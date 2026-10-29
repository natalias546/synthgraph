"""Orchestrator: runs the stages for each paper, saving data/extractions/<paper_id>/<stage>.json.

A stage is skipped when its saved output was produced from the same inputs
(fingerprint). Claude responses are additionally cached by exact request, so a
re-run never pays twice for the same call.
"""
from __future__ import annotations

import traceback
from pathlib import Path

from .. import extract as X
from .config import Config, load_api_key, software_versions
from .export import GraphBuilder, write_ontology_gaps, write_review_queue
from .ingest import ingest_pdf
from .io_utils import StageStore, paper_id_from_filename, read_json, sha256_file, sha256_json, write_json
from .llm import ClaudeExtractor
from ..ontology_loader import Ontology, Profile
from .resolve import resolve_paper
from .schemas import build_schemas
from .structure import build_structure
from .validate import validate_paper

PIPELINE_VERSION = "0.2.0"  # bump when a stage's code changes its output, so saved stages are redone
STAGES = ["ingest", "structure", "tables", "pass1", "pass2", "pass3", "validate", "resolve"]


class Pipeline:
    def __init__(self, cfg: Config, mock_responder=None):
        self.cfg = cfg
        self.ontology = Ontology(cfg.ontology_file)
        self.profile = Profile(self.ontology, cfg.profile_file)
        self.schemas = build_schemas(self.profile)
        self.system = X.system_prompt(self.profile)
        self.fewshot = X.load_fewshot(cfg.fewshot_file)
        self.llm = ClaudeExtractor(cfg.model, cfg.cache_dir, load_api_key(cfg.project_root),
                                   max_output_tokens=cfg.max_output_tokens, allow_live_calls=cfg.allow_live_calls,
                                   use_prompt_caching=cfg.use_prompt_caching, mock_responder=mock_responder)
        self.use_vision_for_tables = cfg.extra.get("use_vision_for_tables", True)

    # ------------------------------------------------------------------
    def papers(self) -> list[Path]:
        return sorted(p for p in self.cfg.papers_dir.rglob("*.pdf"))  # PDFs may sit in subfolders

    def update_checksums(self) -> dict:
        """Record SHA-256 of every file in data/publications/ in checksums.json (relative paths as keys)."""
        path = self.cfg.checksums_file
        current = read_json(path) if path.exists() and path.read_text(encoding="utf-8").strip() else {}
        for f in sorted(self.cfg.papers_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                current[f.relative_to(self.cfg.project_root).as_posix()] = sha256_file(f)
        write_json(path, dict(sorted(current.items())))
        return current

    def manifest(self) -> dict:
        papers = self.papers()
        return {
            "pipeline_version": PIPELINE_VERSION,
            "config": self.cfg.as_dict(),
            "software": software_versions(),
            "ontology": self.profile.summary(),
            "llm": self.llm.status,
            "system_prompt_sha256": sha256_json(self.system),
            "ontology_notes": self.cfg.notes,
            "papers": [{"paper_id": paper_id_from_filename(p), "file": p.name, "sha256": sha256_file(p)} for p in papers],
        }

    # ------------------------------------------------------------------
    def run_paper(self, pdf: Path, until: str = "resolve", force: tuple = ()) -> dict:
        pid = paper_id_from_filename(pdf)
        store = StageStore(self.cfg.outputs_dir, pid)
        out = {"paper_id": pid}
        stop_at = STAGES.index(until)

        def stage(name, fp_inputs, fn):
            fp = sha256_json({"v": PIPELINE_VERSION, "stage": name, **fp_inputs})
            cached = None if name in force else store.load(name, fp)
            if cached is not None:
                out[name] = cached
                return cached, fp
            res = store.save(name, fn(), fp)
            out[name] = res
            return res, fp

        pdf_hash = sha256_file(pdf)
        ing, fp_ing = stage("ingest", {"pdf": pdf_hash}, lambda: ingest_pdf(pdf))
        if stop_at < 1:
            return out
        st, fp_st = stage("structure", {"up": fp_ing}, lambda: build_structure(ing))
        if stop_at < 2:
            return out

        def do_tables():
            if not self.use_vision_for_tables:
                return {"paragraphs": [], "skipped": True}
            return {"paragraphs": X.run_table_vision(self.llm, self.schemas, pdf, st, pid, self.system),
                    "pages": X.table_pages(st)}
        tb, fp_tb = stage("tables", {"up": fp_st, "model": self.cfg.model, "vision": self.use_vision_for_tables}, do_tables)
        paragraphs = st["paragraphs"] + tb["paragraphs"]
        if stop_at < 3:
            return out

        structure_all = {**st, "paragraphs": paragraphs}
        docs = X.build_documents(structure_all, st["title"], self.cfg.skip_sections, self.cfg.max_chars_per_call)
        llm_fp = {"docs": sha256_json([d["text"] for d in docs]), "system": sha256_json(self.system),
                  "fewshot": sha256_json(self.fewshot), "cfg": self.cfg.fingerprint_fields(),
                  "schemas": sha256_json({k: v.model_json_schema() for k, v in self.schemas.items()})}
        p1, fp1 = stage("pass1", llm_fp, lambda: X.run_pass1(self.llm, self.schemas, self.system, docs, self.fewshot, pid))
        if stop_at < 4:
            return out
        p2, fp2 = stage("pass2", {**llm_fp, "up": fp1},
                        lambda: X.run_pass2(self.llm, self.schemas, self.system, docs, self.fewshot, p1, pid))
        if stop_at < 5:
            return out
        p3, fp3 = stage("pass3", {**llm_fp, "up": fp2},
                        lambda: X.run_pass3(self.llm, self.schemas, self.system, docs, self.fewshot, self.profile, p1, p2, pid))
        if stop_at < 6:
            return out
        val, fp_val = stage("validate", {"up": fp3, "profile": sha256_json(self.profile.summary()),
                                         "min_conf": self.cfg.min_confidence_for_auto_accept},
                            lambda: validate_paper({"pass1": p1, "pass2": p2, "pass3": p3}, paragraphs, self.profile,
                                                   self.cfg.min_confidence_for_auto_accept))
        if stop_at < 7:
            return out
        stage("resolve", {"up": fp_val}, lambda: resolve_paper(val))
        out["meta"] = {"title": st["title"], "pdf": pdf.name, "sha256": pdf_hash, "documents": [
            {"chunk": d["chunk"], "chars": d["chars"]} for d in docs]}
        return out

    def run_all(self, until="resolve", force=(), papers=None, stop_on_error=False) -> dict:
        results, errors = {}, {}
        self.update_checksums()
        for pdf in papers or self.papers():
            print(f"[{pdf.name}]", flush=True)
            try:
                results[paper_id_from_filename(pdf)] = self.run_paper(pdf, until=until, force=force)
            except Exception as exc:
                errors[pdf.name] = f"{type(exc).__name__}: {exc}"
                print("  ERROR:", errors[pdf.name], flush=True)
                if stop_on_error:
                    traceback.print_exc()
                    raise
        return {"results": results, "errors": errors}

    # ------------------------------------------------------------------
    def export(self, results: dict, out_dir: Path | None = None, prune_classes=True) -> dict:
        """Write data/graph/: nodes.jsonl, relationships.jsonl, graph.ttl, review_queue.csv,
        ontology_gaps.csv, run_manifest.json."""
        out_dir = Path(out_dir or self.cfg.graph_dir)
        gb = GraphBuilder(self.profile)
        resolved = {}
        for pid, r in results.items():
            if "resolve" in r:
                resolved[pid] = r["resolve"]
                gb.add_paper(pid, r.get("meta", {}), r["resolve"])
        if prune_classes:
            gb.prune_unused_classes()
        info = gb.write(out_dir)
        info["rdf_triples"] = gb.write_turtle(out_dir / "graph.ttl")
        info["review_queue_rows"] = write_review_queue(resolved, out_dir / "review_queue.csv")
        info["ontology_gap_rows"] = len(write_ontology_gaps(resolved, out_dir / "ontology_gaps.csv"))
        write_json(out_dir / "run_manifest.json", self.manifest())
        return info


def load_results(outputs_dir: Path) -> dict:
    """Reload saved stage outputs (e.g. in a fresh session) without running anything."""
    res = {}
    for d in sorted(Path(outputs_dir).iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            r = {"paper_id": d.name}
            for s in STAGES:
                p = d / f"{s}.json"
                if p.exists():
                    r[s] = read_json(p)
            if "ingest" in r and "structure" in r:
                r["meta"] = {"title": r["structure"].get("title"), "pdf": r["ingest"]["source"]["pdf"],
                             "sha256": r["ingest"]["source"]["sha256"]}
            res[d.name] = r
    return res
