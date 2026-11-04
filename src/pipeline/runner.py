"""Orchestrator: runs the stages for each paper, saving data/extractions/<paper_id>/<stage>.json.

A stage is skipped when its saved output was produced from the same inputs
(fingerprint). Claude responses are additionally cached by exact request, so a
re-run never pays twice for the same call.
"""
from __future__ import annotations

import traceback
from functools import cached_property
from pathlib import Path

from .. import extract as X
from .config import Config, load_api_key, software_versions
from .export import GraphBuilder, write_ontology_gaps, write_review_queue
from .ingest import ingest_pdf
from .io_utils import StageStore, paper_id_from_filename, read_json, sha256_file, sha256_json, write_json
from .observations import write_observations
from .llm import ClaudeExtractor
from ..ontology_loader import Ontology, Profile
from ..ontology_resolver import OntologyResolver, annotate_validation
from .resolve import resolve_paper
from .schemas import build_schemas
from .structure import build_structure
from .validate import validate_paper

PIPELINE_VERSION = "0.3.2"  # bump when a stage's code changes its output, so saved stages are redone
STAGES = ["ingest", "structure", "visuals", "pass1", "pass2", "pass3", "ontology_resolve", "validate", "resolve"]


class Pipeline:
    def __init__(self, cfg: Config, mock_responder=None):
        self.cfg = cfg
        self._mock_responder = mock_responder
        self._docling = None
        if cfg.parser not in ("docling", "pdfplumber"):
            raise ValueError(f"parser must be 'docling' or 'pdfplumber', not {cfg.parser!r}")

    # The ontology and Claude client load on first use, so parsing (until="structure")
    # runs without an ontology file or API key.
    @cached_property
    def ontology(self):
        if self.cfg.ontology_file is None:
            raise FileNotFoundError(f"No ontology for the LLM stages: {'; '.join(self.cfg.notes)}. Put the .owl in "
                                    "ontology/ or set ONTOLOGY_FILE; parsing alone needs neither (until='structure').")
        return Ontology(self.cfg.ontology_file)

    @cached_property
    def profile(self):
        return Profile(self.ontology, self.cfg.profile_file)

    @cached_property
    def schemas(self):
        return build_schemas(self.profile)

    @cached_property
    def ontology_resolver(self):
        """Local vocabulary matcher: no API calls, no changes to the OWL ontology."""
        return OntologyResolver(self.profile)

    @cached_property
    def system(self):
        return X.system_prompt(self.profile)

    @cached_property
    def fewshot(self):
        return X.load_fewshot(self.cfg.fewshot_file)

    @cached_property
    def llm(self):
        cfg = self.cfg
        return ClaudeExtractor(cfg.model, cfg.cache_dir, load_api_key(cfg.project_root),
                               max_output_tokens=cfg.max_output_tokens, allow_live_calls=cfg.allow_live_calls,
                               use_prompt_caching=cfg.use_prompt_caching, mock_responder=self._mock_responder)

    @property
    def docling(self):
        """One Docling converter per Pipeline (its models load once, on first use)."""
        if self._docling is None:
            from .docling_ingest import DoclingParser
            self._docling = DoclingParser(self.cfg.docling)
        return self._docling

    # ------------------------------------------------------------------
    def papers(self) -> list[Path]:
        """All PDFs under data/publications/, including subfolders."""
        return sorted((p for p in self.cfg.papers_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"),
                      key=lambda p: p.name.lower())

    def find_papers(self, names) -> list[Path]:
        """Resolve file names (or stems / paper IDs) to PDFs anywhere under data/publications/."""
        all_pdfs = self.papers()
        out, missing = [], []
        for n in names:
            hits = [p for p in all_pdfs if n in (p.name, p.stem, paper_id_from_filename(p))]
            (out.extend(hits) if hits else missing.append(n))
        if missing:
            raise FileNotFoundError(f"Not found under {self.cfg.papers_dir}: {missing}")
        return out

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
        if self.cfg.parser == "docling":
            from .docling_ingest import load_docling_document, structure_from_docling
            ing, fp_ing = stage("ingest", {"pdf": pdf_hash, "parser": "docling", "docling": self.docling.fingerprint()},
                                lambda: self.docling.convert(pdf, store.dir))
            if stop_at < 1:
                return out
            st, fp_st = stage("structure", {"up": fp_ing},
                              lambda: structure_from_docling(load_docling_document(store.dir / "docling.json"), ing, pdf))
        else:
            ing, fp_ing = stage("ingest", {"pdf": pdf_hash, "parser": "pdfplumber"}, lambda: ingest_pdf(pdf))
            if stop_at < 1:
                return out
            st, fp_st = stage("structure", {"up": fp_ing}, lambda: build_structure(ing))
        # Flat reading-order list for the LLM notebooks (same shape as the old text pass).
        write_observations(st, store.dir / "observations.json")
        if stop_at < 2:
            return out

        def do_visuals():
            if self.cfg.parser == "docling":
                return X.run_visuals_docling(self.llm, self.schemas, st, store.dir, pid, self.system,
                                             self.cfg.table_vision, self.cfg.figure_vision)
            if self.cfg.table_vision == "never":
                return {"overrides": {}, "added": [], "skipped": True}
            return {"overrides": {}, "added": X.run_table_vision(self.llm, self.schemas, pdf, st, pid, self.system),
                    "pages": X.table_pages(st)}
        vis, fp_vis = stage("visuals", {"up": fp_st, "model": self.cfg.model, "table_vision": self.cfg.table_vision,
                                        "figure_vision": self.cfg.figure_vision}, do_visuals)
        if stop_at < 3:
            return out

        paragraphs, docs, llm_fp = self._llm_inputs(st, vis)
        p1, fp1 = stage("pass1", llm_fp, lambda: X.run_pass1(self.llm, self.schemas, self.system, docs, self.fewshot, pid, self.profile))
        if stop_at < 4:
            return out
        p2, fp2 = stage("pass2", {**llm_fp, "up": fp1},
                        lambda: X.run_pass2(self.llm, self.schemas, self.system, docs, self.fewshot, p1, pid, self.profile))
        if stop_at < 5:
            return out
        p3, fp3 = stage("pass3", {**llm_fp, "up": fp2},
                        lambda: X.run_pass3(self.llm, self.schemas, self.system, docs, self.fewshot, self.profile, p1, p2, pid))
        if stop_at < STAGES.index("ontology_resolve"):
            return out
        # Resolve source scientific terms against the v11 OWL vocabulary without an
        # additional model call. Only unique exact labels/approved aliases get IRIs.
        aligned, fp_aligned = stage("ontology_resolve",
                                    {"up": fp3, "resolver": self.ontology_resolver.fingerprint},
                                    lambda: self.ontology_resolver.resolve_extraction(
                                        {"pass1": p1, "pass2": p2, "pass3": p3}))
        if stop_at < STAGES.index("validate"):
            return out

        def do_validate():
            validated = validate_paper(aligned, paragraphs, self.profile,
                                       self.cfg.min_confidence_for_auto_accept)
            return annotate_validation(validated, aligned)

        val, fp_val = stage("validate", {"up": fp_aligned,
                                       "profile": sha256_json(self.profile.summary()),
                                       "min_conf": self.cfg.min_confidence_for_auto_accept}, do_validate)
        if stop_at < STAGES.index("resolve"):
            return out
        stage("resolve", {"up": fp_val}, lambda: resolve_paper(val))
        out["meta"] = {"title": st["title"], "pdf": pdf.name, "sha256": pdf_hash, "documents": [
            {"chunk": d["chunk"], "chars": d["chars"]} for d in docs]}
        return out

    def _llm_inputs(self, st, vis):
        """Paragraphs (with table/figure vision applied), the documents sent to Claude, and the fingerprint
        inputs of the Claude passes."""
        paragraphs = [{**p, **vis["overrides"].get(p["id"], {})} for p in st["paragraphs"]] + vis["added"]
        docs = X.build_documents({**st, "paragraphs": paragraphs}, st["title"], self.cfg.skip_sections,
                                 self.cfg.max_chars_per_call)
        llm_fp = {"docs": sha256_json([d["text"] for d in docs]), "system": sha256_json(self.system),
                  "fewshot": sha256_json(self.fewshot), "cfg": self.cfg.fingerprint_fields(),
                  "schemas": sha256_json({k: v.model_json_schema() for k, v in self.schemas.items()})}
        return paragraphs, docs, llm_fp

    def _pass_saved(self, pid, st, passes) -> bool:
        """True if pass1..passes are saved for these exact inputs (same check run_paper uses to skip them)."""
        store = StageStore(self.cfg.outputs_dir, pid)
        if not passes:
            return store.path("visuals").exists()
        vis = store.load("visuals")
        if vis is None:
            return False
        llm_fp, up = self._llm_inputs(st, vis)[2], None
        for k in range(1, passes + 1):
            up = sha256_json({"v": PIPELINE_VERSION, "stage": f"pass{k}", **llm_fp, **({"up": up} if up else {})})
        return store.load(f"pass{passes}", up) is not None

    def estimate_cost(self, papers=None, until="resolve", show=True) -> dict:
        """What the Claude stages up to `until` would cost for these papers, before calling anything.
        Parses papers that aren't parsed yet (free). See src/pipeline/cost.py for the assumptions."""
        from . import cost
        stop = STAGES.index(until)
        passes = max(0, min(3, stop - STAGES.index("visuals")))
        count = cost.TokenCounter(self.llm.client, self.cfg.model)
        system_tokens = count(self.system)
        skip_figures = {"logo", "signature", "qr_code", "bar_code", "stamp", "icon"}
        rows = []
        for pdf in papers or self.papers():
            pid = paper_id_from_filename(pdf)
            st = self.run_paper(pdf, until="structure")["structure"]
            docs = X.build_documents(st, st["title"], self.cfg.skip_sections, self.cfg.max_chars_per_call)
            images = 0
            if stop >= STAGES.index("visuals") and self.cfg.parser == "docling":
                images = sum(1 for t in st.get("tables", []) if t.get("crop") and (
                    self.cfg.table_vision == "always" or (self.cfg.table_vision == "fallback" and t.get("needs_vision"))))
                if self.cfg.figure_vision:
                    images += sum(1 for f in st.get("figures", []) if f.get("crop") and
                                  (f.get("classification") or {}).get("class", "") not in skip_figures)
            rows.append({"paper_id": pid, "doc_tokens": [count(d["text"]) for d in docs] if passes else [],
                         "images": images, "saved": self._pass_saved(pid, st, passes)})
        outputs = cost.output_range(cost.observed_outputs(self.cfg.cache_dir, self.cfg.model))
        result = cost.estimate(self.cfg.model, system_tokens, rows, outputs, passes)
        result["text"] = cost.summary_text(self.cfg.model, system_tokens, count.exact, outputs, result)
        if show:
            print(result["text"], flush=True)
        return result

    def run_all(self, until="resolve", force=(), papers=None, stop_on_error=False, max_cost_usd=None) -> dict:
        """Run the stages for each paper. Before the first Claude call, the cost estimate is printed;
        with max_cost_usd (argument or Config), the run stops if the high estimate is above it."""
        results, errors = {}, {}
        self.update_checksums()
        if STAGES.index(until) >= STAGES.index("visuals") and (self.llm.client or self.cfg.allow_live_calls):
            from .cost import CostLimitExceeded
            est = self.estimate_cost(papers, until)
            limit = max_cost_usd if max_cost_usd is not None else self.cfg.max_cost_usd
            if limit is not None and est["high_usd"] > limit:
                raise CostLimitExceeded(f"High estimate ${est['high_usd']:.2f} is above max_cost_usd=${limit:.2f}. "
                                        "Raise the limit or run fewer papers.")
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
