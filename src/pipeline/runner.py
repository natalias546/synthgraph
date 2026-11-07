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
from .evidence_context import enrich_structure
from .pre_llm_review import (make_candidates, read_or_create_review, reviewed_selection,
                             review_summary, approved_documents, apply_reviewed_mappings, ReviewRequired)
from .validate import validate_paper

PIPELINE_VERSION = "0.5.0"  # bump when a stage's code changes its output, so saved stages are redone
STAGES = ["ingest", "structure", "pre_llm_candidates", "human_review", "visuals", "pass1", "pass2", "pass3", "ontology_resolve", "validate", "resolve"]


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
                              lambda: enrich_structure(structure_from_docling(load_docling_document(store.dir / "docling.json"), ing, pdf)))
        else:
            ing, fp_ing = stage("ingest", {"pdf": pdf_hash, "parser": "pdfplumber"}, lambda: ingest_pdf(pdf))
            if stop_at < 1:
                return out
            st, fp_st = stage("structure", {"up": fp_ing}, lambda: enrich_structure(build_structure(ing)))
        # Flat reading-order list for the LLM notebooks (same shape as the old text pass).
        write_observations(st, store.dir / "observations.json")
        if stop_at < STAGES.index("pre_llm_candidates"):
            return out
        candidates, fp_candidates = stage(
            "pre_llm_candidates", {"up": fp_st, "resolver": self.ontology_resolver.fingerprint},
            lambda: make_candidates(st, self.ontology_resolver))
        if stop_at < STAGES.index("human_review"):
            return out
        review_path = store.dir / "pre_llm_review.json"
        review = read_or_create_review(review_path, st, candidates)
        out["human_review"] = {"path": str(review_path), "summary": review_summary(review, candidates)}
        if stop_at < STAGES.index("visuals"):
            return out
        # Fail closed before any Claude/vision client or prompt is constructed.
        selection = reviewed_selection(st, candidates, review, self.ontology_resolver)
        approved_visuals = set(selection["approved_visuals"])
        # A visual crop must be explicitly approved by its source paragraph ID.
        relevant_tables = [t for t in st.get("tables", []) if t.get("paragraph_id") in approved_visuals]
        relevant_figures = [f for f in st.get("figures", []) if f.get("paragraph_id") in approved_visuals]

        def do_visuals():
            if not approved_visuals or (not relevant_tables and not relevant_figures and self.cfg.parser == "docling"):
                return {"overrides": {}, "added": [], "skipped": True}
            if self.cfg.parser == "docling":
                chosen = {**st, "tables": relevant_tables, "figures": relevant_figures}
                return X.run_visuals_docling(self.llm, self.schemas, chosen, store.dir, pid, self.system,
                                             self.cfg.table_vision, self.cfg.figure_vision)
            if self.cfg.table_vision == "never":
                return {"overrides": {}, "added": [], "skipped": True}
            filtered = {**st, "paragraphs": [p for p in st["paragraphs"] if p["id"] in approved_visuals]}
            if not filtered["paragraphs"] or not X.table_pages(filtered):
                return {"overrides": {}, "added": [], "skipped": True}
            return {"overrides": {}, "added": X.run_table_vision(self.llm, self.schemas, pdf, filtered, pid, self.system),
                    "pages": X.table_pages(filtered)}
        vis, fp_vis = stage("visuals", {"up": fp_candidates, "review": selection["review_fingerprint"],
                                        "approved_visuals": sorted(approved_visuals),
                                        "model": self.cfg.model, "table_vision": self.cfg.table_vision,
                                        "figure_vision": self.cfg.figure_vision}, do_visuals)
        if stop_at < STAGES.index("pass1"):
            return out

        paragraphs, docs, llm_fp = self._llm_inputs(st, vis, selection)
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
                                    lambda: apply_reviewed_mappings(
                                        self.ontology_resolver.resolve_extraction(
                                            {"pass1": p1, "pass2": p2, "pass3": p3}),
                                        selection, self.ontology_resolver))
        if stop_at < STAGES.index("validate"):
            return out

        def do_validate():
            validated = validate_paper(aligned, paragraphs, self.profile,
                                       self.cfg.min_confidence_for_auto_accept,
                                       approved_primary_sentence_ids={e["id"] for e in selection["extract"]},
                                       approved_visual_paragraph_ids=set(selection["approved_visuals"]))
            return annotate_validation(validated, aligned)

        val, fp_val = stage("validate", {"up": fp_aligned,
                                       "profile": sha256_json(self.profile.summary()),
                                       "min_conf": self.cfg.min_confidence_for_auto_accept}, do_validate)
        if stop_at < STAGES.index("resolve"):
            return out
        stage("resolve", {"up": fp_val}, lambda: resolve_paper(val))
        out["meta"] = {"title": st["title"], "pdf": pdf.name, "sha256": pdf_hash,
                       "review_fingerprint": selection["review_fingerprint"],
                       "approved_primary_sentence_ids": [e["id"] for e in selection["extract"]],
                       "approved_visual_paragraph_ids": selection["approved_visuals"],
                       "documents": [{"chunk": d["chunk"], "chars": d["chars"]} for d in docs]}
        return out

    def _llm_inputs(self, st, vis, selection):
        """Paragraphs (with table/figure vision applied), the documents sent to Claude, and the fingerprint
        inputs of the Claude passes."""
        from collections import defaultdict
        cues = defaultdict(set)
        for ev in st.get("semantic_evidence", []):
            cues[ev["paragraph_id"]].update(ev.get("scientific_roles", []))
        paragraphs = [{**p, **vis["overrides"].get(p["id"], {}),
                       "scientific_cues": sorted(cues[p["id"]])} for p in st["paragraphs"]] + vis["added"]
        docs = approved_documents(st, selection, self.cfg.max_chars_per_call)
        # Transcriptions from explicitly approved vision assets may supplement
        # the source sentence packets; they are never silently included.
        visual_paragraphs = [p for p in paragraphs if p.get("text_source") == "claude_vision"
                             and p["id"] in selection["approved_visuals"]]
        visual_paragraphs += [p for p in vis.get("added", []) if p.get("provenance") == "claude_vision"]
        if visual_paragraphs:
            extras = X.build_documents({**st, "paragraphs": visual_paragraphs}, st["title"],
                                       self.cfg.skip_sections, self.cfg.max_chars_per_call)
            for extra in extras:
                extra["chunk"] = len(docs)
                docs.append(extra)
        llm_fp = {"review": selection["review_fingerprint"], "candidates": self.ontology_resolver.fingerprint,
                  "docs": sha256_json([d["text"] for d in docs]), "system": sha256_json(self.system),
                  "fewshot": sha256_json(self.fewshot), "cfg": self.cfg.fingerprint_fields(),
                  "schemas": sha256_json({k: v.model_json_schema() for k, v in self.schemas.items()})}
        return paragraphs, docs, llm_fp

    def _pass_saved(self, pid, st, passes, selection=None) -> bool:
        """True only when the exact reviewed input and output-stage hashes match."""
        store = StageStore(self.cfg.outputs_dir, pid)
        if selection is None:
            return False
        can = store.load("pre_llm_candidates")
        if not can:
            return False
        visual_fp = sha256_json({"v": PIPELINE_VERSION, "stage": "visuals",
                                 "up": can.get("_fingerprint"),
                                 "review": selection["review_fingerprint"],
                                 "approved_visuals": sorted(selection["approved_visuals"]),
                                 "model": self.cfg.model,
                                 "table_vision": self.cfg.table_vision,
                                 "figure_vision": self.cfg.figure_vision})
        vis = store.load("visuals", visual_fp)
        if not vis:
            return False
        if passes == 0:
            return True
        llm_fp = self._llm_inputs(st, vis, selection)[2]
        up = None
        for k in range(1, passes + 1):
            inputs = {**llm_fp, **({"up": up} if up else {})}
            up = sha256_json({"v": PIPELINE_VERSION, "stage": f"pass{k}", **inputs})
            if store.load(f"pass{k}", up) is None:
                return False
        return True

    def estimate_cost(self, papers=None, until="resolve", show=True) -> dict:
        """Estimate *approved* text/vision only. No paid call without valid review."""
        from . import cost
        stop = STAGES.index(until)
        passes = max(0, min(3, stop - STAGES.index("visuals")))
        # Fail closed for the complete batch before contacting token-counting APIs.
        approved = []
        for pdf in papers or self.papers():
            pid = paper_id_from_filename(pdf)
            r = self.run_paper(pdf, until="human_review")
            st, candidates = r["structure"], r["pre_llm_candidates"]
            review = read_or_create_review(StageStore(self.cfg.outputs_dir, pid).dir / "pre_llm_review.json", st, candidates)
            selection = reviewed_selection(st, candidates, review, self.ontology_resolver)
            approved.append((pdf, pid, st, selection))
        count = cost.TokenCounter(self.llm.client, self.cfg.model)
        system_tokens = count(self.system)
        rows = []
        for pdf, pid, st, selection in approved:
            docs = approved_documents(st, selection, self.cfg.max_chars_per_call)
            approved_assets = set(selection["approved_visuals"])
            images = 0
            if stop >= STAGES.index("visuals"):
                if self.cfg.parser == "docling":
                    if self.cfg.table_vision != "never":
                        images += sum(1 for t in st.get("tables", []) if t.get("paragraph_id") in approved_assets
                                      and t.get("crop") and (self.cfg.table_vision == "always"
                                      or (self.cfg.table_vision == "fallback" and t.get("needs_vision"))))
                    if self.cfg.figure_vision:
                        images += sum(1 for f in st.get("figures", []) if f.get("paragraph_id") in approved_assets and f.get("crop"))
                elif self.cfg.table_vision != "never":
                    images = len({p["page"] for p in st["paragraphs"] if p["id"] in approved_assets
                                  and (p["kind"] == "table" or p.get("caption_of", "").startswith("table"))})
            rows.append({"paper_id": pid, "doc_tokens": [count(d["text"]) for d in docs] if passes else [],
                         "images": images, "saved": self._pass_saved(pid, st, passes, selection)})
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
        if STAGES.index(until) >= STAGES.index("visuals"):
            from .cost import CostLimitExceeded
            # estimate_cost first validates every human review manifest, then
            # optionally counts tokens; no paid work can precede approval.
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
        resolved = {pid: r["resolve"] for pid, r in results.items() if "resolve" in r}
        def graph_for(statuses, destination):
            gb = GraphBuilder(self.profile, include_statuses=statuses)
            for pid, res in resolved.items():
                gb.add_paper(pid, results[pid].get("meta", {}), res)
            if prune_classes:
                gb.prune_unused_classes()
            info = gb.write(destination)
            info["rdf_triples"] = gb.write_turtle(destination / "graph.ttl")
            return info
        # The canonical graph contains only mechanically auto-accepted assertions.
        # They remain unverified scientific facts; reviewed/provisional claims are segregated.
        info = graph_for(("auto_accepted",), out_dir)
        candidate_dir = out_dir / "candidates"
        info["candidate_graph"] = graph_for(("auto_accepted", "needs_review"), candidate_dir)
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
