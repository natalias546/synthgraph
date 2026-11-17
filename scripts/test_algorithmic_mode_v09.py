"""v0.9 regression: two local registries + two context-only LLM calls.

Run: python scripts/test_algorithmic_mode_v09.py --owl ontology/matsci-v11-evidence-context-enriched.owl
No paid API/vision calls. Uses a local mocked provider and a generated PDF.
"""
from __future__ import annotations

import argparse
import copy
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.algorithmic_extraction import build_algorithmic_registry
from src.pipeline.algorithmic_context import CONTEXT_SCHEMAS, CONTEXT_SYSTEM, run_algorithmic_context
from src.pipeline.evidence_context import enrich_structure
from src.pipeline.pre_llm_review import new_review, reviewed_selection, transmission_tables
from src.pipeline.config import Config
from src.pipeline.runner import Pipeline
from types import SimpleNamespace


def tiny_structure():
    texts = [
        ("P1", "methods", "Single crystals of Ni3In2S2 were grown using a self-flux method at 1123 K."),
        ("P2", "results", "The magnetoresistance of Ni3In2S2 reached 850% at 2 K and 9 T."),
        ("P3", "results", "The resistivity was measured as 0.2 Ω cm at 3 K."),
        ("P4", "results", "Our result was consistent with the previously reported 55% [12]."),
        ("P5", "results", "Magnetoresistance and resistivity changed by 5% at 2 K."),
    ]
    st = {"title": "Small scientific paper", "sections": [
        {"id": "S1", "role": "results", "heading": "Results"}],
        "paragraphs": [{"id": pid, "section_id": "S1", "role": role,
                        "kind": "text", "page": 1, "text": txt}
                       for pid, role, txt in texts], "figures": [], "tables": []}
    return enrich_structure(st)


def test_registry():
    st = tiny_structure()
    cand = {"candidate_fingerprint": "C", "resolver_fingerprint": "R", "candidates": []}
    selection = reviewed_selection(st, cand, new_review(st, cand), SimpleNamespace(entries={}, by_iri={}))
    reg = build_algorithmic_registry(st, cand, selection)
    assert reg["pass1"]["materials"] and reg["pass1"]["samples"]
    assert reg["pass1"]["process_steps"]
    mr = next(m for m in reg["pass2"]["measurements"] if m["value_text"] == "850%")
    assert mr["property_name_in_text"].lower() == "magnetoresistance"
    assert {c["value_text"] for c in mr["conditions"]} == {"2 K", "9 T"}
    assert mr["confidence"] < 0.85
    assert all(m["property_class"] == "OTHER" for m in reg["pass2"]["measurements"])
    assert any("more than one possible measured property" in a["reason"] for a in reg["ambiguities"])
    assert all(a["evidence_id"] in {e["id"] for e in st["semantic_evidence"]}
               for a in reg["ambiguities"])
    # Explicit rejection never leaks into algorithmic p1/2.
    excluded = copy.deepcopy(selection)
    excluded["extract"] = [x for x in selection["extract"] if x["id"] != "P2:s1"]
    no_mr = build_algorithmic_registry(st, cand, excluded)
    assert all(m["value_text"] != "850%" for m in no_mr["pass2"]["measurements"])
    for schema in CONTEXT_SCHEMAS.values():
        js = str(schema.model_json_schema())
        assert "ontology_class" not in js and "ontology_iri" not in js
        assert "entity_group" not in js and "predicate_iri" not in js
    # Regression: >80 ambiguous evidence proposals must all reach their
    # owning source packet, never be silently truncated.
    many = copy.deepcopy(reg)
    many["ambiguities"] = [{"id": f"AMB_{i}", "paragraph_id": "P2",
                            "evidence_id": "P2:s1", "reason": "unresolved condition",
                            "candidate_terms": ["MR"], "sentence": st["paragraphs"][1]["text"]}
                           for i in range(95)]
    class ContextMock:
        def __init__(self): self.tasks=[]
        def ask(self, label, *, system, document, task, schema, paper_id):
            self.tasks.append(task)
            return ({"entities": [], "contextual_links": [], "assessments": []}
                    if label.startswith("alg_context3a") else
                    {"applications": [], "comparisons": [], "assessments": []})
    mock = ContextMock()
    run_algorithmic_context(mock, [{"chunk": 0, "paragraph_ids": ["P2"],
                                    "text": st["paragraphs"][1]["text"]}], many, st, "PAPER")
    assert len(mock.tasks) == 2 and "AMB_94" in mock.tasks[0] and "AMB_94" in mock.tasks[1]
    print("PASS: deterministic materials/samples/measurements/conditions, ambiguity and exclusions")
    print("PASS: no ambiguity truncation even with >80 source candidates")


def test_pipeline(owl: Path):
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        print("SKIP reportlab PDF integration (optional dependency missing)")
        return
    # CI/minimal test environments may omit Anthropic; production SDK supplies
    # the same schema conversion. This shim is confined to this offline test.
    if "anthropic" not in sys.modules:
        try:
            import anthropic
        except ImportError:
            sys.modules["anthropic"] = types.SimpleNamespace(
                transform_schema=lambda schema: schema.model_json_schema())
    with tempfile.TemporaryDirectory(prefix="synthgraph-v09-test-") as tmp:
        root = Path(tmp)
        pdf = root / "data" / "publications" / "alg_test.pdf"
        pdf.parent.mkdir(parents=True)
        c = canvas.Canvas(str(pdf))
        c.setFont("Helvetica-Bold", 15)
        c.drawString(50, 760, "Ni3In2S2 measurement study")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 725, "Methods")
        c.setFont("Helvetica", 11)
        c.drawString(50, 700, "Single crystals of Ni3In2S2 were grown using a self-flux method at 1123 K.")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 665, "Results")
        c.setFont("Helvetica", 11)
        c.drawString(50, 640, "Magnetoresistance of Ni3In2S2 reached 850% at 2 K and 9 T.")
        c.drawString(50, 620, "Our measurements were consistent with prior work [12].")
        c.drawString(50, 600, "Surface states were proposed as a possible explanation.")
        c.save()
        cfg = Config(project_root=root, ontology_file=owl,
                     profile_file=ROOT / "ontology" / "slice_manifest.json",
                     parser="pdfplumber", table_vision="never", figure_vision=False,
                     allow_live_calls=False, extraction_mode="algorithmic")
        calls = []
        def responder(label, schema, task, document):
            calls.append((label, task, document))
            assert "ALLOWED ONTOLOGY LABELS" not in task
            assert "ontology_class" not in str(schema.model_json_schema())
            assert "".join(["ontology", " glossary"]) not in CONTEXT_SYSTEM.lower()
            if label.startswith("alg_context3a"):
                assert "AMBIGUITIES TO ASSESS" in task
                return {"entities": [], "contextual_links": [], "assessments": []}
            if label.startswith("alg_context3b"):
                return {"applications": [], "comparisons": [], "assessments": []}
            raise AssertionError("Algorithmic mode called an unexpected model stage: " + label)
        pipe = Pipeline(cfg, mock_responder=responder)
        pre = pipe.run_paper(pdf, until="algorithmic_registry")
        assert "algorithmic_registry" in pre
        assert "llm" not in pipe.__dict__, "Preflight unexpectedly constructed an LLM client"
        assert pre["algorithmic_registry"]["mode"] == "algorithmic"
        from src.pipeline.io_utils import StageStore
        from src.pipeline.pre_llm_review import read_or_create_review
        rpath = StageStore(cfg.outputs_dir, pre["paper_id"]).dir / "pre_llm_review.json"
        review = read_or_create_review(rpath, pre["structure"], pre["pre_llm_candidates"])
        sel = reviewed_selection(pre["structure"], pre["pre_llm_candidates"], review, pipe.ontology_resolver)
        transmit = transmission_tables(pre["structure"], sel, cfg.max_chars_per_call)
        package = {"paper_id": pre["paper_id"], "structure": pre["structure"],
                   "selection": sel, "documents": transmit["packets_raw"],
                   "candidates": pre["pre_llm_candidates"],
                   "algorithmic_registry": pre["algorithmic_registry"]}
        est = pipe.estimate_cost_from_preflight(package, show=False)
        assert est["calls"] == 2*len(transmit["packets_raw"]), est
        assert est["rows"][0]["registry_tokens"] > 0
        assert est["extraction_mode"] == "algorithmic"
        complete = pipe.run_paper(pdf, until="resolve")
        assert "ontology_resolve" in complete and "validate" in complete
        assert len(calls) == 2, calls
        assert "pass1" in complete and "pass2" in complete and "pass3" in complete
        assert not any(k.startswith("ontology_") for k in complete["pass3"].get("context_links", [{}])[0]) if complete["pass3"].get("context_links") else True
        assert complete["meta"]["extraction_mode"] == "algorithmic"
        # The final resolver, not the LLM, maps a natural-language relation
        # phrase to an OWL predicate, subject to exact group compatibility.
        mat = complete["pass1"]["materials"][0]
        sample = complete["pass1"]["samples"][0]
        assert mat["id"] and sample["id"]
        rel_text = {"subject_text": sample["name_in_paper"], "object_text": mat["name"],
                    "relation_phrase": "composed of", "scope": "directly stated",
                    "evidence": sample["evidence"], "probability": 0.7}
        test_p3 = {"entities": [], "relations": [], "applications": [],
                   "unresolved_terms": [], "context_links": [rel_text],
                   "context_extraction_mode": "context_only_no_ontology_labels"}
        aligned_relation = pipe.ontology_resolver.resolve_extraction({
            "pass1": complete["pass1"], "pass2": complete["pass2"], "pass3": test_p3})
        assert aligned_relation["pass3"]["relations"][0]["predicate"] == "composed_of"
        assert aligned_relation["pass3"]["context_links"][0]["resolution_status"] == "exact_profile_predicate"
        # An incompatible class-role pairing MUST NOT generate an OWL edge.
        bad = copy.deepcopy(test_p3)
        bad["context_links"][0]["subject_text"] = mat["name"]
        bad["context_links"][0]["object_text"] = sample["name_in_paper"]
        invalid = pipe.ontology_resolver.resolve_extraction({
            "pass1": complete["pass1"], "pass2": complete["pass2"], "pass3": bad})
        assert not invalid["pass3"]["relations"]
        assert invalid["pass3"]["context_links"][0]["resolution_status"] == "unresolved_context_relation"
        # Re-running should read stage caches and not call the mocked model again.
        repeated = pipe.run_paper(pdf, until="resolve")
        assert len(calls) == 2

        # Legacy baseline on the SAME source is still four text calls, never
        # reusing algorithmic stages because mode/version are fingerprinted.
        legacy_cfg = Config(project_root=root, ontology_file=owl,
                            profile_file=ROOT / "ontology" / "slice_manifest.json",
                            parser="pdfplumber", table_vision="never", figure_vision=False,
                            allow_live_calls=False, extraction_mode="four_pass")
        old_calls = []
        def baseline(label, schema, task, document):
            old_calls.append(label)
            if label.startswith("pass1"):
                return {"materials": [], "samples": [], "process_steps": [], "unresolved_notes": []}
            if label.startswith("pass2"):
                return {"measurements": [], "unresolved_notes": []}
            return {"entities": [], "relations": [], "rejected_candidates": [],
                    "unresolved_terms": [], "applications": [], "comparisons": []}
        baseline_pipe = Pipeline(legacy_cfg, mock_responder=baseline)
        baseline_pipe.run_paper(pdf, until="resolve")
        assert len(old_calls) == 4, old_calls
        print("PASS: no paid calls in algorithmic preflight, two context-only calls, offline OWL resolution")
        print("PASS: two-call cost estimate, exact packet preview, cache isolation, baseline four calls")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", type=Path, default=ROOT / "ontology/matsci-v11-evidence-context-enriched.owl")
    args = ap.parse_args()
    test_registry()
    if args.owl.exists():
        test_pipeline(args.owl.resolve())
    else:
        print("SKIP ontology-backed integration: OWL file not found")

if __name__ == "__main__":
    main()
