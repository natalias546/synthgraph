"""Offline regression for candidate review, edits, stale review, and paid-call gate.

Run: python scripts/test_pre_llm_review.py --owl ontology/matsci-v11-parsing-aligned-cleaned.owl
No Anthropic key, LLM, or Neo4j server is required.
"""
from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ontology_loader import Ontology, Profile
from src.ontology_resolver import OntologyResolver
from src.pipeline.config import Config
from src.pipeline.runner import Pipeline
from src.pipeline.evidence_context import enrich_structure
from src.pipeline.pre_llm_review import (
    ReviewRequired, add_manual_candidate, apply_reviewed_mappings, approved_documents, assert_current,
    make_candidates, new_review, read_or_create_review, reviewed_selection,
    set_candidate_decision, set_evidence_decision, set_visual_decision,
)
from src.pipeline.io_utils import read_json, write_json


def test_review(owl: Path):
    root = Path(__file__).resolve().parents[1]
    ontology = Ontology(owl)
    profile = Profile(ontology, root / "ontology" / "slice_manifest.json")
    resolver = OntologyResolver(profile)
    st = enrich_structure({"title": "Local scientific evidence test",
                           "sections": [{"id": "SEC1", "heading": "Results", "role": "results"}],
                           "paragraphs": [{"id": "P1", "section_id": "SEC1", "kind": "paragraph",
                                           "role": "results", "page": 1,
                                           "text": ("Resistivity was measured at 2 K. "
                                                    "Magnetoresistance increased to 25% under magnetic field. "
                                                    "A previous study reported a different result.")}]})
    stage = make_candidates(st, resolver)
    terms = {(c["term"].casefold(), c["match_status"]) for c in stage["candidates"]}
    assert ("resistivity", "exact_match") in terms
    assert ("magnetoresistance", "missing_class") in terms
    rev = new_review(st, stage)
    try:
        reviewed_selection(st, stage, rev, resolver)
    except ReviewRequired:
        pass
    else:
        raise AssertionError("Missing human approval did not block the paid stage")
    set_evidence_decision(rev, "P1:s1", "extract")
    set_evidence_decision(rev, "P1:s2", "extract")
    set_evidence_decision(rev, "P1:s3", "context_only")
    exact = next(c for c in stage["candidates"] if c["term"].casefold() == "resistivity")
    set_candidate_decision(rev, stage, resolver, exact["id"], "approve", selected_iri=exact["matched_iri"])
    missing = next(c for c in stage["candidates"] if c["term"].casefold() == "magnetoresistance")
    set_candidate_decision(rev, stage, resolver, missing["id"], "edit",
                           selected_group="property", term_override="magnetoresistance (MR)",
                           note="Property not yet present as an exact OWL class")
    manual = add_manual_candidate(rev, st, resolver, "P1:s2", "field-dependent transport response", "property")
    set_candidate_decision(rev, stage, resolver, manual, "reject", note="redundant term")
    selected = reviewed_selection(st, stage, rev, resolver)
    docs = approved_documents(st, selected, max_chars=5000)
    assert len(docs) == 1
    source = docs[0]["text"]
    assert "Resistivity was measured" in source
    assert "Magnetoresistance increased" in source
    assert "CONTEXT ONLY P1:s3" in source
    assert exact["matched_iri"] in source
    assert "magnetoresistance (MR)" in source
    assert "field-dependent transport response" not in source
    assert "APPROVED" not in source  # no falsely asserted scientific truth
    # A genuinely reviewer-approved mapping may override the post-LLM 'OTHER'
    # mapping, but only for an exact term and quote from that PRIMARY sentence.
    # This example intentionally uses a broad class, demonstrating the mechanics
    # rather than asserting that magnetoresistance is equivalent to resistivity.
    from src.ontology_resolver import annotate_validation
    from src.pipeline.validate import validate_paper
    review2 = copy.deepcopy(rev)
    set_candidate_decision(review2, stage, resolver, missing["id"], "approve",
                           selected_iri=exact["matched_iri"], selected_group="property",
                           note="Manually approved broad conceptual alignment for testing")
    select2 = reviewed_selection(st, stage, review2, resolver)
    m2 = {"id": "ME1", "property_class": "OTHER", "property_iri": None,
          "proposed_property_label": "magnetoresistance", "property_name_in_text": "Magnetoresistance",
          "technique_class": "OTHER", "technique_name_in_text": "transport measurement",
          "value": 25.0, "value_text": "25%", "unit": "%", "confidence": .9,
          "sample_ids": [], "material_ids": [], "settings": [], "conditions": [],
          "data_origin": "experimental", "source_location": "text", "polarity": "reported",
          "evidence": [{"paragraph_id": "P1", "quote": "Magnetoresistance increased to 25% under magnetic field."}]}
    extraction = {"pass1": {"materials": [], "samples": [], "process_steps": []},
                  "pass2": {"measurements": [m2]},
                  "pass3": {"entities": [], "applications": [], "comparisons": [],
                            "unresolved_terms": [], "relations": []}}
    aligned = apply_reviewed_mappings(resolver.resolve_extraction(extraction), select2, resolver)
    r = next(r for r in aligned["ontology_resolution"] if r["role"] == "measurement_property")
    assert r["status"] == "human_reviewed_mapping" and not r["requires_review"]
    assert aligned["pass2"]["measurements"][0]["property_iri"] == exact["matched_iri"]
    # The human mapping does not silently bypass other scientific checks.
    valid = annotate_validation(validate_paper(aligned, st["paragraphs"], profile, .85,
                   approved_primary_sentence_ids={"P1:s1", "P1:s2"}), aligned)
    assert next(x for x in valid["items"] if x["kind"] == "measurement")["status"] != "rejected"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "review.json"
        write_json(path, rev)
        loaded = read_or_create_review(path, st, stage)
        assert loaded == rev
        altered = copy.deepcopy(st)
        altered["paragraphs"][0]["text"] += " changed"
        try:
            assert_current(loaded, altered, stage)
        except ReviewRequired:
            pass
        else:
            raise AssertionError("Stale evidence was accepted")
    print("PASS: pre-LLM OWL candidate matching, human edits, optional unmapped terms")
    print("PASS: fail-closed review, literal evidence, selection + context + provenance")
    print("PASS: persisted review and stale-source detection")

    # Exercise real stage ordering using a fresh local PDF and no API key.
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        print("SKIP: optional reportlab PDF integration test (not installed)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        papers = work / "data" / "publications"
        papers.mkdir(parents=True)
        pdf = papers / "scientific_test.pdf"
        c = canvas.Canvas(str(pdf))
        c.setFont("Helvetica-Bold", 16)
        c.drawString(60, 720, "Materials Property Study")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, 690, "Results")
        c.setFont("Helvetica", 11)
        c.drawString(60, 665, "Resistivity was measured at 2 K.")
        c.drawString(60, 648, "Magnetoresistance increased at higher magnetic field.")
        c.save()
        cfg = Config(project_root=work, ontology_file=owl,
                     profile_file=root / "ontology" / "slice_manifest.json",
                     parser="pdfplumber", table_vision="never", allow_live_calls=False)
        pl = Pipeline(cfg)
        local = pl.run_paper(pdf, until="human_review")
        assert "human_review" in local and "pre_llm_candidates" in local
        try:
            pl.run_paper(pdf, until="visuals")
        except ReviewRequired:
            pass
        else:
            raise AssertionError("Pipeline did not stop before vision/LLM")
        # A batch run must fail before creating any Claude client/token counter.
        try:
            pl.run_all(until="pass1", papers=[pdf])
        except ReviewRequired:
            pass
        else:
            raise AssertionError("Batch pipeline did not enforce pre-LLM review")
        assert "llm" not in pl.__dict__
        local_review = read_json(Path(local["human_review"]["path"]))
        assert local["structure"]["semantic_evidence"]
        set_evidence_decision(local_review, local["structure"]["semantic_evidence"][0]["id"], "extract")
        write_json(local["human_review"]["path"], local_review)
        resumed = pl.run_paper(pdf, until="visuals")
        assert resumed["visuals"].get("skipped") is True
        assert pl._llm_inputs(resumed["structure"], resumed["visuals"], reviewed_selection(
            resumed["structure"], resumed["pre_llm_candidates"],
            local_review, pl.ontology_resolver))[1]
        def responder(label, schema, task, document):
            assert "PRIMARY" in document
            if label.startswith("pass1"):
                return {"materials": [], "samples": [], "process_steps": [], "unresolved_notes": []}
            if label.startswith("pass2"):
                return {"measurements": [], "unresolved_notes": []}
            return {"entities": [], "relations": [], "rejected_candidates": [],
                    "applications": [], "comparisons": [], "unresolved_terms": []}
        # The production requirements install Anthropic. For a minimal offline
        # test environment, replace only its JSON-schema helper with Pydantic.
        import sys, types
        if "anthropic" not in sys.modules:
            try:
                import anthropic
            except ImportError:
                sys.modules["anthropic"] = types.SimpleNamespace(
                    transform_schema=lambda schema: schema.model_json_schema())
        mocked = Pipeline(cfg, mock_responder=responder)
        whole = mocked.run_paper(pdf, until="resolve")
        assert "ontology_resolve" in whole and "validate" in whole and "resolve" in whole
        assert len(mocked.llm.ledger) == 3
        # Even a literally matching quote is invalid when it cites context-only evidence.
        from src.pipeline.validate import validate_paper
        prop = profile.groups["property"].labels[0]
        measurement = {"id": "ME1", "property_class": prop, "technique_class": "OTHER",
                       "proposed_property_label": None, "proposed_technique_label": "some technique",
                       "property_name_in_text": "Resistance", "technique_name_in_text": "measurement",
                       "sample_ids": [], "material_ids": [], "value": None, "value_text": None,
                       "unit": None, "conditions": [], "settings": [], "data_origin": "experimental",
                       "polarity": "reported", "evidence": [{"paragraph_id": "P1", "quote": st["semantic_evidence"][2]["sentence"]}],
                       "confidence": 0.99}
        unapproved = validate_paper({"pass1": {"materials": [], "samples": [], "process_steps": []},
                                     "pass2": {"measurements": [measurement]},
                                     "pass3": {"entities": [], "applications": [], "comparisons": [],
                                               "unresolved_terms": [], "relations": []}},
                                    st["paragraphs"], profile, .85,
                                    approved_primary_sentence_ids={"P1:s1", "P1:s2"})
        assert unapproved["items"][0]["status"] == "rejected"
        print("PASS: real PDF stages obey approval gate; no paid vision/LLM calls")
        print("PASS: all downstream stages via offline mock + only reviewed primary evidence accepted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", type=Path, required=True)
    args = ap.parse_args()
    test_review(args.owl.resolve())


if __name__ == "__main__":
    main()
