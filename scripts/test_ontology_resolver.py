"""Offline regression smoke test for the v11 ontology resolver (deterministic tier, a scripted
adjudicator standing in for the model, and resolution memory).

Run from the project root:
    python scripts/test_ontology_resolver.py
or provide an explicitly installed v11 OWL:
    python scripts/test_ontology_resolver.py --owl ontology/matsci-v11-parsing-aligned-cleaned.owl

No LLM, internet, Neo4j, PDF corpus or API key is required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ontology_loader import Ontology, Profile, OTHER
from src.ontology_resolver import OntologyResolver, ResolutionMemory, annotate_validation
from src.pipeline.schemas import build_schemas
from src.pipeline.validate import validate_paper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", type=Path, default=ROOT / "ontology" / "matsci-v11-parsing-aligned-cleaned.owl")
    ap.add_argument("--manifest", type=Path, default=ROOT / "ontology" / "slice_manifest.json")
    args = ap.parse_args()
    if not args.owl.is_file() or not args.manifest.is_file():
        ap.error("Install v11 OWL and slice_manifest.json in ontology/, or pass --owl and --manifest")
    ontology = Ontology(args.owl)
    profile = Profile(ontology, args.manifest)
    assert not profile.warnings, profile.warnings
    assert len(profile.groups["application"].class_iris) > 0
    schemas = build_schemas(profile)
    assert "measurements" in schemas and "relations" in schemas
    resolver = OntologyResolver(profile)
    r1 = resolver.resolve("Resistivity", "property")
    assert r1["status"] == "exact_match" and r1["matched_iri"] in ontology.classes, r1
    r2 = resolver.resolve("magnetoresistance", "property")
    assert r2["status"] != "exact_match", r2
    r3 = resolver.resolve("residual resistivity ratio", "property")
    assert r3["status"] != "exact_match", r3
    assert all(x["matched_iri"] is None for x in (r2, r3))

    evidence = {"paragraph_id": "P1", "quote": "Resistivity was 5 ohm cm and magnetoresistance was 12%."}
    def meas(mid, term, value, label=OTHER):
        return {"id": mid, "property_name_in_text": term, "property_class": label,
                "property_iri": None, "proposed_property_label": term,
                "technique_class": OTHER, "technique_name_in_text": "measurement",
                "sample_ids": [], "material_ids": [], "data_origin": "experimental",
                "evidence": [evidence], "confidence": 0.95, "value": value, "unit": None}
    extracted = {"pass1": {"materials": [], "samples": [], "process_steps": []},
                 "pass2": {"measurements": [meas("ME1", "Resistivity", 5),
                                             meas("ME2", "magnetoresistance", 12)]},
                 "pass3": {"entities": [], "applications": [], "relations": []}}
    aligned = resolver.resolve_extraction(extracted)
    assert extracted["pass2"]["measurements"][0]["property_class"] == OTHER, "Inputs must not be mutated"
    a, b = aligned["pass2"]["measurements"]
    assert a["property_iri"] == r1["matched_iri"] and a["property_class"] != OTHER
    assert b["property_class"] == OTHER and b["property_iri"] is None
    paragraphs = [{"id": "P1", "text": evidence["quote"], "page": 1, "kind": "paragraph"}]
    validated = validate_paper(aligned, paragraphs, profile, 0.85)
    validated = annotate_validation(validated, aligned)
    ids = {it["id"]: it for it in validated["items"]}
    assert ids["ME2"]["status"] == "needs_review", ids["ME2"]
    assert aligned["ontology_resolution_summary"]["exact_match"] >= 1

    # Adjudication tier with a scripted stand-in for the model + shared memory reuse.
    import json, tempfile
    calls = []
    def fake_ask(label, task):
        calls.append(label)
        out = []
        for m in json.loads(task[task.index("["):]):
            pick = next((c["key"] for c in m["candidates"]
                         if c["label"].casefold() == "superconducting critical temperature"), "none")
            out.append({"mention_id": m["mention_id"], "choice": pick,
                        "match_type": "same_concept" if pick != "none" else "none",
                        "confidence": 0.95, "context_dependent": False, "reason": "test"})
        return {"decisions": out}
    tc = {"pass1": {"materials": [], "samples": [], "process_steps": []},
          "pass2": {"measurements": [{**meas("ME1", "superconducting transition temperature", 2.5),
                                      "property_term": "superconducting transition temperature"}]},
          "pass3": {"entities": [], "applications": [], "relations": []}}
    with tempfile.TemporaryDirectory() as tmp:
        memory = ResolutionMemory(Path(tmp) / "memory.json")
        first = resolver.resolve_extraction(tc, ask=fake_ask, memory=memory, paper_id="A")
        m = first["pass2"]["measurements"][0]
        assert m["property_class"].casefold() == "superconducting critical temperature", m
        assert m["property_iri"] in ontology.classes
        n_calls = len(calls)
        again = resolver.resolve_extraction(tc, ask=fake_ask, memory=memory, paper_id="B")
        assert len(calls) == n_calls, "second paper should be served from resolution memory"
        assert again["ontology_resolution"][0]["method"] == "memory"
    print("PASS — adjudication + memory reuse:", m["property_class"])
    print("PASS — ontology classes:", len(ontology.classes))
    print("PASS — application classes:", len(profile.groups["application"].class_iris))
    print("PASS — statuses:", aligned["ontology_resolution_summary"])
    print("PASS — no network, no OWL edits, no invented IRIs")


if __name__ == "__main__":
    main()
