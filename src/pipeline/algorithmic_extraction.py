"""Deterministic, evidence-preserving Pass 1 / Pass 2 registry for algorithmic mode.

No ontology class assignments or LLM calls are made here. A candidate is NOT
an endorsed scientific fact. Any ambiguous association is retained explicitly
for downstream contextual extraction and human review.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from .scientific_context import quantities
from ..ontology_resolver.resolver import normalize

# Do not confuse formulas with English abbreviations or units.
_FORMULA = re.compile(r"(?<![A-Za-z])(?:[A-Z][a-z]?\d*(?:\.\d+)?){2,9}(?![A-Za-z])")
_SPECIMEN = re.compile(r"\b(?:single[ -]crystals?|polycrystalline\s+(?:samples?|pellets?)|thin\s+films?|"
                       r"samples?|specimens?|crystals?|wafers?|flakes?)\b", re.I)
_PROCESS = re.compile(r"\b(?:grow|grown|growth|synthesi[sz](?:e|ed|ing)|prepar(?:e|ed|ation)|"
                      r"anneal(?:ed|ing)?|deposit(?:ed|ion)?|fabricat(?:e|ed|ion)|"
                      r"measur(?:e|ed|ement)|perform(?:ed)?|irradiat(?:e|ed|ion)|"
                      r"cleav(?:e|ed|age)|calculat(?:e|ed|ion)|refin(?:e|ed|ement))\b", re.I)
_PRIMARY_PROPERTY = re.compile(r"\b(?:magnetoresistance|resistivity|resistance|conductivity|"
                               r"susceptibility|magnetization|mobility|carrier\s+density|"
                               r"hall\s+coefficient|band\s+gap|lattice\s+(?:parameter|constant)|"
                               r"coherence\s+length|fermi\s+level|rrr|mr)\b", re.I)
_NON_PROPERTY = re.compile(r"^(?:temperature|pressure|magnetic field|time|field|"
                           r"annealing temperature|heating rate|duration|wavelength)$", re.I)
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_UNIT = re.compile(r"(?:°\s*C|K|Torr|mTorr|Pa|GPa|MPa|mT|T|eV|meV|nm|mm|µm|μm|Å|"
                   r"mΩ\s*cm|Ω\s*cm|Ω|%|s|h|min|Oe|kOe)\s*$", re.I)


def _id(prefix: str, *args) -> str:
    return prefix + hashlib.sha256("|".join(map(str, args)).encode()).hexdigest()[:12]


def _ev(e: dict, quote: str | None = None) -> list[dict]:
    """Evidence quotes must always be literal substrings of the source."""
    s = e.get("sentence", "")
    if quote is None:
        words = list(re.finditer(r"\S+", s))
        quote = s if len(words) <= 30 else s[:words[29].end()]
    if quote not in s:
        raise ValueError("Evidence quote must be a verbatim source substring")
    return [{"paragraph_id": e["paragraph_id"], "quote": quote}]


def _numeric(raw: str):
    m = _NUMBER.search(raw.replace("−", "-"))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def _unit(raw: str):
    m = _UNIT.search(raw)
    return m.group().strip() if m else None


def _contextual_problem(issues: list, e: dict, reason: str, candidates=None, priority="medium"):
    issues.append({"id": _id("AMB_", e["id"], reason, *(candidates or [])),
                   "evidence_id": e["id"], "paragraph_id": e["paragraph_id"],
                   "sentence": e["sentence"], "reason": reason,
                   "candidate_terms": list(candidates or []), "priority": priority,
                   "resolution_status": "needs_context_not_class"})


def build_algorithmic_registry(structure: dict, candidate_stage: dict, selection: dict) -> dict:
    """Return pass1/pass2-shaped registries and explicit source-grounded ambiguities.

    No broad text generation; no OWL IRI/class inference. Explicit source
    exclusions apply BEFORE constructing any registry record.
    """
    extracted = {e["id"] for e in selection.get("extract", [])}
    if selection.get("full_paper"):
        extracted = {e["id"] for e in structure.get("semantic_evidence", [])
                     if e.get("section_role") not in {"references", "acknowledgements", "front_matter_meta"}}
    evidence = [e for e in structure.get("semantic_evidence", [])
                if e["id"] in extracted and e.get("source_kind") != "bibliographic_entry"
                and e.get("section_role") not in {"references", "acknowledgements"}]
    evidence_by_id = {e["id"]: e for e in evidence}
    candidates_by_evidence = defaultdict(list)
    for c in candidate_stage.get("candidates", []):
        if c.get("evidence_id") in extracted:
            candidates_by_evidence[c["evidence_id"]].append(c)

    p1 = {"materials": [], "samples": [], "process_steps": [], "unresolved_notes": []}
    p2 = {"measurements": [], "unresolved_notes": []}
    issues = []
    materials = {}
    samples = {}
    sample_forms = defaultdict(set)
    for e in evidence:
        txt = e.get("sentence", "")
        # Formula recognition only, without assigning a formula to a nearby
        # citation or another paper's material as a verified current specimen.
        formula_hits = [m.group() for m in _FORMULA.finditer(txt)
                        if any(ch.isdigit() for ch in m.group()) and len(m.group()) > 3]
        for formula in dict.fromkeys(formula_hits):
            key = normalize(formula).replace(" ", "")
            if key not in materials:
                mid = f"M{len(materials)+1}"
                materials[key] = mid
                p1["materials"].append({"id": mid, "name": formula,
                    "formula": formula, "original_formula_notation": formula,
                    "type_term": "material", "ontology_class": "OTHER",
                    "proposed_class_label": "material", "aliases": [], "evidence": _ev(e),
                    "confidence": 0.65, "extraction_method": "formula_pattern",
                    "review_status": "not reviewed"})
        specimens = list(_SPECIMEN.finditer(txt))
        for specimen in specimens:
            near = [f for f in formula_hits if txt.find(f) >= specimen.start() - 50
                    and txt.find(f) <= specimen.end() + 100]
            if len(set(near)) != 1:
                _contextual_problem(issues, e, "specimen reference lacks a unique local formula", [specimen.group()], "medium")
                continue
            formula = near[0]
            material_id = materials.get(normalize(formula).replace(" ", ""))
            # Distinct explicit treatment identifiers should not collapse.
            window = txt[max(0, specimen.start()-65):specimen.end()+100]
            state = re.search(r"\b(?:as[- ]grown|as[- ]prepared|annealed|untreated|treated|sample\s+[A-Z0-9]+)\b", window, re.I)
            state_name = state.group().lower() if state else "unspecified_state"
            skey = (material_id, state_name)
            if skey not in samples:
                sid = f"S{len(samples)+1}"
                samples[skey] = sid
                sample_forms[material_id].add(sid)
                p1["samples"].append({"id": sid, "name_in_paper": specimen.group(),
                    "type_term": specimen.group(), "ontology_class": "OTHER",
                    "proposed_class_label": specimen.group(), "aliases": [],
                    "description": (formula + ("; " + state_name if state else "; state not specified")),
                    "material_ids": [material_id] if material_id else [],
                    "is_simulated_only": False, "evidence": _ev(e),
                    "confidence": 0.60, "extraction_method": "specimen_formula_context",
                    "review_status": "not reviewed"})
            if len(sample_forms[material_id]) > 1:
                _contextual_problem(issues, e, "multiple possible specimen states for the same material", [formula], "high")
    for e in evidence:
        txt = e.get("sentence", "")
        # Only true experimental/manufacturing verbs, not standalone OWL group assignments.
        if e.get("section_role") == "methods":
            process = _PROCESS.search(txt)
            if process:
                formula_ids = [m["id"] for m in p1["materials"] if m["formula"] in txt]
                sample_ids = sorted({sid for mid in formula_ids for sid in sample_forms[mid]})
                if len(sample_ids) > 1:
                    _contextual_problem(issues, e, "process links to multiple sample states", sample_ids, "high")
                params = []
                for i, q in enumerate(e.get("quantity_candidates", []) or quantities(txt), 1):
                    raw = q.get("text", "")
                    params.append({"id": "", "name_in_text": q.get("quantity_kind", "parameter"),
                        "type_term": q.get("quantity_kind", "parameter"),
                        "attribute_class": "OTHER", "proposed_class_label": q.get("quantity_kind", "parameter"),
                        "symbol": None,
                        "value": _numeric(raw), "value_max": None,
                        "value_text": raw, "unit": _unit(raw), "uncertainty_text": None})
                prid = f"PR{len(p1['process_steps'])+1}"
                for i, prm in enumerate(params, 1):
                    prm["id"] = f"{prid}.p{i}"
                p1["process_steps"].append({"id": prid, "name_in_text": process.group(),
                    "type_term": process.group(), "ontology_class": "OTHER",
                    "proposed_class_label": process.group(), "step_order": None,
                    "follows_step_id": None, "input_ids": [],
                    "output_sample_ids": sample_ids[:1] if len(sample_ids)==1 else [],
                    "precursor_material_ids": formula_ids,
                    "parameters": params, "equipment": None, "software": None,
                    "polarity": "reported", "evidence": _ev(e), "confidence": 0.55,
                    "extraction_method": "method_verb_and_local_conditions"})

    # Explicit measurement candidates: a detectable quantity and a locally
    # supported property mention. Quantity-only sentences remain unresolved.
    for e in evidence:
        txt = e.get("sentence", "")
        qs = e.get("quantity_candidates", []) or quantities(txt)
        terms = [(c.get("term", ""), c) for c in candidates_by_evidence[e["id"]]
                 if c.get("group") == "property"
                 and not _NON_PROPERTY.fullmatch(c.get("term", "").strip())]
        terms.extend((m.group(), {}) for m in _PRIMARY_PROPERTY.finditer(txt))
        distinct = {}
        for term, c in terms:
            if term:
                distinct.setdefault(normalize(term), (term, c))
        properties = list(distinct.values())
        techniques = [(c["term"], c) for c in candidates_by_evidence[e["id"]]
                      if c.get("group") == "technique"]
        techniques = list({normalize(t): (t, c) for t, c in techniques}.values())
        if not qs and properties and e.get("scientific_roles"):
            _contextual_problem(issues, e, "qualitative property statement; no explicit numerical result",
                                [t for t, _ in properties])
            continue
        if not qs:
            continue
        if not properties:
            # An isolated condition/parameter must not be invented as a measurement.
            if "result" in e.get("scientific_roles", []):
                _contextual_problem(issues, e, "quantity with no explicit linked measured property",
                                    [q["text"] for q in qs], "high")
            continue
        if len(properties) > 1:
            _contextual_problem(issues, e, "more than one possible measured property",
                                [t for t, _ in properties], "high")
            continue
        prop = properties[0][0]
        if len(techniques) > 1:
            _contextual_problem(issues, e, "more than one possible technique",
                                [t for t, _ in techniques], "high")
        # Do not treat temperature/magnetic-field conditions as measured
        # property values when the target quantity is resistance, MR, etc.
        values = [q for q in qs if q.get("quantity_kind") == "scientific_quantity"]
        if not values:
            _contextual_problem(issues, e, "property present but all numbers appear to be conditions",
                                [t for t, _ in properties])
            continue
        if len(values) != 1:
            _contextual_problem(issues, e, "multiple candidate numeric values for one property",
                                [q["text"] for q in values], "high")
            continue
        q = values[0]
        own = [a for a in structure.get("source_assertions", [])
               if e["id"] in a.get("evidence_ids", [])]
        origins = {a.get("claim_origin") for a in own}
        origin = "unclear"
        if origins == {"current_study"}:
            origin = "computed" if all(a.get("claim_kind") == "calculation" for a in own) else "experimental"
        elif origins == {"cited_study"}:
            origin = "cited_from_other_work"
        elif origins:
            _contextual_problem(issues, e, "publication attribution ambiguous or mixed", sorted(origins), "high")
        if re.search(r"\b(?:our|we|this work|present study)\b", txt, re.I) and re.search(
                r"\b(?:previously|earlier|reported by|prior work|literature)\b", txt, re.I):
            # Mixed-origin sentences are not safe as individual numeric findings
            # without grammatical attribution of the particular value.
            origin = "unclear"
            _contextual_problem(issues, e, "same sentence contains current and cited study results",
                                [prop, q["text"]], "high")
        cited_ref_labels = sorted({r.get("target_label") for r in structure.get("document_references", [])
                                   if r.get("evidence_id") == e["id"] and r.get("kind") == "bibliography"
                                   and r.get("target_label")})
        mids = [m["id"] for m in p1["materials"] if m["formula"] in txt]
        sids = sorted({sid for mid in mids for sid in sample_forms[mid]})
        if len(sids) > 1:
            _contextual_problem(issues, e, "measurement sample ambiguous", sids, "high")
        if not sids:
            _contextual_problem(issues, e, "measurement lacks an unambiguous specimen link", [prop])
        settings = [{"name": q2.get("quantity_kind", "condition"), "value_text": q2["text"],
                     "symbol": None, "unit": _unit(q2["text"]), "uncertainty_text": None}
                    for q2 in qs if q2 is not q]
        p2["measurements"].append({
            "id": f"ME{len(p2['measurements'])+1}",
            "technique_name_in_text": techniques[0][0] if len(techniques)==1 else "",
            "technique_term": techniques[0][0] if len(techniques)==1 else "",
            "instrument": None, "software": None, "computational_method": None,
            "settings": [], "sample_ids": sids[:1] if len(sids)==1 else [],
            "material_ids": mids, "property_name_in_text": prop, "property_term": prop,
            "property_class": "OTHER", "proposed_property_label": prop,
            "technique_class": "OTHER", "proposed_technique_label": techniques[0][0] if len(techniques)==1 else None,
            "value": _numeric(q["text"]), "value_max": None, "value_text": q["text"],
            "unit": _unit(q["text"]), "uncertainty_text": None,
            "fitted_model_name": None, "fitted_parameter_definition": None,
            "measurement_kind": "reported_value", "parameter_symbol": None,
            "conditions": settings, "data_origin": origin,
            "attributed_reference": ("; ".join(cited_ref_labels) if origin == "cited_from_other_work"
                                     and cited_ref_labels else None),
            "cited_reference_ids": cited_ref_labels if origin == "cited_from_other_work" else [],
            "polarity": "reported", "source_location": "text",
            "evidence": _ev(e), "confidence": 0.65 if origin != "unclear" and sids else 0.45,
            "extraction_method": "local_property_quantity_candidate",
            "candidate_status": "provisional_not_verified"})

    # Preserve OWL mismatch and cross-claim ambiguity candidates, but never
    # route a generic stopword as an unresolved physical measurement.
    for e in evidence:
        terms = [c for c in candidates_by_evidence[e["id"]]
                 if c.get("match_status") != "exact_match"]
        if terms:
            _contextual_problem(issues, e, "unmatched or ambiguous ontology surface; context only",
                                [c["term"] for c in terms], "medium")
    for a in structure.get("source_assertions", []):
        if any(eid in extracted for eid in a.get("evidence_ids", [])) and a.get("claim_origin") in {
                "undetermined", "mixed_current_and_cited"} and a.get("claim_kind") in {
                "observation", "calculation", "comparison", "interpretation"}:
            e = next((evidence_by_id[i] for i in a.get("evidence_ids", []) if i in evidence_by_id), None)
            if e:
                _contextual_problem(issues, e, "unresolved assertion provenance or comparison scope",
                                    [a.get("clause", "")[:150]], "medium")
    # Identical candidate associations collapse, but distinct reasons survive.
    issues = list({i["id"]: i for i in issues}.values())
    issues.sort(key=lambda i: (i["paragraph_id"], i["evidence_id"], i["reason"]))
    p1["unresolved_notes"] = [x["reason"] + " [" + x["evidence_id"] + "]" for x in issues
                              if "sample" in x["reason"] or "specimen" in x["reason"] or "process" in x["reason"]]
    p2["unresolved_notes"] = [x["reason"] + " [" + x["evidence_id"] + "]" for x in issues
                              if "property" in x["reason"] or "measurement" in x["reason"] or "quantity" in x["reason"]]
    return {"version": "1.0", "pass1": p1, "pass2": p2,
            "ambiguities": issues, "summary": {"materials": len(p1["materials"]),
            "samples": len(p1["samples"]), "process_steps": len(p1["process_steps"]),
            "measurements": len(p2["measurements"]), "unresolved": len(issues)},
            "mode": "algorithmic", "candidate_status": "unverified_source_candidates"}
