"""Stage 5 — Validation: code checks what the model claims (form and provenance, not truth).

Checks per item:
  * evidence quotes exist in the cited paragraph (no fuzzy repair; near-misses only diagnosed)
  * referenced IDs exist; relation subject/object types match the profile's domain/range
  * the reported number actually appears in the cited evidence paragraph(s)
  * OTHER classes carry a proposed label (-> ontology gap list)
  * units parse (pint) -> SI value for cross-paper comparison
Status: auto_accepted | needs_review | rejected. `verified_semantic_truth` is always False.
"""
from __future__ import annotations

import re
from collections import Counter

from .clean import match_key
from ..ontology_loader import OTHER, Profile

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
_UREG = None


def _ureg():
    global _UREG
    if _UREG is None:
        import pint
        u = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
        for line in ("sccm = 1e-6 * meter**3 / minute", "slm = 1e-3 * meter**3 / minute",
                     "sun = 1000 * watt / meter**2", "at_percent = 0.01 = at%", "wt_percent = 0.01 = wt%",
                     "ohm_per_square = ohm = ohm_sq", "ppm_atomic = 1e-6", "Torr = torr"):
            try:
                u.define(line)
            except Exception:
                pass
        _UREG = u
    return _UREG


def normalize_unit_text(unit: str) -> str:
    u = unit.strip()
    rep = {"^{": "^", "}": "", "²": "^2", "³": "^3", "⁻": "-", "¹": "1", "°C": "degC", "° C": "degC", "℃": "degC",
           "°": "deg", "Ω": "ohm", "µ": "u", "μ": "u", "Å": "angstrom", "×": "*", "·": "*",
           "%": "percent", "Ω/□": "ohm/sq", "ohm/sq": "ohm_per_square", "Ω/sq": "ohm_per_square"}
    for k, v in rep.items():
        u = u.replace(k, v)
    u = re.sub(r"\bat\.?\s*percent", "at_percent", u)
    u = re.sub(r"\bwt\.?\s*percent", "wt_percent", u)
    u = re.sub(r"([A-Za-z])(-?\d)\b", r"\1^\2", u)          # cm-3 -> cm^-3, cm2 -> cm^2
    u = re.sub(r"\s+", " ", u)
    u = u.replace(" ", "*") if not re.search(r"[*/]", u) and " " in u else u
    return u


def normalize_quantity(value, unit):
    """Return (value_si, unit_si, unit_normalized) or (None, None, reason)."""
    if value is None or not unit:
        return None, None, None
    try:
        ureg = _ureg()
        q = ureg.Quantity(float(value), normalize_unit_text(unit))
        base = q.to_base_units()
        return float(base.magnitude), f"{base.units:~}", f"{q.units:~}"
    except Exception as exc:
        return None, None, f"unparsed unit: {type(exc).__name__}"


# ---------------------------------------------------------------------------
# Evidence anchoring
# ---------------------------------------------------------------------------
def _numbers(text: str) -> set[str]:
    t = match_key(text or "").replace(",", "")
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+(?:\.\d+)?", t)}


def _fmt_number(x: float) -> set[str]:
    out = {repr(x), f"{x:g}"}
    if float(x).is_integer():
        out.add(str(int(x)))
    return {s.lstrip("0") or "0" for s in out}


class Anchorer:
    def __init__(self, paragraphs: list[dict]):
        self.text = {p["id"]: p["text"] for p in paragraphs}
        self.key = {pid: match_key(t) for pid, t in self.text.items()}
        self.page = {p["id"]: p["page"] for p in paragraphs}
        self.kind = {p["id"]: p["kind"] for p in paragraphs}

    def anchor(self, ev: dict) -> dict:
        q = match_key(ev.get("quote", ""))
        pid = ev.get("paragraph_id")
        res = {**ev, "page": self.page.get(pid)}
        if not q:
            return {**res, "anchor": "empty_quote"}
        if pid in self.key and q in self.key[pid]:
            return {**res, "anchor": "exact", "source_kind": self.kind[pid]}
        other = next((p for p, k in self.key.items() if q in k), None)
        if other:
            return {**res, "anchor": "found_in_other_paragraph", "actual_paragraph_id": other, "page": self.page[other],
                    "source_kind": self.kind[other]}
        words = q.split()
        if pid in self.key and words:
            hit = sum(w in self.key[pid] for w in words) / len(words)
            if hit >= 0.8:
                return {**res, "anchor": "near_miss_not_accepted", "word_overlap": round(hit, 2)}
        return {**res, "anchor": "not_found"}


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------
def _group_of(item_id: str, kinds: dict[str, str]) -> str | None:
    if item_id.startswith("PROP:"):
        return "property"
    return kinds.get(item_id)


def validate_paper(extraction: dict, paragraphs: list[dict], profile: Profile, min_conf: float) -> dict:
    anch = Anchorer(paragraphs)
    p1, p2, p3 = extraction["pass1"], extraction["pass2"], extraction["pass3"]
    items: list[dict] = []
    kinds: dict[str, str] = {}
    for m in p1["materials"]:
        kinds[m["id"]] = "material"
    for s in p1["samples"]:
        kinds[s["id"]] = "sample"
    for p in p1["process_steps"]:
        kinds[p["id"]] = "process"
        for prm in p["parameters"]:
            kinds[prm["id"]] = "process_attribute"
    for m in p2["measurements"]:
        kinds[m["id"]] = "measurement"
    other_label_group = {}
    for gname in ("application", "other"):
        for lab in profile.groups[gname].labels:
            other_label_group.setdefault(lab, gname)
    for e in p3["entities"]:
        kinds[e["id"]] = other_label_group.get(e["ontology_class"], "other")
    property_labels = set(profile.groups["property"].labels)
    dup_ids = [i for i, c in Counter(
        [x["id"] for x in p1["materials"] + p1["samples"] + p1["process_steps"] + p2["measurements"] + p3["entities"]]
    ).items() if c > 1]

    def check(kind, obj, class_field=None, group=None, proposed_field="proposed_class_label", refs=()):
        issues, notes = [], []
        anchors = [anch.anchor(e) for e in obj.get("evidence", [])]
        n_exact = sum(a["anchor"] in ("exact", "found_in_other_paragraph") for a in anchors)
        if not anchors:
            issues.append("no evidence")
        elif n_exact == 0:
            issues.append("no evidence quote found in the paper")
        elif n_exact < len(anchors):
            notes.append("some quotes not found")
        if any(a["anchor"] == "found_in_other_paragraph" for a in anchors):
            notes.append("quote found in a different paragraph than cited")
        if obj.get("id") in dup_ids:
            notes.append("duplicate ID")
        for f, allowed in refs:
            vals = obj.get(f)
            vals = vals if isinstance(vals, list) else ([vals] if vals else [])
            for v in vals:
                g = _group_of(v, kinds)
                if g is None:
                    issues.append(f"{f}: unknown ID {v}")
                elif allowed and g not in allowed:
                    notes.append(f"{f}: {v} is a {g}, expected {'/'.join(allowed)}")
        gap = None
        if class_field:
            cls = obj.get(class_field)
            if cls == OTHER:
                prop = obj.get(proposed_field)
                if not prop:
                    notes.append(f"{class_field}=OTHER without a proposed label")
                gap = {"group": group, "proposed_label": prop, "text": obj.get("name") or obj.get("name_in_text")
                       or obj.get("name_in_paper") or obj.get("property_name_in_text") or obj.get("technique_name_in_text")}
                notes.append("class not in ontology (gap)")
        conf = obj.get("confidence")
        if conf is not None and not (0 <= conf <= 1):
            notes.append(f"confidence out of range ({conf})")
            obj["confidence"] = min(1.0, max(0.0, conf))
        pol = obj.get("polarity")
        if pol in ("negated", "hypothetical"):
            notes.append(f"polarity={pol}")
        status = "rejected" if issues else (
            "auto_accepted" if not notes and (obj.get("confidence") or 0) >= min_conf else "needs_review")
        return {"kind": kind, "id": obj.get("id"), "status": status, "issues": issues, "notes": notes,
                "evidence_anchors": anchors, "ontology_gap": gap, "verified_semantic_truth": False, "data": obj}

    for m in p1["materials"]:
        items.append(check("material", m, "ontology_class", "material"))
    for s in p1["samples"]:
        items.append(check("sample", s, "ontology_class", "sample", refs=[("material_ids", ["material"])]))
    for p in p1["process_steps"]:
        it = check("process_step", p, "ontology_class", "process", refs=[
            ("input_ids", ["sample", "material"]), ("output_sample_ids", ["sample"]),
            ("precursor_material_ids", ["material"]), ("follows_step_id", ["process"])])
        for prm in p["parameters"]:
            pv = check("parameter", {**prm, "confidence": p.get("confidence")}, "attribute_class", "process_attribute")
            if prm.get("value") is not None:
                ev_nums = set().union(*[_numbers(anch.text.get(a.get("actual_paragraph_id") or a["paragraph_id"], ""))
                                        for a in pv["evidence_anchors"]] or [set()])
                if not (_fmt_number(prm["value"]) & ev_nums):
                    pv["notes"].append("numeric value not found in evidence paragraph")
                    pv["status"] = "needs_review" if pv["status"] == "auto_accepted" else pv["status"]
            si = normalize_quantity(prm.get("value"), prm.get("unit"))
            prm["value_si"], prm["unit_si"], prm["unit_normalized"] = si
            pv["parent_id"] = p["id"]
            items.append(pv)
        items.append(it)
    for m in p2["measurements"]:
        it = check("measurement", m, "property_class", "property", proposed_field="proposed_property_label",
                   refs=[("sample_ids", ["sample"]), ("material_ids", ["material"])])
        if m["technique_class"] == OTHER:
            it["notes"].append("technique not in ontology (gap)")
            it["technique_gap"] = {"group": "technique", "proposed_label": m.get("proposed_technique_label"),
                                   "text": m.get("technique_name_in_text")}
        if not m["sample_ids"] and not m["material_ids"]:
            it["notes"].append("value not linked to a sample or material")
        if m.get("data_origin") != "experimental":
            it["notes"].append(f"data_origin={m.get('data_origin')}")
        if m.get("value") is not None:
            ev_nums = set().union(*[_numbers(anch.text.get(a.get("actual_paragraph_id") or a["paragraph_id"], ""))
                                    for a in it["evidence_anchors"]] or [set()])
            if not (_fmt_number(m["value"]) & ev_nums):
                it["notes"].append("numeric value not found in evidence paragraph")
        if m.get("value") is not None and m.get("value_max") is not None and m["value_max"] < m["value"]:
            it["notes"].append("value_max < value")
        m["value_si"], m["unit_si"], m["unit_normalized"] = normalize_quantity(m.get("value"), m.get("unit"))
        if it["notes"] and it["status"] == "auto_accepted":
            it["status"] = "needs_review"
        items.append(it)
    for e in p3["entities"]:
        items.append(check("entity", e, "ontology_class", kinds.get(e["id"], "other")))
    for i, r in enumerate(p3["relations"], 1):
        r.setdefault("id", f"R{i}")
        spec = profile.relations.get(r["predicate"])
        subj_groups = spec.subject_groups if spec else None
        obj_groups = spec.object_groups if spec else None
        it = check("relation", r, refs=[("subject_id", subj_groups), ("object_id", obj_groups)])
        if r["object_id"].startswith("PROP:") and r["object_id"][5:] not in property_labels:
            it["notes"].append(f"unknown property label {r['object_id'][5:]}")
        if spec and spec.ontology_iri is None:
            it["notes"].append("predicate not in ontology (gap)")
        if r.get("support_type") == "indirect_document":
            it["notes"].append("indirect inference")
        if it["notes"] and it["status"] == "auto_accepted":
            it["status"] = "needs_review"
        items.append(it)

    gaps = [dict(it["ontology_gap"], item=it["id"], kind=it["kind"]) for it in items if it.get("ontology_gap")]
    gaps += [dict(it["technique_gap"], item=it["id"], kind=it["kind"]) for it in items if it.get("technique_gap")]
    counts = Counter((it["kind"], it["status"]) for it in items)
    return {
        "items": items,
        "rejected_candidates": p3.get("rejected_candidates", []),
        "ontology_gaps": gaps,
        "summary": [{"kind": k, "status": s, "count": c} for (k, s), c in sorted(counts.items())],
        "unresolved_notes": p1.get("unresolved_notes", []) + p2.get("unresolved_notes", []),
    }
