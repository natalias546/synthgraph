"""Stage 8 — Evaluation against hand-annotated gold files (gold/<paper_id>.json).

Gold format (see gold/README.md):
{
  "paper_id": "...",
  "samples":       [{"name": "sample B", "aliases": ["annealed film"], "ontology_class": "thin film"}],
  "process_steps": [{"ontology_class": "annealing", "sample": "sample B",
                     "parameters": [{"attribute_class": "process temperature", "value": 400, "unit": "°C"}]}],
  "measurements":  [{"property_class": "carrier lifetime", "technique_class": "quasi-steady-state photoconductance",
                     "sample": "sample B", "value": 2.3, "unit": "ms"}],
  "relations":     [{"subject": "annealing", "predicate": "influences", "object": "carrier lifetime", "trend": "increases"}]
}
Scores are precision / recall / F1 per item type. Matching is deliberately simple and
transparent; inspect the per-item match tables, not only the scores.
"""
from __future__ import annotations

import json
from pathlib import Path

from .resolve import material_key


def _close(a, b, rel=0.02):
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


def _norm(s):
    return material_key(str(s or ""))


def _prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else None
    r = tp / n_gold if n_gold else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if (p == 0 or r == 0) else None)
    return {"tp": tp, "predicted": n_pred, "gold": n_gold, "precision": p, "recall": r, "f1": f}


def _greedy(pred, gold, match):
    used, pairs = set(), []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if j not in used and match(p, g):
                used.add(j)
                pairs.append((i, j))
                break
    return pairs


def evaluate_paper(resolved: dict, gold: dict, include=("auto_accepted", "needs_review")) -> dict:
    items = [it for it in resolved["items"] if it["status"] in include]
    samples = [it["data"] for it in items if it["kind"] == "sample"]
    names = {s["id"]: {_norm(s["name_in_paper"]), *map(_norm, s.get("aliases", []))} for s in samples}
    steps = [it["data"] for it in items if it["kind"] == "process_step"]
    meas = [it["data"] for it in items if it["kind"] == "measurement"]
    rels = [it["data"] for it in items if it["kind"] == "relation"]
    ents = {it["data"]["id"]: {_norm(it["data"]["name"]), *map(_norm, it["data"].get("aliases", []))}
            for it in items if it["kind"] == "entity"}
    out, details = {}, {}

    gs = gold.get("samples", [])
    pairs = _greedy(samples, gs, lambda p, g: bool(names[p["id"]] & ({_norm(g["name"])} | set(map(_norm, g.get("aliases", []))))))
    out["samples"] = _prf(len(pairs), len(samples), len(gs))
    out["samples_class_accuracy"] = (sum(samples[i]["ontology_class"] == gs[j].get("ontology_class") for i, j in pairs) / len(pairs)) if pairs else None

    gp = gold.get("process_steps", [])
    pairs = _greedy(steps, gp, lambda p, g: p["ontology_class"] == g["ontology_class"])
    out["process_steps"] = _prf(len(pairs), len(steps), len(gp))
    pred_params = [(p["ontology_class"], x["attribute_class"], x.get("value")) for p in steps for x in p["parameters"]]
    gold_params = [(g["ontology_class"], x["attribute_class"], x.get("value")) for g in gp for x in g.get("parameters", [])]
    pp = _greedy(pred_params, gold_params, lambda p, g: p[0] == g[0] and p[1] == g[1] and _close(p[2], g[2]))
    out["process_parameters"] = _prf(len(pp), len(pred_params), len(gold_params))

    gm = gold.get("measurements", [])
    pairs = _greedy(meas, gm, lambda p, g: p["property_class"] == g["property_class"] and _close(p.get("value"), g.get("value")))
    out["measurements_value"] = _prf(len(pairs), len(meas), len(gm))
    linked = 0
    for i, j in pairs:
        gname = _norm(gm[j].get("sample"))
        if gname and any(gname in names.get(sid, set()) for sid in meas[i]["sample_ids"]):
            linked += 1
    out["measurement_sample_link_accuracy"] = linked / len(pairs) if pairs else None
    out["measurement_technique_accuracy"] = (sum(meas[i]["technique_class"] == gm[j].get("technique_class") for i, j in pairs) / len(pairs)) if pairs else None
    details["measurement_matches"] = [{"pred": meas[i]["id"], "gold_index": j} for i, j in pairs]

    gr = gold.get("relations", [])

    def rel_ends(r):
        def name(x):
            if x.startswith("PROP:"):
                return {_norm(x[5:])}
            for s in samples:
                if s["id"] == x:
                    return names[s["id"]]
            for p in steps:
                if p["id"] == x:
                    return {_norm(p["ontology_class"]), _norm(p["name_in_text"])}
                for prm in p["parameters"]:
                    if prm["id"] == x:
                        return {_norm(prm["attribute_class"]), _norm(prm["name_in_text"]), _norm(p["ontology_class"])}
            if x in ents:
                return ents[x]
            return {_norm(x)}
        return name(r["subject_id"]), name(r["object_id"])

    pairs = _greedy(rels, gr, lambda p, g: p["predicate"] == g["predicate"]
                    and _norm(g["subject"]) in rel_ends(p)[0] and _norm(g["object"]) in rel_ends(p)[1])
    out["relations"] = _prf(len(pairs), len(rels), len(gr))
    return {"scores": out, "details": details}


def draft_gold_from_extraction(resolved: dict, path: Path):
    """Write the extraction in gold format as a STARTING POINT for a human annotator.

    Correct it against the paper and save as gold/<paper_id>.json. Beware anchoring
    bias: also add what the model missed, don't only delete what it got wrong.
    """
    data = lambda k: [it["data"] for it in resolved["items"] if it["kind"] == k and it["status"] != "rejected"]  # noqa: E731
    sname = {s["id"]: s["name_in_paper"] for s in data("sample")}
    readable = dict(sname)
    for p in data("process_step"):
        readable[p["id"]] = p["ontology_class"]
        for x in p["parameters"]:
            readable[x["id"]] = x["attribute_class"]
    for e in data("entity"):
        readable[e["id"]] = e["name"]
    nm = lambda x: x[5:] if x.startswith("PROP:") else readable.get(x, x)  # noqa: E731
    gold = {
        "_note": "DRAFT generated from model output. Correct, complete, then rename to <paper_id>.json.",
        "samples": [{"name": s["name_in_paper"], "aliases": s.get("aliases", []), "ontology_class": s["ontology_class"]}
                    for s in data("sample")],
        "process_steps": [{"ontology_class": p["ontology_class"],
                           "sample": ", ".join(sname.get(x, x) for x in p["output_sample_ids"]),
                           "parameters": [{"attribute_class": x["attribute_class"], "value": x.get("value"), "unit": x.get("unit")}
                                          for x in p["parameters"]]} for p in data("process_step")],
        "measurements": [{"property_class": m["property_class"], "technique_class": m["technique_class"],
                          "sample": ", ".join(sname.get(x, x) for x in m["sample_ids"]),
                          "value": m.get("value"), "unit": m.get("unit")} for m in data("measurement")],
        "relations": [{"subject": nm(r["subject_id"]), "predicate": r["predicate"], "object": nm(r["object_id"]),
                       "trend": r.get("trend")} for r in data("relation")],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
