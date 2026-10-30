"""Stage 6 — Entity resolution.

Within a paper: duplicate measurements (e.g. the same value from parsed text and from
a transcribed table) are merged, keeping all evidence.
Across papers: MATERIALS are merged into global nodes by (ontology class, normalized
name/formula). Samples, process steps and measurements are NEVER merged across papers:
they are paper-specific instances. Every original surface form is kept.
"""
from __future__ import annotations

import copy
import hashlib
import re
import unicodedata

from ..ontology_loader import Profile

# Edit/extend: common spelling variants that should resolve to the same material key.
MATERIAL_SYNONYMS = {
    "c-si": "crystalline silicon", "crystalline si": "crystalline silicon", "monocrystalline silicon": "crystalline silicon",
    "mono-si": "crystalline silicon", "mc-si": "multicrystalline silicon", "poly-si": "polycrystalline silicon",
    "polysilicon": "polycrystalline silicon", "a-si:h": "hydrogenated amorphous silicon", "a-si": "amorphous silicon",
    "si": "silicon", "sinx": "silicon nitride", "sinx:h": "silicon nitride", "si3n4": "silicon nitride",
    "sio2": "silicon dioxide", "siox": "silicon oxide", "al2o3": "aluminium oxide", "alox": "aluminium oxide",
    "aluminum oxide": "aluminium oxide", "ito": "indium tin oxide",
}


def material_key(name: str, formula: str | None = None) -> str:
    s = unicodedata.normalize("NFKC", (formula or name or "")).strip().casefold()
    s = re.sub(r"\s+", " ", s).replace("–", "-")
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"\s*(wafers?|films?|layers?|substrates?|thin films?)$", "", s)
    return MATERIAL_SYNONYMS.get(s, s)


def _value_key(m: dict):
    v = m.get("value_si") if m.get("value_si") is not None else m.get("value")
    return (tuple(sorted(m.get("sample_ids") or [])), tuple(sorted(m.get("material_ids") or [])),
            m.get("property_class"), None if v is None else round(float(v), 6), m.get("data_origin"))


def resolve_paper(validated: dict) -> dict:
    """Merge duplicate measurements within one paper (keeps all evidence and both IDs)."""
    validated = copy.deepcopy(validated)
    kept, seen, merges = [], {}, []
    for it in validated["items"]:
        if it["kind"] == "measurement" and it["status"] != "rejected" and it["data"].get("value") is not None:
            k = _value_key(it["data"])
            if k in seen:
                first = seen[k]
                first["evidence_anchors"] += it["evidence_anchors"]
                first.setdefault("merged_ids", []).append(it["id"])
                merges.append({"kept": first["id"], "merged": it["id"]})
                continue
            seen[k] = it
        kept.append(it)
    return {**validated, "items": kept, "measurement_merges": merges}


def global_material_id(profile: Profile, ontology_class: str, name: str, formula: str | None) -> str:
    iri = profile.iri_for("material", ontology_class) or ontology_class
    key = f"{iri}|{material_key(name, formula)}"
    return "MAT_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
