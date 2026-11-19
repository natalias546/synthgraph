"""Link extracted materials to Materials Project (MP) entries.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Callable

from .io_utils import sha256_json

MATERIALS_LINK_VERSION = "1.0.0"
MP_URL = "https://next-gen.materialsproject.org/materials/{}"
MP_FIELDS = ["material_id", "formula_pretty", "symmetry", "energy_above_hull", "is_stable",
             "theoretical", "deprecated", "band_gap", "database_IDs"]
LINKED = ("linked",)
REVIEW_STATUSES = ("ambiguous", "space_group_conflict", "invalid_formula", "variable_composition",
                   "not_looked_up", "lookup_error")
MAX_ATOMS = 200            # "CN2010" parses as C1N2010: a label, not a compound

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_FORMULA_CHARS = re.compile(r"^(?:[A-Z][a-z]?|\d+(?:\.\d+)?|[()\[\]])+$")
_VARIABLE = re.compile(r"[xyzδ±,+]|(?<=\d)-|-(?=[xyzδ\d])")
_SG_KEYWORD = re.compile(r"space[\s-]*group|symmetry|crystalli[sz]es?\s+in|structure\s+type", re.I)
# The keyword must sit right before the symbol: at most a few descriptive words in between.
_SG_LEAD = re.compile(r"(?:space[\s-]*group|symmetry|crystalli[sz](?:es|ed|ing)?\s+in(?:to)?|structure\s+type)"
                      r"(?:\s+(?:the|a|an|of|is|with|hexagonal|cubic|tetragonal|orthorhombic|trigonal|rhombohedral|"
                      r"monoclinic|triclinic|centrosymmetric|non-?centrosymmetric|polar|chiral)|\s*[=:,])*\s*$", re.I)
_SG_TOKEN = re.compile(r"(?<![A-Za-z0-9])([PIFCABR]\s?[-\d₀-₉_/a-z̄¯\u0305]{1,10})(?![A-Za-z0-9/])")
_SG_NUMBER = re.compile(r"\(?\s*No\.?\s*(\d{1,3})\s*\)?", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


# ---------------------------------------------------------------------------
# 1. Formula normalization (local)
# ---------------------------------------------------------------------------
def _element_names() -> dict[str, str]:
    from pymatgen.core.periodic_table import Element
    return {e.long_name.casefold(): e.symbol for e in Element}


def normalize_formula(formula: str | None, name: str | None = None) -> dict:
    """Return {"status": "ok"|..., "reduced_formula", "elements", "input", "reason"}.

    "ok" means the string is a real, fixed-stoichiometry composition MP can be searched for.
    """
    from pymatgen.core import Composition
    from pymatgen.core.periodic_table import Element

    raw = (formula or "").strip()
    if not raw and name:
        single = _element_names().get(name.strip().casefold())
        if single:
            raw = single
    if not raw:
        candidate = (name or "").strip()
        if candidate and _FORMULA_CHARS.match(unicodedata.normalize("NFKC", candidate).translate(_SUBSCRIPTS).replace(" ", "")):
            raw = candidate
    if not raw:
        return {"status": "no_formula", "input": formula or name, "reduced_formula": None,
                "elements": [], "reason": "no chemical formula given"}
    s = unicodedata.normalize("NFKC", raw.translate(_SUBSCRIPTS)).replace(" ", "")
    if _VARIABLE.search(s):
        return {"status": "variable_composition", "input": raw, "reduced_formula": None, "elements": [],
                "reason": "variable, doped or mixed-site notation; MP holds fixed stoichiometries"}
    if not _FORMULA_CHARS.match(s):
        return {"status": "invalid_formula", "input": raw, "reduced_formula": None, "elements": [],
                "reason": "contains characters that are not part of a chemical formula"}
    symbols = re.findall(r"[A-Z][a-z]?", s)
    bad = [x for x in symbols if not Element.is_valid_symbol(x)]
    if bad:
        return {"status": "invalid_formula", "input": raw, "reduced_formula": None, "elements": [],
                "reason": f"not element symbols: {', '.join(sorted(set(bad)))}"}
    try:
        comp = Composition(s)
    except Exception as exc:  # pymatgen raises several exception types
        return {"status": "invalid_formula", "input": raw, "reduced_formula": None, "elements": [],
                "reason": f"pymatgen could not parse it ({type(exc).__name__})"}
    if comp.num_atoms > MAX_ATOMS:
        return {"status": "invalid_formula", "input": raw, "reduced_formula": None, "elements": [],
                "reason": f"{comp.num_atoms:g} atoms per formula unit looks like a label, not a compound"}
    if any(abs(a - round(a)) > 1e-6 for a in comp.get_el_amt_dict().values()):
        return {"status": "variable_composition", "input": raw, "reduced_formula": None, "elements": [],
                "reason": "non-integer stoichiometry (doped/disordered); MP holds ordered structures"}
    return {"status": "ok", "input": raw, "reduced_formula": comp.reduced_formula,
            "elements": sorted(e.symbol for e in comp.elements), "reason": None}


# ---------------------------------------------------------------------------
# 2. Space-group cues from the paper's own text
# ---------------------------------------------------------------------------
def _short_symbol(n: int) -> str:
    """'P12_1/c1' -> 'P2_1/c' (the short form MP and most papers print)."""
    from pymatgen.symmetry.groups import sg_symbol_from_int_number
    sym = sg_symbol_from_int_number(n)
    return re.sub(r"^([PCABIFR])1(.+)1$", r"\1\2", sym) if 3 <= n <= 15 else sym


def parse_space_group(token: str) -> tuple[int, str] | None:
    """Validated (international number, symbol) for a Hermann-Mauguin token, else None.

    Accepts common print variants: bar as a combining overline or macron (R3̄m), screw axes
    without underscore (P63/mmc, P21/c), and the old cubic notation without a bar (Fm3m)."""
    from pymatgen.symmetry.groups import SpaceGroup
    t = unicodedata.normalize("NFD", token.strip().replace(" ", "").translate(_SUBSCRIPTS))
    t = re.sub(r"(\d)[̄̅¯]", r"-\1", t)          # 3̄ -> -3
    t = unicodedata.normalize("NFC", t)
    screw = re.sub(r"(?<=[2346])([1-5])(?=[/a-z]|$)", r"_\1", t)  # P21/c -> P2_1/c, P63 -> P6_3
    for variant in dict.fromkeys((t, screw, re.sub(r"(?<=[a-z])3", "-3", t), re.sub(r"(?<=[a-z])3", "-3", screw))):
        try:
            n = SpaceGroup(variant).int_number
            return n, _short_symbol(n)
        except Exception:
            continue
    return None


def space_group_cues(text: str) -> list[dict]:
    """Space groups stated in `text`. A token counts only after a keyword ("space group",
    "crystallizes in", ...) within 40 characters, or as an explicit "(No. N)" after one, or when it
    is unmistakable on its own (contains / - or _ and is 4+ characters, e.g. P6/mmm, R-3m)."""
    cues = []
    for m in _SG_TOKEN.finditer(text):
        token = m.group(1)
        before = text[max(0, m.start() - 60):m.start()]
        distinctive = len(token) >= 4 and any(c in token for c in "/-_̄¯")
        if not (_SG_LEAD.search(before) or distinctive):
            continue
        parsed = parse_space_group(token)
        if parsed:
            cues.append({"number": parsed[0], "symbol": parsed[1], "text": token, "start": m.start()})
    for m in _SG_NUMBER.finditer(text):
        n = int(m.group(1))
        before = text[max(0, m.start() - 60):m.start()]
        if 1 <= n <= 230 and _SG_KEYWORD.search(re.split(r"[.;]\s", before)[-1]):
            cues.append({"number": n, "symbol": _short_symbol(n),
                         "text": m.group(0).strip(), "start": m.start()})
    return cues


def _mentions(text: str, forms: set[str]) -> bool:
    return any(re.search(r"(?<![A-Za-z0-9])" + re.escape(f) + r"(?![a-z0-9])", text) for f in forms if f)


def material_space_groups(paragraphs: list[dict], forms_by_material: dict[str, set[str]]) -> dict[str, list[dict]]:
    """Attach cues to materials: same sentence as the formula, or the same paragraph when that
    paragraph mentions only one of the paper's materials."""
    out = {mid: [] for mid in forms_by_material}
    for p in paragraphs:
        text = p.get("text") or ""
        if not text or not space_group_cues(text):
            continue
        present = [mid for mid, forms in forms_by_material.items() if _mentions(text, forms)]
        for sentence in _SENTENCE.split(text):
            cues = space_group_cues(sentence)
            if not cues:
                continue
            here = [mid for mid in present if _mentions(sentence, forms_by_material[mid])]
            targets = here or (present if len(present) == 1 else [])
            for mid in targets:
                for c in cues:
                    out[mid].append({**c, "paragraph_id": p.get("id"), "sentence": sentence[:300]})
    return out


# ---------------------------------------------------------------------------
# 3. Materials Project lookup (batched, cached, injectable)
# ---------------------------------------------------------------------------
def _compact(doc: dict) -> dict:
    sym = doc.get("symmetry") or {}
    if not isinstance(sym, dict):
        sym = getattr(sym, "model_dump", lambda: {})() or {}
    ids = doc.get("database_IDs") or {}
    icsd = [str(x) for x in (ids.get("icsd") or [])][:5] if isinstance(ids, dict) else []
    mid = str(doc.get("material_id"))
    return {"material_id": mid, "formula": doc.get("formula_pretty"),
            "space_group_symbol": sym.get("symbol"), "space_group_number": sym.get("number"),
            "crystal_system": str(sym.get("crystal_system")) if sym.get("crystal_system") else None,
            "energy_above_hull_ev": doc.get("energy_above_hull"), "is_stable": doc.get("is_stable"),
            "theoretical": doc.get("theoretical"), "band_gap_ev": doc.get("band_gap"),
            "icsd_ids": icsd, "deprecated": bool(doc.get("deprecated")), "url": MP_URL.format(mid)}


class MPLookup:
    """Cached Materials Project formula search.

    `search_fn(formulas: list[str]) -> list[dict]` performs the request; by default it is
    mp_api's MPRester (key from MP_API_KEY). Tests inject a fake. Results — including empty ones —
    are cached per reduced formula in cache_dir/<db_version or 'unpinned'>/<formula>.json.
    """

    def __init__(self, cache_dir: Path, api_key: str | None = None, db_version: str | None = None,
                 allow_live: bool = True, search_fn: Callable[[list[str]], list[dict]] | None = None):
        self.api_key = api_key if api_key is not None else (os.getenv("MP_API_KEY") or "").strip() or None
        self.db_version = db_version
        self.allow_live = allow_live
        self._search_fn = search_fn
        self.cache_dir = Path(cache_dir) / (db_version or "unpinned")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.requests = 0
        self.observed_db_version = db_version

    @property
    def can_query(self) -> bool:
        return self.allow_live and (self._search_fn is not None or bool(self.api_key))

    def fingerprint(self) -> dict:
        return {"v": MATERIALS_LINK_VERSION, "db_version": self.db_version, "can_query": self.can_query,
                "fields": MP_FIELDS}

    def _path(self, formula: str) -> Path:
        return self.cache_dir / (re.sub(r"[^A-Za-z0-9]", "_", formula) + ".json")

    def _mprester_search(self, formulas: list[str]) -> list[dict]:
        from mp_api.client import MPRester
        kw = {"api_key": self.api_key, "use_document_model": False, "mute_progress_bars": True}
        if self.db_version:
            kw["db_version"] = self.db_version
        with MPRester(**kw) as mpr:
            try:
                self.observed_db_version = mpr.get_database_version()
            except Exception:
                pass
            docs = mpr.materials.summary.search(formula=formulas, deprecated=False, fields=MP_FIELDS)
        return [d if isinstance(d, dict) else d.model_dump() for d in docs]

    def search(self, formulas: list[str]) -> tuple[dict[str, list[dict]], list[str], str | None]:
        """(results by reduced formula, formulas that could not be looked up, error message)."""
        from pymatgen.core import Composition
        found, missing = {}, []
        for f in dict.fromkeys(formulas):
            p = self._path(f)
            if p.exists():
                found[f] = json.loads(p.read_text(encoding="utf-8"))["entries"]
            else:
                missing.append(f)
        if not missing:
            return found, [], None
        if not self.can_query:
            return found, missing, ("MP_API_KEY not set" if not self.api_key else "live MP lookups disabled")
        try:
            docs = (self._search_fn or self._mprester_search)(missing)
            self.requests += 1
        except Exception as exc:
            return found, missing, f"{type(exc).__name__}: {str(exc)[:200]}"
        grouped: dict[str, list[dict]] = {f: [] for f in missing}
        for d in docs:
            try:
                key = Composition(d.get("formula_pretty") or "").reduced_formula
            except Exception:
                continue
            if key in grouped and not d.get("deprecated"):
                grouped[key].append(_compact(d))
        for f, entries in grouped.items():
            entries.sort(key=lambda e: (e["energy_above_hull_ev"] is None, e["energy_above_hull_ev"] or 0))
            self._path(f).write_text(json.dumps({"formula": f, "db_version": self.observed_db_version,
                                                 "entries": entries}, indent=1), encoding="utf-8")
            found[f] = entries
        return found, [], None


# ---------------------------------------------------------------------------
# 4. Per-paper linking
# ---------------------------------------------------------------------------
def _decide(entries: list[dict], cues: list[dict]) -> dict:
    numbers = sorted({c["number"] for c in cues})
    cue_note = {"paper_space_groups": [f"{c['symbol']} (No. {c['number']})" for c in
                                       {c["number"]: c for c in cues}.values()]}
    if not entries:
        return {"status": "not_in_mp", "mp_id": None, "match_basis": None, **cue_note}
    if len(entries) == 1:
        e = entries[0]
        if numbers and e["space_group_number"] not in numbers:
            return {"status": "space_group_conflict", "mp_id": None, "match_basis": None,
                    "reason": f"paper states {', '.join(cue_note['paper_space_groups'])}; "
                              f"the only MP entry is {e['space_group_symbol']}", **cue_note}
        return {"status": "linked", "mp_id": e["material_id"], "match_basis": "unique_formula", **cue_note}
    if len(numbers) == 1:
        hits = [e for e in entries if e["space_group_number"] == numbers[0]]
        if len(hits) == 1:
            return {"status": "linked", "mp_id": hits[0]["material_id"], "match_basis": "space_group", **cue_note}
        reason = (f"{len(hits)} MP entries share space group No. {numbers[0]}" if hits else
                  f"no MP polymorph has the stated space group No. {numbers[0]}")
        return {"status": "ambiguous" if hits else "space_group_conflict", "mp_id": None,
                "match_basis": None, "reason": reason, **cue_note}
    reason = (f"{len(entries)} MP polymorphs; the paper states several space groups"
              if len(numbers) > 1 else f"{len(entries)} MP polymorphs and no space group stated near this material")
    return {"status": "ambiguous", "mp_id": None, "match_basis": None, "reason": reason, **cue_note}


def link_paper(resolved: dict, structure: dict, lookup: MPLookup) -> dict:
    """Link every non-rejected material item of one paper. Never raises on lookup failure."""
    materials = [it for it in resolved.get("items", []) if it["kind"] == "material" and it["status"] != "rejected"]
    norm = {it["id"]: normalize_formula(it["data"].get("formula"), it["data"].get("name")) for it in materials}
    forms = {}
    for it in materials:
        d, n = it["data"], norm[it["id"]]
        forms[it["id"]] = {x for x in (d.get("formula"), d.get("name"), n.get("reduced_formula"),
                                       *(d.get("aliases") or [])) if isinstance(x, str) and len(x) > 1}
    cues = material_space_groups(structure.get("paragraphs", []), forms)
    queryable = sorted({n["reduced_formula"] for n in norm.values() if n["status"] == "ok"})
    results, missing, error = lookup.search(queryable) if queryable else ({}, [], None)

    links = []
    for it in materials:
        d, n = it["data"], norm[it["id"]]
        row = {"local_id": it["id"], "name": d.get("name"), "formula": d.get("formula"),
               "item_status": it["status"], "formula_check": n["status"],
               "reduced_formula": n["reduced_formula"], "elements": n["elements"],
               "space_group_cues": cues.get(it["id"], [])}
        if n["status"] != "ok":
            row.update({"status": n["status"] if n["status"] != "no_formula" else "invalid_formula",
                        "mp_id": None, "match_basis": None, "candidates": [], "reason": n["reason"]})
        elif n["reduced_formula"] in missing:
            row.update({"status": "lookup_error" if error and "not set" not in error and "disabled" not in error
                        else "not_looked_up", "mp_id": None, "match_basis": None, "candidates": [],
                        "reason": error})
        else:
            entries = results.get(n["reduced_formula"], [])
            row.update(_decide(entries, row["space_group_cues"]))
            row["candidates"] = entries
            row.setdefault("reason", None)
        row["requires_review"] = row["status"] in REVIEW_STATUSES
        links.append(row)

    summary: dict[str, int] = {}
    for r in links:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    return {"links": links, "summary": summary, "db_version": lookup.observed_db_version,
            "mp_requests": lookup.requests, "lookup_error": error,
            "source": "Materials Project (computed, DFT)", "version": MATERIALS_LINK_VERSION,
            "fingerprint": sha256_json(lookup.fingerprint())}


def write_links_csv(links_by_paper: dict, path: Path) -> int:
    """One row per (paper, material): what was linked, how, and what needs a look."""
    import csv
    cols = ["paper_id", "local_id", "name", "formula", "reduced_formula", "status", "mp_id", "match_basis",
            "paper_space_groups", "n_candidates", "candidates", "reason", "db_version"]
    rows = []
    for pid, res in links_by_paper.items():
        for r in res.get("links", []):
            rows.append({"paper_id": pid, **{k: r.get(k) for k in ("local_id", "name", "formula", "reduced_formula",
                                                                   "status", "mp_id", "match_basis", "reason")},
                         "paper_space_groups": "; ".join(r.get("paper_space_groups", [])),
                         "n_candidates": len(r.get("candidates", [])),
                         "candidates": "; ".join(f"{c['material_id']} {c['space_group_symbol']} "
                                                 f"Ehull={c['energy_above_hull_ev']}" for c in r.get("candidates", [])[:8]),
                         "db_version": res.get("db_version")})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
