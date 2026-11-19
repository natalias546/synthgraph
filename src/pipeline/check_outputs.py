"""Sanity checks on saved outputs: data/extractions/<paper_id>/{ingest,structure,observations}.json.

Errors mean the output is broken (missing files or fields, duplicate IDs, text lost when
splitting a paragraph across pages); warnings are quality flags worth a look (no abstract,
pages with no observations, garbled glyphs). Run after the pipeline, before the LLM step:

    python -m src.pipeline.check_outputs                  # every paper
    python -m src.pipeline.check_outputs QM039 QM056      # some papers
    python -m src.pipeline.check_outputs --dir other/out  # another outputs folder

Exits 1 if any paper has errors. From a notebook: check_all(cfg.outputs_dir).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if __package__:
    from .io_utils import read_json
    from .observations import REQUIRED
else:  # run as a file (python src/pipeline/check_outputs.py), not with -m from the project root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.pipeline.io_utils import read_json
    from src.pipeline.observations import REQUIRED

GARBLED = re.compile(r"�|\(cid:\d+\)|[-]")  # replacement char, unmapped glyph, private use
OTHER_SHARE = 0.4  # warn when more than this share of observations sit in "other" sections


def _alnum(t: str) -> str:
    return re.sub(r"[^0-9a-z]", "", t.casefold())


def check_paper(d: Path) -> dict:
    """Return {"paper_id", "errors": [...], "warnings": [...], "stats": {...}} for one output folder."""
    errors, warnings, stats = [], [], {}
    res = {"paper_id": d.name, "errors": errors, "warnings": warnings, "stats": stats}
    files = {}
    for name in ("ingest", "structure", "observations"):
        p = d / f"{name}.json"
        if not p.exists():
            errors.append(f"{p.name} missing")
            continue
        try:
            files[name] = read_json(p)
        except ValueError as exc:
            errors.append(f"{p.name} is not valid JSON: {exc}")
    if "structure" not in files or "observations" not in files:
        return res
    st, obs = files["structure"], files["observations"]

    # --- observations: the shape the LLM notebooks rely on
    stats["observations"] = len(obs)
    if not obs:
        errors.append("observations.json is empty")
        return res
    for o in obs:
        missing = REQUIRED - o.keys()
        if missing:
            errors.append(f"{o.get('id')}: missing {sorted(missing)}")
        elif not str(o["text"]).strip():
            errors.append(f"{o['id']}: empty text")
    ids = [o.get("id") for o in obs]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        errors.append(f"duplicate ids: {dups[:5]}")
    if [o.get("reading_order") for o in obs] != list(range(1, len(obs) + 1)):
        errors.append("reading_order is not 1..n in file order")
    known = set(ids)
    dangling = [o["id"] for o in obs if o.get("continues") and o["continues"] not in known]
    if dangling:
        errors.append(f"'continues' points to a missing id: {dangling[:5]}")
    if sum(o.get("kind") == "title" or str(o.get("text", "")).casefold() == st.get("title", "").casefold() for o in obs) > 1:
        warnings.append("title appears more than once")

    # --- structure: paragraphs belong to known sections; page splits keep all the text
    section_ids = {s["id"] for s in st["sections"]}
    orphans = [p["id"] for p in st["paragraphs"] if p["section_id"] not in section_ids]
    if orphans:
        errors.append(f"paragraphs with unknown section_id: {orphans[:5]}")
    split = [p for p in st["paragraphs"] if p.get("page_texts")]
    stats["split_paragraphs"] = len(split)
    for p in split:
        whole, parts = _alnum(p["text"]), _alnum("".join(x["text"] for x in p["page_texts"]))
        if abs(len(whole) - len(parts)) > 0.05 * len(whole):
            errors.append(f"{p['id']}: page pieces have {len(parts)} chars, paragraph has {len(whole)}")

    # --- quality flags
    if not st.get("title"):
        warnings.append("no title")
    roles = {o.get("role") for o in obs}
    for role in ("abstract", "methods", "references"):
        if role not in roles:
            warnings.append(f"no {role} section")
    other = sum(o.get("role") == "other" for o in obs) / len(obs)
    stats["other_share"] = round(other, 2)
    if other > OTHER_SHARE:
        warnings.append(f"{other:.0%} of observations are in 'other' sections (headings not recognised?)")
    pages = (files.get("ingest") or {}).get("source", {}).get("page_count")
    if pages:
        stats["pages"] = pages
        empty = sorted(set(range(1, pages + 1)) - {o.get("page") for o in obs})
        if empty:
            warnings.append(f"pages with no observations (blank, or text lost?): {empty}")
    garbled = [o.get("id") for o in obs if GARBLED.search(str(o.get("text", "")))]
    if garbled:
        warnings.append(f"{len(garbled)} observation(s) with unmapped glyphs, e.g. {garbled[:3]}")
    warnings.extend(f"structure: {w}" for w in st.get("warnings", []))
    return res


def check_all(outputs_dir: Path, papers=None) -> list[dict]:
    dirs = sorted(d for d in Path(outputs_dir).iterdir() if d.is_dir() and not d.name.startswith("_"))
    if papers:
        dirs = [d for d in dirs if any(p in d.name for p in papers)]
    return [check_paper(d) for d in dirs]


def report(results: list[dict], show_warnings: bool = True) -> int:
    """Print a per-paper report; return the number of papers with errors."""
    bad = 0
    for r in results:
        status = "FAIL" if r["errors"] else ("warn" if r["warnings"] else "ok")
        bad += bool(r["errors"])
        print(f"[{status:4}] {r['paper_id']}  {r['stats']}")
        for e in r["errors"]:
            print(f"    ERROR  {e}")
        for w in r["warnings"] if show_warnings else []:
            print(f"    warn   {w}")
    n_warn = sum(bool(r["warnings"]) for r in results)
    print(f"\n{len(results)} papers: {bad} with errors, {n_warn} with warnings")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("papers", nargs="*", help="paper IDs (or any part, e.g. QM039) to check; default all")
    ap.add_argument("--dir", default=Path(__file__).resolve().parents[2] / "data" / "extractions", type=Path)
    ap.add_argument("--errors-only", action="store_true", help="hide warnings")
    args = ap.parse_args(argv)
    if not args.dir.is_dir():
        print(f"No outputs folder at {args.dir}; run the pipeline first.")
        return 1
    return 1 if report(check_all(args.dir, args.papers), not args.errors_only) else 0


if __name__ == "__main__":
    sys.exit(main())
