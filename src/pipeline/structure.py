"""Stage 2 — Publication structure: title, sections with roles, paragraphs, captions, tables.

Follows the publication-structure notebook's ideas (typography-relative headings,
unidentified content kept, every block owned exactly once) in a simpler form.
Section *roles* drive routing later: process details come from `methods`,
values from `results` and tables, applications from `abstract`/`introduction`.

Steps: running headers/footers -> title -> headings -> paragraphs -> inferred
abstract -> coverage check (every ingested block has exactly one owner).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .clean import clean_text, join_lines

ROLE_PATTERNS = [
    ("abstract", r"abstract|summary of the work"),
    ("introduction", r"introduction|background|motivation"),
    ("methods", r"experimental|experiments?|experiment(al)? (section|details|procedures?|methods?)|methods?|"
                r"materials and methods|methodology|measurement methods?|sample preparation|fabrication|synthesis|"
                r"device fabrication|characteri[sz]ation|computational (details|methods?)|simulation (details|methods?)|"
                r"growth|deposition"),
    ("results", r"results?( and discussion)?|discussion|analysis|performance"),
    ("conclusion", r"conclusions?|concluding remarks|summary|outlook"),
    ("references", r"references|bibliography|literature cited|works cited"),
    ("acknowledgements", r"acknowledg(e)?ments?|funding"),
    ("author_info", r"author contributions?|conflicts? of interest|competing interests?|declaration of .*|"
                    r"(author )?declarations|author information|associated content|additional information|notes|"
                    r"data availability|credit authorship.*|orcid"),
    ("supplementary", r"supporting information|supplementary (material|information)|appendix.*"),
]
NUMBERED = re.compile(r"^(?P<num>(\d+(\.\d+)*|[IVX]+)\.?)\s+(?P<title>[A-Z].{1,120})$")
CAPTION = re.compile(r"^(?P<kind>fig(?:ure)?s?|table|scheme)\.?\s*(?P<num>S?\d+[a-z]?)\s*[.:|\-–]?\s*", re.I)

MARGIN = 0.08               # top/bottom share of the page where running headers and footers sit
FURNITURE_MIN_SHARE = 0.3   # ...and the share of pages they must repeat on
MIN_ABSTRACT_WORDS = 80


def role_for(heading: str) -> str | None:
    h = re.sub(r"[\x00-\x1f]+", " ", heading).strip()  # control characters some PDFs put between words
    h = re.sub(r"^(\d+(\.\d+)*|[IVX]+)\.?\s+", "", h).strip(" .:").lower()
    for role, pat in ROLE_PATTERNS:
        if re.fullmatch(pat, h):
            return role
    for role, pat in ROLE_PATTERNS:  # looser: keyword at start, e.g. "Results and discussion of ..."
        if re.match(pat, h) and len(h.split()) <= 8:
            return role
    return None


def _styled(b: dict, body: float) -> bool:
    return b["bold_ratio"] >= 0.7 or b["font_size"] >= body * 1.08


def _same_style(a: dict, b: dict) -> bool:
    return a["font"] == b["font"] and abs(a["font_size"] - b["font_size"]) <= 0.3


def _is_heading(b: dict, body: float) -> bool:
    t = b["text"].strip()
    if not (2 <= len(t) <= 120) or len(t.split()) > 14 or CAPTION.match(t):
        return False
    if re.search(r"[.;,]$", t) and not NUMBERED.match(t):
        return False
    if re.match(r"^(doi|https?://|e-?mail|\*|corresponding)", t, re.I) or "@" in t:
        return False
    # Figure labels and equation pieces ("( b )", "L a R u 3 S i 2", "E =", "(cid:12)") have no real word.
    if not re.search(r"[A-Za-z]{3}", t) or "=" in t or "(cid:" in t:
        return False
    styled = _styled(b, body)
    numbered = bool(NUMBERED.match(t)) and (styled or role_for(t) is not None)
    known = role_for(t) is not None and (styled or t.isupper() or len(t.split()) <= 4)
    return styled and len(t.split()) <= 14 or numbered or known


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def _furniture_key(text: str) -> str:
    """Running-header identity: page numbers and dates change from page to page, so drop digits."""
    return re.sub(r"[\d\s]+", " ", clean_text(text)).strip().casefold()


def find_furniture(pages: list[dict]) -> list[dict]:
    """Running headers/footers ("Chemistry of Materials", "pubs.acs.org/cm", page numbers):
    margin lines that repeat on several pages, plus bare numbers in the margin."""
    def in_margin(b, page):
        return b["top"] < MARGIN * page["height"] or b["bottom"] > (1 - MARGIN) * page["height"]

    margin = [(page["page"], b) for page in pages for b in page["blocks"]
              if b["kind"] == "line" and in_margin(b, page)]
    pages_with = Counter(key for _, key in {(pno, _furniture_key(b["text"])) for pno, b in margin})
    repeat = max(2, math.ceil(FURNITURE_MIN_SHARE * len(pages)))
    return [{"id": b["id"], "page": pno, "text": b["text"]} for pno, b in margin
            if not (key := _furniture_key(b["text"])) or pages_with[key] >= repeat]


def find_title(pages: list[dict], body: float, skip: set[str]) -> list[dict]:
    """Title lines: the largest text on page 1 that reads like a title. A lone glyph
    (a drop cap, a journal logo such as "■") is skipped in favour of the next size down."""
    first = [b for b in pages[0]["blocks"] if b["kind"] == "line" and b["id"] not in skip] if pages else []
    for size in sorted({b["font_size"] for b in first}, reverse=True):
        if size <= body:
            break
        group = [b for b in first if b["font_size"] == size][:4]
        text = join_lines([b["text"] for b in group])
        if len(text.split()) >= 3 and sum(ch.isalpha() for ch in text) >= 15:
            return group
    return []


def find_headings(pages: list[dict], body: float, skip: set[str]) -> set[str]:
    """IDs of lines that start a section.

    A run of three or more consecutive styled lines in one font is styled body text
    (Nature sets the abstract in bold), not a stack of headings; numbered or named
    headings inside such a run still count.
    """
    lines = [b for p in pages for b in p["blocks"] if b["kind"] == "line" and b["id"] not in skip]
    ids = {b["id"] for b in lines if _is_heading(b, body)}
    # Stray glyph fragments (overlays, stamps: "m", "-") must not break a run of body lines.
    lines = [b for b in lines if len(b["text"].strip()) >= 3]
    run: list[dict] = []
    for b in lines + [None]:
        if b is not None and _styled(b, body) and run and _same_style(run[-1], b):
            run.append(b)
            continue
        if len(run) >= 3:
            ids -= {x["id"] for x in run if not (NUMBERED.match(x["text"].strip()) or role_for(x["text"]))}
        run = [b] if b is not None and _styled(b, body) else []
    return ids


def _continues_heading(prev: dict, b: dict) -> bool:
    """A heading wrapped onto a second line: same style and flow, directly below, no number or role of its own."""
    t = b["text"].strip()
    return (_same_style(prev, b) and prev["_page"] == b["_page"] and prev["flow"] == b["flow"]
            and 0 <= b["top"] - prev["bottom"] < 0.6 * b["font_size"]
            and not NUMBERED.match(t) and role_for(t) is None)


def _heading_level(t: str) -> int:
    m = NUMBERED.match(t)
    if not (m and m.group("num")[0].isdigit()):
        return 1
    return m.group("num").rstrip(".").count(".") + 1


def segment(pages: list[dict], body: float, skip: set[str], headings: set[str]) -> tuple[list[dict], list[dict]]:
    """Walk the blocks in reading order, opening a section at each heading and
    splitting running text into paragraphs, captions and tables."""
    sections = [{"id": "SEC00", "heading": "(front matter)", "role": "front_matter", "level": 0,
                 "page_start": 1, "page_end": 1, "line_ids": []}]
    paragraphs: list[dict] = []
    current = sections[0]
    buf: list[dict] = []
    buf_kind = "text"

    def flush():
        nonlocal buf, buf_kind
        if buf:
            text = join_lines([b["text"] for b in buf])
            if text:
                paragraphs.append({
                    "id": f"P{len(paragraphs) + 1:04d}",
                    "section_id": current["id"],
                    "role": current["role"],
                    "kind": buf_kind,
                    "page": buf[0]["_page"],
                    "page_end": buf[-1]["_page"],
                    "text": text,
                    "line_ids": [b["id"] for b in buf],
                })
        buf, buf_kind = [], "text"

    def open_section(heading, role, level, page, line_ids):
        nonlocal current
        current = {"id": f"SEC{len(sections):02d}", "heading": heading, "role": role, "level": level,
                   "page_start": page, "page_end": page, "line_ids": line_ids}
        sections.append(current)

    prev = last_heading = None
    for page in pages:
        for b in page["blocks"]:
            if b["id"] in skip:
                continue  # title and running headers; prev is kept so paragraphs continue across pages
            b = dict(b, _page=page["page"])
            if b["id"] in headings and last_heading and _continues_heading(last_heading, b):
                current["heading"] = clean_text(current["heading"] + " " + b["text"].strip())
                current["line_ids"].append(b["id"])
                last_heading = b
                continue
            last_heading = b if b["id"] in headings else None
            if b["kind"] == "table":
                flush()
                rows = b["rows"]
                md = "\n".join("| " + " | ".join(clean_text(c).replace("\n", " ") for c in r) + " |" for r in rows)
                paragraphs.append({
                    "id": f"P{len(paragraphs) + 1:04d}", "section_id": current["id"], "role": current["role"],
                    "kind": "table", "page": page["page"], "page_end": page["page"], "text": md, "rows": rows,
                    "line_ids": [b["id"]],
                })
                prev = None
                continue
            t = b["text"].strip()
            if t.lower().startswith("abstract") and current["role"] == "front_matter" and len(t.split()) > 3:
                # Inline "Abstract: ..." without a separate heading; the line itself stays body text.
                flush()
                open_section("Abstract", "abstract", 1, page["page"], [])
            elif b["id"] in headings:
                flush()
                role = role_for(t)
                level = _heading_level(t)
                if role is None:
                    # Unknown heading: a subsection inherits its parent's role; a new top-level
                    # heading before the abstract stays front matter; otherwise "other".
                    if current["role"] == "front_matter":
                        role = "front_matter"
                    elif level > 1 or current["role"] in ("methods", "results", "supplementary"):
                        role = current["role"]
                    else:
                        role = "other"
                open_section(clean_text(t), role, level, page["page"], [b["id"]])
                prev = None
                continue
            current["page_end"] = page["page"]
            cm = CAPTION.match(t)
            # "Fig. 3 shows ..." in body text is not a caption: require a separator, smaller font or bold label.
            starts_caption = bool(cm) and (
                b["font_size"] < body * 0.97 or b["bold_ratio"] > 0.05
                or bool(re.match(r"^(fig(?:ure)?s?|table|scheme)\.?\s*S?\d+[a-z]?\s*[.:|]", t, re.I))
            )
            new_par = False
            if starts_caption:
                new_par = True
            elif prev is not None:
                same_flow = prev["flow"] == b["flow"] and prev["_page"] == b["_page"]
                gap = b["top"] - prev["bottom"]
                ended = bool(re.search(r"[.:?!]$", prev["text"].strip()))
                indent = b["x0"] - prev["x0"] > 0.8 * b["font_size"]
                if same_flow and gap > 0.9 * b["font_size"]:
                    new_par = True
                elif same_flow and ended and indent:
                    new_par = True
                elif not same_flow and re.search(r"[.?!]$", prev["text"].strip()):
                    new_par = True  # column/page break at a sentence end
                elif buf_kind == "caption" and abs(b["font_size"] - buf[0]["font_size"]) > 0.6:
                    new_par = True  # caption ended, body text resumes
            if new_par:
                flush()
            if starts_caption:
                buf_kind = "caption"
            buf.append(b)
            prev = b
    flush()
    return sections, paragraphs


def infer_abstract(sections: list[dict], paragraphs: list[dict]) -> dict | None:
    """Nature-style papers have no "Abstract" heading. If none was found, the first long,
    sentence-like front-matter paragraph on pages 1-2 becomes the abstract (flagged inferred)."""
    if any(s["role"] == "abstract" for s in sections):
        return None
    p = next((p for p in paragraphs if p["role"] == "front_matter" and p["kind"] == "text" and p["page"] <= 2
              and len(p["text"].split()) >= MIN_ABSTRACT_WORDS and p["text"].count(". ") >= 2), None)
    if p is None:
        return None
    sec = {"id": f"SEC{len(sections):02d}", "heading": "(abstract, inferred)", "role": "abstract", "level": 1,
           "page_start": p["page"], "page_end": p["page_end"], "line_ids": [], "inferred": True}
    sections.insert(next(i for i, s in enumerate(sections) if s["id"] == p["section_id"]) + 1, sec)
    p["section_id"], p["role"] = sec["id"], "abstract"
    # Whatever follows an unheaded abstract is body text, even before the first named heading.
    later = paragraphs[paragraphs.index(p) + 1:]
    later_sections = {q["section_id"] for q in later}
    for s in sections:
        if s["role"] == "front_matter" and s["id"] in later_sections and s["id"] != "SEC00":
            s["role"] = "other"
    for q in later:
        if q["role"] == "front_matter":
            q["role"] = "other"
    return sec


def check_coverage(pages: list[dict], owned: list[list[str]]) -> dict:
    """Every ingested block must have exactly one owner; otherwise text was lost or duplicated."""
    counts = Counter(i for ids in owned for i in ids)
    blocks = [b["id"] for p in pages for b in p["blocks"]]
    missing = [i for i in blocks if counts[i] == 0]
    duplicated = [i for i, n in counts.items() if n > 1]
    unknown = sorted(set(counts) - set(blocks))
    if missing or duplicated or unknown:
        raise ValueError(f"Structure coverage failed: {len(missing)} unplaced, {len(duplicated)} duplicated, "
                         f"{len(unknown)} unknown blocks (first: {(missing + duplicated + unknown)[:5]})")
    return {"blocks": len(blocks), "complete": True}


# ---------------------------------------------------------------------------
def build_structure(ingested: dict) -> dict:
    body = ingested["body_font_size"]
    pages = ingested["pages"]

    furniture = find_furniture(pages)
    skip = {f["id"] for f in furniture}
    title_lines = find_title(pages, body, skip)
    title = join_lines([b["text"] for b in title_lines])
    skip |= {b["id"] for b in title_lines}
    sections, paragraphs = segment(pages, body, skip, find_headings(pages, body, skip))
    abstract = infer_abstract(sections, paragraphs)

    # Caption metadata (figure/table number) for later linking.
    for p in paragraphs:
        if p["kind"] == "caption":
            m = CAPTION.match(p["text"])
            p["caption_of"] = f"{'table' if m.group('kind').lower().startswith('tab') else 'figure'} {m.group('num')}"

    coverage = check_coverage(pages, [[f["id"] for f in furniture], [b["id"] for b in title_lines]]
                              + [s["line_ids"] for s in sections] + [p["line_ids"] for p in paragraphs])

    roles_found = sorted({s["role"] for s in sections})
    warnings = []
    if not title:
        warnings.append("No title found on page 1.")
    if abstract:
        warnings.append(f"No abstract heading; inferred the abstract from paragraph on page {abstract['page_start']}.")
    if "methods" not in roles_found:
        warnings.append("No methods/experimental section detected; process extraction will scan all body text.")
    return {
        "title": title,
        "title_line_ids": [b["id"] for b in title_lines],
        "sections": sections,
        "paragraphs": paragraphs,
        "furniture": furniture,
        "roles_found": roles_found,
        "coverage": coverage,
        "warnings": warnings,
    }
