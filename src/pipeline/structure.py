"""Stage 2 — Publication structure: title, sections with roles, paragraphs, captions, tables.

Follows the publication-structure notebook's ideas (typography-relative headings,
unidentified content kept, every block owned exactly once) in a simpler form.
Section *roles* drive routing later: process details come from `methods`,
values from `results` and tables, applications from `abstract`/`introduction`.
"""
from __future__ import annotations

import re

from .clean import clean_text, join_lines

ROLE_PATTERNS = [
    ("abstract", r"abstract|summary of the work"),
    ("introduction", r"introduction|background|motivation"),
    ("methods", r"experimental|experiment(al)? (section|details|procedures?|methods?)|methods?|materials and methods|"
                r"methodology|sample preparation|fabrication|synthesis|device fabrication|characteri[sz]ation|"
                r"computational (details|methods?)|simulation (details|methods?)|growth|deposition"),
    ("results", r"results?( and discussion)?|discussion|analysis|performance"),
    ("conclusion", r"conclusions?|concluding remarks|summary|outlook"),
    ("references", r"references|bibliography|literature cited|works cited"),
    ("acknowledgements", r"acknowledg(e)?ments?|funding"),
    ("author_info", r"author contributions?|conflicts? of interest|competing interests?|declaration of .*|"
                    r"data availability|credit authorship.*|orcid"),
    ("supplementary", r"supporting information|supplementary (material|information)|appendix.*"),
]
NUMBERED = re.compile(r"^(?P<num>(\d+(\.\d+)*|[IVX]+)\.?)\s+(?P<title>[A-Z].{1,120})$")
CAPTION = re.compile(r"^(?P<kind>fig(?:ure)?s?|table|scheme)\.?\s*(?P<num>S?\d+[a-z]?)\s*[.:|\-–]?\s*", re.I)


def role_for(heading: str) -> str | None:
    h = re.sub(r"^(\d+(\.\d+)*|[IVX]+)\.?\s+", "", heading.strip()).strip(" .:").lower()
    for role, pat in ROLE_PATTERNS:
        if re.fullmatch(pat, h):
            return role
    for role, pat in ROLE_PATTERNS:  # looser: keyword at start, e.g. "Results and discussion of ..."
        if re.match(pat, h) and len(h.split()) <= 8:
            return role
    return None


def _is_heading(b: dict, body: float) -> bool:
    t = b["text"].strip()
    if not (2 <= len(t) <= 120) or len(t.split()) > 14 or CAPTION.match(t):
        return False
    if re.search(r"[.;,]$", t) and not NUMBERED.match(t):
        return False
    if re.match(r"^(doi|https?://|e-?mail|\*|corresponding)", t, re.I) or "@" in t:
        return False
    styled = b["bold_ratio"] >= 0.7 or b["font_size"] >= body * 1.08
    numbered = bool(NUMBERED.match(t)) and (styled or role_for(t) is not None)
    known = role_for(t) is not None and (styled or t.isupper() or len(t.split()) <= 4)
    return styled and len(t.split()) <= 14 or numbered or known


def build_structure(ingested: dict) -> dict:
    body = ingested["body_font_size"]
    pages = ingested["pages"]
    # Title: largest text on page 1 (front matter).
    first = [b for b in pages[0]["blocks"] if b["kind"] == "line"] if pages else []
    title = ""
    if first:
        top_size = max(b["font_size"] for b in first)
        title = join_lines([b["text"] for b in first if b["font_size"] == top_size][:4])

    sections = [{"id": "SEC00", "heading": "(front matter)", "role": "front_matter", "level": 0,
                 "page_start": 1, "page_end": 1}]
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
                })
        buf, buf_kind = [], "text"

    prev = None
    for page in pages:
        for b in page["blocks"]:
            b = dict(b, _page=page["page"])
            if b["kind"] == "table":
                flush()
                rows = b["rows"]
                md = "\n".join("| " + " | ".join(clean_text(c).replace("\n", " ") for c in r) + " |" for r in rows)
                paragraphs.append({
                    "id": f"P{len(paragraphs) + 1:04d}", "section_id": current["id"], "role": current["role"],
                    "kind": "table", "page": page["page"], "page_end": page["page"], "text": md, "rows": rows,
                })
                prev = None
                continue
            t = b["text"].strip()
            if t.lower().startswith("abstract") and current["role"] == "front_matter" and len(t.split()) > 3:
                # Inline "Abstract: ..." without a separate heading.
                flush()
                current = {"id": f"SEC{len(sections):02d}", "heading": "Abstract", "role": "abstract", "level": 1,
                           "page_start": page["page"], "page_end": page["page"]}
                sections.append(current)
            elif _is_heading(b, body) and not (page["page"] == 1 and title and clean_text(t) in title):
                flush()
                role = role_for(t)
                m = NUMBERED.match(t)
                level = (m.group("num").count(".") + 1 if m and m.group("num")[0].isdigit() else 1)
                if m and m.group("num").rstrip(".").count(".") >= 1:
                    level = m.group("num").rstrip(".").count(".") + 1
                if role is None:
                    # Unknown heading: a subsection inherits its parent's role; a new top-level
                    # heading before the abstract stays front matter; otherwise "other".
                    if current["role"] == "front_matter":
                        role = "front_matter"
                    elif level > 1 or current["role"] in ("methods", "results", "supplementary"):
                        role = current["role"]
                    else:
                        role = "other"
                current = {"id": f"SEC{len(sections):02d}", "heading": clean_text(t), "role": role, "level": level,
                           "page_start": page["page"], "page_end": page["page"]}
                sections.append(current)
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

    # Caption metadata (figure/table number) for later linking.
    for p in paragraphs:
        if p["kind"] == "caption":
            m = CAPTION.match(p["text"])
            p["caption_of"] = f"{'table' if m.group('kind').lower().startswith('tab') else 'figure'} {m.group('num')}"

    roles_found = sorted({s["role"] for s in sections})
    warnings = []
    if "methods" not in roles_found:
        warnings.append("No methods/experimental section detected; process extraction will scan all body text.")
    return {
        "title": title,
        "sections": sections,
        "paragraphs": paragraphs,
        "roles_found": roles_found,
        "warnings": warnings,
    }
