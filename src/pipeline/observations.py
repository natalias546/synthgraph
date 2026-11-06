"""structure.json -> observations.json: the flat, reading-order list the LLM notebooks read.

One observation per heading, paragraph, caption, table or figure, in reading order;
a paragraph that crosses a page break becomes one observation per page. The notebooks
need id, page, reading_order and text; kind, role, section, page_end and continues
are extra context. Works for both parsers (docling and pdfplumber).
"""
from __future__ import annotations

from pathlib import Path

from .io_utils import write_json

REQUIRED = {"id", "page", "reading_order", "text"}


def observations_from_structure(structure: dict) -> list[dict]:
    sections = structure["sections"]
    order = {s["id"]: i for i, s in enumerate(sections)}
    by_id = {s["id"]: s for s in sections}
    out: list[dict] = []
    if structure.get("title"):
        out.append({"id": "TITLE", "page": 1, "page_end": 1, "kind": "title", "role": "front_matter",
                    "section_id": "SEC00", "section": "(front matter)", "text": structure["title"]})
    emitted = 0  # sections[0] is the synthetic front-matter section; it has no heading text

    title = structure.get("title", "").casefold()

    def heading(s):
        # Inferred sections (e.g. an unheaded abstract) have no heading line in the PDF,
        # and Docling sometimes labels the title as a heading too.
        if s["id"] != "SEC00" and not s.get("inferred") and s["heading"].casefold() != title:
            out.append({"id": s["id"], "page": s["page_start"], "page_end": s["page_start"], "kind": "heading",
                        "role": s["role"], "section_id": s["id"], "section": s["heading"], "text": s["heading"]})

    for p in structure["paragraphs"]:
        # Headings come out just before their first paragraph; empty sections (a heading
        # directly followed by a subheading) are emitted in section order on the way.
        idx = order.get(p["section_id"], emitted)
        while emitted < idx:
            emitted += 1
            heading(sections[emitted])
        sec = by_id.get(p["section_id"], {})
        base = {"kind": p["kind"], "role": p["role"], "section_id": p["section_id"], "section": sec.get("heading", "")}
        if p.get("page_texts"):
            # A paragraph crossing a page break: one observation per page, so every page
            # has its own text and quotes cite the right page; later pieces point back.
            for k, part in enumerate(p["page_texts"]):
                out.append({"id": p["id"] if k == 0 else f"{p['id']}.p{part['page']}", "page": part["page"],
                            "page_end": part["page"], **base, "text": part["text"],
                            **({"continues": p["id"]} if k else {})})
        else:
            out.append({"id": p["id"], "page": p["page"], "page_end": p.get("page_end", p["page"]), **base,
                        "text": p["text"]})
    for s in sections[emitted + 1:]:  # trailing sections with no paragraphs
        heading(s)

    out = [o for o in out if o["text"].strip()]
    for i, o in enumerate(out, 1):
        o["reading_order"] = i
    return out


def write_observations(structure: dict, path: Path) -> list[dict]:
    observations = observations_from_structure(structure)
    for o in observations:
        missing = REQUIRED - o.keys()
        if missing:
            raise ValueError(f"Observation {o.get('id')} is missing {missing}")
    write_json(Path(path), observations)
    return observations
