"""Stages 1-2 with Docling: PDF -> DoclingDocument -> the pipeline's structure format.

Docling (IBM) runs a layout model (docling-layout-heron) on every page to find
titles, section headers, paragraphs, lists, captions, footnotes, formulas, tables
and pictures, puts them in reading order (handling one/two-column and mixed pages),
and runs TableFormer to recover table rows/columns/headers. Captions are attached to
their table/figure. Page headers/footers go to a separate "furniture" layer, and text
inside figures (axis labels, tick numbers, legends) stays inside the picture.

What this module adds on top:
  * one reusable converter (models load once per run), options recorded for reproducibility
  * per-paper outputs: docling.json (full DoclingDocument), figure/table crops as PNG
  * re-extraction of each text block / table cell from the PDF characters inside Docling's
    bounding box, keeping superscripts as ^{...} (Docling's text flattens "10^15 cm^-3" to
    "1015 cm-3"). The re-extracted text is used only when it agrees with Docling's text.
  * mapping to sections (with roles: methods/results/...), paragraphs, tables and figures
  * quality flags (tables Docling could not structure -> optional Claude vision fallback)
  * debug page overlays showing what was detected
"""
from __future__ import annotations

import difflib
import importlib.metadata
import re
from collections import Counter
from pathlib import Path

from .clean import clean_text, dehyphenate, join_lines
from .structure import CAPTION, find_title, infer_abstract, role_for

DOCLING_DEFAULTS = {
    "do_ocr": False,                 # born-digital PDFs don't need OCR; True for scanned pages (slower)
    "table_mode": "accurate",        # TableFormer "accurate" or "fast"
    "do_cell_matching": True,        # map table cells back to PDF text cells
    "heading_hierarchy": True,       # infer heading levels from bookmarks / numbering / font style
    "picture_classification": True,  # label figures (line chart, photo, diagram, ...); small model
    "formula_enrichment": False,     # formulas -> LaTeX with a VLM (slow on CPU)
    "images_scale": 2.0,             # crop resolution (2.0 = 144 dpi)
    "device": "auto",                # auto | cpu | cuda | mps
    "num_threads": 4,
    "document_timeout": None,        # seconds per PDF, None = no limit
    "artifacts_path": None,          # folder of pre-downloaded models (docling-tools models download -o ...)
    "reextract_text": True,          # re-read text from PDF chars to keep superscripts (see above)
}


def docling_versions() -> dict:
    out = {}
    for p in ("docling", "docling-core", "docling-ibm-models", "docling-parse"):
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


class DoclingParser:
    """Wraps a DocumentConverter; build once and reuse for all papers."""

    def __init__(self, options: dict | None = None):
        self.options = {**DOCLING_DEFAULTS, **(options or {})}
        self._converter = None

    def fingerprint(self) -> dict:
        return {"options": {k: v for k, v in self.options.items() if k != "artifacts_path"},
                "versions": docling_versions()}

    def converter(self):
        if self._converter is None:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (HeadingHierarchyOptions, PdfPipelineOptions,
                                                            TableFormerMode, TableStructureOptions)
            from docling.document_converter import DocumentConverter, PdfFormatOption

            o = self.options
            po = PdfPipelineOptions(
                do_ocr=o["do_ocr"],
                do_table_structure=True,
                table_structure_options=TableStructureOptions(
                    mode=TableFormerMode(o["table_mode"]), do_cell_matching=o["do_cell_matching"]),
                heading_hierarchy_options=HeadingHierarchyOptions(enabled=o["heading_hierarchy"]),
                generate_parsed_pages=o["heading_hierarchy"],   # needed by the font-style heading signal
                do_picture_classification=o["picture_classification"],
                do_formula_enrichment=o["formula_enrichment"],
                generate_picture_images=True,
                generate_page_images=True,      # TableItem.get_image() crops tables from page images
                images_scale=o["images_scale"],
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice(o["device"]),
                                                       num_threads=o["num_threads"]),
                document_timeout=o["document_timeout"],
            )
            if o.get("artifacts_path"):
                po.artifacts_path = Path(o["artifacts_path"])
            self._converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)},
            )
        return self._converter

    def convert(self, pdf_path: Path, out_dir: Path) -> dict:
        """Convert one PDF; write docling.json and crops into out_dir; return the ingest record."""
        from docling.datamodel.base_models import ConversionStatus
        from docling_core.types.doc import ImageRefMode

        from .io_utils import sha256_file

        pdf_path, out_dir = Path(pdf_path), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        res = self.converter().convert(pdf_path, raises_on_error=False)
        doc = res.document
        status = res.status.value if hasattr(res.status, "value") else str(res.status)
        errors = [str(getattr(e, "error_message", e)) for e in (res.errors or [])]
        if res.status == ConversionStatus.FAILURE or doc is None:
            raise RuntimeError(f"Docling failed on {pdf_path.name}: {errors or status}")

        crops = {}
        crop_dir = out_dir / "crops"
        crop_dir.mkdir(exist_ok=True)
        for kind, items in (("figure", doc.pictures), ("table", doc.tables)):
            for i, item in enumerate(items, 1):
                img = item.get_image(doc)
                if img is not None:
                    name = f"{kind}_{i:03d}_p{item.prov[0].page_no if item.prov else 0}.png"
                    img.save(crop_dir / name)
                    crops[item.self_ref] = f"crops/{name}"
        doc.save_as_json(out_dir / "docling.json", image_mode=ImageRefMode.PLACEHOLDER)

        conf = res.confidence.model_dump(mode="json") if getattr(res, "confidence", None) else {}
        pages_conf = conf.pop("pages", {}) if conf else {}
        return {
            "source": {"pdf": pdf_path.name, "sha256": sha256_file(pdf_path), "page_count": doc.num_pages()},
            "parser": "docling",
            "docling": self.fingerprint(),
            "status": status,
            "errors": errors,
            "confidence": conf,                       # mean/low grade: poor | fair | good | excellent
            "page_confidence": pages_conf,
            "counts": {"texts": len(doc.texts), "tables": len(doc.tables), "pictures": len(doc.pictures)},
            "docling_json": "docling.json",
            "crops": crops,
            "warnings": ([f"docling status {status}: {errors}"] if status != "success" else []),
        }


# ---------------------------------------------------------------------------
# Text re-extraction from PDF characters (keeps superscripts)
# ---------------------------------------------------------------------------
class CharReader:
    """Reads text inside a bounding box from the PDF's characters with pdfplumber."""

    def __init__(self, pdf_path: Path):
        import pdfplumber
        self._pdf = pdfplumber.open(pdf_path)
        self._pages = {}

    def close(self):
        self._pdf.close()

    def _chars(self, page_no: int):
        if page_no not in self._pages:
            page = self._pdf.pages[page_no - 1].dedupe_chars(tolerance=1)
            self._pages[page_no] = ([c for c in page.chars if c.get("text", "")], page.height)
        return self._pages[page_no]

    def text_in(self, page_no: int, bbox_tl: tuple, pad: float = 1.5) -> str:
        from .ingest import _group_lines, _line_text
        chars, _ = self._chars(page_no)
        l, t, r, b = bbox_tl
        inside = [c for c in chars if l - pad <= (c["x0"] + c["x1"]) / 2 <= r + pad
                  and t - pad <= (c["top"] + c["bottom"]) / 2 <= b + pad]
        if not inside:
            return ""
        lines = sorted(_group_lines(inside), key=lambda ln: min(c["top"] for c in ln))
        return join_lines([_line_text(ln)[0] for ln in lines])

    def title(self) -> str:
        """Fallback when Docling labels no title: the largest text on page 1 that reads like a title."""
        from .ingest import _group_lines, _line_text
        chars, _ = self._chars(1)
        sizes = Counter(round(c["size"], 1) for c in chars if c["text"].strip())
        body = sizes.most_common(1)[0][0] if sizes else 10.0
        blocks = []
        for i, ln in enumerate(sorted(_group_lines(chars), key=lambda ln: min(c["top"] for c in ln))):
            text, style = _line_text(ln)
            if text:
                blocks.append({"kind": "line", "id": f"t{i}", "text": text, **style})
        return clean_text(join_lines([b["text"] for b in find_title([{"blocks": blocks}], body, set())]))


def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.replace("^{", "").replace("}", "").casefold())


def _agrees(a: str, b: str, threshold: float = 0.9) -> bool:
    x, y = _alnum(a), _alnum(b)
    if not x or not y:
        return False
    return difflib.SequenceMatcher(None, x, y, autojunk=False).ratio() >= threshold


# ---------------------------------------------------------------------------
# DoclingDocument -> structure
# ---------------------------------------------------------------------------
def _boxes(item, doc) -> list[dict]:
    out = []
    for p in item.prov or []:
        h = doc.pages[p.page_no].size.height if p.page_no in doc.pages else None
        bb = p.bbox.to_top_left_origin(h) if h is not None else p.bbox
        out.append({"page": p.page_no, "bbox": [round(bb.l, 1), round(bb.t, 1), round(bb.r, 1), round(bb.b, 1)]})
    return out


def _page_texts(pages: list[int], parts: list[str]) -> list[dict] | None:
    """Per-page pieces of a block that crosses a page break (None for single-page blocks)."""
    if len(set(pages)) < 2:
        return None
    merged: dict[int, list[str]] = {}
    for page, part in zip(pages, parts):
        merged.setdefault(page, []).append(part)
    return [{"page": page, "text": clean_text(join_lines(ps))} for page, ps in merged.items()]


def _best_text(item, doc, reader: CharReader | None) -> tuple[str, str, list[dict] | None]:
    raw = item.text or ""
    base = clean_text(dehyphenate(raw))
    pages = [p.page_no for p in item.prov or []]
    # Docling records which characters of the block sit on which page (charspan).
    base_pages = _page_texts(pages, [dehyphenate(raw[p.charspan[0]:p.charspan[1]]) for p in item.prov or []])
    if reader is None or not item.prov:
        return base, "docling", base_pages
    parts = []
    for bx in _boxes(item, doc):
        parts.append(reader.text_in(bx["page"], bx["bbox"]))
    re_text = clean_text(join_lines(parts))
    if "^{" in re_text and _agrees(re_text, base):
        return re_text, "pdf_chars", _page_texts(pages, parts)
    return base, "docling", base_pages


def _table_grid(table, doc, reader: CharReader | None) -> tuple[list[list[str]], int]:
    """Cell texts row by row (merged cells repeated); re-extract cells to keep superscripts."""
    grid = table.data.grid
    rows, upgraded = [], 0
    page_no = table.prov[0].page_no if table.prov else None
    h = doc.pages[page_no].size.height if page_no in doc.pages else None
    for row in grid:
        out = []
        for cell in row:
            txt = clean_text(cell.text or "")
            if reader is not None and cell.bbox is not None and page_no and h is not None and txt:
                bb = cell.bbox.to_top_left_origin(h)
                alt = clean_text(reader.text_in(page_no, (bb.l, bb.t, bb.r, bb.b), pad=0.5))
                if "^{" in alt and _agrees(alt, txt, 0.85):
                    txt, upgraded = alt, upgraded + 1
            out.append(txt)
        rows.append(out)
    return rows, upgraded


def _md(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(c.replace("|", "/").replace("\n", " ") for c in rows[0]) + " |",
             "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(c.replace("|", "/").replace("\n", " ") for c in r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _table_quality(rows: list[list[str]]) -> dict:
    n_rows = len(rows)
    n_cols = max((len(r) for r in rows), default=0)
    cells = [c for r in rows for c in r]
    empty = sum(1 for c in cells if not c.strip()) / len(cells) if cells else 1.0
    needs = n_rows < 2 or n_cols < 2 or empty > 0.5
    return {"rows": n_rows, "cols": n_cols, "empty_fraction": round(empty, 2), "needs_vision": needs}


def _classification(pic) -> dict | None:
    meta = getattr(pic, "meta", None)
    cls = getattr(meta, "classification", None) if meta else None
    preds = getattr(cls, "predictions", None) if cls else None
    if preds:
        top = max(preds, key=lambda p: p.confidence or 0)
        return {"class": top.class_name, "confidence": round(top.confidence or 0, 3)}
    return None


def structure_from_docling(doc, ingest: dict, pdf_path: Path | None = None) -> dict:
    """Map a DoclingDocument to {title, sections, paragraphs, tables, figures, ...}."""
    from docling_core.types.doc import DocItemLabel
    from docling_core.types.doc.document import PictureItem, SectionHeaderItem, TableItem, TextItem, TitleItem

    reader = CharReader(pdf_path) if (pdf_path and ingest.get("docling", {}).get("options", {}).get("reextract_text", True)) else None
    crops = ingest.get("crops", {})
    sections = [{"id": "SEC00", "heading": "(front matter)", "role": "front_matter", "level": 0,
                 "page_start": 1, "page_end": 1}]
    current = sections[0]
    paragraphs, tables, figures = [], [], []
    title = ""
    caption_refs = set()
    for it in list(doc.tables) + list(doc.pictures):
        for c in it.captions or []:
            caption_refs.add(c.cref)
        for f in getattr(it, "footnotes", None) or []:
            caption_refs.add(f.cref)
    stats = {"text_from_pdf_chars": 0, "table_cells_from_pdf_chars": 0}

    def new_section(heading, level, page):
        nonlocal current
        role = role_for(heading)
        if role is None:
            if current["role"] == "front_matter":
                role = "front_matter"
            elif level > 1 or current["role"] in ("methods", "results", "supplementary"):
                role = current["role"]
            else:
                role = "other"
        current = {"id": f"SEC{len(sections):02d}", "heading": heading, "role": role, "level": level,
                   "page_start": page, "page_end": page}
        sections.append(current)

    def add_par(kind, text, item, **extra):
        boxes = _boxes(item, doc)
        page = boxes[0]["page"] if boxes else current["page_end"]
        current["page_end"] = max(current["page_end"], boxes[-1]["page"] if boxes else page)
        role = current["role"]
        if kind == "reference":
            role = "references"
        p = {"id": f"P{len(paragraphs) + 1:04d}", "section_id": current["id"], "role": role, "kind": kind,
             "page": page, "page_end": boxes[-1]["page"] if boxes else page, "text": text,
             "boxes": boxes, "docling_ref": item.self_ref, **extra}
        paragraphs.append(p)
        return p

    try:
        for item, _level in doc.iterate_items():
            ref = getattr(item, "self_ref", None)
            if ref in caption_refs:
                continue
            page = item.prov[0].page_no if getattr(item, "prov", None) else current["page_end"]
            if isinstance(item, TitleItem):
                title = title or clean_text(item.text)
                continue
            if isinstance(item, SectionHeaderItem):
                heading = clean_text(item.text)
                if not title and page == 1 and current["role"] == "front_matter" and len(sections) == 1 \
                        and role_for(heading) is None and len(heading.split()) > 4:
                    title = heading   # some papers' titles are detected as headers
                    continue
                new_section(heading, max(1, int(item.level or 1)), page)
                continue
            if isinstance(item, TableItem):
                rows, up = _table_grid(item, doc, reader)
                stats["table_cells_from_pdf_chars"] += up
                cap = clean_text(item.caption_text(doc) or "")
                foot = " ".join(clean_text(doc.texts[int(f.cref.split("/")[-1])].text)
                                for f in (item.footnotes or []) if f.cref.startswith("#/texts/"))
                q = _table_quality(rows)
                m = CAPTION.match(cap)
                label = f"table {m.group('num')}" if m else f"table (unnumbered #{len(tables) + 1})"
                text = (cap + "\n" if cap else "") + _md(rows) + (f"\nFootnotes: {foot}" if foot else "")
                p = add_par("table", text, item, rows=rows, caption=cap, footnotes=foot, caption_of=label,
                            quality=q, crop=crops.get(ref))
                tables.append({"paragraph_id": p["id"], "label": label, "caption": cap, "page": p["page"],
                               "crop": crops.get(ref), **q})
                continue
            if isinstance(item, PictureItem):
                cap = clean_text(item.caption_text(doc) or "")
                cls = _classification(item)
                m = CAPTION.match(cap)
                label = f"figure {m.group('num')}" if m else f"figure (unnumbered #{len(figures) + 1})"
                head = f"[{label.capitalize()}" + (f" | {cls['class']}" if cls else "") + "]"
                p = add_par("figure", f"{head} {cap}".strip(), item, caption=cap, caption_of=label,
                            classification=cls, crop=crops.get(ref))
                figures.append({"paragraph_id": p["id"], "label": label, "caption": cap, "page": p["page"],
                                "classification": cls, "crop": crops.get(ref), "has_caption": bool(cap)})
                continue
            if isinstance(item, TextItem):
                lab = item.label
                kind = {DocItemLabel.CAPTION: "caption", DocItemLabel.FOOTNOTE: "footnote",
                        DocItemLabel.FORMULA: "formula", DocItemLabel.LIST_ITEM: "list_item",
                        DocItemLabel.REFERENCE: "reference", DocItemLabel.CODE: "code"}.get(lab, "text")
                text, src, page_texts = _best_text(item, doc, reader)
                if not text:
                    continue
                if src == "pdf_chars":
                    stats["text_from_pdf_chars"] += 1
                if kind == "text" and current["role"] == "front_matter" and re.match(r"^abstract\b", text, re.I):
                    new_section("Abstract", 1, page)
                if kind == "formula" and getattr(item, "orig", None) and not text.strip():
                    text = clean_text(item.orig)
                extra = {"text_source": src}
                if page_texts:
                    extra["page_texts"] = page_texts
                if kind == "caption":
                    m = CAPTION.match(text)
                    if m:
                        extra["caption_of"] = f"{'table' if m.group('kind').lower().startswith('tab') else 'figure'} {m.group('num')}"
                add_par(kind, text, item, **extra)
        title_source = "docling" if title else None
        if not title and pdf_path:
            fallback = reader or CharReader(pdf_path)
            try:
                title = fallback.title()
            finally:
                if fallback is not reader:
                    fallback.close()
            title_source = "pdf_font_size" if title else None
    finally:
        if reader:
            reader.close()

    # Nature-style papers have no "Abstract" header for Docling to find.
    abstract = infer_abstract(sections, paragraphs)

    roles = sorted({s["role"] for s in sections})
    warnings = list(ingest.get("warnings", []))
    if abstract:
        warnings.append(f"No abstract heading; inferred the abstract from paragraph on page {abstract['page_start']}.")
    if "methods" not in roles:
        warnings.append("No methods/experimental section detected; process extraction will scan all body text.")
    if not title:
        warnings.append("No title detected.")
    elif title_source == "pdf_font_size":
        warnings.append("Docling found no title; used the largest text on page 1.")
    uncaptioned = sum(1 for f in figures if not f["has_caption"])
    if uncaptioned:
        warnings.append(f"{uncaptioned} figure(s) without a caption (logos, inset images or caption not linked).")
    bad_tables = sum(1 for t in tables if t["needs_vision"])
    if bad_tables:
        warnings.append(f"{bad_tables} table(s) Docling could not structure well (vision fallback candidates).")
    return {
        "title": title,
        "title_source": title_source,
        "sections": sections,
        "paragraphs": paragraphs,
        "tables": tables,
        "figures": figures,
        "roles_found": roles,
        "stats": stats,
        "warnings": warnings,
    }


def load_docling_document(path: Path):
    from docling_core.types.doc import DoclingDocument
    return DoclingDocument.load_from_json(Path(path))


# ---------------------------------------------------------------------------
# Debug overlays: draw what was detected on each page
# ---------------------------------------------------------------------------
KIND_COLORS = {"text": "#2563eb", "list_item": "#0891b2", "caption": "#d97706", "table": "#16a34a",
               "figure": "#dc2626", "formula": "#7c3aed", "footnote": "#64748b", "reference": "#94a3b8",
               "code": "#0f766e", "heading": "#db2777"}


def render_debug_pages(pdf_path: Path, structure: dict, doc, out_dir: Path, pages: list[int] | None = None,
                       resolution: int = 100) -> list[Path]:
    """Write one PNG per page with boxes for paragraphs, tables, figures and headings."""
    import pdfplumber

    from docling_core.types.doc.document import SectionHeaderItem, TitleItem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_page: dict[int, list] = {}
    for p in structure["paragraphs"]:
        for bx in p.get("boxes", []):
            by_page.setdefault(bx["page"], []).append((p["kind"], p["id"], bx["bbox"]))
    for item, _ in doc.iterate_items():
        if isinstance(item, (SectionHeaderItem, TitleItem)):
            tag = f"H{item.level}" if isinstance(item, SectionHeaderItem) else "TITLE"
            for bx in _boxes(item, doc):
                by_page.setdefault(bx["page"], []).append(("heading", tag, bx["bbox"]))
    written = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno in pages or range(1, len(pdf.pages) + 1):
            im = pdf.pages[pno - 1].to_image(resolution=resolution)
            for kind, label, bb in by_page.get(pno, []):
                color = KIND_COLORS.get(kind, "#000000")
                im.draw_rect(tuple(bb), stroke=color, fill=None, stroke_width=2)
            path = out_dir / f"page_{pno:03d}.png"
            img = im.annotated
            try:
                from PIL import ImageDraw
                d = ImageDraw.Draw(img)
                s = resolution / 72
                for kind, label, bb in by_page.get(pno, []):
                    tag = f"{label} {kind}" if kind != "heading" else label
                    x, y = bb[0] * s, max(0, bb[1] * s - 11)
                    w = d.textlength(tag)
                    d.rectangle([x, y, x + w + 4, y + 11], fill=KIND_COLORS.get(kind, "#000"))
                    d.text((x + 2, y), tag, fill="#ffffff")
            except Exception:
                pass
            img.save(path)
            written.append(path)
    return written
