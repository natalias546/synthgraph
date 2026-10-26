"""Helpers for the publication structure notebook.

Moved from notebooks/scientific_publication_structure_worked_example.ipynb so the
notebook only makes calls.
"""

import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema
import pdfplumber
from IPython.display import HTML, JSON, display

# Project root is one level up from src/.
ROOT = Path(__file__).resolve().parent.parent
PUBLICATIONS = ROOT / "data" / "publications" / "Publications"
SCHEMA_PATH = ROOT / "notebooks" / "scientific_publication.schema.json"
STRUCTURES = ROOT / "outputs" / "publication_structures"


# Build the paths and fingerprint for one paper; every output path comes from its filename.
def configure(pdf_name):
    pdf = PUBLICATIONS / pdf_name
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    output = STRUCTURES / pdf.stem
    output.mkdir(parents=True, exist_ok=True)
    return {
        "pdf": pdf,
        "output": output,
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }


# Render a compact comparison table while escaping source text before inserting it into HTML.
def show_table(rows, columns):
    """Small dependency-free table; escape source text before rendering it."""
    header = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
        + "</tr>"
        for row in rows
    )
    display(
        HTML(
            "<style>table.pub{border-collapse:collapse;font-size:14px}table.pub th,table.pub td{border:1px solid #cbd5e1;padding:7px;text-align:left;vertical-align:top}table.pub th{background:#e2e8f0}</style><table class='pub'><thead><tr>"
            + header
            + "</tr></thead><tbody>"
            + body
            + "</tbody></table>"
        )
    )


# Save and display the same complete stage result, with a direct Jupyter download link.
def publish_json(path, value, label):
    """Save and display the same complete object, with a Jupyter download link."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == value
    # The download link only works for files inside the project folder.
    link = ""
    if path.is_relative_to(ROOT):
        href = "/files/" + path.relative_to(ROOT).as_posix()
        link = f'<p><a href="{html.escape(href, quote=True)}" download="{html.escape(path.name, quote=True)}" target="_blank">Download complete {html.escape(path.name)}</a></p>'
    display(HTML(f"<h3>{html.escape(label)}</h3>{link}"))
    display(JSON(value, expanded=False))
    print("Saved:", path)
    return path


# Construct the less-informed baseline directly from each PDF page.
def flat_text(config):
    pdf_path = config["pdf"]
    with pdfplumber.open(pdf_path) as pdf:
        flat_document = {
            "source": {
                "pdf": str(pdf_path),
                "sha256": config["sha256"],
                "pages": len(pdf.pages),
            },
            "pages": [
                {"page": i + 1, "text": page.extract_text() or ""}
                for i, page in enumerate(pdf.pages)
            ],
        }
    (config["output"] / "flat_text_baseline.json").write_text(
        json.dumps(flat_document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return flat_document


# Fail loudly when a page-text extraction looks unusable.
def check_flat_text(flat_document, min_chars_per_page=200):
    if not isinstance(flat_document, dict):
        raise TypeError(
            f"Expected the flat_document dict, got {type(flat_document).__name__}"
        )
    pages = flat_document["pages"]
    if not pages:
        raise ValueError("PDF has no pages")
    empty = [p["page"] for p in pages if len(p["text"].strip()) < min_chars_per_page]
    if len(empty) == len(pages):
        raise ValueError("No text on any page; this PDF is probably scanned images")
    if empty:
        print(f"Warning: little or no text on pages {empty}")


# ---------------------------------------------------------------------------
# 4.1 Node construction and structural validation
# ---------------------------------------------------------------------------


# Create one schema element with an inclusive physical-page interval and child list.
def node(identifier, role, start, end=None, **fields):
    return dict(
        id=identifier,
        type=role,
        page_start=start,
        page_end=start if end is None else end,
        children=[],
        **fields
    )


# Traverse the hierarchy in preorder so every element can be inspected uniformly.
def walk(n):
    yield n
    for c in n["children"]:
        yield from walk(c)


# Build an ID lookup for resolving parents, evidence links and replacement targets.
def index(tree):
    return {n["id"]: n for n in walk(tree["document"])}


# Propagate child page intervals upward so each container encloses its contents.
def expand_ranges(n):
    for c in n["children"]:
        expand_ranges(c)
    if n["children"]:
        n["page_start"] = min(
            [n["page_start"]] + [c["page_start"] for c in n["children"]]
        )
        n["page_end"] = max([n["page_end"]] + [c["page_end"] for c in n["children"]])


# Check the schema and mechanical invariants; this does not certify interpretation accuracy.
def validate_structure(tree, schema):
    jsonschema.Draft202012Validator(schema).validate(tree)

    # Track node IDs while checking page bounds, containment and geometry exclusion.
    seen = set()

    def visit(n, parent=None):
        assert n["id"] not in seen, "Duplicate node ID: " + n["id"]
        seen.add(n["id"])
        assert (
            1 <= n["page_start"] <= n["page_end"] <= tree["source"]["pages"]
        ), "Invalid page interval"
        assert (
            "source" not in n and "bbox" not in n
        ), "Geometry is forbidden in structural output"
        if parent:
            assert (
                parent["page_start"]
                <= n["page_start"]
                <= n["page_end"]
                <= parent["page_end"]
            ), "Child outside parent"
        for c in n["children"]:
            visit(c, n)

    visit(tree["document"])

    # Check affiliation links after collecting the complete set of nodes.
    ids = index(tree)
    for n in ids.values():
        for aid in n.get("affiliation_ids", []):
            assert (
                aid in ids and ids[aid]["type"] == "affiliation"
            ), "Broken affiliation link"
    return {
        "schema_valid": True,
        "page_ranges_valid": True,
        "unique_ids": True,
        "nodes": len(seen),
    }


# ---------------------------------------------------------------------------
# 4.2 Lines, typography and reading order
# ---------------------------------------------------------------------------


# Group PDF characters into text lines while retaining their private geometric evidence.
def char_line_records(chars):
    if not chars:
        return []
    from pdfplumber.utils import chars_to_textmap

    # pdfplumber lists a ligature glyph ("ffi") once per letter; rebuilding text from
    # those copies would print it three times ("diffiffifficult").
    chars = list({id(c): c for c in chars}.values())
    return chars_to_textmap(
        chars, layout=False, x_tolerance_ratio=0.15, y_tolerance=3
    ).extract_text_lines(strip=True, return_chars=True)


# Return just the text when the caller does not need character-level evidence.
def char_lines(chars):
    return [line["text"] for line in char_line_records(chars)]


# Infer reading order within page bands, allowing both column text and spanning content.
def ordered_page_lines(page, pno, body_font_size=None):
    ordered = []
    summary = {
        "page": pno,
        "method": "local_geometry_fallback",
        "review_required": True,
    }
    # Analyze separate vertical bands. Long horizontal rules often delimit a header,
    # spanning table, or figure; one page-wide gutter can otherwise be dominated by table columns.

    # Start local layout analysis from the PDF text and horizontal separators.
    raw_lines = page.extract_text_lines(layout=False, strip=True)
    # PDF table rules can be stored as many adjoining short segments.
    rows = {}
    for edge in page.edges:
        if abs(edge.get("bottom", 0) - edge.get("top", 0)) <= 1:
            rows.setdefault(round(float(edge["top"]), 1), []).append(
                (edge["x0"], edge["x1"])
            )
    rule_y = []
    for y, intervals in rows.items():
        start = end = None
        for x0, x1 in sorted(intervals):
            if start is None:
                start, end = x0, x1
            elif x0 <= end + 2.5:
                end = max(end, x1)
            else:
                if end - start >= page.width * 0.5:
                    rule_y.append(y)
                start, end = x0, x1
        if start is not None and end - start >= page.width * 0.5:
            rule_y.append(y)
    boundaries = [0.0]
    for y in sorted(rule_y):
        if y > boundaries[-1] + 2 and y < page.height - 2:
            boundaries.append(y)
    boundaries.append(float(page.height))

    # Score plausible gutters using body-sized characters rather than small table text.
    candidates = [page.width * (0.30 + i * 0.01) for i in range(41)]
    content_size = (
        body_font_size
        or Counter(round(c.get("size", 0), 1) for c in page.chars).most_common(1)[0][0]
        if page.chars
        else 10
    )

    def gap_support(line, split):
        # Use body-size text for gutter voting so small table cells cannot dominate the body layout.
        chars = [
            c
            for c in line.get("chars", [])
            if c.get("text", "").strip() and c.get("size", 0) >= content_size * 0.95
        ]
        left = [c for c in chars if c["x1"] <= split]
        right = [c for c in chars if c["x0"] >= split]
        return bool(
            left
            and right
            and not any(c["x0"] < split < c["x1"] for c in chars)
            and min(c["x0"] for c in right) - max(c["x1"] for c in left)
            >= page.width * 0.015
        )

    order = 0
    band_layouts = []

    # Process each vertical band independently; one page can mix different layouts.
    for low, high in zip(boundaries, boundaries[1:]):
        band = [
            line
            for line in raw_lines
            if low <= min(c["top"] for c in line["chars"]) < high
        ]
        if not band:
            continue
        # Bibliographies often use a smaller font than the article body.
        band_sizes = Counter(
            round(c.get("size", 0), 1)
            for line in band
            for c in line["chars"]
            if c.get("text", "").strip()
        )
        if band_sizes and sum(
            any(c.get("size", 0) >= content_size * 0.95 for c in line["chars"])
            for line in band
        ) < max(3, len(band) * 0.12):
            content_size = band_sizes.most_common(1)[0][0]
        scores = [sum(gap_support(line, s) for line in band) for s in candidates]
        best = max(
            range(len(scores)),
            key=lambda k: (scores[k], -abs(candidates[k] - page.width / 2)),
        )
        split = candidates[best]
        eligible = sum(
            any(
                c.get("text", "").strip() and c.get("size", 0) >= content_size * 0.95
                for c in line["chars"]
            )
            for line in band
        )
        two = scores[best] >= max(3, eligible * 0.12)
        if not two:
            for line in band:
                order += 1
                ordered.append(
                    dict(
                        line,
                        flow="spanning_or_single",
                        region_kind="text",
                        reading_order=order,
                    )
                )
            band_layouts.append("spanning_or_single")
            continue

        # Buffer column lines and flush them before a spanning line interrupts the columns.
        pending_left = []
        pending_right = []
        had_span = False

        def flush():
            nonlocal order
            for flow, items in [("left", pending_left), ("right", pending_right)]:
                for ln in items:
                    order += 1
                    ordered.append(
                        dict(ln, flow=flow, region_kind="text", reading_order=order)
                    )
            pending_left.clear()
            pending_right.clear()

        for line in band:
            chars = line.get("chars", [])
            left = [c for c in chars if (c["x0"] + c["x1"]) / 2 < split]
            right = [c for c in chars if (c["x0"] + c["x1"]) / 2 >= split]
            # Test the actual character gap here, not the body-font voting subset.
            nonspace = [c for c in chars if c.get("text", "").strip()]
            l = [c for c in nonspace if c["x1"] <= split]
            r = [c for c in nonspace if c["x0"] >= split]
            actual_gap = bool(
                l
                and r
                and not any(c["x0"] < split < c["x1"] for c in nonspace)
                and min(c["x0"] for c in r) - max(c["x1"] for c in l)
                >= page.width * 0.015
            )
            spanning = bool(left and right) and not actual_gap
            if spanning:
                flush()
                order += 1
                had_span = True
                ordered.append(
                    dict(line, flow="spanning", region_kind="text", reading_order=order)
                )
            else:
                pending_left.extend(char_line_records(left))
                pending_right.extend(char_line_records(right))
        flush()
        band_layouts.append("mixed" if had_span else "two_column")

    # Report the inferred layout and keep its uncertainty available to later stages.
    summary.update(
        layout=(
            "mixed"
            if (
                "mixed" in band_layouts
                or (
                    "two_column" in band_layouts
                    and "spanning_or_single" in band_layouts
                )
            )
            else "two_column" if "two_column" in band_layouts else "single_or_uncertain"
        ),
        band_count=len(band_layouts),
        band_layouts=band_layouts,
        raw_line_count=len(raw_lines),
    )
    ordered = attach_scripts(ordered)
    for line in ordered:
        line["text"] = clean_text(line["text"])
    return ordered, summary


# Repair text-layer defects that pdfplumber passes through unchanged.
def clean_text(text):
    if "\x00" in text:
        text = repair_ligatures(text)
    if "(cid:" in text:
        # A glyph without a Unicode mapping prints as "(cid:N)", often repeated because the
        # PDF draws it several times. Show each as one replacement character instead.
        text = re.sub(r"(\(cid:\d+\))\1*", "�", text)
    return text


# Fold subscript/superscript lines back into the line they belong to, so that
# "Ni In" + "3" becomes "Ni3In". A script line is set in a clearly smaller font
# and sits inside the height and width of a larger-font host line.
def attach_scripts(lines):
    def size(line):
        sizes = sorted(c.get("size", 0) for c in line["chars"] if c.get("text", "").strip())
        return sizes[len(sizes) // 2] if sizes else 0

    def bounds(line):
        chars = line["chars"]
        return (
            min(c["x0"] for c in chars),
            min(c["top"] for c in chars),
            max(c["x1"] for c in chars),
            max(c["bottom"] for c in chars),
        )

    info = {id(l): (size(l), bounds(l)) for l in lines if l.get("chars")}
    hosts = {}
    for script in lines:
        if id(script) not in info:
            continue
        s_size, (sx0, stop, _, sbottom) = info[id(script)]
        best, best_overlap = None, 0
        for host in lines:
            if host is script or id(host) not in info or id(host) in hosts:
                continue
            h_size, (hx0, htop, hx1, hbottom) = info[id(host)]
            if not s_size or s_size > h_size * 0.85:
                continue
            # A script at the end of a line extends past it, so only its start must fall inside.
            if not hx0 - 2 <= sx0 <= hx1 + 2:
                continue
            overlap = min(sbottom, hbottom) - max(stop, htop)
            if overlap > 0.4 * (sbottom - stop) and overlap > best_overlap:
                best, best_overlap = host, overlap
        if best is not None:
            hosts[id(script)] = best

    if not hosts:
        return lines
    merged = {}
    for script in lines:
        host = hosts.get(id(script))
        if host is not None:
            merged.setdefault(id(host), [host]).append(script)
    result = []
    for line in lines:
        if id(line) in hosts:
            continue
        group = merged.get(id(line))
        if group:
            unique = {id(c): c for l in group for c in l["chars"]}
            chars = sorted(unique.values(), key=lambda c: c["x0"])
            # Give every character the host's vertical position so they read as one line.
            first = group[0]["chars"][0]
            rebuilt = char_line_records(
                [
                    dict(c, top=first["top"], bottom=first["bottom"], doctop=first.get("doctop", first["top"]))
                    for c in chars
                ]
            )
            line = dict(
                line,
                text=" ".join(r["text"] for r in rebuilt),
                chars=chars,
                top=min(c["top"] for c in chars),
                bottom=max(c["bottom"] for c in chars),
            )
        result.append(line)
    return result


_WORDS = None


# Some fonts map the fi/fl ligature glyphs to a NUL character ("signi\x00cant").
# Pick the ligature that turns the word into a dictionary word; default to "fi".
def repair_ligatures(text):
    global _WORDS
    if _WORDS is None:
        try:
            _WORDS = set(Path("/usr/share/dict/words").read_text().lower().split())
        except OSError:
            _WORDS = set()

    def fix(match):
        word = match.group(0)
        for ligature in ("fi", "fl", "ff", "ffi", "ffl"):
            candidate = word.replace("\x00", ligature)
            stem = candidate.lower().rstrip("s")
            if candidate.lower() in _WORDS or stem in _WORDS:
                return candidate
        return word.replace("\x00", "fi")

    return re.sub(r"\w*\x00[\w\x00]*", fix, text)


# ---------------------------------------------------------------------------
# 4.3 First structural pass
# ---------------------------------------------------------------------------


# Learn the paper typography, propose heading roles, and construct the initial document hierarchy.
def first_pass(pdf_path, ctx):
    from statistics import median

    # Fingerprint the source and learn the dominant body size and font family.
    pages = []
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        sizes = Counter(
            Counter(
                round(c.get("size", 0), 1)
                for c in page.chars
                if c.get("text", "").strip()
            ).most_common(1)[0][0]
            for page in pdf.pages
            # A page can hold only whitespace characters, which leaves nothing to count.
            if any(c.get("text", "").strip() for c in page.chars)
        )
        body_size = sizes.most_common(1)[0][0] if sizes else 10
        families = Counter(
            c.get("fontname", "").split("+")[-1]
            for page in pdf.pages
            for c in page.chars
            if abs(c.get("size", 0) - body_size) < 0.6
        )
        body_family = families.most_common(1)[0][0] if families else ""

        # Visit physical pages in order and remove duplicate PDF glyphs before analysis.
        for pno, page in enumerate(pdf.pages, 1):
            page = page.dedupe_chars(tolerance=1)
            ordered, layout = ordered_page_lines(page, pno, body_size)
            lines = []
            last_bottom = {}

            # Measure each line: font, emphasis, spacing, reading flow and private bounds.
            for k, line in enumerate(ordered, 1):
                lc = [c for c in line.get("chars", []) if c.get("text", "").strip()]
                font_size = round(median(c.get("size", 0) for c in lc), 1) if lc else 0
                italic_ratio = (
                    sum(
                        any(
                            tag in c.get("fontname", "").lower()
                            for tag in ("italic", "oblique")
                        )
                        for c in lc
                    )
                    / len(lc)
                    if lc
                    else 0
                )
                font_family = (
                    Counter(
                        c.get("fontname", "").split("+")[-1] for c in lc
                    ).most_common(1)[0][0]
                    if lc
                    else ""
                )
                box = (
                    (
                        min(c["x0"] for c in lc),
                        min(c["top"] for c in lc),
                        max(c["x1"] for c in lc),
                        max(c["bottom"] for c in lc),
                    )
                    if lc
                    else (0, 0, 0, 0)
                )
                bold_ratio = (
                    sum("bold" in c.get("fontname", "").lower() for c in lc) / len(lc)
                    if lc
                    else 0
                )
                top = min(c["top"] for c in lc) if lc else 0
                bottom = max(c["bottom"] for c in lc) if lc else top
                flow = line["flow"]
                gap = top - last_bottom.get(flow, top)
                last_bottom[flow] = bottom
                text = line["text"].strip()
                # A short, separate typographic unit, not inline emphasis, captions or author lists.
                short = (
                    3 <= len(text) <= 125
                    and len(text.split()) <= 16
                    and not re.search(r"[.!?;]$", text)
                )
                excluded = (
                    bool(
                        re.match(
                            r"^(?:fig(?:ure)?\.?|table|https?://|doi\b)", text, re.I
                        )
                    )
                    or "·" in text
                    or "@" in text
                )
                supported = (
                    (
                        bold_ratio >= 0.7
                        or italic_ratio >= 0.7
                        or font_size > body_size * 1.08
                        or font_family != body_family
                        or line.get("region_kind") == "heading"
                    )
                    and short
                    and not excluded
                )
                if (
                    font_size <= body_size * 1.05
                    and gap < font_size * 0.4
                    and line.get("region_kind") != "heading"
                ):
                    supported = False
                title_candidate = (
                    pno == 1
                    and top / page.height < 0.40
                    and font_size >= body_size * 1.35
                    and supported
                )

                # Retain observations and features so later decisions can be traced to source lines.
                item = {
                    "id": f"p{pno}_l{k}",
                    "page": pno,
                    "text": text,
                    "flow": flow,
                    "reading_order": line["reading_order"],
                    "region_kind": line["region_kind"],
                    "heading_typography": supported,
                    "font_size": font_size,
                    "bold_ratio": round(bold_ratio, 3),
                    "gap_before": round(gap, 2),
                    "title_candidate": title_candidate,
                    "margin_candidate": top < page.height * 0.055
                    or bottom > page.height * 0.94,
                    "italic_ratio": round(italic_ratio, 3),
                    "font_family": font_family,
                    "_box": box,
                }
                inside = ctx.asset_containing(pno, box)
                if inside:
                    item["region_kind"] = inside["kind"]
                if item["region_kind"] in (
                    "figure",
                    "table",
                    "equation",
                    "header",
                    "footer",
                ):
                    item["heading_typography"] = False
                lines.append(item)
            pages.append(
                {
                    "page": pno,
                    "lines": lines,
                    "raster_images": len(page.images),
                    "vector_objects": len(page.curves) + len(page.rects),
                    "layout": layout,
                }
            )
    # Alignment is relative to the observed flow's prose extent, not fixed page coordinates.
    for page in pages:
        for flow in {l["flow"] for l in page["lines"]}:
            members = [l for l in page["lines"] if l["flow"] == flow]
            prose = [
                l
                for l in members
                if len(l["text"]) > 60
                and l["font_size"] >= body_size * 0.9
                and l["region_kind"] == "text"
            ] or members
            left = min((l["_box"][0] for l in prose), default=0)
            right = max((l["_box"][2] for l in prose), default=0)
            for l in members:
                box = l["_box"]
                centered = (
                    abs((box[0] + box[2] - left - right) / 2) < max(3, body_size * 0.6)
                    and box[2] - box[0] < (right - left) * 0.85
                )
                l["alignment"] = (
                    "center"
                    if centered
                    else (
                        "left"
                        if abs(box[0] - left) < max(3, body_size * 0.6)
                        else "indented_or_other"
                    )
                )
                if (
                    centered
                    and l["gap_before"] >= body_size * 0.6
                    and 3 <= len(l["text"]) <= 100
                    and not re.search(r"[.!?;]$", l["text"])
                    and l["region_kind"] in ("text", "heading")
                ):
                    l["heading_typography"] = True

    # Keep repeated running headers/footers outside scientific sections; retain their text for review.
    def furniture_key(text):
        return re.sub(r"[\W\d_]+", "", text.casefold())

    repeated = Counter(
        furniture_key(line["text"])
        for page in pages
        for line in page["lines"]
        if line["margin_candidate"] and len(furniture_key(line["text"])) >= 5
    )
    furniture_ids = {
        line["id"]
        for page in pages
        for line in page["lines"]
        if line["margin_candidate"]
        and repeated[furniture_key(line["text"])] >= max(3, total * 0.2)
    }
    # These are generic schema roles, not a paper-specific section inventory.
    special = {
        "abstract": ("abstract", "front"),
        "keywords": ("keywords", "front"),
        "references": ("references", "back"),
        "bibliography": ("references", "back"),
        "acknowledgements": ("acknowledgements", "back"),
        "acknowledgments": ("acknowledgements", "back"),
        "funding": ("funding", "back"),
        "data availability": ("data_availability", "back"),
        "code availability": ("code_availability", "back"),
        "ethics statement": ("ethics_statement", "back"),
        "author contributions": ("author_contributions", "back"),
        "competing interests": ("competing_interests", "back"),
        "declarations": ("section", "back"),
    }

    # Propose standard roles or numbered headings; typography remains supporting evidence.
    candidates = {}
    for page in pages:
        for line in page["lines"]:
            if line["id"] in furniture_ids or line["title_candidate"]:
                continue
            text = line["text"]
            canonical = re.sub(r"\s+", " ", text).strip(": ").casefold()
            if all(len(token) == 1 for token in canonical.split()):
                canonical = canonical.replace(" ", "")
            if not line["heading_typography"] and canonical not in special:
                continue
            numbered = re.match(
                r"^(\d+(?:\.\d+){0,2})[.)]?\s+([^\W\d_][^\n]{2,110})$", text
            )
            if canonical in special:
                candidates[line["id"]] = {
                    "role": special[canonical][0],
                    "container": special[canonical][1],
                    "depth": 1,
                }
            elif re.match(r"^appendix\b", text, re.I):
                label = re.match(
                    r"^appendix\s+[A-Za-z]((?:\.\d+){0,2})\s*:", text, re.I
                )
                depth = 1 + label.group(1).count(".") if label else 1
                candidates[line["id"]] = {
                    "role": ["appendix", "subsection", "subsubsection"][depth - 1],
                    "container": "back",
                    "depth": depth,
                }
            elif numbered:
                depth = numbered.group(1).count(".") + 1
                candidates[line["id"]] = {
                    "role": ["section", "subsection", "subsubsection"][depth - 1],
                    "container": "body",
                    "depth": depth,
                }
            elif line["heading_typography"] and re.match(r"[^\W\d_]", text):
                candidates[line["id"]] = {
                    "role": None,
                    "container": "body",
                    "depth": None,
                    "continuation_only": text[0].islower(),
                }

    # Infer levels for unnumbered headings from relative font sizes.
    styles = sorted(
        {
            line["font_size"]
            for page in pages
            for line in page["lines"]
            if line["id"] in candidates and candidates[line["id"]]["role"] is None
        },
        reverse=True,
    )
    for page in pages:
        for line in page["lines"]:
            candidate = candidates.get(line["id"])
            if candidate and candidate["role"] is None:
                depth = min(3, styles.index(line["font_size"]) + 1)
                candidate.update(
                    role=["section", "subsection", "subsubsection"][depth - 1],
                    depth=depth,
                )

    # Create the front/body/back scaffold before placing recognized elements.
    article = node("article", "article", 1, total)
    front, body, back = (
        node("front", "front_matter", 1),
        node("body", "body", 1, total),
        node("back", "back_matter", total),
    )
    article["children"] = [front, body, back]
    containers = {"front": front, "body": body, "back": back}
    title_lines = [
        line for page in pages for line in page["lines"] if line["title_candidate"]
    ]
    recognized = {line["id"] for line in title_lines}
    if title_lines:
        front["children"].append(
            node(
                "publication_title",
                "title",
                1,
                text=" ".join(line["text"] for line in title_lines),
                attributes={
                    "method": "largest bold front-page typography",
                    "observation_ids": [line["id"] for line in title_lines],
                },
            )
        )

    # Maintain a heading stack; preserve unclassified runs beneath the current context.
    stack = []
    context = body
    current_container = "body"
    prior_heading = None
    for page in pages:
        pending = []
        bundle_count = 0

        def flush_pending():
            nonlocal pending, bundle_count
            if not pending:
                return
            bundle_count += 1
            # References/keywords have restricted child types. Preserve unclassified content beside them.
            holder = (
                context
                if context["type"]
                not in ("references", "keywords", "authors", "affiliations")
                else containers[current_container]
            )
            attrs = {
                "custom_type": "unidentified",
                "method": "unclassified first pass",
                "observation_ids": [x["id"] for x in pending],
                "scope": "Unclassified content in reading order; semantic element boundaries require review",
                "reading_order_method": page["layout"]["method"],
                "page_layout": page["layout"]["layout"],
                "layout_review_required": page["layout"]["review_required"],
                "possible_graphical_content": bool(
                    page["raster_images"] or page["vector_objects"]
                ),
            }
            if holder is not context:
                attrs["intended_parent_id"] = context["id"]
            holder["children"].append(
                node(
                    f'unidentified_p{page["page"]}_{bundle_count}',
                    "custom",
                    page["page"],
                    text="\n".join(x["text"] for x in pending),
                    label="Unidentified content",
                    review_required=True,
                    attributes=attrs,
                )
            )
            context["page_end"] = max(context["page_end"], page["page"])
            pending = []

        for line in page["lines"]:
            if line["id"] in recognized:
                continue
            if line["id"] in furniture_ids:
                body["children"].append(
                    node(
                        f'unidentified_p{page["page"]}_furniture_{line["id"]}',
                        "custom",
                        page["page"],
                        text=line["text"],
                        label="Unidentified publication furniture",
                        review_required=True,
                        attributes={
                            "custom_type": "unidentified",
                            "candidate_role": "running_header_or_footer",
                            "method": "repeated margin text",
                            "observation_ids": [line["id"]],
                        },
                    )
                )
                continue
            candidate = candidates.get(line["id"])
            # Join consecutive heading lines only when typography, flow and tight line spacing match.
            if (
                candidate
                and prior_heading
                and not pending
                and prior_heading["page"] == page["page"]
            ):
                prev = prior_heading["line"]
                if (
                    line["flow"] == prev["flow"]
                    and line["font_size"] == prev["font_size"]
                    and 0 <= line["gap_before"] <= line["font_size"] * 0.4
                ):
                    prior_heading["node"]["title"] += " " + line["text"]
                    prior_heading["node"]["attributes"]["observation_ids"].append(
                        line["id"]
                    )
                    prior_heading["line"] = line
                    recognized.add(line["id"])
                    continue
            if candidate and candidate.get("continuation_only"):
                # An isolated lowercase fragment may be a missed wrap, not a standalone heading.
                pending.append(line)
                prior_heading = None
                continue

            # Flush pending text before changing the current structural container.
            if candidate:
                flush_pending()
                role = candidate["role"]
                depth = candidate["depth"]
                container = candidate["container"]
                if container != current_container:
                    stack = []
                current_container = container
                while stack and stack[-1][0] >= depth:
                    stack.pop()
                # Preserve uncertain skipped levels, rather than inventing hierarchy.
                if depth > 1 and (not stack or stack[-1][0] != depth - 1):
                    pending.append(line)
                    prior_heading = None
                    continue
                parent = (
                    stack[-1][1]
                    if stack and role in ("subsection", "subsubsection")
                    else containers[container]
                )
                h = node(
                    "heading_" + line["id"],
                    role,
                    page["page"],
                    title=line["text"],
                    review_required=page["layout"]["review_required"],
                    attributes={
                        "method": "generic heading typography and spacing",
                        "font_size": line["font_size"],
                        "observation_ids": [line["id"]],
                        "layout_review_required": page["layout"]["review_required"],
                    },
                )
                parent["children"].append(h)
                recognized.add(line["id"])
                context = h
                if role in ("section", "subsection", "subsubsection", "appendix"):
                    stack.append((depth, h))
                else:
                    stack = []
                prior_heading = {"node": h, "page": page["page"], "line": line}
            else:
                pending.append(line)
                prior_heading = None
        flush_pending()
    expand_ranges(article)
    tree = {
        "schema_version": "1.0",
        "source": {
            "pdf": str(pdf_path),
            "pages": total,
            "sha256": pdf_hash,
            "page_numbering": "one-based physical PDF pages",
        },
        "extraction": {
            "library": "pdfplumber",
            "version": pdfplumber.__version__,
            "method": "Layout-aware generic numbered and unnumbered heading extraction against original schema",
            "limitations": [
                "Heading levels are inferred from document typography. Remaining content and layouts require source review."
            ],
            "body_font_size": body_size,
            "body_font_family": body_family,
        },
        "document": article,
    }
    validate_structure(tree, ctx.schema)
    return tree, pages


# ---------------------------------------------------------------------------
# 4.4 Extraction context and graphical candidates
# ---------------------------------------------------------------------------


# Keep extraction settings, private assets and generated IDs together during one run.
@dataclass
class ExtractionContext:
    schema: dict
    out: Path
    assets: list = field(default_factory=list)
    sequence: int = 0

    def __post_init__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.schema_hash = hashlib.sha256(
            json.dumps(self.schema, sort_keys=True).encode()
        ).hexdigest()

    # Allocate a unique node ID and retain a review-required interpretation.
    def make(self, role, page, text=None, **attrs):
        self.sequence += 1
        n = node(f"x{self.sequence:06}", role, page, review_required=True)
        if text is not None:
            n["text"] = text
        n["attributes"] = attrs
        return n

    # Find whether a text region falls predominantly inside a graphical candidate.
    def asset_containing(self, page, box):
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        for a in sorted(self.assets, key=lambda a: a["kind"] != "figure"):
            if a["page"] == page and overlap_fraction(box, a["box"]) >= 0.65:
                return a
        return None


# Measure the fraction of the first rectangle covered by the second rectangle.
def overlap_fraction(a, b):
    return (
        max(0, min(a[2], b[2]) - max(a[0], b[0]))
        * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    )


# Collect embedded images and ruled grids as candidates, not verified scientific objects.
def discover_assets(pdf_path, ctx):
    """Discover candidates; an unlabelled image or grid is not automatically a figure/table."""
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, 1):

            # Retain substantial embedded images and ignore very small decorative objects.
            for k, img in enumerate(page.images, 1):
                box = (img["x0"], img["top"], img["x1"], img["bottom"])
                if (box[2] - box[0]) * (
                    box[3] - box[1]
                ) < page.width * page.height * 0.004:
                    continue
                ctx.assets.append(
                    {
                        "id": f"p{pno}_image{k}",
                        "page": pno,
                        "kind": "figure",
                        "box": box,
                        "rows": None,
                        "method": "embedded_image_geometry",
                        "consumed": False,
                    }
                )

            # Attempt ruled-table extraction; unsupported geometry leaves an empty candidate list.
            try:
                tables = page.find_tables()
            except (ValueError, TypeError):
                tables = []

            # Reject sparse grids and grids already covered by image candidates.
            for k, t in enumerate(tables, 1):
                rows = t.extract(x_tolerance=1, y_tolerance=3)
                if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
                    continue
                if sum(bool(c and c.strip()) for row in rows for c in row) < 4:
                    continue
                box = tuple(t.bbox)
                if any(
                    a["page"] == pno
                    and a["kind"] == "figure"
                    and overlap_fraction(box, a["box"]) > 0.8
                    for a in ctx.assets
                ):
                    continue
                ctx.assets.append(
                    {
                        "id": f"p{pno}_grid{k}",
                        "page": pno,
                        "kind": "table",
                        "box": box,
                        "rows": rows,
                        "method": "ruled_grid_geometry",
                        "consumed": False,
                    }
                )


# ---------------------------------------------------------------------------
# 4.5 Content evidence and candidate rules
# ---------------------------------------------------------------------------


CAPTION = re.compile(
    r"^(?P<kind>fig(?:ure)?s?\.?|tables?)\s*(?P<label>[A-Z]?\d+(?:[a-z])?)\b[.:]?\s*(?P<text>.*)",
    re.I,
)


MENTION = re.compile(
    r"\b(?P<kind>fig(?:ure)?s?\.?|tables?)\s*(?P<label>[A-Z]?\d+(?:[a-z])?)\b", re.I
)


DECLARATIONS = {
    "acknowledgements": "acknowledgements",
    "acknowledgments": "acknowledgements",
    "author contributions": "author_contributions",
    "funding": "funding",
    "data availability": "data_availability",
    "code availability": "code_availability",
    "conflict of interest": "competing_interests",
    "competing interests": "competing_interests",
    "ethics statement": "ethics_statement",
}


# Normalize whitespace and Unicode for comparisons without rewriting stored source text.
def text_key(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


# Record the decision rule and literal source snippets; keep verification explicitly false.
def evidence(rule, lines, **details):
    return {
        "rule": rule,
        "verified": False,
        "review_required": True,
        "source": [
            {"block_id": l["id"], "page": l["page"], "quote": l["text"]} for l in lines
        ],
        **details,
    }


# Use caption syntax, typography and proximity to propose an associated image or table.
def caption_asset(line, ctx, body_size):

    # Require recognizable caption syntax and a style distinguishable from ordinary prose.
    match = CAPTION.match(line["text"])
    if not match:
        return None
    kind = "table" if match["kind"].lower().startswith("table") else "figure"
    styled = (
        line["font_size"] < body_size * 0.99
        or line["bold_ratio"] > 0.4
        or line.get("italic_ratio", 0) > 0.4
        or line["region_kind"] == "caption"
    )
    if not styled:
        return None

    # Rank unused assets of the same kind on the same page by their distance to the caption.
    box = line["_box"]
    scores = []
    for a in ctx.assets:
        if a["page"] != line["page"] or a["kind"] != kind or a["consumed"]:
            continue
        b = a["box"]
        dx = max(b[0] - box[2], box[0] - b[2], 0)
        dy = max(b[1] - box[3], box[1] - b[3], 0)
        if dx + dy <= body_size * 16:
            scores.append((dx + dy, a))
    best = min(scores, key=lambda x: x[0])[1] if scores else None
    return match, kind, best


# Extend a caption only while nearby lines retain compatible typography and flow.
def collect_caption(anchor, lines, asset, ctx):
    selected = [anchor]
    start = lines.index(anchor)
    prev = anchor

    # Stop at a page/column change, new caption, asset, heading, or excessive spacing.
    for l in lines[start + 1 :]:
        if l["page"] != anchor["page"] or l["flow"] != anchor["flow"]:
            break
        if CAPTION.match(l["text"]) or l["region_kind"] in ("table", "figure"):
            break
        if abs(l["font_size"] - anchor["font_size"]) > 0.8 or l["heading_typography"]:
            break
        if l["_box"][1] - prev["_box"][3] > anchor["font_size"] * 1.8:
            break
        if asset:
            b = asset["box"]
            ab = anchor["_box"]
            lb = l["_box"]
            side = ab[2] < b[0] or ab[0] > b[2]
            if side and (
                lb[3] > b[3] + anchor["font_size"]
                or abs(lb[0] - ab[0]) > anchor["font_size"]
            ):
                break
        selected.append(l)
        prev = l
    return selected


# Summarize recurring styles so decisions can be inspected against document-wide evidence.
def learn_style_profile(pages, body_size):

    # Group observations by font family, size and emphasis; retain counts and a few examples.
    groups = {}
    for p in pages:
        for l in p["lines"]:
            key = (
                l["font_family"],
                l["font_size"],
                round(l["bold_ratio"] > 0.6),
                round(l["italic_ratio"] > 0.6),
            )
            g = groups.setdefault(
                key,
                {
                    "font": key[0],
                    "size": key[1],
                    "bold": bool(key[2]),
                    "italic": bool(key[3]),
                    "line_count": 0,
                    "long_prose_lines": 0,
                    "pages": set(),
                    "examples": [],
                },
            )
            g["line_count"] += 1
            g["pages"].add(l["page"])
            if (
                len(l["text"]) > 70
                and not l["heading_typography"]
                and l["region_kind"] == "text"
            ):
                g["long_prose_lines"] += 1
            if len(g["examples"]) < 3:
                g["examples"].append(l["text"][:120])
    for g in groups.values():
        g["pages"] = sorted(g["pages"])
    return {
        "body_size": body_size,
        "styles": list(groups.values()),
        "method": "Repeated per-page body sizes, style differences, spacing and layout; no fixed font names",
    }


# Parse delimiter-separated author names conservatively; reject unsupported syntax.
def parse_author_parts(text):
    """Conservative delimiter-based proposal. Ambiguous or unsupported formats stay unidentified."""
    if "·" not in text and ";" not in text:
        return []

    # Check every proposed name and marker; an ambiguous list is left for later interpretation.
    parts = re.split(r"[·;]", text)
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(.+?)([\d,*†‡]+)?", part)
        name = (m[1] if m else "").strip()
        markers = re.findall(r"\d+", m[2] or "") if m else []
        words = name.split()
        if not 2 <= len(words) <= 6:
            return []
        if not all(re.fullmatch(r"[^\W\d_][\w.'’\-]*", w, re.UNICODE) for w in words):
            return []
        if not words[0][0].isupper() or not words[-1][0].isupper():
            return []
        result.append((name, markers))
    return result if len(result) >= 2 else []


# ---------------------------------------------------------------------------
# 4.6 Interpret local content
# ---------------------------------------------------------------------------


# Assign content roles, preserve unclassified text, and record evidence for proposed associations.
def interpret_content(tree, pages, ctx):
    """Assign roles with source snippets; keep uncertain roles and associations explicit."""

    # Index structural nodes and source lines for role assignment and evidence lookup.
    ids = index(tree)
    body = ids["body"]
    front = ids["front"]
    back = ids["back"]
    lines = [l for p in pages for l in p["lines"]]
    by_id = {l["id"]: l for l in lines}
    # Primary ownership already assigned to recognized elements must not be reassigned to assets.
    consumed = {
        block
        for n in walk(tree["document"])
        if n.get("attributes", {}).get("custom_type") != "unidentified"
        for block in n.get("attributes", {}).get("observation_ids", [])
    }
    floats = []
    body_size = tree["extraction"]["body_font_size"]
    unknown = [
        (p, c)
        for p in walk(tree["document"])
        for c in p["children"]
        if c.get("attributes", {}).get("custom_type") == "unidentified"
    ]
    # Delimited names before the abstract: syntactic author evidence, with review required.
    abstract_order = min(
        (i for i, l in enumerate(lines) if l["text"].strip().casefold() == "abstract"),
        default=0,
    )
    author_lines = [l for l in lines[:abstract_order] if parse_author_parts(l["text"])]
    author_container = None
    labels = set()
    if author_lines:
        author_container = ctx.make(
            "authors",
            1,
            observation_ids=[l["id"] for l in author_lines],
            decision=evidence(
                "Delimited name lists between title and abstract", author_lines
            ),
        )
        front["children"].append(author_container)
        for l in author_lines:
            for name, markers in parse_author_parts(l["text"]):
                a = ctx.make(
                    "author",
                    l["page"],
                    name,
                    evidence_ids=[l["id"]],
                    decision=evidence("Name/marker syntax", [l]),
                )
                a["affiliation_labels"] = markers
                labels.update(markers)
                author_container["children"].append(a)
            consumed.add(l["id"])

    # Resolve institutional notes using author markers, typography and vocabulary.
    affiliation_container = None
    affiliation_nodes = {}
    institution = re.compile(
        r"\b(university|institute|laborator|school|department|college|centre|center|science|research)\w*|s\.?p\.?a\.?",
        re.I,
    )
    for i, l in enumerate(lines):
        if l["page"] != 1 or l["id"] in consumed:
            continue
        m = re.match(r"^(\d+)\s*(.*)$", l["text"])
        if not m or m[1] not in labels or l["font_size"] > body_size:
            continue
        proposal = [l]
        for next_line in lines[i + 1 : i + 5]:
            if next_line["page"] != 1 or next_line["flow"] != l["flow"]:
                break
            if (
                re.match(r"^\d+(?:\s|$)", next_line["text"])
                or next_line["font_size"] > body_size * 0.97
            ):
                break
            if next_line["_box"][1] - proposal[-1]["_box"][3] > body_size * 1.5:
                break
            proposal.append(next_line)
        if not institution.search(" ".join(x["text"] for x in proposal)):
            continue
        if not affiliation_container:
            affiliation_container = ctx.make(
                "affiliations",
                1,
                decision=evidence("Author markers plus institutional notes", []),
            )
            front["children"].append(affiliation_container)
        n = ctx.make(
            "affiliation",
            1,
            "\n".join(x["text"] for x in proposal),
            observation_ids=[x["id"] for x in proposal],
            decision=evidence(
                "Matching author marker and institutional vocabulary", proposal
            ),
        )
        n["label"] = m[1]
        affiliation_container["children"].append(n)
        affiliation_nodes[m[1]] = n
        consumed.update(x["id"] for x in proposal)

    # Link only resolved affiliation IDs; keep unmatched labels explicit.
    if author_container:
        for a in author_container["children"]:
            a["affiliation_ids"] = [
                affiliation_nodes[label]["id"]
                for label in a["affiliation_labels"]
                if label in affiliation_nodes
            ]
            a["attributes"]["unresolved_affiliation_labels"] = [
                label
                for label in a["affiliation_labels"]
                if label not in affiliation_nodes
            ]
    # Captions anchor figure/table candidates. Geometry is evidence, not a logical-parent rule.
    for l in lines:
        if l["id"] in consumed:
            continue
        found = caption_asset(l, ctx, body_size)
        if not found:
            continue
        match, kind, asset = found
        caption_lines = collect_caption(l, lines, asset, ctx)
        label = match["label"]
        f = ctx.make(
            kind,
            l["page"],
            label=label,
            asset_status="located" if asset else "caption_only",
            logical_parent_status="unresolved",
            decision=evidence(
                "Caption label, typography and adjacent asset", caption_lines
            ),
        )
        f["label"] = ("Table " if kind == "table" else "Figure ") + label
        f["attributes"]["asset_id"] = asset["id"] if asset else None
        caption = ctx.make(
            "table_title" if kind == "table" else "caption",
            l["page"],
            "\n".join(x["text"] for x in caption_lines),
            observation_ids=[x["id"] for x in caption_lines],
            decision=evidence("Caption text in same style and flow", caption_lines),
        )
        f["children"].append(caption)
        consumed.update(x["id"] for x in caption_lines)

        # Preserve asset text ownership and derive table cells only when a grid is available.
        if asset:
            asset["consumed"] = True
            inside = [
                x
                for x in lines
                if x["page"] == l["page"]
                and x["id"] not in consumed
                and overlap_fraction(x["_box"], asset["box"]) >= 0.65
            ]
            if kind == "figure":
                image = ctx.make(
                    "image",
                    l["page"],
                    observation_ids=[x["id"] for x in inside],
                    content_status="unidentified",
                    asset_id=asset["id"],
                )
                f["children"].append(image)
            else:
                f["attributes"]["observation_ids"] = [x["id"] for x in inside]
                rows = asset.get("rows")
                if rows:
                    for row_index, row in enumerate(rows):
                        header = (
                            row_index == 0
                            and sum(
                                bool(cell and re.search(r"[A-Za-z]", cell))
                                for cell in row
                            )
                            >= len(row) / 2
                        )
                        r = ctx.make(
                            "table_header" if header else "table_row",
                            l["page"],
                            row_index=row_index,
                            header_inferred=header,
                            decision=evidence(
                                "Ruled-grid extraction; header is provisional", []
                            ),
                        )
                        for col_index, cell in enumerate(row):
                            r["children"].append(
                                ctx.make(
                                    "table_cell",
                                    l["page"],
                                    cell or "",
                                    column_index=col_index,
                                    missing_text=cell is None,
                                )
                            )
                        f["children"].append(r)
                else:
                    for x in inside:
                        f["children"].append(
                            ctx.make(
                                "table_row_text",
                                x["page"],
                                x["text"],
                                evidence_ids=[x["id"]],
                            )
                        )
                    f["attributes"]["cell_structure_status"] = "unidentified"
            consumed.update(x["id"] for x in inside)
        elif kind == "figure":
            f["children"].append(
                ctx.make(
                    "image",
                    l["page"],
                    content_status="unidentified",
                    asset_status="not_located",
                )
            )
        body["children"].append(f)
        floats.append(f)
    # Preserve unlabelled graphical/grid candidates without inventing semantic labels.
    for a in ctx.assets:
        if a["consumed"]:
            continue
        remaining = [
            l
            for l in lines
            if l["page"] == a["page"]
            and l["id"] not in consumed
            and overlap_fraction(l["_box"], a["box"]) >= 0.65
        ]
        body["children"].append(
            ctx.make(
                "custom",
                a["page"],
                "\n".join(l["text"] for l in remaining),
                custom_type="unidentified",
                candidate_role="graphic_or_grid",
                asset_id=a["id"],
                observation_ids=[l["id"] for l in remaining],
                decision=evidence("Unlabelled asset needs semantic review", remaining),
            )
        )
        consumed.update(l["id"] for l in remaining)
    # Convert confidently prose-like runs; retain equations/small labels and unknown layouts.
    current_reference = {}
    section_for = {}
    paragraphs = []
    parent_map = {c["id"]: p for p in walk(tree["document"]) for c in p["children"]}

    def nearest_section(parent):
        while parent:
            if parent["type"] in ("section", "subsection", "subsubsection", "appendix"):
                return parent["id"]
            parent = parent_map.get(parent["id"])
        return None

    # Visit bundles in reading order: tree order can put an earlier page's bundle last,
    # which would glue it onto the previous page's final reference.
    position = {l["id"]: i for i, l in enumerate(lines)}
    unknown.sort(
        key=lambda pair: min(
            (position[i] for i in pair[1]["attributes"].get("observation_ids", [])),
            default=len(lines),
        )
    )

    # Replace each unidentified bundle with role-consistent runs in source order.
    for parent, old in unknown:
        replacements = []
        run = []
        run_role = None
        actual_parent = ids.get(
            old.get("attributes", {}).get("intended_parent_id"), parent
        )
        section_id = nearest_section(actual_parent)

        def flush():
            nonlocal run, run_role
            if not run:
                return
            role = run_role
            attrs = {
                "observation_ids": [l["id"] for l in run],
                "decision": evidence("Source block role proposal", run),
                "text_fidelity_status": "requires_source_review",
            }
            if role == "custom":
                attrs["custom_type"] = "unidentified"
            n = ctx.make(
                role, run[0]["page"], "\n".join(l["text"] for l in run), **attrs
            )
            n["page_end"] = run[-1]["page"]
            replacements.append(n)
            if role == "paragraph":
                paragraphs.append(n)
                section_for[n["id"]] = section_id
            run = []
            run_role = None

        for block_id in old["attributes"].get("observation_ids", []):
            if block_id in consumed:
                continue
            l = by_id[block_id]

            # Start or extend numbered bibliography entries without inventing missing entries.
            if actual_parent["type"] == "references":
                flush()
                m = re.match(r"^\[?(\d{1,4})[\].)]\s+(.{8,})$", l["text"])
                ref = current_reference.get(actual_parent["id"])
                if m and (ref is None or int(m[1]) > ref.get("number", 0)):
                    ref = ctx.make(
                        "reference",
                        l["page"],
                        m[2],
                        observation_ids=[l["id"]],
                        reference_flow=l["flow"],
                        decision=evidence("Numbered bibliography entry", [l]),
                    )
                    ref["number"] = int(m[1])
                    actual_parent["children"].append(ref)
                    current_reference[actual_parent["id"]] = ref
                elif ref and l["flow"] == ref["attributes"].get("reference_flow"):
                    ref["text"] += "\n" + l["text"]
                    ref["page_end"] = l["page"]
                    ref["attributes"]["observation_ids"].append(l["id"])
                    ref["attributes"]["decision"]["source"].append(
                        {"block_id": l["id"], "page": l["page"], "quote": l["text"]}
                    )
                else:
                    run = [l]
                    run_role = "custom"
                    flush()
                continue

            # Recognize explicit declaration labels before falling back to prose or unidentified text.
            declaration = next(
                (
                    name
                    for name in DECLARATIONS
                    if re.match("^" + re.escape(name) + r"\b", l["text"], re.I)
                ),
                None,
            )
            if declaration:
                flush()
                role = DECLARATIONS[declaration]
                n = ctx.make(
                    role,
                    l["page"],
                    l["text"],
                    observation_ids=[l["id"]],
                    decision=evidence("Explicit declaration label", [l]),
                )
                back["children"].append(n)
                continue
            if old["attributes"].get("candidate_role") == "running_header_or_footer":
                role = "publication_metadata"
            elif (
                l["region_kind"] in ("figure", "table", "notes")
                or l["font_size"] < body_size * 0.92
            ):
                role = "custom"
            elif len(l["text"]) >= 25 and re.search(r"[A-Za-z]{3}", l["text"]):
                role = "paragraph"
            else:
                role = "custom"
            if run and (
                role != run_role
                or l["flow"] != run[-1]["flow"]
                or l["gap_before"] > body_size * 0.7
            ):
                flush()
            run.append(l)
            run_role = role
        flush()
        position = parent["children"].index(old)
        parent["children"][position : position + 1] = replacements
    # Relations contain exact source snippets and stay non-executable/review-required.
    relations = []
    asset_lookup = {}

    # Use explicit textual mentions to propose a section association; ambiguity stays unresolved.
    for f in floats:
        asset_lookup.setdefault(
            (f["type"], f["label"].split()[-1].casefold()), []
        ).append(f)
    for paragraph in paragraphs:
        for block_id in paragraph["attributes"]["observation_ids"]:
            l = by_id[block_id]
            for match in MENTION.finditer(l["text"]):
                kind = (
                    "table" if match["kind"].lower().startswith("table") else "figure"
                )
                targets = asset_lookup.get((kind, match["label"].casefold()), [])
                relations.append(
                    {
                        "type": "refers_to",
                        "from": paragraph["id"],
                        "to": targets[0]["id"] if len(targets) == 1 else None,
                        "label": kind + " " + match["label"],
                        "section_id": section_for[paragraph["id"]],
                        "evidence": {
                            "page": l["page"],
                            "block_id": l["id"],
                            "quote": l["text"],
                        },
                        "verified": False,
                        "review_required": True,
                        "supports_claim": False,
                    }
                )
    for f in floats:
        refs = [r for r in relations if r["to"] == f["id"]]
        candidates = sorted({r["section_id"] for r in refs if r["section_id"]})
        f["attributes"]["referring_sections"] = candidates
        f["attributes"]["association_evidence"] = refs
        if len(candidates) == 1:
            # This is an explicit, provisional association, not an approved semantic mapping.
            target = index(tree).get(candidates[0])
            body["children"].remove(f)
            target["children"].append(f)
            f["attributes"][
                "logical_parent_status"
            ] = "proposed_from_unique_referring_section"
        else:
            f["attributes"]["logical_parent_status"] = (
                "unresolved_no_reference"
                if not candidates
                else "unresolved_multiple_referring_sections"
            )
    expand_ranges(tree["document"])
    return relations


# ---------------------------------------------------------------------------
# 4.7 Citation field parsing
# ---------------------------------------------------------------------------


YEAR = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\b")


DOI = re.compile(r"\b10\.\d{4,9}/[^\s<>]+", re.I)


URL = re.compile(r"https?://[^\s<>]+", re.I)


NAME = re.compile(
    r"^(?:(?:[A-Z]\.[ ]*)+[A-Za-zÀ-ž][\wÀ-ž .\-\u2019\']*|[A-ZÀ-ž][\wÀ-ž\-\u2019\']+(?:\s+(?:[A-ZÀ-ž][\wÀ-ž\-\u2019\']+|de|van|von|di)){1,3})$"
)


# Recover literal bibliographic spans and retain every interval the parser cannot interpret.
def parse_citation(text):
    """Return literal spans and explicit unresolved intervals, never normalized guesses."""

    # Track accepted literal spans and the intervals they occupy in the original reference.
    fields = []
    occupied = []

    def add(role, start, end, label=None):
        while start < end and text[start].isspace():
            start += 1
        while end > start and (
            text[end - 1].isspace()
            or (role == "citation_venue" and text[end - 1] == ",")
        ):
            end -= 1
        if end <= start:
            return
        fields.append(
            dict(
                type=role,
                text=text[start:end],
                start=start,
                end=end,
                **({"label": label} if label else {}),
            )
        )
        occupied.append((start, end))

    # Work on a same-length whitespace view: offsets always address the original.
    flat = text.replace("\n", " ")

    # Extract DOI and URL spans without remote lookup or claims that identifiers are correct.
    for pat, label in [(DOI, "DOI"), (URL, "URL")]:
        for m in pat.finditer(flat):
            end = m.end()
            while end > m.start() and flat[end - 1] in ".,;":
                end -= 1
            if label == "URL" and any(a <= m.start() and end <= b for a, b in occupied):
                continue
            add("citation_identifier", m.start(), end, label)

    # Select a year candidate outside already occupied identifier spans.
    years = list(YEAR.finditer(flat))
    year = next(
        (
            m
            for m in reversed(years)
            if not any(a <= m.start() < b for a, b in occupied)
        ),
        None,
    )
    if year:
        add("citation_year", year.start(), year.end())
    # Nature-style surname, initials author lists.
    names = []
    pos = 0
    surname = re.compile(r"\s*([A-ZÀ-ž][\wÀ-ž'’\- ]{1,40},\s*(?:[A-Z]\.\s*){1,5})")
    while m := surname.match(flat, pos):
        names.append((m.start(1), m.end(1)))
        pos = m.end()
        sep = re.match(r"\s*(?:,|&|and)\s*", flat[pos:])
        if sep:
            pos += sep.end()
        else:
            break
    # Initials-first / full-name comma lists, or one full-name author before a period.
    if not names:
        pos = 0
        for m in re.finditer(r"[^,]+(?:,|$)", flat):
            candidate = m.group().rstrip(",").strip()
            start = m.start() + len(m.group()) - len(m.group().lstrip())
            if NAME.fullmatch(candidate):
                names.append((start, start + len(candidate)))
                pos = m.end()
            elif candidate.lower() in ("et al.", "et al"):
                pos = m.end()
                break
            else:
                break

    # Add author spans only after a supported name-list pattern has matched.
    for a, b in names:
        add("author", a, b)
    if names:
        tail = re.match(
            r"\s*(?:et al\.?\s*)?(?:\((?:18|19|20)\d{2}[a-z]?\)\.?\s*)?", flat[pos:]
        )
        pos += tail.end() if tail else 0
        # Book syntax: title (publisher, year). Do not guess a venue for a book.
        book = re.search(r"\(([^()]+),\s*((?:18|19|20)\d{2})\)\.?\s*$", flat)
        if book and book.start() > pos and not re.search(r"\d", book[1]):
            end = book.start()
            title = flat[pos:end].rstrip(" .,")
            edition = re.search(r",\s*([^,]*\bedition)\s*$", title, re.I)
            if edition:
                add("citation_edition", pos + edition.start(1), pos + edition.end(1))
                title = title[: edition.start()].rstrip(" ,")
            add("title", pos, pos + len(title))
            add("citation_publisher", book.start(1), book.end(1))
        else:
            # A title sentence precedes the venue. Periods inside initials are already consumed.
            boundary = re.search(r"\.\s+(?=[A-Z])", flat[pos:])
            if boundary:
                title_end = pos + boundary.start()
                add("title", pos, title_end)
                venue_start = pos + boundary.end()
                journal = re.search(
                    r"\s+(\d+)\s*(?:\(([^()]+)\))?\s*[, :]\s*([A-Za-z]?\d+(?:[\u2013\u2014\-]\d+)?)",
                    flat[venue_start:],
                )
                if journal:
                    offset = venue_start
                    add("citation_venue", venue_start, offset + journal.start())
                    add(
                        "citation_volume",
                        offset + journal.start(1),
                        offset + journal.end(1),
                    )
                    if journal[2]:
                        add(
                            "citation_issue",
                            offset + journal.start(2),
                            offset + journal.end(2),
                        )
                    add(
                        "citation_pages",
                        offset + journal.start(3),
                        offset + journal.end(3),
                    )
                # APA venue followed by comma, volume(issue), pages.
                else:
                    journal = re.search(
                        r",\s*(\d+)\s*(?:\(([^()]+)\))?\s*,\s*(\d+(?:[\u2013\u2014\-]\d+)?)",
                        flat[venue_start:],
                    )
                    if journal:
                        add(
                            "citation_venue", venue_start, venue_start + journal.start()
                        )
                        for role, g in [
                            ("citation_volume", 1),
                            ("citation_issue", 2),
                            ("citation_pages", 3),
                        ]:
                            if journal[g]:
                                add(
                                    role,
                                    venue_start + journal.start(g),
                                    venue_start + journal.end(g),
                                )
    mask = [False] * len(text)
    for a, b in occupied:
        mask[a:b] = [True] * (b - a)
    residual = []
    start = None
    for i in range(len(text) + 1):
        unknown = i < len(text) and not mask[i]
        if unknown and start is None:
            start = i
        if not unknown and start is not None:
            if re.search(r"\w", text[start:i]):
                residual.append({"text": text[start:i], "start": start, "end": i})
            start = None
    return {
        "fields": fields,
        "unparsed_segments": residual,
        "status": (
            "partial"
            if residual
            or not names
            or not {"title", "citation_year"}.issubset({f["type"] for f in fields})
            else "parsed_requires_review"
        ),
    }


# Attach citation subtrees to references while preserving raw reference text and page evidence.
def enrich_citations(tree, ctx):
    """Attach citation components with precise raw-text offsets and source-page evidence."""

    # Traverse the hierarchy in preorder so every element can be inspected uniformly.
    def walk(n):
        yield n
        for c in n["children"]:
            yield from walk(c)

    refs = [n for n in walk(tree["document"]) if n["type"] == "reference"]
    counts = {}
    partial = 0
    for ref in refs:
        raw = ref.get("text", "")
        parsed = parse_citation(raw)
        citation = ctx.make(
            "citation",
            ref["page_start"],
            parse_status=parsed["status"],
            unparsed_segments=parsed["unparsed_segments"],
            parser="literal-span citation parser v1",
        )
        citation["page_end"] = ref["page_end"]
        ref["children"] = [citation]
        source = ref.get("attributes", {}).get("decision", {}).get("source", [])
        # Reference text omits its leading entry label. Reconcile line lengths by literal matches.
        line_spans = []
        cursor = 0
        for item in source:
            quote = item["quote"]
            literal = (
                re.sub(r"^\[?\d{1,4}[\].)]\s+", "", quote) if cursor == 0 else quote
            )
            start = raw.find(literal, cursor)
            if start >= 0:
                line_spans.append((start, start + len(literal), item))
                cursor = start + len(literal)
        authors = None
        for field in parsed["fields"]:
            matches = [
                item
                for a, b, item in line_spans
                if a < field["end"] and b > field["start"]
            ]
            start = min((item["page"] for item in matches), default=ref["page_start"])
            end = max((item["page"] for item in matches), default=ref["page_end"])
            role = field["type"]
            n = ctx.make(
                role,
                start,
                field["text"],
                reference_text_span={
                    "start": field["start"],
                    "end": field["end"],
                    "policy": "zero-based half-open original reference text",
                },
                decision={
                    "rule": "Citation syntax and literal source span",
                    "source": matches,
                    "verified": False,
                    "review_required": True,
                },
                page_evidence_status=(
                    "matched_source_lines" if matches else "reference_range_fallback"
                ),
            )
            n["page_end"] = end
            if "label" in field:
                n["label"] = field["label"]
            if role == "author":
                if authors is None:
                    authors = ctx.make("authors", start)
                    authors["page_end"] = end
                    citation["children"].append(authors)
                authors["children"].append(n)
                authors["page_start"] = min(authors["page_start"], start)
                authors["page_end"] = max(authors["page_end"], end)
            else:
                citation["children"].append(n)
            counts[role] = counts.get(role, 0) + 1
            assert raw[field["start"] : field["end"]] == n["text"]
        if parsed["status"] == "partial":
            partial += 1
    return {
        "entries": len(refs),
        "field_counts": counts,
        "partial_entries": partial,
        "all_require_review": True,
        "policy": "Preserve raw text; parse literal spans; keep unsupported syntax unresolved; no bibliographic existence verification.",
    }


# ---------------------------------------------------------------------------
# 4.8 Construct the initial JSON
# ---------------------------------------------------------------------------


# Orchestrate local extraction, enrichment, validation and output generation for one input PDF.
def extract_publication(
    pdf_path,
    schema_path=None,
    output_dir=None,
    return_analysis=False,
):
    """Extract any supplied PDF from its text and layout; no model is called."""

    # Resolve the input, schema and destination, and validate the schema itself.
    pdf_path = Path(pdf_path)
    schema_path = Path(schema_path or SCHEMA_PATH)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    output = Path(
        output_dir
        or ROOT / "outputs/publication_structures/layout_first" / pdf_path.stem
    )
    ctx = ExtractionContext(schema, output)

    # Construct the text/layout interpretation.
    discover_assets(pdf_path, ctx)
    initial, pages = first_pass(pdf_path, ctx)
    tree = json.loads(json.dumps(initial))
    relations = interpret_content(tree, pages, ctx)

    # Enrich references using literal citation fields and upgrade the structural schema version.
    citation_report = enrich_citations(tree, ctx)
    tree["schema_version"] = "1.1"
    # Acknowledgment paragraphs can be split into short trailing lines by prose rules.
    for acknowledgement in walk(tree["document"]):
        if acknowledgement["type"] != "acknowledgements":
            continue
        children = acknowledgement["children"]
        if children and all(c["type"] in ("paragraph", "custom") for c in children):
            joined = ctx.make(
                "paragraph",
                children[0]["page_start"],
                "\n".join(c.get("text", "") for c in children),
                observation_ids=[
                    i
                    for c in children
                    for i in c.get("attributes", {}).get("observation_ids", [])
                ],
                decision={
                    "rule": "Preserve acknowledgment content as one paragraph",
                    "review_required": True,
                    "verified": False,
                },
            )
            joined["page_end"] = children[-1]["page_end"]
            # Point figure/table mentions from the merged paragraphs at the joined one.
            merged_ids = {c["id"] for c in children}
            for r in relations:
                if r["from"] in merged_ids:
                    r["from"] = joined["id"]
            acknowledgement["children"] = [joined]

    # Build a coverage ledger: every observed text block should have exactly one primary owner.
    ids = index(tree)
    observed = {l["id"] for p in pages for l in p["lines"]}
    owners = {block: [] for block in observed}
    for n in walk(tree["document"]):
        for block in n.get("attributes", {}).get("observation_ids", []):
            if block not in owners:
                raise ValueError("Unknown source block: " + block)
            owners[block].append(n["id"])
    missing = [block for block, nodes in owners.items() if not nodes]
    duplicates = {block: nodes for block, nodes in owners.items() if len(nodes) > 1}
    for r in relations:
        assert r["from"] in ids and (
            r["to"] is None or r["to"] in ids
        ), "Dangling relation"
    unresolved = [
        n["id"]
        for n in ids.values()
        if n.get("attributes", {}).get("custom_type") == "unidentified"
    ]
    profile = learn_style_profile(pages, tree["extraction"]["body_font_size"])

    # Summarize validation, unresolved material, evidence and source fingerprints.
    report = {
        "source_sha256": tree["source"]["sha256"],
        "schema_sha256": ctx.schema_hash,
        "validation": validate_structure(tree, schema),
        "coverage_complete": not missing and not duplicates,
        "unowned_blocks": missing,
        "multiply_owned_blocks": duplicates,
        "observed_blocks": len(observed),
        "unidentified_nodes": unresolved,
        "type_counts": dict(Counter(n["type"] for n in ids.values())),
        "layout": [p["layout"] for p in pages],
        "independently_verified_correct": False,
        "reference_numbers": [
            n["number"] for n in ids.values() if n["type"] == "reference"
        ],
        "citations": citation_report,
        "association_policy": "Explicit references propose a parent; ambiguity stays unresolved; all relations require review.",
    }
    tree["extraction"].update(
        status="partial_requires_review",
        relations=relations,
        review_required=True,
        limitations=[
            "Local layout and structural classifications are proposals requiring source review.",
            "Text fidelity, merged cells, figure semantics, equation/list structure and scanned-page OCR may remain unresolved.",
            "Figure/table parent associations are provisional, not scientific claim support.",
        ],
    )
    validate_structure(tree, schema)
    ledger = [
        {k: v for k, v in l.items() if not k.startswith("_")}
        for p in pages
        for l in p["lines"]
    ]
    for name, value in [
        ("structure.json", tree),
        ("initial_structure.json", initial),
        ("audit.json", report),
        ("style_profile.json", profile),
        ("observations.json", ledger),
    ]:
        (output / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if not report["coverage_complete"]:
        raise ValueError(
            "Coverage failed; inspect audit.json; extraction is not complete."
        )
    if return_analysis:
        return tree, report, profile, pages, ctx
    return tree, report, profile


# ---------------------------------------------------------------------------
# Running one paper or many
# ---------------------------------------------------------------------------


# Load a saved JSON artifact without changing its contents.
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


REQUIRED = {"id", "page", "reading_order", "text"}


# Check every observation carries the fields Notebook B reads.
def check_observations(path):
    observations = read(path)
    if not observations:
        raise ValueError(f"{path} is empty")
    for o in observations:
        missing = REQUIRED - o.keys()
        if missing:
            raise ValueError(f"Observation {o.get('id')} is missing {missing}")
    return len(observations)


# Run the text pass for one paper, reusing the saved result unless forced.
def run_text_pass(pdf_name, force=False):
    config = configure(pdf_name)
    out = config["output"] / "text_pass"
    if (out / "observations.json").exists() and not force:
        return {"paper": config["pdf"].stem, "status": "cached"}
    tree, report, profile = extract_publication(config["pdf"], output_dir=out)
    return {
        "paper": config["pdf"].stem,
        "status": "ok",
        "lines": check_observations(out / "observations.json"),
        "unidentified": len(report["unidentified_nodes"]),
    }


# Run every PDF in the publications folder; one failure does not stop the rest.
def run_all(force=False):
    results = []
    for pdf in sorted(PUBLICATIONS.glob("*.pdf")):
        try:
            results.append(run_text_pass(pdf.name, force=force))
        except Exception as error:
            results.append(
                {
                    "paper": pdf.stem,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        print(results[-1])
    return results
