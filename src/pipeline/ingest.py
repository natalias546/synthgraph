"""Stage 1 — PDF ingestion: lines with typography, reading order, superscripts, ruled tables.

Design taken from the publication-structure notebook: work from characters, infer
two-column vs. spanning layout per page, keep geometry private, and never drop
content silently. Superscripts are kept as ^{...} instead of deleted, because in
materials papers they are often exponents (cm^{-3}, 10^{15}) rather than citations.
"""
from __future__ import annotations

import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber

from .io_utils import sha256_file


def _body_font_size(pdf) -> float:
    sizes = Counter()
    for page in pdf.pages:
        for c in page.chars:
            if c.get("text", "").strip():
                sizes[round(c.get("size", 0), 1)] += 1
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _line_text(chars: list[dict]) -> tuple[str, dict]:
    """Rebuild text from characters, inserting spaces by gap and marking superscripts."""
    chars = sorted((c for c in chars if c.get("text", "") != ""), key=lambda c: c["x0"])
    visible = [c for c in chars if c["text"].strip()]
    if not visible:
        return "", {}
    med_size = statistics.median(c["size"] for c in visible)
    normal = [c for c in visible if c["size"] >= 0.9 * med_size] or visible
    baseline = statistics.median(c["bottom"] for c in normal)

    def is_super(c):
        return c["size"] < 0.85 * med_size and c["bottom"] < baseline - 0.2 * med_size

    out, prev, in_sup = [], None, False
    for c in chars:
        t = c["text"]
        if not t.strip():
            if out and out[-1] != " ":
                out.append(" ")
            prev = c
            continue
        if prev is not None and c["x0"] - prev["x1"] > 0.22 * max(c["size"], 1) and out and out[-1] != " ":
            if in_sup:
                out.append("}")
                in_sup = False
            out.append(" ")
        sup = is_super(c)
        if sup and not in_sup:
            out.append("^{")
            in_sup = True
        elif not sup and in_sup:
            out.append("}")
            in_sup = False
        out.append(t)
        prev = c
    if in_sup:
        out.append("}")
    text = "".join(out).replace(" }", "} ").strip()
    fonts = Counter(c.get("fontname", "").split("+")[-1] for c in visible)
    style = {
        "font_size": round(med_size, 1),
        "font": fonts.most_common(1)[0][0],
        "bold_ratio": round(sum("bold" in c.get("fontname", "").lower() for c in visible) / len(visible), 2),
        "italic_ratio": round(
            sum(any(k in c.get("fontname", "").lower() for k in ("italic", "oblique")) for c in visible) / len(visible), 2
        ),
        "x0": round(min(c["x0"] for c in visible), 1),
        "x1": round(max(c["x1"] for c in visible), 1),
        "top": round(min(c["top"] for c in visible), 1),
        "bottom": round(max(c["bottom"] for c in visible), 1),
    }
    return text, style


def _group_lines(chars: list[dict], tol: float = 1.2) -> list[list[dict]]:
    """Group characters into lines.

    Phase 1 groups by baseline (char bottom). Phase 2 attaches short runs of smaller
    characters (super/subscripts such as cm^-3 or Al2O3) to the line they sit inside,
    choosing the closest line, so an exponent is never glued to the line above.
    """
    rows: list[list[dict]] = []
    for c in sorted(chars, key=lambda c: c["bottom"]):
        if rows and abs(rows[-1][-1]["bottom"] - c["bottom"]) <= tol:
            rows[-1].append(c)
        else:
            rows.append([c])
    # Split a baseline row at large horizontal gaps (column gutter, table cells), so text
    # from two columns that happens to share a baseline stays in separate lines.
    lines: list[list[dict]] = []
    for row in rows:
        row.sort(key=lambda c: c["x0"])
        cur = [row[0]]
        for c in row[1:]:
            vis = [x for x in cur if x.get("text", "").strip()] or cur
            size = statistics.median(x["size"] for x in vis)
            if c["x0"] - max(x["x1"] for x in cur) > 1.5 * size:
                lines.append(cur)
                cur = [c]
            else:
                cur.append(c)
        lines.append(cur)

    def info(ln):
        vis = [c for c in ln if c.get("text", "").strip()] or ln
        size = statistics.median(c["size"] for c in vis)
        return {"size": size, "top": min(c["top"] for c in vis), "bottom": max(c["bottom"] for c in vis),
                "x0": min(c["x0"] for c in vis), "x1": max(c["x1"] for c in vis), "n": len(vis)}

    infos = [info(ln) for ln in lines]
    hosts = [i for i, inf in enumerate(infos)]
    merged = set()
    for i, inf in enumerate(infos):
        # Candidate fragment: few characters, every one smaller than the surrounding text.
        best, best_d = None, None
        for j in hosts:
            if j == i or j in merged:
                continue
            h = infos[j]
            if not (inf["size"] < 0.9 * h["size"] and h["n"] > inf["n"]):
                continue
            centre = (inf["top"] + inf["bottom"]) / 2
            if not (h["top"] - 0.3 * h["size"] <= centre <= h["bottom"] + 0.2 * h["size"]):
                continue
            if inf["x1"] < h["x0"] - 5 or inf["x0"] > h["x1"] + 5:
                continue
            # Fragments are short pieces interleaved with the host's characters.
            if inf["n"] > 12 and not (h["x0"] <= inf["x0"] and inf["x1"] <= h["x1"]):
                continue
            d = abs(centre - (h["top"] + h["bottom"]) / 2)
            if best_d is None or d < best_d:
                best, best_d = j, d
        if best is not None:
            lines[best].extend(lines[i])
            merged.add(i)
    return [ln for i, ln in enumerate(lines) if i not in merged]


def _find_gutter(page, body_size: float) -> float | None:
    """Return the x position of a two-column gutter, or None for single-column pages.

    Uses line pieces (lines are already split at large horizontal gaps): in a two-column
    page most pieces lie entirely left or right of the gutter and only full-width lines
    (title, abstract, wide tables) cross it; in a single-column page most lines cross.
    Works even when the two columns' baselines are not aligned.
    """
    chars = [c for c in page.chars if c["text"].strip() and c["size"] >= body_size * 0.8]
    if len(chars) < 150:
        return None
    pieces = [(min(c["x0"] for c in ln), max(c["x1"] for c in ln)) for ln in _group_lines(chars) if len(ln) >= 3]
    w = page.width
    best = None
    for i in range(30, 71):
        x = w * i / 100
        left = sum(1 for a, b in pieces if b <= x)
        right = sum(1 for a, b in pieces if a >= x)
        cross = len(pieces) - left - right
        key = (cross, abs(x - w / 2))
        if best is None or key < best[0]:
            best = (key, x, left, right, cross)
    _, x, left, right, cross = best
    if left >= 3 and right >= 3 and cross <= 0.35 * len(pieces):
        return x
    return None


def _inside(c, bbox, pad=1.0):
    x0, top, x1, bottom = bbox
    return x0 - pad <= (c["x0"] + c["x1"]) / 2 <= x1 + pad and top - pad <= (c["top"] + c["bottom"]) / 2 <= bottom + pad


def ingest_pdf(pdf_path: Path) -> dict:
    """Return {source, body_font_size, pages:[{page, layout, blocks:[line|table]}]}."""
    pdf_path = Path(pdf_path)
    result = {
        "source": {"pdf": pdf_path.name, "sha256": sha256_file(pdf_path)},
        "pages": [],
        "warnings": [],
    }
    with pdfplumber.open(pdf_path) as pdf:
        body = _body_font_size(pdf)
        result["body_font_size"] = body
        result["source"]["page_count"] = len(pdf.pages)
        for pno, page in enumerate(pdf.pages, 1):
            page = page.dedupe_chars(tolerance=1)
            # Ruled tables: extract cells, then remove their characters from running text.
            tables = []
            try:
                for t_index, t in enumerate(page.find_tables()):
                    rows = [[(cell or "").strip() for cell in row] for row in t.extract()]
                    if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                        tables.append({"bbox": t.bbox, "rows": rows, "index": t_index})
            except Exception as exc:  # never fail ingestion because of a table
                result["warnings"].append(f"page {pno}: table detection failed ({type(exc).__name__})")
            chars = [c for c in page.chars if not any(_inside(c, t["bbox"]) for t in tables)]
            chars = [dict(c, text=unicodedata.normalize("NFC", c.get("text", ""))) for c in chars]

            gutter = _find_gutter(page, body)
            raw_lines = _group_lines(chars)
            # Classify each line as left/right column or spanning (two-column pages).
            records = []
            for line in raw_lines:
                if gutter is not None:
                    x0 = min(c["x0"] for c in line)
                    x1 = max(c["x1"] for c in line)
                    if x0 < gutter - 2 and x1 > gutter + 2:
                        records.append(("span", line))
                    else:
                        records.append(("left" if (x0 + x1) / 2 < gutter else "right", line))
                else:
                    records.append(("single", line))
            # Reading order: spanning lines cut the page into bands; left column before right in each band.
            records.sort(key=lambda r: min(c["top"] for c in r[1]))
            ordered, pending = [], {"left": [], "right": []}

            def flush():
                ordered.extend(pending["left"])
                ordered.extend(pending["right"])
                pending["left"], pending["right"] = [], []

            for flow, line in records:
                if flow in ("left", "right"):
                    pending[flow].append((flow, line))
                else:
                    flush()
                    ordered.append((flow, line))
            flush()

            blocks = []
            for flow, line in ordered:
                text, style = _line_text(line)
                if text:
                    blocks.append({"kind": "line", "text": text, "flow": flow, **style})
            # Insert tables at their vertical position (approximate reading order).
            for t in tables:
                pos = next((i for i, b in enumerate(blocks) if b["top"] > t["bbox"][1]), len(blocks))
                blocks.insert(pos, {"kind": "table", "rows": t["rows"], "top": round(t["bbox"][1], 1),
                                    "bbox": [round(v, 1) for v in t["bbox"]], "flow": "table"})
            result["pages"].append({
                "page": pno,
                "layout": "two_column" if gutter is not None else "single_or_uncertain",
                "width": page.width, "height": page.height,
                "blocks": blocks,
            })
            if not blocks:
                result["warnings"].append(f"page {pno}: no extractable text (scanned page? needs OCR)")
    return result
