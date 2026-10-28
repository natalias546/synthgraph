"""Text normalization (adapted from the Murari notebook's clean(), tuned for materials text).

Two different normalizations are used on purpose:
  * clean_text(): what the LLM sees and what is stored (light-touch, reversible in spirit)
  * match_key():  aggressive normalization used ONLY to check that quotes exist
"""
from __future__ import annotations

import re
import unicodedata

# Hyphenated prefixes that must keep their hyphen when split across a line break.
# (Edit this list for your corpus; e.g. "as-deposited", "post-annealed", "self-assembled".)
PRESERVE_HYPHEN_PREFIXES = {
    "as", "post", "non", "self", "high", "low", "single", "room", "well", "solution", "spin",
    "state", "near", "wide", "grazing", "plasma", "laser", "ion", "front", "back", "rear",
}
# A right-hand fragment like "ly", "ing", "pared" means the hyphen was ordinary word hyphenation.
_SUFFIX = re.compile(r"^(ly|ing|ed|er|est|ness|tion|sion|ment|ple|pared|sent|ty|ity|ive|al|ous|ize|ise|ure|ence|ance)\b")

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
DASHES = {"−": "-", "‐": "-", "‑": "-", "­": ""}  # minus, hyphens, soft hyphen


def clean_text(text: str) -> str:
    for k, v in {**LIGATURES, **DASHES}.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def join_lines(lines: list[str]) -> str:
    """Join PDF lines into one paragraph, repairing line-break hyphenation conservatively."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        m = re.search(r"([A-Za-z0-9]+)-$", out)
        if m and re.match(r"^[a-z]", line):
            left = m.group(1)
            keep = (
                len(left) == 1                                   # n-type, p-doped, x-ray
                or any(ch.isdigit() for ch in left)              # Si3N4-based, 1550-nm
                or any(ch.isupper() for ch in left[1:])          # SiO2-, PECVD-grown (not "Sili-con")
                or (left.lower() in PRESERVE_HYPHEN_PREFIXES and not _SUFFIX.match(line))
            )
            out = out + line if keep else out[:-1] + line
        else:
            out = out + " " + line
    return clean_text(out)


def match_key(text: str) -> str:
    """Aggressive normalization for quote anchoring only (never stored as content)."""
    text = clean_text(text)
    text = unicodedata.normalize("NFKC", text)  # sub/superscript digits -> digits, µ -> μ
    text = text.replace("^{", "").replace("}", "")
    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"[‘’‚‛′]", "'", text)
    text = re.sub(r"[“”„″]", '"', text)
    text = text.replace("μ", "u").replace("°", " deg ").replace("Å", "A")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([-/=(),;:])\s*", r"\1", text)
    return text.strip().casefold()


_WORD_END = re.compile(r"^[a-z]{0,4}(tion|sion|ity|ing|ed|ly|ment|ness|ance|ence|ous|ive|able|ible|ical|ure|ism|ist|"
                       r"ize|ise|ated|ating|ation|ductor|ductivity|ductance|ent|ant|ary|ory|al|er|ic)s?$")


def dehyphenate(text: str) -> str:
    """Repair line-break hyphenation left inside one text block ("depo- sited" -> "deposited"),
    using the same keep/drop rules as join_lines ("n- type" -> "n-type")."""
    def fix(m):
        left, right = m.group(1), m.group(2)
        keep = (len(left) == 1 or any(ch.isdigit() for ch in left) or any(ch.isupper() for ch in left[1:])
                or (left.lower() in PRESERVE_HYPHEN_PREFIXES and not _SUFFIX.match(right))
                # two long fragments, right one not a word ending: a real compound ("temperature-dependent")
                or (len(left) >= 5 and len(right) >= 5 and not _WORD_END.match(right)))
        return f"{left}-{right}" if keep else f"{left}{right}"
    return re.sub(r"\b([A-Za-z0-9]+)- ([a-z][A-Za-z]*)", fix, text)
