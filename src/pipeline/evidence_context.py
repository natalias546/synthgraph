"""Deterministic, reversible sentence evidence segmentation for structure.json.

No ontology or model dependencies. Original paragraph text is never rewritten;
offsets are half-open offsets into the source paragraph string.
"""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"0-9])")
_SYMBOL = re.compile(r"\b(?:RRR|MR|AHE|WAL|DFT|ARPES|XRD|CVT|CVD)\b")
_NUM_UNIT = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?\s*(?:K|T|eV|meV|nm|GPa|MPa|\u03a9|%|cm\b|\u00c5\b)")
_RESULTS = re.compile(r"\b(?:increase[ds]?|decrease[ds]?|observed|measured|found|shows?|exhibits?|obtained|reported)\b", re.I)
_METHODS = re.compile(r"\b(?:grown|synthesi[sz]ed|annealed|deposited|fabricated|prepared|calculated|performed|using|measured)\b", re.I)
_HYPOTHESIS = re.compile(r"\b(?:may|might|suggests?|possibly|could|hypothes[ie]s|attribut(?:ed|able))\b", re.I)
_CITATION = re.compile(r"\b(?:previous(?:ly)?|prior work|literature|ref\.|et al\.)\b", re.I)


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Preserve byte-exact source strings (Python character offsets), incl. punctuation."""
    if not text or not text.strip():
        return []
    boundaries = [0]
    for match in _SENTENCE_END.finditer(text):
        preceding = text[boundaries[-1]:match.start()].rstrip()
        if re.search(r"(?:\b(?:Fig|Figs|Ref|Refs|Eq|Eqs|Dr|Prof|vs|al|et al)\.|\bet al\.)$", preceding, re.I):
            continue
        boundaries.append(match.end())
    boundaries.append(len(text))
    out = []
    for a, b in zip(boundaries, boundaries[1:]):
        segment = text[a:b]
        left = len(segment) - len(segment.lstrip())
        right = len(segment.rstrip())
        if right > left:
            start, end = a + left, a + right
            out.append((start, end, text[start:end]))
    return out


def semantic_roles(sentence: str, paragraph_role: str) -> list[str]:
    roles = []
    for pattern, label in ((_NUM_UNIT, "quantitative_evidence"), (_RESULTS, "result"),
                           (_METHODS, "method"), (_HYPOTHESIS, "qualified_claim"),
                           (_CITATION, "literature_comparison")):
        if pattern.search(sentence):
            roles.append(label)
    if paragraph_role in ("methods", "results", "conclusion", "introduction"):
        roles.append("section_" + paragraph_role)
    return list(dict.fromkeys(roles))


def enrich_structure(structure: dict) -> dict:
    """Mutate only by adding semantic_evidence and per-paragraph evidence IDs."""
    evidence = []
    paragraphs = structure.get("paragraphs", [])
    for p in paragraphs:
        pid = p["id"]
        spans = split_sentences(p.get("text", ""))
        p["evidence_ids"] = [f"{pid}:s{i}" for i in range(1, len(spans) + 1)]
        for i, (start, end, sentence) in enumerate(spans):
            evidence.append({
                "id": p["evidence_ids"][i], "paragraph_id": pid,
                "section_id": p.get("section_id"), "section_role": p.get("role"),
                "source_kind": p.get("kind"), "page": p.get("page"),
                "span_start": start, "span_end": end, "sentence": sentence,
                "preceding_sentence": spans[i-1][2] if i else None,
                "following_sentence": spans[i+1][2] if i + 1 < len(spans) else None,
                "scientific_roles": semantic_roles(sentence, p.get("role", "")),
                "candidate_terms": list(dict.fromkeys(m.group(0) for m in _SYMBOL.finditer(sentence))),
                "value_mentions": [m.group(0) for m in _NUM_UNIT.finditer(sentence)],
                "review_status": "unreviewed", "verified_semantic_truth": False,
            })
    # Neighboring paragraphs in the same section provide context without changing
    # the source text or merging publication provenance across sections.
    by_para = {p["id"]: [] for p in paragraphs}
    for ev in evidence:
        by_para[ev["paragraph_id"]].append(ev)
    for index, paragraph in enumerate(paragraphs):
        current = by_para[paragraph["id"]]
        if not current:
            continue
        if index > 0 and paragraphs[index - 1].get("section_id") == paragraph.get("section_id"):
            previous = by_para[paragraphs[index - 1]["id"]]
            if previous:
                if current[0]["preceding_sentence"] is None:
                    current[0]["preceding_sentence"] = previous[-1]["sentence"]
                current[0]["preceding_source_id"] = previous[-1]["id"]
        if index + 1 < len(paragraphs) and paragraphs[index + 1].get("section_id") == paragraph.get("section_id"):
            following = by_para[paragraphs[index + 1]["id"]]
            if following:
                if current[-1]["following_sentence"] is None:
                    current[-1]["following_sentence"] = following[0]["sentence"]
                current[-1]["following_source_id"] = following[0]["id"]
    structure["semantic_evidence"] = evidence
    structure["semantic_evidence_version"] = "1.0"
    return structure
