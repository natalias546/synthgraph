"""Cheap, local text tools from the Murari notebook, used for two jobs:

1. Ontology development: mine frequent technical phrases (RAKE-style) and list the
   ones that match no class label or synonym -> candidate classes to review.
2. Retrieval: BM25 search over paragraphs of all processed papers (handy for checking
   coverage, and later for the chatbot's retrieval step).
No API calls here.
"""
from __future__ import annotations

import re
from collections import Counter

STOP = set("""a an the and or of in on at to for from by with without as is are was were be been being this that these
those it its we our their which who whom whose than then there here into onto over under between among within via per
also both each such using used use based after before during while when where how what all any some more most less
very can could may might should would will shall not no only same other another one two three fig figure table et al
respectively however thus therefore hence shown show shows obtained performed measured carried out found observed""".split())


def rake_phrases(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-:+]*|[.,;:()\[\]]", text)
    phrases, cur = [], []
    for w in words:
        if w.lower() in STOP or not re.match(r"[A-Za-z]", w):
            if cur:
                phrases.append(" ".join(cur))
            cur = []
        else:
            cur.append(w)
    if cur:
        phrases.append(" ".join(cur))
    return [p for p in phrases if 1 <= len(p.split()) <= 5 and len(p) > 2]


def ontology_gap_terms(structures: dict, ontology, roles=("methods", "results", "abstract"), top=60):
    """Frequent phrases (across papers) that match no ontology label or synonym."""
    known = set()
    for c in ontology.classes.values():
        known.add(c.label.casefold())
        known.update(s.casefold() for s in c.synonyms)
    counts, papers = Counter(), {}
    for pid, st in structures.items():
        for p in st["paragraphs"]:
            if p["role"] in roles and p["kind"] in ("text", "caption"):
                for ph in rake_phrases(p["text"]):
                    k = ph.casefold()
                    counts[k] += 1
                    papers.setdefault(k, set()).add(pid)
    rows = []
    for k, c in counts.most_common():
        if c < 2 or len(papers[k]) < 2 and len(structures) > 2:
            continue
        if k in known or any(k == kn or k.endswith(" " + kn) and len(k.split()) <= 2 for kn in known):
            continue
        rows.append({"phrase": k, "count": c, "papers": len(papers[k])})
        if len(rows) >= top:
            break
    return rows


class ParagraphSearch:
    def __init__(self, structures: dict):
        from rank_bm25 import BM25Okapi
        self.rows = [(pid, p) for pid, st in structures.items() for p in st["paragraphs"]]
        tok = [self._tok(p["text"]) for _, p in self.rows]
        self.bm25 = BM25Okapi(tok) if tok else None

    @staticmethod
    def _tok(t):
        return [w for w in re.findall(r"[a-z0-9][a-z0-9\-:+.]*", t.lower()) if w not in STOP]

    def search(self, query: str, k: int = 10):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(self._tok(query))
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{"paper_id": self.rows[i][0], "paragraph_id": self.rows[i][1]["id"], "role": self.rows[i][1]["role"],
                 "page": self.rows[i][1]["page"], "score": round(float(scores[i]), 3),
                 "text": self.rows[i][1]["text"][:300]} for i in best if scores[i] > 0]
