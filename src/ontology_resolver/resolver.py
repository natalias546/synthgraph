"""Ontology resolver: maps what the LLM extracted (in the paper's own words) onto the v11 ontology.

Extraction and ontology mapping are separate jobs. Passes 1-3 describe each mention in the paper's
terms (name, a generic `type_term`, a short description, evidence) and never see the ontology. This
module then assigns classes in tiers, cheapest and safest first:

  1. deterministic  unique exact active label / approved alias (scientific_aliases.json) -> exact_match
                    confirmed gaps from review -> missing_class without any model call
  2. memory         a decision already made for the same (group, term) in an earlier paper is reused
                    (data/resolution/resolution_memory.json), so each term is decided once per corpus
  3. retrieval      local candidate search inside the group's slice: labels, synonyms and definitions,
                    plus the parents of the best hits (no model, no network)
  4. adjudication   optional small LLM call (batched, default claude-haiku-5-5) that picks one candidate
                    and says HOW it fits: same_concept | instance_of | narrower_than, or none (= ontology gap)

Policy (unchanged from v1 in spirit): a fuzzy or lexical resemblance is never an automatic equivalence.
Only exact labels / approved aliases are accepted without review; adjudicated mappings are accepted
when confident, except a *concept* typed to a broader class (e.g. "anomalous Hall resistivity" ->
"resistivity"), which stays reviewable and is reported as a narrower-concept ontology gap. Nothing here
edits the OWL file or invents an IRI. Human review (resolution_review.csv) feeds back as approved
aliases / rejections / confirmed gaps, which then apply to every paper.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from ..ontology_loader import OTHER, Profile

RESOLVER_VERSION = "2.0.0"
ALIAS_PATH = Path(__file__).with_name("scientific_aliases.json")

# Groups whose mentions are specific individuals (a sample, a compound, a process run, a parameter
# value, an instrument...). For these, "instance of class X" is the correct typing. For concept groups
# (properties, techniques, phenomena, applications...) a broader class is not an equivalent concept.
INSTANCE_GROUPS = {"material", "sample", "process", "process_attribute",
                   "instrument", "software", "reference", "dataset"}
# When a group's own slice lacks a basic concept, these groups are searched too. Process parameters are
# physical quantities: v11 has plain "temperature" and "pressure" only under the property group.
FALLBACK_GROUPS = {"process_attribute": ("property",)}
CONTEXT_GROUPS = ("phenomenon", "mechanism", "theory_model", "instrument", "software",
                  "phase", "structure", "reference", "dataset", "other", "unmapped")


def normalize(text: str | None) -> str:
    """Conservative normalization; preserve chemical formulas and subscripts."""
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = re.sub(r"[‐‑‒–—−_-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9α-ω]+", normalize(s)))


def _words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9α-ω]+", normalize(s)) if len(w) > 1 and w not in _STOP]


_STOP = set("""a an the and or of in on at to for from by with as is are was were be this that these those it its
we our their which using used use based process measurement material property analysing analyzing analysis
method technique application mention""".split())


def _grams(s: str, n: int = 3) -> Counter:
    s = f" {normalize(s)} "
    return Counter(s[i:i + n] for i in range(max(1, len(s) - n + 1)))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


@dataclass(frozen=True)
class Entry:
    group: str
    label: str
    iri: str
    synonyms: tuple[str, ...]
    definition: str


# ---------------------------------------------------------------------------
# Adjudication output contract (structured output for the small LLM call)
# ---------------------------------------------------------------------------
class MappingDecision(BaseModel):
    mention_id: str = Field(description="The mention_id given in the task, e.g. m3")
    choice: str = Field(description="Key of the chosen candidate (e.g. C2), or 'none'")
    match_type: Literal["same_concept", "instance_of", "narrower_than", "none"] = Field(
        description="same_concept: the mention means exactly this class (synonym, abbreviation, spelling); "
                    "instance_of: the mention is a specific thing (sample, compound, process run, instrument) "
                    "whose kind is this class; narrower_than: the mention is a more specific concept than this "
                    "class; none: no candidate is correct or correctly broader")
    confidence: float = Field(description="0-1: how sure you are that this mapping is correct")
    context_dependent: bool = Field(description="True if the same words could map differently in another paper")
    reason: str = Field(description="One short sentence")


class MappingDecisions(BaseModel):
    decisions: list[MappingDecision]


RESOLVER_SYSTEM = """You map terms extracted from materials-science papers onto classes of an OWL ontology.
For each mention you get: its group (the kind of thing), the term(s) as the paper uses them, a short
description and context sentence, and a numbered list of CANDIDATE classes with their definitions.

Choose at most one candidate per mention and say how it fits:
- same_concept: the mention means exactly the class (a synonym, abbreviation or spelling variant).
- instance_of: the mention is one specific thing (this paper's sample, compound, process run, parameter,
  instrument) and the class is its kind. Pick the MOST SPECIFIC candidate that is still correct.
- narrower_than: the mention is a more specific CONCEPT than the class (e.g. "anomalous Hall resistivity"
  vs "resistivity"). Pick the most specific candidate that correctly contains it.
- none: no candidate is correct or correctly broader.

Rules: judge by the definitions, not by word overlap. A related or sibling concept is NOT a match
(magnetoresistance is not resistivity; a measurement technique is not the property it measures; an
application is not a material). Never choose a class just because it is the only candidate. If unsure,
answer none: a missing mapping is reviewed and fixed later, a wrong mapping silently corrupts the graph.
Confidence is your honest probability that the mapping is right. Mark context_dependent when the same
words could mean something else in another paper (acronyms like MR, PL, CDW...)."""


# ---------------------------------------------------------------------------
# Candidate retrieval (local, deterministic)
# ---------------------------------------------------------------------------
class CandidateIndex:
    """Lexical retrieval over one group's classes: character trigrams on labels/synonyms (morphology,
    spelling) plus IDF-weighted word coverage over labels, synonyms and definitions (meaning)."""

    def __init__(self, entries: list[Entry], parents: dict[str, list[str]]):
        self.entries = entries
        self.parents = parents
        self.name_grams = [[_grams(n) for n in (e.label, *e.synonyms) if n] for e in entries]
        self.name_words = [set(_words(" ".join((e.label, *e.synonyms)))) for e in entries]
        self.def_words = [set(_words(e.definition)) for e in entries]
        df = Counter(w for i in range(len(entries)) for w in self.name_words[i] | self.def_words[i])
        n = max(1, len(entries))
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}

    def search(self, terms: list[str], context: str = "", k: int = 8,
               exclude: set[str] = frozenset()) -> list[tuple[float, Entry]]:
        terms = [t for t in terms if t and t.strip()]
        if not terms or not self.entries:
            return []
        tg = [_grams(t) for t in terms]
        qw = set(_words(" ".join(terms)))
        cw = set(_words(context)) - qw
        qidf = sum(self.idf.get(w, 1.0) for w in qw) or 1.0
        scored = []
        for i, e in enumerate(self.entries):
            if e.iri in exclude:
                continue
            name = max((_cosine(a, b) for a in tg for b in self.name_grams[i]), default=0.0)
            in_name = sum(self.idf.get(w, 1.0) for w in qw & self.name_words[i]) / qidf
            in_def = sum(self.idf.get(w, 1.0) for w in qw & self.def_words[i]) / qidf
            ctx = 0.05 * len(cw & (self.name_words[i] | self.def_words[i])) / (len(cw) or 1)
            score = 0.55 * name + 0.25 * in_name + 0.15 * in_def + ctx
            if score > 0.12:
                scored.append((round(score, 4), e))
        scored.sort(key=lambda s: (-s[0], s[1].iri))
        return scored[:k]


# ---------------------------------------------------------------------------
# Decision memory (shared across papers)
# ---------------------------------------------------------------------------
class ResolutionMemory:
    """(group, normalized term) -> adjudicated decision, reused for every later paper.
    Entries are tied to the ontology's class signature: a new ontology version re-decides them."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.entries: dict[str, dict] = {}
        self._dirty = False
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = data.get("entries", {})

    @staticmethod
    def key(group: str, term: str, paper_id: str | None = None) -> str:
        return f"{group}|{normalize(term)}" + (f"|{paper_id}" if paper_id else "")

    def get(self, key: str, class_signature: str) -> dict | None:
        e = self.entries.get(key)
        return e if e and e.get("class_signature") == class_signature else None

    def put(self, key: str, decision: dict, paper_id: str) -> None:
        old = self.entries.get(key)
        papers = sorted(set((old or {}).get("papers", [])) | {paper_id})
        self.entries[key] = {**decision, "papers": papers}
        self._dirty = True

    def note_use(self, key: str, paper_id: str) -> None:
        e = self.entries.get(key)
        if e is not None and paper_id not in e.get("papers", []):
            e["papers"] = sorted(set(e.get("papers", [])) | {paper_id})
            self._dirty = True

    def forget(self, key: str) -> None:
        if self.entries.pop(key, None) is not None:
            self._dirty = True

    def save(self) -> None:
        if self.path and self._dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": 1, "entries": dict(sorted(self.entries.items()))},
                                      indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
            self._dirty = False


# ---------------------------------------------------------------------------
# Which extracted objects are resolved, and where the result goes
# ---------------------------------------------------------------------------
@dataclass
class Mention:
    record_id: str
    role: str
    group: str
    terms: list[str]                   # in priority order; terms[0] keys the memory
    description: str
    context: str
    apply: Callable[[dict], None]      # writes the resolution into the extracted object
    candidates: list = field(default_factory=list)
    resolution: dict | None = None


def _first(*xs) -> str:
    return next((str(x).strip() for x in xs if isinstance(x, str) and x.strip() and x != OTHER), "")


def _quotes(obj: dict, n: int = 2) -> str:
    return " … ".join(e.get("quote", "") for e in (obj.get("evidence") or [])[:n] if e.get("quote"))


class OntologyResolver:
    """Index the active OWL slice once, then resolve extracted mentions (see module docstring)."""

    def __init__(self, profile: Profile, alias_path: Path | None = None):
        self.profile = profile
        self.alias_path = Path(alias_path) if alias_path is not None else ALIAS_PATH
        self.alias_data = json.loads(self.alias_path.read_text(encoding="utf-8"))
        self.entries: dict[str, list[Entry]] = defaultdict(list)
        self.lookup: dict[tuple[str, str], list[tuple[Entry, str]]] = defaultdict(list)
        self.by_iri: dict[tuple[str, str], Entry] = {}
        self.by_label: dict[tuple[str, str], Entry] = {}
        for group, spec in profile.groups.items():
            for label, iri in spec.label_to_iri.items():
                c = profile.ontology.classes[iri]
                if c.deprecated:
                    continue
                e = Entry(group, label, iri, tuple(c.synonyms), c.definition or "")
                self.entries[group].append(e)
                self.by_iri[group, iri] = e
                self.by_label[group, normalize(label)] = e
                # A synonym is only a retrieval candidate: the source may mark it as RELATED.
                for name, method in [(label, "exact_label"), *[(s, "synonym") for s in c.synonyms]]:
                    key = normalize(name)
                    if key:
                        self.lookup[group, key].append((e, method))
                local = iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                if local and normalize(local) != normalize(label):
                    self.lookup[group, normalize(local)].append((e, "iri_local_name"))
            self.entries[group].sort(key=lambda e: (e.label.casefold(), e.iri))
        self.aliases: dict[tuple[str, str], list[Entry]] = defaultdict(list)
        for a in self.alias_data.get("aliases", []):
            if not a.get("approved", False):
                continue
            group, alias = a["group"], normalize(a["alias"])
            target = a.get("target_iri")
            e = self.entry(group, target)
            if e is None:
                raise ValueError(f"Approved ontology alias {alias!r} targets missing/inactive IRI {target!r}")
            self.aliases[group, alias].append(e)
        self.rejected: dict[tuple[str, str], set[str]] = defaultdict(set)
        for r in self.alias_data.get("rejected", []):
            self.rejected[r["group"], normalize(r["term"])].add(r["iri"])
        self.confirmed_missing = {(m["group"], normalize(m["term"])) for m in self.alias_data.get("confirmed_missing", [])}
        signature = [{"group": e.group, "label": e.label, "iri": e.iri,
                      "synonyms": list(e.synonyms), "definition": e.definition}
                     for group in sorted(self.entries) for e in self.entries[group]]
        self.class_signature = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False)
                                              .encode("utf-8")).hexdigest()
        raw = json.dumps({"version": RESOLVER_VERSION, "classes": self.class_signature,
                          "aliases": self.alias_data}, sort_keys=True, ensure_ascii=False)
        self.fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        self._index: dict[str, CandidateIndex] = {}

    def entry(self, group: str, iri: str | None) -> Entry | None:
        """The active class `iri` if it belongs to `group` or one of its fallback groups."""
        if not iri:
            return None
        for g in (group, *FALLBACK_GROUPS.get(group, ())):
            e = self.by_iri.get((g, iri))
            if e is not None:
                return e
        return None

    # ---- deterministic tier (v1 behaviour, unchanged) -------------------------------------------
    def _hits(self, group: str, text: str) -> list[tuple[Entry, str]]:
        key = normalize(text)
        matches = self.lookup.get((group, key), []) + [
            (e, "approved_alias") for e in self.aliases.get((group, key), [])]
        # Preserve all distinct IRIs; exact label beats a synonym of another entry,
        # but multiple exact labels are genuinely ambiguous.
        best: dict[str, tuple[Entry, str]] = {}
        weights = {"exact_label": 4, "approved_alias": 3, "iri_local_name": 2, "synonym": 1}
        for e, method in matches:
            old = best.get(e.iri)
            if old is None or weights[method] > weights[old[1]]:
                best[e.iri] = (e, method)
        values = list(best.values())
        if any(m == "exact_label" for _, m in values):
            values = [(e, m) for e, m in values if m == "exact_label"]
        return sorted(values, key=lambda x: x[0].iri)

    def _suggestions(self, group: str, term: str, limit: int = 5) -> list[dict]:
        query_tokens = _tokens(term)
        suggestions = []
        for e in self.entries.get(group, []):
            ratio = SequenceMatcher(None, normalize(term), normalize(e.label)).ratio()
            overlap = len(query_tokens & _tokens(e.label)) / max(1, len(query_tokens | _tokens(e.label)))
            score = max(ratio, overlap)
            if score >= 0.50:
                suggestions.append((score, e))
        suggestions.sort(key=lambda s: (-s[0], s[1].iri))
        return [{"iri": e.iri, "label": e.label, "lexical_score": round(score, 3),
                 "meaning": "lexical_candidate_only"} for score, e in suggestions[:limit]]

    def _hint(self, term: str, group: str, context: str) -> dict | None:
        for hint in self.alias_data.get("unresolved_concepts", []):
            if hint.get("group") != group or normalize(hint.get("term")) != normalize(term):
                continue
            required = hint.get("required_context_any", [])
            if required and not any(normalize(s) in normalize(context) for s in required):
                continue
            return hint
        return None

    def _expanded_term(self, term: str, group: str, context: str) -> str:
        hint = self._hint(term, group, context)
        return hint.get("canonical_name", term) if hint else term

    def resolve(self, term: str, group: str, context: str = "",
                selected_label: str | None = None, selected_iri: str | None = None,
                proposed_label: str | None = None) -> dict[str, Any]:
        """Deterministic lookup only (no retrieval, no model): unique exact labels / approved aliases.
        Other hits stay reviewable. Used for ad-hoc lookups and as tier 1 of resolve_extraction."""
        if group not in self.entries:
            return {"status": "unknown_group", "group": group, "term": term,
                    "matched_iri": None, "matched_label": None, "requires_review": True,
                    "candidates": [], "reason": "Group absent from v11 profile"}
        terms = [x for x in (term, proposed_label) if isinstance(x, str) and x.strip() and x != OTHER]
        for value in terms:
            expanded = self._expanded_term(value, group, context)
            hits = self._hits(group, expanded)
            if len(hits) == 1 and hits[0][1] in ("exact_label", "approved_alias"):
                e, method = hits[0]
                return {"status": "exact_match", "group": group, "term": term,
                        "matched_iri": e.iri, "matched_label": e.label,
                        "method": method, "requires_review": False, "candidates": [],
                        "reason": "Unique active ontology label / reviewed alias"}
            if hits:
                return {"status": "ambiguous", "group": group, "term": term,
                        "matched_iri": None, "matched_label": None,
                        "requires_review": True,
                        "candidates": [{"iri": e.iri, "label": e.label, "method": m} for e, m in hits],
                        "reason": "Ambiguous label, unapproved synonym, or IRI local name"}
        original = (self.by_iri.get((group, selected_iri)) if selected_iri else None)
        if original is None and selected_label:
            original = self.by_label.get((group, normalize(selected_label)))
        if original is not None:
            needs_review = group not in INSTANCE_GROUPS
            return {"status": "existing_class_unverified" if needs_review else "existing_instance_type",
                    "group": group, "term": term,
                    "matched_iri": original.iri, "matched_label": original.label,
                    "requires_review": needs_review,
                    "candidates": self._suggestions(group, term) if needs_review else [],
                    "reason": ("Previously selected ontology class exists, but source term is not an exact class match"
                               if needs_review else "Existing typed instance; surface name need not equal its class label")}
        cross = []
        for other in self.entries:
            if other == group:
                continue
            for e, method in self._hits(other, term):
                if method == "exact_label":
                    cross.append({"group": other, "iri": e.iri, "label": e.label})
        return {"status": "out_of_group" if cross else "missing_class",
                "group": group, "term": term, "matched_iri": None, "matched_label": None,
                "requires_review": True, "expanded_term": self._expanded_term(term, group, context),
                "candidates": self._suggestions(group, term), "cross_group_matches": cross[:10],
                "reason": "No exact active class in the selected group; related classes are not equivalent"}

    # ---- retrieval tier ---------------------------------------------------------------------------
    def index(self, group: str) -> CandidateIndex:
        if group not in self._index:
            ont = self.profile.ontology
            in_group = {e.iri for e in self.entries.get(group, [])}
            parents = {e.iri: [p for p in ont.classes[e.iri].parents if p in in_group]
                       for e in self.entries.get(group, [])}
            self._index[group] = CandidateIndex(self.entries.get(group, []), parents)
        return self._index[group]

    def candidates(self, group: str, terms: list[str], context: str = "", k: int = 8) -> list[Entry]:
        """Top-k lexical candidates plus the in-group parents of the best three (so a correct broader
        class is on the list when no specific one exists). Rejected mappings are excluded."""
        exclude = set().union(*[self.rejected.get((group, normalize(t)), set()) for t in terms]) if terms else set()
        scored = []
        for g in (group, *FALLBACK_GROUPS.get(group, ())):
            if g in self.entries:
                scored += self.index(g).search(terms, context, k=k, exclude=exclude)
        scored.sort(key=lambda s: (-s[0], s[1].iri))
        hits = list(dict.fromkeys(e for _, e in scored))[:k]
        extra = []
        for e in hits[:3]:
            for p in self.index(e.group).parents.get(e.iri, []):
                pe = self.by_iri.get((e.group, p))
                if pe and pe not in hits and pe not in extra and pe.iri not in exclude:
                    extra.append(pe)
        return hits + extra[:4]

    # ---- mention collection -----------------------------------------------------------------------
    def _mentions(self, p: dict) -> list[Mention]:
        out: list[Mention] = []
        p1, p2, p3 = p["pass1"], p["pass2"], p["pass3"]

        def setter(obj, field_, iri_field, proposed_field, term):
            def apply(res):
                status = res["status"]
                if res.get("matched_iri"):
                    obj[field_], obj[iri_field] = res["matched_label"], res["matched_iri"]
                    # a concept typed to a broader class keeps its own term as the proposed narrower class
                    obj[proposed_field] = term if res.get("narrower_concept") else None
                else:
                    obj[field_], obj[iri_field] = OTHER, None
                    obj[proposed_field] = obj.get(proposed_field) or term
                obj.setdefault("ontology_resolution", {})[field_] = {
                    "status": status, "method": res.get("method"), "match_type": res.get("match_type"),
                    "confidence": res.get("confidence"), "requires_review": res["requires_review"]}
            return apply

        for m in p1.get("materials", []):
            terms = [t for t in (_first(m.get("formula")), _first(m.get("name")), _first(m.get("type_term"))) if t]
            out.append(Mention(m["id"], "material", "material", terms or [m.get("name", "")],
                               _first(m.get("type_term")), _quotes(m),
                               setter(m, "ontology_class", "ontology_iri", "proposed_class_label",
                                      _first(m.get("type_term"), m.get("name")))))
        for s in p1.get("samples", []):
            terms = [t for t in (_first(s.get("type_term")), _first(s.get("name_in_paper"))) if t]
            out.append(Mention(s["id"], "sample", "sample", terms, _first(s.get("description")), _quotes(s),
                               setter(s, "ontology_class", "ontology_iri", "proposed_class_label",
                                      _first(s.get("type_term"), s.get("name_in_paper")))))
        for proc in p1.get("process_steps", []):
            terms = [t for t in (_first(proc.get("type_term")), _first(proc.get("name_in_text"))) if t]
            out.append(Mention(proc["id"], "process", "process", terms, _first(proc.get("name_in_text")),
                               _quotes(proc), setter(proc, "ontology_class", "ontology_iri", "proposed_class_label",
                                                     _first(proc.get("type_term"), proc.get("name_in_text")))))
            for j, prm in enumerate(proc.get("parameters", []), 1):
                terms = [t for t in (_first(prm.get("type_term")), _first(prm.get("name_in_text"))) if t]
                out.append(Mention(prm.get("id", f"{proc['id']}.p{j}"), "parameter", "process_attribute", terms,
                                   f"{_first(prm.get('name_in_text'))} = {prm.get('value_text', '')} "
                                   f"(parameter of {_first(proc.get('type_term'), proc.get('name_in_text'))})",
                                   _quotes(proc, 1),
                                   setter(prm, "attribute_class", "ontology_iri", "proposed_class_label",
                                          _first(prm.get("type_term"), prm.get("name_in_text")))))
        for m in p2.get("measurements", []):
            terms = [t for t in (_first(m.get("property_term")), _first(m.get("property_name_in_text"))) if t]
            out.append(Mention(m["id"], "measurement_property", "property", terms,
                               f"measured by {_first(m.get('technique_term'), m.get('technique_name_in_text'))}; "
                               f"value {m.get('value_text') or 'n/a'}", _quotes(m),
                               setter(m, "property_class", "property_iri", "proposed_property_label",
                                      _first(m.get("property_term"), m.get("property_name_in_text")))))
            terms = [t for t in (_first(m.get("technique_term")), _first(m.get("technique_name_in_text"))) if t]
            out.append(Mention(m["id"], "measurement_technique", "technique", terms,
                               f"used to measure {_first(m.get('property_term'), m.get('property_name_in_text'))}"
                               + (f"; instrument {m['instrument']}" if m.get("instrument") else ""), _quotes(m),
                               setter(m, "technique_class", "technique_iri", "proposed_technique_label",
                                      _first(m.get("technique_term"), m.get("technique_name_in_text")))))
        for a in p3.get("applications", []):
            out.append(Mention(a["id"], "application", "application", [_first(a.get("name"))] + list(a.get("aliases") or []),
                               _first(a.get("application_status")), _quotes(a),
                               setter(a, "ontology_class", "ontology_iri", "proposed_class_label", _first(a.get("name")))))
        for e in p3.get("entities", []):
            group = e.get("entity_group") or "other"
            if group not in self.entries:
                group = "unmapped" if "unmapped" in self.entries else "other"
            if group in ("other", "unmapped"):    # recover a typed group from a unique exact label elsewhere
                target = [g for g in CONTEXT_GROUPS if g in self.entries and g not in ("other", "unmapped")
                          and self.resolve(e.get("name", ""), g, _quotes(e))["status"] == "exact_match"]
                if len(target) == 1:
                    group = target[0]
            e["entity_group"] = group
            out.append(Mention(e["id"], "scientific_entity", group, [_first(e.get("name"))] + list(e.get("aliases") or []),
                               _first(e.get("term_definition")), _quotes(e),
                               setter(e, "ontology_class", "ontology_iri", "proposed_class_label", _first(e.get("name")))))
        for i, r in enumerate(p3.get("relations", []), 1):
            obj_id = r.get("object_id") or ""
            if obj_id.startswith("PROP:") and obj_id[5:].strip():
                term = obj_id[5:].strip()

                def apply_prop(res, r=r, term=term):
                    r["object_term"] = term
                    r["object_id"] = "PROP:" + (res["matched_label"] if res.get("matched_iri") else term)
                    r.setdefault("ontology_resolution", {})["object_id"] = {
                        "status": res["status"], "method": res.get("method"), "match_type": res.get("match_type"),
                        "confidence": res.get("confidence"), "requires_review": res["requires_review"]}
                out.append(Mention(r.get("id") or f"R{i}", "relation_property", "property", [term],
                                   f"object of '{r.get('predicate')}' ({r.get('scope') or ''})", _quotes(r), apply_prop))
        return out

    # ---- the full resolution ------------------------------------------------------------------
    def _decided(self, mention: Mention, e: Entry | None, status: str, method: str, *, match_type=None,
                 confidence=None, reason="", review_below: float = 0.8, candidates=()) -> dict:
        # A concept (property, technique, phenomenon...) placed under a broader class is not equivalent to
        # it: keep it reviewable and report the narrower concept as an ontology gap. For instance groups
        # (sample, material, process...) a broader class is simply the instance's type.
        concept = mention.group not in INSTANCE_GROUPS
        narrower = e is not None and concept and match_type in ("narrower_than", "instance_of")
        if e is not None and status != "exact_match":
            status = ("adjudicated_broader" if narrower else
                      "adjudicated_instance" if match_type in ("instance_of", "narrower_than") else
                      "adjudicated_match")
        if status == "exact_match":
            review = False
        elif status == "missing_class":
            review = method != "confirmed_missing"
        elif narrower:
            review = True
        else:
            review = confidence is None or confidence < review_below
        return {"status": status, "group": mention.group, "term": mention.terms[0] if mention.terms else "",
                "matched_iri": e.iri if e else None, "matched_label": e.label if e else None,
                "method": method, "match_type": match_type, "confidence": confidence,
                "narrower_concept": narrower, "requires_review": review, "reason": reason,
                "candidates": [{"iri": c.iri, "label": c.label} for c in candidates][:12]}

    def resolve_extraction(self, extraction: dict, ask: Callable[[str, str], dict] | None = None,
                           memory: ResolutionMemory | None = None, paper_id: str = "_",
                           top_k: int = 8, review_below: float = 0.8, batch_size: int = 15,
                           model: str | None = None) -> dict:
        """Resolve every mention of passes 1-3. Returns the passes in the same format with ontology
        fields filled (ontology_class / *_iri / proposed_*), plus an audit list. Never mutates inputs.

        ask(label, task) -> MappingDecisions dict: the adjudication call (None = no model; mentions that
        aren't settled deterministically or from memory stay OTHER and reviewable, with candidates)."""
        p = copy.deepcopy(extraction)
        mentions = self._mentions(p)
        pending: dict[str, list[Mention]] = defaultdict(list)    # memory key -> mentions sharing a decision
        for m in mentions:
            if not m.terms or m.group not in self.entries:
                m.resolution = {"status": "unknown_group" if m.group not in self.entries else "missing_class",
                                "group": m.group, "term": "", "matched_iri": None, "matched_label": None,
                                "method": "no_term", "requires_review": True, "reason": "No usable term"}
                continue
            ctx = f"{m.description} {m.context}"
            # 1. deterministic: exact label / approved alias for any of the mention's terms
            for g in (m.group, *FALLBACK_GROUPS.get(m.group, ())):
                for t in m.terms:
                    r = self.resolve(t, g, ctx)
                    if r["status"] == "exact_match":
                        e = self.by_iri[(g, r["matched_iri"])]
                        m.resolution = self._decided(m, e, "exact_match",
                                                     r["method"] + ("" if g == m.group else f"_in_{g}"),
                                                     match_type="same_concept", confidence=1.0, reason=r["reason"])
                        break
                if m.resolution:
                    break
            if m.resolution:
                continue
            if (m.group, normalize(m.terms[0])) in self.confirmed_missing:
                m.resolution = self._decided(m, None, "missing_class", "confirmed_missing",
                                             reason="Reviewed: concept is not in the ontology (confirmed gap)")
                continue
            # 2. memory
            key = ResolutionMemory.key(m.group, m.terms[0])
            mem = memory.get(key, self.class_signature) if memory else None
            if mem is None and memory:
                mem = memory.get(ResolutionMemory.key(m.group, m.terms[0], paper_id), self.class_signature)
            if mem and mem.get("iri") in self.rejected.get((m.group, normalize(m.terms[0])), set()):
                mem = None
            if mem:
                e = self.entry(m.group, mem.get("iri"))
                if mem.get("iri") is None or e is not None:
                    m.resolution = self._decided(m, e, "missing_class" if e is None else "adjudicated_match",
                                                 "memory", match_type=mem.get("match_type"),
                                                 confidence=mem.get("confidence"), reason=mem.get("reason", ""),
                                                 review_below=review_below)
                    memory.note_use(key if key in memory.entries else
                                    ResolutionMemory.key(m.group, m.terms[0], paper_id), paper_id)
                    continue
            # 3. retrieval
            hint = self._hint(m.terms[0], m.group, ctx)
            terms = list(dict.fromkeys([self._expanded_term(t, m.group, ctx) for t in m.terms]))
            m.candidates = self.candidates(m.group, terms, ctx, k=top_k)
            if hint:
                m.description = (m.description + f" | note: {hint.get('description', '')}").strip(" |")
            if not m.candidates:
                m.resolution = self._decided(m, None, "missing_class", "no_candidates",
                                             reason="No lexically related class in this group")
                continue
            pending[key].append(m)

        # 4. adjudication (batched by group), one decision per memory key
        if pending and ask is not None:
            by_group: dict[str, list[str]] = defaultdict(list)
            for key, ms in pending.items():
                by_group[ms[0].group].append(key)
            for group, keys in sorted(by_group.items()):
                for b in range(0, len(keys), batch_size):
                    chunk = keys[b:b + batch_size]
                    payload = []
                    for j, key in enumerate(chunk, 1):
                        m = pending[key][0]
                        payload.append({
                            "mention_id": f"m{j}", "group": group,
                            "group_description": getattr(self.profile.groups.get(group), "description", ""),
                            "terms": m.terms, "description": m.description, "context": m.context[:400],
                            "candidates": [{"key": f"C{i}", "label": c.label, "definition": c.definition[:260],
                                            "synonyms": list(c.synonyms[:4]),
                                            "parent": ", ".join(self.by_iri[(c.group, x)].label
                                                                for x in self.index(c.group).parents.get(c.iri, [])[:2])}
                                           for i, c in enumerate(m.candidates, 1)]})
                    task = ("Map each mention to at most one of its candidates. Return one decision per mention_id.\n"
                            + json.dumps(payload, ensure_ascii=False, indent=1))
                    result = ask(f"resolve_{group}_{b // batch_size + 1}", task)
                    got = {d["mention_id"]: d for d in result.get("decisions", [])}
                    for j, key in enumerate(chunk, 1):
                        ms, d = pending[key], got.get(f"m{j}")
                        m0 = ms[0]
                        cand = {f"C{i}": c for i, c in enumerate(m0.candidates, 1)}
                        e = cand.get(d["choice"]) if d else None
                        if d is None:
                            res = self._decided(m0, None, "missing_class", "llm_no_answer", candidates=m0.candidates,
                                                reason="Adjudicator returned no decision for this mention")
                        elif e is None or d["match_type"] == "none":
                            res = self._decided(m0, None, "missing_class", "llm_adjudicated", match_type="none",
                                                confidence=d["confidence"], reason=d["reason"], candidates=m0.candidates)
                        else:
                            status = "adjudicated_instance" if d["match_type"] == "instance_of" else "adjudicated_match"
                            res = self._decided(m0, e, status, "llm_adjudicated", match_type=d["match_type"],
                                                confidence=d["confidence"], reason=d["reason"],
                                                review_below=review_below, candidates=m0.candidates)
                        for m in ms:
                            m.resolution = res
                        if memory is not None and d is not None:
                            mkey = key if not d.get("context_dependent") else \
                                ResolutionMemory.key(group, m0.terms[0], paper_id)
                            memory.put(mkey, {"group": group, "term": m0.terms[0], "iri": res["matched_iri"],
                                              "label": res["matched_label"], "status": res["status"],
                                              "match_type": res["match_type"], "confidence": res["confidence"],
                                              "reason": res["reason"], "context_dependent": d.get("context_dependent"),
                                              "candidates": res["candidates"], "context_example": m0.context[:300],
                                              "model": model, "class_signature": self.class_signature,
                                              "decided_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                       paper_id)
        for key, ms in pending.items():
            for m in ms:
                if m.resolution is None:       # no adjudicator: keep the candidates for review
                    m.resolution = self._decided(m, None, "missing_class", "not_adjudicated",
                                                 candidates=m.candidates,
                                                 reason="No exact match; candidates kept for review (no model call)")
        if memory is not None:
            memory.save()

        reports = []
        for m in mentions:
            m.apply(m.resolution)
            reports.append({"id": m.record_id, "role": m.role, **m.resolution,
                            "term": m.terms[0] if m.terms else "", "all_terms": m.terms,
                            "context": m.context[:300]})
        counts = Counter(r["status"] for r in reports)
        return {**p, "ontology_resolution": reports,
                "ontology_resolution_summary": dict(sorted(counts.items())),
                "resolver_fingerprint": self.fingerprint}


def annotate_validation(validated: dict, resolved: dict) -> dict:
    """Flag concept/class mismatches in existing validation without losing provenance.

    Exact matches and confident adjudications pass; everything the resolver marked requires_review
    turns an auto-accepted item into needs_review. Concepts typed to a broader class are added to
    the ontology-gap list as narrower concepts the ontology lacks."""
    index: dict[str, list[dict]] = defaultdict(list)
    for report in resolved.get("ontology_resolution", []):
        index[report["id"]].append(report)
    for item in validated.get("items", []):
        related = index.get(item.get("id"), [])
        review = [r for r in related if r["requires_review"]]
        if review:
            for r in review:
                note = f"ontology resolver ({r['role']}): {r['status']} — {r['reason']}"
                if note not in item["notes"]:
                    item["notes"].append(note)
            if item["status"] == "auto_accepted":
                item["status"] = "needs_review"
    gaps = validated.setdefault("ontology_gaps", [])
    for r in resolved.get("ontology_resolution", []):
        if r.get("narrower_concept"):
            gaps.append({"group": r["group"], "proposed_label": r["term"], "text": r["term"], "item": r["id"],
                         "kind": "narrower_than_ontology_class", "parent_label": r["matched_label"]})
    # Validation computes its summary before these review-status changes.
    # Recalculate it to keep notebook tables and run exports consistent.
    counts = Counter((it["kind"], it["status"]) for it in validated.get("items", []))
    validated["summary"] = [
        {"kind": kind, "status": status, "count": count}
        for (kind, status), count in sorted(counts.items())
    ]
    validated["ontology_resolution_summary"] = resolved.get("ontology_resolution_summary", {})
    validated["ontology_resolution"] = resolved.get("ontology_resolution", [])
    return validated


# ---------------------------------------------------------------------------
# Human review: export decisions, then apply the reviewer's actions as aliases / rejections / gaps
# ---------------------------------------------------------------------------
REVIEW_COLUMNS = ["group", "term", "status", "match_type", "class_label", "class_iri", "confidence",
                  "mentions", "papers", "method", "reason", "example_context", "candidates_shown",
                  "action", "remap_to_iri", "reviewer_note"]
ACTIONS = ("approve", "reject", "remap", "gap", "")


def write_resolution_review(resolved_by_paper: dict, path: Path, include_exact: bool = False) -> int:
    """One row per (group, term, decision) across all papers, most frequent first. Fill in `action`:
    approve (mapping is right) | reject (mapping is wrong; re-decide without that class) |
    remap (put the right class IRI in remap_to_iri) | gap (concept truly missing from the ontology)."""
    agg: dict[tuple, dict] = {}
    for pid, res in resolved_by_paper.items():
        for r in res.get("ontology_resolution", []):
            if r["status"] == "exact_match" and not include_exact:
                continue
            k = (r["group"], normalize(r["term"]), r["status"], r.get("matched_iri"))
            row = agg.setdefault(k, {"group": r["group"], "term": r["term"], "status": r["status"],
                                     "match_type": r.get("match_type") or "", "class_label": r.get("matched_label") or "",
                                     "class_iri": r.get("matched_iri") or "", "confidence": r.get("confidence"),
                                     "mentions": 0, "papers": set(), "method": r.get("method") or "",
                                     "reason": r.get("reason", ""), "example_context": r.get("context", "")[:250],
                                     "candidates_shown": "; ".join(c["label"] for c in r.get("candidates", [])[:8]),
                                     "action": "", "remap_to_iri": "", "reviewer_note": ""})
            row["mentions"] += 1
            row["papers"].add(pid)
    rows = sorted(agg.values(), key=lambda x: (-x["mentions"], x["group"], x["term"].casefold()))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "papers": len(r["papers"])})
    return len(rows)


def apply_resolution_review(review_csv: Path, resolver: OntologyResolver,
                            memory: ResolutionMemory | None = None, alias_path: Path | None = None) -> dict:
    """Turn reviewed rows into durable knowledge in scientific_aliases.json (tracked in git):
    approve/remap -> approved alias (applies as an exact match everywhere), reject -> rejected mapping
    (that class is never proposed again for the term), gap -> confirmed missing (no more model calls).
    Re-run the pipeline afterwards: only the ontology_resolve stage and later stages are redone."""
    alias_path = Path(alias_path or resolver.alias_path)
    data = json.loads(alias_path.read_text(encoding="utf-8"))
    data.setdefault("aliases", [])
    data.setdefault("rejected", [])
    data.setdefault("confirmed_missing", [])
    done, errors = Counter(), []
    today = date.today().isoformat()
    with open(review_csv, encoding="utf-8-sig", newline="") as f:
        for n, row in enumerate(csv.DictReader(f), 2):
            action = (row.get("action") or "").strip().lower()
            if action not in ACTIONS:
                errors.append(f"line {n}: unknown action {action!r} (use approve, reject, remap, gap)")
                continue
            if not action:
                continue
            group, term = row["group"], row["term"]
            iri = (row.get("remap_to_iri") or "").strip() if action == "remap" else row.get("class_iri", "")
            key = (group, normalize(term))
            if action in ("approve", "remap"):
                if resolver.entry(group, iri) is None:
                    errors.append(f"line {n}: {iri!r} is not an active class of group {group!r}")
                    continue
                if not any(a["group"] == group and normalize(a["alias"]) == key[1] and a.get("target_iri") == iri
                           for a in data["aliases"]):
                    data["aliases"].append({"group": group, "alias": term, "target_iri": iri, "approved": True,
                                            "source": "resolution_review", "reviewed": today,
                                            "note": row.get("reviewer_note", "")})
            elif action == "reject":
                if not iri:
                    errors.append(f"line {n}: nothing to reject (no class_iri)")
                    continue
                data["rejected"].append({"group": group, "term": term, "iri": iri, "reviewed": today,
                                         "note": row.get("reviewer_note", "")})
            elif action == "gap":
                data["confirmed_missing"].append({"group": group, "term": term, "reviewed": today,
                                                  "note": row.get("reviewer_note", "")})
            if memory is not None:
                memory.forget(ResolutionMemory.key(group, term))
            done[action] += 1
    data["aliases"] = sorted(data["aliases"], key=lambda a: (a["group"], normalize(a["alias"])))
    alias_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if memory is not None:
        memory.save()
    return {"applied": dict(done), "errors": errors, "alias_file": str(alias_path)}
