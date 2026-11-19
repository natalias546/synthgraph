"""Conservative ontology resolver for the v11 materials-science extraction pipeline.

The LLM has already extracted source terms, evidence and candidate class labels.
This module does not call a model, alter the OWL file or invent an IRI. It first
matches the *scientific term*, then consults the model's proposed class, and
retains every unresolved/mismatched case for review.

A fuzzy name or hierarchy resemblance is NEVER an automatic equivalence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..ontology_loader import OTHER, Profile

RESOLVER_VERSION = "1.2.0"
# Reviewed lexical *predicate* cues, not OWL class assignments or a semantic
# similarity model. Unknown phrases remain reviewable, never asserted.
RELATION_PHRASE_ALIASES = {
    "is part of": "part_of", "part of": "part_of",
    "composed of": "composed_of", "consists of": "composed_of",
    "made of": "composed_of", "is made of": "composed_of",
    "intended for": "intended_for", "developed for": "intended_for",
    "affects": "influences", "influences": "influences",
    "derived from": "derived_from", "prepared from": "derived_from",
    "compared with": "compared_with", "compared to": "compared_with",
    "in comparison with": "compared_with", "consistent with": "compared_with",
}
ALIAS_PATH = Path(__file__).with_name("scientific_aliases.json")


def normalize(text: str | None) -> str:
    """Conservative normalization; preserve chemical formulas and subscripts."""
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = re.sub(r"[‐‑‒–—−_-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9α-ω]+", normalize(s)))


@dataclass(frozen=True)
class Entry:
    group: str
    label: str
    iri: str
    synonyms: tuple[str, ...]
    definition: str


class OntologyResolver:
    """Index the active OWL slice once, then resolve extracted mentions by evidence."""

    def __init__(self, profile: Profile, alias_path: Path | None = None):
        self.profile = profile
        path = Path(alias_path) if alias_path is not None else ALIAS_PATH
        self.alias_data = json.loads(path.read_text(encoding="utf-8"))
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
            e = self.by_iri.get((group, target))
            if e is None:
                raise ValueError(f"Approved ontology alias {alias!r} targets missing/inactive IRI {target!r}")
            self.aliases[group, alias].append(e)
        signature = [{"group": e.group, "label": e.label, "iri": e.iri,
                      "synonyms": list(e.synonyms), "definition": e.definition}
                     for group in sorted(self.entries) for e in self.entries[group]]
        raw = json.dumps({"version": RESOLVER_VERSION, "classes": signature,
                          "aliases": self.alias_data}, sort_keys=True, ensure_ascii=False)
        self.fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()

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

    def _suggestions(self, group: str, term: str, limit: int = 5, context: str = "") -> list[dict]:
        query_tokens = _tokens(term)
        suggestions = []
        for e in self.entries.get(group, []):
            ratio = SequenceMatcher(None, normalize(term), normalize(e.label)).ratio()
            overlap = len(query_tokens & _tokens(e.label)) / max(1, len(query_tokens | _tokens(e.label)))
            score = max(ratio, overlap)
            # Definitions can provide additional retrieval signal, never equivalence.
            # Scientific context is only a small tie-breaker for already plausible names.
            if score >= 0.50 and context and e.definition:
                context_tokens = _tokens(context)
                definition_tokens = _tokens(e.definition)
                score += min(0.04, 0.01 * len(context_tokens & definition_tokens))
            if score >= 0.50:
                suggestions.append((score, e))
        suggestions.sort(key=lambda s: (-s[0], s[1].iri))
        return [{"iri": e.iri, "label": e.label, "lexical_score": round(score, 3),
                 "meaning": "lexical_candidate_only"} for score, e in suggestions[:limit]]

    def _expanded_term(self, term: str, group: str, context: str) -> str:
        for hint in self.alias_data.get("unresolved_concepts", []):
            if hint.get("group") != group or normalize(hint.get("term")) != normalize(term):
                continue
            required = hint.get("required_context_any", [])
            if required and not any(normalize(s) in normalize(context) for s in required):
                continue
            return hint.get("canonical_name", term)
        return term

    def resolve(self, term: str, group: str, context: str = "",
                selected_label: str | None = None, selected_iri: str | None = None,
                proposed_label: str | None = None) -> dict[str, Any]:
        """Resolve only unique exact labels/approved aliases. Other hits stay reviewable."""
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
                # Source-aware physical-context rule: exact lexical identity alone
                # must not turn "potential applications" into electrical potential.
                from ..pipeline.scientific_context import safe_context
                compatibility = safe_context(value, group, context)
                if not compatibility["allowed"]:
                    return {"status": "context_rejected", "group": group, "term": term,
                            "matched_iri": None, "matched_label": None,
                            "requires_review": True, "candidates": [],
                            "reason": compatibility["reason"]}
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
        # If the original LLM selected a valid class but the scientific term did not
        # match it, preserve it, but explicitly require review (broader != exact).
        original = (self.by_iri.get((group, selected_iri)) if selected_iri else None)
        if original is None and selected_label:
            original = self.by_label.get((group, normalize(selected_label)))
        if original is not None:
            # Material/sample/process names denote instances, not their class labels.
            # Exact surface-name matching is meaningful for reported properties and
            # techniques, but would incorrectly flag every named specimen as a gap.
            instance_groups = {"material", "sample", "process", "process_attribute",
                               "instrument", "software", "reference", "dataset"}
            needs_review = group not in instance_groups
            return {"status": "existing_class_unverified" if needs_review else "existing_instance_type",
                    "group": group, "term": term,
                    "matched_iri": original.iri, "matched_label": original.label,
                    "requires_review": needs_review,
                    "candidates": self._suggestions(group, term, context=context) if needs_review else [],
                    "reason": ("Previously selected ontology class exists, but source term is not an exact class match"
                               if needs_review else "Existing typed instance; surface name need not equal its class label")}
        # Look for an exact label in a different ontology group, without automatically
        # moving a measurement's property to a phenomenon or a material constituent.
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
                "candidates": self._suggestions(group, term, context=context), "cross_group_matches": cross[:10],
                "reason": "No exact active class in the selected group; related classes are not equivalent"}

    @staticmethod
    def _context(obj: dict) -> str:
        parts = [obj.get("property_name_in_text"), obj.get("technique_name_in_text"),
                 obj.get("name_in_text"), obj.get("name"), obj.get("description")]
        parts.extend(e.get("quote", "") for e in obj.get("evidence", []))
        parts.extend(str(obj.get(k, "")) for k in ("unit", "value_text", "parameter_symbol", "data_origin"))
        for condition in obj.get("conditions", []) + obj.get("settings", []):
            if isinstance(condition, dict):
                parts.extend(str(condition.get(k, "")) for k in ("name", "symbol", "unit", "value_text"))
        return " ".join(str(s) for s in parts if s)

    def _apply(self, obj: dict, group: str, field: str, iri_field: str,
               proposed_field: str, term: str, record_id: str, role: str,
               reports: list[dict], extra: dict | None = None) -> None:
        resolution = self.resolve(term=term or obj.get(proposed_field) or obj.get(field) or "",
                                  group=group, context=self._context(obj),
                                  selected_label=obj.get(field), selected_iri=obj.get(iri_field),
                                  proposed_label=obj.get(proposed_field))
        reports.append({"id": record_id, "role": role, **(extra or {}), **resolution})
        if resolution["status"] == "exact_match":
            obj[field] = resolution["matched_label"]
            obj[iri_field] = resolution["matched_iri"]
            if obj.get(proposed_field) and normalize(obj[proposed_field]) == normalize(resolution["matched_label"]):
                obj[proposed_field] = None
        elif obj.get(field) == OTHER:
            # Keep the original text when the ontology lacks the exact concept.
            obj[iri_field] = None
            obj[proposed_field] = obj.get(proposed_field) or term

    def resolve_extraction(self, extraction: dict) -> dict:
        """Return passes in the existing format, plus an audit; never mutate inputs."""
        p = copy.deepcopy(extraction)
        reports: list[dict] = []
        p1, p2, p3 = p["pass1"], p["pass2"], p["pass3"]
        for m in p1.get("materials", []):
            self._apply(m, "material", "ontology_class", "ontology_iri", "proposed_class_label",
                        m.get("name", ""), m["id"], "material", reports)
        for s in p1.get("samples", []):
            self._apply(s, "sample", "ontology_class", "ontology_iri", "proposed_class_label",
                        s.get("name_in_paper", ""), s["id"], "sample", reports)
        for proc in p1.get("process_steps", []):
            self._apply(proc, "process", "ontology_class", "ontology_iri", "proposed_class_label",
                        proc.get("name_in_text", ""), proc["id"], "process", reports)
            for j, param in enumerate(proc.get("parameters", []), 1):
                self._apply(param, "process_attribute", "attribute_class", "ontology_iri",
                            "proposed_class_label", param.get("name_in_text", ""),
                            param.get("id", f"{proc['id']}:parameter:{j}"), "parameter", reports)
        for m in p2.get("measurements", []):
            self._apply(m, "property", "property_class", "property_iri", "proposed_property_label",
                        m.get("property_name_in_text", ""), m["id"], "measurement_property", reports)
            self._apply(m, "technique", "technique_class", "technique_iri", "proposed_technique_label",
                        m.get("technique_name_in_text", ""), m["id"], "measurement_technique", reports)
        for a in p3.get("applications", []):
            self._apply(a, "application", "ontology_class", "ontology_iri", "proposed_class_label",
                        a.get("name", ""), a["id"], "application", reports)
        context_groups = ("phenomenon", "mechanism", "theory_model", "instrument", "software",
                          "phase", "structure", "reference", "dataset", "other", "unmapped")
        for e in p3.get("entities", []):
            group = e.get("entity_group", "other")
            if group not in self.entries:
                group = "unmapped" if "unmapped" in self.entries else "other"
            if group in ("other", "unmapped"):
                target = []
                for candidate in context_groups:
                    if candidate not in self.entries or candidate in ("other", "unmapped"):
                        continue
                    r = self.resolve(e.get("name", ""), candidate, self._context(e))
                    if r["status"] == "exact_match":
                        target.append(candidate)
                if len(target) == 1:
                    group = target[0]
                    e["entity_group"] = group
            self._apply(e, group, "ontology_class", "ontology_iri", "proposed_class_label",
                        e.get("name", ""), e["id"], "scientific_entity", reports)
        for u in p3.get("unresolved_terms", []):
            reports.append({"id": u["id"], "role": "unresolved_term", "status": "unresolved_term",
                            "group": "unmapped", "term": u.get("raw_term", ""),
                            "matched_iri": None, "matched_label": None, "requires_review": True,
                            "candidates": [], "reason": u.get("reason_unresolved", "Not mapped")})
        # Algorithmic mode: Claude returns only relation phrases and confidence,
        # never ontology predicate labels. Resolve every proposed contextual
        # relation here, after Pass 3b, using exact active profile vocabulary.
        if p3.get("context_extraction_mode") == "context_only_no_ontology_labels":
            name_to_id = defaultdict(set)
            for m in p1.get("materials", []):
                for name in (m.get("name"), m.get("formula"), *m.get("aliases", [])):
                    if name: name_to_id[normalize(name)].add(m["id"])
            for sample in p1.get("samples", []):
                for name in (sample.get("name_in_paper"), *sample.get("aliases", [])):
                    if name: name_to_id[normalize(name)].add(sample["id"])
            for proc in p1.get("process_steps", []):
                name_to_id[normalize(proc.get("name_in_text"))].add(proc["id"])
            for m in p2.get("measurements", []):
                for name in (m.get("property_name_in_text"), m.get("technique_name_in_text")):
                    if name: name_to_id[normalize(name)].add(m["id"])
            for e in p3.get("entities", []):
                name_to_id[normalize(e.get("name"))].add(e["id"])
            id_group = {}
            id_group.update({m["id"]: "material" for m in p1.get("materials", [])})
            id_group.update({m["id"]: "sample" for m in p1.get("samples", [])})
            id_group.update({m["id"]: "process" for m in p1.get("process_steps", [])})
            id_group.update({m["id"]: "measurement" for m in p2.get("measurements", [])})
            id_group.update({m["id"]: m.get("entity_group", "other") for m in p3.get("entities", [])})
            id_group.update({m["id"]: "application" for m in p3.get("applications", [])})
            all_ids = {i for ids in name_to_id.values() for i in ids}
            def exact_id(term):
                if term in all_ids: return term
                ids = name_to_id.get(normalize(term), set())
                return next(iter(ids)) if len(ids) == 1 else None
            # Require unique *exact* predicate name or normalized spelling.
            # Mere semantic resemblance / fuzzy candidate must not be
            # asserted. The source phrase remains in the audit either way.
            rel_by_name = defaultdict(list)
            for rel in self.profile.relations_for("relations"):
                rel_by_name[normalize(rel.name.replace("_", " "))].append(rel)
            for i, link in enumerate(p3.get("context_links", []), 1):
                rid = f"CTXREL{i}"
                subject = exact_id(link.get("subject_text", ""))
                object_id = exact_id(link.get("object_text", ""))
                relation_term = normalize(str(link.get("relation_phrase", "")).replace("_", " "))
                relation_alias = RELATION_PHRASE_ALIASES.get(relation_term)
                canonical = normalize(relation_alias.replace("_", " ")) if relation_alias else relation_term
                matching = rel_by_name.get(canonical, [])
                valid = (len(matching) == 1 and subject and object_id and subject != object_id
                         and id_group.get(subject) in matching[0].subject_groups
                         and id_group.get(object_id) in matching[0].object_groups)
                if valid:
                    rel = matching[0]
                    negated = bool(re.search(r"\b(?:no|not|without|neither)\b", link.get("scope") or "", re.I))
                    hypothetical = bool(re.search(r"\b(?:may|could|might|possible|potential)\b", link.get("scope") or "", re.I))
                    obj = {"subject_id": subject, "predicate": rel.name,
                           "predicate_iri": rel.ontology_iri, "object_id": object_id,
                           "trend": "not_applicable", "scope": link.get("scope"),
                           "polarity": "negated" if negated else "hypothetical" if hypothetical else "affirmed",
                           "support_type": "direct", "bridge_ids": [],
                           "evidence": link.get("evidence", []),
                           "confidence": link.get("probability", 0.0),
                           "extraction_method": "resolver_exact_context_phrase"}
                    p3["relations"].append(obj)
                    link["resolution_status"] = "exact_profile_predicate"
                    reports.append({"id": rid, "role": "context_relation", "status": "exact_match",
                                    "group": "relation", "term": link.get("relation_phrase", ""),
                                    "matched_iri": rel.ontology_iri, "matched_label": rel.name,
                                    "requires_review": False, "reason": "Unique exact active relation and IDs"})
                else:
                    link["resolution_status"] = "unresolved_context_relation"
                    reason = ("ambiguous or unmatched predicate" if len(matching) != 1 else
                              "subject/object name has no unique registry ID or mismatches predicate domain/range")
                    reports.append({"id": rid, "role": "context_relation", "status": "unresolved_term",
                                    "group": "relation", "term": link.get("relation_phrase", ""),
                                    "matched_iri": None, "matched_label": None,
                                    "requires_review": True, "reason": reason, "candidates": []})
        counts = Counter(r["status"] for r in reports)
        return {**p, "ontology_resolution": reports,
                "ontology_resolution_summary": dict(sorted(counts.items())),
                "resolver_fingerprint": self.fingerprint}


def annotate_validation(validated: dict, resolved: dict) -> dict:
    """Flag concept/class mismatches in existing validation without losing provenance.

    Only exact matches can be automatically accepted. This does not change OWL or
    validate scientific truth; domain/range and causal assertions need other checks.
    """
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
