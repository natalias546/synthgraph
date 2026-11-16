"""Pre-LLM ontology candidates and explicitly reviewed scientific evidence.

No API calls. The authoritative PDF text and structure.json are immutable.
Reviewer decisions live in pre_llm_review.json, outside the stage cache. A
review is bound to the source sentences *and* ontology candidate fingerprint.

The proposed class is a hint to an LLM, never a new OWL axiom. Unsupported
fuzzy/broader matches require a user's explicit choice and are not equivalent.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from .io_utils import read_json, sha256_json, write_json
from ..ontology_resolver.resolver import normalize
from .claim_provenance import classify_surface

REVIEW_VERSION = "1.0"
EVIDENCE_DECISIONS = {"pending", "extract", "context_only", "exclude"}
CANDIDATE_DECISIONS = {"pending", "approve", "reject", "edit"}
VISUAL_DECISIONS = {"pending", "approve", "reject"}

# A few scientifically meaningful terms known to be missing from the current
# ontology can still be shown for manual review; they are NEVER assigned IRIs.
_SCIENTIFIC_PHRASES = re.compile(
    r"\b(?:residual\s+resistivity\s+ratio|magneto[ -]?resistance|"
    r"weak\s+anti[ -]?locali[sz]ation|anomalous\s+hall\s+(?:effect|resistivity)|"
    r"topological\s+surface\s+states?|spin[ -]?orbit\s+coupling|"
    r"carrier\s+mobilit(?:y|ies)|magnetic\s+susceptibility|"
    r"hall\s+coefficient|thermal\s+conductivity)\b", re.I)
_TOKENS = re.compile(r"[A-Za-z\u00c0-\u024f\u0370-\u03ff][A-Za-z0-9\u00c0-\u024f\u0370-\u03ff_/-]*")
_GENERIC = {"a", "an", "and", "as", "at", "based", "by", "data", "for", "from", "has",
            "in", "is", "material", "measurement", "method", "of", "on", "or", "other", "process",
            "property", "result", "results", "sample", "the", "to", "using", "was", "were", "with"}
_PRIORITY = ("property", "technique", "phenomenon", "mechanism", "structure", "phase",
             "process_attribute", "material", "process", "application", "theory_model",
             "instrument", "software", "sample", "reference", "dataset", "other", "unmapped")


class ReviewRequired(RuntimeError):
    """A paid operation was attempted before approved, current source evidence."""


def evidence_fingerprint(structure: dict) -> str:
    """Change if the source sentence, offsets, paragraph text, or layout ownership changes."""
    return sha256_json({
        "paragraphs": [(p["id"], p.get("section_id"), p.get("text", ""))
                       for p in structure.get("paragraphs", [])],
        "sentences": [(e["id"], e["paragraph_id"], e.get("span_start"), e.get("span_end"),
                       e.get("sentence", "")) for e in structure.get("semantic_evidence", [])],
    })


def _mentions(sentence: str, evidence: dict, resolver) -> list[tuple[int, int, str, str]]:
    """Find explicit surface spans, using the OWL label index and conservative patterns.

    No NLP inference: for a source token sequence we only ask whether it is a
    known ontology label/synonym/approved alias; no arbitrary phrase is claimed.
    """
    words = list(_TOKENS.finditer(sentence))
    hits: dict[tuple[int, int], tuple[int, int, str, str]] = {}
    for n in (4, 3, 2, 1):
        for k in range(max(0, len(words) - n + 1)):
            a, b = words[k].start(), words[k + n - 1].end()
            raw = sentence[a:b]
            key = normalize(raw)
            # Document references, discourse operators and generic sample pronouns
            # must not become physical ontology classes from matching words alone.
            if classify_surface(raw, sentence)['route'] != 'scientific_entity':
                continue
            if len(key) < 3 or key in _GENERIC or any(ch.isdigit() for ch in key) and n > 1:
                continue
            groups = [group for group in _PRIORITY
                      if resolver.lookup.get((group, key)) or resolver.aliases.get((group, key))]
            if groups and (a, b) not in hits:
                hits[a, b] = (a, b, raw, "owl_surface")
    for term in evidence.get("candidate_terms", []):
        if not isinstance(term, str) or not term.strip():
            continue
        for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", sentence, re.I):
            hits[(match.start(), match.end())] = (match.start(), match.end(), match.group(), "symbol")
    for match in _SCIENTIFIC_PHRASES.finditer(sentence):
        hits[match.start(), match.end()] = (match.start(), match.end(), match.group(), "scientific_phrase")
    # Suppress overlapping generic substrings in favor of longest explicit phrases.
    ranked = sorted(hits.values(), key=lambda v: (-(v[1] - v[0]), v[0], v[2]))
    kept: list[tuple[int, int, str, str]] = []
    for item in ranked:
        if not any(item[0] >= old[0] and item[1] <= old[1] for old in kept):
            kept.append(item)
    return sorted(kept, key=lambda v: (v[0], v[1]))[:16]


def make_candidates(structure: dict, resolver) -> dict:
    """Add candidate records after deterministic scientific evidence segmentation."""
    choices = []
    for ev in structure.get("semantic_evidence", []):
        sentence = ev.get("sentence", "")
        for start, end, term, detected_by in _mentions(sentence, ev, resolver):
            best_group, resolution, options = None, None, []
            group_choices = [g for g in _PRIORITY if g in resolver.entries]
            for group in group_choices:
                r = resolver.resolve(term, group, context=sentence)
                if r["status"] == "exact_match":
                    options.append((group, r))
            if options:
                best_group, resolution = options[0]
            else:
                # Fall back to a domain-specific group, without claiming equivalence.
                low = normalize(term)
                if re.search(r"resist|hall|conductiv|mobilit|rrr|\bmr\b", low):
                    best_group = "property"
                elif re.search(r"wal|anti locali|surface state|spin orbit", low):
                    best_group = "phenomenon"
                elif re.search(r"xrd|arpes|dft|cvt|cvd", low):
                    best_group = "technique"
                else:
                    best_group = next((g for g in group_choices if resolver._hits(g, term)), "property")
                if best_group not in resolver.entries:
                    best_group = group_choices[0]
                resolution = resolver.resolve(term, best_group, context=sentence)
            raw_id = f"{ev['id']}|{start}|{end}|{normalize(term)}"
            choice_id = "OC_" + hashlib.sha256(raw_id.encode()).hexdigest()[:18]
            candidates = []
            if resolution.get("matched_iri"):
                candidates.append({"group": best_group, "iri": resolution["matched_iri"],
                                   "label": resolution["matched_label"], "relation": resolution["status"]})
            candidates.extend({"group": best_group, **c} for c in resolution.get("candidates", []))
            for g, r in options[1:3]:
                candidates.append({"group": g, "iri": r["matched_iri"],
                                   "label": r["matched_label"], "relation": "cross_group_exact"})
            seen = set()
            candidates = [c for c in candidates if not (c.get("iri") in seen or seen.add(c.get("iri")))][:6]
            choices.append({"id": choice_id, "evidence_id": ev["id"], "paragraph_id": ev["paragraph_id"],
                            "term": term, "sentence_span_start": start, "sentence_span_end": end,
                            "detected_by": detected_by, "group": best_group,
                            "match_status": resolution["status"], "matched_iri": resolution.get("matched_iri"),
                            "suggestions": candidates, "reason": resolution.get("reason", "")})
    result = {"version": REVIEW_VERSION, "source_fingerprint": evidence_fingerprint(structure),
              "resolver_fingerprint": resolver.fingerprint, "candidates": choices,
              "evidence_count": len(structure.get("semantic_evidence", [])),
              "summary": dict(sorted(Counter(c["match_status"] for c in choices).items()))}
    result["candidate_fingerprint"] = sha256_json({k: v for k, v in result.items() if k != "candidate_fingerprint"})
    return result


def new_review(structure: dict, candidate_stage: dict) -> dict:
    """Initialize optional human-review overrides; no evidence is approved here."""
    return {"review_version": REVIEW_VERSION,
            "source_fingerprint": evidence_fingerprint(structure),
            "candidate_fingerprint": candidate_stage["candidate_fingerprint"],
            "evidence": {e["id"]: {"decision": "pending", "note": ""}
                         for e in structure.get("semantic_evidence", [])},
            "candidates": {c["id"]: {"decision": "pending", "selected_iri": None,
                                     "selected_group": None, "term_override": None, "note": ""}
                           for c in candidate_stage["candidates"]},
            "manual_candidates": {}, "visuals": {}}


def read_or_create_review(path: Path, structure: dict, candidate_stage: dict) -> dict:
    path = Path(path)
    if not path.exists():
        review = new_review(structure, candidate_stage)
        write_json(path, review)
        return review
    review = read_json(path)
    assert_current(review, structure, candidate_stage)
    return review


def assert_current(review: dict, structure: dict, candidate_stage: dict) -> None:
    if (review.get("source_fingerprint") != evidence_fingerprint(structure)
            or review.get("candidate_fingerprint") != candidate_stage["candidate_fingerprint"]):
        raise ReviewRequired("Saved pre-LLM review is stale: the source evidence or ontology vocabulary changed. "
                             "Review changes before reusing approvals; do not silently overwrite the review.")


def set_evidence_decision(review: dict, evidence_id: str, decision: str, note: str = "") -> None:
    if evidence_id not in review["evidence"]:
        raise KeyError(f"Unknown evidence ID {evidence_id}")
    if decision not in EVIDENCE_DECISIONS:
        raise ValueError(f"Evidence decision must be one of {sorted(EVIDENCE_DECISIONS)}")
    review["evidence"][evidence_id] = {"decision": decision, "note": note.strip(),
                                        "review_status": "reviewed" if decision != "pending" else "not reviewed"}


def set_candidate_decision(review: dict, candidate_stage: dict, resolver, candidate_id: str,
                           decision: str, *, selected_group: str | None = None,
                           selected_iri: str | None = None,
                           term_override: str | None = None, note: str = "") -> None:
    if candidate_id not in review["candidates"]:
        raise KeyError(f"Unknown candidate ID {candidate_id}")
    if decision not in CANDIDATE_DECISIONS:
        raise ValueError(f"Candidate decision must be one of {sorted(CANDIDATE_DECISIONS)}")
    source = next((c for c in candidate_stage["candidates"] if c["id"] == candidate_id), None)
    if source is None:
        source = review.get("manual_candidates", {}).get(candidate_id)
    if source is None:
        raise KeyError(f"Candidate no longer belongs to this review: {candidate_id}")
    group = selected_group or source["group"]
    iri = selected_iri or None
    if group not in resolver.entries:
        raise ValueError(f"Invalid ontology group {group!r}")
    if iri and (group, iri) not in resolver.by_iri:
        raise ValueError(f"Not an active class in group {group!r}: {iri!r}")
    if decision in {"approve", "edit"} and not (iri or (term_override or source["term"]).strip()):
        raise ValueError("Supply a proposed term or an existing class IRI")
    if decision == "approve" and not iri:
        # Approval of a lexical suggestion without a canonical IRI is not approval
        # of an OWL mapping. Keep a reviewed unresolved term as an edit instead.
        raise ValueError("Approve requires a verified OWL IRI; use edit for unresolved terms")
    review["candidates"][candidate_id] = {
        "decision": decision, "selected_group": group if decision in ("approve", "edit") else None,
        "selected_iri": iri if decision in ("approve", "edit") else None,
        "term_override": (term_override or None) if decision in ("approve", "edit") else None,
        "note": note.strip(),
        "review_status": "reviewed" if decision != "pending" else "not reviewed"}


def add_manual_candidate(review: dict, structure: dict, resolver, evidence_id: str,
                         term: str, group: str) -> str:
    """Create a reviewer-supplied term, not a new OWL class or extracted fact."""
    if evidence_id not in review["evidence"]:
        raise KeyError(f"Unknown evidence ID {evidence_id}")
    if group not in resolver.entries:
        raise ValueError(f"Unknown active group {group}")
    term = term.strip()
    if not term:
        raise ValueError("A manual candidate needs a nonempty term")
    # Manual term is a reviewer hypothesis, not necessarily a literal text span.
    # Its provenance must remain explicit rather than pretending it was parsed.
    cid = "OC_MAN_" + hashlib.sha256(f"{evidence_id}|{group}|{normalize(term)}".encode()).hexdigest()[:18]
    review.setdefault("manual_candidates", {})[cid] = {
        "id": cid, "evidence_id": evidence_id,
        "paragraph_id": next(e["paragraph_id"] for e in structure["semantic_evidence"] if e["id"] == evidence_id),
        "term": term, "group": group, "detected_by": "human_added",
        "suggestions": [], "match_status": "human_proposal"}
    review["candidates"][cid] = {"decision": "pending", "selected_iri": None,
                                   "selected_group": group, "term_override": None, "note": ""}
    return cid


def set_visual_decision(review: dict, asset_id: str, decision: str) -> None:
    if decision not in VISUAL_DECISIONS:
        raise ValueError(f"Visual decision must be one of {sorted(VISUAL_DECISIONS)}")
    review.setdefault("visuals", {})[asset_id] = decision


def reviewed_selection(structure: dict, candidate_stage: dict, review: dict, resolver) -> dict:
    """Return only approved source sentences and reviewed candidate hints."""
    assert_current(review, structure, candidate_stage)
    lookup = {e["id"]: e for e in structure.get("semantic_evidence", [])}
    selected = {s: [lookup[k] for k, v in review.get("evidence", {}).items()
                    if v.get("decision") == s and k in lookup] for s in EVIDENCE_DECISIONS}
    if not selected["extract"]:
        raise ReviewRequired("No primary evidence has been approved. Set at least one sentence to 'extract' "
                             "before any paid LLM stage or cost estimation of approved content.")
    hints: dict[str, list[dict]] = {}
    approved_ids = {e["id"] for e in selected["extract"]}
    for c in [*candidate_stage["candidates"], *review.get("manual_candidates", {}).values()]:
        if c["evidence_id"] not in approved_ids:
            continue
        v = review["candidates"].get(c["id"], {})
        if v.get("decision") not in ("approve", "edit"):
            continue
        group = v.get("selected_group") or c["group"]
        iri = v.get("selected_iri")
        if iri and (group, iri) not in resolver.by_iri:
            raise ReviewRequired(f"Reviewer-selected ontology class is no longer active: {iri}")
        hints.setdefault(c["evidence_id"], []).append({
            "term": v.get("term_override") or c["term"], "original_term": c["term"],
            "group": group, "iri": iri,
            "label": resolver.by_iri[(group, iri)].label if iri else None,
            "kind": "reviewed_iri_hint" if iri else "reviewed_unmapped_term"})
    return {"extract": selected["extract"], "context_only": selected["context_only"],
            "hints": hints, "review_fingerprint": sha256_json(review),
            "approved_visuals": [k for k, v in review.get("visuals", {}).items() if v == "approve"],
            "counts": {k: len(v) for k, v in selected.items()}}


def approved_documents(structure: dict, selection: dict, max_chars: int = 120_000) -> list[dict]:
    """Build smaller prompt packets, preserving original source quotes and paragraph IDs.

    An explicitly reviewed context-only sentence can accompany a primary excerpt;
    it must not produce an independent assertion. The full original paragraph
    remains available to the downstream evidence validator.
    """
    sections = {s["id"]: s for s in structure.get("sections", [])}
    paras = {p["id"]: p for p in structure.get("paragraphs", [])}
    primaries = selection["extract"]
    context_ids = {e["id"] for e in selection["context_only"]}
    all_ev = structure.get("semantic_evidence", [])
    order = {e["id"]: i for i, e in enumerate(all_ev)}
    lookup = {e["id"]: e for e in all_ev}
    context_used = set()
    blocks = []
    for e in primaries:
        p = paras[e["paragraph_id"]]
        sec = sections.get(e["section_id"], {})
        block = [f"[{p['id']} | {sec.get('heading', '?')} ({p.get('role', '?')}) | page {p.get('page', '?')} | sentence {e['id']} | PRIMARY]",
                 e["sentence"]]
        for neighbor in (order[e["id"]] - 1, order[e["id"]] + 1):
            if 0 <= neighbor < len(all_ev):
                adjacent = all_ev[neighbor]
                if (adjacent["id"] in context_ids and adjacent["section_id"] == e["section_id"]
                        and adjacent["id"] not in context_used):
                    block += [f"[CONTEXT ONLY {adjacent['id']} | {adjacent['paragraph_id']}]: {adjacent['sentence']}"]
                    context_used.add(adjacent["id"])
        for hint in selection["hints"].get(e["id"], []):
            if hint["iri"]:
                block.append(f"[REVIEWED OWL CANDIDATE: {hint['term']} | {hint['group']} | {hint['label']} | {hint['iri']}; verify against evidence]")
            else:
                block.append(f"[REVIEWED UNMAPPED TERM: {hint['term']} | {hint['group']}; do not invent an IRI]")
        blocks.append((e, "\n".join(block)))
    docs, current, current_ids = [], [], []
    title = f"PAPER TITLE: {structure.get('title', '')}\nOnly PRIMARY passages can support extracted claims; CONTEXT ONLY passages clarify them.\n\n"
    for e, b in blocks:
        if current and len(title) + sum(len(s)+2 for s in current) + len(b) > max_chars:
            text = title + "\n\n".join(current)
            docs.append({"chunk": len(docs), "text": text, "paragraph_ids": list(dict.fromkeys(current_ids)), "chars": len(text)})
            current, current_ids = [], []
        current.append(b)
        current_ids.append(e["paragraph_id"])
    if current:
        text = title + "\n\n".join(current)
        docs.append({"chunk": len(docs), "text": text, "paragraph_ids": list(dict.fromkeys(current_ids)), "chars": len(text)})
    return docs


def review_summary(review: dict, candidates: dict) -> dict:
    ec = Counter(v.get("decision", "pending") for v in review.get("evidence", {}).values())
    cc = Counter(v.get("decision", "pending") for v in review.get("candidates", {}).values())
    return {"evidence": dict(sorted(ec.items())), "candidates": dict(sorted(cc.items())),
            "source_fingerprint": review.get("source_fingerprint"),
            "candidate_fingerprint": candidates.get("candidate_fingerprint")}


def review_widget(structure: dict, candidate_stage: dict, review_path: Path, resolver, _shared_review: dict | None = None):
    """Editable Jupyter evidence/candidate reviewer; requires optional ipywidgets.

    Every Save writes pre_llm_review.json. Original source text remains untouched.
    Candidate editor accepts an existing active OWL IRI, or an unmapped proposed term.
    """
    try:
        import ipywidgets as w
        from IPython.display import display
    except ImportError as exc:
        raise ImportError("Install ipywidgets to use the notebook editor. "
                          "All approval functions are also available directly in Python.") from exc
    review = _shared_review if _shared_review is not None else read_or_create_review(review_path, structure, candidate_stage)
    evidence = structure.get("semantic_evidence", [])
    if not evidence:
        raise ValueError("No sentence evidence to review")
    base_candidates = candidate_stage["candidates"]
    by_id = {e["id"]: e for e in evidence}
    summary = w.HTML()
    select = w.Dropdown(options=[(f"{e['id']} | {e['sentence'][:100]}", e["id"]) for e in evidence],
                        description="Sentence", layout=w.Layout(width="95%"))
    context = w.HTML()
    evidence_action = w.Dropdown(options=[("Pending", "pending"), ("Extract", "extract"),
                                          ("Context only", "context_only"), ("Exclude", "exclude")],
                                 description="Decision")
    evidence_note = w.Text(description="Note", layout=w.Layout(width="75%"))
    save_evidence = w.Button(description="Save sentence", button_style="success")
    candidate_select = w.Dropdown(options=[], description="Concept", layout=w.Layout(width="95%"))
    candidate_status = w.Dropdown(options=[("Pending", "pending"), ("Approve active OWL IRI", "approve"),
                                           ("Edit / unresolved", "edit"), ("Reject", "reject")],
                                  description="Decision")
    term_box = w.Text(description="Term", layout=w.Layout(width="85%"))
    group_box = w.Dropdown(options=sorted(resolver.entries.keys()), description="OWL group")
    candidate_suggestions = w.Dropdown(options=[("No selected class (unmapped)", "")],
                                       description="Suggested", layout=w.Layout(width="95%"))
    iri_box = w.Text(description="OWL IRI", layout=w.Layout(width="95%"))
    candidate_note = w.Text(description="Note", layout=w.Layout(width="85%"))
    save_candidate = w.Button(description="Save candidate", button_style="success")
    new_term = w.Text(description="Add term", layout=w.Layout(width="75%"))
    add_button = w.Button(description="Add candidate")
    visual_box = w.VBox([])
    response = w.Output()

    def persist():
        write_json(review_path, review)
        s = review_summary(review, candidate_stage)
        summary.value = (f"<b>Evidence decisions:</b> {s['evidence']} &nbsp; "
                         f"<b>Candidate decisions:</b> {s['candidates']} &nbsp; "
                         f"<b>Review file:</b> {review_path}")

    def chosen_candidates():
        return [c for c in [*base_candidates, *review.get("manual_candidates", {}).values()]
                if c["evidence_id"] == select.value]

    def show_candidate(*_):
        c = next((x for x in chosen_candidates() if x["id"] == candidate_select.value), None)
        if c is None:
            return
        stored = review["candidates"].get(c["id"], {})
        candidate_status.value = stored.get("decision", "pending")
        term_box.value = stored.get("term_override") or c["term"]
        group_box.value = stored.get("selected_group") or c["group"]
        opts = [("No selected class (unmapped)", "")]
        for item in c.get("suggestions", []):
            if item.get("iri"):
                opts.append((f"{item.get('label', '?')} | {item.get('group', c['group'])}", item["iri"]))
        candidate_suggestions.options = opts
        iri_box.value = stored.get("selected_iri") or c.get("matched_iri") or ""
        candidate_note.value = stored.get("note", "")

    def show_sentence(*_):
        e = by_id[select.value]
        stored = review["evidence"].get(e["id"], {})
        evidence_action.value = stored.get("decision", "pending")
        evidence_note.value = stored.get("note", "")
        # Escape any source text before embedding in the notebook HTML widget.
        import html
        pre, post = e.get("preceding_sentence"), e.get("following_sentence")
        context.value = (f"<div><b>{html.escape(str(e.get('section_role')))} | "
                         f"page {html.escape(str(e.get('page')))} | {html.escape(e['id'])}</b>"
                         f"<p>{html.escape(pre or '')}</p>"
                         f"<p><strong>{html.escape(e['sentence'])}</strong></p>"
                         f"<p>{html.escape(post or '')}</p></div>")
        opts = [(f"{c['term']} | {c['group']} | {c.get('match_status','')}", c["id"])
                for c in chosen_candidates()]
        candidate_select.options = opts
        if opts:
            candidate_select.value = opts[0][1]
            show_candidate()
        else:
            term_box.value = ""
            iri_box.value = ""

    def on_save_evidence(_):
        with response:
            response.clear_output()
            try:
                set_evidence_decision(review, select.value, evidence_action.value, evidence_note.value)
                persist()
                print("Saved sentence", select.value)
            except Exception as exc:
                print("Not saved:", exc)

    def on_save_candidate(_):
        with response:
            response.clear_output()
            try:
                if candidate_select.value is None:
                    raise ValueError("Add or select a candidate first")
                c = next(x for x in chosen_candidates() if x["id"] == candidate_select.value)
                original = c["term"]
                set_candidate_decision(review, candidate_stage, resolver, c["id"], candidate_status.value,
                                       selected_group=group_box.value, selected_iri=iri_box.value.strip() or None,
                                       term_override=term_box.value if term_box.value != original else None,
                                       note=candidate_note.value)
                persist()
                print("Saved concept", c["id"])
            except Exception as exc:
                print("Not saved:", exc)

    def on_add(_):
        with response:
            response.clear_output()
            try:
                cid = add_manual_candidate(review, structure, resolver, select.value, new_term.value, group_box.value)
                persist()
                show_sentence()
                candidate_select.value = cid
                print("Added term for review", cid)
            except Exception as exc:
                print("Not added:", exc)

    def on_suggestion(change):
        if change.get("name") == "value" and change.get("new"):
            iri_box.value = change["new"]

    select.observe(show_sentence, names="value")
    candidate_select.observe(show_candidate, names="value")
    candidate_suggestions.observe(on_suggestion, names="value")
    save_evidence.on_click(on_save_evidence)
    save_candidate.on_click(on_save_candidate)
    add_button.on_click(on_add)

    # Vision approval has a distinct gate. Source paragraph IDs are the stable
    # visual asset keys used by the Docling and pdfplumber extraction backends.
    assets = [("table", x) for x in structure.get("tables", [])] + [
        ("figure", x) for x in structure.get("figures", [])]
    visual_children = []
    for kind, asset in assets:
        asset_id = asset.get("paragraph_id")
        if not asset_id:
            continue
        label = w.Label(f"{kind}: {asset_id} (page {asset.get('page', '?')})")
        drop = w.Dropdown(options=[("Do not call vision", "pending"), ("Approve vision", "approve"),
                                   ("Reject vision", "reject")],
                          value=review.get("visuals", {}).get(asset_id, "pending"))
        def on_visual_change(change, aid=asset_id):
            if change["name"] == "value":
                set_visual_decision(review, aid, change["new"])
                persist()
        drop.observe(on_visual_change, names="value")
        visual_children.append(w.HBox([label, drop]))
    visual_box.children = visual_children or [w.Label("No detected table or figure crops")]
    show_sentence()
    persist()
    return w.VBox([summary, select, context,
                   w.HBox([evidence_action, evidence_note, save_evidence]),
                   w.HTML("<b>Ontology candidate editor (manual IRIs checked against the active OWL):</b>"),
                   candidate_select, candidate_status, term_box, group_box,
                   candidate_suggestions, iri_box, candidate_note,
                   w.HBox([save_candidate, new_term, add_button]),
                   w.HTML("<b>Vision calls (opt in per crop):</b>"), visual_box, response])


def apply_reviewed_mappings(aligned: dict, selection: dict, resolver) -> dict:
    """Use explicit reviewer OWL choices when the extracted term and source quote agree.

    This updates the corresponding Pydantic record and resolver audit, not the
    ontology. It never asserts OWL equivalence: status is human_reviewed_mapping.
    Other scientific validation (numbers, causality, evidence) still applies.
    """
    # Only active IRIs, reviewer-approved before the model, and source-grounded
    # terms from approved PRIMARY sentences can override a model's class guess.
    evidence = {e["id"]: e for e in selection["extract"]}
    reviewed = []
    for evid, hints in selection["hints"].items():
        if evid not in evidence:
            continue
        for h in hints:
            iri, group = h.get("iri"), h["group"]
            if iri and h.get('kind') == 'reviewed_iri_hint' and (group, iri) in resolver.by_iri:
                reviewed.append((evidence[evid], h))
    if not reviewed:
        return aligned

    p1, p2, p3 = aligned["pass1"], aligned["pass2"], aligned["pass3"]
    by_role = {}
    def add(role, records):
        for obj in records:
            by_role[obj["id"], role] = obj
    add("material", p1.get("materials", []))
    add("sample", p1.get("samples", []))
    add("process", p1.get("process_steps", []))
    for proc in p1.get("process_steps", []):
        add("parameter", proc.get("parameters", []))
    add("measurement_property", p2.get("measurements", []))
    add("measurement_technique", p2.get("measurements", []))
    add("application", p3.get("applications", []))
    add("scientific_entity", p3.get("entities", []))
    fields = {"material": ("ontology_class", "ontology_iri"),
              "sample": ("ontology_class", "ontology_iri"),
              "process": ("ontology_class", "ontology_iri"),
              "parameter": ("attribute_class", "ontology_iri"),
              "measurement_property": ("property_class", "property_iri"),
              "measurement_technique": ("technique_class", "technique_iri"),
              "application": ("ontology_class", "ontology_iri"),
              "scientific_entity": ("ontology_class", "ontology_iri")}
    for report in aligned.get("ontology_resolution", []):
        if report.get("role") not in fields:
            continue
        obj = by_role.get((report["id"], report["role"]))
        if obj is None:
            continue
        eligible = []
        for ev, hint in reviewed:
            if hint["group"] != report.get("group"):
                continue
            if normalize(hint["original_term"]) != normalize(report.get("term", "")):
                continue
            if any(q.get("paragraph_id") == ev["paragraph_id"]
                   and isinstance(q.get("quote"), str)
                   and q["quote"].strip() in ev["sentence"]
                   for q in obj.get("evidence", [])):
                eligible.append((ev, hint))
        if len({h["iri"] for _, h in eligible}) != 1:
            continue  # conflicts/ambiguity still require review
        ev, hint = eligible[0]
        label_field, iri_field = fields[report["role"]]
        obj[label_field] = hint["label"]
        obj[iri_field] = hint["iri"]
        report.update(status="human_reviewed_mapping", requires_review=False,
                      matched_iri=hint["iri"], matched_label=hint["label"],
                      method="explicit_reviewed_source_grounded_class_choice",
                      review_evidence_id=ev["id"],
                      reason="Reviewer chose an active OWL class for this exact source concept; scientific truth not verified")
    aligned["ontology_resolution_summary"] = dict(sorted(
        Counter(r["status"] for r in aligned.get("ontology_resolution", [])).items()))
    return aligned

# ---------------------------------------------------------------------------
# v0.6: paper-local concepts and hierarchical review. Keep the existing v0.5
# sentence editor and all public functions backward compatible.
# ---------------------------------------------------------------------------
from .scientific_context import local_acronyms, safe_context
from collections import defaultdict as _defaultdict
import copy as _copy

_make_candidates_v05 = make_candidates
_new_review_v05 = new_review
_reviewed_selection_v05 = reviewed_selection
_review_widget_v05 = review_widget
_review_summary_v05 = review_summary


def make_candidates(structure: dict, resolver) -> dict:
    result = _make_candidates_v05(structure, resolver)
    evidence = structure.get('semantic_evidence', [])
    acronyms = local_acronyms(evidence)
    defined = {}
    for row in acronyms:
        defined.setdefault(row['acronym'].lower(),row)
    registry = {}
    by_id = {e['id']:e for e in evidence}
    kept=[]
    filtered=[]
    for c in result['candidates']:
        ev=by_id.get(c['evidence_id'],{})
        context=safe_context(c['term'], c['group'], ev.get('sentence',''))
        if not context['allowed']:
            filtered.append({'candidate_id':c['id'],'term':c['term'],'reason':context['reason']})
            continue
        term=c['term'].strip()
        local=defined.get(term.lower())
        expanded=local['expansion'] if local else term
        # Normalize local acronym + its expansion to one canonical scientific concept.
        canon=normalize(expanded).replace('anti localization','antilocalization')
        group=c['group']
        # Select category using scientific expansion rather than bare acronym.
        if local:
            if re.search(r'(magnetoresistan|resistivity ratio|magnetoconductiv)',canon):group='property'
            elif 'anti' in canon and 'localiz' in canon:group='phenomenon'
            elif 'surface state' in canon:group='structure' if 'structure' in resolver.entries else c['group']
            elif 'hikari' in canon or 'hln' in canon or 'larkin' in canon:group='theory_model' if 'theory_model' in resolver.entries else c['group']
        # Resolve canonical term against the ontology, not the acronym's spelling.
        if local and group in resolver.entries:
            r=resolver.resolve(expanded,group,context=ev.get('sentence',''))
            c['group']=group
            c['match_status']=r['status']
            c['matched_iri']=r.get('matched_iri')
            c['suggestions']=([{'iri':r['matched_iri'],'label':r['matched_label'],'group':group,'relation':'document_defined_acronym'}] if r.get('matched_iri') else [])+[
                {'group':group,**suggestion} for suggestion in r.get('candidates',[])]
            c['suggestions']=c['suggestions'][:6]
        cid='CC_'+hashlib.sha256((group+'|'+canon).encode()).hexdigest()[:18]
        c['concept_id']=cid
        c['canonical_term']=expanded
        c['context_filter']=context
        item=registry.setdefault(cid,{'id':cid,'canonical_term':expanded,'group':group,
                                      'aliases':[],'mention_candidate_ids':[], 'evidence_ids':[],
                                      'matched_iri':None,'match_status':c['match_status'],
                                      'suggestions':[],'review_status':'not reviewed'})
        item['aliases'].append(term)
        item['mention_candidate_ids'].append(c['id'])
        item['evidence_ids'].append(c['evidence_id'])
        if c.get('matched_iri'):
            item['matched_iri']=c['matched_iri'];item['match_status']='exact_match'
        if c.get('suggestions') and not item['suggestions']:item['suggestions']=c['suggestions']
        kept.append(c)
    for c in registry.values():
        c['aliases']=sorted(set(c['aliases']),key=lambda s:s.lower())
        c['evidence_ids']=list(dict.fromkeys(c['evidence_ids']))
        c['mention_candidate_ids']=list(dict.fromkeys(c['mention_candidate_ids']))
    result['candidates']=kept
    result['acronyms']=acronyms
    result['concept_registry']=list(registry.values())
    result['evidence_blocks']=structure.get('evidence_blocks',[])
    result['context_filtered']=filtered
    result['source_assertions']=structure.get('source_assertions',[])
    result['document_reference_targets']=structure.get('document_reference_targets',[])
    result['relation_candidates']=structure.get('relation_candidates',[])
    result['specimen_mentions']=structure.get('specimen_mentions',[])
    result['discourse_routes_summary']=structure.get('claim_provenance_summary',{}).get('discourse_mention_routes',{})
    result['summary']=dict(sorted(Counter(x['match_status'] for x in kept).items()))
    result['version']='2.0'
    result['candidate_fingerprint']=sha256_json({k:v for k,v in result.items() if k!='candidate_fingerprint'})
    return result


def new_review(structure:dict, candidate_stage:dict)->dict:
    out=_new_review_v05(structure,candidate_stage)
    out['review_version']='2.0'
    out['blocks']={b['id']:{'decision':'pending','note':''} for b in candidate_stage.get('evidence_blocks',[])}
    out['concepts']={c['id']:{'decision':'pending','selected_iri':None,'selected_group':None,
                              'term_override':None,'note':''} for c in candidate_stage.get('concept_registry',[])}
    out['assertions']={a['id']:{'decision':'pending','claim_origin_override':None,'note':''}
                       for a in candidate_stage.get('source_assertions',[])}
    out['relation_candidates']={x['id']:{'decision':'pending','note':''}
                       for x in candidate_stage.get('relation_candidates',[])}
    for kind in ('evidence','candidates','blocks','concepts','assertions','relation_candidates'):
        for item in out.get(kind, {}).values():
            item['review_status'] = 'not reviewed'
    out['source_snapshot'] = _refresh_snapshot(structure, candidate_stage)
    out['resolver_fingerprint'] = candidate_stage.get('resolver_fingerprint')
    return out


def set_block_decision(review:dict, block_id:str, decision:str, note:str='')->None:
    if decision not in EVIDENCE_DECISIONS:raise ValueError(f'Unsupported block decision: {decision}')
    if block_id not in review.get('blocks',{}):raise KeyError(block_id)
    review['blocks'][block_id]={'decision':decision,'note':note.strip(),
                                'review_status':'reviewed' if decision!='pending' else 'not reviewed'}


def set_concept_decision(review:dict, candidate_stage:dict, resolver, concept_id:str,decision:str,
                         *,selected_group:str|None=None,selected_iri:str|None=None,
                         term_override:str|None=None,note:str='')->None:
    c=next((x for x in candidate_stage.get('concept_registry',[]) if x['id']==concept_id),None)
    if c is None or concept_id not in review.get('concepts',{}):raise KeyError(concept_id)
    if decision not in CANDIDATE_DECISIONS:raise ValueError(decision)
    group=selected_group or c['group'];iri=selected_iri or None
    if group not in resolver.entries:raise ValueError('Unknown OWL group: '+group)
    if iri and (group,iri) not in resolver.by_iri:raise ValueError('IRI is not an active class of '+group)
    if decision=='approve' and not iri:raise ValueError('Approval of an OWL class requires an active IRI')
    if decision=='edit' and not ((term_override or c['canonical_term']).strip()):raise ValueError('A term is required')
    review['concepts'][concept_id]={'decision':decision,'selected_group':group if decision in ('approve','edit') else None,
                                   'selected_iri':iri if decision in ('approve','edit') else None,
                                   'term_override':term_override or None,'note':note.strip(),
                                   'review_status':'reviewed' if decision!='pending' else 'not reviewed'}


def set_assertion_decision(review:dict, assertion_id:str, decision:str,
                           *, claim_origin_override:str|None=None, note:str='')->None:
    """Review one claim's paper-of-origin without changing the source quote."""
    if assertion_id not in review.get('assertions',{}):raise KeyError(assertion_id)
    if decision not in {'pending','approve','reject','edit'}:raise ValueError(decision)
    if claim_origin_override not in {None,'current_study','cited_study',
            'background_knowledge','undetermined','mixed_current_and_cited'}:
        raise ValueError('Invalid assertion origin')
    if decision=='edit' and claim_origin_override is None:
        raise ValueError('An origin override is needed when editing')
    review['assertions'][assertion_id]={'decision':decision,
        'claim_origin_override':claim_origin_override if decision in ('approve','edit') else None,
        'note':note.strip(),'review_status':'reviewed' if decision!='pending' else 'not reviewed'}


def set_relation_decision(review:dict,relation_id:str,decision:str,note:str='')->None:
    if relation_id not in review.get('relation_candidates',{}):raise KeyError(relation_id)
    if decision not in {'pending','approve','reject'}:raise ValueError(decision)
    review['relation_candidates'][relation_id]={'decision':decision,'note':note.strip(),
        'review_status':'reviewed' if decision!='pending' else 'not reviewed'}


def reviewed_selection(structure:dict,candidate_stage:dict,review:dict,resolver)->dict:
    """Apply concept/block choices to their members; explicit leaf overrides win."""
    assert_current(review,structure,candidate_stage)
    # Safe refresh creates a wholly new, unreviewed manifest; unconfirmed legacy
    # proposals remain inert. Source packets may still be previewed independently.
    working=_copy.deepcopy(review)
    for b in candidate_stage.get('evidence_blocks',[]):
        choice=review.get('blocks',{}).get(b['id'],{}).get('decision','pending')
        if choice not in EVIDENCE_DECISIONS or choice=='pending':continue
        for evid in b['evidence_ids']:
            if working['evidence'].get(evid,{}).get('decision','pending')=='pending':
                working['evidence'][evid]={'decision':choice,'note':'inherited from '+b['id']}
    for c in candidate_stage.get('candidates',[]):
        if working['candidates'].get(c['id'],{}).get('decision','pending')!='pending':continue
        root=review.get('concepts',{}).get(c.get('concept_id'),{})
        if root.get('decision') not in ('approve','reject','edit'):continue
        working['candidates'][c['id']]={'decision':root['decision'],
                                       'selected_group':root.get('selected_group'),
                                       'selected_iri':root.get('selected_iri'),
                                       'term_override':root.get('term_override'),
                                       'note':'inherited from concept '+c.get('concept_id','')}
    # v0.8: evidence inclusion is independent of human review status. By default
    # include body/caption evidence; do not include bibliography, headers or furniture.
    # Explicit sentence choices override blocks; explicit exclusions always win.
    for ev in structure.get('semantic_evidence', []):
        eid=ev['id']
        if working.get('evidence', {}).get(eid, {}).get('decision', 'pending') == 'pending':
            working['evidence'][eid] = {
                'decision': default_evidence_decision(ev),
                'note': 'automatic source policy; not human-reviewed',
                'review_status': 'not reviewed'}
    result=_reviewed_selection_v05(structure,candidate_stage,working,resolver)
    # Fingerprint actual review and inherited decisions to force fresh LLM input.
    result['review_fingerprint']=sha256_json({'policy':'automatic_evidence_v08', 'saved':review,
                                            'effective':working['evidence'],
                                            'effective_candidates':working['candidates']})
    result['counts']['auto_included'] = sum(1 for e in result['extract']
        if (review.get('evidence', {}).get(e['id'], {}).get('decision', 'pending') == 'pending'
            and review.get('blocks', {}).get('EB_'+e['paragraph_id'], {}).get('decision', 'pending') == 'pending'))
    result['counts']['needs_review'] = len(review_issues(structure, candidate_stage)['issue_ids'])
    for name in ('extract', 'context_only'):
        result[name] = [dict(e, review_status=('reviewed' if
            review_state(review.get('evidence', {}).get(e['id'], {})) == 'reviewed' or
            review_state(review.get('blocks', {}).get('EB_'+e['paragraph_id'], {})) == 'reviewed'
            else 'not reviewed')) for e in result[name]]
    primary_ids={e['id'] for e in result['extract']}
    approved_context_ids=primary_ids|{e['id'] for e in result['context_only']}
    approved_concepts={c['id'] for c in candidate_stage.get('concept_registry',[])
                       if review.get('concepts',{}).get(c['id'],{}).get('decision') in ('approve','edit')}
    result['acronyms']=[a for a in candidate_stage.get('acronyms',[])
                        if any(eid in primary_ids for eid in a['mentions'])
                        and (any(eid in approved_context_ids for eid in a['definition_evidence_ids'])
                             or any(c['id'] in approved_concepts and
                                    (a['acronym'] in c['aliases'] or normalize(a['expansion'])==normalize(c['canonical_term']))
                                    for c in candidate_stage.get('concept_registry',[])))]
    result['source_assertions']=[
        {**a, 'claim_origin': (
             review.get('assertions',{}).get(a['id'],{}).get('claim_origin_override') or a['claim_origin']),
         'origin_review':review_state(review.get('assertions',{}).get(a['id'],{})),
         'origin_decision':review.get('assertions',{}).get(a['id'],{}).get('decision','pending')}
        for a in candidate_stage.get('source_assertions',[])
        if set(a['evidence_ids']).issubset(primary_ids)]
    result['document_references']=[r for r in structure.get('document_references',[])
        if r['evidence_id'] in approved_context_ids]
    result['specimen_mentions']=[s for s in candidate_stage.get('specimen_mentions',[])
        if s['evidence_id'] in approved_context_ids]
    result['relation_candidates']=[{**r, 'review_decision':review.get('relation_candidates',{})
        .get(r['id'],{}).get('decision','pending')} for r in candidate_stage.get('relation_candidates',[])
        if set(r['evidence_ids']).issubset(primary_ids)]
    result['concepts']=[{'id':c['id'],'canonical_term':c['canonical_term'],'group':c['group'],
                         'aliases':c['aliases'],'selected_iri':review.get('concepts',{}).get(c['id'],{}).get('selected_iri')}
                        for c in candidate_stage.get('concept_registry',[])
                        if any(e in {x['id'] for x in result['extract']} for e in c['evidence_ids'])]
    return result


def review_summary(review:dict,candidates:dict)->dict:
    out=_review_summary_v05(review,candidates)
    out['blocks']=dict(Counter(x.get('decision','pending') for x in review.get('blocks',{}).values()))
    out['concepts']=dict(Counter(x.get('decision','pending') for x in review.get('concepts',{}).values()))
    out['assertions']=dict(Counter(x.get('decision','pending') for x in review.get('assertions',{}).values()))
    out['relation_candidates']=dict(Counter(x.get('decision','pending') for x in review.get('relation_candidates',{}).values()))
    out['claim_origins']=dict(Counter(a['claim_origin'] for a in candidates.get('source_assertions',[])))
    out['review_status'] = {kind: dict(Counter(
        review_state(v) for v in review.get(kind, {}).values()))
        for kind in ('evidence','blocks','concepts','candidates','assertions','relation_candidates')}
    out['inclusion_policy'] = 'automatic source evidence; explicit review optional'
    if review.get('refresh'):
        out['refresh']={'status':review['refresh'].get('status'),
                        'counts':review['refresh'].get('counts',{}),
                        'backup_file':review['refresh'].get('backup_file')}
    return out


def review_widget(structure:dict,candidate_stage:dict,review_path:Path,resolver):
    """Hierarchy-first Jupyter panel, legacy per-sentence editor kept for exceptions."""
    try:
        import ipywidgets as w
    except ImportError as exc:
        raise ImportError('Install ipywidgets for hierarchical notebook review') from exc
    review=read_or_create_review(review_path,structure,candidate_stage)
    if review.get('refresh', {}).get('status') == 'awaiting_confirmation':
        # Do not build editable controls from a stale in-memory review: saving
        # them after confirmation could overwrite the newly confirmed file.
        from html import escape
        counts=review['refresh'].get('counts',{})
        verified=sum(v.get('verified',0) for v in counts.values())
        manual=sum(v.get('manual_check',0) for v in counts.values())
        backup=review['refresh'].get('backup_file','(unknown)')
        overview=w.HTML('<b>Saved review was stale and has been safely backed up.</b> '
                        f'Old file: {escape(str(backup))}. '
                        f'{verified} exact matches can be reused after confirmation; '
                        f'{manual} need individual review. Vision approvals always remain pending. '
                        'No LLM or vision call is permitted until migration is confirmed.')
        rows=[]
        for kind, items in review['refresh'].get('proposals',{}).items():
            for current_id,p in items.items():
                if p.get('verified'):continue
                prior=p.get('previous_source') or {}
                new=p.get('current_source') or {}
                before=str(prior.get('sentence') or prior.get('canonical_term') or prior.get('term') or '')
                after=str(new.get('sentence') or new.get('canonical_term') or new.get('term') or '')
                rows.append('<tr><td>'+escape(kind)+'</td><td>'+escape(current_id)+'</td><td>'+
                            escape(p.get('change_kind','unverifiable'))+'</td><td>'+
                            escape(before[:180])+'</td><td>'+escape(after[:180])+'</td></tr>')
                if len(rows)>=16:break
            if len(rows)>=16:break
        detail=(w.HTML('<b>First unverified or changed proposals (full report is saved in pre_llm_review.json):</b>'
                       '<table style="width:100%;font-size:12px"><tr><th>Type</th><th>ID</th><th>Status</th>'
                       '<th>Previous</th><th>Current</th></tr>'+''.join(rows)+'</table>')
                if rows else w.HTML('No unverified source proposals.'))
        confirm_button=w.Button(description='Confirm verified matches only',button_style='warning')
        empty_button=w.Button(description='Start fresh; restore none')
        feedback=w.Output()
        def complete(accept):
            with feedback:
                feedback.clear_output()
                try:
                    result=confirm_refresh(review_path,structure,candidate_stage,resolver,
                                           accept_verified=accept)
                    print('Review migration confirmed. Adopted:',result['applied'])
                    print('Rerun this notebook cell to inspect and edit pending records.')
                    confirm_button.disabled=True;empty_button.disabled=True
                except Exception as exc:
                    print('Could not confirm refresh:',exc)
        confirm_button.on_click(lambda _:complete(True))
        empty_button.on_click(lambda _:complete(False))
        return w.VBox([overview,detail,w.HBox([confirm_button,empty_button]),feedback])
    blocks=candidate_stage.get('evidence_blocks',[])
    concepts=candidate_stage.get('concept_registry',[])
    block_select=w.Dropdown(options=[(f"{b['id']} {b['section_role']} | {b['page']}",b['id']) for b in blocks],
                            description='Block',layout=w.Layout(width='95%'))
    block_action=w.Dropdown(options=sorted(EVIDENCE_DECISIONS),description='Decision')
    block_preview=w.HTML()
    concept_select=w.Dropdown(options=[(f"{c['canonical_term']} ({c['group']}, {len(c['evidence_ids'])} mentions)",c['id']) for c in concepts],
                              description='Concept',layout=w.Layout(width='95%'))
    concept_action=w.Dropdown(options=sorted(CANDIDATE_DECISIONS),description='Decision')
    group_box=w.Dropdown(options=sorted(resolver.entries),description='Group')
    term_box=w.Text(description='Canonical',layout=w.Layout(width='95%'))
    iri_box=w.Text(description='Class IRI',layout=w.Layout(width='95%'))
    feedback=w.Output()
    from html import escape
    evidence_lookup={e['id']:e for e in structure.get('semantic_evidence',[])}
    def show_block(*_):
        b=next((x for x in blocks if x['id']==block_select.value),None)
        if b is None:return
        block_action.value=review.get('blocks',{}).get(b['id'],{}).get('decision','pending')
        block_preview.value='<div>'+''.join('<p>'+escape(evidence_lookup[i]['sentence'])+'</p>' for i in b['evidence_ids'][:12] if i in evidence_lookup)+'</div>'
    def show_concept(*_):
        c=next((x for x in concepts if x['id']==concept_select.value),None)
        if c is None:return
        decision=review.get('concepts',{}).get(c['id'],{})
        concept_action.value=decision.get('decision','pending')
        group_box.value=decision.get('selected_group') or c['group']
        term_box.value=decision.get('term_override') or c['canonical_term']
        iri_box.value=decision.get('selected_iri') or c.get('matched_iri') or ''
    def save_block(_):
        with feedback:
            feedback.clear_output()
            try:
                set_block_decision(review,block_select.value,block_action.value)
                write_json(review_path,review);print('Saved block',block_select.value)
            except Exception as e:print(e)
    def save_concept(_):
        with feedback:
            feedback.clear_output()
            try:
                c=next(x for x in concepts if x['id']==concept_select.value)
                set_concept_decision(review,candidate_stage,resolver,c['id'],concept_action.value,
                                     selected_group=group_box.value,selected_iri=iri_box.value.strip() or None,
                                     term_override=term_box.value.strip() if term_box.value.strip()!=c['canonical_term'] else None)
                write_json(review_path,review);print('Saved concept',c['id'])
            except Exception as e:print(e)
    bs=w.Button(description='Save block',button_style='success')
    cs=w.Button(description='Save concept',button_style='success')
    bs.on_click(save_block);cs.on_click(save_concept)
    block_select.observe(show_block,names='value')
    concept_select.observe(show_concept,names='value')
    show_block();show_concept()
    hierarchical=w.VBox([w.HTML('<b>Review evidence blocks first, then canonical scientific concepts.</b>'),
                         block_select,block_preview,block_action,bs,
                         w.HTML('<b>One concept choice applies to its repeated mentions unless overridden.</b>'),
                         concept_select,concept_action,term_box,group_box,iri_box,cs,feedback])
    # v0.5 per-sentence editor still handles exceptional source-specific overrides and vision approvals.
    legacy=_review_widget_v05(structure,candidate_stage,review_path,resolver,_shared_review=review)
    assertions=candidate_stage.get('source_assertions',[])
    # Evidence blocks remain the primary review unit. A third tab handles
    # attribution for exceptional mixed/current/cited assertions.
    if assertions:
        ar_select=w.Dropdown(options=[(f"{a['source_section_role']} | {a['claim_origin']} | {a['clause'][:80]}",a['id'])
                    for a in assertions],description='Claim',layout=w.Layout(width='95%'))
        ar_status=w.Dropdown(options=['pending','approve','reject','edit'],description='Review')
        ar_origin=w.Dropdown(options=['current_study','cited_study','background_knowledge',
                    'undetermined','mixed_current_and_cited'],description='Origin')
        ar_preview=w.HTML()
        ar_feedback=w.Output()
        def show_assertion(*_):
            a=next((x for x in assertions if x['id']==ar_select.value),None)
            if a is None:return
            v=review.get('assertions',{}).get(a['id'],{})
            ar_status.value=v.get('decision','pending')
            ar_origin.value=v.get('claim_origin_override') or a['claim_origin']
            ar_preview.value='<b>Source:</b> '+escape(a['clause'])+'<p><b>Origin cue:</b> '+escape(a['origin_reason'])+'</p><p>Reference mentions: '+escape(', '.join(a['reference_ids']))+'</p>'
        def save_assertion(_):
            with ar_feedback:
                ar_feedback.clear_output()
                try:
                    set_assertion_decision(review,ar_select.value,ar_status.value,
                        claim_origin_override=ar_origin.value if ar_status.value=='edit' else None)
                    write_json(review_path,review)
                    print('Saved claim attribution',ar_select.value)
                except Exception as exc:print('Review error:',exc)
        ar_select.observe(show_assertion,names='value')
        ar_save=w.Button(description='Save claim attribution',button_style='success')
        ar_save.on_click(save_assertion)
        show_assertion()
        refs=candidate_stage.get('document_reference_targets',[])
        refs_box=w.HTML('<b>Grouped document references:</b> '+escape('; '.join(
            f"{r['kind']} {r['target_label']} ({len(r['mention_ids'])} mentions)" for r in refs[:60])))
        attribution_tab=w.VBox([w.HTML("<b>Claims are source-backed candidates, not verified facts. Choose origin explicitly for mixed or misattributed claims.</b>"),
            refs_box,ar_select,ar_preview,ar_status,ar_origin,ar_save,ar_feedback])
        tabs=w.Tab(children=[hierarchical,attribution_tab,legacy])
        tabs.set_title(0,'Blocks & concepts')
        tabs.set_title(1,'Claims & citations')
        tabs.set_title(2,'Sentence overrides & visuals')
    else:
        tabs=w.Tab(children=[hierarchical,legacy])
        tabs.set_title(0,'Blocks & concepts')
        tabs.set_title(1,'Sentence overrides & visuals')
    return tabs


_approved_documents_v05 = approved_documents

def approved_documents(structure:dict, selection:dict, max_chars:int=120_000)->list[dict]:
    docs=_approved_documents_v05(structure,selection,max_chars)
    acronym_lines=[]
    for a in selection.get('acronyms',[]):
        acronym_lines.append(f"{a['acronym']} = {a['expansion']} [document-defined, source: {','.join(a['definition_evidence_ids'])}; verify source]")
    if selection.get('source_assertions'):
        # Compact claim-origin hints. Source text already lives in evidence
        # packets: do not repeat full quotes (token waste or truncated input).
        for d in docs:
            allowed=set(d.get('paragraph_ids',[]))
            local=[]
            for a in selection['source_assertions']:
                if a.get('paragraph_id') not in allowed:continue
                important=(a.get('claim_origin') in ('mixed_current_and_cited','cited_study')
                           or a.get('claim_kind') in ('comparison','interpretation')
                           or a.get('origin_review') == 'reviewed')
                if important:
                    local.append(f"{a['id']} [{','.join(a['evidence_ids'])}]: "
                                 f"origin={a['claim_origin']}; kind={a['claim_kind']}; "
                                 f"review={a.get('origin_review','pending')}")
            if local:
                d['text']='PREFLIGHT ORIGIN CANDIDATES (not verified):\n'+\
                          '\n'.join(local[:40])+'\n\n'+d['text']
                d['chars']=len(d['text'])
    if acronym_lines:
        glossary='DOCUMENT-LOCAL SCIENTIFIC ACRONYMS (definitions from source, not inferred OWL equivalences):\n'+\
                 '\n'.join(acronym_lines)+'\n\n'
        for d in docs:
            d['text']=glossary+d['text'];d['chars']=len(d['text'])
    return docs

# ---------------------------------------------------------------------------
# v0.6 safe review migration: stale stage fingerprints need not discard reviewer
# work, but an ID alone does not establish that a scientific sentence is equal.
# ---------------------------------------------------------------------------
import os as _os
import shutil as _shutil
from datetime import datetime as _datetime, timezone as _timezone


def _refresh_snapshot(structure: dict, candidate_stage: dict) -> dict:
    """Small, source-grounded fingerprints saved with NEW reviews for future diffs.

    Legacy v0.5/v0.6 reviews lack this snapshot. In that case reusing an ID
    after a source change is only an *unverified suggestion*, never approval.
    """
    paragraphs = {p['id']: p for p in structure.get('paragraphs', [])}
    evidence = {}
    for e in structure.get('semantic_evidence', []):
        p = paragraphs.get(e.get('paragraph_id'), {})
        evidence[e['id']] = {
            'paragraph_id': e.get('paragraph_id'),
            'parent_sha256': hashlib.sha256(p.get('text', '').encode('utf-8')).hexdigest(),
            'section_id': e.get('section_id'),
            'section_role': e.get('section_role'),
            'source_kind': e.get('source_kind'),
            'page': e.get('page'),
            'sentence': e.get('sentence', ''),
        }
    # Do not use sentence *ID* in the signature: segmentation revisions can
    # renumber sentences without changing the exact original source statement.
    def evsig(eid):
        return sha256_json(evidence[eid]) if eid in evidence else None
    blocks = {b['id']: {
        'members': sorted(evsig(i) for i in b.get('evidence_ids', []) if evsig(i)),
        'section_role': b.get('section_role'), 'page': b.get('page'),
    } for b in candidate_stage.get('evidence_blocks', [])}
    candidates = {c['id']: {
        'evidence': evsig(c.get('evidence_id')),
        'term': c.get('term'), 'group': c.get('group'),
        'matched_iri': c.get('matched_iri'),
        'suggestions': sorted([[x.get('group'), x.get('iri')] for x in c.get('suggestions', [])
                               if x.get('group') and x.get('iri')]),
    } for c in candidate_stage.get('candidates', [])}
    concepts = {c['id']: {
        'canonical_term': c.get('canonical_term'), 'group': c.get('group'),
        'evidence': sorted(evsig(i) for i in c.get('evidence_ids', []) if evsig(i)),
        'matched_iri': c.get('matched_iri'),
        'aliases': sorted(normalize(a) for a in c.get('aliases', [])),
    } for c in candidate_stage.get('concept_registry', [])}
    visuals = {a['paragraph_id']: {
        'paragraph_sha256': hashlib.sha256(paragraphs.get(a['paragraph_id'], {}).get('text', '').encode('utf-8')).hexdigest(),
        'page': a.get('page'), 'caption': a.get('caption'), 'crop': a.get('crop'),
    } for a in [*structure.get('tables', []), *structure.get('figures', [])]
               if a.get('paragraph_id')}
    return {'evidence': evidence, 'blocks': blocks, 'candidates': candidates,
            'concepts': concepts, 'visuals': visuals}


def _refresh_proposals(previous: dict, fresh: dict, resolver=None) -> dict:
    """Produce reviewer-visible proposals. No decision is copied here."""
    old_snapshot = previous.get('source_snapshot', {})
    new_snapshot = fresh['source_snapshot']
    old_source_same = previous.get('source_fingerprint') == fresh.get('source_fingerprint')
    old_candidates_same = previous.get('candidate_fingerprint') == fresh.get('candidate_fingerprint')
    ontology_unchanged = (old_candidates_same or
                          (bool(previous.get('resolver_fingerprint')) and
                           previous.get('resolver_fingerprint') == fresh.get('resolver_fingerprint')))
    categories = ('evidence', 'blocks', 'candidates', 'concepts', 'visuals')
    proposals, removed = {}, {}
    for kind in categories:
        old_choices = previous.get(kind, {})
        new_choices = fresh.get(kind, {})
        old_records = old_snapshot.get(kind, {})
        new_records = new_snapshot.get(kind, {})
        # Signature matches are safe only if unique. Duplicate identical
        # statements require an explicit reviewer rather than arbitrary joins.
        index = {}
        for new_id, record in new_records.items():
            index.setdefault(sha256_json(record), []).append(new_id)
        proposals[kind] = {}
        removed[kind] = []
        used = set()
        for old_id, value in old_choices.items():
            if kind == 'visuals':
                val = {'decision': value}
            elif isinstance(value, dict):
                val = _copy.deepcopy(value)
            else:
                continue
            if val.get('decision', 'pending') == 'pending':
                continue
            prior_record = old_records.get(old_id)
            new_id, verified, reason = None, False, 'old source snapshot unavailable'
            if prior_record is not None:
                exact = index.get(sha256_json(prior_record), [])
                if len(exact) == 1:
                    new_id, verified, reason = exact[0], True, 'exact source-and-candidate signature'
                elif len(exact) > 1:
                    reason = 'multiple identical targets; cannot determine source identity'
                elif old_id in new_choices:
                    new_id, reason = old_id, 'existing identifier with changed source or mapping'
            elif old_id in new_choices:
                new_id = old_id
                if kind in ('evidence', 'blocks') and old_source_same:
                    verified, reason = True, 'same full source fingerprint and identifier (legacy review)'
                elif kind in ('candidates', 'concepts') and old_candidates_same:
                    verified, reason = True, 'same full candidate fingerprint and identifier (legacy review)'
                else:
                    reason = 'legacy identifier only; prior text/mapping cannot be verified'
            if new_id is None or new_id in used:
                removed[kind].append(old_id)
                continue
            used.add(new_id)
            # A stable IRI is not proof that its OWL definition/aliases are
            # unchanged. Only compare mappings when the ontology resolver's
            # fingerprint is known stable (or the whole candidate stage agrees).
            if kind in ('candidates', 'concepts') and not ontology_unchanged:
                verified = False
                reason = 'ontology candidate vocabulary changed or its previous version is unverified'
            # An OWL IRI cannot be transferred if it is no longer an active
            # class in the reviewer-selected group.
            iri = val.get('selected_iri')
            group = val.get('selected_group') or new_records.get(new_id, {}).get('group')
            if iri:
                if resolver is None or (group, iri) not in resolver.by_iri:
                    verified = False
                    reason = 'selected ontology class is not verified active in the current profile'
            # Image content isn't hashed in the review snapshot; even if its
            # metadata is equal, paid vision needs an explicit new opt-in.
            if kind == 'visuals':
                verified = False
                reason = 'visual pixels cannot be verified by source metadata alone'
            change_kind = ('unchanged' if verified else
                           'changed' if prior_record is not None and
                           new_records.get(new_id) != prior_record else 'unverifiable')
            proposals[kind][new_id] = {'from_id': old_id, 'decision': val,
                                       'verified': verified, 'change_kind': change_kind,
                                       'reason': reason,
                                       'previous_source': prior_record,
                                       'current_source': new_records.get(new_id)}
    # Human-added candidates are hypotheses; retain them in the diff, but
    # never silently turn them into source-extracted facts or approved IRIs.
    proposals['manual_candidates'] = {}
    removed['manual_candidates'] = []
    for cid, source in previous.get('manual_candidates', {}).items():
        eid = source.get('evidence_id')
        same = old_source_same and eid in fresh['evidence']
        if not same and old_snapshot.get('evidence', {}).get(eid) is not None:
            old_ev = old_snapshot['evidence'][eid]
            same = eid in new_snapshot['evidence'] and new_snapshot['evidence'][eid] == old_ev
        # A previous human-added OWL mapping still needs stable vocabulary.
        # Candidate stage fingerprints often change with segmentation, so
        # compare the independently stored resolver fingerprint when present.
        same = same and ontology_unchanged
        if eid in fresh['evidence']:
            proposals['manual_candidates'][cid] = {
                'from_id': cid, 'verified': bool(same),
                'reason': 'source evidence verified' if same else 'source evidence unverified; recreate manually',
                'source': _copy.deepcopy(source),
                'decision': _copy.deepcopy(previous.get('candidates', {}).get(cid, {'decision': 'pending'}))}
        else:
            removed['manual_candidates'].append(cid)
    return {'proposals': proposals, 'removed': removed}


def refresh_review(path: Path, structure: dict, candidate_stage: dict,
                   resolver=None, *, dry_run: bool = False) -> dict:
    """Safely rebase a stale human review onto NEW parser and ontology outputs.

    The old review is copied to a timestamped backup. The new manifest starts
    entirely pending; prior decisions are reviewable proposals, NEVER active.
    Only explicitly confirmed, source-verifiable proposals may be transferred.
    A legacy review with changed source fingerprints has no provenance snapshot,
    so matching its IDs does not constitute evidence that the text is unchanged.

    Returns {'review': ..., 'summary': ..., 'backup_path': ..., 'status': ...}.
    This function never calls a model or modifies the ontology.
    """
    path = Path(path)
    if not path.exists():
        fresh = new_review(structure, candidate_stage)
        if not dry_run:
            write_json(path, fresh)
        return {'status': 'created', 'review': fresh, 'summary': review_summary(fresh, candidate_stage),
                'backup_path': None}
    previous = read_json(path)
    try:
        assert_current(previous, structure, candidate_stage)
    except ReviewRequired:
        pass
    else:
        return {'status': 'current', 'review': previous,
                'summary': review_summary(previous, candidate_stage), 'backup_path': None}
    fresh = new_review(structure, candidate_stage)
    diff = _refresh_proposals(previous, fresh, resolver)
    counts = {kind: {
        'verified': sum(p['verified'] for p in items.values()),
        'manual_check': sum(not p['verified'] for p in items.values()),
        'removed': len(diff['removed'][kind]),
        'new': len(set(fresh.get(kind, {})) - set(previous.get(kind, {})) - set(items)),
        'unchanged': sum(p.get('change_kind') == 'unchanged' for p in items.values()),
        'changed': sum(p.get('change_kind') == 'changed' for p in items.values()),
        'unverifiable': sum(p.get('change_kind', 'unverifiable') == 'unverifiable' for p in items.values()),
    } for kind, items in diff['proposals'].items()}
    fresh['refresh'] = {
        'status': 'awaiting_confirmation', 'previous_source_fingerprint': previous.get('source_fingerprint'),
        'previous_candidate_fingerprint': previous.get('candidate_fingerprint'),
        'previous_review_version': previous.get('review_version'),
        'proposals': diff['proposals'], 'removed': diff['removed'], 'counts': counts,
        'backup_file': None, 'confirmed_at': None,
    }
    if dry_run:
        return {'status': 'preview', 'review': fresh, 'summary': counts, 'backup_path': None}
    stamp = _datetime.now(_timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    backup = path.with_name(f'{path.stem}.backup_{stamp}{path.suffix}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise FileExistsError(f'Review backup already exists; not overwriting: {backup}')
    # Never silently destroy a review if a backup cannot be created.
    _shutil.copy2(path, backup)
    if not backup.exists() or backup.read_bytes() != path.read_bytes():
        raise OSError('Cannot verify the saved human-review backup; original review was not changed')
    fresh['refresh']['backup_file'] = backup.name
    tmp = path.with_name(path.name + '.refresh_tmp')
    try:
        write_json(tmp, fresh)
        _os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {'status': 'refreshed_pending_confirmation', 'review': fresh,
            'summary': counts, 'backup_path': str(backup)}


def confirm_refresh(path: Path, structure: dict, candidate_stage: dict, resolver,
                    *, accept_verified: bool = True) -> dict:
    """Explicitly activate proven-equal proposals, then close the migration gate.

    Unverified legacy-ID matches and visual approvals stay pending. A reviewer
    can inspect/edit those using the existing hierarchy and sentence editors.
    Calling with accept_verified=False confirms the refreshed manifest but
    intentionally carries over NONE of the old decisions.
    """
    path = Path(path)
    review = read_json(path)
    assert_current(review, structure, candidate_stage)
    info = review.get('refresh', {})
    if info.get('status') != 'awaiting_confirmation':
        raise ReviewRequired('No unconfirmed refreshed review was found at this path.')
    applied = Counter()
    if accept_verified:
        for kind in ('evidence', 'blocks', 'concepts', 'candidates'):
            for new_id, proposal in info.get('proposals', {}).get(kind, {}).items():
                if not proposal.get('verified') or review.get(kind, {}).get(new_id, {}).get('decision') != 'pending':
                    continue
                val = proposal['decision']; decision = val.get('decision', 'pending')
                if decision == 'pending':
                    continue
                try:
                    if kind == 'evidence':
                        set_evidence_decision(review, new_id, decision, val.get('note', ''))
                    elif kind == 'blocks':
                        set_block_decision(review, new_id, decision, val.get('note', ''))
                    elif kind == 'concepts':
                        set_concept_decision(review, candidate_stage, resolver, new_id, decision,
                                             selected_group=val.get('selected_group'),
                                             selected_iri=val.get('selected_iri'),
                                             term_override=val.get('term_override'),
                                             note=val.get('note', ''))
                    elif kind == 'candidates':
                        set_candidate_decision(review, candidate_stage, resolver, new_id, decision,
                                               selected_group=val.get('selected_group'),
                                               selected_iri=val.get('selected_iri'),
                                               term_override=val.get('term_override'),
                                               note=val.get('note', ''))
                except (KeyError, ValueError):
                    continue  # changed/invalid mappings stay pending, not dropped into OWL
                applied[kind] += 1
        for cid, proposal in info.get('proposals', {}).get('manual_candidates', {}).items():
            if not proposal.get('verified'):
                continue
            source = proposal['source']
            try:
                new_id = add_manual_candidate(review, structure, resolver, source['evidence_id'],
                                              source['term'], source['group'])
                val = proposal['decision']
                if val.get('decision', 'pending') != 'pending':
                    set_candidate_decision(review, candidate_stage, resolver, new_id,
                                           val['decision'], selected_group=val.get('selected_group'),
                                           selected_iri=val.get('selected_iri'),
                                           term_override=val.get('term_override'),
                                           note=val.get('note', ''))
                applied['manual_candidates'] += 1
            except (KeyError, ValueError):
                continue
    info['status'] = 'confirmed'
    info['confirmed_at'] = _datetime.now(_timezone.utc).isoformat()
    info['applied'] = dict(sorted(applied.items()))
    review['refresh'] = info
    write_json(path, review)
    return {'status': 'confirmed', 'review': review, 'applied': dict(sorted(applied.items())),
            'summary': review_summary(review, candidate_stage)}


# ---------------------------------------------------------------------------
# v0.8 review is OPTIONAL for source inclusion. "review_status" is always a
# binary human inspection flag; inclusion/mapping decisions are independent.
# ---------------------------------------------------------------------------

# Source categories that are not primary scientific assertions. A figure caption
# is text (and may be transmitted), but crop pixels always require opt-in vision.
_NONPRIMARY_ROLES = {'references', 'bibliography', 'furniture'}
_NONPRIMARY_KINDS = {'reference', 'code'}


def default_evidence_decision(e: dict) -> str:
    """Conservative source policy; never use a review flag to decide inclusion."""
    if (e.get('section_role') in _NONPRIMARY_ROLES or
            e.get('source_kind') in _NONPRIMARY_KINDS or
            'bibliographic_entry' in e.get('scientific_roles', [])):
        return 'exclude'
    if not e.get('sentence', '').strip():
        return 'exclude'
    # Front matter often contains author names, affiliations and paper metadata.
    if e.get('section_role') == 'front_matter' and not e.get('scientific_roles'):
        return 'exclude'
    return 'extract'


def review_state(item: dict) -> str:
    """Binary status, including legacy review manifests."""
    raw = item.get('review_status')
    if raw in ('reviewed', 'not reviewed'):
        return raw
    if item.get('decision', 'pending') not in ('pending', None):
        return 'reviewed'
    return 'not reviewed'


def set_review_status(review: dict, kind: str, item_id: str, reviewed: bool = True) -> None:
    """Mark inspection without implying extraction approval or OWL equivalence."""
    if kind not in ('evidence','blocks','concepts','candidates','assertions','relation_candidates'):
        raise ValueError(f'Unsupported review category: {kind}')
    item = review.get(kind, {}).get(item_id)
    if item is None:
        raise KeyError(f'{kind}: {item_id}')
    item['review_status'] = 'reviewed' if reviewed else 'not reviewed'


def review_issues(structure: dict, candidate_stage: dict) -> dict:
    """Source-backed issue candidates: not a verdict on scientific correctness."""
    by_evidence = _defaultdict(list)
    for c in candidate_stage.get('candidates', []):
        iri_options = {(v.get('group'), v.get('iri')) for v in c.get('suggestions', []) if v.get('iri')}
        status = c.get('match_status', '')
        if status != 'exact_match':
            by_evidence[c['evidence_id']].append('unmatched/approximate OWL: '+c['term'])
        if re.fullmatch(r'(?i)EDAX|EDX|EDS', c.get('term','').strip()):
            by_evidence[c['evidence_id']].append('EDX/EDAX/EDS alias requires context check')
        elif len(iri_options) > 1:
            by_evidence[c['evidence_id']].append('multiple OWL choices: '+c['term'])
    for a in candidate_stage.get('source_assertions', []):
        if a.get('claim_origin') in ('undetermined','mixed_current_and_cited') and (
            a.get('reference_ids') or a.get('claim_kind') in ('comparison','interpretation')):
            for eid in a.get('evidence_ids', []):
                by_evidence[eid].append('ambiguous claim origin')
    for r in candidate_stage.get('relation_candidates', []):
        if r.get('status') in ('requires_operand_review','qualified_claim_only'):
            for eid in r.get('evidence_ids', []):
                by_evidence[eid].append('unresolved relation: '+r.get('relation_type',''))
    for mention in candidate_stage.get('specimen_mentions', []):
        if mention.get('link_status') == 'unresolved':
            by_evidence[mention['evidence_id']].append('unlinked specimen mention')
    # Blocks with no scientific candidates but explicit experimental/result cues
    # should be inspectable too; don't flag every generic paragraph.
    with_candidate = {c['evidence_id'] for c in candidate_stage.get('candidates', [])}
    for ev in structure.get('semantic_evidence', []):
        if (ev['id'] not in with_candidate and default_evidence_decision(ev) == 'extract'
                and set(ev.get('scientific_roles', [])) & {'result','method','quantitative_evidence'}):
            by_evidence[ev['id']].append('scientific evidence without a recognized OWL term')
    blocks = candidate_stage.get('evidence_blocks', [])
    block_issues = {}
    for b in blocks:
        reasons = sorted({r for eid in b.get('evidence_ids', []) for r in by_evidence.get(eid, [])})
        if reasons:
            block_issues[b['id']] = reasons
    return {'issue_ids': sorted(by_evidence), 'evidence_reasons': dict(by_evidence),
            'block_reasons': block_issues}


def preflight_tables(structure: dict, candidate_stage: dict, review: dict | None = None) -> dict:
    """Actual pandas tables for the notebook, no LLM calls and no approval gate."""
    import pandas as pd
    issues = review_issues(structure, candidate_stage)
    evidence = {e['id']:e for e in structure.get('semantic_evidence', [])}
    rows=[]
    for ev in structure.get('semantic_evidence', []):
        choice=(review or {}).get('evidence', {}).get(ev['id'], {})
        block_id='EB_'+str(ev.get('paragraph_id',''))
        override=choice.get('decision','pending')
        block_override=(review or {}).get('blocks', {}).get(block_id, {}).get('decision','pending')
        effective = override if override != 'pending' else (block_override if block_override != 'pending' else default_evidence_decision(ev))
        rows.append({'evidence_id':ev['id'],'block_id':block_id,'page':ev.get('page'),
                     'section':ev.get('section_role'),'kind':ev.get('source_kind'),
                     'review_status':('reviewed' if review_state(choice)=='reviewed' or
                          review_state((review or {}).get('blocks',{}).get(block_id,{}))=='reviewed'
                          else 'not reviewed'),'include_as':effective,
                     'needs_review':bool(issues['evidence_reasons'].get(ev['id'])),
                     'review_reasons':'; '.join(issues['evidence_reasons'].get(ev['id'], [])),
                     'sentence':ev.get('sentence', '')})
    candidates=[]
    for c in candidate_stage.get('candidates', []):
        choice=(review or {}).get('candidates', {}).get(c['id'], {})
        candidates.append({'candidate_id':c['id'],'evidence_id':c['evidence_id'],
                 'page':evidence.get(c['evidence_id'], {}).get('page'),
                 'term':c['term'],'group':c['group'], 'match_status':c['match_status'],
                 'matched_iri':c.get('matched_iri'),
                 'alternatives':'; '.join(x.get('label','') for x in c.get('suggestions', [])),
                 'review_status':('reviewed' if review_state(choice)=='reviewed' or
                      review_state((review or {}).get('concepts',{}).get(c.get('concept_id'),{}))=='reviewed'
                      else 'not reviewed'),
                 'needs_review':c['evidence_id'] in issues['issue_ids']})
    block_rows=[]
    for b in candidate_stage.get('evidence_blocks', []):
        choice=(review or {}).get('blocks', {}).get(b['id'], {})
        block_rows.append({'block_id':b['id'],'page':b.get('page'),'section':b.get('section_role'),
              'kind':b.get('source_kind'),'lines':len(b.get('evidence_ids', [])),
              'needs_review':b['id'] in issues['block_reasons'],
              'review_reasons':'; '.join(issues['block_reasons'].get(b['id'], [])),
              'review_status':review_state(choice),
              'preview':' '.join(evidence[eid].get('sentence','') for eid in b.get('evidence_ids', []) if eid in evidence)[:260]})
    return {'evidence':pd.DataFrame(rows), 'candidates':pd.DataFrame(candidates),
            'blocks':pd.DataFrame(block_rows),
            'assertions':pd.DataFrame(candidate_stage.get('source_assertions', [])),
            'references':pd.DataFrame(candidate_stage.get('document_reference_targets', []))}


def transmission_tables(structure: dict, selection: dict, max_chars: int = 120_000) -> dict:
    """Exact text packets used by the LLM, plus a row-per-primary-sentence view."""
    import pandas as pd
    packets=approved_documents(structure, selection, max_chars)
    sent_rows=[]
    for e in selection.get('extract', []):
        sent_rows.append({'evidence_id':e['id'],'paragraph_id':e['paragraph_id'],
            'page':e.get('page'), 'section':e.get('section_role'),
            'source_kind':e.get('source_kind'), 'send_as':'PRIMARY',
            'review_status':e.get('review_status','not reviewed'),
            'sentence':e.get('sentence','')})
    for e in selection.get('context_only', []):
        if not any(f"[CONTEXT ONLY {e['id']} |" in d['text'] for d in packets):
            continue  # Context-only evidence not adjacent to a primary is not transmitted.
        sent_rows.append({'evidence_id':e['id'],'paragraph_id':e['paragraph_id'],
            'page':e.get('page'), 'section':e.get('section_role'),
            'source_kind':e.get('source_kind'), 'send_as':'CONTEXT ONLY',
            'review_status':e.get('review_status','not reviewed'),
            'sentence':e.get('sentence','')})
    return {'sentences':pd.DataFrame(sent_rows),
      'packets':pd.DataFrame([{'chunk':d['chunk'],'chars':d['chars'],
          'paragraph_ids':', '.join(d['paragraph_ids']), 'exact_llm_input':d['text']} for d in packets]),
      'packets_raw':packets}


def figure_inspector(structure: dict, paper_output_dir: Path):
    """Inspect SAVED Docling crop pixels + structured extractions without paid vision.

    Does not request any model interpretation and never changes vision approvals.
    """
    try:
        import ipywidgets as w
        import pandas as pd
        from IPython.display import Image, display
    except ImportError as exc:
        raise ImportError('Install ipywidgets, pandas and IPython for crop inspection') from exc
    root = Path(paper_output_dir).resolve()
    items = [('figure', x) for x in structure.get('figures', [])] + [('table', x) for x in structure.get('tables', [])]
    if not items:
        return w.HTML('No Docling figures or tables are present in this extraction.')
    opts=[(f"{kind}: {x.get('label', x.get('paragraph_id'))} | page {x.get('page', '?')}", idx)
          for idx,(kind,x) in enumerate(items)]
    choice=w.Dropdown(options=opts, description='Crop',layout=w.Layout(width='95%'))
    details=w.Output(layout=w.Layout(border='1px solid #ddd',padding='8px'))
    def render(*_):
        kind, asset=items[choice.value]
        pid=asset.get('paragraph_id')
        crops=asset.get('crop')
        with details:
            details.clear_output(wait=True)
            if crops:
                candidate=(root / crops).resolve()
                if not candidate.is_relative_to(root):
                    print('Unsafe crop path was rejected')
                elif candidate.exists() and candidate.suffix.lower() in ('.png','.jpg','.jpeg','.webp'):
                    display(Image(filename=str(candidate),width=750))
                else:
                    print('Saved Docling crop not found:',crops)
            else:
                print('No segmented Docling crop was saved for this asset.')
            print('Docling metadata and extracted scientific context (not image verified):')
            display(pd.DataFrame([{'kind':kind,'paragraph_id':pid,'page':asset.get('page'),
               'label':asset.get('label'), 'caption':asset.get('caption'),
               'classification':str(asset.get('classification','')),
               'crop_path':crops}]).T.rename(columns={0:'value'}))
            if kind=='table':
                p=next((x for x in structure.get('paragraphs', []) if x['id']==pid), {})
                if p.get('rows'):display(pd.DataFrame(p['rows']))
            else:
                plots=[x for x in structure.get('plots', []) if x.get('source_paragraph_id')==pid]
                if plots:display(pd.DataFrame(plots).T)
            evidence=[e for e in structure.get('semantic_evidence', []) if e['paragraph_id']==pid]
            if evidence:display(pd.DataFrame([{'evidence_id':e['id'],'sentence':e.get('sentence','')}
                                             for e in evidence]))
    choice.observe(render,names='value');render()
    return w.VBox([w.HTML('<b>Saved Docling segmented crop and extracted data. Inspection does not approve a vision API call.</b>'),
                   choice,details])


def review_widget(structure: dict, candidate_stage: dict, review_path: Path, resolver,
                  *, show_all: bool = False):
    """Default ambiguous/unmatched evidence reviewer; toggle to inspect every line.

    Review status is binary. Decision controls are separate optional overrides.
    """
    try:
        import ipywidgets as w
        from html import escape
    except ImportError as exc:
        raise ImportError('Install ipywidgets to display the optional review widget') from exc
    review=read_or_create_review(review_path,structure,candidate_stage)
    if review.get('refresh', {}).get('status') == 'awaiting_confirmation':
        msg=w.HTML('<b>Previous human decisions were backed up after a source/ontology change.</b> '
                   'The automatic source selection below uses fresh evidence. Prior approvals remain inactive. '
                   'Confirm migration only if you want to restore verified previous review choices.')
        a=w.Button(description='Restore verified choices',button_style='warning')
        b=w.Button(description='Discard old choices')
        out=w.Output()
        def confirm(flag):
            with out:
                out.clear_output()
                try:
                    confirm_refresh(review_path,structure,candidate_stage,resolver,accept_verified=flag)
                    print('Review refresh confirmed. Rerun this cell to edit the current manifest.')
                    a.disabled=b.disabled=True
                except Exception as exc:print(exc)
        a.on_click(lambda _:confirm(True));b.on_click(lambda _:confirm(False))
        return w.VBox([msg,w.HBox([a,b]),out])
    issue=review_issues(structure,candidate_stage)
    evidence={e['id']:e for e in structure.get('semantic_evidence', [])}
    blocks=candidate_stage.get('evidence_blocks',[])
    concepts=candidate_stage.get('concept_registry',[])
    by_concept={c['id']:c for c in concepts}
    flagged_eids=set(issue['issue_ids'])
    flagged_cids={c['concept_id'] for c in candidate_stage.get('candidates', [])
                  if c['evidence_id'] in flagged_eids and c.get('concept_id')}
    flagged_assertions={x['id'] for x in candidate_stage.get('source_assertions', [])
        if any(eid in flagged_eids for eid in x.get('evidence_ids', []))}
    all_mode=w.Checkbox(value=show_all,description='Inspect all lines / concepts / claims',indent=False)
    headline=w.HTML('<b>Default: only ambiguous or unmatched items.</b> Auto-selected primary text needs no manual approval. '
                    'The Review status flag tracks inspection, not whether evidence is included.')
    summary=w.HTML()
    block_sel=w.Dropdown(description='Block',layout=w.Layout(width='95%'))
    block_preview=w.HTML()
    block_decision=w.Dropdown(options=[('Use automatic policy','pending'),('Include as PRIMARY','extract'),
              ('Use as context only','context_only'),('Exclude','exclude')],description='Send as')
    block_reviewed=w.Checkbox(description='Reviewed',indent=False)
    save_block=w.Button(description='Save block',button_style='success')
    concept_sel=w.Dropdown(description='Concept',layout=w.Layout(width='95%'))
    concept_preview=w.HTML()
    concept_choice=w.Dropdown(options=[('Keep automatic match / unresolved candidate','pending'),
          ('Approve active OWL class','approve'),('Edit / unresolved','edit'),('Reject mapping','reject')],description='Mapping')
    concept_reviewed=w.Checkbox(description='Reviewed',indent=False)
    term=w.Text(description='Term',layout=w.Layout(width='95%'))
    group=w.Dropdown(options=sorted(resolver.entries),description='OWL group')
    iri=w.Text(description='OWL IRI',layout=w.Layout(width='95%'))
    save_concept=w.Button(description='Save concept',button_style='success')
    sentence_sel=w.Dropdown(description='Sentence',layout=w.Layout(width='95%'))
    sentence_preview=w.HTML()
    sentence_choice=w.Dropdown(options=[('Inherit block / automatic policy','pending'),
          ('Include as PRIMARY','extract'),('Use as context only','context_only'),('Exclude','exclude')],description='Send as')
    sentence_reviewed=w.Checkbox(description='Reviewed',indent=False)
    save_sentence=w.Button(description='Save sentence',button_style='success')
    claim_sel=w.Dropdown(description='Claim',layout=w.Layout(width='95%'))
    claim_preview=w.HTML()
    claim_reviewed=w.Checkbox(description='Reviewed',indent=False)
    claim_origin=w.Dropdown(options=['current_study','cited_study','background_knowledge',
              'undetermined','mixed_current_and_cited'],description='Origin')
    save_claim=w.Button(description='Save claim',button_style='success')
    assets=[(kind,item) for kind in ('figure','table') for item in structure.get('figures' if kind=='figure' else 'tables', [])
            if item.get('paragraph_id')]
    asset_sel=w.Dropdown(description='Asset',options=[(f"{kind} {a.get('label',a['paragraph_id'])}",a['paragraph_id'])
        for kind,a in assets],layout=w.Layout(width='95%'))
    vision_decision=w.Dropdown(options=[('Do not send crop to vision','pending'),
              ('Approve paid vision analysis','approve'),('Reject paid vision','reject')],description='Vision')
    save_vision=w.Button(description='Save vision choice',button_style='warning')
    messages=w.Output()
    def persist():
        write_json(review_path,review)
        state=review_summary(review,candidate_stage)
        summary.value=(f"<b>Candidate lines:</b> {len(evidence)} &nbsp; "
          f"<b>Flagged:</b> {len(flagged_eids)} &nbsp; "
          f"<b>Manual review:</b> {state['review_status']['evidence']} &nbsp; "
          f"<b>Unreviewed text is included by default (except bibliography/furniture).</b>")
    def refresh_choices(*_):
        selected_blocks=[b for b in blocks if all_mode.value or b['id'] in issue['block_reasons']]
        selected_concepts=[c for c in concepts if all_mode.value or c['id'] in flagged_cids]
        selected_evs=[e for e in structure.get('semantic_evidence',[]) if all_mode.value or e['id'] in flagged_eids]
        selected_claims=[a for a in candidate_stage.get('source_assertions',[]) if all_mode.value or a['id'] in flagged_assertions]
        block_sel.options=[(f"{b['id']} | p{b.get('page')} | {b.get('section_role')}",b['id']) for b in selected_blocks]
        concept_sel.options=[(f"{c['canonical_term']} [{c['group']}; {c['match_status']}]",c['id']) for c in selected_concepts]
        sentence_sel.options=[(f"{e['id']} | {e['sentence'][:110]}",e['id']) for e in selected_evs]
        claim_sel.options=[(f"{a['claim_origin']} | {a['clause'][:105]}",a['id']) for a in selected_claims]
        for control in (block_sel,concept_sel,sentence_sel,claim_sel):
            control.disabled=not bool(control.options)
    def update_block(*_):
        b=next((b for b in blocks if b['id']==block_sel.value),None)
        if not b:return
        saved=review.get('blocks',{}).get(b['id'],{})
        block_reviewed.value=review_state(saved)=='reviewed'
        block_decision.value=saved.get('decision','pending')
        text=''.join('<li>'+escape(evidence[eid]['sentence'])+'</li>' for eid in b.get('evidence_ids',[]) if eid in evidence)
        block_preview.value=('<b>Reason:</b> '+escape('; '.join(issue['block_reasons'].get(b['id'], [])) or 'No flagged mismatch')+
                             '<ol>'+text+'</ol>')
    def update_concept(*_):
        c=by_concept.get(concept_sel.value)
        if not c:return
        val=review.get('concepts',{}).get(c['id'],{})
        concept_reviewed.value=review_state(val)=='reviewed'
        concept_choice.value=val.get('decision','pending')
        term.value=val.get('term_override') or c['canonical_term']
        group.value=val.get('selected_group') or c['group']
        iri.value=val.get('selected_iri') or c.get('matched_iri') or ''
        concept_preview.value='<b>Suggested:</b> '+escape(str(c.get('suggestions', [])[:3]))
    def update_sentence(*_):
        ev=evidence.get(sentence_sel.value)
        if not ev:return
        val=review.get('evidence',{}).get(ev['id'],{})
        sentence_choice.value=val.get('decision','pending')
        sentence_reviewed.value=review_state(val)=='reviewed'
        sentence_preview.value='<b>Reason:</b> '+escape('; '.join(issue['evidence_reasons'].get(ev['id'],[])) or 'No flagged mismatch')+              '<p>'+escape(ev.get('sentence',''))+'</p>'
    def update_claim(*_):
        a=next((x for x in candidate_stage.get('source_assertions',[]) if x['id']==claim_sel.value),None)
        if not a:return
        val=review.get('assertions',{}).get(a['id'],{})
        claim_reviewed.value=review_state(val)=='reviewed'
        claim_origin.value=val.get('claim_origin_override') or a['claim_origin']
        claim_preview.value='<p>'+escape(a['clause'])+'</p><small>'+escape(a.get('origin_reason',''))+'</small>'
    def update_vision(*_):
        vision_decision.value=review.get('visuals',{}).get(asset_sel.value,'pending')
    def save(kind, callback):
        with messages:
            messages.clear_output()
            try:
                callback()
                persist()
                print('Saved',kind,'(review status and inclusion are separate).')
            except Exception as ex:print('Could not save:',ex)
    def do_block():
        set_block_decision(review,block_sel.value,block_decision.value)
        set_review_status(review,'blocks',block_sel.value,block_reviewed.value)
    def do_concept():
        c=by_concept[concept_sel.value]
        set_concept_decision(review,candidate_stage,resolver,c['id'],concept_choice.value,
            selected_group=group.value,selected_iri=iri.value.strip() or None,
            term_override=(term.value.strip() if term.value.strip()!=c['canonical_term'] else None))
        set_review_status(review,'concepts',c['id'],concept_reviewed.value)
    def do_sentence():
        set_evidence_decision(review,sentence_sel.value,sentence_choice.value)
        set_review_status(review,'evidence',sentence_sel.value,sentence_reviewed.value)
    def do_claim():
        a=next(x for x in candidate_stage.get('source_assertions',[]) if x['id']==claim_sel.value)
        override=claim_origin.value != a['claim_origin']
        set_assertion_decision(review,a['id'],'edit' if override else ('approve' if claim_reviewed.value else 'pending'),
            claim_origin_override=claim_origin.value if override else None)
        set_review_status(review,'assertions',a['id'],claim_reviewed.value)
    def do_vision():
        set_visual_decision(review,asset_sel.value,vision_decision.value)
    save_block.on_click(lambda _:save('block',do_block))
    save_concept.on_click(lambda _:save('concept',do_concept))
    save_sentence.on_click(lambda _:save('sentence',do_sentence))
    save_claim.on_click(lambda _:save('claim',do_claim))
    save_vision.on_click(lambda _:save('visual',do_vision))
    block_sel.observe(update_block,names='value')
    concept_sel.observe(update_concept,names='value')
    sentence_sel.observe(update_sentence,names='value')
    claim_sel.observe(update_claim,names='value')
    asset_sel.observe(update_vision,names='value')
    all_mode.observe(refresh_choices,names='value')
    refresh_choices();update_block();update_concept();update_sentence();update_claim();update_vision();persist()
    block_page=w.VBox([block_sel,block_preview,w.HBox([block_decision,block_reviewed]),save_block])
    concepts_page=w.VBox([concept_sel,concept_preview,concept_choice,term,group,iri,
                          concept_reviewed,save_concept])
    line_page=w.VBox([sentence_sel,sentence_preview,sentence_choice,sentence_reviewed,save_sentence])
    claims_page=w.VBox([claim_sel,claim_preview,claim_origin,claim_reviewed,save_claim])
    visual_page=w.VBox([w.HTML('Figure/table image inspection is available through figure_inspector() below. '
        'These controls only govern optional <b>paid vision</b> calls.'),asset_sel,vision_decision,save_vision])
    tabs=w.Tab(children=[block_page,concepts_page,line_page,claims_page,visual_page])
    for n,title in enumerate(('Flagged blocks','OWL concepts','Sentence overrides','Claim attribution','Vision opt-in')):
        tabs.set_title(n,title)
    return w.VBox([headline,all_mode,summary,tabs,messages])
