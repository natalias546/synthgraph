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
    """Start fail-closed; do not grant access to LLM input by default."""
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
    review["evidence"][evidence_id] = {"decision": decision, "note": note.strip()}


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
        "note": note.strip()}


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


def review_widget(structure: dict, candidate_stage: dict, review_path: Path, resolver):
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
    review = read_or_create_review(review_path, structure, candidate_stage)
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
            if iri and (group, iri) in resolver.by_iri:
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
