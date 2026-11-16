"""Offline regression: stale review refresh preserves history, never silently approves.

Run: python scripts/test_review_refresh.py
No ontology, model, or Neo4j service required for the unit tests.
"""
from __future__ import annotations

import copy
import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.io_utils import read_json, write_json
from src.pipeline.pre_llm_review import (
    ReviewRequired, assert_current, confirm_refresh, evidence_fingerprint, new_review,
    read_or_create_review, refresh_review, reviewed_selection, review_summary,
    set_block_decision, set_candidate_decision, set_concept_decision,
    set_evidence_decision, set_visual_decision,
)

IRI = 'https://example.org/Property'


def sample_structure(sentence='Resistivity measured at 2 K.', note='Second result at 5 K.'):
    return {
        'title': 'A test paper',
        'sections': [{'id': 'S', 'heading': 'Results', 'role': 'results'}],
        'paragraphs': [
            {'id': 'P1', 'section_id': 'S', 'role': 'results', 'page': 1, 'text': sentence},
            {'id': 'P2', 'section_id': 'S', 'role': 'results', 'page': 1, 'text': note},
            {'id': 'FIG', 'section_id': 'S', 'role': 'results', 'page': 1,
             'text': '[Figure] X and Y'},
        ],
        'semantic_evidence': [
            {'id': 'P1:s1', 'paragraph_id': 'P1', 'section_id': 'S',
             'section_role': 'results', 'source_kind': 'text', 'page': 1,
             'sentence': sentence, 'span_start': 0, 'span_end': len(sentence)},
            {'id': 'P2:s1', 'paragraph_id': 'P2', 'section_id': 'S',
             'section_role': 'results', 'source_kind': 'text', 'page': 1,
             'sentence': note, 'span_start': 0, 'span_end': len(note)},
        ],
        'figures': [{'paragraph_id': 'FIG', 'page': 1, 'crop': 'figure.png', 'caption': 'X and Y'}],
        'tables': [],
    }


def sample_candidates(st, fp='C1', *, suggestion=IRI, resolver_fp='R1'):
    return {
        'candidate_fingerprint': fp,
        'resolver_fingerprint': resolver_fp,
        'candidates': [{
            'id': 'OC1', 'evidence_id': 'P1:s1', 'paragraph_id': 'P1',
            'term': 'Resistivity', 'group': 'property', 'matched_iri': suggestion,
            'suggestions': [{'group': 'property', 'iri': suggestion, 'label': 'Resistivity'}] if suggestion else [],
        }],
        'concept_registry': [{
            'id': 'CC1', 'canonical_term': 'Resistivity', 'group': 'property',
            'evidence_ids': ['P1:s1'], 'aliases': ['Resistivity'],
            'matched_iri': suggestion,
        }],
        'evidence_blocks': [{
            'id': 'B1', 'evidence_ids': ['P1:s1'], 'section_role': 'results', 'page': 1,
        }],
        'acronyms': [],
    }


def dummy_resolver():
    return SimpleNamespace(entries={'property': []},
                           by_iri={('property', IRI): SimpleNamespace(label='Resistivity')})


def approved_rev(st, candidates, resolver):
    rev = new_review(st, candidates)
    set_evidence_decision(rev, 'P1:s1', 'extract')
    set_evidence_decision(rev, 'P2:s1', 'context_only')
    set_block_decision(rev, 'B1', 'extract')
    set_candidate_decision(rev, candidates, resolver, 'OC1', 'approve', selected_iri=IRI)
    set_concept_decision(rev, candidates, resolver, 'CC1', 'approve', selected_iri=IRI)
    set_visual_decision(rev, 'FIG', 'approve')
    return rev


def expect_clean_defaults(st, c, review, resolver):
    selection = reviewed_selection(st, c, review, resolver)
    assert selection['extract'], 'Fresh evidence must be available without review approval'
    assert not selection['approved_visuals'], 'Old vision approvals must stay inactive'
    assert not selection['hints'], 'Old class approvals must stay inactive'


def main():
    r = dummy_resolver()
    old = sample_structure()
    old_c = sample_candidates(old)
    previous = approved_rev(old, old_c, r)
    with tempfile.TemporaryDirectory() as temp:
        review_path = Path(temp) / 'pre_llm_review.json'
        write_json(review_path, previous)
        assert read_or_create_review(review_path, old, old_c) == read_json(review_path)

        # New ontology vocabulary, identical source text: source decisions are
        # verifiable, but candidate mappings changed and MUST be reviewed.
        new_c = sample_candidates(old, fp='C2', suggestion=None, resolver_fp='R2')
        before = review_path.read_bytes()
        preview = refresh_review(review_path, old, new_c, r, dry_run=True)
        assert preview['status'] == 'preview' and review_path.read_bytes() == before
        result = refresh_review(review_path, old, new_c, r)
        assert result['status'] == 'refreshed_pending_confirmation'
        assert Path(result['backup_path']).read_bytes() == before
        migrated = result['review']
        assert migrated['evidence']['P1:s1']['decision'] == 'pending'
        assert migrated['candidates']['OC1']['decision'] == 'pending'
        assert migrated['refresh']['proposals']['evidence']['P1:s1']['verified']
        assert not migrated['refresh']['proposals']['candidates']['OC1']['verified']
        assert not migrated['refresh']['proposals']['visuals']['FIG']['verified']
        expect_clean_defaults(old, new_c, migrated, r)
        # An explicit new decision may be used, but stale previous choices stay inactive.
        set_evidence_decision(migrated, 'P1:s1', 'extract')
        expect_clean_defaults(old, new_c, migrated, r)
        confirm = confirm_refresh(review_path, old, new_c, r)
        assert confirm['status'] == 'confirmed'
        after = confirm['review']
        assert after['evidence']['P1:s1']['decision'] == 'extract'
        assert after['evidence']['P2:s1']['decision'] == 'context_only'
        assert after['candidates']['OC1']['decision'] == 'pending'
        assert after['visuals'].get('FIG', 'pending') != 'approve'
        reviewed_selection(old, new_c, after, r)  # allowed only now
        assert refresh_review(review_path, old, new_c, r)['status'] == 'current'
        print('PASS: ontology-only migration with verified evidence and pending class/vision hints')

    with tempfile.TemporaryDirectory() as temp:
        review_path = Path(temp) / 'pre_llm_review.json'
        write_json(review_path, previous)
        # Source changed in one paragraph, unchanged in another: match by
        # source snapshot. Source IDs alone must not override a changed quote.
        changed = sample_structure(note='Second result at 8 K.')
        new_c = sample_candidates(changed, 'C3')
        result = refresh_review(review_path, changed, new_c, r)
        proposal = result['review']['refresh']['proposals']
        assert proposal['evidence']['P1:s1']['verified']
        assert not proposal['evidence']['P2:s1']['verified']
        confirmed = confirm_refresh(review_path, changed, new_c, r)['review']
        assert confirmed['evidence']['P1:s1']['decision'] == 'extract'
        assert confirmed['evidence']['P2:s1']['decision'] == 'pending'
        assert confirmed['concepts']['CC1']['decision'] == 'approve'
        assert confirmed['visuals'].get('FIG', 'pending') == 'pending'
        print('PASS: source-level fingerprinting blocks old decisions for altered sentences')

    with tempfile.TemporaryDirectory() as temp:
        review_path = Path(temp) / 'pre_llm_review.json'
        legacy = copy.deepcopy(previous)
        legacy.pop('source_snapshot')  # v0.6 pre-patch review
        write_json(review_path, legacy)
        changed = sample_structure(note='Updated second result at 9 K.')
        result = refresh_review(review_path, changed, sample_candidates(changed, 'C4'), r)
        proposal = result['review']['refresh']['proposals']['evidence']['P1:s1']
        assert not proposal['verified'] and 'legacy' in proposal['reason']
        after = confirm_refresh(review_path, changed, sample_candidates(changed, 'C4'), r)['review']
        assert after['evidence']['P1:s1']['decision'] == 'pending'
        expect_clean_defaults(changed, sample_candidates(changed, 'C4'), after, r)
        print('PASS: legacy reviews with no snapshot retain proposals but never silently restore approvals')

    print('All refresh_review regression tests passed.')


def test_pipeline_refresh(owl: Path):
    """Actual notebook preflight must refresh stale review without paid calls."""
    from reportlab.pdfgen import canvas
    from src.pipeline.config import Config
    from src.pipeline.runner import Pipeline
    def create_pdf(path, content):
        c = canvas.Canvas(str(path))
        c.setFont('Helvetica-Bold', 13)
        c.drawString(70, 740, 'Scientific Investigation')
        c.setFont('Helvetica', 11)
        c.drawString(70, 712, 'Results')
        c.drawString(70, 688, content)
        c.save()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        publications = root / 'data' / 'publications'
        publications.mkdir(parents=True)
        paper = publications / 'paper.pdf'
        create_pdf(paper, 'Resistivity measured at 2 K.')
        cfg = Config(project_root=root, ontology_file=owl,
                     profile_file=Path(__file__).resolve().parents[1] / 'ontology' / 'slice_manifest.json',
                     parser='pdfplumber', table_vision='never', allow_live_calls=False)
        pl = Pipeline(cfg)
        first = pl.run_paper(paper, until='human_review')
        review_path = Path(first['human_review']['path'])
        review = read_json(review_path)
        set_evidence_decision(review, first['structure']['semantic_evidence'][0]['id'], 'extract')
        write_json(review_path, review)
        create_pdf(paper, 'Resistivity measured at 5 K.')
        refreshed = pl.run_paper(paper, until='human_review')
        assert refreshed['human_review']['summary']['refresh']['status'] == 'awaiting_confirmation'
        assert len(list(review_path.parent.glob('pre_llm_review.backup_*.json'))) == 1
        preview = pl.run_paper(paper, until='visuals')
        assert preview['visuals'].get('skipped') is True
        assert 'llm' not in pl.__dict__
        assert not reviewed_selection(preview['structure'], preview['pre_llm_candidates'],
                                      read_json(review_path), pl.ontology_resolver)['approved_visuals']
        print('PASS: notebook preflight refreshes stale reviews without restoring old approvals')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--owl', type=Path, help='Optional full offline PDF integration test')
    args = parser.parse_args()
    main()
    if args.owl:
        test_pipeline_refresh(args.owl.resolve())
