"""Offline v0.8.3 regression: miscaptioned publisher logos and exact-packet cost check.

Run from project root: python scripts/test_v083_logo_cost_hotfix.py
No PDF, API key, network or ontology reasoner required.
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.evidence_context import enrich_structure
from src.pipeline.pre_llm_review import (new_review, reviewed_selection,
                                         transmission_tables, make_candidates)
from src.pipeline.runner import Pipeline
from scripts.test_optional_preflight import fixture


def test_logo_filter():
    structure = {
        'title': 'Springer test',
        'sections': [{'id': 'S1', 'role': 'results', 'heading': 'Results'}],
        'paragraphs': [
            {'id': 'P1', 'section_id': 'S1', 'kind': 'figure', 'role': 'results',
             'page': 6, 'classification': {'class': 'logo'},
             'caption': 'Fig. 6. Magnetoresistance curves at 2 K.',
             'text': '[Figure 6 | logo] Fig. 6. Magnetoresistance curves at 2 K.'},
            {'id': 'P2', 'section_id': 'S1', 'kind': 'text', 'role': 'results',
             'page': 6, 'text': 'The resistivity was measured at 2 K.'},
            {'id': 'P3', 'section_id': 'S1', 'kind': 'caption', 'role': 'results',
             'page': 6, 'text': 'Fig. 6. Magnetoresistance curves at 2 K.'},
            {'id': 'P4', 'section_id': 'S1', 'kind': 'figure', 'role': 'results',
             'page': 6, 'classification': {'class': 'line_chart'},
             'caption': 'Fig. 6. Magnetoresistance curves at 2 K.',
             'text': '[Figure 6 | line_chart] Fig. 6. Magnetoresistance curves at 2 K.'},
        ],
        'figures': [
            {'paragraph_id': 'P1', 'page': 6, 'classification': {'class': 'logo'},
             'caption': 'Fig. 6. Magnetoresistance curves at 2 K.',
             'crop': 'crops/springer-logo.png', 'label': 'figure 6'},
            {'paragraph_id': 'P4', 'page': 6, 'classification': {'class': 'line_chart'},
             'caption': 'Fig. 6. Magnetoresistance curves at 2 K.',
             'crop': 'crops/actual-plot.png', 'label': 'figure 6'},
        ], 'tables': [],
    }
    st = enrich_structure(structure)
    assert 'P1' not in {p['id'] for p in st['paragraphs']}
    assert 'P1' not in {f['paragraph_id'] for f in st['figures']}
    assert 'P4' in {f['paragraph_id'] for f in st['figures']}
    assert 'P3' in {p['id'] for p in st['paragraphs']}  # real caption preserved
    assert all(e['paragraph_id'] != 'P1' for e in st['semantic_evidence'])
    assert all(b['paragraph_id'] != 'P1' for b in st['evidence_blocks'])
    assert 'P1' not in {p['source_paragraph_id'] for p in st['plots']}
    assert st['visual_filter_summary']['excluded_with_attached_caption'] == 1
    log = st['excluded_decorative_pictures'][0]
    assert log['caption'].startswith('Fig. 6')
    assert log['crop'] == 'crops/springer-logo.png'
    assert log['suspected_misattached_caption']
    print('PASS: Springer-like logo with misattached Fig. 6 caption filtered; true plot/caption preserved')


def test_package_cost():
    structure, candidates, resolver = fixture()
    selection = reviewed_selection(structure, candidates, new_review(structure, candidates), resolver)
    preflight = transmission_tables(structure, selection, 120_000)
    docs = preflight['packets_raw']
    assert docs
    package = {'paper_id': 'paper-offline', 'structure': structure,
               'selection': selection, 'documents': docs}
    with tempfile.TemporaryDirectory() as tmp:
        # Bypass constructor's unrelated Docling/schema setup; test the actual
        # Pipeline public method with a no-network TokenCounter fallback.
        pl = object.__new__(Pipeline)
        pl.cfg = SimpleNamespace(model='claude-sonnet-5-5', cache_dir=Path(tmp),
                                 parser='docling', table_vision='never',
                                 figure_vision=False, max_chars_per_call=120_000)
        pl.__dict__['llm'] = SimpleNamespace(client=None)
        pl.__dict__['system'] = 'SYSTEM CONTENT'
        result = pl.estimate_cost_from_preflight(package, show=False)
        assert result['source'] == 'preflight_package'
        assert result['packet_count'] == len(docs)
        assert result['rows'][0]['doc_tokens']
        assert result['rows'][0]['saved'] is False
        assert result['calls'] == len(docs)*4
        assert result['high_usd'] >= result['low_usd'] > 0
        assert 'EXACT preflight preview packets' in result['text']
        # A stale or edited notebook packet cannot silently yield a different
        # price from the evidence currently selected for extraction.
        stale = {**package, 'documents': [{**docs[0], 'text': 'MODIFIED'}]}
        try:
            pl.estimate_cost_from_preflight(stale, show=False)
        except ValueError as exc:
            assert 'rerun the transmission preview' in str(exc)
        else:
            raise AssertionError('Modified packet unexpectedly accepted')
    print('PASS: cost estimate uses exact preflight packets, ignores saved-pass short circuit, rejects stale text')


if __name__ == '__main__':
    test_logo_filter()
    test_package_cost()
