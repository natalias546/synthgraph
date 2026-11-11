"""Offline regression for v0.4: pass-3 survival, evidence, ontology mapping, and export split.

Usage: python scripts/test_semantic_integration.py --owl path/to/v11.owl
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import extract
from src.ontology_loader import Ontology, Profile
from src.ontology_resolver import OntologyResolver, annotate_validation
from src.pipeline.evidence_context import enrich_structure
from src.pipeline.export import GraphBuilder
from src.pipeline.resolve import resolve_paper
from src.pipeline.runner import Pipeline
from src.pipeline.schemas import build_schemas
from src.pipeline.validate import validate_paper

ROOT = Path(__file__).resolve().parents[1]


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--owl', type=Path, default=ROOT / 'ontology' / 'matsci-v11-parsing-aligned-cleaned.owl')
    args = arg.parse_args()
    if not args.owl.exists():
        arg.error('Provide v11 OWL using --owl; this patch does not ship the large ontology.')
    onto = Ontology(args.owl)
    profile = Profile(onto, ROOT / 'ontology' / 'slice_manifest.json')
    schemas = build_schemas(profile)
    assert len(profile.groups['application'].class_iris) > 0
    app = profile.groups['application'].labels[0]
    material = profile.groups['material'].labels[0]
    prop = 'Resistivity' if 'Resistivity' in profile.groups['property'].labels else profile.groups['property'].labels[0]
    evidence_text = ('We measured resistivity at 2 K. Magnetoresistance increased under magnetic field. '
                     'This material may be useful for spintronic applications.')
    structure = enrich_structure({'title': 'Test', 'sections': [{'id': 'SEC1', 'heading': 'Results', 'role': 'results'}],
                                 'paragraphs': [{'id': 'P1', 'kind': 'paragraph', 'role': 'results',
                                                 'page': 1, 'section_id': 'SEC1', 'text': evidence_text}]})
    spans = structure['semantic_evidence']
    assert len(spans) == 3 and spans[0]['sentence'] == 'We measured resistivity at 2 K.'
    assert evidence_text[spans[1]['span_start']:spans[1]['span_end']] == spans[1]['sentence']
    assert spans[2]['preceding_sentence'] == spans[1]['sentence']
    assert spans[2]['scientific_roles']
    quote = 'We measured resistivity at 2 K.'
    e = [{'paragraph_id': 'P1', 'quote': quote}]
    p1 = {'materials': [{'id': 'M1', 'name': 'Ni3In2S2', 'ontology_class': material,
                         'formula': 'Ni3In2S2', 'aliases': [], 'evidence': e, 'confidence': .97}],
          'samples': [], 'process_steps': [], 'unresolved_notes': []}
    p2 = {'measurements': [{'id': 'ME1', 'technique_class': profile.groups['technique'].labels[0],
                            'technique_name_in_text': 'measurement', 'property_class': prop,
                            'property_name_in_text': 'Resistivity', 'value': 4.0,
                            'value_text': '4 Ω', 'unit': 'Ω', 'sample_ids': [], 'material_ids': ['M1'],
                            'conditions': [], 'settings': [], 'data_origin': 'experimental',
                            'source_location': 'text', 'evidence': e, 'confidence': .97}],
          'unresolved_notes': []}
    result = {'entities': [], 'relations': [], 'rejected_candidates': [
                  {'subject_id': 'M1', 'predicate': 'composed_of', 'object_id': 'X2',
                   'reason': 'Topological state is not a material component'}],
              'applications': [{'id': 'A1', 'ontology_class': app, 'name': 'spintro',
                                'material_ids': ['M1'], 'sample_ids': [], 'application_status': 'proposed',
                                'evidence': [{'paragraph_id': 'P1', 'quote': 'may be useful for spintronic applications'}],
                                'confidence': .86}],
              'comparisons': [{'id': 'CMP1', 'subject_id': 'M1', 'comparator_id': 'M1',
                               'property_name': 'magnetoresistance', 'trend': 'different',
                               'baseline_value_text': '3%', 'target_value_text': '5%',
                               'evidence': [{'paragraph_id': 'P1', 'quote': 'Magnetoresistance increased under magnetic field.'}],
                               'confidence': .90}],
              'unresolved_terms': [{'id': 'U1', 'raw_term': 'magnetoresistance',
                                    'reason_unresolved': 'Specific ontology class absent',
                                    'evidence': [{'paragraph_id': 'P1', 'quote': 'Magnetoresistance increased under magnetic field.'}]}]}
    # Verify Pydantic output roundtrips all 6 collections unchanged. Pass 3 is two calls with two schemas
    # (one schema with all six lists exceeds the API's grammar size limit): 3a relations, 3b applications.
    part_a = ('entities', 'relations', 'rejected_candidates', 'unresolved_terms')
    part_b = ('applications', 'comparisons')
    valid = {**schemas['relations'].model_validate({k: result[k] for k in part_a}).model_dump(),
             **schemas['applications_comparisons'].model_validate({k: result[k] for k in part_b}).model_dump()}
    assert set(part_a + part_b) <= valid.keys()

    original = extract._ask_chunks
    try:
        extract._ask_chunks = lambda *args, **kwargs: [({'chunk': 0}, copy.deepcopy(valid))]
        p3 = extract.run_pass3(None, schemas, '', [], {}, profile, p1, p2, 'paper')
    finally:
        extract._ask_chunks = original
    assert all(len(p3[k]) == 1 for k in ('applications', 'comparisons', 'unresolved_terms', 'rejected_candidates'))
    assert p3['applications'][0]['material_ids'] == ['M1']
    original = extract._ask_chunks
    try:
        extract._ask_chunks = lambda *args, **kwargs: [
            ({'chunk': i}, copy.deepcopy(valid)) for i in (0, 1)]
        multi = extract.run_pass3(None, schemas, '', [], {}, profile, p1, p2, 'paper')
    finally:
        extract._ask_chunks = original
    assert [a['id'] for a in multi['applications']] == ['c0_A1', 'c1_A1']
    assert [u['id'] for u in multi['unresolved_terms']] == ['c0_U1', 'c1_U1']
    assert [c['id'] for c in multi['comparisons']] == ['c0_CMP1', 'c1_CMP1']
    assert all(a['material_ids'] == ['M1'] for a in multi['applications'])

    resolver = OntologyResolver(profile)
    aligned = resolver.resolve_extraction({'pass1': p1, 'pass2': p2, 'pass3': p3})
    val = validate_paper(aligned, structure['paragraphs'], profile, .85)
    val = annotate_validation(val, aligned)
    items = {it['kind']: it for it in val['items']}
    for kind in ('application', 'comparison', 'unresolved_term'):
        assert kind in items, f'Lost {kind} through validation'
        assert items[kind]['status'] != 'auto_accepted', kind
    assert items['material']['evidence_anchors'][0]['span_start'] == 0
    assert items['material']['evidence_anchors'][0]['evidence_sentence_ids'] == ['P1:s1']
    assert items['comparison']['status'] == 'needs_review'
    assert items['unresolved_term']['status'] == 'needs_review'
    res = resolve_paper(val)
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)
        # Exercise real Pipeline.export behavior with a real ontology/profile and no LLM.
        pl = Pipeline.__new__(Pipeline)
        pl.__dict__['profile'] = profile
        pl.cfg = SimpleNamespace(graph_dir=out, outputs_dir=out)
        pl.manifest = lambda: {'pipeline_version': 'test'}
        info = pl.export({'paper': {'resolve': res, 'meta': {'title': 'Test', 'pdf':'test.pdf', 'sha256':'deadbeef'}}},
                         out_dir=out)
        assert info['candidate_graph']['nodes'] >= info['nodes']
        curated = (out / 'nodes.jsonl').read_text()
        candidate = (out / 'candidates' / 'nodes.jsonl').read_text()
        assert '"Application"' not in curated and '"Comparison"' not in curated
        assert '"Application"' in candidate and '"Comparison"' in candidate
        assert '"UnresolvedTerm"' in candidate
        assert (out / 'graph.ttl').is_file() and (out / 'candidates' / 'graph.ttl').is_file()
        assert (out / 'review_queue.csv').is_file() and (out / 'ontology_gaps.csv').is_file()
        from rdflib import Graph
        Graph().parse(out / 'graph.ttl', format='turtle')
        Graph().parse(out / 'candidates' / 'graph.ttl', format='turtle')
    print('PASS: v11 ontology/profile and Pydantic structured schema')
    print('PASS: exact sentence offsets, context and scientific roles')
    print('PASS: pass-3 specialized records preserved through validation')
    print('PASS: source quotes linked to sentence IDs, provisional data segregated')
    print('PASS: both RDF graphs parse; no extra LLM calls')

if __name__ == '__main__':
    main()
