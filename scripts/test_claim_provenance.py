"""Regression tests for deterministic claim provenance and routing (offline)."""
from __future__ import annotations
import ast, tempfile, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rdflib import Graph, RDF, URIRef
from src.pipeline.evidence_context import enrich_structure
from src.pipeline.claim_provenance import classify_surface, link_specimen_mentions
from src.pipeline.source_rdf import write_source_rdf


def fake_structure(sentences):
    roles=['results','introduction','methods','results']
    return {'title':'Source provenance test','sections':[{'id':'S','role':'results','heading':'Results'}],
        'paragraphs':[{'id':f'P{i:04d}','section_id':'S','role':roles[(i-1)%len(roles)],
                       'kind':'text','text':t,'page':1} for i,t in enumerate(sentences,1)]}


def main():
    test=[
      'Our measured resistivity is consistent with the value reported by Zhang et al. [12].',
      'Previous studies reported a transition temperature of 14 K [2].',
      'We calculated a band gap of approximately 2.58 eV.',
      'Our SEM measurements of Ni3In2S2 crystal were conducted using Zeiss EVO-50.',
      'See Fig. 2 inset and supplementary information for additional results.',
      'No weak antilocalization effect was observed in our sample.',
      'The Fermi level shifted by 0.5 eV along the c-axis.',
      'The measured result was not found in the cited reference [4].',
    ]
    st=enrich_structure(fake_structure(test))
    evid=st['semantic_evidence']; assert len(evid)>=len(test)
    claims=st['source_assertions']; assert claims
    origins={x['claim_origin'] for x in claims}
    assert {'current_study','cited_study','mixed_current_and_cited'} <= origins, origins
    assert any(r['kind']=='supplement' for r in st['document_references'])
    assert any(r['kind']=='figure' for r in st['document_references'])
    assert any(r['kind']=='bibliography' and r['target_label']=='12' for r in st['document_references'])
    assert any(c['epistemic_status']=='hypothetical' for c in claims) is False
    assert any(c['epistemic_status']=='negated' for c in claims)
    assert any(c['numeric_approximation'] for c in claims)
    assert any(c['claim_kind']=='comparison' for c in claims)
    assert any(r['status']=='requires_operand_review' for r in st['relation_candidates'])
    assert classify_surface('consistent',test[0])['route']=='comparison_operator'
    assert classify_surface('inset',test[4])['route']=='document_reference'
    assert classify_surface('well','A quantum well was observed.')['route']=='scientific_entity'
    assert classify_surface('order','Magnetic order was measured.')['route']=='scientific_entity'
    assert classify_surface('contrast','The SEM image contrast was increased.')['route']=='scientific_entity'
    assert classify_surface('formation','Formation energy decreased.')['route']=='scientific_entity'
    assert classify_surface('sample',test[5])['route']=='sample_reference'
    p1={'materials':[{'id':'M1','formula':'Ni3In2S2','name':'Ni3In2S2'}],
        'samples':[{'id':'S1','material_ids':['M1'],'name_in_paper':'Ni3In2S2 crystal'}]}
    links=link_specimen_mentions(st,p1)
    assert any(x['matched_sample_id']=='S1' for x in links), links
    p1b={**p1,'samples':p1['samples']+[{'id':'S2','material_ids':['M1'],'name_in_paper':'Ni3In2S2 flake'}]}
    links2=link_specimen_mentions(st,p1b)
    assert any(x['match_status']=='ambiguous_multiple_samples' for x in links2)
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/'source.ttl'; write_source_rdf(st,f,'paper_a')
        rdf=Graph().parse(f,format='turtle')
        assert list(rdf.subjects(RDF.type,URIRef('https://w3id.org/matsci/ontology/evidence/ClaimAttributionRecord')))
    # Real supplied paper fixture can be loaded from developer scratch, if present.
    real=Path('/mnt/data/Pasted text.txt')
    if real.exists():
        obj=ast.literal_eval(real.read_text(encoding='utf-8'))
        ss=enrich_structure(obj['structure'])
        assert len(ss['bibliographic_entries'])>=25
        assert len(ss['source_assertions'])>=50
        assert any(r['kind']=='bibliography' for r in ss['document_references'])
        assert all(a['assertion_status']=='candidate_unverified' for a in ss['source_assertions'])
        print('REAL PAPER:',len(ss['source_assertions']),'assertion proposals; ',len(ss['document_reference_targets']),'grouped targets')
    print('PASS source claim provenance, citation groups, specimen links, scientific routing and provisional RDF')

if __name__=='__main__':main()
