"""Offline v0.6 source-aware segmentation / concept review / ontology-fit tests.

No LLM calls, no Docling model downloads, no Neo4j connection.
"""
from __future__ import annotations
import argparse, tempfile
from pathlib import Path
from rdflib import Graph
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ontology_loader import Ontology, Profile
from src.ontology_resolver import OntologyResolver
from src.pipeline.evidence_context import enrich_structure
from src.pipeline.pre_llm_review import (make_candidates,new_review,reviewed_selection,
                                          set_block_decision,set_concept_decision,ReviewRequired,
                                          approved_documents)
from src.pipeline.source_rdf import write_source_rdf
from src.pipeline.scientific_context import local_acronyms


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--owl',type=Path,required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    profile=Profile(Ontology(args.owl),root/'ontology'/'slice_manifest.json')
    assert not profile.warnings,profile.warnings
    resolver=OntologyResolver(profile)
    for group,name in [('property','magnetoresistance'),('property','residual resistivity ratio'),
                       ('phenomenon','weak antilocalization'),('technique','Rietveld refinement')]:
        assert resolver.resolve(name,group)['status']=='exact_match',(group,name)
    structure={'title':'Evidence test','sections':[{'id':'SEC1','heading':'Methods and results','role':'results'}],
               'paragraphs':[{'id':'P1','section_id':'SEC1','role':'results','kind':'text','page':1,
                              'text':'Magnetoresistance (MR) increased to 45% at 2.5 K. It decreased to 25% at 50 K.'},
                             {'id':'P2','section_id':'SEC1','role':'results','kind':'text','page':1,
                              'text':'The tube pressure was 10 -5 Torr with heating at 60 °C/h. The model fitting parameter alpha was recorded.'},
                             {'id':'P3','section_id':'SEC1','role':'references','kind':'reference','page':2,
                              'text':'A. B. Smith, Sci. China Mater. 66, 2032 (2023). K. Jones, Phys. Rev. B (2021).'},
                             {'id':'P4','section_id':'SEC1','role':'results','kind':'table','page':2,
                              'text':'Table 2 HLN fitting', 'caption':'Table 2 HLN fitting',
                              'rows':[['Temperature (K)','α','L φ (nm)'],['2.5','- 0.41','44.9 (5)']]},
                             {'id':'P5','section_id':'SEC1','role':'results','kind':'figure','page':2,
                              'text':'Fig. 5 MR% versus magnetic field','caption':'Fig. 5 MR% versus magnetic field',
                              'classification':{'class':'line_chart','confidence':.98},'crop':'fig5.png'},
                             {'id':'P6','section_id':'SEC1','role':'results','kind':'figure','page':2,
                              'text':'[figure icon]','classification':{'class':'icon','confidence':.99},'caption':''}],
               'tables':[{'paragraph_id':'P4','caption':'Table 2 HLN fitting','page':2}],
               'figures':[{'paragraph_id':'P5','page':2,'caption':'Fig. 5 MR% versus magnetic field',
                           'classification':{'class':'line_chart','confidence':.98},'crop':'fig5.png'},
                          {'paragraph_id':'P6','page':2,'caption':'',
                           'classification':{'class':'icon','confidence':.99},'crop':'icon.png'}]}
    enrich_structure(structure)
    ref=[e for e in structure['semantic_evidence'] if e['paragraph_id']=='P3']
    assert len(ref)==1 and 'Phys. Rev. B' in ref[0]['sentence']
    pressure=next(e for e in structure['semantic_evidence'] if '10 -5 Torr' in e['sentence'])
    assert '10 -5 Torr' in pressure['value_mentions'] and '-5 T' not in pressure['value_mentions']
    assert '60 °C/h' in pressure['value_mentions']
    assert len(structure['plots'])==1
    assert structure['plots'][0]['y_axis']['label']=='MR%'
    assert structure['plots'][0]['x_axis']['label']=='magnetic field'
    assert len(structure['fitted_results'])==2
    assert structure['fitted_results'][1]['uncertainty_notation']=='(5)'
    assert any(a['acronym']=='MR' and a['expansion']=='Magnetoresistance' for a in local_acronyms(structure['semantic_evidence']))
    stage=make_candidates(structure,resolver)
    assert any(c['canonical_term']=='Magnetoresistance' for c in stage['concept_registry'])
    rev=new_review(structure,stage)
    automatic=reviewed_selection(structure,stage,rev,resolver)
    assert len(automatic['extract'])>=2 and not automatic['hints']
    block=next(x for x in stage['evidence_blocks'] if x['paragraph_id']=='P1')
    set_block_decision(rev,block['id'],'extract')
    concept=next(c for c in stage['concept_registry'] if c['canonical_term']=='Magnetoresistance')
    set_concept_decision(rev,stage,resolver,concept['id'],'approve',selected_group='property',
                         selected_iri=str('https://w3id.org/matsci/ontology/evidence/Magnetoresistance'))
    selection=reviewed_selection(structure,stage,rev,resolver)
    assert len(selection['extract'])>=2
    assert any(h.get('iri') for hs in selection['hints'].values() for h in hs)
    assert 'MR = Magnetoresistance' in approved_documents(structure,selection)[0]['text']
    with tempfile.TemporaryDirectory() as d:
        out=Path(d)/'source.ttl';count=write_source_rdf(structure,out,'test_paper')
        g=Graph();g.parse(out,format='turtle')
        assert len(g)==count and count>20
    print('PASS: source segmentation, value parsing, logos/axes, fit parameters, acronym resolution, optional block/concept review, RDF')

if __name__=='__main__':main()
