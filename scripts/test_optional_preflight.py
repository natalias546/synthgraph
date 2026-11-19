"""v0.8 offline tests: automatic source inclusion, binary review, optional inspection.

Usage: python scripts/test_optional_preflight.py
No OWL file or model key needed. ipywidgets rendering is optional.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline.pre_llm_review import (
    new_review, reviewed_selection, approved_documents, preflight_tables,
    transmission_tables, review_issues, review_summary,
    set_block_decision, set_evidence_decision, set_candidate_decision, set_review_status,
    read_or_create_review, figure_inspector
)
from src.pipeline.io_utils import write_json
from src.pipeline.evidence_context import enrich_structure


def fixture():
    st={'title':'Reviewability','sections':[{'id':'S1','role':'results','heading':'Results'}],
        'paragraphs':[], 'semantic_evidence':[], 'figures':[], 'tables':[]}
    records=[('P1','results','text','Our SEM measurement revealed surface morphology.', 'result'),
             ('P2','results','text','EDAX results showed elemental composition.', 'result'),
             ('P3','references','reference','Cited previous work on microscopy.', 'bibliographic_entry'),
             ('P4','results','text','Another measurement found an unknown feature at 10 K.', 'result')]
    for pid,role,kind,sentence,scientific_role in records:
        st['paragraphs'].append({'id':pid,'section_id':'S1','role':role,'kind':kind,'page':1,'text':sentence})
        st['semantic_evidence'].append({'id':pid+':s1','paragraph_id':pid,'section_id':'S1',
            'section_role':role,'source_kind':kind,'page':1,'span_start':0,'span_end':len(sentence),
            'sentence':sentence,'scientific_roles':[scientific_role],'review_status':'not reviewed'})
    candidates={'candidate_fingerprint':'C1','resolver_fingerprint':'R1',
        'candidates':[{'id':'C1','evidence_id':'P1:s1','term':'SEM','group':'technique',
                       'match_status':'exact_match','matched_iri':'https://test/SEM',
                       'suggestions':[{'group':'technique','iri':'https://test/SEM','label':'SEM'}],
                       'concept_id':'CC1'},
                      {'id':'C2','evidence_id':'P2:s1','term':'EDAX','group':'technique',
                       'match_status':'exact_match','matched_iri':'https://test/EDX',
                       'suggestions':[{'group':'technique','iri':'https://test/EDX','label':'EDS'}],
                       'concept_id':'CC2'}],
        'concept_registry':[{'id':'CC1','canonical_term':'SEM','group':'technique',
              'evidence_ids':['P1:s1'],'aliases':['SEM'],'matched_iri':'https://test/SEM',
              'match_status':'exact_match'},
              {'id':'CC2','canonical_term':'EDAX','group':'technique',
              'evidence_ids':['P2:s1'],'aliases':['EDAX'],'matched_iri':'https://test/EDX',
              'match_status':'exact_match'}],
        'evidence_blocks':[{'id':'EB_'+pid,'paragraph_id':pid,'page':1,'section_role':role,
                            'evidence_ids':[pid+':s1'],'source_kind':kind}
                           for pid,role,kind,_,_ in records],
        'acronyms':[],'source_assertions':[],'relation_candidates':[],
        'specimen_mentions':[],'document_reference_targets':[]}
    resolver=SimpleNamespace(entries={'technique':[]}, by_iri={('technique','https://test/SEM'):SimpleNamespace(label='SEM'),
                                                               ('technique','https://test/EDX'):SimpleNamespace(label='EDS')})
    return st,candidates,resolver


def main():
    # A method-section list item is scientific text, not a bibliography entry.
    bullet = enrich_structure({'title':'Methods note', 'sections':[{'id':'S', 'role':'methods', 'heading':'Methods'}],
        'paragraphs':[{'id':'P1','section_id':'S','role':'methods','kind':'list_item',
                       'page':1,'text':'The sample was measured with SEM at 2 K.'}]})
    assert bullet['semantic_evidence'][0]['scientific_roles'] != ['bibliographic_entry']
    from src.pipeline.pre_llm_review import default_evidence_decision
    assert default_evidence_decision(bullet['semantic_evidence'][0]) == 'extract'
    st,c,r=fixture()
    review=new_review(st,c)
    assert {v['review_status'] for v in review['evidence'].values()}=={'not reviewed'}
    assert all(v['decision']=='pending' for v in review['evidence'].values())
    issue=review_issues(st,c)
    assert 'EB_P2' in issue['block_reasons']  # EDAX alias context check
    assert 'EB_P4' in issue['block_reasons']  # scientific sentence without recognized term
    assert 'EB_P1' not in issue['block_reasons']
    selection=reviewed_selection(st,c,review,r)
    assert selection['counts']['extract']==3, selection['counts']
    assert selection['counts']['exclude']==1
    assert not selection['hints']  # no unreviewed IRI is treated as human approved
    assert selection['approved_visuals']==[]
    docs=approved_documents(st,selection)
    assert docs and 'Our SEM' in docs[0]['text']
    assert 'EDAX results' in docs[0]['text']
    assert 'Cited previous work' not in docs[0]['text']
    assert 'unknown feature' in docs[0]['text']
    views=preflight_tables(st,c,review)
    assert len(views['candidates'])==2 and len(views['evidence'])==4
    assert 'review_status' in views['candidates'].columns
    assert set(views['evidence'].review_status)=={'not reviewed'}
    sent=transmission_tables(st,selection)
    assert len(sent['sentences'])==3 and len(sent['packets'])==1
    assert sent['packets'].iloc[0]['exact_llm_input']==docs[0]['text']
    set_review_status(review,'evidence','P1:s1',True)
    assert reviewed_selection(st,c,review,r)['counts']['extract']==3
    set_evidence_decision(review,'P2:s1','exclude',note='ambiguous term')
    set_block_decision(review,'EB_P1','context_only')
    selection=reviewed_selection(st,c,review,r)
    assert selection['counts']['extract']==1
    assert selection['counts']['context_only']==1
    assert selection['counts']['exclude']==2
    assert selection['context_only'][0]['review_status']=='reviewed'
    assert selection['counts']['auto_included']==1
    set_evidence_decision(review,'P1:s1','extract')  # leaf override must beat block
    selection=reviewed_selection(st,c,review,r)
    assert selection['counts']['extract']==2
    set_candidate_decision(review,c,r,'C1','approve',selected_group='technique',selected_iri='https://test/SEM')
    selection=reviewed_selection(st,c,review,r)
    assert any(h['kind']=='reviewed_iri_hint' for values in selection['hints'].values() for h in values)
    assert all(v in ('not reviewed','reviewed') for k in ('evidence','concepts','candidates')
               for v in review_summary(review,c)['review_status'][k])
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'pre_llm_review.json'
        write_json(path,review)
        assert read_or_create_review(path,st,c)['evidence']['P2:s1']['decision']=='exclude'
        # A Docling crop is a saved local file; inspection never opts in to paid vision.
        from PIL import Image
        root=Path(tmp);(root/'crops').mkdir()
        Image.new('RGB',(16,16),(255,255,255)).save(root/'crops'/'figure_001_p1.png')
        st['figures']=[{'paragraph_id':'PF','page':1,'label':'Figure 1',
                       'crop':'crops/figure_001_p1.png','caption':'X-ray intensity vs angle'}]
        try:
            import ipywidgets
        except ImportError:
            print('SKIP optional ipywidgets figure UI (dependency absent); crop source verified')
        else:
            assert figure_inspector(st,root) is not None
        assert (root/'crops'/'figure_001_p1.png').exists()
    print('PASS: optional review, exact pandas packet preview, binary status, exclusions, aliases, unchanged visual gate')

if __name__=='__main__':
    main()
