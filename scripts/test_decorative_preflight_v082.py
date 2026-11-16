"""v0.8.2: decorative Docling pictures never enter preflight or paid stages.

Run: python scripts/test_decorative_preflight_v082.py
No PDF, model key, or OWL file required.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.evidence_context import enrich_structure
from src.pipeline.source_objects import _visual_assessment
from src.pipeline.pre_llm_review import (
    make_candidates, new_review, reviewed_selection, approved_documents,
    review_issues, preflight_tables, figure_inspector)
from src.pipeline.source_rdf import write_source_rdf


def main():
    from types import SimpleNamespace
    st = {'title': 'Test', 'sections': [{'id':'S','heading':'Results','role':'results'}],
          'paragraphs':[], 'figures':[], 'tables':[]}
    examples = [
        ('P1', 'logo', '', 'FERMI LEVEL', True),
        ('P2', 'icon', '', 'DECORATIVE 5 K', True),
        ('P3', 'watermark', '', 'WATERMARK', True),
        ('P4', 'decorative', '', 'PAGE ORNAMENT', True),
        ('P5', 'logo', 'Fig. 5 SEM surface morphology', 'Fig. 5 SEM surface morphology', True),
        ('P6', 'photo', '', '[Figure 6 | photo]', False),
        ('P7', 'scatter_chart', 'Fig. 7 resistivity (%) vs magnetic field (T)',
         'Fig. 7 resistivity (%) vs magnetic field (T)', False),
        ('P8', 'logo', 'An unidentified decorative-like object',
         'An unidentified decorative-like object', True),
    ]
    for pid, cls, cap, text, _ in examples:
        st['paragraphs'].append({'id':pid, 'section_id':'S', 'role':'results', 'kind':'figure',
                                 'page':1, 'text':text, 'caption':cap, 'classification':{'class':cls}})
        st['figures'].append({'paragraph_id':pid, 'classification':{'class':cls}, 'caption':cap,
                              'page':1, 'crop':f'crops/{pid}.png', 'label':f'figure {pid}'})
    st['paragraphs'].append({'id':'P9','section_id':'S','role':'results','kind':'text',
                             'page':1, 'text':'Measured SEM morphology at 5 K.'})
    st['paragraphs'].append({'id':'P10','section_id':'S','role':'results','kind':'figure',
                             'page':1,'text':'[Figure | signature]', 'caption':'',
                             'classification':{'class':'signature'}})
    original_ids = {p['id'] for p in st['paragraphs']}
    st = enrich_structure(st)
    excluded = {'P1','P2','P3','P4','P5','P8','P10'}
    kept = original_ids-excluded
    assert {p['id'] for p in st['paragraphs']} == kept
    assert {f['paragraph_id'] for f in st['figures']} == {'P6','P7'}
    assert {x['paragraph_id'] for x in st['excluded_decorative_pictures']} == excluded
    assert st['visual_filter_summary']['excluded_decorative_pictures'] == 7
    assert 'P5' in {x['paragraph_id'] for x in st['excluded_decorative_pictures']}  # caption is not independent proof
    assert _visual_assessment({'classification':{'class':'logo'},'caption':'Fig. 6 SEM morphology'})[0] == 'decorative_logo'
    assert {e['paragraph_id'] for e in st['semantic_evidence']} == kept
    assert all(e['paragraph_id'] not in excluded for e in st['semantic_evidence'])
    assert all(b['paragraph_id'] not in excluded for b in st['evidence_blocks'])
    assert all(a['paragraph_id'] not in excluded for a in st['source_assertions'])
    assert not {p['source_paragraph_id'] for p in st['plots']}.intersection(excluded)
    # No hidden PR labels or evidence IDs are renumbered after filtering.
    assert 'P9:s1' in {e['id'] for e in st['semantic_evidence']}

    class Resolver:
        lookup={};aliases={}; entries={'property':[]};fingerprint='dummy'
        by_iri={}
        def resolve(self,term,group,context=''):
            return {'status':'unmatched','matched_iri':None,'candidates':[], 'reason':'offline fixture'}
        def _hits(self, group, term):return []
    resolver=Resolver()
    candidates=make_candidates(st,resolver)
    assert not {c['paragraph_id'] for c in candidates['candidates']}.intersection(excluded)
    assert not {b['paragraph_id'] for b in candidates['evidence_blocks']}.intersection(excluded)
    review=new_review(st,candidates)
    assert not {key.partition(':')[0] for key in review['evidence']}.intersection(excluded)
    assert not review_issues(st,candidates)['issue_ids'] or all(
        eid.partition(':')[0] not in excluded for eid in review_issues(st,candidates)['issue_ids'])
    selection=reviewed_selection(st,candidates,review,resolver)
    assert not {e['paragraph_id'] for e in selection['extract']}.intersection(excluded)
    assert not {e['paragraph_id'] for e in selection['context_only']}.intersection(excluded)
    assert not set(selection['approved_visuals']).intersection(excluded)
    packets=approved_documents(st,selection)
    combined='\n'.join(p['text'] for p in packets)
    assert 'DECORATIVE 5 K' not in combined
    assert 'PAGE ORNAMENT' not in combined
    assert 'FERMI LEVEL' not in combined
    views=preflight_tables(st,candidates,review)
    assert not set(views['evidence']['block_id']).intersection({'EB_'+pid for pid in excluded})
    assert st['visual_filter_summary']['excluded_with_attached_caption'] == 2
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        rdf=root/'source_records.ttl'
        write_source_rdf(st,rdf,'test')
        assert 'PLOT_P1' not in rdf.read_text()
        # Re-enriching is idempotent: the audit isn't duplicated.
        again=enrich_structure(st)
        assert len(again['excluded_decorative_pictures']) == 7
        try:
            import ipywidgets
        except ImportError:
            pass
        else:
            widget=figure_inspector(st,root)
            show_all=widget.children[1];choice=widget.children[2]
            assert not any('excluded decorative' in lbl for lbl,_ in choice.options)
            show_all.value=True
            assert sum('excluded decorative' in lbl for lbl,_ in choice.options)==7
    print('PASS: decorative figures excluded before evidence/candidates/review/RDF/LLM; retained in optional audit')

if __name__=='__main__':main()
