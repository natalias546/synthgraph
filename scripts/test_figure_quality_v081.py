"""Offline figure classification, parenthesized units, and review summary tests."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline.source_objects import (
    _visual_assessment, _caption_axis, _apply_axis_unit, _pdf_axis_unit_candidates)
from src.pipeline.evidence_context import enrich_structure
from src.pipeline.pre_llm_review import default_evidence_decision, review_summary_table, review_issues
from src.pipeline.pre_llm_review import new_review
from scripts.test_optional_preflight import fixture


def main():
    x,y,source=_caption_axis('Fig. 2 Magnetoresistance (%) versus magnetic field (T).')
    assert _apply_axis_unit(x,None,source)['unit']=='T'
    assert _apply_axis_unit(y,None,source)['unit']=='%'
    x,y,source=_caption_axis('Intensity (a.u.) vs 2θ (deg)')
    assert _apply_axis_unit(x,None,source)['unit']=='deg'
    assert _apply_axis_unit(y,None,source)['unit']=='a.u.'
    x,y,source=_caption_axis('MR% versus H (T) plots')
    import re
    x=re.sub(r'\s+(?:curve|plots?|graph|diagram)\b.*$', '',x,flags=re.I).strip()
    assert _apply_axis_unit(x,None,source)['unit']=='T'
    assert _apply_axis_unit('Intensity (inset)',None,source)['unit'] is None
    pdf='Magnetic field (T)\nResistivity (μΩ cm)\nLegend: 10 K'
    candidates=_pdf_axis_unit_candidates(pdf)
    assert ('Magnetic field','T') in {(r['label'],r['unit']) for r in candidates}
    assert _apply_axis_unit('Magnetic field',pdf,'caption')['unit']=='T'
    assert _apply_axis_unit('Resistivity',pdf,'caption')['unit']=='μΩ cm'
    assert _apply_axis_unit('Intensity',pdf,'caption')['unit'] is None
    assert _visual_assessment({'classification':{'class':'logo'},
        'caption':'Fig. 3 SEM images of the crystal morphology'})[0]=='decorative_logo'
    assert _visual_assessment({'classification':{'class':'logo'},'caption':''})[0]=='decorative_logo'
    assert _visual_assessment({'classification':{'class':'unknown'},'caption':''})[0]=='unverified_picture'
    st={'title':'Figure quality','sections':[{'id':'S','role':'results','heading':'Results'}],
        'paragraphs':[], 'figures':[], 'tables':[]}
    fixtures=[('P1','[Figure 1 | logo]','',{'class':'logo'},'decorative_logo'),
      ('P2','[Figure 2 | logo] Fig. 2 Resistivity (μΩ cm) versus temperature (K)',
       'Fig. 2 Resistivity (μΩ cm) versus temperature (K)',{'class':'logo'},'decorative_logo'),
      ('P3','[Figure 3]','',{'class':'photo'},'unverified_picture')]
    for pid,text,caption,cls,expected in fixtures:
        st['paragraphs'].append({'id':pid,'section_id':'S','role':'results','kind':'figure',
              'page':1,'text':text,'caption':caption,'classification':cls})
        st['figures'].append({'paragraph_id':pid,'page':1,'caption':caption,'classification':cls})
    enrich_structure(st)
    by_p={e['paragraph_id']:e for e in st['semantic_evidence']}
    for pid,_,_,_,role in fixtures:
        if role in ('decorative_logo', 'decorative_icon'):
            assert pid not in by_p
            assert any(x['paragraph_id']==pid and x['visual_role']==role
                       for x in st['excluded_decorative_pictures'])
        else:
            assert by_p[pid]['visual_role']==role
    assert 'P2' not in by_p  # the caption cannot redeem a publisher logo
    assert default_evidence_decision(by_p['P3'])=='exclude'
    assert len(st['plots'])==0
    assert any(x['paragraph_id']=='P2' and x['suspected_misattached_caption']
               for x in st['excluded_decorative_pictures'])
    paper,candidates,resolver=fixture()
    summary=review_summary_table(new_review(paper,candidates),candidates)
    assert {'category','reviewed','not_reviewed','total'}.issubset(summary.columns)
    assert (summary['reviewed']==0).all()
    print('PASS: logo/figure disagreements, preflight decorative exclusions, caption and vector units, review status summary')

if __name__=='__main__':main()
