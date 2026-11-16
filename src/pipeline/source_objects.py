"""Build explicit source-level Equation, Plot and FittingResult descriptions.

No image digitization, graph point inference, or formula-enrichment model calls.
Axes derived from captions are tagged `caption_inference_unverified`.
"""
from __future__ import annotations
import re
from .scientific_context import quantities

_AXIS_SPLIT = re.compile(r"(?P<y>[A-Za-zαβρΔσ𝜎𝜙φ()/_\s%+-]{1,65}?)\s+(?:versus|vs\.?|against|as a function of)\s+(?P<x>[^,.]{2,90})", re.I)
_FUNC = re.compile(r"(?P<y>[A-Za-zαβρΔσ𝜎𝜙φ]+)\s*\(\s*(?P<x>[TH])\s*\)",re.I)
_EQUAL = re.compile(r"(?:(?:\b[A-ZαβγΔσρ𝜎]\s*\([^)]{1,30}\)|[αβγΔσρ𝜎LφϕBφ][\wφϕ]*)\s*=\s*[^,;\.]{3,180})")
_FORMULA_INTRO = re.compile(r"(?:equation|formula|expression).{0,140}?(?:given\s+by|written\s+as|defined\s+as|as\s+follows)\s*:?\s*$", re.I)
_DEF = re.compile(r"\b(?:where|here|denotes?|represents?|is defined as|refers to|parameter|prefactor|fitting)\b", re.I)
_FIT = re.compile(r"\b(?:Hikami[ -]Larkin[ -]Nagaoka|HLN)\b",re.I)


def _visual_role(fig:dict) -> str:
    name=str(fig.get('classification',{}).get('class','')).lower()
    cap=fig.get('caption','') or ''
    cl=cap.lower()
    if name=='icon' and not cap: return 'decorative_icon'
    if 'logo' in name and cap: return 'caption_fragment_not_verified_as_graphic'
    if 'logo' in name and not re.search(r"\b(?:fig\.?|plot|versus|curve|spectrum|diffraction|schematic|fitted)\b",cl):
        return 'decorative_logo'
    if re.search(r"\b(?:plot|vs\.?|versus|curve|spectrum|spectra|diffraction|magnetoconductivity|fit)\b",cl) or name in ('line_chart','scatter_chart','bar_chart','histogram'):
        return 'scientific_plot_candidate'
    if 'logo' in name and cap:
        return 'caption_fragment_not_verified_as_graphic'
    if 'logo' in name:return 'decorative_logo'
    return 'scientific_figure_or_unknown'


def _caption_axis(text:str)->tuple[str|None,str|None,str]:
    match=_AXIS_SPLIT.search(text)
    if match:
        y=match.group('y').strip(' :-\n')
        x=match.group('x').strip(' :-\n')
        # Avoid copying the entire sentence as an axis.
        y=y.split(' Fig.')[-1]
        y=re.sub(r"^.*?\b(?:Fig\.?\s*\d+[a-z]?)\s*",'',y,flags=re.I)
        x=re.split(r"\s+(?:of|for|in|of synthesized|obtained|at temperatures?)\b", x, maxsplit=1, flags=re.I)[0]
        return x[:90],y[:70], 'caption_inference_unverified'
    f=_FUNC.search(text)
    if f:return f.group('x'),f.group('y'), 'caption_inference_unverified'
    return None,None,'not_extracted'


def enrich_source_objects(structure:dict)->dict:
    paragraphs=structure.get('paragraphs',[])
    by_id={p['id']:p for p in paragraphs}
    evidence=structure.get('semantic_evidence',[])
    eqs=[]
    for index,p in enumerate(paragraphs):
        text=p.get('text','')
        kind=p.get('kind','')
        matches=[]
        if kind=='formula':
            matches=[(0,len(text),text)] if text.strip() else []
        elif kind=='text' and p.get('role')!='references':
            for m in _EQUAL.finditer(text):
                if len(m.group())>5:
                    raw=m.group()
                    cut=re.search(r'\s+(?:suggests?|implies?|yields?|is\s+found|indicates?|shows?|signifies?|represents?)\b',raw,re.I)
                    if cut: raw=raw[:cut.start()]
                    if len(raw)>5:matches.append((m.start(),m.start()+len(raw),raw))
            # A missing formula is still an explicit gap, not a guessed equation.
            if not matches and _FORMULA_INTRO.search(text):
                matches=[(len(text),len(text),'')]
        for n,(a,b,value) in enumerate(matches,1):
            near=[x for x in paragraphs[max(index-1,0):min(index+2,len(paragraphs))] if x['id']!=p['id'] and x.get('section_id')==p.get('section_id') and x.get('kind')=='text']
            nearby_defs=[{'paragraph_id':x['id'],'text':x['text'][:350]} for x in near if _DEF.search(x.get('text',''))]
            nearby_defs += [{'paragraph_id':p['id'],'text':t[:280]} for t in re.split(r'(?<=[.!?])\s+',text) if _DEF.search(t)][:3]
            eqs.append({'id':f'EQ_{p["id"]}_{n}','source_paragraph_id':p['id'],'source_kind':'equation',
                        'section_id':p.get('section_id'),'page':p.get('page'),
                        'span_start':a,'span_end':b,'expression_text':value,
                        'expression_status': 'raw_source_unverified' if value else 'missing_expression_from_pdf_text',
                        'definition_context':nearby_defs[:6], 'evidence_ids':p.get('evidence_ids',[]),
                        'review_status':'not reviewed'})
    plots=[]
    for fig in structure.get('figures',[]):
        pid=fig.get('paragraph_id')
        classification=fig.get('classification') or {}
        role=_visual_role(fig)
        fig['visual_role']=role
        if pid in by_id:by_id[pid]['visual_role']=role
        caption=fig.get('caption') or ''
        x,y,origin=_caption_axis(caption)
        simple=re.search(r'\b([A-Za-zρσM]{1,3})\s*[-–]\s*([HT])\s+plot\b',caption,re.I)
        if simple and not x:
            y,x=simple.group(1),simple.group(2)
            origin='caption_inference_unverified'
        if x:
            x=re.sub(r'\s+(?:curve|plots?|graph|diagram)\b.*$', '', x, flags=re.I).strip()
        if role in {'decorative_logo','decorative_icon'}:
            continue
        typ=('line' if 'line_chart' in str(classification) or re.search(r'\bcurve|plot\b',caption,re.I)
             else 'spectrum' if re.search(r'spectrum|spectra',caption,re.I)
             else 'schematic' if 'schematic' in caption.lower() else 'undetermined')
        panels=[]
        if re.search(r'\binset\s+(?:shows?|display)\b',caption,re.I):
            inset=re.search(r'\b([A-Za-zρσM]{1,3})\s*[-–]\s*([HT])\s+plot\b',caption[caption.lower().find('inset'):],re.I)
            panels.append({'panel_label':'inset', 'x_axis_label':inset.group(2) if inset else None,
                           'y_axis_label':inset.group(1) if inset else None,
                           'source':'caption_inference_unverified'})
        for m in re.finditer(r'(?<!\w)\(?([a-d])\)?\s+(?=\w)',caption):
            if m.group(1) not in [q['panel_label'] for q in panels]:
                panels.append({'panel_label':m.group(1),'source':'caption_inference_unverified'})
        plots.append({'id':'PLOT_'+str(pid),'source_paragraph_id':pid,'page':fig.get('page'),
                      'caption':caption,'title':None,'plot_type':typ,
                      'visual_role':role,'crop':fig.get('crop'),
                      'classification':classification,
                      'x_axis':{'label':x,'unit':None,'source':origin},
                      'y_axis':{'label':y,'unit':None,'source':origin},
                      'panels':panels,'represented_quantity':y,
                      'evidence_ids':by_id.get(pid,{}).get('evidence_ids',[]),
                      'verification_status':'image_not_examined', 'review_status':'not reviewed'})
    # Pair caption-only fragments with the closest science image on the same page;
    # this is a suggestion, never verified as a visual match.
    by_position={p['id']:i for i,p in enumerate(paragraphs)}
    for cap in plots:
        if cap['visual_role']!='caption_fragment_not_verified_as_graphic':continue
        candidates=[x for x in plots if x['visual_role']=='scientific_plot_candidate'
                    and x['page']==cap['page'] and x['source_paragraph_id']!=cap['source_paragraph_id']]
        if candidates:
            nearest=min(candidates,key=lambda x:abs(by_position.get(x['source_paragraph_id'],100000)-by_position.get(cap['source_paragraph_id'],0)))
            cap['paired_image_paragraph_id']=nearest['source_paragraph_id']
            cap['pairing_status']='nearest_same_page_candidate_unverified'
            nearest.setdefault('caption_candidate_ids',[]).append(cap['source_paragraph_id'])
    # Table-native fitting parameters: data provenance is a cell, not a made-up sentence.
    fit_records=[]
    for tab in structure.get('tables',[]):
        pid=tab.get('paragraph_id'); p=by_id.get(pid,{})
        rows=p.get('rows') or []
        caption=(tab.get('caption') or '').strip()
        if not rows or not _FIT.search(caption+' '+p.get('text','')): continue
        header=[x.strip() for x in rows[0]]
        for i,row in enumerate(rows[1:],start=1):
            temp=row[0] if row else ''
            for col in range(1,min(len(row),len(header))):
                raw=str(row[col]).strip(); match=re.search(r'[-+]?\s*\d+(?:\.\d+)?',raw)
                uncertainty=re.search(r'\(\s*(\d+)\s*\)',raw)
                fit_records.append({'id':f'FIT_{pid}_r{i}_c{col}', 'source_paragraph_id':pid,
                                    'source_kind':'model_fitting_result','fitted_model':'HLN',
                                    'fitted_model_name':'Hikami-Larkin-Nagaoka model',
                                    'parameter_symbol':header[col],
                                    'parameter_definition': ('HLN prefactor' if 'α' in header[col] else
                                                             'phase coherence length' if re.search(r'L\s*[φϕ]',header[col]) else None),
                                    'value_text':raw,'value':float(match.group().replace(' ','')) if match else None,
                                    'uncertainty_notation':uncertainty.group() if uncertainty else None,
                                    'unit':'nm' if 'nm' in header[col] else None,
                                    'conditions':[{'name':'temperature','value_text':temp,'unit':'K'}],
                                    'row_index':i,'column_index':col,'caption':caption,'page':p.get('page'),
                                    'evidence_ids':p.get('evidence_ids',[]),'review_status':'not reviewed'})
    structure['equations']=eqs
    structure['plots']=plots
    structure['fitted_results']=fit_records
    structure['source_objects_version']='1.0'
    return structure
