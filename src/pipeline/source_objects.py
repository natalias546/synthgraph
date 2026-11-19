"""Build explicit source-level Equation, Plot and FittingResult descriptions.

No image digitization, graph point inference, or formula-enrichment model calls.
Axes derived from captions are tagged `caption_inference_unverified`.
"""
from __future__ import annotations
import re
from .scientific_context import quantities

_AXIS_SPLIT = re.compile(r"(?P<y>[A-Za-z0-9αβγΔρσ𝜎𝜙φϕθΘµμΩÅ°()/_\s%+.-]{1,65}?)\s+(?:versus|vs\.?|against|as a function of)\s+(?P<x>[^,.]{2,90})", re.I)
_FUNC = re.compile(r"(?P<y>[A-Za-zαβρΔσ𝜎𝜙φ]+)\s*\(\s*(?P<x>[TH])\s*\)",re.I)
_EQUAL = re.compile(r"(?:(?:\b[A-ZαβγΔσρ𝜎]\s*\([^)]{1,30}\)|[αβγΔσρ𝜎LφϕBφ][\wφϕ]*)\s*=\s*[^,;\.]{3,180})")
_FORMULA_INTRO = re.compile(r"(?:equation|formula|expression).{0,140}?(?:given\s+by|written\s+as|defined\s+as|as\s+follows)\s*:?\s*$", re.I)
_DEF = re.compile(r"\b(?:where|here|denotes?|represents?|is defined as|refers to|parameter|prefactor|fitting)\b", re.I)
_FIT = re.compile(r"\b(?:Hikami[ -]Larkin[ -]Nagaoka|HLN)\b",re.I)


# Docling picture classification is a weak hint. Preserve the raw picture,
# while distinguishing an image-only fragment from a supported figure caption.
_SCIENCE_CAPTION = re.compile(
    r"\b(?:fig(?:ure)?s?\.?\s*\d+|plots?|curves?|spectr(?:um|a)|diffraction|"
    r"microscop(?:y|ic)|SEM|TEM|EDX|EDS|EDAX|XPS|XRD|"
    r"magneto(?:resistance|conductivity)|schematic|micrograph|"
    r"dependence|histogram|scatter|intensity|photographs?)\b", re.I)
_CHART_LABELS = {'line_chart', 'scatter_chart', 'bar_chart', 'histogram',
                 'pie_chart', 'heatmap'}
_DECORATIVE_LABELS = {'logo', 'icon', 'signature', 'qr_code',
                      'bar_code', 'stamp', 'decorative', 'decoration',
                      'watermark'}


def _visual_assessment(fig: dict) -> tuple[str, list[str]]:
    """Conservative source-backed classification; never silently discard crops."""
    classification = fig.get('classification') or {}
    name = str(classification.get('class') or '').lower().strip()
    caption = (fig.get('caption') or '').strip()
    scientific_caption = bool(_SCIENCE_CAPTION.search(caption))
    reasons = []
    if name in _DECORATIVE_LABELS:
        # Docling can mistakenly associate a nearby *real figure caption* with
        # a small publisher logo (e.g. Springer above an unrelated Fig. 6).
        # A caption is NOT independent evidence that the picture is scientific.
        # For explicit decorative labels, drop the picture from preflight even
        # when Docling attached a plausible Fig./SEM/plot caption. Preserve
        # the pixels and the suspect caption in the audit for manual recovery.
        role = 'decorative_icon' if name == 'icon' else 'decorative_logo'
        if scientific_caption:
            return role, [
                'Docling decorative picture was linked to a scientific-looking '
                'caption; association is unverified and picture is excluded',
                'Possible publisher logo with a misattached figure caption',
            ]
        if caption:
            return role, [
                'Decorative-classified picture has a caption, but the caption '
                'does not override the decorative classification',
            ]
        return role, ['Decorative picture classification; excluded before preflight']
    if scientific_caption:
        if name in _CHART_LABELS or re.search(r'\b(?:plot|curve|spectrum|spectra|diffraction|dependence)\b',caption,re.I):
            return 'scientific_plot_candidate', reasons
        return 'scientific_figure_candidate', reasons
    if caption:
        reasons.append('Caption lacks a recognizable scientific figure cue')
    else:
        reasons.append('Picture has no caption; may be a logo, inset or fragment')
    return 'unverified_picture', reasons


def _visual_role(fig: dict) -> str:
    return _visual_assessment(fig)[0]


# Prefer source text (caption or embedded PDF vector text); do not use units
# from unrelated legend values, tick labels, or speculative image recognition.
_UNIT_CONTENT = re.compile(
    r"^(?:%|a\.?u\.?|arb\.?\s*(?:units?)?|°|deg(?:rees?)?|"
    r"[pnumkMGTµμ]?\s*(?:K|T|Oe|Hz|eV|V|A|W|Pa|m|s|g|J|F|Ω|ohm|S|"
    r"cm|mm|nm|Å|rad|Torr|emu)(?:[\s*/·.\-^0-9µμΩÅ]+(?:cm|m|g|s|K|Ω|T|Oe|Hz|A|V|nm|mm|Å))?"
    r"|cm\s*[-^]?\s*[0-9]+|emu\s*/\s*g)$", re.I)
_AXIS_UNIT = re.compile(r"^(?P<label>.*?)\s*\(\s*(?P<unit>[^()]{1,28})\s*\)\s*$")
_AXIS_TEXT_PAIR = re.compile(
    r"(?P<label>[A-Za-zαβγΔρσφϕθΘΩµμ0-9_%/]{1,28}(?:\s+[A-Za-zαβγΔρσφϕθΘΩµμ0-9_%/]{1,20}){0,4})"
    r"\s*\(\s*(?P<unit>[^()\n]{1,28})\s*\)")


def _split_axis_unit(raw: str | None) -> tuple[str | None, str | None]:
    """Extract explicit axis suffix units: H (T), ρ (μΩ cm), I (a.u.)."""
    if not raw:
        return None, None
    text = raw.strip().rstrip('.,;:')
    m = _AXIS_UNIT.fullmatch(text)
    if not m or not m.group('label').strip():
        return text, None
    unit = m.group('unit').strip()
    if not _UNIT_CONTENT.fullmatch(unit):
        return text, None
    return m.group('label').strip(), unit


def _pdf_axis_unit_candidates(text: str | None) -> list[dict]:
    """Only explicit label/unit pairs recovered from PDF vector text."""
    out = []
    for line in (text or '').splitlines():
        for m in _AXIS_TEXT_PAIR.finditer(line):
            label, unit = _split_axis_unit(m.group())
            if unit and (label, unit) not in {(r['label'], r['unit']) for r in out}:
                out.append({'label': label, 'unit': unit,
                            'source': 'pdf_vector_text_inside_picture_unverified'})
    return out[:25]


def _apply_axis_unit(axis_label: str | None, pdf_text: str | None,
                     source: str) -> dict:
    label, unit = _split_axis_unit(axis_label)
    if unit:
        return {'label': label, 'unit': unit, 'source': source}
    # Vector-text matches must refer to the SAME explicit axis label. A mere
    # unit elsewhere on the figure might be a tick, legend, or other panel.
    if label and pdf_text:
        norm = lambda t: re.sub(r'\s+', '', t).casefold()
        matches = [r for r in _pdf_axis_unit_candidates(pdf_text)
                   if norm(r['label']) == norm(label)]
        if len({r['unit'] for r in matches}) == 1 and matches:
            return {'label': label, 'unit': matches[0]['unit'],
                    'source': 'pdf_vector_label_match_unverified'}
    return {'label': label, 'unit': None, 'source': source}


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
        role, flags=_visual_assessment(fig)
        fig['visual_role']=role
        fig['visual_review_reasons']=flags
        if pid in by_id:
            by_id[pid]['visual_role']=role
            by_id[pid]['visual_review_reasons']=flags
        caption=fig.get('caption') or ''
        x,y,origin=_caption_axis(caption)
        simple=re.search(r'\b([A-Za-zρσM]{1,3})\s*[-–]\s*([HT])\s+plot\b',caption,re.I)
        if simple and not x:
            y,x=simple.group(1),simple.group(2)
            origin='caption_inference_unverified'
        if x:
            x=re.sub(r'\s+(?:curve|plots?|graph|diagram)\b.*$', '', x, flags=re.I).strip()
        if role in {'decorative_logo','decorative_icon','unverified_picture'}:
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
        x_axis=_apply_axis_unit(x,fig.get('pdf_picture_text'),origin)
        y_axis=_apply_axis_unit(y,fig.get('pdf_picture_text'),origin)
        # If the caption has no axes, retain embedded-text pairs as candidates
        # without asserting x or y orientation (insets/legends are ambiguous).
        pdf_pairs=_pdf_axis_unit_candidates(fig.get('pdf_picture_text'))
        plots.append({'id':'PLOT_'+str(pid),'source_paragraph_id':pid,'page':fig.get('page'),
                      'caption':caption,'title':None,'plot_type':typ,
                      'visual_role':role,'visual_review_reasons':flags, 'crop':fig.get('crop'),
                      'axis_unit_candidates':pdf_pairs,
                      'classification':classification,
                      'x_axis':x_axis, 'y_axis':y_axis,
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
    structure['source_objects_version']='1.1'
    return structure
