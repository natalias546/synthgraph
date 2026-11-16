"""Evidence segmentation with source-aware technical sentence boundaries.

Never edits source paragraph text. Offsets refer to the original paragraph string;
source-type records, quantities and contextual IDs are deterministic candidates.
"""
from __future__ import annotations
import re
from .scientific_context import split_source, quantities, evidence_blocks
from .source_objects import enrich_source_objects
from .claim_provenance import enrich_claim_provenance

_SYMBOL = re.compile(r"\b(?:RRR|MR|AHE|WAL|DFT|ARPES|XRD|CVT|CVD|HLN|TSS|SEM|EDAX|UV[- ]?VIS)\b")
_RESULTS = re.compile(r"\b(?:increase[ds]?|decrease[ds]?|observed|measured|found|shows?|exhibits?|obtained|reported)\b",re.I)
_METHODS = re.compile(r"\b(?:grown|synthesi[sz]ed|annealed|deposited|fabricated|prepared|calculated|performed|using|measured|refined|fitted)\b",re.I)
_HYPOTHESIS = re.compile(r"\b(?:may|might|suggests?|possibly|could|hypothes[ie]s|attribut(?:ed|able))\b",re.I)
_CITATION = re.compile(r"\b(?:previous(?:ly)?|prior work|literature|ref\.|et al\.)\b",re.I)

def enrich_structure(structure:dict)->dict:
    ev=[];paras=structure.get('paragraphs',[])
    for p in paras:
        spans=split_source(p.get('text',''),p.get('kind','text'),p.get('role',''))
        pid=p['id'];p['evidence_ids']=[f'{pid}:s{i}' for i in range(1,len(spans)+1)]
        for i,(start,end,sent) in enumerate(spans):
            roles=[]
            for pat,tag in ((_RESULTS,'result'),(_METHODS,'method'),(_HYPOTHESIS,'qualified_claim'),(_CITATION,'literature_comparison')):
                if pat.search(sent):roles.append(tag)
            qs=quantities(sent)
            if qs:roles.append('quantitative_evidence')
            if p.get('role') in ('methods','results','introduction','conclusion'):roles.append('section_'+p['role'])
            if p.get('role')=='references' or p.get('kind')=='reference': roles=['bibliographic_entry']
            evidence={'id':p['evidence_ids'][i],'paragraph_id':pid,
                      'section_id':p.get('section_id'),'section_role':p.get('role'),
                      'source_kind': 'bibliographic_entry' if roles==['bibliographic_entry'] else p.get('kind'),
                      'page':p.get('page'),'span_start':start,'span_end':end,'sentence':sent,
                      'preceding_sentence':spans[i-1][2] if i else None,
                      'following_sentence':spans[i+1][2] if i+1<len(spans) else None,
                      'scientific_roles':roles,
                      'candidate_terms':list(dict.fromkeys(m.group() for m in _SYMBOL.finditer(sent))),
                      'value_mentions':[v['text'] for v in qs],'quantity_candidates':qs,
                      'review_status':'not reviewed','verified_semantic_truth':False}
            ev.append(evidence)
    by_p={p['id']:[] for p in paras}
    for e in ev:by_p[e['paragraph_id']].append(e)
    for idx,p in enumerate(paras):
        current=by_p[p['id']]
        if not current:continue
        for delta,k,text_key in ((-1,'preceding_source_id','preceding_sentence'),(1,'following_source_id','following_sentence')):
            ii=idx+delta
            if not (0<=ii<len(paras)):continue
            q=paras[ii]
            if q.get('section_id')!=p.get('section_id') or not by_p[q['id']]:continue
            if q.get('kind') in ('figure','table','reference','list_item') or p.get('kind') in ('figure','table','reference','list_item'):
                continue
            target=current[0] if delta<0 else current[-1]
            neighbor=by_p[q['id']][-1] if delta<0 else by_p[q['id']][0]
            target[k]=neighbor['id'];target[text_key]=target[text_key] or neighbor['sentence']
    structure['semantic_evidence']=ev
    structure['semantic_evidence_version']='2.0'
    structure['evidence_blocks']=evidence_blocks(ev)
    return enrich_claim_provenance(enrich_source_objects(structure))

# Kept for downstream Anchorer and legacy notebooks; scientific_context owns the
# source-aware segmentation implementation.
def split_sentences(text: str) -> list[tuple[int,int,str]]:
    return split_source(text)
