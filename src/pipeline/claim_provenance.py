"""Offline clause, citation, specimen, and discourse provenance proposals.

Every assertion is a source-backed *candidate*, never a verified finding.
The source is immutable: all offsets index the original semantic-evidence sentence.
No referenced paper is treated as verified without accessing that paper.
"""
from __future__ import annotations
import hashlib
import re
from collections import Counter, defaultdict

_SOURCE_REF = re.compile(r"\b(?P<kind>Figs?\.?|Figures?|Tables?|Eqs?\.?|Equations?|Refs?\.?|References?|Supp(?:lementary|lemental)?(?:\s+(?:Fig(?:ure)?|Table|Data))?)\s*(?P<identifier>S?\d+(?:[a-z])?)\b",re.I)
_SUPPLEMENT = re.compile(r'\b(?:supplementary|supporting)\s+(?:information|materials?|data|figures?|tables?|files?)\b',re.I)
_NUMBERED_CITE = re.compile(r"(?<!\w)\[(?P<numbers>\d+(?:\s*[-,]\s*\d+)*)\]")
_AUTHOR_CITE = re.compile(r"\b(?:[A-Z][a-z-]+\s+et\s+al\.|(?:previous(?:ly)?|prior|earlier)\s+(?:work|stud(?:y|ies)|reports?|measurements?|literature))",re.I)
_CURRENT_CUE = re.compile(r"\b(?:we\s+(?:report|observe|show|find|measure|calculate|determine|demonstrate|synthesize|obtained|found)|our\s+(?:results?|measurements?|samples?|crystals?|study|data|measured\s+\w+|calculated\s+\w+)|(?:was|were)\s+(?:measured|performed|conducted|obtained|calculated)\s+(?:in\s+this\s+(?:work|study))?)\b",re.I)
_CITED_CUE = re.compile(r"\b(?:previous(?:ly)?|prior|earlier|literature|reported\s+by|according\s+to|as\s+(?:reported|predicted|shown)\s+by|\bet\s+al\.)\b",re.I)
_COMPARISON = re.compile(r"\b(?:consistent\s+with|in\s+(?:agreement|contrast)\s+(?:with|to)|agrees?\s+with|differs?\s+from|compared\s+(?:to|with)|higher\s+than|lower\s+than|whereas|in\s+contrast\s+to|unlike|similar\s+to)\b",re.I)
_INFERENCE = re.compile(r"\b(?:indicat(?:es?|ing)|suggests?|implies?|evidence\s+(?:for|of)|demonstrates?|attributed\s+to|due\s+to|arises?\s+from)\b",re.I)
_CAUSAL = re.compile(r"\b(?:attributed\s+to|due\s+to|caused\s+by|arises?\s+from|results?\s+from)\b",re.I)
_EXPECTED = re.compile(r"\b(?:expected|predicted|may|might|could|potentially|possibly|likely|hypothesized)\b",re.I)
_NEGATION = re.compile(r"\b(?:no\s+evidence|not\s+(?:observed|detected|found)|absence\s+of|without\s+detectable|lacks?)\b",re.I)
_APPROX = re.compile(r"(?<!\w)(?:approximately|roughly|nearly|about|around|~|≈|ca\.)\s*(?=[-+]?\d|\b\w)",re.I)
_MEASURE = re.compile(r"\b(?:measured|observed|measurements?|obtained|experimental(?:ly)?|detected|determined|found)\b",re.I)
_COMPUTED = re.compile(r"\b(?:calculated|computed|simulation|simulated|DFT|density\s+functional|theoretically\s+estimated)\b",re.I)
_FIGURE_CONTEXT = re.compile(r"\b(?:Fig\.?|Figure|plot|panel|inset|micrograph|image|contrast)\b",re.I)
_SPECIMEN = re.compile(r"\b(?:(?:single|polycrystalline|as[ -]?grown|as[ -]?synthesized|thin[ -]?film)\s+)?(?:samples?|crystals?|specimens?|flakes?|wafers?)\b",re.I)
_FORMULA = re.compile(r"\b(?:[A-Z][a-z]?\s*\d*){2,8}\b")
_GENERIC = {"consistent","samples","sample","presence","supplementary","figs","indicating","data","measurements","crystals","inset","contrast","reported","formation","results","see","approximately","attributed","note","order","although","due","determined","ref","whereas","conducted","presented","calculated","expected","well","notably","have","addition","absence","given","single crystals"}
_DOC_WORD = {"supplementary","figs","inset","ref","see","note","presented","given"}
_COMPARE_WORD = {"consistent","contrast","whereas","although","indicating"}
_INTERPRET_WORD = {"presence","absence","attributed","due","expected","determined","calculated"}
_SPEC_WORD = {"sample","samples","crystals","single crystals"}
_MEAS_WORD = {"data","measurements","results","conducted","reported","temperature dependence"}
_EPIS = {"approximately","notably"}


def classify_surface(phrase: str, sentence: str = "") -> dict:
    """A phrase-level routing decision, never an OWL class assignment."""
    key = " ".join(re.findall(r"[\w]+", phrase.casefold()))
    context=sentence.casefold()
    if key=="well" and "quantum well" in context:return {"route":"scientific_entity", "reason":"quantum well is a scientific phrase"}
    if key=="order" and re.search(r"\b(?:magnetic|crystal|structural|ferro\w*)\s+order\b",context):return {"route":"scientific_entity","reason":"scientific compound noun"}
    if key=="contrast" and _FIGURE_CONTEXT.search(sentence) and re.search(r"\b(?:image|micrograph|SEM|TEM|optical)\b",sentence,re.I):return {"route":"scientific_entity","reason":"possible image contrast"}
    if key=="formation" and re.search(r"\bformation\s+(?:energy|enthalpy)\b",context):return {"route":"scientific_entity","reason":"formation energy/enthalpy concept"}
    if key in {"fermi level","c axis","c-axis","tbti3bi4"}:return {"route":"scientific_entity","reason":"specific scientific concept"}
    if key in _DOC_WORD:return {"route":"document_reference","reason":"document navigation, not a material property"}
    if key in _SPEC_WORD:return {"route":"sample_reference","reason":"resolve specimen context before class matching"}
    if key in _COMPARE_WORD:return {"route":"comparison_operator","reason":"needs two claim operands"}
    if key in _INTERPRET_WORD:return {"route":"inference_operator","reason":"claim status or attribution not an entity"}
    if key in _MEAS_WORD:return {"route":"measurement_reference","reason":"needs measurement/assay identity"}
    if key in _EPIS:return {"route":"epistemic_modifier","reason":"claim or numerical qualifier"}
    if key in _GENERIC:return {"route":"generic_discourse","reason":"not an independent class candidate"}
    return {"route":"scientific_entity","reason":"eligible for ontology candidate matching"}


def _stable_id(prefix:str,*values:str)->str:
    return prefix+hashlib.sha256("|".join(str(s) for s in values).encode()).hexdigest()[:16]


def _references(sentence:str, eid:str):
    refs=[]
    for m in _SOURCE_REF.finditer(sentence):
        kind=m.group('kind').lower()
        route=('figure' if kind.startswith(('fig','figure')) else 'table' if kind.startswith('table')
               else 'equation' if kind.startswith(('eq','equation')) else 'bibliography' if kind.startswith(('ref','reference')) else 'supplement')
        refs.append({'id':_stable_id('DR_',eid,str(m.start()),m.group()),'evidence_id':eid,'kind':route,
                     'surface':m.group(),'target_label':m.group('identifier'),'span_start':m.start(),
                     'span_end':m.end(),'verification_status':'source_mention_only'})
    for m in _NUMBERED_CITE.finditer(sentence):
        refs.append({'id':_stable_id('DR_',eid,str(m.start()),m.group()),'evidence_id':eid,'kind':'bibliography',
                     'surface':m.group(),'target_label':m.group('numbers'),'span_start':m.start(),
                     'span_end':m.end(),'verification_status':'source_mention_only'})
    for m in _SUPPLEMENT.finditer(sentence):
        if any(m.start()>=r['span_start'] and m.end()<=r['span_end'] for r in refs):continue
        refs.append({'id':_stable_id('DR_',eid,str(m.start()),m.group()),'evidence_id':eid,
                     'kind':'supplement','surface':m.group(),'target_label':'unspecified',
                     'span_start':m.start(),'span_end':m.end(),
                     'verification_status':'unresolved_supplement_reference'})
    return sorted(refs,key=lambda d:d['span_start'])


def _clauses(sentence:str)->list[tuple[int,int,str]]:
    """Segment only explicit contrast markers and semicolons, never author initials."""
    splits=[0]
    for m in re.finditer(r"\s*;\s*|(?<=,)\s+(?=(?:whereas|while|however|in contrast)\b)",sentence,re.I):
        if m.start()>splits[-1]+15 and len(sentence)-m.end()>15:splits.append(m.end())
    splits.append(len(sentence))
    result=[]
    for a,b in zip(splits,splits[1:]):
        raw=sentence[a:b];offset=len(raw)-len(raw.lstrip());txt=raw.strip()
        if txt:result.append((a+offset,a+offset+len(txt),txt))
    return result


def _classify_claim(text:str, section:str, refs:list)->tuple[str,str,str,str]:
    """Origin, claim_kind, epistemic_status, reason. Conservative: never verify origin."""
    prior=bool(_CITED_CUE.search(text))
    local=bool(_CURRENT_CUE.search(text))
    if (prior or (refs and _COMPARISON.search(text))) and local: origin='mixed_current_and_cited'
    elif prior:origin='cited_study'
    elif local:origin='current_study'
    elif _NUMBERED_CITE.search(text) and (section in ('introduction','abstract')) and not _CURRENT_CUE.search(text):origin='cited_study'
    elif section in ('methods','results') and (_MEASURE.search(text) or _COMPUTED.search(text)) and not _NUMBERED_CITE.search(text):origin='current_study'
    else:origin='undetermined'
    if _COMPARISON.search(text):kind='comparison'
    elif _INFERENCE.search(text):kind='interpretation'
    elif _COMPUTED.search(text):kind='calculation'
    elif _MEASURE.search(text):kind='observation'
    else:kind='scientific_statement'
    epistemic=('hypothetical' if _EXPECTED.search(text) else 'negated' if _NEGATION.search(text)
               else 'reported')
    why=('explicit current and earlier-work cues' if origin=='mixed_current_and_cited' else
         'prior-publication attribution cue' if origin=='cited_study' else
         'current-study pronoun, process, or results-section cue' if origin=='current_study' else
         'insufficient evidence to assign original publication')
    return origin,kind,epistemic,why


def enrich_claim_provenance(structure:dict)->dict:
    """Append derived candidate assertions, reference targets, specimen links, and routing."""
    assertions=[]; refs=[]; sample_mentions=[]; routes=[]; relation_candidates=[]
    evidence=structure.get('semantic_evidence',[])
    by_par=defaultdict(list)
    for e in evidence:by_par[e['paragraph_id']].append(e)
    # Keep specimen linkage local to a paragraph. Cross-paragraph pronouns remain unresolved.
    for e in evidence:
        if e.get('source_kind') in ('bibliographic_entry','figure','table') or e.get('section_role') in ('references','acknowledgements','author_info','front_matter'):
            continue
        sentence=e.get('sentence','');eid=e['id']
        current_refs=_references(sentence,eid);refs.extend(current_refs)
        for m in _SPECIMEN.finditer(sentence):
            near=sentence[max(0,m.start()-100):min(len(sentence),m.end()+80)]
            formulas=[z.group().replace(' ','') for z in _FORMULA.finditer(near)
                      if re.search(r'[A-Z].*\d',z.group()) and len(z.group().replace(' ',''))>=4]
            sample_mentions.append({'id':_stable_id('SM_',eid,str(m.start()),m.group()),'evidence_id':eid,
                                    'surface':m.group(),'span_start':m.start(),'span_end':m.end(),
                                    'candidate_material_formula':formulas[0] if len(set(formulas))==1 else None,
                                    'link_status':'context_candidate' if len(set(formulas))==1 else 'unresolved',
                                    'study_origin':'undetermined'})
        for phrase in re.finditer(r"\b(?:consistent|samples?|presence|supplementary|figs|indicating|data|measurements|crystals|inset|single crystals|contrast|fermi level|reported|formation|results|see|approximately|temperature dependence|attributed|note|order|although|due|determined|refs?|whereas|conducted|presented|calculated|expected|well|notably|have|addition|absence|c[ -]axis|given)\b",sentence,re.I):
            r=classify_surface(phrase.group(),sentence)
            if r['route']!='scientific_entity':routes.append({'evidence_id':eid,'term':phrase.group(),
                     'span_start':phrase.start(),'span_end':phrase.end(),**r})
        if e.get('source_kind') in ('caption','footnote'):
            continue
        for a,b,clause in _clauses(sentence):
            if len(clause)<12:continue
            relevant=[x['id'] for x in current_refs if a<=x['span_start']<b]
            origin,kind,epistemic,reason=_classify_claim(clause,e.get('section_role',''),relevant)
            # Only declare candidate assertions with plausible scientific content.
            if not (e.get('scientific_roles') or relevant or _COMPARISON.search(clause)
                    or _COMPUTED.search(clause) or _SPECIMEN.search(clause)):
                continue
            aid=_stable_id('AS_',eid,str(a),clause)
            row={'id':aid,'evidence_ids':[eid],'paragraph_id':e['paragraph_id'],
                 'source_section_role':e.get('section_role'), 'clause':clause,
                 'span_start':e['span_start']+a,'span_end':e['span_start']+b,
                 'claim_origin':origin,'claim_kind':kind,'epistemic_status':epistemic,
                 'assertion_status':'candidate_unverified','origin_reason':reason,
                 'reference_ids':relevant,'review_status':'not reviewed', 'numeric_approximation':bool(_APPROX.search(clause)),
                 'causal_attribution':bool(_CAUSAL.search(clause))}
            assertions.append(row)
            if kind=='comparison':
                op=_COMPARISON.search(clause)
                relation_candidates.append({'id':_stable_id('REL_',aid,op.group() if op else ''),'assertion_id':aid,
                    'relation_type':('contrasts_with' if re.search('contrast|whereas|unlike|differs|lower|higher',op.group(),re.I) else 'agrees_with') if op else 'compares_with',
                    'cue':op.group() if op else 'comparison','left_assertion_id':None,
                    'right_assertion_id':None,'status':'requires_operand_review',
                    'evidence_ids':[eid]})
            if _CAUSAL.search(clause):
                relation_candidates.append({'id':_stable_id('CAUSE_',aid),'assertion_id':aid,
                    'relation_type':'causal_attribution','cue':_CAUSAL.search(clause).group(),
                    'left_assertion_id':None,'right_assertion_id':None,
                    'status':'qualified_claim_only','evidence_ids':[eid]})
    # Collapse reference targets by (document object type, printed label), retaining every mention.
    bibliography=[{'id':_stable_id('BIB_',p['id']), 'ordinal':i+1,
                 'source_paragraph_id':p['id'], 'literal_reference':p.get('text',''),
                 'resolution_status':'source_bibliography_entry'}
                 for i,p in enumerate(x for x in structure.get('paragraphs',[])
                     if x.get('role')=='references' and x.get('kind') in ('list_item','reference'))]
    bibliography_by_number={str(b['ordinal']):b for b in bibliography}
    for r in refs:
        if r['kind']=='bibliography' and r['target_label'].isdigit():
            hit=bibliography_by_number.get(r['target_label'])
            if hit:
                r['bibliography_entry_id']=hit['id']
                # When reference numbers are inferred from unnumbered bibliography
                # order, the link MUST NOT be treated as verified.
                r['reference_resolution_status']='proposed_ordinal_link_needs_verification'
    refs_by_target=defaultdict(list)
    for r in refs:refs_by_target[(r['kind'],r['target_label'].lower())].append(r['id'])
    targets=[{'id':_stable_id('DT_',*key),'kind':key[0],'target_label':key[1],
              'mention_ids':ids,'resolution_status':'unverified_source_pointer'}
             for key,ids in sorted(refs_by_target.items())]
    structure['bibliographic_entries']=bibliography
    structure['source_assertions']=assertions
    structure['document_references']=refs
    structure['document_reference_targets']=targets
    structure['relation_candidates']=relation_candidates
    structure['specimen_mentions']=sample_mentions
    structure['discourse_routes']=routes
    structure['claim_provenance_version']='1.0'
    structure['claim_provenance_summary']={
        'assertions_by_origin':dict(Counter(a['claim_origin'] for a in assertions)),
        'assertions_by_kind':dict(Counter(a['claim_kind'] for a in assertions)),
        'reference_targets':len(targets), 'specimen_mentions':len(sample_mentions),
        'discourse_mention_routes':dict(Counter(x['route'] for x in routes))}
    return structure


def link_specimen_mentions(structure:dict, pass1:dict)->list[dict]:
    """Candidate-only specimen links to Pass1 sample IDs, never a forced match.

    A scientific formula cannot distinguish multiple differently treated samples.
    Link only when the material formula resolves to precisely one sample in this
    paper; all other references remain unresolved for human/LLM attention.
    """
    samples=pass1.get('samples',[])
    materials={m.get('id'):m for m in pass1.get('materials',[])}
    def canon(s):return re.sub(r'[^A-Za-z0-9]','',(s or '')).casefold()
    formulas=defaultdict(set)
    for s in samples:
        for matid in s.get('material_ids',[]):
            mat=materials.get(matid,{})
            if mat.get('formula'):formulas[canon(mat['formula'])].add(s['id'])
        # Only explicit formula-like sample names, never general specimen words.
        name=s.get('name_in_paper','')
        for m in _FORMULA.finditer(name):
            if re.search(r'\d',m.group()) and len(canon(m.group()))>=4:
                formulas[canon(m.group())].add(s['id'])
    links=[]
    for mention in structure.get('specimen_mentions',[]):
        form=mention.get('candidate_material_formula')
        choices=sorted(formulas.get(canon(form),set())) if form else []
        row={**mention,'candidate_sample_ids':choices,
             'matched_sample_id':choices[0] if len(choices)==1 else None,
             'match_status':'unique_formula_candidate_unverified' if len(choices)==1 else
                'ambiguous_multiple_samples' if len(choices)>1 else 'unresolved'}
        links.append(row)
    return links
