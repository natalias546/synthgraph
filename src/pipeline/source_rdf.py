"""Optional provisional RDF from source-level plot, equation and fit records.

This file is NOT the curated/accepted scientific knowledge graph.
"""
from __future__ import annotations
from pathlib import Path
from rdflib import Graph, Namespace, Literal, URIRef, RDF
from urllib.parse import quote

E=Namespace('https://w3id.org/matsci/ontology/evidence/')
L=Namespace('https://w3id.org/matsci/ontology/literature/')
KG=Namespace('https://w3id.org/matsci/instances/source/')

def write_source_rdf(structure:dict,path:Path,paper_id:str)->int:
    g=Graph();g.bind('ev',E);g.bind('lit',L);g.bind('src',KG)
    def iri(prefix,identifier):return URIRef(str(KG)+quote(paper_id,safe='')+'/'+prefix+'/'+quote(str(identifier),safe=''))
    def literal(subject,name,value):
        if value is not None and value!='':g.add((subject,E[name],Literal(str(value))))
    for p in structure.get('plots',[]):
        node=iri('plot',p['id']);g.add((node,RDF.type,E.ScientificPlotRecord))
        for field,prop in [('caption','plotCaption'),('title','plotTitle'),('plot_type','plotType'),
                           ('visual_role','plotVisualRole'),('crop','plotCropPath'),('verification_status','verificationStatus')]:
            literal(node,prop,p.get(field))
        literal(node,'sourceParagraphId',p.get('source_paragraph_id'))
        for axis_key,role in [('x_axis','x'),('y_axis','y')]:
            axis=p.get(axis_key) or {}
            an=iri('axis',p['id']+'_'+role);g.add((an,RDF.type,E.PlotAxisDescription));g.add((node,E.hasPlotAxis,an))
            literal(an,'axisRole',role)
            for k,prop in [('label','axisLabel'),('unit','axisUnit'),('source','axisInferredFrom')]:literal(an,prop,axis.get(k))
        for i,panel in enumerate(p.get('panels',[])):
            pn=iri('panel',p['id']+'_'+str(i));g.add((pn,RDF.type,E.PlotPanelDescription));g.add((node,E.hasPlotPanel,pn))
            if panel.get('panel_label')=='inset':
                g.add((pn,RDF.type,E.FigureInsetPanel))
                g.add((node,E.hasInsetPanel,pn))
            literal(pn,'plotTitle',panel.get('panel_label'))
    for e in structure.get('equations',[]):
        node=iri('equation',e['id']);g.add((node,RDF.type,E.ScientificEquationRecord))
        for k,p in [('expression_text','expressionText'),('expression_status','expressionStatus'),
                    ('source_paragraph_id','sourceParagraphId')]:literal(node,p,e.get(k))
        for d in e.get('definition_context',[]):
            literal(node,'definitionContextText',d.get('text'))
    for f in structure.get('fitted_results',[]):
        node=iri('fit',f['id']);g.add((node,RDF.type,E.ModelFittingResultRecord));g.add((node,RDF.type,L.ScientificMeasurementRecord))
        for k,p in [('source_paragraph_id','sourceParagraphId'),('value_text','rawValueText'),
                    ('uncertainty_notation','uncertaintyNotation'),('parameter_symbol','fitParameterSymbol'),
                    ('fitted_model_name','fitModelName'),('parameter_definition','fitParameterDefinition'),('unit','quantityUnit')]:
            literal(node,p,f.get(k))
        model_instance=iri('model','HikamiLarkinNagaokaModel')
        g.add((model_instance,RDF.type,E.HikamiLarkinNagaokaModel))
        g.add((node,E.usesFittingModel,model_instance))
        par=iri('fitparameter',f['id']);g.add((par,RDF.type,E.FittedParameterDescription))
        if 'α' in (f.get('parameter_symbol') or ''):
            g.add((par,RDF.type,E.HLNPrefactorDescription))
        elif 'φ' in (f.get('parameter_symbol') or '') or 'ϕ' in (f.get('parameter_symbol') or ''):
            g.add((par,RDF.type,E.PhaseCoherenceLengthDescription))
        g.add((node,E.hasFittedParameter,par))
        literal(par,'fitParameterSymbol',f.get('parameter_symbol'))
        for evidence_id in f.get('evidence_ids',[]):
            evid=iri('evidence',evidence_id)
            g.add((evid,RDF.type,L.LiteratureEvidenceSpan))
            g.add((node,E.hasFitSourceEvidence,evid))
            literal(evid,'sourceEvidenceId',evidence_id)
        for cond in f.get('conditions',[]):literal(node,'definitionContextText',str(cond))
    # Pre-LLM attribution graph is PROVISIONAL. A citation records attribution,
    # not independent verification of the referenced paper's scientific claims.
    paper=iri('publication',paper_id)
    g.add((paper,RDF.type,E.ScientificPublicationRecord))
    literal(paper,'sourcePaperId',paper_id)
    # Resolve explicit figure/table source links only when the printed number
    # identifies exactly one parsed object in this paper.
    doc_by_label={}
    for fig in structure.get('figures',[]):
        name=(fig.get('label') or '').lower()
        m=__import__('re').search(r'figure\s+(\d+)',name)
        if m:doc_by_label.setdefault(('figure',m.group(1)),[]).append(iri('plot','PLOT_'+fig['paragraph_id']))
    for tbl in structure.get('tables',[]):
        name=(tbl.get('label') or '').lower()
        m=__import__('re').search(r'table\s+(\d+)',name)
        if m:
            node=iri('table',tbl['paragraph_id'])
            g.add((node,RDF.type,E.ScientificTableSourceObject))
            literal(node,'sourceParagraphId',tbl['paragraph_id'])
            literal(node,'sourceReferenceText',tbl.get('caption'))
            doc_by_label.setdefault(('table',m.group(1)),[]).append(node)
    biblio={}
    for item in structure.get('bibliographic_entries',[]):
        b=iri('bibliography',item['id']);biblio[item['id']]=b
        g.add((b,RDF.type,E.BibliographicSourceEntry))
        literal(b,'sourceParagraphId',item.get('source_paragraph_id'))
        literal(b,'sourceReferenceText',item.get('literal_reference'))
        literal(b,'referenceOrdinal',item.get('ordinal'))
    reference_map={r['id']:r for r in structure.get('document_references',[])}
    for item in structure.get('source_assertions',[]):
        a=iri('assertion',item['id'])
        g.add((a,RDF.type,URIRef('https://w3id.org/matsci/ontology/SourceAssertion')))
        g.add((a,RDF.type,E.ClaimAttributionRecord))
        g.add((a,E.assertedInPublication,paper))
        origin_nodes={'current_study':E.OriginCurrentStudy,
                      'cited_study':E.OriginCitedStudy,
                      'background_knowledge':E.OriginBackgroundKnowledge,
                      'undetermined':E.OriginUndetermined,
                      'mixed_current_and_cited':E.OriginMixed}
        g.add((a,E.hasClaimOrigin,origin_nodes.get(item.get('claim_origin'),E.OriginUndetermined)))
        for field,prop in [('clause','sourceAssertionText'),('claim_origin','claimOriginLabel'),
                           ('claim_kind','claimKindLabel'),('epistemic_status','epistemicStatus'),
                           ('assertion_status','attributionStatus'),('origin_reason','originReason'),
                           ('review_status','reviewStatus')]:
            literal(a,prop,item.get(field))
        for sid in item.get('evidence_ids',[]):
            en=iri('evidence',sid)
            g.add((en,RDF.type,L.LiteratureEvidenceSpan))
            g.add((a,L.hasEvidence,en))
            literal(en,'sourceEvidenceId',sid)
        for rid in item.get('reference_ids',[]):
            ref=reference_map.get(rid)
            if ref is None:continue
            rn=iri('reference_mention',rid)
            g.add((rn,RDF.type,E.SourceCitationMention))
            g.add((a,E.hasCitationMention,rn))
            literal(rn,'sourceReferenceText',ref.get('surface'))
            literal(rn,'referenceResolutionStatus',ref.get('reference_resolution_status') or 'unverified_source_pointer')
            if ref.get('kind') in ('figure','table'):
                targets=doc_by_label.get((ref['kind'],str(ref.get('target_label'))),[])
                if len(targets)==1:
                    g.add((a,E.refersToDocumentObject,targets[0]))
            elif ref.get('kind')=='supplement':
                supplement=iri('supplement',ref.get('target_label') or rid)
                g.add((supplement,RDF.type,E.SupplementarySourceObject))
                g.add((a,E.refersToDocumentObject,supplement))
                literal(supplement,'verificationStatus','unverified_supplement_reference')
            if ref.get('bibliography_entry_id') in biblio:
                g.add((a,E.citesBibliographicEntry,biblio[ref['bibliography_entry_id']]))
                if item.get('claim_origin')=='cited_study':
                    cited_work=iri('cited_publication',ref['bibliography_entry_id'])
                    g.add((cited_work,RDF.type,E.ScientificPublicationRecord))
                    g.add((a,E.attributedToPublication,cited_work))
                    literal(cited_work,'verificationStatus','source_citation_only_unverified')
                    literal(cited_work,'referenceEntryId',ref['bibliography_entry_id'])
    for row in structure.get('relation_candidates',[]):
        rn=iri('relation_candidate',row['id']);g.add((rn,RDF.type,URIRef('https://w3id.org/matsci/ontology/extraction/ScientificComparisonRecord')))
        literal(rn,'candidateRelation',row.get('relation_type'))
        literal(rn,'relationVerificationStatus',row.get('status'))
        literal(rn,'sourceAssertionId',row.get('assertion_id'))
    for row in structure.get('specimen_mentions',[]):
        n=iri('sample_mention',row['id'])
        g.add((n,RDF.type,E.ScientificConceptMention))
        literal(n,'sampleSurfaceForm',row.get('surface'))
        literal(n,'candidateMaterialFormula',row.get('candidate_material_formula'))
        literal(n,'specimenLinkStatus',row.get('link_status'))
        literal(n,'sourceEvidenceId',row.get('evidence_id'))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);g.serialize(destination=str(path),format='turtle')
    return len(g)
