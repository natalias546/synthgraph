"""Build an additive source-context OWL extension from the v11 curated ontology.

rdflib constructs the small extension. The original RDF/XML body is copied
byte-for-byte, with new RDF/XML statements inserted before the closing tag.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from rdflib import Graph, Namespace, Literal, RDF, RDFS, OWL, URIRef

EV=Namespace('https://w3id.org/matsci/ontology/evidence/')
LIT=Namespace('https://w3id.org/matsci/ontology/literature/')
CORE=Namespace('https://w3id.org/matsci/ontology/')
EX=Namespace('https://w3id.org/matsci/ontology/extraction/')

# Record descriptions are information entities; physical qualities/state/phenomena
# are subclassed separately to avoid an OWL category error.
CLASSES={
 'ScientificSourceObject':(str(LIT.ScientificExtractionRecord),'Original equation/plot/table bibliographic source object.'),
 'BibliographicSourceEntry':(str(EV.ScientificSourceObject),'One reference entry, independent of internal periods.'),
 'ScientificEquationRecord':(str(EV.ScientificSourceObject),'Source mathematical expression with nearby explanatory context.'),
 'ScientificPlotRecord':(str(EV.ScientificSourceObject),'Source figure/plot description, not digitized pixels.'),
 'PlotAxisDescription':(str(EV.ScientificSourceObject),'Candidate axis role, label and unit.'),
 'PlotPanelDescription':(str(EV.ScientificSourceObject),'Panel or inset of a source plot.'),
 'FigureVisualClassification':(str(EV.ScientificSourceObject),'Source-aware figure versus logo classification.'),
 'ModelFittingResultRecord':(str(LIT.ScientificMeasurementRecord),'Fitted parameter value with scientific model and conditions.'),
 'FittedParameterDescription':(str(LIT.ScientificParameterDescription),'Named fitted-model parameter, not an automatically measured value.'),
 'ScientificAcronymDefinition':(str(LIT.ScientificTerminologyEntry),'Document-grounded acronym expansion.'),
 'CanonicalScientificConcept':(str(LIT.ScientificTerminologyEntry),'One paper-local canonical scientific concept.'),
 'ScientificConceptMention':(str(LIT.ScientificTerminologyEntry),'One source-local lexical mention of a canonical concept.'),
 'ScientificEvidenceBlock':(str(LIT.ScientificExtractionRecord),'Collection of source evidence for review and model input.'),
 'ScientificReviewDecision':(str(LIT.ScientificExtractionRecord),'Explicit user action distinct from scientific truth.'),
 'ElectricalTransportDerivedQuantity':(str(CORE.TransportQuality),'Derived transport response or ratio; computation must be reported.'),
 'Magnetoresistance':(str(EV.ElectricalTransportDerivedQuantity),'Field-dependent normalized resistance/resistivity response.'),
 'ResidualResistivityRatio':(str(EV.ElectricalTransportDerivedQuantity),'Resistivity ratio at stated high and low temperatures.'),
 'Magnetoconductivity':(str(CORE.TransportQuality),'Conductivity as a function of magnetic field.'),
 'WeakAntilocalization':(str(CORE.PhysicalPhenomenon),'Quantum interference transport phenomenon (not a mention record).'),
 'TopologicalSurfaceState':(str(CORE.ElectronicState),'Electronic state localized at a topological boundary.'),
 'HikamiLarkinNagaokaModel':(str(CORE.ScientificModel),'Model of weak (anti)localization corrections to conductivity.'),
 'HLNPrefactorDescription':(str(EV.FittedParameterDescription),'Dimensionless HLN prefactor alpha.'),
 'PhaseCoherenceLengthDescription':(str(EV.FittedParameterDescription),'HLN fitted phase-coherence length L_phi.'),
 'RietveldRefinementMethod':(str(CORE.ExperimentalMethod),'Full-profile crystallographic diffraction refinement method.'),
}
# Object properties: typed record links use domain/range without over-asserting physical causation.
OBJECT={
 'hasSourceEquation':('ScientificSourceObject','ScientificEquationRecord'),
 'hasPlot':('ScientificSourceObject','ScientificPlotRecord'),
 'hasPlotAxis':('ScientificPlotRecord','PlotAxisDescription'),
 'hasPlotPanel':('ScientificPlotRecord','PlotPanelDescription'),
 'hasVisualClassification':('ScientificPlotRecord','FigureVisualClassification'),
 'hasNearbyDefinition':('ScientificEquationRecord',None),
 'hasFittedParameter':('ModelFittingResultRecord','FittedParameterDescription'),
 'usesFittingModel':('ModelFittingResultRecord',None),
 'hasFitSourceEvidence':('ModelFittingResultRecord',None),
 'hasAcronymDefinition':('CanonicalScientificConcept','ScientificAcronymDefinition'),
 'hasConceptMention':('CanonicalScientificConcept','ScientificConceptMention'),
 'denotesScientificConcept':('ScientificConceptMention','CanonicalScientificConcept'),
 'containsEvidenceBlock':('ScientificSourceObject','ScientificEvidenceBlock'),
 'containsEvidenceSpan':('ScientificEvidenceBlock',None),
 'reviewsEvidenceBlock':('ScientificReviewDecision','ScientificEvidenceBlock'),
 'reviewsCanonicalConcept':('ScientificReviewDecision','CanonicalScientificConcept'),
 'referencesPlotAxis':('ModelFittingResultRecord','PlotAxisDescription'),
 'hasModelParameterDefinition':('ModelFittingResultRecord','FittedParameterDescription'),
}
DATATYPE=['sourceKind','sourceParagraphId','sourceEvidenceId','sourcePage','expressionText',
          'expressionStatus','definitionContextText','axisRole','axisLabel','axisUnit',
          'axisInferredFrom','plotType','plotTitle','plotCaption','plotVisualRole','plotCropPath',
          'fitModelName','fitParameterSymbol','fitParameterDefinition','rawValueText',
          'uncertaintyNotation','quantityUnit','reviewDecision','conceptCanonicalTerm',
          'acronymExpansion','verificationStatus']

def build(root:Path, base:Path):
    od=root/'ontology';od.mkdir(parents=True,exist_ok=True)
    extension=od/'matsci-evidence-context-extension.ttl'
    combined=od/'matsci-v11-evidence-context-enriched.owl'
    graph=Graph();graph.bind('ev',EV);graph.bind('matsci',CORE);graph.bind('lit',LIT)
    for name,(parent,definition) in CLASSES.items():
        iri=EV[name]
        graph.add((iri,RDF.type,OWL.Class));graph.add((iri,RDFS.label,Literal(name)))
        graph.add((iri,RDFS.comment,Literal(definition)))
        graph.add((iri,RDFS.subClassOf,URIRef(parent)))
    for name,(dom,rng) in OBJECT.items():
        iri=EV[name];graph.add((iri,RDF.type,OWL.ObjectProperty));graph.add((iri,RDFS.label,Literal(name)))
        graph.add((iri,RDFS.domain,EV[dom]))
        if rng:graph.add((iri,RDFS.range,EV[rng]))
    for name in DATATYPE:
        iri=EV[name];graph.add((iri,RDF.type,OWL.DatatypeProperty));graph.add((iri,RDFS.label,Literal(name)))
    for syn,aliases in {'Magnetoresistance':['MR','magneto-resistance'],
                        'ResidualResistivityRatio':['RRR','residual resistivity ratio'],
                        'WeakAntilocalization':['WAL','weak anti-localization'],
                        'TopologicalSurfaceState':['TSS','topological surface states'],
                        'HikamiLarkinNagaokaModel':['HLN','Hikami-Larkin-Nagaoka'],
                        'RietveldRefinementMethod':['Rietveld refinement']}.items():
        for alias in aliases:
            graph.add((EV[syn],URIRef('http://www.w3.org/2004/02/skos/core#altLabel'),Literal(alias)))
    graph.serialize(destination=str(extension),format='turtle')
    # serialize only new RDF/XML statements, then insert within the old root.
    xml=graph.serialize(format='xml').encode('utf-8')
    start=xml.find(b'>',xml.index(b'<rdf:RDF'))+1
    end=xml.rfind(b'</rdf:RDF>')
    src=base.read_bytes();last=src.rfind(b'</rdf:RDF>')
    if last < 0:raise ValueError('Source OWL must be RDF/XML with closing rdf:RDF')
    body=b'\n<!-- ADDITIVE SynthGraph source-evidence-context ontology extension -->\n'+xml[start:end]+b'\n'
    combined.write_bytes(src[:last]+body+src[last:])
    manifest={'source_owl':base.name,'source_bytes':len(src),'combined_owl':combined.name,
              'combined_bytes':combined.stat().st_size,'source_sha256':hashlib.sha256(src).hexdigest(),
              'extension_namespace':str(EV),'added_classes':len(CLASSES),
              'added_object_properties':len(OBJECT),'added_datatype_properties':len(DATATYPE),
              'extension_triples':len(graph),'terms':sorted(CLASSES)}
    (od/'matsci-evidence-context-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return manifest
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,default=Path(__file__).resolve().parents[1] / 'ontology' / 'matsci-v11-parsing-aligned-cleaned.owl')
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args()
    print(build(args.root,args.base))
