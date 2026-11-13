"""Build additive literature extraction extension, and a dependency-safe cleaned OWL.
Does not edit schemas.py or the source ontology.

Usage:
python build_extension.py /path/to/matsci-v10-literature-materials-project-enriched.owl /output/directory
"""
from __future__ import annotations
from collections import defaultdict, Counter
from datetime import date
from pathlib import Path
import hashlib, json, re, sys
from xml.sax.saxutils import escape, quoteattr
from lxml import etree as ET
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

BASE = 'https://w3id.org/matsci/ontology/'
EX = Namespace(BASE + 'extraction/')
LIT = Namespace(BASE + 'literature/')
PMD = Namespace('https://w3id.org/pmd/co/')
BFO = Namespace('http://purl.obolibrary.org/obo/')
RDFXML_CLOSER = b'</rdf:RDF>'

# Use an information-content root because these are descriptions of application use,
# not the application process itself (e.g., OpticalApplication is already a planned process).
APP_CLASSES = {
 'MaterialsApplication':(None, 'materials application', 'An informational description of a reported, proposed, or intended use of a material; not proof that the application has been demonstrated.'),
 'ElectronicApplication':('MaterialsApplication','electronic application','Use of a material in electronic components, devices or circuits.'),
 'OptoelectronicApplication':('ElectronicApplication','optoelectronic application','Use of a material in coupled optical and electronic technologies.'),
 'PhotovoltaicApplication':('OptoelectronicApplication','photovoltaic application','Use in solar-energy conversion or photovoltaic devices.'),
 'PhotodetectorApplication':('OptoelectronicApplication','photodetector application','Use in light-detection devices or photoconductive detectors.'),
 'PhotonicApplication':('MaterialsApplication','photonic application','Use in photonic systems, optical waveguides or light manipulation.'),
 'ElectrochromicApplication':('OptoelectronicApplication','electrochromic application','Use in electrically controlled light absorption or coloration.'),
 'EnergyStorageApplication':('MaterialsApplication','energy storage application','Use of a material in energy storage.'),
 'BatteryApplication':('EnergyStorageApplication','battery application','Use in battery electrodes, electrolytes or related battery components.'),
 'SupercapacitorApplication':('EnergyStorageApplication','supercapacitor application','Use in electrochemical capacitors.'),
 'HydrogenStorageApplication':('EnergyStorageApplication','hydrogen storage application','Use in hydrogen uptake, storage and release.'),
 'CatalysisApplication':('MaterialsApplication','catalysis application','Use of a material as or within a catalyst.'),
 'PhotocatalysisApplication':('CatalysisApplication','photocatalysis application','Use in reactions assisted by photogenerated charge carriers.'),
 'ElectrocatalysisApplication':('CatalysisApplication','electrocatalysis application','Use in electrochemically driven catalytic reactions.'),
 'SensorApplication':('MaterialsApplication','sensor application','Use in measuring or detecting physical, chemical or biological signals.'),
 'GasSensorApplication':('SensorApplication','gas sensor application','Use in gas detection or gas sensing.'),
 'BiosensorApplication':('SensorApplication','biosensor application','Use in biochemical or biological sensing.'),
 'SpintronicsApplication':('ElectronicApplication','spintronics application','Use of electronic spin or spin-dependent transport in devices.'),
 'MagneticMemoryApplication':('SpintronicsApplication','magnetic memory application','Use in information storage based on magnetic states.'),
 'QuantumTechnologyApplication':('MaterialsApplication','quantum technology application','Use in quantum information, coherent devices or quantum sensing.'),
 'SuperconductingDeviceApplication':('QuantumTechnologyApplication','superconducting device application','Use in technology relying on superconductivity.'),
 'ThermoelectricApplication':('MaterialsApplication','thermoelectric application','Use in thermoelectric energy conversion or cooling.'),
 'ThermalManagementApplication':('MaterialsApplication','thermal management application','Use for heat transfer, thermal insulation, or thermal regulation.'),
 'ProtectiveCoatingApplication':('MaterialsApplication','protective coating application','Use as protective functional coating or barrier.'),
 'OpticalCoatingApplication':('PhotonicApplication','optical coating application','Use in functional optical coatings.'),
 'StructuralEngineeringApplication':('MaterialsApplication','structural engineering application','Use as a structural or load-bearing material.'),
 'BiomedicalMaterialApplication':('MaterialsApplication','biomedical material application','Use in biomedical devices, implants or biocompatible materials.'),
 'PiezoelectricApplication':('MaterialsApplication','piezoelectric application','Use in strain-sensitive electrical transduction and actuation.'),
 'MembraneSeparationApplication':('MaterialsApplication','membrane separation application','Use in filtration or selective separation membranes.'),
 'EnvironmentalRemediationApplication':('MaterialsApplication','environmental remediation application','Use in removing, immobilizing or transforming environmental contaminants.'),
 'RadiationShieldingApplication':('MaterialsApplication','radiation shielding application','Use for attenuation or protection against ionizing radiation.'),
}
# These classes organize extracted *mentions/records*, not the external physical entities themselves.
SCI_CLASSES = {
 'ScientificContextRecord':(str(LIT.ScientificExtractionRecord),'scientific context record','An evidence-linked informational record for concepts that do not fit the primary material/process/technique/property slots.'),
 'ScientificPhenomenonMention':('ScientificContextRecord','scientific phenomenon mention','Mention of an observed or predicted physical/chemical phenomenon.'),
 'ScientificMechanismMention':('ScientificContextRecord','scientific mechanism mention','Mention of a proposed or established explanatory mechanism.'),
 'ScientificTheoryMention':('ScientificContextRecord','scientific theory mention','Mention of an explanatory theoretical framework.'),
 'ScientificModelMention':('ScientificContextRecord','scientific model mention','Mention of a physical, statistical or computational model.'),
 'MaterialPhaseMention':('ScientificContextRecord','material phase mention','Mention of a material phase or phase label.'),
 'CrystalMotifMention':('ScientificContextRecord','crystal motif mention','Mention of a structural motif such as kagome, honeycomb or layered motifs.'),
 'ScientificSoftwareMention':('ScientificContextRecord','scientific software mention','Mention of scientific computation, simulation, or analysis software.'),
 'ScientificInstrumentMention':('ScientificContextRecord','scientific instrument mention','Mention of a named scientific instrument or apparatus.'),
 'ScientificDatasetMention':('ScientificContextRecord','scientific dataset mention','Mention of a dataset, repository, tabulation or benchmark.'),
 'LiteratureReferenceMention':('ScientificContextRecord','literature reference mention','Mention of a document, cited paper, or bibliographic record.'),
 'ScientificConditionMention':('ScientificContextRecord','scientific condition mention','Mention of an experimental or simulated condition and its symbol or value.'),
 'ScientificStructureMention':('ScientificContextRecord','scientific structure mention','Mention of a crystallographic, microstructural or electronic structure.'),
 'FigureOrTableMention':('ScientificContextRecord','figure or table mention','Reference to a figure, panel, equation or table in source literature.'),
 'ElectronicStateMention':('ScientificContextRecord','electronic state mention','Mention of an electronic, topological or magnetic electronic state.'),
 'ScientificApplicationClaim':(str(LIT.QualifiedScientificClaim),'scientific application claim','A source-supported statement asserting, suggesting, rejecting, or hypothetically proposing a material application.'),
 'ScientificComparisonRecord':(str(LIT.ScientificExtractionRecord),'scientific comparison record','Reified statement relating compared entities with scope, direction, metrics and evidence; comparison alone does not establish causation.'),
 'ScientificComparisonCondition':(str(LIT.ExperimentalConditionRecord),'scientific comparison condition','Condition under which a comparison is made; separate from comparable object identity.'),
 'ScientificValueRecord':(str(LIT.ScientificExtractionRecord),'scientific value record','Value, unit and qualifier as reported in literature; distinct from the physical quality measured.'),
 'UnmappedScientificTerm':(str(LIT.ScientificTerminologyEntry),'unmapped scientific term','Source phrase preserved while ontology alignment is unknown or needs human review.'),
 'ScientificAcronymCandidate':(str(LIT.ScientificTerminologyEntry),'scientific acronym candidate','Acronym with candidate expansion and evidence; not automatically an exact synonym.'),
 'KagomeMotifMention':('CrystalMotifMention','kagome structural motif mention','Evidence-linked mention of a kagome structural motif; not an assertion that a sample has that structure.'),
 'HallEffectMention':('ScientificPhenomenonMention','Hall effect mention','Evidence-linked mention of ordinary or anomalous Hall effect.'),
 'WeakAntilocalizationMention':('ScientificPhenomenonMention','weak antilocalization mention','Evidence-linked mention of weak antilocalization.'),
 'TopologicalPhaseMention':('MaterialPhaseMention','topological phase mention','Evidence-linked mention of a topological phase or state.'),
}

OBJPROPS={
 'compared_with':(None,None,'Direct comparison between two identified resources within a paper; neither symmetry nor causality is assumed. For metrics and conditions, also create a ScientificComparisonRecord.'),
 'hasComparisonSubject':('ScientificComparisonRecord',None,'Record to its primary compared entity, without forcing all objects into the same upper ontology.'),
 'hasComparisonReference':('ScientificComparisonRecord',None,'Record to its reference, baseline or comparator entity.'),
 'hasComparisonEvidence':('ScientificComparisonRecord',str(LIT.LiteratureEvidenceSpan),'Source passage supporting the comparison.'),
 'hasComparisonCondition':('ScientificComparisonRecord','ScientificComparisonCondition','Experimental or computational conditions for the comparison.'),
 'hasComparedValue':('ScientificComparisonRecord','ScientificValueRecord','Reported measured/computed or qualitative value used in a comparison.'),
 'hasApplicationCategory':('ScientificApplicationClaim','MaterialsApplication','Reported application category of an evidence-linked claim.'),
 'hasApplicationEvidence':('ScientificApplicationClaim',str(LIT.LiteratureEvidenceSpan),'Source span supporting a reported application claim.'),
 'hasApplicationClaim':(str(LIT.MaterialMention),'ScientificApplicationClaim','Associates the extracted material mention with an evidence-linked claim about its use.'),
 'refersToApplication':(str(LIT.ScientificExtractionRecord),'MaterialsApplication','An extraction record refers to a materials application concept.'),
 'referencesScientificContext':(str(LIT.ScientificExtractionRecord),'ScientificContextRecord','Record references contextual phenomenon, instrument, software, theory or structure.'),
 'hasReportedPropertyValue':(str(LIT.ScientificExtractionRecord),'ScientificValueRecord','Connects a claim or measurement record to a value record (not the physical has-quality relation).'),
 'hasProvisionalTerm':(str(LIT.ScientificExtractionRecord),'UnmappedScientificTerm','Associates an extracted record with terminology pending ontology review.'),
 'wasMeasuredOn':(str(LIT.ScientificMeasurementRecord),str(LIT.MaterialMention),'Relates a recorded measurement to the material mention described by the source.'),
 'intended_for':(None,'MaterialsApplication','Informationally reported intended application of a material or specimen; not identical to RO_0000085 has function.'),
 'influences':(None,None,'Source-reported association or influence; do not infer causality from this property. Qualify using evidence and trend records.'),
}

DATA_PROPS={
 'comparisonDirection':('ScientificComparisonRecord',XSD.string,'direction or trend reported in the source'),
 'comparisonBasis':('ScientificComparisonRecord',XSD.string,'property or variable being compared'),
 'comparisonScope':('ScientificComparisonRecord',XSD.string,'conditions, sample selection and other applicability restrictions'),
 'comparisonConclusion':('ScientificComparisonRecord',XSD.string,'source-supported comparison statement'),
 'applicationStatus':('ScientificApplicationClaim',XSD.string,'reported, proposed, demonstrated, negated or hypothetical'),
 'applicationEvidenceText':('ScientificApplicationClaim',XSD.string,'literal source quote supporting application claim'),
 'termResolutionStatus':('UnmappedScientificTerm',XSD.string,'unresolved, provisional, mapped, reviewed, rejected'),
 'candidateScientificCategory':('ScientificContextRecord',XSD.string,'intermediate coarse scientific category for hierarchical label selection'),
}


def build_graph():
 g=Graph(); g.bind('ex',EX);g.bind('lit',LIT);g.bind('owl',OWL);g.bind('rdfs',RDFS);g.bind('xsd',XSD)
 ontology=URIRef(str(EX).rstrip('/'))
 g.add((ontology,RDF.type,OWL.Ontology));g.add((ontology,RDFS.label,Literal('Materials-science schema alignment and application vocabulary',lang='en')))
 g.add((ontology,RDFS.comment,Literal('Additive vocabulary for scientific parsing and classification. Application classes describe reported intended uses, not physical functions. No change to schemas.py.',lang='en')))
 g.add((ontology,OWL.versionInfo,Literal('2026-11-03')))
 for key,(parent,label,comment) in APP_CLASSES.items():
  uri=EX[key];g.add((uri,RDF.type,OWL.Class));g.add((uri,RDFS.label,Literal(label,lang='en')));g.add((uri,RDFS.comment,Literal(comment,lang='en')))
  g.add((uri,RDFS.subClassOf,EX[parent] if parent else BFO.IAO_0000030))
 for key,(parent,label,comment) in SCI_CLASSES.items():
  uri=EX[key];g.add((uri,RDF.type,OWL.Class));g.add((uri,RDFS.label,Literal(label,lang='en')));g.add((uri,RDFS.comment,Literal(comment,lang='en')))
  g.add((uri,RDFS.subClassOf,EX[parent] if parent and not parent.startswith('http') else URIRef(parent)))
 for key,(domain,ran,comment) in OBJPROPS.items():
  uri=EX[key];g.add((uri,RDF.type,OWL.ObjectProperty));g.add((uri,RDFS.label,Literal(key,lang='en')));g.add((uri,RDFS.comment,Literal(comment,lang='en')))
  if domain:g.add((uri,RDFS.domain,EX[domain] if not domain.startswith('http') else URIRef(domain)))
  if ran:g.add((uri,RDFS.range,EX[ran] if not str(ran).startswith('http') else URIRef(ran)))
 for key,(domain,ran,comment) in DATA_PROPS.items():
  uri=EX[key];g.add((uri,RDF.type,OWL.DatatypeProperty));g.add((uri,RDFS.label,Literal(key,lang='en')));g.add((uri,RDFS.comment,Literal(comment,lang='en')))
  g.add((uri,RDFS.domain,EX[domain]));g.add((uri,RDFS.range,ran))
 # Cross-references to existing domain terms; no new subclass assertions for processes masquerading as uses.
 g.add((EX.PhotonicApplication,RDFS.seeAlso,URIRef(BASE+'OpticalApplication')))
 g.add((EX.BiomedicalMaterialApplication,RDFS.seeAlso,URIRef(BASE+'BiomedicalApplication')))
 g.add((EX.ScientificInstrumentMention,RDFS.seeAlso,URIRef(BASE+'MeasuringDevice')))
 g.add((EX.ScientificSoftwareMention,RDFS.seeAlso,URIRef(BASE+'Software')))
 g.add((EX.ScientificStructureMention,RDFS.seeAlso,URIRef(BASE+'CrystalStructure')))
 g.add((EX.ScientificPhenomenonMention,RDFS.seeAlso,URIRef(BASE+'PhysicalPhenomenon')))
 return g


def safe_deprecated(source):
 """Identify only rdf:Description subjects explicitly marked deprecated and with zero incoming rdf:resource mentions."""
 root='{http://www.w3.org/1999/02/22-rdf-syntax-ns#}';owl='{http://www.w3.org/2002/07/owl#}'
 deprecated={}
 for _,el in ET.iterparse(str(source),events=('end',),tag=root+'Description',huge_tree=True):
  a=el.get(root+'about')
  if a:
   flags=[(x.text or '').strip().lower() for x in el if x.tag==owl+'deprecated']
   if 'true' in flags or '1' in flags:
    deprecated[a]=[x.text for x in el if x.tag=='{http://www.w3.org/2000/01/rdf-schema#}label']
  el.clear()
  while el.getprevious() is not None:del el.getparent()[0]
 inbound=Counter(); examples=defaultdict(list)
 for _,el in ET.iterparse(str(source),events=('end',),tag=root+'Description',huge_tree=True):
  subject=el.get(root+'about') or el.get(root+'nodeID') or ''
  for child in el.iter():
   r=child.get(root+'resource')
   if r in deprecated and child is not el:
    inbound[r]+=1
    if len(examples[r])<20:examples[r].append({'source':subject,'predicate':child.tag.split('}')[-1]})
  el.clear()
  while el.getprevious() is not None:del el.getparent()[0]
 return deprecated,inbound,examples


def xml_fragment(g):
 # Existing RDF/XML defines the rdf/rdfs/owl prefixes, so no new XML namespaces are required.
 preds=[RDF.type,RDFS.label,RDFS.comment,RDFS.subClassOf,RDFS.subPropertyOf,RDFS.domain,RDFS.range,RDFS.seeAlso,OWL.versionInfo]
 nsmap={str(RDF):'rdf',str(RDFS):'rdfs',str(OWL):'owl'}
 bysub=defaultdict(list)
 for s,p,o in g: bysub[str(s)].append((p,o))
 lines=['  <!-- 2026-11-03: additive materials-science extraction and application schema -->\n']
 for s in sorted(bysub):
  lines.append('  <rdf:Description rdf:about='+quoteattr(s)+'>\n')
  for p,o in sorted(bysub[s],key=lambda x:(preds.index(x[0]) if x[0] in preds else 99,str(x[1]))):
   uri=str(p)
   pref,name=next(((v,uri[len(k):]) for k,v in nsmap.items() if uri.startswith(k)),(None,None))
   if not pref:raise ValueError('new unexpected predicate '+uri)
   tag=pref+':'+name
   if isinstance(o,URIRef):lines.append('    <'+tag+' rdf:resource='+quoteattr(str(o))+'/>\n')
   else:
    attr=(' xml:lang='+quoteattr(o.language) if o.language else ' rdf:datatype='+quoteattr(str(o.datatype)) if o.datatype else '')
    lines.append('    <'+tag+attr+'>'+escape(str(o))+'</'+tag+'>\n')
  lines.append('  </rdf:Description>\n')
 return ''.join(lines).encode('utf-8')


def make_output(source:Path,out:Path,g:Graph,deprec:set):
 # Editing line-by-line preserves the remaining 164 MB RDF/XML bytes exactly.
 # We only skip a subject's top-level rdf:Description if it is unreferenced.
 frag=xml_fragment(g)
 start_re=re.compile(br'^\s*<rdf:Description rdf:about="([^"]+)"')
 n_skip=0; count_subjects=Counter();skipping=False
 with source.open('rb') as inp,out.open('wb') as target:
  for line in inp:
   if not skipping:
    match=start_re.match(line)
    if match and match.group(1).decode('utf-8') in deprec:
     n_skip+=1;count_subjects[match.group(1).decode('utf-8')]+=1
     if b'</rdf:Description>' not in line:skipping=True
     continue
   else:
    if b'</rdf:Description>' in line:skipping=False
    continue
   if line.strip()==RDFXML_CLOSER:target.write(frag)
   target.write(line)
 if skipping:raise ValueError('unclosed skipped RDF description')
 if n_skip!=len(deprec):raise ValueError('Skipped blocks count mismatched: '+str(n_skip)+' vs '+str(len(deprec)))
 return n_skip

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()


def main():
 source=Path(sys.argv[1]);outdir=Path(sys.argv[2]);outdir.mkdir(parents=True,exist_ok=True)
 g=build_graph();base=outdir/'matsci-extraction-application-extension.ttl';base.write_text(g.serialize(format='turtle'),encoding='utf-8')
 deprec,counts,examples=safe_deprecated(source)
 droppable={x for x in deprec if counts[x]==0}; retained=set(deprec)-droppable
 out=outdir/'matsci-v11-parsing-aligned-cleaned.owl'
 make_output(source,out,g,droppable)
 manifest={
   'source':str(source.name), 'source_sha256':sha(source),'source_bytes':source.stat().st_size,
   'result':str(out.name), 'result_sha256':sha(out),'result_bytes':out.stat().st_size,
   'extension':base.name,'new_classes':len(APP_CLASSES)+len(SCI_CLASSES),'application_classes':len(APP_CLASSES),'scientific_context_classes':len(SCI_CLASSES),
   'new_object_properties':len(OBJPROPS),'new_datatype_properties':len(DATA_PROPS),'extension_triples':len(g),
   'removed_unreferenced_deprecated_iris':sorted(droppable),
   'retained_referenced_deprecated_iris':{x:{'incoming_reference_count':counts[x],'label':deprec[x],'incoming_examples':examples[x]} for x in sorted(retained)},
   'original_source_untouched':True,
   'rdfxml_bytes_preserved_except_deleted_safe_descriptions_and_added_extension':True,
   'notes':['No Pydantic schemas were edited.','OWL XML syntax validation is not OWL reasoning or Neo4j ingestion.','Reference scan checks rdf:resource links; also audit external clients before deleting public IRIs.'],
 }
 (outdir/'matsci-v11-change-manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
 with out.open('rb') as f:
  for _,el in ET.iterparse(f,events=('end',),tag='{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description',huge_tree=True):
   el.clear()
   while el.getprevious() is not None:del el.getparent()[0]
 # Parse standalone TTL to confirm.
 test=Graph();test.parse(base,format='turtle')
 assert len(test)==len(g)
 print(json.dumps({k:manifest[k] for k in ['source_bytes','result_bytes','new_classes','new_object_properties','new_datatype_properties','extension_triples']},indent=2))
 print('removed',len(droppable),'retained',len(retained),'sha256',manifest['result_sha256'])

if __name__=='__main__':main()
