"""Reproducible, narrow RDF/XML refactor of SynthGraph v11 enriched ontology.

Only existing named-class direct rdfs:subClassOf axioms are replaced. Other triples
and source formatting remain intact. New axioms are added from a small RDFLib graph.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, re
from pathlib import Path
from lxml import etree
from rdflib import Graph, Namespace, RDF, RDFS, OWL, SKOS, XSD, Literal, URIRef

FCORE=Namespace('https://w3id.org/matsci/feder-derived/core/')
FMAT=Namespace('https://w3id.org/matsci/feder-derived/materials/')
FMD=Namespace('https://w3id.org/matsci/feder-derived/measurement-diagnostics/')
MAT=Namespace('https://w3id.org/matsci/ontology/')
MT=Namespace('https://w3id.org/matsci/ontology/measurement/')
EV=Namespace('https://w3id.org/matsci/ontology/evidence/')
LIT=Namespace('https://w3id.org/matsci/ontology/literature/')
PMD=Namespace('https://w3id.org/pmd/co/')
OBO=Namespace('http://purl.obolibrary.org/obo/')
DCT=Namespace('http://purl.org/dc/terms/')

# None means leave all original parents (including relevant existing generic methods).
# For each changed existing class, list the final asserted direct parents, not new additions.
PARENTS={
 str(FCORE.MeasurementTechnique): [str(OBO.IAO_0000033)],
 str(FMAT.ElectronMicroscopy): [str(FMAT.Microscopy)],
 str(FMAT.ScanningElectronMicroscopy): [str(FMAT.ElectronMicroscopy)],
 str(FMAT.TransmissionElectronMicroscopy): [str(FMAT.ElectronMicroscopy)],
 str(FMD.GrazingIncidenceXRayDiffraction): [str(FMAT.XRayDiffraction)],
 str(FMAT.XRayDiffraction): [str(MT.DiffractionTechnique)],
 str(FMAT.DifferentialScanningCalorimetry): [str(MT.ThermalAnalysisTechnique)],
 str(FMAT.SecondaryIonMassSpectrometry): [str(MT.MassSpectrometryTechnique)],
 # Keep this class's existing IAO plan-method parent, which is not redundant
 # with the spectroscopy specialization.
 str(LIT.UVVisibleSpectroscopyMethod): [str(MAT.ExperimentalMethod),str(MT.OpticalSpectroscopyTechnique)],
 # Rietveld is a data analysis method, not an XRD instrument acquisition process.
 str(EV.RietveldRefinementMethod): [str(MAT.ExperimentalMethod),str(MT.MeasurementDataAnalysisTechnique)],
}

# Existing target category classes have disjoint BFO branches (structures vs qualities).
# A broad named target superclass accommodates both, without equating the categories.
TARGET_SUPERCLASSES={
 str(FMAT.MaterialProperty): str(MT.CharacterizationTarget),
 str(MAT.PhysicalMaterialStructure): str(MT.CharacterizationTarget),
 str(PMD.PMD_0000551): str(MT.CharacterizationTarget),
}

PAT=re.compile(r'  <rdf:Description rdf:about="([^"]+)">\n(.*?)  </rdf:Description>',re.S)
RE_PARENT=re.compile(r'^    <rdfs:subClassOf rdf:resource="([^"]+)"/>\n',re.M)

def make_additions() -> Graph:
 g=Graph()
 for pref,ns in [('mt',MT),('ev',EV),('fcore',FCORE),('fmat',FMAT),('lit',LIT),('pmd',PMD),('matsci',MAT),('obo',OBO),('dcterms',DCT),('rdf',RDF),('rdfs',RDFS),('owl',OWL),('skos',SKOS),('xsd',XSD)]:
  g.bind(pref,ns)
 def cls(uri, label, parent, desc):
  g.add((uri,RDF.type,OWL.Class))
  g.add((uri,RDFS.label,Literal(label,lang='en')))
  g.add((uri,RDFS.comment,Literal(desc,lang='en')))
  if parent:g.add((uri,RDFS.subClassOf,parent))
 def obj(uri,label,domain,range_,desc,symmetric=False):
  g.add((uri,RDF.type,OWL.ObjectProperty));g.add((uri,RDFS.label,Literal(label,lang='en')))
  g.add((uri,RDFS.domain,domain));g.add((uri,RDFS.range,range_))
  g.add((uri,RDFS.comment,Literal(desc,lang='en')))
  if symmetric:g.add((uri,RDF.type,OWL.SymmetricProperty))
 def data(uri,label,domain,range_,desc):
  g.add((uri,RDF.type,OWL.DatatypeProperty));g.add((uri,RDFS.label,Literal(label,lang='en')))
  g.add((uri,RDFS.domain,domain));g.add((uri,RDFS.range,range_));g.add((uri,RDFS.comment,Literal(desc,lang='en')))
 root=FMAT.MaterialCharacterizationTechnique
 cls(MT.DiffractionTechnique,'diffraction technique',root,'Technique specification based on diffraction; not a diffraction measurement event.')
 cls(MT.SpectroscopyTechnique,'spectroscopy technique',root,'Technique specification for acquiring spectra or energy-resolved measurements.')
 cls(MT.OpticalSpectroscopyTechnique,'optical spectroscopy technique',MT.SpectroscopyTechnique,'Spectroscopy using optical excitation, absorption, emission or transmission.')
 cls(MT.XRaySpectroscopyTechnique,'X-ray spectroscopy technique',MT.SpectroscopyTechnique,'Spectroscopy analyzing characteristic X-ray radiation.')
 cls(MT.PhotoelectronSpectroscopyTechnique,'photoelectron spectroscopy technique',MT.SpectroscopyTechnique,'Spectroscopy detecting emitted electrons following excitation.')
 cls(MT.ThermalAnalysisTechnique,'thermal analysis technique',root,'Technique for determining temperature-dependent material behavior and thermodynamic transitions.')
 cls(MT.MassSpectrometryTechnique,'mass spectrometry technique',root,'Technique measuring mass-to-charge distribution, distinct from optical/X-ray spectroscopy.')
 cls(MT.MeasurementDataAnalysisTechnique,'measurement data analysis technique',root,'Specified analysis of data from characterization; does not entail acquiring new raw observations.')
 cls(MT.EnergyDispersiveXRaySpectroscopy,'energy-dispersive X-ray spectroscopy',MT.XRaySpectroscopyTechnique,
     'Energy-resolved measurement of characteristic X-rays for elemental analysis; a technique, not an EDS detector or EDAX-branded instrument.')
 cls(MT.XRayPhotoelectronSpectroscopy,'X-ray photoelectron spectroscopy',MT.PhotoelectronSpectroscopyTechnique,
     'X-ray-excited photoelectron spectroscopy used to probe elemental composition and chemical/electronic states near material surfaces.')
 cls(MT.CharacterizationTarget,'material characterization target',OBO.BFO_0000001,
     'Material characteristic or feature investigated by an activity, encompassing physical properties, material composition, and structure. This is a broad common superclass, not an assertion that all targets are qualities.')
 g.add((MT.EnergyDispersiveXRaySpectroscopy,SKOS.altLabel,Literal('EDS')))
 g.add((MT.EnergyDispersiveXRaySpectroscopy,SKOS.altLabel,Literal('EDX')))
 g.add((MT.EnergyDispersiveXRaySpectroscopy,SKOS.altLabel,Literal('EDAX')))
 g.add((MT.EnergyDispersiveXRaySpectroscopy,SKOS.altLabel,Literal('energy-dispersive X-ray analysis')))
 g.add((MT.EnergyDispersiveXRaySpectroscopy,SKOS.altLabel,Literal('energy dispersive X-ray spectroscopy')))
 g.add((MT.XRayPhotoelectronSpectroscopy,SKOS.altLabel,Literal('XPS')))
 g.add((LIT.UVVisibleSpectroscopyMethod,SKOS.altLabel,Literal('UV-Vis')))
 g.add((LIT.UVVisibleSpectroscopyMethod,SKOS.altLabel,Literal('UV/VIS')))
 g.add((LIT.UVVisibleSpectroscopyMethod,SKOS.altLabel,Literal('UV–Vis spectroscopy')))
 g.add((FMAT.ScanningElectronMicroscopy,SKOS.altLabel,Literal('SEM')))
 g.add((FMAT.TransmissionElectronMicroscopy,SKOS.altLabel,Literal('TEM')))

 obj(MT.characterizesProperty,'characterizes property or feature',FMAT.MaterialCharacterizationProcess,MT.CharacterizationTarget,
     'Links a particular characterization process to the particular material property, composition quality or structural feature that is investigated. Relates individuals; not technique class IRIs to property class IRIs.')
 obj(MT.canCharacterize,'can characterize',FCORE.MeasurementTechnique,MT.CharacterizationTarget,
     'Capability of an individual technique specification to characterize a target. Capability is not evidence of measurement having occurred; do not apply this object property directly between two OWL class IRIs.')
 obj(MT.canBeCombinedWith,'can be combined with',FCORE.MeasurementTechnique,FCORE.MeasurementTechnique,
     'Symmetric technique-specification compatibility, not a statement that a combination was performed, nor a transitive relation.',symmetric=True)
 cls(EV.TechniqueCombinationRecord,'technique combination record',LIT.ScientificExtractionRecord,
     'Information record describing a particular proposed or verified co-use of multiple techniques in a process, with evidence and review state.')
 obj(EV.hasTechniqueCombinationRecord,'has technique combination record',FMAT.MaterialCharacterizationProcess,EV.TechniqueCombinationRecord,
     'A material characterization activity is reported to use a combination of techniques. Multiple member technique specifications are recorded on the combination record.')
 obj(EV.combinationMemberTechnique,'combination member technique',EV.TechniqueCombinationRecord,FCORE.MeasurementTechnique,
     'Links a combination record to an individual technique specification; should have at least two members, with provenance and approval before asserted use.')
 cls(EV.TechniqueCapabilityRecord,'technique capability record',LIT.ScientificExtractionRecord,
     'Information record connecting a technique class and target class as a documented domain capability, not as evidence of a particular experimental measurement.')

 cls(EV.TechniqueAliasRecord,'technique alias record',LIT.ScientificTerminologyEntry,
     'A scoped lexical alias used to propose technique or instrument interpretations; separate from OWL sameAs or equivalentClass.')
 data(EV.techniqueAliasText,'technique alias text',EV.TechniqueAliasRecord,XSD.string,'Literal surface form to be matched in a paper.')
 data(EV.aliasRole,'alias role',EV.TechniqueAliasRecord,XSD.string,'Intended candidate role such as technique or instrument.')
 data(EV.aliasResolutionPolicy,'alias resolution policy',EV.TechniqueAliasRecord,XSD.string,
      'Machine-readable policy: contextual_exact, review_if_unexpanded, or review_if_instrument_context. The downstream pre-LLM reviewer must implement this policy.')
 data(EV.aliasContextHint,'alias context hint',EV.TechniqueAliasRecord,XSD.string,'Contextual guidance for an acronym and possible competing interpretations.')
 data(EV.aliasRequiresContextReview,'alias requires context review',EV.TechniqueAliasRecord,XSD.boolean,
      'True when an unqualified occurrence must not become a confirmed technique without local context or human review.')
 for uri,label,desc in [(EV.aliasCandidateClass,'alias candidate class','Annotation referencing a candidate OWL class IRI; avoids punning a class IRI into a MeasurementTechnique individual.'),
                        (EV.capabilityTechniqueClass,'capability technique class','Class-level pointer for a documented possible technique capability; does not imply an actual observation.'),
                        (EV.capabilityTargetClass,'capability target class','Class-level pointer for a documented possible characterization target.')]:
  g.add((uri,RDF.type,OWL.AnnotationProperty));g.add((uri,RDFS.label,Literal(label,lang='en')));g.add((uri,RDFS.comment,Literal(desc,lang='en')))

 alias_rows=[
  ('EDX','technique',MT.EnergyDispersiveXRaySpectroscopy,'review_if_unexpanded',True,
   'Typically energy-dispersive X-ray spectroscopy. Confirm when the passage does not expand it or describes hardware.'),
  ('EDS','technique',MT.EnergyDispersiveXRaySpectroscopy,'review_if_unexpanded',True,
   'Common spectroscopy abbreviation; require X-ray/elemental-analysis context or an explicit definition.'),
  ('EDAX','technique',MT.EnergyDispersiveXRaySpectroscopy,'review_if_instrument_context',True,
   'Can denote energy-dispersive X-ray analysis or a brand/product. Ask whether a technique, detector, instrument, or vendor is intended.'),
  ('SEM','technique',FMAT.ScanningElectronMicroscopy,'review_if_instrument_context',True,
   'May denote scanning electron microscopy (technique) or scanning electron microscope (equipment).'),
  ('SEM','instrument',PMD.PMD_0000922,'review_if_instrument_context',True,
   'When SEM is equipment, propose the existing PMD scanning electron microscope instrument class, not the technique.'),
  ('TEM','technique',FMAT.TransmissionElectronMicroscopy,'review_if_instrument_context',True,
   'May be a microscopy method or a transmission electron microscope instrument.'),
  ('XPS','technique',MT.XRayPhotoelectronSpectroscopy,'review_if_unexpanded',True,
   'Generally X-ray photoelectron spectroscopy, but confirm role when instrument hardware is intended.'),
  ('XRD','technique',FMAT.XRayDiffraction,'contextual_exact',False,
   'X-ray diffraction; prefer explicit acronym expansion and diffraction context.'),
  ('UV-Vis','technique',LIT.UVVisibleSpectroscopyMethod,'contextual_exact',False,
   'Ultraviolet-visible spectroscopy; variants UV/VIS and UV–Vis.'),
  ('UV/VIS','technique',LIT.UVVisibleSpectroscopyMethod,'contextual_exact',False,
   'Ultraviolet-visible spectroscopy.'),
 ]
 for i,(lex,role,candidate,policy,needs_review,hint) in enumerate(alias_rows,1):
  u=URIRef(str(EV)+f'technique-alias/{i:03d}')
  g.add((u,RDF.type,EV.TechniqueAliasRecord));g.add((u,RDFS.label,Literal(f'{lex} [{role}]')))
  g.add((u,EV.techniqueAliasText,Literal(lex)))
  g.add((u,EV.aliasRole,Literal(role)))
  g.add((u,EV.aliasCandidateClass,candidate))
  g.add((u,EV.aliasResolutionPolicy,Literal(policy)))
  g.add((u,EV.aliasContextHint,Literal(hint)))
  g.add((u,EV.aliasRequiresContextReview,Literal(needs_review,datatype=XSD.boolean)))
 return g


def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()


def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('source',type=Path)
 ap.add_argument('--out',required=True,type=Path)
 ap.add_argument('--ttl',required=True,type=Path)
 ap.add_argument('--report',required=True,type=Path)
 args=ap.parse_args()
 for p in [args.out,args.ttl,args.report]:p.parent.mkdir(parents=True,exist_ok=True)
 txt=args.source.read_text(encoding='utf-8')
 changes=[]; seen=set()
 def replace(m):
  iri,body=m.group(1),m.group(2)
  if iri not in PARENTS and iri not in TARGET_SUPERCLASSES:return m.group(0)
  seen.add(iri)
  old=RE_PARENT.findall(body)
  if iri in PARENTS:
   new=PARENTS[iri]
  else:
   new=list(dict.fromkeys(old+[TARGET_SUPERCLASSES[iri]]))
  if old == new:return m.group(0)
  # Remove only direct class->named class statements; preserve blank-node constraints and annotations.
  body=RE_PARENT.sub('',body)
  line=''.join(f'    <rdfs:subClassOf rdf:resource="{p}"/>\n' for p in new)
  i=body.find('    <rdfs:label')
  if i<0:i=0
  body=body[:i]+line+body[i:]
  changes.append({'iri':iri,'old_direct_parents':old,'new_direct_parents':new,
                  'removed':sorted(set(old)-set(new)),'added':sorted(set(new)-set(old))})
  return f'  <rdf:Description rdf:about="{iri}">\n{body}  </rdf:Description>'
 out,n=PAT.subn(replace,txt)
 expected=set(PARENTS)|set(TARGET_SUPERCLASSES)
 missing=expected-seen
 if missing:raise RuntimeError('Required target class not found: '+repr(missing))
 # New triples from RDFLib; append each serialized RDF/XML child with all its
 # namespace bindings local to the child so original root namespace map is unchanged.
 g=make_additions()
 args.ttl.write_text(g.serialize(format='turtle'),encoding='utf-8')
 xml=g.serialize(format='xml')
 r=etree.fromstring(xml.encode('utf-8'))
 parts=[]
 for el in r:
  parts.append('  '+etree.tostring(el,encoding='unicode',pretty_print=True).strip().replace('\n','\n  ')+'\n')
 closing='</rdf:RDF>'
 assert out.rstrip().endswith(closing)
 ix=out.rfind(closing)
 out=out[:ix]+'\n  <!-- Measurement technique consolidation: additive v11 ontology module -->\n'+''.join(parts)+out[ix:]
 args.out.write_text(out,encoding='utf-8')
 report={'source':str(args.source),'output':str(args.out),'original_sha256':sha256(args.source),
         'updated_sha256':sha256(args.out),'original_bytes':args.source.stat().st_size,
         'updated_bytes':args.out.stat().st_size,'changed_existing_class_count':len(changes),
         'changed_classes':changes,'added_triples':len(g),
         'added_class_count':len(set(g.subjects(RDF.type,OWL.Class))),
         'added_object_property_count':len(set(g.subjects(RDF.type,OWL.ObjectProperty))),
         'added_datatype_property_count':len(set(g.subjects(RDF.type,OWL.DatatypeProperty))),
         'added_annotation_property_count':len(set(g.subjects(RDF.type,OWL.AnnotationProperty))),
         'added_alias_records':len(set(g.subjects(RDF.type,EV.TechniqueAliasRecord))),
         'alias_records':[{'text':str(next(g.objects(u,EV.techniqueAliasText))),
                           'role':str(next(g.objects(u,EV.aliasRole))),
                           'candidate_iri':str(next(g.objects(u,EV.aliasCandidateClass))),
                           'review_policy':str(next(g.objects(u,EV.aliasResolutionPolicy)))}
                          for u in sorted(set(g.subjects(RDF.type,EV.TechniqueAliasRecord)),key=str)]}
 args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['changed_existing_class_count','added_triples','added_class_count','added_object_property_count','added_datatype_property_count','added_alias_records','updated_bytes']},indent=2))

if __name__=='__main__':main()
