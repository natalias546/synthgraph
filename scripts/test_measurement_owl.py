"""Streaming structural and RDF-usage checks for restructured measurement ontology.

Does not run an OWL DL reasoner; does check exact asserted paths, domain/range,
classes/aliases, and XML well-formedness without loading 164MB RDF into RAM.
"""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
from lxml import etree
from rdflib import Graph, RDF, RDFS, OWL, SKOS, URIRef

FCORE='https://w3id.org/matsci/feder-derived/core/'
FMAT='https://w3id.org/matsci/feder-derived/materials/'
FMD='https://w3id.org/matsci/feder-derived/measurement-diagnostics/'
MAT='https://w3id.org/matsci/ontology/'
MT=MAT+'measurement/'
EV=MAT+'evidence/'
LIT=MAT+'literature/'
OBO='http://purl.obolibrary.org/obo/'
OWL_NS='http://www.w3.org/2002/07/owl#'
RDF_NS='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
RDFS_NS='http://www.w3.org/2000/01/rdf-schema#'
SKOS_NS='http://www.w3.org/2004/02/skos/core#'


def scan(path):
 rdf_about=f'{{{RDF_NS}}}about';rdf_resource=f'{{{RDF_NS}}}resource';
 properties=collections.defaultdict(dict);parents=collections.defaultdict(set)
 types=collections.Counter();classes=set();uris=set();alt=collections.defaultdict(set);aliases=collections.defaultdict(dict)
 count=0
 # End events and previous-sibling clearing: bounded memory usage for the large OWL.
 context=etree.iterparse(str(path),events=('end',),tag=f'{{{RDF_NS}}}Description',huge_tree=True)
 for _,el in context:
  uri=el.get(rdf_about)
  if uri:
   uris.add(uri);count+=1
   for child in el:
    tag=child.tag
    if tag==f'{{{RDF_NS}}}type':
     t=child.get(rdf_resource)
     if t and t.startswith(OWL_NS):types[t.rsplit('#',1)[1]]+=1
     if t == OWL_NS+'Class':classes.add(uri)
    elif tag==f'{{{RDFS_NS}}}subClassOf':
     p=child.get(rdf_resource)
     if p:parents[uri].add(p)
    elif tag==f'{{{SKOS_NS}}}altLabel':
     if child.text:alt[uri].add(child.text.strip())
    elif tag in (f'{{{RDFS_NS}}}domain',f'{{{RDFS_NS}}}range',f'{{{OWL_NS}}}inverseOf'):
     properties[uri][tag.split('}',1)[1]]=child.get(rdf_resource)
    elif uri.startswith(EV+'technique-alias/') and tag.startswith('{'+EV+'}'):
     aliases[uri][tag.split('}',1)[1]]=child.get(rdf_resource) if child.get(rdf_resource) else (child.text or '').strip()
  el.clear()
  while el.getprevious() is not None:
   del el.getparent()[0]
 return {'parents':parents,'types':types,'properties':properties,'classes':classes,'uris':uris,'aliases':aliases,'alt':alt,'descriptions':count}


def validate_structure(d):
 p=d['parents'];props=d['properties'];cls=d['classes']
 def parent(a,b):
  assert b in p[a], f'Expected subclass: {a} => {b}; actual={p[a]}'
 def notparent(a,b):
  assert b not in p[a],f'Redundant superclass remains: {a} => {b}'
 def ancestor(a,b):
  stack=list(p[a]);seen=set()
  while stack:
   x=stack.pop()
   if x==b:return
   if x in seen:continue
   seen.add(x);stack.extend(p[x])
  raise AssertionError(f'No asserted/inherited class path from {a} to {b}')
 def relation(a,domain,range_):
  assert props[a]['domain']==domain,(a,props[a])
  assert props[a]['range']==range_,(a,props[a])
 parent(FCORE+'MeasurementTechnique',OBO+'IAO_0000033')
 assert p[FCORE+'MeasurementTechnique']=={OBO+'IAO_0000033'},p[FCORE+'MeasurementTechnique']
 notparent(FCORE+'MeasurementTechnique',OBO+'BFO_0000001')
 ancestor(FMAT+'ScanningElectronMicroscopy',FCORE+'MeasurementTechnique')
 ancestor(FMAT+'TransmissionElectronMicroscopy',FCORE+'MeasurementTechnique')
 ancestor(FMAT+'XRayDiffraction',FCORE+'MeasurementTechnique')
 ancestor(FMAT+'DifferentialScanningCalorimetry',FCORE+'MeasurementTechnique')
 ancestor(FMAT+'SecondaryIonMassSpectrometry',FCORE+'MeasurementTechnique')
 ancestor(LIT+'UVVisibleSpectroscopyMethod',FCORE+'MeasurementTechnique')
 ancestor(EV+'RietveldRefinementMethod',FCORE+'MeasurementTechnique')
 ancestor(MT+'EnergyDispersiveXRaySpectroscopy',FCORE+'MeasurementTechnique')
 ancestor(MT+'XRayPhotoelectronSpectroscopy',FCORE+'MeasurementTechnique')
 parent(FMAT+'XRayDiffraction',MT+'DiffractionTechnique')
 parent(FMAT+'DifferentialScanningCalorimetry',MT+'ThermalAnalysisTechnique')
 parent(FMAT+'SecondaryIonMassSpectrometry',MT+'MassSpectrometryTechnique')
 parent(FMAT+'ElectronMicroscopy',FMAT+'Microscopy')
 assert p[FMAT+'ElectronMicroscopy']=={FMAT+'Microscopy'}
 assert p[FMAT+'ScanningElectronMicroscopy']=={FMAT+'ElectronMicroscopy'}
 assert p[FMAT+'TransmissionElectronMicroscopy']=={FMAT+'ElectronMicroscopy'}
 assert p[FMD+'GrazingIncidenceXRayDiffraction']=={FMAT+'XRayDiffraction'}
 # Maintain physical process vs information-specification separation.
 parent(FMAT+'MaterialCharacterizationProcess',FCORE+'MeasurementProcess')
 parent(FCORE+'MeasurementProcess',OBO+'BFO_0000015')
 assert OBO+'BFO_0000015' not in p[FCORE+'MeasurementTechnique']
 relation(FCORE+'isSpecifiedBy',FCORE+'MeasurementProcess',FCORE+'MeasurementTechnique')
 relation(FMAT+'usesCharacterizationTechnique',FMAT+'MaterialCharacterizationProcess',FCORE+'MeasurementTechnique')
 relation(MT+'characterizesProperty',FMAT+'MaterialCharacterizationProcess',MT+'CharacterizationTarget')
 relation(MT+'canBeCombinedWith',FCORE+'MeasurementTechnique',FCORE+'MeasurementTechnique')
 relation(MT+'canCharacterize',FCORE+'MeasurementTechnique',MT+'CharacterizationTarget')
 for feature in [FMAT+'MaterialProperty',MAT+'PhysicalMaterialStructure','https://w3id.org/pmd/co/PMD_0000551']:
  parent(feature,MT+'CharacterizationTarget')
 for c in [MT+'CharacterizationTarget',MT+'EnergyDispersiveXRaySpectroscopy',MT+'XRayPhotoelectronSpectroscopy',EV+'TechniqueAliasRecord']:
  assert c in cls,c
 for synonym in ['EDX','EDAX','EDS']:
  assert synonym in d['alt'][MT+'EnergyDispersiveXRaySpectroscopy'],(synonym,d['alt'][MT+'EnergyDispersiveXRaySpectroscopy'])
 assert len(d['aliases'])==10,d['aliases']
 edax=[a for a in d['aliases'].values() if a.get('techniqueAliasText')=='EDAX']
 assert len(edax)==1 and edax[0]['aliasRequiresContextReview']=='true' and edax[0]['aliasResolutionPolicy']=='review_if_instrument_context'
 sem=[a for a in d['aliases'].values() if a.get('techniqueAliasText')=='SEM']
 assert {a['aliasRole'] for a in sem}=={'technique','instrument'}
 assert all(a.get('aliasCandidateClass','').startswith('https://') for a in d['aliases'].values())
 assert 'canBeCombinedWith' in {u.rsplit('/',1)[-1] for u in props}
 return len(cls)


def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('ontology',type=Path)
 ap.add_argument('--ttl',type=Path)
 args=ap.parse_args()
 if args.ttl:
  g=Graph().parse(args.ttl,format='turtle')
  assert (URIRef(MT+'canBeCombinedWith'),RDF.type,OWL.SymmetricProperty) in g
  assert (URIRef(MT+'EnergyDispersiveXRaySpectroscopy'),SKOS.altLabel,None) in g
  assert (URIRef(EV+'aliasCandidateClass'),RDF.type,OWL.AnnotationProperty) in g
  print(f'Extension Turtle parses: {len(g)} triples')
 d=scan(args.ontology)
 class_count=validate_structure(d)
 print(json.dumps({'status':'PASS','rdf_description_count':d['descriptions'],
 'owl_classes':len(d['classes']),'owl_object_property_assertions':d['types']['ObjectProperty'],
 'owl_datatype_property_assertions':d['types']['DatatypeProperty'],
 'alias_records':len(d['aliases']),'technique_hierarchy':'verified',
 'technique_process_separation':'verified','characterization_targets':'verified',
 'aliases_and_context_policies':'verified'},indent=2))

if __name__=='__main__':main()
