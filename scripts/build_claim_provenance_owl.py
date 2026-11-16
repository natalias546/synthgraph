"""Add an RDF Turtle extension to the last, already-restructured v11 RDF/XML OWL.

Do not apply twice. This generator intentionally does not alter existing triples.
Run: python scripts/build_claim_provenance_owl.py [--source ...] [--destination ...]
"""
from pathlib import Path
import argparse
from rdflib import Graph

def build(source:Path,extension:Path,destination:Path)->None:
    content=source.read_bytes()
    closing=b'</rdf:RDF>'
    if content.count(closing)!=1:raise ValueError('Expected a single RDF/XML root closing tag')
    if b'https://w3id.org/matsci/ontology/evidence/ClaimAttributionRecord' in content:
        raise ValueError('This extension is already in the OWL')
    g=Graph().parse(str(extension),format='turtle')
    # rdflib output is a self-contained rdf:RDF element; inject only its body.
    xml=g.serialize(format='xml',encoding='utf-8')
    marker=b'<rdf:RDF'
    a=xml.find(marker)
    if a<0:raise ValueError('RDF/XML serialization failed')
    head=xml.find(b'>',a)+1
    body=xml[head:xml.rfind(closing)]
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('wb') as f:
        f.write(content[:content.rfind(closing)])
        f.write(b'\n<!-- Additive claim provenance extension (v0.7) -->\n')
        f.write(body)
        f.write(b'\n')
        f.write(closing+b'\n')
    print(f'{len(g)} extension triples; wrote {destination}')

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default=str(root/'ontology/matsci-v11-evidence-context-enriched.owl'))
    parser.add_argument('--extension',default=str(root/'ontology/claim-provenance-extension.ttl'))
    parser.add_argument('--destination',default=str(root/'ontology/matsci-v11-provenance-enriched.owl'))
    args=parser.parse_args()
    build(Path(args.source),Path(args.extension),Path(args.destination))
