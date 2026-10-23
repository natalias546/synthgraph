// Neo4j self-hosted + compatible n10s plugin. Commands executed separately.
// Import one of these two paths, not both: FULL OWL *or* extension-only TTL
CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS FOR (r:Resource) REQUIRE r.uri IS UNIQUE;

// Execute once in a fresh graph:
CALL n10s.graphconfig.init({handleRDFTypes:'LABELS_AND_NODES'});

// Option A: fully integrated ontology including original source records:
CALL n10s.rdf.import.fetch('file:///matsci-v10-literature-enriched.owl','RDF/XML');
// Option B: only when the base ontology is already imported:
// CALL n10s.rdf.import.fetch('file:///matsci-literature-extension-only.ttl','Turtle');

// After notebook extraction, copy results/16_materials_science_instances.ttl to Neo4j import/
CALL n10s.rdf.import.fetch('file:///16_materials_science_instances.ttl','Turtle');

// Explore quote-grounded literature nodes, regardless of class namespaced labels:
MATCH (x:Resource)-[edge]->(ev:Resource)
WHERE type(edge) CONTAINS 'hasEvidence'
RETURN x.uri AS record, ev.uri AS evidence LIMIT 50;

// Audit the extracted literature node count
MATCH (r:Resource)
WHERE r.uri STARTS WITH 'https://w3id.org/matsci/ontology/literature/instance/'
RETURN count(r) AS literature_resources;
