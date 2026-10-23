# Materials-science literature extraction + ontology extension (DSE203)

## Release contents

* `spangher_openrouter_matsci_ontology.ipynb` — original notebook workflow preserved, plus materials-science record extraction and Turtle instance export.
* `matsci-v10-literature-enriched.owl` — **full original RDF/XML** file with additive OWL classes/properties and annotation crosswalks; preserves original classes, URIs and individuals.
* `matsci-literature-extension-only.ttl` — lightweight **only the added schema vocabulary** (the same IRIs).
* `matsci-literature-owl-mapping.json` — exact existing-class and new-class IRI crosswalk.
* `matsci_pydantic_schemas.py` + `matsci_json_schemas.json` — Pydantic v2 types and the API response JSON Schema contracts.
* `neo4j_matsci_import.cypher` — sample commands for a clean Neo4j/n10s import.

## Setup

Install Python requirements: `pip install nbformat pdfplumber python-dotenv pydantic openai rdflib numpy pandas matplotlib`.

The **original notebook setup** uses `ROOT = Path.cwd().parent`, `PDF_DIR = ROOT / 'data' / 'publications'` and `.env` at `ROOT/'.env'`. Set your Jupyter working folder accordingly, or change `ROOT` in setup cell 2 to suit your installation. Place `matsci-literature-owl-mapping.json` directly under that `ROOT`; the large OWL does **not** have to be loaded in Jupyter for extraction.

Example environment variables in `ROOT/.env`:

```
OPENROUTER_API_KEY=<your actual key>
OPENROUTER_MODEL=openrouter/free
OPENROUTER_MAX_TOKENS=4096
SCIENCE_CHUNK_CHARS=1200
SCIENCE_OVERLAP=120
SCIENCE_MAX_PAGES=1
RUN_SCIENCE_EXTRACTION=1
```

Set `SCIENCE_MAX_PAGES=0` only after successfully testing one or two pages. Free routing is not reproducible and JSON-schema support depends on the selected model; a pinned model supporting JSON Schema is preferable. Please read the OpenRouter terms and avoid placing keys in notebooks or JSON results.

## Notebook execution

1. Set up and run original code cells through the *original noun-phrase extraction cell*. The original noun-phrase refinement and subsequent entity/relationship exercises are retained **after** the new section. The original file had no dedicated Neo4j Cypher extraction cell; a separate Cypher import/query script is included.
2. Execute the **6A** section: contracts → chunked OpenRouter calls → quote anchoring and review → RDF/Turtle export.
3. Review `results/15_materials_science_review.json` and `results/14_materials_science_records.json` before ingestion. All entries in the `unanchored` bucket are deliberately excluded from `results/16_materials_science_instances.ttl`.
4. The generated files will be in `ROOT/results/`:
   * `13_materials_science_schemas.json` — schemas
   * `14_materials_science_records.json` — quote-anchored structured observations
   * `15_materials_science_review.json` — unanchored predictions and request failures
   * `16_materials_science_instances.ttl` — n10s-compatible RDF instance graph
   * `17_materials_science_ontology_mapping.json` — crosswalk

A claim may be exact-quote anchored while remaining scientifically wrong or misconstrued. Expert review is required. No real PDF or live OpenRouter call was tested in this release.

## Ontology extension

The original XML/RDF is preserved; added resources use namespace:

`https://w3id.org/matsci/ontology/literature/`

New classes: 30, object properties: 24, datatype properties: 59. New class declarations contain `rdfs:subClassOf` links to existing ontology classes where safe. Additional `rdfs:seeAlso` links document conceptual mappings for material/crystallographic/process concepts without asserting scientifically false equivalence between **physical qualities** and **literature records**.

The ontology adds provenance evidence spans, material/composition mentions, crystallography and atomic sites, measurement vs simulation records, software settings, experimental conditions, symbolic temperature/doping parameters, qualified claims/trends, and novel/acronym terms. It does not automatically infer or validate facts about physical materials.

## Neo4j

See `neo4j_matsci_import.cypher`.

Use **n10s.rdf.import.fetch** with RDF/XML for the full OWL and Turtle for instances. Use **n10s.rdf.import.fetch** for the extension-only TTL if the original ontology is already present. **Do not** import the full ontology and then separately import the same extension vocabulary unless you intend duplicate processing. Put source files in Neo4j's configured import folder (not necessarily the notebook folder). The 150+ MB OWL contains a large amount of reference data and may require appropriate Neo4j resources.

`n10s.onto.import.fetch` imports only a simplified schema view, not individuals and all OWL annotations. To preserve both original source individuals and the new literature instance graph, this guide uses `n10s.rdf.import.fetch` for both.

## Known limitations

* `pdfplumber` does not read image-only PDFs or accurately recover every chart, complex table, math symbol or subscript. The new workflow passes source `raw_text` to the model while anchoring into the original notebook's normalized `text`; both are stored.
* Some fields, such as lattice site occupancy and settings, retain a lexical JSON/string alongside typed subnodes. No independent scientific normalization, crystallographic validity, DOI lookup, or canonical term approval is performed.
* OpenRouter structured output support, quotas, routing and model behavior can change. A truncated call is retried with smaller source pieces but failures are logged; they are **not** silently counted as negative observations.
* A production Neo4j setup should add SHACL validation, controlled term curation, stable sample identities, tested cross-paper entity resolution, and non-text figure/table extraction.
* The retained original demonstration notebook includes Spangher/plasma-specific teaching examples; those **are not** a scientific materials benchmark. Run the new section independently.

## Validation performed

* Full combined OWL parsed successfully as XML.
* Incremental Turtle OWL extension parsed as RDF with the expected class/property declarations.
* All notebook Python cells passed syntax parsing.
* Simulated three-call scientific extraction returned all eight categories and serialized a syntactically valid Turtle graph of 141 RDF triples.
* **No live OpenRouter API calls and no actual publication PDF run** were performed; no extraction accuracy claim is made.
