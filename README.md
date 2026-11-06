# feder-kg

Knowledge graph for FEDER data: an OWL ontology loaded into Neo4j, populated with entities/relationships extracted
from publications by Claude (constrained to the ontology, every item traced to a page and an exact quote),
exposed through a FastAPI service and a chatbot.

## Layout

| Path | Purpose |
|------|---------|
| `ontology/` | `.owl` files + `slice_manifest.json` (edit in Protégé) |
| `ontology/slice_manifest.json` | Which ontology file and subtrees/relations the extractor may use |
| `ontology/starter_matsci_ontology.ttl` | Stand-in ontology, used until the real `.owl` is in `ontology/` |
| `feder-graph/` | Existing graph export (`nodes`/`relationships` jsonl) |
| `notebooks/` | Class notebooks (chunking, OpenAI extraction) + `master_pipeline.ipynb` |
| `data/publications/` | Source publications (not committed — see `checksums.json`) |
| `data/extractions/` | Per-paper stage outputs (`<paper_id>/<stage>.json`, `docling.json`, figure/table `crops/`, `debug/` pages) |
| `data/graph/` | Extracted graph (`nodes`/`relationships` jsonl, `graph.ttl`, review queue, ontology gaps) |
| `data/gold/` | Hand-annotated papers for evaluation |
| `data/fewshot_examples.json` | Positive + "wrong" examples that teach Claude the extraction policy |
| `src/ontology_loader.py` | Ontology + slice manifest → Neo4j schema (constraints, class hierarchy) |
| `src/extract.py` | Claude entity/relationship extraction (runs the full pipeline) |
| `src/load_graph.py` | Writes extracted graph jsonl into Neo4j |
| `src/pipeline/docling_ingest.py` | PDF parsing with Docling: layout, reading order, tables, figures, captions; debug page overlays |
| `src/pipeline/` | Other extraction internals: schemas, Claude client, validation, export (`ingest.py` = old pdfplumber parser) |
| `src/api.py` | FastAPI wrapping the query functions |
| `src/chatbot.py` | LLM that calls the API as tools |
| `scripts/` | One-off transfer/restore scripts |
| `checksums.json` | Checksums of data files (updated by `src.extract`) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill in your values (incl. ANTHROPIC_API_KEY)
docker compose up -d             # local Neo4j at http://localhost:7474
```

Put PDFs in `data/publications/` and the ontology in `ontology/` (RDF/XML, Turtle or OWL/XML). Map the extractor's
groups (material, sample, process, technique, property, …) to your class labels in `ontology/slice_manifest.json`.

PDFs are parsed with [Docling](https://github.com/docling-project/docling). Its models (layout, TableFormer, figure
classifier) download from Hugging Face on first use; to fetch them up front:

```bash
docling-tools models download layout tableformer picture_classifier
```

## Build the graph

```bash
python -m src.extract --until structure --debug-pages   # local only (no API): parse PDFs, draw detected regions
python -m src.ontology_loader              # Neo4j constraints + ontology class hierarchy
python -m src.extract                      # PDFs → data/extractions/ → data/graph/
python -m src.load_graph --replace         # data/graph/*.jsonl → Neo4j
python -m scripts.transfer_feder_neo4j     # optional: existing feder-graph/ export → Neo4j
```

Useful flags: `python -m src.ontology_loader --dry-run` (slice summary only),
`python -m src.extract --until structure` (local stages only, no API calls),
`--debug-pages` (coloured boxes for detected text/headings/tables/figures in `data/extractions/<paper>/debug/`),
`--papers <file.pdf>` (selected papers), `--no-live` (replay cached Claude responses only),
`--table-vision fallback|always|never` (Claude re-reads table crops; default: only tables Docling couldn't structure),
`--figure-vision` (Claude describes each figure: axes, samples, conditions, no values), `--ocr` (scanned PDFs),
`--parser pdfplumber` (old heuristic parser), `--model claude-haiku-5-5` / `claude-opus-5-5` (default `claude-sonnet-5-5`).

To inspect each stage, review results and evaluate against `data/gold/`, use `notebooks/master_pipeline.ipynb`.

## Run the API

```bash
uvicorn src.api:app --reload
```
