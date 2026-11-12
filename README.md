# feder-kg

Knowledge graph for FEDER data: an OWL ontology loaded into Neo4j, populated with entities/relationships extracted
from publications by Claude (constrained to the ontology, every item traced to a page and an exact quote),
exposed through a FastAPI service and a chatbot.

## Layout

| Path | Purpose |
|------|---------|
| `ontology/` | `.owl` files + `slice_manifest.json` (edit in Protégé) |
| `ontology/slice_manifest.json` | Which ontology file and subtrees/relations the extractor may use |
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
| `src/ontology_resolver/` | Maps extracted terms to ontology classes (no API calls); `scientific_aliases.json` holds approved aliases |
| `src/pipeline/pre_llm_review.py` | Optional sentence-by-sentence review before any Claude call (`--pre-llm-review`) |
| `compressed_owl/` | Zipped v11 ontology (the `.owl` is ~160 MB, so it is not committed directly) |
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

Extract the v11 ontology into `ontology/` (`-j` drops the zip's folders; don't unzip it into the repo root, it
would overwrite this README):

```bash
unzip -j compressed_owl/matsci_v11_parsing_aligned_package.zip '*.owl' -d ontology/
```

Put PDFs in `data/publications/`. `ontology/slice_manifest.json` names the ontology file to load and maps the
extractor's groups (material, sample, process, technique, property, phenomenon, …) to subtrees of the ontology.

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

`--pre-llm-review` (Claude reads only the sentences and table/figure crops approved in
`data/extractions/<paper>/pre_llm_review.json`; edit it with the widget in `master_pipeline.ipynb`. Default: the full paper).

To inspect each stage, review results and evaluate against `data/gold/`, use `notebooks/master_pipeline.ipynb`.

### How terms get ontology classes

After the three Claude passes, the `ontology_resolve` stage checks each extracted term against the active
ontology slice. A term gets a class IRI only if it matches exactly one class label or an approved alias in
`src/ontology_resolver/scientific_aliases.json`. Anything else (ambiguous, broader-only, or missing) is kept with
candidate classes and flagged for review, and only auto-accepted items go into `data/graph/`; items needing review
are written to `data/graph/candidates/`. To see how a term would map:

```bash
python -m src.ontology_resolver lookup property "magnetic susceptibility"
```

Offline tests (no API key): `python scripts/test_ontology_resolver.py`, `scripts/test_semantic_integration.py`,
`scripts/test_pre_llm_review.py` (each takes `--owl ontology/matsci-v11-parsing-aligned-cleaned.owl`).

Prompt caching: the system prompt (the ontology glossary, most of the input) is cached for an hour. Each pass
writes it once per run and every later paper reads it, so the first paper costs more than the rest. Check
`cache_read_input_tokens` in `pl.llm.usage_table()` after a run of two or more papers.

## Run the API

```bash
uvicorn src.api:app --reload
```
