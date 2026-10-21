# feder-kg

Knowledge graph for FEDER data: an OWL ontology loaded into Neo4j, populated with
entities/relationships extracted by OpenAI, exposed through a FastAPI service and a chatbot.

## Layout

| Path | Purpose |
|------|---------|
| `ontology/` | `.owl` files + `slice_manifest.json` (edit in Protégé) |
| `feder-graph/` | Existing graph export (`nodes`/`relationships` jsonl) |
| `notebooks/` | Class notebooks (chunking, OpenAI extraction) |
| `data/publications/` | Source publications (not committed — see `checksums.json`) |
| `src/ontology_loader.py` | owlready2 → Neo4j schema |
| `src/extract.py` | OpenAI entity/relationship extraction |
| `src/load_graph.py` | Writes extracted triples into Neo4j |
| `src/api.py` | FastAPI wrapping the query functions |
| `src/chatbot.py` | LLM that calls the API as tools |
| `scripts/` | One-off transfer/restore scripts |
| `checksums.json` | Checksums of data files |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in your values
docker compose up -d        # local Neo4j at http://localhost:7474
```

## Run the API

```bash
uvicorn src.api:app --reload
```
