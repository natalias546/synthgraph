"""Write extracted graph data into Neo4j.

Reads node/relationship JSON-lines files in Neo4j's APOC export format, i.e. what
`python -m src.extract` writes to data/graph/ (and the format of the feder-graph/
export, see scripts/transfer_feder_neo4j.py):

    {"type": "node", "id": "...", "labels": ["KGNode", "Sample"], "properties": {...}}
    {"type": "relationship", "id": "...", "label": "MEASURED_ON",
     "start": {"id": "..."}, "end": {"id": "..."}, "properties": {...}}

Every node is MERGEd on (:KGNode {kg_key}). Ontology classes created by
`python -m src.ontology_loader` use their IRI as kg_key, so INSTANCE_OF edges from
extracted nodes attach to the ontology hierarchy already in the database.

Usage:
    python -m src.ontology_loader          # once: constraints + ontology classes
    python -m src.load_graph               # load data/graph/*.jsonl
    python -m src.load_graph --replace     # first remove previously loaded extraction nodes
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _q(name: str) -> str:
    """Backtick-quote a label / relationship type for Cypher."""
    return "`" + str(name).replace("`", "``") + "`"


def _clean(props: dict) -> dict:
    out = {}
    for k, v in (props or {}).items():
        if v is None:
            continue
        if isinstance(v, dict) or (isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v)):
            v = json.dumps(v, ensure_ascii=False)
        out[k] = v
    return out


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(driver, nodes_file: Path, rels_file: Path | None = None, *, source: str = "extraction",
               key_prefix: str = "", use_property_key: bool = True, database: str | None = None,
               replace: bool = False, batch: int = 500) -> dict:
    """Load APOC-format JSONL into Neo4j.

    key: properties.kg_key when present and use_property_key, else key_prefix + record id.
    source: stored on every non-ontology node; `replace=True` first deletes nodes with that source.
    """
    nodes = [r for r in read_jsonl(nodes_file) if r.get("type", "node") == "node"]
    rels = [r for r in read_jsonl(rels_file) if r.get("type") == "relationship"] if rels_file and Path(rels_file).exists() else []
    if rels_file is None:  # single mixed file
        rels = [r for r in read_jsonl(nodes_file) if r.get("type") == "relationship"]

    def key_of(rec_id, props=None):
        if use_property_key and props and props.get("kg_key"):
            return props["kg_key"]
        return f"{key_prefix}{rec_id}"

    keys = {}
    by_labels: dict[tuple, list] = {}
    for n in nodes:
        props = _clean(n.get("properties", {}))
        k = key_of(n["id"], props)
        keys[str(n["id"])] = k
        labels = tuple(sorted(set(n.get("labels", [])) - {"KGNode"}))
        if "OntologyClass" not in labels:
            props["source"] = source
        props["kg_key"] = k
        by_labels.setdefault(labels, []).append({"key": k, "props": props})

    by_type: dict[str, list] = {}
    for r in rels:
        s, t = keys.get(str(r["start"]["id"])), keys.get(str(r["end"]["id"]))
        if s is None or t is None:  # endpoint not in this file (e.g. an ontology class): use the raw id
            s = s or key_of(r["start"]["id"], {"kg_key": str(r["start"]["id"])} if use_property_key else None)
            t = t or key_of(r["end"]["id"], {"kg_key": str(r["end"]["id"])} if use_property_key else None)
        by_type.setdefault(r["label"], []).append({"s": s, "t": t, "k": f"{source}:{r.get('id')}",
                                                    "props": _clean(r.get("properties", {}))})

    with driver.session(database=database) as session:
        session.run("CREATE CONSTRAINT kgnode_key IF NOT EXISTS FOR (n:KGNode) REQUIRE n.kg_key IS UNIQUE")
        if replace:
            session.run("MATCH (n:KGNode {source: $s}) DETACH DELETE n", s=source)
        for labels, rows in by_labels.items():
            set_labels = "".join(f" SET n:{_q(lb)}" for lb in labels)
            for i in range(0, len(rows), batch):
                session.run(f"UNWIND $rows AS r MERGE (n:KGNode {{kg_key: r.key}}){set_labels} SET n += r.props",
                            rows=rows[i:i + batch])
        missing = 0
        for typ, rows in by_type.items():
            for i in range(0, len(rows), batch):
                res = session.run(
                    f"UNWIND $rows AS r MATCH (a:KGNode {{kg_key: r.s}}), (b:KGNode {{kg_key: r.t}}) "
                    f"MERGE (a)-[e:{_q(typ)} {{edge_key: r.k}}]->(b) SET e += r.props RETURN count(e) AS c",
                    rows=rows[i:i + batch]).single()
                missing += len(rows[i:i + batch]) - (res["c"] if res else 0)
    return {"nodes": len(nodes), "relationships": len(rels), "relationships_with_missing_endpoint": missing}


def driver_from_env():
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(ROOT / ".env")
    return GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))


def main():
    ap = argparse.ArgumentParser(description="Load extracted graph JSONL into Neo4j.")
    ap.add_argument("--graph-dir", default=str(ROOT / "data" / "graph"))
    ap.add_argument("--replace", action="store_true", help="delete previously loaded extraction nodes first")
    ap.add_argument("--database", default=None)
    args = ap.parse_args()
    d = Path(args.graph_dir)
    with driver_from_env() as driver:
        print(load_jsonl(driver, d / "nodes.jsonl", d / "relationships.jsonl", source="extraction",
                         replace=args.replace, database=args.database))


if __name__ == "__main__":
    main()
