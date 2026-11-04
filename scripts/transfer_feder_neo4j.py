"""Transfer the existing FEDER graph export (feder-graph/) into Neo4j.

Expects Neo4j APOC JSON-lines exports in feder-graph/: either a nodes file and a
relationships file (any *.jsonl whose name contains "node" / "rel"), or one mixed
file. Records use APOC's internal ids, so nodes are keyed as "feder:<id>" and tagged
source="feder-graph"; this keeps them apart from pipeline output (source="extraction").

Usage (from the repo root):
    python -m scripts.transfer_feder_neo4j            # add/update
    python -m scripts.transfer_feder_neo4j --replace  # delete earlier feder-graph nodes first
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_graph import driver_from_env, load_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "feder-graph"))
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args()
    files = sorted(Path(args.dir).glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files in {args.dir}")
    nodes = [f for f in files if "node" in f.name.lower()]
    rels = [f for f in files if "rel" in f.name.lower()]
    with driver_from_env() as driver:
        if nodes:
            print(load_jsonl(driver, nodes[0], rels[0] if rels else None, source="feder-graph",
                             key_prefix="feder:", use_property_key=False, replace=args.replace))
        else:  # one mixed file
            print(load_jsonl(driver, files[0], None, source="feder-graph", key_prefix="feder:",
                             use_property_key=False, replace=args.replace))


if __name__ == "__main__":
    main()
