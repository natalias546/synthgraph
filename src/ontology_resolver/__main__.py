"""Inspect how the resolver maps a term (no Claude calls).

    python -m src.ontology_resolver lookup property "magnetic susceptibility"

Approved aliases live in src/ontology_resolver/scientific_aliases.json (tracked in git).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    from ..pipeline.config import Config
    from ..pipeline.runner import Pipeline

    ap = argparse.ArgumentParser(description="Show how a term would be resolved against the active ontology slice.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("lookup", help="exact match (if any) plus lexical candidates for a term")
    lk.add_argument("group")
    lk.add_argument("term")
    args = ap.parse_args()

    r = Pipeline(Config(project_root=ROOT)).ontology_resolver
    print(json.dumps(r.resolve(args.term, args.group), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
