"""Review loop for ontology decisions (no Claude calls).

    python -m src.ontology_resolver review          # write data/resolution/resolution_review.csv
    python -m src.ontology_resolver apply           # apply the filled-in `action` column
    python -m src.ontology_resolver lookup property "magnetic susceptibility"   # candidates for a term

In the CSV, set `action` per row: approve (mapping is right) | reject (wrong class; it won't be proposed
again for this term) | remap (put the right class IRI in remap_to_iri) | gap (concept is missing from
the ontology). Approvals/remaps become approved aliases in src/ontology_resolver/scientific_aliases.json
(tracked in git, so the whole team gets them). Then re-run `python -m src.extract`: only the
ontology_resolve stage and the stages after it are redone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    from ..pipeline.config import Config
    from ..pipeline.runner import Pipeline, load_results

    ap = argparse.ArgumentParser(description="Review and correct ontology mappings.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("review", help="write data/resolution/resolution_review.csv from all extracted papers")
    a = sub.add_parser("apply", help="apply the actions filled into the review CSV")
    a.add_argument("--csv", type=Path, default=None)
    lk = sub.add_parser("lookup", help="show how a term would be resolved (exact match + candidates)")
    lk.add_argument("group")
    lk.add_argument("term")
    args = ap.parse_args()

    pl = Pipeline(Config(project_root=ROOT))
    if args.cmd == "review":
        n = pl.write_resolution_review(load_results(pl.cfg.outputs_dir))
        print(f"{n} rows -> {pl.cfg.resolution_dir / 'resolution_review.csv'}")
    elif args.cmd == "apply":
        print(json.dumps(pl.apply_resolution_review(args.csv), indent=2))
    else:
        r = pl.ontology_resolver
        print(json.dumps(r.resolve(args.term, args.group), indent=2, ensure_ascii=False)[:1500])
        for c in r.candidates(args.group, [args.term]):
            print(f"  candidate: {c.label}  <{c.iri}>")


if __name__ == "__main__":
    main()
