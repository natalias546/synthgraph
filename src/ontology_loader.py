"""Load the OWL ontology and the slice manifest; create the matching Neo4j schema.

Protégé stays the source of truth. This module reads classes, labels, definitions,
synonyms, the subclass hierarchy and object properties (rdflib, with an owlready2
fallback for formats rdflib cannot read, e.g. OWL/XML), then uses
ontology/slice_manifest.json to decide which subtrees the LLM may use for each kind
of entity and which relations it may return. The allowed class labels become `enum`s in the
structured-output schema, so the model cannot invent class names; anything that
does not fit is returned as OTHER with a proposed class (an ontology-gap signal).
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


OTHER = "OTHER"

_DEF_PROPS = (
    "http://purl.obolibrary.org/obo/IAO_0000115",      # IAO definition (BFO/OBO style)
    "http://www.w3.org/2004/02/skos/core#definition",
    "http://www.w3.org/2000/01/rdf-schema#comment",
)
_SYN_PROPS = (
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://purl.obolibrary.org/obo/IAO_0000118",      # IAO alternative term
)


def _local_name(iri: str) -> str:
    name = re.split(r"[#/]", iri.rstrip("/#"))[-1]
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).replace("_", " ")
    return name.strip()


def _pick_literal(values, lang="en"):
    vals = list(values)
    for v in vals:
        if getattr(v, "language", None) in (lang, None):
            return str(v)
    return str(vals[0]) if vals else None


@dataclass
class OntClass:
    iri: str
    label: str
    definition: str | None = None
    synonyms: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)


@dataclass
class OntProperty:
    iri: str
    label: str
    kind: str  # object | data
    domain: list[str] = field(default_factory=list)
    range: list[str] = field(default_factory=list)
    definition: str | None = None


class Ontology:
    def __init__(self, path: Path):
        import rdflib
        from rdflib.namespace import OWL, RDF, RDFS, SKOS

        self.path = Path(path)
        self.loaded_with = "rdflib"
        fmt = {".ttl": "turtle", ".owl": "xml", ".rdf": "xml", ".xml": "xml", ".nt": "nt", ".jsonld": "json-ld"}.get(
            self.path.suffix.lower(), None)
        raw = self.path.read_bytes()[:4000].decode("utf-8", errors="ignore")
        owl_xml = "<Ontology" in raw and 'xmlns="http://www.w3.org/2002/07/owl#"' in raw
        g = None
        if not owl_xml:
            try:
                g = rdflib.Graph()
                g.parse(self.path, format=fmt)
            except Exception:
                g = None
        if g is None:
            g = self._load_with_owlready2()
        self.graph = g

        def label_of(s):
            lab = _pick_literal(g.objects(s, RDFS.label)) or _pick_literal(g.objects(s, SKOS.prefLabel))
            return (lab or _local_name(str(s))).strip()

        def defn(s):
            for p in _DEF_PROPS:
                v = _pick_literal(g.objects(s, rdflib.URIRef(p)))
                if v:
                    return v.strip()
            return None

        self.classes: dict[str, OntClass] = {}
        for s in set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class)):
            if isinstance(s, rdflib.URIRef):
                syns = sorted({str(o) for p in _SYN_PROPS for o in g.objects(s, rdflib.URIRef(p))})
                parents = [str(o) for o in g.objects(s, RDFS.subClassOf) if isinstance(o, rdflib.URIRef)]
                self.classes[str(s)] = OntClass(str(s), label_of(s), defn(s), syns, parents)
        # Classes that only appear as subClassOf targets (e.g. imported BFO terms)
        for c in list(self.classes.values()):
            for p in c.parents:
                if p not in self.classes and not p.endswith("#Thing"):
                    self.classes[p] = OntClass(p, label_of(rdflib.URIRef(p)), defn(rdflib.URIRef(p)))

        self.properties: dict[str, OntProperty] = {}
        for kind, t in (("object", OWL.ObjectProperty), ("data", OWL.DatatypeProperty)):
            for s in g.subjects(RDF.type, t):
                if isinstance(s, rdflib.URIRef):
                    self.properties[str(s)] = OntProperty(
                        str(s), label_of(s), kind,
                        [str(o) for o in g.objects(s, RDFS.domain) if isinstance(o, rdflib.URIRef)],
                        [str(o) for o in g.objects(s, RDFS.range) if isinstance(o, rdflib.URIRef)],
                        defn(s),
                    )

        self.children: dict[str, list[str]] = defaultdict(list)
        for c in self.classes.values():
            for p in c.parents:
                self.children[p].append(c.iri)
        self._by_label: dict[str, list[str]] = defaultdict(list)
        for c in self.classes.values():
            self._by_label[c.label.casefold()].append(c.iri)
            for syn in c.synonyms:
                self._by_label.setdefault("syn::" + syn.casefold(), []).append(c.iri)

    def _load_with_owlready2(self):
        """Fallback for OWL/XML and other formats rdflib can't parse (owlready2 -> rdflib view)."""
        try:
            import owlready2
        except ImportError as exc:
            raise ValueError(f"rdflib could not read {self.path.name} and owlready2 is not installed. "
                             "Install owlready2 or save the ontology from Protégé as RDF/XML or Turtle.") from exc
        world = owlready2.World()
        try:
            world.get_ontology(self.path.resolve().as_uri()).load(only_local=True)
        except Exception as exc:
            raise ValueError(f"Could not load {self.path.name} with rdflib or owlready2 ({type(exc).__name__}: {exc}). "
                             "If it imports other ontologies, put them in ontology/ or save a merged copy from Protégé "
                             "(File > Save as, RDF/XML).") from exc
        self.loaded_with = "owlready2"
        return world.as_rdflib_graph()

    # ---- lookups -------------------------------------------------------------
    def find_class(self, ref: str) -> str | None:
        if ref in self.classes:
            return ref
        hits = self._by_label.get(ref.casefold())
        return hits[0] if hits else None

    def find_property(self, ref: str) -> str | None:
        if ref in self.properties:
            return ref
        for p in self.properties.values():
            if p.label.casefold() == ref.casefold() or _local_name(p.iri).casefold() == ref.casefold():
                return p.iri
        return None

    def descendants(self, iri: str, include_self=True) -> list[str]:
        out, stack, seen = [], [iri], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            if x != iri or include_self:
                out.append(x)
            stack.extend(self.children.get(x, []))
        return out

    def ancestors(self, iri: str) -> set[str]:
        out, stack = set(), [iri]
        while stack:
            x = stack.pop()
            for p in self.classes.get(x, OntClass(x, x)).parents:
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    def is_a(self, iri: str, ancestor: str) -> bool:
        return iri == ancestor or ancestor in self.ancestors(iri)

    def tree_text(self, root_ref: str | None = None, max_depth: int = 8) -> str:
        """Indented class tree, like Protégé's class hierarchy view."""
        roots = [self.find_class(root_ref)] if root_ref else [
            c.iri for c in self.classes.values() if not any(p in self.classes for p in c.parents)]
        lines = []

        def walk(i, d):
            if d > max_depth:
                return
            lines.append("  " * d + self.classes[i].label)
            for ch in sorted(self.children.get(i, []), key=lambda x: self.classes[x].label.lower()):
                walk(ch, d + 1)

        for r in sorted(filter(None, roots), key=lambda x: self.classes[x].label.lower()):
            walk(r, 0)
        return "\n".join(lines)


@dataclass
class EntityGroup:
    name: str
    description: str
    class_iris: list[str]
    labels: list[str]               # enum values offered to the model (unique)
    label_to_iri: dict[str, str]
    missing_roots: list[str]


@dataclass
class RelationSpec:
    name: str
    definition: str
    subject_groups: list[str]
    object_groups: list[str]
    ontology_iri: str | None        # None = not in the ontology yet (ontology gap)
    pass_name: str                  # which extraction pass may return it


class Profile:
    """The slice manifest: which ontology subtrees each extraction pass may use."""

    def __init__(self, ontology: Ontology, manifest_path: Path):
        self.ontology = ontology
        self.raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.groups: dict[str, EntityGroup] = {}
        self.warnings: list[str] = []
        for name, spec in self.raw["entity_groups"].items():
            iris, missing = [], []
            excluded = set()
            for ref in spec.get("exclude", []) or []:
                x = ontology.find_class(ref)
                if x:
                    excluded.update(ontology.descendants(x))
            for ref in spec.get("roots", []):
                x = ontology.find_class(ref)
                if not x:
                    missing.append(ref)
                    continue
                cand = ontology.descendants(x, include_self=spec.get("include_root", True)) \
                    if spec.get("include_descendants", True) else [x]
                iris += [c for c in cand if c not in excluded and c not in iris]
            if missing:
                self.warnings.append(f"group '{name}': roots not found in ontology: {missing}")
            # Deterministic order => identical prompts/schemas across runs => cache hits.
            iris.sort(key=lambda i: (ontology.classes[i].label.casefold(), i))
            labels, l2i = [], {}
            for i in iris:
                lab = ontology.classes[i].label
                if lab in l2i:  # duplicate label: disambiguate with the IRI local name
                    lab = f"{lab} ({_local_name(i)})"
                labels.append(lab)
                l2i[lab] = i
            labels.append(OTHER)
            self.groups[name] = EntityGroup(name, spec.get("description", ""), iris, labels, l2i, missing)
            if len(labels) == 1:
                self.warnings.append(f"group '{name}' has no ontology classes; every item will be OTHER "
                                     "(add the classes in Protégé or edit the profile roots).")

        self.relations: dict[str, RelationSpec] = {}
        for r in self.raw["relations"]:
            iri = None
            for ref in r.get("ontology_property", []) or []:
                iri = ontology.find_property(ref)
                if iri:
                    break
            self.relations[r["name"]] = RelationSpec(
                r["name"], r["definition"].strip(), r["subject_groups"], r["object_groups"], iri, r.get("pass", "relations"))
        gaps = [n for n, r in self.relations.items() if r.ontology_iri is None]
        if gaps:
            self.warnings.append("relations with no matching object property in the ontology (add in Protégé "
                                 f"or map via 'ontology_property'): {gaps}")

    def iri_for(self, group: str, label: str) -> str | None:
        return self.groups[group].label_to_iri.get(label)

    def class_glossary(self, group: str, max_def_chars: int = 160) -> str:
        """Labels with short definitions, for the prompt."""
        g = self.groups[group]
        lines = []
        for lab in g.labels:
            if lab == OTHER:
                lines.append(f"- {OTHER}: nothing above fits; give proposed_class_label")
                continue
            c = self.ontology.classes[g.label_to_iri[lab]]
            d = (c.definition or "").replace("\n", " ")
            d = (d[:max_def_chars] + "...") if len(d) > max_def_chars else d
            syn = f" (synonyms: {', '.join(c.synonyms[:4])})" if c.synonyms else ""
            lines.append(f"- {lab}{syn}{': ' + d if d else ''}")
        return "\n".join(lines)

    def relations_for(self, pass_name: str) -> list[RelationSpec]:
        return [r for r in self.relations.values() if r.pass_name == pass_name]

    def summary(self) -> dict:
        return {
            "ontology_file": self.ontology.path.name,
            "loaded_with": self.ontology.loaded_with,
            "classes_in_ontology": len(self.ontology.classes),
            "object_properties_in_ontology": sum(p.kind == "object" for p in self.ontology.properties.values()),
            "groups": {n: {"allowed_classes": len(g.labels) - 1, "missing_roots": g.missing_roots}
                       for n, g in self.groups.items()},
            "relations": {n: ("ontology: " + r.ontology_iri if r.ontology_iri else "NOT IN ONTOLOGY (gap)")
                          for n, r in self.relations.items()},
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Locating the ontology file
# ---------------------------------------------------------------------------
STARTER = "starter_matsci_ontology.ttl"


def resolve_ontology_file(ontology_dir: Path, manifest_path: Path | None = None) -> tuple[Path, list[str]]:
    """Pick the ontology file: ONTOLOGY_FILE env var > slice_manifest.json 'ontology_file' >
    first other .owl/.rdf/.ttl in ontology/ > the starter ontology. Returns (path, notes)."""
    ontology_dir = Path(ontology_dir)
    notes = []
    wanted = os.getenv("ONTOLOGY_FILE")
    if not wanted and manifest_path and Path(manifest_path).exists():
        wanted = json.loads(Path(manifest_path).read_text(encoding="utf-8")).get("ontology_file")
    if wanted:
        p = Path(wanted) if Path(wanted).is_absolute() else ontology_dir / wanted
        if p.exists():
            return p, notes
        notes.append(f"{wanted} not found in {ontology_dir}")
    others = [p for ext in ("*.owl", "*.rdf", "*.ttl") for p in sorted(ontology_dir.glob(ext)) if p.name != STARTER]
    if others:
        notes.append(f"using {others[0].name}")
        return others[0], notes
    starter = ontology_dir / STARTER
    if starter.exists():
        notes.append("using the STARTER ontology (a stand-in); put your Protégé export in ontology/")
        return starter, notes
    raise FileNotFoundError(f"No ontology file in {ontology_dir}")


# ---------------------------------------------------------------------------
# Neo4j schema
# ---------------------------------------------------------------------------
def create_neo4j_schema(ontology: Ontology, driver, database: str | None = None, batch: int = 1000) -> dict:
    """Constraints/indexes for the KG, plus the class hierarchy as (:OntologyClass)-[:SUBCLASS_OF]->().

    All pipeline nodes carry the label KGNode and a unique `kg_key`; ontology classes use
    their IRI as kg_key, so extracted instances link to them with INSTANCE_OF.
    """
    rows = [{"key": c.iri, "props": {"kg_key": c.iri, "id": c.iri, "name": c.label, "definition": c.definition,
                                     "synonyms": c.synonyms}} for c in ontology.classes.values()]
    for r in rows:
        r["props"] = {k: v for k, v in r["props"].items() if v not in (None, [])}
    edges = [{"s": c.iri, "t": p} for c in ontology.classes.values() for p in c.parents if p in ontology.classes]
    with driver.session(database=database) as s:
        s.run("CREATE CONSTRAINT kgnode_key IF NOT EXISTS FOR (n:KGNode) REQUIRE n.kg_key IS UNIQUE")
        for label, prop in (("Measurement", "property"), ("Measurement", "paper_id"), ("Sample", "paper_id"),
                            ("ProcessStep", "ontology_class"), ("Parameter", "attribute"), ("OntologyClass", "name")):
            s.run(f"CREATE INDEX {label.lower()}_{prop} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")
        for i in range(0, len(rows), batch):
            s.run("UNWIND $rows AS r MERGE (n:KGNode {kg_key: r.key}) SET n:OntologyClass SET n += r.props",
                  rows=rows[i:i + batch])
        for i in range(0, len(edges), batch):
            s.run("UNWIND $rows AS r MATCH (a:KGNode {kg_key: r.s}), (b:KGNode {kg_key: r.t}) "
                  "MERGE (a)-[:SUBCLASS_OF]->(b)", rows=edges[i:i + batch])
    return {"classes": len(rows), "subclass_edges": len(edges)}


def main():
    import argparse

    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Load the ontology, report the slice, and create the Neo4j schema.")
    ap.add_argument("--dry-run", action="store_true", help="only print the ontology/slice summary")
    args = ap.parse_args()
    manifest = root / "ontology" / "slice_manifest.json"
    path, notes = resolve_ontology_file(root / "ontology", manifest)
    for n in notes:
        print("note:", n)
    onto = Ontology(path)
    prof = Profile(onto, manifest)
    print(json.dumps(prof.summary(), indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(root / ".env")
    with GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])) as d:
        print(create_neo4j_schema(onto, d))


if __name__ == "__main__":
    main()
