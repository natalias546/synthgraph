"""Stage 7 — Export: nodes/relationships JSONL (APOC JSON-lines format, like feder-graph/),
RDF Turtle, review queue and ontology-gap report. Loading into Neo4j: src/load_graph.py.

Every node and edge keeps provenance: paper_id, status, confidence and the evidence
quotes (paragraph, page, quote, anchor result). Nothing is marked verified.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..ontology_loader import OTHER, Profile
from .resolve import global_material_id

INCLUDE_STATUSES = ("auto_accepted", "needs_review")


def _ev(it: dict) -> str:
    return json.dumps([{k: a.get(k) for k in ("paragraph_id", "page", "quote", "anchor")}
                       for a in it.get("evidence_anchors", [])], ensure_ascii=False)


class GraphBuilder:
    def __init__(self, profile: Profile, include_statuses=INCLUDE_STATUSES):
        self.profile = profile
        self.include = set(include_statuses)
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self._add_ontology()

    # -- primitives ------------------------------------------------------------
    def node(self, nid: str, label: str, **props):
        n = self.nodes.setdefault(nid, {"id": nid, "label": label})
        for k, v in props.items():
            if v is not None and v != [] and k not in n:
                n[k] = v
        return n

    def edge(self, src: str, typ: str, dst: str, **props):
        if src in self.nodes and dst in self.nodes:
            rel = self.profile.relations.get(typ.lower())
            self.edges.append({"source": src, "type": typ.upper(), "target": dst,
                               "ontology_property": rel.ontology_iri if rel else None, **props})

    def _add_ontology(self):
        o = self.profile.ontology
        for c in o.classes.values():
            self.node(c.iri, "OntologyClass", name=c.label, definition=c.definition)
        for c in o.classes.values():
            for p in c.parents:
                if p in self.nodes:
                    self.edges.append({"source": c.iri, "type": "SUBCLASS_OF", "target": p})

    def _class_edge(self, nid, group, label, proposed=None):
        iri = self.profile.iri_for(group, label)
        if iri:
            self.edge(nid, "INSTANCE_OF", iri)
        elif label == OTHER:
            self.nodes[nid]["proposed_class"] = proposed

    # -- one paper --------------------------------------------------------------
    def add_paper(self, paper_id: str, meta: dict, resolved: dict):
        self.node(paper_id, "Paper", title=meta.get("title"), pdf=meta.get("pdf"), sha256=meta.get("sha256"))
        items = [it for it in resolved["items"] if it["status"] in self.include]
        local = {}  # local ID -> graph node ID

        def base(it):
            d = it["data"]
            return {"paper_id": paper_id, "local_id": d.get("id"), "status": it["status"],
                    "confidence": d.get("confidence"), "evidence": _ev(it), "verified": False}

        for it in items:
            d = it["data"]
            if it["kind"] == "material":
                gid = global_material_id(self.profile, d["ontology_class"], d["name"], d.get("formula"))
                self.node(gid, "Material", name=d["name"], formula=d.get("formula"),
                          ontology_class=d["ontology_class"])
                self.nodes[gid].setdefault("aliases", [])
                self.nodes[gid]["aliases"] = sorted(set(self.nodes[gid]["aliases"]) | set(d.get("aliases", [])) | {d["name"]})
                self._class_edge(gid, "material", d["ontology_class"], d.get("proposed_class_label"))
                local[d["id"]] = gid
                self.edge(gid, "MENTIONED_IN", paper_id, **base(it))
            elif it["kind"] == "sample":
                nid = f"{paper_id}::{d['id']}"
                self.node(nid, "Sample", name=d["name_in_paper"], description=d["description"],
                          aliases=d.get("aliases"), is_simulated_only=d.get("is_simulated_only"),
                          ontology_class=d["ontology_class"], **base(it))
                self._class_edge(nid, "sample", d["ontology_class"], d.get("proposed_class_label"))
                local[d["id"]] = nid
                self.edge(nid, "REPORTED_IN", paper_id)
            elif it["kind"] == "process_step":
                nid = f"{paper_id}::{d['id']}"
                self.node(nid, "ProcessStep", name=d["name_in_text"], step_order=d.get("step_order"),
                          equipment=d.get("equipment"), polarity=d.get("polarity"),
                          ontology_class=d["ontology_class"], **base(it))
                self._class_edge(nid, "process", d["ontology_class"], d.get("proposed_class_label"))
                local[d["id"]] = nid
                self.edge(nid, "REPORTED_IN", paper_id)
            elif it["kind"] == "measurement":
                nid = f"{paper_id}::{d['id']}"
                self.node(nid, "Measurement", property=d["property_class"], technique=d["technique_class"],
                          property_in_text=d["property_name_in_text"], technique_in_text=d["technique_name_in_text"],
                          value=d.get("value"), value_max=d.get("value_max"), value_text=d.get("value_text"),
                          unit=d.get("unit"), value_si=d.get("value_si"), unit_si=d.get("unit_si"),
                          uncertainty=d.get("uncertainty_text"), data_origin=d.get("data_origin"),
                          polarity=d.get("polarity"), source_location=d.get("source_location"),
                          conditions=json.dumps(d.get("conditions", []), ensure_ascii=False),
                          settings=json.dumps(d.get("settings", []), ensure_ascii=False),
                          instrument=d.get("instrument"), merged_ids=it.get("merged_ids"),
                          proposed_property=d.get("proposed_property_label"),
                          proposed_technique=d.get("proposed_technique_label"), **base(it))
                local[d["id"]] = nid
                self.edge(nid, "REPORTED_IN", paper_id)
            elif it["kind"] == "entity":
                nid = f"{paper_id}::{d['id']}"
                self.node(nid, "Entity", name=d["name"], aliases=d.get("aliases"),
                          ontology_class=d["ontology_class"], **base(it))
                g = "application" if self.profile.iri_for("application", d["ontology_class"]) else "other"
                self._class_edge(nid, g, d["ontology_class"], d.get("proposed_class_label"))
                local[d["id"]] = nid
                self.edge(nid, "REPORTED_IN", paper_id)

        # parameters (need their parent step to exist)
        for it in items:
            if it["kind"] == "parameter" and it.get("parent_id") in local:
                d = it["data"]
                nid = f"{paper_id}::{d['id']}"
                self.node(nid, "Parameter", name=d["name_in_text"], attribute=d["attribute_class"],
                          value=d.get("value"), value_max=d.get("value_max"), value_text=d.get("value_text"),
                          unit=d.get("unit"), value_si=d.get("value_si"), unit_si=d.get("unit_si"), **base(it))
                self._class_edge(nid, "process_attribute", d["attribute_class"], d.get("proposed_class_label"))
                local[d["id"]] = nid
                self.edge(local[it["parent_id"]], "HAS_PARAMETER", nid, paper_id=paper_id)

        # structural edges from passes 1-2
        for it in items:
            d, k = it["data"], it["kind"]
            me = local.get(d.get("id"))
            if me is None:
                continue
            prov = {"paper_id": paper_id, "status": it["status"], "evidence": _ev(it)}
            if k == "sample":
                for mid in d.get("material_ids", []):
                    if mid in local:
                        self.edge(me, "COMPOSED_OF", local[mid], **prov)
            elif k == "process_step":
                for x in d.get("input_ids", []) + d.get("precursor_material_ids", []):
                    if x in local:
                        self.edge(me, "HAS_INPUT", local[x], **prov)
                for x in d.get("output_sample_ids", []):
                    if x in local:
                        self.edge(me, "HAS_OUTPUT", local[x], **prov)
                if d.get("follows_step_id") in local:
                    self.edge(local[d["follows_step_id"]], "PRECEDES", me, **prov)
            elif k == "measurement":
                for x in d.get("sample_ids", []):
                    if x in local:
                        self.edge(me, "MEASURED_ON", local[x], **prov)
                for x in d.get("material_ids", []):
                    if x in local:
                        self.edge(me, "OF_MATERIAL", local[x], **prov)
                t = self.profile.iri_for("technique", d["technique_class"])
                p = self.profile.iri_for("property", d["property_class"])
                if t:
                    self.edge(me, "USES_TECHNIQUE", t, paper_id=paper_id)
                if p:
                    self.edge(me, "MEASURES", p, paper_id=paper_id)

        # pass-3 relations
        for it in items:
            if it["kind"] != "relation":
                continue
            d = it["data"]
            s = local.get(d["subject_id"])
            o = local.get(d["object_id"])
            if d["object_id"].startswith("PROP:"):
                o = self.profile.iri_for("property", d["object_id"][5:])
            if s and o:
                self.edge(s, d["predicate"], o, paper_id=paper_id, status=it["status"], confidence=d.get("confidence"),
                          trend=d.get("trend"), scope=d.get("scope"), polarity=d.get("polarity"),
                          support_type=d.get("support_type"), evidence=_ev(it), verified=False)

    # -- writers ------------------------------------------------------------------
    def prune_unused_classes(self):
        """Keep only ontology classes that are used (plus their ancestors), for a lighter graph."""
        used = {e["target"] for e in self.edges if e["type"] in ("INSTANCE_OF", "USES_TECHNIQUE", "MEASURES")}
        used |= {e["target"] for e in self.edges if self.nodes.get(e["target"], {}).get("label") == "OntologyClass"}
        keep = set(used)
        for c in list(used):
            keep |= self.profile.ontology.ancestors(c)
        self.nodes = {k: v for k, v in self.nodes.items() if v["label"] != "OntologyClass" or k in keep}
        self.edges = [e for e in self.edges if e["source"] in self.nodes and e["target"] in self.nodes]

    def write(self, out_dir: Path) -> dict:
        """Write nodes.jsonl and relationships.jsonl in Neo4j APOC JSON-lines export format.

        Node record:  {"type": "node", "id": <kg_key>, "labels": [...], "properties": {...}}
        Rel record:   {"type": "relationship", "id": "...", "label": TYPE, "start": {"id": ...},
                       "end": {"id": ...}, "properties": {...}}
        Every node carries the KGNode label and a unique kg_key property.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def props(d, drop=()):
            out = {}
            for k, v in d.items():
                if k in drop or v is None:
                    continue
                if isinstance(v, dict) or (isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v)):
                    v = json.dumps(v, ensure_ascii=False)
                out[k] = v
            return out

        with open(out_dir / "nodes.jsonl", "w", encoding="utf-8") as f:
            for n in self.nodes.values():
                p = props(n, drop=("label",))
                p["kg_key"] = n["id"]
                rec = {"type": "node", "id": n["id"], "labels": ["KGNode", n["label"]], "properties": p}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(out_dir / "relationships.jsonl", "w", encoding="utf-8") as f:
            for i, e in enumerate(self.edges):
                rec = {"type": "relationship", "id": f"e{i}", "label": e["type"],
                       "start": {"id": e["source"], "labels": ["KGNode", self.nodes[e["source"]]["label"]]},
                       "end": {"id": e["target"], "labels": ["KGNode", self.nodes[e["target"]]["label"]]},
                       "properties": props(e, drop=("source", "target", "type"))}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"nodes": len(self.nodes), "relationships": len(self.edges), "dir": str(out_dir)}

    def write_turtle(self, path: Path, base="http://example.org/matsci-kg/"):
        """RDF export typed by your ontology's class IRIs (importable with Neo4j neosemantics too)."""
        import rdflib
        from rdflib import Literal, Namespace, URIRef
        from rdflib.namespace import RDF, RDFS, XSD
        KG = Namespace(base)
        g = rdflib.Graph()
        g.bind("kg", KG)

        def ref(nid):
            return URIRef(nid) if nid.startswith("http") else KG[nid.replace("::", "/").replace(" ", "_")]

        for n in self.nodes.values():
            if n["label"] == "OntologyClass":
                continue
            s = ref(n["id"])
            g.add((s, RDF.type, KG[n["label"]]))
            if n.get("name"):
                g.add((s, RDFS.label, Literal(n["name"])))
            for k in ("value", "value_max", "value_si"):
                if isinstance(n.get(k), (int, float)):
                    g.add((s, KG[k], Literal(n[k], datatype=XSD.double)))
            for k in ("value_text", "unit", "unit_si", "data_origin", "status", "paper_id", "evidence", "proposed_class"):
                if n.get(k):
                    g.add((s, KG[k], Literal(n[k])))
        for e in self.edges:
            s, o = ref(e["source"]), ref(e["target"])
            if e["type"] == "INSTANCE_OF":
                g.add((s, RDF.type, o))
            elif e["type"] == "SUBCLASS_OF":
                continue
            else:
                p = URIRef(e["ontology_property"]) if e.get("ontology_property") else KG[e["type"].lower()]
                g.add((s, p, o))
        g.serialize(path, format="turtle")
        return len(g)


def write_review_queue(resolved_by_paper: dict, path: Path):
    rows = []
    for pid, res in resolved_by_paper.items():
        for it in res["items"]:
            if it["status"] == "auto_accepted":
                continue
            d = it["data"]
            ev = it["evidence_anchors"][0] if it["evidence_anchors"] else {}
            if it["kind"] == "measurement":
                summary = f"{d.get('property_class')} = {d.get('value_text')} ({d.get('technique_class')}) on {d.get('sample_ids')}"
            elif it["kind"] == "relation":
                summary = f"{d.get('subject_id')} {d.get('predicate')} {d.get('object_id')} [{d.get('trend')}]"
            elif it["kind"] == "parameter":
                summary = f"{it.get('parent_id')}: {d.get('attribute_class')} = {d.get('value_text')}"
            else:
                summary = f"{d.get('name') or d.get('name_in_paper') or d.get('name_in_text')} [{d.get('ontology_class')}]"
            rows.append({
                "paper_id": pid, "kind": it["kind"], "id": it["id"], "status": it["status"],
                "summary": summary,
                "issues": "; ".join(it["issues"] + it["notes"]), "confidence": d.get("confidence"),
                "page": ev.get("page"), "quote": ev.get("quote"), "anchor": ev.get("anchor"),
                "reviewer_decision": "", "reviewer_comment": "",
            })
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["paper_id"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_ontology_gaps(resolved_by_paper: dict, path: Path):
    from collections import defaultdict
    agg = defaultdict(lambda: {"count": 0, "papers": set(), "examples": set()})
    for pid, res in resolved_by_paper.items():
        for gap in res["ontology_gaps"]:
            key = (gap["group"], (gap.get("proposed_label") or "(none given)").strip().lower())
            agg[key]["count"] += 1
            agg[key]["papers"].add(pid)
            if gap.get("text"):
                agg[key]["examples"].add(gap["text"])
    rows = [{"group": g, "proposed_class": lab, "count": v["count"], "papers": len(v["papers"]),
             "examples_in_text": "; ".join(sorted(v["examples"])[:5])} for (g, lab), v in agg.items()]
    rows.sort(key=lambda r: (-r["count"], r["group"]))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["group", "proposed_class", "count", "papers", "examples_in_text"])
        w.writeheader()
        w.writerows(rows)
    return rows
