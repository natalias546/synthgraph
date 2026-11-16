"""Source-aware scientific text parsing, terminology, and quantity candidates.

This module makes no LLM calls and never rewrites source quotations. Inferred
axes, acronyms, and units remain candidates until independently checked.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

# Protect common journal abbreviations, initials, figure/equation references,
# Latin abbreviations, and scientific notation during sentence boundary detection.
_PROTECTED = re.compile(
    r"\b(?:Fig|Figs|Eq|Eqs|Ref|Refs|Prof|Dr|Mr|Ms|vs|et al|e\.g|i\.e|"
    r"Sci|Mater|Phys|Chem|Comput|Nat|Commun|Appl|Rev|Lett|Proc|J|Adv|Npj|"
    r"Inc|Ltd|No|Vol|Co)\.$", re.I)
_ABBREVIATION = re.compile(r"(?:\b[A-Z]\.){1,5}$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'\(\[A-Z0-9αβγΔ𝛥])")
_QUANTITY = re.compile(
    r"(?<![\w])(?:[±~≈<>]?\s*[-+]?\d+(?:\.\d+)?(?:\s*\(\s*\d+\s*\))?"
    r"(?:\s*(?:[×x]\s*10\s*(?:\^\s*\{?[-+]?\d+\}?|[-+]\s*\d+)))?)"
    r"\s*(?:°\s*C\s*/\s*h|K\s*/\s*h|°\s*C|°C|K|Torr|mTorr|Oe|kOe|Tesla|T\b|eV|meV|nm|µm|μm|mm|cm|Å|"
    r"GPa|MPa|Pa|mΩ\s*cm|Ω\s*cm|Ω|%|h\b|hours?\b|min\b|s\b|"
    r"°C\s*/\s*h|K\s*/\s*h|g\s*/\s*cm\^?\{?3\}?|cm\^?\{?3\}?)",
    re.I)
_PRESSURE_POWER = re.compile(r"(?<!\w)\d+\s*[-−]\s*\d+\s*Torr\b", re.I)
_SIMPLE_VALUE = re.compile(r"(?<!\w)(?:α|L\s*[_φϕ]|Lφ|Lϕ|R\s*wp|R\s*p|a|b|c)\s*=\s*[-+]?\d+(?:\.\d+)?(?:\s*\(\d+\))?",re.I)
_ACRONYM_DEF = re.compile(r"(?P<expanded>[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,7})\s*\((?P<short>[A-Z][A-Z0-9-]{1,8})\)")
_ACRONYM_LEAD = re.compile(r"\b(?P<short>[A-Z]{2,8})\s*\((?P<expanded>[A-Za-z][A-Za-z -]{5,75})\)")
_STOP_ACRONYMS = {"NI", "IN", "SI", "TI", "FM", "MC", "FC", "WL", "MR"}  # context needed, not globally forced


def split_source(text: str, kind: str = "text", role: str = "") -> list[tuple[int,int,str]]:
    """Retain bibliography/figure/table/formula as a single source object.

    Outputs exact Python character offsets into the provided source string.
    This is a conservative boundary detector, not a grammar-proof segmenter.
    """
    if not text or not text.strip():
        return []
    if kind in {"reference", "list_item", "formula", "equation", "figure", "table", "caption", "code"} or role == "references":
        a=len(text)-len(text.lstrip()); b=len(text.rstrip())
        return [(a,b,text[a:b])] if b>a else []
    positions=[0]
    for m in _SENTENCE_BREAK.finditer(text):
        prefix=text[positions[-1]:m.start()].rstrip()
        last=prefix.split()[-1] if prefix.split() else ""
        numeric_unit_ending=bool(re.search(r"\d+(?:\.\d+)?\s*(?:K|T|h|s)\.$",prefix))
        if _PROTECTED.search(prefix) or ((_ABBREVIATION.search(prefix) or re.fullmatch(r"(?:[A-Z]\.)+",last)) and not numeric_unit_ending):
            continue
        # Citation strings like (K. Kumar ...) and chained initials should not split.
        tail=prefix[-35:]
        if not numeric_unit_ending and re.search(r"\b(?:[A-Z]\.\s*){1,4}$", tail):
            continue
        positions.append(m.end())
    positions.append(len(text))
    spans=[]
    for a,b in zip(positions,positions[1:]):
        frag=text[a:b]; l=len(frag)-len(frag.lstrip()); r=len(frag.rstrip())
        if r>l: spans.append((a+l,a+r,text[a+l:a+r]))
    return spans


def quantities(sentence: str) -> list[dict]:
    found=[]; seen=set()
    for rx,kind in ((_PRESSURE_POWER,"scientific_notation_pressure"),(_QUANTITY,"quantity"),(_SIMPLE_VALUE,"parameter_assignment")):
        for m in rx.finditer(sentence):
            raw=m.group(0).strip(); key=(m.start(),m.end())
            if key in seen or (kind=="quantity" and any(m.start()>=a and m.end()<=b for a,b in seen)): continue
            seen.add(key)
            # Source form stays unchanged: '10 -5 Torr' is pressure, never '-5 T'.
            category=("pressure" if re.search(r"Torr|\bPa\b", raw, re.I) else
                      "magnetic_field" if re.search(r"(?:\bT\b|Tesla|\bOe\b)",raw,re.I) else
                      "temperature_ramp_rate" if re.search(r"(?:°C\s*/\s*h|K\s*/\s*h)",raw) else
                      "temperature" if re.search(r"(?:°C|\bK\b)",raw) else
                      "time_or_rate" if re.search(r"(?:\bh\b|hours?|°C/h|K/h)",raw,re.I) else
                      "scientific_quantity")
            found.append({"text":raw,"span_start":m.start(),"span_end":m.end(),
                          "kind":kind,"quantity_kind":category,"normalization_status":"raw_unverified"})
    return sorted(found,key=lambda x:x["span_start"])


def local_acronyms(evidence: list[dict]) -> list[dict]:
    """Extract abbreviation evidence from definitions, do not hallucinate expansions."""
    entries={}
    for e in evidence:
        sentence=e.get("sentence", "")
        if e.get("source_kind") in {"reference","list_item"} or e.get("section_role")=="references":
            continue
        for rx in (_ACRONYM_DEF,_ACRONYM_LEAD):
            for m in rx.finditer(sentence):
                short=m.group("short").upper(); expansion=m.group("expanded").strip()
                if len(short)<2 or len(expansion)<5:continue
                # Avoid greedy word capture from the full preceding clause using acronym initial letters.
                parts=re.findall(r"[A-Za-z]+",expansion)
                # Search a suffix matching acronym's letters, tolerating fillers.
                chosen=None
                for n in range(1,min(len(parts),len(short)+4)+1):
                    tail=parts[-n:]
                    initials="".join(x[0].upper() for x in tail if x.lower() not in {"of","the","and","in","for","on"})
                    if initials==short:
                        chosen=" ".join(tail);break
                if chosen is None:
                    # Common materials-science expansions may encode internal letters
                    # (MR=magnetoresistance) or omit fillers (WAL=weak antilocalization).
                    fragment=expansion.lower()
                    recognized={"MR":r"magneto[ -]?resistan\w*", "WAL":r"weak\s+anti[ -]?locali\w*",
                                "RRR":r"residual\s+resistivity\s+ratio",
                                "TSS":r"topological\s+surface\s+states?",
                                "HLN":r"hikami[ -]larkin[ -]nagaoka"}
                    rule=recognized.get(short)
                    if rule and (hit:=re.search(rule,fragment)):
                        chosen=expansion[hit.start():hit.end()]
                    else:
                        if rx is _ACRONYM_DEF: continue
                        chosen=expansion
                # Collapse orthographic variants of the same term into a single definition.
                canonical=re.sub(r"[\s-]+", " ", chosen.lower())
                canonical=canonical.replace("anti localization", "antilocalization")
                key=(short,canonical)
                entries.setdefault(key,{"acronym":short,"expansion":chosen,"definition_evidence_ids":[],
                                        "status":"document_defined_unreviewed"})["definition_evidence_ids"].append(e["id"])
    result=list(entries.values())
    for item in result:
        item["definition_evidence_ids"]=list(dict.fromkeys(item["definition_evidence_ids"]))
        item["mentions"]= [e["id"] for e in evidence if
                           e.get("source_kind") not in {"reference","list_item"} and
                           re.search(r"(?<!\w)"+re.escape(item["acronym"])+r"(?!\w)",e.get("sentence",""))]
    return sorted(result,key=lambda x:(x["acronym"],x["expansion"]))


def safe_context(term: str, group: str, sentence: str) -> dict:
    """Reject lexical homonyms before accepting OWL surface matches."""
    t=term.strip().lower(); s=sentence.lower()
    if t=="potential" and re.search(r"\bpotential\s+(?:for\s+)?applications?\b",s):
        return {"allowed":False,"reason":"potential applications is not electric potential"}
    if t=="transport" and group=="process" and re.search(r"(?:electrical|magneto|charge|electron)\s*[- ]?transport",s):
        return {"allowed":False,"reason":"electrical transport is a response/domain, not a transport process"}
    if t=="growth" and group=="property" and re.search(r"\b(?:crystal|film|sample)\s+growth\b",s):
        return {"allowed":False,"reason":"growth describes a synthesis process"}
    if t in {"mr","rrr","wal","tss","hln"}:
        return {"allowed":True,"reason":"technical acronym: use document-local expansion to resolve"}
    return {"allowed":True,"reason":"no deterministic homonym conflict"}


def evidence_blocks(evidence: list[dict]) -> list[dict]:
    """Group source-provenance-consistent evidence for hierarchical review."""
    groups=defaultdict(list)
    for e in evidence:
        # A paragraph is the default block; table, figure, and bibliographic items remain separate.
        groups[(e.get("section_id"),e.get("paragraph_id"))].append(e)
    blocks=[]
    for (sec,pid),items in groups.items():
        if not items:continue
        roles=sorted({r for e in items for r in e.get("scientific_roles",[])})
        blocks.append({"id":"EB_"+pid,"section_id":sec,"section_role":items[0].get("section_role"),
                       "paragraph_id":pid,"source_kind":items[0].get("source_kind"),
                       "evidence_ids":[e["id"] for e in items],"page":items[0].get("page"),
                       "scientific_roles":roles,"review_status":"pending"})
    return blocks
