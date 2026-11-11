"""Structured-output schemas for the Claude extraction passes (v11 profile).

Extraction and ontology mapping are separate jobs. These schemas ask Claude for the paper's own
scientific terms: what each thing is called, a generic `type_term` (e.g. "single crystal", "flux growth",
"magnetic susceptibility measurement"), descriptions and evidence. They contain NO ontology class lists.
src/ontology_resolver then fills the ontology fields (ontology_class / ontology_iri /
proposed_class_label, property_class / property_iri, technique_class / technique_iri, attribute_class)
that validation, export and evaluation read.

Coarse typing still comes from the extraction: where a mention sits (material, sample, process step,
parameter, measurement property/technique, application) or `entity_group` for pass-3 entities.

RULES FOR CHANGING THESE SCHEMAS (the API compiles each one into a grammar and rejects it with
400 "compiled grammar is too large" past a size limit; pass 1 has hit it twice):
  1. Only ask Claude for what only Claude can know: what the paper says, and quotes. Anything code can
     compute (pages, character offsets, IDs, ontology classes/IRIs) does not belong here.
  2. Avoid optional numbers (Optional[int/float]) and optional nested objects: each is a union in the
     grammar, and a union inside a model that is nested in every record (like Evidence) is multiplied
     by every record type. Optional strings are fine (llm.py turns them into plain strings, "" = none).
  3. If a pass needs much more output, add a pass instead of growing a schema.
  4. After any change, run `Pipeline(cfg).check_schemas()` (about $0.001; run_all also runs it before
     any paid call). It sends each schema once with a tiny prompt and names any that is too large.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, create_model

from ..ontology_loader import OTHER, Profile


class Evidence(BaseModel):
    paragraph_id: str = Field(description="Stable paragraph/table/caption ID, e.g. P0012")
    quote: str = Field(description="Exact quote from the source (short, usually <=30 words)")
    # Keep Evidence minimal: it is nested in every record, so each field here is multiplied in the compiled
    # output grammar. page / span_start / span_end are computed by validate.py from the quote;
    # source_artifact_id duplicated paragraph_id and original_notation duplicated the exact quote. With
    # those five fields, passes 1 and 3 exceeded the API's grammar size limit (checked against the API).


class Condition(BaseModel):
    name: str = Field(description="Condition name, e.g. temperature, wavelength, substrate")
    value_text: str = Field(description="Exact value with unit as printed, e.g. '300 K' or 'AM1.5G'")
    symbol: Optional[str] = Field(None, description="Symbol as printed, e.g. T_c, T_a, H_parallel")
    unit: Optional[str] = None
    uncertainty_text: Optional[str] = None


Polarity = Literal["reported", "negated", "hypothetical"]
ApplicationStatus = Literal["proposed", "demonstrated", "cited", "hypothetical", "negated", "unclear"]
ReviewStatus = Literal["unreviewed", "approved", "rejected", "unresolved"]

# Claude/OpenRouter grammar size limits: do not enumerate hundreds of class names.
MAX_ENUM = 60
# Scientific context groups are dedicated OWL subtrees; no giant GDC/OTHER catch-all.
CONTEXT_GROUPS = (
    "phenomenon", "mechanism", "theory_model", "instrument", "software",
    "phase", "structure", "reference", "dataset", "other", "unmapped",
)


def _lit(labels: list[str]):
    return Literal[tuple(labels or [OTHER])]  # type: ignore[valid-type]


def _cls(labels: list[str], group: str):
    if len(labels) <= MAX_ENUM:
        return _lit(labels)
    return Annotated[str, Field(description=(
        f"Select an exact active label from allowed ontology group '{group}', "
        f"or {OTHER}. Canonical URI is assigned by snap_classes after extraction."
    ))]


def _other_labels(profile: Profile) -> list[str]:
    """Legacy helper name: union of typed scientific context groups (not application)."""
    groups = profile.groups
    labels = [label for group in CONTEXT_GROUPS if group in groups
              for label in groups[group].labels if label != OTHER]
    return list(dict.fromkeys(labels + [OTHER]))


def build_schemas(profile: Profile) -> dict[str, type[BaseModel]]:
    g = profile.groups
    required = {"material", "sample", "process", "process_attribute", "technique",
                "property", "application", "other"}
    missing = required.difference(g)
    if missing:
        raise ValueError(f"v11 ontology profile groups missing: {sorted(missing)}")

    GroupName = _lit([name for name in CONTEXT_GROUPS if name in g])
    rel_names = [r.name for r in profile.relations_for("relations")] or ["related_to"]
    RelName = _lit(rel_names)

    def type_term(examples: str):
        return (str, Field(..., description=f"Generic type in plain scientific words, not a name or formula: {examples}. "
                                            "Do not try to match any ontology; a later step does that."))

    Parameter = create_model(
        "Parameter",
        name_in_text=(str, ...),
        type_term=type_term("the kind of parameter, e.g. 'temperature', 'time duration', 'heating rate', 'pressure'"),
        symbol=(Optional[str], None),
        value=(Optional[float], Field(None, description="Lower bound when a numeric range")),
        value_max=(Optional[float], None),
        value_text=(str, Field(..., description="Exact text and unit as written")),
        unit=(Optional[str], None),
        uncertainty_text=(Optional[str], None),
    )
    Material = create_model(
        "Material",
        id=(str, Field(..., description="M1, M2, ...")),
        name=(str, ...), formula=(Optional[str], None),
        type_term=type_term("the kind of material, e.g. 'kagome metal', 'elemental substance', 'oxide', "
                            "'flux', 'precursor powder', 'substrate'"),
        original_formula_notation=(Optional[str], None),
        aliases=(list[str], Field(default_factory=list)),
        evidence=(list[Evidence], ...), confidence=(float, ...),
    )
    Sample = create_model(
        "Sample",
        id=(str, Field(..., description="S1, S2, ...")),
        name_in_paper=(str, Field(..., description="Sample name used in paper")),
        type_term=type_term("the kind of specimen, e.g. 'single crystal', 'polycrystalline pellet', 'thin film', "
                            "'flake', 'powder'"),
        aliases=(list[str], Field(default_factory=list)),
        description=(str, Field(..., description="State, formula, form, composition, dopants")),
        material_ids=(list[str], Field(default_factory=list)),
        is_simulated_only=(bool, False),
        evidence=(list[Evidence], ...), confidence=(float, ...),
    )
    ProcessStep = create_model(
        "ProcessStep",
        id=(str, Field(..., description="PR1, PR2, ...")),
        name_in_text=(str, ...),
        type_term=type_term("the generic kind of process, e.g. 'flux growth', 'annealing', "
                            "'chemical vapour transport', 'solid-state reaction', 'cutting', 'DFT calculation'"),
        step_order=(Optional[int], None), follows_step_id=(Optional[str], None),
        input_ids=(list[str], Field(default_factory=list)),
        output_sample_ids=(list[str], Field(default_factory=list)),
        precursor_material_ids=(list[str], Field(default_factory=list)),
        parameters=(list[Parameter], Field(default_factory=list)),
        equipment=(Optional[str], None),
        software=(Optional[str], None),
        polarity=(Polarity, "reported"),
        evidence=(list[Evidence], ...), confidence=(float, ...),
    )
    SamplesProcesses = create_model(
        "SamplesProcesses", materials=(list[Material], ...),
        samples=(list[Sample], ...), process_steps=(list[ProcessStep], ...),
        unresolved_notes=(list[str], Field(default_factory=list)),
    )
    Measurement = create_model(
        "Measurement",
        id=(str, Field(..., description="ME1, ME2, ...")),
        technique_name_in_text=(str, ...),
        technique_term=type_term("generic name of the technique, e.g. 'x-ray diffraction', "
                                 "'magnetic susceptibility measurement', 'ARPES', 'four-probe resistivity measurement'"),
        instrument=(Optional[str], None),
        software=(Optional[str], None), computational_method=(Optional[str], None),
        settings=(list[Condition], Field(default_factory=list)),
        sample_ids=(list[str], Field(default_factory=list)),
        material_ids=(list[str], Field(default_factory=list)),
        property_name_in_text=(str, ...),
        property_term=type_term("generic name of the measured/computed quantity, e.g. 'superconducting "
                                "transition temperature', 'resistivity', 'lattice parameter', 'band gap'"),
        value=(Optional[float], None), value_max=(Optional[float], None),
        value_text=(Optional[str], None), unit=(Optional[str], None),
        uncertainty_text=(Optional[str], None),
        parameter_symbol=(Optional[str], Field(None, description="e.g. T_c, T_a")),
        conditions=(list[Condition], Field(default_factory=list)),
        data_origin=(Literal["experimental", "computed", "cited_from_other_work", "unclear"], ...),
        polarity=(Polarity, "reported"),
        source_location=(Literal["text", "table", "figure_caption", "figure_only_described"], ...),
        evidence=(list[Evidence], ...), confidence=(float, ...),
    )
    Measurements = create_model(
        "Measurements", measurements=(list[Measurement], ...),
        unresolved_notes=(list[str], Field(default_factory=list)),
    )

    OtherEntity = create_model(
        "OtherEntity",
        id=(str, Field(..., description="X1, X2, ...")),
        entity_group=(GroupName, "other"),
        name=(str, ...), aliases=(list[str], Field(default_factory=list)),
        original_symbol=(Optional[str], None),
        term_definition=(Optional[str], None),
        evidence=(list[Evidence], ...), confidence=(float, ...),
    )   # review_status is set by the pipeline (snap_classes / run_pass3), not asked of Claude
    ApplicationEntity = create_model(
        "ApplicationEntity",
        id=(str, Field(..., description="A1, A2, ...")),
        name=(str, Field(..., description="the application as a generic field, e.g. 'spintronics', 'quantum computing'")),
        aliases=(list[str], Field(default_factory=list)),
        application_status=(ApplicationStatus, ...),
        material_ids=(list[str], Field(default_factory=list)),
        sample_ids=(list[str], Field(default_factory=list)),
        evidence=(list[Evidence], ...), confidence=(float, ...),
    )
    UnresolvedTerm = create_model(
        "UnresolvedTerm",
        id=(str, Field(..., description="U1, U2, ...")),
        raw_term=(str, ...), possible_definition=(Optional[str], None),
        reason_unresolved=(str, ...),
        evidence=(list[Evidence], ...),
    )
    Relation = create_model(
        "Relation",
        subject_id=(str, ...), predicate=(RelName, ...),
        object_id=(str, Field(..., description="an ID, or PROP:<generic property name> for 'influences'")),
        trend=(Literal["increases", "decreases", "non_monotonic", "no_effect", "not_applicable"], "not_applicable"),
        scope=(Optional[str], None),
        polarity=(Literal["affirmed", "negated", "hypothetical"], ...),
        support_type=(Literal["direct", "indirect_document"], ...),
        bridge_ids=(list[str], Field(default_factory=list)),
        evidence=(list[Evidence], ...), confidence=(float, ...),
    )
    ComparisonRecord = create_model(
        "ComparisonRecord",
        id=(str, Field(..., description="CMP1, CMP2, ...")),
        subject_id=(str, ...), comparator_id=(str, ...),
        property_name=(Optional[str], None),
        baseline_value_text=(Optional[str], None),
        target_value_text=(Optional[str], None),
        trend=(Literal["increases", "decreases", "equal", "different", "non_monotonic", "unspecified"], "unspecified"),
        conditions=(list[Condition], Field(default_factory=list)),
        scope=(Optional[str], None),
        causal_status=(Literal["not_established", "claimed", "unknown"], "not_established"),
        evidence=(list[Evidence], ...), confidence=(float, ...),
    )

    class Rejected(BaseModel):
        subject_id: str
        predicate: str
        object_id: str
        reason: str

    # Pass 3 is two calls (one schema with all six lists exceeded the API's grammar size limit):
    # 3a entities/relations/rejected/unresolved terms, 3b applications/comparisons.
    RelationsOut = create_model(
        "RelationsOut",
        entities=(list[OtherEntity], ...),
        relations=(list[Relation], ...),
        rejected_candidates=(list[Rejected], ...),
        unresolved_terms=(list[UnresolvedTerm], Field(default_factory=list)),
    )
    ApplicationsComparisons = create_model(
        "ApplicationsComparisons",
        applications=(list[ApplicationEntity], Field(default_factory=list)),
        comparisons=(list[ComparisonRecord], Field(default_factory=list)),
    )

    class TableTranscription(BaseModel):
        table_label: str
        caption: str
        column_headers: list[str]
        rows: list[list[str]]
        footnotes: list[str] = Field(default_factory=list)
        unreadable_parts: list[str] = Field(default_factory=list)

    class PageTables(BaseModel):
        tables: list[TableTranscription]

    class Axis(BaseModel):
        quantity: str
        unit: Optional[str] = None

    class Panel(BaseModel):
        panel: str
        plot_type: str
        x_axis: Optional[Axis] = None
        y_axis: Optional[Axis] = None
        series_or_samples: list[str] = Field(default_factory=list)
        conditions: list[str] = Field(default_factory=list)
        what_it_shows: str

    class FigureDescription(BaseModel):
        panels: list[Panel]
        materials_printed: list[str] = Field(default_factory=list)
        summary: str

    return {
        "figure_description": FigureDescription,
        "samples_processes": SamplesProcesses,
        "measurements": Measurements,
        "relations": RelationsOut,
        "applications_comparisons": ApplicationsComparisons,
        "page_tables": PageTables,
    }


def _canonical(s: str) -> str:
    return re.sub(r"[\s_-]+", " ", unicodedata.normalize("NFKC", s).strip()).casefold()


def _match(value, labels: list[str]) -> str | None:
    """Match a label exactly or via unique canonicalized spelling; never fuzzy-match."""
    if not isinstance(value, str):
        return None
    if value in labels:
        return value
    k = _canonical(value)
    hits = [lab for lab in labels if _canonical(lab) == k]
    if len(hits) == 1:
        return hits[0]
    # Original label prior to disambiguation e.g. 'crystal (PMD_0020003)'.
    hits = [lab for lab in labels if _canonical(lab.split(" (")[0]) == k]
    return hits[0] if len(hits) == 1 else None


def _group_match(value: str | None, profile: Profile, group: str) -> tuple[str, str] | None:
    if not isinstance(value, str) or not value.strip() or value == OTHER:
        return None
    g = profile.groups[group]
    if value in g.class_iris:
        iri = value
        return next((lab, iri) for lab, i in g.label_to_iri.items() if i == iri)
    label = _match(value, g.labels)
    if label is not None and label != OTHER:
        return label, g.label_to_iri[label]
    # Only exact ontology synonyms, alternate labels, and local IRI names.
    needle = _canonical(value)
    candidates = []
    for lab, iri in g.label_to_iri.items():
        c = profile.ontology.classes[iri]
        local = re.split(r"[/#]", iri)[-1]
        names = list(c.synonyms) + [local]
        if needle in {_canonical(name) for name in names}:
            candidates.append((lab, iri))
    return candidates[0] if len(candidates) == 1 else None


def _snap_group(obj: dict, field: str, group: str, proposed_field: str,
                profile: Profile, iri_field: str = "ontology_iri") -> None:
    original = obj.get(field)
    # Explicit URI wins when valid for this group; otherwise preserve it as unresolved.
    requested = obj.get(iri_field)
    resolved = _group_match(requested, profile, group) if requested else None
    if resolved is None:
        resolved = _group_match(original, profile, group)
    if resolved:
        obj[field], obj[iri_field] = resolved
        if obj.get(proposed_field) == obj[field]:
            obj[proposed_field] = None
    else:
        if not obj.get(proposed_field) and original not in (None, OTHER):
            obj[proposed_field] = original
        obj[field], obj[iri_field] = OTHER, None


def _snap(obj: dict, field: str, labels: list[str], proposed: str) -> None:
    """Legacy utility used elsewhere: group-free label snapping only."""
    hit = _match(obj.get(field), labels)
    if hit is None:
        if not obj.get(proposed) and obj.get(field) not in (None, OTHER):
            obj[proposed] = obj[field]
        hit = OTHER
    obj[field] = hit


def snap_classes(data: dict, profile: Profile, kind: str) -> dict:
    """Attach canonical active ontology IRIs; keep unresolved terminology for review.

    Call on the parsed dict *before* final validation/storage. This does not assert
    semantic truth; it merely ensures identifiers resolve into the declared slice.
    """
    g = profile.groups
    if kind == "samples_processes":
        for m in data.get("materials", []):
            _snap_group(m, "ontology_class", "material", "proposed_class_label", profile)
        for s in data.get("samples", []):
            _snap_group(s, "ontology_class", "sample", "proposed_class_label", profile)
        for p in data.get("process_steps", []):
            _snap_group(p, "ontology_class", "process", "proposed_class_label", profile)
            for prm in p.get("parameters", []):
                _snap_group(prm, "attribute_class", "process_attribute", "proposed_class_label", profile)
    elif kind == "measurements":
        for m in data.get("measurements", []):
            _snap_group(m, "technique_class", "technique", "proposed_technique_label",
                        profile, "technique_iri")
            _snap_group(m, "property_class", "property", "proposed_property_label",
                        profile, "property_iri")
    elif kind == "relations":
        for a in data.get("applications", []):
            _snap_group(a, "ontology_class", "application", "proposed_class_label", profile)
        for e in data.get("entities", []):
            e.setdefault("review_status", "unreviewed")
            group = e.get("entity_group") or "other"
            raw_class = e.get("ontology_class")
            if group not in g or group not in CONTEXT_GROUPS:
                group = "unmapped"
            # Recover group if model left default 'other' but selected an unambiguous class from a typed group.
            if group == "other" and raw_class not in (None, OTHER):
                matches = [name for name in CONTEXT_GROUPS if name in g
                           and _group_match(raw_class, profile, name)]
                if len(matches) == 1:
                    group = matches[0]
            e["entity_group"] = group
            _snap_group(e, "ontology_class", group, "proposed_class_label", profile)
            if e.get("ontology_class") == OTHER:
                e["review_status"] = "unresolved"
        for r in data.get("relations", []):
            spec = profile.relations.get(r.get("predicate"))
            if spec:
                r["predicate_iri"] = spec.ontology_iri
    return data
