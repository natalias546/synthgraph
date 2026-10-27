"""Structured-output schemas, generated from the ontology profile.

Following the Spangher notebook: define the output contract before writing the
prompt; every item carries evidence (paragraph ID + exact quote), polarity and an
uncalibrated confidence. Ontology class fields are enums built from YOUR ontology.
Claude's structured outputs do not support numeric min/max, so ranges such as
confidence 0..1 are checked in validate.py instead of in the schema.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model

from ..ontology_loader import OTHER, Profile


class Evidence(BaseModel):
    paragraph_id: str = Field(description="ID of the paragraph/table the quote comes from, e.g. P0012")
    quote: str = Field(description="Short EXACT quote copied from that paragraph (max ~30 words)")


class Condition(BaseModel):
    name: str = Field(description="e.g. temperature, illumination, wavelength, injection level, substrate")
    value_text: str = Field(description="value with unit exactly as written, e.g. '300 K', 'AM1.5G', '1e15 cm^-3'")


Polarity = Literal["reported", "negated", "hypothetical"]


def _lit(labels: list[str]):
    return Literal[tuple(labels)]  # type: ignore[valid-type]


def build_schemas(profile: Profile) -> dict[str, type[BaseModel]]:
    g = profile.groups
    MatClass, SampleClass = _lit(g["material"].labels), _lit(g["sample"].labels)
    ProcClass, AttrClass = _lit(g["process"].labels), _lit(g["process_attribute"].labels)
    TechClass, PropClass = _lit(g["technique"].labels), _lit(g["property"].labels)
    other_labels = [l for l in g["application"].labels + g["other"].labels if l != OTHER] + [OTHER]
    OtherClass = _lit(list(dict.fromkeys(other_labels)))
    rel_names = [r.name for r in profile.relations_for("relations")] or ["related_to"]
    RelName = _lit(rel_names)
    proposed = (Optional[str], Field(None, description=f"Required when ontology_class is {OTHER}: a concise class name that should be added to the ontology"))

    Parameter = create_model(
        "Parameter",
        attribute_class=(AttrClass, ...),
        proposed_class_label=proposed,
        name_in_text=(str, ...),
        value=(Optional[float], Field(None, description="numeric value (lower bound if a range); null if not numeric")),
        value_max=(Optional[float], Field(None, description="upper bound if a range, else null")),
        value_text=(str, Field(..., description="value exactly as written, with unit")),
        unit=(Optional[str], None),
        evidence=(list[Evidence], ...),
    )
    Material = create_model(
        "Material",
        id=(str, Field(..., description="M1, M2, ...")),
        ontology_class=(MatClass, ...),
        proposed_class_label=proposed,
        name=(str, ...),
        formula=(Optional[str], None),
        aliases=(list[str], Field(default_factory=list)),
        evidence=(list[Evidence], ...),
        confidence=(float, ...),
    )
    Sample = create_model(
        "Sample",
        id=(str, Field(..., description="S1, S2, ...")),
        ontology_class=(SampleClass, ...),
        proposed_class_label=proposed,
        name_in_paper=(str, Field(..., description="how the paper refers to it, e.g. 'sample B', 'the annealed film'")),
        aliases=(list[str], Field(default_factory=list)),
        description=(str, Field(..., description="composition, form, thickness, dopants, substrate, state (as-deposited/annealed)")),
        material_ids=(list[str], Field(default_factory=list, description="IDs of Materials it is made of")),
        is_simulated_only=(bool, Field(False, description="True if this 'sample' is only a computational model")),
        evidence=(list[Evidence], ...),
        confidence=(float, ...),
    )
    ProcessStep = create_model(
        "ProcessStep",
        id=(str, Field(..., description="PR1, PR2, ...")),
        ontology_class=(ProcClass, ...),
        proposed_class_label=proposed,
        name_in_text=(str, ...),
        step_order=(Optional[int], Field(None, description="order within the sample's process history, 1 = first")),
        follows_step_id=(Optional[str], Field(None, description="ID of the immediately preceding step, if stated")),
        input_ids=(list[str], Field(default_factory=list, description="Sample/Material IDs going in (incl. substrate)")),
        output_sample_ids=(list[str], Field(default_factory=list)),
        precursor_material_ids=(list[str], Field(default_factory=list)),
        parameters=(list[Parameter], Field(default_factory=list)),
        equipment=(Optional[str], None),
        polarity=(Polarity, "reported"),
        evidence=(list[Evidence], ...),
        confidence=(float, ...),
    )
    SamplesProcesses = create_model(
        "SamplesProcesses",
        materials=(list[Material], ...),
        samples=(list[Sample], ...),
        process_steps=(list[ProcessStep], ...),
        unresolved_notes=(list[str], Field(default_factory=list, description="ambiguities a human should check")),
    )

    Measurement = create_model(
        "Measurement",
        id=(str, Field(..., description="ME1, ME2, ...")),
        technique_class=(TechClass, ...),
        proposed_technique_label=(Optional[str], Field(None, description=f"Required when technique_class is {OTHER}")),
        technique_name_in_text=(str, ...),
        instrument=(Optional[str], None),
        settings=(list[Condition], Field(default_factory=list, description="instrument settings, e.g. beam energy, incidence angle")),
        sample_ids=(list[str], Field(default_factory=list)),
        material_ids=(list[str], Field(default_factory=list, description="use when the value describes a material in general, not a specific sample")),
        property_class=(PropClass, ...),
        proposed_property_label=(Optional[str], Field(None, description=f"Required when property_class is {OTHER}")),
        property_name_in_text=(str, ...),
        value=(Optional[float], Field(None, description="numeric value (lower bound if range); null if qualitative/not reported")),
        value_max=(Optional[float], None),
        value_text=(Optional[str], Field(None, description="exactly as written incl. unit, e.g. '1.12 eV', '2.3 ± 0.1 ms'")),
        unit=(Optional[str], None),
        uncertainty_text=(Optional[str], None),
        conditions=(list[Condition], Field(default_factory=list)),
        data_origin=(Literal["experimental", "computed", "cited_from_other_work", "unclear"], ...),
        polarity=(Polarity, "reported"),
        source_location=(Literal["text", "table", "figure_caption", "figure_only_described"], ...),
        evidence=(list[Evidence], ...),
        confidence=(float, ...),
    )
    Measurements = create_model(
        "Measurements",
        measurements=(list[Measurement], ...),
        unresolved_notes=(list[str], Field(default_factory=list)),
    )

    OtherEntity = create_model(
        "OtherEntity",
        id=(str, Field(..., description="X1, X2, ...")),
        ontology_class=(OtherClass, ...),
        proposed_class_label=proposed,
        name=(str, ...),
        aliases=(list[str], Field(default_factory=list)),
        evidence=(list[Evidence], ...),
        confidence=(float, ...),
    )
    Relation = create_model(
        "Relation",
        subject_id=(str, ...),
        predicate=(RelName, ...),
        object_id=(str, ...),
        trend=(Literal["increases", "decreases", "non_monotonic", "no_effect", "not_applicable"], "not_applicable"),
        scope=(Optional[str], Field(None, description="qualifications: range studied, conditions, which samples")),
        polarity=(Literal["affirmed", "negated", "hypothetical"], ...),
        support_type=(Literal["direct", "indirect_document"], ...),
        bridge_ids=(list[str], Field(default_factory=list, description="intermediate IDs for indirect support")),
        evidence=(list[Evidence], ...),
        confidence=(float, ...),
    )

    class Rejected(BaseModel):
        subject_id: str
        predicate: str
        object_id: str
        reason: str

    RelationsOut = create_model(
        "RelationsOut",
        entities=(list[OtherEntity], ...),
        relations=(list[Relation], ...),
        rejected_candidates=(list[Rejected], ...),
    )

    class TableTranscription(BaseModel):
        table_label: str = Field(description="e.g. 'Table 1'")
        caption: str
        column_headers: list[str]
        rows: list[list[str]]
        footnotes: list[str] = Field(default_factory=list)
        unreadable_parts: list[str] = Field(default_factory=list)

    class PageTables(BaseModel):
        tables: list[TableTranscription]

    return {
        "samples_processes": SamplesProcesses,
        "measurements": Measurements,
        "relations": RelationsOut,
        "page_tables": PageTables,
    }
