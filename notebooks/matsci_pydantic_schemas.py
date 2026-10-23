"""Pydantic v2 scientific extraction data contracts matching the notebook and OWL crosswalk."""
from pydantic import BaseModel, Field
from typing import Literal, Optional



class LatticeValue(BaseModel):
    symbol: str = Field(description="a, b, c, alpha, beta, or gamma exactly as reported")
    value_original: str = Field(description="Literal reported value, including uncertainty")
    unit: str = ""

class MaterialItem(BaseModel):
    material_name: str
    formula_original: str = ""
    formula_ascii: str = ""
    doping_variable: str = ""
    doping_level: str = ""
    evidence_quote: str

class CrystallographyItem(BaseModel):
    material_name: str = ""
    structure_type: str = ""
    space_group_symbol: str = ""
    space_group_number: str = ""
    symmetry: str = ""
    lattice_parameters: list[LatticeValue] = Field(default_factory=list)
    wyckoff_sites: list[str] = Field(default_factory=list)
    evidence_quote: str

class MeasurementItem(BaseModel):
    material_name: str = ""
    property_name: str
    measurement_type: str = ""  # e.g. UV-VIS, transport, XRD, optical band-gap
    value_original: str = ""
    unit: str = ""
    uncertainty: str = ""
    symbol: str = ""  # e.g. T_c, Ta: do not infer meaning without context
    symbol_meaning: str = ""
    instrument: str = ""
    temperature_context: str = ""
    sample_conditions: str = ""
    evidence_quote: str

class SimulationItem(BaseModel):
    material_name: str = ""
    method: str = ""  # e.g. DFT, Monte Carlo, ab initio
    software_name: str = ""
    software_version: str = ""
    settings: str = ""  # e.g. XC functional, k-point mesh, cutoff
    computed_property: str = ""
    value_original: str = ""
    unit: str = ""
    evidence_quote: str

class ProcessItem(BaseModel):
    material_name: str = ""
    process_type: str  # e.g. CVT, CVD, annealing, sputtering
    acronym: str = ""
    procedure_description: str = ""
    equipment: str = ""
    temperature_symbol: str = ""
    temperature_value: str = ""
    temperature_unit: str = ""
    conditions: str = ""
    evidence_quote: str

class TrendItem(BaseModel):
    material_name: str = ""
    independent_variable: str
    independent_change: str  # e.g. increases, decreases, varying
    dependent_variable: str
    dependent_change: str
    relationship_text: str
    causal_status: Literal["observed_trend", "asserted_cause", "predicted", "unknown"]
    conditions: str = ""
    evidence_quote: str

class ClaimItem(BaseModel):
    claim_text: str
    epistemic_status: Literal["observed", "computed", "hypothesized", "inferred", "author_proposed", "unclear"]
    modality: str = ""  # e.g. likely, might, appears, necessary
    polarity: Literal["affirmed", "negated", "uncertain"]
    material_name: str = ""
    evidence_quote: str

class TermItem(BaseModel):
    term: str
    acronym: str = ""
    expansion: str = ""  # empty if not explicitly expanded
    concept_type: str = ""  # method, structure, effect, equipment, software, other
    definition: str = ""  # no guessed definitions
    is_new_term: bool = False
    evidence_quote: str

class MaterialsAndCrystals(BaseModel):
    materials: list[MaterialItem] = Field(default_factory=list)
    crystallography: list[CrystallographyItem] = Field(default_factory=list)

class MeasurementsAndMethods(BaseModel):
    measurements: list[MeasurementItem] = Field(default_factory=list)
    simulations: list[SimulationItem] = Field(default_factory=list)
    processes: list[ProcessItem] = Field(default_factory=list)

class ScientificStatements(BaseModel):
    trends: list[TrendItem] = Field(default_factory=list)
    claims: list[ClaimItem] = Field(default_factory=list)
    terminology: list[TermItem] = Field(default_factory=list)


SCIENCE_GROUPS = (("materials_crystals", MaterialsAndCrystals), ("measurements_methods", MeasurementsAndMethods), ("statements_terms", ScientificStatements))
