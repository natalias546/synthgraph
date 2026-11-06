"""Deterministic, evidence-preserving ontology alignment (no LLM/API calls)."""

from .resolver import OntologyResolver, annotate_validation

__all__ = ["OntologyResolver", "annotate_validation"]
