"""Ontology resolution: map extracted mentions (paper's own terms) onto the v11 OWL classes.

Deterministic exact labels / approved aliases first, then a shared decision memory, local candidate
retrieval and an optional small LLM adjudication. See resolver.py."""

from .resolver import (RESOLVER_SYSTEM, MappingDecisions, OntologyResolver, ResolutionMemory,
                       annotate_validation, apply_resolution_review, write_resolution_review)

__all__ = ["OntologyResolver", "ResolutionMemory", "MappingDecisions", "RESOLVER_SYSTEM",
           "annotate_validation", "write_resolution_review", "apply_resolution_review"]
