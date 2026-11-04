"""Pipeline configuration and run manifest."""
from __future__ import annotations

import importlib.metadata
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Paths follow the repo layout (all relative to the repo root)."""
    project_root: Path
    papers_dir: Path | None = None             # data/publications/
    ontology_file: Path | None = None          # resolved from ontology/slice_manifest.json if None
    profile_file: Path | None = None           # ontology/slice_manifest.json
    fewshot_file: Path | None = None           # data/fewshot_examples.json
    outputs_dir: Path | None = None            # data/extractions/<paper_id>/<stage>.json
    graph_dir: Path | None = None              # data/graph/ (combined export)
    gold_dir: Path | None = None               # data/gold/
    cache_dir: Path | None = None              # data/cache/ (Claude responses + prompts)
    checksums_file: Path | None = None         # checksums.json
    resolution_dir: Path | None = None         # data/resolution/ (decision memory + review CSV)

    # PDF parsing
    parser: str = "docling"                    # "docling" (layout model) or "pdfplumber" (heuristic fallback)
    docling: dict = field(default_factory=dict)  # overrides for docling_ingest.DOCLING_DEFAULTS
    table_vision: str = "fallback"             # Claude reads table crops: "fallback" (only tables Docling
                                               # couldn't structure), "always", or "never"
    figure_vision: bool = False                # Claude describes each figure crop (axes, samples; no values)

    # Claude settings
    model: str = "claude-sonnet-5-5"           # cheaper bulk option: claude-haiku-5-5; strongest: claude-opus-5-5
    max_output_tokens: int = 16000
    allow_live_calls: bool = True              # False = replay cache only (no charges)
    use_prompt_caching: bool = True            # reuse the paper text across the 3 passes
    max_cost_usd: float | None = None          # stop before any Claude call if the high cost estimate is above this

    # Ontology resolution (after the Claude passes; see src/ontology_resolver)
    resolver_llm: bool = True                  # adjudicate unmatched terms with a small model; False = exact
                                               # labels / aliases / memory only, the rest stays OTHER for review
    resolver_model: str = "claude-haiku-5-5"   # cheap model for choosing among retrieved candidate classes
    resolver_top_k: int = 8                    # candidate classes retrieved per term (plus their parents)
    resolver_review_below: float = 0.8         # adjudications below this confidence go to review

    # Text handling
    max_chars_per_call: int = 120_000          # long papers are split into section chunks
    skip_sections: tuple = ("references", "acknowledgements", "author_info", "front_matter_meta")

    # Validation
    min_confidence_for_auto_accept: float = 0.85

    extra: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def __post_init__(self):
        from ..ontology_loader import resolve_ontology_file
        r = Path(self.project_root).resolve()
        self.project_root = r
        self.papers_dir = Path(self.papers_dir or r / "data" / "publications")
        self.profile_file = Path(self.profile_file or r / "ontology" / "slice_manifest.json")
        self.fewshot_file = Path(self.fewshot_file or r / "data" / "fewshot_examples.json")
        self.outputs_dir = Path(self.outputs_dir or r / "data" / "extractions")
        self.graph_dir = Path(self.graph_dir or r / "data" / "graph")
        self.gold_dir = Path(self.gold_dir or r / "data" / "gold")
        self.cache_dir = Path(self.cache_dir or r / "data" / "cache")
        self.checksums_file = Path(self.checksums_file or r / "checksums.json")
        self.resolution_dir = Path(self.resolution_dir or r / "data" / "resolution")
        load_env(r)
        if self.ontology_file is None:
            try:
                self.ontology_file, self.notes = resolve_ontology_file(r / "ontology", self.profile_file)
            except FileNotFoundError as exc:
                # Parsing doesn't need the ontology; Pipeline.ontology raises when the LLM stages do.
                self.notes = [str(exc)]
        if self.ontology_file is not None:
            self.ontology_file = Path(self.ontology_file)
        for d in (self.outputs_dir, self.cache_dir, self.graph_dir, self.resolution_dir):
            d.mkdir(parents=True, exist_ok=True)

    def fingerprint_fields(self) -> dict:
        """Settings that change extraction results (used for stage skipping)."""
        return {
            "parser": self.parser,
            "docling": self.docling,
            "table_vision": self.table_vision,
            "figure_vision": self.figure_vision,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "max_chars_per_call": self.max_chars_per_call,
            "skip_sections": list(self.skip_sections),
        }

    def as_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def load_api_key(project_root: Path) -> str | None:
    """ANTHROPIC_API_KEY from the environment, else from a local .env file. Never printed."""
    load_env(project_root)
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    return key if key and not key.endswith("...") else None


def load_env(project_root: Path) -> None:
    """Load <repo>/.env into os.environ (python-dotenv; existing variables win)."""
    env = Path(project_root) / ".env"
    if not env.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env, override=False)
    except ImportError:
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() and v.strip() and not os.getenv(k.strip()):
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")


def software_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("anthropic", "pdfplumber", "pydantic", "rdflib", "owlready2", "pint", "rank-bm25", "pandas", "neo4j"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out
