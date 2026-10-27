"""Small I/O helpers: hashing, JSON save/load with read-back checks, stage skipping."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))


def portable(value: Any) -> Any:
    """Convert to plain JSON types; NaN/inf become null (as in the Murari notebook)."""
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [portable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return portable(value.model_dump())
    return value


def write_json(path: Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = portable(value)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == value, f"Read-back mismatch: {path}"
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paper_id_from_filename(path: Path) -> str:
    """Stable, filesystem-safe paper ID from the PDF filename."""
    stem = Path(path).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")[:80]


class StageStore:
    """Per-paper stage outputs at data/extractions/<paper_id>/<stage>.json.

    A stage is skipped when its saved output exists and was produced from the same
    input fingerprint (e.g. the PDF hash + config + upstream stage hash).
    """

    def __init__(self, root: Path, paper_id: str):
        self.dir = Path(root) / paper_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str) -> Path:
        return self.dir / f"{stage}.json"

    def load(self, stage: str, fingerprint: str | None = None) -> Any | None:
        p = self.path(stage)
        if not p.exists():
            return None
        record = read_json(p)
        if fingerprint is not None and record.get("_fingerprint") != fingerprint:
            return None
        return record

    def save(self, stage: str, value: dict, fingerprint: str | None = None) -> dict:
        value = dict(value)
        value["_stage"] = stage
        value["_fingerprint"] = fingerprint
        write_json(self.path(stage), value)
        return value
