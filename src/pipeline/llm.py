"""Claude API wrapper: structured outputs, content-addressed cache, prompt log, usage ledger.

Mirrors the Spangher notebook's `ask()`:
  * the cache key hashes model + system + document + task + images + schema + max_tokens,
    so re-running unchanged cells replays saved responses at no cost;
  * every exact request is written to prompts/ for inspection;
  * incomplete (max_tokens) or refused responses raise instead of becoming empty results;
  * the API key is never printed or saved.
Prompt caching: the system prompt and the paper text are marked cacheable, so the
three extraction passes over one paper pay full price for the paper text only once.
"""
from __future__ import annotations

import base64
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from .io_utils import sha256_bytes, sha256_json, write_json


class IncompleteOutput(RuntimeError):
    """Output hit max_tokens; the caller should split the input into smaller chunks."""


class NoCachedResponse(RuntimeError):
    pass


def _enum_values(schema: dict) -> dict[str, str]:
    """casefold -> canonical value for every enum string in a JSON schema."""
    out = {}

    def walk(x):
        if isinstance(x, dict):
            for v in x.get("enum", []) or []:
                if isinstance(v, str):
                    out.setdefault(v.casefold(), v)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(schema)
    return out


def _repair_enum_case(data, enums: dict[str, str]):
    """Structured outputs may differ from enum values only in capitalization; fix that."""
    if isinstance(data, dict):
        return {k: _repair_enum_case(v, enums) for k, v in data.items()}
    if isinstance(data, list):
        return [_repair_enum_case(v, enums) for v in data]
    if isinstance(data, str) and data not in enums.values() and data.casefold() in enums:
        return enums[data.casefold()]
    return data


class ClaudeExtractor:
    def __init__(self, model: str, cache_dir: Path, api_key: str | None, *, max_output_tokens: int = 16000,
                 allow_live_calls: bool = True, use_prompt_caching: bool = True,
                 mock_responder: Callable | None = None):
        self.model = model
        self.cache_dir = Path(cache_dir)
        (self.cache_dir / "responses").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "prompts").mkdir(parents=True, exist_ok=True)
        self.max_output_tokens = max_output_tokens
        self.allow_live_calls = allow_live_calls
        self.use_prompt_caching = use_prompt_caching
        self.mock_responder = mock_responder
        self.ledger: list[dict] = []
        self.client = None
        if api_key and mock_responder is None:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key, max_retries=4, timeout=900)

    @property
    def status(self) -> dict:
        return {"model": self.model, "api_key_available": self.client is not None,
                "live_calls_enabled": self.allow_live_calls, "mock": self.mock_responder is not None}

    def ask(self, label: str, *, system: str, task: str, schema: type[BaseModel],
            document: str | None = None, images: list[bytes] | None = None,
            paper_id: str = "_", max_tokens: int | None = None) -> dict:
        from anthropic import transform_schema

        max_tokens = max_tokens or self.max_output_tokens
        json_schema = transform_schema(schema)
        image_hashes = [sha256_bytes(b) for b in images or []]
        request = {
            "model": self.model, "system": system, "document": document, "task": task,
            "images_sha256": image_hashes, "schema": json_schema, "max_tokens": max_tokens,
        }
        digest = sha256_json(request)
        cache_path = self.cache_dir / "responses" / f"{digest}.json"
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        write_json(self.cache_dir / "prompts" / paper_id / f"{safe_label}.json", {"request_sha256": digest, **request})

        if cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            self.ledger.append({"label": label, "paper_id": paper_id, "cached": True, **record.get("usage", {})})
            return schema.model_validate(record["data"]).model_dump()

        if self.mock_responder is not None:
            data = self.mock_responder(label=label, schema=schema, task=task, document=document)
            data = schema.model_validate(data).model_dump()
            write_json(cache_path, {"request_sha256": digest, "label": label, "paper_id": paper_id,
                                    "model": "MOCK", "usage": {}, "stop_reason": "mock", "data": data})
            self.ledger.append({"label": label, "paper_id": paper_id, "cached": False, "mock": True})
            return data

        if not self.allow_live_calls or self.client is None:
            raise NoCachedResponse(
                f"{label}: no cached response. Set ANTHROPIC_API_KEY (environment or .env) and allow_live_calls=True.")

        cc = {"cache_control": {"type": "ephemeral"}} if self.use_prompt_caching else {}
        content = []
        if document:
            content.append({"type": "text", "text": document, **cc})
        for b in images or []:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": base64.b64encode(b).decode("ascii")}})
        content.append({"type": "text", "text": task})

        t0 = time.time()
        # Streaming avoids the SDK's 10-minute limit for long non-streamed outputs.
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, **cc}],
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": json_schema}},
        ) as stream:
            response = stream.get_final_message()
        usage = response.usage.model_dump() if response.usage else {}
        if response.stop_reason == "max_tokens":
            raise IncompleteOutput(f"{label}: output hit max_tokens={max_tokens}")
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{label}: the model refused this request")
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        raw = json.loads(text)
        try:
            data = schema.model_validate(raw).model_dump()
        except ValidationError:
            data = schema.model_validate(_repair_enum_case(raw, _enum_values(json_schema))).model_dump()
        record = {
            "request_sha256": digest, "label": label, "paper_id": paper_id,
            "response_id": response.id, "model": response.model, "usage": usage,
            "stop_reason": response.stop_reason, "seconds": round(time.time() - t0, 1),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data": data,
        }
        write_json(cache_path, record)
        self.ledger.append({"label": label, "paper_id": paper_id, "cached": False, **usage})
        print(f"  Claude call completed: {label} ({record['seconds']} s)", flush=True)
        return data

    def usage_table(self):
        import pandas as pd
        cols = ["label", "paper_id", "cached", "input_tokens", "output_tokens",
                "cache_creation_input_tokens", "cache_read_input_tokens"]
        df = pd.DataFrame(self.ledger)
        return df[[c for c in cols if c in df.columns]] if len(df) else df
