"""Cost estimate for the Claude stages, shown before any call is made.

Per paper and document chunk the pipeline makes three calls (passes 1-3), plus one vision call per
table crop sent to Claude (and per figure with figure_vision). Every call sends the system prompt
(rules + ontology glossary) and the paper text, both marked for prompt caching. Each pass has its own
output schema, which is part of the cached prefix, so the system prompt is written once per pass in a
run and read for later papers, and the paper text is written by every pass. The task text (with the
registry of earlier passes) is never cached.

Input sizes are counted exactly with the token-counting endpoint when an API key is set (free), else
estimated from characters. Output length can't be known in advance: until there are saved responses
for this model, a range is assumed per call type; after the pilot, the median and 90th percentile of
your own saved responses are used instead.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

# USD per million tokens: (input, output, cache read). Writing the 5-minute cache costs 1.25 x input,
# the 1-hour cache 2 x input. Check https://www.anthropic.com/pricing before large runs.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
CACHE_WRITE = 1.25      # 5-minute entries (not used by the pipeline now; kept for reference)
CACHE_WRITE_1H = 2.0    # 1-hour entries (the system prompt; Config.system_cache_ttl)

# Output tokens per call (including thinking) assumed until real responses exist: (low, high).
OUTPUT_GUESS = {"pass1": (3000, 12000), "pass2": (4000, 16000), "pass3": (2000, 8000), "pass3b": (800, 4000),
                "table": (1000, 3000), "figure": (500, 1500), "resolve": (600, 2000)}
# Ontology adjudication (small model, after pass 3): batches of up to 15 unmatched terms with their
# candidate classes. Calls per paper drop as the shared decision memory fills up.
RESOLVE_CALLS_PER_PAPER = (1, 5)
RESOLVE_INPUT_TOKENS = 4500
TASK_TOKENS = 700          # fixed task instructions + few-shot examples, per pass call
REGISTRY_SHARE = 0.5       # the registry passed to passes 2-3 is roughly half the earlier passes' output
IMAGE_TOKENS = 1600        # a table/figure crop after the API's resize (about the maximum)
CHARS_PER_TOKEN = 2.5      # fallback when token counting isn't available; papers with formulas measured ~2.6


class CostLimitExceeded(RuntimeError):
    pass


def _call_type(label: str) -> str | None:
    for k in ("pass1", "pass2", "pass3b", "pass3"):   # pass3b before pass3 (prefix)
        if label.startswith(k):
            return k
    if label.startswith("table"):
        return "table"
    if label.startswith("figure"):
        return "figure"
    if label.startswith("resolve_"):
        return "resolve"
    return None


def observed_outputs(cache_dir: Path, model: str) -> dict:
    """Output tokens of saved responses for this model, per call type."""
    seen: dict[str, list[int]] = {}
    for f in (Path(cache_dir) / "responses").glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        kind = _call_type(r.get("label", ""))
        out = (r.get("usage") or {}).get("output_tokens")
        if kind and out and str(r.get("model", "")).startswith(model):
            seen.setdefault(kind, []).append(out)
    return seen


def output_range(seen: dict, min_samples: int = 3) -> dict:
    """(low, high, source) per call type: median and 90th percentile once there are enough samples."""
    out = {}
    for kind, guess in OUTPUT_GUESS.items():
        xs = sorted(seen.get(kind, []))
        if len(xs) >= min_samples:
            p90 = xs[min(len(xs) - 1, math.ceil(0.9 * len(xs)) - 1)]
            out[kind] = (statistics.median(xs), max(p90, statistics.median(xs)), f"{len(xs)} saved responses")
        else:
            out[kind] = (*guess, "assumed")
    return out


class TokenCounter:
    def __init__(self, client=None, model: str | None = None):
        self.client, self.model, self.exact = client, model, client is not None

    def __call__(self, text: str) -> int:
        if self.exact:
            try:
                return self.client.messages.count_tokens(
                    model=self.model, messages=[{"role": "user", "content": text}]).input_tokens
            except Exception:       # no network, unknown model...: fall back for the rest of the run
                self.exact = False
        return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate(model: str, system_tokens: int, papers: list[dict], outputs: dict, passes: int = 3) -> dict:
    """papers: [{"paper_id", "doc_tokens": [per chunk], "images": n, "saved": bool}]; passes: how many of
    passes 1-3 the run goes up to. Returns per-paper rows and totals in USD (low/high)."""
    if model not in PRICES:
        raise KeyError(f"No prices for {model}; add it to PRICES in src/pipeline/cost.py")
    p_in, p_out, p_read = (x / 1e6 for x in PRICES[model])
    rows, first = [], True
    for p in papers:
        if p["saved"]:
            rows.append({**p, "calls": 0, "low_usd": 0.0, "high_usd": 0.0})
            continue
        cost = {"low": 0.0, "high": 0.0}
        calls = 0
        for doc in p["doc_tokens"] if passes else []:
            for bound, i in (("low", 0), ("high", 1)):
                outs = [outputs[k][i] for k in ("pass1", "pass2", "pass3")][:passes]
                n_calls = passes + (passes == 3)   # pass 3 is two calls (3a relations, 3b applications)
                if passes == 3:
                    outs.append(outputs["pass3b"][i])
                # The output schema is part of the cached prefix, so each pass has its own system-prompt
                # entry (1-hour TTL): written by the first paper, read by every later one (assumes the same
                # pass recurs <1 h apart). The paper text is not cached (each pass sends a different schema).
                sys_cost = n_calls * system_tokens * (CACHE_WRITE_1H * p_in if first else p_read)
                doc_cost = n_calls * doc * p_in
                registries = sum(sum(outs[:k]) for k in range(1, passes))   # pass 2 sees pass 1, pass 3 sees 1+2
                task_cost = (n_calls * TASK_TOKENS + REGISTRY_SHARE * registries) * p_in
                cost[bound] += sys_cost + doc_cost + task_cost + sum(outs) * p_out
            first = False
            calls += passes + (passes == 3)
        for _ in range(p["images"]):
            for bound, i in (("low", 0), ("high", 1)):
                cost[bound] += (system_tokens * p_read + (IMAGE_TOKENS + TASK_TOKENS) * p_in
                                + outputs["table"][i] * p_out)
            calls += 1
        rows.append({**p, "calls": calls, "low_usd": round(cost["low"], 3), "high_usd": round(cost["high"], 3)})
    return {"rows": rows, "low_usd": sum(r["low_usd"] for r in rows), "high_usd": sum(r["high_usd"] for r in rows),
            "calls": sum(r["calls"] for r in rows)}


def resolver_estimate(model: str, n_papers: int, outputs: dict | None = None) -> dict:
    """Low/high USD for the ontology adjudication calls of n_papers (assumption-based, see constants)."""
    if model not in PRICES or n_papers == 0:
        return {"calls": (0, 0), "low_usd": 0.0, "high_usd": 0.0}
    p_in, p_out, _ = (x / 1e6 for x in PRICES[model])
    out = (outputs or {}).get("resolve") or OUTPUT_GUESS["resolve"]
    lo_calls, hi_calls = (n_papers * c for c in RESOLVE_CALLS_PER_PAPER)
    return {"calls": (lo_calls, hi_calls),
            "low_usd": round(lo_calls * (RESOLVE_INPUT_TOKENS * p_in + out[0] * p_out), 3),
            "high_usd": round(hi_calls * (RESOLVE_INPUT_TOKENS * p_in + out[1] * p_out), 3)}


def summary_text(model: str, system_tokens: int, exact: bool, outputs: dict, result: dict) -> str:
    new = [r for r in result["rows"] if not r["saved"]]
    saved = len(result["rows"]) - len(new)
    docs = sum(sum(r["doc_tokens"]) for r in new)
    lines = [
        f"COST ESTIMATE before any Claude call ({model}; prices in src/pipeline/cost.py)",
        f"  papers to extract: {len(new)}" + (f"  (+{saved} already extracted, assumed reused at no cost)" if saved else ""),
        f"  Claude calls: {result['calls']}  |  system prompt: {system_tokens:,} tokens (cached)  |  "
        f"paper text: {docs:,} tokens ({'counted exactly' if exact else 'estimated from characters'})",
        "  output per call: " + ", ".join(f"{k} {lo:,.0f}-{hi:,.0f} ({src})" for k, (lo, hi, src) in outputs.items()
                                          if k not in ("figure", "resolve")),
        *([f"  ontology adjudication ({result['resolver']['model']}): {result['resolver']['calls'][0]}-"
           f"{result['resolver']['calls'][1]} small calls, ${result['resolver']['low_usd']:.2f} - "
           f"${result['resolver']['high_usd']:.2f} (included below)"] if result.get("resolver") else []),
        f"  ESTIMATED COST: ${result['low_usd']:.2f} - ${result['high_usd']:.2f}",
    ]
    return "\n".join(lines)
