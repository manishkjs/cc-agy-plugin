"""Vertex AI model-to-region routing, pricing table, and token budget helpers."""

from __future__ import annotations

import os
from typing import Any

# Preview and 3.5-series Vertex models that return 404 NOT_FOUND on regional
# endpoints (such as asia-south1 or europe-west4) and require the "global"
# Vertex endpoint unless explicitly overridden by a dedicated regional endpoint.
GLOBAL_ONLY_MODELS: frozenset[str] = frozenset(
    {
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite-preview",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
    }
)

# Standard Vertex regional locations supported by GA Gemini 2.5 models.
VALID_VERTEX_REGIONS: frozenset[str] = frozenset(
    {
        "global",
        "us-central1",
        "us-east1",
        "us-east4",
        "us-east5",
        "us-south1",
        "us-west1",
        "us-west4",
        "northamerica-northeast1",
        "southamerica-east1",
        "europe-west1",
        "europe-west2",
        "europe-west3",
        "europe-west4",
        "europe-west8",
        "europe-west9",
        "europe-north1",
        "europe-central2",
        "europe-southwest1",
        "asia-south1",
        "asia-southeast1",
        "asia-northeast1",
        "asia-northeast3",
        "asia-east1",
        "asia-east2",
        "australia-southeast1",
        "me-central1",
        "me-central2",
        "me-west1",
    }
)

# Per-1M token pricing in USD on Vertex AI (standard tier <= 200k context).
# Cached input tokens are billed at 10% of the base input rate on Gemini 2.5/3.x.
MODEL_PRICING_PER_1M_USD: dict[str, dict[str, float]] = {
    "gemini-2.5-flash-lite": {
        "input": 0.10,
        "cached_input": 0.01,
        "output": 0.40,
    },
    "gemini-3.5-flash-lite": {
        "input": 0.10,
        "cached_input": 0.01,
        "output": 0.40,
    },
    "gemini-3.1-flash-lite-preview": {
        "input": 0.10,
        "cached_input": 0.01,
        "output": 0.40,
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "cached_input": 0.03,
        "output": 2.50,
    },
    "gemini-3-flash-preview": {
        "input": 0.50,
        "cached_input": 0.05,
        "output": 3.00,
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "cached_input": 0.125,
        "output": 10.00,
    },
    "gemini-3.1-pro-preview": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
    },
}

# Claude Code warns when a single MCP tool response exceeds 10,000 tokens
# and truncates at 25,000 tokens. We keep digests below 8,000 tokens.
MAX_TOOL_RESPONSE_TOKENS: int = 8000
DEFAULT_DIFF_PAGE_TOKENS: int = 4000


def estimate_tokens(text: str) -> int:
    """Estimate token count from text (approx 4 bytes per token, minimum 1 for non-empty)."""
    if not text:
        return 0
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def truncate_to_token_budget(text: str, max_tokens: int = MAX_TOOL_RESPONSE_TOKENS) -> tuple[str, bool, int]:
    """Truncate text so its estimated token count does not exceed max_tokens.

    Returns (truncated_text, was_truncated, original_estimated_tokens).
    """
    orig_tokens = estimate_tokens(text)
    if orig_tokens <= max_tokens:
        return text, False, orig_tokens
    max_bytes = max(16, max_tokens * 4)
    encoded = text.encode("utf-8")[:max_bytes]
    clipped = encoded.decode("utf-8", errors="ignore")
    return clipped, True, orig_tokens


def resolve_vertex_location(
    model: str,
    requested_location: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    """Resolve a valid Vertex AI location for the given model.

    Returns (resolved_location, reroute_note).
    If a global-only model (such as gemini-3.5-flash-lite or gemini-3.1-pro-preview)
    is requested against a regional endpoint like asia-south1 or us-central1,
    automatically routes to 'global' and returns an explanatory note.
    """
    source_env = env if env is not None else os.environ
    raw_loc = (
        requested_location
        or source_env.get("GOOGLE_CLOUD_LOCATION")
        or source_env.get("VERTEX_LOCATION")
        or "us-central1"
    ).strip()

    # Handle unsubstituted Claude Code template placeholders gracefully
    if not raw_loc or raw_loc.startswith("${"):
        raw_loc = "us-central1"

    normalized_model = (model or "").strip().lower()
    if normalized_model.startswith("models/"):
        normalized_model = normalized_model[len("models/") :]

    if normalized_model in GLOBAL_ONLY_MODELS or normalized_model.endswith("-preview"):
        if raw_loc != "global":
            note = (
                f"Auto-routed model '{model}' from regional endpoint '{raw_loc}' "
                f"to 'global' (model requires global Vertex endpoint)."
            )
            return "global", note
        return "global", None

    if raw_loc not in VALID_VERTEX_REGIONS:
        raise ValueError(
            f"Unsupported Vertex AI region '{raw_loc}'. "
            f"Expected 'global' or a valid regional endpoint such as 'us-central1'."
        )
    return raw_loc, None


def resolve_vertex_project(
    requested_project: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> str:
    """Resolve the Google Cloud Project ID from arguments or environment."""
    source_env = env if env is not None else os.environ
    for candidate in (
        requested_project,
        source_env.get("GOOGLE_CLOUD_PROJECT"),
        source_env.get("VERTEX_PROJECT"),
        source_env.get("GCLOUD_PROJECT"),
    ):
        if candidate:
            cleaned = candidate.strip()
            if cleaned and not cleaned.startswith("${"):
                return cleaned
    raise ValueError(
        "Missing Google Cloud project ID. Set userConfig.google_cloud_project in Claude Code "
        "or export GOOGLE_CLOUD_PROJECT."
    )


def estimate_cost_usd(model: str, usage: dict[str, Any]) -> dict[str, float]:
    """Calculate exact USD and cents cost from raw Vertex usage_metadata token counts.

    Vertex prompt_token_count includes cached_content_token_count. Uncached input
    tokens equal max(0, prompt_token_count - cached_content_token_count).
    Thinking tokens (thoughts_token_count) are billed at the output token rate.
    """
    normalized_model = (model or "").strip().lower()
    if normalized_model.startswith("models/"):
        normalized_model = normalized_model[len("models/") :]

    rates = MODEL_PRICING_PER_1M_USD.get(
        normalized_model,
        MODEL_PRICING_PER_1M_USD["gemini-3.1-pro-preview"],
    )

    prompt_tokens = int(usage.get("prompt_token_count") or 0)
    cached_tokens = int(usage.get("cached_content_token_count") or 0)
    candidates_tokens = int(usage.get("candidates_token_count") or 0)
    thoughts_tokens = int(usage.get("thoughts_token_count") or 0)

    uncached_input_tokens = max(0, prompt_tokens - cached_tokens)
    billable_output_tokens = candidates_tokens + thoughts_tokens

    input_cost = (uncached_input_tokens / 1_000_000.0) * rates["input"]
    cached_cost = (cached_tokens / 1_000_000.0) * rates["cached_input"]
    output_cost = (billable_output_tokens / 1_000_000.0) * rates["output"]
    total_usd = input_cost + cached_cost + output_cost

    return {
        "input_cost_usd": round(input_cost, 8),
        "cached_cost_usd": round(cached_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(total_usd, 8),
        "total_cost_cents": round(total_usd * 100.0, 6),
    }
