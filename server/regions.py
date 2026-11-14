"""Vertex AI model-to-region routing, pricing table, and token budget helpers."""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping
import urllib.error
import urllib.parse
import urllib.request

from server.env_flags import (
    COST_PROFILE_ENV_KEYS as COST_PROFILE_ENV_KEYS,
    STRICT_MODE_ENV_KEYS as STRICT_MODE_ENV_KEYS,
    VALID_COST_PROFILES as VALID_COST_PROFILES,
    env_flag as env_flag,
    first_valid_env as first_valid_env,
    is_strict_mode_enabled as is_strict_mode_enabled,
    resolve_cost_profile as resolve_cost_profile,
)
from server.rate_card import (
    MODEL_PRICING_PER_1M_USD as MODEL_PRICING_PER_1M_USD,
    check_promotional_rate_expiry as check_promotional_rate_expiry,
    clamp_gemini_thinking_level as clamp_gemini_thinking_level,
    clamp_output_tokens_to_headroom as clamp_output_tokens_to_headroom,
    compute_compaction_threshold as compute_compaction_threshold,
    resolve_model_input_modalities as resolve_model_input_modalities,
    resolve_model_token_limits as resolve_model_token_limits,
    supported_gemini_thinking_levels as supported_gemini_thinking_levels,
)


def _real_account_home() -> Path | None:
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except Exception:
        try:
            return Path.home().resolve()
        except Exception:
            return None


_REAL_ACCOUNT_HOME: Path | None = _real_account_home()

# Vertex models that return 404 NOT_FOUND on regional endpoints (such as
# asia-south1 or europe-west4) and require the "global" Vertex endpoint.
GLOBAL_ONLY_MODELS: frozenset[str] = frozenset(
    {
        "gemini-4-argon",
        "argon",
        "gemini-4-carbon",
        "carbon",
        "gemini-4-carbon-high",
        "gemini-4-carbon-medium",
        "gemini-4-carbon-low",
        "gemini-4-carbon-zero",
        "carbon-g",
        "carbon-g-zero",
        "bon-experiment-arm-226",
        "gemini-flash-next",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite-preview",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
    }
)

# Models that support multi-segment sequential continuationToken decode (up to 65,536
# output tokens per streamed segment and 3x default output token budget).
LONG_DECODE_MODELS: frozenset[str] = frozenset(
    {
        "gemini-4-carbon",
        "carbon",
        "gemini-4-carbon-high",
        "gemini-4-carbon-medium",
        "gemini-4-carbon-low",
        "gemini-4-carbon-zero",
        "carbon-g",
        "carbon-g-zero",
    }
)


def is_long_decode_model(model: str | None) -> bool:
    """Return True when `model` supports multi-segment `continuationToken` long-decode."""
    if not model:
        return False
    cleaned = str(model).strip().lower()
    if "/models/" in cleaned:
        cleaned = cleaned[cleaned.rfind("/models/") + len("/models/") :]
    elif cleaned.startswith("models/"):
        cleaned = cleaned[len("models/") :]
    return cleaned in LONG_DECODE_MODELS or cleaned.startswith("gemini-4-carbon")


def resolve_ask_thinking_level(
    thinking_level: str | None,
    model: str | None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve thinking level for `ask_gemini`, defaulting `gemini-4-carbon` to `MEDIUM`."""
    if thinking_level and str(thinking_level).strip():
        return str(thinking_level).strip().upper()
    source_env = env if env is not None else os.environ
    cleaned = str(model or "").strip().lower()
    if "/models/" in cleaned:
        cleaned = cleaned[cleaned.rfind("/models/") + len("/models/") :]
    elif cleaned.startswith("models/"):
        cleaned = cleaned[len("models/") :]
    if cleaned.endswith("-zero"):
        return "MINIMAL"
    if cleaned.endswith("-low"):
        return "LOW"
    if cleaned.endswith("-medium"):
        return "MEDIUM"
    if cleaned.endswith("-high"):
        return "HIGH"
    if cleaned in LONG_DECODE_MODELS or "carbon" in cleaned:
        env_lvl = (
            source_env.get("CC_AGY_PRO_THINKING_LEVEL")
            or source_env.get("GEMINI4_PRO_THINKING_LEVEL")
            or "MEDIUM"
        ).strip().upper()
        if env_lvl in ("MINIMAL", "LOW", "MEDIUM", "HIGH", "EXTRA_HIGH"):
            return env_lvl
        return "MEDIUM"
    return None

# Default Flash worker model (cost_profile economy/balanced and single-shot asks).
DEFAULT_FLASH_MODEL = "gemini-flash-next"
# Step-up model when a Flash turn fails its checks (balanced profile). Carbon with zero thinking
# solved every task Flash solved plus 2 more in the 26-task suite; override with CC_AGY_ESCALATE_MODEL_ON_FAILURE.
DEFAULT_ESCALATION_MODEL = "gemini-4-carbon-zero"


def normalize_wire_model_name(model: str, *, custom_gateway: bool = False) -> str:
    """Return the upstream API model ID for `model` while preserving plugin tier aliases on gateways.

    On direct Google AI Studio and Vertex AI endpoints (`custom_gateway=False`), publisher models
    use canonical IDs such as `gemini-4-carbon` while thinking depth is carried via `thinking_config`
    (`thinking_budget=0` or `thinking_level`). Custom Vertex gateways (`custom_gateway=True`) receive
    the tier-qualified alias verbatim so gateway routers can dispatch per-tier model targets.
    """
    cleaned = str(model or "").strip()
    if not cleaned:
        return cleaned
    low_prefix = cleaned.lower()
    if "/models/" in low_prefix:
        cleaned = cleaned[low_prefix.rfind("/models/") + len("/models/") :]
    elif low_prefix.startswith("models/"):
        cleaned = cleaned[len("models/") :]
    if not cleaned or custom_gateway:
        return cleaned
    low = cleaned.lower()
    if low in (
        "carbon",
        "gemini-4-carbon-zero",
        "gemini-4-carbon-low",
        "gemini-4-carbon-medium",
        "gemini-4-carbon-high",
        "carbon-g-zero",
        "carbon-g",
    ):
        return "gemini-4-carbon"
    if low == "argon":
        return "gemini-4-argon"
    if low.endswith("-zero") and len(cleaned) > 5:
        return cleaned[:-5]
    return cleaned

# Standard Vertex regional locations supported by GA Gemini models.
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

# Per-1M token pricing in USD on Vertex AI is loaded from `server/rate_card.json`
# via `server.rate_card.MODEL_PRICING_PER_1M_USD` (re-exported above for callers).

# Claude Code warns when a single MCP tool response exceeds 10,000 tokens
# and truncates at 25,000 tokens. We keep digests below 8,000 tokens.
MAX_TOOL_RESPONSE_TOKENS: int = 8000
DEFAULT_DIFF_PAGE_TOKENS: int = 4000


def normalize_model_id(model: str | None) -> str:
    """Normalize a Vertex AI model identifier, stripping resource prefixes like publishers/google/models/."""
    raw = (model or "").strip().lower()
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    return raw


def estimate_tokens(text: str) -> int:
    """Estimate token count from text (approx 4 UTF-8 bytes per token, minimum 1 for non-empty)."""
    if not text:
        return 0
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def paginate_text_by_tokens(
    text: str,
    page_tokens: int = DEFAULT_DIFF_PAGE_TOKENS,
) -> list[str]:
    """Split text into UTF-8 character-safe pages where each page is <= page_tokens."""
    if not text:
        return [""]
    budget_tokens = max(64, min(int(page_tokens), MAX_TOOL_RESPONSE_TOKENS))
    max_bytes = budget_tokens * 4
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text]

    pages: list[str] = []
    idx = 0
    total_len = len(encoded)
    while idx < total_len:
        end = min(idx + max_bytes, total_len)
        # Back up if `end` lands in the middle of a multi-byte UTF-8 sequence (10xxxxxx continuation byte)
        while end > idx and end < total_len and (encoded[end] & 0xC0) == 0x80:
            end -= 1
        if end == idx:
            end = min(idx + max_bytes, total_len)
        pages.append(encoded[idx:end].decode("utf-8", errors="replace"))
        idx = end
    return pages or [""]


def truncate_to_token_budget(
    text: str,
    max_tokens: int = MAX_TOOL_RESPONSE_TOKENS,
) -> tuple[str, bool, int]:
    """Truncate text on a valid UTF-8 codepoint boundary so estimated tokens <= max_tokens.

    Returns (truncated_text, was_truncated, original_estimated_tokens).
    """
    orig_tokens = estimate_tokens(text)
    if orig_tokens <= max_tokens:
        return text, False, orig_tokens
    pages = paginate_text_by_tokens(text, page_tokens=max_tokens)
    return pages[0], True, orig_tokens


def resolve_vertex_location(
    model: str,
    requested_location: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    """Resolve a valid Vertex AI location for the given model.

    Returns (resolved_location, reroute_note).
    If a global-only model (such as gemini-3.8-flash or gemini-4-argon)
    is requested against a regional endpoint like asia-south1 or us-central1,
    automatically routes to 'global' and returns an explanatory note.
    """
    source_env = env if env is not None else os.environ
    req_clean = (requested_location or "").strip()
    if req_clean and not req_clean.startswith("${"):
        raw_loc = req_clean
    else:
        raw_loc = first_valid_env(
            source_env,
            "GOOGLE_CLOUD_LOCATION",
            "VERTEX_LOCATION",
            "CLAUDE_PLUGIN_OPTION_GOOGLE_CLOUD_LOCATION",
            "CC_AGY_USERCFG_GOOGLE_CLOUD_LOCATION",
            default="us-central1",
        )
    raw_loc = raw_loc.lower()

    normalized_model = normalize_model_id(model)

    if normalized_model in GLOBAL_ONLY_MODELS:
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


def resolve_default_delegate_model(
    requested_model: str | None = None,
    *,
    env: Any = None,
) -> str:
    """Resolve the worker model from explicit argument, environment/userConfig override, or cost_profile."""
    source = env if env is not None else os.environ
    override = first_valid_env(
        source,
        "CC_AGY_DEFAULT_DELEGATE_MODEL",
        "CLAUDE_PLUGIN_OPTION_DEFAULT_DELEGATE_MODEL",
        "CC_AGY_USERCFG_DEFAULT_DELEGATE_MODEL",
        default="",
    )
    req_clean = (requested_model or "").strip()
    if req_clean and not req_clean.startswith("${"):
        if override and req_clean == "gemini-3.8-flash" and "flash" in override.lower():
            return override
        return req_clean
    if override:
        return override
    profile = resolve_cost_profile(source)
    if profile == "quality":
        return "gemini-4-argon"
    return DEFAULT_FLASH_MODEL


def resolve_default_ask_model(
    requested_model: str | None = None,
    *,
    env: Any = None,
) -> str:
    """Resolve the single-shot ask_gemini model from explicit argument, env/userConfig override, or cost_profile."""
    source = env if env is not None else os.environ
    override = first_valid_env(
        source,
        "CC_AGY_DEFAULT_ASK_MODEL",
        "CLAUDE_PLUGIN_OPTION_DEFAULT_ASK_MODEL",
        "CC_AGY_USERCFG_DEFAULT_ASK_MODEL",
        default="",
    )
    req_clean = (requested_model or "").strip()
    if req_clean and not req_clean.startswith("${"):
        if override and req_clean == "gemini-3.8-flash" and "flash" in override.lower():
            return override
        return req_clean
    if override:
        return override
    profile = resolve_cost_profile(source)
    if profile == "quality":
        return "gemini-4-argon"
    return DEFAULT_FLASH_MODEL


def sanitize_placeholder_env() -> None:
    """Remove unsubstituted Claude Code ${user_config...} placeholders from credential env vars."""
    for var in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "CLAUDE_PLUGIN_OPTION_GEMINI_API_KEY",
        "CC_AGY_USERCFG_GEMINI_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "VERTEX_PROJECT",
        "GCLOUD_PROJECT",
        "CLOUD_ML_PROJECT_ID",
        "CLAUDE_PLUGIN_OPTION_GOOGLE_CLOUD_PROJECT",
        "CC_AGY_USERCFG_GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "VERTEX_LOCATION",
        "CLAUDE_PLUGIN_OPTION_GOOGLE_CLOUD_LOCATION",
        "CC_AGY_USERCFG_GOOGLE_CLOUD_LOCATION",
        "CC_AGY_COST_PROFILE",
        "CLAUDE_PLUGIN_OPTION_COST_PROFILE",
        "CC_AGY_USERCFG_COST_PROFILE",
        "CC_AGY_DEFAULT_DELEGATE_MODEL",
        "CC_AGY_USERCFG_DEFAULT_DELEGATE_MODEL",
        "CC_AGY_DEFAULT_ASK_MODEL",
        "CC_AGY_USERCFG_DEFAULT_ASK_MODEL",
        "CC_AGY_STRICT_MODE",
        "CC_AGY_USERCFG_STRICT_MODE",
        "CC_AGY_VERTEX_BASE_URL",
        "GOOGLE_VERTEX_BASE_URL",
        "CLAUDE_PLUGIN_OPTION_VERTEX_BASE_URL",
        "CC_AGY_USERCFG_VERTEX_BASE_URL",
        "CC_AGY_GEMINI_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
        "CLAUDE_PLUGIN_OPTION_GEMINI_BASE_URL",
        "CC_AGY_USERCFG_GEMINI_BASE_URL",
        "CC_AGY_GATEWAY_WIRE",
        "CC_AGY_GATEWAY_STYLE",
        "CLAUDE_PLUGIN_OPTION_GATEWAY_WIRE",
        "CC_AGY_USERCFG_GATEWAY_WIRE",
        "CC_AGY_VERTEX_BASE_URL_TOKEN",
        "CLAUDE_PLUGIN_OPTION_VERTEX_BASE_URL_TOKEN",
        "CC_AGY_USERCFG_VERTEX_BASE_URL_TOKEN",
        "CC_AGY_BIFROST_VIRTUAL_KEY",
        "CLAUDE_PLUGIN_OPTION_BIFROST_VIRTUAL_KEY",
        "CC_AGY_USERCFG_BIFROST_VIRTUAL_KEY",
        "CC_AGY_VERTEX_BASE_URL_AUDIENCE",
        "CLAUDE_PLUGIN_OPTION_VERTEX_BASE_URL_AUDIENCE",
        "CC_AGY_USERCFG_VERTEX_BASE_URL_AUDIENCE",
        "CC_AGY_VERTEX_OIDC",
        "CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT",
        "CC_AGY_TELEMETRY_SINK",
        "CLAUDE_PLUGIN_OPTION_TELEMETRY_SINK",
        "CC_AGY_USERCFG_TELEMETRY_SINK",
        "CC_AGY_TELEMETRY_TOKEN",
        "CLAUDE_PLUGIN_OPTION_TELEMETRY_TOKEN",
        "CC_AGY_USERCFG_TELEMETRY_TOKEN",
        "CC_AGY_USER_ID",
        "CC_AGY_AUTH_MODE",
        "CLAUDE_PLUGIN_OPTION_AUTH_MODE",
        "CC_AGY_USERCFG_AUTH_MODE",
        "CC_AGY_GEMINI_ENTERPRISE",
        "CC_AGY_CLI_BIN",
        "CLAUDE_PLUGIN_OPTION_AGY_CLI_PATH",
        "CC_AGY_USERCFG_AGY_CLI_PATH",
        "CC_AGY_CLI_CONFIG_DIR",
        "CC_AGY_CLI_INSTALL_DIR",
        "ANTIGRAVITY_OAUTH_TOKEN",
        "CC_AGY_ENTERPRISE_ACCESS_TOKEN",
        "CC_AGY_MAX_DAILY_CENTS",
        "CLAUDE_PLUGIN_OPTION_MAX_DAILY_CENTS",
        "CC_AGY_USERCFG_MAX_DAILY_CENTS",
        "CC_AGY_ALLOWED_PROJECTS",
        "CLAUDE_PLUGIN_OPTION_ALLOWED_PROJECTS",
        "CC_AGY_USERCFG_ALLOWED_PROJECTS",
        "CC_AGY_BLOCKED_PROJECTS",
        "CLAUDE_PLUGIN_OPTION_BLOCKED_PROJECTS",
        "CC_AGY_USERCFG_BLOCKED_PROJECTS",
        "CC_AGY_AUTO_APPLY_ON_PASS",
        "CLAUDE_PLUGIN_OPTION_AUTO_APPLY_ON_PASS",
        "CC_AGY_USERCFG_AUTO_APPLY_ON_PASS",
        "CC_AGY_ALLOW_IN_PLACE",
        "CC_AGY_DATA_DIR",
        "CC_AGY_ENTERPRISE_ADC",
        "CLAUDE_PLUGIN_OPTION_ENTERPRISE_ADC",
        "CC_AGY_USERCFG_ENTERPRISE_ADC",
        "CC_AGY_ENTERPRISE_PROJECT",
        "CLAUDE_PLUGIN_OPTION_ENTERPRISE_PROJECT",
        "CC_AGY_USERCFG_ENTERPRISE_PROJECT",
        "CC_AGY_ENTERPRISE_LOCATION",
        "CLAUDE_PLUGIN_OPTION_ENTERPRISE_LOCATION",
        "CC_AGY_USERCFG_ENTERPRISE_LOCATION",
        "CC_AGY_ENTERPRISE_USER_TIER",
        "CLAUDE_PLUGIN_OPTION_ENTERPRISE_USER_TIER",
        "CC_AGY_USERCFG_ENTERPRISE_USER_TIER",
        "CC_AGY_BIFROST_POLICY_URL",
        "CLAUDE_PLUGIN_OPTION_BIFROST_POLICY_URL",
        "CC_AGY_USERCFG_BIFROST_POLICY_URL",
        "CC_AGY_BIFROST_AGY_VIRTUAL_KEY",
        "CLAUDE_PLUGIN_OPTION_BIFROST_AGY_VIRTUAL_KEY",
        "CC_AGY_USERCFG_BIFROST_AGY_VIRTUAL_KEY",
        "CC_AGY_BIFROST_POLICY_USER",
        "CLAUDE_PLUGIN_OPTION_BIFROST_POLICY_USER",
        "CC_AGY_USERCFG_BIFROST_POLICY_USER",
        "CC_AGY_BIFROST_POLICY_PASSWORD",
        "CLAUDE_PLUGIN_OPTION_BIFROST_POLICY_PASSWORD",
        "CC_AGY_USERCFG_BIFROST_POLICY_PASSWORD",
        "CC_AGY_BIFROST_AGY_POLICY_MODEL",
        "CLAUDE_PLUGIN_OPTION_BIFROST_AGY_POLICY_MODEL",
        "CC_AGY_USERCFG_BIFROST_AGY_POLICY_MODEL",
        "CC_AGY_WORKER_SANDBOX",
        "CLAUDE_PLUGIN_OPTION_WORKER_SANDBOX",
        "CC_AGY_USERCFG_WORKER_SANDBOX",
    ):
        val = os.environ.get(var)
        if val is not None and (
            val.strip().startswith("${")
            or val.strip().lower() in ("<api-key>", "dummy-key", "dummy", "none", "null")
        ):
            os.environ.pop(var, None)


def resolve_api_key(
    requested_key: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> str | None:
    """Resolve a Google AI Studio Gemini API key from arguments or environment."""
    if requested_key is not None and not isinstance(requested_key, str):
        raise ValueError("api_key must be a string")
    source_env = env if env is not None else os.environ
    for candidate in (
        requested_key,
        source_env.get("GEMINI_API_KEY"),
        source_env.get("GOOGLE_API_KEY"),
        source_env.get("CLAUDE_PLUGIN_OPTION_GEMINI_API_KEY"),
        source_env.get("CC_AGY_USERCFG_GEMINI_API_KEY"),
    ):
        if isinstance(candidate, str) and candidate:
            cleaned = candidate.strip()
            if (
                cleaned
                and not cleaned.startswith("${")
                and cleaned.lower() not in ("<api-key>", "dummy-key", "dummy", "none", "null")
            ):
                return cleaned
    return None


GATEWAY_WIRE_CHOICES: tuple[str, ...] = ("auto", "vertex", "gemini")
GATEWAY_WIRE_CLI_CHOICES: tuple[str, ...] = (
    "auto",
    "vertex",
    "gemini",
    "litellm-vertex",
    "litellm-gemini",
    "bifrost",
)
_GATEWAY_WIRE_VERTEX_ALIASES: frozenset[str] = frozenset(
    {"vertex", "vertex_ai", "vertexai", "litellm_vertex"}
)
_GATEWAY_WIRE_GEMINI_ALIASES: frozenset[str] = frozenset(
    {"gemini", "gemini_api", "genai", "bifrost", "litellm_gemini", "ai_studio", "aistudio"}
)


def _clean_base_url_candidate(candidate: Any) -> str | None:
    """Return a stripped gateway URL candidate when non-empty and not an unsubstituted placeholder."""
    if not isinstance(candidate, str) or not candidate:
        return None
    cleaned = candidate.strip()
    if (
        cleaned
        and not cleaned.startswith("${")
        and cleaned.lower() not in ("<base-url>", "dummy", "none", "null")
    ):
        return cleaned
    return None


def resolve_vertex_base_url(
    requested_base_url: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an optional custom Vertex AI / proxy base URL from arguments or environment."""
    source_env = env if env is not None else os.environ
    for candidate in (
        requested_base_url,
        source_env.get("CC_AGY_VERTEX_BASE_URL"),
        source_env.get("GOOGLE_VERTEX_BASE_URL"),
        source_env.get("CLAUDE_PLUGIN_OPTION_VERTEX_BASE_URL"),
        source_env.get("CC_AGY_USERCFG_VERTEX_BASE_URL"),
    ):
        cleaned = _clean_base_url_candidate(candidate)
        if cleaned:
            return cleaned
    return None


_STARTUP_JETSKI_SHIM_GEMINI_URL: str | None = (
    (os.environ.get("GOOGLE_GEMINI_BASE_URL") or "").strip()
    if (os.environ.get("GEMINI_API_KEY") or "").strip().lower() in ("dummy-key", "dummy", "<api-key>")
    else None
)


def _usable_google_gemini_base_url(source_env: Mapping[str, str]) -> str | None:
    """Return `GOOGLE_GEMINI_BASE_URL` while ignoring host-injected `dummy-key` loopback shims."""
    cleaned = _clean_base_url_candidate(source_env.get("GOOGLE_GEMINI_BASE_URL"))
    if not cleaned:
        return None
    raw_gem_key = (source_env.get("GEMINI_API_KEY") or "").strip().lower()
    if raw_gem_key in ("dummy-key", "dummy", "<api-key>"):
        return None
    # Also ignore the host's startup loopback shim if `os.environ` is passed directly without `GEMINI_API_KEY`.
    if source_env is os.environ and _STARTUP_JETSKI_SHIM_GEMINI_URL and cleaned == _STARTUP_JETSKI_SHIM_GEMINI_URL:
        return None
    return cleaned


def resolve_gemini_base_url(
    requested_base_url: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an optional Gemini API-wire gateway URL (for example Bifrost `/genai` or LiteLLM `/gemini`)."""
    source_env = env if env is not None else os.environ
    for candidate in (
        requested_base_url,
        source_env.get("CC_AGY_GEMINI_BASE_URL"),
        _usable_google_gemini_base_url(source_env),
    ):
        cleaned = _clean_base_url_candidate(candidate)
        if cleaned:
            return cleaned
    return None


def resolve_gateway_base_url(
    requested_base_url: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve any configured AI gateway base URL across Vertex-wire and Gemini-wire settings."""
    source_env = env if env is not None else os.environ
    for candidate in (
        requested_base_url,
        source_env.get("CC_AGY_VERTEX_BASE_URL"),
        source_env.get("CC_AGY_GEMINI_BASE_URL"),
        source_env.get("GOOGLE_VERTEX_BASE_URL"),
        _usable_google_gemini_base_url(source_env),
        source_env.get("CLAUDE_PLUGIN_OPTION_VERTEX_BASE_URL"),
        source_env.get("CC_AGY_USERCFG_VERTEX_BASE_URL"),
    ):
        cleaned = _clean_base_url_candidate(candidate)
        if cleaned:
            return cleaned
    return None


def resolve_effective_gateway_url(
    auth_or_req: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return the effective gateway base URL for an auth/request mapping without hijacking direct api_key calls."""
    explicit = _clean_base_url_candidate(auth_or_req.get("base_url"))
    if explicit:
        return explicit
    if "base_url" in auth_or_req or auth_or_req.get("api_key") or auth_or_req.get("auth_mode") == "api_key":
        return ""
    return resolve_gateway_base_url(env=env) or ""


def resolve_bifrost_virtual_key(
    requested_virtual_key: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an optional Bifrost governance virtual key (`x-bf-vk`) from arguments or environment."""
    source_env = env if env is not None else os.environ
    for candidate in (
        requested_virtual_key,
        source_env.get("CC_AGY_BIFROST_VIRTUAL_KEY"),
        source_env.get("CLAUDE_PLUGIN_OPTION_BIFROST_VIRTUAL_KEY"),
        source_env.get("CC_AGY_USERCFG_BIFROST_VIRTUAL_KEY"),
    ):
        if isinstance(candidate, str) and candidate:
            cleaned = candidate.strip()
            if (
                cleaned
                and not cleaned.startswith("${")
                and cleaned.lower() not in ("<virtual-key>", "<token>", "dummy-key", "dummy", "none", "null")
            ):
                return cleaned
    return None


def resolve_gateway_wire(
    requested_wire: str | None = None,
    *,
    base_url: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the HTTP wire contract (`'vertex'` or `'gemini'`) for a custom AI gateway.

    - `'vertex'`: `/v1beta1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent`
      (LiteLLM `/vertex_ai`, Cloud Run Vertex proxies, Apigee Vertex routes).
    - `'gemini'`: `/v1beta/models/{model}:generateContent`
      (Bifrost `/genai`, LiteLLM `/gemini`, `GOOGLE_GEMINI_BASE_URL` / `CC_AGY_GEMINI_BASE_URL`).
    """
    source_env = env if env is not None else os.environ
    for candidate in (
        requested_wire,
        source_env.get("CC_AGY_GATEWAY_WIRE"),
        source_env.get("CC_AGY_GATEWAY_STYLE"),
        source_env.get("CLAUDE_PLUGIN_OPTION_GATEWAY_WIRE"),
        source_env.get("CC_AGY_USERCFG_GATEWAY_WIRE"),
    ):
        if isinstance(candidate, str) and candidate:
            cleaned = candidate.strip()
            if not cleaned or cleaned.startswith("${") or cleaned.lower() in ("none", "null"):
                continue
            normalized = cleaned.lower().replace("-", "_")
            if normalized == "auto":
                break
            if normalized in _GATEWAY_WIRE_VERTEX_ALIASES:
                return "vertex"
            if normalized in _GATEWAY_WIRE_GEMINI_ALIASES:
                return "gemini"
            raise ValueError(
                f"Unsupported gateway wire mode {cleaned!r}; expected one of {GATEWAY_WIRE_CLI_CHOICES}."
            )

    explicit_base = _clean_base_url_candidate(base_url)
    eff_base = explicit_base or resolve_gateway_base_url(env=source_env) or ""
    if eff_base:
        parsed_path = urllib.parse.urlsplit(eff_base).path.rstrip("/").lower()
        segments = [seg for seg in parsed_path.split("/") if seg]
        for seg in reversed(segments):
            if seg in ("genai", "gemini"):
                return "gemini"
            if seg in ("vertex_ai", "vertex", "v1internal"):
                return "vertex"

    gemini_base = resolve_gemini_base_url(env=source_env)
    vertex_base = resolve_vertex_base_url(env=source_env)
    if gemini_base and (
        (explicit_base and explicit_base == gemini_base)
        or (not explicit_base and not vertex_base)
    ):
        return "gemini"
    if resolve_bifrost_virtual_key(env=source_env) and not vertex_base:
        return "gemini"
    return "vertex"


_OIDC_CACHE_LOCK = threading.Lock()
_OIDC_TOKEN_CACHE: dict[tuple[str, ...], tuple[str, float, float]] = {}
_OIDC_KEY_LOCKS: dict[tuple[str, ...], threading.Lock] = {}
_OIDC_SOURCE_CHOICES: tuple[str, ...] = ("auto", "gac", "gcloud", "metadata")
_ALLOWED_AUDIENCE_PREFIXES: tuple[str, ...] = (
    "https://",
    "http://localhost",
    "http://127.0.0.1",
    "http://[::1]",
    "iap://",
    "api://",
)
_AUDIENCE_METACHAR_RE = re.compile(r"[\s\x00-\x1f\x7f&|;<>`$\"'%^!()]")


class _NoMetadataRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        raise urllib.error.HTTPError(req.full_url, code, f"Metadata server redirect rejected ({code})", headers, fp)


def clear_oidc_token_cache() -> None:
    """Clear cached OIDC identity tokens (used by tests and credential rotation)."""
    with _OIDC_CACHE_LOCK:
        _OIDC_TOKEN_CACHE.clear()


def _validate_oidc_audience(audience: str) -> str:
    cleaned = (audience or "").strip()
    if not cleaned:
        raise ValueError("OIDC audience must be a non-empty URL or URI")
    if cleaned.startswith("-"):
        raise ValueError("Invalid OIDC audience: must not start with '-'")
    authority_source = cleaned.split("://", 1)[1] if "://" in cleaned else cleaned
    authority_part = authority_source.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if "@" in authority_part:
        raise ValueError("Invalid OIDC audience: URL must not embed userinfo credentials")
    if _AUDIENCE_METACHAR_RE.search(cleaned):
        raise ValueError(
            "Invalid OIDC audience: must not contain whitespace, control characters, or shell metacharacters"
        )
    low = cleaned.lower()
    if not low.startswith(_ALLOWED_AUDIENCE_PREFIXES):
        raise ValueError(
            f"Unsupported OIDC audience {cleaned!r}: expected 'https://...' (or 'iap://...', 'api://...', or loopback 'http://localhost')"
        )
    parsed_aud = urllib.parse.urlsplit(cleaned)
    if parsed_aud.username is not None or parsed_aud.password is not None or "@" in (parsed_aud.netloc or ""):
        raise ValueError("Invalid OIDC audience: URL must not embed userinfo credentials")
    return cleaned


def resolve_vertex_base_url_audience(
    requested_audience: str | None = None,
    *,
    base_url: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an optional OIDC target audience for zero-key Cloud Run / WIF gateway auth."""
    source_env = env if env is not None else os.environ
    raw_aud: str | None = None
    for candidate in (
        requested_audience,
        source_env.get("CC_AGY_VERTEX_BASE_URL_AUDIENCE"),
        source_env.get("CLAUDE_PLUGIN_OPTION_VERTEX_BASE_URL_AUDIENCE"),
        source_env.get("CC_AGY_USERCFG_VERTEX_BASE_URL_AUDIENCE"),
    ):
        if isinstance(candidate, str) and candidate:
            cleaned = candidate.strip()
            if (
                cleaned
                and not cleaned.startswith("${")
                and cleaned.lower() not in ("<audience>", "dummy", "none", "null", "off", "false", "0")
            ):
                raw_aud = cleaned
                break

    if raw_aud is None and env_flag("CC_AGY_VERTEX_OIDC", source_env) is True:
        raw_aud = "auto"

    if raw_aud is None:
        return None

    if raw_aud.lower() in ("auto", "true", "1", "yes", "on"):
        eff_base = resolve_gateway_base_url(base_url, env=source_env)
        if not eff_base:
            return None
        base_authority_src = eff_base.split("://", 1)[1] if "://" in eff_base else eff_base
        if "@" in base_authority_src.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]:
            raise ValueError("CC_AGY_VERTEX_BASE_URL must not embed userinfo credentials when deriving OIDC audience")
        parsed = urllib.parse.urlsplit(eff_base)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Cannot derive OIDC audience from base_url {eff_base!r}")
        if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
            raise ValueError("CC_AGY_VERTEX_BASE_URL must not embed userinfo credentials when deriving OIDC audience")
        raw_aud = f"{parsed.scheme}://{parsed.netloc}"

    return _validate_oidc_audience(raw_aud)


def _resolve_oidc_source(source_env: Mapping[str, str]) -> str:
    raw = str(source_env.get("CC_AGY_VERTEX_OIDC_SOURCE") or "").strip().lower()
    if not raw:
        return "auto"
    if raw not in _OIDC_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported CC_AGY_VERTEX_OIDC_SOURCE={raw!r}; expected one of {_OIDC_SOURCE_CHOICES}."
        )
    return raw


def _is_managed_workload_env(source_env: Mapping[str, str]) -> bool:
    return any(
        bool(str(source_env.get(k) or "").strip())
        for k in ("K_SERVICE", "CLOUD_RUN_JOB", "CLOUD_RUN_EXECUTION")
    )


def _oidc_cache_key(validated_aud: str, source_env: Mapping[str, str]) -> tuple[str, ...]:
    return (
        validated_aud,
        str(source_env.get("CLOUDSDK_CONFIG") or ""),
        first_valid_env(source_env, "GOOGLE_APPLICATION_CREDENTIALS", default=""),
        first_valid_env(source_env, "CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT", default=""),
        first_valid_env(source_env, "CLOUDSDK_CORE_ACCOUNT", default=""),
        _resolve_oidc_source(source_env),
        "workload" if _is_managed_workload_env(source_env) else "dev",
    )


def peek_cached_oidc_token(
    *,
    env: Mapping[str, str] | None = None,
    audience: str | None = None,
    base_url: str | None = None,
) -> str | None:
    """Return a non-expired cached OIDC identity token for `env` without blocking or minting."""
    source_env = env if env is not None else os.environ
    try:
        resolved_aud = resolve_vertex_base_url_audience(audience, base_url=base_url, env=source_env)
        cache_key = _oidc_cache_key(resolved_aud, source_env) if resolved_aud else None
    except ValueError:
        return None
    if not cache_key:
        return None
    now = time.time()
    with _OIDC_CACHE_LOCK:
        entry = _OIDC_TOKEN_CACHE.get(cache_key)
        if entry is not None:
            cached_tok, refresh_at, _minted_ttl = entry
            if now < refresh_at:
                return cached_tok
    return None


def _decode_jwt_segment_dict(segment: str) -> dict[str, Any] | None:
    if not segment or not re.fullmatch(r"[A-Za-z0-9_-]+", segment):
        return None
    padded = segment + ("=" * ((4 - (len(segment) % 4)) % 4))
    try:
        obj = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


def _jwt_exp_epoch(token: str) -> float | None:
    parts = token.split(".")
    if len(parts) != 3 or not parts[1]:
        return None
    payload = _decode_jwt_segment_dict(parts[1])
    if payload is None:
        return None
    exp = payload.get("exp")
    if isinstance(exp, bool) or not isinstance(exp, (int, float)):
        return None
    exp_f = float(exp)
    return exp_f if math.isfinite(exp_f) else None


def _jwt_email_claim(token: str) -> str:
    parts = token.split(".")
    if len(parts) != 3 or not parts[1]:
        return ""
    payload = _decode_jwt_segment_dict(parts[1])
    if payload is None:
        return ""
    email = payload.get("email")
    return email.strip() if isinstance(email, str) else ""


def _is_valid_oidc_jwt(
    token: str,
    *,
    expected_audience: str | None = None,
    require_audience: bool = False,
) -> bool:
    if not token or any(ch.isspace() for ch in token):
        return False
    parts = token.split(".")
    if len(parts) != 3 or not all(parts):
        return False
    hdr = _decode_jwt_segment_dict(parts[0])
    payload = _decode_jwt_segment_dict(parts[1])
    if hdr is None or payload is None:
        return False
    alg = str(hdr.get("alg") or "").strip()
    if not alg or alg.lower() == "none":
        return False
    if _jwt_exp_epoch(token) is None:
        return False
    aud = payload.get("aud")
    has_nonempty_aud = False
    if isinstance(aud, str):
        has_nonempty_aud = bool(aud.strip())
    elif isinstance(aud, list):
        has_nonempty_aud = any(isinstance(item, str) and bool(item.strip()) for item in aud)
    has_identity_claim = has_nonempty_aud or any(
        isinstance(payload.get(k), str) and bool(str(payload.get(k)).strip())
        for k in ("iss", "sub", "email")
    )
    if not has_identity_claim:
        return False
    if expected_audience:
        if "aud" in payload and aud is not None:
            if isinstance(aud, str):
                return aud.strip() == expected_audience
            if isinstance(aud, list):
                return any(isinstance(item, str) and item.strip() == expected_audience for item in aud)
            return False
        if require_audience:
            return False
    elif require_audience and not has_nonempty_aud:
        return False
    return True


def _jwt_refresh_deadline(token: str, now: float) -> float:
    exp_f = _jwt_exp_epoch(token)
    if exp_f is not None:
        remaining = exp_f - now
        if remaining <= 60.0:
            return now
        if remaining < 600.0:
            return now + (remaining * 0.5)
        return min(now + 3300.0, exp_f - 300.0)
    return now + 1800.0


def _is_cached_oidc_fresh(
    cached_tok: str,
    refresh_at: float,
    minted_ttl: float,
    now: float,
    min_remaining_seconds: float,
) -> bool:
    if now >= refresh_at:
        return False
    if min_remaining_seconds > 0.0:
        exp_f = _jwt_exp_epoch(cached_tok)
        if exp_f is not None:
            eff_min_rem = (
                min(min_remaining_seconds, max(0.0, minted_ttl - 60.0))
                if minted_ttl > 0.0
                else min_remaining_seconds
            )
            return (exp_f - now) >= eff_min_rem
        return (refresh_at - now) >= min_remaining_seconds
    return True


def _fetch_metadata_identity_token(audience: str, *, timeout: float = 3.0) -> str:
    query = urllib.parse.urlencode({"audience": audience, "format": "full"})
    meta_url = f"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?{query}"
    req = urllib.request.Request(
        meta_url,
        headers={"Metadata-Flavor": "Google"},
        method="GET",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoMetadataRedirectHandler(),
    )
    with opener.open(req, timeout=timeout) as resp:
        token = resp.read(16_384).decode("utf-8", errors="replace").strip()
    if not _is_valid_oidc_jwt(token, expected_audience=audience):
        raise RuntimeError("Metadata server returned an empty or malformed OIDC identity token")
    return token


def _extract_impersonated_principal(impersonation_url: str) -> str:
    match = re.search(r"/serviceAccounts/([^/:?]+):generateAccessToken", impersonation_url or "")
    return match.group(1).strip() if match else ""


def _peek_credentials_file_type(gac_path: str) -> str:
    try:
        gac_file = Path(gac_path).expanduser()
        if not gac_file.is_file():
            return ""
        loaded = json.loads(gac_file.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return str(loaded.get("type") or "").strip().lower()
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _mint_from_credentials_file(
    gac_path: str,
    validated_aud: str,
    *,
    impersonate_sa: str = "",
) -> str:
    """Mint an OIDC ID token directly from a service-account, WIF, or impersonated JSON file."""
    from google.auth import aws, identity_pool, impersonated_credentials, pluggable
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2 import credentials as oauth2_credentials
    from google.oauth2 import service_account

    cloud_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    user_oidc_scopes = [
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/cloud-platform",
    ]
    gac_file = Path(gac_path).expanduser()
    if not gac_file.is_file():
        raise RuntimeError(f"Credentials file {gac_path!r} does not exist or is not a regular file")
    try:
        loaded = json.loads(gac_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Failed to read credentials JSON {gac_path!r}: {exc}") from exc
    if not isinstance(loaded, dict) or not loaded:
        raise RuntimeError(f"Credentials file {gac_path!r} is not a non-empty JSON object")

    raw_json = loaded
    cred_type = str(raw_json.get("type") or "").strip().lower()
    req = GoogleAuthRequest()

    if cred_type == "service_account" and not impersonate_sa:
        sa_id_creds = service_account.IDTokenCredentials.from_service_account_info(
            raw_json,
            target_audience=validated_aud,
        )
        sa_id_creds.refresh(req)
        tok = str(getattr(sa_id_creds, "token", "") or "").strip()
        if _is_valid_oidc_jwt(tok, expected_audience=validated_aud):
            return tok
        raise RuntimeError(f"Service account file {gac_path!r} did not return a valid OIDC JWT for {validated_aud!r}")

    if cred_type not in (
        "service_account",
        "external_account",
        "impersonated_service_account",
        "authorized_user",
    ):
        raise RuntimeError(
            f"Unsupported credentials file type {cred_type!r} in {gac_path!r} for OIDC identity token minting"
        )

    target_sa = impersonate_sa.strip()
    imp_creds = None
    base_creds = None
    if cred_type == "impersonated_service_account":
        imp_creds = impersonated_credentials.Credentials.from_impersonated_service_account_info(
            raw_json,
            scopes=cloud_scopes,
        )
        if target_sa:
            imp_creds = impersonated_credentials.Credentials(
                source_credentials=imp_creds,
                target_principal=target_sa,
                target_scopes=cloud_scopes,
            )
    else:
        if cred_type == "service_account":
            base_creds = service_account.Credentials.from_service_account_info(
                raw_json,
                scopes=cloud_scopes,
            )
        elif cred_type == "authorized_user":
            base_creds = oauth2_credentials.Credentials.from_authorized_user_info(
                raw_json,
                scopes=user_oidc_scopes,
            )
        elif cred_type == "external_account":
            if not target_sa:
                target_sa = _extract_impersonated_principal(
                    str(raw_json.get("service_account_impersonation_url") or "")
                )
            if not target_sa:
                raise RuntimeError(
                    "Workload Identity Federation (external_account) OIDC minting requires "
                    "'service_account_impersonation_url' in the credentials JSON or "
                    "CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT."
                )
            base_info = {k: v for k, v in raw_json.items() if k != "service_account_impersonation_url"}
            cred_src = base_info.get("credential_source")
            if isinstance(cred_src, dict) and "environment_id" in cred_src:
                base_creds = aws.Credentials.from_info(base_info, scopes=cloud_scopes)
            elif isinstance(cred_src, dict) and "executable" in cred_src:
                base_creds = pluggable.Credentials.from_info(base_info, scopes=cloud_scopes)
            else:
                base_creds = identity_pool.Credentials.from_info(base_info, scopes=cloud_scopes)

        if base_creds is not None and target_sa:
            imp_creds = impersonated_credentials.Credentials(
                source_credentials=base_creds,
                target_principal=target_sa,
                target_scopes=cloud_scopes,
            )

    if imp_creds is not None:
        id_creds = impersonated_credentials.IDTokenCredentials(
            imp_creds,
            target_audience=validated_aud,
            include_email=True,
        )
        id_creds.refresh(req)
        tok = str(getattr(id_creds, "token", "") or "").strip()
        if _is_valid_oidc_jwt(tok, expected_audience=validated_aud):
            if impersonate_sa.strip() and _jwt_email_claim(tok).lower() != impersonate_sa.strip().lower():
                raise RuntimeError(
                    f"Impersonated credentials returned principal {_jwt_email_claim(tok) or 'unknown'!r} "
                    f"instead of CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT={impersonate_sa!r}"
                )
            return tok
        raise RuntimeError(
            f"Impersonated credentials for {target_sa!r} did not return a valid OIDC JWT for {validated_aud!r}"
        )

    if base_creds is not None:
        base_creds.refresh(req)
        direct_id_tok = str(getattr(base_creds, "id_token", None) or "").strip()
        if direct_id_tok and _is_valid_oidc_jwt(
            direct_id_tok,
            expected_audience=validated_aud,
            require_audience=True,
        ):
            return direct_id_tok

    raise RuntimeError(
        f"Credentials file {gac_path!r} ({cred_type}) did not yield a signed OIDC identity token "
        f"matching audience {validated_aud!r}; set CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT to mint an "
        "audience-scoped service account ID token."
    )


def _mint_oidc_identity_token(
    audience: str,
    *,
    env: Mapping[str, str] | None = None,
    force_refresh: bool = False,
    min_remaining_seconds: float = 0.0,
) -> str:
    """Mint and cache a Google OIDC identity token for `audience` via WIF/SA, `gcloud`, or GCE metadata."""
    validated_aud = _validate_oidc_audience(audience)
    source_env = env if env is not None else os.environ
    oidc_source = _resolve_oidc_source(source_env)
    cache_key = _oidc_cache_key(validated_aud, source_env)
    min_rem = max(0.0, float(min_remaining_seconds or 0.0))

    with _OIDC_CACHE_LOCK:
        if not force_refresh and cache_key in _OIDC_TOKEN_CACHE:
            cached_tok, refresh_at, minted_ttl = _OIDC_TOKEN_CACHE[cache_key]
            if _is_cached_oidc_fresh(cached_tok, refresh_at, minted_ttl, time.time(), min_rem):
                return cached_tok
        key_lock = _OIDC_KEY_LOCKS.setdefault(cache_key, threading.Lock())

    with key_lock:
        with _OIDC_CACHE_LOCK:
            if not force_refresh and cache_key in _OIDC_TOKEN_CACHE:
                cached_tok, refresh_at, minted_ttl = _OIDC_TOKEN_CACHE[cache_key]
                if _is_cached_oidc_fresh(cached_tok, refresh_at, minted_ttl, time.time(), min_rem):
                    return cached_tok

        errors: list[str] = []
        minted: str | None = None
        gac_path = cache_key[2]
        impersonate_sa = cache_key[3]
        workload_env = cache_key[6] == "workload"

        if oidc_source in ("auto", "gac"):
            if gac_path:
                gac_type = _peek_credentials_file_type(gac_path)
                try:
                    minted = _mint_from_credentials_file(
                        gac_path,
                        validated_aud,
                        impersonate_sa=impersonate_sa,
                    )
                except RuntimeError as exc:
                    if oidc_source == "auto" and gac_type == "authorized_user" and not impersonate_sa:
                        errors.append(f"GOOGLE_APPLICATION_CREDENTIALS ({gac_type}): {exc}")
                    else:
                        raise
            elif oidc_source == "gac":
                raise RuntimeError("CC_AGY_VERTEX_OIDC_SOURCE='gac' requires GOOGLE_APPLICATION_CREDENTIALS.")

        user_gcloud_fallback_args: tuple[str, dict[str, str]] | None = None
        human_gcloud_session_broken = False
        impersonation_gcloud_failed = False
        expected_sa = impersonate_sa.strip().lower()
        if minted is None and oidc_source in ("auto", "gcloud"):
            path_val = source_env.get("PATH") or os.environ.get("PATH")
            gcloud_bin = shutil.which("gcloud", path=path_val) or shutil.which("gcloud.cmd", path=path_val)
            if gcloud_bin:
                sub_env = dict(os.environ)
                for k, v in source_env.items():
                    if isinstance(v, str):
                        sub_env[k] = v
                try:
                    cmd = [gcloud_bin, "auth", "print-identity-token", f"--audiences={validated_aud}", "--quiet"]
                    if impersonate_sa:
                        cmd.extend([f"--impersonate-service-account={impersonate_sa}", "--include-email"])
                    proc = subprocess.run(
                        cmd,
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=15,
                        check=False,
                        env=sub_env,
                    )
                    out = (proc.stdout or "").strip()
                    if proc.returncode == 0 and _is_valid_oidc_jwt(out, expected_audience=validated_aud):
                        tok_email = _jwt_email_claim(out).lower()
                        if expected_sa and tok_email != expected_sa:
                            impersonation_gcloud_failed = True
                            errors.append(
                                f"gcloud returned principal '{tok_email or 'unknown'}' instead of CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT='{impersonate_sa}'"
                            )
                        else:
                            minted = out
                    elif proc.returncode == 0:
                        if impersonate_sa:
                            impersonation_gcloud_failed = True
                        errors.append("gcloud returned a non-JWT or audience-mismatched OIDC token")
                    else:
                        stderr_text = (proc.stderr or proc.stdout or "").strip()
                        low_err = stderr_text.lower()
                        if impersonate_sa:
                            impersonation_gcloud_failed = True
                            errors.append(f"gcloud (exit {proc.returncode}): {stderr_text[-240:]}")
                        elif (
                            "invalid account type" in low_err
                            or "wrongaccounttypeerror" in low_err
                            or "requires valid service account" in low_err
                        ):
                            user_gcloud_fallback_args = (gcloud_bin, sub_env)
                        else:
                            # Google Cloud SDK appends 'Please run: $ gcloud auth login' to base
                            # AuthenticationException even when no account is configured or when a
                            # scopeless GCE VM fails metadata token lookup; exclude those non-human
                            # headless errors so GCE/GKE VMs still reach metadata identity tokens.
                            is_non_human_gcloud_error = any(
                                marker in low_err
                                for marker in (
                                    "you do not currently have an active account selected",
                                    "no credentialed accounts",
                                    "metadata.google.internal",
                                    "computemetadata",
                                    "compute engine metadata service",
                                    "no service account scopes",
                                )
                            )
                            if not is_non_human_gcloud_error and any(
                                marker in low_err
                                for marker in (
                                    "reauth",
                                    "invalid_grant",
                                    "expired or revoked",
                                    "refresherror",
                                    "problem refreshing auth tokens",
                                    "problem refreshing your current auth tokens",
                                )
                            ):
                                human_gcloud_session_broken = True
                            errors.append(f"gcloud (exit {proc.returncode}): {stderr_text[-240:]}")
                except subprocess.TimeoutExpired as exc:
                    if impersonate_sa:
                        impersonation_gcloud_failed = True
                    else:
                        human_gcloud_session_broken = True
                    errors.append(f"gcloud: TimeoutExpired: {exc}")
                except Exception as exc:  # noqa: BLE001
                    if impersonate_sa:
                        impersonation_gcloud_failed = True
                    errors.append(f"gcloud: {type(exc).__name__}: {exc}")
            else:
                errors.append("gcloud CLI not found on PATH")

        def _try_user_gcloud() -> str | None:
            nonlocal human_gcloud_session_broken
            if user_gcloud_fallback_args is None:
                return None
            gcloud_bin_fb, sub_env_fb = user_gcloud_fallback_args
            try:
                fallback_proc = subprocess.run(
                    [gcloud_bin_fb, "auth", "print-identity-token", "--quiet"],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                    env=sub_env_fb,
                )
                fb_out = (fallback_proc.stdout or "").strip()
                if fallback_proc.returncode == 0 and _is_valid_oidc_jwt(fb_out):
                    return fb_out
                human_gcloud_session_broken = True
                if fallback_proc.returncode == 0:
                    errors.append("gcloud user fallback returned a non-JWT OIDC token")
                else:
                    fb_err = (fallback_proc.stderr or fallback_proc.stdout or "").strip()[-240:]
                    errors.append(f"gcloud user fallback (exit {fallback_proc.returncode}): {fb_err}")
            except Exception as exc:  # noqa: BLE001
                human_gcloud_session_broken = True
                errors.append(f"gcloud user fallback: {type(exc).__name__}: {exc}")
            return None

        def _try_metadata() -> str | None:
            try:
                meta_out = _fetch_metadata_identity_token(validated_aud)
                if _is_valid_oidc_jwt(meta_out, expected_audience=validated_aud):
                    if expected_sa:
                        meta_email = _jwt_email_claim(meta_out).lower()
                        if meta_email != expected_sa:
                            errors.append(
                                f"metadata server principal '{meta_email or 'unknown'}' does not match "
                                f"CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT='{impersonate_sa}'"
                            )
                            return None
                    return meta_out
                errors.append("metadata server returned a non-JWT or audience-mismatched OIDC token")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"metadata server: {type(exc).__name__}: {exc}")
            return None

        if minted is None:
            if oidc_source == "metadata":
                minted = _try_metadata()
            elif oidc_source == "gcloud":
                minted = _try_user_gcloud()
            elif oidc_source == "auto":
                if workload_env:
                    # Managed Cloud Run container runtimes use the workload service account first,
                    # unless an explicit gcloud SA impersonation call just failed.
                    if not impersonation_gcloud_failed:
                        minted = _try_metadata()
                    if minted is None:
                        minted = _try_user_gcloud()
                else:
                    # Interactive developer environments preserve an active human gcloud login and never
                    # silently swap an expired/broken human or impersonation session onto a VM service account.
                    minted = _try_user_gcloud()
                    if (
                        minted is None
                        and not human_gcloud_session_broken
                        and not impersonation_gcloud_failed
                    ):
                        minted = _try_metadata()

        if not minted:
            detail = "; ".join(errors) if errors else "no credential provider succeeded"
            raise RuntimeError(
                f"Failed to mint OIDC identity token for gateway audience '{validated_aud}' ({detail}). "
                "Run `gcloud auth login`, set `CC_AGY_VERTEX_OIDC_SOURCE=metadata`, set "
                "`CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT`, configure Workload Identity Federation "
                "(`GOOGLE_APPLICATION_CREDENTIALS`), or export `CC_AGY_VERTEX_BASE_URL_TOKEN`."
            )

        now = time.time()
        exp_f = _jwt_exp_epoch(minted)
        minted_ttl = max(0.0, exp_f - now) if exp_f is not None else 1800.0
        refresh_at = _jwt_refresh_deadline(minted, now)
        if refresh_at > now:
            with _OIDC_CACHE_LOCK:
                _OIDC_TOKEN_CACHE[cache_key] = (minted, refresh_at, minted_ttl)
        return minted


def resolve_static_vertex_base_url_token(
    requested_token: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an explicitly configured static bearer/virtual-key token without minting OIDC."""
    source_env = env if env is not None else os.environ
    for candidate in (
        requested_token,
        source_env.get("CC_AGY_VERTEX_BASE_URL_TOKEN"),
        source_env.get("CLAUDE_PLUGIN_OPTION_VERTEX_BASE_URL_TOKEN"),
        source_env.get("CC_AGY_USERCFG_VERTEX_BASE_URL_TOKEN"),
    ):
        if isinstance(candidate, str) and candidate:
            cleaned = candidate.strip()
            if (
                cleaned
                and not cleaned.startswith("${")
                and cleaned.lower() not in ("<token>", "dummy-key", "dummy", "none", "null")
            ):
                return cleaned
    return None


def resolve_vertex_base_url_token(
    requested_token: str | None = None,
    *,
    base_url: str | None = None,
    requested_audience: str | None = None,
    env: Mapping[str, str] | None = None,
    min_remaining_seconds: float = 0.0,
) -> str | None:
    """Resolve a static bearer/virtual-key token or mint a cached OIDC identity token for an AI gateway."""
    source_env = env if env is not None else os.environ
    static_tok = resolve_static_vertex_base_url_token(requested_token, env=source_env)
    if static_tok:
        return static_tok

    audience = resolve_vertex_base_url_audience(
        requested_audience,
        base_url=base_url,
        env=source_env,
    )
    if audience:
        return _mint_oidc_identity_token(
            audience,
            env=source_env,
            min_remaining_seconds=min_remaining_seconds,
        )
    return None


def resolve_gateway_credentials(
    auth_or_req: Mapping[str, Any],
    *,
    base_url: str,
    wire: str,
    zero_thinking: bool = False,
    min_remaining_seconds: float = 0.0,
    env: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Resolve `(effective_token, http_headers)` for Vertex-wire and Gemini-wire AI gateways.

    Supports static gateway bearer/virtual keys (`CC_AGY_VERTEX_BASE_URL_TOKEN`), refreshing
    Cloud Run / WIF OIDC tokens (`CC_AGY_VERTEX_BASE_URL_AUDIENCE`), Bifrost governance virtual
    keys (`CC_AGY_BIFROST_VIRTUAL_KEY` -> `x-bf-vk`, composable alongside Cloud Run OIDC), and
    `api_key` credentials when `GOOGLE_GEMINI_BASE_URL` / `CC_AGY_GEMINI_BASE_URL` is paired with
    `GEMINI_API_KEY` in `auto` mode without a gateway token or virtual key, while never letting an
    ambient personal `GEMINI_API_KEY` outrank a configured gateway credential or leak into
    `auth_mode='gateway'` / `'vertex'`.
    """
    source_env = env if env is not None else os.environ
    bearer_or_oidc = resolve_vertex_base_url_token(
        auth_or_req.get("base_url_token"),
        base_url=base_url,
        requested_audience=auth_or_req.get("base_url_audience"),
        env=source_env,
        min_remaining_seconds=min_remaining_seconds,
    )
    bf_vk = resolve_bifrost_virtual_key(
        auth_or_req.get("bifrost_virtual_key"),
        env=source_env,
    )
    explicit_api_key = _clean_base_url_candidate(auth_or_req.get("api_key"))
    if explicit_api_key and explicit_api_key.lower() in ("<api-key>", "dummy-key"):
        explicit_api_key = None

    non_jwt_bearer = (
        bearer_or_oidc
        if (bearer_or_oidc and not bearer_or_oidc.startswith("eyJ"))
        else None
    )
    auth_bearer = bearer_or_oidc or bf_vk or explicit_api_key or "custom-endpoint"
    gemini_key = non_jwt_bearer or bf_vk or explicit_api_key
    effective_token = (gemini_key or "custom-endpoint") if wire == "gemini" else auth_bearer

    headers: dict[str, str] = {"Authorization": f"Bearer {auth_bearer}"}
    if wire == "gemini" and gemini_key:
        headers["x-goog-api-key"] = gemini_key
    elif wire == "vertex" and non_jwt_bearer:
        # LiteLLM `/vertex_ai` pass-through and API-key gateways accept `x-goog-api-key` or `Authorization`.
        headers["x-goog-api-key"] = non_jwt_bearer
    if bf_vk:
        headers["x-bf-vk"] = bf_vk
    if zero_thinking:
        headers["X-CC-AGY-Thinking-Budget"] = "0"
    return effective_token, headers


def resolve_max_daily_cents(
    env: Mapping[str, str] | None = None,
) -> float | None:
    """Resolve an optional workspace daily spend ceiling in cents (> 0), or None when unset."""
    import math

    source_env = env if env is not None else os.environ
    for candidate in (
        source_env.get("CC_AGY_MAX_DAILY_CENTS"),
        source_env.get("CLAUDE_PLUGIN_OPTION_MAX_DAILY_CENTS"),
        source_env.get("CC_AGY_USERCFG_MAX_DAILY_CENTS"),
    ):
        if candidate is None:
            continue
        cleaned = str(candidate).strip()
        if not cleaned or cleaned.startswith("${"):
            continue
        if cleaned.lower() in ("none", "null", "off", "0", "0.0"):
            return None
        try:
            parsed = float(cleaned)
        except ValueError as exc:
            raise ValueError(f"CC_AGY_MAX_DAILY_CENTS must be a positive number of cents (got {cleaned!r})") from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"CC_AGY_MAX_DAILY_CENTS must be a finite positive number of cents (got {cleaned!r})")
        return round(parsed, 6)
    return None


def _parse_project_list(raw_values: tuple[Any, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    parsed: list[str] = []
    for raw in raw_values:
        if not isinstance(raw, str):
            continue
        cleaned = raw.strip()
        if not cleaned or cleaned.startswith("${") or cleaned.lower() in ("none", "null", "off"):
            continue
        for token in cleaned.replace(",", " ").split():
            proj = token.strip()
            if not proj or proj.startswith("${") or proj.lower() in ("none", "null", "off", "custom-endpoint"):
                continue
            low = proj.lower()
            if low not in seen:
                seen.add(low)
                parsed.append(proj)
    return tuple(parsed)


def resolve_allowed_projects(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Resolve optional comma/space-separated GCP project allowlist (`CC_AGY_ALLOWED_PROJECTS`)."""
    source_env = env if env is not None else os.environ
    return _parse_project_list(
        (
            source_env.get("CC_AGY_ALLOWED_PROJECTS"),
            source_env.get("CLAUDE_PLUGIN_OPTION_ALLOWED_PROJECTS"),
            source_env.get("CC_AGY_USERCFG_ALLOWED_PROJECTS"),
        )
    )


def resolve_blocked_projects(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Resolve optional comma/space-separated GCP project blocklist (`CC_AGY_BLOCKED_PROJECTS`)."""
    source_env = env if env is not None else os.environ
    return _parse_project_list(
        (
            source_env.get("CC_AGY_BLOCKED_PROJECTS"),
            source_env.get("CLAUDE_PLUGIN_OPTION_BLOCKED_PROJECTS"),
            source_env.get("CC_AGY_USERCFG_BLOCKED_PROJECTS"),
        )
    )


def resolve_auto_apply_override(env: Mapping[str, str] | None = None) -> bool | None:
    """Return explicit user override for auto-apply (`True`, `False`, or `None` when unset)."""
    source_env = env if env is not None else os.environ
    for key in (
        "CC_AGY_AUTO_APPLY_ON_PASS",
        "CLAUDE_PLUGIN_OPTION_AUTO_APPLY_ON_PASS",
        "CC_AGY_USERCFG_AUTO_APPLY_ON_PASS",
    ):
        val = env_flag(key, source_env)
        if val is not None:
            return val
    return None


def resolve_auto_apply_on_pass(env: Mapping[str, str] | None = None) -> bool:
    """Return effective default auto-apply behavior (`CC_AGY_AUTO_APPLY_ON_PASS` or strict-mode default)."""
    override = resolve_auto_apply_override(env)
    if override is not None:
        return override
    return is_strict_mode_enabled(env)


def is_in_place_allowed(env: Mapping[str, str] | None = None) -> bool:
    """Return whether `in_place=True` direct workspace mutation is permitted by user configuration."""
    source_env = env if env is not None else os.environ
    explicit_in_place = env_flag("CC_AGY_ALLOW_IN_PLACE", source_env)
    if explicit_in_place is not None:
        return explicit_in_place
    if resolve_auto_apply_override(source_env) is False:
        return False
    return True


def read_adc_quota_project(
    source_env: Mapping[str, str],
    *,
    explicit_env: Mapping[str, str] | None = None,
) -> str | None:
    """Read non-secret `quota_project_id` from Application Default Credentials JSON when present."""
    return read_adc_metadata(
        source_env, explicit_env=explicit_env, require_quota_project=True
    ).get("quota_project_id")


def read_adc_metadata(
    source_env: Mapping[str, str],
    *,
    explicit_env: Mapping[str, str] | None = None,
    require_quota_project: bool = False,
) -> dict[str, str | None]:
    """Read the non-secret `type`, `quota_project_id`, and path of the first readable ADC JSON file.

    Never returns `client_secret`, `refresh_token`, or `private_key`. All three values are
    `None` when no Application Default Credentials file is found. With
    `require_quota_project`, files without a `quota_project_id` are skipped.
    """
    import json

    empty: dict[str, str | None] = {"type": None, "quota_project_id": None, "path": None}
    candidates: list[Path] = []
    explicit_adc = (source_env.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if explicit_adc and not explicit_adc.startswith("${"):
        candidates.append(Path(explicit_adc).expanduser())
    gcloud_cfg = (source_env.get("CLOUDSDK_CONFIG") or "").strip()
    if gcloud_cfg and not gcloud_cfg.startswith("${"):
        candidates.append(Path(gcloud_cfg).expanduser() / "application_default_credentials.json")
    home_override = first_valid_env(source_env, "HOME", "USERPROFILE", default="")
    if home_override:
        candidates.append(Path(home_override).expanduser() / ".config" / "gcloud" / "application_default_credentials.json")
    elif explicit_env is None and "PYTEST_CURRENT_TEST" not in os.environ:
        candidates.append(Path.home() / ".config" / "gcloud" / "application_default_credentials.json")
    appdata = (source_env.get("APPDATA") or "").strip()
    if appdata and not appdata.startswith("${"):
        candidates.append(Path(appdata) / "gcloud" / "application_default_credentials.json")
    for cand in candidates:
        try:
            if cand.is_file():
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    qp = data.get("quota_project_id")
                    has_qp = isinstance(qp, str) and bool(qp.strip())
                    if require_quota_project and not has_qp:
                        continue
                    adc_type = data.get("type")
                    return {
                        "type": adc_type.strip() if isinstance(adc_type, str) and adc_type.strip() else None,
                        "quota_project_id": qp.strip() if has_qp else None,
                        "path": str(cand),
                    }
        except (OSError, ValueError):
            continue
    return empty


def enforce_project_controls(
    *,
    auth_mode: str,
    project: str | None,
    source_env: Mapping[str, str],
    explicit_env: Mapping[str, str] | None = None,
) -> None:
    """Enforce `CC_AGY_BLOCKED_PROJECTS` and `CC_AGY_ALLOWED_PROJECTS` at 0 tokens before any call."""
    allowed = resolve_allowed_projects(source_env)
    blocked = resolve_blocked_projects(source_env)
    if not allowed and not blocked:
        return

    blocked_lower = {p.lower() for p in blocked}
    allowed_lower = {p.lower() for p in allowed}
    candidates: list[tuple[str, str]] = []

    real_proj = project.strip() if isinstance(project, str) and project.strip() and project != "custom-endpoint" else None
    if real_proj:
        candidates.append(("project", real_proj))

    companion_proj: str | None = None
    if auth_mode == "gemini_enterprise":
        from server.agy_cli import inspect_agy_cli_offline, resolve_enterprise_adc_seat

        ent_info = inspect_agy_cli_offline(source_env)
        raw_comp = ent_info.get("companion_project")
        if isinstance(raw_comp, str) and raw_comp.strip():
            companion_proj = raw_comp.strip()
            candidates.append(("agy_cli.cloudaicompanionProject", companion_proj))
        seat = resolve_enterprise_adc_seat(source_env)
        if seat["requested"] and seat["project"]:
            # Seat mode bills and reports against this project, so it is the one to check.
            companion_proj = str(seat["project"])
            candidates.append(("enterprise_adc.project", companion_proj))
    elif auth_mode == "vertex" and not resolve_gateway_base_url(env=source_env):
        adc_qp = read_adc_quota_project(source_env, explicit_env=explicit_env)
        if adc_qp:
            candidates.append(("adc.quota_project_id", adc_qp))

    for source_label, cand_proj in candidates:
        if cand_proj.lower() in blocked_lower:
            raise ValueError(
                f"Blocked GCP project '{cand_proj}' (detected via {source_label}, matched CC_AGY_BLOCKED_PROJECTS). "
                "Switch credentials (`--auth gemini-enterprise`, `GEMINI_API_KEY`, or `--project <allowed-id>`) "
                "or update CC_AGY_BLOCKED_PROJECTS."
            )

    if not allowed:
        return

    if auth_mode == "vertex":
        if resolve_gateway_base_url(env=source_env) and real_proj is None:
            return
        if real_proj is None or real_proj.lower() not in allowed_lower:
            raise ValueError(
                f"GCP project {real_proj!r} is not in CC_AGY_ALLOWED_PROJECTS ({', '.join(allowed)}). "
                "Pass an approved `--project` or update CC_AGY_ALLOWED_PROJECTS."
            )
    elif auth_mode == "gemini_enterprise":
        primary_ent = companion_proj or real_proj
        if primary_ent is None:
            raise ValueError(
                f"CC_AGY_ALLOWED_PROJECTS ({', '.join(allowed)}) is set, so Gemini Enterprise mode requires a "
                "verifiable `cloudaicompanionProject` in `agy` onboarding.json (or `--project <allowed-id>`). "
                "Complete `agy` sign-in or use CC_AGY_BLOCKED_PROJECTS instead."
            )
        if primary_ent.lower() not in allowed_lower:
            raise ValueError(
                f"GCP project '{primary_ent}' (from `agy` CLI / environment) is not in CC_AGY_ALLOWED_PROJECTS "
                f"({', '.join(allowed)})."
            )
    elif real_proj is not None and real_proj.lower() not in allowed_lower:
        raise ValueError(
            f"GCP project '{real_proj}' is not in CC_AGY_ALLOWED_PROJECTS ({', '.join(allowed)})."
        )


def resolve_vertex_project(
    requested_project: str | None = None,
    *,
    env: dict[str, str] | None = None,
    allow_missing: bool = False,
) -> str | None:
    """Resolve the Google Cloud Project ID from arguments or environment."""
    if requested_project is not None and not isinstance(requested_project, str):
        raise ValueError("project must be a string")
    source_env = env if env is not None else os.environ
    for candidate in (
        requested_project,
        source_env.get("GOOGLE_CLOUD_PROJECT"),
        source_env.get("VERTEX_PROJECT"),
        source_env.get("GCLOUD_PROJECT"),
        source_env.get("CLOUD_ML_PROJECT_ID"),
        source_env.get("CLAUDE_PLUGIN_OPTION_GOOGLE_CLOUD_PROJECT"),
        source_env.get("CC_AGY_USERCFG_GOOGLE_CLOUD_PROJECT"),
    ):
        if isinstance(candidate, str) and candidate:
            cleaned = candidate.strip()
            if cleaned and not cleaned.startswith("${") and cleaned != "<paying-project>":
                return cleaned
    if (
        resolve_vertex_base_url(env=source_env)
        or _clean_base_url_candidate(source_env.get("CC_AGY_GEMINI_BASE_URL"))
        or (
            _usable_google_gemini_base_url(source_env)
            and (
                resolve_configured_auth_mode(env=source_env) == "gateway"
                or resolve_bifrost_virtual_key(env=source_env)
                or resolve_static_vertex_base_url_token(env=source_env)
                or resolve_vertex_base_url_audience(
                    base_url=_usable_google_gemini_base_url(source_env),
                    env=source_env,
                )
            )
        )
    ):
        return "custom-endpoint"
    if allow_missing:
        return None
    raise ValueError(
        "Missing Google Cloud project ID. Set userConfig.google_cloud_project in Claude Code "
        "or export GOOGLE_CLOUD_PROJECT."
    )


_GEMINI_ENTERPRISE_ALIASES: frozenset[str] = frozenset(
    {
        "gemini_enterprise",
        "gemini-enterprise",
        "enterprise",
        "agy_cli",
        "agy-cli",
        "agy",
        "oauth_business",
        "oauth-business",
    }
)
_API_KEY_MODE_ALIASES: frozenset[str] = frozenset({"api_key", "api-key", "ai_studio", "aistudio"})
_VERTEX_MODE_ALIASES: frozenset[str] = frozenset({"vertex", "vertex_ai", "vertex-ai", "gcp"})
_GATEWAY_MODE_ALIASES: frozenset[str] = frozenset({"gateway", "proxy"})


def resolve_configured_auth_mode(
    requested_auth_mode: str | Mapping[str, str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an explicit authentication mode override ('gemini_enterprise', 'api_key', 'vertex', 'gateway', or None for auto)."""
    if isinstance(requested_auth_mode, Mapping) and env is None:
        env = requested_auth_mode
        requested_auth_mode = None
    source_env = env if env is not None else os.environ
    candidates = (
        requested_auth_mode,
        source_env.get("CC_AGY_AUTH_MODE"),
        source_env.get("CLAUDE_PLUGIN_OPTION_AUTH_MODE"),
        source_env.get("CC_AGY_USERCFG_AUTH_MODE"),
    )
    for raw in candidates:
        if not isinstance(raw, str):
            continue
        cleaned = raw.strip().lower()
        if not cleaned or cleaned.startswith("${") or cleaned in ("auto", "default", "none", "null"):
            continue
        if cleaned in _GEMINI_ENTERPRISE_ALIASES:
            return "gemini_enterprise"
        if cleaned in _API_KEY_MODE_ALIASES:
            return "api_key"
        if cleaned in _VERTEX_MODE_ALIASES:
            return "vertex"
        if cleaned in _GATEWAY_MODE_ALIASES:
            return "gateway"
        raise ValueError(
            f"Unsupported auth_mode {raw!r}. Expected 'auto', 'gemini_enterprise', 'api_key', 'vertex', or 'gateway'."
        )
    if env_flag("CC_AGY_GEMINI_ENTERPRISE", source_env) is True:
        return "gemini_enterprise"
    return None


def is_ge_mode_pinned(env: Mapping[str, str] | None = None) -> bool:
    """Return True when Gemini Enterprise mode is explicitly pinned in environment or plugin config."""
    try:
        return resolve_configured_auth_mode(None, env=env) == "gemini_enterprise"
    except ValueError:
        return False


def _resolve_explicit_project_only(
    requested_project: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an explicit GCP project ID without synthesizing `'custom-endpoint'`."""
    source_env = env if env is not None else os.environ
    for candidate in (
        requested_project,
        source_env.get("GOOGLE_CLOUD_PROJECT"),
        source_env.get("VERTEX_PROJECT"),
        source_env.get("GCLOUD_PROJECT"),
        source_env.get("CLOUD_ML_PROJECT_ID"),
        source_env.get("CLAUDE_PLUGIN_OPTION_GOOGLE_CLOUD_PROJECT"),
        source_env.get("CC_AGY_USERCFG_GOOGLE_CLOUD_PROJECT"),
    ):
        if isinstance(candidate, str) and candidate:
            cleaned = candidate.strip()
            if (
                cleaned
                and not cleaned.startswith("${")
                and cleaned not in ("<paying-project>", "custom-endpoint")
            ):
                return cleaned
    return None


def _should_auto_detect_agy_cli(
    explicit_env: Mapping[str, str] | None,
    source_env: Mapping[str, str],
) -> bool:
    flag = env_flag("CC_AGY_AUTO_DETECT_AGY_CLI", source_env)
    if flag is True:
        return True
    if flag is False:
        return False
    if first_valid_env(
        source_env,
        "CC_AGY_CLI_BIN",
        "CLAUDE_PLUGIN_OPTION_AGY_CLI_PATH",
        "CC_AGY_USERCFG_AGY_CLI_PATH",
        "CC_AGY_CLI_CONFIG_DIR",
        "CC_AGY_CLI_INSTALL_DIR",
        default="",
    ):
        return True
    if explicit_env is not None:
        explicit_home = first_valid_env(explicit_env, "HOME", "USERPROFILE", default="")
        if explicit_home and _REAL_ACCOUNT_HOME is not None:
            try:
                if Path(explicit_home).expanduser().resolve() != _REAL_ACCOUNT_HOME:
                    return True
            except OSError:
                pass
        return False
    if "PYTEST_CURRENT_TEST" in os.environ:
        env_home = first_valid_env(source_env, "HOME", "USERPROFILE", default="")
        if env_home and _REAL_ACCOUNT_HOME is not None:
            try:
                if Path(env_home).expanduser().resolve() != _REAL_ACCOUNT_HOME:
                    return True
            except OSError:
                pass
        return False
    return True


def resolve_auth_config(
    requested_project: str | None = None,
    requested_location: str | None = None,
    requested_api_key: str | None = None,
    *,
    requested_auth_mode: str | None = None,
    model: str = "gemini-2.5-flash",
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve authentication credentials for Gemini Enterprise (`agy` CLI), AI Studio, or Vertex AI.

    Supports `auth_mode` values:
    - `'gemini_enterprise'`: standalone Antigravity CLI (`agy`) backed by human Google sign-in,
      consuming included Gemini Enterprise Standard/Plus seat capacity.
    - `'api_key'`: Google AI Studio (`GEMINI_API_KEY` / `GOOGLE_API_KEY`).
    - `'vertex'`: Google Cloud Vertex AI (or custom Vertex-compatible AI gateway).
    """
    source_env = env if env is not None else os.environ
    resolved_key = resolve_api_key(requested_api_key, env=source_env)
    env_key = resolve_api_key(None, env=source_env)
    # A `requested_api_key` equal to the ambient `env_key` is a pass-1 echo (`_execute_job` / `followup_job`),
    # not a distinct per-call override, so configured Gemini gateways stay attached across two-pass worker calls.
    is_per_call_key_override = bool(requested_api_key and resolved_key != env_key)
    resolved_proj = resolve_vertex_project(requested_project, env=source_env, allow_missing=True)

    # Clean requested_project so scrubbed/placeholder values do not count as an explicit project override
    explicit_project = None
    if isinstance(requested_project, str):
        cleaned_req_proj = requested_project.strip()
        if cleaned_req_proj and not cleaned_req_proj.startswith("${") and cleaned_req_proj != "<paying-project>":
            explicit_project = cleaned_req_proj

    env_cfg_mode = resolve_configured_auth_mode(None, env=source_env)
    if env_cfg_mode == "gemini_enterprise" and requested_auth_mode is not None:
        req_mode_norm = resolve_configured_auth_mode(requested_auth_mode, env={})
        if req_mode_norm not in (None, "gemini_enterprise"):
            raise ValueError(
                "Gemini Enterprise mode is pinned (CC_AGY_AUTH_MODE=gemini_enterprise); "
                f"requested_auth_mode override {requested_auth_mode!r} is forbidden to prevent out-of-pool billing."
            )
    cfg_mode = resolve_configured_auth_mode(requested_auth_mode, env=source_env)
    # A requested `api_key` mode that merely echoes pass-1 output (same ambient key + Gemini gateway)
    # keeps the gateway so two-pass worker resolution is idempotent; env `CC_AGY_AUTH_MODE=api_key`
    # or a different per-call key still forces direct AI Studio.
    is_explicit_api_key_mode = bool(
        env_cfg_mode == "api_key"
        or (
            cfg_mode == "api_key"
            and not (env_key and resolved_key == env_key and resolve_gemini_base_url(env=source_env))
        )
    )

    def _finish(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("auth_mode") == "vertex":
            usable_gem = _usable_google_gemini_base_url(source_env)
            gem_gateway_signal = bool(
                usable_gem
                and (
                    resolve_static_vertex_base_url_token(env=source_env)
                    or resolve_bifrost_virtual_key(env=source_env)
                    or resolve_vertex_base_url_audience(base_url=usable_gem, env=source_env)
                )
            )
            gw_base = (
                resolve_gateway_base_url(env=source_env)
                if (
                    cfg_mode == "gateway"
                    or env_cfg_mode == "gateway"
                    or result.get("project") == "custom-endpoint"
                    or gem_gateway_signal
                )
                else (
                    resolve_vertex_base_url(env=source_env)
                    or _clean_base_url_candidate(source_env.get("CC_AGY_GEMINI_BASE_URL"))
                )
            )
            if gw_base:
                result.setdefault("base_url", gw_base)
                result.setdefault("gateway_wire", resolve_gateway_wire(base_url=gw_base, env=source_env))
                gw_aud = resolve_vertex_base_url_audience(base_url=gw_base, env=source_env)
                if gw_aud:
                    result.setdefault("base_url_audience", gw_aud)
                gw_tok = resolve_static_vertex_base_url_token(env=source_env)
                if gw_tok:
                    result.setdefault("base_url_token", gw_tok)
                bf_vk = resolve_bifrost_virtual_key(env=source_env)
                if bf_vk:
                    result.setdefault("bifrost_virtual_key", bf_vk)
                if (
                    result.get("gateway_wire") == "gemini"
                    and not resolve_vertex_base_url(env=source_env)
                    and not gw_tok
                    and not bf_vk
                    and resolved_key
                    and (cfg_mode == "gateway" or env_cfg_mode == "gateway")
                ):
                    result["api_key"] = resolved_key
        elif (
            result.get("auth_mode") == "api_key"
            and not is_per_call_key_override
            and not is_explicit_api_key_mode
        ):
            # Token and Bifrost-VK gateways route via the `vertex` branch above (`has_vertex_or_bifrost_gateway`).
            gem_base = resolve_gemini_base_url(env=source_env)
            if gem_base:
                result.setdefault("base_url", gem_base)
                result.setdefault("gateway_wire", resolve_gateway_wire(base_url=gem_base, env=source_env))
                gw_aud = resolve_vertex_base_url_audience(base_url=gem_base, env=source_env)
                if gw_aud:
                    result.setdefault("base_url_audience", gw_aud)
        result.setdefault("base_url", None)
        result.setdefault("gateway_wire", None)
        enforce_project_controls(
            auth_mode=str(result.get("auth_mode") or ""),
            project=result.get("project"),
            source_env=source_env,
            explicit_env=env,
        )
        return result

    if cfg_mode == "gemini_enterprise":
        if is_per_call_key_override:
            raise ValueError(
                "Gemini Enterprise mode is pinned (CC_AGY_AUTH_MODE=gemini_enterprise); "
                "per-call api_key overrides are forbidden to prevent out-of-pool billing."
            )
        from server.agy_cli import resolve_enterprise_adc_seat

        seat = resolve_enterprise_adc_seat(source_env)
        explicit_req_proj = (
            requested_project.strip()
            if isinstance(requested_project, str)
            and requested_project.strip()
            and not requested_project.strip().startswith("${")
            and requested_project.strip() not in ("<paying-project>", "custom-endpoint")
            else None
        )
        configured_seat_proj = (
            str(seat["project"]).strip() if (seat.get("requested") and seat.get("project")) else None
        )
        if explicit_req_proj and cfg_mode == "gemini_enterprise":
            allowed_projs = {p.lower() for p in resolve_allowed_projects(source_env)}
            if (
                (not configured_seat_proj or explicit_req_proj.lower() != configured_seat_proj.lower())
                and explicit_req_proj.lower() not in allowed_projs
            ):
                raise ValueError(
                    "Gemini Enterprise mode does not allow arbitrary project overrides: "
                    f"per-call project override {explicit_req_proj!r} is forbidden unless it matches "
                    "the configured enterprise seat project or is listed in CC_AGY_ALLOWED_PROJECTS."
                )
        ent_proj = explicit_req_proj or configured_seat_proj
        req_loc_clean = (requested_location or "").strip().lower()
        if req_loc_clean.startswith("${"):
            req_loc_clean = ""
        ent_loc = req_loc_clean or (
            str(seat["location"]) if (seat.get("requested") and seat.get("location")) else None
        )
        return _finish(
            {
                "auth_mode": "gemini_enterprise",
                "api_key": None,
                "project": ent_proj,
                "location": ent_loc,
                "reroute_note": None,
            }
        )

    if resolved_key and is_per_call_key_override:
        return _finish(
            {
                "auth_mode": "api_key",
                "api_key": resolved_key,
                "project": None,
                "location": None,
                "reroute_note": None,
            }
        )

    if is_explicit_api_key_mode:
        if not resolved_key:
            raise ValueError(
                "Missing Gemini API key for auth_mode='api_key'. Set userConfig.gemini_api_key or GEMINI_API_KEY."
            )
        return _finish(
            {
                "auth_mode": "api_key",
                "api_key": resolved_key,
                "project": None,
                "location": None,
                "reroute_note": None,
            }
        )

    if cfg_mode == "gateway":
        if not resolve_gateway_base_url(env=source_env):
            raise ValueError(
                "Missing gateway URL for auth_mode='gateway'. "
                "Set userConfig.vertex_base_url, CC_AGY_VERTEX_BASE_URL, or CC_AGY_GEMINI_BASE_URL."
            )
        gw_proj = resolved_proj or "custom-endpoint"
        resolved_loc, reroute_note = resolve_vertex_location(model, requested_location, env=source_env)
        return _finish(
            {
                "auth_mode": "vertex",
                "api_key": None,
                "project": gw_proj,
                "location": resolved_loc,
                "reroute_note": reroute_note,
            }
        )

    if cfg_mode == "vertex":
        if not resolved_proj:
            raise ValueError(
                "Missing Google Cloud project ID for auth_mode='vertex'. "
                "Set userConfig.google_cloud_project or GOOGLE_CLOUD_PROJECT."
            )
        resolved_loc, reroute_note = resolve_vertex_location(model, requested_location, env=source_env)
        return _finish(
            {
                "auth_mode": "vertex",
                "api_key": None,
                "project": resolved_proj,
                "location": resolved_loc,
                "reroute_note": reroute_note,
            }
        )

    has_vertex_or_bifrost_gateway = bool(
        resolve_vertex_base_url(env=source_env)
        or resolve_bifrost_virtual_key(env=source_env)
        or (
            resolve_gemini_base_url(env=source_env)
            and resolve_static_vertex_base_url_token(env=source_env)
        )
    )

    # Use API key when explicitly passed, or when present in the environment without an explicit project or gateway token/VK.
    # (If GOOGLE_GEMINI_BASE_URL or CC_AGY_GEMINI_BASE_URL is also set, `_finish` routes API-key calls through that Gemini gateway.)
    if resolved_key and (is_per_call_key_override or (not explicit_project and not has_vertex_or_bifrost_gateway)):
        return _finish(
            {
                "auth_mode": "api_key",
                "api_key": resolved_key,
                "project": None,
                "location": None,
                "reroute_note": None,
            }
        )

    if resolved_proj:
        resolved_loc, reroute_note = resolve_vertex_location(model, requested_location, env=source_env)
        return _finish(
            {
                "auth_mode": "vertex",
                "api_key": None,
                "project": resolved_proj,
                "location": resolved_loc,
                "reroute_note": reroute_note,
            }
        )

    if _should_auto_detect_agy_cli(env, source_env):
        from server.agy_cli import inspect_agy_cli_offline

        if inspect_agy_cli_offline(source_env).get("ok"):
            return _finish(
                {
                    "auth_mode": "gemini_enterprise",
                    "api_key": None,
                    "project": None,
                    "location": None,
                    "reroute_note": None,
                }
            )

    raise ValueError(
        "Missing credentials. Provide a Google AI Studio Gemini API key "
        "(userConfig.gemini_api_key or GEMINI_API_KEY), a Google Cloud project ID "
        "(userConfig.google_cloud_project or GOOGLE_CLOUD_PROJECT), or sign in with the "
        "Antigravity CLI (`agy`) for Gemini Enterprise seat quota (CC_AGY_AUTH_MODE=gemini_enterprise)."
    )


def _safe_tok_int(val: Any) -> int:
    try:
        return max(0, int(float(val or 0)))
    except (ValueError, TypeError):
        return 0


def _cost_for_turn(
    rates: dict[str, float],
    turn_usage: dict[str, Any],
    *,
    apply_long_context: bool,
) -> tuple[float, float, float]:
    prompt_tokens = _safe_tok_int(turn_usage.get("prompt_token_count"))
    cached_tokens = _safe_tok_int(turn_usage.get("cached_content_token_count"))
    candidates_tokens = _safe_tok_int(turn_usage.get("candidates_token_count"))
    thoughts_tokens = _safe_tok_int(turn_usage.get("thoughts_token_count"))

    in_mult = 1.0
    out_mult = 1.0
    if apply_long_context:
        in_mult = float(rates.get("long_context_input_multiplier", 1.0))
        out_mult = float(rates.get("long_context_output_multiplier", 1.0))

    uncached_input_tokens = max(0, prompt_tokens - cached_tokens)
    billable_output_tokens = candidates_tokens + thoughts_tokens

    input_cost = (uncached_input_tokens / 1_000_000.0) * rates["input"] * in_mult
    cached_cost = (cached_tokens / 1_000_000.0) * rates["cached_input"] * in_mult
    output_cost = (billable_output_tokens / 1_000_000.0) * rates["output"] * out_mult
    return input_cost, cached_cost, output_cost


def _resolve_rates_for_model(model_str: str) -> tuple[dict[str, float], bool]:
    norm = normalize_model_id(model_str)
    known = norm in MODEL_PRICING_PER_1M_USD
    resolved = MODEL_PRICING_PER_1M_USD.get(norm)
    if resolved is None:
        for known_prefix, known_rates in sorted(
            MODEL_PRICING_PER_1M_USD.items(),
            key=lambda kv: len(kv[0]),
            reverse=True,
        ):
            if norm.startswith(known_prefix):
                resolved = known_rates
                known = True
                break
    if resolved is None:
        resolved = MODEL_PRICING_PER_1M_USD["gemini-2.5-flash"]
    return resolved, known


def estimate_cost_usd(
    model: str,
    usage: dict[str, Any],
    *,
    single_request: bool = False,
) -> dict[str, Any]:
    """Calculate estimated USD and cents cost from Vertex usage_metadata token counts.

    Vertex prompt_token_count includes cached_content_token_count. Uncached input
    tokens equal max(0, prompt_token_count - cached_content_token_count).
    Thinking tokens (thoughts_token_count) are billed at the output token rate.

    Long-context (>200k prompt tokens) pricing is evaluated per single model call rather than
    on aggregated multi-step turn totals: the >200k tier applies only when `single_request=True`
    (such as `ask_gemini`) or on individual `usage["requests"]` entries marked `per_call=True`
    / `single_request=True`. Total token volume is always anchored to the turn totals so partial
    per-step entries never undercount total usage.
    """
    rates, pricing_known = _resolve_rates_for_model(model)

    prompt_tokens = _safe_tok_int(usage.get("prompt_token_count"))
    cached_tokens = _safe_tok_int(usage.get("cached_content_token_count"))
    cand_tokens = _safe_tok_int(usage.get("candidates_token_count"))
    think_tokens = _safe_tok_int(usage.get("thoughts_token_count"))
    requests_list = usage.get("requests")
    long_context_applied = False

    if isinstance(requests_list, list) and requests_list:
        req_input_cost = 0.0
        req_cached_cost = 0.0
        req_output_cost = 0.0
        sum_req_prompt = 0
        sum_req_cached = 0
        sum_req_out = 0
        for req_u in requests_list:
            if not isinstance(req_u, dict):
                continue
            req_prompt = _safe_tok_int(req_u.get("prompt_token_count"))
            req_cached = _safe_tok_int(req_u.get("cached_content_token_count"))
            req_out = _safe_tok_int(req_u.get("candidates_token_count")) + _safe_tok_int(req_u.get("thoughts_token_count"))
            sum_req_prompt += req_prompt
            sum_req_cached += req_cached
            sum_req_out += req_out
            is_single_call = bool(
                single_request
                or req_u.get("per_call")
                or req_u.get("single_request")
            )
            req_long = is_single_call and (req_prompt > 200_000)
            if req_long:
                long_context_applied = True
            req_model_raw = req_u.get("model")
            if isinstance(req_model_raw, str) and req_model_raw.strip():
                req_rates, _ = _resolve_rates_for_model(req_model_raw)
            else:
                req_rates = rates
            ic, cc, oc = _cost_for_turn(
                req_rates,
                req_u,
                apply_long_context=req_long,
            )
            req_input_cost += ic
            req_cached_cost += cc
            req_output_cost += oc

        # Anchor total priced volume to top-level turn totals when provided:
        # 1) If per-step entries sum to MORE than the turn totals (e.g. multiple steps
        #    reporting cumulative/shared usage for one invocation), scale costs down proportionally.
        total_out = cand_tokens + think_tokens
        if prompt_tokens > 0 and sum_req_prompt > prompt_tokens:
            req_input_cost *= prompt_tokens / sum_req_prompt
        if sum_req_cached > cached_tokens and (cached_tokens > 0 or prompt_tokens > 0):
            req_cached_cost *= (cached_tokens / sum_req_cached) if cached_tokens > 0 else 0.0
        if total_out > 0 and sum_req_out > total_out:
            req_output_cost *= total_out / sum_req_out

        # 2) If top-level totals exceed the sum of reported per-call entries, price the remainder at base rates
        rem_prompt = max(0, prompt_tokens - sum_req_prompt)
        rem_cached = max(0, cached_tokens - sum_req_cached)
        rem_out = max(0, total_out - sum_req_out)
        if rem_prompt > 0 or rem_cached > 0 or rem_out > 0:
            rem_ic, rem_cc, rem_oc = _cost_for_turn(
                rates,
                {
                    "prompt_token_count": rem_prompt,
                    "cached_content_token_count": rem_cached,
                    "candidates_token_count": rem_out,
                    "thoughts_token_count": 0,
                },
                apply_long_context=False,
            )
            req_input_cost += rem_ic
            req_cached_cost += rem_cc
            req_output_cost += rem_oc
        input_cost, cached_cost, output_cost = req_input_cost, req_cached_cost, req_output_cost
    else:
        is_single_call = bool(
            single_request
            or usage.get("per_call")
            or usage.get("single_request")
        )
        long_context_applied = is_single_call and (prompt_tokens > 200_000)
        input_cost, cached_cost, output_cost = _cost_for_turn(
            rates,
            usage,
            apply_long_context=long_context_applied,
        )

    total_usd = input_cost + cached_cost + output_cost

    return {
        "input_cost_usd": round(input_cost, 8),
        "cached_cost_usd": round(cached_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(total_usd, 8),
        "total_cost_cents": round(total_usd * 100.0, 6),
        "pricing_known": pricing_known,
        "long_context_multiplier_applied": long_context_applied,
    }
