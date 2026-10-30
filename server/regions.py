"""Vertex AI model-to-region routing, pricing table, and token budget helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


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
        "gemini-4-carbon-high",
        "gemini-4-carbon-medium",
        "gemini-4-carbon-low",
        "gemini-4-carbon-zero",
        "carbon-g",
        "carbon-g-zero",
        "bon-experiment-arm-226",
        "gemini-flash-next",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite-preview",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
    }
)

# Default Flash worker model (cost_profile economy/balanced and single-shot asks).
DEFAULT_FLASH_MODEL = "gemini-flash-next"

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

# Per-1M token pricing in USD on Vertex AI (standard tier <= 200k context).
# Cached input tokens are billed at 10% of the base input rate on Gemini 2.5/3.x.
MODEL_PRICING_PER_1M_USD: dict[str, dict[str, float]] = {
    "gemini-2.5-flash-lite": {
        "input": 0.10,
        "cached_input": 0.01,
        "output": 0.40,
    },
    "gemini-2.5-flash-lite-preview": {
        "input": 0.10,
        "cached_input": 0.01,
        "output": 0.40,
    },
    "gemini-3.5-flash-lite": {
        "input": 0.10,
        "cached_input": 0.01,
        "output": 0.40,
    },
    "gemini-3.5-flash-lite-preview": {
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
    "gemini-2.5-flash-preview": {
        "input": 0.30,
        "cached_input": 0.03,
        "output": 2.50,
    },
    "gemini-3-flash-preview": {
        "input": 0.50,
        "cached_input": 0.05,
        "output": 3.00,
    },
    "gemini-flash-next": {
        "input": 0.50,
        "cached_input": 0.05,
        "output": 3.00,
    },
    "gemini-3.8-flash": {
        "input": 0.50,
        "cached_input": 0.05,
        "output": 3.00,
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "cached_input": 0.125,
        "output": 10.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "gemini-3-pro-preview": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "gemini-3.1-pro-preview": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "gemini-4-argon": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "argon": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "gemini-4-carbon": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "gemini-4-carbon-high": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "gemini-4-carbon-medium": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "gemini-4-carbon-low": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "gemini-4-carbon-zero": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "carbon-g": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "carbon-g-zero": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    "bon-experiment-arm-226": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
}

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


def first_valid_env(source_env: Any, *keys: str, default: str = "") -> str:
    """Return the first non-empty, non-placeholder (${...}) value from source_env for keys."""
    for key in keys:
        val = source_env.get(key)
        if isinstance(val, str):
            cleaned = val.strip()
            if cleaned and not cleaned.startswith("${"):
                return cleaned
    return default


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


_TRUTHY_FLAG_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSY_FLAG_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off"})


def env_flag(name: str, env: Any = None) -> bool | None:
    """Parse a boolean env flag: True for 1/true/yes/on, False for 0/false/no/off, None if unset or unrecognized."""
    source = env if env is not None else os.environ
    val = str(source.get(name) or "").strip().lower()
    if val in _TRUTHY_FLAG_VALUES:
        return True
    if val in _FALSY_FLAG_VALUES:
        return False
    return None


STRICT_MODE_ENV_KEYS: tuple[str, ...] = (
    "CLAUDE_PLUGIN_OPTION_STRICT_DELEGATION_MODE",
    "CC_AGY_STRICT_MODE",
    "CC_AGY_USERCFG_STRICT_MODE",
)

VALID_COST_PROFILES: tuple[str, ...] = ("economy", "balanced", "quality")

COST_PROFILE_ENV_KEYS: tuple[str, ...] = (
    "CC_AGY_COST_PROFILE",
    "CLAUDE_PLUGIN_OPTION_COST_PROFILE",
    "CC_AGY_USERCFG_COST_PROFILE",
)


def resolve_cost_profile(env: Any = None, *, strict: bool = False) -> str:
    """Resolve the active cost/quality preset ('economy', 'balanced', or 'quality')."""
    source = env if env is not None else os.environ
    raw = first_valid_env(source, *COST_PROFILE_ENV_KEYS, default="balanced").lower()
    if raw in VALID_COST_PROFILES:
        return raw
    if strict:
        raise ValueError(
            f"Unsupported cost_profile {raw!r}. Expected one of {list(VALID_COST_PROFILES)}."
        )
    return "balanced"


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


def is_strict_mode_enabled(env: Any = None) -> bool:
    """Return True when strict orchestrator mode is on via any supported route.

    Strict mode can arrive as `CC_AGY_STRICT_MODE` (shell env), `CC_AGY_USERCFG_STRICT_MODE`
    (`.mcp.json` userConfig passthrough), or `CLAUDE_PLUGIN_OPTION_STRICT_DELEGATION_MODE`
    (Claude Code plugin option). Every server and hook check goes through this one helper.
    """
    source = env if env is not None else os.environ
    return any(env_flag(key, source) is True for key in STRICT_MODE_ENV_KEYS)


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
        "CC_AGY_VERTEX_BASE_URL_TOKEN",
        "CLAUDE_PLUGIN_OPTION_VERTEX_BASE_URL_TOKEN",
        "CC_AGY_USERCFG_VERTEX_BASE_URL_TOKEN",
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
        "CC_AGY_DATA_DIR",
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
        if isinstance(candidate, str) and candidate:
            cleaned = candidate.strip()
            if (
                cleaned
                and not cleaned.startswith("${")
                and cleaned.lower() not in ("<base-url>", "dummy", "none", "null")
            ):
                return cleaned
    return None


def resolve_vertex_base_url_token(
    requested_token: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an optional bearer/virtual-key token for a custom Vertex AI gateway."""
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
    if resolve_vertex_base_url(env=source_env):
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
    resolved_proj = resolve_vertex_project(requested_project, env=source_env, allow_missing=True)

    # Clean requested_project so scrubbed/placeholder values do not count as an explicit project override
    explicit_project = None
    if isinstance(requested_project, str):
        cleaned_req_proj = requested_project.strip()
        if cleaned_req_proj and not cleaned_req_proj.startswith("${") and cleaned_req_proj != "<paying-project>":
            explicit_project = cleaned_req_proj

    if resolved_key and requested_api_key:
        return {
            "auth_mode": "api_key",
            "api_key": resolved_key,
            "project": None,
            "location": None,
            "reroute_note": None,
        }

    cfg_mode = resolve_configured_auth_mode(requested_auth_mode, env=source_env)
    if cfg_mode == "gemini_enterprise":
        ent_proj = _resolve_explicit_project_only(requested_project, env=source_env)
        req_loc_clean = (requested_location or "").strip().lower()
        if req_loc_clean.startswith("${"):
            req_loc_clean = ""
        ent_loc = req_loc_clean or None
        return {
            "auth_mode": "gemini_enterprise",
            "api_key": None,
            "project": ent_proj,
            "location": ent_loc,
            "reroute_note": None,
        }

    if cfg_mode == "api_key":
        if not resolved_key:
            raise ValueError(
                "Missing Gemini API key for auth_mode='api_key'. Set userConfig.gemini_api_key or GEMINI_API_KEY."
            )
        return {
            "auth_mode": "api_key",
            "api_key": resolved_key,
            "project": None,
            "location": None,
            "reroute_note": None,
        }

    if cfg_mode == "gateway":
        if not resolve_vertex_base_url(env=source_env):
            raise ValueError(
                "Missing gateway URL for auth_mode='gateway'. "
                "Set userConfig.vertex_base_url or CC_AGY_VERTEX_BASE_URL."
            )
        gw_proj = resolved_proj or "custom-endpoint"
        resolved_loc, reroute_note = resolve_vertex_location(model, requested_location, env=source_env)
        return {
            "auth_mode": "vertex",
            "api_key": None,
            "project": gw_proj,
            "location": resolved_loc,
            "reroute_note": reroute_note,
        }

    if cfg_mode == "vertex":
        if not resolved_proj:
            raise ValueError(
                "Missing Google Cloud project ID for auth_mode='vertex'. "
                "Set userConfig.google_cloud_project or GOOGLE_CLOUD_PROJECT."
            )
        resolved_loc, reroute_note = resolve_vertex_location(model, requested_location, env=source_env)
        return {
            "auth_mode": "vertex",
            "api_key": None,
            "project": resolved_proj,
            "location": resolved_loc,
            "reroute_note": reroute_note,
        }

    has_gateway = bool(resolve_vertex_base_url(env=source_env))

    # Use API key when explicitly passed, or when present in the environment without an explicit project or gateway
    if resolved_key and (requested_api_key or (not explicit_project and not has_gateway)):
        return {
            "auth_mode": "api_key",
            "api_key": resolved_key,
            "project": None,
            "location": None,
            "reroute_note": None,
        }

    if resolved_proj:
        resolved_loc, reroute_note = resolve_vertex_location(model, requested_location, env=source_env)
        return {
            "auth_mode": "vertex",
            "api_key": None,
            "project": resolved_proj,
            "location": resolved_loc,
            "reroute_note": reroute_note,
        }

    if _should_auto_detect_agy_cli(env, source_env):
        from server.agy_cli import inspect_agy_cli_offline

        if inspect_agy_cli_offline(source_env).get("ok"):
            return {
                "auth_mode": "gemini_enterprise",
                "api_key": None,
                "project": None,
                "location": None,
                "reroute_note": None,
            }

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
        for known_prefix, known_rates in MODEL_PRICING_PER_1M_USD.items():
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
