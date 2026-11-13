"""Lightweight, zero-dependency environment flag and profile resolvers.

Shared by both the MCP server (`server.regions`) and fast-path CLI hooks
(`scripts/pre_tool_guard.py`) so strict-mode and cost-profile semantics stay
identical with sub-millisecond import overhead.
"""

from __future__ import annotations

import os
from typing import Any

_TRUTHY_FLAG_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSY_FLAG_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off"})

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


def first_valid_env(source_env: Any, *keys: str, default: str = "") -> str:
    """Return the first non-empty, non-placeholder (`${...}`) value from `source_env`."""
    for key in keys:
        val = source_env.get(key)
        if isinstance(val, str):
            cleaned = val.strip()
            if cleaned and not cleaned.startswith("${"):
                return cleaned
    return default


def env_flag(name: str, env: Any = None) -> bool | None:
    """Parse a boolean env flag (`1/true/yes/on` -> True, `0/false/no/off` -> False, else None)."""
    source = env if env is not None else os.environ
    val = str(source.get(name) or "").strip().lower()
    if val in _TRUTHY_FLAG_VALUES:
        return True
    if val in _FALSY_FLAG_VALUES:
        return False
    return None


def resolve_cost_profile(env: Any = None, *, strict: bool = False) -> str:
    """Resolve the active cost/quality preset (`economy`, `balanced`, or `quality`).

    Args:
        env: Optional environment mapping; defaults to `os.environ`.
        strict: When True, raise `ValueError` on unrecognized non-empty profile values.

    Returns:
        Normalized cost profile name (`'economy'`, `'balanced'`, or `'quality'`).
    """
    source = env if env is not None else os.environ
    raw = first_valid_env(source, *COST_PROFILE_ENV_KEYS, default="balanced").lower()
    if raw in VALID_COST_PROFILES:
        return raw
    if strict:
        raise ValueError(
            f"Unsupported cost_profile {raw!r}. Expected one of {list(VALID_COST_PROFILES)}."
        )
    return "balanced"


def is_strict_mode_enabled(env: Any = None) -> bool:
    """Return True when strict orchestrator mode is on via any supported route.

    Strict mode can arrive as `CC_AGY_STRICT_MODE` (shell env), `CC_AGY_USERCFG_STRICT_MODE`
    (`.mcp.json` userConfig passthrough), or `CLAUDE_PLUGIN_OPTION_STRICT_DELEGATION_MODE`
    (Claude Code plugin option). Every server and hook check goes through this one helper.
    """
    source = env if env is not None else os.environ
    return any(env_flag(key, source) is True for key in STRICT_MODE_ENV_KEYS)
