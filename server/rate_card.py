"""Declarative model pricing rate card, promo-expiry checks, and Gemini thinking-level clamping.

Loads `server/rate_card.json` using the Python standard library (`json`) so both Gemini worker
pricing (`server/regions.py`) and Claude Code lead-session cost accounting (`server/true_cost.py`)
share a single declarative source of truth with zero third-party YAML dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

RATE_CARD_PATH: Path = Path(__file__).resolve().with_name("rate_card.json")

ORDERED_THINKING_LEVELS: tuple[str, ...] = (
    "MINIMAL",
    "LOW",
    "MEDIUM",
    "HIGH",
    "EXTRA_HIGH",
)

_LEVEL_RANK: dict[str, int] = {lvl: idx for idx, lvl in enumerate(ORDERED_THINKING_LEVELS)}


def _normalize_model_lookup(model: str | None) -> str:
    raw = (model or "").strip().lower()
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    return raw


def _parse_ref_date(as_of: date | str | None) -> date:
    if as_of is None:
        return datetime.now(timezone.utc).date()
    if isinstance(as_of, str):
        return date.fromisoformat(as_of.strip())
    if isinstance(as_of, datetime):
        return as_of.date()
    return as_of


def validate_rate_card(card: Mapping[str, Any]) -> list[str]:
    """Validate rate-card schema and numeric invariants, returning any error messages."""
    errors: list[str] = []
    if not isinstance(card, Mapping):
        return ["rate_card root must be a JSON object"]
    if card.get("schema_version") != 1:
        errors.append(f"unsupported schema_version: {card.get('schema_version')!r}")

    gemini_models = card.get("gemini_models")
    if not isinstance(gemini_models, Mapping) or not gemini_models:
        errors.append("gemini_models must be a non-empty JSON object")
    else:
        for model_name, rates in gemini_models.items():
            if not isinstance(rates, Mapping):
                errors.append(f"gemini_models[{model_name!r}] must be an object")
                continue
            for req_key in ("input", "cached_input", "output"):
                val = rates.get(req_key)
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) < 0:
                    errors.append(f"gemini_models[{model_name!r}].{req_key} must be a non-negative finite number")
            for mult_key in ("long_context_input_multiplier", "long_context_output_multiplier"):
                if mult_key in rates:
                    mval = rates[mult_key]
                    if isinstance(mval, bool) or not isinstance(mval, (int, float)) or not math.isfinite(float(mval)) or float(mval) < 1.0:
                        errors.append(f"gemini_models[{model_name!r}].{mult_key} must be >= 1.0")

    claude_models = card.get("claude_lead_models")
    if not isinstance(claude_models, Mapping) or not claude_models:
        errors.append("claude_lead_models must be a non-empty JSON object")
    else:
        for model_key, tuple_vals in claude_models.items():
            if not isinstance(tuple_vals, list) or len(tuple_vals) != 5:
                errors.append(f"claude_lead_models[{model_key!r}] must be a 5-element array")
                continue
            for idx, item in enumerate(tuple_vals):
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) < 0:
                    errors.append(f"claude_lead_models[{model_key!r}][{idx}] must be a non-negative finite number")

    promos = card.get("promotional_rates")
    if promos is not None:
        if not isinstance(promos, list):
            errors.append("promotional_rates must be a list")
        else:
            for idx, entry in enumerate(promos):
                if not isinstance(entry, Mapping):
                    errors.append(f"promotional_rates[{idx}] must be an object")
                    continue
                exp = str(entry.get("expires_after") or "").strip()
                try:
                    date.fromisoformat(exp)
                except ValueError:
                    errors.append(f"promotional_rates[{idx}].expires_after must be YYYY-MM-DD (got {exp!r})")
                models_list = entry.get("models")
                if not isinstance(models_list, list) or not models_list or not all(isinstance(m, str) and m.strip() for m in models_list):
                    errors.append(f"promotional_rates[{idx}].models must be a non-empty list of model strings")
                std_rates = entry.get("standard_rates")
                if std_rates is not None:
                    if not isinstance(std_rates, list) or len(std_rates) != 5:
                        errors.append(f"promotional_rates[{idx}].standard_rates must be a 5-element array")
                    else:
                        for s_idx, s_item in enumerate(std_rates):
                            if isinstance(s_item, bool) or not isinstance(s_item, (int, float)) or not math.isfinite(float(s_item)) or float(s_item) < 0:
                                errors.append(f"promotional_rates[{idx}].standard_rates[{s_idx}] must be a non-negative finite number")

    thinking_cfg = card.get("gemini_thinking_levels")
    if not isinstance(thinking_cfg, Mapping):
        errors.append("gemini_thinking_levels must be an object")
    else:
        default_levels = thinking_cfg.get("default")
        if not isinstance(default_levels, list) or not default_levels or not all(lvl in _LEVEL_RANK for lvl in default_levels):
            errors.append("gemini_thinking_levels.default must be a non-empty list of valid thinking levels")
        prefixes = thinking_cfg.get("prefixes")
        if not isinstance(prefixes, Mapping):
            errors.append("gemini_thinking_levels.prefixes must be an object")
        else:
            for pfx, lvls in prefixes.items():
                if not isinstance(lvls, list) or not lvls or not all(lvl in _LEVEL_RANK for lvl in lvls):
                    errors.append(f"gemini_thinking_levels.prefixes[{pfx!r}] must be a non-empty list of valid thinking levels")

    return errors


def load_rate_card(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate `rate_card.json` from disk."""
    target = Path(path) if path is not None else RATE_CARD_PATH
    loaded = json.loads(target.read_text(encoding="utf-8"))
    errors = validate_rate_card(loaded)
    if errors:
        raise ValueError(f"Invalid rate card {target}: {'; '.join(errors)}")
    return loaded


def check_promotional_rate_expiry(
    as_of: date | str | None = None,
    *,
    card: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return informational notices for any promotional pricing entries past `expires_after`."""
    ref_date = _parse_ref_date(as_of)
    active_card = card if card is not None else _RATE_CARD
    promos = active_card.get("promotional_rates") or []
    notices: list[str] = []
    for entry in promos:
        if not isinstance(entry, Mapping):
            continue
        exp_raw = str(entry.get("expires_after") or "").strip()
        if not exp_raw:
            continue
        exp_date = date.fromisoformat(exp_raw)
        if ref_date > exp_date:
            models_str = ", ".join(str(m) for m in (entry.get("models") or []))
            note = str(entry.get("note") or "").strip()
            msg = (
                f"Promotional pricing for [{models_str}] expired after {exp_raw} "
                f"(checked as of {ref_date.isoformat()})."
            )
            if note:
                msg = f"{msg} {note}"
            notices.append(msg)
    return notices


def resolve_claude_lead_rates(
    as_of: date | str | None = None,
    *,
    card: Mapping[str, Any] | None = None,
) -> dict[str, tuple[float, float, float, float, float]]:
    """Return Claude lead pricing table as of `as_of`, reverting expired promos with `standard_rates`."""
    ref_date = _parse_ref_date(as_of)
    active_card = card if card is not None else _RATE_CARD
    rates: dict[str, tuple[float, float, float, float, float]] = {
        str(model_key): (
            float(vals[0]),
            float(vals[1]),
            float(vals[2]),
            float(vals[3]),
            float(vals[4]),
        )
        for model_key, vals in (active_card.get("claude_lead_models") or {}).items()
    }
    for entry in active_card.get("promotional_rates") or []:
        if not isinstance(entry, Mapping):
            continue
        std_vals = entry.get("standard_rates")
        exp_raw = str(entry.get("expires_after") or "").strip()
        if not exp_raw or not isinstance(std_vals, list) or len(std_vals) != 5:
            continue
        if ref_date > date.fromisoformat(exp_raw):
            std_tuple = (
                float(std_vals[0]),
                float(std_vals[1]),
                float(std_vals[2]),
                float(std_vals[3]),
                float(std_vals[4]),
            )
            for model_key in entry.get("models") or []:
                if isinstance(model_key, str) and model_key in rates:
                    rates[model_key] = std_tuple
    return rates


def supported_gemini_thinking_levels(
    model: str | None,
    *,
    allow_extra_high: bool = False,
    card: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return ordered `ThinkingLevel` names accepted by `model` (and SDK capability)."""
    active_card = card if card is not None else _RATE_CARD
    thinking_cfg = active_card.get("gemini_thinking_levels") or {}
    prefixes = thinking_cfg.get("prefixes") or {}
    norm = _normalize_model_lookup(model)

    matched: list[str] | None = None
    if norm:
        for prefix, levels in sorted(prefixes.items(), key=lambda kv: len(kv[0]), reverse=True):
            if norm.startswith(prefix.lower()):
                matched = list(levels)
                break
    if matched is None:
        matched = list(thinking_cfg.get("default") or ("LOW", "MEDIUM", "HIGH"))

    if not allow_extra_high:
        matched = [lvl for lvl in matched if lvl != "EXTRA_HIGH"]
    if not matched:
        matched = ["LOW", "MEDIUM", "HIGH"]
    return tuple(matched)


def clamp_gemini_thinking_level(
    model: str | None,
    thinking_level: str | None,
    *,
    allow_extra_high: bool = False,
    map_zero_to_minimal: bool = False,
    card: Mapping[str, Any] | None = None,
) -> str | None:
    """Clamp a requested `thinking_level` to the nearest level supported by `model` and the target SDK.

    Prevents upstream HTTP 400 (`Thinking level is unsupported`) when callers request `MINIMAL`,
    `MEDIUM` (on `gemini-3-pro-preview`), or `EXTRA_HIGH` on models whose published thinking table
    does not support that level, and avoids `google-genai` enum `UserWarning`s when `EXTRA_HIGH`
    is not defined in the installed SDK (`allow_extra_high=False`). Returns `None` when
    `thinking_level` is `None`, empty, `'DEFAULT'`, `'NONE'`, or `'THINKING_LEVEL_UNSPECIFIED'`.
    """
    if thinking_level is None:
        return None
    norm_tl = str(thinking_level).strip().upper().replace("-", "_")
    if norm_tl.startswith("THINKING_LEVEL_"):
        norm_tl = norm_tl[len("THINKING_LEVEL_"):]
    if not norm_tl or norm_tl in ("DEFAULT", "NONE", "UNSPECIFIED"):
        return None
    if norm_tl == "XHIGH":
        norm_tl = "EXTRA_HIGH"
    if norm_tl in ("ZERO", "0", "OFF"):
        if not map_zero_to_minimal:
            return "ZERO"
        norm_tl = "MINIMAL"

    if norm_tl not in _LEVEL_RANK:
        return norm_tl

    allowed = supported_gemini_thinking_levels(
        model,
        allow_extra_high=allow_extra_high,
        card=card,
    )
    if norm_tl in allowed:
        return norm_tl

    target_rank = _LEVEL_RANK[norm_tl]
    higher_or_equal = [lvl for lvl in allowed if _LEVEL_RANK[lvl] >= target_rank]
    if higher_or_equal:
        return min(higher_or_equal, key=lambda lvl: _LEVEL_RANK[lvl])
    return max(allowed, key=lambda lvl: _LEVEL_RANK[lvl])


_RATE_CARD: dict[str, Any] = load_rate_card()

MODEL_PRICING_PER_1M_USD: dict[str, dict[str, float]] = {
    str(model_key): {str(k): float(v) for k, v in rates.items()}
    for model_key, rates in _RATE_CARD["gemini_models"].items()
}

DEFAULT_LEAD_RATES: dict[str, tuple[float, float, float, float, float]] = {
    str(model_key): (
        float(vals[0]),
        float(vals[1]),
        float(vals[2]),
        float(vals[3]),
        float(vals[4]),
    )
    for model_key, vals in _RATE_CARD["claude_lead_models"].items()
}
