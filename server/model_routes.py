"""Which model does which job, in one place.

Decision table the plugin documents for the Lead:

=============================================  ==========================================
Change                                          Route
=============================================  ==========================================
1-5 files                                       Lead edits directly (no delegation)
6+ files with a clear per-file plan             ``delegate_file_batch`` (single-shot writers)
Read-only investigation, test-suite debugging   ``delegate_start`` (full agent, bounded shell)
Doc, tests from cases, one edit repeated        ``draft_document`` / ``write_tests_from_cases``
                                                / ``replicate_edit``
=============================================  ==========================================
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

WRITER_KINDS = ("genai", "agent")
DEFAULT_BATCH_WRITER = "genai"
DEFAULT_BATCH_MODEL = "gemini-3.8-flash"
MECHANICAL_BATCH_MODEL = "gemini-3.8-flash"
DEFAULT_BATCH_THINKING = "LOW"
DEFAULT_CLAUDE_WRITER_MODEL = "opus"
DEFAULT_MAX_PARALLEL = 4
MAX_PARALLEL_CAP = 16
DEFAULT_MAX_CLAUDE_ESCALATIONS = 10
MAX_READ_BUDGET = 12
CHECK_TIMEOUT_RANGE = (5.0, 3600.0)


def _env_first(*keys: str) -> str:
    for key in keys:
        val = (os.environ.get(key) or "").strip()
        if val and not val.startswith("${"):
            return val
    return ""


@dataclass(frozen=True)
class BatchSettings:
    writer: str = DEFAULT_BATCH_WRITER
    model: str = DEFAULT_BATCH_MODEL
    thinking_level: str = DEFAULT_BATCH_THINKING
    claude_model: str = DEFAULT_CLAUDE_WRITER_MODEL
    claude_writer: bool = True
    max_parallel: int = DEFAULT_MAX_PARALLEL
    check_timeout: float = 300.0
    max_cost_cents: float | None = None
    max_claude_escalations: int = DEFAULT_MAX_CLAUDE_ESCALATIONS
    read_budget: int = 0


def resolve_batch_settings(args: dict[str, Any] | None = None) -> BatchSettings:
    """Per-call arguments win over env (CC_AGY_BATCH_*) and userConfig, which win over defaults."""
    args = args or {}
    has_arg_read_budget = args.get("read_budget") is not None
    raw_read_budget = args.get("read_budget") if has_arg_read_budget else (_env_first("CC_AGY_BATCH_READ_BUDGET") or 0)
    try:
        read_budget = max(0, min(MAX_READ_BUDGET, int(raw_read_budget)))
    except (TypeError, ValueError) as exc:
        raise ValueError("read_budget must be an integer") from exc
    arg_writer = str(args.get("writer") or "").strip().lower()
    env_writer = _env_first("CC_AGY_BATCH_WRITER", "CLAUDE_PLUGIN_OPTION_BATCH_WRITER").strip().lower()
    if arg_writer == "genai" and not has_arg_read_budget:
        read_budget = 0
    if read_budget > 0:
        if arg_writer and arg_writer != "agent":
            raise ValueError("read_budget needs writer='agent' (the 'genai' writer is a single call and cannot read files)")
        writer = "agent"
    else:
        writer = str(arg_writer or env_writer or DEFAULT_BATCH_WRITER)
    if writer not in WRITER_KINDS:
        raise ValueError(f"writer must be one of {list(WRITER_KINDS)}, got {writer!r}")
    raw_model = str(args.get("model") or _env_first("CC_AGY_BATCH_MODEL", "CLAUDE_PLUGIN_OPTION_BATCH_MODEL") or DEFAULT_BATCH_MODEL)
    model = MECHANICAL_BATCH_MODEL if raw_model.lower() in ("mechanical", "flash-lite") else raw_model
    thinking = str(args.get("thinking_level") or _env_first("CC_AGY_BATCH_THINKING") or DEFAULT_BATCH_THINKING).upper()
    if thinking not in ("MINIMAL", "LOW", "MEDIUM", "HIGH"):
        raise ValueError(f"thinking_level must be MINIMAL, LOW, MEDIUM or HIGH, got {thinking!r}")
    claude_model = str(args.get("claude_model") or _env_first("CC_AGY_BATCH_CLAUDE_MODEL") or DEFAULT_CLAUDE_WRITER_MODEL)
    claude_writer = args.get("claude_writer")
    if claude_writer is None:
        claude_writer = _env_first("CC_AGY_BATCH_CLAUDE_WRITER").lower() not in ("0", "false", "no")
    try:
        max_parallel = int(args.get("max_parallel") or _env_first("CC_AGY_BATCH_MAX_PARALLEL") or DEFAULT_MAX_PARALLEL)
    except ValueError as exc:
        raise ValueError("max_parallel must be an integer") from exc
    max_parallel = max(1, min(MAX_PARALLEL_CAP, max_parallel))
    try:
        check_timeout = float(args.get("check_timeout_seconds") or 300.0)
        max_cost = args.get("max_cost_cents")
        max_cost_cents = float(max_cost) if max_cost not in (None, "") else None
        max_claude = int(args.get("max_claude_escalations") if args.get("max_claude_escalations") is not None
                         else _env_first("CC_AGY_BATCH_MAX_CLAUDE") or DEFAULT_MAX_CLAUDE_ESCALATIONS)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric batch setting: {exc}") from exc
    check_timeout = max(CHECK_TIMEOUT_RANGE[0], min(CHECK_TIMEOUT_RANGE[1], check_timeout))
    if max_cost_cents is not None and max_cost_cents <= 0:
        raise ValueError("max_cost_cents must be positive")
    return BatchSettings(
        writer=writer, model=model, thinking_level=thinking, claude_model=claude_model,
        claude_writer=bool(claude_writer), max_parallel=max_parallel, check_timeout=check_timeout,
        max_cost_cents=max_cost_cents, max_claude_escalations=max(0, max_claude),
        read_budget=read_budget,
    )
