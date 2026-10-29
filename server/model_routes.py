"""Canonical routing table, worker budget defaults, and batch writer settings.

See :data:`ROUTING_DECISION_ROWS` and :func:`render_routing_markdown_table` for the
single source of truth synchronized across ``README.md``, skills, and subagent docs
via ``scripts/sync_docs.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

DEFAULT_MAX_TOOL_CALLS = 100
DEFAULT_FLASH_MAX_TOOL_CALLS = 55
DEFAULT_ECONOMY_FLASH_MAX_TOOL_CALLS = 100
DEFAULT_MAX_TOTAL_TOKENS = 5_000_000

ROUTING_TABLE_BEGIN = "<!-- SHARED:ROUTING_TABLE:BEGIN -->"
ROUTING_TABLE_END = "<!-- SHARED:ROUTING_TABLE:END -->"

ROUTING_DECISION_ROWS: tuple[tuple[str, str], ...] = (
    (
        "1 to 5 files with a known local fix",
        "Edit directly on the Lead outside strict mode; delegate with `delegate_start` when repo exploration, test debugging, or strict mode is active.",
    ),
    (
        "6+ files with a clear per-file plan",
        "Read at most 1 to 2 interface files, write `/tmp/<task>-plan.json`, and call `delegate_file_batch`.",
    ),
    (
        "Open-ended investigation, unknown files, or failing test suite",
        "Delegate on Turn 1 or Turn 2 with `delegate_start` (or `ask_gemini` for a fast single-shot review without a worker copy).",
    ),
    (
        "One new doc, tests from named cases, or one edit repeated across files",
        "Call `draft_document`, `write_tests_from_cases`, or `replicate_edit`.",
    ),
)


def render_routing_markdown_table() -> str:
    """Render the canonical Markdown routing decision table shared across docs and skills."""
    lines = [
        "| Change | Route |",
        "|---|---|",
    ]
    for change, route in ROUTING_DECISION_ROWS:
        lines.append(f"| {change} | {route} |")
    return "\n".join(lines)


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
        raw_timeout = args.get("check_timeout_seconds")
        check_timeout = float(raw_timeout) if raw_timeout not in (None, "") else 300.0
        max_cost = args.get("max_cost_cents")
        max_cost_cents = float(max_cost) if max_cost not in (None, "") else None
        max_claude = int(args.get("max_claude_escalations") if args.get("max_claude_escalations") is not None
                         else _env_first("CC_AGY_BATCH_MAX_CLAUDE") or DEFAULT_MAX_CLAUDE_ESCALATIONS)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric batch setting: {exc}") from exc
    # NaN compares False against everything, so "nan <= 0" passes and a NaN budget never trips.
    if not math.isfinite(check_timeout):
        raise ValueError(f"check_timeout_seconds must be a finite number, got {check_timeout!r}")
    check_timeout = max(CHECK_TIMEOUT_RANGE[0], min(CHECK_TIMEOUT_RANGE[1], check_timeout))
    if max_cost_cents is not None and (not math.isfinite(max_cost_cents) or max_cost_cents <= 0):
        raise ValueError("max_cost_cents must be a positive, finite number of cents")
    return BatchSettings(
        writer=writer, model=model, thinking_level=thinking, claude_model=claude_model,
        claude_writer=bool(claude_writer), max_parallel=max_parallel, check_timeout=check_timeout,
        max_cost_cents=max_cost_cents, max_claude_escalations=max(0, max_claude),
        read_budget=read_budget,
    )
