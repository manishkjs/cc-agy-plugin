#!/usr/bin/env python3
"""Fast-path PreToolUse hook guard and delegation prompt linter entrypoint.

Keeps cold-start execution under ~55ms on non-strict Read/Grep/Glob/Bash calls by
importing only `server.env_flags` up front and loading prompt-lint, session-state,
and Bash static-analysis modules on demand.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from server.env_flags import (  # noqa: E402
    is_strict_mode_enabled as is_strict_mode_enabled,
    resolve_cost_profile as resolve_cost_profile,
)

_VIDEO_READ_SUFFIXES: frozenset[str] = frozenset({".mp4", ".mov", ".webm", ".mpeg"})
_DELEGATE_LINT_SUFFIXES: tuple[str, ...] = (
    "delegate_start",
    "delegate_to_gemini",
    "delegate_continue",
)
_LAZY_SUBMODULES: tuple[str, ...] = (
    "server.guard_session_state",
    "server.guard_prompt_lint",
    "server.guard_bash",
)


def _eval_non_strict_read_guard(
    tool_input: dict[str, Any],
    cwd: str,
    source_env: dict[str, str],
) -> dict[str, Any]:
    """Check video Read blocks and optional line-count guards without heavy imports."""
    raw_fp = str(tool_input.get("file_path") or tool_input.get("path") or "").strip()
    if not raw_fp:
        return {}
    dot_idx = raw_fp.rfind(".")
    ext = raw_fp[dot_idx:].lower() if dot_idx != -1 else ""
    if ext in _VIDEO_READ_SUFFIXES:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"Claude Read cannot inspect video file '{raw_fp}' ({ext}). "
                    f"Call `ask_gemini(prompt=..., media=[{raw_fp!r}])` for visual analysis or "
                    f"`delegate_start(prompt=..., media_files=[{raw_fp!r}], wait=true)` for implementation."
                ),
            }
        }
    read_guard = (
        source_env.get("CC_AGY_READ_GUARD")
        or source_env.get("GEMINI4_READ_GUARD")
        or "off"
    ).strip().lower()
    if (
        read_guard not in ("suggest", "block")
        or tool_input.get("offset") is not None
        or tool_input.get("limit") is not None
    ):
        return {}

    from server.guard_session_state import resolve_project_root

    ws_root = resolve_project_root(cwd, env=source_env)
    try:
        max_lines = int(
            source_env.get("CC_AGY_READ_GUARD_LINES")
            or source_env.get("GEMINI4_READ_GUARD_LINES")
            or 400
        )
    except ValueError:
        max_lines = 400
    cand = Path(raw_fp) if Path(raw_fp).is_absolute() else (ws_root / raw_fp)
    if not cand.is_file():
        return {}
    try:
        with cand.open("rb") as fh:
            line_count = sum(1 for _ in fh)
    except OSError:
        return {}
    if line_count < max_lines:
        return {}
    msg = (
        f"File '{raw_fp}' is {line_count} lines (>= {max_lines}-line threshold). "
        "Pass `offset` and `limit` to Read only the needed slice, or delegate bulk "
        f"analysis via `ask_gemini(prompt=..., context_files=[{raw_fp!r}])` / `delegate_start`."
    )
    if read_guard == "block":
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": msg,
            }
        }
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": msg,
        }
    }


def evaluate_pre_tool_use(
    payload: dict[str, Any],
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate a PreToolUse event and return `hookSpecificOutput` if blocked."""
    if not isinstance(payload, dict):
        return {}
    source_env = env if env is not None else os.environ  # type: ignore[assignment]
    tool_name = str(payload.get("tool_name") or "")
    raw_tool_input = payload.get("tool_input")
    tool_input = dict(raw_tool_input) if isinstance(raw_tool_input, dict) else {}

    if tool_name.endswith(_DELEGATE_LINT_SUFFIXES):
        from server.guard_prompt_lint import lint_delegation_prompt

        plan_items = tool_input.get("plan") if isinstance(tool_input.get("plan"), list) else []
        checklist_items = (
            tool_input.get("checklist") if isinstance(tool_input.get("checklist"), list) else []
        )
        prompt_text = "\n".join(
            [str(tool_input.get("prompt") or tool_input.get("follow_up_prompt") or "")]
            + [str(step) for step in plan_items]
            + [str(item) for item in checklist_items]
        )
        lint = lint_delegation_prompt(prompt_text)
        if not lint["ok"]:
            auto_repair = str(source_env.get("CC_AGY_REPAIR_CODE_DICTATION") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if not auto_repair:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": lint["reason"],
                    }
                }

    if not is_strict_mode_enabled(source_env):
        if tool_name == "Read":
            cwd = str(payload.get("cwd") or os.getcwd())
            return _eval_non_strict_read_guard(tool_input, cwd, source_env)
        return {}

    from server.guard_session_state import evaluate_strict_pre_tool_use, resolve_project_root

    cwd = str(payload.get("cwd") or os.getcwd())
    ws_root = resolve_project_root(cwd, env=source_env)
    out_dir = source_env.get("CC_AGY_DATA_DIR") or source_env.get("CLAUDE_PLUGIN_DATA")
    return evaluate_strict_pre_tool_use(tool_name, tool_input, cwd, ws_root, out_dir, source_env)


def __getattr__(name: str) -> Any:
    """Lazily resolve helper symbols split into `server.guard_*` submodules."""
    for mod_name in _LAZY_SUBMODULES:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, name):
            val = getattr(mod, name)
            globals()[name] = val
            return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> int:
    """CLI entrypoint reading PreToolUse JSON from stdin and writing decisions to stdout."""
    raw = sys.stdin.read().strip()
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0

    decision = evaluate_pre_tool_use(payload)
    if decision:
        sys.stdout.write(json.dumps(decision) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
