#!/usr/bin/env python3
"""SessionStart hook for cc-agy-plugin.

Records an HMAC-signed per-session baseline (`baselines/session-<session_id>.json`)
capturing `HEAD` and any pre-existing human WIP at the start of a new Claude Code
session, while refusing to overwrite an already-recorded baseline for the same
`session_id`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from server.jobs import capture_live_workspace_diff
from server.receipts import ReceiptStore, extract_diff_provenance, resolve_workspace_root


def verify_mcp_server_health() -> dict[str, Any]:
    """Perform a fast self-check verifying that the cc-agy-plugin MCP stdio tool catalog loads cleanly."""
    try:
        from server.mcp_stdio import get_tool_definitions

        tools = get_tool_definitions()
        tool_names = [str(t.get("name") or "") for t in tools if isinstance(t, dict)]
        required = {"delegate_start", "delegate_continue", "delegate_apply", "get_diff"}
        missing = sorted(required - set(tool_names))
        if missing:
            return {
                "healthy": False,
                "tool_count": len(tool_names),
                "error": f"Missing required MCP tools: {', '.join(missing)}",
            }
        return {"healthy": True, "tool_count": len(tool_names), "tools": tool_names}
    except Exception as exc:
        return {"healthy": False, "tool_count": 0, "error": f"{type(exc).__name__}: {exc}"}


_CHECKLIST_STEP_1 = (
    "1. Before delegating, include the user's verbatim task specification in the delegate prompt "
    "(do not paraphrase away exact sentences, formulas, or examples), and also write every behavioral "
    "requirement from the task as a numbered list (names, signatures, defaults, error cases, edge cases). "
    "Put the whole list in the delegate prompt. "
    "Ask the worker to add one test per item, and to search for every caller and consumer of each API it "
    "changes and update them too.\n"
)

# Strict mode applies a job automatically when its tests pass, and an applied job cannot be continued,
# so the checklist has to match whichever flow is active.
REVIEW_AFTER_APPLY_GUIDANCE = (
    "cc-agy-plugin delegation checklist:\n"
    + _CHECKLIST_STEP_1
    + "2. A job whose tests pass is applied to the workspace automatically. Then check each numbered item "
    "on the workspace: read the changed files and run pytest.\n"
    "3. If any item is missing or wrong, start one follow-up job with delegate_start that names the exact "
    "items to fix (or delegate_revert if the result is wrong overall). Never delegate_continue a job that "
    "was applied.\n"
    "4. If a job was not applied (for example its tests failed or it touched files outside its scope), "
    "review it with get_diff and fix it with "
    "delegate_continue(acceptance=[...]), or call delegate_apply if only the test command was wrong.\n"
    "5. Finish only after every item is covered."
)
REVIEW_BEFORE_APPLY_GUIDANCE = (
    "cc-agy-plugin delegation checklist:\n"
    + _CHECKLIST_STEP_1
    + "2. When the worker returns, check each numbered item against get_diff (use focus_symbols) and the "
    "worker's tests. Host pytest still runs the old code until you apply; to test the worker's version, "
    "pass test commands as acceptance to delegate_continue.\n"
    "3. If any item is missing or wrong, send one delegate_continue that names the exact items to fix.\n"
    "4. Apply only after every item is covered."
)
# Kept for callers/tests that refer to the default text.
REVIEW_LOOP_GUIDANCE = REVIEW_AFTER_APPLY_GUIDANCE

_TRUTHY = ("1", "true", "yes", "on")


def _auto_apply_on_pass(env: dict[str, str]) -> bool:
    """Mirror JobManager._should_auto_apply(None): explicit env wins, otherwise strict mode turns it on."""
    raw = str(env.get("CC_AGY_AUTO_APPLY_ON_PASS") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in _TRUTHY:
        return True
    return any(
        str(env.get(k) or "").strip().lower() in _TRUTHY
        for k in ("CLAUDE_PLUGIN_OPTION_STRICT_DELEGATION_MODE", "CC_AGY_STRICT_MODE", "CC_AGY_USERCFG_STRICT_MODE")
    )


def review_guidance_for(env: dict[str, str]) -> str:
    return REVIEW_AFTER_APPLY_GUIDANCE if _auto_apply_on_pass(env) else REVIEW_BEFORE_APPLY_GUIDANCE


def _review_guidance_enabled(env: dict[str, str]) -> bool:
    raw = str(env.get("CC_AGY_REVIEW_LOOP_GUIDANCE") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def handle_session_start(
    payload: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    store: ReceiptStore | None = None,
) -> dict[str, Any]:
    """Record the workspace baseline for a new Claude Code session_id and verify MCP server health."""
    source_env = env if env is not None else os.environ
    health = verify_mcp_server_health()
    out: dict[str, Any] = {}
    if _review_guidance_enabled(dict(source_env)):
        out["hookSpecificOutput"] = {
            "hookEventName": "SessionStart",
            "additionalContext": review_guidance_for(dict(source_env)),
        }
    if not health.get("healthy"):
        out["systemMessage"] = (
            f"cc-agy-plugin MCP self-check failed ({health.get('error')}). "
            "Check plugin installation and Python environment before delegating."
        )
    cwd = str(payload.get("cwd") or os.getcwd())
    session_id = str(payload.get("session_id") or "").strip() or None
    ws_path = resolve_workspace_root(cwd, env=source_env)

    git_check = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(ws_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if git_check.returncode != 0 or git_check.stdout.strip().lower() != "true":
        return out

    head_res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ws_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    head_sha = head_res.stdout.strip() if head_res.returncode == 0 else None
    receipt_store = store or ReceiptStore(workspace_root=ws_path)
    # The strict-mode Lead read budget is per session: reset the counter and mark when this session
    # started, so review reads are granted only for worker jobs started in this session.
    try:
        (receipt_store.data_dir / ".lead_read_count").write_text("0", encoding="utf-8")
        (receipt_store.data_dir / ".lead_session_started").write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass
    live_diff = capture_live_workspace_diff(ws_path, baseline_head_sha=head_sha)
    prov = extract_diff_provenance(live_diff)
    receipt_store.record_session_baseline(
        ws_path,
        prov,
        baseline_head_sha=head_sha,
        session_id=session_id,
    )
    return out


def main() -> int:
    try:
        raw = sys.stdin.read().strip()
        payload: dict[str, Any] = json.loads(raw) if raw else {}
        if not isinstance(payload, dict):
            payload = {}
        res = handle_session_start(payload)
        if res:
            sys.stdout.write(json.dumps(res) + "\n")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
