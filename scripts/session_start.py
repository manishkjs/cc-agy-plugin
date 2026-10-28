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
from server.regions import is_strict_mode_enabled


def verify_mcp_server_health(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Perform a fast self-check verifying that the cc-agy-plugin MCP stdio tool catalog loads cleanly."""
    try:
        from server.mcp_stdio import get_tool_definitions

        tools = get_tool_definitions(env)
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
    "1. Delegate on Turn 1 without pre-reading target files when their paths are already named. "
    "The plugin automatically attaches the user's verbatim task specification from the session prompt "
    "(and inlines any spec_files/context_files you pass to delegate_start), so do not re-type the full "
    "specification in your delegate_start prompt: keep your delegate_start call concise (<150 output tokens) "
    "and pass checklist with the key behavioral requirements and edge cases (names, signatures, defaults, "
    "error cases). Ask the worker to add one test per item, and to search for every caller and consumer "
    "of each API it changes and update them too.\n"
)

# Strict mode applies a job automatically when its tests pass, and an applied job cannot be continued,
# so the checklist has to match whichever flow is active.
REVIEW_AFTER_APPLY_GUIDANCE = (
    "cc-agy-plugin delegation checklist:\n"
    + _CHECKLIST_STEP_1
    + "2. A job whose tests pass is applied to the workspace automatically. If applied is true, "
    "real_acceptance_passed is true, auto_apply_skipped is absent, and contract_warnings is empty, finish "
    "immediately in one short sentence without re-reading the changed files or re-running pytest on the host "
    "(only read the changed files and run pytest if a numbered item was not covered by acceptance). If "
    "real_acceptance_passed is absent (verification is smoke_only, related_tests or unverified, for example in a "
    "repository without tests), review diff_preview or get_diff against every numbered item before finishing. If "
    "contract_warnings is "
    "non-empty in the worker digest, call get_diff(focus_symbols=[...]) on the flagged class/attribute; if the "
    "worker shadowed an existing class/public attribute not requested by the spec, fix it via a targeted "
    "delegate_start follow-up job before finishing.\n"
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
    "worker's tests (when verification is not acceptance_tests, nothing but a smoke check or no check ran, so "
    "this review is the only verification); if contract_warnings is non-empty in the worker digest, call "
    "get_diff(focus_symbols=[...]) "
    "on the flagged class/attribute and fix any unrequested class/public attribute shadowing via "
    "delegate_continue before applying. Host pytest still runs the old code until you apply; to test the "
    "worker's version, pass test commands as acceptance to delegate_continue.\n"
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
    return is_strict_mode_enabled(env)


ZERO_READ_GUIDANCE = (
    "\nLead file reads are off in this session (Lead read limit 0): call delegate_start on Turn 1 without "
    "Read, Grep, Glob, or Bash file inspection. The worker gets a repository map and reads the code itself. "
    "A few review reads open up only if a worker job fails."
)
ECONOMY_SCOUT_GUIDANCE = (
    "\nEconomy Lead scouting budget (up to 2 Glob/Grep or bounded Read(limit=120) calls; pre-delegation Bash is off): "
    "when target files and test paths are already clear from the prompt, call delegate_start immediately on Turn 1 "
    "with 0 reads. Use your <=2 Glob/Grep/Read(limit=120) calls ONLY when CLI entrypoints, existing API/query "
    "helpers, or test file locations are uncertain so you pass accurate allowed_paths and acceptance commands."
)
BALANCED_SCOUT_GUIDANCE = (
    "\nBalanced Lead scouting budget (up to 2 Glob/Grep or bounded Read(limit=120) calls; pre-delegation Bash is off): "
    "when target files and test paths are clear, call delegate_start on Turn 1 with 0 reads. Use <=2 Glob/Grep/Read(limit=120) "
    "calls ONLY to locate uncertain CLI entrypoints, existing API/query helpers, or test paths; NEVER prescribe speculative "
    "implementation details or guessed API/query names from partial reads — state behavioral outcomes and invariants only."
)


def review_guidance_for(env: dict[str, str]) -> str:
    base = REVIEW_AFTER_APPLY_GUIDANCE if _auto_apply_on_pass(env) else REVIEW_BEFORE_APPLY_GUIDANCE
    if is_strict_mode_enabled(env):
        from scripts.pre_tool_guard import resolve_lead_read_limit
        from server.regions import COST_PROFILE_ENV_KEYS, resolve_cost_profile

        limit = resolve_lead_read_limit(env)
        if limit == 0:
            return base + ZERO_READ_GUIDANCE
        profile = resolve_cost_profile(env)
        if profile == "economy" and limit is not None and limit <= 2:
            return base + ECONOMY_SCOUT_GUIDANCE
        if (
            profile == "balanced"
            and limit is not None
            and limit <= 2
            and any(str(env.get(k) or "").strip().lower() == "balanced" for k in COST_PROFILE_ENV_KEYS)
        ):
            return base + BALANCED_SCOUT_GUIDANCE
    return base


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
    health = verify_mcp_server_health(dict(source_env))
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
    # started on fresh startup/clear (not on compact/resume, which are mid-session events).
    session_source = str(payload.get("source") or "startup").strip().lower()
    if session_source in ("startup", "clear", ""):
        try:
            (receipt_store.data_dir / ".lead_read_count").write_text("0", encoding="utf-8")
            (receipt_store.data_dir / ".lead_escalated_edit_count").write_text("0", encoding="utf-8")
            marker_payload = json.dumps({"ts": round(time.time(), 3), "session_id": session_id})
            (receipt_store.data_dir / ".lead_session_started").write_text(marker_payload, encoding="utf-8")
            (receipt_store.data_dir / ".lead_read_files.json").write_text("[]", encoding="utf-8")
            (receipt_store.data_dir / ".lead_read_windows.json").write_text("{}", encoding="utf-8")
            (receipt_store.data_dir / ".lead_git_diff_spotcheck_used").unlink(missing_ok=True)
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
    if "--doctor" in sys.argv[1:]:
        from server.doctor import run_doctor

        report = run_doctor()
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0 if report.get("healthy") else 1
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
