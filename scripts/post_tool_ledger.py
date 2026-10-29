#!/usr/bin/env python3
"""PostToolUse hook logger and verified-apply completion guard for cc-agy-plugin."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from server.receipts import ReceiptStore, resolve_workspace_root, validate_job_id  # noqa: E402
from server.regions import env_flag, is_strict_mode_enabled  # noqa: E402

_AUTO_APPLY_DELEGATE_SUFFIXES = (
    "delegate_start",
    "delegate_continue",
    "delegate_result",
    "delegate_to_gemini",
)

_BLOCKED_RE = re.compile(r"\bBLOCKED\b", re.I)


def _extract_response_dict(raw_resp: Any) -> dict[str, Any] | None:
    """Parse an MCP tool_response payload (dict, JSON string, or content-block list) into a dict."""
    if isinstance(raw_resp, dict):
        if "applied" in raw_resp or "job_id" in raw_resp or "status" in raw_resp:
            return raw_resp
        content = raw_resp.get("content")
        if isinstance(content, list):
            return _extract_response_dict(content)
        return None
    if isinstance(raw_resp, list):
        for item in raw_resp:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parsed = _extract_response_dict(item["text"])
                if parsed is not None:
                    return parsed
        return None
    if isinstance(raw_resp, str):
        stripped = raw_resp.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _has_pending_parallel_tool_uses(transcript_path: str | None, current_tool_use_id: str | None) -> bool:
    """Return True if the latest assistant message issued multiple tool_use blocks not yet all resolved.

    Claude Code writes one JSONL line per streaming content block (all sharing the same `message.id`),
    so we collect every assistant entry belonging to that `message.id` rather than stopping at the last line.
    """
    if not transcript_path:
        return False
    t_path = Path(transcript_path)
    if not t_path.is_file():
        return False
    try:
        lines = t_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False

    target_msg_id: str | None = None
    earliest_assistant_idx = -1
    found_assistant = False
    tool_use_ids: set[str] = set()

    for idx in range(len(lines) - 1, -1, -1):
        raw_line = lines[idx].strip()
        if not raw_line:
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        etype = entry.get("type")
        if etype == "assistant":
            msg = entry.get("message")
            msg_id = str(msg.get("id") or "") if isinstance(msg, dict) else ""
            if not found_assistant:
                found_assistant = True
                target_msg_id = msg_id or None
            else:
                if target_msg_id is not None:
                    if msg_id != target_msg_id:
                        break
                else:
                    # Without message.id, group only contiguous assistant entries.
                    pass
            earliest_assistant_idx = idx
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                        tool_use_ids.add(str(block["id"]))
        elif found_assistant:
            if target_msg_id is None:
                break
            # Even with message.id, stop once we pass a non-tool-result user prompt from an earlier turn.
            if etype == "user":
                msg = entry.get("message")
                content = msg.get("content") if isinstance(msg, dict) else None
                has_tool_result = isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                )
                if not has_tool_result:
                    break

    if len(tool_use_ids) <= 1 or earliest_assistant_idx < 0:
        return False

    completed_ids: set[str] = set()
    if current_tool_use_id:
        completed_ids.add(str(current_tool_use_id))
    for idx in range(earliest_assistant_idx + 1, len(lines)):
        raw_line = lines[idx].strip()
        if not raw_line:
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get("type") != "user":
            continue
        msg = entry.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id"):
                    completed_ids.add(str(block["tool_use_id"]))
    return not tool_use_ids.issubset(completed_ids)


def _inspect_receipt_apply_state(store: ReceiptStore, job_id: str | None) -> tuple[bool, bool]:
    """Return `(has_explicit_acceptance_pass, has_worker_caveat)` from the stored receipt."""
    if not job_id:
        return False, False
    try:
        safe_id = validate_job_id(str(job_id))
        rec = store.read_receipt(safe_id)
    except Exception:
        return False, False
    summary_txt = str(rec.get("summary") or "")
    has_caveat = bool(
        _BLOCKED_RE.search(summary_txt)
        or "?" in summary_txt
        or rec.get("status") != "ok"
        or bool(rec.get("contract_warnings"))
    )
    from server.verification_policy import satisfies_post_tool_auto_stop

    return satisfies_post_tool_auto_stop(rec), has_caveat


def evaluate_post_tool_use(
    payload: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    store: ReceiptStore | None = None,
) -> dict[str, Any]:
    """Record the PostToolUse audit entry and guide/stop the Lead when verified edits are auto-applied."""
    source_env = os.environ if env is None else env
    cwd = str(payload.get("cwd") or source_env.get("CLAUDE_PROJECT_DIR") or os.getcwd())
    ws_root = resolve_workspace_root(cwd)
    active_store = store or ReceiptStore(
        data_dir=source_env.get("CC_AGY_DATA_DIR") or None,
        workspace_root=ws_root,
    )
    audit_path = active_store.data_dir / "hook_audit.jsonl"
    tool_name = str(payload.get("tool_name") or "")
    entry = {
        "timestamp_epoch": round(time.time(), 3),
        "hook_event_name": payload.get("hook_event_name", "PostToolUse"),
        "tool_name": payload.get("tool_name"),
        "session_id": payload.get("session_id"),
        "cwd": cwd,
        "workspace_root": str(ws_root),
    }
    try:
        with open(audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError:
        pass

    explicit_flag = env_flag("CC_AGY_AUTO_STOP_ON_APPLY", source_env)
    if explicit_flag is False:
        return {}
    strict_enabled = is_strict_mode_enabled(source_env)
    if explicit_flag is not True and not strict_enabled:
        return {}

    if not any(tool_name == sfx or tool_name.endswith(f"__{sfx}") for sfx in _AUTO_APPLY_DELEGATE_SUFFIXES):
        return {}

    raw_resp = payload.get("tool_response")
    if raw_resp is None:
        raw_resp = payload.get("tool_result")
    resp_dict = _extract_response_dict(raw_resp)
    if not isinstance(resp_dict, dict):
        return {}

    if resp_dict.get("applied") is not True or resp_dict.get("status", "ok") != "ok":
        return {}
    if (
        resp_dict.get("scope_violations")
        or resp_dict.get("missing_prompt_symbols")
        or resp_dict.get("contract_warnings")
        or resp_dict.get("has_worker_caveat")
    ):
        return {}

    resp_summary = str(resp_dict.get("summary") or "")
    if _BLOCKED_RE.search(resp_summary) or "?" in resp_summary:
        return {}
    for p_item in (resp_dict.get("progress") or []):
        if isinstance(p_item, dict) and str(p_item.get("kind") or "").lower() in ("question", "blocker"):
            return {}

    rec_pass, rec_caveat = _inspect_receipt_apply_state(active_store, resp_dict.get("job_id"))
    if rec_caveat:
        return {}

    if "real_acceptance_passed" in resp_dict:
        has_real_pass = resp_dict.get("real_acceptance_passed") is True and resp_dict.get("acceptance_passed") is True
    else:
        has_real_pass = rec_pass
    if not has_real_pass:
        return {}

    if _has_pending_parallel_tool_uses(
        str(payload.get("transcript_path") or "") or None,
        str(payload.get("tool_use_id") or "") or None,
    ):
        return {}

    applied_files = resp_dict.get("applied_files") or resp_dict.get("files_changed") or []
    files_note = (
        f" ({', '.join(str(f) for f in applied_files[:5])})"
        if isinstance(applied_files, list) and applied_files
        else ""
    )
    if explicit_flag is True:
        return {
            "continue": False,
            "stopReason": f"Delegated changes passed acceptance checks and were applied to the workspace{files_note}.",
        }
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": (
                f"Delegated changes{files_note} passed explicit acceptance checks and were applied to the workspace. "
                "If every part of the user's request is now complete, finish immediately in one concise sentence "
                "with zero further tool calls; only continue if the user's prompt has an unaddressed part."
            ),
        }
    }


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0
    decision = evaluate_post_tool_use(payload)
    if decision:
        sys.stdout.write(json.dumps(decision) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
