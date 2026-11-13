#!/usr/bin/env python3
"""Claude Code UserPromptSubmit hook for cc-agy-plugin.

Records the user's raw prompt into `<data_dir>/.last_user_prompt.json` so `delegate_start`
can attach the verbatim specification to the Gemini worker even if the Lead orchestrator
paraphrases or summarizes the user's prompt.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from server.receipts import ReceiptStore, resolve_workspace_root


def record_user_prompt(payload: dict[str, Any], env: dict[str, str] | None = None) -> Path | None:
    if not isinstance(payload, dict):
        return None
    source_env = env if env is not None else os.environ
    raw_prompt = str(
        payload.get("prompt")
        or payload.get("user_prompt")
        or payload.get("message")
        or ""
    ).strip()
    if not raw_prompt:
        return None
    cwd = payload.get("cwd") or source_env.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    ws_root = resolve_workspace_root(cwd)
    out_dir = source_env.get("CC_AGY_DATA_DIR") or source_env.get("CLAUDE_PLUGIN_DATA")
    store = ReceiptStore(out_dir or None, workspace_root=ws_root)
    target = store.data_dir / ".last_user_prompt.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    had_prior_prompt = target.is_file()
    record = {
        "ts": round(time.time(), 3),
        "session_id": payload.get("session_id"),
        "prompt": raw_prompt,
    }
    target.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    # On subsequent user prompts in an interactive multi-task session, reset the per-prompt
    # Lead read counter and cached file lists so each new prompt gets a fresh 2-read budget
    # without leaking prior tasks' context files into unrelated delegations.
    if had_prior_prompt:
        try:
            (store.data_dir / ".lead_read_count").write_text("0", encoding="utf-8")
            (store.data_dir / ".lead_read_files.json").write_text("[]", encoding="utf-8")
            (store.data_dir / ".lead_read_windows.json").write_text("{}", encoding="utf-8")
            (store.data_dir / ".lead_git_diff_spotcheck_used").unlink(missing_ok=True)
        except OSError:
            pass
    return target


def evaluate_user_prompt_submit(
    payload: dict[str, Any], env: dict[str, str] | None = None
) -> dict[str, Any]:
    """Emit immediate Turn-1 delegation guidance when multimodal files appear in the user prompt."""
    if not isinstance(payload, dict):
        return {}
    source_env = env if env is not None else os.environ
    raw_prompt = str(
        payload.get("prompt")
        or payload.get("user_prompt")
        or payload.get("message")
        or ""
    ).strip()
    if not raw_prompt:
        return {}
    cwd = payload.get("cwd") or source_env.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    ws_root = resolve_workspace_root(cwd)
    try:
        from server.genai_ask import _PROMPT_MEDIA_PATH_RE, extract_workspace_media_paths

        existing_media = extract_workspace_media_paths(ws_root, raw_prompt)
        mentioned_media = list(
            dict.fromkeys(
                [
                    *existing_media,
                    *(m.group(1) for m in _PROMPT_MEDIA_PATH_RE.finditer(raw_prompt)),
                ]
            )
        )
    except Exception:
        mentioned_media = []
    if not mentioned_media:
        return {}
    media_csv = ", ".join(repr(p) for p in mentioned_media[:6])
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                f"Multimodal asset(s) detected in prompt ({media_csv}). "
                "Do NOT inspect video or large visual/spec assets with multiple Lead Read turns: "
                "for code changes or UI fixes, delegate immediately on Turn 1 via "
                f"`delegate_start(prompt=..., media_files=[{media_csv}], wait=true)` "
                "(untracked media files are copied into the worker worktree and excluded from git diffs); "
                f"for read-only visual/spec analysis, call `ask_gemini(prompt=..., media=[{media_csv}])`."
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
    if isinstance(payload, dict):
        try:
            record_user_prompt(payload)
        except Exception:
            pass
        try:
            out = evaluate_user_prompt_submit(payload)
            if out:
                sys.stdout.write(json.dumps(out) + "\n")
                sys.stdout.flush()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
