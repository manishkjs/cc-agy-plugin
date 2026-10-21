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
        except OSError:
            pass
    return target


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
