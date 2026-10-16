#!/usr/bin/env python3
"""PostToolUse hook logger for cc-agy-plugin MCP tool calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from server.receipts import ReceiptStore, resolve_workspace_root  # noqa: E402


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0

    cwd = str(payload.get("cwd") or os.getcwd())
    ws_root = resolve_workspace_root(cwd)
    store = ReceiptStore(workspace_root=ws_root)
    audit_path = store.data_dir / "hook_audit.jsonl"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
