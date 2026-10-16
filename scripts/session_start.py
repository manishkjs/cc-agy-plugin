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
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from server.jobs import capture_live_workspace_diff
from server.receipts import ReceiptStore, extract_diff_provenance, resolve_workspace_root


def handle_session_start(
    payload: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    store: ReceiptStore | None = None,
) -> dict[str, Any]:
    """Record the workspace baseline for a new Claude Code session_id."""
    source_env = env if env is not None else os.environ
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
        return {}

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
    live_diff = capture_live_workspace_diff(ws_path, baseline_head_sha=head_sha)
    prov = extract_diff_provenance(live_diff)
    receipt_store.record_session_baseline(
        ws_path,
        prov,
        baseline_head_sha=head_sha,
        session_id=session_id,
    )
    return {}


def main() -> int:
    try:
        raw = sys.stdin.read().strip()
        payload: dict[str, Any] = json.loads(raw) if raw else {}
        if not isinstance(payload, dict):
            payload = {}
        handle_session_start(payload)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
