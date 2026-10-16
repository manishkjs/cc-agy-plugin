#!/usr/bin/env python3
"""Stop-hook outcome-based diff provenance verifier and dictation auditor."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from scripts.pre_tool_guard import is_strict_mode_enabled, lint_delegation_prompt  # noqa: E402
from server.receipts import ReceiptStore, extract_diff_provenance  # noqa: E402


def _current_workspace_diff(workspace_root: Path) -> str:
    res = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=str(workspace_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout if res.returncode == 0 else ""


def verify_workspace_provenance(
    workspace_root: str | Path,
    store: ReceiptStore | None = None,
) -> dict[str, Any]:
    """Verify that every modified file and added line in `git diff HEAD` came from an applied worker receipt.

    Also verifies that no recorded delegation prompt violated the 9-line code dictation limit.
    """
    ws_path = Path(workspace_root).resolve()
    receipt_store = store or ReceiptStore()
    receipts = receipt_store.list_receipts()

    dictation_violations: list[dict[str, Any]] = []
    applied_files_allowed: set[str] = set()
    applied_line_hashes_by_file: dict[str, Counter[str]] = {}

    for r in receipts:
        if r.get("kind") == "delegate":
            for prompt_str in r.get("prompts") or []:
                lint = lint_delegation_prompt(str(prompt_str))
                if not lint["ok"]:
                    dictation_violations.append(
                        {
                            "job_id": r.get("job_id"),
                            "max_consecutive_code_lines": lint["max_consecutive_code_lines"],
                            "reason": lint["reason"],
                        }
                    )

        if r.get("applied"):
            prov = r.get("provenance") or {}
            for fpath in prov.get("files_changed") or r.get("applied_files") or []:
                applied_files_allowed.add(fpath)
            per_file = prov.get("per_file") or {}
            for fpath, info in per_file.items():
                applied_files_allowed.add(fpath)
                counter = applied_line_hashes_by_file.setdefault(fpath, Counter())
                for h in info.get("added_line_sha256") or []:
                    counter[h] += 1

    live_diff = _current_workspace_diff(ws_path)
    live_prov = extract_diff_provenance(live_diff)

    unverified_files: list[str] = []
    unverified_lines_by_file: dict[str, int] = {}

    for fpath in live_prov["files_changed"]:
        if fpath.startswith(".cc-agy-data/") or fpath.startswith(".worktrees/"):
            continue
        if fpath not in applied_files_allowed:
            unverified_files.append(fpath)
            continue

        live_hashes = live_prov["per_file"].get(fpath, {}).get("added_line_sha256", [])
        allowed_counter = Counter(applied_line_hashes_by_file.get(fpath, Counter()))
        unmatched_count = 0
        for h in live_hashes:
            if allowed_counter[h] > 0:
                allowed_counter[h] -= 1
            else:
                unmatched_count += 1
        if unmatched_count > 0:
            unverified_lines_by_file[fpath] = unmatched_count

    ok = (
        not unverified_files
        and not unverified_lines_by_file
        and not dictation_violations
    )
    reasons: list[str] = []
    if unverified_files:
        reasons.append(
            f"Files modified without an applied Vertex worker receipt: {unverified_files}"
        )
    if unverified_lines_by_file:
        reasons.append(
            f"Files contain added lines not matching any applied Vertex worker receipt: "
            f"{unverified_lines_by_file}"
        )
    if dictation_violations:
        reasons.append(
            f"Delegation prompt dictation violations: {dictation_violations}"
        )

    return {
        "ok": ok,
        "unverified_files": unverified_files,
        "unverified_lines_by_file": unverified_lines_by_file,
        "dictation_violations": dictation_violations,
        "reason": "; ".join(reasons) if reasons else None,
    }


def main() -> int:
    raw = sys.stdin.read().strip()
    payload: dict[str, Any] = {}
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}

    # Avoid infinite Stop hook loops when stop_hook_active is already true
    if payload.get("stop_hook_active"):
        return 0

    if not is_strict_mode_enabled():
        return 0

    cwd = str(payload.get("cwd") or os.getcwd())
    verdict = verify_workspace_provenance(cwd)
    if not verdict["ok"]:
        sys.stdout.write(
            json.dumps(
                {
                    "decision": "block",
                    "reason": (
                        f"Provenance verification failed: {verdict['reason']}. "
                        f"Revert unverified host edits or apply changes through delegate_apply."
                    ),
                }
            )
            + "\n"
        )
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
