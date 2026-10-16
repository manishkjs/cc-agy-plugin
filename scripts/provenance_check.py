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
from server.jobs import capture_live_workspace_diff  # noqa: E402
from server.receipts import (  # noqa: E402
    ReceiptStore,
    extract_diff_provenance,
    resolve_workspace_root,
)


def _is_git_repository(path: Path) -> bool:
    if not path.exists():
        return False
    res = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0


def verify_workspace_provenance(
    workspace_root: str | Path | None = None,
    store: ReceiptStore | None = None,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify that every modified/untracked/deleted file and line in the workspace came from an HMAC-verified applied worker receipt (or pre-existing baseline).

    Also verifies that no recorded delegation prompt violated the 9-line code dictation limit.
    """
    ws_path = resolve_workspace_root(workspace_root, env=env)
    if not _is_git_repository(ws_path):
        return {
            "ok": False,
            "unverified_files": [],
            "unverified_lines_by_file": {},
            "unverified_deletions_by_file": {},
            "forged_receipts": [],
            "dictation_violations": [],
            "reason": f"Workspace path {ws_path} is not inside a git repository.",
        }

    receipt_store = store or ReceiptStore(workspace_root=ws_path)
    receipts = receipt_store.list_receipts(workspace_root=ws_path)

    dictation_violations: list[dict[str, Any]] = []
    forged_receipts: list[str] = []
    applied_files_allowed: set[str] = set()
    applied_added_hashes_by_file: dict[str, Counter[str]] = {}
    applied_removed_hashes_by_file: dict[str, Counter[str]] = {}
    latest_applied_file_hash: dict[str, str | None] = {}

    def _accumulate_provenance(prov: dict[str, Any], extra_files: list[str] | None = None) -> None:
        for fpath in (prov.get("files_changed") or []) + (extra_files or []):
            applied_files_allowed.add(fpath)
        per_file = prov.get("per_file") or {}
        for fpath, info in per_file.items():
            applied_files_allowed.add(fpath)
            add_counter = applied_added_hashes_by_file.setdefault(fpath, Counter())
            for h in info.get("added_line_sha256") or []:
                add_counter[h] += 1
            rem_counter = applied_removed_hashes_by_file.setdefault(fpath, Counter())
            for h in info.get("removed_line_sha256") or []:
                rem_counter[h] += 1

    # Include pre-existing uncommitted baseline state captured at session start
    baseline_rec = receipt_store.read_baseline_record(ws_path)
    baseline_head_sha = (baseline_rec or {}).get("baseline_head_sha")
    baseline_prov = (baseline_rec or {}).get("provenance")
    if isinstance(baseline_prov, dict):
        _accumulate_provenance(baseline_prov)

    for r in receipts:
        job_id = str(r.get("job_id") or "unknown")
        if not receipt_store.verify_receipt_signature(r):
            forged_receipts.append(job_id)
            continue

        if r.get("kind") == "delegate":
            for prompt_str in r.get("prompts") or []:
                lint = lint_delegation_prompt(str(prompt_str))
                if not lint["ok"]:
                    dictation_violations.append(
                        {
                            "job_id": job_id,
                            "max_consecutive_code_lines": lint["max_consecutive_code_lines"],
                            "reason": lint["reason"],
                        }
                    )

        if r.get("applied"):
            diff_file = receipt_store.diff_path(job_id)
            has_valid_diff = diff_file.exists()
            has_hashes = isinstance(r.get("applied_file_hashes"), dict) and bool(
                r.get("baseline_commit_sha")
            )
            if not (has_valid_diff and has_hashes):
                forged_receipts.append(job_id)
                continue
            prov = r.get("provenance") or {}
            _accumulate_provenance(prov, list(r.get("applied_files") or []))
            for fpath, fhash in (r.get("applied_file_hashes") or {}).items():
                latest_applied_file_hash[fpath] = fhash
        for hist_prov in r.get("applied_history") or []:
            if isinstance(hist_prov, dict):
                _accumulate_provenance(hist_prov)

    live_diff = capture_live_workspace_diff(ws_path, baseline_head_sha=baseline_head_sha)
    live_prov = extract_diff_provenance(live_diff)

    unverified_files: list[str] = []
    unverified_lines_by_file: dict[str, int] = {}
    unverified_deletions_by_file: dict[str, int] = {}

    from server.receipts import sha256_bytes

    for fpath in live_prov["files_changed"]:
        if fpath.startswith(".cc-agy-data/") or fpath.startswith(".worktrees/"):
            continue
        if fpath not in applied_files_allowed:
            unverified_files.append(fpath)
            continue

        file_info = live_prov["per_file"].get(fpath, {})
        live_added = file_info.get("added_line_sha256", [])
        allowed_added = Counter(applied_added_hashes_by_file.get(fpath, Counter()))
        unmatched_added = 0
        for h in live_added:
            if allowed_added[h] > 0:
                allowed_added[h] -= 1
            else:
                unmatched_added += 1
        if unmatched_added > 0:
            unverified_lines_by_file[fpath] = unmatched_added

        live_removed = file_info.get("removed_line_sha256", [])
        allowed_removed = Counter(applied_removed_hashes_by_file.get(fpath, Counter()))
        unmatched_removed = 0
        for h in live_removed:
            if allowed_removed[h] > 0:
                allowed_removed[h] -= 1
            else:
                unmatched_removed += 1

        # Also check if a post-apply file was modified or had worker-added lines deleted
        if fpath in latest_applied_file_hash:
            expected_hash = latest_applied_file_hash[fpath]
            live_file = ws_path / fpath
            if live_file.is_symlink():
                actual_hash: str | None = f"symlink:{os.readlink(live_file)}"
            elif live_file.exists() and live_file.is_file():
                actual_hash = sha256_bytes(live_file.read_bytes())
            else:
                actual_hash = None
            if actual_hash != expected_hash and unmatched_added == 0 and unmatched_removed == 0:
                unmatched_removed += max(1, sum(allowed_added.values()))

        if unmatched_removed > 0:
            unverified_deletions_by_file[fpath] = unmatched_removed

    ok = (
        not unverified_files
        and not unverified_lines_by_file
        and not unverified_deletions_by_file
        and not forged_receipts
        and not dictation_violations
    )
    reasons: list[str] = []
    if forged_receipts:
        reasons.append(f"Receipts failed HMAC signature verification: {forged_receipts}")
    if unverified_files:
        reasons.append(
            f"Files modified/created/deleted without an applied Vertex worker receipt: {unverified_files}"
        )
    if unverified_lines_by_file:
        reasons.append(
            f"Files contain added lines not matching any applied Vertex worker receipt: "
            f"{unverified_lines_by_file}"
        )
    if unverified_deletions_by_file:
        reasons.append(
            f"Files contain deleted lines not matching any applied Vertex worker receipt: "
            f"{unverified_deletions_by_file}"
        )
    if dictation_violations:
        reasons.append(
            f"Delegation prompt dictation violations: {dictation_violations}"
        )

    return {
        "ok": ok,
        "unverified_files": unverified_files,
        "unverified_lines_by_file": unverified_lines_by_file,
        "unverified_deletions_by_file": unverified_deletions_by_file,
        "forged_receipts": forged_receipts,
        "dictation_violations": dictation_violations,
        "reason": "; ".join(reasons) if reasons else None,
    }


def evaluate_stop_hook(
    payload: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    store: ReceiptStore | None = None,
) -> dict[str, Any]:
    """Evaluate a Claude Code Stop event and return the structured hook decision."""
    source_env = env if env is not None else os.environ
    if not is_strict_mode_enabled(source_env):
        return {}

    cwd = str(payload.get("cwd") or os.getcwd())
    verdict = verify_workspace_provenance(cwd, store=store, env=source_env)
    if verdict["ok"]:
        return {}

    # When stop_hook_active is true, Claude Code is already continuing after a prior Stop block;
    # emit a prominent systemMessage warning while respecting the loop guard.
    if payload.get("stop_hook_active"):
        return {
            "systemMessage": (
                f"WARNING (cc-agy-plugin provenance check): {verdict['reason']}"
            )
        }

    return {
        "decision": "block",
        "reason": (
            f"Provenance verification failed: {verdict['reason']}. "
            f"Revert unverified host edits or apply changes through delegate_apply."
        ),
    }


def main() -> int:
    raw = sys.stdin.read().strip()
    payload: dict[str, Any] = {}
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}

    out = evaluate_stop_hook(payload)
    if out:
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
