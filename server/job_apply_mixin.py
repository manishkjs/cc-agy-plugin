"""Diff pagination, conflict-safe apply, atomic rollback, and auto-apply mixin for `JobManager`."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from server.acceptance_runner import (
    _TEST_RUNNER_CMD_RE,
    _extract_failing_test_nodes_and_files,
    _extract_pytest_runner_prefix,
    run_acceptance,
)
from server.ast_regression_guards import (
    _is_test_file_path,
    _prompt_forbids_existing_test_updates,
)
from server.job_git_utils import (
    _env_int,
    _explicitly_targets_runtime_artifact,
    _git_raw,
    _hash_path,
    _is_ephemeral_build_or_test_artifact,
    _is_gitignored_on_host,
    _is_referenced_by_files_or_prompt,
    _is_tracked_in_git,
    _matches_allowed_paths,
    _matches_file_filter,
    _split_diff_by_file,
    validate_allowed_paths,
)
from server.receipts import (
    extract_diff_provenance,
    parse_diff_git_header,
    parse_unified_header_path,
    receipt_owner_pid_alive as _receipt_owner_pid_alive,
    sha256_bytes,
    validate_job_id,
)
from server.regions import (
    DEFAULT_DIFF_PAGE_TOKENS,
    MAX_TOOL_RESPONSE_TOKENS,
    env_flag,
    estimate_tokens,
    is_strict_mode_enabled,
    paginate_text_by_tokens,
    resolve_cost_profile,
    truncate_to_token_budget,
)
from server.test_discovery import _collect_workspace_node_bin_dirs
from server.verification_policy import (
    classify_verification,
    has_real_acceptance_pass,
    is_non_smoke_passing_entry,
    satisfies_auto_apply_verification,
)
from server.worker_prompt_builder import (
    escalate_after_failures,
    is_quota_or_rate_limit_error,
    worker_turn_failed,
)


def _copy2(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
    """Delegate to `server.jobs.shutil.copy2` so `mock.patch('server.jobs.shutil.copy2')` intercepts apply copies."""
    jobs_mod = sys.modules.get("server.jobs")
    target_shutil = getattr(jobs_mod, "shutil", shutil) if jobs_mod is not None else shutil
    return target_shutil.copy2(src, dst, *args, **kwargs)


class JobApplyMixin:
    """Diff inspection, conflict-safe apply, atomic rollback, and auto-apply helpers."""

    @staticmethod
    def _filter_diff_by_symbols(diff_text: str, focus_symbols: list[str]) -> str:
        clean_syms = [str(s).strip() for s in focus_symbols if isinstance(s, str) and str(s).strip()]
        if not clean_syms or not diff_text:
            return diff_text
        sym_patterns = [
            re.compile(rf"(?<![A-Za-z0-9_]){re.escape(sym)}(?![A-Za-z0-9_])")
            for sym in clean_syms
        ]
        matched_files: list[str] = []
        file_chunks: list[list[str]] = []
        cur_file_lines: list[str] = []
        for line in diff_text.splitlines(keepends=True):
            if line.startswith("diff --git "):
                if cur_file_lines:
                    file_chunks.append(cur_file_lines)
                cur_file_lines = [line]
            else:
                cur_file_lines.append(line)
        if cur_file_lines:
            file_chunks.append(cur_file_lines)
        for chunk in file_chunks:
            header_lines: list[str] = []
            hunks: list[list[str]] = []
            cur_hunk: list[str] = []
            for ln in chunk:
                if ln.startswith("@@ "):
                    if cur_hunk:
                        hunks.append(cur_hunk)
                    cur_hunk = [ln]
                elif cur_hunk:
                    cur_hunk.append(ln)
                else:
                    header_lines.append(ln)
            if cur_hunk:
                hunks.append(cur_hunk)
            kept_hunks = [
                "".join(h)
                for h in hunks
                if any(pat.search("".join(h[1:]) if len(h) > 1 else "".join(h)) for pat in sym_patterns)
            ]
            if kept_hunks:
                matched_files.append("".join(header_lines) + "".join(kept_hunks))
            elif not hunks and any(pat.search("".join(header_lines)) for pat in sym_patterns):
                matched_files.append("".join(header_lines))
        return "".join(matched_files)

    def get_diff(
        self,
        job_id: str,
        *,
        file_filter: str | None = None,
        page: int = 1,
        page_tokens: int = DEFAULT_DIFF_PAGE_TOKENS,
        summary_only: bool = False,
        focus_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return paginated unified git diff for job_id measured in tokens, with per-file line stats
        and optional summary-only or symbol-focused hunk filtering."""
        safe_id = validate_job_id(job_id)
        receipt = self.store.read_receipt(safe_id)
        diff_path = self.store.diff_path(safe_id)
        full_diff = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""

        if file_filter and full_diff:
            filtered_chunks: list[str] = []
            current_chunk: list[str] = []
            keep_chunk = False
            in_hunk = False
            for line in full_diff.splitlines(keepends=True):
                if line.startswith("diff --git "):
                    if current_chunk and keep_chunk:
                        filtered_chunks.append("".join(current_chunk))
                    current_chunk = [line]
                    hdr = parse_diff_git_header(line)
                    cand_path = (hdr[1] or hdr[0]) if hdr else ""
                    keep_chunk = bool(cand_path and _matches_file_filter(cand_path, file_filter))
                    in_hunk = False
                else:
                    if line.startswith("@@ "):
                        in_hunk = True
                    elif not in_hunk and line.startswith("+++ "):
                        cand_path = parse_unified_header_path(line, "+++")
                        if cand_path and cand_path != "/dev/null" and _matches_file_filter(cand_path, file_filter):
                            keep_chunk = True
                    elif not in_hunk and line.startswith("--- "):
                        cand_path = parse_unified_header_path(line, "---")
                        if cand_path and cand_path != "/dev/null" and _matches_file_filter(cand_path, file_filter):
                            keep_chunk = True
                    current_chunk.append(line)
            if current_chunk and keep_chunk:
                filtered_chunks.append("".join(current_chunk))
            full_diff = "".join(filtered_chunks)
            if not full_diff:
                omitted_diff_path = diff_path.with_suffix(".omitted.diff")
                if omitted_diff_path.is_file():
                    omitted_raw = omitted_diff_path.read_text(encoding="utf-8", errors="replace")
                    for rel_c, chunk in _split_diff_by_file(omitted_raw):
                        if rel_c and _matches_file_filter(rel_c, file_filter):
                            filtered_chunks.append(chunk)
                    full_diff = "".join(filtered_chunks)

        if focus_symbols and full_diff:
            full_diff = self._filter_diff_by_symbols(full_diff, focus_symbols)

        prov_per_file = (extract_diff_provenance(full_diff).get("per_file") or {}) if full_diff else {}
        file_stats = {
            fpath: {
                "added": int(meta.get("added_count") or 0),
                "removed": int(meta.get("removed_count") or 0),
            }
            for fpath, meta in prov_per_file.items()
        }

        pages = paginate_text_by_tokens(full_diff, page_tokens=page_tokens)
        total_pages = len(pages)
        clamped_page = max(1, min(int(page), total_pages))
        page_slice = "" if summary_only else pages[clamped_page - 1]
        effective_files = (
            list(file_stats.keys())
            if (file_filter or focus_symbols)
            else receipt.get("files_changed", [])
        )

        out: dict[str, Any] = {
            "job_id": safe_id,
            "status": receipt.get("status"),
            "file_filter": file_filter,
            "page": clamped_page,
            "total_pages": total_pages,
            "page_tokens_estimated": estimate_tokens(page_slice),
            "total_diff_tokens_estimated": estimate_tokens(full_diff),
            "files_changed": effective_files,
            "file_stats": file_stats,
            "diff": page_slice,
        }
        if summary_only:
            out["summary_only"] = True
        if focus_symbols:
            out["focus_symbols"] = list(focus_symbols)
        return out

    def delegate_apply(
        self,
        job_id: str,
        *,
        cleanup_worktree: bool = True,
        allow_scope_violations: bool = False,
        include_paths: list[str] | None = None,
        include_unverified_new_tests: bool = False,
    ) -> dict[str, Any]:
        """Apply a verified worker job's changes to the main workspace with serialized conflict detection."""
        safe_id = validate_job_id(job_id)
        with self._lock:
            fut = self._futures.get(safe_id)
        if fut is not None and not fut.done():
            raise RuntimeError(f"Job '{safe_id}' is still running; wait for completion before applying.")

        lock_file_path = self.store.data_dir / ".apply.lock"
        with self._apply_lock:
            with open(lock_file_path, "w", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    return self._delegate_apply_locked(
                        safe_id,
                        cleanup_worktree=cleanup_worktree,
                        allow_scope_violations=allow_scope_violations,
                        include_paths=include_paths,
                        include_unverified_new_tests=include_unverified_new_tests,
                    )
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _omitted_dir(self, safe_id: str) -> Path:
        d = self.store.data_dir / "omitted" / safe_id
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        return d

    def _archive_omitted_worktree_files(self, safe_id: str, wt_dir: Path, rel_paths: list[str]) -> None:
        if not rel_paths or not wt_dir.exists():
            return
        odir = self._omitted_dir(safe_id)
        wt_res = wt_dir.resolve()
        odir_res = odir.resolve()
        for rel in rel_paths:
            src = wt_dir / rel
            dst = odir / rel
            try:
                if not src.exists() or src.is_symlink():
                    continue
                if not src.resolve().is_relative_to(wt_res) or not dst.resolve().is_relative_to(odir_res):
                    continue
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    _copy2(src, dst)
            except OSError:
                continue

    def _delegate_apply_locked(
        self,
        safe_id: str,
        *,
        cleanup_worktree: bool = True,
        allow_scope_violations: bool = False,
        include_paths: list[str] | None = None,
        include_unverified_new_tests: bool = False,
    ) -> dict[str, Any]:
        receipt = self.store.read_receipt(safe_id)
        include_paths_norm: list[str] = validate_allowed_paths(include_paths or [])
        if receipt.get("applied"):
            wt_prev = Path(str(receipt.get("worktree_dir") or ""))
            odir_prev = self.store.data_dir / "omitted" / safe_id
            prev_applied = list(receipt.get("applied_files") or [])
            prev_omitted = list(receipt.get("omitted_untracked_out_of_scope") or [])
            prev_excl = list(receipt.get("excluded_runtime_artifacts") or [])
            prev_omitted_tests = list(receipt.get("omitted_unverified_new_test_files") or [])
            if include_paths_norm or include_unverified_new_tests:
                recoverable = list(
                    dict.fromkeys(
                        [*prev_omitted, *prev_excl, *prev_omitted_tests, *(receipt.get("files_changed") or [])]
                    )
                )
                to_recover = [
                    f
                    for f in recoverable
                    if (
                        (include_paths_norm and _matches_allowed_paths(f, include_paths_norm))
                        or (include_unverified_new_tests and f in prev_omitted_tests)
                    )
                    and (f not in prev_applied or not (self.workspace_root / f).exists())
                ]
                if to_recover:
                    allowed_prev = validate_allowed_paths(receipt.get("allowed_paths") or [])
                    for rel_inc in to_recover:
                        if not allow_scope_violations and allowed_prev and not _matches_allowed_paths(rel_inc, allowed_prev):
                            raise ValueError(
                                f"Refusing to apply out-of-scope file {rel_inc!r} (allowed_paths={allowed_prev}). "
                                f"Pass allow_scope_violations=True to delegate_apply if you have reviewed the diff and want to apply it."
                            )
                        dst_inc = self._safe_destination(rel_inc)
                        if not dst_inc.resolve().is_relative_to(self.workspace_root.resolve()) and not dst_inc.is_symlink():
                            raise ValueError(f"Path traversal blocked for {rel_inc!r}")
                        src_inc = (wt_prev / rel_inc) if (wt_prev.exists() and (wt_prev / rel_inc).exists()) else (odir_prev / rel_inc)
                        if not src_inc.exists() or not src_inc.is_file():
                            continue
                        inc_snap = self._snapshot_pre_apply(safe_id, [rel_inc])
                        pre_map = dict(receipt.get("pre_apply") or {})
                        pre_map.update(inc_snap)
                        receipt["pre_apply"] = pre_map
                        dst_inc.parent.mkdir(parents=True, exist_ok=True)
                        if dst_inc.is_symlink():
                            dst_inc.unlink()  # replace a link at the path; never copy through it
                        _copy2(src_inc, dst_inc)
                        if rel_inc not in prev_applied:
                            prev_applied.append(rel_inc)
                        hashes_map = dict(receipt.get("applied_file_hashes") or {})
                        hashes_map[rel_inc] = sha256_bytes(dst_inc.read_bytes())
                        receipt["applied_file_hashes"] = hashes_map
                        prov_map = dict(receipt.get("provenance") or {})
                        prov_files = list(prov_map.get("files_changed") or [])
                        if rel_inc not in prov_files:
                            prov_files.append(rel_inc)
                        prov_map["files_changed"] = prov_files
                        try:
                            raw_inc_bytes = dst_inc.read_bytes()
                            if b"\x00" in raw_inc_bytes[:8192]:
                                bin_list = list(prov_map.get("binary_files") or [])
                                if rel_inc not in bin_list:
                                    bin_list.append(rel_inc)
                                prov_map["binary_files"] = bin_list
                            else:
                                inc_lines = raw_inc_bytes.decode("utf-8", errors="replace").splitlines()
                                per_file_map = dict(prov_map.get("per_file") or {})
                                per_file_map[rel_inc] = {
                                    "added_line_sha256": [sha256_bytes(ln.encode("utf-8")) for ln in inc_lines],
                                    "removed_line_sha256": [],
                                }
                                prov_map["per_file"] = per_file_map
                        except OSError:
                            pass
                        receipt["provenance"] = prov_map
                        if rel_inc not in (receipt.get("files_changed") or []):
                            receipt["files_changed"] = [*list(receipt.get("files_changed") or []), rel_inc]
                    receipt["applied_files"] = prev_applied
                    rem_omitted = [f for f in prev_omitted if f not in set(to_recover)]
                    if rem_omitted:
                        receipt["omitted_untracked_out_of_scope"] = rem_omitted
                    else:
                        receipt.pop("omitted_untracked_out_of_scope", None)
                    rem_excl = [f for f in prev_excl if f not in set(to_recover)]
                    if rem_excl:
                        receipt["excluded_runtime_artifacts"] = rem_excl
                    else:
                        receipt.pop("excluded_runtime_artifacts", None)
                    rem_omitted_tests = [f for f in prev_omitted_tests if f not in set(to_recover)]
                    if rem_omitted_tests:
                        receipt["omitted_unverified_new_test_files"] = rem_omitted_tests
                    else:
                        receipt.pop("omitted_unverified_new_test_files", None)
                    self.store.write_receipt(
                        safe_id,
                        receipt,
                        scrub=bool(receipt.get("scrub_receipt", False)),
                        workspace_root=str(self.workspace_root),
                    )
            prov = receipt.get("provenance") or {}
            acc_list_prev = receipt.get("acceptance_results")
            real_prev = (
                receipt.get("acceptance_passed") is True
                and has_real_acceptance_pass(acc_list_prev)
            )
            return {
                "job_id": safe_id,
                "applied": True,
                "already_applied": True,
                "applied_files": receipt.get("applied_files", []),
                "diff_sha256": prov.get("diff_sha256"),
                "acceptance_passed": receipt.get("acceptance_passed"),
                **({"real_acceptance_passed": True} if real_prev else {}),
                **(
                    {"omitted_untracked_out_of_scope": list(receipt["omitted_untracked_out_of_scope"])}
                    if receipt.get("omitted_untracked_out_of_scope")
                    else {}
                ),
                **(
                    {"omitted_unverified_new_test_files": list(receipt["omitted_unverified_new_test_files"])}
                    if receipt.get("omitted_unverified_new_test_files")
                    else {}
                ),
            }

        status = receipt.get("status")
        wt_dir = Path(str(receipt.get("worktree_dir") or ""))
        baseline_sha = str(receipt.get("baseline_commit_sha") or "HEAD")
        if status == "running" and self._futures.get(safe_id) is None and wt_dir.exists():
            orphan_allowed = validate_allowed_paths(receipt.get("allowed_paths") or [])
            orphan_prompt = "\n".join(str(p) for p in (receipt.get("prompts") or []) if p)
            rec_diff, rec_files, rec_escapes = self._collect_worktree_diff(
                wt_dir, baseline_sha, allowed_paths=orphan_allowed, prompt_text=orphan_prompt
            )
            if rec_files and not rec_escapes:
                orphan_violations = [
                    f for f in rec_files
                    if orphan_allowed and not _matches_allowed_paths(f, orphan_allowed)
                ]
                self.store.diff_path(safe_id).write_text(rec_diff, encoding="utf-8")
                receipt["status"] = "scope_violation" if orphan_violations else "timeout"
                receipt["files_changed"] = rec_files
                receipt["symlink_escapes"] = rec_escapes
                receipt["scope_violations"] = orphan_violations
                status = receipt["status"]
        has_clean_timeout_edits = (
            status in ("timeout", "interrupted")
            and bool(receipt.get("files_changed"))
            and (not receipt.get("scope_violations") or allow_scope_violations)
            and not receipt.get("symlink_escapes")
        )
        has_approved_scope_override = (
            status == "scope_violation"
            and allow_scope_violations
            and not receipt.get("symlink_escapes")
        )
        if (
            status not in ("ok", "checkpoint", "needs_input")
            and not has_clean_timeout_edits
            and not has_approved_scope_override
        ):
            raise ValueError(
                f"Cannot apply job '{safe_id}' with status={status!r} "
                f"(error={receipt.get('error')!r}). Only ok / checkpoint / needs_input jobs "
                f"(or timeout jobs with clean edits, or scope_violation jobs with allow_scope_violations=True) can be applied."
            )

        omitted_unverified_new_tests: list[str] = list(receipt.get("omitted_unverified_new_test_files") or [])
        if include_unverified_new_tests and omitted_unverified_new_tests:
            for rel_ot in list(omitted_unverified_new_tests):
                if rel_ot not in (receipt.get("files_changed") or []):
                    receipt["files_changed"] = [*list(receipt.get("files_changed") or []), rel_ot]
            omitted_unverified_new_tests = []
            receipt.pop("omitted_unverified_new_test_files", None)
        elif (
            receipt.get("acceptance_passed") is not True
            and not include_unverified_new_tests
            and (
                receipt.get("acceptance_passed") is False
                or status != "ok"
                or bool(receipt.get("acceptance_commands"))
                or bool(receipt.get("acceptance_results"))
            )
        ):
            newly_omitted_tests = self._filter_unverified_new_tests_for_salvage(
                receipt,
                wt_dir,
                exclude_paths=include_paths_norm,
            )
            if newly_omitted_tests:
                omitted_unverified_new_tests = list(
                    dict.fromkeys([*omitted_unverified_new_tests, *newly_omitted_tests])
                )

        diff_path = self.store.diff_path(safe_id)
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""
        allowed_paths: list[str] = validate_allowed_paths(receipt.get("allowed_paths") or [])
        prompt_text_apply = "\n".join(str(p) for p in (receipt.get("prompts") or []) if p)
        raw_files_changed: list[str] = list(receipt.get("files_changed") or [])
        for rel_excl in list(receipt.get("excluded_runtime_artifacts") or []):
            if include_paths_norm and _matches_allowed_paths(rel_excl, include_paths_norm) and rel_excl not in raw_files_changed:
                raw_files_changed.append(rel_excl)
        for rel_ot in list(omitted_unverified_new_tests):
            if include_paths_norm and _matches_allowed_paths(rel_ot, include_paths_norm) and rel_ot not in raw_files_changed:
                raw_files_changed.append(rel_ot)
        ref_root_apply = wt_dir if wt_dir.exists() else self.workspace_root
        files_changed = [
            f
            for f in raw_files_changed
            if _explicitly_targets_runtime_artifact(
                f,
                allowed_paths,
                prompt_text=prompt_text_apply,
                ref_files=raw_files_changed,
                root=ref_root_apply,
            )
            or (include_paths_norm and _matches_allowed_paths(f, include_paths_norm))
            or not _is_ephemeral_build_or_test_artifact(
                f, root=ref_root_apply
            )
        ]
        baseline_hashes: dict[str, str | None] = self._compute_changed_baseline_hashes(
            wt_dir,
            baseline_sha,
            files_changed,
            dict(receipt.get("baseline_file_hashes") or {}),
        )
        omitted_untracked_out_of_scope: list[str] = []
        if allow_scope_violations and allowed_paths:
            in_scope_changed = [f for f in files_changed if _matches_allowed_paths(f, allowed_paths)]
            untracked_oos = [
                f
                for f in files_changed
                if not _matches_allowed_paths(f, allowed_paths)
                and not (include_paths_norm and _matches_allowed_paths(f, include_paths_norm))
                and baseline_hashes.get(f) is None
                and not (self.workspace_root / f).exists()
                and not _is_tracked_in_git(self.workspace_root, f)
                and not _is_referenced_by_files_or_prompt(
                    f,
                    in_scope_changed,
                    ref_root_apply,
                    prompt_text=prompt_text_apply,
                )
            ]
            if untracked_oos:
                omitted_untracked_out_of_scope = untracked_oos
                self._archive_omitted_worktree_files(safe_id, wt_dir, omitted_untracked_out_of_scope)
                omitted_set = set(omitted_untracked_out_of_scope)
                files_changed = [f for f in files_changed if f not in omitted_set]
                baseline_hashes = {k: v for k, v in baseline_hashes.items() if k not in omitted_set}
                if diff_text:
                    split_chunks = _split_diff_by_file(diff_text)
                    if split_chunks:
                        diff_text = "".join(
                            chunk for rel_c, chunk in split_chunks if rel_c not in omitted_set
                        )

        # 1. Pre-flight validation: scope, path traversal, symlinks, and baseline conflict check
        conflicted_files: list[str] = []
        for rel_path in files_changed:
            if not allow_scope_violations and not _matches_allowed_paths(rel_path, allowed_paths):
                raise ValueError(
                    f"Refusing to apply out-of-scope file {rel_path!r} (allowed_paths={allowed_paths}). "
                    f"Pass allow_scope_violations=True to delegate_apply if you have reviewed the diff and want to apply it."
                )
            dst_raw = self.workspace_root / rel_path
            dst_resolved = dst_raw.resolve()
            if not dst_resolved.is_relative_to(self.workspace_root):
                raise ValueError(f"Path traversal blocked for {rel_path!r}")

            if wt_dir.exists():
                src_raw = wt_dir / rel_path
                if src_raw.is_symlink():
                    link_target = (src_raw.parent / os.readlink(src_raw)).resolve()
                    if not link_target.is_relative_to(wt_dir.resolve()):
                        raise ValueError(
                            f"Refusing to apply symlink {rel_path!r} pointing outside worktree"
                        )

            expected_baseline_sha = baseline_hashes.get(rel_path)
            if dst_raw.exists() and dst_raw.is_file() and not dst_raw.is_symlink():
                current_host_sha = sha256_bytes(dst_raw.read_bytes())
                if expected_baseline_sha != current_host_sha:
                    # If the worker file already has the exact same bytes as the host file, no merge is needed.
                    if wt_dir.exists():
                        src_raw = wt_dir / rel_path
                        if src_raw.exists() and src_raw.is_file() and not src_raw.is_symlink():
                            if sha256_bytes(src_raw.read_bytes()) == current_host_sha:
                                continue
                    conflicted_files.append(rel_path)
            elif expected_baseline_sha is not None and not dst_raw.exists():
                conflicted_files.append(rel_path)

        # 2. If host files changed since baseline, attempt in-memory 3-way merge via `git merge-file -p`
        # without ever touching the host repository's git index.
        merged_payloads: dict[str, bytes] = {}
        skipped_ignored_conflicts: list[str] = []
        if conflicted_files:
            if not wt_dir.exists():
                raise ValueError(
                    f"Conflict detected: host files {conflicted_files} were modified since job '{safe_id}' started."
                )
            for rel_path in conflicted_files:
                dst_raw = self.workspace_root / rel_path
                src_raw = wt_dir / rel_path
                if not (dst_raw.exists() and dst_raw.is_file() and src_raw.exists() and src_raw.is_file()):
                    raise ValueError(
                        f"Conflict detected: host file {rel_path!r} was modified or deleted since job '{safe_id}' started."
                    )
                base_show = _git_raw(
                    ["show", f"{baseline_sha}:{rel_path}"],
                    cwd=str(wt_dir),
                    capture_output=True,
                    check=False,
                )
                if base_show.returncode != 0:
                    if _is_gitignored_on_host(self.workspace_root, rel_path):
                        # Generated output (gitignored) written by both sides, e.g. a test report:
                        # keep the host copy rather than failing the whole apply.
                        skipped_ignored_conflicts.append(rel_path)
                        continue
                    raise ValueError(
                        f"Conflict detected: host file {rel_path!r} was concurrently created on both host and worker."
                    )
                with tempfile.TemporaryDirectory(prefix="cc-agy-merge-") as mdir:
                    mpath = Path(mdir)
                    f_cur = mpath / "current"
                    f_base = mpath / "base"
                    f_wrk = mpath / "worker"
                    f_cur.write_bytes(dst_raw.read_bytes())
                    f_base.write_bytes(base_show.stdout)
                    f_wrk.write_bytes(src_raw.read_bytes())
                    merge_res = _git_raw(
                        ["merge-file", "-p", str(f_cur), str(f_base), str(f_wrk)],
                        capture_output=True,
                        check=False,
                    )
                    if merge_res.returncode != 0:
                        raise ValueError(
                            f"Conflict detected: host files {conflicted_files} were modified since job "
                            f"'{safe_id}' started and could not be cleanly merged."
                        )
                    merged_payloads[rel_path] = merge_res.stdout

        if skipped_ignored_conflicts:
            files_changed = [f for f in files_changed if f not in skipped_ignored_conflicts]

        # Record each target's pre-apply state (bytes backed up under data_dir/backups/<job_id>/) so
        # delegate_revert can restore the workspace exactly, including uncommitted host edits.
        pre_apply = self._snapshot_pre_apply(safe_id, files_changed)

        applied_files: list[str] = []
        odir_apply = self.store.data_dir / "omitted" / safe_id
        if wt_dir.exists() or odir_apply.exists():
            # All-or-nothing: if any write fails part way, every file already touched goes back to its
            # pre_apply backup before the error propagates, so the workspace is never half applied.
            from pathlib import PurePosixPath

            from server.batch_plan import is_protected_path
            from server.writers import symlink_component

            in_progress: str | None = None
            ws_resolved = self.workspace_root.resolve()
            try:
                for rel_path in files_changed:
                    in_progress = rel_path
                    src_raw = (
                        (wt_dir / rel_path)
                        if (wt_dir.exists() and ((wt_dir / rel_path).exists() or (wt_dir / rel_path).is_symlink()))
                        else (odir_apply / rel_path)
                    )
                    dst_raw = self.workspace_root / rel_path
                    # Never write through links in the host tree. Git never tracks a path beyond a
                    # symlinked directory, so a linked parent (even one pointing inside the workspace,
                    # such as an alias of .git/hooks) means the path is not what the diff describes.
                    # A link at the path itself is replaced (pre_apply recorded it).
                    parent_rel = PurePosixPath(rel_path).parent.as_posix()
                    if parent_rel not in ("", ".") and symlink_component(self.workspace_root, parent_rel) is not None:
                        raise ValueError(f"refusing to apply {rel_path!r}: a parent directory in the workspace is a symlink")
                    if not dst_raw.parent.resolve().is_relative_to(ws_resolved):
                        raise ValueError(f"refusing to apply {rel_path!r}: its directory resolves outside the workspace")
                    if src_raw.is_symlink():
                        link_dest = (src_raw.parent / os.readlink(src_raw)).resolve()
                        if wt_dir.exists() and link_dest.is_relative_to(wt_dir.resolve()):
                            link_rel = link_dest.relative_to(wt_dir.resolve()).as_posix()
                            if is_protected_path(link_rel):
                                raise ValueError(f"refusing to apply symlink {rel_path!r}: it points at protected {link_rel!r}")
                    if dst_raw.is_symlink() and not src_raw.is_symlink():
                        dst_raw.unlink()
                    if rel_path in merged_payloads:
                        dst_raw.parent.mkdir(parents=True, exist_ok=True)
                        dst_raw.write_bytes(merged_payloads[rel_path])
                        applied_files.append(rel_path)
                    elif src_raw.is_symlink():
                        dst_raw.parent.mkdir(parents=True, exist_ok=True)
                        if dst_raw.exists() or dst_raw.is_symlink():
                            dst_raw.unlink()
                        os.symlink(os.readlink(src_raw), dst_raw)
                        applied_files.append(rel_path)
                    elif src_raw.exists() and src_raw.is_file():
                        dst_raw.parent.mkdir(parents=True, exist_ok=True)
                        _copy2(src_raw, dst_raw)
                        applied_files.append(rel_path)
                    elif wt_dir.exists() and not (wt_dir / rel_path).exists():
                        if dst_raw.exists() or dst_raw.is_symlink():
                            dst_raw.unlink()
                        applied_files.append(rel_path)
                in_progress = None
            except BaseException as exc:
                touched = list(dict.fromkeys([*applied_files, *([in_progress] if in_progress else [])]))
                failed_restores = self._rollback_partial_apply(safe_id, pre_apply, touched)
                if hasattr(exc, "add_note"):  # Python 3.11+; the plugin still supports 3.10
                    exc.add_note(
                        f"delegate_apply rolled back {len(touched)} already-written file(s) to their pre-apply state"
                        + (f"; could not restore {failed_restores}" if failed_restores else "")
                    )
                raise
        elif diff_text.strip():
            _git_raw(
                ["apply", "--binary", "--whitespace=nowarn", "-"],
                input=diff_text,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                check=True,
            )
            applied_files = list(files_changed)

        if cleanup_worktree and wt_dir.exists():
            self._cleanup_worktree(wt_dir)

        # Compute exact provenance and post-apply SHA-256 hashes of the applied files
        applied_prov = extract_diff_provenance(diff_text)
        if skipped_ignored_conflicts:
            applied_prov["files_changed"] = [
                f for f in applied_prov.get("files_changed", []) if f not in skipped_ignored_conflicts
            ]
        applied_file_hashes: dict[str, str | None] = {}
        for fpath in applied_files:
            if fpath not in applied_prov["files_changed"]:
                applied_prov["files_changed"].append(fpath)
            dst_p = self.workspace_root / fpath
            if dst_p.is_symlink():
                applied_file_hashes[fpath] = f"symlink:{os.readlink(dst_p)}"
            elif dst_p.exists() and dst_p.is_file():
                raw_applied_bytes = dst_p.read_bytes()
                applied_file_hashes[fpath] = sha256_bytes(raw_applied_bytes)
                if (
                    baseline_hashes.get(fpath) is None
                    and fpath not in (applied_prov.get("per_file") or {})
                    and fpath not in (applied_prov.get("binary_files") or [])
                ):
                    if b"\x00" in raw_applied_bytes[:8192]:
                        bin_list = list(applied_prov.get("binary_files") or [])
                        if fpath not in bin_list:
                            bin_list.append(fpath)
                        applied_prov["binary_files"] = bin_list
                    else:
                        inc_lines = raw_applied_bytes.decode("utf-8", errors="replace").splitlines()
                        per_file_map = dict(applied_prov.get("per_file") or {})
                        per_file_map[fpath] = {
                            "added_line_sha256": [sha256_bytes(ln.encode("utf-8")) for ln in inc_lines],
                            "removed_line_sha256": [],
                        }
                        applied_prov["per_file"] = per_file_map
            else:
                applied_file_hashes[fpath] = None

        receipt["applied"] = True
        receipt["applied_at_epoch"] = round(time.time(), 3)
        receipt["applied_files"] = applied_files
        receipt["applied_file_hashes"] = applied_file_hashes
        receipt["baseline_file_hashes"] = baseline_hashes
        receipt["provenance"] = applied_prov
        receipt["pre_apply"] = {k: v for k, v in pre_apply.items() if k in applied_files}
        if omitted_untracked_out_of_scope:
            receipt["omitted_untracked_out_of_scope"] = omitted_untracked_out_of_scope
        if omitted_unverified_new_tests:
            rem_omitted_tests_init = [
                f for f in omitted_unverified_new_tests if f not in set(applied_files)
            ]
            if rem_omitted_tests_init:
                receipt["omitted_unverified_new_test_files"] = rem_omitted_tests_init
                omitted_unverified_new_tests = rem_omitted_tests_init
            else:
                receipt.pop("omitted_unverified_new_test_files", None)
                omitted_unverified_new_tests = []
        if receipt.get("excluded_runtime_artifacts"):
            rem_excl_init = [
                f for f in list(receipt.get("excluded_runtime_artifacts") or []) if f not in set(applied_files)
            ]
            if rem_excl_init:
                receipt["excluded_runtime_artifacts"] = rem_excl_init
            else:
                receipt.pop("excluded_runtime_artifacts", None)
        failing_new_test_files: list[str] = []
        acc_list_apply = receipt.get("acceptance_results")
        if receipt.get("acceptance_passed") is False and isinstance(acc_list_apply, list):
            new_test_candidates = [
                f
                for f in applied_files
                if _is_test_file_path(f)
                and baseline_hashes.get(f) is None
                and (pre_apply.get(f) or {}).get("state") == "absent"
            ]
            for ntf in new_test_candidates:
                ntf_name = Path(ntf).name
                if any(
                    isinstance(r, dict)
                    and (not r.get("passed") or r.get("timed_out"))
                    and (
                        ntf in str(r.get("command") or "")
                        or ntf_name in str(r.get("command") or "")
                        or ntf in str(r.get("output_tail") or "")
                        or ntf_name in str(r.get("output_tail") or "")
                    )
                    for r in acc_list_apply
                ):
                    failing_new_test_files.append(ntf)
            if failing_new_test_files:
                receipt["failing_new_test_files"] = failing_new_test_files
        self.store.write_receipt(
            safe_id,
            receipt,
            scrub=bool(receipt.get("scrub_receipt", False)),
            workspace_root=str(self.workspace_root),
        )

        real_apply_pass = (
            receipt.get("acceptance_passed") is True
            and has_real_acceptance_pass(acc_list_apply)
        )
        return {
            "job_id": safe_id,
            "applied": True,
            "applied_files": applied_files,
            **({"kept_host_copy": skipped_ignored_conflicts} if skipped_ignored_conflicts else {}),
            **(
                {"omitted_untracked_out_of_scope": omitted_untracked_out_of_scope}
                if omitted_untracked_out_of_scope
                else {}
            ),
            **(
                {
                    "omitted_unverified_new_test_files": omitted_unverified_new_tests,
                    "omitted_unverified_new_tests_warning": (
                        f"Omitted unverified newly created test file(s) {omitted_unverified_new_tests} "
                        "because they failed or timed out during verification. "
                        f"Inspect with get_diff(job_id={safe_id!r}, file_filter=...) or pass "
                        "include_unverified_new_tests=True / include_paths=[...] to delegate_apply if needed."
                    ),
                }
                if omitted_unverified_new_tests
                else {}
            ),
            **(
                {
                    "failing_new_test_files": failing_new_test_files,
                    "failing_new_tests_warning": (
                        f"Newly created test file(s) {failing_new_test_files} failed in acceptance_results. "
                        "Inspect and fix or remove broken worker-authored test assertions before concluding."
                    ),
                }
                if failing_new_test_files
                else {}
            ),
            "diff_sha256": applied_prov["diff_sha256"],
            "acceptance_passed": receipt.get("acceptance_passed"),
            **({"real_acceptance_passed": True} if real_apply_pass else {}),
            **(
                {"contract_warnings": list(receipt["contract_warnings"])}
                if receipt.get("contract_warnings")
                else {}
            ),
            **(
                {"baseline_acceptance_warnings": list(receipt["baseline_acceptance_warnings"])}
                if receipt.get("baseline_acceptance_warnings")
                else {}
            ),
            "revert_hint": f"delegate_revert(job_id={safe_id!r}) restores these files to their pre-apply state.",
        }

    def _escalation_fields(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Count consecutive failed turns and, at the threshold, mark the job for Lead (Opus) takeover."""
        try:
            prior = self.store.read_receipt(job_id)
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
            prior = {}
        prior_failed = int(prior.get("failed_attempts") or 0)
        status = str(payload.get("status") or "")
        acc_passed = payload.get("acceptance_passed")
        failed = worker_turn_failed(status, acc_passed) or (
            acc_passed is not True
            and status in ("checkpoint", "timeout", "scope_violation", "failed", "error", "spec_conflict")
        )
        # A plugin same-model retry is the plugin's own extra turn: a failed retry must not use up the
        # Lead's failure allowance, or the Lead's first targeted delegate_continue gets refused and
        # the expensive fallback (Opus writing the code itself) becomes the only path.
        sources = [str(s) for s in (payload.get("prompt_sources") or []) if s]
        is_plugin_retry_turn = bool(sources) and sources[-1] == "plugin_retry"
        if failed:
            count = prior_failed if is_plugin_retry_turn else prior_failed + 1
        else:
            count = 0
        out: dict[str, Any] = {"failed_attempts": count}
        threshold = escalate_after_failures()
        if status == "error" and is_quota_or_rate_limit_error(str(payload.get("error") or "")):
            out["escalate_to_lead"] = True
            out["escalation"] = {
                "escalate_to_lead": True,
                "reason": "quota_or_rate_limit",
                "failed_attempts": count,
                "threshold": threshold,
                "last_status": payload.get("status"),
                "last_acceptance_passed": acc_passed,
                "recommendation": (
                    "Gemini hit a rate limit or quota error (429/503). Do not retry the worker right now: "
                    f"inspect get_diff(job_id={job_id!r}) and finish the change directly on the host agent, "
                    "or pass force_retry=true to delegate_continue after switching credentials."
                ),
            }
            return out
        unverified_stop = (
            acc_passed is not True
            and status in ("ok", "checkpoint", "timeout", "scope_violation", "failed", "error", "spec_conflict")
            and (
                status != "ok"
                or acc_passed is False
                or bool(payload.get("vacuous_acceptance"))
                or bool(payload.get("test_only_diff"))
            )
        )
        session_lead_slices = 0
        near_session_deadline = False
        if (
            unverified_stop
            and threshold > 0
            and (
                self._is_bounded_lead_session()
                or is_strict_mode_enabled()
                or resolve_cost_profile() == "economy"
            )
        ):
            session_lead_slices = sum(1 for s in sources if s == "lead")
            session_deadline, session_elapsed, _rem_w = self._session_deadline_budget()
            if session_deadline is not None and (session_deadline - session_elapsed) < 150.0:
                near_session_deadline = True
            session_since = self._session_since_epoch()
            for r_prev in self.store.list_receipts(workspace_root=self.workspace_root):
                if r_prev.get("kind", "delegate") != "delegate" or str(r_prev.get("job_id") or "") == job_id:
                    continue
                r_started = float(r_prev.get("started_at_epoch") or 0.0)
                if r_started > 0.0 and r_started < session_since:
                    continue
                if r_prev.get("cancelled") or r_prev.get("status") == "cancelled":
                    continue
                if (
                    r_prev.get("status") == "ok"
                    and r_prev.get("acceptance_passed") is True
                    and not r_prev.get("scope_violations")
                    and not r_prev.get("vacuous_acceptance")
                    and not r_prev.get("test_only_diff")
                ):
                    continue
                session_lead_slices += sum(1 for s in (r_prev.get("prompt_sources") or []) if s == "lead")
        should_escalate = bool(
            threshold > 0
            and (
                count >= threshold
                or (
                    unverified_stop
                    and (session_lead_slices >= max(2, threshold) or near_session_deadline)
                )
            )
        )
        if should_escalate:
            eff_count = max(count, session_lead_slices) if session_lead_slices >= 2 else max(1, count)
            has_partial_files = bool(payload.get("files_changed"))
            recommendation = (
                f"The Gemini worker failed {eff_count} attempt(s) in a row after modifying "
                f"{len(payload.get('files_changed') or [])} file(s). Do NOT call delegate_cancel (which discards "
                f"partial work): inspect get_diff(job_id={job_id!r}), then either call "
                f"delegate_apply(job_id={job_id!r}, allow_scope_violations=true) to land the partial changes and "
                f"finish with up to 3 targeted Edit calls on those files, or call "
                f"delegate_continue(job_id={job_id!r}, follow_up_prompt=..., force_retry=true). "
                "escalate_to_lead=true: Host Edit, Write, Read, Grep, Glob, and test commands are now unlocked "
                "so you can apply small targeted fixes directly without spawning another worker. "
                "delegate_continue refuses further Gemini attempts on this job unless force_retry=true."
                if has_partial_files
                else (
                    f"The Gemini worker failed {eff_count} attempt(s) in a row without landing passing edits. "
                    f"Stop delegating blindly: inspect why the worker stalled, then either apply up to 3 targeted "
                    f"Edit calls directly on the target file(s) or call "
                    f"delegate_continue(job_id={job_id!r}, follow_up_prompt=..., force_retry=true). "
                    "escalate_to_lead=true: Host Edit, Write, Read, Grep, Glob, and test commands are now unlocked "
                    "so you can apply small targeted fixes directly without spawning another worker. "
                    "delegate_continue refuses further Gemini attempts on this job unless force_retry=true."
                )
            )
            out["escalate_to_lead"] = True
            out["escalation"] = {
                "escalate_to_lead": True,
                "failed_attempts": eff_count,
                "threshold": threshold,
                "last_status": payload.get("status"),
                "last_acceptance_passed": acc_passed,
                **(
                    {"reason": "session_deadline_near"}
                    if near_session_deadline and eff_count < (threshold or 2)
                    else {}
                ),
                "recommendation": recommendation,
            }
        return out

    def _backup_dir(self, safe_id: str) -> Path:
        d = self.store.data_dir / "backups" / safe_id
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        return d

    def _snapshot_pre_apply(self, safe_id: str, rel_paths: list[str]) -> dict[str, dict[str, Any]]:
        """Back up the host bytes of every file an apply is about to touch."""
        snap: dict[str, dict[str, Any]] = {}
        bdir: Path | None = None
        for rel in rel_paths:
            p = self.workspace_root / rel
            if p.is_symlink():
                snap[rel] = {"state": "symlink", "target": os.readlink(p), "sha256": f"symlink:{os.readlink(p)}"}
            elif p.exists() and p.is_file():
                data = p.read_bytes()
                bdir = bdir or self._backup_dir(safe_id)
                name = sha256_bytes(rel.encode("utf-8"))[:24]
                (bdir / name).write_bytes(data)
                snap[rel] = {"state": "file", "sha256": sha256_bytes(data), "backup": name,
                             "mode": p.stat().st_mode & 0o7777}
            else:
                snap[rel] = {"state": "absent", "sha256": None}
        return snap

    def _safe_destination(self, rel: str) -> Path:
        """Workspace path for a write, refusing one whose parent directory is a symlink (git never tracks
        a path beyond a symlinked directory, so it cannot be what a job recorded; an alias of .git/hooks
        would otherwise receive the write) or resolves outside the workspace."""
        from pathlib import PurePosixPath

        from server.writers import symlink_component

        parent_rel = PurePosixPath(rel).parent.as_posix()
        if parent_rel not in ("", ".") and symlink_component(self.workspace_root, parent_rel) is not None:
            raise ValueError(f"refusing to write {rel!r}: a parent directory in the workspace is a symlink")
        dst = self.workspace_root / rel
        if not dst.parent.resolve().is_relative_to(self.workspace_root.resolve()):
            raise ValueError(f"refusing to write {rel!r}: its directory resolves outside the workspace")
        return dst

    def _restore_pre_state(self, bdir: Path, rel: str, pre: dict[str, Any]) -> None:
        """Put one workspace path back to a recorded pre-apply state (absent, symlink, file or git blob)."""
        dst = self._safe_destination(rel)
        state = pre.get("state")
        if dst.is_symlink() or (dst.exists() and state in ("absent", "symlink")):
            dst.unlink()
        if state == "absent":
            pass
        elif state == "symlink":
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(pre.get("target")), dst)
        elif state == "file":
            data = (bdir / str(pre.get("backup"))).read_bytes()
            if sha256_bytes(data) != pre.get("sha256"):
                raise ValueError(f"Backup for {rel!r} is corrupt (hash mismatch); refusing to restore it.")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_file():
                dst.unlink()  # replace rather than overwrite, so a read-only applied file is no obstacle
            dst.write_bytes(data)
            if pre.get("mode") is not None:
                os.chmod(dst, int(pre["mode"]))
        elif state == "git_blob":
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_file():
                dst.unlink()
            dst.write_bytes(pre["data"])
            if pre.get("mode") is not None:
                os.chmod(dst, int(pre["mode"]))

    def _rollback_partial_apply(self, safe_id: str, pre_apply: dict[str, dict[str, Any]], rels: list[str]) -> list[str]:
        """Best-effort undo of a failed apply; returns the paths that could not be restored."""
        bdir = self.store.data_dir / "backups" / safe_id
        failed: list[str] = []
        for rel in reversed(rels):
            pre = pre_apply.get(rel)
            if not pre:
                failed.append(rel)
                continue
            try:
                self._restore_pre_state(bdir, rel, pre)
            except (OSError, ValueError):
                failed.append(rel)
        return failed

    def delegate_revert(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
        skip_dirty: bool = False,
    ) -> dict[str, Any]:
        """Restore every file an applied job touched to its pre-apply state.

        Refuses (or skips with skip_dirty=True) any file whose current content no longer matches
        what the job applied, so later human or agent edits are never silently overwritten.
        """
        safe_id = validate_job_id(job_id)
        lock_file_path = self.store.data_dir / ".apply.lock"
        with self._apply_lock:
            with open(lock_file_path, "w", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    return self._delegate_revert_locked(safe_id, dry_run=dry_run, skip_dirty=skip_dirty)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _pre_state_for_in_place(self, receipt: dict[str, Any], rel: str) -> dict[str, Any]:
        base_hash = (receipt.get("baseline_file_hashes") or {}).get(rel)
        if base_hash is None:
            return {"state": "absent", "sha256": None}
        sha = str(receipt.get("baseline_commit_sha") or "HEAD")
        show = _git_raw(["show", f"{sha}:{rel}"], cwd=str(self.workspace_root),
                        capture_output=True, check=False)
        if show.returncode != 0 or sha256_bytes(show.stdout) != base_hash:
            return {"state": "unknown", "sha256": base_hash}
        blob_mode = 0o644
        tree_res = _git_raw(
            ["ls-tree", sha, "--", rel],
            cwd=str(self.workspace_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if tree_res.returncode == 0 and tree_res.stdout:
            head_tok = tree_res.stdout.strip().split(None, 1)[0]
            if head_tok == "100755":
                blob_mode = 0o755
            elif head_tok == "100644":
                blob_mode = 0o644
        return {"state": "git_blob", "sha256": base_hash, "data": show.stdout, "mode": blob_mode}

    def _delegate_revert_locked(self, safe_id: str, *, dry_run: bool, skip_dirty: bool) -> dict[str, Any]:
        receipt = self.store.read_receipt(safe_id)
        if not self.store.verify_receipt_signature(receipt):
            raise ValueError(f"Receipt for job '{safe_id}' failed its HMAC check; refusing to revert from it.")
        if not receipt.get("applied"):
            raise ValueError(f"Job '{safe_id}' was never applied; nothing to revert (use delegate_cancel).")
        already = list(receipt.get("reverted_files") or [])
        if receipt.get("reverted") and not receipt.get("revert_skipped"):
            return {"job_id": safe_id, "reverted": True, "already_reverted": True,
                    "reverted_files": already}
        already_set = set(already)
        applied_hashes: dict[str, str | None] = dict(receipt.get("applied_file_hashes") or {})
        pre_apply: dict[str, dict[str, Any]] = dict(receipt.get("pre_apply") or {})
        in_place = bool(receipt.get("in_place"))
        bdir = self.store.data_dir / "backups" / safe_id

        plan: list[dict[str, Any]] = []
        dirty: list[dict[str, Any]] = []
        for rel in list(receipt.get("applied_files") or []):
            if rel in already_set:
                # Restored by an earlier skip_dirty revert; only the files skipped then are left to do.
                continue
            try:
                dst = self._safe_destination(rel)
            except ValueError as exc:
                dirty.append({"file": rel, "reason": str(exc)})
                continue
            current = _hash_path(dst)
            expected = applied_hashes.get(rel)
            pre = pre_apply.get(rel) or (self._pre_state_for_in_place(receipt, rel) if in_place else {"state": "unknown"})
            if current != expected:
                diff = ""
                if dst.is_file() and not dst.is_symlink():
                    diff = f"current sha256={current}, applied sha256={expected}"
                dirty.append({"file": rel, "reason": "modified since apply", "detail": diff})
                continue
            if pre.get("state") == "unknown":
                dirty.append({"file": rel, "reason": "pre-apply content not recorded (older receipt or dirty in_place baseline)"})
                continue
            plan.append({"file": rel, "restore_to": pre.get("state"), "pre": pre})

        if dirty and not skip_dirty:
            return {
                "job_id": safe_id,
                "reverted": False,
                "refused": True,
                "dirty_files": dirty,
                "would_restore": [{"file": x["file"], "restore_to": x["restore_to"]} for x in plan],
                "message": "Some files changed after this job was applied. Nothing was written. "
                           "Pass skip_dirty=True to revert only the clean files.",
            }
        if dry_run:
            return {
                "job_id": safe_id,
                "dry_run": True,
                "would_restore": [{"file": x["file"], "restore_to": x["restore_to"]} for x in plan],
                "skipped_dirty": dirty,
            }

        reverted: list[str] = []
        for item in plan:
            self._restore_pre_state(bdir, item["file"], item["pre"])
            reverted.append(item["file"])

        # A skip_dirty revert is partial: the job stays revertable so the skipped files can be
        # restored later, once they match what the job applied again.
        receipt["reverted"] = not dirty
        receipt["partially_reverted"] = bool(dirty)
        receipt["reverted_at_epoch"] = round(time.time(), 3)
        receipt["reverted_files"] = list(dict.fromkeys([*already, *reverted]))
        receipt["revert_skipped"] = dirty
        self.store.write_receipt(
            safe_id,
            receipt,
            scrub=bool(receipt.get("scrub_receipt", False)),
            workspace_root=str(self.workspace_root),
        )
        return {"job_id": safe_id, "reverted": not dirty, "partially_reverted": bool(dirty),
                "reverted_files": reverted, "skipped_dirty": dirty}

    def _should_auto_apply(self, auto_apply: bool | None) -> bool:
        from server.regions import resolve_auto_apply_on_pass, resolve_auto_apply_override

        user_override = resolve_auto_apply_override()
        if user_override is False:
            return False
        if auto_apply is not None:
            return bool(auto_apply)
        return resolve_auto_apply_on_pass()

    def _maybe_auto_apply_digest(
        self,
        job_id: str,
        digest: dict[str, Any],
        *,
        auto_apply: bool | None,
        in_place: bool,
    ) -> dict[str, Any]:
        if not self._should_auto_apply(auto_apply) or in_place or digest.get("applied"):
            return digest
        escalated = bool((digest.get("escalation") or {}).get("escalate_to_lead"))
        if escalated:
            digest["auto_apply_skipped"] = "escalated to the Lead after repeated failed attempts"
            return digest
        if digest.get("acceptance_passed") is False:
            digest["auto_apply_skipped"] = "acceptance checks failed"
            return digest
        qa_info = digest.get("qa_review")
        if isinstance(qa_info, dict) and qa_info.get("passed") is False and not qa_info.get("resumed"):
            digest["auto_apply_skipped"] = "qa_review reported NEEDS_FIXES without a fix turn"
            return digest
        if digest.get("missing_prompt_symbols"):
            digest["auto_apply_skipped"] = f"missing prompt symbols: {digest['missing_prompt_symbols']}"
            return digest
        if digest.get("vacuous_acceptance"):
            digest["auto_apply_skipped"] = "acceptance checks were vacuous (no modified files or 0 tests collected)"
            return digest
        if digest.get("test_only_diff"):
            digest["auto_apply_skipped"] = (
                "worker modified only test files without changing any production source files"
            )
            return digest
        stub_warns = [str(w) for w in (digest.get("stub_mutation_warnings") or []) if w]
        if stub_warns:
            has_stub_mock_warn = any(w.startswith("ExistingTestStubsModified:") for w in stub_warns)
            if has_stub_mock_warn:
                digest["auto_apply_skipped"] = (
                    "worker modified mock/stub expectations across >=3 existing test blocks; inspect get_diff before applying"
                )
                return digest
            qa_verified = isinstance(qa_info, dict) and qa_info.get("passed") is True
            escalated_authorized_test_update = False
            if digest.get("escalated_from_model") and not qa_verified:
                try:
                    full_r = self.store.read_receipt(job_id) or {}
                except Exception:
                    full_r = {}
                intent_txt = "\n".join(
                    str(x)
                    for x in [full_r.get("raw_user_prompt") or "", *((full_r.get("prompts") or [])[:1])]
                    if x
                )
                rec_allowed = [
                    str(ap).replace("\\", "/").strip("/")
                    for ap in (full_r.get("allowed_paths") or digest.get("allowed_paths") or [])
                    if ap
                ]
                changed_tests = [
                    str(fc).replace("\\", "/").strip("/")
                    for fc in (digest.get("files_changed") or [])
                    if _is_test_file_path(str(fc))
                ]
                changed_prod = [
                    str(fc).replace("\\", "/").strip("/")
                    for fc in (digest.get("files_changed") or [])
                    if not _is_test_file_path(str(fc))
                ]
                explicit_test_paths_allowed = bool(changed_tests) and all(
                    ct in rec_allowed for ct in changed_tests
                )
                has_substantive_prod_edit = False
                if changed_prod:
                    dp = self.store.diff_path(job_id)
                    dt = dp.read_text(encoding="utf-8", errors="replace") if dp.exists() else ""
                    in_prod_file = False
                    for dln in dt.splitlines():
                        if dln.startswith("diff --git "):
                            hdr = parse_diff_git_header(dln)
                            cp = ((hdr[1] or hdr[0]) if hdr else "").replace("\\", "/").strip("/")
                            in_prod_file = cp in changed_prod
                            continue
                        if in_prod_file and dln.startswith(("+", "-")) and not dln.startswith(("+++", "---")):
                            code_s = dln[1:].strip()
                            if code_s and not code_s.startswith(("#", "//", "/*", "*", "*/")):
                                has_substantive_prod_edit = True
                                break
                escalated_authorized_test_update = (
                    explicit_test_paths_allowed
                    and has_substantive_prod_edit
                    and not _prompt_forbids_existing_test_updates(intent_txt)
                )
            if not qa_verified and not escalated_authorized_test_update:
                digest["auto_apply_skipped"] = (
                    "worker altered pre-existing test expectations across >=2 test functions; inspect get_diff before applying"
                )
                return digest
        if digest.get("status") == "spec_conflict":
            digest["auto_apply_skipped"] = "worker reported a specification conflict requiring Lead resolution"
            return digest
        if (
            digest.get("status") == "ok"
            and digest.get("files_changed")
            and digest.get("acceptance_passed") is not False
            and not digest.get("scope_violations")
        ):
            apply_res = self.delegate_apply(job_id)
            digest["applied"] = True
            digest["applied_files"] = apply_res.get("applied_files", [])
            digest["diff_sha256"] = apply_res.get("diff_sha256")
            if apply_res.get("kept_host_copy"):
                digest["kept_host_copy"] = apply_res["kept_host_copy"]
            real_acc_passed = False
            has_worker_caveat = False
            raw_summary_text = str(digest.get("summary") or "").strip()
            if re.search(r"\bBLOCKED\b", raw_summary_text, re.I) or "?" in raw_summary_text:
                has_worker_caveat = True
            for p_item in (digest.get("progress") or digest.get("recent_progress") or []):
                if isinstance(p_item, dict) and str(p_item.get("kind") or "").lower() in ("question", "blocker"):
                    has_worker_caveat = True
            if digest.get("acceptance_passed") is True:
                try:
                    full_rec = self.store.read_receipt(job_id)
                    real_acc_passed = satisfies_auto_apply_verification(
                        full_rec.get("acceptance_results"),
                        True,
                    )
                except Exception:
                    real_acc_passed = False
            if real_acc_passed:
                digest["real_acceptance_passed"] = True
                if has_worker_caveat:
                    digest["has_worker_caveat"] = True
                verbose_applied = (
                    env_flag("CC_AGY_VERBOSE_APPLIED_DIGEST") is True
                    or "CC_AGY_APPLIED_DIFF_PREVIEW_TOKENS" in os.environ
                )
                if verbose_applied:
                    applied_preview_budget = _env_int("CC_AGY_APPLIED_DIFF_PREVIEW_TOKENS", 250)
                    if applied_preview_budget > 0 and isinstance(digest.get("diff_preview"), str):
                        short_prev, was_trunc, _ = truncate_to_token_budget(
                            digest["diff_preview"], applied_preview_budget
                        )
                        digest["diff_preview"] = short_prev
                        if was_trunc:
                            digest["diff_preview_complete"] = False
                else:
                    one_line_summary = " ".join(raw_summary_text.split())
                    if one_line_summary:
                        if has_worker_caveat:
                            if len(one_line_summary) > 240:
                                head_s = one_line_summary[:240]
                                if not (re.search(r"\bBLOCKED\b", head_s, re.I) or "?" in head_s):
                                    one_line_summary = f"{one_line_summary[:90]} … {one_line_summary[-140:]}"
                                else:
                                    one_line_summary = f"{head_s}…"
                        elif len(one_line_summary) > 140:
                            one_line_summary = f"{one_line_summary[:140]}…"
                    for heavy_key in (
                        "kind",
                        "diff_preview",
                        "diff_preview_complete",
                        "diff_tokens_estimated",
                        "summary",
                        "summary_truncated",
                        "summary_tokens_estimated",
                        "usage",
                        "cost",
                        "acceptance_results",
                        "recent_progress",
                        "thoughts_count",
                        "budget",
                        "containment",
                        "containment_warning",
                        "vertex_location",
                        "reroute_note",
                        "auth_mode",
                        "allowed_paths",
                        "thinking_level",
                        "tool_call_count",
                        "stop_reason",
                        "error",
                        "failed_attempts",
                    ):
                        digest.pop(heavy_key, None)
                    if not has_worker_caveat:
                        digest.pop("progress", None)
                    if one_line_summary:
                        digest["summary"] = one_line_summary
                    if not digest.get("in_place"):
                        digest.pop("in_place", None)
                    if not digest.get("scope_violations"):
                        digest.pop("scope_violations", None)
                if digest.get("contract_warnings"):
                    digest["next_action"] = (
                        f"Changes were applied, but contract_warnings flagged potential class-attribute shadowing. "
                        f"Call get_diff(job_id={job_id!r}, focus_symbols=[...]) on the flagged class/attribute; "
                        f"if unrequested by the task specification, fix it via a targeted delegate_start follow-up job "
                        f"before finishing."
                    )
                elif not has_worker_caveat:
                    digest["next_action"] = (
                        "Applied after acceptance passed. Finish now in one concise sentence without host reads or pytest."
                    )
                else:
                    digest.pop("next_action", None)
            else:
                # Applied without an explicit test pass (no tests in the repo, or only smoke/compile checks ran):
                # the Lead must review the diff against the request before finishing.
                digest["review_required"] = True
                digest["next_action"] = (
                    f"Applied, but no explicit acceptance test verified it (verification="
                    f"{digest.get('verification', 'unverified')!r}). Review diff_preview (or "
                    f"get_diff(job_id={job_id!r})) against every requirement before finishing; if something is "
                    f"wrong, call delegate_revert(job_id={job_id!r}) or start a targeted delegate_start follow-up."
                )
        return digest
