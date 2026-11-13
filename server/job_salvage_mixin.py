"""Cancellation, live progress digest, worktree salvage, and shutdown mixin for `JobManager`."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
import time
from typing import Any

from server.acceptance_runner import (
    _TEST_RUNNER_CMD_RE,
    _extract_failing_test_nodes_and_files,
    _extract_pytest_runner_prefix,
    run_acceptance,
)
from server.agy_runner import read_progress
from server.ast_regression_guards import _is_test_file_path
from server.job_git_utils import (
    _env_int,
    _is_tracked_in_git,
    _matches_allowed_paths,
    _split_diff_by_file,
    validate_allowed_paths,
)
from server.receipts import (
    extract_diff_provenance,
    parse_diff_git_header,
    receipt_owner_pid_alive as _receipt_owner_pid_alive,
    validate_job_id,
)
from server.regions import (
    MAX_TOOL_RESPONSE_TOKENS,
    estimate_tokens,
    truncate_to_token_budget,
)
from server.test_discovery import _collect_workspace_node_bin_dirs
from server.verification_policy import (
    classify_verification,
    has_real_acceptance_pass,
    is_non_smoke_passing_entry,
)


class JobSalvageMixin:
    """Cancellation, crash-recovery salvage, compact digest rendering, and pool shutdown."""

    def delegate_cancel(self, job_id: str, *, cleanup_worktree: bool = True) -> dict[str, Any]:
        """Cancel a job, immediately terminate any active worker process, and clean up its worktree."""
        safe_id = validate_job_id(job_id)
        with self._lock:
            self._cancelled.add(safe_id)
            self._wait_sliced_auto_apply.pop(safe_id, None)
            cancel_fn = self._cancel_hooks.get(safe_id)
            fut = self._futures.get(safe_id)
            if fut is not None:
                fut.cancel()
            still_running = fut is not None and not fut.done()

        had_cancel_hook = callable(cancel_fn)
        if had_cancel_hook:
            try:
                cancel_fn()
            except Exception:
                pass
        if still_running and fut is not None:
            try:
                fut.result(timeout=2.0 if had_cancel_hook else 0.05)
            except Exception:
                pass
            still_running = not fut.done()

        receipt = self.store.read_receipt(safe_id)
        wt_dir = Path(str(receipt.get("worktree_dir") or ""))
        if not receipt.get("applied") and not receipt.get("in_place") and wt_dir.exists() and not still_running:
            baseline_sha = str(receipt.get("baseline_commit_sha") or "HEAD")
            try:
                rec_diff, rec_files, rec_escapes = self._collect_worktree_diff(
                    wt_dir,
                    baseline_sha,
                    allowed_paths=receipt.get("allowed_paths"),
                )
                if rec_files and not rec_escapes:
                    self.store.diff_path(safe_id).write_text(rec_diff, encoding="utf-8")
                    receipt["files_changed"] = rec_files
                    receipt["symlink_escapes"] = rec_escapes
                    receipt["baseline_file_hashes"] = self._compute_changed_baseline_hashes(
                        wt_dir,
                        baseline_sha,
                        rec_files,
                        dict(receipt.get("baseline_file_hashes") or {}),
                    )
                    receipt["provenance"] = extract_diff_provenance(rec_diff)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        # Only remove wt_dir immediately if the worker thread is not currently running inside it;
        # if it is still running, _execute_job's finally block removes wt_dir upon exit.
        if cleanup_worktree and wt_dir.exists() and not still_running:
            self._cleanup_worktree(wt_dir)

        if not receipt.get("applied"):
            receipt["cancelled"] = True
            receipt["status"] = "cancelled"
            receipt["error"] = "Cancelled by user via delegate_cancel."
            receipt["finished_at_epoch"] = round(time.time(), 3)
            self.store.write_receipt(
                safe_id,
                receipt,
                scrub=bool(receipt.get("scrub_receipt", False)),
                workspace_root=str(self.workspace_root),
            )

        return {
            "job_id": safe_id,
            "status": receipt.get("status"),
            "cancelled": True,
        }

    def _read_live_worker_progress(self, job_id: str, receipt: dict[str, Any]) -> int:
        """Return live cumulative tool_call_count while a worker phase is running."""
        base_calls = int(receipt.get("prior_tool_call_count") or receipt.get("tool_call_count") or 0)
        try:
            prog_p = self.store.progress_path(job_id)
            live_path = Path(f"{prog_p}.live")
            if not live_path.is_file():
                live_path = prog_p.with_suffix(".live.json")
            if not live_path.is_file():
                return int(receipt.get("tool_call_count") or 0)
            live_data = json.loads(live_path.read_text(encoding="utf-8"))
            if not isinstance(live_data, dict):
                return int(receipt.get("tool_call_count") or 0)
            phase_calls = int(live_data.get("straitjacket_calls") or 0)
            save_dir_str = str(live_data.get("save_dir") or "").strip()
            if save_dir_str:
                save_p = Path(save_dir_str)
                if save_p.is_dir():
                    from server.agy_runner import extract_sqlite_session_telemetry

                    sql_tel = extract_sqlite_session_telemetry(save_p)
                    phase_calls = max(phase_calls, len(sql_tel.get("tool_calls") or []))
            return max(int(receipt.get("tool_call_count") or 0), base_calls + phase_calls)
        except (OSError, ValueError, TypeError, KeyError):
            return int(receipt.get("tool_call_count") or 0)

    def _denied_actions_next_action(self, job_id: str, denied_actions: list[dict[str, Any]]) -> str:
        """Explain how the Lead recovers from actions the organization's `agy` policy blocked."""
        try:
            worker_copy = str(self.store.worktree_path(job_id))
        except ValueError:
            worker_copy = "the job's isolated worker copy"
        return (
            f"The worker's `agy` run stopped because the organization's Gemini Enterprise policy blocked "
            f"{len(denied_actions)} action(s) (see denied_actions). Nothing was applied to the live workspace. "
            "Show each blocked action to the user and ask whether to approve it. If approved, run the command "
            f"or read the file yourself inside the isolated worker copy at {worker_copy} (never in the live "
            f"workspace), then call delegate_continue(job_id={job_id!r}, follow_up_prompt=<the output or file "
            "contents, and what to do next>), which resumes the same `agy` conversation. If refused, call "
            f"delegate_continue(job_id={job_id!r}, follow_up_prompt=...) telling the worker to finish without "
            f"that action. Call delegate_apply(job_id={job_id!r}) only once the diff is complete."
        )

    def _digest_with_progress(self, receipt: dict[str, Any], *, include_summary: bool = True) -> dict[str, Any]:
        """Compact digest plus the worker's latest report_to_lead signals (visible while it is running)."""
        digest = self._compact_digest(receipt, include_summary=include_summary)
        job_id = str(receipt.get("job_id") or "")
        if job_id:
            try:
                digest["progress"] = read_progress(self.store.progress_path(job_id), limit=5)
            except ValueError:
                digest["progress"] = []
        if digest.get("status") == "running":
            if job_id:
                digest["tool_call_count"] = self._read_live_worker_progress(job_id, receipt)
            sources = list(receipt.get("prompt_sources") or [])
            attempt_num = max(1, len(sources) or len(receipt.get("prompts") or []) or int(receipt.get("attempt") or 1))
            max_calls = digest.get("cumulative_max_tool_calls") or (digest.get("budget") or {}).get("max_tool_calls")
            calls_note = (
                f"{digest.get('tool_call_count', 0)}/{max_calls} tool calls used"
                if max_calls
                else f"{digest.get('tool_call_count', 0)} tool calls used"
            )
            if attempt_num > 1 and sources:
                src_label = sources[-1]
                digest["poll_hint"] = (
                    f"Attempt {attempt_num} ({src_label}) is actively running ({calls_note}). "
                    f"Do NOT read files, run host commands, or call delegate_apply yet. Poll again with "
                    f"delegate_result(job_id={job_id!r}, wait=true) (waits up to 240s, which keeps your prompt cache warm)."
                )
            else:
                digest["poll_hint"] = (
                    f"Still running ({calls_note}). Poll again with delegate_result(wait=true) (waits up to 240s, "
                    "which keeps your prompt cache warm), or check progress with delegate_status."
                )
        elif (
            job_id
            and digest.get("status") in ("ok", "checkpoint", "needs_input", "timeout", "scope_violation")
            and digest.get("files_changed")
        ):
            try:
                diff_path = self.store.diff_path(job_id)
                full_diff = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""
            except OSError:
                full_diff = ""
            if full_diff.strip():
                # Order non-test implementation file diffs ahead of test file diffs in the preview
                chunks: list[tuple[int, str]] = []
                cur: list[str] = []
                cur_is_test = 0
                for dline in full_diff.splitlines(keepends=True):
                    if dline.startswith("diff --git "):
                        if cur:
                            chunks.append((cur_is_test, "".join(cur)))
                        cur = [dline]
                        hdr = parse_diff_git_header(dline)
                        cpath = (hdr[1] or hdr[0]) if hdr else ""
                        cur_is_test = 1 if (cpath.startswith(("tests/", "test/")) or "/test_" in cpath or cpath.endswith(("_test.py", ".test.ts", ".spec.ts"))) else 0
                    else:
                        cur.append(dline)
                if cur:
                    chunks.append((cur_is_test, "".join(cur)))
                ordered_diff = "".join(chunk for _is_t, chunk in sorted(chunks, key=lambda x: x[0])) if chunks else full_diff
                max_preview_tokens = _env_int("CC_AGY_DIFF_PREVIEW_TOKENS", 1800)
                used_tokens = estimate_tokens(json.dumps(digest, indent=2, sort_keys=True))
                preview_budget = max(0, min(max_preview_tokens, MAX_TOOL_RESPONSE_TOKENS - used_tokens - 150))
                if preview_budget >= 80:
                    preview_txt, was_trunc, _ = truncate_to_token_budget(ordered_diff, preview_budget)
                    digest["diff_preview"] = preview_txt
                    digest["diff_preview_complete"] = not was_trunc
            missing_syms = digest.get("missing_prompt_symbols") or []
            contract_warns = digest.get("contract_warnings") or []
            stub_warns = digest.get("stub_mutation_warnings") or []
            extra_notes: list[str] = []
            if missing_syms:
                extra_notes.append(
                    f" Note: prompt symbol(s) {missing_syms} were not found in the modified files; verify whether "
                    f"they are required and call delegate_continue(job_id={job_id!r}, follow_up_prompt=...) if missing."
                )
            if stub_warns:
                extra_notes.append(
                    f" Note: stub_mutation_warnings flagged existing test stub/mock expectation edits: inspect "
                    f"get_diff(job_id={job_id!r}) to verify existing caller contracts were preserved before applying."
                )
            if contract_warns and len(contract_warns) > len(stub_warns):
                extra_notes.append(
                    f" Note: contract_warnings flagged potential class-attribute shadowing: inspect with "
                    f"get_diff(job_id={job_id!r}, focus_symbols=[...]) and fix via "
                    f"delegate_continue(job_id={job_id!r}, follow_up_prompt=...) if unrequested."
                )
            missing_note = "".join(extra_notes)
            if not digest.get("applied"):
                if digest.get("test_only_diff"):
                    digest["next_action"] = (
                        f"Worker modified only test file(s) ({digest.get('files_changed')}) without changing any "
                        f"production source file. Call delegate_continue(job_id={job_id!r}, follow_up_prompt=...) "
                        f"instructing the worker to implement the production source fix, or call "
                        f"delegate_apply(job_id={job_id!r}) if the task genuinely requested test changes only.{missing_note}"
                    )
                elif (
                    digest.get("status") == "ok"
                    and digest.get("acceptance_passed") is not False
                    and not digest.get("scope_violations")
                    and has_real_acceptance_pass(receipt.get("acceptance_results"))
                ):
                    digest["next_action"] = (
                        f"Acceptance checks passed and scope_violations=[]. Call delegate_apply(job_id={job_id!r}) "
                        f"directly to land these changes without re-running the same test suite.{missing_note}"
                    )
                elif (
                    digest.get("status") == "ok"
                    and digest.get("acceptance_passed") is not False
                    and not digest.get("scope_violations")
                ):
                    digest["next_action"] = (
                        f"No explicit acceptance test verified this change (verification="
                        f"{digest.get('verification', 'unverified')!r}). Review diff_preview (or "
                        f"get_diff(job_id={job_id!r})) against every requirement, then call "
                        f"delegate_apply(job_id={job_id!r}) or send the missing items with "
                        f"delegate_continue(job_id={job_id!r}, follow_up_prompt=...).{missing_note}"
                    )
                elif digest.get("status") == "ok" and digest.get("acceptance_passed") is False:
                    digest["next_action"] = (
                        f"Worker modified {len(digest.get('files_changed') or [])} file(s), but an acceptance check "
                        f"failed (see acceptance_results). Call delegate_continue(job_id={job_id!r}, "
                        f"follow_up_prompt=...) to fix the failing test in the existing worktree.{missing_note}"
                    )
                elif digest.get("status") == "checkpoint":
                    digest["next_action"] = (
                        f"Worker reached the tool-call checkpoint after modifying {len(digest.get('files_changed') or [])} "
                        f"file(s) (acceptance_passed={digest.get('acceptance_passed')!r}). Call "
                        f"delegate_continue(job_id={job_id!r}, follow_up_prompt=...) if any task requirement or test "
                        f"is still unfinished, or call delegate_apply(job_id={job_id!r}) if the diff is complete.{missing_note}"
                    )
                elif digest.get("status") == "needs_input":
                    digest["next_action"] = (
                        f"Worker paused with status='needs_input' (see progress). Call "
                        f"delegate_continue(job_id={job_id!r}, follow_up_prompt=...) to answer and finish, or "
                        f"delegate_apply(job_id={job_id!r}) if the edits are already complete.{missing_note}"
                    )
                elif digest.get("status") == "scope_violation" or digest.get("scope_violations"):
                    digest["next_action"] = (
                        f"Worker modified file(s) outside allowed_paths ({digest.get('scope_violations')}). Review "
                        f"diff_preview and call delegate_apply(job_id={job_id!r}, allow_scope_violations=True) to "
                        f"approve, or call delegate_continue(job_id={job_id!r}, follow_up_prompt=..., allowed_paths=...).{missing_note}"
                    )
                elif digest.get("status") == "timeout":
                    if digest.get("acceptance_passed") is True:
                        digest["next_action"] = (
                            f"Worker timed out after modifying {len(digest.get('files_changed') or [])} file(s) and passing "
                            f"acceptance checks. Do NOT read or grep repository files yourself; immediately call "
                            f"delegate_continue(job_id={job_id!r}, follow_up_prompt=...) if any task item or unit test is "
                            f"still missing, or call delegate_apply(job_id={job_id!r}, allow_scope_violations=True) now.{missing_note}"
                        )
                    else:
                        digest["next_action"] = (
                            f"Worker timed out mid-implementation with {len(digest.get('files_changed') or [])} modified "
                            f"file(s) preserved in its worktree. Do NOT read or grep repository files yourself; "
                            f"immediately call delegate_continue(job_id={job_id!r}, follow_up_prompt=...) so the worker "
                            f"can finish the remaining edits and run acceptance tests.{missing_note}"
                        )
        elif job_id and digest.get("status") in ("timeout", "checkpoint", "needs_input") and not digest.get("applied"):
            digest["next_action"] = (
                f"Worker stopped with status={digest.get('status')!r} before modifying files. Do NOT read or grep "
                f"repository files yourself; immediately call delegate_continue(job_id={job_id!r}, "
                "follow_up_prompt=...) instructing the worker to apply its changes directly with edit_file/create_file."
            )
        if digest.get("denied_actions") and not digest.get("applied") and digest.get("status") != "running":
            digest["next_action"] = self._denied_actions_next_action(job_id, digest["denied_actions"])
        if digest.get("escalate_to_lead") or (digest.get("escalation") or {}).get("escalate_to_lead"):
            esc_hint = (
                "escalate_to_lead=true: Host Edit, Write, Read, Grep, Glob, and test commands are now unlocked "
                "so you can apply small targeted fixes directly without spawning another worker."
            )
            cur_na = str(digest.get("next_action") or "")
            if cur_na:
                cur_na = cur_na.replace(
                    "Do NOT read or grep repository files yourself; immediately call ",
                    "Call ",
                )
                if "escalate_to_lead=true" not in cur_na:
                    digest["next_action"] = f"{cur_na} {esc_hint}"
                else:
                    digest["next_action"] = cur_na
            else:
                digest["next_action"] = esc_hint
        return digest

    def salvage_unapplied_worktree(
        self,
        *,
        owner_instance_id: str | None = None,
    ) -> dict[str, Any] | None:
        """If the workspace has zero applied jobs on shutdown, salvage the best unapplied worker job
        that produced clean file edits so bounded benchmark/CLI runs never exit with an empty patch."""
        all_receipts = self.store.list_receipts(workspace_root=self.workspace_root)
        if any(bool(r.get("applied")) for r in all_receipts):
            return None
        if any(
            r.get("status") == "running"
            and (r.get("owner_instance_id") == self._instance_id or _receipt_owner_pid_alive(r))
            for r in all_receipts
        ):
            return None
        candidates: list[dict[str, Any]] = []
        with self._lock:
            cancelled_snapshot = set(self._cancelled)
        for r in all_receipts:
            if r.get("kind", "delegate") != "delegate":
                continue
            if owner_instance_id is not None and r.get("owner_instance_id") != owner_instance_id:
                continue
            if (
                r.get("applied")
                or r.get("cancelled")
                or r.get("symlink_escapes")
                or r.get("external_worktree_tamper")
                or not r.get("files_changed")
            ):
                continue
            if r.get("status") not in ("ok", "checkpoint", "needs_input", "timeout", "interrupted", "scope_violation"):
                continue
            jid = str(r.get("job_id") or "")
            if not jid or jid in cancelled_snapshot:
                continue
            wt_p = Path(str(r.get("worktree_dir") or ""))
            if not wt_p.exists() and not self.store.diff_path(jid).exists():
                continue
            candidates.append(r)
        if not candidates:
            return None
        candidates.sort(
            key=lambda r: (
                1 if r.get("acceptance_passed") is True else (0 if r.get("acceptance_passed") is None else -1),
                1 if r.get("status") == "ok" else 0,
                len(r.get("files_changed") or []),
                float(r.get("started_at_epoch") or 0.0),
            ),
            reverse=True,
        )
        best = candidates[0]
        best_id = str(best["job_id"])
        dropped_new_tests = self._filter_unverified_new_tests_for_salvage(best)
        try:
            lock_file_path = self.store.data_dir / ".apply.lock"
            with self._apply_lock:
                with open(lock_file_path, "w", encoding="utf-8") as lock_handle:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                    try:
                        res = self._delegate_apply_locked(
                            best_id,
                            cleanup_worktree=False,
                            allow_scope_violations=True,
                        )
                    finally:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            rec = self.store.read_receipt(best_id)
            rec["salvaged_on_shutdown"] = True
            if dropped_new_tests:
                rec["salvage_omitted_unverified_tests"] = dropped_new_tests
            self.store.write_receipt(
                best_id,
                rec,
                scrub=bool(rec.get("scrub_receipt", False)),
                workspace_root=str(self.workspace_root),
            )
            return res
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _filter_unverified_new_tests_for_salvage(
        self,
        receipt: dict[str, Any],
        wt_path: Path | None = None,
        *,
        exclude_paths: list[str] | None = None,
    ) -> list[str]:
        """When applying or salvaging an unverified job that modified both implementation files and brand-new untracked
        test files, smoke-check or omit newly created test files that fail or time out so a deadlocking or broken test
        file is never dropped into the host workspace."""
        if receipt.get("acceptance_passed") is True:
            return []
        files = [str(f) for f in (receipt.get("files_changed") or []) if f]
        base_hashes = dict(receipt.get("baseline_file_hashes") or {})
        excl_norm = validate_allowed_paths(exclude_paths or [])
        new_test_files = [
            f
            for f in files
            if _is_test_file_path(f)
            and not (excl_norm and _matches_allowed_paths(f, excl_norm))
            and base_hashes.get(f) is None
            and not (self.workspace_root / f).exists()
            and not _is_tracked_in_git(self.workspace_root, f)
        ]
        if not new_test_files or len(new_test_files) >= len(files):
            return []
        wt_p = wt_path if (wt_path is not None and str(wt_path)) else Path(str(receipt.get("worktree_dir") or ""))
        acc_res = [r for r in (receipt.get("acceptance_results") or []) if isinstance(r, dict)]
        dropped: list[str] = []
        nm_link = self._link_host_node_modules_for_checks(wt_p) if wt_p.is_dir() else None
        try:
            for ntf in new_test_files:
                bname = Path(ntf).name
                timed_out_in_acc = any(
                    (r.get("timed_out") is True or r.get("diagnostic_timed_out") is True)
                    and (
                        ntf in str(r.get("effective_command") or r.get("command") or "")
                        or bname in str(r.get("effective_command") or r.get("command") or "")
                        or ntf in str(r.get("output_tail") or "")
                        or bname in str(r.get("output_tail") or "")
                    )
                    for r in acc_res
                )
                if timed_out_in_acc:
                    dropped.append(ntf)
                    continue
                failed_in_acc = False
                ref_cwd_extract = wt_p if wt_p.is_dir() else self.workspace_root
                for r in acc_res:
                    if r.get("passed") is not False or r.get("timed_out"):
                        continue
                    out_tail = str(r.get("output_tail") or "")
                    _, f_files = _extract_failing_test_nodes_and_files(out_tail, ref_cwd_extract)
                    if any(ff == ntf or Path(ff).name == bname for ff in f_files):
                        failed_in_acc = True
                        break
                    eff_c = str(r.get("effective_command") or r.get("command") or "")
                    eff_rc = (
                        r.get("diagnostic_exit_code")
                        if r.get("diagnostic_exit_code") is not None
                        else r.get("exit_code")
                    )
                    if (
                        int(eff_rc or 0) not in (0, 4)
                        and (ntf in eff_c or bname in eff_c)
                    ):
                        try:
                            toks = shlex.split(eff_c)
                        except ValueError:
                            toks = []
                        test_file_toks = [
                            t.split("::", 1)[0]
                            for t in toks
                            if not t.startswith("-") and _is_test_file_path(t.split("::", 1)[0])
                        ]
                        if len(test_file_toks) == 1 and (
                            test_file_toks[0] == ntf or Path(test_file_toks[0]).name == bname
                        ):
                            failed_in_acc = True
                            break
                if failed_in_acc:
                    dropped.append(ntf)
                    continue
                already_passed = any(
                    is_non_smoke_passing_entry(r)
                    and not r.get("baseline_neutralized")
                    and (
                        int((r.get("counts") or {}).get("passed") or 0) > 0
                        or bool(_TEST_RUNNER_CMD_RE.search(str(r.get("effective_command") or r.get("command") or "")))
                    )
                    and (
                        ntf in str(r.get("effective_command") or r.get("command") or "")
                        or bname in str(r.get("effective_command") or r.get("command") or "")
                    )
                    for r in acc_res
                )
                if already_passed:
                    continue
                if not wt_p.is_dir() or not (wt_p / ntf).is_file():
                    dropped.append(ntf)
                    continue
                smoke_cmd: str | None = None
                used_fallback_sys_pytest = False
                if ntf.endswith(".py"):
                    acc_pytest_pref: str | None = None
                    for raw_ac in [
                        *(receipt.get("acceptance_commands") or []),
                        *(r.get("effective_command") or r.get("command") for r in acc_res),
                    ]:
                        if not raw_ac:
                            continue
                        extracted_pref = _extract_pytest_runner_prefix(str(raw_ac))
                        if extracted_pref:
                            try:
                                pref_toks = shlex.split(extracted_pref)
                                for idx_t, pt in enumerate(pref_toks):
                                    if pt.startswith("PYTHONPATH=") and "." not in pt.split("=", 1)[1].split(":"):
                                        pref_toks[idx_t] = f"{pt}:."
                                acc_pytest_pref = shlex.join(pref_toks)
                            except ValueError:
                                acc_pytest_pref = extracted_pref
                            break
                    if acc_pytest_pref:
                        runner_pref = acc_pytest_pref
                    elif (wt_p / ".venv" / "bin" / "pytest").is_file():
                        runner_pref = ".venv/bin/pytest"
                    elif (self.workspace_root / ".venv" / "bin" / "pytest").is_file():
                        runner_pref = str(self.workspace_root / ".venv" / "bin" / "pytest")
                    else:
                        runner_pref = "python3 -m pytest"
                        used_fallback_sys_pytest = True
                    smoke_cmd = f"{runner_pref} -q {shlex.quote(ntf)}"
                elif ntf.endswith(".go") and shutil.which("go"):
                    rel_pkg = "./" + str(Path(ntf).parent).replace("\\", "/").strip(".")
                    smoke_cmd = f"go test {shlex.quote(rel_pkg.rstrip('/') or '.')}"
                elif ntf.endswith((".ts", ".tsx", ".js", ".jsx")):
                    node_bins = _collect_workspace_node_bin_dirs(wt_p) + _collect_workspace_node_bin_dirs(
                        self.workspace_root
                    )
                    if any((Path(b) / "vitest").is_file() for b in node_bins):
                        smoke_cmd = f"vitest run {shlex.quote(ntf)}"
                    elif any((Path(b) / "jest").is_file() for b in node_bins):
                        smoke_cmd = f"jest {shlex.quote(ntf)} --watchAll=false"
                    elif ntf.endswith((".js", ".mjs", ".cjs")) and (wt_p / "package.json").is_file():
                        try:
                            pkg_txt = (wt_p / "package.json").read_text(encoding="utf-8", errors="ignore")
                        except OSError:
                            pkg_txt = ""
                        if "node --test" in pkg_txt:
                            smoke_cmd = f"node --test {shlex.quote(ntf)}"
                if smoke_cmd is not None:
                    s_res = run_acceptance([smoke_cmd], wt_p, timeout=8.0, total_wall_budget=8.0)
                    if (
                        not s_res
                        or not s_res[0].get("passed")
                        or s_res[0].get("timed_out")
                        or s_res[0].get("no_tests_collected")
                    ):
                        s0_out = str((s_res[0] if s_res else {}).get("output_tail") or "")
                        if (
                            used_fallback_sys_pytest
                            and s_res
                            and not s_res[0].get("timed_out")
                            and re.search(r"\b(?:ModuleNotFoundError|ImportError)\b", s0_out)
                        ):
                            pass
                        else:
                            dropped.append(ntf)
        finally:
            self._unlink_checks_node_modules(nm_link)
        if not dropped:
            return []
        jid = str(receipt.get("job_id") or "")
        if jid and wt_p.is_dir():
            self._archive_omitted_worktree_files(jid, wt_p, dropped)
        kept_files = [f for f in files if f not in set(dropped)]
        diff_p = self.store.diff_path(jid) if jid else None
        if diff_p is not None and diff_p.is_file():
            try:
                orig_diff = diff_p.read_text(encoding="utf-8", errors="replace")
                split_chunks = _split_diff_by_file(orig_diff)
                kept_chunks = [chunk for rel_c, chunk in split_chunks if rel_c not in set(dropped)]
                omitted_chunks = [chunk for rel_c, chunk in split_chunks if rel_c in set(dropped)]
                if omitted_chunks:
                    omitted_p = diff_p.with_suffix(".omitted.diff")
                    prev_omitted_diff = (
                        omitted_p.read_text(encoding="utf-8", errors="replace")
                        if omitted_p.is_file()
                        else ""
                    )
                    omitted_p.write_text(prev_omitted_diff + "".join(omitted_chunks), encoding="utf-8")
                new_diff = "".join(kept_chunks)
                diff_p.write_text(new_diff, encoding="utf-8")
                receipt["provenance"] = extract_diff_provenance(new_diff)
            except (OSError, ValueError, TypeError):
                pass
        receipt["files_changed"] = kept_files
        prev_omitted_list = list(receipt.get("omitted_unverified_new_test_files") or [])
        receipt["omitted_unverified_new_test_files"] = list(dict.fromkeys([*prev_omitted_list, *dropped]))
        if jid:
            self.store.write_receipt(
                jid,
                receipt,
                scrub=bool(receipt.get("scrub_receipt", False)),
                workspace_root=str(self.workspace_root),
            )
        return dropped

    def mark_running_interrupted(self) -> list[str]:
        """Called when the MCP server is shutting down: terminate active workers first, record unfinished jobs as
        interrupted instead of leaving them 'running', preserve any live worktree diff, and salvage clean edits."""
        marked: list[str] = []
        with self._lock:
            self._shutting_down = True
            live_futs = {jid: fut for jid, fut in self._futures.items() if not fut.done()}
            live = set(live_futs.keys())
            owned_snapshot = set(self._owned_job_ids)
            cancelled_ids = set(self._cancelled)
            live_hooks = [
                self._cancel_hooks[jid]
                for jid in live
                if jid not in cancelled_ids and jid in self._cancel_hooks
            ]
            for jid, fut in live_futs.items():
                if jid not in cancelled_ids:
                    fut.cancel()
        for cancel_fn in live_hooks:
            try:
                cancel_fn()
            except Exception:
                pass
        for jid, fut in live_futs.items():
            if jid not in cancelled_ids and not fut.done():
                try:
                    fut.result(timeout=1.5)
                except Exception:
                    pass
        for r_existing in self.store.list_receipts(workspace_root=self.workspace_root):
            r_id = str(r_existing.get("job_id") or "")
            if (
                r_id
                and r_existing.get("status") == "running"
                and (r_id in owned_snapshot or r_existing.get("owner_instance_id") == self._instance_id)
            ):
                live.add(r_id)
        for jid in live:
            try:
                receipt = self.store.read_receipt(jid)
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if jid in cancelled_ids or receipt.get("cancelled") or receipt.get("status") == "cancelled":
                # Keep the user's explicit cancel; never relabel it as a server interruption.
                continue
            if receipt.get("status") not in ("running", "interrupted"):
                # _execute_job already finalized this receipt cleanly during the 1.5s drain window.
                continue
            if not receipt.get("in_place") and not receipt.get("applied"):
                wt_dir = Path(str(receipt.get("worktree_dir") or ""))
                baseline_sha = str(receipt.get("baseline_commit_sha") or "HEAD")
                if wt_dir.exists():
                    try:
                        rec_diff, rec_files, rec_escapes = self._collect_worktree_diff(
                            wt_dir,
                            baseline_sha,
                            allowed_paths=receipt.get("allowed_paths"),
                        )
                        if rec_files and not rec_escapes:
                            self.store.diff_path(jid).write_text(rec_diff, encoding="utf-8")
                            receipt["files_changed"] = rec_files
                            receipt["symlink_escapes"] = rec_escapes
                            receipt["provenance"] = extract_diff_provenance(rec_diff)
                    except (OSError, ValueError, subprocess.SubprocessError):
                        pass
            receipt["status"] = "interrupted"
            receipt["error"] = (
                "The MCP server stopped before this job finished; the unfinished turn's token usage was not "
                "recorded by the SDK."
            )
            receipt["finished_at_epoch"] = round(time.time(), 3)
            try:
                self.store.write_receipt(
                    jid, receipt, scrub=bool(receipt.get("scrub_receipt", False)),
                    workspace_root=str(self.workspace_root),
                )
                marked.append(jid)
            except (OSError, ValueError, TypeError, KeyError):
                continue
        from server.regions import resolve_auto_apply_override

        raw_salvage = os.environ.get("CC_AGY_SALVAGE_ON_SHUTDOWN", "").strip().lower()
        salvage_disabled = raw_salvage in ("0", "false", "no", "off") or (
            resolve_auto_apply_override() is False and raw_salvage not in ("1", "true", "yes", "on")
        )
        should_salvage = (
            not salvage_disabled
            and bool(owned_snapshot or marked)
            and (self._is_bounded_lead_session() or raw_salvage in ("1", "true", "yes", "on"))
        )
        if should_salvage:
            try:
                self.salvage_unapplied_worktree(owner_instance_id=self._instance_id)
            except Exception:
                pass
        return marked

    def shutdown(self, wait: bool = True) -> None:
        """Terminate any active worker sessions and shutdown the underlying thread pool executor."""
        with self._lock:
            self._shutting_down = True
            hooks = list(self._cancel_hooks.values())
            qa_threads = list(self._qa_bg_threads)
            for fut in self._futures.values():
                if not fut.done():
                    fut.cancel()
        for fn in hooks:
            try:
                fn()
            except Exception:
                pass
        if wait:
            for qt in qa_threads:
                if qt.is_alive() and qt is not threading.current_thread():
                    qt.join(timeout=15.0)
        self._executor.shutdown(wait=wait)

    @staticmethod
    def _compact_digest(
        receipt: dict[str, Any],
        *,
        include_summary: bool = True,
        include_acceptance_details: bool = True,
    ) -> dict[str, Any]:
        """Return a token-bounded digest suitable for Claude Code's context window."""
        is_running = receipt.get("status") == "running"
        prompt_sources = [str(s) for s in (receipt.get("prompt_sources") or []) if s]
        attempt_num = max(
            1,
            len(prompt_sources)
            or len(receipt.get("prompts") or [])
            or int(receipt.get("attempt") or 1),
        )
        if not include_acceptance_details and not include_summary:
            raw_usage = receipt.get("usage") or {}
            raw_cost = receipt.get("cost") or {}
            slim: dict[str, Any] = {
                "job_id": receipt.get("job_id"),
                "kind": receipt.get("kind", "delegate"),
                "status": receipt.get("status"),
                "auth_mode": receipt.get("auth_mode", "vertex"),
                "model": receipt.get("model"),
                "files_changed": receipt.get("files_changed", []),
                "applied": bool(receipt.get("applied", False)),
                "in_place": bool(receipt.get("in_place", False)),
                "usage": {
                    "prompt_token_count": int(raw_usage.get("prompt_token_count") or 0),
                    "cached_content_token_count": int(raw_usage.get("cached_content_token_count") or 0),
                    "candidates_token_count": int(raw_usage.get("candidates_token_count") or 0),
                    "thoughts_token_count": int(raw_usage.get("thoughts_token_count") or 0),
                    "total_token_count": int(raw_usage.get("total_token_count") or 0),
                },
                "cost": {
                    "total_cost_usd": float(raw_cost.get("total_cost_usd") or 0.0),
                    "total_cost_cents": float(raw_cost.get("total_cost_cents") or 0.0),
                },
            }
            if attempt_num > 1:
                slim["attempt"] = attempt_num
            if prompt_sources and (attempt_num > 1 or any(s != "lead" for s in prompt_sources)):
                slim["prompt_sources"] = prompt_sources
            if not is_running and receipt.get("acceptance_passed") is not None:
                slim["acceptance_passed"] = receipt.get("acceptance_passed")
            if (receipt.get("escalation") or {}).get("escalate_to_lead"):
                slim["escalate_to_lead"] = True
            if receipt.get("reverted"):
                slim["reverted"] = True
            if receipt.get("partially_reverted"):
                slim["partially_reverted"] = True
            if receipt.get("error"):
                slim["error"] = receipt.get("error")
            if not is_running and receipt.get("stop_reason"):
                slim["stop_reason"] = receipt.get("stop_reason")
            if not is_running and receipt.get("denied_actions"):
                slim["denied_actions"] = receipt["denied_actions"]
            if receipt.get("escalated_from_model"):
                slim["escalated_from_model"] = receipt.get("escalated_from_model")
            if receipt.get("opencode_session_id"):
                slim["opencode_session_id"] = receipt.get("opencode_session_id")
            return slim

        digest_usage = dict(receipt.get("usage") or {})
        raw_reqs = digest_usage.get("requests")
        if isinstance(raw_reqs, list):
            def _safe_req_tokens(req_obj: Any) -> int:
                if not isinstance(req_obj, dict):
                    return 0
                try:
                    return int(float(req_obj.get("prompt_token_count") or 0))
                except (ValueError, TypeError):
                    return 0

            digest_usage["requests_count"] = len(raw_reqs)
            digest_usage["over_200k_requests_count"] = sum(
                1
                for r in raw_reqs
                if _safe_req_tokens(r) > 200_000
            )
            if len(raw_reqs) > 10:
                digest_usage["requests"] = raw_reqs[-10:]
                digest_usage["requests_truncated"] = True
        if isinstance(digest_usage.get("turns"), list) and len(digest_usage["turns"]) > 10:
            digest_usage["turns"] = digest_usage["turns"][-10:]

        digest: dict[str, Any] = {
            "job_id": receipt.get("job_id"),
            "kind": receipt.get("kind", "delegate"),
            "status": receipt.get("status"),
            "auth_mode": receipt.get("auth_mode", "vertex"),
            "model": receipt.get("model"),
            "vertex_location": receipt.get("vertex_location"),
            "reroute_note": receipt.get("reroute_note"),
            "allowed_paths": receipt.get("allowed_paths", []),
            "scope_violations": receipt.get("scope_violations", []),
            "files_changed": receipt.get("files_changed", []),
            "diff_tokens_estimated": receipt.get("diff_tokens_estimated", 0),
            "tool_call_count": receipt.get("tool_call_count", 0),
            "usage": digest_usage,
            "cost": receipt.get("cost", {}),
            "applied": bool(receipt.get("applied", False)),
            "in_place": bool(receipt.get("in_place", False)),
            "thinking_level": receipt.get("thinking_level"),
            "stop_reason": None if is_running else receipt.get("stop_reason"),
            "budget": receipt.get("budget"),
            "containment": receipt.get("containment", "none"),
            "containment_warning": receipt.get("containment_warning"),
            "acceptance_passed": None if is_running else receipt.get("acceptance_passed"),
            "error": receipt.get("error"),
        }
        if receipt.get("denied_actions") and not is_running:
            digest["denied_actions"] = receipt["denied_actions"]
        if receipt.get("usage_estimated"):
            digest["usage_estimated"] = True
        if attempt_num > 1 or (prompt_sources and any(s != "lead" for s in prompt_sources)):
            digest["attempt"] = attempt_num
            if prompt_sources:
                digest["prompt_sources"] = prompt_sources
        if receipt.get("cumulative_max_tool_calls"):
            digest["cumulative_max_tool_calls"] = int(receipt["cumulative_max_tool_calls"])
        if receipt.get("files_changed") and not is_running:
            digest["verification"] = classify_verification(
                receipt.get("acceptance_results"), receipt.get("acceptance_passed")
            )
        if receipt.get("preloaded_files"):
            digest["preloaded_files"] = list(receipt.get("preloaded_files") or [])
        if receipt.get("escalated_from_model"):
            digest["escalated_from_model"] = receipt.get("escalated_from_model")
        if receipt.get("opencode_session_id"):
            digest["opencode_session_id"] = receipt.get("opencode_session_id")
        if not is_running and receipt.get("vacuous_acceptance"):
            digest["vacuous_acceptance"] = True
        if not is_running and receipt.get("test_only_diff"):
            digest["test_only_diff"] = True
        if receipt.get("spec_conflict"):
            digest["spec_conflict"] = receipt.get("spec_conflict")
        if not is_running and receipt.get("budget_exhausted"):
            digest["budget_exhausted"] = True
        if not is_running and receipt.get("recovered_from_timeout"):
            digest["recovered_from_timeout"] = True
        if not is_running and receipt.get("recovered_from_checkpoint"):
            digest["recovered_from_checkpoint"] = True
        if not is_running and receipt.get("missing_prompt_symbols"):
            digest["missing_prompt_symbols"] = list(receipt.get("missing_prompt_symbols") or [])
        if not is_running and receipt.get("stub_mutation_warnings"):
            digest["stub_mutation_warnings"] = list(receipt.get("stub_mutation_warnings") or [])
        if not is_running and isinstance(receipt.get("qa_review"), dict):
            digest["qa_review"] = dict(receipt["qa_review"])
        if not is_running and receipt.get("contract_warnings"):
            digest["contract_warnings"] = list(receipt.get("contract_warnings") or [])
        if receipt.get("baseline_acceptance_warnings"):
            digest["baseline_acceptance_warnings"] = list(receipt.get("baseline_acceptance_warnings") or [])
        if receipt.get("acceptance_remapped"):
            digest["acceptance_remapped"] = list(receipt.get("acceptance_remapped") or [])
        if receipt.get("excluded_runtime_artifacts"):
            digest["excluded_runtime_artifacts"] = list(receipt.get("excluded_runtime_artifacts") or [])
        if receipt.get("preserved_runtime_artifacts"):
            digest["preserved_runtime_artifacts"] = list(receipt.get("preserved_runtime_artifacts") or [])
        if receipt.get("omitted_untracked_out_of_scope"):
            digest["omitted_untracked_out_of_scope"] = list(receipt.get("omitted_untracked_out_of_scope") or [])
        if receipt.get("omitted_unverified_new_test_files"):
            digest["omitted_unverified_new_test_files"] = list(receipt.get("omitted_unverified_new_test_files") or [])
        if receipt.get("applied"):
            digest["applied_files"] = list(receipt.get("applied_files") or receipt.get("files_changed") or [])
            prov = receipt.get("provenance") or {}
            if prov.get("diff_sha256"):
                digest["diff_sha256"] = prov.get("diff_sha256")
        if receipt.get("failed_attempts"):
            digest["failed_attempts"] = int(receipt.get("failed_attempts") or 0)
        if receipt.get("escalate_to_lead") or (receipt.get("escalation") or {}).get("escalate_to_lead"):
            digest["escalate_to_lead"] = True
        if receipt.get("escalation"):
            digest["escalation"] = receipt.get("escalation")
        if receipt.get("reverted"):
            digest["reverted"] = True
        if receipt.get("partially_reverted"):
            digest["partially_reverted"] = True
        if receipt.get("reverted") or receipt.get("partially_reverted"):
            digest["reverted_files"] = receipt.get("reverted_files", [])
        if include_acceptance_details:
            digest["acceptance_results"] = [] if is_running else (receipt.get("acceptance_results") or [])
        if (
            not receipt.get("applied")
            and receipt.get("files_changed")
            and receipt.get("status") in ("timeout", "checkpoint")
        ):
            digest["apply_hint"] = (
                f"Worker modified {len(receipt.get('files_changed', []))} file(s). "
                f"Inspect the diff with get_diff(job_id={receipt.get('job_id')!r}) and apply clean edits directly via "
                f"delegate_apply(job_id={receipt.get('job_id')!r}) if the implementation is ready."
            )
        if receipt.get("in_place"):
            digest["in_place_notes"] = receipt.get("in_place_notes", [])
            digest["undo_hint"] = receipt.get("undo_hint")
        if include_summary:
            summary_text = "" if is_running else str(receipt.get("summary") or "")
            if not is_running and not summary_text.strip():
                for pa in reversed(list(receipt.get("prior_attempts") or [])):
                    if isinstance(pa, dict) and str(pa.get("summary") or "").strip():
                        summary_text = str(pa.get("summary") or "")
                        break
            meta_tokens = estimate_tokens(json.dumps(digest, indent=2, sort_keys=True))
            max_summary_tokens = _env_int("CC_AGY_DIGEST_SUMMARY_TOKENS", 300)
            summary_budget = max(64, min(max_summary_tokens, MAX_TOOL_RESPONSE_TOKENS - meta_tokens - 250))
            clipped_summary, truncated, est_tokens = truncate_to_token_budget(
                summary_text, summary_budget
            )
            digest["summary"] = clipped_summary
            digest["summary_truncated"] = False if is_running else bool(receipt.get("summary_truncated") or truncated)
            digest["summary_tokens_estimated"] = est_tokens
        return digest
