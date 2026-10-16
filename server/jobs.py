"""Git-worktree isolated job manager for Antigravity SDK delegations."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Callable
import uuid

from server.agy_runner import run_antigravity_worker
from server.receipts import ReceiptStore, extract_diff_provenance
from server.regions import (
    DEFAULT_DIFF_PAGE_TOKENS,
    MAX_TOOL_RESPONSE_TOKENS,
    estimate_tokens,
    resolve_vertex_location,
    resolve_vertex_project,
    truncate_to_token_budget,
)


def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=check,
    )


def _has_git_head(repo_dir: Path) -> bool:
    res = _run_git(["rev-parse", "--verify", "HEAD"], cwd=repo_dir, check=False)
    return res.returncode == 0


def _matches_allowed_paths(rel_path: str, allowed_paths: list[str]) -> bool:
    """Return True if rel_path is inside or matches any entry in allowed_paths."""
    if not allowed_paths:
        return True
    norm_path = rel_path.strip().lstrip("./")
    for pattern in allowed_paths:
        norm_pat = pattern.strip().lstrip("./").rstrip("/")
        if not norm_pat or norm_pat in (".", "*"):
            return True
        if norm_path == norm_pat or norm_path.startswith(norm_pat + "/"):
            return True
        if fnmatch.fnmatch(norm_path, norm_pat):
            return True
    return False


class JobManager:
    """Manages isolated git worktrees, background worker execution, and diff application."""

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        store: ReceiptStore | None = None,
        worker_backend: Callable[[dict[str, Any]], Any] | None = None,
        max_workers: int = 4,
    ) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.store = store or ReceiptStore()
        self.worker_backend = worker_backend
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cc-agy-job")
        self._lock = threading.Lock()
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._cancelled: set[str] = set()

    def _prepare_worktree(self, job_id: str) -> Path:
        """Create an isolated git worktree (or isolated clone) for job_id."""
        wt_dir = (self.store.worktrees_dir / job_id).resolve()
        if wt_dir.exists():
            shutil.rmtree(wt_dir, ignore_errors=True)

        if _has_git_head(self.workspace_root):
            res = _run_git(
                ["worktree", "add", "--detach", str(wt_dir), "HEAD"],
                cwd=self.workspace_root,
                check=False,
            )
            if res.returncode == 0:
                self._sync_uncommitted_state_into_worktree(wt_dir)
                return wt_dir

        # Fallback when workspace has no commits yet or git worktree add is unavailable
        shutil.copytree(
            self.workspace_root,
            wt_dir,
            ignore=shutil.ignore_patterns(".git", ".cc-agy-data", ".worktrees", "__pycache__"),
        )
        _run_git(["init", "-b", "main"], cwd=wt_dir, check=False)
        _run_git(["config", "user.email", "worker@cc-agy-plugin.local"], cwd=wt_dir, check=False)
        _run_git(["config", "user.name", "CC-AGY Worker"], cwd=wt_dir, check=False)
        _run_git(["add", "-A"], cwd=wt_dir, check=False)
        _run_git(["commit", "--allow-empty", "-m", "baseline"], cwd=wt_dir, check=False)
        return wt_dir

    def _sync_uncommitted_state_into_worktree(self, wt_dir: Path) -> None:
        """Copy dirty tracked/untracked files from workspace_root into wt_dir and baseline-commit."""
        diff_res = _run_git(["diff", "--binary", "HEAD"], cwd=self.workspace_root, check=False)
        if diff_res.returncode == 0 and diff_res.stdout.strip():
            subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                input=diff_res.stdout,
                cwd=str(wt_dir),
                capture_output=True,
                text=True,
                check=False,
            )

        untracked_res = _run_git(
            ["ls-files", "--others", "--exclude-standard"],
            cwd=self.workspace_root,
            check=False,
        )
        if untracked_res.returncode == 0:
            for rel in untracked_res.stdout.splitlines():
                rel = rel.strip()
                if not rel or rel.startswith(".cc-agy-data") or rel.startswith(".worktrees"):
                    continue
                src = self.workspace_root / rel
                dst = wt_dir / rel
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

        _run_git(["config", "user.email", "worker@cc-agy-plugin.local"], cwd=wt_dir, check=False)
        _run_git(["config", "user.name", "CC-AGY Worker"], cwd=wt_dir, check=False)
        _run_git(["add", "-A"], cwd=wt_dir, check=False)
        status_res = _run_git(["status", "--porcelain"], cwd=wt_dir, check=False)
        if status_res.stdout.strip():
            _run_git(["commit", "-m", "cc-agy-baseline-dirty-state"], cwd=wt_dir, check=False)

    def _collect_worktree_diff(self, wt_dir: Path) -> tuple[str, list[str]]:
        """Stage all worker edits (including new untracked files via intent-to-add) and return unified diff."""
        if not wt_dir.exists():
            return "", []
        _run_git(["add", "-N", "."], cwd=wt_dir, check=False)
        diff_res = _run_git(["diff", "HEAD"], cwd=wt_dir, check=False)
        diff_text = diff_res.stdout if diff_res.returncode == 0 else ""

        name_res = _run_git(["diff", "--name-only", "HEAD"], cwd=wt_dir, check=False)
        files = [
            line.strip()
            for line in (name_res.stdout.splitlines() if name_res.returncode == 0 else [])
            if line.strip()
        ]
        return diff_text, files

    def _cleanup_worktree(self, wt_dir: Path) -> None:
        if not wt_dir.exists():
            return
        _run_git(["worktree", "remove", "--force", str(wt_dir)], cwd=self.workspace_root, check=False)
        if wt_dir.exists():
            shutil.rmtree(wt_dir, ignore_errors=True)

    def _execute_job(
        self,
        *,
        job_id: str,
        prompts: list[str],
        model: str,
        project: str,
        location: str,
        reroute_note: str | None,
        allowed_paths: list[str] | None,
        timeout_seconds: float,
        enabled_tools: list[str] | None,
        disabled_tools: list[str] | None,
        enable_subagents: bool,
        wt_dir: Path,
        started_at: float,
        prior_usage: dict[str, int] | None = None,
        prior_tool_calls: list[dict[str, Any]] | None = None,
        scrub_receipt: bool = False,
    ) -> dict[str, Any]:
        usage: dict[str, int] = {
            "prompt_token_count": int((prior_usage or {}).get("prompt_token_count", 0)),
            "cached_content_token_count": int((prior_usage or {}).get("cached_content_token_count", 0)),
            "candidates_token_count": int((prior_usage or {}).get("candidates_token_count", 0)),
            "thoughts_token_count": int((prior_usage or {}).get("thoughts_token_count", 0)),
            "total_token_count": int((prior_usage or {}).get("total_token_count", 0)),
        }
        tool_calls: list[dict[str, Any]] = list(prior_tool_calls or [])
        thoughts: list[str] = []
        response_text = ""
        status = "error"
        error_msg: str | None = None
        diff_text = ""
        files_changed: list[str] = []
        scope_violations: list[str] = []

        try:
            with self._lock:
                if job_id in self._cancelled:
                    status = "cancelled"
                    error_msg = "Job cancelled before execution."
                    raise RuntimeError(error_msg)

            worker_out = run_antigravity_worker(
                prompts=prompts,
                workspace_dir=wt_dir,
                model=model,
                project=project,
                location=location,
                timeout_seconds=timeout_seconds,
                enabled_tools=enabled_tools,
                disabled_tools=disabled_tools,
                enable_subagents=enable_subagents,
                backend=self.worker_backend,
            )

            response_text = str(worker_out.get("text") or "")
            new_usage = worker_out.get("usage") or {}
            for k in usage:
                usage[k] += int(new_usage.get(k) or 0)
            if usage["total_token_count"] == 0:
                usage["total_token_count"] = (
                    usage["prompt_token_count"]
                    + usage["candidates_token_count"]
                    + usage["thoughts_token_count"]
                )
            tool_calls.extend(worker_out.get("tool_calls") or [])
            thoughts.extend(worker_out.get("thoughts") or [])

            with self._lock:
                if job_id in self._cancelled:
                    status = "cancelled"
                    error_msg = "Job cancelled during execution."
                else:
                    status = "ok"

        except TimeoutError as exc:
            status = "timeout"
            error_msg = f"TimeoutError: {exc}"
            self._merge_partial_exception_data(exc, usage, tool_calls, thoughts)
        except Exception as exc:
            if status != "cancelled":
                status = "error"
                error_msg = f"{type(exc).__name__}: {exc}"
            self._merge_partial_exception_data(exc, usage, tool_calls, thoughts)
        finally:
            diff_text, files_changed = self._collect_worktree_diff(wt_dir)
            if allowed_paths:
                for fpath in files_changed:
                    if not _matches_allowed_paths(fpath, allowed_paths):
                        scope_violations.append(fpath)
                if scope_violations and status == "ok":
                    status = "scope_violation"
                    error_msg = (
                        f"Worker modified files outside allowed_paths {allowed_paths}: "
                        f"{scope_violations}"
                    )

            provenance = extract_diff_provenance(diff_text)
            diff_path = self.store.receipts_dir / f"{job_id}.diff"
            diff_path.write_text(diff_text, encoding="utf-8")

            finished_at = round(time.time(), 3)
            clipped_text, was_truncated, orig_tokens = truncate_to_token_budget(
                response_text, MAX_TOOL_RESPONSE_TOKENS
            )
            receipt_payload = {
                "job_id": job_id,
                "kind": "delegate",
                "status": status,
                "model": model,
                "vertex_project": project,
                "vertex_location": location,
                "reroute_note": reroute_note,
                "prompts": prompts,
                "allowed_paths": allowed_paths or [],
                "scope_violations": scope_violations,
                "worktree_dir": str(wt_dir),
                "started_at_epoch": started_at,
                "finished_at_epoch": finished_at,
                "duration_seconds": round(max(0.0, finished_at - started_at), 3),
                "usage": usage,
                "tool_call_count": len(tool_calls),
                "tool_calls": tool_calls,
                "thoughts_count": len(thoughts),
                "summary": clipped_text,
                "summary_truncated": was_truncated,
                "summary_tokens_estimated": orig_tokens,
                "files_changed": files_changed,
                "diff_tokens_estimated": estimate_tokens(diff_text),
                "provenance": provenance,
                "applied": False,
                "error": error_msg,
            }
            saved = self.store.write_receipt(
                job_id,
                receipt_payload,
                scrub=scrub_receipt,
                workspace_root=str(self.workspace_root),
            )

        return saved

    @staticmethod
    def _merge_partial_exception_data(
        exc: BaseException,
        usage: dict[str, int],
        tool_calls: list[dict[str, Any]],
        thoughts: list[str],
    ) -> None:
        partial_u = getattr(exc, "partial_usage", None)
        if isinstance(partial_u, dict):
            for k in usage:
                usage[k] += int(partial_u.get(k) or 0)
            if usage["total_token_count"] == 0:
                usage["total_token_count"] = (
                    usage["prompt_token_count"]
                    + usage["candidates_token_count"]
                    + usage["thoughts_token_count"]
                )
        partial_tc = getattr(exc, "partial_tool_calls", None)
        if isinstance(partial_tc, list):
            tool_calls.extend(partial_tc)
        partial_th = getattr(exc, "partial_thoughts", None)
        if isinstance(partial_th, list):
            thoughts.extend(partial_th)

    def delegate_start(
        self,
        prompt: str,
        *,
        model: str | None = None,
        allowed_paths: list[str] | None = None,
        project: str | None = None,
        location: str | None = None,
        timeout_seconds: float = 300.0,
        enabled_tools: list[str] | None = None,
        disabled_tools: list[str] | None = None,
        enable_subagents: bool = False,
        wait: bool = False,
        scrub_receipt: bool = False,
    ) -> dict[str, Any]:
        """Start an isolated Antigravity worker job in a dedicated git worktree."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        chosen_model = (
            model
            or os.environ.get("CC_AGY_DEFAULT_DELEGATE_MODEL")
            or "gemini-3.1-pro-preview"
        ).strip()
        if chosen_model.startswith("${"):
            chosen_model = "gemini-3.1-pro-preview"

        resolved_project = resolve_vertex_project(project)
        resolved_location, reroute_note = resolve_vertex_location(chosen_model, location)

        job_id = f"job-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        started_at = round(time.time(), 3)
        wt_dir = self._prepare_worktree(job_id)

        initial_receipt = {
            "job_id": job_id,
            "kind": "delegate",
            "status": "running",
            "model": chosen_model,
            "vertex_project": resolved_project,
            "vertex_location": resolved_location,
            "reroute_note": reroute_note,
            "prompts": [prompt.strip()],
            "allowed_paths": allowed_paths or [],
            "scope_violations": [],
            "worktree_dir": str(wt_dir),
            "started_at_epoch": started_at,
            "usage": {
                "prompt_token_count": 0,
                "cached_content_token_count": 0,
                "candidates_token_count": 0,
                "thoughts_token_count": 0,
                "total_token_count": 0,
            },
            "tool_call_count": 0,
            "tool_calls": [],
            "files_changed": [],
            "applied": False,
            "error": None,
        }
        self.store.write_receipt(
            job_id,
            initial_receipt,
            scrub=scrub_receipt,
            workspace_root=str(self.workspace_root),
        )

        if wait:
            final_receipt = self._execute_job(
                job_id=job_id,
                prompts=[prompt.strip()],
                model=chosen_model,
                project=resolved_project,
                location=resolved_location,
                reroute_note=reroute_note,
                allowed_paths=allowed_paths,
                timeout_seconds=timeout_seconds,
                enabled_tools=enabled_tools,
                disabled_tools=disabled_tools,
                enable_subagents=enable_subagents,
                wt_dir=wt_dir,
                started_at=started_at,
                scrub_receipt=scrub_receipt,
            )
            return self._compact_digest(final_receipt)

        fut = self._executor.submit(
            self._execute_job,
            job_id=job_id,
            prompts=[prompt.strip()],
            model=chosen_model,
            project=resolved_project,
            location=resolved_location,
            reroute_note=reroute_note,
            allowed_paths=allowed_paths,
            timeout_seconds=timeout_seconds,
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
            enable_subagents=enable_subagents,
            wt_dir=wt_dir,
            started_at=started_at,
            scrub_receipt=scrub_receipt,
        )
        with self._lock:
            self._futures[job_id] = fut

        return {
            "job_id": job_id,
            "status": "running",
            "model": chosen_model,
            "vertex_location": resolved_location,
            "reroute_note": reroute_note,
            "allowed_paths": allowed_paths or [],
            "worktree_dir": str(wt_dir),
        }

    def delegate_status(self, job_id: str | None = None) -> dict[str, Any]:
        """Return status for a single job or list all active/recent jobs."""
        if job_id:
            receipt = self.store.read_receipt(job_id)
            return self._compact_digest(receipt)

        summaries = [self._compact_digest(r) for r in self.store.list_receipts()]
        return {
            "jobs": summaries,
            "totals": self.store.summarize_costs(),
        }

    def delegate_result(self, job_id: str, *, wait: bool = True, timeout: float = 600.0) -> dict[str, Any]:
        """Return the compact completion digest for job_id, optionally waiting if still running."""
        with self._lock:
            fut = self._futures.get(job_id)
        if fut is not None and wait and not fut.done():
            fut.result(timeout=timeout)
        receipt = self.store.read_receipt(job_id)
        return self._compact_digest(receipt)

    def get_diff(
        self,
        job_id: str,
        *,
        file_filter: str | None = None,
        page: int = 1,
        page_tokens: int = DEFAULT_DIFF_PAGE_TOKENS,
    ) -> dict[str, Any]:
        """Return paginated unified git diff for job_id measured in tokens."""
        receipt = self.store.read_receipt(job_id)
        diff_path = self.store.receipts_dir / f"{job_id}.diff"
        full_diff = diff_path.read_text(encoding="utf-8") if diff_path.exists() else ""

        if file_filter and full_diff:
            filtered_chunks: list[str] = []
            current_chunk: list[str] = []
            keep_chunk = False
            for line in full_diff.splitlines(keepends=True):
                if line.startswith("diff --git "):
                    if current_chunk and keep_chunk:
                        filtered_chunks.append("".join(current_chunk))
                    current_chunk = [line]
                    keep_chunk = file_filter in line
                else:
                    if line.startswith("+++ b/") and file_filter in line:
                        keep_chunk = True
                    current_chunk.append(line)
            if current_chunk and keep_chunk:
                filtered_chunks.append("".join(current_chunk))
            full_diff = "".join(filtered_chunks)

        budget_tokens = max(250, min(int(page_tokens), MAX_TOOL_RESPONSE_TOKENS))
        bytes_per_page = budget_tokens * 4
        encoded = full_diff.encode("utf-8")
        total_pages = max(1, (len(encoded) + bytes_per_page - 1) // bytes_per_page)
        clamped_page = max(1, min(int(page), total_pages))
        start_b = (clamped_page - 1) * bytes_per_page
        end_b = start_b + bytes_per_page
        page_slice = encoded[start_b:end_b].decode("utf-8", errors="ignore")

        return {
            "job_id": job_id,
            "status": receipt.get("status"),
            "file_filter": file_filter,
            "page": clamped_page,
            "total_pages": total_pages,
            "page_tokens_estimated": estimate_tokens(page_slice),
            "total_diff_tokens_estimated": estimate_tokens(full_diff),
            "files_changed": receipt.get("files_changed", []),
            "diff": page_slice,
        }

    def delegate_apply(self, job_id: str, *, cleanup_worktree: bool = True) -> dict[str, Any]:
        """Apply a verified worker job's changes from its isolated worktree to the main workspace."""
        with self._lock:
            fut = self._futures.get(job_id)
        if fut is not None and not fut.done():
            raise RuntimeError(f"Job '{job_id}' is still running; wait for completion before applying.")

        receipt = self.store.read_receipt(job_id)
        status = receipt.get("status")
        if status != "ok":
            raise ValueError(
                f"Cannot apply job '{job_id}' with status={status!r} "
                f"(error={receipt.get('error')!r}). Only status='ok' jobs can be applied."
            )

        wt_dir = Path(str(receipt.get("worktree_dir") or ""))
        diff_path = self.store.receipts_dir / f"{job_id}.diff"
        diff_text = diff_path.read_text(encoding="utf-8") if diff_path.exists() else ""
        files_changed: list[str] = list(receipt.get("files_changed") or [])
        allowed_paths: list[str] = list(receipt.get("allowed_paths") or [])

        for rel_path in files_changed:
            if not _matches_allowed_paths(rel_path, allowed_paths):
                raise ValueError(
                    f"Refusing to apply out-of-scope file {rel_path!r} (allowed_paths={allowed_paths})"
                )

        applied_files: list[str] = []
        if wt_dir.exists():
            for rel_path in files_changed:
                src = (wt_dir / rel_path).resolve()
                dst = (self.workspace_root / rel_path).resolve()
                if not str(dst).startswith(str(self.workspace_root)):
                    raise ValueError(f"Path traversal blocked for {rel_path!r}")
                if src.exists() and src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    applied_files.append(rel_path)
                elif not src.exists() and dst.exists():
                    dst.unlink()
                    applied_files.append(rel_path)
        elif diff_text.strip():
            subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                input=diff_text,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                check=True,
            )
            applied_files = list(files_changed)

        if cleanup_worktree and wt_dir.exists():
            self._cleanup_worktree(wt_dir)

        receipt["applied"] = True
        receipt["applied_at_epoch"] = round(time.time(), 3)
        receipt["applied_files"] = applied_files
        receipt["provenance"] = extract_diff_provenance(diff_text)
        self.store.write_receipt(
            job_id,
            receipt,
            workspace_root=str(self.workspace_root),
        )

        return {
            "job_id": job_id,
            "applied": True,
            "applied_files": applied_files,
            "diff_sha256": receipt["provenance"]["diff_sha256"],
        }

    def delegate_continue(
        self,
        job_id: str,
        follow_up_prompt: str,
        *,
        timeout_seconds: float = 300.0,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Send a follow-up prompt into the same job's isolated git worktree."""
        if not follow_up_prompt or not follow_up_prompt.strip():
            raise ValueError("follow_up_prompt must be a non-empty string")

        receipt = self.store.read_receipt(job_id)
        wt_dir = Path(str(receipt.get("worktree_dir") or ""))
        if not wt_dir.exists():
            wt_dir = self._prepare_worktree(job_id)

        prompts = list(receipt.get("prompts") or [])
        prompts.append(follow_up_prompt.strip())
        model = str(receipt.get("model") or "gemini-3.1-pro-preview")
        project = resolve_vertex_project(str(receipt.get("vertex_project") or ""))
        location, reroute_note = resolve_vertex_location(
            model, str(receipt.get("vertex_location") or "")
        )
        allowed_paths = list(receipt.get("allowed_paths") or [])
        started_at = float(receipt.get("started_at_epoch") or time.time())

        if wait:
            updated = self._execute_job(
                job_id=job_id,
                prompts=prompts,
                model=model,
                project=project,
                location=location,
                reroute_note=reroute_note,
                allowed_paths=allowed_paths,
                timeout_seconds=timeout_seconds,
                enabled_tools=None,
                disabled_tools=None,
                enable_subagents=False,
                wt_dir=wt_dir,
                started_at=started_at,
                prior_usage=receipt.get("usage"),
                prior_tool_calls=receipt.get("tool_calls"),
            )
            return self._compact_digest(updated)

        fut = self._executor.submit(
            self._execute_job,
            job_id=job_id,
            prompts=prompts,
            model=model,
            project=project,
            location=location,
            reroute_note=reroute_note,
            allowed_paths=allowed_paths,
            timeout_seconds=timeout_seconds,
            enabled_tools=None,
            disabled_tools=None,
            enable_subagents=False,
            wt_dir=wt_dir,
            started_at=started_at,
            prior_usage=receipt.get("usage"),
            prior_tool_calls=receipt.get("tool_calls"),
        )
        with self._lock:
            self._futures[job_id] = fut
        return {"job_id": job_id, "status": "running", "prompts_count": len(prompts)}

    def delegate_cancel(self, job_id: str, *, cleanup_worktree: bool = True) -> dict[str, Any]:
        """Cancel a job and clean up its worktree while preserving its receipt."""
        with self._lock:
            self._cancelled.add(job_id)
            fut = self._futures.get(job_id)
            if fut is not None:
                fut.cancel()

        receipt = self.store.read_receipt(job_id)
        wt_dir = Path(str(receipt.get("worktree_dir") or ""))
        if cleanup_worktree and wt_dir.exists():
            self._cleanup_worktree(wt_dir)

        if receipt.get("status") == "running":
            receipt["status"] = "cancelled"
            receipt["error"] = "Cancelled by user via delegate_cancel."
            receipt["finished_at_epoch"] = round(time.time(), 3)
            self.store.write_receipt(
                job_id,
                receipt,
                workspace_root=str(self.workspace_root),
            )

        return {
            "job_id": job_id,
            "status": receipt.get("status"),
            "cancelled": True,
        }

    @staticmethod
    def _compact_digest(receipt: dict[str, Any]) -> dict[str, Any]:
        """Return a token-bounded digest suitable for Claude Code's context window."""
        summary_text = str(receipt.get("summary") or "")
        clipped_summary, truncated, est_tokens = truncate_to_token_budget(
            summary_text, MAX_TOOL_RESPONSE_TOKENS
        )
        return {
            "job_id": receipt.get("job_id"),
            "kind": receipt.get("kind", "delegate"),
            "status": receipt.get("status"),
            "model": receipt.get("model"),
            "vertex_location": receipt.get("vertex_location"),
            "reroute_note": receipt.get("reroute_note"),
            "allowed_paths": receipt.get("allowed_paths", []),
            "scope_violations": receipt.get("scope_violations", []),
            "files_changed": receipt.get("files_changed", []),
            "diff_tokens_estimated": receipt.get("diff_tokens_estimated", 0),
            "tool_call_count": receipt.get("tool_call_count", 0),
            "usage": receipt.get("usage", {}),
            "cost": receipt.get("cost", {}),
            "summary": clipped_summary,
            "summary_truncated": bool(receipt.get("summary_truncated") or truncated),
            "summary_tokens_estimated": est_tokens,
            "applied": bool(receipt.get("applied", False)),
            "error": receipt.get("error"),
        }
