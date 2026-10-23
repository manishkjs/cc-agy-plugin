"""Run a validated batch plan inside an isolated job copy and record it like any other job.

The Lead plans; this module types. For each task, the writer ladder produces one file, the plugin
writes it into the job copy and runs the task's checks; only a passing file counts as landed. The
finished job gets the same receipt, diff, 3-way-merge apply and revert path as ``delegate_start``
jobs, plus per-task records (state, attempts, writer, attempt history, added-line hashes).

Concurrency model: writer calls run in parallel (they are network bound), while writing a candidate
into the job copy, running its checks, and keeping or rolling it back happens one task at a time
under ``apply_lock``. When a task's ``context_files`` overlaps another task's target path, prompt
rendering also snapshots under ``apply_lock`` so neither checks nor prompts ever see another task's
unverified candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import threading
import time
from typing import Any, Callable
import uuid

from server.batch_journal import BatchJournal
from server.batch_plan import BatchPlan, FileTask, needs_review
from server.model_routes import BatchSettings
from server.receipts import enforce_daily_budget
from server.repo_map import build_repo_map
from server.task_graph import DONE_STATES, run_task_graph
from server.verify import failure_tail, has_behavioral_check, run_commands
from server.writers import (
    AppliedAnswer,
    WRITER_SYSTEM_PROMPT,
    Writer,
    WriterAuthError,
    WriterRequest,
    create_shared_cache,
    delete_shared_cache,
    make_agent_writer,
    make_claude_writer,
    make_genai_writer,
    render_task_prompt,
    run_writer_ladder,
)

WriterFactory = Callable[[str, BatchSettings, dict[str, Any]], Writer | None]
ProgressFn = Callable[[int, int, str], None]
# Returns a problem description when the written bytes break a tool-specific rule, else None.
ContentGate = Callable[[FileTask, bytes, Path], "str | None"]
CacheFactory = Callable[[dict[str, Any], str, str, str], "str | None"]

DIGEST_TAIL_CHARS = 1_200  # ~300 tokens per failing task
DIGEST_TAIL_TASKS = 3  # only the first few failures carry a full tail; the rest point at batch_task_log
DIGEST_CHAR_BUDGET = 2_400  # ~600 tokens
SHARED_CONTEXT_CAP_BYTES = 200_000
RESUMABLE_KINDS = frozenset({"file_batch", "draft_document", "write_tests_from_cases", "replicate_edit"})
_PLAN_LOCKS: dict[str, threading.Lock] = {}
_PLAN_LOCKS_GUARD = threading.Lock()


def default_writer_factory(name: str, settings: BatchSettings, auth: dict[str, Any]) -> Writer | None:
    if name == "genai":
        return make_genai_writer(auth, settings.model, thinking_level=settings.thinking_level,
                                 cached_content=auth.get("_shared_cache"))
    if name == "agent":
        return make_agent_writer(
            auth,
            settings.model,
            thinking_level=settings.thinking_level,
            read_budget=settings.read_budget,
            workspace_dir=auth.get("_workspace_dir"),
        )
    if name == "claude":
        return make_claude_writer(settings.claude_model) if settings.claude_writer else None
    return None


@dataclass
class TaskRecord:
    task: FileTask
    state: str = "pending"
    attempts: int = 0
    writer: str = ""
    detail: str = ""
    cost_cents: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    pricing_known: bool = True


class BatchControls:
    """Batch-wide throttle, budget and Claude escalation cap shared by every task."""

    def __init__(self, settings: BatchSettings, stop: threading.Event) -> None:
        self.settings = settings
        self.stop = stop
        self._lock = threading.Lock()
        self._cool_until = 0.0
        self.spent_cents = 0.0
        self.claude_calls = 0
        self.stop_reason = ""

    def before_call(self, writer_name: str) -> None:
        with self._lock:
            wait = self._cool_until - time.time()
        if wait > 0:
            time.sleep(min(wait, 60.0))
        if writer_name == "claude":
            with self._lock:
                self.claude_calls += 1

    def on_transient(self, delay: float) -> None:
        with self._lock:
            self._cool_until = max(self._cool_until, time.time() + delay)

    def claude_allowed(self) -> bool:
        with self._lock:
            return self.claude_calls < self.settings.max_claude_escalations

    def add_cost(self, cents: float) -> None:
        with self._lock:
            self.spent_cents += cents
            limit = self.settings.max_cost_cents
            if limit is not None and self.spent_cents >= limit and not self.stop.is_set():
                self.stop_reason = f"cost budget of {limit} cents reached"
                self.stop.set()


class _LadderWriters(dict):
    """Writer table whose Claude rung disappears once the batch's escalation cap is used up."""

    def __init__(self, base: dict[str, Writer | None], controls: BatchControls) -> None:
        super().__init__(base)
        self._controls = controls

    def get(self, key: str, default: Any = None) -> Any:
        if key == "claude" and not self._controls.claude_allowed():
            return None
        return super().get(key, default)


def _workspace_state(root: Path) -> str:
    """Fingerprint of HEAD plus uncommitted state, so a moved workspace never resumes a stale job copy."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, check=False).stdout
    status = subprocess.run(["git", "status", "--porcelain=v1", "-z"], cwd=str(root), capture_output=True, check=False).stdout
    diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=str(root), capture_output=True, check=False).stdout
    return hashlib.sha256(head.encode() + b"\0" + status + b"\0" + diff).hexdigest()


class FileBatchRunner:
    def __init__(
        self,
        job_manager: Any,
        *,
        writer_factory: WriterFactory | None = None,
        progress: ProgressFn | None = None,
        cache_factory: CacheFactory | None = None,
        stop: threading.Event | None = None,
    ) -> None:
        self.jm = job_manager
        self.store = job_manager.store
        self.root = Path(job_manager.workspace_root).resolve()
        self.writer_factory = writer_factory or default_writer_factory
        self.progress = progress
        self.stop = stop or threading.Event()
        # Real Gemini context caching only pairs with the real writers; injected writers get none by default.
        if cache_factory is not None:
            self.cache_factory, self.cache_cleanup = cache_factory, (lambda auth, name: None)
        elif writer_factory is None:
            self.cache_factory, self.cache_cleanup = create_shared_cache, delete_shared_cache
        else:
            self.cache_factory, self.cache_cleanup = None, (lambda auth, name: None)

    # ------------------------------------------------------------------ job copy

    def _find_resumable(self, plan: BatchPlan, ws_state: str) -> dict[str, Any] | None:
        for rec in reversed(self.store.list_receipts(workspace_root=self.root)):
            if (rec.get("kind") in RESUMABLE_KINDS and rec.get("plan_fingerprint") == plan.raw_fingerprint
                    and rec.get("workspace_state") == ws_state and not rec.get("applied")
                    and rec.get("status") != "error"
                    and Path(str(rec.get("worktree_dir") or "")).is_dir()):
                return rec
        return None

    def _open_job(self, plan: BatchPlan, resume_job_id: str | None, ws_state: str):
        if resume_job_id:
            rec = self.store.read_receipt(resume_job_id)
            if rec.get("kind") not in RESUMABLE_KINDS or rec.get("applied"):
                raise ValueError(f"job {resume_job_id!r} is not an unapplied batch job")
            wt = Path(str(rec.get("worktree_dir") or ""))
            if not wt.is_dir():
                raise ValueError(f"job {resume_job_id!r} has no job copy left to resume")
            return (resume_job_id, wt, str(rec.get("baseline_commit_sha") or "HEAD"),
                    dict(rec.get("baseline_file_hashes") or {}), rec)
        prior = self._find_resumable(plan, ws_state)
        if prior:
            return (str(prior["job_id"]), Path(str(prior["worktree_dir"])), str(prior.get("baseline_commit_sha") or "HEAD"),
                    dict(prior.get("baseline_file_hashes") or {}), prior)
        job_id = f"job-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        wt, baseline_sha, baseline_hashes = self.jm._prepare_worktree(job_id)
        return job_id, wt, baseline_sha, baseline_hashes, None

    def _plan_lock(self, plan: BatchPlan) -> tuple[threading.Lock, Path]:
        with _PLAN_LOCKS_GUARD:
            lock = _PLAN_LOCKS.setdefault(plan.raw_fingerprint, threading.Lock())
        lock_dir = self.store.data_dir / "file_batches"
        lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        return lock, lock_dir / f"{plan.raw_fingerprint[:24]}.lock"

    # ------------------------------------------------------------------ per task

    def _shared_text(self, plan: BatchPlan, wt: Path) -> str:
        parts, used = [], 0
        for rel in plan.shared_context:
            path = wt / rel
            if not path.is_file():
                continue
            body = path.read_text(encoding="utf-8", errors="replace")
            if used + len(body) > SHARED_CONTEXT_CAP_BYTES:
                break
            parts.append(f"## Shared `{rel}`\n{body}")
            used += len(body)
        return "\n\n".join(parts)

    def _make_verify(self, task, wt, plan, settings, log_dir, apply_lock, content_gate=None):
        def restore(answer: AppliedAnswer, target: Path) -> None:
            if answer.previous_bytes is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(answer.previous_bytes)

        def verify(answer: AppliedAnswer) -> tuple[bool, str]:
            with apply_lock:
                target = wt / task.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(answer.new_bytes)
                if content_gate is not None:
                    problem = content_gate(task, answer.new_bytes, wt)
                    if problem:
                        restore(answer, target)
                        return False, problem
                cmds: list[str] = []
                if plan.formatter:
                    cmds.append(plan.formatter.replace("{path}", shlex.quote(task.path)))
                cmds.extend(task.checks)
                if task.repro_test:
                    cmds.append(task.repro_test)
                results = run_commands(cmds, wt, timeout=settings.check_timeout, log_dir=log_dir) if cmds else []
                if all(r.passed for r in results):
                    return True, ""
                restore(answer, target)
                return False, failure_tail(results)
        return verify

    @staticmethod
    def _reaches(by_id: dict[str, FileTask], start: str, target: str) -> bool:
        todo, seen = [start], set()
        while todo:
            node = todo.pop()
            if node == target:
                return True
            if node in seen or node not in by_id:
                continue
            seen.add(node)
            todo.extend(by_id[node].depends_on)
        return False

    def _already_done(self, task: FileTask, plan: BatchPlan, journal: BatchJournal, wt: Path) -> bool:
        """Unchanged if its own output is still on disk, or a later task in the same file chain is."""
        if journal.is_unchanged(task, wt):
            return True
        own = journal.entry(task.task_id)
        if not own or own.get("task_fingerprint") != task.fingerprint():
            return False
        by_id = {t.task_id: t for t in plan.tasks}
        for later in plan.tasks:
            if later.path != task.path or later.task_id == task.task_id:
                continue
            if not self._reaches(by_id, later.task_id, task.task_id):
                continue
            entry = journal.entry(later.task_id)
            if (entry and float(entry.get("landed_at_epoch") or 0) >= float(own.get("landed_at_epoch") or 0)
                    and journal.is_unchanged(later, wt)):
                return True
        return False

    # ------------------------------------------------------------------ main

    def run(
        self,
        plan: BatchPlan,
        settings: BatchSettings,
        auth: dict[str, Any],
        *,
        allowed_paths: list[str] | None = None,
        extra_acceptance: list[str] | None = None,
        resume_job_id: str | None = None,
        edits_only: bool = False,
        kind: str = "file_batch",
        extra_receipt: dict[str, Any] | None = None,
        content_gate: ContentGate | None = None,
        shared_note: str = "",
    ) -> dict[str, Any]:
        enforce_daily_budget(self.store, workspace_root=self.root)
        lock, lock_file = self._plan_lock(plan)
        with lock, open(lock_file, "w", encoding="utf-8") as handle:
            # One run per plan at a time (threads and processes); a concurrent identical call waits,
            # then resumes the finished job and finds every task unchanged.
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                return self._run_locked(
                    plan, settings, auth, allowed_paths=allowed_paths, extra_acceptance=extra_acceptance,
                    resume_job_id=resume_job_id, edits_only=edits_only, kind=kind, extra_receipt=extra_receipt or {},
                    content_gate=content_gate, shared_note=shared_note,
                )
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _run_locked(self, plan, settings, auth, *, allowed_paths, extra_acceptance, resume_job_id, edits_only,
                    kind, extra_receipt, content_gate, shared_note) -> dict[str, Any]:
        ws_state = _workspace_state(self.root)
        job_id, wt, baseline_sha, baseline_hashes, prior = self._open_job(plan, resume_job_id, ws_state)
        started = round(time.time(), 3)
        try:
            return self._execute(
                plan, settings, auth, job_id=job_id, wt=wt, baseline_sha=baseline_sha, baseline_hashes=baseline_hashes,
                prior=prior, started=started, ws_state=ws_state, allowed_paths=allowed_paths,
                extra_acceptance=extra_acceptance, edits_only=edits_only, kind=kind, extra_receipt=extra_receipt,
                content_gate=content_gate, shared_note=shared_note,
            )
        except Exception as exc:  # noqa: BLE001 - never leave a job copy without a receipt
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
            self.store.write_receipt(job_id, {
                "job_id": job_id, "kind": kind, "status": "error", "model": settings.model,
                "worktree_dir": str(wt), "baseline_commit_sha": baseline_sha, "baseline_file_hashes": baseline_hashes,
                "started_at_epoch": started, "finished_at_epoch": round(time.time(), 3), "files_changed": [],
                "applied": False, "in_place": False, "plan_fingerprint": plan.raw_fingerprint, "workspace_state": ws_state,
                "prompts": [f"{t.task_id}: {t.intent}" for t in plan.tasks], "error": error, "usage": {},
                "precomputed_cost": {"total_cost_cents": 0.0, "total_cost_usd": 0.0, "pricing_known": False},
            }, workspace_root=str(self.root), api_key=auth.get("api_key"))
            return {"job_id": job_id, "status": "error", "error": error, "is_error": True}

    def _execute(self, plan, settings, auth, *, job_id, wt, baseline_sha, baseline_hashes, prior, started, ws_state,
                 allowed_paths, extra_acceptance, edits_only, kind, extra_receipt, content_gate, shared_note):
        journal = BatchJournal(self.store.data_dir, job_id)
        log_dir = self.store.data_dir / "file_batches" / f"{job_id}.logs"
        records = {t.task_id: TaskRecord(t) for t in plan.tasks}
        by_id = {t.task_id: t for t in plan.tasks}
        controls = BatchControls(settings, self.stop)
        apply_lock = threading.Lock()
        auth_error: list[str] = []
        lock = threading.Lock()
        settled = {"n": 0}
        landed_paths = journal.landed_paths()

        use_repo_map = os.environ.get("CC_AGY_REPO_MAP", "1").strip().lower() not in ("0", "false", "no")
        # Map the live workspace (stable path, so the cache hits across jobs), not the per-job copy.
        repo_map_text = build_repo_map(self.root, cache_dir=self.store.data_dir) if use_repo_map else ""
        repo_map_block = f"## Repository Map (Cached)\n{repo_map_text}" if repo_map_text else ""
        shared_text = "\n\n".join(x for x in (repo_map_block, shared_note.strip(), self._shared_text(plan, wt)) if x)
        writer_auth = dict(auth)
        writer_auth["_workspace_dir"] = wt
        cache_name = None
        if settings.writer == "genai" and shared_text and len(plan.tasks) > 1 and self.cache_factory:
            cache_name = self.cache_factory(auth, settings.model, WRITER_SYSTEM_PROMPT, shared_text)
            writer_auth["_shared_cache"] = cache_name
        writers = _LadderWriters({
            settings.writer: self.writer_factory(settings.writer, settings, writer_auth),
            "claude": self.writer_factory("claude", settings, auth) if settings.claude_writer else None,
        }, controls)

        def dependency_context(task: FileTask) -> tuple[str, ...]:
            seen: list[str] = []
            todo = list(task.depends_on)
            while todo:
                dep = by_id.get(todo.pop(0))
                if dep is None or dep.path in seen or dep.path == task.path:
                    continue
                seen.append(dep.path)
                todo.extend(dep.depends_on)
            return tuple(seen[:6])

        target_paths = {t.path for t in plan.tasks}

        def build_request(task: FileTask, writer_name: str, tail: str) -> WriterRequest:
            uses_cache = writer_name == "genai" and cache_name
            dep_ctx = dependency_context(task)
            if any(c in target_paths for c in task.context_files):
                with apply_lock:
                    rendered = render_task_prompt(task, wt, tail, extra_context=dep_ctx)
            else:
                rendered = render_task_prompt(task, wt, tail, extra_context=dep_ctx)
            return WriterRequest(
                task=task,
                prompt=rendered,
                shared_prefix="" if uses_cache else shared_text,
                failure_tail=tail,
                model=settings.claude_model if writer_name == "claude" else settings.model,
            )

        def run_one(task: FileTask) -> str:
            rec = records[task.task_id]
            try:
                if self._already_done(task, plan, journal, wt):
                    rec.state = "unchanged"
                    return "unchanged"
                if task.repro_test:
                    with apply_lock:
                        pre = run_commands([task.repro_test], wt, timeout=settings.check_timeout, log_dir=log_dir)
                    if pre and pre[0].passed:
                        rec.state = "failed"
                        rec.detail = f"repro_test already passes before the change: {task.repro_test}"
                        rec.history.append({"writer": "", "result": "repro_already_passes", "detail": task.repro_test})
                        return "failed"
                outcome = run_writer_ladder(
                    task, routed=settings.writer, writers=writers,
                    verify=self._make_verify(task, wt, plan, settings, log_dir, apply_lock, content_gate),
                    build_request=build_request, root=wt, edits_only=edits_only,
                    allow_existing=task.path in landed_paths,
                    before_call=controls.before_call, on_transient=controls.on_transient,
                )
            except WriterAuthError as exc:
                with lock:
                    auth_error.append(str(exc))
                self.stop.set()
                rec.state, rec.detail = "failed", str(exc)
                rec.history.append({"writer": "", "result": "auth_error", "detail": str(exc)[-300:]})
                return "failed"
            except Exception as exc:  # noqa: BLE001 - a crashing writer fails its task, visibly
                rec.state, rec.detail = "failed", f"{type(exc).__name__}: {exc}"[-DIGEST_TAIL_CHARS:]
                rec.history.append({"writer": "", "result": "writer_error", "detail": rec.detail[-300:]})
                return "failed"
            rec.attempts, rec.writer, rec.detail = outcome.attempts, outcome.writer_used, outcome.detail
            rec.cost_cents, rec.usage, rec.history = outcome.cost_cents, outcome.usage, outcome.history
            rec.pricing_known = outcome.pricing_known
            rec.state = outcome.state
            controls.add_cost(outcome.cost_cents)
            if outcome.state == "landed":
                journal.record_landed(task, wt)
            return outcome.state

        def on_settle(task: FileTask, state: str) -> None:
            rec = records[task.task_id]
            if state == "held" and rec.state == "pending":
                rec.state = "held"
                if controls.stop_reason:
                    rec.detail = controls.stop_reason
            with lock:
                settled["n"] += 1
                done = settled["n"]
            if self.progress is not None:
                self.progress(done, len(plan.tasks), f"{task.task_id}: {state}")

        from server.jobs import link_host_node_modules_farm, unlink_host_node_modules_farm

        nm_links = (
            link_host_node_modules_farm(self.root, wt)
            if os.environ.get("CC_AGY_LINK_NODE_MODULES", "1").strip().lower() not in ("0", "false", "no", "off")
            else []
        )
        try:
            try:
                run_task_graph(list(plan.tasks), run_one, max_parallel=settings.max_parallel, on_settle=on_settle,
                               warm_first=bool(cache_name) or settings.claude_writer, stop=self.stop)
            finally:
                if cache_name:
                    self.cache_cleanup(auth, cache_name)

            acceptance_cmds = list(plan.acceptance) + list(extra_acceptance or [])
            acceptance_results: list[dict[str, Any]] = []
            any_done = any(r.state in DONE_STATES for r in records.values())
            if acceptance_cmds and any_done and not auth_error:
                acceptance_results = [r.as_dict() for r in run_commands(
                    acceptance_cmds, wt, timeout=max(settings.check_timeout, 600.0), stop_on_failure=False, log_dir=log_dir)]
            acceptance_passed = all(r["passed"] for r in acceptance_results) if acceptance_results else None
        finally:
            unlink_host_node_modules_farm(nm_links)

        return self._finish(
            job_id=job_id, wt=wt, baseline_sha=baseline_sha, baseline_hashes=baseline_hashes, plan=plan,
            settings=settings, auth=auth, records=records, started=started, prior=prior, ws_state=ws_state,
            allowed_paths=allowed_paths, acceptance_cmds=acceptance_cmds, acceptance_results=acceptance_results,
            acceptance_passed=acceptance_passed, auth_error=auth_error[0] if auth_error else None, kind=kind,
            extra_receipt=extra_receipt, stop_reason=controls.stop_reason,
        )

    def _finish(self, *, job_id, wt, baseline_sha, baseline_hashes, plan, settings, auth, records, started, prior,
                ws_state, allowed_paths, acceptance_cmds, acceptance_results, acceptance_passed, auth_error, kind,
                extra_receipt, stop_reason):
        from server.jobs import _matches_allowed_paths
        from server.receipts import extract_diff_provenance

        shutil.rmtree(Path(wt) / ".ctx-session-reads", ignore_errors=True)
        diff_text, files_changed, symlink_escapes = self.jm._collect_worktree_diff(
            wt, baseline_sha, allowed_paths=allowed_paths)
        baseline_hashes = self.jm._compute_changed_baseline_hashes(wt, baseline_sha, files_changed, baseline_hashes)
        provenance = extract_diff_provenance(diff_text)
        for fpath in files_changed:
            if fpath not in provenance["files_changed"]:
                provenance["files_changed"].append(fpath)
        self.store.diff_path(job_id).write_text(diff_text, encoding="utf-8")
        scope_violations = list(symlink_escapes)
        if allowed_paths:
            scope_violations += [f for f in files_changed
                                 if not _matches_allowed_paths(f, allowed_paths) and f not in scope_violations]

        prior_rows = {r.get("task_id"): r for r in (prior or {}).get("tasks") or [] if isinstance(r, dict)}
        counts = {s: 0 for s in ("landed", "unchanged", "failed", "held")}
        usage_run: dict[str, int] = {}
        run_cents = 0.0
        priced = True
        has_behav_acceptance = (
            bool(acceptance_cmds)
            and acceptance_passed is True
            and has_behavioral_check(acceptance_cmds)
        )
        task_rows: list[dict[str, Any]] = []
        for rec in records.values():
            counts[rec.state] = counts.get(rec.state, 0) + 1
            run_cents += rec.cost_cents
            priced = priced and rec.pricing_known
            for k, v in rec.usage.items():
                usage_run[k] = usage_run.get(k, 0) + int(v)
            per_file = (provenance.get("per_file") or {}).get(rec.task.path) or {}
            earlier = prior_rows.get(rec.task.task_id) or {}
            reused = rec.state == "unchanged"
            history = list(earlier.get("history") or []) + ([] if reused else rec.history)
            task_cmds = list(rec.task.checks) + ([rec.task.repro_test] if rec.task.repro_test else [])
            task_done = rec.state in DONE_STATES
            has_any_check = task_done and (bool(task_cmds) or (bool(acceptance_cmds) and acceptance_passed is True))
            behav = task_done and (has_behav_acceptance or has_behavioral_check(task_cmds))
            verif = "behavioral" if behav else ("syntax_only" if has_any_check else "unverified")
            task_rows.append({
                "task_id": rec.task.task_id, "path": rec.task.path, "state": rec.state,
                "attempts": int(earlier.get("attempts") or 0) if reused else rec.attempts,
                "writer": str(earlier.get("writer") or "") if reused else rec.writer,
                "detail": rec.detail[-DIGEST_TAIL_CHARS:],
                "cost_cents": round(rec.cost_cents + float(earlier.get("cost_cents") or 0), 4),
                "added_line_sha256": list(per_file.get("added_line_sha256") or []),
                "task_fingerprint": rec.task.fingerprint(), "history": history[-12:],
                "review_required": needs_review(rec.task.path),
                "verification": verif,
            })

        # Accumulate spend across resumed runs so the receipt always carries the job's whole cost.
        prior_cost = (prior or {}).get("cost") or {}
        prior_usage = (prior or {}).get("usage") or {}
        total_cents = run_cents + float(prior_cost.get("total_cost_cents") or 0.0)
        usage_total = {k: int(v) for k, v in prior_usage.items() if isinstance(v, (int, float))}
        for k, v in usage_run.items():
            usage_total[k] = int(usage_total.get(k, 0)) + int(v)
        priced = priced and bool(prior_cost.get("pricing_known", True))

        all_done = all(r.state in DONE_STATES for r in records.values())
        if auth_error:
            status = "error"
        elif scope_violations:
            status = "scope_violation"
        elif all_done and acceptance_passed is not False:
            status = "ok"
        else:
            status = "checkpoint"
        finished = round(time.time(), 3)
        review = sorted({r["path"] for r in task_rows if r["review_required"] and r["state"] in DONE_STATES})
        receipt = {
            "job_id": job_id, "kind": kind, "status": status, "auth_mode": auth.get("auth_mode"),
            "model": settings.model, "vertex_project": auth.get("project"), "vertex_location": auth.get("location"),
            "prompts": [f"{t.task_id}: {t.intent}" for t in plan.tasks],
            "allowed_paths": list(allowed_paths or []), "scope_violations": scope_violations,
            "symlink_escapes": list(symlink_escapes), "worktree_dir": str(wt), "baseline_commit_sha": baseline_sha,
            "baseline_file_hashes": baseline_hashes,
            "started_at_epoch": float((prior or {}).get("started_at_epoch") or started), "finished_at_epoch": finished,
            "duration_seconds": round(finished - started, 3), "usage": usage_total,
            "precomputed_cost": {"total_cost_cents": round(total_cents, 4), "total_cost_usd": round(total_cents / 100.0, 6),
                                 "pricing_known": priced},
            "files_changed": files_changed, "provenance": provenance, "applied": False, "in_place": False,
            "acceptance": acceptance_cmds, "acceptance_results": acceptance_results, "acceptance_passed": acceptance_passed,
            "error": auth_error or (stop_reason or None), "plan_fingerprint": plan.raw_fingerprint,
            "workspace_state": ws_state, "tasks": task_rows, "resumed": prior is not None, "review_required": review,
            "writer_route": {"writer": settings.writer, "model": settings.model, "thinking_level": settings.thinking_level,
                             "read_budget": settings.read_budget,
                             "claude_writer": settings.claude_model if settings.claude_writer else None},
            "external_worktree_tamper": False, "tool_call_count": 0, "tool_calls": [],
        }
        receipt.update(extra_receipt)
        self.store.write_receipt(job_id, receipt, workspace_root=str(self.root), api_key=auth.get("api_key"))
        return self._digest(job_id, status, task_rows, counts, run_cents, total_cents, usage_run, acceptance_passed,
                            auth_error or stop_reason, files_changed, scope_violations, review)

    @staticmethod
    def _digest(job_id, status, rows, counts, run_cents, total_cents, usage, acceptance_passed, error, files_changed,
                scope_violations, review) -> dict[str, Any]:
        tokens = sum(int(v) for k, v in usage.items() if k.endswith("_tokens") or k == "total_token_count")
        tasks: list[dict[str, Any]] = []
        tails_left = DIGEST_TAIL_TASKS
        for row in rows:
            item = {"task_id": row["task_id"], "state": row["state"], "attempts": row["attempts"], "writer": row["writer"]}
            if row["state"] == "failed" and row["detail"]:
                item["check_tail"] = row["detail"][-(DIGEST_TAIL_CHARS if tails_left > 0 else 160):]
                tails_left -= 1
            tasks.append(item)
        digest: dict[str, Any] = {
            "job_id": job_id, "status": status,
            "totals": {**counts, "files_changed": len(files_changed), "cost_cents": round(run_cents, 4),
                       "job_cost_cents": round(total_cents, 4), "tokens": tokens},
            "acceptance_passed": acceptance_passed, "tasks": tasks,
        }
        if error:
            digest["error"] = error
        if scope_violations:
            digest["scope_violations"] = scope_violations[:10]
        if review:
            digest["review_required"] = review[:10]
        syntax_only = [
            r["task_id"] for r in rows
            if r.get("verification") in ("syntax_only", "unverified") and r["state"] in DONE_STATES
        ]
        if syntax_only:
            digest["syntax_only_tasks"] = syntax_only[:10]
        if status in ("ok", "checkpoint"):
            digest["next"] = f"get_diff(job_id={job_id!r}) then delegate_apply(job_id={job_id!r})"
        if any(t["state"] == "failed" for t in tasks):
            digest["details"] = f"batch_task_log(job_id={job_id!r}, task_id=...) for attempt history and logs"
        # Keep the reply near ~600 tokens on large batches: collapse quiet rows, then trim tails, then held rows.
        if len(json.dumps(digest)) > DIGEST_CHAR_BUDGET:
            quiet = [t for t in tasks if t["state"] in ("landed", "unchanged") and t["attempts"] <= 1]
            digest["tasks"] = [t for t in tasks if t not in quiet]
            digest["quiet_task_ids"] = [t["task_id"] for t in quiet]
        if len(json.dumps(digest)) > DIGEST_CHAR_BUDGET:
            for t in digest["tasks"]:
                if "check_tail" in t:
                    t["check_tail"] = t["check_tail"][-300:]
        if len(json.dumps(digest)) > DIGEST_CHAR_BUDGET:
            held = [t["task_id"] for t in digest["tasks"] if t["state"] == "held"]
            digest["tasks"] = [t for t in digest["tasks"] if t["state"] != "held"]
            if held:
                digest["held_task_ids"] = held
        if len(json.dumps(digest)) > DIGEST_CHAR_BUDGET:
            failed = [t for t in digest["tasks"] if t["state"] == "failed"]
            keep = failed[:DIGEST_TAIL_TASKS]
            digest["tasks"] = [t for t in digest["tasks"] if t["state"] != "failed" or t in keep]
            digest["more_failed_task_ids"] = [t["task_id"] for t in failed[DIGEST_TAIL_TASKS:]]
        return digest


def batch_task_log(store: Any, job_id: str, task_id: str) -> dict[str, Any]:
    """Full per-task record for a batch job: attempt history, last failure, recent check log paths."""
    rec = store.read_receipt(job_id)
    if rec.get("kind") not in RESUMABLE_KINDS:
        raise ValueError(f"job {job_id!r} is not a batch job")
    row = next((t for t in rec.get("tasks") or [] if t.get("task_id") == task_id), None)
    if row is None:
        raise ValueError(f"task {task_id!r} is not in job {job_id!r}")
    log_dir = Path(store.data_dir) / "file_batches" / f"{job_id}.logs"
    logs: list[str] = []
    if log_dir.is_dir():
        logs = [str(p) for p in sorted(log_dir.glob("check-*.log"), key=lambda p: p.stat().st_mtime)[-6:]]
    return {"job_id": job_id, "task": {k: v for k, v in row.items() if k != "added_line_sha256"}, "recent_check_logs": logs}


PROBE_TASK = FileTask(task_id="probe", path="probe.txt", action="create", intent="Write the single word ok.")


def probe_writers(settings: BatchSettings, auth: dict[str, Any], *, writer_factory: WriterFactory | None = None) -> dict[str, Any]:
    """Send one tiny request through each configured writer and report ready / warn / fail with a fix."""
    from dataclasses import replace
    from server.writers import WriterConfigError, WriterTransportError

    factory = writer_factory or default_writer_factory
    probe_settings = replace(settings, read_budget=0)
    names = [probe_settings.writer] + (["claude"] if probe_settings.claude_writer else [])
    rows: list[dict[str, Any]] = []
    for name in names:
        row: dict[str, Any] = {"writer": name, "model": probe_settings.claude_model if name == "claude" else probe_settings.model}
        try:
            writer = factory(name, probe_settings, auth)
            if writer is None:
                row.update(state="fail",
                           fix="Install Claude Code (`claude` on PATH) or set CC_AGY_CLAUDE_BIN, or pass claude_writer=false.")
            else:
                reply = writer(WriterRequest(task=PROBE_TASK, prompt='Reply with {"path": "probe.txt", "content": "ok"}',
                                             max_output_tokens=64))
                row.update(state="ready", cost_cents=round(reply.cost_cents, 4), answered=bool(reply.text.strip()))
        except WriterAuthError as exc:
            if auth.get("api_key"):
                fix = "Check GEMINI_API_KEY / userConfig.gemini_api_key."
            elif auth.get("auth_mode") == "gemini_enterprise":
                fix = (
                    "Run `~/.local/share/cc-agy-plugin/bin/agy` once in a terminal to sign in with your "
                    "Gemini Enterprise Google account (or set `ANTIGRAVITY_OAUTH_TOKEN`)."
                )
            else:
                fix = f"Run `gcloud auth application-default login` and enable Vertex AI on project {auth.get('project')!r}."
            if name == "claude":
                fix = "Run `claude` once to sign in, or set ANTHROPIC_API_KEY."
            row.update(state="fail", error=str(exc), fix=fix)
        except WriterConfigError as exc:
            if name == "claude":
                cfg_fix = "Check the `claude` CLI (2.1+) and that `claude -p` runs in a terminal."
            elif auth.get("auth_mode") == "gemini_enterprise":
                cfg_fix = (
                    "Run `python3 -m server.install --auth gemini-enterprise` "
                    "(or `bash scripts/install_agy_cli.sh`) to install the external `agy` CLI."
                )
            else:
                cfg_fix = str(exc)
            row.update(state="fail", error=str(exc), fix=cfg_fix)
        except WriterTransportError as exc:
            row.update(state="warn", error=str(exc), fix="Vendor busy or rate limited; batches will wait and retry.")
        except Exception as exc:  # noqa: BLE001 - report anything else as a failed probe with its type
            row.update(state="fail", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        rows.append(row)
    failed = [r for r in rows if r["state"] == "fail"]
    return {"status": "fail" if failed else "ok", "auth_mode": auth.get("auth_mode"), "writers": rows,
            "is_error": bool(failed)}
