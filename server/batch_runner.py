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
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import threading
import time
from typing import Any, Callable
import uuid

from server import file_lock as fcntl

from server.batch_journal import BatchJournal
from server.batch_plan import BatchPlan, FileTask, needs_review
from server.model_routes import BatchSettings
from server.receipts import BudgetExceededError, enforce_daily_budget, resolve_max_daily_cents
from server.repo_map import build_repo_map
from server.task_graph import DONE_STATES, run_task_graph
from server.verification_policy import classify_commands_for_batch
from server.verify import CommandResult, _prepare_parent_within, _rmtree_writable, failure_tail, run_commands
from server.writers import (
    AppliedAnswer,
    LadderOutcome,
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
    symlink_component,
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


def _claude_allowed_for_auth(auth: dict[str, Any]) -> bool:
    if auth.get("auth_mode") == "gemini_enterprise":
        return os.environ.get("CC_AGY_ALLOW_CLAUDE_IN_GE", "").strip().lower() in ("1", "true", "yes")
    return True


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
        if not settings.claude_writer or not _claude_allowed_for_auth(auth):
            return None
        return make_claude_writer(settings.claude_model)
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


# Writer-ladder history results that mean a writer call returned a reply (and was paid for).
_REPLY_RESULTS = frozenset({"landed", "checks_failed", "answer_rejected", "cut_off"})


class BatchControls:
    """Batch-wide throttle, budget and Claude escalation cap shared by every task.

    The cost cap covers the whole batch job, resumed runs included: spend, Claude escalations and
    per-writer call prices are seeded from the prior receipt. Each call's spend is recorded the moment
    it returns (``after_call``). The first call of each writer (or every call of a writer whose replies
    cost nothing so far) runs alone, so a price is known before parallel calls of that writer start.
    After that a call is admitted while the spend so far, plus what is reserved for calls in flight,
    plus one worst-case call of this writer (its most expensive call seen) fits the cap. When it does
    not fit only because of calls in flight, the caller waits for them to settle; the batch stops when
    even one more call cannot fit. Usage is only known once a reply arrives, so the cap is best effort:
    a call that costs more than every earlier call of its writer can still push spend past it.
    Claude escalation slots are reserved atomically in ``before_call`` and given back when the call
    produced no reply (a transient error or timeout), so retries never burn the cap.
    """

    _WAIT_SECONDS = 1.0

    def __init__(
        self,
        settings: BatchSettings,
        stop: threading.Event,
        prior: dict[str, Any] | None = None,
        *,
        store: Any = None,
        job_id: str = "",
        workspace_root: Path | None = None,
    ) -> None:
        self.settings = settings
        self.stop = stop
        self.store = store
        self.job_id = job_id
        self.workspace_root = workspace_root
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._local = threading.local()
        self._cool_until = 0.0
        self.spent_cents = 0.0
        self.claude_calls = 0
        self.stop_reason = ""
        self.in_flight = 0
        self.reserved_cents = 0.0
        self.unpriced_in_flight = 0
        self.unpriced_waiting = 0
        self.completed_calls = 0
        self.max_call_cents_by_writer: dict[str, float] = {}
        if prior:
            self._seed_from_prior(prior)

    def _seed_from_prior(self, prior: dict[str, Any]) -> None:
        self.spent_cents = _finite_cents((prior.get("cost") or {}).get("total_cost_cents"))
        history_claude = 0
        for row in prior.get("tasks") or []:
            for item in (row.get("history") or []) if isinstance(row, dict) else []:
                if not isinstance(item, dict) or str(item.get("result") or "") not in _REPLY_RESULTS:
                    continue
                writer = str(item.get("writer") or "")
                history_claude += writer == "claude"
                cents = _finite_cents(item.get("cost_cents"))
                if writer and cents > 0.0:
                    self.max_call_cents_by_writer[writer] = max(self.max_call_cents_by_writer.get(writer, 0.0), cents)
        recorded = prior.get("claude_calls")
        self.claude_calls = max(history_claude, int(recorded) if isinstance(recorded, int) else 0)

    @property
    def max_call_cents(self) -> float:
        return max(self.max_call_cents_by_writer.values(), default=0.0)

    def before_call(self, writer_name: str) -> bool:
        """Wait out any shared cool-down, then admit (True) or refuse (False) one writer call."""
        with self._lock:
            wait = self._cool_until - time.time()
        if wait > 0:
            time.sleep(min(wait, 60.0))
        daily_cap = resolve_max_daily_cents()
        with self._cond:
            admitted, estimate = self._admit_locked(writer_name, daily_cap=daily_cap)
        if not admitted:
            return False
        budget_guard = getattr(self.store, "budget", None) if self.store is not None else None
        if budget_guard is not None and daily_cap is not None:
            try:
                res_id = budget_guard.reserve(
                    self.job_id or "batch",
                    estimate_cents=estimate,
                    unpriced=estimate <= 0.0,
                    workspace_root=self.workspace_root,
                )
                self._local.daily_res_id = res_id
            except BudgetExceededError as exc:
                with self._cond:
                    self._release_local_slot_locked(writer_name, had_reply=False)
                    self._stop_for_daily_budget(exc)
                    self._cond.notify_all()
                return False
        else:
            self._local.daily_res_id = None
        return True

    def _admit_locked(self, writer_name: str, *, daily_cap: float | None = None) -> tuple[bool, float]:
        # Caller holds self._cond.
        estimate = 0.0
        waiting_unpriced = False
        has_any_cap = (self.settings.max_cost_cents is not None) or (daily_cap is not None)
        try:
            while True:
                if self.stop.is_set():
                    return False, 0.0
                if writer_name == "claude" and self.claude_calls >= self.settings.max_claude_escalations:
                    return False, 0.0
                limit = self.settings.max_cost_cents
                if not has_any_cap:
                    break
                if limit is not None and self.spent_cents >= limit:
                    self._stop_for_budget(limit, self.spent_cents)
                    return False, 0.0
                if self.unpriced_in_flight > 0:
                    # A call of a writer with no known price is running: nothing else starts until it
                    # reports what it cost.
                    self._cond.wait(timeout=self._WAIT_SECONDS)
                    continue
                estimate = self.max_call_cents_by_writer.get(writer_name, 0.0)
                if waiting_unpriced and (estimate > 0.0 or self.in_flight == 0):
                    # No longer waiting as an unpriced call (its writer got a price, or the slot is
                    # free): drop the registration before checking it, or this caller waits on itself.
                    waiting_unpriced = False
                    self.unpriced_waiting = max(0, self.unpriced_waiting - 1)
                    self._cond.notify_all()
                if estimate <= 0.0:
                    if self.in_flight > 0:
                        # No price for this writer yet: wait until its first call can run alone, and
                        # keep priced calls from starting meanwhile so it is not postponed forever.
                        if not waiting_unpriced:
                            waiting_unpriced = True
                            self.unpriced_waiting += 1
                        self._cond.wait(timeout=self._WAIT_SECONDS)
                        continue
                    break
                if self.unpriced_waiting > 0:
                    self._cond.wait(timeout=self._WAIT_SECONDS)
                    continue
                if limit is not None:
                    if self.spent_cents + estimate > limit:
                        self._stop_for_budget(limit, self.spent_cents + estimate)
                        return False, 0.0
                    if self.spent_cents + self.reserved_cents + estimate > limit and self.in_flight > 0:
                        # Only the calls in flight stand in the way: wait for them to settle, then re-check.
                        self._cond.wait(timeout=self._WAIT_SECONDS)
                        continue
                break
        finally:
            if waiting_unpriced:
                self.unpriced_waiting = max(0, self.unpriced_waiting - 1)
                self._cond.notify_all()
        if writer_name == "claude":
            self.claude_calls += 1
        unpriced = has_any_cap and estimate <= 0.0
        self.in_flight += 1
        self.unpriced_in_flight += int(unpriced)
        self.reserved_cents += estimate
        self._local.reserved = estimate
        self._local.unpriced = unpriced
        return True, estimate

    def _release_local_slot_locked(self, writer_name: str, *, had_reply: bool) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.reserved_cents = max(0.0, self.reserved_cents - float(getattr(self._local, "reserved", 0.0)))
        self._local.reserved = 0.0
        if getattr(self._local, "unpriced", False):
            self.unpriced_in_flight = max(0, self.unpriced_in_flight - 1)
            self._local.unpriced = False
        if not had_reply and writer_name == "claude":
            self.claude_calls = max(0, self.claude_calls - 1)

    def after_call(self, writer_name: str, reply: Any | None) -> None:
        """Release the call's in-flight slot and reservation, and record what it cost (0 when it raised)."""
        cents = _finite_cents(getattr(reply, "cost_cents", 0.0) if reply is not None else 0.0)
        daily_res_id = getattr(self._local, "daily_res_id", None)
        self._local.daily_res_id = None
        budget_guard = getattr(self.store, "budget", None) if self.store is not None else None
        if budget_guard is not None and (daily_res_id is not None or cents > 0.0):
            budget_guard.settle(
                daily_res_id,
                job_id=self.job_id or "batch",
                actual_cents=cents,
                workspace_root=self.workspace_root,
                record_live=True,
            )
        daily_exceeded: BudgetExceededError | None = None
        if self.store is not None and cents > 0.0 and resolve_max_daily_cents() is not None:
            try:
                enforce_daily_budget(self.store, workspace_root=self.workspace_root)
            except BudgetExceededError as exc:
                daily_exceeded = exc
        with self._cond:
            self._release_local_slot_locked(writer_name, had_reply=(reply is not None))
            if reply is not None:
                self.completed_calls += 1
                self.max_call_cents_by_writer[writer_name] = max(self.max_call_cents_by_writer.get(writer_name, 0.0), cents)
            self._add_cost_locked(cents)
            if daily_exceeded is not None:
                self._stop_for_daily_budget(daily_exceeded)
            self._cond.notify_all()

    def on_transient(self, delay: float) -> None:
        with self._lock:
            self._cool_until = max(self._cool_until, time.time() + delay)

    def claude_allowed(self) -> bool:
        with self._lock:
            return self.claude_calls < self.settings.max_claude_escalations

    def add_cost(self, cents: float) -> None:
        with self._cond:
            self._add_cost_locked(cents)
            self._cond.notify_all()

    def _add_cost_locked(self, cents: float) -> None:
        self.spent_cents += _finite_cents(cents)
        limit = self.settings.max_cost_cents
        if limit is not None and self.spent_cents >= limit:
            self._stop_for_budget(limit, self.spent_cents)

    def _stop_for_budget(self, limit: float, projected: float) -> None:
        # Caller holds self._lock.
        if not self.stop.is_set():
            self.stop_reason = (
                f"cost budget of {limit} cents reached (spent {self.spent_cents:.4f} cents; "
                f"one more call could bring it to {projected:.4f} cents)"
            )
            self.stop.set()

    def _stop_for_daily_budget(self, exc: BudgetExceededError) -> None:
        # Caller holds self._lock.
        if not self.stop.is_set():
            self.stop_reason = (
                f"daily cost budget of {exc.max_daily_cents:.2f} cents reached on {exc.utc_date} UTC "
                f"(spent {exc.spent_today_cents:.4f} cents today)"
            )
            self.stop.set()


def _finite_cents(value: Any) -> float:
    try:
        cents = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return cents if math.isfinite(cents) and cents > 0.0 else 0.0


class RunLedger:
    """Spend of the current run, recorded per writer call as it happens.

    Kept outside the task records so an error receipt can still report what the run paid for, and
    bucketed by UTC day so a run that crosses midnight is charged to the right day.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cents = 0.0
        self.usage: dict[str, int] = {}
        self.pricing_known = True
        self.by_utc_day: dict[str, float] = {}

    def record(self, reply: Any, *, now: float | None = None) -> None:
        cents = _finite_cents(getattr(reply, "cost_cents", 0.0))
        day = datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            self.cents += cents
            self.pricing_known = self.pricing_known and bool(getattr(reply, "pricing_known", True))
            for key, val in (getattr(reply, "usage", None) or {}).items():
                if isinstance(val, (int, float)) and math.isfinite(float(val)):
                    self.usage[key] = int(self.usage.get(key, 0)) + int(val)
            if cents > 0.0:
                self.by_utc_day[day] = self.by_utc_day.get(day, 0.0) + cents

    def merged_days(self, prior: dict[str, Any] | None) -> dict[str, float]:
        """Prior receipt's per-day spend plus this run's, rounded for the receipt."""
        merged: dict[str, float] = {}
        prior_days = (prior or {}).get("cost_by_utc_day")
        if isinstance(prior_days, dict) and prior_days:
            for day, cents in prior_days.items():
                merged[str(day)] = merged.get(str(day), 0.0) + _finite_cents(cents)
        elif prior:
            prior_cents = _finite_cents(((prior.get("cost") or {}).get("total_cost_cents")))
            stamp = float(prior.get("finished_at_epoch") or prior.get("started_at_epoch") or 0.0)
            if prior_cents > 0.0 and stamp > 0.0:
                day = datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d")
                merged[day] = prior_cents
        with self._lock:
            for day, cents in self.by_utc_day.items():
                merged[day] = merged.get(day, 0.0) + cents
        return {day: round(cents, 6) for day, cents in sorted(merged.items()) if cents > 0.0}


class _LadderWriters(dict):
    """Writer table whose Claude rung disappears once the batch's escalation cap is used up."""

    def __init__(self, base: dict[str, Writer | None], controls: BatchControls) -> None:
        super().__init__(base)
        self._controls = controls

    def get(self, key: str, default: Any = None) -> Any:
        if key == "claude" and not self._controls.claude_allowed():
            return None
        return super().get(key, default)


# Plugin-owned directories that may sit inside a workspace; their churn must not change its fingerprint.
_STATE_SKIP_PREFIXES = (".cc-agy-data/", ".worktrees/", ".ctx-session-reads/", ".cc-agy-review/", ".git/")
_UNTRACKED_HASH_MAX_FILES = 5_000
_UNTRACKED_HASH_MAX_FILE_BYTES = 16 * 1024 * 1024
_UNTRACKED_HASH_TOTAL_BYTES = 64 * 1024 * 1024


def _git_out(root: Path, *args: str) -> subprocess.CompletedProcess:
    from server.verify import run_git

    return run_git(args, root, capture_output=True, check=False)


def _hash_meta_into(digest: Any, path: Path) -> None:
    try:
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            target = os.readlink(path).encode("utf-8", "surrogateescape")
            digest.update(f"L{st.st_mode:o}:".encode() + target)
        else:
            digest.update(f"M{st.st_mode:o}:{st.st_size}:{st.st_mtime_ns}:{st.st_ctime_ns}".encode())
    except OSError:
        digest.update(b"?")


def _hash_file_into(digest: Any, path: Path, budget: list[int]) -> None:
    try:
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            digest.update(f"L{st.st_mode:o}:".encode() + os.readlink(path).encode("utf-8", "surrogateescape"))
            return
        digest.update(f"M{st.st_mode:o}:".encode())
        if st.st_size > _UNTRACKED_HASH_MAX_FILE_BYTES or st.st_size > budget[0]:
            digest.update(f"S{st.st_size}:{st.st_mtime_ns}:{st.st_ctime_ns}".encode())
            return
        budget[0] -= st.st_size
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        digest.update(b"?")


def _workspace_state(root: Path, skip_prefixes: tuple[str, ...] = ()) -> str:
    """Fingerprint of HEAD plus uncommitted state, so a moved workspace never resumes a stale job copy.

    Covers tracked changes (status, staged and unstaged binary diffs) and the content of untracked,
    non-ignored files: ``git status`` only names untracked files, so editing one would otherwise leave
    the fingerprint unchanged. Plugin-owned directories are excluded everywhere, so a receipt write or a
    scratch directory inside the workspace never changes it. A workspace that is not a git checkout is
    fingerprinted by walking its files. Very large files and untracked files past the content-hash cap
    contribute full stat metadata (mode, size, mtime_ns, ctime_ns, symlink target) instead of bytes.
    """
    skip = _STATE_SKIP_PREFIXES + tuple(skip_prefixes)
    budget = [_UNTRACKED_HASH_TOTAL_BYTES]
    if _git_out(root, "rev-parse", "--is-inside-work-tree").returncode != 0:
        digest = hashlib.sha256(b"nogit\0")
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            rel_dir = "" if rel_dir == "." else rel_dir + "/"
            dirnames[:] = sorted(d for d in dirnames if not (rel_dir + d + "/").startswith(skip)
                                 and d not in ("node_modules", "__pycache__", ".venv"))
            for name in sorted(filenames):
                rel = rel_dir + name
                if rel.startswith(skip):
                    continue
                count += 1
                digest.update(b"\0F\0" + rel.encode("utf-8", "surrogateescape") + b"\0")
                if count > _UNTRACKED_HASH_MAX_FILES:
                    # Past the content cap: full metadata so edits/renames/chmod anywhere change the fingerprint.
                    _hash_meta_into(digest, Path(dirpath) / name)
                    continue
                _hash_file_into(digest, Path(dirpath) / name, budget)
        return digest.hexdigest()
    head_res = _git_out(root, "rev-parse", "--verify", "-q", "HEAD")
    head = head_res.stdout if head_res.returncode == 0 else b"unborn"
    pathspec = ["--", ".", *(f":(exclude,literal){p.rstrip('/')}" for p in skip if p != ".git/")]
    status = _git_out(root, "status", "--porcelain=v1", "-z", *pathspec).stdout
    if head_res.returncode == 0:
        diff = _git_out(root, "diff", "HEAD", "--binary", *pathspec).stdout
    else:
        # No commit yet: staged content plus unstaged edits on top of it.
        diff = (_git_out(root, "diff", "--cached", "--binary", *pathspec).stdout + b"\0"
                + _git_out(root, "diff", "--binary", *pathspec).stdout)
    digest = hashlib.sha256(head + b"\0" + status + b"\0" + diff)
    listed = _git_out(root, "ls-files", "--others", "--exclude-standard", "-z", *pathspec)
    untracked = sorted(
        p for p in listed.stdout.decode("utf-8", "surrogateescape").split("\0")
        if p and not p.startswith(skip)
    ) if listed.returncode == 0 else []
    for rel in untracked[:_UNTRACKED_HASH_MAX_FILES]:
        digest.update(b"\0U\0" + rel.encode("utf-8", "surrogateescape") + b"\0")
        _hash_file_into(digest, root / rel, budget)
    for rel in untracked[_UNTRACKED_HASH_MAX_FILES:]:
        digest.update(b"\0U\0" + rel.encode("utf-8", "surrogateescape") + b"\0")
        _hash_meta_into(digest, root / rel)
    return digest.hexdigest()


def _restore_applied_answer(wt: Path, answer: AppliedAnswer, target: Path) -> None:
    """Restore ``target`` to ``answer.previous_bytes`` and ``answer.previous_mode`` safely."""
    _prepare_parent_within(
        wt,
        target.parent,
        create=(answer.previous_bytes is not None),
        unlink_planted_links=True,
    )
    if target.is_symlink():
        target.unlink(missing_ok=True)
    elif target.is_dir():
        _rmtree_writable(target)  # a broken formatter or check replaced the file with a directory
    elif target.exists() and answer.previous_bytes is None:
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        target.unlink(missing_ok=True)
    if answer.previous_bytes is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                os.chmod(target, (answer.previous_mode or 0o644) | 0o200)
            except OSError:
                pass
        target.write_bytes(answer.previous_bytes)
        if answer.previous_mode is not None:
            try:
                os.chmod(target, answer.previous_mode)
            except OSError:
                pass


def _is_targeted_check(cmd: str, rel_path: str) -> bool:
    """Return True when ``cmd`` appears specific to ``rel_path`` rather than a repo-wide shared check."""
    norm = rel_path.replace("\\", "/")
    if norm and norm in cmd:
        return True
    p = Path(norm)
    if p.name and p.name in cmd:
        return True
    if len(p.stem) >= 3 and p.stem in cmd:
        return True
    return False


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
        summaries_fn = getattr(self.store, "list_receipt_summaries", None)
        entries = (
            summaries_fn(workspace_root=self.root)
            if callable(summaries_fn)
            else self.store.list_receipts(workspace_root=self.root)
        )
        for rec in reversed(entries):
            if (rec.get("kind") in RESUMABLE_KINDS and rec.get("plan_fingerprint") == plan.raw_fingerprint
                    and rec.get("workspace_state") == ws_state and not rec.get("applied")
                    and rec.get("status") != "error"
                    and Path(str(rec.get("worktree_dir") or "")).is_dir()):
                if callable(summaries_fn) and rec.get("job_id"):
                    return self.store.read_receipt(str(rec["job_id"]))
                return rec
        return None

    def _open_job(self, plan: BatchPlan, resume_job_id: str | None, ws_state: str):
        if resume_job_id:
            rec = self.store.read_receipt(resume_job_id)
            if rec.get("kind") not in RESUMABLE_KINDS or rec.get("applied"):
                raise ValueError(f"job {resume_job_id!r} is not an unapplied batch job")
            # Same compatibility rules as an automatic resume: a job copy made from another plan or an
            # older workspace would mix stale files into this run's checks and diff.
            if rec.get("plan_fingerprint") != plan.raw_fingerprint:
                raise ValueError(
                    f"job {resume_job_id!r} was started from a different plan; re-send that exact plan to "
                    "resume it, or omit resume_job_id to start a new job for this plan"
                )
            if rec.get("workspace_state") != ws_state:
                raise ValueError(
                    f"the workspace changed since job {resume_job_id!r} made its job copy (commit, tracked or "
                    "untracked files); omit resume_job_id to start a new job from the current workspace"
                )
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

    def _make_verify(
        self,
        task,
        wt,
        plan,
        settings,
        log_dir,
        apply_lock,
        content_gate=None,
        side_effects=None,
        *,
        checks_override: list[str] | tuple[str, ...] | None = None,
        stage_deferred: dict[str, AppliedAnswer] | None = None,
        skip_formatter: bool = False,
    ):
        def verify(answer: AppliedAnswer) -> tuple[bool, str]:
            with apply_lock:
                link = symlink_component(wt, task.path)
                if link is not None:
                    return False, f"{task.path!r} goes through a symlink ({link!r}) in the job copy; not writing it"
                target = wt / task.path
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and not target.is_symlink() and not target.is_dir():
                    try:
                        os.chmod(target, (answer.previous_mode or 0o644) | 0o200)
                    except OSError:
                        pass
                target.write_bytes(answer.new_bytes)
                if answer.previous_mode is not None:
                    try:
                        os.chmod(target, answer.previous_mode)
                    except OSError:
                        pass
                if content_gate is not None:
                    problem = content_gate(task, answer.new_bytes, wt)
                    if problem:
                        _restore_applied_answer(wt, answer, target)
                        return False, problem
                raw_checks = (
                    list(checks_override)
                    if checks_override is not None
                    else (list(task.checks) + ([task.repro_test] if task.repro_test else []))
                )
                check_cmds = list(dict.fromkeys(c for c in raw_checks if c and c.strip()))
                # Consolidate formatter + checks into one snapshot session: only the formatter step may
                # rewrite task.path; every subsequent check runs with allowed_mutations=() so a test
                # rewriting task.path is flagged and restored.
                if plan.formatter and not skip_formatter:
                    fmt_cmd = plan.formatter.replace("{path}", shlex.quote(task.path))
                    suite_cmds = [fmt_cmd, *check_cmds]
                    per_cmd_allowed: list[tuple[str, ...]] | None = [(task.path,), *[() for _ in check_cmds]]
                else:
                    suite_cmds = check_cmds
                    per_cmd_allowed = None
                results: list[CommandResult] = (
                    run_commands(
                        suite_cmds,
                        wt,
                        timeout=settings.check_timeout,
                        log_dir=log_dir,
                        allowed_mutations=(),
                        per_command_allowed_mutations=per_cmd_allowed,
                        change_set=(task.path,),
                    )
                    if suite_cmds
                    else []
                )
                restored = sorted({p for r in results for p in r.side_effects_restored})
                if restored and side_effects is not None:
                    seen = side_effects.setdefault(task.task_id, [])
                    seen.extend(p for p in restored if p not in seen)
                if all(r.passed for r in results):
                    if stage_deferred is not None:
                        # Stage formatted bytes in memory and restore the job copy to baseline so
                        # concurrent non-leaf or immediate checks never see unverified leaf edits.
                        staged_bytes = target.read_bytes()
                        staged_mode = (
                            os.lstat(target).st_mode & 0o777
                            if answer.previous_mode is not None
                            else None
                        )
                        stage_deferred[task.task_id] = AppliedAnswer(
                            task.path, staged_bytes, answer.previous_bytes, staged_mode
                        )
                        _restore_applied_answer(wt, answer, target)
                    return True, ""
                _restore_applied_answer(wt, answer, target)
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
        enforce_daily_budget(
            self.store,
            workspace_root=self.root,
            auth_mode=str(auth.get("auth_mode") or ""),
        )
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

    def _state_skip_prefixes(self) -> tuple[str, ...]:
        """Plugin data and job-copy directories that live inside this workspace, as path prefixes."""
        prefixes: list[str] = []
        for candidate in (getattr(self.store, "data_dir", None), getattr(self.store, "worktrees_dir", None)):
            if candidate is None:
                continue
            try:
                rel = Path(candidate).resolve().relative_to(self.root)
            except (OSError, ValueError):
                continue
            if rel.parts:
                prefixes.append(rel.as_posix().rstrip("/") + "/")
        return tuple(prefixes)

    def _run_locked(self, plan, settings, auth, *, allowed_paths, extra_acceptance, resume_job_id, edits_only,
                    kind, extra_receipt, content_gate, shared_note) -> dict[str, Any]:
        ws_state = _workspace_state(self.root, self._state_skip_prefixes())
        job_id, wt, baseline_sha, baseline_hashes, prior = self._open_job(plan, resume_job_id, ws_state)
        started = round(time.time(), 3)
        ledger = RunLedger()
        try:
            return self._execute(
                plan, settings, auth, job_id=job_id, wt=wt, baseline_sha=baseline_sha, baseline_hashes=baseline_hashes,
                prior=prior, started=started, ws_state=ws_state, allowed_paths=allowed_paths,
                extra_acceptance=extra_acceptance, edits_only=edits_only, kind=kind, extra_receipt=extra_receipt,
                content_gate=content_gate, shared_note=shared_note, ledger=ledger,
            )
        except Exception as exc:  # noqa: BLE001 - never leave a job copy without a receipt
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
            # Keep what was already paid for: the resumed job's earlier spend plus every call this run made.
            prior_cost = (prior or {}).get("cost") or {}
            total_cents = _finite_cents(prior_cost.get("total_cost_cents")) + ledger.cents
            usage_total = {k: int(v) for k, v in ((prior or {}).get("usage") or {}).items() if isinstance(v, (int, float))}
            for key, val in ledger.usage.items():
                usage_total[key] = int(usage_total.get(key, 0)) + int(val)
            priced = ledger.pricing_known and bool(prior_cost.get("pricing_known", True))
            prior_oc_sid = (prior or {}).get("opencode_session_id")
            receipt = {k: v for k, v in (prior or {}).items() if k not in ("receipt_hmac", "cost", "updated_at_epoch")}
            receipt.update({
                "job_id": job_id, "kind": kind, "status": "error", "model": settings.model,
                "worktree_dir": str(wt), "baseline_commit_sha": baseline_sha, "baseline_file_hashes": baseline_hashes,
                "started_at_epoch": float((prior or {}).get("started_at_epoch") or started),
                "finished_at_epoch": round(time.time(), 3), "files_changed": [],
                "applied": False, "in_place": False, "plan_fingerprint": plan.raw_fingerprint, "workspace_state": ws_state,
                "prompts": [f"{t.task_id}: {t.intent}" for t in plan.tasks], "error": error, "usage": usage_total,
                "precomputed_cost": {"total_cost_cents": round(total_cents, 4), "total_cost_usd": round(total_cents / 100.0, 6),
                                     "pricing_known": priced},
                "cost_by_utc_day": ledger.merged_days(prior),
            })
            if extra_receipt:
                receipt.update(extra_receipt)
            if isinstance(prior_oc_sid, str) and prior_oc_sid.strip():
                receipt["opencode_session_id"] = prior_oc_sid.strip()
            raw_err_sid = receipt.get("opencode_session_id")
            persisted_err_sid = (
                raw_err_sid.strip()
                if isinstance(raw_err_sid, str) and raw_err_sid.strip()
                else None
            )
            self.store.write_receipt(job_id, receipt, workspace_root=str(self.root), api_key=auth.get("api_key"))
            err_out: dict[str, Any] = {
                "job_id": job_id, "status": "error", "error": error, "is_error": True,
                "totals": {"cost_cents": round(ledger.cents, 4), "job_cost_cents": round(total_cents, 4)},
            }
            if persisted_err_sid:
                err_out["opencode_session_id"] = persisted_err_sid
            return err_out

    def _execute(self, plan, settings, auth, *, job_id, wt, baseline_sha, baseline_hashes, prior, started, ws_state,
                 allowed_paths, extra_acceptance, edits_only, kind, extra_receipt, content_gate, shared_note,
                 ledger: RunLedger | None = None):
        ledger = ledger or RunLedger()
        journal = BatchJournal(self.store.data_dir, job_id)
        log_dir = self.store.data_dir / "file_batches" / f"{job_id}.logs"
        records = {t.task_id: TaskRecord(t) for t in plan.tasks}
        by_id = {t.task_id: t for t in plan.tasks}
        controls = BatchControls(
            settings,
            self.stop,
            prior=prior,
            store=self.store,
            job_id=job_id,
            workspace_root=self.root,
        )
        apply_lock = threading.Lock()
        check_side_effects: dict[str, list[str]] = {}
        auth_error: list[str] = []
        lock = threading.Lock()
        settled = {"n": 0}
        landed_paths = journal.landed_paths()

        # Partition leaf-task checks into immediate targeted checks vs repeated shared checks
        # so identical repo-wide suites run once per wave instead of once per task.
        depended_on_ids = {dep_id for t in plan.tasks for dep_id in t.depends_on}
        referenced_ctx_paths = {ctx_p for t in plan.tasks for ctx_p in t.context_files} | set(plan.shared_context)
        path_counts: dict[str, int] = {}
        for t in plan.tasks:
            path_counts[t.path] = path_counts.get(t.path, 0) + 1
        leaf_ids = {
            t.task_id
            for t in plan.tasks
            if t.task_id not in depended_on_ids
            and t.path not in referenced_ctx_paths
            and path_counts.get(t.path, 0) == 1
        }
        shared_cmd_counts: dict[str, int] = {}
        for t in plan.tasks:
            if t.task_id not in leaf_ids:
                continue
            for c in dict.fromkeys(x for x in t.checks if x and x.strip()):
                if not _is_targeted_check(c, t.path):
                    shared_cmd_counts[c] = shared_cmd_counts.get(c, 0) + 1
        shared_pool = {c for c, n in shared_cmd_counts.items() if n >= 2}
        immediate_checks: dict[str, list[str]] = {}
        deferred_checks: dict[str, list[str]] = {}
        for t in plan.tasks:
            if t.task_id in leaf_ids and shared_pool:
                imm: list[str] = []
                dfn: list[str] = []
                for c in dict.fromkeys(x for x in t.checks if x and x.strip()):
                    if c in shared_pool and not _is_targeted_check(c, t.path):
                        dfn.append(c)
                    else:
                        imm.append(c)
                if t.repro_test and t.repro_test not in imm:
                    imm.append(t.repro_test)
                if dfn:
                    immediate_checks[t.task_id] = imm
                    deferred_checks[t.task_id] = dfn

        staged_deferred: dict[str, AppliedAnswer] = {}
        landed_rungs: dict[str, int] = {}

        use_repo_map = os.environ.get("CC_AGY_REPO_MAP", "1").strip().lower() not in ("0", "false", "no")
        # Map the live workspace (stable path, so the cache hits across jobs), not the per-job copy.
        repo_map_text = build_repo_map(self.root, cache_dir=self.store.data_dir) if use_repo_map else ""
        repo_map_block = f"## Repository Map (Cached)\n{repo_map_text}" if repo_map_text else ""
        shared_text = "\n\n".join(x for x in (repo_map_block, shared_note.strip(), self._shared_text(plan, wt)) if x)
        writer_auth = dict(auth)
        writer_auth["_workspace_dir"] = wt
        cache_name = None
        if settings.writer == "genai" and shared_text and len(plan.tasks) > 1 and self.cache_factory:
            enforce_daily_budget(
            self.store,
            workspace_root=self.root,
            auth_mode=str(auth.get("auth_mode") or ""),
        )
            cache_name = self.cache_factory(auth, settings.model, WRITER_SYSTEM_PROMPT, shared_text)
            writer_auth["_shared_cache"] = cache_name
        use_claude = bool(settings.claude_writer and _claude_allowed_for_auth(auth))
        writers = _LadderWriters({
            settings.writer: self.writer_factory(settings.writer, settings, writer_auth),
            "claude": self.writer_factory("claude", settings, auth) if use_claude else None,
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

        def make_after_call(rec: TaskRecord):
            spent_lock = threading.Lock()

            def _after_call(writer_name: str, reply: Any | None) -> None:
                # Record spend per call, so the budget sees it at once and a later exception in the
                # ladder cannot drop what this task already paid for.
                controls.after_call(writer_name, reply)
                if reply is None:
                    return
                ledger.record(reply)
                with spent_lock:
                    rec.cost_cents += _finite_cents(reply.cost_cents)
                    rec.pricing_known = rec.pricing_known and bool(reply.pricing_known)
                    for key, val in (reply.usage or {}).items():
                        if isinstance(val, (int, float)):
                            rec.usage[key] = int(rec.usage.get(key, 0)) + int(val)

            return _after_call

        def run_one(task: FileTask) -> str:
            rec = records[task.task_id]
            after_call = make_after_call(rec)
            has_deferred = task.task_id in deferred_checks

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
                    verify=self._make_verify(
                        task,
                        wt,
                        plan,
                        settings,
                        log_dir,
                        apply_lock,
                        content_gate,
                        side_effects=check_side_effects,
                        checks_override=immediate_checks.get(task.task_id) if has_deferred else None,
                        stage_deferred=staged_deferred if has_deferred else None,
                    ),
                    build_request=build_request, root=wt, edits_only=edits_only,
                    allow_existing=task.path in landed_paths,
                    before_call=controls.before_call, on_transient=controls.on_transient,
                    after_call=after_call, should_stop=self.stop.is_set,
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
            rec.history = outcome.history
            rec.state = outcome.state
            if outcome.state == "held" and controls.stop_reason:
                rec.detail = controls.stop_reason
            if outcome.state == "landed":
                landed_rungs[task.task_id] = outcome.rung_index
                if not has_deferred:
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
        def _finalize_staged_deferred(*, allow_writer_retry: bool) -> None:
            pending_deferred = [
                t
                for t in plan.tasks
                if records[t.task_id].state == "landed"
                and deferred_checks.get(t.task_id)
                and t.task_id in staged_deferred
            ]
            consolidated_cmds = list(
                dict.fromkeys(c for t in pending_deferred for c in deferred_checks[t.task_id])
            )
            if not pending_deferred or not consolidated_cmds:
                return
            with apply_lock:
                for t in pending_deferred:
                    st_ans = staged_deferred[t.task_id]
                    tgt = wt / t.path
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    if tgt.exists() and not tgt.is_symlink() and not tgt.is_dir():
                        try:
                            os.chmod(tgt, (st_ans.previous_mode or 0o644) | 0o200)
                        except OSError:
                            pass
                    tgt.write_bytes(st_ans.new_bytes)
                    if st_ans.previous_mode is not None:
                        try:
                            os.chmod(tgt, st_ans.previous_mode)
                        except OSError:
                            pass
                shared_results = run_commands(
                    consolidated_cmds,
                    wt,
                    timeout=settings.check_timeout,
                    stop_on_failure=False,
                    log_dir=log_dir,
                    allowed_mutations=(),
                    change_set=tuple(t.path for t in pending_deferred),
                )
            shared_restored = sorted({p for r in shared_results for p in r.side_effects_restored})
            if shared_restored:
                seen_se = check_side_effects.setdefault("shared_checks", [])
                seen_se.extend(p for p in shared_restored if p not in seen_se)
            if all(r.passed for r in shared_results):
                for t in pending_deferred:
                    journal.record_landed(t, wt)
                    staged_deferred.pop(t.task_id, None)
                return

            # Roll back all deferred leaf candidates to baseline, then verify each staged
            # candidate against its own deferred checks so innocent leaves land with 0 extra
            # calls while only culprit leaves escalate up the writer ladder (or become held).
            with apply_lock:
                for t in reversed(pending_deferred):
                    _restore_applied_answer(wt, staged_deferred[t.task_id], wt / t.path)
            for t in pending_deferred:
                rec = records[t.task_id]
                ans = staged_deferred.pop(t.task_id, None)
                deferred_only_verify = self._make_verify(
                    t,
                    wt,
                    plan,
                    settings,
                    log_dir,
                    apply_lock,
                    content_gate,
                    side_effects=check_side_effects,
                    checks_override=deferred_checks[t.task_id],
                    skip_formatter=True,
                )
                ok, tail = deferred_only_verify(ans) if ans is not None else (False, "missing candidate")
                if ok:
                    journal.record_landed(t, wt)
                    continue
                if rec.history and rec.history[-1].get("result") == "landed":
                    rec.history[-1] = {
                        **rec.history[-1],
                        "result": "checks_failed",
                        "detail": tail[-300:],
                    }
                if not allow_writer_retry or auth_error or self.stop.is_set():
                    rec.state = "held"
                    stop_note = auth_error[0] if auth_error else (controls.stop_reason or "batch stopped before retry")
                    rec.detail = f"{tail[-200:]} ({stop_note})" if tail else stop_note
                    continue
                full_verify = self._make_verify(
                    t,
                    wt,
                    plan,
                    settings,
                    log_dir,
                    apply_lock,
                    content_gate,
                    side_effects=check_side_effects,
                )
                try:
                    retry_out: LadderOutcome = run_writer_ladder(
                        t,
                        routed=settings.writer,
                        writers=writers,
                        verify=full_verify,
                        build_request=build_request,
                        root=wt,
                        edits_only=edits_only,
                        allow_existing=t.path in landed_paths,
                        before_call=controls.before_call,
                        on_transient=controls.on_transient,
                        after_call=make_after_call(rec),
                        should_stop=self.stop.is_set,
                        start_rung=landed_rungs.get(t.task_id, 0) + 1,
                        initial_failure_tail=tail,
                    )
                except WriterAuthError as exc:
                    with lock:
                        auth_error.append(str(exc))
                    self.stop.set()
                    rec.state, rec.detail = "failed", str(exc)
                    rec.history.append({"writer": "", "result": "auth_error", "detail": str(exc)[-300:]})
                    continue
                except Exception as exc:  # noqa: BLE001
                    rec.state = "failed"
                    rec.detail = f"{type(exc).__name__}: {exc}"[-DIGEST_TAIL_CHARS:]
                    rec.history.append({"writer": "", "result": "writer_error", "detail": rec.detail[-300:]})
                    continue
                rec.attempts += retry_out.attempts
                rec.history.extend(retry_out.history)
                if retry_out.writer_used:
                    rec.writer = retry_out.writer_used
                rec.state = retry_out.state
                rec.detail = retry_out.detail or tail
                if retry_out.state == "held" and controls.stop_reason:
                    rec.detail = controls.stop_reason
                if retry_out.state == "landed":
                    journal.record_landed(t, wt)

        try:
            try:
                try:
                    # With a cost cap (per-batch or daily), the first call runs alone so the per-call
                    # reservation has a real price to work with before parallel calls start.
                    has_daily_cap = resolve_max_daily_cents() is not None
                    run_task_graph(
                        list(plan.tasks),
                        run_one,
                        max_parallel=settings.max_parallel,
                        on_settle=on_settle,
                        warm_first=bool(cache_name) or settings.claude_writer or settings.max_cost_cents is not None or has_daily_cap,
                        stop=self.stop,
                    )
                    _finalize_staged_deferred(allow_writer_retry=not auth_error and not self.stop.is_set())
                finally:
                    if staged_deferred:
                        try:
                            _finalize_staged_deferred(allow_writer_retry=False)
                        except Exception:  # noqa: BLE001
                            pass
                        if staged_deferred:
                            with apply_lock:
                                for tid, st_ans in list(staged_deferred.items()):
                                    t_obj = by_id.get(tid)
                                    if t_obj is not None:
                                        _restore_applied_answer(wt, st_ans, wt / t_obj.path)
                                    rec = records.get(tid)
                                    if rec is not None and rec.state == "landed":
                                        rec.state = "held"
                                        if not rec.detail:
                                            rec.detail = "staged candidate could not run deferred verification"
                                    staged_deferred.pop(tid, None)
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
            extra_receipt=extra_receipt, stop_reason=controls.stop_reason, ledger=ledger,
            check_side_effects=check_side_effects, claude_calls=controls.claude_calls,
        )

    def _finish(self, *, job_id, wt, baseline_sha, baseline_hashes, plan, settings, auth, records, started, prior,
                ws_state, allowed_paths, acceptance_cmds, acceptance_results, acceptance_passed, auth_error, kind,
                extra_receipt, stop_reason, ledger: RunLedger | None = None,
                check_side_effects: dict[str, list[str]] | None = None, claude_calls: int = 0):
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
        strength = {"unverified": 0, "syntax_only": 1, "behavioral": 2}
        acc_verif = (
            classify_commands_for_batch(acceptance_cmds)
            if (acceptance_cmds and acceptance_passed is True)
            else "unverified"
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
            if task_done:
                task_verif = classify_commands_for_batch(task_cmds)
                verif = acc_verif if strength[acc_verif] >= strength[task_verif] else task_verif
            else:
                verif = "unverified"
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
        # The weakest evidence across finished files: "ok" alone must never read as "verified".
        done_rows = [r for r in task_rows if r["state"] in DONE_STATES]
        strength = {"unverified": 0, "syntax_only": 1, "behavioral": 2}
        verification = min((r["verification"] for r in done_rows), key=strength.__getitem__) if done_rows else None
        unverified_ids = [r["task_id"] for r in done_rows if r["verification"] == "unverified"]
        # Tracked files that checks modified and run_commands put back: never in the diff, but reported.
        side_effects = {tid: list(paths) for tid, paths in (check_side_effects or {}).items() if paths}
        acc_restored = sorted({p for r in acceptance_results for p in (r.get("side_effects_restored") or [])})
        if acc_restored:
            side_effects["acceptance"] = acc_restored
        for row in task_rows:
            if row["task_id"] in side_effects:
                row["check_side_effects"] = side_effects[row["task_id"]][:20]
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
            "cost_by_utc_day": (ledger or RunLedger()).merged_days(prior),
            "files_changed": files_changed, "provenance": provenance, "applied": False, "in_place": False,
            "acceptance": acceptance_cmds, "acceptance_results": acceptance_results, "acceptance_passed": acceptance_passed,
            "error": auth_error or (stop_reason or None), "plan_fingerprint": plan.raw_fingerprint,
            "workspace_state": ws_state, "tasks": task_rows, "resumed": prior is not None, "review_required": review,
            "verification": verification, "unverified_task_ids": unverified_ids, "check_side_effects": side_effects,
            "claude_calls": int(claude_calls),
            "writer_route": {"writer": settings.writer, "model": settings.model, "thinking_level": settings.thinking_level,
                             "read_budget": settings.read_budget,
                             "claude_writer": settings.claude_model if settings.claude_writer else None},
            "external_worktree_tamper": False, "tool_call_count": 0, "tool_calls": [],
        }
        prior_oc_sid = (prior or {}).get("opencode_session_id")
        receipt.update(extra_receipt)
        if isinstance(prior_oc_sid, str) and prior_oc_sid.strip():
            receipt["opencode_session_id"] = prior_oc_sid.strip()
        raw_persisted_sid = receipt.get("opencode_session_id")
        persisted_oc_sid = (
            raw_persisted_sid.strip()
            if isinstance(raw_persisted_sid, str) and raw_persisted_sid.strip()
            else None
        )
        self.store.write_receipt(job_id, receipt, workspace_root=str(self.root), api_key=auth.get("api_key"))
        return self._digest(job_id, status, task_rows, counts, run_cents, total_cents, usage_run, acceptance_passed,
                            auth_error or stop_reason, files_changed, scope_violations, review,
                            verification=verification, unverified_ids=unverified_ids, side_effects=side_effects,
                            opencode_session_id=persisted_oc_sid)

    @staticmethod
    def _digest(job_id, status, rows, counts, run_cents, total_cents, usage, acceptance_passed, error, files_changed,
                scope_violations, review, *, verification: str | None = None,
                unverified_ids: list[str] | tuple[str, ...] = (),
                side_effects: dict[str, list[str]] | None = None,
                opencode_session_id: str | None = None) -> dict[str, Any]:
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
        if opencode_session_id:
            digest["opencode_session_id"] = opencode_session_id
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
        if verification:
            digest["verification"] = verification
        if side_effects:
            digest["check_side_effects"] = {k: v[:5] for k, v in list(side_effects.items())[:10]}
            digest["side_effects_note"] = (
                "Files changed while checks ran (tracked files or files earlier tasks created); those edits "
                "were undone and are not in the diff. A check that rewrites source files does not count as passing."
            )
        if unverified_ids:
            digest["unverified_task_ids"] = list(unverified_ids)[:10]
            digest["warning"] = (
                f"{len(unverified_ids)} finished file(s) had no check and no passing acceptance command covered "
                "them: nothing verified these edits. Review the diff before applying."
            )
        if status in ("ok", "checkpoint"):
            if unverified_ids:
                digest["next"] = (
                    f"review get_diff(job_id={job_id!r}) (unverified files listed in unverified_task_ids), "
                    f"then delegate_apply(job_id={job_id!r})"
                )
            else:
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


def probe_writers(
    settings: BatchSettings,
    auth: dict[str, Any],
    *,
    writer_factory: WriterFactory | None = None,
    store: Any = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Send one tiny request through each configured writer and report ready / warn / fail with a fix."""
    from dataclasses import replace
    from server.writers import WriterConfigError, WriterTransportError

    if store is not None:
        enforce_daily_budget(
            store,
            workspace_root=workspace_root,
            auth_mode=str(auth.get("auth_mode") or ""),
        )
    factory = writer_factory or default_writer_factory
    probe_settings = replace(settings, read_budget=0)
    use_claude = bool(probe_settings.claude_writer and _claude_allowed_for_auth(auth))
    names = [probe_settings.writer] + (["claude"] if use_claude else [])
    rows: list[dict[str, Any]] = []
    budget_guard = getattr(store, "budget", None) if store is not None else None
    for name in names:
        row: dict[str, Any] = {"writer": name, "model": probe_settings.claude_model if name == "claude" else probe_settings.model}
        res_id: str | None = None
        probe_job_id = f"probe-{uuid.uuid4().hex[:8]}"
        try:
            writer = factory(name, probe_settings, auth)
            if writer is None:
                row.update(state="fail",
                           fix="Install Claude Code (`claude` on PATH) or set CC_AGY_CLAUDE_BIN, or pass claude_writer=false.")
            else:
                if budget_guard is not None:
                    res_id = budget_guard.reserve(
                        probe_job_id,
                        estimate_cents=0.0,
                        unpriced=resolve_max_daily_cents() is not None,
                        workspace_root=workspace_root,
                    )
                reply = writer(WriterRequest(task=PROBE_TASK, prompt='Reply with {"path": "probe.txt", "content": "ok"}',
                                             max_output_tokens=64))
                if budget_guard is not None:
                    budget_guard.settle(
                        res_id,
                        job_id=probe_job_id,
                        actual_cents=_finite_cents(reply.cost_cents),
                        workspace_root=workspace_root,
                        record_live=True,
                    )
                    res_id = None
                row.update(state="ready", cost_cents=round(reply.cost_cents, 4), answered=bool(reply.text.strip()))
        except BudgetExceededError:
            raise
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
        finally:
            if budget_guard is not None and res_id is not None:
                budget_guard.settle(res_id, job_id=probe_job_id, actual_cents=0.0, workspace_root=workspace_root, record_live=False)
        rows.append(row)
    failed = [r for r in rows if r["state"] == "fail"]
    return {"status": "fail" if failed else "ok", "auth_mode": auth.get("auth_mode"), "writers": rows,
            "is_error": bool(failed)}
