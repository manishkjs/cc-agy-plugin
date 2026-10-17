"""Isolated job manager for Antigravity SDK delegations."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import fcntl
import fnmatch
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable
import uuid

from server.agy_runner import (
    is_straitjacket_available,
    read_progress,
    resolve_thinking_level,
    run_antigravity_worker,
)
from server.receipts import (
    ReceiptStore,
    extract_diff_provenance,
    parse_diff_git_header,
    parse_unified_header_path,
    resolve_workspace_root,
    sha256_bytes,
    unquote_git_path,
    validate_job_id,
)
from server.regions import (
    DEFAULT_DIFF_PAGE_TOKENS,
    MAX_TOOL_RESPONSE_TOKENS,
    estimate_tokens,
    first_valid_env,
    paginate_text_by_tokens,
    resolve_api_key,
    resolve_auth_config,
    resolve_vertex_location,
    resolve_vertex_project,
    truncate_to_token_budget,
)


def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _strip_dot_slash(path_str: str) -> str:
    """Strip leading `./` prefixes without stripping `../`."""
    s = path_str.strip()
    while s.startswith("./"):
        s = s[2:]
    return s


def validate_allowed_paths(allowed_paths: Any) -> list[str]:
    """Validate allowed_paths parameter, rejecting non-lists, parent traversal, or empty/root wildcards."""
    if allowed_paths is None:
        return []
    if not isinstance(allowed_paths, list):
        raise ValueError(
            f"allowed_paths must be a list of relative path strings, got {type(allowed_paths).__name__}"
        )
    cleaned: list[str] = []
    for item in allowed_paths:
        if not isinstance(item, str):
            raise ValueError(f"allowed_paths entries must be strings, got {type(item).__name__}")
        stripped = item.strip()
        norm = _strip_dot_slash(stripped).rstrip("/")
        if (
            not norm
            or norm in (".", "..")
            or stripped.startswith("/")
            or ".." in Path(norm).parts
        ):
            raise ValueError(
                f"Invalid allowed_paths entry {item!r}: must be a non-empty workspace-relative path without '..'"
            )
        cleaned.append(norm)
    return cleaned


def _matches_allowed_paths(rel_path: str, allowed_paths: list[str] | str | None) -> bool:
    """Return True if rel_path is inside or matches any entry in allowed_paths."""
    if allowed_paths is None:
        return True
    if isinstance(allowed_paths, str):
        raw_list = [allowed_paths]
    elif isinstance(allowed_paths, list):
        raw_list = allowed_paths
    else:
        return False
    if not raw_list:
        return True
    try:
        validated = validate_allowed_paths(raw_list)
    except ValueError:
        return False
    if not validated:
        return True
    stripped = rel_path.strip()
    if stripped.startswith("/"):
        return False
    norm_path = _strip_dot_slash(stripped)
    if not norm_path or norm_path in (".", "..") or ".." in Path(norm_path).parts:
        return False
    for norm_pat in validated:
        if norm_pat == "*":
            return True
        if norm_path == norm_pat or norm_path.startswith(norm_pat + "/"):
            return True
        if fnmatch.fnmatch(norm_path, norm_pat):
            return True
    return False


def _matches_file_filter(rel_path: str, file_filter: str) -> bool:
    """Match a diff file path against file_filter without substring false positives (a.py vs data.py)."""
    norm_path = _strip_dot_slash(rel_path.strip())
    norm_filter = _strip_dot_slash(file_filter.strip()).rstrip("/")
    if not norm_filter:
        return True
    if norm_path == norm_filter or norm_path.startswith(norm_filter + "/"):
        return True
    if Path(norm_path).name == norm_filter:
        return True
    if fnmatch.fnmatch(norm_path, norm_filter):
        return True
    # Also check if norm_filter matches a path suffix at a directory boundary
    if norm_path.endswith("/" + norm_filter):
        return True
    return False


def capture_live_workspace_diff(
    workspace_root: Path,
    *,
    baseline_head_sha: str | None = None,
) -> str:
    """Capture the unified diff of workspace_root against baseline_head_sha (or HEAD) including untracked files.

    Uses a temporary GIT_INDEX_FILE copy so the host repository's real git index is
    never mutated.
    """
    git_dir_res = _run_git(["rev-parse", "--git-dir"], cwd=workspace_root, check=False)
    if git_dir_res.returncode != 0:
        return ""
    raw_git_dir = Path(git_dir_res.stdout.strip())
    git_dir = raw_git_dir if raw_git_dir.is_absolute() else (workspace_root / raw_git_dir).resolve()
    real_index = git_dir / "index"

    target_ref = "HEAD"
    if baseline_head_sha:
        verify_ref = _run_git(["rev-parse", "--verify", f"{baseline_head_sha}^{{commit}}"], cwd=workspace_root, check=False)
        if verify_ref.returncode == 0:
            target_ref = baseline_head_sha

    with tempfile.NamedTemporaryFile(prefix="cc-agy-idx-", delete=False) as tmp_idx:
        tmp_index_path = Path(tmp_idx.name)
    try:
        if real_index.exists():
            shutil.copy2(real_index, tmp_index_path)
        else:
            tmp_index_path.unlink(missing_ok=True)
        env = dict(os.environ)
        env["GIT_INDEX_FILE"] = str(tmp_index_path)
        subprocess.run(
            ["git", "-c", "core.quotePath=false", "add", "-N", "."],
            cwd=str(workspace_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        diff_res = subprocess.run(
            ["git", "-c", "core.quotePath=false", "diff", "--binary", "--no-renames", target_ref],
            cwd=str(workspace_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return diff_res.stdout if diff_res.returncode == 0 else ""
    finally:
        tmp_index_path.unlink(missing_ok=True)


def _snapshot_directory_tree(root: Path) -> dict[str, bytes]:
    """Capture a byte snapshot of files inside `root` (used to protect host .git/hooks and .git/config)."""
    snap: dict[str, bytes] = {}
    if not root.exists():
        return snap
    if root.is_file():
        try:
            snap[""] = root.read_bytes()
        except OSError:
            pass
        return snap
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            rel = str(p.relative_to(root))
            try:
                snap[rel] = p.read_bytes()
            except OSError:
                pass
    return snap


def _restore_directory_tree(root: Path, snap: dict[str, bytes]) -> bool:
    """Verify `root` matches `snap`; if tampered, restore `root` to `snap` and return True."""
    current = _snapshot_directory_tree(root)
    if current == snap:
        return False
    if root.is_file() or ("" in snap):
        if "" in snap:
            root.parent.mkdir(parents=True, exist_ok=True)
            root.write_bytes(snap[""])
        elif root.exists():
            root.unlink(missing_ok=True)
        return True

    if root.exists():
        for p in list(root.rglob("*")):
            if p.is_file() or p.is_symlink():
                rel = str(p.relative_to(root))
                if rel not in snap:
                    p.unlink(missing_ok=True)
    for rel, content in snap.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return True


_EPHEMERAL_CACHE_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".tox",
        ".nox",
        ".venv",
        ".gradle",
        ".next",
        ".nuxt",
        ".ctx-session-reads",
        ".cc-agy-data",
        ".worktrees",
    }
)

_EPHEMERAL_CACHE_SUFFIXES: tuple[str, ...] = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".egg-info",
    ".coverage",
    ".DS_Store",
)

_IGNORED_BUILD_OUTPUT_DIRS: frozenset[str] = frozenset(
    {
        "build",
        "dist",
        "target",
        "out",
        "coverage",
        "venv",
        ".cache",
        ".parcel-cache",
        ".turbo",
        ".svelte-kit",
        "htmlcov",
    }
)

_SAFE_NOTE_EXTENSIONS: tuple[str, ...] = (
    ".md",
    ".txt",
    ".rst",
    ".org",
    ".adoc",
    ".todo",
)


def _is_ephemeral_build_or_test_artifact(rel_path: str, root: Path | None = None) -> bool:
    """Return True if `rel_path` is a transient test/compiler cache artifact that should never be captured or applied."""
    norm = rel_path.strip("/")
    if not norm:
        return False
    parts = norm.split("/")
    for part in parts:
        if part in _EPHEMERAL_CACHE_DIRS or part.endswith(".egg-info"):
            return True
    name = parts[-1]
    if (
        name in (".coverage", ".DS_Store")
        or name.startswith(".coverage.")
        or name.endswith(_EPHEMERAL_CACHE_SUFFIXES)
    ):
        return True
    if root is not None:
        candidate = root / norm
        if candidate.is_file() and not candidate.is_symlink():
            try:
                if candidate.stat().st_size <= 16384:
                    with open(candidate, "rb") as fh:
                        head = fh.read(128)
                    if head.startswith(b"# TLS secrets log file, generated by OpenSSL"):
                        return True
            except OSError:
                pass
    return False


def _is_ignored_build_artifact(rel_path: str, root: Path | None = None) -> bool:
    """Return True if an already-.gitignored path is inside a build/output/cache directory."""
    if _is_ephemeral_build_or_test_artifact(rel_path, root=root):
        return True
    norm = rel_path.strip("/")
    if not norm:
        return False
    for part in norm.split("/"):
        if part in _IGNORED_BUILD_OUTPUT_DIRS:
            return True
    return False


DEFAULT_MAX_TOOL_CALLS = 100
DEFAULT_MAX_TOTAL_TOKENS = 5_000_000
DEFAULT_ACCEPTANCE_TIMEOUT = 300.0
MAX_ACCEPTANCE_COMMANDS = 5
MAX_PLAN_STEPS = 15
ACCEPTANCE_OUTPUT_TOKENS = 600
CHECKPOINT_STOP_REASONS = (
    "MAX_TOOL_CALLS_EXCEEDED",
    "MAX_TOTAL_TOKENS_EXCEEDED",
    "MAX_INPUT_TOKENS_EXCEEDED",
    "MAX_OUTPUT_TOKENS_EXCEEDED",
    "MAX_MODEL_CALLS_EXCEEDED",
)


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    if not raw or raw.startswith("${"):
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def resolve_budget(
    max_tool_calls: int | None, max_total_tokens: int | None
) -> tuple[int | None, int | None]:
    """Per-turn worker caps (defaults 100 tool calls / 5M tokens; env overrides; 0 disables a cap)."""
    calls = max_tool_calls if max_tool_calls is not None else _env_int("CC_AGY_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS)
    tokens = (
        max_total_tokens
        if max_total_tokens is not None
        else _env_int("CC_AGY_MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS)
    )
    if int(calls) < 0 or int(tokens) < 0:
        raise ValueError("max_tool_calls and max_total_tokens must be >= 0 (0 disables the cap)")
    return (int(calls) or None), (int(tokens) or None)


def validate_plan(plan: Any) -> list[str]:
    if plan is None:
        return []
    if not isinstance(plan, list) or not all(isinstance(s, str) for s in plan):
        raise ValueError("plan must be a list of step strings")
    steps = [s.strip() for s in plan if s.strip()]
    if len(steps) > MAX_PLAN_STEPS:
        raise ValueError(f"plan has {len(steps)} steps; keep it to at most {MAX_PLAN_STEPS}")
    return steps


def validate_acceptance(acceptance: Any, workdir: Path) -> list[str]:
    """Acceptance commands must be read-only checks (tests, linters); mutating commands are refused.

    The plugin itself runs them, so a mutating command would be an unguarded way to edit the repo.
    """
    if acceptance is None:
        return []
    if not isinstance(acceptance, list) or not all(isinstance(c, str) for c in acceptance):
        raise ValueError("acceptance must be a list of shell command strings")
    cmds = [c.strip() for c in acceptance if c.strip()]
    if len(cmds) > MAX_ACCEPTANCE_COMMANDS:
        raise ValueError(f"acceptance has {len(cmds)} commands; at most {MAX_ACCEPTANCE_COMMANDS} allowed")
    from scripts.pre_tool_guard import detect_bash_workspace_mutation

    for cmd in cmds:
        verdict = detect_bash_workspace_mutation(cmd, workdir, cwd=workdir)
        if verdict.get("blocked"):
            raise ValueError(
                f"acceptance command {cmd!r} is refused because it can modify the workspace: "
                f"{verdict.get('reason')}"
            )
    return cmds


def render_worker_prompt(
    brief: str,
    *,
    plan: list[str],
    acceptance: list[str],
    max_tool_calls: int | None,
    max_total_tokens: int | None,
    progress_channel: bool = True,
) -> str:
    """Wrap the lead's brief with its plan, acceptance checks, budget, and reporting rules."""
    parts = [brief.strip()]
    if plan:
        parts.append("## Plan from the lead\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1)))
    if acceptance:
        parts.append(
            "## Acceptance\nWhen you finish, the plugin runs these commands and reports the results to the lead. "
            "Make them pass:\n"
            + "\n".join(f"- `{c}`" for c in acceptance)
            + "\nIMPORTANT: Create or update all requested implementation and test files FIRST, and verify your "
            "new/changed test files directly before running any broad repository test suite. Never modify unrelated "
            "files to fix pre-existing test failures outside your task scope or allowed_paths."
        )
    budget_bits = []
    if max_tool_calls:
        budget_bits.append(f"{max_tool_calls} tool calls")
    if max_total_tokens:
        budget_bits.append(f"{max_total_tokens:,} total tokens")
    if budget_bits:
        parts.append(
            "## Budget\nThis turn stops automatically after " + " or ".join(budget_bits) + ". Work efficiently: "
            "write all required implementation and test files early so your changes are saved on disk before budget "
            "runs out, read only the code you need (targeted line ranges, not whole large files), run focused test files "
            "(e.g. `pytest tests/specific_test.py`) instead of the entire repository test suite on every edit, "
            "do not re-read files you have already seen, and remove any temporary scratch scripts before finishing."
        )
    if progress_channel:
        parts.append(
            "## Reporting\nUse the report_to_lead tool: kind=\"milestone\" after each plan step or when tests "
            "pass; kind=\"question\" if a requirement is ambiguous; kind=\"blocker\" if you cannot proceed "
            "(questions and blockers end your turn). Finish with a short summary that marks each plan step "
            "DONE or BLOCKED."
        )
    if (
        os.environ.get("CC_AGY_CONTAINMENT", "1").lower() not in ("0", "false", "no")
        and is_straitjacket_available()
    ):
        parts.append(
            "## Shell Commands & Evidence\n"
            "Commands execute with bounded output. If a test or build produces large output, it returns a concise "
            "digest and an address (e.g. `run:<id>#stdout`). Use `ctx_get(address, start_line, end_line)` or "
            "`ctx_search(address, pattern)` to retrieve specific omitted lines or tracebacks instead of rerunning commands."
        )
    return "\n\n".join(parts)


def _parse_test_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "errors": 0}
    for key, pat in (("passed", r"(\d+) passed"), ("failed", r"(\d+) failed"), ("errors", r"(\d+) errors?\b")):
        found = re.findall(pat, output)
        if found:
            counts[key] = int(found[-1])
    return counts


def _build_workdir_exec_env(cwd: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment dict that prepends cwd (and cwd/src) to PYTHONPATH and cwd/.venv/bin to PATH."""
    env = dict(os.environ if base_env is None else base_env)
    cwd_resolved = cwd.resolve()
    py_parts = [str(cwd_resolved)]
    src_dir = cwd_resolved / "src"
    if src_dir.is_dir():
        py_parts.append(str(src_dir))
    existing_py = (env.get("PYTHONPATH") or "").strip()
    if existing_py:
        py_parts.append(existing_py)
    env["PYTHONPATH"] = ":".join(py_parts)
    venv_dir = cwd_resolved / ".venv"
    venv_bin = venv_dir / "bin"
    if venv_bin.is_dir():
        existing_path = (env.get("PATH") or "").strip()
        env["PATH"] = f"{venv_bin}:{existing_path}" if existing_path else str(venv_bin)
        env["VIRTUAL_ENV"] = str(venv_dir)
    return env


def run_acceptance(cmds: list[str], cwd: Path, timeout: float | None = None) -> list[dict[str, Any]]:
    """Run acceptance checks with a hard timeout (whole process group is killed), then delete any new
    untracked files the checks themselves created (e.g. caches or TLS key logs), so they never land in
    the diff or block later turns."""
    limit = float(timeout or _env_int("CC_AGY_ACCEPTANCE_TIMEOUT", int(DEFAULT_ACCEPTANCE_TIMEOUT)))
    before = set(_snapshot_untracked_hashes(cwd))
    exec_env = _build_workdir_exec_env(cwd)
    results: list[dict[str, Any]] = []
    for cmd in cmds:
        started = time.time()
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            cwd=str(cwd),
            env=exec_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        timed_out = False
        try:
            out, _ = proc.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            out, _ = proc.communicate()
        tail, _, _ = truncate_to_token_budget((out or "")[-20000:], ACCEPTANCE_OUTPUT_TOKENS)
        results.append(
            {
                "command": cmd,
                "exit_code": None if timed_out else proc.returncode,
                "timed_out": timed_out,
                "passed": (not timed_out) and proc.returncode == 0,
                "duration_seconds": round(time.time() - started, 1),
                "counts": _parse_test_counts(out or ""),
                "output_tail": tail,
            }
        )
    created = [rel for rel in _snapshot_untracked_hashes(cwd) if rel not in before]
    for rel in created:
        p = cwd / rel
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
    if created and results:
        results[-1]["cleaned_artifacts"] = created[:20]
    return results


def _hash_path(p: Path) -> str | None:
    """Return a content fingerprint for a workspace path (symlink target, file SHA-256, or None if absent)."""
    if p.is_symlink():
        return f"symlink:{os.readlink(p)}"
    if p.exists() and p.is_file():
        return sha256_bytes(p.read_bytes())
    return None


def _snapshot_untracked_hashes(root: Path) -> dict[str, str | None]:
    """Fingerprint every untracked, non-ignored file so in_place jobs can skip ones they did not touch."""
    res = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=str(root),
        capture_output=True,
        check=False,
    )
    snap: dict[str, str | None] = {}
    if res.returncode != 0:
        return snap
    for raw in res.stdout.split(b"\x00"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="surrogateescape")
        snap[rel] = _hash_path(root / rel)
    return snap


def _split_diff_by_file(diff_text: str) -> list[tuple[str, str]]:
    """Split a unified git diff into (path, chunk) pairs keyed by the `b/` path of each header."""
    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    current_path = ""
    in_hunk = False
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                chunks.append((current_path, "".join(current)))
            current = [line]
            hdr = parse_diff_git_header(line)
            current_path = (hdr[1] or hdr[0]) if hdr else ""
            in_hunk = False
        elif current:
            if line.startswith("@@ "):
                in_hunk = True
            elif not in_hunk and line.startswith("+++ "):
                plus_path = parse_unified_header_path(line, "+++")
                if plus_path and plus_path != "/dev/null":
                    current_path = plus_path
            elif not in_hunk and line.startswith("--- ") and not current_path:
                minus_path = parse_unified_header_path(line, "---")
                if minus_path and minus_path != "/dev/null":
                    current_path = minus_path
            current.append(line)
    if current:
        chunks.append((current_path, "".join(current)))
    return chunks


class JobManager:
    """Manages isolated worker clones, background execution, conflict-safe apply, and provenance."""

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        store: ReceiptStore | None = None,
        worker_backend: Callable[[dict[str, Any]], Any] | None = None,
        max_workers: int = 4,
    ) -> None:
        self.workspace_root = resolve_workspace_root(workspace_root)
        self.store = store or ReceiptStore(workspace_root=self.workspace_root)
        self.worker_backend = worker_backend
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cc-agy-job")
        self._lock = threading.Lock()
        self._apply_lock = threading.Lock()
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._cancelled: set[str] = set()
        self._raw_projects: dict[str, str] = {}
        self._raw_api_keys: dict[str, str] = {}
        self._in_place_job: str | None = None
        self._ensure_baseline_recorded()

    def _ensure_baseline_recorded(self) -> None:
        if self.store.is_baseline_tampered(self.workspace_root):
            return
        head_res = _run_git(["rev-parse", "HEAD"], cwd=self.workspace_root, check=False)
        head_sha = head_res.stdout.strip() if head_res.returncode == 0 else None
        existing = self.store.read_baseline_record(self.workspace_root)
        if existing is not None:
            receipts = self.store.list_receipts(workspace_root=self.workspace_root)
            for r in receipts:
                if r.get("kind") == "delegate" and not r.get("applied") and r.get("status") == "running":
                    return
            existing_sha = existing.get("baseline_head_sha")
            if head_sha and existing_sha and head_sha != existing_sha:
                # HEAD moved since the prior baseline was recorded. Verify existing_sha is an
                # ancestor of head_sha (normal human commit between sessions vs amend/reset).
                anc_res = _run_git(
                    ["merge-base", "--is-ancestor", str(existing_sha), str(head_sha)],
                    cwd=self.workspace_root,
                    check=False,
                )
                if anc_res.returncode != 0:
                    return
            else:
                # Same HEAD commit: refuse to overwrite if tracked files or untracked code files
                # have unverified modifications (prevents laundering host edits via MCP server restart).
                existing_prov = existing.get("provenance") or {}
                existing_prov_files = set(existing_prov.get("files_changed") or [])
                applied_files: set[str] = set()
                latest_applied_hash: dict[str, str | None] = {}
                for r in receipts:
                    if r.get("applied") and self.store.verify_receipt_signature(r):
                        applied_files.update(r.get("applied_files") or [])
                        for f_k, f_h in (r.get("applied_file_hashes") or {}).items():
                            latest_applied_hash[f_k] = f_h
                    for hist_prov in r.get("applied_history") or []:
                        if isinstance(hist_prov, dict):
                            applied_files.update(hist_prov.get("files_changed") or [])

                tracked_diff_res = _run_git(
                    ["diff", "--name-only", "HEAD"],
                    cwd=self.workspace_root,
                    check=False,
                )
                if tracked_diff_res.returncode == 0 and tracked_diff_res.stdout.strip():
                    for fpath in tracked_diff_res.stdout.splitlines():
                        clean_f = fpath.strip()
                        if not clean_f:
                            continue
                        if clean_f not in existing_prov_files and clean_f not in applied_files:
                            return
                        if clean_f in latest_applied_hash:
                            expected_h = latest_applied_hash[clean_f]
                            live_p = self.workspace_root / clean_f
                            actual_h = (
                                sha256_bytes(live_p.read_bytes())
                                if live_p.exists() and live_p.is_file() and not live_p.is_symlink()
                                else None
                            )
                            if actual_h != expected_h:
                                return

                untracked_res = _run_git(
                    ["ls-files", "--others", "--exclude-standard"],
                    cwd=self.workspace_root,
                    check=False,
                )
                if untracked_res.returncode == 0 and untracked_res.stdout.strip():
                    for line in untracked_res.stdout.splitlines():
                        rel_u = line.strip()
                        if (
                            not rel_u
                            or rel_u.startswith((".cc-agy-data/", ".worktrees/"))
                            or _is_ephemeral_build_or_test_artifact(rel_u, root=self.workspace_root)
                            or rel_u in existing_prov_files
                            or rel_u in applied_files
                        ):
                            continue
                        if not rel_u.lower().endswith(_SAFE_NOTE_EXTENSIONS):
                            return

        live_diff = capture_live_workspace_diff(self.workspace_root, baseline_head_sha=head_sha)
        prov = extract_diff_provenance(live_diff)
        self.store.record_baseline_dirty_state(
            self.workspace_root,
            prov,
            baseline_head_sha=head_sha,
            overwrite=True,
        )

    def _prepare_worktree(self, job_id: str) -> tuple[Path, str, dict[str, str | None]]:
        """Create a standalone isolated git repository copy for `job_id`.

        Uses `git ls-files -co --exclude-standard -z` so gitignored directories (like
        `node_modules/` or `.venv/`) are never copied into the worker tree, and computes
        baseline file hashes on demand for changed files rather than bloating the receipt JSON.
        """
        safe_id = validate_job_id(job_id)
        wt_dir = self.store.worktree_path(safe_id).resolve()
        if wt_dir.exists():
            shutil.rmtree(wt_dir, ignore_errors=True)
        wt_dir.mkdir(parents=True, exist_ok=True)

        ls_res = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=str(self.workspace_root),
            capture_output=True,
            check=False,
        )
        if ls_res.returncode == 0:
            rel_entries = [
                entry.decode("utf-8", errors="surrogateescape")
                for entry in ls_res.stdout.split(b"\x00")
                if entry
            ]
            for rel in rel_entries:
                if rel.startswith((".git/", ".cc-agy-data/", ".worktrees/", ".ctx-session-reads/", ".venv/", "venv/")) or rel in (".venv", "venv", ".ctx-session-reads"):
                    continue
                src_p = self.workspace_root / rel
                dst_p = wt_dir / rel
                if src_p.is_symlink():
                    dst_p.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(os.readlink(src_p), dst_p)
                elif src_p.is_file():
                    dst_p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_p, dst_p)
        else:
            shutil.rmtree(wt_dir, ignore_errors=True)
            shutil.copytree(
                self.workspace_root,
                wt_dir,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    ".git", ".cc-agy-data", ".worktrees", ".ctx-session-reads", "__pycache__", "node_modules", ".venv", "venv"
                ),
            )

        _run_git(["init", "-b", "main"], cwd=wt_dir, check=True)
        _run_git(["config", "user.email", "worker@cc-agy-plugin.local"], cwd=wt_dir, check=True)
        _run_git(["config", "user.name", "CC-AGY Worker"], cwd=wt_dir, check=True)
        _run_git(["add", "-A"], cwd=wt_dir, check=True)
        _run_git(["commit", "--allow-empty", "-m", "cc-agy-baseline"], cwd=wt_dir, check=True)
        sha_res = _run_git(["rev-parse", "HEAD"], cwd=wt_dir, check=True)
        baseline_sha = sha_res.stdout.strip()

        exclude_file = wt_dir / ".git" / "info" / "exclude"
        try:
            exclude_file.parent.mkdir(parents=True, exist_ok=True)
            existing_excl = exclude_file.read_text(encoding="utf-8", errors="replace") if exclude_file.exists() else ""
            extra_excl = "\n".join(
                [
                    ".venv",
                    ".venv/",
                    ".pytest_cache/",
                    "__pycache__/",
                    ".mypy_cache/",
                    ".ruff_cache/",
                    ".ctx-session-reads",
                    ".ctx-session-reads/",
                ]
            )
            exclude_file.write_text(
                (existing_excl.rstrip("\n") + "\n" + extra_excl + "\n").lstrip("\n"),
                encoding="utf-8",
            )
        except OSError:
            pass

        host_venv = self.workspace_root / ".venv"
        wt_venv = wt_dir / ".venv"
        if host_venv.exists() and not wt_venv.exists() and not wt_venv.is_symlink():
            try:
                os.symlink(str(host_venv.resolve()), str(wt_venv))
            except OSError:
                pass

        return wt_dir, baseline_sha, {}

    def _compute_changed_baseline_hashes(
        self,
        wt_dir: Path,
        baseline_sha: str,
        files_changed: list[str],
        prior_hashes: dict[str, str | None] | None = None,
    ) -> dict[str, str | None]:
        """Compute baseline SHA-256 hashes only for the files touched by the worker."""
        hashes: dict[str, str | None] = dict(prior_hashes or {})
        if not wt_dir.exists():
            return hashes
        for rel in files_changed:
            if rel in hashes:
                continue
            show_res = subprocess.run(
                ["git", "show", f"{baseline_sha}:{rel}"],
                cwd=str(wt_dir),
                capture_output=True,
                check=False,
            )
            if show_res.returncode == 0:
                hashes[rel] = sha256_bytes(show_res.stdout)
            else:
                hashes[rel] = None
        return hashes

    def _collect_worktree_diff(
        self,
        wt_dir: Path,
        baseline_sha: str,
        allowed_paths: list[str] | None = None,
    ) -> tuple[str, list[str], list[str]]:
        """Collect unified binary diff with `--no-renames` against `baseline_sha`.

        Also force-stages any new `.gitignore`d files written by the worker inside `wt_dir`
        (excluding transient build/test caches such as `__pycache__/` or `.pytest_cache/`).
        Returns (diff_text, files_changed, symlink_escapes).
        """
        if not wt_dir.exists():
            return "", [], []

        symlink_escapes: list[str] = []
        wt_resolved = wt_dir.resolve()
        for p in list(wt_dir.rglob("*")):
            if ".git" in p.parts:
                continue
            rel = str(p.relative_to(wt_dir))
            if _is_ephemeral_build_or_test_artifact(rel, root=wt_dir):
                continue
            if p.is_symlink():
                try:
                    link_target = (p.parent / os.readlink(p)).resolve()
                    if not link_target.is_relative_to(wt_resolved):
                        symlink_escapes.append(rel)
                except OSError:
                    symlink_escapes.append(rel)

        _run_git(["reset", "-q", baseline_sha], cwd=wt_dir, check=False)
        untracked_to_stage: list[str] = []
        others_res = _run_git(
            ["ls-files", "--others", "--exclude-standard"],
            cwd=wt_dir,
            check=False,
        )
        if others_res.returncode == 0 and others_res.stdout.strip():
            for line in others_res.stdout.splitlines():
                rel_u = line.strip()
                if not rel_u:
                    continue
                if _is_ephemeral_build_or_test_artifact(rel_u, root=wt_dir):
                    eph_p = wt_dir / rel_u
                    if eph_p.is_file() and not eph_p.is_symlink():
                        eph_p.unlink(missing_ok=True)
                    continue
                untracked_to_stage.append(rel_u)

        ign_res = _run_git(
            ["ls-files", "--others", "--ignored", "--exclude-standard"],
            cwd=wt_dir,
            check=False,
        )
        if ign_res.returncode == 0 and ign_res.stdout.strip():
            for line in ign_res.stdout.splitlines():
                rel_i = line.strip()
                if (
                    rel_i
                    and not _is_ignored_build_artifact(rel_i, root=wt_dir)
                    and rel_i not in untracked_to_stage
                ):
                    untracked_to_stage.append(rel_i)

        if untracked_to_stage:
            _run_git(["add", "-f", "-N", "--", *untracked_to_stage], cwd=wt_dir, check=False)

        diff_res = _run_git(
            ["diff", "--binary", "--no-renames", baseline_sha],
            cwd=wt_dir,
            check=False,
        )
        diff_text = diff_res.stdout if diff_res.returncode == 0 else ""

        name_res = _run_git(
            ["diff", "--name-only", "--no-renames", baseline_sha],
            cwd=wt_dir,
            check=False,
        )
        if untracked_to_stage:
            _run_git(["reset", "-q", baseline_sha], cwd=wt_dir, check=False)
        files = [
            line.strip()
            for line in (name_res.stdout.splitlines() if name_res.returncode == 0 else [])
            if line.strip()
        ]
        for esc in symlink_escapes:
            if esc not in files:
                files.append(esc)
        return diff_text, files, symlink_escapes

    def _cleanup_worktree(self, wt_dir: Path) -> None:
        if not wt_dir.exists():
            return
        # Never delete anything outside the plugin's own worktrees directory. In in_place mode
        # the "worktree" is the user's real repository, so this guard must never be removed.
        try:
            resolved = wt_dir.resolve()
            if not resolved.is_relative_to(self.store.worktrees_dir.resolve()):
                return
            if resolved == self.workspace_root.resolve():
                return
        except (OSError, ValueError):
            return
        shutil.rmtree(wt_dir, ignore_errors=True)

    def _snapshot_branches(self) -> tuple[str | None, list[str]]:
        """Capture (current symbolic branch name or None, list of local branch names)."""
        sym_res = _run_git(["symbolic-ref", "-q", "--short", "HEAD"], cwd=self.workspace_root, check=False)
        orig_branch = sym_res.stdout.strip() if sym_res.returncode == 0 and sym_res.stdout.strip() else None
        refs_res = _run_git(
            ["for-each-ref", "--format=%(refname:short)", "refs/heads/"],
            cwd=self.workspace_root,
            check=False,
        )
        branches = [b.strip() for b in refs_res.stdout.splitlines() if b.strip()] if refs_res.returncode == 0 else []
        return orig_branch, branches

    def _check_in_place_ready(self) -> tuple[str, dict[str, str | None]]:
        """Verify the host repo can be edited in place; return (HEAD sha, untracked file hashes)."""
        head_res = _run_git(["rev-parse", "--verify", "HEAD"], cwd=self.workspace_root, check=False)
        if head_res.returncode != 0 or not head_res.stdout.strip():
            raise ValueError("in_place mode needs a git repository with at least one commit.")
        dirty_res = _run_git(
            ["status", "--porcelain", "--untracked-files=no"],
            cwd=self.workspace_root,
            check=False,
        )
        dirty = [line for line in dirty_res.stdout.splitlines() if line.strip()]
        if dirty:
            raise ValueError(
                "in_place mode needs a clean working tree for tracked files, so git can undo the "
                "worker's edits. If they come from an earlier in_place job, send it a follow-up with "
                "delegate_continue instead. Otherwise commit or stash them, or use the default isolated copy: "
                f"{dirty[:10]}"
            )
        return head_res.stdout.strip(), _snapshot_untracked_hashes(self.workspace_root)

    def _acquire_in_place_slot(self, job_id: str) -> None:
        with self._lock:
            active = self._in_place_job
            if active is not None and active != job_id:
                active_fut = self._futures.get(active)
                if active_fut is None or not active_fut.done():
                    raise RuntimeError(
                        f"Another in_place job ({active}) is still running in this repository. "
                        f"Wait for it to finish or use the default isolated copy."
                    )
            self._in_place_job = job_id

    def _release_in_place_slot(self, job_id: str) -> None:
        with self._lock:
            if self._in_place_job == job_id:
                self._in_place_job = None

    def _finalize_in_place(
        self,
        baseline_sha: str,
        pre_untracked: dict[str, str | None],
        orig_branch: str | None = None,
        pre_branches: list[str] | None = None,
    ) -> tuple[str, list[str], list[str], list[str]]:
        """Collect the worker's in-place edits against baseline_sha.

        Undoes any commits or branch switches the worker made (keeping their changes in the
        working tree), restores the index, and skips untracked files that existed before the job
        and are unchanged. Returns (diff_text, files_changed, symlink_escapes, notes).
        """
        root = self.workspace_root
        notes: list[str] = []
        sym_res = _run_git(["symbolic-ref", "-q", "--short", "HEAD"], cwd=root, check=False)
        cur_branch = sym_res.stdout.strip() if sym_res.returncode == 0 and sym_res.stdout.strip() else None
        head_res = _run_git(["rev-parse", "HEAD"], cwd=root, check=False)
        head_now = head_res.stdout.strip() if head_res.returncode == 0 else ""

        if orig_branch:
            _run_git(["update-ref", f"refs/heads/{orig_branch}", baseline_sha], cwd=root, check=False)
            if cur_branch != orig_branch:
                _run_git(["symbolic-ref", "HEAD", f"refs/heads/{orig_branch}"], cwd=root, check=False)
                notes.append(
                    f"Worker switched branch to {cur_branch or 'detached HEAD'!r}; "
                    f"restored HEAD to branch {orig_branch!r} at {baseline_sha[:12]}."
                )
        elif cur_branch:
            _run_git(["symbolic-ref", "--delete", "HEAD"], cwd=root, check=False)
            _run_git(["update-ref", "--no-deref", "HEAD", baseline_sha], cwd=root, check=False)
            notes.append(
                f"Worker switched to branch {cur_branch!r}; restored detached HEAD at {baseline_sha[:12]}."
            )

        if pre_branches is not None:
            refs_res = _run_git(
                ["for-each-ref", "--format=%(refname:short)", "refs/heads/"],
                cwd=root,
                check=False,
            )
            now_branches = (
                [b.strip() for b in refs_res.stdout.splitlines() if b.strip()]
                if refs_res.returncode == 0
                else []
            )
            pre_set = set(pre_branches)
            for b in now_branches:
                if b not in pre_set and b != orig_branch:
                    b_rev = _run_git(["rev-parse", "--verify", f"refs/heads/{b}"], cwd=root, check=False)
                    b_sha = b_rev.stdout.strip() if b_rev.returncode == 0 else ""
                    if b_sha and b_sha != baseline_sha and b_sha != head_now:
                        _run_git(["checkout", b_sha, "--", "."], cwd=root, check=False)
                        notes.append(
                            f"Recovered committed changes from worker branch {b!r} ({b_sha[:12]}) "
                            f"into working tree before deleting branch."
                        )
                    _run_git(["branch", "-D", b], cwd=root, check=False)
                    notes.append(f"Deleted worker-created branch {b!r}.")

        if head_now and head_now != baseline_sha:
            notes.append(
                f"Worker moved HEAD to {head_now[:12]}; its commits were undone with "
                f"`git reset --mixed {baseline_sha[:12]}` and the changes kept in the working tree."
            )
        _run_git(["reset", "-q", baseline_sha], cwd=root, check=False)

        live_diff = capture_live_workspace_diff(root, baseline_head_sha=baseline_sha)
        kept_chunks: list[str] = []
        files: list[str] = []
        for rel, chunk in _split_diff_by_file(live_diff):
            if not rel or rel in files:
                continue
            if rel.startswith((".cc-agy-data/", ".worktrees/", ".ctx-session-reads/")):
                continue
            # Unlike the isolated copy, caches the worker writes here (e.g. __pycache__) really are
            # in the user's tree, so they are recorded rather than filtered (and never deleted).
            if rel in pre_untracked and _hash_path(root / rel) == pre_untracked[rel]:
                continue
            kept_chunks.append(chunk)
            files.append(rel)

        escapes: list[str] = []
        root_resolved = root.resolve()
        for rel in files:
            p = root / rel
            if p.is_symlink():
                try:
                    target = (p.parent / os.readlink(p)).resolve()
                    if not target.is_relative_to(root_resolved):
                        escapes.append(rel)
                except OSError:
                    escapes.append(rel)
        return "".join(kept_chunks), files, escapes, notes

    def _execute_job(
        self,
        *,
        job_id: str,
        worker_prompts: list[str],
        recorded_prompts: list[str],
        model: str,
        auth_mode: str = "vertex",
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        reroute_note: str | None,
        allowed_paths: list[str],
        timeout_seconds: float,
        enabled_tools: list[str] | None,
        disabled_tools: list[str] | None,
        enable_subagents: bool,
        wt_dir: Path,
        baseline_sha: str,
        baseline_file_hashes: dict[str, str | None],
        started_at: float,
        prior_usage: dict[str, Any] | None = None,
        prior_tool_calls: list[dict[str, Any]] | None = None,
        scrub_receipt: bool = False,
        in_place: bool = False,
        pre_untracked: dict[str, str | None] | None = None,
        thinking_level: str | None = None,
        plan: list[str] | None = None,
        acceptance: list[str] | None = None,
        max_tool_calls: int | None = None,
        max_total_tokens: int | None = None,
    ) -> dict[str, Any]:
        token_keys = (
            "prompt_token_count",
            "cached_content_token_count",
            "candidates_token_count",
            "thoughts_token_count",
            "total_token_count",
        )
        usage: dict[str, Any] = {
            k: int((prior_usage or {}).get(k, 0)) for k in token_keys
        }
        prior_requests = list((prior_usage or {}).get("requests") or []) if isinstance(prior_usage, dict) else []
        prior_max_req_prompt = int((prior_usage or {}).get("max_request_prompt_tokens") or 0) if isinstance(prior_usage, dict) else 0
        if prior_requests:
            usage["requests"] = prior_requests
        if prior_max_req_prompt > 0:
            usage["max_request_prompt_tokens"] = prior_max_req_prompt

        tool_calls: list[dict[str, Any]] = list(prior_tool_calls or [])
        thoughts: list[str] = []
        response_text = ""
        status = "error"
        error_msg: str | None = None
        diff_text = ""
        files_changed: list[str] = []
        scope_violations: list[str] = []
        in_place_notes: list[str] = []
        symlink_escapes: list[str] = []
        stop_reason: str | None = None
        acceptance_results: list[dict[str, Any]] = []
        progress_path = self.store.progress_path(job_id)
        progress_before = len(read_progress(progress_path, limit=10**6))

        host_git_dir = self.workspace_root / ".git"
        hooks_snap = _snapshot_directory_tree(host_git_dir / "hooks")
        config_snap = _snapshot_directory_tree(host_git_dir / "config")
        orig_branch, pre_branches = self._snapshot_branches() if in_place else (None, [])
        worker_out: dict[str, Any] = {}
        workspace_dir_accessed = False

        class _TrackedReq(dict[str, Any]):
            def __getitem__(self, key: str) -> Any:
                nonlocal workspace_dir_accessed
                if key == "workspace_dir":
                    workspace_dir_accessed = True
                return super().__getitem__(key)

            def get(self, key: str, default: Any = None) -> Any:
                nonlocal workspace_dir_accessed
                if key == "workspace_dir":
                    workspace_dir_accessed = True
                return super().get(key, default)

        raw_backend = self.worker_backend
        if raw_backend is None:
            workspace_dir_accessed = True
            tracked_backend = None
        else:
            def tracked_backend(req: dict[str, Any]) -> Any:
                return raw_backend(_TrackedReq(req))

        try:
            with self._lock:
                if job_id in self._cancelled:
                    status = "cancelled"
                    error_msg = "Job cancelled before execution."
                    raise RuntimeError(error_msg)

            worker_out = run_antigravity_worker(
                prompts=worker_prompts,
                workspace_dir=wt_dir,
                model=model,
                project=project,
                location=location,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                enabled_tools=enabled_tools,
                disabled_tools=disabled_tools,
                enable_subagents=enable_subagents,
                backend=tracked_backend,
                thinking_level=thinking_level,
                max_tool_calls=max_tool_calls,
                max_total_tokens=max_total_tokens,
                progress_path=progress_path,
            )

            response_text = str(worker_out.get("text") or "")
            new_usage = worker_out.get("usage") or {}
            for k in token_keys:
                usage[k] += int(new_usage.get(k) or 0)
            usage["total_token_count"] = max(
                usage["total_token_count"],
                usage["prompt_token_count"]
                + usage["candidates_token_count"]
                + usage["thoughts_token_count"],
            )
            new_requests = new_usage.get("requests")
            if isinstance(new_requests, list) and new_requests:
                usage.setdefault("requests", []).extend(new_requests)
            new_max_req = int(new_usage.get("max_request_prompt_tokens") or 0)
            if new_max_req > 0 or usage.get("max_request_prompt_tokens"):
                usage["max_request_prompt_tokens"] = max(
                    int(usage.get("max_request_prompt_tokens") or 0),
                    new_max_req,
                )
            tool_calls.extend(worker_out.get("tool_calls") or [])
            thoughts.extend(worker_out.get("thoughts") or [])
            stop_reason = worker_out.get("stop_reason")

            with self._lock:
                if job_id in self._cancelled:
                    status = "cancelled"
                    error_msg = "Job cancelled during execution."
                else:
                    status = "ok"
                    new_signals = read_progress(progress_path, limit=10**6)[progress_before:]
                    if stop_reason in CHECKPOINT_STOP_REASONS:
                        # Budget reached: not an error. The lead reviews and continues or redirects.
                        status = "checkpoint"
                    elif new_signals and new_signals[-1].get("kind") in ("question", "blocker"):
                        status = "needs_input"

        except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError) as exc:
            with self._lock:
                is_canc = job_id in self._cancelled
            if is_canc:
                status = "cancelled"
                error_msg = "Job cancelled during execution."
            else:
                status = "timeout"
                error_msg = f"TimeoutError: {exc}"
            self._merge_partial_exception_data(exc, usage, tool_calls, thoughts)
            if getattr(exc, "partial_containment", None):
                worker_out["containment"] = getattr(exc, "partial_containment")
            if getattr(exc, "partial_containment_warning", None):
                worker_out["containment_warning"] = getattr(exc, "partial_containment_warning")
        except Exception as exc:
            with self._lock:
                is_canc = job_id in self._cancelled
            if is_canc or status == "cancelled":
                status = "cancelled"
                error_msg = "Job cancelled by user."
            else:
                status = "error"
                error_msg = f"{type(exc).__name__}: {exc}"
            self._merge_partial_exception_data(exc, usage, tool_calls, thoughts)
            if getattr(exc, "partial_containment", None):
                worker_out["containment"] = getattr(exc, "partial_containment")
            if getattr(exc, "partial_containment_warning", None):
                worker_out["containment_warning"] = getattr(exc, "partial_containment_warning")
        finally:
            try:
                hooks_tampered = _restore_directory_tree(host_git_dir / "hooks", hooks_snap)
                config_tampered = _restore_directory_tree(host_git_dir / "config", config_snap)
                if hooks_tampered or config_tampered:
                    status = "security_violation"
                    error_msg = "Worker attempted to modify host .git/hooks or .git/config; restored."

                if in_place:
                    diff_text, files_changed, symlink_escapes, in_place_notes = self._finalize_in_place(
                        baseline_sha,
                        pre_untracked or {},
                        orig_branch=orig_branch,
                        pre_branches=pre_branches,
                    )
                    baseline_file_hashes = self._compute_changed_baseline_hashes(
                        self.workspace_root, baseline_sha, files_changed, {}
                    )
                else:
                    diff_text, files_changed, symlink_escapes = self._collect_worktree_diff(
                        wt_dir, baseline_sha, allowed_paths=allowed_paths
                    )
                    baseline_file_hashes = self._compute_changed_baseline_hashes(
                        wt_dir, baseline_sha, files_changed, baseline_file_hashes
                    )
            except Exception as fin_exc:
                diff_text, files_changed, symlink_escapes = "", [], []
                if status in ("running", "ok"):
                    status = "error"
                    error_msg = f"Failed to collect worktree diff: {type(fin_exc).__name__}: {fin_exc}"
            if acceptance and (
                status in ("ok", "checkpoint", "needs_input")
                or (status == "timeout" and files_changed)
            ):
                try:
                    acceptance_results = run_acceptance(acceptance, self.workspace_root if in_place else wt_dir)
                except Exception as acc_exc:
                    acceptance_results = [
                        {"command": "<acceptance>", "passed": False, "error": f"{type(acc_exc).__name__}: {acc_exc}"}
                    ]
                try:
                    if in_place:
                        diff_text, files_changed, symlink_escapes, in_place_notes = self._finalize_in_place(
                            baseline_sha,
                            pre_untracked or {},
                            orig_branch=orig_branch,
                            pre_branches=pre_branches,
                        )
                        baseline_file_hashes = self._compute_changed_baseline_hashes(
                            self.workspace_root, baseline_sha, files_changed, {}
                        )
                    else:
                        diff_text, files_changed, symlink_escapes = self._collect_worktree_diff(
                            wt_dir, baseline_sha, allowed_paths=allowed_paths
                        )
                        baseline_file_hashes = self._compute_changed_baseline_hashes(
                            wt_dir, baseline_sha, files_changed, baseline_file_hashes
                        )
                except Exception:
                    pass
            if symlink_escapes:
                for esc in symlink_escapes:
                    if esc not in scope_violations:
                        scope_violations.append(esc)
                if status == "ok":
                    status = "scope_violation"
                    error_msg = f"Worker created symlink escaping worktree: {symlink_escapes}"

            if allowed_paths:
                for fpath in files_changed:
                    if not _matches_allowed_paths(fpath, allowed_paths):
                        if fpath not in scope_violations:
                            scope_violations.append(fpath)
                if scope_violations and status == "ok":
                    status = "scope_violation"
                    error_msg = (
                        f"Worker modified files outside allowed_paths {allowed_paths}: "
                        f"{scope_violations}. Review the diff with get_diff and either call "
                        f"delegate_apply(job_id, allow_scope_violations=True) to approve, or call "
                        f"delegate_continue(job_id, follow_up_prompt=..., allowed_paths=...) to adjust scope."
                    )

            provenance = extract_diff_provenance(diff_text)
            for fpath in files_changed:
                if fpath not in provenance["files_changed"]:
                    provenance["files_changed"].append(fpath)

            diff_path = self.store.diff_path(job_id)
            diff_path.write_text(diff_text, encoding="utf-8")

            finished_at = round(time.time(), 3)
            clipped_text, was_truncated, orig_tokens = truncate_to_token_budget(
                response_text, MAX_TOOL_RESPONSE_TOKENS
            )
            receipt_payload = {
                "job_id": job_id,
                "kind": "delegate",
                "status": status,
                "auth_mode": auth_mode,
                "model": model,
                "vertex_project": project,
                "vertex_location": location,
                "reroute_note": reroute_note,
                "prompts": recorded_prompts,
                "allowed_paths": allowed_paths,
                "enabled_tools": enabled_tools,
                "disabled_tools": disabled_tools,
                "enable_subagents": enable_subagents,
                "scrub_receipt": scrub_receipt,
                "scope_violations": scope_violations,
                "symlink_escapes": symlink_escapes,
                "worktree_dir": str(wt_dir),
                "baseline_commit_sha": baseline_sha,
                "baseline_file_hashes": baseline_file_hashes,
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
                "thinking_level": thinking_level,
                "plan": plan or [],
                "acceptance": acceptance or [],
                "budget": {"max_tool_calls": max_tool_calls, "max_total_tokens": max_total_tokens},
                "stop_reason": stop_reason,
                "containment": worker_out.get("containment", "none") if isinstance(worker_out, dict) else "none",
                "containment_warning": (
                    worker_out.get("containment_warning") if isinstance(worker_out, dict) else None
                ),
                "acceptance_results": acceptance_results,
                "acceptance_passed": (
                    all(r.get("passed") for r in acceptance_results) if acceptance_results else None
                ),
                "external_worktree_tamper": bool(not in_place and files_changed and not workspace_dir_accessed),
            }
            if in_place:
                # The edits are already in the user's working tree, so the job counts as applied
                # the moment it finishes (even on error or timeout, partial edits are on disk).
                applied_file_hashes = {
                    fpath: _hash_path(self.workspace_root / fpath) for fpath in files_changed
                }
                new_files = [f for f in files_changed if baseline_file_hashes.get(f) is None]
                tracked_files = [f for f in files_changed if baseline_file_hashes.get(f) is not None]
                undo_parts: list[str] = []
                if tracked_files:
                    undo_parts.append(
                        f"git checkout {baseline_sha[:12]} -- "
                        + " ".join(shlex.quote(f) for f in tracked_files)
                    )
                if new_files:
                    undo_parts.append("rm -- " + " ".join(shlex.quote(f) for f in new_files))
                receipt_payload.update(
                    {
                        "in_place": True,
                        "in_place_notes": in_place_notes,
                        "pre_untracked_hashes": pre_untracked or {},
                        "applied": True,
                        "applied_at_epoch": finished_at,
                        "applied_files": list(files_changed),
                        "applied_file_hashes": applied_file_hashes,
                        "undo_hint": " && ".join(undo_parts) if undo_parts else None,
                    }
                )
                if status not in ("ok", "checkpoint", "needs_input") and files_changed:
                    receipt_payload["error"] = (
                        f"{error_msg or status}. Partial in_place edits remain in the working tree; "
                        f"see undo_hint to revert them."
                    )
            saved = self.store.write_receipt(
                job_id,
                receipt_payload,
                scrub=scrub_receipt,
                workspace_root=str(self.workspace_root),
                api_key=api_key,
            )
            with self._lock:
                if job_id in self._cancelled and wt_dir.exists():
                    self._cleanup_worktree(wt_dir)
            if in_place:
                self._release_in_place_slot(job_id)

        return saved

    @staticmethod
    def _merge_partial_exception_data(
        exc: BaseException,
        usage: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        thoughts: list[str],
    ) -> None:
        partial_u = getattr(exc, "partial_usage", None)
        if isinstance(partial_u, dict):
            for k in (
                "prompt_token_count",
                "cached_content_token_count",
                "candidates_token_count",
                "thoughts_token_count",
                "total_token_count",
            ):
                usage[k] = int(usage.get(k) or 0) + int(partial_u.get(k) or 0)
            usage["total_token_count"] = max(
                usage["total_token_count"],
                usage["prompt_token_count"]
                + usage["candidates_token_count"]
                + usage["thoughts_token_count"],
            )
            partial_reqs = partial_u.get("requests")
            if isinstance(partial_reqs, list) and partial_reqs:
                usage.setdefault("requests", []).extend(partial_reqs)
            partial_max_req = int(partial_u.get("max_request_prompt_tokens") or 0)
            if partial_max_req > 0:
                usage["max_request_prompt_tokens"] = max(
                    int(usage.get("max_request_prompt_tokens") or 0),
                    partial_max_req,
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
        api_key: str | None = None,
        timeout_seconds: float = 1200.0,
        enabled_tools: list[str] | None = None,
        disabled_tools: list[str] | None = None,
        enable_subagents: bool = False,
        wait: bool = False,
        scrub_receipt: bool = False,
        in_place: bool = False,
        thinking_level: str | None = None,
        plan: list[str] | None = None,
        acceptance: list[str] | None = None,
        max_tool_calls: int | None = None,
        max_total_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Start an Antigravity worker job.

        By default the worker edits an isolated standalone copy of the repo and its diff is
        applied later with delegate_apply. With in_place=True the worker edits the real working
        tree directly (requires a clean tracked tree, one in_place job at a time, and no
        allowed_paths); the job is recorded as applied as soon as it finishes.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        validated_paths = validate_allowed_paths(allowed_paths)
        resolved_thinking = resolve_thinking_level(thinking_level)
        plan_steps = validate_plan(plan)
        acceptance_cmds = validate_acceptance(acceptance, self.workspace_root)
        budget_calls, budget_tokens = resolve_budget(max_tool_calls, max_total_tokens)
        worker_prompt = render_worker_prompt(
            prompt,
            plan=plan_steps,
            acceptance=acceptance_cmds,
            max_tool_calls=budget_calls,
            max_total_tokens=budget_tokens,
        )
        if in_place and validated_paths:
            raise ValueError(
                "allowed_paths cannot be enforced in in_place mode (edits land directly in your "
                "files). Drop allowed_paths or use the default isolated copy."
            )
        req_model_clean = (model or "").strip()
        if req_model_clean and not req_model_clean.startswith("${"):
            chosen_model = req_model_clean
        else:
            chosen_model = first_valid_env(
                os.environ,
                "CC_AGY_DEFAULT_DELEGATE_MODEL",
                "CLAUDE_PLUGIN_OPTION_DEFAULT_DELEGATE_MODEL",
                "CC_AGY_USERCFG_DEFAULT_DELEGATE_MODEL",
                default="gemini-3.1-pro-preview",
            )

        auth = resolve_auth_config(
            requested_project=project,
            requested_location=location,
            requested_api_key=api_key,
            model=chosen_model,
        )
        auth_mode = auth["auth_mode"]
        resolved_key = auth.get("api_key")
        resolved_project = auth.get("project")
        resolved_location = auth.get("location")
        reroute_note = auth.get("reroute_note")

        job_id = f"job-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        started_at = round(time.time(), 3)
        pre_untracked: dict[str, str | None] = {}
        if in_place:
            self._acquire_in_place_slot(job_id)
            try:
                baseline_sha, pre_untracked = self._check_in_place_ready()
            except Exception:
                self._release_in_place_slot(job_id)
                raise
            wt_dir, baseline_hashes = self.workspace_root, {}
        else:
            wt_dir, baseline_sha, baseline_hashes = self._prepare_worktree(job_id)

        with self._lock:
            if resolved_project:
                self._raw_projects[job_id] = resolved_project
            if resolved_key:
                self._raw_api_keys[job_id] = resolved_key

        initial_receipt = {
            "job_id": job_id,
            "kind": "delegate",
            "status": "running",
            "auth_mode": auth_mode,
            "model": chosen_model,
            "vertex_project": resolved_project,
            "vertex_location": resolved_location,
            "reroute_note": reroute_note,
            "prompts": [prompt.strip()],
            "allowed_paths": validated_paths,
            "enabled_tools": enabled_tools,
            "disabled_tools": disabled_tools,
            "enable_subagents": enable_subagents,
            "scrub_receipt": scrub_receipt,
            "scope_violations": [],
            "worktree_dir": str(wt_dir),
            "baseline_commit_sha": baseline_sha,
            "baseline_file_hashes": baseline_hashes,
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
            "in_place": in_place,
            "thinking_level": resolved_thinking or "DEFAULT",
            "plan": plan_steps,
            "acceptance": acceptance_cmds,
            "budget": {"max_tool_calls": budget_calls, "max_total_tokens": budget_tokens},
            "error": None,
        }
        self.store.write_receipt(
            job_id,
            initial_receipt,
            scrub=scrub_receipt,
            workspace_root=str(self.workspace_root),
            api_key=resolved_key,
        )

        fut = self._executor.submit(
            self._execute_job,
            job_id=job_id,
            worker_prompts=[worker_prompt],
            recorded_prompts=[prompt.strip()],
            model=chosen_model,
            auth_mode=auth_mode,
            api_key=resolved_key,
            project=resolved_project,
            location=resolved_location,
            reroute_note=reroute_note,
            allowed_paths=validated_paths,
            timeout_seconds=timeout_seconds,
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
            enable_subagents=enable_subagents,
            wt_dir=wt_dir,
            baseline_sha=baseline_sha,
            baseline_file_hashes=baseline_hashes,
            started_at=started_at,
            scrub_receipt=scrub_receipt,
            in_place=in_place,
            pre_untracked=pre_untracked,
            thinking_level=resolved_thinking or "DEFAULT",
            plan=plan_steps,
            acceptance=acceptance_cmds,
            max_tool_calls=budget_calls,
            max_total_tokens=budget_tokens,
        )
        with self._lock:
            self._futures[job_id] = fut

        if wait:
            self._wait_for_future_or_cancel(job_id, fut, timeout=max(0.5, float(timeout_seconds) + 30.0))
            receipt = self.store.read_receipt(job_id)
            return self._digest_with_progress(receipt)

        return {
            "job_id": job_id,
            "status": "running",
            "auth_mode": auth_mode,
            "model": chosen_model,
            "vertex_location": resolved_location,
            "reroute_note": reroute_note,
            "allowed_paths": validated_paths,
            "worktree_dir": str(wt_dir),
            "in_place": in_place,
        }

    def _wait_for_future_or_cancel(
        self,
        job_id: str,
        fut: Future[dict[str, Any]],
        *,
        timeout: float,
    ) -> None:
        """Wait for `fut` to complete or for `job_id` to be added to `self._cancelled`."""
        deadline = time.monotonic() + max(0.1, float(timeout))
        while not fut.done():
            with self._lock:
                if job_id in self._cancelled:
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                fut.result(timeout=min(0.05, remaining))
                return
            except (TimeoutError, FuturesTimeoutError):
                continue
            except (Exception, asyncio.CancelledError):
                return

    def delegate_status(
        self,
        job_id: str | None = None,
        *,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return status for a single job or list recent jobs within the MCP token cap."""
        baseline_rec = self.store.read_baseline_record(self.workspace_root) or {}
        active_baseline_sha = baseline_rec.get("baseline_head_sha")
        if job_id:
            safe_id = validate_job_id(job_id)
            receipt = self.store.read_receipt(safe_id)
            digest = self._digest_with_progress(receipt, include_summary=True)
            digest["baseline_head_sha"] = active_baseline_sha
            return digest

        all_receipts = self.store.list_receipts(workspace_root=self.workspace_root)
        max_items = max(1, min(int(limit), 50))
        recent = all_receipts[-max_items:]
        totals = self.store.summarize_costs(workspace_root=self.workspace_root)
        summaries: list[dict[str, Any]] = []
        for r in reversed(recent):
            item = self._compact_digest(r, include_summary=False, include_acceptance_details=False)
            candidate_list = [item] + summaries
            probe = {
                "jobs": candidate_list,
                "returned_jobs": len(candidate_list),
                "total_jobs": len(all_receipts),
                "baseline_head_sha": active_baseline_sha,
                "totals": totals,
            }
            if summaries and estimate_tokens(json.dumps(probe, indent=2, sort_keys=True)) > MAX_TOOL_RESPONSE_TOKENS - 200:
                break
            summaries.insert(0, item)
        return {
            "jobs": summaries,
            "returned_jobs": len(summaries),
            "total_jobs": len(all_receipts),
            "baseline_head_sha": active_baseline_sha,
            "totals": totals,
        }

    def delegate_result(
        self,
        job_id: str,
        *,
        wait: bool = True,
        timeout: float = 240.0,
    ) -> dict[str, Any]:
        """Return the compact completion digest for job_id, optionally waiting if still running."""
        safe_id = validate_job_id(job_id)
        with self._lock:
            fut = self._futures.get(safe_id)
        if fut is not None and wait and not fut.done():
            capped_timeout = max(0.1, min(float(timeout), 540.0))
            self._wait_for_future_or_cancel(safe_id, fut, timeout=capped_timeout)
        receipt = self.store.read_receipt(safe_id)
        return self._digest_with_progress(receipt)

    def get_diff(
        self,
        job_id: str,
        *,
        file_filter: str | None = None,
        page: int = 1,
        page_tokens: int = DEFAULT_DIFF_PAGE_TOKENS,
    ) -> dict[str, Any]:
        """Return paginated unified git diff for job_id measured in tokens."""
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

        pages = paginate_text_by_tokens(full_diff, page_tokens=page_tokens)
        total_pages = len(pages)
        clamped_page = max(1, min(int(page), total_pages))
        page_slice = pages[clamped_page - 1]

        return {
            "job_id": safe_id,
            "status": receipt.get("status"),
            "file_filter": file_filter,
            "page": clamped_page,
            "total_pages": total_pages,
            "page_tokens_estimated": estimate_tokens(page_slice),
            "total_diff_tokens_estimated": estimate_tokens(full_diff),
            "files_changed": receipt.get("files_changed", []),
            "diff": page_slice,
        }

    def delegate_apply(
        self,
        job_id: str,
        *,
        cleanup_worktree: bool = True,
        allow_scope_violations: bool = False,
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
                    )
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _delegate_apply_locked(
        self,
        safe_id: str,
        *,
        cleanup_worktree: bool = True,
        allow_scope_violations: bool = False,
    ) -> dict[str, Any]:
        receipt = self.store.read_receipt(safe_id)
        if receipt.get("applied"):
            prov = receipt.get("provenance") or {}
            return {
                "job_id": safe_id,
                "applied": True,
                "already_applied": True,
                "applied_files": receipt.get("applied_files", []),
                "diff_sha256": prov.get("diff_sha256"),
            }

        status = receipt.get("status")
        has_clean_timeout_edits = (
            status == "timeout"
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

        wt_dir = Path(str(receipt.get("worktree_dir") or ""))
        baseline_sha = str(receipt.get("baseline_commit_sha") or "HEAD")
        diff_path = self.store.diff_path(safe_id)
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""
        raw_files_changed: list[str] = list(receipt.get("files_changed") or [])
        files_changed = [
            f
            for f in raw_files_changed
            if not _is_ephemeral_build_or_test_artifact(
                f, root=wt_dir if wt_dir.exists() else self.workspace_root
            )
        ]
        allowed_paths: list[str] = validate_allowed_paths(receipt.get("allowed_paths") or [])
        baseline_hashes: dict[str, str | None] = self._compute_changed_baseline_hashes(
            wt_dir,
            baseline_sha,
            files_changed,
            dict(receipt.get("baseline_file_hashes") or {}),
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
                base_show = subprocess.run(
                    ["git", "show", f"{baseline_sha}:{rel_path}"],
                    cwd=str(wt_dir),
                    capture_output=True,
                    check=False,
                )
                if base_show.returncode != 0:
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
                    merge_res = subprocess.run(
                        ["git", "merge-file", "-p", str(f_cur), str(f_base), str(f_wrk)],
                        capture_output=True,
                        check=False,
                    )
                    if merge_res.returncode != 0:
                        raise ValueError(
                            f"Conflict detected: host files {conflicted_files} were modified since job "
                            f"'{safe_id}' started and could not be cleanly merged."
                        )
                    merged_payloads[rel_path] = merge_res.stdout

        applied_files: list[str] = []
        if wt_dir.exists():
            for rel_path in files_changed:
                src_raw = wt_dir / rel_path
                dst_raw = self.workspace_root / rel_path
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
                    shutil.copy2(src_raw, dst_raw)
                    applied_files.append(rel_path)
                elif not src_raw.exists():
                    if dst_raw.exists() or dst_raw.is_symlink():
                        dst_raw.unlink()
                    applied_files.append(rel_path)
        elif diff_text.strip():
            subprocess.run(
                ["git", "apply", "--binary", "--whitespace=nowarn", "-"],
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
        applied_file_hashes: dict[str, str | None] = {}
        for fpath in applied_files:
            if fpath not in applied_prov["files_changed"]:
                applied_prov["files_changed"].append(fpath)
            dst_p = self.workspace_root / fpath
            if dst_p.is_symlink():
                applied_file_hashes[fpath] = f"symlink:{os.readlink(dst_p)}"
            elif dst_p.exists() and dst_p.is_file():
                applied_file_hashes[fpath] = sha256_bytes(dst_p.read_bytes())
            else:
                applied_file_hashes[fpath] = None

        receipt["applied"] = True
        receipt["applied_at_epoch"] = round(time.time(), 3)
        receipt["applied_files"] = applied_files
        receipt["applied_file_hashes"] = applied_file_hashes
        receipt["baseline_file_hashes"] = baseline_hashes
        receipt["provenance"] = applied_prov
        self.store.write_receipt(
            safe_id,
            receipt,
            scrub=bool(receipt.get("scrub_receipt", False)),
            workspace_root=str(self.workspace_root),
        )

        return {
            "job_id": safe_id,
            "applied": True,
            "applied_files": applied_files,
            "diff_sha256": applied_prov["diff_sha256"],
        }

    def delegate_continue(
        self,
        job_id: str,
        follow_up_prompt: str,
        *,
        allowed_paths: list[str] | None = None,
        acceptance: list[str] | None = None,
        timeout_seconds: float = 600.0,
        wait: bool = True,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Send a follow-up prompt into the same job's isolated worktree without re-running prior prompts."""
        safe_id = validate_job_id(job_id)
        if not follow_up_prompt or not follow_up_prompt.strip():
            raise ValueError("follow_up_prompt must be a non-empty string")

        with self._lock:
            fut = self._futures.get(safe_id)
            if fut is not None and not fut.done():
                raise RuntimeError(f"Job '{safe_id}' is currently running; wait before continuing.")
            self._cancelled.discard(safe_id)

        receipt = self.store.read_receipt(safe_id)
        in_place = bool(receipt.get("in_place"))
        pre_untracked: dict[str, str | None] = {}
        if in_place:
            baseline_sha = str(receipt.get("baseline_commit_sha") or "HEAD")
            stray_untracked = self._verify_in_place_unchanged(receipt)
            self._acquire_in_place_slot(safe_id)
            wt_dir = self.workspace_root
            baseline_hashes: dict[str, str | None] = {}
            pre_untracked = dict(receipt.get("pre_untracked_hashes") or {})
            # New untracked files that someone else created (e.g. a test run's artifacts) are treated
            # as pre-existing: never credited to the worker, and the Stop hook still checks them.
            pre_untracked.update(stray_untracked)
        else:
            if receipt.get("applied"):
                raise ValueError(
                    f"Job '{safe_id}' has already been applied to the workspace. "
                    f"Start a new job with delegate_start to preserve applied provenance."
                )

            wt_dir = Path(str(receipt.get("worktree_dir") or ""))
            baseline_sha = str(receipt.get("baseline_commit_sha") or "HEAD")
            baseline_hashes = dict(receipt.get("baseline_file_hashes") or {})
            if not wt_dir.exists():
                wt_dir, baseline_sha, baseline_hashes = self._prepare_worktree(safe_id)

        prior_prompts = list(receipt.get("prompts") or [])
        clean_followup = follow_up_prompt.strip()
        recorded_prompts = prior_prompts + [clean_followup]

        # Attach a concise context note of the original task and latest turn so the fresh
        # Antigravity session retains task context without re-executing prior turns.
        if prior_prompts:
            orig_task = prior_prompts[0]
            last_turn = prior_prompts[-1] if len(prior_prompts) > 1 else None
            prior_summary = (
                f"original_task={orig_task!r}, latest_turn={last_turn!r}"
                if last_turn is not None
                else f"{orig_task}"
            )
            context_note = (
                f"[Prior turn context (baseline {baseline_sha}, files_changed={receipt.get('files_changed', [])}, "
                f"run `git diff {baseline_sha}` to inspect prior work): "
                f"{prior_summary}]\n{clean_followup}"
            )
            worker_prompts = [context_note]
        else:
            worker_prompts = [clean_followup]
        plan_steps = list(receipt.get("plan") or [])
        if acceptance is not None:
            acceptance_cmds = validate_acceptance(acceptance, self.workspace_root)
        else:
            acceptance_cmds = list(receipt.get("acceptance") or [])
        budget = receipt.get("budget") or {}
        # A stored None means the lead disabled that cap; a missing key (older receipt) means defaults.
        budget_calls, budget_tokens = resolve_budget(
            (budget.get("max_tool_calls") or 0) if "max_tool_calls" in budget else None,
            (budget.get("max_total_tokens") or 0) if "max_total_tokens" in budget else None,
        )
        worker_prompts = [
            render_worker_prompt(
                worker_prompts[0],
                plan=plan_steps,
                acceptance=acceptance_cmds,
                max_tool_calls=budget_calls,
                max_total_tokens=budget_tokens,
            )
        ]

        model = str(receipt.get("model") or "gemini-3.1-pro-preview")
        with self._lock:
            raw_proj = self._raw_projects.get(safe_id)
            raw_key = self._raw_api_keys.get(safe_id)
        prior_auth_mode = receipt.get("auth_mode")
        if prior_auth_mode == "api_key" and not project:
            if not resolve_api_key(api_key or raw_key):
                raise ValueError(
                    "Job was started in api_key mode, but no Gemini API key is available after server restart. "
                    "Pass api_key=... or set GEMINI_API_KEY."
                )
        inherited_project = project
        if not inherited_project and prior_auth_mode != "api_key":
            inherited_project = raw_proj or receipt.get("vertex_project")
        auth = resolve_auth_config(
            requested_project=inherited_project,
            requested_location=location or receipt.get("vertex_location"),
            requested_api_key=api_key or raw_key,
            model=model,
        )
        auth_mode = auth["auth_mode"]
        resolved_key = auth.get("api_key")
        resolved_project = auth.get("project")
        resolved_location = auth.get("location")
        reroute_note = auth.get("reroute_note")

        with self._lock:
            if resolved_project:
                self._raw_projects[safe_id] = resolved_project
            if resolved_key:
                self._raw_api_keys[safe_id] = resolved_key

        if allowed_paths is not None:
            if in_place and allowed_paths:
                raise ValueError("allowed_paths cannot be combined with in_place=True")
            resolved_allowed_paths = validate_allowed_paths(allowed_paths)
        else:
            resolved_allowed_paths = validate_allowed_paths(receipt.get("allowed_paths") or [])
        allowed_paths = resolved_allowed_paths
        enabled_tools = receipt.get("enabled_tools")
        disabled_tools = receipt.get("disabled_tools")
        enable_subagents = bool(receipt.get("enable_subagents", False))
        scrub_receipt = bool(receipt.get("scrub_receipt", False))
        started_at = float(receipt.get("started_at_epoch") or time.time())

        if wait:
            updated = self._execute_job(
                job_id=safe_id,
                worker_prompts=worker_prompts,
                recorded_prompts=recorded_prompts,
                model=model,
                auth_mode=auth_mode,
                api_key=resolved_key,
                project=resolved_project,
                location=resolved_location,
                reroute_note=reroute_note,
                allowed_paths=allowed_paths,
                timeout_seconds=timeout_seconds,
                enabled_tools=enabled_tools,
                disabled_tools=disabled_tools,
                enable_subagents=enable_subagents,
                wt_dir=wt_dir,
                baseline_sha=baseline_sha,
                baseline_file_hashes=baseline_hashes,
                started_at=started_at,
                prior_usage=receipt.get("usage"),
                prior_tool_calls=receipt.get("tool_calls"),
                scrub_receipt=scrub_receipt,
                in_place=in_place,
                pre_untracked=pre_untracked,
                thinking_level=receipt.get("thinking_level"),
                plan=plan_steps,
                acceptance=acceptance_cmds,
                max_tool_calls=budget_calls,
                max_total_tokens=budget_tokens,
            )
            return self._digest_with_progress(updated)

        fut = self._executor.submit(
            self._execute_job,
            job_id=safe_id,
            worker_prompts=worker_prompts,
            recorded_prompts=recorded_prompts,
            model=model,
            auth_mode=auth_mode,
            api_key=resolved_key,
            project=resolved_project,
            location=resolved_location,
            reroute_note=reroute_note,
            allowed_paths=allowed_paths,
            timeout_seconds=timeout_seconds,
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
            enable_subagents=enable_subagents,
            wt_dir=wt_dir,
            baseline_sha=baseline_sha,
            baseline_file_hashes=baseline_hashes,
            started_at=started_at,
            prior_usage=receipt.get("usage"),
            prior_tool_calls=receipt.get("tool_calls"),
            scrub_receipt=scrub_receipt,
            in_place=in_place,
            pre_untracked=pre_untracked,
            thinking_level=receipt.get("thinking_level"),
            plan=plan_steps,
            acceptance=acceptance_cmds,
            max_tool_calls=budget_calls,
            max_total_tokens=budget_tokens,
        )
        with self._lock:
            self._futures[safe_id] = fut
        return {
            "job_id": safe_id,
            "status": "running",
            "auth_mode": auth_mode,
            "prompts_count": len(recorded_prompts),
        }

    def _verify_in_place_unchanged(self, receipt: dict[str, Any]) -> dict[str, str | None]:
        """Refuse to continue an in_place job if anyone else changed the tree since its last turn."""
        root = self.workspace_root
        baseline_sha = str(receipt.get("baseline_commit_sha") or "")
        head_res = _run_git(["rev-parse", "HEAD"], cwd=root, check=False)
        if head_res.stdout.strip() != baseline_sha:
            raise ValueError(
                "HEAD moved since this in_place job ran (a commit was made). "
                "Start a new in_place job instead of continuing this one."
            )
        applied_files = set(receipt.get("applied_files") or [])
        for fpath, expected in (receipt.get("applied_file_hashes") or {}).items():
            if _hash_path(root / fpath) != expected:
                raise ValueError(
                    f"{fpath} changed after this in_place job finished. Commit or revert that edit, "
                    f"then start a new in_place job."
                )
        tracked_res = _run_git(["diff", "--name-only", "--no-renames", baseline_sha], cwd=root, check=False)
        extra_tracked = [
            f.strip() for f in tracked_res.stdout.splitlines() if f.strip() and f.strip() not in applied_files
        ]
        pre_untracked = receipt.get("pre_untracked_hashes") or {}
        extra_untracked = [
            rel
            for rel, h in _snapshot_untracked_hashes(root).items()
            if rel not in applied_files
            and not rel.startswith((".cc-agy-data/", ".worktrees/"))
            and not _is_ephemeral_build_or_test_artifact(rel, root=root)
            and pre_untracked.get(rel, "__absent__") != h
        ]
        if extra_tracked:
            raise ValueError(
                "Tracked files outside this in_place job changed since its last turn: "
                f"{extra_tracked[:10]}. Commit or revert them, then start a new job."
            )
        current = _snapshot_untracked_hashes(root)
        return {rel: current.get(rel) for rel in extra_untracked}

    def delegate_cancel(self, job_id: str, *, cleanup_worktree: bool = True) -> dict[str, Any]:
        """Cancel a job and clean up its worktree while preserving its cancelled receipt status."""
        safe_id = validate_job_id(job_id)
        with self._lock:
            self._cancelled.add(safe_id)
            fut = self._futures.get(safe_id)
            if fut is not None:
                fut.cancel()
            still_running = fut is not None and not fut.done()

        receipt = self.store.read_receipt(safe_id)
        wt_dir = Path(str(receipt.get("worktree_dir") or ""))
        # Only remove wt_dir immediately if the worker thread is not currently running inside it;
        # if it is still running, _execute_job's finally block removes wt_dir upon exit.
        if cleanup_worktree and wt_dir.exists() and not still_running:
            self._cleanup_worktree(wt_dir)

        if receipt.get("status") == "running":
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
            digest["poll_hint"] = (
                "Still running. Poll again with delegate_result(wait=true) (waits up to 240s, which keeps "
                "your prompt cache warm), or check progress with delegate_status."
            )
        return digest

    def mark_running_interrupted(self) -> list[str]:
        """Called when the MCP server is shutting down: record unfinished jobs as interrupted instead of
        leaving them 'running' forever. Token usage for the unfinished turn cannot be recovered."""
        marked: list[str] = []
        with self._lock:
            live = {jid for jid, fut in self._futures.items() if not fut.done()}
        for jid in live:
            try:
                receipt = self.store.read_receipt(jid)
            except Exception:
                continue
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
            except Exception:
                continue
        return marked

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the underlying thread pool executor."""
        self._executor.shutdown(wait=wait)

    @staticmethod
    def _compact_digest(
        receipt: dict[str, Any],
        *,
        include_summary: bool = True,
        include_acceptance_details: bool = True,
    ) -> dict[str, Any]:
        """Return a token-bounded digest suitable for Claude Code's context window."""
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
            if receipt.get("acceptance_passed") is not None:
                slim["acceptance_passed"] = receipt.get("acceptance_passed")
            if receipt.get("error"):
                slim["error"] = receipt.get("error")
            if receipt.get("stop_reason"):
                slim["stop_reason"] = receipt.get("stop_reason")
            return slim

        digest_usage = dict(receipt.get("usage") or {})
        raw_reqs = digest_usage.get("requests")
        if isinstance(raw_reqs, list):
            digest_usage["requests_count"] = len(raw_reqs)
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
            "stop_reason": receipt.get("stop_reason"),
            "budget": receipt.get("budget"),
            "containment": receipt.get("containment", "none"),
            "containment_warning": receipt.get("containment_warning"),
            "acceptance_passed": receipt.get("acceptance_passed"),
            "error": receipt.get("error"),
        }
        if include_acceptance_details:
            digest["acceptance_results"] = receipt.get("acceptance_results") or []
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
            summary_text = str(receipt.get("summary") or "")
            meta_tokens = estimate_tokens(json.dumps(digest, indent=2, sort_keys=True))
            summary_budget = max(500, MAX_TOOL_RESPONSE_TOKENS - meta_tokens - 250)
            clipped_summary, truncated, est_tokens = truncate_to_token_budget(
                summary_text, summary_budget
            )
            digest["summary"] = clipped_summary
            digest["summary_truncated"] = bool(receipt.get("summary_truncated") or truncated)
            digest["summary_tokens_estimated"] = est_tokens
        return digest
