"""Isolated job manager for Antigravity SDK delegations."""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import fnmatch
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import uuid

from server.agy_runner import (
    is_read_only_tool_set,
    is_straitjacket_available,
    read_progress,
    resolve_thinking_level,
    run_antigravity_worker,
    verify_worker_run_record,
)
from server.model_routes import (
    DEFAULT_ECONOMY_FLASH_MAX_TOOL_CALLS,
    DEFAULT_FLASH_MAX_TOOL_CALLS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TOTAL_TOKENS,
)
from server.repo_map import build_focused_file_outline, build_repo_map
from server.receipts import (
    BudgetExceededError,
    ReceiptStore,
    enforce_daily_budget,
    extract_diff_provenance,
    parse_diff_git_header,
    parse_unified_header_path,
    receipt_owner_pid_alive as _receipt_owner_pid_alive,
    resolve_workspace_root,
    sha256_bytes,
    validate_job_id,
)
from server.verification_policy import (
    classify_verification,
    command_strength,
    has_real_acceptance_pass,
    is_non_smoke_passing_entry,
    satisfies_auto_apply_verification,
    satisfies_timeout_promotion,
)
from server.regions import (
    DEFAULT_ESCALATION_MODEL,
    DEFAULT_DIFF_PAGE_TOKENS,
    MAX_TOOL_RESPONSE_TOKENS,
    estimate_cost_usd,
    estimate_tokens,
    env_flag,
    is_ge_mode_pinned,
    is_strict_mode_enabled,
    paginate_text_by_tokens,
    resolve_api_key,
    resolve_auth_config,
    resolve_cost_profile,
    resolve_default_delegate_model,
    truncate_to_token_budget,
)


from server.job_git_utils import (
    ACCEPTANCE_OUTPUT_TOKENS as ACCEPTANCE_OUTPUT_TOKENS,
    CHECKPOINT_STOP_REASONS as CHECKPOINT_STOP_REASONS,
    DEFAULT_ACCEPTANCE_TIMEOUT as DEFAULT_ACCEPTANCE_TIMEOUT,
    DEFAULT_BOUNDED_ACCEPTANCE_TIMEOUT as DEFAULT_BOUNDED_ACCEPTANCE_TIMEOUT,
    DEFAULT_MAX_WAIT_SLICE_SECONDS as DEFAULT_MAX_WAIT_SLICE_SECONDS,
    MAX_ACCEPTANCE_COMMANDS as MAX_ACCEPTANCE_COMMANDS,
    MAX_PLAN_STEPS as MAX_PLAN_STEPS,
    _EPHEMERAL_CACHE_DIRS as _EPHEMERAL_CACHE_DIRS,
    _EPHEMERAL_CACHE_SUFFIXES as _EPHEMERAL_CACHE_SUFFIXES,
    _SAFE_NOTE_EXTENSIONS as _SAFE_NOTE_EXTENSIONS,
    _UNTRACKED_MODEL_OR_DATA_DUMP_SUFFIXES as _UNTRACKED_MODEL_OR_DATA_DUMP_SUFFIXES,
    _env_int as _env_int,
    _explicitly_targets_runtime_artifact as _explicitly_targets_runtime_artifact,
    _git_raw as _git_raw,
    _hash_path as _hash_path,
    _is_ephemeral_build_or_test_artifact as _is_ephemeral_build_or_test_artifact,
    _is_gitignored_on_host as _is_gitignored_on_host,
    _is_ignored_build_artifact as _is_ignored_build_artifact,
    _is_referenced_by_files_or_prompt as _is_referenced_by_files_or_prompt,
    _is_tracked_in_git as _is_tracked_in_git,
    _is_untracked_runtime_artifact_name as _is_untracked_runtime_artifact_name,
    _matches_allowed_paths as _matches_allowed_paths,
    _matches_file_filter as _matches_file_filter,
    _normalize_readonly_go_build_acceptance as _normalize_readonly_go_build_acceptance,
    _run_git as _run_git,
    _snapshot_untracked_hashes as _snapshot_untracked_hashes,
    _split_diff_by_file as _split_diff_by_file,
    _strip_dot_slash as _strip_dot_slash,
    resolve_budget as resolve_budget,
    validate_acceptance as validate_acceptance,
    validate_allowed_paths as validate_allowed_paths,
    validate_plan as validate_plan,
)
from server.worker_prompt_builder import (
    QA_DIFF_AUDIT_SYSTEM as QA_DIFF_AUDIT_SYSTEM,
    QA_REVIEW_MAX_TOKENS as QA_REVIEW_MAX_TOKENS,
    _load_safe_workspace_files as _load_safe_workspace_files,
    build_qa_diff_audit_system as build_qa_diff_audit_system,
    escalate_after_failures as escalate_after_failures,
    is_quota_or_rate_limit_error as is_quota_or_rate_limit_error,
    parse_qa_audit_verdict as parse_qa_audit_verdict,
    render_worker_prompt as render_worker_prompt,
    select_candidate_files as select_candidate_files,
    worker_turn_failed as worker_turn_failed,
)
from server.ast_regression_guards import (
    _collect_init_self_writes as _collect_init_self_writes,
    _is_test_file_path as _is_test_file_path,
    _is_test_or_doc_path as _is_test_or_doc_path,
    _prompt_forbids_existing_test_updates as _prompt_forbids_existing_test_updates,
    _prompt_requests_existing_test_updates as _prompt_requests_existing_test_updates,
    _prompt_requests_test_authoring as _prompt_requests_test_authoring,
    _symbols_defined_in_baseline as _symbols_defined_in_baseline,
    _unexempted_stub_mutation_warnings as _unexempted_stub_mutation_warnings,
    detect_class_attribute_shadowing as detect_class_attribute_shadowing,
    detect_existing_test_stub_mutations as detect_existing_test_stub_mutations,
    find_missing_prompt_symbols as find_missing_prompt_symbols,
)
from server.test_discovery import (
    COMPILE_SMOKE_ENV_FAILURE_RE as COMPILE_SMOKE_ENV_FAILURE_RE,
    _CHANGED_FILE_ERROR_FILTER as _CHANGED_FILE_ERROR_FILTER,
    _JS_TS_EXTS as _JS_TS_EXTS,
    _JS_TS_TEST_RE as _JS_TS_TEST_RE,
    _build_workdir_exec_env as _build_workdir_exec_env,
    _collect_workspace_node_bin_dirs as _collect_workspace_node_bin_dirs,
    _detect_default_pytest_runner as _detect_default_pytest_runner,
    _detect_node_runner_info as _detect_node_runner_info,
    _list_tracked_repo_files as _list_tracked_repo_files,
    build_compile_smoke_commands as build_compile_smoke_commands,
    build_import_smoke_command as build_import_smoke_command,
    detect_project_test_runner as detect_project_test_runner,
    discover_related_test_commands as discover_related_test_commands,
)
from server.acceptance_runner import (
    _TEST_RUNNER_CMD_RE as _TEST_RUNNER_CMD_RE,
    _extract_failing_test_nodes_and_files as _extract_failing_test_nodes_and_files,
    _extract_pytest_runner_prefix as _extract_pytest_runner_prefix,
    _invalidate_python_bytecode_for_paths as _invalidate_python_bytecode_for_paths,
    _is_compiler_cli_config_or_help_failure as _is_compiler_cli_config_or_help_failure,
    _is_linter_or_static_check_cmd as _is_linter_or_static_check_cmd,
    _only_compiler_cli_config_failures as _only_compiler_cli_config_failures,
    _parse_test_counts as _parse_test_counts,
    _resolve_pytest_runner_for_subdir as _resolve_pytest_runner_for_subdir,
    _run_bounded_probe_cmd as _run_bounded_probe_cmd,
    evaluate_post_worker_baseline_failures as evaluate_post_worker_baseline_failures,
    probe_baseline_acceptance as probe_baseline_acceptance,
    remap_nonexistent_pytest_acceptance_cmds as remap_nonexistent_pytest_acceptance_cmds,
    run_acceptance as run_acceptance,
)
from server.job_apply_mixin import JobApplyMixin
from server.job_qa_mixin import JobQAMixin
from server.job_salvage_mixin import JobSalvageMixin

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
        index_env = {"GIT_INDEX_FILE": str(tmp_index_path)}
        _git_raw(
            ["-c", "core.quotePath=false", "add", "-N", "."],
            cwd=str(workspace_root),
            env_overrides=index_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        diff_res = _git_raw(
            ["-c", "core.quotePath=false", "diff", "--binary", "--no-renames", target_ref],
            cwd=str(workspace_root),
            env_overrides=index_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return diff_res.stdout if diff_res.returncode == 0 else ""
    finally:
        tmp_index_path.unlink(missing_ok=True)


class _DirTreeSnapshot(dict[str, bytes]):
    """Byte and metadata snapshot of a file or directory tree under .git."""

    def __init__(self) -> None:
        super().__init__()
        self.kind: str = "absent"
        self.root_mode: int | None = None
        self.link_target: str | None = None
        self.link_target_digest: str | None = None
        self.modes: dict[str, int] = {}
        self.subdirs: dict[str, int] = {}
        self.symlinks: dict[str, str] = {}
        self.unreadable: bool = False


def _snapshot_directory_tree(root: Path) -> dict[str, bytes]:
    """Capture a byte and mode snapshot of `root` (used to protect .git/hooks, .git/config, .git/info)."""
    snap = _DirTreeSnapshot()
    try:
        st = root.lstat()
    except FileNotFoundError:
        return snap
    except OSError:
        snap.unreadable = True
        return snap
    if stat.S_ISLNK(st.st_mode):
        from server.verify import _symlink_target_digest

        snap.kind = "symlink"
        try:
            snap.link_target = os.readlink(root)
            snap.link_target_digest = _symlink_target_digest(root)
            if snap.link_target_digest is None:
                snap.unreadable = True
        except OSError:
            snap.unreadable = True
        return snap
    if stat.S_ISREG(st.st_mode):
        snap.kind = "file"
        snap.root_mode = st.st_mode & 0o7777
        snap.modes[""] = snap.root_mode
        try:
            snap[""] = root.read_bytes()
        except OSError:
            snap.unreadable = True
        return snap
    if not stat.S_ISDIR(st.st_mode):
        snap.unreadable = True
        return snap
    snap.kind = "dir"
    snap.root_mode = st.st_mode & 0o7777

    def _on_walk_err(_err: OSError) -> None:
        snap.unreadable = True

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=_on_walk_err, followlinks=False):
        dp = Path(dirpath)
        dirnames.sort()
        filenames.sort()
        real_dirs: list[str] = []
        for d in dirnames:
            sub = dp / d
            rel_d = sub.relative_to(root).as_posix()
            try:
                 dst = sub.lstat()
            except OSError:
                snap.unreadable = True
                continue
            if stat.S_ISLNK(dst.st_mode):
                try:
                    snap.symlinks[rel_d] = os.readlink(sub)
                except OSError:
                    snap.unreadable = True
            elif stat.S_ISDIR(dst.st_mode):
                snap.subdirs[rel_d] = dst.st_mode & 0o7777
                real_dirs.append(d)
            else:
                snap.unreadable = True
        dirnames[:] = real_dirs
        for f in filenames:
            fp = dp / f
            rel_f = fp.relative_to(root).as_posix()
            try:
                fst = fp.lstat()
            except OSError:
                snap.unreadable = True
                continue
            if stat.S_ISLNK(fst.st_mode):
                try:
                    snap.symlinks[rel_f] = os.readlink(fp)
                except OSError:
                    snap.unreadable = True
            elif stat.S_ISREG(fst.st_mode):
                snap.modes[rel_f] = fst.st_mode & 0o7777
                try:
                    snap[rel_f] = fp.read_bytes()
                except OSError:
                    snap.unreadable = True
            else:
                snap.unreadable = True
    return snap


def _make_dir_owner_writable(d: Path) -> None:
    if not d.is_symlink() and d.is_dir():
        try:
            os.chmod(d, (d.lstat().st_mode & 0o7777) | 0o700)
        except OSError:
            pass


def _force_remove_tree_or_entry(p: Path) -> None:
    try:
        st = p.lstat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.S_ISLNK(st.st_mode):
        p.unlink(missing_ok=True)
        return
    if stat.S_ISDIR(st.st_mode):
        _make_dir_owner_writable(p)
        for dirpath, dirnames, filenames in os.walk(p, topdown=True, followlinks=False):
            dp = Path(dirpath)
            for d in dirnames:
                _make_dir_owner_writable(dp / d)
            for f in filenames:
                fp = dp / f
                if not fp.is_symlink() and fp.is_file():
                    try:
                        os.chmod(fp, 0o600)
                    except OSError:
                        pass
        shutil.rmtree(p, ignore_errors=True)
        return
    if stat.S_ISREG(st.st_mode):
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
    p.unlink(missing_ok=True)


def _restore_directory_tree(root: Path, snap: dict[str, bytes]) -> bool:
    """Verify `root` matches `snap`; if tampered (including planted symlinks or read-only modes), restore `root` and return True."""
    if root.parent.is_symlink():
        return True
    current = _snapshot_directory_tree(root)
    assert isinstance(current, _DirTreeSnapshot)
    if isinstance(snap, _DirTreeSnapshot):
        expected_kind = snap.kind
        expected_root_mode = snap.root_mode
        expected_link = snap.link_target
        expected_modes = snap.modes
        expected_subdirs = snap.subdirs
        expected_symlinks = snap.symlinks
        if (
            not snap.unreadable
            and not current.unreadable
            and current.kind == expected_kind
            and current.link_target == expected_link
            and (expected_kind != "symlink" or current.link_target_digest == snap.link_target_digest)
            and (expected_root_mode is None or current.root_mode == expected_root_mode)
            and dict(current) == dict(snap)
            and current.modes == expected_modes
            and current.subdirs == expected_subdirs
            and current.symlinks == expected_symlinks
        ):
            return False
    else:
        expected_kind = "file" if "" in snap else ("dir" if snap else "absent_or_empty")
        expected_root_mode = None
        expected_link = None
        expected_modes = {}
        expected_subdirs = {}
        expected_symlinks = {}
        if not current.unreadable and not current.symlinks:
            if expected_kind == "file" and current.kind == "file" and dict(current) == snap:
                return False
            if expected_kind == "dir" and current.kind == "dir" and dict(current) == snap:
                return False
            if expected_kind == "absent_or_empty" and (
                current.kind == "absent" or (current.kind == "dir" and not current)
            ):
                return False

    if root.parent.exists():
        _make_dir_owner_writable(root.parent)
    else:
        root.parent.mkdir(parents=True, exist_ok=True)

    if expected_kind == "absent":
        _force_remove_tree_or_entry(root)
        return True
    if expected_kind == "absent_or_empty":
        if root.is_dir() and not root.is_symlink():
            _make_dir_owner_writable(root)
            for child in list(root.iterdir()):
                _force_remove_tree_or_entry(child)
        else:
            _force_remove_tree_or_entry(root)
        return True
    if expected_kind == "symlink" and expected_link is not None:
        _force_remove_tree_or_entry(root)
        os.symlink(expected_link, root)
        return True
    if expected_kind == "file":
        _force_remove_tree_or_entry(root)
        root.write_bytes(snap[""])
        if expected_root_mode is not None:
            os.chmod(root, expected_root_mode)
        return True

    # expected_kind == "dir"
    if not root.is_dir() or root.is_symlink():
        _force_remove_tree_or_entry(root)
        root.mkdir(parents=True, exist_ok=True)
    _make_dir_owner_writable(root)
    for dirpath, dirnames, _filenames in os.walk(root, topdown=True, followlinks=False):
        dp = Path(dirpath)
        for d in dirnames:
            _make_dir_owner_writable(dp / d)

    needed_dirs: set[str] = set(expected_subdirs)
    for rel_key in (*snap.keys(), *expected_symlinks.keys()):
        if not rel_key:
            continue
        parts = Path(rel_key).parts
        for idx in range(1, len(parts)):
            needed_dirs.add(Path(*parts[:idx]).as_posix())

    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        dp = Path(dirpath)
        for name in filenames:
            fp = dp / name
            rel = fp.relative_to(root).as_posix()
            keep_file = rel in snap and fp.is_file() and not fp.is_symlink()
            keep_link = rel in expected_symlinks and fp.is_symlink() and os.readlink(fp) == expected_symlinks[rel]
            if not (keep_file or keep_link):
                _force_remove_tree_or_entry(fp)
        for name in dirnames:
            sub = dp / name
            rel = sub.relative_to(root).as_posix()
            if sub.is_symlink():
                if rel not in expected_symlinks or os.readlink(sub) != expected_symlinks[rel]:
                    sub.unlink(missing_ok=True)
            elif rel not in needed_dirs:
                _force_remove_tree_or_entry(sub)

    for drel in sorted(needed_dirs):
        dp = root / drel
        if dp.is_symlink() or (dp.exists() and not dp.is_dir()):
            _force_remove_tree_or_entry(dp)
        dp.mkdir(parents=True, exist_ok=True)
        _make_dir_owner_writable(dp)

    for srel, target_str in sorted(expected_symlinks.items()):
        sp = root / srel
        if not (sp.is_symlink() and os.readlink(sp) == target_str):
            _force_remove_tree_or_entry(sp)
            os.symlink(target_str, sp)

    for rel, content in sorted(snap.items()):
        if not rel:
            continue
        tp = root / rel
        if tp.is_symlink() or (tp.exists() and not tp.is_file()):
            _force_remove_tree_or_entry(tp)
        need_write = True
        if tp.is_file() and not tp.is_symlink():
            try:
                if tp.read_bytes() == content:
                    need_write = False
            except OSError:
                need_write = True
        if need_write:
            if tp.exists():
                try:
                    os.chmod(tp, (tp.lstat().st_mode & 0o7777) | 0o600)
                except OSError:
                    pass
            tp.write_bytes(content)
        if rel in expected_modes:
            os.chmod(tp, expected_modes[rel])

    for drel, dmode in sorted(expected_subdirs.items(), reverse=True):
        dp = root / drel
        if dp.is_dir() and not dp.is_symlink():
            try:
                os.chmod(dp, dmode)
            except OSError:
                pass
    if expected_root_mode is not None:
        try:
            os.chmod(root, expected_root_mode)
        except OSError:
            pass
    return True


_NM_TRANSIENT_CACHE_ENTRIES: frozenset[str] = frozenset(
    {
        ".vite",
        ".vite-temp",
        ".cache",
        ".tmp",
        ".eslintcache",
        ".parcel-cache",
        ".turbo",
        ".package-lock.json",
        ".modules.yaml",
        ".yarn-state.yml",
        ".pnp.cjs",
        ".pnp.loader.mjs",
        ".cc-agy-sparse-ws",
        ".cc-agy-ext-store",
    }
)

_NM_COPY_MAX_FILES: int = 10000
_NM_COPY_MAX_BYTES: int = 128 * 1024 * 1024

_NM_WALK_PRUNE_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        ".git",
        "dist",
        "build",
        ".next",
        ".nuxt",
        ".turbo",
        ".cache",
        ".parcel-cache",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".worktrees",
        ".cc-agy-data",
        ".cc-agy-review",
        ".ctx-session-reads",
    }
)


def _workspace_internal_symlink_rel(link_path: Path, host_resolved: Path) -> Path | None:
    """If `link_path` is a symlink resolving inside `host_resolved` outside `node_modules`, return its host-relative path."""
    if not link_path.is_symlink():
        return None
    try:
        raw_target = os.readlink(link_path)
        resolved = (link_path.parent / raw_target).resolve()
        rel_to_host = resolved.relative_to(host_resolved)
    except (OSError, ValueError):
        return None
    if "node_modules" in rel_to_host.parts:
        return None
    return rel_to_host


def _probe_nm_within_copy_budget(
    root_path: Path,
    *,
    max_files: int | None = None,
    max_bytes: int | None = None,
) -> bool:
    """Return True when `root_path` is small enough to materialize isolated copies in a worktree."""
    limit_files = _NM_COPY_MAX_FILES if max_files is None else max_files
    limit_bytes = _NM_COPY_MAX_BYTES if max_bytes is None else max_bytes
    if limit_files <= 0 or limit_bytes <= 0:
        return False
    files_seen = 0
    bytes_seen = 0
    root_str = str(root_path)
    try:
        for cur_r, sub_dirs, file_names in os.walk(root_str, topdown=True, followlinks=False):
            if cur_r == root_str:
                sub_dirs[:] = [d for d in sub_dirs if d not in _NM_TRANSIENT_CACHE_ENTRIES]
            files_seen += len(file_names) + len(sub_dirs)
            if files_seen > limit_files:
                return False
            for fn in file_names:
                try:
                    st = os.lstat(os.path.join(cur_r, fn))
                    bytes_seen += int(st.st_size)
                    if bytes_seen > limit_bytes:
                        return False
                except OSError:
                    continue
    except OSError:
        return False
    return True


def _try_cow_clone_dir(src_dir: Path, dst_dir: Path) -> bool:
    """Attempt an O(1) copy-on-write clone (`cp -cR` on APFS, `cp -a --reflink=always` on Btrfs/XFS)."""
    if os.name != "posix":
        return False
    cmd = (
        ["cp", "-cR", str(src_dir), str(dst_dir)]
        if sys.platform == "darwin"
        else ["cp", "-a", "--reflink=always", str(src_dir), str(dst_dir)]
    )
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=5.0, check=False)
        if res.returncode == 0 and dst_dir.exists():
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    shutil.rmtree(dst_dir, ignore_errors=True)
    return False


def _fs_supports_cow_between(src_dir: Path, dst_dir: Path) -> bool:
    """Probe whether O(1) CoW cloning works from `src_dir` into `dst_dir`."""
    if os.name != "posix":
        return False
    probe_src: Path | None = None
    try:
        for entry in os.scandir(src_dir):
            if entry.name in _NM_TRANSIENT_CACHE_ENTRIES:
                continue
            if entry.is_file(follow_symlinks=False):
                probe_src = Path(entry.path)
                break
    except OSError:
        return False
    if probe_src is None:
        return False
    probe_dst = dst_dir / ".cc-agy-cow-probe"
    ok = _try_cow_clone_dir(probe_src, probe_dst)
    try:
        probe_dst.unlink(missing_ok=True)
    except OSError:
        shutil.rmtree(probe_dst, ignore_errors=True)
    return ok


def _copy_isolated_pkg_entry(src_path: Path, dst_path: Path, *, try_cow: bool = True) -> bool:
    """Materialize an isolated copy of `src_path` at `dst_path` (CoW clone when supported)."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.is_dir() and not src_path.is_symlink():
        if try_cow and _try_cow_clone_dir(src_path, dst_path):
            return True
        shutil.copytree(src_path, dst_path, symlinks=True, ignore_dangling_symlinks=True)
        return True
    if src_path.is_file() and not src_path.is_symlink():
        shutil.copy2(src_path, dst_path)
        return True
    return False


def _snapshot_host_nm_metadata(
    host_root: Path,
    *,
    max_entries: int = 150000,
) -> dict[str, tuple[int, int, int]] | None:
    """Snapshot `{rel_path: (st_mode, st_size, st_mtime_ns)}` across `host_root/**/node_modules`."""
    if not host_root.is_dir():
        return {}
    host_resolved = host_root.resolve()
    snap: dict[str, tuple[int, int, int]] = {}
    for cur_root, dirs, _files in os.walk(str(host_resolved), topdown=True, followlinks=False):
        cur_path = Path(cur_root)
        has_nm = "node_modules" in dirs
        dirs[:] = sorted(
            d for d in dirs if d not in _NM_WALK_PRUNE_DIRS and not d.startswith(".cc-agy-")
        )
        if not has_nm:
            continue
        host_nm = cur_path / "node_modules"
        if host_nm.is_symlink() or not host_nm.is_dir():
            continue
        nm_str = str(host_nm)
        for nm_r, nm_dirs, nm_files in os.walk(nm_str, topdown=True, followlinks=False):
            if nm_r == nm_str:
                nm_dirs[:] = sorted(d for d in nm_dirs if d not in _NM_TRANSIENT_CACHE_ENTRIES)
            else:
                nm_dirs.sort()
            for name in (*nm_dirs, *sorted(nm_files)):
                full_p = os.path.join(nm_r, name)
                try:
                    st = os.lstat(full_p)
                    rel_k = os.path.relpath(full_p, str(host_resolved))
                    snap[rel_k] = (int(st.st_mode), int(st.st_size), int(st.st_mtime_ns))
                except OSError:
                    continue
                if len(snap) > max_entries:
                    return None
    return snap


def _verify_host_nm_metadata_unchanged(
    host_root: Path,
    snapshot: dict[str, tuple[int, int, int]] | None,
) -> bool:
    """Return True when `host_root/**/node_modules` matches `snapshot` byte-size/mode/mtime."""
    if snapshot is None:
        return True
    after = _snapshot_host_nm_metadata(host_root, max_entries=max(150000, len(snapshot) + 100))
    if after is None:
        return False
    return after == snapshot


def _nm_links_contain_host_symlinks(created_dirs: list[Path], host_root: Path) -> bool:
    """Return True if any entry inside `created_dirs` is a symlink resolving into `host_root`."""
    if not created_dirs or not host_root.is_dir():
        return False
    host_resolved = host_root.resolve()
    for nm_dir in created_dirs:
        if not nm_dir.is_dir():
            continue
        try:
            for entry in os.scandir(nm_dir):
                if entry.name.startswith(".cc-agy-"):
                    continue
                ep = Path(entry.path)
                if ep.is_symlink():
                    try:
                        if ep.resolve().is_relative_to(host_resolved):
                            return True
                    except OSError:
                        continue
                elif entry.name.startswith("@") and entry.is_dir(follow_symlinks=False):
                    for sub in os.scandir(ep):
                        sp = Path(sub.path)
                        if sp.is_symlink():
                            try:
                                if sp.resolve().is_relative_to(host_resolved):
                                    return True
                            except OSError:
                                continue
        except OSError:
            continue
    return False


def _iter_pnpm_nm_symlinks(nm_dir: Path) -> list[Path]:
    """Yield direct package symlinks inside `.pnpm/<pkg>/node_modules` (including `@scope/*`)."""
    out: list[Path] = []
    try:
        for entry in os.scandir(nm_dir):
            ep = Path(entry.path)
            if entry.is_symlink():
                out.append(ep)
            elif entry.name.startswith("@") and entry.is_dir(follow_symlinks=False):
                for sub in os.scandir(ep):
                    if sub.is_symlink():
                        out.append(Path(sub.path))
    except OSError:
        pass
    return out


def _copy_pnpm_virtual_store_closure(
    pnpm_nm_anc: Path,
    ext_pnpm_root: Path,
    *,
    fs_cow: bool,
    allow_shallow_symlinks: bool,
    copy_budget: list[int] | None = None,
) -> bool:
    """Materialize `pnpm_nm_anc` plus its transitive sibling `.pnpm/<dep>/node_modules` closure."""
    pnpm_store_root = pnpm_nm_anc.parent.parent
    if pnpm_store_root.name != ".pnpm" or not pnpm_nm_anc.is_dir():
        return False
    visited: set[Path] = set()
    to_copy: list[Path] = []
    queue: list[Path] = [pnpm_nm_anc]
    budget = copy_budget if (isinstance(copy_budget, list) and len(copy_budget) >= 2) else [0, 0]
    start_files, start_bytes = int(budget[0]), int(budget[1])
    while queue:
        cur_nm = queue.pop(0)
        if cur_nm in visited:
            continue
        visited.add(cur_nm)
        dst_cur_nm = ext_pnpm_root / cur_nm.parent.name / "node_modules"
        if not dst_cur_nm.exists():
            if not fs_cow:
                cur_str = str(cur_nm)
                try:
                    for cr, sdirs, fnames in os.walk(cur_str, topdown=True, followlinks=False):
                        budget[0] += len(fnames) + len(sdirs)
                        if budget[0] > _NM_COPY_MAX_FILES:
                            budget[0], budget[1] = start_files, start_bytes
                            return False
                        for fn in fnames:
                            try:
                                budget[1] += int(os.lstat(os.path.join(cr, fn)).st_size)
                                if budget[1] > _NM_COPY_MAX_BYTES:
                                    budget[0], budget[1] = start_files, start_bytes
                                    return False
                            except OSError:
                                continue
                except OSError:
                    budget[0], budget[1] = start_files, start_bytes
                    return False
            to_copy.append(cur_nm)
        for s_link in _iter_pnpm_nm_symlinks(cur_nm):
            try:
                s_res = (s_link.parent / os.readlink(s_link)).resolve()
            except OSError:
                continue
            if not s_res.exists():
                continue
            if s_res.is_relative_to(pnpm_store_root):
                for anc in s_res.parents:
                    if anc.name == "node_modules" and anc.parent.parent == pnpm_store_root:
                        if anc not in visited:
                            queue.append(anc)
                        break

    for cur_nm in to_copy:
        dst_cur_nm = ext_pnpm_root / cur_nm.parent.name / "node_modules"
        if not dst_cur_nm.exists():
            _copy_isolated_pkg_entry(cur_nm, dst_cur_nm, try_cow=fs_cow)
    for cur_nm in to_copy:
        dst_cur_nm = ext_pnpm_root / cur_nm.parent.name / "node_modules"
        for dst_link in _iter_pnpm_nm_symlinks(dst_cur_nm):
            if dst_link.exists():
                continue
            rel_in_nm = dst_link.relative_to(dst_cur_nm)
            src_link = cur_nm / rel_in_nm
            try:
                s_res = (src_link.parent / os.readlink(src_link)).resolve()
            except OSError:
                continue
            if not s_res.exists():
                continue
            dst_link.unlink(missing_ok=True)
            if fs_cow or _probe_nm_within_copy_budget(s_res):
                _copy_isolated_pkg_entry(s_res, dst_link, try_cow=fs_cow)
            elif allow_shallow_symlinks:
                os.symlink(str(s_res), str(dst_link))
    return True


def link_host_node_modules_farm(
    host_root: Path,
    wt_dir: Path,
    *,
    allow_shallow_symlinks: bool = True,
) -> list[Path]:
    """Populate isolated `node_modules` directories in `wt_dir` from `host_root`.

    Uses O(1) CoW clones (`cp -cR` / `cp --reflink=always`) or bounded isolated package copies
    (`<= 10,000` entries and `<= 128 MiB`) so worker writes or `rm -rf` inside `wt_dir/node_modules`
    cannot touch `host_root`. On oversized non-CoW trees, skips linking when
    `allow_shallow_symlinks=False` (`agy` CLI `--dangerously-skip-permissions` workers) or falls
    back to shallow package symlinks guarded by Straitjacket + post-run host integrity checks.
    """
    if not host_root.is_dir() or not wt_dir.is_dir():
        return []
    host_resolved = host_root.resolve()
    wt_resolved = wt_dir.resolve()
    if host_resolved == wt_resolved:
        return []

    created_dirs: list[Path] = []
    cumulative_pnpm_budget: list[int] = [0, 0]
    for cur_root, dirs, _files in os.walk(str(host_resolved), topdown=True, followlinks=False):
        cur_path = Path(cur_root)
        has_nm = "node_modules" in dirs
        dirs[:] = sorted(
            d for d in dirs if d not in _NM_WALK_PRUNE_DIRS and not d.startswith(".cc-agy-")
        )
        if not has_nm:
            continue
        host_nm = cur_path / "node_modules"
        if host_nm.is_symlink() or not host_nm.is_dir():
            continue
        try:
            rel_parent = cur_path.relative_to(host_resolved)
        except ValueError:
            continue
        wt_parent = wt_resolved if rel_parent == Path(".") else (wt_resolved / rel_parent)
        wt_nm = wt_parent / "node_modules"
        if not wt_parent.is_dir() or wt_nm.exists() or wt_nm.is_symlink():
            continue

        within_budget = _probe_nm_within_copy_budget(host_nm)
        try:
            wt_nm.mkdir(parents=False, exist_ok=False)
        except OSError:
            continue
        fs_cow = _fs_supports_cow_between(host_nm, wt_nm)
        isolate_copies = within_budget or fs_cow
        if not isolate_copies and not allow_shallow_symlinks:
            shutil.rmtree(wt_nm, ignore_errors=True)
            continue

        created_dirs.append(wt_nm)
        host_nm_resolved = host_nm.resolve()

        def _populate_entry(src_p: Path, dst_p: Path, dst_parent: Path) -> None:
            ws_rel = _workspace_internal_symlink_rel(src_p, host_resolved)
            if ws_rel is not None:
                wt_target = wt_resolved / ws_rel
                if wt_target.exists() or wt_target.is_symlink():
                    os.symlink(os.path.relpath(wt_target, dst_parent), str(dst_p))
                    return
                resolved_ws_pkg = src_p.resolve()
                if resolved_ws_pkg.exists() and (isolate_copies or _probe_nm_within_copy_budget(resolved_ws_pkg)):
                    sparse_copy = wt_nm / ".cc-agy-sparse-ws" / ws_rel
                    if not sparse_copy.exists():
                        _copy_isolated_pkg_entry(resolved_ws_pkg, sparse_copy, try_cow=fs_cow)
                    os.symlink(os.path.relpath(sparse_copy, dst_parent), str(dst_p))
                    return
                if allow_shallow_symlinks and resolved_ws_pkg.exists():
                    os.symlink(str(resolved_ws_pkg), str(dst_p))
                return

            if src_p.is_symlink():
                raw_target = os.readlink(src_p)
                resolved_target = (src_p.parent / raw_target).resolve()
                if resolved_target.is_relative_to(host_nm_resolved):
                    if isolate_copies:
                        rel_inside_nm = resolved_target.relative_to(host_nm_resolved)
                        os.symlink(os.path.relpath(wt_nm / rel_inside_nm, dst_parent), str(dst_p))
                    elif allow_shallow_symlinks:
                        os.symlink(str(src_p), str(dst_p))
                    return
                if resolved_target.is_relative_to(host_resolved):
                    wt_equiv = wt_resolved / resolved_target.relative_to(host_resolved)
                    if wt_equiv.exists() or wt_equiv.is_symlink():
                        os.symlink(os.path.relpath(wt_equiv, dst_parent), str(dst_p))
                        return
                if "/.pnpm/" in resolved_target.as_posix() and resolved_target.exists():
                    pnpm_nm_anc: Path | None = None
                    for anc in resolved_target.parents:
                        if anc.name == "node_modules" and anc.parent.parent.name == ".pnpm":
                            pnpm_nm_anc = anc
                            break
                    ext_pnpm_root = wt_nm / ".cc-agy-ext-store" / ".pnpm"
                    if (
                        pnpm_nm_anc is not None
                        and _copy_pnpm_virtual_store_closure(
                            pnpm_nm_anc,
                            ext_pnpm_root,
                            fs_cow=fs_cow,
                            allow_shallow_symlinks=allow_shallow_symlinks,
                            copy_budget=cumulative_pnpm_budget,
                        )
                    ):
                        ext_pnpm_nm = ext_pnpm_root / pnpm_nm_anc.parent.name / "node_modules"
                        ext_pkg = ext_pnpm_nm / resolved_target.relative_to(pnpm_nm_anc)
                        os.symlink(os.path.relpath(ext_pkg, dst_parent), str(dst_p))
                        return
                    if allow_shallow_symlinks:
                        os.symlink(str(resolved_target), str(dst_p))
                        return
                    return
                if resolved_target.exists() and (isolate_copies or _probe_nm_within_copy_budget(resolved_target)):
                    ext_copy = wt_nm / ".cc-agy-ext-store" / dst_p.relative_to(wt_nm)
                    if not ext_copy.exists():
                        _copy_isolated_pkg_entry(resolved_target, ext_copy, try_cow=fs_cow)
                    os.symlink(os.path.relpath(ext_copy, dst_parent), str(dst_p))
                    return
                if allow_shallow_symlinks:
                    os.symlink(str(src_p), str(dst_p))
                return

            if isolate_copies:
                _copy_isolated_pkg_entry(src_p, dst_p, try_cow=fs_cow)
            elif allow_shallow_symlinks:
                os.symlink(str(src_p), str(dst_p))

        try:
            entries = sorted(os.scandir(host_nm), key=lambda e: e.name)
        except OSError:
            continue

        for entry in entries:
            name = entry.name
            if name in _NM_TRANSIENT_CACHE_ENTRIES:
                continue
            entry_path = Path(entry.path)
            if name == ".bin" and entry.is_dir(follow_symlinks=True):
                wt_bin = wt_nm / ".bin"
                try:
                    wt_bin.mkdir(parents=False, exist_ok=True)
                    bin_entries = sorted(os.scandir(entry_path), key=lambda e: e.name)
                except OSError:
                    continue
                for b_ent in bin_entries:
                    b_src = Path(b_ent.path)
                    b_dst = wt_bin / b_ent.name
                    try:
                        if b_src.is_symlink():
                            raw_b_target = os.readlink(b_src)
                            res_b_target = (b_src.parent / raw_b_target).resolve()
                            if res_b_target.is_relative_to(host_nm_resolved):
                                rel_in_nm = res_b_target.relative_to(host_nm_resolved)
                                os.symlink(os.path.relpath(wt_nm / rel_in_nm, wt_bin), str(b_dst))
                            elif not os.path.isabs(raw_b_target):
                                os.symlink(raw_b_target, str(b_dst))
                            elif res_b_target.is_file():
                                shutil.copy2(res_b_target, b_dst)
                        elif b_src.is_file():
                            shutil.copy2(b_src, b_dst)
                    except OSError:
                        continue
                continue
            if name.startswith("@") and entry.is_dir(follow_symlinks=False):
                wt_scope = wt_nm / name
                try:
                    wt_scope.mkdir(parents=False, exist_ok=True)
                    sub_entries = sorted(os.scandir(entry_path), key=lambda e: e.name)
                except OSError:
                    continue
                for sub in sub_entries:
                    if sub.name in _NM_TRANSIENT_CACHE_ENTRIES:
                        continue
                    try:
                        _populate_entry(Path(sub.path), wt_scope / sub.name, wt_scope)
                    except OSError:
                        continue
                continue

            try:
                _populate_entry(entry_path, wt_nm / name, wt_nm)
            except OSError:
                continue

    return created_dirs


def unlink_host_node_modules_farm(created_dirs: list[Path] | Path | None) -> None:
    """Remove shallow `node_modules` farm directories created in a worktree without touching host files."""
    if created_dirs is None:
        return
    items = [created_dirs] if isinstance(created_dirs, Path) else list(created_dirs)
    for d in sorted(items, key=lambda p: len(p.parts), reverse=True):
        try:
            if d.is_symlink():
                d.unlink(missing_ok=True)
            elif d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


class JobManager(JobApplyMixin, JobQAMixin, JobSalvageMixin):
    """Manages isolated worker clones, background execution, conflict-safe apply, and provenance."""

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        store: ReceiptStore | None = None,
        worker_backend: Callable[[dict[str, Any]], Any] | None = None,
        max_workers: int = 4,
        ask_backend: Any = None,
    ) -> None:
        self.workspace_root = resolve_workspace_root(workspace_root)
        self.store = store or ReceiptStore(workspace_root=self.workspace_root)
        self.worker_backend = worker_backend
        self.ask_backend = ask_backend
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cc-agy-job")
        self._lock = threading.Lock()
        self._apply_lock = threading.Lock()
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._cancelled: set[str] = set()
        self._cancel_hooks: dict[str, Callable[[], None]] = {}
        self._shutting_down: bool = False
        self._raw_projects: dict[str, str] = {}
        self._raw_api_keys: dict[str, str] = {}
        self._wait_sliced_auto_apply: dict[str, bool | None] = {}
        self._qa_bg_threads: list[threading.Thread] = []
        self._sparse_worktrees: dict[str, int] = {}
        self._in_place_job: str | None = None
        self._instance_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._owned_job_ids: set[str] = set()
        self._created_at = time.monotonic()
        self._created_at_epoch = time.time()
        self._ensure_baseline_recorded()

    def _resolve_wait_slice_seconds(
        self,
        timeout_seconds: float,
        *,
        elapsed_seconds: float = 0.0,
        allow_retry_grace: bool = False,
    ) -> float:
        """Clamp synchronous blocking wait slices (default <=240s, <=275s on auto-retry) so the host 5m prompt cache never expires."""
        full_wait = max(0.5, float(timeout_seconds) + 30.0)
        raw_slice = os.environ.get("CC_AGY_MAX_WAIT_SLICE_SECONDS", "").strip()
        if raw_slice:
            try:
                max_slice = float(raw_slice)
            except ValueError:
                max_slice = DEFAULT_MAX_WAIT_SLICE_SECONDS
            if max_slice <= 0:
                return full_wait
        elif self._is_bounded_lead_session() and self.worker_backend is None:
            max_slice = DEFAULT_MAX_WAIT_SLICE_SECONDS
        else:
            return full_wait
        elapsed_clamped = max(0.0, float(elapsed_seconds))
        if allow_retry_grace and not raw_slice and max_slice >= 200.0:
            max_slice = min(275.0, max(max_slice, elapsed_clamped + 45.0))
        remaining_slice = max(0.05, max_slice - elapsed_clamped)
        return min(full_wait, remaining_slice)

    def _is_bounded_lead_session(self) -> bool:
        if env_flag("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE") is True:
            return True
        if self.worker_backend is not None:
            return False
        if is_strict_mode_enabled():
            return True
        return bool(os.environ.get("CC_AGY_FORCE_WORKER_THINKING_LEVEL", "").strip())

    def _should_run_default_acceptance(self) -> bool:
        if os.environ.get("CC_AGY_AUTO_REGRESSION_TESTS", "1").strip().lower() in ("0", "false", "no"):
            return False
        raw = os.environ.get("CC_AGY_AUTO_DEFAULT_ACCEPTANCE", "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        return self.worker_backend is None

    def _resolve_escalation_target_model(self, current_model: str) -> str | None:
        """Return the escalated worker model (default `DEFAULT_ESCALATION_MODEL`, Carbon zero-thinking) when escalating from Flash, or None."""
        if "flash" not in (current_model or "").lower():
            return None
        raw_esc = os.environ.get("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "").strip()
        if raw_esc.lower() in ("0", "false", "no", "off"):
            return None
        if raw_esc:
            target = (
                DEFAULT_ESCALATION_MODEL
                if raw_esc.lower() in ("1", "true", "yes", "on")
                else raw_esc
            )
        elif resolve_cost_profile() == "economy":
            return None
        elif self._is_bounded_lead_session() or self.worker_backend is None:
            target = DEFAULT_ESCALATION_MODEL
        else:
            return None
        return target if target != current_model else None

    def _same_model_retry_limit(self) -> int:
        """Plugin-side retries on the same worker model before handing a failed turn back to the Lead.

        Defaults to 1 in the economy profile (which has no Argon escalation) and 0 elsewhere;
        `CC_AGY_SAME_MODEL_RETRIES` overrides it.
        """
        raw = os.environ.get("CC_AGY_SAME_MODEL_RETRIES", "").strip()
        if raw and not raw.startswith("${"):
            try:
                return max(0, int(raw))
            except ValueError:
                pass
        return 1 if resolve_cost_profile() == "economy" else 0

    def _session_deadline_budget(self) -> tuple[float | None, float, float]:
        """Return `(session_deadline, session_elapsed, remaining_worker_budget)` when `CC_AGY_SESSION_DEADLINE_SECONDS` is set."""
        raw_deadline = os.environ.get("CC_AGY_SESSION_DEADLINE_SECONDS", "").strip()
        if not raw_deadline:
            return None, 0.0, float("inf")
        try:
            session_deadline = float(raw_deadline)
        except ValueError:
            session_deadline = 1500.0
        if session_deadline <= 0:
            return None, 0.0, float("inf")
        raw_reserve = os.environ.get("CC_AGY_DEADLINE_RESERVE_SECONDS", "").strip()
        try:
            reserve_sec = max(15.0, float(raw_reserve)) if raw_reserve else 120.0
        except ValueError:
            reserve_sec = 120.0
        elapsed = max(0.0, time.monotonic() - self._created_at)
        remaining_worker = (session_deadline - reserve_sec) - elapsed
        return session_deadline, elapsed, remaining_worker

    def _same_model_retry_reason(self, receipt: dict[str, Any]) -> str | None:
        """Return why a finished worker turn should get a plugin-side same-model retry, or None.

        Only voluntary status='ok' turns using < 60% of the tool-call budget with failed acceptance checks
        or vacuous acceptance qualify. Checkpoints (MAX_TOOL_CALLS_EXCEEDED), high-call-count slices,
        compiler CLI config/help errors alongside passing test runners, errors, timeouts, quota hits,
        questions, scope violations, spec conflicts, and Lead escalations are left to the Lead.
        """
        limit = self._same_model_retry_limit()
        if limit <= 0:
            return None
        used = sum(1 for s in (receipt.get("prompt_sources") or []) if s == "plugin_retry")
        if used >= limit:
            return None
        if (
            receipt.get("escalate_to_lead")
            or (receipt.get("escalation") or {}).get("escalate_to_lead")
            or receipt.get("applied")
        ):
            return None
        session_deadline, session_elapsed, remaining_worker = self._session_deadline_budget()
        if session_deadline is not None and (
            session_elapsed > 0.55 * session_deadline or remaining_worker < 180.0
        ):
            return None
        autonomous_solo = env_flag("CC_AGY_AUTONOMOUS_SOLO") is True
        status = str(receipt.get("status") or "")
        if autonomous_solo and status == "checkpoint":
            if (
                receipt.get("acceptance_passed") is False
                or receipt.get("vacuous_acceptance")
                or not receipt.get("files_changed")
            ):
                return "tool_call_checkpoint"
        if status != "ok":
            return None
        if not receipt.get("files_changed"):
            if autonomous_solo and receipt.get("vacuous_acceptance"):
                return "vacuous_acceptance"
            return None
        tool_calls_used = int(
            receipt.get("tool_call_count") or len(receipt.get("tool_calls") or [])
        )
        max_tool_calls = int((receipt.get("budget") or {}).get("max_tool_calls") or 100)
        call_cap = min(60, max(1, int(0.6 * max_tool_calls)))
        if not autonomous_solo and tool_calls_used >= call_cap:
            return None
        acc_results = [
            r for r in (receipt.get("acceptance_results") or []) if isinstance(r, dict)
        ]
        if acc_results and _only_compiler_cli_config_failures(acc_results):
            return None
        if receipt.get("acceptance_passed") is False:
            return "acceptance_failed"
        if receipt.get("vacuous_acceptance"):
            return "vacuous_acceptance"
        return None

    def _session_since_epoch(self) -> float:
        session_since = self._created_at_epoch - 1.0
        try:
            marker = self.store.data_dir / ".lead_session_started"
            if marker.exists():
                session_since = marker.stat().st_mtime
                try:
                    m_obj = json.loads(marker.read_text(encoding="utf-8"))
                    if isinstance(m_obj, dict) and m_obj.get("ts"):
                        session_since = float(m_obj["ts"])
                except (OSError, ValueError, TypeError):
                    pass
        except OSError:
            pass
        return session_since

    def _session_has_completed_delegate_job(self) -> bool:
        """Return True if this active session already has a finished prior delegate job."""
        session_since = self._session_since_epoch()
        for r in self.store.list_receipts(workspace_root=self.workspace_root):
            if r.get("kind", "delegate") != "delegate":
                continue
            r_started = float(r.get("started_at_epoch") or 0.0)
            if r_started > 0.0 and r_started < session_since:
                continue
            st = str(r.get("status") or "")
            if st and st not in ("running", "cancelled"):
                return True
        return False

    def _should_escalate_new_delegate_start(
        self,
        prompt: str,
        *,
        allowed_paths: list[str] | None = None,
        context_files: list[str] | None = None,
        spec_files: list[str] | None = None,
        acceptance: list[str] | None = None,
    ) -> bool:
        """Return True when a new `delegate_start` is a retry or follow-up fix on a prior session job,
        and False when it is an independent task after cleanly applied work."""
        session_since = self._session_since_epoch()
        session_jobs: list[dict[str, Any]] = []
        for r in self.store.list_receipts(workspace_root=self.workspace_root):
            if r.get("kind", "delegate") != "delegate":
                continue
            r_started = float(r.get("started_at_epoch") or 0.0)
            if r_started > 0.0 and r_started < session_since:
                continue
            st = str(r.get("status") or "")
            if st and st not in ("running", "cancelled"):
                session_jobs.append(r)

        if not session_jobs:
            return False

        def _job_failed_or_unapplied(r: dict[str, Any]) -> bool:
            st = str(r.get("status") or "")
            return (
                st != "ok"
                or r.get("acceptance_passed") is False
                or bool(r.get("reverted"))
                or not bool(r.get("applied"))
            )

        prior_paths: list[str] = []
        for r in session_jobs:
            for key in ("files_changed", "applied_files", "allowed_paths"):
                for p in r.get(key) or []:
                    norm_p = str(p or "").replace("\\", "/").strip("/")
                    if norm_p and norm_p not in prior_paths:
                        prior_paths.append(norm_p)

        explicit_targets: list[str] = []
        for seq in (allowed_paths, context_files, spec_files):
            for p in seq or []:
                norm_p = str(p or "").replace("\\", "/").strip("/")
                if norm_p and norm_p not in explicit_targets:
                    explicit_targets.append(norm_p)

        mentioned_targets: list[str] = list(explicit_targets)
        runner_names = {"pytest.py", "manage.py", "runtests.py", "run_tests.py", "setup.py"}
        scan_text = "\n".join([prompt or "", *(acceptance or [])])
        for tok in re.findall(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8}", scan_text):
            norm_tok = tok.strip(".,:;()[]{}'\"").replace("\\", "/").strip("/")
            if not norm_tok or Path(norm_tok).name in runner_names or ".." in Path(norm_tok).parts:
                continue
            if norm_tok in prior_paths or (self.workspace_root / norm_tok).exists():
                if norm_tok not in mentioned_targets:
                    mentioned_targets.append(norm_tok)

        if mentioned_targets and prior_paths:
            overlaps_prior = any(
                _matches_allowed_paths(nt, prior_paths) or _matches_allowed_paths(pp, mentioned_targets)
                for nt in mentioned_targets
                for pp in prior_paths
            )
            if overlaps_prior:
                return True

        followup_intent_re = re.compile(
            r"\b(?:fix\s+edge\s+case|post-apply\s+review|regression|still\s+(?:fails?|broken)|"
            r"missed\s+(?:item|requirement|case)|follow[- ]?up\s+fix|previous\s+job|prior\s+attempt)\b",
            re.I,
        )
        if followup_intent_re.search(prompt or ""):
            return True

        norm_new_acc = {c.strip() for c in (acceptance or []) if c.strip()}
        acc_overlaps_prior = False
        if norm_new_acc:
            for r in session_jobs:
                norm_old_acc = {str(c).strip() for c in (r.get("acceptance") or []) if str(c).strip()}
                if norm_new_acc & norm_old_acc:
                    acc_overlaps_prior = True
                    break
        if acc_overlaps_prior:
            return True

        # Check whether a new human user prompt arrived after prior session jobs finished.
        up_path = self.store.data_dir / ".last_user_prompt.json"
        if up_path.is_file():
            try:
                up_obj = json.loads(up_path.read_text(encoding="utf-8"))
                if isinstance(up_obj, dict):
                    up_ts = float(up_obj.get("ts") or 0.0)
                    if up_ts >= session_since:
                        active_turn_jobs = [
                            r
                            for r in session_jobs
                            if max(
                                float(r.get("applied_at_epoch") or 0.0),
                                float(r.get("finished_at_epoch") or 0.0),
                                float(r.get("started_at_epoch") or 0.0),
                            )
                            >= up_ts - 0.02
                        ]
                        if not active_turn_jobs:
                            # Brand-new user prompt after all prior jobs, with disjoint targets & no fix intent -> stay on Flash!
                            return False
                        if mentioned_targets and prior_paths:
                            return False
                        return True
            except (OSError, ValueError, TypeError, KeyError):
                pass

        # When called directly without hook state: disjoint explicit/mentioned targets stay on Flash even if an unrelated job failed.
        if mentioned_targets and prior_paths:
            return False

        return any(_job_failed_or_unapplied(r) for r in session_jobs)

    def _clamp_worker_timeout(self, timeout_seconds: float) -> float:
        """Clamp worker turn timeout against an explicit session deadline (reserving headroom for review & apply).

        Only `CC_AGY_SESSION_DEADLINE_SECONDS` enables the clamp. Session deadlines belong to the
        harness that owns the wall clock, not to strict mode, so interactive sessions are never clamped.
        """
        t_val = max(0.1, float(timeout_seconds))
        raw_deadline = os.environ.get("CC_AGY_SESSION_DEADLINE_SECONDS", "").strip()
        if not raw_deadline:
            return t_val
        try:
            session_deadline = float(raw_deadline)
        except ValueError:
            session_deadline = 1500.0
        if session_deadline <= 0:
            return t_val
        raw_reserve = os.environ.get("CC_AGY_DEADLINE_RESERVE_SECONDS", "").strip()
        try:
            reserve_sec = max(15.0, float(raw_reserve)) if raw_reserve else 120.0
        except ValueError:
            reserve_sec = 120.0
        elapsed = max(0.0, time.monotonic() - self._created_at)
        worker_budget = max(45.0, (session_deadline - reserve_sec) - elapsed)
        return min(t_val, worker_budget)

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

    def _select_sparse_worktree_entries(
        self,
        rel_entries: list[str],
        *,
        allowed_paths: list[str],
        context_files: list[str] | None = None,
        spec_files: list[str] | None = None,
    ) -> list[str]:
        """Select a self-contained subset of repository files for large monorepos when `allowed_paths` is scoped.

        Includes:
        1. All top-level manifest/configuration files (e.g. `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, `.gitignore`).
        2. All files matching `allowed_paths`, `context_files`, or `spec_files`, plus sibling files in their immediate directories.
        3. Ancestor package markers (`__init__.py`, `conftest.py`, `package.json`, `tsconfig*.json`, `Cargo.toml`, `BUILD*`).
        4. Matching unit test files (`test_<stem>.*`, `<stem>_test.*`, `<stem>.test.*`, `<stem>.spec.*`).
        """
        seed_scopes = [
            p.replace("\\", "/").strip("/")
            for p in [*(allowed_paths or []), *(context_files or []), *(spec_files or [])]
            if str(p or "").strip()
        ]
        target_dirs: set[str] = set()
        ancestor_dirs: set[str] = {""}
        stems: set[str] = set()
        for sp in seed_scopes:
            sp_path = self.workspace_root / sp
            if sp_path.is_dir():
                target_dirs.add(sp)
                cur_parts = Path(sp).parts
            else:
                parent_str = Path(sp).parent.as_posix()
                if parent_str and parent_str != ".":
                    target_dirs.add(parent_str)
                cur_parts = Path(sp).parts[:-1]
                stem_s = Path(sp).stem
                if stem_s and stem_s not in ("__init__", "index", "mod", "main"):
                    stems.add(stem_s)
            acc = []
            for part in cur_parts:
                acc.append(part)
                ancestor_dirs.add("/".join(acc))

        test_names: set[str] = set()
        for st in stems:
            test_names.update(
                {
                    f"test_{st}.py",
                    f"{st}_test.py",
                    f"{st}_test.go",
                    f"{st}_test.rs",
                    *(f"{st}.test{ext}" for ext in _JS_TS_EXTS),
                    *(f"{st}.spec{ext}" for ext in _JS_TS_EXTS),
                }
            )
        anc_marker_names = {
            "__init__.py",
            "conftest.py",
            "pytest.ini",
            "pyproject.toml",
            "setup.cfg",
            "package.json",
            "tsconfig.json",
            "tsconfig.base.json",
            "Cargo.toml",
            "go.mod",
            "go.sum",
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "Makefile",
            "BUILD",
            "BUILD.bazel",
            ".gitignore",
        }
        selected: list[str] = []
        dir_siblings_cap: dict[str, int] = {}
        for rel in rel_entries:
            norm = rel.replace("\\", "/").strip("/")
            if not norm:
                continue
            if "/" not in norm:
                # Top-level config/manifest files
                selected.append(rel)
                continue
            if _matches_allowed_paths(norm, seed_scopes):
                selected.append(rel)
                st_m = Path(norm).stem
                if len(stems) < 24 and st_m and st_m not in ("__init__", "index"):
                    stems.add(st_m)
                    test_names.update({f"test_{st_m}.py", f"{st_m}_test.py", f"{st_m}_test.go"})
                continue
            parent_rel = Path(norm).parent.as_posix()
            bname = Path(norm).name
            if parent_rel in ancestor_dirs and (bname in anc_marker_names or bname.startswith("BUILD.")):
                selected.append(rel)
                continue
            if parent_rel in target_dirs and dir_siblings_cap.get(parent_rel, 0) < 200:
                dir_siblings_cap[parent_rel] = dir_siblings_cap.get(parent_rel, 0) + 1
                selected.append(rel)
                continue
            if bname in test_names or norm.startswith(("bin/test",)):
                selected.append(rel)

        # 5. Bounded 2-hop workspace import tracing so cross-package local imports are materialized.
        rel_entry_map: dict[str, str] = {}
        for r_item in rel_entries:
            n_item = r_item.replace("\\", "/").strip("/")
            if n_item and n_item not in rel_entry_map:
                rel_entry_map[n_item] = r_item
        selected_norms: set[str] = {r_sel.replace("\\", "/").strip("/") for r_sel in selected}

        def _add_dep_norm(dep_norm: str, next_frontier: list[str]) -> bool:
            if dep_norm not in rel_entry_map or dep_norm in selected_norms:
                return False
            selected_norms.add(dep_norm)
            selected.append(rel_entry_map[dep_norm])
            next_frontier.append(dep_norm)
            # Ensure ancestor package markers (__init__.py, package.json, BUILD*) are included for imported modules
            acc_p: list[str] = []
            for pt in Path(dep_norm).parts[:-1]:
                acc_p.append(pt)
                anc_dir = "/".join(acc_p)
                for mk in ("__init__.py", "conftest.py", "package.json", "tsconfig.json", "BUILD", "BUILD.bazel"):
                    mk_norm = f"{anc_dir}/{mk}"
                    if mk_norm in rel_entry_map and mk_norm not in selected_norms:
                        selected_norms.add(mk_norm)
                        selected.append(rel_entry_map[mk_norm])
            return True

        frontier = [
            n_s
            for n_s in selected_norms
            if n_s.endswith((".py", *_JS_TS_EXTS))
        ][:80]
        added_deps = 0
        js_rel_re = re.compile(r"""(?:from|import|require\s*\()\s*['"](\.[^'"]+)['"]""")
        for _hop in range(2):
            if not frontier or added_deps >= 200:
                break
            next_frontier: list[str] = []
            for src_norm in frontier[:80]:
                if added_deps >= 200:
                    break
                src_file = self.workspace_root / src_norm
                try:
                    if not src_file.is_file() or src_file.stat().st_size > 256_000:
                        continue
                    src_txt = src_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if src_norm.endswith(".py"):
                    try:
                        tree = ast.parse(src_txt)
                    except SyntaxError:
                        continue
                    src_parts = Path(src_norm).parts
                    for node in ast.walk(tree):
                        cand_mod_paths: list[str] = []
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                mod_slash = alias.name.replace(".", "/").strip("/")
                                if mod_slash:
                                    cand_mod_paths.extend([mod_slash, f"src/{mod_slash}"])
                        elif isinstance(node, ast.ImportFrom):
                            if node.level and node.level > 0:
                                if node.level <= len(src_parts):
                                    base_parts = list(src_parts[: len(src_parts) - node.level])
                                    if node.module:
                                        base_parts.extend(p for p in node.module.split(".") if p)
                                    base_slash = "/".join(base_parts)
                                    if base_slash:
                                        cand_mod_paths.append(base_slash)
                                    for alias in node.names:
                                        if alias.name != "*":
                                            sub_slash = "/".join([*base_parts, alias.name])
                                            if sub_slash:
                                                cand_mod_paths.append(sub_slash)
                            elif node.module:
                                mod_slash = node.module.replace(".", "/").strip("/")
                                if mod_slash:
                                    for pref in ("", "src/"):
                                        cand_mod_paths.append(f"{pref}{mod_slash}")
                                        for alias in node.names:
                                            if alias.name != "*":
                                                cand_mod_paths.append(f"{pref}{mod_slash}/{alias.name}")
                        for cm in cand_mod_paths:
                            for cand_file in (f"{cm}.py", f"{cm}/__init__.py"):
                                if _add_dep_norm(cand_file, next_frontier):
                                    added_deps += 1
                else:
                    parent_posix = Path(src_norm).parent.as_posix()
                    for m_js in js_rel_re.finditer(src_txt):
                        raw_spec = m_js.group(1)
                        joined = os.path.normpath(
                            f"{parent_posix}/{raw_spec}" if parent_posix and parent_posix != "." else raw_spec
                        ).replace("\\", "/")
                        if joined.startswith("..") or joined == ".":
                            continue
                        for sfx in ("", *_JS_TS_EXTS, *(f"/index{ext}" for ext in _JS_TS_EXTS)):
                            if _add_dep_norm(f"{joined}{sfx}", next_frontier):
                                added_deps += 1
                                break
            frontier = next_frontier
        return selected

    def _prepare_worktree(
        self,
        job_id: str,
        *,
        allowed_paths: list[str] | None = None,
        context_files: list[str] | None = None,
        spec_files: list[str] | None = None,
        media_files: list[str] | None = None,
    ) -> tuple[Path, str, dict[str, str | None]]:
        """Create a standalone isolated git repository copy for `job_id`.

        Uses `git ls-files -co --exclude-standard -z` so gitignored directories (like
        `node_modules/` or `.venv/`) are never copied into the worker tree, supports opt-in
        scoped sparse materialization on large monorepos when `allowed_paths` is set, and
        computes baseline file hashes on demand for changed files.
        """
        safe_id = validate_job_id(job_id)
        wt_dir = self.store.worktree_path(safe_id).resolve()
        if wt_dir.exists():
            shutil.rmtree(wt_dir, ignore_errors=True)
        wt_dir.mkdir(parents=True, exist_ok=True)

        ls_res = _git_raw(
            ["ls-files", "-co", "--exclude-standard", "-z"],
            cwd=str(self.workspace_root),
            capture_output=True,
            check=False,
        )
        sparse_used = False
        if ls_res.returncode == 0:
            rel_entries = [
                entry.decode("utf-8", errors="surrogateescape")
                for entry in ls_res.stdout.split(b"\x00")
                if entry
            ]
            sparse_flag = env_flag("CC_AGY_SPARSE_WORKTREE")
            sparse_threshold = _env_int("CC_AGY_SPARSE_THRESHOLD_FILES", 0)
            if (
                allowed_paths
                and sparse_flag is not False
                and (sparse_flag is True or (sparse_threshold > 0 and len(rel_entries) > sparse_threshold))
            ):
                rel_entries = self._select_sparse_worktree_entries(
                    rel_entries,
                    allowed_paths=allowed_paths,
                    context_files=context_files,
                    spec_files=spec_files,
                )
                sparse_used = True

            created_dirs: set[Path] = {wt_dir}
            copied_count = 0
            for rel in rel_entries:
                if rel.startswith((".git/", ".cc-agy-data/", ".worktrees/", ".ctx-session-reads/", ".cc-agy-review/", ".venv/", "venv/")) or rel in (".venv", "venv", ".ctx-session-reads", ".cc-agy-review"):
                    continue
                src_p = self.workspace_root / rel
                dst_p = wt_dir / rel
                dst_parent = dst_p.parent
                if src_p.is_symlink():
                    if dst_parent not in created_dirs:
                        dst_parent.mkdir(parents=True, exist_ok=True)
                        created_dirs.add(dst_parent)
                    os.symlink(os.readlink(src_p), dst_p)
                    copied_count += 1
                elif src_p.is_file():
                    if dst_parent not in created_dirs:
                        dst_parent.mkdir(parents=True, exist_ok=True)
                        created_dirs.add(dst_parent)
                    shutil.copy2(src_p, dst_p)
                    copied_count += 1
            with self._lock:
                if sparse_used:
                    self._sparse_worktrees[safe_id] = copied_count
                else:
                    self._sparse_worktrees.pop(safe_id, None)
            # Also copy any pre-existing gitignored `_version.py` / `__version__.py` / `*_version*.py`
            # or compiled Cython/C extension (`*.so`, `*.pyd`) files generated during editable install
            # so they are present in baseline_sha and importable inside the worktree.
            copied_ignored_rels: list[str] = []
            ign_ver_res = _git_raw(
                [
                    "ls-files",
                    "-o",
                    "-i",
                    "--exclude-standard",
                    "-z",
                    "--",
                    ":(glob)**/_version.py",
                    ":(glob)**/__version__.py",
                    ":(glob)**/*_version*.py",
                    ":(glob)**/*.so",
                    ":(glob)**/*.pyd",
                    "_version.py",
                    "__version__.py",
                ],
                cwd=str(self.workspace_root),
                capture_output=True,
                check=False,
            )
            if ign_ver_res.returncode == 0:
                for entry in ign_ver_res.stdout.split(b"\x00"):
                    if not entry:
                        continue
                    rel_v = entry.decode("utf-8", errors="surrogateescape")
                    if rel_v.startswith((".git/", ".cc-agy-data/", ".worktrees/", ".venv/", "venv/", "node_modules/", "build/", "dist/")):
                        continue
                    if sparse_used and allowed_paths and not _matches_allowed_paths(rel_v, allowed_paths) and "/" in rel_v:
                        continue
                    v_name = Path(rel_v).name
                    if (
                        v_name in ("_version.py", "__version__.py")
                        or ("_version" in v_name and v_name.endswith(".py"))
                        or v_name.endswith((".so", ".pyd"))
                    ):
                        src_v = self.workspace_root / rel_v
                        dst_v = wt_dir / rel_v
                        if src_v.is_file() and not src_v.is_symlink() and not dst_v.exists():
                            dst_v.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_v, dst_v)
                            copied_ignored_rels.append(rel_v)
        else:
            copied_ignored_rels = []
            shutil.rmtree(wt_dir, ignore_errors=True)
            shutil.copytree(
                self.workspace_root,
                wt_dir,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    ".git", ".cc-agy-data", ".worktrees", ".ctx-session-reads", ".cc-agy-review", "__pycache__", "node_modules", ".venv", "venv"
                ),
            )

        _run_git(["init", "-b", "main"], cwd=wt_dir, check=True)
        _run_git(["config", "user.email", "worker@cc-agy-plugin.local"], cwd=wt_dir, check=True)
        _run_git(["config", "user.name", "CC-AGY Worker"], cwd=wt_dir, check=True)
        host_objects_dir = self.workspace_root / ".git" / "objects"
        fast_seeded = False
        if ls_res.returncode == 0 and host_objects_dir.is_dir():
            try:
                alt_file = wt_dir / ".git" / "objects" / "info" / "alternates"
                alt_file.parent.mkdir(parents=True, exist_ok=True)
                alt_file.write_text(str(host_objects_dir.resolve()) + "\n", encoding="utf-8")
                # In sparse mode, NEVER read-tree the full host HEAD (which would mark omitted files as deleted!).
                if not sparse_used:
                    lsv_res = _git_raw(
                        ["ls-files", "-v", "-z"],
                        cwd=str(self.workspace_root),
                        capture_output=True,
                        check=False,
                    )
                    has_special_index_flags = (
                        lsv_res.returncode != 0
                        or any(e and not e.startswith(b"H ") for e in lsv_res.stdout.split(b"\x00"))
                    )
                    host_head_res = _run_git(["rev-parse", "--verify", "HEAD"], cwd=self.workspace_root, check=False)
                    host_head_sha = host_head_res.stdout.strip() if host_head_res.returncode == 0 else ""
                    if host_head_sha and not has_special_index_flags:
                        rt_res = _run_git(["read-tree", host_head_sha], cwd=wt_dir, check=False)
                        if rt_res.returncode == 0:
                            # Remove index entries that were excluded from worktree copy or are submodule gitlinks (160000).
                            wt_ls = _git_raw(
                                ["ls-files", "-s", "-z"],
                                cwd=str(wt_dir),
                                capture_output=True,
                                check=False,
                            )
                            if wt_ls.returncode == 0:
                                idx_remove: list[str] = []
                                for raw_s in wt_ls.stdout.split(b"\x00"):
                                    if not raw_s:
                                        continue
                                    s_line = raw_s.decode("utf-8", errors="surrogateescape")
                                    meta, sep, idx_path = s_line.partition("\t")
                                    if not sep or not idx_path:
                                        continue
                                    mode_str = meta.split(" ", 1)[0]
                                    path_parts = Path(idx_path).parts
                                    if (
                                        mode_str == "160000"
                                        or idx_path.startswith(
                                            (
                                                ".git/",
                                                ".cc-agy-data/",
                                                ".worktrees/",
                                                ".ctx-session-reads/",
                                                ".cc-agy-review/",
                                                ".venv/",
                                                "venv/",
                                            )
                                        )
                                        or idx_path in (".venv", "venv", ".ctx-session-reads", ".cc-agy-review")
                                        or "__pycache__" in path_parts
                                        or "node_modules" in path_parts
                                    ):
                                        idx_remove.append(idx_path)
                                if idx_remove:
                                    _run_git(
                                        ["rm", "--cached", "-r", "-q", "--ignore-unmatch", "--", *idx_remove],
                                        cwd=wt_dir,
                                        check=False,
                                    )
                            delta_res = _git_raw(
                                ["status", "--porcelain", "-z"],
                                cwd=str(self.workspace_root),
                                capture_output=True,
                                check=False,
                            )
                            if delta_res.returncode == 0:
                                raw_entries = [
                                    e.decode("utf-8", errors="surrogateescape")
                                    for e in delta_res.stdout.split(b"\x00")
                                    if e
                                ]
                                delta_paths: list[str] = []
                                for entry in raw_entries:
                                    if len(entry) > 3 and entry[2] == " ":
                                        delta_paths.append(entry[3:])
                                    elif entry and not entry.startswith(" "):
                                        delta_paths.append(entry)
                                delta_paths = [
                                    p for p in dict.fromkeys([*delta_paths, *copied_ignored_rels])
                                    if not p.startswith((".git/", ".cc-agy-data/", ".worktrees/", ".ctx-session-reads/", ".cc-agy-review/", ".venv/", "venv/"))
                                    and p not in (".venv", "venv", ".ctx-session-reads", ".cc-agy-review")
                                ]
                                if delta_paths:
                                    _run_git(["add", "-f", "-A", "--", *delta_paths], cwd=wt_dir, check=True)
                                fast_seeded = True
            except (OSError, subprocess.SubprocessError):
                fast_seeded = False
        untracked_media_rels: list[str] = []
        for rel_m in media_files or []:
            chk_m = _git_raw(
                ["ls-files", "--error-unmatch", "--", rel_m],
                cwd=str(self.workspace_root),
                capture_output=True,
                check=False,
            )
            if chk_m.returncode != 0:
                untracked_media_rels.append(rel_m)

        if not fast_seeded:
            # When populated from `git ls-files`, force-add (`-f`) so tracked files that also match `.gitignore`
            # patterns (e.g. `tests/test_apps/.env` in Flask) are committed to `baseline_sha`.
            _run_git(["add", "-f", "-A"] if ls_res.returncode == 0 else ["add", "-A"], cwd=wt_dir, check=True)
        if untracked_media_rels:
            _run_git(
                ["rm", "--cached", "-q", "--ignore-unmatch", "--", *untracked_media_rels],
                cwd=wt_dir,
                check=False,
            )
        baseline_sha = ""
        if fast_seeded:
            wt_tree = _run_git(["write-tree"], cwd=wt_dir, check=False)
            if wt_tree.returncode == 0 and wt_tree.stdout.strip():
                wt_commit = _run_git(
                    ["commit-tree", wt_tree.stdout.strip(), "-m", "cc-agy-baseline"],
                    cwd=wt_dir,
                    check=False,
                )
                if wt_commit.returncode == 0 and wt_commit.stdout.strip():
                    cand_sha = wt_commit.stdout.strip()
                    up_ref = _run_git(["update-ref", "HEAD", cand_sha], cwd=wt_dir, check=False)
                    if up_ref.returncode == 0:
                        baseline_sha = cand_sha
        if not baseline_sha:
            _run_git(["commit", "--allow-empty", "-m", "cc-agy-baseline"], cwd=wt_dir, check=True)
            sha_res = _run_git(["rev-parse", "HEAD"], cwd=wt_dir, check=True)
            baseline_sha = sha_res.stdout.strip()

        for rel_m in media_files or []:
            src_m = self.workspace_root / rel_m
            dst_m = wt_dir / rel_m
            if src_m.is_file() and not src_m.is_symlink() and not dst_m.exists():
                dst_m.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_m, dst_m)

        exclude_file = wt_dir / ".git" / "info" / "exclude"
        try:
            exclude_file.parent.mkdir(parents=True, exist_ok=True)
            existing_excl = exclude_file.read_text(encoding="utf-8", errors="replace") if exclude_file.exists() else ""
            extra_excl = "\n".join(
                [
                    ".venv",
                    ".venv/",
                    "node_modules",
                    "node_modules/",
                    ".pytest_cache/",
                    "__pycache__/",
                    ".mypy_cache/",
                    ".ruff_cache/",
                    ".ctx-session-reads",
                    ".ctx-session-reads/",
                    ".cc-agy-review",
                    ".cc-agy-review/",
                    *untracked_media_rels,
                ]
            )
            exclude_file.write_text(
                (existing_excl.rstrip("\n") + "\n" + extra_excl + "\n").lstrip("\n"),
                encoding="utf-8",
            )
            if untracked_media_rels:
                (wt_dir / ".git" / "info" / "exclude-media").write_text(
                    "\n".join(untracked_media_rels) + "\n",
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
        git_cwd = wt_dir if wt_dir.exists() else (self.workspace_root if self.workspace_root.exists() else None)
        if git_cwd is None:
            return hashes
        ref_sha = baseline_sha
        if git_cwd != wt_dir:
            chk = _git_raw(
                ["cat-file", "-e", f"{baseline_sha}^{{commit}}"],
                cwd=str(git_cwd),
                capture_output=True,
                check=False,
            )
            if chk.returncode != 0:
                ref_sha = "HEAD"
        for rel in files_changed:
            if rel in hashes:
                continue
            show_res = _git_raw(
                ["show", f"{ref_sha}:{rel}"],
                cwd=str(git_cwd),
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
        prompt_text: str = "",
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
        for cur_root, dirs, files in os.walk(str(wt_dir), topdown=True, followlinks=False):
            dirs[:] = [
                d
                for d in dirs
                if d != ".git" and d not in _EPHEMERAL_CACHE_DIRS and not d.endswith(".egg-info")
            ]
            cur_path = Path(cur_root)
            for name in list(dirs) + list(files):
                p = cur_path / name
                if not p.is_symlink():
                    continue
                rel = str(p.relative_to(wt_dir))
                if _is_ephemeral_build_or_test_artifact(rel, root=wt_dir):
                    continue
                try:
                    link_target = (p.parent / os.readlink(p)).resolve()
                    if not link_target.is_relative_to(wt_resolved):
                        symlink_escapes.append(rel)
                except OSError:
                    symlink_escapes.append(rel)

        _run_git(["reset", "-q", baseline_sha], cwd=wt_dir, check=False)
        tracked_res = _run_git(
            ["diff", "--name-only", "--no-renames", baseline_sha],
            cwd=wt_dir,
            check=False,
        )
        tracked_changed = [
            ln.strip()
            for ln in (tracked_res.stdout.splitlines() if tracked_res.returncode == 0 else [])
            if ln.strip()
        ]
        untracked_to_stage: list[str] = []
        excluded_runtime_artifacts: list[str] = []
        preserved_runtime_artifacts: list[str] = []
        others_res = _run_git(
            ["ls-files", "--others", "--exclude-standard"],
            cwd=wt_dir,
            check=False,
        )
        excluded_media_set: set[str] = set()
        excl_media_file = wt_dir / ".git" / "info" / "exclude-media"
        if excl_media_file.is_file():
            try:
                excluded_media_set = {
                    ln.strip()
                    for ln in excl_media_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    if ln.strip()
                }
            except OSError:
                excluded_media_set = set()
        raw_untracked = [
            ln.strip()
            for ln in (others_res.stdout.splitlines() if others_res.returncode == 0 else [])
            if ln.strip() and ln.strip() not in excluded_media_set
        ]
        ref_candidates = list(dict.fromkeys([
            *tracked_changed,
            *[u for u in raw_untracked if not _is_ephemeral_build_or_test_artifact(u, root=wt_dir)],
        ]))
        for rel_u in raw_untracked:
            explicitly_kept_rt = _explicitly_targets_runtime_artifact(
                rel_u,
                allowed_paths,
                prompt_text=prompt_text,
                ref_files=ref_candidates,
                root=wt_dir,
            )
            if not explicitly_kept_rt and _is_ephemeral_build_or_test_artifact(rel_u, root=wt_dir):
                eph_p = wt_dir / rel_u
                if _is_untracked_runtime_artifact_name(rel_u):
                    excluded_runtime_artifacts.append(rel_u)
                elif (
                    eph_p.is_file()
                    and not eph_p.is_symlink()
                    and not rel_u.lower().endswith(_UNTRACKED_MODEL_OR_DATA_DUMP_SUFFIXES)
                ):
                    eph_p.unlink(missing_ok=True)
                continue
            if explicitly_kept_rt:
                preserved_runtime_artifacts.append(rel_u)
            untracked_to_stage.append(rel_u)

        ign_res = _run_git(
            ["ls-files", "--others", "--ignored", "--exclude-standard"],
            cwd=wt_dir,
            check=False,
        )
        dropped_ignored: list[str] = []
        if ign_res.returncode == 0 and ign_res.stdout.strip():
            for line in ign_res.stdout.splitlines():
                rel_i = line.strip()
                if not rel_i or rel_i in untracked_to_stage or rel_i in excluded_media_set:
                    continue
                explicitly_kept_ign = _explicitly_targets_runtime_artifact(
                    rel_i,
                    allowed_paths,
                    prompt_text=prompt_text,
                    ref_files=ref_candidates,
                    root=wt_dir,
                )
                if explicitly_kept_ign or not _is_ignored_build_artifact(rel_i, root=wt_dir):
                    if allowed_paths and not _matches_allowed_paths(rel_i, allowed_paths):
                        dropped_ignored.append(rel_i)
                        continue
                    if explicitly_kept_ign and rel_i not in preserved_runtime_artifacts:
                        preserved_runtime_artifacts.append(rel_i)
                    untracked_to_stage.append(rel_i)
                elif _is_untracked_runtime_artifact_name(rel_i) and rel_i not in excluded_runtime_artifacts:
                    excluded_runtime_artifacts.append(rel_i)
        if not hasattr(self, "_dropped_ignored_by_wt"):
            self._dropped_ignored_by_wt: dict[str, list[str]] = {}
        self._dropped_ignored_by_wt[str(wt_dir)] = dropped_ignored
        if not hasattr(self, "_excluded_runtime_artifacts_by_wt"):
            self._excluded_runtime_artifacts_by_wt: dict[str, list[str]] = {}
        self._excluded_runtime_artifacts_by_wt[str(wt_dir)] = excluded_runtime_artifacts
        if not hasattr(self, "_preserved_runtime_artifacts_by_wt"):
            self._preserved_runtime_artifacts_by_wt: dict[str, list[str]] = {}
        self._preserved_runtime_artifacts_by_wt[str(wt_dir)] = preserved_runtime_artifacts

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

    def _link_host_node_modules_for_checks(
        self,
        wt_dir: Path,
        *,
        allow_shallow_symlinks: bool = True,
    ) -> list[Path]:
        """Populate isolated `node_modules` in `wt_dir` from `self.workspace_root`."""
        if env_flag("CC_AGY_LINK_NODE_MODULES") is False:
            return []
        return link_host_node_modules_farm(
            self.workspace_root,
            wt_dir,
            allow_shallow_symlinks=allow_shallow_symlinks,
        )

    @staticmethod
    def _unlink_checks_node_modules(links: list[Path] | Path | None) -> None:
        """Remove shallow `node_modules` farm directories created in a worktree (never the host directory itself)."""
        unlink_host_node_modules_farm(links)

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
            if rel.startswith((".cc-agy-data/", ".worktrees/", ".ctx-session-reads/", ".cc-agy-review/")):
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
        preloaded_files: list[str] | None = None,
        prompt_sources: list[str] | None = None,
        recorded_budget: dict[str, int | None] | None = None,
        baseline_acceptance_probe: list[dict[str, Any]] | None = None,
        acceptance_remapped: list[dict[str, str]] | None = None,
        agy_conversation_id: str | None = None,
        agy_prior_conversation_usage: dict[str, Any] | None = None,
        agy_prior_conversation_turns: int | None = None,
        turn_instructions: Mapping[str, str] | None = None,
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
        denied_actions: list[dict[str, Any]] = []
        acceptance_results: list[dict[str, Any]] = []
        baseline_acceptance_warnings: list[str] = []
        contract_warnings: list[str] = []
        progress_path = self.store.progress_path(job_id)
        progress_before = len(read_progress(progress_path, limit=10**6))

        from server.verify import _git_dirs as _resolve_git_dirs

        _PROTECTED_GIT_SUBPATHS = ("hooks", "config", "config.worktree", "info")
        resolved_host_git_dirs = list(_resolve_git_dirs(self.workspace_root).values()) or [self.workspace_root / ".git"]
        host_git_dir_modes: list[tuple[Path, int | None]] = [
            (gd, (gd.lstat().st_mode & 0o7777) if (gd.is_dir() and not gd.is_symlink()) else None)
            for gd in resolved_host_git_dirs
        ]
        host_git_snaps: list[tuple[Path, dict[str, bytes]]] = [
            (gd / sub, _snapshot_directory_tree(gd / sub))
            for gd in resolved_host_git_dirs
            for sub in _PROTECTED_GIT_SUBPATHS
        ]
        wt_git_dir = wt_dir / ".git"
        wt_git_dir_mode: int | None = (
            (wt_git_dir.lstat().st_mode & 0o7777)
            if (not in_place and wt_git_dir.is_dir() and not wt_git_dir.is_symlink())
            else None
        )
        wt_git_snaps: list[tuple[Path, dict[str, bytes]]] = (
            [(wt_git_dir / sub, _snapshot_directory_tree(wt_git_dir / sub)) for sub in _PROTECTED_GIT_SUBPATHS]
            if (not in_place and wt_git_dir.exists())
            else []
        )
        orig_branch, pre_branches = self._snapshot_branches() if in_place else (None, [])
        worker_out: dict[str, Any] = {}
        active_agy_conv_id: str | None = agy_conversation_id
        active_agy_conv_usage: dict[str, int] | None = (
            {k: int(agy_prior_conversation_usage.get(k) or 0) for k in token_keys}
            if isinstance(agy_prior_conversation_usage, dict)
            else None
        )
        active_agy_conv_turns: int | None = (
            int(agy_prior_conversation_turns) if agy_prior_conversation_turns else None
        )
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

        def _register_cancel_hook(fn: Callable[[], None] | None) -> None:
            should_fire_immediately = False
            with self._lock:
                if fn is None:
                    self._cancel_hooks.pop(job_id, None)
                else:
                    self._cancel_hooks[job_id] = fn
                    should_fire_immediately = job_id in self._cancelled
            if should_fire_immediately and fn is not None:
                try:
                    fn()
                except Exception:
                    pass

        worker_nm_links: list[Path] = []
        host_nm_snap: dict[str, tuple[int, int, int]] | None = None
        budget_guard = getattr(self.store, "budget", None)
        try:
            with self._lock:
                if job_id in self._cancelled:
                    status = "cancelled"
                    error_msg = "Job cancelled before execution."
                    raise RuntimeError(error_msg)

            # Delegate workers run as multi-turn CLI/SDK sessions: enforce the shared daily cap
            # right before each worker phase (initial, continuation, or model escalation) without
            # holding a multi-minute reservation that would stall concurrent unpriced batch/ask calls.
            enforce_daily_budget(self.store, workspace_root=self.workspace_root, auth_mode=auth_mode)

            if not in_place:
                os_sandbox_active = False
                if auth_mode == "gemini_enterprise":
                    from server.worker_sandbox import detect_sandbox_backend, resolve_sandbox_mode

                    try:
                        os_sandbox_active = (
                            resolve_sandbox_mode(os.environ) != "off"
                            and detect_sandbox_backend(os.environ)[0] is not None
                        )
                    except ValueError:
                        os_sandbox_active = False
                allow_shallow = not (
                    auth_mode == "gemini_enterprise"
                    and not os_sandbox_active
                    and not is_read_only_tool_set(enabled_tools, disabled_tools)
                    and env_flag("CC_AGY_CLI_LINK_NODE_MODULES") is not True
                )
                worker_nm_links = self._link_host_node_modules_for_checks(
                    wt_dir,
                    allow_shallow_symlinks=allow_shallow,
                )
                if _nm_links_contain_host_symlinks(worker_nm_links, self.workspace_root):
                    host_nm_snap = _snapshot_host_nm_metadata(self.workspace_root)

            worker_out = run_antigravity_worker(
                prompts=worker_prompts,
                workspace_dir=wt_dir,
                model=model,
                project=project,
                location=location,
                api_key=api_key,
                auth_mode=auth_mode,
                timeout_seconds=timeout_seconds,
                enabled_tools=enabled_tools,
                disabled_tools=disabled_tools,
                enable_subagents=enable_subagents,
                backend=tracked_backend,
                thinking_level=thinking_level,
                max_tool_calls=max_tool_calls,
                max_total_tokens=max_total_tokens,
                progress_path=progress_path,
                acceptance=acceptance,
                allowed_paths=allowed_paths,
                agy_conversation_id=agy_conversation_id,
                agy_prior_conversation_usage=agy_prior_conversation_usage,
                agy_prior_conversation_turns=agy_prior_conversation_turns,
                register_cancel=_register_cancel_hook,
                host_workspace_root=self.workspace_root,
                nm_linked=bool(worker_nm_links),
                isolated_worktree=(not in_place),
                turn_instructions=turn_instructions,
            )
            if worker_out.get("agy_conversation_id"):
                active_agy_conv_id = str(worker_out["agy_conversation_id"])
            if isinstance(worker_out.get("agy_conversation_usage"), dict):
                active_agy_conv_usage = {
                    k: int(worker_out["agy_conversation_usage"].get(k) or 0) for k in token_keys
                }
            if worker_out.get("agy_conversation_turns"):
                active_agy_conv_turns = int(worker_out["agy_conversation_turns"])
            if worker_out.get("vertex_project"):
                project = str(worker_out["vertex_project"])
                with self._lock:
                    self._raw_projects[job_id] = project
            if worker_out.get("vertex_location"):
                location = str(worker_out["vertex_location"])
            if worker_out.get("reroute_note") and not reroute_note:
                reroute_note = str(worker_out["reroute_note"])

            response_text = str(worker_out.get("text") or "")
            raw_new_usage = worker_out.get("usage") or {}
            if worker_out.get("usage_is_cumulative") and prior_usage:
                new_usage = {
                    k: max(0, int(raw_new_usage.get(k) or 0) - int((prior_usage or {}).get(k) or 0))
                    for k in token_keys
                }
                new_usage["total_token_count"] = (
                    new_usage["prompt_token_count"]
                    + new_usage["candidates_token_count"]
                    + new_usage["thoughts_token_count"]
                )
            else:
                new_usage = raw_new_usage
            for k in token_keys:
                usage[k] += int(new_usage.get(k) or 0)
            usage["total_token_count"] = max(
                usage["total_token_count"],
                usage["prompt_token_count"]
                + usage["candidates_token_count"]
                + usage["thoughts_token_count"],
            )
            self._append_phase_usage_requests(usage, new_usage, model)
            if (
                budget_guard is not None
                and auth_mode != "gemini_enterprise"
                and any(int(new_usage.get(k) or 0) > 0 for k in token_keys)
            ):
                phase_cost = estimate_cost_usd(model, new_usage, single_request=False)
                phase_cents = float(phase_cost.get("total_cost_cents") or 0.0)
                if phase_cents > 0.0:
                    budget_guard.settle(
                        None,
                        job_id=job_id,
                        actual_cents=phase_cents,
                        workspace_root=self.workspace_root,
                        record_live=True,
                    )
            new_max_req = int(new_usage.get("max_request_prompt_tokens") or 0)
            if new_max_req > 0 or usage.get("max_request_prompt_tokens"):
                usage["max_request_prompt_tokens"] = max(
                    int(usage.get("max_request_prompt_tokens") or 0),
                    new_max_req,
                )
            tool_calls.extend(worker_out.get("tool_calls") or [])
            thoughts.extend(worker_out.get("thoughts") or [])
            stop_reason = worker_out.get("stop_reason")
            denied_actions = [d for d in (worker_out.get("denied_actions") or []) if isinstance(d, dict)]

            with self._lock:
                if job_id in self._cancelled:
                    status = "cancelled"
                    error_msg = "Job cancelled during execution."
                else:
                    status = "ok"
                    new_signals = read_progress(progress_path, limit=10**6)[progress_before:]
                    if stop_reason == "DENIED_ACTIONS":
                        # Policy blocked part of the work: pause for the user instead of applying a partial diff.
                        status = "needs_input"
                    elif stop_reason in CHECKPOINT_STOP_REASONS:
                        # Budget reached: not an error. The lead reviews and continues or redirects.
                        status = "checkpoint"
                    elif new_signals and new_signals[-1].get("kind") == "spec_conflict":
                        status = "spec_conflict"
                    elif new_signals and new_signals[-1].get("kind") in ("question", "blocker"):
                        status = "needs_input"

        except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError) as exc:
            with self._lock:
                is_canc = job_id in self._cancelled
                is_shutting_down = self._shutting_down
            if is_canc:
                status = "cancelled"
                error_msg = "Job cancelled during execution."
            elif is_shutting_down:
                status = "interrupted"
                error_msg = "MCP server shut down before worker completed."
            else:
                status = "timeout"
                error_msg = f"TimeoutError: {exc}"
            active_agy_conv_id, active_agy_conv_usage, active_agy_conv_turns = (
                self._recover_partial_worker_exception_state(
                    exc,
                    usage=usage,
                    tool_calls=tool_calls,
                    thoughts=thoughts,
                    worker_out=worker_out,
                    token_keys=token_keys,
                    active_agy_conv_id=active_agy_conv_id,
                    active_agy_conv_usage=active_agy_conv_usage,
                    active_agy_conv_turns=active_agy_conv_turns,
                    model=model,
                )
            )
        except Exception as exc:
            with self._lock:
                is_canc = job_id in self._cancelled
            if is_canc or status == "cancelled":
                status = "cancelled"
                error_msg = "Job cancelled by user."
            else:
                status = "error"
                error_msg = f"{type(exc).__name__}: {exc}"
            active_agy_conv_id, active_agy_conv_usage, active_agy_conv_turns = (
                self._recover_partial_worker_exception_state(
                    exc,
                    usage=usage,
                    tool_calls=tool_calls,
                    thoughts=thoughts,
                    worker_out=worker_out,
                    token_keys=token_keys,
                    active_agy_conv_id=active_agy_conv_id,
                    active_agy_conv_usage=active_agy_conv_usage,
                    active_agy_conv_turns=active_agy_conv_turns,
                    model=model,
                )
            )
        finally:
            self._unlink_checks_node_modules(worker_nm_links)
            try:
                def _restore_git_snap_group(
                    dir_modes: list[tuple[Path, int | None]],
                    snaps: list[tuple[Path, dict[str, bytes]]],
                ) -> bool:
                    tampered = False
                    for gd, orig_mode in dir_modes:
                        if gd.is_symlink():
                            tampered = True
                        elif orig_mode is not None and gd.is_dir():
                            try:
                                now_m = gd.lstat().st_mode & 0o7777
                                if now_m != orig_mode:
                                    tampered = True
                                os.chmod(gd, orig_mode | 0o700)
                            except OSError:
                                tampered = True
                    for snap_path, snap_data in snaps:
                        try:
                            if _restore_directory_tree(snap_path, snap_data):
                                tampered = True
                        except OSError:
                            tampered = True
                    for gd, orig_mode in dir_modes:
                        if orig_mode is not None and gd.is_dir() and not gd.is_symlink():
                            try:
                                os.chmod(gd, orig_mode)
                            except OSError:
                                tampered = True
                    return tampered

                host_git_tampered = _restore_git_snap_group(host_git_dir_modes, host_git_snaps)
                if wt_git_snaps:
                    _restore_git_snap_group([(wt_git_dir, wt_git_dir_mode)], wt_git_snaps)
                nm_tampered = (
                    host_nm_snap is not None
                    and not _verify_host_nm_metadata_unchanged(self.workspace_root, host_nm_snap)
                )
                if host_git_tampered:
                    status = "security_violation"
                    error_msg = "Worker attempted to modify host .git/hooks or .git/config; restored."
                elif nm_tampered:
                    status = "security_violation"
                    error_msg = "Worker modified host node_modules during isolated execution."

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
                        wt_dir,
                        baseline_sha,
                        allowed_paths=allowed_paths,
                        prompt_text="\n".join(recorded_prompts) if recorded_prompts else "",
                    )
                    baseline_file_hashes = self._compute_changed_baseline_hashes(
                        wt_dir, baseline_sha, files_changed, baseline_file_hashes
                    )
            except Exception as fin_exc:
                diff_text, files_changed, symlink_escapes = "", [], []
                if status in ("running", "ok"):
                    status = "error"
                    error_msg = f"Failed to collect worktree diff: {type(fin_exc).__name__}: {fin_exc}"
            is_mutating_job = not is_read_only_tool_set(enabled_tools, disabled_tools)
            vacuous_acceptance = False
            runnable_acc_status = (
                status in ("ok", "checkpoint", "needs_input")
                or (status == "timeout" and bool(files_changed))
            )
            effective_acceptance = list(acceptance or [])
            auto_default_acceptance = False
            compile_smoke_cmds: set[str] = set()
            target_acc_cwd = self.workspace_root if in_place else wt_dir
            # The worker never sees host node_modules, but the plugin's own checks (repo test runners,
            # TypeScript compile smoke) need installed packages, so link them for the acceptance phase only.
            acc_node_modules_link = self._link_host_node_modules_for_checks(wt_dir) if not in_place else None
            if not effective_acceptance and runnable_acc_status and files_changed and self._should_run_default_acceptance():
                effective_acceptance = discover_related_test_commands(target_acc_cwd, files_changed, [])
                if os.environ.get("CC_AGY_AUTO_IMPORT_CHECK", "1").strip().lower() not in ("0", "false", "no", "off"):
                    for icmd in build_import_smoke_command(target_acc_cwd, files_changed):
                        if icmd not in effective_acceptance:
                            effective_acceptance.append(icmd)
                if os.environ.get("CC_AGY_AUTO_COMPILE_CHECK", "1").strip().lower() not in ("0", "false", "no", "off"):
                    for ccmd in build_compile_smoke_commands(target_acc_cwd, files_changed):
                        if ccmd not in effective_acceptance:
                            effective_acceptance.append(ccmd)
                            compile_smoke_cmds.add(ccmd)
                auto_default_acceptance = bool(effective_acceptance)
            active_acceptance_remapped = list(acceptance_remapped or [])
            diagnostic_remaps: dict[str, tuple[str, list[str]]] = {}
            if effective_acceptance and active_acceptance_remapped:
                restored_cmds: list[str] = []
                still_remapped: list[dict[str, str]] = []
                remap_by_new = {
                    str(item.get("remapped_command") or "").strip(): item
                    for item in active_acceptance_remapped
                    if isinstance(item, dict) and item.get("remapped_command")
                }
                remap_by_orig: dict[str, list[dict[str, str]]] = {}
                for item in active_acceptance_remapped:
                    if isinstance(item, dict) and item.get("original_command") and item.get("remapped_command"):
                        orig_k = str(item.get("original_command") or "").strip()
                        remap_by_orig.setdefault(orig_k, []).append(item)
                for ecmd in effective_acceptance:
                    cur_cmd = ecmd.strip()
                    chain: list[dict[str, str]] = []
                    seen_cmds: set[str] = set()
                    while cur_cmd in remap_by_new and cur_cmd not in seen_cmds:
                        seen_cmds.add(cur_cmd)
                        entry_rm = remap_by_new[cur_cmd]
                        chain.append(entry_rm)
                        prev_c = str(entry_rm.get("original_command") or "").strip()
                        if not prev_c:
                            break
                        cur_cmd = prev_c
                    if chain:
                        chain.reverse()
                        orig_cmd_str = str(chain[0].get("original_command") or "").strip() or ecmd
                        diag_cmd_str = ecmd
                    elif cur_cmd in remap_by_orig:
                        # Forward walk when acceptance already preserves the original explicit command.
                        fwd_cmd = cur_cmd
                        while fwd_cmd in remap_by_orig and fwd_cmd not in seen_cmds:
                            seen_cmds.add(fwd_cmd)
                            entry_rm = remap_by_orig[fwd_cmd][-1]
                            chain.append(entry_rm)
                            nxt_c = str(entry_rm.get("remapped_command") or "").strip()
                            if not nxt_c:
                                break
                            fwd_cmd = nxt_c
                        orig_cmd_str = ecmd
                        diag_cmd_str = str(chain[-1].get("remapped_command") or "").strip() if chain else ecmd
                    else:
                        restored_cmds.append(ecmd)
                        continue
                    missing_still: list[str] = []
                    for entry_rm in chain:
                        orig_missing = str(entry_rm.get("missing_target") or "").strip()
                        if orig_missing and not (target_acc_cwd / orig_missing).exists():
                            if orig_missing not in missing_still:
                                missing_still.append(orig_missing)
                    if missing_still:
                        # Keep the full chain intact whenever any target in it remains missing so
                        # multi-step remaps never break lookup keys across continuation turns.
                        still_remapped.extend(chain)
                    restored_cmds.append(orig_cmd_str)
                    if missing_still and orig_cmd_str.strip() != diag_cmd_str.strip():
                        diagnostic_remaps[orig_cmd_str.strip()] = (diag_cmd_str, missing_still)
                effective_acceptance = restored_cmds
                if not auto_default_acceptance:
                    acceptance = list(restored_cmds)
                active_acceptance_remapped = still_remapped
            if effective_acceptance and runnable_acc_status and is_mutating_job and not files_changed:
                # Never run acceptance commands on an unmodified worktree for a mutating coding job:
                # pre-existing baseline tests passing with zero edits is vacuous (e.g. an early tool-budget checkpoint).
                vacuous_acceptance = True
                effective_acceptance = []
            has_timed_out_baseline_cmd = False
            if effective_acceptance and runnable_acc_status and files_changed and baseline_acceptance_probe:
                timed_out_baseline_cmds = {
                    str(bp.get("command") or "").strip()
                    for bp in baseline_acceptance_probe
                    if isinstance(bp, dict) and bp.get("timed_out") is True and bp.get("command")
                }
                if timed_out_baseline_cmds:
                    scoped_acceptance: list[str] = []
                    for orig_cmd in effective_acceptance:
                        if orig_cmd.strip() not in timed_out_baseline_cmds:
                            if orig_cmd not in scoped_acceptance:
                                scoped_acceptance.append(orig_cmd)
                            continue
                        related_cmds = discover_related_test_commands(target_acc_cwd, files_changed, [])
                        py_test_edits = [
                            rel
                            for rel in files_changed
                            if rel.endswith(".py")
                            and _is_test_file_path(rel)
                            and (target_acc_cwd / rel).is_file()
                        ]
                        if py_test_edits and "pytest" in orig_cmd:
                            runner_pref = _extract_pytest_runner_prefix(orig_cmd) or "pytest"
                            uncovered_py_edits = [
                                rel
                                for rel in py_test_edits
                                if not any(rel in rc or Path(rel).name in rc for rc in related_cmds)
                            ]
                            if uncovered_py_edits:
                                quoted_py = " ".join(shlex.quote(p) for p in uncovered_py_edits)
                                related_cmds.append(f"{runner_pref} -q {quoted_py}")
                        if related_cmds:
                            for rcmd in related_cmds:
                                if rcmd not in scoped_acceptance:
                                    scoped_acceptance.append(rcmd)
                        else:
                            has_timed_out_baseline_cmd = True
                            if orig_cmd not in scoped_acceptance:
                                scoped_acceptance.append(orig_cmd)
                    effective_acceptance = scoped_acceptance
            if effective_acceptance and runnable_acc_status:
                if not in_place and files_changed:
                    _invalidate_python_bytecode_for_paths(target_acc_cwd, files_changed)
                try:
                    reused_acc = (
                        self._try_reuse_worker_acceptance(
                            target_acc_cwd,
                            files_changed,
                            effective_acceptance,
                            worker_runs=worker_out.get("command_runs") if isinstance(worker_out, dict) else None,
                        )
                        if os.environ.get("CC_AGY_REUSE_WORKER_ACCEPTANCE", "1").strip().lower() not in ("0", "false", "no")
                        else None
                    )
                    if reused_acc is not None:
                        acceptance_results = reused_acc
                    else:
                        acc_timeout_override = 45.0 if has_timed_out_baseline_cmd else None
                        sess_dl, sess_el, _ = self._session_deadline_budget()
                        acc_wall_budget = max(5.0, (sess_dl - 15.0) - sess_el) if sess_dl is not None else None
                        acceptance_results = run_acceptance(
                            effective_acceptance,
                            target_acc_cwd,
                            timeout=acc_timeout_override,
                            total_wall_budget=acc_wall_budget,
                            restore_tracked=not in_place,
                            change_set=list(files_changed or []),
                            quarantine_dir=(self.store.data_dir / "quarantine" / job_id) if in_place else None,
                            diagnostic_remaps=diagnostic_remaps or None,
                        )
                    norm_prompt_for_missing = "\n".join(str(p) for p in (recorded_prompts or []) if p).replace("\\", "/")
                    for r_item in acceptance_results:
                        if not isinstance(r_item, dict):
                            continue
                        fmp: list[str] = []
                        for raw_mp in (r_item.get("filtered_missing_paths") or []):
                            clean_mp = str(raw_mp or "").replace("\\", "/").strip("/")
                            if not clean_mp:
                                continue
                            bname_mp = Path(clean_mp).name
                            in_allowed = bool(allowed_paths and _matches_allowed_paths(clean_mp, allowed_paths))
                            in_prompt = bool(
                                norm_prompt_for_missing
                                and (
                                    re.search(
                                        r"(?<![A-Za-z0-9_./-])" + re.escape(clean_mp) + r"(?![A-Za-z0-9_./-])",
                                        norm_prompt_for_missing,
                                    )
                                    or (
                                        bname_mp
                                        and re.search(
                                            r"\b(?:create|add|write|new)\b[^.\n]{0,120}\b" + re.escape(bname_mp) + r"\b",
                                            norm_prompt_for_missing,
                                            re.I,
                                        )
                                    )
                                )
                            )
                            if in_allowed or in_prompt:
                                fmp.append(clean_mp)
                        if fmp:
                            r_item["passed"] = False
                            r_item["missing_allowed_test_files"] = fmp
                            prev_t = str(r_item.get("output_tail") or "").rstrip()
                            r_item["output_tail"] = (
                                f"{prev_t}\nERROR: required test file(s) not created: {fmp}\n"
                                if prev_t
                                else f"ERROR: required test file(s) not created: {fmp}\n"
                            )
                    any_acc_timed_out = any(
                        isinstance(r, dict) and r.get("timed_out") is True for r in acceptance_results
                    )
                    if not any_acc_timed_out and not auto_default_acceptance and files_changed:
                        baseline_acceptance_warnings = evaluate_post_worker_baseline_failures(
                            acceptance_results,
                            baseline_acceptance_probe,
                            target_acc_cwd,
                            files_changed,
                        )
                    if auto_default_acceptance:
                        for dr in acceptance_results:
                            dr["auto_discovered_default"] = True
                            dr_counts = dr.get("counts") if isinstance(dr.get("counts"), dict) else {}
                            dr_tail = str(dr.get("output_tail") or "")
                            if dr.get("command") in compile_smoke_cmds:
                                dr["compile_smoke"] = True
                                if not dr.get("passed") and (
                                    dr.get("exit_code") == 127 or COMPILE_SMOKE_ENV_FAILURE_RE.search(dr_tail)
                                ):
                                    dr["passed"] = True
                                    dr["runner_incompatible_skipped"] = True
                                continue
                            if dr.get("exit_code") == 5:
                                dr["passed"] = True
                                dr["no_tests_collected"] = True
                            elif dr.get("exit_code") == 127 or (
                                not dr.get("passed")
                                and int(dr_counts.get("passed") or 0) == 0
                                and int(dr_counts.get("failed") or 0) == 0
                                and (
                                    "command not found" in dr_tail.lower()
                                    or "not recognized as an internal or external command" in dr_tail.lower()
                                    or "no module named pytest" in dr_tail.lower()
                                )
                            ):
                                dr["passed"] = True
                                dr["runner_incompatible_skipped"] = True
                            elif (
                                not dr.get("passed")
                                and int(dr_counts.get("passed") or 0) == 0
                                and int(dr_counts.get("failed") or 0) == 0
                                and int(dr_counts.get("errors") or 0) > 0
                                and "module 'py' has no attribute 'test'" in dr_tail
                            ):
                                dr["passed"] = True
                                dr["runner_incompatible_skipped"] = True
                    elif (
                        not any_acc_timed_out
                        and files_changed
                        and os.environ.get("CC_AGY_AUTO_REGRESSION_TESTS", "1").strip().lower()
                        not in ("0", "false", "no")
                    ):
                        extra_reg_cmds = discover_related_test_commands(
                            target_acc_cwd,
                            files_changed,
                            effective_acceptance,
                        )
                        if extra_reg_cmds:
                            sess_dl2, sess_el2, _ = self._session_deadline_budget()
                            reg_wall = max(5.0, (sess_dl2 - 15.0) - sess_el2) if sess_dl2 is not None else None
                            extra_res = run_acceptance(
                                extra_reg_cmds, target_acc_cwd, total_wall_budget=reg_wall,
                                restore_tracked=not in_place, change_set=list(files_changed or []),
                                quarantine_dir=(self.store.data_dir / "quarantine" / job_id) if in_place else None,
                            )
                            for er in extra_res:
                                er["auto_discovered_regression"] = True
                            acceptance_results.extend(extra_res)
                            if any(isinstance(r, dict) and r.get("timed_out") is True for r in extra_res):
                                any_acc_timed_out = True
                    if (
                        not any_acc_timed_out
                        and not auto_default_acceptance
                        and files_changed
                        and os.environ.get("CC_AGY_AUTO_IMPORT_CHECK", "").strip().lower()
                        in ("1", "true", "yes", "on", "always")
                    ):
                        import_cmds = build_import_smoke_command(target_acc_cwd, files_changed)
                        if import_cmds:
                            imp_res = run_acceptance(
                                import_cmds, target_acc_cwd, restore_tracked=not in_place,
                                change_set=list(files_changed or []),
                                quarantine_dir=(self.store.data_dir / "quarantine" / job_id) if in_place else None,
                            )
                            for ir in imp_res:
                                ir["auto_import_smoke"] = True
                            acceptance_results.extend(imp_res)
                except Exception as acc_exc:
                    acceptance_results = [
                        {"command": "<acceptance>", "passed": False, "error": f"{type(acc_exc).__name__}: {acc_exc}"}
                    ]
                self._unlink_checks_node_modules(acc_node_modules_link)
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
                            wt_dir,
                            baseline_sha,
                            allowed_paths=allowed_paths,
                            prompt_text="\n".join(recorded_prompts) if recorded_prompts else "",
                        )
                        baseline_file_hashes = self._compute_changed_baseline_hashes(
                            wt_dir, baseline_sha, files_changed, baseline_file_hashes
                        )
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            self._unlink_checks_node_modules(acc_node_modules_link)
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
                        f"delegate_apply(job_id, allow_scope_violations=True, include_paths=[...]) to approve, or call "
                        f"delegate_continue(job_id, follow_up_prompt=..., allowed_paths=...) to adjust scope."
                    )

            provenance = extract_diff_provenance(diff_text)
            for fpath in files_changed:
                if fpath not in provenance["files_changed"]:
                    provenance["files_changed"].append(fpath)

            diff_path = self.store.diff_path(job_id)
            diff_path.write_text(diff_text, encoding="utf-8")

            lead_only_prompts = [
                p
                for idx, p in enumerate(recorded_prompts or [])
                if idx == 0 or (prompt_sources and idx < len(prompt_sources) and prompt_sources[idx] == "lead")
            ]
            missing_prompt_symbols = (
                find_missing_prompt_symbols(
                    "\n".join(lead_only_prompts),
                    diff_text,
                    worktree_root=self.workspace_root if in_place else wt_dir,
                    files_changed=files_changed,
                    baseline_ref=baseline_sha or None,
                )
                if files_changed
                else []
            )
            contract_warnings = (
                detect_class_attribute_shadowing(
                    self.workspace_root if in_place else wt_dir,
                    files_changed,
                    baseline_ref=baseline_sha or "HEAD",
                )
                if files_changed
                else []
            )
            prompt_text_joined = "\n".join(recorded_prompts) if recorded_prompts else ""
            existing_rec_for_intent: dict[str, Any] = {}
            prior_raw_user_prompt = ""
            try:
                existing_rec_for_intent = self.store.read_receipt(job_id) or {}
                prior_raw_user_prompt = str(existing_rec_for_intent.get("raw_user_prompt") or "")
            except (OSError, ValueError, TypeError, KeyError):
                prior_raw_user_prompt = ""
            task_intent_text = "\n".join(
                s for s in [prior_raw_user_prompt, *(recorded_prompts[:1])] if s
            )
            prompt_requests_tests = _prompt_requests_test_authoring(task_intent_text or prompt_text_joined)
            allowed_only_tests = bool(
                allowed_paths and all(_is_test_file_path(ap) for ap in allowed_paths)
            )
            raw_stub_warnings = (
                detect_existing_test_stub_mutations(
                    self.workspace_root if in_place else wt_dir,
                    diff_text,
                    files_changed,
                    baseline_ref=baseline_sha or "HEAD",
                )
                if (files_changed and not allowed_only_tests)
                else []
            )
            stub_mutation_warnings = _unexempted_stub_mutation_warnings(
                raw_stub_warnings,
                task_intent_text or prompt_text_joined,
            )
            for sw in stub_mutation_warnings:
                if sw not in contract_warnings:
                    contract_warnings.append(sw)

            test_only_diff = bool(
                is_mutating_job
                and files_changed
                and all(_is_test_file_path(f) for f in files_changed)
                and not allowed_only_tests
                and not prompt_requests_tests
            )

            with self._lock:
                if job_id in self._cancelled:
                    status = "cancelled"
                    error_msg = error_msg or "Cancelled by user via delegate_cancel."

            external_tamper = bool(not in_place and files_changed and not workspace_dir_accessed)
            real_acc_passed_on_job = satisfies_timeout_promotion(
                acceptance_results,
                test_only_diff=test_only_diff,
            )
            recovered_from_timeout = False
            recovered_from_checkpoint = False
            timeout_note: str | None = None
            _qa_flag_chk = env_flag("CC_AGY_BALANCED_QA_REVIEW")
            _esc_dl_chk, _, _rem_b_chk = self._session_deadline_budget()
            _qa_will_audit_checkpoint = (
                not in_place
                and not existing_rec_for_intent.get("escalated_from_model")
                and not existing_rec_for_intent.get("skip_balanced_qa")
                and not existing_rec_for_intent.get("qa_review")
                and "flash" in (model or "").lower()
                and resolve_cost_profile() == "balanced"
                and _qa_flag_chk is not False
                and (self.worker_backend is None or _qa_flag_chk is True)
                and (_esc_dl_chk is None or _rem_b_chk >= 90.0)
            )
            if (
                status == "timeout"
                and files_changed
                and not scope_violations
                and not symlink_escapes
                and not missing_prompt_symbols
                and not external_tamper
                and real_acc_passed_on_job
                and env_flag("CC_AGY_PROMOTE_VERIFIED_TIMEOUT") is not False
            ):
                status = "ok"
                recovered_from_timeout = True
                timeout_note = error_msg
                error_msg = None
            elif (
                status == "checkpoint"
                and stop_reason == "MAX_TOOL_CALLS_EXCEEDED"
                and files_changed
                and not test_only_diff
                and not vacuous_acceptance
                and not scope_violations
                and not symlink_escapes
                and not missing_prompt_symbols
                and not contract_warnings
                and not external_tamper
                and real_acc_passed_on_job
                and (self._resolve_escalation_target_model(model) is None or _qa_will_audit_checkpoint)
                and not all(
                    re.match(r"^\s*(?:grep|egrep|fgrep|rg|test|\[|ls|stat)\b", str(r.get("command") or ""))
                    for r in acceptance_results
                    if isinstance(r, dict)
                )
                and env_flag("CC_AGY_PROMOTE_VERIFIED_CHECKPOINT") is not False
            ):
                status = "ok"
                recovered_from_checkpoint = True
                error_msg = None

            finished_at = round(time.time(), 3)
            clipped_text, was_truncated, orig_tokens = truncate_to_token_budget(
                response_text, MAX_TOOL_RESPONSE_TOKENS
            )
            effective_prompt_sources = list(prompt_sources or ["lead"] * len(recorded_prompts))
            if len(effective_prompt_sources) < len(recorded_prompts):
                effective_prompt_sources.extend(["lead"] * (len(recorded_prompts) - len(effective_prompt_sources)))
            effective_budget = (
                dict(recorded_budget)
                if isinstance(recorded_budget, dict)
                else {"max_tool_calls": max_tool_calls, "max_total_tokens": max_total_tokens}
            )
            spec_conflict_msg: str | None = None
            if status == "spec_conflict":
                latest_sigs = read_progress(progress_path, limit=8)
                for sig in reversed(latest_sigs):
                    if sig.get("kind") == "spec_conflict" and sig.get("message"):
                        spec_conflict_msg = str(sig["message"])
                        break
            if any(isinstance(r, dict) and r.get("vacuous") for r in acceptance_results):
                vacuous_acceptance = True
            computed_acc_passed = (
                False
                if (vacuous_acceptance and acceptance_results)
                else (all(r.get("passed") for r in acceptance_results) if acceptance_results else None)
            )
            clipped_thoughts_tail = (
                []
                if scrub_receipt
                else [
                    str(t).strip()[-600:]
                    for t in thoughts[-4:]
                    if str(t).strip()
                ]
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
                "prompt_sources": effective_prompt_sources,
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
                "thoughts_tail": clipped_thoughts_tail,
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
                "timeout_seconds": timeout_seconds,
                "budget": effective_budget,
                "stop_reason": stop_reason,
                **({"denied_actions": denied_actions} if denied_actions else {}),
                **(
                    {"worker_sandbox": dict(worker_out["worker_sandbox"])}
                    if isinstance(worker_out, dict) and isinstance(worker_out.get("worker_sandbox"), dict)
                    else {}
                ),
                **({"usage_estimated": True} if isinstance(worker_out, dict) and worker_out.get("usage_estimated") else {}),
                "containment": worker_out.get("containment", "none") if isinstance(worker_out, dict) else "none",
                "containment_warning": (
                    worker_out.get("containment_warning") if isinstance(worker_out, dict) else None
                ),
                "acceptance_results": acceptance_results,
                "acceptance_passed": computed_acc_passed,
                **({"vacuous_acceptance": True} if vacuous_acceptance else {}),
                **({"test_only_diff": True} if test_only_diff else {}),
                **({"spec_conflict": spec_conflict_msg} if spec_conflict_msg else {}),
                "missing_prompt_symbols": missing_prompt_symbols,
                "contract_warnings": contract_warnings,
                **({"stub_mutation_warnings": stub_mutation_warnings} if stub_mutation_warnings else {}),
                **({"baseline_acceptance_probe": baseline_acceptance_probe} if baseline_acceptance_probe else {}),
                **({"baseline_acceptance_warnings": baseline_acceptance_warnings} if baseline_acceptance_warnings else {}),
                **({"acceptance_remapped": active_acceptance_remapped} if active_acceptance_remapped else {}),
                "external_worktree_tamper": external_tamper,
                "preloaded_files": list(preloaded_files or []),
                **({"agy_conversation_id": active_agy_conv_id} if active_agy_conv_id else {}),
                **(
                    {"agy_conversation_usage": dict(active_agy_conv_usage)}
                    if isinstance(active_agy_conv_usage, dict)
                    else {}
                ),
                **(
                    {"agy_conversation_turns": int(active_agy_conv_turns)}
                    if active_agy_conv_turns
                    else {}
                ),
                **(
                    {"recovered_from_timeout": True, "timeout_note": timeout_note}
                    if recovered_from_timeout
                    else {}
                ),
                **({"recovered_from_checkpoint": True} if recovered_from_checkpoint else {}),
            }
            dropped_ign = getattr(self, "_dropped_ignored_by_wt", {}).pop(str(wt_dir), [])
            if dropped_ign:
                receipt_payload["dropped_out_of_scope_ignored"] = dropped_ign
            excl_art = getattr(self, "_excluded_runtime_artifacts_by_wt", {}).pop(str(wt_dir), [])
            if excl_art:
                receipt_payload["excluded_runtime_artifacts"] = excl_art
                self._archive_omitted_worktree_files(job_id, wt_dir, excl_art)
            pres_art = getattr(self, "_preserved_runtime_artifacts_by_wt", {}).pop(str(wt_dir), [])
            if pres_art:
                receipt_payload["preserved_runtime_artifacts"] = pres_art
            with self._lock:
                sparse_count = self._sparse_worktrees.get(job_id)
            if sparse_count is not None:
                receipt_payload["sparse_worktree"] = True
                receipt_payload["worktree_file_count"] = sparse_count
            receipt_payload.update(self._escalation_fields(job_id, receipt_payload))
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
            try:
                prior_disk_rec = self.store.read_receipt(job_id)
            except (OSError, ValueError, TypeError, KeyError):
                prior_disk_rec = {}
            with self._lock:
                explicitly_cancelled = job_id in self._cancelled
            if prior_disk_rec.get("cancelled") or prior_disk_rec.get("status") == "cancelled" or explicitly_cancelled:
                receipt_payload["status"] = "cancelled"
                receipt_payload["cancelled"] = True
                if not receipt_payload.get("error"):
                    receipt_payload["error"] = prior_disk_rec.get("error") or "Cancelled by user via delegate_cancel."
            elif prior_disk_rec.get("status") == "interrupted" or status == "interrupted":
                receipt_payload["status"] = "interrupted"
                if not receipt_payload.get("error"):
                    receipt_payload["error"] = prior_disk_rec.get("error") or error_msg
            elif status == "cancelled":
                receipt_payload["status"] = "cancelled"
                receipt_payload["cancelled"] = True
                if not receipt_payload.get("error"):
                    receipt_payload["error"] = "Cancelled."
            if prior_disk_rec.get("escalated_from_model") and not receipt_payload.get("escalated_from_model"):
                receipt_payload["escalated_from_model"] = prior_disk_rec["escalated_from_model"]
            for ctx_key in (
                "raw_user_prompt",
                "checklist",
                "spec_files",
                "context_files",
                "baseline_acceptance_probe",
                "prior_attempts",
                "cumulative_max_tool_calls",
                "cumulative_max_total_tokens",
                "explicit_max_tool_calls",
                "explicit_max_total_tokens",
                "explicit_thinking_level",
                "skip_balanced_qa",
                "qa_review",
                "owner_instance_id",
                "opencode_session_id",
            ):
                if prior_disk_rec.get(ctx_key) and not receipt_payload.get(ctx_key):
                    if (
                        ctx_key == "qa_review"
                        and isinstance(prior_disk_rec["qa_review"], dict)
                        and prior_disk_rec["qa_review"].get("passed") is False
                        and prior_disk_rec["qa_review"].get("resumed")
                        and receipt_payload.get("status") == "ok"
                        and receipt_payload.get("acceptance_passed") is True
                    ):
                        receipt_payload["qa_review"] = {
                            **prior_disk_rec["qa_review"],
                            "resolved_by_escalation": True,
                        }
                    else:
                        receipt_payload[ctx_key] = prior_disk_rec[ctx_key]
            if prior_disk_rec.get("applied") and not in_place:
                for preserve_key in (
                    "applied",
                    "applied_at_epoch",
                    "applied_files",
                    "applied_file_hashes",
                    "baseline_file_hashes",
                    "provenance",
                    "pre_apply",
                    "salvaged_on_shutdown",
                ):
                    if preserve_key in prior_disk_rec:
                        receipt_payload[preserve_key] = prior_disk_rec[preserve_key]
            try:
                Path(f"{progress_path}.live").unlink(missing_ok=True)
                progress_path.with_suffix(".live.json").unlink(missing_ok=True)
            except OSError:
                pass
            wrote_receipt = False
            try:
                saved = self.store.write_receipt(
                    job_id,
                    receipt_payload,
                    scrub=scrub_receipt,
                    workspace_root=str(self.workspace_root),
                    api_key=api_key,
                )
                wrote_receipt = True
            finally:
                if not wrote_receipt and budget_guard is not None:
                    try:
                        budget_guard.clear_job(job_id, workspace_root=self.workspace_root)
                    except (OSError, ValueError, KeyError):
                        pass
            with self._lock:
                if job_id in self._cancelled and wt_dir.exists():
                    self._cleanup_worktree(wt_dir)
            if in_place:
                self._release_in_place_slot(job_id)

        return saved

    @staticmethod
    def _append_phase_usage_requests(
        usage: dict[str, Any],
        phase_usage: dict[str, Any],
        model: str,
    ) -> None:
        """Append per-call requests from `phase_usage` (tagged with `model`) plus any unitemized remainder."""
        phase_reqs = phase_usage.get("requests")
        phase_p = int(phase_usage.get("prompt_token_count") or 0)
        phase_c = int(phase_usage.get("cached_content_token_count") or 0)
        phase_cand = int(phase_usage.get("candidates_token_count") or 0)
        phase_th = int(phase_usage.get("thoughts_token_count") or 0)
        phase_tot = max(
            int(phase_usage.get("total_token_count") or 0),
            phase_p + phase_cand + phase_th,
        )
        if isinstance(phase_reqs, list) and phase_reqs:
            tagged_reqs = [
                ({**r, "model": r.get("model") or model} if isinstance(r, dict) else r)
                for r in phase_reqs
            ]
            usage.setdefault("requests", []).extend(tagged_reqs)
            sum_p = sum(int(r.get("prompt_token_count") or 0) for r in tagged_reqs if isinstance(r, dict))
            sum_c = sum(int(r.get("cached_content_token_count") or 0) for r in tagged_reqs if isinstance(r, dict))
            sum_cand = sum(int(r.get("candidates_token_count") or 0) for r in tagged_reqs if isinstance(r, dict))
            sum_th = sum(int(r.get("thoughts_token_count") or 0) for r in tagged_reqs if isinstance(r, dict))
            rem_p = max(0, phase_p - sum_p)
            rem_c = min(rem_p, max(0, phase_c - sum_c))
            rem_cand = max(0, phase_cand - sum_cand)
            rem_th = max(0, phase_th - sum_th)
            if rem_p > 0 or rem_cand > 0 or rem_th > 0:
                usage["requests"].append(
                    {
                        "prompt_token_count": rem_p,
                        "cached_content_token_count": rem_c,
                        "candidates_token_count": rem_cand,
                        "thoughts_token_count": rem_th,
                        "total_token_count": rem_p + rem_cand + rem_th,
                        "model": model,
                    }
                )
        elif usage.get("requests") and (phase_p > 0 or phase_cand > 0 or phase_th > 0):
            usage["requests"].append(
                {
                    "prompt_token_count": phase_p,
                    "cached_content_token_count": phase_c,
                    "candidates_token_count": phase_cand,
                    "thoughts_token_count": phase_th,
                    "total_token_count": phase_tot,
                    "model": model,
                }
            )

    @classmethod
    def _merge_partial_exception_data(
        cls,
        exc: BaseException,
        usage: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        thoughts: list[str],
        *,
        model: str = "",
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
            cls._append_phase_usage_requests(usage, partial_u, model)
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

    @classmethod
    def _recover_partial_worker_exception_state(
        cls,
        exc: BaseException,
        *,
        usage: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        thoughts: list[str],
        worker_out: dict[str, Any],
        token_keys: tuple[str, ...] | list[str],
        active_agy_conv_id: str | None,
        active_agy_conv_usage: dict[str, int] | None,
        active_agy_conv_turns: int | None,
        model: str = "",
    ) -> tuple[str | None, dict[str, int] | None, int | None]:
        cls._merge_partial_exception_data(exc, usage, tool_calls, thoughts, model=model)
        if getattr(exc, "partial_containment", None):
            worker_out["containment"] = getattr(exc, "partial_containment")
        if getattr(exc, "partial_containment_warning", None):
            worker_out["containment_warning"] = getattr(exc, "partial_containment_warning")
        if getattr(exc, "partial_command_runs", None):
            worker_out["command_runs"] = getattr(exc, "partial_command_runs")
        if getattr(exc, "partial_agy_conversation_id", None):
            active_agy_conv_id = str(getattr(exc, "partial_agy_conversation_id"))
        partial_conv_u = getattr(exc, "partial_agy_conversation_usage", None)
        if isinstance(partial_conv_u, dict):
            active_agy_conv_usage = {k: int(partial_conv_u.get(k) or 0) for k in token_keys}
        if getattr(exc, "partial_agy_conversation_turns", None):
            active_agy_conv_turns = int(getattr(exc, "partial_agy_conversation_turns"))
        return active_agy_conv_id, active_agy_conv_usage, active_agy_conv_turns

    @staticmethod
    def _try_reuse_worker_acceptance(
        target_cwd: Path,
        files_changed: list[str],
        acceptance_cmds: list[str],
        worker_runs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]] | None:
        """If the worker already ran every acceptance command via `run_command` after its last file edit
        and all exited 0, reuse those passing results instead of re-spawning the test suite."""
        if not acceptance_cmds:
            return None
        records: list[dict[str, Any]] = []
        if isinstance(worker_runs, list) and worker_runs:
            for rec in worker_runs:
                if isinstance(rec, dict) and verify_worker_run_record(rec):
                    records.append(rec)
        else:
            runs_log = target_cwd / ".ctx-session-reads" / "worker_runs.jsonl"
            if not runs_log.is_file():
                return None
            try:
                for line in runs_log.read_text(encoding="utf-8", errors="replace").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if isinstance(rec, dict) and verify_worker_run_record(rec):
                        records.append(rec)
            except (OSError, ValueError, TypeError):
                return None
        if not records:
            return None
        max_mtime = 0.0
        missing_rels: list[str] = []
        current_hashes: dict[str, str] = {}
        for rel in files_changed:
            fp = target_cwd / rel
            try:
                if fp.exists() and fp.is_file():
                    max_mtime = max(max_mtime, fp.stat().st_mtime)
                    current_hashes[rel] = sha256_bytes(fp.read_bytes())
                else:
                    missing_rels.append(rel)
                    parent_dir = fp.parent if fp.parent.exists() else target_cwd
                    max_mtime = max(max_mtime, parent_dir.stat().st_mtime)
            except OSError:
                return None
        from server.verify import is_behavioral_check, vacuous_test_run

        _PYTEST_OPTS_WITH_VALUE = frozenset({
            "-k", "-m", "-c", "-o", "-p", "-W", "-r", "-n",
            "--rootdir", "--confcutdir", "--basetemp", "--maxfail",
            "--tb", "--import-mode", "--cov", "--cov-report",
            "--cov-config", "--junitxml", "--log-level", "--capture",
            "--durations", "--durations-min", "--deselect", "--ignore",
            "--ignore-glob", "--override-ini",
        })
        reused: list[dict[str, Any]] = []
        for acc_cmd in acceptance_cmds:
            if "pytest" in acc_cmd or "py.test" in acc_cmd:
                try:
                    acc_toks = shlex.split(acc_cmd)
                except ValueError:
                    acc_toks = acc_cmd.split()
                runner_idx: int | None = None
                for idx_t, tok in enumerate(acc_toks):
                    if Path(tok).name in ("pytest", "py.test"):
                        runner_idx = idx_t
                        break
                if runner_idx is not None:
                    skip_val = False
                    end_opts = False
                    for tok in acc_toks[runner_idx + 1:]:
                        if skip_val:
                            skip_val = False
                            continue
                        if not end_opts and tok == "--":
                            end_opts = True
                            continue
                        if not end_opts and tok.startswith("-"):
                            if tok in _PYTEST_OPTS_WITH_VALUE:
                                skip_val = True
                            continue
                        base_tok = tok.split("::", 1)[0].strip()
                        if not base_tok or not (target_cwd / base_tok).exists():
                            return None
                else:
                    for tok in acc_toks:
                        if tok.startswith("-"):
                            continue
                        base_tok = tok.split("::", 1)[0].strip()
                        if base_tok.endswith(".py") and not (target_cwd / base_tok).exists():
                            return None
            norm_acc = " ".join(acc_cmd.strip().split())
            matched_rec: dict[str, Any] | None = None
            for rec in reversed(records):
                rec_ts = float(rec.get("ts") or 0.0)
                if max_mtime > 0.0 and rec_ts < max_mtime:
                    continue
                if rec.get("exit_code") != 0 or rec.get("timed_out"):
                    continue
                rec_wd = str(rec.get("working_directory") or "").strip()
                if rec_wd not in ("", "."):
                    continue
                present_at_run = rec.get("present_files")
                if missing_rels:
                    deleted_at_run = rec.get("deleted_files")
                    if isinstance(deleted_at_run, list):
                        if any(m not in deleted_at_run for m in missing_rels):
                            continue
                    elif isinstance(present_at_run, list):
                        if any(m in present_at_run for m in missing_rels):
                            continue
                    else:
                        continue
                if isinstance(present_at_run, list):
                    if any(
                        not isinstance(pf, str)
                        or not pf
                        or Path(pf).is_absolute()
                        or ".." in Path(pf).parts
                        or not (target_cwd / pf).exists()
                        for pf in present_at_run
                    ):
                        continue
                rec_hashes = rec.get("file_hashes")
                if not isinstance(rec_hashes, dict):
                    continue
                if set(rec_hashes.keys()) != set(current_hashes.keys()):
                    continue
                if any(rec_hashes.get(rel) != sha_val for rel, sha_val in current_hashes.items()):
                    continue
                if any(rel in rec_hashes for rel in missing_rels):
                    continue
                rec_cmd = " ".join(str(rec.get("command") or "").strip().split())
                if rec_cmd == norm_acc:
                    matched_rec = rec
                    break
            if matched_rec is None:
                return None
            out_tail = str(matched_rec.get("output_tail") or "")
            counts = _parse_test_counts(out_tail)
            if is_behavioral_check(acc_cmd) and vacuous_test_run(out_tail, 0, False, counts):
                return None
            reused.append(
                {
                    "command": acc_cmd,
                    "passed": True,
                    "exit_code": 0,
                    "timed_out": False,
                    "duration_seconds": round(float(matched_rec.get("duration_seconds") or 0.0), 3),
                    "counts": counts,
                    "output_tail": out_tail,
                    "reused_from_worker": True,
                }
            )
        return reused

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
        preload_files: bool = True,
        auto_apply: bool | None = None,
        spec_files: list[str] | None = None,
        context_files: list[str] | None = None,
        media_files: list[str] | None = None,
        checklist: list[str] | None = None,
        opencode_session_id: str | None = None,
        on_job_allocated: Callable[[str], None] | None = None,
        progress_callback: Callable[[int, int | None, str], None] | None = None,
        stop_event: threading.Event | None = None,
        _skip_balanced_qa: bool = False,
    ) -> dict[str, Any]:
        """Start an Antigravity worker job.

        By default the worker edits an isolated standalone copy of the repo and its diff is
        applied later with delegate_apply (or automatically when auto_apply=True / CC_AGY_AUTO_APPLY_ON_PASS=1).
        With in_place=True the worker edits the real working tree directly (requires a clean tracked tree,
        one in_place job at a time, and no allowed_paths); the job is recorded as applied as soon as it finishes.
        """
        rpc_started_mono = time.monotonic()
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("MCP server is shutting down; refusing to start new worker job.")
        if wait and not in_place and self._is_bounded_lead_session():
            delegate_receipts = [
                r
                for r in self.store.list_receipts(workspace_root=self.workspace_root)
                if r.get("kind", "delegate") == "delegate"
            ]
            if delegate_receipts:
                prev_rec = delegate_receipts[-1]
                prev_jid = str(prev_rec.get("job_id") or "")
                prev_wt = prev_rec.get("worktree_dir")
                prev_status = str(prev_rec.get("status") or "")
                with self._lock:
                    prev_is_cancelled = bool(
                        prev_rec.get("cancelled")
                        or prev_status == "cancelled"
                        or prev_jid in self._cancelled
                    )
                prev_continuable = (
                    prev_status in ("timeout", "checkpoint", "needs_input")
                    or (prev_status == "ok" and prev_rec.get("acceptance_passed") is False)
                )
                prev_allowed = prev_rec.get("allowed_paths")
                new_allowed_probe = validate_allowed_paths(allowed_paths) if allowed_paths else None
                paths_clearly_disjoint = bool(
                    prev_allowed
                    and new_allowed_probe
                    and not any(_matches_allowed_paths(p, new_allowed_probe) for p in prev_allowed)
                    and not any(_matches_allowed_paths(p, prev_allowed) for p in new_allowed_probe)
                )
                if (
                    prev_jid
                    and not prev_is_cancelled
                    and prev_continuable
                    and not paths_clearly_disjoint
                    and prev_rec.get("files_changed")
                    and not prev_rec.get("applied")
                    and not (prev_rec.get("escalation") or {}).get("escalate_to_lead")
                    and prev_wt
                    and Path(prev_wt).is_dir()
                ):
                    return self.delegate_continue(
                        job_id=prev_jid,
                        follow_up_prompt=prompt,
                        allowed_paths=allowed_paths,
                        acceptance=acceptance,
                        timeout_seconds=timeout_seconds,
                        wait=True,
                        project=project,
                        location=location,
                        api_key=api_key,
                        scrub_receipt=scrub_receipt,
                        auto_apply=auto_apply,
                        opencode_session_id=opencode_session_id,
                        on_job_allocated=on_job_allocated,
                        progress_callback=progress_callback,
                        stop_event=stop_event,
                    )
        timeout_seconds = self._clamp_worker_timeout(timeout_seconds)

        validated_paths = validate_allowed_paths(allowed_paths)
        if in_place and validated_paths:
            raise ValueError(
                "allowed_paths cannot be enforced in in_place mode (edits land directly in your "
                "files). Drop allowed_paths or use the default isolated copy."
            )
        chosen_model = resolve_default_delegate_model(model, env=os.environ)
        escalated_from_model: str | None = None
        target_esc_model = self._resolve_escalation_target_model(chosen_model)
        if target_esc_model and self._should_escalate_new_delegate_start(
            prompt,
            allowed_paths=validated_paths,
            context_files=context_files,
            spec_files=spec_files,
            acceptance=acceptance,
        ):
            escalated_from_model = chosen_model
            chosen_model = target_esc_model
        resolved_thinking = resolve_thinking_level(
            thinking_level,
            model=chosen_model,
            read_only=is_read_only_tool_set(enabled_tools, disabled_tools),
        )
        plan_steps = validate_plan(plan)
        acceptance_cmds = validate_acceptance(acceptance, self.workspace_root)
        budget_calls, budget_tokens = resolve_budget(
            max_tool_calls,
            max_total_tokens,
            model=chosen_model,
        )
        preloaded_pairs: list[tuple[str, str]] = (
            select_candidate_files(self.workspace_root, prompt) if preload_files else []
        )
        preloaded_rel_paths = [rel for rel, _ in preloaded_pairs]
        use_repo_map = os.environ.get("CC_AGY_REPO_MAP", "1").strip().lower() not in ("0", "false", "no")
        repo_map_text = (
            build_repo_map(self.workspace_root, cache_dir=self.store.data_dir)
            if use_repo_map
            else ""
        )
        focus_targets = list(dict.fromkeys([*(validated_paths or []), *preloaded_rel_paths]))
        if not focus_targets and use_repo_map:
            focus_targets = [rel for rel, _ in select_candidate_files(self.workspace_root, prompt, max_files=6)]
        raw_user_prompt_text = ""
        if not spec_files:
            user_prompt_path = self.store.data_dir / ".last_user_prompt.json"
            if user_prompt_path.is_file():
                try:
                    up_obj = json.loads(user_prompt_path.read_text(encoding="utf-8"))
                    if isinstance(up_obj, dict):
                        up_ts = float(up_obj.get("ts") or 0.0)
                        up_text = str(up_obj.get("prompt") or "").strip()
                        up_sid = str(up_obj.get("session_id") or "").strip()
                        session_since = self._created_at_epoch - 1.0
                        active_sid = ""
                        marker = self.store.data_dir / ".lead_session_started"
                        if marker.exists():
                            session_since = marker.stat().st_mtime
                            try:
                                m_obj = json.loads(marker.read_text(encoding="utf-8"))
                                if isinstance(m_obj, dict):
                                    active_sid = str(m_obj.get("session_id") or "").strip()
                                    if m_obj.get("ts"):
                                        session_since = float(m_obj["ts"])
                            except (OSError, ValueError, TypeError):
                                pass
                        sid_matches = not (active_sid and up_sid and active_sid != up_sid)
                        if (
                            up_text
                            and sid_matches
                            and up_ts >= session_since
                            and len(up_text) > len(prompt.strip()) * 1.25
                            and up_text not in prompt
                        ):
                            clipped_up, _, _ = truncate_to_token_budget(up_text, 3000)
                            raw_user_prompt_text = clipped_up
                except (OSError, ValueError, TypeError):
                    raw_user_prompt_text = ""
        lead_read_paths: list[str] = []
        lead_read_windows: dict[str, tuple[int, int]] = {}
        lead_rf_path = self.store.data_dir / ".lead_read_files.json"
        if lead_rf_path.is_file():
            try:
                parsed_lrf = json.loads(lead_rf_path.read_text(encoding="utf-8"))
                if isinstance(parsed_lrf, list):
                    lead_read_paths = [str(x) for x in parsed_lrf if isinstance(x, str)]
            except (OSError, ValueError, TypeError):
                lead_read_paths = []
        lead_rw_path = self.store.data_dir / ".lead_read_windows.json"
        if lead_rw_path.is_file():
            try:
                parsed_lrw = json.loads(lead_rw_path.read_text(encoding="utf-8"))
                if isinstance(parsed_lrw, dict):
                    for k, v in parsed_lrw.items():
                        if isinstance(k, str) and isinstance(v, list) and len(v) == 2:
                            lead_read_windows[k] = (int(v[0]), int(v[1]))
            except (OSError, ValueError, TypeError):
                lead_read_windows = {}
        merged_ctx_paths = list(dict.fromkeys([*(context_files or []), *lead_read_paths]))
        if (
            os.environ.get("CC_AGY_INLINE_CANDIDATE_FILES", "").strip().lower() in ("1", "true", "yes")
            and len(preloaded_rel_paths) <= 4
        ):
            for pr_path in preloaded_rel_paths:
                if pr_path not in merged_ctx_paths:
                    merged_ctx_paths.append(pr_path)
        spec_inlined = _load_safe_workspace_files(self.workspace_root, spec_files, max_total_tokens=8000)
        spec_rel_set = {r for r, _ in spec_inlined}
        raw_ctx_inlined = _load_safe_workspace_files(
            self.workspace_root,
            [p for p in merged_ctx_paths if p not in spec_rel_set],
            max_total_tokens=4000,
            line_windows=lead_read_windows,
        )
        ctx_inlined: list[tuple[str, str]] = []
        for rel_c, content_c in raw_ctx_inlined:
            stripped_c = content_c.strip()
            if (
                "[truncated:" not in content_c
                and len(stripped_c) >= 40
                and (stripped_c in prompt or (raw_user_prompt_text and stripped_c in raw_user_prompt_text))
            ):
                continue
            ctx_inlined.append((rel_c, content_c))
        fully_inlined_rels = {
            r for r, c in (*spec_inlined, *ctx_inlined) if "[truncated:" not in c
        }
        outline_targets = [p for p in focus_targets if p not in fully_inlined_rels]
        focused_outline_text = (
            build_focused_file_outline(self.workspace_root, outline_targets, max_tokens=500)
            if use_repo_map and outline_targets
            else ""
        )
        clean_checklist = [str(item).strip() for item in (checklist or []) if str(item).strip()]
        from server.genai_ask import (
            MEDIA_TYPES,
            _resolve_safe_workspace_target,
            extract_workspace_media_paths,
            is_secret_file,
        )

        validated_media: list[str] = []
        if media_files is not None:
            if not isinstance(media_files, list):
                raise ValueError("media_files must be a list of workspace-relative media file paths.")
            for raw_m in media_files:
                if not isinstance(raw_m, str) or not raw_m.strip():
                    raise ValueError("Each entry in media_files must be a non-empty path string.")
                rel_m = _strip_dot_slash(raw_m.strip().replace("\\", "/"))
                if is_secret_file(rel_m):
                    raise ValueError(f"Secret file blocked in media_files: '{raw_m}'")
                target_m = _resolve_safe_workspace_target(self.workspace_root, rel_m)
                if not target_m.is_file():
                    raise ValueError(f"Media file not found: '{raw_m}'")
                ext_m = target_m.suffix.lower()
                if ext_m not in MEDIA_TYPES:
                    raise ValueError(
                        f"Unsupported media extension '{ext_m}' for '{raw_m}'. "
                        f"Supported: {', '.join(sorted(MEDIA_TYPES))}"
                    )
                norm_m = target_m.relative_to(self.workspace_root).as_posix()
                if norm_m not in validated_media:
                    validated_media.append(norm_m)
        for auto_m in extract_workspace_media_paths(self.workspace_root, prompt, raw_user_prompt_text):
            if auto_m not in validated_media:
                validated_media.append(auto_m)

        auth = resolve_auth_config(
            requested_project=project,
            requested_location=location,
            requested_api_key=api_key,
            model=chosen_model,
        )
        auth_mode = auth["auth_mode"]
        enforce_daily_budget(self.store, workspace_root=self.workspace_root, auth_mode=auth_mode)
        resolved_key = auth.get("api_key")
        resolved_project = auth.get("project")
        resolved_location = auth.get("location")
        reroute_note = auth.get("reroute_note")

        job_id = f"job-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        started_at = round(time.time(), 3)
        pre_untracked: dict[str, str | None] = {}
        if in_place:
            from server.regions import is_in_place_allowed

            if not is_in_place_allowed():
                raise ValueError(
                    "in_place=True is disabled by user policy (CC_AGY_AUTO_APPLY_ON_PASS=0 / --no-auto-apply or "
                    "CC_AGY_ALLOW_IN_PLACE=0). Run in an isolated git worktree (`in_place=False`) and review the "
                    "diff before calling `delegate_apply`."
                )
            self._acquire_in_place_slot(job_id)
            try:
                baseline_sha, pre_untracked = self._check_in_place_ready()
            except Exception:
                self._release_in_place_slot(job_id)
                raise
            wt_dir, baseline_hashes = self.workspace_root, {}
        else:
            wt_dir, baseline_sha, baseline_hashes = self._prepare_worktree(
                job_id,
                allowed_paths=validated_paths,
                context_files=merged_ctx_paths,
                spec_files=spec_files,
                media_files=validated_media,
            )

        baseline_probe: list[dict[str, Any]] = []
        acceptance_remapped: list[dict[str, str]] = []
        if (
            acceptance_cmds
            and not in_place
            and not is_read_only_tool_set(enabled_tools, disabled_tools)
            and env_flag("CC_AGY_BASELINE_ACCEPTANCE_PROBE") is not False
        ):
            per_probe_limit = min(15.0, max(0.3, float(timeout_seconds) * 0.4))
            total_probe_limit = min(25.0, max(0.5, float(timeout_seconds) * 0.5))
            probe_nm_link = self._link_host_node_modules_for_checks(wt_dir)
            try:
                baseline_probe = probe_baseline_acceptance(
                    acceptance_cmds,
                    wt_dir,
                    per_cmd_timeout=per_probe_limit,
                    total_timeout=total_probe_limit,
                    reset_tracked=True,
                )
                remapped_cmds, did_remap = remap_nonexistent_pytest_acceptance_cmds(
                    acceptance_cmds,
                    wt_dir,
                    baseline_probe,
                    allowed_paths=validated_paths,
                    prompt_text=f"{prompt}\n{raw_user_prompt_text}",
                    remap_details=acceptance_remapped,
                )
                if did_remap:
                    # Keep acceptance_cmds equal to the caller's explicit commands so missing targets never
                    # silently satisfy acceptance; probe distinct remapped targets for diagnostic context only.
                    diag_only_cmds = [
                        rcmd for rcmd, ocmd in zip(remapped_cmds, acceptance_cmds) if rcmd.strip() != ocmd.strip()
                    ]
                    if diag_only_cmds:
                        diag_probes = probe_baseline_acceptance(
                            diag_only_cmds,
                            wt_dir,
                            per_cmd_timeout=per_probe_limit,
                            total_timeout=total_probe_limit,
                            reset_tracked=True,
                        )
                        for dp in diag_probes:
                            if isinstance(dp, dict) and not dp.get("passed"):
                                dp_copy = dict(dp)
                                dp_copy["diagnostic_only"] = True
                                baseline_probe.append(dp_copy)
            finally:
                self._unlink_checks_node_modules(probe_nm_link)

        worker_prompt = render_worker_prompt(
            prompt,
            plan=plan_steps,
            acceptance=acceptance_cmds,
            max_tool_calls=budget_calls,
            max_total_tokens=budget_tokens,
            preloaded_files=preloaded_pairs,
            repo_map=repo_map_text,
            focused_outline=focused_outline_text,
            spec_files_inlined=spec_inlined,
            context_files_inlined=ctx_inlined,
            checklist=clean_checklist,
            raw_user_prompt=raw_user_prompt_text,
            baseline_acceptance_probe=baseline_probe,
            allowed_paths=validated_paths,
            media_files=validated_media,
        )

        with self._lock:
            if resolved_project:
                self._raw_projects[job_id] = resolved_project
            if resolved_key:
                self._raw_api_keys[job_id] = resolved_key
            sparse_init_count = self._sparse_worktrees.get(job_id)

        spec_rel_list = [r for r, _ in spec_inlined]
        ctx_rel_list = [r for r, _ in ctx_inlined]
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
            "prompt_sources": ["lead"],
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
            "owner_instance_id": self._instance_id,
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
            "timeout_seconds": timeout_seconds,
            "budget": {"max_tool_calls": budget_calls, "max_total_tokens": budget_tokens},
            "explicit_max_tool_calls": bool(max_tool_calls is not None),
            "explicit_max_total_tokens": bool(max_total_tokens is not None),
            "explicit_thinking_level": bool(thinking_level is not None and str(thinking_level).strip()),
            **({"skip_balanced_qa": True} if _skip_balanced_qa else {}),
            "preloaded_files": preloaded_rel_paths,
            **({"raw_user_prompt": raw_user_prompt_text} if raw_user_prompt_text else {}),
            **({"checklist": clean_checklist} if clean_checklist else {}),
            **({"spec_files": spec_rel_list} if spec_rel_list else {}),
            **({"context_files": ctx_rel_list} if ctx_rel_list else {}),
            **({"media_files": validated_media} if validated_media else {}),
            **({"baseline_acceptance_probe": baseline_probe} if baseline_probe else {}),
            **({"acceptance_remapped": acceptance_remapped} if acceptance_remapped else {}),
            **({"opencode_session_id": opencode_session_id} if opencode_session_id else {}),
            "error": None,
        }
        if sparse_init_count is not None:
            initial_receipt["sparse_worktree"] = True
            initial_receipt["worktree_file_count"] = sparse_init_count
        if escalated_from_model:
            initial_receipt["escalated_from_model"] = escalated_from_model
        self.store.write_receipt(
            job_id,
            initial_receipt,
            scrub=scrub_receipt,
            workspace_root=str(self.workspace_root),
            api_key=resolved_key,
        )

        with self._lock:
            if self._shutting_down:
                if in_place:
                    self._release_in_place_slot(job_id)
                elif wt_dir.exists():
                    self._cleanup_worktree(wt_dir)
                try:
                    self.store.receipt_path(job_id).unlink(missing_ok=True)
                except OSError:
                    pass
                raise RuntimeError("MCP server is shutting down; refusing to start new worker job.")
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
                preloaded_files=preloaded_rel_paths,
                prompt_sources=["lead"],
                recorded_budget={"max_tool_calls": budget_calls, "max_total_tokens": budget_tokens},
                baseline_acceptance_probe=baseline_probe,
                acceptance_remapped=acceptance_remapped,
            )
            self._futures[job_id] = fut
            self._owned_job_ids.add(job_id)

        if callable(on_job_allocated):
            try:
                on_job_allocated(job_id)
            except Exception:
                pass
        if stop_event is not None and stop_event.is_set():
            self.delegate_cancel(job_id)

        if wait:
            setup_elapsed = max(0.0, time.monotonic() - rpc_started_mono)
            wait_slice = self._resolve_wait_slice_seconds(
                timeout_seconds,
                elapsed_seconds=setup_elapsed,
            )
            if progress_callback is None and stop_event is None:
                self._wait_for_future_or_cancel(job_id, fut, timeout=wait_slice)
            else:
                self._wait_for_future_or_cancel(
                    job_id,
                    fut,
                    timeout=wait_slice,
                    progress_callback=progress_callback,
                    stop_event=stop_event,
                    max_tool_calls=budget_calls,
                )
            receipt = self.store.read_receipt(job_id)
            if not fut.done() or receipt.get("status") == "running":
                with self._lock:
                    self._wait_sliced_auto_apply[job_id] = auto_apply
                return self._digest_with_progress(receipt)
            return self._finalize_completed_wait_job(
                job_id,
                receipt,
                auto_apply=auto_apply,
                in_place=in_place,
                wait_elapsed=max(0.0, time.monotonic() - rpc_started_mono),
                timeout_seconds=timeout_seconds,
            )

        return {
            "job_id": job_id,
            "status": "running",
            "auth_mode": auth_mode,
            "model": chosen_model,
            **({"escalated_from_model": escalated_from_model} if escalated_from_model else {}),
            "vertex_location": resolved_location,
            "reroute_note": reroute_note,
            "allowed_paths": validated_paths,
            **({"media_files": validated_media} if validated_media else {}),
            "worktree_dir": str(wt_dir),
            "in_place": in_place,
            **({"opencode_session_id": opencode_session_id} if opencode_session_id else {}),
        }

    def _wait_for_future_or_cancel(
        self,
        job_id: str,
        fut: Future[dict[str, Any]],
        *,
        timeout: float,
        progress_callback: Callable[[int, int | None, str], None] | None = None,
        stop_event: threading.Event | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        """Wait for `fut` to complete or for `job_id` to be cancelled, streaming live progress when requested."""
        started_mono = time.monotonic()
        deadline = started_mono + max(0.01, float(timeout))
        last_reported_calls = -1
        last_reported_sig_count = 0
        last_heartbeat_mono = started_mono
        prog_path = self.store.progress_path(job_id)
        live_path = Path(f"{prog_path}.live")
        if callable(progress_callback):
            try:
                progress_callback(0, max_tool_calls, f"Job {job_id}: worker started")
            except Exception:
                pass
        while not fut.done():
            if stop_event is not None and stop_event.is_set():
                try:
                    self.delegate_cancel(job_id)
                except Exception:
                    pass
                return
            with self._lock:
                if job_id in self._cancelled:
                    return
            if callable(progress_callback):
                now_m = time.monotonic()
                live_calls = 0
                last_tool_desc = ""
                try:
                    if live_path.is_file():
                        ldata = json.loads(live_path.read_text(encoding="utf-8"))
                        if isinstance(ldata, dict):
                            live_calls = int(ldata.get("straitjacket_calls") or 0)
                            lt = str(ldata.get("last_tool") or "").strip()
                            ltarg = str(ldata.get("last_target") or "").strip()
                            if lt:
                                last_tool_desc = f"{lt} {ltarg}".strip()
                except (OSError, ValueError, TypeError):
                    pass
                sigs = read_progress(prog_path, limit=16)
                if len(sigs) > last_reported_sig_count:
                    last_reported_sig_count = len(sigs)
                    latest_sig = sigs[-1]
                    sig_msg = str(latest_sig.get("message") or "").strip()[:140]
                    try:
                        progress_callback(
                            max(0, live_calls),
                            max_tool_calls,
                            f"Job {job_id} [{latest_sig.get('kind', 'progress')}]: {sig_msg}",
                        )
                        last_heartbeat_mono = now_m
                    except Exception:
                        pass
                elif live_calls > last_reported_calls and live_calls > 0:
                    last_reported_calls = live_calls
                    step_note = f": {last_tool_desc}" if last_tool_desc else ""
                    try:
                        progress_callback(
                            live_calls,
                            max_tool_calls,
                            f"Job {job_id}: step {live_calls}{step_note} ({int(now_m - started_mono)}s)",
                        )
                        last_heartbeat_mono = now_m
                    except Exception:
                        pass
                elif now_m - last_heartbeat_mono >= 5.0:
                    last_heartbeat_mono = now_m
                    try:
                        progress_callback(
                            max(0, live_calls),
                            max_tool_calls,
                            f"Job {job_id}: working ({max(0, live_calls)} tool calls, {int(now_m - started_mono)}s)",
                        )
                    except Exception:
                        pass
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
            if receipt.get("status") != "running":
                with self._lock:
                    was_wait_sliced = safe_id in self._wait_sliced_auto_apply
                    sliced_auto_apply = self._wait_sliced_auto_apply.pop(safe_id, None)
                if was_wait_sliced:
                    digest = self._finalize_completed_wait_job(
                        safe_id,
                        receipt,
                        auto_apply=sliced_auto_apply,
                        in_place=bool(receipt.get("in_place")),
                        wait_elapsed=0.0,
                        timeout_seconds=float(receipt.get("timeout_seconds") or 600.0),
                    )
                    digest["baseline_head_sha"] = active_baseline_sha
                    return digest
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

    def _finalize_completed_wait_job(
        self,
        job_id: str,
        receipt: dict[str, Any],
        *,
        auto_apply: bool | None,
        in_place: bool,
        wait_elapsed: float = 0.0,
        timeout_seconds: float | None = None,
        wait_budget: float | None = None,
    ) -> dict[str, Any]:
        """Handle post-completion escalation (Flash -> Argon) or auto-apply for a synchronous/wait-sliced job."""
        allow_turn1_inplace_escalation = (
            self.worker_backend is None
            or is_strict_mode_enabled()
            or bool(os.environ.get("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "").strip())
        )
        with self._lock:
            is_cancelled_now = (job_id in self._cancelled) or bool(receipt.get("cancelled"))
            shutting_down_now = self._shutting_down
        if shutting_down_now:
            return self._digest_with_progress(receipt)
        acc_results_list = [
            r for r in (receipt.get("acceptance_results") or []) if isinstance(r, dict)
        ]
        only_cli_cfg_err = (
            receipt.get("status") == "ok"
            and not receipt.get("vacuous_acceptance")
            and bool(receipt.get("files_changed"))
            and bool(acc_results_list)
            and _only_compiler_cli_config_failures(acc_results_list)
        )
        prompt_text_for_intent = "\n".join(
            str(x)
            for x in [
                receipt.get("raw_user_prompt") or "",
                *((receipt.get("prompts") or [])[:1]),
            ]
            if x
        )
        unexempted_stub_warns = _unexempted_stub_mutation_warnings(
            receipt.get("stub_mutation_warnings"),
            prompt_text_for_intent,
        )
        has_stub_mutation_for_esc = bool(unexempted_stub_warns)
        needs_turn1_escalation = (
            (
                worker_turn_failed(receipt.get("status"), receipt.get("acceptance_passed"))
                or bool(receipt.get("vacuous_acceptance"))
                or receipt.get("status") in ("checkpoint", "needs_input")
                or has_stub_mutation_for_esc
            )
            and not only_cli_cfg_err
        )
        esc_session_deadline, _esc_elapsed, rem_esc_budget = self._session_deadline_budget()
        has_escalation_time = esc_session_deadline is None or rem_esc_budget >= 120.0
        if (
            self._is_bounded_lead_session()
            and allow_turn1_inplace_escalation
            and not in_place
            and not is_cancelled_now
            and not receipt.get("applied")
            and (receipt.get("escalation") or {}).get("reason") != "quota_or_rate_limit"
            and not receipt.get("escalated_from_model")
            and self._resolve_escalation_target_model(str(receipt.get("model") or "")) is not None
            and needs_turn1_escalation
            and has_escalation_time
            and receipt.get("status") not in ("cancelled", "interrupted", "spec_conflict")
        ):
            acc_failures = [
                f"$ {r.get('command', '')} (exit {r.get('exit_code', 1)}):\n{r.get('output_tail', '')}"
                for r in acc_results_list
                if not r.get("passed")
            ]
            stub_warn_list = [str(w) for w in unexempted_stub_warns if w]
            if acc_failures:
                fail_detail = "\n\nFailing acceptance commands from the Flash attempt:\n" + "\n\n".join(acc_failures)
            elif stub_warn_list:
                fail_detail = "\n\nPre-existing test mutation warnings from the Flash attempt:\n" + "\n".join(
                    f"- {w}" for w in stub_warn_list
                )
            else:
                fail_detail = (
                    f"\n\nPrior Flash attempt ended with status={receipt.get('status')!r}, error={receipt.get('error')!r}."
                )
            prior_flash_summary = " ".join(str(receipt.get("summary") or "").split())[:500]
            flash_summary_note = (
                f"\n\nPrior Flash diagnostic summary:\n{prior_flash_summary}"
                if prior_flash_summary
                else ""
            )
            esc_target_name = self._resolve_escalation_target_model(str(receipt.get("model") or "")) or "Gemini 4 Argon"
            worker_timeout = float(timeout_seconds or receipt.get("timeout_seconds") or 600.0)
            if esc_session_deadline is not None:
                worker_timeout = min(worker_timeout, max(60.0, rem_esc_budget))
            try:
                return self.delegate_continue(
                    job_id=job_id,
                    follow_up_prompt=(
                        "The initial Gemini Flash attempt did not complete all requirements or failed verification checks. "
                        f"You are now escalated to {esc_target_name}. Inspect the existing changes in the worktree, "
                        "complete all requirements from the original task specification, and ensure all acceptance "
                        "and unit tests pass. When a test fails, trace and fix the root cause in production code "
                        "(including imported helper or caller modules) rather than weakening test inputs or altering "
                        "pre-existing test expectations (unless the specification explicitly requires updating existing "
                        "test expectations)." + flash_summary_note + fail_detail
                    ),
                    allowed_paths=receipt.get("allowed_paths"),
                    acceptance=None,
                    timeout_seconds=worker_timeout,
                    wait=True,
                    scrub_receipt=bool(receipt.get("scrub_receipt", False)),
                    auto_apply=auto_apply,
                    _wait_elapsed=wait_elapsed,
                    _wait_budget=wait_budget,
                    _prompt_source="plugin_escalation",
                )
            except BudgetExceededError as exc:
                digest = self._digest_with_progress(receipt)
                digest["budget_exceeded"] = True
                digest["budget_exceeded_note"] = str(exc)
                return digest
        retry_reason = (
            self._same_model_retry_reason(receipt)
            if (
                self._is_bounded_lead_session()
                and allow_turn1_inplace_escalation
                and not in_place
                and not is_cancelled_now
                and self._resolve_escalation_target_model(str(receipt.get("model") or "")) is None
            )
            else None
        )
        if retry_reason:
            # Economy has no Argon escalation: give the same worker one more turn inside the plugin so the
            # Opus Lead is not woken (and billed) just to say "keep going".
            acc_failures = [
                f"$ {r.get('command', '')} (exit {r.get('exit_code', 1)}):\n{r.get('output_tail', '')}"
                for r in (receipt.get("acceptance_results") or [])
                if isinstance(r, dict) and not r.get("passed")
            ]
            if retry_reason == "tool_call_checkpoint":
                if receipt.get("acceptance_passed") is True and receipt.get("files_changed"):
                    lead_in = (
                        "Your previous turn hit its tool-call limit after modifying "
                        f"{receipt.get('files_changed')} and passing the configured acceptance command. "
                        "Check in <=2 calls (`git status --short` or targeted test) whether every requirement "
                        "and unit test from the original task is already complete; if anything is still missing, "
                        "add only that missing piece and run the acceptance check, otherwise stop immediately."
                    )
                else:
                    lead_in = (
                        "Your previous turn hit its tool-call limit before you finished. Continue in the same "
                        "worktree: run `git diff` to see what you already changed, finish every remaining "
                        "requirement from the original task, and make the acceptance checks pass."
                    )
            elif retry_reason == "vacuous_acceptance":
                lead_in = (
                    "Your previous turn did not produce a verifiable change (no modified files, or the "
                    "acceptance checks collected 0 tests). Re-read the original task, make the required "
                    "edits in this worktree, and make sure the acceptance checks actually run them. If the "
                    "task truly needs no change, say so and stop without editing."
                )
            else:
                lead_in = (
                    "Your previous turn finished but the acceptance checks failed. Fix the failures below "
                    "in this worktree without discarding correct work, and complete every requirement "
                    "from the original task."
                )
            fail_detail = (
                "\n\nFailing acceptance commands:\n" + "\n\n".join(acc_failures) if acc_failures else ""
            )
            worker_timeout = float(timeout_seconds or receipt.get("timeout_seconds") or 600.0)
            session_deadline, _sess_elapsed, rem_worker_budget = self._session_deadline_budget()
            if session_deadline is not None:
                worker_timeout = min(worker_timeout, 240.0, max(45.0, 0.5 * rem_worker_budget))
            elif resolve_cost_profile() == "economy":
                worker_timeout = min(worker_timeout, 240.0)
            try:
                return self.delegate_continue(
                    job_id=job_id,
                    follow_up_prompt=lead_in + fail_detail,
                    allowed_paths=receipt.get("allowed_paths"),
                    acceptance=None,
                    timeout_seconds=worker_timeout,
                    wait=True,
                    scrub_receipt=bool(receipt.get("scrub_receipt", False)),
                    auto_apply=auto_apply,
                    _wait_elapsed=wait_elapsed,
                    _wait_budget=wait_budget,
                    _prompt_source="plugin_retry",
                )
            except BudgetExceededError as exc:
                digest = self._digest_with_progress(receipt)
                digest["budget_exceeded"] = True
                digest["budget_exceeded_note"] = str(exc)
                return digest
        qa_escalated_digest = self._maybe_run_balanced_pre_apply_qa(
            job_id,
            receipt,
            auto_apply=auto_apply,
            in_place=in_place,
            is_cancelled_now=is_cancelled_now,
            wait_elapsed=wait_elapsed,
            timeout_seconds=timeout_seconds,
            wait_budget=wait_budget,
        )
        if qa_escalated_digest is not None:
            return qa_escalated_digest
        receipt = self.store.read_receipt(job_id)
        digest = self._digest_with_progress(receipt)
        return self._maybe_auto_apply_digest(
            job_id,
            digest,
            auto_apply=auto_apply,
            in_place=in_place,
        )


    def delegate_result(
        self,
        job_id: str,
        *,
        wait: bool = True,
        timeout: float = 240.0,
        on_job_allocated: Callable[[str], None] | None = None,
        progress_callback: Callable[[int, int | None, str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Return the compact completion digest for job_id, optionally waiting if still running."""
        safe_id = validate_job_id(job_id)
        if callable(on_job_allocated):
            try:
                on_job_allocated(safe_id)
            except Exception:
                pass
        if stop_event is not None and stop_event.is_set():
            self.delegate_cancel(safe_id)
        wait_started_mono = time.monotonic()
        with self._lock:
            fut = self._futures.get(safe_id)
            has_sliced_state = safe_id in self._wait_sliced_auto_apply
        capped_timeout = self._resolve_wait_slice_seconds(min(float(timeout), 540.0))
        if capped_timeout < 1.0 and float(timeout) <= 30.0 and has_sliced_state:
            capped_timeout = max(capped_timeout, float(timeout))
        if fut is not None and wait and not fut.done():
            if progress_callback is None and stop_event is None:
                self._wait_for_future_or_cancel(safe_id, fut, timeout=capped_timeout)
            else:
                self._wait_for_future_or_cancel(
                    safe_id,
                    fut,
                    timeout=capped_timeout,
                    progress_callback=progress_callback,
                    stop_event=stop_event,
                )
        receipt = self.store.read_receipt(safe_id)
        if (fut is not None and not fut.done()) or receipt.get("status") == "running":
            return self._digest_with_progress(receipt)
        with self._lock:
            was_wait_sliced = safe_id in self._wait_sliced_auto_apply
            sliced_auto_apply = self._wait_sliced_auto_apply.pop(safe_id, None)
        if was_wait_sliced:
            return self._finalize_completed_wait_job(
                safe_id,
                receipt,
                auto_apply=sliced_auto_apply,
                in_place=bool(receipt.get("in_place")),
                wait_elapsed=time.monotonic() - wait_started_mono,
                timeout_seconds=float(receipt.get("timeout_seconds") or 600.0),
                wait_budget=capped_timeout,
            )
        return self._digest_with_progress(receipt)

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
        force_retry: bool = False,
        scrub_receipt: bool | None = None,
        auto_apply: bool | None = None,
        opencode_session_id: str | None = None,
        on_job_allocated: Callable[[str], None] | None = None,
        progress_callback: Callable[[int, int | None, str], None] | None = None,
        stop_event: threading.Event | None = None,
        _wait_elapsed: float = 0.0,
        _wait_budget: float | None = None,
        _prompt_source: str = "lead",
    ) -> dict[str, Any]:
        """Send a follow-up prompt into the same job's isolated worktree without re-running prior prompts."""
        cont_started_mono = time.monotonic()
        safe_id = validate_job_id(job_id)
        if not follow_up_prompt or not follow_up_prompt.strip():
            raise ValueError("follow_up_prompt must be a non-empty string")
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("MCP server is shutting down; refusing to continue worker job.")
        timeout_seconds = self._clamp_worker_timeout(timeout_seconds)
        try:
            prior_rec_init = self.store.read_receipt(safe_id)
            kind_probe = prior_rec_init.get("kind")
            prior_auth_probe = str(prior_rec_init.get("auth_mode") or "")
        except FileNotFoundError:
            kind_probe = None
            prior_auth_probe = ""
        if kind_probe in ("file_batch", "draft_document", "write_tests_from_cases", "replicate_edit"):
            raise ValueError(
                f"Job '{safe_id}' is a batch job; re-send it with delegate_file_batch(resume_job_id={safe_id!r}) "
                "or fix the failed files yourself. delegate_continue only drives agent jobs."
            )
        if not force_retry:
            try:
                prior_receipt = self.store.read_receipt(safe_id)
            except FileNotFoundError:
                prior_receipt = {}
            prior_esc = prior_receipt.get("escalation") or {}
            if prior_esc.get("escalate_to_lead") and (
                prior_esc.get("reason") == "quota_or_rate_limit"
                or self._resolve_escalation_target_model(str(prior_receipt.get("model") or "")) is None
            ):
                digest = self._compact_digest(prior_receipt, include_summary=False)
                digest["status"] = "escalated"
                digest["continued"] = False
                return digest

        enforce_daily_budget(self.store, workspace_root=self.workspace_root, auth_mode=prior_auth_probe)

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
            effective_cont_allowed = (
                validate_allowed_paths(allowed_paths)
                if allowed_paths is not None
                else receipt.get("allowed_paths")
            )
            if not wt_dir.exists():
                wt_dir, baseline_sha, baseline_hashes = self._prepare_worktree(
                    safe_id,
                    allowed_paths=effective_cont_allowed,
                )
            elif receipt.get("sparse_worktree") and allowed_paths is not None:
                saved_worker_edits: dict[str, tuple[str, Any, int]] = {}
                for rel_chg in receipt.get("files_changed") or []:
                    norm_chg = str(rel_chg or "").replace("\\", "/").strip("/")
                    if not norm_chg or ".." in Path(norm_chg).parts:
                        continue
                    p_chg = wt_dir / norm_chg
                    if p_chg.is_symlink():
                        saved_worker_edits[norm_chg] = ("symlink", os.readlink(p_chg), 0)
                    elif p_chg.is_file():
                        saved_worker_edits[norm_chg] = ("file", p_chg.read_bytes(), p_chg.stat().st_mode)
                    else:
                        saved_worker_edits[norm_chg] = ("deleted", None, 0)
                merged_sparse_allowed = (
                    []
                    if effective_cont_allowed == []
                    else list(
                        dict.fromkeys(
                            [
                                *(receipt.get("allowed_paths") or []),
                                *(effective_cont_allowed or []),
                                *saved_worker_edits.keys(),
                            ]
                        )
                    )
                )
                wt_dir, baseline_sha, baseline_hashes = self._prepare_worktree(
                    safe_id,
                    allowed_paths=merged_sparse_allowed,
                )
                for norm_chg, (kind_chg, payload_chg, mode_chg) in saved_worker_edits.items():
                    dst_chg = wt_dir / norm_chg
                    if kind_chg == "symlink":
                        if dst_chg.exists() or dst_chg.is_symlink():
                            dst_chg.unlink()
                        dst_chg.parent.mkdir(parents=True, exist_ok=True)
                        os.symlink(str(payload_chg), dst_chg)
                    elif kind_chg == "file":
                        if dst_chg.is_symlink():
                            dst_chg.unlink()
                        dst_chg.parent.mkdir(parents=True, exist_ok=True)
                        dst_chg.write_bytes(payload_chg)
                        if mode_chg:
                            try:
                                os.chmod(dst_chg, mode_chg & 0o777)
                            except OSError:
                                pass
                    else:
                        if dst_chg.exists() or dst_chg.is_symlink():
                            dst_chg.unlink()
                if saved_worker_edits:
                    baseline_hashes = self._compute_changed_baseline_hashes(
                        wt_dir, baseline_sha, list(saved_worker_edits.keys()), baseline_hashes
                    )
                with self._lock:
                    is_still_sparse = safe_id in self._sparse_worktrees
                receipt["sparse_worktree"] = is_still_sparse
                receipt["baseline_commit_sha"] = baseline_sha
                receipt["baseline_file_hashes"] = baseline_hashes

        if receipt.get("cancelled") or receipt.get("status") in ("cancelled", "interrupted"):
            # A new turn supersedes a prior cancel or a server-restart interruption; otherwise
            # _execute_job's shutdown-race guard would keep relabelling every later turn. This runs only
            # after the applied / in-place checks above, so a refused continue leaves the receipt as-is.
            receipt.pop("cancelled", None)
            receipt["status"] = "running"
            receipt["error"] = None
            self.store.write_receipt(
                safe_id,
                receipt,
                scrub=bool(receipt.get("scrub_receipt", False)),
                workspace_root=str(self.workspace_root),
            )

        prior_prompts = list(receipt.get("prompts") or [])
        prior_sources = list(receipt.get("prompt_sources") or ["lead"] * len(prior_prompts))
        if len(prior_sources) < len(prior_prompts):
            prior_sources.extend(["lead"] * (len(prior_prompts) - len(prior_sources)))
        elif len(prior_sources) > len(prior_prompts):
            prior_sources = prior_sources[: len(prior_prompts)]
        clean_followup = follow_up_prompt.strip()
        recorded_prompts = prior_prompts + [clean_followup]
        recorded_prompt_sources = prior_sources + [str(_prompt_source or "lead")]

        prior_inspected_files: list[str] = []
        wt_str_prefix = str(wt_dir).rstrip("/") + "/"
        ws_str_prefix = str(self.workspace_root).rstrip("/") + "/"
        for tc in receipt.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            t_args = tc.get("args") or {}
            if not isinstance(t_args, dict):
                continue
            raw_fp = str(t_args.get("file_path") or t_args.get("path") or "").strip()
            if not raw_fp:
                continue
            for pref in (wt_str_prefix, ws_str_prefix):
                if raw_fp.startswith(pref):
                    raw_fp = raw_fp[len(pref):]
                    break
            raw_fp = raw_fp.lstrip("/")
            if raw_fp and raw_fp not in prior_inspected_files and len(prior_inspected_files) < 8:
                prior_inspected_files.append(raw_fp)

        prior_model = str(receipt.get("model") or "gemini-flash-next")
        model = prior_model
        escalated_from: str | None = None
        target_esc_model = self._resolve_escalation_target_model(model)
        prior_acc_results = [
            r for r in (receipt.get("acceptance_results") or []) if isinstance(r, dict)
        ]
        prior_only_cli_cfg_err = (
            receipt.get("status") == "ok"
            and not receipt.get("vacuous_acceptance")
            and bool(receipt.get("files_changed"))
            and bool(prior_acc_results)
            and _only_compiler_cli_config_failures(prior_acc_results)
        )
        prior_prompt_for_intent = "\n".join(
            str(x)
            for x in [
                receipt.get("raw_user_prompt") or "",
                *(prior_prompts[:1]),
            ]
            if x
        )
        prior_has_stub_mutation = bool(
            _unexempted_stub_mutation_warnings(
                receipt.get("stub_mutation_warnings"),
                prior_prompt_for_intent,
            )
        )
        prior_turn_needs_escalation = (
            (
                worker_turn_failed(receipt.get("status"), receipt.get("acceptance_passed"))
                or bool(receipt.get("vacuous_acceptance"))
                or receipt.get("status") in ("checkpoint", "needs_input", "scope_violation")
                or bool(receipt.get("missing_prompt_symbols"))
                or prior_has_stub_mutation
                or _prompt_source == "plugin_qa_review"
                or (isinstance(receipt.get("qa_review"), dict) and receipt["qa_review"].get("passed") is not True)
            )
            and receipt.get("status") != "spec_conflict"
            and not prior_only_cli_cfg_err
        )
        if target_esc_model and prior_turn_needs_escalation:
            escalated_from = str(receipt.get("escalated_from_model") or model)
            model = target_esc_model

        # Attach a concise context note of the original task and latest turn so the fresh
        # Antigravity session retains task context without re-executing prior turns.
        turn_reminders: dict[str, str] = {}
        if prior_prompts:
            orig_task = prior_prompts[0]
            last_turn = prior_prompts[-1] if len(prior_prompts) > 1 else None
            prior_summary = (
                f"original_task={orig_task!r}, latest_turn={last_turn!r}"
                if last_turn is not None
                else f"{orig_task}"
            )
            no_edits_warning = ""
            if not receipt.get("files_changed") and prior_inspected_files:
                no_edits_warning = (
                    f" IMPORTANT: The prior turn timed out before writing any file edits "
                    f"(already inspected: {prior_inspected_files}). Do NOT re-read all files from scratch; "
                    f"open at most 1-2 target sections and immediately apply your changes with edit_file / create_file."
                )
            diffstat_note = ""
            if escalated_from is not None and receipt.get("files_changed") and wt_dir.exists():
                try:
                    ds_res = _run_git(["diff", "--stat", baseline_sha], cwd=wt_dir, check=False)
                    if ds_res.returncode == 0 and ds_res.stdout.strip():
                        diffstat_note = f"\nWorktree diffstat from prior {escalated_from} attempt:\n{ds_res.stdout.strip()}"
                except (OSError, subprocess.SubprocessError):
                    diffstat_note = ""
            # Note: `thoughts_tail` and `output_tail` originate from the prior worker run in the same
            # isolated worktree (same trust class as `prior_summary`; API keys are redacted on receipt write).
            from server.agy_runner import render_keyed_turn_instructions

            _cont_thought_clip = 300
            _acc_head_clip = 180
            _acc_tail_clip = 215
            raw_prior_thoughts = [
                str(t).strip()[-_cont_thought_clip:]
                for t in (receipt.get("thoughts_tail") or [])[-2:]
                if str(t).strip()
            ]
            if raw_prior_thoughts:
                joined_thoughts = " | ".join(raw_prior_thoughts)
                turn_reminders["prior_worker_reasoning_tail"] = f"Prior worker reasoning tail: {joined_thoughts}"
            for acc_r in (receipt.get("acceptance_results") or []):
                if isinstance(acc_r, dict) and not acc_r.get("passed", True):
                    cmd_s = str(acc_r.get("command") or "").strip()
                    raw_tail = str(acc_r.get("output_tail") or "").strip()
                    if len(raw_tail) > (_acc_head_clip + 5 + _acc_tail_clip):
                        tail_s = raw_tail[:_acc_head_clip] + " ... " + raw_tail[-_acc_tail_clip:]
                    else:
                        tail_s = raw_tail
                    if cmd_s:
                        turn_reminders["prior_failed_acceptance"] = f"Prior failed acceptance `{cmd_s}`: {tail_s}"
                        break
            reminder_block = render_keyed_turn_instructions(turn_reminders)
            reminder_suffix = f"\n{reminder_block}" if reminder_block else ""
            context_note = (
                f"[Prior turn context (baseline {baseline_sha}, files_changed={receipt.get('files_changed', [])}, "
                f"run `git diff {baseline_sha}` to inspect prior work): "
                f"{prior_summary}]{no_edits_warning}{diffstat_note}{reminder_suffix}\n{clean_followup}"
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
        explicit_calls = bool(receipt.get("explicit_max_tool_calls"))
        explicit_tokens = bool(receipt.get("explicit_max_total_tokens"))
        stored_calls = (budget.get("max_tool_calls") or 0) if "max_tool_calls" in budget else None
        if escalated_from is not None and not explicit_calls:
            prior_flash_default, _ = resolve_budget(None, None, model=escalated_from)
            if stored_calls in (DEFAULT_FLASH_MAX_TOOL_CALLS, 40, 55, 75, prior_flash_default):
                stored_calls = None
        # A stored None means the lead disabled that cap; a missing key (older receipt) means defaults.
        budget_calls, budget_tokens = resolve_budget(
            stored_calls,
            (budget.get("max_total_tokens") or 0) if "max_total_tokens" in budget else None,
            model=model,
        )
        effective_max_calls = (
            max(budget_calls, int(receipt.get("cumulative_max_tool_calls") or 0))
            if budget_calls is not None
            else None
        )
        effective_max_tokens = (
            max(budget_tokens, int(receipt.get("cumulative_max_total_tokens") or 0))
            if budget_tokens is not None
            else None
        )
        prior_calls_used = max(
            int(receipt.get("tool_call_count") or 0),
            len(receipt.get("tool_calls") or []),
        )
        prior_tokens_used = int((receipt.get("usage") or {}).get("total_token_count") or 0)
        if not force_retry and _prompt_source not in ("lead", "plugin_retry", "plugin_qa_review") and (
            (
                effective_max_tokens is not None
                and prior_tokens_used >= effective_max_tokens
                and (escalated_from is None or explicit_tokens)
            )
            or (
                effective_max_calls is not None
                and prior_calls_used >= effective_max_calls
                and (escalated_from is None or explicit_calls)
            )
        ):
            if receipt.get("status") not in ("ok", "error", "cancelled", "interrupted", "spec_conflict"):
                receipt["status"] = "checkpoint"
            receipt["stop_reason"] = (
                "MAX_TOTAL_TOKENS_EXCEEDED"
                if (effective_max_tokens is not None and prior_tokens_used >= effective_max_tokens)
                else "MAX_TOOL_CALLS_EXCEEDED"
            )
            receipt["budget_exhausted"] = True
            self.store.write_receipt(
                safe_id,
                receipt,
                scrub=bool(receipt.get("scrub_receipt", False)),
                workspace_root=str(self.workspace_root),
            )
            digest = self._digest_with_progress(receipt)
            digest["continued"] = False
            digest["budget_exhausted"] = True
            return digest

        phase_budget_calls: int | None = budget_calls
        if budget_calls is not None and effective_max_calls is not None:
            rem_calls = effective_max_calls - prior_calls_used
            # A same-model plugin retry, the turn right after one, or any same-model turn after a tool-call
            # checkpoint gets a fresh `budget_calls` slice. Handing it the remainder gave a worker that
            # stopped at 39 of 40 calls exactly 1 call to finish with, and let a plugin retry eat the Lead's
            # own continue. Other continuations and escalations to another model keep the cumulative
            # remainder.
            fresh_slice = escalated_from is None and (
                _prompt_source in ("plugin_retry", "plugin_qa_review")
                or (bool(prior_sources) and prior_sources[-1] == "plugin_retry")
                or receipt.get("status") == "checkpoint"
                or receipt.get("stop_reason") == "MAX_TOOL_CALLS_EXCEEDED"
            )
            if (escalated_from is not None or _prompt_source == "plugin_qa_review") and not explicit_calls:
                # Guarantee Argon at least 60 tool calls even after a 75-call Flash checkpoint
                # (100 - 75 = 25 would starve Argon on multi-file tasks), while still bounding it.
                phase_budget_calls = max(min(60, budget_calls), rem_calls)
            elif rem_calls > 0 and not fresh_slice:
                phase_budget_calls = rem_calls
            else:
                phase_budget_calls = budget_calls

        phase_budget_tokens: int | None = budget_tokens
        if budget_tokens is not None and effective_max_tokens is not None:
            rem_tokens = effective_max_tokens - prior_tokens_used
            if escalated_from is not None and not explicit_tokens:
                phase_budget_tokens = max(min(2_500_000, budget_tokens), rem_tokens)
            elif budget_calls is not None and effective_max_calls is not None and fresh_slice:
                phase_budget_tokens = budget_tokens
            elif rem_tokens > 0:
                phase_budget_tokens = rem_tokens
            else:
                phase_budget_tokens = budget_tokens

        use_repo_map = os.environ.get("CC_AGY_REPO_MAP", "1").strip().lower() not in ("0", "false", "no")
        repo_map_text = (
            build_repo_map(self.workspace_root, cache_dir=self.store.data_dir)
            if use_repo_map
            else ""
        )
        cont_focus_targets = list(
            dict.fromkeys([
                *(receipt.get("files_changed") or []),
                *prior_inspected_files,
                *(receipt.get("allowed_paths") or []),
                *(receipt.get("preloaded_files") or []),
            ])
        )
        cont_focused_outline = (
            build_focused_file_outline(wt_dir if wt_dir.exists() else self.workspace_root, cont_focus_targets, max_tokens=500)
            if use_repo_map and cont_focus_targets
            else ""
        )
        stored_raw_user_prompt = str(receipt.get("raw_user_prompt") or "").strip()
        if stored_raw_user_prompt and stored_raw_user_prompt in clean_followup:
            stored_raw_user_prompt = ""
        stored_checklist = [
            str(item).strip()
            for item in (receipt.get("checklist") or [])
            if str(item).strip()
        ]
        stored_spec_paths = [str(p) for p in (receipt.get("spec_files") or []) if isinstance(p, str)]
        cont_spec_inlined = (
            _load_safe_workspace_files(self.workspace_root, stored_spec_paths, max_total_tokens=8000)
            if stored_spec_paths
            else []
        )
        stored_ctx_paths = [str(p) for p in (receipt.get("context_files") or []) if isinstance(p, str)]
        cont_ctx_inlined: list[tuple[str, str]] = []
        ctx_root = wt_dir if wt_dir.exists() else self.workspace_root
        spec_set_cont = {r for r, _ in cont_spec_inlined}
        if stored_ctx_paths and (
            _prompt_source == "plugin_escalation"
            or escalated_from is not None
            or not receipt.get("files_changed")
        ):
            cont_ctx_inlined = _load_safe_workspace_files(
                ctx_root,
                [p for p in stored_ctx_paths if p not in spec_set_cont],
                max_total_tokens=4000,
            )
        if receipt.get("files_changed") and wt_dir.exists():
            inlined_rel_set = {r for r, _ in cont_ctx_inlined} | spec_set_cont
            mod_impl_paths = [
                str(p)
                for p in (receipt.get("files_changed") or [])
                if isinstance(p, str)
                and p not in inlined_rel_set
                and not _is_test_file_path(p)
            ][:3]
            if mod_impl_paths:
                cont_ctx_inlined.extend(
                    _load_safe_workspace_files(
                        wt_dir,
                        mod_impl_paths,
                        max_total_tokens=3500,
                    )
                )
        prior_baseline_probe = (
            list(receipt.get("baseline_acceptance_probe"))
            if isinstance(receipt.get("baseline_acceptance_probe"), list)
            else []
        )
        prior_acceptance_remapped = (
            list(receipt.get("acceptance_remapped"))
            if (
                (acceptance is None or list(acceptance) == list(receipt.get("acceptance") or []))
                and isinstance(receipt.get("acceptance_remapped"), list)
            )
            else []
        )
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
            rec_proj = receipt.get("vertex_project")
            if rec_proj == "<paying-project>":
                rec_proj = None
            inherited_project = raw_proj or rec_proj
        if prior_auth_mode == "gemini_enterprise" or is_ge_mode_pinned():
            if prior_auth_mode in ("api_key", "vertex") and is_ge_mode_pinned():
                raise ValueError(
                    f"Gemini Enterprise mode is now pinned, so pre-pin {prior_auth_mode!r} job '{safe_id}' "
                    "cannot be continued on a pay-as-you-go backend; start a new Gemini Enterprise job instead."
                )
            if api_key and str(api_key).strip():
                raise ValueError(
                    "Gemini Enterprise job cannot be continued with a per-call api_key override "
                    "to prevent out-of-pool billing."
                )
            inherited_auth_mode: str | None = "gemini_enterprise"
        else:
            inherited_auth_mode = (
                str(prior_auth_mode)
                if prior_auth_mode in ("api_key", "vertex") and not project and not api_key
                else None
            )
        auth = resolve_auth_config(
            requested_project=inherited_project,
            requested_location=location or receipt.get("vertex_location"),
            requested_api_key=api_key or raw_key,
            requested_auth_mode=inherited_auth_mode,
            model=model,
        )
        auth_mode = auth["auth_mode"]
        enforce_daily_budget(self.store, workspace_root=self.workspace_root, auth_mode=auth_mode)
        resolved_key = auth.get("api_key")
        resolved_project = auth.get("project")
        resolved_location = auth.get("location")
        reroute_note = auth.get("reroute_note")

        resumed_agy_conv_id: str | None = None
        prior_agy_conv_usage: dict[str, Any] | None = None
        prior_agy_conv_turns: int | None = None
        if auth_mode == "gemini_enterprise" and receipt.get("agy_conversation_id"):
            resumed_agy_conv_id = str(receipt["agy_conversation_id"]).strip() or None
            if resumed_agy_conv_id:
                if isinstance(receipt.get("agy_conversation_usage"), dict):
                    prior_agy_conv_usage = dict(receipt["agy_conversation_usage"])
                elif isinstance(receipt.get("usage"), dict):
                    prior_agy_conv_usage = dict(receipt["usage"])
                prior_agy_conv_turns = int(receipt.get("agy_conversation_turns") or max(1, len(prior_prompts)))

        worker_prompts = [
            render_worker_prompt(
                clean_followup if resumed_agy_conv_id else worker_prompts[0],
                plan=[] if resumed_agy_conv_id else plan_steps,
                acceptance=acceptance_cmds,
                max_tool_calls=phase_budget_calls,
                max_total_tokens=phase_budget_tokens,
                repo_map="" if resumed_agy_conv_id else repo_map_text,
                focused_outline=cont_focused_outline,
                spec_files_inlined=[] if resumed_agy_conv_id else cont_spec_inlined,
                context_files_inlined=cont_ctx_inlined if not resumed_agy_conv_id else [],
                checklist=[] if resumed_agy_conv_id else stored_checklist,
                raw_user_prompt="" if resumed_agy_conv_id else stored_raw_user_prompt,
                baseline_acceptance_probe=prior_baseline_probe,
                allowed_paths=list(receipt.get("allowed_paths") or []),
            )
        ]

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
        if escalated_from is not None:
            receipt["escalated_from_model"] = escalated_from
            receipt["model"] = model
            receipt["failed_attempts"] = 0
            receipt.pop("escalation", None)
            try:
                self.store.write_receipt(
                    safe_id,
                    receipt,
                    scrub=bool(receipt.get("scrub_receipt", False)),
                    workspace_root=str(self.workspace_root),
                )
            except (OSError, ValueError, KeyError):
                pass
        enabled_tools = receipt.get("enabled_tools")
        disabled_tools = receipt.get("disabled_tools")
        enable_subagents = bool(receipt.get("enable_subagents", False))
        eff_scrub = bool(scrub_receipt) if scrub_receipt is not None else bool(receipt.get("scrub_receipt", False))
        started_at = float(receipt.get("started_at_epoch") or time.time())
        prior_usage_snapshot = dict(receipt.get("usage") or {})
        if (
            not prior_usage_snapshot.get("requests")
            and any(
                int(prior_usage_snapshot.get(k) or 0) > 0
                for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")
            )
        ):
            prior_usage_snapshot["requests"] = [
                {
                    "prompt_token_count": int(prior_usage_snapshot.get("prompt_token_count") or 0),
                    "cached_content_token_count": int(prior_usage_snapshot.get("cached_content_token_count") or 0),
                    "candidates_token_count": int(prior_usage_snapshot.get("candidates_token_count") or 0),
                    "thoughts_token_count": int(prior_usage_snapshot.get("thoughts_token_count") or 0),
                    "total_token_count": int(prior_usage_snapshot.get("total_token_count") or 0),
                    "model": prior_model,
                }
            ]
        elif isinstance(prior_usage_snapshot.get("requests"), list):
            tagged_prior_reqs = [
                ({**r, "model": r.get("model") or prior_model} if isinstance(r, dict) else r)
                for r in prior_usage_snapshot["requests"]
            ]
            sum_pp = sum(int(r.get("prompt_token_count") or 0) for r in tagged_prior_reqs if isinstance(r, dict))
            sum_pc = sum(int(r.get("cached_content_token_count") or 0) for r in tagged_prior_reqs if isinstance(r, dict))
            sum_pcand = sum(int(r.get("candidates_token_count") or 0) for r in tagged_prior_reqs if isinstance(r, dict))
            sum_pth = sum(int(r.get("thoughts_token_count") or 0) for r in tagged_prior_reqs if isinstance(r, dict))
            rem_pp = max(0, int(prior_usage_snapshot.get("prompt_token_count") or 0) - sum_pp)
            rem_pc = min(rem_pp, max(0, int(prior_usage_snapshot.get("cached_content_token_count") or 0) - sum_pc))
            rem_pcand = max(0, int(prior_usage_snapshot.get("candidates_token_count") or 0) - sum_pcand)
            rem_pth = max(0, int(prior_usage_snapshot.get("thoughts_token_count") or 0) - sum_pth)
            if rem_pp > 0 or rem_pcand > 0 or rem_pth > 0:
                tagged_prior_reqs.append(
                    {
                        "prompt_token_count": rem_pp,
                        "cached_content_token_count": rem_pc,
                        "candidates_token_count": rem_pcand,
                        "thoughts_token_count": rem_pth,
                        "total_token_count": rem_pp + rem_pcand + rem_pth,
                        "model": prior_model,
                    }
                )
            prior_usage_snapshot["requests"] = tagged_prior_reqs
        prior_tool_calls_snapshot = list(receipt.get("tool_calls") or [])
        explicit_thinking = bool(receipt.get("explicit_thinking_level"))
        if escalated_from is not None and not explicit_thinking:
            prior_thinking = (
                resolve_thinking_level(
                    None,
                    model=model,
                    read_only=is_read_only_tool_set(enabled_tools, disabled_tools),
                )
                or "DEFAULT"
            )
        else:
            prior_thinking = receipt.get("thinking_level")
        prior_preloaded = receipt.get("preloaded_files")
        cumulative_max_calls = (
            max(effective_max_calls, prior_calls_used + phase_budget_calls)
            if (effective_max_calls is not None and phase_budget_calls is not None)
            else budget_calls
        )
        cumulative_max_tokens = (
            max(effective_max_tokens, prior_tokens_used + phase_budget_tokens)
            if (effective_max_tokens is not None and phase_budget_tokens is not None)
            else budget_tokens
        )
        prior_attempts = list(receipt.get("prior_attempts") or [])
        prior_attempts.append(
            {
                "attempt": max(1, len(recorded_prompt_sources) - 1),
                "model": prior_model,
                "status": receipt.get("status"),
                "stop_reason": receipt.get("stop_reason"),
                "acceptance_passed": receipt.get("acceptance_passed"),
                "tool_call_count": len(prior_tool_calls_snapshot),
                "thoughts_tail": list(receipt.get("thoughts_tail") or [])[-4:],
                "summary": str(receipt.get("summary") or "")[:1500],
                "acceptance_results": list(receipt.get("acceptance_results") or [])[:5],
            }
        )
        receipt["prior_attempts"] = prior_attempts[-5:]
        receipt["prior_tool_call_count"] = len(prior_tool_calls_snapshot)
        receipt["status"] = "running"
        receipt["error"] = None
        receipt["stop_reason"] = None
        receipt.pop("denied_actions", None)
        receipt["completed_at"] = None
        receipt["finished_at_epoch"] = None
        receipt["acceptance_passed"] = None
        receipt["acceptance_results"] = []
        receipt["summary"] = ""
        receipt["summary_truncated"] = False
        receipt["missing_prompt_symbols"] = []
        receipt["contract_warnings"] = []
        receipt.pop("budget_exhausted", None)
        receipt.pop("vacuous_acceptance", None)
        receipt.pop("test_only_diff", None)
        receipt.pop("stub_mutation_warnings", None)
        receipt.pop("recovered_from_timeout", None)
        receipt.pop("recovered_from_checkpoint", None)
        receipt["model"] = model
        receipt["thinking_level"] = prior_thinking
        receipt["prompts"] = recorded_prompts
        receipt["prompt_sources"] = recorded_prompt_sources
        receipt["attempt"] = len(recorded_prompt_sources)
        receipt["attempt_started_at_epoch"] = round(time.time(), 3)
        receipt["owner_instance_id"] = self._instance_id
        # Set-once session provenance: record the first OpenCode session ID seen on this job
        if opencode_session_id and not receipt.get("opencode_session_id"):
            receipt["opencode_session_id"] = opencode_session_id
        receipt["budget"] = {
            "max_tool_calls": budget_calls,
            "max_total_tokens": budget_tokens,
        }
        receipt["explicit_max_tool_calls"] = explicit_calls
        receipt["explicit_max_total_tokens"] = explicit_tokens
        receipt["explicit_thinking_level"] = explicit_thinking
        if cumulative_max_calls is not None and budget_calls is not None and cumulative_max_calls > budget_calls:
            receipt["cumulative_max_tool_calls"] = cumulative_max_calls
        if cumulative_max_tokens is not None and budget_tokens is not None and cumulative_max_tokens > budget_tokens:
            receipt["cumulative_max_total_tokens"] = cumulative_max_tokens
        try:
            self.store.write_receipt(
                safe_id,
                receipt,
                scrub=bool(receipt.get("scrub_receipt", False)),
                workspace_root=str(self.workspace_root),
            )
        except (OSError, ValueError, KeyError):
            pass

        with self._lock:
            if self._shutting_down:
                if in_place:
                    self._release_in_place_slot(safe_id)
                receipt["status"] = "interrupted"
                receipt["error"] = "MCP server is shutting down; continuation turn was not started."
                try:
                    self.store.write_receipt(
                        safe_id,
                        receipt,
                        scrub=bool(receipt.get("scrub_receipt", False)),
                        workspace_root=str(self.workspace_root),
                    )
                except (OSError, ValueError, KeyError):
                    pass
                raise RuntimeError("MCP server is shutting down; refusing to continue worker job.")
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
                prior_usage=prior_usage_snapshot,
                prior_tool_calls=prior_tool_calls_snapshot,
                scrub_receipt=eff_scrub,
                in_place=in_place,
                pre_untracked=pre_untracked,
                thinking_level=prior_thinking,
                plan=plan_steps,
                acceptance=acceptance_cmds,
                max_tool_calls=phase_budget_calls,
                max_total_tokens=phase_budget_tokens,
                preloaded_files=prior_preloaded,
                prompt_sources=recorded_prompt_sources,
                recorded_budget={
                    "max_tool_calls": budget_calls,
                    "max_total_tokens": budget_tokens,
                },
                baseline_acceptance_probe=prior_baseline_probe,
                acceptance_remapped=prior_acceptance_remapped,
                agy_conversation_id=resumed_agy_conv_id,
                agy_prior_conversation_usage=prior_agy_conv_usage,
                agy_prior_conversation_turns=prior_agy_conv_turns,
                turn_instructions=turn_reminders,
            )
            self._futures[safe_id] = fut
            self._owned_job_ids.add(safe_id)

        if callable(on_job_allocated):
            try:
                on_job_allocated(safe_id)
            except Exception:
                pass
        if stop_event is not None and stop_event.is_set():
            self.delegate_cancel(safe_id)

        if wait:
            slice_target = float(_wait_budget) if _wait_budget is not None else float(timeout_seconds)
            total_elapsed = max(0.0, float(_wait_elapsed) + max(0.0, time.monotonic() - cont_started_mono))
            wait_slice = self._resolve_wait_slice_seconds(
                slice_target,
                elapsed_seconds=total_elapsed,
                allow_retry_grace=(_prompt_source in ("plugin_retry", "plugin_escalation", "plugin_qa_review")),
            )
            if _wait_budget is not None and wait_slice < 1.0 and float(_wait_budget) <= 30.0:
                wait_slice = max(wait_slice, float(_wait_budget) - float(_wait_elapsed))
            if progress_callback is None and stop_event is None:
                self._wait_for_future_or_cancel(safe_id, fut, timeout=wait_slice)
            else:
                self._wait_for_future_or_cancel(
                    safe_id,
                    fut,
                    timeout=wait_slice,
                    progress_callback=progress_callback,
                    stop_event=stop_event,
                    max_tool_calls=phase_budget_calls,
                )
            updated = self.store.read_receipt(safe_id)
            if not fut.done() or updated.get("status") == "running":
                with self._lock:
                    self._wait_sliced_auto_apply[safe_id] = auto_apply
                return self._digest_with_progress(updated)
            digest = self._digest_with_progress(updated)
            if updated.get("escalated_from_model") or (
                _prompt_source == "plugin_retry" and env_flag("CC_AGY_AUTONOMOUS_SOLO") is True
            ):
                # Escalated or autonomous-solo retry turns get one audit of their final diff before auto-apply.
                post_esc_digest = self._maybe_run_balanced_pre_apply_qa(
                    safe_id,
                    updated,
                    auto_apply=auto_apply,
                    in_place=in_place,
                    is_cancelled_now=bool(updated.get("cancelled")),
                    wait_elapsed=max(0.0, float(_wait_elapsed) + max(0.0, time.monotonic() - cont_started_mono)),
                    timeout_seconds=timeout_seconds,
                    wait_budget=_wait_budget,
                )
                if post_esc_digest is not None:
                    return post_esc_digest
                digest = self._digest_with_progress(self.store.read_receipt(safe_id))
            return self._maybe_auto_apply_digest(
                safe_id,
                digest,
                auto_apply=auto_apply,
                in_place=in_place,
            )

        return {
            "job_id": safe_id,
            "status": "running",
            "auth_mode": auth_mode,
            "prompts_count": len(recorded_prompts),
            **(
                {"opencode_session_id": str(receipt["opencode_session_id"])}
                if receipt.get("opencode_session_id")
                else {}
            ),
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


_EXTRACTED_JOB_SUBMODULES: tuple[str, ...] = (
    "server.job_git_utils",
    "server.worker_prompt_builder",
    "server.ast_regression_guards",
    "server.test_discovery",
    "server.acceptance_runner",
)


def __getattr__(name: str) -> Any:
    """Resolve private helpers moved into `server.job_*` / `server.*` domain modules."""
    import importlib

    for mod_name in _EXTRACTED_JOB_SUBMODULES:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, name):
            val = getattr(mod, name)
            globals()[name] = val
            return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

