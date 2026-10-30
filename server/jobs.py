"""Isolated job manager for Antigravity SDK delegations."""

from __future__ import annotations

import ast
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
    has_real_acceptance_pass,
    is_non_smoke_passing_entry,
    satisfies_auto_apply_verification,
    satisfies_timeout_promotion,
)
from server.regions import (
    DEFAULT_DIFF_PAGE_TOKENS,
    MAX_TOOL_RESPONSE_TOKENS,
    estimate_tokens,
    env_flag,
    is_strict_mode_enabled,
    paginate_text_by_tokens,
    resolve_api_key,
    resolve_auth_config,
    resolve_cost_profile,
    resolve_default_delegate_model,
    truncate_to_token_budget,
)


def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    # Hooks, fsmonitor, external diff/textconv, and clean/smudge filters stay off and the environment is
    # credential-free: worker or check commands can rewrite a job copy's .git/config or hooks.
    from server.verify import run_git

    return run_git(
        ["-c", "core.quotePath=false", *args],
        cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _git_raw(args: list[str], cwd: Path | str | None = None, **kwargs: Any) -> subprocess.CompletedProcess:
    """A plugin git call with the caller's own ``subprocess.run`` options, guarded like ``_run_git``
    (see ``verify.run_git``)."""
    from server.verify import run_git

    return run_git(args, cwd, **kwargs)


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


_EPHEMERAL_CACHE_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".hypothesis",
        ".benchmarks",
        ".mypy_cache",
        ".ruff_cache",
        ".pytype",
        ".eggs",
        "node_modules",
        ".vite",
        ".vite-temp",
        ".eslintcache",
        ".tox",
        ".nox",
        ".venv",
        ".gradle",
        ".next",
        ".nuxt",
        ".ctx-session-reads",
        ".cc-agy-review",
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
    ".tsbuildinfo",
)

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
    except Exception:
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

# Report files that test runners write at the repo root (pytest-cov, coverage.py, nose). The
# host and the worker both produce them when they run the same tests, so capturing one turns
# into a "created on both sides" apply conflict that throws away the worker's real edits.
# Only matched at the repo root, so fixtures such as tests/fixtures/coverage.xml still apply.
_GENERATED_REPORT_NAMES: frozenset[str] = frozenset(
    {
        "coverage.xml",
        "coverage.json",
        "coverage.lcov",
        "lcov.info",
        "nosetests.xml",
    }
)
_FIXTURE_PATH_SEGMENTS: frozenset[str] = frozenset(
    {"testdata", "fixtures", "fixture", "snapshots", "__snapshots__", "golden"}
)
_UNTRACKED_MODEL_OR_DATA_DUMP_SUFFIXES: tuple[str, ...] = (
    ".joblib",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".h5",
    ".onnx",
    ".parquet",
)


def _is_untracked_runtime_artifact_name(rel_path: str) -> bool:
    """Return True for transient runtime log, pid, or serialized model/dataset dump files unless in a fixture dir."""
    norm = rel_path.replace("\\", "/").strip("/")
    if not norm:
        return False
    parts = norm.split("/")
    if any(p.lower() in _FIXTURE_PATH_SEGMENTS for p in parts[:-1]):
        return False
    low_name = parts[-1].lower()
    if low_name.endswith((".log", ".pid", ".pid.lock", *_UNTRACKED_MODEL_OR_DATA_DUMP_SUFFIXES)):
        return True
    return False


_GENERIC_MODULE_STEMS: frozenset[str] = frozenset(
    {"__init__", "init", "index", "main", "test", "tests", "utils", "common", "base", "types", "config", "constants"}
)
_IMPORTABLE_MODULE_EXTS: frozenset[str] = frozenset(
    {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs", ".java", ".kt", ".c", ".cc", ".cpp", ".h", ".hpp", ".rb", ".php", ".cs", ".swift"}
)


def _is_referenced_by_files_or_prompt(
    rel_cand: str,
    ref_files: list[str] | None,
    root: Path | None,
    *,
    prompt_text: str = "",
) -> bool:
    """Return True if `rel_cand` (or its basename with extension, or its module stem for importable code files) is referenced in `prompt_text` or `ref_files`."""
    cand_norm = rel_cand.replace("\\", "/").strip("/")
    if not cand_norm:
        return False
    cand_name = Path(cand_norm).name
    cand_sfx = Path(cand_norm).suffix.lower()
    cand_stem = Path(cand_norm).stem
    cand_no_ext = str(Path(cand_norm).with_suffix("")).replace("\\", "/")
    cand_dotted = cand_no_ext.replace("/", ".")
    prompt_low = (prompt_text or "").lower()
    if prompt_low:
        if cand_norm.lower() in prompt_low or (len(cand_name) >= 4 and "." in cand_name and cand_name.lower() in prompt_low):
            return True
    if not ref_files or root is None or not root.exists():
        return False
    check_stem = (
        cand_sfx in _IMPORTABLE_MODULE_EXTS
        and not _is_untracked_runtime_artifact_name(cand_norm)
        and len(cand_stem) >= 3
        and cand_stem.lower() not in _GENERIC_MODULE_STEMS
    )
    stem_re = re.compile(rf"\b{re.escape(cand_stem)}\b") if check_stem else None
    for rf in ref_files:
        rf_norm = str(rf or "").replace("\\", "/").strip("/")
        if not rf_norm or rf_norm == cand_norm:
            continue
        if cand_sfx == ".go" and Path(rf_norm).suffix.lower() == ".go" and Path(rf_norm).parent == Path(cand_norm).parent:
            return True
        rf_p = root / rf_norm
        try:
            if not rf_p.is_file() or rf_p.is_symlink():
                continue
            with rf_p.open("rb") as fh:
                raw_bytes = fh.read(262_144)
            txt = raw_bytes.decode("utf-8", errors="ignore")
        except OSError:
            continue
        if cand_norm in txt or ("." in cand_name and cand_name in txt):
            return True
        if check_stem and (
            cand_dotted in txt
            or cand_no_ext in txt
            or (stem_re is not None and stem_re.search(txt))
        ):
            return True
    return False


def _explicitly_targets_runtime_artifact(
    rel_cand: str,
    allowed_paths: list[str] | None,
    *,
    prompt_text: str = "",
    ref_files: list[str] | None = None,
    root: Path | None = None,
) -> bool:
    """Return True when `allowed_paths`, `prompt_text`, or `ref_files` explicitly names or references `rel_cand`."""
    if not _is_untracked_runtime_artifact_name(rel_cand):
        return False
    cand_norm = rel_cand.replace("\\", "/").strip("/")
    cand_sfx = Path(cand_norm).suffix.lower()
    for pat in allowed_paths or []:
        p_norm = str(pat or "").replace("\\", "/").strip("/")
        if not p_norm or p_norm in (".", "*", "**", "**/*"):
            continue
        if (
            p_norm == cand_norm
            or (
                cand_sfx
                and p_norm.lower().endswith(cand_sfx)
                and _matches_allowed_paths(cand_norm, [p_norm])
            )
            or (
                "/" in cand_norm
                and p_norm.split("/", 1)[0] == cand_norm.split("/", 1)[0]
                and _matches_allowed_paths(cand_norm, [p_norm])
            )
        ):
            return True
    if (prompt_text or ref_files) and _is_referenced_by_files_or_prompt(
        cand_norm, ref_files, root, prompt_text=prompt_text
    ):
        return True
    return False


def _is_tracked_in_git(root: Path, rel_path: str) -> bool:
    try:
        res = _git_raw(
            ["ls-files", "--error-unmatch", "--", rel_path],
            cwd=str(root),
            capture_output=True,
            check=False,
            timeout=5,
        )
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


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
        or (len(parts) == 1 and name in _GENERATED_REPORT_NAMES)
        or name.startswith(".coverage.")
        or name.endswith(_EPHEMERAL_CACHE_SUFFIXES)
    ):
        return True
    if root is not None and _is_untracked_runtime_artifact_name(norm):
        if not _is_tracked_in_git(root, norm):
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


def _is_gitignored_on_host(workspace_root: Path, rel_path: str) -> bool:
    """True if `rel_path` is ignored by the host repo's .gitignore rules (False when not a git repo)."""
    try:
        res = _git_raw(
            ["check-ignore", "-q", "--", rel_path],
            cwd=str(workspace_root),
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0


def _is_ignored_build_artifact(rel_path: str, root: Path | None = None) -> bool:
    """Return True if an already-.gitignored path is inside a build/output/cache directory or a setuptools-scm _version.py."""
    if _is_ephemeral_build_or_test_artifact(rel_path, root=root):
        return True
    norm = rel_path.strip("/")
    if not norm:
        return False
    parts = norm.split("/")
    for part in parts:
        if part in _IGNORED_BUILD_OUTPUT_DIRS:
            return True
    if parts[-1] in ("_version.py", "__version__.py"):
        return True
    return False


# DEFAULT_MAX_TOOL_CALLS, DEFAULT_FLASH_MAX_TOOL_CALLS, DEFAULT_ECONOMY_FLASH_MAX_TOOL_CALLS,
# and DEFAULT_MAX_TOTAL_TOKENS are imported from server.model_routes (single source of truth).
DEFAULT_MAX_WAIT_SLICE_SECONDS = 240.0
DEFAULT_ACCEPTANCE_TIMEOUT = 300.0
DEFAULT_BOUNDED_ACCEPTANCE_TIMEOUT = 90.0
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
    max_tool_calls: int | None,
    max_total_tokens: int | None,
    *,
    model: str | None = None,
) -> tuple[int | None, int | None]:
    """Per-turn worker caps (defaults 55 tool calls on Flash in balanced, 100 on Flash in economy and on Pro/Argon/Carbon,
    5M tokens; env overrides; 0 disables)."""
    if max_tool_calls is not None:
        calls = max_tool_calls
    elif "flash" in (model or "").lower() and not str(os.environ.get("CC_AGY_MAX_TOOL_CALLS") or "").strip():
        flash_default = (
            DEFAULT_ECONOMY_FLASH_MAX_TOOL_CALLS
            if resolve_cost_profile() == "economy"
            else DEFAULT_FLASH_MAX_TOOL_CALLS
        )
        calls = _env_int("CC_AGY_FLASH_MAX_TOOL_CALLS", flash_default)
    else:
        calls = _env_int("CC_AGY_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS)
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


_GO_BUILD_HEAD_RE = re.compile(
    r"^([\s(]*(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s;|&()]+)\s+)*"
    r"(?:\S*/)?go(?:\s+(?:-C|--C)(?:=(?:\"[^\"]*\"|'[^']*'|[^\s;|&()]+)|\s+(?:\"[^\"]*\"|'[^']*'|[^\s;|&()]+)))*"
    r"\s+build)(?=\s|$|\))"
)


def _normalize_readonly_go_build_acceptance(cmd: str) -> str:
    """Insert `-o /dev/null` into bare `go build ...` segments (with no `-o`/`--o`) without re-quoting shell syntax."""
    parts = re.split(r"(\s*(?:&&|\|\||;|\||\n)\s*)", cmd)
    changed = False
    rebuilt: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1 or not part.strip():
            rebuilt.append(part)
            continue
        m = _GO_BUILD_HEAD_RE.match(part)
        if not m:
            rebuilt.append(part)
            continue
        rest = part[m.end() :]
        rest_for_tokens = re.sub(r"\s*(?:\d*>>?\s*(?:&\d+|\S+)|\d*<\s*\S+)\s*", " ", rest.rstrip(") \t"))
        try:
            rest_tokens = shlex.split(rest_for_tokens)
        except ValueError:
            rest_tokens = rest_for_tokens.split()
        if any(a in ("-o", "--o") or a.startswith(("-o=", "--o=")) for a in rest_tokens):
            rebuilt.append(part)
            continue
        rebuilt.append(f"{m.group(1)} -o /dev/null{rest}")
        changed = True
    return "".join(rebuilt) if changed else cmd


def validate_acceptance(acceptance: Any, workdir: Path) -> list[str]:
    """Acceptance commands must be read-only checks (tests, linters); mutating commands are refused.

    The plugin itself runs them, so a mutating command would be an unguarded way to edit the repo.
    """
    if acceptance is None:
        return []
    if not isinstance(acceptance, list) or not all(isinstance(c, str) for c in acceptance):
        raise ValueError("acceptance must be a list of shell command strings")
    cmds = [_normalize_readonly_go_build_acceptance(c.strip()) for c in acceptance if c.strip()]
    if len(cmds) > MAX_ACCEPTANCE_COMMANDS:
        raise ValueError(f"acceptance has {len(cmds)} commands; at most {MAX_ACCEPTANCE_COMMANDS} allowed")
    from scripts.pre_tool_guard import detect_bash_workspace_mutation

    for cmd in cmds:
        verdict = detect_bash_workspace_mutation(
            cmd, workdir, cwd=workdir, allow_workspace_inline_imports=True
        )
        if verdict.get("blocked"):
            raise ValueError(
                f"acceptance command {cmd!r} is refused because it can modify the workspace: "
                f"{verdict.get('reason')}"
            )
    return cmds


_PRELOAD_SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".pdf",
    ".whl", ".tar", ".gz", ".zip", ".pyc", ".so", ".dylib", ".lock",
}
_PRELOAD_BOILERPLATE_STEMS = frozenset({
    "license", "licence", "copying", "notice", "dockerfile", "makefile", "gnumakefile",
    "changelog", "changes", "history", "authors", "contributors", "contributing",
    "code_of_conduct", "security",
})
_PRELOAD_BOILERPLATE_EXACT = frozenset({
    ".dockerignore", ".editorconfig", ".gitattributes", ".gitignore",
    ".prettierrc", ".eslintrc", ".flake8", ".yamllint",
})
_PRELOAD_ENTRYPOINT_NAMES = frozenset({
    "__main__.py", "cli.py", "main.py", "app.py", "main.go", "root.go",
    "index.ts", "index.tsx", "index.js", "main.ts", "cli.ts", "main.rs", "lib.rs",
})


def _is_preload_boilerplate(rel_path: str, prompt_low: str) -> bool:
    """Exclude license/Docker/Makefile/README/changelog boilerplate unless explicitly referenced in the prompt."""
    norm_low = rel_path.replace("\\", "/").strip("/").lower()
    bname = norm_low.rsplit("/", 1)[-1]
    if norm_low in prompt_low or bname in prompt_low:
        return False
    if bname in _PRELOAD_BOILERPLATE_EXACT:
        return True
    stem = bname.split(".", 1)[0]
    if stem in _PRELOAD_BOILERPLATE_STEMS:
        return True
    if stem == "readme" and "readme" not in prompt_low:
        return True
    return False

# Reviewer context cap for qa_review_model diff audits.
QA_REVIEW_MAX_TOKENS = 25_000


def escalate_after_failures() -> int:
    """Consecutive failed worker turns before the job is handed back to the Lead (default 2)."""
    raw = os.environ.get("CC_AGY_ESCALATE_AFTER_FAILURES", "2").strip()
    try:
        val = int(raw)
    except ValueError:
        val = 2
    return max(0, val)


def worker_turn_failed(status: str | None, acceptance_passed: bool | None) -> bool:
    """A turn fails when its acceptance checks fail or the worker errors or times out."""
    return acceptance_passed is False or status in ("error", "timeout")

_PROMPT_ASKS_FOR_TESTS_RE = re.compile(
    r"\b(?:unit\s+tests?|integration\s+tests?|test\s+cases?|test\s+suite|write\s+tests?|add\s+tests?|pytest|vitest|jest)\b",
    re.IGNORECASE,
)

_QUOTA_OR_RATE_LIMIT_RE = re.compile(
    r"(?:\b429\b|RESOURCE_EXHAUSTED|QUOTA_EXHAUSTED|\brate[- ]limit(?:ed|s)?\b|\b503\b.*?\b(?:UNAVAILABLE|overloaded)\b|\bUNAVAILABLE\b.*?\b503\b)",
    re.IGNORECASE | re.DOTALL,
)


def is_quota_or_rate_limit_error(error_text: str | None) -> bool:
    """Return True when an error string indicates a Gemini 429/503 quota or rate-limit failure."""
    return bool(error_text and _QUOTA_OR_RATE_LIMIT_RE.search(error_text))


QA_DIFF_AUDIT_SYSTEM = """You are a senior software engineer performing a 1-shot blocker-only code review of a git diff against a task specification.
Report ONLY High or Medium functional blockers:
1. Missing requirements, cross-module wiring, or edge cases explicitly stated in the task specification (check every branch and requirement in the Original User Specification even if abbreviated in the Lead brief).
2. Subtle logic bugs, off-by-one errors, state leaks, security bugs, or broken imports/exports.
3. Weakened or altered pre-existing test assertions/expected outputs made to match broken behavior instead of fixing production code, or missing unit test coverage when the repository has automated tests.

Do NOT report Low, Optional, polish, naming, style, formatting, or future-hardening suggestions.
Output format:
- If there are no High or Medium blockers, start with `VERDICT: PASS` on line 1.
- If there is at least one concrete blocker, start with `VERDICT: NEEDS_FIXES` on line 1, followed by concise bullet points (`- ISSUE: <file:line>: <exact bug and how to fix it>`)."""


def build_qa_diff_audit_system(*, verification: str | None = None, prompt: str = "") -> str:
    """Build the 1-shot QA review system prompt, skipping test-coverage demands on testless repos."""
    ver = str(verification or "").strip().lower()
    if ver in ("smoke_only", "unverified") and not _PROMPT_ASKS_FOR_TESTS_RE.search(prompt or ""):
        return (
            "You are a senior software engineer performing a 1-shot blocker-only code review of a git diff against a task specification.\n"
            "Report ONLY High or Medium functional blockers:\n"
            "1. Missing requirements or edge cases explicitly stated in the task specification (check every branch and requirement in the Original User Specification even if abbreviated in the Lead brief).\n"
            "2. Subtle logic bugs, off-by-one errors, state leaks, security bugs, or broken imports/exports.\n\n"
            f"This repository has no automated test suite covering this change (verification={ver!r}). "
            "Do NOT fail the diff for missing unit tests unless the task specification explicitly asked to add tests.\n"
            "Do NOT report Low, Optional, polish, naming, style, formatting, or future-hardening suggestions.\n"
            "Output format:\n"
            "- If there are no High or Medium blockers, start with `VERDICT: PASS` on line 1.\n"
            "- If there is at least one concrete blocker, start with `VERDICT: NEEDS_FIXES` on line 1, followed by concise bullet points (`- ISSUE: <file:line>: <exact bug and how to fix it>`)."
        )
    return QA_DIFF_AUDIT_SYSTEM


def parse_qa_audit_verdict(text: str) -> dict[str, Any]:
    """Parse a 1-shot QA review response into verdict, passed boolean, and issue bullets."""
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    # Reviewers sometimes add a preamble ("Reviewed the diff.", "Format reminder: VERDICT: PASS means ...")
    # before the real verdict, so prefer the LAST line that starts with a VERDICT marker (markdown
    # emphasis allowed), then fall back to any line mentioning it, then line 0.
    starts = [i for i, ln in enumerate(lines) if re.match(r"^[\s*#>_`]*VERDICT:", ln, re.IGNORECASE)]
    mentions = [i for i, ln in enumerate(lines) if "VERDICT:" in ln.upper()]
    verdict_idx = starts[-1] if starts else (mentions[0] if mentions else 0)
    first = lines[verdict_idx].upper() if lines else ""
    passed = "VERDICT: PASS" in first and "NEEDS_FIXES" not in first and "FAIL" not in first
    rest = lines[verdict_idx + 1:]
    issues = [
        ln
        for ln in rest
        if re.match(r"^(?:[-*•]|\d+[.)])\s+", ln) or "ISSUE:" in ln.upper()
    ]
    # Fail closed: an explicit ISSUE: line anywhere contradicts a PASS verdict.
    if passed and any("ISSUE:" in ln.upper() for ln in lines):
        passed = False
    if not passed and not issues and len(rest) > 0:
        issues = [ln for ln in rest if not ln.upper().startswith("VERDICT:")]
    return {
        "verdict": "PASS" if passed else "NEEDS_FIXES",
        "passed": passed,
        "issues": issues,
    }


def select_candidate_files(
    workspace_root: Path,
    task_prompt: str,
    *,
    max_files: int = 12,
    token_budget: int = 120_000,
) -> list[tuple[str, str]]:
    """Rank tracked repository files by keyword overlap with task_prompt and return (rel_path, content) pairs."""
    res = _git_raw(
        ["ls-files"],
        cwd=str(workspace_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    all_files = [
        f.strip()
        for f in res.stdout.splitlines()
        if f.strip() and not _is_ephemeral_build_or_test_artifact(f.strip(), root=workspace_root)
    ]
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", task_prompt)
    keywords = {t.lower().strip("._/-") for t in raw_tokens if len(t) >= 3}
    prompt_low = task_prompt.lower()

    scored: list[tuple[float, str, str]] = []
    for rel in all_files[:5000]:
        p = workspace_root / rel
        if p.suffix.lower() in _PRELOAD_SKIP_EXTS or not p.is_file() or p.is_symlink():
            continue
        if _is_preload_boilerplate(rel, prompt_low):
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz == 0 or sz > 400_000:
            continue
        rel_low = rel.lower()
        bname_low = p.name.lower()
        parts_low = rel_low.split("/")
        score = 0.0
        if rel_low in prompt_low:
            score += 50.0
        for kw in keywords:
            if kw and kw in rel_low:
                score += 6.0
        try:
            with p.open("rb") as fh:
                sample_txt = fh.read(65_536).decode("utf-8", errors="replace")
        except OSError:
            continue
        txt_low = sample_txt.lower()
        for kw in keywords:
            if kw and kw in txt_low:
                score += 1.0
        if score > 0 or bname_low in _PRELOAD_ENTRYPOINT_NAMES:
            if bname_low in _PRELOAD_ENTRYPOINT_NAMES and not rel_low.startswith(("docs/", "examples/", "tests/", "test/")):
                score += 4.0
            elif any(
                tok in ("api", "client", "queries", "query", "graphql", "rpc", "routes", "router", "handler", "handlers", "cmd", "cli")
                for seg in parts_low
                for tok in re.split(r"[._-]+", seg.split(".", 1)[0])
                if tok
            ) and score > 0:
                score += 3.0
        if score > 0 and not rel_low.startswith(("docs/", "examples/", ".github/")):
            score += 0.5
        if score > 0:
            scored.append((score, rel, sample_txt))

    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen: list[tuple[str, str]] = []
    used_tokens = 0
    for _score, rel, sample_txt in scored[: max_files * 2]:
        if len(chosen) >= max_files:
            break
        est = estimate_tokens(sample_txt)
        if used_tokens + est > token_budget:
            continue
        chosen.append((rel, sample_txt))
        used_tokens += est
    return chosen


def _load_safe_workspace_files(
    workspace_root: Path,
    rel_paths: list[str] | None,
    *,
    max_total_tokens: int = 8000,
    line_windows: dict[str, tuple[int, int]] | None = None,
) -> list[tuple[str, str]]:
    """Safely load workspace-relative text files up to `max_total_tokens` without escaping `workspace_root`."""
    if not rel_paths:
        return []
    ws_resolved = workspace_root.resolve()
    secret_basenames = {
        ".env", ".netrc", ".npmrc", ".pypirc", ".mcp.json", ".hmac_key",
        "id_rsa", "id_ed25519", "id_ecdsa", "credentials.json",
    }
    candidates: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for raw_p in rel_paths:
        clean_p = str(raw_p or "").strip()
        if not clean_p or ".." in Path(clean_p).parts:
            continue
        try:
            cand = (ws_resolved / clean_p).resolve()
            rel_obj = cand.relative_to(ws_resolved)
        except Exception:
            continue
        rel_str = rel_obj.as_posix()
        if not rel_str or rel_str in seen:
            continue
        bname = rel_obj.name.lower()
        parts_low = [p.lower() for p in rel_obj.parts]
        if (
            bname in secret_basenames
            or bname.startswith(".env.")
            or bname.endswith((".pem", ".key", ".p12", ".pfx"))
            or ".git" in parts_low
            or ".claude" in parts_low
            or ".ssh" in parts_low
            or ".cc-agy-data" in parts_low
            or ".worktrees" in parts_low
            or not cand.is_file()
        ):
            continue
        try:
            raw_bytes = cand.read_bytes()
            if b"\x00" in raw_bytes[:4096]:
                continue
            text = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            continue
        seen.add(rel_str)
        candidates.append((rel_str, text, estimate_tokens(text)))

    if not candidates:
        return []

    # Reserve token budget for smaller files so an oversized first file never starves subsequent small files.
    small_cutoff = max(250, max_total_tokens // 2)
    remaining_small_tokens = sum(est for _, _, est in candidates if est <= small_cutoff)
    loaded: list[tuple[str, str]] = []
    used_tokens = 0
    for rel_str, text, est in candidates:
        is_small = est <= small_cutoff
        if is_small:
            remaining_small_tokens = max(0, remaining_small_tokens - est)
        avail_for_this = max_total_tokens - used_tokens - (0 if is_small else min(remaining_small_tokens, max_total_tokens // 2))
        if avail_for_this <= 0:
            continue
        if est > avail_for_this:
            if avail_for_this < 200 and loaded:
                continue
            all_lines = text.splitlines()
            total_lines = max(1, len(all_lines))
            win = (line_windows or {}).get(rel_str)
            if win and total_lines > 1:
                w_start = max(1, min(int(win[0]), total_lines))
                w_limit = max(50, int(win[1]) if int(win[1]) > 0 else 250)
                w_end = min(total_lines, w_start + w_limit - 1)
                window_text = "\n".join(all_lines[w_start - 1 : w_end])
                trunc_text, _was_tr, _orig_t = truncate_to_token_budget(window_text, max(100, avail_for_this - 30))
                shown_count = max(1, len(trunc_text.splitlines()))
                actual_end = min(total_lines, w_start + shown_count - 1)
                text = (
                    trunc_text.rstrip("\n")
                    + f"\n... [truncated: showing lines {w_start}-{actual_end} of {total_lines}; use view_file for remaining lines]"
                )
            else:
                trunc_text, _was_tr, _orig_t = truncate_to_token_budget(text, max(100, avail_for_this - 30))
                shown_lines = max(1, len(trunc_text.splitlines()))
                text = (
                    trunc_text.rstrip("\n")
                    + f"\n... [truncated: showing lines 1-{shown_lines} of {total_lines}; use view_file for remaining lines]"
                )
            est = estimate_tokens(text)
        loaded.append((rel_str, text))
        used_tokens += est
    return loaded


def render_worker_prompt(
    brief: str,
    *,
    plan: list[str],
    acceptance: list[str],
    max_tool_calls: int | None,
    max_total_tokens: int | None,
    progress_channel: bool = True,
    preloaded_files: list[tuple[str, str]] | None = None,
    repo_map: str = "",
    focused_outline: str = "",
    spec_files_inlined: list[tuple[str, str]] | None = None,
    context_files_inlined: list[tuple[str, str]] | None = None,
    checklist: list[str] | None = None,
    raw_user_prompt: str = "",
    baseline_acceptance_probe: list[dict[str, Any]] | None = None,
    allowed_paths: list[str] | None = None,
) -> str:
    """Wrap the lead's brief with repository map, budget/reporting rules, inlined context, plan, and acceptance."""
    static_parts: list[str] = []
    if repo_map and repo_map.strip():
        static_parts.append("## Repository Map (Cached)\n" + repo_map.strip())
    budget_bits = []
    if max_tool_calls:
        budget_bits.append(f"{max_tool_calls} tool calls")
    if max_total_tokens:
        budget_bits.append(f"{max_total_tokens:,} total tokens")
    if budget_bits:
        static_parts.append(
            "## Budget\nThis turn stops automatically after " + " or ".join(budget_bits) + ". Work efficiently: "
            "make your first file edit within 8-10 tool calls and implement changes incrementally file-by-file "
            "so your edits are saved on disk early (never read more than 6-8 files before editing, and NEVER draft "
            "full file implementations or long code blocks inside internal thinking; emit create_file/edit_file "
            "tool calls directly and iterate against test output), read files <=260 lines in a single view_file "
            "call (start_line=1, end_line=260) or use the Focused File Outline line numbers to read 200-260 line "
            "spans on larger files instead of paging in small slices, avoid interactive one-liner REPL probes, "
            "run only the focused test file(s) needed for your changes, do not re-read files you have already seen, "
            "and remove any temporary scratch scripts before finishing."
        )
    if progress_channel:
        static_parts.append(
            "## Reporting\nUse the report_to_lead tool only when blocked (kind=\"question\" if a requirement is "
            "ambiguous; kind=\"blocker\" if you cannot proceed; questions and blockers end your turn) or "
            "kind=\"milestone\" between distinct plan steps on multi-step plans. Once your implementation and "
            "tests pass, finish your turn directly with a short summary instead of calling report_to_lead."
        )
    if (
        os.environ.get("CC_AGY_CONTAINMENT", "1").lower() not in ("0", "false", "no")
        and is_straitjacket_available()
    ):
        static_parts.append(
            "## Shell Commands & Evidence\n"
            "Commands and file reads execute with bounded output via Straitjacket (`run_command`, `view_file`, "
            "`search_dir`, `list_dir`, `find_file`, `ctx_query`, `ctx_get`, `ctx_search`). If a test or build "
            "produces large output, it returns a concise digest and an address (e.g. `run:<id>#stdout`). Use "
            "`ctx_get(address, start_line, end_line)` or `ctx_search(address, pattern)` to retrieve specific "
            "omitted lines or tracebacks instead of rerunning commands."
        )

    dynamic_parts: list[str] = []
    if raw_user_prompt and raw_user_prompt.strip():
        if allowed_paths:
            dynamic_parts.append(
                "## Background User Context (Reference Only: Scope Restricted to allowed_paths)\n"
                f"Scope restriction: implement ONLY the Lead's brief above and modify ONLY files within "
                f"`allowed_paths={list(allowed_paths)}`. Ignore any parts of this background request that "
                "target other files/directories or ask to run multiple workers:\n"
                + raw_user_prompt.strip()
            )
        else:
            dynamic_parts.append(
                "## Original User Request (Verbatim: Do Not Omit Any Requirement)\n"
                "If the Lead's brief above omits a branch or guesses an unverified flag/helper name, follow this Original User Request and existing repository helpers/callers.\n"
                + raw_user_prompt.strip()
            )
    if spec_files_inlined:
        full_specs = [(r, c) for r, c in spec_files_inlined if "[truncated:" not in c]
        trunc_specs = [(r, c) for r, c in spec_files_inlined if "[truncated:" in c]
        if full_specs:
            spec_blocks = [f"### `{rel}`\n```\n{content.rstrip()}\n```" for rel, content in full_specs]
            dynamic_parts.append(
                "## Task Specification Files (Verbatim: Do Not Re-Read)\n" + "\n\n".join(spec_blocks)
            )
        if trunc_specs:
            tspec_blocks = [f"### `{rel}`\n```\n{content.rstrip()}\n```" for rel, content in trunc_specs]
            dynamic_parts.append(
                "## Partially Inlined Specification Files (Truncated to Token Budget: Use view_file for Omitted Lines)\n"
                + "\n\n".join(tspec_blocks)
            )
    if checklist:
        dynamic_parts.append(
            "## Behavioral Checklist\n"
            + "\n".join(f"{i}. {item}" for i, item in enumerate(checklist, 1))
        )
    if focused_outline and focused_outline.strip():
        dynamic_parts.append("## Focused File Outline (Line-Numbered)\n" + focused_outline.strip())
    if preloaded_files:
        # Path hints only: full file contents pasted into turn 1 get re-read on every later turn.
        path_lines = [f"- `{rel}`" for rel, _content in preloaded_files]
        dynamic_parts.append(
            "## Candidate Repository Files\n"
            "These files matched the task's identifiers. Open only the relevant sections:\n"
            + "\n".join(path_lines)
        )
    if context_files_inlined:
        full_ctx = [(r, c) for r, c in context_files_inlined if "[truncated:" not in c]
        trunc_ctx = [(r, c) for r, c in context_files_inlined if "[truncated:" in c]
        if full_ctx:
            ctx_blocks = [f"### `{rel}`\n```\n{content.rstrip()}\n```" for rel, content in full_ctx]
            dynamic_parts.append(
                "## Pre-Read Context Files (Already Inlined: Do Not Call view_file on These Again; Edit Directly)\n"
                + "\n\n".join(ctx_blocks)
            )
        if trunc_ctx:
            tctx_blocks = [f"### `{rel}`\n```\n{content.rstrip()}\n```" for rel, content in trunc_ctx]
            dynamic_parts.append(
                "## Partially Inlined Context Files (Truncated to Token Budget: Use view_file with start_line/end_line for Omitted Lines)\n"
                + "\n\n".join(tctx_blocks)
            )
    if plan:
        dynamic_parts.append("## Plan from the lead\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1)))
    if acceptance:
        baseline_notes: list[str] = []
        for bp in baseline_acceptance_probe or []:
            if not isinstance(bp, dict) or bp.get("passed"):
                continue
            bp_cmd = str(bp.get("command") or "").strip()
            if bp.get("timed_out"):
                if bp_cmd:
                    baseline_notes.append(
                        f"[Baseline Acceptance Timeout Note] Command `{bp_cmd}` exceeded the 15s baseline probe "
                        f"on clean HEAD because the full test suite is large. Do NOT run the full `{bp_cmd}` "
                        f"suite during iteration; run only targeted test file(s) covering your modified modules "
                        f"(post-turn acceptance will also scope to related tests automatically)."
                    )
                continue
            bp_rc = bp.get("exit_code")
            if bp_rc in (None, 0):
                continue
            bp_tail = " | ".join(
                ln.strip() for ln in str(bp.get("error_tail") or "").splitlines() if ln.strip()
            )[:280]
            cwd_hint = ""
            if bp.get("subdir_cwd_passed") and bp.get("passing_cwd"):
                p_cwd = str(bp["passing_cwd"])
                cwd_hint = (
                    f" Note: existing tests under `{p_cwd}` pass (`rc=0`) on HEAD when run with "
                    f"`working_directory={p_cwd!r}`. Verify `{p_cwd}` with `working_directory={p_cwd!r}` and run "
                    f"your new/modified tests (e.g. `--ignore={p_cwd}`), then stop once both pass; post-worker "
                    f"acceptance automatically accounts for this pre-existing HEAD `cwd` quirk."
                )
            baseline_notes.append(
                f"[Baseline Acceptance Note] Command `{bp_cmd}` already exited non-zero (`rc={bp_rc}`) on "
                f"unmodified HEAD before any edits ({bp_tail or 'failed on baseline'}).{cwd_hint} Inspect how "
                f"existing tests configure `cwd`, fixtures, or environment variables before editing; NEVER modify "
                f"or shadow existing production class/module attributes or default paths in `__init__` just to "
                f"work around a test runner's current working directory (`cwd`)."
            )
        baseline_block = ("\n" + "\n".join(baseline_notes)) if baseline_notes else ""
        dynamic_parts.append(
            "## Acceptance\nWhen you finish, the plugin runs these commands and reports the results to the lead. "
            "Make them pass:\n"
            + "\n".join(f"- `{c}`" for c in acceptance)
            + baseline_block
            + "\nIMPORTANT: Create or update all requested implementation and test files (you may inspect how an "
            "existing test file sets up fixtures or `cwd` when needed), and verify your new/changed test files "
            "plus the Acceptance command(s) above. Once they pass and any temporary scratch files are removed, "
            "stop and report immediately; do NOT run unrequested linters (e.g. ruff, mypy, flake8, eslint) or "
            "unrelated full-repository suites not listed above. Never modify unrelated files or production default "
            "paths to work around pre-existing test failures outside your task scope or allowed_paths."
        )
    return "\n\n".join([brief.strip(), *static_parts, *dynamic_parts])


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _parse_test_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "errors": 0}
    text = _ANSI_ESCAPE_RE.sub("", output or "")

    # 1. Check anchored pytest summary lines first so prose like "expected 2 errors" in a unittest/stestr
    # log never hijacks the count parser.
    pytest_summary_lines = [
        ln
        for ln in text.splitlines()
        if re.search(r"=+.*\bin\s+\d+(?:\.\d+)?s\b.*=+", ln)
        or re.search(
            r"^\s*\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?)\b.*\bin\s+\d+(?:\.\d+)?s\b",
            ln,
        )
    ]
    if pytest_summary_lines:
        for ln in pytest_summary_lines:
            for key, pat in (
                ("passed", r"(\d+)\s+passed\b"),
                ("failed", r"(\d+)\s+failed\b"),
                ("errors", r"(\d+)\s+errors?\b"),
            ):
                for m in re.finditer(pat, ln):
                    counts[key] += int(m.group(1))
        return counts

    # 2. stestr / subunit summary blocks (e.g. "- Passed: 162", "- Failed: 1")
    st_passed = re.findall(r"-\s*Passed:\s*(\d+)", text)
    st_failed = re.findall(r"-\s*Failed:\s*(\d+)", text)
    st_errors = re.findall(r"-\s*(?:Errors?|Unexpected Success(?:es)?):\s*(\d+)", text)
    if st_passed or st_failed or st_errors:
        if st_passed:
            counts["passed"] = int(st_passed[-1])
        if st_failed:
            counts["failed"] = int(st_failed[-1])
        if st_errors:
            counts["errors"] = sum(int(x) for x in st_errors)
        return counts

    # 3. Standard library unittest summary ("Ran N tests in 0.123s" + "OK" / "FAILED (...)")
    # Sum across multiple unittest suite invocations in the same output and tolerate trailing teardown lines.
    ran_iter = list(re.finditer(r"Ran (\d+) tests? in [\d.]+s", text))
    if ran_iter:
        matched_any_suite = False
        for idx_r, m_ran in enumerate(ran_iter):
            suite_total = int(m_ran.group(1))
            block_start = m_ran.end()
            block_end = ran_iter[idx_r + 1].start() if idx_r + 1 < len(ran_iter) else len(text)
            suite_tail = text[block_start:block_end]
            m_fail = re.search(r"(?m)^\s*FAILED\s*\(([^)]+)\)", suite_tail)
            m_ok = re.search(r"(?m)^\s*OK(?:\s*\(([^)]*)\))?\s*$", suite_tail)
            if m_fail:
                matched_any_suite = True
                fail_meta = m_fail.group(1)
                f_m = re.search(r"failures=(\d+)", fail_meta)
                e_m = re.search(r"errors=(\d+)", fail_meta)
                s_m = re.search(r"skipped=(\d+)", fail_meta)
                f_cnt = int(f_m.group(1)) if f_m else 0
                e_cnt = int(e_m.group(1)) if e_m else 0
                s_cnt = int(s_m.group(1)) if s_m else 0
                counts["failed"] += f_cnt
                counts["errors"] += e_cnt
                counts["passed"] += max(0, suite_total - f_cnt - e_cnt - s_cnt)
            elif m_ok:
                matched_any_suite = True
                ok_meta = m_ok.group(1) or ""
                s_m = re.search(r"skipped=(\d+)", ok_meta)
                s_cnt = int(s_m.group(1)) if s_m else 0
                counts["passed"] += max(0, suite_total - s_cnt)
        if matched_any_suite:
            return counts

    # 4. Rust cargo test ("test result: ok. 12 passed; 0 failed; 0 ignored; ...")
    cargo_matches = list(
        re.finditer(r"test result:\s*(?:ok|FAILED)\.\s*(\d+)\s+passed;\s*(\d+)\s+failed;", text)
    )
    if cargo_matches:
        for cm in cargo_matches:
            counts["passed"] += int(cm.group(1))
            counts["failed"] += int(cm.group(2))
        return counts

    # 5. Jest / Vitest ("Tests: 1 failed, 14 passed, 15 total" or "Tests  14 passed (14)")
    jest_lines = [
        ln for ln in text.splitlines() if re.match(r"^\s*Tests:?\s+", ln) and re.search(r"\b(?:passed|failed)\b", ln)
    ]
    if jest_lines:
        last_jest = jest_lines[-1]
        m_jp = re.search(r"(\d+)\s+passed\b", last_jest)
        m_jf = re.search(r"(\d+)\s+failed\b", last_jest)
        if m_jp:
            counts["passed"] = int(m_jp.group(1))
        if m_jf:
            counts["failed"] = int(m_jf.group(1))
        return counts

    # 6. Go `go test` ("--- PASS: TestX", "--- FAIL: TestY", or package-level "ok  pkg  0.012s" / "FAIL pkg 0.012s")
    go_case_pass = len(re.findall(r"(?m)^---\s+PASS:\s+\S+", text))
    go_case_fail = len(re.findall(r"(?m)^---\s+FAIL:\s+\S+", text))
    if go_case_pass or go_case_fail:
        counts["passed"] = go_case_pass
        counts["failed"] = go_case_fail
        return counts
    go_pkg_ok = len(re.findall(r"(?m)^ok\s+\S+\s+(?:\d+(?:\.\d+)?s|\(cached\))", text))
    go_pkg_fail = len(re.findall(r"(?m)^FAIL\s+\S+\s+\d+(?:\.\d+)?s", text))
    if go_pkg_ok or go_pkg_fail:
        counts["passed"] = go_pkg_ok
        counts["failed"] = go_pkg_fail
        return counts

    # 7. Node.js `node --test` TAP/spec footer ("ℹ pass 5" / "# pass 5", "ℹ fail 1" / "# fail 1")
    node_pass = re.findall(r"(?m)^[\sℹ#]*pass\s+(\d+)\s*$", text)
    node_fail = re.findall(r"(?m)^[\sℹ#]*fail\s+(\d+)\s*$", text)
    if node_pass or node_fail:
        if node_pass:
            counts["passed"] = int(node_pass[-1])
        if node_fail:
            counts["failed"] = int(node_fail[-1])
        return counts

    # 8. Maven Surefire ("Tests run: 10, Failures: 1, Errors: 0, Skipped: 0")
    mvn_matches = list(
        re.finditer(
            r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)",
            text,
        )
    )
    if mvn_matches:
        last_m = mvn_matches[-1]
        total_m, fail_m, err_m, skip_m = (int(last_m.group(i)) for i in range(1, 5))
        counts["failed"] = fail_m
        counts["errors"] = err_m
        counts["passed"] = max(0, total_m - fail_m - err_m - skip_m)
        return counts

    # 9. Gradle ("5 tests completed, 1 failed, 1 skipped")
    gradle_matches = list(
        re.finditer(
            r"(\d+)\s+tests?\s+completed(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+skipped)?",
            text,
        )
    )
    if gradle_matches:
        total_g = sum(int(m.group(1)) for m in gradle_matches)
        fail_g = sum(int(m.group(2) or 0) for m in gradle_matches)
        skip_g = sum(int(m.group(3) or 0) for m in gradle_matches)
        counts["failed"] = fail_g
        counts["passed"] = max(0, total_g - fail_g - skip_g)
        return counts

    # 10. Fallback for minimal/custom pytest output without a duration footer
    for key, pat in (("passed", r"(\d+) passed\b"), ("failed", r"(\d+) failed\b"), ("errors", r"(\d+) errors?\b")):
        found = re.findall(pat, text)
        if found:
            counts[key] = int(found[-1])
    return counts


def _collect_workspace_node_bin_dirs(ws_root: Path, max_sub_dirs: int = 16) -> list[str]:
    """Return root `node_modules/.bin` followed by bounded 1-to-2-level monorepo subpackage `.bin` directories."""
    out: list[str] = []
    seen: set[str] = set()

    def _safe_is_dir(p: Path) -> bool:
        try:
            return p.is_dir()
        except OSError:
            return False

    def _add_bin(p: Path) -> None:
        if not _safe_is_dir(p):
            return
        resolved_str: str | None = None
        try:
            resolved_str = str(p.resolve())
        except OSError:
            resolved_str = None
        for cand in (str(p), resolved_str):
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)

    _add_bin(ws_root / "node_modules" / ".bin")
    skip_names = {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        "out",
        "coverage",
        "target",
    }
    sub_bins: list[Path] = []
    try:
        lvl1 = sorted(os.scandir(ws_root), key=lambda e: e.name)
    except OSError:
        lvl1 = []
    for e1 in lvl1:
        try:
            is_d1 = e1.is_dir(follow_symlinks=False)
        except OSError:
            is_d1 = False
        if e1.name.startswith(".") or e1.name in skip_names or not is_d1:
            continue
        p1 = Path(e1.path)
        b1 = p1 / "node_modules" / ".bin"
        if _safe_is_dir(b1):
            sub_bins.append(b1)
        try:
            lvl2 = sorted(os.scandir(p1), key=lambda e: e.name)
        except OSError:
            continue
        for e2 in lvl2:
            try:
                is_d2 = e2.is_dir(follow_symlinks=False)
            except OSError:
                is_d2 = False
            if e2.name.startswith(".") or e2.name in skip_names or not is_d2:
                continue
            b2 = Path(e2.path) / "node_modules" / ".bin"
            if _safe_is_dir(b2):
                sub_bins.append(b2)
    for sb in sub_bins[:max_sub_dirs]:
        _add_bin(sb)
    return out


def _build_workdir_exec_env(cwd: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment dict that prepends cwd (and cwd/src) to PYTHONPATH and node_modules/.bin + .venv/bin to PATH.

    Without ``base_env`` it starts from ``verify.credential_free_environ()``: every command this module
    runs in a workspace (acceptance, baseline probes, runner detection, smoke checks) gets an
    environment with plugin internals and credentials removed.
    """
    if base_env is None:
        from server.verify import credential_free_environ

        env = credential_free_environ()
    else:
        env = dict(base_env)
    cwd_resolved = cwd.resolve()
    py_parts = [str(cwd_resolved)]
    src_dir = cwd_resolved / "src"
    if src_dir.is_dir():
        py_parts.append(str(src_dir))
    existing_py = (env.get("PYTHONPATH") or "").strip()
    if existing_py:
        py_parts.append(existing_py)
    env["PYTHONPATH"] = ":".join(py_parts)
    path_prefixes: list[str] = _collect_workspace_node_bin_dirs(cwd_resolved)
    venv_dir = cwd_resolved / ".venv"
    venv_bin = venv_dir / "bin"
    if venv_bin.is_dir():
        path_prefixes.append(str(venv_bin))
        env["VIRTUAL_ENV"] = str(venv_dir)
    if path_prefixes:
        existing_path = (env.get("PATH") or "").strip()
        prefix_joined = ":".join(path_prefixes)
        env["PATH"] = f"{prefix_joined}:{existing_path}" if existing_path else prefix_joined
    return env


def _invalidate_python_bytecode_for_paths(root: Path, rel_paths: list[str] | tuple[str, ...]) -> None:
    """Delete stale `__pycache__/<stem>.*.pyc` bytecode for modified `.py` files inside `root`.

    CPython PEP 552 `.pyc` headers compare 32-bit integer `int(st_mtime)` seconds and `st_size`, so
    two edits within the same wall-clock second that keep the same byte length (`a * b` -> `a + b`)
    would otherwise execute stale bytecode from an earlier turn.
    """
    try:
        resolved_root = root.resolve()
    except OSError:
        return
    for raw_rel in rel_paths:
        norm = str(raw_rel or "").replace("\\", "/").strip("/")
        if not norm.endswith(".py"):
            continue
        src = root / norm
        try:
            resolved_src = src.resolve()
        except OSError:
            continue
        if not resolved_src.is_relative_to(resolved_root):
            continue
        stem = resolved_src.stem
        pycache = resolved_src.parent / "__pycache__"
        if pycache.is_dir() and not pycache.is_symlink():
            for pyc in pycache.glob(f"{stem}.*.pyc"):
                try:
                    pyc.unlink(missing_ok=True)
                except OSError:
                    pass
        legacy_pyc = resolved_src.with_suffix(".pyc")
        if legacy_pyc.is_file() and not legacy_pyc.is_symlink():
            try:
                legacy_pyc.unlink(missing_ok=True)
            except OSError:
                pass


_PYTEST_VALUE_FLAGS = frozenset({
    "-k",
    "-m",
    "-o",
    "-c",
    "-p",
    "--rootdir",
    "--confcutdir",
    "--import-mode",
    "--basetemp",
    "--tb",
    "--maxfail",
    "--durations",
    "--ignore",
    "--ignore-glob",
    "--deselect",
    "--cov",
    "--cov-report",
    "--cov-config",
})


def _strip_missing_pytest_file_targets(
    cmd: str,
    output: str,
    cwd: Path,
) -> tuple[str | None, list[str]]:
    """When a pytest command fails with `exit_code == 4` (`ERROR: file or directory not found: <path>`),
    strip non-existent file/dir target arguments if at least one existing test target remains in `cmd`."""
    if "file or directory not found:" not in (output or "").lower():
        return None, []
    stripped = cmd.strip()
    if any(sep in stripped for sep in ("&&", "||", ";", "|", "\n")):
        return None, []
    runner_pref = _extract_pytest_runner_prefix(stripped)
    if not runner_pref:
        return None, []
    try:
        tokens = shlex.split(stripped)
        pref_tokens = shlex.split(runner_pref)
    except ValueError:
        return None, []
    if len(tokens) <= len(pref_tokens):
        return None, []
    reported_missing = {
        m.strip().rstrip(",'\"")
        for m in re.findall(r"ERROR:\s+file or directory not found:\s+(\S+)", output or "", re.I)
        if m.strip()
    }
    kept_tokens: list[str] = list(tokens[: len(pref_tokens)])
    existing_targets: list[str] = []
    missing_targets: list[str] = []
    skip_next = False
    for tok in tokens[len(pref_tokens) :]:
        if skip_next:
            kept_tokens.append(tok)
            skip_next = False
            continue
        if tok.startswith("-"):
            kept_tokens.append(tok)
            if tok in _PYTEST_VALUE_FLAGS:
                skip_next = True
            continue
        file_part = tok.split("::", 1)[0].strip()
        cand_path = cwd / file_part
        if file_part and cand_path.exists():
            existing_targets.append(file_part)
            kept_tokens.append(tok)
        elif file_part and (
            file_part in reported_missing
            or tok in reported_missing
            or "/" in file_part
            or file_part.endswith(".py")
        ):
            missing_targets.append(file_part)
        else:
            kept_tokens.append(tok)
    if not missing_targets or not existing_targets:
        return None, []
    return shlex.join(kept_tokens), missing_targets


def run_acceptance(
    cmds: list[str],
    cwd: Path,
    timeout: float | None = None,
    *,
    total_wall_budget: float | None = None,
    restore_tracked: bool = True,
    change_set: list[str] | tuple[str, ...] = (),
    quarantine_dir: Path | None = None,
    diagnostic_remaps: dict[str, tuple[str, list[str]]] | None = None,
) -> list[dict[str, Any]]:
    """Run acceptance checks via ``verify.execute_command_suite`` so both delegation and batch paths
    share identical credential scrubbing, bounded output streaming, unconditional process-group
    termination, per-command workspace snapshots, and untracked artifact cleanup.

    If an explicit pytest acceptance command exits with code 4 because one or more requested targets
    are missing, any reduced or remapped command runs for diagnostic output only; ``passed`` remains
    ``False`` and ``exit_code`` stays ``4`` so a missing required test never becomes a passing result.
    """
    from server.verify import execute_check_process, execute_command_suite

    default_timeout = (
        DEFAULT_BOUNDED_ACCEPTANCE_TIMEOUT
        if (
            os.environ.get("CC_AGY_SESSION_DEADLINE_SECONDS", "").strip()
            or resolve_cost_profile() == "economy"
        )
        else DEFAULT_ACCEPTANCE_TIMEOUT
    )
    limit = float(timeout or _env_int("CC_AGY_ACCEPTANCE_TIMEOUT", int(default_timeout)))

    def _missing_target_diagnostics(
        cmd: str,
        rc: int | None,
        _timed_out: bool,
        out: str,
        rem_timeout: float,
        exec_env: Any,
    ) -> dict[str, Any] | None:
        if rc != 4:
            return None
        filtered_cmd, filtered_missing = _strip_missing_pytest_file_targets(cmd, out or "", cwd)
        if not filtered_cmd and diagnostic_remaps:
            remap_pair = diagnostic_remaps.get(cmd.strip())
            if remap_pair:
                filtered_cmd, filtered_missing = remap_pair[0], list(remap_pair[1])
        if not filtered_cmd:
            diag_details: list[dict[str, str]] = []
            remapped_list, did_diag_remap = remap_nonexistent_pytest_acceptance_cmds(
                [cmd],
                cwd,
                [{"command": cmd, "exit_code": 4, "passed": False, "error_tail": out or ""}],
                remap_details=diag_details,
            )
            if did_diag_remap and remapped_list and remapped_list[0].strip() != cmd.strip():
                filtered_cmd = remapped_list[0]
                filtered_missing = list(
                    dict.fromkeys(
                        str(d.get("missing_target") or "").strip()
                        for d in diag_details
                        if isinstance(d, dict) and d.get("missing_target")
                    )
                )
        if not filtered_cmd or not filtered_missing:
            return None
        diag_rc, diag_to, diag_out, _ = execute_check_process(
            filtered_cmd,
            cwd,
            exec_env,
            timeout=max(1.0, rem_timeout),
            max_tail_chars=20_000,
        )
        diag_header = (
            f"\n[cc-agy diagnostic: requested target(s) {filtered_missing} not found; "
            f"diagnostic run of {filtered_cmd}]\n"
        )
        combined = f"{(out or '').rstrip()}{diag_header}{diag_out or ''}"
        return {
            "force_failed": True,
            "effective_command": filtered_cmd,
            "filtered_missing_paths": filtered_missing,
            "diagnostic_exit_code": diag_rc,
            "diagnostic_timed_out": diag_to,
            "diagnostic_counts": _parse_test_counts(diag_out or ""),
            "combined_output": combined,
        }

    suite_items, cleanup_summary = execute_command_suite(
        cmds,
        cwd,
        timeout=limit,
        total_wall_budget=total_wall_budget,
        stop_on_failure=False,
        stop_on_timeout=True,
        clean_new_files=True,
        restore_tracked=restore_tracked,
        change_set=change_set,
        quarantine_dir=quarantine_dir,
        max_tail_chars=20_000,
        check_all_vacuous=True,
        diagnostic_hook=_missing_target_diagnostics,
    )
    results: list[dict[str, Any]] = []
    for cmd_res, meta in suite_items:
        if meta.get("wall_exhausted_skipped"):
            results.append(
                {
                    "command": cmd_res.command,
                    "exit_code": None,
                    "timed_out": True,
                    "passed": False,
                    "duration_seconds": 0.0,
                    "counts": {"passed": 0, "failed": 0, "errors": 0},
                    "output_tail": cmd_res.tail,
                }
            )
            continue
        raw_tail = str(meta.get("raw_tail") or "")
        tail, _, _ = truncate_to_token_budget(raw_tail[-20_000:], ACCEPTANCE_OUTPUT_TOKENS)
        if cmd_res.untrusted_side_effects:
            from server.verify import side_effect_note

            tail = (tail or "") + side_effect_note(cmd_res.untrusted_side_effects, restored=restore_tracked)
        filtered_cmd = meta.get("effective_command")
        filtered_missing_paths = meta.get("filtered_missing_paths")
        entry: dict[str, Any] = {
            "command": cmd_res.command,
            "exit_code": cmd_res.exit_code,
            "timed_out": cmd_res.timed_out,
            "passed": cmd_res.passed,
            "duration_seconds": round(cmd_res.duration_seconds, 1),
            "counts": dict(meta.get("counts") or {"passed": 0, "failed": 0, "errors": 0}),
            "output_tail": tail,
            **({"vacuous": True} if cmd_res.vacuous else {}),
            **(
                {
                    "side_effects_restored" if restore_tracked else "tracked_side_effects": list(
                        cmd_res.side_effects_restored[:50]
                    )
                }
                if cmd_res.side_effects_restored
                else {}
            ),
            **(
                {"untrusted_side_effects": list(cmd_res.untrusted_side_effects[:50])}
                if cmd_res.untrusted_side_effects
                else {}
            ),
            **(
                {"credential_env_removed": list(cmd_res.credential_env_removed)}
                if cmd_res.credential_env_removed
                else {}
            ),
            **(
                {
                    "effective_command": filtered_cmd,
                    "filtered_missing_paths": list(filtered_missing_paths),
                    **({"diagnostic_counts": dict(meta["diagnostic_counts"])} if "diagnostic_counts" in meta else {}),
                    **({"diagnostic_exit_code": meta["diagnostic_exit_code"]} if "diagnostic_exit_code" in meta else {}),
                    **({"diagnostic_timed_out": True} if meta.get("diagnostic_timed_out") else {}),
                }
                if filtered_cmd and filtered_missing_paths
                else {}
            ),
        }
        results.append(entry)

    q_report = cleanup_summary.get("quarantine") or {}
    if q_report.get("moved") and results:
        results[-1]["quarantined_files"] = {"dir": q_report["quarantine_dir"], "files": q_report["moved"][:20]}
    if q_report.get("left_in_place") and results:
        results[-1]["untracked_left_in_place"] = q_report["left_in_place"][:20]
    created = cleanup_summary.get("created") or []
    if created and results:
        results[-1]["cleaned_artifacts"] = created[:20]
    return results


_PROBE_SKIP_TOP_DIRS = frozenset({
    ".git",
    ".venv",
    ".cc-agy-data",
    ".worktrees",
    ".ctx-session-reads",
    ".cc-agy-review",
})


def _snapshot_probe_worktree_state(cwd: Path) -> tuple[dict[str, bytes | None], set[str]]:
    """Snapshot all untracked files (including small gitignored files) and directory paths in `cwd`."""
    files_snap: dict[str, bytes | None] = {}

    def _record_untracked_file(rel: str) -> None:
        p = cwd / rel
        data: bytes | None = None
        try:
            if not p.is_symlink() and p.is_file() and p.stat().st_size <= 1_048_576:
                data = p.read_bytes()
        except OSError:
            data = None
        files_snap[rel] = data

    res = _git_raw(
        ["ls-files", "--others", "-z"],
        cwd=str(cwd),
        capture_output=True,
        check=False,
    )
    if res.returncode == 0:
        for raw in res.stdout.split(b"\x00"):
            if not raw:
                continue
            rel = raw.decode("utf-8", errors="surrogateescape")
            top = rel.split("/", 1)[0]
            if top not in _PROBE_SKIP_TOP_DIRS:
                _record_untracked_file(rel)
    dirs_set: set[str] = set()
    for cur_root, dirnames, filenames in os.walk(str(cwd)):
        rel_root = os.path.relpath(cur_root, str(cwd))
        if rel_root == ".":
            dirnames[:] = [d for d in dirnames if d not in _PROBE_SKIP_TOP_DIRS]
            rel_prefix = ""
        else:
            rel_prefix = Path(rel_root).as_posix()
            dirs_set.add(rel_prefix)
        if res.returncode != 0:
            for fn in filenames:
                _record_untracked_file(f"{rel_prefix}/{fn}" if rel_prefix else fn)
    return files_snap, dirs_set


def _restore_probe_worktree_state(
    cwd: Path,
    before_files: dict[str, bytes | None] | set[str],
    before_dirs: set[str],
    *,
    reset_tracked: bool = True,
) -> list[str]:
    """Restore clean worktree state after an acceptance probe by reverting tracked/untracked mutations and removing new files/dirs."""
    if reset_tracked:
        _run_git(["checkout", "--", "."], cwd=cwd, check=False)
    after_files, after_dirs = _snapshot_probe_worktree_state(cwd)
    before_keys = set(before_files.keys()) if isinstance(before_files, dict) else set(before_files)
    removed: list[str] = []
    for rel in sorted(set(after_files.keys()) - before_keys):
        p = cwd / rel
        try:
            if p.is_symlink() or p.is_file():
                p.unlink(missing_ok=True)
                removed.append(rel)
        except OSError:
            pass
    if isinstance(before_files, dict):
        for rel, prev_bytes in before_files.items():
            if prev_bytes is None:
                continue
            p = cwd / rel
            try:
                if p.is_symlink():
                    p.unlink(missing_ok=True)
                if not p.is_file() or after_files.get(rel) != prev_bytes:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(prev_bytes)
            except OSError:
                pass
    for rel_dir in sorted(after_dirs - before_dirs, key=lambda d: d.count("/"), reverse=True):
        dpath = cwd / rel_dir
        try:
            if dpath.is_symlink():
                dpath.unlink(missing_ok=True)
            elif dpath.is_dir():
                shutil.rmtree(dpath, ignore_errors=True)
        except OSError:
            pass
    return removed


_COMPILER_CLI_CONFIG_ERROR_RE = re.compile(
    r"(?:"
    r"Syntax:\s+tsc\s+\[options\]"
    r"|Initializes a TypeScript project and creates a tsconfig\.json"
    r"|error TS(?:5023|5024|5042|5058|5083|6053|18002|18003)\b"
    r")",
    re.I,
)


def _extract_short_error_tail(output: str, max_lines: int = 3) -> str:
    """Return a compact 3-line summary of the failure from ANSI-stripped test output."""
    clean = _ANSI_ESCAPE_RE.sub("", output or "")
    non_empty = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    if not non_empty:
        return ""
    priority = [
        ln
        for ln in non_empty
        if re.match(r"^(?:FAILED|ERROR)\b", ln)
        or ln.startswith("E   ")
        or re.search(r"\b(?:[A-Za-z]+Error|Exception|AssertionError):", ln)
        or _COMPILER_CLI_CONFIG_ERROR_RE.search(ln)
    ]
    chosen = priority[-max_lines:] if priority else non_empty[-max_lines:]
    return "\n".join(ln[:180] for ln in chosen)


def _extract_failing_test_nodes_and_files(output: str, cwd: Path) -> tuple[list[str], list[str]]:
    """Extract failing test node IDs (e.g. `tests/sub/test_x.py::test_y`) and existing relative test file paths."""
    clean = _ANSI_ESCAPE_RE.sub("", output or "")
    nodes: list[str] = []
    files: list[str] = []

    def _record_file(cand: str) -> str | None:
        norm = cand.strip().lstrip("./").replace("\\", "/")
        if not norm:
            return None
        p = (cwd / norm).resolve()
        try:
            rel_norm = p.relative_to(cwd.resolve()).as_posix()
        except ValueError:
            return None
        if p.is_file():
            if rel_norm not in files:
                files.append(rel_norm)
            return rel_norm
        return None

    for m in re.finditer(r"(?m)^(?:FAILED|ERROR(?:\s+at\s+\w+\s+of|\s+collecting)?)\s+([^\s]+\.py(?:::[^\s]+)*)", clean):
        raw_node = m.group(1).strip().rstrip(",")
        file_part = raw_node.split("::", 1)[0]
        rel_f = _record_file(file_part)
        if rel_f:
            suffix = raw_node[len(file_part):]
            norm_node = f"{rel_f}{suffix}"
            if norm_node not in nodes:
                nodes.append(norm_node)

    # Traceback location fallback for `pytest -x -q` or custom formatters that omit `FAILED <node>` summary lines
    current_func: str | None = None
    for ln in clean.splitlines():
        m_hdr = re.match(r"^_{3,}\s+([A-Za-z_][\w.\[\]-]*)\s+_{3,}$", ln.strip())
        if m_hdr:
            current_func = m_hdr.group(1)
            continue
        m_loc = re.match(r"^([\w./\\-]+?\.py):(\d+):", ln.strip())
        if m_loc:
            cand_f = m_loc.group(1)
            base_n = Path(cand_f).name
            if base_n.startswith("test_") or base_n.endswith("_test.py"):
                rel_f = _record_file(cand_f)
                if rel_f:
                    fallback_node = f"{rel_f}::{current_func.replace('.', '::')}" if current_func else rel_f
                    if not any(n == fallback_node or n.startswith(f"{rel_f}::") for n in nodes):
                        nodes.append(fallback_node)

    # Standard library unittest header: `FAIL: test_foo (pkg.sub.test_bar.TestCls)`
    for m_ut in re.finditer(r"(?m)^(?:FAIL|ERROR):\s+([A-Za-z_]\w*)\s+\(([\w.]+)\)", clean):
        func_name, qual_cls = m_ut.group(1), m_ut.group(2)
        parts = qual_cls.split(".")
        for split_idx in range(len(parts), 0, -1):
            cand_rel = "/".join(parts[:split_idx]) + ".py"
            rel_f = _record_file(cand_rel)
            if rel_f:
                cls_part = "::".join(parts[split_idx:])
                ut_node = f"{rel_f}::{cls_part}::{func_name}" if cls_part else f"{rel_f}::{func_name}"
                if ut_node not in nodes:
                    nodes.append(ut_node)
                break

    return nodes, files


def _resolve_pytest_runner_for_subdir(runner_prefix: str, root_cwd: Path) -> str:
    """Resolve repo-relative runner executables (e.g. `.venv/bin/pytest`) without dereferencing venv symlinks."""
    try:
        tokens = shlex.split(runner_prefix)
    except ValueError:
        return runner_prefix
    if not tokens:
        return runner_prefix
    changed = False
    for idx, tok in enumerate(tokens):
        if idx > 0 and tokens[idx - 1] != "run":
            if all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t) for t in tokens[:idx]):
                pass
            else:
                continue
        if "/" in tok and not os.path.isabs(tok):
            cand = os.path.abspath(os.path.join(str(root_cwd), tok))
            if os.path.lexists(cand):
                tokens[idx] = cand
                changed = True
    return shlex.join(tokens) if changed else runner_prefix


def _extract_pytest_runner_prefix(cmd: str) -> str | None:
    """If `cmd` is a single simple pytest invocation, return its runner prefix (e.g. `.venv/bin/pytest`)."""
    stripped = cmd.strip()
    if any(sep in stripped for sep in ("&&", "||", ";", "|", "\n")):
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return None
    if not tokens:
        return None
    idx = 0
    while idx < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[idx]):
        idx += 1
    if idx >= len(tokens):
        return None
    head_base = Path(tokens[idx]).name
    if head_base in ("pytest", "py.test"):
        return shlex.join(tokens[: idx + 1])
    if (
        head_base.startswith("python")
        and idx + 2 < len(tokens)
        and tokens[idx + 1] == "-m"
        and tokens[idx + 2] in ("pytest", "py.test")
    ):
        return shlex.join(tokens[: idx + 3])
    if (
        head_base == "uv"
        and idx + 2 < len(tokens)
        and tokens[idx + 1] == "run"
        and Path(tokens[idx + 2]).name in ("pytest", "py.test")
    ):
        return shlex.join(tokens[: idx + 3])
    return None


def _strip_pytest_exitfirst_flags(cmd: str) -> str:
    """Return `cmd` with `-x`, `--exitfirst`, and `--maxfail=...` removed so remaining tests all execute."""
    try:
        tokens = shlex.split(cmd.strip())
    except ValueError:
        return cmd
    cleaned: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in ("-x", "--exitfirst"):
            continue
        if tok == "--maxfail":
            skip_next = True
            continue
        if tok.startswith("--maxfail="):
            continue
        if tok.startswith("-") and not tok.startswith("--") and len(tok) > 2 and "x" in tok[1:]:
            rem = "-" + tok[1:].replace("x", "")
            if rem != "-":
                cleaned.append(rem)
            continue
        cleaned.append(tok)
    return shlex.join(cleaned) if cleaned else cmd


def _run_bounded_probe_cmd(
    cmd: str,
    *,
    cwd: Path,
    exec_env: dict[str, str],
    timeout_s: float,
) -> tuple[int | None, bool, str]:
    from server.verify import execute_check_process

    rc, timed_out, out, _ = execute_check_process(
        cmd,
        cwd,
        exec_env,
        timeout=max(0.1, float(timeout_s)),
        max_tail_chars=20_000,
    )
    return rc, timed_out, (out or "")


_LINTER_OR_STATIC_CHECK_CMD_RE = re.compile(
    r"\b(?:"
    r"go\s+vet"
    r"|golangci-lint"
    r"|staticcheck"
    r"|flake8"
    r"|ruff(?:\s+check)?"
    r"|mypy"
    r"|pyright"
    r"|pylint"
    r"|bandit"
    r"|eslint"
    r"|biome"
    r"|stylelint"
    r"|tsc"
    r"|cargo\s+clippy"
    r"|rubocop"
    r"|shellcheck"
    r")\b",
    re.I,
)

_TEST_RUNNER_CMD_RE = re.compile(
    r"\b(?:"
    r"pytest|py\.test|unittest|go\s+test|jest|vitest|mocha|ava|cargo\s+test"
    r"|npm\s+(?:run\s+)?test|yarn\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|rspec"
    r")\b",
    re.I,
)


def _is_linter_or_static_check_cmd(cmd: str) -> bool:
    if not cmd or _TEST_RUNNER_CMD_RE.search(cmd):
        return False
    return bool(_LINTER_OR_STATIC_CHECK_CMD_RE.search(cmd))


def _extract_linter_diagnostic_files(output: str, cwd: Path) -> list[str]:
    """Extract existing relative file paths from `path:line[:col]` or `path(line,col)` linter/compiler output."""
    clean = _ANSI_ESCAPE_RE.sub("", output or "")
    files: list[str] = []
    cwd_res = cwd.resolve()

    def _add_cand(raw_cand: str) -> None:
        norm = raw_cand.strip().lstrip("./").replace("\\", "/")
        if not norm or norm.startswith("-"):
            return
        try:
            p = (cwd / norm).resolve()
            rel_norm = p.relative_to(cwd_res).as_posix()
        except (ValueError, OSError):
            return
        if p.is_file() and rel_norm not in files:
            files.append(rel_norm)

    for m in re.finditer(r"(?:^|[\s(])([A-Za-z0-9_./\\-]+?\.[A-Za-z0-9]+):(\d+)(?::\d+)?(?::|\b)", clean, re.M):
        _add_cand(m.group(1))
    for m in re.finditer(r"(?:^|[\s])([A-Za-z0-9_./\\-]+?\.[A-Za-z0-9]+)\((\d+),\d+\):", clean, re.M):
        _add_cand(m.group(1))
    return files


def probe_baseline_acceptance(
    cmds: list[str],
    cwd: Path,
    *,
    per_cmd_timeout: float = 15.0,
    total_timeout: float = 25.0,
    reset_tracked: bool = True,
) -> list[dict[str, Any]]:
    """Probe explicit acceptance commands on clean HEAD (`0` model tokens) and restore 100% worktree hygiene."""
    if not cmds or not cwd.is_dir():
        return []
    before_files, before_dirs = _snapshot_probe_worktree_state(cwd)
    exec_env = _build_workdir_exec_env(cwd)
    probe_started = time.monotonic()
    probes: list[dict[str, Any]] = []
    try:
        for cmd in cmds:
            elapsed = time.monotonic() - probe_started
            remaining = total_timeout - elapsed
            if remaining < 0.2:
                break
            cmd_limit = min(per_cmd_timeout, remaining)
            rc, timed_out, out = _run_bounded_probe_cmd(
                cmd, cwd=cwd, exec_env=exec_env, timeout_s=cmd_limit
            )
            _restore_probe_worktree_state(cwd, before_files, before_dirs, reset_tracked=reset_tracked)
            if timed_out:
                probes.append({"command": cmd, "exit_code": None, "timed_out": True, "passed": False})
                continue
            counts = _parse_test_counts(out)
            if rc == 0:
                probes.append({"command": cmd, "exit_code": 0, "timed_out": False, "passed": True, "counts": counts})
                continue
            nodes, files = _extract_failing_test_nodes_and_files(out, cwd)
            if not files:
                files = _extract_linter_diagnostic_files(out, cwd)
            short_tail = _extract_short_error_tail(out, max_lines=3)
            clean_probe_out = _ANSI_ESCAPE_RE.sub("", out or "")
            is_cli_cfg_err = bool(
                _COMPILER_CLI_CONFIG_ERROR_RE.search(clean_probe_out)
                and not re.search(r"\berror TS(?:1|2|4|7)\d{3}\b", clean_probe_out)
            )
            entry: dict[str, Any] = {
                "command": cmd,
                "exit_code": rc,
                "timed_out": False,
                "passed": False,
                "counts": counts,
                "error_tail": short_tail,
                "failing_nodes": nodes[:20],
                "failing_files": files[:10],
                **({"compiler_cli_config_error": True} if is_cli_cfg_err else {}),
            }
            pytest_runner = _extract_pytest_runner_prefix(cmd)
            if pytest_runner and files:
                cand_dirs = list(
                    dict.fromkeys(
                        str(Path(f).parent.as_posix())
                        for f in files
                        if str(Path(f).parent.as_posix()) not in ("", ".")
                    )
                )
                rem_after = total_timeout - (time.monotonic() - probe_started)
                if len(cand_dirs) == 1 and rem_after >= 0.4:
                    sub_rel = cand_dirs[0]
                    sub_dir = cwd / sub_rel
                    if sub_dir.is_dir():
                        sub_runner = _resolve_pytest_runner_for_subdir(pytest_runner, cwd)
                        sub_rc, sub_to, sub_out = _run_bounded_probe_cmd(
                            f"{sub_runner} -q",
                            cwd=sub_dir,
                            exec_env=exec_env,
                            timeout_s=min(10.0, rem_after),
                        )
                        _restore_probe_worktree_state(cwd, before_files, before_dirs, reset_tracked=reset_tracked)
                        sub_counts = _parse_test_counts(sub_out)
                        if (
                            not sub_to
                            and sub_rc == 0
                            and int(sub_counts.get("passed") or 0) >= 1
                            and int(sub_counts.get("failed") or 0) == 0
                            and int(sub_counts.get("errors") or 0) == 0
                        ):
                            entry["subdir_cwd_passed"] = True
                            entry["passing_cwd"] = sub_rel
                            entry["subdir_counts"] = sub_counts
            probes.append(entry)
    finally:
        _restore_probe_worktree_state(cwd, before_files, before_dirs, reset_tracked=reset_tracked)
    return probes


_PYTEST_NOT_FOUND_RE = re.compile(r"file or directory not found:\s*([^\s'\"`]+)", re.I)


def remap_nonexistent_pytest_acceptance_cmds(
    cmds: list[str],
    cwd: Path,
    probes: list[dict[str, Any]] | None,
    *,
    allowed_paths: list[str] | None = None,
    prompt_text: str = "",
    remap_details: list[dict[str, str]] | None = None,
) -> tuple[list[str], bool]:
    """Rewrite nonexistent `exit_code == 4` pytest target paths to unique tracked basename matches."""
    if not cmds or not probes or not cwd.is_dir():
        return list(cmds), False
    probe_by_cmd = {
        str(p.get("command") or "").strip(): p
        for p in probes
        if isinstance(p, dict) and not p.get("passed") and p.get("exit_code") == 4
    }
    if not probe_by_cmd:
        return list(cmds), False
    norm_prompt = prompt_text.replace("\\", "/") if prompt_text else ""
    tracked: list[str] | None = None
    remapped_any = False
    updated_cmds: list[str] = []
    for cmd in cmds:
        pr = probe_by_cmd.get(cmd.strip())
        if not pr or "pytest" not in cmd:
            updated_cmds.append(cmd)
            continue
        tail = str(pr.get("error_tail") or "")
        missing_tokens = list(dict.fromkeys(_PYTEST_NOT_FOUND_RE.findall(tail)))
        if not missing_tokens:
            try:
                toks = shlex.split(cmd)
            except ValueError:
                toks = cmd.split()
            for tok in toks:
                if tok.startswith("-"):
                    continue
                base_tok = tok.split("::", 1)[0].strip()
                if base_tok.endswith(".py") and not (cwd / base_tok).exists():
                    missing_tokens.append(tok)
        new_cmd = cmd
        for raw_missing in missing_tokens:
            file_part, sep, node_part = raw_missing.partition("::")
            clean_missing = file_part.replace("\\", "/").strip("/")
            if not clean_missing or (cwd / clean_missing).exists():
                continue
            if allowed_paths and _matches_allowed_paths(clean_missing, allowed_paths):
                continue
            target_bname = Path(clean_missing).name
            if not target_bname:
                continue
            if norm_prompt:
                if re.search(
                    r"(?<![A-Za-z0-9_./-])" + re.escape(clean_missing) + r"(?![A-Za-z0-9_./-])",
                    norm_prompt,
                ):
                    continue
                if re.search(
                    r"\b(?:create|add|write|new)\b[^.\n]{0,120}\b" + re.escape(target_bname) + r"\b",
                    norm_prompt,
                    re.I,
                ):
                    continue
            if tracked is None:
                tracked = _list_tracked_repo_files(cwd)
            try:
                cur_toks = shlex.split(new_cmd)
            except ValueError:
                cur_toks = new_cmd.split()
            existing_cmd_targets = {
                tok.split("::", 1)[0].replace("\\", "/").strip("/")
                for tok in cur_toks
                if not tok.startswith("-") and tok.split("::", 1)[0].endswith(".py")
            }
            matches = [
                tf
                for tf in tracked
                if Path(tf).name == target_bname
                and (cwd / tf).is_file()
                and tf not in existing_cmd_targets
            ]
            chosen_tf: str | None = None
            if len(matches) == 1:
                chosen_tf = matches[0]
            elif len(matches) > 1:
                miss_parts = list(reversed(Path(clean_missing).parts[:-1]))
                ranked: list[tuple[int, int, str]] = []
                for tf in matches:
                    tf_parts = list(reversed(Path(tf).parts[:-1]))
                    shared = 0
                    for a_p, b_p in zip(miss_parts, tf_parts):
                        if a_p == b_p:
                            shared += 1
                        else:
                            break
                    ranked.append((-shared, len(Path(tf).parts), tf))
                ranked.sort()
                if len(ranked) >= 2 and ranked[0][0] < ranked[1][0]:
                    chosen_tf = ranked[0][2]
            if chosen_tf:
                replacement = f"{chosen_tf}{sep}{node_part}" if sep else chosen_tf
                if raw_missing in new_cmd:
                    prev_cmd = new_cmd
                    new_cmd = new_cmd.replace(raw_missing, replacement, 1)
                    remapped_any = True
                    if remap_details is not None:
                        remap_details.append(
                            {
                                "original_command": prev_cmd,
                                "remapped_command": new_cmd,
                                "missing_target": clean_missing,
                                "remapped_target": chosen_tf,
                            }
                        )
        updated_cmds.append(new_cmd)
    return updated_cmds, remapped_any


def evaluate_post_worker_baseline_failures(
    acceptance_results: list[dict[str, Any]],
    baseline_probe: list[dict[str, Any]] | None,
    cwd: Path,
    files_changed: list[str],
    *,
    timeout: float = 30.0,
) -> list[str]:
    """Account for pre-existing clean-HEAD `cwd` or untouched baseline test/linter failures without hiding regressions."""
    from server.verify import check_side_effects, snapshot_workspace

    if not acceptance_results or not files_changed or not cwd.is_dir():
        return []
    probe_by_cmd = {
        str(p.get("command") or "").strip(): p
        for p in (baseline_probe or [])
        if isinstance(p, dict) and not p.get("passed") and not p.get("timed_out") and p.get("exit_code") not in (None, 0)
    }
    any_real_test_passed = any(
        is_non_smoke_passing_entry(x)
        and (
            int((x.get("counts") or {}).get("passed") or 0) > 0
            or bool(_TEST_RUNNER_CMD_RE.search(str(x.get("command") or "")))
        )
        for x in acceptance_results
    )
    changed_norm = {
        str(f).strip().lstrip("./").replace("\\", "/")
        for f in files_changed
        if f
    }

    exec_env = _build_workdir_exec_env(cwd)
    warnings: list[str] = []

    for r in acceptance_results:
        if not isinstance(r, dict) or r.get("passed") or r.get("timed_out"):
            continue
        cmd = str(r.get("command") or "").strip()
        probe = probe_by_cmd.get(cmd)
        post_tail = str(r.get("output_tail") or "")
        r_counts = r.get("counts") or {}
        has_zero_test_counts = (
            int(r_counts.get("passed") or 0) == 0
            and int(r_counts.get("failed") or 0) == 0
            and int(r_counts.get("errors") or 0) == 0
        )

        # Case 2b ([R5-2]): Untouched baseline linter / static-check failure (e.g. `go vet ./pkg/...`,
        # `golangci-lint`, `flake8`, `ruff`, `mypy`, `eslint`) where the command ALSO failed on clean HEAD
        # (`probe is not None`), every `path:line` diagnostic is a subset of the clean-HEAD diagnostics and
        # outside `files_changed`, and at least one real test-runner command passed.
        if (
            probe is not None
            and any_real_test_passed
            and has_zero_test_counts
            and _is_linter_or_static_check_cmd(cmd)
            and r.get("exit_code") not in (None, 0)
        ):
            diag_files = _extract_linter_diagnostic_files(post_tail, cwd)
            probe_diag_files = [
                str(pf).strip().lstrip("./").replace("\\", "/")
                for pf in (probe.get("failing_files") or [])
                if pf
            ]
            probe_err_tail = str(probe.get("error_tail") or "").strip()
            matches_clean_head_diag = (
                set(diag_files) <= set(probe_diag_files)
                if probe_diag_files
                else (bool(probe_err_tail) and all(df in probe_err_tail for df in diag_files))
            )
            untouched_diag = (
                bool(diag_files)
                and matches_clean_head_diag
                and all(df not in changed_norm for df in diag_files)
            )
            identical_clean_head_err = bool(
                not diag_files
                and probe.get("exit_code") == r.get("exit_code")
                and probe_err_tail
                and probe_err_tail in post_tail
            )
            if untouched_diag or identical_clean_head_err:
                r["original_exit_code"] = r.get("exit_code")
                r["exit_code"] = 0
                r["passed"] = True
                r["baseline_neutralized"] = True
                r["preexisting_baseline_failure_ignored"] = True
                warnings.append(
                    f"Untouched baseline linter/static-check failure in `{cmd}` neutralized: "
                    f"diagnostic file(s) {diag_files or ['<clean-HEAD match>']} are outside `files_changed` "
                    "and matched clean-HEAD baseline diagnostics while test-runner acceptance passed."
                )
                continue

        if not probe:
            continue
        post_nodes, post_files = _extract_failing_test_nodes_and_files(post_tail, cwd)
        pytest_runner = _extract_pytest_runner_prefix(cmd)

        # Case 1: Pre-existing clean-HEAD subdir `cwd` quirk (passed on HEAD when run from `passing_cwd`).
        if probe.get("subdir_cwd_passed") and probe.get("passing_cwd") and pytest_runner:
            passing_cwd = str(probe["passing_cwd"])
            sub_dir = cwd / passing_cwd
            # Root run must only have failed inside `passing_cwd` (not on a new root test before it).
            if sub_dir.is_dir() and post_files and all(
                f == passing_cwd or f.startswith(f"{passing_cwd}/") for f in post_files
            ):
                before_files, before_dirs = _snapshot_probe_worktree_state(cwd)
                # Post-worker probes run on top of the worker's edits, so tracked files cannot be reset
                # wholesale; snapshot the change set instead and undo whatever the probe touched.
                probe_snapshot = snapshot_workspace(cwd)
                try:
                    sub_runner = _resolve_pytest_runner_for_subdir(pytest_runner, cwd)
                    sub_rc, sub_to, sub_out = _run_bounded_probe_cmd(
                        f"{sub_runner} -q",
                        cwd=sub_dir,
                        exec_env=exec_env,
                        timeout_s=min(20.0, timeout),
                    )
                finally:
                    _restore_probe_worktree_state(cwd, before_files, before_dirs, reset_tracked=False)
                    check_side_effects(probe_snapshot)
                sub_counts = _parse_test_counts(sub_out)
                sub_ok = (
                    not sub_to
                    and sub_rc == 0
                    and int(sub_counts.get("passed") or 0) >= 1
                    and int(sub_counts.get("failed") or 0) == 0
                    and int(sub_counts.get("errors") or 0) == 0
                )
                if sub_ok:
                    rest_cmd = f"{_strip_pytest_exitfirst_flags(cmd)} --ignore={shlex.quote(passing_cwd)}"
                    before_files2, before_dirs2 = _snapshot_probe_worktree_state(cwd)
                    probe_snapshot2 = snapshot_workspace(cwd)
                    try:
                        rest_rc, rest_to, rest_out = _run_bounded_probe_cmd(
                            rest_cmd,
                            cwd=cwd,
                            exec_env=exec_env,
                            timeout_s=min(30.0, timeout),
                        )
                    finally:
                        _restore_probe_worktree_state(cwd, before_files2, before_dirs2, reset_tracked=False)
                        check_side_effects(probe_snapshot2)
                    rest_counts = _parse_test_counts(rest_out)
                    rest_ok = (
                        not rest_to
                        and rest_rc in (0, 5)
                        and int(rest_counts.get("failed") or 0) == 0
                        and int(rest_counts.get("errors") or 0) == 0
                    )
                    if rest_ok:
                        total_passed = int(sub_counts.get("passed") or 0) + int(rest_counts.get("passed") or 0)
                        r["original_exit_code"] = r.get("exit_code")
                        r["exit_code"] = 0
                        r["passed"] = True
                        r["preexisting_baseline_failure_ignored"] = True
                        r["passing_cwd"] = passing_cwd
                        r["counts"] = {"passed": total_passed, "failed": 0, "errors": 0}
                        warnings.append(
                            f"Baseline `cwd` failure in `{cmd}` ignored: `{passing_cwd}` already failed from repo "
                            f"root on clean HEAD and passed from `{passing_cwd}` ({sub_counts.get('passed', 0)} "
                            f"passed in `{passing_cwd}`, {rest_counts.get('passed', 0)} passed in remaining suite)."
                        )
                        continue

        # Case 2: Pre-existing clean-HEAD compiler CLI config/help error (e.g. `tsc --noEmit` without root
        # tsconfig.json) alongside at least one passing test-runner command: ignore so real passing tests
        # are not blocked by a missing root compiler config.
        if (
            _is_compiler_cli_config_or_help_failure(r)
            and _is_compiler_cli_config_or_help_failure(probe)
            and any_real_test_passed
        ):
            r["original_exit_code"] = r.get("exit_code")
            r["exit_code"] = 0
            r["passed"] = True
            r["preexisting_baseline_failure_ignored"] = True
            warnings.append(
                f"Baseline compiler CLI config failure in `{cmd}` ignored: `{cmd}` already failed on clean HEAD "
                "without test failures while test-runner acceptance passed."
            )
            continue

        # Case 3: Non-subdir clean-HEAD failures remain failing: NEVER flip `passed` to True
        # (an unfixed TDD acceptance test must never auto-apply even if the worker also added
        # unrelated passing tests). Record an informational diagnostic for the Lead only.
        base_nodes = [str(n) for n in (probe.get("failing_nodes") or []) if n]
        if base_nodes and post_nodes and set(post_nodes) <= set(base_nodes):
            warnings.append(
                f"Diagnostic: failing node(s) {post_nodes} in `{cmd}` also failed on clean HEAD "
                f"(acceptance remains failed so unfixed TDD targets are never auto-applied)."
            )

    return warnings


def _is_compiler_cli_config_or_help_failure(res: dict[str, Any]) -> bool:
    if not isinstance(res, dict) or res.get("passed") or res.get("timed_out"):
        return False
    counts = res.get("counts") or {}
    if (
        int(counts.get("passed") or 0) > 0
        or int(counts.get("failed") or 0) > 0
        or int(counts.get("errors") or 0) > 0
    ):
        return False
    tail = str(res.get("output_tail") or res.get("error_tail") or "")
    if re.search(r"\berror TS(?:1|2|4|7)\d{3}\b", tail):
        return False
    return bool(res.get("compiler_cli_config_error")) or bool(_COMPILER_CLI_CONFIG_ERROR_RE.search(tail))


def _only_compiler_cli_config_failures(acc_results: list[dict[str, Any]]) -> bool:
    if not acc_results:
        return False
    has_passing_test_runner = any(
        is_non_smoke_passing_entry(r)
        and int(r.get("exit_code") if r.get("exit_code") is not None else 0) == 0
        and int((r.get("counts") or {}).get("passed") or 0) > 0
        for r in acc_results
    )
    if not has_passing_test_runner:
        return False
    failing = [r for r in acc_results if isinstance(r, dict) and not r.get("passed")]
    return bool(failing) and all(_is_compiler_cli_config_or_help_failure(r) for r in failing)


_DESCRIPTOR_CALL_NAMES = frozenset({"property", "classmethod", "staticmethod", "cached_property"})
_TRY_STAR_TYPES = (ast.Try, getattr(ast, "TryStar", ast.Try))
_MATCH_TYPES = (getattr(ast, "Match", ()),) if hasattr(ast, "Match") else ()


def _is_descriptor_expr(expr: ast.AST | None) -> bool:
    if not isinstance(expr, ast.Call):
        return False
    fn = expr.func
    if isinstance(fn, ast.Name):
        return fn.id in _DESCRIPTOR_CALL_NAMES
    if isinstance(fn, ast.Attribute):
        return fn.attr in _DESCRIPTOR_CALL_NAMES
    return False


def _collect_ast_class_map(tree: ast.Module) -> dict[str, ast.ClassDef]:
    """Map qualified class names (e.g. `Runner` or `Outer.Inner`) in `tree` to their `ast.ClassDef`."""
    mapping: dict[str, ast.ClassDef] = {}

    def _visit_block(stmts: list[ast.stmt], prefix: str) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.ClassDef):
                qual = f"{prefix}.{stmt.name}" if prefix else stmt.name
                mapping[qual] = stmt
                _visit_block(stmt.body, qual)

    _visit_block(tree.body, "")
    return mapping


def _collect_direct_class_attrs(cls_node: ast.ClassDef) -> set[str]:
    """Return non-dunder class-body attributes defined with concrete values directly on `cls_node`."""
    attrs: set[str] = set()

    def _add_target(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            name = t.id
            if not (name.startswith("__") and name.endswith("__")):
                attrs.add(name)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                _add_target(elt)

    for stmt in cls_node.body:
        if isinstance(stmt, ast.Assign) and not _is_descriptor_expr(stmt.value):
            for t in stmt.targets:
                _add_target(t)
        elif (
            isinstance(stmt, ast.AnnAssign)
            and stmt.value is not None
            and not _is_descriptor_expr(stmt.value)
        ):
            _add_target(stmt.target)
    return attrs


def _collect_param_is_not_none_guards(test_expr: ast.AST, param_names: set[str]) -> frozenset[str]:
    """Return optional parameter names guaranteed `is not None` when `test_expr` holds."""
    if (
        isinstance(test_expr, ast.Compare)
        and len(test_expr.ops) == 1
        and isinstance(test_expr.ops[0], ast.IsNot)
        and len(test_expr.comparators) == 1
        and isinstance(test_expr.comparators[0], ast.Constant)
        and test_expr.comparators[0].value is None
        and isinstance(test_expr.left, ast.Name)
        and test_expr.left.id in param_names
    ):
        return frozenset({test_expr.left.id})
    if isinstance(test_expr, ast.BoolOp) and isinstance(test_expr.op, ast.And):
        found: set[str] = set()
        for val in test_expr.values:
            found |= _collect_param_is_not_none_guards(val, param_names)
        return frozenset(found)
    return frozenset()


def _rhs_reads_unshadowed_class_attr(
    val_expr: ast.AST | None,
    attr_name: str,
    self_name: str,
    cls_name: str,
    *,
    already_written: bool,
) -> bool:
    """Return True if `val_expr` copies/wraps `self.<attr_name>` or `<cls_name>.<attr_name>` without conditional fallback."""
    if val_expr is None or already_written:
        return False
    has_ref = False
    for node in ast.walk(val_expr):
        if isinstance(node, (ast.IfExp, ast.BoolOp, ast.Lambda)):
            return False
        if (
            isinstance(node, ast.Attribute)
            and node.attr == attr_name
            and isinstance(node.value, ast.Name)
            and node.value.id in (self_name, cls_name)
        ):
            has_ref = True
    return has_ref


def _dotted_expr_parts(expr: ast.AST) -> list[str] | None:
    """Unroll a dotted `ast.Name` / `ast.Attribute` chain (e.g. `pkg.base.Base` -> `['pkg', 'base', 'Base']`)."""
    parts: list[str] = []
    cur: ast.AST = expr
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


def _collect_init_self_writes(
    cls_node: ast.ClassDef,
) -> dict[str, list[tuple[str, int, bool]]]:
    """Collect `self.<attr>` writes in `__init__` and `self.<helper>` methods called from `__init__`.

    Returns `{attr: [(method_name, lineno, is_guarded_or_class_default_copy), ...]}`.
    """
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        stmt.name: stmt
        for stmt in cls_node.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "__init__" not in methods:
        return {}

    writes: dict[str, list[tuple[str, int, bool]]] = {}
    visited: set[str] = set()
    queue: list[str] = ["__init__"]

    while queue:
        m_name = queue.pop(0)
        if m_name in visited or m_name not in methods:
            continue
        visited.add(m_name)
        fn_node = methods[m_name]
        all_pos = list(fn_node.args.posonlyargs) + list(fn_node.args.args)
        if not all_pos:
            continue
        self_name = all_pos[0].arg

        # Identify parameters with default `None` on `__init__` for caller-override guard detection
        none_default_params: set[str] = set()
        if m_name == "__init__":
            defaults = list(fn_node.args.defaults)
            non_self = all_pos[1:]
            pad = len(non_self) - len(defaults)
            for idx_p, arg_node in enumerate(non_self):
                def_idx = idx_p - pad
                if def_idx >= 0 and isinstance(defaults[def_idx], ast.Constant) and defaults[def_idx].value is None:
                    none_default_params.add(arg_node.arg)
            for kw_arg, kw_def in zip(fn_node.args.kwonlyargs, fn_node.args.kw_defaults):
                if isinstance(kw_def, ast.Constant) and kw_def.value is None:
                    none_default_params.add(kw_arg.arg)

        def _is_guarded_rhs(attr_name: str, val_expr: ast.AST | None, guarded_params: frozenset[str]) -> bool:
            if guarded_params and val_expr is not None:
                if any(isinstance(n, ast.Name) and n.id in guarded_params for n in ast.walk(val_expr)):
                    return True
            return _rhs_reads_unshadowed_class_attr(
                val_expr,
                attr_name,
                self_name,
                cls_node.name,
                already_written=(attr_name in writes),
            )

        def _scan_expr(expr: ast.AST | None, guarded_params: frozenset[str], fallback_lineno: int) -> None:
            if expr is None:
                return
            for sub in ast.walk(expr):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(sub, ast.Call):
                    if (
                        isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == self_name
                        and sub.func.attr in methods
                        and sub.func.attr not in visited
                    ):
                        queue.append(sub.func.attr)
                    elif (
                        isinstance(sub.func, ast.Name)
                        and sub.func.id == "setattr"
                        and len(sub.args) == 3
                        and isinstance(sub.args[0], ast.Name)
                        and sub.args[0].id == self_name
                        and isinstance(sub.args[1], ast.Constant)
                        and isinstance(sub.args[1].value, str)
                    ):
                        s_attr = sub.args[1].value
                        if s_attr and not (s_attr.startswith("__") and s_attr.endswith("__")):
                            is_g = _is_guarded_rhs(s_attr, sub.args[2], guarded_params)
                            writes.setdefault(s_attr, []).append(
                                (m_name, getattr(sub, "lineno", fallback_lineno), is_g)
                            )

        def _walk_stmts(stmts: list[ast.stmt], guarded_params: frozenset[str]) -> None:
            for st in stmts:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(st, ast.If):
                    _scan_expr(st.test, guarded_params, st.lineno)
                    g_params = guarded_params | _collect_param_is_not_none_guards(st.test, none_default_params)
                    _walk_stmts(st.body, g_params)
                    _walk_stmts(st.orelse, guarded_params)
                    continue
                if _MATCH_TYPES and isinstance(st, _MATCH_TYPES):
                    _scan_expr(getattr(st, "subject", None), guarded_params, st.lineno)
                    for mc in getattr(st, "cases", []):
                        _scan_expr(getattr(mc, "guard", None), guarded_params, st.lineno)
                        _walk_stmts(mc.body, guarded_params)
                    continue
                if isinstance(st, (ast.For, ast.AsyncFor)):
                    _scan_expr(st.iter, guarded_params, st.lineno)
                    _walk_stmts(st.body, guarded_params)
                    _walk_stmts(st.orelse, guarded_params)
                    continue
                if isinstance(st, ast.While):
                    _scan_expr(st.test, guarded_params, st.lineno)
                    _walk_stmts(st.body, guarded_params)
                    _walk_stmts(st.orelse, guarded_params)
                    continue
                if isinstance(st, (ast.With, ast.AsyncWith)):
                    for item in st.items:
                        _scan_expr(item.context_expr, guarded_params, st.lineno)
                    _walk_stmts(st.body, guarded_params)
                    continue
                if isinstance(st, _TRY_STAR_TYPES):
                    _walk_stmts(st.body, guarded_params)
                    for h in getattr(st, "handlers", []):
                        if isinstance(h, ast.ExceptHandler):
                            _walk_stmts(h.body, guarded_params)
                    _walk_stmts(st.orelse, guarded_params)
                    _walk_stmts(st.finalbody, guarded_params)
                    continue

                # Simple statement: scan expressions for helper calls and `setattr(self, ...)`
                _scan_expr(st, guarded_params, st.lineno)
                if isinstance(st, ast.Assign):
                    for tgt in st.targets:
                        targets_iter = tgt.elts if isinstance(tgt, (ast.Tuple, ast.List)) else [tgt]
                        for item_t in targets_iter:
                            if (
                                isinstance(item_t, ast.Attribute)
                                and isinstance(item_t.value, ast.Name)
                                and item_t.value.id == self_name
                            ):
                                is_g = _is_guarded_rhs(item_t.attr, st.value, guarded_params)
                                writes.setdefault(item_t.attr, []).append((m_name, st.lineno, is_g))
                elif (
                    isinstance(st, ast.AnnAssign)
                    and st.value is not None
                    and isinstance(st.target, ast.Attribute)
                    and isinstance(st.target.value, ast.Name)
                    and st.target.value.id == self_name
                ):
                    is_g = _is_guarded_rhs(st.target.attr, st.value, guarded_params)
                    writes.setdefault(st.target.attr, []).append((m_name, st.lineno, is_g))
                elif (
                    isinstance(st, ast.AugAssign)
                    and isinstance(st.target, ast.Attribute)
                    and isinstance(st.target.value, ast.Name)
                    and st.target.value.id == self_name
                ):
                    writes.setdefault(st.target.attr, []).append((m_name, st.lineno, False))

        _walk_stmts(fn_node.body, frozenset())

    return writes


def _collect_top_level_import_map(
    tree: ast.Module,
    current_rel: str,
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Return `(symbol_imports, module_aliases)` from top-level imports in `tree`.

    `symbol_imports[local_name] = (module_dotted_base, exported_symbol_name)`
    `module_aliases[local_alias] = module_dotted_base`
    """
    pkg_parts = list(Path(current_rel).parent.parts)
    if pkg_parts == ["."]:
        pkg_parts = []
    sym_map: dict[str, tuple[str, str]] = {}
    mod_map: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            base_parts = list(pkg_parts)
            if stmt.level > 0:
                trim = stmt.level - 1
                base_parts = base_parts[: max(0, len(base_parts) - trim)] if trim > 0 else base_parts
                mod_parts = base_parts + (stmt.module.split(".") if stmt.module else [])
            else:
                mod_parts = stmt.module.split(".") if stmt.module else []
            mod_dotted = ".".join(p for p in mod_parts if p)
            if not mod_dotted:
                continue
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                sym_map[local] = (mod_dotted, alias.name)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.asname:
                    mod_map[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    mod_map[top] = top
    return sym_map, mod_map


def detect_class_attribute_shadowing(
    worktree_root: Path,
    files_changed: list[str],
    *,
    baseline_ref: str = "HEAD",
) -> list[str]:
    """Detect non-test `.py` edits where `C.__init__` (or helpers it calls) newly shadows an existing `HEAD` class attribute."""
    if not files_changed or not worktree_root.is_dir():
        return []

    parsed_head_cache: dict[str, tuple[ast.Module, dict[str, ast.ClassDef]] | None] = {}

    def _load_head_module(rel_path: str) -> tuple[ast.Module, dict[str, ast.ClassDef]] | None:
        if rel_path in parsed_head_cache:
            return parsed_head_cache[rel_path]
        res = _run_git(["show", f"{baseline_ref}:{rel_path}"], cwd=worktree_root, check=False)
        if res.returncode != 0 or not res.stdout:
            parsed_head_cache[rel_path] = None
            return None
        try:
            mod_tree = ast.parse(res.stdout, filename=rel_path)
        except SyntaxError:
            parsed_head_cache[rel_path] = None
            return None
        entry = (mod_tree, _collect_ast_class_map(mod_tree))
        parsed_head_cache[rel_path] = entry
        return entry

    def _resolve_dotted_module_rel(mod_dotted: str) -> str | None:
        rel_stem = mod_dotted.replace(".", "/")
        cands = [f"{rel_stem}.py", f"{rel_stem}/__init__.py"]
        if not rel_stem.startswith("src/"):
            cands.extend([f"src/{rel_stem}.py", f"src/{rel_stem}/__init__.py"])
        for cand in cands:
            if _load_head_module(cand) is not None:
                return cand
        return None

    def _resolve_imported_chain(full_parts: list[str]) -> tuple[str, str] | None:
        for split_idx in range(len(full_parts) - 1, 0, -1):
            cand_rel = _resolve_dotted_module_rel(".".join(full_parts[:split_idx]))
            if cand_rel is not None:
                return cand_rel, ".".join(full_parts[split_idx:])
        return None

    contract_cache: dict[tuple[str, str], tuple[set[str], set[str]]] = {}

    def _effective_head_contract(
        rel_path: str,
        qual_cls: str,
        visiting: frozenset[tuple[str, str]] = frozenset(),
    ) -> tuple[set[str], set[str]]:
        key = (rel_path, qual_cls)
        if key in contract_cache:
            return contract_cache[key]
        if key in visiting:
            return set(), set()
        loaded = _load_head_module(rel_path)
        if loaded is None:
            return set(), set()
        mod_tree, mod_classes = loaded
        sym_imports, mod_aliases = _collect_top_level_import_map(mod_tree, rel_path)
        next_visiting = visiting | {key}
        cls_node = mod_classes.get(qual_cls)
        if cls_node is None:
            q_parts = qual_cls.split(".")
            reexp_chain: list[str] = []
            if q_parts[0] in sym_imports:
                imp_mod, orig_sym = sym_imports[q_parts[0]]
                reexp_chain = [*imp_mod.split("."), orig_sym, *q_parts[1:]]
            elif q_parts[0] in mod_aliases and len(q_parts) >= 2:
                reexp_chain = [*mod_aliases[q_parts[0]].split("."), *q_parts[1:]]
            resolved_reexp = _resolve_imported_chain(reexp_chain) if reexp_chain else None
            if resolved_reexp is not None:
                res = _effective_head_contract(resolved_reexp[0], resolved_reexp[1], next_visiting)
                contract_cache[key] = res
                return res
            return set(), set()

        eff_attrs = set(_collect_direct_class_attrs(cls_node))
        eff_unguarded = {
            a
            for a, locs in _collect_init_self_writes(cls_node).items()
            if any(not is_g for _, _, is_g in locs)
        }
        parent_prefix = qual_cls.rsplit(".", 1)[0] if "." in qual_cls else ""

        for base_expr in cls_node.bases:
            b_rel: str | None = None
            b_qual: str | None = None
            b_parts = _dotted_expr_parts(base_expr)
            if not b_parts:
                continue
            joined_base = ".".join(b_parts)
            scoped_cand = f"{parent_prefix}.{joined_base}" if parent_prefix else joined_base
            if scoped_cand in mod_classes:
                b_rel, b_qual = rel_path, scoped_cand
            elif joined_base in mod_classes:
                b_rel, b_qual = rel_path, joined_base
            else:
                imp_chain: list[str] = []
                if b_parts[0] in sym_imports:
                    imp_mod, orig_sym = sym_imports[b_parts[0]]
                    imp_chain = [*imp_mod.split("."), orig_sym, *b_parts[1:]]
                elif b_parts[0] in mod_aliases and len(b_parts) >= 2:
                    imp_chain = [*mod_aliases[b_parts[0]].split("."), *b_parts[1:]]
                resolved_imp = _resolve_imported_chain(imp_chain) if imp_chain else None
                if resolved_imp is not None:
                    b_rel, b_qual = resolved_imp
            if b_rel and b_qual:
                inh_attrs, inh_unguarded = _effective_head_contract(b_rel, b_qual, next_visiting)
                eff_attrs |= inh_attrs
                eff_unguarded |= inh_unguarded

        contract_cache[key] = (eff_attrs, eff_unguarded)
        return eff_attrs, eff_unguarded

    warnings: list[str] = []
    for rel in files_changed:
        if not rel.endswith(".py") or _is_test_or_doc_path(rel):
            continue
        wt_file = worktree_root / rel
        if not wt_file.is_file() or wt_file.is_symlink():
            continue
        loaded_head = _load_head_module(rel)
        if loaded_head is None:
            continue
        _head_tree, head_classes = loaded_head
        try:
            wt_tree = ast.parse(wt_file.read_text(encoding="utf-8", errors="replace"), filename=rel)
        except SyntaxError:
            continue
        wt_classes = _collect_ast_class_map(wt_tree)
        for qual_cls in head_classes:
            wt_cls = wt_classes.get(qual_cls)
            if wt_cls is None:
                continue
            head_attrs, head_unguarded_writes = _effective_head_contract(rel, qual_cls)
            if not head_attrs:
                continue
            wt_init_writes = _collect_init_self_writes(wt_cls)
            for attr, locs in wt_init_writes.items():
                if attr not in head_attrs or attr in head_unguarded_writes:
                    continue
                unguarded_locs = [(m_name, ln) for m_name, ln, is_guarded in locs if not is_guarded]
                if not unguarded_locs:
                    continue
                m_name, lineno = unguarded_locs[0]
                loc_method = f"{qual_cls}.__init__" if m_name == "__init__" else f"{qual_cls}.{m_name} (called from __init__)"
                warnings.append(
                    f"ClassAttributeShadowed: {rel}:{lineno}: {loc_method} newly assigns self.{attr}, "
                    f"shadowing existing class attribute {qual_cls}.{attr} on HEAD "
                    f"(breaks monkeypatch.setattr({qual_cls}, {attr!r}, ...) unless only set when explicitly passed by caller)."
                )
    return warnings


_TEST_DIR_SEGMENTS = frozenset({"tests", "test", "__tests__"})
_TEST_FILENAME_RE = re.compile(
    r"^(?:test_.+\.(?:py|rb|rs|c|cc|cpp|sh)|.+_(?:test|spec)\.(?:py|go|rs|ts|tsx|js|jsx|rb)|.+\.(?:test|spec)\.(?:ts|tsx|js|jsx|mjs|cjs)|.+Test\.(?:java|kt))$"
)
_NEW_TEST_BLOCK_LINE_RE = re.compile(
    r"^\+\s*(?:(?:async\s+)?def\s+test_\w+|func\s+Test\w+|(?:it|test|describe)\s*\(|#\[test\]|@Test\b|name:\s*[\"'])"
)
_STUB_OR_MOCK_MUTATION_RE = re.compile(
    r"(?:\b(?:httpmock|Mockito|mockito|GoMock|gomock|Sinon|sinon|nock|respx|mocker|monkeypatch|MagicMock|AsyncMock|Mock|mock|mocks|mocked|unmock|Stub|stub|stubs|stubbed)\b"
    r"|\b(?:mock|stub|Mock|Stub)_[A-Za-z0-9_]+\b"
    r"|\b[A-Za-z0-9_]+_(?:mock|stub|mocks|stubs)\b"
    r"|\b(?:mock|stub|Mock|Stub)[A-Z][A-Za-z0-9_]*\b"
    r"|\b[a-z][A-Za-z0-9_]*(?:Mock|Stub|Mocks|Stubs)\b"
    r"|\b(?:patch|spyOn)\s*\(|\bresponses\.add\b)"
)
_EXISTING_TEST_EXPECTATION_RE = re.compile(
    r"(?:\b(?:wantStdout|wantStderr|wantOut|wantErr|expectedOutput|expectedStdout|expectedStderr|expectedError)\s*[:=]"
    r"|\bassert(?:Equal|Equals|StrictEqual|DeepEqual|That|True|False)?\b"
    r"|\bexpect\s*\([^)]*\)\s*\.\s*(?:toEqual|toStrictEqual|toBe|toMatch|toThrow)\b)"
)
_TEST_FUNC_HEADER_RE = re.compile(
    r"(?:(?:async\s+)?def\s+(test_\w+)|func\s+(Test\w+)|(?:it|test)\s*\(\s*(['\"`][^'\"`]+['\"`])|(?:public\s+)?void\s+(test\w+))"
)
_SUBTEST_CASE_RE = re.compile(r"^\s*name\s*:\s*(['\"`][^'\"`]+['\"`])")
_EXPLICIT_TESTS_ONLY_RE = re.compile(
    r"\b(?:only|solely)\s+(?:add|write|create|update)\s+(?:unit\s+|integration\s+|regression\s+)?tests?\b"
    r"|\bwithout\s+modifying\s+(?:any\s+)?(?:production|source|implementation)\b"
    r"|\bdo\s+not\s+modify\s+(?:any\s+)?(?:production|source|implementation)\b",
    re.I,
)
_EXPLICIT_EXISTING_TEST_UPDATE_RE = re.compile(
    r"\b(?:update|modify|adjust|adapt|rewrite|sync)\s+"
    r"(?:the\s+|any\s+|all\s+)?(?:existing\s+|affected\s+|corresponding\s+|related\s+)?"
    r"(?:unit\s+|integration\s+|regression\s+)?(?:test(?:s|\s+files?|\s+cases?|\s+expectations?|\s+assertions?|\s+snapshots?)|test_\w+\.py|\S+_test\.go|\S+\.(?:test|spec)\.[jt]sx?)\b"
    r"|\b(?:change|fix)\s+(?:the\s+|any\s+|all\s+)?existing\s+(?:unit\s+|integration\s+|regression\s+)?test(?:s|\s+expectations?|\s+assertions?|\s+cases?|\s+files?)\b"
    r"|\btests?\s+to\s+expect\b",
    re.I,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[.;:!?\n]|\b(?:but|however)\b", re.I)
_CLAUSE_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|don['\u2019]t|does\s+not|doesn['\u2019]t|did\s+not|didn['\u2019]t"
    r"|never|must\s+not|mustn['\u2019]t|should\s+not|shouldn['\u2019]t"
    r"|can\s*not|cannot|can['\u2019]t|avoid|refrain\s+from|without|no|not"
    r"|rather\s+than|instead\s+of|forbid\w*|prohibit\w*)\b",
    re.I,
)
_TEST_AUTHORING_INTENT_RE = re.compile(
    r"\b(?:write|add|create|expand|increase|improve)\s+"
    r"(?:a\s+|an\s+|new\s+|more\s+|missing\s+|additional\s+)?"
    r"(?:(?:unit|integration|regression|edge[- ]case|table[- ]driven|parameterized|parametrized)\s+)*tests?\b"
    r"|\b(?:increase|improve|expand|add)\s+(?:unit\s+|branch\s+|line\s+)?test\s+coverage\b",
    re.I,
)
_PROD_CHANGE_INTENT_RE = re.compile(
    r"\b(?:fix|implement|refactor|support|resolve|repair|patch|migrate|convert|replace"
    r"|make|handle|change|ensure|rename|remove|delete|allow|enable|disable|expose|export|preserve|prevent|avoid)\b"
    r"|\badd\s+(?:a\s+|an\s+|the\s+)?(?:--?[A-Za-z0-9_-]+|[A-Za-z0-9_-]+\s+(?:flag|option|parameter|argument|field|method|function|endpoint|command|subcommand|route|header|hook|handler|helper|class|module|config|setting|support))\b"
    r"|\bupdate\s+(?:the\s+)?(?:implementation|function|method|class|handler|logic|module|behavior)\b",
    re.I,
)


def _prompt_requests_test_authoring(prompt_text: str) -> bool:
    """Return True when the prompt explicitly asks for test-only edits or ordinary test/coverage authoring."""
    if not prompt_text:
        return False
    if _EXPLICIT_TESTS_ONLY_RE.search(prompt_text):
        return True
    if _TEST_AUTHORING_INTENT_RE.search(prompt_text) and not _PROD_CHANGE_INTENT_RE.search(prompt_text):
        return True
    return False


_FORBID_EXISTING_TEST_UPDATE_RE = re.compile(
    r"\b(?:do\s+not|don['\u2019]t|never|must\s+not|should\s+not|cannot|can['\u2019]t|without|avoid|refrain\s+from)\s+"
    r"(?:ever\s+)?(?:update|updating|modify|modifying|alter|altering|change|changing|rewrite|rewriting|edit|editing|touch|touching|weaken|weakening)\s+"
    r"(?:the\s+|any\s+|all\s+)?(?:pre-?existing\s+|existing\s+)?(?:unit\s+|integration\s+|regression\s+)?test",
    re.I,
)


def _prompt_forbids_existing_test_updates(prompt_text: str) -> bool:
    """Return True when the prompt explicitly forbids modifying/rewriting existing tests."""
    if not prompt_text:
        return False
    return bool(_FORBID_EXISTING_TEST_UPDATE_RE.search(prompt_text))


def _prompt_requests_existing_test_updates(prompt_text: str) -> bool:
    """Return True when the prompt explicitly instructs updating existing test expectations/assertions (not negated)."""
    if not prompt_text:
        return False
    for m in _EXPLICIT_EXISTING_TEST_UPDATE_RE.finditer(prompt_text):
        prefix = prompt_text[: m.start()]
        clause_parts = _CLAUSE_BOUNDARY_RE.split(prefix)
        local_clause = clause_parts[-1] if clause_parts else prefix
        if _CLAUSE_NEGATION_RE.search(local_clause):
            continue
        return True
    return False


def _unexempted_stub_mutation_warnings(warnings: list[str] | None, prompt_text: str) -> list[str]:
    """Filter test-mutation warnings against explicit prompt intent."""
    if not warnings:
        return []
    if _prompt_requests_test_authoring(prompt_text):
        return []
    if _prompt_requests_existing_test_updates(prompt_text):
        return [w for w in warnings if not str(w).startswith("ExistingTestExpectationsModified:")]
    return list(warnings)


def _is_test_file_path(rel_path: str) -> bool:
    """Return True when `rel_path` is a unit/integration/spec test file across supported languages."""
    norm = rel_path.replace("\\", "/").strip("/")
    if not norm:
        return False
    parts = norm.split("/")
    if any(seg in _TEST_DIR_SEGMENTS for seg in parts[:-1]):
        return True
    return bool(_TEST_FILENAME_RE.match(parts[-1]))


def detect_existing_test_stub_mutations(
    worktree_root: Path,
    diff_text: str,
    files_changed: list[str],
    *,
    baseline_ref: str = "HEAD",
) -> list[str]:
    """Warn when a worker alters mock/stub expectations across >= 3 existing test blocks in a pre-existing test file."""
    if not files_changed or not diff_text.strip() or not worktree_root.is_dir():
        return []
    test_targets = {
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if _is_test_file_path(rel)
    }
    if not test_targets:
        return []

    preexisting_tests: set[str] = set()
    for rel in test_targets:
        cat_res = _run_git(["cat-file", "-e", f"{baseline_ref}:{rel}"], cwd=worktree_root, check=False)
        if cat_res.returncode == 0:
            preexisting_tests.add(rel)
    if not preexisting_tests:
        return []

    file_chunks: dict[str, list[str]] = {}
    cur_rel: str | None = None
    for dline in diff_text.splitlines():
        if dline.startswith("diff --git "):
            hdr = parse_diff_git_header(dline)
            cpath = ((hdr[1] or hdr[0]) if hdr else "").replace("\\", "/").strip("/")
            cur_rel = cpath if cpath in preexisting_tests else None
            if cur_rel is not None:
                file_chunks.setdefault(cur_rel, [])
            continue
        if cur_rel is not None:
            file_chunks[cur_rel].append(dline)

    from collections import Counter

    warnings: list[str] = []
    for rel in sorted(file_chunks):
        lines = file_chunks[rel]
        hunks: list[tuple[int, str, list[str]]] = []
        cur_old_len = 0
        cur_hdr_ctx = ""
        cur_hunk: list[str] | None = None
        for ln in lines:
            m_hunk = re.match(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$", ln)
            if m_hunk:
                if cur_hunk is not None:
                    hunks.append((cur_old_len, cur_hdr_ctx, cur_hunk))
                cur_old_len = int(m_hunk.group(2)) if m_hunk.group(2) is not None else 1
                cur_hdr_ctx = (m_hunk.group(5) or "").strip()
                cur_hunk = []
                continue
            if cur_hunk is not None and not ln.startswith(("--- ", "+++ ")):
                cur_hunk.append(ln)
        if cur_hunk is not None:
            hunks.append((cur_old_len, cur_hdr_ctx, cur_hunk))

        mutated_stub_blocks = 0
        mutated_expectation_funcs: set[str] = set()
        for hunk_idx, (old_len, hdr_ctx, hlines) in enumerate(hunks):
            if old_len <= 0:
                continue
            m_hdr_fn = _TEST_FUNC_HEADER_RE.search(hdr_ctx)
            base_fn_id = (
                f"fn:{next(g for g in m_hdr_fn.groups() if g)}"
                if m_hdr_fn
                else f"hunk:{hunk_idx}"
            )
            cur_block_id = base_fn_id
            clusters: list[tuple[str, list[str]]] = []
            cur_cluster: list[str] = []
            cur_cluster_block_id = cur_block_id
            for hl in hlines:
                body = hl[1:] if hl else ""
                if hl.startswith((" ", "-")):
                    m_fn = _TEST_FUNC_HEADER_RE.search(body)
                    if m_fn and not hl.startswith("+"):
                        if cur_cluster and hl.startswith(" "):
                            clusters.append((cur_cluster_block_id, cur_cluster))
                            cur_cluster = []
                        base_fn_id = f"fn:{next(g for g in m_fn.groups() if g)}"
                        cur_block_id = base_fn_id
                    else:
                        m_sub = _SUBTEST_CASE_RE.match(body)
                        if m_sub:
                            if cur_cluster and hl.startswith(" "):
                                clusters.append((cur_cluster_block_id, cur_cluster))
                                cur_cluster = []
                            cur_block_id = f"{base_fn_id}::{m_sub.group(1)}"
                if hl.startswith(("+", "-")):
                    if not cur_cluster:
                        cur_cluster_block_id = cur_block_id
                    cur_cluster.append(hl)
                elif cur_cluster:
                    clusters.append((cur_cluster_block_id, cur_cluster))
                    cur_cluster = []
            if cur_cluster:
                clusters.append((cur_cluster_block_id, cur_cluster))
            for block_id, cluster in clusters:
                if not any(cl.startswith("-") for cl in cluster):
                    continue
                if any(cl.startswith("+") and _NEW_TEST_BLOCK_LINE_RE.search(cl) for cl in cluster):
                    continue
                if any(_STUB_OR_MOCK_MUTATION_RE.search(cl[1:]) for cl in cluster):
                    mutated_stub_blocks += 1
                minus_exp = [
                    cl[1:].strip()
                    for cl in cluster
                    if cl.startswith("-") and _EXISTING_TEST_EXPECTATION_RE.search(cl[1:])
                ]
                plus_exp = [
                    cl[1:].strip()
                    for cl in cluster
                    if cl.startswith("+") and _EXISTING_TEST_EXPECTATION_RE.search(cl[1:])
                ]
                if minus_exp:
                    minus_counts = Counter(minus_exp)
                    plus_counts = Counter(plus_exp)
                    if any(plus_counts[m] < cnt for m, cnt in minus_counts.items()):
                        mutated_expectation_funcs.add(block_id)
        if mutated_stub_blocks >= 3:
            warnings.append(
                f"ExistingTestStubsModified: {rel} modified mock/stub expectations across {mutated_stub_blocks} "
                f"pre-existing test blocks; inspect get_diff (focus_symbols) to verify no unrequested API/HTTP "
                f"calls or breaking contract changes were masked."
            )
        elif len(mutated_expectation_funcs) >= 2:
            warnings.append(
                f"ExistingTestExpectationsModified: {rel} altered pre-existing test expectations across "
                f"{len(mutated_expectation_funcs)} test functions; verify production code matches the specification "
                f"rather than rewriting existing test assertions."
            )
    return warnings


import builtins as _py_builtins

_PROMPT_SYMBOL_STOPLIST = frozenset({
    "True", "False", "None", "self", "cls", "args", "kwargs", "dict", "list", "tuple", "set",
    "str", "int", "float", "bool", "bytes", "object", "type", "super", "Path", "Any", "Optional",
    "Union", "Callable", "Iterable", "Iterator", "Sequence", "Mapping", "MutableMapping", "Literal",
    "Exception", "BaseException", "ValueError", "TypeError", "RuntimeError", "KeyError", "IndexError",
    "AttributeError", "FileNotFoundError", "OSError", "NotImplementedError", "StopIteration",
    "pytest", "unittest", "git", "python", "python3", "bash", "sh", "node", "npm", "npx", "cargo",
    "rustc", "go", "pip", "uv", "poetry", "ruff", "mypy", "flake8", "black", "isort",
    "main", "init", "setup", "test", "tests", "src", "pkg", "app", "lib", "core", "utils", "helpers",
    "todo", "note", "warning", "error", "info", "debug", "true", "false", "null", "none",
    "view_file", "edit_file", "create_file", "run_command", "search_dir", "list_dir", "find_file",
    "ctx_get", "ctx_search", "ctx_query", "report_to_lead", "delegate_start", "delegate_continue",
    "delegate_apply", "delegate_result", "delegate_status", "delegate_cancel", "delegate_revert",
    "delegate_to_gemini", "ask_gemini", "get_diff", "cost_report", "probe_writers",
    "delegate_file_batch", "draft_document", "write_tests_from_cases", "replicate_edit", "list_jobs",
    "allowed_paths", "files_changed", "acceptance", "acceptance_passed", "acceptance_results",
    "scope_violations", "scope_violation", "missing_prompt_symbols", "auto_apply", "in_place",
    "follow_up_prompt", "job_id", "timeout_seconds", "max_tool_calls", "max_total_tokens",
    "thinking_level", "allow_scope_violations", "include_paths", "force_retry", "resume_job_id",
} | {name for name in dir(_py_builtins) if not name.startswith("_")})


def _repo_has_explicit_pytest_config(cwd: Path) -> bool:
    """Return True when `cwd` declares explicit pytest configuration."""
    if (cwd / "pytest.ini").is_file() or (cwd / "conftest.py").is_file():
        return True
    for cfg_name, marker in (("pyproject.toml", "[tool.pytest"), ("setup.cfg", "[tool:pytest]"), ("tox.ini", "[pytest]")):
        cfg_p = cwd / cfg_name
        if cfg_p.is_file():
            try:
                if marker in cfg_p.read_text(encoding="utf-8", errors="replace"):
                    return True
            except OSError:
                pass
    return False


def _has_python_bin_test_runner(cwd: Path) -> bool:
    """Return True if `cwd/bin/test` is a Python test runner script."""
    bin_test = cwd / "bin" / "test"
    if not bin_test.is_file():
        return False
    try:
        head = bin_test.read_bytes()[:256].decode("utf-8", errors="replace")
    except OSError:
        return False
    first_line = head.splitlines()[0] if head else ""
    return "python" in first_line.lower()


def _detect_default_pytest_runner(cwd: Path) -> str:
    """Detect the best Python test runner prefix for `cwd` when the caller supplied no acceptance commands.
    Returns `""` if no compatible Python test runner is installed in `cwd`'s environment."""
    # Repositories with a dedicated Python `bin/test` runner and legacy `utilities/pytest.py` shims
    # (without modern pytest config) fail under modern pytest (`module 'py' has no attribute 'test'`).
    has_bin_test = _has_python_bin_test_runner(cwd)
    has_legacy_pytest_shim = (
        has_bin_test
        and not _repo_has_explicit_pytest_config(cwd)
        and next(cwd.glob("*/utilities/pytest.py"), None) is not None
    )
    if has_legacy_pytest_shim:
        for py_cand in (".venv/bin/python", "venv/bin/python"):
            if (cwd / py_cand).is_file():
                return f"{py_cand} bin/test"
        return "python3 bin/test"
    for rel_cand in (".venv/bin/pytest", "venv/bin/pytest"):
        if (cwd / rel_cand).is_file():
            return rel_cand
    exec_env = _build_workdir_exec_env(cwd)
    if shutil.which("pytest", path=exec_env.get("PATH")):
        return "pytest"
    for py_cand in (".venv/bin/python", "venv/bin/python"):
        if (cwd / py_cand).is_file():
            probe = subprocess.run(
                [str(cwd / py_cand), "-c", "import pytest"],
                cwd=str(cwd),
                env=exec_env,
                capture_output=True,
                check=False,
            )
            if probe.returncode == 0:
                return f"{py_cand} -m pytest"
    probe_sys = subprocess.run(
        ["python3", "-c", "import pytest"],
        cwd=str(cwd),
        env=exec_env,
        capture_output=True,
        check=False,
    )
    if probe_sys.returncode == 0:
        return "python3 -m pytest"
    if has_bin_test:
        for py_cand in (".venv/bin/python", "venv/bin/python"):
            if (cwd / py_cand).is_file():
                return f"{py_cand} bin/test"
        return "python3 bin/test"
    return ""


_JS_TS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_JS_TS_TEST_RE = re.compile(r"\.(?:test|spec)\.(?:ts|tsx|js|jsx|mjs|cjs)$", re.I)


def _detect_node_runner_info(cwd: Path, exec_env: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Inspect `package.json` and lockfiles to detect the Node/TypeScript test runner for `cwd`."""
    pkg_json = cwd / "package.json"
    if not pkg_json.is_file():
        return None
    env_map = exec_env if exec_env is not None else _build_workdir_exec_env(cwd)
    path_str = env_map.get("PATH")
    try:
        pkg = json.loads(pkg_json.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        pkg = {}
    if not isinstance(pkg, dict):
        pkg = {}
    scripts = pkg.get("scripts") if isinstance(pkg.get("scripts"), dict) else {}
    test_script = str((scripts or {}).get("test") or "").strip()
    is_placeholder = "no test specified" in test_script.lower()
    deps: dict[str, Any] = {}
    for dep_key in ("dependencies", "devDependencies"):
        sub = pkg.get(dep_key)
        if isinstance(sub, dict):
            deps.update(sub)
    pm_field = str(pkg.get("packageManager") or "").lower()
    if (cwd / "pnpm-lock.yaml").is_file() or pm_field.startswith("pnpm@"):
        pm = "pnpm"
    elif (cwd / "yarn.lock").is_file() or pm_field.startswith("yarn@"):
        pm = "yarn"
    elif (cwd / "bun.lockb").is_file() or (cwd / "bun.lock").is_file() or pm_field.startswith("bun@"):
        pm = "bun"
    else:
        pm = "npm"

    framework = ""
    if "vitest" in test_script or "vitest" in deps:
        framework = "vitest"
    elif "jest" in test_script or "jest" in deps:
        framework = "jest"
    elif "mocha" in test_script or "mocha" in deps:
        framework = "mocha"
    elif "node --test" in test_script:
        framework = "node-test"
    elif pm == "bun" and (not test_script or "bun test" in test_script):
        framework = "bun-test"

    node_bin_dir = cwd / "node_modules" / ".bin"
    pm_available = bool(shutil.which(pm, path=path_str))
    node_available = bool(shutil.which("node", path=path_str))
    if framework in ("vitest", "jest", "mocha"):
        local_bin = node_bin_dir / framework
        available = local_bin.is_file() or ((cwd / "node_modules").is_dir() and pm_available)
    elif framework == "node-test":
        available = node_available
    elif framework == "bun-test":
        available = bool(shutil.which("bun", path=path_str))
    else:
        available = bool(test_script and not is_placeholder and pm_available and (cwd / "node_modules").is_dir())

    primary_cmd = ""
    if framework == "bun-test":
        primary_cmd = "bun test"
    elif framework == "node-test":
        primary_cmd = "node --test"
    elif framework == "vitest" and test_script.strip() == "vitest":
        primary_cmd = "./node_modules/.bin/vitest run" if (node_bin_dir / "vitest").is_file() else "npx vitest run"
    elif test_script and not is_placeholder:
        primary_cmd = f"{pm} test"
    elif framework == "vitest":
        primary_cmd = "./node_modules/.bin/vitest run" if (node_bin_dir / "vitest").is_file() else "npx vitest run"
    elif framework == "jest":
        primary_cmd = (
            "./node_modules/.bin/jest --watchAll=false"
            if (node_bin_dir / "jest").is_file()
            else "npx jest --watchAll=false"
        )

    return {
        "ecosystem": "node",
        "package_manager": pm,
        "framework": framework or ("npm-script" if (test_script and not is_placeholder) else "none"),
        "primary_command": primary_cmd,
        "available": available,
    }


def detect_project_test_runner(cwd: Path) -> dict[str, Any]:
    """Detect repository language ecosystems and the best read-only test command for `cwd`."""
    root = Path(cwd).resolve()
    exec_env = _build_workdir_exec_env(root)
    path_str = exec_env.get("PATH")
    ecosystems: list[str] = []
    runners: list[dict[str, Any]] = []

    py_markers = ("pyproject.toml", "pytest.ini", "setup.py", "setup.cfg", "manage.py", "requirements.txt")
    has_py = (
        any((root / m).is_file() for m in py_markers)
        or _has_python_bin_test_runner(root)
        or next(root.glob("tests/test_*.py"), None) is not None
        or next(root.glob("*/tests/test_*.py"), None) is not None
    )
    if has_py:
        ecosystems.append("python")
        if (root / "manage.py").is_file() and not _repo_has_explicit_pytest_config(root):
            py_bin = ".venv/bin/python" if (root / ".venv/bin/python").is_file() else "python3"
            runners.append(
                {
                    "ecosystem": "python",
                    "framework": "django",
                    "primary_command": f"{py_bin} manage.py test",
                    "available": bool(shutil.which(py_bin, path=path_str) or (root / py_bin).is_file()),
                }
            )
        else:
            py_runner = _detect_default_pytest_runner(root)
            if py_runner:
                runners.append(
                    {
                        "ecosystem": "python",
                        "framework": "bin/test" if py_runner.endswith("bin/test") else "pytest",
                        "primary_command": py_runner if py_runner.endswith("bin/test") else f"{py_runner} -q",
                        "available": True,
                    }
                )
            else:
                runners.append(
                    {
                        "ecosystem": "python",
                        "framework": "unittest",
                        "primary_command": "python3 -m unittest discover",
                        "available": bool(shutil.which("python3", path=path_str)),
                    }
                )

    node_info = _detect_node_runner_info(root, exec_env)
    if node_info is not None:
        ecosystems.append("node")
        if node_info.get("primary_command"):
            runners.append(node_info)

    if (root / "go.mod").is_file():
        ecosystems.append("go")
        runners.append(
            {
                "ecosystem": "go",
                "framework": "go-test",
                "primary_command": "go test ./...",
                "available": bool(shutil.which("go", path=path_str)),
            }
        )

    if (root / "Cargo.toml").is_file():
        ecosystems.append("rust")
        runners.append(
            {
                "ecosystem": "rust",
                "framework": "cargo-test",
                "primary_command": "cargo test --quiet",
                "available": bool(shutil.which("cargo", path=path_str)),
            }
        )

    if (root / "gradlew").is_file() or (root / "build.gradle").is_file() or (root / "build.gradle.kts").is_file():
        ecosystems.append("jvm")
        use_wrapper = (root / "gradlew").is_file() and os.access(root / "gradlew", os.X_OK)
        cmd_jvm = "./gradlew test" if use_wrapper else "gradle test"
        avail_jvm = (use_wrapper and bool(shutil.which("java", path=path_str))) or bool(
            shutil.which("gradle", path=path_str)
        )
        runners.append(
            {
                "ecosystem": "jvm",
                "framework": "gradle",
                "primary_command": cmd_jvm,
                "available": avail_jvm,
            }
        )
    elif (root / "pom.xml").is_file():
        ecosystems.append("jvm")
        use_mvnw = (root / "mvnw").is_file() and os.access(root / "mvnw", os.X_OK)
        cmd_mvn = "./mvnw -q test" if use_mvnw else "mvn -q test"
        avail_mvn = (use_mvnw and bool(shutil.which("java", path=path_str))) or bool(
            shutil.which("mvn", path=path_str)
        )
        runners.append(
            {
                "ecosystem": "jvm",
                "framework": "maven",
                "primary_command": cmd_mvn,
                "available": avail_mvn,
            }
        )

    makefile_p = root / "Makefile"
    if makefile_p.is_file():
        try:
            mk_txt = makefile_p.read_text(encoding="utf-8", errors="replace")
            if re.search(r"(?m)^test\s*:", mk_txt):
                if "make" not in ecosystems:
                    ecosystems.append("make")
                runners.append(
                    {
                        "ecosystem": "make",
                        "framework": "make",
                        "primary_command": "make test",
                        "available": bool(shutil.which("make", path=path_str)),
                    }
                )
        except OSError:
            pass

    primary = next((r for r in runners if r.get("available")), runners[0] if runners else None)
    return {
        "ecosystems": ecosystems,
        "primary_command": str(primary.get("primary_command") or "") if primary else "",
        "framework": str(primary.get("framework") or "") if primary else "",
        "available": bool(primary.get("available")) if primary else False,
        "runners": runners,
    }


def _list_tracked_repo_files(cwd: Path) -> list[str]:
    try:
        tracked_res = _git_raw(
            ["ls-files"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if tracked_res.returncode == 0 and tracked_res.stdout.strip():
            return [line.strip() for line in tracked_res.stdout.splitlines() if line.strip()]
    except OSError:
        pass
    skip_dirs = {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "target",
        "dist",
        "build",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }
    fallback: list[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(cwd):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".agy")]
            rel_dir = Path(dirpath).relative_to(cwd)
            for fname in sorted(filenames):
                rel_p = (rel_dir / fname).as_posix() if rel_dir != Path(".") else fname
                fallback.append(rel_p)
                if len(fallback) >= 5000:
                    return fallback
    except OSError:
        return []
    return fallback


def discover_related_test_commands(
    cwd: Path,
    files_changed: list[str],
    existing_cmds: list[str],
    *,
    max_files: int = 6,
) -> list[str]:
    """Discover targeted unit/regression test commands across Python, Node/TS, Go, Rust, and JVM
    for modified source or test files that are not already covered by `existing_cmds`."""
    if not files_changed:
        return []
    exec_env = _build_workdir_exec_env(cwd)
    path_str = exec_env.get("PATH")
    tracked_cache: list[str] | None = None

    def _get_tracked() -> list[str]:
        nonlocal tracked_cache
        if tracked_cache is None:
            tracked_cache = _list_tracked_repo_files(cwd)
        return tracked_cache

    out_cmds: list[str] = []

    # --- 1. Python (.py) ---
    py_changed = [
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if rel.replace("\\", "/").strip("/").endswith(".py")
    ]
    if py_changed:
        pytest_cmds: list[str] = []
        skip_py = False
        if existing_cmds:
            pytest_cmds = [c for c in existing_cmds if re.search(r"\bpytest\b", c)]
            if not pytest_cmds:
                skip_py = True
            else:
                for pcmd in pytest_cmds:
                    if not re.search(r"[\w./-]+\.py\b", pcmd) and not re.search(r"(?:^|\s)-k(?:\s|=|$)", pcmd):
                        skip_py = True
                        break
        if not skip_py:
            discovered_py: list[str] = []
            stems_to_check: list[str] = []
            pkg_test_names: set[str] = set()
            mirrored_test_dirs: list[str] = []
            for norm in py_changed:
                bname = Path(norm).name
                if bname.startswith("test_") or bname.endswith("_test.py"):
                    if not existing_cmds and (cwd / norm).is_file() and norm not in discovered_py:
                        discovered_py.append(norm)
                    continue
                if bname in ("__init__.py", "conftest.py", "setup.py"):
                    continue
                parts = Path(norm).parts
                if any(p in ("tests", "test", "docs", "examples", "benchmarks") for p in parts[:-1]):
                    continue
                stem = Path(norm).stem
                if stem and stem not in stems_to_check:
                    stems_to_check.append(stem)
                dir_parts = list(parts[:-1])
                if dir_parts and dir_parts[0] in ("src", "lib", "python"):
                    dir_parts = dir_parts[1:]
                for pkg_seg in dir_parts:
                    if pkg_seg and pkg_seg.isidentifier():
                        pkg_test_names.add(f"test_{pkg_seg}.py")
                        pkg_test_names.add(f"{pkg_seg}_test.py")
                if dir_parts:
                    rel_pkg = "/".join(dir_parts)
                    for t_root in ("tests", "test"):
                        for cand_mir in (f"{t_root}/{rel_pkg}", *( [f"{t_root}/{'/'.join(dir_parts[1:])}"] if len(dir_parts) > 1 else [] )):
                            if cand_mir not in mirrored_test_dirs:
                                mirrored_test_dirs.append(cand_mir)

            def _is_py_test_covered(tf_rel: str) -> bool:
                tf_bname = Path(tf_rel).name
                return any(
                    re.search(rf"(?:^|[\s/\"'=]){re.escape(tf_bname)}(?:$|[\s\"':])", c)
                    or re.search(rf"(?:^|[\s\"'=]){re.escape(tf_rel)}(?:$|[\s\"':])", c)
                    for c in pytest_cmds
                )

            if stems_to_check and len(discovered_py) < max_files:
                for stem in stems_to_check:
                    target_names = {f"test_{stem}.py", f"{stem}_test.py"}
                    for tf in _get_tracked():
                        tf_name = Path(tf).name
                        if tf_name in target_names and tf not in discovered_py:
                            if not _is_py_test_covered(tf) and (cwd / tf).is_file():
                                discovered_py.append(tf)
                                if len(discovered_py) >= max_files:
                                    break
                    if len(discovered_py) >= max_files:
                        break

            # Include sibling package-level test files (e.g. tests/<pkg>/<subpkg>/test_<subpkg>.py)
            # and compact mirrored test directories (<= 8 test files) so package round-trip suites run.
            if (pkg_test_names or mirrored_test_dirs) and len(discovered_py) < max_files:
                tracked_list = _get_tracked()
                for tf in tracked_list:
                    if len(discovered_py) >= max_files:
                        break
                    tf_parent = Path(tf).parent.as_posix()
                    tf_name = Path(tf).name
                    if tf_name in pkg_test_names and (
                        tf_parent in mirrored_test_dirs
                        or tf.startswith(("tests/", "test/"))
                    ):
                        if tf not in discovered_py and not _is_py_test_covered(tf) and (cwd / tf).is_file():
                            discovered_py.append(tf)
                for mdir in mirrored_test_dirs:
                    if len(discovered_py) >= max_files:
                        break
                    dir_tests = [
                        tf
                        for tf in tracked_list
                        if Path(tf).parent.as_posix() == mdir
                        and (Path(tf).name.startswith("test_") or Path(tf).name.endswith("_test.py"))
                        and (cwd / tf).is_file()
                    ]
                    if 0 < len(dir_tests) <= 8:
                        for tf in dir_tests:
                            if len(discovered_py) >= max_files:
                                break
                            if tf not in discovered_py and not _is_py_test_covered(tf):
                                discovered_py.append(tf)

            if discovered_py:
                discovered_py = discovered_py[:max_files]
                if pytest_cmds:
                    first_pytest = pytest_cmds[0].strip()
                    m_pref = re.match(r"^(.*?\bpytest)\b", first_pytest)
                    runner_prefix = m_pref.group(1).strip() if m_pref else "pytest"
                else:
                    runner_prefix = _detect_default_pytest_runner(cwd)
                quoted_files = " ".join(shlex.quote(f) for f in discovered_py)
                if runner_prefix:
                    if runner_prefix.endswith("bin/test"):
                        out_cmds.append(f"{runner_prefix} {quoted_files}")
                    else:
                        out_cmds.append(f"{runner_prefix} -q {quoted_files}")
                elif not existing_cmds and shutil.which("python3", path=path_str):
                    out_cmds.append(f"python3 -m unittest {quoted_files}")

    # --- 2. Node.js / TypeScript (.ts, .tsx, .js, .jsx, .mjs, .cjs) ---
    js_changed = [
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if rel.replace("\\", "/").strip("/").lower().endswith(_JS_TS_EXTS)
    ]
    if js_changed and (cwd / "package.json").is_file():
        node_info = _detect_node_runner_info(cwd, exec_env)
        if node_info and node_info.get("available"):
            fw = str(node_info.get("framework") or "")
            pm = str(node_info.get("package_manager") or "npm")
            has_existing_node_cmd = any(
                re.search(r"\b(?:npm|pnpm|yarn|bun|npx|vitest|jest|mocha|node\s+--test)\b", c)
                for c in existing_cmds
            )
            # Only synthesize a Node test command when caller passed none, or when caller ran targeted
            # vitest/jest test files and missed a sibling `<stem>.test.*` / `<stem>.spec.*`.
            specific_js_cmds = [
                c
                for c in existing_cmds
                if re.search(r"\b(?:vitest|jest)\b", c) and _JS_TS_TEST_RE.search(c)
            ]
            if not existing_cmds or (has_existing_node_cmd and specific_js_cmds and len(specific_js_cmds) == len(existing_cmds)):
                discovered_js: list[str] = []
                js_stems: list[str] = []
                for norm in js_changed:
                    if _JS_TS_TEST_RE.search(norm):
                        if not existing_cmds and (cwd / norm).is_file() and norm not in discovered_js:
                            discovered_js.append(norm)
                        continue
                    bname = Path(norm).name
                    if bname.startswith(("index.", "vite.config.", "vitest.config.", "jest.config.", "eslint.config.")):
                        continue
                    stem = Path(norm).stem
                    if stem and stem not in js_stems:
                        js_stems.append(stem)
                if js_stems and len(discovered_js) < max_files:
                    for stem in js_stems:
                        cand_names = {
                            f"{stem}.test{ext}" for ext in _JS_TS_EXTS
                        } | {
                            f"{stem}.spec{ext}" for ext in _JS_TS_EXTS
                        }
                        for tf in _get_tracked():
                            tf_name = Path(tf).name
                            if tf_name in cand_names and tf not in discovered_js:
                                already_cov = any(
                                    re.search(rf"(?:^|[\s/\"'=]){re.escape(tf_name)}(?:$|[\s\"':])", c)
                                    for c in existing_cmds
                                )
                                if not already_cov and (cwd / tf).is_file():
                                    discovered_js.append(tf)
                                    if len(discovered_js) >= max_files:
                                        break
                        if len(discovered_js) >= max_files:
                            break
                if discovered_js:
                    q_js = " ".join(shlex.quote(f) for f in discovered_js[:max_files])
                    if fw == "vitest":
                        bin_v = "./node_modules/.bin/vitest" if (cwd / "node_modules/.bin/vitest").is_file() else "npx vitest"
                        out_cmds.append(f"{bin_v} run {q_js}")
                    elif fw == "jest":
                        bin_j = "./node_modules/.bin/jest" if (cwd / "node_modules/.bin/jest").is_file() else "npx jest"
                        out_cmds.append(f"{bin_j} --watchAll=false {q_js}")
                    elif fw == "bun-test":
                        out_cmds.append(f"bun test {q_js}")
                    elif fw == "node-test":
                        out_cmds.append(f"node --test {q_js}")
                    elif not existing_cmds and node_info.get("primary_command"):
                        out_cmds.append(str(node_info["primary_command"]))
                elif not existing_cmds and node_info.get("primary_command") and pm in ("npm", "pnpm", "yarn", "bun"):
                    out_cmds.append(str(node_info["primary_command"]))

    # --- 3. Go (.go) ---
    go_changed = [
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if rel.replace("\\", "/").strip("/").endswith(".go")
    ]
    if go_changed and (cwd / "go.mod").is_file() and shutil.which("go", path=path_str):
        go_cmds = [c for c in existing_cmds if re.search(r"\bgo\s+test\b", c)]
        full_go_suite = any("./..." in c for c in go_cmds)
        if (not existing_cmds) or (go_cmds and not full_go_suite):
            pkg_dirs: list[str] = []
            for norm in go_changed:
                parent_rel = Path(norm).parent.as_posix()
                pkg_dir = cwd if parent_rel in ("", ".") else (cwd / parent_rel)
                if not pkg_dir.is_dir() or next(pkg_dir.glob("*_test.go"), None) is None:
                    continue
                pkg_spec = "." if parent_rel in ("", ".") else f"./{parent_rel}"
                already_cov = any(
                    re.search(rf"(?:^|\s){re.escape(pkg_spec)}(?:$|/|\s)", c) for c in go_cmds
                )
                if not already_cov and pkg_spec not in pkg_dirs:
                    pkg_dirs.append(pkg_spec)
                    if len(pkg_dirs) >= max_files:
                        break
            if pkg_dirs:
                out_cmds.append("go test " + " ".join(shlex.quote(p) for p in pkg_dirs))

    # --- 4. Rust (.rs) ---
    rs_changed = [
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if rel.replace("\\", "/").strip("/").endswith(".rs")
    ]
    if not existing_cmds and rs_changed and (cwd / "Cargo.toml").is_file() and shutil.which("cargo", path=path_str):
        crate_names: list[str] = []
        for norm in rs_changed:
            cur_dir = (cwd / norm).parent
            while cur_dir != cwd and cur_dir.is_relative_to(cwd):
                sub_toml = cur_dir / "Cargo.toml"
                if sub_toml.is_file():
                    try:
                        toml_txt = sub_toml.read_text(encoding="utf-8", errors="replace")
                        m_pkg = re.search(r'(?ms)^\[package\].*?^\s*name\s*=\s*["\']([A-Za-z0-9_-]+)["\']', toml_txt)
                        if m_pkg and m_pkg.group(1) not in crate_names:
                            crate_names.append(m_pkg.group(1))
                    except OSError:
                        pass
                    break
                cur_dir = cur_dir.parent
        if crate_names:
            pkg_flags = " ".join(f"-p {shlex.quote(cn)}" for cn in crate_names[:max_files])
            out_cmds.append(f"cargo test --quiet {pkg_flags}")
        else:
            out_cmds.append("cargo test --quiet")

    # --- 5. Java / Kotlin (.java, .kt) ---
    jvm_changed = [
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if rel.replace("\\", "/").strip("/").lower().endswith((".java", ".kt"))
    ]
    if not existing_cmds and jvm_changed:
        jvm_test_classes: list[str] = []
        for norm in jvm_changed:
            stem = Path(norm).stem
            if not stem:
                continue
            if stem.endswith(("Test", "Tests", "IT")):
                if stem not in jvm_test_classes:
                    jvm_test_classes.append(stem)
                continue
            cand_jvm = {f"{stem}Test.java", f"{stem}Tests.java", f"{stem}Test.kt", f"{stem}Tests.kt"}
            for tf in _get_tracked():
                tf_bname = Path(tf).name
                if tf_bname in cand_jvm and (cwd / tf).is_file():
                    cls_name = Path(tf_bname).stem
                    if cls_name not in jvm_test_classes:
                        jvm_test_classes.append(cls_name)
                        break
        jvm_test_classes = jvm_test_classes[:max_files]
        is_multi_gradle = False
        for sg_name in ("settings.gradle", "settings.gradle.kts"):
            sg_p = cwd / sg_name
            if sg_p.is_file():
                try:
                    if re.search(r"\binclude\s*[\(\"'']", sg_p.read_text(encoding="utf-8", errors="replace")):
                        is_multi_gradle = True
                        break
                except OSError:
                    pass
        if (cwd / "gradlew").is_file() and os.access(cwd / "gradlew", os.X_OK) and shutil.which("java", path=path_str):
            if jvm_test_classes and not is_multi_gradle:
                flags = " ".join(f"--tests {shlex.quote('*' + c)}" for c in jvm_test_classes)
                out_cmds.append(f"./gradlew test {flags}")
            else:
                out_cmds.append("./gradlew test")
        elif ((cwd / "build.gradle").is_file() or (cwd / "build.gradle.kts").is_file()) and shutil.which("gradle", path=path_str):
            if jvm_test_classes and not is_multi_gradle:
                flags = " ".join(f"--tests {shlex.quote('*' + c)}" for c in jvm_test_classes)
                out_cmds.append(f"gradle test {flags}")
            else:
                out_cmds.append("gradle test")
        elif (cwd / "pom.xml").is_file():
            use_mvnw = (cwd / "mvnw").is_file() and os.access(cwd / "mvnw", os.X_OK) and bool(shutil.which("java", path=path_str))
            mvn_bin = "./mvnw" if use_mvnw else ("mvn" if shutil.which("mvn", path=path_str) else "")
            if mvn_bin:
                if jvm_test_classes:
                    out_cmds.append(
                        f"{mvn_bin} -q test -Dtest={','.join(jvm_test_classes)} "
                        f"-Dsurefire.failIfNoSpecifiedTests=false -DfailIfNoTests=false"
                    )
                else:
                    out_cmds.append(f"{mvn_bin} -q test")

    return out_cmds[:MAX_ACCEPTANCE_COMMANDS]


_IMPORT_SKIP_DIRS = frozenset({
    "tests", "test", "docs", "examples", "benchmarks", "scripts", "migrations", "tools", "bin",
})
_IMPORT_SKIP_FILES = frozenset({
    "setup.py", "conftest.py", "__main__.py", "manage.py", "noxfile.py", "tasks.py", "fabfile.py",
    "wsgi.py", "asgi.py", "sitecustomize.py", "usercustomize.py",
})


def build_import_smoke_command(
    cwd: Path,
    files_changed: list[str],
    *,
    max_modules: int = 8,
) -> list[str]:
    """Build a fast Python import smoke-check command for modified or created Python modules."""
    if not files_changed:
        return []
    modules: list[str] = []
    for rel in files_changed:
        norm = rel.replace("\\", "/").strip("/")
        if not norm.endswith(".py") or not (cwd / norm).is_file():
            continue
        bname = Path(norm).name
        if bname in _IMPORT_SKIP_FILES or bname.startswith("test_") or bname.endswith("_test.py"):
            continue
        parts = list(Path(norm).parts)
        if any(p in _IMPORT_SKIP_DIRS or p.startswith(".") for p in parts[:-1]):
            continue
        base_dir = cwd
        if parts[0] == "src" and len(parts) > 1:
            base_dir = cwd / "src"
            parts = parts[1:]
        if len(parts) <= 1:
            # Never import top-level non-package .py scripts (e.g. run.py), which can execute CLI/server side effects.
            continue
        # Verify every parent directory is a valid Python package with __init__.py
        cur = base_dir
        valid_pkg = True
        for pkg_part in parts[:-1]:
            cur = cur / pkg_part
            if not (cur / "__init__.py").is_file() or not pkg_part.isidentifier():
                valid_pkg = False
                break
        if not valid_pkg:
            continue
        if bname == "__init__.py":
            mod_name = ".".join(parts[:-1])
        else:
            stem = Path(bname).stem
            if not stem.isidentifier():
                continue
            mod_name = ".".join((*parts[:-1], stem))
        if mod_name and mod_name not in modules:
            modules.append(mod_name)
            if len(modules) >= max_modules:
                break
    if not modules:
        return []
    py_bin = "python3"
    for cand in (".venv/bin/python", "venv/bin/python"):
        if (cwd / cand).is_file():
            py_bin = cand
            break
    return [
        f"{py_bin} -c 'import {', '.join(modules)}'"
    ]


# Output signatures that mean a compile check could not run in this environment (missing toolchain,
# offline dependency fetch, no lockfile entry), as opposed to the worker's code failing to compile.
COMPILE_SMOKE_ENV_FAILURE_RE = re.compile(
    r"command not found|not recognized as an internal or external command|"
    r"cannot find module 'typescript'|could not resolve host|failed to download|"
    r"no matching package named|unable to update registry|dial tcp|network is unreachable|"
    r"missing go\.sum entry|cannot find module providing package|"
    r"attempting to make an http request, but --offline|failed to get .* as a dependency",
    re.I,
)

# Exits 0 unless an error line from the compiler names one of the files passed as arguments, so errors that
# already existed elsewhere in the project never fail a smoke check for an unrelated change.
_CHANGED_FILE_ERROR_FILTER = (
    "import re,sys\n"
    "rc=int(sys.argv[1]); files=sys.argv[2:]; out=sys.stdin.read()\n"
    "pats=[re.compile(r'^\\s*(?:-->\\s*)?(?:\\./)?'+re.escape(f)+r'(?::\\d+|\\(\\d+,\\d+\\))') for f in files]\n"
    "hits=[l for l in out.splitlines() if any(p.search(l) for p in pats)]\n"
    "print('\\n'.join(hits[-40:]) if hits else out[-2000:])\n"
    "sys.exit(1 if rc and hits else 0)\n"
)


def build_compile_smoke_commands(
    cwd: Path,
    files_changed: list[str],
    *,
    max_files: int = 8,
) -> list[str]:
    """Build fast compile/syntax checks for modified files across common languages.

    These run when a job has no explicit acceptance commands (for example a repository without tests), so
    a change that does not parse or compile still fails acceptance and triggers escalation instead of
    passing silently. Every command is read-only, uses toolchains that are already installed in the
    workspace or on PATH, and never downloads packages. Results are recorded as smoke checks, never as an
    explicit test pass.
    """
    if not files_changed:
        return []
    exec_env = _build_workdir_exec_env(cwd)
    path_str = exec_env.get("PATH")
    norm_files: list[str] = []
    for rel in files_changed:
        norm = rel.replace("\\", "/").strip("/")
        if norm and norm not in norm_files and (cwd / norm).is_file():
            norm_files.append(norm)

    def _pick(exts: tuple[str, ...]) -> list[str]:
        return [f for f in norm_files if f.lower().endswith(exts)][:max_files]

    def _quoted(files: list[str]) -> str:
        return " ".join(shlex.quote(f) for f in files)

    py_bin = "python3"
    for cand in (".venv/bin/python", "venv/bin/python"):
        if (cwd / cand).is_file():
            py_bin = cand
            break

    cmds: list[str] = []

    # Python: parse every modified file, including top-level scripts the import smoke check skips.
    py_files = _pick((".py",))
    if py_files:
        parse_script = "import ast,sys\nfor p in sys.argv[1:]:\n    ast.parse(open(p,'rb').read(), p)\n"
        cmds.append(f"{py_bin} -c {shlex.quote(parse_script)} {_quoted(py_files)}")

    # TypeScript: type-check the project with the repository's own tsc, failing only on changed files.
    ts_files = _pick((".ts", ".tsx", ".mts", ".cts"))
    tsc_bin = cwd / "node_modules" / ".bin" / "tsc"
    if ts_files and (cwd / "tsconfig.json").is_file() and tsc_bin.is_file():
        cmds.append(
            "out=$(./node_modules/.bin/tsc --noEmit --pretty false -p tsconfig.json 2>&1); rc=$?; "
            f"printf '%s\\n' \"$out\" | python3 -c {shlex.quote(_CHANGED_FILE_ERROR_FILTER)} \"$rc\" "
            f"{_quoted(ts_files)}"
        )

    # Plain JavaScript: syntax check with node (JSX needs a build step, so it is skipped).
    js_files = _pick((".js", ".mjs", ".cjs"))
    if js_files and shutil.which("node", path=path_str):
        cmds.append(" && ".join(f"node --check {shlex.quote('./' + f)}" for f in js_files))

    # Go: compile the changed packages without writing a binary (test-only edits are left to `go test`).
    go_files = [f for f in _pick((".go",)) if not f.endswith("_test.go")]
    if go_files and (cwd / "go.mod").is_file() and shutil.which("go", path=path_str):
        pkgs: list[str] = []
        for f in go_files:
            parent = Path(f).parent.as_posix()
            spec = "." if parent in ("", ".") else f"./{parent}"
            if spec not in pkgs:
                pkgs.append(spec)
        cmds.append(f"go build -o /dev/null {' '.join(shlex.quote(p) for p in pkgs)}")

    # Rust: only with a warm host target directory, so the check never triggers a cold full build.
    rs_files = _pick((".rs",))
    if rs_files and (cwd / "Cargo.toml").is_file() and shutil.which("cargo", path=path_str):
        if (cwd / "target").is_dir():
            cmds.append("cargo check --offline --quiet")

    # Shell scripts and JSON documents (JSON-with-comments config files are skipped).
    sh_files = _pick((".sh", ".bash"))
    if sh_files and shutil.which("bash", path=path_str):
        cmds.append(" && ".join(f"bash -n {shlex.quote('./' + f)}" for f in sh_files))
    json_files = [
        f
        for f in _pick((".json",))
        if not re.match(r"^(?:tsconfig|jsconfig|\.eslintrc|devcontainer|\.babelrc)", Path(f).name, re.I)
        and not any(part in (".vscode", ".devcontainer") for part in Path(f).parts[:-1])
    ]
    if json_files:
        json_script = "import json,sys\nfor p in sys.argv[1:]:\n    json.load(open(p,encoding='utf-8-sig'))\n"
        cmds.append(f"python3 -c {shlex.quote(json_script)} {_quoted(json_files)}")

    return cmds


def find_missing_prompt_symbols(
    prompt: str,
    diff_text: str,
    *,
    worktree_root: Path | None = None,
    files_changed: list[str] | None = None,
    baseline_ref: str | None = None,
) -> list[str]:
    """Identify code symbols explicitly backticked in `prompt` (e.g. `` `my_func` ``, `` `MyClass.method` ``)
    that are completely absent from both `diff_text` and the modified files in `worktree_root`."""
    if not prompt or not prompt.strip():
        return []

    # Collect file stems mentioned in the prompt or files_changed so module names are not mistaken for missing symbols
    file_stems: set[str] = set()
    for f_match in re.findall(r"[\w./-]+\.(?:py|ts|tsx|js|jsx|rs|go|java|rb|c|cc|cpp|h|hpp)\b", prompt):
        file_stems.add(Path(f_match).stem)
    for fc in files_changed or []:
        file_stems.add(Path(fc).stem)

    candidate_symbols: list[str] = []
    # Method half of a backticked `ClassName.method` -> ClassName, used to scope the baseline check.
    member_owners: dict[str, str] = {}
    for raw_span in re.findall(r"(?<!`)`([^`\n]+)`(?!`)", prompt):
        span = raw_span.strip()
        if not span or span.startswith("-"):
            continue
        span = re.sub(r"^(?:async\s+def|def|class)\s+", "", span).rstrip(":").strip()
        has_call_parens = "(" in span and ")" in span
        if "(" in span:
            span = span.split("(", 1)[0].strip()
        if not span or len(span) < 3 or len(span) > 80:
            continue
        if " " in span or "/" in span or "\\" in span:
            continue
        if re.search(r"\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml|toml|cfg|ini|sh|txt|rst|html|css|sql)\b", span, re.IGNORECASE):
            continue
        # Handle dotted identifiers like `ClassName.method_name` or `module.func_name`
        parts = span.split(".")
        if len(parts) > 2 or not all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", p) for p in parts):
            continue
        tokens_to_check = [parts[-1]] if len(parts) == 2 else [parts[0]]
        if len(parts) == 2 and re.match(r"^[A-Z][A-Za-z0-9_]*$", parts[0]) and any(c.islower() for c in parts[0]):
            tokens_to_check.insert(0, parts[0])
            member_owners.setdefault(parts[-1], parts[0])
        for tok in tokens_to_check:
            if len(tok) < 4 or tok in _PROMPT_SYMBOL_STOPLIST or tok in file_stems:
                continue
            # Ignore all-uppercase constants/env-vars like `TRUE`, `JSON`, `HTTP`, `CC_AGY_...`
            if tok.isupper():
                continue
            is_snake = "_" in tok and any(c.islower() for c in tok)
            is_camel = bool(re.match(r"^[A-Z][A-Za-z0-9]+$", tok)) and any(c.islower() for c in tok)
            if (is_snake or is_camel or has_call_parens) and tok not in candidate_symbols:
                candidate_symbols.append(tok)

    if not candidate_symbols:
        return []

    corpus_parts: list[str] = [diff_text or ""]
    if worktree_root is not None and worktree_root.is_dir():
        for rel in files_changed or []:
            fp = worktree_root / rel
            if fp.is_file() and not fp.is_symlink():
                try:
                    corpus_parts.append(fp.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    pass
    corpus = "\n".join(corpus_parts)
    if not corpus.strip():
        return candidate_symbols[:10]

    missing: list[str] = [sym for sym in candidate_symbols if not re.search(rf"\b{re.escape(sym)}\b", corpus)]
    if missing and worktree_root is not None:
        # Symbols already *defined* in non-test baseline code are references to existing code
        # ("follow `_DiverseFailableLawSpec`", "do NOT extend `SwappableN`"), not new symbols the
        # worker was asked to create, so they must not block auto-apply or trigger extra turns.
        # Mere mentions (test imports, docs) do not count: a TDD test that imports a not-yet-written
        # function must still surface that function as missing.
        defined = _symbols_defined_in_baseline(
            worktree_root, missing, ref=baseline_ref or "HEAD", owners=member_owners
        )
        missing = [sym for sym in missing if sym not in defined]
    # Cap only after filtering so real misses are never crowded out by excluded references.
    return missing[:10]


_DOC_SUFFIXES = (".md", ".rst", ".txt", ".adoc")


def _is_test_or_doc_path(rel: str) -> bool:
    p = Path(rel)
    name = p.name
    if any(part in ("tests", "test", "testing", "__tests__", "docs", "doc") for part in p.parts[:-1]):
        return True
    if name.startswith("test_") or name.endswith(("_test.py", "_test.go", "_spec.rb")):
        return True
    if ".test." in name or ".spec." in name or name == "conftest.py":
        return True
    return name.lower().endswith(_DOC_SUFFIXES)


def _symbols_defined_in_baseline(
    repo_root: Path,
    symbols: list[str],
    *,
    ref: str = "HEAD",
    owners: dict[str, str] | None = None,
) -> set[str]:
    """Return the subset of `symbols` that are defined in non-test, non-doc files of the committed `ref`
    tree of `repo_root`.

    A definition is a declaration at the start of a non-comment line (def/class/function/type/...;
    indentation allowed so methods count) or a column-0 top-level assignment. A symbol listed in `owners`
    (the method half of a backticked `Owner.method`) counts only if one file defines both `class Owner`
    and `method`, so `Order.to_payload` is not excused by an unrelated `User.to_payload`.
    """
    if not symbols or not repo_root.is_dir():
        return set()
    owner_map = {sym: own for sym, own in (owners or {}).items() if sym in symbols}
    search_terms = list(dict.fromkeys(list(symbols) + list(owner_map.values())))
    cmd = ["grep", "--no-color", "-I", "-z", "-w", "-F"]
    for term in search_terms:
        cmd.extend(["-e", term])
    cmd.extend([ref, "--"])
    try:
        res = _git_raw(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    # git grep exits 1 when nothing matches and >1 on errors (e.g. not a git repo); both mean no exclusions.
    if res.returncode != 0:
        return set()
    decl_prefix = r"^\s*(?:(?:export|default|async|pub|public|private|protected|static|abstract|final)\s+)*"
    def_patterns = {
        term: (
            re.compile(
                rf"{decl_prefix}(?:def|class|function|interface|type|struct|enum|trait|fn|func)\s+{re.escape(term)}\b"
            ),
            re.compile(rf"^(?:(?:export|const|let|var|final|static|pub)\s+)*{re.escape(term)}\s*(?::[^=]*)?=(?!=)"),
        )
        for term in search_terms
    }
    comment_prefixes = ("#", "//", "/*", "*", "--", '"""', "'''")
    defined_by_file: dict[str, set[str]] = {}
    prefix = f"{ref}:"
    for line in (res.stdout or "").splitlines():
        # With -z and a tree-ish, each hit is `<ref>:<path>\0<content>`.
        if "\x00" not in line:
            continue
        loc, content = line.split("\x00", 1)
        rel = loc[len(prefix):] if loc.startswith(prefix) else loc
        if _is_test_or_doc_path(rel) or content.lstrip().startswith(comment_prefixes):
            continue
        for term, (decl_re, assign_re) in def_patterns.items():
            if decl_re.search(content) or assign_re.search(content):
                defined_by_file.setdefault(rel, set()).add(term)
    found: set[str] = set()
    for sym in symbols:
        owner = owner_map.get(sym)
        for defs in defined_by_file.values():
            if sym in defs and (owner is None or owner in defs):
                found.add(sym)
                break
    return found


def _hash_path(p: Path) -> str | None:
    """Return a content fingerprint for a workspace path (symlink target, file SHA-256, or None if absent)."""
    if p.is_symlink():
        return f"symlink:{os.readlink(p)}"
    if p.exists() and p.is_file():
        return sha256_bytes(p.read_bytes())
    return None


def _snapshot_untracked_hashes(root: Path) -> dict[str, str | None]:
    """Fingerprint every untracked, non-ignored file so in_place jobs can skip ones they did not touch."""
    res = _git_raw(
        ["ls-files", "--others", "--exclude-standard", "-z"],
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
        """Return the escalated worker model (e.g. 'gemini-4-argon') when escalating from Flash, or None."""
        if "flash" not in (current_model or "").lower():
            return None
        raw_esc = os.environ.get("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "").strip()
        if raw_esc.lower() in ("0", "false", "no", "off"):
            return None
        if raw_esc:
            target = (
                "gemini-4-argon"
                if raw_esc.lower() in ("1", "true", "yes", "on")
                else raw_esc
            )
        elif resolve_cost_profile() == "economy":
            return None
        elif self._is_bounded_lead_session() or self.worker_backend is None:
            target = "gemini-4-argon"
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
                except Exception:
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
            except Exception:
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
            except Exception:
                fast_seeded = False
        if not fast_seeded:
            # When populated from `git ls-files`, force-add (`-f`) so tracked files that also match `.gitignore`
            # patterns (e.g. `tests/test_apps/.env` in Flask) are committed to `baseline_sha`.
            _run_git(["add", "-f", "-A"] if ls_res.returncode == 0 else ["add", "-A"], cwd=wt_dir, check=True)
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
        raw_untracked = [
            ln.strip()
            for ln in (others_res.stdout.splitlines() if others_res.returncode == 0 else [])
            if ln.strip()
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
                if not rel_i or rel_i in untracked_to_stage:
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
        try:
            with self._lock:
                if job_id in self._cancelled:
                    status = "cancelled"
                    error_msg = "Job cancelled before execution."
                    raise RuntimeError(error_msg)

            if not in_place:
                allow_shallow = not (
                    auth_mode == "gemini_enterprise"
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
                except Exception:
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
            except Exception:
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
            except Exception:
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
            except Exception:
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
        checklist: list[str] | None = None,
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
        enforce_daily_budget(self.store, workspace_root=self.workspace_root)
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
                            except Exception:
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
                except Exception:
                    raw_user_prompt_text = ""
        lead_read_paths: list[str] = []
        lead_read_windows: dict[str, tuple[int, int]] = {}
        lead_rf_path = self.store.data_dir / ".lead_read_files.json"
        if lead_rf_path.is_file():
            try:
                parsed_lrf = json.loads(lead_rf_path.read_text(encoding="utf-8"))
                if isinstance(parsed_lrf, list):
                    lead_read_paths = [str(x) for x in parsed_lrf if isinstance(x, str)]
            except Exception:
                lead_read_paths = []
        lead_rw_path = self.store.data_dir / ".lead_read_windows.json"
        if lead_rw_path.is_file():
            try:
                parsed_lrw = json.loads(lead_rw_path.read_text(encoding="utf-8"))
                if isinstance(parsed_lrw, dict):
                    for k, v in parsed_lrw.items():
                        if isinstance(k, str) and isinstance(v, list) and len(v) == 2:
                            lead_read_windows[k] = (int(v[0]), int(v[1]))
            except Exception:
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
            **({"baseline_acceptance_probe": baseline_probe} if baseline_probe else {}),
            **({"acceptance_remapped": acceptance_remapped} if acceptance_remapped else {}),
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

        if wait:
            setup_elapsed = max(0.0, time.monotonic() - rpc_started_mono)
            wait_slice = self._resolve_wait_slice_seconds(
                timeout_seconds,
                elapsed_seconds=setup_elapsed,
            )
            self._wait_for_future_or_cancel(job_id, fut, timeout=wait_slice)
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
        deadline = time.monotonic() + max(0.01, float(timeout))
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

    def record_qa_review_on_receipt(
        self,
        job_id: str,
        *,
        qa_model: str,
        qa_meta: dict[str, Any],
        qa_usage: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge a 1-shot QA review's usage and metadata onto `job-*.json` without creating a separate receipt."""
        safe_id = validate_job_id(job_id)
        receipt = self.store.read_receipt(safe_id)
        prior_model = str(receipt.get("model") or "gemini-flash-next")
        u = dict(receipt.get("usage") or {})
        pq = int((qa_usage or {}).get("prompt_token_count") or 0)
        cq = int((qa_usage or {}).get("cached_content_token_count") or 0)
        oq = int((qa_usage or {}).get("candidates_token_count") or 0)
        tq = int((qa_usage or {}).get("thoughts_token_count") or 0)
        tot_q = max(int((qa_usage or {}).get("total_token_count") or 0), pq + oq + tq)
        if pq > 0 or oq > 0 or tq > 0 or tot_q > 0:
            if not u.get("requests") and any(
                int(u.get(k) or 0) > 0
                for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")
            ):
                u["requests"] = [
                    {
                        "prompt_token_count": int(u.get("prompt_token_count") or 0),
                        "cached_content_token_count": int(u.get("cached_content_token_count") or 0),
                        "candidates_token_count": int(u.get("candidates_token_count") or 0),
                        "thoughts_token_count": int(u.get("thoughts_token_count") or 0),
                        "total_token_count": int(u.get("total_token_count") or 0),
                        "model": prior_model,
                    }
                ]
            elif isinstance(u.get("requests"), list):
                tagged = [
                    ({**r, "model": r.get("model") or prior_model} if isinstance(r, dict) else r)
                    for r in u["requests"]
                ]
                sp = sum(int(r.get("prompt_token_count") or 0) for r in tagged if isinstance(r, dict))
                sc = sum(int(r.get("cached_content_token_count") or 0) for r in tagged if isinstance(r, dict))
                so = sum(int(r.get("candidates_token_count") or 0) for r in tagged if isinstance(r, dict))
                st = sum(int(r.get("thoughts_token_count") or 0) for r in tagged if isinstance(r, dict))
                rp = max(0, int(u.get("prompt_token_count") or 0) - sp)
                rc = min(rp, max(0, int(u.get("cached_content_token_count") or 0) - sc))
                ro = max(0, int(u.get("candidates_token_count") or 0) - so)
                rt = max(0, int(u.get("thoughts_token_count") or 0) - st)
                if rp > 0 or ro > 0 or rt > 0:
                    tagged.append(
                        {
                            "prompt_token_count": rp,
                            "cached_content_token_count": rc,
                            "candidates_token_count": ro,
                            "thoughts_token_count": rt,
                            "total_token_count": rp + ro + rt,
                            "model": prior_model,
                        }
                    )
                u["requests"] = tagged
            else:
                u["requests"] = []
            u["requests"].append(
                {
                    "per_call": True,
                    "single_request": True,
                    "kind": "qa_review",
                    "model": qa_model,
                    "prompt_token_count": pq,
                    "cached_content_token_count": cq,
                    "candidates_token_count": oq,
                    "thoughts_token_count": tq,
                    "total_token_count": tot_q,
                }
            )
            u["prompt_token_count"] = int(u.get("prompt_token_count") or 0) + pq
            u["cached_content_token_count"] = int(u.get("cached_content_token_count") or 0) + cq
            u["candidates_token_count"] = int(u.get("candidates_token_count") or 0) + oq
            u["thoughts_token_count"] = int(u.get("thoughts_token_count") or 0) + tq
            u["total_token_count"] = int(u.get("total_token_count") or 0) + tot_q
            receipt["usage"] = u
            precomputed = receipt.get("precomputed_cost")
            if isinstance(precomputed, dict) and "total_cost_cents" in precomputed:
                # Receipts that price their own calls (batch jobs) ignore usage when costed, so the QA
                # review's spend has to be added to the precomputed total or it never shows up.
                from server.regions import estimate_cost_usd

                qa_cost = estimate_cost_usd(
                    qa_model,
                    {"prompt_token_count": pq, "cached_content_token_count": cq, "candidates_token_count": oq,
                     "thoughts_token_count": tq, "total_token_count": tot_q},
                    single_request=True,
                )
                qa_cents = float(qa_cost.get("total_cost_cents") or 0.0)
                total_cents = float(precomputed.get("total_cost_cents") or 0.0) + qa_cents
                receipt["precomputed_cost"] = {
                    **precomputed,
                    "total_cost_cents": round(total_cents, 4),
                    "total_cost_usd": round(total_cents / 100.0, 6),
                    "pricing_known": bool(precomputed.get("pricing_known", True))
                    and bool(qa_cost.get("pricing_known", True)),
                }
        receipt["qa_review"] = dict(qa_meta)
        with self._lock:
            raw_key = self._raw_api_keys.get(safe_id)
        return self.store.write_receipt(
            safe_id,
            receipt,
            scrub=bool(receipt.get("scrub_receipt", False)),
            workspace_root=str(self.workspace_root),
            api_key=raw_key,
        )

    def _maybe_run_balanced_pre_apply_qa(
        self,
        job_id: str,
        receipt: dict[str, Any],
        *,
        auto_apply: bool | None,
        in_place: bool,
        is_cancelled_now: bool,
        wait_elapsed: float = 0.0,
        timeout_seconds: float | None = None,
        wait_budget: float | None = None,
    ) -> dict[str, Any] | None:
        """Run a 1-shot pre-apply QA diff audit on un-escalated Flash passes in `balanced`."""
        qa_flag = env_flag("CC_AGY_BALANCED_QA_REVIEW")
        if qa_flag is False:
            return None
        autonomous_solo = env_flag("CC_AGY_AUTONOMOUS_SOLO") is True
        if resolve_cost_profile() != "balanced" and not autonomous_solo:
            return None
        if self.worker_backend is not None and qa_flag is not True:
            return None
        # One extra audit round on the escalated model's final diff: escalated turns otherwise ship
        # without any review even when the stronger model leaves a requirement half-done.
        prior_qa = receipt.get("qa_review") if isinstance(receipt.get("qa_review"), dict) else None
        post_escalation_round = bool(
            receipt.get("escalated_from_model")
            and not (prior_qa or {}).get("post_escalation")
            and env_flag("CC_AGY_POST_ESCALATION_QA") is True
        )
        if (
            in_place
            or is_cancelled_now
            or receipt.get("applied")
            or receipt.get("skip_balanced_qa")
            or (receipt.get("qa_review") and not post_escalation_round)
            or (receipt.get("escalated_from_model") and not post_escalation_round)
            or receipt.get("status") != "ok"
            or receipt.get("acceptance_passed") is not True
            or not receipt.get("files_changed")
        ):
            return None
        cur_model = str(receipt.get("model") or "")
        if "flash" not in cur_model.lower() and not post_escalation_round:
            return None
        qa_model = self._resolve_escalation_target_model(cur_model) or (
            cur_model if (post_escalation_round or autonomous_solo) else None
        )
        if not qa_model:
            return None
        qa_started_mono = time.monotonic()
        esc_deadline, _esc_elapsed, rem_budget = self._session_deadline_budget()
        if esc_deadline is not None and rem_budget < (180.0 if post_escalation_round else 90.0):
            return None
        diff_path = self.store.diff_path(job_id)
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""
        if not diff_text.strip():
            return None

        from server.genai_ask import ask_gemini
        from server.review_bundle import build_review_evidence_dir

        wt_candidate = Path(str(receipt.get("worktree_dir") or ""))
        if not wt_candidate.is_dir():
            wt_candidate = self.store.worktree_path(job_id)
        target_wt = wt_candidate if wt_candidate.is_dir() else self.workspace_root

        raw_user_spec = str(receipt.get("raw_user_prompt") or "").strip()
        lead_brief = str((receipt.get("prompts") or [""])[0] or "").strip()
        spec_sections: list[str] = []
        if raw_user_spec and raw_user_spec != lead_brief:
            spec_sections.append(f"### Original User Specification\n{raw_user_spec}")
            if lead_brief:
                spec_sections.append(f"### Lead Delegation Brief (Summary: Original User Specification Takes Precedence)\n{lead_brief}")
        elif lead_brief or raw_user_spec:
            spec_sections.append(lead_brief or raw_user_spec)
        spec_rel_files = [str(p) for p in (receipt.get("spec_files") or []) if p]
        for rel_sp in spec_rel_files[:4]:
            cand_sp = (target_wt / rel_sp).resolve()
            if cand_sp.is_relative_to(target_wt.resolve()) and cand_sp.is_file():
                sp_txt, _, _ = truncate_to_token_budget(
                    cand_sp.read_text(encoding="utf-8", errors="replace"),
                    1500,
                )
                spec_sections.append(f"### Spec File: {rel_sp}\n{sp_txt}")
        checklist_items = [str(c) for c in (receipt.get("checklist") or []) if c]
        if checklist_items:
            spec_sections.append("### Checklist\n" + "\n".join(f"- {c}" for c in checklist_items))
        warn_items = [
            str(w)
            for w in [*(receipt.get("stub_mutation_warnings") or []), *(receipt.get("contract_warnings") or [])]
            if w
        ]
        if receipt.get("recovered_from_checkpoint"):
            warn_items.append(
                "Worker hit MAX_TOOL_CALLS_EXCEEDED right as acceptance tests passed; verify that every "
                "requirement, edge case, and caller in the Original User Specification is completely implemented."
            )
        changed_rel = [str(p) for p in (receipt.get("files_changed") or []) if p]
        if changed_rel and not any(_is_test_file_path(p) for p in changed_rel):
            warn_items.append(
                "The diff changes production code but adds or updates no test file, so the passing suite may "
                "never exercise the requested behavior. Name the existing test that covers each requirement, "
                "or report the uncovered requirement as an ISSUE."
            )
        if autonomous_solo and not warn_items:
            return None
        if warn_items:
            spec_sections.append("### Automated Diff Warnings\n" + "\n".join(f"- {w}" for w in warn_items))
        full_spec_text = "\n\n".join(spec_sections).strip()

        acc_results = [r for r in (receipt.get("acceptance_results") or []) if isinstance(r, dict)]
        test_out_lines = [
            f"$ {r.get('command', '')} (exit {r.get('exit_code', 0)})\n{r.get('output_tail', '')}".strip()
            for r in acc_results
        ]
        test_out_str = "\n\n".join(test_out_lines)
        review_bundle = build_review_evidence_dir(
            target_wt,
            task_prompt=full_spec_text,
            diff_text=diff_text,
            test_output=test_out_str,
            files_changed=list(receipt.get("files_changed") or []),
        )
        qa_diff, qa_truncated, _qa_orig = truncate_to_token_budget(diff_text, QA_REVIEW_MAX_TOKENS)
        slices_text = str(review_bundle.get("enclosing_slices") or "")
        slices_tok = int(review_bundle.get("enclosing_slices_tokens") or 0)
        slices_block = (
            f"\n\n## Enclosing Modified Functions & Classes\n{slices_text}"
            if (slices_text and not qa_truncated and (_qa_orig + slices_tok) <= QA_REVIEW_MAX_TOKENS)
            else ""
        )
        acc_block, _, _ = truncate_to_token_budget(test_out_str, 1500) if test_out_str else ("", False, 0)
        acc_section = f"\n\n## Acceptance Command Output\n```\n{acc_block}\n```" if acc_block else ""
        qa_prompt = (
            f"## Task Specification\n{full_spec_text}\n\n"
            f"## Candidate Git Diff{' (truncated)' if qa_truncated else ''}\n```diff\n{qa_diff}\n```"
            f"{acc_section}{slices_block}"
        )
        qa_system = build_qa_diff_audit_system(
            verification=classify_verification(acc_results, receipt.get("acceptance_passed")),
            prompt=full_spec_text,
        )
        with self._lock:
            raw_proj = self._raw_projects.get(job_id)
            raw_key = self._raw_api_keys.get(job_id)
        eff_proj = raw_proj or (
            str(receipt.get("vertex_project"))
            if receipt.get("vertex_project") and receipt.get("vertex_project") != "[REDACTED]"
            else None
        )
        explicit_qa_env = (os.environ.get("CC_AGY_QA_THINKING_LEVEL") or "").strip().upper()
        raw_qa_thinking = (
            explicit_qa_env
            or (os.environ.get("CC_AGY_FORCE_WORKER_THINKING_LEVEL") or "").strip().upper()
        )
        if raw_qa_thinking in ("0", "OFF"):
            raw_qa_thinking = "ZERO"
        qa_model_low = str(qa_model).strip().lower()
        if qa_model_low.endswith("-zero") and not explicit_qa_env:
            qa_thinking = "ZERO"
        elif raw_qa_thinking in ("ZERO", "MINIMAL", "LOW", "MEDIUM", "HIGH"):
            qa_thinking = raw_qa_thinking
        elif qa_model_low.endswith("-zero"):
            qa_thinking = "ZERO"
        elif qa_model_low.endswith("-high"):
            qa_thinking = "HIGH"
        elif qa_model_low.endswith("-medium"):
            qa_thinking = "MEDIUM"
        else:
            qa_thinking = "LOW"
        qa_timeout_ms = max(5_000, _env_int("CC_AGY_QA_TIMEOUT_MS", 90_000))

        def _should_escalate_qa(
            p_qa: dict[str, Any],
            q_err: str | None,
            rem_sec: float,
            has_dl: bool,
        ) -> bool:
            if p_qa["passed"] is False:
                return (not has_dl) or rem_sec >= 120.0
            if (
                p_qa["passed"] is None
                and not is_quota_or_rate_limit_error(q_err or "")
                and (
                    bool(receipt.get("recovered_from_checkpoint"))
                    or max(int(receipt.get("tool_call_count") or 0), len(receipt.get("tool_calls") or [])) >= 15
                )
            ):
                return (not has_dl) or rem_sec >= 180.0
            return False

        def _invoke_qa_once() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
            err_s: str | None = None
            agg_u: dict[str, int] = {
                "prompt_token_count": 0,
                "cached_content_token_count": 0,
                "candidates_token_count": 0,
                "thoughts_token_count": 0,
                "total_token_count": 0,
            }
            res_q: dict[str, Any] = {}
            parsed_q: dict[str, Any] = {"verdict": "ERROR", "passed": None, "issues": []}
            for _qa_try in range(2):
                err_s = None
                try:
                    res_q = ask_gemini(
                        prompt=qa_prompt,
                        model=qa_model,
                        system_instruction=qa_system,
                        thinking_level=qa_thinking,
                        timeout_ms=qa_timeout_ms,
                        api_key=raw_key,
                        project=eff_proj,
                        location=receipt.get("vertex_location"),
                        workspace_root=self.workspace_root,
                        store=self.store,
                        scrub_receipt=bool(receipt.get("scrub_receipt", False)),
                        backend=self.ask_backend,
                        record_receipt=False,
                    )
                    cur_u = dict(res_q.get("usage") or {})
                    for _uk in agg_u:
                        agg_u[_uk] += int(cur_u.get(_uk) or 0)
                    parsed_q = parse_qa_audit_verdict(str(res_q.get("text") or ""))
                    break
                except Exception as qa_exc:
                    err_s = f"{type(qa_exc).__name__}: {qa_exc}"
                    res_q = {}
                    partial_u = getattr(qa_exc, "partial_usage", None)
                    if isinstance(partial_u, dict):
                        for _uk in agg_u:
                            agg_u[_uk] += int(partial_u.get(_uk) or 0)
                    parsed_q = {"verdict": "ERROR", "passed": None, "issues": []}
                    with self._lock:
                        cancelled_retry = (job_id in self._cancelled) or self._shutting_down
                    _dl_r, _, _rem_r = self._session_deadline_budget()
                    if (
                        _qa_try == 0
                        and not cancelled_retry
                        and not is_quota_or_rate_limit_error(err_s)
                        and (_dl_r is None or _rem_r >= 150.0)
                    ):
                        continue
                    break
            u_dict: dict[str, Any] = dict(agg_u) if any(agg_u.values()) else {}
            return res_q, u_dict, parsed_q, err_s

        def _build_qa_followup(parsed_q: dict[str, Any]) -> str:
            if parsed_q["issues"]:
                issues_block = "\n".join(parsed_q["issues"])
                return (
                    f"A pre-apply QA diff audit ({qa_model}) identified the following blockers against the task specification:\n"
                    f"{issues_block}\n\n"
                    f"{review_bundle['pointer_prompt']}\n\n"
                    "Fix each issue above cleanly in production code (including imported helper/caller modules) "
                    "without weakening test inputs or altering pre-existing test expectations (unless explicitly "
                    "required by the specification), and verify all tests pass."
                )
            if parsed_q.get("verdict") == "ERROR":
                return (
                    f"A pre-apply QA diff audit ({qa_model}) timed out or errored before verifying Flash's candidate patch.\n\n"
                    f"{review_bundle['pointer_prompt']}\n\n"
                    "Audit the candidate diff against every requirement and edge case in the Original User Request, "
                    "fix any missing branches or bugs in production code without weakening pre-existing test "
                    "expectations (unless explicitly required by the specification), and ensure all tests pass."
                )
            return (
                f"A pre-apply QA diff audit ({qa_model}) returned VERDICT: NEEDS_FIXES against the task specification.\n\n"
                f"{review_bundle['pointer_prompt']}\n\n"
                "Verify every requirement and edge case in the original task specification, fix any root causes in "
                "production code without weakening pre-existing test expectations (unless explicitly required by "
                "the specification), and ensure all tests pass."
            )

        slice_target = float(wait_budget) if wait_budget is not None else float(timeout_seconds or receipt.get("timeout_seconds") or 600.0)
        full_wait_cap = max(0.5, slice_target + 30.0)
        rem_qa_slice = self._resolve_wait_slice_seconds(
            slice_target,
            elapsed_seconds=wait_elapsed,
            allow_retry_grace=False,
        )
        saved_turn1_status = str(receipt.get("status") or "ok")

        if rem_qa_slice < full_wait_cap:
            qa_fut: Future[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]] = Future()

            def _qa_runner() -> None:
                if not qa_fut.set_running_or_notify_cancel():
                    return
                qa_fut.set_result(_invoke_qa_once())

            qa_thread = threading.Thread(target=_qa_runner, name=f"cc-agy-qa-{job_id}", daemon=True)
            with self._lock:
                self._qa_bg_threads = [t for t in self._qa_bg_threads if t.is_alive()] + [qa_thread]
            qa_thread.start()
            self._wait_for_future_or_cancel(job_id, qa_fut, timeout=rem_qa_slice)
            if not qa_fut.done():
                chain_fut: Future[dict[str, Any]] = Future()
                chain_fut.set_running_or_notify_cancel()
                running_rec = self.store.read_receipt(job_id)
                running_rec["status"] = "running"
                with self._lock:
                    raw_k_slice = self._raw_api_keys.get(job_id)
                    self._futures[job_id] = chain_fut
                    self._wait_sliced_auto_apply[job_id] = auto_apply
                running_saved = self.store.write_receipt(
                    job_id,
                    running_rec,
                    scrub=bool(running_rec.get("scrub_receipt", False)),
                    workspace_root=str(self.workspace_root),
                    api_key=raw_k_slice,
                )

                def _finish_sliced_qa() -> None:
                    try:
                        q_res, q_usage, p_qa, q_err = qa_fut.result()
                        with self._lock:
                            cancelled_bg = (job_id in self._cancelled) or self._shutting_down
                        cur_rec = self.store.read_receipt(job_id)
                        cur_rec["status"] = "cancelled" if cancelled_bg else saved_turn1_status
                        with self._lock:
                            rk = self._raw_api_keys.get(job_id)
                        self.store.write_receipt(
                            job_id,
                            cur_rec,
                            scrub=bool(cur_rec.get("scrub_receipt", False)),
                            workspace_root=str(self.workspace_root),
                            api_key=rk,
                        )
                        esc_dl_bg, _, rem_bg = self._session_deadline_budget()
                        do_esc = (
                            not cancelled_bg
                            and _should_escalate_qa(p_qa, q_err, rem_bg, esc_dl_bg is not None)
                        )
                        q_meta: dict[str, Any] = {
                            "model": qa_model,
                            "job_id": q_res.get("job_id"),
                            "verdict": p_qa["verdict"],
                            "passed": p_qa["passed"],
                            "issues": p_qa["issues"],
                            "resumed": bool(do_esc),
                            **({"post_escalation": True} if post_escalation_round else {}),
                            "review_dir": review_bundle["review_dir"],
                            "pointer_prompt_tokens": review_bundle["pointer_prompt_tokens"],
                            "usage": q_usage or None,
                            "cost": q_res.get("cost"),
                        }
                        if q_err:
                            q_meta["error"] = q_err
                        merged_rec = self.record_qa_review_on_receipt(
                            job_id,
                            qa_model=qa_model,
                            qa_meta=q_meta,
                            qa_usage=q_usage,
                        )
                        if cancelled_bg:
                            if not in_place and target_wt != self.workspace_root and target_wt.exists():
                                self._cleanup_worktree(target_wt)
                            chain_fut.set_result(merged_rec)
                            return
                        if not do_esc:
                            chain_fut.set_result(merged_rec)
                            return
                        wt_to = float(timeout_seconds or receipt.get("timeout_seconds") or 600.0)
                        if esc_dl_bg is not None:
                            wt_to = min(wt_to, max(60.0, rem_bg))
                        with self._lock:
                            if self._futures.get(job_id) is chain_fut:
                                self._futures.pop(job_id, None)
                        try:
                            cont_bg = self.delegate_continue(
                                job_id=job_id,
                                follow_up_prompt=_build_qa_followup(p_qa),
                                allowed_paths=receipt.get("allowed_paths"),
                                acceptance=None,
                                timeout_seconds=wt_to,
                                wait=False,
                                scrub_receipt=bool(receipt.get("scrub_receipt", False)),
                                auto_apply=auto_apply,
                                _prompt_source="plugin_qa_review",
                            )
                        except Exception:
                            q_meta["resumed"] = False
                            fallback_rec = self.record_qa_review_on_receipt(
                                job_id,
                                qa_model=qa_model,
                                qa_meta=q_meta,
                                qa_usage=None,
                            )
                            chain_fut.set_result(fallback_rec)
                            return
                        if cont_bg.get("continued") is False:
                            q_meta["resumed"] = False
                            fallback_rec = self.record_qa_review_on_receipt(
                                job_id,
                                qa_model=qa_model,
                                qa_meta=q_meta,
                                qa_usage=None,
                            )
                            chain_fut.set_result(fallback_rec)
                            return
                        with self._lock:
                            self._wait_sliced_auto_apply[job_id] = auto_apply
                            next_fut = self._futures.get(job_id)
                        if next_fut is not None and next_fut is not chain_fut:
                            try:
                                chain_fut.set_result(next_fut.result())
                            except Exception as nf_exc:
                                chain_fut.set_exception(nf_exc)
                        else:
                            chain_fut.set_result(self.store.read_receipt(job_id))
                    except Exception as bg_exc:
                        if not chain_fut.done():
                            chain_fut.set_exception(bg_exc)

                chain_thread = threading.Thread(
                    target=_finish_sliced_qa,
                    name=f"cc-agy-qa-chain-{job_id}",
                    daemon=True,
                )
                with self._lock:
                    self._qa_bg_threads = [t for t in self._qa_bg_threads if t.is_alive()] + [chain_thread]
                chain_thread.start()
                return self._digest_with_progress(running_saved)
            qa_res, qa_usage, parsed_qa, qa_error = qa_fut.result()
        else:
            qa_res, qa_usage, parsed_qa, qa_error = _invoke_qa_once()

        will_escalate = _should_escalate_qa(parsed_qa, qa_error, rem_budget, esc_deadline is not None)
        qa_meta: dict[str, Any] = {
            "model": qa_model,
            "job_id": qa_res.get("job_id"),
            "verdict": parsed_qa["verdict"],
            "passed": parsed_qa["passed"],
            "issues": parsed_qa["issues"],
            "resumed": bool(will_escalate),
            **({"post_escalation": True} if post_escalation_round else {}),
            "review_dir": review_bundle["review_dir"],
            "pointer_prompt_tokens": review_bundle["pointer_prompt_tokens"],
            "usage": qa_usage or None,
            "cost": qa_res.get("cost"),
        }
        if qa_error:
            qa_meta["error"] = qa_error

        self.record_qa_review_on_receipt(
            job_id,
            qa_model=qa_model,
            qa_meta=qa_meta,
            qa_usage=qa_usage,
        )
        if not will_escalate:
            return None

        followup = _build_qa_followup(parsed_qa)
        post_qa_elapsed = max(0.0, float(wait_elapsed) + max(0.0, time.monotonic() - qa_started_mono))
        esc_deadline_after_qa, _esc_elapsed_after_qa, rem_budget_after_qa = self._session_deadline_budget()
        worker_timeout = float(timeout_seconds or receipt.get("timeout_seconds") or 600.0)
        if esc_deadline_after_qa is not None:
            worker_timeout = min(worker_timeout, max(60.0, rem_budget_after_qa))
        try:
            cont_digest = self.delegate_continue(
                job_id=job_id,
                follow_up_prompt=followup,
                allowed_paths=receipt.get("allowed_paths"),
                acceptance=None,
                timeout_seconds=worker_timeout,
                wait=True,
                scrub_receipt=bool(receipt.get("scrub_receipt", False)),
                auto_apply=auto_apply,
                _wait_elapsed=post_qa_elapsed,
                _wait_budget=wait_budget,
                _prompt_source="plugin_qa_review",
            )
            if cont_digest.get("continued") is False:
                qa_meta["resumed"] = False
                self.record_qa_review_on_receipt(
                    job_id,
                    qa_model=qa_model,
                    qa_meta=qa_meta,
                    qa_usage=None,
                )
                cont_digest["qa_review"] = dict(qa_meta)
            return cont_digest
        except BudgetExceededError as exc:
            qa_meta["resumed"] = False
            updated_rec = self.record_qa_review_on_receipt(
                job_id,
                qa_model=qa_model,
                qa_meta=qa_meta,
                qa_usage=None,
            )
            digest = self._digest_with_progress(updated_rec)
            digest["budget_exceeded"] = True
            digest["budget_exceeded_note"] = str(exc)
            return digest

    def delegate_result(
        self,
        job_id: str,
        *,
        wait: bool = True,
        timeout: float = 240.0,
    ) -> dict[str, Any]:
        """Return the compact completion digest for job_id, optionally waiting if still running."""
        safe_id = validate_job_id(job_id)
        wait_started_mono = time.monotonic()
        with self._lock:
            fut = self._futures.get(safe_id)
            has_sliced_state = safe_id in self._wait_sliced_auto_apply
        capped_timeout = self._resolve_wait_slice_seconds(min(float(timeout), 540.0))
        if capped_timeout < 1.0 and float(timeout) <= 30.0 and has_sliced_state:
            capped_timeout = max(capped_timeout, float(timeout))
        if fut is not None and wait and not fut.done():
            self._wait_for_future_or_cancel(safe_id, fut, timeout=capped_timeout)
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
                    shutil.copy2(src, dst)
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
                        shutil.copy2(src_inc, dst_inc)
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
                        shutil.copy2(src_raw, dst_raw)
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
        return {"state": "git_blob", "sha256": base_hash, "data": show.stdout}

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
            kind_probe = self.store.read_receipt(safe_id).get("kind")
        except FileNotFoundError:
            kind_probe = None
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

        enforce_daily_budget(self.store, workspace_root=self.workspace_root)

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
                except Exception:
                    diffstat_note = ""
            context_note = (
                f"[Prior turn context (baseline {baseline_sha}, files_changed={receipt.get('files_changed', [])}, "
                f"run `git diff {baseline_sha}` to inspect prior work): "
                f"{prior_summary}]{no_edits_warning}{diffstat_note}\n{clean_followup}"
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
        inherited_auth_mode = (
            str(prior_auth_mode)
            if prior_auth_mode in ("gemini_enterprise", "api_key", "vertex") and not project and not api_key
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
            except Exception:
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
                "summary": str(receipt.get("summary") or "")[:1500],
                "acceptance_results": list(receipt.get("acceptance_results") or [])[:5],
            }
        )
        receipt["prior_attempts"] = prior_attempts[-5:]
        receipt["prior_tool_call_count"] = len(prior_tool_calls_snapshot)
        receipt["status"] = "running"
        receipt["error"] = None
        receipt["stop_reason"] = None
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
        except Exception:
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
                except Exception:
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
            )
            self._futures[safe_id] = fut
            self._owned_job_ids.add(safe_id)

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
            self._wait_for_future_or_cancel(safe_id, fut, timeout=wait_slice)
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

        if callable(cancel_fn):
            try:
                cancel_fn()
            except Exception:
                pass
        if still_running and fut is not None:
            try:
                fut.result(timeout=2.0)
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
            except Exception:
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
        except Exception:
            return int(receipt.get("tool_call_count") or 0)

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
            except Exception:
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
        except Exception:
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
            except Exception:
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
            except Exception:
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
                    except Exception:
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
            except Exception:
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
            if receipt.get("escalated_from_model"):
                slim["escalated_from_model"] = receipt.get("escalated_from_model")
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
