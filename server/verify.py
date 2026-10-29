"""Run verification commands for file tasks in a job copy.

Each command runs in its own process group with vendor credentials removed from the environment and
is killed as a group when it overruns its timeout (and after it exits, so stray background children
never outlive the check). Output is kept as a bounded tail; the full log is written next to the job
so it stays addressable when the tail is not enough.
"""
from __future__ import annotations

import atexit
from collections import deque
from collections.abc import Collection, Mapping, Sequence
import codecs
from dataclasses import dataclass, field
import hashlib
import os
import posixpath
import re
import selectors
import shutil
import sys
import tempfile
import threading
from typing import Any, Callable
from pathlib import Path, PurePosixPath
import signal
import stat
import subprocess
import time
import uuid

TAIL_CHARS = 3_600
SECRET_ENV_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "CC_AGY_VERTEX_BASE_URL_TOKEN",
    "ANTIGRAVITY_OAUTH_TOKEN",
    "CC_AGY_ENTERPRISE_ACCESS_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)
# Credential-looking names (vendor keys, tokens, passwords, connection strings) that check and tool
# commands never need. Matched on whole underscore-delimited words, so TOKENIZERS_PARALLELISM,
# MAX_TOKENS, PATH or GIT_AUTHOR_NAME stay untouched.
_SECRET_ENV_NAME_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|ACCESS_?KEY(?:_?ID)?|SECRETS?(?:_?ACCESS)?(?:_?KEY)?|PRIVATE_?KEY|KEY|TOKEN|"
    r"AUTH_?TOKEN|AUTH(?:ORIZATION)?|PASSWORD|PASSWD|PWD|PASSPHRASE|CREDENTIALS?|DSN|"
    r"CONNECTION_?STRING|PAT)(?:_|$)",
    re.I,
)
# A URL value carrying a password or a bare token as user info, e.g. postgres://app:s3cret@db/app.
_CREDENTIAL_URL_RE = re.compile(r"[a-z][a-z0-9+.-]*://(?:[^/\s:@]*:[^/\s@]+|[^/\s:@]{16,})@", re.I)
_EXTRA_SECRET_ENV_NAMES = frozenset({"PGPASSWORD", "MYSQL_PWD", "DOCKER_AUTH_CONFIG", "KRB5CCNAME"})
_NEVER_SECRET_ENV_NAMES = frozenset({"PWD", "OLDPWD"})
# Comma-separated variable names a workspace owner explicitly lets through to check commands.
CHECK_ENV_ALLOW_VAR = "CC_AGY_CHECK_ENV_ALLOW"
# Paths the plugin itself writes inside a job copy; never treated as check side effects.
_PLUGIN_SCRATCH_PREFIXES = (".ctx-session-reads/", ".cc-agy-review/")


def _env_allowlist(environ: Mapping[str, str]) -> frozenset[str]:
    raw = str(environ.get(CHECK_ENV_ALLOW_VAR) or "")
    return frozenset(
        name.strip() for name in raw.split(",")
        if name.strip() and not name.strip().startswith(("CC_AGY_", "CLAUDE_PLUGIN_"))
    )


def secret_env_names(environ: Mapping[str, str] | None = None) -> set[str]:
    """Names in ``environ`` whose name or value looks like a credential, minus CC_AGY_CHECK_ENV_ALLOW."""
    src = os.environ if environ is None else environ
    allow = _env_allowlist(src)
    found: set[str] = set()
    for name, value in src.items():
        if name in allow or name in _NEVER_SECRET_ENV_NAMES:
            continue
        if (name in SECRET_ENV_KEYS or name in _EXTRA_SECRET_ENV_NAMES or _SECRET_ENV_NAME_RE.search(name)
                or _CREDENTIAL_URL_RE.search(str(value or ""))):
            found.add(name)
    return found


def credential_free_environ(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy of ``environ`` without plugin internals or credentials: the base for every command the
    plugin runs in a workspace (acceptance checks, batch checks, baseline probes, runner detection).

    Known vendor keys stay as empty strings so a repo ``.env`` loader cannot fill them back in; other
    credential-looking variables are dropped. CC_AGY_CHECK_ENV_ALLOW lists names to keep.
    """
    src = dict(os.environ if environ is None else environ)
    secrets = secret_env_names(src)
    allow = _env_allowlist(src)
    env = {
        k: v for k, v in src.items()
        if not k.startswith(("CC_AGY_", "CLAUDE_PLUGIN_")) and k not in secrets
    }
    for key in SECRET_ENV_KEYS:
        if key not in allow:
            env[key] = ""
    return env


@dataclass
class CommandResult:
    command: str
    passed: bool
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    tail: str
    log_path: str = ""
    vacuous: bool = False
    side_effects_restored: list[str] = field(default_factory=list)
    untrusted_side_effects: list[str] = field(default_factory=list)
    credential_env_removed: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "command": self.command,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "tail": self.tail,
            "log_path": self.log_path,
        }
        if self.vacuous:
            out["vacuous"] = True
        if self.side_effects_restored:
            out["side_effects_restored"] = list(self.side_effects_restored)
        if self.untrusted_side_effects:
            out["untrusted_side_effects"] = list(self.untrusted_side_effects)
        if self.credential_env_removed:
            out["credential_env_removed"] = list(self.credential_env_removed)
        return out


def scrubbed_env(cwd: Path, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for check commands: ``credential_free_environ`` plus the repo venv first on PATH."""
    from server.jobs import _build_workdir_exec_env

    env = credential_free_environ(base_env)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return _build_workdir_exec_env(cwd, env)


# ------------------------------------------------------------------------------------------------
# Side effects of check commands on the files that will ship
# ------------------------------------------------------------------------------------------------

_SNAPSHOT_MAX_BYTES = 256 * 1024 * 1024
_SNAPSHOT_MAX_FILES = 2_000
# Files test runs legitimately rewrite that do not change what the tests exercised.
_GENERATED_ARTIFACT_NAMES = frozenset({
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml", "bun.lock", "bun.lockb",
    "poetry.lock", "Pipfile.lock", "uv.lock", "pdm.lock", "Cargo.lock", "go.sum", "Gemfile.lock",
    "composer.lock", "mix.lock", "pubspec.lock", "Podfile.lock", "packages.lock.json",
    "coverage.xml", "lcov.info", ".coverage", "coverage-final.json", "junit.xml",
})


_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
# Hooks and fsmonitor are off for every git call the plugin makes around checks: a check may have
# planted them in the job copy, and they would otherwise run with the plugin's own environment.
_GIT_EXEC_GUARD = (
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.fsmonitor=false",
    "-c", "core.pager=cat",
    "-c", "core.sshCommand=false",
    "-c", "commit.gpgsign=false",
    "-c", "tag.gpgSign=false",
    "-c", "protocol.allow=never",
)
_GIT_SAFE_ARGS = (*_GIT_EXEC_GUARD, "-c", "core.quotePath=false")
# Git metadata a check could rewrite to make later git commands run code (hooks, fsmonitor, filters).
_GIT_META_FILES = ("config", "config.worktree", "info/attributes", "info/exclude")
# In-progress operation state in the per-worktree git dir; a check must not start or end one.
_GIT_STATE_PATHS = ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "REBASE_HEAD", "rebase-merge", "rebase-apply")
_GIT_TWO_ARG_OPTS = frozenset({"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env"})
_GLOBAL_FILTER_LOCK = threading.Lock()
_GLOBAL_FILTER_CACHE: tuple[tuple[tuple[str, int, int], ...], tuple[str, ...]] | None = None


def git_safe_env() -> dict[str, str]:
    """Credential-free environment for git subprocesses, never redirected by inherited GIT_* paths."""
    env = credential_free_environ()
    for key in (
        "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR",
        "GIT_CONFIG", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT",
        "GIT_EXTERNAL_DIFF", "GIT_EXEC_PATH", "GIT_TEMPLATE_DIR",
        "GIT_ASKPASS", "SSH_ASKPASS", "GIT_SSH", "GIT_SSH_COMMAND", "GIT_EDITOR",
    ):
        env.pop(key, None)
    for key in [k for k in env if k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))]:
        env.pop(key, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_PAGER"] = "cat"
    return env


def _inject_git_subcommand_guards(args: Sequence[str]) -> list[str]:
    out = list(args)
    i = 0
    while i < len(out):
        tok = out[i]
        if tok in _GIT_TWO_ARG_OPTS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if tok in ("diff", "show"):
            tail = out[i + 1:]
            extra = [flag for flag in ("--no-ext-diff", "--no-textconv") if flag not in tail]
            if extra:
                out[i + 1:i + 1] = extra
        break
    return out


def _config_file_looks_filter_capable(cfg: Path) -> bool:
    try:
        if cfg.is_symlink():
            return True
        if not cfg.is_file():
            return False
        low = cfg.read_bytes().lower()
        return b"filter" in low or b"include" in low
    except OSError:
        return True


def _local_git_configs_may_define_filters(cwd: Path | str | None) -> bool:
    if cwd is None:
        return False
    try:
        cur = Path(cwd).resolve()
    except OSError:
        return True
    for candidate in (cur, *cur.parents):
        dot_git = candidate / ".git"
        try:
            if dot_git.is_symlink():
                return True
            if dot_git.is_dir():
                if (
                    _config_file_looks_filter_capable(dot_git / "config")
                    or _config_file_looks_filter_capable(dot_git / "config.worktree")
                ):
                    return True
                continue
            if dot_git.is_file():
                line = dot_git.read_text(encoding="utf-8", errors="surrogateescape").strip()
                if not line.startswith("gitdir:"):
                    return True
                raw_gd = line[len("gitdir:"):].strip()
                git_dir = Path(raw_gd) if os.path.isabs(raw_gd) else (candidate / raw_gd).resolve()
                common_dir = git_dir
                commondir_file = git_dir / "commondir"
                if commondir_file.is_symlink():
                    return True
                if commondir_file.is_file():
                    rel_common = commondir_file.read_text(encoding="utf-8", errors="surrogateescape").strip()
                    common_dir = Path(rel_common) if os.path.isabs(rel_common) else (git_dir / rel_common).resolve()
                if (
                    _config_file_looks_filter_capable(git_dir / "config")
                    or _config_file_looks_filter_capable(git_dir / "config.worktree")
                    or _config_file_looks_filter_capable(common_dir / "config")
                ):
                    return True
                continue
        except OSError:
            return True
    return False


def _query_filter_names(cwd: Path | str | None, env: dict[str, str]) -> tuple[str, ...]:
    try:
        res = subprocess.run(
            ["git", *_GIT_EXEC_GUARD, "config", "--includes", "--null", "--name-only",
             "--get-regexp", r"^filter\..*\.(clean|smudge|process|required)$"],
            cwd=None if cwd is None else str(cwd),
            env=env,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if res.returncode not in (0, 1) or not res.stdout:
        return ()
    names: set[str] = set()
    for raw in res.stdout.split(b"\0"):
        if not raw:
            continue
        key = raw.decode("utf-8", "surrogateescape")
        if key.startswith("filter.") and key.count(".") >= 2:
            names.add(key[len("filter."):].rsplit(".", 1)[0])
    return tuple(sorted(names))


def _global_filter_names(base_env: Mapping[str, str]) -> tuple[str, ...]:
    global _GLOBAL_FILTER_CACHE
    home = Path(base_env.get("HOME") or Path.home())
    xdg = Path(base_env.get("XDG_CONFIG_HOME") or (home / ".config"))
    user_configs = (home / ".gitconfig", xdg / "git" / "config")
    candidates = (Path("/etc/gitconfig"), *user_configs)
    user_has_include = False
    sig_items: list[tuple[str, int, int]] = []
    for p in candidates:
        try:
            st = p.stat()
            sig_items.append((str(p), st.st_mtime_ns, st.st_size))
            if p in user_configs and b"include" in p.read_bytes().lower():
                user_has_include = True
        except OSError:
            sig_items.append((str(p), -1, -1))
    sig = tuple(sig_items)
    if not user_has_include:
        with _GLOBAL_FILTER_LOCK:
            if _GLOBAL_FILTER_CACHE is not None and _GLOBAL_FILTER_CACHE[0] == sig:
                return _GLOBAL_FILTER_CACHE[1]
    probe_env = dict(base_env)
    probe_env["GIT_DIR"] = os.devnull
    found = _query_filter_names(None, probe_env)
    if not user_has_include:
        with _GLOBAL_FILTER_LOCK:
            _GLOBAL_FILTER_CACHE = (sig, found)
    return found


def _apply_filter_neutralizers(cwd: Path | str | None, env: dict[str, str]) -> list[str]:
    names = set(_global_filter_names(env))
    if "GIT_DIR" in env or "GIT_CONFIG" in env or _local_git_configs_may_define_filters(cwd):
        names.update(_query_filter_names(cwd, env))
    if not names:
        return []
    cli_overrides: list[str] = []
    env_pairs: list[tuple[str, str]] = []
    for name in sorted(names):
        if "=" in name or "\n" in name:
            for attr, val in (("required", "false"), ("clean", ""), ("smudge", ""), ("process", "")):
                env_pairs.append((f"filter.{name}.{attr}", val))
        else:
            cli_overrides.extend((
                "-c", f"filter.{name}.required=false",
                "-c", f"filter.{name}.clean=",
                "-c", f"filter.{name}.smudge=",
                "-c", f"filter.{name}.process=",
            ))
    if env_pairs:
        env["GIT_CONFIG_COUNT"] = str(len(env_pairs))
        for idx, (k, v) in enumerate(env_pairs):
            env[f"GIT_CONFIG_KEY_{idx}"] = k
            env[f"GIT_CONFIG_VALUE_{idx}"] = v
    return cli_overrides


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return run_git(["-c", "core.quotePath=false", *args], cwd, capture_output=True, check=False)


def run_git(args: Sequence[str], cwd: Path | str | None = None, *,
            env_overrides: Mapping[str, str] | None = None, **kwargs: Any) -> subprocess.CompletedProcess:
    """Run one of the plugin's own git commands with hooks, fsmonitor, external diff/textconv, and
    clean/smudge/process filters neutralized inside a credential-free environment; other ``subprocess.run``
    options pass through. ``env_overrides`` (such as ``GIT_INDEX_FILE``) is applied on top."""
    env = git_safe_env()
    if env_overrides:
        env.update(env_overrides)
    filter_args = _apply_filter_neutralizers(cwd, env)
    guarded_args = _inject_git_subcommand_guards(args)
    return subprocess.run(["git", *_GIT_EXEC_GUARD, *filter_args, *guarded_args],
                          cwd=None if cwd is None else str(cwd), env=env, **kwargs)


def _git_paths(cwd: Path, *args: str) -> list[str] | None:
    res = _git(cwd, *args)
    if res.returncode != 0:
        return None
    return [p for p in res.stdout.decode("utf-8", "surrogateescape").split("\0") if p]


def _git_text(cwd: Path, *args: str) -> str | None:
    res = _git(cwd, *args)
    return res.stdout.decode("utf-8", "surrogateescape").strip() if res.returncode == 0 else None


def _symlink_in_path(root: Path, rel: str) -> bool:
    current = root
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


@dataclass
class WorkspaceSnapshot:
    """The change set as it stood before one check command ran.

    Covers the whole git checkout that contains ``cwd`` (a check run from a subdirectory can still
    touch its siblings): every tracked file that differs from HEAD, every untracked non-ignored file
    (in a job copy, the files the worker or earlier batch tasks created), the index, and the git
    metadata a check could use to make later git commands run code. Files are backed up to a temp
    directory; anything a check changes on top of this state is a side effect.
    """

    root: Path
    cwd_prefix: str
    base: str
    entries: dict[str, tuple[str, str | None, int | None]]
    head_sha: str | None = None
    head_ref: str | None = None
    index_tree: str | None = None
    index_raw_sha: str | None = None
    index_was_absent: bool = False
    git_dirs: dict[str, Path] = field(default_factory=dict)
    git_dir_modes: dict[str, int | None] = field(default_factory=dict)
    git_meta: dict[str, tuple[str, str | None, int | None]] = field(default_factory=dict)
    hooks: dict[str, frozenset[str]] = field(default_factory=dict)
    hooks_dir_state: dict[str, tuple[bool, int | None, str | None, str | None]] = field(default_factory=dict)
    info_dir_state: dict[str, tuple[bool, int | None, str | None, str | None]] = field(default_factory=dict)
    git_state: frozenset[str] = frozenset()
    index_digest: str | None = None
    index_backup: str | None = None
    index_bytes: bytes | None = None
    index_ns: tuple[int, int] | None = None
    dirty: frozenset[str] = frozenset()
    backup_dir: Path | None = None
    budget: int = _SNAPSHOT_MAX_BYTES
    keep_backups: bool = False
    inline_bytes: dict[str, bytes] = field(default_factory=dict)
    inline_budget: int = 32 << 20
    backup_hashes: dict[str, str] = field(default_factory=dict)
    backup_fds: dict[str, Any] = field(default_factory=dict)
    clean_tracked: dict[str, tuple[int, int, int, int, int, int, str | None]] | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _symlink_target_digest(link_path: Path) -> str | None:
    """Read-only digest of a pre-existing symlink target so writes through the link are detected."""
    try:
        resolved = link_path.resolve(strict=False)
        if not resolved.exists() and not resolved.is_symlink():
            return "absent"
        st = resolved.lstat()
    except OSError:
        return None
    h = hashlib.sha256()
    if resolved.is_file() and not resolved.is_symlink():
        try:
            payload = _sha256_file(resolved) if st.st_size <= (16 << 20) else f"{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            return None
        h.update(f"file:{st.st_mode & 0o7777}:{payload}\0".encode("utf-8", "surrogateescape"))
        return h.hexdigest()
    if not resolved.is_dir() or resolved.is_symlink():
        return None
    h.update(f"dir:{st.st_mode & 0o7777}\0".encode("utf-8"))
    walk_err = False

    def _on_err(_e: OSError) -> None:
        nonlocal walk_err
        walk_err = True

    count = 0
    for dirpath, dirnames, filenames in os.walk(resolved, topdown=True, onerror=_on_err, followlinks=False):
        dp = Path(dirpath)
        dirnames.sort()
        filenames.sort()
        kept: list[str] = []
        for d in dirnames:
            count += 1
            if count > 2_000:
                return None
            sub = dp / d
            rel_d = sub.relative_to(resolved).as_posix()
            try:
                if sub.is_symlink():
                    h.update(f"L:{rel_d}:{os.readlink(sub)}\0".encode("utf-8", "surrogateescape"))
                else:
                    h.update(f"D:{rel_d}:{sub.lstat().st_mode & 0o7777}\0".encode("utf-8", "surrogateescape"))
                    kept.append(d)
            except OSError:
                return None
        dirnames[:] = kept
        for fn in filenames:
            count += 1
            if count > 2_000:
                return None
            fp = dp / fn
            rel_f = fp.relative_to(resolved).as_posix()
            try:
                if fp.is_symlink():
                    h.update(f"L:{rel_f}:{os.readlink(fp)}\0".encode("utf-8", "surrogateescape"))
                elif fp.is_file():
                    fst = fp.lstat()
                    fpay = _sha256_file(fp) if fst.st_size <= (16 << 20) else f"{fst.st_size}:{fst.st_mtime_ns}"
                    h.update(f"F:{rel_f}:{fst.st_mode & 0o7777}:{fpay}\0".encode("utf-8", "surrogateescape"))
                else:
                    return None
            except OSError:
                return None
    return None if walk_err else h.hexdigest()


def _files_equal_bytes(a: Path, b: Path | str) -> bool:
    """Compare two regular files byte-for-byte without consulting ``filecmp._cache``."""
    pb = Path(b)
    try:
        sa = a.lstat()
        sb = pb.lstat()
        if not stat.S_ISREG(sa.st_mode) or not stat.S_ISREG(sb.st_mode) or sa.st_size != sb.st_size:
            return False
        with open(a, "rb") as fa, open(pb, "rb") as fb:
            while True:
                ca = fa.read(65_536)
                cb = fb.read(65_536)
                if ca != cb:
                    return False
                if not ca:
                    return True
    except OSError:
        return False


def _copy_to_unlinked_tmp_and_sha256(src: Path) -> tuple[Any, str]:
    """Copy ``src`` into an unlinked O_CLOEXEC ``TemporaryFile`` while computing its SHA-256."""
    digest = hashlib.sha256()
    tmp_fh = tempfile.TemporaryFile()
    try:
        with open(src, "rb") as f_in:
            while True:
                chunk = f_in.read(262_144)
                if not chunk:
                    break
                digest.update(chunk)
                tmp_fh.write(chunk)
        tmp_fh.flush()
        return tmp_fh, digest.hexdigest()
    except BaseException:
        tmp_fh.close()
        raise


def _close_backup_fds(fds: Mapping[str, Any]) -> None:
    for fh in list(fds.values()):
        try:
            fh.close()
        except OSError:
            pass


def _capture(
    snap: WorkspaceSnapshot,
    path: Path,
    *,
    backup_cache: dict[str, tuple[str, int, int, int, int, int, int, bytes | None, str | None, Any]] | None = None,
    cache_key: str | None = None,
    session_fds: dict[str, Any] | None = None,
) -> tuple[str, str | None, int | None] | None:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return ("absent", None, None)
    except OSError:
        return None
    if stat.S_ISLNK(st.st_mode):
        return ("symlink", os.readlink(path), None)
    if stat.S_ISREG(st.st_mode):
        mode = st.st_mode & 0o7777
        if st.st_size > snap.budget or len(snap.entries) >= _SNAPSHOT_MAX_FILES:
            # Past the backup budget: record size and mtime only (detected, reported, not restorable).
            return ("stat", f"{st.st_size}:{st.st_mtime_ns}", mode)
        if backup_cache is not None and cache_key is not None:
            cached = backup_cache.get(cache_key)
            if cached is not None:
                saved_str, c_dev, c_ino, c_size, c_mtime_ns, c_ctime_ns, _c_mode, c_bytes, c_sha, c_fh = cached
                if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) == (
                    c_dev, c_ino, c_size, c_mtime_ns, c_ctime_ns
                ):
                    if c_bytes is not None and snap.inline_budget >= len(c_bytes):
                        if path.read_bytes() == c_bytes:
                            snap.budget -= st.st_size
                            snap.inline_budget -= len(c_bytes)
                            snap.inline_bytes[saved_str] = c_bytes
                            return ("file", saved_str, mode)
                    elif c_sha is not None and c_fh is not None and not getattr(c_fh, "closed", True):
                        if _sha256_file(path) == c_sha:
                            snap.budget -= st.st_size
                            snap.backup_hashes[saved_str] = c_sha
                            snap.backup_fds[saved_str] = c_fh
                            return ("file", saved_str, mode)
        snap.budget -= st.st_size
        slot = f"mem:{uuid.uuid4().hex[:12]}" if backup_cache is not None else f"mem:{len(snap.entries) + len(snap.git_meta):06d}"
        raw_inline: bytes | None = None
        sha_hex: str | None = None
        unlinked_fh: Any = None
        if st.st_size <= 262_144 and snap.inline_budget >= st.st_size:
            raw_inline = path.read_bytes()
            snap.inline_budget -= len(raw_inline)
            snap.inline_bytes[slot] = raw_inline
        else:
            unlinked_fh, sha_hex = _copy_to_unlinked_tmp_and_sha256(path)
            snap.backup_hashes[slot] = sha_hex
            snap.backup_fds[slot] = unlinked_fh
            if session_fds is not None:
                session_fds[slot] = unlinked_fh
        if backup_cache is not None and cache_key is not None:
            old = backup_cache.get(cache_key)
            if old is not None and old[9] is not None and old[9] is not unlinked_fh:
                try:
                    old[9].close()
                except OSError:
                    pass
                if session_fds is not None:
                    session_fds.pop(old[0], None)
            backup_cache[cache_key] = (
                slot, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, mode, raw_inline, sha_hex, unlinked_fh
            )
        return ("file", slot, mode)
    if stat.S_ISDIR(st.st_mode):
        return ("dir", None, st.st_mode & 0o7777)
    return None


def checkout_root(cwd: Path) -> Path | None:
    """Top of the git checkout containing ``cwd``, or None outside git."""
    top = _git_text(cwd, "rev-parse", "--show-toplevel")
    return Path(top) if top else None


def _resolve_git_path(base: Path, value: str) -> Path:
    return Path(value) if os.path.isabs(value) else (Path(base) / value).resolve()


def _repo_layout(cwd: Path) -> tuple[Path, dict[str, Path]] | None:
    """Checkout root plus the per-worktree git dir and, for a linked worktree (.git is a file), the
    shared common dir that holds config and hooks, from one rev-parse call (each check command takes a
    snapshot, so fewer git processes keep per-command overhead low). No --path-format: git before 2.31
    echoes unknown options and still exits 0; a relative common dir is resolved against cwd."""
    res = _git(cwd, "rev-parse", "--show-toplevel", "--absolute-git-dir", "--git-common-dir")
    lines = res.stdout.decode("utf-8", "surrogateescape").splitlines() if res.returncode == 0 else []
    if len(lines) != 3:
        return None
    root, own, common = Path(lines[0]), Path(lines[1]), _resolve_git_path(Path(cwd), lines[2])
    dirs = {".git": own}
    if common.resolve() != own.resolve():
        dirs[".git-common"] = common
    return root, dirs


def _git_dirs(root: Path) -> dict[str, Path]:
    """Git dirs for ``root`` keyed by the label used in reports (see ``_repo_layout``)."""
    layout = _repo_layout(root)
    return layout[1] if layout is not None else {}


def _index_digest(root: Path, *, index_file: str | None = None) -> str | None:
    overrides = {"GIT_INDEX_FILE": index_file} if index_file is not None else None
    res = run_git(["-c", "core.quotePath=false", "ls-files", "--stage", "-z"], root,
                  env_overrides=overrides, capture_output=True, check=False)
    return hashlib.sha256(res.stdout).hexdigest() if res.returncode == 0 else None


def _write_tree_for_index(root: Path, *, index_file: str | None = None) -> str | None:
    overrides = {"GIT_INDEX_FILE": index_file} if index_file is not None else None
    res = run_git(["write-tree"], root, env_overrides=overrides, capture_output=True, check=False)
    return res.stdout.decode("utf-8", "surrogateescape").strip() if res.returncode == 0 else None


def _read_head_ref(own_dir: Path | None, root: Path, status_branch_head: str | None = None) -> str | None:
    """Return ``symbolic-ref -q HEAD`` without spawning git on normal files-backend checkouts."""
    if own_dir is not None:
        head_file = own_dir / "HEAD"
        try:
            if not head_file.is_symlink() and head_file.is_file():
                raw = head_file.read_text(encoding="utf-8", errors="surrogateescape").strip()
                if raw.startswith("ref:"):
                    ref = raw[4:].strip()
                    if ref and ref != "refs/heads/.invalid":
                        return ref
                else:
                    return None
        except OSError:
            pass
    if status_branch_head == "(detached)":
        return None
    return _git_text(root, "symbolic-ref", "-q", "HEAD") or None


def _git_status_porcelain(
    root: Path,
    own_dir: Path | None,
    *,
    include_untracked: bool = False,
) -> tuple[str | None, str | None, list[str], list[str]] | None:
    """Query HEAD sha/ref, dirty tracked paths, and (optionally) untracked paths in one git call."""
    untracked_arg = "--untracked-files=all" if include_untracked else "--untracked-files=no"
    res = run_git(
        ["--no-optional-locks", "-c", "core.quotePath=false", "status",
         "--porcelain=v2", "--branch", "--no-renames", "-z", untracked_arg],
        root,
        capture_output=True,
        check=False,
    )
    if res.returncode != 0:
        return None
    tokens = res.stdout.decode("utf-8", "surrogateescape").split("\0")
    head_sha: str | None = None
    branch_head: str | None = None
    dirty: list[str] = []
    untracked: list[str] = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        idx += 1
        if not tok:
            continue
        if tok.startswith("# branch.oid "):
            oid = tok[len("# branch.oid "):].strip()
            head_sha = None if (not oid or oid == "(initial)") else oid
        elif tok.startswith("# branch.head "):
            branch_head = tok[len("# branch.head "):].strip()
        elif tok.startswith("1 "):
            parts = tok.split(" ", 8)
            if len(parts) == 9 and parts[8]:
                dirty.append(parts[8])
        elif tok.startswith("2 "):
            parts = tok.split(" ", 9)
            if len(parts) == 10 and parts[9]:
                dirty.append(parts[9])
            if idx < len(tokens):
                orig_p = tokens[idx]
                idx += 1
                if orig_p:
                    dirty.append(orig_p)
        elif tok.startswith("u "):
            parts = tok.split(" ", 10)
            if len(parts) == 11 and parts[10]:
                dirty.append(parts[10])
        elif tok.startswith("? "):
            rel_u = tok[2:]
            if rel_u:
                untracked.append(rel_u)
    head_ref = _read_head_ref(own_dir, root, branch_head)
    return head_ref, head_sha, dirty, untracked


def _hook_names(git_dir: Path | None) -> frozenset[str]:
    hooks = git_dir / "hooks" if git_dir is not None else None
    if hooks is None or hooks.is_symlink() or not hooks.is_dir():
        return frozenset()
    return frozenset(p.name for p in hooks.iterdir() if not p.name.endswith(".sample"))


def untracked_baseline(cwd: Path, *, _root: Path | None = None) -> frozenset[str] | None:
    """Untracked, non-ignored files of the checkout containing ``cwd``, relative to its root; taken
    once when a run starts, so files earlier commands of the same run create count as run artifacts."""
    root = _root if _root is not None else checkout_root(cwd)
    if root is None:
        return None
    paths = _git_paths(root, "ls-files", "--others", "--exclude-standard", "-z")
    return frozenset(paths) if paths is not None else None


def _with_writable_parent(root: Path, target: Path, action: Callable[[], object]) -> bool:
    """Run ``action``, which takes ``target`` out of its directory (unlink or move). If the check made
    ancestor directories (including ``root`` itself) read-only, make them owner-writable (only inside
    the checkout, never through a link) and try once more. Returns False when it still fails."""
    try:
        action()
        return True
    except PermissionError:
        try:
            if root.is_symlink():
                return False
            root_res = root.resolve()
            try:
                rel_parts = target.parent.relative_to(root).parts
            except ValueError:
                rel_parts = target.parent.resolve().relative_to(root_res).parts
            cur = root
            for step in (None, *rel_parts):
                if step is not None:
                    cur = cur / step
                if cur.is_symlink() or not cur.resolve().is_relative_to(root_res):
                    return False
                mode = cur.stat().st_mode & 0o7777
                if (mode & 0o700) != 0o700:
                    os.chmod(cur, mode | 0o700)
            action()
            return True
        except (OSError, ValueError):
            pass
    except OSError:
        pass
    return False


def _clear_quarantine_dest(dest: Path) -> None:
    """Remove any partial read-only file left at ``dest`` by a failed cross-device ``shutil.move``."""
    try:
        if dest.is_symlink():
            dest.unlink(missing_ok=True)
        elif dest.is_file():
            os.chmod(dest, 0o600)
            dest.unlink(missing_ok=True)
    except OSError:
        pass


def _unlink_forcefully(root: Path, target: Path) -> bool:
    """Unlink a file or link a check created, even from a directory the check made read-only."""
    return _with_writable_parent(root, target, lambda: target.unlink(missing_ok=True))


def _prune_empty_dirs(top: Path) -> None:
    """Remove empty directories under ``top`` and ``top`` itself if it ends up empty; never a file."""
    for dirpath, _dirnames, _filenames in os.walk(top, topdown=False):
        try:
            os.rmdir(dirpath)
        except OSError:
            pass


def _canonical_non_symlink_key(path: Path) -> Path | None:
    try:
        if path.is_symlink():
            return None
        return path.parent.resolve() / path.name
    except OSError:
        return None


def remove_new_untracked(
    cwd: Path,
    baseline: Collection[str] | None,
    *,
    whole_checkout: bool = True,
    quarantine_dir: Path | None = None,
    report: dict[str, Any] | None = None,
    preserve_paths: Collection[Path] = (),
    _root: Path | None = None,
) -> list[str]:
    """Remove untracked, non-ignored files that appeared since ``baseline`` (relative to the checkout
    root), so they never land in the job's diff; return the removed paths.

    In an isolated job copy the whole checkout is cleaned, since checks run from a subdirectory can
    write next to it. In the user's own tree (``whole_checkout=False``) only files under ``cwd`` are
    touched: run artifacts (reports, coverage, caches) and links are deleted, and any other new file,
    which may be a person's work, is moved into a fresh ``run-*`` directory under ``quarantine_dir``
    (so runs never collide, and no live link is ever placed there). A file that cannot be moved is left
    where it is rather than deleted. ``report`` receives ``quarantine_dir``, ``moved`` and
    ``left_in_place``.
    """
    root = _root if _root is not None else checkout_root(cwd)
    if root is None or baseline is None:
        return []
    try:
        prefix = Path(cwd).resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        prefix = ""
    scope = "" if whole_checkout or prefix in ("", ".") else prefix.rstrip("/") + "/"
    now = _git_paths(root, "ls-files", "--others", "--exclude-standard", "-z") or []
    preserved = {k for p in preserve_paths if (k := _canonical_non_symlink_key(Path(p))) is not None}
    removed: list[str] = []
    moved: list[str] = []
    left: list[str] = []
    run_dir: Path | None = None
    for rel in sorted(set(now) - set(baseline)):
        if rel.startswith(_PLUGIN_SCRATCH_PREFIXES) or (scope and not rel.startswith(scope)):
            continue
        target = root / rel
        if not (target.is_file() or target.is_symlink()):
            continue
        if preserved and _canonical_non_symlink_key(target) in preserved:
            continue
        keep_copy = (not whole_checkout and quarantine_dir is not None and not target.is_symlink()
                     and not _is_run_artifact(rel))
        if not keep_copy:
            if _unlink_forcefully(root, target):
                removed.append(rel)
            else:
                left.append(rel)
            continue
        try:
            if run_dir is None:
                Path(quarantine_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
                run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(quarantine_dir)))
            dest = run_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            left.append(rel)
            continue
        def _move_to_dest() -> None:
            _clear_quarantine_dest(dest)
            shutil.move(str(target), str(dest))

        if _with_writable_parent(root, target, _move_to_dest):
            moved.append(rel)
            removed.append(rel)
        else:
            _clear_quarantine_dest(dest)
            left.append(rel)
    if run_dir is not None:
        # Directories made for moves that failed; the run directory itself goes when nothing was moved.
        _prune_empty_dirs(run_dir)
        if not run_dir.exists():
            run_dir = None
            try:
                os.rmdir(quarantine_dir)
            except OSError:
                pass
    if report is not None:
        report.update({"quarantine_dir": str(run_dir) if run_dir else None, "moved": moved, "left_in_place": left})
    return removed


def _capture_git_subdir_state(sub: Path) -> tuple[bool, int | None, str | None, str | None]:
    try:
        if sub.is_symlink():
            return (True, None, os.readlink(sub), _symlink_target_digest(sub))
        if sub.is_dir():
            return (False, sub.lstat().st_mode & 0o7777, None, None)
    except OSError:
        pass
    return (False, None, None, None)


def _build_workspace_snapshot(
    root: Path,
    cwd_prefix: str,
    git_dirs: dict[str, Path],
    head_ref: str | None,
    head_sha: str | None,
    dirty: Collection[str],
    untracked: Collection[str],
    *,
    extra_paths: Collection[str] = (),
    backup_dir: Path | None = None,
    backup_cache: dict[str, tuple[str, int, int, int, int, int, int, bytes | None, str | None, Any]] | None = None,
    session_fds: dict[str, Any] | None = None,
    keep_backups: bool = False,
) -> WorkspaceSnapshot:
    base = head_sha or _EMPTY_TREE
    snap = WorkspaceSnapshot(
        root=root,
        cwd_prefix=cwd_prefix,
        base=base,
        entries={},
        git_dirs=git_dirs,
        dirty=frozenset(dirty),
        head_sha=head_sha,
        head_ref=head_ref,
        backup_dir=backup_dir,
        keep_backups=keep_backups,
    )
    for rel in dict.fromkeys([*dirty, *sorted(untracked), *sorted(extra_paths)]):
        if not rel or rel == "." or rel.startswith(_PLUGIN_SCRATCH_PREFIXES):
            continue
        try:
            entry = _capture(snap, root / rel, backup_cache=backup_cache, cache_key=f"wt:{rel}", session_fds=session_fds)
        except OSError:
            continue
        if entry is not None:
            snap.entries[rel] = entry
    own_dir = snap.git_dirs.get(".git")
    if own_dir is not None:
        snap.git_state = frozenset(n for n in _GIT_STATE_PATHS if (own_dir / n).exists() or (own_dir / n).is_symlink())
        idx_file = own_dir / "index"
        try:
            if idx_file.is_symlink():
                snap.index_tree = _git_text(root, "write-tree")
            elif idx_file.is_file():
                idx_st = idx_file.stat()
                snap.index_ns = (idx_st.st_atime_ns, idx_st.st_mtime_ns)
                cached_idx = backup_cache.get("git-index") if backup_cache is not None else None
                if (
                    cached_idx is not None
                    and (idx_st.st_dev, idx_st.st_ino, idx_st.st_size, idx_st.st_mtime_ns, idx_st.st_ctime_ns)
                    == cached_idx[1:6]
                    and (
                        (cached_idx[7] is not None and idx_file.read_bytes() == cached_idx[7])
                        or (
                            cached_idx[7] is None
                            and cached_idx[8] is not None
                            and cached_idx[9] is not None
                            and not getattr(cached_idx[9], "closed", True)
                            and _sha256_file(idx_file) == cached_idx[8]
                        )
                    )
                ):
                    snap.index_bytes = cached_idx[7]
                    snap.index_raw_sha = cached_idx[8]
                    if cached_idx[9] is not None:
                        snap.backup_fds["git-index"] = cached_idx[9]
                elif idx_st.st_size <= (8 << 20):
                    raw_idx = idx_file.read_bytes()
                    idx_sha = hashlib.sha256(raw_idx).hexdigest()
                    snap.index_bytes = raw_idx
                    snap.index_raw_sha = idx_sha
                    if backup_cache is not None:
                        old_idx = backup_cache.get("git-index")
                        if old_idx is not None and old_idx[9] is not None:
                            try:
                                old_idx[9].close()
                            except OSError:
                                pass
                            if session_fds is not None:
                                session_fds.pop("git-index", None)
                        backup_cache["git-index"] = (
                            "git-index",
                            idx_st.st_dev,
                            idx_st.st_ino,
                            idx_st.st_size,
                            idx_st.st_mtime_ns,
                            idx_st.st_ctime_ns,
                            0o600,
                            raw_idx,
                            idx_sha,
                            None,
                        )
                else:
                    idx_fh, idx_sha = _copy_to_unlinked_tmp_and_sha256(idx_file)
                    snap.index_raw_sha = idx_sha
                    snap.backup_fds["git-index"] = idx_fh
                    if session_fds is not None:
                        old_fh = session_fds.pop("git-index", None)
                        if old_fh is not None and old_fh is not idx_fh:
                            try:
                                old_fh.close()
                            except OSError:
                                pass
                        session_fds["git-index"] = idx_fh
                    if backup_cache is not None:
                        backup_cache["git-index"] = (
                            "git-index",
                            idx_st.st_dev,
                            idx_st.st_ino,
                            idx_st.st_size,
                            idx_st.st_mtime_ns,
                            idx_st.st_ctime_ns,
                            0o600,
                            None,
                            idx_sha,
                            idx_fh,
                        )
            elif not idx_file.exists():
                snap.index_was_absent = True
        except OSError:
            snap.index_tree = _git_text(root, "write-tree")
    for label, git_dir in snap.git_dirs.items():
        snap.git_dir_modes[label] = (
            (git_dir.lstat().st_mode & 0o7777) if (git_dir.is_dir() and not git_dir.is_symlink()) else None
        )
        snap.hooks_dir_state[label] = _capture_git_subdir_state(git_dir / "hooks")
        snap.info_dir_state[label] = _capture_git_subdir_state(git_dir / "info")
        hooks = _hook_names(git_dir)
        snap.hooks[label] = hooks
        meta_names = [
            n for n in _GIT_META_FILES
            if not (n.startswith("info/") and snap.info_dir_state[label][0])
        ]
        for name in (*meta_names, *(f"hooks/{h}" for h in sorted(hooks))):
            meta_key = f"{label}/{name}"
            try:
                entry = _capture(
                    snap,
                    git_dir / name,
                    backup_cache=backup_cache,
                    cache_key=f"meta:{meta_key}",
                    session_fds=session_fds,
                )
            except OSError:
                continue
            if entry is not None:
                snap.git_meta[meta_key] = entry
    return snap


def snapshot_workspace(cwd: Path, untracked: Collection[str] | None = None) -> WorkspaceSnapshot | None:
    """Back up the change set before a check; None when ``cwd`` is not inside a git checkout.

    ``untracked`` is the run-start baseline from ``untracked_baseline``; without it the current
    untracked files are used. Past 2,000 files or a 256 MB backup budget, files are recorded by size
    and mtime only: a change is still detected and reported, but cannot be undone.
    """
    layout = _repo_layout(cwd)
    if layout is None:
        return None
    root, git_dirs = layout
    try:
        prefix = Path(cwd).resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        prefix = ""
    own_dir = git_dirs.get(".git")
    status_info = _git_status_porcelain(root, own_dir, include_untracked=(untracked is None))
    if status_info is None:
        return None
    head_ref, head_sha, dirty, status_untracked = status_info
    eff_untracked = status_untracked if untracked is None else untracked
    return _build_workspace_snapshot(
        root,
        "" if prefix == "." else prefix,
        git_dirs,
        head_ref,
        head_sha,
        dirty,
        eff_untracked,
    )


def _entry_matches(
    path: Path,
    kind: str,
    payload: str | None,
    mode: int | None,
    *,
    inline_bytes: bytes | None = None,
    expected_sha: str | None = None,
) -> bool:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return kind == "absent"
    except OSError:
        return False
    try:
        if kind == "absent":
            return False
        if kind == "dir":
            return stat.S_ISDIR(st.st_mode)
        if kind == "symlink":
            return stat.S_ISLNK(st.st_mode) and os.readlink(path) == payload
        if not stat.S_ISREG(st.st_mode) or payload is None:
            return False
        if mode is not None and (st.st_mode & 0o7777) != mode:
            return False
        if kind == "stat":
            return f"{st.st_size}:{st.st_mtime_ns}" == payload
        if inline_bytes is not None:
            return st.st_size == len(inline_bytes) and path.read_bytes() == inline_bytes
        if expected_sha is not None:
            return _sha256_file(path) == expected_sha
        return _files_equal_bytes(path, payload)
    except OSError:
        return False


def _rmtree_writable(path: Path) -> None:
    """Remove directory tree ``path`` without following symlinks, making read-only subdirectories writable first."""
    if path.is_symlink():
        path.unlink(missing_ok=True)
        return
    if not path.is_dir():
        return
    try:
        os.chmod(path, (path.lstat().st_mode & 0o7777) | 0o700)
    except OSError:
        pass
    for dirpath, dirnames, filenames in os.walk(path, topdown=True, followlinks=False):
        dp = Path(dirpath)
        for d in dirnames:
            sub = dp / d
            if not sub.is_symlink() and sub.is_dir():
                try:
                    os.chmod(sub, (sub.lstat().st_mode & 0o7777) | 0o700)
                except OSError:
                    pass
        for f in filenames:
            fp = dp / f
            if not fp.is_symlink() and fp.is_file():
                try:
                    os.chmod(fp, 0o600)
                except OSError:
                    pass
    shutil.rmtree(path, ignore_errors=True)


def _prepare_parent_within(top: Path, target_parent: Path, *, create: bool, unlink_planted_links: bool = False) -> bool:
    """Ensure every directory from ``top`` to ``target_parent`` is a real owner-writable directory."""
    if top.is_symlink():
        return False
    try:
        top_res = top.resolve()
        try:
            rel_parts = target_parent.relative_to(top).parts
        except ValueError:
            rel_parts = target_parent.resolve().relative_to(top_res).parts
    except (OSError, ValueError):
        return False
    cur = top
    for step in (None, *rel_parts):
        if step is not None:
            cur = cur / step
        if cur.is_symlink():
            if step is not None and unlink_planted_links:
                cur.unlink(missing_ok=True)
                if create:
                    cur.mkdir(mode=0o755, exist_ok=True)
                continue
            return False
        if cur.exists() and not cur.is_dir():
            if step is not None and unlink_planted_links:
                try:
                    os.chmod(cur, 0o600)
                except OSError:
                    pass
                cur.unlink(missing_ok=True)
                if create:
                    cur.mkdir(mode=0o755, exist_ok=True)
                continue
            return False
        if cur.is_dir():
            if not cur.resolve().is_relative_to(top_res):
                return False
            try:
                dmode = cur.lstat().st_mode & 0o7777
                if (dmode & 0o700) != 0o700:
                    os.chmod(cur, dmode | 0o700)
            except OSError:
                return False
        elif create:
            try:
                cur.mkdir(mode=0o755, exist_ok=True)
            except OSError:
                return False
    return True


def _read_verified_backup_fd(backup_fh: Any, expected_sha: str | None) -> Path | None:
    """Materialize an unlinked backup ``TemporaryFile`` into a fresh temporary file after verifying SHA-256."""
    if backup_fh is None or getattr(backup_fh, "closed", True):
        return None
    fd_out, tmp_str = tempfile.mkstemp(prefix="cc-agy-restore-")
    tmp_path = Path(tmp_str)
    digest = hashlib.sha256()
    try:
        backup_fh.seek(0)
        with os.fdopen(fd_out, "wb") as f_out:
            while True:
                chunk = backup_fh.read(262_144)
                if not chunk:
                    break
                digest.update(chunk)
                f_out.write(chunk)
        if expected_sha is not None and digest.hexdigest() != expected_sha:
            tmp_path.unlink(missing_ok=True)
            return None
        return tmp_path
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        return None


def _restore_entry(
    path: Path,
    kind: str,
    payload: str | None,
    mode: int | None,
    *,
    top: Path | None = None,
    unlink_planted_links: bool = False,
    inline_bytes: bytes | None = None,
    expected_sha: str | None = None,
    backup_fh: Any = None,
) -> None:
    staged_tmp: Path | None = None
    try:
        if kind == "file" and inline_bytes is None:
            if backup_fh is not None:
                staged_tmp = _read_verified_backup_fd(backup_fh, expected_sha)
                if staged_tmp is None:
                    return
            elif payload is None or not Path(payload).is_file() or Path(payload).is_symlink():
                return
            elif expected_sha is not None and _sha256_file(Path(payload)) != expected_sha:
                return
        need_parent = kind in ("symlink", "file") and (payload is not None or inline_bytes is not None or staged_tmp is not None)
        if top is not None:
            if not _prepare_parent_within(top, path.parent, create=need_parent, unlink_planted_links=unlink_planted_links):
                return
        if path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_file():
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            path.unlink(missing_ok=True)
        elif path.is_dir():
            _rmtree_writable(path)
        elif path.exists():
            path.unlink(missing_ok=True)
        if kind == "symlink" and payload is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(payload, path)
        elif kind == "file" and (inline_bytes is not None or staged_tmp is not None or payload is not None):
            path.parent.mkdir(parents=True, exist_ok=True)
            if inline_bytes is not None:
                path.write_bytes(inline_bytes)
            elif staged_tmp is not None:
                shutil.copyfile(staged_tmp, path)
            elif payload is not None:
                shutil.copyfile(payload, path)
            if mode is not None:
                os.chmod(path, mode)
    finally:
        if staged_tmp is not None:
            staged_tmp.unlink(missing_ok=True)


_RUN_ARTIFACT_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis",
                                 ".tox", ".nox", "htmlcov", ".nyc_output", "coverage", ".cache"})


def _is_run_artifact(rel: str) -> bool:
    """Files test runs leave behind: generated artifacts, caches and reports, never source files."""
    path = PurePosixPath(rel)
    return (is_generated_artifact(rel) or bool(_RUN_ARTIFACT_DIRS & set(path.parts[:-1]))
            or path.suffix in (".log", ".pyc") or path.name.startswith(".coverage"))


def is_generated_artifact(rel: str) -> bool:
    """Lockfiles, test snapshots and coverage reports: rewritten by test runs without changing what
    the tests exercised."""
    path = PurePosixPath(rel)
    return path.name in _GENERATED_ARTIFACT_NAMES or "__snapshots__" in path.parts or path.suffix in (".snap", ".pyc")


def _restore_head(root: Path, snapshot: WorkspaceSnapshot) -> None:
    """Point HEAD (and the branch it named) back where they were when the snapshot was taken."""
    if snapshot.head_ref:
        _git(root, "symbolic-ref", "HEAD", snapshot.head_ref)
        if snapshot.head_sha:
            _git(root, "update-ref", snapshot.head_ref, snapshot.head_sha)
        else:
            _git(root, "update-ref", "-d", snapshot.head_ref)  # the branch was unborn
    elif snapshot.head_sha:
        _git(root, "update-ref", "--no-deref", "HEAD", snapshot.head_sha)


def _join_prefix(prefix: str, rel: str) -> str:
    return posixpath.normpath(f"{prefix}/{rel}" if prefix else rel)


def _head_matches_fast(snapshot: WorkspaceSnapshot) -> bool:
    own_dir = snapshot.git_dirs.get(".git")
    if own_dir is None:
        return False
    common_dir = snapshot.git_dirs.get(".git-common") or own_dir
    head_file = own_dir / "HEAD"
    try:
        if head_file.is_symlink() or not head_file.is_file():
            return False
        raw = head_file.read_text(encoding="utf-8", errors="surrogateescape").strip()
        if raw.startswith("ref:"):
            ref = raw[4:].strip()
            if ref != snapshot.head_ref or not ref.startswith("refs/") or ".." in ref:
                return False
            ref_file = common_dir / ref
            if ref_file.is_symlink():
                return False
            if ref_file.is_file():
                oid = ref_file.read_text(encoding="utf-8", errors="surrogateescape").strip()
                return bool(snapshot.head_sha and oid == snapshot.head_sha)
            if not ref_file.exists() and snapshot.head_sha is None:
                return not (common_dir / "packed-refs").exists()
            return False
        return snapshot.head_ref is None and snapshot.head_sha is not None and raw == snapshot.head_sha
    except OSError:
        return False


def _clean_tracked_intact(
    root: Path,
    clean_tracked: dict[str, tuple[int, int, int, int, int, int, str | None]],
) -> bool:
    checked_dirs: set[str] = set()
    for rel, (c_dev, c_ino, c_mode, c_size, c_mtime_ns, c_ctime_ns, c_extra) in clean_tracked.items():
        try:
            slash = rel.find("/")
            while slash != -1:
                d_rel = rel[:slash]
                if d_rel not in checked_dirs:
                    st_d = os.lstat(root / d_rel)
                    if not stat.S_ISDIR(st_d.st_mode):
                        return False
                    checked_dirs.add(d_rel)
                slash = rel.find("/", slash + 1)
            p = root / rel
            st = os.lstat(p)
            if c_dev == -1:
                if not stat.S_ISLNK(st.st_mode) or (st.st_mode & 0o7777) != c_mode or os.readlink(p) != c_extra:
                    return False
                continue
            if not stat.S_ISREG(st.st_mode):
                return False
            if (st.st_dev, st.st_ino, st.st_mode & 0o7777, st.st_size, st.st_mtime_ns, st.st_ctime_ns) != (
                c_dev,
                c_ino,
                c_mode,
                c_size,
                c_mtime_ns,
                c_ctime_ns,
            ):
                return False
            if c_extra is not None and _sha256_file(p) != c_extra:
                return False
        except OSError:
            return False
    return True


def check_side_effects(
    snapshot: WorkspaceSnapshot | None,
    *,
    allowed: Collection[str] = (),
    restore: bool = True,
    change_set: Collection[str] = (),
) -> tuple[list[str], list[str]]:
    """Compare the checkout with ``snapshot`` after one check; return ``(changed, untrusted)``."""
    if snapshot is None:
        return [], []
    try:
        root = snapshot.root
        allowed_set = {_join_prefix(snapshot.cwd_prefix, a) for a in allowed}
        # change_set may come relative to cwd or to the checkout root; accept either reading.
        declared = {_join_prefix(snapshot.cwd_prefix, c) for c in change_set} | {posixpath.normpath(c) for c in change_set}
        changed: list[str] = []
        untrusted: list[str] = []

        def counts_against(rel: str) -> bool:
            if rel in snapshot.dirty or rel in declared:
                return True
            return restore and not is_generated_artifact(rel)

        def flag(key: str) -> None:
            changed.append(key)
            untrusted.append(key)

        # 1. Git metadata first, before any git command reads a config or hook the check planted.
        _check_git_metadata(snapshot, flag)
        own_dir = snapshot.git_dirs.get(".git")

        if own_dir is not None:
            for name in _GIT_STATE_PATHS:
                path = own_dir / name
                exists = path.exists() or path.is_symlink()
                if exists == (name in snapshot.git_state):
                    continue
                flag(f".git/{name}")  # a merge, rebase or cherry-pick started or ended by the check
                if restore and exists:
                    try:
                        _rmtree_writable(path) if path.is_dir() and not path.is_symlink() else path.unlink()
                    except OSError:
                        pass

        idx_file = (own_dir / "index") if own_dir is not None else None
        index_untouched = False
        if idx_file is not None and not idx_file.is_symlink():
            try:
                if snapshot.index_was_absent and not idx_file.exists():
                    index_untouched = True
                elif snapshot.index_raw_sha is not None and idx_file.is_file():
                    index_untouched = _sha256_file(idx_file) == snapshot.index_raw_sha
            except OSError:
                index_untouched = False

        # 2. Fast-path clean checkouts without spawning git when HEAD, index, metadata, and clean tracked
        # files are all unchanged; otherwise run git status / rediff to inspect and restore.
        if (
            not allowed_set
            and not changed
            and index_untouched
            and snapshot.clean_tracked is not None
            and _head_matches_fast(snapshot)
            and _clean_tracked_intact(root, snapshot.clean_tracked)
        ):
            now_dirty = list(snapshot.dirty)
        else:
            status_info = _git_status_porcelain(root, own_dir, include_untracked=False)
            need_rediff = status_info is None
            if status_info is not None:
                head_ref_now, head_sha_now, status_dirty, _ = status_info
            else:
                head_ref_now = _git_text(root, "symbolic-ref", "-q", "HEAD") or None
                head_sha_now = _git_text(root, "rev-parse", "-q", "--verify", "HEAD^{commit}") or None
                status_dirty = []
            if (head_ref_now, head_sha_now) != (snapshot.head_ref, snapshot.head_sha):
                # A commit, reset or checkout made by a check rewrites history with the user's identity.
                flag(".git/HEAD")
                if restore:
                    _restore_head(root, snapshot)
                    need_rediff = True

            if not index_untouched:
                if restore and own_dir is not None:
                    (own_dir / "index.lock").unlink(missing_ok=True)
                idx_was_symlink = idx_file is not None and idx_file.is_symlink()
                if idx_was_symlink:
                    flag(".git/index")
                    if restore and idx_file is not None:
                        idx_file.unlink(missing_ok=True)
                orig_tree: str | None = None
                tmp_idx_path: Path | None = None
                try:
                    eff_idx_backup: str | None = None
                    idx_fh = snapshot.backup_fds.get("git-index")
                    if snapshot.index_bytes is not None:
                        fd_i, raw_tmp_idx = tempfile.mkstemp(prefix="cc-agy-idx-")
                        try:
                            os.write(fd_i, snapshot.index_bytes)
                        finally:
                            os.close(fd_i)
                        tmp_idx_path = Path(raw_tmp_idx)
                        eff_idx_backup = raw_tmp_idx
                    elif idx_fh is not None:
                        tmp_idx_path = _read_verified_backup_fd(idx_fh, snapshot.index_raw_sha)
                        if tmp_idx_path is not None:
                            eff_idx_backup = str(tmp_idx_path)
                    elif snapshot.index_backup and Path(snapshot.index_backup).is_file() and not Path(snapshot.index_backup).is_symlink():
                        if snapshot.index_raw_sha is None or _sha256_file(Path(snapshot.index_backup)) == snapshot.index_raw_sha:
                            eff_idx_backup = snapshot.index_backup

                    if snapshot.index_was_absent:
                        orig_tree = _EMPTY_TREE
                    elif eff_idx_backup:
                        orig_tree = _write_tree_for_index(root, index_file=eff_idx_backup)
                    elif snapshot.index_tree:
                        orig_tree = snapshot.index_tree
                    now_tree = _write_tree_for_index(root) if not idx_was_symlink else None
                    if orig_tree is not None:
                        if not idx_was_symlink and now_tree != orig_tree:
                            flag(".git/index")
                    else:
                        orig_dig = snapshot.index_digest or (
                            _index_digest(root, index_file=eff_idx_backup) if eff_idx_backup else None
                        )
                        if not idx_was_symlink and orig_dig is not None and _index_digest(root) != orig_dig:
                            flag(".git/index")
                    if restore and own_dir is not None and idx_file is not None:
                        need_rediff = True
                        try:
                            if snapshot.index_was_absent:
                                idx_file.unlink(missing_ok=True)
                            elif snapshot.index_bytes is not None:
                                idx_file.write_bytes(snapshot.index_bytes)
                                if snapshot.index_ns is not None:
                                    os.utime(idx_file, ns=snapshot.index_ns)
                            elif eff_idx_backup:
                                bak_idx = Path(eff_idx_backup)
                                bak_st = bak_idx.stat()
                                shutil.copyfile(bak_idx, idx_file)
                                os.utime(
                                    idx_file,
                                    ns=snapshot.index_ns or (bak_st.st_atime_ns, bak_st.st_mtime_ns),
                                )
                            elif snapshot.index_tree:
                                _git(root, "read-tree", snapshot.index_tree)
                        except OSError:
                            pass
                finally:
                    if tmp_idx_path is not None:
                        tmp_idx_path.unlink(missing_ok=True)

            # 3. Files. Compared after HEAD and the index are back, so a path that was only unstaged is not
            # reported. If git cannot list changes any more, the git state itself was tampered with: say
            # so, and still restore every backed-up file.
            if need_rediff:
                now_dirty_opt = _git_paths(root, "diff", "--name-only", "--no-renames", "-z", snapshot.base)
                if now_dirty_opt is None:
                    flag(".git")
                    now_dirty = []
                else:
                    now_dirty = now_dirty_opt
            else:
                now_dirty = status_dirty
        from_base: list[str] = []
        extra_planted: list[str] = []
        if snapshot.clean_tracked is not None:
            checked_parent_links: dict[str, bool] = {}
            for c_rel in snapshot.clean_tracked:
                p_dir = posixpath.dirname(c_rel)
                if not p_dir:
                    continue
                is_linked = checked_parent_links.get(p_dir)
                if is_linked is None:
                    is_linked = _symlink_in_path(root, p_dir)
                    checked_parent_links[p_dir] = is_linked
                if is_linked:
                    extra_planted.append(c_rel)
        for rel in sorted(set(now_dirty) | set(snapshot.entries) | set(extra_planted)):
            if rel.startswith(_PLUGIN_SCRATCH_PREFIXES):
                continue
            if rel in allowed_set and not _symlink_in_path(root, rel):
                try:
                    st_al = os.lstat(root / rel)
                except OSError:
                    st_al = None
                if st_al is not None and stat.S_ISREG(st_al.st_mode) and (st_al.st_mode & 0o600) == 0o600:
                    continue
            path = root / rel
            entry = snapshot.entries.get(rel)
            if entry is None:
                from_base.append(rel)
                continue
            kind, payload, mode = entry
            parent_rel = posixpath.dirname(rel)
            parent_has_symlink = bool(parent_rel and _symlink_in_path(root, parent_rel))
            inlined = snapshot.inline_bytes.get(payload) if (kind == "file" and payload is not None) else None
            exp_sha = snapshot.backup_hashes.get(payload) if (kind == "file" and payload is not None) else None
            if not parent_has_symlink and _entry_matches(
                path, kind, payload, mode, inline_bytes=inlined, expected_sha=exp_sha
            ):
                continue
            changed.append(rel)
            if counts_against(rel):
                untrusted.append(rel)
            if restore and kind not in ("stat", "dir"):
                bak_fh = snapshot.backup_fds.get(payload) if (kind == "file" and payload is not None) else None
                try:
                    _restore_entry(
                        path,
                        kind,
                        payload,
                        mode,
                        top=root,
                        unlink_planted_links=True,
                        inline_bytes=inlined,
                        expected_sha=exp_sha,
                        backup_fh=bak_fh,
                    )
                except OSError:
                    pass
        for start in range(0, len(from_base), 200):
            chunk = from_base[start:start + 200]
            changed.extend(chunk)
            untrusted.extend(p for p in chunk if counts_against(p))
            if restore and snapshot.base != _EMPTY_TREE:
                # These matched HEAD before the check; paths HEAD does not have were only staged by the
                # check and became untracked again when the index was restored above.
                in_base = _git_paths(root, "ls-tree", "-r", "-z", "--name-only", snapshot.base, "--", *chunk) or []
                if in_base:
                    for rel_base in in_base:
                        _prepare_parent_within(root, (root / rel_base).parent, create=True, unlink_planted_links=True)
                        base_target = root / rel_base
                        if base_target.is_symlink():
                            base_target.unlink(missing_ok=True)
                        elif base_target.is_dir():
                            _rmtree_writable(base_target)
                    _git(root, "checkout", snapshot.base, "--", *in_base)
        return sorted(set(changed)), sorted(set(untrusted))
    finally:
        if not snapshot.keep_backups:
            _close_backup_fds(snapshot.backup_fds)
            if snapshot.backup_dir is not None:
                _rmtree_writable(snapshot.backup_dir)


def _check_git_metadata(snapshot: WorkspaceSnapshot, flag: Callable[[str], None]) -> None:
    """Undo planted git config, attributes/exclude, and hooks (always, even in report-only mode)."""
    for label, git_dir in snapshot.git_dirs.items():
        if git_dir.is_symlink():
            flag(label)
            continue
        was_git_mode = snapshot.git_dir_modes.get(label)
        if git_dir.is_dir():
            now_git_mode = git_dir.lstat().st_mode & 0o7777
            if was_git_mode is not None and now_git_mode != was_git_mode:
                flag(label)
            try:
                os.chmod(git_dir, (was_git_mode or 0o755) | 0o700)
            except OSError:
                pass
        for sub_name, state_map in (("hooks", snapshot.hooks_dir_state), ("info", snapshot.info_dir_state)):
            sub_dir = git_dir / sub_name
            was_link, was_mode, orig_target, orig_digest = state_map.get(label, (False, None, None, None))
            if was_link:
                same_link = False
                if sub_dir.is_symlink():
                    try:
                        same_link = os.readlink(sub_dir) == orig_target
                    except OSError:
                        same_link = False
                if not same_link:
                    flag(f"{label}/{sub_name}")
                    if sub_name == "hooks" and sub_dir.is_dir() and not sub_dir.is_symlink():
                        try:
                            os.chmod(sub_dir, 0o700)
                        except OSError:
                            pass
                        for planted_h in sorted(_hook_names(git_dir)):
                            flag(f"{label}/hooks/{planted_h}")
                    if sub_dir.is_symlink():
                        try:
                            sub_dir.unlink()
                        except OSError:
                            pass
                    elif sub_dir.is_dir():
                        _rmtree_writable(sub_dir)
                    elif sub_dir.exists():
                        try:
                            os.chmod(sub_dir, 0o600)
                        except OSError:
                            pass
                        sub_dir.unlink(missing_ok=True)
                    if orig_target is not None:
                        try:
                            os.symlink(orig_target, sub_dir)
                        except OSError:
                            pass
                else:
                    now_digest = _symlink_target_digest(sub_dir)
                    if orig_digest is None or now_digest is None or now_digest != orig_digest:
                        flag(f"{label}/{sub_name}")
                continue
            if sub_dir.is_symlink():
                flag(f"{label}/{sub_name}")
                try:
                    sub_dir.unlink()  # drop the link itself, never anything inside its target
                    if was_mode is not None:
                        sub_dir.mkdir(mode=was_mode | 0o700)
                except OSError:
                    pass
            elif sub_dir.exists() and not sub_dir.is_dir() and not sub_dir.is_symlink():
                flag(f"{label}/{sub_name}")
                try:
                    os.chmod(sub_dir, 0o600)
                    sub_dir.unlink()
                    if was_mode is not None:
                        sub_dir.mkdir(mode=was_mode | 0o700)
                except OSError:
                    pass
            elif sub_dir.is_dir() and not sub_dir.is_symlink() and was_mode is None:
                flag(f"{label}/{sub_name}")
                try:
                    os.chmod(sub_dir, 0o700)
                except OSError:
                    pass
            elif sub_dir.is_dir() and not sub_dir.is_symlink() and was_mode is not None:
                now_mode = sub_dir.lstat().st_mode & 0o7777
                if now_mode != was_mode:
                    flag(f"{label}/{sub_name}")
                try:
                    os.chmod(sub_dir, was_mode | 0o700)
                except OSError:
                    pass
    for key, (kind, payload, mode) in snapshot.git_meta.items():
        label, name = key.split("/", 1)
        git_dir = snapshot.git_dirs[label]
        if git_dir.is_symlink():
            continue
        meta_path = git_dir / name
        meta_inlined = snapshot.inline_bytes.get(payload) if (kind == "file" and payload is not None) else None
        meta_sha = snapshot.backup_hashes.get(payload) if (kind == "file" and payload is not None) else None
        if _entry_matches(meta_path, kind, payload, mode, inline_bytes=meta_inlined, expected_sha=meta_sha):
            continue
        flag(key)
        if kind != "stat":
            meta_fh = snapshot.backup_fds.get(payload) if (kind == "file" and payload is not None) else None
            try:
                _restore_entry(
                    meta_path,
                    kind,
                    payload,
                    mode,
                    top=git_dir,
                    unlink_planted_links=False,
                    inline_bytes=meta_inlined,
                    expected_sha=meta_sha,
                    backup_fh=meta_fh,
                )
            except OSError:
                pass
    for label, git_dir in snapshot.git_dirs.items():
        if git_dir.is_symlink():
            continue
        info_was_link = snapshot.info_dir_state.get(label, (False, None, None, None))[0]
        hooks_was_link = snapshot.hooks_dir_state.get(label, (False, None, None, None))[0]
        for name in _GIT_META_FILES:
            if name.startswith("info/") and info_was_link:
                continue
            key = f"{label}/{name}"
            meta_path = git_dir / name
            if key not in snapshot.git_meta and (meta_path.exists() or meta_path.is_symlink()):
                flag(key)
                try:
                    _restore_entry(meta_path, "absent", None, None, top=git_dir, unlink_planted_links=False)
                except OSError:
                    pass
        if not hooks_was_link:
            for hook in sorted(_hook_names(git_dir) - snapshot.hooks.get(label, frozenset())):
                flag(f"{label}/hooks/{hook}")
                hook_path = git_dir / "hooks" / hook
                try:
                    _restore_entry(hook_path, "absent", None, None, top=git_dir, unlink_planted_links=False)
                except OSError:
                    pass
        for sub_name, state_map in (("hooks", snapshot.hooks_dir_state), ("info", snapshot.info_dir_state)):
            was_link, was_mode, _orig_target, _orig_digest = state_map.get(label, (False, None, None, None))
            if was_link:
                continue
            sub_dir = git_dir / sub_name
            if was_mode is None and (sub_dir.exists() or sub_dir.is_symlink()):
                _rmtree_writable(sub_dir)
            elif was_mode is not None and sub_dir.is_dir() and not sub_dir.is_symlink():
                try:
                    os.chmod(sub_dir, was_mode)
                except OSError:
                    pass
        was_git_mode = snapshot.git_dir_modes.get(label)
        if was_git_mode is not None and git_dir.is_dir() and not git_dir.is_symlink():
            try:
                os.chmod(git_dir, was_git_mode)
            except OSError:
                pass


def side_effect_note(untrusted: list[str], *, restored: bool) -> str:
    shown = ", ".join(untrusted[:5]) + (f" and {len(untrusted) - 5} more" if len(untrusted) > 5 else "")
    action = "they were restored" if restored else "they were left in place"
    return (
        f"\n[cc-agy] Files changed while the check ran ({shown}); {action}. The check does not count as "
        "passing, because it did not test the files that will ship."
    )


# ------------------------------------------------------------------------------------------------
# Vacuous test runs (exit 0, but nothing executed)
# ------------------------------------------------------------------------------------------------

_SKIPPED_ONLY_RE = re.compile(
    r"\b(?:[1-9]\d*\s+skipped|skipped\s*[=:]\s*[1-9]\d*|Skipped:\s*[1-9]\d*|[1-9]\d*\s+ignored)\b",
    re.I,
)
_ZERO_TESTS_RE = re.compile(r"\b(?:ran\s+0\s+tests|collected\s+0\s+items|no\s+tests\s+ran)\b", re.I)
# Signs that at least one test executed, from runners whose summary the count parser cannot read.
_RAN_SOME_TEST_RES = (
    re.compile(r"\b[1-9]\d*\s+(?:passed|passing|succeeded)\b", re.I),  # pytest, jest, vitest, mocha, cargo
    re.compile(r"\bPassed:\s*[1-9]"),                                  # dotnet test (VSTest), stestr
    re.compile(r"^ok\s+\S+", re.M),                                    # go test
    re.compile(r"\bOK \([1-9]\d* tests?\b"),                           # PHPUnit, all green
)
_PHPUNIT_SUMMARY_RE = re.compile(r"\bTests:\s*(\d+),.*?\bSkipped:\s*(\d+)", re.S)


def _ran_some_test(text: str) -> bool:
    if any(rx.search(text) for rx in _RAN_SOME_TEST_RES):
        return True
    phpunit = _PHPUNIT_SUMMARY_RE.search(text)
    return bool(phpunit and int(phpunit.group(1)) > int(phpunit.group(2)))


_CARGO_RUNNING_RE = re.compile(r"^running (\d+) tests?$", re.M)
_GO_PACKAGE_LINE_RE = re.compile(r"^(?:ok|\?)\s+\S+.*$", re.M)
_GO_TEST_RESULT_RE = re.compile(r"^\s*--- (?:PASS|FAIL)", re.M)


def _go_or_cargo_ran_nothing(text: str) -> bool:
    """cargo test where every test binary ran 0 tests, or go test where every package reported
    [no test files] or [no tests to run] (go still prints PASS and ok for those)."""
    cargo = _CARGO_RUNNING_RE.findall(text)
    if cargo and all(int(n) == 0 for n in cargo):
        return True
    go_lines = _GO_PACKAGE_LINE_RE.findall(text)
    return bool(
        go_lines
        and all("[no test files]" in line or "[no tests to run]" in line for line in go_lines)
        and not _GO_TEST_RESULT_RE.search(text)
    )


def vacuous_test_run(output: str, returncode: int | None, timed_out: bool, counts: Mapping[str, Any]) -> bool:
    """True when a test command exited cleanly but executed no test: zero collected, or all skipped.

    Go and cargo runs that report no tests are vacuous even though they print PASS or ok. The
    skipped-only rule applies only when nothing in the output shows a test ran, so passing PHPUnit,
    dotnet test or go test runs that also skip a test are not mistaken for empty runs.
    """
    if timed_out or returncode not in (0, 5):
        return False
    text = output or ""
    if returncode == 0 and _go_or_cargo_ran_nothing(text):
        return True
    if any(int(counts.get(key) or 0) for key in ("passed", "failed", "errors")):
        return False
    if _ZERO_TESTS_RE.search(text):
        return True
    return returncode == 0 and bool(_SKIPPED_ONLY_RE.search(text)) and not _ran_some_test(text)


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


_PROC_TICKS_LOCK = threading.Lock()
_PROC_START_TICKS_CACHE: dict[tuple[int, int, int], int] = {}


def _proc_start_ticks(pid: int, st_ino: int, st_ctime_ns: int) -> int | None:
    """Return Linux ``/proc/<pid>/stat`` monotonic boot start ticks (field 22), cached per procfs inode."""
    key = (pid, st_ino, st_ctime_ns)
    with _PROC_TICKS_LOCK:
        cached = _PROC_START_TICKS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read(4_096)
        rp = data.rfind(b")")
        if rp < 0:
            return None
        val = int(data[rp + 2:].split()[19])
    except (OSError, IndexError, ValueError):
        return None
    with _PROC_TICKS_LOCK:
        if len(_PROC_START_TICKS_CACHE) > 8_192:
            _PROC_START_TICKS_CACHE.clear()
        _PROC_START_TICKS_CACHE[key] = val
    return val


def _environ_has_entry(env_bytes: bytes, entry: bytes) -> bool:
    idx = env_bytes.find(entry)
    while idx >= 0:
        end = idx + len(entry)
        if (idx == 0 or env_bytes[idx - 1] == 0) and (end == len(env_bytes) or env_bytes[end] == 0):
            return True
        idx = env_bytes.find(entry, idx + 1)
    return False


def _kill_check_descendants(
    pipe_dev: int,
    pipe_ino: int,
    run_id: str,
    *,
    min_start_ticks: int | None = None,
) -> None:
    """Kill detached ``setsid`` descendants on Linux via either the per-check env marker or open stdout pipe."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return
    my_pid = os.getpid()
    my_pgid = os.getpgrp()
    my_uid = os.getuid()
    pipe_link = f"pipe:[{pipe_ino}]"
    cookie_entry = f"_AGY_CHECK_RUN_ID={run_id}".encode("ascii") if run_id else b""
    for _pass in range(2):
        killed_any = False
        try:
            entries = os.listdir(proc_root)
        except OSError:
            return
        for name in entries:
            if not name.isdigit():
                continue
            pid = int(name)
            if pid == my_pid:
                continue
            p_dir = proc_root / name
            try:
                pst = os.lstat(p_dir)
            except OSError:
                continue
            if pst.st_uid != my_uid:
                continue
            if min_start_ticks is not None:
                st_ticks = _proc_start_ticks(pid, pst.st_ino, pst.st_ctime_ns)
                if st_ticks is not None and st_ticks < min_start_ticks:
                    continue
            matched = False
            if cookie_entry:
                try:
                    with open(p_dir / "environ", "rb") as efh:
                        matched = _environ_has_entry(efh.read(2_097_152), cookie_entry)
                except OSError:
                    pass
            if not matched:
                fd_dir = p_dir / "fd"
                try:
                    fd_names = os.listdir(fd_dir)
                except OSError:
                    fd_names = []
                for fd_name in fd_names:
                    fd_path = fd_dir / fd_name
                    try:
                        if os.readlink(fd_path) == pipe_link:
                            st = os.stat(fd_path)
                            if st.st_dev == pipe_dev and st.st_ino == pipe_ino:
                                matched = True
                                break
                    except OSError:
                        continue
            if matched:
                killed_any = True
                try:
                    pgid = os.getpgid(pid)
                    if pgid > 1 and pgid != my_pgid:
                        os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if not killed_any:
            break


def _append_bounded_chunk(
    chunk: str,
    *,
    log_handle: Any,
    tail: deque[str],
    kept_ref: list[int],
    max_chars: int,
    io_lock: threading.Lock,
    io_closed: list[bool],
) -> None:
    if not chunk:
        return
    with io_lock:
        if io_closed[0]:
            return
        if log_handle is not None and not getattr(log_handle, "closed", False):
            log_handle.write(chunk)
        tail.append(chunk)
        kept_ref[0] += len(chunk)
        while kept_ref[0] - len(tail[0]) > max_chars and len(tail) > 1:
            kept_ref[0] -= len(tail.popleft())
        if len(tail) == 1 and len(tail[0]) > max_chars * 2:
            only = tail.popleft()
            trimmed = only[-max_chars:]
            tail.append(trimmed)
            kept_ref[0] = len(trimmed)


def _drain(stream: Any, log_handle: Any, tail: deque[str], max_chars: int = TAIL_CHARS) -> None:
    """Copy a process's output to the log file while keeping only a bounded tail in memory."""
    kept_ref = [0]
    io_lock = threading.Lock()
    io_closed = [False]
    for chunk in iter(lambda: stream.read(8192), ""):
        _append_bounded_chunk(
            chunk,
            log_handle=log_handle,
            tail=tail,
            kept_ref=kept_ref,
            max_chars=max_chars,
            io_lock=io_lock,
            io_closed=io_closed,
        )


_SUBREAPER_C_SOURCE = r"""
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef PR_SET_CHILD_SUBREAPER
#define PR_SET_CHILD_SUBREAPER 36
#endif

static volatile sig_atomic_t got_term = 0;
static int sig_pipe[2] = {-1, -1};

static void on_sig(int sig) {
    int saved_errno = errno;
    if (sig == SIGTERM || sig == SIGINT) {
        got_term = 1;
    }
    if (sig_pipe[1] >= 0) {
        char b = 1;
        ssize_t unused = write(sig_pipe[1], &b, 1);
        (void)unused;
    }
    errno = saved_errno;
}

static void kill_direct_children(pid_t my_pid, pid_t my_pgrp) {
    DIR *d = opendir("/proc");
    if (!d) return;
    struct dirent *de;
    char path[128];
    char buf[4096];
    pid_t pids[4096];
    pid_t pgrps[4096];
    size_t count = 0;
    while ((de = readdir(d)) != NULL) {
        const char *name = de->d_name;
        if (!isdigit((unsigned char)name[0])) continue;
        char *endp = NULL;
        long pid_l = strtol(name, &endp, 10);
        if (!endp || *endp != '\0' || pid_l <= 1 || (pid_t)pid_l == my_pid) continue;
        snprintf(path, sizeof(path), "/proc/%ld/stat", pid_l);
        int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) continue;
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        close(fd);
        if (n <= 0) continue;
        buf[n] = '\0';
        char *rp = strrchr(buf, ')');
        if (!rp || rp[1] != ' ') continue;
        char state = '\0';
        int ppid = -1, pgrp = -1;
        if (sscanf(rp + 2, "%c %d %d", &state, &ppid, &pgrp) != 3) continue;
        if ((pid_t)ppid == my_pid && count < 4096) {
            pids[count] = (pid_t)pid_l;
            pgrps[count] = (pid_t)pgrp;
            count++;
        }
    }
    closedir(d);
    for (size_t i = 0; i < count; i++) {
        if (pgrps[i] > 1 && pgrps[i] != my_pgrp) kill(-pgrps[i], SIGSTOP);
        kill(pids[i], SIGSTOP);
    }
    for (size_t i = 0; i < count; i++) {
        if (pgrps[i] > 1 && pgrps[i] != my_pgrp) kill(-pgrps[i], SIGKILL);
        kill(pids[i], SIGKILL);
    }
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--probe-subreaper") == 0) {
        return prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0 ? 0 : 1;
    }
    if (argc < 2) return 2;
    if (prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0) return 125;
    if (pipe2(sig_pipe, O_NONBLOCK | O_CLOEXEC) != 0) return 125;

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_sig;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGCHLD, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);

    pid_t my_pid = getpid();
    pid_t my_pgrp = getpgrp();

    pid_t child = fork();
    if (child < 0) return 126;
    if (child == 0) {
        setpgid(0, 0);
        signal(SIGCHLD, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
        signal(SIGINT, SIG_DFL);
        execvp(argv[1], &argv[1]);
        _exit(127);
    }
    (void)setpgid(child, child);

    int dn = open("/dev/null", O_WRONLY | O_CLOEXEC);
    if (dn >= 0) {
        if (dn != STDOUT_FILENO) (void)dup2(dn, STDOUT_FILENO);
        if (dn != STDERR_FILENO) (void)dup2(dn, STDERR_FILENO);
        if (dn > STDERR_FILENO) close(dn);
    }

    int main_status = 0;
    int have_main = 0;
    while (!have_main && !got_term) {
        char drain[64];
        while (read(sig_pipe[0], drain, sizeof(drain)) > 0) {}
        int no_children = 0;
        for (;;) {
            int st = 0;
            pid_t w = waitpid(-1, &st, WNOHANG);
            if (w > 0) {
                if (w == child) {
                    main_status = st;
                    have_main = 1;
                }
                continue;
            }
            if (w < 0 && errno == ECHILD) {
                no_children = 1;
            }
            break;
        }
        if (have_main || got_term || no_children) break;
        struct pollfd pfd;
        pfd.fd = sig_pipe[0];
        pfd.events = POLLIN;
        pfd.revents = 0;
        (void)poll(&pfd, 1, 100);
    }

    kill(-child, SIGSTOP);
    kill(child, SIGSTOP);
    kill(-child, SIGKILL);
    kill(child, SIGKILL);

    for (int iter = 0; iter < 2000; iter++) {
        int reaped_all = 0;
        for (;;) {
            int st = 0;
            pid_t w = waitpid(-1, &st, WNOHANG);
            if (w > 0) {
                if (w == child && !have_main) {
                    main_status = st;
                    have_main = 1;
                }
                continue;
            }
            if (w < 0 && errno == ECHILD) {
                reaped_all = 1;
            }
            break;
        }
        if (reaped_all) break;
        kill_direct_children(my_pid, my_pgrp);
        struct timespec ts = {0, 1000000L};
        nanosleep(&ts, NULL);
    }

    if (got_term && !have_main) return 128 + SIGTERM;
    if (WIFEXITED(main_status)) return WEXITSTATUS(main_status);
    if (WIFSIGNALED(main_status)) return 128 + WTERMSIG(main_status);
    return 1;
}
"""

_PY_SUBREAPER_SCRIPT = r"""
import ctypes, errno, os, select, signal, sys

libc = ctypes.CDLL(None, use_errno=True)
if len(sys.argv) == 2 and sys.argv[1] == "--probe-subreaper":
    sys.exit(0 if libc.prctl(36, 1, 0, 0, 0) == 0 else 1)
if len(sys.argv) < 2 or libc.prctl(36, 1, 0, 0, 0) != 0:
    sys.exit(125)

r_fd, w_fd = os.pipe()
os.set_blocking(r_fd, False)
os.set_blocking(w_fd, False)
signal.set_wakeup_fd(w_fd, warn_on_full_buffer=False)
got_term = False

def _on_sig(sig, _frame):
    global got_term
    if sig in (signal.SIGTERM, signal.SIGINT):
        got_term = True

signal.signal(signal.SIGCHLD, _on_sig)
signal.signal(signal.SIGTERM, _on_sig)
signal.signal(signal.SIGINT, _on_sig)

my_pid = os.getpid()
my_pgrp = os.getpgrp()
child = os.fork()
if child == 0:
    os.setpgid(0, 0)
    signal.set_wakeup_fd(-1)
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        os.execvp(sys.argv[1], sys.argv[1:])
    except OSError:
        os._exit(127)
try:
    os.setpgid(child, child)
except OSError:
    pass
try:
    dn = os.open("/dev/null", os.O_WRONLY)
    os.dup2(dn, 1)
    os.dup2(dn, 2)
    if dn > 2:
        os.close(dn)
except OSError:
    pass

main_status = 0
have_main = False
while not have_main and not got_term:
    try:
        while os.read(r_fd, 64):
            pass
    except BlockingIOError:
        pass
    no_children = False
    while True:
        try:
            w, st = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            no_children = True
            break
        except InterruptedError:
            continue
        if w == 0:
            break
        if w == child:
            main_status = st
            have_main = True
    if have_main or got_term or no_children:
        break
    try:
        select.select([r_fd], [], [], 0.05)
    except InterruptedError:
        pass

def _kill_direct():
    targets = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        if pid <= 1 or pid == my_pid:
            continue
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                data = fh.read(4096)
            rp = data.rfind(b")")
            if rp < 0:
                continue
            fields = data[rp + 2:].split()
            if int(fields[1]) == my_pid:
                targets.append((pid, int(fields[2])))
        except (OSError, IndexError, ValueError):
            continue
    for pid, pgrp in targets:
        for sig in (signal.SIGSTOP,):
            if pgrp > 1 and pgrp != my_pgrp:
                try:
                    os.killpg(pgrp, sig)
                except OSError:
                    pass
            try:
                os.kill(pid, sig)
            except OSError:
                pass
    for pid, pgrp in targets:
        if pgrp > 1 and pgrp != my_pgrp:
            try:
                os.killpg(pgrp, signal.SIGKILL)
            except OSError:
                pass
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

for sig in (signal.SIGSTOP, signal.SIGKILL):
    try:
        os.killpg(child, sig)
    except OSError:
        pass
    try:
        os.kill(child, sig)
    except OSError:
        pass

for _ in range(2000):
    reaped_all = False
    while True:
        try:
            w, st = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            reaped_all = True
            break
        except InterruptedError:
            continue
        if w == 0:
            break
        if w == child and not have_main:
            main_status = st
            have_main = True
    if reaped_all:
        break
    _kill_direct()
    select.select([], [], [], 0.001)

if got_term and not have_main:
    sys.exit(128 + signal.SIGTERM)
if os.WIFEXITED(main_status):
    sys.exit(os.WEXITSTATUS(main_status))
if os.WIFSIGNALED(main_status):
    sys.exit(128 + os.WTERMSIG(main_status))
sys.exit(1)
"""

_SUBREAPER_LOCK = threading.Lock()
_SUBREAPER_BIN_INFO: tuple[str, int, int, int, int, int, str] | None = None
_SUBREAPER_C_DISABLED: bool = False
_PY_SUBREAPER_AVAILABLE: bool | None = None
_FORCE_PYTHON_SUBREAPER: bool = False


def _popen_sync_rc(argv: Sequence[str], *, timeout: float = 10.0) -> int | None:
    try:
        p = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            return None
    except OSError:
        return None


def _ensure_c_subreaper_bin() -> str | None:
    global _SUBREAPER_BIN_INFO, _SUBREAPER_C_DISABLED
    if _FORCE_PYTHON_SUBREAPER or sys.platform != "linux":
        return None
    with _SUBREAPER_LOCK:
        if _SUBREAPER_BIN_INFO is not None:
            bpath, bdev, bino, bsize, bmtime, bctime, bsha = _SUBREAPER_BIN_INFO
            try:
                st = os.lstat(bpath)
                if (
                    stat.S_ISREG(st.st_mode)
                    and (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
                    == (bdev, bino, bsize, bmtime, bctime)
                    and _sha256_file(Path(bpath)) == bsha
                ):
                    return bpath
            except OSError:
                pass
            _SUBREAPER_BIN_INFO = None
        if _SUBREAPER_C_DISABLED:
            return None
        compiler: str | None = None
        for cand in ("cc", "gcc", "clang"):
            found = shutil.which(cand)
            if found:
                compiler = found
                break
        if compiler is None:
            _SUBREAPER_C_DISABLED = True
            return None
        sr_dir = Path(tempfile.mkdtemp(prefix="cc-agy-subreaper-"))
        atexit.register(_rmtree_writable, sr_dir)
        src_file = sr_dir / "subreaper.c"
        bin_file = sr_dir / "subreaper"
        try:
            src_file.write_text(_SUBREAPER_C_SOURCE, encoding="utf-8")
            rc = _popen_sync_rc([compiler, "-O2", str(src_file), "-o", str(bin_file)], timeout=15.0)
            src_file.unlink(missing_ok=True)
            if rc != 0 or not bin_file.is_file():
                _rmtree_writable(sr_dir)
                _SUBREAPER_C_DISABLED = True
                return None
            os.chmod(bin_file, 0o500)
            os.chmod(sr_dir, 0o500)
            if _popen_sync_rc([str(bin_file), "--probe-subreaper"], timeout=3.0) != 0:
                _rmtree_writable(sr_dir)
                _SUBREAPER_C_DISABLED = True
                return None
            st = os.lstat(bin_file)
            bsha = _sha256_file(bin_file)
            _SUBREAPER_BIN_INFO = (
                str(bin_file),
                st.st_dev,
                st.st_ino,
                st.st_size,
                st.st_mtime_ns,
                st.st_ctime_ns,
                bsha,
            )
            return str(bin_file)
        except OSError:
            _rmtree_writable(sr_dir)
            _SUBREAPER_C_DISABLED = True
            return None


def _python_subreaper_ok() -> bool:
    global _PY_SUBREAPER_AVAILABLE
    if sys.platform != "linux":
        return False
    with _SUBREAPER_LOCK:
        if _PY_SUBREAPER_AVAILABLE is None:
            _PY_SUBREAPER_AVAILABLE = (
                _popen_sync_rc([sys.executable, "-I", "-S", "-c", _PY_SUBREAPER_SCRIPT, "--probe-subreaper"], timeout=5.0)
                == 0
            )
        return _PY_SUBREAPER_AVAILABLE


def _build_check_launch_argv(cmd: str, proc_env: Mapping[str, str]) -> tuple[list[str], bool]:
    bash_bin = shutil.which("bash", path=proc_env.get("PATH")) or shutil.which("bash") or "/bin/bash"
    c_bin = _ensure_c_subreaper_bin()
    if c_bin is not None:
        return [c_bin, bash_bin, "-c", cmd], True
    if _python_subreaper_ok():
        return [sys.executable, "-I", "-S", "-c", _PY_SUBREAPER_SCRIPT, bash_bin, "-c", cmd], True
    return [bash_bin, "-c", cmd], False


def execute_check_process(
    cmd: str,
    cwd: Path,
    env: Mapping[str, str],
    *,
    timeout: float,
    log_handle: Any = None,
    max_tail_chars: int = TAIL_CHARS,
) -> tuple[int | None, bool, str, float]:
    """Run ``bash -c cmd`` under a dedicated session + Linux ``PR_SET_CHILD_SUBREAPER`` supervisor,
    stream output into a bounded tail, and kill every descendant process (even clean-env ``setsid``
    daemons with redirected stdio) before returning."""
    started = time.time()
    tail: deque[str] = deque()
    kept_ref = [0]
    io_lock = threading.Lock()
    io_closed = [False]
    proc_exited = threading.Event()
    run_id = uuid.uuid4().hex
    proc_env = dict(env)
    proc_env["_AGY_CHECK_RUN_ID"] = run_id
    launch_argv, has_subreaper = _build_check_launch_argv(cmd, proc_env)
    proc = subprocess.Popen(
        launch_argv,
        cwd=str(cwd),
        env=proc_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        start_new_session=True,
    )
    min_start_ticks: int | None = None
    if not has_subreaper:
        try:
            proc_st = os.lstat(f"/proc/{proc.pid}")
            min_start_ticks = _proc_start_ticks(proc.pid, proc_st.st_ino, proc_st.st_ctime_ns)
        except OSError:
            min_start_ticks = None
    assert proc.stdout is not None
    raw_fd = proc.stdout.fileno()
    pipe_st = os.fstat(raw_fd)
    pipe_dev, pipe_ino = pipe_st.st_dev, pipe_st.st_ino
    os.set_blocking(raw_fd, False)

    def _drain_fd() -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        sel = selectors.DefaultSelector()
        eof = False
        leaked_writer = False
        try:
            sel.register(raw_fd, selectors.EVENT_READ)
            while not eof:
                exited = proc_exited.is_set()
                if not exited:
                    sel.select(timeout=0.05)
                    exited = proc_exited.is_set()
                drained_post_exit = 0
                while True:
                    try:
                        data = os.read(raw_fd, 65_536)
                    except BlockingIOError:
                        if exited:
                            leaked_writer = True
                        break
                    except OSError:
                        eof = True
                        break
                    if not data:
                        eof = True
                        break
                    text_part = decoder.decode(data, final=False)
                    _append_bounded_chunk(
                        text_part,
                        log_handle=log_handle,
                        tail=tail,
                        kept_ref=kept_ref,
                        max_chars=max_tail_chars,
                        io_lock=io_lock,
                        io_closed=io_closed,
                    )
                    if exited:
                        drained_post_exit += len(data)
                        if drained_post_exit >= 524_288:
                            leaked_writer = True
                            break
                if exited:
                    break
            if leaked_writer:
                _kill_check_descendants(pipe_dev, pipe_ino, run_id, min_start_ticks=None)
            rem_text = decoder.decode(b"", final=True)
            if rem_text:
                _append_bounded_chunk(
                    rem_text,
                    log_handle=log_handle,
                    tail=tail,
                    kept_ref=kept_ref,
                    max_chars=max_tail_chars,
                    io_lock=io_lock,
                    io_closed=io_closed,
                )
        finally:
            try:
                sel.close()
            except OSError:
                pass

    reader = threading.Thread(target=_drain_fd, daemon=True)
    reader.start()
    timed_out = False
    try:
        try:
            proc.wait(timeout=max(0.1, float(timeout)))
        except subprocess.TimeoutExpired:
            timed_out = True
    finally:
        subreaper_cleaned = False
        if has_subreaper:
            if proc.poll() is not None:
                subreaper_cleaned = True
            else:
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    proc.wait(timeout=3.0)
                    subreaper_cleaned = True
                except subprocess.TimeoutExpired:
                    subreaper_cleaned = False
        if not subreaper_cleaned:
            _kill_group(proc.pid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_group(proc.pid)
                proc.wait()
            _kill_check_descendants(pipe_dev, pipe_ino, run_id, min_start_ticks=min_start_ticks)
        proc_exited.set()
    reader.join(timeout=2.0)
    with io_lock:
        io_closed[0] = True
        tail_str = "".join(tail)
    if not reader.is_alive():
        try:
            proc.stdout.close()
        except OSError:
            pass
    duration = time.time() - started
    return (None if timed_out else proc.returncode), timed_out, tail_str, duration


class _SnapshotSession:
    """Reusable workspace snapshot session across sequential check commands in ``cwd``."""

    def __init__(self, cwd: Path, *, extra_paths: Collection[str] = ()) -> None:
        self.cwd = cwd
        self.root: Path | None = None
        self.cwd_prefix: str = ""
        self.git_dirs: dict[str, Path] = {}
        self.baseline: frozenset[str] | None = None
        self.backup_dir: Path | None = None
        self.backup_cache: dict[str, tuple[str, int, int, int, int, int, int, bytes | None, str | None, Any]] = {}
        self.backup_fds: dict[str, Any] = {}
        self._extra_root_paths: tuple[str, ...] = ()
        self._initial_status: tuple[str | None, str | None, list[str], list[str]] | None = None
        self._clean_tracked: dict[str, tuple[int, int, int, int, int, int, str | None]] | None = None
        self._excluded_dirty: set[str] = set()
        self._current_snap: WorkspaceSnapshot | None = None
        self._can_reuse_prev: bool = False

        layout = _repo_layout(cwd)
        if layout is None:
            return
        root, git_dirs = layout
        self.root = root
        self.git_dirs = git_dirs
        try:
            prefix = Path(cwd).resolve().relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            prefix = ""
        self.cwd_prefix = "" if prefix == "." else prefix
        extra_norm: list[str] = []
        for raw_p in extra_paths:
            if not raw_p:
                continue
            for cand in (_join_prefix(self.cwd_prefix, raw_p), posixpath.normpath(raw_p)):
                if cand and cand != "." and not cand.startswith("..") and cand not in extra_norm:
                    extra_norm.append(cand)
        self._extra_root_paths = tuple(extra_norm)
        status_info = _git_status_porcelain(root, git_dirs.get(".git"), include_untracked=True)
        if status_info is not None:
            self._initial_status = status_info
            self.baseline = frozenset(status_info[3])
            self._excluded_dirty = set(status_info[2])
            self._clean_tracked = self._load_clean_tracked(root, self._excluded_dirty)
        else:
            self.baseline = untracked_baseline(cwd, _root=root)

    @staticmethod
    def _record_clean_entry(
        root: Path,
        rel: str,
        now_ns: int,
    ) -> tuple[int, int, int, int, int, int, str | None] | None:
        parent_rel = posixpath.dirname(rel)
        if parent_rel and _symlink_in_path(root, parent_rel):
            return None
        p = root / rel
        st = os.lstat(p)
        if stat.S_ISDIR(st.st_mode):
            return None  # gitlink / submodule: use git status per command
        if stat.S_ISLNK(st.st_mode):
            return (-1, 0, st.st_mode & 0o7777, 0, 0, 0, os.readlink(p))
        if not stat.S_ISREG(st.st_mode):
            return None
        extra_sha: str | None = None
        if now_ns - st.st_ctime_ns <= 1_000_000_000:
            if st.st_size > (4 << 20):
                return None
            extra_sha = _sha256_file(p)
        return (st.st_dev, st.st_ino, st.st_mode & 0o7777, st.st_size, st.st_mtime_ns, st.st_ctime_ns, extra_sha)

    @classmethod
    def _load_clean_tracked(
        cls,
        root: Path,
        dirty_set: set[str],
    ) -> dict[str, tuple[int, int, int, int, int, int, str | None]] | None:
        if _local_git_configs_may_define_filters(root):
            return None
        tracked = _git_paths(root, "ls-files", "-z")
        if tracked is None or len(tracked) > 5_000:
            return None
        now_ns = time.time_ns()
        clean: dict[str, tuple[int, int, int, int, int, int, str | None]] = {}
        for rel in tracked:
            if rel in dirty_set:
                continue
            try:
                rec = cls._record_clean_entry(root, rel, now_ns)
            except OSError:
                return None
            if rec is None:
                return None
            clean[rel] = rec
        return clean

    def _ensure_backup_dir(self) -> Path:
        if self.backup_dir is None or not self.backup_dir.is_dir():
            self.backup_dir = Path(tempfile.mkdtemp(prefix="cc-agy-check-snap-"))
        return self.backup_dir

    def before_command(self) -> WorkspaceSnapshot | None:
        if self.root is None:
            return None
        if self._initial_status is not None:
            head_ref, head_sha, dirty, _ = self._initial_status
            self._initial_status = None
            snap = _build_workspace_snapshot(
                self.root,
                self.cwd_prefix,
                self.git_dirs,
                head_ref,
                head_sha,
                dirty,
                self.baseline or (),
                extra_paths=self._extra_root_paths,
                backup_cache=self.backup_cache,
                session_fds=self.backup_fds,
                keep_backups=True,
            )
            snap.clean_tracked = self._clean_tracked
            self._current_snap = snap
            self._can_reuse_prev = False
            return snap
        if self._can_reuse_prev and self._current_snap is not None:
            self._can_reuse_prev = False
            return self._current_snap
        status_info = _git_status_porcelain(self.root, self.git_dirs.get(".git"), include_untracked=False)
        if status_info is None:
            self._clean_tracked = None
            self._current_snap = None
            self._can_reuse_prev = False
            return None
        head_ref, head_sha, dirty, _ = status_info
        new_dirty_set = set(dirty)
        if self._clean_tracked is not None:
            if self._excluded_dirty - new_dirty_set:
                # A previously dirty file became clean wrt HEAD (e.g. restore_tracked=False); invalidate
                # _clean_tracked so it never lacks a clean tracked file.
                self._clean_tracked = None
            else:
                for d_rel in new_dirty_set - self._excluded_dirty:
                    self._clean_tracked.pop(d_rel, None)
                self._excluded_dirty = new_dirty_set
        snap = _build_workspace_snapshot(
            self.root,
            self.cwd_prefix,
            self.git_dirs,
            head_ref,
            head_sha,
            dirty,
            self.baseline or (),
            extra_paths=self._extra_root_paths,
            backup_cache=self.backup_cache,
            session_fds=self.backup_fds,
            keep_backups=True,
        )
        snap.clean_tracked = self._clean_tracked
        self._current_snap = snap
        self._can_reuse_prev = False
        return snap

    def after_command(
        self,
        snap: WorkspaceSnapshot | None,
        *,
        allowed: Collection[str] = (),
        restore: bool = True,
        change_set: Collection[str] = (),
    ) -> tuple[list[str], list[str]]:
        changed, untrusted = check_side_effects(snap, allowed=allowed, restore=restore, change_set=change_set)
        if not restore or allowed:
            self._clean_tracked = None
        elif changed and self._clean_tracked is not None and self.root is not None:
            now_ns = time.time_ns()
            for c_rel in changed:
                if c_rel.startswith(".git"):
                    self._clean_tracked = None
                    break
                if c_rel in self._clean_tracked:
                    try:
                        rec = self._record_clean_entry(self.root, c_rel, now_ns)
                    except OSError:
                        rec = None
                    if rec is None:
                        self._clean_tracked = None
                        break
                    self._clean_tracked[c_rel] = rec
                elif c_rel not in self._excluded_dirty:
                    self._clean_tracked = None
                    break
        self._can_reuse_prev = bool(snap is not None and restore and not allowed and not changed)
        return changed, untrusted

    def close(self) -> None:
        _close_backup_fds(self.backup_fds)
        self.backup_fds.clear()
        if self.backup_dir is not None:
            _rmtree_writable(self.backup_dir)
            self.backup_dir = None
        self.backup_cache.clear()
        self._current_snap = None


def _non_git_files(
    cwd: Path,
    *,
    max_entries: int = 20_000,
    make_dirs_readable: bool = False,
    skip_unreadable_dirs: Collection[str] = (),
    unreadable_dirs_out: set[str] | None = None,
    dirs_out: set[str] | None = None,
) -> set[str] | None:
    """List relative file and symlink paths inside a non-git ``cwd`` (without following symlinks)."""
    if cwd.is_symlink() or not cwd.is_dir():
        return None
    try:
        os.listdir(cwd)
    except OSError:
        return None
    skip_dirs = {".git", ".ctx-session-reads", ".cc-agy-review", "node_modules", ".venv", ".tox", ".nox"}
    skip_unreadable = set(skip_unreadable_dirs)
    found: set[str] = set()

    def _on_err(err: OSError) -> None:
        if unreadable_dirs_out is not None and getattr(err, "filename", None):
            try:
                rel_err = Path(err.filename).relative_to(cwd).as_posix()
                if rel_err != ".":
                    unreadable_dirs_out.add(rel_err)
            except ValueError:
                pass

    for dirpath, dirnames, filenames in os.walk(cwd, topdown=True, onerror=_on_err, followlinks=False):
        dp = Path(dirpath)
        kept_dirs: list[str] = []
        for d in dirnames:
            if d in skip_dirs:
                continue
            sub = dp / d
            rel_d = sub.relative_to(cwd).as_posix()
            if sub.is_symlink():
                found.add(rel_d)
                continue
            if rel_d in skip_unreadable:
                continue
            if dirs_out is not None:
                dirs_out.add(rel_d)
            if make_dirs_readable and sub.is_dir():
                try:
                    smode = sub.lstat().st_mode & 0o7777
                    if (smode & 0o700) != 0o700:
                        os.chmod(sub, smode | 0o700)
                except OSError:
                    pass
            kept_dirs.append(d)
        dirnames[:] = kept_dirs
        for f in filenames:
            rel = (dp / f).relative_to(cwd).as_posix()
            if not rel.startswith(_PLUGIN_SCRATCH_PREFIXES):
                found.add(rel)
        if len(found) > max_entries:
            return None
    return found


def _untracked(cwd: Path) -> set[str]:
    paths = _git_paths(cwd, "ls-files", "--others", "--exclude-standard", "-z")
    if paths is None:
        return _non_git_files(cwd) or set()
    return {
        p for p in paths
        if not p.startswith((".ctx-session-reads/", ".cc-agy-review/"))
        and p not in (".ctx-session-reads", ".cc-agy-review")
    }


def execute_command_suite(
    commands: Sequence[str],
    cwd: Path,
    *,
    timeout: float = 300.0,
    total_wall_budget: float | None = None,
    stop_on_failure: bool = True,
    stop_on_timeout: bool = True,
    log_dir: Path | None = None,
    clean_new_files: bool = True,
    allowed_mutations: Collection[str] = (),
    restore_tracked: bool = True,
    change_set: Collection[str] = (),
    quarantine_dir: Path | None = None,
    max_tail_chars: int = TAIL_CHARS,
    check_all_vacuous: bool = False,
    diagnostic_hook: Callable[[str, int | None, bool, str, float, Mapping[str, str]], dict[str, Any] | None] | None = None,
) -> tuple[list[tuple[CommandResult, dict[str, Any]]], dict[str, Any]]:
    """Shared execution engine for batch checks (``run_commands``) and delegation checks (``run_acceptance``)."""
    from server.jobs import _parse_test_counts

    items: list[tuple[CommandResult, dict[str, Any]]] = []
    cleanup_summary: dict[str, Any] = {"created": [], "quarantine": {}}
    env = scrubbed_env(cwd)
    removed_env = sorted(secret_env_names())[:30]
    before: set[str] | None = None
    before_unreadable_dirs: set[str] = set()
    before_dirs: set[str] = set()
    own_logs: set[Path] = set()
    suite_started_mono = time.monotonic()
    session = _SnapshotSession(cwd, extra_paths=(*allowed_mutations, *change_set))
    baseline = session.baseline
    if clean_new_files and baseline is None:
        before = _non_git_files(cwd, unreadable_dirs_out=before_unreadable_dirs, dirs_out=before_dirs)
    try:
        for cmd in commands:
            eff_timeout = float(timeout)
            if total_wall_budget is not None:
                rem_wall = float(total_wall_budget) - max(0.0, time.monotonic() - suite_started_mono)
                if rem_wall < 1.0:
                    skipped_res = CommandResult(
                        command=cmd,
                        passed=False,
                        exit_code=None,
                        timed_out=True,
                        duration_seconds=0.0,
                        tail="Skipped: acceptance wall budget exhausted before session deadline.",
                    )
                    items.append((
                        skipped_res,
                        {
                            "counts": {"passed": 0, "failed": 0, "errors": 0},
                            "raw_tail": skipped_res.tail,
                            "wall_exhausted_skipped": True,
                        },
                    ))
                    break
                eff_timeout = min(eff_timeout, max(1.0, rem_wall))

            log_path = ""
            log_handle = None
            if log_dir is not None:
                log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                log_file = log_dir / f"check-{uuid.uuid4().hex[:10]}.log"
                log_handle = open(log_file, "w", encoding="utf-8")
                log_path = str(log_file)
                key_log = _canonical_non_symlink_key(log_file)
                if key_log is not None:
                    own_logs.add(key_log)

            snapshot = session.before_command()
            changed: list[str] = []
            untrusted: list[str] = []
            diag_extra: dict[str, Any] = {}
            cmd_started = time.time()
            try:
                rc, timed_out, out, _ = execute_check_process(
                    cmd,
                    cwd,
                    env,
                    timeout=eff_timeout,
                    log_handle=log_handle,
                    max_tail_chars=max_tail_chars,
                )
                if diagnostic_hook is not None and not timed_out:
                    rem_diag = eff_timeout
                    if total_wall_budget is not None:
                        rem_diag = min(
                            float(timeout),
                            max(1.0, float(total_wall_budget) - max(0.0, time.monotonic() - suite_started_mono)),
                        )
                    hook_res = diagnostic_hook(cmd, rc, timed_out, out, rem_diag, env)
                    if hook_res:
                        diag_extra.update(hook_res)
            finally:
                if log_handle is not None:
                    log_handle.close()
                changed, untrusted = session.after_command(
                    snapshot,
                    allowed=allowed_mutations,
                    restore=restore_tracked,
                    change_set=change_set,
                )
            total_dur = time.time() - cmd_started
            counts = _parse_test_counts(out)
            vacuous = (
                not timed_out
                and (check_all_vacuous or is_behavioral_check(cmd))
                and vacuous_test_run(out, rc, timed_out, counts)
            )
            display_out = str(diag_extra.get("combined_output") or out)
            tail_text = display_out[-max_tail_chars:]
            if vacuous:
                tail_text += (
                    "\n[cc-agy] The command exited cleanly but ran no test (0 collected or every test skipped); "
                    "it does not count as a passing check."
                )
            if untrusted:
                tail_text += side_effect_note(untrusted, restored=restore_tracked)
            force_failed = bool(diag_extra.get("force_failed"))
            passed = (not timed_out) and rc == 0 and not vacuous and not untrusted and not force_failed
            result = CommandResult(
                command=cmd,
                passed=passed,
                exit_code=None if timed_out else rc,
                timed_out=timed_out,
                duration_seconds=round(total_dur, 2),
                tail=tail_text,
                log_path=log_path,
                vacuous=vacuous,
                side_effects_restored=changed[:50],
                untrusted_side_effects=untrusted[:50],
                credential_env_removed=[] if passed else removed_env,
            )
            meta = {
                "counts": counts,
                "raw_tail": display_out[-max_tail_chars:],
                **diag_extra,
            }
            items.append((result, meta))
            if stop_on_failure and not result.passed:
                break
            if stop_on_timeout and timed_out:
                break
    finally:
        session.close()
        if clean_new_files:
            if baseline is not None:
                q_report: dict[str, Any] = {}
                created = remove_new_untracked(
                    cwd,
                    baseline,
                    whole_checkout=restore_tracked,
                    quarantine_dir=quarantine_dir,
                    report=q_report,
                    preserve_paths=own_logs,
                    _root=session.root,
                )
                cleanup_summary["created"] = created
                cleanup_summary["quarantine"] = q_report
            elif before is not None:
                created_ng: list[str] = []
                if restore_tracked and not cwd.is_symlink() and cwd.is_dir():
                    try:
                        cmode = cwd.lstat().st_mode & 0o7777
                        if (cmode & 0o700) != 0o700:
                            os.chmod(cwd, cmode | 0o700)
                    except OSError:
                        pass
                after_dirs: set[str] = set()
                after = _non_git_files(
                    cwd,
                    make_dirs_readable=restore_tracked,
                    skip_unreadable_dirs=before_unreadable_dirs,
                    dirs_out=after_dirs,
                )
                if after is not None:
                    for rel in sorted(after - before):
                        target = cwd / rel
                        if own_logs and _canonical_non_symlink_key(target) in own_logs:
                            continue
                        if not (target.is_file() or target.is_symlink()):
                            continue
                        if restore_tracked or target.is_symlink() or _is_run_artifact(rel):
                            if _unlink_forcefully(cwd, target):
                                created_ng.append(rel)
                    if restore_tracked:
                        for drel in sorted(after_dirs - before_dirs, reverse=True):
                            try:
                                os.rmdir(cwd / drel)
                            except OSError:
                                pass
                cleanup_summary["created"] = created_ng
    return items, cleanup_summary


def run_commands(
    commands: list[str] | tuple[str, ...],
    cwd: Path,
    *,
    timeout: float = 300.0,
    stop_on_failure: bool = True,
    log_dir: Path | None = None,
    clean_new_files: bool = True,
    allowed_mutations: Collection[str] = (),
    restore_tracked: bool = True,
    change_set: Collection[str] = (),
) -> list[CommandResult]:
    """Run commands in order inside cwd; stop at the first failure unless stop_on_failure is False."""
    items, _ = execute_command_suite(
        commands,
        cwd,
        timeout=timeout,
        stop_on_failure=stop_on_failure,
        stop_on_timeout=stop_on_failure,
        log_dir=log_dir,
        clean_new_files=clean_new_files,
        allowed_mutations=allowed_mutations,
        restore_tracked=restore_tracked,
        change_set=change_set,
        max_tail_chars=TAIL_CHARS,
        check_all_vacuous=False,
    )
    return [res for res, _ in items]


_ERROR_LINE_RE = re.compile(
    r"(Traceback \(most recent call last\)|^E\s|Error[:\s]|error[:\[]|FAILED|FAIL:|AssertionError|panic:|undefined|cannot find)",
    re.MULTILINE,
)


_SYNTAX_ONLY_RE = re.compile(
    r"^\s*(?:python3?\s+-m\s+(?:py_compile|compileall)|ruff\b|flake8\b|black\b|isort\b|mypy\b|pyright\b|"
    r"(?:npx\s+)?(?:tsc|eslint|prettier|biome)\b|gofmt\b|go\s+vet\b|cargo\s+check\b|rustfmt\b|test\s+-[efds]\b)",
    re.I,
)

_BEHAVIORAL_TEST_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*(?:(?:uv|poetry|pipenv)\s+run\s+|bundle\s+exec\s+|env\s+)*(?:\S+/)?(?:"
    r"pytest\b|py\.test\b|python[0-9.]*\s+-m\s+(?:pytest|unittest)\b|"
    r"tox\b|nox\b|go\s+test\b|cargo\s+(?:test|nextest)\b|"
    r"(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test(?::[a-z0-9_:-]+)?\b|"
    r"(?:npx\s+)?(?:vitest|jest|mocha|ava|playwright|cypress)\b|"
    r"rspec\b|rake\s+test\b|"
    r"mvn\s+test\b|gradle(?:w)?\s+test\b|ctest\b|dotnet\s+test\b|bats\b|phpunit\b"
    r")",
    re.I,
)


_NON_EXEC_TEST_FLAG_RE = re.compile(
    r"(?:^|\s)(?:--collect-only|--co|--setup-only|--no-run|--dry-run|--listTests|--fixtures|--markers|--version|-V|--help|-h|-run=\^\$)(?:\s|$)"
    r"|\bgo\s+test\b[^\n;|&]*\s(?:-c|-list)(?:\s|$)"
)
_EXIT_MASK_RE = re.compile(
    r"\|\|\s*(?:true\b|:(?=\s|$)|exit\s+0\b|echo\b)|;\s*(?:exit\s+0\b|true\b|:(?=\s|$))",
    re.I,
)


def is_behavioral_check(cmd: str) -> bool:
    """Return True if ``cmd`` executes a behavioral test runner rather than syntax/type/lint checks."""
    text = (cmd or "").strip()
    if not text or text.startswith("#") or _SYNTAX_ONLY_RE.match(text):
        return False
    if _EXIT_MASK_RE.search(text) or _NON_EXEC_TEST_FLAG_RE.search(text):
        return False
    has_pipefail = bool(re.search(r"\bset\s+-o\s+pipefail\b", text))
    has_errexit = bool(re.search(r"\bset\s+-[a-z]*e", text))
    semi_parts = [s.strip() for s in re.split(r";|\n", text) if s.strip() and not s.strip().startswith("#")]
    if not semi_parts:
        return False
    candidate_semi = semi_parts if has_errexit else [semi_parts[-1]]
    for semi_seg in candidate_semi:
        for and_seg in semi_seg.split("&&"):
            pipe_parts = [p.strip() for p in re.split(r"(?<!\|)\|(?!\|)", and_seg) if p.strip()]
            if not pipe_parts:
                continue
            if len(pipe_parts) > 1 and not has_pipefail:
                continue
            cleaned = pipe_parts[0].lstrip("(").strip()
            if cleaned and not cleaned.startswith("#") and _BEHAVIORAL_TEST_RE.match(cleaned):
                return True
    return False


def has_behavioral_check(cmds: list[str] | tuple[str, ...]) -> bool:
    return any(is_behavioral_check(c) for c in cmds)


def failure_tail(results: list[CommandResult], max_chars: int = 1_200) -> str:
    """Short text for the first failing command: the first error block plus the closing summary."""
    for res in results:
        if res.passed:
            continue
        head = f"$ {res.command}\n" + ("(timed out)\n" if res.timed_out else f"(exit {res.exit_code})\n")
        body = res.tail
        budget = max(200, max_chars - len(head))
        match = _ERROR_LINE_RE.search(body)
        if match and len(body) > budget:
            first = body[match.start(): match.start() + budget * 2 // 3]
            summary = body[-(budget - len(first)):]
            return head + first + ("\n...\n" + summary if summary and summary not in first else "")
        return head + body[-budget:]
    return ""
