"""Run verification commands for file tasks in a job copy.

Each command runs in its own process group with vendor credentials removed from the environment and
is killed as a group when it overruns its timeout (and after it exits, so stray background children
never outlive the check). Output is kept as a bounded tail; the full log is written next to the job
so it stays addressable when the tail is not enough.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
import filecmp
import hashlib
import os
import posixpath
import re
import shutil
import tempfile
import threading
from typing import Any, Callable
from pathlib import Path, PurePosixPath
import signal
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
    git_dirs: dict[str, Path] = field(default_factory=dict)
    git_dir_modes: dict[str, int | None] = field(default_factory=dict)
    git_meta: dict[str, tuple[str, str | None, int | None]] = field(default_factory=dict)
    hooks: dict[str, frozenset[str]] = field(default_factory=dict)
    hooks_dir_state: dict[str, tuple[bool, int | None, str | None, str | None]] = field(default_factory=dict)
    info_dir_state: dict[str, tuple[bool, int | None, str | None, str | None]] = field(default_factory=dict)
    git_state: frozenset[str] = frozenset()
    index_digest: str | None = None
    index_backup: str | None = None
    dirty: frozenset[str] = frozenset()
    backup_dir: Path | None = None
    budget: int = _SNAPSHOT_MAX_BYTES


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


def _capture(snap: WorkspaceSnapshot, path: Path) -> tuple[str, str | None, int | None] | None:
    if path.is_symlink():
        return ("symlink", os.readlink(path), None)
    if path.is_file():
        st = path.stat()
        if st.st_size > snap.budget or len(snap.entries) >= _SNAPSHOT_MAX_FILES:
            # Past the backup budget: record size and mtime only (detected, reported, not restorable).
            return ("stat", f"{st.st_size}:{st.st_mtime_ns}", st.st_mode & 0o7777)
        snap.budget -= st.st_size
        if snap.backup_dir is None:
            snap.backup_dir = Path(tempfile.mkdtemp(prefix="cc-agy-check-snap-"))
        saved = snap.backup_dir / f"{len(snap.entries) + len(snap.git_meta):06d}"
        shutil.copyfile(path, saved)
        return ("file", str(saved), st.st_mode & 0o7777)
    if not path.exists():
        return ("absent", None, None)
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


def _index_digest(root: Path) -> str | None:
    res = _git(root, "ls-files", "--stage", "-z")
    return hashlib.sha256(res.stdout).hexdigest() if res.returncode == 0 else None


def _hook_names(git_dir: Path | None) -> frozenset[str]:
    hooks = git_dir / "hooks" if git_dir is not None else None
    if hooks is None or hooks.is_symlink() or not hooks.is_dir():
        return frozenset()
    return frozenset(p.name for p in hooks.iterdir() if not p.name.endswith(".sample"))


def untracked_baseline(cwd: Path) -> frozenset[str] | None:
    """Untracked, non-ignored files of the checkout containing ``cwd``, relative to its root; taken
    once when a run starts, so files earlier commands of the same run create count as run artifacts."""
    root = checkout_root(cwd)
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
    root = checkout_root(cwd)
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
    # Pin the commit by sha: a check may move HEAD (commit, reset, checkout) before we compare.
    head_sha = _git_text(root, "rev-parse", "-q", "--verify", "HEAD^{commit}") or None
    head_ref = _git_text(root, "symbolic-ref", "-q", "HEAD") or None
    base = head_sha or _EMPTY_TREE
    dirty = _git_paths(root, "diff", "--name-only", "--no-renames", "-z", base)
    if dirty is None:
        return None
    if untracked is None:
        untracked = _git_paths(root, "ls-files", "--others", "--exclude-standard", "-z") or []
    snap = WorkspaceSnapshot(
        root=root, cwd_prefix="" if prefix == "." else prefix, base=base, entries={},
        index_tree=_git_text(root, "write-tree"), git_dirs=git_dirs, dirty=frozenset(dirty),
        head_sha=head_sha, head_ref=head_ref,
    )
    for rel in dict.fromkeys([*dirty, *sorted(untracked)]):
        if rel.startswith(_PLUGIN_SCRATCH_PREFIXES):
            continue
        try:
            entry = _capture(snap, root / rel)
        except OSError:
            continue
        if entry is not None:
            snap.entries[rel] = entry
    own_dir = snap.git_dirs.get(".git")
    if own_dir is not None:
        snap.git_state = frozenset(n for n in _GIT_STATE_PATHS if (own_dir / n).exists() or (own_dir / n).is_symlink())
        if snap.index_tree is None and (own_dir / "index").is_file():
            # Unmerged entries make write-tree fail: compare the staged listing and keep the raw index.
            snap.index_digest = _index_digest(root)
            if snap.backup_dir is None:
                snap.backup_dir = Path(tempfile.mkdtemp(prefix="cc-agy-check-snap-"))
            snap.index_backup = str(snap.backup_dir / "git-index")
            shutil.copyfile(own_dir / "index", snap.index_backup)
    def _capture_git_subdir_state(sub: Path) -> tuple[bool, int | None, str | None, str | None]:
        try:
            if sub.is_symlink():
                return (True, None, os.readlink(sub), _symlink_target_digest(sub))
            if sub.is_dir():
                return (False, sub.lstat().st_mode & 0o7777, None, None)
        except OSError:
            pass
        return (False, None, None, None)

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
            try:
                entry = _capture(snap, git_dir / name)
            except OSError:
                continue
            if entry is not None:
                snap.git_meta[f"{label}/{name}"] = entry
    return snap


def _entry_matches(path: Path, kind: str, payload: str | None, mode: int | None) -> bool:
    try:
        if kind == "absent":
            return not path.exists() and not path.is_symlink()
        if kind == "symlink":
            return path.is_symlink() and os.readlink(path) == payload
        if path.is_symlink() or not path.is_file() or payload is None:
            return False
        if mode is not None and (path.stat().st_mode & 0o7777) != mode:
            return False
        if kind == "stat":
            st = path.stat()
            return f"{st.st_size}:{st.st_mtime_ns}" == payload
        return filecmp.cmp(path, payload, shallow=False)
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


def _restore_entry(
    path: Path,
    kind: str,
    payload: str | None,
    mode: int | None,
    *,
    top: Path | None = None,
    unlink_planted_links: bool = False,
) -> None:
    need_parent = kind in ("symlink", "file") and payload is not None
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
    elif kind == "file" and payload is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(payload, path)
        if mode is not None:
            os.chmod(path, mode)


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


def check_side_effects(
    snapshot: WorkspaceSnapshot | None,
    *,
    allowed: Collection[str] = (),
    restore: bool = True,
    change_set: Collection[str] = (),
) -> tuple[list[str], list[str]]:
    """Compare the checkout with ``snapshot`` after one check; return ``(changed, untrusted)``.

    ``changed`` lists every path (relative to the checkout root) the check modified, deleted, replaced
    or staged, plus ``.git/...`` entries for rewritten git metadata; unless ``restore`` is False all of
    it is put back (index first, then files). ``allowed`` paths (relative to the check's cwd, e.g. the
    task file a formatter rewrites) are skipped while they stay plain paths; turning one into a symlink
    is always a side effect. ``untrusted`` is the subset that makes the check's verdict meaningless,
    because then the check did not test the bytes that will ship: changes to tracked files the change
    set already modified, to git metadata, HEAD or the index, and (when restoring, i.e. in an isolated
    job copy) to any other file except generated artifacts (lockfiles, snapshots, coverage). In
    report-only mode (``restore=False``: a check running in the user's own tree, where a person may be
    editing too) only tracked change-set files, paths in ``change_set`` (relative to cwd, e.g. files
    the worker created), git metadata, HEAD and the index count. Planted git config and hooks are put
    back even in report-only mode; HEAD is compared by sha and restored only in an isolated job copy.
    The snapshot's backups are always removed.
    """
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
        # 2. HEAD, then in-progress operations, then the index.
        head_ref_now = _git_text(root, "symbolic-ref", "-q", "HEAD") or None
        head_sha_now = _git_text(root, "rev-parse", "-q", "--verify", "HEAD^{commit}") or None
        if (head_ref_now, head_sha_now) != (snapshot.head_ref, snapshot.head_sha):
            # A commit, reset or checkout made by a check rewrites history with the user's identity.
            flag(".git/HEAD")
            if restore:
                _restore_head(root, snapshot)
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
        if snapshot.index_tree:
            if _git_text(root, "write-tree") != snapshot.index_tree:
                # A check that stages or unstages files changes what a later commit or diff would ship.
                flag(".git/index")
                if restore:
                    _git(root, "read-tree", snapshot.index_tree)
        elif snapshot.index_digest is not None and _index_digest(root) != snapshot.index_digest:
            flag(".git/index")
            if restore and snapshot.index_backup and own_dir is not None:
                try:
                    shutil.copyfile(snapshot.index_backup, own_dir / "index")
                except OSError:
                    pass
        # 3. Files. Compared after HEAD and the index are back, so a path that was only unstaged is not
        # reported. If git cannot list changes any more, the git state itself was tampered with: say
        # so, and still restore every backed-up file.
        now_dirty = _git_paths(root, "diff", "--name-only", "--no-renames", "-z", snapshot.base)
        if now_dirty is None:
            flag(".git")
            now_dirty = []
        from_base: list[str] = []
        for rel in sorted(set(now_dirty) | set(snapshot.entries)):
            if rel.startswith(_PLUGIN_SCRATCH_PREFIXES):
                continue
            if rel in allowed_set and not _symlink_in_path(root, rel) and not (root / rel).is_dir():
                continue
            path = root / rel
            entry = snapshot.entries.get(rel)
            if entry is None:
                from_base.append(rel)
                continue
            kind, payload, mode = entry
            if _entry_matches(path, kind, payload, mode):
                continue
            changed.append(rel)
            if counts_against(rel):
                untrusted.append(rel)
            if restore and kind != "stat":
                try:
                    _restore_entry(path, kind, payload, mode, top=root, unlink_planted_links=True)
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
                    _git(root, "checkout", snapshot.base, "--", *in_base)
        return sorted(set(changed)), sorted(set(untrusted))
    finally:
        if snapshot.backup_dir is not None:
            shutil.rmtree(snapshot.backup_dir, ignore_errors=True)


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
        if _entry_matches(meta_path, kind, payload, mode):
            continue
        flag(key)
        if kind != "stat":
            try:
                _restore_entry(meta_path, kind, payload, mode, top=git_dir, unlink_planted_links=False)
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


def _drain(stream: Any, log_handle: Any, tail: deque[str]) -> None:
    """Copy a process's output to the log file while keeping only a bounded tail in memory."""
    kept = 0
    for chunk in iter(lambda: stream.read(8192), ""):
        if log_handle is not None:
            log_handle.write(chunk)
        tail.append(chunk)
        kept += len(chunk)
        while kept - len(tail[0]) > TAIL_CHARS and len(tail) > 1:
            kept -= len(tail.popleft())


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
) -> list[CommandResult]:
    """Run commands in order inside cwd; stop at the first failure unless stop_on_failure is False.

    Output streams to a log file (when log_dir is set) and only a bounded tail is held in memory.
    Untracked files the commands create (caches, reports) are removed afterwards. Before each command
    the change set is snapshotted; files the command modifies or deletes are restored (except
    ``allowed_mutations``) and listed on its result, and a command that changed anything other than a
    generated artifact fails, because it did not test the files that will ship. A behavioral test
    command that exits 0 but runs no test (zero collected, or every test skipped) also fails and is
    marked ``vacuous``.
    """
    from server.jobs import _parse_test_counts

    results: list[CommandResult] = []
    env = scrubbed_env(cwd)
    removed_env = sorted(secret_env_names())[:30]
    before: set[str] | None = None
    before_unreadable_dirs: set[str] = set()
    before_dirs: set[str] = set()
    own_logs: set[Path] = set()
    baseline = untracked_baseline(cwd)
    if clean_new_files and baseline is None:
        before = _non_git_files(cwd, unreadable_dirs_out=before_unreadable_dirs, dirs_out=before_dirs)
    try:
        for cmd in commands:
            started = time.time()
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
            tail: deque[str] = deque()
            snapshot = snapshot_workspace(cwd, baseline)
            changed: list[str] = []
            untrusted: list[str] = []
            try:
                proc = subprocess.Popen(
                    ["bash", "-c", cmd], cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", start_new_session=True,
                )
                reader = threading.Thread(target=_drain, args=(proc.stdout, log_handle, tail), daemon=True)
                reader.start()
                timed_out = False
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_group(proc.pid)
                    proc.wait()
                _kill_group(proc.pid)
                reader.join(timeout=10)
                if proc.stdout is not None and not reader.is_alive():
                    proc.stdout.close()
            finally:
                if log_handle is not None:
                    log_handle.close()
                changed, untrusted = check_side_effects(snapshot, allowed=allowed_mutations, restore=restore_tracked)
            out = "".join(tail)
            vacuous = (
                not timed_out
                and is_behavioral_check(cmd)
                and vacuous_test_run(out, proc.returncode, timed_out, _parse_test_counts(out))
            )
            tail_text = out[-TAIL_CHARS:]
            if vacuous:
                tail_text += (
                    "\n[cc-agy] The command exited cleanly but ran no test (0 collected or every test skipped); "
                    "it does not count as a passing check."
                )
            if untrusted:
                tail_text += side_effect_note(untrusted, restored=restore_tracked)
            passed = (not timed_out) and proc.returncode == 0 and not vacuous and not untrusted
            result = CommandResult(
                command=cmd,
                passed=passed,
                exit_code=None if timed_out else proc.returncode,
                timed_out=timed_out,
                duration_seconds=round(time.time() - started, 2),
                tail=tail_text,
                log_path=log_path,
                vacuous=vacuous,
                side_effects_restored=changed[:50],
                untrusted_side_effects=untrusted[:50],
                credential_env_removed=[] if passed else removed_env,
            )
            results.append(result)
            if stop_on_failure and not result.passed:
                break
    finally:
        if clean_new_files:
            if baseline is not None:
                remove_new_untracked(cwd, baseline, whole_checkout=restore_tracked, preserve_paths=own_logs)
            elif before is not None:
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
                        if restore_tracked or target.is_symlink() or _is_run_artifact(rel):
                            _unlink_forcefully(cwd, target)
                    if restore_tracked:
                        for drel in sorted(after_dirs - before_dirs, reverse=True):
                            try:
                                os.rmdir(cwd / drel)
                            except OSError:
                                pass
    return results


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
