"""OS-level confinement for Gemini Enterprise (`agy` CLI) workers.

A write worker runs `agy --dangerously-skip-permissions`, so `agy` never stops to ask
before it edits a file or runs a command. Telling the model which files it may touch is
only an instruction. This module makes the operating system enforce it instead:

- Linux uses `bwrap` (bubblewrap). The whole filesystem is mounted read-only, the job's
  isolated repo copy and `agy`'s own runtime state directory are writable (while persistent
  `settings.json`, `mcp/`, `plugins/`, `builtin/`, `bin/`, and `updater/` inside `agy`'s
  config directory stay read-only), `/tmp`, `~/.cache`, the plugin data directory
  (`.hmac_key` and receipts), and sibling worktrees are replaced with private `tmpfs`
  mounts, and credential stores (including symlinked dotfiles) are hidden.
- macOS uses `sandbox-exec` with a Seatbelt profile that denies writes outside the same
  places (while protecting `agy`'s persistent settings/plugins/MCP subpaths) and denies
  reads of the same credential stores and plugin data/worktree directories.

`agy` must reach Google over the network, and neither tool can filter by host, so the
network stays on. Remote git access is blocked by masking `git-remote-http(s)` and `ssh`
binaries inside `bwrap` and by `git_guard_environment` as defense in depth.

`CC_AGY_WORKER_SANDBOX` picks the policy:

- `auto` (default): sandbox when a backend works; otherwise run `agy` without
  `--dangerously-skip-permissions`, so actions that need approval come back as
  `denied_actions` for the lead agent to handle.
- `required`: refuse to start a write worker (at 0 tokens) when no backend works.
- `off`: the previous behavior, no sandbox and no permission prompts. Explicit opt-out.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any

SANDBOX_MODES = ("auto", "required", "off")
DEFAULT_SANDBOX_MODE = "auto"
BACKEND_BWRAP = "bwrap"
BACKEND_SEATBELT = "seatbelt"
_SEATBELT_BIN = "/usr/bin/sandbox-exec"
_PROBE_TIMEOUT_SECONDS = 5.0

# Credential and agent-token stores a worker never needs. Both symlink paths and their
# resolved targets are hidden when they exist; nothing missing is ever created.
_HIDDEN_HOME_DIRS = (
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".kube",
    ".docker",
    ".helm",
    ".terraform.d",
    ".pulumi",
    ".sops",
    ".age",
    ".password-store",
    ".local/share/keyrings",
    ".local/share/opencode",
    ".config/gh",
    ".config/gcloud",
    ".config/opencode",
    ".config/cursor",
    ".config/claude",
    ".config/anthropic",
    ".config/openai",
    ".config/hub",
    ".config/glab-cli",
    ".config/doctl",
    ".config/heroku",
    ".config/stripe",
    ".config/npm",
    ".config/pip",
    ".config/pypoetry",
    ".config/uv",
    ".claude",
    ".codex",
    ".cursor",
    ".anthropic",
    ".openai",
    ".gem",
    "Library/Keychains",
)
_HIDDEN_HOME_FILES = (
    ".netrc",
    ".git-credentials",
    ".pypirc",
    ".npmrc",
    ".yarnrc.yml",
    ".vault-token",
    ".docker/config.json",
    ".kube/config",
    ".cargo/credentials",
    ".cargo/credentials.toml",
    ".gem/credentials",
    ".bundle/config",
    ".composer/auth.json",
    ".pgpass",
    ".my.cnf",
    ".dbshell",
    ".psql_history",
    ".bash_history",
    ".zsh_history",
    ".python_history",
    ".lesshst",
    ".viminfo",
    ".wget-hsts",
)

# Subpaths inside `~/.gemini/antigravity-cli` that a worker must not modify so it cannot
# persist settings, instructions, skills, MCP servers, hooks, or binaries across sessions.
_READONLY_CONFIG_DIRS = (
    "mcp",
    "plugins",
    "builtin",
    "bin",
    "updater",
    "skills",
    "extensions",
    "rules",
    "agents",
    "commands",
    "hooks",
    "workflows",
    "prompts",
)
_READONLY_CONFIG_FILES = (
    "settings.json",
    "config.json",
    "mcp.json",
    "mcp_config.json",
    "hooks.json",
    "GEMINI.md",
    "AGENTS.md",
    "CLAUDE.md",
    "SYSTEM.md",
    "instructions.md",
)

# Remote transport binaries masked inside Linux `bwrap` so even `env -u GIT_CONFIG_COUNT`
# cannot push over HTTP(S), FTP, or SSH.
_MASKED_EXEC_PATHS = (
    "/usr/bin/ssh",
    "/usr/bin/scp",
    "/usr/bin/sftp",
    "/usr/lib/git-core/git-remote-http",
    "/usr/lib/git-core/git-remote-https",
    "/usr/lib/git-core/git-remote-ftp",
    "/usr/lib/git-core/git-remote-ftps",
    "/usr/libexec/git-core/git-remote-http",
    "/usr/libexec/git-core/git-remote-https",
    "/usr/libexec/git-core/git-remote-ftp",
    "/usr/libexec/git-core/git-remote-ftps",
)

# Git settings applied to every `agy` subprocess as defense in depth: no credential helpers
# or prompts, no network transports by default, and push URLs rewritten to a blocked scheme.
_PUSH_BLOCK_PREFIXES = ("/", ".", "~", "file:", "http:", "https:", "ssh:", "git:", "git@")
_BLOCKED_PUSH_SCHEME = "cc-agy-push-blocked://"

_PROBE_LOCK = threading.Lock()
_PROBE_CACHE: dict[tuple[str, str], tuple[bool, str]] = {}


class WorkerSandboxError(RuntimeError):
    """No OS sandbox is usable while `CC_AGY_WORKER_SANDBOX=required`."""


@dataclass(frozen=True)
class SandboxPlan:
    """How one worker will be confined.

    Attributes:
        mode: The configured policy (`auto`, `required`, or `off`).
        backend: `bwrap`, `seatbelt`, or None when the worker is not sandboxed.
        skip_permissions: Whether `agy` may run with `--dangerously-skip-permissions`.
        command_prefix: Arguments to put in front of the `agy` command line.
        warning: A short note for receipts when the worker is not sandboxed, or None.
    """

    mode: str
    backend: str | None
    skip_permissions: bool
    command_prefix: tuple[str, ...] = field(default_factory=tuple)
    warning: str | None = None

    def wrap(self, command: list[str]) -> list[str]:
        """Return `command` prefixed with the sandbox launcher (unchanged when there is none)."""
        return [*self.command_prefix, *command]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe description for receipts and digests."""
        return {"mode": self.mode, "backend": self.backend, "skip_permissions": self.skip_permissions}


def _setting(env: Mapping[str, str] | None, *names: str) -> str:
    for source in (env, os.environ):
        if source is None:
            continue
        for name in names:
            cleaned = str(source.get(name) or "").strip()
            if cleaned and not cleaned.startswith("${"):
                return cleaned
    return ""


def resolve_sandbox_mode(env: Mapping[str, str] | None = None) -> str:
    """Return the worker sandbox policy from `CC_AGY_WORKER_SANDBOX` (default `auto`).

    Args:
        env: Optional environment mapping; falls back to `os.environ` when None.

    Returns:
        One of `auto`, `required`, or `off`.

    Raises:
        ValueError: If the value is not one of `auto`, `required`, or `off`.
    """
    raw = _setting(
        env, "CC_AGY_WORKER_SANDBOX", "CLAUDE_PLUGIN_OPTION_WORKER_SANDBOX", "CC_AGY_USERCFG_WORKER_SANDBOX"
    ).lower()
    aliases = {
        "": DEFAULT_SANDBOX_MODE,
        "on": "auto",
        "1": "auto",
        "true": "auto",
        "require": "required",
        "strict": "required",
        "0": "off",
        "false": "off",
        "none": "off",
        "disabled": "off",
    }
    mode = aliases.get(raw, raw)
    if mode not in SANDBOX_MODES:
        raise ValueError(f"CC_AGY_WORKER_SANDBOX must be one of {', '.join(SANDBOX_MODES)} (got {raw!r}).")
    return mode


def _probe(command: list[str]) -> tuple[bool, str, bool]:
    """Run `command` and return `(works, detail, cacheable)`."""
    try:
        proc = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True, timeout=_PROBE_TIMEOUT_SECONDS, check=False
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"TimeoutExpired: {exc}", False
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}", True
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        return False, detail[-1] if detail else f"exit code {proc.returncode}", True
    return True, "", True


def detect_sandbox_backend(env: Mapping[str, str] | None = None) -> tuple[str | None, str | None, str]:
    """Find a working sandbox launcher for this machine.

    The launcher is probed with the exact namespace and mount flags used at launch (`--proc`,
    `--tmpfs /tmp`, `--unshare-pid`, `--unshare-ipc`, `--new-session`), because `bwrap` can be
    installed but blocked inside restricted containers. Transient timeouts are not cached.

    Args:
        env: Optional environment mapping; falls back to `os.environ` when None.

    Returns:
        `(backend, launcher_path, reason)`. `backend` and `launcher_path` are None when
        nothing works, and `reason` then says why.
    """
    if sys.platform.startswith("linux"):
        launcher = _setting(env, "CC_AGY_BWRAP_BIN") or shutil.which("bwrap") or ""
        if not launcher:
            return None, None, "bubblewrap (`bwrap`) is not installed"
        probe_cmd = [
            launcher,
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--unshare-pid",
            "--unshare-ipc",
            "--new-session",
            "--die-with-parent",
            "true",
        ]
        backend = BACKEND_BWRAP
    elif sys.platform == "darwin":
        launcher = _SEATBELT_BIN if Path(_SEATBELT_BIN).exists() else ""
        if not launcher:
            return None, None, "`sandbox-exec` is not available"
        probe_cmd = [launcher, "-p", "(version 1)(allow default)", "/usr/bin/true"]
        backend = BACKEND_SEATBELT
    else:
        return None, None, f"no supported sandbox on platform {sys.platform!r}"

    cache_key = (backend, launcher)
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get(cache_key)
    if cached is None:
        works, detail, cacheable = _probe(probe_cmd)
        if cacheable:
            with _PROBE_LOCK:
                _PROBE_CACHE[cache_key] = (works, detail)
    else:
        works, detail = cached
    if not works:
        return None, None, f"`{launcher}` is installed but does not work here ({detail})"
    return backend, launcher, ""


def _resolve_existing_dir(home: Path, relative: str) -> tuple[Path, ...]:
    """Return existing directory path(s) for `home / relative`, resolving symlinks."""
    path = home / relative
    results: list[Path] = []
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return ()
    if not resolved.is_dir():
        return ()
    results.append(resolved)
    if path.exists() and not path.is_symlink() and path != resolved:
        results.append(path)
    return tuple(results)


def _resolve_existing_file(home: Path, relative: str) -> tuple[Path, ...]:
    """Return existing file path(s) for `home / relative`, resolving symlinks."""
    path = home / relative
    results: list[Path] = []
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return ()
    if not resolved.is_file():
        return ()
    results.append(resolved)
    if path.exists() and not path.is_symlink() and path != resolved:
        results.append(path)
    return tuple(results)


def _is_worktrees_container(directory: Path) -> bool:
    name = directory.name
    return (
        name == "worktrees"
        or name.startswith(".cc-agy-worktrees-")
        or name.startswith("cc-agy-worktrees-")
    )


def _has_plugin_data_markers(directory: Path) -> bool:
    return (
        (directory / ".hmac_key").exists()
        or (directory / "ledger.jsonl").exists()
        or (directory / "receipts").is_dir()
    )


def _data_dir_for(workspace: Path) -> Path | None:
    """Return the plugin data dir associated with `workspace`, or None when not recognizable."""
    parent = workspace.parent
    if not _is_worktrees_container(parent):
        return None
    grandparent = parent.parent
    if grandparent == Path(grandparent.anchor) or grandparent == Path.home():
        return None
    if parent.name == "worktrees" or _has_plugin_data_markers(grandparent):
        return grandparent
    try:
        for child in grandparent.iterdir():
            if child != parent and child.is_dir() and _has_plugin_data_markers(child):
                return child
    except OSError:
        pass
    return None


def _extra_xdg_hidden_dirs(env: Mapping[str, str] | None) -> tuple[Path, ...]:
    """Return existing XDG credential/config directories when XDG_*_HOME is customized."""
    found: list[Path] = []
    xdg_data = _setting(env, "XDG_DATA_HOME")
    if xdg_data:
        base_data = Path(xdg_data).expanduser()
        for rel in ("opencode", "keyrings"):
            found.extend(_resolve_existing_dir(base_data, rel))
    xdg_cfg = _setting(env, "XDG_CONFIG_HOME")
    if xdg_cfg:
        base_cfg = Path(xdg_cfg).expanduser()
        for rel in (
            "gcloud",
            "gh",
            "opencode",
            "cursor",
            "claude",
            "anthropic",
            "openai",
            "hub",
            "glab-cli",
            "doctl",
            "heroku",
            "stripe",
            "npm",
            "pip",
            "pypoetry",
            "uv",
        ):
            found.extend(_resolve_existing_dir(base_cfg, rel))
    return tuple(found)


_SAFE_DOTENV_SUFFIXES = (".example", ".sample", ".template", ".dist")


def _outside_dotenv_files(home: Path, workspace: Path, *, max_files: int = 64) -> tuple[Path, ...]:
    """Find `.env` and `.env.*` files under `home` (up to depth 3) outside `workspace`."""
    try:
        ws_resolved = workspace.resolve()
    except OSError:
        ws_resolved = workspace
    results: list[Path] = []
    search_dirs: list[Path] = [home]
    frontier: list[Path] = [home]
    for _depth in range(3):
        next_frontier: list[Path] = []
        for parent in frontier:
            try:
                for child in parent.iterdir():
                    if len(search_dirs) >= 96:
                        break
                    if child.name.startswith(".") or child.is_symlink():
                        continue
                    if child.is_dir():
                        search_dirs.append(child)
                        next_frontier.append(child)
            except OSError:
                continue
            if len(search_dirs) >= 96:
                break
        frontier = next_frontier
        if not frontier or len(search_dirs) >= 96:
            break
    for directory in search_dirs:
        try:
            for entry in directory.iterdir():
                if len(results) >= max_files:
                    return tuple(results)
                name = entry.name
                if not (name == ".env" or name.startswith(".env.")):
                    continue
                if name.lower().endswith(_SAFE_DOTENV_SUFFIXES):
                    continue
                resolved = entry.resolve(strict=True)
                if not resolved.is_file():
                    continue
                if resolved == ws_resolved or ws_resolved in resolved.parents:
                    continue
                results.append(resolved)
                if entry.exists() and not entry.is_symlink() and entry != resolved:
                    results.append(entry)
        except OSError:
            continue
    return tuple(results)


def _is_active_conversation_entry(entry_name: str, active: str) -> bool:
    """Return True when `entry_name` belongs to `active` (including `.db`, `.db-wal`, `.db-shm`, `.pb`)."""
    if not active:
        return False
    return entry_name == active or entry_name.startswith(f"{active}.")


def _foreign_conversation_files(config_dir: Path, active_conversation_id: str | None) -> tuple[Path, ...]:
    """Return existing conversation files in `config_dir/conversations` not belonging to this job."""
    conv_dir = config_dir / "conversations"
    if not conv_dir.is_dir():
        return ()
    active = (active_conversation_id or "").strip()
    masked: list[Path] = []
    try:
        for entry in conv_dir.iterdir():
            if _is_active_conversation_entry(entry.name, active):
                continue
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                continue
            if resolved.is_file():
                masked.append(resolved)
    except OSError:
        return ()
    return tuple(masked)


def _foreign_brain_dirs(config_dir: Path, active_conversation_id: str | None) -> tuple[Path, ...]:
    """Return existing `config_dir/brain/<id>` directories belonging to other conversations.

    Real `agy` stores per-conversation transcripts and scratch artifacts under
    `config_dir/brain/<conversation_id>/`. Masking only foreign `<other_id>` subdirectories
    keeps `brain/` writable when Turn 1 allocates a new conversation ID while preserving
    `brain/<conversation_id>/` across `delegate_continue` and resumed `ask_gemini` turns.
    """
    brain_dir = config_dir / "brain"
    if not brain_dir.is_dir():
        return ()
    active = (active_conversation_id or "").strip()
    masked: list[Path] = []
    try:
        for entry in brain_dir.iterdir():
            if _is_active_conversation_entry(entry.name, active):
                continue
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                continue
            if resolved.is_dir():
                masked.append(resolved)
    except OSError:
        return ()
    return tuple(masked)


def _protected_plugin_dirs(
    workspace: Path,
    home: Path,
    *,
    data_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Return plugin data and sibling-worktree directories that must be masked before binding `workspace`."""
    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(data_dir)
    explicit_data = _setting(env, "CC_AGY_DATA_DIR", "CLAUDE_PLUGIN_DATA")
    if explicit_data:
        candidates.append(Path(explicit_data).expanduser())
    inferred = _data_dir_for(workspace)
    if inferred is not None:
        candidates.append(inferred)
    if _is_worktrees_container(workspace.parent):
        candidates.append(workspace.parent)
    claude_plugin_data = home / ".claude" / "plugins" / "data"
    if claude_plugin_data.is_dir():
        candidates.append(claude_plugin_data)

    seen: set[str] = set()
    ordered: list[Path] = []
    for raw in candidates:
        try:
            resolved = raw.resolve()
        except (OSError, RuntimeError):
            continue
        if not resolved.is_dir():
            continue
        if resolved in (Path(resolved.anchor), home.resolve()):
            continue
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            ordered.append(resolved)
    return tuple(ordered)


def _prepare_config_dir_guards(config_dir: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Ensure critical settings/plugin subpaths inside `config_dir` exist so they can be bound read-only."""
    ro_dirs: list[Path] = []
    ro_files: list[Path] = []
    if not config_dir.is_dir():
        return (), ()
    for subdir in _READONLY_CONFIG_DIRS:
        target_dir = config_dir / subdir
        try:
            if not target_dir.exists() and not target_dir.is_symlink():
                target_dir.mkdir(mode=0o700, exist_ok=True)
            resolved_dir = target_dir.resolve(strict=True)
            if resolved_dir.is_dir():
                ro_dirs.append(resolved_dir)
        except OSError:
            continue
    for filename in _READONLY_CONFIG_FILES:
        target_file = config_dir / filename
        try:
            if not target_file.exists() and not target_file.is_symlink():
                stub = "{}\n" if filename.endswith(".json") else ""
                target_file.write_text(stub, encoding="utf-8")
                target_file.chmod(0o600)
            resolved_file = target_file.resolve(strict=True)
            if resolved_file.is_file():
                ro_files.append(resolved_file)
        except OSError:
            continue
    return tuple(ro_dirs), tuple(ro_files)


def build_bwrap_prefix(
    launcher: str,
    *,
    workspace: Path,
    config_dir: Path,
    home: Path,
    adc_path: Path | None,
    readonly_paths: tuple[Path, ...] = (),
    data_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
    read_only_workspace: bool = False,
    conversation_id: str | None = None,
) -> list[str]:
    """Build the `bwrap` arguments that confine one worker. Order matters: later mounts win."""
    args = [
        launcher,
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--unshare-pid",
        "--unshare-ipc",
        "--new-session",
        "--die-with-parent",
    ]
    for cache_dir in _resolve_existing_dir(home, ".cache"):
        args += ["--tmpfs", str(cache_dir)]
    for protected_dir in _protected_plugin_dirs(workspace, home, data_dir=data_dir, env=env):
        args += ["--tmpfs", str(protected_dir)]

    masked_dirs_seen: set[str] = set()
    for rel in _HIDDEN_HOME_DIRS:
        for hidden_dir in _resolve_existing_dir(home, rel):
            key = str(hidden_dir)
            if key not in masked_dirs_seen:
                masked_dirs_seen.add(key)
                args += ["--tmpfs", key]
    for xdg_dir in _extra_xdg_hidden_dirs(env):
        key = str(xdg_dir)
        if key not in masked_dirs_seen:
            masked_dirs_seen.add(key)
            args += ["--tmpfs", key]

    masked_files_seen: set[str] = set()
    for rel in _HIDDEN_HOME_FILES:
        for hidden_file in _resolve_existing_file(home, rel):
            key = str(hidden_file)
            if key not in masked_files_seen:
                masked_files_seen.add(key)
                # `--dev-bind` keeps /dev/null readable (a plain bind mount is `nodev`).
                args += ["--dev-bind", "/dev/null", key]
    for dotenv_file in _outside_dotenv_files(home, workspace):
        key = str(dotenv_file)
        if key not in masked_files_seen:
            masked_files_seen.add(key)
            args += ["--dev-bind", "/dev/null", key]

    if adc_path is not None and adc_path.is_file():
        args += ["--ro-bind", str(adc_path), str(adc_path)]
    for path in readonly_paths:
        # Re-expose paths a private mount above may hide (for example an `agy` binary under /tmp).
        if path.exists():
            args += ["--ro-bind", str(path), str(path)]

    for exec_str in _MASKED_EXEC_PATHS:
        exec_path = Path(exec_str)
        if exec_path.is_file() and not exec_path.is_symlink():
            args += ["--dev-bind", "/dev/null", str(exec_path)]

    # `agy` writes logs, crash reports, and conversation state in `config_dir`, while its
    # persistent settings, MCP configs, plugins, and binaries are re-bound read-only on top.
    args += ["--bind", str(config_dir), str(config_dir)]
    ro_cfg_dirs, ro_cfg_files = _prepare_config_dir_guards(config_dir)
    for cfg_dir in ro_cfg_dirs:
        args += ["--ro-bind", str(cfg_dir), str(cfg_dir)]
    for cfg_file in ro_cfg_files:
        args += ["--ro-bind", str(cfg_file), str(cfg_file)]
    brain_dir = config_dir / "brain"
    try:
        if not brain_dir.exists() and not brain_dir.is_symlink():
            brain_dir.mkdir(mode=0o700, exist_ok=True)
    except OSError:
        pass
    for foreign_brain in _foreign_brain_dirs(config_dir, conversation_id):
        args += ["--tmpfs", str(foreign_brain)]
    for foreign_conv in _foreign_conversation_files(config_dir, conversation_id):
        args += ["--dev-bind", "/dev/null", str(foreign_conv)]

    ws_flag = "--ro-bind" if read_only_workspace else "--bind"
    args += [ws_flag, str(workspace), str(workspace), "--chdir", str(workspace), "--"]
    return args


def _sbpl_string(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_seatbelt_profile(
    *,
    workspace: Path,
    config_dir: Path,
    home: Path,
    adc_path: Path | None,
    data_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
    read_only_workspace: bool = False,
    conversation_id: str | None = None,
) -> str:
    """Build a `sandbox-exec` profile. In SBPL the last matching rule wins."""
    resolved_home = home.resolve() if home.exists() else home
    writable = [config_dir, Path("/private/tmp"), Path("/private/var/folders")]
    if not read_only_workspace:
        writable.insert(0, workspace)
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
        "(allow file-write* "
        + " ".join(f"(subpath {_sbpl_string(p)})" for p in writable)
        + ' (literal "/dev/null") (literal "/dev/tty") (regex #"^/dev/fd/"))',
    ]

    ro_cfg_dirs = [config_dir / sub for sub in _READONLY_CONFIG_DIRS]
    ro_cfg_files = [config_dir / fn for fn in _READONLY_CONFIG_FILES]
    lines.append(
        "(deny file-write* "
        + " ".join(f"(subpath {_sbpl_string(p)})" for p in ro_cfg_dirs)
        + " "
        + " ".join(f"(literal {_sbpl_string(p)})" for p in ro_cfg_files)
        + ")"
    )

    hidden_dirs: list[Path] = list(_foreign_brain_dirs(config_dir, conversation_id))
    for base in (home, resolved_home):
        hidden_dirs.append(base / ".cache")
        hidden_dirs.append(base / "Library" / "Caches")
        for rel in _HIDDEN_HOME_DIRS:
            cand = base / rel
            hidden_dirs.append(cand)
            for resolved_dir in _resolve_existing_dir(base, rel):
                hidden_dirs.append(resolved_dir)
    for xdg_dir in _extra_xdg_hidden_dirs(env):
        hidden_dirs.append(xdg_dir)
    for protected_dir in _protected_plugin_dirs(workspace, home, data_dir=data_dir, env=env):
        hidden_dirs.append(protected_dir)

    hidden_files: list[Path] = []
    for base in (home, resolved_home):
        for rel in _HIDDEN_HOME_FILES:
            hidden_files.append(base / rel)
            for resolved_file in _resolve_existing_file(base, rel):
                hidden_files.append(resolved_file)
    for dotenv_file in _outside_dotenv_files(home, workspace):
        hidden_files.append(dotenv_file)
    for foreign_conv in _foreign_conversation_files(config_dir, conversation_id):
        hidden_files.append(foreign_conv)

    dedup_dirs = list(dict.fromkeys(hidden_dirs))
    dedup_files = list(dict.fromkeys(hidden_files))
    lines.append(
        "(deny file-read* file-write* "
        + " ".join(f"(subpath {_sbpl_string(p)})" for p in dedup_dirs)
        + " "
        + " ".join(f"(literal {_sbpl_string(p)})" for p in dedup_files)
        + ")"
    )
    reopen = [workspace]
    if adc_path is not None:
        reopen.append(adc_path)
    if not read_only_workspace:
        lines.append(f"(allow file-write* (subpath {_sbpl_string(workspace)}))")
    lines.append("(allow file-read* " + " ".join(f"(subpath {_sbpl_string(p)})" for p in reopen) + ")")
    return "\n".join(lines)


def plan_worker_sandbox(
    *,
    workspace_dir: str | Path,
    config_dir: str | Path,
    env: Mapping[str, str] | None = None,
    adc_path: str | Path | None = None,
    readonly_paths: tuple[str | Path, ...] = (),
    data_dir: str | Path | None = None,
    read_only_workspace: bool = False,
    conversation_id: str | None = None,
) -> SandboxPlan:
    """Decide how to confine one `agy` worker.

    Args:
        workspace_dir: The job's isolated repo copy; the only writable project path.
        config_dir: `agy`'s state directory, which it must be able to write for runtime logs.
        env: Environment to read settings from (defaults to `os.environ` when None).
        adc_path: An Application Default Credentials file to keep readable (seat mode), or None.
        readonly_paths: Extra paths to keep readable, such as the folder holding `agy`.
        data_dir: Explicit plugin data directory holding `.hmac_key` and receipts, or None.
        read_only_workspace: Whether `workspace_dir` itself should be mounted read-only (`plan` mode).
        conversation_id: Active `agy` conversation ID to keep accessible inside `config_dir/conversations`.

    Returns:
        A `SandboxPlan` describing the launcher prefix and permission mode.

    Raises:
        WorkerSandboxError: If the policy is `required` and no backend works.
        ValueError: If `CC_AGY_WORKER_SANDBOX` has an unknown value.
    """
    mode = resolve_sandbox_mode(env)
    if mode == "off":
        return SandboxPlan(
            mode=mode,
            backend=None,
            skip_permissions=not read_only_workspace,
            warning="Worker sandbox is off (CC_AGY_WORKER_SANDBOX=off): `agy` can edit any file your account can.",
        )
    backend, launcher, reason = detect_sandbox_backend(env)
    if backend is None or launcher is None:
        if mode == "required":
            raise WorkerSandboxError(
                f"CC_AGY_WORKER_SANDBOX=required, but no OS sandbox is usable: {reason}. Install bubblewrap "
                "(Linux) or set CC_AGY_WORKER_SANDBOX=auto to run without permission skipping instead."
            )
        return SandboxPlan(
            mode=mode,
            backend=None,
            skip_permissions=False,
            warning=(
                f"No OS sandbox is usable ({reason}), so `agy` runs without --dangerously-skip-permissions. "
                "Actions that need approval come back as denied_actions."
            ),
        )

    workspace = Path(workspace_dir).resolve()
    config = Path(config_dir).expanduser().resolve()
    config.mkdir(parents=True, exist_ok=True)
    home = Path(_setting(env, "HOME") or Path.home()).expanduser()
    adc = Path(adc_path).resolve() if adc_path else None
    resolved_data_dir = Path(data_dir).expanduser().resolve() if data_dir else None
    if backend == BACKEND_BWRAP:
        prefix = build_bwrap_prefix(
            launcher,
            workspace=workspace,
            config_dir=config,
            home=home,
            adc_path=adc,
            readonly_paths=tuple(Path(p).resolve() for p in readonly_paths),
            data_dir=resolved_data_dir,
            env=env,
            read_only_workspace=read_only_workspace,
            conversation_id=conversation_id,
        )
    else:
        profile = build_seatbelt_profile(
            workspace=workspace,
            config_dir=config,
            home=home,
            adc_path=adc,
            data_dir=resolved_data_dir,
            env=env,
            read_only_workspace=read_only_workspace,
            conversation_id=conversation_id,
        )
        prefix = [launcher, "-p", profile]
    return SandboxPlan(
        mode=mode,
        backend=backend,
        skip_permissions=not read_only_workspace,
        command_prefix=tuple(prefix),
    )


def git_guard_environment() -> dict[str, str]:
    """Git settings for worker subprocesses (defense in depth alongside OS credential/helper masking).

    Local operations (commit, diff, log, fetching from a path on disk) keep working.
    """
    entries: list[tuple[str, str]] = [
        ("credential.helper", ""),
        ("core.askPass", ""),
        ("core.sshCommand", "false"),
        ("protocol.allow", "never"),
        ("protocol.file.allow", "always"),
    ]
    entries += [(f"url.{_BLOCKED_PUSH_SCHEME}.pushInsteadOf", prefix) for prefix in _PUSH_BLOCK_PREFIXES]
    out = {"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": str(len(entries))}
    for idx, (key, value) in enumerate(entries):
        out[f"GIT_CONFIG_KEY_{idx}"] = key
        out[f"GIT_CONFIG_VALUE_{idx}"] = value
    return out


_SANDBOX_REMEDIATION = (
    "Install bubblewrap (`bwrap`) on Linux or use macOS `sandbox-exec`, or set CC_AGY_WORKER_SANDBOX=auto."
)


def inspect_worker_sandbox(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Summarize the worker sandbox for `doctor` (runs the cached launcher probe, never `agy`)."""
    try:
        mode = resolve_sandbox_mode(env)
    except ValueError as exc:
        return {
            "ok": False,
            "active": False,
            "mode": None,
            "backend": None,
            "launcher": None,
            "reason": str(exc),
            "error": str(exc),
        }
    if mode == "off":
        return {
            "ok": True,
            "active": False,
            "mode": "off",
            "backend": None,
            "launcher": None,
            "reason": "disabled by configuration",
            "error": None,
        }
    backend, launcher, reason = detect_sandbox_backend(env)
    err = (
        f"CC_AGY_WORKER_SANDBOX=required, and no OS sandbox is available ({reason}). {_SANDBOX_REMEDIATION}"
        if (backend is None and mode == "required")
        else None
    )
    return {
        "ok": backend is not None or mode == "auto",
        "active": backend is not None,
        "mode": mode,
        "backend": backend,
        "launcher": launcher,
        "reason": reason or None,
        "error": err,
    }


def clear_probe_cache() -> None:
    """Drop cached launcher probe results (used by tests)."""
    with _PROBE_LOCK:
        _PROBE_CACHE.clear()
