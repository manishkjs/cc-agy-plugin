"""Per-harness contract tests: one plugin, five host loaders.

cc-agy-plugin ships to Claude Code, Cursor, Codex, OpenCode, and Antigravity. Each host
loads it differently, and each has loader quirks that were only found in live testing.
These tests pin every host's loading contract so a fix for one host cannot silently
break another.

The plugin is built in three layers:

1. Layer 1: the shared stdio MCP server (`server/main.py`, started directly or through
   `scripts/run_mcp_server.sh`). Every host must launch this same server.
2. Layer 2: per-host adapter files (manifests, MCP entries, hooks, entry points).
3. Layer 3: `server/install.py --host <h>`, which writes a host's user-level config.

Every test is parametrized by host (or named after one), so CI can run one host per job
with `pytest tests/test_harness_contracts.py -k <host>`.

All tests are hermetic: fake HOME and XDG directories, a fake `claude` CLI, no network,
no model calls, and no real host CLIs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import pytest

from server import install

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_MAIN = REPO_ROOT / "server" / "main.py"
SHARED_LAUNCHER = REPO_ROOT / "scripts" / "run_mcp_server.sh"
HARNESSES = ("claude", "cursor", "codex", "opencode", "antigravity")
# A placeholder credential that lets the installer and server pick `api_key` mode.
# It is never sent anywhere: no test here makes a model call.
_DUMMY_API_KEY = "contract-test-dummy-key-0000"
_DEAD_PIP_INDEX = "http://127.0.0.1:9/simple"

# `--host` value that `server/install.py` uses for each harness.
_INSTALL_HOST = {h: h for h in HARNESSES}

# Strings that identify one harness's adapter directory. Another harness's adapter files
# must never contain them (the installer and build backend are Layer 3 and exempt).
_ADAPTER_MARKERS: dict[str, tuple[str, ...]] = {
    "claude": (".claude-plugin", "CLAUDE_PLUGIN_ROOT"),
    "cursor": (".cursor-plugin", "CURSOR_PLUGIN_ROOT", "cursor/rules"),
    "codex": (".codex-plugin", "plugins/cc-agy-plugin", ".agents/plugins"),
    "opencode": ("opencode-plugin", ".opencode/", "@opencode-ai/plugin"),
    "antigravity": (".gemini/config", "mcp_config.json"),
}


def _adapter_files(harness: str) -> list[Path]:
    """Return the Layer 2 files one harness owns.

    Shared content folders (`commands/`, `agents/`, `skills/`) are used by more than one
    host and are not adapter files. Antigravity's only adapter file in the repository is
    its hooks template: `server/install.py` generates the rest under `~/.gemini/config/`.
    """
    fixed: dict[str, tuple[str, ...]] = {
        "claude": (".claude-plugin/plugin.json", ".claude-plugin/marketplace.json", ".mcp.json", "hooks/hooks.json"),
        "cursor": (".cursor-plugin/plugin.json", ".cursor-plugin/mcp.json", ".cursor-plugin/hooks.json"),
        "codex": (
            "plugins/cc-agy-plugin/.codex-plugin/plugin.json",
            ".agents/plugins/marketplace.json",
            "plugins/cc-agy-plugin/hooks/codex-hooks.json",
        ),
        "opencode": ("package.json",),
        "antigravity": ("plugins/antigravity/hooks.json",),
    }
    globs: dict[str, tuple[str, ...]] = {
        "cursor": ("cursor/rules/*.mdc",),
        "opencode": ("opencode-plugin/*.ts", ".opencode/**/*.md"),
    }
    files = [REPO_ROOT / rel for rel in fixed[harness]]
    for pattern in globs.get(harness, ()):
        files.extend(sorted(REPO_ROOT.glob(pattern)))
    return files


def _pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert match, "pyproject.toml has no version"
    return match.group(1)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(text)


def _frontmatter(path: Path) -> str:
    match = re.match(r"^---\r?\n(.*?)\r?\n---", path.read_text(encoding="utf-8"), re.S)
    assert match, f"{path.relative_to(REPO_ROOT)} has no YAML frontmatter"
    return match.group(1)


def _assert_skill_dir(skills_dir: Path) -> None:
    """A host loads each `skills/<name>/SKILL.md` that has `name` and `description` frontmatter."""
    skill_files = sorted(skills_dir.glob("*/SKILL.md"))
    assert skill_files, f"no */SKILL.md under {skills_dir}"
    for skill in skill_files:
        meta = _frontmatter(skill)
        assert re.search(r"^name:\s*\S", meta, re.M), f"{skill} frontmatter has no name"
        assert re.search(r"^description:\s*\S", meta, re.M), f"{skill} frontmatter has no description"


def _assert_markdown_assets(directory: Path) -> None:
    """Commands and agents are Markdown files whose frontmatter has a description."""
    files = sorted(directory.glob("*.md"))
    assert files, f"no *.md under {directory}"
    for asset in files:
        assert re.search(r"^description:\s*\S", _frontmatter(asset), re.M), f"{asset} has no description"


def _plugin_root_paths(text: str, variable: str) -> list[Path]:
    """Expand every `${VAR}/path` reference (quoted or not) to a repository path."""
    pattern = r"\$\{" + re.escape(variable) + r"\}\"?/([A-Za-z0-9_./-]+)"
    return [REPO_ROOT / rel for rel in re.findall(pattern, text)]


# --------------------------------------------------------------------------------------
# Hermetic sandbox and installer runs (Layer 3)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Sandbox:
    """A fake user account: HOME, XDG dirs, and a PATH with only `git` and a fake `claude`."""

    root: Path

    @property
    def home(self) -> Path:
        return self.root / "home"

    @property
    def xdg_data(self) -> Path:
        return self.root / "xdg_data"

    @property
    def xdg_config(self) -> Path:
        return self.root / "xdg_config"

    @property
    def xdg_cache(self) -> Path:
        return self.root / "xdg_cache"

    @property
    def bin_dir(self) -> Path:
        return self.root / "bin"

    @property
    def claude_log(self) -> Path:
        return self.root / "fake_claude_calls.log"

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    def watched_dirs(self) -> tuple[Path, ...]:
        return (self.home, self.xdg_data, self.xdg_config, self.xdg_cache)

    def env(self, *, path: str | None = None) -> dict[str, str]:
        """Return a from-scratch environment; nothing leaks in from the real user."""
        return {
            "HOME": str(self.home),
            "XDG_DATA_HOME": str(self.xdg_data),
            "XDG_CONFIG_HOME": str(self.xdg_config),
            "XDG_CACHE_HOME": str(self.xdg_cache),
            # Only `git` and the fake `claude`, so no real host CLI (claude, codex, opencode,
            # agy) can ever run, even if one is installed on this machine.
            "PATH": path or str(self.bin_dir),
            "LANG": "C.UTF-8",
            "GEMINI_API_KEY": _DUMMY_API_KEY,
            "CC_AGY_FAKE_CLAUDE_LOG": str(self.claude_log),
            "CC_AGY_STATE_DIR": str(self.root / "agy-state"),
            "CC_AGY_MODEL_CATALOG": "0",
            "CC_AGY_AUTO_BOOTSTRAP": "0",
            "CC_AGY_PYTHON_BIN": sys.executable,
            "PIP_INDEX_URL": _DEAD_PIP_INDEX,
            "PIP_NO_INDEX": "1",
            "PIP_RETRIES": "0",
        }


def _make_sandbox(root: Path) -> _Sandbox:
    sandbox = _Sandbox(root)
    for directory in (*sandbox.watched_dirs(), sandbox.bin_dir, sandbox.workspace):
        directory.mkdir(parents=True, exist_ok=True)
    git_bin = shutil.which("git")
    if git_bin:
        (sandbox.bin_dir / "git").symlink_to(git_bin)
    fake_claude = sandbox.bin_dir / "claude"
    fake_claude.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$CC_AGY_FAKE_CLAUDE_LOG"\necho ok\n', encoding="utf-8"
    )
    fake_claude.chmod(0o755)
    return sandbox


def _snapshot(sandbox: _Sandbox) -> set[str]:
    """Every file, directory, and symlink under the sandbox's HOME and XDG dirs."""
    entries: set[str] = set()
    for base in sandbox.watched_dirs():
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            for name in (*dirnames, *filenames):
                entries.add(str((Path(dirpath) / name).relative_to(sandbox.root)))
    return entries


def _host_config_roots(harness: str, sandbox: _Sandbox) -> tuple[Path, ...]:
    """Directories `install.py --host <harness>` is allowed to write into."""
    roots: dict[str, tuple[Path, ...]] = {
        # Claude Code registers the plugin through its own CLI and never by writing files.
        "claude": (),
        "codex": (sandbox.home / ".codex",),
        "cursor": (sandbox.home / ".cursor",),
        "opencode": (sandbox.xdg_config / "opencode",),
        "antigravity": (sandbox.home / ".gemini" / "config",),
    }
    return roots[harness]


def _shared_state_root(sandbox: _Sandbox) -> Path:
    """Host-neutral plugin state every install may write (`last-install.json`, `current`)."""
    return sandbox.xdg_data / "cc-agy-plugin"


@dataclass(frozen=True)
class _InstallRun:
    sandbox: _Sandbox
    before: set[str]
    after: set[str]
    report: dict[str, Any]

    @property
    def host(self) -> dict[str, Any]:
        hosts = self.report["hosts"]
        assert len(hosts) == 1, f"expected exactly one host report, got {[h.get('host') for h in hosts]}"
        return hosts[0]

    @property
    def created(self) -> set[str]:
        return self.after - self.before


@pytest.fixture(scope="module")
def installed(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., _InstallRun]:
    """Run `server/install.py --host <harness>` once per (harness, dry_run) in a fresh sandbox."""
    cache: dict[tuple[str, bool], _InstallRun] = {}

    def run(harness: str, *, dry_run: bool = False) -> _InstallRun:
        key = (harness, dry_run)
        if key not in cache:
            sandbox = _make_sandbox(tmp_path_factory.mktemp(f"{harness}-{'dry' if dry_run else 'real'}"))
            before = _snapshot(sandbox)
            argv = [
                sys.executable, "-m", "server.install",
                "--host", _INSTALL_HOST[harness],
                "--auth", "api-key",
                "--no-bootstrap", "--non-interactive", "--json",
                *(["--dry-run"] if dry_run else []),
            ]
            proc = subprocess.run(
                argv, cwd=REPO_ROOT, env=sandbox.env(), capture_output=True, text=True, timeout=180, check=False
            )
            try:
                report = json.loads(proc.stdout)
            except json.JSONDecodeError:
                pytest.fail(f"install.py printed no JSON (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
            cache[key] = _InstallRun(sandbox, before, _snapshot(sandbox), report)
        return cache[key]

    return run


# --------------------------------------------------------------------------------------
# MCP launch specs and stdio handshake (Layer 1)
# --------------------------------------------------------------------------------------


def _expand(value: str, variable: str) -> str:
    return value.replace("${" + variable + "}", str(REPO_ROOT))


def _launch_spec(harness: str, installed: Callable[..., _InstallRun]) -> tuple[list[str], dict[str, str]]:
    """Return the argv and extra env a harness uses to start the MCP server.

    Claude Code and Cursor read the repository's own MCP manifests. Codex, OpenCode, and
    Antigravity read what `install.py` wrote into the user's config.
    """
    if harness == "claude":
        entry = _read_json(REPO_ROOT / ".mcp.json")["mcpServers"]["vertex"]
        argv = [entry["command"], *(_expand(a, "CLAUDE_PLUGIN_ROOT") for a in entry["args"])]
        # Values stay as unsubstituted `${user_config.X}` placeholders, which is what the
        # server sees when the user has not set them.
        return argv, dict(entry.get("env") or {})
    if harness == "cursor":
        entry = _read_json(REPO_ROOT / ".cursor-plugin" / "mcp.json")["mcpServers"]["agy"]
        argv = [entry["command"], *(_expand(a, "CURSOR_PLUGIN_ROOT") for a in entry["args"])]
        return argv, dict(entry.get("env") or {})
    sandbox = installed(harness).sandbox
    if harness == "codex":
        table = _load_toml((sandbox.home / ".codex" / "config.toml").read_text(encoding="utf-8"))
        entry = table["mcp_servers"]["agy"]
        return [entry["command"], *entry["args"]], dict(entry.get("env") or {})
    if harness == "opencode":
        entry = _read_json(sandbox.xdg_config / "opencode" / "opencode.json")["mcp"]["agy"]
        return list(entry["command"]), dict(entry.get("environment") or {})
    entry = _read_json(sandbox.home / ".gemini" / "config" / "mcp_config.json")["mcpServers"]["agy"]
    return [entry["command"], *entry["args"]], dict(entry.get("env") or {})


def _resolves_to(arg: str, target: Path) -> bool:
    try:
        return Path(arg).resolve() == target.resolve()
    except (OSError, ValueError):
        return False


def _mcp_handshake(argv: list[str], env: Mapping[str, str], cwd: Path) -> tuple[dict[str, Any], list[str]]:
    """Send `initialize` and `tools/list` over stdio; return (serverInfo, tool names)."""
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "harness-contract-test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    # Closing stdin after the requests makes the server exit; the timeout kills it otherwise.
    proc = subprocess.run(
        argv,
        input="".join(json.dumps(r) + "\n" for r in requests),
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    replies: dict[Any, dict[str, Any]] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            message = json.loads(line)
            replies[message.get("id")] = message
    if 2 not in replies and "ModuleNotFoundError" in proc.stderr:
        pytest.skip(f"MCP server needs Python packages that are not installed here:\n{proc.stderr[-800:]}")
    assert 1 in replies and 2 in replies, f"no MCP replies (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
    assert "error" not in replies[2], replies[2]
    return replies[1]["result"]["serverInfo"], [tool["name"] for tool in replies[2]["result"]["tools"]]


def _server_env(sandbox: _Sandbox, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = sandbox.env(path=os.pathsep.join([str(sandbox.bin_dir), "/usr/bin", "/bin"]))
    env["CC_AGY_AUTH_MODE"] = "api_key"
    env.update(extra or {})
    return env


@pytest.fixture(scope="module")
def baseline_tools(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], list[str]]:
    """The Layer 1 reference: tools the shared server lists when started directly."""
    sandbox = _make_sandbox(tmp_path_factory.mktemp("baseline"))
    return _mcp_handshake([sys.executable, str(SERVER_MAIN)], _server_env(sandbox), sandbox.workspace)


# --------------------------------------------------------------------------------------
# (a) Manifests parse and every path they reference exists
# --------------------------------------------------------------------------------------


def _check_claude_manifest() -> None:
    plugin = _read_json(REPO_ROOT / ".claude-plugin" / "plugin.json")
    market = _read_json(REPO_ROOT / ".claude-plugin" / "marketplace.json")
    entry = next(p for p in market["plugins"] if p["name"] == plugin["name"])
    source = (REPO_ROOT / entry["source"]).resolve()
    assert (source / ".claude-plugin" / "plugin.json").is_file(), f"marketplace source {source} has no plugin.json"
    for key in ("commands", "agents", "skills", "hooks", "mcpServers"):
        value = plugin.get(key)
        for rel in [value] if isinstance(value, str) else (value if isinstance(value, list) else []):
            assert (source / rel).exists(), f"plugin.json {key} -> {rel} is missing"
    # Claude Code also auto-discovers these default locations from the plugin root.
    _assert_markdown_assets(source / "commands")
    _assert_markdown_assets(source / "agents")
    _assert_skill_dir(source / "skills")
    for manifest in (source / ".mcp.json", source / "hooks" / "hooks.json"):
        text = manifest.read_text(encoding="utf-8")
        json.loads(text)
        for path in _plugin_root_paths(text, "CLAUDE_PLUGIN_ROOT"):
            assert path.is_file(), f"{manifest.name} references missing {path.relative_to(REPO_ROOT)}"


def _check_cursor_manifest() -> None:
    plugin = _read_json(REPO_ROOT / ".cursor-plugin" / "plugin.json")
    for key in ("skills", "agents", "commands", "mcpServers", "hooks"):
        assert (REPO_ROOT / plugin[key]).exists(), f".cursor-plugin/plugin.json {key} -> {plugin[key]} is missing"
    _assert_skill_dir(REPO_ROOT / plugin["skills"])
    _assert_markdown_assets(REPO_ROOT / plugin["agents"])
    _assert_markdown_assets(REPO_ROOT / plugin["commands"])
    for rel in (plugin["mcpServers"], plugin["hooks"]):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        json.loads(text)
        for path in _plugin_root_paths(text, "CURSOR_PLUGIN_ROOT"):
            assert path.is_file(), f"{rel} references missing {path.relative_to(REPO_ROOT)}"
    rules = sorted((REPO_ROOT / "cursor" / "rules").glob("*.mdc"))
    assert rules, "cursor/rules/*.mdc is missing (install.py copies it into ~/.cursor/rules)"
    for rule in rules:
        assert re.search(r"^description:\s*\S", _frontmatter(rule), re.M), f"{rule} has no description"


def _check_codex_manifest() -> None:
    market = _read_json(REPO_ROOT / ".agents" / "plugins" / "marketplace.json")
    entry = next(p for p in market["plugins"] if p["name"] == "cc-agy-plugin")
    plugin_dir = (REPO_ROOT / entry["source"]["path"]).resolve()
    plugin = _read_json(plugin_dir / ".codex-plugin" / "plugin.json")
    assert plugin["name"] == entry["name"]
    _assert_skill_dir(plugin_dir / plugin["skills"])
    assert install.CODEX_SKILL_SOURCE.is_file(), "install.py's Codex skill source is missing"


def _check_opencode_manifest() -> None:
    package = _read_json(REPO_ROOT / "package.json")
    assert (REPO_ROOT / package["main"]).is_file(), f"package.json main -> {package['main']} is missing"
    for target in (package.get("exports") or {}).values():
        assert (REPO_ROOT / target).is_file(), f"package.json exports -> {target} is missing"
    for rel in package.get("files") or []:
        assert (REPO_ROOT / rel).exists(), f"package.json files -> {rel} is missing"
    entry_dir = (REPO_ROOT / package["main"]).parent
    for ts_file in sorted(entry_dir.glob("*.ts")):
        for spec in re.findall(r'from\s+"(\.{1,2}/[^"]+)"', ts_file.read_text(encoding="utf-8")):
            assert (ts_file.parent / spec).is_file(), f"{ts_file.name} imports missing {spec}"
    commands = REPO_ROOT / ".opencode" / "commands"
    _assert_markdown_assets(commands)
    _assert_markdown_assets(REPO_ROOT / ".opencode" / "agents")
    _assert_skill_dir(REPO_ROOT / ".opencode" / "skills")
    for name in install._REQUIRED_OPENCODE_COMMANDS:
        assert (commands / f"{name}.md").is_file(), f".opencode/commands/{name}.md is missing"


def _check_antigravity_manifest(installed: Callable[..., _InstallRun]) -> None:
    # No adapter files live in the repository; check the bundle install.py generates.
    plugin_dir = installed("antigravity").sandbox.home / ".gemini" / "config" / "plugins" / "cc-agy-plugin"
    plugin = _read_json(plugin_dir / "plugin.json")
    assert plugin["name"] == "cc-agy-plugin"
    bundle_mcp = _read_json(plugin_dir / "mcp_config.json")["mcpServers"]["agy"]
    assert any(_resolves_to(arg, SERVER_MAIN) for arg in bundle_mcp["args"]), bundle_mcp
    _assert_skill_dir(plugin_dir / "skills")


@pytest.mark.parametrize("harness", HARNESSES)
def test_manifest_parses_and_referenced_paths_exist(harness: str, installed: Callable[..., _InstallRun]) -> None:
    checks: dict[str, Callable[[], None]] = {
        "claude": _check_claude_manifest,
        "cursor": _check_cursor_manifest,
        "codex": _check_codex_manifest,
        "opencode": _check_opencode_manifest,
        "antigravity": lambda: _check_antigravity_manifest(installed),
    }
    checks[harness]()


# --------------------------------------------------------------------------------------
# (b) Every harness launches the same Layer 1 server
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("harness", HARNESSES)
def test_mcp_launch_spec_targets_the_shared_server(harness: str, installed: Callable[..., _InstallRun]) -> None:
    argv, _env = _launch_spec(harness, installed)

    via_launcher = any(_resolves_to(arg, SHARED_LAUNCHER) for arg in argv[1:])
    direct = any(_resolves_to(arg, SERVER_MAIN) for arg in argv[1:])

    assert via_launcher or direct, f"{harness} launches {argv}, not the shared server"
    if via_launcher:
        assert argv[0] == "sh", f"{harness} must start the POSIX launcher with sh: {argv}"
        assert 'server_main="${cc_agy_root}/server/main.py"' in SHARED_LAUNCHER.read_text(encoding="utf-8")
    else:
        assert Path(argv[0]).name.startswith("python"), f"{harness} starts server/main.py with {argv[0]}"


# --------------------------------------------------------------------------------------
# (c) `install.py --host <h>` touches only that host's config
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("harness", HARNESSES)
def test_installer_writes_only_its_own_host_config(harness: str, installed: Callable[..., _InstallRun]) -> None:
    run = installed(harness)
    sandbox = run.sandbox
    own_roots = _host_config_roots(harness, sandbox)
    allowed = (*own_roots, _shared_state_root(sandbox))

    def is_allowed(rel: str) -> bool:
        path = sandbox.root / rel
        return any(path == root or root in path.parents or path in root.parents for root in allowed)

    stray = sorted(rel for rel in run.created if not is_allowed(rel))

    assert run.host["status"] == "installed", run.host
    assert not stray, f"--host {harness} wrote outside its own config: {stray}"
    for root in own_roots:
        assert any((sandbox.root / rel) == root or root in (sandbox.root / rel).parents for rel in run.created), (
            f"--host {harness} wrote nothing under {root.relative_to(sandbox.root)}"
        )
    other_roots = {
        other: _host_config_roots(other, sandbox) for other in HARNESSES if other != harness
    }
    for other, roots in other_roots.items():
        for root in roots:
            assert not root.exists(), f"--host {harness} created {other}'s config dir {root.relative_to(sandbox.root)}"
    if harness != "claude":
        assert not sandbox.claude_log.exists(), f"--host {harness} ran the claude CLI"


@pytest.mark.parametrize("harness", HARNESSES)
def test_installer_dry_run_writes_nothing(harness: str, installed: Callable[..., _InstallRun]) -> None:
    run = installed(harness, dry_run=True)

    assert run.host["status"] == "dry_run", run.host
    assert run.created == set(), f"--host {harness} --dry-run wrote {sorted(run.created)}"
    assert not run.sandbox.claude_log.exists(), f"--host {harness} --dry-run ran the claude CLI"


# --------------------------------------------------------------------------------------
# (d) Harness-specific loader quirks
# --------------------------------------------------------------------------------------


def _strip_ts_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", source)


def test_opencode_entry_exports_only_the_default_factory() -> None:
    """OpenCode (1.19.6+) treats every export of the entry file as a plugin factory."""
    entry_rel = _read_json(REPO_ROOT / "package.json")["main"]
    entry = _strip_ts_comments((REPO_ROOT / entry_rel).read_text(encoding="utf-8"))
    internal = _strip_ts_comments((REPO_ROOT / "opencode-plugin" / "internal.ts").read_text(encoding="utf-8"))

    exports = [line.strip().rstrip(";") for line in re.findall(r"(?m)^\s*export\b[^\n]*", entry)]

    assert "module.exports" not in entry and "exports." not in entry
    assert len(exports) == 1, f"{entry_rel} must export exactly one value, found {exports}"
    match = re.fullmatch(r"export default ([A-Za-z_$][\w$]*)", exports[0])
    assert match, f"{entry_rel} must `export default <factory>`, found {exports[0]!r}"
    factory = match.group(1)
    assert re.search(r"import\s*\{\s*" + re.escape(factory) + r"\s*\}\s*from\s*\"\./internal\.ts\"", entry)
    assert re.search(r"export const " + re.escape(factory) + r"\s*:\s*Plugin\s*=\s*async\s*\(", internal), (
        f"{factory} in internal.ts is not an async plugin factory function"
    )


def _node_candidates() -> list[str]:
    """Every distinct `node` executable on PATH, in PATH order.

    Some distro builds of Node 22+ ship without TypeScript support (`ERR_NO_TYPESCRIPT`),
    and the test conftest puts the Python interpreter's directory (often `/usr/bin`) first
    on PATH, so the first `node` found is not necessarily one that can load the entry file.
    """
    found: list[str] = []
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = shutil.which("node", path=directory) if directory else None
        if candidate:
            resolved = str(Path(candidate).resolve())
            if resolved not in {str(Path(f).resolve()) for f in found}:
                found.append(candidate)
    return found


def _node_ts_command() -> list[str] | None:
    """Return a `node` command that can import TypeScript by stripping types, or None."""
    probe = f"await import({json.dumps((REPO_ROOT / 'opencode-plugin' / 'index.ts').as_uri())}); console.log('ok')"
    for node in _node_candidates():
        for flags in ([], ["--experimental-strip-types"]):
            cmd = [node, "--no-warnings", *flags]
            try:
                proc = subprocess.run(
                    [*cmd, "--input-type=module", "-e", probe],
                    capture_output=True, text=True, timeout=60, check=False,
                    env={"PATH": os.environ.get("PATH", ""), "HOME": str(Path.cwd())},
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if proc.returncode == 0 and proc.stdout.strip().endswith("ok"):
                return cmd
    return None


def test_opencode_entry_loads_in_node_and_registers_the_shared_server(tmp_path: Path) -> None:
    node_cmd = _node_ts_command()
    if node_cmd is None:
        pytest.skip("no node on PATH can import TypeScript (needs Node 22.6+ built with type stripping)")
    home = tmp_path / "home"
    home.mkdir()
    entry_url = (REPO_ROOT / "opencode-plugin" / "index.ts").as_uri()
    internal_url = (REPO_ROOT / "opencode-plugin" / "internal.ts").as_uri()
    fake_env = {"HOME": str(home), "XDG_DATA_HOME": str(tmp_path / "xdg_data")}
    script = (
        f"const entry = await import({json.dumps(entry_url)});\n"
        f"const internal = await import({json.dumps(internal_url)});\n"
        "const hooks = await entry.default({});\n"
        "const cfg = {};\n"
        f"internal.applyConfig(cfg, {json.dumps(fake_env)});\n"
        "console.log(JSON.stringify({exports: Object.keys(entry), defaultType: typeof entry.default,\n"
        "  hooks: Object.keys(hooks), provider: hooks.auth && hooks.auth.provider, mcp: cfg.mcp.agy,\n"
        "  commands: Object.keys(cfg.command), agents: Object.keys(cfg.agent), instructions: cfg.instructions}));\n"
    )

    proc = subprocess.run(
        [*node_cmd, "--input-type=module", "-e", script],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False,
        env={"PATH": os.environ.get("PATH", ""), **fake_env},
    )
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])

    assert loaded["exports"] == ["default"] and loaded["defaultType"] == "function"
    assert {"auth", "config"} <= set(loaded["hooks"]) and loaded["provider"] == "cc-agy"
    assert loaded["mcp"]["type"] == "local" and _resolves_to(loaded["mcp"]["command"][1], SERVER_MAIN)
    assert set(install._REQUIRED_OPENCODE_COMMANDS) <= set(loaded["commands"])
    assert "gemini-delegator" in loaded["agents"]
    assert any(_resolves_to(p, REPO_ROOT / ".opencode" / "skills" / "cc-agy-plugin" / "SKILL.md")
               for p in loaded["instructions"])


def test_opencode_auth_methods_ask_only_for_what_they_need(tmp_path: Path) -> None:
    """OpenCode 1.19.6 appends a required "Enter your API key" prompt to every `type: "api"` method.

    Only the Gemini API key method may be `type: "api"`, and it must not ask for the key a second
    time. The other methods are empty-url "auto" oauth steps that open no browser, and what they
    save reads back exactly like entries saved by earlier plugin versions (`type: "api"`).
    """
    node_cmd = _node_ts_command()
    if node_cmd is None:
        pytest.skip("no node on PATH can import TypeScript (needs Node 22.6+ built with type stripping)")
    internal_url = (REPO_ROOT / "opencode-plugin" / "internal.ts").as_uri()
    entry_url = (REPO_ROOT / "opencode-plugin" / "index.ts").as_uri()
    filled = {
        "Gemini Enterprise seat": {},
        "Google Cloud Vertex AI project": {"project": "my-proj", "location": "us-central1"},
        "AI Gateway": {"baseUrl": "https://gw.example.com/vertex_ai/", "gatewayToken": "gw-token"},
    }
    legacy_seat = {"type": "api", "key": "gemini-enterprise-seat",
                   "metadata": {"authMode": "gemini_enterprise", "bifrostPolicyUrl": ""}}
    script = (
        f"const m = await import({json.dumps(internal_url)});\n"
        f"const entry = await import({json.dumps(entry_url)});\n"
        f"const filled = {json.dumps(filled)};\n"
        "const hooks = await entry.default({});\n"
        "const methods = [];\n"
        "for (const method of hooks.auth.methods) {\n"
        "  const info = {type: method.type, label: method.label,\n"
        "    prompts: (method.prompts || []).map((p) => ({key: p.key, message: p.message}))};\n"
        "  if (method.type === 'oauth') {\n"
        "    const prefix = Object.keys(filled).find((k) => method.label.startsWith(k));\n"
        "    const step = await method.authorize(filled[prefix] || {});\n"
        "    const result = await step.callback();\n"
        "    // What OpenCode then writes to auth.json for a key result.\n"
        "    const saved = {type: 'api', key: result.key, metadata: result.metadata};\n"
        "    Object.assign(info, {url: step.url, method: step.method, result, env: m.buildServerEnvironment(saved),\n"
        "      emptyResult: await (await method.authorize({})).callback()});\n"
        "  }\n"
        "  methods.push(info);\n"
        "}\n"
        f"const legacyEnv = m.buildServerEnvironment({json.dumps(legacy_seat)});\n"
        "const typedKeyEnv = m.buildServerEnvironment({type: 'api', key: ' typed-gemini-key '});\n"
        "console.log(JSON.stringify({methods, legacyEnv, typedKeyEnv}));\n"
    )

    proc = subprocess.run(
        [*node_cmd, "--input-type=module", "-e", script],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "XDG_DATA_HOME": str(tmp_path / "data")},
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    by_label = {m["label"].split(" (")[0]: m for m in out["methods"]}

    api_methods = [m for m in out["methods"] if m["type"] == "api"]
    assert [m["label"] for m in api_methods] == ["Gemini API key (Google AI Studio)"]
    for method in api_methods:
        own_key_prompts = [p for p in method["prompts"] if re.search(r"api.?key", f"{p['key']} {p['message']}", re.I)]
        assert not own_key_prompts, f"{method['label']} asks for the API key twice: {own_key_prompts}"
    for name in filled:
        method = by_label[name]
        assert method["type"] == "oauth", f"{name} must not be type api (OpenCode would ask for an API key)"
        assert method["url"] == "" and method["method"] == "auto", f"{name} must not send the user to a browser"
        assert method["result"]["type"] == "success" and method["result"]["key"], method["result"]
    assert by_label["Google Cloud Vertex AI project"]["emptyResult"] == {"type": "failed"}
    assert by_label["AI Gateway"]["emptyResult"] == {"type": "failed"}
    # A seat entry saved by an older plugin version (type "api") maps to the same MCP env as a new one.
    assert out["legacyEnv"] == by_label["Gemini Enterprise seat"]["env"] == {"CC_AGY_AUTH_MODE": "gemini_enterprise"}
    assert by_label["Google Cloud Vertex AI project"]["env"] == {
        "CC_AGY_AUTH_MODE": "vertex", "GOOGLE_CLOUD_PROJECT": "my-proj", "GOOGLE_CLOUD_LOCATION": "us-central1",
    }
    assert by_label["AI Gateway"]["env"] == {
        "CC_AGY_AUTH_MODE": "gateway", "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com/vertex_ai",
        "CC_AGY_VERTEX_BASE_URL_TOKEN": "gw-token",
    }
    # The key typed at OpenCode's own prompt is saved without metadata and still selects api_key mode.
    assert out["typedKeyEnv"] == {"CC_AGY_AUTH_MODE": "api_key", "GEMINI_API_KEY": "typed-gemini-key"}


def _claude_command_index(cmds: list[list[str]], head: list[str]) -> int:
    matches = [idx for idx, cmd in enumerate(cmds) if cmd[: len(head)] == head]
    assert len(matches) == 1, f"expected one `{' '.join(head)}` command, got {matches}"
    return matches[0]


def test_claude_install_refreshes_the_local_marketplace_cache(installed: Callable[..., _InstallRun]) -> None:
    """`claude plugin install` does not refresh an installed local-marketplace cache by itself."""
    cmds = install.claude_install_commands(profile="balanced", auth_mode="api-key")
    add = _claude_command_index(cmds, ["claude", "plugin", "marketplace", "add"])
    refresh = _claude_command_index(cmds, ["claude", "plugin", "marketplace", "update"])
    plugin_install = _claude_command_index(cmds, ["claude", "plugin", "install"])
    plugin_update = _claude_command_index(cmds, ["claude", "plugin", "update"])

    run = installed("claude")
    calls = run.sandbox.claude_log.read_text(encoding="utf-8").splitlines()

    assert add < refresh < plugin_install < plugin_update
    assert cmds[add][4] == str(install.PLUGIN_ROOT)
    assert calls == [" ".join(cmd[1:]) for cmd in cmds], "the installer must run exactly the planned commands"
    assert run.host["status"] == "installed"


def test_claude_user_config_placeholders_match_plugin_user_config() -> None:
    user_config = _read_json(REPO_ROOT / ".claude-plugin" / "plugin.json")["userConfig"]
    referenced: set[str] = set()
    for manifest in (".mcp.json", "hooks/hooks.json"):
        referenced |= set(re.findall(r"\$\{user_config\.([A-Za-z0-9_]+)\}", (REPO_ROOT / manifest).read_text(encoding="utf-8")))

    missing = sorted(referenced - set(user_config))
    unused = sorted(set(user_config) - referenced)

    assert not missing, f"${{user_config.X}} keys missing from plugin.json userConfig: {missing}"
    assert not unused, f"plugin.json userConfig keys never forwarded to the MCP server: {unused}"
    for key, spec in user_config.items():
        assert spec.get("type") in ("string", "boolean", "number"), f"userConfig.{key} has type {spec.get('type')!r}"
        if re.search(r"(key|token|password|secret)", key):
            assert spec.get("sensitive") is True, f"userConfig.{key} holds a secret but is not sensitive"


def test_codex_installed_toml_block_parses(installed: Callable[..., _InstallRun]) -> None:
    run = installed("codex")
    config = _load_toml((run.sandbox.home / ".codex" / "config.toml").read_text(encoding="utf-8"))
    server = config["mcp_servers"]["agy"]

    assert [_resolves_to(arg, SERVER_MAIN) for arg in server["args"]] == [True]
    assert server["env"]["CC_AGY_AUTH_MODE"] == "api_key"
    assert "GEMINI_API_KEY" in server["env_vars"], "secrets must be forwarded, never written to config.toml"
    assert _DUMMY_API_KEY not in (run.sandbox.home / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert server["tool_timeout_sec"] == install.CODEX_TOOL_TIMEOUT_SEC
    assert server["startup_timeout_sec"] == install.CODEX_STARTUP_TIMEOUT_SEC
    installed_skill = run.sandbox.home / ".codex" / "skills" / "cc-agy-plugin" / "SKILL.md"
    assert installed_skill.read_text(encoding="utf-8") == install.CODEX_SKILL_SOURCE.read_text(encoding="utf-8")


def test_cursor_mcp_json_registers_the_agy_server(installed: Callable[..., _InstallRun]) -> None:
    native = _read_json(REPO_ROOT / ".cursor-plugin" / "mcp.json")["mcpServers"]
    run = installed("cursor")
    user_level = _read_json(run.sandbox.home / ".cursor" / "mcp.json")["mcpServers"]

    assert set(native) == {"agy"}
    assert "agy" in user_level
    assert any(_resolves_to(arg, SERVER_MAIN) for arg in user_level["agy"]["args"]), user_level["agy"]
    assert _DUMMY_API_KEY not in (run.sandbox.home / ".cursor" / "mcp.json").read_text(encoding="utf-8")
    assert (run.sandbox.home / ".cursor" / "rules" / "cc-agy-coexistence.mdc").is_file()


def test_antigravity_mcp_config_registers_the_agy_server(installed: Callable[..., _InstallRun]) -> None:
    run = installed("antigravity")
    cfg_path = run.sandbox.home / ".gemini" / "config" / "mcp_config.json"
    server = _read_json(cfg_path)["mcpServers"]["agy"]

    assert any(_resolves_to(arg, SERVER_MAIN) for arg in server["args"]), server
    assert server["env"]["CC_AGY_AUTH_MODE"] == "api_key"
    assert _DUMMY_API_KEY not in cfg_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# (e) Every manifest carries the pyproject.toml version
# --------------------------------------------------------------------------------------


def _manifest_versions(harness: str, installed: Callable[..., _InstallRun]) -> dict[str, str]:
    if harness == "claude":
        market = _read_json(REPO_ROOT / ".claude-plugin" / "marketplace.json")
        return {
            ".claude-plugin/plugin.json": _read_json(REPO_ROOT / ".claude-plugin" / "plugin.json")["version"],
            ".claude-plugin/marketplace.json": market["plugins"][0]["version"],
        }
    if harness == "cursor":
        return {".cursor-plugin/plugin.json": _read_json(REPO_ROOT / ".cursor-plugin" / "plugin.json")["version"]}
    if harness == "codex":
        rel = "plugins/cc-agy-plugin/.codex-plugin/plugin.json"
        return {rel: _read_json(REPO_ROOT / rel)["version"]}
    if harness == "opencode":
        return {"package.json": _read_json(REPO_ROOT / "package.json")["version"]}
    bundle = installed("antigravity").sandbox.home / ".gemini" / "config" / "plugins" / "cc-agy-plugin" / "plugin.json"
    return {"~/.gemini/config/plugins/cc-agy-plugin/plugin.json": _read_json(bundle)["version"]}


@pytest.mark.parametrize("harness", HARNESSES)
def test_manifest_versions_match_pyproject(harness: str, installed: Callable[..., _InstallRun]) -> None:
    expected = _pyproject_version()

    stale = {name: ver for name, ver in _manifest_versions(harness, installed).items() if ver != expected}

    assert not stale, f"{harness} manifests out of sync with pyproject.toml {expected}: {stale}"


# --------------------------------------------------------------------------------------
# (f) Adapters stay in their own folder
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("harness", HARNESSES)
def test_adapter_files_do_not_reference_other_harnesses(harness: str) -> None:
    violations: list[str] = []
    # This harness's adapter files must not point into another harness's adapter folder ...
    for path in _adapter_files(harness):
        text = path.read_text(encoding="utf-8")
        for other, markers in _ADAPTER_MARKERS.items():
            if other != harness:
                violations += [f"{path.relative_to(REPO_ROOT)} -> {other} ({m})" for m in markers if m in text]
    # ... and no other harness's adapter files may point into this one's.
    for other in HARNESSES:
        if other == harness:
            continue
        for path in _adapter_files(other):
            text = path.read_text(encoding="utf-8")
            violations += [f"{path.relative_to(REPO_ROOT)} -> {harness} ({m})" for m in _ADAPTER_MARKERS[harness] if m in text]

    assert not violations, "cross-harness adapter references:\n" + "\n".join(violations)


# --------------------------------------------------------------------------------------
# (g) The shared server answers the MCP handshake exactly as each harness launches it
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("harness", HARNESSES)
def test_shared_server_handshake_via_harness_launch_spec(
    harness: str,
    installed: Callable[..., _InstallRun],
    baseline_tools: tuple[dict[str, Any], list[str]],
    tmp_path: Path,
) -> None:
    argv, harness_env = _launch_spec(harness, installed)
    sandbox = _make_sandbox(tmp_path / "launch")

    server_info, tools = _mcp_handshake(argv, _server_env(sandbox, harness_env), sandbox.workspace)
    baseline_info, baseline = baseline_tools

    assert server_info["version"] == _pyproject_version() == baseline_info["version"]
    assert tools, f"{harness}: tools/list returned no tools"
    assert sorted(tools) == sorted(baseline), f"{harness} sees a different tool set than the shared server"


# --------------------------------------------------------------------------------------
# (h) Every harness wires the five guard hooks to the shared scripts
# --------------------------------------------------------------------------------------

_CC_HOOK_EVENTS = {"SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop"}
_HOOK_ADAPTER = REPO_ROOT / "scripts" / "hook_adapter.py"


def _hook_commands(node: Any) -> list[str]:
    if isinstance(node, dict):
        own = [node["command"]] if isinstance(node.get("command"), str) else []
        return own + [c for value in node.values() for c in _hook_commands(value)]
    if isinstance(node, list):
        return [c for item in node for c in _hook_commands(item)]
    return []


def _assert_commands_run_the_adapter(commands: list[str], host: str) -> None:
    import shlex

    assert commands, f"{host}: no hook commands"
    for command in commands:
        argv = shlex.split(command)
        assert Path(argv[0]).name.startswith("python"), command
        assert Path(argv[1]).is_absolute() and _resolves_to(argv[1], _HOOK_ADAPTER), command
        assert argv[2:4] == ["--host", host], command
    assert _HOOK_ADAPTER.is_file(), "scripts/hook_adapter.py is missing"


def _check_claude_hooks(installed: Callable[..., _InstallRun]) -> None:
    hooks = _read_json(REPO_ROOT / "hooks" / "hooks.json")["hooks"]
    assert set(hooks) == _CC_HOOK_EVENTS


def _check_cursor_hooks(installed: Callable[..., _InstallRun]) -> None:
    plugin = _read_json(REPO_ROOT / ".cursor-plugin" / "plugin.json")
    assert (REPO_ROOT / plugin["hooks"]).is_file()
    json.loads((REPO_ROOT / plugin["hooks"]).read_text(encoding="utf-8"))


def _check_codex_hooks(installed: Callable[..., _InstallRun]) -> None:
    run = installed("codex")
    codex_home = run.sandbox.home / ".codex"
    assert run.host["hooks"]["status"] == "installed", run.host.get("hooks")
    hooks = _read_json(codex_home / "hooks.json")["hooks"]
    assert set(hooks) == _CC_HOOK_EVENTS == set(install.CODEX_HOOK_EVENTS)
    for event in ("PreToolUse", "PostToolUse"):
        assert [g.get("matcher") for g in hooks[event]] == ["^Bash$"], hooks[event]
    _assert_commands_run_the_adapter(_hook_commands(hooks), "codex")
    table = _load_toml((codex_home / "config.toml").read_text(encoding="utf-8"))
    assert table["features"]["codex_hooks"] is True


def _check_opencode_hooks(installed: Callable[..., _InstallRun]) -> None:
    # Static only: the node tests in tests/test_opencode_hooks.py exercise the behavior.
    hooks_ts = REPO_ROOT / "opencode-plugin" / "hooks.ts"
    assert hooks_ts.is_file(), "opencode-plugin/hooks.ts is missing"
    source = _strip_ts_comments(hooks_ts.read_text(encoding="utf-8"))
    for key in ('"tool.execute.before"', '"tool.execute.after"', '"chat.message"', "event:"):
        assert key in source, f"opencode-plugin/hooks.ts does not register {key}"
    assert "hook_adapter.py" in source
    internal = _strip_ts_comments((REPO_ROOT / "opencode-plugin" / "internal.ts").read_text(encoding="utf-8"))
    assert re.search(r'from\s+"\./hooks(?:\.ts)?"', internal), "internal.ts does not import ./hooks"


def _check_antigravity_hooks(installed: Callable[..., _InstallRun]) -> None:
    run = installed("antigravity")
    assert run.host["hooks"]["status"] == "installed", run.host.get("hooks")
    data = _read_json(run.sandbox.home / ".gemini" / "config" / "hooks.json")
    spec = data[install.AGY_HOOKS_KEY]
    assert set(spec) == {"SessionStart", "PreInvocation", "PreToolUse", "PostToolUse", "Stop"}
    for event in ("PreToolUse", "PostToolUse"):
        for group in spec[event]:
            re.compile(r"\A(?:" + group["matcher"] + r")\Z")  # agy anchors matchers this way.
    _assert_commands_run_the_adapter(_hook_commands(spec), "agy")


@pytest.mark.parametrize("harness", HARNESSES)
def test_guard_hooks_are_wired(harness: str, installed: Callable[..., _InstallRun]) -> None:
    checks: dict[str, Callable[[Callable[..., _InstallRun]], None]] = {
        "claude": _check_claude_hooks,
        "cursor": _check_cursor_hooks,
        "codex": _check_codex_hooks,
        "opencode": _check_opencode_hooks,
        "antigravity": _check_antigravity_hooks,
    }
    checks[harness](installed)
