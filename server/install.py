#!/usr/bin/env python3
"""One-command setup of cc-agy-plugin for Claude Code, Codex, Cursor, OpenCode, and Antigravity.

Usage (from a clone of this repository):

    python3 -m server.install --host claude   --auth gemini-enterprise
    python3 -m server.install --host codex    --auth gemini-enterprise
    python3 -m server.install --host cursor   --auth gemini-enterprise
    python3 -m server.install --host opencode --auth gemini-enterprise
    python3 -m server.install --host agy      --auth gemini-enterprise
    python3 -m server.install --host all --profile quality

Automatically sets up a PEP 668-safe per-user Python virtualenv (`~/.local/share/cc-agy-plugin/venv`)
when system Python does not already have `google-antigravity` and `google-genai`, and forwards
non-secret configuration to Codex (`~/.codex/config.toml`), Claude Code (`--config` / `userConfig`),
Cursor (`~/.cursor/mcp.json` and `~/.cursor/skills/cc-agy-plugin/SKILL.md`), OpenCode
(`~/.config/opencode/opencode.json` plus global skill, commands, and subagent), and Antigravity
(`~/.gemini/config/mcp_config.json` and `~/.gemini/config/plugins/cc-agy-plugin/`).
API keys and OAuth tokens are never written into config files on disk.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Literal
from urllib.parse import parse_qsl, unquote, urlsplit

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from server.regions import GATEWAY_WIRE_CLI_CHOICES  # noqa: E402
from server.runtime_env import ensure_sdk_venv, python_has_plugin_sdks, resolve_sdk_python  # noqa: E402

SERVER_ENTRY = PLUGIN_ROOT / "server" / "main.py"
CODEX_SKILL_SOURCE = PLUGIN_ROOT / "plugins" / "cc-agy-plugin" / "skills" / "vertex" / "SKILL.md"
OPENCODE_BUNDLE_DIR = PLUGIN_ROOT / ".opencode"
OPENCODE_SKILL_SOURCE = OPENCODE_BUNDLE_DIR / "skills" / "cc-agy-plugin" / "SKILL.md"
OPENCODE_COMMANDS_DIR = OPENCODE_BUNDLE_DIR / "commands"
OPENCODE_AGENT_SOURCE = OPENCODE_BUNDLE_DIR / "agents" / "gemini-delegator.md"
OPENCODE_SCHEMA_URL = "https://opencode.ai/config.json"
DEFAULT_SERVER_NAME = "agy"
MARKETPLACE_NAME = "cc-agy"
PLUGIN_NAME = "cc-agy-plugin"
COST_PROFILES = ("economy", "balanced", "quality")
AUTH_CHOICES = (
    "auto",
    "gemini-enterprise",
    "gemini_enterprise",
    "agy-cli",
    "agy_cli",
    "agy",
    "api-key",
    "api_key",
    "vertex",
    "gateway",
)

# Forwarded from the user's shell into the Codex-launched MCP server, so secrets never land in config.toml.
CODEX_FORWARDED_ENV_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "CLOUDSDK_CONFIG",
    "CC_AGY_AUTH_MODE",
    "CC_AGY_GEMINI_ENTERPRISE",
    "CC_AGY_CLI_BIN",
    "CC_AGY_CLI_CONFIG_DIR",
    "ANTIGRAVITY_OAUTH_TOKEN",
    "CC_AGY_VERTEX_BASE_URL",
    "GOOGLE_VERTEX_BASE_URL",
    "CC_AGY_GEMINI_BASE_URL",
    "GOOGLE_GEMINI_BASE_URL",
    "CC_AGY_GATEWAY_WIRE",
    "CC_AGY_VERTEX_BASE_URL_TOKEN",
    "CC_AGY_BIFROST_VIRTUAL_KEY",
    "CC_AGY_VERTEX_BASE_URL_AUDIENCE",
    "CC_AGY_VERTEX_OIDC",
    "CC_AGY_TELEMETRY_SINK",
    "CC_AGY_TELEMETRY_TOKEN",
    "CC_AGY_USER_ID",
    "CC_AGY_MAX_DAILY_CENTS",
    "CC_AGY_ALLOWED_PROJECTS",
    "CC_AGY_BLOCKED_PROJECTS",
    "CC_AGY_AUTO_APPLY_ON_PASS",
    "CC_AGY_ALLOW_IN_PLACE",
)
# Worker jobs run for minutes; Codex and OpenCode default MCP tool timeouts are far shorter.
CODEX_TOOL_TIMEOUT_SEC = 1800
CODEX_STARTUP_TIMEOUT_SEC = 30
OPENCODE_TOOL_TIMEOUT_MS = CODEX_TOOL_TIMEOUT_SEC * 1000
OPENCODE_STARTUP_TIMEOUT_MS = CODEX_STARTUP_TIMEOUT_SEC * 1000

_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_OPENCODE_RESERVED_SERVER_NAMES: frozenset[str] = frozenset({"servers", "timeout"})
_OPENCODE_MCP_ENTRY_KEYS: tuple[str, ...] = ("type", "command", "url", "enabled", "disabled")
_REQUIRED_OPENCODE_COMMANDS: tuple[str, ...] = ("delegate", "doctor", "jobs", "cost")
_PLUGIN_MANAGED_ENV_KEYS: frozenset[str] = frozenset(
    {
        "CC_AGY_COST_PROFILE",
        "CC_AGY_AUTH_MODE",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "CC_AGY_VERTEX_BASE_URL",
        "CC_AGY_GEMINI_BASE_URL",
        "CC_AGY_GATEWAY_WIRE",
        "CC_AGY_VERTEX_BASE_URL_AUDIENCE",
        "CC_AGY_TELEMETRY_SINK",
        "CC_AGY_CLI_BIN",
        "CC_AGY_MAX_DAILY_CENTS",
        "CC_AGY_ALLOWED_PROJECTS",
        "CC_AGY_BLOCKED_PROJECTS",
        "CC_AGY_AUTO_APPLY_ON_PASS",
        "CC_AGY_STRICT_MODE",
    }
)
_SECRET_KEY_TOKENS: frozenset[str] = frozenset(
    {
        "key",
        "token",
        "secret",
        "password",
        "passwd",
        "pwd",
        "pass",
        "pw",
        "jwt",
        "sid",
        "assertion",
        "sig",
        "signature",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "credentials",
    }
)
_SECRET_TOKEN_SUFFIXES: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "passwd",
    "apikey",
    "accesskey",
    "authkey",
    "privatekey",
    "secretkey",
    "virtualkey",
    "signature",
    "credential",
    "credentials",
)
_SECRET_COMPOUND_KEYS: frozenset[str] = frozenset(
    {
        "apikey",
        "xapikey",
        "accesstoken",
        "authtoken",
        "idtoken",
        "refreshtoken",
        "bearertoken",
        "sessiontoken",
        "clientsecret",
        "appsecret",
        "apisecret",
        "jwtsecret",
        "privatekey",
        "secretkey",
        "accesskey",
        "authkey",
        "virtualkey",
    }
)
_OPENCODE_MAX_SAFE_INT = 9_007_199_254_740_991
_OPENCODE_V2_PROTOCOLS: frozenset[str] = frozenset({"legacy", "auto"})
_OPENCODE_V2_TIMEOUT_KEYS: tuple[str, ...] = ("startup", "catalog", "execution")
_OPENCODE_V1_TO_V2_OAUTH_KEY_MAP: dict[str, str] = {
    "clientId": "client_id",
    "clientSecret": "client_secret",
    "callbackPort": "callback_port",
    "redirectUri": "redirect_uri",
}
_OPENCODE_V2_OAUTH_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "client_id",
        "client_secret",
        "scope",
        "callback_port",
        "redirect_uri",
        "auth_server_metadata_url",
    }
)
_OPENCODE_ASSET_MANIFEST_NAME = ".cc-agy-plugin-assets.json"
_OPENCODE_SERVER_REF_RE = re.compile(
    r"(?P<backtick_srv>`agy` MCP server)"
    r"|(?P<backtick_tools>`agy` MCP tools)"
    r"|(?P<via_srv>via the agy MCP server)"
    r"|(?P<prefix_doc>`agy_` prefix)"
    r"|(?P<codemode_ns>\btools\.agy\b)"
    r"|(?P<install_flag>--host opencode\b(?!\s+--name\b))"
    r"|(?P<tool_call>\bagy_(?!cli\b)(?P<tool_suffix>[a-z<][a-z0-9_>]*))"
)


def _key_looks_secret(raw_key: str) -> bool:
    """Return True when a URL query or fragment parameter name denotes a credential or secret."""
    decoded = unquote(raw_key).strip()
    if not decoded:
        return False
    acronym_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", decoded)
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", acronym_split)
    tokens = [tok for tok in re.split(r"[^a-zA-Z0-9]+", camel_split.lower()) if tok]
    for tok in tokens:
        tok_alpha = tok.rstrip("0123456789")
        if tok in _SECRET_KEY_TOKENS or tok_alpha in _SECRET_KEY_TOKENS:
            return True
        if re.fullmatch(r"sigv\d+", tok) is not None:
            return True
        if tok_alpha and any(tok_alpha.endswith(sfx) for sfx in _SECRET_TOKEN_SUFFIXES):
            return True
    joined = "".join(tokens)
    joined_alpha = "".join(tok.rstrip("0123456789") for tok in tokens)
    return joined in _SECRET_COMPOUND_KEYS or joined_alpha in _SECRET_COMPOUND_KEYS


def _validate_gateway_base_url_no_secrets(base_url: str | None) -> None:
    """Reject gateway URLs that embed userinfo credentials or secret query/fragment parameters."""
    if not base_url or not base_url.strip():
        return
    parsed = urlsplit(base_url.strip())
    if parsed.username or parsed.password or "@" in (parsed.netloc or ""):
        raise ValueError(
            "Gateway base URL must not embed username or password credentials; "
            "use --gateway-audience (OIDC) or export CC_AGY_VERTEX_BASE_URL_TOKEN at runtime."
        )
    normalized_query = parsed.query.replace(";", "&")
    for query_key, _query_val in parse_qsl(normalized_query, keep_blank_values=True):
        if _key_looks_secret(query_key):
            raise ValueError(
                f"Gateway base URL must not embed secret query parameter {query_key!r}; "
                "export CC_AGY_VERTEX_BASE_URL_TOKEN at runtime instead."
            )
    if parsed.fragment:
        normalized_fragment = parsed.fragment.replace(";", "&")
        for frag_key, _frag_val in parse_qsl(normalized_fragment, keep_blank_values=True):
            if _key_looks_secret(frag_key):
                raise ValueError(
                    f"Gateway base URL must not embed secret fragment parameter {frag_key!r}; "
                    "export CC_AGY_VERTEX_BASE_URL_TOKEN at runtime instead."
                )


def resolve_home_dir(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the user's home directory, honoring `HOME` / `USERPROFILE` in `env` when supplied."""
    if env is not None:
        raw_home = (env.get("HOME") or env.get("USERPROFILE") or "").strip()
        if raw_home:
            return Path(raw_home).expanduser()
    return Path.home()


def expand_path_with_env(raw_path: str | Path, env: Mapping[str, str] | None = None) -> Path:
    """Expand a leading `~` against `env` when provided before falling back to `Path.expanduser()`."""
    path_str = str(raw_path)
    if env is not None and path_str.startswith("~"):
        if path_str == "~":
            return resolve_home_dir(env)
        if path_str.startswith(("~/", "~\\")):
            return resolve_home_dir(env) / path_str[2:]
    return Path(raw_path).expanduser()


def _write_unique_backup_bytes(path: Path, data: bytes, mode: int = 0o600) -> Path:
    """Atomically claim and write raw bytes to a unique `.bak-YYYYMMDD-HHMMSS[-N]` file with `O_CREAT | O_EXCL`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    counter = 0
    while True:
        suffix = f"{stamp}" if counter == 0 else f"{stamp}-{counter}"
        candidate = path.with_name(f"{path.name}.bak-{suffix}")
        try:
            fd = os.open(str(candidate), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        except FileExistsError:
            counter += 1
            continue
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(candidate, mode)
        return candidate


def _write_unique_backup(path: Path, text: str, mode: int = 0o600) -> Path:
    """Atomically claim and write UTF-8 text to a unique `.bak-YYYYMMDD-HHMMSS[-N]` backup file."""
    return _write_unique_backup_bytes(path, text.encode("utf-8"), mode)


def _toml_str(value: str) -> str:
    """TOML basic string (JSON string escapes are a valid subset)."""
    return json.dumps(str(value), ensure_ascii=False)


def _normalize_auth_choice(raw: str | None) -> str:
    val = (raw or "auto").strip().lower()
    if val in (
        "gemini-enterprise",
        "gemini_enterprise",
        "enterprise",
        "agy-cli",
        "agy_cli",
        "agy",
        "oauth-business",
        "oauth_business",
    ):
        return "gemini_enterprise"
    if val in ("api-key", "api_key", "ai_studio", "aistudio"):
        return "api_key"
    if val in ("vertex", "vertex-ai", "vertex_ai", "gcp"):
        return "vertex"
    if val in ("gateway", "proxy"):
        return "gateway"
    return "auto"


def resolve_python_bin() -> str:
    """Pick the interpreter that has google-antigravity: current, managed user venv, or .venv."""
    return resolve_sdk_python(PLUGIN_ROOT)


def _normalize_gateway_wire_choice(gateway_wire: str | None) -> str | None:
    """Validate an optional `--gateway-wire` override (`auto`, `vertex`, `gemini`, `litellm-vertex`, `litellm-gemini`, `bifrost`), returning None when `auto`/unset."""
    if gateway_wire is None or not str(gateway_wire).strip():
        return None
    cleaned = str(gateway_wire).strip().lower().replace("-", "_")
    if cleaned == "auto":
        return None
    from server.regions import resolve_gateway_wire

    return resolve_gateway_wire(cleaned, env={})


def build_codex_server_block(
    *,
    name: str = DEFAULT_SERVER_NAME,
    python_bin: str | None = None,
    profile: str = "balanced",
    auth_mode: str | None = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    gateway_wire: str | None = None,
    gateway_audience: str | None = None,
    telemetry_sink: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
) -> str:
    """Render the `[mcp_servers.<name>]` TOML block for `$CODEX_HOME/config.toml`."""
    _validate_gateway_base_url_no_secrets(base_url)
    norm_wire = _normalize_gateway_wire_choice(gateway_wire)
    py = python_bin or resolve_python_bin()
    norm_auth = _normalize_auth_choice(auth_mode)
    env: dict[str, str] = {"CC_AGY_COST_PROFILE": profile}
    if norm_auth == "gemini_enterprise":
        env["CC_AGY_AUTH_MODE"] = "gemini_enterprise"
    elif norm_auth in ("api_key", "vertex", "gateway"):
        env["CC_AGY_AUTH_MODE"] = norm_auth
    if project:
        env["GOOGLE_CLOUD_PROJECT"] = project
    if location and norm_auth != "gemini_enterprise":
        env["GOOGLE_CLOUD_LOCATION"] = location
    if norm_auth != "api_key":
        if base_url:
            env["CC_AGY_VERTEX_BASE_URL"] = base_url
        if norm_wire:
            env["CC_AGY_GATEWAY_WIRE"] = norm_wire
        if gateway_audience and gateway_audience.strip():
            env["CC_AGY_VERTEX_BASE_URL_AUDIENCE"] = gateway_audience.strip()
    if telemetry_sink and telemetry_sink.strip():
        env["CC_AGY_TELEMETRY_SINK"] = telemetry_sink.strip()
    if agy_cli_path:
        env["CC_AGY_CLI_BIN"] = agy_cli_path
    if max_daily_cents is not None and max_daily_cents > 0:
        formatted = f"{max_daily_cents:g}"
        env["CC_AGY_MAX_DAILY_CENTS"] = formatted
    if allowed_projects and allowed_projects.strip():
        env["CC_AGY_ALLOWED_PROJECTS"] = allowed_projects.strip()
    if blocked_projects and blocked_projects.strip():
        env["CC_AGY_BLOCKED_PROJECTS"] = blocked_projects.strip()
    if no_auto_apply:
        env["CC_AGY_AUTO_APPLY_ON_PASS"] = "0"
    forwarded = [v for v in CODEX_FORWARDED_ENV_VARS if v not in env]
    lines = [
        f"[mcp_servers.{name}]",
        f"command = {_toml_str(py)}",
        f"args = [{_toml_str(str(SERVER_ENTRY))}]",
        f"env_vars = [{', '.join(_toml_str(v) for v in forwarded)}]",
        f"startup_timeout_sec = {CODEX_STARTUP_TIMEOUT_SEC}",
        f"tool_timeout_sec = {CODEX_TOOL_TIMEOUT_SEC}",
        "",
        f"[mcp_servers.{name}.env]",
    ]
    lines.extend(f"{key} = {_toml_str(val)}" for key, val in env.items())
    return "\n".join(lines) + "\n"


class CodexConfigError(RuntimeError):
    """Raised when config.toml cannot be updated without risking the user's existing settings."""


def _parse_toml(text: str) -> dict[str, Any] | None:
    """Parse TOML with the standard library (Python 3.11+); None when tomllib is unavailable."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10: use tomli when installed
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return None
    return tomllib.loads(text)


def _has_server_entry(parsed: dict[str, Any] | None, name: str) -> bool:
    servers = (parsed or {}).get("mcp_servers")
    return isinstance(servers, dict) and name in servers


def upsert_codex_server_block(config_text: str, block: str, *, name: str = DEFAULT_SERVER_NAME) -> str:
    """Replace any existing `[mcp_servers.<name>]` table (and its sub-tables) with `block`, or append it."""
    try:
        parsed_before = _parse_toml(config_text)
    except Exception as exc:
        raise CodexConfigError(f"existing config.toml is not valid TOML ({exc}); fix it first") from exc
    table_header_re = re.compile(r"^\s*\[\[?\s*[A-Za-z0-9_\-.\"' ]+\s*\]\]?\s*(?:#.*)?$")
    own_header_re = re.compile(
        r"^\s*\[\s*mcp_servers\s*\.\s*(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9_-]+))\s*(?:\.[^\]]*)?\]\s*(?:#.*)?$"
    )
    out: list[str] = []
    skipping = False
    replaced_at: int | None = None
    for line in config_text.splitlines():
        m = own_header_re.match(line)
        if m and (m.group(1) or m.group(2) or m.group(3)) == name:
            if replaced_at is None:
                replaced_at = len(out)
            skipping = True
            continue
        if skipping and table_header_re.match(line):
            skipping = False
        if not skipping:
            out.append(line)
    if replaced_at is not None and parsed_before is None:
        raise CodexConfigError(
            f"replacing an existing mcp_servers.{name} entry needs Python 3.11+ or the tomli package to "
            f"validate the result; install tomli or remove the entry and re-run"
        )
    if replaced_at is None and _has_server_entry(parsed_before, name):
        raise CodexConfigError(
            f"config.toml already defines mcp_servers.{name} as dotted keys or an inline table; remove it "
            f"or pass --name with a different server name"
        )
    block_lines = block.rstrip("\n").splitlines()
    insert_at = len(out) if replaced_at is None else replaced_at
    before, after = out[:insert_at], out[insert_at:]
    while before and not before[-1].strip():
        before.pop()
    while after and not after[0].strip():
        after.pop(0)
    merged = before + ([""] if before else []) + block_lines + ([""] if after else []) + after
    result = "\n".join(merged).rstrip("\n") + "\n"
    try:
        parsed_after = _parse_toml(result)
    except Exception as exc:
        raise CodexConfigError(f"refusing to write config.toml: the update would produce invalid TOML ({exc})") from exc
    if parsed_after is not None:
        expected = _parse_toml(block) or {}
        if (parsed_after.get("mcp_servers") or {}).get(name) != (expected.get("mcp_servers") or {}).get(name):
            raise CodexConfigError(f"refusing to write config.toml: mcp_servers.{name} did not round-trip")
        for key, val in (parsed_before or {}).items():
            if key != "mcp_servers" and parsed_after.get(key) != val:
                raise CodexConfigError(f"refusing to write config.toml: the update would change [{key}]")
        for srv, val in ((parsed_before or {}).get("mcp_servers") or {}).items():
            if srv != name and (parsed_after.get("mcp_servers") or {}).get(srv) != val:
                raise CodexConfigError(f"refusing to write config.toml: the update would change mcp_servers.{srv}")
    return result


def _write_private(path: Path, text: str, mode: int = 0o600) -> None:
    """Write `text` to `path`, creating it with `mode` and tightening regular files (never symlink targets) to `mode`."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    if not path.is_symlink():
        os.chmod(path, mode)


def install_codex(
    *,
    name: str,
    profile: str,
    auth_mode: str | None = None,
    project: str | None,
    location: str | None,
    base_url: str | None = None,
    gateway_wire: str | None = None,
    gateway_audience: str | None = None,
    telemetry_sink: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    dry_run: bool,
    codex_home: Path | None = None,
) -> dict[str, Any]:
    home = codex_home or Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex")).expanduser()
    cfg = home / "config.toml"
    block = build_codex_server_block(
        name=name,
        profile=profile,
        auth_mode=auth_mode,
        project=project,
        location=location,
        base_url=base_url,
        gateway_wire=gateway_wire,
        gateway_audience=gateway_audience,
        telemetry_sink=telemetry_sink,
        agy_cli_path=agy_cli_path,
        max_daily_cents=max_daily_cents,
        allowed_projects=allowed_projects,
        blocked_projects=blocked_projects,
        no_auto_apply=no_auto_apply,
    )
    existing = cfg.read_text(encoding="utf-8") if cfg.is_file() else ""
    skill_dst = home / "skills" / PLUGIN_NAME / "SKILL.md"
    result: dict[str, Any] = {
        "host": "codex",
        "config_path": str(cfg),
        "skill_path": str(skill_dst),
        "server_name": name,
        "block": block,
        "marketplace_commands": [
            f"codex plugin marketplace add {PLUGIN_ROOT}",
            f"codex plugin add {PLUGIN_NAME}@{MARKETPLACE_NAME}",
        ],
    }
    try:
        updated = upsert_codex_server_block(existing, block, name=name)
    except CodexConfigError as exc:
        result["status"] = "refused"
        result["error"] = str(exc)
        return result
    if dry_run:
        result["status"] = "dry_run"
        return result
    home.mkdir(parents=True, exist_ok=True)
    cfg_mode = (cfg.stat().st_mode & 0o777) if cfg.is_file() else 0o600
    if existing and existing != updated:
        backup = _write_unique_backup(cfg, existing, 0o600)
        result["backup_path"] = str(backup)
    if existing != updated:
        _write_private(cfg, updated, cfg_mode)
    if CODEX_SKILL_SOURCE.is_file():
        skill_dst.parent.mkdir(parents=True, exist_ok=True)
        skill_text = CODEX_SKILL_SOURCE.read_text(encoding="utf-8")
        if not skill_dst.is_file() or skill_dst.read_text(encoding="utf-8") != skill_text:
            skill_dst.write_text(skill_text, encoding="utf-8")
    result["status"] = "installed" if existing != updated else "unchanged"
    codex_bin = shutil.which("codex")
    if codex_bin:
        probe = subprocess.run(
            [codex_bin, "mcp", "get", name, "--json"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "CODEX_HOME": str(home)},
            timeout=30,
        )
        result["codex_verify_exit_code"] = probe.returncode
        if probe.returncode != 0:
            result["codex_verify_error"] = (probe.stderr or probe.stdout).strip()[-600:]
    return result


def build_claude_user_config(
    *,
    profile: str | None = None,
    auth_mode: str | None = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    gateway_wire: str | None = None,
    gateway_audience: str | None = None,
    telemetry_sink: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Build non-secret Claude Code `userConfig` key/value pairs matching `.claude-plugin/plugin.json`."""
    _validate_gateway_base_url_no_secrets(base_url)
    norm_wire = _normalize_gateway_wire_choice(gateway_wire)
    cfg: dict[str, Any] = {}
    norm_auth = _normalize_auth_choice(auth_mode)
    if norm_auth in ("gemini_enterprise", "api_key", "vertex", "gateway"):
        cfg["auth_mode"] = norm_auth
    if profile:
        cfg["cost_profile"] = profile
    if project:
        cfg["google_cloud_project"] = project
    if location and norm_auth != "gemini_enterprise":
        cfg["google_cloud_location"] = location
    if norm_auth != "api_key":
        if base_url:
            cfg["vertex_base_url"] = base_url
        if norm_wire:
            cfg["gateway_wire"] = norm_wire
        if gateway_audience and gateway_audience.strip():
            cfg["vertex_base_url_audience"] = gateway_audience.strip()
    if telemetry_sink and telemetry_sink.strip():
        cfg["telemetry_sink"] = telemetry_sink.strip()
    if agy_cli_path:
        cfg["agy_cli_path"] = agy_cli_path
    if max_daily_cents is not None and max_daily_cents > 0:
        cfg["max_daily_cents"] = f"{max_daily_cents:g}"
    if allowed_projects and allowed_projects.strip():
        cfg["allowed_projects"] = allowed_projects.strip()
    if blocked_projects and blocked_projects.strip():
        cfg["blocked_projects"] = blocked_projects.strip()
    if no_auto_apply:
        cfg["auto_apply_on_pass"] = False
    if strict:
        cfg["strict_delegation_mode"] = True
    return cfg


def claude_install_commands(
    *,
    profile: str | None = None,
    auth_mode: str | None = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    gateway_wire: str | None = None,
    gateway_audience: str | None = None,
    telemetry_sink: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    strict: bool = False,
) -> list[list[str]]:
    cfg = build_claude_user_config(
        profile=profile,
        auth_mode=auth_mode,
        project=project,
        location=location,
        base_url=base_url,
        gateway_wire=gateway_wire,
        gateway_audience=gateway_audience,
        telemetry_sink=telemetry_sink,
        agy_cli_path=agy_cli_path,
        max_daily_cents=max_daily_cents,
        allowed_projects=allowed_projects,
        blocked_projects=blocked_projects,
        no_auto_apply=no_auto_apply,
        strict=strict,
    )
    install_cmd = ["claude", "plugin", "install", f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"]
    for key, val in cfg.items():
        rendered = "true" if val is True else ("false" if val is False else str(val))
        install_cmd.extend(["--config", f"{key}={rendered}"])
    return [
        ["claude", "plugin", "marketplace", "add", str(PLUGIN_ROOT)],
        install_cmd,
    ]


def install_claude(
    *,
    dry_run: bool,
    profile: str | None = None,
    auth_mode: str | None = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    gateway_wire: str | None = None,
    gateway_audience: str | None = None,
    telemetry_sink: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    user_cfg = build_claude_user_config(
        profile=profile,
        auth_mode=auth_mode,
        project=project,
        location=location,
        base_url=base_url,
        gateway_wire=gateway_wire,
        gateway_audience=gateway_audience,
        telemetry_sink=telemetry_sink,
        agy_cli_path=agy_cli_path,
        max_daily_cents=max_daily_cents,
        allowed_projects=allowed_projects,
        blocked_projects=blocked_projects,
        no_auto_apply=no_auto_apply,
        strict=strict,
    )
    cmds = claude_install_commands(
        profile=profile,
        auth_mode=auth_mode,
        project=project,
        location=location,
        base_url=base_url,
        gateway_wire=gateway_wire,
        gateway_audience=gateway_audience,
        telemetry_sink=telemetry_sink,
        agy_cli_path=agy_cli_path,
        max_daily_cents=max_daily_cents,
        allowed_projects=allowed_projects,
        blocked_projects=blocked_projects,
        no_auto_apply=no_auto_apply,
        strict=strict,
    )
    result: dict[str, Any] = {
        "host": "claude",
        "commands": [" ".join(c) for c in cmds],
        "user_config": user_cfg,
        "fallback": f"claude --plugin-dir {PLUGIN_ROOT}",
        "note": (
            "If `claude` is already open in another tab, run `/reload-plugins` (or `claude --continue`) "
            "and verify with `/cc-agy-plugin:doctor`."
        ),
    }
    claude_bin = shutil.which("claude")
    if dry_run or not claude_bin:
        result["status"] = "dry_run" if dry_run else "claude_cli_not_found"
        return result
    outputs: list[dict[str, Any]] = []
    for idx, cmd in enumerate(cmds):
        proc = subprocess.run([claude_bin, *cmd[1:]], capture_output=True, text=True, check=False, timeout=120)
        combined = (proc.stdout + proc.stderr).strip()
        # Graceful fallback if an older Claude CLI does not accept `--config` on `plugin install`.
        lower_out = combined.lower()
        unknown_config_flag = "--config" in lower_out and any(
            phrase in lower_out
            for phrase in (
                "unknown option",
                "unrecognized option",
                "unexpected argument",
                "invalid option",
                "no such option",
            )
        )
        if idx == 1 and proc.returncode != 0 and "--config" in cmd and unknown_config_flag:
            plain_install = ["plugin", "install", f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"]
            proc = subprocess.run(
                [claude_bin, *plain_install], capture_output=True, text=True, check=False, timeout=120
            )
            outputs.append(
                {
                    "command": f"claude {' '.join(plain_install)}",
                    "exit_code": proc.returncode,
                    "output": (proc.stdout + proc.stderr).strip()[-400:],
                }
            )
            if proc.returncode == 0 and user_cfg:
                cfg_cmd = ["plugin", "configure", f"{PLUGIN_NAME}@{MARKETPLACE_NAME}", "--values-stdin"]
                cfg_proc = subprocess.run(
                    [claude_bin, *cfg_cmd],
                    input=json.dumps(user_cfg),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
                outputs.append(
                    {
                        "command": f"claude {' '.join(cfg_cmd)}",
                        "exit_code": cfg_proc.returncode,
                        "output": (cfg_proc.stdout + cfg_proc.stderr).strip()[-400:],
                    }
                )
            break
        outputs.append({"command": " ".join(cmd), "exit_code": proc.returncode, "output": combined[-400:]})
        if proc.returncode != 0:
            break
    result["runs"] = outputs
    result["status"] = "installed" if outputs and all(o["exit_code"] == 0 for o in outputs) else "failed"
    return result


class AgyConfigError(RuntimeError):
    """Raised when ~/.gemini/config/mcp_config.json cannot be updated without risking existing settings."""


def build_agy_server_entry(
    *,
    python_bin: str | None = None,
    profile: str = "balanced",
    auth_mode: str | None = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    gateway_wire: str | None = None,
    gateway_audience: str | None = None,
    telemetry_sink: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Build the stdio MCP server entry for Antigravity's `~/.gemini/config/mcp_config.json`."""
    _validate_gateway_base_url_no_secrets(base_url)
    norm_wire = _normalize_gateway_wire_choice(gateway_wire)
    py = python_bin or resolve_python_bin()
    norm_auth = _normalize_auth_choice(auth_mode)
    env: dict[str, str] = {"CC_AGY_COST_PROFILE": profile}
    if norm_auth == "gemini_enterprise":
        env["CC_AGY_AUTH_MODE"] = "gemini_enterprise"
    elif norm_auth in ("api_key", "vertex", "gateway"):
        env["CC_AGY_AUTH_MODE"] = norm_auth
    if project:
        env["GOOGLE_CLOUD_PROJECT"] = project
    if location and norm_auth != "gemini_enterprise":
        env["GOOGLE_CLOUD_LOCATION"] = location
    if norm_auth != "api_key":
        if base_url:
            env["CC_AGY_VERTEX_BASE_URL"] = base_url
        if norm_wire:
            env["CC_AGY_GATEWAY_WIRE"] = norm_wire
        if gateway_audience and gateway_audience.strip():
            env["CC_AGY_VERTEX_BASE_URL_AUDIENCE"] = gateway_audience.strip()
    if telemetry_sink and telemetry_sink.strip():
        env["CC_AGY_TELEMETRY_SINK"] = telemetry_sink.strip()
    if agy_cli_path:
        env["CC_AGY_CLI_BIN"] = agy_cli_path
    if max_daily_cents is not None and max_daily_cents > 0:
        env["CC_AGY_MAX_DAILY_CENTS"] = f"{max_daily_cents:g}"
    if allowed_projects and allowed_projects.strip():
        env["CC_AGY_ALLOWED_PROJECTS"] = allowed_projects.strip()
    if blocked_projects and blocked_projects.strip():
        env["CC_AGY_BLOCKED_PROJECTS"] = blocked_projects.strip()
    if no_auto_apply:
        env["CC_AGY_AUTO_APPLY_ON_PASS"] = "0"
    if strict:
        env["CC_AGY_STRICT_MODE"] = "true"
    return {
        "command": py,
        "args": [str(SERVER_ENTRY)],
        "env": env,
    }


def upsert_agy_mcp_config(
    config_text: str,
    entry: dict[str, Any],
    *,
    name: str = DEFAULT_SERVER_NAME,
) -> str:
    """Upsert `mcpServers[name]` inside an Antigravity `mcp_config.json` document without touching other keys."""
    stripped = config_text.strip()
    if not stripped:
        parsed: dict[str, Any] = {}
    else:
        try:
            loaded = json.loads(stripped)
        except Exception as exc:
            raise AgyConfigError(f"existing mcp_config.json is not valid JSON ({exc}); fix it first") from exc
        if not isinstance(loaded, dict):
            raise AgyConfigError("existing mcp_config.json top-level value must be a JSON object")
        parsed = dict(loaded)
    servers = parsed.get("mcpServers")
    if servers is None:
        servers_map: dict[str, Any] = {}
    elif isinstance(servers, dict):
        servers_map = dict(servers)
    else:
        raise AgyConfigError("existing mcp_config.json 'mcpServers' field must be a JSON object")
    servers_map[name] = entry
    parsed["mcpServers"] = servers_map
    return json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"


def install_agy(
    *,
    name: str = DEFAULT_SERVER_NAME,
    profile: str = "balanced",
    auth_mode: str | None = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    gateway_wire: str | None = None,
    gateway_audience: str | None = None,
    telemetry_sink: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    strict: bool = False,
    dry_run: bool = False,
    agy_config_dir: Path | None = None,
    python_bin: str | None = None,
) -> dict[str, Any]:
    """Register cc-agy-plugin in Antigravity's documented `~/.gemini/config/` MCP, plugin, and skill directories."""
    cfg_dir = agy_config_dir or Path(
        os.environ.get("AGY_CONFIG_DIR") or (Path.home() / ".gemini" / "config")
    ).expanduser()
    mcp_cfg_path = cfg_dir / "mcp_config.json"
    skill_dst = cfg_dir / "skills" / PLUGIN_NAME / "SKILL.md"
    plugin_dir = cfg_dir / "plugins" / PLUGIN_NAME
    plugin_json_path = plugin_dir / "plugin.json"
    plugin_mcp_path = plugin_dir / "mcp_config.json"
    plugin_skill_path = plugin_dir / "skills" / "vertex" / "SKILL.md"

    entry = build_agy_server_entry(
        python_bin=python_bin,
        profile=profile,
        auth_mode=auth_mode,
        project=project,
        location=location,
        base_url=base_url,
        gateway_wire=gateway_wire,
        gateway_audience=gateway_audience,
        telemetry_sink=telemetry_sink,
        agy_cli_path=agy_cli_path,
        max_daily_cents=max_daily_cents,
        allowed_projects=allowed_projects,
        blocked_projects=blocked_projects,
        no_auto_apply=no_auto_apply,
        strict=strict,
    )
    env_flags = [f"--env {k}={v}" for k, v in entry["env"].items()]
    cli_cmd = f"agy mcp add {' '.join(env_flags)} {name} {entry['command']} {SERVER_ENTRY}"

    existing = mcp_cfg_path.read_text(encoding="utf-8") if mcp_cfg_path.is_file() else ""
    result: dict[str, Any] = {
        "host": "agy",
        "config_path": str(mcp_cfg_path),
        "skill_path": str(skill_dst),
        "plugin_dir": str(plugin_dir),
        "server_name": name,
        "entry": entry,
        "cli_command": cli_cmd,
    }
    try:
        updated = upsert_agy_mcp_config(existing, entry, name=name)
    except AgyConfigError as exc:
        result["status"] = "refused"
        result["error"] = str(exc)
        return result
    if dry_run:
        result["status"] = "dry_run"
        return result

    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_mode = (mcp_cfg_path.stat().st_mode & 0o777) if mcp_cfg_path.is_file() else 0o600
    if existing and existing != updated:
        backup = _write_unique_backup(mcp_cfg_path, existing, 0o600)
        result["backup_path"] = str(backup)
    if existing != updated:
        _write_private(mcp_cfg_path, updated, cfg_mode)

    skill_text = CODEX_SKILL_SOURCE.read_text(encoding="utf-8") if CODEX_SKILL_SOURCE.is_file() else ""
    if skill_text:
        skill_dst.parent.mkdir(parents=True, exist_ok=True)
        if not skill_dst.is_file() or skill_dst.read_text(encoding="utf-8") != skill_text:
            skill_dst.write_text(skill_text, encoding="utf-8")

    plugin_dir.mkdir(parents=True, exist_ok=True)
    plugin_manifest = {
        "name": PLUGIN_NAME,
        "displayName": PLUGIN_NAME,
        "version": "0.3.0",
        "description": (
            "Delegate coding, multi-file edits, and test loops to fast, low-cost Gemini workers "
            "in isolated git worktrees with signed cost receipts."
        ),
    }
    plugin_json_text = json.dumps(plugin_manifest, indent=2, ensure_ascii=False) + "\n"
    if not plugin_json_path.is_file() or plugin_json_path.read_text(encoding="utf-8") != plugin_json_text:
        plugin_json_path.write_text(plugin_json_text, encoding="utf-8")
    plugin_mcp_text = json.dumps({"mcpServers": {name: entry}}, indent=2, ensure_ascii=False) + "\n"
    if not plugin_mcp_path.is_file() or plugin_mcp_path.read_text(encoding="utf-8") != plugin_mcp_text:
        _write_private(plugin_mcp_path, plugin_mcp_text, 0o600)
    if skill_text:
        plugin_skill_path.parent.mkdir(parents=True, exist_ok=True)
        if not plugin_skill_path.is_file() or plugin_skill_path.read_text(encoding="utf-8") != skill_text:
            plugin_skill_path.write_text(skill_text, encoding="utf-8")

    result["status"] = "installed" if existing != updated else "unchanged"
    return result


class CursorConfigError(RuntimeError):
    """Raised when ~/.cursor/mcp.json cannot be updated without risking existing settings."""


def build_cursor_server_entry(
    *,
    python_bin: str | None = None,
    profile: str = "balanced",
    auth_mode: str | None = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    gateway_wire: str | None = None,
    gateway_audience: str | None = None,
    telemetry_sink: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Build the stdio MCP server entry for Cursor's `~/.cursor/mcp.json`."""
    return build_agy_server_entry(
        python_bin=python_bin,
        profile=profile,
        auth_mode=auth_mode,
        project=project,
        location=location,
        base_url=base_url,
        gateway_wire=gateway_wire,
        gateway_audience=gateway_audience,
        telemetry_sink=telemetry_sink,
        agy_cli_path=agy_cli_path,
        max_daily_cents=max_daily_cents,
        allowed_projects=allowed_projects,
        blocked_projects=blocked_projects,
        no_auto_apply=no_auto_apply,
        strict=strict,
    )


def upsert_cursor_mcp_config(
    config_text: str,
    entry: dict[str, Any],
    *,
    name: str = DEFAULT_SERVER_NAME,
) -> str:
    """Upsert `mcpServers[name]` inside a Cursor `mcp.json` document without touching other keys."""
    stripped = config_text.strip()
    if not stripped:
        parsed: dict[str, Any] = {}
    else:
        try:
            loaded = json.loads(stripped)
        except Exception as exc:
            raise CursorConfigError(f"existing mcp.json is not valid JSON ({exc}); fix it first") from exc
        if not isinstance(loaded, dict):
            raise CursorConfigError("existing mcp.json top-level value must be a JSON object")
        parsed = dict(loaded)
    servers = parsed.get("mcpServers")
    if servers is None:
        servers_map: dict[str, Any] = {}
    elif isinstance(servers, dict):
        servers_map = dict(servers)
    else:
        raise CursorConfigError("existing mcp.json 'mcpServers' field must be a JSON object")
    servers_map[name] = entry
    parsed["mcpServers"] = servers_map
    return json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"


def install_cursor(
    *,
    name: str = DEFAULT_SERVER_NAME,
    profile: str = "balanced",
    auth_mode: str | None = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    gateway_wire: str | None = None,
    gateway_audience: str | None = None,
    telemetry_sink: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    strict: bool = False,
    dry_run: bool = False,
    cursor_config_dir: Path | None = None,
    python_bin: str | None = None,
) -> dict[str, Any]:
    """Register cc-agy-plugin in Cursor's `~/.cursor/mcp.json` and `~/.cursor/skills/cc-agy-plugin/SKILL.md`."""
    cfg_dir = cursor_config_dir or Path(
        os.environ.get("CURSOR_CONFIG_DIR") or (Path.home() / ".cursor")
    ).expanduser()
    mcp_cfg_path = cfg_dir / "mcp.json"
    skill_dst = cfg_dir / "skills" / PLUGIN_NAME / "SKILL.md"

    entry = build_cursor_server_entry(
        python_bin=python_bin,
        profile=profile,
        auth_mode=auth_mode,
        project=project,
        location=location,
        base_url=base_url,
        gateway_wire=gateway_wire,
        gateway_audience=gateway_audience,
        telemetry_sink=telemetry_sink,
        agy_cli_path=agy_cli_path,
        max_daily_cents=max_daily_cents,
        allowed_projects=allowed_projects,
        blocked_projects=blocked_projects,
        no_auto_apply=no_auto_apply,
        strict=strict,
    )

    existing = mcp_cfg_path.read_text(encoding="utf-8") if mcp_cfg_path.is_file() else ""
    result: dict[str, Any] = {
        "host": "cursor",
        "config_path": str(mcp_cfg_path),
        "skill_path": str(skill_dst),
        "server_name": name,
        "entry": entry,
    }
    try:
        updated = upsert_cursor_mcp_config(existing, entry, name=name)
    except CursorConfigError as exc:
        result["status"] = "refused"
        result["error"] = str(exc)
        return result
    if dry_run:
        result["status"] = "dry_run"
        return result

    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_mode = (mcp_cfg_path.stat().st_mode & 0o777) if mcp_cfg_path.is_file() else 0o600
    if existing and existing != updated:
        backup = _write_unique_backup(mcp_cfg_path, existing, 0o600)
        result["backup_path"] = str(backup)
    if existing != updated:
        _write_private(mcp_cfg_path, updated, cfg_mode)

    skill_changed = False
    skill_text = CODEX_SKILL_SOURCE.read_text(encoding="utf-8") if CODEX_SKILL_SOURCE.is_file() else ""
    if skill_text:
        skill_dst.parent.mkdir(parents=True, exist_ok=True)
        if not skill_dst.is_file() or skill_dst.read_text(encoding="utf-8") != skill_text:
            skill_dst.write_text(skill_text, encoding="utf-8")
            skill_changed = True

    result["status"] = "installed" if (existing != updated or skill_changed) else "unchanged"
    return result


class OpenCodeConfigError(RuntimeError):
    """Raised when an OpenCode opencode.json / opencode.jsonc file cannot be safely updated."""


def _validate_opencode_server_name(name: str) -> None:
    """Validate `name` for OpenCode MCP config keys, tool prefixes, and YAML frontmatter."""
    if not isinstance(name, str) or not _SERVER_NAME_RE.fullmatch(name):
        raise OpenCodeConfigError(
            f"invalid OpenCode server name {name!r}; only ASCII letters, digits, '-' and '_' are allowed"
        )
    if name in _OPENCODE_RESERVED_SERVER_NAMES:
        raise OpenCodeConfigError(
            f"server name {name!r} is reserved by OpenCode's mcp config schema; pass a different --name"
        )


def resolve_opencode_config_dir(
    env: Mapping[str, str] | None = None,
    explicit_dir: Path | None = None,
) -> Path:
    """Resolve OpenCode's global config directory (`OPENCODE_CONFIG_DIR` -> `XDG_CONFIG_HOME/opencode` -> `~/.config/opencode`)."""
    if explicit_dir is not None:
        return expand_path_with_env(explicit_dir, env)
    source = env if env is not None else os.environ
    override = (source.get("OPENCODE_CONFIG_DIR") or "").strip()
    if override:
        return expand_path_with_env(override, env)
    xdg = (source.get("XDG_CONFIG_HOME") or "").strip()
    if xdg:
        return expand_path_with_env(xdg, env) / "opencode"
    return resolve_home_dir(env) / ".config" / "opencode"


def resolve_opencode_config_path(
    cfg_dir: Path,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the active OpenCode config file (`OPENCODE_CONFIG` when passed, `opencode.jsonc` when alone, else `opencode.json`)."""
    if env is not None:
        explicit_cfg = (env.get("OPENCODE_CONFIG") or "").strip()
        if explicit_cfg:
            return expand_path_with_env(explicit_cfg, env)
    json_path = cfg_dir / "opencode.json"
    jsonc_path = cfg_dir / "opencode.jsonc"
    if jsonc_path.is_file() and not json_path.is_file():
        return jsonc_path
    return json_path


def _strip_jsonc(text: str) -> str:
    """Strip UTF-8 BOM, `//` and `/* */` comments, and trailing commas outside JSON string literals."""
    src = text.lstrip("\ufeff")
    n = len(src)
    out: list[str] = []
    idx = 0

    def _next_meaningful_char(pos: int) -> str:
        cur = pos
        while cur < n:
            ch = src[cur]
            if ch in " \t\r\n":
                cur += 1
            elif src[cur : cur + 2] == "//":
                nl = src.find("\n", cur + 2)
                cur = n if nl == -1 else nl + 1
            elif src[cur : cur + 2] == "/*":
                end = src.find("*/", cur + 2)
                if end == -1:
                    return ""
                cur = end + 2
            else:
                return ch
        return ""

    while idx < n:
        ch = src[idx]
        if ch == '"':
            out.append(ch)
            idx += 1
            while idx < n:
                s_ch = src[idx]
                out.append(s_ch)
                if s_ch == "\\" and idx + 1 < n:
                    idx += 1
                    out.append(src[idx])
                elif s_ch == '"':
                    idx += 1
                    break
                idx += 1
        elif src[idx : idx + 2] == "//":
            nl = src.find("\n", idx + 2)
            if nl == -1:
                break
            out.append("\n")
            idx = nl + 1
        elif src[idx : idx + 2] == "/*":
            end = src.find("*/", idx + 2)
            if end == -1:
                raise json.JSONDecodeError("Unterminated /* comment in JSONC", src, idx)
            out.append("\n" * src[idx + 2 : end].count("\n"))
            idx = end + 2
        elif ch == ",":
            if _next_meaningful_char(idx + 1) not in ("}", "]"):
                out.append(ch)
            idx += 1
        else:
            out.append(ch)
            idx += 1
    return "".join(out)


def _loads_jsonc(text: str) -> Any:
    """Parse a JSONC string (allowing UTF-8 BOM, comments, and trailing commas)."""
    return json.loads(_strip_jsonc(text))


def build_opencode_server_entry(
    *,
    python_bin: str | None = None,
    profile: str = "balanced",
    auth_mode: str | None = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    gateway_wire: str | None = None,
    gateway_audience: str | None = None,
    telemetry_sink: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    strict: bool = False,
    v2: bool = False,
    pool: bool | None = None,
    codemode: bool | None = None,
    protocol: str | None = None,
) -> dict[str, Any]:
    """Build the local MCP server entry for OpenCode's `opencode.json` (`mcp.<name>` or V2 `mcp.servers.<name>`).

    Fresh configs default to OpenCode's legacy-compatible V1 `mcp.<name>` shape (`v2=False`, schema-valid
    across both OpenCode v1.x and v2, using a single `1800000` ms timeout so long `delegate_result` runs do
    not time out). Passing `v2=True`, any explicit V2 option (`pool`, `codemode`, `protocol`), or calling
    `build_opencode_v2_server_entry` builds an explicit OpenCode V2 (`2.2.6`) `Mcp.LocalConfig` entry with
    split `{startup, execution}` timeouts and `pool: False` by default (because the `agy` MCP server binds
    its workspace root at startup).
    """
    if protocol is not None:
        if not isinstance(protocol, str) or protocol not in _OPENCODE_V2_PROTOCOLS:
            raise ValueError(
                f"Unsupported OpenCode MCP protocol {protocol!r} for cc-agy-plugin; expected one of "
                f"{sorted(_OPENCODE_V2_PROTOCOLS)} ('2026-07-28' requires MCP revision 2026-07-28 and fails "
                "handshake against this stdio server's 2024-11-05 / 2025-03-26 protocol)"
            )
    eff_v2 = bool(v2 or pool is not None or codemode is not None or protocol is not None)
    agy_entry = build_agy_server_entry(
        python_bin=python_bin,
        profile=profile,
        auth_mode=auth_mode,
        project=project,
        location=location,
        base_url=base_url,
        gateway_wire=gateway_wire,
        gateway_audience=gateway_audience,
        telemetry_sink=telemetry_sink,
        agy_cli_path=agy_cli_path,
        max_daily_cents=max_daily_cents,
        allowed_projects=allowed_projects,
        blocked_projects=blocked_projects,
        no_auto_apply=no_auto_apply,
        strict=strict,
    )
    if eff_v2:
        v2_entry: dict[str, Any] = {
            "type": "local",
            "command": [agy_entry["command"], *agy_entry["args"]],
            "environment": dict(agy_entry["env"]),
            "disabled": False,
            "timeout": {
                "startup": OPENCODE_STARTUP_TIMEOUT_MS,
                "execution": OPENCODE_TOOL_TIMEOUT_MS,
            },
            "pool": bool(pool) if pool is not None else False,
        }
        if isinstance(codemode, bool):
            v2_entry["codemode"] = codemode
        if protocol is not None:
            v2_entry["protocol"] = protocol
        return v2_entry
    return {
        "type": "local",
        "command": [agy_entry["command"], *agy_entry["args"]],
        "environment": dict(agy_entry["env"]),
        "enabled": True,
        "timeout": OPENCODE_TOOL_TIMEOUT_MS,
    }


def build_opencode_v2_server_entry(
    *,
    pool: bool = False,
    codemode: bool | None = None,
    protocol: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build an OpenCode V2 (`2.2.6`) `mcp.servers.<name>` entry (`disabled`, split `timeout`, `pool=False`)."""
    return build_opencode_server_entry(
        v2=True,
        pool=pool,
        codemode=codemode,
        protocol=protocol,
        **kwargs,
    )


def _is_v1_toggle_only_stub(val: Any) -> bool:
    """Return True when `val` is a V1 disable/enable stub `{"enabled": bool}` allowed under `mcp.additionalProperties`."""
    return isinstance(val, Mapping) and set(val.keys()) == {"enabled"} and isinstance(val.get("enabled"), bool)


def _migrate_v1_remote_oauth_to_v2(oauth_val: Any) -> dict[str, Any] | Literal[False] | None:
    """Convert a V1 `McpOAuthConfig` mapping (`clientId`, etc.) into a schema-valid V2 `Mcp.OAuthConfigEncoded`."""
    if oauth_val is False:
        return False
    if not isinstance(oauth_val, Mapping):
        return None
    converted: dict[str, Any] = {}
    for raw_k, raw_v in oauth_val.items():
        if raw_k in _OPENCODE_V2_OAUTH_ALLOWED_KEYS:
            target_k = raw_k
        elif raw_k in _OPENCODE_V1_TO_V2_OAUTH_KEY_MAP:
            target_k = _OPENCODE_V1_TO_V2_OAUTH_KEY_MAP[raw_k]
            if target_k in oauth_val:
                continue
        else:
            continue
        if target_k == "callback_port":
            if isinstance(raw_v, int) and not isinstance(raw_v, bool) and 1 <= raw_v <= 65535:
                converted[target_k] = raw_v
        elif isinstance(raw_v, str):
            converted[target_k] = raw_v
    return converted


def _has_v1_mcp_entry_fields(obj: Mapping[str, Any]) -> bool:
    """Return True when `obj` has scalar or list V1 `McpLocalConfig` / `McpRemoteConfig` fields (not nested server mappings)."""
    return any(k in obj and not isinstance(obj[k], Mapping) for k in _OPENCODE_MCP_ENTRY_KEYS)


def _clean_opencode_v2_top_timeout(raw_mcp_t: Any) -> dict[str, int] | None:
    """Sanitize a top-level OpenCode V2 `mcp.timeout` mapping or integer into a schema-valid `Mcp.Timeout` object."""
    if isinstance(raw_mcp_t, Mapping) and not _has_v1_mcp_entry_fields(raw_mcp_t):
        cleaned = {
            k: int(raw_mcp_t[k])
            for k in _OPENCODE_V2_TIMEOUT_KEYS
            if k in raw_mcp_t and _is_valid_opencode_timeout_int(raw_mcp_t[k])
        }
        return cleaned or None
    if _is_valid_opencode_timeout_int(raw_mcp_t):
        return {"execution": int(raw_mcp_t)}
    return None


def _is_opencode_v2_servers_envelope(mcp_obj: Mapping[str, Any]) -> bool:
    """Return True when `mcp` uses OpenCode's V2 `{servers?: {...}, timeout?: {...}}` envelope (allowing toggle-only `{"enabled": bool}` siblings)."""
    if not mcp_obj:
        return False
    if not ("servers" in mcp_obj or "timeout" in mcp_obj):
        return False
    for k, v in mcp_obj.items():
        if isinstance(v, Mapping) and k not in ("servers", "timeout") and not _is_v1_toggle_only_stub(v):
            return False
    if "servers" in mcp_obj:
        servers = mcp_obj["servers"]
        if not isinstance(servers, dict):
            return False
        if _has_v1_mcp_entry_fields(servers):
            return False
    if "timeout" in mcp_obj:
        timeout_val = mcp_obj["timeout"]
        if not isinstance(timeout_val, dict):
            return False
        if _has_v1_mcp_entry_fields(timeout_val):
            return False
    return True


def _merge_opencode_env(existing_env: Any, plugin_env: Mapping[str, str]) -> dict[str, str]:
    """Preserve user-added environment keys while refreshing all plugin-managed keys."""
    preserved: dict[str, str] = {}
    if isinstance(existing_env, Mapping):
        for key, val in existing_env.items():
            if key not in _PLUGIN_MANAGED_ENV_KEYS and isinstance(val, str):
                preserved[str(key)] = val
    return {**preserved, **dict(plugin_env)}


def _is_valid_opencode_timeout_int(val: Any) -> bool:
    """Return True when `val` is a positive JSON-safe integer within OpenCode's `Number.MAX_SAFE_INTEGER` schema cap."""
    return isinstance(val, int) and not isinstance(val, bool) and 0 < val <= _OPENCODE_MAX_SAFE_INT


def _merge_opencode_v1_entry(existing_srv: Any, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Merge `entry` into an existing V1 `mcp.<name>` object, keeping only schema-valid `McpLocalConfig` keys."""
    src = existing_srv if isinstance(existing_srv, Mapping) else {}
    is_local = src.get("type") in (None, "local")
    if isinstance(src.get("enabled"), bool):
        enabled_val = src["enabled"]
    elif isinstance(src.get("disabled"), bool):
        enabled_val = not src["disabled"]
    elif isinstance(entry.get("enabled"), bool):
        enabled_val = entry["enabled"]
    elif isinstance(entry.get("disabled"), bool):
        enabled_val = not entry["disabled"]
    else:
        enabled_val = True
    existing_timeout = src.get("timeout") if is_local else None
    raw_entry_timeout = entry.get("timeout", OPENCODE_TOOL_TIMEOUT_MS)
    if isinstance(raw_entry_timeout, Mapping):
        raw_entry_timeout = raw_entry_timeout.get("execution", OPENCODE_TOOL_TIMEOUT_MS)
    entry_timeout_int = (
        int(raw_entry_timeout)
        if _is_valid_opencode_timeout_int(raw_entry_timeout)
        else OPENCODE_TOOL_TIMEOUT_MS
    )
    timeout_val = (
        int(existing_timeout)
        if _is_valid_opencode_timeout_int(existing_timeout)
        else entry_timeout_int
    )
    merged: dict[str, Any] = {
        "type": "local",
        "command": list(entry["command"]),
        "environment": (
            _merge_opencode_env(src.get("environment"), entry.get("environment") or {})
            if is_local
            else dict(entry.get("environment") or {})
        ),
        "enabled": enabled_val,
        "timeout": timeout_val,
    }
    if is_local and isinstance(src.get("cwd"), str):
        merged["cwd"] = src["cwd"]
    return merged


def _merge_opencode_v2_entry(
    existing_srv: Any,
    entry: Mapping[str, Any],
    *,
    pool_override: bool | None = None,
) -> dict[str, Any]:
    """Merge `entry` into an existing V2 `mcp.servers.<name>` object, keeping only valid `Mcp.LocalConfigEncoded` keys.

    Precedence: explicit opt-ins/overrides (`pool_override` bool, `entry["pool"] is True`, `codemode` bool,
    or valid `protocol` string) take precedence over `existing_srv`; omitted options preserve valid values
    from `existing_srv`, while invalid existing types are dropped. Partial `timeout` mappings in `existing_srv`
    are merged over `default_timeout` so `"execution": 1800000` is never lost.
    """
    src = existing_srv if isinstance(existing_srv, Mapping) else {}
    is_local = src.get("type") in (None, "local")
    if isinstance(src.get("disabled"), bool):
        disabled_val = src["disabled"]
    elif isinstance(src.get("enabled"), bool):
        disabled_val = not src["enabled"]
    elif isinstance(entry.get("disabled"), bool):
        disabled_val = entry["disabled"]
    else:
        disabled_val = not bool(entry.get("enabled", True))
    default_timeout: dict[str, int] = {
        "startup": OPENCODE_STARTUP_TIMEOUT_MS,
        "execution": int(
            entry["timeout"].get("execution", OPENCODE_TOOL_TIMEOUT_MS)
            if isinstance(entry.get("timeout"), Mapping)
            else entry.get("timeout", OPENCODE_TOOL_TIMEOUT_MS)
        ),
    }
    existing_timeout = src.get("timeout") if is_local else None
    if isinstance(existing_timeout, Mapping):
        cleaned_timeout = {
            key: int(existing_timeout[key])
            for key in _OPENCODE_V2_TIMEOUT_KEYS
            if key in existing_timeout and _is_valid_opencode_timeout_int(existing_timeout[key])
        }
        timeout_val = {**default_timeout, **cleaned_timeout} if cleaned_timeout else default_timeout
    elif _is_valid_opencode_timeout_int(existing_timeout):
        timeout_val = {
            "startup": OPENCODE_STARTUP_TIMEOUT_MS,
            "execution": int(existing_timeout),
        }
    else:
        timeout_val = default_timeout
    merged: dict[str, Any] = {
        "type": "local",
        "command": list(entry["command"]),
        "environment": (
            _merge_opencode_env(src.get("environment"), entry.get("environment") or {})
            if is_local
            else dict(entry.get("environment") or {})
        ),
        "disabled": disabled_val,
        "timeout": timeout_val,
    }
    if is_local:
        if isinstance(src.get("cwd"), str):
            merged["cwd"] = src["cwd"]
        if isinstance(pool_override, bool):
            merged["pool"] = pool_override
        elif entry.get("pool") is True:
            merged["pool"] = True
        elif isinstance(src.get("pool"), bool):
            merged["pool"] = src["pool"]
        elif isinstance(entry.get("pool"), bool):
            merged["pool"] = entry["pool"]
        if isinstance(entry.get("codemode"), bool):
            merged["codemode"] = entry["codemode"]
        elif isinstance(src.get("codemode"), bool):
            merged["codemode"] = src["codemode"]
        entry_proto = entry.get("protocol")
        src_proto = src.get("protocol")
        if isinstance(entry_proto, str) and entry_proto in _OPENCODE_V2_PROTOCOLS:
            merged["protocol"] = entry_proto
        elif isinstance(src_proto, str) and src_proto in _OPENCODE_V2_PROTOCOLS:
            merged["protocol"] = src_proto
    return merged


def upsert_opencode_mcp_config(
    config_text: str,
    entry: dict[str, Any],
    *,
    name: str = DEFAULT_SERVER_NAME,
    filename: str = "opencode.json",
    v2: bool = False,
    pool_override: bool | None = None,
) -> str:
    """Upsert `mcp[name]` (or V2 `mcp.servers[name]`) inside an OpenCode config without touching other settings."""
    _validate_opencode_server_name(name)
    stripped = config_text.lstrip("\ufeff").strip()
    if not stripped:
        parsed: dict[str, Any] = {"$schema": OPENCODE_SCHEMA_URL}
    else:
        try:
            loaded = json.loads(stripped)
        except ValueError as exc:
            raise OpenCodeConfigError(
                f"existing {filename} is not strict JSON ({exc}); if it uses JSONC comments or trailing commas, "
                f"add mcp.{name} manually so comments are preserved"
            ) from exc
        if not isinstance(loaded, dict):
            raise OpenCodeConfigError(f"existing {filename} top-level value must be a JSON object")
        parsed = dict(loaded)

    mcp_val = parsed.get("mcp")
    use_v2_envelope = bool(v2) or isinstance(entry.get("timeout"), Mapping) or "disabled" in entry
    if mcp_val is None:
        if use_v2_envelope:
            parsed["mcp"] = {"servers": {name: _merge_opencode_v2_entry(None, entry, pool_override=pool_override)}}
        else:
            parsed["mcp"] = {name: _merge_opencode_v1_entry(None, entry)}
    elif not isinstance(mcp_val, dict):
        raise OpenCodeConfigError(f"existing {filename} 'mcp' field must be a JSON object")
    elif _is_opencode_v2_servers_envelope(mcp_val):
        mcp_map = dict(mcp_val)
        if "timeout" in mcp_map:
            cleaned_env_t = _clean_opencode_v2_top_timeout(mcp_map.get("timeout"))
            if cleaned_env_t:
                mcp_map["timeout"] = cleaned_env_t
            else:
                mcp_map.pop("timeout", None)
        servers_map = dict(mcp_val.get("servers") or {})
        servers_map[name] = _merge_opencode_v2_entry(servers_map.get(name), entry, pool_override=pool_override)
        mcp_map["servers"] = servers_map
        parsed["mcp"] = mcp_map
    elif use_v2_envelope:
        for reserved_k in ("servers", "timeout"):
            reserved_val = mcp_val.get(reserved_k)
            if isinstance(reserved_val, Mapping) and _has_v1_mcp_entry_fields(reserved_val):
                raise OpenCodeConfigError(
                    f"existing {filename} defines a V1 MCP server under reserved V2 key 'mcp.{reserved_k}'; "
                    "rename that server entry before migrating to OpenCode V2"
                )
        nested_servers = mcp_val.get("servers")
        servers_map = dict(nested_servers) if isinstance(nested_servers, dict) else {}
        preserved_v1_stubs: dict[str, dict[str, Any]] = {}
        for srv_k, srv_v in mcp_val.items():
            if srv_k in ("servers", "timeout") or srv_k == name:
                continue
            if _is_v1_toggle_only_stub(srv_v):
                preserved_v1_stubs[srv_k] = {"enabled": srv_v["enabled"]}
            elif isinstance(srv_v, Mapping):
                if srv_v.get("type") not in ("local", "remote"):
                    preserved_v1_stubs[srv_k] = dict(srv_v)
                    continue
                migrated = dict(srv_v)
                if isinstance(migrated.get("enabled"), bool):
                    en_bool = migrated.pop("enabled")
                    if "disabled" not in migrated:
                        migrated["disabled"] = not en_bool
                if _is_valid_opencode_timeout_int(migrated.get("timeout")):
                    migrated["timeout"] = {
                        "startup": OPENCODE_STARTUP_TIMEOUT_MS,
                        "execution": int(migrated["timeout"]),
                    }
                if migrated.get("type") == "remote" and "oauth" in migrated:
                    mig_oauth = _migrate_v1_remote_oauth_to_v2(migrated.get("oauth"))
                    if mig_oauth is None:
                        migrated.pop("oauth", None)
                    else:
                        migrated["oauth"] = mig_oauth
                servers_map[srv_k] = migrated
        existing_self = mcp_val.get(name) if isinstance(mcp_val.get(name), Mapping) else servers_map.get(name)
        servers_map[name] = _merge_opencode_v2_entry(existing_self, entry, pool_override=pool_override)
        new_mcp: dict[str, Any] = {}
        for extra_k, extra_v in mcp_val.items():
            if extra_k in ("servers", "timeout") or extra_k == name:
                continue
            if not isinstance(extra_v, Mapping):
                new_mcp[extra_k] = extra_v
        new_mcp.update(preserved_v1_stubs)
        if "timeout" in mcp_val:
            cleaned_mig_t = _clean_opencode_v2_top_timeout(mcp_val.get("timeout"))
            if cleaned_mig_t:
                new_mcp["timeout"] = cleaned_mig_t
        new_mcp["servers"] = servers_map
        parsed["mcp"] = new_mcp
    else:
        mcp_map = dict(mcp_val)
        mcp_map[name] = _merge_opencode_v1_entry(mcp_map.get(name), entry)
        nested_servers = mcp_map.get("servers")
        if (
            isinstance(nested_servers, dict)
            and not _has_v1_mcp_entry_fields(nested_servers)
            and name in nested_servers
        ):
            cleaned_nested = dict(nested_servers)
            cleaned_nested.pop(name, None)
            if cleaned_nested:
                mcp_map["servers"] = cleaned_nested
            else:
                mcp_map.pop("servers", None)
        parsed["mcp"] = mcp_map
    return json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"


def _rewrite_opencode_server_refs(text: str, server_name: str) -> str:
    """Adapt pristine bundled `.opencode/` markdown source text for a custom `--name` in one regex pass.

    Precondition: `text` must be the unmodified bundled source markdown (`OPENCODE_SKILL_SOURCE`,
    `OPENCODE_COMMANDS_DIR/*.md`, or `OPENCODE_AGENT_SOURCE`), never already-rewritten output.
    """
    _validate_opencode_server_name(server_name)
    if server_name == DEFAULT_SERVER_NAME:
        return text

    def _replace_match(match: re.Match[str]) -> str:
        if match.group("backtick_srv") is not None:
            return f"`{server_name}` MCP server"
        if match.group("backtick_tools") is not None:
            return f"`{server_name}` MCP tools"
        if match.group("via_srv") is not None:
            return f"via the {server_name} MCP server"
        if match.group("prefix_doc") is not None:
            return f"`{server_name}_` prefix"
        if match.group("codemode_ns") is not None:
            js_ident = re.sub(r"[^A-Za-z0-9_]", "_", server_name)
            if js_ident and js_ident[0].isdigit():
                js_ident = f"_{js_ident}"
            return f"tools.{js_ident}"
        if match.group("install_flag") is not None:
            return f"--host opencode --name {server_name}"
        return f"{server_name}_{match.group('tool_suffix')}"

    return _OPENCODE_SERVER_REF_RE.sub(_replace_match, text)


def _opencode_doc_defines_server(parsed_doc: Any, name: str) -> bool:
    """Return True when a parsed OpenCode config dict defines `mcp.<name>` or `mcp.servers.<name>`."""
    if not isinstance(parsed_doc, Mapping):
        return False
    mcp_obj = parsed_doc.get("mcp")
    if not isinstance(mcp_obj, Mapping):
        return False
    nested = mcp_obj.get("servers")
    if (
        isinstance(nested, Mapping)
        and not _has_v1_mcp_entry_fields(nested)
        and name in nested
    ):
        return True
    return name in mcp_obj


def _check_opencode_jsonc_shadow_conflict(cfg_dir: Path, name: str, *, target_cfg_path: Path) -> None:
    """Refuse writing `opencode.json` if a sibling `opencode.jsonc` would override `mcp.<name>`."""
    json_path = cfg_dir / "opencode.json"
    jsonc_path = cfg_dir / "opencode.jsonc"
    if target_cfg_path != json_path or not jsonc_path.is_file():
        return
    raw_jsonc = jsonc_path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff").strip()
    if not raw_jsonc:
        return
    try:
        parsed_jsonc = _loads_jsonc(raw_jsonc)
    except ValueError as exc:
        raise OpenCodeConfigError(
            f"opencode.jsonc in {cfg_dir} is not valid JSONC ({exc}); "
            f"OpenCode merges opencode.jsonc on top of opencode.json, so fix opencode.jsonc first"
        ) from exc
    if _opencode_doc_defines_server(parsed_jsonc, name):
        raise OpenCodeConfigError(
            f"opencode.jsonc in {cfg_dir} already defines mcp.{name} "
            f"(which overrides opencode.json in OpenCode); remove or update the entry in opencode.jsonc"
        )


def _sha256_bytes(data: bytes) -> str:
    """Return the hex-encoded SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def _load_opencode_asset_manifest(cfg_dir: Path) -> dict[str, str]:
    """Load recorded SHA-256 hashes of plugin-installed OpenCode assets, returning `{}` if missing or unreadable."""
    manifest_path = cfg_dir / _OPENCODE_ASSET_MANIFEST_NAME
    if not manifest_path.is_file():
        return {}
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(parsed, Mapping):
        return {}
    files = parsed.get("files")
    if not isinstance(files, Mapping):
        return {}
    return {str(k): str(v) for k, v in files.items() if isinstance(k, str) and isinstance(v, str)}


def _save_opencode_asset_manifest(cfg_dir: Path, files_map: Mapping[str, str]) -> None:
    """Persist SHA-256 hashes of plugin-installed OpenCode assets with private `0o600` permissions."""
    manifest_path = cfg_dir / _OPENCODE_ASSET_MANIFEST_NAME
    payload = json.dumps(
        {"version": 1, "files": dict(sorted(files_map.items()))},
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    _write_private(manifest_path, payload, 0o600)


def _write_opencode_asset(
    dst: Path,
    rel_key: str,
    new_text: str,
    asset_backups: list[str],
    manifest_files: dict[str, str],
) -> bool:
    """Write `new_text` to `dst`, backing up raw bytes of any non-matching file and unlinking symlinks first."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        new_bytes = new_text.encode("utf-8")
        new_hash = _sha256_bytes(new_bytes)
        was_symlink = dst.is_symlink()
        if dst.is_file():
            old_bytes = dst.read_bytes()
            if old_bytes == new_bytes and not was_symlink:
                manifest_files[rel_key] = new_hash
                return False
            old_hash = _sha256_bytes(old_bytes)
            if manifest_files.get(rel_key) != old_hash:
                backup = _write_unique_backup_bytes(dst, old_bytes, 0o600)
                asset_backups.append(str(backup))
        if was_symlink:
            dst.unlink()
        dst.write_bytes(new_bytes)
        manifest_files[rel_key] = new_hash
        return True
    except OSError as exc:
        raise OSError(f"{dst}: {exc}") from exc


def _resolve_opencode_target_config(
    cfg_dir: Path,
    mcp_cfg_path: Path,
    entry: dict[str, Any],
    *,
    name: str,
    pool_override: bool | None = None,
) -> tuple[Path, str, bytes, str, str | None]:
    """Read, conflict-check, and merge the target OpenCode config file without writing to disk."""
    try:
        existing_bytes = mcp_cfg_path.read_bytes() if mcp_cfg_path.is_file() else b""
        existing = existing_bytes.decode("utf-8") if existing_bytes else ""
    except (OSError, UnicodeDecodeError) as exc:
        raise OSError(f"{mcp_cfg_path}: {exc}") from exc
    existing_for_upsert = existing
    preserved_jsonc_path: str | None = None
    if mcp_cfg_path == cfg_dir / "opencode.jsonc" and existing.lstrip("\ufeff").strip():
        try:
            json.loads(existing.lstrip("\ufeff").strip())
        except ValueError:
            try:
                parsed_standalone_jsonc = _loads_jsonc(existing)
            except ValueError as exc:
                raise OpenCodeConfigError(f"existing opencode.jsonc is not valid JSONC ({exc}); fix it first") from exc
            if not isinstance(parsed_standalone_jsonc, dict):
                raise OpenCodeConfigError("existing opencode.jsonc top-level value must be a JSON object")
            if _opencode_doc_defines_server(parsed_standalone_jsonc, name):
                raise OpenCodeConfigError(
                    f"existing opencode.jsonc uses comments or trailing commas and already defines mcp.{name}; "
                    f"update mcp.{name} in opencode.jsonc manually so comments are preserved"
                )
            preserved_jsonc_path = str(mcp_cfg_path)
            mcp_cfg_path = cfg_dir / "opencode.json"
            try:
                existing_bytes = mcp_cfg_path.read_bytes() if mcp_cfg_path.is_file() else b""
                existing = existing_bytes.decode("utf-8") if existing_bytes else ""
            except (OSError, UnicodeDecodeError) as exc:
                raise OSError(f"{mcp_cfg_path}: {exc}") from exc
            existing_for_upsert = existing
            standalone_mcp = parsed_standalone_jsonc.get("mcp")
            if not existing and isinstance(standalone_mcp, Mapping) and _is_opencode_v2_servers_envelope(standalone_mcp):
                existing_for_upsert = json.dumps({"$schema": OPENCODE_SCHEMA_URL, "mcp": {"servers": {}}})

    _check_opencode_jsonc_shadow_conflict(cfg_dir, name, target_cfg_path=mcp_cfg_path)
    updated = upsert_opencode_mcp_config(
        existing_for_upsert,
        entry,
        name=name,
        filename=mcp_cfg_path.name,
        pool_override=pool_override,
    )
    return mcp_cfg_path, existing, existing_bytes, updated, preserved_jsonc_path


def _sync_opencode_assets(
    cfg_dir: Path,
    command_sources: list[Path],
    *,
    name: str,
    asset_backups: list[str],
) -> bool:
    """Install OpenCode skill, commands, and subagent, persisting SHA-256 manifest updates even on partial failure."""
    manifest_before = _load_opencode_asset_manifest(cfg_dir)
    manifest_files = dict(manifest_before)
    assets_changed = False
    try:
        skill_text = _rewrite_opencode_server_refs(OPENCODE_SKILL_SOURCE.read_text(encoding="utf-8"), name)
        if _write_opencode_asset(
            cfg_dir / "skills" / PLUGIN_NAME / "SKILL.md",
            f"skills/{PLUGIN_NAME}/SKILL.md",
            skill_text,
            asset_backups,
            manifest_files,
        ):
            assets_changed = True

        for src_cmd in command_sources:
            cmd_text = _rewrite_opencode_server_refs(src_cmd.read_text(encoding="utf-8"), name)
            if _write_opencode_asset(
                cfg_dir / "commands" / src_cmd.name,
                f"commands/{src_cmd.name}",
                cmd_text,
                asset_backups,
                manifest_files,
            ):
                assets_changed = True

        agent_text = _rewrite_opencode_server_refs(OPENCODE_AGENT_SOURCE.read_text(encoding="utf-8"), name)
        if _write_opencode_asset(
            cfg_dir / "agents" / "gemini-delegator.md",
            "agents/gemini-delegator.md",
            agent_text,
            asset_backups,
            manifest_files,
        ):
            assets_changed = True
    except OSError:
        if manifest_files != manifest_before:
            try:
                _save_opencode_asset_manifest(cfg_dir, manifest_files)
            except OSError:
                pass
        raise
    if manifest_files != manifest_before:
        _save_opencode_asset_manifest(cfg_dir, manifest_files)
    return assets_changed


def install_opencode(
    *,
    name: str = DEFAULT_SERVER_NAME,
    profile: str = "balanced",
    auth_mode: str | None = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    gateway_wire: str | None = None,
    gateway_audience: str | None = None,
    telemetry_sink: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    strict: bool = False,
    dry_run: bool = False,
    opencode_config_dir: Path | None = None,
    python_bin: str | None = None,
    v2: bool = False,
    pool: bool | None = None,
    codemode: bool | None = None,
    protocol: str | None = None,
) -> dict[str, Any]:
    """Register cc-agy-plugin in OpenCode's global config, skills, commands, and agents directories."""
    cfg_dir = resolve_opencode_config_dir(explicit_dir=opencode_config_dir)
    active_env = os.environ if opencode_config_dir is None else None
    mcp_cfg_path = resolve_opencode_config_path(cfg_dir, env=active_env)
    commands_dst_dir = cfg_dir / "commands"

    eff_v2 = bool(v2 or pool is not None or codemode is not None or protocol is not None)
    command_sources = sorted(OPENCODE_COMMANDS_DIR.glob("*.md")) if OPENCODE_COMMANDS_DIR.is_dir() else []
    result: dict[str, Any] = {
        "host": "opencode",
        "config_path": str(mcp_cfg_path),
        "skill_path": str(cfg_dir / "skills" / PLUGIN_NAME / "SKILL.md"),
        "commands_dir": str(commands_dst_dir),
        "command_paths": [str(commands_dst_dir / src.name) for src in command_sources],
        "agent_path": str(cfg_dir / "agents" / "gemini-delegator.md"),
        "server_name": name,
        "mcp_key_path": f"mcp.servers.{name}" if eff_v2 else f"mcp.{name}",
    }
    try:
        entry = build_opencode_server_entry(
            python_bin=python_bin,
            profile=profile,
            auth_mode=auth_mode,
            project=project,
            location=location,
            base_url=base_url,
            gateway_wire=gateway_wire,
            gateway_audience=gateway_audience,
            telemetry_sink=telemetry_sink,
            agy_cli_path=agy_cli_path,
            max_daily_cents=max_daily_cents,
            allowed_projects=allowed_projects,
            blocked_projects=blocked_projects,
            no_auto_apply=no_auto_apply,
            strict=strict,
            v2=eff_v2,
            pool=pool,
            codemode=codemode,
            protocol=protocol,
        )
    except ValueError as exc:
        result["status"] = "refused"
        result["error"] = str(exc)
        return result
    result["entry"] = entry
    if active_env is not None and (active_env.get("OPENCODE_CONFIG") or "").strip() and not mcp_cfg_path.is_absolute():
        result["status"] = "refused"
        result["error"] = (
            f"OPENCODE_CONFIG ({mcp_cfg_path}) must be an absolute path (or start with ~/...) "
            "so the installer and OpenCode resolve the same config file"
        )
        return result

    try:
        _validate_opencode_server_name(name)
    except OpenCodeConfigError as exc:
        result["status"] = "refused"
        result["error"] = str(exc)
        return result

    missing_assets = [
        str(asset_path)
        for asset_path in (
            OPENCODE_SKILL_SOURCE,
            OPENCODE_AGENT_SOURCE,
            *(OPENCODE_COMMANDS_DIR / f"{req_cmd}.md" for req_cmd in _REQUIRED_OPENCODE_COMMANDS),
        )
        if not asset_path.is_file()
    ]
    if missing_assets:
        result["status"] = "failed"
        result["error"] = f"OpenCode bundle assets missing from plugin checkout: {', '.join(missing_assets)}"
        return result

    asset_backups: list[str] = []
    try:
        mcp_cfg_path, existing, existing_bytes, updated, preserved_jsonc = _resolve_opencode_target_config(
            cfg_dir, mcp_cfg_path, entry, name=name, pool_override=pool if eff_v2 else None
        )
        if preserved_jsonc is not None:
            result["preserved_jsonc_path"] = preserved_jsonc
            result["config_path"] = str(mcp_cfg_path)
        updated_mcp = json.loads(updated).get("mcp") or {}
        servers_sub = updated_mcp.get("servers")
        wrote_v2_servers = isinstance(servers_sub, dict) and name in servers_sub
        result["mcp_key_path"] = (
            f"mcp.servers.{name}" if wrote_v2_servers else f"mcp.{name}"
        )
        merged_self = servers_sub.get(name) if wrote_v2_servers else updated_mcp.get(name)
        if isinstance(merged_self, dict):
            result["entry"] = merged_self
        if dry_run:
            result["status"] = "dry_run"
            return result

        updated_bytes = updated.encode("utf-8")
        config_changed = existing_bytes != updated_bytes
        mcp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        if (
            mcp_cfg_path.is_file()
            and not mcp_cfg_path.is_symlink()
            and (mcp_cfg_path.stat().st_mode & 0o777) != 0o600
        ):
            os.chmod(mcp_cfg_path, 0o600)
            result["mode_tightened"] = True
        if existing_bytes and config_changed:
            backup = _write_unique_backup_bytes(mcp_cfg_path, existing_bytes, 0o600)
            result["backup_path"] = str(backup)
        if config_changed:
            _write_private(mcp_cfg_path, updated, 0o600)

        assets_changed = _sync_opencode_assets(
            cfg_dir,
            command_sources,
            name=name,
            asset_backups=asset_backups,
        )
    except OpenCodeConfigError as exc:
        result["status"] = "refused"
        result["error"] = str(exc)
        return result
    except (OSError, UnicodeDecodeError) as exc:
        if asset_backups:
            result["asset_backups"] = asset_backups
        result["status"] = "failed"
        result["error"] = f"OpenCode config/asset I/O failed: {exc}"
        return result

    if asset_backups:
        result["asset_backups"] = asset_backups
    result["status"] = "installed" if (config_changed or assets_changed) else "unchanged"
    return result


def preflight(
    env: dict[str, str] | None = None,
    *,
    auth_mode: str | None = None,
    project: str | None = None,
    base_url: str | None = None,
    agy_cli_path: str | None = None,
) -> dict[str, Any]:
    """Zero-network readiness: Python, SDKs / `agy` CLI, git and credentials."""
    from server.agy_cli import inspect_agy_cli_offline
    from server.regions import (
        _clean_base_url_candidate,
        _resolve_explicit_project_only,
        _should_auto_detect_agy_cli,
        resolve_api_key,
        resolve_configured_auth_mode,
        resolve_gateway_base_url,
        resolve_vertex_base_url,
    )

    source = dict(env if env is not None else os.environ)
    if agy_cli_path:
        source["CC_AGY_CLI_BIN"] = agy_cli_path
    if base_url:
        source["CC_AGY_VERTEX_BASE_URL"] = base_url

    py_bin = resolve_python_bin()
    issues: list[str] = []
    warnings: list[str] = []
    if sys.version_info < (3, 10):
        issues.append(f"Python 3.10+ is required (found {sys.version.split()[0]}).")
    if not shutil.which("git"):
        issues.append("git is not on PATH (worker jobs run in git copies of your repository).")

    try:
        cfg_mode = resolve_configured_auth_mode(auth_mode, env=source)
    except ValueError:
        cfg_mode = None
    norm_auth = cfg_mode or _normalize_auth_choice(auth_mode)

    resolved_key = resolve_api_key(env=source)
    resolved_gateway = resolve_gateway_base_url(base_url, env=source)
    has_explicit_or_plugin_gateway = bool(
        _clean_base_url_candidate(base_url)
        or resolve_vertex_base_url(env=source)
        or _clean_base_url_candidate(source.get("CC_AGY_GEMINI_BASE_URL"))
    )
    explicit_proj_arg = _resolve_explicit_project_only(project, env={}) if project else None
    resolved_proj = _resolve_explicit_project_only(project, env=source)

    has_key = bool(resolved_key)
    has_gateway = bool(resolved_gateway)
    has_project = bool(resolved_proj or has_gateway)

    agy_cli_status: dict[str, Any] | None = None
    if norm_auth == "gemini_enterprise":
        auth = "gemini_enterprise"
        agy_cli_status = inspect_agy_cli_offline(source)
        if not agy_cli_status.get("ok"):
            err = str(agy_cli_status.get("error") or "Antigravity CLI (`agy`) sign-in is missing.")
            rem = str(agy_cli_status.get("remediation") or "")
            issues.append(f"{err} {rem}".strip())
    elif norm_auth == "api_key":
        auth = "api_key" if has_key else "missing"
        if not has_key:
            issues.append("Export `GEMINI_API_KEY` in your shell for API-key mode.")
        conflict_msg = (
            "Explicit `--auth api-key` (`CC_AGY_AUTH_MODE=api_key`) always connects directly to Google AI Studio "
            "and ignores `--base-url` / gateway URLs. Use `--auth auto` (with `GOOGLE_GEMINI_BASE_URL` or "
            "`CC_AGY_GEMINI_BASE_URL`) or `--auth gateway` to route through a gateway."
        )
        if has_explicit_or_plugin_gateway:
            issues.append(conflict_msg)
        elif has_gateway:
            warnings.append(conflict_msg)
    elif norm_auth == "gateway":
        auth = "gateway" if has_gateway else "missing"
        if not has_gateway:
            issues.append(
                "Pass `--base-url` or export `CC_AGY_VERTEX_BASE_URL` / `CC_AGY_GEMINI_BASE_URL` for gateway mode."
            )
    elif norm_auth == "vertex":
        auth = "vertex" if has_project else "missing"
        if not has_project:
            issues.append("Pass `--project` or export `GOOGLE_CLOUD_PROJECT` for Vertex AI mode.")
    elif explicit_proj_arg:
        auth = "vertex"
    elif has_gateway:
        auth = "gateway"
    elif has_key:
        auth = "api_key"
    elif has_project:
        auth = "vertex"
    elif _should_auto_detect_agy_cli(env if not agy_cli_path else source, source) and inspect_agy_cli_offline(source).get("ok"):
        auth = "gemini_enterprise"
        agy_cli_status = inspect_agy_cli_offline(source)
    else:
        auth = "missing"
        issues.append(
            "No credentials found. Pass `--auth gemini-enterprise` (uses the standalone `agy` CLI with your "
            "Google sign-in), export `GEMINI_API_KEY` (AI Studio), pass `--project` / export `GOOGLE_CLOUD_PROJECT` "
            "(Vertex AI), or pass `--base-url` (AI gateway)."
        )

    sdk_ok = python_has_plugin_sdks(py_bin)
    if not sdk_ok and auth != "gemini_enterprise":
        issues.append("Antigravity SDK missing. Run `python3 -m server.install` (without `--no-bootstrap`).")

    out: dict[str, Any] = {
        "python_bin": py_bin,
        "auth_mode": auth,
        "issues": issues,
        "warnings": warnings,
        "ready": not issues,
    }
    if agy_cli_status is not None:
        out["agy_cli"] = {
            "binary_found": bool(agy_cli_status.get("binary_found")),
            "agy_bin": agy_cli_status.get("agy_bin"),
            "auth_configured": bool(agy_cli_status.get("auth_configured")),
            "credential_source": agy_cli_status.get("credential_source"),
            "companion_project": agy_cli_status.get("companion_project"),
            "tier_id": agy_cli_status.get("tier_id"),
        }
    return out


def run_interactive_wizard(
    *,
    input_fn: Any = input,
    output_fn: Any = print,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Ask 4 plain-English setup questions when cc-agy-install runs with no flags in a terminal."""
    from server.agy_cli import inspect_agy_cli_offline, managed_agy_bin_dir, MANAGED_CLI_BIN_NAME

    source = env if env is not None else os.environ
    output_fn("cc-agy-plugin quick setup (press Enter for the default in brackets)\n")

    host_raw = str(
        input_fn("1. Coding tool [1=All, 2=Codex, 3=Claude Code, 4=Antigravity, 5=Cursor, 6=OpenCode] (1): ") or ""
    ).strip().lower()
    host_map = {
        "": "all",
        "1": "all",
        "all": "all",
        "both": "all",
        "2": "codex",
        "codex": "codex",
        "3": "claude",
        "claude": "claude",
        "4": "agy",
        "agy": "agy",
        "antigravity": "agy",
        "jetski": "agy",
        "5": "cursor",
        "cursor": "cursor",
        "6": "opencode",
        "opencode": "opencode",
        "open-code": "opencode",
    }
    host = host_map.get(host_raw, "all")

    from server.regions import (
        _clean_base_url_candidate,
        _should_auto_detect_agy_cli,
        resolve_api_key,
        resolve_bifrost_virtual_key,
        resolve_gateway_base_url,
        resolve_gemini_base_url,
        resolve_static_vertex_base_url_token,
        resolve_vertex_base_url,
    )

    detected_proj = str(source.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    detected_gw = str(resolve_gateway_base_url(env=source) or "").strip()
    detected_vtx_gw = str(resolve_vertex_base_url(env=source) or "").strip()
    detected_gem_gw = str(resolve_gemini_base_url(env=source) or "").strip()
    has_dedicated_gw_url = bool(
        detected_vtx_gw or _clean_base_url_candidate(source.get("CC_AGY_GEMINI_BASE_URL"))
    )
    has_gw_cred = bool(
        resolve_static_vertex_base_url_token(env=source)
        or resolve_bifrost_virtual_key(env=source)
        or str(source.get("CC_AGY_VERTEX_BASE_URL_AUDIENCE") or "").strip()
    )
    has_real_key = bool(resolve_api_key(env=source))
    explicit_ent = _normalize_auth_choice(str(source.get("CC_AGY_AUTH_MODE") or "")) == "gemini_enterprise"
    auto_ent = _should_auto_detect_agy_cli(env, source) and bool(inspect_agy_cli_offline(source).get("ok"))
    has_key = any(str(source.get(k) or "").strip() for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    is_gemini_cli_key_pairing = bool(
        detected_gw and not has_dedicated_gw_url and has_real_key and not has_gw_cred
    )
    if explicit_ent:
        default_auth_label = "gemini-enterprise"
    elif detected_gw and not is_gemini_cli_key_pairing:
        default_auth_label = "gateway"
    elif has_key:
        default_auth_label = "api_key"
    elif detected_proj:
        default_auth_label = "vertex"
    elif auto_ent or env is None:
        default_auth_label = "gemini-enterprise"
    else:
        default_auth_label = "api_key"
    auth_raw = str(
        input_fn(
            "2. Gemini connection [1=API key in shell, 2=Google Cloud project, 3=AI gateway URL, "
            f"4=Gemini Enterprise subscription (`agy` CLI)] ({default_auth_label}): "
        )
        or ""
    ).strip().lower()
    auth_flag: str | None = None
    project: str | None = None
    base_url: str | None = None
    gateway_wire: str | None = None
    if auth_raw in ("4", "gemini-enterprise", "gemini_enterprise", "enterprise", "agy", "agy-cli", "agy_cli") or (
        not auth_raw and default_auth_label == "gemini-enterprise"
    ):
        auth_flag = "gemini-enterprise"
        ent_status = inspect_agy_cli_offline(source)
        if not ent_status.get("auth_configured"):
            agy_hint = ent_status.get("agy_bin") or str(managed_agy_bin_dir(source) / MANAGED_CLI_BIN_NAME)
            output_fn(
                f"   Tip: after setup installs `agy`, run `{agy_hint}` once in a terminal to sign in with your Google work account."
            )
    elif auth_raw in ("2", "vertex", "gcp") or (not auth_raw and default_auth_label == "vertex"):
        proj_prompt = f"   Google Cloud project ID ({detected_proj}): " if detected_proj else "   Google Cloud project ID: "
        entered_proj = str(input_fn(proj_prompt) or "").strip() or detected_proj
        if entered_proj:
            project = entered_proj
    elif auth_raw in ("3", "gateway", "proxy") or (not auth_raw and default_auth_label == "gateway"):
        gw_prompt = f"   AI gateway base URL ({detected_gw}): " if detected_gw else "   AI gateway base URL: "
        entered_gw = str(input_fn(gw_prompt) or "").strip() or detected_gw
        if entered_gw:
            base_url = entered_gw
            if not detected_vtx_gw and detected_gem_gw and entered_gw == detected_gem_gw:
                gateway_wire = "gemini"
        if not has_gw_cred:
            output_fn(
                "   Tip: export CC_AGY_VERTEX_BASE_URL_TOKEN='your-token' or CC_AGY_BIFROST_VIRTUAL_KEY='your-vk' "
                "in your shell (or pass --gateway-audience for OIDC)."
            )
    elif not has_key:
        output_fn("   Tip: export GEMINI_API_KEY='your-key' in your shell (keys are never saved to config files).")

    prof_raw = str(
        input_fn("3. Model preset [1=balanced (Flash then Carbon), 2=economy (Flash only), 3=quality (Argon first)] (1): ")
        or ""
    ).strip().lower()
    prof_map = {
        "": "balanced",
        "1": "balanced",
        "balanced": "balanced",
        "2": "economy",
        "economy": "economy",
        "3": "quality",
        "quality": "quality",
    }
    profile = prof_map.get(prof_raw, "balanced")

    cap_raw = str(input_fn("4. Optional daily budget cap in cents, e.g. 500 for $5/day (Enter for none): ") or "").strip()
    built = ["--host", host, "--profile", profile]
    if auth_flag:
        built.extend(["--auth", auth_flag])
    if project:
        built.extend(["--project", project])
    if base_url:
        built.extend(["--base-url", base_url])
    if gateway_wire:
        built.extend(["--gateway-wire", gateway_wire])
    if cap_raw:
        built.extend(["--max-daily-cents", cap_raw])
    return built


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv and argv is None and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            raw_argv = run_interactive_wizard()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            return 1

    parser = argparse.ArgumentParser(
        prog="cc-agy-install",
        description="Set up cc-agy-plugin for Claude Code, Codex, Cursor, OpenCode, and/or Antigravity.",
    )
    parser.add_argument(
        "--host",
        choices=("codex", "claude", "cursor", "opencode", "agy", "antigravity", "jetski", "all"),
        default="all",
    )
    parser.add_argument(
        "--auth",
        choices=AUTH_CHOICES,
        default="auto",
        help="Connection mode: 'gemini-enterprise' / 'agy-cli' (included seat quota via standalone `agy` CLI), 'api-key', 'vertex', 'gateway', or 'auto'.",
    )
    parser.add_argument("--profile", choices=COST_PROFILES, default="balanced",
                        help="economy: Flash only; balanced: Flash, escalating to Carbon Zero-Think on failure; quality: Argon first.")
    parser.add_argument("--project", default=None, help="Google Cloud project ID (required for pay-as-you-go Vertex AI).")
    parser.add_argument("--location", default=None, help="Vertex AI region (for example 'global' or 'us-central1').")
    parser.add_argument("--base-url", default=None, help="Optional AI gateway base URL (sets CC_AGY_VERTEX_BASE_URL).")
    parser.add_argument(
        "--gateway-wire",
        choices=GATEWAY_WIRE_CLI_CHOICES,
        default=None,
        help="Gateway wire protocol: 'auto' (detects /genai and /gemini vs Vertex paths), 'vertex' / 'litellm-vertex', or 'gemini' / 'litellm-gemini' / 'bifrost'.",
    )
    parser.add_argument("--gateway-audience", default=None,
                        help="Optional OIDC target audience URL (or 'auto') for zero-key Cloud Run / WIF gateway auth.")
    parser.add_argument("--telemetry-sink", default=None,
                        help="Optional local NDJSON path or https:// webhook URL for opt-in receipt telemetry.")
    parser.add_argument("--agy-cli-path", default=None, help="Optional path to an external Antigravity CLI (`agy`) binary.")
    parser.add_argument("--install-agy-cli", action="store_true",
                        help="Download and verify the standalone Antigravity CLI (`agy`) into ~/.local/share/cc-agy-plugin/bin.")
    parser.add_argument("--max-daily-cents", type=float, default=None,
                        help="Optional daily spend limit in US cents (for example 500 for $5.00/day).")
    parser.add_argument("--allowed-projects", default=None,
                        help="Optional comma-separated allowlist of GCP project IDs (sets CC_AGY_ALLOWED_PROJECTS).")
    parser.add_argument("--blocked-projects", default=None,
                        help="Optional comma-separated blocklist of GCP project IDs (sets CC_AGY_BLOCKED_PROJECTS).")
    parser.add_argument("--no-auto-apply", action="store_true",
                        help="Require manual diff review (`get_diff` then `delegate_apply`) before any worker change touches your repo.")
    parser.add_argument("--strict", action="store_true",
                        help="Enable strict orchestrator mode in Claude Code userConfig.")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Do not automatically install ~/.local/share/cc-agy-plugin/venv or standalone `agy` CLI.")
    parser.add_argument(
        "--name",
        default=DEFAULT_SERVER_NAME,
        help="MCP server name for Codex, Cursor, OpenCode, and Antigravity (default: agy).",
    )
    parser.add_argument("--opencode-v2", action="store_true",
                        help="Write OpenCode 2.2.6+ mcp.servers.<name> envelope with split startup/execution timeouts.")
    parser.add_argument("--opencode-pool", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable or disable OpenCode 2.2.6+ MCP daemon pooling (pool: true/false; enable only for single-workspace use).")
    parser.add_argument("--opencode-codemode", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable or disable OpenCode 2.2.6+ Code Mode (codemode: true/false) on the MCP server entry.")
    parser.add_argument("--opencode-protocol", choices=sorted(_OPENCODE_V2_PROTOCOLS), default=None,
                        help="Optional OpenCode 2.2.6+ MCP protocol negotiation mode ('auto' or 'legacy').")
    parser.add_argument("--non-interactive", action="store_true", help="Never prompt, even when run with no flags in a TTY.")
    parser.add_argument("--doctor", action="store_true", help="Run environment, auth, and telemetry diagnostics as JSON and exit.")
    parser.add_argument("--probe-oidc", action="store_true", help="With --doctor, mint a live OIDC token once to verify and display the resolved principal.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing anything.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    args = parser.parse_args(raw_argv)
    if args.probe_oidc and not args.doctor:
        parser.error("--probe-oidc requires --doctor")
    if args.doctor:
        from server.doctor import run_doctor

        report = run_doctor(workspace_root=os.getcwd(), probe_oidc=bool(args.probe_oidc))
        print(json.dumps(report, indent=2))
        return 0 if report.get("healthy") else 1
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("--name may only contain letters, digits, '-' and '_'")
    if args.max_daily_cents is not None:
        import math

        if not math.isfinite(args.max_daily_cents) or args.max_daily_cents <= 0:
            parser.error("--max-daily-cents must be a finite number greater than 0")
    try:
        _validate_gateway_base_url_no_secrets(args.base_url)
        norm_gateway_wire = _normalize_gateway_wire_choice(args.gateway_wire)
    except ValueError as exc:
        parser.error(str(exc))
    if args.gateway_audience:
        from server.regions import resolve_gateway_base_url, resolve_vertex_base_url_audience

        try:
            resolve_vertex_base_url_audience(
                args.gateway_audience,
                base_url=resolve_gateway_base_url(args.base_url, env=os.environ),
                env=os.environ,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.telemetry_sink:
        from server.telemetry import validate_webhook_sink_url, _resolve_local_sink_path

        try:
            if args.telemetry_sink.strip().lower().startswith(("http://", "https://")):
                validate_webhook_sink_url(args.telemetry_sink, check_dns=False)
            else:
                _resolve_local_sink_path(args.telemetry_sink)
        except ValueError as exc:
            parser.error(str(exc))

    bootstrap_info: dict[str, Any] | None = None
    agy_cli_bootstrap: dict[str, Any] | None = None
    norm_requested_auth = _normalize_auth_choice(args.auth)
    if not args.dry_run and not args.no_bootstrap:
        if norm_requested_auth == "gemini_enterprise" or args.install_agy_cli:
            from server.agy_cli import ensure_agy_cli_installed

            boot_env = dict(os.environ)
            if args.agy_cli_path:
                boot_env["CC_AGY_CLI_BIN"] = args.agy_cli_path
            agy_cli_bootstrap = ensure_agy_cli_installed(
                PLUGIN_ROOT,
                env=boot_env,
                output_fn=None if args.json else print,
            )
        if norm_requested_auth != "gemini_enterprise":
            bootstrap_info = ensure_sdk_venv(
                PLUGIN_ROOT,
                output_fn=None if args.json else print,
            )

    pre = preflight(
        auth_mode=args.auth,
        project=args.project,
        base_url=args.base_url,
        agy_cli_path=args.agy_cli_path,
    )
    if bootstrap_info and not bootstrap_info.get("ok") and bootstrap_info.get("error") and pre.get("auth_mode") != "gemini_enterprise":
        pre["issues"].append(str(bootstrap_info["error"]))
        pre["ready"] = False
    if agy_cli_bootstrap and not agy_cli_bootstrap.get("ok") and agy_cli_bootstrap.get("error"):
        err_str = str(agy_cli_bootstrap["error"])
        if err_str not in pre["issues"]:
            pre["issues"].append(err_str)
        pre["ready"] = False

    report: dict[str, Any] = {
        "preflight": pre,
        **({"bootstrap": bootstrap_info} if bootstrap_info is not None else {}),
        **({"agy_cli_bootstrap": agy_cli_bootstrap} if agy_cli_bootstrap is not None else {}),
        "hosts": [],
    }
    if args.host in ("codex", "all"):
        report["hosts"].append(
            install_codex(
                name=args.name,
                profile=args.profile,
                auth_mode=args.auth,
                project=args.project,
                location=args.location,
                base_url=args.base_url,
                gateway_wire=norm_gateway_wire,
                gateway_audience=args.gateway_audience,
                telemetry_sink=args.telemetry_sink,
                agy_cli_path=args.agy_cli_path,
                max_daily_cents=args.max_daily_cents,
                allowed_projects=args.allowed_projects,
                blocked_projects=args.blocked_projects,
                no_auto_apply=args.no_auto_apply,
                dry_run=args.dry_run,
            )
        )
    if args.host in ("claude", "all"):
        report["hosts"].append(
            install_claude(
                dry_run=args.dry_run,
                profile=args.profile,
                auth_mode=args.auth,
                project=args.project,
                location=args.location,
                base_url=args.base_url,
                gateway_wire=norm_gateway_wire,
                gateway_audience=args.gateway_audience,
                telemetry_sink=args.telemetry_sink,
                agy_cli_path=args.agy_cli_path,
                max_daily_cents=args.max_daily_cents,
                allowed_projects=args.allowed_projects,
                blocked_projects=args.blocked_projects,
                no_auto_apply=args.no_auto_apply,
                strict=args.strict,
            )
        )
    if args.host in ("cursor", "all"):
        report["hosts"].append(
            install_cursor(
                name=args.name,
                profile=args.profile,
                auth_mode=args.auth,
                project=args.project,
                location=args.location,
                base_url=args.base_url,
                gateway_wire=norm_gateway_wire,
                gateway_audience=args.gateway_audience,
                telemetry_sink=args.telemetry_sink,
                agy_cli_path=args.agy_cli_path,
                max_daily_cents=args.max_daily_cents,
                allowed_projects=args.allowed_projects,
                blocked_projects=args.blocked_projects,
                no_auto_apply=args.no_auto_apply,
                strict=args.strict,
                dry_run=args.dry_run,
            )
        )
    if args.host in ("opencode", "all"):
        report["hosts"].append(
            install_opencode(
                name=args.name,
                profile=args.profile,
                auth_mode=args.auth,
                project=args.project,
                location=args.location,
                base_url=args.base_url,
                gateway_wire=norm_gateway_wire,
                gateway_audience=args.gateway_audience,
                telemetry_sink=args.telemetry_sink,
                agy_cli_path=args.agy_cli_path,
                max_daily_cents=args.max_daily_cents,
                allowed_projects=args.allowed_projects,
                blocked_projects=args.blocked_projects,
                no_auto_apply=args.no_auto_apply,
                strict=args.strict,
                dry_run=args.dry_run,
                v2=bool(
                    args.opencode_v2
                    or args.opencode_pool is not None
                    or args.opencode_codemode is not None
                    or args.opencode_protocol is not None
                ),
                pool=args.opencode_pool,
                codemode=args.opencode_codemode,
                protocol=args.opencode_protocol,
            )
        )
    if args.host in ("agy", "antigravity", "jetski", "all"):
        report["hosts"].append(
            install_agy(
                name=args.name,
                profile=args.profile,
                auth_mode=args.auth,
                project=args.project,
                location=args.location,
                base_url=args.base_url,
                gateway_wire=norm_gateway_wire,
                gateway_audience=args.gateway_audience,
                telemetry_sink=args.telemetry_sink,
                agy_cli_path=args.agy_cli_path,
                max_daily_cents=args.max_daily_cents,
                allowed_projects=args.allowed_projects,
                blocked_projects=args.blocked_projects,
                no_auto_apply=args.no_auto_apply,
                strict=args.strict,
                dry_run=args.dry_run,
            )
        )

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if _report_ok(report) else 1

    print(f"cc-agy-plugin at {PLUGIN_ROOT}")
    print(f"  python: {pre['python_bin']}  auth: {pre['auth_mode']}  profile: {args.profile}")
    for issue in pre["issues"]:
        print(f"  ! {issue}")
    for warning in pre.get("warnings", []):
        print(f"  ~ {warning}")
    for host in report["hosts"]:
        if host["host"] == "codex":
            print(f"\nCodex: {host['status']} -> {host['config_path']} [mcp_servers.{host['server_name']}]")
            if host.get("error"):
                print(f"  ! {host['error']}")
            if host.get("backup_path"):
                print(f"  backup: {host['backup_path']}")
            if host.get("codex_verify_error"):
                print(f"  ! codex mcp get failed: {host['codex_verify_error']}")
            if host["status"] == "dry_run":
                print(host["block"])
            print('  Next: run `codex` in your repository and ask: "Run doctor to verify my Gemini setup."')
        elif host["host"] == "cursor":
            print(f"\nCursor: {host['status']} -> {host['config_path']} [mcpServers.{host['server_name']}]")
            if host.get("error"):
                print(f"  ! {host['error']}")
            if host.get("backup_path"):
                print(f"  backup: {host['backup_path']}")
            print(f"  skill: {host['skill_path']}")
            print('  Next: open your repository in Cursor and ask: "Run doctor to verify my Gemini setup."')
        elif host["host"] == "opencode":
            key_path = host.get("mcp_key_path") or f"mcp.{host['server_name']}"
            print(f"\nOpenCode: {host['status']} -> {host['config_path']} [{key_path}]")
            if host.get("error"):
                print(f"  ! {host['error']}")
            if host.get("preserved_jsonc_path"):
                print(f"  preserved jsonc: {host['preserved_jsonc_path']}")
            if host.get("backup_path"):
                print(f"  backup: {host['backup_path']}")
            for asset_bak in host.get("asset_backups") or []:
                print(f"  asset backup: {asset_bak}")
            print(f"  skill: {host['skill_path']}")
            print(f"  commands: {host['commands_dir']} (/delegate, /doctor, /jobs, /cost)")
            print(f"  subagent: {host['agent_path']}")
            print(
                '  Next: run `opencode` in your repository and type `/doctor` '
                '(or ask: "Run doctor to verify my Gemini setup.").'
            )
        elif host["host"] == "agy":
            print(f"\nAntigravity: {host['status']} -> {host['config_path']} [mcpServers.{host['server_name']}]")
            if host.get("error"):
                print(f"  ! {host['error']}")
            if host.get("backup_path"):
                print(f"  backup: {host['backup_path']}")
            print(f"  plugin bundle: {host['plugin_dir']}")
            print(f"  or via CLI: $ {host['cli_command']}")
            print('  Next: start `agy` in your repository and ask: "Run doctor to verify my Gemini setup."')
        else:
            print(f"\nClaude Code: {host['status']}")
            for cmd in host["commands"]:
                print(f"  $ {cmd}")
            print(f"  or for one session: {host['fallback']}")
            print(f"  {host['note']}")
    return 0 if _report_ok(report) else 1


def _report_ok(report: dict[str, Any]) -> bool:
    return bool(report["preflight"]["ready"]) and all(
        h.get("status") not in ("refused", "failed") for h in report["hosts"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
