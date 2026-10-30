#!/usr/bin/env python3
"""One-command setup of cc-agy-plugin for Claude Code, Codex, and Antigravity.

Usage (from a clone of this repository):

    python3 -m server.install --host claude --auth gemini-enterprise
    python3 -m server.install --host codex  --auth gemini-enterprise
    python3 -m server.install --host agy    --auth gemini-enterprise
    python3 -m server.install --host all --profile quality

Automatically sets up a PEP 668-safe per-user Python virtualenv (`~/.local/share/cc-agy-plugin/venv`)
when system Python does not already have `google-antigravity` and `google-genai`, and forwards
non-secret configuration to Codex (`~/.codex/config.toml`), Claude Code (`--config` / `userConfig`),
and Antigravity (`~/.gemini/config/mcp_config.json` and `~/.gemini/config/plugins/cc-agy-plugin/`).
API keys and OAuth tokens are never written into config files on disk.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from server.runtime_env import ensure_sdk_venv, python_has_plugin_sdks, resolve_sdk_python  # noqa: E402

SERVER_ENTRY = PLUGIN_ROOT / "server" / "main.py"
CODEX_SKILL_SOURCE = PLUGIN_ROOT / "plugins" / "cc-agy-plugin" / "skills" / "vertex" / "SKILL.md"
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
    "CC_AGY_VERTEX_BASE_URL_TOKEN",
    "CC_AGY_MAX_DAILY_CENTS",
    "CC_AGY_ALLOWED_PROJECTS",
    "CC_AGY_BLOCKED_PROJECTS",
    "CC_AGY_AUTO_APPLY_ON_PASS",
    "CC_AGY_ALLOW_IN_PLACE",
)
# Worker jobs run for minutes; Codex's default MCP tool timeout is far shorter.
CODEX_TOOL_TIMEOUT_SEC = 1800
CODEX_STARTUP_TIMEOUT_SEC = 30


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


def build_codex_server_block(
    *,
    name: str = DEFAULT_SERVER_NAME,
    python_bin: str | None = None,
    profile: str = "balanced",
    auth_mode: str | None = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
) -> str:
    """Render the `[mcp_servers.<name>]` TOML block for `$CODEX_HOME/config.toml`."""
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
    if base_url:
        env["CC_AGY_VERTEX_BASE_URL"] = base_url
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
    """Write `text` to `path`, creating it with `mode` and tightening an existing file to `mode`."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, mode)


def install_codex(
    *,
    name: str,
    profile: str,
    auth_mode: str | None = None,
    project: str | None,
    location: str | None,
    base_url: str | None = None,
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
        backup = cfg.with_name(f"config.toml.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        _write_private(backup, existing, 0o600)
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
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Build non-secret Claude Code `userConfig` key/value pairs matching `.claude-plugin/plugin.json`."""
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
    if base_url:
        cfg["vertex_base_url"] = base_url
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
    agy_cli_path: str | None = None,
    max_daily_cents: float | None = None,
    allowed_projects: str | None = None,
    blocked_projects: str | None = None,
    no_auto_apply: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Build the stdio MCP server entry for Antigravity's `~/.gemini/config/mcp_config.json`."""
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
    if base_url:
        env["CC_AGY_VERTEX_BASE_URL"] = base_url
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
        backup = mcp_cfg_path.with_name(f"mcp_config.json.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        _write_private(backup, existing, 0o600)
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
        _resolve_explicit_project_only,
        _should_auto_detect_agy_cli,
        resolve_api_key,
        resolve_configured_auth_mode,
        resolve_vertex_base_url,
    )

    source = dict(env if env is not None else os.environ)
    if agy_cli_path:
        source["CC_AGY_CLI_BIN"] = agy_cli_path
    if base_url:
        source["CC_AGY_VERTEX_BASE_URL"] = base_url

    py_bin = resolve_python_bin()
    issues: list[str] = []
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
    resolved_gateway = resolve_vertex_base_url(base_url, env=source)
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
    elif norm_auth == "gateway":
        auth = "gateway" if has_gateway else "missing"
        if not has_gateway:
            issues.append("Pass `--base-url` or export `CC_AGY_VERTEX_BASE_URL` for gateway mode.")
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

    out: dict[str, Any] = {"python_bin": py_bin, "auth_mode": auth, "issues": issues, "ready": not issues}
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
        input_fn("1. Coding tool [1=All, 2=Codex, 3=Claude Code, 4=Antigravity] (1): ") or ""
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
    }
    host = host_map.get(host_raw, "all")

    from server.regions import _should_auto_detect_agy_cli

    detected_proj = str(source.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    detected_gw = str(source.get("CC_AGY_VERTEX_BASE_URL") or source.get("GOOGLE_VERTEX_BASE_URL") or "").strip()
    explicit_ent = _normalize_auth_choice(str(source.get("CC_AGY_AUTH_MODE") or "")) == "gemini_enterprise"
    auto_ent = _should_auto_detect_agy_cli(env, source) and bool(inspect_agy_cli_offline(source).get("ok"))
    has_key = any(str(source.get(k) or "").strip() for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    if explicit_ent:
        default_auth_label = "gemini-enterprise"
    elif detected_gw:
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
    elif not has_key:
        output_fn("   Tip: export GEMINI_API_KEY='your-key' in your shell (keys are never saved to config files).")

    prof_raw = str(
        input_fn("3. Model preset [1=balanced (Flash then Argon), 2=economy (Flash only), 3=quality (Argon first)] (1): ")
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
        description="Set up cc-agy-plugin for Claude Code, Codex, and/or Antigravity.",
    )
    parser.add_argument(
        "--host",
        choices=("codex", "claude", "agy", "antigravity", "jetski", "all"),
        default="all",
    )
    parser.add_argument(
        "--auth",
        choices=AUTH_CHOICES,
        default="auto",
        help="Connection mode: 'gemini-enterprise' / 'agy-cli' (included seat quota via standalone `agy` CLI), 'api-key', 'vertex', 'gateway', or 'auto'.",
    )
    parser.add_argument("--profile", choices=COST_PROFILES, default="balanced",
                        help="economy: Flash only; balanced: Flash, escalating to Argon on failure; quality: Argon first.")
    parser.add_argument("--project", default=None, help="Google Cloud project ID (required for pay-as-you-go Vertex AI).")
    parser.add_argument("--location", default=None, help="Vertex AI region (for example 'global' or 'us-central1').")
    parser.add_argument("--base-url", default=None, help="Optional AI gateway base URL (sets CC_AGY_VERTEX_BASE_URL).")
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
    parser.add_argument("--name", default=DEFAULT_SERVER_NAME, help="Codex MCP server name (default: agy).")
    parser.add_argument("--non-interactive", action="store_true", help="Never prompt, even when run with no flags in a TTY.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing anything.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    args = parser.parse_args(raw_argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("--name may only contain letters, digits, '-' and '_'")
    if args.max_daily_cents is not None:
        import math

        if not math.isfinite(args.max_daily_cents) or args.max_daily_cents <= 0:
            parser.error("--max-daily-cents must be a finite number greater than 0")

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
                agy_cli_path=args.agy_cli_path,
                max_daily_cents=args.max_daily_cents,
                allowed_projects=args.allowed_projects,
                blocked_projects=args.blocked_projects,
                no_auto_apply=args.no_auto_apply,
                strict=args.strict,
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
