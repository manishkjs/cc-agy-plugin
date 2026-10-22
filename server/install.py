#!/usr/bin/env python3
"""One-command setup of cc-agy-plugin for Codex and Claude Code.

Usage (from a clone of this repository):

    python3 -m server.install --host codex                  # register the MCP server with Codex
    python3 -m server.install --host claude                 # print/run the Claude Code plugin install
    python3 -m server.install --host all --profile quality  # both hosts, Argon-first routing

Only four settings exist: auth (a Gemini API key or a Google Cloud project, read from the environment),
`--profile` (economy | balanced | quality), `--location` (Vertex region, optional) and `--strict`
(Claude Code only, because Codex has no plugin hooks). API keys are never written to disk: Codex forwards
them from your shell through `env_vars`, and Claude Code stores them through its own `userConfig` prompt.
"""

from __future__ import annotations

import argparse
import importlib.util
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
SERVER_ENTRY = PLUGIN_ROOT / "server" / "main.py"
CODEX_SKILL_SOURCE = PLUGIN_ROOT / "plugins" / "cc-agy-plugin" / "skills" / "vertex" / "SKILL.md"
DEFAULT_SERVER_NAME = "agy"
MARKETPLACE_NAME = "cc-agy"
PLUGIN_NAME = "cc-agy-plugin"
COST_PROFILES = ("economy", "balanced", "quality")

# Forwarded from the user's shell into the Codex-launched MCP server, so secrets never land in config.toml.
CODEX_FORWARDED_ENV_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "CLOUDSDK_CONFIG",
    "CC_AGY_VERTEX_BASE_URL",
    "GOOGLE_VERTEX_BASE_URL",
    "CC_AGY_VERTEX_BASE_URL_TOKEN",
    "CC_AGY_MAX_DAILY_CENTS",
)
# Worker jobs run for minutes; Codex's default MCP tool timeout is far shorter.
CODEX_TOOL_TIMEOUT_SEC = 1800
CODEX_STARTUP_TIMEOUT_SEC = 30


def _toml_str(value: str) -> str:
    """TOML basic string (JSON string escapes are a valid subset)."""
    return json.dumps(str(value), ensure_ascii=False)


def _venv_has_antigravity(python_bin: Path) -> bool:
    venv_dir = python_bin.parent.parent
    return any(venv_dir.glob("lib/python3*/site-packages/google/antigravity/__init__.py"))


def resolve_python_bin() -> str:
    """Pick the interpreter that has google-antigravity: this one, else a plugin-adjacent virtualenv."""
    if importlib.util.find_spec("google.antigravity") is not None and sys.executable:
        return sys.executable
    candidates = [PLUGIN_ROOT / ".venv" / "bin" / "python", PLUGIN_ROOT.parent / ".venv" / "bin" / "python"]
    usable = [c for c in candidates if c.is_file() and os.access(c, os.X_OK)]
    for cand in usable:
        if _venv_has_antigravity(cand):
            return str(cand)
    return str(usable[0]) if usable else (sys.executable or "python3")


def build_codex_server_block(
    *,
    name: str = DEFAULT_SERVER_NAME,
    python_bin: str | None = None,
    profile: str = "balanced",
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
    max_daily_cents: float | None = None,
) -> str:
    """Render the `[mcp_servers.<name>]` TOML block for `$CODEX_HOME/config.toml`."""
    py = python_bin or resolve_python_bin()
    env: dict[str, str] = {"CC_AGY_COST_PROFILE": profile}
    if project:
        env["GOOGLE_CLOUD_PROJECT"] = project
    if location:
        env["GOOGLE_CLOUD_LOCATION"] = location
    if base_url:
        env["CC_AGY_VERTEX_BASE_URL"] = base_url
    if max_daily_cents is not None and max_daily_cents > 0:
        formatted = f"{max_daily_cents:g}"
        env["CC_AGY_MAX_DAILY_CENTS"] = formatted
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
    """Replace any existing `[mcp_servers.<name>]` table (and its sub-tables) with `block`, or append it.

    Every result is validated with tomllib (when available). If the existing entry is written in a form
    the line-based rewrite cannot remove safely (dotted keys or an inline table), CodexConfigError is
    raised and nothing is written.
    """
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
    project: str | None,
    location: str | None,
    base_url: str | None = None,
    max_daily_cents: float | None = None,
    dry_run: bool,
    codex_home: Path | None = None,
) -> dict[str, Any]:
    home = codex_home or Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex")).expanduser()
    cfg = home / "config.toml"
    block = build_codex_server_block(
        name=name,
        profile=profile,
        project=project,
        location=location,
        base_url=base_url,
        max_daily_cents=max_daily_cents,
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
        # The backup can hold other servers' tokens, so it is never more readable than owner-only.
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


def claude_install_commands() -> list[list[str]]:
    return [
        ["claude", "plugin", "marketplace", "add", str(PLUGIN_ROOT)],
        ["claude", "plugin", "install", f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"],
    ]


def install_claude(*, dry_run: bool) -> dict[str, Any]:
    cmds = claude_install_commands()
    result: dict[str, Any] = {
        "host": "claude",
        "commands": [" ".join(c) for c in cmds],
        "fallback": f"claude --plugin-dir {PLUGIN_ROOT}",
        "note": (
            "Claude Code asks for gemini_api_key / google_cloud_project, cost_profile and "
            "strict_delegation_mode when the plugin is enabled (userConfig)."
        ),
    }
    claude_bin = shutil.which("claude")
    if dry_run or not claude_bin:
        result["status"] = "dry_run" if dry_run else "claude_cli_not_found"
        return result
    outputs: list[dict[str, Any]] = []
    for cmd in cmds:
        proc = subprocess.run([claude_bin, *cmd[1:]], capture_output=True, text=True, check=False, timeout=120)
        outputs.append({"command": " ".join(cmd), "exit_code": proc.returncode, "output": (proc.stdout + proc.stderr).strip()[-400:]})
        if proc.returncode != 0:
            break
    result["runs"] = outputs
    result["status"] = "installed" if outputs and all(o["exit_code"] == 0 for o in outputs) else "failed"
    return result


def preflight(
    env: dict[str, str] | None = None,
    *,
    project: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Zero-network readiness: Python, SDKs, git and credentials."""
    source = env if env is not None else os.environ
    py_bin = resolve_python_bin()
    issues: list[str] = []
    if sys.version_info < (3, 10):
        issues.append(f"Python 3.10+ is required (found {sys.version.split()[0]}).")
    if py_bin == (sys.executable or "python3"):
        sdk_ok = importlib.util.find_spec("google.antigravity") is not None
    else:
        sdk_ok = subprocess.run(
            [py_bin, "-c", "import google.antigravity"], capture_output=True, check=False, timeout=60
        ).returncode == 0
    if not sdk_ok:
        issues.append(f'Antigravity SDK missing. Run: {py_bin} -m pip install -e "{PLUGIN_ROOT}[antigravity]"')
    if not shutil.which("git"):
        issues.append("git is not on PATH (worker jobs run in git copies of your repository).")
    has_key = any(str(source.get(k) or "").strip() for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    has_gateway = bool(
        base_url
        or any(str(source.get(k) or "").strip() for k in ("CC_AGY_VERTEX_BASE_URL", "GOOGLE_VERTEX_BASE_URL"))
    )
    has_project = bool(project or str(source.get("GOOGLE_CLOUD_PROJECT") or "").strip())
    if has_gateway:
        auth = "gateway"
    elif has_key:
        auth = "api_key"
    elif has_project:
        auth = "vertex"
    else:
        auth = "missing"
    if auth == "missing":
        issues.append(
            "No credentials found. Export GEMINI_API_KEY (Google AI Studio), pass --project / export "
            "GOOGLE_CLOUD_PROJECT (Vertex AI), or pass --base-url (AI gateway)."
        )
    return {"python_bin": py_bin, "auth_mode": auth, "issues": issues, "ready": not issues}


def run_interactive_wizard(
    *,
    input_fn: Any = input,
    output_fn: Any = print,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Ask 4 plain-English setup questions when cc-agy-install runs with no flags in a terminal."""
    source = env if env is not None else os.environ
    output_fn("cc-agy-plugin quick setup (press Enter for the default in brackets)\n")

    host_raw = str(input_fn("1. Coding tool [1=Both, 2=Codex, 3=Claude Code] (1): ") or "").strip().lower()
    host_map = {"": "all", "1": "all", "all": "all", "both": "all", "2": "codex", "codex": "codex", "3": "claude", "claude": "claude"}
    host = host_map.get(host_raw, "all")

    detected_proj = str(source.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    detected_gw = str(source.get("CC_AGY_VERTEX_BASE_URL") or source.get("GOOGLE_VERTEX_BASE_URL") or "").strip()
    has_key = any(str(source.get(k) or "").strip() for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    default_auth_label = "gateway" if detected_gw else ("api_key" if has_key else ("vertex" if detected_proj else "api_key"))
    auth_raw = str(
        input_fn(
            f"2. Gemini connection [1=API key in shell, 2=Google Cloud project, 3=AI gateway URL] ({default_auth_label}): "
        )
        or ""
    ).strip().lower()
    project: str | None = None
    base_url: str | None = None
    if auth_raw in ("2", "vertex", "gcp") or (not auth_raw and default_auth_label == "vertex"):
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

    parser = argparse.ArgumentParser(prog="cc-agy-install", description="Set up cc-agy-plugin for Codex and/or Claude Code.")
    parser.add_argument("--host", choices=("codex", "claude", "all"), default="all")
    parser.add_argument("--profile", choices=COST_PROFILES, default="balanced",
                        help="economy: Flash only; balanced: Flash, escalating to Argon on failure; quality: Argon first.")
    parser.add_argument("--project", default=None, help="Google Cloud project for Vertex AI (omit when using GEMINI_API_KEY).")
    parser.add_argument("--location", default=None, help="Vertex AI region (optional; global-only models are routed automatically).")
    parser.add_argument("--base-url", default=None, help="Optional AI gateway base URL (sets CC_AGY_VERTEX_BASE_URL).")
    parser.add_argument("--max-daily-cents", type=float, default=None,
                        help="Optional daily spend limit in US cents (for example 500 for $5.00/day).")
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

    report: dict[str, Any] = {
        "preflight": preflight(project=args.project, base_url=args.base_url),
        "hosts": [],
    }
    if args.host in ("codex", "all"):
        report["hosts"].append(
            install_codex(
                name=args.name,
                profile=args.profile,
                project=args.project,
                location=args.location,
                base_url=args.base_url,
                max_daily_cents=args.max_daily_cents,
                dry_run=args.dry_run,
            )
        )
    if args.host in ("claude", "all"):
        report["hosts"].append(install_claude(dry_run=args.dry_run))

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if _report_ok(report) else 1

    pre = report["preflight"]
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
            print("  Next: run `codex` in your repository and ask it to use delegate_start.")
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
