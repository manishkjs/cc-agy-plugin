"""Interactive and two-step headless Gemini Enterprise OAuth login for `agy`.

Supports three seamless sign-in paths without requiring users to navigate full-screen TUI menus:
1. Local desktop browser loopback (`agy acp` `authenticate` with `methodId="oauth-business"`):
   automatically opens the browser on local macOS/Windows/Linux desktops and saves credentials
   with zero code copy-pasting.
2. Interactive terminal PKCE (`AGY_CLI_CDE_AUTH_ACTION=login agy` attached to `/dev/tty`):
   prints a clickable Google OAuth URL directly in the terminal and reads the pasted code or
   callback URL.
3. Two-step agent/chat PKCE (`--start` / `--complete` for Claude Code, Cursor, OpenCode, or SSH
   sessions without a controlling TTY): spawns a background `agy` PKCE session holding the code
   verifier in memory over a `0600` UNIX socket, immediately returns the clickable OAuth URL so
   the host agent can show it to the user, and completes token exchange when `--complete <code>`
   is invoked.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

_PLUGIN_ROOT = str(Path(__file__).resolve().parent.parent)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from server.agy_cli import agy_cli_config_dir, inspect_agy_cli_offline, resolve_agy_cli_binary

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_OAUTH_URL_RE = re.compile(r"(https://accounts\.google\.com/\S+)")
_EMAIL_RE = re.compile(r"Successfully authenticated as\s+(\S+?)\.?\s*$", re.MULTILINE)
_DEFAULT_HOLD_TIMEOUT_SECONDS = 600
_URL_CAPTURE_TIMEOUT_SECONDS = 15.0
BAIC_FETCH_LICENSES_URL = "https://businessaicode.googleapis.com/v1beta:fetchLicenses"


def _baic_host_for_location(location: str) -> str:
    loc = (location or "global").strip().lower()
    if not loc or loc == "global":
        return "businessaicode.googleapis.com"
    return f"{loc}-businessaicode.googleapis.com"


def fetch_enterprise_licenses(
    access_token: str,
    *,
    timeout_seconds: float = 8.0,
) -> list[dict[str, str]]:
    """Query `GET https://businessaicode.googleapis.com/v1beta:fetchLicenses` with `access_token`."""
    tok = (access_token or "").strip()
    if not tok:
        return []
    req = urllib.request.Request(
        BAIC_FETCH_LICENSES_URL,
        headers={
            "Authorization": f"Bearer {tok}",
            "Accept": "application/json",
            "User-Agent": "cc-agy-plugin/0.3",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return []
    raw_items = payload.get("licenses") if isinstance(payload, Mapping) else None
    if not isinstance(raw_items, list):
        return []
    licenses: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        user_tier = str(item.get("userTier") or "").strip()
        project_id = str(item.get("projectId") or "").strip()
        if not user_tier and not project_id:
            continue
        licenses.append(
            {
                "user_tier": user_tier,
                "tier_display_name": str(item.get("tierDisplayName") or user_tier).strip(),
                "project_id": project_id,
                "region": str(item.get("location") or "global").strip() or "global",
            }
        )
    return licenses


def self_assign_enterprise_license(
    access_token: str,
    project_id: str,
    location: str = "global",
    *,
    timeout_seconds: float = 8.0,
) -> dict[str, str] | None:
    """Attempt `POST /v1beta/projects/{project}/locations/{loc}:selfAssignLicense` on BAIC."""
    tok = (access_token or "").strip()
    proj = (project_id or "").strip()
    loc = (location or "global").strip().lower() or "global"
    if not tok or not proj or "-" in loc:
        return None
    host = _baic_host_for_location(loc)
    url = f"https://{host}/v1beta/projects/{proj}/locations/{loc}:selfAssignLicense"
    body = json.dumps({"parent": f"projects/{proj}/locations/{loc}"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "cc-agy-plugin/0.3",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    lic = payload.get("license") if isinstance(payload, Mapping) else None
    if not isinstance(lic, Mapping):
        return None
    user_tier = str(lic.get("userTier") or "").strip()
    if not user_tier:
        return None
    return {
        "user_tier": user_tier,
        "tier_display_name": str(lic.get("tierDisplayName") or user_tier).strip(),
        "project_id": str(lic.get("projectId") or proj).strip(),
        "region": str(lic.get("location") or loc).strip() or "global",
    }


def _configured_enterprise_project_and_location(
    config_dir: Path,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str, str]:
    """Return `(project, location, user_tier)` from env vars or `<config_dir>/settings.json`."""
    source = os.environ if env is None else env
    project = ""
    # Only explicit plugin settings. Ambient variables such as GOOGLE_CLOUD_PROJECT or
    # ANTIGRAVITY_PROJECT_ID often name an unrelated project (or an Antigravity workspace ID),
    # and matching against them would pick the wrong license.
    for key in (
        "CC_AGY_ENTERPRISE_PROJECT",
        "CLAUDE_PLUGIN_OPTION_ENTERPRISE_PROJECT",
        "CC_AGY_USERCFG_ENTERPRISE_PROJECT",
    ):
        val = (source.get(key) or "").strip()
        if val:
            project = val
            break
    location = ""
    for key in (
        "CC_AGY_ENTERPRISE_LOCATION",
        "CLAUDE_PLUGIN_OPTION_ENTERPRISE_LOCATION",
        "CC_AGY_USERCFG_ENTERPRISE_LOCATION",
    ):
        val = (source.get(key) or "").strip()
        if val:
            location = val
            break
    user_tier = ""
    for key in (
        "CC_AGY_ENTERPRISE_USER_TIER",
        "CLAUDE_PLUGIN_OPTION_ENTERPRISE_USER_TIER",
        "CC_AGY_USERCFG_ENTERPRISE_USER_TIER",
        "BAICODE_USER_TIER",
    ):
        val = (source.get(key) or "").strip()
        if val:
            user_tier = val
            break

    settings_file = config_dir / "settings.json"
    if settings_file.is_file():
        try:
            settings_data = json.loads(settings_file.read_text(encoding="utf-8"))
            gcp_cfg = settings_data.get("gcp") if isinstance(settings_data, Mapping) else None
            if isinstance(gcp_cfg, Mapping):
                if not project:
                    project = str(gcp_cfg.get("project") or "").strip()
                if not location:
                    location = str(gcp_cfg.get("location") or "").strip()
        except (OSError, ValueError):
            pass
    return project, location, user_tier


def resolve_enterprise_license_for_token(
    stored_token: Mapping[str, Any],
    *,
    config_dir: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 8.0,
) -> dict[str, str]:
    """Resolve `user_tier`, `tier_display_name`, `project_id`, and `region` for a GCP OAuth token.

    Matches `Manager.resolveEnterpriseLicense` in `agy` (`acpauth/manager.go`):
    1. Queries `GET https://businessaicode.googleapis.com/v1beta:fetchLicenses` with the bearer token.
    2. Selects the license matching any configured project/location (or calls `selfAssignLicense` when
       a project is configured and not yet assigned, or picks `licenses[0]`).
    """
    existing_tier = str(stored_token.get("user_tier") or "").strip()
    existing_display = str(stored_token.get("tier_display_name") or existing_tier).strip()
    existing_proj = str(stored_token.get("project_id") or "").strip()
    existing_region = str(stored_token.get("region") or "").strip()
    if existing_tier and existing_proj:
        return {
            "user_tier": existing_tier,
            "tier_display_name": existing_display or existing_tier,
            "project_id": existing_proj,
            "region": existing_region or "global",
        }

    cfg_proj, cfg_loc, cfg_tier = _configured_enterprise_project_and_location(config_dir, env)
    want_proj = existing_proj or cfg_proj
    want_loc = existing_region or cfg_loc

    tok_obj = stored_token.get("token")
    access_token = str(tok_obj.get("access_token") or "").strip() if isinstance(tok_obj, Mapping) else ""
    source = os.environ if env is None else env
    lookup_disabled = (source.get("CC_AGY_LICENSE_LOOKUP") or "").strip().lower() in ("0", "false", "off", "no")
    allow_network = not lookup_disabled and (
        not bool(source.get("PYTEST_CURRENT_TEST"))
        or (source.get("CC_AGY_ALLOW_TEST_LICENSE_FETCH") or "").strip() == "1"
    )

    chosen: dict[str, str] | None = None
    if access_token and allow_network:
        licenses = fetch_enterprise_licenses(access_token, timeout_seconds=timeout_seconds)
        if want_proj and want_loc:
            for lic in licenses:
                if lic["project_id"] == want_proj and lic["region"] in (want_loc, "", "global"):
                    chosen = lic
                    break
            if chosen is None:
                chosen = self_assign_enterprise_license(
                    access_token,
                    want_proj,
                    want_loc,
                    timeout_seconds=timeout_seconds,
                )
        elif want_proj:
            for lic in licenses:
                if lic["project_id"] == want_proj:
                    chosen = lic
                    break
            if chosen is None:
                chosen = self_assign_enterprise_license(
                    access_token,
                    want_proj,
                    want_loc or "global",
                    timeout_seconds=timeout_seconds,
                )
        elif licenses:
            chosen = licenses[0]

    if chosen is not None and (chosen["user_tier"] or cfg_tier):
        return {
            "user_tier": chosen["user_tier"] or cfg_tier,
            "tier_display_name": chosen["tier_display_name"] or chosen["user_tier"] or cfg_tier,
            "project_id": chosen["project_id"] or want_proj,
            "region": chosen["region"] or want_loc or "global",
        }

    # Fallback when the lookup found nothing: only if the user configured BOTH a tier and a
    # project. A project without a tier would switch agy to pay-as-you-go Agent Platform
    # billing (`ResolveAuthMode` -> `AuthModeBusinessAgentPlatformPayGo`), so never save one.
    if cfg_tier and want_proj:
        return {
            "user_tier": cfg_tier,
            "tier_display_name": cfg_tier,
            "project_id": want_proj,
            "region": want_loc or "global",
        }
    return {"user_tier": "", "tier_display_name": "", "project_id": "", "region": ""}


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from `text`."""
    return _ANSI_ESCAPE_RE.sub("", text)


def extract_oauth_code(raw_value: str) -> str:
    """Extract an OAuth authorization code from a pasted code string (or URL if `?code=` is still present)."""
    cleaned = (raw_value or "").strip()
    if not cleaned:
        raise ValueError("Authorization code cannot be empty.")
    if cleaned.startswith(("http://", "https://")) or ("code=" in cleaned and ("?" in cleaned or "&" in cleaned)):
        query_str = urllib.parse.urlsplit(cleaned).query if "://" in cleaned else cleaned.lstrip("?")
        params = urllib.parse.parse_qs(query_str)
        codes = params.get("code")
        if codes and codes[0].strip():
            return codes[0].strip()
        raise ValueError(
            "The browser address bar strips `?code=` after loading. Click the 'Copy to Clipboard' "
            "button inside the Antigravity callback page and paste that 4/... code instead."
        )
    return cleaned


def _home_dir(env: Mapping[str, str]) -> Path:
    raw_home = (env.get("HOME") or "").strip()
    return Path(raw_home).expanduser() if raw_home else Path.home()


def _login_state_dir(env: Mapping[str, str]) -> Path:
    """Return a short `0700` per-user directory for the UNIX domain socket (<= 100 bytes for `AF_UNIX`)."""
    home_key = hashlib.sha256(str(_home_dir(env)).encode("utf-8")).hexdigest()[:12]
    runtime_base = Path(tempfile.gettempdir()) / f"cc-agy-login-{os.getuid()}-{home_key}"
    runtime_base.mkdir(parents=True, exist_ok=True)
    try:
        runtime_base.chmod(0o700)
    except OSError:
        pass
    return runtime_base


def _atomic_write_private(target: Path, payload_bytes: bytes) -> None:
    """Atomically write `payload_bytes` to `target` with `0600` permissions."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload_bytes)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def finalize_enterprise_onboarding(
    config_dir: Path,
    home_dir: Path,
    *,
    had_standalone_token_before: bool = False,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Sync standalone CDE token into `agy`'s CLI token file, resolve BAIC license/project, and update `onboarding.json`.

    When `AGY_CLI_CDE_AUTH_ACTION=login` runs on a headless host without an OS keyring, it writes
    `~/.gemini/jetski-standalone-oauth-token` with empty `user_tier` and `project_id`, whereas
    `agy` CLI (`keyringAuth.TryAuth` in `keyring.go`) reads `<config_dir>/antigravity-oauth-token`
    and requires at least one of `user_tier` or `project_id` to be non-empty for `auth_method="gcp"`.
    """
    standalone_token_file = home_dir / ".gemini" / "jetski-standalone-oauth-token"
    cli_token_file = config_dir / "antigravity-oauth-token"
    bak_token_file = config_dir / "antigravity-oauth-token.bak"
    prev_auth_method = "gcp"

    if not cli_token_file.is_file() and bak_token_file.is_file() and bak_token_file.stat().st_size > 10:
        _atomic_write_private(cli_token_file, bak_token_file.read_bytes())
        bak_token_file.unlink(missing_ok=True)

    parsed_token: dict[str, Any] | None = None
    if standalone_token_file.is_file() and standalone_token_file.stat().st_size > 10:
        raw_bytes = standalone_token_file.read_bytes()
        try:
            loaded_tok = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(loaded_tok, dict):
                parsed_token = loaded_tok
                if parsed_token.get("wif_provider"):
                    prev_auth_method = "wif"
        except (UnicodeDecodeError, ValueError):
            parsed_token = None
        if parsed_token is None:
            _atomic_write_private(cli_token_file, raw_bytes)
        if not had_standalone_token_before:
            standalone_token_file.unlink(missing_ok=True)
    elif cli_token_file.is_file() and cli_token_file.stat().st_size > 10:
        try:
            loaded_tok = json.loads(cli_token_file.read_text(encoding="utf-8"))
            if isinstance(loaded_tok, dict):
                parsed_token = loaded_tok
                if parsed_token.get("wif_provider"):
                    prev_auth_method = "wif"
        except (OSError, ValueError):
            parsed_token = None

    resolved_license = {"user_tier": "", "tier_display_name": "", "project_id": "", "region": ""}
    if isinstance(parsed_token, dict):
        resolved_license = resolve_enterprise_license_for_token(
            parsed_token,
            config_dir=config_dir,
            env=env,
        )
        if resolved_license["user_tier"]:
            parsed_token["user_tier"] = resolved_license["user_tier"]
        if resolved_license["tier_display_name"]:
            parsed_token["tier_display_name"] = resolved_license["tier_display_name"]
        if resolved_license["project_id"]:
            parsed_token["project_id"] = resolved_license["project_id"]
        if resolved_license["region"]:
            parsed_token["region"] = resolved_license["region"]
        _atomic_write_private(
            cli_token_file,
            json.dumps(parsed_token).encode("utf-8"),
        )

    onboarding_file = config_dir / "cache" / "onboarding.json"
    existing_onboarding: dict[str, Any] = {}
    if onboarding_file.is_file():
        try:
            loaded = json.loads(onboarding_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_onboarding = loaded
        except (OSError, ValueError):
            existing_onboarding = {}

    effective_tier = (
        resolved_license["user_tier"]
        or str(existing_onboarding.get("tierId") or "").strip()
    )
    effective_display = (
        resolved_license["tier_display_name"]
        or str(existing_onboarding.get("tierName") or "").strip()
    )
    effective_proj = (
        resolved_license["project_id"]
        or str(existing_onboarding.get("cloudaicompanionProject") or "").strip()
    )
    effective_region = resolved_license["region"] or "global"

    updated_onboarding: dict[str, Any] = {
        **existing_onboarding,
        "consumerOnboardingComplete": False,
        "enterpriseOnboardingComplete": True,
        "onboardingComplete": bool(existing_onboarding.get("onboardingComplete", False)),
        "previousAuthMethod": prev_auth_method,
    }
    if effective_proj:
        updated_onboarding["cloudaicompanionProject"] = effective_proj
    if effective_tier:
        updated_onboarding["tierId"] = effective_tier
    if effective_display:
        updated_onboarding["tierName"] = effective_display
    _atomic_write_private(
        onboarding_file,
        (json.dumps(updated_onboarding, indent=2) + "\n").encode("utf-8"),
    )

    if effective_proj:
        settings_file = config_dir / "settings.json"
        existing_settings: dict[str, Any] = {}
        if settings_file.is_file():
            try:
                loaded_settings = json.loads(settings_file.read_text(encoding="utf-8"))
                if isinstance(loaded_settings, dict):
                    existing_settings = loaded_settings
            except (OSError, ValueError):
                existing_settings = {}
        existing_gcp = existing_settings.get("gcp")
        gcp_dict = dict(existing_gcp) if isinstance(existing_gcp, Mapping) else {}
        gcp_dict["project"] = effective_proj
        gcp_dict["location"] = gcp_dict.get("location") or effective_region
        existing_settings["gcp"] = gcp_dict
        _atomic_write_private(
            settings_file,
            (json.dumps(existing_settings, indent=2) + "\n").encode("utf-8"),
        )

    return {
        "config_dir": str(config_dir),
        "previous_auth_method": prev_auth_method,
        "user_tier": effective_tier or "gcp-enterprise",
        "tier_display_name": effective_display or effective_tier or "Gemini Enterprise",
        "project_id": effective_proj or None,
        "region": effective_region if effective_proj else None,
    }


def can_use_local_browser_loopback(env: Mapping[str, str]) -> bool:
    """Return True when this machine has a local GUI browser and is not an SSH session."""
    if any((env.get(k) or "").strip() for k in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return False
    if (env.get("CC_AGY_FORCE_URL_LOGIN") or "").strip().lower() in ("1", "true", "yes"):
        return False
    if sys.platform == "darwin":
        return shutil.which("open") is not None
    if sys.platform == "win32":
        return True
    if sys.platform.startswith("linux"):
        has_display = bool((env.get("DISPLAY") or "").strip() or (env.get("WAYLAND_DISPLAY") or "").strip())
        return has_display and shutil.which("xdg-open") is not None
    return False


def try_acp_business_browser_login(
    agy_bin: str,
    *,
    env: Mapping[str, str],
    timeout_seconds: float = 110.0,
) -> tuple[bool, str]:
    """Attempt zero-copy-paste browser loopback login via `agy acp` `authenticate(oauth-business)`."""
    proc = subprocess.Popen(
        [agy_bin, "acp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env),
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientInfo": {"name": "cc-agy-plugin", "version": "0.3.0"},
            },
        }
        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()
        init_line = proc.stdout.readline()
        if not init_line:
            return False, "agy acp closed before initialize completed"

        auth_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "authenticate",
            "params": {"methodId": "oauth-business"},
        }
        proc.stdin.write(json.dumps(auth_req) + "\n")
        proc.stdin.flush()

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                return False, "agy acp exited during browser sign-in"
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("id") != 2:
                continue
            if "error" in msg and msg["error"]:
                err_obj = msg["error"]
                err_msg = err_obj.get("message") if isinstance(err_obj, Mapping) else str(err_obj)
                return False, str(err_msg or "agy acp authenticate failed")
            finalize_enterprise_onboarding(
                agy_cli_config_dir(env),
                _home_dir(env),
                had_standalone_token_before=True,
                env=env,
            )
            return True, ""
        return False, "Timed out waiting for browser sign-in"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _spawn_cde_login_and_capture_url(
    agy_bin: str,
    *,
    env: Mapping[str, str],
    timeout_seconds: float = _URL_CAPTURE_TIMEOUT_SECONDS,
) -> tuple[subprocess.Popen[str], str]:
    """Start `AGY_CLI_CDE_AUTH_ACTION=login agy` and read stdout until the Google OAuth URL appears."""
    child_env = {**env, "AGY_CLI_CDE_AUTH_ACTION": "login"}
    child_env.pop("AGY_CDE_ENTERPRISE_AUTH", None)
    proc = subprocess.Popen(
        [agy_bin],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )
    assert proc.stdout is not None
    collected: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
            continue
        clean = strip_ansi(line)
        collected.append(clean)
        match = _OAUTH_URL_RE.search(clean)
        if match:
            return proc, match.group(1).strip()
    try:
        proc.kill()
    except OSError:
        pass
    joined = "".join(collected).strip()
    raise RuntimeError(f"Could not obtain OAuth sign-in URL from `{agy_bin}`: {joined or 'no output'}")


def _exchange_code_with_running_cde_proc(
    proc: subprocess.Popen[str],
    raw_code_or_url: str,
    *,
    config_dir: Path,
    home_dir: Path,
    had_standalone_before: bool,
    timeout_seconds: float = 30.0,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Feed the extracted authorization code to the waiting `agy` login process and finalize credentials."""
    code = extract_oauth_code(raw_code_or_url)
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(code + "\n")
    proc.stdin.flush()
    # Do not call proc.stdin.close() before proc.communicate(): on Python 3.10/3.11,
    # Popen.communicate() calls self.stdin.flush() before closing stdin and raises
    # ValueError("I/O operation on closed file") if stdin was already closed.
    remaining_out, _ = proc.communicate(timeout=timeout_seconds)
    clean_out = strip_ansi(remaining_out or "")
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": clean_out.strip() or f"agy login exited with status {proc.returncode}",
        }
    onboarding = finalize_enterprise_onboarding(
        config_dir,
        home_dir,
        had_standalone_token_before=had_standalone_before,
        env=env,
    )
    email_match = _EMAIL_RE.search(clean_out)
    email = email_match.group(1) if email_match else None
    result: dict[str, Any] = {
        "ok": True,
        "email": email,
        "tier_id": onboarding.get("user_tier") or "gcp-enterprise",
        "tier_display_name": onboarding.get("tier_display_name"),
        "project_id": onboarding.get("project_id"),
        "message": clean_out.strip(),
    }
    if not onboarding.get("project_id"):
        result["warning"] = (
            "Signed in, but no Gemini Enterprise license was found for this account. "
            "Set CC_AGY_ENTERPRISE_PROJECT=<project with your seats> and run "
            "`/cc-agy-plugin:login --force` again."
        )
    return result


def _cleanup_stale_hold_session(state_dir: Path) -> None:
    session_file = state_dir / "session.json"
    sock_file = state_dir / "submit.sock"
    if session_file.is_file():
        try:
            meta = json.loads(session_file.read_text(encoding="utf-8"))
            old_pid = int(meta.get("pid") or 0)
            if old_pid > 1 and old_pid != os.getpid():
                os.kill(old_pid, signal.SIGTERM)
                time.sleep(0.1)
        except (OSError, ValueError):
            pass
    session_file.unlink(missing_ok=True)
    sock_file.unlink(missing_ok=True)


def run_hold_daemon(state_dir: Path, *, timeout_seconds: int = _DEFAULT_HOLD_TIMEOUT_SECONDS) -> int:
    """Background PKCE holder: starts `agy` login, writes `session.json`, and waits for `--complete`."""
    env = dict(os.environ)
    agy_bin, _ = resolve_agy_cli_binary(env)
    if not agy_bin:
        _atomic_write_private(
            state_dir / "session.json",
            json.dumps({"ok": False, "error": "External `agy` binary is not installed."}).encode("utf-8"),
        )
        return 1

    home_dir = _home_dir(env)
    config_dir = agy_cli_config_dir(env)
    had_standalone = (home_dir / ".gemini" / "jetski-standalone-oauth-token").is_file()
    sock_path = state_dir / "submit.sock"
    sock_path.unlink(missing_ok=True)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    try:
        sock_path.chmod(0o600)
    except OSError:
        pass
    srv.listen(2)
    srv.settimeout(float(timeout_seconds))

    try:
        proc, auth_url = _spawn_cde_login_and_capture_url(agy_bin, env=env)
    except Exception as exc:
        srv.close()
        sock_path.unlink(missing_ok=True)
        _atomic_write_private(
            state_dir / "session.json",
            json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"),
        )
        return 1

    _atomic_write_private(
        state_dir / "session.json",
        json.dumps(
            {
                "ok": True,
                "pid": os.getpid(),
                "auth_url": auth_url,
                "socket": str(sock_path),
                "created_at": time.time(),
                "expires_at": time.time() + timeout_seconds,
            },
            indent=2,
        ).encode("utf-8"),
    )

    try:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            remaining = max(1.0, deadline - time.monotonic())
            srv.settimeout(remaining)
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                break
            with conn:
                raw_req = conn.recv(8192).decode("utf-8", errors="replace").strip()
                try:
                    extract_oauth_code(raw_req)
                except ValueError as exc:
                    conn.sendall((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode("utf-8"))
                    continue
                try:
                    result = _exchange_code_with_running_cde_proc(
                        proc,
                        raw_req,
                        config_dir=config_dir,
                        home_dir=home_dir,
                        had_standalone_before=had_standalone,
                        env=env,
                    )
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                conn.sendall((json.dumps(result) + "\n").encode("utf-8"))
                return 0 if result.get("ok") else 1
        return 1
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except OSError:
            pass
        srv.close()
        sock_path.unlink(missing_ok=True)
        (state_dir / "session.json").unlink(missing_ok=True)


def heal_enterprise_token(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Repair a saved `auth_method="gcp"` token that `agy`'s `keyringAuth` would reject.

    `keyringAuth.TryAuth` treats a GCP token with empty `user_tier` AND empty `project_id` as
    "incomplete GCP onboarding" and signs the user out. If the token's access token can still
    reach `businessaicode.googleapis.com/v1beta:fetchLicenses`, this fills in the license so no
    new browser sign-in is needed.

    Returns:
        A dict with `healed` (bool), `already_complete` (bool), and the resolved fields.
    """
    source = dict(os.environ if env is None else env)
    config_dir = agy_cli_config_dir(source)
    token_file = config_dir / "antigravity-oauth-token"
    bak_file = config_dir / "antigravity-oauth-token.bak"
    if not token_file.is_file() and bak_file.is_file() and bak_file.stat().st_size > 10:
        _atomic_write_private(token_file, bak_file.read_bytes())
        bak_file.unlink(missing_ok=True)
    try:
        stored = json.loads(token_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"healed": False, "already_complete": False, "reason": "no readable token file"}
    if not isinstance(stored, dict) or stored.get("auth_method") != "gcp":
        return {"healed": False, "already_complete": False, "reason": "not a Gemini Enterprise token"}
    if stored.get("user_tier") or stored.get("project_id"):
        return {
            "healed": False,
            "already_complete": True,
            "user_tier": stored.get("user_tier"),
            "project_id": stored.get("project_id"),
        }
    onboarding = finalize_enterprise_onboarding(
        config_dir,
        _home_dir(source),
        had_standalone_token_before=True,
        env=source,
    )
    healed = bool(onboarding.get("project_id"))
    reason = None
    if not healed:
        reason = "fetchLicenses returned no license for this account (no seat, or the saved token expired)"
        tok_obj = stored.get("token")
        expiry_raw = str(tok_obj.get("expiry") or "") if isinstance(tok_obj, Mapping) else ""
        try:
            expiry = datetime.fromisoformat(expiry_raw.replace("Z", "+00:00"))
            if expiry.tzinfo is not None and expiry < datetime.now(timezone.utc):
                reason = (
                    "the saved access token expired, so the license can't be looked up offline; "
                    "run `/cc-agy-plugin:login` to sign in again"
                )
        except ValueError:
            pass
    return {
        "healed": healed,
        "already_complete": False,
        "user_tier": onboarding.get("user_tier") if healed else None,
        "project_id": onboarding.get("project_id"),
        "reason": reason,
    }


def start_two_step_login(
    env: Mapping[str, str] | None = None,
    *,
    timeout_seconds: int = _DEFAULT_HOLD_TIMEOUT_SECONDS,
    force: bool = False,
) -> dict[str, Any]:
    """Launch the detached PKCE holder daemon and return the clickable Google OAuth URL.

    Args:
        env: Environment to use (defaults to `os.environ`).
        timeout_seconds: How long the holder waits for the pasted code.
        force: Skip the "already signed in" short-circuit and always start a fresh sign-in.
    """
    source = dict(os.environ if env is None else env)
    if not force:
        heal_enterprise_token(source)
        status = inspect_agy_cli_offline(source)
        if status.get("auth_configured") and not status.get("consumer_account_detected"):
            return {
                "ok": True,
                "already_signed_in": True,
                "tier_id": status.get("tier_id"),
                "credential_source": status.get("credential_source"),
            }
    else:
        status = inspect_agy_cli_offline(source)
    if not status.get("agy_bin"):
        return {"ok": False, "error": status.get("error") or "External `agy` binary is not installed."}

    state_dir = _login_state_dir(source)
    _cleanup_stale_hold_session(state_dir)
    session_file = state_dir / "session.json"
    daemon_log = state_dir / "daemon.log"
    _atomic_write_private(daemon_log, b"")

    with (
        open(os.devnull, "rb") as devnull_in,
        open(os.devnull, "wb") as devnull_out,
        open(daemon_log, "ab") as log_err,
    ):
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "server.agy_login",
                "--hold-daemon",
                str(state_dir),
                "--timeout",
                str(timeout_seconds),
            ],
            cwd=str(Path(__file__).resolve().parent.parent),
            env=source,
            stdin=devnull_in,
            stdout=devnull_out,
            stderr=log_err,
            start_new_session=True,
            close_fds=True,
        )

    deadline = time.monotonic() + _URL_CAPTURE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if session_file.is_file():
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and ("auth_url" in data or "error" in data):
                    return data
            except (OSError, ValueError):
                pass
        time.sleep(0.05)
    return {"ok": False, "error": "Timed out waiting for background `agy` OAuth URL generation."}


def complete_two_step_login(
    raw_code_or_url: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Send the user's pasted authorization code (or callback URL) to the waiting PKCE holder."""
    source = dict(os.environ if env is None else env)
    try:
        clean_code = extract_oauth_code(raw_code_or_url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    state_dir = _login_state_dir(source)
    sock_path = state_dir / "submit.sock"
    if not sock_path.exists():
        return {
            "ok": False,
            "error": (
                "No active Gemini Enterprise sign-in session is waiting (it may have expired or already completed). "
                "Run `python3 -m server.agy_login --start` to generate a fresh sign-in link."
            ),
        }

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(35.0)
            client.connect(str(sock_path))
            client.sendall((clean_code + "\n").encode("utf-8"))
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(8192)
                if not chunk:
                    break
                chunks.append(chunk)
    except OSError as exc:
        return {
            "ok": False,
            "error": (
                f"Sign-in helper socket closed ({type(exc).__name__}: {exc}). "
                "Run `python3 -m server.agy_login --start` to generate a fresh sign-in link."
            ),
        }
    raw_reply = b"".join(chunks).decode("utf-8", errors="replace").strip()
    try:
        parsed = json.loads(raw_reply)
        if isinstance(parsed, dict):
            return parsed
    except ValueError:
        pass
    log_tail = ""
    daemon_log = state_dir / "daemon.log"
    if daemon_log.is_file():
        try:
            log_tail = daemon_log.read_text(encoding="utf-8", errors="replace").strip()[-400:]
        except OSError:
            log_tail = ""
    detail = raw_reply or log_tail or "Empty response from login helper."
    return {"ok": False, "error": detail}


def _open_controlling_tty() -> tuple[Any, Any] | None:
    if sys.platform == "win32":
        return None
    try:
        tty_in = open("/dev/tty", "r", encoding="utf-8")
        tty_out = open("/dev/tty", "w", encoding="utf-8", buffering=1)
        return tty_in, tty_out
    except OSError:
        return None


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Stop at the first 301/302 redirect so we can verify `Location == auth_url`."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def shorten_and_verify_oauth_url(auth_url: str, env: Mapping[str, str] | None = None) -> str | None:
    """Create a ~20-char single-line HTTPS redirect URL and verify its `Location` matches `auth_url` verbatim.

    Long 500+ char OAuth URLs wrap across 4 indented lines inside TUI hosts (such as Claude Code in
    macOS `Terminal.app`), breaking both `Cmd+Click` URL detection and mouse selection. A verified
    20-char URL fits on one line on every terminal width. Disable via `CC_AGY_SHORT_URL=0`.
    """
    source = os.environ if env is None else env
    if (source.get("CC_AGY_SHORT_URL") or "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    if source.get("PYTEST_CURRENT_TEST"):
        return None

    endpoints = (
        "https://is.gd/create.php?" + urllib.parse.urlencode({"format": "simple", "url": auth_url}),
        "https://tinyurl.com/api-create.php?" + urllib.parse.urlencode({"url": auth_url}),
    )
    no_redirect_opener = urllib.request.build_opener(_NoRedirectHandler)
    for api_url in endpoints:
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "cc-agy-plugin/0.3"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                candidate = resp.read(256).decode("utf-8", errors="replace").strip()
            if not candidate.startswith("https://") or len(candidate) > 48 or any(c.isspace() for c in candidate):
                continue
            location = ""
            try:
                check_req = urllib.request.Request(candidate, headers={"User-Agent": "cc-agy-plugin/0.3"})
                no_redirect_opener.open(check_req, timeout=2.5)
            except urllib.error.HTTPError as redirect_exc:
                if redirect_exc.code in (301, 302, 303, 307, 308):
                    location = str(redirect_exc.headers.get("Location") or "")
            if location == auth_url:
                return candidate
        except Exception:
            continue
    return None


def copy_url_to_terminal_clipboard_osc52(url: str) -> bool:
    """Emit an OSC 52 clipboard sequence to `/dev/tty` or an ancestor PTY (`/proc/<pid>/fd/1`).

    When run inside an SSH session or a TUI agent like Claude Code whose own stdout is a pipe,
    walking same-UID ancestor `/proc/<pid>/fd/{1,2}` finds the underlying SSH PTY (`/dev/pts/N`)
    so the full OAuth URL is placed straight into the user's laptop clipboard (`Cmd+V`).
    """
    if not url or sys.platform == "win32":
        return False
    encoded = base64.b64encode(url.encode("utf-8")).decode("ascii")
    osc52_bytes = f"\x1b]52;c;{encoded}\x07".encode("ascii")

    try:
        with open("/dev/tty", "wb", buffering=0) as tty_handle:
            tty_handle.write(osc52_bytes)
            return True
    except OSError:
        pass

    if not Path("/proc").is_dir():
        return False
    my_uid = os.getuid()
    pid = os.getppid()
    visited: set[int] = set()
    for _depth in range(8):
        if pid <= 1 or pid in visited:
            break
        visited.add(pid)
        proc_dir = Path(f"/proc/{pid}")
        try:
            if proc_dir.stat().st_uid != my_uid:
                break
            for fd_num in (1, 2, 0):
                fd_link = proc_dir / "fd" / str(fd_num)
                target = os.readlink(fd_link)
                if target.startswith(("/dev/pts/", "/dev/tty")):
                    with open(target, "wb", buffering=0) as pty_handle:
                        pty_handle.write(osc52_bytes)
                        return True
            stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
            rparen = stat_text.rfind(")")
            pid = int(stat_text[rparen + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return False


def run_interactive_or_url_login(
    env: Mapping[str, str] | None = None,
    *,
    browser_timeout_seconds: float = 110.0,
    force: bool = False,
) -> int:
    """Automatic onboarding entrypoint used by `quickstart.sh` and `/cc-agy-plugin:login`."""
    source = dict(os.environ if env is None else env)
    if (source.get("CC_AGY_SKIP_SIGNIN") or "").strip() == "1" or source.get("PYTEST_CURRENT_TEST"):
        return 0

    if not force:
        healed = heal_enterprise_token(source)
        if healed.get("healed"):
            print(f"Repaired saved Gemini Enterprise sign-in (project={healed.get('project_id')}).")
    status = inspect_agy_cli_offline(source)
    if not force and status.get("auth_configured") and not status.get("consumer_account_detected"):
        print(f"Gemini Enterprise sign-in verified ({status.get('tier_id') or 'enterprise'}).")
        return 0

    agy_bin = status.get("agy_bin")
    if not agy_bin:
        print("Antigravity CLI (`agy`) is not installed yet; skipping sign-in.", file=sys.stderr)
        return 0

    # 1. Local desktop with GUI browser (not SSH): zero-copy-paste browser loopback!
    if can_use_local_browser_loopback(source):
        print("Opening your browser for one-click Gemini Enterprise sign-in...", flush=True)
        ok, _reason = try_acp_business_browser_login(
            agy_bin,
            env=source,
            timeout_seconds=browser_timeout_seconds,
        )
        if ok:
            print("Signed in to Gemini Enterprise (tier=gcp-enterprise).")
            return 0

    # 2. Interactive terminal (`/dev/tty` available, e.g. `curl ... | bash` in an SSH shell):
    tty_pair = _open_controlling_tty()
    if tty_pair is not None:
        tty_in, tty_out = tty_pair
        try:
            home_dir = _home_dir(source)
            config_dir = agy_cli_config_dir(source)
            had_standalone = (home_dir / ".gemini" / "jetski-standalone-oauth-token").is_file()
            proc, auth_url = _spawn_cde_login_and_capture_url(agy_bin, env=source)
            short_url = shorten_and_verify_oauth_url(auth_url, source)
            copied_osc52 = copy_url_to_terminal_clipboard_osc52(auth_url)
            primary_link = short_url or auth_url
            clip_note = " (also copied to your clipboard — press Cmd+V in your browser!)" if copied_osc52 else ""
            tty_out.write(
                "\n"
                "  Gemini Enterprise Sign-In\n"
                f"  1. Open this link in your browser{clip_note}:\n\n"
                f"     {primary_link}\n\n"
                "  2. Click 'Copy to Clipboard' on the Antigravity page and paste the code below:\n"
                "  Authorization code: "
            )
            tty_out.flush()
            pasted = tty_in.readline()
            if not pasted:
                proc.kill()
                tty_out.write("\nSign-in skipped (no code entered).\n")
                return 0
            result = _exchange_code_with_running_cde_proc(
                proc,
                pasted,
                config_dir=config_dir,
                home_dir=home_dir,
                had_standalone_before=had_standalone,
                env=source,
            )
            if result.get("ok"):
                who = f" as {result['email']}" if result.get("email") else ""
                tty_out.write(
                    f"\nSigned in to Gemini Enterprise{who} "
                    f"(tier={result.get('tier_id')}, project={result.get('project_id') or 'unknown'})!\n\n"
                )
                if result.get("warning"):
                    tty_out.write(f"Warning: {result['warning']}\n")
                return 0
            tty_out.write(f"\nSign-in failed: {result.get('error')}\n")
            return 1
        finally:
            tty_in.close()
            tty_out.close()

    # 3. Non-interactive host (e.g. Claude Code / Cursor over SSH with no `/dev/tty`):
    #    Start the 2-step PKCE session immediately and print the clickable OAuth URL!
    started = start_two_step_login(source, force=force)
    if not started.get("ok"):
        print(f"Could not start Gemini Enterprise sign-in: {started.get('error')}", file=sys.stderr)
        return 1
    if started.get("already_signed_in"):
        print(f"Gemini Enterprise sign-in verified ({started.get('tier_id') or 'enterprise'}).")
        return 0

    plugin_root = Path(__file__).resolve().parent.parent
    complete_cmd = f'{sys.executable} "{plugin_root}/server/agy_login.py" --complete "<PASTE_CODE>"'
    auth_url = started["auth_url"]
    short_url = shorten_and_verify_oauth_url(auth_url, source)
    copied_osc52 = copy_url_to_terminal_clipboard_osc52(auth_url)
    short_line = f"Short 1-line URL (never wraps): {short_url}\n" if short_url else ""
    osc52_line = "Clipboard: Full sign-in URL was ALREADY copied to the user's clipboard via OSC 52 (they can press Cmd+V / Ctrl+V in their browser!).\n" if copied_osc52 else ""
    print(
        "\n"
        "========================================================================\n"
        "GEMINI ENTERPRISE SIGN-IN READY:\n\n"
        f"{short_line}"
        f"Markdown link: [Sign in with Google]({auth_url})\n"
        f"{osc52_line}\n"
        "AGENT / CLAUDE CODE — DISPLAY THIS EXACTLY SO THE LINK IS EASY TO OPEN:\n"
        "  1. Show the short 1-line URL on its own line (so it never wraps in macOS\n"
        "     Terminal.app: tell the user they can press Cmd+V in their browser right\n"
        "     now because it's already in their clipboard, drag-select the short URL\n"
        "     to copy it, or hold Fn+Cmd+Double-Click in Terminal.app / Cmd+Click in\n"
        "     iTerm2/Ghostty/VS Code on `[Sign in with Google](...)`).\n"
        "  2. Ask them to sign in with their Gemini Enterprise account, click\n"
        "     'Copy to Clipboard' on the Antigravity page, and paste the code here.\n"
        "  3. As soon as they paste the code, run:\n"
        f"       {complete_cmd}\n"
        "========================================================================\n",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `python3 -m server.agy_login` and `python3 server/agy_login.py`."""
    parser = argparse.ArgumentParser(description="Interactive Gemini Enterprise OAuth helper for cc-agy-plugin.")
    parser.add_argument("--start", action="store_true", help="Start a 2-step PKCE login and print JSON/text with URL.")
    parser.add_argument("--complete", metavar="CODE_OR_URL", help="Complete a waiting 2-step PKCE login with the code.")
    parser.add_argument("--json", action="store_true", help="Format `--start` or `--complete` output as JSON.")
    parser.add_argument("--hold-daemon", metavar="STATE_DIR", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=_DEFAULT_HOLD_TIMEOUT_SECONDS, help="Session timeout in seconds.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any saved sign-in and always start a fresh Gemini Enterprise sign-in.",
    )
    parser.add_argument(
        "--heal",
        action="store_true",
        help="Repair a saved token that agy rejects (missing license/project) without a new sign-in.",
    )
    args = parser.parse_args(argv)

    if args.hold_daemon:
        return run_hold_daemon(Path(args.hold_daemon), timeout_seconds=args.timeout)

    if args.heal:
        res = heal_enterprise_token()
        if args.json:
            print(json.dumps(res, indent=2))
        elif res.get("healed"):
            print(f"[OK] Repaired saved sign-in (tier={res.get('user_tier')}, project={res.get('project_id')}).")
        elif res.get("already_complete"):
            print(f"[OK] Saved sign-in is already complete (project={res.get('project_id')}).")
        else:
            print(f"[FAIL] {res.get('reason')}", file=sys.stderr)
        return 0 if (res.get("healed") or res.get("already_complete")) else 1

    if args.complete is not None:
        res = complete_two_step_login(args.complete)
        if args.json:
            print(json.dumps(res, indent=2))
        elif res.get("ok"):
            who = f" as {res['email']}" if res.get("email") else ""
            print(
                f"[OK] Signed in to Gemini Enterprise{who} "
                f"(tier={res.get('tier_id', 'gcp-enterprise')}, project={res.get('project_id') or 'unknown'})."
            )
            if res.get("warning"):
                print(f"[WARN] {res['warning']}", file=sys.stderr)
        else:
            print(f"[FAIL] {res.get('error')}", file=sys.stderr)
        return 0 if res.get("ok") else 1

    if args.start:
        res = start_two_step_login(timeout_seconds=args.timeout, force=args.force)
        if res.get("ok") and res.get("auth_url"):
            res["short_url"] = shorten_and_verify_oauth_url(res["auth_url"])
            res["copied_to_clipboard"] = copy_url_to_terminal_clipboard_osc52(res["auth_url"])
        if args.json:
            print(json.dumps(res, indent=2))
        elif res.get("already_signed_in"):
            print(f"[OK] Already signed in to Gemini Enterprise (tier={res.get('tier_id')}).")
        elif res.get("ok"):
            if res.get("short_url"):
                print(res["short_url"])
            print(res["auth_url"])
        else:
            print(f"[FAIL] {res.get('error')}", file=sys.stderr)
        return 0 if res.get("ok") else 1

    return run_interactive_or_url_login(force=args.force)


if __name__ == "__main__":
    # Support direct `python3 /path/to/repo/server/agy_login.py` invocation as well as `-m server.agy_login`
    sys.exit(main())
