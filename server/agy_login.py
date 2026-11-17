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
import shlex
import shutil
import signal
import socket
import stat
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

from server.agy_cli import (
    _platform_keeps_token_in_keychain,
    agy_cli_config_dir,
    inspect_agy_cli_offline,
    resolve_agy_cli_binary,
)

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


class LicenseLookupError(RuntimeError):
    """The license service could not be asked (network down, token rejected, bad reply)."""


def fetch_enterprise_licenses(
    access_token: str,
    *,
    timeout_seconds: float = 8.0,
) -> list[dict[str, str]]:
    """Query `GET https://businessaicode.googleapis.com/v1beta:fetchLicenses` with `access_token`.

    Returns:
        The licenses on the account; an empty list means the service answered "no license".

    Raises:
        LicenseLookupError: The question could not be answered (network, HTTP error, bad JSON),
            which callers must not report as "no license".
    """
    tok = (access_token or "").strip()
    if not tok:
        raise LicenseLookupError("no access token to look up the license with")
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
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise LicenseLookupError(
                f"the license service rejected the saved token (HTTP {exc.code}); it has probably expired"
            ) from exc
        raise LicenseLookupError(f"the license service returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise LicenseLookupError(f"couldn't reach the license service ({exc})") from exc
    except ValueError as exc:
        raise LicenseLookupError("the license service sent an unreadable reply") from exc
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


def _first_env(source: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        val = (source.get(key) or "").strip()
        if val:
            return val
    return ""


def _configured_enterprise_project_and_location(
    config_dir: Path,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str, str, str]:
    """Return `(explicit_project, settings_project, location, user_tier)`.

    `explicit_project` comes only from plugin settings the user set (`CC_AGY_ENTERPRISE_PROJECT`
    and its Claude/OpenCode mirrors). Ambient variables such as GOOGLE_CLOUD_PROJECT or
    ANTIGRAVITY_PROJECT_ID are ignored: they often name an unrelated project, or an Antigravity
    workspace ID. `settings_project` is agy's `gcp.project`, which this plugin also writes, so it
    is a preference, not an order.
    """
    source = os.environ if env is None else env
    explicit = _first_env(
        source,
        "CC_AGY_ENTERPRISE_PROJECT",
        "CLAUDE_PLUGIN_OPTION_ENTERPRISE_PROJECT",
        "CC_AGY_USERCFG_ENTERPRISE_PROJECT",
    )
    location = _first_env(
        source,
        "CC_AGY_ENTERPRISE_LOCATION",
        "CLAUDE_PLUGIN_OPTION_ENTERPRISE_LOCATION",
        "CC_AGY_USERCFG_ENTERPRISE_LOCATION",
    )
    user_tier = _first_env(
        source,
        "CC_AGY_ENTERPRISE_USER_TIER",
        "CLAUDE_PLUGIN_OPTION_ENTERPRISE_USER_TIER",
        "CC_AGY_USERCFG_ENTERPRISE_USER_TIER",
        "BAICODE_USER_TIER",
    )
    settings_project = _first_env(source, "CC_AGY_HOST_ENTERPRISE_PROJECT")
    settings_file = config_dir / "settings.json"
    if settings_file.is_file():
        try:
            settings_data = json.loads(settings_file.read_text(encoding="utf-8"))
            gcp_cfg = settings_data.get("gcp") if isinstance(settings_data, Mapping) else None
            if isinstance(gcp_cfg, Mapping):
                settings_project = str(gcp_cfg.get("project") or "").strip() or settings_project
                if not location:
                    location = str(gcp_cfg.get("location") or "").strip()
        except (OSError, ValueError):
            pass
    return explicit, settings_project, location, user_tier


def _license_lookup_allowed(source: Mapping[str, str]) -> bool:
    if (source.get("CC_AGY_LICENSE_LOOKUP") or "").strip().lower() in ("0", "false", "off", "no"):
        return False
    if source.get("PYTEST_CURRENT_TEST"):
        return (source.get("CC_AGY_ALLOW_TEST_LICENSE_FETCH") or "").strip() == "1"
    return True


def resolve_enterprise_license_for_token(
    stored_token: Mapping[str, Any],
    *,
    config_dir: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 8.0,
) -> dict[str, Any]:
    """Resolve `user_tier`, `tier_display_name`, `project_id`, and `region` for a GCP OAuth token.

    Follows `Manager.resolveEnterpriseLicense` in agy (`acpauth/manager.go`): ask
    `businessaicode.googleapis.com/v1beta:fetchLicenses`, take the license for the project the
    user chose, otherwise the account's license. Differences, on purpose:
    - `selfAssignLicense` is tried only for a project the user set explicitly, never for a
      project this plugin wrote into settings.json itself.
    - A saved license is re-resolved when the user's explicit project differs from it.
    - A lookup failure is reported in `error`, never as "no license".

    Returns:
        A dict with the four license fields (empty strings when unresolved), `alternatives`
        (other seat projects on the account), and `error` (why the lookup failed, or "").
    """
    source = os.environ if env is None else env
    existing_tier = str(stored_token.get("user_tier") or "").strip()
    existing_display = str(stored_token.get("tier_display_name") or existing_tier).strip()
    existing_proj = str(stored_token.get("project_id") or "").strip()
    existing_region = str(stored_token.get("region") or "").strip()
    explicit_proj, settings_proj, cfg_loc, cfg_tier = _configured_enterprise_project_and_location(config_dir, source)

    empty: dict[str, Any] = {
        "user_tier": "",
        "tier_display_name": "",
        "project_id": "",
        "region": "",
        "alternatives": [],
        "error": "",
    }
    if existing_tier and existing_proj and (not explicit_proj or explicit_proj == existing_proj):
        return {
            **empty,
            "user_tier": existing_tier,
            "tier_display_name": existing_display or existing_tier,
            "project_id": existing_proj,
            "region": existing_region or "global",
        }

    preferred_proj = explicit_proj or existing_proj or settings_proj
    preferred_loc = cfg_loc or existing_region

    tok_obj = stored_token.get("token")
    access_token = str(tok_obj.get("access_token") or "").strip() if isinstance(tok_obj, Mapping) else ""

    chosen: dict[str, str] | None = None
    alternatives: list[str] = []
    error = ""
    if _license_lookup_allowed(source):
        try:
            licenses = fetch_enterprise_licenses(access_token, timeout_seconds=timeout_seconds)
        except LicenseLookupError as exc:
            licenses = []
            error = str(exc)
        if not error:
            matching = [lic for lic in licenses if preferred_proj and lic["project_id"] == preferred_proj]
            if preferred_loc:
                matching.sort(key=lambda lic: lic["region"] not in (preferred_loc, "", "global"))
            if matching:
                chosen = matching[0]
            elif explicit_proj:
                chosen = self_assign_enterprise_license(
                    access_token,
                    explicit_proj,
                    preferred_loc or "global",
                    timeout_seconds=timeout_seconds,
                )
                if chosen is None:
                    error = (
                        f"no Gemini Enterprise license for project '{explicit_proj}' "
                        "(CC_AGY_ENTERPRISE_PROJECT), and self-assigning one failed"
                    )
            elif licenses:
                chosen = licenses[0]
            if chosen is not None:
                alternatives = sorted(
                    {lic["project_id"] for lic in licenses if lic["project_id"] and lic["project_id"] != chosen["project_id"]}
                )

    if chosen is not None and (chosen["user_tier"] or cfg_tier):
        return {
            **empty,
            "user_tier": chosen["user_tier"] or cfg_tier,
            "tier_display_name": chosen["tier_display_name"] or chosen["user_tier"] or cfg_tier,
            "project_id": chosen["project_id"] or preferred_proj,
            "region": chosen["region"] or preferred_loc or "global",
            "alternatives": alternatives,
        }

    # Fallback when the lookup found nothing: only if the user configured BOTH a tier and a
    # project explicitly. A project without a tier would switch agy to pay-as-you-go Agent
    # Platform billing (`ResolveAuthMode` -> `AuthModeBusinessAgentPlatformPayGo`).
    if cfg_tier and explicit_proj:
        return {
            **empty,
            "user_tier": cfg_tier,
            "tier_display_name": cfg_tier,
            "project_id": explicit_proj,
            "region": preferred_loc or "global",
        }
    return {**empty, "error": error}


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


def _owned_private_dir(path: Path) -> Path | None:
    """Create or reuse `path` as a 0700 directory owned by this user; None if it isn't safe."""
    try:
        path.mkdir(mode=0o700, parents=False)
    except FileExistsError:
        pass
    except OSError:
        return None
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        return None
    if stat.S_IMODE(info.st_mode) & 0o077:
        os.chmod(path, 0o700)
    return path


def _login_state_dir(env: Mapping[str, str]) -> Path:
    """Return a short `0700` per-user directory for the sign-in socket (<= 100 bytes for `AF_UNIX`).

    The /tmp name is predictable, so on a shared machine another user could create it first and
    swap in their own socket to collect this user's sign-in code. A directory we don't own is
    never used; the helper falls back to one under the user's home instead.
    """
    home_key = hashlib.sha256(str(_home_dir(env)).encode("utf-8")).hexdigest()[:12]
    candidates = (
        Path(tempfile.gettempdir()) / f"cc-agy-login-{os.getuid()}-{home_key}",
        _home_dir(env) / ".cache" / f"cc-agy-login-{home_key}",
    )
    for candidate in candidates:
        if candidate.name.startswith("cc-agy-login-") and candidate.parent.name == ".cache":
            candidate.parent.mkdir(parents=True, exist_ok=True)
        owned = _owned_private_dir(candidate)
        if owned is not None:
            return owned
    raise RuntimeError(
        f"Refusing to use {candidates[0]} (not a directory owned by you) and couldn't create "
        f"{candidates[1]}. Set TMPDIR to a private directory and retry."
    )


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
    # Use the standalone file only when it is the newest sign-in: a leftover from an older
    # attempt must never replace a working CLI token.
    standalone_is_newest = standalone_token_file.is_file() and standalone_token_file.stat().st_size > 10 and (
        not cli_token_file.is_file()
        or standalone_token_file.stat().st_mtime_ns > cli_token_file.stat().st_mtime_ns
    )
    if standalone_is_newest:
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

    resolved_license: dict[str, Any] = {
        "user_tier": "",
        "tier_display_name": "",
        "project_id": "",
        "region": "",
        "alternatives": [],
        "error": "",
    }
    token_on_disk = isinstance(parsed_token, dict)
    if token_on_disk:
        resolved_license = resolve_enterprise_license_for_token(
            parsed_token,
            config_dir=config_dir,
            env=env,
        )
        for field in ("user_tier", "tier_display_name", "project_id", "region"):
            if resolved_license[field]:
                parsed_token[field] = resolved_license[field]
        _atomic_write_private(
            cli_token_file,
            json.dumps(parsed_token).encode("utf-8"),
        )

    effective_tier = resolved_license["user_tier"]
    effective_display = resolved_license["tier_display_name"]
    effective_proj = resolved_license["project_id"]
    effective_region = resolved_license["region"] or "global"
    # Only a license we actually resolved counts. Writing "onboarding complete" without one
    # makes doctor report "signed in" while agy rejects every call.
    verified = bool(effective_tier and effective_proj)

    if verified:
        onboarding_file = config_dir / "cache" / "onboarding.json"
        existing_onboarding: dict[str, Any] = {}
        if onboarding_file.is_file():
            try:
                loaded = json.loads(onboarding_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing_onboarding = loaded
            except (OSError, ValueError):
                existing_onboarding = {}
        updated_onboarding: dict[str, Any] = {
            **existing_onboarding,
            "consumerOnboardingComplete": False,
            "enterpriseOnboardingComplete": True,
            "onboardingComplete": bool(existing_onboarding.get("onboardingComplete", False)),
            "previousAuthMethod": prev_auth_method,
            "cloudaicompanionProject": effective_proj,
            "tierId": effective_tier,
            "tierName": effective_display or effective_tier,
        }
        _atomic_write_private(
            onboarding_file,
            (json.dumps(updated_onboarding, indent=2) + "\n").encode("utf-8"),
        )

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
        gcp_dict["location"] = effective_region
        existing_settings["gcp"] = gcp_dict
        _atomic_write_private(
            settings_file,
            (json.dumps(existing_settings, indent=2) + "\n").encode("utf-8"),
        )

    if verified:
        problem = None
    elif not token_on_disk:
        # agy saved the token in the OS keychain (macOS/Windows), where this plugin can't read it.
        problem = "keychain"
    elif resolved_license["error"]:
        problem = "lookup_failed"
    else:
        problem = "no_license"
    return {
        "config_dir": str(config_dir),
        "previous_auth_method": prev_auth_method,
        "verified": verified,
        "problem": problem,
        "error": resolved_license["error"] or None,
        "user_tier": effective_tier or None,
        "tier_display_name": (effective_display or effective_tier) or None,
        "project_id": effective_proj or None,
        "region": effective_region if effective_proj else None,
        "alternatives": list(resolved_license.get("alternatives") or []),
    }


def describe_onboarding_problem(onboarding: Mapping[str, Any]) -> str:
    """One plain-English sentence explaining an unverified sign-in, with the fix."""
    problem = onboarding.get("problem")
    if problem == "keychain":
        return (
            "Signed in, but agy saved the token in your OS keychain, so the plugin can't check "
            "your license from a pasted code. Run `/cc-agy-plugin:login --force` on this computer and "
            "finish the sign-in in the browser window it opens (agy then records the license itself). "
            "If no browser window appears on your screen, set CC_AGY_TOKEN_IN_KEYCHAIN=0 only if agy "
            "keeps its token in ~/.gemini/antigravity-cli on this machine."
        )
    if problem == "lookup_failed":
        return (
            f"Signed in, but the license check failed: {onboarding.get('error')}. "
            "Check your network, then run `/cc-agy-plugin:login` again."
        )
    return (
        "Signed in, but this Google account has no Gemini Enterprise license. Sign in with the "
        "work account that has the seat (`/cc-agy-plugin:login --force`), or set "
        "CC_AGY_ENTERPRISE_PROJECT=<project with your seats> and run it again."
    )


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


# Variables an ambient shell may carry that must not reach agy's own sign-in. agy's CDE login
# saves GOOGLE_CLOUD_PROJECT as the token's project with an empty tier, which switches agy to
# pay-as-you-go Agent Platform billing; ANTIGRAVITY_PROJECT_ID is often a workspace ID, and API
# keys would change which backend agy picks.
_LOGIN_ENV_SCRUB = (
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "CLOUDSDK_CORE_PROJECT",
    "GOOGLE_CLOUD_QUOTA_PROJECT",
    "ANTIGRAVITY_PROJECT_ID",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
)


def _login_child_env(env: Mapping[str, str]) -> dict[str, str]:
    child = dict(env)
    for name in _LOGIN_ENV_SCRUB:
        child.pop(name, None)
    return child


def try_acp_business_browser_login(
    agy_bin: str,
    *,
    env: Mapping[str, str],
    timeout_seconds: float = 110.0,
) -> tuple[bool, str]:
    """Sign in through agy's own browser flow (`agy acp` authenticate `oauth-business`).

    Every read has a deadline: `agy acp` can stay silent (browser closed, sign-in abandoned),
    and a blocking `readline()` would hang the caller, which on the agent path is the user's
    Claude Code session.
    """
    import queue
    import threading

    proc = subprocess.Popen(
        [agy_bin, "acp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=_login_child_env(env),
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def _pump() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            lines.put(raw)
        lines.put(None)

    threading.Thread(target=_pump, daemon=True).start()

    def _await_reply(request_id: int, deadline: float) -> tuple[dict[str, Any] | None, str]:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "timed out waiting for the browser sign-in"
            try:
                raw = lines.get(timeout=remaining)
            except queue.Empty:
                return None, "timed out waiting for the browser sign-in"
            if raw is None:
                return None, "agy acp exited before the sign-in finished"
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if isinstance(msg, dict) and msg.get("id") == request_id:
                return msg, ""

    try:
        assert proc.stdin is not None
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": 1, "clientInfo": {"name": "cc-agy-plugin", "version": "0.3.2"}},
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        reply, reason = _await_reply(1, time.monotonic() + min(20.0, timeout_seconds))
        if reply is None:
            return False, reason
        proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "authenticate", "params": {"methodId": "oauth-business"}})
            + "\n"
        )
        proc.stdin.flush()
        reply, reason = _await_reply(2, time.monotonic() + timeout_seconds)
        if reply is None:
            return False, reason
        if reply.get("error"):
            err_obj = reply["error"]
            err_msg = err_obj.get("message") if isinstance(err_obj, Mapping) else str(err_obj)
            return False, str(err_msg or "agy acp authenticate failed")

        # authenticate(oauth-business) only succeeds after agy's own resolveEnterpriseLicense
        # found and saved a license. A leftover broken token file from an earlier attempt would
        # now shadow that sign-in for doctor, so drop it.
        config_dir = agy_cli_config_dir(env)
        token_file = config_dir / "antigravity-oauth-token"
        from server.agy_cli import stored_token_problem

        if token_file.exists() and stored_token_problem(token_file) is not None:
            token_file.unlink(missing_ok=True)
        onboarding = finalize_enterprise_onboarding(
            config_dir,
            _home_dir(env),
            had_standalone_token_before=True,
            env=env,
        )
        if onboarding.get("problem") == "keychain":
            # The token went to the OS keychain, which this plugin can't read; agy confirmed it.
            _mark_enterprise_onboarding_complete(config_dir)
        return True, ""
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
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
    child_env = {**_login_child_env(env), "AGY_CLI_CDE_AUTH_ACTION": "login"}
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
    if not onboarding.get("verified"):
        return {
            "ok": False,
            "signed_in_unverified": True,
            "email": email,
            "problem": onboarding.get("problem"),
            "error": describe_onboarding_problem(onboarding),
        }
    result: dict[str, Any] = {
        "ok": True,
        "email": email,
        "tier_id": onboarding.get("user_tier"),
        "tier_display_name": onboarding.get("tier_display_name"),
        "project_id": onboarding.get("project_id"),
        "message": clean_out.strip(),
    }
    if onboarding.get("alternatives"):
        result["note"] = (
            f"Using seat project {onboarding.get('project_id')}. This account also has seats in: "
            f"{', '.join(onboarding['alternatives'])}. To switch, set CC_AGY_ENTERPRISE_PROJECT and run "
            "`/cc-agy-plugin:login` again."
        )
    return result


def _env_truthy(source: Mapping[str, str], name: str) -> bool:
    return (source.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _set_agy_settings_project(config_dir: Path, project: str) -> None:
    """Write `gcp.project` into agy's settings.json, which agy's own license lookup reads."""
    settings_file = config_dir / "settings.json"
    data: dict[str, Any] = {}
    if settings_file.is_file():
        try:
            loaded = json.loads(settings_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
    gcp = dict(data.get("gcp")) if isinstance(data.get("gcp"), Mapping) else {}
    gcp["project"] = project
    gcp.setdefault("location", "global")
    data["gcp"] = gcp
    _atomic_write_private(settings_file, (json.dumps(data, indent=2) + "\n").encode("utf-8"))


def _mark_enterprise_onboarding_complete(config_dir: Path) -> None:
    """Record that agy itself confirmed the enterprise sign-in (token kept in the OS keychain)."""
    onboarding_file = config_dir / "cache" / "onboarding.json"
    existing: dict[str, Any] = {}
    if onboarding_file.is_file():
        try:
            loaded = json.loads(onboarding_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, ValueError):
            existing = {}
    existing.update(
        {
            "consumerOnboardingComplete": False,
            "enterpriseOnboardingComplete": True,
            "previousAuthMethod": "gcp",
        }
    )
    # agy resolved the license for the project in its settings.json; record it, not an older one.
    try:
        settings = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
        project = settings.get("gcp", {}).get("project") if isinstance(settings, dict) else None
        if project:
            existing["cloudaicompanionProject"] = str(project)
    except (OSError, ValueError, AttributeError):
        pass
    _atomic_write_private(onboarding_file, (json.dumps(existing, indent=2) + "\n").encode("utf-8"))


def _cleanup_stale_hold_session(state_dir: Path) -> None:
    session_file = state_dir / "session.json"
    sock_file = state_dir / "submit.sock"
    if session_file.is_file():
        try:
            meta = json.loads(session_file.read_text(encoding="utf-8"))
            old_pid = int(meta.get("pid") or 0)
            if _is_hold_daemon(old_pid, state_dir):
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
                "enterprise_project": _configured_enterprise_project_and_location(config_dir, env)[0],
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
                exchange_env = env
                if raw_req.startswith("{"):
                    # {"code": ..., "env": {...}}: settings the user changed after this session
                    # started (e.g. CC_AGY_ENTERPRISE_PROJECT) apply to the license lookup.
                    try:
                        payload = json.loads(raw_req)
                        overrides = payload.get("env") if isinstance(payload, dict) else None
                        if isinstance(overrides, dict):
                            exchange_env = {
                                **env,
                                **{k: str(v) for k, v in overrides.items() if k in _COMPLETE_ENV_OVERRIDES and v},
                            }
                        raw_req = str(payload.get("code") or "") if isinstance(payload, dict) else ""
                    except ValueError:
                        raw_req = ""
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
                        env=exchange_env,
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
    new browser sign-in is needed. It also re-resolves a saved license when the user set
    CC_AGY_ENTERPRISE_PROJECT to a different seat project.

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
    explicit_proj, _settings_proj, _loc, _tier = _configured_enterprise_project_and_location(config_dir, source)
    saved_proj = str(stored.get("project_id") or "")
    if (stored.get("user_tier") or saved_proj) and (not explicit_proj or explicit_proj == saved_proj):
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
    healed = bool(onboarding.get("verified"))
    reason = None
    if not healed:
        reason = onboarding.get("error") or (
            "the license service found no Gemini Enterprise license for this account"
        )
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
        "project_id": onboarding.get("project_id") if healed else None,
        "alternatives": onboarding.get("alternatives") or [],
        "reason": reason,
    }


def _process_command_line(pid: int) -> str:
    """Return a process's command line (macOS and Linux), or "" when it isn't running."""
    try:
        proc = subprocess.run(
            # -ww: without a terminal, ps truncates command lines at 80 columns.
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        proc = None
    if proc is not None and proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    try:  # Slim containers often have no `ps`.
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _is_hold_daemon(pid: int, state_dir: Path) -> bool:
    """True only for this plugin's sign-in holder for `state_dir`, never an unrelated process."""
    if pid <= 1 or pid == os.getpid():
        return False
    cmdline = _process_command_line(pid)
    return "--hold-daemon" in cmdline and str(state_dir) in cmdline


def _live_hold_session(state_dir: Path) -> dict[str, Any] | None:
    """Return the waiting sign-in session, if its holder is alive and not expired."""
    session_file = state_dir / "session.json"
    try:
        meta = json.loads(session_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not meta.get("auth_url"):
        return None
    if float(meta.get("expires_at") or 0) <= time.time() + 30:
        return None
    if not (state_dir / "submit.sock").exists() or not _is_hold_daemon(int(meta.get("pid") or 0), state_dir):
        return None
    return meta


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
    explicit_proj = _configured_enterprise_project_and_location(agy_cli_config_dir(source), source)[0]
    if not force:
        healed = heal_enterprise_token(source)
        if explicit_proj and not (healed.get("healed") or healed.get("already_complete")):
            reason = healed.get("reason") or "the license lookup failed"
            if "no readable token" not in reason and "not a Gemini Enterprise token" not in reason:
                # The user asked for a specific seat project and the switch failed: say so
                # instead of reporting the old sign-in as fine.
                return {"ok": False, "error": f"Couldn't switch to seat project '{explicit_proj}': {reason}."}
        status = inspect_agy_cli_offline(source)
        project_change_pending = bool(
            explicit_proj
            and not healed.get("healed")
            and explicit_proj != (healed.get("project_id") or status.get("companion_project") or "")
        )
        if status.get("auth_configured") and not status.get("consumer_account_detected") and not project_change_pending:
            return {
                "ok": True,
                "already_signed_in": True,
                "tier_id": status.get("tier_id"),
                "project_id": healed.get("project_id") or status.get("companion_project"),
                "credential_source": status.get("credential_source"),
            }
    else:
        status = inspect_agy_cli_offline(source)
    agy_bin = status.get("agy_bin")
    if not agy_bin:
        return {"ok": False, "error": status.get("error") or "External `agy` binary is not installed."}

    keyring_machine = _platform_keeps_token_in_keychain(source)
    fallback_note = ""
    if can_use_local_browser_loopback(source) or (keyring_machine and _env_truthy(source, "CC_AGY_FORCE_URL_LOGIN")):
        # On a local desktop, agy's own browser sign-in records the license even when the token
        # goes to the OS keychain (macOS), which the copy-paste flow below can't verify; so on such
        # a machine the browser is tried first even when the link flow was requested.
        if explicit_proj:
            # agy's resolveEnterpriseLicense reads gcp.project from its own settings.json.
            _set_agy_settings_project(agy_cli_config_dir(source), explicit_proj)
        # 90s, not longer: agent shells (Claude Code's Bash tool) kill commands after ~120s by
        # default, and the user must see the error rather than a killed command.
        signed_in, browser_error = try_acp_business_browser_login(agy_bin, env=source, timeout_seconds=90.0)
        if signed_in:
            return {"ok": True, "signed_in": True, "method": "browser"}
        # Browser sign-in didn't finish (window on a display the user can't see, SSO took longer
        # than the wait, tab closed). Fall back to the link: if agy then saves the token to a
        # keychain the plugin can't read, --complete says so plainly.
        fallback_note = f"The browser sign-in didn't finish ({browser_error}), so here is a link instead."
    else:
        fallback_note = ""

    state_dir = _login_state_dir(source)
    if not force:
        live = _live_hold_session(state_dir)
        if live is not None and (live.get("enterprise_project") or "") == explicit_proj:
            # A second tab, or the agent showing the link again: hand back the same link instead
            # of killing the session the user may be signing in to right now.
            return {**live, "reused": True, **({"note": fallback_note} if fallback_note else {})}
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
                    if fallback_note and data.get("auth_url"):
                        data["note"] = fallback_note
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
                "Run `/cc-agy-plugin:login` to get a fresh sign-in link."
            ),
        }

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(35.0)
            client.connect(str(sock_path))
            overrides = {k: source[k] for k in _COMPLETE_ENV_OVERRIDES if (source.get(k) or "").strip()}
            client.sendall((json.dumps({"code": clean_code, "env": overrides}) + "\n").encode("utf-8"))
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
                "Run `/cc-agy-plugin:login` to get a fresh sign-in link."
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
    20-char URL fits on one line on every terminal width. Off by default because it sends the
    sign-in URL to a third-party service; opt in with `CC_AGY_SHORT_URL=1`.
    """
    source = os.environ if env is None else env
    if (source.get("CC_AGY_SHORT_URL") or "0").strip().lower() not in ("1", "true", "yes", "on"):
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


_COMPLETE_ENV_OVERRIDES = (
    "CC_AGY_ENTERPRISE_PROJECT",
    "CC_AGY_ENTERPRISE_LOCATION",
    "CC_AGY_ENTERPRISE_USER_TIER",
    "CLAUDE_PLUGIN_OPTION_ENTERPRISE_PROJECT",
    "CC_AGY_USERCFG_ENTERPRISE_PROJECT",
)
# CURSOR_TRACE_ID is left out on purpose: Cursor also sets it in terminals a person types in.
_AGENT_ENV_MARKERS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "OPENCODE", "CODEX_SANDBOX")


def is_interactive_terminal(env: Mapping[str, str]) -> bool:
    """True only for a person at a terminal, never an agent's shell tool.

    Agent tool runners often inherit the user's `/dev/tty` while piping stdout. Prompting on
    that tty hangs the agent forever, so also require stdout to be a terminal and no agent
    marker variable to be set.
    """
    if any((env.get(name) or "").strip() for name in _AGENT_ENV_MARKERS):
        return False
    try:
        return os.isatty(1)
    except OSError:
        return False


def _print_login_result(result: Mapping[str, Any], write: Any) -> None:
    if result.get("ok"):
        who = f" as {result['email']}" if result.get("email") else ""
        write(
            f"\nSigned in to Gemini Enterprise{who} "
            f"(tier={result.get('tier_id')}, project={result.get('project_id')}).\n"
        )
        if result.get("note"):
            write(f"{result['note']}\n")
    else:
        write(f"\nSign-in did not finish: {result.get('error')}\n")


def run_interactive_or_url_login(
    env: Mapping[str, str] | None = None,
    *,
    browser_timeout_seconds: float = 110.0,
    force: bool = False,
) -> int:
    """Automatic onboarding entrypoint used by `quickstart.sh` and `agy_signin.sh`."""
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

    # 1. Local desktop with a browser (not SSH): agy's own browser sign-in, no copy-paste.
    if can_use_local_browser_loopback(source):
        print("Opening your browser for Gemini Enterprise sign-in...", flush=True)
        ok, reason = try_acp_business_browser_login(agy_bin, env=source, timeout_seconds=browser_timeout_seconds)
        if ok:
            print("Signed in to Gemini Enterprise.")
            return 0
        print(f"Browser sign-in didn't finish ({reason}); showing a link instead.", flush=True)

    # 2. A person at a terminal (`curl ... | bash` in an SSH shell): print the link, read the code.
    tty_pair = _open_controlling_tty() if is_interactive_terminal(source) else None
    if tty_pair is not None:
        tty_in, tty_out = tty_pair
        proc: subprocess.Popen[str] | None = None
        try:
            home_dir = _home_dir(source)
            config_dir = agy_cli_config_dir(source)
            had_standalone = (home_dir / ".gemini" / "jetski-standalone-oauth-token").is_file()
            proc, auth_url = _spawn_cde_login_and_capture_url(agy_bin, env=source)
            short_url = shorten_and_verify_oauth_url(auth_url, source)
            copied_osc52 = copy_url_to_terminal_clipboard_osc52(auth_url)
            tty_out.write("\n  Gemini Enterprise sign-in\n  1. Open this link in your browser:\n\n")
            if short_url:
                tty_out.write(f"{short_url}\n\n")
            # The full URL goes at column 0 so terminals can detect and open it.
            tty_out.write(f"{auth_url}\n\n")
            if copied_osc52:
                tty_out.write("  (If your terminal supports it, the link is also on your clipboard.)\n")
            tty_out.write("  2. Sign in, click 'Copy to Clipboard' on the Antigravity page, and paste the code here.\n")
            while True:
                tty_out.write("  Authorization code (or press Ctrl+D to skip): ")
                tty_out.flush()
                pasted = tty_in.readline()
                if not pasted:
                    tty_out.write("\nSign-in skipped. Run `/cc-agy-plugin:login` later to finish.\n")
                    return 0
                try:
                    extract_oauth_code(pasted)
                except ValueError as exc:
                    tty_out.write(f"  {exc}\n")
                    continue
                break
            result = _exchange_code_with_running_cde_proc(
                proc,
                pasted,
                config_dir=config_dir,
                home_dir=home_dir,
                had_standalone_before=had_standalone,
                env=source,
            )
            proc = None
            _print_login_result(result, tty_out.write)
            return 0 if result.get("ok") else 1
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
            tty_in.close()
            tty_out.close()

    # 3. CI or an explicit --non-interactive run: never leave a sign-in helper waiting.
    if (source.get("CI") or "").strip().lower() in ("1", "true", "yes") or (
        source.get("CC_AGY_NONINTERACTIVE") or ""
    ).strip() == "1":
        print("Gemini Enterprise sign-in is still needed. Run `/cc-agy-plugin:login` in Claude Code.")
        return 0

    # 4. An agent's shell (Claude Code, Cursor, OpenCode): start the 2-step sign-in and print
    #    the link for the agent to show.
    started = start_two_step_login(source, force=force)
    if not started.get("ok"):
        print(f"Could not start Gemini Enterprise sign-in: {started.get('error')}", file=sys.stderr)
        return 1
    if started.get("already_signed_in") or started.get("signed_in"):
        print("Gemini Enterprise sign-in verified.")
        return 0

    plugin_root = Path(__file__).resolve().parent.parent
    complete_cmd = f'{shlex.quote(sys.executable)} "{plugin_root}/server/agy_login.py" --complete \'<PASTE_CODE>\''
    auth_url = started["auth_url"]
    short_url = shorten_and_verify_oauth_url(auth_url, source)
    short_line = f"Short link: {short_url}\n" if short_url else ""
    print(
        "\n"
        "========================================================================\n"
        "GEMINI ENTERPRISE SIGN-IN READY\n\n"
        f"{short_line}"
        f"Sign-in link (put it on its own line, unindented):\n{auth_url}\n\n"
        "AGENT: show the sign-in link above to the user on its own line, and also as\n"
        "[Sign in with Google](<the link>). Ask them to sign in with their Gemini Enterprise\n"
        "work account, click 'Copy to Clipboard' on the Antigravity page, and paste the code.\n"
        "When they paste it, run:\n"
        f"  {complete_cmd}\n"
        "Never tell the user to sign in with `agy -p`.\n"
        "========================================================================\n",
        flush=True,
    )
    return 0


def _apply_host_settings_to_environ() -> None:
    """Fill unset CC_AGY_* settings from the host's saved plugin config (e.g. ~/.claude/settings.json).

    The MCP server gets these from the host; a slash command's shell does not, so without this
    an enterprise project chosen at install time would be ignored by sign-in.
    """
    try:
        from server.doctor import _discover_installed_host_mcp_env

        discovered = _discover_installed_host_mcp_env(dict(os.environ))
    except Exception:
        return
    for key, value in (discovered or {}).items():
        if not value or str(value).startswith("${"):
            continue
        if key in ("CC_AGY_USERCFG_ENTERPRISE_PROJECT", "CLAUDE_PLUGIN_OPTION_ENTERPRISE_PROJECT", "CC_AGY_ENTERPRISE_PROJECT"):
            # A project saved at install time is a preference: it must not override a later
            # `/login --project` switch, and on a machine with several hosts it may be another
            # host's setting.
            os.environ.setdefault("CC_AGY_HOST_ENTERPRISE_PROJECT", str(value))
        elif key.startswith(("CC_AGY_", "CLAUDE_PLUGIN_OPTION_")) and not os.environ.get(key):
            os.environ[key] = str(value)


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
    parser.add_argument(
        "--project",
        default=None,
        help="Gemini Enterprise seat project to use (same as CC_AGY_ENTERPRISE_PROJECT).",
    )
    args = parser.parse_args(argv)
    _apply_host_settings_to_environ()
    if args.project:
        os.environ["CC_AGY_ENTERPRISE_PROJECT"] = args.project.strip()

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
                f"(tier={res.get('tier_id')}, project={res.get('project_id')})."
            )
            if res.get("note"):
                print(f"[NOTE] {res['note']}")
        else:
            print(f"[FAIL] {res.get('error')}", file=sys.stderr)
        return 0 if res.get("ok") else 1

    if args.start:
        try:
            res = start_two_step_login(timeout_seconds=args.timeout, force=args.force)
        except RuntimeError as exc:
            res = {"ok": False, "error": str(exc)}
        if res.get("ok") and res.get("auth_url"):
            res["short_url"] = shorten_and_verify_oauth_url(res["auth_url"])
            # True only means the clipboard request was written to a terminal; terminals such as
            # macOS Terminal.app ignore it, so callers must word this as "may be on your clipboard".
            res["clipboard_requested"] = copy_url_to_terminal_clipboard_osc52(res["auth_url"])
            res["complete_command"] = (
                f'{shlex.quote(sys.executable)} "{Path(__file__).resolve().parent.parent}/server/agy_login.py" '
                "--complete '<PASTE_CODE>'"
            )
        if args.json:
            print(json.dumps(res, indent=2))
        elif res.get("already_signed_in"):
            print(f"[OK] Already signed in to Gemini Enterprise (tier={res.get('tier_id')}).")
        elif res.get("signed_in"):
            print("[OK] Signed in to Gemini Enterprise in your browser.")
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
