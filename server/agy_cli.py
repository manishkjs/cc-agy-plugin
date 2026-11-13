"""Standalone Antigravity CLI (`agy`) runtime adapter for Gemini Enterprise seats.

Cloud Code PA rejects `useragent.SDK` requests on the Business OAuth path
(`The Antigravity SDK is not available on an enterprise account`), while
explicitly permitting `useragent.CLI` (`Antigravity CLI` / `agy`). This module
discovers or bootstraps the external `agy` binary (skipping Google-internal
`jetski` builds), inspects local OAuth readiness without echoing tokens, maps
`agy` headless JSON / `stream-json` envelopes and token usage to Vertex/GenAI
accounting without double-counting thinking or cached tokens, and runs
cancelled-cleanly subprocesses for workers, single-shot Q&A, and batch writers.
"""

from __future__ import annotations

from collections import OrderedDict
import contextlib
import json
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence

DEFAULT_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
MANAGED_CLI_BIN_NAME = "agy.exe" if os.name == "nt" else "agy"

_USAGE_KEYS = (
    "prompt_token_count",
    "cached_content_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)

_SCRUBBED_SUBPROCESS_ENV_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_VERTEX_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
        "CC_AGY_GEMINI_BASE_URL",
        "CC_AGY_VERTEX_BASE_URL_TOKEN",
        "CC_AGY_BIFROST_VIRTUAL_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "VERTEX_PROJECT",
        "CLOUD_ML_PROJECT_ID",
        "GOOGLE_CLOUD_QUOTA_PROJECT",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_OPENAI_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "AGY_USE_BAIC_FETCH_CONFIG",
        "AGY_LLM_GATEWAY_URL",
        "AGY_LLM_GATEWAY_API_KEY",
        "AGY_ADC_AUTH",
        "BAICODE_USER_TIER",
        "BIFROST_VK_AGY",
        "BIFROST_VK_DEFAULT",
        "BIFROST_POLICY_USER",
        "BIFROST_POLICY_PASSWORD",
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "SSH_ASKPASS",
        "SUDO_ASKPASS",
        "GPG_AGENT_INFO",
        "GIT_ASKPASS",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_TERMINAL_PROMPT",
        "KUBECONFIG",
        "DOCKER_AUTH_CONFIG",
        "PGPASSWORD",
        "MYSQL_PWD",
        "REDISCLI_AUTH",
        "VAULT_TOKEN",
        "NPM_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GITLAB_TOKEN",
        "BITBUCKET_TOKEN",
        "SLACK_TOKEN",
        "SLACK_BOT_TOKEN",
    }
)
# Prefixes of variables that carry git config overrides (credential helpers, URL rewrites).
_SCRUBBED_SUBPROCESS_ENV_PREFIXES = ("CC_AGY_", "CLAUDE_PLUGIN_", "GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
_SCRUBBED_SECRET_ENV_RE = re.compile(
    r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|APIKEY|CREDENTIAL|CREDENTIALS|PRIVATE_KEY|AUTH_CONFIG)(?:$|_)",
    re.IGNORECASE,
)
_ALLOWED_SECRET_ENV_KEYS = frozenset({"ANTIGRAVITY_OAUTH_TOKEN"})

# Gemini Enterprise seat tiers `agy` accepts in `BAICODE_USER_TIER` for Developer tools metrics.
ENTERPRISE_USER_TIERS: frozenset[str] = frozenset({"gcp-ge-plus-tier", "gcp-ge-standard-tier"})
ENTERPRISE_LOCATIONS: frozenset[str] = frozenset({"global", "us", "eu"})
DEFAULT_ENTERPRISE_LOCATION = "global"
_HUMAN_ADC_TYPE = "authorized_user"

_EFFORT_SUFFIXES = frozenset({"low", "medium", "high", "xhigh", "max"})

_THINKING_TO_EFFORT: dict[str, str] = {
    "ZERO": "low",
    "MINIMAL": "low",
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high",
    "EXTRA_HIGH": "xhigh",
    "XHIGH": "xhigh",
    "MAX": "max",
}

_BINARY_PROBE_LOCK = threading.Lock()
_BINARY_PROBE_CACHE: dict[tuple[str, int, int], tuple[bool, str]] = {}

_MODEL_FALLBACK_LOCK = threading.Lock()
_MODEL_FALLBACK_CACHE: dict[tuple[str, str, str, str], str] = {}

class _BoundedLedger(OrderedDict):
    """Mapping that keeps only the `max_entries` most recently used keys."""

    def __init__(self, max_entries: int) -> None:
        super().__init__()
        self._max_entries = max_entries

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self._max_entries:
            self.popitem(last=False)

    def get(self, key: Any, default: Any = None) -> Any:
        if key not in self:
            return default
        self.move_to_end(key)
        return super().__getitem__(key)


_CONV_LEDGER_LOCK = threading.Lock()
_CONVERSATION_USAGE_LEDGER: _BoundedLedger = _BoundedLedger(max_entries=512)
_MIN_ATTEMPT_SECONDS = 1.0

_QUOTA_OR_TRANSIENT_PATTERNS = (
    "429",
    "resource_exhausted",
    "quota exhausted",
    "quota exceeded",
    "rate limit",
    "too many requests",
    "overloaded",
    "503",
    "service unavailable",
)

_AUTH_ERROR_PATTERNS = (
    "not logged in",
    "not authenticated",
    "unauthenticated",
    "authentication required",
    "authentication failed",
    "login required",
    "please sign in",
    "sign in with google",
    "invalid oauth",
    "expired oauth",
    "missing oauth",
    "invalid_grant",
    "token has been expired or revoked",
    "expired or revoked",
    "credentials missing",
    "no stored credentials",
    "reauthentication required",
    "missing required authentication credential",
    "antigravity sdk is not available on an enterprise account",
    "unauthorized",
)

_AUTH_CONTEXT_HTTP_RE = re.compile(
    r"(?:googleapis\.com|cloudcode|antigravity|oauth|unauthenticated)"
    r"[^\n]{0,120}\b(?:401|403|unauthorized)\b"
    r"|\b(?:401|403|unauthorized)\b[^\n]{0,120}"
    r"(?:googleapis\.com|cloudcode|antigravity|oauth|sign[- ]in)"
    r"|(?:http(?:/\d(?:\.\d)?)?|status(?:\s+code)?|error(?:\s+code)?)\s*[:=]?\s*(?:401|403)\b"
    r"|\b(?:401\s+unauthorized|403\s+forbidden)\b",
    re.IGNORECASE,
)


def _clean_env_str(env: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        val = env.get(key)
        if val is None:
            continue
        cleaned = str(val).strip()
        if cleaned and not cleaned.startswith("${"):
            return cleaned
    return ""


def _env_flag(name: str, env: Mapping[str, str]) -> bool | None:
    raw = _clean_env_str(env, name).lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def _home_dir(env: Mapping[str, str]) -> Path:
    explicit_home = _clean_env_str(env, "HOME", "USERPROFILE")
    if explicit_home:
        return Path(explicit_home).expanduser()
    return Path.home()


def managed_agy_bin_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the per-user managed directory where `install_agy_cli.sh` places `agy`."""
    source = env if env is not None else os.environ
    explicit = _clean_env_str(source, "CC_AGY_CLI_INSTALL_DIR")
    if explicit:
        return Path(explicit).expanduser()
    home = _home_dir(source)
    if os.name == "nt":
        local_app = _clean_env_str(source, "LOCALAPPDATA")
        base = Path(local_app).expanduser() if local_app else (home / "AppData" / "Local")
        return base / "cc-agy-plugin" / "bin"
    xdg_data = _clean_env_str(source, "XDG_DATA_HOME")
    base = Path(xdg_data).expanduser() if xdg_data else (home / ".local" / "share")
    return base / "cc-agy-plugin" / "bin"


def agy_cli_config_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the active `agy` CLI state directory (`~/.gemini/antigravity-cli`)."""
    source = env if env is not None else os.environ
    explicit = _clean_env_str(source, "CC_AGY_CLI_CONFIG_DIR")
    if explicit:
        return Path(explicit).expanduser()
    home = _home_dir(source)
    default_dir = home / ".gemini" / "antigravity-cli"
    if os.name == "nt" and not default_dir.exists():
        for win_var in ("APPDATA", "LOCALAPPDATA"):
            win_base = _clean_env_str(source, win_var)
            if win_base:
                cand = Path(win_base).expanduser() / "antigravity-cli"
                if cand.exists():
                    return cand
    return default_dir


def is_external_agy_binary(
    path: str | Path,
    env: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Verify `path` is a runnable external `agy` CLI and not an internal `jetski` build."""
    cand = Path(path).expanduser()
    if not (cand.is_file() and os.access(cand, os.X_OK)):
        return False, "not_executable"
    try:
        resolved = cand.resolve()
        st = resolved.stat()
    except OSError as exc:
        return False, f"stat_error:{exc}"

    cache_key = (str(resolved), st.st_mtime_ns, st.st_size)
    with _BINARY_PROBE_LOCK:
        cached = _BINARY_PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    probe_env = build_agy_subprocess_env(env)
    try:
        proc = subprocess.run(
            [str(cand), "--help"],
            cwd=tempfile.gettempdir(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            env=probe_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        verdict = (False, f"probe_failed:{exc}")
        with _BINARY_PROBE_LOCK:
            _BINARY_PROBE_CACHE[cache_key] = verdict
        return verdict

    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    if "--is_google_internal" in combined or "--use_stubby_auth" in combined:
        verdict = (False, "internal_jetski_binary")
    elif proc.returncode != 0:
        verdict = (False, f"exit_{proc.returncode}")
    elif "--output-format" not in combined:
        verdict = (False, "missing_output_format_flag")
    else:
        verdict = (True, "ok")

    with _BINARY_PROBE_LOCK:
        _BINARY_PROBE_CACHE[cache_key] = verdict
    return verdict


def _explicit_cli_bin_override(source: Mapping[str, str]) -> str:
    return _clean_env_str(
        source,
        "CC_AGY_CLI_BIN",
        "CLAUDE_PLUGIN_OPTION_AGY_CLI_PATH",
        "CC_AGY_USERCFG_AGY_CLI_PATH",
    )


def resolve_agy_cli_binary(
    env: Mapping[str, str] | None = None,
) -> tuple[str | None, list[str]]:
    """Locate the external `agy` binary, returning `(valid_path, skipped_internal_paths)`."""
    source = env if env is not None else os.environ
    skipped_internal: list[str] = []
    explicit = _explicit_cli_bin_override(source)

    if explicit:
        ok, reason = is_external_agy_binary(explicit, env=source)
        if ok:
            return str(Path(explicit).expanduser()), skipped_internal
        if reason == "internal_jetski_binary":
            skipped_internal.append(explicit)
        return None, skipped_internal

    candidates: list[Path] = []
    candidates.append(managed_agy_bin_dir(source) / MANAGED_CLI_BIN_NAME)

    path_var = source.get("PATH") if env is not None else os.environ.get("PATH")
    if path_var:
        which_hit = shutil.which("agy", path=path_var)
        if which_hit:
            candidates.append(Path(which_hit))

    home = _home_dir(source)
    candidates.append(home / ".local" / "bin" / MANAGED_CLI_BIN_NAME)

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        ok, reason = is_external_agy_binary(cand, env=source)
        if ok:
            return str(cand), skipped_internal
        if reason == "internal_jetski_binary":
            skipped_internal.append(str(cand))

    return None, skipped_internal


def _read_onboarding_metadata(config_dir: Path) -> tuple[bool, str | None, str | None]:
    """Read non-secret onboarding flags, `cloudaicompanionProject`, and `tierId` from `cache/onboarding.json`."""
    onboarding_file = config_dir / "cache" / "onboarding.json"
    try:
        if onboarding_file.is_file() and onboarding_file.stat().st_size > 2:
            data = json.loads(onboarding_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                raw_proj = data.get("cloudaicompanionProject")
                comp_proj = str(raw_proj).strip() if isinstance(raw_proj, str) and str(raw_proj).strip() else None
                raw_tier = data.get("tierId")
                tier_id = str(raw_tier).strip() if isinstance(raw_tier, str) and str(raw_tier).strip() else None
                complete = any(
                    bool(data.get(flag_key))
                    for flag_key in (
                        "onboardingComplete",
                        "enterpriseOnboardingComplete",
                        "consumerOnboardingComplete",
                    )
                ) or bool(tier_id or comp_proj)
                return complete, comp_proj, tier_id
    except (OSError, ValueError):
        pass
    return False, None, None


def _plugin_option(source: Mapping[str, str], suffix: str) -> str:
    """Read one plugin option from its env var, Claude `userConfig`, or `.mcp.json` mirror."""
    return _clean_env_str(
        source,
        f"CC_AGY_{suffix}",
        f"CLAUDE_PLUGIN_OPTION_{suffix}",
        f"CC_AGY_USERCFG_{suffix}",
    )


def _gemini_adc_setting_enabled(config_dir: Path) -> bool:
    """Return True when `<gemini_dir>/config/config.json` sets `userSettings.enableAdc` to true."""
    settings_file = config_dir.parent / "config" / "config.json"
    try:
        if not settings_file.is_file():
            return False
        data = json.loads(settings_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    user_settings = data.get("userSettings") if isinstance(data, dict) else None
    return isinstance(user_settings, dict) and user_settings.get("enableAdc") is True


def resolve_enterprise_adc_seat(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Resolve the opt-in Gemini Enterprise human ADC seat mode for `agy` without network calls.

    Seat mode is requested by `CC_AGY_ENTERPRISE_ADC=true`, by `AGY_ADC_AUTH=true` in the
    caller's shell, or automatically when `agy` has `userSettings.enableAdc` set and the ADC
    file belongs to a human (`authorized_user`). `CC_AGY_ENTERPRISE_ADC=false` turns it off.

    It becomes `active` only when the ADC file is a human credential (a `service_account`
    is never a licensed seat), a project is known (`CC_AGY_ENTERPRISE_PROJECT` or the ADC
    `quota_project_id`), that project passes `CC_AGY_ALLOWED_PROJECTS` and
    `CC_AGY_BLOCKED_PROJECTS`, and the location and tier are valid. No secret is returned.

    Returns:
        A dict with `requested`, `active`, `project`, `location`, `user_tier`, `adc_type`,
        `adc_path`, `source`, and `error` (why a requested seat is not active, else None).
    """
    from server.regions import read_adc_metadata, resolve_allowed_projects, resolve_blocked_projects

    source = env if env is not None else os.environ
    seat: dict[str, Any] = {
        "requested": False,
        "active": False,
        "project": None,
        "location": None,
        "user_tier": None,
        "adc_type": None,
        "adc_path": None,
        "source": None,
        "error": None,
    }

    opt_in: bool | None = None
    for flag_name in ("CC_AGY_ENTERPRISE_ADC", "CLAUDE_PLUGIN_OPTION_ENTERPRISE_ADC", "CC_AGY_USERCFG_ENTERPRISE_ADC"):
        opt_in = _env_flag(flag_name, source)
        if opt_in is not None:
            seat["source"] = f"env:{flag_name}"
            break
    if opt_in is False:
        return seat

    adc = read_adc_metadata(source)
    seat["adc_type"] = adc.get("type")
    seat["adc_path"] = adc.get("path")
    if opt_in is None:
        if _env_flag("AGY_ADC_AUTH", source) is True:
            seat["source"] = "env:AGY_ADC_AUTH"
        elif seat["adc_type"] == _HUMAN_ADC_TYPE and _gemini_adc_setting_enabled(agy_cli_config_dir(source)):
            seat["source"] = "auto:userSettings.enableAdc"
        else:
            return seat
    seat["requested"] = True

    if seat["adc_type"] != _HUMAN_ADC_TYPE:
        found = f"type '{seat['adc_type']}'" if seat["adc_type"] else "no ADC file"
        seat["error"] = (
            f"Enterprise ADC seat mode needs a human `{_HUMAN_ADC_TYPE}` credential, but found {found}. "
            "Run `gcloud auth application-default login` as the licensed user."
        )
        return seat

    project = _plugin_option(source, "ENTERPRISE_PROJECT") or (adc.get("quota_project_id") or "")
    if not project:
        seat["error"] = (
            "Enterprise ADC seat mode needs a project. Set CC_AGY_ENTERPRISE_PROJECT or run "
            "`gcloud auth application-default set-quota-project <project>`."
        )
        return seat
    seat["project"] = project
    if project.lower() in {p.lower() for p in resolve_blocked_projects(source)}:
        seat["error"] = f"Enterprise project '{project}' is listed in CC_AGY_BLOCKED_PROJECTS."
        return seat
    allowed = resolve_allowed_projects(source)
    if allowed and project.lower() not in {p.lower() for p in allowed}:
        seat["error"] = f"Enterprise project '{project}' is not in CC_AGY_ALLOWED_PROJECTS ({', '.join(allowed)})."
        return seat

    location = (_plugin_option(source, "ENTERPRISE_LOCATION") or DEFAULT_ENTERPRISE_LOCATION).lower()
    if location not in ENTERPRISE_LOCATIONS:
        seat["error"] = (
            f"Unsupported CC_AGY_ENTERPRISE_LOCATION '{location}'. "
            f"Expected one of: {', '.join(sorted(ENTERPRISE_LOCATIONS))}."
        )
        return seat
    seat["location"] = location

    user_tier = (_plugin_option(source, "ENTERPRISE_USER_TIER") or _clean_env_str(source, "BAICODE_USER_TIER")).lower()
    if user_tier and user_tier not in ENTERPRISE_USER_TIERS:
        seat["error"] = (
            f"Unsupported CC_AGY_ENTERPRISE_USER_TIER '{user_tier}'. "
            f"Expected one of: {', '.join(sorted(ENTERPRISE_USER_TIERS))}."
        )
        return seat
    seat["user_tier"] = user_tier or None
    seat["active"] = True
    return seat


class EnterpriseSeatError(RuntimeError):
    """Raised when Gemini Enterprise ADC seat mode was requested but cannot be used."""


def enterprise_seat_failure(env: Mapping[str, str] | None = None) -> str | None:
    """Return why a requested ADC seat cannot be used, or None when there is nothing wrong."""
    seat = resolve_enterprise_adc_seat(env)
    if not (seat["requested"] and seat["error"]):
        return None
    return (
        f"Gemini Enterprise ADC seat mode was requested ({seat['source']}) but cannot be used: {seat['error']} "
        "Nothing was sent to Gemini. Fix the setting above, or set CC_AGY_ENTERPRISE_ADC=false to use "
        "your normal `agy` sign-in instead."
    )


def ensure_enterprise_seat_ready(env: Mapping[str, str] | None = None) -> None:
    """Stop at 0 tokens when ADC seat mode was requested but is invalid, instead of falling back.

    Raises:
        EnterpriseSeatError: With the reason and how to fix it.
    """
    failure = enterprise_seat_failure(env)
    if failure:
        raise EnterpriseSeatError(failure)


def _inspect_auth_files(config_dir: Path, source: Mapping[str, str]) -> tuple[bool, str | None]:
    env_tok = _clean_env_str(source, "ANTIGRAVITY_OAUTH_TOKEN")
    if len(env_tok) >= 8:
        return True, "env:ANTIGRAVITY_OAUTH_TOKEN"

    token_file = config_dir / "antigravity-oauth-token"
    try:
        if token_file.is_file() and token_file.stat().st_size > 10:
            raw_head = token_file.read_bytes()[:256].strip()
            if raw_head:
                return True, f"token_file:{token_file}"
    except OSError:
        pass

    onboarding_ok, _comp_proj, _tier_id = _read_onboarding_metadata(config_dir)
    if onboarding_ok:
        onboarding_file = config_dir / "cache" / "onboarding.json"
        return True, f"onboarding_cache:{onboarding_file}"

    seat = resolve_enterprise_adc_seat(source)
    if seat["active"]:
        return True, f"enterprise_adc:{seat['adc_path']}"

    return False, None


def inspect_agy_cli_offline(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Inspect external `agy` binary and sign-in state without network calls or secret echoing."""
    source = env if env is not None else os.environ
    agy_bin, skipped_internal = resolve_agy_cli_binary(source)
    cfg_dir = agy_cli_config_dir(source)
    auth_ok, cred_source = _inspect_auth_files(cfg_dir, source)
    seat = resolve_enterprise_adc_seat(source)
    _onboarding_ok, companion_project, tier_id = _read_onboarding_metadata(cfg_dir)
    managed_bin = managed_agy_bin_dir(source) / MANAGED_CLI_BIN_NAME

    explicit_bin = _explicit_cli_bin_override(source)
    binary_found = agy_bin is not None
    seat_failure = enterprise_seat_failure(source)
    ok = binary_found and auth_ok and seat_failure is None

    error: str | None = None
    remediation: str | None = None
    if not binary_found:
        if explicit_bin and explicit_bin in skipped_internal:
            error = (
                f"Configured CC_AGY_CLI_BIN ({explicit_bin}) points to a Google-internal `jetski` binary "
                "rather than the external Antigravity CLI (`agy`)."
            )
        elif explicit_bin:
            error = f"Configured CC_AGY_CLI_BIN ({explicit_bin}) is missing or not a valid external `agy` binary."
        elif skipped_internal:
            error = (
                f"Skipped internal `jetski` binary on PATH ({skipped_internal[0]}); "
                "no external Antigravity CLI (`agy`) binary found."
            )
        else:
            error = "External Antigravity CLI (`agy`) is not installed yet."
        remediation = (
            "Run `python3 -m server.install --auth gemini-enterprise` (or `bash scripts/install_agy_cli.sh`) "
            f"to install `agy` to `{managed_bin}`, then run `{managed_bin}` once in a terminal to sign in."
        )
    elif not auth_ok:
        error = f"Antigravity CLI is installed at `{agy_bin}`, but no Google sign-in token was found in `{cfg_dir}`."
        remediation = (
            f"Open a terminal and run `{agy_bin}` once to sign in with your Gemini Enterprise Google account "
            "(then `/exit` or Ctrl+C), or set `ANTIGRAVITY_OAUTH_TOKEN`."
        )
        if seat["requested"] and seat["error"]:
            error = f"{error} {seat['error']}"
    elif seat_failure:
        error = seat_failure
        remediation = (
            "Run `gcloud auth application-default login` as the licensed user and check "
            "CC_AGY_ENTERPRISE_PROJECT / CC_AGY_ENTERPRISE_LOCATION / CC_AGY_ENTERPRISE_USER_TIER, "
            "or set CC_AGY_ENTERPRISE_ADC=false."
        )

    return {
        "ok": ok,
        "binary_found": binary_found,
        "agy_bin": agy_bin,
        "managed_bin_path": str(managed_bin),
        "auth_configured": auth_ok,
        "credential_source": cred_source,
        "companion_project": companion_project,
        "tier_id": tier_id,
        "skipped_internal_binaries": skipped_internal,
        "config_dir": str(cfg_dir),
        "error": error,
        "remediation": remediation,
        "adc_seat_enabled": bool(seat["active"]),
        "adc_seat_error": seat["error"],
        "enterprise_project": seat["project"] if seat["active"] else None,
        "enterprise_location": seat["location"] if seat["active"] else None,
        "baicode_user_tier": seat["user_tier"] if seat["active"] else None,
    }


def ensure_agy_cli_installed(
    plugin_root: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    manifest_url: str | None = None,
    configure_shell: bool = False,
    output_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Ensure the external `agy` binary is installed in `managed_agy_bin_dir`."""
    source = env if env is not None else os.environ
    existing_bin, skipped_internal = resolve_agy_cli_binary(source)
    if existing_bin is not None:
        return {
            "ok": True,
            "bootstrapped": False,
            "agy_bin": existing_bin,
            "skipped_internal_binaries": skipped_internal,
        }

    explicit_override = _explicit_cli_bin_override(source)
    if explicit_override:
        return {
            "ok": False,
            "bootstrapped": False,
            "agy_bin": None,
            "skipped_internal_binaries": skipped_internal,
            "error": (
                f"Explicit CC_AGY_CLI_BIN ({explicit_override}) is invalid or missing; "
                "unset CC_AGY_CLI_BIN to allow automatic installation."
            ),
        }

    root = Path(plugin_root).resolve() if plugin_root is not None else DEFAULT_PLUGIN_ROOT.resolve()
    installer = root / "scripts" / "install_agy_cli.sh"
    target_dir = managed_agy_bin_dir(source)
    if not installer.is_file():
        return {
            "ok": False,
            "bootstrapped": False,
            "agy_bin": None,
            "skipped_internal_binaries": skipped_internal,
            "error": f"Installer script missing at {installer}",
        }

    if output_fn is not None:
        output_fn(f"  Installing standalone Antigravity CLI (`agy`) to {target_dir / MANAGED_CLI_BIN_NAME} ...")

    cmd = ["bash", str(installer), "--target-dir", str(target_dir), "--json"]
    effective_manifest = manifest_url or _clean_env_str(source, "CC_AGY_CLI_MANIFEST_URL")
    if effective_manifest:
        cmd.extend(["--manifest-url", effective_manifest])
    if configure_shell:
        cmd.append("--configure-shell")

    run_env = dict(os.environ)
    run_env.update({k: str(v) for k, v in source.items() if v is not None})
    try:
        proc = subprocess.run(
            cmd,
            cwd=tempfile.gettempdir(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
            env=run_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ok": False,
            "bootstrapped": False,
            "agy_bin": None,
            "skipped_internal_binaries": skipped_internal,
            "error": f"Failed to execute install_agy_cli.sh: {exc}",
        }

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-500:]
        return {
            "ok": False,
            "bootstrapped": False,
            "agy_bin": None,
            "skipped_internal_binaries": skipped_internal,
            "error": f"install_agy_cli.sh failed (exit {proc.returncode}): {detail}",
        }

    with _BINARY_PROBE_LOCK:
        _BINARY_PROBE_CACHE.clear()

    resolved_after, skipped_after = resolve_agy_cli_binary(source)
    if resolved_after is None:
        return {
            "ok": False,
            "bootstrapped": False,
            "agy_bin": None,
            "skipped_internal_binaries": skipped_after,
            "error": f"Installed binary under {target_dir} did not pass external `agy` verification.",
        }

    payload_ver: str | None = None
    for line in reversed((proc.stdout or "").splitlines()):
        line_s = line.strip()
        if line_s.startswith("{") and line_s.endswith("}"):
            try:
                parsed = json.loads(line_s)
                if isinstance(parsed, dict):
                    payload_ver = parsed.get("version")
                    break
            except ValueError:
                pass

    return {
        "ok": True,
        "bootstrapped": True,
        "agy_bin": resolved_after,
        "version": payload_ver,
        "skipped_internal_binaries": skipped_after,
    }


def _should_auto_install_on_call(env: Mapping[str, str] | None, source: Mapping[str, str]) -> bool:
    flag = _env_flag("CC_AGY_AUTO_INSTALL_CLI", source)
    if flag is True:
        return True
    if flag is False:
        return False
    if _explicit_cli_bin_override(source):
        return False
    if env is not None or "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return True


def _resolve_or_bootstrap_agy_bin(env: Mapping[str, str] | None = None) -> str:
    source = env if env is not None else os.environ
    agy_bin, skipped_internal = resolve_agy_cli_binary(source)
    if agy_bin is not None:
        return agy_bin

    if _should_auto_install_on_call(env, source):
        boot = ensure_agy_cli_installed(DEFAULT_PLUGIN_ROOT, env=source)
        if boot.get("ok") and boot.get("agy_bin"):
            return str(boot["agy_bin"])

    status = inspect_agy_cli_offline(source)
    err_msg = status.get("error") or "Antigravity CLI (`agy`) is not installed."
    rem = status.get("remediation") or "Run `python3 -m server.install --auth gemini-enterprise`."
    if skipped_internal and "internal" not in err_msg:
        err_msg = f"{err_msg} (skipped internal binary: {skipped_internal[0]})"
    raise RuntimeError(f"{err_msg} {rem}")


def convert_agy_usage_to_vertex(agy_usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Convert `agy` `JSONUsage` into Vertex/GenAI `usage_metadata` without double-counting.

    In `agy` (`steps/json_output.go`):
    - `input_tokens` = uncached prompt tokens
    - `cache_read_tokens` = cached prompt tokens
    - `output_tokens` = total generated tokens INCLUDING `thinking_tokens`
    - `thinking_tokens` = reasoning subset of `output_tokens`
    """
    if not isinstance(agy_usage, Mapping):
        return {
            "prompt_token_count": 0,
            "cached_content_token_count": 0,
            "candidates_token_count": 0,
            "thoughts_token_count": 0,
            "total_token_count": 0,
        }

    src_map: Mapping[str, Any] = agy_usage
    if isinstance(src_map.get("usage"), Mapping):
        src_map = src_map["usage"]

    def _first_nonneg(*keys: str) -> int:
        for key in keys:
            val = src_map.get(key)
            if val is None or isinstance(val, bool):
                continue
            try:
                return max(0, int(val))
            except (TypeError, ValueError):
                continue
        return 0

    if any(
        k in src_map
        for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "cached_content_token_count")
    ) and not any(k in src_map for k in ("input_tokens", "inputTokens", "output_tokens", "outputTokens")):
        p_tok = _first_nonneg("prompt_token_count", "promptTokenCount")
        c_tok = _first_nonneg("cached_content_token_count", "cachedContentTokenCount")
        o_tok = _first_nonneg("candidates_token_count", "candidatesTokenCount")
        t_tok = _first_nonneg("thoughts_token_count", "thoughtsTokenCount")
        return {
            "prompt_token_count": p_tok,
            "cached_content_token_count": c_tok,
            "candidates_token_count": o_tok,
            "thoughts_token_count": t_tok,
            "total_token_count": p_tok + o_tok + t_tok,
        }

    uncached_in = _first_nonneg("input_tokens", "inputTokens")
    cached_in = _first_nonneg("cache_read_tokens", "cacheReadTokens")
    raw_out = _first_nonneg("output_tokens", "outputTokens")
    think_out = _first_nonneg("thinking_tokens", "thinkingTokens")

    prompt_tokens = uncached_in + cached_in
    candidates_tokens = max(0, raw_out - think_out)
    thoughts_tokens = think_out
    total_tokens = prompt_tokens + candidates_tokens + thoughts_tokens
    return {
        "prompt_token_count": prompt_tokens,
        "cached_content_token_count": cached_in,
        "candidates_token_count": candidates_tokens,
        "thoughts_token_count": thoughts_tokens,
        "total_token_count": total_tokens,
    }


def _model_has_explicit_effort(model: str | None) -> bool:
    if not model:
        return False
    m = model.strip()
    lower = m.lower()
    if lower.endswith("-effort"):
        lower = lower[: -len("-effort")]
    if "-" in lower:
        suffix = lower.rsplit("-", 1)[-1]
        if suffix in _EFFORT_SUFFIXES:
            return True
    paren = re.search(r"\((low|medium|high|xhigh|max)\)\s*$", lower)
    return paren is not None


def map_thinking_level_to_effort(
    thinking_level: str | None,
    model: str | None = None,
) -> str | None:
    """Map plugin `thinking_level` (`MINIMAL`..`EXTRA_HIGH`) to `agy --effort`."""
    if _model_has_explicit_effort(model):
        return None
    if not thinking_level:
        return None
    norm = str(thinking_level).strip().upper().replace("-", "_")
    return _THINKING_TO_EFFORT.get(norm)


def build_agy_subprocess_env(
    env: Mapping[str, str] | None = None,
    *,
    workspace_dir: str | Path | None = None,
    cwd: str | Path | None = None,
) -> dict[str, str]:
    """Build a scrubbed subprocess environment for `agy` preserving OAuth, keyring, and proxies.

    Project, quota-project, pay-as-you-go, gateway, and Bifrost variables are always removed.
    When Enterprise ADC seat mode is active (`resolve_enterprise_adc_seat`), the validated
    seat project, location, and optional tier are then set so `agy` signs in with the
    user's own ADC and reports to Gemini Enterprise Developer tools metrics.
    """
    source = env if env is not None else os.environ
    out: dict[str, str] = {}
    for k, v in source.items():
        if v is None:
            continue
        if k in _SCRUBBED_SUBPROCESS_ENV_KEYS or k in ("PWD", "OLDPWD"):
            continue
        if k.startswith(_SCRUBBED_SUBPROCESS_ENV_PREFIXES):
            continue
        if k not in _ALLOWED_SECRET_ENV_KEYS and _SCRUBBED_SECRET_ENV_RE.search(k):
            continue
        val_str = str(v)
        if k == "ANTIGRAVITY_OAUTH_TOKEN" and val_str.strip().startswith("${"):
            continue
        out[k] = val_str
    from server.worker_sandbox import git_guard_environment

    out.update(git_guard_environment())
    seat = resolve_enterprise_adc_seat(source)
    if seat["active"]:
        out["AGY_ADC_AUTH"] = "true"
        out["GOOGLE_CLOUD_PROJECT"] = seat["project"]
        out["GOOGLE_CLOUD_QUOTA_PROJECT"] = seat["project"]
        out["GOOGLE_CLOUD_LOCATION"] = seat["location"]
        if seat["user_tier"]:
            out["BAICODE_USER_TIER"] = seat["user_tier"]
    if not out.get("PATH"):
        out["PATH"] = os.environ.get("PATH") or os.defpath
    eff_ws = workspace_dir if workspace_dir is not None else cwd
    if eff_ws is not None:
        ws_path = Path(eff_ws)
        path_prefixes: list[str] = []
        nm_bin = ws_path / "node_modules" / ".bin"
        if nm_bin.is_dir():
            nm_bin_resolved = str(nm_bin.resolve())
            path_prefixes.append(nm_bin_resolved)
            if str(nm_bin) != nm_bin_resolved:
                path_prefixes.append(str(nm_bin))
        venv_bin = ws_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
        if venv_bin.is_dir():
            path_prefixes.append(str(venv_bin))
        if path_prefixes:
            cur_path = out.get("PATH") or ""
            prefix_joined = os.pathsep.join(path_prefixes)
            out["PATH"] = f"{prefix_joined}{os.pathsep}{cur_path}" if cur_path else prefix_joined
    out.pop("_AGY_SPAWN_COOKIE", None)
    out["NO_COLOR"] = "1"
    out.setdefault("TERM", "dumb")
    return out


def _custom_config_dir_args(env: Mapping[str, str] | None = None) -> list[str]:
    source = env if env is not None else os.environ
    explicit_cfg = _clean_env_str(source, "CC_AGY_CLI_CONFIG_DIR")
    if not explicit_cfg:
        return []
    cfg_path = Path(explicit_cfg).expanduser().resolve()
    return ["--gemini_dir", str(cfg_path.parent), "--app_data_dir", cfg_path.name]


_DENIED_TOOL_KINDS: dict[str, str] = {
    "run_command": "command",
    "run_terminal_command": "command",
    "bash": "command",
    "shell": "command",
    "read_file": "read",
    "view_file": "read",
    "write_file": "write",
    "write_to_file": "write",
    "edit_file": "edit",
    "replace_in_file": "edit",
    "replace_file_content": "edit",
}
_DENIED_NAME_KEYS = ("tool_name", "tool", "name", "action")
_DENIED_ARG_CONTAINER_KEYS = ("parameters", "args", "arguments", "input")
_DENIED_COMMAND_KEYS = ("CommandLine", "command", "command_line", "cmd")
_DENIED_PATH_KEYS = ("Path", "path", "file_path", "AbsolutePath", "TargetFile", "target_file")
_DENIED_WORKDIR_KEYS = ("Cwd", "cwd", "workdir", "working_directory")
_DENIED_REASON_KEYS = ("reason", "message", "policy", "description")
_DENIED_FIELD_MAX_CHARS = 2000


def _first_text(mapping: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        val = mapping.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()[:_DENIED_FIELD_MAX_CHARS]
    return ""


def normalize_denied_actions(
    raw_denied: Sequence[Any] | None,
    tool_calls: Sequence[Mapping[str, Any]],
    *,
    step_indexes: Sequence[int] | None = None,
    conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    """Turn `agy`'s raw `denied_actions` into compact records a Lead agent can act on.

    Each denied entry is matched to at most one recorded tool call, and each tool call is
    claimed at most once, so a turn that denies several tools keeps them apart. Matching
    tries the entry's step index, then its tool name (latest unclaimed call first), then
    falls back to the latest unclaimed call. `correlation` records which rule matched
    (`step_index`, `tool_name`, `position`, or `none`) so callers can judge confidence.
    File contents are never copied into the record.

    Args:
        raw_denied: The `denied_actions` list from the `result` frame.
        tool_calls: Tool calls recorded for the turn, each with `name` and `args`.
        step_indexes: Step index of each tool call, aligned with `tool_calls`.
        conversation_id: The `agy` conversation to resume after the user decides.

    Returns:
        One dict per denied entry with `tool`, `kind`, `correlation`, and, when known,
        `command`, `path`, `workdir`, `reason`, and `agy_conversation_id`.
    """
    records: list[dict[str, Any]] = []
    claimed: set[int] = set()
    aligned_steps = list(step_indexes) if step_indexes is not None and len(step_indexes) == len(tool_calls) else None

    for entry in raw_denied or []:
        fields: Mapping[str, Any] = entry if isinstance(entry, Mapping) else {}
        entry_tool = _first_text(fields, _DENIED_NAME_KEYS)
        raw_step = fields.get("step_index", fields.get("step"))
        entry_step = raw_step if isinstance(raw_step, int) and not isinstance(raw_step, bool) else None

        match_idx: int | None = None
        correlation = "none"
        if entry_step is not None and aligned_steps is not None:
            for idx, step in enumerate(aligned_steps):
                if idx not in claimed and step == entry_step and (
                    not entry_tool or str(tool_calls[idx].get("name") or "") == entry_tool
                ):
                    match_idx, correlation = idx, "step_index"
                    break
        if match_idx is None and entry_tool:
            for idx in range(len(tool_calls) - 1, -1, -1):
                if idx not in claimed and str(tool_calls[idx].get("name") or "") == entry_tool:
                    match_idx, correlation = idx, "tool_name"
                    break
        if match_idx is None and not entry_tool and entry_step is None:
            for idx in range(len(tool_calls) - 1, -1, -1):
                if idx not in claimed:
                    match_idx, correlation = idx, "position"
                    break

        merged_args: dict[str, Any] = {}
        tool_name = entry_tool
        if match_idx is not None:
            claimed.add(match_idx)
            matched_call = tool_calls[match_idx]
            tool_name = tool_name or str(matched_call.get("name") or "")
            call_args = matched_call.get("args")
            if isinstance(call_args, Mapping):
                merged_args.update(call_args)
        for container_key in _DENIED_ARG_CONTAINER_KEYS:
            container = fields.get(container_key)
            if isinstance(container, Mapping):
                merged_args.update(container)
        for key, val in fields.items():
            if isinstance(val, str) and key not in _DENIED_NAME_KEYS:
                merged_args.setdefault(key, val)

        reason = _first_text(fields, _DENIED_REASON_KEYS)
        if not reason and isinstance(entry, str) and entry.strip():
            reason = entry.strip()[:_DENIED_FIELD_MAX_CHARS]

        record: dict[str, Any] = {
            "tool": tool_name or "unknown",
            "kind": _DENIED_TOOL_KINDS.get(tool_name, "other"),
            "correlation": correlation,
        }
        for out_key, arg_keys in (
            ("command", _DENIED_COMMAND_KEYS),
            ("path", _DENIED_PATH_KEYS),
            ("workdir", _DENIED_WORKDIR_KEYS),
        ):
            text = _first_text(merged_args, arg_keys)
            if text:
                record[out_key] = text
        if reason:
            record["reason"] = reason
        if conversation_id:
            record["agy_conversation_id"] = conversation_id
        records.append(record)
    return records


def is_agy_auth_error(message: str) -> bool:
    """Return True if `message` indicates missing, expired, or unauthorized `agy` OAuth credentials."""
    lower = (message or "").lower()
    if any(qpat in lower for qpat in _QUOTA_OR_TRANSIENT_PATTERNS):
        return False
    if any(pat in lower for pat in _AUTH_ERROR_PATTERNS):
        return True
    return bool(_AUTH_CONTEXT_HTTP_RE.search(lower))


def _extract_available_models(error_text: str) -> list[str]:
    if "Available models:" not in error_text:
        return []
    after = error_text.split("Available models:", 1)[1]
    entries: list[str] = []
    for raw_line in after.splitlines():
        if not raw_line.strip():
            if entries:
                break
            continue
        if raw_line.startswith((" ", "\t", "-", "*", "•")):
            item = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", raw_line.strip()).strip("`'\"")
            if item:
                entries.append(item)
        elif entries:
            break
    return entries


def select_fallback_agy_model(
    requested_model: str,
    requested_effort: str | None,
    available_models: Sequence[str],
) -> str | None:
    """Select the closest Gemini model from `agy`'s `Available models:` catalog."""
    if not available_models:
        return None
    req_low = (requested_model or "").lower()
    want_flash = "flash" in req_low
    want_pro = ("argon" in req_low) or ("carbon" in req_low) or ("pro" in req_low) or not want_flash
    eff_low = (requested_effort or "").strip().lower()

    best_score: tuple[int, int, int, int, int] | None = None
    best_entry: str | None = None

    for entry in available_models:
        low = entry.lower()
        if "gemini" not in low:
            continue
        if "lite" in low and "lite" not in req_low:
            continue
        is_flash = "flash" in low
        is_pro_or_argon = ("argon" in low) or ("carbon" in low) or ("pro" in low)
        if want_flash and not is_flash:
            continue
        if want_pro and not want_flash and not is_pro_or_argon:
            continue

        ver_match = re.search(r"gemini[\s\-]*(\d+)(?:\.(\d+))?", low)
        major = int(ver_match.group(1)) if ver_match else 0
        minor = int(ver_match.group(2)) if (ver_match and ver_match.group(2)) else 0
        argon_bonus = 1 if (("argon" in req_low and "argon" in low) or ("carbon" in req_low and "carbon" in low)) else 0

        eff_bonus = 0
        if eff_low:
            if f"({eff_low})" in low or low.endswith(f"-{eff_low}"):
                eff_bonus = 2
        elif "(medium)" in low or low.endswith("-medium") or "(high)" in low or low.endswith("-high"):
            eff_bonus = 1

        base_clean_bonus = 0 if _model_has_explicit_effort(entry) else 1
        score = (argon_bonus, major, minor, eff_bonus, base_clean_bonus)
        if best_score is None or score > best_score:
            best_score = score
            best_entry = entry

    return best_entry


def _close_proc_pipes_safely(
    proc: subprocess.Popen[Any],
    *,
    stdout_thread: threading.Thread | None = None,
    stderr_thread: threading.Thread | None = None,
) -> None:
    """Close subprocess pipes only when no reader thread is blocked in `.read()`/`.readline()`.

    Calling `BufferedReader.close()` from another thread while a reader thread holds the buffer's
    internal C lock deadlocks Python indefinitely if an unkilled child still holds the write fd.
    """
    if proc.stdin is not None:
        try:
            proc.stdin.close()
        except Exception:
            pass
    if proc.stdout is not None and (stdout_thread is None or not stdout_thread.is_alive()):
        try:
            proc.stdout.close()
        except Exception:
            pass
    if proc.stderr is not None and (stderr_thread is None or not stderr_thread.is_alive()):
        try:
            proc.stderr.close()
        except Exception:
            pass


def _snapshot_pipe_inodes(proc: subprocess.Popen[Any]) -> set[tuple[int, int]]:
    inodes: set[tuple[int, int]] = set()
    for pipe in (proc.stdout, proc.stderr):
        if pipe is None:
            continue
        try:
            st = os.fstat(pipe.fileno())
            inodes.add((st.st_dev, st.st_ino))
        except Exception:
            pass
    return inodes


def _find_pipe_or_cookie_survivors(
    *,
    pipe_inodes: set[tuple[int, int]] | None,
    spawn_cookie: str,
    exclude_pid: int,
) -> set[int]:
    """Locate detached (`setsid`) descendants that still hold our stdout/stderr pipes or spawn cookie."""
    found: set[int] = set()
    if os.name == "nt":
        return found
    from server.agy_runner import _proc_fs_available, _ps_pid_matches_save_dir, _ps_process_table

    cookie_token = f"_AGY_SPAWN_COOKIE={spawn_cookie}" if spawn_cookie else ""
    if _proc_fs_available():
        try:
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                if pid == exclude_pid:
                    continue
                if pipe_inodes:
                    matched_pipe = False
                    for fd_num in (1, 2):
                        try:
                            st = os.stat(f"/proc/{pid}/fd/{fd_num}")
                            if (st.st_dev, st.st_ino) in pipe_inodes:
                                found.add(pid)
                                matched_pipe = True
                                break
                        except OSError:
                            continue
                    if matched_pipe:
                        continue
                if cookie_token:
                    try:
                        raw_env = (entry / "environ").read_bytes()
                        if cookie_token.encode("utf-8") in raw_env:
                            found.add(pid)
                    except OSError:
                        continue
        except OSError:
            pass
        return found

    if cookie_token:
        for pid, (_ppid, state, args_s) in _ps_process_table(include_args=True).items():
            if pid == exclude_pid or state.upper().startswith("Z"):
                continue
            if _ps_pid_matches_save_dir(pid, cookie_token, args_s):
                found.add(pid)
    return found


def _scan_process_tree(
    root_pid: int,
    *,
    pipe_inodes: set[tuple[int, int]] | None,
    spawn_cookie: str,
    exclude_pid: int,
) -> set[int]:
    """Find `root_pid`'s descendants plus processes holding our pipes or spawn cookie.

    On Linux this reads `/proc` once. Elsewhere it falls back to the `ps`-based helpers.
    """
    from server.agy_runner import _collect_descendant_pids, _proc_fs_available, _read_proc_ppid_and_state

    if not _proc_fs_available():
        found = set(_collect_descendant_pids({root_pid}, exclude_pid=exclude_pid))
        found.update(
            _find_pipe_or_cookie_survivors(pipe_inodes=pipe_inodes, spawn_cookie=spawn_cookie, exclude_pid=exclude_pid)
        )
        return found

    cookie = f"_AGY_SPAWN_COOKIE={spawn_cookie}".encode("utf-8") if spawn_cookie else b""
    children: dict[int, list[int]] = {}
    found = set()
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == exclude_pid:
            continue
        ppid, state = _read_proc_ppid_and_state(pid)
        if ppid is None or state == "Z":
            continue
        children.setdefault(ppid, []).append(pid)
        if pipe_inodes:
            for fd_num in (1, 2):
                try:
                    st = os.stat(f"/proc/{pid}/fd/{fd_num}")
                except OSError:
                    continue
                if (st.st_dev, st.st_ino) in pipe_inodes:
                    found.add(pid)
                    break
        if cookie and pid not in found:
            try:
                if cookie in (entry / "environ").read_bytes():
                    found.add(pid)
            except OSError:
                pass
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, []):
            if child not in found:
                found.add(child)
                frontier.append(child)
    return found


def _terminate_process_tree(
    proc: subprocess.Popen[Any],
    *,
    spawn_cookie: str = "",
    pipe_inodes: set[tuple[int, int]] | None = None,
) -> None:
    """Terminate `proc`, its POSIX process group, and any detached `setsid` descendants.

    The process table is scanned once per call, and a tree that was already fully reaped
    is not scanned again on later calls for the same process.
    """
    if getattr(proc, "_cc_agy_tree_reaped", False) and proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            from server.agy_runner import _pid_is_alive_non_zombie

            my_pid = os.getpid()
            extra_pids = _scan_process_tree(
                proc.pid, pipe_inodes=pipe_inodes, spawn_cookie=spawn_cookie, exclude_pid=my_pid
            )
            extra_pids.discard(proc.pid)
            extra_pids.discard(my_pid)

            pgid_alive = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pgid_alive = False

            for ep in list(extra_pids):
                try:
                    os.killpg(ep, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    os.kill(ep, signal.SIGTERM)
                except OSError:
                    pass

            if proc.poll() is None:
                try:
                    proc.wait(timeout=0.4)
                except subprocess.TimeoutExpired:
                    pass
            elif pgid_alive or extra_pids:
                deadline_term = time.monotonic() + 0.25
                while time.monotonic() < deadline_term:
                    if not any(_pid_is_alive_non_zombie(ep) for ep in extra_pids):
                        break
                    time.sleep(0.02)

            if pgid_alive or proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
            for ep in list(extra_pids):
                if _pid_is_alive_non_zombie(ep):
                    try:
                        os.killpg(ep, signal.SIGKILL)
                    except OSError:
                        pass
                    try:
                        os.kill(ep, signal.SIGKILL)
                    except OSError:
                        pass
            if proc.poll() is None:
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            if proc.poll() is not None and not any(_pid_is_alive_non_zombie(ep) for ep in extra_pids):
                try:
                    proc._cc_agy_tree_reaped = True  # type: ignore[attr-defined]
                except AttributeError:
                    pass
        else:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
    except OSError:
        pass


def _is_read_only_worker_request(request: Mapping[str, Any]) -> bool:
    from server.agy_runner import is_read_only_tool_set

    return is_read_only_tool_set(
        request.get("enabled_tools"),
        request.get("disabled_tools"),
    )


def _format_cli_worker_prompt(
    prompt: str,
    *,
    system_prompt: str | None,
    allowed_paths: Sequence[str] | None,
    read_only: bool,
    is_followup: bool,
) -> str:
    if is_followup:
        return prompt
    guard_lines = [
        "[Execution Guardrails]",
        "- Stay strictly inside the current workspace directory; never run `git commit`, `git reset`, `git checkout`, or `git push`.",
    ]
    if read_only:
        guard_lines.append("- Read-only task: do NOT create, modify, or delete any files.")
    elif allowed_paths:
        joined = ", ".join(str(p) for p in allowed_paths[:20])
        guard_lines.append(f"- Allowed file scope: {joined}. Do not modify files outside this scope.")
    guard_lines.append("- Keep your final summary concise (under 250 words) listing files changed and checks run.")
    header = "\n".join(guard_lines)
    if system_prompt and system_prompt.strip():
        return f"{system_prompt.strip()}\n\n{header}\n\n{prompt}"
    return f"{header}\n\n{prompt}"


def _format_auth_remediation(raw_err: str, agy_bin: str, env: Mapping[str, str] | None = None) -> str:
    cfg_dir = agy_cli_config_dir(env)
    return (
        f"Gemini Enterprise `agy` CLI authentication failed: {raw_err.strip()}. "
        f"Run `{agy_bin}` once in a terminal to complete Google sign-in (credentials stored in `{cfg_dir}`), "
        "or set `ANTIGRAVITY_OAUTH_TOKEN`."
    )


def _merge_usage(acc: dict[str, int], turn_usage: Mapping[str, int]) -> None:
    for k in _USAGE_KEYS:
        acc[k] = acc.get(k, 0) + int(turn_usage.get(k, 0) or 0)


def _sum_step_usages(step_usages: Mapping[int, dict[str, int]]) -> dict[str, int]:
    acc = convert_agy_usage_to_vertex(None)
    for u in step_usages.values():
        _merge_usage(acc, u)
    return acc


def _start_stdin_writer(proc: subprocess.Popen[bytes], prompt_text: str) -> threading.Thread:
    """Spawn a daemon thread that writes `prompt_text` (UTF-8) to `proc.stdin` and closes the pipe."""

    def _write_stdin() -> None:
        if proc.stdin is None:
            return
        try:
            proc.stdin.write(prompt_text.encode("utf-8"))
            proc.stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    thread = threading.Thread(target=_write_stdin, daemon=True)
    thread.start()
    return thread


def _start_pipe_chunk_drainer(stream: Any, chunks: list[bytes]) -> threading.Thread:
    """Spawn a daemon thread that drains binary chunks from `stream` into `chunks` until EOF."""

    def _drain() -> None:
        assert stream is not None
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)
        except (OSError, ValueError):
            pass

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    return thread


def _compute_conversation_turn_delta(
    *,
    turn_usage: Mapping[str, int],
    step_usage_sum: Mapping[str, int],
    cur_num_turns: int,
    base_usage: Mapping[str, int] | None,
    base_turns: int,
    had_resumed_conversation: bool,
) -> tuple[dict[str, int], dict[str, int], int, bool]:
    """Compute `(delta_usage, updated_cum_usage, updated_turns, unknown_cumulative_baseline)`.

    `agy --conversation` builds `JSONOutput` from `session.Snapshot()` across all trajectory steps
    (`num_turns > 1`), so `turn_usage` is cumulative when `cur_num_turns > base_turns` and every
    counter is `>= base_usage`. If any counter shrinks (e.g. a mock or CLI build reporting per-turn
    numbers despite `num_turns > 1`), treat `turn_usage` as already per-turn so Turn-1 tokens are
    never lost.
    """
    clean_turn = {k: max(0, int(turn_usage.get(k, 0) or 0)) for k in _USAGE_KEYS}
    eff_turns = max(cur_num_turns, base_turns + 1 if had_resumed_conversation else 1)

    if base_usage is not None:
        clean_base = {k: max(0, int(base_usage.get(k, 0) or 0)) for k in _USAGE_KEYS}
        is_monotonic_cumulative = (
            cur_num_turns > max(0, base_turns)
            and cur_num_turns > 1
            and all(clean_turn[k] >= clean_base[k] for k in _USAGE_KEYS)
        )
        if is_monotonic_cumulative:
            delta = {k: max(0, clean_turn[k] - clean_base[k]) for k in _USAGE_KEYS}
            delta["total_token_count"] = (
                delta["prompt_token_count"] + delta["candidates_token_count"] + delta["thoughts_token_count"]
            )
            return delta, dict(clean_turn), eff_turns, False

        cum = dict(clean_base)
        _merge_usage(cum, clean_turn)
        return dict(clean_turn), cum, eff_turns, False

    if had_resumed_conversation and cur_num_turns > 1:
        step_tot = int(step_usage_sum.get("total_token_count", 0) or 0)
        if step_tot > 0:
            delta_from_steps = {k: max(0, int(step_usage_sum.get(k, 0) or 0)) for k in _USAGE_KEYS}
            return delta_from_steps, dict(clean_turn), eff_turns, False
        return dict(clean_turn), dict(clean_turn), eff_turns, True

    return dict(clean_turn), dict(clean_turn), eff_turns, False


def _run_stream_json_turn(
    *,
    agy_bin: str,
    prompt_text: str,
    cwd: str,
    model: str,
    effort: str | None,
    mode: str,
    skip_permissions: bool,
    conversation_id: str | None,
    timeout_seconds: float,
    register_cancel: Callable[[Callable[[], None]], None] | None,
    max_tool_calls: int | None = None,
    max_total_tokens: int | None = None,
    prior_tool_calls_count: int = 0,
    prior_tokens_count: int = 0,
    env: Mapping[str, str] | None = None,
    sandbox_prefix: Sequence[str] = (),
) -> dict[str, Any]:
    if not prompt_text or not prompt_text.strip():
        raise ValueError("Worker prompt cannot be empty.")
    timeout_sec = max(0.05, float(timeout_seconds))
    print_timeout_sec = max(5, int(round(timeout_sec)) + 15)

    cmd: list[str] = [
        *sandbox_prefix,
        agy_bin,
        *_custom_config_dir_args(env),
        "--output-format",
        "stream-json",
        "--disable-slash-commands",
        "--mode",
        mode,
        "--print-timeout",
        f"{print_timeout_sec}s",
    ]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["--effort", effort])
    if conversation_id:
        cmd.extend(["--conversation", conversation_id])

    spawn_cookie = secrets.token_hex(16)
    sub_env = build_agy_subprocess_env(env, workspace_dir=cwd)
    sub_env["_AGY_SPAWN_COOKIE"] = spawn_cookie

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name != "nt"),
        env=sub_env,
    )
    pipe_inodes = _snapshot_pipe_inodes(proc)

    cancelled_or_timed_out = threading.Event()
    timed_out_flag = threading.Event()
    budget_stop_reason: list[str] = []

    def _cancel_proc() -> None:
        cancelled_or_timed_out.set()
        _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)

    if callable(register_cancel):
        register_cancel(_cancel_proc)

    def _on_timeout() -> None:
        timed_out_flag.set()
        _cancel_proc()

    timer = threading.Timer(timeout_sec, _on_timeout)
    timer.daemon = True
    timer.start()

    stderr_chunks: list[bytes] = []
    line_q: queue.Queue[bytes | None] = queue.Queue()

    def _drain_stdout_lines() -> None:
        assert proc.stdout is not None
        try:
            while True:
                ln = proc.stdout.readline()
                if not ln:
                    break
                line_q.put(ln)
        except (OSError, ValueError):
            pass
        finally:
            line_q.put(None)

    t_out = threading.Thread(target=_drain_stdout_lines, daemon=True)
    t_out.start()
    t_err = _start_pipe_chunk_drainer(proc.stderr, stderr_chunks)
    t_in = _start_stdin_writer(proc, prompt_text)

    resolved_conv_id = conversation_id or ""
    step_tool_calls: list[dict[str, Any]] = []
    step_tool_indexes: list[int] = []
    seen_step_tools: dict[tuple[int, str], int] = {}
    step_usages: dict[int, dict[str, int]] = {}
    fallback_tool_calls: list[dict[str, Any]] = []
    thoughts: list[str] = []
    delta_text_parts: list[str] = []
    assistant_text_parts: list[str] = []
    final_result_obj: dict[str, Any] | None = None
    got_result_frame = False
    auto_step_seq = -1
    deadline = time.monotonic() + timeout_sec

    def _check_live_budgets() -> bool:
        if max_tool_calls is not None and max_tool_calls > 0:
            active_calls = len(step_tool_calls) if step_tool_calls else len(fallback_tool_calls)
            live_calls = prior_tool_calls_count + active_calls
            if live_calls > max_tool_calls:
                budget_stop_reason.append("MAX_TOOL_CALLS_EXCEEDED")
                _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
                return True
        if max_total_tokens is not None and max_total_tokens > 0 and step_usages:
            live_toks = prior_tokens_count + _sum_step_usages(step_usages)["total_token_count"]
            if live_toks > max_total_tokens:
                budget_stop_reason.append("MAX_TOTAL_TOKENS_EXCEEDED")
                _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
                return True
        return False

    def _process_ndjson_line(raw_line: bytes, *, allow_result: bool = True) -> str:
        """Parse one NDJSON line; return `'result'`, `'budget'`, or `''`."""
        nonlocal resolved_conv_id, auto_step_seq, got_result_frame, final_result_obj
        line_str = raw_line.decode("utf-8", errors="replace").strip()
        if not line_str or not line_str.startswith("{"):
            return ""
        try:
            evt = json.loads(line_str)
        except ValueError:
            return ""
        if not isinstance(evt, dict):
            return ""

        evt_kind = str(evt.get("event") or evt.get("type") or "").strip().lower()
        if isinstance(evt.get("conversation_id"), str) and evt["conversation_id"]:
            resolved_conv_id = evt["conversation_id"]

        if evt_kind == "step_update":
            su = evt.get("step_update")
            if isinstance(su, dict):
                if isinstance(su.get("conversation_id"), str) and su["conversation_id"]:
                    resolved_conv_id = su["conversation_id"]
                if isinstance(su.get("text_delta"), str) and su["text_delta"]:
                    delta_text_parts.append(su["text_delta"])
                raw_s_idx = su.get("step_index")
                if isinstance(raw_s_idx, int) and not isinstance(raw_s_idx, bool) and raw_s_idx > 0:
                    s_idx = raw_s_idx
                else:
                    s_idx = auto_step_seq
                    auto_step_seq -= 1
                if isinstance(su.get("usage"), dict):
                    step_usages[s_idx] = convert_agy_usage_to_vertex(su["usage"])
                t_info = su.get("tool_info")
                t_name = str(
                    (t_info.get("name") if isinstance(t_info, dict) else None)
                    or su.get("tool_name")
                    or ""
                ).strip()
                if t_name:
                    params = t_info.get("parameters") if isinstance(t_info, dict) else None
                    parsed_args = dict(params) if isinstance(params, dict) else {}
                    if s_idx > 0 and (s_idx, t_name) in seen_step_tools:
                        if parsed_args:
                            step_tool_calls[seen_step_tools[(s_idx, t_name)]]["args"] = parsed_args
                    else:
                        step_tool_calls.append({"name": t_name, "args": parsed_args})
                        step_tool_indexes.append(s_idx)
                        if s_idx > 0:
                            seen_step_tools[(s_idx, t_name)] = len(step_tool_calls) - 1
                if allow_result and _check_live_budgets():
                    return "budget"
        elif evt_kind == "assistant":
            msg = evt.get("message")
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "thinking" and isinstance(block.get("thinking"), str):
                        thoughts.append(block["thinking"])
                    elif btype == "tool_use" and isinstance(block.get("name"), str):
                        inp = block.get("input")
                        fallback_tool_calls.append(
                            {"name": block["name"], "args": dict(inp) if isinstance(inp, dict) else {}}
                        )
                    elif btype == "text" and isinstance(block.get("text"), str):
                        assistant_text_parts.append(block["text"])
                if allow_result and _check_live_budgets():
                    return "budget"
        elif evt_kind == "result" and allow_result:
            got_result_frame = True
            res_payload = evt.get("result")
            if isinstance(res_payload, dict):
                final_result_obj = res_payload
                if isinstance(res_payload.get("conversation_id"), str) and res_payload["conversation_id"]:
                    resolved_conv_id = res_payload["conversation_id"]
            else:
                final_result_obj = evt
            return "result"
        return ""

    try:
        while not cancelled_or_timed_out.is_set() and not budget_stop_reason:
            rem = deadline - time.monotonic()
            if rem <= 0:
                timed_out_flag.set()
                cancelled_or_timed_out.set()
                break
            poll_wait = 0.05 if proc.poll() is not None else min(0.1, rem)
            try:
                raw_line = line_q.get(timeout=poll_wait)
            except queue.Empty:
                if proc.poll() is not None and line_q.empty():
                    # Leader already exited; only orphaned children can still hold stdout open.
                    break
                continue
            if raw_line is None:
                break

            action = _process_ndjson_line(raw_line, allow_result=True)
            if action in ("result", "budget"):
                # `result` is the terminal NDJSON frame; break immediately and give `agy` a
                # graceful window to flush conversation state before cleaning up descendants.
                break

        if got_result_frame and not cancelled_or_timed_out.is_set() and not budget_stop_reason:
            timer.cancel()
            post_result_grace = min(8.0, max(1.0, deadline - time.monotonic()))
            try:
                proc.wait(timeout=post_result_grace)
            except subprocess.TimeoutExpired:
                pass
        elif not cancelled_or_timed_out.is_set() and not budget_stop_reason:
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
    finally:
        timer.cancel()
        _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
        t_in.join(timeout=0.3)
        t_out.join(timeout=0.3)
        t_err.join(timeout=0.3)
        _close_proc_pipes_safely(proc, stdout_thread=t_out, stderr_thread=t_err)

    while True:
        try:
            rem_line = line_q.get_nowait()
        except queue.Empty:
            break
        if rem_line is not None:
            _process_ndjson_line(rem_line, allow_result=not timed_out_flag.is_set())

    stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
    tool_calls = step_tool_calls if step_tool_calls else fallback_tool_calls
    step_usage_sum = _sum_step_usages(step_usages)

    if isinstance(final_result_obj, dict) and isinstance(final_result_obj.get("usage"), dict):
        vertex_usage = convert_agy_usage_to_vertex(final_result_obj["usage"])
        usage_from_result_snapshot = True
    elif step_usages:
        vertex_usage = step_usage_sum
        usage_from_result_snapshot = False
    else:
        vertex_usage = convert_agy_usage_to_vertex(None)
        usage_from_result_snapshot = False

    if timed_out_flag.is_set() and not budget_stop_reason and not got_result_frame:
        err = TimeoutError(f"Antigravity CLI worker timed out after {timeout_sec:.1f}s")
        err.partial_usage = vertex_usage  # type: ignore[attr-defined]
        err.partial_usage_from_snapshot = usage_from_result_snapshot  # type: ignore[attr-defined]
        err.partial_step_usage_sum = step_usage_sum  # type: ignore[attr-defined]
        err.partial_tool_calls = tool_calls  # type: ignore[attr-defined]
        err.partial_thoughts = thoughts  # type: ignore[attr-defined]
        err.partial_containment = "agy_cli"  # type: ignore[attr-defined]
        err.partial_agy_conversation_id = resolved_conv_id or None  # type: ignore[attr-defined]
        raise err

    status_str = ""
    err_str = ""
    response_text = ""
    denied_actions: list[Any] = []
    num_turns = 0
    if isinstance(final_result_obj, dict):
        num_turns = int(final_result_obj.get("num_turns") or 0)
        status_str = str(final_result_obj.get("status") or final_result_obj.get("subtype") or "").upper()
        if final_result_obj.get("is_error") is True and not status_str:
            status_str = "ERROR"
        raw_err = final_result_obj.get("error")
        if isinstance(raw_err, str) and raw_err.strip():
            err_str = raw_err.strip()
        elif isinstance(final_result_obj.get("errors"), list) and final_result_obj["errors"]:
            err_str = "; ".join(str(x) for x in final_result_obj["errors"])
        if isinstance(final_result_obj.get("response"), str):
            response_text = final_result_obj["response"]
        elif isinstance(final_result_obj.get("result"), str):
            response_text = final_result_obj["result"]
        if isinstance(final_result_obj.get("denied_actions"), list):
            denied_actions = final_result_obj["denied_actions"]

    streamed_text = "".join(delta_text_parts) if delta_text_parts else "".join(assistant_text_parts)
    if not response_text and streamed_text:
        response_text = streamed_text

    if budget_stop_reason:
        return {
            "ok": True,
            "text": response_text,
            "usage": vertex_usage,
            "step_usage_sum": step_usage_sum,
            "usage_from_snapshot": usage_from_result_snapshot,
            "num_turns": num_turns,
            "tool_calls": tool_calls,
            "thoughts": thoughts,
            "stop_reason": budget_stop_reason[0],
            "conversation_id": resolved_conv_id,
        }

    missing_envelope = final_result_obj is None and not streamed_text
    has_non_error_result = got_result_frame and status_str != "ERROR" and not err_str
    failed = (status_str == "ERROR") or bool(err_str) or missing_envelope or (proc.returncode != 0 and not has_non_error_result)
    if failed:
        combined_err = (
            err_str
            or stderr_text
            or (
                "Antigravity CLI exited without emitting a result envelope"
                if missing_envelope and proc.returncode == 0
                else f"Antigravity CLI exited with code {proc.returncode}"
            )
        )
        return {
            "ok": False,
            "error": combined_err,
            "returncode": proc.returncode,
            "conversation_id": resolved_conv_id,
            "num_turns": num_turns,
            "usage": vertex_usage,
            "step_usage_sum": step_usage_sum,
            "usage_from_snapshot": usage_from_result_snapshot,
            "tool_calls": tool_calls,
            "thoughts": thoughts,
            "text": response_text,
        }

    stop_reason = "DENIED_ACTIONS" if denied_actions else "STOP"
    if stop_reason == "STOP":
        if max_tool_calls is not None and max_tool_calls > 0 and (prior_tool_calls_count + len(tool_calls)) > max_tool_calls:
            stop_reason = "MAX_TOOL_CALLS_EXCEEDED"
    return {
        "ok": True,
        "text": response_text,
        "usage": vertex_usage,
        "step_usage_sum": step_usage_sum,
        "usage_from_snapshot": usage_from_result_snapshot,
        "num_turns": num_turns,
        "tool_calls": tool_calls,
        "thoughts": thoughts,
        "stop_reason": stop_reason,
        "conversation_id": resolved_conv_id,
        "denied_actions": normalize_denied_actions(
            denied_actions,
            tool_calls,
            step_indexes=step_tool_indexes if step_tool_calls else None,
            conversation_id=resolved_conv_id or None,
        ),
    }


def _resolve_cached_model_for_cli(
    agy_bin: str,
    requested_model: str,
    effort: str | None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str | None, str | None]:
    """Pick the model to run, checking the `agy` catalog and Central Bifrost policy at 0 tokens.

    When the cached `agy models` catalog is known and does not offer the requested model,
    the closest offered model (that Bifrost policy also allows) is chosen up front, so the
    mismatch does not cost a failed first turn. An unknown catalog changes nothing.

    Raises:
        BifrostPolicyError: If Bifrost `/agy` governance is on and denies the model, or
            routes `policy-routed` to a model this seat does not offer.
    """
    from server.agy_state import catalog_offers, load_agy_model_catalog
    from server.bifrost_policy import POLICY_ROUTED_MODEL, BifrostPolicyError, enforce_agy_model_policy

    cfg_key = str(agy_cli_config_dir(env))
    cache_key = (agy_bin, cfg_key, requested_model, effort or "")
    with _MODEL_FALLBACK_LOCK:
        cached = _MODEL_FALLBACK_CACHE.get(cache_key)
    if cached:
        model = cached
        eff_after = None if _model_has_explicit_effort(cached) else effort
        note: str | None = f"Mapped requested model {requested_model!r} to offered Antigravity CLI model {cached!r}."
    else:
        model, eff_after, note = requested_model, effort, None

    catalog = load_agy_model_catalog(agy_bin, env)
    is_policy_routed = model.rsplit("/", 1)[-1] == POLICY_ROUTED_MODEL
    if catalog and not is_policy_routed and not catalog_offers(model, catalog):
        chosen = select_fallback_agy_model(model, eff_after, _filter_models_by_bifrost_policy(list(catalog), env))
        if chosen and chosen != model:
            with _MODEL_FALLBACK_LOCK:
                _MODEL_FALLBACK_CACHE[cache_key] = chosen
            note = f"Mapped requested model {requested_model!r} to offered Antigravity CLI model {chosen!r}."
            model = chosen
            eff_after = None if _model_has_explicit_effort(chosen) else eff_after

    verdict = enforce_agy_model_policy(model, eff_after, env)
    if verdict["policy_routed"]:
        model = str(verdict["model"])
        if catalog and not catalog_offers(model, catalog):
            raise BifrostPolicyError(
                f"Central Bifrost policy routed 'policy-routed' to '{model}', "
                "but `agy models` does not offer that model to this seat."
            )
        eff_after = None if _model_has_explicit_effort(model) else eff_after
        note = f"Central Bifrost policy routed 'policy-routed' to Antigravity CLI model {model!r}."
    return model, eff_after, note


def _filter_models_by_bifrost_policy(models: list[str], env: Mapping[str, str] | None) -> list[str]:
    """Keep only fallback models that Central Bifrost policy allows (all of them when it is off)."""
    from server.bifrost_policy import fetch_allowed_agy_models, is_model_allowed, resolve_bifrost_policy_config

    config = resolve_bifrost_policy_config(env)
    if config is None:
        return models
    allowed = fetch_allowed_agy_models(config)
    return [m for m in models if is_model_allowed(m, None, allowed)]


def _maybe_retry_model_error(
    *,
    err_text: str,
    agy_bin: str,
    requested_model: str,
    effort: str | None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str | None, str] | None:
    lower = err_text.lower()
    available = _filter_models_by_bifrost_policy(_extract_available_models(err_text), env)
    if available and ("not recognized as a known model" in lower or "invalid model selection" in lower):
        chosen = select_fallback_agy_model(requested_model, effort, available)
        if chosen and chosen != requested_model:
            cfg_key = str(agy_cli_config_dir(env))
            with _MODEL_FALLBACK_LOCK:
                _MODEL_FALLBACK_CACHE[(agy_bin, cfg_key, requested_model, effort or "")] = chosen
            eff_after = None if _model_has_explicit_effort(chosen) else effort
            note = f"Mapped requested model {requested_model!r} to offered Antigravity CLI model {chosen!r}."
            return chosen, eff_after, note

    if effort and ("has no " in lower and "effort" in lower or "effort is not supported" in lower):
        note = f"Omitted unsupported --effort={effort!r} for model {requested_model!r}."
        return requested_model, None, note

    return None


def _is_missing_conversation_error(err_text: str) -> bool:
    low = (err_text or "").lower()
    return "conversation" in low and any(
        phrase in low
        for phrase in ("not found", "does not exist", "unknown conversation", "no such conversation", "failed to load")
    )


def run_agy_cli_worker(request: dict[str, Any]) -> dict[str, Any]:
    """Execute an agentic worker job through external `agy --output-format stream-json`.

    Write workers are confined by `server.worker_sandbox` (see `CC_AGY_WORKER_SANDBOX`).
    One deadline covers the whole job, so retries share `timeout_seconds` instead of each
    getting the full amount. Tokens, tool calls, and thoughts from failed attempts that are
    retried still count toward usage, receipts, and `max_total_tokens`.
    """
    from server.worker_sandbox import plan_worker_sandbox

    env = request.get("env") if isinstance(request.get("env"), Mapping) else None
    ensure_enterprise_seat_ready(env)
    agy_bin = _resolve_or_bootstrap_agy_bin(env)
    cfg_key = str(agy_cli_config_dir(env))

    workspace_dir = str(request["workspace_dir"])
    requested_model = str(request.get("model") or "gemini-flash-next")
    thinking_level = request.get("thinking_level")
    effort = map_thinking_level_to_effort(thinking_level, requested_model)
    active_model, active_effort, reroute_note = _resolve_cached_model_for_cli(
        agy_bin, requested_model, effort, env=env
    )

    conv_id: str | None = (
        str(request["agy_conversation_id"]).strip()
        if request.get("agy_conversation_id")
        else None
    )

    read_only = _is_read_only_worker_request(request)
    mode = "plan" if read_only else "auto"
    seat = resolve_enterprise_adc_seat(env)
    sandbox = plan_worker_sandbox(
        workspace_dir=workspace_dir,
        config_dir=agy_cli_config_dir(env),
        env=env,
        adc_path=seat["adc_path"] if seat["active"] else None,
        readonly_paths=(Path(agy_bin).resolve().parent,),
        read_only_workspace=read_only,
        data_dir=request.get("data_dir"),
        conversation_id=conv_id,
    )
    skip_permissions = bool(not read_only and sandbox and sandbox.skip_permissions)
    sandbox_prefix = sandbox.command_prefix if sandbox else ()

    prompts: list[str] = list(request.get("prompts") or [])
    if not prompts and request.get("prompt"):
        prompts = [str(request["prompt"])]
    if not prompts:
        raise ValueError("Worker prompt cannot be empty.")

    timeout_seconds = float(request.get("timeout_seconds") or 600.0)
    job_deadline = time.monotonic() + timeout_seconds
    register_cancel = request.get("register_cancel")
    max_tool_calls = int(request["max_tool_calls"]) if request.get("max_tool_calls") else None
    max_total_tokens = int(request["max_total_tokens"]) if request.get("max_total_tokens") else None

    base_conv_usage: dict[str, int] | None = None
    base_conv_turns = 0
    raw_prior_conv_u = request.get("agy_prior_conversation_usage")
    if raw_prior_conv_u is None:
        raw_prior_conv_u = request.get("prior_agy_usage")
    if isinstance(raw_prior_conv_u, Mapping) and any(int(raw_prior_conv_u.get(k, 0) or 0) > 0 for k in _USAGE_KEYS):
        base_conv_usage = {k: max(0, int(raw_prior_conv_u.get(k, 0) or 0)) for k in _USAGE_KEYS}
        base_conv_turns = max(1, int(request.get("agy_prior_conversation_turns") or 1))
    elif conv_id:
        with _CONV_LEDGER_LOCK:
            ledger_hit = _CONVERSATION_USAGE_LEDGER.get((cfg_key, conv_id))
        if ledger_hit is not None:
            base_conv_turns, ledger_u = ledger_hit
            base_conv_usage = dict(ledger_u)

    agg_usage = convert_agy_usage_to_vertex(None)
    all_tool_calls: list[dict[str, Any]] = []
    all_thoughts: list[str] = []
    last_text = ""
    stop_reason = "STOP"
    denied_actions: list[dict[str, Any]] = []
    unknown_cumulative_baseline = False

    attempts_started = [0]

    def _run_attempt(prompt_text: str, conversation: str | None) -> dict[str, Any]:
        remaining = job_deadline - time.monotonic()
        # The first attempt always runs; later ones need at least a second left to be useful.
        floor = _MIN_ATTEMPT_SECONDS if attempts_started[0] else 0.0
        if remaining <= floor:
            raise TimeoutError(
                f"Antigravity CLI worker used its whole {timeout_seconds:.0f}s timeout across attempts"
            )
        attempts_started[0] += 1
        return _run_stream_json_turn(
            agy_bin=agy_bin,
            prompt_text=prompt_text,
            cwd=workspace_dir,
            model=active_model,
            effort=active_effort,
            mode=mode,
            skip_permissions=skip_permissions,
            conversation_id=conversation,
            timeout_seconds=remaining,
            register_cancel=register_cancel if callable(register_cancel) else None,
            max_tool_calls=max_tool_calls,
            max_total_tokens=max_total_tokens,
            prior_tool_calls_count=len(all_tool_calls),
            prior_tokens_count=agg_usage["total_token_count"],
            env=env,
            sandbox_prefix=sandbox_prefix,
        )

    def _count_failed_attempt(failed: Mapping[str, Any], resumed: bool) -> None:
        """Add a failed attempt's tokens, tool calls, and thoughts before it is retried."""
        raw_usage = failed.get("usage") or {}
        if failed.get("usage_from_snapshot") and resumed and base_conv_usage is not None:
            attempt_usage, _, _, _ = _compute_conversation_turn_delta(
                turn_usage=raw_usage,
                step_usage_sum=failed.get("step_usage_sum") or {},
                cur_num_turns=int(failed.get("num_turns") or 0),
                base_usage=base_conv_usage,
                base_turns=base_conv_turns,
                had_resumed_conversation=True,
            )
        else:
            attempt_usage = {k: max(0, int(raw_usage.get(k, 0) or 0)) for k in _USAGE_KEYS}
        _merge_usage(agg_usage, attempt_usage)
        all_tool_calls.extend(failed.get("tool_calls") or [])
        all_thoughts.extend(failed.get("thoughts") or [])

    for turn_idx, raw_prompt in enumerate(prompts):
        had_resumed = bool(conv_id)
        formatted_prompt = _format_cli_worker_prompt(
            raw_prompt,
            system_prompt=request.get("system_prompt"),
            allowed_paths=request.get("allowed_paths"),
            read_only=read_only,
            is_followup=(turn_idx > 0 or had_resumed),
        )
        try:
            turn_out = _run_attempt(formatted_prompt, conv_id)

            if not turn_out["ok"] and turn_idx == 0 and conv_id and _is_missing_conversation_error(turn_out["error"]):
                _count_failed_attempt(turn_out, had_resumed)
                conv_id = None
                had_resumed = False
                base_conv_usage = None
                base_conv_turns = 0
                formatted_prompt = _format_cli_worker_prompt(
                    raw_prompt,
                    system_prompt=request.get("system_prompt"),
                    allowed_paths=request.get("allowed_paths"),
                    read_only=read_only,
                    is_followup=False,
                )
                turn_out = _run_attempt(formatted_prompt, None)

            seen_worker_states: set[tuple[str, str | None]] = {(active_model, active_effort)}
            for _retry_idx in range(2):
                if turn_out["ok"]:
                    break
                retry_spec = _maybe_retry_model_error(
                    err_text=turn_out["error"],
                    agy_bin=agy_bin,
                    requested_model=active_model,
                    effort=active_effort,
                    env=env,
                )
                if retry_spec is None:
                    break
                next_model, next_effort, reroute_note = retry_spec
                if (next_model, next_effort) in seen_worker_states:
                    break
                seen_worker_states.add((next_model, next_effort))
                _count_failed_attempt(turn_out, had_resumed)
                active_model, active_effort = next_model, next_effort
                turn_out = _run_attempt(formatted_prompt, conv_id)
        except TimeoutError as exc:
            part_u = getattr(exc, "partial_usage", None) or {}
            if getattr(exc, "partial_usage_from_snapshot", False):
                delta_part, partial_cum_u, partial_cum_turns, _ = _compute_conversation_turn_delta(
                    turn_usage=part_u,
                    step_usage_sum=getattr(exc, "partial_step_usage_sum", None) or {},
                    cur_num_turns=base_conv_turns + 1,
                    base_usage=base_conv_usage,
                    base_turns=base_conv_turns,
                    had_resumed_conversation=had_resumed,
                )
            else:
                delta_part = {k: max(0, int(part_u.get(k, 0) or 0)) for k in _USAGE_KEYS}
                partial_cum_u = dict(base_conv_usage or convert_agy_usage_to_vertex(None))
                _merge_usage(partial_cum_u, delta_part)
                has_turn_toks = any(delta_part.get(k, 0) > 0 for k in _USAGE_KEYS)
                partial_cum_turns = base_conv_turns + (1 if has_turn_toks else 0)
            _merge_usage(agg_usage, delta_part)
            all_tool_calls.extend(getattr(exc, "partial_tool_calls", None) or [])
            all_thoughts.extend(getattr(exc, "partial_thoughts", None) or [])
            eff_partial_conv = getattr(exc, "partial_agy_conversation_id", None) or conv_id or None
            if eff_partial_conv and any(partial_cum_u.get(k, 0) > 0 for k in _USAGE_KEYS):
                with _CONV_LEDGER_LOCK:
                    _CONVERSATION_USAGE_LEDGER[(cfg_key, str(eff_partial_conv))] = (
                        max(1, partial_cum_turns),
                        dict(partial_cum_u),
                    )
            exc.partial_usage = agg_usage  # type: ignore[attr-defined]
            exc.partial_tool_calls = all_tool_calls  # type: ignore[attr-defined]
            exc.partial_thoughts = all_thoughts  # type: ignore[attr-defined]
            exc.partial_containment = "agy_cli"  # type: ignore[attr-defined]
            exc.partial_agy_conversation_id = eff_partial_conv  # type: ignore[attr-defined]
            if any(partial_cum_u.get(k, 0) > 0 for k in _USAGE_KEYS):
                exc.partial_agy_conversation_usage = dict(partial_cum_u)  # type: ignore[attr-defined]
                exc.partial_agy_conversation_turns = max(1, partial_cum_turns)  # type: ignore[attr-defined]
            raise

        if turn_out.get("usage_from_snapshot", True):
            delta_u, updated_cum_u, updated_turns, unk_cum = _compute_conversation_turn_delta(
                turn_usage=turn_out["usage"],
                step_usage_sum=turn_out.get("step_usage_sum") or {},
                cur_num_turns=int(turn_out.get("num_turns") or 0),
                base_usage=base_conv_usage,
                base_turns=base_conv_turns,
                had_resumed_conversation=had_resumed,
            )
            if unk_cum:
                unknown_cumulative_baseline = True
        else:
            delta_u = dict(turn_out["usage"])
            updated_cum_u = dict(base_conv_usage or convert_agy_usage_to_vertex(None))
            _merge_usage(updated_cum_u, delta_u)
            updated_turns = max(int(turn_out.get("num_turns") or 0), base_conv_turns + 1)

        _merge_usage(agg_usage, delta_u)
        all_tool_calls.extend(turn_out["tool_calls"])
        all_thoughts.extend(turn_out["thoughts"])
        if turn_out.get("conversation_id"):
            conv_id = str(turn_out["conversation_id"])

        if turn_out["ok"] or any(delta_u.get(k, 0) > 0 for k in _USAGE_KEYS):
            base_conv_usage = updated_cum_u
            base_conv_turns = updated_turns
            if conv_id and any(base_conv_usage.get(k, 0) > 0 for k in _USAGE_KEYS):
                with _CONV_LEDGER_LOCK:
                    _CONVERSATION_USAGE_LEDGER[(cfg_key, conv_id)] = (base_conv_turns, dict(base_conv_usage))

        if not turn_out["ok"]:
            err_msg = str(turn_out["error"])
            if is_agy_auth_error(err_msg):
                err_msg = _format_auth_remediation(err_msg, agy_bin, env=env)
            exc = RuntimeError(err_msg)
            exc.partial_usage = agg_usage  # type: ignore[attr-defined]
            exc.partial_tool_calls = all_tool_calls  # type: ignore[attr-defined]
            exc.partial_thoughts = all_thoughts  # type: ignore[attr-defined]
            exc.partial_containment = "agy_cli"  # type: ignore[attr-defined]
            exc.partial_agy_conversation_id = conv_id or None  # type: ignore[attr-defined]
            if base_conv_usage is not None and any(base_conv_usage.get(k, 0) > 0 for k in _USAGE_KEYS):
                exc.partial_agy_conversation_usage = dict(base_conv_usage)  # type: ignore[attr-defined]
                exc.partial_agy_conversation_turns = max(1, base_conv_turns)  # type: ignore[attr-defined]
            raise exc

        last_text = str(turn_out.get("text") or "")
        stop_reason = str(turn_out.get("stop_reason") or "STOP")
        denied_actions = list(turn_out.get("denied_actions") or [])
        if (
            stop_reason == "STOP"
            and max_total_tokens is not None
            and max_total_tokens > 0
            and agg_usage["total_token_count"] > max_total_tokens
        ):
            stop_reason = "MAX_TOTAL_TOKENS_EXCEEDED"
        if stop_reason in ("MAX_TOOL_CALLS_EXCEEDED", "MAX_TOTAL_TOKENS_EXCEEDED", "DENIED_ACTIONS"):
            break

    return {
        "text": last_text,
        "usage": agg_usage,
        "tool_calls": all_tool_calls,
        "thoughts": all_thoughts,
        "stop_reason": stop_reason,
        "containment": "agy_cli",
        "containment_warning": sandbox.warning if (sandbox and not read_only) else None,
        **({"worker_sandbox": sandbox.summary()} if (sandbox and (not read_only or sandbox.backend)) else {}),
        "command_runs": [],
        "agy_conversation_id": conv_id or None,
        "agy_conversation_usage": dict(base_conv_usage) if base_conv_usage is not None else None,
        "agy_conversation_turns": base_conv_turns if base_conv_turns > 0 else None,
        "usage_is_cumulative": unknown_cumulative_baseline,
        "usage_estimated": unknown_cumulative_baseline,
        "reroute_note": reroute_note,
        "denied_actions": denied_actions,
    }


def _run_json_envelope_once(
    *,
    agy_bin: str,
    prompt_text: str,
    cwd: str,
    model: str,
    effort: str | None,
    json_schema: dict[str, Any] | None,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    if not prompt_text or not prompt_text.strip():
        raise ValueError("Worker prompt cannot be empty.")
    timeout_sec = max(0.05, float(timeout_seconds))
    print_timeout_sec = max(5, int(round(timeout_sec)) + 15)

    from server.worker_sandbox import plan_worker_sandbox

    Path(cwd).mkdir(parents=True, exist_ok=True)
    seat = resolve_enterprise_adc_seat(env)
    sandbox = plan_worker_sandbox(
        workspace_dir=cwd,
        config_dir=agy_cli_config_dir(env),
        env=env,
        adc_path=seat["adc_path"] if seat["active"] else None,
        readonly_paths=(Path(agy_bin).resolve().parent,),
        read_only_workspace=True,
        conversation_id=conversation_id,
    )
    sandbox_prefix = sandbox.command_prefix if sandbox else ()

    cmd: list[str] = [
        *sandbox_prefix,
        agy_bin,
        *_custom_config_dir_args(env),
        "--output-format",
        "json",
        "--mode",
        "plan",
        "--disable-slash-commands",
        "--print-timeout",
        f"{print_timeout_sec}s",
    ]
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["--effort", effort])
    if conversation_id:
        cmd.extend(["--conversation", conversation_id])
    if json_schema is not None:
        cmd.extend(["--json-schema", json.dumps(json_schema, separators=(",", ":"))])

    spawn_cookie = secrets.token_hex(16)
    sub_env = build_agy_subprocess_env(env, workspace_dir=cwd)
    sub_env["_AGY_SPAWN_COOKIE"] = spawn_cookie

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name != "nt"),
        env=sub_env,
    )
    pipe_inodes = _snapshot_pipe_inodes(proc)

    timed_out_flag = threading.Event()
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    t_in = _start_stdin_writer(proc, prompt_text)
    t_out = _start_pipe_chunk_drainer(proc.stdout, stdout_chunks)
    t_err = _start_pipe_chunk_drainer(proc.stderr, stderr_chunks)

    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out_flag.set()
        _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
    finally:
        _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
        t_in.join(timeout=0.3)
        t_out.join(timeout=0.3)
        t_err.join(timeout=0.3)
        _close_proc_pipes_safely(proc, stdout_thread=t_out, stderr_thread=t_err)

    stdout_text = b"".join(stdout_chunks).decode("utf-8", errors="replace").strip()
    stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()

    parsed: dict[str, Any] | None = None
    partial_step_usages: list[dict[str, Any]] = []
    if stdout_text:
        lines = stdout_text.splitlines()
        for candidate in (stdout_text, lines[-1]):
            cand_s = candidate.strip()
            if cand_s.startswith("{") and cand_s.endswith("}"):
                try:
                    obj = json.loads(cand_s)
                    if isinstance(obj, dict):
                        parsed = obj
                        break
                except ValueError:
                    pass
        if parsed is None or not isinstance(parsed.get("usage"), dict):
            for raw_line in lines:
                line_s = raw_line.strip()
                if not (line_s.startswith("{") and line_s.endswith("}")):
                    continue
                try:
                    frame = json.loads(line_s)
                except ValueError:
                    continue
                if isinstance(frame, dict) and isinstance(frame.get("usage"), dict):
                    partial_step_usages.append(frame["usage"])

    raw_usage = parsed.get("usage") if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict) else None
    if raw_usage is not None:
        vertex_usage = convert_agy_usage_to_vertex(raw_usage)
    elif partial_step_usages:
        vertex_usage = _sum_step_usages(partial_step_usages)
    else:
        vertex_usage = convert_agy_usage_to_vertex(None)

    if timed_out_flag.is_set():
        err = TimeoutError(f"Antigravity CLI request timed out after {timeout_sec:.1f}s")
        if any(vertex_usage.values()):
            err.partial_usage = vertex_usage  # type: ignore[attr-defined]
        raise err

    status_str = str(parsed.get("status") or "").upper() if isinstance(parsed, dict) else ""
    err_field = ""
    if isinstance(parsed, dict):
        if isinstance(parsed.get("error"), str) and parsed["error"].strip():
            err_field = parsed["error"].strip()
        elif parsed.get("is_error") is True and isinstance(parsed.get("result"), str):
            err_field = parsed["result"].strip()

    raw_conv = parsed.get("conversation_id") if isinstance(parsed, dict) else None
    raw_turns = parsed.get("num_turns") if isinstance(parsed, dict) else None
    conv_id_out = raw_conv.strip() if isinstance(raw_conv, str) and raw_conv.strip() else None
    num_turns_out = raw_turns if isinstance(raw_turns, int) and not isinstance(raw_turns, bool) else 0

    missing_json_envelope = parsed is None and not stdout_text
    if proc.returncode != 0 or status_str == "ERROR" or missing_json_envelope:
        return {
            "ok": False,
            "error": (
                err_field
                or stderr_text
                or stdout_text
                or (
                    "Antigravity CLI exited without emitting a JSON result envelope"
                    if missing_json_envelope and proc.returncode == 0
                    else f"Antigravity CLI exited with code {proc.returncode}"
                )
            ),
            "usage": vertex_usage,
            "parsed": parsed,
            "conversation_id": conv_id_out,
            "num_turns": num_turns_out,
        }

    text_out = ""
    if isinstance(parsed, dict):
        if "structured_output" in parsed and parsed["structured_output"] is not None:
            so = parsed["structured_output"]
            text_out = so if isinstance(so, str) else json.dumps(so)
        elif isinstance(parsed.get("response"), str):
            text_out = parsed["response"]
        elif isinstance(parsed.get("result"), str):
            text_out = parsed["result"]
    elif stdout_text:
        text_out = stdout_text

    return {
        "ok": True,
        "text": text_out,
        "usage": vertex_usage,
        "parsed": parsed,
        "conversation_id": conv_id_out,
        "num_turns": num_turns_out,
    }


def _ask_resume_key(
    request: Mapping[str, Any],
    model: str,
    effort: str | None,
    system_instruction: str,
    env: Mapping[str, str] | None,
) -> str | None:
    """Return the conversation lookup key for a resumable ask, or None when it cannot resume."""
    if request.get("stateless"):
        return None
    from server.agy_state import conversation_key, conversation_ttl_seconds

    prefix = str(request.get("prompt_prefix") or "")
    session_id = str(request.get("session_id") or "").strip()
    if not prefix.strip() or not str(request.get("prompt_delta") or "").strip() or not session_id:
        return None
    if conversation_ttl_seconds(env) <= 0:
        return None
    conv_tag = str(request.get("conversation_tag") or "").strip()
    effective_session = f"{session_id}:{conv_tag}" if conv_tag else session_id
    return conversation_key(
        session_id=effective_session,
        model=model,
        effort=effort,
        workspace=str(request.get("workspace_root") or ""),
        prefix=f"{system_instruction}\x00{prefix}",
        config_dir=str(agy_cli_config_dir(env)),
    )


def _stage_ask_media_parts(target_dir: str | Path, media_parts: Sequence[Any] | None) -> str:
    """Write inline `media_parts` into `target_dir/.agy_media` so `agy` can inspect them locally."""
    if not media_parts:
        return ""
    import base64

    ext_by_mime = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/heic": ".heic",
        "image/heif": ".heif",
        "application/pdf": ".pdf",
        "audio/wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/mpeg": ".mpeg",
    }
    media_dir = Path(target_dir) / ".agy_media"
    media_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        os.chmod(media_dir, 0o700)
    staged_rels: list[str] = []
    for idx, part in enumerate(media_parts, start=1):
        if not isinstance(part, Mapping):
            continue
        inline = part.get("inlineData") or part.get("inline_data")
        if not isinstance(inline, Mapping):
            continue
        raw_b64 = inline.get("data")
        mime = str(inline.get("mimeType") or inline.get("mime_type") or "").lower()
        if not isinstance(raw_b64, str) or not raw_b64:
            continue
        ext = ext_by_mime.get(mime, ".bin")
        fname = f"attachment_{idx}{ext}"
        try:
            dest_file = media_dir / fname
            dest_file.write_bytes(base64.b64decode(raw_b64))
            with contextlib.suppress(OSError):
                os.chmod(dest_file, 0o600)
            staged_rels.append(f".agy_media/{fname}")
        except (OSError, ValueError):
            continue
    if not staged_rels:
        return ""
    return f"\n\n[Attached media files in workspace: {', '.join(staged_rels)}]"


def run_agy_cli_ask(request: dict[str, Any]) -> dict[str, Any]:
    """Execute an `ask_gemini` / QA audit call via `agy --output-format json --mode plan`.

    Delta resume: when the request splits its prompt into `prompt_prefix` (the stable
    context) and `prompt_delta` (the new question) and names a `session_id`, a follow-up
    on the same context resumes the earlier `agy` conversation and sends only the question.
    A conversation is reused only for the same session, model, effort, workspace, and
    context, and only while it is fresh. If the resume fails for any reason, the stored
    entry is dropped and the full prompt is sent in a new conversation. The prompt always
    goes over stdin. Usage on a resumed turn is reported as that turn's delta.
    """
    from server.agy_state import (
        conversation_workspace_dir,
        forget_conversation,
        lookup_conversation,
        release_conversation_lease,
        remember_conversation,
    )

    env = request.get("env") if isinstance(request.get("env"), Mapping) else None
    ensure_enterprise_seat_ready(env)
    agy_bin = _resolve_or_bootstrap_agy_bin(env)

    requested_model = str(request.get("model") or "gemini-flash-next")
    effort = map_thinking_level_to_effort(request.get("thinking_level"), requested_model)
    active_model, active_effort, reroute_note = _resolve_cached_model_for_cli(
        agy_bin, requested_model, effort, env=env
    )

    prompt = str(request.get("prompt") or "")
    sys_inst = str(request.get("system_instruction") or "").strip()
    media_parts = request.get("media_parts") if isinstance(request.get("media_parts"), Sequence) else None
    no_tools_note = (
        "[Single-Shot Query Mode: Inspect the attached media files and prompt context below without modifying workspace files.]"
        if media_parts
        else "[Single-Shot Query Mode: Answer directly from the prompt context below without invoking workspace tools.]"
    )
    if sys_inst:
        full_prompt = f"{sys_inst}\n\n{no_tools_note}\n\n{prompt}"
    else:
        full_prompt = f"{no_tools_note}\n\n{prompt}"

    if request.get("timeout_seconds") is not None:
        timeout_seconds = float(request["timeout_seconds"])
    elif request.get("timeout_ms") is not None:
        timeout_seconds = float(request["timeout_ms"]) / 1000.0
    else:
        timeout_seconds = 180.0
    ask_deadline = time.monotonic() + timeout_seconds
    attempts_started = [0]

    def _next_attempt_timeout() -> float:
        remaining = ask_deadline - time.monotonic()
        floor = _MIN_ATTEMPT_SECONDS if attempts_started[0] else 0.0
        if remaining <= floor:
            raise TimeoutError(
                f"Antigravity CLI request timed out after {timeout_seconds:.1f}s"
            )
        attempts_started[0] += 1
        return max(0.05, remaining)

    lookup_key = _ask_resume_key(request, active_model, active_effort, sys_inst, env)
    resumed = lookup_conversation(lookup_key, env) if lookup_key else None
    lease_held = resumed is not None and bool(lookup_key)
    failed_attempts_usage = convert_agy_usage_to_vertex(None)

    try:
        with tempfile.TemporaryDirectory(prefix="cc-agy-ask-") as temp_cwd:
            ask_cwd = str(conversation_workspace_dir(lookup_key, env)) if lookup_key else temp_cwd
            media_note = _stage_ask_media_parts(ask_cwd, media_parts)
            try:
                effective_full_prompt = f"{full_prompt}{media_note}" if media_note else full_prompt
                out: dict[str, Any] | None = None
                if resumed is not None and lookup_key:
                    delta_prompt = f"{no_tools_note}\n\n{request['prompt_delta']}{media_note}"
                    resume_budget = _next_attempt_timeout()
                    if timeout_seconds >= _MIN_ATTEMPT_SECONDS * 2.5:
                        resume_budget = min(resume_budget, max(_MIN_ATTEMPT_SECONDS, timeout_seconds * 0.6))
                    try:
                        out = _run_json_envelope_once(
                            agy_bin=agy_bin,
                            prompt_text=delta_prompt,
                            cwd=ask_cwd,
                            model=active_model,
                            effort=active_effort,
                            json_schema=None,
                            timeout_seconds=resume_budget,
                            env=env,
                            conversation_id=resumed["conversation_id"],
                        )
                    except TimeoutError as exc:
                        forget_conversation(lookup_key, env)
                        lease_held = False
                        raw_partial = getattr(exc, "partial_usage", None)
                        if isinstance(raw_partial, Mapping) and any(raw_partial.values()):
                            timed_out_delta, _, _, _ = _compute_conversation_turn_delta(
                                turn_usage=dict(raw_partial),
                                step_usage_sum={},
                                cur_num_turns=int(resumed["turns"]) + 1,
                                base_usage=resumed["usage"],
                                base_turns=resumed["turns"],
                                had_resumed_conversation=True,
                            )
                            _merge_usage(failed_attempts_usage, timed_out_delta)
                            exc.partial_usage = {}  # type: ignore[attr-defined]
                        resumed, out = None, None
                        if ask_deadline - time.monotonic() <= _MIN_ATTEMPT_SECONDS:
                            raise
                    else:
                        if not out["ok"]:
                            forget_conversation(lookup_key, env)
                            lease_held = False
                            failed_delta, _, _, _ = _compute_conversation_turn_delta(
                                turn_usage=out["usage"],
                                step_usage_sum={},
                                cur_num_turns=int(out.get("num_turns") or 0),
                                base_usage=resumed["usage"],
                                base_turns=resumed["turns"],
                                had_resumed_conversation=True,
                            )
                            _merge_usage(failed_attempts_usage, failed_delta)
                            resumed, out = None, None
                if out is None:
                    out = _run_json_envelope_once(
                        agy_bin=agy_bin,
                        prompt_text=effective_full_prompt,
                        cwd=ask_cwd,
                        model=active_model,
                        effort=active_effort,
                        json_schema=None,
                        timeout_seconds=_next_attempt_timeout(),
                        env=env,
                    )
                seen_ask_states: set[tuple[str, str | None]] = {(active_model, active_effort)}
                for _retry_idx in range(2):
                    if out["ok"] or resumed is not None:
                        break
                    retry_spec = _maybe_retry_model_error(
                        err_text=out["error"],
                        agy_bin=agy_bin,
                        requested_model=active_model,
                        effort=active_effort,
                        env=env,
                    )
                    if retry_spec is None:
                        break
                    next_model, next_effort, reroute_note = retry_spec
                    if (next_model, next_effort) in seen_ask_states:
                        break
                    if ask_deadline - time.monotonic() <= _MIN_ATTEMPT_SECONDS:
                        break
                    seen_ask_states.add((next_model, next_effort))
                    _merge_usage(failed_attempts_usage, out["usage"])
                    active_model, active_effort = next_model, next_effort
                    out = _run_json_envelope_once(
                        agy_bin=agy_bin,
                        prompt_text=effective_full_prompt,
                        cwd=ask_cwd,
                        model=active_model,
                        effort=active_effort,
                        json_schema=None,
                        timeout_seconds=_next_attempt_timeout(),
                        env=env,
                    )
            finally:
                if media_note:
                    shutil.rmtree(Path(ask_cwd) / ".agy_media", ignore_errors=True)

        if not out["ok"]:
            err_msg = str(out["error"])
            if is_agy_auth_error(err_msg):
                err_msg = _format_auth_remediation(err_msg, agy_bin, env=env)
            exc = RuntimeError(err_msg)
            partial = dict(out["usage"])
            _merge_usage(partial, failed_attempts_usage)
            exc.partial_usage = partial  # type: ignore[attr-defined]
            raise exc

        turn_usage: dict[str, int] = dict(out["usage"])
        cumulative_usage = dict(out["usage"])
        cumulative_turns = max(1, int(out.get("num_turns") or 0))
        if resumed is not None:
            turn_usage, cumulative_usage, cumulative_turns, _ = _compute_conversation_turn_delta(
                turn_usage=out["usage"],
                step_usage_sum={},
                cur_num_turns=int(out.get("num_turns") or 0),
                base_usage=resumed["usage"],
                base_turns=resumed["turns"],
                had_resumed_conversation=True,
            )
        _merge_usage(turn_usage, failed_attempts_usage)

        conversation_id = out.get("conversation_id") or (resumed["conversation_id"] if resumed else None)
        store_key = _ask_resume_key(request, active_model, active_effort, sys_inst, env)
        if store_key and conversation_id:
            remember_conversation(
                store_key, str(conversation_id), turns=cumulative_turns, usage=cumulative_usage, env=env
            )
            if store_key == lookup_key:
                lease_held = False

        return {
            "text": str(out.get("text") or ""),
            "usage": turn_usage,
            "reroute_note": reroute_note,
            "agy_conversation_id": conversation_id,
            "resumed_conversation": resumed is not None,
        }
    except TimeoutError as exc:
        merged_partial = convert_agy_usage_to_vertex(None)
        raw_exc_partial = getattr(exc, "partial_usage", None)
        if isinstance(raw_exc_partial, Mapping):
            _merge_usage(merged_partial, dict(raw_exc_partial))
        _merge_usage(merged_partial, failed_attempts_usage)
        if any(merged_partial.values()):
            exc.partial_usage = merged_partial  # type: ignore[attr-defined]
        raise
    finally:
        if lease_held and lookup_key:
            release_conversation_lease(lookup_key, env)


def run_agy_cli_structured_writer(
    *,
    prompt: str,
    system_instruction: str,
    model: str,
    thinking_level: str | None,
    json_schema: dict[str, Any],
    timeout_seconds: float = 300.0,
    read_budget: int = 0,
    workspace_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, int]]:
    """Run a structured JSON-schema batch writer turn via `agy --output-format json`."""
    ensure_enterprise_seat_ready(env)
    agy_bin = _resolve_or_bootstrap_agy_bin(env)
    effort = map_thinking_level_to_effort(thinking_level, model)
    active_model, active_effort, _note = _resolve_cached_model_for_cli(
        agy_bin, model, effort, env=env
    )

    if read_budget > 0 and workspace_dir and Path(workspace_dir).is_dir():
        budget_rule = (
            f"[Read-Only Batch Writer Mode: You may read at most {int(read_budget)} workspace file(s) "
            "if needed, must NOT edit files or run commands, and must return valid JSON matching the schema.]"
        )
        full_prompt = f"{system_instruction.strip()}\n\n{budget_rule}\n\n{prompt}"
        out = _consume_writer_envelope(
            agy_bin=agy_bin,
            full_prompt=full_prompt,
            cwd=str(workspace_dir),
            active_model=active_model,
            active_effort=active_effort,
            json_schema=json_schema,
            timeout_seconds=timeout_seconds,
            env=env,
        )
    else:
        budget_rule = (
            "[Pure Generation Batch Writer Mode: Do NOT call any tools; generate the JSON object "
            "matching the schema directly from the provided prompt context.]"
        )
        full_prompt = f"{system_instruction.strip()}\n\n{budget_rule}\n\n{prompt}"
        with tempfile.TemporaryDirectory(prefix="cc-agy-writer-") as empty_cwd:
            out = _consume_writer_envelope(
                agy_bin=agy_bin,
                full_prompt=full_prompt,
                cwd=empty_cwd,
                active_model=active_model,
                active_effort=active_effort,
                json_schema=json_schema,
                timeout_seconds=timeout_seconds,
                env=env,
            )

    if not out["ok"]:
        err_msg = str(out["error"])
        if is_agy_auth_error(err_msg):
            err_msg = _format_auth_remediation(err_msg, agy_bin, env=env)
        exc = RuntimeError(err_msg)
        exc.partial_usage = out["usage"]  # type: ignore[attr-defined]
        raise exc

    return str(out.get("text") or ""), out["usage"]


def _consume_writer_envelope(
    *,
    agy_bin: str,
    full_prompt: str,
    cwd: str,
    active_model: str,
    active_effort: str | None,
    json_schema: dict[str, Any],
    timeout_seconds: float,
    env: Mapping[str, str] | None,
) -> dict[str, Any]:
    cur_model = active_model
    cur_effort = active_effort
    cur_schema: dict[str, Any] | None = json_schema
    seen_model_states: set[tuple[str, str | None]] = {(cur_model, cur_effort)}
    out: dict[str, Any] = {"ok": False, "error": "Writer did not execute", "usage": convert_agy_usage_to_vertex(None)}
    failed_attempts_usage = convert_agy_usage_to_vertex(None)

    def _with_failed_usage(result: dict[str, Any]) -> dict[str, Any]:
        total = dict(result.get("usage") or convert_agy_usage_to_vertex(None))
        _merge_usage(total, failed_attempts_usage)
        return {**result, "usage": total}

    for _attempt in range(4):
        if _attempt:
            _merge_usage(failed_attempts_usage, out["usage"])
        out = _run_json_envelope_once(
            agy_bin=agy_bin,
            prompt_text=full_prompt,
            cwd=cwd,
            model=cur_model,
            effort=cur_effort,
            json_schema=cur_schema,
            timeout_seconds=timeout_seconds,
            env=env,
        )
        if out["ok"]:
            return _with_failed_usage(out)
        err_str = str(out.get("error") or "")
        err_low = err_str.lower()
        if cur_schema is not None and "--json-schema" in err_str and (
            "not supported" in err_low
            or "unknown" in err_low
            or "unrecognized" in err_low
            or "flag provided but not defined" in err_low
        ):
            cur_schema = None
            continue
        retry_spec = _maybe_retry_model_error(
            err_text=err_str,
            agy_bin=agy_bin,
            requested_model=cur_model,
            effort=cur_effort,
            env=env,
        )
        if retry_spec is not None:
            next_model, next_effort, _ = retry_spec
            if (next_model, next_effort) not in seen_model_states:
                seen_model_states.add((next_model, next_effort))
                cur_model, cur_effort = next_model, next_effort
                continue
        return _with_failed_usage(out)
    return _with_failed_usage(out)
