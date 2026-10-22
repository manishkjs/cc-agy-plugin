"""Gemini Enterprise subscription (user OAuth + Business AI Code) adapter.

Routes `google-antigravity` (`VertexEndpoint`) and `google-genai` (`Client(vertexai=True)`)
through an in-process `127.0.0.1` loopback proxy that translates Vertex-shaped requests into
Google Cloud's Business AI Code (`businessaicode.googleapis.com`) protocol so users with a
Gemini Enterprise Standard or Plus license consume their included monthly seat capacity
instead of pay-as-you-go Vertex AI or AI Studio billing.
"""

from __future__ import annotations

import atexit
import copy
from dataclasses import dataclass
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping
import urllib.error
import urllib.parse
import urllib.request

GLOBAL_BAIC_HOST = "https://businessaicode.googleapis.com"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
PROXY_AUTH_HEADER = "X-CC-AGY-Proxy-Token"
PROXY_LOCATION_HEADER = "X-CC-AGY-Enterprise-Location"

_VALID_LOCATION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_VERTEX_MODEL_PATH_RE = re.compile(
    r"/projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)"
    r"(?:/publishers/[^/]+)?/models/(?P<model>[^/:]+):(?P<action>streamGenerateContent|generateContent)"
)
_FALLBACK_MODEL_PATH_RE = re.compile(
    r"models/(?P<model>[^/:]+):(?P<action>streamGenerateContent|generateContent)"
)

_TOKEN_REFRESH_MARGIN_SECONDS = 60.0
_GCLOUD_TOKEN_TTL_SECONDS = 2400.0
_ENTITLEMENT_TTL_SECONDS = 900.0

_GROUNDING_PASSTHROUGH_FIELDS = frozenset(
    {"excludeDomains", "blockingConfidence", "exclude_domains", "blocking_confidence"}
)


class EnterpriseAuthError(RuntimeError):
    """Raised when human Google OAuth credentials or Gemini Enterprise licenses are missing or invalid."""


@dataclass(frozen=True)
class EnterpriseEntitlement:
    """Resolved Gemini Enterprise seat license and serving jurisdiction."""

    project_id: str
    location: str
    user_tier: str
    tier_display_name: str = ""
    reroute_note: str | None = None


class _CachedToken:
    def __init__(self, token: str, expires_at_mono: float, source: str) -> None:
        self.token = token
        self.expires_at_mono = expires_at_mono
        self.source = source


_token_lock = threading.Lock()
_cached_tokens: dict[str, _CachedToken] = {}

_entitlement_lock = threading.Lock()
_cached_entitlements: dict[tuple[str, str, str, str], tuple[EnterpriseEntitlement, float]] = {}


def clear_enterprise_caches() -> None:
    """Clear in-memory token and entitlement caches (used by tests and credential rotation)."""
    with _token_lock:
        _cached_tokens.clear()
    with _entitlement_lock:
        _cached_entitlements.clear()


def _clean_env_str(env: Mapping[str, str], key: str) -> str:
    val = str(env.get(key) or "").strip()
    if not val or val.startswith("${") or val.lower() in ("none", "null", "dummy"):
        return ""
    return val


def baic_host_for_location(location: str | None, *, env: Mapping[str, str] | None = None) -> str:
    """Return the jurisdiction-specific Business AI Code base URL for `location`."""
    source = env if env is not None else os.environ
    override = _clean_env_str(source, "CC_AGY_BAIC_BASE_URL")
    if override:
        return override.rstrip("/")
    loc = (location or "global").strip().lower()
    if not loc or loc == "global":
        return GLOBAL_BAIC_HOST
    if not _VALID_LOCATION_RE.fullmatch(loc):
        raise ValueError(f"Invalid Gemini Enterprise location {location!r}")
    if "-" in loc:
        return f"https://{loc}-businessaicode.googleapis.com"
    return f"https://businessaicode.{loc}.rep.googleapis.com"


def _home_dir(env: Mapping[str, str]) -> Path:
    custom_home = _clean_env_str(env, "HOME") or _clean_env_str(env, "USERPROFILE")
    return Path(custom_home) if custom_home else Path.home()


def _adc_candidates(env: Mapping[str, str]) -> list[tuple[Path, bool]]:
    """Return `(path, is_explicit)` pairs for Google Application Default Credentials."""
    out: list[tuple[Path, bool]] = []
    explicit = _clean_env_str(env, "GOOGLE_APPLICATION_CREDENTIALS")
    if explicit:
        out.append((Path(explicit).expanduser(), True))
    gcloud_cfg = _clean_env_str(env, "CLOUDSDK_CONFIG")
    if gcloud_cfg:
        out.append((Path(gcloud_cfg).expanduser() / "application_default_credentials.json", False))
    out.append((_home_dir(env) / ".config" / "gcloud" / "application_default_credentials.json", False))
    appdata = _clean_env_str(env, "APPDATA")
    if appdata:
        out.append((Path(appdata) / "gcloud" / "application_default_credentials.json", False))
    return out


def _gcloud_config_dir(env: Mapping[str, str]) -> Path:
    gcloud_cfg = _clean_env_str(env, "CLOUDSDK_CONFIG")
    if gcloud_cfg:
        return Path(gcloud_cfg).expanduser()
    appdata = _clean_env_str(env, "APPDATA")
    if os.name == "nt" and appdata:
        return Path(appdata) / "gcloud"
    return _home_dir(env) / ".config" / "gcloud"


def _read_gcloud_config_field_offline(env: Mapping[str, str], field_name: str) -> str | None:
    """Read `field_name` (such as `account` or `project`) from `~/.config/gcloud` without subprocesses."""
    cfg_dir = _gcloud_config_dir(env)
    active_name = "default"
    active_file = cfg_dir / "active_config"
    if active_file.is_file():
        try:
            raw_name = active_file.read_text(encoding="utf-8", errors="replace").strip()
            if raw_name and re.fullmatch(r"[A-Za-z0-9_-]+", raw_name):
                active_name = raw_name
        except OSError:
            pass
    cfg_path = cfg_dir / "configurations" / f"config_{active_name}"
    if not cfg_path.is_file():
        return None
    try:
        for line in cfg_path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if "=" not in stripped or stripped.startswith(("#", ";", "[")):
                continue
            lhs, rhs = stripped.split("=", 1)
            if lhs.strip() == field_name:
                val = rhs.strip()
                if val:
                    return val
    except OSError:
        return None
    return None


def _read_gcloud_active_account_offline(env: Mapping[str, str]) -> str | None:
    """Read the active `gcloud` account from disk without spawning subprocesses or network calls."""
    return _read_gcloud_config_field_offline(env, "account")


def _resolve_quota_project_hint(env: Mapping[str, str], explicit_project: str | None = None) -> str | None:
    """Find a quota project ID for Cloud SDK ADC calls (`X-Goog-User-Project`) when available."""
    if explicit_project and explicit_project.strip():
        return explicit_project.strip()
    for env_key in ("GOOGLE_CLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT", "GCLOUD_PROJECT"):
        cand = _clean_env_str(env, env_key)
        if cand:
            return cand
    for adc_path, _is_explicit in _adc_candidates(env):
        if not adc_path.is_file():
            continue
        try:
            payload = json.loads(adc_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if isinstance(payload, dict):
            qp = str(payload.get("quota_project_id") or "").strip()
            if qp:
                return qp
    return _read_gcloud_config_field_offline(env, "project")


def inspect_enterprise_credentials_offline(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Zero-network check verifying that a human Google OAuth credential is available."""
    source = env if env is not None else os.environ
    if _clean_env_str(source, "CC_AGY_ENTERPRISE_ACCESS_TOKEN"):
        return {"ok": True, "credential_source": "env_access_token", "account": None}

    for adc_path, is_explicit in _adc_candidates(source):
        if is_explicit and not adc_path.is_file():
            return {
                "ok": False,
                "credential_source": "missing_adc_file",
                "error": f"GOOGLE_APPLICATION_CREDENTIALS points to missing file: {adc_path}",
                "remediation": "Unset GOOGLE_APPLICATION_CREDENTIALS and run `gcloud auth application-default login`.",
            }
        if not adc_path.is_file():
            continue
        try:
            payload = json.loads(adc_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        cred_type = str(payload.get("type") or "").strip().lower()
        if cred_type == "service_account":
            sa_email = str(payload.get("client_email") or "service account")
            return {
                "ok": False,
                "credential_source": "service_account_rejected",
                "account": sa_email,
                "error": (
                    f"Found service account credentials ({sa_email}), which cannot use Gemini Enterprise "
                    "subscription seats (licenses belong to human user accounts)."
                ),
                "remediation": (
                    "Unset `GOOGLE_APPLICATION_CREDENTIALS` and run `gcloud auth application-default login` "
                    "with your licensed work email."
                ),
            }
        if cred_type in ("authorized_user", "external_account"):
            return {
                "ok": True,
                "credential_source": f"adc_{cred_type}",
                "account": payload.get("account") or None,
                "quota_project_id": payload.get("quota_project_id"),
            }

    acct = _read_gcloud_active_account_offline(source)
    if acct:
        if acct.lower().endswith(".gserviceaccount.com"):
            return {
                "ok": False,
                "credential_source": "service_account_rejected",
                "account": acct,
                "error": (
                    f"Active gcloud account ({acct}) is a service account, which cannot hold a Gemini Enterprise seat."
                ),
                "remediation": "Run `gcloud auth application-default login` with your licensed work email.",
            }
        if "@" in acct and shutil.which("gcloud"):
            return {"ok": True, "credential_source": "gcloud_user_login", "account": acct}

    return {
        "ok": False,
        "credential_source": "missing",
        "account": None,
        "error": "No human Google sign-in found for Gemini Enterprise subscription mode.",
        "remediation": (
            "Run `gcloud auth application-default login` with your Gemini Enterprise work email "
            "(API keys and service accounts cannot draw from seat subscription quota)."
        ),
    }


def _is_service_account_creds(creds: Any) -> tuple[bool, str]:
    mod_name = type(creds).__module__.lower()
    sa_email = str(
        getattr(creds, "service_account_email", None)
        or getattr(creds, "_service_account_email", None)
        or ""
    ).strip()
    if "service_account" in mod_name or "impersonated_credentials" in mod_name:
        return True, sa_email or "service_account"
    if "compute_engine" in mod_name:
        return True, sa_email or "gce_metadata_service_account"
    if sa_email.lower().endswith(".gserviceaccount.com"):
        return True, sa_email
    return False, ""


def _token_via_gcloud_cli(env: Mapping[str, str]) -> _CachedToken | None:
    gcloud_bin = shutil.which("gcloud")
    if not gcloud_bin:
        return None
    run_env = {**os.environ, **{k: str(v) for k, v in env.items()}}
    acct_proc = subprocess.run(
        [gcloud_bin, "config", "get-value", "account", "--quiet"],
        capture_output=True,
        text=True,
        env=run_env,
        check=False,
        timeout=8,
    )
    acct = (acct_proc.stdout or "").strip()
    if not acct or "@" not in acct or acct.lower().endswith(".gserviceaccount.com"):
        return None
    tok_proc = subprocess.run(
        [gcloud_bin, "auth", "print-access-token", "--quiet"],
        capture_output=True,
        text=True,
        env=run_env,
        check=False,
        timeout=15,
    )
    tok = (tok_proc.stdout or "").strip()
    if tok_proc.returncode != 0 or not tok or " " in tok:
        return None
    return _CachedToken(tok, time.monotonic() + _GCLOUD_TOKEN_TTL_SECONDS, f"gcloud:{acct}")


def resolve_enterprise_access_token(
    *,
    env: Mapping[str, str] | None = None,
    force_refresh: bool = False,
) -> str:
    """Return a valid human Google OAuth access token, rejecting service accounts cleanly."""
    source = env if env is not None else os.environ
    explicit_tok = _clean_env_str(source, "CC_AGY_ENTERPRISE_ACCESS_TOKEN")
    if explicit_tok:
        return explicit_tok

    cache_key = "|".join(
        [
            _clean_env_str(source, "GOOGLE_APPLICATION_CREDENTIALS"),
            _clean_env_str(source, "CLOUDSDK_CONFIG"),
        ]
    )
    now = time.monotonic()
    if not force_refresh:
        with _token_lock:
            cached = _cached_tokens.get(cache_key)
            if cached is not None and cached.expires_at_mono - now > _TOKEN_REFRESH_MARGIN_SECONDS:
                return cached.token

    # Fail fast when GOOGLE_APPLICATION_CREDENTIALS explicitly points to a service account key file.
    explicit_adc = _clean_env_str(source, "GOOGLE_APPLICATION_CREDENTIALS")
    if explicit_adc:
        adc_file = Path(explicit_adc).expanduser()
        if not adc_file.is_file():
            raise EnterpriseAuthError(
                f"GOOGLE_APPLICATION_CREDENTIALS points to missing file {adc_file!s}. "
                "Unset GOOGLE_APPLICATION_CREDENTIALS and run `gcloud auth application-default login`."
            )
        try:
            adc_json = json.loads(adc_file.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            adc_json = {}
        if isinstance(adc_json, dict) and str(adc_json.get("type") or "").strip().lower() == "service_account":
            sa_email = str(adc_json.get("client_email") or "service account")
            raise EnterpriseAuthError(
                f"Gemini Enterprise subscription mode requires a human Google sign-in, not service account "
                f"{sa_email!r} (service accounts have no Gemini Enterprise seat license and bill Vertex AI "
                "pay-as-you-go). Unset GOOGLE_APPLICATION_CREDENTIALS and run: gcloud auth application-default login"
            )

    saw_ambient_sa: str | None = None
    try:
        import google.auth
        from google.auth.transport.requests import Request as GoogleAuthRequest

        creds, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
        is_sa, sa_label = _is_service_account_creds(creds)
        if is_sa:
            saw_ambient_sa = sa_label
        else:
            if not creds.valid or creds.expired or force_refresh or not getattr(creds, "token", None):
                creds.refresh(GoogleAuthRequest())
            tok = str(getattr(creds, "token", "") or "").strip()
            if tok:
                expiry = getattr(creds, "expiry", None)
                if expiry is not None:
                    ttl = max(120.0, float(expiry.timestamp() - time.time()))
                else:
                    ttl = _GCLOUD_TOKEN_TTL_SECONDS
                entry = _CachedToken(tok, time.monotonic() + ttl, "adc_user")
                with _token_lock:
                    _cached_tokens[cache_key] = entry
                return tok
    except EnterpriseAuthError:
        raise
    except Exception:
        pass

    gcloud_entry = _token_via_gcloud_cli(source)
    if gcloud_entry is not None:
        with _token_lock:
            _cached_tokens[cache_key] = gcloud_entry
        return gcloud_entry.token

    if saw_ambient_sa:
        raise EnterpriseAuthError(
            f"Only machine/service-account credentials ({saw_ambient_sa}) were found, which cannot draw from "
            "Gemini Enterprise user seat quota. Run `gcloud auth application-default login` with your licensed "
            "work email."
        )
    raise EnterpriseAuthError(
        "No human Google sign-in found for Gemini Enterprise subscription mode. "
        "Run `gcloud auth application-default login` with your licensed work email "
        "(API keys and service accounts cannot use Gemini Enterprise seat quota)."
    )


def _http_json(
    method: str,
    url: str,
    *,
    access_token: str,
    quota_project: str | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any], str]:
    raw_bytes = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    if quota_project and quota_project.strip():
        headers["X-Goog-User-Project"] = quota_project.strip()
    if raw_bytes is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=raw_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            text = str(exc)
    except urllib.error.URLError as exc:
        raise EnterpriseAuthError(f"Failed to reach Gemini Enterprise licensing endpoint ({url}): {exc}") from exc

    parsed: dict[str, Any] = {}
    if text.strip():
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                parsed = loaded
        except json.JSONDecodeError:
            parsed = {}
    return status, parsed, text


def _parse_license_entry(raw: Any) -> EnterpriseEntitlement | None:
    if not isinstance(raw, dict):
        return None
    user_tier = str(raw.get("userTier") or raw.get("user_tier") or "").strip()
    project_id = str(raw.get("projectId") or raw.get("project_id") or "").strip()
    location = str(raw.get("location") or "global").strip().lower() or "global"
    display = str(raw.get("tierDisplayName") or raw.get("tier_display_name") or "").strip()
    if not user_tier or not project_id:
        return None
    return EnterpriseEntitlement(
        project_id=project_id,
        location=location,
        user_tier=user_tier,
        tier_display_name=display,
    )


def fetch_enterprise_licenses(
    access_token: str,
    *,
    quota_project: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 30.0,
) -> list[EnterpriseEntitlement]:
    """Fetch active Gemini Enterprise licenses (`GET /v1beta:fetchLicenses`) for the signed-in user."""
    source = env if env is not None else os.environ
    host = baic_host_for_location("global", env=source)
    url = f"{host}/v1beta:fetchLicenses"
    qp = _resolve_quota_project_hint(source, quota_project)
    status, payload, raw_text = _http_json(
        "GET", url, access_token=access_token, quota_project=qp, timeout=timeout
    )
    # If an unprivileged quota_project header caused a 403, retry once without X-Goog-User-Project.
    if status == 403 and qp:
        retry_status, retry_payload, retry_raw = _http_json(
            "GET", url, access_token=access_token, quota_project=None, timeout=timeout
        )
        if retry_status < 400:
            status, payload, raw_text = retry_status, retry_payload, retry_raw
    if status in (401, 403):
        err_detail = str((payload.get("error") or {}).get("message") or raw_text[:240]).strip()
        if "quota project" in err_detail.lower():
            raise EnterpriseAuthError(
                f"Gemini Enterprise license lookup needs a quota project for Application Default Credentials "
                f"(HTTP {status}: {err_detail}). Run `gcloud auth application-default set-quota-project YOUR_PROJECT_ID` "
                "or pass `--project YOUR_PROJECT_ID`."
            )
        raise EnterpriseAuthError(
            f"Gemini Enterprise license lookup was rejected (HTTP {status}: {err_detail}). "
            "Sign in with your licensed work email via `gcloud auth application-default login`."
        )
    if status >= 400:
        err_detail = str((payload.get("error") or {}).get("message") or raw_text[:240]).strip()
        raise EnterpriseAuthError(
            f"Gemini Enterprise license lookup failed (HTTP {status}: {err_detail})."
        )
    raw_list = payload.get("licenses")
    if not isinstance(raw_list, list):
        return []
    out: list[EnterpriseEntitlement] = []
    for item in raw_list:
        parsed = _parse_license_entry(item)
        if parsed is not None:
            out.append(parsed)
    return out


def self_assign_enterprise_license(
    access_token: str,
    *,
    project: str,
    location: str = "global",
    env: Mapping[str, str] | None = None,
    timeout: float = 30.0,
) -> EnterpriseEntitlement:
    """Attempt idempotent Gemini Enterprise license self-assignment on `(project, location)`."""
    clean_proj = project.strip()
    if not clean_proj:
        raise EnterpriseAuthError("project is required for Gemini Enterprise license self-assignment.")
    loc = (location or "global").strip().lower() or "global"
    if "-" in loc:
        raise EnterpriseAuthError(
            f"Invalid location {location!r} for Gemini Enterprise license self-assignment: "
            "licensing endpoints require a multi-region jurisdiction ('global', 'us', or 'eu')."
        )
    host = baic_host_for_location(loc, env=env)
    safe_proj = urllib.parse.quote(clean_proj, safe="")
    safe_loc = urllib.parse.quote(loc, safe="")
    url = f"{host}/v1beta/projects/{safe_proj}/locations/{safe_loc}:selfAssignLicense"
    req_body = {"parent": f"projects/{clean_proj}/locations/{loc}"}
    status, payload, raw_text = _http_json(
        "POST",
        url,
        access_token=access_token,
        quota_project=clean_proj,
        body=req_body,
        timeout=timeout,
    )
    if status >= 400:
        err_detail = str((payload.get("error") or {}).get("message") or raw_text[:240]).strip()
        raise EnterpriseAuthError(
            f"No active Gemini Enterprise license on project {clean_proj!r} ({loc}), and self-assignment "
            f"returned HTTP {status}: {err_detail}"
        )
    lic = _parse_license_entry(payload.get("license") or payload)
    if lic is None:
        raise EnterpriseAuthError(
            f"Gemini Enterprise selfAssignLicense on {clean_proj!r} ({loc}) returned an incomplete license payload."
        )
    return lic


def _infer_baic_jurisdiction_from_region(location: str | None) -> str:
    """Map a regional GCP string (e.g. `us-central1`, `europe-west4`) to a BAIC multi-region jurisdiction."""
    loc = (location or "").strip().lower()
    if not loc or loc == "global":
        return "global"
    if loc in ("us", "eu"):
        return loc
    if loc.startswith(("us-", "northamerica-", "southamerica-")):
        return "us"
    if loc.startswith(("europe-", "eu-")):
        return "eu"
    return "global"


def resolve_enterprise_entitlement(
    access_token: str,
    *,
    project: str | None = None,
    location: str | None = None,
    location_explicit: bool = False,
    env: Mapping[str, str] | None = None,
) -> EnterpriseEntitlement:
    """Resolve the active Gemini Enterprise `(project_id, location, user_tier)` for `access_token`."""
    source = env if env is not None else os.environ
    clean_proj = (project or "").strip() or None
    clean_loc = (location or "").strip().lower() or None

    explicit_tier = _clean_env_str(source, "CC_AGY_ENTERPRISE_TIER")
    if explicit_tier and clean_proj:
        resolved_explicit_loc = (
            clean_loc
            if (clean_loc and "-" not in clean_loc)
            else _infer_baic_jurisdiction_from_region(clean_loc)
        )
        return EnterpriseEntitlement(
            project_id=clean_proj,
            location=resolved_explicit_loc,
            user_tier=explicit_tier,
            tier_display_name=explicit_tier,
        )

    tok_hash = hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:16]
    base_override = _clean_env_str(source, "CC_AGY_BAIC_BASE_URL")
    cache_key = (tok_hash, clean_proj or "", f"{clean_loc or ''}:{int(location_explicit)}", base_override)
    now = time.monotonic()
    with _entitlement_lock:
        hit = _cached_entitlements.get(cache_key)
        if hit is not None and hit[1] > now:
            return hit[0]

    licenses = fetch_enterprise_licenses(access_token, quota_project=clean_proj, env=source)
    chosen: EnterpriseEntitlement | None = None

    if clean_proj:
        exact = [
            lic
            for lic in licenses
            if lic.project_id == clean_proj and (clean_loc is None or lic.location == clean_loc)
        ]
        if exact:
            chosen = exact[0]
        else:
            same_proj = [lic for lic in licenses if lic.project_id == clean_proj]
            if same_proj and not location_explicit:
                hint_jur = _infer_baic_jurisdiction_from_region(clean_loc)
                preferred = [lic for lic in same_proj if lic.location == hint_jur]
                base_lic = preferred[0] if preferred else same_proj[0]
                note = (
                    f"Using Gemini Enterprise license jurisdiction '{base_lic.location}' "
                    f"for project '{clean_proj}' (instead of '{clean_loc}')."
                    if clean_loc and clean_loc != base_lic.location
                    else None
                )
                chosen = EnterpriseEntitlement(
                    project_id=base_lic.project_id,
                    location=base_lic.location,
                    user_tier=base_lic.user_tier,
                    tier_display_name=base_lic.tier_display_name,
                    reroute_note=note,
                )
            else:
                if clean_loc and location_explicit:
                    assign_loc = clean_loc
                else:
                    assign_loc = _infer_baic_jurisdiction_from_region(clean_loc)
                try:
                    chosen = self_assign_enterprise_license(
                        access_token, project=clean_proj, location=assign_loc, env=source
                    )
                except EnterpriseAuthError as exc:
                    if licenses:
                        avail = ", ".join(f"{l.project_id} ({l.location})" for l in licenses[:5])
                        raise EnterpriseAuthError(
                            f"{exc} Available licensed projects on this account: {avail}."
                        ) from exc
                    raise
    else:
        if clean_loc and location_explicit:
            filtered = [lic for lic in licenses if lic.location == clean_loc]
            if filtered:
                chosen = filtered[0]
            elif licenses:
                avail = ", ".join(f"{l.project_id} ({l.location})" for l in licenses[:5])
                raise EnterpriseAuthError(
                    f"No Gemini Enterprise license found in jurisdiction {clean_loc!r}. "
                    f"Available licensed projects on this account: {avail}."
                )
        elif licenses:
            if clean_loc:
                hint_jur = _infer_baic_jurisdiction_from_region(clean_loc)
                hint_matches = [lic for lic in licenses if lic.location in (clean_loc, hint_jur)]
                chosen = hint_matches[0] if hint_matches else licenses[0]
            else:
                chosen = licenses[0]

        if chosen is None:
            raise EnterpriseAuthError(
                "No active Gemini Enterprise license was found for this Google account. "
                "Sign in with the work email assigned a Gemini Enterprise Standard or Plus seat "
                "(`gcloud auth application-default login`), or pass `--project YOUR_PROJECT_ID` if your "
                "organization allows license self-assignment."
            )

    with _entitlement_lock:
        _cached_entitlements[cache_key] = (chosen, time.monotonic() + _ENTITLEMENT_TTL_SECONDS)
    return chosen


def to_baic_generate_content_request(
    vertex_body: dict[str, Any],
    *,
    model: str,
    user_tier: str,
) -> dict[str, Any]:
    """Translate a Vertex-shaped `generateContent` JSON body into a Business AI Code request."""
    out = copy.deepcopy(vertex_body)
    out.pop("sessionId", None)
    out.pop("session_id", None)
    out.pop("model", None)

    for gen_key in ("generationConfig", "generation_config"):
        gen_cfg = out.get(gen_key)
        if not isinstance(gen_cfg, dict):
            continue
        gen_cfg.pop("imageConfig", None)
        gen_cfg.pop("image_config", None)
        for think_key in ("thinkingConfig", "thinking_config"):
            think_cfg = gen_cfg.get(think_key)
            if isinstance(think_cfg, dict):
                think_cfg.pop("thinkingLevel", None)
                think_cfg.pop("thinking_level", None)
                if not think_cfg:
                    gen_cfg.pop(think_key, None)
        if not gen_cfg:
            out.pop(gen_key, None)

    raw_tools = out.get("tools")
    if isinstance(raw_tools, list):
        converted_tools: list[Any] = []
        for entry in raw_tools:
            if isinstance(entry, dict) and ("googleSearch" in entry or "google_search" in entry):
                new_entry = {k: v for k, v in entry.items() if k not in ("googleSearch", "google_search")}
                gs_val = entry.get("googleSearch") if "googleSearch" in entry else entry.get("google_search")
                ent_search: dict[str, Any] = {}
                if isinstance(gs_val, dict):
                    ent_search = {k: v for k, v in gs_val.items() if k in _GROUNDING_PASSTHROUGH_FIELDS}
                new_entry["enterpriseWebSearch"] = ent_search
                converted_tools.append(new_entry)
            else:
                converted_tools.append(entry)
        out["tools"] = converted_tools

    clean_model = model.strip()
    if "/" in clean_model:
        clean_model = clean_model.rsplit("/", 1)[-1]
    out["aicode"] = {"experience": clean_model}
    out["entitlement"] = {"user_tier": user_tier}
    return out


_MAX_PROXY_BODY_BYTES = 64 * 1024 * 1024


class _EnterpriseThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class EnterpriseSubscriptionProxy:
    """Thread-safe local `127.0.0.1` HTTP/SSE proxy translating Vertex SDK calls to BAIC."""

    def __init__(self, *, env: Mapping[str, str] | None = None) -> None:
        self._env = dict(env) if env is not None else None
        self.secret = secrets.token_urlsafe(32)
        self._entitlements: dict[tuple[str, str], EnterpriseEntitlement] = {}
        self._lock = threading.Lock()
        self._server = _EnterpriseThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        host, port = self._server.server_address[:2]
        self.base_url = f"http://{host}:{port}"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="cc-agy-enterprise-proxy",
            daemon=True,
        )
        self._thread.start()

    def register_entitlement(self, entitlement: EnterpriseEntitlement) -> None:
        with self._lock:
            self._entitlements[(entitlement.project_id, entitlement.location.lower())] = entitlement

    def lookup_entitlement(self, project_id: str, location: str) -> EnterpriseEntitlement | None:
        clean_p = (project_id or "").strip()
        clean_l = (location or "").strip().lower()
        with self._lock:
            if clean_p and clean_l:
                exact = self._entitlements.get((clean_p, clean_l))
                if exact is not None:
                    return exact
            if clean_p:
                for (reg_p, _reg_l), ent in self._entitlements.items():
                    if reg_p == clean_p:
                        return ent
            elif clean_l:
                loc_matches = [ent for (_reg_p, reg_l), ent in self._entitlements.items() if reg_l == clean_l]
                if len(loc_matches) == 1:
                    return loc_matches[0]
                if len(self._entitlements) == 1:
                    return next(iter(self._entitlements.values()))
            elif len(self._entitlements) == 1:
                return next(iter(self._entitlements.values()))
            return None

    def close(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass
        try:
            self._server.server_close()
        except Exception:
            pass

    def _active_env(self) -> Mapping[str, str]:
        return self._env if self._env is not None else os.environ

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        proxy = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = 30.0

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

            def _read_body_checked(self) -> tuple[int, bytes, str]:
                if self.headers.get("Transfer-Encoding"):
                    return 411, b"", "Transfer-Encoding is not supported; Content-Length is required."
                raw_len = self.headers.get("Content-Length")
                if raw_len is None or raw_len.strip() == "":
                    return 200, b"", ""
                try:
                    n = int(raw_len.strip())
                except ValueError:
                    return 400, b"", "Invalid Content-Length header."
                if n < 0:
                    return 400, b"", "Negative Content-Length header is not allowed."
                if n > _MAX_PROXY_BODY_BYTES:
                    return 413, b"", "Request payload exceeds 64 MiB local proxy limit."
                return 200, (self.rfile.read(n) if n > 0 else b""), ""

            def _send_json_error(self, status: int, message: str) -> None:
                payload = json.dumps({"error": {"code": status, "message": message}}).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                self.wfile.write(payload)
                self.wfile.flush()

            def do_POST(self) -> None:  # noqa: N802
                provided_tok = str(self.headers.get(PROXY_AUTH_HEADER) or "")
                if not provided_tok or not hmac.compare_digest(provided_tok, proxy.secret):
                    self._send_json_error(401, "Invalid local proxy authentication token.")
                    return

                body_status, body_bytes, body_err = self._read_body_checked()
                if body_status != 200:
                    self._send_json_error(body_status, body_err)
                    return

                parsed_url = urllib.parse.urlsplit(self.path)
                m = _VERTEX_MODEL_PATH_RE.search(parsed_url.path)
                req_proj = ""
                req_loc = ""
                if m:
                    req_proj = urllib.parse.unquote(m.group("project"))
                    req_loc = urllib.parse.unquote(m.group("location")).lower()
                    model_id = urllib.parse.unquote(m.group("model"))
                    action = m.group("action")
                else:
                    m_fb = _FALLBACK_MODEL_PATH_RE.search(parsed_url.path)
                    if not m_fb:
                        self._send_json_error(
                            404, f"Unsupported Gemini Enterprise proxy path: {parsed_url.path}"
                        )
                        return
                    model_id = urllib.parse.unquote(m_fb.group("model"))
                    action = m_fb.group("action")

                try:
                    vertex_json = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                    if not isinstance(vertex_json, dict):
                        raise ValueError("JSON body must be an object")
                except Exception as exc:
                    self._send_json_error(400, f"Invalid JSON request body: {exc}")
                    return

                hdr_loc = str(self.headers.get(PROXY_LOCATION_HEADER) or "").strip().lower()
                lookup_loc = hdr_loc or req_loc

                env = proxy._active_env()
                try:
                    access_token = resolve_enterprise_access_token(env=env)
                    ent = proxy.lookup_entitlement(req_proj, lookup_loc)
                    if ent is None:
                        ent = resolve_enterprise_entitlement(
                            access_token,
                            project=req_proj or None,
                            location=lookup_loc or None,
                            location_explicit=False,
                            env=env,
                        )
                        proxy.register_entitlement(ent)
                except EnterpriseAuthError as exc:
                    self._send_json_error(401, str(exc))
                    return
                except Exception as exc:
                    self._send_json_error(500, f"Enterprise license resolution failed: {exc}")
                    return

                baic_payload = to_baic_generate_content_request(
                    vertex_json,
                    model=model_id,
                    user_tier=ent.user_tier,
                )
                encoded_payload = json.dumps(baic_payload).encode("utf-8")
                host = baic_host_for_location(ent.location, env=env)
                safe_p = urllib.parse.quote(ent.project_id, safe="")
                safe_l = urllib.parse.quote(ent.location, safe="")
                is_stream = action == "streamGenerateContent"
                suffix = ":streamGenerateContent?alt=sse" if is_stream else ":generateContent"
                upstream_url = f"{host}/v1beta/projects/{safe_p}/locations/{safe_l}{suffix}"

                for attempt in range(2):
                    req_headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream" if is_stream else "application/json",
                    }
                    if ent.project_id:
                        req_headers["X-Goog-User-Project"] = ent.project_id
                    up_req = urllib.request.Request(
                        upstream_url, data=encoded_payload, headers=req_headers, method="POST"
                    )
                    try:
                        with urllib.request.urlopen(up_req, timeout=600) as up_resp:
                            status = int(up_resp.status)
                            ctype = up_resp.headers.get(
                                "Content-Type",
                                "text/event-stream; charset=utf-8" if is_stream else "application/json",
                            )
                            if is_stream:
                                self.send_response(status)
                                self.send_header("Content-Type", ctype)
                                self.send_header("Cache-Control", "no-cache")
                                self.send_header("Connection", "close")
                                self.end_headers()
                                self.close_connection = True
                                while True:
                                    line = up_resp.readline()
                                    if not line:
                                        break
                                    self.wfile.write(line)
                                    if line in (b"\n", b"\r\n"):
                                        self.wfile.flush()
                                self.wfile.flush()
                            else:
                                data = up_resp.read()
                                self.send_response(status)
                                self.send_header("Content-Type", ctype)
                                self.send_header("Content-Length", str(len(data)))
                                self.send_header("Connection", "close")
                                self.end_headers()
                                self.wfile.write(data)
                                self.wfile.flush()
                            return
                    except urllib.error.HTTPError as http_exc:
                        err_status = int(http_exc.code)
                        try:
                            err_bytes = http_exc.read()
                        except Exception:
                            err_bytes = str(http_exc).encode("utf-8")
                        if err_status == 401 and attempt == 0 and not _clean_env_str(env, "CC_AGY_ENTERPRISE_ACCESS_TOKEN"):
                            try:
                                access_token = resolve_enterprise_access_token(env=env, force_refresh=True)
                                continue
                            except Exception:
                                pass
                        err_ctype = (
                            http_exc.headers.get("Content-Type", "application/json")
                            if http_exc.headers
                            else "application/json"
                        )
                        self.send_response(err_status)
                        self.send_header("Content-Type", err_ctype)
                        self.send_header("Content-Length", str(len(err_bytes)))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.wfile.write(err_bytes)
                        self.wfile.flush()
                        return
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    except Exception as exc:
                        self._send_json_error(502, f"Upstream Gemini Enterprise connection error: {exc}")
                        return

        return _Handler


_proxy_singleton_lock = threading.Lock()
_proxy_singleton: EnterpriseSubscriptionProxy | None = None


def get_or_start_enterprise_proxy() -> EnterpriseSubscriptionProxy:
    """Return the process-wide loopback `EnterpriseSubscriptionProxy`, starting it lazily."""
    global _proxy_singleton
    with _proxy_singleton_lock:
        if _proxy_singleton is None:
            _proxy_singleton = EnterpriseSubscriptionProxy()
            atexit.register(stop_enterprise_proxy)
        return _proxy_singleton


def stop_enterprise_proxy() -> None:
    """Shut down the process-wide `EnterpriseSubscriptionProxy` if running."""
    global _proxy_singleton
    with _proxy_singleton_lock:
        proxy = _proxy_singleton
        _proxy_singleton = None
    if proxy is not None:
        proxy.close()


def prepare_enterprise_endpoint(
    auth_or_request: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve Gemini Enterprise license entitlement and ensure the local loopback proxy is ready."""
    source = env if env is not None else os.environ
    access_token = resolve_enterprise_access_token(env=source)
    req_proj = str(auth_or_request.get("project") or "").strip() or None
    req_loc = str(auth_or_request.get("location") or "").strip().lower() or None
    loc_explicit = bool(auth_or_request.get("location_explicit"))
    entitlement = resolve_enterprise_entitlement(
        access_token,
        project=req_proj,
        location=req_loc,
        location_explicit=loc_explicit,
        env=source,
    )
    proxy = get_or_start_enterprise_proxy()
    proxy.register_entitlement(entitlement)
    return {
        "base_url": proxy.base_url,
        "headers": {
            PROXY_AUTH_HEADER: proxy.secret,
            PROXY_LOCATION_HEADER: entitlement.location,
        },
        "project": entitlement.project_id,
        "location": entitlement.location,
        "user_tier": entitlement.user_tier,
        "tier_display_name": entitlement.tier_display_name,
        "reroute_note": entitlement.reroute_note,
    }
