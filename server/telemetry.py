"""Opt-in non-blocking telemetry receipt sink with verified-identity precedence.

Off by default (`CC_AGY_TELEMETRY_SINK` unset). When enabled, emits sanitized, grounded
receipt summaries (token counts, USD cents, duration, model, verification mode, hashed
workspace ID, and caller identity provenance) to either:
  - a local append-only NDJSON file (`0o600`, leaf-symlink-guarded), or
  - an `https://` collector endpoint (or loopback `http://127.0.0.1` / `localhost` / `[::1]`)
    via a bounded background queue so model/receipt execution never blocks on telemetry I/O.

Never emits prompts, diffs, raw workspace paths, environment variables, secrets, or
LLM-estimated human-hour guesses.
"""

from __future__ import annotations

import atexit
import base64
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import re
import socket
import stat
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from server.regions import first_valid_env, peek_cached_oidc_token

_MAX_QUEUE_SIZE = 256
_MAX_DEDUP_KEYS = 4096
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_NUMERIC_IPV4_SHORTHAND_RE = re.compile(r"^(?:0x[0-9a-f]+|0[0-7]*|[1-9]\d*)(?:\.(?:0x[0-9a-f]+|0[0-7]*|\d+)){0,3}$")
_NAT64_PREFIX = ipaddress.IPv6Network("64:ff9b::/96")
_BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
        "metadata",
        "169.254.169.254",
        "fd00:ec2::254",
    }
)
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

_USAGE_KEYS: tuple[str, ...] = (
    "prompt_token_count",
    "cached_content_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)
_FILE_SINK_LOCK = threading.Lock()
_DEDUP_LOCK = threading.Lock()
_EMITTED_KEYS: OrderedDict[tuple[str, str, float, int], None] = OrderedDict()
_EMITTED_JOB_SPEND: OrderedDict[str, tuple[float, dict[str, int], str]] = OrderedDict()
_JOB_LOCKS: OrderedDict[str, threading.Lock] = OrderedDict()

_WEBHOOK_LOCK = threading.Lock()
_WEBHOOK_QUEUE: queue.Queue[tuple[str, bytes, dict[str, str]] | None] = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
_WEBHOOK_THREAD: threading.Thread | None = None


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject HTTP redirects on telemetry webhook POSTs to prevent SSRF redirect hops."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        raise urllib.error.HTTPError(req.full_url, code, f"Redirects disabled for telemetry sink ({code})", headers, fp)


def resolve_telemetry_sink(env: Mapping[str, str] | None = None) -> str | None:
    """Resolve `CC_AGY_TELEMETRY_SINK` (or `userConfig.telemetry_sink`), returning `None` when unset."""
    source = env if env is not None else os.environ
    raw = first_valid_env(
        source,
        "CC_AGY_TELEMETRY_SINK",
        "CLAUDE_PLUGIN_OPTION_TELEMETRY_SINK",
        "CC_AGY_USERCFG_TELEMETRY_SINK",
        default="",
    )
    if not raw or raw.lower() in ("off", "false", "0", "none", "null", "disabled"):
        return None
    return raw


def resolve_telemetry_token(env: Mapping[str, str] | None = None) -> str | None:
    """Resolve optional `CC_AGY_TELEMETRY_TOKEN` bearer secret for HTTPS webhook sinks."""
    source = env if env is not None else os.environ
    raw = first_valid_env(
        source,
        "CC_AGY_TELEMETRY_TOKEN",
        "CLAUDE_PLUGIN_OPTION_TELEMETRY_TOKEN",
        "CC_AGY_USERCFG_TELEMETRY_TOKEN",
        default="",
    )
    if not raw or raw.lower() in ("none", "null", "dummy", "<api-key>"):
        return None
    return raw


def _b64url_decode_bytes(segment: str) -> bytes | None:
    if not segment or not re.fullmatch(r"[A-Za-z0-9_-]+", segment):
        return None
    padded = segment + ("=" * ((4 - (len(segment) % 4)) % 4))
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, TypeError):
        return None


def _sanitize_identity_value(raw: str | None) -> str | None:
    if not isinstance(raw, str):
        return None
    cleaned = _CONTROL_CHAR_RE.sub("", raw).strip()
    if not cleaned or cleaned.startswith("${") or cleaned.lower() in ("none", "null", "anonymous", "unknown"):
        return None
    return cleaned[:128]


def _extract_oidc_jwt_claim(token: str | None, *, now: float | None = None) -> str | None:
    """Extract `email` or `sub` from a non-expired, non-`alg:none` 3-part JWT with mandatory `exp`."""
    if not isinstance(token, str):
        return None
    cleaned = token.strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()
    parts = cleaned.split(".")
    if len(parts) != 3 or not all(parts):
        return None
    hdr_bytes = _b64url_decode_bytes(parts[0])
    payload_bytes = _b64url_decode_bytes(parts[1])
    sig_bytes = _b64url_decode_bytes(parts[2])
    if hdr_bytes is None or payload_bytes is None or sig_bytes is None or len(sig_bytes) < 16:
        return None
    try:
        hdr = json.loads(hdr_bytes.decode("utf-8"))
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(hdr, dict) or not isinstance(payload, dict):
        return None
    alg = str(hdr.get("alg") or "").strip()
    if not alg or alg.lower() == "none":
        return None
    if "exp" not in payload:
        return None
    ref_now = time.time() if now is None else float(now)
    exp = payload["exp"]
    if (
        isinstance(exp, bool)
        or not isinstance(exp, (int, float))
        or not math.isfinite(float(exp))
        or float(exp) <= ref_now
    ):
        return None
    if "nbf" in payload:
        nbf = payload["nbf"]
        if (
            isinstance(nbf, bool)
            or not isinstance(nbf, (int, float))
            or not math.isfinite(float(nbf))
            or float(nbf) > (ref_now + 60.0)
        ):
            return None
    for key in ("email", "sub"):
        claim = _sanitize_identity_value(payload.get(key) if isinstance(payload.get(key), str) else None)
        if claim:
            return claim
    return None


def _header_case_insensitive(headers: Mapping[str, str] | None, target: str) -> str | None:
    if not headers:
        return None
    low_target = target.lower()
    for k, v in headers.items():
        if str(k).lower() == low_target and isinstance(v, str):
            return v
    return None


def resolve_telemetry_identity(
    *,
    env: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    bearer_token: str | None = None,
    jwt_signature_verified: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """Resolve caller attribution using strict trust-tier precedence.

    Precedence:
      1. `X-Goog-Authenticated-User-Email` proxy header when HTTP gateway request `headers` are supplied
         (`identity_source='iap_header'`, `identity_verified=True`; local stdio `ReceiptStore` writes pass
         `headers=None` and resolve via tiers 2-4).
      2. Signed non-`alg:none` non-expired OIDC JWT `email`/`sub` with mandatory `exp` (`identity_source='oidc_jwt'`,
         `identity_verified=bool(jwt_signature_verified)`).
      3. Self-reported `X-User-Id` header or `CC_AGY_USER_ID` env var (`identity_source='self_reported'`,
         `identity_verified=False`).
      4. `'anonymous'` (`identity_source='anonymous'`, `identity_verified=False`).
    """
    source_env = env if env is not None else os.environ
    iap_raw = _header_case_insensitive(headers, "X-Goog-Authenticated-User-Email")
    if iap_raw:
        stripped = iap_raw.strip()
        if ":" in stripped and stripped.lower().startswith("accounts.google.com:"):
            stripped = stripped.split(":", 1)[1]
        iap_user = _sanitize_identity_value(stripped)
        if iap_user:
            return {
                "user_id": iap_user,
                "identity_source": "iap_header",
                "identity_verified": True,
            }

    candidate_jwt = (
        bearer_token
        or _header_case_insensitive(headers, "Authorization")
        or first_valid_env(
            source_env,
            "CC_AGY_VERTEX_BASE_URL_TOKEN",
            "CLAUDE_PLUGIN_OPTION_VERTEX_BASE_URL_TOKEN",
            "CC_AGY_USERCFG_VERTEX_BASE_URL_TOKEN",
            default="",
        )
        or peek_cached_oidc_token(env=source_env)
    )
    jwt_claim = _extract_oidc_jwt_claim(candidate_jwt, now=now)
    if jwt_claim:
        return {
            "user_id": jwt_claim,
            "identity_source": "oidc_jwt",
            "identity_verified": bool(jwt_signature_verified),
        }

    self_reported = _sanitize_identity_value(
        _header_case_insensitive(headers, "X-User-Id")
        or first_valid_env(source_env, "CC_AGY_USER_ID", default="")
    )
    if self_reported:
        return {
            "user_id": self_reported,
            "identity_source": "self_reported",
            "identity_verified": False,
        }

    return {
        "user_id": "anonymous",
        "identity_source": "anonymous",
        "identity_verified": False,
    }


def _workspace_hash(workspace_root: Any) -> str | None:
    if not isinstance(workspace_root, str) or not workspace_root.strip():
        return None
    raw = workspace_root.strip()
    if raw == "<redacted>":
        return None
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _safe_nonneg_int(val: Any) -> int:
    if isinstance(val, bool) or val is None:
        return 0
    try:
        parsed = float(val)
        if not math.isfinite(parsed):
            return 0
        return max(0, int(parsed))
    except (TypeError, ValueError, OverflowError):
        return 0


def build_telemetry_event(
    receipt: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    bearer_token: str | None = None,
) -> dict[str, Any]:
    """Construct a sanitized, secret-free telemetry event from a persisted receipt."""
    usage_raw = receipt.get("usage") if isinstance(receipt.get("usage"), Mapping) else {}
    usage = {k: _safe_nonneg_int(usage_raw.get(k)) for k in _USAGE_KEYS}
    cost_raw = receipt.get("cost") if isinstance(receipt.get("cost"), Mapping) else {}
    cost_cents_val = receipt.get("cost_cents")
    if cost_cents_val is None:
        cost_cents_val = cost_raw.get("total_cost_cents", 0.0)
    try:
        parsed_cost = float(cost_cents_val or 0.0)
        cost_cents = (
            0.0
            if isinstance(cost_cents_val, bool) or not math.isfinite(parsed_cost)
            else round(max(0.0, parsed_cost), 6)
        )
    except (TypeError, ValueError):
        cost_cents = 0.0

    dur_val = receipt.get("duration_seconds")
    try:
        parsed_dur = float(dur_val or 0.0)
        duration_s = (
            0.0
            if isinstance(dur_val, bool) or not math.isfinite(parsed_dur)
            else round(max(0.0, parsed_dur), 4)
        )
    except (TypeError, ValueError):
        duration_s = 0.0

    created_raw = receipt.get("created_at_epoch")
    created_at_epoch: float | None = None
    if created_raw is not None and not isinstance(created_raw, bool):
        try:
            parsed_created = float(created_raw)
            if math.isfinite(parsed_created) and parsed_created > 0.0:
                created_at_epoch = round(parsed_created, 3)
        except (TypeError, ValueError):
            created_at_epoch = None

    ident = resolve_telemetry_identity(
        env=env,
        headers=headers,
        bearer_token=bearer_token,
    )
    ver_mode = receipt.get("verification_mode")
    if not isinstance(ver_mode, str) or not ver_mode.strip():
        ver_dict = receipt.get("verification") if isinstance(receipt.get("verification"), Mapping) else {}
        ver_mode = str(ver_dict.get("verification_mode") or "none")

    now_ts = time.time()
    iso_timestamp = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    job_id_str = str(receipt.get("job_id") or "").strip()
    status_str = str(receipt.get("status") or "").strip()
    event_id_seed = (
        f"{job_id_str}|{status_str}|{cost_cents:.6f}|{json.dumps(usage, sort_keys=True)}"
        if job_id_str
        else f"anon|{now_ts:.6f}|{status_str}|{cost_cents:.6f}|{json.dumps(usage, sort_keys=True)}"
    )
    event_id = hashlib.sha256(event_id_seed.encode("utf-8")).hexdigest()[:24]
    return {
        "schema_version": 1,
        "event_type": "cc_agy_receipt",
        "event_id": event_id,
        "timestamp": iso_timestamp,
        "emitted_at_epoch": round(now_ts, 3),
        "created_at_epoch": created_at_epoch,
        "job_id": job_id_str,
        "kind": str(receipt.get("kind") or ""),
        "status": status_str,
        "auth_mode": str(receipt.get("auth_mode") or ""),
        "model": str(receipt.get("model") or ""),
        "vertex_location": receipt.get("vertex_location"),
        "thinking_level": receipt.get("thinking_level"),
        "duration_seconds": duration_s,
        "usage": usage,
        "cumulative_usage": dict(usage),
        "cost_cents": cost_cents,
        "cumulative_cost_cents": cost_cents,
        "pricing_known": bool(cost_raw.get("pricing_known", receipt.get("pricing_known", True))),
        "verification_mode": ver_mode,
        "workspace_hash": _workspace_hash(receipt.get("workspace_root")),
        "user_id": ident["user_id"],
        "identity_source": ident["identity_source"],
        "identity_verified": ident["identity_verified"],
    }


def _parse_ip_or_shorthand(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    if _NUMERIC_IPV4_SHORTHAND_RE.fullmatch(host):
        try:
            packed = socket.inet_aton(host)
            return ipaddress.IPv4Address(packed)
        except OSError:
            return None
    return None


def _unwrap_embedded_ipv4(
    ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    if isinstance(ip_obj, ipaddress.IPv6Address):
        if ip_obj.ipv4_mapped is not None:
            return ip_obj.ipv4_mapped
        if ip_obj in _NAT64_PREFIX:
            return ipaddress.IPv4Address(ip_obj.packed[-4:])
        ip_int = int(ip_obj)
        if (ip_int >> 32) == 0 and ip_int > 1:
            return ipaddress.IPv4Address(ip_obj.packed[-4:])
    return ip_obj


def _is_blocked_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    unwrapped = _unwrap_embedded_ipv4(ip_obj)
    for cand in (ip_obj, unwrapped):
        if (
            cand.is_link_local
            or cand.is_unspecified
            or cand.is_multicast
            or str(cand).lower() in _BLOCKED_HOSTNAMES
        ):
            return True
    return False


def validate_webhook_sink_url(url: str, *, check_dns: bool = False) -> str:
    """Validate an HTTPS (or loopback HTTP) telemetry webhook URL and block metadata/link-local SSRF."""
    cleaned = (url or "").strip()
    parsed = urllib.parse.urlsplit(cleaned)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("https", "http"):
        raise ValueError(f"Unsupported telemetry webhook scheme {scheme!r}; expected 'https://' or loopback 'http://'")
    if parsed.username or parsed.password:
        raise ValueError("Telemetry sink URL must not embed credentials in userinfo; use CC_AGY_TELEMETRY_TOKEN instead")
    raw_host = (parsed.hostname or "").strip().lower()
    host = raw_host.rstrip(".")
    if not host:
        raise ValueError("Telemetry sink URL is missing a hostname")
    if host in _BLOCKED_HOSTNAMES or (host.endswith(".internal") and host.startswith("metadata.")):
        raise ValueError(f"Telemetry sink host {host!r} is blocked (cloud metadata endpoint)")

    is_loopback = host in _LOOPBACK_HOSTS
    ip_obj = _parse_ip_or_shorthand(host)

    if ip_obj is not None:
        if _is_blocked_ip(ip_obj):
            raise ValueError(f"Telemetry sink IP {host!r} is blocked (metadata, link-local, unspecified, or multicast)")
        unwrapped_ip = _unwrap_embedded_ipv4(ip_obj)
        if ip_obj.is_loopback or unwrapped_ip.is_loopback:
            is_loopback = True

    if scheme == "http" and not is_loopback:
        raise ValueError(
            f"Plain HTTP telemetry sink {cleaned!r} is only permitted on loopback (127.0.0.1, localhost, [::1]); use https://"
        )

    if check_dns and ip_obj is None and not is_loopback:
        try:
            infos = socket.getaddrinfo(
                host,
                parsed.port or (443 if scheme == "https" else 80),
                proto=socket.IPPROTO_TCP,
            )
        except socket.gaierror as exc:
            raise ValueError(f"Telemetry sink hostname {host!r} failed DNS resolution: {exc}") from exc
        if not infos:
            raise ValueError(f"Telemetry sink hostname {host!r} returned no DNS addresses")
        for info in infos:
            sockaddr = info[4]
            if not sockaddr:
                continue
            resolved_ip = ipaddress.ip_address(sockaddr[0])
            unwrapped_res = _unwrap_embedded_ipv4(resolved_ip)
            if _is_blocked_ip(resolved_ip) or resolved_ip.is_loopback or unwrapped_res.is_loopback:
                raise ValueError(
                    f"Telemetry sink {host!r} resolved to blocked or loopback address {resolved_ip}"
                )

    return cleaned


def _resolve_local_sink_path(sink: str) -> Path:
    cleaned = (sink or "").strip()
    if cleaned.startswith(("\\\\", "//")):
        raise ValueError(f"UNC or network share telemetry sink paths are not allowed (got {sink!r})")
    if cleaned.lower().startswith("file://"):
        parsed = urllib.parse.urlsplit(cleaned)
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            raise ValueError(f"UNC or remote file:// telemetry sink must be local (got host {parsed.netloc!r})")
        raw_path = urllib.parse.unquote(parsed.path or "")
        if raw_path.startswith("//"):
            raise ValueError(f"UNC or network share telemetry sink paths are not allowed (got {sink!r})")
        if os.name == "nt" and re.match(r"^/[A-Za-z]:/", raw_path):
            raw_path = raw_path[1:]
        path = Path(raw_path).expanduser()
    else:
        path = Path(cleaned).expanduser()
    drive, _ = os.path.splitdrive(str(path))
    if drive.startswith(("\\\\", "//")):
        raise ValueError(f"UNC or network share telemetry sink paths are not allowed (got {sink!r})")
    if not path.is_absolute():
        raise ValueError(
            f"Local telemetry file sink must be an absolute path, '~/...', or 'file:///...' (got {sink!r})"
        )
    if path.is_symlink():
        raise ValueError(f"Refusing to write telemetry to symlink target: {path}")
    return path


def validate_telemetry_sink(sink: str, *, check_dns: bool = False) -> str:
    """Validate `CC_AGY_TELEMETRY_SINK` (either an HTTP(S) webhook URL or a local NDJSON file path)."""
    cleaned = (sink or "").strip()
    if not cleaned:
        raise ValueError("Telemetry sink must not be empty")
    if cleaned.lower().startswith(("https://", "http://")):
        return validate_webhook_sink_url(cleaned, check_dns=check_dns)
    resolved = _resolve_local_sink_path(cleaned)
    return str(resolved)


def _append_ndjson_file(path: Path, line_bytes: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing to write telemetry to symlink target: {path}")
    parent_dir = path.parent.resolve(strict=False)
    parent_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    target_path = parent_dir / path.name
    if target_path.is_symlink():
        raise ValueError(f"Refusing to write telemetry to symlink target: {target_path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    with _FILE_SINK_LOCK:
        fd = os.open(str(target_path), flags, 0o600)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ValueError(f"Telemetry sink target is not a regular file: {target_path}")
            try:
                os.fchmod(fd, 0o600)
            except (AttributeError, OSError):
                pass
            os.write(fd, line_bytes)
        finally:
            os.close(fd)


def _post_webhook_once(url: str, body: bytes, headers: Mapping[str, str], *, timeout: float = 3.0) -> None:
    validate_webhook_sink_url(url, check_dns=True)
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    opener = urllib.request.build_opener(_NoRedirectHandler())
    with opener.open(req, timeout=timeout) as resp:
        resp.read(256)


def _webhook_worker_loop() -> None:
    while True:
        item = _WEBHOOK_QUEUE.get()
        try:
            if item is None:
                return
            url, body, headers = item
            try:
                _post_webhook_once(url, body, headers)
            except Exception:  # noqa: BLE001 - telemetry must never crash the host process
                pass
        finally:
            _WEBHOOK_QUEUE.task_done()


def _ensure_webhook_worker() -> None:
    global _WEBHOOK_THREAD
    with _WEBHOOK_LOCK:
        if _WEBHOOK_THREAD is None or not _WEBHOOK_THREAD.is_alive():
            _WEBHOOK_THREAD = threading.Thread(
                target=_webhook_worker_loop,
                name="cc-agy-telemetry-sink",
                daemon=True,
            )
            _WEBHOOK_THREAD.start()


def flush_telemetry(timeout: float = 2.0) -> bool:
    """Wait up to `timeout` seconds for queued background webhook telemetry events to drain."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    while _WEBHOOK_QUEUE.unfinished_tasks > 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    return _WEBHOOK_QUEUE.unfinished_tasks == 0


atexit.register(flush_telemetry, 1.5)


def clear_telemetry_dedup_cache() -> None:
    """Clear in-memory telemetry deduplication state (used by unit tests)."""
    with _DEDUP_LOCK:
        _EMITTED_KEYS.clear()
        _EMITTED_JOB_SPEND.clear()
        _JOB_LOCKS.clear()


def _get_job_lock(job_id: str) -> threading.Lock:
    with _DEDUP_LOCK:
        lock = _JOB_LOCKS.get(job_id)
        if lock is not None:
            _JOB_LOCKS.move_to_end(job_id)
            return lock
        if len(_JOB_LOCKS) >= _MAX_DEDUP_KEYS:
            _JOB_LOCKS.popitem(last=False)
        lock = threading.Lock()
        _JOB_LOCKS[job_id] = lock
        return lock


def _recover_file_sink_job_state(
    sink: str, job_id: str
) -> tuple[float, dict[str, int], str] | None:
    """Recover `(cumulative_cost_cents, cumulative_usage, last_status)` for `job_id` from a local NDJSON sink tail."""
    if not job_id or sink.lower().startswith(("https://", "http://")):
        return None
    sink_path = _resolve_local_sink_path(sink)
    if not sink_path.exists():
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(sink_path), flags)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size <= 0:
            return None
        tail_offset = max(0, st.st_size - 262_144)
        if tail_offset > 0:
            os.lseek(fd, tail_offset, os.SEEK_SET)
        raw = os.read(fd, 262_144).decode("utf-8", errors="replace")
    except OSError:
        return None
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    lines = raw.splitlines()
    if tail_offset > 0 and lines:
        lines = lines[1:]
    matched = False
    max_cost = 0.0
    max_usage = {k: 0 for k in _USAGE_KEYS}
    last_status = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or job_id not in stripped:
            continue
        try:
            row = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(row, dict) or str(row.get("job_id") or "").strip() != job_id:
            continue
        matched = True
        row_cum_cost = row.get("cumulative_cost_cents", row.get("cost_cents", 0.0))
        try:
            parsed_row_cost = float(row_cum_cost or 0.0)
            if math.isfinite(parsed_row_cost) and parsed_row_cost >= 0.0:
                max_cost = max(max_cost, parsed_row_cost)
        except (TypeError, ValueError):
            pass
        u_obj = row.get("cumulative_usage")
        if not isinstance(u_obj, Mapping):
            u_obj = row.get("usage")
        if isinstance(u_obj, Mapping):
            for k in _USAGE_KEYS:
                raw_u = u_obj.get(k, 0)
                if isinstance(raw_u, bool):
                    continue
                try:
                    parsed_u = int(raw_u or 0)
                    if parsed_u >= 0:
                        max_usage[k] = max(max_usage[k], parsed_u)
                except (TypeError, ValueError, OverflowError):
                    pass
        last_status = str(row.get("status") or "").strip().lower()
    if not matched:
        return None
    return (round(max_cost, 6), max_usage, last_status)


def _deliver_sink_payload(sink: str, line_bytes: bytes, env: Mapping[str, str] | None) -> bool:
    low_sink = sink.lower()
    if low_sink.startswith(("https://", "http://")):
        validate_webhook_sink_url(sink, check_dns=False)
        req_headers = {
            "Content-Type": "application/x-ndjson; charset=utf-8",
            "User-Agent": "cc-agy-plugin-telemetry/1.0",
        }
        tok = resolve_telemetry_token(env)
        if tok:
            req_headers["Authorization"] = f"Bearer {tok}"
        _ensure_webhook_worker()
        try:
            _WEBHOOK_QUEUE.put_nowait((sink, line_bytes, req_headers))
        except queue.Full:
            return False
        return True
    sink_path = _resolve_local_sink_path(sink)
    _append_ndjson_file(sink_path, line_bytes)
    return True


def emit_receipt_telemetry(
    receipt: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    bearer_token: str | None = None,
) -> bool:
    """Emit a completed receipt to `CC_AGY_TELEMETRY_SINK` when configured.

    Returns `True` when an event was written or enqueued, or `False` when disabled, intermediate
    (`status` in `{'running', 'starting', 'queued'}`), deduplicated across lifecycle rewrites
    (`delegate_apply` / `delegate_undo`), or dropped when the webhook queue is full.
    Raises `ValueError` or `OSError` when the configured sink URL/path is invalid; callers
    such as `ReceiptStore.write_receipt` wrap this call in a best-effort `try/except`.
    """
    sink = resolve_telemetry_sink(env)
    if not sink:
        return False

    status = str(receipt.get("status") or "").strip().lower()
    if not status or status in ("running", "starting", "queued"):
        return False

    job_id = str(receipt.get("job_id") or "").strip()
    event = build_telemetry_event(receipt, env=env, headers=headers, bearer_token=bearer_token)
    total_cost_cents = round(float(event["cost_cents"]), 6)
    cum_usage = {k: int(event["usage"][k]) for k in _USAGE_KEYS}
    total_tokens = cum_usage["total_token_count"]
    dedup_key = (job_id, status, total_cost_cents, total_tokens)

    if not job_id:
        line_bytes = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
        return _deliver_sink_payload(sink, line_bytes, env)

    job_lock = _get_job_lock(job_id)
    with job_lock:
        with _DEDUP_LOCK:
            if dedup_key in _EMITTED_KEYS:
                return False
            prev_state = _EMITTED_JOB_SPEND.get(job_id)

        if prev_state is None:
            prev_state = _recover_file_sink_job_state(sink, job_id)

        next_cost = total_cost_cents
        next_usage = dict(cum_usage)
        if prev_state is not None:
            prev_cost, prev_usage, prev_status = prev_state
            delta_cost = round(max(0.0, total_cost_cents - prev_cost), 6)
            delta_usage = {
                k: max(0, cum_usage[k] - int(prev_usage.get(k, 0)))
                for k in _USAGE_KEYS
            }
            if delta_cost == 0.0 and not any(delta_usage.values()) and status == prev_status:
                with _DEDUP_LOCK:
                    if len(_EMITTED_KEYS) >= _MAX_DEDUP_KEYS and dedup_key not in _EMITTED_KEYS:
                        _EMITTED_KEYS.popitem(last=False)
                    _EMITTED_KEYS[dedup_key] = None
                    if len(_EMITTED_JOB_SPEND) >= _MAX_DEDUP_KEYS and job_id not in _EMITTED_JOB_SPEND:
                        _EMITTED_JOB_SPEND.popitem(last=False)
                    _EMITTED_JOB_SPEND[job_id] = (prev_cost, dict(prev_usage), prev_status)
                return False
            event["cumulative_cost_cents"] = total_cost_cents
            event["cost_cents"] = delta_cost
            event["cumulative_usage"] = dict(cum_usage)
            event["usage"] = delta_usage
            next_cost = max(prev_cost, total_cost_cents)
            next_usage = {
                k: max(int(prev_usage.get(k, 0)), cum_usage[k])
                for k in _USAGE_KEYS
            }

        line_bytes = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
        if not _deliver_sink_payload(sink, line_bytes, env):
            return False

        with _DEDUP_LOCK:
            if len(_EMITTED_KEYS) >= _MAX_DEDUP_KEYS and dedup_key not in _EMITTED_KEYS:
                _EMITTED_KEYS.popitem(last=False)
            _EMITTED_KEYS[dedup_key] = None
            if len(_EMITTED_JOB_SPEND) >= _MAX_DEDUP_KEYS and job_id not in _EMITTED_JOB_SPEND:
                _EMITTED_JOB_SPEND.popitem(last=False)
            _EMITTED_JOB_SPEND[job_id] = (next_cost, next_usage, status)
        return True
