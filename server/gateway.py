#!/usr/bin/env python3
"""Optional local Anthropic Messages API gateway for cc-agy-plugin (`127.0.0.1:3456`).

Listens on `127.0.0.1:3456` (override with `CC_AGY_GATEWAY_PORT`, `GEMINI4_GATEWAY_PORT`, or `--port`)
for opt-in `ANTHROPIC_BASE_URL=http://127.0.0.1:3456` setups:
  - Implements `POST /v1/messages`, `POST /v1/messages/count_tokens`, `GET /v1/models`, and `GET /health`.
  - Routes requests across `gemini-3.8-flash` (`gemini-flash-next`), `gemini-4-argon`, and `gemini-4-carbon`
    (never `gemini-3.5-flash-lite` by default) or passes through to `api.anthropic.com` when `ANTHROPIC_UPSTREAM_KEY`
    is set.
  - Translates Anthropic `messages`/`tools`/`tool_use`/`tool_result` payloads to Gemini `generateContent`
    and `streamGenerateContent?alt=sse`, including `BoundedLRU` `thoughtSignature` replay, SSE `ping`
    keep-alives, and sequential `continuationToken` multi-segment long-decode for `gemini-4-carbon`.
  - Enforces strict `127.0.0.1` / `localhost` / `[::1]` `Host` validation, rejects any `Origin` header
    (DNS-rebinding protection), and requires a mode-`0600` `O_NOFOLLOW` gateway token on all non-health routes.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable

from server.genai_ask import MAX_CONTINUATIONS, call_gemini_long_decode

VERSION = "2.3.0"
GATEWAY_PORT = int(
    os.environ.get("CC_AGY_GATEWAY_PORT")
    or os.environ.get("GEMINI4_GATEWAY_PORT")
    or "3456"
)

MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_TOKENS = 65536
DEFAULT_MAX_OUTPUT_TOKENS = 8192
SESSION_MAP_MAX_ENTRIES = 2048
THOUGHT_VAULT_MAX_ENTRIES = 2048
PING_INTERVAL_S = 15.0
VERTEX_TOKEN_TTL_S = 45 * 60
MIN_TOKEN_LENGTH = 32

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_TIMEOUT_S = 600
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"

MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]{0,63}$")
GCP_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,62}$")

# Active Gemini routing models (gemini-3.5-flash-lite is intentionally excluded from active routing).
KNOWN_GEMINI_MODELS = frozenset(
    {
        "gemini-4-carbon",
        "gemini-4-argon",
        "gemini-3.8-flash",
        "gemini-flash-next",
    }
)
PRO_CLASS_MODELS = frozenset({"gemini-4-carbon", "gemini-4-argon"})
LONG_DECODE_MODELS = frozenset({"gemini-4-carbon"})
LONG_DECODE_SEGMENT_TOKENS = 65536

NOT_IMPLEMENTED_PREFIXES = (
    "/v1/responses",
    "/v1/chat/completions",
    "/v1/completions",
    "/cursor",
    "/openai/",
)
NOT_IMPLEMENTED_BODY = {
    "error": {
        "type": "not_implemented",
        "message": (
            "This gateway only implements the Anthropic Messages API (/v1/messages). "
            "For Codex CLI and Cursor, use the cc-agy-plugin MCP server."
        ),
    }
}


class BoundedLRU:
    """Thread-safe OrderedDict with a maximum size; least recently used entries are evicted."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max_entries
        self.lock = threading.RLock()
        self._data: collections.OrderedDict[str, Any] = collections.OrderedDict()

    def get(self, key: str, default: Any = None) -> Any:
        with self.lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self.lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def pop(self, key: str, default: Any = None) -> Any:
        with self.lock:
            return self._data.pop(key, default)

    def clear(self) -> None:
        with self.lock:
            self._data.clear()

    def __contains__(self, key: object) -> bool:
        with self.lock:
            return key in self._data

    def __len__(self) -> int:
        with self.lock:
            return len(self._data)


SESSION_PINS = BoundedLRU(SESSION_MAP_MAX_ENTRIES)
SESSION_SPEND_USD = BoundedLRU(SESSION_MAP_MAX_ENTRIES)
THOUGHT_SIGNATURE_VAULT = BoundedLRU(THOUGHT_VAULT_MAX_ENTRIES)

BUDGET_LOCK = threading.Lock()
GLOBAL_SPEND_USD = {"settled": 0.0, "reserved": 0.0}

SECURITY_RE = re.compile(
    r"\b(cve-\d+|cwe-\d+|security\s+audit|vulnerability|auth\s+bypass|sql\s+injection|"
    r"ssrf|xss|race\s+condition|deadlock|taint|cryptograph|privilege\s+escalation)\b",
    re.IGNORECASE,
)
LONG_DECODE_RE = re.compile(
    r"\b(migrate\s+all|full\s+test\s+suite|refactor\s+across|rewrite\s+module|"
    r"monorepo\s+scan|64k|boilerplate\s+for|generate\s+complete|architecture\s+synthesis)\b",
    re.IGNORECASE,
)

DEFAULT_PRICES_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gemini-4-carbon": (2.00, 0.20, 12.00),
    "gemini-4-argon": (2.00, 0.20, 12.00),
    "gemini-3.8-flash": (0.50, 0.05, 3.00),
    "gemini-flash-next": (0.50, 0.05, 3.00),
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "claude-sonnet-5-5": (2.00, 0.20, 10.00),
}
FALLBACK_PRICE_PER_MTOK = (4.00, 0.40, 20.00)

GATEWAY_MODEL_CATALOG: list[dict[str, Any]] = [
    {"id": "claude-gemini-4-carbon", "display_name": "Gemini 4 Carbon", "created_at": "2026-11-10T00:00:00Z", "type": "model"},
    {"id": "claude-gemini-4-argon", "display_name": "Gemini 4 Argon", "created_at": "2026-10-13T00:00:00Z", "type": "model"},
    {"id": "claude-gemini-3.8-flash", "display_name": "Gemini 3.8 Flash", "created_at": "2026-09-01T00:00:00Z", "type": "model"},
    {"id": "claude-auto-gemini", "display_name": "Gemini auto router (flash / argon / carbon)", "created_at": "2026-11-01T00:00:00Z", "type": "model"},
    {"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5 (Anthropic passthrough)", "created_at": "2026-09-15T00:00:00Z", "type": "model"},
    {"id": "claude-sonnet-5-5", "display_name": "Claude Sonnet 5.5 (Anthropic passthrough)", "created_at": "2026-09-15T00:00:00Z", "type": "model"},
]


class GatewayRequestError(Exception):
    """Error that maps to an Anthropic-format error response."""

    def __init__(self, status: int, err_type: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.err_type = err_type
        self.message = message


def anthropic_error_body(err_type: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": err_type, "message": message}}


def session_digest(session_id: Any) -> str:
    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:12]


def log_event(event: str, **fields: Any) -> None:
    """Write one JSON log line to stderr without logging prompt text."""
    if (os.environ.get("CC_AGY_GATEWAY_LOG") or os.environ.get("GEMINI4_GATEWAY_LOG") or "1").strip() == "0":
        return
    allowed = (
        "session",
        "requested",
        "model",
        "rule",
        "status",
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "continuations",
    )
    record: dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event}
    for key in allowed:
        if fields.get(key) is not None:
            record[key] = fields[key]
    try:
        sys.stderr.write(json.dumps(record) + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def token_file_path() -> str:
    override = (os.environ.get("CC_AGY_GATEWAY_TOKEN_FILE") or "").strip()
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".config", "cc-agy-plugin", "gateway-token")


def load_or_create_token() -> tuple[str, str | None, bool]:
    """Return `(token, file_path_or_None, created)` using O_NOFOLLOW + 0600 permissions."""
    env_token = (
        os.environ.get("CC_AGY_GATEWAY_TOKEN")
        or os.environ.get("GEMINI4_GATEWAY_TOKEN")
        or ""
    ).strip()
    if env_token:
        if len(env_token) < MIN_TOKEN_LENGTH:
            raise RuntimeError(f"Gateway token must be at least {MIN_TOKEN_LENGTH} characters.")
        return env_token, None, False
    path = token_file_path()
    for _ in range(2):
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            fd = None
        except OSError as exc:
            raise RuntimeError(
                f"Refusing to use gateway token file {path} ({exc.strerror}); delete it to regenerate."
            ) from exc
        if fd is not None:
            with os.fdopen(fd, "r", encoding="utf-8") as fh:
                st = os.fstat(fh.fileno())
                if hasattr(os, "getuid") and (st.st_uid != os.getuid() or (st.st_mode & 0o077)):
                    raise RuntimeError(
                        f"Gateway token file {path} must be owned by you with mode 0600 (run: chmod 600 {path})."
                    )
                token = fh.read().strip()
            if len(token) < MIN_TOKEN_LENGTH:
                raise RuntimeError(f"Gateway token file {path} is empty or too short; delete it to regenerate.")
            return token, path, False
        parent = os.path.dirname(path)
        os.makedirs(parent, mode=0o700, exist_ok=True)
        try:
            os.chmod(parent, 0o700)
        except OSError:
            pass
        token = secrets.token_urlsafe(32)
        try:
            fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
        return token, path, True
    raise RuntimeError(f"Could not read or create gateway token file {path}.")


def token_matches(presented: str, expected: str) -> bool:
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def strip_virtual_wrappers(model_id: str) -> str:
    cleaned = re.sub(r"\[[^\]]*\]", "", (model_id or "").strip())
    lower = cleaned.lower()
    if lower.startswith("claude-gemini-"):
        return cleaned[len("claude-") :]
    if lower.startswith("litellm-"):
        return cleaned[len("litellm-") :]
    return cleaned


def normalize_model_id(model_id: str) -> str:
    raw = strip_virtual_wrappers(model_id).strip().lower()
    if raw == "carbon":
        return "gemini-4-carbon"
    if raw == "argon":
        return "gemini-4-argon"
    return raw


def is_valid_model_id(model_id: str) -> bool:
    return bool(MODEL_ID_RE.match(model_id or ""))


def _security_model() -> str | None:
    raw = (
        os.environ.get("CC_AGY_SECURITY_MODEL")
        or os.environ.get("GEMINI4_SECURITY_MODEL")
        or ""
    ).strip()
    if not raw:
        return None
    if raw.lower().startswith("gemini/"):
        raw = raw[len("gemini/") :]
    return normalize_model_id(raw)


def allowed_gemini_models() -> frozenset[str]:
    extra = _security_model()
    if extra and is_valid_model_id(extra):
        return KNOWN_GEMINI_MODELS | {extra}
    return KNOWN_GEMINI_MODELS


def resolve_security_target() -> str:
    override = _security_model()
    if override and is_valid_model_id(override):
        return f"gemini/{override}"
    return "gemini/gemini-4-carbon"


def resolve_gemini_model(raw: str) -> str:
    model = normalize_model_id(raw)
    if not is_valid_model_id(model):
        raise GatewayRequestError(400, "invalid_request_error", "Invalid model identifier.")
    if model not in allowed_gemini_models():
        raise GatewayRequestError(400, "invalid_request_error", f"Unknown or disabled Gemini model: {model}")
    return model


def parse_selected_model(selected: str) -> tuple[str, str]:
    provider, sep, model = (selected or "").partition("/")
    if not sep or provider not in ("gemini", "anthropic"):
        raise GatewayRequestError(400, "invalid_request_error", "Invalid model identifier.")
    if provider == "gemini":
        return provider, resolve_gemini_model(model)
    model = normalize_model_id(model)
    if not is_valid_model_id(model):
        raise GatewayRequestError(400, "invalid_request_error", "Invalid model identifier.")
    return provider, model


def _extract_text_from_content(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        chunks = []
        for item in raw_content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks)
    return ""


def estimate_anthropic_input_tokens(body: dict[str, Any]) -> int:
    total_chars = 0
    for key in ("system", "tools", "messages"):
        if body.get(key):
            total_chars += len(json.dumps(body[key], default=str))
    return max(1, total_chars // 4)


def derive_session_id(body: dict[str, Any], headers: Any = None) -> str:
    meta = body.get("metadata")
    if isinstance(meta, dict):
        uid = meta.get("user_id")
        if uid is not None and str(uid).strip():
            return str(uid)[:256]
    if headers is not None:
        hdr = headers.get("x-session-id")
        if hdr and str(hdr).strip():
            return str(hdr)[:256]
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    first = next((m for m in messages if isinstance(m, dict) and m.get("role") != "system"), None)
    material = json.dumps({"system": body.get("system"), "first": first, "metadata": meta}, sort_keys=True, default=str)
    return "anon-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _pin_if_absent(session_id: str, model: str, turns: int) -> bool:
    with SESSION_PINS.lock:
        if session_id in SESSION_PINS:
            return False
        SESSION_PINS.set(session_id, {"model": model, "turns": turns, "failover": False})
        return True


def pin_failover(session_id: str, model: str, turns: int) -> None:
    SESSION_PINS.set(str(session_id), {"model": model, "turns": turns, "failover": True})


def _decision(target: str, rule: str, reason: str, pinned: bool, digest: str) -> dict[str, Any]:
    return {
        "selected_model": target,
        "rule_matched": rule,
        "reason": reason,
        "session_pinned": pinned,
        "prompt_digest": digest,
    }


def classify_and_route(
    *,
    session_id: Any,
    requested_model: str,
    messages: list[dict[str, Any]],
    token_count_est: int,
    is_summary: bool = False,
    has_safeguards: bool = False,
    subagent_flag: bool = False,
    simulate_429_primary: bool = False,
) -> dict[str, Any]:
    """Choose an upstream `provider/model` for a request across Flash 3.8, Argon, and Carbon."""
    del is_summary, has_safeguards
    session_id = str(session_id)
    unwrapped = normalize_model_id(requested_model or "")
    non_system = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    turn_count = len(non_system)
    last_text = ""
    for msg in reversed(non_system):
        if msg.get("role") == "user":
            last_text = _extract_text_from_content(msg.get("content"))
            if last_text.strip():
                break
    digest = hashlib.sha256(last_text.encode("utf-8")).hexdigest()[:16]
    sec_target = resolve_security_target()

    # 1. Haiku-class background calls route to gemini-3.8-flash (never gemini-3.5-flash-lite).
    if "haiku" in unwrapped:
        return _decision(
            "gemini/gemini-3.8-flash",
            "background_haiku_flash",
            "Haiku-class request routed to gemini-3.8-flash.",
            False,
            digest,
        )

    # 2. Auto router within the Gemini family (gemini-3.8-flash / gemini-4-argon / gemini-4-carbon).
    if unwrapped in ("claude-auto-gemini", "auto-gemini"):
        if SECURITY_RE.search(last_text):
            target, tier = sec_target, "security"
        elif token_count_est > 150_000 or LONG_DECODE_RE.search(last_text):
            target, tier = "gemini/gemini-4-carbon", "long_decode_or_large_context"
        else:
            target, tier = "gemini/gemini-3.8-flash", "workhorse_flash"
        return _decision(target, "single_family_complexity_router", f"Auto router tier: {tier}.", False, digest)

    # 3. Explicit Gemini model.
    if unwrapped.startswith("gemini"):
        target = "gemini/" + resolve_gemini_model(unwrapped)
        pinned = False
        if not subagent_flag:
            pinned = _pin_if_absent(session_id, target, turn_count)
        return _decision(target, "explicit_gemini_role_or_agent", "Gemini model requested explicitly.", pinned, digest)

    # 4. Primary provider quota failure: pin the session to Gemini 4 Carbon.
    if simulate_429_primary:
        target = "gemini/gemini-4-carbon"
        pin_failover(session_id, target, turn_count)
        return _decision(
            target,
            "sticky_quota_failover_429",
            "Primary provider returned 429; session pinned to Gemini.",
            True,
            digest,
        )

    # 5. Caller-supplied subagent flag.
    if subagent_flag:
        if SECURITY_RE.search(last_text):
            target, rule = sec_target, "subagent_security_pro"
        elif token_count_est > 150_000 or LONG_DECODE_RE.search(last_text):
            target, rule = "gemini/gemini-4-carbon", "subagent_deep_pro"
        else:
            target, rule = "gemini/gemini-3.8-flash", "subagent_workhorse_flash"
        return _decision(target, rule, "Subagent request.", False, digest)

    # 6. Existing session pin.
    with SESSION_PINS.lock:
        pin = SESSION_PINS.get(session_id)
        if pin is not None and (turn_count > 2 or pin.get("failover")):
            pin["turns"] = turn_count
            target = pin["model"]
            return _decision(
                target,
                "session_pinned_cache_guard",
                "Session pinned to keep one provider per thread.",
                True,
                digest,
            )

    # 7. Anthropic passthrough when ANTHROPIC_UPSTREAM_KEY is configured, else default Gemini model.
    if (os.environ.get("ANTHROPIC_UPSTREAM_KEY") or "").strip():
        target, rule = "anthropic/" + (unwrapped or "claude-opus-5-5"), "anthropic_passthrough"
    else:
        fallback = normalize_model_id(
            os.environ.get("CC_AGY_UNMAPPED_MODEL")
            or os.environ.get("GEMINI4_UNMAPPED_MODEL")
            or "gemini-3.8-flash"
        )
        target, rule = "gemini/" + resolve_gemini_model(fallback), "unmapped_model_default"
    pinned = _pin_if_absent(session_id, target, max(1, turn_count))
    return _decision(target, rule, "Model not mapped to a Gemini tier.", pinned, digest)


def resolve_max_output_tokens(body: dict[str, Any]) -> int:
    value = body.get("max_tokens")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return min(value, MAX_OUTPUT_TOKENS)
    return DEFAULT_MAX_OUTPUT_TOKENS


def _store_thought_signature(tool_id: str, signature: str) -> None:
    if tool_id and signature:
        THOUGHT_SIGNATURE_VAULT.set(tool_id, signature)


def translate_anthropic_to_gemini_payload(body: dict[str, Any], gemini_model_id: str) -> dict[str, Any]:
    """Translate an Anthropic Messages request into a Gemini generateContent payload."""
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    tool_id_to_name: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            for blk in msg["content"]:
                if isinstance(blk, dict) and blk.get("type") == "tool_use" and blk.get("id"):
                    tool_id_to_name[str(blk["id"])] = str(blk.get("name") or "tool")

    embedded_system: list[str] = []
    contents: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        raw_role = msg.get("role", "user")
        raw_content = msg.get("content", "")
        if raw_role == "system":
            piece = _extract_text_from_content(raw_content).strip()
            if piece:
                embedded_system.append(piece)
            continue

        role = "model" if raw_role == "assistant" else "user"
        parts: list[dict[str, Any]] = []
        sentinel_used = False
        if isinstance(raw_content, str):
            parts.append({"text": raw_content.strip() or " "})
        elif isinstance(raw_content, list):
            for block in raw_content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    cleaned = str(block.get("text", "")).strip()
                    if cleaned:
                        parts.append({"text": cleaned})
                elif btype == "image":
                    src = block.get("source") or {}
                    if isinstance(src, dict) and src.get("type") == "base64" and src.get("data"):
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": src.get("media_type", "image/png"),
                                    "data": src["data"],
                                }
                            }
                        )
                elif btype == "tool_use":
                    tid = str(block.get("id", ""))
                    args = block.get("input")
                    fc_part: dict[str, Any] = {
                        "functionCall": {
                            "name": block.get("name", "tool"),
                            "args": args if isinstance(args, dict) else {},
                        }
                    }
                    sig = THOUGHT_SIGNATURE_VAULT.get(tid)
                    if sig:
                        fc_part["thoughtSignature"] = sig
                    elif not sentinel_used:
                        fc_part["thoughtSignature"] = "skip_thought_signature_validator"
                        sentinel_used = True
                    parts.append(fc_part)
                elif btype == "tool_result":
                    tid = str(block.get("tool_use_id", ""))
                    result = block.get("content", "")
                    if isinstance(result, list):
                        result = _extract_text_from_content(result)
                    elif not isinstance(result, str):
                        result = json.dumps(result, default=str)
                    response: dict[str, Any] = {"result": result}
                    if block.get("is_error"):
                        response = {"error": result}
                    parts.append(
                        {
                            "functionResponse": {
                                "name": tool_id_to_name.get(tid, tid or "tool"),
                                "response": response,
                            }
                        }
                    )
        if parts:
            fn_parts = [pt for pt in parts if "functionResponse" in pt]
            other_parts = [pt for pt in parts if "functionResponse" not in pt]
            ordered_parts = (fn_parts + other_parts) if (role == "user" and fn_parts and other_parts) else parts
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(ordered_parts)
            else:
                contents.append({"role": role, "parts": ordered_parts})

    short_model = gemini_model_id.rsplit("/", 1)[-1]
    is_pro_class = short_model in PRO_CLASS_MODELS
    max_out = resolve_max_output_tokens(body)
    if short_model in LONG_DECODE_MODELS:
        max_out = min(max_out, LONG_DECODE_SEGMENT_TOKENS)
    gen_cfg: dict[str, Any] = {"maxOutputTokens": max_out}
    level = "LOW"
    if is_pro_class:
        level = "MEDIUM" if short_model in LONG_DECODE_MODELS else "HIGH"
    gen_cfg["thinkingConfig"] = {"thinkingLevel": level}

    payload: dict[str, Any] = {
        "contents": contents or [{"role": "user", "parts": [{"text": " "}]}],
        "generationConfig": gen_cfg,
    }

    tools = body.get("tools")
    if isinstance(tools, list):
        decls = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("name"):
                decl: dict[str, Any] = {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                }
                if isinstance(tool.get("input_schema"), dict):
                    decl["parametersJsonSchema"] = tool["input_schema"]
                decls.append(decl)
        if decls:
            payload["tools"] = [{"functionDeclarations": decls}]

    system_chunks: list[str] = []
    top_sys = _extract_text_from_content(body.get("system")).strip()
    if top_sys:
        system_chunks.append(top_sys)
    system_chunks.extend(embedded_system)
    if system_chunks:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_chunks)}]}
    return payload


_SAFETY_FINISH_REASONS = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "LANGUAGE"}
)


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def gemini_usage_to_anthropic(meta: Any) -> dict[str, int]:
    meta = meta if isinstance(meta, dict) else {}
    prompt = _as_int(meta.get("promptTokenCount"))
    cached = _as_int(meta.get("cachedContentTokenCount"))
    output = _as_int(meta.get("candidatesTokenCount")) + _as_int(meta.get("thoughtsTokenCount"))
    return {
        "input_tokens": max(0, prompt - cached),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
        "output_tokens": output,
    }


def gemini_response_to_anthropic(gem_resp: Any, model_label: str) -> dict[str, Any]:
    """Convert a generateContent response into an Anthropic Message object."""
    if not isinstance(gem_resp, dict):
        raise GatewayRequestError(502, "api_error", "Gemini returned a malformed response.")
    candidates = gem_resp.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        feedback = gem_resp.get("promptFeedback") or {}
        reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
        if reason:
            raise GatewayRequestError(
                400, "invalid_request_error", f"Gemini blocked the prompt (blockReason={reason})."
            )
        raise GatewayRequestError(502, "api_error", "Gemini returned no candidates.")

    cand = candidates[0]
    finish = str(cand.get("finishReason") or "STOP").upper()
    content = cand.get("content") if isinstance(cand.get("content"), dict) else {}
    blocks: list[dict[str, Any]] = []
    has_tool_use = False
    for part in content.get("parts") or []:
        if not isinstance(part, dict) or part.get("thought"):
            continue
        fc = part.get("functionCall")
        if isinstance(fc, dict):
            tid = "toolu_" + uuid.uuid4().hex[:24]
            if isinstance(part.get("thoughtSignature"), str):
                _store_thought_signature(tid, part["thoughtSignature"])
            args = fc.get("args")
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tid,
                    "name": str(fc.get("name") or "tool"),
                    "input": args if isinstance(args, dict) else {},
                }
            )
            has_tool_use = True
        elif isinstance(part.get("text"), str) and part["text"]:
            if blocks and blocks[-1]["type"] == "text":
                blocks[-1]["text"] += part["text"]
            else:
                blocks.append({"type": "text", "text": part["text"]})

    if has_tool_use:
        stop_reason = "tool_use"
    elif finish in ("MAX_TOKENS", "CONTINUATION"):
        stop_reason = "max_tokens"
    elif finish == "STOP" or blocks:
        stop_reason = "end_turn"
    elif finish in _SAFETY_FINISH_REASONS:
        raise GatewayRequestError(
            400, "invalid_request_error", f"Gemini returned no output (finishReason={finish})."
        )
    else:
        raise GatewayRequestError(502, "api_error", f"Gemini returned no output (finishReason={finish}).")

    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": model_label,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": gemini_usage_to_anthropic(gem_resp.get("usageMetadata")),
    }


def _sse_frame(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode("utf-8")


PING_FRAME = _sse_frame("ping", {"type": "ping"})


def anthropic_sse_events(msg: dict[str, Any]) -> list[bytes]:
    usage = msg.get("usage") or {}
    frames = [
        _sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg.get("id"),
                    "type": "message",
                    "role": "assistant",
                    "model": msg.get("model"),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": usage.get("input_tokens", 0),
                        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                        "output_tokens": 0,
                    },
                },
            },
        )
    ]
    for idx, blk in enumerate(msg.get("content") or []):
        if blk.get("type") == "tool_use":
            start = {"type": "tool_use", "id": blk.get("id"), "name": blk.get("name"), "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(blk.get("input") or {})}
        else:
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": blk.get("text", "")}
        frames.append(
            _sse_frame("content_block_start", {"type": "content_block_start", "index": idx, "content_block": start})
        )
        frames.append(_sse_frame("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": delta}))
        frames.append(_sse_frame("content_block_stop", {"type": "content_block_stop", "index": idx}))
    frames.append(
        _sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": msg.get("stop_reason"), "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
        )
    )
    frames.append(_sse_frame("message_stop", {"type": "message_stop"}))
    return frames


def format_anthropic_sse_stream(msg: dict[str, Any]) -> bytes:
    return b"".join(anthropic_sse_events(msg))


def load_prices() -> dict[str, tuple[float, float, float]]:
    prices = dict(DEFAULT_PRICES_PER_MTOK)
    raw = (
        os.environ.get("CC_AGY_PRICES_JSON")
        or os.environ.get("GEMINI4_PRICES_JSON")
        or ""
    ).strip()
    if not raw:
        return prices
    try:
        overrides = json.loads(raw)
    except ValueError:
        return prices
    if isinstance(overrides, dict):
        for model, rates in overrides.items():
            if (
                isinstance(rates, list)
                and len(rates) == 3
                and all(isinstance(r, (int, float)) and not isinstance(r, bool) and r >= 0 for r in rates)
            ):
                prices[normalize_model_id(str(model))] = (float(rates[0]), float(rates[1]), float(rates[2]))
    return prices


def _price_for(model: str) -> tuple[float, float, float]:
    return load_prices().get(normalize_model_id(model), FALLBACK_PRICE_PER_MTOK)


def _env_float(name: str, legacy_name: str | None = None) -> float | None:
    raw = (os.environ.get(name) or (os.environ.get(legacy_name) if legacy_name else "") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"{name} must be a non-negative number, got {raw!r}.")
    return value


def estimate_reservation_usd(model: str, input_tokens: int, max_output_tokens: int) -> float:
    in_rate, _, out_rate = _price_for(model)
    return (input_tokens * in_rate + max_output_tokens * out_rate) / 1_000_000.0


def cost_from_gemini_usage(model: str, meta: Any) -> float:
    usage = gemini_usage_to_anthropic(meta)
    in_rate, cached_rate, out_rate = _price_for(model)
    return (
        usage["input_tokens"] * in_rate
        + usage["cache_read_input_tokens"] * cached_rate
        + usage["output_tokens"] * out_rate
    ) / 1_000_000.0


def cost_from_anthropic_usage(model: str, usage: dict[str, Any]) -> float:
    in_rate, cached_rate, out_rate = _price_for(model)
    return (
        _as_int(usage.get("input_tokens")) * in_rate
        + _as_int(usage.get("cache_creation_input_tokens")) * in_rate * 1.25
        + _as_int(usage.get("cache_read_input_tokens")) * cached_rate
        + _as_int(usage.get("output_tokens")) * out_rate
    ) / 1_000_000.0


def reserve_budget(session_id: str, model: str, input_tokens: int, max_output_tokens: int) -> tuple[bool, float, str]:
    reservation = estimate_reservation_usd(model, input_tokens, max_output_tokens)
    session_cap = _env_float("CC_AGY_SESSION_BUDGET_USD", "GEMINI4_SESSION_BUDGET_USD")
    global_cap = _env_float("CC_AGY_GLOBAL_BUDGET_USD", "GEMINI4_GLOBAL_BUDGET_USD")
    with BUDGET_LOCK:
        entry = SESSION_SPEND_USD.get(session_id) or {"settled": 0.0, "reserved": 0.0}
        session_total = entry["settled"] + entry["reserved"] + reservation
        global_total = GLOBAL_SPEND_USD["settled"] + GLOBAL_SPEND_USD["reserved"] + reservation
        if session_cap is not None and session_total > session_cap:
            return (
                False,
                0.0,
                f"Session budget exceeded: projected ${session_total:.4f} > session cap ${session_cap:.4f}.",
            )
        if global_cap is not None and global_total > global_cap:
            return (
                False,
                0.0,
                f"Global budget exceeded: projected ${global_total:.4f} > global cap ${global_cap:.4f}.",
            )
        entry = {"settled": entry["settled"], "reserved": entry["reserved"] + reservation}
        SESSION_SPEND_USD.set(session_id, entry)
        GLOBAL_SPEND_USD["reserved"] += reservation
    return True, reservation, ""


def settle_budget(session_id: str, reservation: float, actual_usd: float | None) -> None:
    charge = max(0.0, actual_usd or 0.0)
    with BUDGET_LOCK:
        entry = SESSION_SPEND_USD.get(session_id) or {"settled": 0.0, "reserved": 0.0}
        SESSION_SPEND_USD.set(
            session_id,
            {
                "settled": entry["settled"] + charge,
                "reserved": max(0.0, entry["reserved"] - reservation),
            },
        )
        GLOBAL_SPEND_USD["reserved"] = max(0.0, GLOBAL_SPEND_USD["reserved"] - reservation)
        GLOBAL_SPEND_USD["settled"] += charge


_VERTEX_TOKEN_LOCK = threading.Lock()
_VERTEX_TOKEN_CACHE: dict[str, Any] = {"token": None, "expires": 0.0}


def _vertex_access_token() -> str:
    with _VERTEX_TOKEN_LOCK:
        now = time.time()
        if _VERTEX_TOKEN_CACHE["token"] and now < _VERTEX_TOKEN_CACHE["expires"]:
            return _VERTEX_TOKEN_CACHE["token"]
        try:
            proc = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GatewayRequestError(
                401,
                "authentication_error",
                f"Could not obtain a Vertex AI token from gcloud ({type(exc).__name__}).",
            ) from exc
        token = proc.stdout.strip()
        if proc.returncode != 0 or not token:
            raise GatewayRequestError(
                401,
                "authentication_error",
                "gcloud auth print-access-token failed; run `gcloud auth login`.",
            )
        _VERTEX_TOKEN_CACHE["token"] = token
        _VERTEX_TOKEN_CACHE["expires"] = now + VERTEX_TOKEN_TTL_S
        return token


_SSL_CONTEXT_LOCK = threading.Lock()
_SSL_CONTEXT: ssl.SSLContext | None = None


def _build_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if ctx.cert_store_stats().get("x509_ca", 0) == 0:
        candidates: list[str] = []
        try:
            import certifi  # type: ignore[import-not-found]

            candidates.append(certifi.where())
        except ImportError:
            pass
        candidates += [
            "/etc/ssl/cert.pem",
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
        ]
        for cafile in candidates:
            if os.path.isfile(cafile):
                ctx.load_verify_locations(cafile)
                break
    return ctx


def _ssl_context() -> ssl.SSLContext:
    global _SSL_CONTEXT
    with _SSL_CONTEXT_LOCK:
        if _SSL_CONTEXT is None:
            _SSL_CONTEXT = _build_ssl_context()
        return _SSL_CONTEXT


def _gemini_api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def _vertex_enabled() -> bool:
    return (os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower() == "true" and bool(
        (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    )


def gemini_credentials_available() -> bool:
    return bool(_gemini_api_key()) or _vertex_enabled()


def gemini_endpoint(model: str, *, stream_sse: bool = False) -> tuple[str, dict[str, str]]:
    """Return `(url, headers)` for `generateContent` or `streamGenerateContent?alt=sse`."""
    if not is_valid_model_id(model):
        raise GatewayRequestError(400, "invalid_request_error", "Invalid model identifier.")
    quoted = urllib.parse.quote(model, safe="")
    action = "streamGenerateContent?alt=sse" if stream_sse else "generateContent"
    api_key = _gemini_api_key()
    if api_key:
        return (
            f"{GEMINI_API_BASE}{quoted}:{action}",
            {"Content-Type": "application/json", "x-goog-api-key": api_key},
        )
    if _vertex_enabled():
        from server.bifrost_policy import validate_policy_url
        from server.regions import (
            GLOBAL_ONLY_MODELS,
            enforce_project_controls,
            resolve_bifrost_virtual_key,
            resolve_vertex_base_url,
            resolve_vertex_base_url_token,
        )

        project = os.environ["GOOGLE_CLOUD_PROJECT"].strip()
        location = (os.environ.get("GOOGLE_CLOUD_LOCATION") or "global").strip()
        if not GCP_SEGMENT_RE.match(project) or not GCP_SEGMENT_RE.match(location):
            raise GatewayRequestError(
                400, "invalid_request_error", "Invalid GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION."
            )
        try:
            enforce_project_controls(auth_mode="vertex", project=project, source_env=os.environ)
        except ValueError as exc:
            raise GatewayRequestError(403, "permission_error", str(exc)) from exc
        if model in GLOBAL_ONLY_MODELS:
            location = "global"
        raw_base_url = resolve_vertex_base_url(env=os.environ)
        if raw_base_url:
            try:
                base_url = validate_policy_url(raw_base_url)
            except ValueError as exc:
                raise GatewayRequestError(400, "invalid_request_error", str(exc)) from exc
            url = (
                f"{base_url}/v1beta1/projects/{urllib.parse.quote(project, safe='')}"
                f"/locations/{urllib.parse.quote(location, safe='')}/publishers/google/models/{quoted}:{action}"
            )
            try:
                gw_tok = (
                    resolve_vertex_base_url_token(base_url=base_url, env=os.environ)
                    or resolve_bifrost_virtual_key(env=os.environ)
                )
            except (RuntimeError, ValueError) as exc:
                raise GatewayRequestError(401, "authentication_error", str(exc)) from exc
            bearer = gw_tok if gw_tok else _vertex_access_token()
            headers: dict[str, str] = {"Content-Type": "application/json", "Authorization": f"Bearer {bearer}"}
            bf_vk = resolve_bifrost_virtual_key(env=os.environ)
            if bf_vk:
                headers["x-bf-vk"] = bf_vk
            return url, headers
        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
        url = (
            f"https://{host}/v1/projects/{urllib.parse.quote(project, safe='')}"
            f"/locations/{urllib.parse.quote(location, safe='')}/publishers/google/models/{quoted}:{action}"
        )
        return url, {"Content-Type": "application/json", "Authorization": f"Bearer {_vertex_access_token()}"}
    raise GatewayRequestError(
        401,
        "authentication_error",
        "No Gemini credentials configured. Set GEMINI_API_KEY, or GOOGLE_GENAI_USE_VERTEXAI=true with GOOGLE_CLOUD_PROJECT.",
    )


def _map_upstream_status(code: int, text: str) -> tuple[int, str]:
    if code == 400 and "API_KEY_INVALID" in text.upper():
        return 401, "authentication_error"
    if code in (401, 403):
        return code, "authentication_error"
    if code == 404:
        return 404, "not_found_error"
    if code == 413:
        return 413, "request_too_large"
    if code == 429:
        return 429, "rate_limit_error"
    if code in (503, 529):
        return code, "overloaded_error"
    if 400 <= code < 500:
        return code, "invalid_request_error"
    if 500 <= code < 600:
        return code, "api_error"
    return 502, "api_error"


def _upstream_http_error(provider: str, err: urllib.error.HTTPError) -> GatewayRequestError:
    try:
        text = err.read(65536).decode("utf-8", errors="replace")
    except Exception:
        text = ""
    detail = text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            detail = str(parsed["error"].get("message") or text)
    except ValueError:
        pass
    status, err_type = _map_upstream_status(err.code, text)
    return GatewayRequestError(status, err_type, f"{provider} returned HTTP {err.code}: {detail.strip()[:500]}")


def call_gemini(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    *,
    long_decode: bool = False,
) -> dict[str, Any]:
    """Call Gemini `generateContent` or multi-segment `continuationToken` SSE long-decode."""
    if long_decode:
        stream_url = url.replace(":generateContent", ":streamGenerateContent?alt=sse")
        try:
            ld_res = call_gemini_long_decode(
                endpoint_url=stream_url,
                headers=headers,
                request_body=payload,
                max_continuations=MAX_CONTINUATIONS,
                timeout=timeout,
            )
        except RuntimeError as err:
            raise GatewayRequestError(502, "api_error", str(err)) from err
        usage_dict = ld_res.get("usage") if isinstance(ld_res.get("usage"), dict) else {}
        return {
            "candidates": [
                {
                    "finishReason": ld_res.get("finish_reason") or "STOP",
                    "content": {
                        "role": "model",
                        "parts": [{"text": str(ld_res.get("text") or "")}],
                    },
                }
            ],
            "usageMetadata": {
                "promptTokenCount": int(usage_dict.get("prompt_token_count") or 0),
                "cachedContentTokenCount": int(usage_dict.get("cached_content_token_count") or 0),
                "candidatesTokenCount": int(usage_dict.get("candidates_token_count") or 0),
                "thoughtsTokenCount": int(usage_dict.get("thoughts_token_count") or 0),
                "totalTokenCount": int(usage_dict.get("total_token_count") or 0),
            },
            "continuations": int(ld_res.get("continuations") or 0),
        }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as err:
        raise _upstream_http_error("Gemini", err) from err
    except (socket.timeout, TimeoutError) as err:
        raise GatewayRequestError(504, "api_error", "Gemini request timed out.") from err
    except (urllib.error.URLError, OSError) as err:
        raise GatewayRequestError(502, "api_error", f"Gemini request failed ({type(err).__name__}).") from err
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as err:
        raise GatewayRequestError(502, "api_error", "Gemini returned invalid JSON.") from err


class AnthropicUsageTracker:
    """Extract usage from a relayed Anthropic response (SSE or JSON) without altering it."""

    MAX_BUFFER = 16 * 1024 * 1024

    def __init__(self, is_sse: bool) -> None:
        self.is_sse = is_sse
        self.usage: dict[str, int] = {}
        self._buf = bytearray()
        self._overflow = False

    def _merge(self, usage: Any) -> None:
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    self.usage[key] = value

    def feed(self, chunk: bytes) -> None:
        if self._overflow:
            return
        self._buf.extend(chunk)
        if len(self._buf) > self.MAX_BUFFER:
            self._overflow = True
            self._buf = bytearray()
            return
        if not self.is_sse:
            return
        while b"\n" in self._buf:
            idx = self._buf.index(b"\n")
            line = bytes(self._buf[:idx]).strip()
            del self._buf[: idx + 1]
            if not line.startswith(b"data:"):
                continue
            try:
                event = json.loads(line[5:].decode("utf-8"))
            except ValueError:
                continue
            if isinstance(event, dict):
                if event.get("type") == "message_start" and isinstance(event.get("message"), dict):
                    self._merge(event["message"].get("usage"))
                elif event.get("type") == "message_delta":
                    self._merge(event.get("usage"))

    def finish(self) -> dict[str, int] | None:
        if not self.is_sse and not self._overflow and self._buf:
            try:
                parsed = json.loads(bytes(self._buf).decode("utf-8"))
                if isinstance(parsed, dict):
                    self._merge(parsed.get("usage"))
            except ValueError:
                pass
        return self.usage or None


_RELAYED_RESPONSE_HEADERS = ("request-id", "retry-after", "x-should-retry")


def normalize_gateway_route_path(raw_path: str) -> str:
    path = raw_path.split("?", 1)[0]
    if path.startswith("/anthropic/"):
        path = path[len("/anthropic") :]
    return path


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "cc-agy-gateway/" + VERSION
    sys_version = ""
    protocol_version = "HTTP/1.0"
    timeout = 10

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _begin(self, status: int, content_type: str, extra: dict[str, str] | None = None) -> None:
        self._response_started = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)

    def _send_json(self, status: int, data: dict[str, Any], extra: dict[str, str] | None = None) -> None:
        raw = json.dumps(data).encode("utf-8")
        self._begin(status, "application/json", extra)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except OSError:
            pass

    def _send_error(
        self, status: int, err_type: str, message: str, extra: dict[str, str] | None = None
    ) -> None:
        self._send_json(status, anthropic_error_body(err_type, message), extra)

    def _write(self, data: bytes) -> bool:
        try:
            self.wfile.write(data)
            self.wfile.flush()
            return True
        except OSError:
            return False

    def _check_host_and_origin(self) -> bool:
        port = self.server.server_address[1]
        hosts = self.headers.get_all("Host") or []
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        if len(hosts) != 1 or hosts[0].strip().lower() not in allowed:
            self._send_error(403, "permission_error", "Host header not allowed.")
            return False
        if self.headers.get("Origin") is not None:
            self._send_error(403, "permission_error", "Cross-origin requests are not allowed.")
            return False
        return True

    def _check_auth(self) -> bool:
        expected = getattr(self.server, "gateway_token", None)
        presented = self.headers.get("x-api-key")
        if presented is None:
            auth = self.headers.get("Authorization") or ""
            scheme, _, value = auth.partition(" ")
            if scheme.lower() == "bearer":
                presented = value.strip()
        if expected and presented and token_matches(presented.strip(), expected):
            return True
        self._send_error(401, "authentication_error", "Missing or invalid gateway token.")
        return False

    def _read_json_body(self) -> tuple[bytes, dict[str, Any]] | None:
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            self._send_error(
                400, "invalid_request_error", "Chunked request bodies are not supported; send Content-Length."
            )
            return None
        raw_len = self.headers.get("Content-Length")
        try:
            length = int(raw_len) if raw_len is not None else 0
        except ValueError:
            length = -1
        if length < 0:
            self.close_connection = True
            self._send_error(400, "invalid_request_error", "Invalid Content-Length.")
            return None
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            self._send_error(413, "request_too_large", f"Request body exceeds {MAX_BODY_BYTES} bytes.")
            return None
        try:
            raw = self.rfile.read(length) if length else b""
        except OSError:
            self.close_connection = True
            return None
        if len(raw) != length:
            self.close_connection = True
            self._send_error(400, "invalid_request_error", "Request body shorter than Content-Length.")
            return None
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, RecursionError):
            self._send_error(400, "invalid_request_error", "Request body is not valid JSON.")
            return None
        if not isinstance(body, dict):
            self._send_error(400, "invalid_request_error", "Request body must be a JSON object.")
            return None
        return raw, body

    def _method_not_allowed(self) -> None:
        self.close_connection = True
        self._send_error(405, "invalid_request_error", "Method not allowed.", {"Allow": "GET, POST"})

    do_OPTIONS = _method_not_allowed
    do_PUT = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_HEAD = _method_not_allowed

    def do_GET(self) -> None:
        self._response_started = False
        if not self._check_host_and_origin():
            return
        route = normalize_gateway_route_path(self.path)
        if route == "/health":
            self._send_json(200, {"status": "ok", "version": VERSION})
            return
        if not self._check_auth():
            return
        if route.startswith(NOT_IMPLEMENTED_PREFIXES):
            self._send_json(501, NOT_IMPLEMENTED_BODY)
            return
        if route == "/v1/models":
            self._send_json(
                200,
                {
                    "data": GATEWAY_MODEL_CATALOG,
                    "has_more": False,
                    "first_id": GATEWAY_MODEL_CATALOG[0]["id"],
                    "last_id": GATEWAY_MODEL_CATALOG[-1]["id"],
                },
            )
            return
        if route.startswith("/v1/models/"):
            model_id = urllib.parse.unquote(route[len("/v1/models/") :])
            entry = next((m for m in GATEWAY_MODEL_CATALOG if m["id"] == model_id), None)
            if entry:
                self._send_json(200, entry)
            else:
                self._send_error(404, "not_found_error", "Model not found.")
            return
        self._send_error(404, "not_found_error", "Not found.")

    def do_POST(self) -> None:
        self._response_started = False
        if not self._check_host_and_origin():
            return
        if not self._check_auth():
            self.close_connection = True
            return
        route = normalize_gateway_route_path(self.path)
        if route.startswith(NOT_IMPLEMENTED_PREFIXES):
            self.close_connection = True
            self._send_json(501, NOT_IMPLEMENTED_BODY)
            return
        if route == "/v1/messages/count_tokens":
            parsed = self._read_json_body()
            if parsed is not None:
                self._send_json(200, {"input_tokens": estimate_anthropic_input_tokens(parsed[1])})
            return
        if route == "/v1/messages":
            parsed = self._read_json_body()
            if parsed is None:
                return
            try:
                self._handle_messages(parsed[0], parsed[1])
            except GatewayRequestError as err:
                if not self._response_started:
                    self._send_error(err.status, err.err_type, err.message)
            except Exception as exc:
                log_event("internal_error", status=500, rule=type(exc).__name__)
                if not self._response_started:
                    self._send_error(500, "api_error", "Internal gateway error.")
            return
        self.close_connection = True
        self._send_error(404, "not_found_error", "Not found.")

    def _handle_messages(self, raw: bytes, body: dict[str, Any]) -> None:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise GatewayRequestError(400, "invalid_request_error", "messages must be a non-empty array.")
        requested = body.get("model")
        if not isinstance(requested, str) or not requested.strip():
            raise GatewayRequestError(400, "invalid_request_error", "model is required.")
        max_tokens = body.get("max_tokens")
        if max_tokens is not None and (
            not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1
        ):
            raise GatewayRequestError(400, "invalid_request_error", "max_tokens must be a positive integer.")

        session_id = derive_session_id(body, self.headers)
        est_input = estimate_anthropic_input_tokens(body)
        decision = classify_and_route(
            session_id=session_id,
            requested_model=requested,
            messages=messages,
            token_count_est=est_input,
            is_summary=bool(body.get("context_management")),
            subagent_flag=body.get("isSubagent") is True,
        )
        provider, model = parse_selected_model(decision["selected_model"])
        log_event(
            "route",
            session=session_digest(session_id),
            requested=str(requested)[:80],
            model=f"{provider}/{model}",
            rule=decision["rule_matched"],
            input_tokens=est_input,
        )
        if provider == "anthropic":
            self._proxy_anthropic(raw, body, session_id, decision, model, est_input)
        else:
            self._serve_gemini(body, session_id, decision, model, est_input)

    @staticmethod
    def _decision_header(decision: dict[str, Any]) -> dict[str, str]:
        return {"X-Gemini4-Gateway": f"model={decision['selected_model']}; rule={decision['rule_matched']}"}

    def _serve_gemini(
        self,
        body: dict[str, Any],
        session_id: str,
        decision: dict[str, Any],
        model: str,
        est_input: int,
    ) -> None:
        payload = translate_anthropic_to_gemini_payload(body, model)
        url, headers = gemini_endpoint(model)
        max_out = payload["generationConfig"]["maxOutputTokens"]
        allowed, reservation, message = reserve_budget(session_id, model, est_input, max_out)
        if not allowed:
            raise GatewayRequestError(403, "permission_error", message)
        short_model = model.rsplit("/", 1)[-1]
        timeout_s = 1200 if short_model in PRO_CLASS_MODELS else 300
        use_long_decode = bool(
            short_model in LONG_DECODE_MODELS
            and not payload.get("tools")
            and int(body.get("max_tokens") or 0) > 32768
        )
        outcome: dict[str, Any] = {}
        done = threading.Event()
        digest = session_digest(session_id)

        def worker() -> None:
            try:
                gem_resp = call_gemini(url, headers, payload, timeout_s, long_decode=use_long_decode)
                msg = gemini_response_to_anthropic(gem_resp, f"claude-{model}")
                usage_meta = gem_resp.get("usageMetadata")
                settle_budget(
                    session_id,
                    reservation,
                    cost_from_gemini_usage(model, usage_meta) if usage_meta else reservation,
                )
                usage = msg["usage"]
                log_event(
                    "upstream",
                    session=digest,
                    model=model,
                    rule=decision["rule_matched"],
                    status=200,
                    input_tokens=usage["input_tokens"],
                    cached_tokens=usage["cache_read_input_tokens"],
                    output_tokens=usage["output_tokens"],
                    continuations=gem_resp.get("continuations"),
                )
                outcome["message"] = msg
            except GatewayRequestError as err:
                settle_budget(session_id, reservation, None)
                log_event("upstream", session=digest, model=model, rule=decision["rule_matched"], status=err.status)
                outcome["error"] = err
            except Exception as exc:
                settle_budget(session_id, reservation, None)
                log_event("upstream", session=digest, model=model, status=500, rule=type(exc).__name__)
                outcome["error"] = GatewayRequestError(500, "api_error", "Internal gateway error.")
            finally:
                done.set()

        threading.Thread(target=worker, daemon=True).start()
        extra = self._decision_header(decision)

        if body.get("stream") is not True:
            done.wait()
            if "error" in outcome:
                raise outcome["error"]
            self._send_json(200, outcome["message"], extra)
            return

        if not done.wait(PING_INTERVAL_S):
            self._begin(200, "text/event-stream; charset=utf-8", extra)
            self.end_headers()
            self.close_connection = True
            if not self._write(PING_FRAME):
                return
            while not done.wait(PING_INTERVAL_S):
                if not self._write(PING_FRAME):
                    return
        elif "error" in outcome:
            raise outcome["error"]
        else:
            self._begin(200, "text/event-stream; charset=utf-8", extra)
            self.end_headers()
            self.close_connection = True

        if "error" in outcome:
            err = outcome["error"]
            self._write(_sse_frame("error", anthropic_error_body(err.err_type, err.message)))
            return
        for frame in anthropic_sse_events(outcome["message"]):
            if not self._write(frame):
                return

    def _proxy_anthropic(
        self,
        raw: bytes,
        body: dict[str, Any],
        session_id: str,
        decision: dict[str, Any],
        model: str,
        est_input: int,
    ) -> None:
        key = (os.environ.get("ANTHROPIC_UPSTREAM_KEY") or "").strip()
        if not key:
            raise GatewayRequestError(
                401,
                "authentication_error",
                "ANTHROPIC_UPSTREAM_KEY is not set; cannot forward to api.anthropic.com.",
            )
        model = normalize_model_id(str(body.get("model") or model))
        real_max = (
            body.get("max_tokens")
            if isinstance(body.get("max_tokens"), int)
            else resolve_max_output_tokens(body)
        )
        allowed, reservation, message = reserve_budget(session_id, model, est_input, max(1, real_max))
        if not allowed:
            raise GatewayRequestError(403, "permission_error", message)
        headers = {
            "content-type": self.headers.get("content-type") or "application/json",
            "anthropic-version": self.headers.get("anthropic-version") or "2023-06-01",
            "x-api-key": key,
        }
        beta = self.headers.get("anthropic-beta")
        if beta:
            headers["anthropic-beta"] = beta
        req = urllib.request.Request(ANTHROPIC_MESSAGES_URL, data=raw, headers=headers, method="POST")
        digest = session_digest(session_id)
        try:
            upstream = urllib.request.urlopen(req, timeout=ANTHROPIC_TIMEOUT_S, context=_ssl_context())
        except urllib.error.HTTPError as err:
            settle_budget(session_id, reservation, None)
            log_event("upstream", session=digest, model=model, rule=decision["rule_matched"], status=err.code)
            if (
                err.code in (429, 529)
                and gemini_credentials_available()
                and (
                    os.environ.get("CC_AGY_FAILOVER_TO_GEMINI")
                    or os.environ.get("GEMINI4_FAILOVER_TO_GEMINI")
                    or ""
                ).strip()
                == "1"
            ):
                err.close()
                target = "gemini-4-carbon"
                pin_failover(session_id, f"gemini/{target}", len(body.get("messages") or []))
                decision = dict(
                    decision,
                    selected_model=f"gemini/{target}",
                    rule_matched="live_429_sticky_failover",
                )
                log_event("failover", session=digest, model=target, rule=decision["rule_matched"])
                self._serve_gemini(body, session_id, decision, target, est_input)
                return
            try:
                self._relay(err, None, decision)
            finally:
                err.close()
            return
        except (socket.timeout, TimeoutError) as err:
            settle_budget(session_id, reservation, None)
            raise GatewayRequestError(504, "api_error", "Request to api.anthropic.com timed out.") from err
        except (urllib.error.URLError, OSError) as err:
            settle_budget(session_id, reservation, None)
            raise GatewayRequestError(
                502, "api_error", f"Request to api.anthropic.com failed ({type(err).__name__})."
            ) from err

        ctype = upstream.headers.get("Content-Type") or ""
        tracker = AnthropicUsageTracker(is_sse="text/event-stream" in ctype)
        try:
            self._relay(upstream, tracker, decision)
        finally:
            upstream.close()
            usage = tracker.finish()
            actual = cost_from_anthropic_usage(model, usage) if usage else reservation
            settle_budget(session_id, reservation, actual)
            usage = usage or {}
            log_event(
                "upstream",
                session=digest,
                model=model,
                rule=decision["rule_matched"],
                status=getattr(upstream, "status", None),
                input_tokens=usage.get("input_tokens"),
                cached_tokens=usage.get("cache_read_input_tokens"),
                output_tokens=usage.get("output_tokens"),
            )

    def _relay(self, resp: Any, tracker: AnthropicUsageTracker | None, decision: dict[str, Any]) -> None:
        status = getattr(resp, "status", None) or getattr(resp, "code", 502)
        resp_headers = resp.headers
        extra = self._decision_header(decision)
        for name in _RELAYED_RESPONSE_HEADERS:
            value = resp_headers.get(name)
            if value and "\r" not in value and "\n" not in value:
                extra[name] = value
        for name, value in resp_headers.items():
            if name.lower().startswith("anthropic-ratelimit-") and "\r" not in value and "\n" not in value:
                extra[name] = value
        self._begin(status, resp_headers.get("Content-Type") or "application/json", extra)
        length = resp_headers.get("Content-Length")
        if length and not resp_headers.get("Transfer-Encoding"):
            self.send_header("Content-Length", length)
        self.end_headers()
        self.close_connection = True
        reader = getattr(resp, "read1", None)
        while True:
            chunk = reader(65536) if reader else resp.read(65536)
            if not chunk:
                break
            if tracker is not None:
                tracker.feed(chunk)
            if not self._write(chunk):
                break


MAX_CONNECTIONS = 64


class ReusableGatewayHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    _slots = threading.BoundedSemaphore(MAX_CONNECTIONS)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            try:
                request.close()
            except OSError:
                pass
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def __init__(self, server_address: tuple[str, int], handler_class: Any, token: str | None = None) -> None:
        super().__init__(server_address, handler_class)
        if token is None:
            token, _, _ = load_or_create_token()
        self.gateway_token = token


def run_server(port: int = GATEWAY_PORT) -> None:
    token, path, created = load_or_create_token()
    if path is None:
        print("[cc-agy-gateway] Using token from environment.", file=sys.stderr, flush=True)
    else:
        verb = "Created" if created else "Using"
        print(f"[cc-agy-gateway] {verb} token file: {path}", file=sys.stderr, flush=True)
    server = ReusableGatewayHTTPServer(("127.0.0.1", port), GatewayHandler, token=token)
    print(f"[cc-agy-gateway {VERSION}] Listening on http://127.0.0.1:{port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Anthropic Messages API gateway for cc-agy-plugin.")
    parser.add_argument("--port", type=int, default=GATEWAY_PORT, help="Port to listen on (default: %(default)s)")
    parser.add_argument("--print-token-path", action="store_true", help="Print the gateway token file path and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.print_token_path:
        print(token_file_path())
        return 0
    run_server(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
