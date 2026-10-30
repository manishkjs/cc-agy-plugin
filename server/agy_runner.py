"""Antigravity SDK worker adapter with crash-safe usage/tool extraction."""

from __future__ import annotations

import asyncio
import glob
import hashlib
import hmac
import inspect
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Mapping
import urllib.parse

from server.genai_client import extract_usage_metadata
from server.regions import (
    env_flag,
    is_strict_mode_enabled,
    resolve_auth_config,
    resolve_cost_profile,
    resolve_vertex_base_url,
    resolve_vertex_base_url_token,
    sanitize_placeholder_env,
)

_WORKER_RUNS_SECRET = secrets.token_bytes(32)


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def sign_worker_run_record(rec: dict[str, Any]) -> str:
    """Compute an in-process HMAC over a worker run record so forged worktree files are rejected."""
    raw_hashes = rec.get("file_hashes")
    clean_hashes = (
        {str(k): str(v) for k, v in raw_hashes.items() if isinstance(k, str) and isinstance(v, str)}
        if isinstance(raw_hashes, dict)
        else {}
    )
    payload = json.dumps(
        {
            "ts": round(float(rec.get("ts") or 0.0), 6),
            "command": str(rec.get("command") or ""),
            "working_directory": str(rec.get("working_directory") or ""),
            "exit_code": rec.get("exit_code"),
            "timed_out": bool(rec.get("timed_out", False)),
            "output_tail": str(rec.get("output_tail") or ""),
            "file_hashes": clean_hashes,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hmac.new(_WORKER_RUNS_SECRET, payload, hashlib.sha256).hexdigest()


def verify_worker_run_record(rec: dict[str, Any]) -> bool:
    """Verify that a worker run record was signed by this host process and has a plausible timestamp."""
    if not isinstance(rec, dict):
        return False
    sig = str(rec.get("hmac") or "")
    if not sig:
        return False
    try:
        rec_ts = float(rec.get("ts") or 0.0)
    except (TypeError, ValueError):
        return False
    if rec_ts <= 0.0 or rec_ts > time.time() + 1.0:
        return False
    expected = sign_worker_run_record(rec)
    return hmac.compare_digest(sig, expected)


async def maybe_await(value: Any) -> Any:
    """Resolve value whether it is a plain return value, coroutine, or awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value


THINKING_LEVELS = ("ZERO", "MINIMAL", "LOW", "MEDIUM", "HIGH", "EXTRA_HIGH")


def _normalize_thinking_token(raw: str | None) -> str:
    clean = (raw or "").strip().upper()
    if clean in ("0", "OFF"):
        return "ZERO"
    return clean
PROGRESS_KINDS = ("milestone", "question", "blocker", "spec_conflict")
_progress_lock = threading.Lock()


def append_progress(progress_path: str | Path | None, kind: str, message: str) -> str:
    """Append one worker -> lead signal (milestone / question / blocker / spec_conflict) as a JSON line."""
    if not progress_path:
        return "No progress channel is configured for this job."
    clean_kind = str(kind or "milestone").strip().lower()
    if clean_kind not in PROGRESS_KINDS:
        clean_kind = "milestone"
    entry = {"ts": round(time.time(), 3), "kind": clean_kind, "message": str(message or "")[:2000]}
    with _progress_lock:
        Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
        with open(progress_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    if clean_kind == "spec_conflict":
        return (
            "Specification conflict reported to the lead. Stop now: end your turn immediately with the "
            "verified runtime output that contradicts the prompt/checklist so the lead can resolve it."
        )
    if clean_kind in ("question", "blocker"):
        return (
            "Reported to the lead. Stop now: end your turn with a short summary of what is done and what "
            "you need. The lead will reply with a follow-up."
        )
    return "Milestone recorded. Continue."


def read_progress(progress_path: str | Path | None, limit: int = 8) -> list[dict[str, Any]]:
    """Return the most recent worker -> lead signals for a job."""
    if not progress_path or not Path(progress_path).exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in Path(progress_path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-max(1, limit):]


WORKER_SYSTEM_PROMPT = (
    "Headless coding worker in an isolated repo copy with Straitjacket tools. "
    "Use the repo map, candidate files, and tools; read bounded output refs (run:<id>#stdout) "
    "via ctx_get/ctx_search instead of rerunning; keep edits in allowed_paths; reply <250 words."
)


def resolve_containment_profile(
    profile: str | None = None,
    *,
    enabled_tools: list[str] | None = None,
    disabled_tools: list[str] | None = None,
) -> str:
    """Resolve the active Straitjacket profile (`full`, `read-only`, or `command-output`).

    Defaults to `read-only` when all mutating tools are excluded, and `full` otherwise
    (unless overridden by an explicit `profile` argument or `CC_AGY_CONTAINMENT_PROFILE`).
    """
    explicit = (profile or os.environ.get("CC_AGY_CONTAINMENT_PROFILE") or "").strip().lower()
    if explicit in ("command-output", "full", "read-only"):
        return explicit
    if is_read_only_tool_set(enabled_tools, disabled_tools):
        return "read-only"
    return "full"


def build_worker_system_prompt(
    ws_str: str,
    *,
    profile: str | None = None,
    has_report_to_lead: bool = False,
) -> str:
    """Build a workspace-aware system instruction block for the active Straitjacket tool profile."""
    eff_profile = resolve_containment_profile(profile)
    if eff_profile == "read-only":
        tool_list = "ctx_get, ctx_search, view_file, search_dir, list_dir, find_file, ctx_query"
    elif eff_profile == "full":
        tool_list = "run_command, ctx_get, ctx_search, view_file, search_dir, list_dir, find_file, ctx_query, edit_file, create_file"
    else:
        tool_list = "run_command, ctx_get, ctx_search, view_file, edit_file, create_file"
    if has_report_to_lead:
        tool_list += ", report_to_lead"
    return (
        f"{WORKER_SYSTEM_PROMPT}\n"
        f"- Root: {ws_str}\n"
        f"- Tools ({eff_profile}): {tool_list}.\n"
        "- Edit early: skip view_file on inlined files; batch view_file (1..260 lines); emit edit_file/create_file (<1,500 thinking tokens/turn).\n"
        "- Verify: skip `python3 -c`/`node -e` probes; run Acceptance tests and stop once green.\n"
        "- Contracts: never shadow `Cls.attr` via `self.attr = ...` in `__init__` from `os.getcwd()`/`os.environ`/defaults unless passed by caller (keeps subclassing & `monkeypatch.setattr(Cls, 'attr', ...)` intact); never alter default paths for test `cwd` (use `tmp_path`/`monkeypatch`).\n"
        "- Non-interactive: call report_to_lead(kind='spec_conflict'|'question'|'blocker', ...) only if blocked.\n"
        "- Hygiene: never touch .env*, .mcp.json, .netrc, *.pem, *.key, .claude/settings*.json, or git commit/reset/checkout/clean."
    )


def is_read_only_tool_set(
    enabled_tools: list[str] | None,
    disabled_tools: list[str] | None,
) -> bool:
    """Return True when the caller excluded every file-writing and command-running tool."""
    mutating_names = {"CREATE_FILE", "EDIT_FILE", "RUN_COMMAND"}
    if enabled_tools is not None:
        up = {str(t).strip().upper() for t in enabled_tools if str(t).strip()}
        if up and not (up & mutating_names):
            return True
    if disabled_tools is not None:
        dis = {str(t).strip().upper() for t in disabled_tools if str(t).strip()}
        if mutating_names.issubset(dis):
            return True
    return False


def resolve_thinking_level(
    value: str | None,
    *,
    model: str | None = None,
    read_only: bool = False,
) -> str | None:
    """Return the worker thinking level.

    Defaults to ``LOW`` for mutating coding jobs (`read_only=False`) so multi-turn workers
    act and run tests empirically without 35k+ token single-turn thinking stalls, and
    ``HIGH`` for read-only investigation (`read_only=True`). ``"DEFAULT"`` / ``"NONE"``
    leaves it to the model's own default.
    """
    raw_env = (os.environ.get("CC_AGY_DELEGATE_THINKING_LEVEL") or "").strip()
    if raw_env.startswith("${"):
        raw_env = ""
    model_low = (model or "").strip().lower()
    clean_val = _normalize_thinking_token(value)
    esc_env = _normalize_thinking_token(os.environ.get("CC_AGY_ESCALATION_THINKING_LEVEL"))
    if not read_only and "flash" not in model_low:
        if esc_env and not esc_env.startswith("${") and esc_env in THINKING_LEVELS:
            return clean_val if clean_val in THINKING_LEVELS else esc_env
        if model_low.endswith("-zero"):
            return clean_val if clean_val in THINKING_LEVELS else "ZERO"
    forced_env = _normalize_thinking_token(os.environ.get("CC_AGY_FORCE_WORKER_THINKING_LEVEL"))
    if not read_only and forced_env in THINKING_LEVELS:
        return forced_env
    explicit_model_thinking = (
        "HIGH" if model_low.endswith("-high")
        else "MEDIUM" if model_low.endswith("-medium")
        else "LOW" if model_low.endswith("-low")
        else "ZERO" if model_low.endswith("-zero")
        else None
    )
    if not read_only and explicit_model_thinking:
        if clean_val in THINKING_LEVELS:
            return clean_val
        return explicit_model_thinking
    default_level = "HIGH" if read_only else ("MEDIUM" if resolve_cost_profile() == "quality" else "LOW")
    raw = _normalize_thinking_token(value or raw_env or default_level)
    if raw.startswith("${"):
        raw = default_level
    if raw in ("DEFAULT", "NONE", ""):
        return None
    if raw not in THINKING_LEVELS:
        raise ValueError(f"thinking_level must be one of {THINKING_LEVELS} or DEFAULT, got {value!r}")
    return raw


async def resolve_turn_usage(resp: Any) -> dict[str, int]:
    """Read one chat turn's usage from `resp.usage_metadata` (awaitable on google-antigravity 0.1.x)."""
    usage = {
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "total_token_count": 0,
    }
    if resp is None:
        return usage
    try:
        raw = await maybe_await(getattr(resp, "usage_metadata", None))
    except Exception:
        return usage
    return extract_usage_metadata(raw)


def extract_usage_from_session(agent: Any, resp: Any = None) -> dict[str, int]:
    """Extract raw Vertex token counters from an Antigravity Agent or Response.

    Safe to call inside a `finally:` block even when `resp.resolve()` timed out or
    raised an exception mid-turn, inspecting both the response object and any
    completed turns recorded on the agent session.
    """
    usage = {
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "total_token_count": 0,
    }

    candidates = []
    if resp is not None:
        candidates.append(resp)
        for attr in ("usage_metadata", "usage", "_usage"):
            sub = getattr(resp, attr, None)
            if sub is not None:
                candidates.append(sub)

    if agent is not None:
        candidates.append(agent)
        for attr in ("usage_metadata", "usage", "total_usage", "_usage", "session"):
            sub = getattr(agent, attr, None)
            if sub is not None:
                candidates.append(sub)
        conv = getattr(agent, "_conversation", None)
        if conv is None:
            try:
                conv = getattr(agent, "conversation", None)
            except Exception:
                conv = None
        if conv is not None:
            for attr in ("last_turn_usage", "total_usage", "_last_turn_usage"):
                try:
                    sub = getattr(conv, attr, None)
                    if sub is not None:
                        candidates.append(sub)
                except Exception:
                    pass
            conn = getattr(conv, "_connection", None)
            if conn is not None:
                try:
                    sub = getattr(conn, "cumulative_usage", None)
                    if sub is not None:
                        candidates.append(sub)
                except Exception:
                    pass

    for obj in candidates:
        if isinstance(obj, dict):
            for k in usage:
                val = int(obj.get(k) or 0)
                if val > usage[k]:
                    usage[k] = val
        else:
            for k in usage:
                val = int(getattr(obj, k, 0) or 0)
                if val > usage[k]:
                    usage[k] = val

    if usage["total_token_count"] == 0:
        usage["total_token_count"] = (
            usage["prompt_token_count"]
            + usage["candidates_token_count"]
            + usage["thoughts_token_count"]
        )
    return usage


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    val = 0
    shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7
    return val, pos


def _parse_proto_fields(buf: bytes) -> dict[int, int | bytes]:
    pos = 0
    out: dict[int, int | bytes] = {}
    while pos < len(buf):
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 7
        if wire_type == 0:
            v, pos = _read_varint(buf, pos)
            out[field_num] = v
        elif wire_type == 2:
            ln, pos = _read_varint(buf, pos)
            out[field_num] = buf[pos : pos + ln]
            pos += ln
        elif wire_type == 1:
            pos += 8
        elif wire_type == 5:
            pos += 4
        else:
            break
    return out


_MAX_CREDIBLE_STEP_INPUT_TOKENS = 20_000_000
_MAX_CREDIBLE_STEP_OUTPUT_TOKENS = 5_000_000


def extract_sqlite_session_telemetry(save_dir: str | Path | None) -> dict[str, Any]:
    """Extract per-call usage requests and tool calls from localharness SQLite DB in save_dir."""
    out: dict[str, Any] = {
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "total_token_count": 0,
        "max_request_prompt_tokens": 0,
        "requests": [],
        "tool_calls": [],
    }
    if not save_dir or not Path(save_dir).exists():
        return out
    required_cols = {"idx", "step_type", "metadata", "step_payload"}
    for db_path in sorted(glob.glob(os.path.join(str(save_dir), "*.db"))):
        conn = None
        try:
            ro_uri = "file:" + urllib.parse.quote(os.path.abspath(db_path)) + "?mode=ro"
            conn = sqlite3.connect(ro_uri, uri=True)
            col_rows = conn.execute("PRAGMA table_info(steps)").fetchall()
            col_names = {str(r[1]) for r in col_rows if len(r) > 1}
            if not required_cols.issubset(col_names):
                continue
            rows = conn.execute(
                "SELECT idx, step_type, metadata, step_payload FROM steps ORDER BY idx"
            ).fetchall()
        except Exception:
            continue
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        for _idx, step_type, meta, payload in rows:
            if step_type == 15 and isinstance(meta, (bytes, bytearray)) and meta:
                top = _parse_proto_fields(bytes(meta))
                u_bytes = top.get(9)
                if isinstance(u_bytes, bytes) and u_bytes:
                    u = _parse_proto_fields(u_bytes)
                    uncached_in = int(u.get(2) or 0) if isinstance(u.get(2), int) else 0
                    cached_in = int(u.get(5) or 0) if isinstance(u.get(5), int) else 0
                    thoughts_tok = int(u.get(9) or 0) if isinstance(u.get(9), int) else 0
                    cand_tok = int(u.get(10) or 0) if isinstance(u.get(10), int) else 0
                    if cand_tok == 0 and thoughts_tok == 0 and isinstance(u.get(3), int):
                        cand_tok = int(u.get(3) or 0)
                    prompt_tok = uncached_in + cached_in
                    tot_tok = prompt_tok + cand_tok + thoughts_tok
                    # Guard against protobuf field-number reuse (e.g. a 64-bit nanosecond epoch in tag 2)
                    if (
                        uncached_in < 0
                        or cached_in < 0
                        or cand_tok < 0
                        or thoughts_tok < 0
                        or uncached_in > _MAX_CREDIBLE_STEP_INPUT_TOKENS
                        or cached_in > _MAX_CREDIBLE_STEP_INPUT_TOKENS
                        or prompt_tok > _MAX_CREDIBLE_STEP_INPUT_TOKENS
                        or cand_tok > _MAX_CREDIBLE_STEP_OUTPUT_TOKENS
                        or thoughts_tok > _MAX_CREDIBLE_STEP_OUTPUT_TOKENS
                    ):
                        continue
                    if tot_tok > 0:
                        req_entry = {
                            "prompt_token_count": prompt_tok,
                            "cached_content_token_count": cached_in,
                            "candidates_token_count": cand_tok,
                            "thoughts_token_count": thoughts_tok,
                            "total_token_count": tot_tok,
                            "per_call": True,
                        }
                        out["requests"].append(req_entry)
                        out["prompt_token_count"] += prompt_tok
                        out["cached_content_token_count"] += cached_in
                        out["candidates_token_count"] += cand_tok
                        out["thoughts_token_count"] += thoughts_tok
                        out["total_token_count"] += tot_tok
                        out["max_request_prompt_tokens"] = max(
                            int(out["max_request_prompt_tokens"]),
                            prompt_tok,
                        )
            elif step_type in (5, 8, 21, 103) and isinstance(payload, (bytes, bytearray)) and payload:
                try:
                    top_p = _parse_proto_fields(bytes(payload))
                    inner_b = top_p.get(5)
                    if isinstance(inner_b, bytes):
                        inner_p = _parse_proto_fields(inner_b)
                        tc_b = inner_p.get(4)
                        if isinstance(tc_b, bytes):
                            tc_p = _parse_proto_fields(tc_b)
                            t_name = tc_p.get(2).decode("utf-8", errors="ignore") if isinstance(tc_p.get(2), bytes) else "tool"
                            t_args_raw = tc_p.get(3).decode("utf-8", errors="ignore") if isinstance(tc_p.get(3), bytes) else "{}"
                            try:
                                t_args = json.loads(t_args_raw)
                            except Exception:
                                t_args = {}
                            out["tool_calls"].append({"name": t_name, "args": t_args})
                except Exception:
                    pass
    return out


async def drain_response_streams(
    resp: Any,
    tool_calls_out: list[dict[str, Any]],
    thoughts_out: list[str],
) -> None:
    """Drain async or sync `resp.tool_calls` and `resp.thoughts` streams safely."""
    if resp is None:
        return

    tc_stream = getattr(resp, "tool_calls", None)
    if tc_stream is not None:
        try:
            if hasattr(tc_stream, "__aiter__"):
                async for tc in tc_stream:
                    tool_calls_out.append(_serialize_tool_call(tc))
            elif hasattr(tc_stream, "__iter__"):
                for tc in tc_stream:
                    tool_calls_out.append(_serialize_tool_call(tc))
        except (Exception, asyncio.CancelledError):
            pass

    th_stream = getattr(resp, "thoughts", None)
    if th_stream is not None:
        try:
            if hasattr(th_stream, "__aiter__"):
                async for th in th_stream:
                    thoughts_out.append(str(getattr(th, "text", th)))
            elif hasattr(th_stream, "__iter__"):
                for th in th_stream:
                    thoughts_out.append(str(getattr(th, "text", th)))
        except (Exception, asyncio.CancelledError):
            pass


def _serialize_tool_call(tc: Any) -> dict[str, Any]:
    if isinstance(tc, dict):
        return {
            "name": str(tc.get("name") or "unknown"),
            "args": tc.get("args") or {},
        }
    return {
        "name": str(getattr(tc, "name", None) or getattr(tc, "tool_name", "unknown")),
        "args": getattr(tc, "args", None) or getattr(tc, "arguments", {}) or {},
    }


def is_straitjacket_available() -> bool:
    """Return True if Straitjacket (`ctx.sdk.ToolSession`) is installed and importable."""
    try:
        from ctx.sdk import ToolSession  # noqa: F401
        return True
    except (ImportError, Exception):
        return False


def _coerce_line_number(value: Any, default: int) -> int:
    """Safely coerce a model-provided line number (`None`, int, float, float-string) to a 1-indexed int."""
    if value is None or isinstance(value, bool):
        return max(1, default)
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return max(1, default)


_SECRET_BASENAMES = frozenset({
    ".env", ".netrc", ".npmrc", ".pypirc", ".mcp.json", ".hmac_key",
    "id_rsa", "id_ed25519", "id_ecdsa", "credentials.json",
    "antigravity-oauth-token",
})
_EXTRA_CTX_DENY_GLOBS = (
    ".netrc", "**/.netrc",
    ".npmrc", "**/.npmrc",
    ".pypirc", "**/.pypirc",
    ".mcp.json", "**/.mcp.json",
    ".hmac_key", "**/.hmac_key",
    "id_ecdsa*", "**/id_ecdsa*",
    "credentials.json", "**/credentials.json",
    "antigravity-oauth-token", "**/antigravity-oauth-token",
    ".gemini/antigravity-cli/**", "**/.gemini/antigravity-cli/**",
    "*.p12", "**/*.p12",
    "*.pfx", "**/*.pfx",
    ".claude/settings.local.json", "**/.claude/settings.local.json",
    ".cc-agy-data/**", "**/.cc-agy-data/**",
)


def build_straitjacket_tools(
    workspace_dir: str | Path,
    timeout_seconds: float | str = 300.0,
    *,
    profile: str | None = None,
    is_lead_orchestrator: bool = False,
    lead_read_limit: int = 12,
    acceptance_commands: list[str] | None = None,
    host_workspace_root: str | Path | None = None,
    nm_linked: bool = False,
    isolated_worktree: bool = False,
    live_state_path: str | Path | None = None,
    save_dir: str | Path | None = None,
    max_tool_calls: int | None = None,
) -> tuple[list[Any], list[str]]:
    """Build Straitjacket (ctx.sdk.ToolSession) async tools and return (tools, replaced_builtin_names).

    Fixes three integration hazards:
    1. Clamps per-command timeout to `session.timeout` so a default 300s turn never raises
       `ValueError("command timeout exceeds the session timeout")` when `run_command` defaults to 600s.
    2. Coerces `start_line` / `end_line` in `ctx_get` to 1-indexed ints (`1 <= start_line <= end_line`)
       so 0-indexed, None, or float tool arguments from the model do not fail `ToolSession.get` type checks.
    3. Supports `command-output` (default), `read-only`, and `full` (`CC_AGY_CONTAINMENT_PROFILE=full`) profiles,
       while blocking `ws:` alias escapes, `..` traversal, and credential files.
    """
    import ctx.config as _ctx_cfg
    from ctx.sdk import ToolSession

    merged_globs = tuple(dict.fromkeys(tuple(_ctx_cfg.BUILTIN_DENY_GLOBS) + _EXTRA_CTX_DENY_GLOBS))
    if _ctx_cfg.BUILTIN_DENY_GLOBS != merged_globs:
        _ctx_cfg.BUILTIN_DENY_GLOBS = merged_globs

    resolved_profile = (profile or os.environ.get("CC_AGY_CONTAINMENT_PROFILE") or "command-output").strip().lower()
    if resolved_profile not in ("command-output", "full", "read-only"):
        resolved_profile = "command-output"

    try:
        numeric_timeout = 300.0 if isinstance(timeout_seconds, str) else float(timeout_seconds or 600.0)
    except (TypeError, ValueError):
        numeric_timeout = 300.0
    session_timeout = max(60.0, numeric_timeout)
    env_cmd_timeout = os.environ.get("CC_AGY_DEFAULT_CMD_TIMEOUT", "").strip()
    try:
        parsed_cmd_timeout = float(env_cmd_timeout) if env_cmd_timeout else 600.0
    except ValueError:
        parsed_cmd_timeout = 600.0
    default_cmd_timeout = min(max(1.0, parsed_cmd_timeout), session_timeout)
    ws_resolved = Path(workspace_dir).resolve()
    host_ws_resolved: Path | None = None
    if host_workspace_root:
        try:
            cand_host = Path(host_workspace_root).resolve()
            if cand_host != ws_resolved:
                host_ws_resolved = cand_host
        except Exception:
            host_ws_resolved = None
    extra_blocked_list = [
        p.strip()
        for p in os.environ.get("CC_AGY_BLOCKED_PATHS", "").split(",")
        if p.strip()
    ]
    if isolated_worktree and host_ws_resolved is not None:
        extra_blocked_list.append(str(host_ws_resolved))
    extra_blocked_paths = tuple(dict.fromkeys(extra_blocked_list))
    scrubbed_env: dict[str, str] = {
        "CC_AGY_DATA_DIR": "",
        "CC_AGY_OUT_DIR": "",
        "CC_AGY_WORKTREES_DIR": "",
        "CLAUDE_PLUGIN_DATA": "",
        "GEMINI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "CLAUDE_PLUGIN_OPTION_GEMINI_API_KEY": "",
        "GOOGLE_APPLICATION_CREDENTIALS": "",
        "ANTIGRAVITY_OAUTH_TOKEN": "",
        "PWD": "",
        "OLDPWD": "",
    }
    for env_k in list(os.environ.keys()):
        if env_k.startswith(("CC_AGY_", "CLAUDE_PLUGIN_")) or env_k in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "ANTIGRAVITY_OAUTH_TOKEN",
            "PWD",
            "OLDPWD",
        ):
            scrubbed_env[env_k] = ""
    # Same credential policy as acceptance and batch checks: blank every credential-looking variable
    # (GITHUB_TOKEN, AWS keys, database passwords, ...) unless CC_AGY_CHECK_ENV_ALLOW lists it.
    from server.verify import secret_env_names

    for env_k in secret_env_names(os.environ):
        scrubbed_env[env_k] = ""
    py_parts = [str(ws_resolved)]
    if (ws_resolved / "src").is_dir():
        py_parts.append(str(ws_resolved / "src"))
    existing_py = (os.environ.get("PYTHONPATH") or "").strip()
    if existing_py:
        py_parts.append(existing_py)
    scrubbed_env["PYTHONPATH"] = ":".join(py_parts)
    from server.jobs import _collect_workspace_node_bin_dirs

    path_prefixes: list[str] = _collect_workspace_node_bin_dirs(ws_resolved)
    venv_bin = ws_resolved / ".venv" / "bin"
    if venv_bin.is_dir():
        path_prefixes.append(str(venv_bin))
        scrubbed_env["VIRTUAL_ENV"] = str(ws_resolved / ".venv")
    if path_prefixes:
        existing_path = (os.environ.get("PATH") or "").strip()
        prefix_joined = ":".join(path_prefixes)
        scrubbed_env["PATH"] = f"{prefix_joined}:{existing_path}" if existing_path else prefix_joined
    from dataclasses import replace as _dc_replace
    import hashlib as _hashlib

    sj_result_tokens = max(200, _env_int("CC_AGY_SJ_RESULT_TOKENS", 6000))
    sj_turn_retrieval_tokens = max(
        sj_result_tokens,
        _env_int("CC_AGY_SJ_TURN_RETRIEVAL_TOKENS", 40000),
    )
    sj_max_inline_lines = max(50, _env_int("CC_AGY_SJ_MAX_INLINE_LINES", 400))
    sj_max_inline_bytes = max(4096, _env_int("CC_AGY_SJ_MAX_INLINE_BYTES", 32768))
    sj_session_read_bytes = max(sj_max_inline_bytes, _env_int("CC_AGY_SJ_SESSION_READ_BYTES", 1048576))

    class _BoundedToolSession(ToolSession):
        def _workspace(self):
            ws = super()._workspace()
            b = ws.config.budgets
            budget_updates: dict[str, Any] = {
                "result_tokens": sj_result_tokens,
                "turn_retrieval_tokens": sj_turn_retrieval_tokens,
                "max_inline_lines": sj_max_inline_lines,
                "max_inline_bytes": sj_max_inline_bytes,
            }
            if hasattr(b, "max_tool_output_bytes"):
                budget_updates["max_tool_output_bytes"] = max(
                    int(getattr(b, "max_tool_output_bytes", 16384) or 16384),
                    sj_max_inline_bytes,
                )
            if hasattr(b, "session_read_budget_bytes"):
                budget_updates["session_read_budget_bytes"] = max(
                    int(getattr(b, "session_read_budget_bytes", 262144) or 262144),
                    sj_session_read_bytes,
                )
            return _dc_replace(
                ws,
                config=_dc_replace(
                    ws.config,
                    budgets=_dc_replace(b, **budget_updates),
                ),
            )

    session = _BoundedToolSession(
        ws_resolved,
        timeout=session_timeout,
        env=scrubbed_env,
        producer="cc-agy-plugin",
    )

    disable_web_tools = env_flag("CC_AGY_DISABLE_WEB_TOOLS") is True or (
        is_strict_mode_enabled() and env_flag("CC_AGY_DISABLE_WEB_TOOLS") is not False
    )
    import re as _re
    _net_bin_re = _re.compile(
        r"(?:^|[\s;|&`(]|/)(?:curl|wget|nc|ncat|netcat|ssh|scp|sftp|rsync)(?=$|[\s;|&`)])"
    )
    _git_net_re = _re.compile(
        r"(?:^|[\s;|&`(])git(?:\s+-[^\s]+)*\s+(?:clone|fetch|push|pull|ls-remote)(?=$|[\s;|&`)])"
    )
    _direct_network_tokens = (
        "urllib.request",
        "urllib.urlopen",
        "socket.create_connection",
        "huggingface.co",
        "githubusercontent.com",
        "api.github.com",
    )
    _external_url_re = _re.compile(
        r"https?://(?!(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|testserver|test\b))"
        r"[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    )
    _plugin_internal_escape_tokens = (
        "/tmp/cc-agy-",
        ".cc-agy-worktrees-",
        ".cc-agy-data",
        ".hmac_key",
        "antigravity-oauth-token",
        ".gemini/antigravity-cli",
        "/google/",
    ) + extra_blocked_paths
    _root_scan_re = _re.compile(
        r"(?:^|[\s;|&`(])(?:find|grep|rg|ripgrep)\b[^\n;|&`]*?(?:^|\s)/(?:\s|$|[;|&)>])"
    )

    def _is_outbound_network_command(raw_cmd: str) -> bool:
        if _net_bin_re.search(raw_cmd) or _git_net_re.search(raw_cmd):
            return True
        if any(tok in raw_cmd for tok in _direct_network_tokens):
            return True
        if ("MockTransport" in raw_cmd or "ASGITransport" in raw_cmd or "WSGITransport" in raw_cmd):
            return False
        if any(lib in raw_cmd for lib in ("requests.get(", "requests.post(", "httpx.get(", "httpx.post(", "http.client")):
            if _external_url_re.search(raw_cmd):
                return True
        return False

    def _matches_abs_blocked(cand_p: Path) -> bool:
        s_p = str(cand_p)
        for bp in _plugin_internal_escape_tokens:
            if not bp.startswith("/"):
                continue
            if bp.endswith("-"):
                if s_p.startswith(bp):
                    return True
            else:
                bp_c = bp.rstrip("/")
                if bp_c and (s_p == bp_c or s_p.startswith(bp_c + "/")):
                    return True
        return False

    def _is_host_path_escape(raw_cmd: str) -> bool:
        scrubbed = raw_cmd.replace(str(ws_resolved), "__WS__")
        if any(tok in scrubbed for tok in _plugin_internal_escape_tokens):
            return True
        if _root_scan_re.search(scrubbed):
            return True
        if ".." in scrubbed:
            eff_cwd = ws_resolved
            for seg_c in _re.split(r"(?:&&|\|\||;|\n|\|)", raw_cmd):
                seg_s = seg_c.strip()
                if not seg_s:
                    continue
                try:
                    toks_c = shlex.split(seg_s, posix=True)
                except ValueError:
                    toks_c = seg_s.split()
                if not toks_c:
                    continue
                if toks_c[0] == "cd" and len(toks_c) >= 2 and not toks_c[1].startswith("-"):
                    try:
                        next_cwd = (eff_cwd / toks_c[1]).resolve()
                    except Exception:
                        if isolated_worktree:
                            return True
                        continue
                    if (isolated_worktree and not next_cwd.is_relative_to(ws_resolved)) or _matches_abs_blocked(next_cwd):
                        return True
                    eff_cwd = next_cwd
                    continue
                for tk in toks_c[1:]:
                    if tk.startswith("-"):
                        continue
                    if tk == ".." or tk.startswith("../") or "/../" in tk or tk.endswith("/.."):
                        try:
                            cand_tk = (eff_cwd / tk).resolve()
                            if (isolated_worktree and not cand_tk.is_relative_to(ws_resolved)) or _matches_abs_blocked(cand_tk):
                                return True
                        except Exception:
                            if isolated_worktree:
                                return True
        return False

    _pkg_mutating_subs = frozenset({
        "install", "i", "ci", "add", "remove", "rm", "uninstall", "un",
        "update", "up", "upgrade", "link", "unlink", "rebuild", "prune",
        "dedupe", "clean-install", "install-clean",
    })
    _pkg_arg_flags = frozenset({"--filter", "-F", "-C", "--prefix", "--cwd", "--dir", "--workspace", "-w"})

    def _touches_non_ephemeral_node_modules(arg_p: str) -> bool:
        norm_p = arg_p.strip().replace("\\", "/").lstrip("./").rstrip("/")
        if not norm_p:
            return False
        parts_p = [seg for seg in norm_p.split("/") if seg and seg != "."]
        if "node_modules" not in parts_p:
            return False
        nm_idx = len(parts_p) - 1 - parts_p[::-1].index("node_modules")
        tail_parts = parts_p[nm_idx + 1 :]
        if tail_parts and tail_parts[0] in (".vite", ".vite-temp", ".cache", ".tmp", ".eslintcache", ".parcel-cache", ".turbo"):
            return False
        return True

    _nm_redirect_re = _re.compile(r"(?:>>?|>\|)\s*['\"]?([^\s;|&'\"]+)")
    _nm_script_write_tokens = (
        "write_text",
        "write_bytes",
        "writeFileSync",
        "writeFile",
        "appendFileSync",
        "rmSync",
        "rmdirSync",
        "unlinkSync",
        "rmtree",
        "shutil.move",
    )

    def _is_linked_node_modules_mutation(raw_cmd: str) -> bool:
        for redir_m in _nm_redirect_re.finditer(raw_cmd):
            if _touches_non_ephemeral_node_modules(redir_m.group(1)):
                return True
        if "node_modules" in raw_cmd and any(tok in raw_cmd for tok in _nm_script_write_tokens):
            return True
        for seg_m in _re.split(r"(?:&&|\|\||;|\n|\|)", raw_cmd):
            seg_s = seg_m.strip()
            if not seg_s:
                continue
            try:
                toks_m = shlex.split(seg_s, posix=True)
            except ValueError:
                toks_m = seg_s.split()
            i_m = 0
            while i_m < len(toks_m):
                tk0 = toks_m[i_m]
                if "=" in tk0 and not tk0.startswith("-") and tk0.split("=", 1)[0].isidentifier():
                    i_m += 1
                    continue
                if Path(tk0).name.lower() in ("env", "nohup", "time"):
                    i_m += 1
                    continue
                break
            if i_m >= len(toks_m):
                continue
            exe_m = Path(toks_m[i_m]).name.lower()
            rest_m = toks_m[i_m + 1 :]
            if exe_m in ("npm", "pnpm", "yarn", "bun"):
                j_m = 0
                sub_m = ""
                while j_m < len(rest_m):
                    rt = rest_m[j_m]
                    if rt in _pkg_arg_flags and j_m + 1 < len(rest_m):
                        j_m += 2
                        continue
                    if rt.startswith("-"):
                        j_m += 1
                        continue
                    sub_m = rt.lower()
                    break
                if exe_m == "yarn" and not sub_m:
                    return True
                if sub_m in _pkg_mutating_subs:
                    return True
            elif exe_m in ("rm", "rmdir", "unlink", "mv", "chmod", "chown", "tee", "touch", "truncate"):
                operands = [a for a in rest_m if not a.startswith("-")]
                if any(_touches_non_ephemeral_node_modules(op) for op in operands):
                    return True
            elif exe_m == "sed" and any(a == "-i" or a.startswith("-i") for a in rest_m):
                operands = [a for a in rest_m if not a.startswith("-")]
                if any(_touches_non_ephemeral_node_modules(op) for op in operands):
                    return True
            elif exe_m in ("cp", "ln"):
                operands = [a for a in rest_m if not a.startswith("-")]
                if operands and _touches_non_ephemeral_node_modules(operands[-1]):
                    return True
        return False

    from server.verify import run_git as _git_run
    _session_started_mono = time.monotonic()
    _straitjacket_state = {"calls": 0, "last_git_check": 0.0, "has_edits": False}
    _live_state_p = Path(live_state_path) if live_state_path else None

    def _persist_live_tool_count(calls_count: int) -> None:
        if _live_state_p is None:
            return
        try:
            _live_state_p.parent.mkdir(parents=True, exist_ok=True)
            tmp_p = _live_state_p.with_suffix(".live.tmp")
            tmp_p.write_text(
                json.dumps(
                    {
                        "straitjacket_calls": int(calls_count),
                        "save_dir": str(save_dir or ""),
                        "updated_at_epoch": round(time.time(), 3),
                    }
                ),
                encoding="utf-8",
            )
            os.replace(tmp_p, _live_state_p)
        except Exception:
            pass

    def _maybe_append_turn_budget_note(raw_output: str) -> str:
        if resolved_profile == "read-only":
            if not is_lead_orchestrator:
                return raw_output
            _straitjacket_state["calls"] += 1
            calls = _straitjacket_state["calls"]
            if calls >= 3:
                return (
                    f"{raw_output}\n"
                    f"[Lead budget: {calls}/{lead_read_limit} read/search calls used: once you know the target files, "
                    "call delegate_start with a concise prompt (<500 words, no 'Wait:' self-debate); "
                    "the worker has full repository and test access.]"
                )
            return raw_output
        _straitjacket_state["calls"] += 1
        calls = _straitjacket_state["calls"]
        _persist_live_tool_count(calls)
        elapsed = max(0.0, time.monotonic() - _session_started_mono)
        tool_cap = int(max_tool_calls or 0)
        near_tool_cap = bool(tool_cap > 0 and calls >= max(12, int(tool_cap * 0.75)))
        if calls < 8 and elapsed < session_timeout * 0.35 and not near_tool_cap:
            return raw_output
        now_m = time.monotonic()
        if not _straitjacket_state["has_edits"] and (now_m - _straitjacket_state["last_git_check"] >= 4.0):
            _straitjacket_state["last_git_check"] = now_m
            try:
                gs = _git_run(
                    ["status", "--porcelain", "--untracked-files=normal"],
                    cwd=str(ws_resolved),
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                if (gs.stdout or "").strip():
                    _straitjacket_state["has_edits"] = True
            except Exception:
                pass
        if not _straitjacket_state["has_edits"] and (calls >= 8 or elapsed >= session_timeout * 0.35 or near_tool_cap):
            return (
                f"{raw_output}\n"
                f"[Turn budget: {int(elapsed)}s/{int(session_timeout)}s elapsed, {calls}"
                f"{f'/{tool_cap}' if tool_cap > 0 else ''} Straitjacket calls, "
                "0 files modified in worktree: stop searching/reading and write your implementation with "
                "edit_file / create_file NOW so your edits are saved on disk before timeout.]"
            )
        if _straitjacket_state["has_edits"] and (elapsed >= session_timeout * 0.75 or near_tool_cap):
            cap_str = f", {calls}/{tool_cap} tool calls used" if tool_cap > 0 else ""
            return (
                f"{raw_output}\n"
                f"[Turn budget: {int(elapsed)}s/{int(session_timeout)}s elapsed{cap_str}: finish your current edit, "
                "run the targeted acceptance test, and conclude your turn before timeout.]"
            )
        return raw_output

    command_runs: list[dict[str, Any]] = []

    def _snapshot_run_hashes() -> tuple[list[str] | None, set[str], dict[str, str]]:
        hash_targets: set[str] = set()
        deleted_set: set[str] = set()
        used_git_status = False
        if (ws_resolved / ".git").exists():
            try:
                gs_res = _git_run(
                    ["status", "--porcelain", "-z", "--untracked-files=all"],
                    cwd=str(ws_resolved),
                    capture_output=True,
                    timeout=2.0,
                    check=False,
                )
                if gs_res.returncode == 0:
                    used_git_status = True
                    parts_z = gs_res.stdout.split(b"\x00")
                    idx_z = 0
                    while idx_z < len(parts_z):
                        raw_e = parts_z[idx_z]
                        idx_z += 1
                        if not raw_e:
                            continue
                        e_str = raw_e.decode("utf-8", errors="surrogateescape")
                        if len(e_str) > 3 and e_str[2] == " ":
                            xy = e_str[:2]
                            rel_p = e_str[3:]
                            if "D" in xy:
                                deleted_set.add(rel_p)
                            else:
                                hash_targets.add(rel_p)
                            if xy[0] in ("R", "C") and idx_z < len(parts_z):
                                old_raw = parts_z[idx_z]
                                idx_z += 1
                                if old_raw and xy[0] == "R":
                                    deleted_set.add(old_raw.decode("utf-8", errors="surrogateescape"))
            except Exception:
                used_git_status = False
        _ephemeral_hash_dirs = frozenset({
            ".git",
            ".ctx-session-reads",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".hypothesis",
            "node_modules",
            ".vite",
            ".vite-temp",
            ".next",
            ".turbo",
            ".cache",
            ".parcel-cache",
            ".eslintcache",
        })

        def _is_ephemeral_cache_path(rel_path: str) -> bool:
            parts = rel_path.split("/")
            if any(seg in _ephemeral_hash_dirs for seg in parts):
                return True
            return rel_path.endswith((".pyc", ".pyo", ".tsbuildinfo"))

        deleted_set = {d for d in deleted_set if not _is_ephemeral_cache_path(d)}
        existing_files: list[str] | None = None
        if not used_git_status:
            existing_files = []
            for cur_r, walk_dirs, walk_files in os.walk(str(ws_resolved), topdown=True, followlinks=False):
                walk_dirs[:] = [d for d in walk_dirs if d not in _ephemeral_hash_dirs]
                cur_rp = Path(cur_r)
                for wf in walk_files:
                    fp_w = cur_rp / wf
                    if fp_w.is_symlink() or not fp_w.is_file():
                        continue
                    rel_s = fp_w.relative_to(ws_resolved).as_posix()
                    if not _is_ephemeral_cache_path(rel_s):
                        existing_files.append(rel_s)
            if len(existing_files) <= 500:
                hash_targets.update(existing_files)
        fhashes: dict[str, str] = {}
        for rel_t in hash_targets:
            if _is_ephemeral_cache_path(rel_t):
                continue
            fp_t = ws_resolved / rel_t
            try:
                if fp_t.is_file():
                    fhashes[rel_t] = hashlib.sha256(fp_t.read_bytes()).hexdigest()
            except OSError:
                pass
        return existing_files, deleted_set, fhashes

    norm_acc_cmds = list(
        dict.fromkeys(
            " ".join(str(c).strip().split())
            for c in (acceptance_commands or [])
            if str(c or "").strip()
        )
    )
    post_acc_guard_enabled = (
        bool(norm_acc_cmds)
        and not is_lead_orchestrator
        and resolved_profile != "read-only"
        and env_flag("CC_AGY_POST_ACCEPTANCE_STOP_GUARD") is not False
    )
    _is_git_ws = (ws_resolved / ".git").exists()
    _baseline_non_git_hashes: dict[str, str] = {}
    if post_acc_guard_enabled and not _is_git_ws:
        try:
            _, _, _baseline_non_git_hashes = _snapshot_run_hashes()
        except Exception:
            _baseline_non_git_hashes = {}
    _no_tests_re = _re.compile(r"\bno tests ran\b|\bcollected 0 items\b|\bRan 0 tests\b", _re.I)
    _py_exe_re = _re.compile(r"^python(?:\d+(?:\.\d+)*)?$", _re.I)
    _redir_re = _re.compile(r"^\d*(?:>>?|<<?|>&|&>|\|>).*$")
    _pytest_val_flags = frozenset({
        "-k", "-m", "-c", "-o", "-p", "-n", "-W", "-r", "-d",
        "--tb", "--maxfail", "--rootdir", "--confcutdir", "--basetemp",
        "--durations", "--import-mode", "--capture", "--color", "--cov",
        "--cov-report", "--cov-config", "--junitxml", "--log-cli-level",
        "--timeout", "--tx", "--dist", "--ignore", "--ignore-glob", "--deselect",
    })
    _pager_exes = frozenset({"head", "tail", "cat", "less", "more", "wc", "nl"})

    def _split_shell_segments(cmd: str) -> list[str]:
        segs: list[str] = []
        buf: list[str] = []
        in_sq = False
        in_dq = False
        esc = False
        n = len(cmd)
        idx = 0
        while idx < n:
            ch = cmd[idx]
            if esc:
                buf.append(ch)
                esc = False
                idx += 1
                continue
            if ch == "\\" and not in_sq:
                buf.append(ch)
                esc = True
                idx += 1
                continue
            if ch == "'" and not in_dq:
                in_sq = not in_sq
                buf.append(ch)
                idx += 1
                continue
            if ch == '"' and not in_sq:
                in_dq = not in_dq
                buf.append(ch)
                idx += 1
                continue
            if not in_sq and not in_dq:
                if ch in (";", "\n"):
                    seg_s = "".join(buf).strip()
                    if seg_s:
                        segs.append(seg_s)
                    buf = []
                    idx += 1
                    continue
                if ch in ("&", "|"):
                    if idx + 1 < n and cmd[idx + 1] == ch:
                        seg_s = "".join(buf).strip()
                        if seg_s:
                            segs.append(seg_s)
                        buf = []
                        idx += 2
                        continue
                    if ch == "|":
                        # Avoid splitting `2>&1` or `|&` weirdly: treat `|` (and `|&`) as pipeline separator.
                        seg_s = "".join(buf).strip()
                        if seg_s:
                            segs.append(seg_s)
                        buf = []
                        idx += 2 if (idx + 1 < n and cmd[idx + 1] == "&") else 1
                        continue
            buf.append(ch)
            idx += 1
        tail_s = "".join(buf).strip()
        if tail_s:
            segs.append(tail_s)
        return segs

    def _parse_pytest_args(args: list[str]) -> tuple[str, tuple[str, ...]]:
        pos_targets: list[str] = []
        has_selector = False
        after_dd = False
        i = 0
        while i < len(args):
            tok = args[i]
            if after_dd:
                pos_targets.append(tok)
                i += 1
                continue
            if tok == "--":
                after_dd = True
                i += 1
                continue
            if tok.startswith("-"):
                if (
                    tok in (
                        "-k", "-m", "--lf", "--ff", "--last-failed", "--failed-first",
                        "--co", "--collect-only", "--deselect", "--ignore", "--ignore-glob",
                        "--setup-only", "--setup-plan",
                    )
                    or tok.startswith(("-k=", "-m=", "--deselect=", "--ignore=", "--ignore-glob="))
                    or (len(tok) > 2 and tok[:2] in ("-k", "-m") and not tok.startswith("--"))
                ):
                    has_selector = True
                if "=" not in tok and tok in _pytest_val_flags:
                    i += 2
                    continue
                i += 1
                continue
            pos_targets.append(tok)
            i += 1
        norm_targets: list[str] = []
        for raw_t in pos_targets:
            s_t = raw_t.strip()
            if not s_t:
                continue
            node_part = ""
            path_part = s_t
            if "::" in s_t:
                path_part, node_part = s_t.split("::", 1)
            p_cand = Path(path_part)
            if p_cand.is_absolute():
                try:
                    rel_p = p_cand.resolve().relative_to(ws_resolved).as_posix()
                except Exception:
                    rel_p = path_part.rstrip("/") or "."
            else:
                rel_p = path_part.lstrip("./").rstrip("/") or "."
            norm_targets.append(f"{rel_p}::{node_part}" if node_part else rel_p)
        _suite_roots = frozenset({".", "tests", "test", "src"})
        specific = [
            t for t in norm_targets
            if "::" in t or t not in _suite_roots
        ]
        if has_selector:
            # Selector-filtered run (-k/-m/--lf/--co/--deselect): treat as specific to its exact flag signature.
            return "selector_test", tuple(sorted(specific))
        if specific:
            return "targeted_test", tuple(sorted(specific))
        return "whole_suite", ()

    def _parse_unittest_args(args: list[str]) -> tuple[str, tuple[str, ...]]:
        if not args:
            return "whole_suite", ()
        rem_u = list(args)
        is_discover = False
        if rem_u[0] == "discover":
            is_discover = True
            rem_u = rem_u[1:]
        pos_u: list[str] = []
        has_custom_pattern = False
        i = 0
        while i < len(rem_u):
            tok = rem_u[i]
            if tok in ("-s", "--start-directory", "-t", "--top-level-directory"):
                i += 2
                continue
            if tok in ("-p", "--pattern"):
                if i + 1 < len(rem_u) and rem_u[i + 1] not in ("test*.py", "test_*.py"):
                    has_custom_pattern = True
                i += 2
                continue
            if tok.startswith(("--pattern=",)):
                pat_v = tok.split("=", 1)[1]
                if pat_v not in ("test*.py", "test_*.py"):
                    has_custom_pattern = True
                i += 1
                continue
            if tok.startswith("-"):
                i += 1
                continue
            pos_u.append(tok.lstrip("./") or ".")
            i += 1
        if is_discover:
            return ("selector_test", ()) if has_custom_pattern else ("whole_suite", ())
        if pos_u:
            return "targeted_test", tuple(sorted(pos_u))
        return "whole_suite", ()

    def _classify_post_acc_segment(seg: str) -> tuple[str, tuple[str, ...]]:
        try:
            raw_tokens = shlex.split(seg, posix=True)
        except ValueError:
            return "other", ()
        tokens: list[str] = []
        skip_next = False
        for tok in raw_tokens:
            if skip_next:
                skip_next = False
                continue
            if tok in (">", ">>", "<", "2>", "&>", "1>"):
                skip_next = True
                continue
            if _redir_re.match(tok):
                continue
            tokens.append(tok)
        idx = 0
        n = len(tokens)
        while idx < n:
            tok = tokens[idx]
            if "=" in tok and not tok.startswith("-") and tok.split("=", 1)[0].isidentifier():
                idx += 1
                continue
            base = Path(tok).name.lower()
            if base == "env":
                idx += 1
                while idx < n:
                    et = tokens[idx]
                    if "=" in et and not et.startswith("-"):
                        idx += 1
                    elif et in ("-i", "-0", "--ignore-environment", "--null"):
                        idx += 1
                    elif et in ("-u", "--unset", "-C", "--chdir") and idx + 1 < n:
                        idx += 2
                    elif et.startswith(("--unset=", "--chdir=")):
                        idx += 1
                    else:
                        break
                continue
            if base == "timeout":
                idx += 1
                while idx < n and tokens[idx].startswith("-"):
                    tt = tokens[idx]
                    if tt in ("-k", "-s") and idx + 1 < n:
                        idx += 2
                    else:
                        idx += 1
                if idx < n:
                    idx += 1  # duration
                continue
            if base in ("time", "nohup"):
                idx += 1
                continue
            if base == "nice":
                idx += 1
                if idx + 1 < n and tokens[idx] == "-n":
                    idx += 2
                continue
            if base == "stdbuf":
                idx += 1
                while idx < n and tokens[idx].startswith("-"):
                    idx += 1
                continue
            if base in ("uv", "poetry", "pdm") and idx + 1 < n and tokens[idx + 1] == "run":
                idx += 2
                while idx < n and tokens[idx].startswith("-"):
                    idx += 1
                continue
            break
        if idx >= n:
            return "other", ()
        exe = Path(tokens[idx]).name.lower()
        rest = tokens[idx + 1:]
        if exe in _pager_exes:
            return "pager", ()
        if exe == "git":
            gi = 0
            sub = ""
            while gi < len(rest):
                gt = rest[gi]
                if gt in ("-C", "-c", "--git-dir", "--work-tree", "--namespace") and gi + 1 < len(rest):
                    gi += 2
                    continue
                if gt.startswith("-"):
                    gi += 1
                    continue
                sub = gt.lower()
                break
            if sub in ("diff", "status", "log"):
                return "git_meta", ()
            return "other", ()
        if _py_exe_re.match(exe):
            pi = 0
            while pi < len(rest):
                pt = rest[pi]
                if pt == "-c":
                    return "other", ()
                if pt == "-m" and pi + 1 < len(rest):
                    mod = rest[pi + 1].lower()
                    mod_args = rest[pi + 2:]
                    if mod in ("pytest", "py.test"):
                        return _parse_pytest_args(mod_args)
                    if mod == "unittest":
                        return _parse_unittest_args(mod_args)
                    return "other", ()
                if pt in ("-W", "-X") and pi + 1 < len(rest):
                    pi += 2
                    continue
                if pt.startswith("-"):
                    pi += 1
                    continue
                return "other", ()
            return "other", ()
        if exe in ("pytest", "py.test"):
            return _parse_pytest_args(rest)
        if exe in ("tox", "nox"):
            return "whole_suite", ()
        if exe in ("cargo", "go", "npm", "pnpm", "yarn"):
            if rest and rest[0] == "test":
                extra = [a for a in rest[1:] if not a.startswith("-") and a not in ("--", "./...", ".")]
                return ("targeted_test", tuple(sorted(extra))) if extra else ("whole_suite", ())
            return "other", ()
        return "other", ()

    _acc_covered_targets: set[str] = set()
    _acc_covered_files: set[str] = set()
    _acc_covered_dirs: set[str] = set()
    _acc_target_to_norm_cmds: dict[tuple[str, ...], list[str]] = {}
    for _ac in norm_acc_cmds:
        _ac_segs = _split_shell_segments(_ac)
        if len(_ac_segs) == 1:
            _ac_kind, _ac_tgts = _classify_post_acc_segment(_ac_segs[0])
            if _ac_kind == "targeted_test" and _ac_tgts:
                for _t in _ac_tgts:
                    _acc_covered_targets.add(_t)
                    if "::" not in _t:
                        if _t.endswith((".py", ".txt", ".rst", ".doctest")):
                            _acc_covered_files.add(_t)
                        else:
                            _acc_covered_dirs.add(_t)
                _acc_target_to_norm_cmds.setdefault(_ac_tgts, []).append(_ac)

    _acc_guard_state: dict[str, Any] = {
        "state_key": None,
        "passed_cmds": set(),
        "active": False,
    }

    def _worktree_state_key(
        post_del: set[str] | None = None,
        post_h: dict[str, str] | None = None,
    ) -> tuple[bool, tuple[Any, ...]]:
        if post_del is None or post_h is None:
            try:
                _, post_del, post_h = _snapshot_run_hashes()
            except Exception:
                return False, ((), ())
        if _is_git_ws:
            has_changes = bool(post_h or post_del)
        else:
            has_changes = (post_h != _baseline_non_git_hashes) or bool(post_del)
        return has_changes, (tuple(sorted(post_h.items())), tuple(sorted(post_del)))

    def _is_redundant_post_acc_command(raw_cmd: str) -> bool:
        norm_c = " ".join(raw_cmd.strip().split())
        if norm_c in norm_acc_cmds:
            return True
        segs = _split_shell_segments(raw_cmd)
        if not segs:
            return False
        classified = [_classify_post_acc_segment(s) for s in segs]
        has_git_meta = any(k == "git_meta" for k, _ in classified)
        if has_git_meta and all(k in ("git_meta", "pager") for k, _ in classified):
            return True
        for k, tgts in classified:
            if k == "whole_suite":
                return True
            if k == "targeted_test" and tgts and all(
                (
                    t in _acc_covered_targets
                    or t.split("::", 1)[0] in _acc_covered_files
                    or any(
                        t.split("::", 1)[0] == d or t.split("::", 1)[0].startswith(f"{d}/")
                        for d in _acc_covered_dirs
                    )
                )
                for t in tgts
            ):
                return True
        return False

    def _check_post_acceptance_short_circuit(raw_cmd: str) -> str | None:
        if not post_acc_guard_enabled or not _acc_guard_state["active"]:
            return None
        has_changes, cur_key = _worktree_state_key()
        if not has_changes or cur_key != _acc_guard_state["state_key"]:
            _acc_guard_state["state_key"] = cur_key
            _acc_guard_state["passed_cmds"] = set()
            _acc_guard_state["active"] = False
            return None
        if not _is_redundant_post_acc_command(raw_cmd):
            return None
        return (
            "[ACCEPTANCE_PASSED_STOP_NOW: Command NOT executed (skipped because all configured acceptance "
            "commands already passed on your current modified files). Do NOT claim this skipped command ran "
            "or passed. If all task requirements are implemented, end your turn now; if another requirement "
            "still needs code changes, edit the target file first.]"
        )

    async def run_command(
        command: str,
        working_directory: str = "",
        timeout_seconds: float = default_cmd_timeout,
    ) -> str:
        """Execute a shell command with typed status and bounded output. Large output (>400 lines or >6,000 tokens)
        is saved to a local evidence store and returns a concise digest with a run address (run:<id>#stdout).
        Use ctx_get or ctx_search to retrieve omitted lines."""
        cmd_str = str(command or "")
        if disable_web_tools:
            if _is_outbound_network_command(cmd_str):
                return json.dumps({
                    "status": "error",
                    "exit_code": 126,
                    "error": (
                        "PermissionDenied: Outbound network and web fetching are disabled in strict/offline "
                        "worker mode (CC_AGY_DISABLE_WEB_TOOLS=true). Solve the task strictly using the local "
                        "repository and local Python environment."
                    ),
                })
        if disable_web_tools or isolated_worktree:
            if _is_host_path_escape(cmd_str):
                return json.dumps({
                    "status": "error",
                    "exit_code": 126,
                    "error": (
                        "PermissionDenied: Accessing host filesystem directories outside the isolated workspace "
                        "is disabled in isolated worker mode. All required task specifications are inside the "
                        "current workspace directory; implement the solution directly."
                    ),
                })
        if working_directory:
            try:
                cand_wd = (ws_resolved / str(working_directory)).resolve()
                if not cand_wd.is_relative_to(ws_resolved) or _matches_abs_blocked(cand_wd):
                    return json.dumps({
                        "status": "error",
                        "exit_code": 126,
                        "error": (
                            "PermissionDenied: working_directory must stay within the isolated workspace."
                        ),
                    })
            except Exception:
                pass
        if nm_linked and _is_linked_node_modules_mutation(cmd_str):
            return json.dumps({
                "status": "error",
                "exit_code": 126,
                "error": (
                    "Host node_modules is already linked into this workspace; run your test or typecheck command "
                    "directly without installing packages."
                ),
            })
        sc_msg = _check_post_acceptance_short_circuit(cmd_str)
        if sc_msg:
            return json.dumps({
                "status": "skipped",
                "exit_code": None,
                "short_circuited": True,
                "message": sc_msg,
            })
        try:
            raw_timeout = float(timeout_seconds) if timeout_seconds is not None else default_cmd_timeout
        except (TypeError, ValueError, OverflowError):
            raw_timeout = default_cmd_timeout
        eff_timeout = min(max(1.0, raw_timeout if raw_timeout > 0 else default_cmd_timeout), session_timeout)
        try:
            _pre_files, pre_deleted, pre_hashes = _snapshot_run_hashes()
        except Exception:
            pre_deleted, pre_hashes = set(), {}
        t0_mono = time.monotonic()
        result = await session.run(
            ["bash", "-c", cmd_str],
            working_directory=str(working_directory or ""),
            timeout_seconds=eff_timeout,
        )
        dur_sec = round(max(0.0, time.monotonic() - t0_mono), 3)
        post_deleted: set[str] = set()
        post_hashes: dict[str, str] = {}
        tail_src = ""
        try:
            reads_dir = ws_resolved / ".ctx-session-reads"
            reads_dir.mkdir(parents=True, exist_ok=True)
            tail_src = str(getattr(result, "stdout", None) or getattr(result, "digest", None) or "")
            existing_files, post_deleted, post_hashes = _snapshot_run_hashes()
            file_hashes = {
                k: v for k, v in post_hashes.items() if pre_hashes.get(k) == v
            }
            run_record: dict[str, Any] = {
                "ts": round(time.time(), 6),
                "command": cmd_str,
                "working_directory": str(working_directory or ""),
                "exit_code": getattr(result, "command_exit_code", None),
                "timed_out": bool(getattr(result, "timed_out", False)),
                "duration_seconds": dur_sec,
                "output_tail": tail_src[-2000:],
                "file_hashes": file_hashes,
            }
            if existing_files is not None:
                run_record["present_files"] = existing_files[:5000]
            else:
                run_record["deleted_files"] = sorted(pre_deleted & post_deleted)
            run_record["hmac"] = sign_worker_run_record(run_record)
            command_runs.append(run_record)
            with open(reads_dir / "worker_runs.jsonl", "a", encoding="utf-8") as rfh:
                rfh.write(json.dumps(run_record) + "\n")
        except Exception:
            pass
        if post_acc_guard_enabled:
            has_changes, post_state_key = _worktree_state_key(post_deleted, post_hashes)
            if not has_changes or post_state_key != _acc_guard_state["state_key"]:
                _acc_guard_state["state_key"] = post_state_key
                _acc_guard_state["passed_cmds"] = set()
                _acc_guard_state["active"] = False
            norm_cmd = " ".join(cmd_str.strip().split())
            matched_acc: list[str] = []
            if norm_cmd in norm_acc_cmds:
                matched_acc.append(norm_cmd)
            else:
                _run_segs = _split_shell_segments(cmd_str)
                if len(_run_segs) == 1:
                    _rk, _rtgts = _classify_post_acc_segment(_run_segs[0])
                    if _rk == "targeted_test" and _rtgts:
                        _rset = set(_rtgts)
                        for _atgts, _ac_list in _acc_target_to_norm_cmds.items():
                            if _rset >= set(_atgts):
                                matched_acc.extend(_ac_list)
            if (
                matched_acc
                and has_changes
                and pre_hashes == post_hashes
                and pre_deleted == post_deleted
                and getattr(result, "command_exit_code", None) == 0
                and not bool(getattr(result, "timed_out", False))
                and str(working_directory or "").strip() in ("", ".")
                and not _no_tests_re.search(tail_src)
            ):
                _acc_guard_state["passed_cmds"].update(matched_acc)
                if _acc_guard_state["passed_cmds"] >= set(norm_acc_cmds):
                    _acc_guard_state["active"] = True
                    return (
                        f"{result.to_json()}\n"
                        "[ACCEPTANCE_PASSED_STOP_NOW: All acceptance commands passed (exit_code=0) on your modified "
                        "files. Do NOT run git diff, git status, or broader test suites. End your turn immediately "
                        "with 'Done.' in <=5 words.]"
                    )
        return _maybe_append_turn_budget_note(result.to_json())

    setattr(run_command, "command_runs", command_runs)

    def _normalize_safe_repo_path(rel_path: str) -> tuple[str, str | None]:
        raw_rel = str(rel_path or "").strip()
        if raw_rel.lower().startswith("ws:"):
            return "", "PermissionDenied: External workspace aliases (ws:) are disabled in isolated worker sessions."
        try:
            cand = (ws_resolved / raw_rel).resolve()
            rel_obj = cand.relative_to(ws_resolved)
        except Exception:
            return "", "PermissionDenied: Path must remain within the isolated workspace."
        if rel_obj.parts:
            bname = rel_obj.name.lower()
            rel_norm = rel_obj.as_posix().lower()
            parts_low = [p.lower() for p in rel_obj.parts]
            if (
                bname in _SECRET_BASENAMES
                or bname.startswith(".env.")
                or bname.endswith((".pem", ".key", ".p12", ".pfx"))
                or rel_norm == ".claude/settings.local.json"
                or rel_norm.endswith("/.claude/settings.local.json")
                or ".cc-agy-data" in parts_low
            ):
                return "", f"PermissionDenied: Reading credential or secret file {raw_rel!r} is blocked."
            return rel_obj.as_posix(), None
        return "", None

    def _normalize_safe_address(address: str) -> tuple[str, str | None]:
        addr = str(address or "").strip()
        if addr.lower().startswith("ws:") or "/ws:" in addr.lower():
            return addr, "PermissionDenied: External workspace aliases (ws:) are disabled in isolated worker sessions."
        if addr.lower().startswith("repo:"):
            rest = addr[5:]
            repo_rel, sep, frag = rest.partition("#")
            norm_rel, err = _normalize_safe_repo_path(repo_rel)
            if err:
                return addr, err
            return f"repo:{norm_rel}{sep}{frag}", None
        return addr, None

    def _check_lead_read_cap() -> str | None:
        if resolved_profile == "read-only" and is_lead_orchestrator and _straitjacket_state["calls"] >= lead_read_limit:
            return json.dumps({
                "error": (
                    f"LeadReadBudgetExceeded: Lead read/search limit ({lead_read_limit} calls) reached. Do NOT read more files "
                    "on the Lead orchestrator. Call delegate_start (with a concise prompt <500 words), delegate_continue, "
                    "get_diff, or delegate_apply immediately; the worker has full repository and test access in its worktree."
                )
            })
        return None

    async def ctx_get(address: str, start_line: int = 1, end_line: int = 300) -> str:
        """Retrieve a bounded, inclusive line range from an evidence address (e.g. 'run:<id>#stdout')."""
        cap_err = _check_lead_read_cap()
        if cap_err:
            return cap_err
        norm_addr, err = _normalize_safe_address(address)
        if err:
            return json.dumps({"error": err, "address": str(address or "")})
        s_line = _coerce_line_number(start_line, 1)
        e_line = max(s_line, _coerce_line_number(end_line, 300))
        try:
            out = await session.get(norm_addr, start_line=s_line, end_line=e_line)
            return _maybe_append_turn_budget_note(out)
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}", "address": str(address or "")})

    async def ctx_search(address: str, pattern: str) -> str:
        """Search stored evidence (e.g. 'run:<id>#stdout') or repository files for a regex pattern."""
        cap_err = _check_lead_read_cap()
        if cap_err:
            return cap_err
        norm_addr, err = _normalize_safe_address(address)
        if err:
            return json.dumps({"error": err, "address": str(address or "")})
        try:
            out = await session.search(norm_addr, str(pattern or ""))
            return _maybe_append_turn_budget_note(out)
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}", "address": str(address or "")})

    if resolved_profile == "command-output":
        return [run_command, ctx_get, ctx_search], ["RUN_COMMAND"]

    _seen_view_spans: dict[tuple[str, str], list[tuple[int, int]]] = {}

    def _resolve_linked_node_modules_path(raw_path: str) -> tuple[Path | None, str]:
        """Allow read-only inspection of linked `host_ws_resolved/.../node_modules` without `..` escape."""
        if not nm_linked or host_ws_resolved is None:
            return None, ""
        clean = str(raw_path or "").strip().replace("\\", "/")
        while clean.startswith("./"):
            clean = clean[2:]
        clean = clean.strip("/")
        if not clean or clean.lower().startswith("ws:") or os.path.isabs(str(raw_path or "").strip()):
            return None, ""
        rel_parts = Path(clean).parts
        if ".." in rel_parts or "node_modules" not in rel_parts:
            return None, ""
        try:
            cand = (ws_resolved / clean).resolve()
            host_rel = cand.relative_to(host_ws_resolved)
        except Exception:
            return None, ""
        if "node_modules" not in host_rel.parts:
            return None, ""
        bname = cand.name.lower()
        parts_low = [p.lower() for p in host_rel.parts]
        if (
            bname in _SECRET_BASENAMES
            or bname.startswith(".env.")
            or bname.endswith((".pem", ".key", ".p12", ".pfx"))
            or ".cc-agy-data" in parts_low
        ):
            return None, ""
        return cand, Path(clean).as_posix()

    async def view_file(
        path: str,
        start_line: int = 1,
        end_line: int = 400,
        fresh: bool = False,
    ) -> str:
        """Read a bounded, 1-indexed file span (up to 400 inline lines by default); omitted evidence remains addressable."""
        cap_err = _check_lead_read_cap()
        if cap_err:
            return cap_err
        s_line = _coerce_line_number(start_line, 1)
        e_line = max(s_line, _coerce_line_number(end_line, 400))
        norm_rel, err = _normalize_safe_repo_path(path)
        if err:
            nm_cand, nm_rel = _resolve_linked_node_modules_path(path)
            if nm_cand is not None:
                if not nm_cand.is_file():
                    return json.dumps({"error": f"FileNotFoundError: {nm_rel}", "path": str(path or "")})
                try:
                    raw_lines = nm_cand.read_text(encoding="utf-8", errors="replace").splitlines()
                    total_l = len(raw_lines)
                    clamped_end = min(total_l, e_line, s_line + sj_max_inline_lines - 1)
                    body_l = [
                        f"L{ln_no}: {raw_lines[ln_no - 1]}"
                        for ln_no in range(s_line, clamped_end + 1)
                        if 1 <= ln_no <= total_l
                    ]
                    header_l = f"[ctx read repo:{nm_rel} L{s_line}:{max(s_line, clamped_end)} of {total_l}]"
                    return _maybe_append_turn_budget_note(
                        header_l + ("\n" + "\n".join(body_l) if body_l else "")
                    )
                except Exception as exc:
                    return json.dumps({"error": f"{type(exc).__name__}: {exc}", "path": str(path or "")})
            return json.dumps({"error": err, "path": str(path or "")})
        file_sha = ""
        try:
            target_f = ws_resolved / norm_rel
            if target_f.is_file():
                file_sha = _hashlib.sha256(target_f.read_bytes()).hexdigest()
        except Exception:
            file_sha = ""
        try:
            min_span = max(0, int(os.environ.get("CC_AGY_MIN_VIEW_LINES", "250")))
        except (TypeError, ValueError):
            min_span = 250
        prev_spans = _seen_view_spans.get((norm_rel, file_sha), []) if file_sha else []
        if not fresh and file_sha:
            for prev_s, prev_e in prev_spans:
                if prev_s <= s_line and e_line <= prev_e:
                    dedup_msg = (
                        f"[ctx read repo:{norm_rel} L{s_line}:{e_line} · already in context from earlier "
                        f"view_file(L{prev_s}:{prev_e}) at identical sha256:{file_sha[:12]}; "
                        f"file is unchanged, so use lines L{s_line}..L{e_line} above and proceed with edit_file or run_command]"
                    )
                    return _maybe_append_turn_budget_note(dedup_msg)
        if min_span > 0 and (e_line - s_line + 1) < min_span:
            floor_line = (
                1
                if fresh
                else max([pe + 1 for ps, pe in prev_spans if ps <= s_line <= pe + 20 and e_line > pe], default=1)
            )
            s_line = max(floor_line, s_line - 20)
            e_line = max(e_line, s_line + min_span - 1)
        if not fresh and file_sha:
            for prev_s, prev_e in prev_spans:
                if prev_s <= s_line and e_line <= prev_e:
                    dedup_msg = (
                        f"[ctx read repo:{norm_rel} L{s_line}:{e_line} · already in context from earlier "
                        f"view_file(L{prev_s}:{prev_e}) at identical sha256:{file_sha[:12]}; "
                        f"file is unchanged, so use lines L{s_line}..L{e_line} above and proceed with edit_file or run_command]"
                    )
                    return _maybe_append_turn_budget_note(dedup_msg)
        try:
            out = await session.get(f"repo:{norm_rel}", start_line=s_line, end_line=e_line, fresh=bool(fresh))
            if file_sha:
                delivered_nums = [int(m.group(1)) for m in _re.finditer(r"^L(\d+):", out, _re.M)]
                if delivered_nums:
                    raw_spans = list(_seen_view_spans.get((norm_rel, file_sha), []))
                    seg_s = delivered_nums[0]
                    seg_e = seg_s
                    for ln in delivered_nums[1:]:
                        if ln == seg_e + 1:
                            seg_e = ln
                        else:
                            raw_spans.append((seg_s, seg_e))
                            seg_s = seg_e = ln
                    raw_spans.append((seg_s, seg_e))
                    raw_spans.sort()
                    merged_spans: list[tuple[int, int]] = []
                    for cur_s, cur_e in raw_spans:
                        if merged_spans and cur_s <= merged_spans[-1][1] + 1:
                            merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], cur_e))
                        else:
                            merged_spans.append((cur_s, cur_e))
                    _seen_view_spans[(norm_rel, file_sha)] = merged_spans
            return _maybe_append_turn_budget_note(out)
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}", "path": str(path or "")})

    async def search_dir(pattern: str, path: str = "") -> str:
        """Search repository files with a bounded hit census."""
        cap_err = _check_lead_read_cap()
        if cap_err:
            return cap_err
        norm_rel, err = _normalize_safe_repo_path(path)
        if err:
            nm_cand, nm_rel = _resolve_linked_node_modules_path(path)
            if nm_cand is not None and nm_cand.exists():
                raw_pat = str(pattern or "")
                try:
                    rx = _re.compile(raw_pat)
                except _re.error:
                    rx = _re.compile(_re.escape(raw_pat))
                hits: list[str] = []
                scanned = 0
                targets = [nm_cand] if nm_cand.is_file() else []
                if nm_cand.is_dir():
                    for sr, sdirs, sfiles in os.walk(str(nm_cand), topdown=True, followlinks=False):
                        sdirs[:] = sorted(d for d in sdirs if not d.startswith("."))
                        for sf in sorted(sfiles):
                            targets.append(Path(sr) / sf)
                            if len(targets) >= 500:
                                break
                        if len(targets) >= 500:
                            break
                for tf in targets:
                    scanned += 1
                    try:
                        if tf.stat().st_size > 524288:
                            continue
                        raw_b = tf.read_bytes()
                        if b"\x00" in raw_b[:2048]:
                            continue
                        rel_hit = (
                            nm_rel
                            if nm_cand.is_file()
                            else f"{nm_rel}/{tf.relative_to(nm_cand).as_posix()}"
                        )
                        for lno, ltxt in enumerate(raw_b.decode("utf-8", errors="replace").splitlines(), 1):
                            if rx.search(ltxt):
                                hits.append(f"{rel_hit}:{lno}: {ltxt[:200]}")
                                if len(hits) >= 80:
                                    break
                    except OSError:
                        continue
                    if len(hits) >= 80:
                        break
                summary = f"[ctx search repo:{nm_rel} ({scanned} files scanned, {len(hits)} hits)]"
                return _maybe_append_turn_budget_note(summary + ("\n" + "\n".join(hits) if hits else ""))
            return json.dumps({"error": err, "path": str(path or "")})
        try:
            out = await session.search(f"repo:{norm_rel}", str(pattern or ""))
            return _maybe_append_turn_budget_note(out)
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}", "path": str(path or "")})

    async def list_dir(path: str = "") -> str:
        """Summarise a directory's file counts, languages and size."""
        cap_err = _check_lead_read_cap()
        if cap_err:
            return cap_err
        norm_rel, err = _normalize_safe_repo_path(path)
        if err:
            nm_cand, nm_rel = _resolve_linked_node_modules_path(path)
            if nm_cand is not None and nm_cand.is_dir():
                try:
                    rows_l: list[str] = []
                    for dent in sorted(os.scandir(nm_cand), key=lambda e: e.name)[:200]:
                        suffix_kind = "/" if dent.is_dir(follow_symlinks=True) else f" ({dent.stat(follow_symlinks=True).st_size}B)"
                        rows_l.append(f"{dent.name}{suffix_kind}")
                    return _maybe_append_turn_budget_note(
                        f"[ctx list repo:{nm_rel} ({len(rows_l)} entries)]\n" + "\n".join(rows_l)
                    )
                except Exception as exc:
                    return json.dumps({"error": f"{type(exc).__name__}: {exc}", "path": str(path or "")})
            return json.dumps({"error": err, "path": str(path or "")})
        try:
            out = await session.retrieve("stats", ref=f"repo:{norm_rel}")
            return _maybe_append_turn_budget_note(out)
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}", "path": str(path or "")})

    async def find_file(name: str) -> str:
        """List files matching a path glob, including empty files."""
        cap_err = _check_lead_read_cap()
        if cap_err:
            return cap_err
        raw_name = str(name or "")
        if ".." in raw_name or raw_name.lower().startswith("ws:"):
            return json.dumps({"error": "PermissionDenied: Invalid file glob.", "name": raw_name})
        try:
            out = await session.query(f"corpus --glob {shlex.quote(raw_name)}")
            return _maybe_append_turn_budget_note(out)
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}", "name": raw_name})

    async def ctx_query(pipeline: str) -> str:
        """Compose observation in one bounded query, e.g. refs Foo | group file | top 3."""
        cap_err = _check_lead_read_cap()
        if cap_err:
            return cap_err
        raw_pipe = str(pipeline or "")
        if "ws:" in raw_pipe.lower() or "../" in raw_pipe:
            return json.dumps({"error": "PermissionDenied: External workspace aliases (ws:) are disabled.", "pipeline": raw_pipe})
        try:
            out = await session.query(raw_pipe)
            return _maybe_append_turn_budget_note(out)
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}", "pipeline": raw_pipe})

    if resolved_profile == "read-only":
        return (
            [ctx_get, ctx_search, view_file, search_dir, list_dir, find_file, ctx_query],
            ["VIEW_FILE", "SEARCH_DIR", "LIST_DIR", "FIND_FILE"],
        )

    return (
        [run_command, ctx_get, ctx_search, view_file, search_dir, list_dir, find_file, ctx_query],
        ["RUN_COMMAND", "VIEW_FILE", "SEARCH_DIR", "LIST_DIR", "FIND_FILE"],
    )


def _build_default_run_command_config(
    types_mod: Any,
    *,
    timeout_seconds: float | None = 300.0,
) -> Any | None:
    """Build `types.RunCommandConfig(enable_sandbox=True, ...)` for fallback built-in `RUN_COMMAND`."""
    rc_cls = getattr(types_mod, "RunCommandConfig", None)
    if rc_cls is None:
        return None
    enable_sb = env_flag("CC_AGY_BUILTIN_SANDBOX") is not False
    rc_kwargs: dict[str, Any] = {"enable_sandbox": enable_sb}
    if timeout_seconds is not None:
        try:
            t_val = max(1.0, float(timeout_seconds))
            rc_kwargs["timeout_seconds"] = t_val
        except (TypeError, ValueError):
            pass
    try:
        return rc_cls(**rc_kwargs)
    except TypeError:
        try:
            return rc_cls(enable_sandbox=enable_sb)
        except Exception:
            return None


def resolve_worker_sdk_limits(
    request: dict[str, Any] | None = None,
    *,
    env: dict[str, str] | None = None,
) -> tuple[int, int]:
    """Return `(compaction_threshold_tokens, tool_output_max_tokens)` for `LocalAgentConfig` (`0` disables)."""
    req = request or {}
    source_env = os.environ if env is None else env
    raw_compaction = req.get("compaction_threshold")
    if raw_compaction is None:
        raw_compaction = source_env.get("CC_AGY_COMPACTION_THRESHOLD", "120000")
    try:
        compaction_threshold = max(0, int(float(str(raw_compaction).strip())))
    except (TypeError, ValueError):
        compaction_threshold = 120000

    raw_tool_max = req.get("tool_output_max_tokens")
    if raw_tool_max is None:
        raw_tool_max = source_env.get("CC_AGY_TOOL_OUTPUT_MAX_TOKENS", "4000")
    try:
        tool_max_tokens = max(0, int(float(str(raw_tool_max).strip())))
    except (TypeError, ValueError):
        tool_max_tokens = 4000

    return compaction_threshold, tool_max_tokens


async def _default_antigravity_backend(request: dict[str, Any]) -> dict[str, Any]:
    """Execute an agentic task inside `workspace_dir` using `agy` CLI or `google.antigravity`."""
    sanitize_placeholder_env()
    if request.get("auth_mode") == "gemini_enterprise":
        from server.agy_cli import run_agy_cli_worker

        def _invoke_cli_preserving_exc() -> tuple[bool, Any]:
            # CPython <3.12 `asyncio.futures._convert_future_exc` replaces any `TimeoutError`
            # raised inside `run_in_executor` / `to_thread` with a bare `TimeoutError()`,
            # stripping `.partial_usage` and `.partial_agy_conversation_*` attributes.
            try:
                return True, run_agy_cli_worker(request)
            except BaseException as exc:
                return False, exc

        ok_cli, cli_payload = await asyncio.to_thread(_invoke_cli_preserving_exc)
        if not ok_cli:
            raise cli_payload
        return cli_payload

    from google import antigravity as ag
    from google.antigravity import policy, types

    req_tl = str(request.get("thinking_level") or "").strip().upper()
    is_zero_thinking = req_tl in ("ZERO", "0", "OFF")
    sdk_tl_name = "MINIMAL" if is_zero_thinking else req_tl
    options = (
        types.GeminiModelOptions(thinking_level=getattr(types.ThinkingLevel, sdk_tl_name))
        if sdk_tl_name
        else None
    )

    eff_project = request.get("project")
    eff_location = request.get("location")
    if request.get("api_key"):
        endpoint = types.GeminiAPIEndpoint(
            api_key=request["api_key"],
            options=options,
        )
    else:
        vertex_base_url = resolve_vertex_base_url(request.get("base_url")) or ""
        if vertex_base_url:
            eff_project = eff_project or "custom-endpoint"
            eff_location = eff_location or "global"
        ep_kwargs: dict[str, Any] = {
            "project": eff_project,
            "location": eff_location,
            "options": options,
        }
        if vertex_base_url:
            gw_tok = resolve_vertex_base_url_token() or "custom-endpoint"
            ep_kwargs["base_url"] = vertex_base_url
            ep_kwargs["api_key"] = gw_tok
            hdrs = {"Authorization": f"Bearer {gw_tok}"}
            if is_zero_thinking:
                hdrs["X-CC-AGY-Thinking-Budget"] = "0"
            ep_kwargs["http_headers"] = hdrs
        elif is_zero_thinking:
            ep_kwargs["http_headers"] = {"X-CC-AGY-Thinking-Budget": "0"}
        endpoint = types.VertexEndpoint(**ep_kwargs)
    # google-antigravity 0.1.x: the model is a ModelTarget(name, types, endpoint).
    resolved_model = str(request.get("model") or "")
    model_cfg = types.ModelTarget(
        name=request["model"],
        types=[types.ModelType.TEXT],
        endpoint=endpoint,
    )

    cap_kwargs: dict[str, Any] = {
        "enable_subagents": bool(request.get("enable_subagents", False)),
    }
    def _coerce_builtin_tools(items: list[Any]) -> list[Any]:
        coerced: list[Any] = []
        for item in items:
            if isinstance(item, str):
                b_enum = getattr(types.BuiltinTools, item.strip().upper(), None)
                coerced.append(b_enum if b_enum is not None else item.strip().lower())
            else:
                coerced.append(item)
        return coerced

    if request.get("enabled_tools") is not None:
        cap_kwargs["enabled_tools"] = _coerce_builtin_tools(list(request["enabled_tools"]))
    if request.get("disabled_tools") is not None:
        cap_kwargs["disabled_tools"] = _coerce_builtin_tools(list(request["disabled_tools"]))

    # Always disable interactive/UI/workflow tools that cannot function in a headless background worker,
    # and disable SEARCH_WEB / READ_URL_CONTENT when CC_AGY_DISABLE_WEB_TOOLS or strict mode is active.
    disable_web_tools = env_flag("CC_AGY_DISABLE_WEB_TOOLS") is True or (
        is_strict_mode_enabled() and env_flag("CC_AGY_DISABLE_WEB_TOOLS") is not False
    )
    headless_disabled_names = ["ASK_QUESTION", "GENERATE_IMAGE", "SCHEDULE", "RUN_WORKFLOW"]
    if disable_web_tools:
        headless_disabled_names.extend(["SEARCH_WEB", "READ_URL_CONTENT"])
    base_disabled = list(cap_kwargs.get("disabled_tools") or [])
    base_enabled = list(cap_kwargs.get("enabled_tools") or []) if "enabled_tools" in cap_kwargs else None
    for b_name in headless_disabled_names:
        b_enum = getattr(types.BuiltinTools, b_name, None)
        if b_enum is not None:
            if base_enabled is not None and not disable_web_tools and (b_enum in base_enabled or b_name in base_enabled):
                continue
            if b_enum not in base_disabled:
                base_disabled.append(b_enum)
            if base_enabled is not None:
                base_enabled = [
                    x for x in base_enabled
                    if x != b_enum and str(x).upper() != b_name
                ]
    if base_disabled:
        cap_kwargs["disabled_tools"] = base_disabled
    if base_enabled is not None:
        cap_kwargs["enabled_tools"] = base_enabled

    compaction_threshold, tool_max_tokens = resolve_worker_sdk_limits(request)
    if tool_max_tokens > 0 and hasattr(types, "ToolOutputTruncationConfig"):
        cap_kwargs["tool_output_truncation_config"] = types.ToolOutputTruncationConfig(max_tokens=tool_max_tokens)
    default_rc_cfg = _build_default_run_command_config(
        types,
        timeout_seconds=float(request.get("timeout_seconds") or 300.0),
    )
    if default_rc_cfg is not None:
        cap_kwargs["run_command_config"] = default_rc_cfg

    capabilities = types.CapabilitiesConfig(**cap_kwargs)

    # Keep the SDK's own session/artifact files outside the repo so they never show up in the diff.
    save_dir = tempfile.mkdtemp(prefix="cc-agy-save-")

    # Hard caps so one job cannot read tens of millions of tokens; when hit, the turn stops with a
    # MAX_*_EXCEEDED stop reason and the job becomes a "checkpoint" the lead can continue or redirect.
    budget_kwargs = {
        k: int(request[k]) for k in ("max_tool_calls", "max_total_tokens") if request.get(k)
    }
    budget_cfg = types.BudgetConfig(**budget_kwargs) if budget_kwargs else None

    progress_path = request.get("progress_path")
    live_state_file = Path(f"{progress_path}.live") if progress_path else None
    if live_state_file is not None:
        try:
            live_state_file.parent.mkdir(parents=True, exist_ok=True)
            live_state_file.write_text(
                json.dumps(
                    {
                        "straitjacket_calls": 0,
                        "save_dir": str(save_dir),
                        "updated_at_epoch": round(time.time(), 3),
                    }
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def report_to_lead(kind: str, message: str) -> str:
        """Send a short signal to the lead engineer.

        Args:
            kind: "milestone" for progress (e.g. tests now 40/42 passing), "question" when a requirement
                is ambiguous, "blocker" when you cannot proceed, or "spec_conflict" when a prompt/checklist
                premise contradicts verified runtime/library behavior. Questions, blockers, and spec_conflict
                signals end your turn immediately.
            message: One or two sentences (with exact verified runtime output for spec_conflict) for the lead.
        """
        return append_progress(progress_path, kind, message)

    custom_tools: list[Any] = []
    if progress_path:
        custom_tools.append(report_to_lead)
    extra_tools = request.get("extra_tools")
    if extra_tools:
        custom_tools.extend(extra_tools)

    containment_mode = "none"
    containment_warning: str | None = None
    sj_command_runs: list[dict[str, Any]] = []
    inferred_profile = resolve_containment_profile(
        enabled_tools=request.get("enabled_tools"),
        disabled_tools=request.get("disabled_tools"),
    )
    use_containment = os.environ.get("CC_AGY_CONTAINMENT", "1").lower() not in ("0", "false", "no")
    if use_containment:
        try:
            is_lead_orch = any(
                getattr(fn, "__name__", "") in ("delegate_start", "delegate_to_gemini")
                for fn in (extra_tools or [])
            )
            sj_tools, replaced_names = build_straitjacket_tools(
                Path(request["workspace_dir"]),
                timeout_seconds=float(request.get("timeout_seconds") or 300.0),
                profile=inferred_profile,
                is_lead_orchestrator=is_lead_orch,
                acceptance_commands=request.get("acceptance"),
                host_workspace_root=request.get("host_workspace_root"),
                nm_linked=bool(request.get("nm_linked")),
                isolated_worktree=bool(request.get("isolated_worktree")),
                live_state_path=live_state_file,
                save_dir=save_dir,
                max_tool_calls=int(request["max_tool_calls"]) if request.get("max_tool_calls") else None,
            )
            for sj_fn in sj_tools:
                if getattr(sj_fn, "__name__", "") == "run_command" and isinstance(getattr(sj_fn, "command_runs", None), list):
                    sj_command_runs = getattr(sj_fn, "command_runs")
            custom_tools.extend(sj_tools)
            disabled_list = list(cap_kwargs.get("disabled_tools") or [])
            enabled_list = list(cap_kwargs.get("enabled_tools") or []) if "enabled_tools" in cap_kwargs else None
            for b_name in replaced_names:
                b_enum = getattr(types.BuiltinTools, b_name, None)
                if b_enum is not None:
                    if b_enum not in disabled_list:
                        disabled_list.append(b_enum)
                    if enabled_list is not None:
                        enabled_list = [
                            x for x in enabled_list
                            if x != b_enum and str(x).upper() != b_name
                        ]
            cap_kwargs["disabled_tools"] = disabled_list
            if enabled_list is not None:
                cap_kwargs["enabled_tools"] = enabled_list
            if (
                tool_max_tokens > 0
                and request.get("tool_output_max_tokens") is None
                and "CC_AGY_TOOL_OUTPUT_MAX_TOKENS" not in os.environ
                and hasattr(types, "ToolOutputTruncationConfig")
            ):
                sj_floor = max(tool_max_tokens, _env_int("CC_AGY_SJ_RESULT_TOKENS", 6000) + 500)
                cap_kwargs["tool_output_truncation_config"] = types.ToolOutputTruncationConfig(max_tokens=sj_floor)
            capabilities = types.CapabilitiesConfig(**cap_kwargs)
            containment_mode = "straitjacket"
        except (ImportError, Exception) as sj_exc:
            containment_warning = (
                f"Straitjacket (ctx-harness) unavailable ({type(sj_exc).__name__}: {sj_exc}); "
                f"running with standard built-in tools."
            )

    # For Vertex AI: Always set vertex=True at top-level of LocalAgentConfig so the
    # Antigravity SDK does not inject the default GeminiAPIEndpoint image model
    # (which crashes when GEMINI_API_KEY is unset in Vertex-only environments).
    # For Google AI Studio direct API key: set vertex=False and pass api_key.
    is_api_key = bool(request.get("api_key"))
    ws_str = str(Path(request["workspace_dir"]).resolve())
    agent_policies: list[Any] = [policy.allow_all()]
    if hasattr(policy, "workspace_only"):
        try:
            ws_policies = policy.workspace_only([ws_str])
            if isinstance(ws_policies, list):
                agent_policies = ws_policies + agent_policies
            else:
                agent_policies = [ws_policies] + agent_policies
        except Exception:
            pass
    cfg_kwargs: dict[str, Any] = dict(
        workspaces=[ws_str],
        vertex=not is_api_key,
        model=model_cfg,
        policies=agent_policies,
        capabilities=capabilities,
        save_dir=save_dir,
        env={"CC_AGY_HARNESS_SAVE_DIR": str(save_dir)},
    )
    if is_api_key:
        cfg_kwargs["api_key"] = request["api_key"]
    else:
        cfg_kwargs["project"] = eff_project
        cfg_kwargs["location"] = eff_location
    if budget_cfg is not None:
        cfg_kwargs["budget_config"] = budget_cfg
    if compaction_threshold > 0 and hasattr(types, "CompactionConfig"):
        cfg_kwargs["compaction_config"] = types.CompactionConfig(token_threshold=compaction_threshold)
    if custom_tools:
        cfg_kwargs["tools"] = custom_tools
    use_short_preamble = os.environ.get("CC_AGY_FULL_SDK_PREAMBLE", "").strip().lower() not in ("1", "true", "yes")
    force_custom = bool(request.get("system_prompt")) or (
        os.environ.get("CC_AGY_CUSTOM_SYSTEM_PROMPT", "").strip().lower() in ("1", "true", "yes")
    )
    if use_short_preamble:
        sys_text = str(
            request.get("system_prompt")
            or build_worker_system_prompt(
                ws_str,
                profile=inferred_profile,
                has_report_to_lead=bool(progress_path),
            )
        )
        if force_custom and hasattr(types, "CustomSystemInstructions"):
            cfg_kwargs["system_instructions"] = types.CustomSystemInstructions(text=sys_text)
        elif hasattr(types, "TemplatedSystemInstructions") and hasattr(types, "SystemInstructionSection"):
            cfg_kwargs["system_instructions"] = types.TemplatedSystemInstructions(
                sections=[types.SystemInstructionSection(title="straitjacket_worker_rules", content=sys_text)]
            )
        elif hasattr(types, "CustomSystemInstructions"):
            cfg_kwargs["system_instructions"] = types.CustomSystemInstructions(text=sys_text)
    cfg = ag.LocalAgentConfig(**cfg_kwargs)

    tool_calls_log: list[dict[str, Any]] = []
    thoughts_log: list[str] = []
    usage: dict[str, Any] = {
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "total_token_count": 0,
        "max_request_prompt_tokens": 0,
        "requests": [],
        "turns": [],
    }
    text_out = ""
    resp = None
    agent_ref = None
    timeout_sec = float(request.get("timeout_seconds") or 300.0)
    stop_reason: str | None = None

    agent = ag.Agent(cfg)
    entered = False
    abnormal_exit = False
    pre_children = _direct_child_pids(os.getpid())
    on_agent_ready = request.get("_on_agent_created")
    if callable(on_agent_ready):
        try:
            on_agent_ready(agent, save_dir, pre_children)
        except TypeError:
            on_agent_ready(agent, save_dir)
        except Exception:
            pass
    try:
        agent_ref = await agent.__aenter__()
        entered = True
        for idx, prompt_turn in enumerate(request["prompts"]):
            if idx == 0 or not hasattr(agent_ref, "continue_session"):
                resp = await maybe_await(agent_ref.chat(prompt_turn))
            else:
                resp = await maybe_await(agent_ref.continue_session(prompt_turn))
            await asyncio.wait_for(
                maybe_await(resp.resolve()),
                timeout=timeout_sec,
            )
            await asyncio.wait_for(
                drain_response_streams(resp, tool_calls_log, thoughts_log),
                timeout=3.0,
            )
            text_fn = getattr(resp, "text", None)
            if callable(text_fn):
                text_out = str((await maybe_await(text_fn())) or "")
            elif isinstance(text_fn, str):
                text_out = text_fn
            # Turn.usage_metadata is accumulated across all model invocations in the turn.
            turn_usage = await resolve_turn_usage(resp)
            if not any(turn_usage.values()):
                turn_usage = extract_usage_from_session(agent_ref, resp)
            for k in (
                "prompt_token_count",
                "cached_content_token_count",
                "candidates_token_count",
                "thoughts_token_count",
                "total_token_count",
            ):
                usage[k] += turn_usage[k]
            usage["turns"].append(dict(turn_usage))

            # Extract per-invocation usage from resp.steps when available so >200k long-context
            # pricing is evaluated on single model calls rather than accumulated multi-step turns.
            steps_stream = getattr(resp, "steps", None)
            if steps_stream is not None:
                try:
                    if hasattr(steps_stream, "__aiter__"):
                        async for st in steps_stream:
                            st_u = extract_usage_metadata(await maybe_await(getattr(st, "usage_metadata", None)))
                            if any(st_u.values()):
                                usage["requests"].append({**st_u, "per_call": True, "model": resolved_model})
                                usage["max_request_prompt_tokens"] = max(
                                    int(usage.get("max_request_prompt_tokens") or 0),
                                    int(st_u.get("prompt_token_count") or 0),
                                )
                    elif hasattr(steps_stream, "__iter__"):
                        for st in steps_stream:
                            st_u = extract_usage_metadata(await maybe_await(getattr(st, "usage_metadata", None)))
                            if any(st_u.values()):
                                usage["requests"].append({**st_u, "per_call": True, "model": resolved_model})
                                usage["max_request_prompt_tokens"] = max(
                                    int(usage.get("max_request_prompt_tokens") or 0),
                                    int(st_u.get("prompt_token_count") or 0),
                                )
                except Exception:
                    pass
            stop_reason = await resolve_stop_reason(resp)
        sqlite_telemetry = extract_sqlite_session_telemetry(save_dir)
        pub_tot = int(usage.get("total_token_count") or 0)
        sql_tot = int(sqlite_telemetry.get("total_token_count") or 0)
        if pub_tot > 0 and sql_tot > max(pub_tot * 4, pub_tot + 500_000):
            sqlite_telemetry = {"requests": [], "tool_calls": sqlite_telemetry.get("tool_calls") or []}
        for k in (
            "prompt_token_count",
            "cached_content_token_count",
            "candidates_token_count",
            "thoughts_token_count",
            "total_token_count",
        ):
            usage[k] = max(
                int(usage.get(k) or 0),
                int(sqlite_telemetry.get(k) or 0),
            )
        usage["total_token_count"] = max(
            int(usage.get("total_token_count") or 0),
            int(usage.get("prompt_token_count") or 0)
            + int(usage.get("candidates_token_count") or 0)
            + int(usage.get("thoughts_token_count") or 0),
        )
        _reconcile_phase_usage_requests(usage, sqlite_telemetry, resolved_model)
    except (Exception, asyncio.CancelledError) as exc:
        abnormal_exit = True
        # Attach partial usage and tool calls before closing the agent so the caller's finally
        # block records everything consumed prior to the failure or timeout without blocking.
        try:
            await asyncio.wait_for(
                drain_response_streams(resp, tool_calls_log, thoughts_log),
                timeout=2.0,
            )
        except (Exception, asyncio.CancelledError):
            pass
        partial_usage = extract_usage_from_session(agent_ref, resp)
        sqlite_telemetry = extract_sqlite_session_telemetry(save_dir)
        base_tot = max(
            int(usage.get("total_token_count") or 0),
            int(partial_usage.get("total_token_count") or 0),
        )
        sql_tot = int(sqlite_telemetry.get("total_token_count") or 0)
        if base_tot > 0 and sql_tot > max(base_tot * 4, base_tot + 500_000):
            sqlite_telemetry = {"requests": [], "tool_calls": sqlite_telemetry.get("tool_calls") or []}
        for k in (
            "prompt_token_count",
            "cached_content_token_count",
            "candidates_token_count",
            "thoughts_token_count",
            "total_token_count",
        ):
            usage[k] = max(
                int(usage.get(k) or 0),
                int(partial_usage.get(k) or 0),
                int(sqlite_telemetry.get(k) or 0),
            )
        usage["total_token_count"] = max(
            int(usage.get("total_token_count") or 0),
            int(usage.get("prompt_token_count") or 0)
            + int(usage.get("candidates_token_count") or 0)
            + int(usage.get("thoughts_token_count") or 0),
        )
        _reconcile_phase_usage_requests(usage, sqlite_telemetry, resolved_model)
        if not tool_calls_log and sqlite_telemetry.get("tool_calls"):
            tool_calls_log.extend(sqlite_telemetry["tool_calls"])
        if isinstance(exc, asyncio.CancelledError):
            timeout_exc = TimeoutError(f"Antigravity worker timed out or was cancelled after {timeout_sec}s")
            setattr(timeout_exc, "partial_usage", usage)
            setattr(timeout_exc, "partial_tool_calls", tool_calls_log)
            setattr(timeout_exc, "partial_thoughts", thoughts_log)
            setattr(timeout_exc, "partial_containment", containment_mode)
            setattr(timeout_exc, "partial_containment_warning", containment_warning)
            setattr(timeout_exc, "partial_command_runs", list(sj_command_runs))
            raise timeout_exc from exc
        setattr(exc, "partial_usage", usage)
        setattr(exc, "partial_tool_calls", tool_calls_log)
        setattr(exc, "partial_thoughts", thoughts_log)
        setattr(exc, "partial_containment", containment_mode)
        setattr(exc, "partial_containment_warning", containment_warning)
        setattr(exc, "partial_command_runs", list(sj_command_runs))
        raise
    finally:
        if abnormal_exit:
            # Kill localharness BEFORE agent.__aexit__ so LocalConnection.disconnect()'s
            # synchronous process.wait(timeout=180) cannot block or leak an active wire session.
            _force_terminate_agent_process(agent, save_dir, pre_child_pids=pre_children)
        if entered:
            try:
                await asyncio.wait_for(agent.__aexit__(None, None, None), timeout=3.0)
            except (Exception, asyncio.CancelledError):
                pass
        _force_terminate_agent_process(agent, save_dir, pre_child_pids=pre_children)
        if live_state_file is not None:
            try:
                live_state_file.unlink(missing_ok=True)
            except OSError:
                pass
        shutil.rmtree(save_dir, ignore_errors=True)
        shutil.rmtree(Path(ws_str) / ".ctx-session-reads", ignore_errors=True)

    return {
        "text": text_out,
        "usage": usage,
        "tool_calls": tool_calls_log,
        "thoughts": thoughts_log,
        "stop_reason": stop_reason,
        "containment": containment_mode,
        "containment_warning": containment_warning,
        "command_runs": list(sj_command_runs),
    }


async def resolve_stop_reason(resp: Any) -> str | None:
    """Return the SDK stop reason name (e.g. MAX_TOOL_CALLS_EXCEEDED) or None for a normal finish."""
    try:
        raw = await maybe_await(getattr(resp, "stop_reason", None))
    except Exception:
        return None
    if raw is None:
        return None
    name = str(getattr(raw, "name", None) or getattr(raw, "value", None) or raw).upper()
    name = name.rsplit(".", 1)[-1]
    if name in ("", "NONE", "UNSPECIFIED", "STOP_REASON_UNSPECIFIED"):
        return None
    return name


def _reconcile_phase_usage_requests(
    usage: dict[str, Any],
    sqlite_telemetry: dict[str, Any],
    resolved_model: str,
) -> None:
    """Ensure `usage['requests']` captures all SQLite per-call entries and any unitemized phase remainder."""
    sqlite_reqs = [
        ({**r, "model": r.get("model") or resolved_model, "per_call": bool(r.get("per_call", True))})
        for r in (sqlite_telemetry.get("requests") or [])
        if isinstance(r, dict)
    ]
    cur_reqs = [
        ({**r, "model": r.get("model") or resolved_model} if isinstance(r, dict) else r)
        for r in (usage.get("requests") or [])
    ]
    sum_cur_p = sum(int(r.get("prompt_token_count") or 0) for r in cur_reqs if isinstance(r, dict))
    sum_sql_p = sum(int(r.get("prompt_token_count") or 0) for r in sqlite_reqs)
    if sqlite_reqs and (len(sqlite_reqs) > len(cur_reqs) or sum_sql_p > sum_cur_p):
        cur_reqs = sqlite_reqs
    if cur_reqs:
        usage["requests"] = cur_reqs
        usage["max_request_prompt_tokens"] = max(
            int(usage.get("max_request_prompt_tokens") or 0),
            int(sqlite_telemetry.get("max_request_prompt_tokens") or 0),
            max((int(r.get("prompt_token_count") or 0) for r in cur_reqs if isinstance(r, dict)), default=0),
        )
        sum_p = sum(int(r.get("prompt_token_count") or 0) for r in cur_reqs if isinstance(r, dict))
        sum_c = sum(int(r.get("cached_content_token_count") or 0) for r in cur_reqs if isinstance(r, dict))
        sum_cand = sum(int(r.get("candidates_token_count") or 0) for r in cur_reqs if isinstance(r, dict))
        sum_th = sum(int(r.get("thoughts_token_count") or 0) for r in cur_reqs if isinstance(r, dict))
        rem_p = max(0, int(usage.get("prompt_token_count") or 0) - sum_p)
        rem_c = min(rem_p, max(0, int(usage.get("cached_content_token_count") or 0) - sum_c))
        rem_cand = max(0, int(usage.get("candidates_token_count") or 0) - sum_cand)
        rem_th = max(0, int(usage.get("thoughts_token_count") or 0) - sum_th)
        if rem_p > 0 or rem_cand > 0 or rem_th > 0:
            usage["requests"].append(
                {
                    "prompt_token_count": rem_p,
                    "cached_content_token_count": rem_c,
                    "candidates_token_count": rem_cand,
                    "thoughts_token_count": rem_th,
                    "total_token_count": rem_p + rem_cand + rem_th,
                    "model": resolved_model,
                }
            )


def _proc_fs_available() -> bool:
    """Return True when a Linux-style `/proc` process table is mounted."""
    return Path("/proc").is_dir()


def _ps_process_table(*, include_args: bool = False) -> dict[int, tuple[int, str, str]]:
    """Return `{pid: (ppid, state_str, args_str)}` via POSIX `ps` (used on macOS/BSD when `/proc` is absent)."""
    table: dict[int, tuple[int, str, str]] = {}
    fmt = "pid=,ppid=,state=,args=" if include_args else "pid=,ppid=,state="
    try:
        res = subprocess.run(
            ["ps", "-Ao", fmt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0 if include_args else 3.0,
            check=False,
        )
        if res.returncode != 0:
            if include_args:
                return _ps_process_table(include_args=False)
            return table
        for raw_ln in (res.stdout or "").splitlines():
            parts = raw_ln.strip().split(None, 3 if include_args else 2)
            if len(parts) < 3:
                continue
            try:
                pid_v = int(parts[0])
                ppid_v = int(parts[1])
            except ValueError:
                continue
            state_v = parts[2]
            args_v = parts[3] if (include_args and len(parts) > 3) else ""
            table[pid_v] = (ppid_v, state_v, args_v)
    except subprocess.TimeoutExpired:
        if include_args:
            return _ps_process_table(include_args=False)
    except Exception:
        pass
    return table


def _ps_pid_matches_save_dir(pid: int, save_token: str, ps_args: str = "") -> bool:
    """Check whether `pid` references `save_token` in its command line or (`ps eww`) environment."""
    if save_token and save_token in ps_args:
        return True
    try:
        res = subprocess.run(
            ["ps", "eww", "-p", str(pid)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1.5,
            check=False,
        )
        if res.returncode == 0 and save_token in (res.stdout or ""):
            return True
    except Exception:
        pass
    return False


def _read_proc_ppid_and_state(pid: int) -> tuple[int | None, str]:
    """Return `(ppid, state_char)` from `/proc/<pid>/stat` (or `ps` on macOS/BSD)."""
    if _proc_fs_available():
        try:
            raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="ignore")
            rparen = raw_stat.rfind(")")
            if rparen < 0:
                return None, ""
            fields = raw_stat[rparen + 2 :].split()
            if len(fields) < 2:
                return None, ""
            return int(fields[1]), fields[0]
        except (OSError, ValueError):
            return None, ""
    try:
        res = subprocess.run(
            ["ps", "-o", "ppid=,state=", "-p", str(pid)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0:
            for ln in (res.stdout or "").splitlines():
                fields = ln.strip().split()
                if len(fields) >= 2:
                    return int(fields[0]), fields[1][0]
    except Exception:
        pass
    return None, ""


def _direct_child_pids(parent_pid: int) -> set[int]:
    """Return direct live child PIDs of `parent_pid` via `/proc` on Linux or `ps`/`pgrep` on macOS."""
    if _proc_fs_available():
        out: set[int] = set()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            ppid, state = _read_proc_ppid_and_state(pid)
            if ppid == parent_pid and state != "Z":
                out.add(pid)
        return out
    out_ps: set[int] = set()
    ps_tab = _ps_process_table()
    for pid, (ppid, state, _args) in ps_tab.items():
        if ppid == parent_pid and not state.upper().startswith("Z"):
            out_ps.add(pid)
    if not out_ps:
        try:
            pg = subprocess.run(
                ["pgrep", "-P", str(parent_pid)],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if pg.returncode == 0:
                for tok in (pg.stdout or "").split():
                    if tok.isdigit() and _pid_is_alive_non_zombie(int(tok)):
                        out_ps.add(int(tok))
        except Exception:
            pass
    return out_ps


def _collect_descendant_pids(root_pids: set[int], *, exclude_pid: int | None = None) -> list[int]:
    """Return all live descendant PIDs rooted at `root_pids` (BFS order, excluding `exclude_pid`)."""
    if not root_pids:
        return []
    children_by_ppid: dict[int, list[int]] = {}
    if _proc_fs_available():
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == exclude_pid:
                continue
            ppid, state = _read_proc_ppid_and_state(pid)
            if ppid is not None and state != "Z":
                children_by_ppid.setdefault(ppid, []).append(pid)
    else:
        for pid, (ppid, state, _args) in _ps_process_table().items():
            if pid == exclude_pid or state.upper().startswith("Z"):
                continue
            children_by_ppid.setdefault(ppid, []).append(pid)
    descendants: list[int] = []
    seen = set(root_pids)
    frontier = list(root_pids)
    while frontier:
        cur = frontier.pop(0)
        for ch_pid in children_by_ppid.get(cur, ()):
            if ch_pid not in seen and ch_pid != exclude_pid:
                seen.add(ch_pid)
                descendants.append(ch_pid)
                frontier.append(ch_pid)
    return descendants


def _pid_is_alive_non_zombie(pid: int) -> bool:
    if _proc_fs_available():
        _, state = _read_proc_ppid_and_state(pid)
        return bool(state and state != "Z")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except OSError:
        return False
    _ppid, state = _read_proc_ppid_and_state(pid)
    if state:
        return not state.upper().startswith("Z")
    return True


def _proc_matches_save_dir(entry: Path, save_token: str) -> bool:
    """Match a `/proc/<pid>` entry by `CC_AGY_HARNESS_SAVE_DIR` in `environ`, `cmdline`, or open `fd` links."""
    for fname in ("environ", "cmdline"):
        try:
            raw_txt = (entry / fname).read_bytes().decode("utf-8", errors="ignore")
            if save_token in raw_txt:
                return True
        except OSError:
            pass
    fd_dir = entry / "fd"
    try:
        if fd_dir.is_dir():
            for fd_entry in fd_dir.iterdir():
                try:
                    target = os.readlink(fd_entry)
                    if target == save_token or target.startswith(save_token + "/"):
                        return True
                except OSError:
                    continue
    except OSError:
        pass
    return False


def _force_terminate_agent_process(
    agent: Any,
    save_dir: str | None = None,
    *,
    pre_child_pids: set[int] | None = None,
) -> int:
    """Deterministically terminate `localharness` and its entire descendant process tree."""
    my_pid = os.getpid()
    popen_by_pid: dict[int, Any] = {}
    root_pids: set[int] = set()
    candidates: list[Any] = []
    def _safe_attr(obj: Any, attr_name: str) -> Any:
        try:
            return getattr(obj, attr_name, None)
        except Exception:
            return None

    if agent is not None:
        for conv_attr in ("_conversation", "conversation"):
            conv = _safe_attr(agent, conv_attr)
            if conv is not None and not callable(conv):
                for conn_attr in ("connection", "_connection"):
                    conn = _safe_attr(conv, conn_attr)
                    if conn is not None and not callable(conn):
                        candidates.append(_safe_attr(conn, "_process"))
        for strat_attr in ("_strategy", "_connection_strategy"):
            strat = _safe_attr(agent, strat_attr)
            if strat is not None:
                for conn_attr in ("_connection", "connection"):
                    conn = _safe_attr(strat, conn_attr)
                    if conn is not None and not callable(conn):
                        candidates.append(_safe_attr(conn, "_process"))
        candidates.append(_safe_attr(agent, "_process"))

    for proc in candidates:
        if proc is None:
            continue
        try:
            stdin = getattr(proc, "stdin", None)
            if stdin is not None:
                stdin.close()
        except Exception:
            pass
        pid = getattr(proc, "pid", None)
        if isinstance(pid, int) and pid != my_pid:
            popen_by_pid[pid] = proc
            if proc.poll() is None or _pid_is_alive_non_zombie(pid):
                root_pids.add(pid)

    has_proc = _proc_fs_available()
    proc_root = Path("/proc")
    save_token = str(save_dir).rstrip("/") if save_dir else ""
    if save_token and has_proc:
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == my_pid or pid in root_pids:
                continue
            if _proc_matches_save_dir(entry, save_token) and _pid_is_alive_non_zombie(pid):
                root_pids.add(pid)
        if pre_child_pids is not None:
            for pid in _direct_child_pids(my_pid) - pre_child_pids:
                if pid == my_pid or pid in root_pids:
                    continue
                if _proc_matches_save_dir(proc_root / str(pid), save_token) and _pid_is_alive_non_zombie(pid):
                    root_pids.add(pid)
    elif not has_proc:
        ps_tab = _ps_process_table(include_args=bool(save_token))
        if save_token:
            for pid, (_ppid, state, args_s) in ps_tab.items():
                if pid == my_pid or pid in root_pids or state.upper().startswith("Z"):
                    continue
                if save_token in args_s and _pid_is_alive_non_zombie(pid):
                    root_pids.add(pid)
        if pre_child_pids is not None:
            new_children = {
                pid
                for pid in (_direct_child_pids(my_pid) - pre_child_pids)
                if pid != my_pid and _pid_is_alive_non_zombie(pid)
            }
            for pid in new_children:
                if pid in root_pids:
                    continue
                args_s = ps_tab.get(pid, (0, "", ""))[2]
                if not save_token or _ps_pid_matches_save_dir(pid, save_token, args_s):
                    root_pids.add(pid)
            # Under macOS SIP where `ps eww` may omit child env vars, fall back to direct children
            # spawned during this worker's lifetime when private SDK handles yielded no PID.
            if not root_pids and new_children:
                root_pids.update(new_children)

    if not root_pids:
        return 0

    descendants = _collect_descendant_pids(root_pids, exclude_pid=my_pid)
    # Terminate leaves first alongside roots so children cannot outlive their parent.
    ordered_pids = list(reversed(descendants)) + [p for p in root_pids if p not in descendants]
    killed_pids: set[int] = set()

    for pid in ordered_pids:
        proc = popen_by_pid.get(pid)
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                killed_pids.add(pid)
            else:
                os.kill(pid, signal.SIGTERM)
                killed_pids.add(pid)
        except OSError:
            pass

    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        for pid, proc in list(popen_by_pid.items()):
            try:
                proc.poll()
            except Exception:
                pass
        if not any(_pid_is_alive_non_zombie(p) for p in ordered_pids):
            break
        time.sleep(0.02)

    for pid in ordered_pids:
        proc = popen_by_pid.get(pid)
        if _pid_is_alive_non_zombie(pid):
            try:
                if proc is not None and proc.poll() is None:
                    proc.kill()
                else:
                    os.kill(pid, signal.SIGKILL)
                killed_pids.add(pid)
            except OSError:
                pass
        if proc is not None:
            try:
                proc.wait(timeout=0.3)
            except Exception:
                pass

    return len(killed_pids)


def _run_coro_with_bounded_shutdown(
    coro: Any,
    *,
    register_cancel: Callable[[Callable[[], None] | None], None] | None = None,
    agent_holder: list[Any] | None = None,
    extra_cancel_hooks: list[Callable[[], None]] | None = None,
) -> Any:
    """Run `coro` in an isolated event loop with thread-safe cancellation and bounded shutdown."""
    loop = asyncio.new_event_loop()
    finished_evt = threading.Event()
    try:
        asyncio.set_event_loop(loop)
        main_task = loop.create_task(coro)

        if callable(register_cancel):
            def _cancel_worker() -> None:
                if finished_evt.is_set():
                    return
                try:
                    loop.call_soon_threadsafe(main_task.cancel)
                except RuntimeError:
                    pass
                for extra_fn in list(extra_cancel_hooks or []):
                    try:
                        extra_fn()
                    except Exception:
                        pass
                if agent_holder:
                    def _watchdog() -> None:
                        if not finished_evt.wait(timeout=1.2) and agent_holder:
                            ag_obj = agent_holder[0] if len(agent_holder) > 0 else None
                            sv_dir = agent_holder[1] if len(agent_holder) > 1 else None
                            pre_ch = agent_holder[2] if len(agent_holder) > 2 else None
                            _force_terminate_agent_process(ag_obj, sv_dir, pre_child_pids=pre_ch)

                    threading.Thread(target=_watchdog, daemon=True).start()

            register_cancel(_cancel_worker)

        return loop.run_until_complete(main_task)
    finally:
        finished_evt.set()
        if callable(register_cancel):
            try:
                register_cancel(None)
            except Exception:
                pass
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.wait(pending, timeout=2.0))
            loop.run_until_complete(loop.shutdown_asyncgens())
            try:
                loop.run_until_complete(loop.shutdown_default_executor(timeout=2.0))
            except TypeError:
                pass
        except Exception:
            pass
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def run_antigravity_worker(
    *,
    prompts: list[str],
    workspace_dir: str | Path,
    model: str,
    project: str | None = None,
    location: str | None = None,
    api_key: str | None = None,
    auth_mode: str | None = None,
    timeout_seconds: float = 300.0,
    enabled_tools: list[str] | None = None,
    disabled_tools: list[str] | None = None,
    enable_subagents: bool = False,
    backend: Callable[[dict[str, Any]], Any] | None = None,
    thinking_level: str | None = None,
    max_tool_calls: int | None = None,
    max_total_tokens: int | None = None,
    progress_path: str | Path | None = None,
    extra_tools: list[Any] | None = None,
    system_prompt: str | None = None,
    acceptance: list[str] | None = None,
    allowed_paths: list[str] | None = None,
    agy_conversation_id: str | None = None,
    agy_prior_conversation_usage: Mapping[str, Any] | None = None,
    agy_prior_conversation_turns: int | None = None,
    register_cancel: Callable[[Callable[[], None] | None], None] | None = None,
    host_workspace_root: str | Path | None = None,
    nm_linked: bool = False,
    isolated_worktree: bool = False,
) -> dict[str, Any]:
    """Run an Antigravity worker synchronously or via custom test backend."""
    resolved_auth = resolve_auth_config(
        requested_project=project,
        requested_location=location,
        requested_api_key=api_key,
        requested_auth_mode=auth_mode,
        model=model,
    )
    eff_auth_mode = resolved_auth["auth_mode"]
    resolved_key = resolved_auth.get("api_key")
    resolved_project = resolved_auth.get("project")
    resolved_location = resolved_auth.get("location")
    reroute_note = resolved_auth.get("reroute_note")
    resolved_thinking = resolve_thinking_level(
        thinking_level,
        model=model,
        read_only=is_read_only_tool_set(enabled_tools, disabled_tools),
    )

    agent_holder: list[Any] = [None, None, None]

    def _on_agent_created(ag_obj: Any, sv_dir: str | None, pre_ch: set[int] | None = None) -> None:
        agent_holder[0] = ag_obj
        agent_holder[1] = sv_dir
        agent_holder[2] = pre_ch

    req: dict[str, Any] = {
        "prompts": list(prompts),
        "workspace_dir": str(Path(workspace_dir).resolve()),
        "model": model,
        "auth_mode": eff_auth_mode,
        "api_key": resolved_key,
        "project": resolved_project,
        "location": resolved_location,
        "location_explicit": bool(resolved_auth.get("location_explicit")),
        "reroute_note": reroute_note,
        "timeout_seconds": timeout_seconds,
        "enabled_tools": enabled_tools,
        "disabled_tools": disabled_tools,
        "enable_subagents": enable_subagents,
        "thinking_level": resolved_thinking,
        "max_tool_calls": max_tool_calls,
        "max_total_tokens": max_total_tokens,
        "progress_path": str(progress_path) if progress_path else None,
        "extra_tools": list(extra_tools) if extra_tools else None,
        "system_prompt": system_prompt,
        "acceptance": list(acceptance) if acceptance else [],
        "allowed_paths": list(allowed_paths) if allowed_paths else [],
        "agy_conversation_id": agy_conversation_id,
        "host_workspace_root": str(Path(host_workspace_root).resolve()) if host_workspace_root else None,
        "nm_linked": bool(nm_linked),
        "isolated_worktree": bool(isolated_worktree),
        **(
            {"agy_prior_conversation_usage": dict(agy_prior_conversation_usage)}
            if isinstance(agy_prior_conversation_usage, Mapping)
            else {}
        ),
        **(
            {"agy_prior_conversation_turns": int(agy_prior_conversation_turns)}
            if agy_prior_conversation_turns is not None
            else {}
        ),
    }
    extra_cancel_hooks: list[Callable[[], None]] = []
    in_async_runner = [False]

    def _register_backend_cancel(fn: Callable[[], None] | None) -> None:
        if fn is None:
            extra_cancel_hooks.clear()
            if not in_async_runner[0] and callable(register_cancel):
                register_cancel(None)
            return
        extra_cancel_hooks.append(fn)
        if not in_async_runner[0] and callable(register_cancel):
            register_cancel(fn)

    if backend is None:
        req["_on_agent_created"] = _on_agent_created
    if callable(register_cancel):
        req["register_cancel"] = _register_backend_cancel

    active_backend = backend or _default_antigravity_backend
    started = time.time()
    result = active_backend(req)
    if inspect.isawaitable(result):
        in_async_runner[0] = True
        result = _run_coro_with_bounded_shutdown(
            result,
            register_cancel=register_cancel,
            agent_holder=agent_holder,
            extra_cancel_hooks=extra_cancel_hooks,
        )
    elif callable(register_cancel):
        try:
            register_cancel(None)
        except Exception:
            pass
    elapsed = round(max(0.0, time.time() - started), 3)

    containment_val = result.get("containment", "none")
    containment_warn = result.get("containment_warning")
    use_containment = os.environ.get("CC_AGY_CONTAINMENT", "1").lower() not in ("0", "false", "no")
    if use_containment and containment_val == "none" and not containment_warn and not is_straitjacket_available():
        containment_warn = (
            "Straitjacket (ctx-harness) is not installed; worker executed without ctx shell containment."
        )

    if result.get("vertex_project"):
        resolved_project = str(result["vertex_project"])
    if result.get("vertex_location"):
        resolved_location = str(result["vertex_location"])
    if result.get("reroute_note") and not reroute_note:
        reroute_note = str(result["reroute_note"])

    return {
        "text": str(result.get("text") or ""),
        "usage": dict(result.get("usage") or {}),
        "tool_calls": list(result.get("tool_calls") or []),
        "thoughts": list(result.get("thoughts") or []),
        "stop_reason": result.get("stop_reason"),
        "containment": containment_val,
        "containment_warning": containment_warn,
        "command_runs": list(result.get("command_runs") or []),
        "auth_mode": eff_auth_mode,
        "vertex_project": resolved_project,
        "vertex_location": resolved_location,
        "reroute_note": reroute_note,
        "thinking_level": resolved_thinking,
        "duration_seconds": elapsed,
        **({"agy_conversation_id": str(result["agy_conversation_id"])} if result.get("agy_conversation_id") else {}),
        **(
            {"agy_conversation_usage": dict(result["agy_conversation_usage"])}
            if isinstance(result.get("agy_conversation_usage"), Mapping)
            else {}
        ),
        **(
            {"agy_conversation_turns": int(result["agy_conversation_turns"])}
            if result.get("agy_conversation_turns")
            else {}
        ),
        **({"usage_is_cumulative": True} if result.get("usage_is_cumulative") else {}),
    }
