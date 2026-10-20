"""Antigravity SDK worker adapter with crash-safe usage/tool extraction."""

from __future__ import annotations

import asyncio
import glob
import inspect
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
import time
from typing import Any, Callable

from server.regions import (
    resolve_auth_config,
    resolve_vertex_location,
    resolve_vertex_project,
    sanitize_placeholder_env,
)


async def maybe_await(value: Any) -> Any:
    """Resolve value whether it is a plain return value, coroutine, or awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value


THINKING_LEVELS = ("MINIMAL", "LOW", "MEDIUM", "HIGH", "EXTRA_HIGH")
PROGRESS_KINDS = ("milestone", "question", "blocker")
_progress_lock = threading.Lock()


def append_progress(progress_path: str | Path | None, kind: str, message: str) -> str:
    """Append one worker -> lead signal (milestone / question / blocker) as a JSON line."""
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
    "You are a headless software engineering worker inside an isolated repository copy with "
    "Straitjacket (ctx-harness) tools. Implement or investigate the requested task using "
    "the cached repository map, candidate files, and Straitjacket tools (run_command, ctx_get, ctx_search, "
    "view_file, search_dir, list_dir, find_file, ctx_query). When output is bounded to an "
    "evidence address (e.g. run:<id>#stdout or repo:<path>), inspect spans with ctx_get/ctx_search "
    "instead of rerunning commands; keep edits within allowed_paths and finish with a summary under 250 words."
)


def build_worker_system_prompt(
    ws_str: str,
    *,
    profile: str | None = None,
    has_report_to_lead: bool = False,
) -> str:
    """Build a workspace-aware system instruction block for the active Straitjacket tool profile."""
    eff_profile = (profile or os.environ.get("CC_AGY_CONTAINMENT_PROFILE") or "command-output").strip().lower()
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
        f"- Workspace root (use exact absolute path for file tools): {ws_str}\n"
        f"- Active tools ({eff_profile}): {tool_list}.\n"
        "- Read & edit early: batch view_file calls in parallel (1..500 lines, no 50-100 slices or re-reads); edit within <=8 calls.\n"
        "- Bounded thinking: never draft full code in thinking (<1,500 tokens/turn); emit create_file/edit_file directly.\n"
        "- Direct verification: skip `python3 -c` / `node -e` probes; edit files, run Acceptance tests, and stop once green.\n"
        "- Non-interactive: never ask the user; call report_to_lead(kind='question'|'blocker', ...) only if blocked.\n"
        "- Credential & git hygiene: never read/edit .env*, .mcp.json, .netrc, *.pem, *.key, or .claude/settings*.json; never run git commit/reset/checkout/clean."
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
    forced_env = (os.environ.get("CC_AGY_FORCE_WORKER_THINKING_LEVEL") or "").strip().upper()
    if not read_only and forced_env in THINKING_LEVELS:
        return forced_env
    default_level = "HIGH" if read_only else "LOW"
    raw = (value or raw_env or default_level).strip().upper()
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
    if raw is None:
        return usage
    data = raw.model_dump() if hasattr(raw, "model_dump") else (dict(raw) if isinstance(raw, dict) else {})
    if not data:
        data = {k: getattr(raw, k, 0) for k in usage}
    for k in usage:
        usage[k] = int(data.get(k) or 0)
    if usage["total_token_count"] == 0:
        usage["total_token_count"] = (
            usage["prompt_token_count"] + usage["candidates_token_count"] + usage["thoughts_token_count"]
        )
    return usage


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
    for db_path in sorted(glob.glob(os.path.join(str(save_dir), "*.db"))):
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT idx, step_type, metadata, step_payload FROM steps ORDER BY idx"
            ).fetchall()
            conn.close()
        except Exception:
            continue
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
})
_EXTRA_CTX_DENY_GLOBS = (
    ".netrc", "**/.netrc",
    ".npmrc", "**/.npmrc",
    ".pypirc", "**/.pypirc",
    ".mcp.json", "**/.mcp.json",
    ".hmac_key", "**/.hmac_key",
    "id_ecdsa*", "**/id_ecdsa*",
    "credentials.json", "**/credentials.json",
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
    extra_blocked_paths = tuple(
        p.strip()
        for p in os.environ.get("CC_AGY_BLOCKED_PATHS", "").split(",")
        if p.strip()
    )
    scrubbed_env: dict[str, str] = {
        "CC_AGY_DATA_DIR": "",
        "CC_AGY_OUT_DIR": "",
        "CC_AGY_WORKTREES_DIR": "",
        "CLAUDE_PLUGIN_DATA": "",
        "GEMINI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "CLAUDE_PLUGIN_OPTION_GEMINI_API_KEY": "",
        "GOOGLE_APPLICATION_CREDENTIALS": "",
    }
    for env_k in list(os.environ.keys()):
        if env_k.startswith(("CC_AGY_", "CLAUDE_PLUGIN_")) or env_k in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ):
            scrubbed_env[env_k] = ""
    ws_resolved = Path(workspace_dir).resolve()
    py_parts = [str(ws_resolved)]
    if (ws_resolved / "src").is_dir():
        py_parts.append(str(ws_resolved / "src"))
    existing_py = (os.environ.get("PYTHONPATH") or "").strip()
    if existing_py:
        py_parts.append(existing_py)
    scrubbed_env["PYTHONPATH"] = ":".join(py_parts)
    venv_bin = ws_resolved / ".venv" / "bin"
    if venv_bin.is_dir():
        existing_path = (os.environ.get("PATH") or "").strip()
        scrubbed_env["PATH"] = f"{venv_bin}:{existing_path}" if existing_path else str(venv_bin)
        scrubbed_env["VIRTUAL_ENV"] = str(ws_resolved / ".venv")
    from dataclasses import replace as _dc_replace

    class _ExpandedToolSession(ToolSession):
        def _workspace(self):
            ws = super()._workspace()
            b = ws.config.budgets
            return _dc_replace(
                ws,
                config=_dc_replace(
                    ws.config,
                    budgets=_dc_replace(
                        b,
                        result_tokens=max(b.result_tokens, 6400),
                        turn_retrieval_tokens=max(b.turn_retrieval_tokens, 16000),
                        max_inline_lines=max(b.max_inline_lines, 750),
                        max_inline_bytes=max(b.max_inline_bytes, 40960),
                    ),
                ),
            )

    session = _ExpandedToolSession(
        ws_resolved,
        timeout=session_timeout,
        env=scrubbed_env,
        producer="cc-agy-plugin",
    )

    disable_web_tools = (
        os.environ.get("CC_AGY_DISABLE_WEB_TOOLS", "").strip().lower() in ("1", "true", "yes")
        or (
            os.environ.get("CC_AGY_STRICT_MODE", "").strip().lower() in ("1", "true", "yes")
            and os.environ.get("CC_AGY_DISABLE_WEB_TOOLS", "").strip().lower() not in ("0", "false", "no")
        )
    )
    import re as _re
    _direct_network_tokens = (
        "curl ",
        "wget ",
        "urllib.request",
        "urllib.urlopen",
        "socket.create_connection",
        "git clone ",
        "git fetch ",
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
        "/google/",
        "find / ",
    ) + extra_blocked_paths

    def _is_outbound_network_command(raw_cmd: str) -> bool:
        if any(tok in raw_cmd for tok in _direct_network_tokens):
            return True
        if ("MockTransport" in raw_cmd or "ASGITransport" in raw_cmd or "WSGITransport" in raw_cmd):
            return False
        if any(lib in raw_cmd for lib in ("requests.get(", "requests.post(", "httpx.get(", "httpx.post(", "http.client")):
            if _external_url_re.search(raw_cmd):
                return True
        return False

    def _is_host_path_escape(raw_cmd: str) -> bool:
        scrubbed = raw_cmd.replace(str(ws_resolved), "__WS__")
        if any(tok in scrubbed for tok in _plugin_internal_escape_tokens):
            return True
        return False

    import subprocess as _sp
    _session_started_mono = time.monotonic()
    _straitjacket_state = {"calls": 0, "last_git_check": 0.0, "has_edits": False}

    def _maybe_append_turn_budget_note(raw_output: str) -> str:
        if resolved_profile == "read-only":
            if not is_lead_orchestrator:
                return raw_output
            _straitjacket_state["calls"] += 1
            calls = _straitjacket_state["calls"]
            if calls >= 3:
                return (
                    f"{raw_output}\n"
                    f"[Lead budget: {calls}/{lead_read_limit} read/search calls used — once you know the target files, "
                    "call delegate_start with a concise prompt (<500 words, no 'Wait:' self-debate); "
                    "the worker has full repository and test access.]"
                )
            return raw_output
        _straitjacket_state["calls"] += 1
        calls = _straitjacket_state["calls"]
        elapsed = max(0.0, time.monotonic() - _session_started_mono)
        if calls < 8 and elapsed < session_timeout * 0.35:
            return raw_output
        now_m = time.monotonic()
        if not _straitjacket_state["has_edits"] and (now_m - _straitjacket_state["last_git_check"] >= 4.0):
            _straitjacket_state["last_git_check"] = now_m
            try:
                gs = _sp.run(
                    ["git", "status", "--porcelain", "--untracked-files=normal"],
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
        if not _straitjacket_state["has_edits"] and (calls >= 8 or elapsed >= session_timeout * 0.35):
            return (
                f"{raw_output}\n"
                f"[Turn budget: {int(elapsed)}s/{int(session_timeout)}s elapsed, {calls} Straitjacket calls, "
                "0 files modified in worktree — stop searching/reading and write your implementation with "
                "edit_file / create_file NOW so your edits are saved on disk before timeout.]"
            )
        if _straitjacket_state["has_edits"] and elapsed >= session_timeout * 0.75:
            return (
                f"{raw_output}\n"
                f"[Turn budget: {int(elapsed)}s/{int(session_timeout)}s elapsed — finish your current edit, "
                "run the targeted acceptance test, and conclude your turn before timeout.]"
            )
        return raw_output

    async def run_command(
        command: str,
        working_directory: str = "",
        timeout_seconds: float = default_cmd_timeout,
    ) -> str:
        """Execute a shell command with typed status and bounded output. Large output (>240 lines or >4,000 tokens)
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
            if _is_host_path_escape(cmd_str):
                return json.dumps({
                    "status": "error",
                    "exit_code": 126,
                    "error": (
                        "PermissionDenied: Accessing host filesystem directories outside the isolated workspace "
                        "is disabled in strict worker mode. All required task specifications are inside the "
                        "current workspace directory; implement the solution directly."
                    ),
                })
            if working_directory:
                try:
                    cand_wd = (ws_resolved / str(working_directory)).resolve()
                    if not cand_wd.is_relative_to(ws_resolved):
                        return json.dumps({
                            "status": "error",
                            "exit_code": 126,
                            "error": (
                                "PermissionDenied: working_directory must stay within the isolated workspace."
                            ),
                        })
                except Exception:
                    pass
        try:
            raw_timeout = float(timeout_seconds) if timeout_seconds is not None else default_cmd_timeout
        except (TypeError, ValueError, OverflowError):
            raw_timeout = default_cmd_timeout
        eff_timeout = min(max(1.0, raw_timeout if raw_timeout > 0 else default_cmd_timeout), session_timeout)
        result = await session.run(
            ["bash", "-c", cmd_str],
            working_directory=str(working_directory or ""),
            timeout_seconds=eff_timeout,
        )
        return _maybe_append_turn_budget_note(result.to_json())

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
                    "get_diff, or delegate_apply immediately — the worker has full repository and test access in its worktree."
                )
            })
        return None

    async def ctx_get(address: str, start_line: int = 1, end_line: int = 100) -> str:
        """Retrieve a bounded, inclusive line range from an evidence address (e.g. 'run:<id>#stdout')."""
        cap_err = _check_lead_read_cap()
        if cap_err:
            return cap_err
        norm_addr, err = _normalize_safe_address(address)
        if err:
            return json.dumps({"error": err, "address": str(address or "")})
        s_line = _coerce_line_number(start_line, 1)
        e_line = max(s_line, _coerce_line_number(end_line, 100))
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

    import shlex

    async def view_file(path: str, start_line: int = 1, end_line: int = 500) -> str:
        """Read a bounded, 1-indexed file span (minimum 250 lines by default); omitted evidence remains addressable."""
        cap_err = _check_lead_read_cap()
        if cap_err:
            return cap_err
        norm_rel, err = _normalize_safe_repo_path(path)
        if err:
            return json.dumps({"error": err, "path": str(path or "")})
        s_line = _coerce_line_number(start_line, 1)
        e_line = max(s_line, _coerce_line_number(end_line, 500))
        try:
            min_span = max(0, int(os.environ.get("CC_AGY_MIN_VIEW_LINES", "250")))
        except (TypeError, ValueError):
            min_span = 250
        if min_span > 0 and (e_line - s_line + 1) < min_span:
            s_line = max(1, s_line - 20)
            e_line = max(e_line, s_line + min_span - 1)
        try:
            out = await session.get(f"repo:{norm_rel}", start_line=s_line, end_line=e_line)
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


async def _default_antigravity_backend(request: dict[str, Any]) -> dict[str, Any]:
    """Execute an agentic task inside `workspace_dir` using `google.antigravity`."""
    from google import antigravity as ag
    from google.antigravity import policy, types

    sanitize_placeholder_env()
    options = (
        types.GeminiModelOptions(thinking_level=getattr(types.ThinkingLevel, request["thinking_level"]))
        if request.get("thinking_level")
        else None
    )

    if request.get("api_key"):
        endpoint = types.GeminiAPIEndpoint(
            api_key=request["api_key"],
            options=options,
        )
    else:
        vertex_base_url = (
            request.get("base_url")
            or os.environ.get("CC_AGY_VERTEX_BASE_URL")
            or os.environ.get("GOOGLE_VERTEX_BASE_URL")
            or ""
        ).strip()
        ep_kwargs: dict[str, Any] = {
            "project": request["project"],
            "location": request["location"],
            "options": options,
        }
        if vertex_base_url:
            ep_kwargs["base_url"] = vertex_base_url
            ep_kwargs["api_key"] = os.environ.get("CC_AGY_VERTEX_BASE_URL_TOKEN", "custom-endpoint")
        endpoint = types.VertexEndpoint(**ep_kwargs)
    # google-antigravity 0.1.x: the model is a ModelTarget(name, types, endpoint).
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
    # and disable SEARCH_WEB / READ_URL_CONTENT when CC_AGY_DISABLE_WEB_TOOLS or CC_AGY_STRICT_MODE is active.
    disable_web_tools = (
        os.environ.get("CC_AGY_DISABLE_WEB_TOOLS", "").strip().lower() in ("1", "true", "yes")
        or (
            os.environ.get("CC_AGY_STRICT_MODE", "").strip().lower() in ("1", "true", "yes")
            and os.environ.get("CC_AGY_DISABLE_WEB_TOOLS", "").strip().lower() not in ("0", "false", "no")
        )
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

    def report_to_lead(kind: str, message: str) -> str:
        """Send a short signal to the lead engineer.

        Args:
            kind: "milestone" for progress (e.g. tests now 40/42 passing), "question" when a requirement
                is ambiguous, or "blocker" when you cannot proceed. Questions and blockers end your turn.
            message: One or two sentences for the lead.
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
    inferred_profile = (
        "read-only"
        if (
            not os.environ.get("CC_AGY_CONTAINMENT_PROFILE", "").strip()
            and is_read_only_tool_set(request.get("enabled_tools"), request.get("disabled_tools"))
        )
        else None
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
            )
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
    )
    if is_api_key:
        cfg_kwargs["api_key"] = request["api_key"]
    else:
        cfg_kwargs["project"] = request.get("project")
        cfg_kwargs["location"] = request.get("location")
    if budget_cfg is not None:
        cfg_kwargs["budget_config"] = budget_cfg
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
                                usage["requests"].append({**st_u, "per_call": True})
                                usage["max_request_prompt_tokens"] = max(
                                    int(usage.get("max_request_prompt_tokens") or 0),
                                    int(st_u.get("prompt_token_count") or 0),
                                )
                    elif hasattr(steps_stream, "__iter__"):
                        for st in steps_stream:
                            st_u = extract_usage_metadata(await maybe_await(getattr(st, "usage_metadata", None)))
                            if any(st_u.values()):
                                usage["requests"].append({**st_u, "per_call": True})
                                usage["max_request_prompt_tokens"] = max(
                                    int(usage.get("max_request_prompt_tokens") or 0),
                                    int(st_u.get("prompt_token_count") or 0),
                                )
                except Exception:
                    pass
            stop_reason = await resolve_stop_reason(resp)
        sqlite_telemetry = extract_sqlite_session_telemetry(save_dir)
        if not usage.get("requests") and sqlite_telemetry.get("requests"):
            usage["requests"] = list(sqlite_telemetry["requests"])
            usage["max_request_prompt_tokens"] = max(
                int(usage.get("max_request_prompt_tokens") or 0),
                int(sqlite_telemetry.get("max_request_prompt_tokens") or 0),
            )
    except (Exception, asyncio.CancelledError) as exc:
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
        if not usage.get("requests") and sqlite_telemetry.get("requests"):
            usage["requests"] = list(sqlite_telemetry["requests"])
            usage["max_request_prompt_tokens"] = max(
                int(usage.get("max_request_prompt_tokens") or 0),
                int(sqlite_telemetry.get("max_request_prompt_tokens") or 0),
            )
        if not tool_calls_log and sqlite_telemetry.get("tool_calls"):
            tool_calls_log.extend(sqlite_telemetry["tool_calls"])
        if isinstance(exc, asyncio.CancelledError):
            timeout_exc = TimeoutError(f"Antigravity worker timed out or was cancelled after {timeout_sec}s")
            setattr(timeout_exc, "partial_usage", usage)
            setattr(timeout_exc, "partial_tool_calls", tool_calls_log)
            setattr(timeout_exc, "partial_thoughts", thoughts_log)
            setattr(timeout_exc, "partial_containment", containment_mode)
            setattr(timeout_exc, "partial_containment_warning", containment_warning)
            raise timeout_exc from exc
        setattr(exc, "partial_usage", usage)
        setattr(exc, "partial_tool_calls", tool_calls_log)
        setattr(exc, "partial_thoughts", thoughts_log)
        setattr(exc, "partial_containment", containment_mode)
        setattr(exc, "partial_containment_warning", containment_warning)
        raise
    finally:
        if entered:
            try:
                await asyncio.wait_for(agent.__aexit__(None, None, None), timeout=5.0)
            except (Exception, asyncio.CancelledError):
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


def _run_coro_with_bounded_shutdown(coro: Any) -> Any:
    """Run `coro` in an isolated event loop with bounded executor/task shutdown on timeout."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
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
) -> dict[str, Any]:
    """Run an Antigravity worker synchronously or via custom test backend."""
    auth = resolve_auth_config(
        requested_project=project,
        requested_location=location,
        requested_api_key=api_key,
        model=model,
    )
    auth_mode = auth["auth_mode"]
    resolved_key = auth.get("api_key")
    resolved_project = auth.get("project")
    resolved_location = auth.get("location")
    reroute_note = auth.get("reroute_note")
    resolved_thinking = resolve_thinking_level(
        thinking_level,
        model=model,
        read_only=is_read_only_tool_set(enabled_tools, disabled_tools),
    )

    req = {
        "prompts": list(prompts),
        "workspace_dir": str(Path(workspace_dir).resolve()),
        "model": model,
        "auth_mode": auth_mode,
        "api_key": resolved_key,
        "project": resolved_project,
        "location": resolved_location,
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
    }

    active_backend = backend or _default_antigravity_backend
    started = time.time()
    result = active_backend(req)
    if inspect.isawaitable(result):
        result = _run_coro_with_bounded_shutdown(result)
    elapsed = round(max(0.0, time.time() - started), 3)

    containment_val = result.get("containment", "none")
    containment_warn = result.get("containment_warning")
    use_containment = os.environ.get("CC_AGY_CONTAINMENT", "1").lower() not in ("0", "false", "no")
    if use_containment and containment_val == "none" and not containment_warn and not is_straitjacket_available():
        containment_warn = (
            "Straitjacket (ctx-harness) is not installed; worker executed without ctx shell containment."
        )

    return {
        "text": str(result.get("text") or ""),
        "usage": dict(result.get("usage") or {}),
        "tool_calls": list(result.get("tool_calls") or []),
        "thoughts": list(result.get("thoughts") or []),
        "stop_reason": result.get("stop_reason"),
        "containment": containment_val,
        "containment_warning": containment_warn,
        "auth_mode": auth_mode,
        "vertex_project": resolved_project,
        "vertex_location": resolved_location,
        "reroute_note": reroute_note,
        "thinking_level": resolved_thinking,
        "duration_seconds": elapsed,
    }
