"""Agent Client Protocol (ACP v1) JSON-RPC 2.0 stdio client for `agy acp`.

Provides structured session lifecycle (`initialize`, `session/new`, `session/load`,
`session/prompt`, `session/cancel`), real-time `session/update` tool/thought/text
streaming, workspace-scoped permission enforcement (`session/request_permission`),
and `agy acp --sandbox` integration when `CC_AGY_CLI_PROTOCOL=acp`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import queue
import secrets
import subprocess
import threading
import time
from typing import Any

ACP_PROTOCOL_VERSION = 1
CLI_PROTOCOLS = ("stream-json", "acp")


def resolve_cli_protocol(env: Mapping[str, str] | None = None) -> str:
    """Resolve `CC_AGY_CLI_PROTOCOL` (`stream-json` default, or experimental `acp`)."""
    source = env if env is not None else os.environ
    raw = str(source.get("CC_AGY_CLI_PROTOCOL") or "").strip().lower()
    if raw in ("acp", "agent-client-protocol"):
        return "acp"
    return "stream-json"


def probe_agy_acp_capabilities(
    agy_bin: str,
    *,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Perform an offline 0-token `agy acp --stdio` `initialize` handshake."""
    from server.agy_cli import _custom_config_dir_args, build_agy_subprocess_env

    cmd = [agy_bin, "acp", "--stdio", *_custom_config_dir_args(env)]
    sub_env = build_agy_subprocess_env(env)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=(os.name != "nt"),
            env=sub_env,
        )
    except OSError as exc:
        return {"supported": False, "error": f"{type(exc).__name__}: {exc}"}

    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "clientCapabilities": {},
            "clientInfo": {"name": "cc-agy-plugin", "version": "0.3.0"},
        },
    }
    line_q: queue.Queue[str | None] = queue.Queue()

    def _read_first() -> None:
        if proc.stdout is None:
            line_q.put(None)
            return
        try:
            ln = proc.stdout.readline()
            line_q.put(ln if ln else None)
        except (OSError, ValueError):
            line_q.put(None)

    reader = threading.Thread(target=_read_first, daemon=True)
    reader.start()
    try:
        if proc.stdin is None:
            return {"supported": False, "error": "ACP subprocess stdin pipe unavailable."}
        proc.stdin.write(json.dumps(init_req, separators=(",", ":")) + "\n")
        proc.stdin.flush()
        raw = line_q.get(timeout=max(0.5, float(timeout_seconds)))
    except (OSError, ValueError, queue.Empty) as exc:
        return {"supported": False, "error": f"ACP initialize timed out or failed: {exc}"}
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    if not raw:
        return {"supported": False, "error": "ACP server closed stdout without responding to initialize."}
    try:
        parsed = json.loads(raw.strip())
    except ValueError as exc:
        return {"supported": False, "error": f"Invalid ACP JSON response: {exc}"}
    result = parsed.get("result") if isinstance(parsed, dict) else None
    if not isinstance(result, dict):
        err = parsed.get("error") if isinstance(parsed, dict) else parsed
        return {"supported": False, "error": f"ACP initialize error: {err}"}
    agent_info = result.get("agentInfo") if isinstance(result.get("agentInfo"), dict) else {}
    auth_methods = result.get("authMethods") if isinstance(result.get("authMethods"), list) else []
    return {
        "supported": True,
        "protocol_version": result.get("protocolVersion"),
        "agent_name": agent_info.get("name"),
        "agent_version": agent_info.get("version"),
        "auth_methods": [str(m.get("id")) for m in auth_methods if isinstance(m, dict) and m.get("id")],
        "capabilities": result.get("agentCapabilities") or {},
    }


def _write_live_step_progress(
    progress_path: str | Path | None,
    *,
    tool_calls_count: int,
    last_tool: str = "",
    last_target: str = "",
) -> None:
    if not progress_path:
        return
    try:
        live_file = Path(f"{progress_path}.live")
        live_file.parent.mkdir(parents=True, exist_ok=True)
        live_file.write_text(
            json.dumps(
                {
                    "straitjacket_calls": int(tool_calls_count),
                    "last_tool": last_tool[:80],
                    "last_target": last_target[:120],
                    "updated_at_epoch": round(time.time(), 3),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _select_acp_model_id(
    model: str | None,
    effort: str | None,
    available_models: Sequence[Any],
    *,
    strict: bool = False,
) -> str | None:
    """Resolve `(model, effort)` onto an `agy acp` `availableModels[].modelId` entry."""
    from server.agy_cli import select_fallback_agy_model

    requested = str(model or "").strip()
    if not requested:
        return None
    eff = str(effort or "").strip().lower()
    offered_ids = [
        str(entry.get("modelId")).strip()
        for entry in available_models
        if isinstance(entry, Mapping) and str(entry.get("modelId") or "").strip()
    ]
    if not offered_ids:
        return f"{requested}-{eff}" if eff and not requested.endswith((f"-{eff}", "-low", "-medium", "-high")) else requested
    offered_set = set(offered_ids)
    candidates: list[str] = []
    if eff and not requested.endswith(("-low", "-medium", "-high")):
        candidates.append(f"{requested}-{eff}")
    candidates.append(requested)
    for default_eff in ("medium", "high", "low"):
        candidates.append(f"{requested}-{default_eff}")
    for cand in candidates:
        if cand in offered_set:
            return cand
    if strict:
        return None
    chosen = select_fallback_agy_model(requested, eff or None, offered_ids)
    return chosen if chosen in offered_set else None


def run_agy_acp_turn(
    *,
    agy_bin: str,
    prompt_text: str,
    cwd: str,
    model: str | None = None,
    effort: str | None = None,
    strict_model_selection: bool = False,
    read_only: bool,
    skip_permissions: bool,
    use_agy_sandbox: bool,
    conversation_id: str | None,
    timeout_seconds: float,
    register_cancel: Callable[[Callable[[], None] | None], None] | None = None,
    max_tool_calls: int | None = None,
    prior_tool_calls_count: int = 0,
    env: Mapping[str, str] | None = None,
    sandbox_prefix: Sequence[str] = (),
    progress_path: str | Path | None = None,
    host_workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one worker turn over `agy acp --stdio` and return the normalized turn result."""
    from server.agy_cli import (
        _close_proc_pipes_safely,
        _custom_config_dir_args,
        _snapshot_pipe_inodes,
        _start_pipe_chunk_drainer,
        _terminate_process_tree,
        build_agy_subprocess_env,
        convert_agy_usage_to_vertex,
    )

    if not prompt_text or not prompt_text.strip():
        raise ValueError("Worker prompt cannot be empty.")
    timeout_sec = max(0.1, float(timeout_seconds))
    cmd: list[str] = [
        *sandbox_prefix,
        agy_bin,
        "acp",
        "--stdio",
        *_custom_config_dir_args(env),
    ]
    if use_agy_sandbox:
        cmd.append("--sandbox")
    if skip_permissions and not read_only:
        cmd.append("--dangerously-skip-permissions")

    spawn_cookie = secrets.token_hex(16)
    sub_env = build_agy_subprocess_env(env, workspace_dir=cwd, host_workspace_root=host_workspace_root)
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
    active_session_id: list[str] = [conversation_id or ""]
    write_lock = threading.Lock()

    def _send_rpc(payload: Mapping[str, Any]) -> None:
        if proc.stdin is None:
            return
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with write_lock:
            try:
                proc.stdin.write(encoded)
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    def _cancel_proc() -> None:
        cancelled_or_timed_out.set()
        if active_session_id[0]:
            _send_rpc(
                {
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": {"sessionId": active_session_id[0]},
                }
            )
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
    t_err = _start_pipe_chunk_drainer(proc.stderr, stderr_chunks)
    line_q: queue.Queue[bytes | None] = queue.Queue()

    def _drain_stdout() -> None:
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

    t_out = threading.Thread(target=_drain_stdout, daemon=True)
    t_out.start()

    text_chunks: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    denied_actions: list[dict[str, Any]] = []
    usage_out = convert_agy_usage_to_vertex(None)
    stop_reason = "STOP"
    budget_exceeded = False
    deadline = time.monotonic() + timeout_sec

    def _await_response(expected_id: int) -> dict[str, Any]:
        nonlocal budget_exceeded, stop_reason, usage_out
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0 or timed_out_flag.is_set():
                raise TimeoutError(f"ACP turn exceeded timeout ({timeout_sec:.1f}s).")
            if cancelled_or_timed_out.is_set():
                raise RuntimeError("ACP turn cancelled by caller.")
            try:
                raw_ln = line_q.get(timeout=min(0.2, rem))
            except queue.Empty:
                continue
            if raw_ln is None:
                err_tail = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()[-500:]
                raise RuntimeError(f"ACP server exited unexpectedly: {err_tail or 'EOF'}")
            s_ln = raw_ln.decode("utf-8", errors="replace").strip()
            if not s_ln.startswith("{"):
                continue
            try:
                msg = json.loads(s_ln)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue

            # Handle server-to-client notifications and permission requests
            method = msg.get("method")
            if isinstance(method, str):
                params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
                if method == "session/update":
                    upd = params.get("update") if isinstance(params.get("update"), dict) else params
                    kind = str(upd.get("sessionUpdate") or upd.get("type") or "").strip()
                    if kind == "agent_message_chunk":
                        content = upd.get("content")
                        if isinstance(content, dict) and isinstance(content.get("text"), str):
                            text_chunks.append(content["text"])
                        elif isinstance(upd.get("text"), str):
                            text_chunks.append(upd["text"])
                    elif kind == "agent_thought_chunk":
                        content = upd.get("content")
                        if isinstance(content, dict) and isinstance(content.get("text"), str):
                            thoughts.append(content["text"])
                    elif kind in ("tool_call", "tool_call_update"):
                        t_title = str(upd.get("title") or upd.get("name") or kind).strip()
                        if kind == "tool_call":
                            raw_in = upd.get("rawInput") if isinstance(upd.get("rawInput"), dict) else {}
                            tool_calls.append({"name": t_title, "args": dict(raw_in)})
                            _write_live_step_progress(
                                progress_path,
                                tool_calls_count=len(tool_calls),
                                last_tool=t_title,
                                last_target=str(raw_in.get("path") or raw_in.get("command") or ""),
                            )
                            if (
                                max_tool_calls is not None
                                and max_tool_calls > 0
                                and (prior_tool_calls_count + len(tool_calls)) > max_tool_calls
                            ):
                                budget_exceeded = True
                                stop_reason = "MAX_TOOL_CALLS_EXCEEDED"
                                if active_session_id[0]:
                                    _send_rpc(
                                        {
                                            "jsonrpc": "2.0",
                                            "method": "session/cancel",
                                            "params": {"sessionId": active_session_id[0]},
                                        }
                                    )
                elif method == "session/request_permission" and "id" in msg:
                    req_id = msg["id"]
                    options = params.get("options") if isinstance(params.get("options"), list) else []
                    tc = params.get("toolCall") if isinstance(params.get("toolCall"), dict) else {}
                    t_name = str(tc.get("title") or tc.get("kind") or "tool").strip()
                    allow_opt = next(
                        (
                            o.get("optionId")
                            for o in options
                            if isinstance(o, dict) and str(o.get("kind") or "").startswith("allow")
                        ),
                        None,
                    )
                    reject_opt = next(
                        (
                            o.get("optionId")
                            for o in options
                            if isinstance(o, dict) and str(o.get("kind") or "").startswith("reject")
                        ),
                        None,
                    )
                    if not read_only and skip_permissions and allow_opt:
                        _send_rpc(
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "result": {"outcome": {"outcome": "selected", "optionId": allow_opt}},
                            }
                        )
                    else:
                        denied_actions.append(
                            {
                                "kind": "command" if "exec" in t_name.lower() or "bash" in t_name.lower() else "write",
                                "tool": t_name,
                                "reason": "Blocked in read-only or restricted ACP worker session.",
                            }
                        )
                        stop_reason = "DENIED_ACTIONS"
                        if reject_opt:
                            _send_rpc(
                                {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "result": {"outcome": {"outcome": "selected", "optionId": reject_opt}},
                                }
                            )
                        else:
                            _send_rpc({"jsonrpc": "2.0", "id": req_id, "result": {"outcome": {"outcome": "cancelled"}}})
                elif "id" in msg:
                    _send_rpc(
                        {
                            "jsonrpc": "2.0",
                            "id": msg["id"],
                            "error": {"code": -32601, "message": f"Unsupported client callback: {method}"},
                        }
                    )
                continue

            if msg.get("id") == expected_id:
                if "error" in msg and msg["error"]:
                    err_obj = msg["error"]
                    err_msg = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
                    raise RuntimeError(f"ACP RPC error: {err_msg}")
                res = msg.get("result")
                return res if isinstance(res, dict) else {}

    try:
        _send_rpc(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": ACP_PROTOCOL_VERSION,
                    "clientCapabilities": {},
                    "clientInfo": {"name": "cc-agy-plugin", "version": "0.3.0"},
                },
            }
        )
        _await_response(1)

        resolved_sid = ""
        session_meta: dict[str, Any] = {}
        if conversation_id:
            try:
                _send_rpc(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/load",
                        "params": {"sessionId": conversation_id, "cwd": str(Path(cwd).resolve()), "mcpServers": []},
                    }
                )
                load_res = _await_response(2)
                resolved_sid = str(load_res.get("sessionId") or conversation_id)
                session_meta = load_res
            except RuntimeError:
                resolved_sid = ""
        if not resolved_sid:
            _send_rpc(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "session/new",
                    "params": {"cwd": str(Path(cwd).resolve()), "mcpServers": []},
                }
            )
            new_res = _await_response(3)
            resolved_sid = str(new_res.get("sessionId") or "")
            session_meta = new_res
        active_session_id[0] = resolved_sid

        models_block = session_meta.get("models") if isinstance(session_meta.get("models"), Mapping) else {}
        avail_models = models_block.get("availableModels") if isinstance(models_block.get("availableModels"), Sequence) else ()
        target_acp_model = _select_acp_model_id(model, effort, avail_models, strict=strict_model_selection)
        if strict_model_selection and model and not target_acp_model:
            raise RuntimeError(
                f"Central Bifrost model policy requires '{model}', which is not offered by this `agy acp` session."
            )
        cur_acp_model = str(models_block.get("currentModelId") or "").strip()
        if resolved_sid and target_acp_model and target_acp_model != cur_acp_model:
            try:
                _send_rpc(
                    {
                        "jsonrpc": "2.0",
                        "id": 10,
                        "method": "session/set_model",
                        "params": {"sessionId": resolved_sid, "modelId": target_acp_model},
                    }
                )
                _await_response(10)
            except RuntimeError as exc:
                if strict_model_selection:
                    raise RuntimeError(
                        f"Central Bifrost model policy requires ACP model '{target_acp_model}', "
                        f"but session/set_model failed: {exc}"
                    ) from exc

        modes_block = session_meta.get("modes") if isinstance(session_meta.get("modes"), Mapping) else {}
        avail_mode_ids = {
            str(m.get("id")).strip()
            for m in (modes_block.get("availableModes") or ())
            if isinstance(m, Mapping) and str(m.get("id") or "").strip()
        }
        desired_mode = "plan" if read_only else ("accept-edits" if skip_permissions else "")
        if resolved_sid and desired_mode and (not avail_mode_ids or desired_mode in avail_mode_ids):
            try:
                _send_rpc(
                    {
                        "jsonrpc": "2.0",
                        "id": 11,
                        "method": "session/set_mode",
                        "params": {"sessionId": resolved_sid, "modeId": desired_mode},
                    }
                )
                _await_response(11)
            except RuntimeError:
                pass

        _send_rpc(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/prompt",
                "params": {
                    "sessionId": resolved_sid,
                    "prompt": [{"type": "text", "text": prompt_text}],
                },
            }
        )
        prompt_res = _await_response(4)
        if isinstance(prompt_res.get("usage"), Mapping):
            usage_out = convert_agy_usage_to_vertex(prompt_res["usage"])
        raw_sr = str(prompt_res.get("stopReason") or "").strip()
        if denied_actions:
            stop_reason = "DENIED_ACTIONS"
        elif budget_exceeded:
            stop_reason = "MAX_TOOL_CALLS_EXCEEDED"
        elif raw_sr and raw_sr.lower() not in ("end_turn", "stop"):
            stop_reason = raw_sr.upper()
        return {
            "ok": True,
            "text": "".join(text_chunks).strip(),
            "usage": usage_out,
            "step_usage_sum": usage_out,
            "usage_from_snapshot": False,
            "num_turns": 1,
            "tool_calls": tool_calls,
            "thoughts": thoughts,
            "conversation_id": resolved_sid or None,
            "stop_reason": stop_reason,
            "denied_actions": denied_actions,
            "error": "",
        }
    except TimeoutError as exc:
        exc.partial_usage = usage_out  # type: ignore[attr-defined]
        exc.partial_tool_calls = tool_calls  # type: ignore[attr-defined]
        exc.partial_thoughts = thoughts  # type: ignore[attr-defined]
        exc.partial_agy_conversation_id = active_session_id[0] or None  # type: ignore[attr-defined]
        raise
    except Exception as exc:
        return {
            "ok": False,
            "text": "".join(text_chunks).strip(),
            "usage": usage_out,
            "step_usage_sum": usage_out,
            "usage_from_snapshot": False,
            "num_turns": 0,
            "tool_calls": tool_calls,
            "thoughts": thoughts,
            "conversation_id": active_session_id[0] or None,
            "stop_reason": "ERROR",
            "denied_actions": denied_actions,
            "error": str(exc),
        }
    finally:
        timer.cancel()
        if callable(register_cancel):
            register_cancel(None)
        _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
        t_out.join(timeout=0.3)
        t_err.join(timeout=0.3)
        _close_proc_pipes_safely(proc, stdout_thread=t_out, stderr_thread=t_err)
