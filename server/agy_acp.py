"""Agent Client Protocol (ACP) stdio client and warm session pool for `agy acp`.

Provides structured JSON-RPC 2.0 session creation/resumption (`initialize`,
`session/new`, `session/load`, `session/set_model`, `session/set_mode`,
`session/prompt`, `session/cancel`, `session/close`), warm per-job process
pooling across `delegate_continue` and `/prewalk` phases, permission policy
enforcement (`allowed_paths` + secret-file denylist), and `agy acp --sandbox`
integration when `CC_AGY_CLI_PROTOCOL=acp`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import re
import secrets
import subprocess
import threading
import time
from typing import Any

from server.prewalk import evaluate_acp_permission_policy, extract_edit_path_from_acp_event

ACP_PROTOCOL_VERSION = 1
CLI_PROTOCOLS = ("stream-json", "acp")

_ACP_UPSTREAM_API_ERROR_RE = re.compile(
    r"^\s*(?:⚠\s*|Warning:\s*|Error:\s*)?"
    r"(?:INVALID_ARGUMENT|PERMISSION_DENIED|UNAUTHENTICATED|RESOURCE_EXHAUSTED|"
    r"NOT_FOUND|INTERNAL|UNAVAILABLE|DEADLINE_EXCEEDED|FAILED_PRECONDITION|ABORTED)\s*"
    r"\(code\s+\d{3}\)\s*:",
    re.IGNORECASE,
)


def _extract_acp_turn_error(
    prompt_res: Mapping[str, Any],
    final_text: str,
    *,
    tool_calls: Sequence[Any],
    usage_out: Mapping[str, int],
) -> str:
    """Detect upstream API error banners emitted as `agent_message_chunk` with 0 output tokens."""
    raw_err = prompt_res.get("error")
    if isinstance(raw_err, str) and raw_err.strip():
        return raw_err.strip()
    if isinstance(raw_err, Mapping):
        msg = str(raw_err.get("message") or raw_err).strip()
        if msg:
            return msg
    if prompt_res.get("isError") is True or prompt_res.get("is_error") is True:
        return final_text.strip() or "ACP prompt turn returned isError=true."
    if (
        not tool_calls
        and int(usage_out.get("candidates_token_count", 0) or 0) == 0
        and final_text
        and _ACP_UPSTREAM_API_ERROR_RE.search(final_text)
    ):
        return final_text.strip()
    return ""


def resolve_cli_protocol(env: Mapping[str, str] | None = None) -> str:
    """Return `'acp'` when `CC_AGY_CLI_PROTOCOL=acp`, else default `'stream-json'`."""
    src = os.environ if env is None else env
    raw = (
        src.get("CC_AGY_CLI_PROTOCOL")
        or src.get("CLAUDE_PLUGIN_OPTION_CLI_PROTOCOL")
        or src.get("CC_AGY_USERCFG_CLI_PROTOCOL")
        or ""
    ).strip().lower()
    if raw == "acp":
        return "acp"
    return "stream-json"


@dataclass
class WarmAcpSession:
    """A warm `agy acp --stdio` process bound to an active delegation `job_id`."""

    job_id: str
    agy_bin: str
    cwd: str
    proc: subprocess.Popen[bytes]
    spawn_cookie: str
    pipe_inodes: set[tuple[int, int]]
    line_q: queue.Queue[bytes | None]
    stderr_chunks: list[bytes]
    write_lock: threading.Lock = field(default_factory=threading.Lock)
    turn_lock: threading.Lock = field(default_factory=threading.Lock)
    initialized: bool = False
    session_id: str = ""
    session_meta: dict[str, Any] = field(default_factory=dict)
    current_model_id: str = ""
    current_mode_id: str = ""
    closed: bool = False
    _rpc_seq: int = 10

    def next_rpc_id(self) -> int:
        with self.write_lock:
            self._rpc_seq += 1
            return self._rpc_seq

    def is_alive(self) -> bool:
        return not self.closed and self.proc.poll() is None

    def send_rpc(self, payload: Mapping[str, Any]) -> None:
        if self.closed or self.proc.stdin is None:
            return
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with self.write_lock:
            try:
                self.proc.stdin.write(encoded)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass


class AcpSessionPool:
    """Thread-safe registry of warm `agy acp --stdio` processes keyed by `job_id`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, WarmAcpSession] = {}

    def get(self, job_id: str) -> WarmAcpSession | None:
        with self._lock:
            sess = self._sessions.get(job_id)
            if sess is not None and not sess.is_alive():
                self._sessions.pop(job_id, None)
                return None
            return sess

    def put(self, session: WarmAcpSession) -> None:
        old: WarmAcpSession | None = None
        with self._lock:
            prev = self._sessions.get(session.job_id)
            if prev is not None and prev is not session:
                old = prev
            self._sessions[session.job_id] = session
        if old is not None:
            _terminate_warm_session(old)

    def close_job(self, job_id: str | None) -> bool:
        if not job_id:
            return False
        with self._lock:
            sess = self._sessions.pop(str(job_id), None)
        if sess is None:
            return False
        _terminate_warm_session(sess)
        return True

    def close_all(self) -> int:
        with self._lock:
            items = list(self._sessions.values())
            self._sessions.clear()
        for sess in items:
            _terminate_warm_session(sess)
        return len(items)

    def active_job_ids(self) -> list[str]:
        with self._lock:
            return [jid for jid, sess in self._sessions.items() if sess.is_alive()]


_WARM_ACP_POOL = AcpSessionPool()


def _terminate_warm_session(sess: WarmAcpSession) -> None:
    """Send best-effort `session/close` then terminate the entire ACP process tree."""
    from server.agy_cli import _close_proc_pipes_safely, _terminate_process_tree

    sess.closed = True
    if sess.session_id and sess.proc.poll() is None:
        sess.send_rpc(
            {
                "jsonrpc": "2.0",
                "id": sess.next_rpc_id(),
                "method": "session/close",
                "params": {"sessionId": sess.session_id},
            }
        )
    if sess.proc.stdin is not None:
        try:
            sess.proc.stdin.close()
        except Exception:
            pass
    try:
        sess.proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(sess.proc, spawn_cookie=sess.spawn_cookie, pipe_inodes=sess.pipe_inodes)
    except Exception:
        pass
    _close_proc_pipes_safely(sess.proc)


def close_acp_job_session(job_id: str | None) -> bool:
    """Close and terminate any warm `agy acp` session associated with `job_id`."""
    return _WARM_ACP_POOL.close_job(job_id)


def close_all_acp_sessions() -> int:
    """Close and terminate all warm `agy acp` sessions (used on `JobManager.shutdown`)."""
    return _WARM_ACP_POOL.close_all()


def active_warm_acp_jobs() -> list[str]:
    """Return `job_id`s that currently hold a live warm `agy acp` process."""
    return _WARM_ACP_POOL.active_job_ids()


def _spawn_acp_session_process(
    *,
    job_id: str,
    agy_bin: str,
    cwd: str,
    read_only: bool,
    skip_permissions: bool,
    use_agy_sandbox: bool,
    env: Mapping[str, str] | None,
    sandbox_prefix: Sequence[str],
    host_workspace_root: str | Path | None,
) -> WarmAcpSession:
    from server.agy_cli import (
        _custom_config_dir_args,
        _snapshot_pipe_inodes,
        _start_pipe_chunk_drainer,
        build_agy_subprocess_env,
    )

    cmd: list[str] = [
        *sandbox_prefix,
        agy_bin,
        "acp",
        "--stdio",
        *_custom_config_dir_args(env),
    ]
    if use_agy_sandbox:
        cmd.append("--sandbox")
    # Notice: when allowed_paths or read_only is active, we still enforce policy in
    # `session/request_permission`; passing `--dangerously-skip-permissions` suppresses
    # `session/request_permission` callbacks inside `agy`, so omit it whenever `allowed_paths`
    # requires per-file callback enforcement!
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
    stderr_chunks: list[bytes] = []
    _start_pipe_chunk_drainer(proc.stderr, stderr_chunks)
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

    return WarmAcpSession(
        job_id=job_id,
        agy_bin=agy_bin,
        cwd=str(Path(cwd).resolve()),
        proc=proc,
        spawn_cookie=spawn_cookie,
        pipe_inodes=pipe_inodes,
        line_q=line_q,
        stderr_chunks=stderr_chunks,
    )


def probe_agy_acp_capabilities(
    agy_bin: str,
    *,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    """Perform a lightweight local `agy acp --stdio` `initialize` handshake (0 tokens, 0 network calls)."""
    from server.agy_cli import (
        _close_proc_pipes_safely,
        _custom_config_dir_args,
        _snapshot_pipe_inodes,
        _terminate_process_tree,
        build_agy_subprocess_env,
    )

    cmd = [agy_bin, "acp", "--stdio", *_custom_config_dir_args(env)]
    sub_env = build_agy_subprocess_env(env)
    spawn_cookie = secrets.token_hex(16)
    sub_env["_AGY_SPAWN_COOKIE"] = spawn_cookie

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=(os.name != "nt"),
            env=sub_env,
        )
    except OSError as exc:
        return {"supported": False, "error": f"{type(exc).__name__}: {exc}"}

    pipe_inodes = _snapshot_pipe_inodes(proc)
    line_q: queue.Queue[bytes | None] = queue.Queue()

    def _reader() -> None:
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

    t_out = threading.Thread(target=_reader, daemon=True)
    t_out.start()

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
    try:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(init_req) + "\n").encode("utf-8"))
        proc.stdin.flush()
        deadline = time.monotonic() + max(0.5, float(timeout_seconds))
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                return {"supported": False, "error": "ACP initialize handshake timed out."}
            try:
                raw = line_q.get(timeout=min(0.25, rem))
            except queue.Empty:
                continue
            if raw is None:
                return {"supported": False, "error": "ACP server closed stdout without responding to initialize."}
            text = raw.decode("utf-8", errors="replace").strip()
            if not text.startswith("{"):
                continue
            try:
                msg = json.loads(text)
            except ValueError:
                continue
            if isinstance(msg, dict) and msg.get("id") == 1:
                if "error" in msg and msg["error"]:
                    return {"supported": False, "error": str(msg["error"])}
                res = msg.get("result") if isinstance(msg.get("result"), dict) else {}
                agent_info = res.get("agentInfo") if isinstance(res.get("agentInfo"), dict) else {}
                caps = res.get("agentCapabilities") if isinstance(res.get("agentCapabilities"), dict) else {}
                auth_methods = res.get("authMethods") if isinstance(res.get("authMethods"), list) else []
                return {
                    "supported": True,
                    "protocol_version": res.get("protocolVersion", ACP_PROTOCOL_VERSION),
                    "agent_name": agent_info.get("name"),
                    "agent_version": agent_info.get("version"),
                    "capabilities": caps,
                    "auth_methods": [
                        m.get("id") for m in auth_methods if isinstance(m, dict) and m.get("id")
                    ],
                }
    finally:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
        _close_proc_pipes_safely(proc)


def _write_live_step_progress(
    progress_path: str | Path | None,
    *,
    tool_calls_count: int,
    last_tool: str = "",
    last_target: str = "",
) -> None:
    if not progress_path:
        return
    live_p = Path(f"{progress_path}.live")
    payload = {
        "straitjacket_calls": int(tool_calls_count),
        "last_tool": last_tool[:80],
        "last_target": last_target[:120],
        "updated_at_epoch": round(time.time(), 3),
    }
    try:
        live_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = live_p.with_suffix(f".live.{os.getpid()}.tmp")
        tmp_p.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_p, live_p)
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
    job_id: str | None = None,
    keep_warm: bool = False,
    allowed_paths: Sequence[str] | None = None,
    stop_on_first_edit: bool = False,
) -> dict[str, Any]:
    """Execute one worker turn over `agy acp --stdio`, reusing a warm session when `keep_warm=True`."""
    from server.agy_cli import convert_agy_usage_to_vertex

    if not prompt_text or not prompt_text.strip():
        raise ValueError("Worker prompt cannot be empty.")
    timeout_sec = max(0.1, float(timeout_seconds))
    resolved_cwd = str(Path(cwd).resolve())
    pool_key = str(job_id).strip() if (job_id and keep_warm) else ""

    # When `allowed_paths` restricts write scope, require `agy` to emit `session/request_permission`
    # callbacks so `evaluate_acp_permission_policy` can block out-of-scope edits before disk write!
    effective_cli_skip_perms = bool(skip_permissions and not read_only and not allowed_paths)

    warm_sess = _WARM_ACP_POOL.get(pool_key) if pool_key else None
    created_here = False
    if warm_sess is None or warm_sess.cwd != resolved_cwd or warm_sess.agy_bin != agy_bin:
        if warm_sess is not None and pool_key:
            _WARM_ACP_POOL.close_job(pool_key)
        warm_sess = _spawn_acp_session_process(
            job_id=pool_key or f"ephemeral-{secrets.token_hex(6)}",
            agy_bin=agy_bin,
            cwd=resolved_cwd,
            read_only=read_only,
            skip_permissions=effective_cli_skip_perms,
            use_agy_sandbox=use_agy_sandbox,
            env=env,
            sandbox_prefix=sandbox_prefix,
            host_workspace_root=host_workspace_root,
        )
        created_here = True
        if pool_key:
            _WARM_ACP_POOL.put(warm_sess)

    assert warm_sess is not None
    cancelled_by_caller = threading.Event()
    timed_out_flag = threading.Event()
    destroy_process_on_exit = not bool(pool_key)

    def _hard_cancel_and_kill() -> None:
        nonlocal destroy_process_on_exit
        cancelled_by_caller.set()
        destroy_process_on_exit = True
        if warm_sess.session_id:
            warm_sess.send_rpc(
                {
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": {"sessionId": warm_sess.session_id},
                }
            )
        if pool_key:
            _WARM_ACP_POOL.close_job(pool_key)
        else:
            _terminate_warm_session(warm_sess)

    if callable(register_cancel):
        register_cancel(_hard_cancel_and_kill)

    def _on_timeout() -> None:
        timed_out_flag.set()
        _hard_cancel_and_kill()

    timer = threading.Timer(timeout_sec, _on_timeout)
    timer.daemon = True
    timer.start()

    text_chunks: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    denied_actions: list[dict[str, Any]] = []
    usage_out = convert_agy_usage_to_vertex(None)
    stop_reason = "STOP"
    budget_exceeded = False
    prewalk_first_edit_path: str | None = None
    prewalk_cancel_sent_at: float | None = None
    deadline = time.monotonic() + timeout_sec

    def _await_response(expected_id: int, *, is_prompt_turn: bool = False) -> dict[str, Any]:
        nonlocal budget_exceeded, stop_reason, usage_out, prewalk_first_edit_path, prewalk_cancel_sent_at
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0 or timed_out_flag.is_set():
                raise TimeoutError(f"ACP turn exceeded timeout ({timeout_sec:.1f}s).")
            if cancelled_by_caller.is_set():
                raise RuntimeError("ACP turn cancelled by caller.")
            if prewalk_cancel_sent_at is not None and (time.monotonic() - prewalk_cancel_sent_at) >= 2.5:
                return {"stopReason": "cancelled"}
            wait_slice = min(0.2, rem)
            if prewalk_cancel_sent_at is not None:
                wait_slice = min(wait_slice, max(0.05, 2.5 - (time.monotonic() - prewalk_cancel_sent_at)))
            try:
                raw_ln = warm_sess.line_q.get(timeout=wait_slice)
            except queue.Empty:
                continue
            if raw_ln is None:
                err_tail = b"".join(warm_sess.stderr_chunks).decode("utf-8", errors="replace").strip()[-500:]
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
                            if not raw_in and isinstance(upd.get("args"), dict):
                                raw_in = upd["args"]
                            tool_calls.append({"name": t_title, "args": dict(raw_in)})
                            _write_live_step_progress(
                                progress_path,
                                tool_calls_count=prior_tool_calls_count + len(tool_calls),
                                last_tool=t_title,
                                last_target=str(
                                    raw_in.get("path")
                                    or raw_in.get("Path")
                                    or raw_in.get("targetFile")
                                    or raw_in.get("command")
                                    or ""
                                ),
                            )
                            if (
                                max_tool_calls is not None
                                and max_tool_calls > 0
                                and (prior_tool_calls_count + len(tool_calls)) > max_tool_calls
                            ):
                                budget_exceeded = True
                                stop_reason = "MAX_TOOL_CALLS_EXCEEDED"
                                if warm_sess.session_id:
                                    warm_sess.send_rpc(
                                        {
                                            "jsonrpc": "2.0",
                                            "method": "session/cancel",
                                            "params": {"sessionId": warm_sess.session_id},
                                        }
                                    )
                        if (
                            is_prompt_turn
                            and stop_on_first_edit
                            and prewalk_first_edit_path is None
                        ):
                            edited_rel = extract_edit_path_from_acp_event(
                                upd,
                                workspace_dir=resolved_cwd,
                                allowed_paths=allowed_paths,
                            )
                            if edited_rel is not None:
                                prewalk_first_edit_path = edited_rel
                                stop_reason = "PREWALK_FIRST_EDIT"
                                prewalk_cancel_sent_at = time.monotonic()
                                if warm_sess.session_id:
                                    warm_sess.send_rpc(
                                        {
                                            "jsonrpc": "2.0",
                                            "method": "session/cancel",
                                            "params": {"sessionId": warm_sess.session_id},
                                        }
                                    )
                elif method == "session/request_permission" and "id" in msg:
                    req_id = msg["id"]
                    options = params.get("options") if isinstance(params.get("options"), list) else []
                    tc = params.get("toolCall") if isinstance(params.get("toolCall"), dict) else {}
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
                    allowed_by_policy, denied_entry = evaluate_acp_permission_policy(
                        tc,
                        workspace_dir=resolved_cwd,
                        allowed_paths=allowed_paths,
                        read_only=read_only,
                        skip_permissions=skip_permissions,
                    )
                    if allowed_by_policy and allow_opt:
                        warm_sess.send_rpc(
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "result": {"outcome": {"outcome": "selected", "optionId": allow_opt}},
                            }
                        )
                    else:
                        if denied_entry is not None:
                            denied_actions.append(denied_entry)
                        else:
                            t_name = str(tc.get("title") or tc.get("name") or tc.get("kind") or "tool").strip()
                            denied_actions.append(
                                {
                                    "kind": "write",
                                    "tool": t_name,
                                    "reason": "Blocked in restricted ACP worker session.",
                                }
                            )
                        stop_reason = "DENIED_ACTIONS"
                        if reject_opt:
                            warm_sess.send_rpc(
                                {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "result": {"outcome": {"outcome": "selected", "optionId": reject_opt}},
                                }
                            )
                        else:
                            warm_sess.send_rpc(
                                {"jsonrpc": "2.0", "id": req_id, "result": {"outcome": {"outcome": "cancelled"}}}
                            )
                        if warm_sess.session_id:
                            warm_sess.send_rpc(
                                {
                                    "jsonrpc": "2.0",
                                    "method": "session/cancel",
                                    "params": {"sessionId": warm_sess.session_id},
                                }
                            )
                elif "id" in msg:
                    warm_sess.send_rpc(
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
        with warm_sess.turn_lock:
            if not warm_sess.initialized:
                init_id = warm_sess.next_rpc_id()
                warm_sess.send_rpc(
                    {
                        "jsonrpc": "2.0",
                        "id": init_id,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": ACP_PROTOCOL_VERSION,
                            "clientCapabilities": {},
                            "clientInfo": {"name": "cc-agy-plugin", "version": "0.3.0"},
                        },
                    }
                )
                _await_response(init_id)
                warm_sess.initialized = True

            resolved_sid = warm_sess.session_id
            if conversation_id and resolved_sid and conversation_id != resolved_sid:
                resolved_sid = ""

            if not resolved_sid and conversation_id:
                try:
                    load_id = warm_sess.next_rpc_id()
                    warm_sess.send_rpc(
                        {
                            "jsonrpc": "2.0",
                            "id": load_id,
                            "method": "session/load",
                            "params": {
                                "sessionId": conversation_id,
                                "cwd": resolved_cwd,
                                "mcpServers": [],
                            },
                        }
                    )
                    load_res = _await_response(load_id)
                    resolved_sid = str(load_res.get("sessionId") or conversation_id)
                    warm_sess.session_meta = dict(load_res)
                except RuntimeError:
                    resolved_sid = ""

            if not resolved_sid:
                new_id = warm_sess.next_rpc_id()
                warm_sess.send_rpc(
                    {
                        "jsonrpc": "2.0",
                        "id": new_id,
                        "method": "session/new",
                        "params": {"cwd": resolved_cwd, "mcpServers": []},
                    }
                )
                new_res = _await_response(new_id)
                resolved_sid = str(new_res.get("sessionId") or "")
                warm_sess.session_meta = dict(new_res)

            warm_sess.session_id = resolved_sid
            session_meta = warm_sess.session_meta

            models_block = session_meta.get("models") if isinstance(session_meta.get("models"), Mapping) else {}
            avail_models = (
                models_block.get("availableModels")
                if isinstance(models_block.get("availableModels"), Sequence)
                else ()
            )
            target_acp_model = _select_acp_model_id(model, effort, avail_models, strict=strict_model_selection)
            if strict_model_selection and model and not target_acp_model:
                raise RuntimeError(
                    f"Central Bifrost model policy requires '{model}', which is not offered by this `agy acp` session."
                )
            cur_acp_model = warm_sess.current_model_id or str(models_block.get("currentModelId") or "").strip()
            if resolved_sid and target_acp_model and target_acp_model != cur_acp_model:
                try:
                    set_model_id = warm_sess.next_rpc_id()
                    warm_sess.send_rpc(
                        {
                            "jsonrpc": "2.0",
                            "id": set_model_id,
                            "method": "session/set_model",
                            "params": {"sessionId": resolved_sid, "modelId": target_acp_model},
                        }
                    )
                    _await_response(set_model_id)
                    warm_sess.current_model_id = target_acp_model
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
            cur_acp_mode = warm_sess.current_mode_id or str(modes_block.get("currentModeId") or "").strip()
            if (
                resolved_sid
                and desired_mode
                and desired_mode != cur_acp_mode
                and (not avail_mode_ids or desired_mode in avail_mode_ids)
            ):
                try:
                    set_mode_id = warm_sess.next_rpc_id()
                    warm_sess.send_rpc(
                        {
                            "jsonrpc": "2.0",
                            "id": set_mode_id,
                            "method": "session/set_mode",
                            "params": {"sessionId": resolved_sid, "modeId": desired_mode},
                        }
                    )
                    _await_response(set_mode_id)
                    warm_sess.current_mode_id = desired_mode
                except RuntimeError:
                    pass

            prompt_rpc_id = warm_sess.next_rpc_id()
            warm_sess.send_rpc(
                {
                    "jsonrpc": "2.0",
                    "id": prompt_rpc_id,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": resolved_sid,
                        "prompt": [{"type": "text", "text": prompt_text}],
                    },
                }
            )
            prompt_res = _await_response(prompt_rpc_id, is_prompt_turn=True)
            if isinstance(prompt_res.get("usage"), Mapping):
                usage_out = convert_agy_usage_to_vertex(prompt_res["usage"])
            final_text = "".join(text_chunks).strip()
            acp_turn_err = _extract_acp_turn_error(
                prompt_res,
                final_text,
                tool_calls=tool_calls,
                usage_out=usage_out,
            )
            if acp_turn_err:
                return {
                    "ok": False,
                    "text": final_text,
                    "usage": usage_out,
                    "step_usage_sum": usage_out,
                    "usage_from_snapshot": False,
                    "num_turns": 0,
                    "tool_calls": tool_calls,
                    "thoughts": thoughts,
                    "conversation_id": resolved_sid or None,
                    "active_acp_model": warm_sess.current_model_id or cur_acp_model or (model or ""),
                    "warm_session_reused": not created_here,
                    "prewalk_first_edit_path": None,
                    "stop_reason": "ERROR",
                    "denied_actions": denied_actions,
                    "error": acp_turn_err,
                }
            raw_sr = str(prompt_res.get("stopReason") or "").strip()
            if denied_actions:
                stop_reason = "DENIED_ACTIONS"
            elif budget_exceeded:
                stop_reason = "MAX_TOOL_CALLS_EXCEEDED"
            elif prewalk_first_edit_path is not None:
                stop_reason = "PREWALK_FIRST_EDIT"
            elif raw_sr and raw_sr.lower() not in ("end_turn", "stop"):
                stop_reason = raw_sr.upper()
            return {
                "ok": True,
                "text": final_text,
                "usage": usage_out,
                "step_usage_sum": usage_out,
                "usage_from_snapshot": False,
                "num_turns": 1,
                "tool_calls": tool_calls,
                "thoughts": thoughts,
                "conversation_id": resolved_sid or None,
                "active_acp_model": warm_sess.current_model_id or cur_acp_model or (model or ""),
                "warm_session_reused": not created_here,
                "prewalk_first_edit_path": prewalk_first_edit_path,
                "stop_reason": stop_reason,
                "denied_actions": denied_actions,
                "error": "",
            }
    except TimeoutError as exc:
        destroy_process_on_exit = True
        exc.partial_usage = usage_out  # type: ignore[attr-defined]
        exc.partial_tool_calls = tool_calls  # type: ignore[attr-defined]
        exc.partial_thoughts = thoughts  # type: ignore[attr-defined]
        exc.partial_agy_conversation_id = warm_sess.session_id or conversation_id or None  # type: ignore[attr-defined]
        raise
    except Exception as exc:
        if not warm_sess.is_alive() or cancelled_by_caller.is_set():
            destroy_process_on_exit = True
        return {
            "ok": False,
            "text": "".join(text_chunks).strip(),
            "usage": usage_out,
            "step_usage_sum": usage_out,
            "usage_from_snapshot": False,
            "num_turns": 0,
            "tool_calls": tool_calls,
            "thoughts": thoughts,
            "conversation_id": warm_sess.session_id or conversation_id or None,
            "stop_reason": "ERROR",
            "denied_actions": denied_actions,
            "error": str(exc),
        }
    finally:
        timer.cancel()
        if callable(register_cancel):
            try:
                register_cancel(None)
            except Exception:
                pass
        if destroy_process_on_exit:
            if pool_key:
                _WARM_ACP_POOL.close_job(pool_key)
            else:
                _terminate_warm_session(warm_sess)
