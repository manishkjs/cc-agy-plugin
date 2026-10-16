"""Model Context Protocol (MCP) JSON-RPC 2.0 server for cc-agy-plugin."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
import sys
import threading
from typing import Any, BinaryIO

from server.genai_ask import ask_gemini
from server.jobs import JobManager, validate_allowed_paths
from server.receipts import ReceiptStore

MCP_PROTOCOL_VERSION = "2024-11-05"
SUPPORTED_MCP_VERSIONS = frozenset({"2024-11-05", "2025-03-26"})

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "ask_gemini",
        "description": (
            "Run a fast single-shot Vertex AI Gemini query via google-genai (without the "
            "11,554-token Antigravity SDK system prompt floor). Best for code reviews, "
            "architecture feedback, and second opinions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Question or review request for Gemini.",
                },
                "model": {
                    "type": "string",
                    "description": "Vertex Gemini model ID (default: gemini-2.5-flash).",
                },
                "context_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of workspace-relative file paths to attach.",
                },
                "system_instruction": {
                    "type": "string",
                    "description": "Optional system instruction.",
                },
                "temperature": {
                    "type": "number",
                    "description": "Optional sampling temperature.",
                },
                "location": {
                    "type": "string",
                    "description": "Optional Vertex AI region override (auto-routed to global when required).",
                },
                "scrub_receipt": {
                    "type": "boolean",
                    "description": "If true, redact paying project ID and host workspace paths in the saved receipt.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "delegate_start",
        "description": (
            "Start an isolated Antigravity SDK worker in a dedicated standalone repo copy. "
            "Returns a job_id immediately (or waits if wait=true). Failed runs never "
            "pollute the main working tree."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Outcome-oriented task specification (avoid pasting >8 consecutive code lines).",
                },
                "model": {
                    "type": "string",
                    "description": "Vertex Gemini model ID (default: gemini-3.1-pro-preview).",
                },
                "allowed_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of workspace-relative file paths or directory prefixes this worker may modify.",
                },
                "enabled_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional allowlist of Antigravity worker tools.",
                },
                "disabled_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional denylist of Antigravity worker tools.",
                },
                "enable_subagents": {
                    "type": "boolean",
                    "description": "Enable nested subagents inside the Antigravity worker (default: false).",
                },
                "scrub_receipt": {
                    "type": "boolean",
                    "description": "Redact paying project ID and host paths in the receipt (default: false).",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Per-turn timeout in seconds (default: 300).",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, block until the worker finishes and return the digest.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "delegate_status",
        "description": (
            "Check status, token usage, and estimated cost for a specific job_id, "
            "or omit job_id to list recent jobs and cumulative session costs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Optional job_id to inspect.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of recent jobs to return in list mode (default: 20, max: 50).",
                },
            },
        },
    },
    {
        "name": "delegate_result",
        "description": "Wait for a delegated job to finish and return its token-bounded digest.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID returned by delegate_start.",
                },
                "wait": {
                    "type": "boolean",
                    "description": "Whether to wait if the job is still running (default: true).",
                },
                "timeout": {
                    "type": "number",
                    "description": "Max seconds to wait (default: 540, capped below the 600s MCP timeout).",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "get_diff",
        "description": (
            "Retrieve the unified git diff from a worker job's isolated worktree, "
            "paginated by token budget so responses never exceed Claude Code's MCP token cap."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID to inspect.",
                },
                "file_filter": {
                    "type": "string",
                    "description": "Optional exact relative path, directory prefix, basename, or glob filter.",
                },
                "page": {
                    "type": "integer",
                    "description": "1-based page number (default: 1).",
                },
                "page_tokens": {
                    "type": "integer",
                    "description": "Token budget per page (default: 4000, max: 8000).",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "delegate_apply",
        "description": (
            "Apply a verified worker job's changes from its isolated worktree to the main "
            "workspace tree, record provenance hashes on the receipt, and clean up the worktree."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Completed job_id with status='ok'.",
                },
                "cleanup_worktree": {
                    "type": "boolean",
                    "description": "Remove the temporary worktree after applying (default: true).",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "delegate_continue",
        "description": "Send a follow-up instruction to an existing worker job's worktree.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Existing job_id.",
                },
                "follow_up_prompt": {
                    "type": "string",
                    "description": "Follow-up task or test-fix instruction.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 300).",
                },
                "wait": {
                    "type": "boolean",
                    "description": "Wait for the follow-up turn to finish (default: true).",
                },
            },
            "required": ["job_id", "follow_up_prompt"],
        },
    },
    {
        "name": "delegate_cancel",
        "description": "Cancel a running worker job and remove its temporary worktree.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID to cancel.",
                },
                "cleanup_worktree": {
                    "type": "boolean",
                    "description": "Remove the temporary worktree (default: true).",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "delegate_to_gemini",
        "description": (
            "Synchronous convenience tool: runs an Antigravity worker in an isolated "
            "worktree, waits for completion, and optionally applies the verified diff if status='ok'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Task specification for the Gemini worker.",
                },
                "model": {
                    "type": "string",
                    "description": "Vertex Gemini model ID (default: gemini-3.1-pro-preview).",
                },
                "allowed_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of allowed relative paths.",
                },
                "enabled_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional allowlist of Antigravity worker tools.",
                },
                "disabled_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional denylist of Antigravity worker tools.",
                },
                "enable_subagents": {
                    "type": "boolean",
                    "description": "Enable nested subagents inside the Antigravity worker (default: false).",
                },
                "scrub_receipt": {
                    "type": "boolean",
                    "description": "Redact paying project ID and host paths in the receipt (default: false).",
                },
                "auto_apply": {
                    "type": "boolean",
                    "description": "Automatically apply the worktree diff to the main workspace if status='ok' (default: true).",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 300).",
                },
            },
            "required": ["prompt"],
        },
    },
]

KNOWN_TOOL_NAMES: frozenset[str] = frozenset(t["name"] for t in TOOL_DEFINITIONS)


def parse_bool(value: Any, default: bool = False) -> bool:
    """Safely parse boolean tool arguments, treating string 'false'/'0'/'no' as False."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in ("false", "0", "no", "off", ""):
            return False
        if norm in ("true", "1", "yes", "on"):
            return True
    raise ValueError(f"Expected boolean value, got {value!r}")


class InvalidParamsError(ValueError):
    """Raised when MCP tool call parameters fail schema/type validation (-32602)."""


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidParamsError(f"'{field_name}' must be a non-empty string")
    return value


def _parse_positive_float(value: Any, field_name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise InvalidParamsError(f"'{field_name}' must be a positive number, got bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidParamsError(f"'{field_name}' must be a valid number: {exc}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise InvalidParamsError(f"'{field_name}' must be a finite number > 0, got {value!r}")
    return parsed


def _parse_positive_int(value: Any, field_name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise InvalidParamsError(f"'{field_name}' must be a positive integer, got bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidParamsError(f"'{field_name}' must be a valid integer: {exc}") from exc
    if parsed <= 0:
        raise InvalidParamsError(f"'{field_name}' must be an integer > 0, got {value!r}")
    return parsed


class MCPServer:
    """JSON-RPC 2.0 MCP server dispatching Vertex AI tools."""

    def __init__(
        self,
        job_manager: JobManager | None = None,
        store: ReceiptStore | None = None,
        ask_backend: Any = None,
    ) -> None:
        self.store = store or (job_manager.store if job_manager else ReceiptStore())
        self.job_manager = job_manager or JobManager(store=self.store)
        self.ask_backend = ask_backend

    def handle_Message(self, message: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Process a JSON-RPC 2.0 message (or batch) and return a response dict (or None for notifications)."""
        if isinstance(message, list):
            if not message:
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request: empty batch"},
                }
            responses = []
            for item in message:
                resp = self._handle_single_message(item)
                if resp is not None:
                    responses.append(resp)
            return responses if responses else None

        return self._handle_single_message(message)

    def _handle_single_message(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": f"Invalid Request: expected JSON object, got {type(message).__name__}",
                },
            }

        method = message.get("method")
        msg_id = message.get("id")
        has_id = "id" in message and msg_id is not None

        if not isinstance(method, str) or not method.strip():
            return {
                "jsonrpc": "2.0",
                "id": msg_id if has_id else None,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: missing or non-string 'method'",
                },
            }

        params = message.get("params")
        if params is not None and not isinstance(params, dict):
            if not has_id:
                return None
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params: 'params' must be an object",
                },
            }
        params = params or {}

        # Notifications have no id and expect no response
        if not has_id:
            return None

        try:
            if method == "initialize":
                requested_version = str(params.get("protocolVersion") or "")
                negotiated = (
                    requested_version
                    if requested_version in SUPPORTED_MCP_VERSIONS
                    else MCP_PROTOCOL_VERSION
                )
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": negotiated,
                        "capabilities": {
                            "tools": {"listChanged": False},
                        },
                        "serverInfo": {
                            "name": "cc-agy-plugin-vertex",
                            "version": "0.1.0",
                        },
                    },
                }

            if method == "ping":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": TOOL_DEFINITIONS},
                }

            if method == "tools/call":
                tool_name = str(params.get("name") or "")
                if tool_name not in KNOWN_TOOL_NAMES:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32602,
                            "message": f"Invalid params: unknown tool {tool_name!r}",
                        },
                    }
                raw_args = params.get("arguments")
                if raw_args is not None and not isinstance(raw_args, dict):
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid params: 'arguments' must be an object",
                        },
                    }
                arguments = dict(raw_args or {})
                payload = self._dispatch_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(payload, indent=2, sort_keys=True),
                            }
                        ],
                        "isError": bool(payload.get("is_error", False)),
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }
        except InvalidParamsError as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32602,
                    "message": f"Invalid params: {exc}",
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                                indent=2,
                            ),
                        }
                    ],
                    "isError": True,
                },
            }

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "ask_gemini":
            prompt_str = _require_str(args.get("prompt"), "prompt")
            ctx_files = args.get("context_files")
            if ctx_files is not None and (
                not isinstance(ctx_files, list) or not all(isinstance(x, str) for x in ctx_files)
            ):
                raise InvalidParamsError("context_files must be a list of strings")
            try:
                scrub_b = parse_bool(args.get("scrub_receipt"), False)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            return ask_gemini(
                prompt=prompt_str,
                model=args.get("model"),
                context_files=ctx_files,
                system_instruction=args.get("system_instruction"),
                temperature=args.get("temperature"),
                location=args.get("location"),
                workspace_root=self.job_manager.workspace_root,
                store=self.store,
                scrub_receipt=scrub_b,
                backend=self.ask_backend,
            )

        if name == "delegate_start":
            prompt_str = _require_str(args.get("prompt"), "prompt")
            try:
                allowed = validate_allowed_paths(args.get("allowed_paths"))
                enable_sub = parse_bool(args.get("enable_subagents"), False)
                scrub_b = parse_bool(args.get("scrub_receipt"), False)
                wait_b = parse_bool(args.get("wait"), False)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            timeout_sec = _parse_positive_float(args.get("timeout_seconds"), "timeout_seconds", 300.0)
            return self.job_manager.delegate_start(
                prompt=prompt_str,
                model=args.get("model"),
                allowed_paths=allowed,
                enabled_tools=args.get("enabled_tools"),
                disabled_tools=args.get("disabled_tools"),
                enable_subagents=enable_sub,
                scrub_receipt=scrub_b,
                timeout_seconds=timeout_sec,
                wait=wait_b,
            )

        if name == "delegate_status":
            limit_val = _parse_positive_int(args.get("limit"), "limit", 20)
            return self.job_manager.delegate_status(
                job_id=args.get("job_id"),
                limit=limit_val,
            )

        if name == "delegate_result":
            job_id_str = _require_str(args.get("job_id"), "job_id")
            try:
                wait_b = parse_bool(args.get("wait"), True)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            timeout_val = _parse_positive_float(args.get("timeout"), "timeout", 540.0)
            return self.job_manager.delegate_result(
                job_id=job_id_str,
                wait=wait_b,
                timeout=timeout_val,
            )

        if name == "get_diff":
            job_id_str = _require_str(args.get("job_id"), "job_id")
            page_val = _parse_positive_int(args.get("page"), "page", 1)
            page_tok_val = _parse_positive_int(args.get("page_tokens"), "page_tokens", 4000)
            return self.job_manager.get_diff(
                job_id=job_id_str,
                file_filter=args.get("file_filter"),
                page=page_val,
                page_tokens=page_tok_val,
            )

        if name == "delegate_apply":
            job_id_str = _require_str(args.get("job_id"), "job_id")
            try:
                cleanup_b = parse_bool(args.get("cleanup_worktree"), True)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            return self.job_manager.delegate_apply(
                job_id=job_id_str,
                cleanup_worktree=cleanup_b,
            )

        if name == "delegate_continue":
            job_id_str = _require_str(args.get("job_id"), "job_id")
            followup_str = _require_str(args.get("follow_up_prompt"), "follow_up_prompt")
            try:
                wait_b = parse_bool(args.get("wait"), True)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            timeout_sec = _parse_positive_float(args.get("timeout_seconds"), "timeout_seconds", 300.0)
            return self.job_manager.delegate_continue(
                job_id=job_id_str,
                follow_up_prompt=followup_str,
                timeout_seconds=timeout_sec,
                wait=wait_b,
            )

        if name == "delegate_cancel":
            job_id_str = _require_str(args.get("job_id"), "job_id")
            try:
                cleanup_b = parse_bool(args.get("cleanup_worktree"), True)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            return self.job_manager.delegate_cancel(
                job_id=job_id_str,
                cleanup_worktree=cleanup_b,
            )

        if name == "delegate_to_gemini":
            prompt_str = _require_str(args.get("prompt"), "prompt")
            try:
                allowed = validate_allowed_paths(args.get("allowed_paths"))
                enable_sub = parse_bool(args.get("enable_subagents"), False)
                scrub_b = parse_bool(args.get("scrub_receipt"), False)
                auto_apply = parse_bool(args.get("auto_apply"), True)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            timeout_sec = _parse_positive_float(args.get("timeout_seconds"), "timeout_seconds", 300.0)
            digest = self.job_manager.delegate_start(
                prompt=prompt_str,
                model=args.get("model"),
                allowed_paths=allowed,
                enabled_tools=args.get("enabled_tools"),
                disabled_tools=args.get("disabled_tools"),
                enable_subagents=enable_sub,
                scrub_receipt=scrub_b,
                timeout_seconds=timeout_sec,
                wait=True,
            )
            if auto_apply and digest.get("status") == "ok":
                apply_res = self.job_manager.delegate_apply(str(digest["job_id"]))
                digest["applied"] = True
                digest["applied_files"] = apply_res.get("applied_files", [])
                digest["diff_sha256"] = apply_res.get("diff_sha256")
            elif digest.get("status") != "ok":
                digest["is_error"] = True
            return digest

        raise InvalidParamsError(f"Unknown tool: {name!r}")


def serve_stdio(
    server: MCPServer | None = None,
    in_stream: BinaryIO | None = None,
    out_stream: BinaryIO | None = None,
) -> None:
    """Run the stdio JSON-RPC loop supporting concurrent tool execution and both framing modes."""
    active_server = server or MCPServer()
    reader = in_stream or sys.stdin.buffer
    writer = out_stream or sys.stdout.buffer
    write_lock = threading.Lock()
    in_flight_futures: list[Future[None]] = []

    def _process_and_write(msg_obj: Any, framed: bool) -> None:
        resp = active_server.handle_Message(msg_obj)
        if resp is not None:
            with write_lock:
                _write_response(writer, resp, framed)

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="cc-agy-rpc") as rpc_pool:
        while True:
            first_line = reader.readline()
            if not first_line:
                break

            stripped = first_line.strip()
            if not stripped:
                continue

            use_content_length = False
            if stripped.lower().startswith(b"content-length:"):
                use_content_length = True
                # Always consume remaining headers until blank line
                while True:
                    hdr = reader.readline()
                    if not hdr or hdr.strip() == b"":
                        break
                try:
                    length = int(stripped.split(b":", 1)[1].strip())
                    if length < 0:
                        raise ValueError("Negative Content-Length")
                except ValueError as exc:
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Invalid Content-Length header: {exc}"},
                    }
                    with write_lock:
                        _write_response(writer, err_resp, use_content_length)
                    continue
                body = reader.read(length) if length > 0 else b""
                if length > 0 and not body:
                    break
                try:
                    raw_msg = body.decode("utf-8")
                except UnicodeDecodeError as exc:
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Invalid UTF-8 payload: {exc}"},
                    }
                    with write_lock:
                        _write_response(writer, err_resp, use_content_length)
                    continue
            else:
                try:
                    raw_msg = stripped.decode("utf-8")
                except UnicodeDecodeError as exc:
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Invalid UTF-8 payload: {exc}"},
                    }
                    with write_lock:
                        _write_response(writer, err_resp, use_content_length)
                    continue

            parsed_messages: list[Any] = []
            try:
                parsed_messages.append(json.loads(raw_msg))
            except json.JSONDecodeError as exc:
                # Attempt multi-JSON stream recovery (e.g. concatenated JSON objects on one line)
                decoder = json.JSONDecoder()
                idx = 0
                n_chars = len(raw_msg)
                recovered: list[Any] = []
                recovery_ok = True
                while idx < n_chars:
                    while idx < n_chars and raw_msg[idx].isspace():
                        idx += 1
                    if idx >= n_chars:
                        break
                    try:
                        obj, next_idx = decoder.raw_decode(raw_msg, idx)
                        recovered.append(obj)
                        idx = next_idx
                    except json.JSONDecodeError:
                        recovery_ok = False
                        break
                if recovery_ok and recovered:
                    parsed_messages.extend(recovered)
                else:
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {exc}"},
                    }
                    with write_lock:
                        _write_response(writer, err_resp, use_content_length)
                    continue

            # Dispatch tools/call concurrently so ping, delegate_status, and delegate_cancel
            # never block behind an in-flight synchronous worker call.
            for msg in parsed_messages:
                if isinstance(msg, dict) and msg.get("method") == "tools/call":
                    fut = rpc_pool.submit(_process_and_write, msg, use_content_length)
                    in_flight_futures.append(fut)
                else:
                    _process_and_write(msg, use_content_length)

        for fut in in_flight_futures:
            try:
                fut.result(timeout=60.0)
            except Exception:
                pass


def _write_response(
    writer: BinaryIO,
    response: dict[str, Any] | list[dict[str, Any]],
    use_content_length: bool,
) -> None:
    encoded = json.dumps(response, separators=(",", ":")).encode("utf-8")
    if use_content_length:
        header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
        writer.write(header + encoded)
    else:
        writer.write(encoded + b"\n")
    writer.flush()
