"""Model Context Protocol (MCP) JSON-RPC 2.0 server for cc-agy-plugin."""

from __future__ import annotations

import json
import sys
from typing import Any, BinaryIO

from server.genai_ask import ask_gemini
from server.jobs import JobManager
from server.receipts import ReceiptStore

MCP_PROTOCOL_VERSION = "2024-11-05"

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
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "delegate_start",
        "description": (
            "Start an isolated Antigravity SDK worker in a dedicated git worktree. "
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
            "or omit job_id to list all jobs and cumulative session costs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Optional job_id to inspect.",
                }
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
                    "description": "Max seconds to wait (default: 600).",
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
                    "description": "Optional substring or relative path filter.",
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
            "Apply a verified worker job's changes from its isolated git worktree to the main "
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
            "Synchronous convenience tool: runs an Antigravity worker in an isolated git "
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

    def handle_Message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single JSON-RPC 2.0 message and return a response dict (or None for notifications)."""
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}

        # Notifications have no id and expect no response
        if msg_id is None:
            return None

        try:
            if method == "initialize":
                client_version = params.get("protocolVersion") or MCP_PROTOCOL_VERSION
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": client_version,
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
                arguments = dict(params.get("arguments") or {})
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
            return ask_gemini(
                prompt=str(args.get("prompt") or ""),
                model=args.get("model"),
                context_files=args.get("context_files"),
                system_instruction=args.get("system_instruction"),
                temperature=args.get("temperature"),
                location=args.get("location"),
                workspace_root=self.job_manager.workspace_root,
                store=self.store,
                backend=self.ask_backend,
            )

        if name == "delegate_start":
            return self.job_manager.delegate_start(
                prompt=str(args.get("prompt") or ""),
                model=args.get("model"),
                allowed_paths=args.get("allowed_paths"),
                timeout_seconds=float(args.get("timeout_seconds") or 300.0),
                wait=bool(args.get("wait", False)),
            )

        if name == "delegate_status":
            return self.job_manager.delegate_status(job_id=args.get("job_id"))

        if name == "delegate_result":
            return self.job_manager.delegate_result(
                job_id=str(args.get("job_id") or ""),
                wait=bool(args.get("wait", True)),
                timeout=float(args.get("timeout") or 600.0),
            )

        if name == "get_diff":
            return self.job_manager.get_diff(
                job_id=str(args.get("job_id") or ""),
                file_filter=args.get("file_filter"),
                page=int(args.get("page") or 1),
                page_tokens=int(args.get("page_tokens") or 4000),
            )

        if name == "delegate_apply":
            return self.job_manager.delegate_apply(
                job_id=str(args.get("job_id") or ""),
                cleanup_worktree=bool(args.get("cleanup_worktree", True)),
            )

        if name == "delegate_continue":
            return self.job_manager.delegate_continue(
                job_id=str(args.get("job_id") or ""),
                follow_up_prompt=str(args.get("follow_up_prompt") or ""),
                timeout_seconds=float(args.get("timeout_seconds") or 300.0),
                wait=bool(args.get("wait", True)),
            )

        if name == "delegate_cancel":
            return self.job_manager.delegate_cancel(
                job_id=str(args.get("job_id") or ""),
                cleanup_worktree=bool(args.get("cleanup_worktree", True)),
            )

        if name == "delegate_to_gemini":
            digest = self.job_manager.delegate_start(
                prompt=str(args.get("prompt") or ""),
                model=args.get("model"),
                allowed_paths=args.get("allowed_paths"),
                timeout_seconds=float(args.get("timeout_seconds") or 300.0),
                wait=True,
            )
            auto_apply = bool(args.get("auto_apply", True))
            if auto_apply and digest.get("status") == "ok":
                apply_res = self.job_manager.delegate_apply(str(digest["job_id"]))
                digest["applied"] = True
                digest["applied_files"] = apply_res.get("applied_files", [])
                digest["diff_sha256"] = apply_res.get("diff_sha256")
            elif digest.get("status") != "ok":
                digest["is_error"] = True
            return digest

        raise ValueError(f"Unknown tool: {name!r}")


def serve_stdio(
    server: MCPServer | None = None,
    in_stream: BinaryIO | None = None,
    out_stream: BinaryIO | None = None,
) -> None:
    """Run the stdio JSON-RPC loop supporting both newline-delimited JSON and Content-Length framing."""
    active_server = server or MCPServer()
    reader = in_stream or sys.stdin.buffer
    writer = out_stream or sys.stdout.buffer

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
            length = int(stripped.split(b":", 1)[1].strip())
            # Consume remaining headers until blank line
            while True:
                hdr = reader.readline()
                if not hdr or hdr.strip() == b"":
                    break
            body = reader.read(length)
            if not body:
                break
            raw_msg = body.decode("utf-8")
        else:
            raw_msg = stripped.decode("utf-8")

        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            _write_response(writer, err_resp, use_content_length)
            continue

        resp = active_server.handle_Message(msg)
        if resp is not None:
            _write_response(writer, resp, use_content_length)


def _write_response(
    writer: BinaryIO,
    response: dict[str, Any],
    use_content_length: bool,
) -> None:
    encoded = json.dumps(response, separators=(",", ":")).encode("utf-8")
    if use_content_length:
        header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
        writer.write(header + encoded)
    else:
        writer.write(encoded + b"\n")
    writer.flush()
