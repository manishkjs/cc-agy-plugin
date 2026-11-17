"""Host tool-name and argument tables for the shared hook adapter.

Each host (OpenCode, Codex, agy) names its built-in tools and their arguments
differently. The guard scripts under `scripts/` only understand Claude Code
names (`Read`, `Edit`, `Write`, `MultiEdit`, `Grep`, `Glob`, `Bash`, and MCP
tools spelled `mcp__<server>__<tool>`), so `scripts/hook_adapter.py` uses this
module to rewrite a host tool call into that canonical shape.

Stdlib only; Python 3.9+.
"""

from __future__ import annotations

import re
from typing import Any

# OpenCode built-ins (lower-case ids from `opencode/src/tool/*`).
OPENCODE_TOOLS: dict[str, str] = {
    "read": "Read",
    "edit": "Edit",
    "multiedit": "MultiEdit",
    "write": "Write",
    "patch": "Edit",
    "bash": "Bash",
    "grep": "Grep",
    "glob": "Glob",
    "list": "LS",
    "webfetch": "WebFetch",
    "todowrite": "TodoWrite",
    "todoread": "TodoRead",
    "task": "Task",
}

OPENCODE_ARGS: dict[str, str] = {
    "filePath": "file_path",
    "oldString": "old_string",
    "newString": "new_string",
    "replaceAll": "replace_all",
}

# agy (Jetski cortex) built-ins.
AGY_TOOLS: dict[str, str] = {
    "view_file": "Read",
    "view_file_outline": "Read",
    "view_code_item": "Read",
    "replace_file_content": "Edit",
    "multi_replace_file_content": "MultiEdit",
    "write_to_file": "Write",
    "run_command": "Bash",
    "grep_search": "Grep",
    "find_by_name": "Glob",
    "list_dir": "LS",
    # Jetski IDE spellings.
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
}

AGY_ARGS: dict[str, str] = {
    "CommandLine": "command",
    "AbsolutePath": "file_path",
    "TargetFile": "file_path",
    "File": "file_path",
    "SearchPath": "path",
    "SearchDirectory": "path",
    "DirectoryPath": "path",
    "Query": "pattern",
    "Pattern": "pattern",
    "TargetContent": "old_string",
    "ReplacementContent": "new_string",
    "CodeContent": "content",
    "Cwd": "cwd",
    "Path": "file_path",
    "Target": "old_string",
    "Replacement": "new_string",
    "Content": "content",
}

# Tool names the plugin's own MCP server exposes (suffix after the server).
_PLUGIN_MCP_TOOL_RE = re.compile(r"(?:delegate_[A-Za-z0-9_]+|ask_gemini[A-Za-z0-9_]*)\Z")
_KNOWN_SERVER_PREFIXES: tuple[str, ...] = ("vertex", "agy", "cc-agy-plugin", "cc_agy_plugin")


def _split_server_tool(name: str) -> tuple[str, str] | None:
    """Split a sanitized `<server>_<tool>` name into its parts, or return None."""
    for srv in _KNOWN_SERVER_PREFIXES:
        if name.startswith(srv + "_") and len(name) > len(srv) + 1:
            return srv, name[len(srv) + 1 :]
    match = re.match(r"\A(?P<srv>[A-Za-z0-9_-]+?)_(?P<tool>delegate_[A-Za-z0-9_]+)\Z", name)
    if match:
        return match.group("srv"), match.group("tool")
    return None


def canonical_mcp_name(name: str) -> str | None:
    """Return `mcp__<server>__<tool>` for a host MCP tool name, or None if not MCP-like."""
    if name.startswith("mcp__"):
        return name
    stripped = name[4:] if name.startswith("mcp_") else name
    parts = _split_server_tool(stripped)
    if parts is None:
        return None
    srv, tool = parts
    if name.startswith("mcp_") or srv in _KNOWN_SERVER_PREFIXES or _PLUGIN_MCP_TOOL_RE.match(tool):
        return f"mcp__{srv}__{tool}"
    return None


def _rename_args(args: Any, table: dict[str, str]) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    out: dict[str, Any] = {}
    for key, val in args.items():
        new_key = table.get(key, key)
        if new_key not in out or key == new_key:
            out[new_key] = val
    return out


def _agy_line_window(out: dict[str, Any]) -> None:
    """Translate agy `StartLine`/`EndLine` into CC `offset`/`limit` for Read guards."""
    start, end = out.pop("StartLine", None), out.pop("EndLine", None)
    if start is None and end is None:
        return
    try:
        s = int(start) if start is not None else 1
        out["offset"] = s
        if end is not None:
            out["limit"] = max(1, int(end) - s + 1)
    except (TypeError, ValueError):
        return


def map_tool(host: str, name: str, args: Any) -> tuple[str, dict[str, Any]]:
    """Map a host tool call to `(cc_tool_name, cc_tool_input)`."""
    name = str(name or "")
    if host == "codex":
        cc_input = dict(args) if isinstance(args, dict) else {}
        return "Bash", cc_input
    if host == "opencode":
        cc_name = OPENCODE_TOOLS.get(name) or canonical_mcp_name(name) or name
        cc_input = _rename_args(args, OPENCODE_ARGS)
        return cc_name, cc_input
    if host == "agy":
        raw = args if isinstance(args, dict) else {}
        server = str(raw.get("ServerName") or raw.get("server_name") or "").strip()
        tool = str(raw.get("ToolName") or raw.get("tool_name") or "").strip()
        if name in AGY_TOOLS:
            cc_input = _rename_args(raw, AGY_ARGS)
            if AGY_TOOLS[name] == "Read":
                _agy_line_window(cc_input)
            return AGY_TOOLS[name], cc_input
        if server and tool:
            inner = raw.get("Arguments") or raw.get("arguments") or raw
            return f"mcp__{server}__{tool}", dict(inner) if isinstance(inner, dict) else {}
        return canonical_mcp_name(name) or name, dict(raw)
    return name, dict(args) if isinstance(args, dict) else {}
