"""Tests for MDM-compliant ACP permission callbacks and hybrid MCP elicitation bridge."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

import pytest

from server.agy_acp import run_agy_acp_turn
from server.mcp_stdio import MCPServer


def _make_acp_permission_fixture(
    path: Path,
    *,
    tool_call_payload: dict[str, Any],
    argv_log: Path | None = None,
) -> Path:
    """Create a fake `agy acp --stdio` binary that emits `session/request_permission` mid-turn."""
    script = f"""#!/usr/bin/env python3
import json, pathlib, sys
argv_log = {str(argv_log)!r}
if argv_log != 'None':
    pathlib.Path(argv_log).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
tc_payload = {json.dumps(tool_call_payload)}
prompt_id = None
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    msg = json.loads(raw)
    mid = msg.get("id")
    method = msg.get("method")
    if method == "initialize":
        print(json.dumps({{"jsonrpc": "2.0", "id": mid, "result": {{
            "protocolVersion": 1,
            "agentInfo": {{"name": "Fake ACP", "version": "1.5.10"}},
            "agentCapabilities": {{"loadSession": True}},
        }}}}), flush=True)
    elif method == "session/new":
        print(json.dumps({{"jsonrpc": "2.0", "id": mid, "result": {{
            "sessionId": "acp-elicit-sess",
            "models": {{"currentModelId": "gemini-4-hydrogen-medium", "availableModels": [
                {{"modelId": "gemini-4-hydrogen-medium", "name": "Hydrogen"}}
            ]}},
            "modes": {{"currentModeId": "default", "availableModes": [{{"id": "default"}}, {{"id": "accept-edits"}}]}},
        }}}}), flush=True)
    elif method in ("session/set_model", "session/set_mode"):
        print(json.dumps({{"jsonrpc": "2.0", "id": mid, "result": {{}}}}), flush=True)
    elif method == "session/prompt":
        prompt_id = mid
        print(json.dumps({{"jsonrpc": "2.0", "id": 99, "method": "session/request_permission", "params": {{
            "sessionId": "acp-elicit-sess",
            "toolCall": tc_payload,
            "options": [
                {{"optionId": "allow_once", "kind": "allow_once"}},
                {{"optionId": "reject_once", "kind": "reject_once"}},
            ],
        }}}}), flush=True)
    elif mid == 99 and "result" in msg:
        outcome = (msg["result"] or {{}}).get("outcome") or {{}}
        chosen = outcome.get("optionId")
        if chosen == "allow_once":
            print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{
                "update": {{"sessionUpdate": "agent_message_chunk", "content": {{"type": "text", "text": "Approved inline and finished."}}}}
            }}}}), flush=True)
            print(json.dumps({{"jsonrpc": "2.0", "id": prompt_id, "result": {{
                "stopReason": "end_turn",
                "usage": {{"input_tokens": 50, "output_tokens": 12}},
            }}}}), flush=True)
        else:
            print(json.dumps({{"jsonrpc": "2.0", "id": prompt_id, "result": {{
                "stopReason": "cancelled",
                "usage": {{"input_tokens": 50, "output_tokens": 0}},
            }}}}), flush=True)
"""
    if os.name == "nt":
        from tests.test_ftux_and_decision_guide import _write_windows_console_launcher

        return _write_windows_console_launcher(script, path.with_suffix(".exe"))
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_acp_omits_dangerously_skip_permissions_by_default_and_approves_via_elicitation(
    tmp_path: Path,
) -> None:
    argv_log = tmp_path / "argv.json"
    fake_agy = _make_acp_permission_fixture(
        tmp_path / "fake_agy",
        tool_call_payload={"name": "run_command", "kind": "execute", "rawInput": {"command": "pytest -q"}},
        argv_log=argv_log,
    )
    prompted: list[tuple[dict[str, Any], str]] = []

    def _approve(tc: dict[str, Any], reason: str) -> bool:
        prompted.append((dict(tc), reason))
        return True

    res = run_agy_acp_turn(
        agy_bin=str(fake_agy),
        prompt_text="Run tests",
        cwd=str(tmp_path),
        model="gemini-4-hydrogen",
        effort="medium",
        read_only=False,
        skip_permissions=False,
        use_agy_sandbox=True,
        conversation_id=None,
        timeout_seconds=10.0,
        permission_prompter=_approve,
    )
    assert res["ok"] is True
    assert res["text"] == "Approved inline and finished."
    assert res["denied_actions"] == []
    assert len(prompted) == 1
    recorded_argv = json.loads(argv_log.read_text(encoding="utf-8"))
    assert "--dangerously-skip-permissions" not in recorded_argv


def test_acp_elicitation_decline_marks_user_declined_and_pauses(tmp_path: Path) -> None:
    fake_agy = _make_acp_permission_fixture(
        tmp_path / "fake_agy_decline",
        tool_call_payload={"name": "run_command", "kind": "execute", "rawInput": {"command": "pytest -q"}},
    )
    res = run_agy_acp_turn(
        agy_bin=str(fake_agy),
        prompt_text="Run tests",
        cwd=str(tmp_path),
        model="gemini-4-hydrogen",
        effort="medium",
        read_only=False,
        skip_permissions=False,
        use_agy_sandbox=True,
        conversation_id=None,
        timeout_seconds=10.0,
        permission_prompter=lambda _tc, _reason: False,
    )
    assert res["ok"] is True
    assert res["stop_reason"] == "DENIED_ACTIONS"
    assert len(res["denied_actions"]) == 1
    assert res["denied_actions"][0].get("user_declined") is True


def test_acp_hard_security_deny_never_calls_permission_prompter(tmp_path: Path) -> None:
    fake_agy = _make_acp_permission_fixture(
        tmp_path / "fake_agy_secret",
        tool_call_payload={"name": "read_file", "kind": "read", "rawInput": {"path": ".env"}},
    )
    called = False

    def _should_not_run(_tc: dict[str, Any], _reason: str) -> bool:
        nonlocal called
        called = True
        return True

    res = run_agy_acp_turn(
        agy_bin=str(fake_agy),
        prompt_text="Read env",
        cwd=str(tmp_path),
        model="gemini-4-hydrogen",
        effort="medium",
        read_only=False,
        skip_permissions=True,
        use_agy_sandbox=False,
        conversation_id=None,
        timeout_seconds=10.0,
        permission_prompter=_should_not_run,
    )
    assert called is False
    assert res["stop_reason"] == "DENIED_ACTIONS"
    assert res["denied_actions"][0].get("hard_deny") is True


def test_mcp_server_elicitation_roundtrip_and_fallback_when_unsupported() -> None:
    srv = MCPServer()
    sent: list[dict[str, Any]] = []
    srv.notify = sent.append

    # Without capabilities.elicitation, returns None (falls back to needs_input).
    srv.handle_Message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
        }
    )
    assert srv.supports_elicitation() is False
    assert srv.request_permission_elicitation({"name": "run_command"}, "sandbox gate") is None

    # With capabilities.elicitation, emits elicitation/create and resolves on client reply.
    srv.handle_Message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {"elicitation": {}}},
        }
    )
    assert srv.supports_elicitation() is True

    result_box: dict[str, Any] = {}

    def _run_elicit() -> None:
        result_box["approved"] = srv.request_permission_elicitation(
            {"name": "run_command", "rawInput": {"command": "pytest -q"}},
            "MDM sandbox approval required",
            timeout_seconds=5.0,
        )

    t = threading.Thread(target=_run_elicit)
    t.start()
    while not sent:
        threading.Event().wait(0.01)
    req = sent[0]
    assert req["method"] == "elicitation/create"
    assert "pytest -q" in req["params"]["message"]
    assert srv.resolve_client_response(
        {"jsonrpc": "2.0", "id": req["id"], "result": {"action": "accept", "content": {"approve": True}}}
    )
    t.join(timeout=2.0)
    assert result_box["approved"] is True
