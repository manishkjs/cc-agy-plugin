"""Wire-compatibility tests for `agy acp --stdio` against agy's own ACP server behaviour.

Each case mirrors code in google3/third_party/jetski/cli/acp/:
- messages/usage.go: PromptResult usage is camelCase (cachedReadTokens, thoughtTokens, ...).
- server/session_handlers.go: session/load replays stored steps as session/update before replying.
- server/updates.go + server/prompt.go: a fatal step error is sent as "\\n⚠ <msg>\\n" and the
  turn still ends with stopReason "end_turn".
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import zipfile

from server.agy_acp import run_agy_acp_turn
from server.agy_cli import convert_agy_usage_to_vertex

REPO_ROOT = Path(__file__).resolve().parent.parent

_FIXTURE = r'''#!/usr/bin/env python3
import json, sys
MODE = "__MODE__"
def send(o):
    print(json.dumps(o), flush=True)
def update(kind, **kw):
    send({"jsonrpc": "2.0", "method": "session/update",
          "params": {"sessionId": "sess-prev", "update": {"sessionUpdate": kind, **kw}}})
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    msg = json.loads(raw)
    mid, method = msg.get("id"), msg.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1}})
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "sess-new"}})
    elif method == "session/load":
        update("agent_message_chunk", content={"type": "text", "text": "OLD TURN-1 TEXT. "})
        update("tool_call", title="edit_file", rawInput={"path": "a.py"})
        update("tool_call", title="run_command", rawInput={"command": "pytest -q"})
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
    elif method in ("session/set_model", "session/set_mode"):
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
    elif method == "session/prompt":
        if MODE == "replay":
            update("tool_call", title="edit_file", rawInput={"path": "b.py"})
            update("agent_message_chunk", content={"type": "text", "text": "NEW TURN-2 TEXT"})
            send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn",
                  "usage": {"inputTokens": 100, "outputTokens": 20, "cachedReadTokens": 900,
                            "cachedWriteTokens": 50, "thoughtTokens": 5, "totalTokens": 1070}}})
        else:
            update("agent_message_chunk",
                   content={"type": "text", "text": "\n\u26a0 Agent execution terminated due to error.\n"})
            send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn",
                  "usage": {"inputTokens": 30, "outputTokens": 0, "totalTokens": 30}}})
'''


def _fake_agy(tmp_path: Path, mode: str) -> str:
    script = _FIXTURE.replace("__MODE__", mode)
    if os.name == "nt":
        from tests.test_ftux_and_decision_guide import _write_windows_console_launcher

        return str(_write_windows_console_launcher(script, tmp_path / f"fake_agy_{mode}.exe"))
    path = tmp_path / f"fake_agy_{mode}"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _turn(agy_bin: str, cwd: Path, **kwargs):
    return run_agy_acp_turn(
        agy_bin=agy_bin,
        prompt_text="continue",
        cwd=str(cwd),
        model=None,
        effort=None,
        read_only=False,
        skip_permissions=True,
        use_agy_sandbox=False,
        timeout_seconds=15.0,
        **kwargs,
    )


def test_acp_camelcase_usage_keeps_cached_cache_write_and_thought_tokens() -> None:
    usage = convert_agy_usage_to_vertex(
        {
            "inputTokens": 100,
            "outputTokens": 20,
            "cachedReadTokens": 900,
            "cachedWriteTokens": 50,
            "thoughtTokens": 5,
            "totalTokens": 1070,
        }
    )
    assert usage == {
        "prompt_token_count": 1050,
        "cached_content_token_count": 900,
        "candidates_token_count": 15,
        "thoughts_token_count": 5,
        "total_token_count": 1070,
    }


def test_stream_json_snake_case_usage_is_unchanged() -> None:
    usage = convert_agy_usage_to_vertex(
        {"input_tokens": 100, "cache_read_tokens": 50, "output_tokens": 40, "thinking_tokens": 10}
    )
    assert usage == {
        "prompt_token_count": 150,
        "cached_content_token_count": 50,
        "candidates_token_count": 30,
        "thoughts_token_count": 10,
        "total_token_count": 190,
    }


def test_session_load_history_replay_is_not_counted_in_the_resumed_turn(tmp_path: Path) -> None:
    res = _turn(_fake_agy(tmp_path, "replay"), tmp_path, conversation_id="sess-prev", max_tool_calls=2)
    assert res["ok"] is True
    assert res["text"] == "NEW TURN-2 TEXT"
    assert [t["name"] for t in res["tool_calls"]] == ["edit_file"]
    assert res["stop_reason"] == "STOP"
    assert res["usage"]["cached_content_token_count"] == 900


def test_warning_banner_without_status_code_marks_the_turn_failed(tmp_path: Path) -> None:
    res = _turn(_fake_agy(tmp_path, "error"), tmp_path, conversation_id=None)
    assert res["ok"] is False
    assert res["stop_reason"] == "ERROR"
    assert res["error"] == "⚠ Agent execution terminated due to error."


def test_wheel_bundles_every_guide_the_readme_links_to(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("cc_agy_build_backend", REPO_ROOT / "_build_backend.py")
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    wheel_name = backend.build_wheel(str(tmp_path))
    with zipfile.ZipFile(tmp_path / wheel_name) as zf:
        names = set(zf.namelist())
        readme = zf.read("server/_bundled_assets/README.md").decode("utf-8")
    links = sorted(set(re.findall(r"\]\((guides/[^)#\s]+)", readme)))
    assert links, "README no longer links to guides/"
    assert [ln for ln in links if f"server/_bundled_assets/{ln}" not in names] == []
