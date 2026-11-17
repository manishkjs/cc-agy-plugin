"""Tests for the shared cross-host hook adapter (`scripts/hook_adapter.py`)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ha = _load("hook_adapter")
tm = _load("hook_toolmap")


@pytest.fixture()
def base_env(tmp_path: Path) -> dict[str, str]:
    """A hermetic env: fake HOME with no host configs, private data dir."""
    home = tmp_path / "home"
    home.mkdir()
    return {"HOME": str(home), "PATH": "/usr/bin:/bin", "CC_AGY_DATA_DIR": str(tmp_path / "data")}


class _Recorder:
    """Stand-in for `run_script` that records the CC payload and returns canned output."""

    def __init__(self, out: dict[str, Any] | Exception) -> None:
        self.out = out
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def __call__(self, script: Path, cc: dict[str, Any], env: dict[str, str], timeout: int) -> dict[str, Any]:
        self.calls.append((script.name, dict(cc), dict(env)))
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


def _deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _ctx(event: str, text: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


# ------------------------------------------------------------- golden normalize

GOLDEN: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = [
    (
        "opencode",
        "PreToolUse",
        {"tool": "edit", "sessionID": "s1", "callID": "c1", "directory": "/w", "args": {"filePath": "/w/a.py", "oldString": "a", "newString": "b"}},
        {"session_id": "s1", "cwd": "/w", "tool_name": "Edit", "tool_input": {"file_path": "/w/a.py", "old_string": "a", "new_string": "b"}},
    ),
    (
        "opencode",
        "PreToolUse",
        {"tool": "vertex_delegate_start", "sessionID": "s1", "directory": "/w", "args": {"prompt": "do x"}},
        {"tool_name": "mcp__vertex__delegate_start", "tool_input": {"prompt": "do x"}},
    ),
    (
        "opencode",
        "PostToolUse",
        {"tool": "vertex_delegate_result", "sessionID": "s1", "directory": "/w", "args": {}, "output": {"content": [{"type": "text", "text": "R1"}]}},
        {"tool_name": "mcp__vertex__delegate_result", "tool_response": "R1"},
    ),
    (
        "opencode",
        "PostToolUse",
        {"tool": "bash", "sessionID": "s1", "directory": "/w", "args": {"command": "ls"}, "output": {"title": "t", "output": "out", "metadata": {}}},
        {"tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_response": "out"},
    ),
    (
        "opencode",
        "UserPromptSubmit",
        {"sessionID": "s1", "directory": "/w", "parts": [{"type": "text", "text": "hello"}, {"type": "file", "url": "x"}]},
        {"session_id": "s1", "prompt": "hello"},
    ),
    ("opencode", "SessionStart", {"sessionID": "s1", "directory": "/w"}, {"session_id": "s1", "cwd": "/w", "source": "startup"}),
    ("opencode", "Stop", {"sessionID": "s1", "directory": "/w"}, {"session_id": "s1", "stop_hook_active": False}),
    (
        "codex",
        "PreToolUse",
        {"session_id": "x", "cwd": "/w", "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        {"session_id": "x", "cwd": "/w", "tool_name": "Bash", "tool_input": {"command": "rm -rf /"}, "hook_event_name": "PreToolUse"},
    ),
    ("codex", "Stop", {"session_id": "x", "cwd": "/w", "stop_hook_active": True}, {"stop_hook_active": True}),
    ("codex", "UserPromptSubmit", {"session_id": "x", "cwd": "/w", "prompt": "p"}, {"prompt": "p"}),
    (
        "agy",
        "PreToolUse",
        {"conversationId": "cv", "cwd": "/w", "transcriptPath": "/t", "toolCall": {"id": "t1", "name": "run_command", "args": {"CommandLine": "ls", "Cwd": "/w"}}},
        {"session_id": "cv", "cwd": "/w", "transcript_path": "/t", "tool_name": "Bash", "tool_input": {"command": "ls", "cwd": "/w"}, "tool_use_id": "t1"},
    ),
    (
        "agy",
        "PreToolUse",
        {"conversationId": "cv", "cwd": "/w", "toolCall": {"name": "view_file", "args": {"AbsolutePath": "/w/a", "StartLine": 10, "EndLine": 19}}},
        {"tool_name": "Read", "tool_input": {"file_path": "/w/a", "offset": 10, "limit": 10}},
    ),
    (
        "agy",
        "PreToolUse",
        {"conversationId": "cv", "cwd": "/w", "toolCall": {"name": "mcp_agy_delegate_start", "args": {"prompt": "p"}}},
        {"tool_name": "mcp__agy__delegate_start", "tool_input": {"prompt": "p"}},
    ),
    (
        "agy",
        "PostToolUse",
        {"conversationId": "cv", "cwd": "/w", "toolCall": {"name": "grep_search", "args": {"Query": "x", "SearchPath": "/w"}}, "result": "ok"},
        {"tool_name": "Grep", "tool_input": {"pattern": "x", "path": "/w"}, "tool_response": "ok"},
    ),
    ("agy", "UserPromptSubmit", {"conversationId": "cv", "cwd": "/w", "lastUserInput": "fix it"}, {"session_id": "cv", "prompt": "fix it"}),
    ("agy", "Stop", {"conversationId": "cv", "cwd": "/w", "continuationActive": True}, {"stop_hook_active": True}),
]


@pytest.mark.parametrize(("host", "event", "native", "expected"), GOLDEN)
def test_normalize_golden(host: str, event: str, native: dict[str, Any], expected: dict[str, Any]) -> None:
    cc = ha.normalize(host, event, native)
    assert cc["hook_event_name"] == event
    for key, val in expected.items():
        assert cc[key] == val, key


def test_codex_tool_name_is_always_bash() -> None:
    assert ha.normalize("codex", "PreToolUse", {"tool_name": "shell", "tool_input": {"command": "ls"}})["tool_name"] == "Bash"


@pytest.mark.parametrize(
    ("host", "name", "expected"),
    [
        ("opencode", "read", "Read"),
        ("opencode", "glob", "Glob"),
        ("opencode", "my-srv_delegate_apply", "mcp__my-srv__delegate_apply"),
        ("opencode", "github_create_issue", "github_create_issue"),
        ("agy", "write_to_file", "Write"),
        ("agy", "multi_replace_file_content", "MultiEdit"),
        ("agy", "find_by_name", "Glob"),
        ("agy", "mcp__vertex__delegate_continue", "mcp__vertex__delegate_continue"),
    ],
)
def test_toolmap_names(host: str, name: str, expected: str) -> None:
    assert tm.map_tool(host, name, {})[0] == expected


def test_jetski_ide_names() -> None:
    assert tm.map_tool("agy", "edit_file", {"Path": "/a", "Target": "x", "Replacement": "y"}) == (
        "Edit",
        {"file_path": "/a", "old_string": "x", "new_string": "y"},
    )
    assert tm.map_tool("agy", "read_file", {"Path": "/a", "StartLine": 1, "EndLine": 5}) == ("Read", {"file_path": "/a", "offset": 1, "limit": 5})


def test_agy_mcp_server_tool_args() -> None:
    name, args = tm.map_tool("agy", "call_mcp_tool", {"ServerName": "agy", "ToolName": "delegate_start", "Arguments": {"prompt": "p"}})
    assert (name, args) == ("mcp__agy__delegate_start", {"prompt": "p"})


# --------------------------------------------------------- output translation


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("opencode", {"action": "deny", "reason": "nope", "additionalContext": None, "block": True}),
        ("codex", _deny("nope")),
        ("agy", {"decision": "deny", "reason": "nope"}),
    ],
)
def test_pretool_deny(monkeypatch, base_env, host: str, expected: dict[str, Any]) -> None:
    monkeypatch.setattr(ha, "run_script", _Recorder(_deny("nope")))
    payload = {
        "opencode": {"tool": "bash", "args": {"command": "x"}},
        "codex": {"tool_name": "Bash", "tool_input": {"command": "x"}},
        "agy": {"toolCall": {"name": "run_command", "args": {"CommandLine": "x"}}},
    }[host]
    assert ha.handle(host, "PreToolUse", payload, base_env) == expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("opencode", {"action": "allow", "reason": None, "additionalContext": "slice it", "block": False}),
        ("codex", None),  # Codex rejects PreToolUse additionalContext.
        ("agy", {}),
    ],
)
def test_pretool_context_only(monkeypatch, base_env, host: str, expected: Any) -> None:
    monkeypatch.setattr(ha, "run_script", _Recorder(_ctx("PreToolUse", "slice it")))
    payload = {"tool": "bash", "tool_name": "Bash", "toolCall": {"name": "run_command", "args": {}}, "args": {}}
    assert ha.handle(host, "PreToolUse", payload, base_env) == expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("opencode", {"action": "allow", "reason": None, "additionalContext": "CTX", "block": False}),
        ("codex", {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "CTX"}}),
        ("agy", {"injectSteps": [{"systemMessage": {"systemMessage": "CTX"}}]}),
    ],
)
def test_session_start_context(monkeypatch, base_env, host: str, expected: dict[str, Any]) -> None:
    monkeypatch.setattr(ha, "run_script", _Recorder(_ctx("SessionStart", "CTX")))
    assert ha.handle(host, "SessionStart", {"cwd": "/w"}, base_env) == expected


def test_post_tool_context_per_host(monkeypatch, base_env) -> None:
    monkeypatch.setattr(ha, "run_script", _Recorder(_ctx("PostToolUse", "ledger")))
    oc = ha.handle("opencode", "PostToolUse", {"tool": "vertex_delegate_result", "output": {"output": "r"}}, base_env)
    assert oc["additionalContext"] == "ledger"
    agy = ha.handle("agy", "PostToolUse", {"toolCall": {"name": "mcp_vertex_delegate_result", "args": {}}}, base_env)
    assert agy == {}  # agy PostToolUse cannot inject context.


def test_post_tool_matcher_skips_non_plugin_tools(monkeypatch, base_env) -> None:
    rec = _Recorder(_ctx("PostToolUse", "ledger"))
    monkeypatch.setattr(ha, "run_script", rec)
    assert ha.handle("codex", "PostToolUse", {"tool_name": "Bash", "tool_input": {"command": "ls"}}, base_env) is None
    assert rec.calls == []


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("opencode", {"action": "allow", "reason": "dirty", "additionalContext": None, "block": True}),
        ("codex", {"decision": "block", "reason": "dirty"}),
        ("agy", {"decision": "block", "reason": "dirty"}),
    ],
)
def test_stop_block(monkeypatch, base_env, host: str, expected: dict[str, Any]) -> None:
    monkeypatch.setattr(ha, "run_script", _Recorder({"decision": "block", "reason": "dirty"}))
    assert ha.handle(host, "Stop", {"cwd": "/w"}, base_env) == expected


@pytest.mark.parametrize(
    ("host", "payload", "expected"),
    [
        ("codex", {"stop_hook_active": True}, None),
        ("agy", {"continuationActive": True}, {}),
        ("opencode", {"stop_hook_active": True}, {"action": "allow", "reason": None, "additionalContext": None, "block": False}),
    ],
)
def test_stop_reentry_never_blocks_twice(monkeypatch, base_env, host, payload, expected) -> None:
    monkeypatch.setattr(ha, "run_script", _Recorder({"decision": "block", "reason": "dirty"}))
    assert ha.handle(host, "Stop", payload, base_env) == expected


# --------------------------------------------------------------- fail-safe


def test_script_crash_fails_open(monkeypatch, base_env) -> None:
    monkeypatch.setattr(ha, "run_script", _Recorder(ha.HookScriptError("boom")))
    assert ha.handle("agy", "PreToolUse", {"toolCall": {"name": "run_command", "args": {}}}, base_env) == {}
    assert ha.handle("codex", "Stop", {}, base_env) is None


@pytest.mark.parametrize("host", ["opencode", "codex", "agy"])
def test_strict_mode_crash_denies_pretool(monkeypatch, base_env, host: str) -> None:
    monkeypatch.setattr(ha, "run_script", _Recorder(ha.HookScriptError("boom")))
    env = dict(base_env, CC_AGY_STRICT_MODE="true")
    payload = {"tool": "bash", "tool_name": "Bash", "toolCall": {"name": "run_command", "args": {}}}
    out = ha.handle(host, "PreToolUse", payload, env)
    text = json.dumps(out)
    assert "deny" in text and "strict delegation mode" in text and "boom" in text


def test_strict_mode_from_codex_config(monkeypatch, base_env, tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[mcp_servers.other.env]\nCC_AGY_STRICT_MODE = "false"\n\n'
        "[mcp_servers.agy]\ncommand = \"sh\"\n\n[mcp_servers.agy.env]\n"
        'CC_AGY_USERCFG_STRICT_MODE = "true"\nCC_AGY_USERCFG_COST_PROFILE = \'economy\'\nPATH = "/evil"\n',
        encoding="utf-8",
    )
    rec = _Recorder(ha.HookScriptError("boom"))
    monkeypatch.setattr(ha, "run_script", rec)
    out = ha.handle("codex", "PreToolUse", {"tool_name": "Bash"}, dict(base_env, CC_AGY_HOOK_CODEX_CONFIG=str(cfg)))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    env = rec.calls[0][2]
    assert env["CC_AGY_USERCFG_COST_PROFILE"] == "economy"
    assert env["PATH"] == "/usr/bin:/bin"  # Non-plugin keys are never imported.


def test_parse_codex_inline_env() -> None:
    text = '[mcp_servers."agy"]\ncommand = "sh"\nenv = { CC_AGY_X = "1", "CC_AGY_Y" = "two" }\n'
    assert ha.parse_codex_env(text) == {"CC_AGY_X": "1", "CC_AGY_Y": "two"}


def test_agy_env_does_not_override_and_skips_placeholders(base_env, tmp_path: Path) -> None:
    cfg = tmp_path / "mcp_config.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"agy": {"env": {"CC_AGY_USERCFG_MAX_DAILY_CENTS": "50", "CC_AGY_USERCFG_AUTH_MODE": "${user_config.auth_mode}", "CC_AGY_COST_PROFILE": "quality"}}}}),
        encoding="utf-8",
    )
    env = ha.build_env("agy", dict(base_env, CC_AGY_HOOK_AGY_MCP_CONFIG=str(cfg), CC_AGY_COST_PROFILE="economy", PLUGIN_DATA="/pd"))
    assert env["CC_AGY_USERCFG_MAX_DAILY_CENTS"] == "50"
    assert "CC_AGY_USERCFG_AUTH_MODE" not in env
    assert env["CC_AGY_COST_PROFILE"] == "economy"
    assert env["CC_AGY_DATA_DIR"] == base_env["CC_AGY_DATA_DIR"]  # explicit value wins over PLUGIN_DATA


def test_opencode_shim_env_merges(monkeypatch, base_env) -> None:
    rec = _Recorder({})
    monkeypatch.setattr(ha, "run_script", rec)
    ha.handle("opencode", "SessionStart", {"env": {"CC_AGY_USERCFG_STRICT_MODE": "1", "LD_PRELOAD": "x"}}, base_env)
    env = rec.calls[0][2]
    assert env["CC_AGY_USERCFG_STRICT_MODE"] == "1" and "LD_PRELOAD" not in env


# ------------------------------------------------------------------ dedupe


def test_agy_preinvocation_dedupe(monkeypatch, base_env) -> None:
    rec = _Recorder(_ctx("UserPromptSubmit", "hint"))
    monkeypatch.setattr(ha, "run_script", rec)
    p1 = {"conversationId": "cv", "cwd": "/w", "lastUserInput": "fix the bug"}
    first = ha.handle("agy", "PreInvocation", p1, base_env)
    assert first == {"injectSteps": [{"systemMessage": {"systemMessage": "hint"}}]}
    assert ha.handle("agy", "UserPromptSubmit", p1, base_env) == {}  # same prompt, second model call
    ha.handle("agy", "UserPromptSubmit", dict(p1, conversationId="other"), base_env)
    ha.handle("agy", "UserPromptSubmit", dict(p1, lastUserInput="next"), base_env)
    assert ha.handle("agy", "UserPromptSubmit", dict(p1, lastUserInput=""), base_env) == {}
    assert [c[1]["prompt"] for c in rec.calls] == ["fix the bug", "fix the bug", "next"]
    assert (Path(base_env["CC_AGY_DATA_DIR"]) / "hook_adapter" / "agy_prompts.json").is_file()


# --------------------------------------------------------------- redaction


def test_secret_redaction(monkeypatch, base_env) -> None:
    key = "AIzaSyD" + "x" * 30
    env = dict(base_env, CC_AGY_USERCFG_VERTEX_BASE_URL_TOKEN="tok-abcdefghijkl")
    monkeypatch.setattr(ha, "run_script", _Recorder(_deny(f"leak {key} and tok-abcdefghijkl and Bearer abc.def.ghi")))
    out = json.dumps(ha.handle("agy", "PreToolUse", {"toolCall": {"name": "run_command", "args": {}}}, env))
    assert key not in out and "tok-abcdefghijkl" not in out and "abc.def.ghi" not in out
    assert out.count("[REDACTED]") == 3


def test_crash_message_redacted(monkeypatch, base_env, capsys) -> None:
    env = dict(base_env, CC_AGY_USERCFG_GEMINI_API_KEY="supersecretvalue123")
    monkeypatch.setattr(ha, "run_script", _Recorder(ha.HookScriptError("bad supersecretvalue123")))
    ha.handle("codex", "Stop", {}, env)
    assert "supersecretvalue123" not in capsys.readouterr().err


# ----------------------------------------------------------- spec / end to end


def test_hook_spec_matches_hooks_json() -> None:
    hooks = json.loads((ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    for event in ha.EVENTS:
        script, timeout, matcher = ha.load_hook_spec(event)
        entry = hooks[event][0]
        assert script.name in entry["hooks"][0]["command"] and script.is_file()
        assert timeout == entry["hooks"][0]["timeout"]
        assert matcher == entry.get("matcher")


def test_exit2_maps_to_deny(monkeypatch, base_env) -> None:
    monkeypatch.setattr(ha, "run_script", _Recorder({"_exit2": "blocked via stderr"}))
    out = ha.handle("codex", "PreToolUse", {"tool_name": "Bash"}, base_env)
    assert out == _deny("blocked via stderr")


def _cli(host: str, event: str, payload: Any, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hook_adapter.py"), "--host", host, "--event", event],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_cli_end_to_end_real_scripts(base_env, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    oc = _cli("opencode", "Stop", {"sessionID": "s", "directory": str(ws)}, base_env)
    assert oc.returncode == 0
    assert json.loads(oc.stdout)["action"] == "allow"
    codex = _cli("codex", "PostToolUse", {"tool_name": "Bash", "cwd": str(ws)}, base_env)
    assert codex.returncode == 0 and codex.stdout == ""
    bad = _cli("agy", "PreToolUse", "{not json", base_env)
    assert bad.returncode == 0 and json.loads(bad.stdout) == {}
    pre = _cli("agy", "PreToolUse", {"conversationId": "c", "cwd": str(ws), "toolCall": {"name": "run_command", "args": {"CommandLine": "ls"}}}, base_env)
    assert pre.returncode == 0 and "decision" not in json.loads(pre.stdout)
