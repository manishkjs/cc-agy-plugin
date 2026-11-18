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
_REAL_INPROC = ha.run_pre_tool_guard_inprocess


@pytest.fixture(autouse=True)
def _subprocess_path_only(monkeypatch):
    """Route PreToolUse through `run_script` so `_Recorder` sees it; real-guard tests undo this."""
    monkeypatch.setattr(ha, "run_pre_tool_guard_inprocess", lambda cc, env: None)


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


# ------------------------------------------------------------ round-10 fixes


@pytest.mark.parametrize("marker_env", [{"CC_AGY_IN_WORKER": "1"}, {}])
def test_worker_context_skips_guards_even_in_strict_mode(monkeypatch, base_env, tmp_path, marker_env) -> None:
    rec = _Recorder(ha.HookScriptError("boom"))
    monkeypatch.setattr(ha, "run_script", rec)
    env = dict(base_env, CC_AGY_STRICT_MODE="true", **marker_env)
    cwd = str(tmp_path / "ws") if marker_env else str(tmp_path / ".cc-agy-worktrees-ab12" / "job1")
    payload = {"conversationId": "c", "cwd": cwd, "toolCall": {"name": "run_command", "args": {"CommandLine": "ls"}}}
    assert ha.handle("agy", "PreToolUse", payload, env) == {}
    assert ha.handle("codex", "Stop", {"cwd": cwd}, env) is None
    assert rec.calls == []


def test_strict_without_worker_marker_still_denies(monkeypatch, base_env, tmp_path) -> None:
    monkeypatch.setattr(ha, "run_script", _Recorder(ha.HookScriptError("boom")))
    env = dict(base_env, CC_AGY_STRICT_MODE="true")
    payload = {"conversationId": "c", "cwd": str(tmp_path), "toolCall": {"name": "run_command", "args": {}}}
    assert ha.handle("agy", "PreToolUse", payload, env)["decision"] == "deny"


def test_build_agy_subprocess_env_sets_worker_marker() -> None:
    sys.path.insert(0, str(ROOT))
    from server.agy_cli import WORKER_ENV_MARKER, build_agy_subprocess_env

    assert WORKER_ENV_MARKER == ha.WORKER_ENV_MARKER
    assert build_agy_subprocess_env({"PATH": "/usr/bin", "CC_AGY_IN_WORKER": "0"})[WORKER_ENV_MARKER] == "1"


_PLUGIN_FILE = str(ROOT / "server" / "receipts.py")


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("edit", {"filePath": _PLUGIN_FILE, "oldString": "a", "newString": "b"}),
        ("multiedit", {"filePath": _PLUGIN_FILE, "edits": [{"oldString": "a", "newString": "b"}]}),
        ("patch", {"patchText": f"*** Begin Patch\n*** Update File: {_PLUGIN_FILE}\n@@\n-a\n+b\n*** End Patch"}),
        ("apply_patch", {"patchText": f"*** Begin Patch\n*** Add File: {_PLUGIN_FILE}\n+x\n*** End Patch"}),
    ],
)
def test_opencode_edit_family_denied_in_strict_mode(monkeypatch, base_env, tmp_path, tool, args) -> None:
    monkeypatch.setattr(ha, "run_pre_tool_guard_inprocess", _REAL_INPROC)
    monkeypatch.setattr(ha, "run_script", _Recorder(AssertionError("guard must run in-process")))
    ws = tmp_path / "ws"
    ws.mkdir()
    env = dict(base_env, CC_AGY_STRICT_MODE="true")
    out = ha.handle("opencode", "PreToolUse", {"tool": tool, "sessionID": "s", "directory": str(ws), "args": args}, env)
    assert out["action"] == "deny" and out["block"] is True, out
    assert "Strict orchestrator mode" in out["reason"]


def test_apply_patch_target_extraction() -> None:
    body = "*** Begin Patch\n*** Delete File: a/b.py\n*** Update File: c.py\n*** End Patch"
    assert tm.map_tool("opencode", "apply_patch", {"patchText": body}) == ("Edit", {"patchText": body, "file_path": "a/b.py"})


def test_inprocess_falls_back_to_subprocess(monkeypatch, base_env) -> None:
    rec = _Recorder(_deny("from subprocess"))
    monkeypatch.setattr(ha, "run_script", rec)
    out = ha.handle("codex", "PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "ls"}}, base_env)
    assert out == _deny("from subprocess") and rec.calls[0][0] == "pre_tool_guard.py"


def test_system_message_folds_into_context(monkeypatch, base_env) -> None:
    out_cc = {"systemMessage": "self-check failed", **_ctx("SessionStart", "guide")}
    monkeypatch.setattr(ha, "run_script", _Recorder(out_cc))
    assert ha.handle("agy", "SessionStart", {}, base_env) == {
        "injectSteps": [{"systemMessage": {"systemMessage": "guide\n\nself-check failed"}}]
    }
    monkeypatch.setattr(ha, "run_script", _Recorder({"systemMessage": "only warn"}))
    assert ha.handle("opencode", "SessionStart", {}, base_env)["additionalContext"] == "only warn"


@pytest.mark.parametrize(
    ("host", "payload", "active"),
    [("codex", {}, False), ("agy", {}, False), ("codex", {"stop_hook_active": True}, True), ("agy", {"continuationActive": True}, True)],
)
def test_strict_stop_crash_blocks_once(monkeypatch, base_env, host, payload, active) -> None:
    monkeypatch.setattr(ha, "run_script", _Recorder(ha.HookScriptError("provenance_check.py timed out after 15s")))
    out = ha.handle(host, "Stop", payload, dict(base_env, CC_AGY_STRICT_MODE="1"))
    if active:
        assert not out
    else:
        assert out["decision"] == "block" and "internal error" in out["reason"]
    assert not ha.handle(host, "Stop", payload, base_env)  # non-strict crash fails open


def test_plugin_data_does_not_split_the_data_dir(base_env, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    server_env = {"HOME": base_env["HOME"], "PATH": base_env["PATH"], "PLUGIN_DATA": str(tmp_path / "pd")}
    hook_env = ha.build_env("agy", server_env)
    assert "CC_AGY_DATA_DIR" not in hook_env
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; "
        "from server.receipts import ReceiptStore; "
        "s = ReceiptStore(workspace_root=Path(sys.argv[2])); s.write_receipt if hasattr(s, 'write_receipt') else None; "
        "print(s.data_dir)"
    )
    dirs = [
        subprocess.run([sys.executable, "-c", probe, str(ROOT), str(ws)], env=e, cwd=str(ws), capture_output=True, text=True, check=True).stdout.strip()
        for e in (server_env, hook_env)
    ]
    assert dirs[0] == dirs[1] and dirs[0]
    assert ha.data_dir(hook_env) != Path(server_env["PLUGIN_DATA"])


def test_post_tool_matcher_covers_agy_server_tools() -> None:
    _, _, matcher = ha.load_hook_spec("PostToolUse")
    for name in ("mcp__agy__ask_gemini", "mcp__cc-agy-plugin__delegate_status", "mcp__vertex__x", "mcp__other__delegate_apply"):
        assert ha.matcher_allows(matcher, name), name
    assert not ha.matcher_allows(matcher, "mcp__github__create_issue")


def test_straitjacket_worker_env_sets_marker(monkeypatch, tmp_path) -> None:
    pytest.importorskip("ctx.sdk")
    import asyncio

    sys.path.insert(0, str(ROOT))
    from server import agy_runner

    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("CC_AGY_IN_WORKER", "0")
    tools, _ = agy_runner.build_straitjacket_tools(ws, timeout_seconds=60.0)
    run_cmd = next(fn for fn in tools if getattr(fn, "__name__", "") == "run_command")
    res = json.loads(asyncio.run(run_cmd('echo "marker=$CC_AGY_IN_WORKER"')))
    assert "marker=1" in json.dumps(res), res


def test_straitjacket_session_env_carries_marker_with_stub_sdk(monkeypatch, tmp_path) -> None:
    """Hermetic twin of the test above: stub `ctx.sdk` and capture the env handed to ToolSession."""
    import types
    from unittest import mock

    class _Captured(Exception):
        pass

    class _ToolSession:
        def __init__(self, *args: Any, env: dict[str, str] | None = None, **kwargs: Any) -> None:
            raise _Captured(env)

    ctx_pkg = types.ModuleType("ctx")
    ctx_cfg = mock.MagicMock(name="ctx.config")
    ctx_sdk = types.ModuleType("ctx.sdk")
    ctx_sdk.ToolSession = _ToolSession
    ctx_pkg.config, ctx_pkg.sdk = ctx_cfg, ctx_sdk
    monkeypatch.setitem(sys.modules, "ctx", ctx_pkg)
    monkeypatch.setitem(sys.modules, "ctx.config", ctx_cfg)
    monkeypatch.setitem(sys.modules, "ctx.sdk", ctx_sdk)
    monkeypatch.setenv("CC_AGY_IN_WORKER", "0")
    sys.path.insert(0, str(ROOT))
    from server import agy_runner

    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(_Captured) as captured:
        agy_runner.build_straitjacket_tools(ws, timeout_seconds=60.0)
    env = captured.value.args[0]
    assert env["CC_AGY_IN_WORKER"] == "1"
    assert env.get("CC_AGY_DATA_DIR", "") == ""  # the CC_AGY_* blanking still applies
