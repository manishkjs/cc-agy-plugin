"""OpenCode hook parity: `opencode-plugin/hooks.ts` maps OpenCode plugin hooks onto hooks.json events.

Each test runs a small Node script that imports the TypeScript plugin with type stripping and
drives its hook functions. A fake adapter (selected with `CC_AGY_HOOK_ADAPTER`) records what the
plugin sent and answers with a canned result, so these tests do not depend on the real adapter;
one integration test at the end runs `scripts/hook_adapter.py` itself.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOKS_URL = (REPO_ROOT / "opencode-plugin" / "hooks.ts").as_uri()
INTERNAL_URL = (REPO_ROOT / "opencode-plugin" / "internal.ts").as_uri()
ENTRY_URL = (REPO_ROOT / "opencode-plugin" / "index.ts").as_uri()
REAL_ADAPTER = REPO_ROOT / "scripts" / "hook_adapter.py"

FAKE_ADAPTER = textwrap.dedent(
    """
    import json, os, sys, time
    payload = json.loads(sys.stdin.read() or "{}")
    record = os.environ["FAKE_RECORD"]
    with open(record, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"argv": sys.argv[1:], "payload": payload,
                             "env": {k: v for k, v in os.environ.items() if k.startswith("CC_AGY_")}}) + "\\n")
    mode = os.environ.get("FAKE_MODE", "allow")
    if mode == "sleep":
        time.sleep(30)
    if mode == "crash":
        sys.exit(3)
    result = {"action": "allow", "reason": None, "additionalContext": None, "block": False}
    if mode == "deny":
        result.update(action="deny", reason="fake guard says no", block=True)
    elif mode == "context":
        result.update(additionalContext="ctx from " + sys.argv[-1])
    elif mode == "block":
        result.update(block=True, reason="diff has no provenance receipt")
    print(json.dumps(result))
    """
)


def _node_candidates() -> list[str]:
    found: list[str] = []
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = shutil.which("node", path=directory) if directory else None
        if candidate and str(Path(candidate).resolve()) not in {str(Path(f).resolve()) for f in found}:
            found.append(candidate)
    return found


def _node_ts_command() -> list[str] | None:
    """A `node` command that can import the plugin's TypeScript by stripping types, or None."""
    probe = f"await import({json.dumps(HOOKS_URL)}); console.log('ok')"
    for node in _node_candidates():
        for flags in ([], ["--experimental-strip-types"]):
            cmd = [node, "--no-warnings", *flags]
            try:
                proc = subprocess.run(
                    [*cmd, "--input-type=module", "-e", probe],
                    capture_output=True, text=True, timeout=60, check=False,
                    env={"PATH": os.environ.get("PATH", ""), "HOME": str(Path.cwd())},
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if proc.returncode == 0 and proc.stdout.strip().endswith("ok"):
                return cmd
    return None


NODE_CMD = _node_ts_command()
pytestmark = pytest.mark.skipif(
    NODE_CMD is None, reason="no node on PATH can import TypeScript (needs Node 22.6+ with type stripping)"
)


class Harness:
    """Runs scenario scripts against the plugin with the fake adapter."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.record = tmp_path / "record.jsonl"
        self.adapter = tmp_path / "fake_adapter.py"
        self.adapter.write_text(FAKE_ADAPTER, encoding="utf-8")
        (tmp_path / "home").mkdir()

    def env(self, **extra: str) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.tmp / "home"),
            "XDG_DATA_HOME": str(self.tmp / "data"),
            "CC_AGY_PYTHON_BIN": sys.executable,
            "CC_AGY_HOOK_ADAPTER": str(self.adapter),
            "FAKE_RECORD": str(self.record),
        }
        env.update(extra)
        return env

    def run(self, body: str, **extra: str) -> dict:
        script = (
            f"const hooksMod = await import({json.dumps(HOOKS_URL)});\n"
            f"const internal = await import({json.dumps(INTERNAL_URL)});\n"
            f"const entry = await import({json.dumps(ENTRY_URL)});\n"
            "const toasts = [];\n"
            "const client = {app: {log: async () => {}}, tui: {showToast: async (a) => { toasts.push(a.body.message); }}};\n"
            "const out = {};\n"
            "async function attempt(fn) { try { await fn(); return null; } catch (e) { return String(e.message); } }\n"
            f"{body}\n"
            "out.toasts = toasts;\n"
            "console.log(JSON.stringify(out));\n"
        )
        proc = subprocess.run(
            [*NODE_CMD, "--input-type=module", "-e", script],
            cwd=self.tmp, capture_output=True, text=True, timeout=120, check=False, env=self.env(**extra),
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def calls(self) -> list[dict]:
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


PLUGIN = "const hooks = await entry.default({directory: '/proj', client});\n"


def test_plugin_registers_every_hook_and_can_be_switched_off(harness: Harness) -> None:
    on = harness.run(PLUGIN + "out.keys = Object.keys(hooks);")
    off = harness.run(PLUGIN + "out.keys = Object.keys(hooks);", CC_AGY_OPENCODE_HOOKS="0")

    guard_keys = {"tool.execute.before", "tool.execute.after", "chat.message", "event",
                  "experimental.chat.system.transform"}
    assert guard_keys | {"auth", "config"} <= set(on["keys"])
    assert set(off["keys"]) == {"auth", "config"}


def test_timeouts_mirror_hooks_json(harness: Harness) -> None:
    out = harness.run("out.timeouts = hooksMod.HOOK_TIMEOUTS_MS;")
    hooks_json = json.loads((REPO_ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    expected = {event: groups[0]["hooks"][0]["timeout"] * 1000 for event, groups in hooks_json.items()}
    assert out["timeouts"] == expected


def test_prefilter_follows_hooks_json_matchers(harness: Harness) -> None:
    pre = ["read", "edit", "multiedit", "write", "patch", "apply_patch", "grep", "glob", "bash", "agy_delegate_start",
           "vertex_delegate_continue", "plugin_cc-agy-plugin_vertex_delegate_to_gemini",
           "webfetch", "task", "todowrite", "agy_ask_gemini", "github_create_issue"]
    post = ["agy_delegate_start", "agy_ask_gemini", "vertex_jobs", "other_delegate_apply", "cc-agy-plugin_delegate_status",
            "bash", "read", "github_create_issue", "other_delegate_list"]
    out = harness.run(
        f"out.pre = Object.fromEntries({json.dumps(pre)}.map((t) => [t, hooksMod.matchesPreToolUse(t)]));\n"
        f"out.post = Object.fromEntries({json.dumps(post)}.map((t) => [t, hooksMod.matchesPostToolUse(t)]));\n"
        + PLUGIN
        + "await hooks['tool.execute.before']({tool: 'webfetch', sessionID: 's', callID: 'c'}, {args: {}});\n"
        "await hooks['tool.execute.after']({tool: 'bash', sessionID: 's', callID: 'c', args: {}}, {title: '', output: '', metadata: {}});\n"
    )

    assert [t for t, hit in out["pre"].items() if hit] == pre[:12]
    assert [t for t, hit in out["post"].items() if hit] == post[:5]
    assert harness.calls() == [], "tools outside the hooks.json matchers must not spawn Python"


def test_deny_throws_the_guard_reason_with_the_opencode_payload(harness: Harness) -> None:
    out = harness.run(
        PLUGIN
        + "out.error = await attempt(() => hooks['tool.execute.before']("
        "{tool: 'read', sessionID: 'ses_1', callID: 'call_1'}, {args: {filePath: '/proj/a.py'}}));",
        FAKE_MODE="deny",
    )

    assert out["error"] == "fake guard says no"
    (call,) = harness.calls()
    assert call["argv"][-4:] == ["--host", "opencode", "--event", "PreToolUse"]
    assert call["payload"] == {"directory": "/proj", "tool": "read", "args": {"filePath": "/proj/a.py"},
                               "sessionID": "ses_1", "callID": "call_1"}


def test_after_handles_builtin_and_mcp_result_shapes(harness: Harness) -> None:
    harness.run(
        PLUGIN
        + "await hooks['tool.execute.after']({tool: 'agy_delegate_start', sessionID: 's', callID: 'c1', args: {prompt: 'p'}},"
        " {content: [{type: 'text', text: 'job_123 started'}, {type: 'image', data: 'AAAA'}]});\n"
        "await hooks['tool.execute.after']({tool: 'vertex_jobs', sessionID: 's', callID: 'c2', args: {}},"
        " {title: 'jobs', output: 'none running', metadata: {n: 0}});\n"
    )

    mcp, builtin = harness.calls()
    assert mcp["argv"][-1] == "PostToolUse" and builtin["argv"][-1] == "PostToolUse"
    assert mcp["payload"]["output"] == {"content": [{"type": "text", "text": "job_123 started"}], "isError": False}
    assert mcp["payload"]["args"] == {"prompt": "p"} and mcp["payload"]["callID"] == "c1"
    assert builtin["payload"]["output"] == {"title": "jobs", "output": "none running", "metadata": {"n": 0}}


def test_timeout_and_crash_fail_open_outside_strict_mode(harness: Harness) -> None:
    out = harness.run(
        PLUGIN
        + "const t0 = Date.now();\n"
        "out.error = await attempt(() => hooks['tool.execute.before']({tool: 'bash', sessionID: 's', callID: 'c'}, {args: {command: 'ls'}}));\n"
        "out.elapsed = Date.now() - t0;",
        FAKE_MODE="sleep", CC_AGY_HOOK_TIMEOUT_MS="400",
    )
    crashed = harness.run(
        PLUGIN
        + "out.error = await attempt(() => hooks['tool.execute.before']({tool: 'bash', sessionID: 's', callID: 'c'}, {args: {command: 'ls'}}));",
        FAKE_MODE="crash",
    )

    assert out["error"] is None and out["elapsed"] < 5000
    assert crashed["error"] is None


@pytest.mark.parametrize("mode,extra", [("crash", {}), ("sleep", {"CC_AGY_HOOK_TIMEOUT_MS": "400"})])
def test_strict_mode_denies_when_the_guard_cannot_run(harness: Harness, mode: str, extra: dict) -> None:
    out = harness.run(
        PLUGIN
        + "out.error = await attempt(() => hooks['tool.execute.before']({tool: 'edit', sessionID: 's', callID: 'c'}, {args: {}}));",
        FAKE_MODE=mode, CC_AGY_STRICT_MODE="true", **extra,
    )

    assert out["error"] and "strict delegation mode blocked edit" in out["error"]


def test_strict_mode_from_the_opencode_mcp_entry_reaches_the_adapter(harness: Harness) -> None:
    out = harness.run(
        PLUGIN
        + "const cfg = {mcp: {agy: {type: 'local', command: ['x'], environment: {\n"
        "  CC_AGY_USERCFG_STRICT_MODE: 'true', CC_AGY_USERCFG_COST_PROFILE: 'economy',\n"
        "  CC_AGY_USERCFG_MAX_DAILY_CENTS: '250', CC_AGY_USERCFG_AUTO_APPLY_ON_PASS: '${user_config.x}',\n"
        "  GEMINI_API_KEY: 'not-for-hooks'}}}};\n"
        "await hooks.config(cfg);\n"
        "out.agyKept = cfg.mcp.agy.command;\n"
        "out.error = await attempt(() => hooks['tool.execute.before']({tool: 'write', sessionID: 's', callID: 'c'}, {args: {}}));",
        FAKE_MODE="crash",
    )

    assert out["agyKept"] == ["x"]
    assert out["error"] and "strict delegation mode" in out["error"]
    env = harness.calls()[0]["env"]
    assert env["CC_AGY_USERCFG_COST_PROFILE"] == "economy" and env["CC_AGY_USERCFG_MAX_DAILY_CENTS"] == "250"
    assert "CC_AGY_USERCFG_AUTO_APPLY_ON_PASS" not in env


def test_stored_login_environment_reaches_the_adapter(harness: Harness) -> None:
    auth_dir = harness.tmp / "data" / "opencode"
    auth_dir.mkdir(parents=True)
    (auth_dir / "auth.json").write_text(json.dumps({"cc-agy": {
        "type": "api", "key": "k", "metadata": {"authMode": "vertex", "project": "p1", "location": "us"}}}))
    harness.run(PLUGIN + "await hooks['tool.execute.before']({tool: 'grep', sessionID: 's', callID: 'c'}, {args: {}});")

    assert harness.calls()[0]["env"]["CC_AGY_AUTH_MODE"] == "vertex"


def test_prompt_and_session_context_are_injected_into_the_system_prompt(harness: Harness) -> None:
    out = harness.run(
        PLUGIN
        + "await hooks.event({event: {type: 'session.created', properties: {info: {id: 'ses_9'}}}});\n"
        "await hooks['chat.message']({sessionID: 'ses_9', messageID: 'm1'}, {message: {}, parts: [\n"
        "  {type: 'text', text: 'refactor the parser'}, {type: 'text', text: 'ignored', synthetic: true},\n"
        "  {type: 'file', url: 'file:///x'}]});\n"
        "const first = {system: ['base']}; await hooks['experimental.chat.system.transform']({sessionID: 'ses_9', model: {}}, first);\n"
        "const second = {system: ['base']}; await hooks['experimental.chat.system.transform']({sessionID: 'ses_9', model: {}}, second);\n"
        "out.first = first.system; out.second = second.system;",
        FAKE_MODE="context",
    )

    start, prompt = harness.calls()
    assert start["argv"][-1] == "SessionStart" and start["payload"]["sessionID"] == "ses_9"
    assert prompt["argv"][-1] == "UserPromptSubmit" and prompt["payload"]["prompt"] == "refactor the parser"
    assert out["first"] == ["base", "ctx from SessionStart", "ctx from UserPromptSubmit"]
    assert out["second"] == ["base"], "buffered context is delivered once"


def test_stop_block_is_logged_and_toasted_but_cannot_block(harness: Harness) -> None:
    out = harness.run(
        PLUGIN
        + "out.error = await attempt(() => hooks.event({event: {type: 'session.idle', properties: {sessionID: 'ses_2'}}}));\n"
        "await hooks.event({event: {type: 'message.updated', properties: {}}});",
        FAKE_MODE="block",
    )

    assert out["error"] is None
    assert out["toasts"] == ["diff has no provenance receipt"]
    (call,) = harness.calls()
    assert call["argv"][-1] == "Stop" and call["payload"]["sessionID"] == "ses_2"
    assert call["payload"]["stop_hook_active"] is False


@pytest.mark.skipif(not REAL_ADAPTER.exists(), reason="scripts/hook_adapter.py is not present")
def test_real_adapter_denies_a_video_read_through_the_opencode_hook(harness: Harness) -> None:
    out = harness.run(
        PLUGIN
        + "out.denied = await attempt(() => hooks['tool.execute.before']({tool: 'read', sessionID: 's', callID: 'c'},"
        " {args: {filePath: '/proj/demo.mp4'}}));\n"
        "out.allowed = await attempt(() => hooks['tool.execute.before']({tool: 'read', sessionID: 's', callID: 'c'},"
        " {args: {filePath: '/proj/notes.txt'}}));",
        CC_AGY_HOOK_ADAPTER=str(REAL_ADAPTER), CC_AGY_DATA_DIR=str(harness.tmp / "ccdata"),
    )

    assert out["denied"] and "cannot inspect video file" in out["denied"]
    assert out["allowed"] is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX PATH shim test for macOS/Linux 3.9 vs 3.10+ selection")
def test_resolve_python_skips_python39_and_picks_python310_plus(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    py39 = bin_dir / "python3"
    py39.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    py39.chmod(0o755)
    py312 = bin_dir / "python3.12"
    py312.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    py312.chmod(0o755)

    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    script = (
        f"import {{ resolvePython }} from {json.dumps(INTERNAL_URL)};\n"
        f"const picked = resolvePython({{\n"
        f"  HOME: {json.dumps(str(empty_home))},\n"
        f"  XDG_DATA_HOME: {json.dumps(str(empty_home / 'data'))},\n"
        f"  CC_AGY_VENV_DIR: {json.dumps(str(empty_home / 'no-venv'))},\n"
        f"  CC_AGY_PYTHON_BIN: '${{user_config.python_bin}}',\n"
        f"  PATH: {json.dumps(str(bin_dir))},\n"
        f"}});\n"
        f"console.log(picked);\n"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == str(py312)
