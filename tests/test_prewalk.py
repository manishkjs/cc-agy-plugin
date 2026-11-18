"""Hermetic tests for intra-worker prewalk (Carbon guide, then Hydrogen executor) over `agy acp`.

Every test drives a fake `agy acp --stdio` server written to `tmp_path`; no live model calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import textwrap
from typing import Any

import pytest

from server.agy_acp import active_warm_acp_jobs
from server.agy_cli import run_agy_cli_worker
from server.jobs import JobManager
from server.mcp_stdio import TOOL_DEFINITIONS
from server.prewalk import (
    PREWALK_BLOCK_START,
    is_real_code_edit_target,
    resolve_prewalk_config,
)
from server.receipts import ReceiptStore
from server.regions import estimate_cost_usd

_FAKE_ACP_SERVER = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json
    import os
    import sys

    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        sys.stderr.write("Usage: agy [options]\\n  --output-format string\\n  --mode string\\n"
                         "  acp  Start the Agent Client Protocol (ACP) stdio server\\n")
        raise SystemExit(0)
    if not argv or argv[0] != "acp":
        raise SystemExit(2)

    log_path = os.environ["FAKE_ACP_LOG"]
    mode = os.environ.get("FAKE_ACP_MODE", "edit")
    current = {"model": "gemini-4-hydrogen-medium"}


    def log(entry):
        entry["pid"] = os.getpid()
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\\n")


    def send(obj):
        print(json.dumps(obj), flush=True)


    def update(sid, payload):
        send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": payload}})


    def edit(sid, path):
        update(sid, {"sessionUpdate": "tool_call", "title": "edit_file", "kind": "edit", "rawInput": {"path": path}})
        update(sid, {"sessionUpdate": "tool_call_update", "title": "edit_file", "kind": "edit",
                     "status": "completed", "rawInput": {"path": path}})


    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        msg = json.loads(raw)
        method = msg.get("method")
        mid = msg.get("id")
        log({"method": method, "params": msg.get("params"), "model": current["model"]})
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1}})
        elif method == "session/new":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "sessionId": "acp-sess-1",
                "models": {
                    "availableModels": [
                        {"modelId": "gemini-4-carbon-medium"},
                        {"modelId": "gemini-4-hydrogen-medium"},
                    ],
                    "currentModelId": current["model"],
                },
            }})
        elif method == "session/set_model":
            current["model"] = msg["params"]["modelId"]
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif method == "session/set_mode":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif method == "session/prompt":
            sid = msg["params"]["sessionId"]
            if current["model"].startswith("gemini-4-carbon"):
                edit(sid, "scratch.py")
                edit(sid, "todo.md")
                if mode == "edit":
                    edit(sid, "src/app.py")
                usage = {"input_tokens": 1000, "output_tokens": 100}
            else:
                edit(sid, "src/app.py")
                usage = {"input_tokens": 400, "output_tokens": 50}
            update(sid, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "done"}})
            send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn", "usage": usage}})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
    """
)


def _write_fake_agy(tmp_path: Path) -> Path:
    bin_path = tmp_path / "agy-cfg" / "bin" / "agy"
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_text(_FAKE_ACP_SERVER, encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "data" / "worktrees" / "job-1"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return workspace


def _request(tmp_path: Path, **env_extra: str) -> dict[str, Any]:
    fake_agy = _write_fake_agy(tmp_path)
    log_path = tmp_path / "acp_log.jsonl"
    env = {
        "PATH": os.defpath,
        "CC_AGY_CLI_BIN": str(fake_agy),
        "CC_AGY_CLI_CONFIG_DIR": str(tmp_path / "agy-cfg"),
        "CC_AGY_CLI_PROTOCOL": "acp",
        "CC_AGY_WORKER_SANDBOX": "off",
        "FAKE_ACP_LOG": str(log_path),
        **env_extra,
    }
    return {
        "workspace_dir": str(_workspace(tmp_path)),
        "prompt": "Add input validation to src/app.py",
        "model": "gemini-4-hydrogen",
        "acceptance": ["pytest -q"],
        "env": env,
    }


def _log_entries(tmp_path: Path) -> list[dict[str, Any]]:
    log_path = tmp_path / "acp_log.jsonl"
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except OSError:
        return True
    return state != "Z"


def test_prewalk_swaps_to_executor_on_the_same_warm_session_after_first_real_edit(tmp_path: Path) -> None:
    out = run_agy_cli_worker(_request(tmp_path, CC_AGY_PREWALK="1"))
    entries = _log_entries(tmp_path)
    methods = [e["method"] for e in entries]
    prompts = [e for e in entries if e["method"] == "session/prompt"]
    set_models = [e["params"] for e in entries if e["method"] == "session/set_model"]

    assert len({e["pid"] for e in entries}) == 1
    assert methods.count("initialize") == 1 and methods.count("session/new") == 1
    assert set_models == [
        {"sessionId": "acp-sess-1", "modelId": "gemini-4-carbon-medium"},
        {"sessionId": "acp-sess-1", "modelId": "gemini-4-hydrogen-medium"},
    ]
    assert [p["model"] for p in prompts] == ["gemini-4-carbon-medium", "gemini-4-hydrogen-medium"]
    phase1_text = prompts[0]["params"]["prompt"][0]["text"]
    phase2_text = prompts[1]["params"]["prompt"][0]["text"]
    assert PREWALK_BLOCK_START in phase1_text and "at most 7 items" in phase1_text
    assert PREWALK_BLOCK_START not in phase2_text and "src/app.py" in phase2_text
    assert "session/cancel" in methods
    assert out["stop_reason"] == "STOP"
    assert out["agy_conversation_id"] == "acp-sess-1"
    assert out["prewalk"]["swapped"] is True
    assert out["prewalk"]["first_edit_path"] == "src/app.py"
    assert out["prewalk"]["skipped_reason"] is None


def test_prewalk_keeps_usage_cumulative_and_split_by_model(tmp_path: Path) -> None:
    out = run_agy_cli_worker(_request(tmp_path, CC_AGY_PREWALK="1"))

    by_model = out["prewalk"]["by_model"]
    assert by_model["gemini-4-carbon"]["prompt_token_count"] == 1000
    assert by_model["gemini-4-hydrogen"]["prompt_token_count"] == 400
    assert out["usage"]["prompt_token_count"] == 1400
    assert out["usage"]["candidates_token_count"] == 150
    assert [(r["model"], r["phase"]) for r in out["usage"]["requests"]] == [
        ("gemini-4-carbon", "prewalk_guide"),
        ("gemini-4-hydrogen", "prewalk_executor"),
    ]
    split_cost = estimate_cost_usd("gemini-4-hydrogen", out["usage"])["total_cost_usd"]
    carbon_cost = estimate_cost_usd("gemini-4-carbon", by_model["gemini-4-carbon"])["total_cost_usd"]
    hydrogen_cost = estimate_cost_usd("gemini-4-hydrogen", by_model["gemini-4-hydrogen"])["total_cost_usd"]
    assert split_cost == pytest.approx(carbon_cost + hydrogen_cost)


def test_prewalk_ignores_scratch_edits_and_skips_the_swap_without_a_real_edit(tmp_path: Path) -> None:
    out = run_agy_cli_worker(_request(tmp_path, CC_AGY_PREWALK="1", FAKE_ACP_MODE="noedit"))
    entries = _log_entries(tmp_path)

    assert [e["model"] for e in entries if e["method"] == "session/prompt"] == ["gemini-4-carbon-medium"]
    assert "session/cancel" not in [e["method"] for e in entries]
    assert out["prewalk"]["swapped"] is False
    assert out["prewalk"]["skipped_reason"] == "no_edit_in_phase1"
    assert out["stop_reason"] == "STOP"


def test_prewalk_is_off_by_default(tmp_path: Path) -> None:
    out = run_agy_cli_worker(_request(tmp_path))
    prompts = [e for e in _log_entries(tmp_path) if e["method"] == "session/prompt"]

    assert "prewalk" not in out
    assert len(prompts) == 1 and prompts[0]["model"] == "gemini-4-hydrogen-medium"
    assert PREWALK_BLOCK_START not in prompts[0]["params"]["prompt"][0]["text"]


def test_prewalk_request_flag_enables_it_without_env(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request["prewalk"] = True

    out = run_agy_cli_worker(request)

    assert out["prewalk"]["swapped"] is True


def test_warm_prewalk_process_is_terminated_when_the_worker_returns(tmp_path: Path) -> None:
    run_agy_cli_worker(_request(tmp_path, CC_AGY_PREWALK="1"))
    pid = _log_entries(tmp_path)[0]["pid"]

    assert active_warm_acp_jobs() == []
    assert _pid_alive(pid) is False


def test_resumed_conversation_does_not_prewalk_again(tmp_path: Path) -> None:
    request = _request(tmp_path, CC_AGY_PREWALK="1")
    request["agy_conversation_id"] = "acp-sess-1"

    out = run_agy_cli_worker(request)

    assert out["prewalk"]["skipped_reason"] == "resumed_conversation"
    assert out["prewalk"]["swapped"] is False


@pytest.mark.parametrize(
    ("kwargs", "env", "expected"),
    [
        ({}, {"CC_AGY_PREWALK": "1"}, True),
        ({}, {}, False),
        ({"read_only": True}, {"CC_AGY_PREWALK": "1"}, False),
        ({"requested_model": "gemini-4-carbon"}, {"CC_AGY_PREWALK": "1"}, False),
        ({"requested_model": "gemini-4-carbon", "explicit_prewalk": True}, {}, True),
        ({"explicit_prewalk": False}, {"CC_AGY_PREWALK": "1"}, False),
    ],
)
def test_resolve_prewalk_config(kwargs: dict[str, Any], env: dict[str, str], expected: bool) -> None:
    cfg = resolve_prewalk_config(env, **kwargs)

    assert cfg.enabled is expected
    assert cfg.guide_model == "gemini-4-carbon"
    assert cfg.executor_model == "gemini-4-hydrogen"


@pytest.mark.parametrize(
    ("raw_path", "expected"),
    [
        ("src/app.py", "src/app.py"),
        ("./src/app.py", "src/app.py"),
        ("scratch.py", None),
        ("todo.md", None),
        (".pytest_cache/v/cache/lastfailed", None),
        ("tmp/notes.txt", None),
        (".env", None),
        ("../outside.py", None),
    ],
)
def test_real_code_edit_target_filters_scratch_and_secrets(
    tmp_path: Path, raw_path: str, expected: str | None
) -> None:
    assert is_real_code_edit_target(raw_path, workspace_dir=tmp_path) == expected


def test_delegate_start_schema_exposes_prewalk() -> None:
    delegate_start = next(t for t in TOOL_DEFINITIONS if t["name"] == "delegate_start")

    assert delegate_start["inputSchema"]["properties"]["prewalk"]["type"] == "boolean"


def test_delegate_start_forwards_prewalk_and_records_it_on_the_receipt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "t@e.st"],
        ["git", "config", "user.name", "T"],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    (repo / "a.py").write_text("A = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    store = ReceiptStore(workspace_root=repo, data_dir=tmp_path / "data")
    seen: list[Any] = []

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        seen.append(req.get("prewalk"))
        Path(req["workspace_dir"], "a.py").write_text("A = 2\n", encoding="utf-8")
        return {
            "text": "done",
            "usage": {"prompt_token_count": 10, "total_token_count": 10},
            "prewalk": {"enabled": True, "swapped": True, "first_edit_path": "a.py"},
        }

    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        result = jm.delegate_start("Set A to 2", project="p", wait=True, prewalk=True)
    finally:
        jm.shutdown(wait=True)

    assert seen == [True]
    assert store.read_receipt(result["job_id"])["prewalk"]["swapped"] is True


# ---------------------------------------------------------------------------
# SDK path (google.antigravity over Vertex-wire gateways such as Cloud Code PA)
# ---------------------------------------------------------------------------


def _sdk_phase_result(model: str, *, prompt_tokens: int, first_edit: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "text": f"{model} done",
        "usage": {
            "prompt_token_count": prompt_tokens,
            "cached_content_token_count": 0,
            "candidates_token_count": 10,
            "thoughts_token_count": 0,
            "total_token_count": prompt_tokens + 10,
            "requests": [{"prompt_token_count": prompt_tokens, "candidates_token_count": 10, "model": model}],
        },
        "tool_calls": [{"name": "view_file", "args": {}}],
        "thoughts": [],
        "stop_reason": "PREWALK_FIRST_EDIT" if first_edit else None,
        "containment": "straitjacket",
        "containment_warning": None,
        "command_runs": [],
    }
    if first_edit:
        out["prewalk_first_edit_path"] = first_edit
    return out


def _sdk_request(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "prompts": ["Fix the bug in src/app.py"],
        "workspace_dir": str(tmp_path),
        "model": "gemini-4-hydrogen",
        "auth_mode": "vertex",
        "acceptance": ["pytest -q"],
        "max_tool_calls": 55,
        **extra,
    }


def test_sdk_prewalk_resumes_the_same_conversation_on_the_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from server import agy_runner

    seen: list[dict[str, Any]] = []

    async def fake_worker(req: dict[str, Any]) -> dict[str, Any]:
        seen.append(req)
        if req["model"] == "gemini-4-carbon":
            return _sdk_phase_result("gemini-4-carbon", prompt_tokens=1000, first_edit="src/app.py")
        return _sdk_phase_result("gemini-4-hydrogen", prompt_tokens=400)

    monkeypatch.setattr(agy_runner, "_run_sdk_worker", fake_worker)
    monkeypatch.setenv("CC_AGY_PREWALK", "1")

    out = asyncio.run(agy_runner._default_antigravity_backend(_sdk_request(tmp_path)))

    guide, executor = seen
    assert guide["model"] == "gemini-4-carbon" and guide["_prewalk_stop_on_first_edit"] is True
    assert PREWALK_BLOCK_START in guide["prompts"][0]
    assert executor["model"] == "gemini-4-hydrogen" and executor["_sdk_resume"] is True
    assert PREWALK_BLOCK_START not in executor["prompts"][0] and "src/app.py" in executor["prompts"][0]
    assert guide["_sdk_conversation_id"] == executor["_sdk_conversation_id"]
    assert guide["_sdk_save_dir"] == executor["_sdk_save_dir"]
    assert not Path(guide["_sdk_save_dir"]).exists()
    assert executor["max_tool_calls"] == 54
    assert out["prewalk"]["swapped"] is True and out["prewalk"]["first_edit_path"] == "src/app.py"
    assert out["usage"]["prompt_token_count"] == 1400
    assert [r["model"] for r in out["usage"]["requests"]] == ["gemini-4-carbon", "gemini-4-hydrogen"]
    assert len(out["tool_calls"]) == 2 and out["text"] == "gemini-4-hydrogen done"


def test_sdk_prewalk_without_a_real_edit_keeps_the_guide_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from server import agy_runner

    calls: list[str] = []

    async def fake_worker(req: dict[str, Any]) -> dict[str, Any]:
        calls.append(req["model"])
        return _sdk_phase_result(req["model"], prompt_tokens=500)

    monkeypatch.setattr(agy_runner, "_run_sdk_worker", fake_worker)

    out = asyncio.run(agy_runner._default_antigravity_backend(_sdk_request(tmp_path, prewalk=True)))

    assert calls == ["gemini-4-carbon"]
    assert out["prewalk"]["swapped"] is False and out["prewalk"]["skipped_reason"] == "no_edit_in_phase1"
    assert "prewalk_first_edit_path" not in out


def test_sdk_backend_skips_prewalk_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from server import agy_runner

    calls: list[dict[str, Any]] = []

    async def fake_worker(req: dict[str, Any]) -> dict[str, Any]:
        calls.append(req)
        return _sdk_phase_result(req["model"], prompt_tokens=100)

    monkeypatch.setattr(agy_runner, "_run_sdk_worker", fake_worker)

    out = asyncio.run(agy_runner._default_antigravity_backend(_sdk_request(tmp_path)))

    assert len(calls) == 1 and "_prewalk_stop_on_first_edit" not in calls[0]
    assert "prewalk" not in out


def test_sdk_first_edit_hooks_cancel_only_after_a_real_edit_completes(tmp_path: Path) -> None:
    import asyncio

    from google.antigravity import types as ag_types

    from server.agy_runner import _build_prewalk_first_edit_hooks

    cancels: list[bool] = []

    class _Conversation:
        async def cancel(self) -> None:
            cancels.append(True)

    class _Agent:
        conversation = _Conversation()

    state: dict[str, Any] = {"first_edit_path": None, "pending": {}, "cancel_task": None}
    pre_hook, post_hook = _build_prewalk_first_edit_hooks(str(tmp_path), state, [_Agent()])

    async def drive() -> None:
        for call_id, path, error in (
            ("c1", "scratch.py", None),
            ("c2", "src/app.py", "boom"),
            ("c3", "src/app.py", None),
        ):
            decision = await pre_hook.run(
                context=None, data=ag_types.ToolCall(name="edit_file", id=call_id, args={"TargetFile": path})
            )
            assert decision.allow is True
            await post_hook.run(context=None, data=ag_types.ToolResult(name="edit_file", id=call_id, error=error))
            if state["cancel_task"] is not None:
                await state["cancel_task"]

    asyncio.run(drive())

    assert state["first_edit_path"] == "src/app.py"
    assert cancels == [True]


def test_own_prewalk_cancel_is_tolerated_but_timeouts_and_real_cancels_are_not() -> None:
    import asyncio

    from google.antigravity import types as ag_types

    from server.agy_runner import _is_own_prewalk_cancel

    edited = {"first_edit_path": "src/app.py"}

    async def check(state: dict[str, Any], exc: BaseException) -> bool:
        return _is_own_prewalk_cancel(exc, state)

    assert asyncio.run(check(edited, ag_types.AntigravityCancelledError())) is True
    assert asyncio.run(check({"first_edit_path": None}, ag_types.AntigravityCancelledError())) is False
    assert asyncio.run(check(edited, TimeoutError("slow"))) is False

    async def cancelled_task() -> bool:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        # Python 3.11 keeps a cancel pending until it is delivered at an await, even after
        # uncancel(); take delivery here so only `cancelling()` still reports the real cancel.
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        try:
            return _is_own_prewalk_cancel(ag_types.AntigravityCancelledError(), edited)
        finally:
            task.uncancel()

    assert asyncio.run(cancelled_task()) is False


def test_executor_phase_drops_the_guide_calls_from_the_shared_sqlite_trajectory() -> None:
    from server.agy_runner import drop_sqlite_baseline

    def req(prompt: int, cached: int, out: int) -> dict[str, int]:
        return {
            "prompt_token_count": prompt,
            "cached_content_token_count": cached,
            "candidates_token_count": out,
            "thoughts_token_count": 0,
            "total_token_count": prompt + out,
        }

    telemetry = {
        "requests": [req(8843, 0, 500), req(32759, 8000, 600), req(44492, 40000, 900)],
        "tool_calls": [{"name": "view_file"}, {"name": "edit_file"}, {"name": "run_command"}],
    }

    out = drop_sqlite_baseline(telemetry, {"requests": 2, "tool_calls": 2})

    assert out["prompt_token_count"] == 44492 and out["cached_content_token_count"] == 40000
    assert out["candidates_token_count"] == 900 and len(out["requests"]) == 1
    assert out["tool_calls"] == [{"name": "run_command"}]
    assert drop_sqlite_baseline(telemetry, None) is telemetry


def test_sdk_prewalk_hands_the_guide_sqlite_counts_to_the_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from server import agy_runner

    seen: list[dict[str, Any]] = []

    async def fake_worker(req: dict[str, Any]) -> dict[str, Any]:
        seen.append(req)
        if req["model"] == "gemini-4-carbon":
            out = _sdk_phase_result("gemini-4-carbon", prompt_tokens=1000, first_edit="src/app.py")
            out["_sdk_sqlite_counts"] = {"requests": 3, "tool_calls": 15}
            return out
        out = _sdk_phase_result("gemini-4-hydrogen", prompt_tokens=400)
        out["_sdk_sqlite_counts"] = {"requests": 12, "tool_calls": 31}
        return out

    monkeypatch.setattr(agy_runner, "_run_sdk_worker", fake_worker)

    out = asyncio.run(agy_runner._default_antigravity_backend(_sdk_request(tmp_path, prewalk=True)))

    assert seen[1]["_sdk_sqlite_baseline"] == {"requests": 3, "tool_calls": 15}
    assert "_sdk_sqlite_counts" not in out


def test_acp_upstream_api_error_banner_marks_turn_as_failed() -> None:
    from server.agy_acp import _extract_acp_turn_error

    zero_usage = {"prompt_token_count": 0, "candidates_token_count": 0, "total_token_count": 0}
    banner = (
        "⚠ INVALID_ARGUMENT (code 400): "
        "* GenerateContentRequest.tools[0].function_declarations[56].parameters.properties[Arguments].type: "
        "field is required"
    )
    assert _extract_acp_turn_error({"stopReason": "end_turn"}, banner, tool_calls=[], usage_out=zero_usage) == banner
    assert (
        _extract_acp_turn_error(
            {"stopReason": "end_turn"},
            banner,
            tool_calls=[{"name": "run_command"}],
            usage_out=zero_usage,
        )
        == ""
    )
    assert (
        _extract_acp_turn_error(
            {"stopReason": "end_turn", "isError": True},
            "Upstream connection failed",
            tool_calls=[],
            usage_out=zero_usage,
        )
        == "Upstream connection failed"
    )


def test_worker_sandbox_hides_host_gemini_config_dir(tmp_path: Path) -> None:
    from server.worker_sandbox import build_bwrap_prefix, build_seatbelt_profile

    home = tmp_path / "home"
    host_gemini_cfg = home / ".gemini" / "config"
    cli_cfg = home / ".gemini" / "antigravity-cli"
    ws = tmp_path / "ws"
    for p in (host_gemini_cfg, cli_cfg, ws):
        p.mkdir(parents=True, exist_ok=True)

    bwrap_args = build_bwrap_prefix(
        "/usr/bin/bwrap",
        workspace=ws,
        config_dir=cli_cfg,
        home=home,
        adc_path=None,
    )
    joined_bwrap = " ".join(bwrap_args)
    assert f"--tmpfs {host_gemini_cfg.resolve()}" in joined_bwrap
    assert f"--bind {cli_cfg} {cli_cfg}" in joined_bwrap

    sbpl = build_seatbelt_profile(
        workspace=ws,
        config_dir=cli_cfg,
        home=home,
        adc_path=None,
    )
    assert str(host_gemini_cfg) in sbpl
