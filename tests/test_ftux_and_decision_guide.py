"""End-to-end tests for First-Time User Experience (FTUX) and Decision Maker's Guide guarantees."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any
import zipfile

import pytest

from server import file_lock
from server.agy_acp import probe_agy_acp_capabilities, resolve_cli_protocol, run_agy_acp_turn
from server.agy_cli import (
    _install_agy_cli_pure_python,
    _resolve_or_bootstrap_agy_bin,
    build_agy_subprocess_env,
    inspect_agy_cli_offline,
    probe_agy_cli_version,
)
from server.batch_plan import FileTask, validate_batch_plan
from server.batch_runner import FileBatchRunner, default_writer_factory, probe_writers
from server.doctor import format_doctor_human, run_doctor
from server.gateway import GatewayRequestError, gemini_credentials_available, gemini_endpoint, run_server
from server.install import (
    _materialize_persistent_checkout,
    install_codex,
    install_opencode,
    main as install_main,
    run_interactive_wizard,
)
from server.jobs import JobManager
from server.mcp_stdio import MCPServer, get_tool_definitions, serve_stdio
from server.model_routes import resolve_batch_settings
from server.receipts import BudgetExceededError, ReceiptStore, enforce_daily_budget
from server.regions import is_ge_mode_pinned, resolve_auth_config
from server.runtime_env import candidate_sdk_pythons, resolve_sdk_python
from server.true_cost import true_total_report


def _init_git_repo(repo_dir: Path, files: dict[str, str]) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)
    for rel, content in files.items():
        target = repo_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True, capture_output=True)


def _make_fake_external_agy(path: Path, version: str = "1.5.8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""#!/usr/bin/env python3
import json, sys

args = sys.argv[1:]
if "--help" in args and "acp" not in args:
    print("Usage: agy [flags]\\n  --output-format string\\n  --sandbox\\n  acp  Start the Agent Client Protocol (ACP) stdio server")
    sys.exit(0)
if "--version" in args:
    print("{version}")
    sys.exit(0)
if args and args[0] == "acp":
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
                "agentInfo": {{"name": "Antigravity ACP Server", "version": "{version}"}},
                "authMethods": [{{"id": "oauth-business", "name": "Gemini Enterprise"}}],
                "agentCapabilities": {{"loadSession": True}}
            }}}}), flush=True)
        elif method == "session/new":
            print(json.dumps({{"jsonrpc": "2.0", "id": mid, "result": {{
                "sessionId": "acp-sess-1",
                "models": {{
                    "availableModels": [
                        {{"modelId": "gemini-4-carbon-medium", "name": "Gemini 4 Carbon (Medium)"}},
                        {{"modelId": "gemini-3.8-flash-high", "name": "Gemini 3.8 Flash (High)"}}
                    ],
                    "currentModelId": "gemini-4-carbon-medium"
                }},
                "modes": {{
                    "currentModeId": "default",
                    "availableModes": [{{"id": "default"}}, {{"id": "accept-edits"}}, {{"id": "plan"}}]
                }},
                "_meta": {{"antigravity": {{"tier": {{"name": "gcp-ge-plus-tier", "id": "gcp-ge-plus-tier"}}}}}}
            }}}}), flush=True)
        elif method == "session/load":
            print(json.dumps({{"jsonrpc": "2.0", "id": mid, "result": {{"sessionId": msg["params"]["sessionId"]}}}}), flush=True)
        elif method in ("session/set_model", "session/set_mode"):
            sid = msg["params"]["sessionId"]
            if method == "session/set_model" and msg["params"]["modelId"] not in ("gemini-4-carbon-medium", "gemini-3.8-flash-high"):
                print(json.dumps({{"jsonrpc": "2.0", "id": mid, "error": {{"code": -32602, "message": "Unknown model ID"}}}}), flush=True)
            else:
                print(json.dumps({{"jsonrpc": "2.0", "id": mid, "result": {{"applied": msg["params"]}}}}), flush=True)
        elif method == "session/prompt":
            sid = msg["params"]["sessionId"]
            print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{
                "sessionId": sid,
                "update": {{"sessionUpdate": "tool_call", "title": "edit_file", "rawInput": {{"path": "app.py"}}}}
            }}}}), flush=True)
            print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{
                "sessionId": sid,
                "update": {{"sessionUpdate": "agent_message_chunk", "content": {{"type": "text", "text": "Updated via ACP."}}}}
            }}}}), flush=True)
            print(json.dumps({{"jsonrpc": "2.0", "id": mid, "result": {{
                "stopReason": "end_turn",
                "usage": {{"input_tokens": 120, "output_tokens": 30}}
            }}}}), flush=True)
            sys.exit(0)
    sys.exit(0)
sys.exit(0)
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_ge_pin_fails_closed_against_api_key_project_claude_and_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ge_env = {
        "CC_AGY_AUTH_MODE": "gemini_enterprise",
        "GEMINI_API_KEY": "AIzaAmbientShouldBeIgnored",
        "GOOGLE_CLOUD_PROJECT": "ambient-payg-proj",
        "HOME": str(tmp_path),
    }
    assert is_ge_mode_pinned(ge_env) is True

    # 1. Per-call api_key override must fail closed under GE pin
    with pytest.raises(ValueError, match="per-call api_key overrides are forbidden"):
        resolve_auth_config(requested_api_key="AIzaOverrideLeak999", env=ge_env)

    # 2. Unapproved per-call project override must fail closed under GE pin
    with pytest.raises(ValueError, match="per-call project override 'rogue-payg-project' is forbidden"):
        resolve_auth_config(requested_project="rogue-payg-project", env=ge_env)

    # Approved project in CC_AGY_ALLOWED_PROJECTS is permitted
    ok_auth = resolve_auth_config(
        requested_project="corp-ge-seat-proj",
        env={**ge_env, "CC_AGY_ALLOWED_PROJECTS": "corp-ge-seat-proj"},
    )
    assert ok_auth["auth_mode"] == "gemini_enterprise"
    assert ok_auth["project"] == "corp-ge-seat-proj"

    # 3. Batch settings disable Claude Opus escalation by default and reject claude_writer=True
    settings_default = resolve_batch_settings({}, env=ge_env, auth_mode="gemini_enterprise")
    assert settings_default.claude_writer is False
    assert settings_default.max_claude_escalations == 0

    with pytest.raises(ValueError, match="disables Claude writer escalation"):
        resolve_batch_settings({"claude_writer": True}, env=ge_env, auth_mode="gemini_enterprise")

    assert default_writer_factory("claude", settings_default, {"auth_mode": "gemini_enterprise"}) is None

    probe_out = probe_writers(
        settings_default,
        {"auth_mode": "gemini_enterprise"},
        writer_factory=lambda name, _s, _a: (
            (lambda _req: type("R", (), {"cost_cents": 0.0, "text": '{"path":"probe.txt","content":"ok"}'})())
            if name != "claude"
            else (_ for _ in ()).throw(AssertionError("claude writer should never be probed in GE mode"))
        ),
    )
    assert [w["writer"] for w in probe_out["writers"]] == ["genai"]

    # 4. MCP tools/list strips api_key, project, location, claude_writer, claude_model under GE pin
    tools = get_tool_definitions(ge_env)
    for tdef in tools:
        props = (tdef.get("inputSchema") or {}).get("properties") or {}
        for forbidden in ("api_key", "project", "location", "claude_writer", "claude_model", "max_claude_escalations"):
            assert forbidden not in props, f"{tdef['name']} leaked {forbidden} under GE pin"

    # 5. Gateway refuses startup and requests under GE pin
    monkeypatch.setenv("CC_AGY_AUTH_MODE", "gemini_enterprise")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFakeKey123456")
    assert gemini_credentials_available() is False
    with pytest.raises(GatewayRequestError) as gw_exc:
        gemini_endpoint("gemini-2.5-flash")
    assert gw_exc.value.status == 403
    with pytest.raises(RuntimeError, match="Refusing to start cc-agy-gateway"):
        run_server(0)


def test_mcp_esc_cancellation_and_live_progress_on_delegate_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo, {"calc.py": "x = 1\n"})
    data_dir = tmp_path / "data"
    monkeypatch.setenv("CC_AGY_DATA_DIR", str(data_dir))
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFakeTestKeyForStubWorker")
    monkeypatch.delenv("CC_AGY_AUTH_MODE", raising=False)

    worker_entered = threading.Event()
    worker_cancelled = threading.Event()

    def slow_stub_worker(req: dict[str, Any]) -> dict[str, Any]:
        reg_cancel = req.get("register_cancel")
        if callable(reg_cancel):
            reg_cancel(lambda: worker_cancelled.set())
        prog_path = req.get("progress_path")
        if prog_path:
            Path(f"{prog_path}.live").write_text(
                json.dumps({"straitjacket_calls": 2, "last_tool": "edit_file", "last_target": "calc.py"}),
                encoding="utf-8",
            )
        worker_entered.set()
        if worker_cancelled.wait(timeout=8.0):
            raise TimeoutError("worker cancelled via Esc")
        (Path(req["workspace_dir"]) / "calc.py").write_text("x = 99\n", encoding="utf-8")
        return {
            "text": "done",
            "usage": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15},
            "tool_calls": [{"name": "edit_file", "args": {"path": "calc.py"}}],
            "thoughts": [],
            "stop_reason": "STOP",
            "containment": "straitjacket",
        }

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=slow_stub_worker)
    mcp = MCPServer(job_manager=jm, store=store)

    notifications: list[dict[str, Any]] = []
    mcp.notify = lambda payload: notifications.append(payload)

    # Negotiates MCP 2025-06-18 cleanly
    init_resp = mcp.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}
    )
    assert isinstance(init_resp, dict)
    assert init_resp["result"]["protocolVersion"] == "2025-06-18"

    call_result_box: list[Any] = []

    def _run_call() -> None:
        resp = mcp.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 77,
                "method": "tools/call",
                "params": {
                    "name": "delegate_to_gemini",
                    "arguments": {"prompt": "Update calc.py"},
                    "_meta": {"progressToken": "tok-77"},
                },
            }
        )
        call_result_box.append(resp)

    t = threading.Thread(target=_run_call, daemon=True)
    t.start()
    assert worker_entered.wait(timeout=12.0)
    # Wait briefly so _wait_for_future_or_cancel emits progress from .live
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and len(notifications) < 2:
        time.sleep(0.05)

    # Press Esc (notifications/cancelled for requestId=77)
    mcp.handle_message({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 77}})
    t.join(timeout=4.0)
    assert not t.is_alive(), "delegate_to_gemini did not stop promptly on notifications/cancelled"
    assert worker_cancelled.is_set(), "worker cancel callback was not invoked on Esc"
    assert any(n.get("method") == "notifications/progress" for n in notifications)
    body = json.loads(call_result_box[0]["result"]["content"][0]["text"])
    assert body["status"] == "cancelled"
    assert (repo / "calc.py").read_text(encoding="utf-8") == "x = 1\n"


def test_post_merge_reverification_and_3way_delegate_revert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "merge_repo"
    initial_src = "\n".join([f"def fn_{i}():\n    return {i}\n" for i in range(1, 9)]) + "\n"
    _init_git_repo(
        repo,
        {
            "shared.py": initial_src,
            "test_shared.py": "from shared import fn_1, fn_8\n\ndef test_both():\n    assert fn_1() + fn_8() > 0\n",
        },
    )
    data_dir = tmp_path / "merge_data"
    monkeypatch.setenv("CC_AGY_DATA_DIR", str(data_dir))
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFakeMergeKey")
    monkeypatch.delenv("CC_AGY_AUTH_MODE", raising=False)

    def edit_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        p_text = "\n".join(req.get("prompts") or [])
        txt = (wt / "shared.py").read_text(encoding="utf-8")
        if "EDIT_FN1" in p_text:
            txt = txt.replace("def fn_1():\n    return 1", "def fn_1():\n    return 100")
        elif "EDIT_FN8_GOOD" in p_text:
            txt = txt.replace("def fn_8():\n    return 8", "def fn_8():\n    return 800")
        elif "EDIT_FN8_BREAK_SEMANTICS" in p_text:
            txt = txt.replace("def fn_8():\n    return 8", "def fn_8():\n    return -9999")
        (wt / "shared.py").write_text(txt, encoding="utf-8")
        return {
            "text": "updated",
            "usage": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15},
            "tool_calls": [{"name": "edit_file", "args": {"path": "shared.py"}}],
            "thoughts": [],
            "stop_reason": "STOP",
            "containment": "straitjacket",
        }

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=edit_worker)

    py_q = f"{shlex.quote(sys.executable)} -m pytest -q test_shared.py"
    j1 = jm.delegate_start(prompt="EDIT_FN1", acceptance=[py_q], wait=False, auto_apply=False)
    j2 = jm.delegate_start(prompt="EDIT_FN8_GOOD", acceptance=[py_q], wait=False, auto_apply=False)
    j3 = jm.delegate_start(prompt="EDIT_FN8_BREAK_SEMANTICS", acceptance=[py_q], wait=False, auto_apply=False)
    jm.delegate_result(j1["job_id"], wait=True)
    jm.delegate_result(j2["job_id"], wait=True)
    jm.delegate_result(j3["job_id"], wait=True)

    # Apply Job 1 cleanly
    app1 = jm.delegate_apply(j1["job_id"], cleanup_worktree=True)
    assert app1["applied"] is True
    assert "merged_files" not in app1

    # Job 3's non-overlapping edit merges at text level, bit breaks `fn_1() + fn_8() > 0` (100 + -9999 = -9899)!
    # Post-merge acceptance re-verification MUST block it and leave host untouched!
    with pytest.raises(ValueError, match="failed post-merge verification"):
        jm.delegate_apply(j3["job_id"], cleanup_worktree=False)
    assert "return -9999" not in (repo / "shared.py").read_text(encoding="utf-8")

    # Job 2's non-overlapping edit merges AND passes post-merge verification!
    app2 = jm.delegate_apply(j2["job_id"], cleanup_worktree=True)
    assert app2["applied"] is True
    assert app2["merged_files"] == ["shared.py"]
    assert app2["post_merge_verified"] is True
    merged_text = (repo / "shared.py").read_text(encoding="utf-8")
    assert "return 100" in merged_text and "return 800" in merged_text

    # Reverting Job 1 now 3-way reverse-merges Job 1's lines while keeping Job 2's `return 800`!
    rev1 = jm.delegate_revert(j1["job_id"])
    assert rev1["reverted"] is True
    after_rev = (repo / "shared.py").read_text(encoding="utf-8")
    assert "return 1\n" in after_rev
    assert "return 800" in after_rev


def test_ge_included_seat_zero_cost_split_budget_exemption_and_opencode_lead_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "cost_repo"
    _init_git_repo(repo, {"a.py": "a = 1\n"})
    data_dir = tmp_path / "cost_data"
    xdg_data = tmp_path / "xdg_data"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))
    monkeypatch.setenv("CC_AGY_MAX_DAILY_CENTS", "1.0")

    store = ReceiptStore(data_dir, workspace_root=repo)
    # Record a large GE seat run (list value > 1.0c, billed = 0.0c)
    ge_rec = store.write_receipt(
        "job-ge-1",
        {
            "status": "ok",
            "auth_mode": "gemini_enterprise",
            "model": "gemini-4-argon",
            "started_at_epoch": 1700000100.0,
            "finished_at_epoch": 1700000110.0,
            "usage": {
                "prompt_token_count": 200_000,
                "cached_content_token_count": 0,
                "candidates_token_count": 20_000,
                "thoughts_token_count": 0,
                "total_token_count": 220_000,
            },
        },
        workspace_root=str(repo),
    )
    assert ge_rec["cost"]["billing_model"] == "included_ge_seat"
    assert ge_rec["cost"]["billed_cost_cents"] == 0.0
    assert ge_rec["cost"]["total_cost_cents"] == 0.0
    assert ge_rec["cost"]["list_equivalent_cents"] > 10.0

    # Daily budget is NOT consumed by included GE seat runs
    budget_check = enforce_daily_budget(store, workspace_root=repo, now_epoch=1700000120.0)
    assert budget_check is not None
    assert budget_check["spent_today_cents"] == 0.0
    assert enforce_daily_budget(store, workspace_root=repo, auth_mode="gemini_enterprise") is None

    # Record a PAYG run that exceeds 1.0c
    store.write_receipt(
        "job-payg-1",
        {
            "status": "ok",
            "auth_mode": "api_key",
            "model": "gemini-4-argon",
            "started_at_epoch": 1700000130.0,
            "finished_at_epoch": 1700000140.0,
            "usage": {
                "prompt_token_count": 50_000,
                "cached_content_token_count": 0,
                "candidates_token_count": 5_000,
                "thoughts_token_count": 0,
                "total_token_count": 55_000,
            },
        },
        workspace_root=str(repo),
    )
    # PAYG is now blocked by the 1.0c daily cap, while GE seat STILL passes!
    with pytest.raises(BudgetExceededError):
        enforce_daily_budget(store, workspace_root=repo, auth_mode="api_key", now_epoch=1700000150.0)
    assert enforce_daily_budget(store, workspace_root=repo, auth_mode="gemini_enterprise", now_epoch=1700000150.0) is None

    # Write a synthetic OpenCode Lead session message in XDG storage
    sess_dir = xdg_data / "opencode" / "storage" / "session" / "proj1"
    msg_dir = xdg_data / "opencode" / "storage" / "message" / "ses_oc_1"
    sess_dir.mkdir(parents=True)
    msg_dir.mkdir(parents=True)
    (sess_dir / "ses_oc_1.json").write_text(
        json.dumps({"id": "ses_oc_1", "directory": str(repo.resolve())}),
        encoding="utf-8",
    )
    (msg_dir / "msg_1.json").write_text(
        json.dumps(
            {
                "id": "msg_1",
                "role": "assistant",
                "modelID": "claude-opus-4-6",
                "time": {"created": 1700000105000},
                "tokens": {"input": 1000, "output": 200, "reasoning": 0, "cache": {"read": 0, "write": 0}},
            }
        ),
        encoding="utf-8",
    )

    rep = true_total_report(store, repo, until_epoch=1700000200.0)
    assert rep["dispatched"]["by_auth_mode"]["gemini_enterprise_seat"]["billed_cost_cents"] == 0.0
    assert rep["dispatched"]["by_auth_mode"]["gemini_enterprise_seat"]["list_equivalent_cents"] > 10.0
    assert rep["dispatched"]["by_auth_mode"]["pay_as_you_go"]["billed_cost_cents"] > 1.0
    assert rep["dispatched"]["cost_cents"] == rep["dispatched"]["by_auth_mode"]["pay_as_you_go"]["billed_cost_cents"]
    assert rep["lead"]["messages"] == 1
    assert rep["lead"]["lead_cost_cents"] > 0.0


def test_ftux_doctor_host_config_consumer_block_wheel_assets_and_safe_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    fake_agy = _make_fake_external_agy(fake_home / ".local" / "share" / "cc-agy-plugin" / "bin" / "agy", "1.5.8")
    cfg_dir = fake_home / ".gemini" / "antigravity-cli"
    (cfg_dir / "cache").mkdir(parents=True)

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake_home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(fake_home / ".local" / "share"))
    monkeypatch.setenv("CC_AGY_CLI_BIN", str(fake_agy))
    monkeypatch.setenv("CC_AGY_CLI_CONFIG_DIR", str(cfg_dir))
    monkeypatch.delenv("CC_AGY_AUTH_MODE", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    # Ambient GOOGLE_CLOUD_PROJECT in shell must NOT hijack doctor after `--host opencode --auth gemini-enterprise`!
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "ambient-dev-sandbox")

    opencode_res = install_opencode(auth_mode="gemini-enterprise", agy_cli_path=str(fake_agy))
    assert opencode_res["status"] == "installed"
    assert (Path(opencode_res["commands_dir"]) / "agy.md").is_file()

    # Unsigned first-time install with allow_pending_login=True is healthy (degraded warning, exit 0)
    user_repo = tmp_path / "user_project"
    _init_git_repo(user_repo, {"hello.py": "print('hi')\n"})
    doc_pending = run_doctor(workspace_root=user_repo, allow_pending_login=True)
    assert doc_pending["auth"]["auth_mode"] == "gemini_enterprise"
    assert doc_pending["healthy"] is True
    human_out = format_doctor_human(doc_pending)
    assert "version=1.5.8" in human_out
    assert "fail_closed=True" in human_out
    assert "{'ecosystems'" not in human_out

    # Consumer Gmail onboarding is flagged and blocked in GE mode
    (cfg_dir / "cache" / "onboarding.json").write_text(
        json.dumps({"consumerOnboardingComplete": True, "tierId": "free-tier"}),
        encoding="utf-8",
    )
    cons_status = inspect_agy_cli_offline(os.environ)
    assert cons_status["consumer_account_detected"] is True
    assert cons_status["ok"] is False
    assert "personal/consumer Google account" in (cons_status["error"] or "")

    # Licensed GE tier passes
    (cfg_dir / "cache" / "onboarding.json").write_text(
        json.dumps({"enterpriseOnboardingComplete": True, "tierId": "gcp-ge-plus-tier", "cloudaicompanionProject": "ge-p"}),
        encoding="utf-8",
    )
    ent_status = inspect_agy_cli_offline(os.environ)
    assert ent_status["ok"] is True
    assert ent_status["version"] == "1.5.8"

    # Pinned version mismatch fails closed
    mismatch_status = inspect_agy_cli_offline({**os.environ, "CC_AGY_CLI_VERSION": "9.9.9"})
    assert mismatch_status["ok"] is False
    assert "9.9.9" in (mismatch_status["error"] or "")

    # Auto-install on tool call is OFF by default when binary is missing
    with pytest.raises(RuntimeError, match="not installed yet"):
        _resolve_or_bootstrap_agy_bin({"HOME": str(tmp_path / "empty_home"), "PATH": "/nonexistent"})

    # runtime_env never picks parent directory .venv
    nested_root = tmp_path / "parent_dir" / "plugin_checkout"
    nested_root.mkdir(parents=True)
    parent_venv_bin = tmp_path / "parent_dir" / ".venv" / "bin"
    parent_venv_bin.mkdir(parents=True)
    fake_parent_py = parent_venv_bin / "python"
    fake_parent_py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_parent_py.chmod(0o755)
    cands = candidate_sdk_pythons(nested_root, env={"HOME": str(fake_home)})
    assert all(parent_venv_bin not in p.parents for p in cands)

    # Wizard rejects typos instead of defaulting to "all"
    prompts_iter = iter(["opncode-typo", "6", "4", "1", ""])
    wizard_out: list[str] = []
    built_flags = run_interactive_wizard(input_fn=lambda _p: next(prompts_iter), output_fn=wizard_out.append, env={})
    assert built_flags[:2] == ["--host", "opencode"]
    assert any("Unrecognized host" in line for line in wizard_out)

    # Non-empty non-managed directory is NEVER deleted by _materialize_persistent_checkout or quickstart.sh
    precious = tmp_path / "precious_notes"
    precious.mkdir()
    (precious / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to overwrite non-empty directory"):
        _materialize_persistent_checkout(precious)
    assert (precious / "notes.txt").read_text(encoding="utf-8") == "keep me"

    repo_root = Path(__file__).resolve().parent.parent
    qs_res = subprocess.run(
        ["bash", str(repo_root / "scripts" / "quickstart.sh"), "--dry-run"],
        env={**os.environ, "CC_AGY_INSTALL_REPO_DIR": str(precious)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert qs_res.returncode != 0
    assert (precious / "notes.txt").read_text(encoding="utf-8") == "keep me"

    # Unrelated `codex` binary on PATH is skipped without noisy flag errors
    bad_bin_dir = tmp_path / "bad_codex_bin"
    bad_bin_dir.mkdir()
    unrelated_codex = bad_bin_dir / "codex"
    unrelated_codex.write_text("#!/bin/sh\necho 'Unrelated internal binary --blog --mod'\nexit 0\n", encoding="utf-8")
    unrelated_codex.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bad_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    codex_res = install_codex(
        name="agy",
        profile="balanced",
        project=None,
        location=None,
        dry_run=False,
        codex_home=tmp_path / "codex_home",
        auth_mode="gemini-enterprise",
    )
    assert "codex_verify_error" not in codex_res


def test_wheel_bundles_plugin_assets_and_acp_and_cross_platform_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    build_res = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(repo_root), "--no-deps", "--no-build-isolation", "-w", str(wheel_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build_res.returncode == 0, build_res.stderr
    wheels = list(wheel_dir.glob("cc_agy_plugin-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0], "r") as zf:
        names = set(zf.namelist())
    assert "server/_bundled_assets/.claude-plugin/plugin.json" in names
    assert "server/_bundled_assets/.opencode/commands/agy.md" in names
    assert "server/_bundled_assets/commands/agy.md" in names
    assert "server/_bundled_assets/hooks/hooks.json" in names
    assert "server/agy_acp.py" in names
    assert "server/file_lock.py" in names

    # Pure-Python cross-platform installer verifies SHA-512 & version pin
    stage_src = tmp_path / "archive_src"
    stage_src.mkdir()
    built_agy = _make_fake_external_agy(stage_src / "agy", "1.5.8")
    tar_path = tmp_path / "agy_release.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(built_agy, arcname="agy")
    sha512_hex = hashlib.sha512(tar_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"version": "1.5.8", "url": tar_path.as_uri(), "sha512": sha512_hex}),
        encoding="utf-8",
    )
    install_target = tmp_path / "installed_bin"
    py_inst = _install_agy_cli_pure_python(
        target_dir=install_target,
        source={"CC_AGY_CLI_VERSION": "1.5.8", "CC_AGY_CLI_SHA512": sha512_hex},
        manifest_url=manifest_path.as_uri(),
    )
    assert py_inst["version"] == "1.5.8"
    assert probe_agy_cli_version(py_inst["agy_bin"]) == "1.5.8"

    # ACP probe + turn execution + host .venv/bin PATH injection + portable file_lock fallback
    acp_caps = probe_agy_acp_capabilities(py_inst["agy_bin"])
    assert acp_caps["supported"] is True
    assert acp_caps["agent_version"] == "1.5.8"
    assert resolve_cli_protocol({"CC_AGY_CLI_PROTOCOL": "acp"}) == "acp"

    prog_file = tmp_path / "acp_progress.jsonl"
    acp_turn = run_agy_acp_turn(
        agy_bin=py_inst["agy_bin"],
        prompt_text="Say hello",
        cwd=str(tmp_path),
        model="gemini-3.8-flash",
        effort="high",
        read_only=False,
        skip_permissions=True,
        use_agy_sandbox=True,
        conversation_id=None,
        timeout_seconds=10.0,
        progress_path=prog_file,
    )
    assert acp_turn["ok"] is True
    assert acp_turn["text"] == "Updated via ACP."
    assert acp_turn["conversation_id"] == "acp-sess-1"
    assert len(acp_turn["tool_calls"]) == 1
    assert Path(f"{prog_file}.live").is_file()

    host_ws = tmp_path / "host_ws"
    (host_ws / ".venv" / "bin").mkdir(parents=True)
    wt_ws = tmp_path / "wt_ws"
    wt_ws.mkdir()
    sub_env = build_agy_subprocess_env({}, workspace_dir=wt_ws, host_workspace_root=host_ws)
    assert str((host_ws / ".venv" / "bin").resolve()) in sub_env["PATH"].split(os.pathsep)

    # Verify file_lock works both natively and when _fcntl is None
    lock_target = tmp_path / "portable.lock"
    with lock_target.open("w", encoding="utf-8") as fh:
        file_lock.flock(fh, file_lock.LOCK_EX)
        file_lock.flock(fh, file_lock.LOCK_UN)
        monkeypatch.setattr(file_lock, "_fcntl", None)
        file_lock.flock(fh, file_lock.LOCK_EX)
        file_lock.flock(fh, file_lock.LOCK_UN)


def test_cli_doctor_host_discovery_quickstart_repo_guard_ge_auth_mode_override_and_cross_file_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    fake_home = tmp_path / "clean_home"
    fake_home.mkdir()
    fake_agy = _make_fake_external_agy(fake_home / ".local" / "share" / "cc-agy-plugin" / "bin" / "agy", "1.5.8")
    cfg_dir = fake_home / ".gemini" / "antigravity-cli"
    (cfg_dir / "cache").mkdir(parents=True)

    # 1. `requested_auth_mode` cannot override an env-pinned `CC_AGY_AUTH_MODE=gemini_enterprise`
    for forbidden_mode in ("api_key", "vertex", "gateway"):
        with pytest.raises(ValueError, match="requested_auth_mode override"):
            resolve_auth_config(
                requested_auth_mode=forbidden_mode,
                env={"CC_AGY_AUTH_MODE": "gemini_enterprise", "GEMINI_API_KEY": "AIzaSyFakeKey999999999"},
            )
    # Auto-GE continued job (`requested_auth_mode="gemini_enterprise"`) also rejects rogue per-call `project`
    with pytest.raises(ValueError, match="per-call project override 'rogue-payg-proj' is forbidden"):
        resolve_auth_config(
            requested_auth_mode="gemini_enterprise",
            requested_project="rogue-payg-proj",
            env={},
        )

    # 2. `--fix` without `--doctor` is rejected immediately instead of triggering a 4-minute full install
    fix_proc = subprocess.run(
        [sys.executable, "-m", "server.install", "--fix"],
        cwd=str(repo_root),
        env={"HOME": str(fake_home), "PATH": os.environ.get("PATH", os.defpath)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert fix_proc.returncode == 2
    assert "--fix requires --doctor" in fix_proc.stderr

    # 3. CLI `python3 -m server.install --doctor` preserves `CC_AGY_AUTH_MODE=gemini_enterprise` and discovers `opencode.json`
    user_proj = tmp_path / "user_app"
    _init_git_repo(user_proj, {"main.py": "x = 1\n"})
    clean_cli_env = {
        "HOME": str(fake_home),
        "XDG_CONFIG_HOME": str(fake_home / ".config"),
        "XDG_DATA_HOME": str(fake_home / ".local" / "share"),
        "CC_AGY_CLI_BIN": str(fake_agy),
        "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
        "GOOGLE_CLOUD_PROJECT": "ambient-dev-sandbox",
        "PATH": os.environ.get("PATH", os.defpath),
    }
    inst_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "server.install",
            "--host",
            "opencode",
            "--auth",
            "gemini-enterprise",
            "--agy-cli-path",
            str(fake_agy),
            "--no-bootstrap",
        ],
        cwd=str(repo_root),
        env=clean_cli_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert inst_proc.returncode == 0, inst_proc.stdout + inst_proc.stderr

    doc_cli_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "server.install",
            "--doctor",
            "--json",
            "--allow-pending-login",
            "--workspace",
            str(user_proj),
        ],
        cwd=str(repo_root),
        env=clean_cli_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert doc_cli_proc.returncode == 0, doc_cli_proc.stdout + doc_cli_proc.stderr
    doc_json = json.loads(doc_cli_proc.stdout)
    assert doc_json["auth"]["auth_mode"] == "gemini_enterprise"
    assert doc_json["auth"]["configured_auth_mode"] == "gemini_enterprise"
    assert doc_json["healthy"] is True

    # 4. `quickstart.sh` refuses to touch an existing unmanaged user `.git` repository
    user_git_repo = tmp_path / "precious_git_repo"
    _init_git_repo(user_git_repo, {"work.py": "KEEP = True\n"})
    qs_refuse = subprocess.run(
        ["bash", str(repo_root / "scripts" / "quickstart.sh"), "--host", "opencode", "--auth", "gemini-enterprise"],
        cwd=str(user_proj),
        env={**clean_cli_env, "CC_AGY_INSTALL_REPO_DIR": str(user_git_repo), "PYTHON_BIN": sys.executable},
        capture_output=True,
        text=True,
        check=False,
    )
    assert qs_refuse.returncode != 0
    assert (user_git_repo / "work.py").read_text(encoding="utf-8") == "KEEP = True\n"
    with pytest.raises(RuntimeError, match="Refusing to overwrite non-empty directory"):
        _materialize_persistent_checkout(user_git_repo)

    # 5. End-to-end `quickstart.sh` succeeds (rc=0) on a clean first-time GE laptop
    staged_origin = tmp_path / "staged_origin"
    shutil.copytree(
        repo_root,
        staged_origin,
        symlinks=True,
        ignore_dangling_symlinks=True,
        ignore=shutil.ignore_patterns(".git", "docs", "__pycache__", "*.pyc", ".pytest_cache"),
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=staged_origin, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=staged_origin, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=staged_origin, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=staged_origin, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "staged working tree"], cwd=staged_origin, check=True, capture_output=True)

    qs_checkout = tmp_path / "managed_qs_checkout"
    qs_ok = subprocess.run(
        [
            "bash",
            str(staged_origin / "scripts" / "quickstart.sh"),
            "--host",
            "opencode",
            "--auth",
            "gemini-enterprise",
            "--agy-cli-path",
            str(fake_agy),
            "--no-bootstrap",
        ],
        cwd=str(user_proj),
        env={
            **clean_cli_env,
            "CC_AGY_REPO_URL": staged_origin.as_uri(),
            "CC_AGY_REPO_REF": "main",
            "CC_AGY_INSTALL_REPO_DIR": str(qs_checkout),
            "PYTHON_BIN": sys.executable,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert qs_ok.returncode == 0, qs_ok.stdout + qs_ok.stderr
    assert "gemini_enterprise" in qs_ok.stdout

    # 6. Cross-file post-merge verification catches semantic breakage across distinct files (`a.py` + `b.py`)
    cross_repo = tmp_path / "cross_repo"
    _init_git_repo(
        cross_repo,
        {
            "api.py": "def compute(x: int) -> int:\n    return x + 1\n",
            "caller.py": "from api import compute\n\ndef run() -> int:\n    return compute(5)\n",
            "test_cross.py": "from caller import run\n\ndef test_run():\n    assert run() == 6\n",
        },
    )
    monkeypatch.setenv("CC_AGY_AUTH_MODE", "api_key")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyCrossFileTestKey123456789")

    def cross_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        p = "\n".join(req.get("prompts") or [])
        if "RENAME_API" in p:
            (wt / "api.py").write_text("def compute_v2(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
            (wt / "caller.py").write_text("from api import compute_v2\n\ndef run() -> int:\n    return compute_v2(5)\n", encoding="utf-8")
            changed = "api.py"
        else:
            # Job 2 touches a separate file `extra.py` importing old `compute` from `api.py`!
            (wt / "extra.py").write_text("from api import compute\n\ndef extra_run() -> int:\n    return compute(1)\n", encoding="utf-8")
            (wt / "test_cross.py").write_text(
                "from caller import run\nfrom extra import extra_run\n\ndef test_run():\n    assert run() == 6 and extra_run() == 2\n",
                encoding="utf-8",
            )
            changed = "extra.py"
        return {
            "text": "done",
            "usage": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15},
            "tool_calls": [{"name": "edit_file", "args": {"path": changed}}],
            "thoughts": [],
            "stop_reason": "STOP",
            "containment": "straitjacket",
        }

    c_store = ReceiptStore(tmp_path / "cross_data", workspace_root=cross_repo)
    c_jm = JobManager(workspace_root=cross_repo, store=c_store, worker_backend=cross_worker)
    py_cross = f"{shlex.quote(sys.executable)} -m pytest -q test_cross.py"
    cj1 = c_jm.delegate_start(prompt="RENAME_API", acceptance=[py_cross], wait=False, auto_apply=False)
    cj2 = c_jm.delegate_start(prompt="ADD_EXTRA_CALLING_OLD_API", acceptance=[py_cross], wait=False, auto_apply=False)
    c_jm.delegate_result(cj1["job_id"], wait=True)
    c_jm.delegate_result(cj2["job_id"], wait=True)
    c_jm.delegate_apply(cj1["job_id"], cleanup_worktree=True)
    with pytest.raises(
        ValueError,
        match=r"Post-baseline verification.*concurrent host changes in \['api\.py', 'caller\.py'\]",
    ):
        c_jm.delegate_apply(cj2["job_id"], cleanup_worktree=False)
    assert not (cross_repo / "extra.py").exists()


def test_quickstart_sha_and_tag_pinning_host_deletion_overlay_and_bifrost_acp_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify commit-SHA/tag CC_AGY_REPO_REF pinning, host-deleted file overlay, and strict Bifrost ACP model lock."""
    from server.agy_acp import run_agy_acp_turn

    # 1. Stage a local git origin with annotated tag `v0.3.0` and a commit SHA
    staged_origin = tmp_path / "staged_origin_r4"
    shutil.copytree(
        Path(__file__).resolve().parents[1],
        staged_origin,
        ignore=shutil.ignore_patterns(".git", "docs", "__pycache__", "*.pyc", ".pytest_cache"),
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=staged_origin, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=staged_origin, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=staged_origin, check=True, capture_output=True)
    subprocess.run(["git", "config", "uploadpack.allowReachableSHA1InWant", "true"], cwd=staged_origin, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=staged_origin, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "release v0.3.0"], cwd=staged_origin, check=True, capture_output=True)
    pinned_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=staged_origin, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "tag", "-a", "v0.3.0", "-m", "v0.3.0"], cwd=staged_origin, check=True, capture_output=True)

    fake_agy = _make_fake_external_agy(tmp_path / "agy", "1.5.9")
    user_proj = tmp_path / "user_proj_r4"
    _init_git_repo(user_proj, {"main.py": "print('ok')\n"})
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("CC_AGY_")
        and k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_GEMINI_BASE_URL")
    }
    clean_env.update(
        {
            "HOME": str(tmp_path / "r4_home"),
            "XDG_CONFIG_HOME": str(tmp_path / "r4_home" / ".config"),
            "XDG_DATA_HOME": str(tmp_path / "r4_home" / ".local" / "share"),
            "XDG_CACHE_HOME": str(tmp_path / "r4_home" / ".cache"),
            "PYTHON_BIN": sys.executable,
            "CC_AGY_REPO_URL": staged_origin.as_uri(),
        }
    )

    # 1a. Pin by commit SHA (`CC_AGY_REPO_REF=<sha>`)
    sha_checkout = tmp_path / "sha_checkout"
    qs_sha = subprocess.run(
        [
            "bash",
            str(staged_origin / "scripts" / "quickstart.sh"),
            "--host",
            "opencode",
            "--auth",
            "gemini-enterprise",
            "--agy-cli-path",
            str(fake_agy),
            "--no-bootstrap",
        ],
        cwd=str(user_proj),
        env={**clean_env, "CC_AGY_REPO_REF": pinned_sha, "CC_AGY_INSTALL_REPO_DIR": str(sha_checkout)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert qs_sha.returncode == 0, qs_sha.stdout + qs_sha.stderr
    checked_out_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=sha_checkout, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert checked_out_sha == pinned_sha

    # 1b. Updating existing managed checkout with an invalid pinned ref fails loudly (`rc != 0`)
    qs_bad_ref = subprocess.run(
        [
            "bash",
            str(staged_origin / "scripts" / "quickstart.sh"),
            "--host",
            "opencode",
            "--auth",
            "gemini-enterprise",
            "--agy-cli-path",
            str(fake_agy),
            "--no-bootstrap",
        ],
        cwd=str(user_proj),
        env={**clean_env, "CC_AGY_REPO_REF": "v99.99.99-nonexistent", "CC_AGY_INSTALL_REPO_DIR": str(sha_checkout)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert qs_bad_ref.returncode != 0
    assert "failed to fetch pinned CC_AGY_REPO_REF" in qs_bad_ref.stderr

    # 2. Host deletion since baseline (`P2-b`) + `overlaid_host_files` response field (`P2-c`)
    del_repo = tmp_path / "del_repo"
    _init_git_repo(
        del_repo,
        {
            "legacy_helper.py": "def legacy_val() -> int:\n    return 7\n",
            "keep.py": "def base_val() -> int:\n    return 3\n",
            "test_del.py": "from keep import base_val\n\ndef test_base():\n    assert base_val() == 3\n",
        },
    )
    monkeypatch.setenv("CC_AGY_AUTH_MODE", "api_key")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDelOverlayTestKey123456789")

    def del_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        p = "\n".join(req.get("prompts") or [])
        if "USE_DELETED_HELPER" in p:
            (wt / "feature.py").write_text(
                "from legacy_helper import legacy_val\n\ndef feat() -> int:\n    return legacy_val()\n",
                encoding="utf-8",
            )
            (wt / "test_del.py").write_text(
                "from feature import feat\n\ndef test_base():\n    assert feat() == 7\n",
                encoding="utf-8",
            )
            changed = "feature.py"
        else:
            (wt / "safe_feature.py").write_text("def safe_feat() -> int:\n    return 42\n", encoding="utf-8")
            changed = "safe_feature.py"
        return {
            "text": "done",
            "usage": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15},
            "tool_calls": [{"name": "edit_file", "args": {"path": changed}}],
            "thoughts": [],
            "stop_reason": "STOP",
            "containment": "straitjacket",
        }

    d_store = ReceiptStore(tmp_path / "del_data", workspace_root=del_repo)
    d_jm = JobManager(workspace_root=del_repo, store=d_store, worker_backend=del_worker)
    py_del = f"{shlex.quote(sys.executable)} -m pytest -q test_del.py"
    dj_bad = d_jm.delegate_start(prompt="USE_DELETED_HELPER", acceptance=[py_del], wait=False, auto_apply=False)
    dj_good = d_jm.delegate_start(prompt="SAFE_FEATURE", acceptance=[py_del], wait=False, auto_apply=False)
    d_jm.delegate_result(dj_bad["job_id"], wait=True)
    d_jm.delegate_result(dj_good["job_id"], wait=True)

    # Host deletes `legacy_helper.py` after both jobs captured their baseline!
    (del_repo / "legacy_helper.py").unlink()

    with pytest.raises(ValueError, match=r"concurrent host changes in \['legacy_helper\.py'\]"):
        d_jm.delegate_apply(dj_bad["job_id"], cleanup_worktree=False)
    assert not (del_repo / "feature.py").exists()
    # Worktree copy of `legacy_helper.py` was cleanly restored after the failed overlay check
    bad_wt = Path(d_store.read_receipt(dj_bad["job_id"])["worktree_dir"])
    assert (bad_wt / "legacy_helper.py").is_file()

    good_applied = d_jm.delegate_apply(dj_good["job_id"], cleanup_worktree=True)
    assert good_applied["applied"] is True
    assert good_applied["overlaid_host_files"] == ["legacy_helper.py"]
    assert good_applied["post_merge_verified"] is True

    # 3. Recorded `agy 1.5.9 acp --stdio` wire fixture + strict Bifrost `session/set_model` enforcement
    acp_strict_bin = tmp_path / "fake_agy_acp_159.py"
    acp_strict_bin.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw)\n"
        "    mid = msg.get('id')\n"
        "    m = msg.get('method')\n"
        "    if m == 'initialize':\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {\n"
        "            'protocolVersion': 1,\n"
        "            'agentCapabilities': {'loadSession': True, 'sessionCapabilities': {'resume': {}, 'close': {}, 'load': {}, 'list': {}}},\n"
        "            'agentInfo': {'name': 'Antigravity ACP Server', 'version': '1.5.9'},\n"
        "            'authMethods': [{'id': 'oauth-business'}, {'id': 'agent-platform'}],\n"
        "        }}) + '\\n')\n"
        "        sys.stdout.flush()\n"
        "    elif m == 'session/new':\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {\n"
        "            'sessionId': 'sess-159',\n"
        "            'models': {'currentModelId': 'gemini-4-carbon-high', 'availableModels': [\n"
        "                {'modelId': 'gemini-3.8-flash-high', 'name': 'Gemini 3.8 Flash (High)'},\n"
        "                {'modelId': 'gemini-4-carbon-high', 'name': 'Gemini 4 Carbon (High)'},\n"
        "            ]},\n"
        "            'modes': {'currentModeId': 'default', 'availableModes': [{'id': 'default'}, {'id': 'accept-edits'}, {'id': 'plan'}]},\n"
        "        }}) + '\\n')\n"
        "        sys.stdout.flush()\n"
        "    elif m == 'session/set_model':\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32602, 'message': 'model rejected by server'}}) + '\\n')\n"
        "        sys.stdout.flush()\n",
        encoding="utf-8",
    )
    acp_strict_bin.chmod(0o755)

    res_rej = run_agy_acp_turn(
        agy_bin=str(acp_strict_bin),
        prompt_text="Check strict model failure",
        cwd=str(del_repo),
        model="gemini-3.8-flash",
        effort="high",
        strict_model_selection=True,
        read_only=True,
        skip_permissions=False,
        use_agy_sandbox=True,
        conversation_id=None,
        timeout_seconds=10.0,
    )
    assert res_rej["ok"] is False
    assert "Central Bifrost model policy requires ACP model" in res_rej["error"]

    res_unoffered = run_agy_acp_turn(
        agy_bin=str(acp_strict_bin),
        prompt_text="Check unoffered model under Bifrost",
        cwd=str(del_repo),
        model="gemini-4-argon",
        effort="high",
        strict_model_selection=True,
        read_only=True,
        skip_permissions=False,
        use_agy_sandbox=True,
        conversation_id=None,
        timeout_seconds=10.0,
    )
    assert res_unoffered["ok"] is False
    assert "not offered by this `agy acp` session" in res_unoffered["error"]
