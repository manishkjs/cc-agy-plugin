"""Regression tests for verification policy, MCP schema/dispatch, GenAI client, CLI pipe, receipt scan, and doc-sync consolidations."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.post_tool_ledger import _inspect_receipt_apply_state
from scripts.pre_tool_guard import (
    _receipt_owner_pid_is_alive,
    evaluate_pre_tool_use,
    summarize_session_receipts,
)
from scripts.sync_docs import SHARED_ROUTING_DOC_PATHS, sync_routing_blocks
from server.agy_cli import _start_pipe_chunk_drainer, _start_stdin_writer
from server.agy_runner import extract_usage_metadata as runner_extract_usage
from server.genai_client import build_genai_client, close_genai_client, extract_usage_metadata
from server.harness_validator import validate_plugin_tree
from server.jobs import JobManager
from server.mcp_stdio import MCPServer, TOOL_DEFINITIONS
from server.model_routes import (
    DEFAULT_ECONOMY_FLASH_MAX_TOOL_CALLS,
    DEFAULT_FLASH_MAX_TOOL_CALLS,
    render_routing_markdown_table,
)
from server.receipts import ReceiptStore, receipt_owner_pid_alive
from server.verification_policy import (
    classify_verification,
    evaluate_verification,
    has_real_acceptance_pass,
    is_verified_applied_receipt,
    satisfies_auto_apply_verification,
    satisfies_post_tool_auto_stop,
    satisfies_timeout_promotion,
)
from server import writers

REPO_ROOT = Path(__file__).resolve().parent.parent


def _init_git_repo(repo: Path, files: dict[str, str]) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True, capture_output=True)
    for rel, content in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True)


def test_unified_verification_policy_resolves_smoke_vs_discovered_drift(tmp_path: Path) -> None:
    smoke_entry = [{"command": "python -m py_compile a.py", "passed": True, "compile_smoke": True}]
    assert classify_verification(smoke_entry, True) == "smoke_only"
    assert has_real_acceptance_pass(smoke_entry) is False
    assert satisfies_auto_apply_verification(smoke_entry, acceptance_passed=True) is False
    assert satisfies_timeout_promotion(smoke_entry) is False

    smoke_rec = {
        "job_id": "job-smoke01",
        "status": "ok",
        "applied": True,
        "acceptance_passed": True,
        "acceptance_results": smoke_entry,
    }
    assert is_verified_applied_receipt(smoke_rec) is False
    assert satisfies_post_tool_auto_stop(smoke_rec) is False

    discovered_ok = [
        {
            "command": "pytest -q tests/test_a.py",
            "passed": True,
            "auto_discovered_default": True,
            "counts": {"passed": 4, "failed": 0},
        }
    ]
    ev_disc = evaluate_verification(discovered_ok, True)
    assert ev_disc.tier == "related_tests"
    assert ev_disc.explicit_acceptance_passed is False
    assert ev_disc.behavioral_tests_passed is True
    assert satisfies_auto_apply_verification(discovered_ok, acceptance_passed=True) is True
    assert satisfies_timeout_promotion(discovered_ok) is False

    disc_rec = {
        "job_id": "job-disc01",
        "status": "ok",
        "applied": True,
        "acceptance_passed": True,
        "acceptance_results": discovered_ok,
    }
    assert is_verified_applied_receipt(disc_rec) is True
    assert satisfies_post_tool_auto_stop(disc_rec) is False

    discovered_zero = [
        {
            "command": "pytest -q tests/test_a.py",
            "passed": True,
            "auto_discovered_default": True,
            "counts": {"passed": 0, "failed": 0},
        }
    ]
    assert satisfies_auto_apply_verification(discovered_zero, acceptance_passed=True) is False
    assert satisfies_timeout_promotion(discovered_zero) is False

    explicit_ok = [{"command": "pytest -q tests/test_a.py", "passed": True, "counts": {"passed": 2}}]
    assert classify_verification(explicit_ok, True) == "acceptance_tests"
    assert has_real_acceptance_pass(explicit_ok) is True
    assert satisfies_timeout_promotion(explicit_ok) is True
    assert satisfies_timeout_promotion(explicit_ok, test_only_diff=True) is False

    data_dir = tmp_path / "ledger_data"
    store = ReceiptStore(data_dir)
    store.write_receipt("job-smoke01", smoke_rec)
    store.write_receipt("job-disc01", disc_rec)
    exp_rec = {
        "job_id": "job-exp01",
        "status": "ok",
        "applied": True,
        "acceptance_passed": True,
        "acceptance_results": explicit_ok,
    }
    store.write_receipt("job-exp01", exp_rec)
    assert _inspect_receipt_apply_state(store, "job-smoke01") == (False, False)
    assert _inspect_receipt_apply_state(store, "job-disc01") == (False, False)
    assert _inspect_receipt_apply_state(store, "job-exp01") == (True, False)


def test_delegate_start_and_delegate_to_gemini_share_schema_and_executor(tmp_path: Path) -> None:
    by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
    start_props = by_name["delegate_start"]["inputSchema"]["properties"]
    dtg_props = by_name["delegate_to_gemini"]["inputSchema"]["properties"]

    assert set(start_props.keys()) - {"wait"} == set(dtg_props.keys())
    assert str(DEFAULT_FLASH_MAX_TOOL_CALLS) in start_props["max_tool_calls"]["description"]
    assert str(DEFAULT_ECONOMY_FLASH_MAX_TOOL_CALLS) in dtg_props["max_tool_calls"]["description"]

    repo = tmp_path / "mcp_repo"
    _init_git_repo(repo, {"app.py": "VALUE = 1\n", "test_app.py": "from app import VALUE\nassert VALUE == 2\n"})
    store = ReceiptStore(tmp_path / "mcp_data", workspace_root=repo)

    def ok_worker(req: dict[str, Any]) -> dict[str, Any]:
        (Path(req["workspace_dir"]) / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
        return {"text": "updated VALUE"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=ok_worker)
    srv = MCPServer(
        job_manager=jm,
        store=store,
        ask_backend=lambda _r: {"text": "VERDICT: PASS", "usage": {"prompt_token_count": 10, "total_token_count": 12}},
    )
    try:
        res = srv._dispatch_tool(
            "delegate_to_gemini",
            {
                "prompt": "Set VALUE = 2 in app.py.",
                "project": "p",
                "checklist": ["VALUE must equal 2"],
                "acceptance": ["python3 test_app.py"],
                "qa_review_model": "gemini-4-argon",
            },
        )
        assert res["status"] == "ok"
        assert res["applied"] is True
        assert res["qa_review"]["verdict"] == "PASS"
        assert (repo / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    finally:
        jm.shutdown()


def test_shared_genai_client_factory_cleanup_and_usage_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: list[dict[str, Any]] = []
    closed_flags: list[str] = []

    class FakeHttpx:
        def close(self) -> None:
            closed_flags.append("httpx_sync")

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.append(kwargs)
            self._api_client = SimpleNamespace(_httpx_client=FakeHttpx(), _async_httpx_client=None)
            self.caches = SimpleNamespace(
                create=lambda **_kw: SimpleNamespace(name="cachedContents/123"),
                delete=lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom")),
            )

        def close(self) -> None:
            closed_flags.append("client_close")

    import google.genai as genai_mod

    monkeypatch.setattr(genai_mod, "Client", FakeClient)
    monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL_TOKEN", "tok-secret")

    gw_auth: dict[str, Any] = {
        "auth_mode": "vertex",
        "api_key": None,
        "project": "proj-1",
        "location": "global",
        "base_url": "https://gw.example.internal",
    }
    client = build_genai_client(gw_auth, timeout_ms=45_000, zero_thinking=True)
    assert captured_kwargs[-1]["vertexai"] is True
    assert captured_kwargs[-1]["project"] == "proj-1"
    http_opts = captured_kwargs[-1]["http_options"]
    base_url_val = http_opts.base_url if hasattr(http_opts, "base_url") else http_opts["base_url"]
    headers_val = http_opts.headers if hasattr(http_opts, "headers") else http_opts["headers"]
    client_args_val = http_opts.client_args if hasattr(http_opts, "client_args") else http_opts["client_args"]
    assert base_url_val == "https://gw.example.internal"
    assert headers_val["Authorization"] == "Bearer tok-secret"
    assert headers_val["X-CC-AGY-Thinking-Budget"] == "0"
    assert client_args_val["follow_redirects"] is False
    assert client._api_client._use_google_auth_async() is False

    close_genai_client(client)
    assert "client_close" in closed_flags
    assert "httpx_sync" in closed_flags

    closed_flags.clear()
    cname = writers.create_shared_cache(gw_auth, "gemini-3.8-flash", "sys", "shared body")
    assert cname == "cachedContents/123"
    assert "client_close" in closed_flags

    closed_flags.clear()
    writers.delete_shared_cache(gw_auth, "cachedContents/123")
    assert "client_close" in closed_flags

    raw_obj = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=120,
            cached_content_token_count=40,
            candidates_token_count=30,
            thoughts_token_count=10,
            total_token_count=0,
        )
    )
    norm = extract_usage_metadata(raw_obj)
    assert norm == {
        "prompt_token_count": 120,
        "cached_content_token_count": 40,
        "candidates_token_count": 30,
        "thoughts_token_count": 10,
        "total_token_count": 160,
    }
    assert runner_extract_usage({"usage": norm}) == norm


def test_shared_cli_pipe_helpers_stream_stdin_and_drain_output() -> None:
    proc = subprocess.Popen(
        [
            "python3",
            "-c",
            "import sys; data = sys.stdin.buffer.read(); sys.stdout.buffer.write(data.upper()); sys.stderr.buffer.write(b'err-ok')",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    writer_t = _start_stdin_writer(proc, "hello shared pipe\n")
    out_t = _start_pipe_chunk_drainer(proc.stdout, out_chunks)
    err_t = _start_pipe_chunk_drainer(proc.stderr, err_chunks)
    assert proc.wait(timeout=5) == 0
    writer_t.join(timeout=2)
    out_t.join(timeout=2)
    err_t.join(timeout=2)
    assert b"".join(out_chunks) == b"HELLO SHARED PIPE\n"
    assert b"".join(err_chunks) == b"err-ok"
    closed_sink = io.BytesIO()
    empty_t = _start_stdin_writer(SimpleNamespace(stdin=closed_sink), "")  # type: ignore[arg-type]
    empty_t.join(timeout=2)
    assert closed_sink.closed is True


def test_pre_tool_guard_reads_each_session_receipt_at_most_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert receipt_owner_pid_alive({"owner_instance_id": f"{os.getpid()}-abcd"}) is True
    assert _receipt_owner_pid_is_alive({"owner_instance_id": "inst-999999999-dead"}) is False

    repo = tmp_path / "guard_repo"
    _init_git_repo(repo, {"a.py": "x = 1\n", "b.py": "y = 2\n"})
    data_dir = tmp_path / "guard_data"
    store = ReceiptStore(data_dir, workspace_root=repo)
    (data_dir / ".lead_session_started").write_text("sess-1", encoding="utf-8")

    for idx in range(4):
        store.write_receipt(
            f"job-done0{idx}",
            {
                "job_id": f"job-done0{idx}",
                "status": "ok",
                "kind": "delegate",
                "applied": True,
                "applied_files": ["a.py", "b.py"],
                "files_changed": ["a.py", "b.py"],
                "acceptance_passed": True,
                "acceptance_results": [{"command": "pytest -q", "passed": True, "counts": {"passed": 2}}],
                "started_at_epoch": 1000.0 + idx,
            },
            workspace_root=str(repo),
        )

    summary = summarize_session_receipts(data_dir, repo)
    assert summary.has_terminal_delegate is True
    assert summary.has_multifile_applied_delegate is True
    assert summary.sticky_verified_applied is True
    assert summary.has_lead_escalated_job is False

    read_counts: dict[str, int] = {}
    orig_read_text = Path.read_text

    def counting_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.parent == (data_dir / "receipts") and self.suffix == ".json":
            read_counts[self.name] = read_counts.get(self.name, 0) + 1
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    env = {
        "CC_AGY_STRICT_MODE": "true",
        "CC_AGY_COST_PROFILE": "economy",
        "CC_AGY_DATA_DIR": str(data_dir),
        "CLAUDE_PROJECT_DIR": str(repo),
    }

    # Bash `git diff` triggers multifile spotcheck, escalation check, terminal check, and review bonus check in one call.
    verdict = evaluate_pre_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": "git diff"}, "cwd": str(repo)},
        env=env,
    )
    assert verdict == {}
    assert set(read_counts.values()) == {1}, f"Expected single-pass read per receipt, got {read_counts}"


def test_partial_worker_exception_state_recovered_for_timeout_and_runtime_error(tmp_path: Path) -> None:
    repo = tmp_path / "exc_repo"
    _init_git_repo(repo, {"mod.py": "A = 1\n"})
    store = ReceiptStore(tmp_path / "exc_data", workspace_root=repo)

    for idx, exc_cls in enumerate((TimeoutError, RuntimeError)):

        def failing_worker(req: dict[str, Any], _cls: type[Exception] = exc_cls, _i: int = idx) -> dict[str, Any]:
            (Path(req["workspace_dir"]) / "mod.py").write_text("A = 2\n", encoding="utf-8")
            err = _cls(f"boom-{_i}")
            err.partial_usage = {  # type: ignore[attr-defined]
                "prompt_token_count": 200,
                "cached_content_token_count": 0,
                "candidates_token_count": 50,
                "thoughts_token_count": 10,
                "total_token_count": 260,
            }
            err.partial_tool_calls = [{"name": "edit_file", "args": {"path": "mod.py"}}]  # type: ignore[attr-defined]
            err.partial_containment = "straitjacket"  # type: ignore[attr-defined]
            raise err

        jm = JobManager(workspace_root=repo, store=store, worker_backend=failing_worker)
        try:
            res = jm.delegate_start(
                "Set A = 2 in mod.py.",
                model="gemini-4-argon",
                project="p",
                acceptance=None,
                wait=True,
                auto_apply=False,
            )
            rec = store.read_receipt(res["job_id"])
            assert rec["usage"]["total_token_count"] == 260
            assert rec["tool_call_count"] == 1
            assert rec["containment"] == "straitjacket"
            assert "mod.py" in rec["files_changed"]
            # Compile smoke alone must NEVER promote a timed-out job to status='ok'!
            if exc_cls is TimeoutError:
                assert rec["status"] == "timeout"
            else:
                assert rec["status"] == "error"
        finally:
            jm.shutdown()


def test_shared_routing_docs_and_budget_sync_validator(tmp_path: Path) -> None:
    assert sync_routing_blocks(REPO_ROOT, write=False) == []
    assert validate_plugin_tree(REPO_ROOT) == []

    # Copy shared docs into a temp root and verify drift detection + --write auto-repair
    for rel in SHARED_ROUTING_DOC_PATHS:
        src = REPO_ROOT / rel
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    skill_doc = tmp_path / "skills/vertex/SKILL.md"
    skill_doc.write_text(
        skill_doc.read_text(encoding="utf-8")
        .replace(render_routing_markdown_table(), "| Change | Route |\n|---|---|\n| stale | row |")
        .replace("default `55` on Flash", "default `40` on Flash"),
        encoding="utf-8",
    )
    errs = sync_routing_blocks(tmp_path, write=False)
    assert any("shared routing table out of sync" in e for e in errs)
    assert any("stale Flash tool-call cap 40" in e for e in errs)

    # Restore valid budget text and verify write=True repairs the shared routing block
    skill_doc.write_text(
        skill_doc.read_text(encoding="utf-8").replace("default `40` on Flash", "default `55` on Flash"),
        encoding="utf-8",
    )
    assert sync_routing_blocks(tmp_path, write=True) == []
    assert sync_routing_blocks(tmp_path, write=False) == []

    # Verify stale Flash cap guard covers Codex skill and delegator subagent docs too
    codex_skill = tmp_path / "plugins/cc-agy-plugin/skills/vertex/SKILL.md"
    orig_codex = codex_skill.read_text(encoding="utf-8")
    codex_skill.write_text(orig_codex + "\nCap: 75 calls.\n", encoding="utf-8")
    assert any("plugins/cc-agy-plugin/skills/vertex/SKILL.md: contains stale Flash tool-call cap 75" in e for e in sync_routing_blocks(tmp_path, write=False))
    codex_skill.write_text(orig_codex, encoding="utf-8")
