"""Unit tests for all 6 Iteration 5 plugin reliability, accounting, and guardrail fixes."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from scripts.pre_tool_guard import (
    evaluate_pre_tool_use,
    lint_delegation_prompt,
    repair_dictated_prompt,
)
from scripts.provenance_check import verify_workspace_provenance
from server.agy_runner import (
    _force_terminate_agent_process,
    _reconcile_phase_usage_requests,
    append_progress,
)
from server.jobs import JobManager, run_acceptance
from server.mcp_stdio import MCPServer
from server.receipts import ReceiptStore


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)


# ---------------------------------------------------------------------------
# Fix 1: Orphan post-cancel / post-timeout worker process killer & cancel-aware escalation
# ---------------------------------------------------------------------------


def test_fix1_delegate_cancel_immediately_cancels_running_async_worker_and_kills_subprocess(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo_cancel"
    _init_repo(repo)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)

    # Spawn a real long-sleeping child process attached to a fake Agent hierarchy
    child_proc = subprocess.Popen(["python3", "-c", "import time; time.sleep(60)"])
    assert child_proc.poll() is None

    class _FakeConn:
        _process = child_proc

    class _FakeConv:
        connection = _FakeConn()

    class _FakeAgent:
        _conversation = _FakeConv()

    fake_agent = _FakeAgent()
    worker_started = threading.Event()
    worker_cancelled = threading.Event()

    async def slow_async_backend(req: dict[str, Any]) -> dict[str, Any]:
        reg = req.get("register_cancel")
        if callable(reg):
            reg(lambda: _force_terminate_agent_process(fake_agent))
        worker_started.set()
        try:
            await asyncio.sleep(30.0)
        except asyncio.CancelledError:
            worker_cancelled.set()
            _force_terminate_agent_process(fake_agent)
            raise
        return {"text": "should not finish"}

    store = ReceiptStore(tmp_path / "data_cancel", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=slow_async_backend)
    try:
        started = jm.delegate_start("Long running task", project="p", wait=False)
        job_id = started["job_id"]
        assert worker_started.wait(timeout=3.0)

        t0 = time.monotonic()
        canc = jm.delegate_cancel(job_id)
        elapsed = time.monotonic() - t0

        assert canc["cancelled"] is True
        assert canc["status"] == "cancelled"
        assert elapsed < 3.0
        assert worker_cancelled.wait(timeout=2.0)
        assert child_proc.poll() is not None

        rec = store.read_receipt(job_id)
        assert rec["status"] == "cancelled"
        assert rec.get("cancelled") is True
    finally:
        if child_proc.poll() is None:
            child_proc.kill()
            child_proc.wait(timeout=1.0)
        jm.shutdown()


def test_fix1_delegate_cancel_prevents_flash_to_argon_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo_cancel_no_esc"
    _init_repo(repo)
    (repo / "app.py").write_text("def add(a, b):\n    return 0\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)

    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")

    seen_models: list[str] = []
    phase1_in_flight = threading.Event()
    release_phase1 = threading.Event()

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        seen_models.append(req["model"])
        phase1_in_flight.set()
        release_phase1.wait(timeout=3.0)
        return {
            "text": "interrupted by cancel",
            "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
            "usage": {"prompt_token_count": 20_000, "candidates_token_count": 500, "total_token_count": 20_500},
        }

    store = ReceiptStore(tmp_path / "data_cancel_no_esc", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        started = jm.delegate_start(
            "Implement add(a, b) in app.py",
            model="gemini-3.8-flash",
            project="p",
            wait=False,
            auto_apply=False,
        )
        jid = started["job_id"]
        assert phase1_in_flight.wait(timeout=2.0)
        jm.delegate_cancel(jid)
        release_phase1.set()
        res = jm.delegate_result(jid, wait=True, timeout=5.0)
        assert res["status"] == "cancelled"
        assert seen_models == ["gemini-3.8-flash"]
        assert "escalated_from_model" not in res
    finally:
        release_phase1.set()
        jm.shutdown()


# ---------------------------------------------------------------------------
# Fix 2: Stop-hook provenance ignores plugin-generated escalation prompts
# ---------------------------------------------------------------------------


def test_fix2_provenance_check_skips_plugin_escalation_traceback_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo_prov_esc"
    _init_repo(repo)
    (repo / "auth.py").write_text("def proxy_header():\n    return None\n", encoding="utf-8")
    (repo / "test_auth.py").write_text(
        "import sys, pathlib\n"
        "from auth import proxy_header\n"
        "try:\n"
        "    assert proxy_header() == 'Basic dXNlcjpwYXNz'\n"
        "except Exception:\n"
        "    print(pathlib.Path('auth.py').read_text(encoding='utf-8'))\n"
        "    sys.exit(1)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)

    data_dir = tmp_path / "data_prov_esc"
    monkeypatch.setenv("CC_AGY_STRICT_MODE", "1")
    monkeypatch.setenv("CC_AGY_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")
    monkeypatch.setenv("CC_AGY_AUTO_APPLY_ON_PASS", "1")

    # Simulate Phase 1 failing with a >12-line Python snippet in acceptance output,
    # triggering plugin-generated escalation prompt in receipt["prompts"][1].
    call_num = 0

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_num
        call_num += 1
        wt = Path(req["workspace_dir"])
        if call_num == 1:
            # Write a file whose test failure prints >14 lines of Python source into acceptance output
            (wt / "auth.py").write_text(
                "def proxy_header():\n"
                + "".join(f"    step_{i} = {i}\n" for i in range(14))
                + "    raise RuntimeError('boom')\n",
                encoding="utf-8",
            )
            return {"text": "attempt 1"}
        (wt / "auth.py").write_text("def proxy_header():\n    return 'Basic dXNlcjpwYXNz'\n", encoding="utf-8")
        return {"text": "fixed in argon"}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        res = jm.delegate_start(
            "Return the Basic auth header from proxy_header() in auth.py.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert res["model"] == "gemini-4-argon"
        assert res["applied"] is True
        raw_rec = store.read_receipt(res["job_id"])
        assert raw_rec["prompt_sources"] == ["lead", "plugin_escalation"]
        # Confirm the raw escalation prompt itself would trip lint_delegation_prompt if un-exempted
        assert lint_delegation_prompt(raw_rec["prompts"][1])["ok"] is False

        verdict = verify_workspace_provenance(
            repo,
            store,
            env={"CC_AGY_STRICT_MODE": "true", "CC_AGY_DATA_DIR": str(data_dir)},
            strict=True,
        )
        assert verdict["dictation_violations"] == []
        assert verdict["ok"] is True
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# Fix 3: Complete per-request model pricing & SQLite/exception remainder itemization
# ---------------------------------------------------------------------------


def test_fix3_unitemized_flash_tokens_on_timeout_escalation_are_priced_at_flash_not_argon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo_pricing"
    _init_repo(repo)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)

    # First check _reconcile_phase_usage_requests directly when SQLite has more requests + remainder
    u_phase: dict[str, Any] = {
        "prompt_token_count": 200_000,
        "cached_content_token_count": 0,
        "candidates_token_count": 10_000,
        "thoughts_token_count": 0,
        "total_token_count": 210_000,
        "requests": [{"prompt_token_count": 50_000, "candidates_token_count": 2_000, "model": "gemini-3.8-flash"}],
    }
    sql_tel = {
        "requests": [
            {"prompt_token_count": 50_000, "candidates_token_count": 2_000},
            {"prompt_token_count": 70_000, "candidates_token_count": 3_000},
        ],
        "max_request_prompt_tokens": 70_000,
    }
    _reconcile_phase_usage_requests(u_phase, sql_tel, "gemini-3.8-flash")
    assert sum(r["prompt_token_count"] for r in u_phase["requests"]) == 200_000
    assert all(r["model"] == "gemini-3.8-flash" for r in u_phase["requests"])

    # Now test full Phase 1 Flash timeout (with partial requests < total tokens) -> Phase 2 Argon escalation
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")

    phase = 0

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal phase
        phase += 1
        if phase == 1:
            exc = TimeoutError("flash timed out")
            # 100k itemized + 100k unitemized Flash prompt tokens (total 200k Flash input, 10k Flash output)
            setattr(
                exc,
                "partial_usage",
                {
                    "prompt_token_count": 200_000,
                    "cached_content_token_count": 0,
                    "candidates_token_count": 10_000,
                    "thoughts_token_count": 0,
                    "total_token_count": 210_000,
                    "requests": [
                        {
                            "prompt_token_count": 100_000,
                            "cached_content_token_count": 0,
                            "candidates_token_count": 5_000,
                            "thoughts_token_count": 0,
                            "total_token_count": 105_000,
                        }
                    ],
                },
            )
            raise exc
        wt = Path(req["workspace_dir"])
        (wt / "app.py").write_text("x = 2\n", encoding="utf-8")
        # Phase 2 Argon: 100k input, 10k output
        return {
            "text": "done",
            "usage": {
                "prompt_token_count": 100_000,
                "cached_content_token_count": 0,
                "candidates_token_count": 10_000,
                "thoughts_token_count": 0,
                "total_token_count": 110_000,
            },
        }

    store = ReceiptStore(tmp_path / "data_pricing", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        res = jm.delegate_start(
            "Update x in app.py",
            model="gemini-3.8-flash",
            project="p",
            wait=True,
            auto_apply=False,
        )
        assert res["model"] == "gemini-4-argon"
        # Expected:
        # Flash: 200k input @ $0.50/1M ($0.10) + 10k output @ $3.00/1M ($0.03) = $0.13
        # Argon: 100k input @ $2.00/1M ($0.20) + 10k output @ $12.00/1M ($0.12) = $0.32
        # Total = $0.45 (NOT $0.645 if the 100k unitemized Flash remainder were priced at Argon!)
        assert abs(float(res["cost"]["total_cost_usd"]) - 0.45) < 1e-6
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# Fix 4: Cumulative max_tool_calls and max_total_tokens across escalation phases
# ---------------------------------------------------------------------------


def test_fix4_escalation_subtracts_phase1_tool_calls_and_tokens_from_phase2_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo_cum_budget"
    _init_repo(repo)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)

    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")

    seen_reqs: list[dict[str, Any]] = []

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        seen_reqs.append(dict(req))
        wt = Path(req["workspace_dir"])
        if len(seen_reqs) == 1:
            return {
                "text": "hit flash cap",
                "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
                "tool_calls": [{"name": "view_file", "args": {"path": "app.py"}} for _ in range(55)],
                "usage": {
                    "prompt_token_count": 2_200_000,
                    "candidates_token_count": 150_000,
                    "total_token_count": 2_350_000,
                },
            }
        (wt / "app.py").write_text("x = 2\n", encoding="utf-8")
        return {
            "text": "done in argon",
            "tool_calls": [{"name": "edit_file", "args": {"path": "app.py"}} for _ in range(10)],
            "usage": {
                "prompt_token_count": 500_000,
                "candidates_token_count": 20_000,
                "total_token_count": 520_000,
            },
        }

    store = ReceiptStore(tmp_path / "data_cum_budget", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        res = jm.delegate_start(
            "Update app.py",
            model="gemini-3.8-flash",
            project="p",
            wait=True,
            auto_apply=False,
        )
        assert len(seen_reqs) == 2
        assert seen_reqs[0]["max_tool_calls"] == 55
        assert seen_reqs[0]["max_total_tokens"] == 5_000_000
        # Phase 2 Argon receives max(60 floor, 100 - 55) = 60 calls, and 5M - 2.35M = 2.65M tokens
        assert seen_reqs[1]["max_tool_calls"] == 60
        assert seen_reqs[1]["max_total_tokens"] == 2_650_000
        # Job receipt preserves the cumulative job ceiling (100 / 5,000,000)
        assert res["budget"] == {"max_tool_calls": 100, "max_total_tokens": 5_000_000}
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# Fix 5: Checklist / semicolon code-dictation guard (preserving 1-line API hints) + spec_conflict
# ---------------------------------------------------------------------------


def test_fix5_checklist_and_semicolon_dictation_guarded_while_1line_api_hints_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Concise 1-line inline API hint passes cleanly
    concise = lint_delegation_prompt(
        "Ensure `LogCaptureFixture.clear()` calls `self.records.clear()` and resets handler stream."
    )
    assert concise["ok"] is True

    # 2. Semicolon-packed multi-statement Python implementation across plan/checklist items is flagged
    packed_plan = [
        "BaseExpression.get_refs(): refs = set(); for expr in self.get_source_expressions(): refs |= expr.get_refs(); return refs.",
        "Query.trim_unused(): used = set(); for a in self.annotation_select.values(): used |= a.get_refs(); self.annotations = {k: v for k, v in self.annotations.items() if k in used}; return used.",
    ]
    blocked = evaluate_pre_tool_use(
        {
            "tool_name": "mcp__plugin_cc-agy_vertex-gemini-agent__delegate_start",
            "tool_input": {
                "prompt": "Optimize unused annotations.",
                "checklist": packed_plan,
            },
            "cwd": str(tmp_path),
        },
        env={"CC_AGY_STRICT_MODE": "1"},
    )
    assert blocked["hookSpecificOutput"]["permissionDecision"] == "deny"

    # 3. Auto-repair repairs dictated `checklist` items in MCPServer
    repo = tmp_path / "repo_spec"
    _init_repo(repo)
    (repo / "url.py").write_text("def parse(u):\n    return u\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)

    monkeypatch.setenv("CC_AGY_REPAIR_CODE_DICTATION", "1")
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")

    seen_models: list[str] = []

    def conflict_worker(req: dict[str, Any]) -> dict[str, Any]:
        seen_models.append(req["model"])
        append_progress(
            req["progress_path"],
            "spec_conflict",
            "urllib3.util.parse_url('user:pass@host:80') parses 'user' as scheme, contradicting checklist item 2.",
        )
        return {"text": "Stopped due to verified spec conflict."}

    store = ReceiptStore(tmp_path / "data_spec", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=conflict_worker)
    srv = MCPServer(job_manager=jm, store=store)
    try:
        out = srv._dispatch_tool(
            "delegate_start",
            {
                "prompt": "Fix proxy auth parsing in url.py.",
                "project": "p",
                "wait": True,
                "checklist": packed_plan,
            },
        )
        assert out["dictation_repaired"] is True
        # Structured spec_conflict returns immediately to Lead WITHOUT burning an Argon escalation!
        assert out["status"] == "spec_conflict"
        assert "is_error" not in out
        assert "urllib3.util.parse_url" in out["spec_conflict"]
        assert seen_models == ["gemini-flash-next"]
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# Fix 6: Non-vacuous acceptance guard + single read-only Lead `git diff` spot-check on multi-file tasks
# ---------------------------------------------------------------------------


def test_fix6_vacuous_acceptance_blocked_and_single_multifile_git_diff_spotcheck_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo_vacuous"
    _init_repo(repo)
    (repo / "a.py").write_text("A = 1\n", encoding="utf-8")
    (repo / "b.py").write_text("B = 1\n", encoding="utf-8")
    (repo / "test_ok.py").write_text(
        "import unittest\nclass T(unittest.TestCase):\n    def test_existing(self):\n        self.assertEqual(1, 1)\n"
        "if __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)

    # 6a. `unittest` running 0 tests exits 0 in Python, but run_acceptance marks it vacuous (passed=False)
    zero_res = run_acceptance(["python3 -m unittest discover -s . -p 'test_nonexistent_*.py'"], repo)
    assert zero_res[0]["passed"] is False
    assert zero_res[0].get("vacuous") is True

    # 6b. Worker hitting checkpoint with 0 modified files must NOT report acceptance_passed=True
    data_dir = tmp_path / "data_vacuous"
    store = ReceiptStore(data_dir, workspace_root=repo)

    call_idx = 0

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_idx
        call_idx += 1
        wt = Path(req["workspace_dir"])
        if call_idx == 1:
            # 0 files modified in worktree!
            return {"text": "still reading", "stop_reason": "MAX_TOOL_CALLS_EXCEEDED"}
        (wt / "a.py").write_text("A = 2\n", encoding="utf-8")
        (wt / "b.py").write_text("B = 2\n", encoding="utf-8")
        return {"text": "updated both files"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        d1 = jm.delegate_start(
            "Update A and B",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_ok.py"],
            wait=True,
            auto_apply=False,
        )
        assert d1["files_changed"] == []
        assert d1["acceptance_passed"] is None
        assert d1.get("vacuous_acceptance") is True

        # 6c. Multi-file applied job allows exactly ONE read-only `git diff` spot-check by the Lead
        monkeypatch.setenv("CC_AGY_AUTO_APPLY_ON_PASS", "1")
        env = {
            "CC_AGY_STRICT_MODE": "1",
            "CC_AGY_DATA_DIR": str(data_dir),
            "CC_AGY_LEAD_READ_LIMIT": "2",
        }
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / ".lead_session_started").write_text(json.dumps({"ts": time.time() - 5}), encoding="utf-8")
        # Exhaust the 2 initial Lead reads
        (data_dir / ".lead_read_count").write_text("2", encoding="utf-8")

        d2 = jm.delegate_start(
            "Update both a.py and b.py",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_ok.py"],
            wait=True,
        )
        assert d2["applied"] is True
        assert sorted(d2["applied_files"]) == ["a.py", "b.py"]

        # First read-only `git diff` spot-check is allowed!
        allow_once = evaluate_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "git diff --stat"}, "cwd": str(repo)},
            env=env,
        )
        assert allow_once == {}

        # Second `git diff` is blocked by LeadReadBudgetExceeded!
        deny_second = evaluate_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "git diff"}, "cwd": str(repo)},
            env=env,
        )
        assert deny_second["hookSpecificOutput"]["permissionDecision"] == "deny"
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# Opus Round 1 Adversarial Regressions (P1..P9)
# ---------------------------------------------------------------------------


def test_opus_r1_budget_continuation_exemptions_and_stop_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # P1: Explicit Lead follow-up after token cap does NOT overwrite `ok` receipt and grants a new slice
    r1 = tmp_path / "r_p1"
    _init_repo(r1)
    (r1 / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(r1), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(r1), check=True)

    calls = 0

    def w_tok(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        (Path(req["workspace_dir"]) / "app.py").write_text(f"x = {calls + 1}\n", encoding="utf-8")
        return {
            "text": "ok",
            "usage": {"prompt_token_count": 1400, "candidates_token_count": 100, "total_token_count": 1500},
        }

    jm1 = JobManager(workspace_root=r1, store=ReceiptStore(tmp_path / "d_p1", workspace_root=r1), worker_backend=w_tok)
    try:
        d1 = jm1.delegate_start(
            "Update app.py", model="gemini-4-argon", project="p", wait=True, auto_apply=False, max_total_tokens=1000
        )
        assert d1["status"] == "ok"
        c1 = jm1.delegate_continue(d1["job_id"], "Add docstring.", wait=True, auto_apply=False)
        assert c1.get("continued") is not False
        assert c1["status"] == "ok"
        assert calls == 2
    finally:
        jm1.shutdown()

    # P1c: Flash explicit max_tool_calls=10 + escalation -> auto-escalation halts at cap (`MAX_TOOL_CALLS_EXCEEDED`),
    # while an explicit Lead `delegate_continue` proceeds cleanly onto Argon!
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")
    r2 = tmp_path / "r_p1c"
    _init_repo(r2)
    (r2 / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(r2), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(r2), check=True)

    seen_invocations: list[tuple[str, int | None]] = []

    def w_flash(req: dict[str, Any]) -> dict[str, Any]:
        seen_invocations.append((req["model"], req.get("max_tool_calls")))
        (Path(req["workspace_dir"]) / "app.py").write_text("x = 9\n", encoding="utf-8")
        return {
            "text": "cap",
            "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
            "tool_calls": [{"name": "view_file", "args": {}}] * 10,
            "usage": {"prompt_token_count": 10, "candidates_token_count": 1, "total_token_count": 11},
        }

    jm2 = JobManager(workspace_root=r2, store=ReceiptStore(tmp_path / "d_p1c", workspace_root=r2), worker_backend=w_flash)
    try:
        d2 = jm2.delegate_start(
            "Update app.py", model="gemini-3.8-flash", project="p", wait=True, auto_apply=False, max_tool_calls=10
        )
        assert d2["status"] == "checkpoint"
        assert d2.get("budget_exhausted") is True
        assert seen_invocations == [("gemini-3.8-flash", 10)]

        c2 = jm2.delegate_continue(d2["job_id"], "Finish the remaining edit.", wait=True, auto_apply=False)
        assert c2.get("continued") is not False
        assert seen_invocations == [("gemini-3.8-flash", 10), ("gemini-4-argon", 10)]
    finally:
        jm2.shutdown()


def test_opus_r1_process_tree_and_save_dir_env_termination_plus_shutdown_status(tmp_path: Path) -> None:
    import os
    import sys

    from server.agy_runner import _pid_is_alive_non_zombie

    # P2: Grandchild process spawned by harness is killed along with parent
    gc_pid_file = tmp_path / "gc.pid"
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import subprocess, time; p = subprocess.Popen(['sleep', '120']); "
            f"open({str(gc_pid_file)!r}, 'w').write(str(p.pid)); time.sleep(120)",
        ]
    )
    gc_pid: int | None = None
    try:
        for _ in range(60):
            if gc_pid_file.is_file() and gc_pid_file.read_text(encoding="utf-8").strip():
                gc_pid = int(gc_pid_file.read_text(encoding="utf-8").strip())
                break
            time.sleep(0.05)
        assert gc_pid is not None

        class _Obj:
            pass

        conn = _Obj()
        conn._process = parent
        conv = _Obj()
        conv._connection = conn
        ag = _Obj()
        ag._conversation = conv

        killed = _force_terminate_agent_process(ag)
        assert killed >= 2
        assert parent.poll() is not None
        assert not _pid_is_alive_non_zombie(gc_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
        if gc_pid is not None:
            try:
                os.kill(gc_pid, 9)
            except OSError:
                pass

    # P3: Harness matched via CC_AGY_HARNESS_SAVE_DIR in environ or open fd inside save_dir
    save_dir = tmp_path / "save_env_match"
    save_dir.mkdir(parents=True)
    env_p3 = {**os.environ, "CC_AGY_HARNESS_SAVE_DIR": str(save_dir)}
    lh = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], env=env_p3)
    try:
        time.sleep(0.1)
        killed_env = _force_terminate_agent_process(None, str(save_dir))
        assert killed_env >= 1
        lh.wait(timeout=1.0)
        assert lh.poll() is not None
    finally:
        if lh.poll() is None:
            lh.kill()

    # P9: JobManager.shutdown() without prior mark_running_interrupted labels receipt `interrupted`, not `timeout`
    r9 = tmp_path / "r_p9"
    _init_repo(r9)
    (r9 / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(r9), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(r9), check=True)

    started_evt = threading.Event()

    async def slow_worker(req: dict[str, Any]) -> dict[str, Any]:
        started_evt.set()
        await asyncio.sleep(30.0)
        return {"text": "done"}

    jm9 = JobManager(
        workspace_root=r9, store=ReceiptStore(tmp_path / "d_p9", workspace_root=r9), worker_backend=slow_worker
    )
    d9 = jm9.delegate_start("Long job", project="p", model="gemini-4-argon", wait=False)
    assert started_evt.wait(timeout=3.0)
    jm9.shutdown(wait=True)
    rec9 = jm9.store.read_receipt(d9["job_id"])
    assert rec9["status"] == "interrupted"


def test_opus_r1_provenance_apostrophe_git_diff_and_skipped_acceptance_guards(tmp_path: Path) -> None:
    import sys

    from scripts.pre_tool_guard import _is_readonly_git_diff_command

    # P4: Lead prompt starting with escalation prefix is still flagged when prompt_source == 'lead'
    r4 = tmp_path / "r_p4"
    _init_repo(r4)
    (r4 / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(r4), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(r4), check=True)

    d4 = tmp_path / "d_p4"
    st4 = ReceiptStore(d4, workspace_root=r4)
    jm4 = JobManager(
        workspace_root=r4,
        store=st4,
        worker_backend=lambda req: (
            (Path(req["workspace_dir"]) / "app.py").write_text("x = 2\n", encoding="utf-8") or {"text": "ok"}
        ),
    )
    try:
        smuggled = (
            "The initial Gemini Flash attempt did not complete all requirements. Implement this:\n"
            "def f(v0):\n"
            + "\n".join(f"    v{i} = compute_{i}(v{i - 1})" for i in range(1, 16))
            + "\n    return v15\n"
        )
        jm4.delegate_start(smuggled, project="p", model="gemini-4-argon", wait=True, auto_apply=False)
        verdict = verify_workspace_provenance(
            r4, st4, env={"CC_AGY_STRICT_MODE": "true", "CC_AGY_DATA_DIR": str(d4)}, strict=True
        )
        assert len(verdict["dictation_violations"]) == 1
    finally:
        jm4.shutdown()

    # P5: Possessive apostrophe ("Django's ORM") does NOT mask semicolon-packed Python statements
    packed_with_apostrophe = [
        "BaseExpression.get_refs() (Django's ORM): refs = set(); for expr in self.get_source_expressions(): refs |= expr.get_refs(); return refs.",
        "Query.trim_unused() (Django's ORM): used = set(); for a in self.annotation_select.values(): used |= a.get_refs(); self.annotations = {k: v for k, v in self.annotations.items() if k in used}; return used.",
    ]
    assert lint_delegation_prompt("\n".join(packed_with_apostrophe))["ok"] is False

    # P6: Reject `-c diff.external=...`, `--no-index`, and `--textconv` in _is_readonly_git_diff_command
    assert _is_readonly_git_diff_command("git -c diff.external=/tmp/evil.sh diff") is False
    assert _is_readonly_git_diff_command("git diff --no-index /etc/passwd /dev/null") is False
    assert _is_readonly_git_diff_command("git diff --textconv HEAD") is False
    assert _is_readonly_git_diff_command("git diff --stat HEAD") is True

    # P7: Spot-check marker resets cleanly across new sessions and ignores unapplied jobs
    d7 = tmp_path / "d_p7"
    (d7 / "receipts").mkdir(parents=True)
    r7 = tmp_path / "r_p7"
    _init_repo(r7)
    env7 = {"CC_AGY_STRICT_MODE": "1", "CC_AGY_DATA_DIR": str(d7), "CC_AGY_LEAD_READ_LIMIT": "2"}

    # Unapplied multi-file job does NOT unlock git diff spot-check
    (d7 / ".lead_session_started").write_text(json.dumps({"ts": time.time(), "session_id": "s0"}), encoding="utf-8")
    (d7 / ".lead_read_count").write_text("99", encoding="utf-8")
    time.sleep(0.02)
    (d7 / "receipts" / "job-unapplied.json").write_text(
        json.dumps({"kind": "delegate", "status": "ok", "applied": False, "files_changed": ["a.py", "b.py"]}),
        encoding="utf-8",
    )
    unapplied_dec = evaluate_pre_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": "git diff --stat"}, "cwd": str(r7)}, env=env7
    )
    assert unapplied_dec["hookSpecificOutput"]["permissionDecision"] == "deny"

    # Two consecutive sessions each get exactly 1 spot-check after an applied multi-file job
    for sess_idx in (1, 2):
        (d7 / ".lead_session_started").write_text(
            json.dumps({"ts": time.time(), "session_id": f"sess-{sess_idx}"}), encoding="utf-8"
        )
        time.sleep(0.02)
        (d7 / "receipts" / f"job-applied-{sess_idx}.json").write_text(
            json.dumps({"kind": "delegate", "status": "ok", "applied": True, "applied_files": ["a.py", "b.py"]}),
            encoding="utf-8",
        )
        (d7 / ".lead_read_count").write_text("99", encoding="utf-8")
        first = evaluate_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "git diff --stat"}, "cwd": str(r7)}, env=env7
        )
        second = evaluate_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "git diff"}, "cwd": str(r7)}, env=env7
        )
        assert first == {}
        assert second["hookSpecificOutput"]["permissionDecision"] == "deny"

    # Full guard also blocks `git -c diff.external=... diff` even on a fresh session
    (d7 / ".lead_session_started").write_text(json.dumps({"ts": time.time(), "session_id": "sess-evil"}), encoding="utf-8")
    evil_dec = evaluate_pre_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": "git -c diff.external=/tmp/evil.sh diff"}, "cwd": str(r7)},
        env=env7,
    )
    assert evil_dec["hookSpecificOutput"]["permissionDecision"] == "deny"

    # P8: All-skipped pytest run exits 0 with 0 passed, and run_acceptance marks it vacuous (passed=False)
    r8 = tmp_path / "r_p8"
    _init_repo(r8)
    (r8 / "test_s.py").write_text(
        "import pytest\n@pytest.mark.skip(reason='skip')\ndef test_only_skipped():\n    assert False\n",
        encoding="utf-8",
    )
    skip_res = run_acceptance([f"{sys.executable} -m pytest -q test_s.py"], r8)
    assert skip_res[0]["passed"] is False
    assert skip_res[0].get("vacuous") is True

    # P10: Concurrent job A cleanup never kills job B's live localharness with a different save_dir
    import os
    from server.agy_runner import _direct_child_pids

    lh_sym = tmp_path / "localharness"
    lh_sym.symlink_to("/bin/sleep")
    save_a = tmp_path / "save_a"
    save_b = tmp_path / "save_b"
    save_a.mkdir()
    save_b.mkdir()
    pre_a = _direct_child_pids(os.getpid())
    proc_b = subprocess.Popen([str(lh_sym), "60"], env={**os.environ, "CC_AGY_HARNESS_SAVE_DIR": str(save_b)})
    try:
        time.sleep(0.1)
        killed_cross = _force_terminate_agent_process(None, str(save_a), pre_child_pids=pre_a)
        assert killed_cross == 0
        assert proc_b.poll() is None
    finally:
        if proc_b.poll() is None:
            proc_b.kill()
            proc_b.wait(timeout=1.0)

    # P11: Backtick-wrapped semicolon-packed code in plan/checklist is also caught and repaired
    backtick_packed = [
        "BaseExpression.get_refs(): `refs = set(); for expr in self.get_source_expressions(): refs |= expr.get_refs(); return refs`.",
        "Query.trim_unused(): `used = set(); for a in self.annotation_select.values(): used |= a.get_refs(); self.annotations = {k: v for k, v in self.annotations.items() if k in used}; return used`.",
    ]
    assert lint_delegation_prompt("\n".join(backtick_packed))["ok"] is False
    rep_bt = repair_dictated_prompt("\n".join(backtick_packed))
    assert rep_bt["repaired"] is True
    assert lint_delegation_prompt(rep_bt["prompt"])["ok"] is True

    # P12: Reject `-C` and `--git-dir` in _is_readonly_git_diff_command so spot-check stays in-repo
    assert _is_readonly_git_diff_command("git -C /tmp/other diff") is False
    assert _is_readonly_git_diff_command("git --git-dir=/tmp/other/.git diff") is False


