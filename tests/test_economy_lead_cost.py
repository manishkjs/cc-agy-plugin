"""Regression tests for the economy Lead-cost fixes (Opus 5.5 + Flash 3.8 profile).

Covers:
- Flash gets 100 tool calls per turn in economy (75 elsewhere).
- An explicit continuation after a tool-call checkpoint gets a fresh call slice instead of the remainder.
- Economy gives a failed or capped Flash turn one plugin-side retry on the same model.
- `CC_AGY_LEAD_READ_LIMIT=0` means zero Lead reads (not "no cap"), and economy defaults to 0.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.pre_tool_guard import evaluate_pre_tool_use, resolve_lead_read_limit
from scripts.provenance_check import verify_workspace_provenance
from scripts.session_start import (
    BALANCED_SCOUT_GUIDANCE,
    ECONOMY_SCOUT_GUIDANCE,
    REVIEW_AFTER_APPLY_GUIDANCE,
    ZERO_READ_GUIDANCE,
    review_guidance_for,
)
from server.jobs import JobManager, resolve_budget
from server.receipts import ReceiptStore


def _init_repo(repo: Path, files: dict[str, str]) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)
    for rel, text in files.items():
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "-f", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)


@pytest.fixture
def clean_budget_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in (
        "CC_AGY_MAX_TOOL_CALLS",
        "CC_AGY_FLASH_MAX_TOOL_CALLS",
        "CC_AGY_COST_PROFILE",
        "CLAUDE_PLUGIN_OPTION_COST_PROFILE",
        "CC_AGY_USERCFG_COST_PROFILE",
        "CC_AGY_SAME_MODEL_RETRIES",
        "CC_AGY_ESCALATE_MODEL_ON_FAILURE",
    ):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Flash per-turn tool-call budget
# ---------------------------------------------------------------------------


def test_flash_gets_100_calls_in_economy_and_75_elsewhere(clean_budget_env: pytest.MonkeyPatch) -> None:
    assert resolve_budget(None, None, model="gemini-3.8-flash") == (55, 5_000_000)
    clean_budget_env.setenv("CC_AGY_COST_PROFILE", "economy")
    assert resolve_budget(None, None, model="gemini-3.8-flash") == (100, 5_000_000)
    assert resolve_budget(None, None, model="gemini-4-argon") == (100, 5_000_000)
    assert resolve_budget(None, None, model="gemini-4-carbon-high") == (100, 5_000_000)
    clean_budget_env.setenv("CC_AGY_FLASH_MAX_TOOL_CALLS", "30")
    assert resolve_budget(None, None, model="gemini-3.8-flash") == (30, 5_000_000)
    assert resolve_budget(12, None, model="gemini-3.8-flash") == (12, 5_000_000)


# ---------------------------------------------------------------------------
# Continuation after a checkpoint
# ---------------------------------------------------------------------------


def test_lead_continue_after_checkpoint_gets_full_slice_not_remainder(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    clean_budget_env.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "0")
    repo = tmp_path / "repo"
    _init_repo(repo, {"app.py": "x = 1\n"})
    seen: list[tuple[str, int | None]] = []

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        seen.append((req["model"], req.get("max_tool_calls")))
        (Path(req["workspace_dir"]) / "app.py").write_text(f"x = {len(seen) + 1}\n", encoding="utf-8")
        if len(seen) == 1:
            return {
                "text": "ran out of calls",
                "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
                "tool_calls": [{"name": "view_file", "args": {}}] * 39,
                "usage": {"prompt_token_count": 10, "candidates_token_count": 1, "total_token_count": 11},
            }
        return {"text": "done"}

    jm = JobManager(workspace_root=repo, store=ReceiptStore(tmp_path / "data", workspace_root=repo), worker_backend=worker)
    try:
        first = jm.delegate_start(
            "Update app.py", model="gemini-3.8-flash", project="p", wait=True, auto_apply=False, max_tool_calls=40
        )
        assert first["status"] == "checkpoint"
        jm.delegate_continue(first["job_id"], "Finish the edit.", wait=True, auto_apply=False)
        # Before the fix the continuation got 40 - 39 = 1 call.
        assert seen == [("gemini-3.8-flash", 40), ("gemini-3.8-flash", 40)]
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# Same-model retry in economy
# ---------------------------------------------------------------------------


_AUTH_TEST = (
    "import sys\n"
    "from auth import header\n"
    "sys.exit(0 if header() == 'ok' else 1)\n"
)


def _strict_economy_env(monkeypatch: pytest.MonkeyPatch, data_dir: Path, profile: str = "economy") -> None:
    monkeypatch.setenv("CC_AGY_STRICT_MODE", "1")
    monkeypatch.setenv("CC_AGY_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    monkeypatch.setenv("CC_AGY_AUTO_APPLY_ON_PASS", "1")
    monkeypatch.setenv("CC_AGY_COST_PROFILE", profile)


def test_economy_retries_failed_acceptance_once_on_same_model(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    data_dir = tmp_path / "data"
    _strict_economy_env(clean_budget_env, data_dir)
    models: list[str] = []

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        models.append(req["model"])
        value = "'wrong'" if len(models) == 1 else "'ok'"
        (Path(req["workspace_dir"]) / "auth.py").write_text(f"def header():\n    return {value}\n", encoding="utf-8")
        return {"text": f"attempt {len(models)}"}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        res = jm.delegate_start(
            "Make header() in auth.py return 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert models == ["gemini-3.8-flash", "gemini-3.8-flash"]
        assert res["model"] == "gemini-3.8-flash"
        assert res["applied"] is True
        raw = store.read_receipt(res["job_id"])
        assert raw["prompt_sources"] == ["lead", "plugin_retry"]
        verdict = verify_workspace_provenance(
            repo, store, env={"CC_AGY_STRICT_MODE": "true", "CC_AGY_DATA_DIR": str(data_dir)}, strict=True
        )
        assert verdict["dictation_violations"] == []
    finally:
        jm.shutdown()


def test_economy_same_model_retry_runs_at_most_once(tmp_path: Path, clean_budget_env: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    data_dir = tmp_path / "data"
    _strict_economy_env(clean_budget_env, data_dir)
    calls = 0

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        (Path(req["workspace_dir"]) / "auth.py").write_text("def header():\n    return 'nope'\n", encoding="utf-8")
        return {"text": "still wrong"}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        res = jm.delegate_start(
            "Make header() in auth.py return 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert calls == 2
        assert not res.get("applied")
        assert store.read_receipt(res["job_id"])["prompt_sources"] == ["lead", "plugin_retry"]
    finally:
        jm.shutdown()


def test_economy_retry_after_tool_call_checkpoint_gets_fresh_100_call_slice(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    data_dir = tmp_path / "data"
    _strict_economy_env(clean_budget_env, data_dir)
    seen: list[tuple[str, int | None]] = []

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        seen.append((req["model"], req.get("max_tool_calls")))
        wt = Path(req["workspace_dir"])
        if len(seen) == 1:
            (wt / "auth.py").write_text("def header():\n    return 'half'\n", encoding="utf-8")
            return {
                "text": "cap",
                "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
                "tool_calls": [{"name": "view_file", "args": {}}] * 99,
                "usage": {"prompt_token_count": 10, "candidates_token_count": 1, "total_token_count": 11},
            }
        (wt / "auth.py").write_text("def header():\n    return 'ok'\n", encoding="utf-8")
        return {"text": "finished"}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        res = jm.delegate_start(
            "Make header() in auth.py return 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        # [R4-1]: automatic plugin_retry must NOT fire on MAX_TOOL_CALLS_EXCEEDED checkpoint;
        # Lead delegate_continue then gets a fresh 100-call slice and auto-applies on pass.
        assert seen == [("gemini-3.8-flash", 100)]
        assert res["status"] == "checkpoint"
        assert not res.get("applied")
        cont = jm.delegate_continue(res["job_id"], "Finish header() -> 'ok'.", wait=True)
        assert seen == [("gemini-3.8-flash", 100), ("gemini-3.8-flash", 100)]
        assert cont["applied"] is True
        assert store.read_receipt(res["job_id"])["prompt_sources"] == ["lead", "lead"]
    finally:
        jm.shutdown()


def test_failed_plugin_retry_does_not_use_up_the_leads_continue(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    """Opus review blocker: a failed plugin retry must not trip the 2-failure escalate_to_lead threshold."""
    repo = tmp_path / "repo"
    _init_repo(repo, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    data_dir = tmp_path / "data"
    _strict_economy_env(clean_budget_env, data_dir)
    calls = 0

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        value = "'ok'" if calls == 3 else "'nope'"
        (Path(req["workspace_dir"]) / "auth.py").write_text(f"def header():\n    return {value}\n", encoding="utf-8")
        return {"text": f"attempt {calls}"}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        first = jm.delegate_start(
            "Make header() in auth.py return 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert calls == 2
        assert not (first.get("escalation") or {}).get("escalate_to_lead")
        assert store.read_receipt(first["job_id"])["failed_attempts"] == 1
        follow = jm.delegate_continue(first["job_id"], "header() must return the string 'ok'.", wait=True)
        assert follow.get("status") != "escalated"
        assert calls == 3
        assert follow.get("applied") is True
    finally:
        jm.shutdown()


def test_lead_continue_after_failed_plugin_retry_gets_full_slice(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    """Opus round-2 blocker: the retry's calls must not shrink the Lead's own continue to the remainder."""
    repo = tmp_path / "repo"
    _init_repo(repo, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    data_dir = tmp_path / "data"
    _strict_economy_env(clean_budget_env, data_dir)
    budgets: list[int | None] = []
    used_per_turn = [45, 54, 3]

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        budgets.append(req.get("max_tool_calls"))
        n = len(budgets)
        value = "'ok'" if n == 3 else "'nope'"
        (Path(req["workspace_dir"]) / "auth.py").write_text(f"def header():\n    return {value}\n", encoding="utf-8")
        return {"text": f"attempt {n}", "tool_calls": [{"name": "view_file", "args": {}}] * used_per_turn[n - 1]}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        first = jm.delegate_start(
            "Make header() in auth.py return 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert budgets == [100, 100]
        jm.delegate_continue(first["job_id"], "header() must return the string 'ok'.", wait=True)
        # Without the fix the Lead's continue got 100 - (45 + 54) = 1 call.
        assert budgets == [100, 100, 100]
    finally:
        jm.shutdown()


def test_balanced_without_escalation_does_not_same_model_retry(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    data_dir = tmp_path / "data"
    _strict_economy_env(clean_budget_env, data_dir, profile="balanced")
    clean_budget_env.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "0")
    calls = 0

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        (Path(req["workspace_dir"]) / "auth.py").write_text("def header():\n    return 'nope'\n", encoding="utf-8")
        return {"text": "wrong"}

    jm = JobManager(workspace_root=repo, store=ReceiptStore(data_dir, workspace_root=repo), worker_backend=worker)
    try:
        jm.delegate_start(
            "Make header() in auth.py return 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert calls == 1
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# Lead read limit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, 2),
        ({"CC_AGY_COST_PROFILE": "economy"}, 2),
        ({"CC_AGY_LEAD_READ_LIMIT": "0"}, 0),
        ({"CC_AGY_LEAD_READ_LIMIT": "0", "CC_AGY_COST_PROFILE": "economy"}, 0),
        ({"CC_AGY_LEAD_READ_LIMIT": "3"}, 3),
        ({"CC_AGY_MAX_LEAD_READS": "4"}, 4),
        ({"CC_AGY_LEAD_READ_LIMIT": "off"}, None),
        ({"CC_AGY_LEAD_READ_LIMIT": "-1"}, None),
        ({"CC_AGY_LEAD_READ_LIMIT": "junk"}, 2),
        ({"CC_AGY_LEAD_READ_LIMIT": "junk", "CC_AGY_COST_PROFILE": "economy"}, 2),
        ({"CC_AGY_LEAD_READ_LIMIT": "5", "CC_AGY_COST_PROFILE": "economy"}, 5),
    ],
)
def test_resolve_lead_read_limit(env: dict[str, str], expected: int | None) -> None:
    assert resolve_lead_read_limit(env) == expected


def _read(repo: Path, env: dict[str, str]) -> dict[str, Any]:
    return evaluate_pre_tool_use(
        {"tool_name": "Read", "tool_input": {"file_path": str(repo / "app.py")}, "cwd": str(repo)},
        env=env,
    )


def test_read_limit_zero_blocks_first_read_and_off_allows_it(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo, {"app.py": "x = 1\n"})
    base = {"CC_AGY_STRICT_MODE": "true", "CLAUDE_PROJECT_DIR": str(repo)}

    zero = {**base, "CC_AGY_LEAD_READ_LIMIT": "0", "CC_AGY_DATA_DIR": str(tmp_path / "d0")}
    dec = _read(repo, zero)
    assert dec["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "LeadReadBudgetExceeded" in dec["hookSpecificOutput"]["permissionDecisionReason"]

    economy = {**base, "CC_AGY_COST_PROFILE": "economy", "CC_AGY_DATA_DIR": str(tmp_path / "de")}
    assert _read(repo, economy) == {}
    assert _read(repo, economy) == {}
    assert _read(repo, economy)["hookSpecificOutput"]["permissionDecision"] == "deny"

    off = {**base, "CC_AGY_LEAD_READ_LIMIT": "off", "CC_AGY_DATA_DIR": str(tmp_path / "doff")}
    assert all(_read(repo, off) == {} for _ in range(5))

    balanced = {**base, "CC_AGY_DATA_DIR": str(tmp_path / "db")}
    assert _read(repo, balanced) == {}
    assert _read(repo, balanced) == {}
    assert _read(repo, balanced)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_read_limit_zero_blocks_bash_file_inspection(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo, {"app.py": "x = 1\n"})
    env = {
        "CC_AGY_STRICT_MODE": "true",
        "CLAUDE_PROJECT_DIR": str(repo),
        "CC_AGY_COST_PROFILE": "economy",
        "CC_AGY_DATA_DIR": str(tmp_path / "d"),
    }
    dec = evaluate_pre_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": f"cat {repo / 'app.py'}"}, "cwd": str(repo)},
        env=env,
    )
    assert dec["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_session_guidance_tells_economy_lead_to_delegate_without_reading() -> None:
    strict = {"CC_AGY_STRICT_MODE": "true"}
    assert review_guidance_for(strict) == REVIEW_AFTER_APPLY_GUIDANCE
    economy = review_guidance_for({**strict, "CC_AGY_COST_PROFILE": "economy"})
    assert economy == REVIEW_AFTER_APPLY_GUIDANCE + ECONOMY_SCOUT_GUIDANCE
    balanced = review_guidance_for({**strict, "CC_AGY_COST_PROFILE": "balanced"})
    assert balanced == REVIEW_AFTER_APPLY_GUIDANCE + BALANCED_SCOUT_GUIDANCE
    assert review_guidance_for({**strict, "CC_AGY_COST_PROFILE": "economy", "CC_AGY_LEAD_READ_LIMIT": "0"}) == (
        REVIEW_AFTER_APPLY_GUIDANCE + ZERO_READ_GUIDANCE
    )
    assert review_guidance_for({**strict, "CC_AGY_COST_PROFILE": "economy", "CC_AGY_LEAD_READ_LIMIT": "5"}) == (
        REVIEW_AFTER_APPLY_GUIDANCE
    )


# ---------------------------------------------------------------------------
# Round 1 Pilot Remediation Regressions
# ---------------------------------------------------------------------------


def test_running_plugin_retry_clears_stale_fields_updates_budget_and_shows_live_progress(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    import json
    import threading
    import time

    repo = tmp_path / "repo_retry_live"
    _init_repo(repo, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    data_dir = tmp_path / "data"
    _strict_economy_env(clean_budget_env, data_dir)
    clean_budget_env.setenv("CC_AGY_MAX_WAIT_SLICE_SECONDS", "1.8")

    attempt2_entered = threading.Event()
    release_attempt2 = threading.Event()
    store = ReceiptStore(data_dir, workspace_root=repo)
    calls = 0

    def slow_retry_worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        wt = Path(req["workspace_dir"])
        if calls == 1:
            (wt / "auth.py").write_text("def header():\n    return 'half'\n# attempt 1\n", encoding="utf-8")
            return {
                "text": "attempt 1 finished under 60 calls",
                "tool_calls": [{"name": "view_file", "args": {}}] * 40,
                "usage": {"prompt_token_count": 100, "candidates_token_count": 10, "total_token_count": 110},
            }
        prog_path = Path(req["progress_path"])
        live_path = Path(f"{prog_path}.live")
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_path.write_text(
            json.dumps({"straitjacket_calls": 14, "save_dir": "", "updated_at_epoch": time.time()}),
            encoding="utf-8",
        )
        attempt2_entered.set()
        release_attempt2.wait(timeout=5.0)
        (wt / "auth.py").write_text("def header():\n    return 'ok'\n# attempt 2 complete\n", encoding="utf-8")
        return {
            "text": "attempt 2 done",
            "tool_calls": [{"name": "edit_file", "args": {}}] * 15,
            "usage": {"prompt_token_count": 120, "candidates_token_count": 12, "total_token_count": 132},
        }

    jm = JobManager(workspace_root=repo, store=store, worker_backend=slow_retry_worker)
    try:
        mid = jm.delegate_start(
            "Ensure header() returns 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        # Acceptance now snapshots and checks the change set around every command (a few git calls),
        # so allow more than 3 s on a loaded machine; the event still has to fire.
        assert attempt2_entered.wait(timeout=10.0)
        if mid["status"] != "running":
            pytest.fail(f"Expected wait-sliced Attempt 2 to still be running, got {mid}")

        assert mid["attempt"] == 2
        assert mid["prompt_sources"] == ["lead", "plugin_retry"]
        assert mid["stop_reason"] is None
        assert mid["acceptance_passed"] is None
        assert mid["acceptance_results"] == []
        assert mid["summary"] == ""
        assert mid["budget"]["max_tool_calls"] == 100
        assert mid["cumulative_max_tool_calls"] == 140
        assert mid["tool_call_count"] == 54
        assert "Attempt 2 (plugin_retry) is actively running (54/140 tool calls used)" in mid["poll_hint"]

        status_view = jm.delegate_status(mid["job_id"])
        assert status_view["status"] == "running"
        assert status_view["attempt"] == 2
        assert status_view["stop_reason"] is None
        assert status_view["acceptance_passed"] is None
        assert status_view["tool_call_count"] == 54
        all_status = jm.delegate_status()
        assert all_status["total_jobs"] == 1
        assert [r["job_id"] for r in store.list_receipts(workspace_root=repo)] == [mid["job_id"]]

        release_attempt2.set()
        clean_budget_env.setenv("CC_AGY_MAX_WAIT_SLICE_SECONDS", "10")
        final = jm.delegate_result(mid["job_id"], wait=True, timeout=5.0)
        assert final["status"] == "ok"
        assert final["applied"] is True
        assert final["acceptance_passed"] is True
        raw_rec = store.read_receipt(mid["job_id"])
        assert raw_rec["budget"]["max_tool_calls"] == 100
        assert raw_rec["cumulative_max_tool_calls"] == 140
        assert raw_rec["prior_attempts"][0]["summary"] == "attempt 1 finished under 60 calls"
        assert len(raw_rec["prior_attempts"][0]["acceptance_results"]) == 1
    finally:
        release_attempt2.set()
        jm.shutdown()


def test_lead_review_read_bonus_and_host_bash_blocked_during_running_cancelled_and_auto_discovered_pass(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    import json
    import time
    from scripts.pre_tool_guard import _lead_review_read_bonus

    repo = tmp_path / "repo_guard"
    _init_repo(repo, {"pkg.ts": "export const x = 1;\n"})
    data_dir = tmp_path / "data_guard"
    store = ReceiptStore(data_dir, workspace_root=repo)
    (data_dir / ".lead_session_started").write_text(
        json.dumps({"session_id": "s-guard", "ts": time.time() - 5.0}),
        encoding="utf-8",
    )
    env = {
        "CC_AGY_STRICT_MODE": "true",
        "CC_AGY_COST_PROFILE": "economy",
        "CLAUDE_PROJECT_DIR": str(repo),
        "CC_AGY_DATA_DIR": str(data_dir),
    }

    # 1. Running receipt must NOT unlock +6 reads or allow host test runners
    store.write_receipt("job-1000-aaaaaaaa", {"job_id": "job-1000-aaaaaaaa", "status": "running"}, workspace_root=str(repo))
    assert _lead_review_read_bonus(data_dir, env) == 0
    read_dec = evaluate_pre_tool_use(
        {"tool_name": "Read", "tool_input": {"file_path": str(repo / "pkg.ts")}, "cwd": str(repo)},
        env=env,
    )
    assert read_dec["hookSpecificOutput"]["permissionDecision"] == "deny"
    bash_running = evaluate_pre_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": ".venv/bin/pytest -q tests"}, "cwd": str(repo)},
        env=env,
    )
    assert bash_running["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "still running" in bash_running["hookSpecificOutput"]["permissionDecisionReason"]

    # 2. Superseded cancelled receipt alone must NOT unlock +6 reads
    store.write_receipt(
        "job-1000-aaaaaaaa",
        {"job_id": "job-1000-aaaaaaaa", "status": "cancelled", "cancelled": True},
        workspace_root=str(repo),
    )
    assert _lead_review_read_bonus(data_dir, env) == 0

    # 3. Unapplied failing terminal receipt in Economy blocks host pytest (prevents host artifact pollution)
    store.write_receipt(
        "job-1001-bbbbbbbb",
        {
            "job_id": "job-1001-bbbbbbbb",
            "status": "ok",
            "applied": False,
            "files_changed": ["pkg.ts"],
            "acceptance_passed": False,
            "acceptance_results": [{"command": "pnpm test", "passed": False, "exit_code": 1}],
        },
        workspace_root=str(repo),
    )
    assert _lead_review_read_bonus(data_dir, env) == 6
    bash_unapplied = evaluate_pre_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": ".venv/bin/pytest -q tests"}, "cwd": str(repo)},
        env=env,
    )
    assert bash_unapplied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "no worker diff has been applied" in bash_unapplied["hookSpecificOutput"]["permissionDecisionReason"]

    # 4. Applied clean auto_discovered_default suite with real passing tests (>0, not compile_smoke) returns 0 bonus
    time.sleep(0.02)
    store.write_receipt(
        "job-1002-cccccccc",
        {
            "job_id": "job-1002-cccccccc",
            "status": "ok",
            "applied": True,
            "files_changed": ["pkg.ts"],
            "acceptance_passed": True,
            "acceptance_results": [
                {
                    "command": "pnpm test",
                    "passed": True,
                    "auto_discovered_default": True,
                    "counts": {"passed": 35, "failed": 0, "errors": 0},
                }
            ],
        },
        workspace_root=str(repo),
    )
    assert _lead_review_read_bonus(data_dir, env) == 0
    bash_post_apply = evaluate_pre_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": "cd packages/core && ./node_modules/.bin/vitest run"}, "cwd": str(repo)},
        env=env,
    )
    assert bash_post_apply["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "already verified" in bash_post_apply["hookSpecificOutput"]["permissionDecisionReason"]

    # 5. compile_smoke (0 tests) and empty acceptance_results (vacuous) STILL grant +6 review reads
    time.sleep(0.02)
    store.write_receipt(
        "job-1003-dddddddd",
        {
            "job_id": "job-1003-dddddddd",
            "status": "ok",
            "applied": True,
            "files_changed": ["pkg.ts"],
            "acceptance_passed": True,
            "acceptance_results": [
                {
                    "command": "bash -n deploy.sh",
                    "passed": True,
                    "auto_discovered_default": True,
                    "compile_smoke": True,
                    "counts": {"passed": 0, "failed": 0, "errors": 0},
                }
            ],
        },
        workspace_root=str(repo),
    )
    assert _lead_review_read_bonus(data_dir, env) == 6
    time.sleep(0.02)
    store.write_receipt(
        "job-1004-eeeeeeee",
        {
            "job_id": "job-1004-eeeeeeee",
            "status": "ok",
            "applied": True,
            "files_changed": ["pkg.ts"],
            "acceptance_passed": True,
            "vacuous_acceptance": True,
            "acceptance_results": [],
        },
        workspace_root=str(repo),
    )
    assert _lead_review_read_bonus(data_dir, env) == 6


def test_pnpm_monorepo_subpackage_symlink_preserves_sibling_virtual_store_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import shutil
    import subprocess
    import server.jobs as jobs_mod
    from server.jobs import link_host_node_modules_farm, unlink_host_node_modules_farm

    def _build_real_pnpm_store(pnpm_root: Path) -> Path:
        vitest_nm = pnpm_root / "vitest@4.0.13" / "node_modules"
        utils_nm = pnpm_root / "@vitest+utils@4.0.13" / "node_modules"
        (vitest_nm / "vitest").mkdir(parents=True)
        (vitest_nm / "@vitest").mkdir(parents=True)
        (utils_nm / "@vitest" / "utils").mkdir(parents=True)
        (utils_nm / "@vitest" / "utils" / "index.js").write_text(
            "module.exports = { ok: 42 };\n",
            encoding="utf-8",
        )
        os.symlink(
            "../../../@vitest+utils@4.0.13/node_modules/@vitest/utils",
            str(vitest_nm / "@vitest" / "utils"),
        )
        (vitest_nm / "vitest" / "index.js").write_text(
            "module.exports = require('@vitest/utils');\n",
            encoding="utf-8",
        )
        return vitest_nm

    host = tmp_path / "pnpm_host"
    (host / "packages" / "core").mkdir(parents=True)
    _build_real_pnpm_store(host / "node_modules" / ".pnpm")
    sub_nm = host / "packages" / "core" / "node_modules"
    sub_nm.mkdir(parents=True)
    os.symlink(
        "../../../node_modules/.pnpm/vitest@4.0.13/node_modules/vitest",
        str(sub_nm / "vitest"),
    )

    wt = tmp_path / "pnpm_wt"
    (wt / "packages" / "core").mkdir(parents=True)
    created = link_host_node_modules_farm(host, wt)
    try:
        wt_vitest = (wt / "packages" / "core" / "node_modules" / "vitest").resolve()
        assert wt_vitest.is_relative_to(wt.resolve())
        assert ".cc-agy-ext-store" not in wt_vitest.as_posix()
        assert (wt_vitest.parent / "@vitest" / "utils" / "index.js").is_file()
    finally:
        unlink_host_node_modules_farm(created)

    # Case 2: external .pnpm store outside host_root copies transitive sibling .pnpm/<dep>/node_modules closure
    ext_vitest_nm = _build_real_pnpm_store(tmp_path / "external_pnpm_store" / ".pnpm")
    host2 = tmp_path / "pnpm_ext_host"
    (host2 / "packages" / "core" / "node_modules").mkdir(parents=True)
    os.symlink(str(ext_vitest_nm / "vitest"), str(host2 / "packages" / "core" / "node_modules" / "vitest"))
    wt2 = tmp_path / "pnpm_ext_wt"
    (wt2 / "packages" / "core").mkdir(parents=True)
    created2 = link_host_node_modules_farm(host2, wt2, allow_shallow_symlinks=False)
    try:
        wt2_vitest = (wt2 / "packages" / "core" / "node_modules" / "vitest").resolve()
        assert wt2_vitest.is_relative_to(wt2.resolve())
        sibling_utils = wt2_vitest.parent / "@vitest" / "utils"
        assert sibling_utils.is_symlink() and sibling_utils.exists()
        assert (sibling_utils.resolve() / "index.js").is_file()
        node_bin = shutil.which("node")
        if node_bin:
            proc = subprocess.run(
                [node_bin, "-e", "const v = require('./packages/core/node_modules/vitest'); if (v.ok !== 42) process.exit(2);"],
                cwd=str(wt2),
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stderr
    finally:
        unlink_host_node_modules_farm(created2)

    # Case 3: oversized root node_modules (skipped when allow_shallow_symlinks=False, no CoW) still resolves subpackage
    host3 = tmp_path / "pnpm_big_host"
    (host3 / "packages" / "core" / "node_modules").mkdir(parents=True)
    _build_real_pnpm_store(host3 / "node_modules" / ".pnpm")
    for idx_d in range(25):
        (host3 / "node_modules" / f"extra_{idx_d}.txt").write_text("x", encoding="utf-8")
    os.symlink(
        "../../../node_modules/.pnpm/vitest@4.0.13/node_modules/vitest",
        str(host3 / "packages" / "core" / "node_modules" / "vitest"),
    )
    wt3 = tmp_path / "pnpm_big_wt"
    (wt3 / "packages" / "core").mkdir(parents=True)
    monkeypatch.setattr(jobs_mod, "_NM_COPY_MAX_FILES", 20)
    monkeypatch.setattr(jobs_mod, "_fs_supports_cow_between", lambda _a, _b: False)
    created3 = link_host_node_modules_farm(host3, wt3, allow_shallow_symlinks=False)
    try:
        assert not (wt3 / "node_modules").exists()
        wt3_vitest = (wt3 / "packages" / "core" / "node_modules" / "vitest").resolve()
        assert wt3_vitest.is_relative_to(wt3.resolve())
        assert (wt3_vitest.parent / "@vitest" / "utils" / "index.js").is_file()
    finally:
        unlink_host_node_modules_farm(created3)


def test_validate_acceptance_normalizes_bare_go_build_and_blocks_workspace_binary_output(tmp_path: Path) -> None:
    from server.jobs import _resolve_pytest_runner_for_subdir, validate_acceptance

    repo = tmp_path / "go_repo"
    _init_repo(repo, {"go.mod": "module example.com/m\ngo 1.21\n"})
    normalized = validate_acceptance(
        [
            "go build ./...",
            "CGO_ENABLED=0 go build ./cmd/gh",
            "GOFLAGS=$EXTRA go build ./... 2>&1",
            "(cd cmd && go build ./...)",
            "go -C . build .",
        ],
        repo,
    )
    assert normalized == [
        "go build -o /dev/null ./...",
        "CGO_ENABLED=0 go build -o /dev/null ./cmd/gh",
        "GOFLAGS=$EXTRA go build -o /dev/null ./... 2>&1",
        "(cd cmd && go build -o /dev/null ./...)",
        "go -C . build -o /dev/null .",
    ]
    for bad_cmd in (
        "go build -o ./bin/gh ./cmd/gh",
        "go build --o=pwned .",
        "go build --o pwned .",
        "go -C . build -o ./pwned .",
    ):
        with pytest.raises(ValueError, match="refused"):
            validate_acceptance([bad_cmd], repo)

    venv_pytest = repo / ".venv" / "bin" / "pytest"
    venv_pytest.parent.mkdir(parents=True)
    venv_pytest.write_text("#!/bin/sh\n", encoding="utf-8")
    resolved_multi_env = _resolve_pytest_runner_for_subdir("A=1 B=2 .venv/bin/pytest", repo)
    assert resolved_multi_env == f"A=1 B=2 {venv_pytest}"


def test_round2_economy_cancel_prestart_bash_guard_cross_session_shutdown_and_setup_wait_budget(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    import json
    import threading
    import time

    repo = tmp_path / "r2_repo"
    _init_repo(repo, {"app.py": "def run():\n    return 'orig'\n"})
    data_dir = tmp_path / "r2_data"
    store = ReceiptStore(data_dir, workspace_root=repo)
    (data_dir / ".lead_session_started").write_text(
        json.dumps({"session_id": "s-r2", "ts": time.time() - 5.0}),
        encoding="utf-8",
    )
    env = {
        "CC_AGY_STRICT_MODE": "true",
        "CC_AGY_COST_PROFILE": "economy",
        "CLAUDE_PROJECT_DIR": str(repo),
        "CC_AGY_DATA_DIR": str(data_dir),
    }

    # 1. Pre-delegate_start in Economy blocks host test runners (and package-manager/shell -c variants), allows git log
    for runner_cmd in (
        ".venv/bin/pytest -q tests",
        "pnpm --filter @koota/core test",
        "pnpm -C packages/core test",
        "npm t",
        "python3 -B -m unittest",
        "bash -c 'pytest -q'",
        "go -C sub test ./...",
    ):
        dec = evaluate_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": runner_cmd}, "cwd": str(repo)},
            env=env,
        )
        assert dec.get("hookSpecificOutput", {}).get("permissionDecision") == "deny", f"Expected deny for {runner_cmd!r}"

    git_log_ok = evaluate_pre_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": "git status --short"}, "cwd": str(repo)},
        env=env,
    )
    assert git_log_ok == {}

    # 2. Exact igel scenario: after delegate_cancel (only cancelled receipt in session), host pytest STILL blocked
    store.write_receipt(
        "job-2000-cancelled",
        {"job_id": "job-2000-cancelled", "status": "cancelled", "cancelled": True},
        workspace_root=str(repo),
    )
    after_cancel = evaluate_pre_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": ".venv/bin/pytest -q tests"}, "cwd": str(repo)},
        env=env,
    )
    assert after_cancel["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "no worker diff has been applied" in after_cancel["hookSpecificOutput"]["permissionDecisionReason"]

    # 3. Cross-session JobManager shutdown isolation + _shutting_down entry refusal
    a_entered = threading.Event()
    release_a = threading.Event()

    def session_a_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "app.py").write_text("def run():\n    return 'half-done'\n", encoding="utf-8")
        a_entered.set()
        release_a.wait(timeout=5.0)
        (wt / "app.py").write_text("def run():\n    return 'complete'\n", encoding="utf-8")
        return {"text": "done"}

    clean_budget_env.setenv("CC_AGY_SALVAGE_ON_SHUTDOWN", "1")
    jm_a = JobManager(workspace_root=repo, store=store, worker_backend=session_a_worker)
    jm_b = JobManager(workspace_root=repo, store=store, worker_backend=lambda _r: {"text": "noop"})
    try:
        started_a = jm_a.delegate_start("Finish run()", project="p", wait=False)
        assert a_entered.wait(timeout=3.0)
        # Session B shutting down must NOT interrupt or salvage Session A's active job
        assert jm_b.mark_running_interrupted() == []
        with pytest.raises(RuntimeError, match="shutting down"):
            jm_b.delegate_start("Should refuse after shutdown", project="p", wait=False)
        assert store.read_receipt(started_a["job_id"])["status"] == "running"
        assert (repo / "app.py").read_text(encoding="utf-8") == "def run():\n    return 'orig'\n"
        release_a.set()
        fin_a = jm_a.delegate_result(started_a["job_id"], wait=True, timeout=5.0)
        assert fin_a["status"] == "ok"
    finally:
        release_a.set()
        jm_a.shutdown()
        jm_b.shutdown()

    # 4. RPC-entry wait clock accounts for pre-submit setup time (probe_baseline_acceptance)
    clean_budget_env.setenv("CC_AGY_MAX_WAIT_SLICE_SECONDS", "1.2")
    seen_slices: list[float] = []
    jm_time = JobManager(workspace_root=repo, store=store, worker_backend=lambda _r: {"text": "fast"})
    orig_wait = jm_time._wait_for_future_or_cancel

    def _record_wait(jid: str, fut: Any, timeout: float | None = None) -> None:
        if timeout is not None:
            seen_slices.append(float(timeout))
        return orig_wait(jid, fut, timeout=timeout)

    clean_budget_env.setattr(jm_time, "_wait_for_future_or_cancel", _record_wait)
    try:
        jm_time.delegate_start(
            "Check setup deduction",
            project="p",
            acceptance=["python3 -c \"import time; time.sleep(0.45)\""],
            wait=True,
        )
        assert seen_slices, "Expected wait_for_future_or_cancel to record wait_slice"
        assert seen_slices[0] < 0.9, f"Expected setup time (~0.45s) deducted from 1.2s slice, got {seen_slices[0]}"
    finally:
        jm_time.shutdown()


def test_subdir_cwd_pytest_probe_resolves_repo_relative_venv_runner(tmp_path: Path) -> None:
    import sys
    from server.jobs import evaluate_post_worker_baseline_failures, probe_baseline_acceptance

    repo = tmp_path / "igel_like_repo"
    _init_repo(repo, {"igel.py": "X = 1\n"})
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    wrapper = venv_bin / "pytest"
    wrapper.write_text(f"#!{sys.executable}\nimport pytest, sys\nraise SystemExit(pytest.main(sys.argv[1:]))\n", encoding="utf-8")
    wrapper.chmod(0o755)

    sub_tests = repo / "tests" / "test_igel"
    sub_tests.mkdir(parents=True)
    (sub_tests / "data.txt").write_text("ok\n", encoding="utf-8")
    (sub_tests / "test_cwd.py").write_text(
        "from pathlib import Path\n"
        "def test_requires_subdir_cwd():\n"
        "    assert Path('data.txt').is_file()\n",
        encoding="utf-8",
    )

    probes = probe_baseline_acceptance([".venv/bin/pytest -q tests"], repo)
    assert len(probes) == 1
    assert probes[0]["passed"] is False
    assert probes[0].get("subdir_cwd_passed") is True
    assert probes[0].get("passing_cwd") == "tests/test_igel"

    post_results = [
        {
            "command": ".venv/bin/pytest -q tests",
            "passed": False,
            "exit_code": 1,
            "timed_out": False,
            "output_tail": "FAILED tests/test_igel/test_cwd.py::test_requires_subdir_cwd - AssertionError\n1 failed in 0.05s",
        }
    ]
    warns = evaluate_post_worker_baseline_failures(post_results, probes, repo, ["igel.py"])
    assert post_results[0]["passed"] is True
    assert post_results[0]["preexisting_baseline_failure_ignored"] is True
    assert warns


def test_live_stdin_eof_cancels_worker_promptly_and_salvages_worktree(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    import io
    import json
    import sys
    import threading
    import time
    from server.mcp_stdio import MCPServer, serve_stdio

    repo = tmp_path / "eof_repo"
    _init_repo(repo, {"feature.py": "val = 0\n"})
    data_dir = tmp_path / "eof_data"
    _strict_economy_env(clean_budget_env, data_dir)

    worker_started = threading.Event()
    cancel_called = threading.Event()

    def hanging_worker(req: dict[str, Any]) -> dict[str, Any]:
        reg_cancel = req.get("register_cancel")
        if callable(reg_cancel):
            reg_cancel(lambda: cancel_called.set())
        wt = Path(req["workspace_dir"])
        (wt / "feature.py").write_text("val = 42\n", encoding="utf-8")
        worker_started.set()
        cancel_called.wait(timeout=10.0)
        raise TimeoutError("cancelled on stdin EOF")

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=hanging_worker)
    mcp = MCPServer(job_manager=jm)

    r_fd, w_fd = __import__("os").pipe()
    r_raw = __import__("os").fdopen(r_fd, "rb")
    w_raw = __import__("os").fdopen(w_fd, "wb")

    class _FakeStdin:
        def __init__(self, buf: Any) -> None:
            self.buffer = buf

    clean_budget_env.setattr(sys, "stdin", _FakeStdin(r_raw))
    out_buf = io.BytesIO()

    rpc_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "delegate_start",
            "arguments": {"prompt": "Set val = 42 in feature.py", "model": "gemini-3.8-flash", "project": "p", "wait": True},
        },
    }
    w_raw.write((json.dumps(rpc_req) + "\n").encode("utf-8"))
    w_raw.flush()

    server_done = threading.Event()

    def _run_server() -> None:
        try:
            serve_stdio(server=mcp, in_stream=None, out_stream=out_buf)
        finally:
            server_done.set()

    t = threading.Thread(target=_run_server, daemon=True)
    t.start()
    try:
        assert worker_started.wait(timeout=4.0)
        eof_started = time.monotonic()
        w_raw.close()
        assert server_done.wait(timeout=4.0), "serve_stdio blocked on live stdin EOF instead of cancelling worker!"
        assert (time.monotonic() - eof_started) < 4.0
        assert cancel_called.is_set()
        recs = store.list_receipts(workspace_root=repo)
        assert len(recs) == 1
        assert recs[0]["status"] == "interrupted"
        assert recs[0].get("salvaged_on_shutdown") is True
        assert recs[0].get("applied") is True
        assert (repo / "feature.py").read_text(encoding="utf-8") == "val = 42\n"
    finally:
        try:
            w_raw.close()
        except Exception:
            pass
        try:
            r_raw.close()
        except Exception:
            pass
        jm.shutdown(wait=False)



# ---------------------------------------------------------------------------
# Round 2 Full-26 Generic Remediation Regressions (R2-1 .. R2-5 + S1 .. S4)
# ---------------------------------------------------------------------------


def test_plugin_retry_timeout_capped_to_half_remaining_session_budget_and_skips_past_55pct(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    import time

    repo = tmp_path / "r2_timeout_repo"
    _init_repo(repo, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    data_dir = tmp_path / "r2_timeout_data"
    _strict_economy_env(clean_budget_env, data_dir)
    clean_budget_env.setenv("CC_AGY_SESSION_DEADLINE_SECONDS", "900")
    clean_budget_env.setenv("CC_AGY_DEADLINE_RESERVE_SECONDS", "100")

    seen_timeouts: list[float] = []

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        seen_timeouts.append(float(req["timeout_seconds"]))
        (Path(req["workspace_dir"]) / "auth.py").write_text("def header():\n    return 'nope'\n", encoding="utf-8")
        return {"text": "wrong"}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        # Simulate 400s elapsed out of 900s (< 55% = 495s), leaving 900 - 100 - 400 = 400s worker budget.
        # plugin_retry should cap worker_timeout to 0.5 * 400s = ~200s (leaving ~200s for Lead's Attempt 3).
        jm._created_at = time.monotonic() - 400.0
        jm.delegate_start(
            "Make header() return 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            timeout_seconds=600.0,
            wait=True,
        )
        assert len(seen_timeouts) == 2
        assert 190.0 <= seen_timeouts[1] <= 205.0

        # Simulate 520s elapsed (> 55% of 900s = 495s): plugin_retry must be skipped so Lead keeps remaining time!
        seen_timeouts.clear()
        store_late = ReceiptStore(tmp_path / "r2_timeout_data_late", workspace_root=repo)
        jm_late = JobManager(workspace_root=repo, store=store_late, worker_backend=worker)
        try:
            jm_late._created_at = time.monotonic() - 520.0
            res_late = jm_late.delegate_start(
                "Make header() return 'ok'.",
                model="gemini-3.8-flash",
                project="p",
                acceptance=["python3 test_auth.py"],
                timeout_seconds=600.0,
                wait=True,
            )
            assert len(seen_timeouts) == 1
            assert store_late.read_receipt(res_late["job_id"])["prompt_sources"] == ["lead"]
        finally:
            jm_late.shutdown()
    finally:
        jm.shutdown()


def test_escalate_to_lead_omits_cancel_unlocks_up_to_3_edits_and_passes_provenance(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    from scripts.session_start import handle_session_start

    repo = tmp_path / "r2_esc_repo"
    _init_repo(repo, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    data_dir = tmp_path / "r2_esc_data"
    _strict_economy_env(clean_budget_env, data_dir)
    store = ReceiptStore(data_dir, workspace_root=repo)
    env = {
        "CC_AGY_STRICT_MODE": "true",
        "CC_AGY_COST_PROFILE": "economy",
        "CC_AGY_DATA_DIR": str(data_dir),
        "CLAUDE_PROJECT_DIR": str(repo),
    }
    handle_session_start({"cwd": str(repo), "session_id": "s-esc", "source": "startup"}, env=env, store=store)

    # Before escalation, workspace Edit is strictly blocked.
    pre_edit = evaluate_pre_tool_use(
        {"tool_name": "Edit", "tool_input": {"file_path": str(repo / "auth.py")}, "cwd": str(repo)},
        env=env,
    )
    assert pre_edit["hookSpecificOutput"]["permissionDecision"] == "deny"

    def failing_worker(req: dict[str, Any]) -> dict[str, Any]:
        (Path(req["workspace_dir"]) / "auth.py").write_text("def header():\n    return 'almost'\n", encoding="utf-8")
        return {"text": "partial"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=failing_worker)
    try:
        r1 = jm.delegate_start(
            "Make header() return 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            allowed_paths=["auth.py"],
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        r2 = jm.delegate_continue(r1["job_id"], "Still returning 'almost'; return 'ok'.", wait=True)
        esc = r2.get("escalation") or {}
        assert esc.get("escalate_to_lead") is True
        rec_text = str(esc.get("recommendation") or "")
        assert "Do NOT call delegate_cancel" in rec_text
        assert "or delegate_cancel and start over" not in rec_text

        jm.delegate_apply(r1["job_id"], allow_scope_violations=True)

        # Up to 3 targeted Edit calls on the applied file are unlocked; Write remains blocked; 4th Edit is blocked.
        for _ in range(3):
            dec = evaluate_pre_tool_use(
                {"tool_name": "Edit", "tool_input": {"file_path": str(repo / "auth.py")}, "cwd": str(repo)},
                env=env,
            )
            assert dec == {}
        dec_4th = evaluate_pre_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": str(repo / "auth.py")}, "cwd": str(repo)},
            env=env,
        )
        assert dec_4th["hookSpecificOutput"]["permissionDecision"] == "deny"
        dec_write = evaluate_pre_tool_use(
            {"tool_name": "Write", "tool_input": {"file_path": str(repo / "auth.py")}, "cwd": str(repo)},
            env=env,
        )
        assert dec_write["hookSpecificOutput"]["permissionDecision"] == "deny"

        # Simulate the Lead landing its small surgical fix on auth.py and verify Stop-hook provenance passes.
        (repo / "auth.py").write_text("def header():\n    return 'ok'\n", encoding="utf-8")
        verdict = verify_workspace_provenance(repo, store=store, env=env, strict=True, session_id="s-esc")
        assert verdict["ok"] is True, verdict
    finally:
        jm.shutdown()


def test_delegate_cancel_preserves_diff_and_salvages_unapplied_work_ignoring_dead_pid(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "r2_cancel_repo"
    _init_repo(repo, {"lib.py": "x = 1\n"})
    data_dir = tmp_path / "r2_cancel_data"
    _strict_economy_env(clean_budget_env, data_dir)

    call_n = 0

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_n
        call_n += 1
        (Path(req["workspace_dir"]) / "lib.py").write_text(f"x = {98 + call_n}\n", encoding="utf-8")
        return {"text": f"partial {call_n}"}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        res = jm.delegate_start(
            "Set x = 99 in lib.py",
            model="gemini-3.8-flash",
            project="p",
            auto_apply=False,
            wait=True,
        )
        jid = res["job_id"]
        jm.delegate_cancel(jid, cleanup_worktree=True)
        assert store.diff_path(jid).exists()
        assert "x = 99" in store.diff_path(jid).read_text(encoding="utf-8")
        # User-cancelled jobs must never be silently salvaged on shutdown.
        assert jm.salvage_unapplied_worktree(owner_instance_id=jm._instance_id) is None

        # Write a stale "running" receipt owned by a guaranteed-dead PID (S1) and a live unapplied job —
        # salvage must ignore the dead-PID running receipt and salvage the unapplied job.
        stale_id = "job-1000000000000-deadbeef"
        store.write_receipt(
            stale_id,
            {
                "job_id": stale_id,
                "kind": "delegate",
                "status": "running",
                "owner_instance_id": "inst-99999999-deadbeef",
                "files_changed": [],
                "applied": False,
            },
            workspace_root=str(repo),
        )
        res2 = jm.delegate_start(
            "Set x = 100 in lib.py",
            model="gemini-3.8-flash",
            project="p",
            auto_apply=False,
            wait=True,
        )
        assert res2["applied"] is False

        salvaged = jm.salvage_unapplied_worktree(owner_instance_id=jm._instance_id)
        assert salvaged is not None
        assert salvaged["applied"] is True
        assert (repo / "lib.py").read_text(encoding="utf-8") == "x = 100\n"
    finally:
        jm.shutdown()


def test_delegate_continue_inlines_modified_impl_files_on_fresh_turn(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "r2_cont_inline"
    _init_repo(repo, {"mod.py": "def calc():\n    return 0\n", "test_mod.py": "from mod import calc\nassert calc() == 2\n"})
    data_dir = tmp_path / "r2_cont_data"
    _strict_economy_env(clean_budget_env, data_dir)
    clean_budget_env.setenv("CC_AGY_SAME_MODEL_RETRIES", "0")
    prompts_seen: list[str] = []

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        prompts_seen.append(str(req["prompts"][0]))
        wt = Path(req["workspace_dir"])
        if len(prompts_seen) == 1:
            (wt / "mod.py").write_text("def calc():\n    return 1  # unique_marker_turn1\n", encoding="utf-8")
            return {"text": "turn 1"}
        (wt / "mod.py").write_text("def calc():\n    return 2\n", encoding="utf-8")
        return {"text": "turn 2"}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        r1 = jm.delegate_start(
            "Make calc() return 2.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_mod.py"],
            wait=True,
        )
        r2 = jm.delegate_continue(r1["job_id"], "Fix calc() to return 2.", wait=True)
        assert r2["applied"] is True
        assert "unique_marker_turn1" in prompts_seen[1]
    finally:
        jm.shutdown()


def test_discover_related_test_commands_and_exit4_pytest_remap(tmp_path: Path) -> None:
    from server.jobs import discover_related_test_commands, remap_nonexistent_pytest_acceptance_cmds

    repo = tmp_path / "r2_pkg_tests"
    _init_repo(
        repo,
        {
            "src/psd_tools/psd/layer_and_mask.py": "x = 1\n",
            "tests/psd_tools/psd/test_layer_and_mask.py": "def test_lm(): pass\n",
            "tests/psd_tools/psd/test_psd.py": "def test_psd_roundtrip(): pass\n",
            "IPython/core/magic.py": "m = 1\n",
            "tests/test_magic.py": "def test_m(): pass\n",
        },
    )
    cmds = discover_related_test_commands(repo, ["src/psd_tools/psd/layer_and_mask.py"], [])
    assert len(cmds) == 1
    assert "tests/psd_tools/psd/test_layer_and_mask.py" in cmds[0]
    assert "tests/psd_tools/psd/test_psd.py" in cmds[0]

    probe = [
        {
            "command": "pytest IPython/core/tests/test_magic.py -q",
            "exit_code": 4,
            "passed": False,
            "output_tail": "ERROR: file or directory not found: IPython/core/tests/test_magic.py",
        }
    ]
    remapped, changed = remap_nonexistent_pytest_acceptance_cmds(
        ["pytest IPython/core/tests/test_magic.py -q"],
        repo,
        probe,
        allowed_paths=["IPython/core/magic.py"],
    )
    assert changed is True
    assert remapped == ["pytest tests/test_magic.py -q"]


def test_repo_map_and_candidate_selection_prioritize_entrypoints_and_skip_boilerplate(tmp_path: Path) -> None:
    from server.jobs import select_candidate_files
    from server.repo_map import build_repo_map

    repo = tmp_path / "r2_repomap"
    _init_repo(
        repo,
        {
            "LICENSE.txt": "MIT License user repo fork\n",
            "Dockerfile": "FROM python:3.11\n",
            "makefile": "test:\n\tpytest\n",
            "README.md": "mnamer CLI media organizer\n",
            "mnamer/__main__.py": "def main():\n    return 0\n",
            "api/queries_repo.go": "package api\nfunc RepoNetwork() {}\n",
            "pkg/cmd/repo/fork/fork.go": "package fork\nfunc ForkRun() {}\n",
        },
    )
    rmap = build_repo_map(repo)
    assert "mnamer/__main__.py" in rmap
    assert "api/queries_repo.go" in rmap

    cands = [rel for rel, _ in select_candidate_files(repo, "Fix user fork CLI flag in mnamer", max_files=4)]
    assert "LICENSE.txt" not in cands
    assert "Dockerfile" not in cands
    assert "makefile" not in cands
    assert "README.md" not in cands
    assert "mnamer/__main__.py" in cands


def test_detect_existing_test_stub_mutations_skips_silent_auto_apply(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    orig_test = "\n".join(
        [
            "def setup_one():",
            "    reg.Register(httpmock.GraphQL('query RepoInfo'), resp1)",
            "",
            "def setup_two():",
            "    reg.Register(httpmock.GraphQL('query RepoInfo'), resp2)",
            "",
            "def setup_three():",
            "    reg.Register(httpmock.GraphQL('query RepoInfo'), resp3)",
            "",
            "def test_stub_suite():",
            "    assert True",
            "",
        ]
    )
    mutated_test = orig_test.replace("query RepoInfo", "query RepoForkInfo")
    repo = tmp_path / "r2_stubs"
    _init_repo(
        repo,
        {"fork.py": "x = 1\n", "test_fork.py": orig_test, "test_ok.py": "def test_ok():\n    assert True\n"},
    )
    data_dir = tmp_path / "r2_stubs_data"
    _strict_economy_env(clean_budget_env, data_dir)

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "fork.py").write_text("x = 2\n", encoding="utf-8")
        (wt / "test_fork.py").write_text(mutated_test, encoding="utf-8")
        return {"text": "updated stubs"}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        res = jm.delegate_start(
            "Update fork logic in fork.py.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 -m pytest test_ok.py -q"],
            wait=True,
        )
        assert res.get("applied") is False
        assert res.get("stub_mutation_warnings")
        assert "stub" in str(res.get("auto_apply_skipped") or "").lower()
    finally:
        jm.shutdown()


def test_test_only_diff_green_checkpoint_promotion_and_zero_edit_no_retry(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    from server.jobs import _is_test_file_path

    # 0. Non-test `spec/` source directories must not be misclassified as test files, while RSpec `_spec.rb` is.
    assert _is_test_file_path("src/spec/parser.py") is False
    assert _is_test_file_path("docs/spec/openapi.yaml") is False
    assert _is_test_file_path("spec/models/user_spec.rb") is True

    # 1. Test-only diff on an implementation prompt keeps truthful acceptance_passed=True, does NOT fire
    # plugin_retry to mutate production files blindly, and skips silent auto-apply for Lead inspection.
    repo1 = tmp_path / "r2_test_only"
    _init_repo(repo1, {"impl.py": "x = 1\n", "tests/test_impl.py": "def test_x():\n    assert True\n"})
    d1 = tmp_path / "d1"
    _strict_economy_env(clean_budget_env, d1)
    impl_worker_turns = 0

    def test_only_worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal impl_worker_turns
        impl_worker_turns += 1
        (Path(req["workspace_dir"]) / "tests/test_impl.py").write_text(
            "def test_x():\n    assert 1 == 1\n", encoding="utf-8"
        )
        return {"text": "added test only"}

    jm1 = JobManager(workspace_root=repo1, store=ReceiptStore(d1, workspace_root=repo1), worker_backend=test_only_worker)
    try:
        res1 = jm1.delegate_start(
            "Fix bug in impl.py so x becomes 2.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 -m pytest tests/test_impl.py -q"],
            wait=True,
        )
        assert impl_worker_turns == 1
        assert res1.get("test_only_diff") is True
        assert res1.get("acceptance_passed") is True
        assert res1.get("applied") is False
        assert "only test" in str(res1.get("auto_apply_skipped") or "").lower()
    finally:
        jm1.shutdown()

    # 1b. Ordinary test-authoring prompts ("Write unit tests...", "Add a regression test...", "Increase test coverage...")
    # are NOT flagged as test_only_diff and auto-apply cleanly in 1 worker turn without touching production files.
    for idx, tprompt in enumerate(
        (
            "Write unit tests for header() in auth.py.",
            "Add a regression test covering header() returning 'ok'.",
            "Increase test coverage for auth.py.",
        )
    ):
        repo_t = tmp_path / f"r2_write_tests_{idx}"
        _init_repo(repo_t, {"auth.py": "def header():\n    return 'ok'\n", "tests/test_auth.py": "def test_h():\n    assert True\n"})
        dt = tmp_path / f"dt_{idx}"
        _strict_economy_env(clean_budget_env, dt)
        wt_turns = 0

        def unit_test_worker(req: dict[str, Any]) -> dict[str, Any]:
            nonlocal wt_turns
            wt_turns += 1
            (Path(req["workspace_dir"]) / "tests/test_auth.py").write_text(
                "from auth import header\n\ndef test_h():\n    assert header() == 'ok'\n",
                encoding="utf-8",
            )
            return {"text": "wrote unit tests"}

        jmt = JobManager(workspace_root=repo_t, store=ReceiptStore(dt, workspace_root=repo_t), worker_backend=unit_test_worker)
        try:
            rest = jmt.delegate_start(
                tprompt,
                model="gemini-3.8-flash",
                project="p",
                acceptance=["python3 -m pytest tests/test_auth.py -q"],
                wait=True,
            )
            assert wt_turns == 1
            assert rest.get("test_only_diff") is not True
            assert rest.get("acceptance_passed") is True
            assert rest.get("applied") is True
            assert (repo_t / "auth.py").read_text(encoding="utf-8") == "def header():\n    return 'ok'\n"
        finally:
            jmt.shutdown()

    # 2. Green tool-call checkpoint promotes to ok + auto-applies WITHOUT firing plugin_retry.
    repo2 = tmp_path / "r2_green_chk"
    _init_repo(repo2, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    d2 = tmp_path / "d2"
    _strict_economy_env(clean_budget_env, d2)
    chk_calls = 0

    def green_chk_worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal chk_calls
        chk_calls += 1
        (Path(req["workspace_dir"]) / "auth.py").write_text("def header():\n    return 'ok'\n", encoding="utf-8")
        return {
            "text": "done at limit",
            "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
            "tool_calls": [{"name": "view_file", "args": {}}] * 100,
        }

    jm2 = JobManager(workspace_root=repo2, store=ReceiptStore(d2, workspace_root=repo2), worker_backend=green_chk_worker)
    try:
        res2 = jm2.delegate_start(
            "Make header() return 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert chk_calls == 1
        assert res2["status"] == "ok"
        assert res2.get("recovered_from_checkpoint") is True
        assert res2["applied"] is True
    finally:
        jm2.shutdown()

    # 3. Zero-edit status='ok' turn does not waste a plugin_retry.
    repo3 = tmp_path / "r2_noop"
    _init_repo(repo3, {"auth.py": "def header():\n    return 'ok'\n", "test_auth.py": _AUTH_TEST})
    d3 = tmp_path / "d3"
    _strict_economy_env(clean_budget_env, d3)
    noop_calls = 0

    def noop_worker(_req: dict[str, Any]) -> dict[str, Any]:
        nonlocal noop_calls
        noop_calls += 1
        return {"text": "already complete"}

    jm3 = JobManager(workspace_root=repo3, store=ReceiptStore(d3, workspace_root=repo3), worker_backend=noop_worker)
    try:
        res3 = jm3.delegate_start(
            "Verify header() returns 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert noop_calls == 1
        assert res3.get("vacuous_acceptance") is True
    finally:
        jm3.shutdown()


def test_untracked_runtime_logs_excluded_sticky_bonus_and_config_filename_regex(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    from scripts.pre_tool_guard import _command_invokes_host_test_runner
    from scripts.session_start import handle_session_start

    # S2: config filenames and non-head arguments must not false-match test runner guard,
    # while bash -lc / pnpm -F pkg test / run test:unit must match.
    assert _command_invokes_host_test_runner("cat pytest.ini jest.config.js tox.ini") is False
    assert _command_invokes_host_test_runner("ls -la node_modules/.bin/vitest") is False
    assert _command_invokes_host_test_runner("grep -rn vitest package.json") is False
    assert _command_invokes_host_test_runner("echo 'run make test later'") is False
    assert _command_invokes_host_test_runner('bash -lc "pytest -q"') is True
    assert _command_invokes_host_test_runner("pnpm -F @scope/pkg test") is True
    assert _command_invokes_host_test_runner("pnpm --filter core run test:unit") is True
    assert _command_invokes_host_test_runner("pnpm exec vitest") is True
    assert _command_invokes_host_test_runner("python -X dev -m pytest") is True
    assert _command_invokes_host_test_runner("$(which pytest) -q") is True
    assert _command_invokes_host_test_runner("npm run-script test") is True

    repo = tmp_path / "r2_logs_repo"
    _init_repo(
        repo,
        {
            "auth.py": "def header():\n    return None\n",
            "test_auth.py": _AUTH_TEST,
            "tracked.log": "keep me\n",
            "big.py": "".join(f"line_{i} = {i}\n" for i in range(400)),
        },
    )
    data_dir = tmp_path / "r2_logs_data"
    _strict_economy_env(clean_budget_env, data_dir)
    store = ReceiptStore(data_dir, workspace_root=repo)
    env = {
        "CC_AGY_STRICT_MODE": "true",
        "CC_AGY_COST_PROFILE": "economy",
        "CC_AGY_DATA_DIR": str(data_dir),
        "CLAUDE_PROJECT_DIR": str(repo),
    }
    handle_session_start({"cwd": str(repo), "session_id": "s-logs", "source": "startup"}, env=env, store=store)

    call_idx = 0

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_idx
        call_idx += 1
        wt = Path(req["workspace_dir"])
        if call_idx == 1:
            (wt / "auth.py").write_text("def header():\n    return 'ok'\n", encoding="utf-8")
            (wt / "tracked.log").write_text("keep me updated\n", encoding="utf-8")
            (wt / "initial-state.json").write_text('{"theme": "light"}\n', encoding="utf-8")
            (wt / "e2e_daemon.log").write_text("stray runtime log\n", encoding="utf-8")
            (wt / "worker.pid").write_text("12345\n", encoding="utf-8")
            (wt / "worker.pid.lock").write_text("12345\n", encoding="utf-8")
        return {"text": f"job {call_idx}"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        r1 = jm.delegate_start(
            "Update header(), tracked.log, and initial-state.json",
            model="gemini-3.8-flash",
            project="p",
            allowed_paths=["auth.py", "tracked.log", "initial-state.json"],
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert r1["applied"] is True
        assert sorted(r1["applied_files"]) == ["auth.py", "initial-state.json", "tracked.log"]
        assert (repo / "initial-state.json").exists()
        assert not (repo / "e2e_daemon.log").exists()
        assert not (repo / "worker.pid").exists()
        assert not (repo / "worker.pid.lock").exists()

        # Run a 0-edit follow-up job (like mnamer Job 2): sticky review bonus must remain 0!
        r2 = jm.delegate_start(
            "Double check auth.py",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert r2["files_changed"] == []
        # Lead used 0 of its 2 base reads; a massive sed dump (1,360p = 3 units) exceeds remaining 2 + 0 bonus!
        sed_dec = evaluate_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": f"sed -n '1,360p' {repo / 'big.py'}"}, "cwd": str(repo)},
            env=env,
        )
        assert sed_dec["hookSpecificOutput"]["permissionDecision"] == "deny"
    finally:
        jm.shutdown()


def test_timed_out_baseline_acceptance_scopes_to_related_tests_and_shutdown_guard_and_pnpm_budget(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    from server.jobs import link_host_node_modules_farm

    # 1. R2-5.1: When baseline_acceptance_probe timed out on broad `pytest -q`, _execute_job scopes post-turn
    # acceptance to related + modified test files instead of re-hanging on the slow suite.
    repo = tmp_path / "r2_slow_base"
    _init_repo(
        repo,
        {
            "pkg/__init__.py": "",
            "pkg/calc.py": "def double(x):\n    return x\n",
            "tests/test_calc.py": "from pkg.calc import double\n\ndef test_double():\n    assert double(3) == 6\n",
            "tests/test_hang.py": "import time\n\ndef test_slow():\n    time.sleep(5)\n",
        },
    )
    data_dir = tmp_path / "r2_slow_data"
    _strict_economy_env(clean_budget_env, data_dir)

    def fast_fix_worker(req: dict[str, Any]) -> dict[str, Any]:
        assert "[Baseline Acceptance Timeout Note]" in req["prompts"][0]
        (Path(req["workspace_dir"]) / "pkg/calc.py").write_text(
            "def double(x):\n    return x * 2\n", encoding="utf-8"
        )
        return {"text": "fixed double"}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=fast_fix_worker)
    try:
        # timeout_seconds=1.0 -> per_probe_limit=0.4s so `python3 -m pytest -q` times out on test_hang.py (5s),
        # then post-turn acceptance scopes cleanly to tests/test_calc.py and passes in <0.5s!
        res = jm.delegate_start(
            "Fix double() in pkg/calc.py",
            model="gemini-3.8-flash",
            project="p",
            allowed_paths=["pkg/calc.py"],
            acceptance=["python3 -m pytest -q"],
            timeout_seconds=1.0,
            wait=True,
        )
        assert res["applied"] is True
        rec = store.read_receipt(res["job_id"])
        assert any("tests/test_calc.py" in str(r.get("command")) for r in rec["acceptance_results"])

        # 2. S3: Once shutting down, delegate_start refuses new submissions cleanly.
        jm._shutting_down = True
        with pytest.raises(RuntimeError, match="shutting down"):
            jm.delegate_start("Should not start", model="gemini-3.8-flash", project="p", wait=False)
    finally:
        jm.shutdown()

    # 3. S4: Cumulative pnpm virtual-store copy budget caps total copied files across packages.
    host_ws = tmp_path / "pnpm_host"
    wt_ws = tmp_path / "pnpm_wt"
    (host_ws / "node_modules/.pnpm").mkdir(parents=True)
    wt_ws.mkdir(parents=True)
    for idx in range(5):
        pkg_store = host_ws / f"node_modules/.pnpm/pkg{idx}@1.0.0/node_modules/pkg{idx}"
        pkg_store.mkdir(parents=True)
        for f_i in range(4):
            (pkg_store / f"f{f_i}.js").write_text("export default 1;\n", encoding="utf-8")
        (host_ws / "node_modules" / f"pkg{idx}").symlink_to(
            f".pnpm/pkg{idx}@1.0.0/node_modules/pkg{idx}"
        )
    farm = link_host_node_modules_farm(host_ws, wt_ws)
    assert farm is not None
    assert (wt_ws / "node_modules/.pnpm").is_dir()


def test_tdd_new_pytest_target_is_not_remapped_to_unrelated_existing_test(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "r2_tdd_remap"
    _init_repo(
        repo,
        {
            "config.py": "def load_config():\n    return {}\n",
            "pkg/tests/test_config.py": "def test_unrelated_pkg():\n    assert True\n",
        },
    )
    data_dir = tmp_path / "r2_tdd_data"
    _strict_economy_env(clean_budget_env, data_dir)
    clean_budget_env.setenv("CC_AGY_SAME_MODEL_RETRIES", "0")

    def tdd_buggy_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "tests").mkdir(parents=True, exist_ok=True)
        (wt / "tests/test_config.py").write_text(
            "from config import load_config\n\ndef test_default_debug():\n    assert load_config() == {'debug': False}\n",
            encoding="utf-8",
        )
        # Worker forgot to update config.py -> new TDD test MUST fail, NOT pass via pkg/tests/test_config.py!
        (wt / "config.py").write_text("def load_config():\n    return {'debug': True}\n", encoding="utf-8")
        return {"text": "created tests/test_config.py"}

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=tdd_buggy_worker)
    try:
        res = jm.delegate_start(
            "Create tests/test_config.py asserting load_config() returns {'debug': False} by default, then implement it.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 -m pytest -q tests/test_config.py"],
            wait=True,
        )
        assert res.get("acceptance_passed") is False
        assert res.get("applied") is False
    finally:
        jm.shutdown()


def test_r3_run_acceptance_fail_fast_on_timeout_and_default_90s(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    from server.jobs import DEFAULT_ACCEPTANCE_TIMEOUT, DEFAULT_BOUNDED_ACCEPTANCE_TIMEOUT, run_acceptance

    assert DEFAULT_ACCEPTANCE_TIMEOUT == 300.0
    assert DEFAULT_BOUNDED_ACCEPTANCE_TIMEOUT == 90.0
    marker = tmp_path / "second_cmd_ran.txt"
    results = run_acceptance(
        [
            'python3 -c "import time; time.sleep(5)"',
            f'python3 -c "from pathlib import Path; Path({str(marker)!r}).write_text(\'ran\')"',
        ],
        tmp_path,
        timeout=0.3,
    )
    assert len(results) == 1
    assert results[0].get("timed_out") is True
    assert results[0].get("passed") is False
    assert not marker.exists()


def test_r3_salvage_unapplied_worktree_omits_unverified_broken_new_tests(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "r3_salvage_repo"
    _init_repo(
        repo,
        {
            "mobly/base_test.py": "def run():\n    return 'old'\n",
            "tests/test_existing.py": "from mobly.base_test import run\n\ndef test_old():\n    assert run() in ('old', 'new')\n",
        },
    )
    data_dir = tmp_path / "r3_salvage_data"
    _strict_economy_env(clean_budget_env, data_dir)

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda req: {"text": "ok"})
    try:
        wt_dir = tmp_path / "r3_salvage_wt"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt_dir), "HEAD"],
            cwd=str(repo),
            check=True,
            capture_output=True,
        )
        (wt_dir / "mobly/base_test.py").write_text("def run():\n    return 'new'\n", encoding="utf-8")
        (wt_dir / "tests/test_new_deadlock.py").write_text(
            "def test_broken_new():\n    assert False, 'deadlock/fail in new test'\n",
            encoding="utf-8",
        )
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        receipt = {
            "job_id": "job-r3-salvage",
            "kind": "delegate",
            "status": "interrupted",
            "applied": False,
            "acceptance_passed": False,
            "worktree_dir": str(wt_dir),
            "base_head": head_sha,
            "files_changed": ["mobly/base_test.py", "tests/test_new_deadlock.py"],
            "acceptance_results": [
                {
                    "command": "pytest tests/test_new_deadlock.py",
                    "exit_code": None,
                    "timed_out": True,
                    "passed": False,
                }
            ],
        }
        store.write_receipt("job-r3-salvage", receipt)
        salvaged = jm.salvage_unapplied_worktree()
        assert salvaged is not None
        assert (repo / "mobly/base_test.py").read_text(encoding="utf-8") == "def run():\n    return 'new'\n"
        assert not (repo / "tests/test_new_deadlock.py").exists()
        saved_rec = store.read_receipt("job-r3-salvage")
        assert saved_rec.get("salvage_omitted_unverified_tests") == ["tests/test_new_deadlock.py"]
    finally:
        jm.shutdown()


def test_r3_node_modules_bin_materialized_when_isolate_copies_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import server.jobs as jobs_mod
    from server.jobs import link_host_node_modules_farm

    monkeypatch.setattr(jobs_mod, "_NM_COPY_MAX_FILES", 1)
    host_ws = tmp_path / "host_nm_bin"
    wt_ws = tmp_path / "wt_nm_bin"
    (host_ws / "node_modules/vitest").mkdir(parents=True)
    vitest_cli = host_ws / "node_modules/vitest/vitest.mjs"
    vitest_cli.write_text("#!/usr/bin/env node\nconsole.log('ok');\n", encoding="utf-8")
    (host_ws / "node_modules/vitest/extra1.js").write_text("export default 1;\n", encoding="utf-8")
    (host_ws / "node_modules/vitest/extra2.js").write_text("export default 2;\n", encoding="utf-8")
    (host_ws / "node_modules/.bin").mkdir(parents=True)
    (host_ws / "node_modules/.bin/vitest").symlink_to("../vitest/vitest.mjs")
    wt_ws.mkdir(parents=True)

    farm = link_host_node_modules_farm(host_ws, wt_ws)
    assert farm is not None
    wt_bin = wt_ws / "node_modules/.bin"
    assert wt_bin.is_dir()
    assert not wt_bin.is_symlink()
    wt_vitest = wt_bin / "vitest"
    assert wt_vitest.is_symlink()
    assert os.readlink(wt_vitest) == "../vitest/vitest.mjs"


def test_r3_collect_workspace_node_bin_dirs_and_node_test_runner_acceptance(
    tmp_path: Path,
) -> None:
    from scripts.pre_tool_guard import detect_bash_workspace_mutation
    from server.jobs import _collect_workspace_node_bin_dirs, validate_acceptance

    ws = tmp_path / "monorepo"
    sub_bin = ws / "packages/core/node_modules/.bin"
    sub_bin.mkdir(parents=True)
    bins = _collect_workspace_node_bin_dirs(ws)
    assert str(sub_bin) in bins

    ok_cmds = validate_acceptance(
        [
            "node node_modules/vitest/vitest.mjs run packages/core/tests/snapshot.test.ts",
            "node packages/core/node_modules/.bin/vitest run packages/core/tests/snapshot.test.ts",
            "node node_modules/typescript/lib/tsc.js --noEmit",
        ],
        ws,
    )
    assert len(ok_cmds) == 3
    assert (
        detect_bash_workspace_mutation(
            "node node_modules/vitest/vitest.mjs run packages/core/tests/snapshot.test.ts",
            ws,
        )["blocked"]
        is False
    )
    for blocked_cmd in (
        "node scripts/mutate.js",
        "node node_modules/typescript/lib/tsc.js",
        "node node_modules/typescript/lib/tsc.js --init",
        "node node_modules/typescript/lib/tsc.js --build",
        "node node_modules/.bin/tsc -p .",
        "node node_modules/jest/bin/jest.js --updateSnapshot=true",
        "node node_modules/playwright/cli.js install",
    ):
        assert detect_bash_workspace_mutation(blocked_cmd, ws)["blocked"] is True
        with pytest.raises(ValueError):
            validate_acceptance([blocked_cmd], ws)


def test_r4_1_plugin_retry_skips_high_call_count_and_compiler_cli_help_and_preserves_prior_summary(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    from server.jobs import _only_compiler_cli_config_failures

    repo = tmp_path / "r4_retry_repo"
    _init_repo(repo, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    data_dir = tmp_path / "r4_retry_data"
    _strict_economy_env(clean_budget_env, data_dir)

    calls = 0

    def high_call_worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        (Path(req["workspace_dir"]) / "auth.py").write_text("def header():\n    return 'nope'\n", encoding="utf-8")
        return {
            "text": "Diagnostic summary from Attempt 1 (60 tool calls)",
            "tool_calls": [{"name": "view_file", "args": {}}] * 60,
        }

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=high_call_worker)
    try:
        res = jm.delegate_start(
            "Make header() return 'ok'.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert calls == 1
        assert store.read_receipt(res["job_id"])["prompt_sources"] == ["lead"]
    finally:
        jm.shutdown()

    # Compiler CLI help alongside passing test runner skips plugin_retry
    acc_sample = [
        {
            "command": "./node_modules/.bin/jest --config=tests/jest.config.js",
            "passed": True,
            "exit_code": 0,
            "counts": {"passed": 5, "failed": 0, "errors": 0},
        },
        {
            "command": "./node_modules/.bin/tsc --noEmit",
            "passed": False,
            "exit_code": 1,
            "counts": {"passed": 0, "failed": 0, "errors": 0},
            "output_tail": "Version 4.3.5\nSyntax:   tsc [options] [file...]\n --init Initializes a TypeScript project and creates a tsconfig.json file.",
        },
    ]
    assert _only_compiler_cli_config_failures(acc_sample) is True


def test_r4_2_escalate_to_lead_across_session_slices_and_deadline_unlocks_host_tools(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    import time
    from scripts.pre_tool_guard import _session_has_lead_escalated_job
    from scripts.session_start import handle_session_start

    repo = tmp_path / "r4_esc_repo"
    _init_repo(repo, {"auth.py": "def header():\n    return None\n", "test_auth.py": _AUTH_TEST})
    data_dir = tmp_path / "r4_esc_data"
    _strict_economy_env(clean_budget_env, data_dir)
    clean_budget_env.setenv("CC_AGY_SESSION_DEADLINE_SECONDS", "900")
    store = ReceiptStore(data_dir, workspace_root=repo)
    env = {
        "CC_AGY_STRICT_MODE": "true",
        "CC_AGY_COST_PROFILE": "economy",
        "CC_AGY_DATA_DIR": str(data_dir),
        "CLAUDE_PROJECT_DIR": str(repo),
    }
    handle_session_start({"cwd": str(repo), "session_id": "s-r4", "source": "startup"}, env=env, store=store)

    def checkpoint_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "auth.py").write_text("def header():\n    return 'almost'\n", encoding="utf-8")
        (wt / "tests").mkdir(exist_ok=True)
        (wt / "tests/test_new_group.py").write_text("def test_bad():\n    assert False\n", encoding="utf-8")
        return {
            "text": "stopped at cap",
            "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
            "tool_calls": [{"name": "view_file", "args": {}}] * 97,
        }

    jm = JobManager(workspace_root=repo, store=store, worker_backend=checkpoint_worker)
    try:
        # Simulate near session deadline (< 150s remaining out of 900s)
        jm._created_at = time.monotonic() - 780.0
        r1 = jm.delegate_start(
            "Implement header() and tests.",
            model="gemini-3.8-flash",
            project="p",
            allowed_paths=["auth.py", "tests/test_new_group.py"],
            acceptance=["python3 test_auth.py"],
            wait=True,
        )
        assert r1["escalate_to_lead"] is True
        assert "escalate_to_lead=true" in str(r1.get("next_action") or "")
        assert _session_has_lead_escalated_job(data_dir, repo) is True

        # Host Read, Grep, Edit, Write, and test-runner Bash are now unlocked even after exhausting initial read limit!
        (data_dir / ".lead_read_count").write_text("99", encoding="utf-8")
        for payload in (
            {"tool_name": "Read", "tool_input": {"file_path": str(repo / "auth.py")}, "cwd": str(repo)},
            {"tool_name": "Grep", "tool_input": {"pattern": "header", "path": str(repo)}, "cwd": str(repo)},
            {"tool_name": "Edit", "tool_input": {"file_path": str(repo / "auth.py")}, "cwd": str(repo)},
            {"tool_name": "Write", "tool_input": {"file_path": str(repo / "tests/test_new_group.py")}, "cwd": str(repo)},
            {"tool_name": "Bash", "tool_input": {"command": "pytest -q test_auth.py"}, "cwd": str(repo)},
        ):
            assert evaluate_pre_tool_use(payload, env=env) == {}
    finally:
        jm.shutdown()


def test_r4_3_untracked_model_dumps_and_out_of_scope_untracked_omitted_on_apply(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "r4_igel_repo"
    _init_repo(repo, {"igel.py": "x = 1\n", "test_igel.py": "def test_ok():\n    assert False\n"})
    data_dir = tmp_path / "r4_igel_data"
    _strict_economy_env(clean_budget_env, data_dir)
    store = ReceiptStore(data_dir, workspace_root=repo)

    def polluting_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "igel.py").write_text("x = 2\n", encoding="utf-8")
        (wt / "tests").mkdir(exist_ok=True)
        (wt / "tests/test_feature_schema.py").write_text("def test_schema():\n    assert False\n", encoding="utf-8")
        mr = wt / "model_results"
        mr.mkdir(exist_ok=True)
        (mr / "model.joblib").write_bytes(b"binary-joblib")
        (mr / "predictions.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (wt / "post_req_data.csv").write_text("x,y\n3,4\n", encoding="utf-8")
        return {"text": "done", "tool_calls": [{"name": "edit_file", "args": {}}] * 65}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=polluting_worker)
    try:
        res = jm.delegate_start(
            "Implement feature schema.",
            model="gemini-3.8-flash",
            project="p",
            allowed_paths=["igel.py", "tests/test_feature_schema.py"],
            acceptance=["python3 -m pytest tests/test_feature_schema.py -q"],
            wait=True,
        )
        # 1. model.joblib is excluded at diff-staging time as an untracked binary model dump and surfaced in digest
        assert "model_results/model.joblib" not in res["files_changed"]
        assert res.get("excluded_runtime_artifacts") == ["model_results/model.joblib"]
        # 2. delegate_apply(allow_scope_violations=True) applies in-scope implementation files while skipping untracked out-of-scope CSV dumps AND omitting failing newly created test files by default
        app = jm.delegate_apply(res["job_id"], allow_scope_violations=True)
        assert sorted(app["applied_files"]) == ["igel.py"]
        assert sorted(app["omitted_untracked_out_of_scope"]) == [
            "model_results/predictions.csv",
            "post_req_data.csv",
        ]
        assert app.get("omitted_unverified_new_test_files") == ["tests/test_feature_schema.py"]
        assert not (repo / "tests/test_feature_schema.py").exists()
        assert not (repo / "model_results/predictions.csv").exists()
        assert not (repo / "post_req_data.csv").exists()
        # 3. Diff of omitted test file remains inspectable via get_diff(file_filter=...)
        diff_omitted = jm.get_diff(res["job_id"], file_filter="tests/test_feature_schema.py")
        assert "def test_schema():" in diff_omitted["diff"]
        # 4. Follow-up delegate_apply with include_paths / include_unverified_new_tests on an already-applied job recovers omitted files even after worktree cleanup
        app2 = jm.delegate_apply(
            res["job_id"],
            allow_scope_violations=True,
            include_paths=["post_req_data.csv"],
            include_unverified_new_tests=True,
        )
        assert "post_req_data.csv" in app2["applied_files"]
        assert "tests/test_feature_schema.py" in app2["applied_files"]
        assert (repo / "post_req_data.csv").read_text(encoding="utf-8") == "x,y\n3,4\n"
        assert (repo / "tests/test_feature_schema.py").exists()
    finally:
        jm.shutdown()


def test_r4_4_imported_helper_and_prompt_named_binary_preserved_and_hmac_guards_escalation(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    import json
    import time
    from scripts.pre_tool_guard import _session_has_lead_escalated_job
    from server.jobs import _is_compiler_cli_config_or_help_failure, probe_baseline_acceptance

    # 1. Out-of-scope helper module imported by in-scope file is preserved by delegate_apply(allow_scope_violations=True)
    repo1 = tmp_path / "r4_scope_helper_repo"
    _init_repo(repo1, {"src/__init__.py": "", "src/foo.py": "def run():\n    return 1\n"})
    data1 = tmp_path / "r4_scope_helper_data"
    _strict_economy_env(clean_budget_env, data1)
    store1 = ReceiptStore(data1, workspace_root=repo1)

    def helper_worker(req: dict[str, Any]) -> dict[str, Any]:
        wd = Path(req["workspace_dir"])
        (wd / "src/foo_helpers.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
        (wd / "src/foo.py").write_text(
            "from src.foo_helpers import helper\n\ndef run():\n    return helper()\n",
            encoding="utf-8",
        )
        return {"text": "split helper out"}

    jm1 = JobManager(workspace_root=repo1, store=store1, worker_backend=helper_worker)
    try:
        res1 = jm1.delegate_start(
            "Make run() in src/foo.py return 2 via a helper.",
            model="gemini-3.8-flash",
            project="p",
            allowed_paths=["src/foo.py"],
            wait=True,
            auto_apply=False,
        )
        app1 = jm1.delegate_apply(res1["job_id"], allow_scope_violations=True)
        assert sorted(app1["applied_files"]) == ["src/foo.py", "src/foo_helpers.py"]
        assert (repo1 / "src/foo_helpers.py").exists()
    finally:
        jm1.shutdown()

    # 2. Prompt-named and code-referenced binary file (.npy) is preserved and applied, while bare stems (app -> app.log, model -> model.joblib) never resurrect runtime artifacts
    from server.jobs import _is_referenced_by_files_or_prompt

    stem_dir = tmp_path / "r4_stem_check"
    stem_dir.mkdir(exist_ok=True)
    (stem_dir / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n", encoding="utf-8")
    (stem_dir / "train.py").write_text("def fit(model):\n    return model.fit()\n", encoding="utf-8")
    assert _is_referenced_by_files_or_prompt("app.log", ["app.py"], stem_dir) is False
    assert _is_referenced_by_files_or_prompt("model_results/model.joblib", ["train.py"], stem_dir) is False

    repo2 = tmp_path / "r4_binary_repo"
    _init_repo(repo2, {"load.py": "import numpy\n"})
    data2 = tmp_path / "r4_binary_data"
    _strict_economy_env(clean_budget_env, data2)
    store2 = ReceiptStore(data2, workspace_root=repo2)

    def binary_worker(req: dict[str, Any]) -> dict[str, Any]:
        wd = Path(req["workspace_dir"])
        (wd / "data").mkdir(exist_ok=True)
        (wd / "data/embeddings.npy").write_bytes(b"\x93NUMPY\x01\x00" + bytes(range(16)))
        (wd / "load.py").write_text("import numpy as np\nE = np.load('data/embeddings.npy')\n", encoding="utf-8")
        return {"text": "added embeddings"}

    jm2 = JobManager(workspace_root=repo2, store=store2, worker_backend=binary_worker)
    try:
        res2 = jm2.delegate_start(
            "Add the precomputed embeddings file data/embeddings.npy and load it in load.py.",
            model="gemini-3.8-flash",
            project="p",
            wait=True,
        )
        assert "data/embeddings.npy" in res2["files_changed"]
        assert (repo2 / "data/embeddings.npy").exists()
    finally:
        jm2.shutdown()

    # 3. Multi-line tsc CLI help probe preserves compiler_cli_config_error, and receipt HMAC signs escalate_to_lead
    fake_tsc = repo2 / "fake_tsc.py"
    fake_tsc.write_text(
        "import sys\n"
        "print('Syntax:   tsc [options] [file...]')\n"
        "for i in range(10):\n"
        "    print(f'--opt{i}  desc {i}')\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    probes = probe_baseline_acceptance(["python3 fake_tsc.py"], repo2)
    assert len(probes) == 1
    assert _is_compiler_cli_config_or_help_failure(probes[0]) is True

    store2.write_receipt(
        "job-tamper",
        {
            "job_id": "job-tamper",
            "status": "checkpoint",
            "kind": "delegate",
            "allowed_paths": ["load.py"],
            "updated_at_epoch": time.time(),
        },
        workspace_root=str(repo2),
    )
    raw_p = store2.receipt_path("job-tamper")
    tampered = json.loads(raw_p.read_text(encoding="utf-8"))
    tampered["escalate_to_lead"] = True
    raw_p.write_text(json.dumps(tampered), encoding="utf-8")
    assert _session_has_lead_escalated_job(data2, repo2) is False
    assert _session_has_lead_escalated_job(data2, None) is False


def test_r5_1_delegate_apply_filters_unverified_new_tests_and_pytest_exit4_missing_paths(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    # Part A: pytest exit_code == 4 on guessed non-existent test file path strips missing path and runs existing test files;
    # and delegate_apply omits failing/deadlocking new test file while keeping passing new test file!
    repo = tmp_path / "r5_mobly_repo"
    _init_repo(
        repo,
        {
            "mobly_base.py": "def run_group():\n    return 'old'\n",
            "tests/test_existing.py": "from mobly_base import run_group\n\ndef test_existing():\n    assert run_group() == 'new'\n",
        },
    )
    data_dir = tmp_path / "r5_mobly_data"
    _strict_economy_env(clean_budget_env, data_dir)
    store = ReceiptStore(data_dir, workspace_root=repo)

    def worker_with_bad_and_good_new_tests(req: dict[str, Any]) -> dict[str, Any]:
        wd = Path(req["workspace_dir"])
        (wd / "mobly_base.py").write_text("def run_group():\n    return 'new'\n", encoding="utf-8")
        (wd / "tests/test_good_new.py").write_text(
            "from mobly_base import run_group\n\ndef test_good():\n    assert run_group() == 'new'\n",
            encoding="utf-8",
        )
        (wd / "tests/test_broken_new.py").write_text(
            "def test_deadlock_or_fail():\n    assert False, 'broken worker test'\n",
            encoding="utf-8",
        )
        return {"text": "done", "tool_calls": [{"name": "edit_file", "args": {}}] * 65}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker_with_bad_and_good_new_tests)
    try:
        # Lead guessed non-existent `tests/expects_test.py` alongside `tests/test_existing.py` and `tests/test_broken_new.py`
        res = jm.delegate_start(
            "Implement run_group.",
            model="gemini-3.8-flash",
            project="p",
            allowed_paths=["mobly_base.py", "tests/test_good_new.py", "tests/test_broken_new.py"],
            acceptance=[
                "python3 -m pytest -q tests/test_existing.py tests/test_broken_new.py tests/expects_test.py"
            ],
            wait=True,
        )
        acc0 = res["acceptance_results"][0]
        # Missing `tests/expects_test.py` was stripped so `test_existing.py` and `test_broken_new.py` actually ran!
        assert acc0.get("filtered_missing_paths") == ["tests/expects_test.py"]
        assert "tests/expects_test.py" not in str(acc0.get("effective_command") or "")
        assert acc0["passed"] is False

        # When Lead calls delegate_apply, `mobly_base.py` and `tests/test_good_new.py` (which passes its 8s smoke check)
        # are applied, while `tests/test_broken_new.py` is omitted!
        app = jm.delegate_apply(res["job_id"])
        assert sorted(app["applied_files"]) == ["mobly_base.py", "tests/test_good_new.py"]
        assert app.get("omitted_unverified_new_test_files") == ["tests/test_broken_new.py"]
        assert (repo / "mobly_base.py").read_text(encoding="utf-8") == "def run_group():\n    return 'new'\n"
        assert (repo / "tests/test_good_new.py").exists()
        assert not (repo / "tests/test_broken_new.py").exists()

        # Part B: Preceding `compile_smoke` pass in `acceptance_results` never prevents `_filter_unverified_new_tests_for_salvage`
        # from dropping a test file that failed in `pytest -q`.
        wt_order = tmp_path / "r5_order_wt"
        wt_order.mkdir(exist_ok=True)
        (wt_order / "tests").mkdir(exist_ok=True)
        (wt_order / "tests/test_broken_new.py").write_text("def test_bad(): assert False\n", encoding="utf-8")
        rec_order = {
            "job_id": "job-test-order",
            "status": "checkpoint",
            "acceptance_passed": False,
            "worktree_dir": str(wt_order),
            "files_changed": ["mobly_base.py", "tests/test_broken_new.py"],
            "baseline_file_hashes": {"mobly_base.py": "abc", "tests/test_broken_new.py": None},
            "acceptance_results": [
                {
                    "command": "python3 -m py_compile mobly_base.py tests/test_broken_new.py",
                    "passed": True,
                    "exit_code": 0,
                    "compile_smoke": True,
                },
                {
                    "command": "pytest -q",
                    "passed": False,
                    "exit_code": 1,
                    "output_tail": "FAILED tests/test_broken_new.py::test_bad - AssertionError\n1 failed in 0.05s",
                },
            ],
        }
        store.write_receipt("job-test-order", rec_order, workspace_root=str(repo))
        assert jm._filter_unverified_new_tests_for_salvage(rec_order, wt_order) == ["tests/test_broken_new.py"]
    finally:
        jm.shutdown()


def test_r5_2_untouched_baseline_go_vet_and_linter_failures_neutralized(
    tmp_path: Path, clean_budget_env: pytest.MonkeyPatch
) -> None:
    from server.jobs import evaluate_post_worker_baseline_failures, probe_baseline_acceptance

    repo = tmp_path / "r5_govet_repo"
    _init_repo(
        repo,
        {
            "pkg/cmd/pr/view/view.go": "package view\n",
            "pkg/cmd/pr/shared/Forceable.go": "package shared\n",
        },
    )
    fake_vet = repo / "fake_go_vet.py"
    fake_vet.write_text(
        "import sys\n"
        "sys.stderr.write('# github.com/cli/cli/v2/pkg/cmd/pr/shared\\n')\n"
        "sys.stderr.write('pkg/cmd/pr/shared/Forceable.go:19:6: Forceable redeclared in this block\\n')\n"
        "sys.exit(2)\n",
        encoding="utf-8",
    )
    probes = probe_baseline_acceptance(["python3 fake_go_vet.py"], repo)
    assert len(probes) == 1
    assert probes[0]["failing_files"] == ["pkg/cmd/pr/shared/Forceable.go"]

    baseline_probe_govet = [
        {
            "command": "go vet ./pkg/cmd/pr/...",
            "passed": False,
            "exit_code": 2,
            "timed_out": False,
            "failing_files": ["pkg/cmd/pr/shared/Forceable.go"],
            "error_tail": "pkg/cmd/pr/shared/Forceable.go:19:6: Forceable redeclared in this block",
        }
    ]
    acc_results = [
        {
            "command": "go test ./pkg/cmd/pr/view/...",
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "output_tail": "ok  \tgithub.com/cli/cli/v2/pkg/cmd/pr/view\t0.012s\n",
        },
        {
            "command": "go vet ./pkg/cmd/pr/...",
            "passed": False,
            "exit_code": 2,
            "timed_out": False,
            "output_tail": (
                "# github.com/cli/cli/v2/pkg/cmd/pr/shared\n"
                "pkg/cmd/pr/shared/Forceable.go:19:6: Forceable redeclared in this block\n"
            ),
        },
    ]
    warnings = evaluate_post_worker_baseline_failures(
        acc_results,
        baseline_probe_govet,
        repo,
        ["pkg/cmd/pr/view/view.go", "pkg/cmd/pr/view/view_test.go"],
    )
    assert len(warnings) == 1
    assert "pkg/cmd/pr/shared/Forceable.go" in warnings[0]
    assert acc_results[1]["passed"] is True
    assert acc_results[1]["baseline_neutralized"] is True

    # Negative check 1: if `go vet` PASSED on clean HEAD (`passed: True` in baseline_probe), a cross-package failure
    # in untouched `pkg/cmd/pr/shared/Forceable.go` MUST NOT be neutralized!
    acc_results_clean_head_passed = [
        dict(acc_results[0]),
        {
            "command": "go vet ./pkg/cmd/pr/...",
            "passed": False,
            "exit_code": 2,
            "timed_out": False,
            "output_tail": "pkg/cmd/pr/shared/Forceable.go:19:6: not enough arguments in call\n",
        },
    ]
    warnings_clean_head = evaluate_post_worker_baseline_failures(
        acc_results_clean_head_passed,
        [{"command": "go vet ./pkg/cmd/pr/...", "passed": True, "exit_code": 0, "timed_out": False}],
        repo,
        ["pkg/cmd/pr/view/view.go"],
    )
    assert warnings_clean_head == []
    assert acc_results_clean_head_passed[1]["passed"] is False

    # Negative check 2: if the worker modified the file reported by `go vet`, it MUST NOT be neutralized!
    acc_results_touched = [
        dict(acc_results[0]),
        {
            "command": "go vet ./pkg/cmd/pr/...",
            "passed": False,
            "exit_code": 2,
            "timed_out": False,
            "output_tail": "pkg/cmd/pr/view/view.go:42:2: unreachable code\n",
        },
    ]
    warnings_touched = evaluate_post_worker_baseline_failures(
        acc_results_touched,
        baseline_probe_govet,
        repo,
        ["pkg/cmd/pr/view/view.go"],
    )
    assert warnings_touched == []
    assert acc_results_touched[1]["passed"] is False




