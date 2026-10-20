"""Round 4 fixes: generated test reports on apply, per-session Lead review reads, delegation checklist."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from scripts.pre_tool_guard import evaluate_pre_tool_use
from scripts.session_start import (
    REVIEW_AFTER_APPLY_GUIDANCE,
    REVIEW_BEFORE_APPLY_GUIDANCE,
    _review_guidance_enabled,
    review_guidance_for,
)
from server.jobs import JobManager, _is_ephemeral_build_or_test_artifact
from server.receipts import ReceiptStore


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    (repo / "app.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    (repo / ".gitignore").write_text("coverage.xml\nreports/\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")


def test_root_coverage_reports_are_ephemeral_but_nested_fixtures_are_not() -> None:
    for name in ("coverage.xml", "coverage.json", "lcov.info", "nosetests.xml"):
        assert _is_ephemeral_build_or_test_artifact(name), name
    for name in ("tests/fixtures/coverage.xml", "sub/coverage.json", "src/coverage_tools.py", "coverage.py"):
        assert not _is_ephemeral_build_or_test_artifact(name), name


def test_apply_keeps_host_copy_of_gitignored_file_created_on_both_sides() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        _init_repo(repo)
        store = ReceiptStore(Path(tmp) / "data", workspace_root=repo)

        def worker(req: dict) -> dict:
            wt = Path(req["workspace_dir"])
            (wt / "app.py").write_text("def greet():\n    return 'worker'\n", encoding="utf-8")
            (wt / "reports").mkdir()
            (wt / "reports" / "junit.xml").write_text("<worker/>", encoding="utf-8")
            return {"text": "done"}

        jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
        digest = jm.delegate_start("Edit app.py", project="p", wait=True, auto_apply=False)
        assert digest["status"] == "ok"
        # Host test run writes the same gitignored report after the job started.
        (repo / "reports").mkdir()
        (repo / "reports" / "junit.xml").write_text("<host/>", encoding="utf-8")

        res = jm.delegate_apply(digest["job_id"])
        assert res["applied"] is True
        assert "worker" in (repo / "app.py").read_text(encoding="utf-8")
        assert (repo / "reports" / "junit.xml").read_text(encoding="utf-8") == "<host/>"
        assert res.get("kept_host_copy") == ["reports/junit.xml"]


def test_auto_apply_digest_reports_kept_host_copy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        _init_repo(repo)
        store = ReceiptStore(Path(tmp) / "data", workspace_root=repo)

        def worker(req: dict) -> dict:
            wt = Path(req["workspace_dir"])
            (wt / "app.py").write_text("def greet():\n    return 'worker'\n", encoding="utf-8")
            (wt / "reports").mkdir()
            (wt / "reports" / "junit.xml").write_text("<worker/>", encoding="utf-8")
            # Host test run writes the same gitignored report while the job is running.
            (repo / "reports").mkdir(exist_ok=True)
            (repo / "reports" / "junit.xml").write_text("<host/>", encoding="utf-8")
            return {"text": "done"}

        jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
        digest = jm.delegate_start("Edit app.py", project="p", wait=True, auto_apply=True)
        assert digest.get("applied") is True, digest
        assert digest.get("kept_host_copy") == ["reports/junit.xml"]
        assert (repo / "reports" / "junit.xml").read_text(encoding="utf-8") == "<host/>"


def test_apply_still_refuses_tracked_style_double_create() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        _init_repo(repo)
        store = ReceiptStore(Path(tmp) / "data", workspace_root=repo)

        def worker(req: dict) -> dict:
            (Path(req["workspace_dir"]) / "new_module.py").write_text("A = 1\n", encoding="utf-8")
            return {"text": "done"}

        jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
        digest = jm.delegate_start("Add module", project="p", wait=True, auto_apply=False)
        (repo / "new_module.py").write_text("A = 2\n", encoding="utf-8")
        try:
            jm.delegate_apply(digest["job_id"])
            raise AssertionError("expected a conflict for a non-ignored file created on both sides")
        except ValueError as exc:
            assert "concurrently created" in str(exc)
        assert (repo / "new_module.py").read_text(encoding="utf-8") == "A = 2\n"


def _read(tmp: Path) -> dict:
    return {"tool_name": "Read", "tool_input": {"file_path": str(tmp / "mod.py")}, "cwd": str(tmp)}


def test_lead_gets_review_reads_only_for_jobs_from_this_session(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    data = tmp_path / ".cc-agy-data"
    (data / "receipts").mkdir(parents=True)
    env = {"CC_AGY_STRICT_MODE": "true", "CLAUDE_PROJECT_DIR": str(tmp_path), "CC_AGY_DATA_DIR": str(data)}

    # A job from an earlier session does not grant review reads.
    old = data / "receipts" / "job-0-old.diff"
    old.write_text("", encoding="utf-8")
    past = time.time() - 3600
    os.utime(old, (past, past))
    (data / ".lead_session_started").write_text("now", encoding="utf-8")

    assert evaluate_pre_tool_use(_read(tmp_path), env=env) == {}
    assert evaluate_pre_tool_use(_read(tmp_path), env=env) == {}
    denied = evaluate_pre_tool_use(_read(tmp_path), env=env)
    assert "LeadReadBudgetExceeded" in denied["hookSpecificOutput"]["permissionDecisionReason"]

    # A job from this session grants 6 more reads, then the cap applies again.
    (data / "receipts" / "job-1-new.diff").write_text("", encoding="utf-8")
    for _ in range(6):
        assert evaluate_pre_tool_use(_read(tmp_path), env=env) == {}
    denied = evaluate_pre_tool_use(_read(tmp_path), env=env)
    assert "LeadReadBudgetExceeded" in denied["hookSpecificOutput"]["permissionDecisionReason"]


def test_review_limit_env_override(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    data = tmp_path / ".cc-agy-data"
    (data / "receipts").mkdir(parents=True)
    (data / "receipts" / "job-1-abc.json").write_text("{}", encoding="utf-8")
    env = {
        "CC_AGY_STRICT_MODE": "true",
        "CC_AGY_LEAD_REVIEW_READ_LIMIT": "0",
        "CLAUDE_PROJECT_DIR": str(tmp_path),
        "CC_AGY_DATA_DIR": str(data),
    }
    assert evaluate_pre_tool_use(_read(tmp_path), env=env) == {}
    assert evaluate_pre_tool_use(_read(tmp_path), env=env) == {}
    assert evaluate_pre_tool_use(_read(tmp_path), env=env)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_inline_python_probe_of_project_code_stays_blocked(tmp_path: Path) -> None:
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    env = {"CC_AGY_STRICT_MODE": "true", "CLAUDE_PROJECT_DIR": str(tmp_path), "CC_AGY_DATA_DIR": str(tmp_path / ".d")}
    dec = evaluate_pre_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": "python3 -c \"import mypkg; print(mypkg)\""}, "cwd": str(tmp_path)},
        env=env,
    )
    assert dec["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_checklist_matches_auto_apply_mode() -> None:
    assert review_guidance_for({"CC_AGY_STRICT_MODE": "true"}) == REVIEW_AFTER_APPLY_GUIDANCE
    assert review_guidance_for({"CC_AGY_STRICT_MODE": "true", "CC_AGY_AUTO_APPLY_ON_PASS": "0"}) == REVIEW_BEFORE_APPLY_GUIDANCE
    assert review_guidance_for({}) == REVIEW_BEFORE_APPLY_GUIDANCE
    assert review_guidance_for({"CC_AGY_AUTO_APPLY_ON_PASS": "1"}) == REVIEW_AFTER_APPLY_GUIDANCE
    # After-apply flow must not tell the Lead to continue an applied job.
    assert "Never delegate_continue a job that was applied" in REVIEW_AFTER_APPLY_GUIDANCE
    assert "delegate_start" in REVIEW_AFTER_APPLY_GUIDANCE
    assert "not applied" in REVIEW_AFTER_APPLY_GUIDANCE
    # Before-apply flow tests the worker's version through acceptance, not host pytest.
    assert "acceptance" in REVIEW_BEFORE_APPLY_GUIDANCE
    for text in (REVIEW_AFTER_APPLY_GUIDANCE, REVIEW_BEFORE_APPLY_GUIDANCE):
        assert "one test per item" in text and "caller" in text and "python -c" not in text


def test_review_guidance_toggle() -> None:
    assert _review_guidance_enabled({})
    assert not _review_guidance_enabled({"CC_AGY_REVIEW_LOOP_GUIDANCE": "0"})


def test_session_start_resets_read_counter(tmp_path: Path) -> None:
    from scripts.session_start import handle_session_start

    repo = tmp_path / "repo"
    _init_repo(repo)
    data = tmp_path / "data"
    store = ReceiptStore(data, workspace_root=repo)
    (store.data_dir / ".lead_read_count").write_text("9", encoding="utf-8")
    out = handle_session_start({"cwd": str(repo), "session_id": "s1"}, env={"CC_AGY_STRICT_MODE": "true"}, store=store)
    assert (store.data_dir / ".lead_read_count").read_text(encoding="utf-8") == "0"
    assert (store.data_dir / ".lead_session_started").exists()
    assert out["hookSpecificOutput"]["additionalContext"] == REVIEW_AFTER_APPLY_GUIDANCE
