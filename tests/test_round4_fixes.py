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
    assert "verbatim task specification" in REVIEW_AFTER_APPLY_GUIDANCE


def test_flash_to_argon_escalation_on_followup_delegate_start_and_continue(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo_esc"
    _init_repo(repo)
    (repo / "app.py").write_text("def f(x: int) -> int:\n    return x\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add app"], cwd=str(repo), check=True)

    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")
    monkeypatch.setenv("CC_AGY_AUTO_APPLY_ON_PASS", "1")

    seen_models: list[str] = []

    def worker(req: dict[str, object]) -> dict[str, object]:
        seen_models.append(str(req["model"]))
        wt = Path(str(req["workspace_dir"]))
        (wt / "app.py").write_text(f"def f(x: int) -> int:\n    return x + {len(seen_models)}\n", encoding="utf-8")
        return {
            "text": "updated app.py",
            "usage": {"prompt_token_count": 100, "candidates_token_count": 20, "total_token_count": 120},
        }

    store = ReceiptStore(tmp_path / "data_esc", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        # Job 1 runs on gemini-3.8-flash and auto-applies when its initial acceptance command passes.
        d1 = jm.delegate_start(
            "Initial implementation",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["grep -q def app.py"],
            wait=True,
        )
        assert d1["model"] == "gemini-3.8-flash"
        assert d1["applied"] is True
        assert "escalated_from_model" not in d1

        # Job 2 is a follow-up delegate_start after post-apply review found an edge case:
        # it must automatically escalate from gemini-3.8-flash to gemini-4-argon.
        d2 = jm.delegate_start(
            "Fix edge case found during post-apply review",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["grep -q def app.py"],
            wait=True,
        )
        assert d2["model"] == "gemini-4-argon"
        assert d2["escalated_from_model"] == "gemini-3.8-flash"
        assert seen_models == ["gemini-3.8-flash", "gemini-4-argon"]
    finally:
        jm.shutdown()


def test_bounded_lead_delegate_start_auto_escalates_in_place_on_turn1_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """When Job 1 on Flash fails acceptance or hits checkpoint in a bounded Lead session, delegate_start(wait=True) auto-escalates in-place to Argon."""
    repo = tmp_path / "repo_esc_inplace"
    _init_repo(repo)
    (repo / "app.py").write_text("broken\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add app"], cwd=str(repo), check=True)

    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")
    monkeypatch.setenv("CC_AGY_AUTO_APPLY_ON_PASS", "1")

    seen_models: list[str] = []

    def worker(req: dict[str, object]) -> dict[str, object]:
        model = str(req["model"])
        seen_models.append(model)
        wt = Path(str(req["workspace_dir"]))
        if model == "gemini-3.8-flash":
            # Turn 1 on Flash hits the 100-tool-call checkpoint even though the initial acceptance command passes
            (wt / "app.py").write_text("def partial():\n    return 1\n", encoding="utf-8")
            return {
                "text": "hit 100 tool calls checkpoint",
                "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
                "usage": {"prompt_token_count": 100, "candidates_token_count": 20, "total_token_count": 120},
            }
        # Turn 2 on Argon fixes the code
        (wt / "app.py").write_text("def x():\n    return 2\n", encoding="utf-8")
        return {
            "text": "fixed on argon",
            "usage": {"prompt_token_count": 150, "candidates_token_count": 30, "total_token_count": 180},
        }

    store = ReceiptStore(tmp_path / "data_esc_inplace", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        d1 = jm.delegate_start(
            "Implement feature with Turn 1 failure on flash",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["grep -q def app.py"],
            wait=True,
        )
        assert seen_models == ["gemini-3.8-flash", "gemini-4-argon"]
        assert d1["model"] == "gemini-4-argon"
        assert d1["escalated_from_model"] == "gemini-3.8-flash"
        assert d1["acceptance_passed"] is True
        assert d1["applied"] is True
        raw_rec = store.read_receipt(d1["job_id"])
        assert len(raw_rec.get("prompts") or []) == 2
        assert raw_rec["model"] == "gemini-4-argon"
        assert raw_rec["escalated_from_model"] == "gemini-3.8-flash"
    finally:
        jm.shutdown()


def test_prepare_worktree_commits_tracked_gitignored_files_and_avoids_false_scope_violation(
    tmp_path: Path,
) -> None:
    """Tracked files that match .gitignore (e.g. Flask's tests/test_apps/.env) must be committed in baseline_sha."""
    repo = tmp_path / "repo_flask_env"
    _init_repo(repo)
    (repo / ".gitignore").write_text(".env\n.flaskenv\nsrc/mypkg/_version.py\n", encoding="utf-8")
    (repo / "src" / "mypkg").mkdir(parents=True)
    (repo / "src" / "mypkg" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "tests" / "test_apps").mkdir(parents=True)
    (repo / "tests" / "test_apps" / ".env").write_text("FOO=1\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "src/mypkg/__init__.py"], cwd=str(repo), check=True)
    subprocess.run(["git", "add", "-f", "tests/test_apps/.env"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init with tracked .env"], cwd=str(repo), check=True)

    def worker(req: dict[str, object]) -> dict[str, object]:
        wt = Path(str(req["workspace_dir"]))
        (wt / "src" / "mypkg" / "__init__.py").write_text("x = 2\n", encoding="utf-8")
        # Simulate setuptools-scm writing gitignored _version.py during test run
        (wt / "src" / "mypkg" / "_version.py").write_text("__version__ = '0.1.dev0'\n", encoding="utf-8")
        return {
            "text": "updated x",
            "usage": {"prompt_token_count": 50, "candidates_token_count": 10, "total_token_count": 60},
        }

    store = ReceiptStore(tmp_path / "data_flask_env", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        d = jm.delegate_start(
            "Update x in src/mypkg/__init__.py",
            model="gemini-4-argon",
            project="p",
            allowed_paths=["src/mypkg/__init__.py"],
            wait=True,
        )
        assert d["status"] == "ok"
        assert d.get("scope_violations", []) == []
        assert d["files_changed"] == ["src/mypkg/__init__.py"]
    finally:
        jm.shutdown()


def test_find_missing_prompt_symbols_ignores_builtins_and_mcp_fields(tmp_path: Path) -> None:
    from server.jobs import find_missing_prompt_symbols

    (tmp_path / "rest_api.py").write_text("def settle(balance: float) -> float:\n    return balance\n", encoding="utf-8")
    diff = (
        "--- a/rest_api.py\n+++ b/rest_api.py\n@@ -1,2 +1,2 @@\n"
        " def settle(balance: float) -> float:\n-    return balance\n+    return float(f'{balance:.2f}')\n"
    )
    prompt = "Round floats to 2 decimal places (`round(balance, 2)`) and respect `allowed_paths` and `files_changed`."
    missing = find_missing_prompt_symbols(
        prompt,
        diff,
        worktree_root=tmp_path,
        files_changed=["rest_api.py"],
    )
    assert "round" not in missing
    assert "allowed_paths" not in missing
    assert "files_changed" not in missing


def test_validate_acceptance_allows_workspace_pythonpath_bin_test_and_inline_pkg_import(tmp_path: Path) -> None:
    from server.jobs import validate_acceptance

    (tmp_path / "src" / "mypkg").mkdir(parents=True)
    (tmp_path / "src" / "mypkg" / "__init__.py").write_text("def ok(): return True\n", encoding="utf-8")
    (tmp_path / "bin").mkdir(parents=True)
    (tmp_path / "bin" / "test").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    cmds = [
        "PYTHONPATH=src pytest -q",
        ".venv/bin/python bin/test sympy/core/tests/test_assumptions.py",
        ".venv/bin/python -c \"import mypkg; assert mypkg.test()\"",
    ]
    assert validate_acceptance(cmds, tmp_path) == cmds

    for bad_cmd in (
        "PYTHONPATH=/tmp/outside pytest -q",
        "PYTHONPATH=../other pytest -q",
        "PYTHONPATH=$HOME pytest -q",
        "python ../other/runtests.py",
        "python3 -c \"import importlib; importlib.import_module('shutil').rmtree('src')\"",
        "python3 -c \"import mypkg; mypkg.shutil.rmtree('src')\"",
        "python3 -c \"import setup\"",
        "python3 -c \"import conftest\"",
    ):
        try:
            validate_acceptance([bad_cmd], tmp_path)
            raise AssertionError(f"Expected {bad_cmd!r} to be rejected")
        except ValueError:
            pass


def test_delegate_start_inlines_spec_files_context_files_lead_reads_and_checklist(
    tmp_path: Path,
) -> None:
    import json
    from scripts.pre_tool_guard import evaluate_pre_tool_use

    repo = tmp_path / "repo_inline"
    _init_repo(repo)
    (repo / "PROMPT.md").write_text("Implement `compute_magic(x)` returning `x * 2`.\n", encoding="utf-8")
    (repo / "helper.py").write_text("HELPER_CONST = 7\n", encoding="utf-8")
    (repo / "target.py").write_text("def compute_magic(x: int) -> int:\n    return x\n", encoding="utf-8")
    subprocess.run(["git", "add", "PROMPT.md", "helper.py", "target.py"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add spec and source files"], cwd=str(repo), check=True)

    data_dir = tmp_path / "data_inline"
    store = ReceiptStore(data_dir, workspace_root=repo)
    env = {
        "CC_AGY_STRICT_MODE": "true",
        "CC_AGY_LEAD_READ_LIMIT": "4",
        "CLAUDE_PROJECT_DIR": str(repo),
        "CC_AGY_DATA_DIR": str(data_dir),
    }
    decision = evaluate_pre_tool_use(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": str(repo / "helper.py")},
            "cwd": str(repo),
        },
        env=env,
    )
    assert decision == {}
    lead_rf = json.loads((data_dir / ".lead_read_files.json").read_text(encoding="utf-8"))
    assert lead_rf == ["helper.py"]

    captured_prompts: list[str] = []

    def worker(req: dict[str, object]) -> dict[str, object]:
        prompts = list(req.get("prompts") or [])
        captured_prompts.extend(str(p) for p in prompts)
        wt = Path(str(req["workspace_dir"]))
        (wt / "target.py").write_text("def compute_magic(x: int) -> int:\n    return x * 2\n", encoding="utf-8")
        return {
            "text": "implemented compute_magic",
            "usage": {"prompt_token_count": 80, "candidates_token_count": 15, "total_token_count": 95},
        }

    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        d = jm.delegate_start(
            "Implement task in target.py.",
            model="gemini-4-argon",
            project="p",
            spec_files=["PROMPT.md"],
            context_files=["target.py"],
            checklist=["`compute_magic(x)` returns `x * 2`"],
            wait=True,
        )
        assert d["status"] == "ok"
        assert len(captured_prompts) == 1
        wp = captured_prompts[0]
        # Brief is first, followed immediately by static prefix (Repository Map + Budget) before dynamic sections
        assert wp.startswith("Implement task in target.py.")
        assert wp.index("## Repository Map (Cached)") < wp.index("## Task Specification Files")
        assert "## Task Specification Files" in wp
        assert "Implement `compute_magic(x)` returning `x * 2`." in wp
        assert "## Behavioral Checklist" in wp
        assert "1. `compute_magic(x)` returns `x * 2`" in wp
        assert "## Pre-Read Context Files" in wp
        assert "HELPER_CONST = 7" in wp
        assert "def compute_magic(x: int) -> int:" in wp
    finally:
        jm.shutdown()


def test_straitjacket_full_profile_default_and_view_file_dedup(tmp_path: Path) -> None:
    import asyncio
    from server.agy_runner import build_straitjacket_tools, resolve_containment_profile

    assert resolve_containment_profile() == "full"
    assert resolve_containment_profile(disabled_tools=["CREATE_FILE", "EDIT_FILE", "RUN_COMMAND"]) == "read-only"
    assert resolve_containment_profile("command-output") == "command-output"

    repo = tmp_path / "repo_sj"
    _init_repo(repo)
    lines = [f"val_{i:03d} = {i}" for i in range(1, 301)]
    (repo / "mod.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tools, replaced = build_straitjacket_tools(repo, profile="full")
    assert replaced == ["RUN_COMMAND", "VIEW_FILE", "SEARCH_DIR", "LIST_DIR", "FIND_FILE"]
    tmap = {fn.__name__: fn for fn in tools}

    # First read returns actual lines
    r1 = asyncio.run(tmap["view_file"]("mod.py", start_line=1, end_line=250))
    assert "val_001 = 1" in r1
    assert "already in context" not in r1

    # Second read of a subset span at identical SHA-256 returns deduplicated 1-line pointer
    r2 = asyncio.run(tmap["view_file"]("mod.py", start_line=10, end_line=200))
    assert "already in context" in r2
    assert "val_050 = 50" not in r2

    # Modifying the file invalidates the SHA-256 cache and returns fresh content
    lines[0] = "val_001 = 999"
    (repo / "mod.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    r3 = asyncio.run(tmap["view_file"]("mod.py", start_line=1, end_line=250))
    assert "val_001 = 999" in r3
    assert "already in context" not in r3


def test_worker_acceptance_reuse_and_orphaned_job_scope_check(tmp_path: Path) -> None:
    import json
    import time
    from server.agy_runner import sign_worker_run_record

    repo = tmp_path / "repo_reuse"
    _init_repo(repo)
    (repo / "other.py").write_text("y = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "other.py"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add other.py"], cwd=str(repo), check=True)

    def worker(req: dict[str, object]) -> dict[str, object]:
        import hashlib
        wt = Path(str(req["workspace_dir"]))
        app_bytes = b"def greet() -> str:\n    return 'hi'\n"
        (wt / "app.py").write_bytes(app_bytes)
        time.sleep(0.02)
        rec: dict[str, object] = {
            "ts": round(time.time(), 6),
            "command": "pytest -q",
            "working_directory": "",
            "exit_code": 0,
            "timed_out": False,
            "duration_seconds": 0.03,
            "output_tail": "2 passed in 0.03s\n",
            "present_files": ["app.py", "other.py"],
            "file_hashes": {"app.py": hashlib.sha256(app_bytes).hexdigest()},
        }
        rec["hmac"] = sign_worker_run_record(rec)
        return {
            "text": "done",
            "usage": {"prompt_token_count": 40, "candidates_token_count": 10, "total_token_count": 50},
            "command_runs": [rec],
        }

    store = ReceiptStore(tmp_path / "data_reuse", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        d = jm.delegate_start(
            "Update greet in app.py",
            model="gemini-4-argon",
            project="p",
            acceptance=["pytest -q"],
            wait=True,
        )
        assert d["status"] == "ok"
        assert d["acceptance_passed"] is True
        raw_rec = store.read_receipt(d["job_id"])
        assert raw_rec["acceptance_results"][0].get("reused_from_worker") is True
        assert raw_rec["acceptance_results"][0].get("timed_out") is False
        assert raw_rec["acceptance_results"][0].get("duration_seconds") == 0.03

        # Verify forged unsigned worker_runs.jsonl is rejected
        forged_wt = tmp_path / "forged_wt"
        forged_wt.mkdir()
        (forged_wt / "app.py").write_text("broken(\n", encoding="utf-8")
        (forged_wt / ".ctx-session-reads").mkdir()
        (forged_wt / ".ctx-session-reads" / "worker_runs.jsonl").write_text(
            json.dumps({"ts": time.time() + 1000, "command": "pytest -q", "exit_code": 0, "output_tail": "1 passed"})
            + "\n",
            encoding="utf-8",
        )
        assert JobManager._try_reuse_worker_acceptance(forged_wt, ["app.py"], ["pytest -q"]) is None

        # Simulate an orphaned "running" job that touched out-of-scope `other.py`
        wt_orphan, base_sha, base_hashes = jm._prepare_worktree("job-orphan-1")
        (wt_orphan / "other.py").write_text("y = 2\n", encoding="utf-8")
        store.write_receipt(
            "job-orphan-1",
            {
                "job_id": "job-orphan-1",
                "kind": "delegate",
                "status": "running",
                "allowed_paths": ["app.py"],
                "worktree_dir": str(wt_orphan),
                "baseline_commit_sha": base_sha,
                "baseline_file_hashes": base_hashes,
                "files_changed": [],
                "scope_violations": [],
                "applied": False,
            },
            workspace_root=str(repo),
        )
        try:
            jm.delegate_apply("job-orphan-1")
            raise AssertionError("Expected orphan job touching out-of-scope other.py to be rejected")
        except ValueError as exc:
            assert "scope_violation" in str(exc) or "out-of-scope" in str(exc)
    finally:
        jm.shutdown()


def test_stale_prior_session_receipt_does_not_escalate_turn1_and_multimodel_cost_math(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify (1) stale receipts before .lead_session_started do not escalate Turn 1,
    (2) passing delegate_continue stays on Flash while failing/checkpoint delegate_continue
    escalates to Argon with 100 tool calls and diffstat handoff, and (3) multi-model requests
    are priced per turn model."""
    repo = tmp_path / "repo_session_esc"
    _init_repo(repo)
    (repo / "app.py").write_text("def compute(x: int) -> int:\n    return x\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add app"], cwd=str(repo), check=True)

    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")

    data_dir = tmp_path / "data_session_esc"
    store = ReceiptStore(data_dir, workspace_root=repo)
    # Write a receipt from an older session (1 hour ago)
    store.write_receipt(
        "job-old-session",
        {
            "job_id": "job-old-session",
            "kind": "delegate",
            "status": "ok",
            "model": "gemini-3.8-flash",
            "started_at_epoch": time.time() - 3600,
        },
        workspace_root=str(repo),
    )
    marker = data_dir / ".lead_session_started"
    marker.write_text(str(time.time()), encoding="utf-8")

    seen_reqs: list[dict[str, object]] = []

    def worker(req: dict[str, object]) -> dict[str, object]:
        seen_reqs.append(dict(req))
        wt = Path(str(req["workspace_dir"]))
        call_idx = len(seen_reqs)
        (wt / "app.py").write_text(f"def compute(x: int) -> int:\n    return x + {call_idx}\n", encoding="utf-8")
        if call_idx == 2:
            # Simulate hitting the 40-tool-call Flash checkpoint on Turn 2
            return {
                "text": "checkpoint reached",
                "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
                "usage": {
                    "prompt_token_count": 100_000,
                    "cached_content_token_count": 0,
                    "candidates_token_count": 10_000,
                    "thoughts_token_count": 0,
                    "total_token_count": 110_000,
                },
            }
        return {
            "text": f"turn {call_idx}",
            "usage": {
                "prompt_token_count": 100_000,
                "cached_content_token_count": 0,
                "candidates_token_count": 10_000,
                "thoughts_token_count": 0,
                "total_token_count": 110_000,
            },
        }

    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        # Turn 1 of new session must NOT escalate despite job-old-session on disk, and uses 40 max_tool_calls for Flash
        d1 = jm.delegate_start(
            "Phase 1: update compute",
            model="gemini-3.8-flash",
            project="p",
            wait=True,
            auto_apply=False,
        )
        assert d1["model"] == "gemini-3.8-flash"
        assert "escalated_from_model" not in d1
        assert seen_reqs[0]["max_tool_calls"] == 40

        # Turn 2 via delegate_continue after a passing Turn 1 (status="ok") stays on Flash!
        d2 = jm.delegate_continue(
            d1["job_id"],
            "Phase 2: add another increment",
            project="p",
            wait=True,
            auto_apply=False,
        )
        assert d2["model"] == "gemini-3.8-flash"
        assert d2["status"] == "checkpoint"

        # Turn 3 via delegate_continue after Turn 2 hit "checkpoint" escalates to gemini-4-argon,
        # upgrades max_tool_calls to 100, and includes worktree diffstat in the continuation prompt!
        d3 = jm.delegate_continue(
            d1["job_id"],
            "Finish implementation after checkpoint",
            project="p",
            wait=True,
            auto_apply=False,
        )
        assert d3["model"] == "gemini-4-argon"
        assert d3["escalated_from_model"] == "gemini-3.8-flash"
        assert seen_reqs[2]["max_tool_calls"] == 100
        assert "Worktree diffstat from prior gemini-3.8-flash attempt:" in str(seen_reqs[2]["prompts"][0])

        # Verify multi-model pricing:
        # Turns 1 & 2 ran on gemini-3.8-flash (200k input @ $0.50/1M = $0.10, 20k output @ $3.00/1M = $0.06 -> $0.16)
        # Turn 3 ran on gemini-4-argon (100k input @ $2.00/1M = $0.20, 10k output @ $12.00/1M = $0.12 -> $0.32)
        # Combined cost must be $0.48 (48.0 cents), NOT $0.96 (all-Argon rate)!
        assert abs(float(d3["cost"]["total_cost_usd"]) - 0.48) < 1e-6
    finally:
        jm.shutdown()


def test_wait_slice_clamping_and_auto_apply_on_delegate_result(tmp_path: Path, monkeypatch) -> None:
    """When delegate_start(wait=True) exceeds CC_AGY_MAX_WAIT_SLICE_SECONDS, it returns status='running'
    with poll_hint to keep the host prompt cache warm, and delegate_result(wait=True) auto-applies on completion."""
    repo = tmp_path / "repo_slice"
    _init_repo(repo)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)

    monkeypatch.setenv("CC_AGY_MAX_WAIT_SLICE_SECONDS", "0.08")
    monkeypatch.setenv("CC_AGY_AUTO_APPLY_ON_PASS", "1")

    def slow_worker(req: dict[str, object]) -> dict[str, object]:
        time.sleep(0.30)
        wt = Path(str(req["workspace_dir"]))
        (wt / "app.py").write_text("x = 42\n", encoding="utf-8")
        return {"text": "done slow", "usage": {"prompt_token_count": 50, "candidates_token_count": 10, "total_token_count": 60}}

    store = ReceiptStore(tmp_path / "data_slice", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=slow_worker)
    try:
        d_slice = jm.delegate_start(
            "Update x to 42",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["grep -q 42 app.py"],
            wait=True,
        )
        assert d_slice["status"] == "running"
        assert "poll_hint" in d_slice

        d_final = jm.delegate_result(d_slice["job_id"], wait=True, timeout=5.0)
        assert d_final["status"] == "ok"
        assert d_final["applied"] is True
        assert (repo / "app.py").read_text(encoding="utf-8") == "x = 42\n"
        assert "Finish now in one concise sentence" in str(d_final.get("next_action") or "")
    finally:
        jm.shutdown()


def test_bash_workspace_reads_counted_and_clean_apply_skips_review_bonus_and_user_prompt_captured(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify (1) Bash workspace reads (sed/cat/rg/git diff) count against .lead_read_count and record
    .lead_read_files.json, (2) clean auto-applied .json receipt grants 0 extra review reads, and
    (3) UserPromptSubmit captures verbatim prompt for delegate_start when Lead paraphrases it."""
    from scripts.user_prompt_submit import record_user_prompt

    repo = tmp_path / "repo_bash_cap"
    _init_repo(repo)
    (repo / "mod.py").write_text("class Blueprint:\n    def register(self) -> int:\n        return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add mod"], cwd=str(repo), check=True)

    data_dir = tmp_path / "data_bash_cap"
    env = {
        "CC_AGY_STRICT_MODE": "true",
        "CLAUDE_PROJECT_DIR": str(repo),
        "CC_AGY_DATA_DIR": str(data_dir),
    }

    # 1. Two Bash file-inspection commands consume the 2-call Lead read budget and record mod.py
    assert (
        evaluate_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "sed -n '1,20p' mod.py"}, "cwd": str(repo)},
            env=env,
        )
        == {}
    )
    assert (
        evaluate_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "git diff"}, "cwd": str(repo)},
            env=env,
        )
        == {}
    )
    # 3rd Bash read is denied with LeadReadBudgetExceeded and Focused File Outline
    denied_bash = evaluate_pre_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": "cat mod.py"}, "cwd": str(repo)},
        env=env,
    )
    assert denied_bash["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "LeadReadBudgetExceeded" in denied_bash["hookSpecificOutput"]["permissionDecisionReason"]
    assert "Blueprint" in denied_bash["hookSpecificOutput"]["permissionDecisionReason"]

    # Record a detailed user prompt via UserPromptSubmit hook
    full_user_spec = (
        "Update Blueprint.register in mod.py so that empty blueprint names raise ValueError('name may not be empty'), "
        "preserve parent name prefixing with a dot separator, and return 99 when registered."
    )
    record_user_prompt({"prompt": full_user_spec, "cwd": str(repo), "session_id": "s1"}, env=env)

    seen_prompts: list[str] = []

    def worker(req: dict[str, object]) -> dict[str, object]:
        seen_prompts.append(str(req["prompts"][0]))
        wt = Path(str(req["workspace_dir"]))
        (wt / "mod.py").write_text("class Blueprint:\n    def register(self) -> int:\n        return 99\n", encoding="utf-8")
        return {"text": "updated Blueprint.register"}

    monkeypatch.setenv("CC_AGY_AUTO_APPLY_ON_PASS", "1")
    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        # Lead sends a short paraphrased prompt; worker_prompt must include both the Original User Request
        # AND mod.py in Pre-Read Context Files (captured from the Bash `sed` read above!)
        d = jm.delegate_start(
            "Update Blueprint.register in mod.py",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["grep -q 99 mod.py"],
            wait=True,
        )
        assert d["applied"] is True
        assert d["acceptance_passed"] is True
        assert "## Original User Request (Verbatim" in seen_prompts[0]
        assert "name may not be empty" in seen_prompts[0]
        assert "## Pre-Read Context Files" in seen_prompts[0]

        # Because the job auto-applied cleanly with acceptance_passed=True, post-apply Read is still denied
        denied_post = evaluate_pre_tool_use(
            {"tool_name": "Read", "tool_input": {"file_path": str(repo / "mod.py")}, "cwd": str(repo)},
            env=env,
        )
        assert denied_post["hookSpecificOutput"]["permissionDecision"] == "deny"
    finally:
        jm.shutdown()


def test_wait_slice_preserves_in_place_flash_to_argon_escalation_and_skip_worktree(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify:
    - B1: A wait-sliced Flash job that ends at checkpoint/failure still escalates in-place to Argon on delegate_result.
    - B2: Fast-seeded worktrees do not invent phantom diffs on --skip-worktree or --assume-unchanged files.
    - B3: Auto-apply with acceptance_passed=None omits the 'passed acceptance checks' next_action message.
    - B4 & compact: detect_bash_workspace_read expands globs/braces/subshells/redirects, counts multi-file reads,
      and SessionStart(source='compact') preserves .lead_read_count and .lead_session_started.
    """
    from scripts.pre_tool_guard import detect_bash_workspace_read
    from scripts.session_start import handle_session_start

    # 1. B2: skip-worktree and assume-unchanged files in host repo
    repo = tmp_path / "repo_b1_b2"
    _init_repo(repo)
    (repo / "big_asset.bin").write_text("payload\n", encoding="utf-8")
    (repo / "cfg.ini").write_text("k=committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add asset and cfg"], cwd=str(repo), check=True)
    subprocess.run(["git", "update-index", "--skip-worktree", "big_asset.bin"], cwd=str(repo), check=True)
    (repo / "big_asset.bin").unlink()
    (repo / "cfg.ini").write_text("k=LOCAL_SECRET\n", encoding="utf-8")
    subprocess.run(["git", "update-index", "--assume-unchanged", "cfg.ini"], cwd=str(repo), check=True)

    # 2. B1: Wait-sliced Flash -> Argon escalation
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")
    monkeypatch.setenv("CC_AGY_AUTO_APPLY_ON_PASS", "1")
    monkeypatch.setenv("CC_AGY_MAX_WAIT_SLICE_SECONDS", "0.08")

    seen_models: list[str] = []
    seen_timeouts: list[float] = []

    def worker(req: dict[str, object]) -> dict[str, object]:
        m = str(req["model"])
        seen_models.append(m)
        seen_timeouts.append(float(req.get("timeout_seconds") or 0.0))
        wt = Path(str(req["workspace_dir"]))
        if m == "gemini-3.8-flash":
            time.sleep(0.25)
            (wt / "app.py").write_text("def partial():\n    return 1\n", encoding="utf-8")
            return {
                "text": "checkpoint",
                "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
                "usage": {"prompt_token_count": 100, "candidates_token_count": 20, "total_token_count": 120},
            }
        (wt / "app.py").write_text("def fixed():\n    return 2\n", encoding="utf-8")
        return {
            "text": "fixed",
            "usage": {"prompt_token_count": 150, "candidates_token_count": 30, "total_token_count": 180},
        }

    store = ReceiptStore(tmp_path / "data_b1_b2", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        d1 = jm.delegate_start(
            "Implement fixed in app.py",
            model="gemini-3.8-flash",
            project="p",
            allowed_paths=["app.py"],
            acceptance=["grep -q fixed app.py"],
            timeout_seconds=900.0,
            wait=True,
        )
        assert d1["status"] == "running"
        d2 = jm.delegate_result(d1["job_id"], wait=True, timeout=5.0)
        assert d2["status"] == "ok"
        assert d2["model"] == "gemini-4-argon"
        assert d2["applied"] is True
        assert d2["files_changed"] == ["app.py"]
        assert d2.get("scope_violations", []) == []
        assert seen_models == ["gemini-3.8-flash", "gemini-4-argon"]
        assert seen_timeouts == [900.0, 900.0]
        assert "LOCAL_SECRET" not in str(d2.get("diff_preview") or "")

        # 3. B3: Auto-apply with no acceptance (acceptance_passed is None) omits next_action
        monkeypatch.setenv("CC_AGY_DEFAULT_ACCEPTANCE", "0")
        d3 = jm.delegate_start(
            "Touch notes.txt",
            model="gemini-4-argon",
            project="p",
            wait=True,
        )
        if d3["status"] == "running":
            d3 = jm.delegate_result(d3["job_id"], wait=True, timeout=5.0)
        assert d3.get("acceptance_passed") is None
        assert d3.get("next_action") is None
    finally:
        jm.shutdown()

    # 4. B4: Bash read detection across globs, braces, wrappers, subshells, redirects, and multi-file counting
    p4_repo = tmp_path / "p4_repo"
    (p4_repo / "src").mkdir(parents=True)
    for fname in ("events.py", "app.py", "keys.py"):
        (p4_repo / "src" / fname).write_text("x = 1\n", encoding="utf-8")
    assert len(detect_bash_workspace_read("cat src/*.py", p4_repo, cwd=p4_repo)) == 3
    assert len(detect_bash_workspace_read("sed -n '1,10p' src/{events,app}.py", p4_repo, cwd=p4_repo)) == 2
    assert len(detect_bash_workspace_read("cat src/{events,app}{,}{,}.py", p4_repo, cwd=p4_repo)) == 2
    assert detect_bash_workspace_read(f"rg foo {tmp_path / 'repo_b1_b2'}", p4_repo, cwd=p4_repo) == []
    assert detect_bash_workspace_read("timeout 5 cat src/events.py", p4_repo, cwd=p4_repo) == ["src/events.py"]
    assert detect_bash_workspace_read("bash -c 'cat src/events.py'", p4_repo, cwd=p4_repo) == ["src/events.py"]
    assert detect_bash_workspace_read('echo "$(cat src/events.py)"', p4_repo, cwd=p4_repo) == ["src/events.py"]
    assert detect_bash_workspace_read("while read l; do echo \"$l\"; done < src/events.py", p4_repo, cwd=p4_repo) == [
        "src/events.py"
    ]

    # 5. SessionStart with source='compact' does NOT reset .lead_read_count
    (store.data_dir / ".lead_read_count").write_text("2", encoding="utf-8")
    handle_session_start(
        {"cwd": str(repo), "session_id": "s1", "source": "compact"},
        env={"CC_AGY_STRICT_MODE": "true"},
        store=store,
    )
    assert (store.data_dir / ".lead_read_count").read_text(encoding="utf-8") == "2"









import pytest  # noqa: E402

_STRICT_ROUTE_KEYS = (
    "CC_AGY_STRICT_MODE",
    "CC_AGY_USERCFG_STRICT_MODE",
    "CLAUDE_PLUGIN_OPTION_STRICT_DELEGATION_MODE",
)


@pytest.mark.parametrize("strict_key", _STRICT_ROUTE_KEYS)
def test_strict_mode_behaves_identically_for_every_config_route(tmp_path: Path, monkeypatch, strict_key: str) -> None:
    """Strict mode set via shell env, `.mcp.json` userConfig, or Claude plugin option must enable the same behavior."""
    from scripts.pre_tool_guard import is_strict_mode_enabled as hook_strict
    from scripts.session_start import _auto_apply_on_pass
    from server.mcp_stdio import CORE_TOOL_NAMES, get_tool_definitions
    from server.regions import STRICT_MODE_ENV_KEYS, is_strict_mode_enabled

    for k in (
        *STRICT_MODE_ENV_KEYS,
        "CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE",
        "CC_AGY_FORCE_WORKER_THINKING_LEVEL",
        "CC_AGY_MAX_WAIT_SLICE_SECONDS",
        "CC_AGY_SESSION_DEADLINE_SECONDS",
        "CC_AGY_AUTO_APPLY_ON_PASS",
        "CC_AGY_TOOL_PROFILE",
    ):
        monkeypatch.delenv(k, raising=False)
    assert set(_STRICT_ROUTE_KEYS) == set(STRICT_MODE_ENV_KEYS)
    assert hook_strict is is_strict_mode_enabled
    assert not is_strict_mode_enabled()

    monkeypatch.setenv(strict_key, "true")
    env = {strict_key: "true"}
    assert is_strict_mode_enabled()
    assert is_strict_mode_enabled(env)
    assert _auto_apply_on_pass(env) is True
    assert {t["name"] for t in get_tool_definitions(env)} == set(CORE_TOOL_NAMES)

    repo = tmp_path / "repo"
    _init_repo(repo)
    jm = JobManager(workspace_root=repo, store=ReceiptStore(tmp_path / "data", workspace_root=repo))
    try:
        assert jm._is_bounded_lead_session() is True
        assert jm._resolve_wait_slice_seconds(900.0) == 240.0
        assert jm._should_auto_apply(None) is True
        # Strict mode alone never imposes a session deadline (interactive sessions stay unclamped);
        # only an explicit CC_AGY_SESSION_DEADLINE_SECONDS from the harness does.
        jm._created_at -= 3600.0
        assert jm._clamp_worker_timeout(900.0) == 900.0
        monkeypatch.setenv("CC_AGY_SESSION_DEADLINE_SECONDS", "1500")
        assert jm._clamp_worker_timeout(900.0) == 45.0
    finally:
        jm.shutdown()


def test_env_flag_accepts_on_off_for_web_tools_and_dirty_worktree(tmp_path: Path, monkeypatch) -> None:
    from server.regions import env_flag

    for raw, expected in (("on", True), ("YES", True), ("1", True), ("off", False), ("No", False), ("0", False), ("", None), ("maybe", None)):
        assert env_flag("X_FLAG", {"X_FLAG": raw}) is expected, raw
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "on")
    repo = tmp_path / "repo"
    _init_repo(repo)
    jm = JobManager(
        workspace_root=repo,
        store=ReceiptStore(tmp_path / "data", workspace_root=repo),
        worker_backend=lambda req: {"text": ""},
    )
    try:
        assert jm._is_bounded_lead_session() is True
    finally:
        jm.shutdown()


def test_round7_straitjacket_budgets_and_post_acceptance_stop_guard(tmp_path: Path, monkeypatch) -> None:
    """Verify Round 7 P0 worker optimizations + Opus R6-4 refinements:
    1. Straitjacket view_file reads a 370-line file in 1 call without 130-line pagination, and ctx_get defaults to 300 lines.
    2. Post-acceptance stop guard appends ACCEPTANCE_PASSED_STOP_NOW only after real modified-worktree acceptance passes.
    3. After acceptance passes (R6-4):
       - read-only tools (`view_file`) still work so multi-requirement workers can inspect remaining files;
       - non-test shell commands (`rm -f ...`, `python3 -c ...`) still execute;
       - redundant broad test runs / `git diff` short-circuit with `status='skipped'`, `exit_code=None` (never fake `0`);
       - editing a file resets the guard automatically.
    """
    import asyncio
    import json
    import sys
    from server.agy_runner import build_straitjacket_tools

    repo = tmp_path / "repo_sj_r7"
    _init_repo(repo)
    big_lines = [f"def func_{i}(x: int) -> int:\n    return x + {i}\n" for i in range(185)]
    (repo / "big_mod.py").write_text("".join(big_lines), encoding="utf-8")  # 370 lines
    (repo / "test_mod.py").write_text(
        "from big_mod import func_0\ndef test_ok():\n    assert func_0(1) == 1\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add big_mod"], cwd=str(repo), check=True)

    tools, _ = build_straitjacket_tools(
        repo,
        timeout_seconds=120.0,
        profile="full",
        acceptance_commands=["python3 -m pytest -q test_mod.py"],
    )
    tool_map = {fn.__name__: fn for fn in tools}

    async def _exercise() -> None:
        # 1. Single default view_file reads all 370 lines without pagination
        out_view = await tool_map["view_file"]("big_mod.py")
        assert "L1:" in out_view
        assert "L370:" in out_view

        # 2. Running acceptance on an UNMODIFIED worktree does NOT activate stop guard
        unmod_out = await tool_map["run_command"]("python3 -m pytest -q test_mod.py")
        assert "ACCEPTANCE_PASSED_STOP_NOW" not in unmod_out

        # 3. Modify big_mod.py, verify 0-tests-collected does NOT trigger stop guard
        (repo / "big_mod.py").write_text("".join(big_lines) + "\nADDED_CONST = 77\n", encoding="utf-8")
        tools_fake_zero, _ = build_straitjacket_tools(
            repo,
            timeout_seconds=120.0,
            profile="full",
            acceptance_commands=["python3 -c \"print('collected 0 items')\""],
        )
        fz_out = await {fn.__name__: fn for fn in tools_fake_zero}["run_command"](
            "python3 -c \"print('collected 0 items')\""
        )
        assert "ACCEPTANCE_PASSED_STOP_NOW" not in fz_out

        # 4. Run real passing acceptance command on modified worktree -> triggers ACCEPTANCE_PASSED_STOP_NOW
        pass_out = await tool_map["run_command"]("python3 -m pytest -q test_mod.py")
        assert "ACCEPTANCE_PASSED_STOP_NOW" in pass_out

        # 5. R6-4: Subsequent broad suite run and `git diff` short-circuit WITHOUT spawning shell,
        # returning honest `status='skipped'` and `exit_code=None` (never fake `exit_code=0`).
        marker_file = repo / "should_not_exist.txt"
        blocked_suite_raw = await tool_map["run_command"](f"touch {marker_file.name} && python3 -m pytest -q")
        blocked_suite = json.loads(blocked_suite_raw)
        assert blocked_suite["status"] == "skipped"
        assert blocked_suite["exit_code"] is None
        assert blocked_suite["short_circuited"] is True
        assert "Command NOT executed" in blocked_suite["message"]
        assert not marker_file.exists()

        blocked_git_diff = json.loads(await tool_map["run_command"]("git diff"))
        assert blocked_git_diff["status"] == "skipped"
        assert blocked_git_diff["exit_code"] is None

        # 6. R6-4 + R7-1: Read-only `view_file`, non-test shell commands (`cat pytest.ini`, `grep pytest`,
        # `git show HEAD:...`, `python -c 'import pytest...'`), AND new targeted caller test files NOT in
        # acceptance (`pytest test_callers.py`, `.venv/bin/pytest test_callers.py`) STILL execute!
        # Meanwhile equivalent spellings of already-passed acceptance (`.venv/bin/pytest test_mod.py`)
        # and whole-suite runs (`.venv/bin/pytest -q`) short-circuit consistently!
        (repo / "pytest.ini").write_text("[pytest]\ntestpaths = .\n", encoding="utf-8")
        (repo / "test_callers.py").write_text(
            "from big_mod import func_1\ndef test_caller():\n    assert func_1(2) == 3\n",
            encoding="utf-8",
        )
        (repo / "tests" / "callers").mkdir(parents=True, exist_ok=True)
        (repo / "tests" / "callers" / "test_sub_caller.py").write_text(
            "from big_mod import func_2\ndef test_sub_caller():\n    assert func_2(1) == 3\n",
            encoding="utf-8",
        )
        # Re-establish active guard on current modified worktree (after adding pytest.ini + caller tests)
        re_pass = await tool_map["run_command"]("python3 -m pytest -q test_mod.py")
        assert "ACCEPTANCE_PASSED_STOP_NOW" in re_pass

        still_readable = await tool_map["view_file"]("test_mod.py", fresh=True)
        assert "ACCEPTANCE_PASSED_STOP_NOW" not in still_readable
        assert "test_ok" in still_readable

        for allowed_cmd, expected_substr in [
            ("cat pytest.ini", "[pytest]"),
            ("grep -n pytest pytest.ini", "[pytest]"),
            ("python3 -c 'import pytest; print(\"py_import_ok\")'", "py_import_ok"),
            ("git show HEAD:big_mod.py | head -n 2", "def func_0"),
            ("python3 -m pytest -q test_callers.py", "1 passed"),
            (f"{Path(sys.executable).parent / 'pytest'} test_callers.py", "1 passed"),
            (f"{Path(sys.executable).parent / 'pytest'} tests/callers/", "1 passed"),
        ]:
            cmd_res = await tool_map["run_command"](allowed_cmd)
            assert '"short_circuited": true' not in cmd_res, f"Unexpectedly skipped: {allowed_cmd}"
            assert expected_substr in cmd_res, f"Missing {expected_substr!r} in {allowed_cmd}: {cmd_res}"

        # Equivalent spelling of already-passed `test_mod.py`, top-level `tests/`, or whole-suite `.venv/bin/pytest` IS skipped
        for skipped_cmd in [
            ".venv/bin/pytest test_mod.py",
            "pytest -v test_mod.py",
            "pytest tests/",
            ".venv/bin/pytest -q",
            "git status",
            "git log -n 1 | head -n 5",
        ]:
            sk_res = json.loads(await tool_map["run_command"](skipped_cmd))
            assert sk_res["status"] == "skipped", f"Expected skipped for {skipped_cmd}: {sk_res}"
            assert sk_res["exit_code"] is None

        rm_out = await tool_map["run_command"]("rm -f nonexistent_scratch.tmp")
        assert '"short_circuited": true' not in rm_out
        helper_out = await tool_map["run_command"]("python3 -c \"print('helper_ok')\"")
        assert "helper_ok" in helper_out

        # 7. Editing big_mod.py again automatically resets the guard so pytest runs for real!
        (repo / "big_mod.py").write_text("".join(big_lines) + "\nADDED_CONST = 88\n", encoding="utf-8")
        re_run_pytest = await tool_map["run_command"]("python3 -m pytest -q test_mod.py")
        assert '"short_circuited": true' not in re_run_pytest
        assert "ACCEPTANCE_PASSED_STOP_NOW" in re_run_pytest

    asyncio.run(_exercise())


def test_round7_post_tool_use_autostop_and_slim_applied_digest(tmp_path: Path, monkeypatch) -> None:
    """Verify Round 7 P0 Lead optimizations + Opus R6-1 & R6-3 fixes:
    1. Auto-applied digest with explicit passing acceptance is <180 tokens, omits diff_preview/usage bloat,
       and preserves a concise 1-line `summary` (plus `progress`/`has_worker_caveat` if BLOCKED or `?`).
    2. Default strict mode emits non-blocking `hookSpecificOutput.additionalContext` so multi-part requests stay safe,
       while `CC_AGY_AUTO_STOP_ON_APPLY=1` emits `{'continue': False, 'stopReason': ...}`, and `delegate_apply` is ignored.
    3. Worker caveats (`BLOCKED` or `?` in summary) suppress both `additionalContext` and `continue: False`.
    4. Split-line Claude Code JSONL transcripts (multiple `type: assistant` lines sharing one `message.id`, R6-1)
       suppress completion until every sibling `tool_use` has completed.
    """
    import json
    from scripts.post_tool_ledger import evaluate_post_tool_use
    from server.regions import estimate_tokens

    repo = tmp_path / "repo_autostop"
    _init_repo(repo)
    (repo / "app.py").write_text("def calc(x: int) -> int:\n    return x\n", encoding="utf-8")
    (repo / "test_app.py").write_text(
        "import app\ndef test_calc():\n    assert app.calc(3) == 6\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init app"], cwd=str(repo), check=True)

    monkeypatch.setenv("CC_AGY_STRICT_MODE", "true")

    def worker(req: dict[str, object]) -> dict[str, object]:
        wt = Path(str(req["workspace_dir"]))
        (wt / "app.py").write_text("def calc(x: int) -> int:\n    return x * 2\n", encoding="utf-8")
        return {
            "text": "Updated calc in app.py with detailed explanation " * 20,
            "usage": {"prompt_token_count": 5000, "candidates_token_count": 300, "total_token_count": 5300},
        }

    store = ReceiptStore(tmp_path / "data_autostop", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        slim_digest = jm.delegate_start(
            "Double x in calc",
            model="gemini-4-argon",
            project="p",
            acceptance=["python3 -m pytest -q test_app.py"],
            wait=True,
        )
        assert slim_digest["applied"] is True
        assert slim_digest["acceptance_passed"] is True
        assert slim_digest["real_acceptance_passed"] is True
        assert "diff_preview" not in slim_digest
        assert "usage" not in slim_digest
        assert isinstance(slim_digest.get("summary"), str) and slim_digest["summary"]
        assert estimate_tokens(json.dumps(slim_digest)) < 180

        strict_env = {
            "CC_AGY_STRICT_MODE": "true",
            "CLAUDE_PROJECT_DIR": str(repo),
            "CC_AGY_DATA_DIR": str(store.data_dir),
        }
        hard_stop_env = {**strict_env, "CC_AGY_AUTO_STOP_ON_APPLY": "1"}

        # Default strict mode -> non-blocking hookSpecificOutput.additionalContext (R6-3)
        soft_dec = evaluate_post_tool_use(
            {
                "tool_name": "mcp__plugin_cc-agy-plugin_gemini-vertex__delegate_start",
                "tool_response": slim_digest,
                "cwd": str(repo),
            },
            env=strict_env,
            store=store,
        )
        assert "continue" not in soft_dec
        add_ctx = soft_dec.get("hookSpecificOutput", {}).get("additionalContext", "")
        assert "app.py" in add_ctx
        assert "unaddressed part" in add_ctx

        # Manual delegate_apply does NOT trigger completion/auto-stop (R6-3)
        assert (
            evaluate_post_tool_use(
                {
                    "tool_name": "mcp__gemini-vertex__delegate_apply",
                    "tool_response": slim_digest,
                    "cwd": str(repo),
                },
                env=hard_stop_env,
                store=store,
            )
            == {}
        )

        # Explicit opt-in CC_AGY_AUTO_STOP_ON_APPLY=1 -> hard continue=False across MCP content-list shape
        stop_dec_mcp = evaluate_post_tool_use(
            {
                "tool_name": "mcp__gemini-vertex__delegate_result",
                "tool_response": [{"type": "text", "text": json.dumps(slim_digest)}],
                "cwd": str(repo),
            },
            env=hard_stop_env,
            store=store,
        )
        assert stop_dec_mcp.get("continue") is False
        assert "app.py" in str(stop_dec_mcp.get("stopReason") or "")

        # Explicit opt-out CC_AGY_AUTO_STOP_ON_APPLY=0 returns {}
        assert (
            evaluate_post_tool_use(
                {
                    "tool_name": "mcp__gemini-vertex__delegate_start",
                    "tool_response": slim_digest,
                    "cwd": str(repo),
                },
                env={**strict_env, "CC_AGY_AUTO_STOP_ON_APPLY": "0"},
                store=store,
            )
            == {}
        )

        # Worker caveat (`BLOCKED` or `?` in summary) suppresses completion/auto-stop AND preserves caveat (R6-3)
        (repo / "app.py").write_text("def calc(x: int) -> int:\n    return x\n", encoding="utf-8")

        def caveat_worker(req: dict[str, object]) -> dict[str, object]:
            wt = Path(str(req["workspace_dir"]))
            (wt / "app.py").write_text("def calc(x: int) -> int:\n    return x * 2\n", encoding="utf-8")
            return {"text": "Step 1 done; Step 2 BLOCKED: should negative inputs raise ValueError?"}

        jm.worker_backend = caveat_worker
        caveat_digest = jm.delegate_start(
            "Double x in calc and clarify negatives",
            model="gemini-4-argon",
            project="p",
            acceptance=["python3 -m pytest -q test_app.py"],
            wait=True,
        )
        assert caveat_digest.get("has_worker_caveat") is True
        assert "BLOCKED" in str(caveat_digest.get("summary") or "")
        assert (
            evaluate_post_tool_use(
                {
                    "tool_name": "mcp__gemini-vertex__delegate_start",
                    "tool_response": caveat_digest,
                    "cwd": str(repo),
                },
                env=hard_stop_env,
                store=store,
            )
            == {}
        )
        assert (
            evaluate_post_tool_use(
                {
                    "tool_name": "mcp__gemini-vertex__delegate_start",
                    "tool_response": caveat_digest,
                    "cwd": str(repo),
                },
                env=strict_env,
                store=store,
            )
            == {}
        )

        # R6-1: Split-line Claude Code JSONL transcript (separate `type: assistant` lines sharing one `message.id`)
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_parallel_01",
                        "content": [{"type": "tool_use", "id": "tu_1", "name": "mcp__gemini-vertex__delegate_start"}],
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_parallel_01",
                        "content": [{"type": "tool_use", "id": "tu_2", "name": "mcp__gemini-vertex__delegate_start"}],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        assert (
            evaluate_post_tool_use(
                {
                    "tool_name": "mcp__gemini-vertex__delegate_start",
                    "tool_use_id": "tu_1",
                    "transcript_path": str(transcript),
                    "tool_response": slim_digest,
                    "cwd": str(repo),
                },
                env=hard_stop_env,
                store=store,
            )
            == {}
        )
        # Also suppressed in default strict mode while tu_2 is pending!
        assert (
            evaluate_post_tool_use(
                {
                    "tool_name": "mcp__gemini-vertex__delegate_start",
                    "tool_use_id": "tu_1",
                    "transcript_path": str(transcript),
                    "tool_response": slim_digest,
                    "cwd": str(repo),
                },
                env=strict_env,
                store=store,
            )
            == {}
        )
        # Once tu_1's tool_result is recorded and tu_2 completes, completion fires!
        with open(transcript, "a", encoding="utf-8") as tfh:
            tfh.write(
                json.dumps(
                    {
                        "type": "user",
                        "message": {"content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}]},
                    }
                )
                + "\n"
            )
        stop_after_sibling = evaluate_post_tool_use(
            {
                "tool_name": "mcp__gemini-vertex__delegate_start",
                "tool_use_id": "tu_2",
                "transcript_path": str(transcript),
                "tool_response": slim_digest,
                "cwd": str(repo),
            },
            env=hard_stop_env,
            store=store,
        )
        assert stop_after_sibling.get("continue") is False

        # Verbose opt-in preserves diff_preview
        jm.worker_backend = worker
        monkeypatch.setenv("CC_AGY_VERBOSE_APPLIED_DIGEST", "1")
        (repo / "app.py").write_text("def calc(x: int) -> int:\n    return x\n", encoding="utf-8")
        verbose_digest = jm.delegate_start(
            "Double x again",
            model="gemini-4-argon",
            project="p",
            acceptance=["python3 -m pytest -q test_app.py"],
            wait=True,
        )
        assert "diff_preview" in verbose_digest
    finally:
        jm.shutdown()


def test_round7_verified_timeout_promotion_and_dedup_prompt_and_compact_core_schema(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify Round 7 P1 & P2 optimizations + Opus R6-2 & R6-5 fixes:
    1. A worker job that hits TimeoutError AFTER writing clean edits that pass EXPLICIT Lead acceptance checks promotes
       status='timeout' -> 'ok' (recovered_from_timeout=True) and auto-applies without escalating.
    2. R6-2: A worker job that hits TimeoutError when NO explicit Lead acceptance was supplied (only pre-existing
       `auto_discovered_default` repo tests pass) is NOT promoted from status='timeout' and does NOT set real_acceptance_passed!
    3. A worker job that hits TimeoutError with failing explicit acceptance remains status='timeout'.
    4. Candidate/context files already verbatim in raw_user_prompt are NOT duplicated in Pre-Read Context Files,
       and fully inlined files are omitted from Focused File Outline.
    5. R6-5: Strict-mode CORE_TOOL_NAMES exposes all 8 hinted delegation/recovery tools in <1,250 tokens.
    """
    import json
    from scripts.user_prompt_submit import record_user_prompt
    from server.mcp_stdio import CORE_TOOL_NAMES, get_tool_definitions
    from server.regions import estimate_tokens

    # R6-5: Check compact core-tool schema (8 delegation/recovery tools + doctor)
    strict_defs = get_tool_definitions(env={"CC_AGY_STRICT_MODE": "true"})
    assert set(t["name"] for t in strict_defs) == {
        "delegate_start",
        "delegate_continue",
        "delegate_apply",
        "delegate_result",
        "delegate_status",
        "delegate_cancel",
        "delegate_revert",
        "get_diff",
        "doctor",
    }
    assert set(CORE_TOOL_NAMES) == {t["name"] for t in strict_defs}
    assert estimate_tokens(json.dumps(strict_defs)) < 1250

    repo = tmp_path / "repo_timeout_promo"
    _init_repo(repo)
    spec_doc = (
        "# Feature Spec\n"
        "Implement `compute_total(items)` in `billing.py` so it sums positive integers and ignores negatives.\n"
    )
    (repo / ".docs").mkdir()
    (repo / ".docs" / "instructions.md").write_text(spec_doc, encoding="utf-8")
    (repo / "billing.py").write_text(
        "def compute_total(items: list[int]) -> int:\n    return 0\n\n"
        "def helper_unused(val: int) -> int:\n    return val + 1\n",
        encoding="utf-8",
    )
    (repo / "test_billing.py").write_text(
        "import billing\ndef test_compute_total():\n    assert billing.compute_total([3, -2, 5]) == 8\n",
        encoding="utf-8",
    )
    (repo / "test_existing_unrelated.py").write_text(
        "import billing\ndef test_helper_unused():\n    assert billing.helper_unused(2) == 3\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init billing"], cwd=str(repo), check=True)

    data_dir = tmp_path / "data_timeout_promo"
    monkeypatch.setenv("CC_AGY_STRICT_MODE", "true")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")
    monkeypatch.setenv("CC_AGY_INLINE_CANDIDATE_FILES", "1")

    record_user_prompt(
        {
            "prompt": f"Please implement the following task:\n\n{spec_doc}\nEnsure edge cases pass.",
            "cwd": str(repo),
            "session_id": "s_r7",
        },
        env={"CC_AGY_STRICT_MODE": "true", "CLAUDE_PROJECT_DIR": str(repo), "CC_AGY_DATA_DIR": str(data_dir)},
    )

    seen_models: list[str] = []
    seen_worker_prompts: list[str] = []

    def timed_out_but_passing_worker(req: dict[str, object]) -> dict[str, object]:
        seen_models.append(str(req["model"]))
        seen_worker_prompts.append(str(req["prompts"][0]))
        wt = Path(str(req["workspace_dir"]))
        (wt / "billing.py").write_text(
            "def compute_total(items: list[int]) -> int:\n    return sum(x for x in items if x > 0)\n\n"
            "def helper_unused(val: int) -> int:\n    return val + 1\n",
            encoding="utf-8",
        )
        raise TimeoutError("Antigravity worker timed out after 1200s while running slow extra suite")

    store = ReceiptStore(data_dir, workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=timed_out_but_passing_worker)
    try:
        d_ok = jm.delegate_start(
            "Implement compute_total in billing.py",
            model="gemini-3.8-flash",
            project="p",
            context_files=[".docs/instructions.md", "billing.py"],
            acceptance=["python3 -m pytest -q test_billing.py"],
            wait=True,
        )
        # Promoted from timeout -> ok on Flash without escalating to Argon!
        assert seen_models == ["gemini-3.8-flash"]
        assert d_ok["status"] == "ok"
        assert d_ok["applied"] is True
        assert d_ok.get("recovered_from_timeout") is True
        rec_ok = store.read_receipt(d_ok["job_id"])
        assert rec_ok["recovered_from_timeout"] is True

        # Prompt deduplication checks:
        wp = seen_worker_prompts[0]
        assert "## Original User Request" in wp
        # .docs/instructions.md was already verbatim inside raw_user_prompt, so it must NOT appear a 2nd time!
        assert wp.count("sums positive integers and ignores negatives") == 1
        # billing.py was fully inlined in Pre-Read Context Files, so Focused File Outline must skip billing.py!
        assert "## Pre-Read Context Files" in wp
        assert "`billing.py`" not in (wp.split("## Focused File Outline")[1].split("##")[0] if "## Focused File Outline" in wp else "")

        # R6-2: Timeout with NO explicit acceptance (only pre-existing auto_discovered_default test passes)
        # MUST remain status='timeout' and NOT auto-promote a half-finished worker!
        monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "")
        (repo / "billing.py").write_text(
            "def compute_total(items: list[int]) -> int:\n    return 0\n\n"
            "def helper_unused(val: int) -> int:\n    return val + 1\n",
            encoding="utf-8",
        )
        # Remove failing test_billing.py so auto_discovered_default pytest passes 100% on half-finished code
        (repo / "test_billing.py").unlink()
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
        subprocess.run(["git", "commit", "-qm", "keep only unrelated passing test"], cwd=str(repo), check=True)

        def half_finished_timeout_worker(req: dict[str, object]) -> dict[str, object]:
            wt = Path(str(req["workspace_dir"]))
            (wt / "billing.py").write_text(
                "def compute_total(items: list[int]) -> int:\n    return -1  # half-finished!\n\n"
                "def helper_unused(val: int) -> int:\n    return val + 1\n",
                encoding="utf-8",
            )
            raise TimeoutError("worker timed out mid-implementation")

        jm.worker_backend = half_finished_timeout_worker
        d_autodisc = jm.delegate_start(
            "Implement compute_total without explicit acceptance",
            model="gemini-4-argon",
            project="p",
            wait=True,
        )
        assert d_autodisc["status"] == "timeout"
        assert not d_autodisc.get("applied")
        assert not d_autodisc.get("recovered_from_timeout")
        assert not d_autodisc.get("real_acceptance_passed")

        # Now verify a timeout with FAILING explicit acceptance is also NOT promoted!
        def timed_out_failing_worker(req: dict[str, object]) -> dict[str, object]:
            wt = Path(str(req["workspace_dir"]))
            (wt / "billing.py").write_text("def compute_total(items):\n    return -999\n", encoding="utf-8")
            raise TimeoutError("worker timed out with broken code")

        jm.worker_backend = timed_out_failing_worker
        d_fail = jm.delegate_start(
            "Break compute_total",
            model="gemini-4-argon",
            project="p",
            acceptance=["python3 -c 'import sys; sys.exit(1)'"],
            wait=True,
        )
        assert d_fail["status"] == "timeout"
        assert not d_fail.get("applied")
        assert not d_fail.get("recovered_from_timeout")
    finally:
        jm.shutdown()
