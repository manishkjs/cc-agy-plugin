"""Tests for file-backed review bundles (review_bundle.py) and plugin pre-flight guardrails (harness_validator.py)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from server.agy_runner import WORKER_SYSTEM_PROMPT, build_worker_system_prompt
from server.harness_validator import (
    autotrigger_skills,
    check_always_on_budget,
    validate_hooks_config,
    validate_plugin_tree,
    validate_script_refs,
    validate_skills,
)
from server.jobs import _EPHEMERAL_CACHE_DIRS, _is_ephemeral_build_or_test_artifact
from server.regions import estimate_tokens
from server.review_bundle import (
    POINTER_PROMPT_MAX_TOKENS,
    REVIEW_DIR_NAME,
    build_review_evidence_dir,
    extract_enclosing_diff_slices,
)
from server.verify import _untracked

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_always_on_budget_passes_under_350_tokens() -> None:
    assert estimate_tokens(WORKER_SYSTEM_PROMPT) < 350
    assert check_always_on_budget([WORKER_SYSTEM_PROMPT], max_tokens=350) == []


def test_always_on_budget_fails_over_limit() -> None:
    bloated = "word " * 1600
    errors = check_always_on_budget([bloated], max_tokens=1200)
    assert len(errors) == 1
    assert "always-on prompt token budget exceeded" in errors[0]


def test_worker_system_prompt_stays_lean_without_injecting_skill_bodies() -> None:
    prompt_a = build_worker_system_prompt("/tmp/ws", profile="command-output", has_report_to_lead=True)
    prompt_b = build_worker_system_prompt("/tmp/ws", profile="read-only", has_report_to_lead=False)
    assert estimate_tokens(prompt_a) < 350
    assert estimate_tokens(prompt_b) < 350


def test_autotrigger_skills_matches_only_relevant_skills(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    (skills_dir / "pytest-debugging").mkdir(parents=True)
    (skills_dir / "pytest-debugging" / "SKILL.md").write_text(
        "---\n"
        "name: pytest-debugging\n"
        "description: Use when debugging failing pytest unit tests and assertion tracebacks.\n"
        "---\n"
        "# Pytest Debugging Guide\nRun pytest -q -k <test_name>.\n",
        encoding="utf-8",
    )
    (skills_dir / "sql-migrations").mkdir(parents=True)
    (skills_dir / "sql-migrations" / "SKILL.md").write_text(
        "---\n"
        "name: sql-migrations\n"
        "description: Use when authoring PostgreSQL schema migrations and alembic revisions.\n"
        "---\n"
        "# SQL Migration Guide\nAlways wrap DDL in transactions.\n",
        encoding="utf-8",
    )

    matched = autotrigger_skills(
        "Fix the failing pytest assertion traceback in tests/test_parser.py",
        skills_dir,
        allowed_paths=["tests/test_parser.py"],
    )
    assert [name for name, _body in matched] == ["pytest-debugging"]
    assert "Pytest Debugging Guide" in matched[0][1]

    unrelated = autotrigger_skills(
        "Update README typo",
        skills_dir,
        allowed_paths=["README.md"],
    )
    assert unrelated == []


def test_build_review_evidence_dir_writes_index_and_stays_under_500_tokens(tmp_path: Path) -> None:
    src_dir = tmp_path / "pkg"
    src_dir.mkdir(parents=True)
    calc_py = src_dir / "calc.py"
    calc_py.write_text(
        "def add(a: int, b: int) -> int:\n"
        "    result = a + b\n"
        "    return result\n\n"
        "def multiply(a: int, b: int) -> int:\n"
        "    return a * b\n",
        encoding="utf-8",
    )

    diff_text = (
        "--- a/pkg/calc.py\n"
        "+++ b/pkg/calc.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def add(a: int, b: int) -> int:\n"
        "-    result = a - b\n"
        "+    result = a + b\n"
        "     return result\n"
    )
    # Also include a large synthetic diff section to verify pointer prompt stays < 500 tokens
    large_diff = diff_text + ("\n" + "+# padding line\n" * 4000)

    bundle = build_review_evidence_dir(
        tmp_path,
        task_prompt="Fix `add(a, b)` in `pkg/calc.py` so it adds instead of subtracting.",
        diff_text=large_diff,
        test_output="1 passed in 0.03s",
        files_changed=["pkg/calc.py"],
    )

    review_dir = Path(bundle["review_dir"])
    assert review_dir == tmp_path / REVIEW_DIR_NAME
    assert (review_dir / "INDEX.md").is_file()
    assert (review_dir / "task.md").is_file()
    assert (review_dir / "changes.diff").is_file()
    assert (review_dir / "test_output.txt").is_file()
    assert (review_dir / "enclosing_slices.txt").is_file()

    assert bundle["pointer_prompt_tokens"] <= POINTER_PROMPT_MAX_TOKENS
    assert f"{REVIEW_DIR_NAME}/INDEX.md" in bundle["pointer_prompt"]
    assert "def add(a: int, b: int) -> int:" in bundle["enclosing_slices"]


def test_extract_enclosing_diff_slices_returns_enclosing_python_function(tmp_path: Path) -> None:
    mod = tmp_path / "service.py"
    mod.write_text(
        "class OrderProcessor:\n"
        "    def validate(self, amount: int) -> bool:\n"
        "        if amount <= 0:\n"
        "            return False\n"
        "        return True\n\n"
        "    def unrelated(self) -> str:\n"
        "        return 'ok'\n",
        encoding="utf-8",
    )
    diff_text = (
        "--- a/service.py\n"
        "+++ b/service.py\n"
        "@@ -3,3 +3,3 @@\n"
        "         if amount <= 0:\n"
        "-            return True\n"
        "+            return False\n"
        "         return True\n"
    )
    slices = extract_enclosing_diff_slices(tmp_path, diff_text)
    assert "service.py (def validate, L2-L5)" in slices
    assert "def validate(self, amount: int) -> bool:" in slices
    assert "def unrelated" not in slices


def test_cc_agy_review_excluded_from_ephemeral_and_untracked_filters(tmp_path: Path) -> None:
    assert REVIEW_DIR_NAME in _EPHEMERAL_CACHE_DIRS
    assert _is_ephemeral_build_or_test_artifact(f"{REVIEW_DIR_NAME}/INDEX.md")
    assert _is_ephemeral_build_or_test_artifact(f"{REVIEW_DIR_NAME}/changes.diff")

    subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
    (tmp_path / REVIEW_DIR_NAME).mkdir()
    (tmp_path / REVIEW_DIR_NAME / "INDEX.md").write_text("# Manifest\n", encoding="utf-8")
    (tmp_path / "real_new_file.py").write_text("x = 1\n", encoding="utf-8")

    untracked = _untracked(tmp_path)
    assert "real_new_file.py" in untracked
    assert all(not p.startswith(REVIEW_DIR_NAME) for p in untracked)


def test_validate_skills_catches_orphan_and_missing_trigger(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    (skills_dir / "orphan-skill").mkdir(parents=True)
    (skills_dir / "orphan-skill" / "SKILL.md").write_text(
        "---\n"
        "name: orphan-skill\n"
        "description: A helper skill with no trigger phrase.\n"
        "---\n"
        "Body text.\n",
        encoding="utf-8",
    )
    errors = validate_skills(
        skills_dir,
        always_on_text="Always use vertex tools for delegation.",
        require_trigger_in_always_on=True,
    )
    assert any("lacks a trigger phrase" in err for err in errors)
    assert any("orphan skill 'orphan-skill'" in err for err in errors)


def test_validate_script_refs_detects_missing_file_and_syntax_error(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    broken_py = scripts_dir / "broken.py"
    broken_py.write_text("def broken(:\n    pass\n", encoding="utf-8")

    doc = tmp_path / "SKILL.md"
    doc.write_text(
        "Run `python3 scripts/missing.py` and `python3 \"${CLAUDE_PLUGIN_ROOT}\"/scripts/broken.py`.\n",
        encoding="utf-8",
    )
    errors = validate_script_refs(tmp_path, [doc])
    assert any("missing referenced script 'scripts/missing.py'" in err for err in errors)
    assert any("syntax error in 'scripts/broken.py'" in err for err in errors)


def test_validate_hooks_config_detects_unknown_event(tmp_path: Path) -> None:
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir(parents=True)
    bad_hooks = hooks_dir / "hooks.json"
    bad_hooks.write_text(
        json.dumps(
            {
                "hooks": {
                    "InvalidEventName": [
                        {
                            "hooks": [
                                {"type": "command", "command": "echo hi"}
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    errors = validate_hooks_config(bad_hooks, plugin_root=tmp_path)
    assert len(errors) == 1
    assert "unknown hook event 'InvalidEventName'" in errors[0]


def test_validate_plugin_tree_passes_on_cc_agy_plugin_repo() -> None:
    assert validate_plugin_tree(REPO_ROOT) == []


def test_run_antigravity_worker_forwards_extra_tools_and_system_prompt(tmp_path: Path) -> None:
    from server.agy_runner import run_antigravity_worker

    captured: dict[str, object] = {}

    def dummy_tool(x: str) -> str:
        return x

    def fake_backend(req: dict[str, object]) -> dict[str, object]:
        captured.update(req)
        return {"text": "done", "containment": "straitjacket"}

    res = run_antigravity_worker(
        prompts=["hello"],
        workspace_dir=tmp_path,
        model="gemini-4-argon",
        project="test-proj",
        thinking_level="HIGH",
        disabled_tools=["CREATE_FILE", "EDIT_FILE", "RUN_COMMAND"],
        extra_tools=[dummy_tool],
        system_prompt="Custom Lead Prompt",
        backend=fake_backend,
    )
    assert res["text"] == "done"
    assert captured["extra_tools"] == [dummy_tool]
    assert captured["system_prompt"] == "Custom Lead Prompt"
    assert captured["thinking_level"] == "HIGH"


def test_straitjacket_run_command_preserves_pythonpath_and_blocks_root_find(
    tmp_path: Path, monkeypatch
) -> None:
    import asyncio
    from server.agy_runner import build_straitjacket_tools

    monkeypatch.setenv("CC_AGY_DISABLE_WEB_TOOLS", "true")
    pkg_dir = tmp_path / "mypkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("VAL = 42\n", encoding="utf-8")
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()

    tool_list, _ = build_straitjacket_tools(tmp_path)
    tools = {fn.__name__: fn for fn in tool_list}
    run_cmd = tools["run_command"]

    blocked_raw = asyncio.run(run_cmd(command="find / -name pytest 2>/dev/null"))
    blocked = json.loads(blocked_raw)
    assert blocked["status"] == "error"
    assert blocked["exit_code"] == 126

    ok_raw = asyncio.run(
        run_cmd(
            command='python3 -c "import mypkg, os; print(mypkg.VAL, os.path.dirname(mypkg.__file__))"',
            working_directory="sub",
        )
    )
    assert f"42 {pkg_dir}" in ok_raw


def test_straitjacket_lead_orchestrator_read_budget_cap(tmp_path: Path) -> None:
    import asyncio
    from server.agy_runner import build_straitjacket_tools

    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    tool_list, _ = build_straitjacket_tools(
        tmp_path,
        profile="read-only",
        is_lead_orchestrator=True,
        lead_read_limit=6,
    )
    tools = {fn.__name__: fn for fn in tool_list}
    vf = tools["view_file"]

    out_1 = asyncio.run(vf(path="mod.py"))
    assert "[Lead budget:" not in out_1
    _ = asyncio.run(vf(path="mod.py"))
    out_3 = asyncio.run(vf(path="mod.py"))
    assert "[Lead budget: 3/6 read/search calls used" in out_3
    _ = asyncio.run(vf(path="mod.py"))
    _ = asyncio.run(vf(path="mod.py"))
    _ = asyncio.run(vf(path="mod.py"))
    out_7 = asyncio.run(vf(path="mod.py"))
    assert "LeadReadBudgetExceeded" in out_7

    from scripts.pre_tool_guard import evaluate_pre_tool_use

    env = {
        "CC_AGY_STRICT_MODE": "true",
        "CC_AGY_LEAD_READ_LIMIT": "6",
        "CLAUDE_PROJECT_DIR": str(tmp_path),
        "CC_AGY_DATA_DIR": str(tmp_path / ".cc-agy-data"),
    }
    for _ in range(6):
        dec = evaluate_pre_tool_use(
            {"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "mod.py")}, "cwd": str(tmp_path)},
            env=env,
        )
        assert dec == {}
    dec_7 = evaluate_pre_tool_use(
        {"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "mod.py")}, "cwd": str(tmp_path)},
        env=env,
    )
    assert dec_7["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "LeadReadBudgetExceeded" in dec_7["hookSpecificOutput"]["permissionDecisionReason"]


def _init_temp_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo), check=True)
    (repo / "app.py").write_text("def greet() -> str:\n    return 'hello'\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)


def test_auto_apply_in_delegate_start_and_continue_and_shutdown_salvage(
    tmp_path: Path, monkeypatch
) -> None:
    import threading
    from server.jobs import JobManager
    from server.receipts import ReceiptStore

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_temp_git_repo(repo)
    store = ReceiptStore(tmp_path / "data", workspace_root=repo)

    def ok_worker(req: dict[str, object]) -> dict[str, object]:
        wt = Path(str(req["workspace_dir"]))
        (wt / "app.py").write_text("def greet() -> str:\n    return 'auto_applied'\n", encoding="utf-8")
        return {"text": "done"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=ok_worker)
    res = jm.delegate_start(
        "Update greet in app.py",
        project="test-proj",
        wait=True,
        auto_apply=True,
        acceptance=["python3 -m py_compile app.py"],
    )
    assert res["status"] == "ok"
    assert res["applied"] is True
    assert res["applied_files"] == ["app.py"]
    assert "auto_applied" in (repo / "app.py").read_text(encoding="utf-8")
    jm.shutdown()

    # Shutdown salvage on an interrupted running worker in a bounded lead session
    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    _init_temp_git_repo(repo2)
    store2 = ReceiptStore(tmp_path / "data2", workspace_root=repo2)
    gate = threading.Event()
    written = threading.Event()

    def slow_worker(req: dict[str, object]) -> dict[str, object]:
        wt = Path(str(req["workspace_dir"]))
        (wt / "app.py").write_text("def greet() -> str:\n    return 'salvaged_from_wt'\n", encoding="utf-8")
        written.set()
        gate.wait(5)
        return {"text": "late"}

    monkeypatch.setenv("CC_AGY_SALVAGE_ON_SHUTDOWN", "1")
    jm2 = JobManager(workspace_root=repo2, store=store2, worker_backend=slow_worker)
    started = jm2.delegate_start("Update greet before shutdown", project="test-proj", wait=False)
    assert written.wait(5)
    interrupted_ids = jm2.mark_running_interrupted()
    assert interrupted_ids == [started["job_id"]]
    rec = store2.read_receipt(started["job_id"])
    assert rec["applied"] is True
    assert rec.get("salvaged_on_shutdown") is True
    assert "salvaged_from_wt" in (repo2 / "app.py").read_text(encoding="utf-8")
    gate.set()
    jm2.shutdown(wait=True)
    rec_after = store2.read_receipt(started["job_id"])
    assert rec_after["status"] == "interrupted"
    assert rec_after["applied"] is True
    assert rec_after.get("salvaged_on_shutdown") is True
    assert rec_after.get("applied_files") == ["app.py"]
    assert rec_after.get("applied_file_hashes", {}).get("app.py") is not None


def test_dirty_worktree_continuation_on_checkpoint_and_failed_acceptance_plus_next_action(
    tmp_path: Path, monkeypatch
) -> None:
    from server.jobs import JobManager
    from server.receipts import ReceiptStore

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_temp_git_repo(repo)
    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")

    calls: list[dict[str, object]] = []

    def multi_turn_worker(req: dict[str, object]) -> dict[str, object]:
        calls.append(req)
        wt = Path(str(req["workspace_dir"]))
        if len(calls) == 1:
            (wt / "app.py").write_text("def greet() -> str:\n    return 'part1'\n", encoding="utf-8")
            return {"text": "paused at checkpoint", "stop_reason": "MAX_TOOL_CALLS_EXCEEDED"}
        # Turn 2 must run in the SAME worktree where part1 already exists
        assert "part1" in (wt / "app.py").read_text(encoding="utf-8")
        (wt / "app.py").write_text("def greet() -> str:\n    return 'part2_complete'\n", encoding="utf-8")
        return {"text": "completed"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=multi_turn_worker)
    d1 = jm.delegate_start("Implement part1", project="test-proj", wait=True)
    assert d1["status"] == "checkpoint"
    assert "delegate_continue" in d1.get("next_action", "")

    # Calling delegate_start again in a bounded lead session automatically continues d1's dirty worktree
    d2 = jm.delegate_start("Finish part2", project="test-proj", wait=True)
    assert d2["job_id"] == d1["job_id"]
    assert d2["status"] == "ok"
    assert len(calls) == 2
    jm.shutdown()


def test_straitjacket_view_file_enforces_minimum_250_line_window(tmp_path: Path) -> None:
    import asyncio
    from server.agy_runner import build_straitjacket_tools

    lines = [f"line_{i:03d} = {i}" for i in range(1, 401)]
    (tmp_path / "large.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tool_list, _ = build_straitjacket_tools(tmp_path, profile="read-only")
    tools = {fn.__name__: fn for fn in tool_list}
    vf = tools["view_file"]

    # Request a tiny 16-line micro-slice (lines 50..65); view_file must widen it to >= 250 lines
    out = asyncio.run(vf(path="large.py", start_line=50, end_line=65))
    assert "line_030 = 30" in out
    assert "line_275 = 275" in out

    # Exact 250-line inclusive span (lines 10..259) must NOT be shifted backward to line 1
    out_exact = asyncio.run(vf(path="large.py", start_line=10, end_line=259))
    assert "line_009 = 9" not in out_exact
    assert "line_010 = 10" in out_exact
    assert "line_259 = 259" in out_exact
    assert "line_260 = 260" not in out_exact


def test_auto_discover_related_regression_tests_in_acceptance(tmp_path: Path) -> None:
    from server.jobs import JobManager, discover_related_test_commands
    from server.receipts import ReceiptStore

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_temp_git_repo(repo)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "engine.py").write_text("def compute(x: int) -> int:\n    return x * 2\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_new_feature.py").write_text(
        "from pkg.engine import compute\ndef test_new():\n    assert compute(3) == 6\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_engine.py").write_text(
        "from pkg.engine import compute\ndef test_existing_regression():\n    assert compute(0) == 0\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add engine and tests"], cwd=str(repo), check=True)

    extra = discover_related_test_commands(
        repo,
        files_changed=["pkg/engine.py", "tests/test_new_feature.py"],
        existing_cmds=["python3 -m pytest -q tests/test_new_feature.py"],
    )
    assert extra == ["python3 -m pytest -q tests/test_engine.py"]

    # Verify _execute_job runs the auto-discovered regression test and fails acceptance if broken
    store = ReceiptStore(tmp_path / "data", workspace_root=repo)

    def buggy_worker(req: dict[str, object]) -> dict[str, object]:
        wt = Path(str(req["workspace_dir"]))
        # Breaks compute(0) == 0 while passing compute(3) == 6
        (wt / "pkg" / "engine.py").write_text(
            "def compute(x: int) -> int:\n    return x * 2 if x > 0 else 999\n",
            encoding="utf-8",
        )
        return {"text": "edited engine"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=buggy_worker)
    res = jm.delegate_start(
        "Update compute in pkg/engine.py",
        project="test-proj",
        wait=True,
        acceptance=["python3 -m pytest -q tests/test_new_feature.py"],
    )
    assert res["status"] == "ok"
    assert res["acceptance_passed"] is False
    assert len(res["acceptance_results"]) == 2
    assert res["acceptance_results"][0]["passed"] is True
    assert res["acceptance_results"][1]["passed"] is False
    assert res["acceptance_results"][1].get("auto_discovered_regression") is True
    assert "delegate_continue" in res.get("next_action", "")
    jm.shutdown()


def test_find_missing_prompt_symbols_and_delegate_start_qa_review_model(tmp_path: Path) -> None:
    from server.jobs import JobManager, find_missing_prompt_symbols
    from server.mcp_stdio import MCPServer
    from server.receipts import ReceiptStore

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_temp_git_repo(repo)
    store = ReceiptStore(tmp_path / "data", workspace_root=repo)

    missing = find_missing_prompt_symbols(
        "Implement `parse_token` and ` ExecutionResult ` and `validate_payload(data)` in `app.py`.",
        diff_text="+def parse_token(x):\n+    return x\n",
        worktree_root=repo,
        files_changed=["app.py"],
    )
    assert missing == ["ExecutionResult", "validate_payload"]

    worker_calls: list[dict[str, object]] = []

    def two_pass_worker(req: dict[str, object]) -> dict[str, object]:
        worker_calls.append(req)
        wt = Path(str(req["workspace_dir"]))
        if len(worker_calls) == 1:
            (wt / "app.py").write_text("def parse_token(x):\n    return x\n", encoding="utf-8")
            return {"text": "partial"}
        (wt / "app.py").write_text(
            "def parse_token(x):\n    return x\n\ndef validate_payload(d):\n    return True\n",
            encoding="utf-8",
        )
        return {"text": "complete"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=two_pass_worker)
    srv = MCPServer(
        job_manager=jm,
        store=store,
        ask_backend=lambda _r: {
            "text": "VERDICT: FAIL\n- Missing `validate_payload` function in `app.py`.",
            "usage": {"prompt_token_count": 50, "candidates_token_count": 10, "total_token_count": 60},
        },
    )
    resp = srv.handle_Message(
        {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "delegate_start",
                "arguments": {
                    "prompt": "Implement `parse_token` and `validate_payload` in `app.py`.",
                    "project": "test-proj",
                    "wait": True,
                    "qa_review_model": "gemini-4-argon",
                    "auto_apply": True,
                },
            },
        }
    )
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["status"] == "ok"
    assert payload["qa_review"]["resumed"] is True
    assert payload["applied"] is True
    assert payload.get("missing_prompt_symbols") in (None, [])
    assert "validate_payload" in (repo / "app.py").read_text(encoding="utf-8")
    jm.shutdown()


def test_find_missing_prompt_symbols_ignores_symbols_already_in_baseline(tmp_path: Path) -> None:
    """Live DeepSWE regression: a prompt saying "do NOT extend `SwappableN`" and "just like
    `_DiverseFailableLawSpec`" flagged both existing classes as missing, which blocked auto-apply and
    cost the Lead two extra delegate_continue turns before it timed out."""
    from server.jobs import find_missing_prompt_symbols

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_temp_git_repo(repo)
    (repo / "failable.py").write_text(
        "class SwappableN:\n    pass\n\nclass _DiverseFailableLawSpec:\n    pass\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "baseline interfaces"], cwd=str(repo), check=True)
    (repo / "validated.py").write_text("class ValidatedLikeN:\n    pass\n", encoding="utf-8")

    prompt = (
        "Create `ValidatedLikeN` and `bind_validated`. Do NOT extend `SwappableN`; define law specs "
        "just like `_DiverseFailableLawSpec`."
    )
    missing = find_missing_prompt_symbols(
        prompt,
        diff_text="+class ValidatedLikeN:\n+    pass\n",
        worktree_root=repo,
        files_changed=["validated.py"],
    )
    # Existing baseline symbols are references; only the genuinely new, still-absent symbol is reported.
    assert missing == ["bind_validated"]

    # Without a git repo to consult, behaviour is unchanged (all absent symbols reported).
    no_repo = find_missing_prompt_symbols(prompt, diff_text="+class ValidatedLikeN:\n+    pass\n")
    assert no_repo == ["bind_validated", "SwappableN", "_DiverseFailableLawSpec"]


def test_opus_round1_symbol_shutdown_and_qa_regressions(tmp_path: Path) -> None:
    """Opus 5.5 review findings: TDD test imports must not hide missing symbols, the 10-item cap must
    apply after baseline filtering, `interrupted` must not stick across delegate_continue, shutdown must
    not relabel user cancels, and a QA preamble line must not flip PASS to NEEDS_FIXES."""
    import threading

    from server.jobs import JobManager, find_missing_prompt_symbols, parse_qa_audit_verdict
    from server.receipts import ReceiptStore

    # (a) A committed test that imports `compute_total` is a mention, not a definition.
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_temp_git_repo(repo)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_lib.py").write_text(
        "from lib import compute_total\n\ndef test_total():\n    assert compute_total([1]) == 1\n", encoding="utf-8"
    )
    (repo / "README.md").write_text("Call `compute_total` to sum values.\n", encoding="utf-8")
    existing = [f"existing_helper_{i:02d}" for i in range(11)]
    (repo / "core.py").write_text("".join(f"def {name}():\n    pass\n" for name in existing), encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "tdd baseline"], cwd=str(repo), check=True)
    assert find_missing_prompt_symbols(
        "Implement `compute_total` in `lib.py`.", diff_text="+x = 1\n", worktree_root=repo, files_changed=["lib.py"]
    ) == ["compute_total"]

    # (b) 11 references to existing helpers plus one genuinely new symbol: the new one must survive the cap.
    prompt_b = "Reuse " + ", ".join(f"`{n}`" for n in existing) + " and add `brand_new_entry`."
    assert find_missing_prompt_symbols(
        prompt_b, diff_text="+x = 1\n", worktree_root=repo, files_changed=["lib.py"]
    ) == ["brand_new_entry"]

    # (c) A preamble before the verdict line must not turn PASS into NEEDS_FIXES.
    assert parse_qa_audit_verdict("Reviewed the diff.\nVERDICT: PASS\nLooks good.")["passed"] is True
    fails = parse_qa_audit_verdict("Reviewed the diff.\nVERDICT: NEEDS_FIXES\n- missing guard")
    assert fails["passed"] is False and fails["issues"] == ["- missing guard"]

    # (d) A job marked `interrupted` by a server restart must end `ok` after a successful continue.
    store = ReceiptStore(tmp_path / "data", workspace_root=repo)

    def checkpoint_worker(req: dict[str, object]) -> dict[str, object]:
        (Path(str(req["workspace_dir"])) / "lib.py").write_text("def compute_total(x):\n    return 0\n", encoding="utf-8")
        return {"text": "cp", "stop_reason": "MAX_TOOL_CALLS_EXCEEDED"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=checkpoint_worker)
    job = jm.delegate_start("Implement compute_total", project="p", wait=True)
    assert job["status"] == "checkpoint"
    rec = store.read_receipt(job["job_id"])
    rec["status"] = "interrupted"
    store.write_receipt(job["job_id"], rec, workspace_root=str(repo))
    jm.shutdown()

    jm_restarted = JobManager(workspace_root=repo, store=store, worker_backend=lambda _r: {"text": "done"})
    cont = jm_restarted.delegate_continue(job["job_id"], "Finish compute_total", project="p", wait=True)
    assert cont["status"] == "ok"
    assert store.read_receipt(job["job_id"])["status"] == "ok"
    jm_restarted.shutdown()

    # (e) Shutdown while a cancelled job's thread is still exiting must keep `cancelled`.
    repo_e = tmp_path / "repo_e"
    repo_e.mkdir()
    _init_temp_git_repo(repo_e)
    store_e = ReceiptStore(tmp_path / "data_e", workspace_root=repo_e)
    release = threading.Event()
    started = threading.Event()

    def slow_worker(_req: dict[str, object]) -> dict[str, object]:
        started.set()
        release.wait(10)
        return {"text": "late"}

    jm_e = JobManager(workspace_root=repo_e, store=store_e, worker_backend=slow_worker)
    job_e = jm_e.delegate_start("Slow task", project="p", wait=False)
    assert started.wait(10)
    jm_e.delegate_cancel(job_e["job_id"], cleanup_worktree=False)
    jm_e.mark_running_interrupted()
    assert store_e.read_receipt(job_e["job_id"])["status"] == "cancelled"
    release.set()
    jm_e.shutdown()


def test_opus_round2_continue_owner_scoping_comments_and_qa_fail_closed(tmp_path: Path) -> None:
    """Opus 5.5 round-2 findings: a refused delegate_continue must not leave the receipt stuck at
    `running`; `Order.to_payload` must not be excused by `User.to_payload`; indented locals, kwargs and
    comments are not definitions; a format-reminder preamble must not win over the real verdict."""
    from server.jobs import JobManager, find_missing_prompt_symbols, parse_qa_audit_verdict
    from server.receipts import ReceiptStore

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_temp_git_repo(repo)
    (repo / "models.py").write_text(
        "class User:\n"
        "    def to_payload(self):\n"
        "        retry_budget = 3\n"
        "        return dict(\n"
        "            cache_window=retry_budget,\n"
        "        )\n"
        "\n"
        "# TODO: class CacheManager for later\n"
        "TOP_LEVEL_SETTING = 1\n"
        "default_timeout = 30\n",
        encoding="utf-8",
    )
    (repo / "orders.py").write_text("class Order:\n    pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "models"], cwd=str(repo), check=True)

    def missing_for(prompt: str) -> list[str]:
        return find_missing_prompt_symbols(prompt, diff_text="+x = 1\n", worktree_root=repo, files_changed=["lib.py"])

    # E1: method on a different class does not count; the same class's method does.
    assert missing_for("Add `Order.to_payload()` to orders.") == ["to_payload"]
    assert missing_for("Mirror `User.to_payload()` exactly.") == []
    # E2: indented local variable and a kwarg line are not definitions; a column-0 assignment is.
    assert missing_for("Add `retry_budget` and `cache_window` settings.") == ["retry_budget", "cache_window"]
    assert missing_for("Respect `default_timeout` everywhere.") == []
    # E3: a planned class named only in a comment is still missing.
    assert missing_for("Create `CacheManager`.") == ["CacheManager"]

    # QA: the last line starting with VERDICT wins, and an ISSUE: line forces NEEDS_FIXES.
    reminder = parse_qa_audit_verdict(
        "Format reminder: VERDICT: PASS means no bugs.\nVERDICT: NEEDS_FIXES\n- ISSUE: a.py:1 bug"
    )
    assert reminder["passed"] is False and reminder["issues"] == ["- ISSUE: a.py:1 bug"]
    contradictory = parse_qa_audit_verdict("**VERDICT: PASS**\nISSUE: off-by-one in a.py:3")
    assert contradictory["passed"] is False
    assert parse_qa_audit_verdict("VERDICT: PASS")["passed"] is True

    # C: continuing an interrupted job that was already salvaged/applied is refused and must leave the
    # receipt's interrupted status and error intact.
    store = ReceiptStore(tmp_path / "data", workspace_root=repo)

    def edit_worker(req: dict[str, object]) -> dict[str, object]:
        (Path(str(req["workspace_dir"])) / "lib.py").write_text("VALUE = 1\n", encoding="utf-8")
        return {"text": "done"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=edit_worker)
    job = jm.delegate_start("Add lib.py", project="p", wait=True, auto_apply=True)
    assert job["applied"] is True
    rec = store.read_receipt(job["job_id"])
    rec["status"] = "interrupted"
    rec["error"] = "server stopped"
    store.write_receipt(job["job_id"], rec, workspace_root=str(repo))
    try:
        jm.delegate_continue(job["job_id"], "more", project="p", wait=True)
        raised = False
    except ValueError:
        raised = True
    assert raised
    after = store.read_receipt(job["job_id"])
    assert after["status"] == "interrupted" and after["error"] == "server stopped"
    jm.shutdown()


def test_mcp_tool_profile_core_reduces_schema_bloat() -> None:
    from server.mcp_stdio import CORE_TOOL_NAMES, TOOL_DEFINITIONS, get_tool_definitions

    full_defs = get_tool_definitions(env={})
    core_defs = get_tool_definitions(env={"CC_AGY_TOOL_PROFILE": "core"})
    assert len(full_defs) == len(TOOL_DEFINITIONS)
    assert [t["name"] for t in core_defs] == sorted(
        [t["name"] for t in core_defs],
        key=lambda n: [t["name"] for t in TOOL_DEFINITIONS].index(n),
    )
    assert set(t["name"] for t in core_defs) == set(CORE_TOOL_NAMES)
    full_tokens = estimate_tokens(json.dumps(full_defs))
    core_tokens = estimate_tokens(json.dumps(core_defs))
    assert core_tokens < full_tokens * 0.5


def test_round1_argon_critique_regressions(tmp_path: Path, monkeypatch) -> None:
    from server.jobs import (
        JobManager,
        discover_related_test_commands,
        find_missing_prompt_symbols,
        parse_qa_audit_verdict,
    )
    from server.mcp_stdio import MCPServer
    from server.receipts import ReceiptStore

    # 1. find_missing_prompt_symbols: short <3-char backtick spans (`x`, `id`) must not phase-shift
    # pairing, and multi-argument signatures (`validate_payload(data, strict=True)`) must be extracted
    syms = find_missing_prompt_symbols(
        "When `x` is 0 or `id` is None, implement `compute_total` and `validate_payload(data, strict=True)`.",
        diff_text="+def other_helper():\n+    return 1\n",
    )
    assert syms == ["compute_total", "validate_payload"]

    # 2. discover_related_test_commands: preserve `uv run` and env prefixes, handle `-k`, and include
    # related test files even when also touched by the worker
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_temp_git_repo(repo)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "engine.py").write_text("def run(): return 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_engine.py").write_text("def test_run(): assert True\n", encoding="utf-8")
    (repo / "tests" / "test_other.py").write_text("def test_other(): assert True\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add pkg and tests"], cwd=str(repo), check=True)

    uv_cmds = discover_related_test_commands(
        repo,
        files_changed=["pkg/engine.py", "tests/test_engine.py"],
        existing_cmds=["uv run pytest -q tests/test_other.py"],
    )
    assert uv_cmds == ["uv run pytest -q tests/test_engine.py"]

    env_cmds = discover_related_test_commands(
        repo,
        files_changed=["pkg/engine.py"],
        existing_cmds=["PYTHONPATH=src python3 -m pytest -k test_other"],
    )
    assert env_cmds == ["PYTHONPATH=src python3 -m pytest -q tests/test_engine.py"]

    # 3. Dirty-worktree continuation: workspace isolation, no stale pre-apply resurrection,
    # batch receipt safety, and allowed_paths forwarding
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    shared_store = ReceiptStore(tmp_path / "shared_data", workspace_root=repo)

    repo_b = tmp_path / "repo_b"
    repo_b.mkdir()
    _init_temp_git_repo(repo_b)

    def cp_worker(req: dict[str, object]) -> dict[str, object]:
        wt = Path(str(req["workspace_dir"]))
        (wt / "app.py").write_text("def greet(): return 'cp'\n", encoding="utf-8")
        return {"text": "cp", "stop_reason": "MAX_TOOL_CALLS_EXCEEDED"}

    jm_a = JobManager(workspace_root=repo, store=shared_store, worker_backend=cp_worker)
    jm_b = JobManager(workspace_root=repo_b, store=shared_store, worker_backend=lambda _r: {"text": "ok_b"})

    ja1 = jm_a.delegate_start("Task A1", project="p", wait=True)
    assert ja1["status"] == "checkpoint"

    # Workspace B must NOT hijack Workspace A's checkpoint job
    jb1 = jm_b.delegate_start("Task B1", project="p", wait=True)
    assert jb1["job_id"] != ja1["job_id"]

    # Forwarding allowed_paths on auto-continuation in Workspace A
    jm_a.worker_backend = lambda _r: {"text": "ok_a2"}
    ja1_cont = jm_a.delegate_start("Task A1 continue", project="p", wait=True, allowed_paths=["app.py"])
    assert ja1_cont["job_id"] == ja1["job_id"]
    assert shared_store.read_receipt(ja1["job_id"])["allowed_paths"] == ["app.py"]
    jm_a.delegate_apply(ja1["job_id"])

    # If an older job is unapplied (e.g. j_stale) but a newer job j_applied was already applied,
    # a subsequent delegate_start must start a fresh job rather than resurrecting j_stale
    jm_a.worker_backend = cp_worker
    j_stale = jm_a.delegate_start("Stale checkpoint", project="p", wait=True)
    assert j_stale["status"] == "checkpoint"

    # Temporarily disable auto-continue to create & apply a newer job j_newer
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "0")
    def newer_worker(req: dict[str, object]) -> dict[str, object]:
        wt = Path(str(req["workspace_dir"]))
        (wt / "app.py").write_text("def greet(): return 'newer'\n", encoding="utf-8")
        return {"text": "newer"}

    jm_a.worker_backend = newer_worker
    j_newer = jm_a.delegate_start("Newer task", project="p", wait=True, auto_apply=True)
    assert j_newer["job_id"] != j_stale["job_id"]
    assert j_newer["applied"] is True

    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    j_third = jm_a.delegate_start("Third task", project="p", wait=True)
    assert j_third["job_id"] not in (j_stale["job_id"], j_newer["job_id"])

    # 4. Cancelled checkpoint job must NEVER be salvaged by salvage_unapplied_worktree
    repo_c = tmp_path / "repo_c"
    repo_c.mkdir()
    _init_temp_git_repo(repo_c)
    store_c = ReceiptStore(tmp_path / "data_c", workspace_root=repo_c)
    jm_c = JobManager(workspace_root=repo_c, store=store_c, worker_backend=cp_worker)
    j_canc = jm_c.delegate_start("Checkpoint then cancel", project="p", wait=True)
    assert j_canc["status"] == "checkpoint"
    jm_c.delegate_cancel(j_canc["job_id"])
    assert jm_c.salvage_unapplied_worktree() is None
    assert "cp" not in (repo_c / "app.py").read_text(encoding="utf-8")

    # 5. delegate_to_gemini must NOT auto-apply when acceptance_passed is False,
    # parse_qa_audit_verdict must parse numbered issues, and _maybe_run_qa_review must tolerate ask_gemini errors
    parsed_num = parse_qa_audit_verdict("VERDICT: NEEDS_FIXES\n1. Missing edge case check in app.py")
    assert parsed_num["passed"] is False
    assert len(parsed_num["issues"]) == 1

    srv_c = MCPServer(
        job_manager=jm_c,
        store=store_c,
        ask_backend=lambda _r: (_ for _ in ()).throw(RuntimeError("QA endpoint transient 503")),
    )
    jm_c.worker_backend = newer_worker
    d_fail_acc = srv_c._dispatch_tool(
        "delegate_to_gemini",
        {
            "prompt": "Edit app.py",
            "project": "p",
            "auto_apply": True,
            "qa_review_model": "gemini-4-argon",
            "acceptance": ["python3 -c 'import sys; sys.exit(1)'"],
        },
    )
    assert d_fail_acc["acceptance_passed"] is False
    assert d_fail_acc["applied"] is False
    assert "error" in d_fail_acc["qa_review"]

    jm_a.shutdown()
    jm_b.shutdown()
    jm_c.shutdown()


def test_whole_plugin_strict_mode_defaults(tmp_path: Path, monkeypatch) -> None:
    from scripts.pre_tool_guard import evaluate_pre_tool_use
    from server.jobs import JobManager
    from server.mcp_stdio import CORE_TOOL_NAMES, MCPServer, get_tool_definitions
    from server.receipts import ReceiptStore

    # 1. get_tool_definitions defaults to 'core' (8 tools) when strict mode is active
    strict_defs = get_tool_definitions(env={"CC_AGY_STRICT_MODE": "true"})
    assert set(t["name"] for t in strict_defs) == set(CORE_TOOL_NAMES)

    # 2. evaluate_pre_tool_use defaults Lead read cap to 2 in strict mode
    strict_env = {
        "CC_AGY_STRICT_MODE": "true",
        "CLAUDE_PROJECT_DIR": str(tmp_path),
        "CC_AGY_DATA_DIR": str(tmp_path / ".cc-agy-data"),
    }
    for _ in range(2):
        dec = evaluate_pre_tool_use(
            {"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "app.py")}, "cwd": str(tmp_path)},
            env=strict_env,
        )
        assert dec == {}
    dec_3 = evaluate_pre_tool_use(
        {"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "app.py")}, "cwd": str(tmp_path)},
        env=strict_env,
    )
    assert dec_3["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "LeadReadBudgetExceeded" in dec_3["hookSpecificOutput"]["permissionDecisionReason"]

    # 3. delegate_start defaults preload_files=True and auto_apply=True (when wait=True in strict mode)
    repo = tmp_path / "repo_defaults"
    repo.mkdir()
    _init_temp_git_repo(repo)
    store = ReceiptStore(tmp_path / "data_defaults", workspace_root=repo)

    def edit_worker(req: dict[str, object]) -> dict[str, object]:
        wt = Path(str(req["workspace_dir"]))
        (wt / "app.py").write_text("def greet() -> str:\n    return 'strict_auto_applied'\n", encoding="utf-8")
        return {"text": "done"}

    monkeypatch.setenv("CC_AGY_STRICT_MODE", "true")
    jm = JobManager(workspace_root=repo, store=store, worker_backend=edit_worker)
    srv = MCPServer(job_manager=jm, store=store)
    digest = srv._dispatch_tool(
        "delegate_start",
        {
            "prompt": "Update greet in app.py",
            "project": "test-proj",
            "wait": True,
        },
    )
    assert digest["status"] == "ok"
    assert digest.get("preloaded_files") == ["app.py"]
    assert digest["applied"] is True
    assert "strict_auto_applied" in (repo / "app.py").read_text(encoding="utf-8")
    jm.shutdown()


def test_extract_usage_metadata_handles_dict_model_and_attr_bag() -> None:
    from server.agy_runner import extract_usage_metadata

    got = extract_usage_metadata({"prompt_token_count": 120, "candidates_token_count": 30, "thoughts_token_count": 5})
    assert got["prompt_token_count"] == 120
    assert got["total_token_count"] == 155

    class Bag:
        prompt_token_count = 7
        cached_content_token_count = 0
        candidates_token_count = 1
        thoughts_token_count = 0
        total_token_count = 8

    assert extract_usage_metadata(Bag())["total_token_count"] == 8
    assert extract_usage_metadata(None)["total_token_count"] == 0
    assert extract_usage_metadata({"prompt_token_count": "bad"})["prompt_token_count"] == 0


def test_worker_sdk_limits_and_test_count_parsers() -> None:
    from server.agy_runner import resolve_worker_sdk_limits
    from server.jobs import _parse_test_counts

    # Default limits: 120000 compaction threshold, 4000 tool output tokens
    defaults = resolve_worker_sdk_limits({}, env={})
    assert defaults == (120_000, 4000)

    # Custom env overrides and 0-disable
    custom = resolve_worker_sdk_limits(
        {},
        env={"CC_AGY_TOOL_OUTPUT_MAX_TOKENS": "2500", "CC_AGY_COMPACTION_THRESHOLD": "0"},
    )
    assert custom == (0, 2500)

    # stestr / subunit parser (e.g. bandit)
    stestr_out = (
        "======\nTotals\n======\nRan: 163 tests in 1.4200 sec.\n"
        " - Passed: 161\n - Skipped: 0\n - Expected Fail: 0\n - Unexpected Success: 1\n - Failed: 1\n"
    )
    assert _parse_test_counts(stestr_out) == {"passed": 161, "failed": 1, "errors": 1}

    # unittest OK and FAILED parser (subtracts skipped from passed)
    ut_ok = "...............\n----------------------------------------------------------------------\nRan 15 tests in 0.042s\n\nOK (skipped=1)\n"
    assert _parse_test_counts(ut_ok) == {"passed": 14, "failed": 0, "errors": 0}
    ut_fail = "Ran 10 tests in 0.05s\n\nFAILED (failures=2, errors=1)\n"
    assert _parse_test_counts(ut_fail) == {"passed": 7, "failed": 2, "errors": 1}


def test_default_acceptance_import_smoke_and_model_escalation(tmp_path: Path, monkeypatch) -> None:
    from server.jobs import JobManager, build_import_smoke_command, discover_related_test_commands
    from server.receipts import ReceiptStore

    repo = tmp_path / "repo_acc"
    repo.mkdir()
    _init_temp_git_repo(repo)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "calc.py").write_text("def double(x: int) -> int:\n    return x * 2\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_calc.py").write_text(
        "from pkg.calc import double\ndef test_double():\n    assert double(3) == 6\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add pkg.calc"], cwd=str(repo), check=True)

    # 1. Default test discovery when caller passes existing_cmds=[]
    default_cmds = discover_related_test_commands(repo, ["pkg/calc.py"], [])
    assert len(default_cmds) == 1
    assert "pytest -q tests/test_calc.py" in default_cmds[0]

    # 2. Import smoke command generation for changed package module
    imp_cmds = build_import_smoke_command(repo, ["pkg/calc.py", "tests/test_calc.py"])
    assert len(imp_cmds) == 1
    assert "pkg.calc" in imp_cmds[0]
    assert "test_calc" not in imp_cmds[0]

    # 3. Auto default acceptance + Flash -> Argon model escalation on delegate_continue
    monkeypatch.setenv("CC_AGY_AUTO_DEFAULT_ACCEPTANCE", "1")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")
    monkeypatch.setenv("CC_AGY_SESSION_DEADLINE_SECONDS", "600")
    monkeypatch.setenv("CC_AGY_DEADLINE_RESERVE_SECONDS", "100")

    seen_models: list[str] = []
    seen_timeouts: list[float] = []

    def broken_then_fixed_worker(req: dict[str, object]) -> dict[str, object]:
        seen_models.append(str(req["model"]))
        seen_timeouts.append(float(req["timeout_seconds"]))
        wt = Path(str(req["workspace_dir"]))
        if len(seen_models) == 1:
            # Break pkg/calc.py so auto-discovered default acceptance fails
            (wt / "pkg" / "calc.py").write_text("def double(x: int) -> int:\n    return x * 999\n", encoding="utf-8")
        else:
            (wt / "pkg" / "calc.py").write_text("def double(x: int) -> int:\n    return int(x) * 2\n", encoding="utf-8")
        return {
            "text": "updated calc",
            "usage": {
                "prompt_token_count": 250_000,
                "candidates_token_count": 100,
                "total_token_count": 250_100,
                "requests": [
                    {"prompt_token_count": 180_000},
                    {"prompt_token_count": 220_000},
                ],
            },
        }

    store = ReceiptStore(tmp_path / "data_acc", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=broken_then_fixed_worker)

    d1 = jm.delegate_start(
        "Update double in pkg/calc.py",
        model="gemini-3.8-flash",
        project="test-proj",
        timeout_seconds=1200.0,
        wait=True,
    )
    # Timeout was clamped below (600 - 100) = 500s
    assert seen_timeouts[0] <= 500.0
    assert d1["acceptance_passed"] is False
    assert any(r.get("auto_discovered_default") for r in d1.get("acceptance_results", []))
    assert d1["usage"]["over_200k_requests_count"] == 1

    d2 = jm.delegate_continue(
        d1["job_id"],
        "Fix double in pkg/calc.py so tests pass",
        project="test-proj",
        timeout_seconds=1200.0,
        wait=True,
    )
    assert seen_models == ["gemini-3.8-flash", "gemini-4-argon"]
    assert seen_timeouts[1] <= 500.0
    assert d2["model"] == "gemini-4-argon"
    assert d2.get("escalated_from_model") == "gemini-3.8-flash"
    assert d2["acceptance_passed"] is True
    jm.shutdown()


def test_guard_pytest_ini_plugin_fence_dictation_repair_and_diff_summary(tmp_path: Path, monkeypatch) -> None:
    from scripts.pre_tool_guard import (
        PLUGIN_ROOT,
        detect_bash_workspace_mutation,
        evaluate_pre_tool_use,
        lint_delegation_prompt,
        repair_dictated_prompt,
    )
    from scripts.session_start import verify_mcp_server_health
    from server.jobs import JobManager
    from server.mcp_stdio import MCPServer
    from server.receipts import ReceiptStore

    # 1. SessionStart MCP health self-check
    health = verify_mcp_server_health()
    assert health["healthy"] is True
    assert health["tool_count"] >= 8

    # 2. pytest -o addopts=... is allowed while unsafe -o cache_dir=... or -p plugin is blocked
    v_safe = detect_bash_workspace_mutation('pytest -q -o addopts="" tests/', tmp_path)
    assert v_safe["blocked"] is False
    v_safe2 = detect_bash_workspace_mutation("pytest --override-ini=xfail_strict=true tests/", tmp_path)
    assert v_safe2["blocked"] is False
    v_unsafe = detect_bash_workspace_mutation("pytest -o cache_dir=/tmp/outside tests/", tmp_path)
    assert v_unsafe["blocked"] is True
    v_unsafe_plug = detect_bash_workspace_mutation('pytest -o "addopts=-p evil_plugin" tests/', tmp_path)
    assert v_unsafe_plug["blocked"] is True

    # 3. Direct Python import of plugin internals in Bash or reading PLUGIN_ROOT is blocked in strict mode
    v_imp = detect_bash_workspace_mutation(
        'python3 -c "from server.mcp_stdio import MCPServer; print(MCPServer)"',
        tmp_path,
    )
    assert v_imp["blocked"] is True
    assert "plugin internals" in str(v_imp["reason"])

    strict_env = {
        "CC_AGY_STRICT_MODE": "true",
        "CLAUDE_PROJECT_DIR": str(tmp_path),
        "CC_AGY_DATA_DIR": str(tmp_path / ".cc-agy-data"),
    }
    dec_plugin_read = evaluate_pre_tool_use(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": str(PLUGIN_ROOT / "server" / "mcp_stdio.py")},
            "cwd": str(tmp_path),
        },
        env=strict_env,
    )
    assert dec_plugin_read["hookSpecificOutput"]["permissionDecision"] == "deny"

    # 4. LeadReadBudgetExceeded auto-attaches Focused File Outline for the requested workspace file
    (tmp_path / "service.py").write_text(
        "class OrderProcessor:\n    def process_order(self, order_id: str) -> bool:\n        return True\n",
        encoding="utf-8",
    )
    strict_env_cap1 = {**strict_env, "CC_AGY_LEAD_READ_LIMIT": "1"}
    assert (
        evaluate_pre_tool_use(
            {"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "service.py")}, "cwd": str(tmp_path)},
            env=strict_env_cap1,
        )
        == {}
    )
    dec_exceed = evaluate_pre_tool_use(
        {"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "service.py")}, "cwd": str(tmp_path)},
        env=strict_env_cap1,
    )
    reason_exceed = dec_exceed["hookSpecificOutput"]["permissionDecisionReason"]
    assert "LeadReadBudgetExceeded" in reason_exceed
    assert "Focused File Outline" in reason_exceed
    assert "OrderProcessor" in reason_exceed

    # 5. Dictated code block auto-repair (repair_dictated_prompt + CC_AGY_REPAIR_CODE_DICTATION=1)
    dictated = (
        "Please update app.py with this implementation:\n"
        "```python\n"
        "def a1(x):\n"
        "    y = x + 1\n"
        "    y = y + 2\n"
        "    y = y + 3\n"
        "    y = y + 4\n"
        "    y = y + 5\n"
        "    y = y + 6\n"
        "    y = y + 7\n"
        "    y = y + 8\n"
        "    return y\n"
        "```\n"
    )
    assert lint_delegation_prompt(dictated)["ok"] is False
    rep = repair_dictated_prompt(dictated, snippet_dir=tmp_path / "snippets")
    assert rep["repaired"] is True
    assert lint_delegation_prompt(rep["prompt"])["ok"] is True
    assert len(rep["snippet_paths"]) == 1
    assert Path(rep["snippet_paths"][0]).is_file()

    # With CC_AGY_REPAIR_CODE_DICTATION=1, PreToolUse allows the call through and MCPServer repairs it
    env_repair = {**strict_env, "CC_AGY_REPAIR_CODE_DICTATION": "1"}
    assert (
        evaluate_pre_tool_use(
            {"tool_name": "mcp__plugin_agy_agy__delegate_start", "tool_input": {"prompt": dictated}, "cwd": str(tmp_path)},
            env=env_repair,
        )
        == {}
    )

    # 6. get_diff with file_stats, summary_only=True, and focus_symbols
    repo = tmp_path / "repo_diff"
    repo.mkdir()
    _init_temp_git_repo(repo)
    (repo / "extra.py").write_text("def helper_one() -> int:\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "add extra.py"], cwd=str(repo), check=True)

    def multi_file_worker(req: dict[str, object]) -> dict[str, object]:
        wt = Path(str(req["workspace_dir"]))
        (wt / "app.py").write_text("def greet() -> str:\n    return 'updated_greet'\n", encoding="utf-8")
        (wt / "extra.py").write_text("def helper_one() -> int:\n    return 42\n", encoding="utf-8")
        return {"text": "edited both"}

    monkeypatch.setenv("CC_AGY_REPAIR_CODE_DICTATION", "1")
    store = ReceiptStore(tmp_path / "data_diff", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=multi_file_worker)
    srv = MCPServer(job_manager=jm, store=store)

    d_rep = srv._dispatch_tool(
        "delegate_start",
        {"prompt": dictated, "project": "p", "wait": True, "auto_apply": False},
    )
    assert d_rep.get("dictation_repaired") is True

    diff_summary = srv._dispatch_tool(
        "get_diff",
        {"job_id": d_rep["job_id"], "summary_only": True},
    )
    assert diff_summary["summary_only"] is True
    assert diff_summary["diff"] == ""
    assert set(diff_summary["file_stats"].keys()) == {"app.py", "extra.py"}
    assert diff_summary["file_stats"]["app.py"]["added"] >= 1

    diff_focused = srv._dispatch_tool(
        "get_diff",
        {"job_id": d_rep["job_id"], "focus_symbols": ["helper_one"]},
    )
    assert "helper_one" in diff_focused["diff"]
    assert "updated_greet" not in diff_focused["diff"]
    assert list(diff_focused["file_stats"].keys()) == ["extra.py"]
    assert diff_focused["files_changed"] == ["extra.py"]
    jm.shutdown()


def test_gauntlet_round1_regressions(tmp_path: Path, monkeypatch) -> None:
    """Regression suite for all Round 1 Gauntlet findings across Opus 5.5, Fable 5.1, and Argon."""
    from scripts.pre_tool_guard import (
        PLUGIN_ROOT,
        detect_bash_workspace_mutation,
        evaluate_pre_tool_use,
        lint_delegation_prompt,
        repair_dictated_prompt,
    )
    from server.agy_runner import resolve_worker_sdk_limits
    from server.jobs import JobManager, _parse_test_counts, build_import_smoke_command
    from server.mcp_stdio import MCPServer
    from server.receipts import ReceiptStore

    # 1. Pytest addopts bypasses, clustered short flags (-qo, -qp, -qc), testpaths, and python -m pytest
    for blocked_cmd in (
        'pytest -o "addopts=-c /tmp/evil.ini" tests/',
        'pytest -o "addopts=--noconftest" tests/',
        'pytest -o "addopts=--cov=pkg --cov-report=html:/tmp/cov" tests/',
        'pytest -o "addopts=--log-file=/tmp/pwn.log" tests/',
        'pytest -o "addopts=/tmp/evil_test.py" tests/',
        "pytest -o testpaths=/tmp tests/",
        "pytest -c /tmp/evil.ini tests/",
        "pytest -qc /tmp/evil.ini tests/",
        "pytest -qp evil_plugin tests/",
        "pytest -qo cache_dir=/tmp/outside tests/",
        "python3 -m pytest -p evil_plugin tests/",
        "python3 -m pytest --confcutdir=/tmp tests/",
        "python3 -m pytest -o cache_dir=/tmp/outside tests/",
        'PYTEST_ADDOPTS="-o cache_dir=/tmp/outside" pytest tests/',
        'env PYTEST_ADDOPTS="-o cache_dir=/tmp/outside" pytest tests/',
        'export PYTEST_ADDOPTS="-o cache_dir=/tmp/outside" && pytest tests/',
        "PYTEST_PLUGINS=evil_plugin pytest tests/",
        "pytest-3 -o cache_dir=/tmp/outside tests/",
    ):
        res_b = detect_bash_workspace_mutation(blocked_cmd, tmp_path)
        assert res_b["blocked"] is True, f"Expected blocked for: {blocked_cmd}"

    for allowed_cmd in (
        "pytest -q -p no:cacheprovider tests/",
        "pytest -qp no:cacheprovider tests/",
        'pytest -qo "addopts=-q -p no:cacheprovider" tests/',
        'python3 -m pytest -q -o "addopts=-q" tests/',
        'SAFE_VAR="a b" pytest -q tests/',
    ):
        res_a = detect_bash_workspace_mutation(allowed_cmd, tmp_path)
        assert res_a["blocked"] is False, f"Expected allowed for: {allowed_cmd} ({res_a.get('reason')})"

    # 2. Scoped MCPServer / JobManager / PLUGIN_ROOT fence: grep in user repo allowed, python import or cat PLUGIN_ROOT blocked
    assert detect_bash_workspace_mutation("grep -rn 'JobManager(' server/", tmp_path)["blocked"] is False
    assert (
        detect_bash_workspace_mutation(
            'python3 -c "from server import jobs; print(jobs.JobManager)"',
            tmp_path,
        )["blocked"]
        is True
    )
    assert (
        detect_bash_workspace_mutation(
            f"cat {PLUGIN_ROOT / 'server' / 'mcp_stdio.py'}",
            tmp_path,
        )["blocked"]
        is True
    )

    strict_env = {
        "CC_AGY_STRICT_MODE": "true",
        "CLAUDE_PROJECT_DIR": str(tmp_path),
        "CC_AGY_DATA_DIR": str(tmp_path / ".cc-agy-data"),
    }
    # Grep with regex pattern is allowed even when pattern has slashes
    assert (
        evaluate_pre_tool_use(
            {"tool_name": "Grep", "tool_input": {"pattern": "server/jobs"}, "cwd": str(tmp_path)},
            env=strict_env,
        )
        == {}
    )
    # Grep with cwd inside PLUGIN_ROOT and no path is denied when workspace is tmp_path
    dec_grep_plugin = evaluate_pre_tool_use(
        {"tool_name": "Grep", "tool_input": {"pattern": "MCPServer"}, "cwd": str(PLUGIN_ROOT)},
        env=strict_env,
    )
    assert dec_grep_plugin["hookSpecificOutput"]["permissionDecision"] == "deny"

    # 3. Dictation repair: mismatched inner fence (~~~ inside ```), unfenced across short fence, no host path leak
    mismatched_fence_prompt = (
        "Update parser:\n"
        "```python\n"
        "x1 = 1\n"
        "x2 = 2\n"
        "x3 = 3\n"
        "x4 = 4\n"
        "~~~\n"
        "x5 = 5\n"
        "x6 = 6\n"
        "x7 = 7\n"
        "x8 = 8\n"
        "x9 = 9\n"
        "```\n"
    )
    assert lint_delegation_prompt(mismatched_fence_prompt)["ok"] is False
    rep_m = repair_dictated_prompt(mismatched_fence_prompt, snippet_dir=tmp_path / "snip_m")
    assert rep_m["repaired"] is True
    assert lint_delegation_prompt(rep_m["prompt"])["ok"] is True
    assert str(tmp_path / "snip_m") not in rep_m["prompt"]

    split_unfenced_prompt = (
        "x1 = 1\n"
        "x2 = 2\n"
        "x3 = 3\n"
        "x4 = 4\n"
        "x5 = 5\n"
        "```\n"
        "note\n"
        "```\n"
        "x6 = 6\n"
        "x7 = 7\n"
        "x8 = 8\n"
        "x9 = 9\n"
    )
    assert lint_delegation_prompt(split_unfenced_prompt)["ok"] is False
    rep_s = repair_dictated_prompt(split_unfenced_prompt, snippet_dir=tmp_path / "snip_s")
    assert rep_s["repaired"] is True
    assert lint_delegation_prompt(rep_s["prompt"])["ok"] is True

    # Cross-boundary prompt + plan dictation repair in MCPServer._maybe_repair_dictation
    monkeypatch.setenv("CC_AGY_REPAIR_CODE_DICTATION", "1")
    repo = tmp_path / "repo_r1"
    repo.mkdir()
    _init_temp_git_repo(repo)
    (repo / "extra.py").write_text("def check() -> bool:\n    valid = True\n    return valid\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init r1"], cwd=str(repo), check=True)
    store = ReceiptStore(tmp_path / "data_r1", workspace_root=repo)
    jm = JobManager(
        workspace_root=repo,
        store=store,
        worker_backend=lambda req: (
            (Path(str(req["workspace_dir"])) / "app.py").write_text("def by_id(id):\n    return id\n", encoding="utf-8"),
            (Path(str(req["workspace_dir"])) / "extra.py").write_text("def check() -> bool:\n    valid = False\n    return valid\n", encoding="utf-8"),
            {"text": "ok"},
        )[-1],
    )
    srv = MCPServer(job_manager=jm, store=store)
    p_tail = "a1 = 1\na2 = 2\na3 = 3\na4 = 4\na5 = 5"
    plan_head = ["b1 = 1\nb2 = 2\nb3 = 3\nb4 = 4"]
    assert lint_delegation_prompt("\n".join([p_tail, *plan_head]))["ok"] is False
    new_p, new_pl, rmeta = srv._maybe_repair_dictation(p_tail, plan_head)
    assert rmeta["repaired"] is True
    assert lint_delegation_prompt("\n".join([new_p, *new_pl]))["ok"] is True

    # 4. get_diff focus_symbols word-boundary matching ("id" matches `by_id(id)` in app.py, not `valid` in extra.py)
    d_job = jm.delegate_start("Update app and extra", project="p", wait=True)
    diff_id = jm.get_diff(d_job["job_id"], focus_symbols=["id"])
    assert diff_id["files_changed"] == ["app.py"]
    assert list(diff_id["file_stats"].keys()) == ["app.py"]
    assert "by_id" in diff_id["diff"]
    assert "valid" not in diff_id["diff"]

    # 5. _parse_test_counts with prose "2 errors" inside unittest output + trailing teardown + multi-suite
    ut_with_prose_and_teardown = (
        "test_validation (tests.TestParser) ... expected 2 errors in malformed input\n"
        "ok\n"
        "----------------------------------------------------------------------\n"
        "Ran 12 tests in 0.031s\n\n"
        "OK (skipped=2)\n"
        "Destroying test database for alias 'default'...\n"
    )
    assert _parse_test_counts(ut_with_prose_and_teardown) == {"passed": 10, "failed": 0, "errors": 0}
    ut_multi = (
        "Ran 5 tests in 0.01s\n\nOK\n"
        "Ran 4 tests in 0.02s\n\nFAILED (failures=1, errors=1, skipped=1)\n"
    )
    assert _parse_test_counts(ut_multi) == {"passed": 6, "failed": 1, "errors": 1}

    # 6. build_import_smoke_command skips top-level scripts like run.py
    (repo / "run.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    smoke_cmds = build_import_smoke_command(repo, ["run.py", "pkg/mod.py"])
    assert len(smoke_cmds) == 1
    assert "pkg.mod" in smoke_cmds[0]
    assert "run" not in smoke_cmds[0]

    # 7. _clamp_worker_timeout floor when elapsed >= deadline, and CC_AGY_SESSION_DEADLINE_SECONDS=0
    monkeypatch.setenv("CC_AGY_SESSION_DEADLINE_SECONDS", "100")
    monkeypatch.setenv("CC_AGY_DEADLINE_RESERVE_SECONDS", "30")
    jm._created_at = jm._created_at - 500.0
    assert jm._clamp_worker_timeout(600.0) == 45.0
    monkeypatch.setenv("CC_AGY_SESSION_DEADLINE_SECONDS", "0")
    assert jm._clamp_worker_timeout(600.0) == 600.0
    assert jm._is_bounded_lead_session() is False

    # 8. resolve_worker_sdk_limits float coercion & _compact_digest malformed prompt_token_count resilience
    limits = resolve_worker_sdk_limits({"compaction_threshold": 150000.0, "tool_output_max_tokens": 3500.0})
    assert limits == (150000, 3500)
    rec = store.read_receipt(d_job["job_id"])
    rec["usage"] = {"requests": [{"prompt_token_count": "n/a"}, {"prompt_token_count": 250000}]}
    store.write_receipt(d_job["job_id"], rec, workspace_root=str(repo))
    st_digest = jm.delegate_status(d_job["job_id"])
    assert st_digest["usage"]["over_200k_requests_count"] == 1

    # 9. Round 2: new broken module (`SyntaxError` in `pkg/new_broken.py`) fails import smoke (acceptance_passed=False)
    # and never touches untracked files in workspace_root
    monkeypatch.setenv("CC_AGY_AUTO_DEFAULT_ACCEPTANCE", "1")
    (repo / "MY_NOTES.md").write_text("keep me\n", encoding="utf-8")
    jm_broken = JobManager(
        workspace_root=repo,
        store=store,
        worker_backend=lambda req: (
            (Path(str(req["workspace_dir"])) / "pkg").mkdir(exist_ok=True),
            (Path(str(req["workspace_dir"])) / "pkg" / "__init__.py").write_text("", encoding="utf-8"),
            (Path(str(req["workspace_dir"])) / "pkg" / "new_broken.py").write_text("def broken(:\n", encoding="utf-8"),
            {"text": "created broken module"},
        )[-1],
    )
    d_broken = jm_broken.delegate_start("Create pkg/new_broken.py", project="p", wait=True)
    assert d_broken["acceptance_passed"] is False
    assert (repo / "MY_NOTES.md").is_file()
    jm_broken.shutdown()
    jm.shutdown()
