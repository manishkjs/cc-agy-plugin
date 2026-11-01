"""Regression tests for Round 2 external review findings (Issues 1-5, E1-E3, and PATH portability)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from scripts.post_tool_ledger import evaluate_post_tool_use
import server.batch_runner as batch_runner
from server.batch_plan import validate_batch_plan
from server.batch_runner import FileBatchRunner, _workspace_state, probe_writers
from server.doctor import run_doctor
from server.genai_ask import ask_gemini
from server.jobs import JobManager
from server.mcp_stdio import MCPServer, _select_tool_lane, serve_stdio
from server.model_routes import resolve_batch_settings
from server.receipts import BudgetExceededError, ReceiptStore
from server.verification_policy import (
    classify_commands_for_batch,
    classify_verification,
    command_strength,
    has_real_acceptance_pass,
    is_verified_applied_receipt,
    satisfies_auto_apply_verification,
    satisfies_post_tool_auto_stop,
    satisfies_timeout_promotion,
)
from server.writers import WriterAuthError, WriterReply, WriterRequest


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo), check=True)


def _git_commit_all(repo: Path, msg: str = "init") -> None:
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", msg], cwd=str(repo), check=True)


# ---------------------------------------------------------------------------
# Issue 1 + E3: Verification strength classification & unified policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("true", "noop"),
        ("/bin/true", "noop"),
        ("'true'", "noop"),
        ("\\true", "noop"),
        ("sh -c true", "noop"),
        ("/usr/bin/env true", "noop"),
        ("! false", "noop"),
        ("test 1", "noop"),
        ("python3 -c 'assert True'", "noop"),
        (":", "noop"),
        ("echo '5 passed'", "noop"),
        ("pytest -q || true", "noop"),
        ("pytest -q || /bin/true", "noop"),
        ("pytest -q || ( true )", "noop"),
        ("if pytest; then :; fi", "noop"),
        ("pytest -q || exit 0", "noop"),
        ("pytest -q; true", "noop"),
        ("pytest --collect-only", "noop"),
        ("pytest --version", "noop"),
        ("go test -run=^$ ./...", "noop"),
        ("go test -run '^$' ./...", "noop"),
        ("pytest -q | tee out.log", "noop"),
        ("set -o pipefail && pytest -q | tee out.log", "behavioral"),
        ("python3 -m py_compile app.py", "syntax_only"),
        ("python3 -m mypy pkg", "syntax_only"),
        ("python3 -m flake8 pkg", "syntax_only"),
        ("python3 -m ruff check .", "syntax_only"),
        ("python3 -m black --check .", "syntax_only"),
        ("npm run lint", "syntax_only"),
        ("npm run build", "syntax_only"),
        ("npm run typecheck", "syntax_only"),
        ("make lint", "syntax_only"),
        ("cd pkg && python3 -m py_compile a.py", "syntax_only"),
        ("node --check app.js", "syntax_only"),
        ("bash -n run.sh", "syntax_only"),
        ("tsc --noEmit", "syntax_only"),
        ("ruff check src", "syntax_only"),
        ("mypy src", "syntax_only"),
        ("go build ./...", "syntax_only"),
        ("cargo check", "syntax_only"),
        ("test -f app.py", "syntax_only"),
        ("python3 -c \"import ast; ast.parse(open('app.py').read())\"", "syntax_only"),
        ("python3 -c \"import app; assert app.greet() == 'ok'\"", "custom"),
        ("make test", "custom"),
        ("./scripts/run_tests.sh", "custom"),
        ("python3 check_suite.py", "custom"),
        ("pytest -q", "behavioral"),
        (f"{shlex.quote(sys.executable)} -m pytest -q", "behavioral"),
        ("npm test", "behavioral"),
    ],
)
def test_command_strength_matrix(cmd: str, expected: str) -> None:
    assert command_strength(cmd) == expected


@pytest.mark.parametrize("trivial_cmd", ["true", "python3 -m py_compile app.py"])
def test_compile_only_and_true_keep_diff_preview_under_auto_apply(
    tmp_path: Path, trivial_cmd: str
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "app.py").write_text("def greet() -> str:\n    return 'old'\n", encoding="utf-8")
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)

    def ok_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "app.py").write_text("def greet() -> str:\n    return 'new'\n", encoding="utf-8")
        return {"text": "Updated greet() in app.py."}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=ok_worker)
    try:
        res = jm.delegate_start(
            "Update greet() in app.py",
            project="test-proj",
            wait=True,
            auto_apply=True,
            acceptance=[trivial_cmd],
        )
        assert res["status"] == "ok"
        assert res["applied"] is True
        assert "real_acceptance_passed" not in res
        assert "diff_preview" in res and "+    return 'new'" in res["diff_preview"]

        rec = store.read_receipt(res["job_id"])
        acc_results = rec["acceptance_results"]
        assert has_real_acceptance_pass(acc_results) is False
        assert satisfies_auto_apply_verification(acc_results, True) is False
        assert satisfies_timeout_promotion(acc_results) is False
        assert is_verified_applied_receipt(rec) is False
        assert satisfies_post_tool_auto_stop(rec) is False

        expected_tier = "unverified" if trivial_cmd == "true" else "smoke_only"
        assert classify_verification(acc_results, True) == expected_tier

        # PostToolUse hook must NOT emit an auto-stop block decision on compile-only / noop checks.
        hook_out = evaluate_post_tool_use(
            {
                "cwd": str(repo),
                "tool_name": "mcp__plugin_cc-agy_cc-agy__delegate_start",
                "tool_response": res,
            },
            env={"CC_AGY_STRICT_MODE": "true"},
            store=store,
        )
        assert hook_out.get("decision") != "block"
    finally:
        jm.shutdown()


def test_behavioral_custom_python_assert_sets_real_acceptance_passed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "app.py").write_text("def greet() -> str:\n    return 'old'\n", encoding="utf-8")
    (repo / "run_tests.py").write_text("import app\nassert app.greet() == 'ok'\n", encoding="utf-8")
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)

    def ok_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "app.py").write_text("def greet() -> str:\n    return 'ok'\n", encoding="utf-8")
        return {"text": "Updated greet()."}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=ok_worker)
    try:
        behav_cmd = f"{shlex.quote(sys.executable)} run_tests.py"
        assert command_strength(behav_cmd) == "custom"
        res = jm.delegate_start(
            "Make greet() return 'ok' in app.py",
            project="test-proj",
            wait=True,
            auto_apply=True,
            acceptance=[behav_cmd],
        )
        assert res["status"] == "ok"
        assert res["applied"] is True
        assert res.get("real_acceptance_passed") is True
        rec = store.read_receipt(res["job_id"])
        assert is_verified_applied_receipt(rec) is True
    finally:
        jm.shutdown()


def test_batch_and_delegation_unify_verification_tiers(tmp_path: Path) -> None:
    assert classify_commands_for_batch([]) == "unverified"
    assert classify_commands_for_batch(["true", ":"]) == "unverified"
    assert classify_commands_for_batch(["python3 -m py_compile a.py"]) == "syntax_only"
    assert classify_commands_for_batch(["./run_tests.sh"]) == "behavioral"

    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda r: {"text": "ok"})
    try:
        def fake_factory(name: str, settings: Any, auth: dict[str, Any]):
            if name != settings.writer:
                return None
            return lambda req: WriterReply(
                text=json.dumps({"path": req.task.path, "content": "X = 1\n"}),
                model="gemini-3.8-flash",
                usage={"prompt_token_count": 50, "candidates_token_count": 10},
                cost_cents=0.01,
            )

        plan_noop = validate_batch_plan(
            {
                "tasks": [
                    {"task_id": "t1", "path": "a.py", "action": "create", "intent": "set X to 1", "checks": ["true"]},
                ]
            },
            repo,
        )
        settings = resolve_batch_settings({})
        out_noop = FileBatchRunner(jm, writer_factory=fake_factory).run(
            plan_noop, settings, {"auth_mode": "api_key", "api_key": "fake"}
        )
        assert out_noop["verification"] == "unverified"
        assert out_noop["unverified_task_ids"] == ["t1"]

        plan_syn = validate_batch_plan(
            {
                "tasks": [
                    {
                        "task_id": "t2",
                        "path": "b.py",
                        "action": "create",
                        "intent": "set X to 1",
                        "checks": [f"{shlex.quote(sys.executable)} -m py_compile b.py"],
                    },
                ]
            },
            repo,
        )
        out_syn = FileBatchRunner(jm, writer_factory=fake_factory).run(
            plan_syn, settings, {"auth_mode": "api_key", "api_key": "fake"}
        )
        assert out_syn["verification"] == "syntax_only"
        assert "unverified_task_ids" not in out_syn
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# Issue 2: Shared daily spend cap checked & reserved before every paid call
# ---------------------------------------------------------------------------


def test_daily_cap_stops_batch_after_first_paid_call_and_blocks_next_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CC_AGY_MAX_DAILY_CENTS", "1")
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda r: {"text": "ok"})
    paid_calls = {"n": 0}
    call_lock = threading.Lock()

    def two_cent_writer_factory(name: str, settings: Any, auth: dict[str, Any]):
        if name != settings.writer:
            return None

        def _writer(req: WriterRequest) -> WriterReply:
            with call_lock:
                paid_calls["n"] += 1
            return WriterReply(
                text=json.dumps({"path": req.task.path, "content": f"# {req.task.task_id}\n"}),
                model="gemini-3.8-flash",
                usage={"prompt_token_count": 1000, "candidates_token_count": 200},
                cost_cents=2.0,
            )

        return _writer

    try:
        plan = validate_batch_plan(
            {
                "tasks": [
                    {"task_id": "t1", "path": "f1.py", "action": "create", "intent": "write f1"},
                    {"task_id": "t2", "path": "f2.py", "action": "create", "intent": "write f2"},
                    {"task_id": "t3", "path": "f3.py", "action": "create", "intent": "write f3"},
                ]
            },
            repo,
        )
        settings = resolve_batch_settings({"max_parallel": 4})
        digest = FileBatchRunner(jm, writer_factory=two_cent_writer_factory).run(
            plan, settings, {"auth_mode": "api_key", "api_key": "fake"}
        )

        # Must stop after 1 call (2c >= 1c daily cap) rather than running all 3 tasks (6c).
        assert paid_calls["n"] == 1
        assert digest["status"] == "checkpoint"
        assert digest["totals"]["landed"] == 1
        assert digest["totals"]["held"] == 2
        assert digest["totals"]["cost_cents"] == pytest.approx(2.0)

        today = store.daily_spend_summary(workspace_root=repo)
        assert today["spent_today_cents"] == pytest.approx(2.0)

        # Subsequent ask_gemini and probe_writers calls must also fail immediately with BudgetExceededError.
        with pytest.raises(BudgetExceededError):
            ask_gemini(
                "hello",
                api_key="fake",
                workspace_root=repo,
                store=store,
                backend=lambda req: {"text": "never", "usage": {}},
            )
        with pytest.raises(BudgetExceededError):
            probe_writers(
                settings,
                {"auth_mode": "api_key", "api_key": "fake"},
                writer_factory=two_cent_writer_factory,
                store=store,
                workspace_root=repo,
            )
    finally:
        jm.shutdown()


def test_daily_cap_across_concurrent_threads_and_processes_and_dead_pid_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CC_AGY_MAX_DAILY_CENTS", "1.5")
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git_commit_all(repo)

    data_dir = tmp_path / "shared_data"
    store = ReceiptStore(data_dir, workspace_root=repo)

    # 1. Inject a stale unpriced reservation owned by a non-existent PID; reserve() must prune it immediately.
    day = time.strftime("%Y-%m-%d", time.gmtime())
    budget_dir = data_dir / "budget"
    budget_dir.mkdir(parents=True, exist_ok=True)
    ws_key = str(repo.resolve())
    (budget_dir / f"{day}.json").write_text(
        json.dumps(
            {
                "utc_date": day,
                "workspaces": {
                    ws_key: {
                        "live": {},
                        "reservations": {
                            "res-dead": {
                                "job_id": "dead-job",
                                "cents": 0.0,
                                "unpriced": True,
                                "pid": 999_999_991,
                                "proc_start": 1,
                                "ts": time.time(),
                                "utc_day": day,
                                "ws_key": ws_key,
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    # 2. Launch 2 independent OS processes running distinct batch plans against the same shared data_dir.
    worker_script = tmp_path / "proc_batch.py"
    repo_root = Path(__file__).resolve().parent.parent
    worker_script.write_text(
        "import json, os, sys, time\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from pathlib import Path\n"
        "from server.batch_plan import validate_batch_plan\n"
        "from server.batch_runner import FileBatchRunner\n"
        "from server.jobs import JobManager\n"
        "from server.model_routes import resolve_batch_settings\n"
        "from server.receipts import BudgetExceededError, ReceiptStore\n"
        "from server.writers import WriterReply\n"
        "idx = sys.argv[1]\n"
        f"repo = Path({str(repo)!r})\n"
        f"store = ReceiptStore(Path({str(data_dir)!r}), workspace_root=repo)\n"
        "jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda r: {'text': 'ok'})\n"
        "calls = 0\n"
        "def wf(name, settings, auth):\n"
        "    if name != settings.writer: return None\n"
        "    def w(req):\n"
        "        global calls\n"
        "        calls += 1\n"
        "        time.sleep(0.15)\n"
        "        return WriterReply(text=json.dumps({'path': req.task.path, 'content': 'x = 1\\n'}),\n"
        "                           model='gemini-3.8-flash', usage={'prompt_token_count': 100}, cost_cents=2.0)\n"
        "    return w\n"
        "plan = validate_batch_plan({'tasks': [\n"
        "    {'task_id': f'a{idx}', 'path': f'p{idx}_1.py', 'action': 'create', 'intent': 'create module a'},\n"
        "    {'task_id': f'b{idx}', 'path': f'p{idx}_2.py', 'action': 'create', 'intent': 'create module b'},\n"
        "]}, repo)\n"
        "try:\n"
        "    FileBatchRunner(jm, writer_factory=wf).run(plan, resolve_batch_settings({}), {'auth_mode': 'api_key', 'api_key': 'k'})\n"
        "except BudgetExceededError:\n"
        "    pass\n"
        "finally:\n"
        "    jm.shutdown()\n"
        "print(json.dumps({'calls': calls}))\n",
        encoding="utf-8",
    )

    env = {**os.environ, "CC_AGY_MAX_DAILY_CENTS": "1.5"}
    p1 = subprocess.Popen([sys.executable, str(worker_script), "1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    p2 = subprocess.Popen([sys.executable, str(worker_script), "2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out1, err1 = p1.communicate(timeout=30)
    out2, err2 = p2.communicate(timeout=30)
    assert p1.returncode == 0, err1
    assert p2.returncode == 0, err2
    total_calls = json.loads(out1.strip())["calls"] + json.loads(out2.strip())["calls"]
    assert total_calls == 1
    assert store.daily_spend_summary(workspace_root=repo)["spent_today_cents"] == pytest.approx(2.0)


def test_resumed_batch_daily_spend_never_double_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CC_AGY_MAX_DAILY_CENTS", "1.5")
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda r: {"text": "ok"})

    def writer_factory(name: str, settings: Any, auth: dict[str, Any]):
        if name != settings.writer:
            return None
        return lambda req: WriterReply(
            text=json.dumps({"path": req.task.path, "content": f"VAL = {req.task.task_id!r}\n"}),
            model="gemini-3.8-flash",
            usage={"prompt_token_count": 100},
            cost_cents=2.0,
        )

    try:
        plan = validate_batch_plan(
            {
                "tasks": [
                    {"task_id": "t1", "path": "m1.py", "action": "create", "intent": "create m1"},
                    {"task_id": "t2", "path": "m2.py", "action": "create", "intent": "create m2"},
                ]
            },
            repo,
        )
        settings = resolve_batch_settings({})
        d1 = FileBatchRunner(jm, writer_factory=writer_factory).run(
            plan, settings, {"auth_mode": "api_key", "api_key": "k"}
        )
        assert d1["status"] == "checkpoint"
        assert store.daily_spend_summary(workspace_root=repo)["spent_today_cents"] == pytest.approx(2.0)

        # Raise the daily cap and resume: only t2 runs (+2.0c), total today must be 4.0c (not 6.0c).
        monkeypatch.setenv("CC_AGY_MAX_DAILY_CENTS", "10")
        d2 = FileBatchRunner(jm, writer_factory=writer_factory).run(
            plan, settings, {"auth_mode": "api_key", "api_key": "k"}
        )
        assert d2["status"] == "ok"
        assert d2["job_id"] == d1["job_id"]
        assert d2["totals"]["unchanged"] == 1
        assert d2["totals"]["landed"] == 1
        assert store.daily_spend_summary(workspace_root=repo)["spent_today_cents"] == pytest.approx(4.0)
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# Issue 3: Executable file permissions preserved on failed checks, retries & revert
# ---------------------------------------------------------------------------


def test_executable_mode_preserved_on_failed_batch_check_and_retry_and_in_place_revert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    script = repo / "bin" / "run.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    orig_body = "#!/bin/sh\necho baseline\n"
    script.write_text(orig_body, encoding="utf-8")
    script.chmod(0o755)
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda r: {"text": "ok"})
    try:
        # 1. Failed batch check restores both original bytes AND 0o755 mode (zero dirty mode diff).
        def bad_writer_factory(name: str, settings: Any, auth: dict[str, Any]):
            if name != settings.writer:
                return None
            return lambda req: WriterReply(
                text=json.dumps({"path": "bin/run.sh", "content": "#!/bin/sh\nexit 1\n"}),
                model="gemini-3.8-flash",
                usage={"prompt_token_count": 50},
                cost_cents=0.01,
            )

        plan_fail = validate_batch_plan(
            {
                "tasks": [
                    {
                        "task_id": "s1",
                        "path": "bin/run.sh",
                        "action": "modify",
                        "intent": "update script",
                        "checks": ["./bin/run.sh"],
                    }
                ]
            },
            repo,
        )
        settings = resolve_batch_settings({})
        res_fail = FileBatchRunner(jm, writer_factory=bad_writer_factory).run(
            plan_fail, settings, {"auth_mode": "api_key", "api_key": "k"}
        )
        assert res_fail["status"] == "checkpoint"
        assert res_fail["totals"]["files_changed"] == 0
        rec_fail = store.read_receipt(res_fail["job_id"])
        wt_script = Path(rec_fail["worktree_dir"]) / "bin" / "run.sh"
        assert wt_script.read_text(encoding="utf-8") == orig_body
        assert (wt_script.stat().st_mode & 0o777) == 0o755

        # 2. Retry where attempt 1 fails and attempt 2 fixes it: `./bin/run.sh` remains executable!
        attempts = {"n": 0}

        def retry_writer_factory(name: str, settings: Any, auth: dict[str, Any]):
            if name != settings.writer:
                return None

            def _w(req: WriterRequest) -> WriterReply:
                attempts["n"] += 1
                body = "#!/bin/sh\nexit 2\n" if attempts["n"] == 1 else "#!/bin/sh\necho fixed\n"
                return WriterReply(
                    text=json.dumps({"path": "bin/run.sh", "content": body}),
                    model="gemini-3.8-flash",
                    usage={"prompt_token_count": 50},
                    cost_cents=0.01,
                )

            return _w

        plan_retry = validate_batch_plan(
            {
                "tasks": [
                    {
                        "task_id": "s2",
                        "path": "bin/run.sh",
                        "action": "modify",
                        "intent": "print fixed status",
                        "checks": ["./bin/run.sh && test \"$(./bin/run.sh)\" = fixed"],
                    }
                ]
            },
            repo,
        )
        res_ok = FileBatchRunner(jm, writer_factory=retry_writer_factory).run(
            plan_retry, settings, {"auth_mode": "api_key", "api_key": "k"}
        )
        assert res_ok["status"] == "ok"
        assert attempts["n"] == 2
        jm.delegate_apply(res_ok["job_id"])
        assert (script.stat().st_mode & 0o777) == 0o755
        assert "fixed" in script.read_text(encoding="utf-8")
        _git_commit_all(repo, "commit fixed script")

        # 3. in_place delegation followed by delegate_revert preserves 0o755 from git_blob.
        monkeypatch.setenv("CC_AGY_ALLOW_IN_PLACE", "true")

        def inplace_worker(req: dict[str, Any]) -> dict[str, Any]:
            p = Path(req["workspace_dir"]) / "bin" / "run.sh"
            p.write_text("#!/bin/sh\necho mutated\n", encoding="utf-8")
            p.chmod(0o644)
            return {"text": "mutated"}

        jm_ip = JobManager(workspace_root=repo, store=store, worker_backend=inplace_worker)
        try:
            d_ip = jm_ip.delegate_start(
                "Edit bin/run.sh in place",
                project="p",
                in_place=True,
                wait=True,
            )
            assert d_ip["status"] == "ok"
            assert (script.stat().st_mode & 0o777) == 0o644
            rev = jm_ip.delegate_revert(d_ip["job_id"])
            assert rev["reverted"] is True
            assert (script.stat().st_mode & 0o777) == 0o755
            assert "fixed" in script.read_text(encoding="utf-8")
        finally:
            jm_ip.shutdown()
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# Issue 4: Control & status operations never block behind 8 busy workers
# ---------------------------------------------------------------------------


def test_control_and_status_calls_bypass_saturated_worker_pool(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    all_eight_entered = threading.Barrier(9, timeout=10.0)
    release_eight = threading.Event()
    job_started = threading.Event()

    def slow_worker(req: dict[str, Any]) -> dict[str, Any]:
        job_started.set()
        release_eight.wait(timeout=10.0)
        return {"text": "done"}

    def slow_ask_backend(req: dict[str, Any]) -> dict[str, Any]:
        all_eight_entered.wait(timeout=10.0)
        release_eight.wait(timeout=10.0)
        return {"text": "pong", "usage": {"prompt_token_count": 10, "candidates_token_count": 5}}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=slow_worker)
    mcp = MCPServer(job_manager=jm, store=store, ask_backend=slow_ask_backend)

    c2s_r, c2s_w = os.pipe()
    s2c_r, s2c_w = os.pipe()
    in_stream = os.fdopen(c2s_r, "rb", buffering=0)
    out_stream = os.fdopen(s2c_w, "wb", buffering=0)
    client_w = os.fdopen(c2s_w, "wb", buffering=0)
    client_r = os.fdopen(s2c_r, "rb")

    responses: dict[Any, dict[str, Any]] = {}
    resp_cond = threading.Condition()

    def _reader_loop() -> None:
        while True:
            line = client_r.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line.decode("utf-8"))
            if "id" in obj:
                with resp_cond:
                    responses[obj["id"]] = obj
                    resp_cond.notify_all()

    server_thread = threading.Thread(target=serve_stdio, args=(mcp, in_stream, out_stream), daemon=True)
    reader_thread = threading.Thread(target=_reader_loop, daemon=True)
    server_thread.start()
    reader_thread.start()

    def _send(msg: dict[str, Any]) -> None:
        client_w.write((json.dumps(msg) + "\n").encode("utf-8"))
        client_w.flush()

    def _wait_id(req_id: Any, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with resp_cond:
            while req_id not in responses:
                rem = deadline - time.monotonic()
                assert rem > 0, f"Timed out waiting for JSON-RPC id={req_id!r} after {timeout}s"
                resp_cond.wait(timeout=rem)
            return responses[req_id]

    try:
        # Start 1 background worker job so we have a real running job_id to query and cancel.
        _send(
            {
                "jsonrpc": "2.0",
                "id": "bg-start",
                "method": "tools/call",
                "params": {
                    "name": "delegate_start",
                    "arguments": {"prompt": "Slow edit", "project": "p", "wait": False},
                },
            }
        )
        start_resp = _wait_id("bg-start", 5.0)
        bg_job_id = json.loads(start_resp["result"]["content"][0]["text"])["job_id"]
        assert job_started.wait(timeout=5.0)

        # Now saturate all 8 threads of rpc_pool with 8 blocking ask_gemini calls.
        for i in range(8):
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": f"slow-{i}",
                    "method": "tools/call",
                    "params": {
                        "name": "ask_gemini",
                        "arguments": {"prompt": f"hold-{i}", "api_key": "fake"},
                    },
                }
            )
        all_eight_entered.wait(timeout=5.0)

        # While all 8 rpc_pool threads are blocked, control calls must complete in < 0.5s!
        t0 = time.monotonic()
        _send(
            {
                "jsonrpc": "2.0",
                "id": "ctl-status",
                "method": "tools/call",
                "params": {"name": "delegate_status", "arguments": {"job_id": bg_job_id}},
            }
        )
        _send(
            {
                "jsonrpc": "2.0",
                "id": "ctl-cancel",
                "method": "tools/call",
                "params": {"name": "delegate_cancel", "arguments": {"job_id": bg_job_id}},
            }
        )
        status_resp = _wait_id("ctl-status", 0.5)
        cancel_resp = _wait_id("ctl-cancel", 0.5)
        elapsed_ctl = time.monotonic() - t0
        assert elapsed_ctl < 0.5
        assert json.loads(status_resp["result"]["content"][0]["text"])["job_id"] == bg_job_id
        assert json.loads(cancel_resp["result"]["content"][0]["text"])["cancelled"] is True
    finally:
        release_eight.set()
        for i in range(8):
            _wait_id(f"slow-{i}", 5.0)
        client_w.close()
        server_thread.join(timeout=5.0)
        out_stream.close()
        client_r.close()
        in_stream.close()
        jm.shutdown()


# ---------------------------------------------------------------------------
# Issue 5: Workspace fingerprint detects changes past _UNTRACKED_HASH_MAX_FILES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_git", [True, False])
def test_workspace_fingerprint_detects_edits_past_untracked_file_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_git: bool
) -> None:
    monkeypatch.setattr(batch_runner, "_UNTRACKED_HASH_MAX_FILES", 3)
    ws = tmp_path / ("git_ws" if is_git else "nogit_ws")
    if is_git:
        _init_git_repo(ws)
        (ws / "tracked.txt").write_text("base\n", encoding="utf-8")
        _git_commit_all(ws)
    else:
        ws.mkdir(parents=True, exist_ok=True)

    for idx in range(6):
        (ws / f"u_{idx:02d}.txt").write_text(f"val-{idx:02d}\n", encoding="utf-8")

    s0 = _workspace_state(ws)

    # 1. Edit content of file #5 (beyond cap=3) changing size -> must change fingerprint.
    f5 = ws / "u_05.txt"
    f5.write_text("val-05-longer\n", encoding="utf-8")
    s1 = _workspace_state(ws)
    assert s1 != s0

    # 2. Same-size content edit on file #5 with mtime restored via os.utime -> ctime_ns changes fingerprint.
    st_before = f5.stat()
    time.sleep(0.01)
    f5.write_text("VAL-05-LONGER\n", encoding="utf-8")
    os.utime(f5, ns=(st_before.st_atime_ns, st_before.st_mtime_ns))
    s2 = _workspace_state(ws)
    assert s2 != s1

    # 3. Permission change on file #5 AND on file #0 -> both change fingerprint.
    f5.chmod(0o755)
    s3 = _workspace_state(ws)
    assert s3 != s2

    f0 = ws / "u_00.txt"
    f0.chmod(0o755)
    s4 = _workspace_state(ws)
    assert s4 != s3

    # 4. Rename file #5 -> changes fingerprint.
    f5.rename(ws / "u_05_renamed.txt")
    s5 = _workspace_state(ws)
    assert s5 != s4


# ---------------------------------------------------------------------------
# Efficiency E1 & E2: Check consolidation & ReceiptStore stat-keyed cache
# ---------------------------------------------------------------------------


def test_shared_checks_consolidate_and_isolate_culprit_on_failure(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    marker_file = tmp_path / "shared_check_invocations.log"
    (repo / "seed.py").write_text("SEED = 1\n", encoding="utf-8")
    (repo / "run_tests.py").write_text(
        "import glob, pathlib, sys\n"
        f"with open({str(marker_file)!r}, 'a', encoding='utf-8') as fh:\n"
        "    fh.write('1\\n')\n"
        "bad = [p for p in glob.glob('m*.py') if 'BROKEN' in pathlib.Path(p).read_text(encoding='utf-8')]\n"
        "sys.exit(1 if bad else 0)\n",
        encoding="utf-8",
    )
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda r: {"text": "ok"})
    shared_check = f"{shlex.quote(sys.executable)} run_tests.py"

    try:
        # Part 1: Happy path — 5 leaf tasks with identical shared check run it ONCE instead of 5 times.
        happy_calls = 0

        def happy_factory(name: str, settings: Any, auth: dict[str, Any]):
            if name != settings.writer:
                return None

            def _w(req: WriterRequest) -> WriterReply:
                nonlocal happy_calls
                happy_calls += 1
                return WriterReply(
                    text=json.dumps({"path": req.task.path, "content": f"OK = {req.task.task_id!r}\n"}),
                    model="gemini-3.8-flash",
                    usage={"prompt_token_count": 50},
                    cost_cents=0.01,
                )

            return _w

        plan_happy = validate_batch_plan(
            {
                "tasks": [
                    {
                        "task_id": f"h{i}",
                        "path": f"m{i}.py",
                        "action": "create",
                        "intent": "create module",
                        "checks": [shared_check],
                    }
                    for i in range(5)
                ]
            },
            repo,
        )
        settings = resolve_batch_settings({"max_parallel": 4})
        res_happy = FileBatchRunner(jm, writer_factory=happy_factory).run(
            plan_happy, settings, {"auth_mode": "api_key", "api_key": "k"}
        )
        assert res_happy["status"] == "ok"
        assert happy_calls == 5
        assert len(marker_file.read_text(encoding="utf-8").splitlines()) == 1

        # Part 2: Culprit isolation — Task c2 writes BROKEN on Rung 1 and fixes it on Rung 2;
        # innocent tasks c0, c1, c3 must NOT re-call the writer (1 call each, 2 calls for c2 = 5 total).
        per_task_calls: dict[str, int] = {}
        call_lock = threading.Lock()

        def culprit_factory(name: str, settings: Any, auth: dict[str, Any]):
            if name != settings.writer:
                return None

            def _w(req: WriterRequest) -> WriterReply:
                with call_lock:
                    n = per_task_calls.get(req.task.task_id, 0) + 1
                    per_task_calls[req.task.task_id] = n
                if req.task.task_id == "c2" and n == 1:
                    body = "BROKEN = True\n"
                else:
                    body = f"CLEAN = {req.task.task_id!r}\n"
                return WriterReply(
                    text=json.dumps({"path": req.task.path, "content": body}),
                    model="gemini-3.8-flash",
                    usage={"prompt_token_count": 50},
                    cost_cents=0.01,
                )

            return _w

        plan_culprit = validate_batch_plan(
            {
                "tasks": [
                    {
                        "task_id": f"c{i}",
                        "path": f"m_c{i}.py",
                        "action": "create",
                        "intent": "create clean module",
                        "checks": [shared_check],
                    }
                    for i in range(4)
                ]
            },
            repo,
        )
        res_culprit = FileBatchRunner(jm, writer_factory=culprit_factory).run(
            plan_culprit, settings, {"auth_mode": "api_key", "api_key": "k"}
        )
        assert res_culprit["status"] == "ok"
        assert per_task_calls == {"c0": 1, "c1": 1, "c2": 2, "c3": 1}
    finally:
        jm.shutdown()


def test_formatter_and_check_share_snapshot_session_while_blocking_check_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    (repo / "test_tamper.py").write_text(
        "import pathlib\n"
        "pathlib.Path('mod.py').write_text('tampered\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (repo / "test_formatted.py").write_text(
        "import pathlib\n"
        "assert pathlib.Path('mod2.py').read_text(encoding='utf-8') == 'x = 1\\n'\n",
        encoding="utf-8",
    )
    _git_commit_all(repo)

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_black = fake_bin / "black"
    fake_black.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "p = pathlib.Path(sys.argv[-1])\n"
        "p.write_text('x = 1\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_black.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda r: {"text": "ok"})
    try:
        def factory(name: str, settings: Any, auth: dict[str, Any]):
            if name != settings.writer:
                return None
            return lambda req: WriterReply(
                text=json.dumps({"path": req.task.path, "content": "x=1\n"}),
                model="gemini-3.8-flash",
                usage={"prompt_token_count": 20},
                cost_cents=0.01,
            )

        # Formatter rewrites {path} cleanly (allowed), but check mutates {path} (forbidden -> fails!).
        mutating_check = f"{shlex.quote(sys.executable)} test_tamper.py"
        plan_bad = validate_batch_plan(
            {
                "formatter": "black {path}",
                "tasks": [
                    {
                        "task_id": "t1",
                        "path": "mod.py",
                        "action": "create",
                        "intent": "create mod",
                        "checks": [mutating_check],
                    }
                ],
            },
            repo,
        )
        res_bad = FileBatchRunner(jm, writer_factory=factory).run(
            plan_bad, resolve_batch_settings({}), {"auth_mode": "api_key", "api_key": "k"}
        )
        assert res_bad["status"] == "checkpoint"
        assert res_bad["totals"]["files_changed"] == 0

        # With a pure read-only check verifying the formatted output, it passes in 1 session!
        clean_check = f"{shlex.quote(sys.executable)} test_formatted.py"
        plan_good = validate_batch_plan(
            {
                "formatter": "black {path}",
                "tasks": [
                    {
                        "task_id": "t2",
                        "path": "mod2.py",
                        "action": "create",
                        "intent": "create mod2",
                        "checks": [clean_check],
                    }
                ],
            },
            repo,
        )
        res_good = FileBatchRunner(jm, writer_factory=factory).run(
            plan_good, resolve_batch_settings({}), {"auth_mode": "api_key", "api_key": "k"}
        )
        assert res_good["status"] == "ok"
    finally:
        jm.shutdown()


def test_receipt_store_stat_cache_zero_reparse_mutation_safety_and_doctor_procless_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    _init_git_repo(ws)
    (ws / "a.txt").write_text("a\n", encoding="utf-8")
    _git_commit_all(ws)

    store = ReceiptStore(tmp_path / "data", workspace_root=ws)
    now = time.time()
    for i in range(200):
        store.write_receipt(
            f"job-bench-{i:04d}",
            {
                "job_id": f"job-bench-{i:04d}",
                "kind": "ask",
                "status": "ok",
                "model": "gemini-3.8-flash",
                "started_at_epoch": now - 10,
                "finished_at_epoch": now - 5 + i * 0.001,
                "usage": {"prompt_token_count": 100, "candidates_token_count": 20},
                "precomputed_cost": {"total_cost_cents": 0.05, "total_cost_usd": 0.0005, "pricing_known": True},
            },
            workspace_root=str(ws),
        )

    # Cold call populates stat cache.
    c1 = store.summarize_costs(workspace_root=ws)
    assert c1["total_jobs"] == 200

    # Warm calls must NOT re-parse any JSON files on disk.
    orig_loads = json.loads
    loads_calls = {"n": 0}

    def counted_loads(*args: Any, **kwargs: Any) -> Any:
        loads_calls["n"] += 1
        return orig_loads(*args, **kwargs)

    import server.receipts as receipts_mod

    monkeypatch.setattr(receipts_mod.json, "loads", counted_loads)
    c2 = store.summarize_costs(workspace_root=ws)
    s2 = store.daily_spend_summary(workspace_root=ws)
    assert c2["total_jobs"] == 200
    assert s2["spent_today_cents"] == pytest.approx(10.0)
    assert loads_calls["n"] == 0

    # Mutating a returned list_receipts() dict must never poison cached state.
    recs = store.list_receipts(workspace_root=ws)
    recs[0]["status"] = "CORRUPTED_IN_MEMORY"
    assert store.list_receipts(workspace_root=ws)[0]["status"] == "ok"

    # External deletion invalidates the entry on next scan.
    (store.receipts_dir / "job-bench-0199.json").unlink()
    assert store.summarize_costs(workspace_root=ws)["total_jobs"] == 199

    # Doctor warns on non-/proc platforms when neither pgrep nor ps is available.
    orig_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self: False if str(self) == "/proc" else orig_is_dir(self))
    orig_which = batch_runner.shutil.which
    monkeypatch.setattr(
        "server.doctor.shutil.which",
        lambda cmd: None if cmd in ("pgrep", "ps") else orig_which(cmd),
    )
    doc = run_doctor(workspace_root=ws, store=store, env={"GEMINI_API_KEY": "fake-key"})
    assert any("pgrep" in w or "ps" in w for w in doc["warnings"])


def test_mixed_dag_consolidated_checks_never_blame_non_leaf_tasks(tmp_path: Path) -> None:
    """a, b, c have no deps and d depends on c; all check run_tests.py. When only leaf a is broken,
    non-leaf c and its dependent d must still land in 1 call each without wasting retries."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "seed.py").write_text("SEED = 1\n", encoding="utf-8")
    (repo / "run_tests.py").write_text(
        "import glob, pathlib, sys\n"
        "bad = [p for p in glob.glob('m_*.py') if 'BROKEN' in pathlib.Path(p).read_text(encoding='utf-8')]\n"
        "sys.exit(1 if bad else 0)\n",
        encoding="utf-8",
    )
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda r: {"text": "ok"})
    shared_check = f"{shlex.quote(sys.executable)} run_tests.py"
    calls: dict[str, int] = {}

    def wf(name: str, settings: Any, auth: dict[str, Any]):
        if name != settings.writer:
            return None

        def _w(req: WriterRequest) -> WriterReply:
            tid = req.task.task_id
            calls[tid] = calls.get(tid, 0) + 1
            body = "BROKEN = True\n" if tid == "a" else f"OK = {tid!r}\n"
            return WriterReply(
                text=json.dumps({"path": req.task.path, "content": body}),
                model="gemini-3.8-flash",
                usage={"prompt_token_count": 40},
                cost_cents=0.01,
            )

        return _w

    try:
        plan = validate_batch_plan(
            {
                "tasks": [
                    {"task_id": "a", "path": "m_a.py", "action": "create", "intent": "create a", "checks": [shared_check]},
                    {"task_id": "b", "path": "m_b.py", "action": "create", "intent": "create b", "checks": [shared_check]},
                    {"task_id": "c", "path": "m_c.py", "action": "create", "intent": "create c", "checks": [shared_check]},
                    {
                        "task_id": "d",
                        "path": "m_d.py",
                        "action": "create",
                        "intent": "create d",
                        "depends_on": ["c"],
                        "checks": [shared_check],
                    },
                ]
            },
            repo,
        )
        settings = resolve_batch_settings({"max_parallel": 1})
        res = FileBatchRunner(jm, writer_factory=wf).run(plan, settings, {"auth_mode": "api_key", "api_key": "k"})
        assert res["status"] == "checkpoint"
        states = {t["task_id"]: (t["state"], t["attempts"]) for t in res["tasks"]}
        assert states == {"a": ("failed", 2), "b": ("landed", 1), "c": ("landed", 1), "d": ("landed", 1)}
        assert calls == {"a": 2, "b": 1, "c": 1, "d": 1}
    finally:
        jm.shutdown()


def test_budget_guard_cross_process_commit_race_and_no_delegate_unpriced_stall_and_cost_report_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. cost_report routes to worker lane so slow transcript scans never clog the 4-thread control lane.
    assert _select_tool_lane({"params": {"name": "cost_report", "arguments": {}}}) == "worker"
    assert _select_tool_lane({"params": {"name": "delegate_status", "arguments": {"job_id": "j1"}}}) == "control"

    # 2. Running delegate_start worker must NEVER stall an unpriced batch/ask reservation, AND failed
    # write_receipt must clean up any live spend for that job.
    monkeypatch.setenv("CC_AGY_MAX_DAILY_CENTS", "10")
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    _git_commit_all(repo)

    data_dir = tmp_path / "data"
    store = ReceiptStore(data_dir, workspace_root=repo)
    in_worker = threading.Event()
    release_worker = threading.Event()

    def slow_delegate_worker(req: dict[str, Any]) -> dict[str, Any]:
        in_worker.set()
        release_worker.wait(timeout=5.0)
        return {"text": "done", "usage": {"prompt_token_count": 1000, "candidates_token_count": 100}}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=slow_delegate_worker)
    try:
        bg = jm.delegate_start("Background task", project="p", wait=False)
        assert in_worker.wait(timeout=3.0)

        t0 = time.monotonic()
        rid = store.budget.reserve("batch-1", estimate_cents=0.0, unpriced=True, workspace_root=repo, wait_timeout=3.0)
        elapsed = time.monotonic() - t0
        assert rid is not None
        assert elapsed < 0.5
        store.budget.settle(rid, job_id="batch-1", actual_cents=8.0, workspace_root=repo, record_live=True)

        # Commit batch-1 receipt (8c) while racing a 5c reservation under cap=10c -> must reject!
        store.write_receipt(
            "batch-1",
            {
                "job_id": "batch-1",
                "kind": "file_batch",
                "status": "ok",
                "model": "gemini-3.8-flash",
                "started_at_epoch": time.time() - 1,
                "finished_at_epoch": time.time(),
                "precomputed_cost": {"total_cost_cents": 8.0, "total_cost_usd": 0.08, "pricing_known": True},
            },
            workspace_root=str(repo),
        )
        with pytest.raises(BudgetExceededError):
            store.budget.reserve("job-over", estimate_cents=5.0, unpriced=False, workspace_root=repo)

        release_worker.set()
        jm.delegate_result(bg["job_id"], wait=True, timeout=5.0)

        # Verify clear_job drops leftover unspent reservations (unblocking waiters immediately) while
        # preserving already-spent live cents if a receipt write ever raises.
        rid_leak = store.budget.reserve("job-leak", estimate_cents=0.4, unpriced=False, workspace_root=repo)
        assert rid_leak is not None
        store.budget.settle(None, job_id="job-leak", actual_cents=1.5, workspace_root=repo, record_live=True)
        # Clean up rid_leak via clear_job: reservation is removed immediately without waiting for timeout,
        # while the 1.5c live spend remains enforced against subsequent reservations.
        store.budget.clear_job("job-leak", workspace_root=repo)
        t_after = time.monotonic()
        rid_ok = store.budget.reserve("probe-after", estimate_cents=0.2, unpriced=False, workspace_root=repo, wait_timeout=2.0)
        assert rid_ok is not None
        assert (time.monotonic() - t_after) < 0.2
        store.budget.settle(rid_ok, job_id="probe-after", actual_cents=0.0, workspace_root=repo, record_live=False)
        assert store.budget.unsettled_live_cents(workspace_root=repo) == pytest.approx(1.5)
        with pytest.raises(BudgetExceededError):
            store.budget.reserve("probe-still-capped", estimate_cents=0.6, unpriced=False, workspace_root=repo)
    finally:
        release_worker.set()
        jm.shutdown()


def test_staged_deferred_leaves_preserved_and_journaled_on_auth_error_and_exception(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "seed.py").write_text("SEED = 1\n", encoding="utf-8")
    (repo / "run_tests.py").write_text(
        "import glob, pathlib, sys\n"
        "bad = [p for p in glob.glob('pkg/*.py') if 'BROKEN' in pathlib.Path(p).read_text(encoding='utf-8')]\n"
        "sys.exit(1 if bad else 0)\n",
        encoding="utf-8",
    )
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda r: {"text": "ok"})
    shared_check = f"{shlex.quote(sys.executable)} run_tests.py"

    try:
        # 1. Three leaf tasks sharing a check: a writes clean code, b writes BROKEN code, c raises WriterAuthError.
        #    a must be materialized to disk, included in files_changed, and journaled;
        #    b must be rolled back on disk and marked 'held' (NEVER 'landed');
        #    c must be marked 'failed'.
        auth_fixed = {"ok": False}
        call_counts: dict[str, int] = {}

        def auth_fail_factory(name: str, settings: Any, auth: dict[str, Any]):
            if name != settings.writer:
                return None

            def _w(req: WriterRequest) -> WriterReply:
                tid = req.task.task_id
                call_counts[tid] = call_counts.get(tid, 0) + 1
                if tid == "c" and not auth_fixed["ok"]:
                    raise WriterAuthError("401 expired credential")
                if tid == "b" and not auth_fixed["ok"]:
                    body = "BROKEN = True\n"
                else:
                    body = f"CLEAN = {tid!r}\n"
                return WriterReply(
                    text=json.dumps({"path": req.task.path, "content": body}),
                    model="gemini-3.8-flash",
                    usage={"prompt_token_count": 30},
                    cost_cents=0.5,
                )

            return _w

        plan_dict = {
            "tasks": [
                {"task_id": "a", "path": "pkg/alpha.py", "action": "create", "intent": "alpha", "checks": [shared_check]},
                {"task_id": "b", "path": "pkg/beta.py", "action": "create", "intent": "beta", "checks": [shared_check]},
                {"task_id": "c", "path": "pkg/gamma.py", "action": "create", "intent": "gamma", "checks": [shared_check]},
            ]
        }
        plan = validate_batch_plan(plan_dict, repo)
        settings = resolve_batch_settings({"max_parallel": 1})
        r1 = FileBatchRunner(jm, writer_factory=auth_fail_factory).run(
            plan, settings, {"auth_mode": "api_key", "api_key": "k"}
        )
        assert r1["status"] == "error"
        states_r1 = {t["task_id"]: t["state"] for t in r1["tasks"]}
        assert states_r1 == {"a": "landed", "b": "held", "c": "failed"}
        rec1 = store.read_receipt(r1["job_id"])
        assert rec1["files_changed"] == ["pkg/alpha.py"]
        wt1 = Path(rec1["worktree_dir"])
        assert (wt1 / "pkg/alpha.py").is_file()
        assert not (wt1 / "pkg/beta.py").exists()

        # Resume after fixing credentials: 'a' was journaled so it is reused with 0 new writer calls!
        auth_fixed["ok"] = True
        r2 = FileBatchRunner(jm, writer_factory=auth_fail_factory).run(
            plan, settings, {"auth_mode": "api_key", "api_key": "k"}, resume_job_id=r1["job_id"]
        )
        assert r2["status"] == "ok"
        states_r2 = {t["task_id"]: t["state"] for t in r2["tasks"]}
        assert states_r2 == {"a": "unchanged", "b": "landed", "c": "landed"}
        assert call_counts["a"] == 1
    finally:
        jm.shutdown()


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("pytest &", "noop"),
        ("pytest & wait", "noop"),
        ("set -e; pytest &", "noop"),
        ("nice -n 5 true", "noop"),
        ("exec true", "noop"),
        ("xargs true </dev/null", "noop"),
        ("make -C sub lint", "syntax_only"),
        ("make -f Makefile.ci -j4 build", "syntax_only"),
        ("python3 -m http.server 0 & sleep 0.1; pytest -q", "behavioral"),
    ],
)
def test_command_strength_background_and_wrapper_residuals(cmd: str, expected: str) -> None:
    assert command_strength(cmd) == expected
