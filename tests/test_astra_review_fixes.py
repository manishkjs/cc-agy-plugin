"""Regression tests for the external (Astra) code review of commit 8ec46f2.

One test class per finding: check-environment credentials, check side effects on tracked files,
per-call batch budget, spend kept on errors, vacuous checks, stale resumes, writer symlink aliases,
atomic apply, partial revert, per-day spend, the Claude escalation race, non-finite settings and
leaked check streams. Each test fails on 8ec46f2 and passes with the fixes, except the few marked
"regression guard", which pin behavior the first fix pass got wrong and 8ec46f2 already had right.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
import warnings

from server.batch_plan import FileTask
from server.batch_runner import BatchControls, FileBatchRunner, _workspace_state
from server.jobs import JobManager, run_acceptance
from server.mcp_stdio import MCPServer
from server.model_routes import resolve_batch_settings
from server.receipts import ReceiptStore
import server.verify as verify_mod
from server.verify import run_commands
from server.writers import AnswerRejected, WriterReply, apply_writer_answer, run_writer_ladder

PY = sys.executable


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repo(tmp: str) -> tuple[Path, ReceiptStore]:
    root = Path(tmp) / "repo"
    (root / "pkg").mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "core.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "pkg" / "data.txt").write_text("original\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root, ReceiptStore(Path(tmp) / "data", workspace_root=root)


def _writer_factory(contents: dict[str, list[str]], calls: list | None = None, *, cost: float = 0.5,
                    raise_on_call: int | None = None):
    """Writers that answer with scripted whole-file contents per path (the last one repeats)."""
    counter = {"n": 0}

    def factory(name, settings, auth):
        def write(req):
            counter["n"] += 1
            if calls is not None:
                calls.append((name, req.task.task_id))
            if raise_on_call is not None and counter["n"] == raise_on_call:
                raise RuntimeError("writer crashed after earlier paid calls")
            queue = contents[req.task.path]
            body = queue.pop(0) if len(queue) > 1 else queue[0]
            return WriterReply(text=json.dumps({"path": req.task.path, "content": body}), cost_cents=cost,
                               usage={"prompt_token_count": 10, "total_token_count": 12})
        return write
    return factory


def _server(root, store, factory):
    jm = JobManager(workspace_root=root, store=store, worker_backend=lambda _r: {"text": "unused"})
    return MCPServer(job_manager=jm, store=store, batch_writer_factory=factory), jm


def _call(srv, name, args):
    resp = srv.handle_Message({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                               "params": {"name": name, "arguments": args}})
    return json.loads(resp["result"]["content"][0]["text"])


def _task(tid, path, **kw):
    return {"task_id": tid, "path": path, "action": kw.pop("action", "create"), "intent": f"Write {path}.", **kw}


_PRINT_ENV = (
    f"{PY} -c \"import os; print('|'.join(os.environ.get(k, '<unset>') for k in "
    "('GEMINI_API_KEY', 'GITHUB_TOKEN', 'ACME_DB_PASSWORD', 'TOKENIZERS_PARALLELISM')))\""
)
_SECRETS = {
    "GEMINI_API_KEY": "dummy-gemini-secret",
    "GITHUB_TOKEN": "dummy-github-secret",
    "ACME_DB_PASSWORD": "dummy-db-secret",
    "TOKENIZERS_PARALLELISM": "false",
}


class CheckEnvironmentCredentialTests(unittest.TestCase):
    """HP-1: acceptance and batch checks must not see credentials."""

    def test_run_acceptance_hides_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.dict(os.environ, _SECRETS):
            root, _ = _repo(tmp)
            out = run_acceptance([_PRINT_ENV], root)[0]["output_tail"]
            for secret in ("dummy-gemini-secret", "dummy-github-secret", "dummy-db-secret"):
                self.assertNotIn(secret, out)
            self.assertIn("false", out, "non-secret variables must still reach the check")

    def test_batch_checks_hide_pattern_matched_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.dict(os.environ, _SECRETS):
            root, _ = _repo(tmp)
            tail = run_commands([_PRINT_ENV], root)[0].tail
            self.assertNotIn("dummy-db-secret", tail)
            self.assertNotIn("dummy-github-secret", tail)

    def test_allowlist_lets_a_named_variable_through(self) -> None:
        env = dict(_SECRETS, CC_AGY_CHECK_ENV_ALLOW="ACME_DB_PASSWORD")
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.dict(os.environ, env):
            root, _ = _repo(tmp)
            out = run_acceptance([_PRINT_ENV], root)[0]["output_tail"]
            self.assertIn("dummy-db-secret", out)
            self.assertNotIn("dummy-gemini-secret", out)

    def test_credential_values_and_more_names_are_caught(self) -> None:
        env = {
            "DATABASE_URL": "postgres://app:s3cret@db.internal/app", "REDIS_URL": "redis://localhost:6379/0",
            "GITHUB_PAT": "x", "RAILS_MASTER_KEY": "x", "SENTRY_DSN": "x", "NPM_AUTHTOKEN": "x",
            "GIT_AUTHOR_NAME": "Dev", "PWD": "/repo", "OLDPWD": "/", "HOME": "/home/dev",
        }
        found = verify_mod.secret_env_names(env)
        self.assertTrue({"DATABASE_URL", "GITHUB_PAT", "RAILS_MASTER_KEY", "SENTRY_DSN", "NPM_AUTHTOKEN"} <= found)
        self.assertFalse({"REDIS_URL", "GIT_AUTHOR_NAME", "PWD", "OLDPWD", "HOME"} & found)

    def test_every_workspace_command_environment_is_credential_free(self) -> None:
        from server.jobs import _build_workdir_exec_env

        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.dict(os.environ, _SECRETS):
            env = _build_workdir_exec_env(Path(tmp))  # baseline probes and runner detection use this
            self.assertEqual(env.get("GEMINI_API_KEY"), "")
            self.assertNotIn("ACME_DB_PASSWORD", env)
            self.assertEqual(env.get("TOKENIZERS_PARALLELISM"), "false")

    def test_secret_name_pattern(self) -> None:
        env = {name: "x" for name in (
            "GEMINI_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "NPM_TOKEN", "JWT_SECRET",
            "PGPASSWORD", "GOOGLE_APPLICATION_CREDENTIALS", "TOKENIZERS_PARALLELISM", "MAX_TOKENS",
            "PATH", "HOME", "SSH_AUTH_SOCK",
        )}
        found = verify_mod.secret_env_names(env)
        self.assertTrue({"GEMINI_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "NPM_TOKEN",
                         "JWT_SECRET", "PGPASSWORD", "GOOGLE_APPLICATION_CREDENTIALS"} <= found)
        # The SSH agent socket lets a command sign in as the user, so it is treated as a credential too.
        self.assertIn("SSH_AUTH_SOCK", found)
        self.assertFalse({"TOKENIZERS_PARALLELISM", "MAX_TOKENS", "PATH", "HOME"} & found)


class CheckSideEffectTests(unittest.TestCase):
    """HP-2: tracked files a check modifies never stay modified (or enter the diff)."""

    def test_run_acceptance_restores_clean_and_worker_edited_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            worker_version = "def add(a, b):\n    return b + a\n"
            (root / "pkg" / "core.py").write_text(worker_version, encoding="utf-8")
            cmd = "echo tampered > pkg/data.txt && echo tampered > pkg/core.py && rm pkg/__init__.py"
            res = run_acceptance([cmd], root)
            self.assertEqual((root / "pkg" / "data.txt").read_text(encoding="utf-8"), "original\n")
            self.assertEqual((root / "pkg" / "core.py").read_text(encoding="utf-8"), worker_version)
            self.assertTrue((root / "pkg" / "__init__.py").exists())
            self.assertEqual(res[-1]["side_effects_restored"], ["pkg/__init__.py", "pkg/core.py", "pkg/data.txt"])
            self.assertFalse(res[-1]["passed"], "a check that rewrote the files under test cannot pass")
            self.assertEqual(res[-1]["untrusted_side_effects"], ["pkg/__init__.py", "pkg/core.py", "pkg/data.txt"])

    def test_in_place_acceptance_reports_without_restoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            res = run_acceptance(["echo tampered > pkg/data.txt"], root, restore_tracked=False)
            self.assertEqual((root / "pkg" / "data.txt").read_text(encoding="utf-8"), "tampered\n")
            self.assertEqual(res[-1]["tracked_side_effects"], ["pkg/data.txt"])
            # In the user's own tree a file outside the change set may be a person's edit: report only.
            self.assertTrue(res[-1]["passed"])

    def test_run_commands_keeps_allowed_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            res = run_commands(["echo fmt >> pkg/core.py && echo x > pkg/data.txt"], root,
                               allowed_mutations=("pkg/core.py",))
            self.assertIn("fmt", (root / "pkg" / "core.py").read_text(encoding="utf-8"))
            self.assertEqual((root / "pkg" / "data.txt").read_text(encoding="utf-8"), "original\n")
            self.assertEqual(res[-1].side_effects_restored, ["pkg/data.txt"])
            self.assertEqual(res[-1].untrusted_side_effects, ["pkg/data.txt"])
            self.assertFalse(res[-1].passed)

    def test_lockfile_churn_is_restored_but_keeps_the_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "package-lock.json").write_text("{}\n", encoding="utf-8")
            _git(root, "add", "-A")
            _git(root, "commit", "-qm", "lockfile")
            res = run_acceptance(["echo '{\"v\": 2}' > package-lock.json"], root)[-1]
            self.assertTrue(res["passed"])
            self.assertEqual(res["side_effects_restored"], ["package-lock.json"])
            self.assertEqual((root / "package-lock.json").read_text(encoding="utf-8"), "{}\n")

    def test_check_that_rewrites_a_change_set_file_cannot_pass(self) -> None:
        # The worker broke add() and wrote a test whose import repairs core.py before asserting.
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            broken = "def add(a, b):\n    return a - b\n"
            (root / "pkg" / "core.py").write_text(broken, encoding="utf-8")
            (root / "test_heal.py").write_text(
                "import pathlib\n"
                "pathlib.Path('pkg/core.py').write_text('def add(a, b):\\n    return a + b\\n')\n"
                "import unittest\nfrom pkg.core import add\n\n\nclass T(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(add(2, 2), 4)\n", encoding="utf-8")
            res = run_acceptance([f"{PY} -m unittest -q test_heal"], root)[-1]
            self.assertFalse(res["passed"])
            self.assertIn("pkg/core.py", res["untrusted_side_effects"])
            self.assertEqual((root / "pkg" / "core.py").read_text(encoding="utf-8"), broken)

    def test_side_effect_on_a_file_created_earlier_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "pkg" / "new_mod.py").write_text("X = 1\n", encoding="utf-8")  # created by the worker
            res = run_commands(["echo 'X = 2' >> pkg/new_mod.py"], root)[-1]
            self.assertFalse(res.passed)
            self.assertEqual(res.untrusted_side_effects, ["pkg/new_mod.py"])
            self.assertEqual((root / "pkg" / "new_mod.py").read_text(encoding="utf-8"), "X = 1\n")

    def test_failed_batch_check_side_effects_leave_no_trace_in_the_job_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            # Plan validation refuses visible writes, so the side effect hides inside a failing test.
            (root / "tests_x").mkdir()
            (root / "tests_x" / "__init__.py").write_text("", encoding="utf-8")
            (root / "tests_x" / "test_sneaky.py").write_text(
                "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                "        open('pkg/core.py', 'a').write('# tampered\\n')\n        self.fail('always red')\n",
                encoding="utf-8")
            _git(root, "add", "-A")
            _git(root, "commit", "-qm", "sneaky check")
            srv, jm = _server(root, store, _writer_factory({"pkg/a.py": ["A = 1\n"]}))
            check = "python3 -m unittest -q tests_x.test_sneaky"
            out = _call(srv, "delegate_file_batch", {"tasks": [_task("a", "pkg/a.py", checks=[check])],
                                                     "project": "p", "claude_writer": False})
            self.assertEqual(out["tasks"][0]["state"], "failed")
            diff = store.diff_path(out["job_id"]).read_text(encoding="utf-8")
            self.assertNotIn("pkg/core.py", diff)
            self.assertIn("pkg/core.py", out["check_side_effects"]["a"])
            jm.shutdown()


class BatchCreatedFileSideEffectTests(unittest.TestCase):
    def test_check_editing_a_file_an_earlier_task_created_never_reaches_the_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            (root / "tests_x").mkdir()
            (root / "tests_x" / "__init__.py").write_text("", encoding="utf-8")
            (root / "tests_x" / "test_inject.py").write_text(
                "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                "        open('pkg/a.py', 'a').write('INJECTED = 1\\n')\n", encoding="utf-8")
            _git(root, "add", "-A")
            _git(root, "commit", "-qm", "inject test")
            srv, jm = _server(root, store, _writer_factory({"pkg/a.py": ["A = 1\n"], "pkg/b.py": ["B = 1\n"]}))
            tasks = [_task("a", "pkg/a.py"),
                     _task("b", "pkg/b.py", depends_on=["a"], checks=["python3 -m unittest -q tests_x.test_inject"])]
            out = _call(srv, "delegate_file_batch", {"tasks": tasks, "project": "p", "claude_writer": False})
            diff = store.diff_path(out["job_id"]).read_text(encoding="utf-8")
            self.assertNotIn("INJECTED", diff)
            self.assertIn("pkg/a.py", out["check_side_effects"]["b"])
            jm.shutdown()


class BatchBudgetTests(unittest.TestCase):
    """HP-3: the batch budget is enforced per writer call, not after the whole retry ladder."""

    def test_retries_stop_as_soon_as_the_budget_is_spent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            calls: list = []
            factory = _writer_factory({"pkg/a.py": ["bad (\n"]}, calls, cost=2.0)
            srv, jm = _server(root, store, factory)
            out = _call(srv, "delegate_file_batch", {
                "tasks": [_task("a", "pkg/a.py", checks=["python3 -m py_compile pkg/a.py"])],
                "project": "p", "max_cost_cents": 1.0, "max_parallel": 1,
            })
            self.assertEqual(len(calls), 1, f"no retry may start once the budget is spent: {calls}")
            self.assertEqual(out["tasks"][0]["state"], "held")
            self.assertIn("budget", out["error"])
            self.assertAlmostEqual(out["totals"]["cost_cents"], 2.0)
            jm.shutdown()

    @staticmethod
    def _admit_in_thread(controls: BatchControls, name: str = "genai") -> tuple[threading.Thread, list]:
        verdict: list = []
        thread = threading.Thread(target=lambda: verdict.append(controls.before_call(name)), daemon=True)
        thread.start()
        return thread, verdict

    def test_reservation_waits_for_in_flight_calls_instead_of_stopping(self) -> None:
        controls = BatchControls(resolve_batch_settings({"max_cost_cents": 10}), threading.Event())
        self.assertTrue(controls.before_call("genai"))
        controls.after_call("genai", WriterReply(text="", cost_cents=4.0))
        self.assertTrue(controls.before_call("genai"))           # 4 + (0 + 1) * 4 = 8 <= 10
        thread, verdict = self._admit_in_thread(controls)       # 4 + (1 + 1) * 4 = 12 > 10: wait
        thread.join(0.3)
        self.assertTrue(thread.is_alive(), "a call that only fails to fit because of in-flight calls waits")
        self.assertFalse(controls.stop.is_set())
        controls.after_call("genai", WriterReply(text="", cost_cents=1.0))  # spent 5: 5 + 4 <= 10
        thread.join(3)
        self.assertEqual(verdict, [True])
        self.assertFalse(controls.stop.is_set())

    def test_batch_stops_only_when_one_more_call_cannot_fit(self) -> None:
        controls = BatchControls(resolve_batch_settings({"max_cost_cents": 10}), threading.Event())
        self.assertTrue(controls.before_call("genai"))
        controls.after_call("genai", WriterReply(text="", cost_cents=4.0))
        self.assertTrue(controls.before_call("genai"))
        thread, verdict = self._admit_in_thread(controls)
        controls.after_call("genai", WriterReply(text="", cost_cents=4.0))  # spent 8: 8 + 4 > 10
        thread.join(3)
        self.assertEqual(verdict, [False])
        self.assertTrue(controls.stop.is_set())
        self.assertIn("budget", controls.stop_reason)

    def test_first_priced_call_runs_alone(self) -> None:
        # On a resume the warm-up task can be "unchanged" (no call), so the controls themselves must
        # hold parallel calls until one price is known.
        controls = BatchControls(resolve_batch_settings({"max_cost_cents": 5}), threading.Event())
        self.assertTrue(controls.before_call("genai"))
        thread, verdict = self._admit_in_thread(controls)
        thread.join(0.3)
        self.assertTrue(thread.is_alive())
        controls.after_call("genai", WriterReply(text="", cost_cents=4.0))  # 4 + 4 > 5: stop
        thread.join(3)
        self.assertEqual(verdict, [False])

    def test_budgeted_parallel_batch_keeps_going_until_the_budget_is_used(self) -> None:
        # Regression guard: the first fix pass stopped this batch at 3 of 8 files with half the budget left.
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            prices = iter([30.0] + [10.0] * 20)
            price_lock = threading.Lock()

            def factory(name, settings, auth):
                def write(req):
                    time.sleep(0.2)
                    with price_lock:
                        cost = next(prices)
                    return WriterReply(text=json.dumps({"path": req.task.path, "content": "X = 1\n"}), cost_cents=cost)
                return write

            srv, jm = _server(root, store, factory)
            tasks = [_task(f"t{i}", f"pkg/t{i}.py") for i in range(8)]
            out = _call(srv, "delegate_file_batch", {"tasks": tasks, "project": "p", "max_cost_cents": 100,
                                                     "max_parallel": 4, "claude_writer": False})
            self.assertLessEqual(out["totals"]["cost_cents"], 100.0)
            self.assertGreaterEqual(out["totals"]["landed"], 5, out)
            jm.shutdown()


class SpendKeptOnErrorTests(unittest.TestCase):
    """HP-4: paid usage survives task-level and batch-level exceptions."""

    def test_task_exception_keeps_cost_of_earlier_rungs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            factory = _writer_factory({"pkg/a.py": ["bad (\n"]}, cost=3.0, raise_on_call=2)
            srv, jm = _server(root, store, factory)
            out = _call(srv, "delegate_file_batch", {
                "tasks": [_task("a", "pkg/a.py", checks=["python3 -m py_compile pkg/a.py"])],
                "project": "p", "claude_writer": False,
            })
            self.assertEqual(out["tasks"][0]["state"], "failed")
            rec = store.read_receipt(out["job_id"])
            self.assertAlmostEqual(rec["tasks"][0]["cost_cents"], 3.0)
            self.assertAlmostEqual(rec["cost"]["total_cost_cents"], 3.0)
            jm.shutdown()

    def test_batch_error_receipt_keeps_the_run_spend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = _server(root, store, _writer_factory({"pkg/a.py": ["A = 1\n"]}, cost=1.5))
            with unittest.mock.patch.object(jm, "_collect_worktree_diff", side_effect=RuntimeError("disk full")):
                out = _call(srv, "delegate_file_batch", {"tasks": [_task("a", "pkg/a.py")], "project": "p"})
            self.assertEqual(out["status"], "error")
            rec = store.read_receipt(out["job_id"])
            self.assertAlmostEqual(rec["cost"]["total_cost_cents"], 1.5)
            self.assertEqual(rec["usage"]["prompt_token_count"], 10)
            jm.shutdown()


class VacuousCheckTests(unittest.TestCase):
    """HP-5: zero-test and all-skipped runs do not verify anything, and the digest says so."""

    def test_all_skipped_pytest_run_is_not_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "test_skip.py").write_text(
                "import pytest\n\n@pytest.mark.skip(reason='later')\ndef test_x():\n    assert False\n", encoding="utf-8")
            res = run_commands([f"{PY} -m pytest -q -p no:cacheprovider test_skip.py"], root)[0]
            self.assertEqual(res.exit_code, 0)
            self.assertFalse(res.passed)
            self.assertTrue(res.vacuous)

    def test_zero_test_unittest_run_is_not_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            res = run_commands([f"{PY} -m unittest discover -s pkg -p 'nothing_*.py'"], root)[0]
            self.assertFalse(res.passed)
            self.assertTrue(res.vacuous)

    def test_passing_runs_that_skip_some_tests_are_not_vacuous(self) -> None:
        zero = {"passed": 0, "failed": 0, "errors": 0}
        phpunit = "OK, but incomplete, skipped, or risky tests!\nTests: 10, Assertions: 20, Skipped: 1.\n"
        dotnet = "Passed!  - Failed:     0, Passed:     5, Skipped:     1, Total:     6, Duration: 1 s\n"
        jest = "Tests:       1 skipped, 5 passed, 6 total\n"
        for out in (phpunit, dotnet, jest):
            self.assertFalse(verify_mod.vacuous_test_run(out, 0, False, zero), out)
        all_skipped = "Tests: 3, Assertions: 0, Skipped: 3.\n"
        self.assertTrue(verify_mod.vacuous_test_run(all_skipped, 0, False, zero))

    def test_batch_without_checks_is_flagged_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = _server(root, store, _writer_factory({"pkg/a.py": ["A = 1\n"]}))
            out = _call(srv, "delegate_file_batch", {"tasks": [_task("a", "pkg/a.py")], "project": "p"})
            self.assertEqual(out["verification"], "unverified")
            self.assertEqual(out["unverified_task_ids"], ["a"])
            self.assertIn("nothing verified", out["warning"])
            self.assertTrue(out["next"].startswith("review "))
            self.assertEqual(store.read_receipt(out["job_id"])["verification"], "unverified")
            jm.shutdown()


class ResumeCompatibilityTests(unittest.TestCase):
    """HP-6: resumes never reuse a job copy from another plan or an older workspace."""

    def test_untracked_content_changes_the_workspace_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "notes.txt").write_text("a\n", encoding="utf-8")
            (root / ".cc-agy-data").mkdir()
            (root / ".cc-agy-data" / "receipt.json").write_text("{}", encoding="utf-8")
            first = _workspace_state(root)
            (root / "notes.txt").write_text("b\n", encoding="utf-8")
            self.assertNotEqual(first, _workspace_state(root))
            second = _workspace_state(root)
            (root / ".cc-agy-data" / "receipt.json").write_text('{"status": "ok"}', encoding="utf-8")
            (root / ".ctx-session-reads").mkdir()
            (root / ".ctx-session-reads" / "r1").write_text("x", encoding="utf-8")
            self.assertEqual(second, _workspace_state(root), "plugin data inside the workspace is ignored")

    def test_explicit_resume_checks_plan_and_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = _server(root, store, _writer_factory({"pkg/a.py": ["bad (\n"], "pkg/b.py": ["B = 1\n"]}))
            task_a = _task("a", "pkg/a.py", checks=["python3 -m py_compile pkg/a.py"])
            first = _call(srv, "delegate_file_batch", {"tasks": [task_a], "project": "p", "claude_writer": False})
            self.assertEqual(first["status"], "checkpoint")
            other_plan = _call(srv, "delegate_file_batch", {"tasks": [_task("b", "pkg/b.py")], "project": "p",
                                                            "resume_job_id": first["job_id"]})
            self.assertIn("different plan", json.dumps(other_plan))
            (root / "scratch.txt").write_text("new untracked file\n", encoding="utf-8")
            moved = _call(srv, "delegate_file_batch", {"tasks": [task_a], "project": "p", "claude_writer": False,
                                                       "resume_job_id": first["job_id"]})
            self.assertIn("workspace changed", json.dumps(moved))
            jm.shutdown()


class WriterSymlinkTests(unittest.TestCase):
    """HP-7: a writer answer can never land through a symlink alias."""

    def test_alias_to_protected_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
            os.symlink(".env", root / "safe.txt")
            task = FileTask(task_id="t", path="safe.txt", action="modify", intent="edit")
            with self.assertRaises(AnswerRejected) as ctx:
                apply_writer_answer(task, json.dumps({"path": "safe.txt", "content": "pwned\n"}), root)
            self.assertIn("symlink", str(ctx.exception))
            self.assertEqual((root / ".env").read_text(encoding="utf-8"), "SECRET=1\n")

    def test_linked_directory_inside_the_workspace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            os.symlink("pkg", root / "alias")
            task = FileTask(task_id="t", path="alias/new.py", action="create", intent="create")
            with self.assertRaises(AnswerRejected):
                apply_writer_answer(task, json.dumps({"path": "alias/new.py", "content": "X = 1\n"}), root)


def _applied_job_fixture(tmp: str):
    """A finished delegate job that changed pkg/core.py and pkg/data.txt (not applied yet)."""
    root, store = _repo(tmp)

    def worker(req: dict) -> dict:
        wt = Path(req["workspace_dir"])
        (wt / "pkg" / "core.py").write_text("def add(a, b):\n    return b + a\n", encoding="utf-8")
        (wt / "pkg" / "data.txt").write_text("changed\n", encoding="utf-8")
        return {"text": "done", "usage": {"prompt_token_count": 100, "candidates_token_count": 20,
                                          "total_token_count": 120}, "tool_calls": [], "thoughts": []}

    jm = JobManager(workspace_root=root, store=store, worker_backend=worker)
    d = jm.delegate_start(prompt="Swap add operands and update data", model="gemini-3.8-flash", project="p",
                          wait=True)
    return root, store, jm, d


class ApplyAndRevertTests(unittest.TestCase):
    """O-1: apply is all-or-nothing (O-2's partial revert is covered in test_cc_agy_plugin.py)."""

    def test_failed_write_rolls_back_files_already_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store, jm, d = _applied_job_fixture(tmp)
            before = {p: (root / p).read_bytes() for p in ("pkg/core.py", "pkg/data.txt")}
            import shutil as _shutil
            real_copy = _shutil.copy2
            seen = {"n": 0}

            def flaky_copy(src, dst, *a, **kw):
                seen["n"] += 1
                if seen["n"] == 2:
                    raise OSError("disk full")
                return real_copy(src, dst, *a, **kw)

            with unittest.mock.patch("server.jobs.shutil.copy2", side_effect=flaky_copy):
                with self.assertRaises(OSError):
                    jm.delegate_apply(d["job_id"])
            self.assertEqual({p: (root / p).read_bytes() for p in before}, before)
            self.assertFalse(store.read_receipt(d["job_id"]).get("applied"))
            jm.shutdown()


class RevertDigestTests(unittest.TestCase):
    def test_full_revert_keeps_reverted_files_in_status(self) -> None:
        # Regression guard: the first fix pass dropped reverted_files from a fully reverted job's status.
        with tempfile.TemporaryDirectory() as tmp:
            root, store, jm, d = _applied_job_fixture(tmp)
            jm.delegate_apply(d["job_id"])
            jm.delegate_revert(d["job_id"])
            status = jm.delegate_status(d["job_id"])
            self.assertTrue(status.get("reverted"))
            self.assertEqual(sorted(status.get("reverted_files") or []), ["pkg/core.py", "pkg/data.txt"])
            jm.shutdown()


class DailySpendTests(unittest.TestCase):
    """O-3: a job resumed across midnight is charged to each day it ran."""

    def test_resumed_receipt_splits_spend_by_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            now = time.time()
            yesterday = now - 86_400
            base = {"kind": "file_batch", "status": "checkpoint", "model": "gemini-3.8-flash"}
            store.write_receipt("job-1-aaaaaaaa", {**base, "started_at_epoch": yesterday, "finished_at_epoch": yesterday,
                                                   "precomputed_cost": {"total_cost_cents": 10.0}},
                                workspace_root=str(root))
            store.write_receipt("job-1-aaaaaaaa", {**base, "started_at_epoch": yesterday, "finished_at_epoch": now,
                                                   "precomputed_cost": {"total_cost_cents": 25.0}},
                                workspace_root=str(root))
            today = store.daily_spend_summary(workspace_root=root, now_epoch=now)
            self.assertAlmostEqual(today["spent_today_cents"], 15.0)
            earlier = store.daily_spend_summary(workspace_root=root, now_epoch=yesterday)
            self.assertAlmostEqual(earlier["spent_today_cents"], 10.0)


class DailySpendRewriteTests(unittest.TestCase):
    def test_rewrite_with_stale_buckets_still_charges_the_new_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            now = time.time()
            store.write_receipt("job-2-bbbbbbbb", {"kind": "delegate", "status": "ok", "model": "gemini-3.8-flash",
                                                   "started_at_epoch": now, "finished_at_epoch": now,
                                                   "precomputed_cost": {"total_cost_cents": 10.0}},
                                workspace_root=str(root))
            rec = store.read_receipt("job-2-bbbbbbbb")
            rec["precomputed_cost"] = {"total_cost_cents": 15.0}  # e.g. a QA review appended later
            store.write_receipt("job-2-bbbbbbbb", rec, workspace_root=str(root))
            again = store.read_receipt("job-2-bbbbbbbb")
            self.assertAlmostEqual(sum(again["cost_by_utc_day"].values()), 15.0)
            self.assertAlmostEqual(store.daily_spend_summary(workspace_root=root, now_epoch=now)["spent_today_cents"], 15.0)


class ClaudeEscalationRaceTests(unittest.TestCase):
    """O-4: the escalation slot is reserved atomically."""

    def test_second_claude_call_is_refused_even_after_both_passed_the_check(self) -> None:
        settings = resolve_batch_settings({"max_claude_escalations": 1})
        controls = BatchControls(settings, threading.Event())
        self.assertTrue(controls.claude_allowed())
        self.assertTrue(controls.claude_allowed())   # both tasks see a free slot
        self.assertTrue(controls.before_call("claude"))
        self.assertFalse(controls.before_call("claude"))
        self.assertEqual(controls.claude_calls, 1)

    def test_a_call_without_reply_gives_the_escalation_slot_back(self) -> None:
        controls = BatchControls(resolve_batch_settings({"max_claude_escalations": 1}), threading.Event())
        self.assertTrue(controls.before_call("claude"))
        controls.after_call("claude", None)  # transient error or timeout: nothing came back
        self.assertTrue(controls.claude_allowed())
        self.assertTrue(controls.before_call("claude"))

    def test_backoff_sleep_does_not_hold_a_budget_slot(self) -> None:
        from server.writers import WriterTransportError

        controls = BatchControls(resolve_batch_settings({"max_cost_cents": 100}), threading.Event())
        task = FileTask(task_id="t", path="a.py", action="create", intent="x")
        attempts = {"n": 0}
        seen_in_flight: list[int] = []

        def flaky(req):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise WriterTransportError(429, "busy")
            return WriterReply(text=json.dumps({"path": "a.py", "content": "X = 1\n"}), cost_cents=1.0)

        with tempfile.TemporaryDirectory() as tmp:
            outcome = run_writer_ladder(task, routed="genai", writers={"genai": flaky}, verify=lambda ans: (True, ""),
                                        root=tmp, before_call=controls.before_call, after_call=controls.after_call,
                                        should_stop=controls.stop.is_set,
                                        sleep=lambda _d: seen_in_flight.append(controls.in_flight))
        self.assertEqual(outcome.state, "landed")
        self.assertEqual(seen_in_flight, [0])

    def test_ladder_skips_a_refused_rung_without_holding_the_task(self) -> None:
        task = FileTask(task_id="t", path="a.py", action="create", intent="x")
        replies = {"routed": 0}

        def routed(req):
            replies["routed"] += 1
            return WriterReply(text=json.dumps({"path": "a.py", "content": "bad (\n"}))

        writers = {"genai": routed, "claude": lambda req: self.fail("refused rung must not be called")}
        with tempfile.TemporaryDirectory() as tmp:
            outcome = run_writer_ladder(task, routed="genai", writers=writers, verify=lambda ans: (False, "nope"),
                                        root=tmp, before_call=lambda name: name != "claude",
                                        should_stop=lambda: False)
        self.assertEqual(outcome.state, "failed")
        self.assertEqual(replies["routed"], 2)


class NonFiniteSettingsTests(unittest.TestCase):
    """O-5: NaN and infinity are rejected for numeric batch settings."""

    def test_non_finite_values_are_rejected(self) -> None:
        for args in ({"max_cost_cents": "nan"}, {"max_cost_cents": float("inf")},
                     {"check_timeout_seconds": "nan"}, {"check_timeout_seconds": "inf"}):
            with self.assertRaises(ValueError, msg=str(args)):
                resolve_batch_settings(args)


class CheckStreamTests(unittest.TestCase):
    """O-6: check runs close their output pipe."""

    def test_run_commands_leaves_no_unclosed_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                run_commands(["echo hello"], root)
                gc.collect()
            leaks = [w for w in caught if issubclass(w.category, ResourceWarning) and "unclosed file" in str(w.message)]
            self.assertEqual(leaks, [])


if __name__ == "__main__":
    unittest.main()


# ------------------------------------------------------------------------------------------------
# Round 2 critique findings (Gemini Pro Next and Claude Fable 5.1)
# ------------------------------------------------------------------------------------------------


class GitHardeningTests(unittest.TestCase):
    """Restores and later plugin git calls never run repo hooks, and planted git metadata is undone."""

    def test_restore_does_not_run_repo_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.dict(os.environ, _SECRETS):
            root, _ = _repo(tmp)
            leak = Path(tmp) / "hook_env.txt"
            hook = root / ".git" / "hooks" / "post-checkout"
            hook.write_text(f"#!/bin/sh\necho \"$GEMINI_API_KEY\" > {leak}\n", encoding="utf-8")
            hook.chmod(0o755)
            res = run_commands(["echo dirty > pkg/data.txt"], root)[-1]
            self.assertFalse(res.passed)
            self.assertEqual((root / "pkg" / "data.txt").read_text(encoding="utf-8"), "original\n")
            self.assertFalse(leak.exists(), "git checkout during the restore must not run repo hooks")

    def test_check_planting_git_hooks_or_config_is_undone_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            config_before = (root / ".git" / "config").read_text(encoding="utf-8")
            cmd = ("printf '#!/bin/sh\\necho pwned\\n' > .git/hooks/pre-commit && "
                   "git config core.fsmonitor 'sh -c \"echo pwned\"'")
            res = run_commands([cmd], root)[-1]
            self.assertFalse(res.passed)
            self.assertIn(".git/hooks/pre-commit", res.untrusted_side_effects)
            self.assertIn(".git/config", res.untrusted_side_effects)
            self.assertFalse((root / ".git" / "hooks" / "pre-commit").exists())
            self.assertEqual((root / ".git" / "config").read_text(encoding="utf-8"), config_before)


class SideEffectBypassTests(unittest.TestCase):
    def test_staged_edit_is_detected_and_fully_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            res = run_commands(["echo STAGED > pkg/data.txt && git add pkg/data.txt"], root)[-1]
            self.assertFalse(res.passed)
            self.assertIn("pkg/data.txt", res.untrusted_side_effects)
            self.assertEqual((root / "pkg" / "data.txt").read_text(encoding="utf-8"), "original\n")
            staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True)
            self.assertEqual(staged.stdout.strip(), "")

    def test_check_run_from_a_subdirectory_cannot_touch_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "sub").mkdir()
            res = run_commands(["echo TAMPER > ../pkg/data.txt"], root / "sub")[-1]
            self.assertFalse(res.passed)
            self.assertIn("pkg/data.txt", res.untrusted_side_effects)
            self.assertEqual((root / "pkg" / "data.txt").read_text(encoding="utf-8"), "original\n")

    def test_allowed_target_turned_into_a_symlink_is_restored_not_followed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            outside = Path(tmp) / "outside.txt"
            outside.write_text("DO_NOT_TOUCH\n", encoding="utf-8")
            (root / "pkg" / "core.py").write_text("NEW = 1\n", encoding="utf-8")  # the task's candidate
            cmd = f"rm pkg/core.py && ln -s {outside} pkg/core.py"
            res = run_commands([cmd], root, allowed_mutations=("pkg/core.py",))[-1]
            self.assertFalse(res.passed)
            self.assertIn("pkg/core.py", res.untrusted_side_effects)
            self.assertFalse((root / "pkg" / "core.py").is_symlink())
            self.assertEqual((root / "pkg" / "core.py").read_text(encoding="utf-8"), "NEW = 1\n")
            self.assertEqual(outside.read_text(encoding="utf-8"), "DO_NOT_TOUCH\n")


class ResumeBudgetTests(unittest.TestCase):
    def test_resumed_batch_counts_prior_spend_and_escalations(self) -> None:
        prior = {"cost": {"total_cost_cents": 2.0}, "tasks": [{"history": [
            {"writer": "genai", "result": "checks_failed", "cost_cents": 0.5},
            {"writer": "claude", "result": "checks_failed", "cost_cents": 1.5},
            {"writer": "claude", "result": "not_admitted", "cost_cents": 0.0},
        ]}]}
        settings = resolve_batch_settings({"max_cost_cents": 2.0, "max_claude_escalations": 1})
        controls = BatchControls(settings, threading.Event(), prior=prior)
        self.assertEqual(controls.claude_calls, 1)
        self.assertAlmostEqual(controls.max_call_cents_by_writer["claude"], 1.5)
        self.assertFalse(controls.before_call("genai"))
        self.assertIn("budget", controls.stop_reason)

    def test_free_replies_keep_calls_serialized_under_a_cap(self) -> None:
        controls = BatchControls(resolve_batch_settings({"max_cost_cents": 10}), threading.Event())
        self.assertTrue(controls.before_call("genai"))
        controls.after_call("genai", WriterReply(text="", cost_cents=0.0))
        self.assertTrue(controls.before_call("genai"))
        verdict: list = []
        waiter = threading.Thread(target=lambda: verdict.append(controls.before_call("genai")), daemon=True)
        waiter.start()
        waiter.join(0.3)
        self.assertTrue(waiter.is_alive(), "an unpriced writer must not fan out under a cost cap")
        controls.after_call("genai", WriterReply(text="", cost_cents=0.0))
        waiter.join(3)
        self.assertEqual(verdict, [True])

    def test_first_claude_call_waits_for_cheaper_calls_in_flight(self) -> None:
        controls = BatchControls(resolve_batch_settings({"max_cost_cents": 100}), threading.Event())
        self.assertTrue(controls.before_call("genai"))
        controls.after_call("genai", WriterReply(text="", cost_cents=1.0))
        self.assertTrue(controls.before_call("genai"))
        verdict: list = []
        waiter = threading.Thread(target=lambda: verdict.append(controls.before_call("claude")), daemon=True)
        waiter.start()
        waiter.join(0.3)
        self.assertTrue(waiter.is_alive(), "Claude has no price yet, so its first call runs alone")
        controls.after_call("genai", WriterReply(text="", cost_cents=1.0))
        waiter.join(3)
        self.assertEqual(verdict, [True])

    def test_zero_check_timeout_is_clamped_not_replaced(self) -> None:
        self.assertEqual(resolve_batch_settings({"check_timeout_seconds": 0}).check_timeout, 5.0)


class QaReviewSpendTests(unittest.TestCase):
    def test_qa_review_cost_is_added_to_a_precomputed_receipt(self) -> None:
        from server.regions import estimate_cost_usd

        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = _server(root, store, _writer_factory({"pkg/a.py": ["A = 1\n"]}, cost=1.5))
            out = _call(srv, "delegate_file_batch", {"tasks": [_task("a", "pkg/a.py")], "project": "p"})
            usage = {"prompt_token_count": 50_000, "candidates_token_count": 2_000, "total_token_count": 52_000}
            jm.record_qa_review_on_receipt(out["job_id"], qa_model="gemini-3.8-flash", qa_meta={"verdict": "PASS"},
                                           qa_usage=usage)
            rec = store.read_receipt(out["job_id"])
            qa_cents = float(estimate_cost_usd("gemini-3.8-flash", usage, single_request=True)["total_cost_cents"])
            self.assertGreater(qa_cents, 0.0)
            self.assertAlmostEqual(rec["cost"]["total_cost_cents"], round(1.5 + qa_cents, 4), places=3)
            self.assertAlmostEqual(sum(rec["cost_by_utc_day"].values()), rec["cost"]["total_cost_cents"], places=3)
            jm.shutdown()


class ApplyRevertHardeningTests(unittest.TestCase):
    def test_revert_replaces_a_read_only_applied_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store, jm, d = _applied_job_fixture(tmp)
            before = (root / "pkg" / "core.py").read_bytes()
            jm.delegate_apply(d["job_id"])
            os.chmod(root / "pkg" / "core.py", 0o444)
            out = jm.delegate_revert(d["job_id"])
            self.assertTrue(out["reverted"])
            self.assertEqual((root / "pkg" / "core.py").read_bytes(), before)
            jm.shutdown()

    def test_apply_replaces_a_host_symlink_instead_of_writing_through_it(self) -> None:
        # The link points inside the workspace at a file with the baseline bytes, so no conflict is
        # raised; 8ec46f2 wrote the job's bytes through it into secret.txt.
        with tempfile.TemporaryDirectory() as tmp:
            root, store, jm, d = _applied_job_fixture(tmp)
            secret = root / "secret.txt"
            secret.write_text((root / "pkg" / "data.txt").read_text(encoding="utf-8"), encoding="utf-8")
            (root / "pkg" / "data.txt").unlink()
            os.symlink("../secret.txt", root / "pkg" / "data.txt")
            try:
                jm.delegate_apply(d["job_id"])
            except Exception:  # noqa: BLE001 - refusing is also safe; writing through is not
                pass
            self.assertEqual(secret.read_text(encoding="utf-8"), "original\n")
            jm.shutdown()

    def test_revert_flags_are_signed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store, jm, d = _applied_job_fixture(tmp)
            jm.delegate_apply(d["job_id"])
            rec = store.read_receipt(d["job_id"])
            self.assertTrue(store.verify_receipt_signature(rec))
            rec["reverted"] = True
            self.assertFalse(store.verify_receipt_signature(rec), "a forged reverted flag must break the HMAC")
            jm.shutdown()


class GoCargoVacuousTests(unittest.TestCase):
    def test_go_and_cargo_runs_without_tests_are_vacuous(self) -> None:
        zero = {"passed": 0, "failed": 0, "errors": 0}
        go_no_files = "?   \texample.com/pkg\t[no test files]\n"
        go_none_matched = "testing: warning: no tests to run\nPASS\nok  \texample.com/pkg\t0.004s [no tests to run]\n"
        cargo_none = "running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored\n"
        for out in (go_no_files, go_none_matched, cargo_none):
            self.assertTrue(verify_mod.vacuous_test_run(out, 0, False, {"passed": 1, "failed": 0, "errors": 0}
                                                        if "PASS" in out else zero), out)
        go_real = "=== RUN   TestX\n--- PASS: TestX (0.00s)\nPASS\nok  \texample.com/pkg\t0.004s\n"
        cargo_real = "running 2 tests\ntest a ... ok\ntest b ... ok\n\ntest result: ok. 2 passed; 0 failed\n"
        self.assertFalse(verify_mod.vacuous_test_run(go_real, 0, False, {"passed": 1, "failed": 0, "errors": 0}))
        self.assertFalse(verify_mod.vacuous_test_run(cargo_real, 0, False, {"passed": 2, "failed": 0, "errors": 0}))


class WorkspaceFingerprintEdgeTests(unittest.TestCase):
    def test_non_git_workspace_fingerprint_tracks_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("1\n", encoding="utf-8")
            first = _workspace_state(root)
            (root / "a.txt").write_text("2\n", encoding="utf-8")
            self.assertNotEqual(first, _workspace_state(root))

    def test_unborn_head_staged_edit_changes_the_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git(root, "init", "-q", "-b", "main")
            (root / "a.txt").write_text("1\n", encoding="utf-8")
            _git(root, "add", "a.txt")
            first = _workspace_state(root)
            (root / "a.txt").write_text("2\n", encoding="utf-8")
            _git(root, "add", "a.txt")
            self.assertNotEqual(first, _workspace_state(root))


class RunArtifactTests(unittest.TestCase):
    """Fable Round 2: files a run itself creates, and generated artifacts, never fail a correct check."""

    def test_report_written_by_two_commands_of_one_run_is_not_a_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            cmds = ["echo '<testsuite tests=\"1\"/>' > junit.xml", "echo '<testsuite tests=\"2\"/>' > junit.xml"]
            res = run_commands(cmds, root, stop_on_failure=False)
            self.assertEqual([r.passed for r in res], [True, True])
            acc = run_acceptance(cmds, root)
            self.assertEqual([r["passed"] for r in acc], [True, True])
            self.assertFalse((root / "junit.xml").exists(), "run artifacts are still cleaned up")

    def test_pre_existing_untracked_coverage_file_is_a_generated_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / ".coverage").write_text("old\n", encoding="utf-8")
            res = run_commands(["echo new > .coverage"], root)[-1]
            self.assertTrue(res.passed)
            self.assertEqual(res.side_effects_restored, [".coverage"])


class InPlaceConcurrentEditTests(unittest.TestCase):
    """Fable Round 2: in the user's own tree, an unrelated file a person saves mid-check is not blamed."""

    def test_unrelated_untracked_edit_is_reported_but_does_not_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "notes.md").write_text("draft\n", encoding="utf-8")
            res = run_acceptance(["echo more >> notes.md"], root, restore_tracked=False,
                                 change_set=["pkg/core.py"])[-1]
            self.assertTrue(res["passed"])
            self.assertEqual(res["tracked_side_effects"], ["notes.md"])
            self.assertNotIn("untrusted_side_effects", res)

    def test_change_set_file_edit_still_fails_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "pkg" / "new_mod.py").write_text("X = 1\n", encoding="utf-8")  # created by the worker
            res = run_acceptance(["echo 'X = 2' >> pkg/new_mod.py"], root, restore_tracked=False,
                                 change_set=["pkg/new_mod.py"])[-1]
            self.assertFalse(res["passed"])
            self.assertEqual(res["untrusted_side_effects"], ["pkg/new_mod.py"])
            self.assertIn("Files changed while the check ran", res["output_tail"])


class CredentialReportingTests(unittest.TestCase):
    def test_url_with_empty_user_and_password_is_caught(self) -> None:
        found = verify_mod.secret_env_names({"REDIS_URL": "redis://:devpass@localhost:6379/0"})
        self.assertEqual(found, {"REDIS_URL"})

    def test_failed_check_names_the_removed_variables_without_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.dict(os.environ, _SECRETS):
            root, _ = _repo(tmp)
            res = run_commands(["exit 1"], root)[-1]
            self.assertIn("ACME_DB_PASSWORD", res.credential_env_removed)
            self.assertNotIn("dummy-db-secret", json.dumps(res.as_dict()))
            ok = run_commands(["true"], root)[-1]
            self.assertEqual(ok.credential_env_removed, [])


# ------------------------------------------------------------------------------------------------
# Round 3 critique findings (Gemini Pro Next)
# ------------------------------------------------------------------------------------------------


class LinkedWorktreeTests(unittest.TestCase):
    def test_check_poisoning_the_shared_git_dir_of_a_linked_worktree_fails_and_is_undone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            linked = Path(tmp) / "linked"
            _git(root, "worktree", "add", "-q", str(linked), "-b", "feature")
            self.assertTrue((linked / ".git").is_file())
            common_config = root / ".git" / "config"
            config_before = common_config.read_text(encoding="utf-8")
            cmd = ("git config core.fsmonitor 'sh -c true' && "
                   "printf '#!/bin/sh\\nexit 0\\n' > \"$(git rev-parse --git-common-dir)/hooks/pre-commit\"")
            res = run_commands([cmd], linked)[-1]
            self.assertFalse(res.passed)
            self.assertIn(".git-common/config", res.untrusted_side_effects)
            self.assertIn(".git-common/hooks/pre-commit", res.untrusted_side_effects)
            self.assertEqual(common_config.read_text(encoding="utf-8"), config_before)
            self.assertFalse((root / ".git" / "hooks" / "pre-commit").exists())


class SubdirectoryAndIndexTests(unittest.TestCase):
    def test_files_a_subdirectory_check_creates_beside_it_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "sub").mkdir()
            run_commands(["echo X > ../pkg/sneaky1.py"], root / "sub")
            run_acceptance(["echo X > ../pkg/sneaky2.py"], root / "sub")
            self.assertFalse((root / "pkg" / "sneaky1.py").exists())
            self.assertFalse((root / "pkg" / "sneaky2.py").exists())

    def test_change_set_given_relative_to_the_checkout_root_matches_from_a_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "sub").mkdir()
            (root / "pkg" / "new_mod.py").write_text("X = 1\n", encoding="utf-8")
            res = run_acceptance(["echo 'X = 2' >> ../pkg/new_mod.py"], root / "sub", restore_tracked=False,
                                 change_set=["pkg/new_mod.py"])[-1]
            self.assertFalse(res["passed"])
            self.assertIn("pkg/new_mod.py", res["untrusted_side_effects"])

    def test_index_only_changes_are_untrusted_in_both_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "pkg" / "core.py").write_text("def add(a, b):\n    return b + a\n", encoding="utf-8")
            res = run_commands(["git add pkg/core.py"], root)[-1]
            self.assertFalse(res.passed)
            self.assertIn(".git/index", res.untrusted_side_effects)
            staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True)
            self.assertEqual(staged.stdout.strip(), "", "the index is restored in a job copy")
            inplace = run_acceptance(["git rm -q --cached pkg/data.txt"], root, restore_tracked=False)[-1]
            self.assertFalse(inplace["passed"])
            self.assertIn(".git/index", inplace["untrusted_side_effects"])

    def test_allowed_target_replaced_by_a_directory_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "pkg" / "core.py").write_text("NEW = 1\n", encoding="utf-8")
            res = run_commands(["rm pkg/core.py && mkdir pkg/core.py"], root, allowed_mutations=("pkg/core.py",))[-1]
            self.assertFalse(res.passed)
            self.assertTrue((root / "pkg" / "core.py").is_file())
            self.assertEqual((root / "pkg" / "core.py").read_text(encoding="utf-8"), "NEW = 1\n")


class UnpricedCallAloneTests(unittest.TestCase):
    def test_priced_calls_wait_while_an_unpriced_call_is_in_flight(self) -> None:
        controls = BatchControls(resolve_batch_settings({"max_cost_cents": 5}), threading.Event())
        self.assertTrue(controls.before_call("genai"))
        controls.after_call("genai", WriterReply(text="", cost_cents=1.0))  # genai is priced now
        self.assertTrue(controls.before_call("claude"))                     # first Claude call: unpriced
        verdict: list = []
        waiter = threading.Thread(target=lambda: verdict.append(controls.before_call("genai")), daemon=True)
        waiter.start()
        waiter.join(0.3)
        self.assertTrue(waiter.is_alive(), "nothing may start while a call with no known price runs")
        controls.after_call("claude", WriterReply(text="", cost_cents=4.5))  # spent 5.5 >= 5
        waiter.join(3)
        self.assertEqual(verdict, [False])
        self.assertLessEqual(controls.spent_cents, 5.5)


class ApplyThroughInternalLinkTests(unittest.TestCase):
    def test_apply_refuses_a_path_through_a_symlinked_directory_inside_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)

            def worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                (wt / "hooks_alias").mkdir()
                (wt / "hooks_alias" / "pre-commit").write_text("#!/bin/sh\necho pwned\n", encoding="utf-8")
                return {"text": "done", "usage": {"prompt_token_count": 10, "candidates_token_count": 2,
                                                  "total_token_count": 12}, "tool_calls": [], "thoughts": []}

            jm = JobManager(workspace_root=root, store=store, worker_backend=worker)
            d = jm.delegate_start(prompt="add a helper script", model="gemini-3.8-flash", project="p", wait=True)
            os.symlink(".git/hooks", root / "hooks_alias")  # the host tree has an alias of .git/hooks
            with self.assertRaises(Exception):
                jm.delegate_apply(d["job_id"], allow_scope_violations=True)
            self.assertFalse((root / ".git" / "hooks" / "pre-commit").exists())
            jm.shutdown()


# ------------------------------------------------------------------------------------------------
# Round 3 critique findings (Claude Fable 5.1)
# ------------------------------------------------------------------------------------------------


class HeadMoveTests(unittest.TestCase):
    def test_check_that_commits_fails_and_head_is_restored_in_a_job_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
            (root / "pkg" / "core.py").write_text("def add(a, b):\n    return b + a\n", encoding="utf-8")
            cmd = "git add -A && git -c user.email=t@e -c user.name=T commit -qm 'made by a test'"
            res = run_commands([cmd], root)[-1]
            self.assertFalse(res.passed)
            self.assertIn(".git/HEAD", res.untrusted_side_effects)
            now = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
            self.assertEqual(now, head)
            self.assertEqual((root / "pkg" / "core.py").read_text(encoding="utf-8"), "def add(a, b):\n    return b + a\n")
            status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout
            self.assertEqual(status.strip(), "M pkg/core.py")

    def test_commit_in_the_users_tree_is_reported_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "pkg" / "m0.py").write_text("M = 0\n", encoding="utf-8")
            cmd = "git add -A && git -c user.email=t@e -c user.name=T commit -qm 'made by a test'"
            res = run_acceptance([cmd], root, restore_tracked=False, change_set=["pkg/m0.py"])[-1]
            self.assertFalse(res["passed"])
            self.assertIn(".git/HEAD", res["untrusted_side_effects"])


class InPlaceLinkedWorktreeTests(unittest.TestCase):
    def test_hook_planted_in_the_common_dir_is_removed_even_in_report_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            linked = Path(tmp) / "linked"
            _git(root, "worktree", "add", "-q", str(linked), "-b", "feature")
            cmd = "printf '#!/bin/sh\\nexit 0\\n' > \"$(git rev-parse --git-common-dir)/hooks/pre-commit\""
            res = run_acceptance([cmd], linked, restore_tracked=False)[-1]
            self.assertFalse(res["passed"])
            self.assertIn(".git-common/hooks/pre-commit", res["untrusted_side_effects"])
            self.assertFalse((root / ".git" / "hooks" / "pre-commit").exists())


class UnpricedFairnessTests(unittest.TestCase):
    def test_waiting_unpriced_call_is_not_overtaken_by_new_priced_calls(self) -> None:
        # before_call and after_call for one call run on the same thread (the task's ladder), as here.
        controls = BatchControls(resolve_batch_settings({"max_cost_cents": 100}), threading.Event())
        self.assertTrue(controls.before_call("genai"))
        controls.after_call("genai", WriterReply(text="", cost_cents=1.0))
        self.assertTrue(controls.before_call("genai"))          # one priced call in flight (main thread)
        order: list[str] = []
        claude_done = threading.Event()

        def claude_task() -> None:
            order.append(f"claude:{controls.before_call('claude')}")
            claude_done.wait(3)
            controls.after_call("claude", WriterReply(text="", cost_cents=5.0))

        claude = threading.Thread(target=claude_task, daemon=True)
        claude.start()
        time.sleep(0.2)                                          # Claude is now waiting for a solo slot
        genai = threading.Thread(target=lambda: order.append(f"genai:{controls.before_call('genai')}"), daemon=True)
        genai.start()
        time.sleep(0.2)
        self.assertEqual(order, [], "the new priced call must queue behind the waiting unpriced call")
        controls.after_call("genai", WriterReply(text="", cost_cents=1.0))
        deadline = time.time() + 3
        while order != ["claude:True"] and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(order, ["claude:True"])
        claude_done.set()
        claude.join(3)
        genai.join(3)
        self.assertEqual(order, ["claude:True", "genai:True"])


class InPlaceCleanupScopeTests(unittest.TestCase):
    def test_in_place_cleanup_removes_only_run_artifacts_under_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "svc").mkdir()
            cmd = ("mkdir -p .pytest_cache && echo c > .pytest_cache/v && echo r > junit.xml && "
                   "echo draft > draft.md && echo note > ../user_note.md")
            quarantine = Path(tmp) / "quarantine"
            res = run_acceptance([cmd], root / "svc", restore_tracked=False, quarantine_dir=quarantine)[-1]
            self.assertFalse((root / "svc" / ".pytest_cache" / "v").exists())
            self.assertFalse((root / "svc" / "junit.xml").exists())
            # A new non-artifact file may be a person's work: kept out of the diff, but moved, not deleted.
            self.assertFalse((root / "svc" / "draft.md").exists())
            run_dir = Path(res["quarantined_files"]["dir"])
            self.assertEqual(run_dir.parent, quarantine)
            self.assertEqual((run_dir / "svc" / "draft.md").read_text(encoding="utf-8"), "draft\n")
            self.assertEqual(res["quarantined_files"]["files"], ["svc/draft.md"])
            self.assertTrue((root / "user_note.md").exists(), "files outside the in-place workspace are left alone")


# ------------------------------------------------------------------------------------------------
# Round 4 critique findings (Gemini Pro Next and Claude Fable 5.1)
# ------------------------------------------------------------------------------------------------


class UnpricedTwoWaiterTests(unittest.TestCase):
    def test_two_waiters_for_one_unpriced_writer_do_not_deadlock(self) -> None:
        controls = BatchControls(resolve_batch_settings({"max_cost_cents": 1000}), threading.Event())
        self.assertTrue(controls.before_call("genai"))
        controls.after_call("genai", WriterReply(text="", cost_cents=5.0))
        self.assertTrue(controls.before_call("genai"))           # a priced call in flight (main thread)
        done: list[str] = []

        def claude_task(name: str) -> None:
            if controls.before_call("claude"):
                time.sleep(0.1)
                controls.after_call("claude", WriterReply(text="", cost_cents=7.0))
                done.append(name)

        def late_genai() -> None:
            time.sleep(0.4)
            if controls.before_call("genai"):
                controls.after_call("genai", WriterReply(text="", cost_cents=5.0))
                done.append("genai")

        threads = [threading.Thread(target=claude_task, args=(n,), daemon=True) for n in ("c1", "c2")]
        threads.append(threading.Thread(target=late_genai, daemon=True))
        for t in threads:
            t.start()
        time.sleep(0.3)
        controls.after_call("genai", WriterReply(text="", cost_cents=5.0))
        for t in threads:
            t.join(8)
        self.assertFalse(any(t.is_alive() for t in threads), f"stuck threads; done={done}")
        self.assertEqual(sorted(done), ["c1", "c2", "genai"])
        self.assertEqual(controls.unpriced_waiting, 0)


class GitStateTamperingTests(unittest.TestCase):
    def test_check_making_the_repo_bare_still_fails_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            res = run_commands(["echo BACKDOOR > pkg/data.txt && git config core.bare true"], root)[-1]
            self.assertFalse(res.passed)
            self.assertIn(".git/config", res.untrusted_side_effects)
            self.assertEqual((root / "pkg" / "data.txt").read_text(encoding="utf-8"), "original\n")
            bare = subprocess.run(["git", "config", "--get", "core.bare"], cwd=root, capture_output=True, text=True)
            self.assertEqual(bare.stdout.strip(), "false")

    def test_hooks_dir_replaced_by_a_symlink_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            elsewhere = Path(tmp) / "evil_hooks"
            elsewhere.mkdir()
            (elsewhere / "pre-commit").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            res = run_commands([f"rm -rf .git/hooks && ln -s {elsewhere} .git/hooks"], root)[-1]
            self.assertFalse(res.passed)
            self.assertIn(".git/hooks", res.untrusted_side_effects)
            self.assertFalse((root / ".git" / "hooks").is_symlink())
            self.assertTrue((elsewhere / "pre-commit").exists(), "the link target itself is never touched")

    def test_hook_planted_behind_a_read_only_hooks_dir_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            mode_before = (root / ".git" / "hooks").stat().st_mode & 0o7777
            cmd = "printf '#!/bin/sh\\nexit 0\\n' > .git/hooks/pre-commit && chmod 0555 .git/hooks"
            res = run_commands([cmd], root)[-1]
            self.assertFalse(res.passed)
            self.assertFalse((root / ".git" / "hooks" / "pre-commit").exists())
            self.assertEqual((root / ".git" / "hooks").stat().st_mode & 0o7777, mode_before)

    def test_check_starting_a_merge_fails_and_the_merge_state_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            _git(root, "checkout", "-q", "-b", "other")
            (root / "pkg" / "data.txt").write_text("other\n", encoding="utf-8")
            _git(root, "commit", "-qam", "other")
            _git(root, "checkout", "-q", "main")
            res = run_commands(["git -c user.email=t@e -c user.name=T merge --no-ff --no-commit other || true"],
                               root)[-1]
            self.assertFalse(res.passed)
            self.assertIn(".git/MERGE_HEAD", res.untrusted_side_effects)
            self.assertFalse((root / ".git" / "MERGE_HEAD").exists())
            self.assertEqual((root / "pkg" / "data.txt").read_text(encoding="utf-8"), "original\n")

    def test_common_dir_is_resolved_from_a_linked_worktree_subdirectory(self) -> None:
        # No --path-format: git before 2.31 echoes unknown options to stdout and still exits 0.
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            linked = Path(tmp) / "linked wt"
            _git(root, "worktree", "add", "-q", str(linked), "-b", "feature")
            layout = verify_mod._repo_layout(linked / "pkg")
            self.assertIsNotNone(layout)
            top, dirs = layout
            self.assertEqual(top.resolve(), linked.resolve())
            self.assertEqual(dirs[".git-common"].resolve(), (root / ".git").resolve())


class RevertThroughLinkedParentTests(unittest.TestCase):
    def test_revert_never_writes_through_a_parent_that_became_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            (root / "sub").mkdir()
            (root / "sub" / "pre-commit").write_text("#!/bin/sh\necho pwned\n", encoding="utf-8")
            os.chmod(root / "sub" / "pre-commit", 0o755)
            _git(root, "add", "-A")
            _git(root, "commit", "-qm", "script")

            def worker(req: dict) -> dict:
                (Path(req["workspace_dir"]) / "sub" / "pre-commit").unlink()  # the job deletes the script
                return {"text": "done", "usage": {"prompt_token_count": 10, "candidates_token_count": 2,
                                                  "total_token_count": 12}, "tool_calls": [], "thoughts": []}

            jm = JobManager(workspace_root=root, store=store, worker_backend=worker)
            d = jm.delegate_start(prompt="remove the script", model="gemini-3.8-flash", project="p", wait=True)
            jm.delegate_apply(d["job_id"])
            (root / "sub").rmdir()
            os.symlink(".git/hooks", root / "sub")  # later, sub becomes an alias of .git/hooks
            out = jm.delegate_revert(d["job_id"], skip_dirty=True)
            self.assertFalse(out.get("reverted"))
            self.assertFalse((root / ".git" / "hooks" / "pre-commit").exists())
            jm.shutdown()



# ------------------------------------------------------------------------------------------------
# Round 5 hardening notes (Gemini Pro Next)
# ------------------------------------------------------------------------------------------------


class Round5HardeningTests(unittest.TestCase):
    def test_run_commands_in_report_only_mode_keeps_files_outside_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / "svc").mkdir()
            run_commands(["echo note > ../sibling_note.md"], root / "svc", restore_tracked=False)
            self.assertTrue((root / "sibling_note.md").exists())

    def test_hooks_dir_created_by_a_check_is_removed_even_when_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            subprocess.run(["rm", "-rf", str(root / ".git" / "hooks")], check=True)
            cmd = "mkdir .git/hooks && printf '#!/bin/sh\\nexit 0\\n' > .git/hooks/pre-commit && chmod 0555 .git/hooks"
            res = run_commands([cmd], root)[-1]
            self.assertFalse(res.passed)
            self.assertFalse((root / ".git" / "hooks").exists())

    def test_check_editing_info_exclude_is_undone_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            exclude = root / ".git" / "info" / "exclude"
            before = exclude.read_text(encoding="utf-8") if exclude.exists() else None
            res = run_commands(["mkdir -p .git/info && echo 'conftest.py' >> .git/info/exclude && echo x > conftest.py"],
                               root)[-1]
            self.assertFalse(res.passed)
            self.assertIn(".git/info/exclude", res.untrusted_side_effects)
            after = exclude.read_text(encoding="utf-8") if exclude.exists() else None
            self.assertEqual(after, before)
            self.assertFalse((root / "conftest.py").exists(), "the file it tried to hide is cleaned up")


# ------------------------------------------------------------------------------------------------
# Round 6 critique findings (Gemini Pro Next, Claude Fable 5.1)
# ------------------------------------------------------------------------------------------------


class QuarantineSafetyTests(unittest.TestCase):
    def test_links_are_deleted_never_quarantined_and_runs_never_collide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            quarantine = Path(tmp) / "q"
            first = run_acceptance([f"ln -s {root / '.git' / 'hooks'} pre-commit && echo v1 > draft.md"], root,
                                   restore_tracked=False, quarantine_dir=quarantine)[-1]
            second = run_acceptance(["echo v2 > draft.md && echo payload > pre-commit"], root,
                                    restore_tracked=False, quarantine_dir=quarantine)[-1]
            self.assertFalse((root / ".git" / "hooks" / "pre-commit").exists(), "nothing was written through a link")
            dirs = {Path(first["quarantined_files"]["dir"]), Path(second["quarantined_files"]["dir"])}
            self.assertEqual(len(dirs), 2, "each run gets its own quarantine directory")
            contents = sorted((d / "draft.md").read_text(encoding="utf-8") for d in dirs)
            self.assertEqual(contents, ["v1\n", "v2\n"])
            links = [p for p in quarantine.rglob("*") if p.is_symlink()]
            self.assertEqual(links, [])

    def test_read_only_directory_created_by_a_check_does_not_crash_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            res = run_commands(["mkdir ro_dir && echo x > ro_dir/f.txt && chmod 0555 ro_dir"], root)
            self.assertEqual(len(res), 1)
            self.assertFalse((root / "ro_dir" / "f.txt").exists())


class PlantedGitConfigTests(unittest.TestCase):
    """A worker or check can rewrite the checkout's git config; the plugin's own git calls must not then
    run the planted command with the plugin's credentials in its environment."""

    def _planted_repo(self, tmp: str) -> tuple[Path, Path]:
        root, _ = _repo(tmp)
        marker = Path(tmp) / "fsmonitor-ran.txt"
        hook = Path(tmp) / "fsmonitor.sh"
        hook.write_text(f'#!/bin/sh\necho "ran key=${{GEMINI_API_KEY:-unset}}" >> "{marker}"\nexit 1\n',
                        encoding="utf-8")
        hook.chmod(0o755)
        _git(root, "config", "core.fsmonitor", str(hook))
        subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, check=False)
        if not marker.exists():
            self.skipTest("this git does not run core.fsmonitor")
        marker.unlink()
        return root, marker

    def test_plugin_git_helpers_do_not_run_a_planted_fsmonitor(self) -> None:
        from server import batch_plan, jobs, repo_map

        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.dict(os.environ, _SECRETS):
            root, marker = self._planted_repo(tmp)
            (root / "new.txt").write_text("x\n", encoding="utf-8")
            jobs._snapshot_untracked_hashes(root)
            jobs._is_tracked_in_git(root, "pkg/core.py")
            jobs._is_gitignored_on_host(root, "new.txt")
            jobs._list_tracked_repo_files(root)
            self.assertIn("new.txt", jobs.capture_live_workspace_diff(root))
            batch_plan._git_ignored(root, "new.txt")
            repo_map._git_info(root)
            repo_map.repo_map_cache_key(root)
            ran = marker.read_text(encoding="utf-8") if marker.exists() else ""
            self.assertEqual(ran, "", "a planted core.fsmonitor ran inside a plugin git call")

    def test_every_git_subprocess_in_server_is_guarded(self) -> None:
        import ast

        server_dir = Path(__file__).resolve().parents[1] / "server"
        guard_names = {"_GIT_SAFE_ARGS", "_GIT_EXEC_GUARD"}
        problems: list[str] = []
        for path in sorted(server_dir.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for node in ast.walk(tree):
                if not (isinstance(node, (ast.List, ast.Tuple)) and node.elts
                        and isinstance(node.elts[0], ast.Constant) and node.elts[0].value == "git"):
                    continue
                consts = {e.value for e in node.elts if isinstance(e, ast.Constant)}
                starred = {e.value.id for e in node.elts
                           if isinstance(e, ast.Starred) and isinstance(e.value, ast.Name)}
                guarded = bool(starred & guard_names) or {"core.hooksPath=/dev/null", "core.fsmonitor=false"} <= consts
                call = parents.get(node)
                has_env = isinstance(call, ast.Call) and any(k.arg == "env" for k in call.keywords)
                if not (guarded and has_env):
                    problems.append(f"{path.name}:{node.lineno}")
        self.assertEqual(problems, [], "git argv built outside verify.run_git / jobs._run_git")


class Round8CleanupTests(unittest.TestCase):
    def test_quarantine_moves_a_file_out_of_a_read_only_directory(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            baseline = verify_mod.untracked_baseline(root)
            (root / "ro").mkdir()
            (root / "ro" / "f.txt").write_text("keep me\n", encoding="utf-8")
            (root / "ro").chmod(0o555)
            report: dict = {}
            try:
                verify_mod.remove_new_untracked(root, baseline, whole_checkout=False,
                                                quarantine_dir=Path(tmp) / "q", report=report)
            finally:
                (root / "ro").chmod(0o755)
            self.assertEqual((report["moved"], report["left_in_place"]), (["ro/f.txt"], []))
            self.assertFalse((root / "ro" / "f.txt").exists())
            saved = Path(report["quarantine_dir"]) / "ro" / "f.txt"
            self.assertEqual(saved.read_text(encoding="utf-8"), "keep me\n")

    def test_no_empty_quarantine_directory_when_nothing_could_be_moved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            baseline = verify_mod.untracked_baseline(root)
            (root / "drafts").mkdir()
            (root / "drafts" / "work.txt").write_text("draft\n", encoding="utf-8")
            quarantine = Path(tmp) / "q"
            report: dict = {}
            with unittest.mock.patch.object(verify_mod.shutil, "move", side_effect=OSError("disk full")):
                verify_mod.remove_new_untracked(root, baseline, whole_checkout=False,
                                                quarantine_dir=quarantine, report=report)
            self.assertEqual(report["left_in_place"], ["drafts/work.txt"])
            self.assertIsNone(report["quarantine_dir"])
            self.assertTrue((root / "drafts" / "work.txt").exists(), "a file that cannot be moved stays put")
            self.assertEqual(list(quarantine.rglob("*")) if quarantine.exists() else [], [])

    def test_cleanup_without_a_baseline_handles_read_only_directories(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            try:
                with unittest.mock.patch.object(verify_mod, "untracked_baseline", return_value=None):
                    run_commands(["mkdir ro_dir && echo x > ro_dir/f.txt && chmod 0555 ro_dir"], root)
                    run_acceptance(["mkdir ro_acc && echo x > ro_acc/f.txt && chmod 0555 ro_acc"], root)
            finally:
                for name in ("ro_dir", "ro_acc"):
                    if (root / name).is_dir():
                        (root / name).chmod(0o755)
            self.assertFalse((root / "ro_dir" / "f.txt").exists())
            self.assertFalse((root / "ro_acc" / "f.txt").exists())


class DelegateApplyIncludePathsTests(unittest.TestCase):
    def test_include_paths_reaches_delegate_apply_instead_of_a_name_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, _ = _server(root, store, _writer_factory({}))
            for value in (["extra.csv"], "extra.csv"):
                resp = srv.handle_Message({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                           "params": {"name": "delegate_apply",
                                                      "arguments": {"job_id": "job-missing", "include_paths": value}}})
                text = json.dumps(resp)
                self.assertNotIn("_require_str_list", text)
                self.assertNotIn("NameError", text)


class Round9GitConfigAndExdevTests(unittest.TestCase):
    def test_planted_diff_external_textconv_and_filters_even_via_include_do_not_execute_or_spoof_diffs(self) -> None:
        from server import assist_tools, batch_runner, jobs

        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.dict(os.environ, _SECRETS):
            root, _ = _repo(tmp)
            marker = Path(tmp) / "markers.log"
            scripts: dict[str, Path] = {}
            for kind, extra in (
                ("diff_external", 'echo "SPOOFED_DIFF"\nexit 0'),
                ("diff_cmd", 'echo "SPOOFED_DRV"\nexit 0'),
                ("textconv", "cat"),
                ("filter_clean", "cat"),
                ("filter_smudge", "cat"),
                ("filter_eq", "cat"),
            ):
                script = Path(tmp) / f"{kind}.sh"
                script.write_text(f'#!/bin/sh\necho "{kind}" >> "{marker}"\n{extra}\n', encoding="utf-8")
                script.chmod(0o755)
                scripts[kind] = script

            (root / ".gitattributes").write_text(
                "*.py filter=evil diff=evildiff\neq.txt filter=with=eq\n", encoding="utf-8"
            )
            (root / "eq.txt").write_text("base\n", encoding="utf-8")
            _git(root, "add", ".gitattributes", "eq.txt")
            _git(root, "commit", "-qm", "attrs")

            evil_cfg = Path(tmp) / "included_evil.cfg"
            evil_cfg.write_text(
                f'[diff]\n\texternal = {scripts["diff_external"]}\n'
                f'[diff "evildiff"]\n\tcommand = {scripts["diff_cmd"]}\n\ttextconv = {scripts["textconv"]}\n'
                f'[filter "evil"]\n\tclean = {scripts["filter_clean"]}\n\tsmudge = {scripts["filter_smudge"]}\n'
                f'[filter "with=eq"]\n\tclean = {scripts["filter_eq"]}\n',
                encoding="utf-8",
            )
            with open(root / ".git" / "config", "a", encoding="utf-8") as cfg:
                cfg.write(f"\n[include]\n\tpath = {evil_cfg}\n")

            (root / "pkg" / "core.py").write_text("def add(a, b):\n    return a + b + 1\n", encoding="utf-8")
            (root / "eq.txt").write_text("changed\n", encoding="utf-8")

            snap = verify_mod.snapshot_workspace(root)
            self.assertIsNotNone(snap)
            live_diff = jobs.capture_live_workspace_diff(root)
            self.assertIn("@@", live_diff)
            self.assertNotIn("SPOOFED", live_diff)
            raw_diff = jobs._run_git(["diff", "--binary", "--no-renames", "HEAD"], cwd=root, check=False).stdout
            self.assertIn("@@", raw_diff)
            self.assertNotIn("SPOOFED", raw_diff)
            batch_runner._workspace_state(root)
            plan, shared, _ = assist_tools.prepare_replication(
                root, example_path="pkg/core.py", targets=["pkg/data.txt"]
            )
            self.assertEqual(len(plan.tasks), 1)
            self.assertIn("@@", shared)
            self.assertNotIn("SPOOFED", shared)

            fired = marker.read_text(encoding="utf-8").splitlines() if marker.exists() else []
            self.assertEqual(fired, [], f"planted git drivers/filters executed: {fired}")

    def test_isolated_worker_poisoning_worktree_git_config_cannot_execute_or_spoof_get_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            marker = Path(tmp) / "worker_pwn.log"
            ext_script = Path(tmp) / "spoof_diff.sh"
            ext_script.write_text(f'#!/bin/sh\necho diff_ext >> "{marker}"\necho "SPOOFED_DIFF"\nexit 0\n',
                                  encoding="utf-8")
            ext_script.chmod(0o755)
            clean_script = Path(tmp) / "clean_pwn.sh"
            clean_script.write_text(f'#!/bin/sh\necho filter_clean >> "{marker}"\ncat\n', encoding="utf-8")
            clean_script.chmod(0o755)

            def poisoning_worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                with open(wt / ".git" / "config", "a", encoding="utf-8") as cfg:
                    cfg.write(
                        f'\n[diff]\n\texternal = {ext_script}\n'
                        f'[filter "pwn"]\n\tclean = {clean_script}\n'
                    )
                (wt / ".git" / "info").mkdir(parents=True, exist_ok=True)
                (wt / ".git" / "info" / "attributes").write_text("* filter=pwn\n", encoding="utf-8")
                (wt / "pkg" / "core.py").write_text("def add(a, b):\n    return (a + b)\n", encoding="utf-8")
                return {"text": "done", "usage": {"prompt_token_count": 10, "total_token_count": 12}}

            jm = JobManager(workspace_root=root, store=store, worker_backend=poisoning_worker)
            digest = jm.delegate_start("Update pkg/core.py", project="p", allowed_paths=["pkg/core.py"], wait=True)
            self.assertEqual(digest["status"], "ok")
            diff_payload = jm.get_diff(digest["job_id"])
            self.assertNotIn("SPOOFED_DIFF", diff_payload["diff"])
            self.assertIn("pkg/core.py", diff_payload["diff"])
            fired = marker.read_text(encoding="utf-8").splitlines() if marker.exists() else []
            self.assertEqual(fired, [])

    def test_restore_directory_tree_removes_planted_symlinks_without_writing_through_them(self) -> None:
        from server import jobs

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            victim = base / "victim.txt"
            victim.write_text("DO_NOT_OVERWRITE", encoding="utf-8")

            cfg = base / "config"
            cfg.write_text("[core]\n\tbare = false\n", encoding="utf-8")
            cfg_snap = jobs._snapshot_directory_tree(cfg)
            cfg.unlink()
            cfg.symlink_to(victim)
            self.assertTrue(jobs._restore_directory_tree(cfg, cfg_snap))
            self.assertFalse(cfg.is_symlink())
            self.assertEqual(cfg.read_text(encoding="utf-8"), "[core]\n\tbare = false\n")
            self.assertEqual(victim.read_text(encoding="utf-8"), "DO_NOT_OVERWRITE")

            hooks = base / "hooks"
            hooks.mkdir()
            hooks_snap = jobs._snapshot_directory_tree(hooks)
            (hooks / "post-checkout").symlink_to(victim)
            self.assertTrue(jobs._restore_directory_tree(hooks, hooks_snap))
            self.assertFalse((hooks / "post-checkout").exists())
            self.assertFalse((hooks / "post-checkout").is_symlink())
            self.assertEqual(victim.read_text(encoding="utf-8"), "DO_NOT_OVERWRITE")

    def test_cross_filesystem_quarantine_moves_0444_file_out_of_0555_directory_and_leaves_no_orphan_run_dir(self) -> None:
        import errno

        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            baseline = verify_mod.untracked_baseline(root)
            mod_dir = root / "modcache" / "pkg"
            mod_dir.mkdir(parents=True)
            mod_file = mod_dir / "mod.go"
            mod_file.write_text("package mod\n", encoding="utf-8")
            mod_file.chmod(0o444)
            mod_dir.chmod(0o555)
            quarantine = Path(tmp) / "cross_fs_q"
            report: dict = {}

            def exdev_rename(src, dst, *a, **kw):
                raise OSError(errno.EXDEV, "Invalid cross-device link")

            try:
                with unittest.mock.patch.object(verify_mod.shutil.os, "rename", side_effect=exdev_rename):
                    verify_mod.remove_new_untracked(root, baseline, whole_checkout=False,
                                                    quarantine_dir=quarantine, report=report)
            finally:
                if mod_dir.exists():
                    mod_dir.chmod(0o755)
            self.assertEqual((report["moved"], report["left_in_place"]), (["modcache/pkg/mod.go"], []))
            self.assertFalse(mod_file.exists())
            saved = Path(report["quarantine_dir"]) / "modcache" / "pkg" / "mod.go"
            self.assertEqual(saved.read_text(encoding="utf-8"), "package mod\n")

            # Also verify that if unlink fails with EPERM after EXDEV copy2 wrote a 0444 dest, no orphan run-* remains.
            mod_dir.mkdir(parents=True, exist_ok=True)
            stuck = mod_dir / "stuck.go"
            stuck.write_text("package stuck\n", encoding="utf-8")
            stuck.chmod(0o444)
            quarantine2 = Path(tmp) / "cross_fs_q_fail"
            report2: dict = {}
            real_unlink = os.unlink

            def eperm_source_unlink(p, *a, **kw):
                if Path(p).resolve() == stuck.resolve():
                    raise OSError(errno.EPERM, "Operation not permitted")
                return real_unlink(p, *a, **kw)

            with (
                unittest.mock.patch.object(verify_mod.shutil.os, "rename", side_effect=exdev_rename),
                unittest.mock.patch.object(verify_mod.shutil.os, "unlink", side_effect=eperm_source_unlink),
            ):
                verify_mod.remove_new_untracked(root, baseline, whole_checkout=False,
                                                quarantine_dir=quarantine2, report=report2)
            self.assertEqual(report2["left_in_place"], ["modcache/pkg/stuck.go"])
            self.assertIsNone(report2["quarantine_dir"])
            self.assertTrue(stuck.exists())
            self.assertEqual(list(quarantine2.rglob("*")) if quarantine2.exists() else [], [])


class Round10GitRestoreAndRootModeTests(unittest.TestCase):
    def test_host_git_restore_does_not_short_circuit_and_survives_readonly_or_deleted_metadata(self) -> None:
        from server import jobs

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # 1. Directory replaced by symlink, file, deleted, 0555 readonly, 0111 exec-only, and mode-only tamper
            for case in ("symlink", "file", "deleted", "readonly", "execonly", "chmod_hook"):
                hooks = base / f"hooks_{case}"
                hooks.mkdir()
                sample = hooks / "pre-commit"
                sample.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                sample.chmod(0o644 if case == "chmod_hook" else 0o755)
                snap = jobs._snapshot_directory_tree(hooks)
                if case == "symlink":
                    victim = base / f"victim_{case}"
                    victim.write_text("KEEP", encoding="utf-8")
                    subprocess.run(["rm", "-rf", str(hooks)], check=True)
                    hooks.symlink_to(victim)
                elif case == "file":
                    subprocess.run(["rm", "-rf", str(hooks)], check=True)
                    hooks.write_text("not a dir\n", encoding="utf-8")
                    hooks.chmod(0o444)
                elif case == "deleted":
                    subprocess.run(["rm", "-rf", str(hooks)], check=True)
                elif case == "readonly":
                    p = hooks / "planted"
                    p.write_text("bad\n", encoding="utf-8")
                    p.chmod(0o444)
                    hooks.chmod(0o555)
                elif case == "execonly":
                    (hooks / "planted").write_text("bad\n", encoding="utf-8")
                    hooks.chmod(0o111)
                elif case == "chmod_hook":
                    sample.chmod(0o755)
                self.assertTrue(jobs._restore_directory_tree(hooks, snap), case)
                self.assertTrue(hooks.is_dir() and not hooks.is_symlink(), case)
                self.assertEqual((hooks / "pre-commit").read_text(encoding="utf-8"), "#!/bin/sh\nexit 0\n")
                expected_m = 0o644 if case == "chmod_hook" else 0o755
                self.assertEqual((hooks / "pre-commit").stat().st_mode & 0o7777, expected_m, case)
                self.assertFalse((hooks / "planted").exists(), case)

            # Pre-existing legitimate symlink (e.g. Husky hooks dir) is preserved when untouched
            shared_hooks = base / "shared_hooks"
            shared_hooks.mkdir()
            link_hooks = base / "husky_link"
            link_hooks.symlink_to(shared_hooks)
            link_snap = jobs._snapshot_directory_tree(link_hooks)
            self.assertFalse(jobs._restore_directory_tree(link_hooks, link_snap))
            self.assertTrue(link_hooks.is_symlink() and os.readlink(link_hooks) == str(shared_hooks))

            cfg_ro = base / "config_ro"
            cfg_ro.write_text("[core]\n\tbare = false\n", encoding="utf-8")
            cfg_snap = jobs._snapshot_directory_tree(cfg_ro)
            cfg_ro.write_text("[core]\n\tfsmonitor = evil\n", encoding="utf-8")
            cfg_ro.chmod(0o444)
            self.assertTrue(jobs._restore_directory_tree(cfg_ro, cfg_snap))
            self.assertEqual(cfg_ro.read_text(encoding="utf-8"), "[core]\n\tbare = false\n")

            # 2. End-to-end delegate_start tampering with hooks + config + info + 0555/0444 modes simultaneously
            root, store = _repo(tmp)
            orig_cfg = (root / ".git" / "config").read_text(encoding="utf-8")

            def multi_tamper_worker(_req: dict) -> dict:
                h_git = root / ".git"
                (h_git / "hooks").mkdir(parents=True, exist_ok=True)
                (h_git / "hooks" / "post-commit").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
                (h_git / "hooks").chmod(0o555)
                (h_git / "config").write_text(orig_cfg + "\n[core]\n\tfsmonitor = /tmp/evil\n", encoding="utf-8")
                (h_git / "config").chmod(0o444)
                (h_git / "info").mkdir(parents=True, exist_ok=True)
                (h_git / "info" / "attributes").write_text("* filter=evil\n", encoding="utf-8")
                (h_git / "info").chmod(0o555)
                h_git.chmod(0o555)
                return {"text": "tampered", "usage": {"prompt_token_count": 5, "total_token_count": 6}}

            jm = JobManager(workspace_root=root, store=store, worker_backend=multi_tamper_worker)
            digest = jm.delegate_start("Attempt multi-target host git tamper", project="p", wait=True)
            self.assertEqual(digest["status"], "security_violation")
            self.assertFalse((root / ".git" / "hooks" / "post-commit").exists())
            self.assertEqual((root / ".git" / "config").read_text(encoding="utf-8"), orig_cfg)
            self.assertFalse((root / ".git" / "info" / "attributes").exists())
            jm.shutdown()

            # 3. Isolated worktree worker chmodding wt/.git/config 0444 is silently restored without breaking job
            def wt_ro_worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                wt_cfg = wt / ".git" / "config"
                wt_cfg.write_text(wt_cfg.read_text(encoding="utf-8") + "\n[diff]\n\texternal = /bin/true\n", encoding="utf-8")
                wt_cfg.chmod(0o444)
                (wt / "pkg" / "data.txt").write_text("updated\n", encoding="utf-8")
                return {"text": "ok", "usage": {"prompt_token_count": 5, "total_token_count": 6}}

            jm_wt = JobManager(workspace_root=root, store=store, worker_backend=wt_ro_worker)
            wt_digest = jm_wt.delegate_start("Edit data.txt", project="p", allowed_paths=["pkg/data.txt"], wait=True)
            self.assertEqual(wt_digest["status"], "ok")
            jm_wt.shutdown()

    def test_verify_check_git_metadata_restores_readonly_git_info_and_root_0555_cleanup(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            res = run_commands(
                ["mkdir -p .git/info && echo '* filter=pwn' > .git/info/attributes && chmod 0555 .git/info"],
                root,
            )
            self.assertFalse(res[0].passed)
            self.assertFalse((root / ".git" / "info" / "attributes").exists())

            victim_info = Path(tmp) / "victim_info"
            victim_info.mkdir()
            (victim_info / "exclude").write_text("KEEP_ME\n", encoding="utf-8")
            res_link = run_commands([f"rm -rf .git/info && ln -s {victim_info} .git/info"], root)
            self.assertFalse(res_link[0].passed)
            self.assertFalse((root / ".git" / "info").is_symlink())
            self.assertEqual((victim_info / "exclude").read_text(encoding="utf-8"), "KEEP_ME\n")

            # Checkout root itself chmodded 0555 after creating a root-level untracked file
            baseline = verify_mod.untracked_baseline(root)
            (root / "at_root.txt").write_text("stray\n", encoding="utf-8")
            root.chmod(0o555)
            try:
                removed = verify_mod.remove_new_untracked(root, baseline, whole_checkout=True)
            finally:
                root.chmod(0o755)
            self.assertEqual(removed, ["at_root.txt"])
            self.assertFalse((root / "at_root.txt").exists())

            # Non-git directory itself chmodded 0555 during run_commands and run_acceptance
            nongit = Path(tmp) / "nongit"
            nongit.mkdir()
            try:
                run_commands(["echo x > stray.txt && chmod 0555 ."], nongit)
                self.assertFalse((nongit / "stray.txt").exists())
                run_acceptance(["echo y > stray2.txt && chmod 0555 ."], nongit)
                self.assertFalse((nongit / "stray2.txt").exists())
            finally:
                nongit.chmod(0o755)


class Round11LogDirSymlinkAndNonGitTests(unittest.TestCase):
    def test_run_commands_preserves_own_log_path_when_log_dir_is_inside_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # 1. Non-git cwd with log_dir inside cwd
            nongit = base / "nongit"
            nongit.mkdir()
            res_ng = run_commands(["echo nongit_out && echo leak > stray.log"], nongit, log_dir=nongit / "logs")
            self.assertTrue(res_ng[0].passed)
            self.assertTrue(Path(res_ng[0].log_path).is_file())
            self.assertIn("nongit_out", Path(res_ng[0].log_path).read_text(encoding="utf-8"))
            self.assertFalse((nongit / "stray.log").exists())

            # 2. Git cwd (clean HOME with no global gitignore) for both restore_tracked=True and False
            clean_home = base / "clean_home"
            clean_home.mkdir()
            with unittest.mock.patch.dict(os.environ, {"HOME": str(clean_home), "XDG_CONFIG_HOME": str(clean_home)}):
                for mode_flag in (True, False):
                    repo, _ = _repo(str(base / f"g_{mode_flag}"))
                    res_g = run_commands(
                        ["echo git_out && echo artifact > other.log"],
                        repo,
                        log_dir=repo / "logs",
                        restore_tracked=mode_flag,
                    )
                    self.assertTrue(res_g[0].passed)
                    self.assertTrue(Path(res_g[0].log_path).is_file())
                    self.assertFalse((repo / "other.log").exists())

    def test_verify_preserves_and_restores_preexisting_symlinked_git_hooks_and_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            evil_hooks = base / "evil_hooks"
            evil_hooks.mkdir()
            (evil_hooks / "post-checkout").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            (evil_hooks / "post-checkout").chmod(0o755)

            # 2a: .git/hooks starts as symlink ../.githooks; check retargets to evil_hooks
            repo_a, _ = _repo(str(base / "a"))
            (repo_a / ".githooks").mkdir()
            _git(repo_a, "add", ".githooks")
            hooks_a = repo_a / ".git" / "hooks"
            verify_mod._rmtree_writable(hooks_a)
            os.symlink("../.githooks", hooks_a)
            res_a = run_commands([f"ln -sfn {evil_hooks} .git/hooks"], repo_a, restore_tracked=True)
            self.assertFalse(res_a[0].passed)
            self.assertIn(".git/hooks", res_a[0].untrusted_side_effects)
            self.assertTrue(hooks_a.is_symlink())
            self.assertEqual(os.readlink(hooks_a), "../.githooks")

            # 2b: .git/hooks starts as symlink; check replaces with real dir + hook (even with restore_tracked=False)
            repo_b, _ = _repo(str(base / "b"))
            (repo_b / ".githooks").mkdir()
            hooks_b = repo_b / ".git" / "hooks"
            verify_mod._rmtree_writable(hooks_b)
            os.symlink("../.githooks", hooks_b)
            res_b = run_commands(
                ["rm .git/hooks && mkdir .git/hooks && printf '#!/bin/sh\n' > .git/hooks/post-commit && chmod +x .git/hooks/post-commit"],
                repo_b,
                restore_tracked=False,
            )
            self.assertFalse(res_b[0].passed)
            self.assertIn(".git/hooks", res_b[0].untrusted_side_effects)
            self.assertTrue(hooks_b.is_symlink())
            self.assertEqual(os.readlink(hooks_b), "../.githooks")

            # 2c: .git/info starts as symlink; check retargets to evil_info with attributes filter
            evil_info = base / "evil_info"
            evil_info.mkdir()
            (evil_info / "attributes").write_text("* filter=evil\n", encoding="utf-8")
            legit_info = base / "legit_info"
            legit_info.mkdir()
            repo_c, _ = _repo(str(base / "c"))
            info_c = repo_c / ".git" / "info"
            verify_mod._rmtree_writable(info_c)
            os.symlink(str(legit_info), info_c)
            res_c = run_commands([f"ln -sfn {evil_info} .git/info"], repo_c, restore_tracked=True)
            self.assertFalse(res_c[0].passed)
            self.assertIn(".git/info", res_c[0].untrusted_side_effects)
            self.assertTrue(info_c.is_symlink())
            self.assertEqual(os.readlink(info_c), str(legit_info))

            # 2d: writing THROUGH a pre-existing .git/hooks or .git/info symlink without retargeting is flagged
            res_d = run_commands(
                ["printf '* filter=evil\n' > .git/info/attributes"],
                repo_c,
                restore_tracked=True,
            )
            self.assertFalse(res_d[0].passed)
            self.assertIn(".git/info", res_d[0].untrusted_side_effects)
            import server.jobs as _jobs_mod
            snap_d = _jobs_mod._snapshot_directory_tree(info_c)
            self.assertFalse(_jobs_mod._restore_directory_tree(info_c, snap_d))
            (legit_info / "exclude").write_text("*.secret\n", encoding="utf-8")
            self.assertTrue(_jobs_mod._restore_directory_tree(info_c, snap_d))

    def test_nongit_cleanup_unlocks_unreadable_subdirs_without_aborting_sibling_cleanup(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ng1 = base / "ng1"
            ng1.mkdir()
            run_commands(
                ["mkdir locked && echo x > locked/in.txt && chmod 0111 locked && echo y > stray.txt"],
                ng1,
                restore_tracked=True,
            )
            self.assertFalse((ng1 / "stray.txt").exists())
            self.assertFalse((ng1 / "locked" / "in.txt").exists())

            ng2 = base / "ng2"
            ng2.mkdir()
            run_acceptance(
                ["mkdir locked && echo x > locked/in.txt && chmod 0000 locked && echo y > stray.txt"],
                ng2,
                restore_tracked=True,
            )
            self.assertFalse((ng2 / "stray.txt").exists())
            self.assertFalse((ng2 / "locked" / "in.txt").exists())
