"""Regression tests for the round-1 critic findings on the batch writer path."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import threading
import time
import unittest
import unittest.mock

from server.batch_plan import BatchPlanError, FileTask, is_protected_path, validate_batch_plan
from server.jobs import JobManager
from server.mcp_stdio import MCPServer
from server.model_routes import resolve_batch_settings
from server.receipts import ReceiptStore
from server.verify import run_commands
from server.writers import (
    WriterConfigError,
    WriterReply,
    WriterTransportError,
    apply_writer_answer,
    make_claude_writer,
    run_writer_ladder,
)


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
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root, ReceiptStore(Path(tmp) / "data", workspace_root=root)


def _content_writer(contents: dict[str, list[str]], calls: list | None = None, *, delay: float = 0.0):
    """Factory whose writers answer with scripted whole-file contents per path (last one repeats)."""
    def factory(name, settings, auth):
        def write(req):
            if calls is not None:
                calls.append((name, req.task.task_id, req.prompt))
            if delay:
                time.sleep(delay)
            queue = contents[req.task.path]
            body = queue.pop(0) if len(queue) > 1 else queue[0]
            return WriterReply(text=json.dumps({"path": req.task.path, "content": body}), cost_cents=0.5,
                               usage={"prompt_token_count": 10, "total_token_count": 12})
        return write
    return factory


def _server(root, store, factory):
    jm = JobManager(workspace_root=root, store=store, worker_backend=lambda _r: {"text": "unused"})
    return MCPServer(job_manager=jm, store=store, batch_writer_factory=factory), jm


def _call(srv, name, args, *, msg_id=1, meta=None):
    params = {"name": name, "arguments": args}
    if meta:
        params["_meta"] = meta
    resp = srv.handle_Message({"jsonrpc": "2.0", "id": msg_id, "method": "tools/call", "params": params})
    return json.loads(resp["result"]["content"][0]["text"])


def _task(tid, path, **kw):
    return {"task_id": tid, "path": path, "action": kw.pop("action", "create"), "intent": f"Write {path}.", **kw}


class RunnerHardeningTests(unittest.TestCase):
    def test_crashing_writer_fails_the_task_and_never_reports_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)

            def factory(name, settings, auth):
                def write(req):
                    raise RuntimeError("400 INVALID_ARGUMENT: model not found")
                return write

            srv, jm = _server(root, store, factory)
            out = _call(srv, "delegate_file_batch", {"tasks": [_task("a", "pkg/a.py")], "project": "p"})
            self.assertEqual(out["status"], "checkpoint")
            self.assertEqual(out["tasks"][0]["state"], "failed")
            self.assertIn("RuntimeError", out["tasks"][0]["check_tail"])
            jm.shutdown()

    def test_checks_never_see_another_tasks_unverified_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            factory = _content_writer({"pkg/a.py": ["A = 1\n"], "pkg/b.py": ["def broken(:\n", "B = 2\n"]}, delay=0.05)
            srv, jm = _server(root, store, factory)
            tasks = [_task("a", "pkg/a.py", checks=["python3 -m compileall -q pkg"]),
                     _task("b", "pkg/b.py", checks=["python3 -m compileall -q pkg"])]
            out = _call(srv, "delegate_file_batch", {"tasks": tasks, "project": "p", "max_parallel": 2, "claude_writer": False})
            rows = {t["task_id"]: (t["state"], t["attempts"]) for t in out["tasks"]}
            self.assertEqual(rows["a"], ("landed", 1))
            self.assertEqual(rows["b"], ("landed", 2))
            jm.shutdown()

    def test_job_copy_never_left_without_a_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = _server(root, store, _content_writer({"pkg/a.py": ["A = 1\n"]}))
            with unittest.mock.patch("server.batch_runner.FileBatchRunner._execute", side_effect=OSError("disk full")):
                out = _call(srv, "delegate_file_batch", {"tasks": [_task("a", "pkg/a.py")], "project": "p"})
            self.assertEqual(out["status"], "error")
            rec = store.read_receipt(out["job_id"])
            self.assertEqual(rec["status"], "error")
            self.assertIn("disk full", rec["error"])
            jm.shutdown()

    def test_tracked_file_changed_by_a_check_fails_the_check_and_is_restored(self) -> None:
        # A check that edits another tracked file used to leave that edit in the job copy (and so in
        # the diff, where only allowed_paths could catch it). Now the edit is undone and reported, and
        # the check fails: it did not test the files that will ship.
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            (root / "tests_x").mkdir()
            (root / "tests_x" / "__init__.py").write_text("", encoding="utf-8")
            (root / "tests_x" / "test_sneaky.py").write_text(
                "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                "        open('pkg/core.py', 'a').write('# touched\\n')\n", encoding="utf-8")
            _git(root, "add", "-A")
            _git(root, "commit", "-qm", "sneaky test")
            srv, jm = _server(root, store, _content_writer({"pkg/a.py": ["A = 1\n"]}))
            out = _call(srv, "delegate_file_batch", {"tasks": [_task("a", "pkg/a.py", checks=["python3 -m unittest -q tests_x.test_sneaky"])],
                                                     "project": "p", "allowed_paths": ["pkg/a.py"]})
            self.assertNotIn("pkg/core.py", out.get("scope_violations") or [])
            self.assertEqual(out["tasks"][0]["state"], "failed")
            self.assertEqual(out["check_side_effects"], {"a": ["pkg/core.py"]})
            diff = store.diff_path(out["job_id"]).read_text(encoding="utf-8")
            self.assertNotIn("pkg/core.py", diff)
            jm.shutdown()

    def test_receipt_hmac_covers_task_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = _server(root, store, _content_writer({"pkg/a.py": ["A = 1\n"]}))
            out = _call(srv, "delegate_file_batch", {"tasks": [_task("a", "pkg/a.py")], "project": "p"})
            rec = store.read_receipt(out["job_id"])
            self.assertTrue(store.verify_receipt_signature(rec))
            rec["tasks"][0]["writer"] = "forged"
            self.assertFalse(store.verify_receipt_signature(rec))
            jm.shutdown()

    def test_digest_stays_small_when_many_tasks_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            paths = {f"pkg/m{i}.py": ["x" * 50 + " (\n"] for i in range(12)}
            srv, jm = _server(root, store, _content_writer(paths))
            tasks = [_task(f"m{i}", f"pkg/m{i}.py", checks=[f"python3 -m py_compile pkg/m{i}.py"]) for i in range(12)]
            out = _call(srv, "delegate_file_batch", {"tasks": tasks, "project": "p", "claude_writer": False})
            self.assertEqual(out["totals"]["failed"], 12)
            self.assertLessEqual(len(json.dumps(out)), 2400 + 400)
            jm.shutdown()

    def test_concurrent_identical_calls_share_one_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            calls: list = []
            srv, jm = _server(root, store, _content_writer({"pkg/a.py": ["A = 1\n"]}, calls, delay=0.2))
            results: list[dict] = []
            args = {"tasks": [_task("a", "pkg/a.py")], "project": "p", "claude_writer": False}
            threads = [threading.Thread(target=lambda: results.append(_call(srv, "delegate_file_batch", dict(args)))) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(len({r["job_id"] for r in results}), 1)
            self.assertEqual(len(calls), 1)
            jm.shutdown()

    def test_resume_keeps_cost_and_skips_a_same_file_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            calls: list = []

            def factory(name, settings, auth):
                def write(req):
                    calls.append(req.task.task_id)
                    if req.task.action == "create":
                        return WriterReply(text=json.dumps({"path": req.task.path, "content": "A = 1\n"}), cost_cents=0.25)
                    return WriterReply(text=json.dumps({"path": req.task.path, "edits": [{"find": "A = 1", "replace": "A = 2"}]}),
                                       cost_cents=0.25)
                return write

            srv, jm = _server(root, store, factory)
            tasks = [_task("t1", "pkg/chain.py"), _task("t2", "pkg/chain.py", action="modify", depends_on=["t1"])]
            first = _call(srv, "delegate_file_batch", {"tasks": tasks, "project": "p"})
            second = _call(srv, "delegate_file_batch", {"tasks": tasks, "project": "p"})
            self.assertEqual(second["job_id"], first["job_id"])
            self.assertEqual(second["totals"]["unchanged"], 2)
            self.assertEqual(calls, ["t1", "t2"])
            self.assertAlmostEqual(store.read_receipt(first["job_id"])["cost"]["total_cost_cents"], 0.5)
            jm.shutdown()

    def test_moved_workspace_starts_a_fresh_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = _server(root, store, _content_writer({"pkg/a.py": ["A = 1\n"]}))
            args = {"tasks": [_task("a", "pkg/a.py")], "project": "p"}
            first = _call(srv, "delegate_file_batch", dict(args))
            (root / "pkg" / "core.py").write_text("def add(a, b):\n    return b + a\n", encoding="utf-8")
            second = _call(srv, "delegate_file_batch", dict(args))
            self.assertNotEqual(first["job_id"], second["job_id"])
            jm.shutdown()

    def test_dependencies_landed_files_are_shown_to_dependents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            calls: list = []
            srv, jm = _server(root, store, _content_writer({"pkg/a.py": ["MAGIC_SYMBOL = 1\n"], "pkg/b.py": ["B = 2\n"]}, calls))
            tasks = [_task("a", "pkg/a.py"), _task("b", "pkg/b.py", depends_on=["a"])]
            _call(srv, "delegate_file_batch", {"tasks": tasks, "project": "p"})
            prompt_b = next(p for _, tid, p in calls if tid == "b")
            self.assertIn("MAGIC_SYMBOL", prompt_b)
            jm.shutdown()

    def test_cost_budget_and_claude_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            paths = {f"pkg/b{i}.py": [f"B{i} = 1\n"] for i in range(6)}
            srv, jm = _server(root, store, _content_writer(paths))
            tasks = [_task(f"b{i}", f"pkg/b{i}.py") for i in range(6)]
            out = _call(srv, "delegate_file_batch", {"tasks": tasks, "project": "p", "max_cost_cents": 1.0, "max_parallel": 1})
            self.assertIn("budget", out["error"])
            self.assertGreater(out["totals"]["held"], 0)
            jm.shutdown()
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            calls: list = []
            paths = {f"pkg/c{i}.py": ["bad (\n"] for i in range(3)}
            srv, jm = _server(root, store, _content_writer(paths, calls))
            tasks = [_task(f"c{i}", f"pkg/c{i}.py", checks=[f"python3 -m py_compile pkg/c{i}.py"]) for i in range(3)]
            _call(srv, "delegate_file_batch", {"tasks": tasks, "project": "p", "max_claude_escalations": 1, "max_parallel": 1})
            self.assertEqual(sum(1 for name, _, _ in calls if name == "claude"), 1)
            jm.shutdown()

    def test_notifications_cancelled_stops_the_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            paths = {f"pkg/s{i}.py": [f"S{i} = 1\n"] for i in range(5)}
            srv, jm = _server(root, store, _content_writer(paths, delay=0.3))
            tasks = [_task(f"s{i}", f"pkg/s{i}.py") for i in range(5)]
            holder: list[dict] = []
            t = threading.Thread(target=lambda: holder.append(_call(srv, "delegate_file_batch",
                                                                    {"tasks": tasks, "project": "p", "max_parallel": 1}, msg_id=42)))
            t.start()
            time.sleep(0.5)
            srv.handle_Message({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 42}})
            t.join(timeout=30)
            self.assertGreater(holder[0]["totals"]["held"], 0)
            jm.shutdown()

    def test_delegate_continue_refuses_batch_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = _server(root, store, _content_writer({"pkg/a.py": ["A = 1\n"]}))
            out = _call(srv, "delegate_file_batch", {"tasks": [_task("a", "pkg/a.py")], "project": "p"})
            with self.assertRaises(ValueError):
                jm.delegate_continue(job_id=out["job_id"], follow_up_prompt="more", project="p")
            jm.shutdown()

    def test_task_log_returns_attempt_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = _server(root, store, _content_writer({"pkg/a.py": ["bad (\n", "A = 1\n"]}))
            out = _call(srv, "delegate_file_batch", {"tasks": [_task("a", "pkg/a.py", checks=["python3 -m py_compile pkg/a.py"])],
                                                     "project": "p"})
            log = _call(srv, "batch_task_log", {"job_id": out["job_id"], "task_id": "a"})
            self.assertEqual([h["result"] for h in log["task"]["history"]], ["checks_failed", "landed"])
            self.assertTrue(log["recent_check_logs"])
            jm.shutdown()


class PlanAndAnswerHardeningTests(unittest.TestCase):
    def test_protected_paths_any_depth_and_case(self) -> None:
        for path in ("vendor/.git/hooks/pre-commit", "sub/.claude/settings.json", ".claude/settings.prod.json",
                     ".ENV", ".Env.local", ".envrc", "keys/server.pem", ".cc-agy-data/x"):
            self.assertTrue(is_protected_path(path), path)
        self.assertFalse(is_protected_path("src/environment.py"))

    def test_symlinked_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
            os.symlink(".env", root / "pkg" / "link_env.py")
            with self.assertRaises(BatchPlanError) as ctx:
                validate_batch_plan({"tasks": [_task("a", "pkg/link_env.py", action="modify")]}, root)
            self.assertIn("symlink", str(ctx.exception))

    def test_formatter_must_be_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            for bad in ("rm -rf pkg && echo {path}", "black {path}; rm x", "echo {path}"):
                with self.assertRaises(BatchPlanError):
                    validate_batch_plan({"formatter": bad, "tasks": [_task("a", "pkg/a.py")]}, root)
            validate_batch_plan({"formatter": "black {path}", "tasks": [_task("a", "pkg/a.py")]}, root)

    def test_interface_cannot_carry_an_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            body = "\n".join(f"    line_{i} = {i}" for i in range(30))
            with self.assertRaises(BatchPlanError) as ctx:
                validate_batch_plan({"tasks": [_task("a", "pkg/a.py", interface="def f():\n" + body)]}, root)
            self.assertIn("$.tasks[0].interface", str(ctx.exception))

    def test_same_path_ordering_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            reverse = {"tasks": [_task("two", "pkg/new.py", action="modify", depends_on=["one"]), _task("one", "pkg/new.py")]}
            self.assertEqual([t.task_id for t in validate_batch_plan(reverse, root).tasks], ["two", "one"])
            fork = {"tasks": [_task("t1", "pkg/core.py", action="modify", depends_on=["t2"]),
                              _task("t2", "pkg/core.py", action="modify"),
                              _task("t3", "pkg/core.py", action="modify", depends_on=["t2"])]}
            with self.assertRaises(BatchPlanError):
                validate_batch_plan(fork, root)
            double_create = {"tasks": [_task("a", "pkg/n.py"), _task("b", "pkg/n.py", depends_on=["a"])]}
            with self.assertRaises(BatchPlanError) as ctx:
                validate_batch_plan(double_create, root)
            self.assertIn("use 'modify'", str(ctx.exception))

    def test_private_plan_files_only_under_tmp(self) -> None:
        from server.batch_plan import load_batch_plan

        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _repo(tmp)
            plan = Path(tmp) / "plan.json"
            plan.write_text(json.dumps({"tasks": [_task("a", "pkg/a.py")]}), encoding="utf-8")
            os.chmod(plan, 0o666)
            with self.assertRaises(BatchPlanError):
                load_batch_plan(plan, root)
            os.chmod(plan, 0o600)
            self.assertEqual(len(load_batch_plan(plan, root).tasks), 1)

    def test_answer_existence_rules_and_mixed_eol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "m.py").write_bytes(b"one\r\ntwo\nthree\n")
            out = apply_writer_answer(FileTask("t", "m.py", "modify", "x"), {"path": "m.py", "edits": [{"find": "two", "replace": "TWO"}]}, root)
            self.assertEqual(out.new_bytes, b"one\r\nTWO\nthree\n")
            with self.assertRaises(Exception):
                apply_writer_answer(FileTask("t", "gone.py", "modify", "x"), {"path": "gone.py", "content": "x\n"}, root)
            with self.assertRaises(Exception):
                apply_writer_answer(FileTask("t", "m.py", "create", "x"), {"path": "m.py", "content": "x\n"}, root)
            ok = apply_writer_answer(FileTask("t", "m.py", "create", "x"), {"path": "m.py", "content": "x\n"}, root, allow_existing=True)
            self.assertEqual(ok.new_bytes, b"x\n")

    def test_settings_are_clamped_and_validated(self) -> None:
        self.assertEqual(resolve_batch_settings({"check_timeout_seconds": -5}).check_timeout, 5.0)
        self.assertEqual(resolve_batch_settings({"check_timeout_seconds": 99999}).check_timeout, 3600.0)
        with self.assertRaises(ValueError):
            resolve_batch_settings({"max_cost_cents": -1})


class WriterHardeningTests(unittest.TestCase):
    def test_transient_exhaustion_moves_to_claude(self) -> None:
        def genai(req):
            raise WriterTransportError(503)

        def claude(req):
            return WriterReply(text=json.dumps({"path": "n.py", "content": "ok\n"}))

        with tempfile.TemporaryDirectory() as tmp:
            out = run_writer_ladder(FileTask("t", "n.py", "create", "x"), routed="genai", writers={"genai": genai, "claude": claude},
                                    verify=lambda a: (True, ""), root=tmp, sleep=lambda s: None, jitter=lambda: 1.0)
        self.assertEqual((out.state, out.writer_used), ("landed", "claude"))
        self.assertEqual(out.history[0]["result"], "transient_exhausted")

    def test_config_error_skips_the_rung_without_counting(self) -> None:
        def genai(req):
            return WriterReply(text=json.dumps({"path": "n.py", "content": "bad\n"}))

        def claude(req):
            raise WriterConfigError("claude -p exited 2")

        with tempfile.TemporaryDirectory() as tmp:
            out = run_writer_ladder(FileTask("t", "n.py", "create", "x"), routed="genai", writers={"genai": genai, "claude": claude},
                                    verify=lambda a: (False, "nope"), root=tmp, sleep=lambda s: None)
        self.assertEqual(out.attempts, 2)
        self.assertEqual(out.history[-1]["result"], "writer_error")

    def test_claude_writer_uses_a_prompt_file_and_reports_cli_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake_claude.py"
            fake.write_text(
                "#!/usr/bin/env python3\nimport json, sys\nargs = sys.argv[1:]\n"
                "path = args[args.index('--system-prompt-file') + 1]\nsize = len(open(path).read())\n"
                "print(json.dumps({'type': 'result', 'is_error': False, 'api_error_status': None,\n"
                "  'structured_output': {'path': 'n.py', 'content': f'size={size}'}, 'stop_reason': 'end_turn',\n"
                "  'usage': {'input_tokens': 10, 'output_tokens': 5},\n"
                "  'modelUsage': {'claude-opus-x': {'inputTokens': 10, 'outputTokens': 5}, 'mystery': {'inputTokens': 1}}}))\n",
                encoding="utf-8")
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            writer = make_claude_writer("opus", binary=str(fake))
            from server.writers import WriterRequest
            reply = writer(WriterRequest(task=FileTask("t", "n.py", "create", "x"), prompt="go", shared_prefix="x" * 300_000))
            self.assertIn("size=", reply.text)
            self.assertGreater(int(json.loads(reply.text)["content"].split("=")[1]), 300_000)
            self.assertFalse(reply.pricing_known)
            self.assertIn("claude_input_tokens", reply.usage)
            broken = Path(tmp) / "broken_claude.sh"
            broken.write_text("#!/bin/sh\necho 'error: unknown option' >&2\nexit 2\n", encoding="utf-8")
            broken.chmod(broken.stat().st_mode | stat.S_IEXEC)
            with self.assertRaises(WriterConfigError):
                make_claude_writer("opus", binary=str(broken))(WriterRequest(task=FileTask("t", "n.py", "create", "x"), prompt="go"))

    def test_check_output_streams_to_log_with_bounded_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = run_commands(["python3 -c \"import sys; sys.stdout.write('y' * 5_000_000)\""], Path(tmp),
                               timeout=60, log_dir=Path(tmp) / "logs")
            self.assertLessEqual(len(res[0].tail), 3600)
            self.assertEqual(Path(res[0].log_path).stat().st_size, 5_000_000)


if __name__ == "__main__":
    unittest.main()
