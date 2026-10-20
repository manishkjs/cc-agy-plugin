"""End-to-end tests for delegate_file_batch with fake writers (server/batch_runner.py + MCP wiring)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.provenance_check import verify_workspace_provenance
from server.jobs import JobManager
from server.mcp_stdio import MCPServer
from server.receipts import ReceiptStore
from server.writers import WriterAuthError, WriterReply


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repo(tmp: str) -> tuple[Path, ReceiptStore]:
    root = Path(tmp) / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "core.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "checks").mkdir()
    (root / "checks" / "__init__.py").write_text("", encoding="utf-8")
    test_tpl = ("import unittest\nfrom pkg.core import add\n\n\nclass T(unittest.TestCase):\n"
                "    def test_it(self):\n        self.assertEqual(add({a}, 2), 3)\n")
    (root / "checks" / "test_coerce.py").write_text(test_tpl.format(a="'1'"), encoding="utf-8")
    (root / "checks" / "test_ints.py").write_text(test_tpl.format(a="1"), encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root, ReceiptStore(Path(tmp) / "data", workspace_root=root)


def _content_for(prompt: str, path: str) -> str:
    return f"# generated for {path}\nVALUE = {len(path)}\n"


class FakeWriters:
    """Writer factory returning scripted writers and recording every call."""

    def __init__(self, *, bad_paths: tuple[str, ...] = (), auth_fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.bad_paths = bad_paths
        self.auth_fail = auth_fail

    def __call__(self, name: str, settings, auth):
        def write(req):
            self.calls.append((name, req.task.task_id))
            if self.auth_fail:
                raise WriterAuthError(403, "permission denied on project")
            if req.task.path in self.bad_paths and name != "claude":
                return WriterReply(text=json.dumps({"path": req.task.path, "content": "this is not python (\n"}),
                                   usage={"prompt_token_count": 100, "candidates_token_count": 10}, cost_cents=0.01)
            if req.task.action == "modify":
                return WriterReply(text=json.dumps({"path": req.task.path, "edits": [
                    {"find": "return a + b", "replace": "return int(a) + int(b)"}]}), cost_cents=0.01)
            return WriterReply(text=json.dumps({"path": req.task.path, "content": _content_for(req.prompt, req.task.path)}),
                               usage={"prompt_token_count": 100, "candidates_token_count": 10}, cost_cents=0.01)
        return write


def _plan(root: Path, tasks: list[dict], **extra) -> str:
    plan = {"tasks": tasks, **extra}
    # Plans live outside the repo so they never show up as untracked changes.
    path = root.parent / "batch_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return str(path)


def _five_tasks() -> list[dict]:
    compile_check = "python3 -m py_compile {p}"
    return [
        {"task_id": "a", "path": "pkg/a.py", "action": "create", "intent": "Module a.", "checks": [compile_check.format(p="pkg/a.py")]},
        {"task_id": "b", "path": "pkg/b.py", "action": "create", "intent": "Module b.", "depends_on": ["a"],
         "checks": [compile_check.format(p="pkg/b.py")]},
        {"task_id": "c", "path": "pkg/c.py", "action": "create", "intent": "Module c.", "checks": [compile_check.format(p="pkg/c.py")]},
        {"task_id": "d", "path": "pkg/d.py", "action": "create", "intent": "Module d.", "depends_on": ["b", "c"],
         "checks": [compile_check.format(p="pkg/d.py")]},
        {"task_id": "core", "path": "pkg/core.py", "action": "modify", "intent": "Coerce inputs to int.",
         "checks": ["python3 -m unittest -q checks.test_coerce"]},
    ]


class FileBatchTests(unittest.TestCase):
    def _server(self, root: Path, store: ReceiptStore, writers: FakeWriters) -> tuple[MCPServer, JobManager]:
        jm = JobManager(workspace_root=root, store=store, worker_backend=lambda _r: {"text": "unused"})
        return MCPServer(job_manager=jm, store=store, batch_writer_factory=writers), jm

    def _call(self, srv: MCPServer, name: str, args: dict) -> dict:
        resp = srv.handle_Message({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})
        return json.loads(resp["result"]["content"][0]["text"])

    def test_five_task_batch_lands_applies_and_passes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            writers = FakeWriters()
            srv, jm = self._server(root, store, writers)
            digest = self._call(srv, "delegate_file_batch", {"plan_path": _plan(root, _five_tasks()), "project": "p"})
            self.assertEqual(digest["status"], "ok", digest)
            self.assertEqual(digest["totals"]["landed"], 5)
            self.assertLess(len(json.dumps(digest)) / 4, 600, "digest must stay under ~600 tokens")
            receipt = store.read_receipt(digest["job_id"])
            self.assertEqual(receipt["kind"], "file_batch")
            self.assertEqual(sorted(receipt["files_changed"]), ["pkg/a.py", "pkg/b.py", "pkg/c.py", "pkg/core.py", "pkg/d.py"])
            self.assertTrue(all(t["added_line_sha256"] for t in receipt["tasks"] if t["state"] == "landed"))
            self.assertFalse((root / "pkg" / "a.py").exists(), "nothing touches the workspace before delegate_apply")
            diff = self._call(srv, "get_diff", {"job_id": digest["job_id"]})
            self.assertIn("pkg/d.py", json.dumps(diff))
            applied = self._call(srv, "delegate_apply", {"job_id": digest["job_id"]})
            self.assertTrue(applied["applied"])
            self.assertIn("int(a)", (root / "pkg" / "core.py").read_text(encoding="utf-8"))
            self.assertTrue(store.verify_receipt_signature(store.read_receipt(digest["job_id"])))
            prov = verify_workspace_provenance(root, store)
            self.assertTrue(prov["ok"], prov)
            # b waited for a; d waited for b and c.
            order = [tid for _, tid in writers.calls]
            self.assertLess(order.index("a"), order.index("b"))
            self.assertLess(order.index("b"), order.index("d"))
            jm.shutdown()

    def test_task_that_fails_twice_lands_on_claude(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            writers = FakeWriters(bad_paths=("pkg/c.py",))
            srv, jm = self._server(root, store, writers)
            digest = self._call(srv, "delegate_file_batch", {"plan_path": _plan(root, _five_tasks()), "project": "p"})
            task_c = next(t for t in digest["tasks"] if t["task_id"] == "c")
            self.assertEqual((task_c["state"], task_c["attempts"], task_c["writer"]), ("landed", 3, "claude"))
            self.assertEqual([n for n, tid in writers.calls if tid == "c"], ["genai", "genai", "claude"])
            self.assertEqual(digest["status"], "ok")
            jm.shutdown()

    def test_failed_task_holds_dependents_and_reports_a_bounded_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            writers = FakeWriters(bad_paths=("pkg/a.py",))
            srv, jm = self._server(root, store, writers)
            digest = self._call(srv, "delegate_file_batch", {"plan_path": _plan(root, _five_tasks()), "project": "p",
                                                             "claude_writer": False})
            states = {t["task_id"]: t["state"] for t in digest["tasks"]}
            self.assertEqual(states["a"], "failed")
            self.assertEqual(states["b"], "held")
            self.assertEqual(states["d"], "held")
            self.assertEqual(states["c"], "landed")
            failed = next(t for t in digest["tasks"] if t["task_id"] == "a")
            self.assertIn("py_compile", failed["check_tail"])
            self.assertLessEqual(len(failed["check_tail"]), 1200)
            self.assertEqual(digest["status"], "checkpoint")
            jm.shutdown()

    def test_resending_the_same_batch_skips_landed_tasks_at_zero_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            writers = FakeWriters()
            srv, jm = self._server(root, store, writers)
            plan_path = _plan(root, _five_tasks())
            first = self._call(srv, "delegate_file_batch", {"plan_path": plan_path, "project": "p"})
            calls_after_first = len(writers.calls)
            second = self._call(srv, "delegate_file_batch", {"plan_path": plan_path, "project": "p"})
            self.assertEqual(second["job_id"], first["job_id"])
            self.assertEqual(second["totals"]["unchanged"], 5)
            self.assertEqual(second["totals"]["cost_cents"], 0.0)
            self.assertEqual(len(writers.calls), calls_after_first)
            jm.shutdown()

    def test_auth_failure_stops_the_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            writers = FakeWriters(auth_fail=True)
            srv, jm = self._server(root, store, writers)
            digest = self._call(srv, "delegate_file_batch", {"plan_path": _plan(root, _five_tasks()), "project": "p",
                                                             "max_parallel": 1})
            self.assertEqual(digest["status"], "error")
            self.assertIn("403", digest["error"])
            self.assertEqual(len(writers.calls), 1, "no attempts after the first 403")
            jm.shutdown()

    def test_invalid_plan_returns_every_problem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = self._server(root, store, FakeWriters())
            bad = [{"task_id": "x", "path": ".env", "action": "create", "intent": "nope"},
                   {"task_id": "y", "path": "pkg/y.py", "action": "create", "intent": "y", "depends_on": ["zzz"]}]
            digest = self._call(srv, "delegate_file_batch", {"plan_path": _plan(root, bad), "project": "p"})
            self.assertEqual(digest["status"], "invalid_plan")
            paths = {p["json_path"] for p in digest["problems"]}
            self.assertEqual(paths, {"$.tasks[0].path", "$.tasks[1].depends_on[0]"})
            jm.shutdown()

    def test_repro_test_must_fail_before_the_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = self._server(root, store, FakeWriters())
            repro = "python3 -m unittest -q checks.test_coerce"
            already_passing = "python3 -m unittest -q checks.test_ints"
            tasks = [
                {"task_id": "fix", "path": "pkg/core.py", "action": "modify", "intent": "Coerce to int.", "repro_test": repro},
            ]
            ok = self._call(srv, "delegate_file_batch", {"plan_path": _plan(root, tasks), "project": "p"})
            self.assertEqual(ok["tasks"][0]["state"], "landed")
            tasks[0]["repro_test"] = already_passing
            tasks[0]["intent"] = "Coerce to int (second plan)."
            bad = self._call(srv, "delegate_file_batch", {"plan_path": _plan(root, tasks), "project": "p"})
            self.assertEqual(bad["tasks"][0]["state"], "failed")
            self.assertIn("already passes", bad["tasks"][0]["check_tail"])
            jm.shutdown()

    def test_progress_is_reported_per_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = self._server(root, store, FakeWriters())
            events: list[dict] = []
            srv.notify = events.append
            resp = srv.handle_Message({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {
                "name": "delegate_file_batch", "arguments": {"plan_path": _plan(root, _five_tasks()), "project": "p"},
                "_meta": {"progressToken": "tok-1"}}})
            self.assertIn("result", resp)
            progress = [e for e in events if e.get("method") == "notifications/progress"]
            self.assertGreaterEqual(len(progress), 5)
            self.assertTrue(all(e["params"]["progressToken"] == "tok-1" for e in progress))
            self.assertEqual(progress[-1]["params"]["progress"], 5)
            self.assertEqual(progress[-1]["params"]["total"], 5)
            jm.shutdown()

    def test_probe_writers_reports_ready_fail_and_fix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = self._server(root, store, FakeWriters())
            ok = self._call(srv, "probe_writers", {"project": "p"})
            self.assertEqual(ok["status"], "ok")
            self.assertEqual([w["writer"] for w in ok["writers"]], ["genai", "claude"])
            srv.batch_writer_factory = FakeWriters(auth_fail=True)
            bad = self._call(srv, "probe_writers", {"project": "p", "claude_writer": False})
            self.assertEqual(bad["status"], "fail")
            self.assertIn("gcloud auth application-default login", bad["writers"][0]["fix"])
            srv.batch_writer_factory = lambda name, settings, auth: None if name == "claude" else FakeWriters()(name, settings, auth)
            no_cli = self._call(srv, "probe_writers", {"project": "p"})
            self.assertEqual(no_cli["writers"][1]["state"], "fail")
            self.assertIn("CC_AGY_CLAUDE_BIN", no_cli["writers"][1]["fix"])
            jm.shutdown()

    def test_in_place_is_refused_and_revert_accepts_receipt_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = self._server(root, store, FakeWriters())
            resp = srv.handle_Message({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "delegate_file_batch", "arguments": {"plan_path": _plan(root, _five_tasks()), "in_place": True}}})
            self.assertIn("isolated job copy", json.dumps(resp))
            digest = self._call(srv, "delegate_file_batch", {"plan_path": _plan(root, _five_tasks()), "project": "p"})
            self._call(srv, "delegate_apply", {"job_id": digest["job_id"]})
            reverted = self._call(srv, "delegate_revert", {"receipt_id": digest["job_id"]})
            self.assertTrue(reverted["reverted"])
            self.assertFalse((root / "pkg" / "a.py").exists())
            self.assertNotIn("int(a)", (root / "pkg" / "core.py").read_text(encoding="utf-8"))
            jm.shutdown()


if __name__ == "__main__":
    unittest.main()
