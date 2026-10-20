"""Tests for the batch plan validator (server/batch_plan.py). Written before the implementation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from server.batch_plan import BatchPlanError, FileTask, validate_batch_plan


def _repo(tmp: str) -> Path:
    root = Path(tmp) / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    (root / "pkg").mkdir()
    (root / "pkg" / "core.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    return root


def _task(task_id: str, path: str, action: str = "create", **extra) -> dict:
    base = {"task_id": task_id, "path": path, "action": action, "intent": f"Write {path} for the demo."}
    base.update(extra)
    return base


class ValidateBatchPlanTests(unittest.TestCase):
    def test_valid_plan_expands_into_ordered_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _repo(tmp)
            plan = {
                "shared_context": ["pkg/core.py"],
                "tasks": [
                    _task("api", "pkg/api.py", interface="from pkg.api import serve  # serve(port: int) -> None"),
                    _task("cli", "pkg/cli.py", depends_on=["api"], checks=["python -m py_compile pkg/cli.py"]),
                    _task("core", "pkg/core.py", action="modify", context_files=["README.md"]),
                ],
            }
            compiled = validate_batch_plan(plan, root)
            self.assertEqual([t.task_id for t in compiled.tasks], ["api", "cli", "core"])
            self.assertIsInstance(compiled.tasks[0], FileTask)
            self.assertEqual(compiled.tasks[1].depends_on, ("api",))
            self.assertEqual(compiled.shared_context, ("pkg/core.py",))
            # Fingerprints are stable and change when the task changes.
            again = validate_batch_plan(json.loads(json.dumps(plan)), root)
            self.assertEqual(compiled.tasks[0].fingerprint(), again.tasks[0].fingerprint())
            plan["tasks"][0]["intent"] = "Something else entirely."
            changed = validate_batch_plan(plan, root)
            self.assertNotEqual(compiled.tasks[0].fingerprint(), changed.tasks[0].fingerprint())

    def test_reports_every_problem_with_its_json_path_in_one_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _repo(tmp)
            plan = {
                "tasks": [
                    _task("a", "pkg/a.py"),
                    _task("a", "pkg/b.py"),                                # duplicate id
                    _task("c", "../escape.py"),                            # outside workspace
                    _task("d", ".env.local"),                              # hard-deny
                    _task("e", ".git/config", action="modify"),           # hard-deny
                    _task("f", "pkg/f.py", depends_on=["nope"]),          # unknown dependency
                    _task("g", "pkg/core.py"),                             # create over an existing file
                    _task("h", "pkg/missing.py", action="modify"),        # modify a missing file
                    _task("i", "pkg/i.py", action="rename"),              # bad action
                    {"task_id": "j", "path": "pkg/j.py", "action": "create"},  # missing intent
                    _task("k", "pkg/k.py", context_files=["no/such/file.py"]),
                    _task("l", "/abs/path.py"),
                    _task("m", ".claude/settings.local.json"),
                    _task("n", ".mcp.json"),
                ]
            }
            with self.assertRaises(BatchPlanError) as ctx:
                validate_batch_plan(plan, root)
            paths = {p.json_path for p in ctx.exception.problems}
            for expected in (
                "$.tasks[1].task_id",
                "$.tasks[2].path",
                "$.tasks[3].path",
                "$.tasks[4].path",
                "$.tasks[5].depends_on[0]",
                "$.tasks[6].action",
                "$.tasks[7].action",
                "$.tasks[8].action",
                "$.tasks[9].intent",
                "$.tasks[10].context_files[0]",
                "$.tasks[11].path",
                "$.tasks[12].path",
                "$.tasks[13].path",
            ):
                self.assertIn(expected, paths)
            self.assertIn("$.tasks[2].path", str(ctx.exception))

    def test_dependency_cycle_is_reported_with_the_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _repo(tmp)
            plan = {"tasks": [
                _task("a", "pkg/a.py", depends_on=["c"]),
                _task("b", "pkg/b.py", depends_on=["a"]),
                _task("c", "pkg/c.py", depends_on=["b"]),
                _task("s", "pkg/s.py", depends_on=["s"]),
            ]}
            with self.assertRaises(BatchPlanError) as ctx:
                validate_batch_plan(plan, root)
            messages = " | ".join(p.message for p in ctx.exception.problems)
            self.assertIn("cycle", messages)
            self.assertIn("a -> c -> b -> a", messages.replace("b -> a -> c -> b", "a -> c -> b -> a").replace(
                "c -> b -> a -> c", "a -> c -> b -> a"))
            self.assertTrue(any(p.json_path == "$.tasks[3].depends_on[0]" for p in ctx.exception.problems))

    def test_same_path_twice_requires_an_explicit_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _repo(tmp)
            unordered = {"tasks": [_task("one", "pkg/new.py"), _task("two", "pkg/new.py", action="modify")]}
            with self.assertRaises(BatchPlanError) as ctx:
                validate_batch_plan(unordered, root)
            self.assertTrue(any(p.json_path == "$.tasks[1].path" for p in ctx.exception.problems))
            ordered = {"tasks": [
                _task("one", "pkg/new.py"),
                _task("two", "pkg/new.py", action="modify", depends_on=["one"]),
            ]}
            compiled = validate_batch_plan(ordered, root)
            self.assertEqual([t.task_id for t in compiled.tasks], ["one", "two"])

    def test_allowed_paths_and_code_dictation_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _repo(tmp)
            dictated = "\n".join(f"    x{i} = {i}" for i in range(12))
            plan = {"tasks": [
                _task("a", "docs/a.md"),
                _task("b", "pkg/b.py", intent="Write this exactly:\n" + dictated),
            ]}
            with self.assertRaises(BatchPlanError) as ctx:
                validate_batch_plan(plan, root, allowed_paths=["pkg/**"])
            paths = {p.json_path for p in ctx.exception.problems}
            self.assertIn("$.tasks[0].path", paths)
            self.assertIn("$.tasks[1].intent", paths)

    def test_mutating_checks_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _repo(tmp)
            plan = {"tasks": [_task("a", "pkg/a.py", checks=["rm -rf pkg"], repro_test="python -m pytest -q")]}
            with self.assertRaises(BatchPlanError) as ctx:
                validate_batch_plan(plan, root)
            self.assertTrue(any(p.json_path == "$.tasks[0].checks[0]" for p in ctx.exception.problems))

    def test_plan_must_be_an_object_with_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _repo(tmp)
            for bad in ([], {"tasks": []}, {"tasks": "x"}, "nope"):
                with self.assertRaises(BatchPlanError):
                    validate_batch_plan(bad, root)


if __name__ == "__main__":
    unittest.main()
