"""Tests for task verification commands (server/verify.py), the batch journal and the dependency scheduler."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from server.batch_journal import BatchJournal
from server.batch_plan import FileTask
from server.task_graph import run_task_graph
from server.verify import run_commands, scrubbed_env


def _task(task_id: str, path: str, depends_on: tuple[str, ...] = ()) -> FileTask:
    return FileTask(task_id=task_id, path=path, action="create", intent="x", depends_on=depends_on)


class VerifyTests(unittest.TestCase):
    def test_pass_fail_and_bounded_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = run_commands(["true", "python3 -c \"print('x' * 200000); raise SystemExit(3)\""], Path(tmp), timeout=30)
            self.assertTrue(res[0].passed)
            self.assertFalse(res[1].passed)
            self.assertEqual(res[1].exit_code, 3)
            self.assertLessEqual(len(res[1].tail), 4000)

    def test_stops_after_first_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = run_commands(["false", "true"], Path(tmp), timeout=30)
            self.assertEqual(len(res), 1)

    def test_timeout_kills_the_whole_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "child_alive"
            cmd = f"(sleep 3; touch {marker}) & sleep 30"
            started = time.time()
            res = run_commands([cmd], Path(tmp), timeout=1)
            self.assertTrue(res[0].timed_out)
            self.assertLess(time.time() - started, 10)
            time.sleep(3.5)
            self.assertFalse(marker.exists(), "background child survived the timeout")

    def test_vendor_credentials_are_scrubbed(self) -> None:
        os.environ["GEMINI_API_KEY"] = "secret-gemini"
        os.environ["ANTHROPIC_API_KEY"] = "secret-anthropic"
        try:
            env = scrubbed_env(Path("/tmp"))
            self.assertEqual(env.get("GEMINI_API_KEY", ""), "")
            self.assertEqual(env.get("ANTHROPIC_API_KEY", ""), "")
            with tempfile.TemporaryDirectory() as tmp:
                res = run_commands(["test -z \"$GEMINI_API_KEY\" && test -z \"$ANTHROPIC_API_KEY\""], Path(tmp), timeout=30)
                self.assertTrue(res[0].passed, res[0].tail)
        finally:
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)


class JournalTests(unittest.TestCase):
    def test_landed_task_resumes_as_unchanged_only_when_file_and_task_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "copy"
            root.mkdir()
            (root / "a.py").write_text("A = 1\n", encoding="utf-8")
            journal = BatchJournal(Path(tmp) / "data", "job-1-abcdef12")
            task = _task("a", "a.py")
            self.assertFalse(journal.is_unchanged(task, root))
            journal.record_landed(task, root)
            self.assertTrue(journal.is_unchanged(task, root))
            reopened = BatchJournal(Path(tmp) / "data", "job-1-abcdef12")
            self.assertTrue(reopened.is_unchanged(task, root))
            changed_task = FileTask(task_id="a", path="a.py", action="create", intent="different")
            self.assertFalse(reopened.is_unchanged(changed_task, root))
            (root / "a.py").write_text("A = 2\n", encoding="utf-8")
            self.assertFalse(reopened.is_unchanged(task, root))

    def test_corrupt_lines_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = BatchJournal(Path(tmp) / "data", "job-1-abcdef12")
            journal.path.parent.mkdir(parents=True, exist_ok=True)
            journal.path.write_text("not json\n{\"task_id\": 3}\n", encoding="utf-8")
            root = Path(tmp)
            (root / "a.py").write_text("x", encoding="utf-8")
            self.assertFalse(BatchJournal(Path(tmp) / "data", "job-1-abcdef12").is_unchanged(_task("a", "a.py"), root))


class TaskGraphTests(unittest.TestCase):
    def test_respects_dependencies_and_parallelism(self) -> None:
        tasks = [_task("a", "a.py"), _task("b", "b.py"), _task("c", "c.py", ("a", "b")), _task("d", "d.py", ("c",))]
        order: list[str] = []
        active = {"n": 0, "max": 0}
        lock = threading.Lock()

        def run(task: FileTask) -> str:
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.05)
            with lock:
                active["n"] -= 1
                order.append(task.task_id)
            return "landed"

        states = run_task_graph(tasks, run, max_parallel=4)
        self.assertEqual(states, {"a": "landed", "b": "landed", "c": "landed", "d": "landed"})
        self.assertLess(order.index("a"), order.index("c"))
        self.assertLess(order.index("b"), order.index("c"))
        self.assertLess(order.index("c"), order.index("d"))
        self.assertEqual(active["max"], 2)

    def test_same_path_never_runs_concurrently(self) -> None:
        tasks = [_task("one", "same.py"), _task("two", "same.py", ("one",)), _task("x", "x.py"), _task("y", "y.py")]
        running_paths: list[str] = []
        overlap = {"hit": False}
        lock = threading.Lock()

        def run(task: FileTask) -> str:
            with lock:
                if task.path in running_paths:
                    overlap["hit"] = True
                running_paths.append(task.path)
            time.sleep(0.03)
            with lock:
                running_paths.remove(task.path)
            return "landed"

        run_task_graph(tasks, run, max_parallel=4)
        self.assertFalse(overlap["hit"])

    def test_failure_holds_every_dependent(self) -> None:
        tasks = [_task("a", "a.py"), _task("b", "b.py", ("a",)), _task("c", "c.py", ("b",)), _task("z", "z.py")]
        ran: list[str] = []

        def run(task: FileTask) -> str:
            ran.append(task.task_id)
            return "failed" if task.task_id == "a" else "landed"

        settled: list[tuple[str, str]] = []
        states = run_task_graph(tasks, run, max_parallel=2, on_settle=lambda t, s: settled.append((t.task_id, s)))
        self.assertEqual(states, {"a": "failed", "b": "held", "c": "held", "z": "landed"})
        self.assertNotIn("b", ran)
        self.assertNotIn("c", ran)
        self.assertEqual(sorted(s for s in settled), sorted(states.items()))

    def test_unchanged_counts_as_done_and_exceptions_fail_the_task(self) -> None:
        tasks = [_task("a", "a.py"), _task("b", "b.py", ("a",)), _task("c", "c.py")]

        def run(task: FileTask) -> str:
            if task.task_id == "c":
                raise RuntimeError("boom")
            return "unchanged" if task.task_id == "a" else "landed"

        states = run_task_graph(tasks, run, max_parallel=2)
        self.assertEqual(states, {"a": "unchanged", "b": "landed", "c": "failed"})

    def test_first_task_can_run_alone_to_warm_caches(self) -> None:
        tasks = [_task(f"t{i}", f"t{i}.py") for i in range(4)]
        timeline: list[tuple[str, str]] = []
        lock = threading.Lock()

        def run(task: FileTask) -> str:
            with lock:
                timeline.append(("start", task.task_id))
            time.sleep(0.03)
            with lock:
                timeline.append(("end", task.task_id))
            return "landed"

        run_task_graph(tasks, run, max_parallel=4, warm_first=True)
        self.assertEqual(timeline[0], ("start", "t0"))
        self.assertEqual(timeline[1], ("end", "t0"))

    def test_stop_event_holds_remaining_tasks(self) -> None:
        tasks = [_task("a", "a.py"), _task("b", "b.py", ("a",))]
        stop = threading.Event()

        def run(task: FileTask) -> str:
            stop.set()
            return "landed"

        states = run_task_graph(tasks, run, max_parallel=1, stop=stop)
        self.assertEqual(states["a"], "landed")
        self.assertEqual(states["b"], "held")


if __name__ == "__main__":
    unittest.main()
