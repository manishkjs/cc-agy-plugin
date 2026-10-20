"""Tests for the mid-chat helper tools: draft_document, write_tests_from_cases, replicate_edit."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from server.jobs import JobManager
from server.mcp_stdio import MCPServer
from server.receipts import ReceiptStore
from server.writers import WriterReply


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
    (root / "pkg" / "price.py").write_text(
        "RATE = 0.2\n\n\ndef with_tax(amount):\n    return round(amount * (1 + RATE), 2)\n", encoding="utf-8")
    for name in ("alpha", "beta", "gamma"):
        (root / "pkg" / f"{name}.py").write_text(
            f"import logging\n\nLOG = logging.getLogger('{name}')\n\n\ndef run():\n    LOG.info('start')\n    return '{name}'\n",
            encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "intro.md").write_text("# Intro\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root, ReceiptStore(Path(tmp) / "data", workspace_root=root)


class ScriptedWriters:
    """Writer factory that returns scripted answers per (task path, attempt)."""

    def __init__(self, answers: dict[str, list[dict]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def __call__(self, name, settings, auth):
        def write(req):
            self.calls.append((name, req.task.path))
            queue = self.answers[req.task.path]
            answer = queue.pop(0) if len(queue) > 1 else queue[0]
            return WriterReply(text=json.dumps(answer), cost_cents=0.01)
        return write


class AssistToolTests(unittest.TestCase):
    def _call(self, srv: MCPServer, name: str, args: dict) -> dict:
        resp = srv.handle_Message({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})
        return json.loads(resp["result"]["content"][0]["text"])

    def _server(self, root, store, writers):
        jm = JobManager(workspace_root=root, store=store, worker_backend=lambda _r: {"text": "unused"})
        return MCPServer(job_manager=jm, store=store, batch_writer_factory=writers), jm

    # ---------------------------------------------------------------- draft_document

    def test_draft_document_rejects_facts_not_found_verbatim_before_any_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            writers = ScriptedWriters({})
            srv, jm = self._server(root, store, writers)
            out = self._call(srv, "draft_document", {
                "path": "docs/pricing.md", "purpose": "Explain tax handling.", "audience": "support engineers",
                "sections": ["Overview"], "project": "p",
                "facts": [{"claim": "Tax is 20%", "source": "pkg/price.py", "quote": "RATE = 0.25"}],
            })
            self.assertEqual(out["status"], "invalid_request")
            self.assertIn("facts[0]", json.dumps(out))
            self.assertEqual(writers.calls, [])
            jm.shutdown()

    def test_draft_document_gate_requires_every_section_and_existing_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            good = "# Pricing\n\n## Overview\nSee [intro](intro.md) and `pkg/price.py`.\n\n## Rounding\nTwo decimals.\n"
            missing_section = "# Pricing\n\n## Overview\nOnly one section.\n"
            dead_link = "# Pricing\n\n## Overview\nSee [gone](nowhere.md).\n\n## Rounding\nx\n"
            writers = ScriptedWriters({"docs/pricing.md": [
                {"path": "docs/pricing.md", "content": missing_section},
                {"path": "docs/pricing.md", "content": dead_link},
                {"path": "docs/pricing.md", "content": good},
            ]})
            srv, jm = self._server(root, store, writers)
            out = self._call(srv, "draft_document", {
                "path": "docs/pricing.md", "purpose": "Explain tax handling.", "audience": "support engineers",
                "sections": ["Overview", "Rounding"], "project": "p",
                "facts": [{"claim": "Tax rate", "source": "pkg/price.py", "quote": "RATE = 0.2"}],
            })
            task = out["tasks"][0]
            self.assertEqual((task["state"], task["attempts"], task["writer"]), ("landed", 3, "claude"))
            self.assertEqual(out["status"], "ok")
            jm.shutdown()

    # ---------------------------------------------------------------- write_tests_from_cases

    def test_tests_from_cases_requires_every_case_unchanged_expectations_and_a_passing_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            base = ("import unittest\nfrom pkg.price import with_tax\n\n\nclass PriceTests(unittest.TestCase):\n")
            missing_case = base + "    def test_zero(self):\n        self.assertEqual(with_tax(0), 0.0)\n"
            edited_expect = base + ("    def test_zero(self):\n        self.assertEqual(with_tax(0), 0.0)\n\n"
                                    "    def test_hundred(self):\n        self.assertEqual(with_tax(100), 121.0)\n")
            good = base + ("    def test_zero(self):\n        self.assertEqual(with_tax(0), 0.0)\n\n"
                           "    def test_hundred(self):\n        self.assertEqual(with_tax(100), 120.0)\n")
            writers = ScriptedWriters({"tests_price/test_price.py": [
                {"path": "tests_price/test_price.py", "content": missing_case},
                {"path": "tests_price/test_price.py", "content": edited_expect},
                {"path": "tests_price/test_price.py", "content": good},
            ]})
            srv, jm = self._server(root, store, writers)
            (root / "tests_price").mkdir()
            (root / "tests_price" / "__init__.py").write_text("", encoding="utf-8")
            _git(root, "add", "-A")
            _git(root, "commit", "-qm", "pkg for tests")
            out = self._call(srv, "write_tests_from_cases", {
                "test_path": "tests_price/test_price.py", "target_path": "pkg/price.py", "project": "p",
                "test_command": "python3 -m unittest -q tests_price.test_price",
                "cases": [{"name": "zero", "given": "with_tax(0)", "expect": "0.0"},
                          {"name": "hundred", "given": "with_tax(100)", "expect": "120.0"}],
            })
            task = out["tasks"][0]
            self.assertEqual((task["state"], task["attempts"]), ("landed", 3), out)
            jm.shutdown()

    def test_tests_from_cases_fails_when_the_suite_does_not_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            wrong = ("import unittest\nfrom pkg.price import with_tax\n\n\nclass PriceTests(unittest.TestCase):\n"
                     "    def test_hundred(self):\n        self.assertEqual(with_tax(100), 999.0)  # expect 120.0\n")
            writers = ScriptedWriters({"test_price_x.py": [{"path": "test_price_x.py", "content": wrong}]})
            srv, jm = self._server(root, store, writers)
            out = self._call(srv, "write_tests_from_cases", {
                "test_path": "test_price_x.py", "target_path": "pkg/price.py", "project": "p", "claude_writer": False,
                "test_command": "python3 -m unittest -q test_price_x",
                "cases": [{"name": "hundred", "given": "with_tax(100)", "expect": "120.0"}],
            })
            self.assertEqual(out["tasks"][0]["state"], "failed")
            self.assertIn("unittest", out["tasks"][0]["check_tail"])
            jm.shutdown()

    # ---------------------------------------------------------------- replicate_edit

    def test_replicate_edit_requires_a_hand_made_example_and_exact_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, store = _repo(tmp)
            srv, jm = self._server(root, store, ScriptedWriters({}))
            no_change = self._call(srv, "replicate_edit", {
                "example_path": "pkg/alpha.py", "targets": ["pkg/beta.py"], "project": "p"})
            self.assertEqual(no_change["status"], "invalid_request")
            self.assertIn("no uncommitted change", json.dumps(no_change))
            jm.shutdown()

            alpha = root / "pkg" / "alpha.py"
            alpha.write_text(alpha.read_text(encoding="utf-8").replace("LOG.info('start')", "LOG.info('start', extra={'run': 1})"),
                             encoding="utf-8")
            edit = {"find": "LOG.info('start')", "replace": "LOG.info('start', extra={'run': 1})"}
            writers = ScriptedWriters({
                "pkg/beta.py": [{"path": "pkg/beta.py", "content": "whole file rewrite\n"},
                                {"path": "pkg/beta.py", "edits": [edit]}],
                "pkg/gamma.py": [{"path": "pkg/gamma.py", "edits": [edit]}],
            })
            srv, jm = self._server(root, store, writers)
            out = self._call(srv, "replicate_edit", {
                "example_path": "pkg/alpha.py", "targets": ["pkg/beta.py", "pkg/gamma.py"], "project": "p",
                "check_command": "python3 -m py_compile pkg/beta.py pkg/gamma.py",
            })
            states = {t["task_id"]: (t["state"], t["attempts"]) for t in out["tasks"]}
            self.assertEqual(states["beta"], ("landed", 2))
            self.assertEqual(states["gamma"], ("landed", 1))
            self.assertTrue(out["acceptance_passed"])
            receipt = store.read_receipt(out["job_id"])
            self.assertEqual(receipt["kind"], "replicate_edit")
            self.assertNotIn("pkg/alpha.py", receipt["files_changed"])
            jm.shutdown()


if __name__ == "__main__":
    unittest.main()
