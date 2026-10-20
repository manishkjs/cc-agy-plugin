"""Tests for single-shot writers: answer parsing/application and the fallback ladder (server/writers.py)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from server.batch_plan import FileTask
from server.writers import (
    AnswerRejected,
    LadderOutcome,
    WriterAuthError,
    WriterReply,
    WriterTransportError,
    WRITER_LADDER,
    apply_writer_answer,
    classify_status,
    run_writer_ladder,
)


def _task(action: str = "modify", path: str = "pkg/mod.py") -> FileTask:
    return FileTask(task_id="t1", path=path, action=action, intent="Make it better.")


class ApplyAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "pkg").mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_create_writes_new_content(self) -> None:
        out = apply_writer_answer(_task("create", "pkg/new.py"), json.dumps({"path": "pkg/new.py", "content": "X = 1\n"}), self.root)
        self.assertEqual(out.new_bytes, b"X = 1\n")

    def test_modify_edits_must_match_exactly_once(self) -> None:
        (self.root / "pkg" / "mod.py").write_text("a = 1\nb = 1\n", encoding="utf-8")
        ok = apply_writer_answer(_task(), {"path": "pkg/mod.py", "edits": [{"find": "a = 1", "replace": "a = 2"}]}, self.root)
        self.assertEqual(ok.new_bytes, b"a = 2\nb = 1\n")
        with self.assertRaises(AnswerRejected) as twice:
            apply_writer_answer(_task(), {"path": "pkg/mod.py", "edits": [{"find": "= 1", "replace": "= 3"}]}, self.root)
        self.assertIn("2 times", str(twice.exception))
        with self.assertRaises(AnswerRejected) as never:
            apply_writer_answer(_task(), {"path": "pkg/mod.py", "edits": [{"find": "zzz", "replace": "y"}]}, self.root)
        self.assertIn("0 times", str(never.exception))

    def test_edits_apply_in_sequence_against_the_updated_text(self) -> None:
        (self.root / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
        out = apply_writer_answer(_task(), {"path": "pkg/mod.py", "edits": [
            {"find": "x = 1", "replace": "x = 2"}, {"find": "x = 2", "replace": "x = 3"}]}, self.root)
        self.assertEqual(out.new_bytes, b"x = 3\n")

    def test_crlf_files_keep_crlf(self) -> None:
        (self.root / "pkg" / "mod.py").write_bytes(b"line1\r\nline2\r\n")
        out = apply_writer_answer(_task(), {"path": "pkg/mod.py", "edits": [{"find": "line1\nline2", "replace": "one\ntwo"}]}, self.root)
        self.assertEqual(out.new_bytes, b"one\r\ntwo\r\n")
        whole = apply_writer_answer(_task(), {"path": "pkg/mod.py", "content": "a\nb\n"}, self.root)
        self.assertEqual(whole.new_bytes, b"a\r\nb\r\n")

    def test_rejects_without_repair(self) -> None:
        (self.root / "pkg" / "mod.py").write_text("a\n", encoding="utf-8")
        cases = [
            ('```json\n{"path": "pkg/mod.py", "content": "b"}\n```', "JSON"),        # fenced: no stripping
            ({"path": "pkg/other.py", "content": "b"}, "path"),                         # wrong path
            ({"path": "../pkg/mod.py", "content": "b"}, "path"),                        # unsafe
            ({"path": "pkg/mod.py", "content": ""}, "empty"),
            ({"path": "pkg/mod.py"}, "content"),                                        # nothing to do
            ({"path": "pkg/mod.py", "content": "b", "edits": [{"find": "a", "replace": "b"}]}, "both"),
            ({"path": "pkg/mod.py", "edits": [{"find": "", "replace": "b"}]}, "find"),
            ([1, 2], "object"),
        ]
        for raw, needle in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(AnswerRejected) as ctx:
                    apply_writer_answer(_task(), raw if not isinstance(raw, str) else raw, self.root)
                self.assertIn(needle, str(ctx.exception))

    def test_create_must_not_use_edits_and_edits_only_mode(self) -> None:
        with self.assertRaises(AnswerRejected):
            apply_writer_answer(_task("create", "pkg/n.py"), {"path": "pkg/n.py", "edits": [{"find": "a", "replace": "b"}]}, self.root)
        (self.root / "pkg" / "mod.py").write_text("a\n", encoding="utf-8")
        with self.assertRaises(AnswerRejected) as ctx:
            apply_writer_answer(_task(), {"path": "pkg/mod.py", "content": "b\n"}, self.root, edits_only=True)
        self.assertIn("edits", str(ctx.exception))


class StatusClassificationTests(unittest.TestCase):
    def test_codes(self) -> None:
        for code in (408, 429, 500, 502, 503, 504, 529):
            self.assertEqual(classify_status(code), "transient")
        for code in (401, 403):
            self.assertEqual(classify_status(code), "auth")
        for code in (400, 404, 422, None):
            self.assertEqual(classify_status(code), "other")


class LadderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sleeps: list[float] = []

    def _run(self, writers, verify, task=None) -> LadderOutcome:
        return run_writer_ladder(
            task or _task("create", "pkg/n.py"),
            routed="genai",
            writers=writers,
            verify=verify,
            sleep=self.sleeps.append,
            jitter=lambda: 1.0,
        )

    def test_ladder_shape(self) -> None:
        self.assertEqual(WRITER_LADDER, ("routed", "routed", "claude"))

    def test_first_attempt_lands(self) -> None:
        calls = []

        def genai(req):
            calls.append(req)
            return WriterReply(text='{"path": "pkg/n.py", "content": "X = 1\\n"}', usage={"prompt_token_count": 10})

        out = self._run({"genai": genai, "claude": None}, verify=lambda answer: (True, ""))
        self.assertEqual((out.state, out.attempts, out.writer_used), ("landed", 1, "genai"))
        self.assertEqual(len(calls), 1)

    def test_second_attempt_gets_failure_tail_then_claude_lands_third(self) -> None:
        seen_tails = []

        def genai(req):
            seen_tails.append(req.failure_tail)
            return WriterReply(text='{"path": "pkg/n.py", "content": "bad\\n"}')

        def claude(req):
            seen_tails.append(req.failure_tail)
            return WriterReply(text='{"path": "pkg/n.py", "content": "good\\n"}')

        def verify(answer):
            return (answer.new_bytes == b"good\n", "AssertionError: expected good")

        out = self._run({"genai": genai, "claude": claude}, verify)
        self.assertEqual((out.state, out.attempts, out.writer_used), ("landed", 3, "claude"))
        self.assertEqual(seen_tails[0], "")
        self.assertIn("expected good", seen_tails[1])
        self.assertIn("expected good", seen_tails[2])

    def test_transient_failures_wait_and_do_not_count(self) -> None:
        state = {"n": 0}

        def genai(req):
            state["n"] += 1
            if state["n"] <= 3:
                raise WriterTransportError(429 if state["n"] < 3 else 503)
            return WriterReply(text='{"path": "pkg/n.py", "content": "ok\\n"}')

        out = self._run({"genai": genai, "claude": None}, verify=lambda a: (True, ""))
        self.assertEqual((out.state, out.attempts), ("landed", 1))
        self.assertEqual(len(self.sleeps), 3)
        self.assertTrue(all(0 < s <= 60 for s in self.sleeps))
        self.assertLess(self.sleeps[0], self.sleeps[2])

    def test_backoff_is_capped_and_bounded(self) -> None:
        def genai(req):
            raise WriterTransportError(503)

        out = self._run({"genai": genai, "claude": None}, verify=lambda a: (True, ""))
        self.assertEqual(out.state, "failed")
        self.assertEqual(out.attempts, 0)
        self.assertIn("unavailable", out.detail)
        self.assertTrue(all(s <= 60 for s in self.sleeps))

    def test_cut_off_reply_skips_to_next_writer_without_counting(self) -> None:
        calls = []

        def genai(req):
            calls.append("genai")
            return WriterReply(text='{"path": "pkg/n.py", "con', cut_off=True)

        def claude(req):
            calls.append("claude")
            return WriterReply(text='{"path": "pkg/n.py", "content": "ok\\n"}')

        out = self._run({"genai": genai, "claude": claude}, verify=lambda a: (True, ""))
        self.assertEqual(calls, ["genai", "claude"])
        self.assertEqual((out.state, out.attempts, out.writer_used), ("landed", 1, "claude"))

    def test_auth_failure_stops_everything(self) -> None:
        def genai(req):
            raise WriterAuthError(403, "Vertex AI API not enabled for project p")

        with self.assertRaises(WriterAuthError) as ctx:
            self._run({"genai": genai, "claude": None}, verify=lambda a: (True, ""))
        self.assertIn("403", str(ctx.exception))

    def test_rejected_answers_count_as_attempts(self) -> None:
        def genai(req):
            return WriterReply(text="not json")

        out = self._run({"genai": genai, "claude": genai}, verify=lambda a: (True, ""))
        self.assertEqual((out.state, out.attempts), ("failed", 3))
        self.assertIn("JSON", out.detail)

    def test_missing_claude_writer_ends_after_two(self) -> None:
        def genai(req):
            return WriterReply(text='{"path": "pkg/n.py", "content": "x\\n"}')

        out = self._run({"genai": genai}, verify=lambda a: (False, "boom"))
        self.assertEqual((out.state, out.attempts), ("failed", 2))
        self.assertIn("boom", out.detail)


if __name__ == "__main__":
    unittest.main()
