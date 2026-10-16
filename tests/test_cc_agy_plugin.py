"""Comprehensive unit and integration tests for cc-agy-plugin."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.pre_tool_guard import (
    classify_path,
    detect_bash_workspace_mutation,
    evaluate_pre_tool_use,
    lint_delegation_prompt,
)
from scripts.provenance_check import verify_workspace_provenance
from server.genai_ask import ask_gemini
from server.jobs import JobManager
from server.mcp_stdio import MCPServer, serve_stdio
from server.receipts import ReceiptStore, extract_diff_provenance, scrub_receipt_for_export
from server.regions import (
    estimate_cost_usd,
    estimate_tokens,
    resolve_vertex_location,
    resolve_vertex_project,
    truncate_to_token_budget,
)

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _init_git_repo(repo_dir: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
    )
    (repo_dir / "app.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
    )


class TestPluginManifestAndStructure(unittest.TestCase):
    """Verify Claude Code plugin packaging rules and frontmatter constraints."""

    def test_plugin_json_schema_and_name_rules(self) -> None:
        manifest_path = PLUGIN_ROOT / ".claude-plugin" / "plugin.json"
        self.assertTrue(manifest_path.is_file(), "Missing .claude-plugin/plugin.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        name = manifest["name"]
        self.assertEqual(name, "cc-agy-plugin")
        for forbidden_prefix in ("claude-", "anthropic-", "anthropics-", "cc-plugin-"):
            self.assertFalse(
                name.startswith(forbidden_prefix),
                f"Plugin name {name!r} must not start with reserved prefix {forbidden_prefix!r}",
            )

        # Component folders must live at plugin root, NOT inside .claude-plugin/
        for child in (PLUGIN_ROOT / ".claude-plugin").iterdir():
            self.assertEqual(child.name, "plugin.json")

        for required_path in (
            ".mcp.json",
            "hooks/hooks.json",
            "skills/vertex/SKILL.md",
            "agents/gemini-delegator.md",
            "commands/delegate.md",
            "commands/jobs.md",
            "commands/cost.md",
            "server/main.py",
        ):
            self.assertTrue(
                (PLUGIN_ROOT / required_path).exists(),
                f"Missing required plugin file: {required_path}",
            )

    def test_subagent_frontmatter_respects_claude_code_security_restrictions(self) -> None:
        agent_md = (PLUGIN_ROOT / "agents" / "gemini-delegator.md").read_text(encoding="utf-8")
        parts = agent_md.split("---", 2)
        self.assertGreaterEqual(len(parts), 3)
        frontmatter = parts[1]
        # Plugin subagents do not support hooks, mcpServers, or permissionMode
        for forbidden_field in ("hooks:", "mcpServers:", "permissionMode:"):
            self.assertNotIn(forbidden_field, frontmatter)
        self.assertIn("disallowedTools: Write, Edit, MultiEdit, NotebookEdit", frontmatter)

    def test_hooks_json_quotes_plugin_root_and_avoids_shell_user_config(self) -> None:
        hooks_raw = (PLUGIN_ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8")
        self.assertNotIn("${user_config.", hooks_raw)
        hooks_data = json.loads(hooks_raw)
        self.assertIn("hooks", hooks_data)
        for event_name in ("PreToolUse", "PostToolUse", "Stop"):
            self.assertIn(event_name, hooks_data["hooks"])
            for matcher_group in hooks_data["hooks"][event_name]:
                for hook in matcher_group["hooks"]:
                    self.assertIn('"${CLAUDE_PLUGIN_ROOT}"', hook["command"])


class TestRegionsAndCostAccounting(unittest.TestCase):
    """Verify automatic per-model Vertex region routing and token cost math."""

    def test_global_only_models_reroute_automatically(self) -> None:
        for model_id in (
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
        ):
            loc, note = resolve_vertex_location(model_id, "asia-south1")
            self.assertEqual(loc, "global")
            self.assertIsNotNone(note)
            self.assertIn("global", note or "")

    def test_regional_models_preserve_valid_region(self) -> None:
        loc, note = resolve_vertex_location("gemini-2.5-flash", "asia-south1")
        self.assertEqual(loc, "asia-south1")
        self.assertIsNone(note)

        with self.assertRaises(ValueError):
            resolve_vertex_location("gemini-2.5-flash", "invalid-region-99")

    def test_resolve_vertex_project_rejects_unsubstituted_template(self) -> None:
        with self.assertRaises(ValueError):
            resolve_vertex_project("${user_config.google_cloud_project}", env={})
        self.assertEqual(
            resolve_vertex_project(None, env={"GOOGLE_CLOUD_PROJECT": "my-gcp-proj"}),
            "my-gcp-proj",
        )

    def test_estimate_cost_usd_accounts_for_cached_and_thoughts_tokens(self) -> None:
        usage = {
            "prompt_token_count": 100_000,
            "cached_content_token_count": 60_000,
            "candidates_token_count": 5_000,
            "thoughts_token_count": 5_000,
        }
        # gemini-2.5-flash: input=$0.30/1M (40k uncached = $0.012),
        # cached=$0.03/1M (60k = $0.0018), output=$2.50/1M (10k = $0.025)
        cost = estimate_cost_usd("gemini-2.5-flash", usage)
        self.assertAlmostEqual(cost["input_cost_usd"], 0.012, places=6)
        self.assertAlmostEqual(cost["cached_cost_usd"], 0.0018, places=6)
        self.assertAlmostEqual(cost["output_cost_usd"], 0.025, places=6)
        self.assertAlmostEqual(cost["total_cost_usd"], 0.0388, places=6)
        self.assertAlmostEqual(cost["total_cost_cents"], 3.88, places=4)


class TestReceiptsAndCrashSafety(unittest.TestCase):
    """Verify finally-block receipt writing on error/timeout and scrubbing."""

    def test_ask_gemini_writes_receipt_on_backend_error_with_partial_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(tmp)

            def failing_backend(_req: dict) -> dict:
                err = RuntimeError("429 RESOURCE_EXHAUSTED mid-stream")
                setattr(
                    err,
                    "partial_usage",
                    {
                        "prompt_token_count": 4000,
                        "cached_content_token_count": 0,
                        "candidates_token_count": 200,
                        "thoughts_token_count": 100,
                        "total_token_count": 4300,
                    },
                )
                raise err

            with self.assertRaises(RuntimeError):
                ask_gemini(
                    "Review this architecture",
                    model="gemini-3.5-flash-lite",
                    project="secret-corp-proj-123",
                    location="asia-south1",
                    store=store,
                    backend=failing_backend,
                )

            receipts = store.list_receipts()
            self.assertEqual(len(receipts), 1)
            r = receipts[0]
            self.assertEqual(r["status"], "error")
            self.assertEqual(r["vertex_location"], "global")
            self.assertEqual(r["usage"]["prompt_token_count"], 4000)
            self.assertEqual(r["usage"]["thoughts_token_count"], 100)
            self.assertIn("429 RESOURCE_EXHAUSTED", r["error"])

            summary = store.summarize_costs()
            self.assertEqual(summary["failed_or_timed_out_jobs"], 1)
            self.assertEqual(summary["prompt_token_count"], 4000)
            self.assertGreater(summary["total_cost_usd"], 0.0)

    def test_scrub_receipt_redacts_project_and_workspace_paths(self) -> None:
        raw = {
            "job_id": "job-123",
            "vertex_project": "internal-billing-project-99",
            "worktree_dir": "/tmp/my_ws/.cc-agy-data/worktrees/job-123",
            "error": "Failed in /tmp/my_ws/src/main.py under project internal-billing-project-99",
        }
        scrubbed = scrub_receipt_for_export(
            raw,
            project_id="internal-billing-project-99",
            workspace_root="/tmp/my_ws",
        )
        self.assertEqual(scrubbed["vertex_project"], "<paying-project>")
        self.assertNotIn("internal-billing-project-99", json.dumps(scrubbed))
        self.assertIn("/workspace", scrubbed["worktree_dir"])


class TestWorktreeIsolationAndJobLifecycle(unittest.TestCase):
    """Verify git worktree isolation, scope enforcement, diff pagination, and apply."""

    def test_failed_worker_never_dirties_main_workspace_and_records_partial_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            store = ReceiptStore(repo_dir / ".cc-agy-data")

            def crashing_worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                (wt / "app.py").write_text("CORRUPTED BY CRASHING WORKER\n", encoding="utf-8")
                err = TimeoutError("Worker timed out after 300s")
                setattr(
                    err,
                    "partial_usage",
                    {
                        "prompt_token_count": 12000,
                        "cached_content_token_count": 8000,
                        "candidates_token_count": 500,
                        "thoughts_token_count": 900,
                        "total_token_count": 13400,
                    },
                )
                setattr(
                    err,
                    "partial_tool_calls",
                    [{"name": "edit_file", "args": {"path": "app.py"}}],
                )
                raise err

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=crashing_worker)
            digest = jm.delegate_start(
                "Refactor greet() in app.py",
                project="test-proj",
                wait=True,
            )
            self.assertEqual(digest["status"], "timeout")
            self.assertEqual(digest["tool_call_count"], 1)
            self.assertEqual(digest["usage"]["prompt_token_count"], 12000)

            # Main workspace file MUST remain completely untouched
            self.assertEqual(
                (repo_dir / "app.py").read_text(encoding="utf-8"),
                "def greet():\n    return 'hello'\n",
            )
            with self.assertRaises(ValueError):
                jm.delegate_apply(digest["job_id"])

    def test_scope_violation_blocks_out_of_slot_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            store = ReceiptStore(repo_dir / ".cc-agy-data")

            def out_of_scope_worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                (wt / "forbidden.py").write_text("x = 1\n", encoding="utf-8")
                return {
                    "text": "Edited forbidden.py",
                    "usage": {"prompt_token_count": 100, "candidates_token_count": 20},
                    "tool_calls": [{"name": "write_file", "args": {"path": "forbidden.py"}}],
                }

            jm = JobManager(
                workspace_root=repo_dir,
                store=store,
                worker_backend=out_of_scope_worker,
            )
            digest = jm.delegate_start(
                "Only edit app.py",
                allowed_paths=["app.py"],
                project="test-proj",
                wait=True,
            )
            self.assertEqual(digest["status"], "scope_violation")
            self.assertIn("forbidden.py", digest["scope_violations"])
            self.assertFalse((repo_dir / "forbidden.py").exists())
            with self.assertRaises(ValueError):
                jm.delegate_apply(digest["job_id"])

    def test_successful_worker_diff_pagination_apply_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            store = ReceiptStore(repo_dir / ".cc-agy-data")

            def good_worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                (wt / "app.py").write_text(
                    "def greet(name: str = 'world') -> str:\n    return f'hello {name}'\n",
                    encoding="utf-8",
                )
                return {
                    "text": "Updated greet() signature and return value.",
                    "usage": {
                        "prompt_token_count": 11600,
                        "cached_content_token_count": 10000,
                        "candidates_token_count": 150,
                        "thoughts_token_count": 200,
                    },
                    "tool_calls": [{"name": "edit_file", "args": {"path": "app.py"}}],
                }

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=good_worker)
            digest = jm.delegate_start(
                "Add optional name parameter to greet in app.py",
                allowed_paths=["app.py"],
                project="test-proj",
                wait=True,
            )
            self.assertEqual(digest["status"], "ok")
            self.assertEqual(digest["files_changed"], ["app.py"])

            # Main workspace is still unchanged before delegate_apply
            self.assertNotIn("world", (repo_dir / "app.py").read_text(encoding="utf-8"))

            diff_page = jm.get_diff(digest["job_id"], page=1, page_tokens=500)
            self.assertIn("+def greet(name: str = 'world') -> str:", diff_page["diff"])

            apply_res = jm.delegate_apply(digest["job_id"])
            self.assertTrue(apply_res["applied"])
            self.assertIn("world", (repo_dir / "app.py").read_text(encoding="utf-8"))

            # Stop-hook provenance verification must pass on the applied diff
            prov_check = verify_workspace_provenance(repo_dir, store=store)
            self.assertTrue(prov_check["ok"], f"Expected provenance ok, got: {prov_check}")

            # Now simulate an unverified manual edit by the host driver on top of app.py
            with open(repo_dir / "app.py", "a", encoding="utf-8") as f:
                f.write("# Sneaky unverified host line\n")
            prov_after_tamper = verify_workspace_provenance(repo_dir, store=store)
            self.assertFalse(prov_after_tamper["ok"])
            self.assertIn("app.py", prov_after_tamper["unverified_lines_by_file"])


class TestPreToolGuardAndTmpOrdering(unittest.TestCase):
    """Verify /tmp path ordering, extended Bash mutation guard, and 9-line dictation linter."""

    def test_classify_path_checks_workdir_and_outdir_before_tmp(self) -> None:
        # Simulate a git repo or worktree located under /tmp on Linux
        tmp_repo = Path("/tmp/cc-agy-test-workspace-123")
        tmp_out = tmp_repo / ".cc-agy-data"

        self.assertEqual(
            classify_path("/tmp/cc-agy-test-workspace-123/src/app.py", tmp_repo, tmp_out),
            "workspace",
        )
        self.assertEqual(
            classify_path("src/app.py", tmp_repo, tmp_out),
            "workspace",
        )
        self.assertEqual(
            classify_path("/tmp/cc-agy-test-workspace-123/.cc-agy-data/receipts/j1.json", tmp_repo, tmp_out),
            "out_dir",
        )
        self.assertEqual(
            classify_path("/tmp/unrelated-scratch-file.txt", tmp_repo, tmp_out),
            "tmp_scratch",
        )

    def test_bash_guard_blocks_cp_mv_python_inline_and_git_checkout_in_tmp_workspace(self) -> None:
        tmp_repo = "/tmp/cc-agy-test-workspace-123"
        out_dir = "/tmp/cc-agy-test-workspace-123/.cc-agy-data"

        blocked_commands = [
            "cp /tmp/other.py src/app.py",
            "mv src/app.py src/app_bak.py",
            "rm -f src/app.py",
            "touch src/new_file.py",
            "install -m 644 /tmp/a.py src/a.py",
            "rsync -a /tmp/stage/ src/",
            "ln -s /tmp/a.py src/link.py",
            "truncate -s 0 src/app.py",
            "python3 -c \"open('src/app.py', 'w').write('x')\"",
            "node -e \"fs.writeFileSync('src/app.py', 'x')\"",
            "perl -pi -e 's/foo/bar/g' src/app.py",
            "sed -i 's/foo/bar/' src/app.py",
            "git checkout -- src/app.py",
            "git restore src/app.py",
            "git reset --hard HEAD~1",
            "git clean -fd",
            "echo 'x = 1' > src/app.py",
            "cat << 'EOF' > src/app.py\nx=1\nEOF",
        ]
        for cmd in blocked_commands:
            verdict = detect_bash_workspace_mutation(cmd, tmp_repo, out_dir)
            self.assertTrue(
                verdict["blocked"],
                f"Expected command to be blocked in strict mode: {cmd!r}",
            )

        allowed_commands = [
            "pytest -q",
            "git status --short",
            "git diff HEAD",
            "ls -la src/",
            "cp src/app.py /tmp/external-scratch.py",
            "echo 'ok' > /tmp/external-log.txt",
        ]
        for cmd in allowed_commands:
            verdict = detect_bash_workspace_mutation(cmd, tmp_repo, out_dir)
            self.assertFalse(
                verdict["blocked"],
                f"Expected read-only or external-tmp command to be allowed: {cmd!r} ({verdict['reason']})",
            )

    def test_dictation_linter_enforces_8_line_ceiling(self) -> None:
        short_prompt = (
            "Update `greet()` in `app.py` to accept an optional `name` parameter.\n"
            "```python\n"
            "def greet(name: str = 'world') -> str:\n"
            "    return f'hello {name}'\n"
            "```\n"
        )
        self.assertTrue(lint_delegation_prompt(short_prompt)["ok"])

        nine_line_code = "\n".join(f"x_{i} = {i}" for i in range(9))
        long_prompt = f"Paste this exact code into app.py:\n```python\n{nine_line_code}\n```"
        lint_res = lint_delegation_prompt(long_prompt)
        self.assertFalse(lint_res["ok"])
        self.assertEqual(lint_res["max_consecutive_code_lines"], 9)

        hook_out = evaluate_pre_tool_use(
            {
                "tool_name": "mcp__plugin_cc-agy-plugin_vertex__delegate_start",
                "tool_input": {"prompt": long_prompt},
                "cwd": "/tmp/ws",
            },
            env={"CC_AGY_STRICT_MODE": "false"},
        )
        self.assertEqual(
            hook_out["hookSpecificOutput"]["permissionDecision"],
            "deny",
        )


class TestMCPStdioServer(unittest.TestCase):
    """Verify JSON-RPC 2.0 MCP stdio protocol over both newline and Content-Length framing."""

    def test_mcp_stdio_roundtrip_ask_and_delegate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            store = ReceiptStore(repo_dir / ".cc-agy-data")

            def fake_ask(req: dict) -> dict:
                return {
                    "text": f"Reviewed on {req['location']} using {req['model']}",
                    "usage": {
                        "prompt_token_count": 250,
                        "cached_content_token_count": 0,
                        "candidates_token_count": 50,
                        "thoughts_token_count": 25,
                        "total_token_count": 325,
                    },
                }

            def fake_worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                (wt / "app.py").write_text("def greet():\n    return 'hi from vertex'\n", encoding="utf-8")
                return {
                    "text": "Updated app.py in isolated worktree.",
                    "usage": {
                        "prompt_token_count": 11554,
                        "cached_content_token_count": 9000,
                        "candidates_token_count": 120,
                        "thoughts_token_count": 180,
                    },
                    "tool_calls": [{"name": "edit_file", "args": {"path": "app.py"}}],
                }

            old_proj = os.environ.get("GOOGLE_CLOUD_PROJECT")
            os.environ["GOOGLE_CLOUD_PROJECT"] = "unit-test-proj"
            try:
                jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=fake_worker)
                mcp = MCPServer(job_manager=jm, store=store, ask_backend=fake_ask)

                requests = [
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": "ask_gemini",
                            "arguments": {
                                "prompt": "Check app.py",
                                "model": "gemini-3.5-flash-lite",
                                "location": "asia-south1",
                                "context_files": ["app.py"],
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "tools/call",
                        "params": {
                            "name": "delegate_to_gemini",
                            "arguments": {
                                "prompt": "Update greet() in app.py to return 'hi from vertex'",
                                "allowed_paths": ["app.py"],
                                "auto_apply": True,
                            },
                        },
                    },
                ]
                in_buf = io.BytesIO(
                    b"".join((json.dumps(r) + "\n").encode("utf-8") for r in requests)
                )
                out_buf = io.BytesIO()
                serve_stdio(server=mcp, in_stream=in_buf, out_stream=out_buf)

                lines = [
                    json.loads(line.decode("utf-8"))
                    for line in out_buf.getvalue().splitlines()
                    if line.strip()
                ]
                # 4 requests had an id (the notification produces no response)
                self.assertEqual(len(lines), 4)

                tools_list = lines[1]["result"]["tools"]
                tool_names = {t["name"] for t in tools_list}
                self.assertIn("ask_gemini", tool_names)
                self.assertIn("delegate_start", tool_names)
                self.assertIn("delegate_apply", tool_names)
                self.assertIn("delegate_to_gemini", tool_names)

                ask_payload = json.loads(lines[2]["result"]["content"][0]["text"])
                self.assertEqual(ask_payload["vertex_location"], "global")
                self.assertIn("Reviewed on global", ask_payload["text"])

                del_payload = json.loads(lines[3]["result"]["content"][0]["text"])
                self.assertEqual(del_payload["status"], "ok")
                self.assertTrue(del_payload["applied"])
                self.assertIn("hi from vertex", (repo_dir / "app.py").read_text(encoding="utf-8"))
            finally:
                if old_proj is None:
                    os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
                else:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = old_proj


if __name__ == "__main__":
    unittest.main()
