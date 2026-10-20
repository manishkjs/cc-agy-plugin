"""Comprehensive unit and adversarial regression tests for cc-agy-plugin."""

from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import unittest
import unittest.mock

from scripts.pre_tool_guard import (
    classify_path,
    detect_bash_workspace_mutation,
    evaluate_pre_tool_use,
    lint_delegation_prompt,
)
from scripts.provenance_check import evaluate_stop_hook, verify_workspace_provenance
from server.genai_ask import ask_gemini
from server.jobs import JobManager, validate_allowed_paths
from server.mcp_stdio import MCPServer, parse_bool, serve_stdio
from server.receipts import (
    ReceiptStore,
    extract_diff_provenance,
    scrub_receipt_for_export,
    validate_job_id,
)
from server.regions import (
    estimate_cost_usd,
    normalize_model_id,
    paginate_text_by_tokens,
    resolve_api_key,
    resolve_auth_config,
    resolve_vertex_location,
    resolve_vertex_project,
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
    """Verify Claude Code plugin packaging rules, matchers, and frontmatter constraints."""

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
        for forbidden_field in ("hooks:", "mcpServers:", "permissionMode:"):
            self.assertNotIn(forbidden_field, frontmatter)
        self.assertIn("disallowedTools: Write, Edit, MultiEdit, NotebookEdit", frontmatter)

    def test_hooks_json_matchers_cover_both_host_tools_and_mcp_delegate_tools(self) -> None:
        hooks_raw = (PLUGIN_ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8")
        self.assertNotIn("${user_config.", hooks_raw)
        hooks_data = json.loads(hooks_raw)
        self.assertIn("hooks", hooks_data)

        pre_matchers = [g["matcher"] for g in hooks_data["hooks"]["PreToolUse"]]
        # Verify at least one PreToolUse matcher matches the full Claude Code MCP delegate tool names
        for mcp_tool in (
            "mcp__plugin_cc-agy-plugin_vertex__delegate_start",
            "mcp__plugin_cc-agy-plugin_vertex__delegate_to_gemini",
            "mcp__plugin_cc-agy-plugin_vertex__delegate_continue",
            "mcp__vertex__delegate_start",
        ):
            matched = any(re.fullmatch(m, mcp_tool) for m in pre_matchers)
            self.assertTrue(matched, f"No PreToolUse matcher matched {mcp_tool!r}")


class TestRegionsAndCostAccounting(unittest.TestCase):
    """Verify automatic per-model Vertex region routing, normalization, and token cost math."""

    def test_global_only_models_and_full_resource_paths_reroute_automatically(self) -> None:
        for model_id in (
            "gemini-3.5-flash-lite",
            "publishers/google/models/gemini-3.5-flash-lite",
            "projects/p/locations/us-central1/publishers/google/models/gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
        ):
            loc, note = resolve_vertex_location(model_id, "ASIA-SOUTH1")
            self.assertEqual(loc, "global")
            self.assertIsNotNone(note)

    def test_regional_models_and_case_insensitive_regions(self) -> None:
        loc, note = resolve_vertex_location("gemini-2.5-flash", "US-CENTRAL1")
        self.assertEqual(loc, "us-central1")
        self.assertIsNone(note)

        # Regional preview models not in GLOBAL_ONLY_MODELS stay in their valid region
        loc2, note2 = resolve_vertex_location("gemini-2.5-flash-preview", "us-central1")
        self.assertEqual(loc2, "us-central1")
        self.assertIsNone(note2)

        with self.assertRaises(ValueError):
            resolve_vertex_location("gemini-2.5-flash", "invalid-region-99")

    def test_resolve_vertex_project_rejects_unsubstituted_and_scrubbed_placeholders(self) -> None:
        with self.assertRaises(ValueError):
            resolve_vertex_project("${user_config.google_cloud_project}", env={})
        with self.assertRaises(ValueError):
            resolve_vertex_project("<paying-project>", env={})
        self.assertEqual(
            resolve_vertex_project("<paying-project>", env={"GOOGLE_CLOUD_PROJECT": "my-gcp-proj"}),
            "my-gcp-proj",
        )

    def test_estimate_cost_usd_handles_resource_prefixes_unknown_models_and_long_context(self) -> None:
        usage = {
            "prompt_token_count": 100_000,
            "cached_content_token_count": 60_000,
            "candidates_token_count": 5_000,
            "thoughts_token_count": 5_000,
        }
        cost = estimate_cost_usd("publishers/google/models/gemini-2.5-flash", usage)
        self.assertTrue(cost["pricing_known"])
        self.assertAlmostEqual(cost["total_cost_usd"], 0.0388, places=6)

        flash38_cost = estimate_cost_usd("gemini-3.8-flash", usage)
        self.assertTrue(flash38_cost["pricing_known"])
        self.assertAlmostEqual(flash38_cost["total_cost_usd"], 0.0530, places=6)

        unknown_cost = estimate_cost_usd("custom-experimental-model-v9", usage)
        self.assertFalse(unknown_cost["pricing_known"])

        # >200k long-context multiplier on Pro/Argon models applies per request (2.0x input, 1.5x output)
        long_usage = {"prompt_token_count": 250_000, "candidates_token_count": 10_000}
        pro_single_cost = estimate_cost_usd("gemini-3.1-pro-preview", long_usage, single_request=True)
        self.assertAlmostEqual(pro_single_cost["input_cost_usd"], 250_000 / 1e6 * 2.0 * 2.0, places=6)
        self.assertAlmostEqual(pro_single_cost["output_cost_usd"], 10_000 / 1e6 * 12.0 * 1.5, places=6)

        # Aggregated multi-turn worker usage (e.g. 1M input + 100k output across 25 turns) uses standard tier
        multi_turn_usage = {"prompt_token_count": 1_000_000, "candidates_token_count": 100_000}
        argon_multi_cost = estimate_cost_usd("gemini-4-argon", multi_turn_usage)
        self.assertAlmostEqual(argon_multi_cost["total_cost_usd"], 3.20, places=6)


class TestReceiptsAndCrashSafety(unittest.TestCase):
    """Verify finally-block receipt writing, HMAC verification, job_id validation, and scrubbing."""

    def test_validate_job_id_blocks_traversal_and_dots(self) -> None:
        for bad_id in ("", ".", "..", "./id", "../evil", "a/b", "a\\b", "-flag"):
            with self.assertRaises(ValueError):
                validate_job_id(bad_id)
        self.assertEqual(validate_job_id("job-123_abc"), "job-123_abc")

    def test_ask_gemini_writes_receipt_on_backend_error_and_blocks_sibling_prefix_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ws = base / "repo"
            sibling = base / "repo-secrets"
            ws.mkdir()
            sibling.mkdir()
            (sibling / "creds.txt").write_text("TOP_SECRET_TOKEN", encoding="utf-8")
            store = ReceiptStore(base / "data", workspace_root=ws)

            # Sibling prefix traversal (../repo-secrets/creds.txt) must be rejected
            with self.assertRaises(ValueError):
                ask_gemini(
                    "Read this",
                    context_files=["../repo-secrets/creds.txt"],
                    project="proj-1",
                    workspace_root=ws,
                    store=store,
                    backend=lambda _r: {"text": "ok"},
                )

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
                    workspace_root=ws,
                    store=store,
                    backend=failing_backend,
                )

            receipts = store.list_receipts()
            self.assertEqual(len(receipts), 1)
            r = receipts[0]
            self.assertTrue(store.verify_receipt_signature(r))
            self.assertEqual(r["status"], "error")
            self.assertEqual(r["vertex_location"], "global")
            self.assertEqual(r["usage"]["prompt_token_count"], 4000)

    def test_updated_at_epoch_advances_on_rewrite_and_hmac_detects_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(tmp)
            r1 = store.write_receipt("job-1", {"status": "running", "updated_at_epoch": 100.0})
            self.assertGreater(r1["updated_at_epoch"], 100.0)
            time.sleep(0.01)
            r2 = store.write_receipt("job-1", r1)
            self.assertGreaterEqual(r2["updated_at_epoch"], r1["updated_at_epoch"])
            self.assertTrue(store.verify_receipt_signature(r2))

            tampered = dict(r2)
            tampered["applied"] = True
            self.assertFalse(store.verify_receipt_signature(tampered))


class TestWorktreeIsolationAndAdversarialEdgeCases(unittest.TestCase):
    """Verify standalone repo isolation, conflict detection, renames, deletes, symlinks, and continue."""

    def test_worker_cannot_corrupt_host_git_hooks_or_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo_dir)

            def evil_git_worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                # Write a hook inside the worker's own .git/hooks and also try to touch host .git/hooks
                (wt / ".git" / "hooks" / "pre-commit").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
                return {"text": "done", "usage": {"prompt_token_count": 10}}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=evil_git_worker)
            digest = jm.delegate_start("Try modifying git hooks", project="p", wait=True)
            self.assertEqual(digest["status"], "ok")
            self.assertFalse((repo_dir / ".git" / "hooks" / "pre-commit").exists())

            # Now test a worker that directly writes to the host's .git/hooks/pre-commit
            def direct_host_tamper_worker(_req: dict) -> dict:
                (repo_dir / ".git" / "hooks" / "pre-commit").write_text("evil", encoding="utf-8")
                return {"text": "tampered"}

            jm2 = JobManager(workspace_root=repo_dir, store=store, worker_backend=direct_host_tamper_worker)
            digest2 = jm2.delegate_start("Tamper host git", project="p", wait=True)
            self.assertEqual(digest2["status"], "security_violation")
            self.assertFalse((repo_dir / ".git" / "hooks" / "pre-commit").exists())

    def test_worker_git_commit_renames_deletes_and_binary_files_captured_and_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            (repo_dir / "old_name.py").write_text("x = 1\n", encoding="utf-8")
            (repo_dir / "to_delete.py").write_text("delete_me = True\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), check=True)
            subprocess.run(["git", "commit", "-m", "add files"], cwd=str(repo_dir), check=True)
            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo_dir)

            def committing_and_renaming_worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                subprocess.run(["git", "mv", "old_name.py", "new_name.py"], cwd=str(wt), check=True)
                subprocess.run(["git", "rm", "to_delete.py"], cwd=str(wt), check=True)
                (wt / "binary.dat").write_bytes(b"\x00\x01\x02\xff\xfe")
                subprocess.run(["git", "add", "-A"], cwd=str(wt), check=True)
                subprocess.run(["git", "commit", "-m", "worker commit"], cwd=str(wt), check=True)
                return {"text": "Committed rename, delete, and binary file"}

            jm = JobManager(
                workspace_root=repo_dir,
                store=store,
                worker_backend=committing_and_renaming_worker,
            )
            digest = jm.delegate_start("Rename, delete, and add binary", project="p", wait=True)
            self.assertEqual(digest["status"], "ok")
            self.assertIn("old_name.py", digest["files_changed"])
            self.assertIn("new_name.py", digest["files_changed"])
            self.assertIn("to_delete.py", digest["files_changed"])
            self.assertIn("binary.dat", digest["files_changed"])

            # Test fallback apply from stored binary diff when worktree is already removed
            shutil.rmtree(store.worktree_path(digest["job_id"]))
            apply_res = jm.delegate_apply(digest["job_id"])
            self.assertTrue(apply_res["applied"])
            self.assertFalse((repo_dir / "old_name.py").exists())
            self.assertFalse((repo_dir / "to_delete.py").exists())
            self.assertTrue((repo_dir / "new_name.py").exists())
            self.assertEqual((repo_dir / "binary.dat").read_bytes(), b"\x00\x01\x02\xff\xfe")

            # Calling delegate_apply a second time is idempotent
            reapply_res = jm.delegate_apply(digest["job_id"])
            self.assertTrue(reapply_res["already_applied"])

            # Provenance check passes on renamed, deleted, and binary files
            prov = verify_workspace_provenance(repo_dir, store=store)
            self.assertTrue(prov["ok"], f"Expected provenance ok, got: {prov}")

    def test_delegate_apply_detects_host_conflict_and_refuses_to_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo_dir)

            def worker_edit(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                (wt / "app.py").write_text("def greet():\n    return 'worker edit'\n", encoding="utf-8")
                return {"text": "edited"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_edit)
            digest = jm.delegate_start("Edit app.py", project="p", wait=True)

            # Simulate conflicting user edit on the same line of app.py before apply
            (repo_dir / "app.py").write_text("def greet():\n    return 'conflicting user edit'\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                jm.delegate_apply(digest["job_id"])
            # User edit is preserved!
            self.assertIn("conflicting user edit", (repo_dir / "app.py").read_text(encoding="utf-8"))

    def test_symlink_escape_in_worktree_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            secret_file = Path(tmp) / "secret_key.pem"
            secret_file.write_text("PRIVATE_KEY_DATA", encoding="utf-8")
            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo_dir)

            def symlink_worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                os.symlink(str(secret_file), str(wt / "leaked.pem"))
                return {"text": "created symlink"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=symlink_worker)
            digest = jm.delegate_start("Create symlink", project="p", wait=True)
            self.assertEqual(digest["status"], "scope_violation")
            self.assertIn("leaked.pem", digest["scope_violations"])
            with self.assertRaises(ValueError):
                jm.delegate_apply(digest["job_id"])

    def test_delegate_continue_sends_only_followup_and_preserves_settings_and_scrubbed_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo_dir)
            calls: list[dict] = []

            def tracking_worker(req: dict) -> dict:
                calls.append(dict(req))
                wt = Path(req["workspace_dir"])
                with open(wt / "app.py", "a", encoding="utf-8") as f:
                    f.write(f"# turn {len(calls)}\n")
                return {"text": f"turn {len(calls)}"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=tracking_worker)
            d1 = jm.delegate_start(
                "Turn 1 prompt",
                project="real-secret-project",
                enabled_tools=["edit_file"],
                scrub_receipt=True,
                wait=True,
            )
            self.assertEqual(len(calls), 1)
            # The lead's brief is sent first, followed by the plugin's budget and reporting sections.
            self.assertEqual(len(calls[0]["prompts"]), 1)
            self.assertTrue(calls[0]["prompts"][0].startswith("Turn 1 prompt"))
            self.assertIn("## Budget", calls[0]["prompts"][0])
            self.assertEqual(store.read_receipt(d1["job_id"])["prompts"], ["Turn 1 prompt"])
            saved_r1 = store.read_receipt(d1["job_id"])
            self.assertEqual(saved_r1["vertex_project"], "<paying-project>")

            d2 = jm.delegate_continue(d1["job_id"], "Turn 2 follow-up", wait=True)
            self.assertEqual(d2["status"], "ok")
            self.assertEqual(len(calls), 2)
            # A single prompt is sent on Turn 2 with prior context summary + follow-up, and real project/tools preserved!
            self.assertEqual(len(calls[1]["prompts"]), 1)
            self.assertIn("Turn 1 prompt", calls[1]["prompts"][0])
            self.assertIn("Turn 2 follow-up", calls[1]["prompts"][0])
            self.assertEqual(calls[1]["project"], "real-secret-project")
            self.assertEqual(calls[1]["enabled_tools"], ["edit_file"])

            jm.delegate_apply(d1["job_id"])
            # Continuing an already-applied job must raise ValueError so provenance is never clobbered
            with self.assertRaises(ValueError):
                jm.delegate_continue(d1["job_id"], "Turn 3 after apply")

    def test_subdirectory_workspace_root_and_preexisting_dirty_baseline_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            sub_dir = repo_dir / "packages" / "api"
            sub_dir.mkdir(parents=True)
            (sub_dir / "handler.py").write_text("v = 1\n", encoding="utf-8")
            (repo_dir / "user_dirty.py").write_text("clean = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), check=True)
            subprocess.run(["git", "commit", "-m", "add subdir"], cwd=str(repo_dir), check=True)

            # User has uncommitted local edits in user_dirty.py BEFORE JobManager starts
            (repo_dir / "user_dirty.py").write_text("dirty_before_session = 2\n", encoding="utf-8")
            store = ReceiptStore(Path(tmp) / "data", workspace_root=sub_dir)

            def sub_worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                (wt / "packages" / "api" / "handler.py").write_text("v = 2\n", encoding="utf-8")
                return {"text": "updated handler"}

            # Initialize JobManager from subdirectory `sub_dir`
            jm = JobManager(workspace_root=sub_dir, store=store, worker_backend=sub_worker)
            res = jm.delegate_start("Update handler", project="p", wait=True)
            jm.delegate_apply(res["job_id"])

            self.assertEqual((sub_dir / "handler.py").read_text(encoding="utf-8"), "v = 2\n")
            self.assertFalse((sub_dir / "packages").exists())

            # Provenance check must pass even though user_dirty.py had pre-existing dirty edits!
            prov = verify_workspace_provenance(sub_dir, store=store)
            self.assertTrue(prov["ok"], f"Expected dirty-baseline provenance ok, got: {prov}")

            # Now if the host creates an untracked backdoor.py or deletes a line in handler.py, provenance MUST fail
            (repo_dir / "backdoor.py").write_text("evil = 1\n", encoding="utf-8")
            prov_untracked = verify_workspace_provenance(repo_dir, store=store)
            self.assertFalse(prov_untracked["ok"])
            self.assertIn("backdoor.py", prov_untracked["unverified_files"])
            (repo_dir / "backdoor.py").unlink()

            (sub_dir / "handler.py").write_text("", encoding="utf-8")
            prov_deleted_line = verify_workspace_provenance(repo_dir, store=store)
            self.assertFalse(prov_deleted_line["ok"])
            self.assertIn("packages/api/handler.py", prov_deleted_line["unverified_deletions_by_file"])


class TestPreToolGuardAndTmpOrdering(unittest.TestCase):
    """Verify /tmp path ordering, extended Bash mutation guard, and 9-line dictation linter."""

    def test_classify_path_checks_workdir_and_outdir_before_tmp_and_handles_cwd_drift(self) -> None:
        tmp_repo = Path("/tmp/cc-agy-test-workspace-123")
        tmp_out = tmp_repo / ".cc-agy-data"

        self.assertEqual(
            classify_path("/tmp/cc-agy-test-workspace-123/src/app.py", tmp_repo, tmp_out),
            "workspace",
        )
        # Simulate cwd drift: Claude ran `cd src`, then targeted `../setup.py`
        self.assertEqual(
            classify_path(
                "../setup.py",
                tmp_repo,
                tmp_out,
                cwd=tmp_repo / "src",
                env={"CLAUDE_PROJECT_DIR": str(tmp_repo)},
            ),
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

    def test_bash_guard_blocks_bypass_vectors_and_allows_readonly_commands(self) -> None:
        tmp_repo = "/tmp/cc-agy-test-workspace-123"
        out_dir = "/tmp/cc-agy-test-workspace-123/.cc-agy-data"

        blocked_commands = [
            "cp /tmp/other.py src/app.py",
            "cp -t src /tmp/other.py",
            "mv src/app.py src/app_bak.py",
            "rm -f src/app.py",
            "touch src/new_file.py",
            "install -m 644 /tmp/a.py src/a.py",
            "rsync -a /tmp/stage/ src/",
            "ln -s /tmp/a.py src/link.py",
            "truncate -s 0 src/app.py",
            "python3 -c \"open('src/app.py', 'w').write('x')\"",
            "python3 /tmp/writer.py",
            "node -e \"fs.writeFileSync('src/app.py', 'x')\"",
            "perl -pi -e 's/foo/bar/g' src/app.py",
            "sed -i 's/foo/bar/' src/app.py",
            "git checkout -- src/app.py",
            "git -C . checkout -- src/app.py",
            "git switch -c feature",
            "git rm -f src/app.py",
            "git mv src/app.py src/b.py",
            "git commit -am wip",
            "git push --force origin HEAD",
            "git pull",
            "echo 'x = 1' > src/app.py",
            "echo 'x = 1' >| src/app.py",
            "cat << 'EOF' > src/app.py\nx=1\nEOF",
            "sh -c 'rm src/app.py'",
            "bash -c 'cp /tmp/x src/app.py'",
            "eval 'rm src/app.py'",
            "(rm src/app.py)",
            "{ rm src/app.py; }",
            "exec rm src/app.py",
            "time rm src/app.py",
            "sudo -u root rm src/app.py",
            "printf src/app.py | xargs rm",
            "find src -name '*.py' -delete",
            "curl -sL https://example.com/a.py -o src/app.py",
            "wget -O src/app.py https://example.com/a.py",
            "tar -xf /tmp/a.tar -C src",
            "unzip -o /tmp/a.zip",
            "cd src && rm ../app.py",
            "dd if=/dev/zero of=src/app.py bs=1 count=1",
            # Writing to out_dir (receipt forgery attempt) must also be blocked!
            f"echo '{{}}' > {out_dir}/receipts/forged.json",
        ]
        for cmd in blocked_commands:
            verdict = detect_bash_workspace_mutation(cmd, tmp_repo, out_dir)
            self.assertTrue(
                verdict["blocked"],
                f"Expected command to be blocked in strict mode: {cmd!r}",
            )

        allowed_commands = [
            "pytest -q",
            "python3 -m unittest discover -s tests -v",
            "python3 -c 'print(1 + 1)'",
            "git status --short",
            "git -C . diff HEAD",
            "ls -la src/",
            "cp src/app.py /tmp/external-scratch.py",
            "echo 'ok' > /tmp/external-log.txt",
            "dd if=/dev/zero of=/tmp/external-scratch.bin bs=1 count=1",
            "grep -c foo <<< 'hello foo'",
            "cat << 'EOF'\nhello stdout\nEOF",
        ]
        for cmd in allowed_commands:
            verdict = detect_bash_workspace_mutation(cmd, tmp_repo, out_dir)
            self.assertFalse(
                verdict["blocked"],
                f"Expected read-only or external-tmp command to be allowed: {cmd!r} ({verdict['reason']})",
            )

    def test_dictation_linter_catches_tilde_fences_method_calls_indented_blocks_and_comments(self) -> None:
        # 1. Tilde fence with 9 lines
        tilde_prompt = "~~~\n" + "\n".join(f"self.items.append(build({i}))" for i in range(9)) + "\n~~~"
        self.assertFalse(lint_delegation_prompt(tilde_prompt)["ok"])

        # 2. Unfenced method calls
        calls_prompt = "\n".join(f"self.items.append(build_item({i}))" for i in range(10))
        self.assertFalse(lint_delegation_prompt(calls_prompt)["ok"])

        # 3. 4-space indented code block
        indented_prompt = "Here is the code:\n\n" + "\n".join("    pass" for _ in range(10))
        self.assertFalse(lint_delegation_prompt(indented_prompt)["ok"])

        # 4. Interleaved comments every 3rd line inside 10 lines of code
        interleaved = "\n".join(
            f"# comment {i}\nx_{i} = {i}" if i % 3 == 0 else f"x_{i} = {i}"
            for i in range(9)
        )
        self.assertFalse(lint_delegation_prompt(interleaved)["ok"])

        # 5. delegate_continue follow_up_prompt is also linted by evaluate_pre_tool_use
        hook_out = evaluate_pre_tool_use(
            {
                "tool_name": "mcp__plugin_cc-agy-plugin_vertex__delegate_continue",
                "tool_input": {"job_id": "job-1", "follow_up_prompt": calls_prompt},
                "cwd": "/tmp/ws",
            },
            env={"CC_AGY_STRICT_MODE": "false"},
        )
        self.assertEqual(hook_out["hookSpecificOutput"]["permissionDecision"], "deny")


class TestMCPStdioServer(unittest.TestCase):
    """Verify JSON-RPC 2.0 MCP stdio protocol, malformed input resilience, and concurrent cancel."""

    def test_mcp_stdio_survives_malformed_frames_and_validates_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo_dir)
            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=lambda _r: {"text": "ok"})
            mcp = MCPServer(job_manager=jm, store=store)

            # Non-object JSON, empty batch, missing method, unknown version, unknown tool, string allowed_paths
            self.assertEqual(mcp.handle_Message(42)["error"]["code"], -32600)
            self.assertEqual(mcp.handle_Message([])["error"]["code"], -32600)
            self.assertEqual(mcp.handle_Message({"jsonrpc": "2.0", "id": 1})["error"]["code"], -32600)

            init_resp = mcp.handle_Message(
                {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "2099-01-01"}}
            )
            self.assertEqual(init_resp["result"]["protocolVersion"], "2024-11-05")

            unk_tool = mcp.handle_Message(
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "no_such_tool"}}
            )
            self.assertEqual(unk_tool["error"]["code"], -32602)

            bad_allowed = mcp.handle_Message(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "delegate_start",
                        "arguments": {"prompt": "hi", "allowed_paths": "src/"},
                    },
                }
            )
            self.assertEqual(bad_allowed["error"]["code"], -32602)

            # Test bad Content-Length and invalid UTF-8 on wire without crashing serve_stdio
            raw_stream = (
                b"Content-Length: abc\r\n\r\n"
                b"\xff\xfe\xfd\n"
                + (json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping"}) + "\n").encode("utf-8")
            )
            out_buf = io.BytesIO()
            serve_stdio(server=mcp, in_stream=io.BytesIO(raw_stream), out_stream=out_buf)
            raw_out = out_buf.getvalue().decode("utf-8")
            self.assertIn("-32700", raw_out)
            self.assertIn("Invalid Content-Length header", raw_out)
            self.assertIn("Invalid UTF-8 payload", raw_out)
            self.assertIn('"id":99', raw_out)

    def test_parse_bool_and_utf8_pagination_and_file_filter(self) -> None:
        self.assertFalse(parse_bool("false", True))
        self.assertFalse(parse_bool("0", True))
        self.assertTrue(parse_bool("true", False))

        # Multi-byte UTF-8 characters on page boundary must not be dropped
        multi_byte_text = "αβγδεζηθ" * 200
        pages = paginate_text_by_tokens(multi_byte_text, page_tokens=64)
        self.assertGreater(len(pages), 1)
        self.assertEqual("".join(pages), multi_byte_text)

    def test_round2_merge_apply_concurrency_and_untracked_merge(self) -> None:
        import concurrent.futures

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            base_lines = [f"line_{i}\n" for i in range(1, 21)]
            (repo_dir / "shared.py").write_text("".join(base_lines), encoding="utf-8")
            (repo_dir / "untracked.py").write_text("".join(base_lines), encoding="utf-8")
            subprocess.run(["git", "add", "shared.py"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "shared"], cwd=repo_dir, check=True, capture_output=True)

            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo_dir)

            def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                wt = Path(req["workspace_dir"])
                if req["prompts"][0].startswith("edit top"):
                    lines = (wt / "shared.py").read_text(encoding="utf-8").splitlines(True)
                    lines[0] = "TOP_EDIT\n"
                    (wt / "shared.py").write_text("".join(lines), encoding="utf-8")
                    u_lines = (wt / "untracked.py").read_text(encoding="utf-8").splitlines(True)
                    u_lines[0] = "UNTRACKED_TOP\n"
                    (wt / "untracked.py").write_text("".join(u_lines), encoding="utf-8")
                else:
                    lines = (wt / "shared.py").read_text(encoding="utf-8").splitlines(True)
                    lines[-1] = "BOTTOM_EDIT\n"
                    (wt / "shared.py").write_text("".join(lines), encoding="utf-8")
                return {"text": "done", "input_tokens": 10, "output_tokens": 5}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
            r1 = jm.delegate_start("edit top", project="p", wait=True)
            r2 = jm.delegate_start("edit bottom", project="p", wait=True)

            # Mutate bottom of untracked.py on host before applying r1
            u_host = (repo_dir / "untracked.py").read_text(encoding="utf-8").splitlines(True)
            u_host[-1] = "UNTRACKED_HOST_BOTTOM\n"
            (repo_dir / "untracked.py").write_text("".join(u_host), encoding="utf-8")

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(jm.delegate_apply, r1["job_id"])
                f2 = pool.submit(jm.delegate_apply, r2["job_id"])
                a1, a2 = f1.result(), f2.result()

            self.assertTrue(a1["applied"])
            self.assertTrue(a2["applied"])
            merged_shared = (repo_dir / "shared.py").read_text(encoding="utf-8")
            self.assertIn("TOP_EDIT\n", merged_shared)
            self.assertIn("BOTTOM_EDIT\n", merged_shared)
            merged_untracked = (repo_dir / "untracked.py").read_text(encoding="utf-8")
            self.assertIn("UNTRACKED_TOP\n", merged_untracked)
            self.assertIn("UNTRACKED_HOST_BOTTOM\n", merged_untracked)
            # Host git index must remain clean (unstaged) after 3-way merge
            cached_diff = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(cached_diff, "")

    def test_round2_gitignored_node_modules_and_worker_created_ignored_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            (repo_dir / ".gitignore").write_text("node_modules/\n*.env\n", encoding="utf-8")
            subprocess.run(["git", "add", ".gitignore"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "ignore"], cwd=repo_dir, check=True, capture_output=True)
            nm = repo_dir / "node_modules" / "pkg"
            nm.mkdir(parents=True)
            (nm / "bundle.js").write_text("x" * 1024, encoding="utf-8")

            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo_dir)
            saw_node_modules = []

            def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                wt = Path(req["workspace_dir"])
                saw_node_modules.append((wt / "node_modules").exists())
                (wt / "prod.env").write_text("PORT=8080\n", encoding="utf-8")
                return {"text": "created prod.env", "input_tokens": 5, "output_tokens": 5}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
            res = jm.delegate_start("create prod.env", project="p", wait=True)
            self.assertEqual(saw_node_modules, [False])
            self.assertIn("prod.env", res["files_changed"])
            applied = jm.delegate_apply(res["job_id"])
            self.assertTrue(applied["applied"])
            self.assertTrue((repo_dir / "prod.env").exists())

    def test_round2_hmac_key_guard_and_commit_laundering_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            data_dir = Path(tmp) / "data"
            store = ReceiptStore(data_dir, workspace_root=repo_dir)
            _ = JobManager(workspace_root=repo_dir, store=store, worker_backend=lambda _r: {"text": "ok"})

            # Reading .hmac_key via Read tool must be denied in strict mode
            env = {
                "CC_AGY_STRICT_MODE": "true",
                "CLAUDE_PROJECT_DIR": str(repo_dir),
                "CC_AGY_OUT_DIR": str(data_dir),
            }
            read_decision = evaluate_pre_tool_use(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": str(data_dir / ".hmac_key")},
                    "cwd": str(repo_dir),
                },
                env=env,
            )
            self.assertEqual(
                read_decision["hookSpecificOutput"]["permissionDecision"],
                "deny",
            )

            # Quote-aware semicolon inside python3 -c must be allowed
            py_check = detect_bash_workspace_mutation(
                "python3 -c 'import sys; print(sys.version)'",
                repo_dir,
                data_dir,
                env=env,
            )
            self.assertFalse(py_check["blocked"])

            # Direct git commit without delegation receipt must fail provenance check
            (repo_dir / "app.py").write_text("print('laundered')\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "sneak"], cwd=repo_dir, check=True, capture_output=True)
            prov = verify_workspace_provenance(repo_dir, store=store)
            self.assertFalse(prov["ok"])
            self.assertIn("app.py", prov["unverified_files"])

    def test_round4_tracked_and_untracked_build_dirs_and_cross_session_commits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            for rel in ("build/gen.py", "dist/index.html", "coverage/__init__.py", "src/build/__init__.py"):
                p = repo_dir / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("v = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "tracked build dirs"], cwd=repo_dir, check=True, capture_output=True)

            # Host-only untracked file under build/
            (repo_dir / "build" / "notes.md").write_text("keep me\n", encoding="utf-8")

            data_dir = Path(tmp) / "data"
            store = ReceiptStore(data_dir, workspace_root=repo_dir)

            # Before server start in strict mode: distinct missing-baseline message
            pre_verdict = verify_workspace_provenance(
                repo_dir,
                store=store,
                env={"CC_AGY_STRICT_MODE": "1", "CLAUDE_PROJECT_DIR": str(repo_dir)},
            )
            self.assertFalse(pre_verdict["ok"])
            self.assertIn("No session baseline recorded", pre_verdict["reason"] or "")

            def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                wt = Path(req["workspace_dir"])
                (wt / "app.py").write_text("print('v2')\n", encoding="utf-8")
                (wt / "__pycache__").mkdir(exist_ok=True)
                (wt / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"\x00pyc")
                return {"text": "updated app.py"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
            res = jm.delegate_start("update app.py", project="p", wait=True)
            self.assertEqual(res["files_changed"], ["app.py"])
            applied = jm.delegate_apply(res["job_id"])
            self.assertEqual(applied["applied_files"], ["app.py"])
            self.assertTrue((repo_dir / "build" / "gen.py").exists())
            self.assertTrue((repo_dir / "dist" / "index.html").exists())
            self.assertTrue((repo_dir / "coverage" / "__init__.py").exists())
            self.assertTrue((repo_dir / "build" / "notes.md").exists())

            # Human commits between sessions -> new session refreshes baseline cleanly
            subprocess.run(["git", "add", "app.py"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "human commit"], cwd=repo_dir, check=True, capture_output=True)
            store2 = ReceiptStore(data_dir, workspace_root=repo_dir)
            jm2 = JobManager(workspace_root=repo_dir, store=store2, worker_backend=worker_fn)
            self.assertTrue(verify_workspace_provenance(repo_dir, store=store2, strict=True)["ok"])

            # Abandoned unapplied job + new untracked note in next session -> still ok
            _ = jm2.delegate_start("unapplied experiment", project="p", wait=True)
            (repo_dir / "todo.md").write_text("my notes\n", encoding="utf-8")
            store3 = ReceiptStore(data_dir, workspace_root=repo_dir)
            _ = JobManager(workspace_root=repo_dir, store=store3, worker_backend=worker_fn)
            self.assertTrue(verify_workspace_provenance(repo_dir, store=store3, strict=True)["ok"])

    def test_round5_non_utf8_diffs_gitignored_build_outputs_and_session_start_baseline(self) -> None:
        from scripts.session_start import handle_session_start

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            repo_dir.mkdir()
            _init_git_repo(repo_dir)
            (repo_dir / ".gitignore").write_text("target/\ndist/\nbuild/\n.gradle/\n", encoding="utf-8")
            (repo_dir / "legacy.csv").write_bytes("name\nJos\xe9\n".encode("latin-1"))
            subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "add gitignore and latin1"], cwd=repo_dir, check=True, capture_output=True)

            # Dirty Latin-1 file on host before JobManager starts must not crash UTF-8 decode
            (repo_dir / "legacy.csv").write_bytes("name\nJos\xe9\nRen\xe9e\n".encode("latin-1"))
            data_dir = Path(tmp) / "data"
            store = ReceiptStore(data_dir, workspace_root=repo_dir)

            def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                wt = Path(req["workspace_dir"])
                (wt / "app.py").write_text("print('v3')\n", encoding="utf-8")
                for rel, raw in (
                    ("target/debug/app", b"\x7fELF\x00bin"),
                    ("dist/bundle.js", b"console.log(1)"),
                    ("build/classes/App.class", b"\xca\xfe\xba\xbe"),
                ):
                    p = wt / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(raw)
                return {"text": "compiled and tested"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
            res = jm.delegate_start("edit app.py and build", allowed_paths=["app.py"], project="p", wait=True)
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["files_changed"], ["app.py"])
            applied = jm.delegate_apply(res["job_id"])
            self.assertEqual(applied["applied_files"], ["app.py"])
            self.assertFalse((repo_dir / "target" / "debug" / "app").exists())

            # SessionStart hook with a new session_id records human WIP cleanly
            (repo_dir / "app.py").write_text("print('human wip')\n", encoding="utf-8")
            env = {"CC_AGY_STRICT_MODE": "1", "CLAUDE_PROJECT_DIR": str(repo_dir), "CC_AGY_DATA_DIR": str(data_dir)}
            handle_session_start({"cwd": str(repo_dir), "session_id": "sess-alpha-1"}, env=env, store=store)
            self.assertEqual(
                evaluate_stop_hook(
                    {"cwd": str(repo_dir), "session_id": "sess-alpha-1", "stop_hook_active": False},
                    env=env,
                    store=store,
                ),
                {},
            )


def _git(repo_dir: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(repo_dir), check=True, capture_output=True, text=True
    ).stdout


class TestInPlaceMode(unittest.TestCase):
    """in_place=True: the worker edits the real working tree instead of an isolated copy."""

    def _setup(self, tmp: str) -> tuple[Path, ReceiptStore]:
        repo_dir = Path(tmp) / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        return repo_dir, ReceiptStore(Path(tmp) / "data", workspace_root=repo_dir)

    def test_in_place_edits_real_tree_undoes_worker_commits_and_skips_preexisting_untracked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            (repo_dir / "notes.md").write_text("my own notes\n", encoding="utf-8")
            baseline = _git(repo_dir, "rev-parse", "HEAD").strip()
            seen_dirs: list[str] = []

            def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                wd = Path(req["workspace_dir"])
                seen_dirs.append(str(wd))
                (wd / "app.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
                (wd / "new.py").write_text("X = 1\n", encoding="utf-8")
                (wd / "__pycache__").mkdir(exist_ok=True)
                (wd / "__pycache__" / "app.cpython-313.pyc").write_bytes(b"\x00pyc")
                subprocess.run(["git", "add", "-A"], cwd=str(wd), check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", "worker"], cwd=str(wd), check=True, capture_output=True)
                return {"text": "done"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
            res = jm.delegate_start("edit greet", project="p", wait=True, in_place=True)

            self.assertEqual(seen_dirs, [str(repo_dir.resolve())])
            self.assertEqual(res["status"], "ok")
            self.assertTrue(res["in_place"])
            self.assertTrue(res["applied"])
            self.assertEqual(
                sorted(res["files_changed"]), ["__pycache__/app.cpython-313.pyc", "app.py", "new.py"]
            )
            self.assertTrue(any("undone" in n for n in res["in_place_notes"]))
            self.assertIn("git checkout", res["undo_hint"])
            self.assertIn("new.py", res["undo_hint"].split("rm -- ", 1)[1])
            # Worker's commit was unwound: HEAD back at baseline, index clean, edits kept on disk.
            self.assertEqual(_git(repo_dir, "rev-parse", "HEAD").strip(), baseline)
            self.assertEqual(_git(repo_dir, "diff", "--cached", "--name-only").strip(), "")
            self.assertIn("'hi'", (repo_dir / "app.py").read_text(encoding="utf-8"))
            self.assertEqual((repo_dir / "notes.md").read_text(encoding="utf-8"), "my own notes\n")
            # Nothing left to apply, and the Stop-hook provenance check accepts the worker's lines.
            self.assertTrue(jm.delegate_apply(res["job_id"])["already_applied"])
            self.assertTrue(verify_workspace_provenance(repo_dir, store=store, strict=True)["ok"])

    def test_in_place_refuses_dirty_tree_allowed_paths_and_concurrent_jobs(self) -> None:
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            gate = threading.Event()

            def slow_worker(req: dict[str, Any]) -> dict[str, Any]:
                gate.wait(5)
                return {"text": "ok"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=slow_worker)
            with self.assertRaises(ValueError):
                jm.delegate_start("x", project="p", in_place=True, allowed_paths=["app.py"])

            first = jm.delegate_start("first", project="p", in_place=True)
            with self.assertRaises(RuntimeError):
                jm.delegate_start("second", project="p", in_place=True)
            gate.set()
            jm.delegate_result(first["job_id"], wait=True, timeout=10)

            (repo_dir / "app.py").write_text("dirty\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                jm.delegate_start("x", project="p", in_place=True)
            self.assertIn("clean working tree", str(ctx.exception))

    def test_in_place_cancel_and_cleanup_never_delete_the_real_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=lambda _r: {"text": "ok"})
            res = jm.delegate_start("noop", project="p", wait=True, in_place=True)
            jm.delegate_cancel(res["job_id"], cleanup_worktree=True)
            jm._cleanup_worktree(repo_dir)
            self.assertTrue((repo_dir / "app.py").exists())
            self.assertTrue((repo_dir / ".git").exists())

    def test_in_place_partial_failure_keeps_edits_marked_applied_with_undo_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)

            def failing_worker(req: dict[str, Any]) -> dict[str, Any]:
                (Path(req["workspace_dir"]) / "half.py").write_text("partial = True\n", encoding="utf-8")
                raise RuntimeError("model crashed")

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=failing_worker)
            res = jm.delegate_start("x", project="p", wait=True, in_place=True)
            self.assertEqual(res["status"], "error")
            self.assertTrue(res["applied"])
            self.assertEqual(res["files_changed"], ["half.py"])
            self.assertIn("Partial in_place edits", res["error"])
            self.assertIn("rm -- half.py", res["undo_hint"])

    def test_in_place_continue_is_cumulative_and_refuses_if_tree_changed_by_someone_else(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            turns: list[int] = []

            def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                turns.append(1)
                wd = Path(req["workspace_dir"])
                with open(wd / "app.py", "a", encoding="utf-8") as fh:
                    fh.write(f"TURN_{len(turns)} = True\n")
                return {"text": "ok"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
            first = jm.delegate_start("turn one", project="p", wait=True, in_place=True)
            second = jm.delegate_continue(first["job_id"], "turn two", wait=True)
            self.assertEqual(second["files_changed"], ["app.py"])
            text = (repo_dir / "app.py").read_text(encoding="utf-8")
            self.assertIn("TURN_1", text)
            self.assertIn("TURN_2", text)
            self.assertTrue(verify_workspace_provenance(repo_dir, store=store, strict=True)["ok"])

            # A stray untracked file (e.g. written by a test run) no longer blocks recovery: it is treated
            # as pre-existing and never credited to the worker.
            (repo_dir / "README.md").write_text("human edit\n", encoding="utf-8")
            third = jm.delegate_continue(first["job_id"], "turn three", wait=True)
            self.assertEqual(third["files_changed"], ["app.py"])
            self.assertNotIn("README.md", third["files_changed"])

            # A foreign change to a *tracked* file still blocks the in_place continue.
            (repo_dir / "README.md").unlink()
            subprocess.run(["git", "stash", "-q"], cwd=repo_dir, check=True, capture_output=True)
            (repo_dir / "notes.txt").write_text("n\n", encoding="utf-8")
            subprocess.run(["git", "add", "notes.txt"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-qm", "notes"], cwd=repo_dir, check=True, capture_output=True)
            fresh = jm.delegate_start("turn a", project="p", wait=True, in_place=True)
            (repo_dir / "notes.txt").write_text("someone else\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                jm.delegate_continue(fresh["job_id"], "turn b", wait=True)

    def test_mcp_tools_expose_in_place(self) -> None:
        from server.mcp_stdio import TOOL_DEFINITIONS

        with_in_place = {t["name"] for t in TOOL_DEFINITIONS if "in_place" in t["inputSchema"]["properties"]}
        self.assertEqual(with_in_place, {"delegate_start", "delegate_to_gemini"})


class TestCostControls(unittest.TestCase):
    """Budget checkpoints, worker -> lead signals, plugin-run acceptance checks, structured plans."""

    def _setup(self, tmp: str) -> tuple[Path, ReceiptStore]:
        repo_dir = Path(tmp) / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        return repo_dir, ReceiptStore(Path(tmp) / "data", workspace_root=repo_dir)

    def test_budget_stop_becomes_checkpoint_with_default_caps_and_can_be_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            seen: list[dict[str, Any]] = []

            def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                seen.append(req)
                (Path(req["workspace_dir"]) / "app.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
                return {"text": "", "stop_reason": "MAX_TOOL_CALLS_EXCEEDED"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
            res = jm.delegate_start("edit greet", project="p", wait=True)
            self.assertEqual(res["status"], "checkpoint")
            self.assertEqual(res["stop_reason"], "MAX_TOOL_CALLS_EXCEEDED")
            self.assertEqual(res["budget"], {"max_tool_calls": 100, "max_total_tokens": 5_000_000})
            self.assertEqual((seen[0]["max_tool_calls"], seen[0]["max_total_tokens"]), (100, 5_000_000))
            self.assertTrue(jm.delegate_apply(res["job_id"])["applied"])

            no_cap = jm.delegate_start("x", project="p", wait=True, max_tool_calls=0, max_total_tokens=0)
            self.assertEqual(no_cap["budget"], {"max_tool_calls": None, "max_total_tokens": None})
            self.assertNotIn("## Budget", seen[-1]["prompts"][0])

    def test_report_to_lead_question_sets_needs_input_and_progress_is_visible(self) -> None:
        import threading

        from server.agy_runner import append_progress

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            gate = threading.Event()
            reported = threading.Event()

            def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                append_progress(req["progress_path"], "milestone", "read the code")
                reported.set()
                gate.wait(5)
                msg = append_progress(req["progress_path"], "question", "Should empty bodies raise?")
                self.assertIn("Stop now", msg)
                return {"text": "need an answer"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
            started = jm.delegate_start("x", project="p")
            reported.wait(5)
            live = jm.delegate_status(started["job_id"])
            self.assertEqual(live["status"], "running")
            self.assertEqual(live["progress"][-1]["message"], "read the code")
            gate.set()
            done = jm.delegate_result(started["job_id"], wait=True, timeout=10)
            self.assertEqual(done["status"], "needs_input")
            self.assertEqual(done["progress"][-1]["kind"], "question")

    def test_acceptance_runs_after_worker_reports_pass_fail_timeout_and_cleans_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            (repo_dir / "test_check.py").write_text(
                "open('artifact.out', 'w').write('x')\nprint('3 passed in 0.1s')\n", encoding="utf-8"
            )
            subprocess.run(["git", "add", "test_check.py"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-qm", "check"], cwd=repo_dir, check=True, capture_output=True)

            def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                (Path(req["workspace_dir"]) / "app.py").write_text("x = 2\n", encoding="utf-8")
                return {"text": "done"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
            os.environ["CC_AGY_ACCEPTANCE_TIMEOUT"] = "2"
            try:
                res = jm.delegate_start(
                    "x",
                    project="p",
                    wait=True,
                    in_place=True,
                    acceptance=["python3 test_check.py", "python3 -c 'import sys; sys.exit(1)'", "sleep 30"],
                )
            finally:
                os.environ.pop("CC_AGY_ACCEPTANCE_TIMEOUT", None)
            results = res["acceptance_results"]
            self.assertEqual([r["passed"] for r in results], [True, False, False])
            self.assertEqual(results[0]["counts"]["passed"], 3)
            self.assertTrue(results[2]["timed_out"])
            self.assertLess(results[2]["duration_seconds"], 10)
            self.assertFalse(res["acceptance_passed"])
            self.assertEqual(res["files_changed"], ["app.py"])
            self.assertFalse((repo_dir / "artifact.out").exists())
            self.assertIn("artifact.out", results[-1]["cleaned_artifacts"])

    def test_acceptance_commands_that_modify_the_repo_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=lambda _r: {"text": "ok"})
            for bad in (
                "cp /tmp/x app.py",
                "rm -rf .",
                "echo hi > app.py",
                "python3 -c \"import sys, operator as o; m=o.attrgetter('modules')(sys); o.attrgetter('system')(m['os'])('touch pwned')\"",
                "python3 -c \"import sys, operator as o; o.methodcaller('system','touch pwned')(o.attrgetter('modules')(sys)['os'])\"",
                "python3 -c \"import sys, string; F=string.Formatter(); m=F.get_field('0.modules', (sys,), {})[0]; F.get_field('0.system', (m['os'],), {})[0]('touch pwned_fmt')\"",
                "echo 'touch pwned' | sh",
                "echo 'touch pwned' | bash",
                "printf 'import os\\nos.system(\\\"touch pwned\\\")' | python3",
                "echo 'touch pwned' | xargs -I{} sh -c {}",
            ):
                with self.assertRaises(ValueError, msg=f"Expected ValueError for {bad!r}"):
                    jm.delegate_start("x", project="p", acceptance=[bad])
            from server.jobs import validate_acceptance
            for good_acc in (
                "pytest -q 2>&1 | tail -5",
                "make test 2>&1 | tail -5",
                "pytest -q > /tmp/out.txt 2>&1",
                "pytest -q > /dev/null",
            ):
                validate_acceptance([good_acc], repo_dir)
                self.assertFalse(detect_bash_workspace_mutation(good_acc, repo_dir, cwd=repo_dir)["blocked"])
            self.assertTrue(detect_bash_workspace_mutation("pytest /tmp/evil_test.py", repo_dir, cwd=repo_dir)["blocked"])
            jm.shutdown()

    def test_plan_and_acceptance_are_rendered_for_worker_and_carried_into_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            prompts: list[str] = []

            def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                prompts.append(req["prompts"][0])
                return {"text": "ok"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
            res = jm.delegate_start(
                "Implement greet()", project="p", wait=True,
                plan=["Read app.py", "Change greet"], acceptance=["python3 -c 'pass'"], max_tool_calls=12,
            )
            self.assertIn("## Plan from the lead\n1. Read app.py\n2. Change greet", prompts[0])
            self.assertIn("- `python3 -c 'pass'`", prompts[0])
            self.assertIn("12 tool calls", prompts[0])
            self.assertIn("report_to_lead", prompts[0])
            jm.delegate_continue(res["job_id"], "Also add a docstring", wait=True)
            self.assertIn("Also add a docstring", prompts[1])
            self.assertIn("12 tool calls", prompts[1])
            self.assertIn("2. Change greet", prompts[1])
            with self.assertRaises(ValueError):
                jm.delegate_start("x", project="p", plan=[f"step {i}" for i in range(20)])

    def test_unfinished_jobs_are_marked_interrupted_on_shutdown(self) -> None:
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            gate = threading.Event()
            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=lambda _r: (gate.wait(5), {"text": ""})[1])
            started = jm.delegate_start("x", project="p")
            self.assertEqual(jm.mark_running_interrupted(), [started["job_id"]])
            self.assertEqual(store.read_receipt(started["job_id"])["status"], "interrupted")
            gate.set()
            jm.shutdown(wait=True)

    def test_mcp_schema_and_guard_cover_plan_acceptance_budget(self) -> None:
        from server.mcp_stdio import TOOL_DEFINITIONS

        start = next(t for t in TOOL_DEFINITIONS if t["name"] == "delegate_start")["inputSchema"]["properties"]
        for field in ("plan", "acceptance", "max_tool_calls", "max_total_tokens"):
            self.assertIn(field, start)
        code_plan = [f"x{i} = compute({i})" for i in range(12)]
        decision = evaluate_pre_tool_use(
            {"tool_name": "mcp__vertex__delegate_start", "tool_input": {"prompt": "do it", "plan": code_plan}, "cwd": "/tmp"},
            env={},
        )
        self.assertEqual(decision["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_straitjacket_containment_prompt_and_receipt(self) -> None:
        import asyncio
        from server.agy_runner import build_straitjacket_tools

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            prompts: list[str] = []

            def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                prompts.append(req["prompts"][0])
                return {"text": "ok", "containment": "straitjacket"}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
            res = jm.delegate_start("Test containment", project="p", wait=True)
            self.assertIn("## Shell Commands & Evidence", prompts[0])
            self.assertIn("ctx_get", prompts[0])
            self.assertIn("ctx_search", prompts[0])
            receipt = store.read_receipt(res["job_id"])
            self.assertEqual(receipt.get("containment"), "straitjacket")

            # Live verification of build_straitjacket_tools with default 300s turn timeout
            tools, replaced = build_straitjacket_tools(repo_dir, timeout_seconds=300.0)
            self.assertEqual(replaced, ["RUN_COMMAND"])
            tool_map = {fn.__name__: fn for fn in tools}
            self.assertEqual(set(tool_map.keys()), {"run_command", "ctx_get", "ctx_search"})

            # 1. Default timeout (300s session) + oversized timeout_seconds (900s) must clamp, not raise ValueError
            out_json = asyncio.run(
                tool_map["run_command"](
                    "python3 -c 'for i in range(1, 350): print(f\"LINE_{i:03d}_PAYLOAD_TEXT\")'",
                    timeout_seconds=900.0,
                )
            )
            parsed = json.loads(out_json)
            self.assertEqual(parsed["command_exit_code"], 0)
            ref = parsed["ref"]
            self.assertTrue(ref.startswith("run:"))

            # 2. ctx_get with 0-indexed start_line=0 coerces cleanly to 1-indexed and retrieves lines
            slice_text = asyncio.run(tool_map["ctx_get"](f"{ref}#stdout", start_line=0, end_line=5))
            self.assertIn("LINE_001_PAYLOAD_TEXT", slice_text)
            self.assertIn("LINE_005_PAYLOAD_TEXT", slice_text)

            # 3. ctx_search locates matching lines in the stored stdout artifact
            search_text = asyncio.run(tool_map["ctx_search"](f"{ref}#stdout", "LINE_250_PAYLOAD_TEXT"))
            self.assertIn("LINE_250_PAYLOAD_TEXT", search_text)

    def test_resolve_api_key_and_auth_config(self) -> None:
        self.assertEqual(resolve_api_key("my-api-key"), "my-api-key")
        self.assertEqual(resolve_api_key(None, env={"GEMINI_API_KEY": "gem-env-key"}), "gem-env-key")
        self.assertEqual(resolve_api_key(None, env={"GOOGLE_API_KEY": "goog-env-key"}), "goog-env-key")
        self.assertIsNone(resolve_api_key("${user_config.gemini_api_key}", env={}))
        self.assertIsNone(resolve_api_key("<api-key>", env={}))
        self.assertIsNone(resolve_api_key("", env={}))

        auth_key = resolve_auth_config(requested_api_key="my-key", env={})
        self.assertEqual(auth_key["auth_mode"], "api_key")
        self.assertEqual(auth_key["api_key"], "my-key")
        self.assertIsNone(auth_key["location"])

        auth_vertex = resolve_auth_config(requested_project="my-proj", env={"GOOGLE_CLOUD_LOCATION": "us-central1"})
        self.assertEqual(auth_vertex["auth_mode"], "vertex")
        self.assertEqual(auth_vertex["project"], "my-proj")
        self.assertEqual(auth_vertex["location"], "us-central1")

        with self.assertRaises(ValueError):
            resolve_auth_config(env={})

    def test_scrub_receipt_scrubs_api_key(self) -> None:
        raw_receipt = {
            "vertex_project": "my-secret-project",
            "api_key": "AIzaSySecretTestingKey1234567890abc",
            "summary": "Invoked model with key AIzaSySecretTestingKey1234567890abc in project my-secret-project",
        }
        scrubbed = scrub_receipt_for_export(raw_receipt, api_key="AIzaSySecretTestingKey1234567890abc")
        self.assertEqual(scrubbed["vertex_project"], "<paying-project>")
        self.assertEqual(scrubbed["api_key"], "<api-key>")
        self.assertNotIn("AIzaSySecretTestingKey1234567890abc", json.dumps(scrubbed))
        self.assertNotIn("my-secret-project", json.dumps(scrubbed))

    def test_api_key_mode_for_ask_gemini_and_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)

            def mock_ask_backend(req: dict[str, Any]) -> dict[str, Any]:
                self.assertEqual(req.get("api_key"), "AIzaTestKey")
                self.assertEqual(req.get("auth_mode"), "api_key")
                return {"text": "Answer from API key", "usage": {"total_token_count": 50}}

            ask_res = ask_gemini(
                "Review this code",
                api_key="AIzaTestKey",
                workspace_root=repo_dir,
                store=store,
                backend=mock_ask_backend,
            )
            self.assertEqual(ask_res["status"], "ok")
            self.assertEqual(ask_res["auth_mode"], "api_key")
            ask_rcpt = store.read_receipt(ask_res["job_id"])
            self.assertEqual(ask_rcpt["auth_mode"], "api_key")

            def mock_worker(req: dict[str, Any]) -> dict[str, Any]:
                self.assertEqual(req.get("api_key"), "AIzaTestKey")
                self.assertEqual(req.get("auth_mode"), "api_key")
                (Path(req["workspace_dir"]) / "output.txt").write_text("done\n", encoding="utf-8")
                return {"text": "Worker done", "usage": {"total_token_count": 100}}

            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=mock_worker)
            start_res = jm.delegate_start(
                "Implement feature",
                api_key="AIzaTestKey",
                wait=True,
            )
            self.assertEqual(start_res["status"], "ok")
            self.assertEqual(start_res["auth_mode"], "api_key")
            job_rcpt = store.read_receipt(start_res["job_id"])
            self.assertEqual(job_rcpt["auth_mode"], "api_key")

            # delegate_continue should automatically reuse the job's api_key
            cont_res = jm.delegate_continue(
                start_res["job_id"],
                "Fix extra detail",
                wait=True,
            )
            self.assertEqual(cont_res["status"], "ok")
            self.assertEqual(cont_res["auth_mode"], "api_key")

    def test_mcp_schema_and_stdio_expose_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)
            jm = JobManager(workspace_root=repo_dir, store=store, worker_backend=lambda _r: {"text": "ok"})
            server = MCPServer(job_manager=jm, store=store)

            # Check tools have api_key in inputSchema
            tools_list = server.handle_Message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
            schemas = {t["name"]: t["inputSchema"]["properties"] for t in tools_list}
            self.assertIn("api_key", schemas["ask_gemini"])
            self.assertIn("api_key", schemas["delegate_start"])
            self.assertIn("api_key", schemas["delegate_continue"])
            self.assertIn("api_key", schemas["delegate_to_gemini"])

            called_auth = {}

            def mock_ask(req: dict[str, Any]) -> dict[str, Any]:
                called_auth["api_key"] = req.get("api_key")
                called_auth["auth_mode"] = req.get("auth_mode")
                return {"text": "mcp ask ok"}

            server.ask_backend = mock_ask
            res = server._dispatch_tool(
                "ask_gemini",
                {"prompt": "hello", "api_key": "mcp-test-key"},
            )
            self.assertEqual(res["status"], "ok")
            self.assertEqual(called_auth["api_key"], "mcp-test-key")
            self.assertEqual(called_auth["auth_mode"], "api_key")

    def test_api_key_hardening_redaction_and_restart_continuity(self) -> None:
        from server.regions import sanitize_placeholder_env

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir, store = self._setup(tmp)

            # 1. Unconditional API key redaction even when scrub=False
            special_key = 'my-custom-key-with"quotes\\and-secret'
            aiza_key = "AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6"
            rcpt = store.write_receipt(
                "rcpt-unscrubbed-test",
                {
                    "tool": "ask_gemini",
                    "status": "error",
                    "auth_mode": "api_key",
                    "error": f"Request failed with key={special_key} and url?key={aiza_key}",
                },
                scrub=False,
                api_key=special_key,
            )
            self.assertNotIn(special_key, json.dumps(rcpt))
            self.assertNotIn(aiza_key, json.dumps(rcpt))
            disk_raw = (store.receipts_dir / "rcpt-unscrubbed-test.json").read_text(encoding="utf-8")
            self.assertNotIn(special_key, disk_raw)
            self.assertNotIn(aiza_key, disk_raw)
            self.assertTrue(store.verify_receipt_signature(rcpt))

            # 2. Cross-mode flip guard on delegate_continue across restart when both GEMINI_API_KEY and GOOGLE_CLOUD_PROJECT are set
            with unittest.mock.patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "env-gemini-secret-key-999",
                    "GOOGLE_CLOUD_PROJECT": "ambient-gcloud-proj",
                },
                clear=False,
            ):
                seen_turns: list[dict[str, Any]] = []

                def worker_fn(req: dict[str, Any]) -> dict[str, Any]:
                    seen_turns.append({"auth_mode": req.get("auth_mode"), "api_key": req.get("api_key"), "project": req.get("project")})
                    return {"text": "ok"}

                jm1 = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
                first = jm1.delegate_start("initial turn", wait=True)
                self.assertEqual(first["auth_mode"], "api_key")
                self.assertIsNone(store.read_receipt(first["job_id"]).get("vertex_project"))

                # Simulate MCP server restart with a fresh JobManager (empty _raw_api_keys)
                jm2 = JobManager(workspace_root=repo_dir, store=store, worker_backend=worker_fn)
                second = jm2.delegate_continue(first["job_id"], "follow up after restart", wait=True)
                self.assertEqual(second["auth_mode"], "api_key")
                self.assertEqual(seen_turns[-1]["auth_mode"], "api_key")
                self.assertEqual(seen_turns[-1]["api_key"], "env-gemini-secret-key-999")
                self.assertIsNone(seen_turns[-1]["project"])
                jm1.shutdown()
                jm2.shutdown()

            # 2b. Argument-only API key across server restart must NOT silently flip to ambient GOOGLE_CLOUD_PROJECT
            with unittest.mock.patch.dict(
                os.environ,
                {"GOOGLE_CLOUD_PROJECT": "ambient-gcloud-proj"},
                clear=False,
            ):
                os.environ.pop("GEMINI_API_KEY", None)
                os.environ.pop("GOOGLE_API_KEY", None)
                jm_a = JobManager(workspace_root=repo_dir, store=store, worker_backend=lambda _r: {"text": "ok"})
                arg_job = jm_a.delegate_start("arg key turn", api_key="AIzaArgOnlyKey000000000", wait=True)
                self.assertEqual(arg_job["auth_mode"], "api_key")
                jm_b = JobManager(workspace_root=repo_dir, store=store, worker_backend=lambda _r: {"text": "ok"})
                with self.assertRaises(ValueError) as ctx_err:
                    jm_b.delegate_continue(arg_job["job_id"], "continue after restart without key", wait=True)
                self.assertIn("api_key mode", str(ctx_err.exception))
                jm_a.shutdown()
                jm_b.shutdown()

            # 3. Placeholder env sanitization
            with unittest.mock.patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "${user_config.gemini_api_key}"},
                clear=False,
            ):
                sanitize_placeholder_env()
                self.assertNotIn("GEMINI_API_KEY", os.environ)

            # 4. Non-string api_key / project in MCP tools/call returns -32602
            jm_mcp = JobManager(workspace_root=repo_dir, store=store, worker_backend=lambda _r: {"text": "ok"})
            server = MCPServer(job_manager=jm_mcp, store=store)
            bad_resp = server.handle_Message(
                {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "method": "tools/call",
                    "params": {
                        "name": "ask_gemini",
                        "arguments": {"prompt": "hi", "api_key": 12345},
                    },
                }
            )
            self.assertEqual(bad_resp["error"]["code"], -32602)
            self.assertIn("api_key", bad_resp["error"]["message"])
            jm_mcp.shutdown()


class TestGauntletRound1Regressions(unittest.TestCase):
    """Regression suite covering all Round 1 adversarial findings across Opus, Fable, and Argon."""

    def test_non_ascii_and_shell_metachar_filenames_and_in_place_branch_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _init_git_repo(repo)
            (repo / "r\u00e9sum\u00e9.py").write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "non-ascii"], cwd=repo, check=True, capture_output=True)

            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo)
            # Verify worker worktree is isolated outside store.data_dir so ../../.hmac_key is unreachable
            wt_sample = store.worktree_path("job-1")
            self.assertNotEqual((wt_sample / ".." / ".." / ".hmac_key").resolve(), store.hmac_key_path.resolve())

            def isolated_worker(req: dict[str, Any]) -> dict[str, Any]:
                wd = Path(req["workspace_dir"])
                (wd / "r\u00e9sum\u00e9.py").write_text("x = 2\n", encoding="utf-8")
                (wd / "caf\u00e9.py").write_text("c = 3\n", encoding="utf-8")
                return {"text": "updated non-ascii"}

            jm = JobManager(workspace_root=repo, store=store, worker_backend=isolated_worker)
            res = jm.delegate_start("non-ascii isolated", project="p", wait=True)
            self.assertEqual(res["status"], "ok")
            self.assertIn("r\u00e9sum\u00e9.py", res["files_changed"])
            self.assertIn("caf\u00e9.py", res["files_changed"])
            diff_page = jm.get_diff(res["job_id"], file_filter="caf\u00e9.py")
            self.assertIn("+c = 3", diff_page["diff"])
            ap = jm.delegate_apply(res["job_id"])
            self.assertTrue(ap["applied"])
            self.assertEqual((repo / "r\u00e9sum\u00e9.py").read_text(encoding="utf-8"), "x = 2\n")
            self.assertEqual((repo / "caf\u00e9.py").read_text(encoding="utf-8"), "c = 3\n")
            jm.shutdown()

        # Now test in_place mode with shell metacharacters in filename and worker branch switching
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _init_git_repo(repo)
            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo)

            def inplace_worker(req: dict[str, Any]) -> dict[str, Any]:
                wd = Path(req["workspace_dir"])
                subprocess.run(["git", "checkout", "-b", "evil-branch"], cwd=wd, check=True, capture_output=True)
                (wd / "my notes.py").write_text("a = 1\n", encoding="utf-8")
                (wd / "$(touch PWNED).py").write_text("b = 2\n", encoding="utf-8")
                subprocess.run(["git", "add", "."], cwd=wd, check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", "branch commit"], cwd=wd, check=True, capture_output=True)
                return {"text": "in-place done"}

            jm = JobManager(workspace_root=repo, store=store, worker_backend=inplace_worker)
            res = jm.delegate_start("in-place branch & metachar", project="p", in_place=True, wait=True)
            self.assertEqual(res["status"], "ok")
            cur_branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(cur_branch, "main")
            branches = subprocess.run(
                ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads/"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            self.assertEqual(branches, ["main"])
            # Executing undo_hint must cleanly remove the files without executing $(touch PWNED)
            self.assertIsNotNone(res.get("undo_hint"))
            subprocess.run(["bash", "-c", res["undo_hint"]], cwd=repo, check=True, capture_output=True)
            self.assertFalse((repo / "PWNED").exists())
            self.assertFalse((repo / "my notes.py").exists())
            self.assertFalse((repo / "$(touch PWNED).py").exists())
            jm.shutdown()

    def test_guard_aliasing_sed_ln_glob_and_linter_prefixes(self) -> None:
        from server.jobs import _matches_file_filter, validate_acceptance
        from server.receipts import extract_diff_provenance, redact_api_keys

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            _init_git_repo(ws)
            out_dir = ws / ".cc-agy-data"

            for cmd in (
                "python3 -c \"import os; os.system('touch pwned')\"",
                "python3 -c \"import os as o; o.system('touch pwned')\"",
                "python3 -c \"from os import system; system('touch pwned')\"",
                "python3 -c \"import importlib; importlib.import_module('os').system('touch pwned')\"",
                "node -e \"require('child_process').execSync('touch pwned')\"",
                "sed -E -i \"s/a/b/\" app.py",
                "sed -e \"s/a/b/\" -i app.py",
                "python3 -c\"open('app.py','w').write('x')\"",
                "ln -s /tmp/evil.py",
            ):
                verdict = detect_bash_workspace_mutation(cmd, ws, out_dir, cwd=ws)
                self.assertTrue(verdict["blocked"], f"Expected blocked=True for {cmd!r}")
                with self.assertRaises(ValueError):
                    validate_acceptance([cmd], ws)

            env = {"CC_AGY_STRICT_MODE": "true", "CLAUDE_PROJECT_DIR": str(ws), "CC_AGY_OUT_DIR": str(out_dir)}
            for tool_payload in (
                {"tool_name": "Glob", "tool_input": {"path": str(ws), "pattern": ".cc-agy-data/.hmac_key"}, "cwd": str(ws)},
                {"tool_name": "Grep", "tool_input": {"path": str(ws), "pattern": ".*", "glob": ".cc-agy-data/.hmac_key"}, "cwd": str(ws)},
            ):
                decision = evaluate_pre_tool_use(tool_payload, env=env)
                self.assertEqual(decision["hookSpecificOutput"]["permissionDecision"], "deny")

            # Non-dict JSON payloads must not crash hook entrypoints
            self.assertEqual(evaluate_pre_tool_use([]), {})  # type: ignore[arg-type]

        # Blockquoted and bulleted 10-line code blocks must be caught by lint_delegation_prompt
        fn_lines = [
            "def apply_discount(order):",
            "    total = 0",
            "    for item in order.items:",
            "        total += item.price * item.qty",
            "    if order.coupon:",
            "        total *= 0.9",
            "    if total > 100:",
            "        total -= 5",
            "    order.total = round(total, 2)",
            "    return order",
        ]
        bq_code = "\n".join(f"> {l}" for l in fn_lines)
        bullet_code = "\n".join(f"- `{l.strip()}`" for l in bq_code.splitlines())
        self.assertFalse(lint_delegation_prompt(bq_code)["ok"])
        self.assertFalse(lint_delegation_prompt(bullet_code)["ok"])
        prose_parens = "\n".join(f"Ensure helper_{i}(x) returns the right value" for i in range(9))
        self.assertTrue(lint_delegation_prompt(prose_parens)["ok"])

        # Short API key redaction must not corrupt JSON keys
        rec = {"status": "ok", "api_key": "st", "model": "gemini-2.5-flash"}
        redacted = redact_api_keys(rec, api_key="st")
        self.assertIn("status", redacted)
        self.assertEqual(redacted["api_key"], "<api-key>")

        # _matches_file_filter must not strip leading dots from dotfiles
        self.assertFalse(_matches_file_filter(".env", "env"))
        self.assertFalse(_matches_file_filter("env", ".env"))
        self.assertFalse(_matches_file_filter(".github/ci.yml", "github/ci.yml"))

        # extract_diff_provenance must hash ++ / -- content lines inside a hunk
        sample_diff = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-x = 1\n"
            "---foo\n"
            "+++bar\n"
            "+y = 2\n"
        )
        prov = extract_diff_provenance(sample_diff)
        self.assertEqual(prov["per_file"]["a.py"]["added_count"], 2)
        self.assertEqual(prov["per_file"]["a.py"]["removed_count"], 2)

    def test_workspace_scoped_status_cancel_continue_and_mcp_boundary(self) -> None:
        from server.agy_runner import append_progress, build_straitjacket_tools

        with tempfile.TemporaryDirectory() as tmp:
            repo1 = Path(tmp) / "repo1"
            repo2 = Path(tmp) / "repo2"
            repo1.mkdir()
            repo2.mkdir()
            _init_git_repo(repo1)
            _init_git_repo(repo2)
            shared_data = Path(tmp) / "shared_data"
            s1 = ReceiptStore(shared_data, workspace_root=repo1)
            s2 = ReceiptStore(shared_data, workspace_root=repo2)

            s1.write_receipt(
                "j-chk",
                {"kind": "delegate", "status": "checkpoint", "model": "gemini-2.5-flash", "usage": {"prompt_token_count": 100}},
                workspace_root=str(repo1),
            )
            s2.write_receipt(
                "j-other",
                {"kind": "delegate", "status": "ok", "model": "gemini-2.5-flash", "usage": {"prompt_token_count": 9999}},
                workspace_root=str(repo2),
            )
            tot1 = s1.summarize_costs()
            self.assertEqual(tot1["total_jobs"], 1)
            self.assertEqual(tot1["in_progress_or_paused_jobs"], 1)
            self.assertEqual(tot1["prompt_token_count"], 100)

            jm = JobManager(workspace_root=repo1, store=s1, worker_backend=lambda _r: {"text": "ok"})
            # delegate_continue after delegate_cancel must succeed
            j = jm.delegate_start("first", project="p", wait=True)
            jm.delegate_cancel(j["job_id"])
            resumed = jm.delegate_continue(j["job_id"], "resume after cancel", wait=True)
            self.assertEqual(resumed["status"], "ok")

            # Missing context_files in ask_gemini raises ValueError; valid context_files are echoed
            with self.assertRaises(ValueError):
                ask_gemini(
                    "review",
                    context_files=["missing_file.py"],
                    project="p",
                    workspace_root=repo1,
                    store=s1,
                    backend=lambda _r: {"text": "ok"},
                )

            # MCP boundary checks: Infinity page, string enabled_tools, string plan, in_place+auto_apply=False, needs_input not is_error
            def q_worker(req: dict[str, Any]) -> dict[str, Any]:
                append_progress(req["progress_path"], "question", "clarify?")
                return {"text": "waiting for answer"}

            jm_q = JobManager(workspace_root=repo1, store=s1, worker_backend=q_worker)
            srv = MCPServer(job_manager=jm_q, store=s1)
            for bad_call in (
                {"name": "get_diff", "arguments": {"job_id": j["job_id"], "page": float("inf")}},
                {"name": "delegate_start", "arguments": {"prompt": "do", "project": "p", "enabled_tools": "RUN_COMMAND"}},
                {"name": "delegate_start", "arguments": {"prompt": "do", "project": "p", "plan": "not-a-list"}},
                {"name": "delegate_to_gemini", "arguments": {"prompt": "do", "project": "p", "in_place": True, "auto_apply": False}},
            ):
                resp = srv.handle_Message({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": bad_call})
                self.assertEqual(resp["error"]["code"], -32602, f"Expected -32602 for {bad_call}")

            d_needs = srv._dispatch_tool("delegate_to_gemini", {"prompt": "ask q", "project": "p"})
            self.assertEqual(d_needs["status"], "needs_input")
            self.assertFalse(d_needs.get("is_error", False))

            # Straitjacket tool argument coercion and env scrubbing
            with unittest.mock.patch.dict(os.environ, {"CC_AGY_DATA_DIR": str(shared_data), "GEMINI_API_KEY": "secret-key-val"}):
                tools, _ = build_straitjacket_tools(repo1, timeout_seconds=10.0)
                tmap = {f.__name__: f for f in tools}
                out_none = asyncio.run(tmap["ctx_get"]("run:deadbeef#stdout", None, None))
                self.assertIn("error", out_none)
                out_str = asyncio.run(tmap["ctx_get"]("run:deadbeef#stdout", "abc", "5.0"))
                self.assertIn("error", out_str)
                out_cmd = json.loads(asyncio.run(tmap["run_command"]('printf "%s|%s" "$CC_AGY_DATA_DIR" "$GEMINI_API_KEY"', None, 0)))
                self.assertEqual(out_cmd.get("command_exit_code"), 0)
                self.assertEqual(out_cmd.get("stdout", "").strip(), "|")

            jm.shutdown()
            jm_q.shutdown()

    def test_round2_gauntlet_regressions(self) -> None:
        from server.jobs import MAX_TOOL_RESPONSE_TOKENS, validate_acceptance
        from server.regions import estimate_tokens, resolve_api_key, resolve_auth_config, resolve_vertex_project
        import stat
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _init_git_repo(repo)
            data_dir = Path(tmp) / "data"
            store = ReceiptStore(data_dir, workspace_root=repo)

            # 1. Worktrees dir mode is 0o700 and classified as out_dir
            self.assertEqual(stat.S_IMODE(os.stat(store.worktrees_dir).st_mode), 0o700)
            wt_file = store.worktrees_dir / "job-x" / "app.py"
            self.assertEqual(classify_path(str(wt_file), repo, str(data_dir), cwd=repo), "out_dir")
            self.assertTrue(
                detect_bash_workspace_mutation(f"cp /tmp/x {wt_file}", repo, str(data_dir), cwd=repo)["blocked"]
            )

            # 2. One worker turn with 1M accumulated input tokens + 100k output tokens on Argon costs $3.20
            turn_u = {
                "prompt_token_count": 1_000_000,
                "cached_content_token_count": 0,
                "candidates_token_count": 100_000,
                "thoughts_token_count": 0,
                "total_token_count": 1_100_000,
            }
            jm_cost = JobManager(
                workspace_root=repo,
                store=store,
                worker_backend=lambda r: (
                    r["workspace_dir"],
                    {"text": "ok", "usage": {**turn_u, "max_request_prompt_tokens": 1_000_000, "requests": [dict(turn_u)]}},
                )[1],
            )
            res_cost = jm_cost.delegate_start("argon turn", model="gemini-4-argon", project="p", wait=True)
            self.assertAlmostEqual(res_cost["cost"]["total_cost_usd"], 3.20, places=4)
            jm_cost.shutdown()

            # Partial per-call requests must not undercount turn totals, and over-reported per-step sums scale down to turn totals
            partial_u = {
                "prompt_token_count": 1_000_000,
                "candidates_token_count": 100_000,
                "turns": [{"prompt_token_count": 1_000_000, "candidates_token_count": 100_000}],
                "requests": [{"prompt_token_count": 50_000, "candidates_token_count": 1_000, "per_call": True}],
            }
            self.assertAlmostEqual(estimate_cost_usd("gemini-4-argon", partial_u)["total_cost_usd"], 3.20, places=4)
            over_u1 = {
                "prompt_token_count": 1_000_000,
                "candidates_token_count": 100_000,
                "requests": [
                    {"prompt_token_count": 1_000_000, "candidates_token_count": 100_000, "per_call": True},
                    {"prompt_token_count": 1_000_000, "candidates_token_count": 100_000, "per_call": True},
                ],
            }
            self.assertAlmostEqual(estimate_cost_usd("gemini-4-argon", over_u1)["total_cost_usd"], 5.80, places=4)
            over_u2 = {
                "prompt_token_count": 1_000_000,
                "candidates_token_count": 100_000,
                "requests": [{"prompt_token_count": 150_000, "candidates_token_count": 10_000, "per_call": True}] * 10,
            }
            self.assertAlmostEqual(estimate_cost_usd("gemini-4-argon", over_u2)["total_cost_usd"], 3.20, places=4)

            # 3. AST inline Python check blocks obfuscated mutations and allows benign inline prints
            for bad_inline in (
                "python3 -c \"import sys; sys.modules['o'+'s'].__dict__['sys'+'tem']('touch pwned')\"",
                "python3 -c \"print('x', file=__builtins__.__dict__['op'+'en']('pwned','w'))\"",
                "node -e \"require('child_pro'+'cess')['exec'+'Sync']('touch pwned')\"",
            ):
                self.assertTrue(detect_bash_workspace_mutation(bad_inline, repo, str(data_dir), cwd=repo)["blocked"])
                with self.assertRaises(ValueError):
                    validate_acceptance([bad_inline], repo)

            for good_inline in (
                "python3 -c \"import sys; print(sys.version)\"",
                "python3 -c \"print('hello os')\"",
            ):
                self.assertFalse(detect_bash_workspace_mutation(good_inline, repo, str(data_dir), cwd=repo)["blocked"])
                validate_acceptance([good_inline], repo)

            # 4. Bulleted `* code` lines are caught by lint_delegation_prompt
            fn = [
                "def f(a):",
                "    b = a + 1",
                "    c = b * 2",
                "    if c > 3:",
                "        c -= 1",
                "    d = c ** 2",
                "    e = d % 7",
                "    print(e)",
                "    return e",
            ]
            self.assertFalse(lint_delegation_prompt("\n".join(f"* {l}" for l in fn))["ok"])

            # 5. Auth precedence and fallback env vars
            self.assertEqual(resolve_api_key(None, env={"CLAUDE_PLUGIN_OPTION_GEMINI_API_KEY": "k123"}), "k123")
            self.assertEqual(resolve_vertex_project(None, env={"CLOUD_ML_PROJECT_ID": "p123"}), "p123")
            self.assertEqual(
                resolve_auth_config(requested_project="p1", requested_api_key="k1", env={})["auth_mode"],
                "api_key",
            )

            # 6. External worktree laundering when worker never touches workspace_dir fails provenance check
            gate = threading.Event()
            jm_launder = JobManager(
                workspace_root=repo,
                store=store,
                worker_backend=lambda _r: (gate.wait(5), {"text": "did nothing"})[1],
            )
            started = jm_launder.delegate_start("noop", project="p")
            wt_launder = Path(started["worktree_dir"])
            (wt_launder / "app.py").write_text("LAUNDERED = True\n", encoding="utf-8")
            gate.set()
            jm_launder.delegate_result(started["job_id"], wait=True, timeout=10)
            jm_launder.delegate_apply(started["job_id"])
            prov_verdict = verify_workspace_provenance(
                repo,
                store,
                env={"CC_AGY_STRICT_MODE": "true", "CC_AGY_DATA_DIR": str(data_dir), "CLAUDE_PROJECT_DIR": str(repo)},
                strict=True,
            )
            self.assertFalse(prov_verdict["ok"])
            jm_launder.shutdown()

            # 7. in_place worker committing on a branch and switching back preserves edits, and list status stays under 8k tokens
            repo_ip = Path(tmp) / "repo_ip"
            repo_ip.mkdir()
            _init_git_repo(repo_ip)
            (repo_ip / "a.py").write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo_ip, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "add a"], cwd=repo_ip, check=True, capture_output=True)
            subprocess.run(["git", "checkout", "-b", "dev"], cwd=repo_ip, check=True, capture_output=True)
            store_ip = ReceiptStore(Path(tmp) / "data_ip", workspace_root=repo_ip)

            def branch_switch_worker(req: dict[str, Any]) -> dict[str, Any]:
                wd = Path(req["workspace_dir"])
                subprocess.run(["git", "checkout", "-b", "worker-feature"], cwd=wd, check=True, capture_output=True)
                (wd / "a.py").write_text("x = 2\n", encoding="utf-8")
                subprocess.run(["git", "commit", "-am", "w"], cwd=wd, check=True, capture_output=True)
                subprocess.run(["git", "checkout", "main"], cwd=wd, check=True, capture_output=True)
                return {"text": "ok"}

            jm_ip = JobManager(workspace_root=repo_ip, store=store_ip, worker_backend=branch_switch_worker)
            res_ip = jm_ip.delegate_start("e", project="p", wait=True, in_place=True)
            self.assertEqual(res_ip["status"], "ok")
            self.assertEqual((repo_ip / "a.py").read_text(encoding="utf-8").strip(), "x = 2")
            for idx in range(25):
                store_ip.write_receipt(
                    f"job-bulk-{idx}",
                    {"kind": "delegate", "status": "ok", "model": "gemini-3.5-flash-lite", "usage": {"prompt_token_count": 1000}},
                    workspace_root=str(repo_ip),
                )
            st_list = jm_ip.delegate_status(limit=50)
            self.assertLessEqual(
                estimate_tokens(json.dumps(st_list, indent=2, sort_keys=True)),
                MAX_TOOL_RESPONSE_TOKENS,
            )
            jm_ip.shutdown()

    def test_mcp_json_usercfg_separation_and_worktree_venv_symlink(self) -> None:
        """Verify .mcp.json does not clobber shell env vars and _prepare_worktree symlinks .venv safely."""
        mcp_cfg = json.loads((PLUGIN_ROOT / ".mcp.json").read_text(encoding="utf-8"))
        mcp_env = mcp_cfg["mcpServers"]["vertex"]["env"]
        for shell_var in (
            "GEMINI_API_KEY",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "CC_AGY_DEFAULT_DELEGATE_MODEL",
            "CC_AGY_DEFAULT_ASK_MODEL",
            "CC_AGY_STRICT_MODE",
            "CC_AGY_DATA_DIR",
        ):
            self.assertNotIn(
                shell_var,
                mcp_env,
                f".mcp.json must not clobber shell env var {shell_var!r}",
            )

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo_venv"
            repo.mkdir()
            _init_git_repo(repo)
            (repo / "pkg.py").write_text("VAL = 'host'\n", encoding="utf-8")
            (repo / "test_pkg.py").write_text(
                "import unittest\nimport pkg\nclass T(unittest.TestCase):\n    def test_v(self):\n        self.assertEqual(pkg.VAL, 'worker_wt')\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "add pkg"], cwd=repo, check=True, capture_output=True)

            # Create a dummy .venv directory in repo
            venv_bin = repo / ".venv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "marker").write_text("ok\n", encoding="utf-8")

            store = ReceiptStore(Path(tmp) / "data_venv", workspace_root=repo)

            def edit_pkg_worker(req: dict[str, Any]) -> dict[str, Any]:
                wd = Path(req["workspace_dir"])
                assert (wd / ".venv").is_symlink(), "Expected .venv symlink inside worktree"
                (wd / "pkg.py").write_text("VAL = 'worker_wt'\n", encoding="utf-8")
                return {"text": "updated pkg"}

            old_env = dict(os.environ)
            try:
                os.environ.update(mcp_env)
                os.environ["CC_AGY_DEFAULT_DELEGATE_MODEL"] = "gemini-4-argon"
                os.environ["GOOGLE_CLOUD_PROJECT"] = "test-proj-argon"
                jm = JobManager(workspace_root=repo, store=store, worker_backend=edit_pkg_worker)
                res = jm.delegate_start(
                    "update pkg",
                    wait=True,
                    acceptance=["python3 -m unittest test_pkg.py"],
                )
                self.assertEqual(res["status"], "ok")
                self.assertEqual(res["model"], "gemini-4-argon")
                self.assertEqual(res["files_changed"], ["pkg.py"])
                self.assertTrue(res["acceptance_passed"])
                jm.delegate_apply(res["job_id"])
                (repo / ".pytest_cache").mkdir(exist_ok=True)
                (repo / ".pytest_cache" / "README.md").write_text("pytest cache\n", encoding="utf-8")
                prov_res = verify_workspace_provenance(
                    repo,
                    store,
                    env={"CC_AGY_STRICT_MODE": "true", "CC_AGY_DATA_DIR": str(store.data_dir), "CLAUDE_PROJECT_DIR": str(repo)},
                    strict=True,
                )
                self.assertTrue(prov_res["ok"], f"Expected provenance ok even with .pytest_cache, got {prov_res}")
                jm.shutdown()
            finally:
                os.environ.clear()
                os.environ.update(old_env)

    def test_cancelled_error_timeout_recovery_and_sqlite_telemetry_and_readonly_linters(self) -> None:
        from server.agy_runner import drain_response_streams, extract_sqlite_session_telemetry
        from server.jobs import validate_acceptance

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _init_git_repo(repo)
            (repo / "pkg.py").write_text("VAL = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

            # 1. Verify drain_response_streams catches asyncio.CancelledError after buffered items
            class _CancelledStream:
                def __aiter__(self):
                    async def _gen():
                        yield {"name": "edit_file", "args": {"path": "pkg.py"}}
                        raise asyncio.CancelledError()
                    return _gen()

            class _FakeResp:
                tool_calls = _CancelledStream()
                thoughts = None

            tc_out: list[dict[str, Any]] = []
            th_out: list[str] = []
            asyncio.run(drain_response_streams(_FakeResp(), tc_out, th_out))
            self.assertEqual(len(tc_out), 1)
            self.assertEqual(tc_out[0]["name"], "edit_file")

            # 2. Verify JobManager and MCPServer handle asyncio.CancelledError (BaseException) with partial usage & clean edits
            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo)

            def cancelled_worker(req: dict[str, Any]) -> dict[str, Any]:
                wd = Path(req["workspace_dir"])
                (wd / "pkg.py").write_text("VAL = 2\n", encoding="utf-8")
                # Simulate Straitjacket creating .ctx-session-reads/engagement.json inside the worktree
                ctx_dir = wd / ".ctx-session-reads"
                ctx_dir.mkdir(exist_ok=True)
                (ctx_dir / "engagement.json").write_text('{"activated_by": "truncation"}\n', encoding="utf-8")
                exc = asyncio.CancelledError("Simulated SDK wait_for cancellation")
                setattr(
                    exc,
                    "partial_usage",
                    {
                        "prompt_token_count": 12000,
                        "cached_content_token_count": 8000,
                        "candidates_token_count": 400,
                        "thoughts_token_count": 200,
                        "total_token_count": 12600,
                        "requests": [
                            {
                                "prompt_token_count": 12000,
                                "cached_content_token_count": 8000,
                                "candidates_token_count": 400,
                                "thoughts_token_count": 200,
                                "total_token_count": 12600,
                                "per_call": True,
                            }
                        ],
                        "max_request_prompt_tokens": 12000,
                    },
                )
                setattr(exc, "partial_tool_calls", [{"name": "replace_file_content", "args": {}}])
                raise exc

            jm = JobManager(workspace_root=repo, store=store, worker_backend=cancelled_worker)
            digest = jm.delegate_start(
                "edit pkg",
                model="gemini-3.8-flash",
                project="p",
                allowed_paths=["pkg.py"],
                wait=True,
            )
            self.assertEqual(digest["status"], "timeout")
            self.assertEqual(digest["scope_violations"], [])
            self.assertEqual(digest["usage"]["total_token_count"], 12600)
            self.assertEqual(len(digest["usage"]["requests"]), 1)
            self.assertEqual(digest["tool_call_count"], 1)
            self.assertEqual(digest["files_changed"], ["pkg.py"])
            self.assertGreater(digest["cost"]["total_cost_cents"], 0.0)

            # Verify timeout job with clean edits can be applied after diff inspection
            apply_res = jm.delegate_apply(digest["job_id"])
            self.assertTrue(apply_res["applied"])
            self.assertEqual((repo / "pkg.py").read_text(encoding="utf-8"), "VAL = 2\n")
            self.assertFalse((repo / ".ctx-session-reads").exists())
            jm.shutdown()

            # 3. Verify validate_acceptance allows python -m mypy / ruff check / ruff format --check / pytest -p no:cacheprovider and blocks ruff --fix / pytest -p evil_mod
            allowed_cmds = validate_acceptance(
                [
                    ".venv/bin/python -m mypy pkg.py",
                    ".venv/bin/python -m ruff check pkg.py",
                    "ruff format --check pkg.py",
                    ".venv/bin/pytest -q -p no:cacheprovider pkg.py",
                ],
                repo,
            )
            self.assertEqual(len(allowed_cmds), 4)
            with self.assertRaises(ValueError):
                validate_acceptance(["ruff format pkg.py"], repo)
            with self.assertRaises(ValueError):
                validate_acceptance([".venv/bin/python -m ruff check --fix pkg.py"], repo)
            with self.assertRaises(ValueError):
                validate_acceptance([".venv/bin/pytest -p evil_mod pkg.py"], repo)

    def test_bugs_6_7_8_9_tls_keylog_index_reset_digest_cap_and_scope_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@t.local"], cwd=str(repo), check=True)
            subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), check=True)
            (repo / "pkg.py").write_text("VAL = 1\n", encoding="utf-8")
            (repo / "extra.py").write_text("EXTRA = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True)

            store = ReceiptStore(repo / ".cc-agy-data", workspace_root=repo)

            # 1. Bug #6 + #7: TLS keylog 'test', .coverage.worker1, and 150 requests in usage
            def worker_with_tls_and_many_reqs(req: dict) -> dict:
                wd = Path(req["workspace_dir"])
                (wd / "pkg.py").write_text("VAL = 2\n", encoding="utf-8")
                (wd / "test").write_text(
                    "# TLS secrets log file, generated by OpenSSL / Python\nCLIENT_RANDOM deadbeef 0123\n",
                    encoding="utf-8",
                )
                (wd / ".coverage.worker1").write_text("covdata", encoding="utf-8")
                reqs = [
                    {
                        "prompt_token_count": 1000,
                        "cached_content_token_count": 800,
                        "candidates_token_count": 50,
                        "thoughts_token_count": 20,
                        "total_token_count": 1070,
                        "per_call": True,
                    }
                    for _ in range(150)
                ]
                return {
                    "text": "updated pkg.py",
                    "usage": {
                        "prompt_token_count": 150000,
                        "cached_content_token_count": 120000,
                        "candidates_token_count": 7500,
                        "thoughts_token_count": 3000,
                        "total_token_count": 160500,
                        "requests": reqs,
                    },
                    "tool_calls": [{"name": "edit_file", "args": {"path": "pkg.py"}}],
                }

            jm = JobManager(workspace_root=repo, store=store, worker_backend=worker_with_tls_and_many_reqs)
            digest = jm.delegate_start(
                "update pkg",
                model="gemini-4-argon",
                project="p",
                allowed_paths=["pkg.py"],
                wait=True,
            )
            self.assertEqual(digest["status"], "ok")
            self.assertEqual(digest["files_changed"], ["pkg.py"])
            self.assertEqual(digest["scope_violations"], [])
            # Digest caps usage.requests to 10 while recording requests_count=150
            self.assertEqual(digest["usage"]["requests_count"], 150)
            self.assertTrue(digest["usage"]["requests_truncated"])
            self.assertEqual(len(digest["usage"]["requests"]), 10)
            # Full on-disk receipt preserves all 150 requests
            full_rcpt = store.read_receipt(digest["job_id"])
            self.assertEqual(len(full_rcpt["usage"]["requests"]), 150)

            # Simulate pytest on host creating OpenSSL TLS keylog 'test' before apply
            (repo / "test").write_text(
                "# TLS secrets log file, generated by OpenSSL / Python\nCLIENT_RANDOM hostbeef 9999\n",
                encoding="utf-8",
            )
            apply_res = jm.delegate_apply(digest["job_id"])
            self.assertTrue(apply_res["applied"])
            prov = verify_workspace_provenance(repo, store, strict=True)
            self.assertTrue(prov["ok"], f"Expected provenance ok=True, got: {prov}")
            jm.shutdown()

            # 2. Bug #8: scope_violation override via delegate_apply(allow_scope_violations=True) and delegate_continue(allowed_paths=...)
            def worker_out_of_scope(req: dict) -> dict:
                wd = Path(req["workspace_dir"])
                (wd / "pkg.py").write_text("VAL = 3\n", encoding="utf-8")
                (wd / "extra.py").write_text("EXTRA = 2\n", encoding="utf-8")
                return {
                    "text": "updated both",
                    "usage": {"prompt_token_count": 100, "candidates_token_count": 20, "total_token_count": 120},
                    "tool_calls": [],
                }

            jm2 = JobManager(workspace_root=repo, store=store, worker_backend=worker_out_of_scope)
            d2 = jm2.delegate_start(
                "update both",
                model="gemini-3.8-flash",
                project="p",
                allowed_paths=["pkg.py"],
                wait=True,
            )
            self.assertEqual(d2["status"], "scope_violation")
            self.assertEqual(d2["scope_violations"], ["extra.py"])
            with self.assertRaises(ValueError):
                jm2.delegate_apply(d2["job_id"])

            # Widen allowed_paths via delegate_continue
            d2_cont = jm2.delegate_continue(
                d2["job_id"],
                "verify both files",
                allowed_paths=["pkg.py", "extra.py"],
                wait=True,
            )
            self.assertEqual(d2_cont["status"], "ok")
            self.assertEqual(d2_cont["scope_violations"], [])
            jm2.delegate_apply(d2["job_id"])
            self.assertEqual((repo / "extra.py").read_text(encoding="utf-8"), "EXTRA = 2\n")

            # Direct allow_scope_violations=True on delegate_apply
            def worker_out_of_scope_2(req: dict) -> dict:
                wd = Path(req["workspace_dir"])
                (wd / "extra.py").write_text("EXTRA = 3\n", encoding="utf-8")
                return {
                    "text": "updated extra",
                    "usage": {"prompt_token_count": 100, "candidates_token_count": 20, "total_token_count": 120},
                    "tool_calls": [],
                }

            jm3 = JobManager(workspace_root=repo, store=store, worker_backend=worker_out_of_scope_2)
            d3 = jm3.delegate_start(
                "update extra",
                model="gemini-3.8-flash",
                project="p",
                allowed_paths=["pkg.py"],
                wait=True,
            )
            self.assertEqual(d3["status"], "scope_violation")
            res3 = jm3.delegate_apply(d3["job_id"], allow_scope_violations=True)
            self.assertTrue(res3["applied"])
            self.assertEqual((repo / "extra.py").read_text(encoding="utf-8"), "EXTRA = 3\n")
            prov3 = verify_workspace_provenance(repo, store, strict=True)
            self.assertTrue(prov3["ok"], f"Expected provenance ok=True after allow_scope_violations, got: {prov3}")
            jm2.shutdown()
            jm3.shutdown()

            # 3. Bug #10: Concurrent ReceiptStore initialization locks .hmac_key and reloads on-disk key
            race_dir = Path(tmp) / "race_data"
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                stores = list(pool.map(lambda _: ReceiptStore(race_dir, workspace_root=repo), range(8)))
            keys = {s._active_hmac_key() for s in stores}
            self.assertEqual(len(keys), 1)
            rcpt_a = stores[0].write_receipt("job-race-1", {"kind": "delegate", "status": "ok", "model": "gemini-4-argon"})
            self.assertTrue(stores[-1].verify_receipt_signature(rcpt_a))

    def test_disable_web_tools_and_timeout_containment_preservation(self) -> None:
        """Verifies CC_AGY_DISABLE_WEB_TOOLS / CC_AGY_STRICT_MODE blocks outbound network commands
        in Straitjacket run_command and preserves partial_containment on worker TimeoutError."""
        from unittest import mock
        from server.agy_runner import build_straitjacket_tools

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
            (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

            import inspect
            with mock.patch.dict(os.environ, {"CC_AGY_STRICT_MODE": "true"}, clear=False):
                os.environ.pop("CC_AGY_BLOCKED_PATHS", None)
                os.environ.pop("CC_AGY_DEFAULT_CMD_TIMEOUT", None)
                tools, _ = build_straitjacket_tools(repo, timeout_seconds=600.0)
                run_cmd_fn = tools[0]
                self.assertEqual(
                    inspect.signature(run_cmd_fn).parameters["timeout_seconds"].default,
                    600.0,
                )
                blocked_json = asyncio.run(
                    run_cmd_fn("python -c \"import urllib.request; urllib.request.urlopen('https://huggingface.co')\"")
                )
                blocked_obj = json.loads(blocked_json)
                self.assertEqual(blocked_obj["status"], "error")
                self.assertEqual(blocked_obj["exit_code"], 126)
                self.assertIn("Outbound network and web fetching are disabled", blocked_obj["error"])

                # Offline httpx.Client with MockTransport must be allowed
                httpx_mock_json = asyncio.run(
                    run_cmd_fn("echo \"python3 -c 'import httpx; c = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))'\"")
                )
                self.assertEqual(json.loads(httpx_mock_json).get("command_exit_code"), 0)

                # Ordinary reads like ls ../ and cat /etc/os-release are allowed when CC_AGY_BLOCKED_PATHS is unset
                parent_ls_json = asyncio.run(run_cmd_fn("ls ../"))
                self.assertEqual(json.loads(parent_ls_json).get("command_exit_code"), 0)

                wd_escape_json = asyncio.run(
                    run_cmd_fn("ls -la", working_directory="/tmp")
                )
                wd_escape_obj = json.loads(wd_escape_json)
                self.assertEqual(wd_escape_obj["status"], "error")
                self.assertEqual(wd_escape_obj["exit_code"], 126)
                self.assertIn("working_directory must stay within the isolated workspace", wd_escape_obj["error"])

                cross_wt_json = asyncio.run(
                    run_cmd_fn("ls -d /tmp/cc-agy-worktrees-*")
                )
                cross_wt_obj = json.loads(cross_wt_json)
                self.assertEqual(cross_wt_obj["status"], "error")
                self.assertEqual(cross_wt_obj["exit_code"], 126)

                allowed_json = asyncio.run(run_cmd_fn("echo local_ok"))
                self.assertIn("local_ok", allowed_json)

            with mock.patch.dict(
                os.environ,
                {
                    "CC_AGY_STRICT_MODE": "true",
                    "CC_AGY_BLOCKED_PATHS": "/usr/local,/tmp/.brench",
                    "CC_AGY_DEFAULT_CMD_TIMEOUT": "75",
                },
                clear=False,
            ):
                tools_bp, _ = build_straitjacket_tools(repo, timeout_seconds=600.0)
                run_cmd_bp = tools_bp[0]
                self.assertEqual(
                    inspect.signature(run_cmd_bp).parameters["timeout_seconds"].default,
                    75.0,
                )
                host_escape_json = asyncio.run(
                    run_cmd_bp("find /usr/local/share/ -name '*ewald*'")
                )
                host_escape_obj = json.loads(host_escape_json)
                self.assertEqual(host_escape_obj["status"], "error")
                self.assertEqual(host_escape_obj["exit_code"], 126)
                self.assertIn("Accessing host filesystem directories outside the isolated workspace", host_escape_obj["error"])

            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo)

            def timeout_worker(req: dict) -> dict:
                _ = req["workspace_dir"]
                exc = TimeoutError("worker timed out")
                setattr(exc, "partial_usage", {"prompt_token_count": 500, "total_token_count": 500})
                setattr(exc, "partial_tool_calls", [{"name": "view_file", "args": {"path": "a.py"}}])
                setattr(exc, "partial_containment", "straitjacket")
                raise exc

            jm = JobManager(workspace_root=repo, store=store, worker_backend=timeout_worker)
            digest = jm.delegate_start("test timeout", model="gemini-4-argon", project="p", wait=True)
            self.assertEqual(digest["status"], "timeout")
            rcpt = store.read_receipt(digest["job_id"])
            self.assertEqual(rcpt["containment"], "straitjacket")
            jm.shutdown()

    def test_preload_files_and_qa_review_pipeline_in_delegate_to_gemini(self) -> None:
        from server.genai_ask import _close_genai_client
        from server.jobs import parse_qa_audit_verdict, select_candidate_files

        class DummyClient:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        dc = DummyClient()
        _close_genai_client(dc)
        self.assertTrue(dc.closed)

        self.assertTrue(parse_qa_audit_verdict("VERDICT: PASS\nLooks great.")["passed"])
        nf = parse_qa_audit_verdict("VERDICT: NEEDS_FIXES\n- ISSUE: app.py:1: missing exclamation")
        self.assertFalse(nf["passed"])
        self.assertEqual(len(nf["issues"]), 1)

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _init_git_repo(repo)
            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo)

            cands = select_candidate_files(repo, "Update greet in app.py")
            self.assertEqual([rel for rel, _ in cands], ["app.py"])

            worker_calls: list[dict] = []

            def fake_worker(req: dict) -> dict:
                worker_calls.append(req)
                wt = Path(req["workspace_dir"])
                if len(worker_calls) == 1:
                    (wt / "app.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
                else:
                    (wt / "app.py").write_text("def greet():\n    return 'hi!'\n", encoding="utf-8")
                return {
                    "text": "updated app.py",
                    "usage": {"prompt_token_count": 200, "candidates_token_count": 50, "total_token_count": 250},
                    "tool_calls": [],
                    "thoughts": [],
                }

            ask_calls: list[dict] = []

            def fake_ask(req: dict) -> dict:
                ask_calls.append(req)
                return {
                    "text": "VERDICT: NEEDS_FIXES\n- ISSUE: app.py:2: return 'hi!' with exclamation",
                    "usage": {"prompt_token_count": 300, "candidates_token_count": 40, "thoughts_token_count": 60, "total_token_count": 400},
                }

            jm = JobManager(workspace_root=repo, store=store, worker_backend=fake_worker)
            srv = MCPServer(job_manager=jm, store=store, ask_backend=fake_ask)
            resp = srv.handle_Message(
                {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "method": "tools/call",
                    "params": {
                        "name": "delegate_to_gemini",
                        "arguments": {
                            "prompt": "Update greet in app.py to return 'hi!'",
                            "model": "gemini-3.8-flash",
                            "project": "test-proj",
                            "preload_files": True,
                            "qa_review_model": "gemini-4-argon",
                            "auto_apply": True,
                        },
                    },
                }
            )
            payload = json.loads(resp["result"]["content"][0]["text"])
            self.assertEqual(payload["status"], "ok")
            self.assertTrue(payload["applied"])
            self.assertEqual(payload.get("preloaded_files"), ["app.py"])
            self.assertIn("## Candidate Repository Files", worker_calls[0]["prompts"][0])
            self.assertIn("- `app.py`", worker_calls[0]["prompts"][0])
            # Path hints only: file contents must not be pasted into turn 1.
            self.assertNotIn("def greet():", worker_calls[0]["prompts"][0])
            # Fix pass is a fresh worker session with a compact context note, not a replay of turn 1.
            self.assertEqual(len(worker_calls[1]["prompts"]), 1)
            self.assertIn("[Prior turn context", worker_calls[1]["prompts"][0])
            self.assertNotIn("## Candidate Repository Files", worker_calls[1]["prompts"][0])
            self.assertEqual(len(worker_calls), 2)
            self.assertEqual(len(ask_calls), 1)
            self.assertEqual(ask_calls[0].get("thinking_level"), "HIGH")
            self.assertTrue(payload["qa_review"]["resumed"])
            self.assertFalse(payload["qa_review"]["passed"])
            self.assertEqual((repo / "app.py").read_text(encoding="utf-8"), "def greet():\n    return 'hi!'\n")

            # Also verify checkpoint status on Turn 1 triggers qa_review_model + delegate_continue
            checkpoint_calls: list[dict] = []

            def fake_checkpoint_worker(req: dict) -> dict:
                checkpoint_calls.append(req)
                wt = Path(req["workspace_dir"])
                if len(checkpoint_calls) == 1:
                    (wt / "app.py").write_text("def greet():\n    return 'partial'\n", encoding="utf-8")
                    return {
                        "text": "hit cap",
                        "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
                        "usage": {"prompt_token_count": 100, "candidates_token_count": 20, "total_token_count": 120},
                        "tool_calls": [{"name": "view_file", "args": {"path": "app.py"}}],
                        "thoughts": [],
                    }
                (wt / "app.py").write_text("def greet():\n    return 'completed!'\n", encoding="utf-8")
                return {
                    "text": "completed",
                    "status": "ok",
                    "usage": {"prompt_token_count": 120, "candidates_token_count": 30, "total_token_count": 150},
                    "tool_calls": [],
                    "thoughts": [],
                }

            jm_cp = JobManager(workspace_root=repo, store=store, worker_backend=fake_checkpoint_worker)
            srv_cp = MCPServer(
                job_manager=jm_cp,
                store=store,
                ask_backend=lambda _r: {
                    "text": "VERDICT: PASS\nNo spec defects in partial diff.",
                    "usage": {"prompt_token_count": 100, "candidates_token_count": 10, "thoughts_token_count": 10, "total_token_count": 120},
                },
            )
            resp_cp = srv_cp.handle_Message(
                {
                    "jsonrpc": "2.0",
                    "id": 100,
                    "method": "tools/call",
                    "params": {
                        "name": "delegate_to_gemini",
                        "arguments": {
                            "prompt": "Finish greet in app.py",
                            "model": "gemini-3.8-flash",
                            "project": "test-proj",
                            "qa_review_model": "gemini-4-argon",
                            "auto_apply": True,
                        },
                    },
                }
            )
            payload_cp = json.loads(resp_cp["result"]["content"][0]["text"])
            self.assertEqual(payload_cp["status"], "ok")
            self.assertTrue(payload_cp["applied"])
            self.assertTrue(payload_cp["qa_review"]["resumed"])
            self.assertEqual(len(checkpoint_calls), 2)
            self.assertEqual((repo / "app.py").read_text(encoding="utf-8"), "def greet():\n    return 'completed!'\n")
            jm_cp.shutdown()
            jm.shutdown()

    def test_qa_review_diff_is_capped_at_25k_tokens(self) -> None:
        """Reviewer context is truncated to QA_REVIEW_MAX_TOKENS with HIGH thinking."""
        from server.jobs import QA_REVIEW_MAX_TOKENS
        from server.regions import estimate_tokens

        self.assertEqual(QA_REVIEW_MAX_TOKENS, 25_000)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _init_git_repo(repo)
            store = ReceiptStore(Path(tmp) / "data", workspace_root=repo)
            big_line = "x = '" + ("abcdefgh " * 20) + "'\n"

            def fake_worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                (wt / "big.py").write_text(big_line * 6000, encoding="utf-8")
                return {"text": "done", "usage": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15},
                        "tool_calls": [], "thoughts": []}

            ask_calls: list[dict] = []

            def fake_ask(req: dict) -> dict:
                ask_calls.append(req)
                return {"text": "VERDICT: PASS", "usage": {"prompt_token_count": 1, "candidates_token_count": 1, "total_token_count": 2}}

            jm = JobManager(workspace_root=repo, store=store, worker_backend=fake_worker)
            srv = MCPServer(job_manager=jm, store=store, ask_backend=fake_ask)
            resp = srv.handle_Message({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {
                "name": "delegate_to_gemini",
                "arguments": {"prompt": "Add big.py", "model": "gemini-3.8-flash", "project": "test-proj",
                              "qa_review_model": "gemini-4-argon", "auto_apply": False}}})
            payload = json.loads(resp["result"]["content"][0]["text"])
            self.assertEqual(len(ask_calls), 1)
            self.assertEqual(ask_calls[0].get("thinking_level"), "HIGH")
            sent = str(ask_calls[0].get("prompt") or "")
            self.assertIn("(truncated)", sent)
            self.assertLessEqual(estimate_tokens(sent), QA_REVIEW_MAX_TOKENS + 200)
            self.assertTrue(payload["qa_review"]["passed"])
            jm.shutdown()


class TestEscalationRevertAndTrueCost(unittest.TestCase):
    """Dispatched vs true total cost, escalation after repeated failures, and revert from provenance."""

    def _repo(self, tmp: str) -> tuple[Path, ReceiptStore]:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        _init_git_repo(repo)
        return repo, ReceiptStore(Path(tmp) / "data", workspace_root=repo)

    @staticmethod
    def _ok_usage() -> dict:
        return {"prompt_token_count": 100, "candidates_token_count": 20, "total_token_count": 120}

    def test_true_cost_report_splits_dispatched_and_lead(self) -> None:
        from server.true_cost import collect_lead_usage, project_hash, true_total_report

        self.assertEqual(project_hash("/Users/you/repro-v0.6.0"), "-Users-you-repro-v0-6-0")
        with tempfile.TemporaryDirectory() as tmp:
            repo, store = self._repo(tmp)
            now = time.time()
            store.write_receipt(
                "job-1-aaaaaaaa",
                {"kind": "delegate", "status": "ok", "model": "gemini-3.8-flash",
                 "usage": {"prompt_token_count": 1_000_000, "candidates_token_count": 0, "total_token_count": 1_000_000},
                 "started_at_epoch": now - 60, "prompts": ["x"]},
                workspace_root=str(repo),
            )
            dispatched = float(store.read_receipt("job-1-aaaaaaaa")["cost"]["total_cost_cents"])
            self.assertGreater(dispatched, 0.0)

            tdir = Path(tmp) / "transcripts"
            tdir.mkdir()

            def iso(t: float) -> str:
                return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".000Z"

            lines = [
                # Opus message written twice (two content blocks, same id): counted once.
                {"type": "assistant", "timestamp": iso(now - 30), "message": {"id": "m1", "model": "claude-opus-5-5",
                 "usage": {"input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000, "output_tokens": 100_000,
                           "cache_creation_input_tokens": 300_000,
                           "cache_creation": {"ephemeral_5m_input_tokens": 100_000, "ephemeral_1h_input_tokens": 200_000}}}},
                {"type": "assistant", "timestamp": iso(now - 30), "message": {"id": "m1", "model": "claude-opus-5-5",
                 "usage": {"input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000, "output_tokens": 100_000}}},
                {"type": "assistant", "timestamp": iso(now - 20), "message": {"id": "m2", "model": "claude-haiku-4",
                 "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}},
                {"type": "assistant", "timestamp": iso(now - 20), "message": {"id": "m3", "model": "mystery-model",
                 "usage": {"input_tokens": 5, "output_tokens": 5}}},
                # Outside the window (before the first receipt): ignored.
                {"type": "assistant", "timestamp": iso(now - 3600), "message": {"id": "m0", "model": "claude-opus-5-5",
                 "usage": {"input_tokens": 9_000_000, "output_tokens": 0}}},
                {"type": "user", "timestamp": iso(now - 25), "message": {"content": "hi"}},
            ]
            (tdir / "session.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

            lead = collect_lead_usage(repo, since_epoch=now - 60, transcripts_dir=tdir)
            # Opus: 1M*5 + 1M*0.5 + 100k*6.25 + 200k*10 + 100k*25 = 5 + 0.5 + 0.625 + 2 + 2.5 = 10.625 USD
            self.assertAlmostEqual(lead["per_model"]["claude-opus-5-5"]["cost_cents"], 1062.5, places=3)
            self.assertEqual(lead["per_model"]["claude-opus-5-5"]["messages"], 1)
            self.assertAlmostEqual(lead["per_model"]["claude-haiku-4"]["cost_cents"], 100.0, places=3)
            self.assertFalse(lead["per_model"]["mystery-model"]["priced"])
            self.assertFalse(lead["all_lead_pricing_known"])
            self.assertAlmostEqual(lead["lead_cost_cents"], 1162.5, places=3)

            # Default window: whole sessions overlapping the jobs, so session.jsonl's early 9M-token
            # planning message (before the first receipt) counts; another session that ended before
            # the first job does not.
            old_session = [{"type": "assistant", "timestamp": iso(now - 7200), "message": {"id": "z1",
                            "model": "claude-opus-5-5", "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}}]
            (tdir / "old.jsonl").write_text(json.dumps(old_session[0]) + "\n", encoding="utf-8")
            rep = true_total_report(store, repo, transcripts_dir=tdir)
            self.assertAlmostEqual(rep["dispatched"]["cost_cents"], dispatched, places=3)
            self.assertAlmostEqual(rep["lead"]["lead_cost_cents"], 1162.5 + 4500.0, places=3)
            self.assertEqual(rep["lead"]["transcript_files"], 1)
            self.assertAlmostEqual(rep["true_total_cost_cents"], dispatched + 5662.5, places=3)
            rep_explicit = true_total_report(store, repo, transcripts_dir=tdir, since_epoch=now - 60)
            self.assertAlmostEqual(rep_explicit["true_total_cost_cents"], dispatched + 1162.5, places=3)
            self.assertGreater(rep["lead_share_of_true_total"], 0.5)

            jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda _r: {"text": "ok"})
            srv = MCPServer(job_manager=jm, store=store)
            names = [t["name"] for t in srv.handle_Message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]]
            self.assertIn("cost_report", names)
            self.assertIn("delegate_revert", names)
            resp = srv.handle_Message({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "cost_report", "arguments": {"transcripts_dir": str(tdir)}}})
            payload = json.loads(resp["result"]["content"][0]["text"])
            self.assertAlmostEqual(payload["true_total_cost_cents"], dispatched + 5662.5, places=3)
            jm.shutdown()

    def test_escalates_to_lead_after_two_failed_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, store = self._repo(tmp)
            calls: list[dict] = []

            def worker(req: dict) -> dict:
                calls.append(req)
                wt = Path(req["workspace_dir"])
                (wt / "app.py").write_text(f"def greet():\n    return 'try {len(calls)}'\n", encoding="utf-8")
                return {"text": "done", "usage": self._ok_usage(), "tool_calls": [], "thoughts": []}

            jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
            failing = ["grep -q NEVER_PRESENT_TOKEN app.py"]
            d1 = jm.delegate_start(prompt="Make greet say hi", model="gemini-3.8-flash", project="p",
                                   acceptance=failing, wait=True)
            self.assertIs(d1["acceptance_passed"], False)
            self.assertEqual(d1.get("failed_attempts"), 1)
            self.assertNotIn("escalation", d1)

            d2 = jm.delegate_continue(job_id=d1["job_id"], follow_up_prompt="Fix it", project="p")
            self.assertEqual(d2.get("failed_attempts"), 2)
            self.assertTrue(d2["escalation"]["escalate_to_lead"])
            self.assertEqual(len(calls), 2)

            d3 = jm.delegate_continue(job_id=d1["job_id"], follow_up_prompt="Try again", project="p")
            self.assertEqual(d3["status"], "escalated")
            self.assertEqual(len(calls), 2, "an escalated job must not spend another Gemini attempt")

            d4 = jm.delegate_continue(job_id=d1["job_id"], follow_up_prompt="Try again", project="p", force_retry=True)
            self.assertEqual(len(calls), 3)
            self.assertEqual(d4.get("failed_attempts"), 3)

            # A passing turn resets the streak.
            d5 = jm.delegate_continue(job_id=d1["job_id"], follow_up_prompt="Finish", project="p",
                                      acceptance=["grep -q greet app.py"], force_retry=True)
            self.assertIs(d5["acceptance_passed"], True)
            self.assertNotIn("failed_attempts", d5)
            self.assertNotIn("escalation", d5)
            jm.shutdown()

    def test_revert_restores_pre_apply_state_and_refuses_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, store = self._repo(tmp)
            original = (repo / "app.py").read_bytes()

            def worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                (wt / "app.py").write_text("def greet():\n    return 'hi!'\n", encoding="utf-8")
                (wt / "new_mod.py").write_text("X = 1\n", encoding="utf-8")
                return {"text": "done", "usage": self._ok_usage(), "tool_calls": [], "thoughts": []}

            jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
            srv = MCPServer(job_manager=jm, store=store)
            d = jm.delegate_start(prompt="Change greet and add new_mod", model="gemini-3.8-flash", project="p", wait=True)
            applied = jm.delegate_apply(d["job_id"])
            self.assertIn("revert_hint", applied)
            self.assertTrue((repo / "new_mod.py").exists())
            rec = store.read_receipt(d["job_id"])
            self.assertEqual(rec["pre_apply"]["new_mod.py"]["state"], "absent")
            self.assertEqual(rec["pre_apply"]["app.py"]["state"], "file")
            self.assertTrue(store.verify_receipt_signature(rec))

            dry = jm.delegate_revert(d["job_id"], dry_run=True)
            self.assertEqual(sorted(x["file"] for x in dry["would_restore"]), ["app.py", "new_mod.py"])
            self.assertTrue((repo / "new_mod.py").exists(), "dry_run must not write")

            # A later edit makes new_mod.py dirty: revert refuses and writes nothing.
            (repo / "new_mod.py").write_text("X = 2  # human edit\n", encoding="utf-8")
            refused = jm.delegate_revert(d["job_id"])
            self.assertTrue(refused["refused"])
            self.assertEqual([x["file"] for x in refused["dirty_files"]], ["new_mod.py"])
            self.assertNotEqual((repo / "app.py").read_bytes(), original)

            # skip_dirty reverts only the clean file, through the MCP tool.
            resp = srv.handle_Message({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "delegate_revert", "arguments": {"job_id": d["job_id"], "skip_dirty": True}}})
            out = json.loads(resp["result"]["content"][0]["text"])
            self.assertTrue(out["reverted"])
            self.assertEqual(out["reverted_files"], ["app.py"])
            self.assertEqual((repo / "app.py").read_bytes(), original)
            self.assertEqual((repo / "new_mod.py").read_text(encoding="utf-8"), "X = 2  # human edit\n")
            again = jm.delegate_revert(d["job_id"])
            self.assertTrue(again["already_reverted"])
            self.assertTrue(store.verify_receipt_signature(store.read_receipt(d["job_id"])))
            jm.shutdown()

    def test_full_revert_deletes_created_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, store = self._repo(tmp)
            original = (repo / "app.py").read_bytes()

            def worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                (wt / "app.py").write_text("def greet():\n    return 'yo'\n", encoding="utf-8")
                (wt / "pkg").mkdir(exist_ok=True)
                (wt / "pkg" / "extra.py").write_text("Y = 1\n", encoding="utf-8")
                return {"text": "done", "usage": self._ok_usage(), "tool_calls": [], "thoughts": []}

            jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
            d = jm.delegate_start(prompt="edit", model="gemini-3.8-flash", project="p", wait=True)
            jm.delegate_apply(d["job_id"])
            out = jm.delegate_revert(d["job_id"])
            self.assertEqual(sorted(out["reverted_files"]), ["app.py", "pkg/extra.py"])
            self.assertFalse((repo / "pkg" / "extra.py").exists())
            self.assertEqual((repo / "app.py").read_bytes(), original)
            jm.shutdown()

    def test_revert_in_place_job_restores_from_git_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, store = self._repo(tmp)
            original = (repo / "app.py").read_bytes()

            def worker(req: dict) -> dict:
                wt = Path(req["workspace_dir"])
                (wt / "app.py").write_text("def greet():\n    return 'inplace'\n", encoding="utf-8")
                (wt / "added.py").write_text("Z = 3\n", encoding="utf-8")
                return {"text": "done", "usage": self._ok_usage(), "tool_calls": [], "thoughts": []}

            jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
            d = jm.delegate_start(prompt="edit in place", model="gemini-3.8-flash", project="p", wait=True, in_place=True)
            self.assertTrue(store.read_receipt(d["job_id"]).get("in_place"))
            out = jm.delegate_revert(d["job_id"])
            self.assertTrue(out["reverted"], out)
            self.assertEqual(sorted(out["reverted_files"]), ["added.py", "app.py"])
            self.assertEqual((repo / "app.py").read_bytes(), original)
            self.assertFalse((repo / "added.py").exists())
            jm.shutdown()

    def test_scout_repo_map_low_thinking_and_straitjacket_read_only(self) -> None:
        from server.agy_runner import build_straitjacket_tools, is_straitjacket_available, resolve_thinking_level
        from server.model_routes import resolve_batch_settings
        from server.regions import estimate_tokens
        from server.repo_map import build_repo_map
        from server.verify import is_behavioral_check

        self.assertEqual(resolve_thinking_level(None, model="gemini-3.8-flash", read_only=False), "LOW")
        self.assertEqual(resolve_thinking_level(None, model="gemini-3.8-flash", read_only=True), "HIGH")
        self.assertEqual(resolve_thinking_level(None, model="gemini-4-argon", read_only=False), "LOW")
        self.assertEqual(resolve_thinking_level(None, model="gemini-4-argon", read_only=True), "HIGH")
        self.assertEqual(resolve_batch_settings({"model": "mechanical"}).model, "gemini-3.8-flash")
        self.assertEqual(resolve_batch_settings({"read_budget": 6}).writer, "agent")
        self.assertFalse(is_behavioral_check("python3 -m py_compile pkg/a.py"))
        self.assertTrue(is_behavioral_check("python3 -m pytest -q tests/test_a.py"))

        with tempfile.TemporaryDirectory() as tmp:
            repo, store = self._repo(tmp)
            rmap = build_repo_map(repo, cache_dir=store.data_dir)
            self.assertIn("`app.py`", rmap)
            if is_straitjacket_available():
                sj_tools, replaced = build_straitjacket_tools(repo, profile="read-only")
                tool_names = [t.__name__ for t in sj_tools]
                self.assertIn("view_file", tool_names)
                self.assertNotIn("run_command", tool_names)
                self.assertEqual(replaced, ["VIEW_FILE", "SEARCH_DIR", "LIST_DIR", "FIND_FILE"])

            jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda _r: {
                "text": "word " * 1200, "usage": self._ok_usage(), "tool_calls": [], "thoughts": []
            })
            d = jm.delegate_start(prompt="edit", model="gemini-3.8-flash", project="p", wait=True)
            self.assertEqual(d["thinking_level"], "LOW")
            self.assertTrue(d["summary_truncated"])
            self.assertLessEqual(estimate_tokens(d["summary"]), 300)
            jm.shutdown()

    def test_read_only_detection_uses_sdk_tool_names_and_budget_rules(self) -> None:
        from server.agy_runner import (
            build_straitjacket_tools,
            build_worker_system_prompt,
            is_read_only_tool_set,
            is_straitjacket_available,
            resolve_thinking_level,
        )
        from server.model_routes import resolve_batch_settings
        from server.repo_map import build_repo_map, repo_map_cache_key
        from server.verify import is_behavioral_check

        # google-antigravity names its mutating tools CREATE_FILE, EDIT_FILE, and RUN_COMMAND.
        self.assertTrue(is_read_only_tool_set(None, ["CREATE_FILE", "EDIT_FILE", "RUN_COMMAND"]))
        self.assertFalse(is_read_only_tool_set(None, ["CREATE_FILE", "EDIT_FILE"]))
        self.assertFalse(is_read_only_tool_set(None, ["EDIT_FILE"]))
        self.assertTrue(is_read_only_tool_set(["VIEW_FILE", "SEARCH_DIR"], None))
        self.assertFalse(is_read_only_tool_set(["VIEW_FILE", "CREATE_FILE"], None))
        self.assertEqual(
            resolve_thinking_level(
                None,
                model="gemini-3.8-flash",
                read_only=is_read_only_tool_set(None, ["CREATE_FILE", "EDIT_FILE", "RUN_COMMAND"]),
            ),
            "HIGH",
        )
        self.assertEqual(resolve_thinking_level("MEDIUM", model="gemini-3.8-flash"), "MEDIUM")
        with self.assertRaises(ValueError):
            resolve_batch_settings({"read_budget": 6, "writer": "genai"})

        # Behavioral check classification must reject non-test commands and collection-only flags.
        self.assertFalse(is_behavioral_check("true"))
        self.assertFalse(is_behavioral_check("echo ok"))
        self.assertFalse(is_behavioral_check("pytest --collect-only"))
        self.assertFalse(is_behavioral_check("pytest --co"))
        self.assertFalse(is_behavioral_check("cargo test --no-run"))
        self.assertFalse(is_behavioral_check("go test -c ./..."))
        self.assertTrue(is_behavioral_check("pytest -q"))

        sys_ro = build_worker_system_prompt("/tmp/ws", profile="read-only", has_report_to_lead=True)
        self.assertIn("/tmp/ws", sys_ro)
        self.assertIn("ctx_query", sys_ro)
        self.assertIn("report_to_lead", sys_ro)

        with tempfile.TemporaryDirectory() as tmp:
            repo, store = self._repo(tmp)
            key1 = repo_map_cache_key(repo)
            os.utime(repo / "app.py", None)  # touching a file must not bust the cache
            self.assertEqual(repo_map_cache_key(repo), key1)
            (repo / "app.py").write_text("def greet(token='sk-secret-123'):\n    return 'changed'\n", encoding="utf-8")
            key2 = repo_map_cache_key(repo)
            self.assertNotEqual(key2, key1)  # 1st edit busts cache
            (repo / "app.py").write_text("def greet(token='sk-secret-456'):\n    return 'changed-v2'\n", encoding="utf-8")
            key3 = repo_map_cache_key(repo)
            self.assertNotEqual(key3, key2)  # 2nd edit to already-dirty file also busts cache
            (repo / "untracked_mod.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            rmap = build_repo_map(repo, cache_dir=store.data_dir)
            self.assertIn("`untracked_mod.py`", rmap)
            self.assertNotIn("sk-secret", rmap)
            self.assertEqual(len(list(Path(store.data_dir).glob("repo_map_*.txt"))), 1)

            if is_straitjacket_available():
                (repo / ".mcp.json").write_text('{"secret": "top-secret-mcp"}\n', encoding="utf-8")
                ro_tools, _ = build_straitjacket_tools(repo, profile="read-only")
                ro_map = {t.__name__: t for t in ro_tools}
                ws_esc = asyncio.run(ro_map["ctx_get"]("ws:host/repo:secret.txt"))
                self.assertIn("PermissionDenied", ws_esc)
                sec_read = asyncio.run(ro_map["view_file"](".mcp.json"))
                self.assertIn("PermissionDenied", sec_read)
                self.assertNotIn("top-secret-mcp", sec_read)

            # Without git, node_modules is never walked.
            plain = Path(tmp) / "plain"
            (plain / "node_modules" / "dep").mkdir(parents=True)
            (plain / "node_modules" / "dep" / "index.js").write_text("x", encoding="utf-8")
            (plain / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
            rmap = build_repo_map(plain)
            self.assertIn("`main.py`", rmap)
            self.assertNotIn("node_modules", rmap)

    def test_line_numbered_repo_map_focused_outline_and_diff_preview(self) -> None:
        from server.repo_map import build_focused_file_outline, build_repo_map

        with tempfile.TemporaryDirectory() as tmp:
            repo, store = self._repo(tmp)
            # Create a file with 7 private helpers at top followed by a public class
            helper_lines = "\n".join(f"def _priv_{i}(x: int) -> int:\n    return x + {i}\n" for i in range(7))
            class_body = (
                "\nclass CoreEngine:\n"
                "    def __init__(self, cfg: str) -> None:\n"
                "        self.cfg = cfg\n\n"
                "    def execute(self, item: str) -> str:\n"
                "        return item.upper()\n"
            )
            (repo / "engine.py").write_text(helper_lines + class_body, encoding="utf-8")
            rmap = build_repo_map(repo, cache_dir=store.data_dir, refresh=True)
            # Class definition must appear first in repo_map with line span despite 7 _priv_* helpers above it
            self.assertIn("class CoreEngine (L", rmap)
            outline = build_focused_file_outline(repo, ["engine.py"])
            self.assertIn("`engine.py`", outline)
            self.assertIn("execute@L", outline)

            def worker_edit(req: dict[str, Any]) -> dict[str, Any]:
                wt = Path(req["workspace_dir"])
                (wt / "app.py").write_text("def greet(name: str = 'world') -> str:\n    return f'hi {name}'\n", encoding="utf-8")
                return {
                    "text": "updated greet",
                    "usage": self._ok_usage(),
                    "tool_calls": [{"name": "edit_file"}],
                    "thoughts": [],
                }

            jm = JobManager(workspace_root=repo, store=store, worker_backend=worker_edit)
            d = jm.delegate_start(
                prompt="update greet in app.py",
                model="gemini-3.8-flash",
                project="p",
                allowed_paths=["app.py"],
                acceptance=["python3 -m py_compile app.py"],
                wait=True,
            )
            self.assertEqual(d["status"], "ok")
            self.assertTrue(d["acceptance_passed"])
            self.assertIn("diff_preview", d)
            self.assertIn("+    return f'hi {name}'", d["diff_preview"])
            self.assertTrue(d["diff_preview_complete"])
            self.assertIn("delegate_apply", d.get("next_action", ""))
            jm.shutdown()


if __name__ == "__main__":
    unittest.main()
