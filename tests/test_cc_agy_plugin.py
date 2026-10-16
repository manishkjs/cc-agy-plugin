"""Comprehensive unit and adversarial regression tests for cc-agy-plugin."""

from __future__ import annotations

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

        unknown_cost = estimate_cost_usd("custom-experimental-model-v9", usage)
        self.assertFalse(unknown_cost["pricing_known"])

        # >200k long-context multiplier (2x) on Pro models
        long_usage = {"prompt_token_count": 250_000, "candidates_token_count": 10_000}
        pro_cost = estimate_cost_usd("gemini-3.1-pro-preview", long_usage)
        self.assertAlmostEqual(pro_cost["input_cost_usd"], 250_000 / 1e6 * 2.0 * 2.0, places=6)


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
            self.assertEqual(calls[0]["prompts"], ["Turn 1 prompt"])
            saved_r1 = store.read_receipt(d1["job_id"])
            self.assertEqual(saved_r1["vertex_project"], "<paying-project>")

            d2 = jm.delegate_continue(d1["job_id"], "Turn 2 follow-up", wait=True)
            self.assertEqual(d2["status"], "ok")
            self.assertEqual(len(calls), 2)
            # Only the follow-up prompt was sent on Turn 2, with the real project and enabled_tools preserved!
            self.assertEqual(calls[1]["prompts"], ["Turn 2 follow-up"])
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


if __name__ == "__main__":
    unittest.main()
