"""Productization & language/repository generality tests for cc-agy-plugin."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.pre_tool_guard import detect_bash_workspace_mutation
from server.agy_runner import resolve_thinking_level
from server.doctor import run_doctor
from server.genai_ask import ask_gemini
from server.harness_validator import validate_plugin_tree
from server.jobs import (
    JobManager,
    _parse_test_counts,
    detect_project_test_runner,
    discover_related_test_commands,
    validate_acceptance,
)
from server.mcp_stdio import MCPServer
from server.receipts import ReceiptStore
from server.regions import (
    resolve_cost_profile,
    resolve_default_ask_model,
    resolve_default_delegate_model,
    sanitize_placeholder_env,
)
from server.repo_map import build_focused_file_outline, build_repo_map

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True, capture_output=True)


def _git_commit_all(repo: Path, msg: str = "init") -> None:
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=str(repo), check=True, capture_output=True)


def test_multilang_runner_detection_and_related_tests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_bin = tmp_path / "fake_toolchain_bin"
    fake_bin.mkdir()
    for tool in ("go", "cargo", "java", "mvn", "make"):
        stub = fake_bin / tool
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
    orig_path = __import__("os").environ.get("PATH", "")
    monkeypatch.setenv("PATH", f"{fake_bin}:{orig_path}")

    # 1. Generic Python repo with custom `bin/test` wrapper (not hardcoded to sympy/)
    py_custom = tmp_path / "custom_math_lib"
    (py_custom / "bin").mkdir(parents=True)
    (py_custom / "mymath" / "utilities").mkdir(parents=True)
    (py_custom / "mymath" / "tests").mkdir(parents=True)
    (py_custom / "bin" / "test").write_text("#!/usr/bin/env python3\nimport sys\n", encoding="utf-8")
    (py_custom / "mymath" / "utilities" / "pytest.py").write_text("# custom runner\n", encoding="utf-8")
    (py_custom / "mymath" / "solver.py").write_text("def solve(): return 1\n", encoding="utf-8")
    (py_custom / "mymath" / "tests" / "test_solver.py").write_text("def test_solve(): assert True\n", encoding="utf-8")
    assert detect_project_test_runner(py_custom)["primary_command"] == "python3 bin/test"
    rel_py = discover_related_test_commands(py_custom, ["mymath/solver.py"], [])
    assert rel_py == ["python3 bin/test mymath/tests/test_solver.py"]

    # 2. Node/TypeScript repo with Vitest + pnpm lockfile
    ts_repo = tmp_path / "ts_app"
    (ts_repo / "src").mkdir(parents=True)
    (ts_repo / "node_modules" / ".bin").mkdir(parents=True)
    (ts_repo / "node_modules" / ".bin" / "vitest").write_text("#!/bin/sh\n", encoding="utf-8")
    (ts_repo / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
    (ts_repo / "package.json").write_text(
        json.dumps({"name": "ts-app", "devDependencies": {"vitest": "^2.0.0"}, "scripts": {"test": "vitest"}}),
        encoding="utf-8",
    )
    (ts_repo / "src" / "cart.ts").write_text("export const total = () => 0;\n", encoding="utf-8")
    (ts_repo / "src" / "cart.test.ts").write_text("test('total', () => {});\n", encoding="utf-8")
    assert detect_project_test_runner(ts_repo)["primary_command"] == "./node_modules/.bin/vitest run"
    assert discover_related_test_commands(ts_repo, ["src/cart.ts"], []) == [
        "./node_modules/.bin/vitest run src/cart.test.ts"
    ]

    # 3. Node repo with Jest
    jest_repo = tmp_path / "jest_app"
    (jest_repo / "src" / "__tests__").mkdir(parents=True)
    (jest_repo / "node_modules" / ".bin").mkdir(parents=True)
    (jest_repo / "node_modules" / ".bin" / "jest").write_text("#!/bin/sh\n", encoding="utf-8")
    (jest_repo / "package.json").write_text(
        json.dumps({"name": "jest-app", "devDependencies": {"jest": "^29.0.0"}}),
        encoding="utf-8",
    )
    (jest_repo / "src" / "widget.tsx").write_text("export function Widget() { return null; }\n", encoding="utf-8")
    (jest_repo / "src" / "__tests__" / "widget.spec.tsx").write_text("it('renders', () => {});\n", encoding="utf-8")
    assert detect_project_test_runner(jest_repo)["primary_command"] == "./node_modules/.bin/jest --watchAll=false"
    assert discover_related_test_commands(jest_repo, ["src/widget.tsx"], []) == [
        "./node_modules/.bin/jest --watchAll=false src/__tests__/widget.spec.tsx"
    ]

    # 4. Go module
    go_repo = tmp_path / "go_svc"
    (go_repo / "internal" / "auth").mkdir(parents=True)
    (go_repo / "go.mod").write_text("module example.com/svc\ngo 1.22\n", encoding="utf-8")
    (go_repo / "internal" / "auth" / "token.go").write_text("package auth\n", encoding="utf-8")
    (go_repo / "internal" / "auth" / "token_test.go").write_text("package auth\n", encoding="utf-8")
    assert detect_project_test_runner(go_repo)["primary_command"] == "go test ./..."
    assert discover_related_test_commands(go_repo, ["internal/auth/token.go"], []) == [
        "go test ./internal/auth"
    ]

    # 5. Rust crate
    rs_repo = tmp_path / "rs_crate"
    (rs_repo / "src").mkdir(parents=True)
    (rs_repo / "Cargo.toml").write_text('[package]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8")
    (rs_repo / "src" / "parser.rs").write_text("pub fn parse() {}\n", encoding="utf-8")
    assert detect_project_test_runner(rs_repo)["primary_command"] == "cargo test --quiet"
    assert discover_related_test_commands(rs_repo, ["src/parser.rs"], []) == ["cargo test --quiet"]

    # 6. Gradle wrapper & Makefile
    gradle_repo = tmp_path / "gradle_svc"
    (gradle_repo / "src" / "test" / "java" / "com" / "acme").mkdir(parents=True)
    (gradle_repo / "src" / "main" / "java" / "com" / "acme").mkdir(parents=True)
    gradlew_p = gradle_repo / "gradlew"
    gradlew_p.write_text("#!/bin/sh\n", encoding="utf-8")
    gradlew_p.chmod(0o755)
    (gradle_repo / "build.gradle.kts").write_text("plugins { java }\n", encoding="utf-8")
    (gradle_repo / "src" / "main" / "java" / "com" / "acme" / "OrderService.java").write_text(
        "package com.acme; public class OrderService {}\n", encoding="utf-8"
    )
    (gradle_repo / "src" / "test" / "java" / "com" / "acme" / "OrderServiceTest.java").write_text(
        "package com.acme; public class OrderServiceTest {}\n", encoding="utf-8"
    )
    assert detect_project_test_runner(gradle_repo)["primary_command"] == "./gradlew test"
    assert discover_related_test_commands(
        gradle_repo, ["src/main/java/com/acme/OrderService.java"], []
    ) == ["./gradlew test --tests '*OrderServiceTest'"]

    make_repo = tmp_path / "c_lib"
    make_repo.mkdir()
    (make_repo / "Makefile").write_text("all:\n\tcc main.c\ntest:\n\t./run_tests\n", encoding="utf-8")
    assert detect_project_test_runner(make_repo)["primary_command"] == "make test"


def test_multilang_test_count_parsers_and_acceptance_validation(tmp_path: Path) -> None:
    # ANSI colored Vitest / Jest
    assert _parse_test_counts("\x1b[32m✓\x1b[39m src/cart.test.ts\n Tests  14 passed (14)\n") == {
        "passed": 14,
        "failed": 0,
        "errors": 0,
    }
    assert _parse_test_counts("Tests:       2 failed, 9 passed, 11 total\n") == {
        "passed": 9,
        "failed": 2,
        "errors": 0,
    }
    # Cargo test across unit + doc suites
    assert _parse_test_counts(
        "test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
        "test result: FAILED. 3 passed; 1 failed; 0 ignored\n"
    ) == {"passed": 11, "failed": 1, "errors": 0}
    # Go test case-level & package-level
    assert _parse_test_counts("--- PASS: TestLogin (0.00s)\n--- FAIL: TestRefresh (0.01s)\nFAIL\n") == {
        "passed": 1,
        "failed": 1,
        "errors": 0,
    }
    assert _parse_test_counts("ok  \texample.com/a\t0.014s\nok  \texample.com/b\t(cached)\n") == {
        "passed": 2,
        "failed": 0,
        "errors": 0,
    }
    # Node --test
    assert _parse_test_counts("ℹ tests 6\nℹ pass 5\nℹ fail 1\n") == {
        "passed": 5,
        "failed": 1,
        "errors": 0,
    }
    # Maven Surefire
    assert _parse_test_counts("Tests run: 12, Failures: 1, Errors: 2, Skipped: 1\n") == {
        "passed": 8,
        "failed": 1,
        "errors": 2,
    }

    # Read-only multi-language acceptance commands allowed
    ws = tmp_path / "ws"
    ws.mkdir()
    allowed_cmds = [
        "npx vitest run src/cart.test.ts",
        "cargo test --quiet",
        "go test ./...",
        "bun test",
        "./gradlew test",
    ]
    assert validate_acceptance(allowed_cmds, ws) == allowed_cmds

    # Mutating commands blocked by detect_bash_workspace_mutation & validate_acceptance
    for mutating in (
        "cargo fmt",
        "go fmt ./...",
        "go test -o ./bin/out ./...",
        "npx -y create-react-app foo",
        "npx jest -u",
        "npx eslint --fix src/",
        "bun install",
        "mvn clean package",
    ):
        assert detect_bash_workspace_mutation(mutating, ws)["blocked"] is True, f"Expected block for {mutating}"
        with pytest.raises(ValueError):
            validate_acceptance([mutating], ws)


def test_cost_profiles_and_shipped_default_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in (
        "CC_AGY_COST_PROFILE",
        "CLAUDE_PLUGIN_OPTION_COST_PROFILE",
        "CC_AGY_USERCFG_COST_PROFILE",
        "CC_AGY_DEFAULT_DELEGATE_MODEL",
        "CLAUDE_PLUGIN_OPTION_DEFAULT_DELEGATE_MODEL",
        "CC_AGY_USERCFG_DEFAULT_DELEGATE_MODEL",
        "CC_AGY_DEFAULT_ASK_MODEL",
        "CLAUDE_PLUGIN_OPTION_DEFAULT_ASK_MODEL",
        "CC_AGY_USERCFG_DEFAULT_ASK_MODEL",
        "CC_AGY_DELEGATE_THINKING_LEVEL",
        "CC_AGY_FORCE_WORKER_THINKING_LEVEL",
    ):
        monkeypatch.delenv(key, raising=False)

    # Default profile is balanced -> gemini-3.8-flash for both worker and ask_gemini
    assert resolve_cost_profile({}) == "balanced"
    assert resolve_default_delegate_model(None, env={}) == "gemini-3.8-flash"
    assert resolve_default_ask_model(None, env={}) == "gemini-3.8-flash"

    # Unexpanded ${user_config.*} placeholders are sanitized and fall back to cost_profile
    placeholder_env = {
        "CC_AGY_USERCFG_COST_PROFILE": "quality",
        "CC_AGY_USERCFG_DEFAULT_DELEGATE_MODEL": "${user_config.default_delegate_model}",
        "CC_AGY_USERCFG_DEFAULT_ASK_MODEL": "${user_config.default_ask_model}",
    }
    assert resolve_cost_profile(placeholder_env) == "quality"
    assert resolve_default_delegate_model(None, env=placeholder_env) == "gemini-4-argon"
    assert resolve_default_ask_model(None, env=placeholder_env) == "gemini-4-argon"

    # Explicit default_delegate_model still wins over cost_profile
    explicit_env = {
        "CC_AGY_USERCFG_COST_PROFILE": "economy",
        "CC_AGY_DEFAULT_DELEGATE_MODEL": "gemini-4-argon",
    }
    assert resolve_default_delegate_model(None, env=explicit_env) == "gemini-4-argon"

    # Flash family environment overrides explicitly supersede hardcoded gemini-3.8-flash prompt requests
    flash_next_env = {
        "CC_AGY_DEFAULT_DELEGATE_MODEL": "gemini-flash-next",
        "CC_AGY_DEFAULT_ASK_MODEL": "gemini-flash-next",
    }
    assert resolve_default_delegate_model("gemini-3.8-flash", env=flash_next_env) == "gemini-flash-next"
    assert resolve_default_ask_model("gemini-3.8-flash", env=flash_next_env) == "gemini-flash-next"
    assert resolve_default_delegate_model("gemini-4-argon", env=flash_next_env) == "gemini-4-argon"

    # sanitize_placeholder_env cleans CC_AGY_USERCFG_COST_PROFILE when unexpanded
    monkeypatch.setenv("CC_AGY_USERCFG_COST_PROFILE", "${user_config.cost_profile}")
    sanitize_placeholder_env()
    assert "CC_AGY_USERCFG_COST_PROFILE" not in __import__("os").environ

    # Thinking level respects cost_profile="quality" on mutating Flash calls
    monkeypatch.setenv("CC_AGY_COST_PROFILE", "quality")
    assert resolve_thinking_level(None, model="gemini-3.8-flash", read_only=False) == "MEDIUM"
    monkeypatch.setenv("CC_AGY_COST_PROFILE", "balanced")
    assert resolve_thinking_level(None, model="gemini-3.8-flash", read_only=False) == "LOW"

    # End-to-end ask_gemini default model check
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git_commit_all(repo)
    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    seen_models: list[str] = []

    def fake_ask(req: dict[str, Any]) -> dict[str, Any]:
        seen_models.append(str(req["model"]))
        return {"text": "looks good", "usage": {"prompt_token_count": 10, "candidates_token_count": 5}}

    res = ask_gemini("Review a.py", project="test-p", workspace_root=repo, store=store, backend=fake_ask)
    assert res["model"] == "gemini-3.8-flash"
    assert seen_models == ["gemini-3.8-flash"]

    # plugin.json must not hardcode default on default_delegate_model / default_ask_model
    manifest = json.loads((PLUGIN_ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    ucfg = manifest["userConfig"]
    assert ucfg["cost_profile"]["default"] == "balanced"
    assert ucfg["cost_profile"]["type"] == "string"
    assert all(p in ucfg["cost_profile"]["description"] for p in ("economy", "balanced", "quality"))
    assert "default" not in ucfg["default_delegate_model"]
    assert "default" not in ucfg["default_ask_model"]


def test_smart_escalation_preserves_flash_for_unrelated_second_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for k in (
        "CC_AGY_COST_PROFILE",
        "CC_AGY_DEFAULT_DELEGATE_MODEL",
        "CC_AGY_ESCALATION_MODEL",
        "CC_AGY_ESCALATE_MODEL_ON_FAILURE",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CC_AGY_AUTO_APPLY_ON_PASS", "1")
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")

    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "auth.py").write_text("def login(): return True\n", encoding="utf-8")
    (repo / "billing.py").write_text("def charge(): return 0\n", encoding="utf-8")
    (repo / "test_auth.py").write_text(
        "import auth\ndef test_login():\n    assert auth.login('u') is True\n",
        encoding="utf-8",
    )
    (repo / "test_billing.py").write_text(
        "import billing\ndef test_charge():\n    assert billing.charge() == 100\n",
        encoding="utf-8",
    )
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    models_used: list[str] = []

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        models_used.append(str(req["model"]))
        wt = Path(req["workspace_dir"])
        prompt_joined = "\n".join(req.get("prompts") or [])
        if "Implement charge amount" in prompt_joined or "billing.py edge case" in prompt_joined:
            (wt / "billing.py").write_text("def charge(): return 100\n", encoding="utf-8")
        else:
            (wt / "auth.py").write_text("def login(user='anon'): return bool(user)\n", encoding="utf-8")
        return {"text": "done", "usage": {"prompt_token_count": 50, "candidates_token_count": 20}}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)

    # Task 1: clean applied job on auth.py -> runs on gemini-3.8-flash
    d1 = jm.delegate_start(
        "Update login in auth.py",
        allowed_paths=["auth.py"],
        acceptance=["pytest -q test_auth.py"],
        project="p",
        wait=True,
    )
    assert d1["applied"] is True
    assert d1["model"] == "gemini-3.8-flash"

    # Task 2: completely unrelated task on billing.py in the same session -> MUST stay on gemini-3.8-flash!
    d2 = jm.delegate_start(
        "Implement charge amount in billing.py",
        allowed_paths=["billing.py"],
        acceptance=["pytest -q test_billing.py"],
        project="p",
        wait=True,
    )
    assert d2["applied"] is True
    assert d2["model"] == "gemini-3.8-flash"
    assert "escalated_from_model" not in d2

    # Task 3: post-apply bugfix on billing.py (overlapping file + fix intent) -> escalates to gemini-4-argon!
    d3 = jm.delegate_start(
        "Fix regression in billing.py edge case",
        allowed_paths=["billing.py"],
        acceptance=["pytest -q test_billing.py"],
        project="p",
        wait=True,
    )
    assert d3["model"] == "gemini-4-argon"
    assert d3.get("escalated_from_model") == "gemini-3.8-flash"

    # Economy profile disables auto-escalation even on overlapping follow-up
    monkeypatch.setenv("CC_AGY_COST_PROFILE", "economy")
    d4 = jm.delegate_start(
        "Fix another billing.py edge case",
        allowed_paths=["billing.py"],
        project="p",
        wait=True,
    )
    assert d4["model"] == "gemini-3.8-flash"
    assert "escalated_from_model" not in d4


def test_disjoint_allowed_paths_skips_dirty_worktree_hijack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "pkg_a").mkdir()
    (repo / "pkg_b").mkdir()
    (repo / "pkg_a" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (repo / "pkg_b" / "b.py").write_text("b = 1\n", encoding="utf-8")
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)

    def checkpoint_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        if "Edit pkg_a" in "\n".join(req.get("prompts") or []):
            (wt / "pkg_a" / "a.py").write_text("a = 2\n", encoding="utf-8")
            return {"text": "paused", "stop_reason": "MAX_TOOL_CALLS_EXCEEDED"}
        (wt / "pkg_b" / "b.py").write_text("b = 2\n", encoding="utf-8")
        return {"text": "done b"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=checkpoint_worker)
    j1 = jm.delegate_start("Edit pkg_a", allowed_paths=["pkg_a/a.py"], project="p", wait=True)
    assert j1["status"] == "checkpoint"

    # Disjoint allowed_paths must start a brand-new job, NOT hijack j1's worktree
    j2 = jm.delegate_start("Edit pkg_b", allowed_paths=["pkg_b/b.py"], project="p", wait=True)
    assert j2["job_id"] != j1["job_id"]
    assert j2["status"] == "ok"


def test_sparse_worktree_monorepo_isolation_apply_and_revert(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CC_AGY_SPARSE_THRESHOLD_FILES", "12")
    repo = tmp_path / "monorepo"
    _init_git_repo(repo)
    (repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (repo / "pkg_a").mkdir()
    (repo / "pkg_a" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg_a" / "target.py").write_text("def calc():\n    return 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_target.py").write_text(
        "from pkg_a.target import calc\ndef test_calc():\n    assert calc() == 42\n",
        encoding="utf-8",
    )
    # Create 25 unrelated files in pkg_b/ and pkg_c/ to exceed threshold=12
    for pkg_name in ("pkg_b", "pkg_c"):
        pdir = repo / pkg_name
        pdir.mkdir()
        for idx in range(15):
            (pdir / f"mod_{idx}.py").write_text(f"VAL = {idx}\n", encoding="utf-8")
    _git_commit_all(repo)

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    observed_wt: dict[str, Any] = {}

    def sparse_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        observed_wt["has_target"] = (wt / "pkg_a" / "target.py").is_file()
        observed_wt["has_test"] = (wt / "tests" / "test_target.py").is_file()
        observed_wt["has_pyproject"] = (wt / "pyproject.toml").is_file()
        observed_wt["has_pkg_b"] = (wt / "pkg_b").exists()
        observed_wt["has_pkg_c"] = (wt / "pkg_c").exists()
        (wt / "pkg_a" / "target.py").write_text("def calc():\n    return 42\n", encoding="utf-8")
        return {"text": "updated calc to 42"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=sparse_worker)
    res = jm.delegate_start(
        "Update calc() in pkg_a/target.py to return 42",
        allowed_paths=["pkg_a/target.py"],
        acceptance=["pytest -q tests/test_target.py"],
        project="p",
        wait=True,
        auto_apply=False,
    )
    assert res["status"] == "ok"
    assert res["acceptance_passed"] is True
    assert res["files_changed"] == ["pkg_a/target.py"]
    assert observed_wt == {
        "has_target": True,
        "has_test": True,
        "has_pyproject": True,
        "has_pkg_b": False,
        "has_pkg_c": False,
    }
    rcpt = store.read_receipt(res["job_id"])
    assert rcpt["sparse_worktree"] is True
    assert rcpt["worktree_file_count"] < 15

    # Apply must update ONLY pkg_a/target.py and leave all 30 omitted files in pkg_b/ and pkg_c/ untouched!
    applied = jm.delegate_apply(res["job_id"])
    assert applied["applied"] is True
    assert "return 42" in (repo / "pkg_a" / "target.py").read_text(encoding="utf-8")
    assert (repo / "pkg_b" / "mod_0.py").read_text(encoding="utf-8") == "VAL = 0\n"
    assert (repo / "pkg_c" / "mod_14.py").read_text(encoding="utf-8") == "VAL = 14\n"

    # Revert must cleanly restore pkg_a/target.py
    reverted = jm.delegate_revert(res["job_id"])
    assert reverted["reverted"] is True
    assert reverted["reverted_files"] == ["pkg_a/target.py"]
    assert "return 1" in (repo / "pkg_a" / "target.py").read_text(encoding="utf-8")


def test_auto_default_acceptance_missing_binary_does_not_block_apply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CC_AGY_AUTO_DEFAULT_ACCEPTANCE", "1")
    repo = tmp_path / "rust_no_cargo_env"
    _init_git_repo(repo)
    (repo / "Cargo.toml").write_text('[package]\nname = "x"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("pub fn v() -> i32 { 1 }\n", encoding="utf-8")
    _git_commit_all(repo)

    # Force PATH inside run_acceptance via a mocked discover_related_test_commands returning a nonexistent binary
    monkeypatch.setattr(
        "server.jobs.discover_related_test_commands",
        lambda *_a, **_kw: ["nonexistent_test_runner_xyz_999 --quiet"],
    )
    store = ReceiptStore(tmp_path / "data", workspace_root=repo)

    def edit_rs(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "src" / "lib.rs").write_text("pub fn v() -> i32 { 2 }\n", encoding="utf-8")
        return {"text": "updated lib.rs"}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=edit_rs)
    res = jm.delegate_start("Update v() to return 2", project="p", wait=True, auto_apply=True)
    assert res["applied"] is True
    rcpt = store.read_receipt(res["job_id"])
    assert rcpt["acceptance_passed"] is True
    assert rcpt["acceptance_results"][0]["runner_incompatible_skipped"] is True


def test_doctor_diagnostics_redacts_secrets_and_works_via_mcp_and_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "go_project"
    _init_git_repo(repo)
    (repo / "go.mod").write_text("module example.com/demo\ngo 1.22\n", encoding="utf-8")
    (repo / "main.go").write_text("package main\nfunc main() {}\n", encoding="utf-8")
    _git_commit_all(repo)
    store = ReceiptStore(tmp_path / "data", workspace_root=repo)

    secret_key = "AIzaSuperSecretNeverLeak999999"
    report = run_doctor(
        workspace_root=repo,
        env={"GEMINI_API_KEY": secret_key, "CC_AGY_COST_PROFILE": "quality"},
        store=store,
    )
    serialized = json.dumps(report)
    assert secret_key not in serialized
    assert report["healthy"] is True
    assert report["cost_profile"] == "quality"
    assert report["default_delegate_model"] == "gemini-4-argon"
    assert report["auth"]["auth_mode"] == "api_key"
    assert report["auth"]["api_key_configured"] is True
    assert report["workspace"]["is_git_repo"] is True
    assert report["workspace"]["tracked_files"] == 2
    assert report["workspace"]["detected_test_runner"]["primary_command"] == "go test ./..."
    assert report["receipt_store"]["writable"] is True

    # Missing auth surfaces error + remediation
    bad_report = run_doctor(workspace_root=repo, env={}, store=store)
    assert bad_report["healthy"] is False
    assert bad_report["status"] == "error"
    assert any("GEMINI_API_KEY" in r for r in bad_report["remediations"])

    # MCP server dispatch for `doctor`
    monkeypatch.setenv("GEMINI_API_KEY", secret_key)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=lambda _r: {"text": "ok"})
    srv = MCPServer(job_manager=jm, store=store)
    mcp_resp = srv.handle_message(
        {"jsonrpc": "2.0", "id": 77, "method": "tools/call", "params": {"name": "doctor", "arguments": {}}}
    )
    assert isinstance(mcp_resp, dict)
    mcp_text = mcp_resp["result"]["content"][0]["text"]
    assert secret_key not in mcp_text
    mcp_payload = json.loads(mcp_text)
    assert mcp_payload["workspace"]["detected_test_runner"]["primary_command"] == "go test ./..."

    # CLI `scripts/session_start.py --doctor`
    cli_env = {
        "PATH": __import__("os").environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "GEMINI_API_KEY": secret_key,
        "CC_AGY_DATA_DIR": str(tmp_path / "cli_data"),
    }
    cli_res = subprocess.run(
        [sys.executable, str(PLUGIN_ROOT / "scripts" / "session_start.py"), "--doctor"],
        cwd=str(repo),
        env=cli_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli_res.returncode == 0
    assert secret_key not in cli_res.stdout
    assert json.loads(cli_res.stdout)["healthy"] is True


def test_repo_map_java_kotlin_signatures_and_plugin_tree_validator(tmp_path: Path) -> None:
    repo = tmp_path / "jvm_repo"
    _init_git_repo(repo)
    (repo / "pom.xml").write_text("<project></project>\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "PaymentProcessor.java").write_text(
        "package com.demo;\n"
        "public class PaymentProcessor {\n"
        "    public boolean captureOrder(String orderId, int cents) {\n"
        "        return true;\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "src" / "Ledger.kt").write_text(
        "class Ledger {\n"
        "    suspend fun recordEntry(id: String): Boolean = true\n"
        "}\n",
        encoding="utf-8",
    )
    _git_commit_all(repo)

    rmap = build_repo_map(repo)
    assert "Detected checks: `mvn -q test`" in rmap
    assert "PaymentProcessor" in rmap
    assert "Ledger" in rmap
    outline = build_focused_file_outline(repo, ["src/PaymentProcessor.java", "src/Ledger.kt"])
    assert "captureOrder" in outline
    assert "recordEntry" in outline

    # Validate the entire shipped plugin tree (budgets, SKILL.md triggers, script refs, hooks.json)
    assert validate_plugin_tree(PLUGIN_ROOT) == []


def test_user_prompt_submit_resets_lead_read_budget_across_interactive_turns(tmp_path: Path) -> None:
    from scripts.user_prompt_submit import record_user_prompt

    repo = tmp_path / "interactive_repo"
    _init_git_repo(repo)
    (repo / "main.py").write_text("x = 1\n", encoding="utf-8")
    _git_commit_all(repo)

    data_dir = tmp_path / "data"
    store = ReceiptStore(data_dir, workspace_root=repo)
    # Pre-prompt reads before Prompt 1 are preserved on the first user prompt of the session
    (store.data_dir / ".lead_read_count").write_text("1", encoding="utf-8")
    (store.data_dir / ".lead_read_files.json").write_text('["pre_prompt.py"]', encoding="utf-8")
    first = record_user_prompt(
        {"cwd": str(repo), "session_id": "sess-1", "prompt": "Initial task on main.py"},
        env={"CC_AGY_DATA_DIR": str(data_dir)},
    )
    assert first is not None and first.is_file()
    assert (store.data_dir / ".lead_read_count").read_text(encoding="utf-8") == "1"

    # Subsequent interactive prompt (Prompt 2+) resets per-turn Lead read counters cleanly
    (store.data_dir / ".lead_read_count").write_text("2", encoding="utf-8")
    (store.data_dir / ".lead_read_files.json").write_text('["old_task.py"]', encoding="utf-8")
    (store.data_dir / ".lead_read_windows.json").write_text('{"old_task.py": [1, 50]}', encoding="utf-8")

    saved = record_user_prompt(
        {"cwd": str(repo), "session_id": "sess-1", "prompt": "Now add metrics to metrics.py"},
        env={"CC_AGY_DATA_DIR": str(data_dir)},
    )
    assert saved is not None and saved.is_file()
    assert (store.data_dir / ".lead_read_count").read_text(encoding="utf-8") == "0"
    assert json.loads((store.data_dir / ".lead_read_files.json").read_text(encoding="utf-8")) == []
    assert json.loads((store.data_dir / ".lead_read_windows.json").read_text(encoding="utf-8")) == {}


def test_opus_round1_regressions_sparse_imports_widen_mvnw_gradle_and_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from server.mcp_stdio import get_tool_definitions

    # 1. `doctor` is exposed in strict/core tool profile
    core_names = {t["name"] for t in get_tool_definitions(env={"CC_AGY_STRICT_MODE": "true"})}
    assert "doctor" in core_names

    # 2. Gradle test count parser + `./mvnw` wrapper + multi-module Gradle / Surefire flags
    assert _parse_test_counts("> Task :test\n5 tests completed, 1 failed, 1 skipped\n") == {
        "passed": 3,
        "failed": 1,
        "errors": 0,
    }
    fake_bin = tmp_path / "jvm_only_bin"
    fake_bin.mkdir()
    java_stub = fake_bin / "java"
    java_stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    java_stub.chmod(0o755)
    orig_path = __import__("os").environ.get("PATH", "")
    monkeypatch.setenv("PATH", str(fake_bin))

    mvnw_repo = tmp_path / "mvnw_svc"
    (mvnw_repo / "src" / "main" / "java" / "p").mkdir(parents=True)
    (mvnw_repo / "src" / "test" / "java" / "p").mkdir(parents=True)
    (mvnw_repo / "pom.xml").write_text("<project/>\n", encoding="utf-8")
    mvnw_script = mvnw_repo / "mvnw"
    mvnw_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mvnw_script.chmod(0o755)
    (mvnw_repo / "src" / "main" / "java" / "p" / "Billing.java").write_text("class Billing {}\n", encoding="utf-8")
    (mvnw_repo / "src" / "test" / "java" / "p" / "BillingTest.java").write_text("class BillingTest {}\n", encoding="utf-8")
    det_mvnw = detect_project_test_runner(mvnw_repo)
    assert det_mvnw["primary_command"] == "./mvnw -q test"
    assert det_mvnw["available"] is True
    assert "Detected checks: `./mvnw -q test`" in build_repo_map(mvnw_repo)
    rel_mvnw = discover_related_test_commands(mvnw_repo, ["src/main/java/p/Billing.java"], [])
    assert rel_mvnw == [
        "./mvnw -q test -Dtest=BillingTest -Dsurefire.failIfNoSpecifiedTests=false -DfailIfNoTests=false"
    ]

    multi_gradle = tmp_path / "multi_gradle"
    (multi_gradle / "subA" / "src" / "test" / "java").mkdir(parents=True)
    gw = multi_gradle / "gradlew"
    gw.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    gw.chmod(0o755)
    (multi_gradle / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")
    (multi_gradle / "settings.gradle").write_text("include(':subA', ':subB')\n", encoding="utf-8")
    (multi_gradle / "subA" / "src" / "test" / "java" / "SubATest.java").write_text("class SubATest {}\n", encoding="utf-8")
    assert discover_related_test_commands(multi_gradle, ["subA/src/test/java/SubATest.java"], []) == [
        "./gradlew test"
    ]
    monkeypatch.setenv("PATH", orig_path)

    # 3. `pre_tool_guard` mutating snapshot/autofix blocks & read-only runner allowances
    ws = tmp_path / "guard_ws"
    ws.mkdir()
    for blocked_cmd in (
        "npm run lint -- --fix",
        "npm run format:fix",
        "npm pkg set name=x",
        "npm pkg delete scripts.test",
        "npm pkg fix",
        "npm shrinkwrap",
        "npm config set foo=bar --location=project",
        "yarn set version stable",
        "yarn plugin import typescript",
        "yarn config set nodeLinker node-modules",
        "pnpm import",
        "pnpm test -- --update-snapshots",
        "yarn dlx create-next-app demo",
        "bunx eslint --fix .",
        "bunx prettier --write .",
        "pnpx eslint --fix .",
        "bunx vitest run -u",
        "pytest --snapshot-update",
        "pytest --inline-snapshot=fix",
        "cargo insta accept",
        "go test -update ./...",
        "ruby -r ./evil.rb test/foo_test.rb",
    ):
        assert detect_bash_workspace_mutation(blocked_cmd, ws)["blocked"] is True, f"Should block: {blocked_cmd}"
    for allowed_cmd in (
        "node --test",
        "ruby -Itest test/foo_test.rb",
        "php artisan test",
        "npm test -- --grep update",
        "npm test -- -t update",
        "yarn test --testNamePattern update",
        "npm pkg get name",
        "yarn config get nodeLinker",
        "npm ls",
        "pnpm why vitest",
        "yarn why jest",
        "npm outdated",
        "bunx vitest run",
        "pnpx eslint src/",
    ):
        assert detect_bash_workspace_mutation(allowed_cmd, ws)["blocked"] is False, f"Should allow: {allowed_cmd}"

    # 4. Sparse worktree: opt-in by default, traces cross-package imports when enabled, AND supports
    # widening `allowed_paths` on `delegate_continue` after Turn 1 already modified a file!
    monkeypatch.delenv("CC_AGY_SPARSE_WORKTREE", raising=False)
    monkeypatch.delenv("CC_AGY_SPARSE_THRESHOLD_FILES", raising=False)
    srepo = tmp_path / "sparse_import_repo"
    _init_git_repo(srepo)
    (srepo / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (srepo / "pkg" / "a").mkdir(parents=True)
    (srepo / "pkg" / "b").mkdir(parents=True)
    (srepo / "pkg" / "c").mkdir(parents=True)
    (srepo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (srepo / "pkg" / "a" / "__init__.py").write_text("", encoding="utf-8")
    (srepo / "pkg" / "b" / "__init__.py").write_text("", encoding="utf-8")
    (srepo / "pkg" / "c" / "__init__.py").write_text("", encoding="utf-8")
    (srepo / "pkg" / "b" / "util.py").write_text("def bonus() -> int:\n    return 10\n", encoding="utf-8")
    (srepo / "pkg" / "c" / "other.py").write_text("def scale() -> int:\n    return 2\n", encoding="utf-8")
    (srepo / "pkg" / "a" / "foo.py").write_text(
        "from pkg.b.util import bonus\ndef compute() -> int:\n    return bonus() + 1\n",
        encoding="utf-8",
    )
    (srepo / "tests").mkdir()
    (srepo / "tests" / "test_foo.py").write_text(
        "from pkg.a.foo import compute\ndef test_compute() -> None:\n    assert compute() == 50\n",
        encoding="utf-8",
    )
    for i in range(15):
        (srepo / "pkg" / "c" / f"extra_{i}.py").write_text(f"X = {i}\n", encoding="utf-8")
    _git_commit_all(srepo)

    s_store = ReceiptStore(tmp_path / "sdata", workspace_root=srepo)
    monkeypatch.setenv("CC_AGY_SPARSE_WORKTREE", "1")
    seen_turn1_util = []

    def two_turn_sparse_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        joined = "\n".join(req.get("prompts") or [])
        if "Widen to pkg/c/other.py" not in joined:
            # Turn 1: `pkg/b/util.py` MUST be materialized via cross-package import tracing!
            seen_turn1_util.append((wt / "pkg" / "b" / "util.py").is_file())
            assert not (wt / "pkg" / "c" / "other.py").exists()
            (wt / "pkg" / "a" / "foo.py").write_text(
                "from pkg.b.util import bonus\nfrom pkg.c.other import scale\ndef compute() -> int:\n    return bonus() * scale()\n",
                encoding="utf-8",
            )
            return {"text": "paused", "stop_reason": "MAX_TOOL_CALLS_EXCEEDED"}
        # Turn 2 after widening allowed_paths to include pkg/c/other.py: BOTH Turn-1 edits on foo.py
        # AND baseline pkg/c/other.py must exist cleanly!
        assert "from pkg.c.other import scale" in (wt / "pkg" / "a" / "foo.py").read_text(encoding="utf-8")
        assert (wt / "pkg" / "c" / "other.py").is_file()
        (wt / "pkg" / "c" / "other.py").write_text("def scale() -> int:\n    return 5\n", encoding="utf-8")
        return {"text": "finished both"}

    jm_s = JobManager(workspace_root=srepo, store=s_store, worker_backend=two_turn_sparse_worker)
    t1 = jm_s.delegate_start(
        "Update compute in pkg/a/foo.py",
        allowed_paths=["pkg/a/foo.py"],
        project="p",
        wait=True,
        auto_apply=False,
    )
    assert seen_turn1_util == [True]
    assert t1["status"] == "checkpoint"

    t2 = jm_s.delegate_continue(
        t1["job_id"],
        "Widen to pkg/c/other.py and set scale() to 5",
        allowed_paths=["pkg/a/foo.py", "pkg/c/other.py"],
        acceptance=["pytest -q tests/test_foo.py"],
        project="p",
        wait=True,
        auto_apply=False,
    )
    assert t2["status"] == "ok"
    assert t2["acceptance_passed"] is True
    app_s = jm_s.delegate_apply(t1["job_id"])
    assert app_s["applied"] is True
    assert sorted(app_s["applied_files"]) == ["pkg/a/foo.py", "pkg/c/other.py"]

    # 5. Unrelated Task 2 after a FAILED Task 1 stays on Flash (does not escalate every subsequent session job)
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "1")
    monkeypatch.delenv("CC_AGY_SPARSE_WORKTREE", raising=False)
    frepo = tmp_path / "fail_session_repo"
    _init_git_repo(frepo)
    (frepo / "mod_fail.py").write_text("a = 1\n", encoding="utf-8")
    (frepo / "mod_clean.py").write_text("b = 1\n", encoding="utf-8")
    (frepo / "test_mod_fail.py").write_text(
        "import mod_fail\ndef test_a() -> None:\n    assert mod_fail.a == 999\n",
        encoding="utf-8",
    )
    (frepo / "test_mod_clean.py").write_text(
        "import mod_clean\ndef test_b() -> None:\n    assert mod_clean.b == 2\n",
        encoding="utf-8",
    )
    _git_commit_all(frepo)
    fstore = ReceiptStore(tmp_path / "fdata", workspace_root=frepo)

    def mixed_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        if "mod_fail.py" in "\n".join(req.get("prompts") or []):
            (wt / "mod_fail.py").write_text("a = 2\n", encoding="utf-8")
        else:
            (wt / "mod_clean.py").write_text("b = 2\n", encoding="utf-8")
        return {"text": "ok"}

    jm_f = JobManager(workspace_root=frepo, store=fstore, worker_backend=mixed_worker)
    jf1 = jm_f.delegate_start(
        "Update mod_fail.py",
        allowed_paths=["mod_fail.py"],
        acceptance=["pytest -q test_mod_fail.py"],
        project="p",
        wait=True,
        auto_apply=False,
    )
    assert jf1["acceptance_passed"] is False
    assert jf1["model"] == "gemini-3.8-flash"

    # Disjoint Task 2 on mod_clean.py stays on Flash
    jf2 = jm_f.delegate_start(
        "Update mod_clean.py",
        allowed_paths=["mod_clean.py"],
        acceptance=["pytest -q test_mod_clean.py"],
        project="p",
        wait=True,
        auto_apply=False,
    )
    assert jf2["model"] == "gemini-3.8-flash"
    assert "escalated_from_model" not in jf2


