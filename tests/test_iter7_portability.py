"""Tests for testless-repository verification and host-portable install (Codex and Claude Code)."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.session_start import REVIEW_AFTER_APPLY_GUIDANCE, REVIEW_BEFORE_APPLY_GUIDANCE
from server.doctor import run_doctor
from server.install import (
    CodexConfigError,
    build_codex_server_block,
    install_codex,
    preflight,
    run_interactive_wizard,
    upsert_codex_server_block,
)
from server.jobs import (
    _CHANGED_FILE_ERROR_FILTER,
    QA_DIFF_AUDIT_SYSTEM,
    JobManager,
    build_compile_smoke_commands,
    build_qa_diff_audit_system,
    classify_verification,
    has_real_acceptance_pass,
    is_quota_or_rate_limit_error,
    run_acceptance,
)
from server.mcp_stdio import HOST_REVIEW_INSTRUCTIONS, MCPServer, server_instructions_for_host
from server.receipts import BudgetExceededError, ReceiptStore, resolve_workspace_root
from server.regions import (
    resolve_auth_config,
    resolve_max_daily_cents,
    resolve_vertex_base_url,
    resolve_vertex_base_url_token,
)
from server.writers import _build_genai_client


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)


def _commit_all(repo: Path, msg: str = "init") -> None:
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", msg], cwd=str(repo), check=True)


# ---------------------------------------------------------------------------
# Compile/syntax smoke checks for repositories without tests
# ---------------------------------------------------------------------------


def test_compile_smoke_catches_syntax_errors_across_languages(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "run.py").write_text("def main(:\n    pass\n", encoding="utf-8")
    (repo / "deploy.sh").write_text("if true; then\n  echo hi\n", encoding="utf-8")
    (repo / "settings.json").write_text('{"a": 1,}', encoding="utf-8")
    files = ["run.py", "deploy.sh", "settings.json"]

    cmds = build_compile_smoke_commands(repo, files)
    assert any("ast.parse" in c for c in cmds)
    assert any(c.startswith("bash -n") for c in cmds)
    assert any("json.load" in c for c in cmds)
    results = run_acceptance(cmds, repo)
    assert results and all(r["passed"] is False for r in results)

    (repo / "run.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    (repo / "deploy.sh").write_text("if true; then\n  echo hi\nfi\n", encoding="utf-8")
    (repo / "settings.json").write_text('{"a": 1}', encoding="utf-8")
    results_ok = run_acceptance(build_compile_smoke_commands(repo, files), repo)
    assert results_ok and all(r["passed"] is True for r in results_ok)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_compile_smoke_checks_plain_javascript_with_node(tmp_path: Path) -> None:
    repo = tmp_path / "jsrepo"
    repo.mkdir()
    (repo / "app.js").write_text("function f( {\n", encoding="utf-8")
    cmds = build_compile_smoke_commands(repo, ["app.js"])
    assert cmds == ["node --check ./app.js"]
    assert run_acceptance(cmds, repo)[0]["passed"] is False
    (repo / "app.js").write_text("function f() { return 1; }\n", encoding="utf-8")
    assert run_acceptance(cmds, repo)[0]["passed"] is True


@pytest.mark.skipif(shutil.which("go") is None, reason="go not installed")
def test_compile_smoke_builds_changed_go_packages_and_skips_test_only_edits(tmp_path: Path) -> None:
    repo = tmp_path / "gorepo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "go.mod").write_text("module ex\n\ngo 1.21\n", encoding="utf-8")
    (repo / "pkg" / "util.go").write_text('package pkg\n\nfunc A() int { return "x" }\n', encoding="utf-8")
    (repo / "pkg" / "util_test.go").write_text("package pkg\n", encoding="utf-8")
    assert build_compile_smoke_commands(repo, ["pkg/util_test.go"]) == []
    cmds = build_compile_smoke_commands(repo, ["pkg/util.go"])
    assert cmds == ["go build -o /dev/null ./pkg"]
    assert run_acceptance(cmds, repo)[0]["passed"] is False
    (repo / "pkg" / "util.go").write_text("package pkg\n\nfunc A() int { return 1 }\n", encoding="utf-8")
    assert run_acceptance(cmds, repo)[0]["passed"] is True


def test_compile_smoke_skips_json_with_comments_configs(tmp_path: Path) -> None:
    repo = tmp_path / "jsonc"
    (repo / ".vscode").mkdir(parents=True)
    (repo / ".vscode" / "settings.json").write_text('{ // comment\n "a": 1 }', encoding="utf-8")
    (repo / "tsconfig.build.json").write_text('{ /* c */ }', encoding="utf-8")
    assert build_compile_smoke_commands(repo, [".vscode/settings.json", "tsconfig.build.json"]) == []


def test_compile_smoke_skips_typescript_without_local_tsc_and_rust_without_warm_target(tmp_path: Path) -> None:
    repo = tmp_path / "tsrepo"
    repo.mkdir()
    (repo / "tsconfig.json").write_text("{}", encoding="utf-8")
    (repo / "a.ts").write_text("export const x: number = 1;\n", encoding="utf-8")
    (repo / "Cargo.toml").write_text('[package]\nname = "x"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    cmds = build_compile_smoke_commands(repo, ["a.ts", "src/main.rs"])
    # Never `npx tsc` (which can download packages) and never a cold `cargo check`.
    assert not any("tsc" in c for c in cmds)
    assert not any("cargo" in c for c in cmds)


def test_typescript_error_filter_ignores_errors_outside_changed_files(tmp_path: Path) -> None:
    def _run(rc: int, output: str, files: list[str]) -> int:
        cmd = (
            f"printf '%s\\n' {shlex.quote(output)} | python3 -c {shlex.quote(_CHANGED_FILE_ERROR_FILTER)} "
            f"{rc} {' '.join(shlex.quote(f) for f in files)}"
        )
        return subprocess.run(["bash", "-c", cmd], cwd=str(tmp_path), capture_output=True, check=False).returncode

    legacy_only = "src/legacy.ts(3,1): error TS2304: Cannot find name 'foo'."
    assert _run(2, legacy_only, ["src/new.ts"]) == 0
    changed = "src/new.ts(10,5): error TS2322: Type 'string' is not assignable to type 'number'."
    assert _run(2, legacy_only + "\n" + changed, ["src/new.ts"]) == 1
    assert _run(0, "", ["src/new.ts"]) == 0
    assert _run(1, "./pkg/util.go:12:3: undefined: Foo", ["pkg/util.go"]) == 1
    # An error in lib/a.ts must not fail a change to a.ts (suffix collision).
    assert _run(2, "lib/a.ts(3,1): error TS2322: pre-existing", ["a.ts"]) == 0


def test_classify_verification_levels() -> None:
    explicit = [{"command": "pytest -q", "passed": True, "counts": {"passed": 3}}]
    assert has_real_acceptance_pass(explicit)
    assert classify_verification(explicit, True) == "acceptance_tests"
    related = [{"command": "pytest -q tests/test_a.py", "passed": True, "auto_discovered_default": True, "counts": {"passed": 4}}]
    assert classify_verification(related, True) == "related_tests"
    smoke = [{"command": "python3 -c ...", "passed": True, "auto_discovered_default": True, "compile_smoke": True}]
    assert not has_real_acceptance_pass(smoke)
    assert classify_verification(smoke, True) == "smoke_only"
    skipped = [{"command": "go build", "passed": True, "auto_discovered_default": True, "runner_incompatible_skipped": True}]
    assert classify_verification(skipped, True) == "unverified"
    assert classify_verification([], None) == "unverified"
    assert classify_verification(explicit, False) == "failed"


def test_testless_repo_syntax_error_fails_acceptance_and_valid_edit_requires_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CC_AGY_AUTO_DEFAULT_ACCEPTANCE", "1")
    monkeypatch.setenv("CC_AGY_AUTO_APPLY_ON_PASS", "1")
    repo = tmp_path / "scripts_repo"
    _init_repo(repo)
    (repo / "deploy.sh").write_text("#!/bin/bash\necho deploy\n", encoding="utf-8")
    _commit_all(repo)

    def broken_worker(req: dict[str, Any]) -> dict[str, Any]:
        (Path(req["workspace_dir"]) / "deploy.sh").write_text("#!/bin/bash\nif true; then\n echo x\n", encoding="utf-8")
        return {"text": "Updated deploy.sh.", "usage": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15}}

    store = ReceiptStore(tmp_path / "data_broken", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=broken_worker)
    try:
        bad = jm.delegate_start("Add a dry-run flag to deploy.sh.", project="p", wait=True)
    finally:
        jm.shutdown(wait=True)
    assert bad["acceptance_passed"] is False
    assert bad.get("verification") == "failed"
    assert not bad.get("applied")
    assert (repo / "deploy.sh").read_text(encoding="utf-8") == "#!/bin/bash\necho deploy\n"

    def good_worker(req: dict[str, Any]) -> dict[str, Any]:
        (Path(req["workspace_dir"]) / "deploy.sh").write_text(
            "#!/bin/bash\nif [ \"$1\" = --dry-run ]; then\n  echo dry\nfi\necho deploy\n", encoding="utf-8"
        )
        return {"text": "Added --dry-run.", "usage": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15}}

    store2 = ReceiptStore(tmp_path / "data_good", workspace_root=repo)
    jm2 = JobManager(workspace_root=repo, store=store2, worker_backend=good_worker)
    try:
        good = jm2.delegate_start("Add a dry-run flag to deploy.sh.", project="p", wait=True)
    finally:
        jm2.shutdown(wait=True)
    assert good["acceptance_passed"] is True
    assert good.get("verification") == "smoke_only"
    assert good.get("applied") is True
    assert "real_acceptance_passed" not in good
    assert good.get("review_required") is True
    assert "no explicit acceptance test verified it" in str(good.get("next_action"))
    assert good.get("diff_preview"), "diff must stay visible for Lead review when no test verified it"


def test_unapplied_testless_job_does_not_tell_lead_to_apply_blindly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CC_AGY_AUTO_APPLY_ON_PASS", raising=False)
    monkeypatch.delenv("CC_AGY_STRICT_MODE", raising=False)
    monkeypatch.setenv("CC_AGY_AUTO_DEFAULT_ACCEPTANCE", "0")
    repo = tmp_path / "docs_repo"
    _init_repo(repo)
    (repo / "notes.txt").write_text("v1\n", encoding="utf-8")
    _commit_all(repo)

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        (Path(req["workspace_dir"]) / "notes.txt").write_text("v2\n", encoding="utf-8")
        return {"text": "Updated notes.", "usage": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15}}

    store = ReceiptStore(tmp_path / "data_docs", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        digest = jm.delegate_start("Bump notes to v2.", project="p", wait=True)
    finally:
        jm.shutdown(wait=True)
    assert digest["acceptance_passed"] is None
    assert digest.get("verification") == "unverified"
    assert not digest.get("applied")
    action = str(digest.get("next_action"))
    assert "No explicit acceptance test verified this change" in action
    assert "directly to land these changes" not in action


def test_host_node_modules_linked_only_for_plugin_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CC_AGY_AUTO_APPLY_ON_PASS", raising=False)
    monkeypatch.delenv("CC_AGY_STRICT_MODE", raising=False)
    repo = tmp_path / "jsapp"
    _init_repo(repo)
    (repo / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (repo / "index.txt").write_text("v1\n", encoding="utf-8")
    _commit_all(repo)
    (repo / "node_modules" / "dep").mkdir(parents=True)
    (repo / "node_modules" / "dep" / "index.js").write_text("module.exports = 1;\n", encoding="utf-8")
    seen_by_worker: list[tuple[bool, bool]] = []

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        nm_dir = wt / "node_modules"
        seen_by_worker.append((nm_dir.is_dir(), nm_dir.is_symlink()))
        (wt / "index.txt").write_text("v2\n", encoding="utf-8")
        return {"text": "done", "usage": {"prompt_token_count": 1, "candidates_token_count": 1, "total_token_count": 2}}

    store = ReceiptStore(tmp_path / "data_nm", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        digest = jm.delegate_start("Bump index.", project="p", wait=True, acceptance=["test -f node_modules/dep/index.js"])
        # Also verify CC_AGY_LINK_NODE_MODULES=0 disables linking during worker execution
        monkeypatch.setenv("CC_AGY_LINK_NODE_MODULES", "0")
        _ = jm.delegate_start("Bump index disabled link.", project="p", wait=True)
    finally:
        jm.shutdown(wait=True)
    assert seen_by_worker == [(True, False), (False, False)]
    assert digest["acceptance_passed"] is True
    assert digest["files_changed"] == ["index.txt"]
    assert not digest.get("scope_violations")
    assert (repo / "node_modules" / "dep" / "index.js").is_file()


def test_review_guidance_keys_on_real_acceptance_pass() -> None:
    assert "real_acceptance_passed is true" in REVIEW_AFTER_APPLY_GUIDANCE
    assert "repository without tests" in REVIEW_AFTER_APPLY_GUIDANCE
    assert "verification is not acceptance_tests" in REVIEW_BEFORE_APPLY_GUIDANCE


# ---------------------------------------------------------------------------
# Host portability: MCP instructions, workspace root, Codex installer
# ---------------------------------------------------------------------------


def test_mcp_instructions_only_for_hosts_without_hooks() -> None:
    assert server_instructions_for_host({}) == HOST_REVIEW_INSTRUCTIONS
    assert server_instructions_for_host({"CLAUDE_PLUGIN_ROOT": "/opt/plugin"}) is None
    assert server_instructions_for_host({"CLAUDECODE": "1"}) is None
    assert server_instructions_for_host({"CLAUDE_PLUGIN_ROOT": "${CLAUDE_PLUGIN_ROOT}"}) == HOST_REVIEW_INSTRUCTIONS
    assert server_instructions_for_host({"CLAUDECODE": "1", "CC_AGY_MCP_INSTRUCTIONS": "1"}) == HOST_REVIEW_INSTRUCTIONS
    assert server_instructions_for_host({"CC_AGY_MCP_INSTRUCTIONS": "0"}) is None


def test_workspace_root_env_override(tmp_path: Path) -> None:
    ws = tmp_path / "proj"
    _init_repo(ws)
    assert resolve_workspace_root(env={"CC_AGY_WORKSPACE_ROOT": str(ws)}) == ws.resolve()
    other = tmp_path / "other"
    _init_repo(other)
    assert resolve_workspace_root(env={"CLAUDE_PROJECT_DIR": str(other), "CC_AGY_WORKSPACE_ROOT": str(ws)}) == other.resolve()


def test_codex_block_never_stores_api_keys_and_sets_long_tool_timeout() -> None:
    block = build_codex_server_block(name="agy", python_bin="/venv/bin/python", profile="quality", project="my-proj")
    assert "[mcp_servers.agy]" in block and "[mcp_servers.agy.env]" in block
    assert 'CC_AGY_COST_PROFILE = "quality"' in block
    assert 'GOOGLE_CLOUD_PROJECT = "my-proj"' in block
    assert "tool_timeout_sec = 1800" in block
    assert '"GEMINI_API_KEY"' in block.split("env_vars = ", 1)[1].splitlines()[0]
    assert "GEMINI_API_KEY =" not in block


def test_codex_upsert_replaces_only_its_own_tables() -> None:
    existing = (
        'model = "gpt-5"\n\n[mcp_servers.other]\ncommand = "echo"\n\n'
        '[mcp_servers.agy]\ncommand = "old"\n\n[mcp_servers.agy.env]\nX = "1"\n\n[profiles.fast]\nmodel = "o3"\n'
    )
    block = build_codex_server_block(name="agy", python_bin="/py", profile="balanced")
    updated = upsert_codex_server_block(existing, block, name="agy")
    assert 'command = "old"' not in updated and 'X = "1"' not in updated
    assert '[mcp_servers.other]\ncommand = "echo"' in updated
    assert '[profiles.fast]\nmodel = "o3"' in updated
    assert updated.count("[mcp_servers.agy]") == 1
    assert upsert_codex_server_block(updated, block, name="agy") == updated
    fresh = upsert_codex_server_block("", block, name="agy")
    assert fresh.startswith("[mcp_servers.agy]")


@pytest.mark.skipif(shutil.which("codex") is None, reason="codex CLI not installed")
def test_codex_cli_parses_installed_block(tmp_path: Path) -> None:
    home = tmp_path / "codex_home"
    home.mkdir()
    (home / "config.toml").write_text(build_codex_server_block(name="agy", python_bin="/py", profile="economy"), encoding="utf-8")
    proc = subprocess.run(
        ["codex", "mcp", "get", "agy", "--json"],
        capture_output=True,
        text=True,
        check=False,
        env={**__import__("os").environ, "CODEX_HOME": str(home)},
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    cfg = json.loads(proc.stdout[proc.stdout.index("{"):])
    assert cfg["tool_timeout_sec"] == 1800.0
    assert "GEMINI_API_KEY" in cfg["transport"]["env_vars"]
    assert cfg["transport"]["env"]["CC_AGY_COST_PROFILE"] == "economy"


def test_claude_marketplace_manifest_points_at_repo_root() -> None:
    root = Path(__file__).resolve().parent.parent
    market = json.loads((root / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    plugin = json.loads((root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    entry = market["plugins"][0]
    assert entry["name"] == plugin["name"] == "cc-agy-plugin"
    assert entry["source"] == "./"
    assert entry["version"] == plugin["version"]


def test_smoke_checks_resist_option_like_names_and_accept_bom_json(tmp_path: Path) -> None:
    repo = tmp_path / "odd"
    repo.mkdir()
    (repo / "--version.sh").write_text("echo ok\n", encoding="utf-8")
    (repo / "data.json").write_bytes(b"\xef\xbb\xbf" + b'{"a": 1}')
    cmds = build_compile_smoke_commands(repo, ["--version.sh", "data.json"])
    assert "bash -n ./--version.sh" in cmds
    results = run_acceptance(cmds, repo)
    assert all(r["passed"] is True for r in results), results


@pytest.mark.parametrize(
    "existing",
    [
        '[mcp_servers.agy]  # my server\ncommand = "old"\n\n[mcp_servers.agy.env]\nX = "1"\n',
        "[mcp_servers.'agy']\ncommand = \"old\"\n",
        '[mcp_servers.agy]\ncommand = "old"\nargs = [\n  ["a", "b"],\n  ["c"],\n]\n\n[other]\nk = 1\n',
    ],
)
def test_codex_upsert_handles_comments_quotes_and_nested_arrays(existing: str) -> None:
    tomllib = pytest.importorskip("tomllib")
    block = build_codex_server_block(name="agy", python_bin="/py", profile="balanced")
    updated = upsert_codex_server_block(existing, block, name="agy")
    parsed = tomllib.loads(updated)
    assert parsed["mcp_servers"]["agy"]["command"] == "/py"
    if "[other]" in existing:
        assert parsed["other"] == {"k": 1}


@pytest.mark.parametrize(
    "existing",
    ['[mcp_servers]\nagy.command = "old"\nagy.args = []\n', 'mcp_servers.agy = { command = "old" }\n'],
)
def test_codex_upsert_refuses_forms_it_cannot_rewrite(existing: str) -> None:
    pytest.importorskip("tomllib")
    block = build_codex_server_block(name="agy", python_bin="/py", profile="balanced")
    with pytest.raises(CodexConfigError):
        upsert_codex_server_block(existing, block, name="agy")


def test_codex_install_backup_is_owner_only(tmp_path: Path) -> None:
    home = tmp_path / "codex_home"
    home.mkdir()
    cfg = home / "config.toml"
    cfg.write_text('[mcp_servers.other]\ncommand = "x"\n[mcp_servers.other.env]\nTOKEN = "secret"\n', encoding="utf-8")
    cfg.chmod(0o600)
    res = install_codex(name="agy", profile="balanced", project=None, location=None, dry_run=False, codex_home=home)
    assert res["status"] == "installed"
    assert (Path(res["backup_path"]).stat().st_mode & 0o777) == 0o600
    assert (cfg.stat().st_mode & 0o777) == 0o600


# ---------------------------------------------------------------------------
# Iteration 8: Codex native plugin, AI gateway support, blocker-only QA,
# immediate 429/503 quota escalation, daily spend cap, and validate.sh
# ---------------------------------------------------------------------------


def test_codex_native_plugin_and_marketplace_manifests() -> None:
    root = Path(__file__).resolve().parent.parent
    codex_market = json.loads((root / ".agents" / "plugins" / "marketplace.json").read_text(encoding="utf-8"))
    codex_plugin = json.loads(
        (root / "plugins" / "cc-agy-plugin" / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    claude_plugin = json.loads((root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    mcp_cfg = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))

    assert codex_market["interface"]["displayName"]
    entry = codex_market["plugins"][0]
    assert entry["name"] == codex_plugin["name"] == "cc-agy-plugin"
    assert entry["source"] == {"source": "local", "path": "./plugins/cc-agy-plugin"}
    assert entry["policy"] == {"installation": "AVAILABLE", "authentication": "ON_INSTALL"}
    assert entry["category"] == "Productivity"

    assert codex_plugin["version"] == claude_plugin["version"]
    assert codex_plugin["skills"] == "./skills/"
    assert "mcpServers" not in codex_plugin
    prompts = codex_plugin["interface"]["defaultPrompt"]
    assert 1 <= len(prompts) <= 3
    assert all(len(p.encode("utf-8")) <= 128 for p in prompts)
    assert (root / "plugins" / "cc-agy-plugin" / "LICENSE").read_bytes() == (root / "LICENSE").read_bytes()
    assert (root / "plugins" / "cc-agy-plugin" / "skills" / "vertex" / "SKILL.md").is_file()

    user_cfg = claude_plugin["userConfig"]
    for key in ("vertex_base_url", "vertex_base_url_token", "max_daily_cents"):
        assert key in user_cfg
    assert user_cfg["vertex_base_url_token"]["sensitive"] is True
    mcp_env = mcp_cfg["mcpServers"]["vertex"]["env"]
    assert mcp_env["CC_AGY_USERCFG_VERTEX_BASE_URL"] == "${user_config.vertex_base_url}"
    assert mcp_env["CC_AGY_USERCFG_VERTEX_BASE_URL_TOKEN"] == "${user_config.vertex_base_url_token}"
    assert mcp_env["CC_AGY_USERCFG_MAX_DAILY_CENTS"] == "${user_config.max_daily_cents}"


def test_codex_install_forwards_gateway_vars_and_syncs_skill(tmp_path: Path) -> None:
    default_block = build_codex_server_block(name="agy", python_bin="/py", profile="balanced")
    for var_name in (
        "CC_AGY_VERTEX_BASE_URL",
        "GOOGLE_VERTEX_BASE_URL",
        "CC_AGY_VERTEX_BASE_URL_TOKEN",
        "CC_AGY_MAX_DAILY_CENTS",
    ):
        assert f'"{var_name}"' in default_block

    home = tmp_path / "codex_home"
    first = install_codex(
        name="agy",
        profile="balanced",
        project=None,
        location=None,
        base_url="https://gw.example.internal",
        max_daily_cents=500.0,
        dry_run=False,
        codex_home=home,
    )
    assert first["status"] == "installed"
    skill_dst = home / "skills" / "cc-agy-plugin" / "SKILL.md"
    assert skill_dst.is_file()
    cfg_text = (home / "config.toml").read_text(encoding="utf-8")
    for var_name in ("GOOGLE_VERTEX_BASE_URL", "CC_AGY_VERTEX_BASE_URL_TOKEN"):
        assert f'"{var_name}"' in cfg_text
    assert 'CC_AGY_VERTEX_BASE_URL = "https://gw.example.internal"' in cfg_text
    assert 'CC_AGY_MAX_DAILY_CENTS = "500"' in cfg_text
    assert "CC_AGY_VERTEX_BASE_URL_TOKEN =" not in cfg_text

    second = install_codex(
        name="agy",
        profile="balanced",
        project=None,
        location=None,
        base_url="https://gw.example.internal",
        max_daily_cents=500.0,
        dry_run=False,
        codex_home=home,
    )
    assert second["status"] == "unchanged"
    assert preflight({"GEMINI_API_KEY": "ambient-key"}, base_url="https://gw.example.internal")["auth_mode"] == "gateway"


def test_interactive_wizard_builds_expected_cli_flags() -> None:
    answers = iter(["2", "3", "https://gw.example.internal", "1", "750"])
    printed: list[str] = []
    built = run_interactive_wizard(input_fn=lambda _p: next(answers), output_fn=printed.append, env={})
    assert built == [
        "--host",
        "codex",
        "--profile",
        "balanced",
        "--base-url",
        "https://gw.example.internal",
        "--max-daily-cents",
        "750",
    ]

    defaults = iter(["", "", "", ""])
    built_default = run_interactive_wizard(
        input_fn=lambda _p: next(defaults),
        output_fn=printed.append,
        env={"GEMINI_API_KEY": "present"},
    )
    assert built_default == ["--host", "all", "--profile", "balanced"]


def test_gateway_env_and_writer_client_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_vertex_base_url(env={"GOOGLE_VERTEX_BASE_URL": "https://gw.example.internal"}) == "https://gw.example.internal"
    assert resolve_vertex_base_url(env={"CC_AGY_VERTEX_BASE_URL": "${user_config.vertex_base_url}"}) is None
    assert resolve_vertex_base_url_token(env={"CC_AGY_VERTEX_BASE_URL_TOKEN": "vk-123"}) == "vk-123"
    assert resolve_vertex_base_url_token(env={"CC_AGY_VERTEX_BASE_URL_TOKEN": "${user_config.vertex_base_url_token}"}) is None

    # Configured AI gateway takes priority over an ambient GEMINI_API_KEY in the shell
    auth = resolve_auth_config(
        env={"GOOGLE_VERTEX_BASE_URL": "https://gw.example.internal", "GEMINI_API_KEY": "ambient-key"}
    )
    assert auth["auth_mode"] == "vertex"
    assert auth["project"] == "custom-endpoint"

    # Explicit requested_api_key override still selects api_key mode
    explicit_key_auth = resolve_auth_config(
        requested_api_key="explicit-key",
        env={"GOOGLE_VERTEX_BASE_URL": "https://gw.example.internal"},
    )
    assert explicit_key_auth["auth_mode"] == "api_key"

    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    from google import genai

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setenv("GOOGLE_VERTEX_BASE_URL", "https://gw.example.internal")
    monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL_TOKEN", "vk-live-456")
    monkeypatch.delenv("CC_AGY_VERTEX_BASE_URL", raising=False)

    _build_genai_client({"auth_mode": "vertex", "project": "custom-endpoint", "location": "global"}, timeout_ms=15000)
    assert captured["vertexai"] is True
    assert captured["project"] == "custom-endpoint"
    http_opts = captured["http_options"]
    assert http_opts.base_url == "https://gw.example.internal"
    assert http_opts.headers["Authorization"] == "Bearer vk-live-456"


def test_qa_diff_audit_system_skips_unit_tests_on_testless_repos() -> None:
    assert "Report ONLY High or Medium functional blockers" in QA_DIFF_AUDIT_SYSTEM
    assert "Do NOT report Low, Optional, polish" in QA_DIFF_AUDIT_SYSTEM

    testless = build_qa_diff_audit_system(verification="smoke_only", prompt="Add a --dry-run flag to deploy.sh")
    assert "Do NOT fail the diff for missing unit tests" in testless

    asked_for_tests = build_qa_diff_audit_system(
        verification="smoke_only", prompt="Add a --dry-run flag and write unit tests"
    )
    assert asked_for_tests == QA_DIFF_AUDIT_SYSTEM

    with_suite = build_qa_diff_audit_system(verification="acceptance_tests", prompt="Add a --dry-run flag")
    assert with_suite == QA_DIFF_AUDIT_SYSTEM


def test_quota_rate_limit_error_escalates_immediately_to_lead(tmp_path: Path) -> None:
    assert is_quota_or_rate_limit_error("429 RESOURCE_EXHAUSTED: Quota exceeded for metric")
    assert is_quota_or_rate_limit_error("503 Service Unavailable (UNAVAILABLE): model overloaded")
    assert not is_quota_or_rate_limit_error("AssertionError: expected 2 got 3")

    repo = tmp_path / "quota_repo"
    _init_repo(repo)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    _commit_all(repo)

    calls = 0

    def quota_worker(_req: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded for gemini-3.8-flash")

    store = ReceiptStore(tmp_path / "data_quota", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=quota_worker)
    try:
        first = jm.delegate_start("Set x = 2 in app.py.", project="p", wait=True)
        assert first["failed_attempts"] == 1
        esc = first["escalation"]
        assert esc["escalate_to_lead"] is True
        assert esc["reason"] == "quota_or_rate_limit"
        assert "force_retry=true" in esc["recommendation"]
        assert calls == 1

        second = jm.delegate_continue(first["job_id"], "Please retry", wait=True)
        assert second["status"] == "escalated"
        assert second["continued"] is False
        assert calls == 1
    finally:
        jm.shutdown(wait=True)


def test_daily_budget_cap_blocks_paid_dispatches_and_surfaces_in_doctor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for bad_cap in ("nan", "inf", "-5"):
        with pytest.raises(ValueError):
            resolve_max_daily_cents({"CC_AGY_MAX_DAILY_CENTS": bad_cap})
    assert resolve_max_daily_cents({"CC_AGY_MAX_DAILY_CENTS": "0", "CC_AGY_USERCFG_MAX_DAILY_CENTS": "500"}) is None

    monkeypatch.setenv("CC_AGY_MAX_DAILY_CENTS", "5.0")
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    repo = tmp_path / "budget_repo"
    _init_repo(repo)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    _commit_all(repo)

    calls = 0

    def paid_worker(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        (Path(req["workspace_dir"]) / "app.py").write_text("x = 2\n", encoding="utf-8")
        return {
            "text": "Updated x.",
            "usage": {"prompt_token_count": 200_000, "candidates_token_count": 1_000, "total_token_count": 201_000},
        }

    store = ReceiptStore(tmp_path / "data_budget", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=paid_worker)
    srv = MCPServer(job_manager=jm, store=store)
    try:
        # Acceptance fails so balanced profile attempts Flash->Argon escalation after Flash spent > 5c;
        # the finished Flash job's job_id and diff must be preserved instead of lost.
        first = jm.delegate_start(
            "Set x = 2.",
            model="gemini-3.8-flash",
            project="p",
            acceptance=["python3 -c 'import sys; sys.exit(1)'"],
            wait=True,
        )
        assert first["status"] == "ok"
        assert first["job_id"]
        assert first["budget_exceeded"] is True
        assert calls == 1

        with pytest.raises(BudgetExceededError):
            jm.delegate_start("Set x = 3.", model="gemini-3.8-flash", project="p", wait=True)
        assert calls == 1

        rpc = srv.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {"name": "delegate_start", "arguments": {"prompt": "Set x = 4.", "wait": True}},
            }
        )
        assert isinstance(rpc, dict)
        assert rpc["result"]["isError"] is True
        err_payload = json.loads(rpc["result"]["content"][0]["text"])
        assert err_payload["status"] == "budget_exceeded"
        assert err_payload["max_daily_cents"] == 5.0
        assert calls == 1

        monkeypatch.setattr("server.doctor._module_available", lambda _name: True)
        doc = run_doctor(workspace_root=repo, store=store)
        assert doc["healthy"] is True
        assert doc["daily_budget"]["exceeded"] is True
        assert doc["daily_budget"]["max_daily_cents"] == 5.0
        assert any("Daily Gemini budget reached" in w for w in doc["warnings"])
    finally:
        jm.shutdown(wait=True)


def test_validate_script_passes_end_to_end(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "validate.sh"
    assert script.is_file()
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Validation passed." in proc.stdout
