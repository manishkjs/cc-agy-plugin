"""Tests for first-class Cursor IDE support in cc-agy-plugin."""

from __future__ import annotations

import io
import json
from pathlib import Path
import re
from contextlib import redirect_stdout
import pytest

from server.doctor import run_doctor
from server.harness_validator import parse_frontmatter, validate_cursor_plugin, validate_plugin_tree
from server.install import (
    CursorConfigError,
    build_cursor_server_entry,
    install_cursor,
    main as install_main,
    run_interactive_wizard,
    upsert_cursor_mcp_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cursor_plugin_manifest_mcp_and_hooks_contracts() -> None:
    manifest_path = REPO_ROOT / ".cursor-plugin" / "plugin.json"
    mcp_path = REPO_ROOT / ".cursor-plugin" / "mcp.json"
    hooks_path = REPO_ROOT / ".cursor-plugin" / "hooks.json"

    assert manifest_path.is_file()
    assert mcp_path.is_file()
    assert hooks_path.is_file()

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m_ver = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    assert m_ver is not None
    expected_version = m_ver.group(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "cc-agy-plugin"
    assert manifest["version"] == expected_version
    assert manifest["skills"] == "./skills/"
    assert manifest["agents"] == "./agents/"
    assert manifest["commands"] == "./commands/"
    assert manifest["mcpServers"] == "./.cursor-plugin/mcp.json"
    assert manifest["hooks"] == "./.cursor-plugin/hooks.json"

    raw_mcp = mcp_path.read_text(encoding="utf-8")
    assert "${CURSOR_PLUGIN_ROOT}/server/main.py" in raw_mcp
    assert "${PLUGIN_ROOT}" not in raw_mcp
    assert "${PLUGIN_DATA}" not in raw_mcp
    assert "${user_config." not in raw_mcp
    mcp_data = json.loads(raw_mcp)
    assert mcp_data["mcpServers"]["agy"]["command"] == "python3"
    assert mcp_data["mcpServers"]["agy"]["args"] == ["${CURSOR_PLUGIN_ROOT}/server/main.py"]

    hooks_data = json.loads(hooks_path.read_text(encoding="utf-8"))
    assert hooks_data == {"version": 1, "hooks": {}}

    for cmd_md in sorted((REPO_ROOT / "commands").glob("*.md")):
        meta, _body = parse_frontmatter(cmd_md.read_text(encoding="utf-8"))
        assert meta.get("name") == cmd_md.stem
        assert meta.get("description")

    codex_skill = (REPO_ROOT / "plugins" / "cc-agy-plugin" / "skills" / "vertex" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "agy CLI" in codex_skill.splitlines()[2]
    assert "Cursor" in codex_skill

    for rel in (
        ".cursor-plugin/plugin.json",
        "README.md",
        "skills/vertex/SKILL.md",
        "plugins/cc-agy-plugin/skills/vertex/SKILL.md",
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "\u2014" not in text, f"Em-dash found in {rel}"

    assert validate_cursor_plugin(REPO_ROOT) == []
    assert validate_plugin_tree(REPO_ROOT) == []


def test_validate_cursor_plugin_rejects_known_compatibility_traps(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('name = "cc-agy-plugin"\nversion = "0.3.0"\n', encoding="utf-8")
    for sub in ("skills", "agents", "commands", "server", "hooks", ".cursor-plugin"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "server" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {"vertex": {"env": {"X": "${user_config.x}"}}}}\n', encoding="utf-8")
    (tmp_path / "hooks" / "hooks.json").write_text('{"hooks": {"SessionStart": []}}\n', encoding="utf-8")

    # Trap 1: pointing at Claude root .mcp.json + omitting hooks + wrong version
    (tmp_path / ".cursor-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "cc-agy-plugin",
                "version": "0.1.0",
                "skills": "./skills/",
                "agents": "./agents/",
                "commands": "./commands/",
                "mcpServers": "./.mcp.json",
            }
        ),
        encoding="utf-8",
    )
    errs = validate_cursor_plugin(tmp_path)
    assert any("does not match pyproject.toml" in e for e in errs), errs
    assert any("must not point to Claude Code root '.mcp.json'" in e for e in errs), errs
    assert any("auto-discover Claude's hooks/hooks.json" in e for e in errs), errs

    # Trap 2: ${PLUGIN_ROOT} + ${user_config.*} in Cursor mcp.json + PascalCase hook in Cursor hooks.json
    (tmp_path / ".cursor-plugin" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "agy": {
                        "command": "python3",
                        "args": ["${PLUGIN_ROOT}/server/main.py"],
                        "env": {"GEMINI_API_KEY": "${user_config.gemini_api_key}"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / ".cursor-plugin" / "hooks.json").write_text(
        json.dumps({"version": 1, "hooks": {"SessionStart": [{"command": "echo hi"}]}}),
        encoding="utf-8",
    )
    (tmp_path / ".cursor-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "cc-agy-plugin",
                "version": "0.3.0",
                "skills": "./skills/",
                "agents": "./agents/",
                "commands": "./commands/",
                "mcpServers": "./.cursor-plugin/mcp.json",
                "hooks": "./.cursor-plugin/hooks.json",
            }
        ),
        encoding="utf-8",
    )
    errs2 = validate_cursor_plugin(tmp_path)
    assert any("user_config" in e for e in errs2), errs2
    assert any("PLUGIN_ROOT" in e for e in errs2), errs2
    assert any("SessionStart" in e for e in errs2), errs2

    # Clean config passes
    (tmp_path / ".cursor-plugin" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "agy": {
                        "command": "python3",
                        "args": ["${CURSOR_PLUGIN_ROOT}/server/main.py"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / ".cursor-plugin" / "hooks.json").write_text(
        json.dumps({"version": 1, "hooks": {}}),
        encoding="utf-8",
    )
    assert validate_cursor_plugin(tmp_path) == []


def test_install_cursor_lifecycle_idempotence_backups_and_secret_hygiene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-google-key")
    monkeypatch.setenv("ANTIGRAVITY_OAUTH_TOKEN", "secret-oauth-token")
    monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL_TOKEN", "secret-gateway-token")

    entry = build_cursor_server_entry(
        python_bin="/venv/bin/python",
        profile="balanced",
        auth_mode="gemini_enterprise",
        agy_cli_path="/opt/bin/agy",
        project="proj-a",
        base_url="https://gw.example.internal",
        max_daily_cents=350.0,
        allowed_projects="proj-a,proj-b",
        blocked_projects="proj-blocked",
        no_auto_apply=True,
        strict=True,
    )
    assert entry["command"] == "/venv/bin/python"
    assert entry["env"]["CC_AGY_AUTH_MODE"] == "gemini_enterprise"
    assert entry["env"]["CC_AGY_ALLOWED_PROJECTS"] == "proj-a,proj-b"
    assert entry["env"]["CC_AGY_BLOCKED_PROJECTS"] == "proj-blocked"
    assert entry["env"]["CC_AGY_AUTO_APPLY_ON_PASS"] == "0"
    for forbidden in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTIGRAVITY_OAUTH_TOKEN",
        "CC_AGY_VERTEX_BASE_URL_TOKEN",
    ):
        assert forbidden not in entry["env"]

    with pytest.raises(CursorConfigError):
        upsert_cursor_mcp_config("{broken json", entry)
    with pytest.raises(CursorConfigError):
        upsert_cursor_mcp_config("[]", entry)
    with pytest.raises(CursorConfigError):
        upsert_cursor_mcp_config('{"mcpServers": "bad"}', entry)

    cursor_dir = tmp_path / "cursor_cfg"
    dry = install_cursor(
        cursor_config_dir=cursor_dir,
        python_bin="/venv/bin/python",
        dry_run=True,
    )
    assert dry["status"] == "dry_run"
    assert not cursor_dir.exists()

    cursor_dir.mkdir()
    mcp_path = cursor_dir / "mcp.json"
    mcp_path.write_text(
        json.dumps(
            {
                "customSetting": True,
                "mcpServers": {"github": {"command": "npx", "args": ["-y", "srv-gh"]}},
            }
        ),
        encoding="utf-8",
    )
    mcp_path.chmod(0o600)

    first = install_cursor(
        name="agy",
        profile="balanced",
        auth_mode="gateway",
        base_url="https://gw.example.internal",
        max_daily_cents=500.0,
        cursor_config_dir=cursor_dir,
        python_bin="/venv/bin/python",
    )
    assert first["status"] == "installed"
    assert Path(first["backup_path"]).is_file()
    assert (Path(first["backup_path"]).stat().st_mode & 0o777) == 0o600
    assert (mcp_path.stat().st_mode & 0o777) == 0o600

    written_raw = mcp_path.read_text(encoding="utf-8")
    for secret_val in (
        "secret-gemini-key",
        "secret-google-key",
        "secret-oauth-token",
        "secret-gateway-token",
    ):
        assert secret_val not in written_raw
    written = json.loads(written_raw)
    assert written["customSetting"] is True
    assert written["mcpServers"]["github"]["command"] == "npx"
    assert written["mcpServers"]["agy"]["env"]["CC_AGY_COST_PROFILE"] == "balanced"
    assert written["mcpServers"]["agy"]["env"]["CC_AGY_AUTH_MODE"] == "gateway"
    assert written["mcpServers"]["agy"]["env"]["CC_AGY_VERTEX_BASE_URL"] == "https://gw.example.internal"
    assert written["mcpServers"]["agy"]["env"]["CC_AGY_MAX_DAILY_CENTS"] == "500"

    skill_file = cursor_dir / "skills" / "cc-agy-plugin" / "SKILL.md"
    assert skill_file.is_file()

    second = install_cursor(
        name="agy",
        profile="balanced",
        auth_mode="gateway",
        base_url="https://gw.example.internal",
        max_daily_cents=500.0,
        cursor_config_dir=cursor_dir,
        python_bin="/venv/bin/python",
    )
    assert second["status"] == "unchanged"

    # Re-running after skill deletion restores SKILL.md and marks status installed
    skill_file.unlink()
    repaired = install_cursor(
        name="agy",
        profile="balanced",
        auth_mode="gateway",
        base_url="https://gw.example.internal",
        max_daily_cents=500.0,
        cursor_config_dir=cursor_dir,
        python_bin="/venv/bin/python",
    )
    assert repaired["status"] == "installed"
    assert skill_file.is_file()

    # Refusal on corrupted existing mcp.json leaves broken file untouched
    mcp_path.write_text("{corrupt", encoding="utf-8")
    refused = install_cursor(cursor_config_dir=cursor_dir, python_bin="/venv/bin/python")
    assert refused["status"] == "refused"
    assert "not valid JSON" in refused["error"]


def test_install_cli_cursor_and_all_with_env_override_and_wizard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers_num = iter(["5", "3", "https://gw.example.internal", "2", "400"])
    built_num = run_interactive_wizard(input_fn=lambda _p: next(answers_num), output_fn=lambda _m: None, env={})
    assert built_num == [
        "--host",
        "cursor",
        "--profile",
        "economy",
        "--base-url",
        "https://gw.example.internal",
        "--max-daily-cents",
        "400",
    ]

    answers_word = iter(["cursor", "1", "3", ""])
    built_word = run_interactive_wizard(
        input_fn=lambda _p: next(answers_word),
        output_fn=lambda _m: None,
        env={"GEMINI_API_KEY": "present"},
    )
    assert built_word == ["--host", "cursor", "--profile", "quality"]

    codex_dir = tmp_path / "codex"
    cursor_dir = tmp_path / "cursor"
    agy_dir = tmp_path / "agy"
    monkeypatch.setenv("CODEX_HOME", str(codex_dir))
    monkeypatch.setenv("CURSOR_CONFIG_DIR", str(cursor_dir))
    monkeypatch.setenv("AGY_CONFIG_DIR", str(agy_dir))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-never-written")

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = install_main(
            [
                "--host",
                "cursor",
                "--profile",
                "balanced",
                "--base-url",
                "https://gw.example.internal",
                "--no-bootstrap",
                "--non-interactive",
            ]
        )
    out = buf.getvalue()
    assert "Cursor: installed ->" in out
    assert (cursor_dir / "mcp.json").is_file()
    assert (cursor_dir / "skills" / "cc-agy-plugin" / "SKILL.md").is_file()
    assert "test-key-never-written" not in (cursor_dir / "mcp.json").read_text(encoding="utf-8")
    # rc depends only on whether SDKs are installed in active env; verify JSON report structure directly
    buf_all = io.StringIO()
    with redirect_stdout(buf_all):
        install_main(
            [
                "--host",
                "all",
                "--profile",
                "economy",
                "--base-url",
                "https://gw.example.internal",
                "--no-bootstrap",
                "--non-interactive",
                "--json",
            ]
        )
    report_all = json.loads(buf_all.getvalue())
    host_names = [h["host"] for h in report_all["hosts"]]
    assert host_names == ["codex", "claude", "cursor", "agy"]
    assert rc in (0, 1)


def test_doctor_reports_cursor_codex_and_agy_host_configs(tmp_path: Path) -> None:
    cursor_dir = tmp_path / "cursor"
    codex_dir = tmp_path / "codex"
    agy_dir = tmp_path / "agy"
    env = {
        "CURSOR_CONFIG_DIR": str(cursor_dir),
        "CODEX_HOME": str(codex_dir),
        "AGY_CONFIG_DIR": str(agy_dir),
        "GEMINI_API_KEY": "dummy-key",
        "CC_AGY_DATA_DIR": str(tmp_path / "data"),
    }
    before = run_doctor(workspace_root=REPO_ROOT, env=env)
    assert before["mcp"]["host_configs"]["cursor"] == {
        "path": str(cursor_dir / "mcp.json"),
        "exists": False,
    }

    install_cursor(cursor_config_dir=cursor_dir, python_bin="/usr/bin/python3")
    after = run_doctor(workspace_root=REPO_ROOT, env=env)
    assert after["mcp"]["host_configs"]["cursor"] == {
        "path": str(cursor_dir / "mcp.json"),
        "exists": True,
    }
