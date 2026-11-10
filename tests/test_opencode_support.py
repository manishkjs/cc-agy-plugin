"""Tests for first-class OpenCode (`opencode.ai`) support in cc-agy-plugin."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pytest

from server.doctor import run_doctor
from server.harness_validator import (
    parse_frontmatter,
    validate_opencode_bundle,
    validate_plugin_tree,
)
from server.install import (
    OPENCODE_SCHEMA_URL,
    OPENCODE_STARTUP_TIMEOUT_MS,
    OPENCODE_TOOL_TIMEOUT_MS,
    OpenCodeConfigError,
    _write_unique_backup_bytes,
    build_opencode_server_entry,
    install_opencode,
    main as install_main,
    resolve_opencode_config_dir,
    resolve_opencode_config_path,
    run_interactive_wizard,
    upsert_opencode_mcp_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_opencode_bundle_contracts_and_zero_em_dashes() -> None:
    bundle_dir = REPO_ROOT / ".opencode"
    skill_path = bundle_dir / "skills" / "cc-agy-plugin" / "SKILL.md"
    agent_path = bundle_dir / "agents" / "gemini-delegator.md"
    commands_dir = bundle_dir / "commands"

    assert skill_path.is_file()
    assert agent_path.is_file()
    for stem in ("delegate", "doctor", "jobs", "cost"):
        cmd_path = commands_dir / f"{stem}.md"
        assert cmd_path.is_file()
        meta, body = parse_frontmatter(cmd_path.read_text(encoding="utf-8"))
        assert meta.get("description")
        assert "name" not in meta
        assert "argument-hint" not in meta
        assert "agy_" in body

    skill_meta, skill_body = parse_frontmatter(skill_path.read_text(encoding="utf-8"))
    assert skill_meta["name"] == "cc-agy-plugin" == skill_path.parent.name
    assert 1 <= len(skill_meta["description"]) <= 1024
    assert "agy_delegate_start" in skill_body

    agent_meta, agent_body = parse_frontmatter(agent_path.read_text(encoding="utf-8"))
    assert agent_meta["mode"] == "subagent"
    assert "permissions" not in agent_meta
    assert "edit: deny" in agent_meta["permission"]
    assert "agy_delegate_start" in agent_body

    codex_skill = (REPO_ROOT / "plugins" / "cc-agy-plugin" / "skills" / "vertex" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "agy CLI" in codex_skill.splitlines()[2]
    assert "OpenCode" in codex_skill

    for rel in (
        ".opencode/skills/cc-agy-plugin/SKILL.md",
        ".opencode/agents/gemini-delegator.md",
        ".opencode/commands/delegate.md",
        ".opencode/commands/doctor.md",
        ".opencode/commands/jobs.md",
        ".opencode/commands/cost.md",
        "README.md",
        "skills/vertex/SKILL.md",
        "plugins/cc-agy-plugin/skills/vertex/SKILL.md",
        "commands/delegate.md",
        "commands/doctor.md",
        "commands/jobs.md",
        "commands/cost.md",
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "\u2014" not in text, f"Em-dash found in {rel}"

    assert validate_opencode_bundle(REPO_ROOT) == []
    assert validate_plugin_tree(REPO_ROOT) == []


def test_validate_opencode_bundle_rejects_known_schema_traps(tmp_path: Path) -> None:
    # Missing .opencode directory
    assert any("does not exist" in e for e in validate_opencode_bundle(tmp_path))

    bundle = tmp_path / ".opencode"
    (bundle / "skills" / "cc-agy-plugin").mkdir(parents=True)
    (bundle / "commands").mkdir(parents=True)
    (bundle / "agents").mkdir(parents=True)

    # Trap 1: skill name mismatch with parent folder + invalid uppercase name + missing trigger phrase
    (bundle / "skills" / "cc-agy-plugin" / "SKILL.md").write_text(
        "---\nname: Wrong_Name\ndescription: Short text without trigger.\n---\nBody\n",
        encoding="utf-8",
    )
    # Trap 2: command using Claude-only frontmatter keys (`name`, `argument-hint`) and `vertex` server
    for stem in ("delegate", "doctor", "jobs", "cost"):
        (bundle / "commands" / f"{stem}.md").write_text(
            f"---\nname: {stem}\ndescription: Desc\nargument-hint: <arg>\n---\nCall `doctor` on the `vertex` MCP server.\n",
            encoding="utf-8",
        )
    # Trap 3: agent using V2 plural `permissions:` and Claude-only `disallowedTools`
    (bundle / "agents" / "gemini-delegator.md").write_text(
        "---\ndescription: Subagent\nmode: subagent\npermissions:\n  edit: deny\ndisallowedTools: Edit\n---\nBody\n",
        encoding="utf-8",
    )

    errs = validate_opencode_bundle(tmp_path)
    assert any("must match ^[a-z0-9]+(-[a-z0-9]+)*$" in e for e in errs), errs
    assert any("must match directory name 'cc-agy-plugin'" in e for e in errs), errs
    assert any("lacks a trigger phrase" in e for e in errs), errs
    assert any("argument-hint" in e for e in errs), errs
    assert any("references Claude Code's 'vertex' MCP server" in e for e in errs), errs
    assert any("plural 'permissions'" in e for e in errs), errs
    assert any("disallowedTools" in e for e in errs), errs

    # Clean OpenCode bundle passes
    (bundle / "skills" / "cc-agy-plugin" / "SKILL.md").write_text(
        "---\nname: cc-agy-plugin\ndescription: Use when delegating tasks from OpenCode to Gemini.\n---\nBody\n",
        encoding="utf-8",
    )
    for stem in ("delegate", "doctor", "jobs", "cost"):
        (bundle / "commands" / f"{stem}.md").write_text(
            f"---\ndescription: Run {stem}\n---\nCall `agy_{stem}` on the `agy` MCP server.\n",
            encoding="utf-8",
        )
    (bundle / "agents" / "gemini-delegator.md").write_text(
        "---\ndescription: Subagent\nmode: subagent\npermission:\n  edit: deny\n---\nBody\n",
        encoding="utf-8",
    )
    assert validate_opencode_bundle(tmp_path) == []


def test_resolve_opencode_config_dir_and_path_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    # Default ~/.config/opencode
    assert resolve_opencode_config_dir(env={}) == fake_home / ".config" / "opencode"

    # XDG_CONFIG_HOME/opencode overrides ~/.config/opencode
    xdg = tmp_path / "xdg"
    assert resolve_opencode_config_dir(env={"XDG_CONFIG_HOME": str(xdg)}) == xdg / "opencode"

    # OPENCODE_CONFIG_DIR overrides XDG_CONFIG_HOME
    custom = tmp_path / "custom-oc"
    assert (
        resolve_opencode_config_dir(
            env={"XDG_CONFIG_HOME": str(xdg), "OPENCODE_CONFIG_DIR": str(custom)}
        )
        == custom
    )

    # Explicit dir overrides env
    explicit = tmp_path / "explicit-oc"
    assert (
        resolve_opencode_config_dir(
            env={"OPENCODE_CONFIG_DIR": str(custom)}, explicit_dir=explicit
        )
        == explicit
    )

    # Config file selection: opencode.jsonc when it exists alone; otherwise opencode.json
    custom.mkdir()
    assert resolve_opencode_config_path(custom) == custom / "opencode.json"
    (custom / "opencode.jsonc").write_text("{}\n", encoding="utf-8")
    assert resolve_opencode_config_path(custom) == custom / "opencode.jsonc"
    (custom / "opencode.json").write_text("{}\n", encoding="utf-8")
    assert resolve_opencode_config_path(custom) == custom / "opencode.json"


def test_upsert_opencode_mcp_config_v1_v2_and_error_guards() -> None:
    entry = build_opencode_server_entry(
        python_bin="/venv/bin/python",
        profile="balanced",
        auth_mode="gateway",
        base_url="https://gw.example.internal",
        max_daily_cents=500.0,
        no_auto_apply=True,
        strict=True,
    )
    assert entry == {
        "type": "local",
        "command": ["/venv/bin/python", str(REPO_ROOT / "server" / "main.py")],
        "environment": {
            "CC_AGY_COST_PROFILE": "balanced",
            "CC_AGY_AUTH_MODE": "gateway",
            "CC_AGY_VERTEX_BASE_URL": "https://gw.example.internal",
            "CC_AGY_MAX_DAILY_CENTS": "500",
            "CC_AGY_AUTO_APPLY_ON_PASS": "0",
            "CC_AGY_STRICT_MODE": "true",
        },
        "enabled": True,
        "timeout": OPENCODE_TOOL_TIMEOUT_MS,
    }

    # Empty file initializes $schema + V1 mcp.<name>
    fresh = json.loads(upsert_opencode_mcp_config("", entry, name="agy"))
    assert fresh["$schema"] == OPENCODE_SCHEMA_URL
    assert fresh["mcp"]["agy"] == entry

    # Existing V2 envelope (`mcp.servers`) stays in V2 format and preserves sibling servers
    v2_existing = json.dumps(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "mcp": {
                "timeout": {"startup": 10000, "execution": 60000},
                "servers": {
                    "github": {"type": "local", "command": ["npx", "gh-mcp"], "disabled": False}
                },
            },
        }
    )
    v2_updated = json.loads(upsert_opencode_mcp_config(v2_existing, entry, name="agy"))
    assert v2_updated["model"] == "anthropic/claude-sonnet-4-5"
    assert v2_updated["mcp"]["servers"]["github"]["command"] == ["npx", "gh-mcp"]
    assert v2_updated["mcp"]["servers"]["agy"] == {
        "type": "local",
        "command": entry["command"],
        "environment": entry["environment"],
        "disabled": False,
        "timeout": {
            "startup": OPENCODE_STARTUP_TIMEOUT_MS,
            "execution": OPENCODE_TOOL_TIMEOUT_MS,
        },
    }

    # Mixed V1 + stale nested `servers.agy` removes the duplicate nested key so OpenCode never warns
    mixed = json.dumps(
        {
            "mcp": {
                "other": {"type": "remote", "url": "https://mcp.example.com"},
                "servers": {"agy": {"type": "local", "command": ["old"]}},
            }
        }
    )
    mixed_updated = json.loads(upsert_opencode_mcp_config(mixed, entry, name="agy"))
    assert mixed_updated["mcp"]["other"]["url"] == "https://mcp.example.com"
    assert mixed_updated["mcp"]["agy"] == entry
    assert "servers" not in mixed_updated["mcp"]

    # Error guards: reserved names, JSONC comments, non-dict top-level, non-dict mcp
    for reserved in ("servers", "timeout"):
        with pytest.raises(OpenCodeConfigError, match="reserved"):
            upsert_opencode_mcp_config("{}", entry, name=reserved)
    with pytest.raises(OpenCodeConfigError, match="not strict JSON"):
        upsert_opencode_mcp_config("// user comment\n{\n  \"mcp\": {}\n}\n", entry, filename="opencode.jsonc")
    with pytest.raises(OpenCodeConfigError, match="JSON object"):
        upsert_opencode_mcp_config("[]", entry)
    with pytest.raises(OpenCodeConfigError, match="'mcp' field must be a JSON object"):
        upsert_opencode_mcp_config('{"mcp": "invalid"}', entry)


def test_install_opencode_lifecycle_idempotence_backups_jsonc_and_secret_hygiene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-google-key")
    monkeypatch.setenv("ANTIGRAVITY_OAUTH_TOKEN", "secret-oauth-token")
    monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL_TOKEN", "secret-gateway-token")
    monkeypatch.setenv("CC_AGY_TELEMETRY_TOKEN", "secret-telemetry-token")

    oc_dir = tmp_path / "opencode_cfg"
    dry = install_opencode(
        opencode_config_dir=oc_dir,
        python_bin="/venv/bin/python",
        dry_run=True,
    )
    assert dry["status"] == "dry_run"
    assert not oc_dir.exists()

    # Pre-existing opencode.json with user settings and another MCP server
    oc_dir.mkdir()
    cfg_path = oc_dir / "opencode.json"
    cfg_path.write_text(
        json.dumps(
            {
                "$schema": OPENCODE_SCHEMA_URL,
                "theme": "tokyonight",
                "mcp": {"context7": {"type": "remote", "url": "https://mcp.context7.com/mcp"}},
            }
        ),
        encoding="utf-8",
    )
    cfg_path.chmod(0o600)

    first = install_opencode(
        name="agy",
        profile="balanced",
        auth_mode="gemini_enterprise",
        agy_cli_path="/opt/bin/agy",
        max_daily_cents=450.0,
        allowed_projects="proj-ok",
        blocked_projects="proj-deny",
        no_auto_apply=True,
        opencode_config_dir=oc_dir,
        python_bin="/venv/bin/python",
    )
    assert first["status"] == "installed"
    assert Path(first["backup_path"]).is_file()
    assert (Path(first["backup_path"]).stat().st_mode & 0o777) == 0o600
    assert (cfg_path.stat().st_mode & 0o777) == 0o600

    raw_written = cfg_path.read_text(encoding="utf-8")
    for secret_val in (
        "secret-gemini-key",
        "secret-google-key",
        "secret-oauth-token",
        "secret-gateway-token",
        "secret-telemetry-token",
    ):
        assert secret_val not in raw_written

    parsed = json.loads(raw_written)
    assert parsed["theme"] == "tokyonight"
    assert parsed["mcp"]["context7"]["url"] == "https://mcp.context7.com/mcp"
    assert parsed["mcp"]["agy"]["type"] == "local"
    assert parsed["mcp"]["agy"]["enabled"] is True
    assert parsed["mcp"]["agy"]["timeout"] == OPENCODE_TOOL_TIMEOUT_MS
    assert parsed["mcp"]["agy"]["environment"]["CC_AGY_AUTH_MODE"] == "gemini_enterprise"
    assert parsed["mcp"]["agy"]["environment"]["CC_AGY_CLI_BIN"] == "/opt/bin/agy"
    assert parsed["mcp"]["agy"]["environment"]["CC_AGY_MAX_DAILY_CENTS"] == "450"
    assert parsed["mcp"]["agy"]["environment"]["CC_AGY_ALLOWED_PROJECTS"] == "proj-ok"
    assert parsed["mcp"]["agy"]["environment"]["CC_AGY_BLOCKED_PROJECTS"] == "proj-deny"
    assert parsed["mcp"]["agy"]["environment"]["CC_AGY_AUTO_APPLY_ON_PASS"] == "0"

    # Global skill, 4 commands, and subagent are all installed
    assert (oc_dir / "skills" / "cc-agy-plugin" / "SKILL.md").is_file()
    assert (oc_dir / "agents" / "gemini-delegator.md").is_file()
    for stem in ("delegate", "doctor", "jobs", "cost"):
        assert (oc_dir / "commands" / f"{stem}.md").is_file()

    # 2nd run is idempotent (`unchanged`)
    second = install_opencode(
        name="agy",
        profile="balanced",
        auth_mode="gemini_enterprise",
        agy_cli_path="/opt/bin/agy",
        max_daily_cents=450.0,
        allowed_projects="proj-ok",
        blocked_projects="proj-deny",
        no_auto_apply=True,
        opencode_config_dir=oc_dir,
        python_bin="/venv/bin/python",
    )
    assert second["status"] == "unchanged"

    # Deleting a command or agent self-heals on next install
    (oc_dir / "commands" / "doctor.md").unlink()
    repaired = install_opencode(
        name="agy",
        profile="balanced",
        auth_mode="gemini_enterprise",
        agy_cli_path="/opt/bin/agy",
        max_daily_cents=450.0,
        allowed_projects="proj-ok",
        blocked_projects="proj-deny",
        no_auto_apply=True,
        opencode_config_dir=oc_dir,
        python_bin="/venv/bin/python",
    )
    assert repaired["status"] == "installed"
    assert (oc_dir / "commands" / "doctor.md").is_file()

    # Sibling opencode.jsonc that overrides mcp.agy is refused rather than silently shadowed
    (oc_dir / "opencode.jsonc").write_text(
        json.dumps({"mcp": {"agy": {"type": "local", "command": ["stale"]}}}),
        encoding="utf-8",
    )
    conflict = install_opencode(opencode_config_dir=oc_dir, python_bin="/venv/bin/python")
    assert conflict["status"] == "refused"
    assert "opencode.jsonc" in conflict["error"]

    # Standalone strict-JSON opencode.jsonc (when opencode.json does not exist) updates in place
    jsonc_only_dir = tmp_path / "jsonc_only"
    jsonc_only_dir.mkdir()
    jsonc_file = jsonc_only_dir / "opencode.jsonc"
    jsonc_file.write_text('{"theme": "gruvbox"}\n', encoding="utf-8")
    res_jsonc = install_opencode(opencode_config_dir=jsonc_only_dir, python_bin="/venv/bin/python")
    assert res_jsonc["status"] == "installed"
    assert res_jsonc["config_path"] == str(jsonc_file)
    assert not (jsonc_only_dir / "opencode.json").exists()
    assert json.loads(jsonc_file.read_text(encoding="utf-8"))["mcp"]["agy"]["type"] == "local"

    # Standalone opencode.jsonc without conflicting mcp.agy keeps comments untouched and writes opencode.json
    commented_dir = tmp_path / "commented_jsonc"
    commented_dir.mkdir()
    commented_file = commented_dir / "opencode.jsonc"
    original_commented = '// keep my comment mentioning "mcp"\n{\n  "theme": "nord",\n}\n'
    commented_file.write_text(original_commented, encoding="utf-8")
    preserved_comment = install_opencode(opencode_config_dir=commented_dir, python_bin="/venv/bin/python")
    assert preserved_comment["status"] == "installed"
    assert preserved_comment["preserved_jsonc_path"] == str(commented_file)
    assert preserved_comment["config_path"] == str(commented_dir / "opencode.json")
    assert commented_file.read_text(encoding="utf-8") == original_commented
    assert json.loads((commented_dir / "opencode.json").read_text(encoding="utf-8"))["mcp"]["agy"]["type"] == "local"
    assert (commented_dir / "commands" / "doctor.md").is_file()

    # Standalone commented opencode.jsonc that ALREADY defines mcp.agy refuses so comments are not stripped
    commented_conflict_dir = tmp_path / "commented_conflict_jsonc"
    commented_conflict_dir.mkdir()
    commented_conflict_file = commented_conflict_dir / "opencode.jsonc"
    original_conflict = '// user mcp comment\n{\n  "mcp": {\n    "agy": {"type": "local", "command": ["custom"]},\n  }\n}\n'
    commented_conflict_file.write_text(original_conflict, encoding="utf-8")
    refused_comment = install_opencode(opencode_config_dir=commented_conflict_dir, python_bin="/venv/bin/python")
    assert refused_comment["status"] == "refused"
    assert commented_conflict_file.read_text(encoding="utf-8") == original_conflict
    assert not (commented_conflict_dir / "opencode.json").exists()

    # Custom `--name` rewrites tool prefixes in installed OpenCode commands, skill, and subagent
    custom_name_dir = tmp_path / "custom_name"
    res_custom = install_opencode(
        name="gemini-worker",
        opencode_config_dir=custom_name_dir,
        python_bin="/venv/bin/python",
    )
    assert res_custom["status"] == "installed"
    custom_doc_text = (custom_name_dir / "commands" / "doctor.md").read_text(encoding="utf-8")
    custom_agent_text = (custom_name_dir / "agents" / "gemini-delegator.md").read_text(encoding="utf-8")
    custom_skill_text = (custom_name_dir / "skills" / "cc-agy-plugin" / "SKILL.md").read_text(encoding="utf-8")
    assert "gemini-worker_doctor" in custom_doc_text
    assert "`gemini-worker` MCP server" in custom_doc_text
    assert "gemini-worker_delegate_start" in custom_agent_text
    assert "`gemini-worker_` prefix" in custom_agent_text
    assert "gemini-worker_delegate_file_batch" in custom_skill_text
    assert "`gemini-worker` MCP tools" in custom_skill_text
    assert "--host opencode --name gemini-worker" in custom_skill_text
    assert "standalone `agy` CLI" in custom_skill_text
    assert "`agy` MCP" not in custom_skill_text


def test_install_cli_opencode_wizard_and_doctor_host_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers_num = iter(["6", "3", "https://gw.example.internal", "1", "250"])
    built_num = run_interactive_wizard(
        input_fn=lambda _p: next(answers_num), output_fn=lambda _m: None, env={}
    )
    assert built_num == [
        "--host",
        "opencode",
        "--profile",
        "balanced",
        "--base-url",
        "https://gw.example.internal",
        "--max-daily-cents",
        "250",
    ]

    answers_word = iter(["opencode", "1", "3", ""])
    built_word = run_interactive_wizard(
        input_fn=lambda _p: next(answers_word),
        output_fn=lambda _m: None,
        env={"GEMINI_API_KEY": "present"},
    )
    assert built_word == ["--host", "opencode", "--profile", "quality"]

    oc_dir = tmp_path / "opencode"
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(oc_dir))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-never-written")

    doctor_env = {
        "OPENCODE_CONFIG_DIR": str(oc_dir),
        "CODEX_HOME": str(tmp_path / "codex"),
        "CURSOR_CONFIG_DIR": str(tmp_path / "cursor"),
        "AGY_CONFIG_DIR": str(tmp_path / "agy"),
        "GEMINI_API_KEY": "dummy-key",
        "CC_AGY_DATA_DIR": str(tmp_path / "data"),
    }
    before = run_doctor(workspace_root=REPO_ROOT, env=doctor_env)
    assert before["mcp"]["host_configs"]["opencode"] == {
        "path": str(oc_dir / "opencode.json"),
        "exists": False,
    }

    monkeypatch.setattr("server.install.python_has_plugin_sdks", lambda _py: True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = install_main(
            [
                "--host",
                "opencode",
                "--profile",
                "balanced",
                "--base-url",
                "https://gw.example.internal",
                "--no-bootstrap",
                "--non-interactive",
            ]
        )
    out = buf.getvalue()
    assert "OpenCode: installed ->" in out
    assert "[mcp.agy]" in out
    assert "/delegate, /doctor, /jobs, /cost" in out
    assert (oc_dir / "opencode.json").is_file()
    assert "test-key-never-written" not in (oc_dir / "opencode.json").read_text(encoding="utf-8")
    assert rc == 0

    after = run_doctor(workspace_root=REPO_ROOT, env=doctor_env)
    assert after["mcp"]["host_configs"]["opencode"] == {
        "path": str(oc_dir / "opencode.json"),
        "exists": True,
    }

    # Reserved --name exits with status 1
    buf_bad = io.StringIO()
    with redirect_stdout(buf_bad):
        rc_bad = install_main(
            [
                "--host",
                "opencode",
                "--name",
                "servers",
                "--base-url",
                "https://gw.example.internal",
                "--no-bootstrap",
                "--non-interactive",
                "--json",
            ]
        )
    assert rc_bad == 1
    assert json.loads(buf_bad.getvalue())["hosts"][0]["status"] == "refused"


def test_opencode_user_asset_backups_and_collision_safe_bak_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("server.install.time.strftime", lambda _fmt: "20261106-120000")
    oc_dir = tmp_path / "oc_user_assets"
    (oc_dir / "commands").mkdir(parents=True)
    (oc_dir / "agents").mkdir(parents=True)

    user_doctor_v1 = "---\ndescription: My personal OpenCode doctor command\n---\nRun npm run check.\n"
    user_agent_v1 = "---\ndescription: Custom local helper\nmode: subagent\n---\nCustom prompt.\n"
    (oc_dir / "commands" / "doctor.md").write_text(user_doctor_v1, encoding="utf-8")
    (oc_dir / "agents" / "gemini-delegator.md").write_text(user_agent_v1, encoding="utf-8")

    res1 = install_opencode(opencode_config_dir=oc_dir, python_bin="/venv/bin/python")
    assert res1["status"] == "installed"
    bak_cmd_1 = oc_dir / "commands" / "doctor.md.bak-20261106-120000"
    bak_agent_1 = oc_dir / "agents" / "gemini-delegator.md.bak-20261106-120000"
    assert bak_cmd_1.read_text(encoding="utf-8") == user_doctor_v1
    assert bak_agent_1.read_text(encoding="utf-8") == user_agent_v1
    assert (bak_cmd_1.stat().st_mode & 0o777) == 0o600
    assert (bak_agent_1.stat().st_mode & 0o777) == 0o600
    assert str(bak_cmd_1) in res1["asset_backups"]
    assert str(bak_agent_1) in res1["asset_backups"]

    # Re-running when plugin-managed files are already installed creates zero extra asset backups
    res_plugin_update = install_opencode(
        name="agy_prod",
        opencode_config_dir=oc_dir,
        python_bin="/venv/bin/python",
    )
    assert res_plugin_update["status"] == "installed"
    assert "asset_backups" not in res_plugin_update
    # Config backup within the same second uses collision-safe suffix
    cfg_bak_1 = oc_dir / "opencode.json.bak-20261106-120000"
    assert cfg_bak_1.is_file()

    # Second config change in the exact same second produces `-1` instead of clobbering `cfg_bak_1`
    res_plugin_update_2 = install_opencode(
        name="agy_stage",
        opencode_config_dir=oc_dir,
        python_bin="/venv/bin/python",
    )
    assert res_plugin_update_2["status"] == "installed"
    cfg_bak_2 = oc_dir / "opencode.json.bak-20261106-120000-1"
    assert cfg_bak_2.is_file()
    assert "agy_prod" in cfg_bak_2.read_text(encoding="utf-8")
    assert "agy_prod" not in cfg_bak_1.read_text(encoding="utf-8")

    # Replacing doctor.md again with a second user-authored command in the same second uses `.bak-...-1`
    user_doctor_v2 = "---\ndescription: Second user doctor\n---\nRun cargo check.\n"
    (oc_dir / "commands" / "doctor.md").write_text(user_doctor_v2, encoding="utf-8")
    res_user_v2 = install_opencode(
        name="agy_stage",
        opencode_config_dir=oc_dir,
        python_bin="/venv/bin/python",
    )
    bak_cmd_2 = oc_dir / "commands" / "doctor.md.bak-20261106-120000-1"
    assert bak_cmd_2.read_text(encoding="utf-8") == user_doctor_v2
    assert bak_cmd_1.read_text(encoding="utf-8") == user_doctor_v1
    assert str(bak_cmd_2) in res_user_v2["asset_backups"]


@pytest.mark.parametrize("bad_name", ["", "servers", "timeout", "bad name", "x: y", "a\nkey: val", "agy😀"])
def test_opencode_rejects_unsafe_or_non_ascii_server_names(tmp_path: Path, bad_name: str) -> None:
    entry = build_opencode_server_entry(python_bin="/venv/bin/python")
    with pytest.raises(OpenCodeConfigError):
        upsert_opencode_mcp_config("{}", entry, name=bad_name)
    res = install_opencode(name=bad_name, opencode_config_dir=tmp_path / "bad", python_bin="/venv/bin/python")
    assert res["status"] == "refused"


def test_opencode_single_pass_rewrite_avoids_agy_prefix_double_substitution(tmp_path: Path) -> None:
    oc_dir = tmp_path / "agy_prod_dir"
    res = install_opencode(name="agy_prod", opencode_config_dir=oc_dir, python_bin="/venv/bin/python")
    assert res["status"] == "installed"
    for md_path in [
        oc_dir / "skills" / "cc-agy-plugin" / "SKILL.md",
        oc_dir / "agents" / "gemini-delegator.md",
        *sorted((oc_dir / "commands").glob("*.md")),
    ]:
        content = md_path.read_text(encoding="utf-8")
        assert "agy_prod_prod" not in content, f"Double substitution in {md_path}"
        assert "agy_prod_" in content


def test_opencode_v1_v2_merge_preserves_user_fields_bom_jsonc_and_cli_key_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("server.install.python_has_plugin_sdks", lambda _py: True)
    entry_initial = build_opencode_server_entry(
        python_bin="/venv/bin/python",
        profile="balanced",
        max_daily_cents=500.0,
        no_auto_apply=True,
    )
    entry_refreshed = build_opencode_server_entry(
        python_bin="/venv2/bin/python",
        profile="quality",
    )

    # 1. V1 existing entry with enabled=False, custom timeout, cwd, and custom env var + UTF-8 BOM
    #    (non-V1 keys like `codemode` or remote `url`/`headers`/`oauth` are dropped so McpLocalConfig stays schema-valid)
    v1_doc = "\ufeff" + json.dumps(
        {
            "$schema": OPENCODE_SCHEMA_URL,
            "mcp": {
                "agy": {
                    "type": "local",
                    "command": ["/old/python", "/old/main.py"],
                    "enabled": False,
                    "timeout": 900000,
                    "cwd": "/workspace/custom",
                    "codemode": True,
                    "environment": {
                        "CUSTOM_USER_VAR": "keep-me",
                        "CC_AGY_MAX_DAILY_CENTS": "500",
                        "CC_AGY_AUTO_APPLY_ON_PASS": "0",
                    },
                }
            },
        }
    )
    v1_merged = json.loads(upsert_opencode_mcp_config(v1_doc, entry_refreshed, name="agy"))["mcp"]["agy"]
    assert v1_merged["command"][0] == "/venv2/bin/python"
    assert v1_merged["enabled"] is False
    assert v1_merged["timeout"] == 900000
    assert v1_merged["cwd"] == "/workspace/custom"
    assert "codemode" not in v1_merged
    assert v1_merged["environment"] == {
        "CUSTOM_USER_VAR": "keep-me",
        "CC_AGY_COST_PROFILE": "quality",
    }

    # Merging over an existing remote server named `agy` drops remote-only keys (`url`, `headers`, `oauth`)
    v1_remote_doc = json.dumps(
        {
            "$schema": OPENCODE_SCHEMA_URL,
            "mcp": {
                "agy": {
                    "type": "remote",
                    "url": "https://remote.example.com/mcp",
                    "headers": {"Authorization": "Bearer x"},
                    "oauth": {"clientId": "abc"},
                    "enabled": False,
                }
            },
        }
    )
    v1_from_remote = json.loads(upsert_opencode_mcp_config(v1_remote_doc, entry_refreshed, name="agy"))["mcp"]["agy"]
    assert v1_from_remote["type"] == "local"
    assert v1_from_remote["enabled"] is False
    assert set(v1_from_remote.keys()) <= {"type", "command", "cwd", "environment", "enabled", "timeout"}

    # 2. V2 timeout-only envelope (`{"mcp": {"timeout": {"startup": 15000}}}`) stays in V2 `mcp.servers`
    v2_timeout_only = json.dumps({"mcp": {"timeout": {"startup": 15000, "execution": 120000}}})
    v2_init = json.loads(upsert_opencode_mcp_config(v2_timeout_only, entry_initial, name="agy"))
    assert v2_init["mcp"]["timeout"] == {"startup": 15000, "execution": 120000}
    assert v2_init["mcp"]["servers"]["agy"]["disabled"] is False
    assert v2_init["mcp"]["servers"]["agy"]["environment"]["CC_AGY_MAX_DAILY_CENTS"] == "500"

    # Re-upserting V2 preserves `disabled: True`, `cwd`, `codemode: True`, `pool: True`, custom `timeout`, and user env vars
    v2_init["mcp"]["servers"]["agy"]["disabled"] = True
    v2_init["mcp"]["servers"]["agy"]["cwd"] = "/srv/repo"
    v2_init["mcp"]["servers"]["agy"]["codemode"] = True
    v2_init["mcp"]["servers"]["agy"]["pool"] = True
    v2_init["mcp"]["servers"]["agy"]["timeout"] = {"startup": 45000, "execution": 2400000}
    v2_init["mcp"]["servers"]["agy"]["environment"]["MY_CORP_PROXY"] = "http://proxy.internal"
    v2_remerged = json.loads(upsert_opencode_mcp_config(json.dumps(v2_init), entry_refreshed, name="agy"))
    srv_agy = v2_remerged["mcp"]["servers"]["agy"]
    assert srv_agy["disabled"] is True
    assert srv_agy["cwd"] == "/srv/repo"
    assert srv_agy["codemode"] is True
    assert srv_agy["pool"] is True
    assert srv_agy["timeout"] == {"startup": 45000, "execution": 2400000}
    assert srv_agy["environment"] == {
        "MY_CORP_PROXY": "http://proxy.internal",
        "CC_AGY_COST_PROFILE": "quality",
    }

    # V2 remote -> local merge also strips `url`, `headers`, and `oauth`
    v2_remote_doc = json.dumps(
        {
            "$schema": OPENCODE_SCHEMA_URL,
            "mcp": {
                "servers": {
                    "agy": {
                        "type": "remote",
                        "url": "https://remote.example.com/mcp",
                        "headers": {"X-Test": "1"},
                        "oauth": {"clientId": "abc"},
                        "disabled": True,
                    }
                }
            },
        }
    )
    v2_from_remote = json.loads(upsert_opencode_mcp_config(v2_remote_doc, entry_refreshed, name="agy"))["mcp"]["servers"]["agy"]
    assert v2_from_remote["type"] == "local"
    assert v2_from_remote["disabled"] is True
    assert set(v2_from_remote.keys()) <= {
        "type",
        "command",
        "cwd",
        "environment",
        "disabled",
        "codemode",
        "timeout",
        "protocol",
        "pool",
    }

    # 3. CLI prints `[mcp.servers.agy]` when updating a V2 config and tightens 0o644 -> 0o600
    oc_v2_dir = tmp_path / "oc_v2_cli"
    oc_v2_dir.mkdir()
    v2_cfg_path = oc_v2_dir / "opencode.json"
    v2_cfg_path.write_text(v2_timeout_only, encoding="utf-8")
    v2_cfg_path.chmod(0o644)
    # Sibling opencode.jsonc with `"mcp"` only inside a // comment + trailing comma does NOT false-positive refuse
    (oc_v2_dir / "opencode.jsonc").write_text(
        '// note: "mcp" servers live in opencode.json; "agy" is managed\n'
        '{\n  "mcp": {\n    "servers": {\n      "github": {"type": "local", "command": ["gh-mcp"],},\n    },\n  },\n}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(oc_v2_dir))
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyUnitTestKeyValue123456")
    buf_v2 = io.StringIO()
    with redirect_stdout(buf_v2):
        rc_v2 = install_main(["--host", "opencode", "--no-bootstrap", "--non-interactive"])
    assert rc_v2 == 0
    assert "[mcp.servers.agy]" in buf_v2.getvalue()
    assert (v2_cfg_path.stat().st_mode & 0o777) == 0o600


def test_doctor_env_home_isolation_opencode_config_env_url_secret_guard_and_missing_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_home = tmp_path / "isolated_home"
    (isolated_home / ".config" / "opencode").mkdir(parents=True)
    (isolated_home / ".config" / "opencode" / "opencode.json").write_text("{}\n", encoding="utf-8")

    # Passing `env={"HOME": ...}` without mutating os.environ isolates all host paths in run_doctor
    rep_home = run_doctor(
        workspace_root=REPO_ROOT,
        env={
            "HOME": str(isolated_home),
            "GEMINI_API_KEY": "k",
            "CC_AGY_DATA_DIR": str(tmp_path / "data1"),
        },
    )
    assert rep_home["mcp"]["host_configs"]["opencode"] == {
        "path": str(isolated_home / ".config" / "opencode" / "opencode.json"),
        "exists": True,
    }
    assert rep_home["mcp"]["host_configs"]["codex"]["path"] == str(isolated_home / ".codex" / "config.toml")
    assert rep_home["mcp"]["host_configs"]["cursor"]["path"] == str(isolated_home / ".cursor" / "mcp.json")

    # `OPENCODE_CONFIG` env var overrides config file path in resolve_opencode_config_path and run_doctor
    custom_cfg_file = tmp_path / "my-custom-opencode.json"
    custom_cfg_file.write_text("{}\n", encoding="utf-8")
    rep_cfg_env = run_doctor(
        workspace_root=REPO_ROOT,
        env={
            "HOME": str(isolated_home),
            "OPENCODE_CONFIG": str(custom_cfg_file),
            "GEMINI_API_KEY": "k",
            "CC_AGY_DATA_DIR": str(tmp_path / "data2"),
        },
    )
    assert rep_cfg_env["mcp"]["host_configs"]["opencode"] == {
        "path": str(custom_cfg_file),
        "exists": True,
    }

    # Gateway `--base-url` with embedded user:pass or `?api_key=` is rejected before writing any config
    for secret_url in (
        "https://user:supersecret@gw.example.internal/v1",
        "https://gw.example.internal/v1?api_key=AIzaSecret",
        "https://gw.example.internal/v1?token=BearerSecret",
    ):
        with pytest.raises(ValueError, match="must not embed"):
            build_opencode_server_entry(python_bin="/venv/bin/python", base_url=secret_url)

    # Missing `.opencode/` bundle asset returns status="failed" rather than falsely reporting "installed"
    monkeypatch.setattr("server.install.OPENCODE_SKILL_SOURCE", tmp_path / "nonexistent" / "SKILL.md")
    missing_res = install_opencode(opencode_config_dir=tmp_path / "missing_bundle", python_bin="/venv/bin/python")
    assert missing_res["status"] == "failed"
    assert "missing" in missing_res["error"]


def test_validate_opencode_bundle_catches_wildcard_permission_override_extra_agents_and_yaml_comments(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / ".opencode"
    (bundle / "skills" / "cc-agy-plugin").mkdir(parents=True)
    (bundle / "commands").mkdir(parents=True)
    (bundle / "agents").mkdir(parents=True)

    # Valid YAML block scalar (`>-`) + inline `#` comments pass cleanly
    (bundle / "skills" / "cc-agy-plugin" / "SKILL.md").write_text(
        "---\nname: cc-agy-plugin # inline comment\ndescription: >-\n  Use when delegating coding tasks from OpenCode to Gemini.\n---\nBody\n",
        encoding="utf-8",
    )
    for stem in ("delegate", "doctor", "jobs", "cost"):
        (bundle / "commands" / f"{stem}.md").write_text(
            f"---\ndescription: Run {stem} # comment\n---\nCall `agy_{stem}`.\n",
            encoding="utf-8",
        )
    (bundle / "agents" / "gemini-delegator.md").write_text(
        '---\ndescription: Subagent\nmode: subagent # inline comment\npermission:\n  "*": allow\n  edit: deny # last rule wins\n---\nBody\n',
        encoding="utf-8",
    )
    assert validate_opencode_bundle(tmp_path) == []

    # Reversing permission order (`edit: deny` followed by `"*": allow`) is rejected because last rule wins!
    (bundle / "agents" / "gemini-delegator.md").write_text(
        '---\ndescription: Subagent\nmode: subagent\npermission:\n  edit: deny\n  "*": allow\n---\nBody\n',
        encoding="utf-8",
    )
    perm_errs = validate_opencode_bundle(tmp_path)
    assert any("effective 'permission.edit' must evaluate to 'deny'" in e for e in perm_errs), perm_errs

    # Restore gemini-delegator.md, then add an invalid sibling `.opencode/agents/helper.md` + unsupported skill key
    (bundle / "agents" / "gemini-delegator.md").write_text(
        "---\ndescription: Subagent\nmode: subagent\npermission:\n  edit: deny\n---\nBody\n",
        encoding="utf-8",
    )
    (bundle / "agents" / "helper.md").write_text(
        "---\ndescription: Bad sibling\nmode: primary\npermission:\n  edit: allow\n---\nBody\n",
        encoding="utf-8",
    )
    (bundle / "skills" / "cc-agy-plugin" / "SKILL.md").write_text(
        "---\nname: cc-agy-plugin\ndescription: Use when delegating.\nallowed-tools: Bash\n---\nBody\n",
        encoding="utf-8",
    )
    sibling_errs = validate_opencode_bundle(tmp_path)
    assert any("helper.md: expected mode 'subagent'" in e for e in sibling_errs), sibling_errs
    assert any("allowed-tools" in e for e in sibling_errs), sibling_errs


def test_opencode_sha256_asset_manifest_backs_up_mcp_lookalikes_and_user_edits(tmp_path: Path) -> None:
    oc_dir = tmp_path / "oc_manifest"
    (oc_dir / "commands").mkdir(parents=True)

    # 1. User's own `doctor.md` calling `github_doctor` on the `github` MCP server AND user `jobs.md` mentioning `cc-agy-plugin`
    github_doctor = (
        "---\ndescription: Health check for my GitHub tools\n---\n"
        "Call `github_doctor` on the github MCP server and summarize.\n"
    )
    user_jobs = (
        "---\ndescription: My custom jobs helper\n---\n"
        "List my jobs (unrelated note mentioning cc-agy-plugin once).\n"
    )
    (oc_dir / "commands" / "doctor.md").write_text(github_doctor, encoding="utf-8")
    (oc_dir / "commands" / "jobs.md").write_text(user_jobs, encoding="utf-8")

    res1 = install_opencode(opencode_config_dir=oc_dir, python_bin="/venv/bin/python")
    assert res1["status"] == "installed"
    assert len(res1.get("asset_backups") or []) == 2
    doctor_baks = sorted((oc_dir / "commands").glob("doctor.md.bak-*"))
    jobs_baks = sorted((oc_dir / "commands").glob("jobs.md.bak-*"))
    assert len(doctor_baks) == 1 and doctor_baks[0].read_text(encoding="utf-8") == github_doctor
    assert len(jobs_baks) == 1 and jobs_baks[0].read_text(encoding="utf-8") == user_jobs
    assert ((oc_dir / ".cc-agy-plugin-assets.json").stat().st_mode & 0o777) == 0o600

    # 2. User hand-edits the installed `commands/doctor.md`, then re-runs installer -> user edit is backed up!
    installed_doctor = (oc_dir / "commands" / "doctor.md").read_text(encoding="utf-8")
    hand_edited = installed_doctor + "\nCustom user note added after install.\n"
    (oc_dir / "commands" / "doctor.md").write_text(hand_edited, encoding="utf-8")

    res2 = install_opencode(opencode_config_dir=oc_dir, python_bin="/venv/bin/python")
    assert res2["status"] == "installed"
    assert len(res2.get("asset_backups") or []) == 1
    doctor_baks_after = sorted((oc_dir / "commands").glob("doctor.md.bak-*"))
    assert len(doctor_baks_after) == 2
    assert any(p.read_text(encoding="utf-8") == hand_edited for p in doctor_baks_after)


def _init_minimal_opencode_bundle(root: Path) -> Path:
    bundle = root / ".opencode"
    (bundle / "skills" / "cc-agy-plugin").mkdir(parents=True, exist_ok=True)
    (bundle / "commands").mkdir(parents=True, exist_ok=True)
    (bundle / "agents").mkdir(parents=True, exist_ok=True)
    (bundle / "skills" / "cc-agy-plugin" / "SKILL.md").write_text(
        "---\nname: cc-agy-plugin\ndescription: Use when delegating tasks.\n---\nBody\n",
        encoding="utf-8",
    )
    for stem in ("delegate", "doctor", "jobs", "cost"):
        (bundle / "commands" / f"{stem}.md").write_text(
            f"---\ndescription: Run {stem}\n---\nCall `agy_{stem}`.\n",
            encoding="utf-8",
        )
    return bundle


def test_opencode_permission_evaluator_nested_patterns_and_flow_maps(tmp_path: Path) -> None:
    bundle = _init_minimal_opencode_bundle(tmp_path)
    agent_md = bundle / "agents" / "gemini-delegator.md"

    # Nested `bash: {"*": deny}` with NO `edit` rule must be flagged
    agent_md.write_text(
        '---\ndescription: Subagent\nmode: subagent\npermission:\n  bash:\n    "*": deny\n---\nBody\n',
        encoding="utf-8",
    )
    assert validate_opencode_bundle(tmp_path) != []

    # `edit: deny` + nested `bash: {"*": allow}` passes
    agent_md.write_text(
        '---\ndescription: Subagent\nmode: subagent\npermission:\n  edit: deny\n  bash:\n    "*": allow\n---\nBody\n',
        encoding="utf-8",
    )
    assert validate_opencode_bundle(tmp_path) == []

    # Nested `edit: {"*": deny, "*.py": allow}` is flagged because `*.py` permits edits
    agent_md.write_text(
        '---\ndescription: Subagent\nmode: subagent\npermission:\n  edit:\n    "*": deny\n    "*.py": allow\n---\nBody\n',
        encoding="utf-8",
    )
    assert validate_opencode_bundle(tmp_path) != []

    # Nested `edit: {"*": deny}` alone passes
    agent_md.write_text(
        '---\ndescription: Subagent\nmode: subagent\npermission:\n  edit:\n    "*": deny\n---\nBody\n',
        encoding="utf-8",
    )
    assert validate_opencode_bundle(tmp_path) == []

    # Nested wildcard `"*": {"*": allow}` AFTER `edit: deny` overrides `edit` and is flagged,
    # whereas `"*": {"*": allow}` BEFORE `edit: deny` leaves `edit` denied and passes
    agent_md.write_text(
        '---\ndescription: Subagent\nmode: subagent\npermission:\n  edit: deny\n  "*":\n    "*": allow\n---\nBody\n',
        encoding="utf-8",
    )
    assert validate_opencode_bundle(tmp_path) != []

    agent_md.write_text(
        '---\ndescription: Subagent\nmode: subagent\npermission:\n  "*":\n    "*": allow\n  edit: deny\n---\nBody\n',
        encoding="utf-8",
    )
    assert validate_opencode_bundle(tmp_path) == []

    # Inline flow-style `permission: {edit: deny}` passes while `{edit: deny, "*": allow}` is flagged
    agent_md.write_text(
        "---\ndescription: Subagent\nmode: subagent\npermission: {edit: deny}\n---\nBody\n",
        encoding="utf-8",
    )
    assert validate_opencode_bundle(tmp_path) == []

    agent_md.write_text(
        '---\ndescription: Subagent\nmode: subagent\npermission: {edit: deny, "*": allow}\n---\nBody\n',
        encoding="utf-8",
    )
    assert validate_opencode_bundle(tmp_path) != []


def test_gateway_base_url_secret_guard_accepts_benign_keys_and_blocks_secrets() -> None:
    for benign_url in (
        "https://gw.example.internal/v1?region=global",
        "https://gw.example.internal/v1?keyword=routing",
        "https://gw.example.internal/v1?author=alice&authority=corp",
        "https://gw.example.internal/v1?authuser=0",
        "https://gw.example.internal/v1?design=v2&signal=1&monkey=true&oauth_version=1.0",
    ):
        built = build_opencode_server_entry(python_bin="/venv/bin/python", base_url=benign_url)
        assert built["environment"]["CC_AGY_VERTEX_BASE_URL"] == benign_url

    for secret_url in (
        "https://gw.example.internal/v1?client_secret=x",
        "https://gw.example.internal/v1?refresh_token=x",
        "https://gw.example.internal/v1?access-token=x",
        "https://gw.example.internal/v1?x-api-key=x",
        "https://gw.example.internal/v1?apiKey=x",
        "https://gw.example.internal/v1?X-Goog-Signature=x",
        "https://gw.example.internal/v1?%74oken=x",
        "https://gw.example.internal/v1?region=us;token=x",
        "https://gw.example.internal/v1#token=x",
        "https://gw.example.internal/v1#apikey",
    ):
        with pytest.raises(ValueError, match="must not embed"):
            build_opencode_server_entry(python_bin="/venv/bin/python", base_url=secret_url)


def test_opencode_relative_config_path_refused_in_temp_cwd_and_warned_in_doctor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "rel_cfg_dir"))
    monkeypatch.setenv("OPENCODE_CONFIG", "rel-opencode.json")
    rel_res = install_opencode(python_bin="/venv/bin/python")
    assert rel_res["status"] == "refused"
    assert "absolute path" in rel_res["error"]
    assert not (tmp_path / "rel-opencode.json").exists()

    doc_rel = run_doctor(
        workspace_root=REPO_ROOT,
        env={
            "HOME": str(tmp_path / "rel_home"),
            "OPENCODE_CONFIG": "rel-opencode.json",
            "GEMINI_API_KEY": "k",
            "CC_AGY_DATA_DIR": str(tmp_path / "rel_data"),
        },
    )
    assert any("OPENCODE_CONFIG" in w and "relative path" in w for w in doc_rel["warnings"])


def test_opencode_v2_integer_timeout_coercion_protocol_filter_and_commented_jsonc_seeding(
    tmp_path: Path,
) -> None:
    entry = build_opencode_server_entry(python_bin="/venv/bin/python")

    # Legacy integer `timeout` inside V2 `mcp.servers.<name>` coerces to V2 `{startup, execution}` object,
    # supported `protocol` ("auto") is preserved on `agy`, while foreign `mcp.servers.other` keeps "2026-07-28" untouched
    v2_int_timeout_doc = json.dumps(
        {
            "$schema": OPENCODE_SCHEMA_URL,
            "mcp": {
                "servers": {
                    "agy": {
                        "type": "local",
                        "command": ["old"],
                        "timeout": 900000,
                        "protocol": "auto",
                    },
                    "other": {
                        "type": "local",
                        "command": ["other"],
                        "protocol": "2026-07-28",
                    },
                }
            },
        }
    )
    coerced_doc = json.loads(upsert_opencode_mcp_config(v2_int_timeout_doc, entry, name="agy"))
    coerced = coerced_doc["mcp"]["servers"]["agy"]
    assert coerced["timeout"] == {"startup": OPENCODE_STARTUP_TIMEOUT_MS, "execution": 900000}
    assert coerced["protocol"] == "auto"
    assert coerced_doc["mcp"]["servers"]["other"]["protocol"] == "2026-07-28"

    # Pre-existing unsupported `"protocol": "2026-07-28"` on `agy` itself is stripped on re-upsert so startup does not fail
    v2_unsupported_agy_proto = json.dumps(
        {
            "mcp": {
                "servers": {
                    "agy": {
                        "type": "local",
                        "command": ["old"],
                        "protocol": "2026-07-28",
                    }
                }
            }
        }
    )
    healed = json.loads(upsert_opencode_mcp_config(v2_unsupported_agy_proto, entry, name="agy"))["mcp"]["servers"]["agy"]
    assert "protocol" not in healed

    v2_bad_meta_doc = json.dumps(
        {
            "mcp": {
                "servers": {
                    "agy": {
                        "type": "local",
                        "command": ["old"],
                        "timeout": {"startup": -5, "bogus": 100},
                        "protocol": "stdio",
                    }
                }
            }
        }
    )
    sanitized = json.loads(upsert_opencode_mcp_config(v2_bad_meta_doc, entry, name="agy"))["mcp"]["servers"]["agy"]
    assert sanitized["timeout"] == {
        "startup": OPENCODE_STARTUP_TIMEOUT_MS,
        "execution": OPENCODE_TOOL_TIMEOUT_MS,
    }
    assert "protocol" not in sanitized

    # Standalone commented V2 `opencode.jsonc` initializes sibling `opencode.json` in V2 `mcp.servers`
    v2_jsonc_dir = tmp_path / "v2_commented_jsonc"
    v2_jsonc_dir.mkdir()
    (v2_jsonc_dir / "opencode.jsonc").write_text(
        '// V2 user config with comments\n{\n  "mcp": {\n    "servers": {\n      "github": {"type": "local", "command": ["gh-mcp"],},\n    },\n  },\n}\n',
        encoding="utf-8",
    )
    res_v2_sibling = install_opencode(opencode_config_dir=v2_jsonc_dir, python_bin="/venv/bin/python")
    assert res_v2_sibling["status"] == "installed"
    assert res_v2_sibling["mcp_key_path"] == "mcp.servers.agy"
    sibling_parsed = json.loads((v2_jsonc_dir / "opencode.json").read_text(encoding="utf-8"))
    assert "agy" in sibling_parsed["mcp"]["servers"]


def test_opencode_mode_tightening_and_config_symlink_preservation(tmp_path: Path) -> None:
    oc_dir = tmp_path / "tighten_dir"
    install_opencode(opencode_config_dir=oc_dir, python_bin="/venv/bin/python")
    (oc_dir / "opencode.json").chmod(0o644)

    res_tighten = install_opencode(opencode_config_dir=oc_dir, python_bin="/venv/bin/python")
    assert res_tighten["status"] == "unchanged"
    assert res_tighten["mode_tightened"] is True
    assert ((oc_dir / "opencode.json").stat().st_mode & 0o777) == 0o600

    sym_target = tmp_path / "dotfiles_opencode.json"
    sym_target.write_text((oc_dir / "opencode.json").read_text(encoding="utf-8"), encoding="utf-8")
    sym_target.chmod(0o644)
    (oc_dir / "opencode.json").unlink()
    (oc_dir / "opencode.json").symlink_to(sym_target)

    res_sym = install_opencode(opencode_config_dir=oc_dir, python_bin="/venv/bin/python")
    assert res_sym["status"] == "unchanged"
    assert (sym_target.stat().st_mode & 0o777) == 0o644


def test_write_unique_backup_bytes_o_excl_collision_and_dangling_symlink_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("server.install.time.strftime", lambda _fmt: "20261106-130000")
    target = tmp_path / "doctor.md"
    # Pre-occupy `.bak-20261106-130000` with a regular file and `.bak-20261106-130000-1` with a dangling symlink
    first_slot = tmp_path / "doctor.md.bak-20261106-130000"
    first_slot.write_bytes(b"slot-0")
    dangling_slot = tmp_path / "doctor.md.bak-20261106-130000-1"
    dangling_slot.symlink_to(tmp_path / "missing-dotfiles-target.md")

    created = _write_unique_backup_bytes(target, b"raw-\xe9\r\n", 0o600)
    assert created == tmp_path / "doctor.md.bak-20261106-130000-2"
    assert created.read_bytes() == b"raw-\xe9\r\n"
    assert (created.stat().st_mode & 0o777) == 0o600
    assert first_slot.read_bytes() == b"slot-0"
    assert dangling_slot.is_symlink() and not (tmp_path / "missing-dotfiles-target.md").exists()


def test_opencode_raw_byte_backups_preserve_crlf_latin1_symlinks_corrupt_manifest_and_partial_io_failure(
    tmp_path: Path,
) -> None:
    # 1. CRLF + Latin-1 (`0xe9`) user command + symlinked `cost.md` into dotfiles + corrupt manifest
    dotfiles_dir = tmp_path / "dotfiles"
    dotfiles_dir.mkdir()
    dotfiles_cost = dotfiles_dir / "cost.md"
    dotfiles_cost_bytes = b"---\r\ndescription: Dotfiles cost caf\xe9\r\n---\r\nDotfiles body.\r\n"
    dotfiles_cost.write_bytes(dotfiles_cost_bytes)

    oc_dir = tmp_path / "oc_raw_bytes"
    (oc_dir / "commands").mkdir(parents=True)
    raw_latin1_doctor = b"---\r\ndescription: Local doctor caf\xe9\r\n---\r\nRun check.\r\n"
    (oc_dir / "commands" / "doctor.md").write_bytes(raw_latin1_doctor)
    (oc_dir / "commands" / "cost.md").symlink_to(dotfiles_cost)
    (oc_dir / ".cc-agy-plugin-assets.json").write_text("{corrupt-json", encoding="utf-8")

    # Make `agents` a regular file so `_sync_opencode_assets` fails mid-way AFTER backing up `commands/*`
    (oc_dir / "agents").write_text("blocker-not-a-directory", encoding="utf-8")
    partial = install_opencode(opencode_config_dir=oc_dir, python_bin="/venv/bin/python")
    assert partial["status"] == "failed"
    assert "gemini-delegator.md" in partial["error"]
    assert len(partial.get("asset_backups") or []) == 2

    # Both raw backups preserved CRLF + Latin-1 bytes verbatim, and `dotfiles/cost.md` was NOT overwritten!
    doctor_baks = sorted((oc_dir / "commands").glob("doctor.md.bak-*"))
    cost_baks = sorted((oc_dir / "commands").glob("cost.md.bak-*"))
    assert len(doctor_baks) == 1 and doctor_baks[0].read_bytes() == raw_latin1_doctor
    assert len(cost_baks) == 1 and cost_baks[0].read_bytes() == dotfiles_cost_bytes
    assert dotfiles_cost.read_bytes() == dotfiles_cost_bytes
    assert not (oc_dir / "commands" / "cost.md").is_symlink()

    # Fixing the blocker and retrying finishes cleanly with zero duplicate backups
    (oc_dir / "agents").unlink()
    recovered = install_opencode(opencode_config_dir=oc_dir, python_bin="/venv/bin/python")
    assert recovered["status"] == "installed"
    assert "asset_backups" not in recovered
    assert len(list((oc_dir / "commands").glob("doctor.md.bak-*"))) == 1
    assert len(list((oc_dir / "commands").glob("cost.md.bak-*"))) == 1
    assert (oc_dir / "agents" / "gemini-delegator.md").is_file()

    # 2. Corrupt non-UTF-8 `opencode.json` returns `status="failed"` naming `opencode.json`
    bad_utf8_dir = tmp_path / "bad_utf8_dir"
    bad_utf8_dir.mkdir()
    (bad_utf8_dir / "opencode.json").write_bytes(b"{\xff\xfe invalid utf8")
    res_bad_utf8 = install_opencode(opencode_config_dir=bad_utf8_dir, python_bin="/venv/bin/python")
    assert res_bad_utf8["status"] == "failed"
    assert "opencode.json" in res_bad_utf8["error"]


def test_opencode_r4_timeout_bounds_partial_manifest_save_crlf_config_backup_and_secret_mutants(
    tmp_path: Path,
) -> None:
    entry = build_opencode_server_entry(python_bin="/venv/bin/python")
    max_safe = 9_007_199_254_740_991

    # 1. V1 and V2 reject <= 0, bool, and > Number.MAX_SAFE_INTEGER timeouts while keeping MAX_SAFE_INTEGER
    for bad_timeout in (0, -5, True, False, 2**60, max_safe + 1):
        v1_bad = json.dumps({"mcp": {"agy": {"type": "local", "command": ["old"], "timeout": bad_timeout}}})
        v1_res = json.loads(upsert_opencode_mcp_config(v1_bad, entry, name="agy"))["mcp"]["agy"]
        assert v1_res["timeout"] == OPENCODE_TOOL_TIMEOUT_MS

        v2_bad_int = json.dumps(
            {"mcp": {"servers": {"agy": {"type": "local", "command": ["old"], "timeout": bad_timeout}}}}
        )
        v2_res_int = json.loads(upsert_opencode_mcp_config(v2_bad_int, entry, name="agy"))["mcp"]["servers"]["agy"]
        assert v2_res_int["timeout"] == {
            "startup": OPENCODE_STARTUP_TIMEOUT_MS,
            "execution": OPENCODE_TOOL_TIMEOUT_MS,
        }

        v2_bad_map = json.dumps(
            {
                "mcp": {
                    "servers": {
                        "agy": {
                            "type": "local",
                            "command": ["old"],
                            "timeout": {"startup": bad_timeout, "execution": bad_timeout},
                        }
                    }
                }
            }
        )
        v2_res_map = json.loads(upsert_opencode_mcp_config(v2_bad_map, entry, name="agy"))["mcp"]["servers"]["agy"]
        assert v2_res_map["timeout"] == {
            "startup": OPENCODE_STARTUP_TIMEOUT_MS,
            "execution": OPENCODE_TOOL_TIMEOUT_MS,
        }

    v1_max = json.dumps({"mcp": {"agy": {"type": "local", "command": ["old"], "timeout": max_safe}}})
    assert json.loads(upsert_opencode_mcp_config(v1_max, entry, name="agy"))["mcp"]["agy"]["timeout"] == max_safe

    # 2. Partial failure persists manifest hashes so retrying with a different `--name` does NOT treat
    # plugin-written files as user edits and create spurious `.bak-*` backups (kills M11 and M12)
    m11_dir = tmp_path / "m11_partial"
    m11_dir.mkdir()
    (m11_dir / "agents").write_text("file-blocking-agents-mkdir", encoding="utf-8")
    first_try = install_opencode(opencode_config_dir=m11_dir, name="agy", python_bin="/venv/bin/python")
    assert first_try["status"] == "failed"
    assert (m11_dir / ".cc-agy-plugin-assets.json").is_file()
    (m11_dir / "agents").unlink()
    second_try = install_opencode(opencode_config_dir=m11_dir, name="customgw", python_bin="/venv/bin/python")
    assert second_try["status"] == "installed"
    assert "asset_backups" not in second_try
    assert list((m11_dir / "commands").glob("*.bak-*")) == []
    assert list((m11_dir / "skills" / "cc-agy-plugin").glob("*.bak-*")) == []

    # 3. CRLF `opencode.json` backup preserves exact `\r\n` bytes (`P3-2`)
    crlf_dir = tmp_path / "crlf_cfg"
    crlf_dir.mkdir()
    raw_crlf_cfg = b'{\r\n  "theme": "tokyonight"\r\n}\r\n'
    (crlf_dir / "opencode.json").write_bytes(raw_crlf_cfg)
    crlf_res = install_opencode(opencode_config_dir=crlf_dir, python_bin="/venv/bin/python")
    assert crlf_res["status"] == "installed"
    assert Path(crlf_res["backup_path"]).read_bytes() == raw_crlf_cfg

    # 4. Concatenated / acronym / digit-suffixed secret query keys are blocked (kills M3d & M3e),
    # while benign substrings (`bypass`, `compass`, `sidecar`) are accepted
    for benign_url in (
        "https://gw.example.internal/v1?bypass=1&compass=north&sidecar=true",
        "https://gw.example.internal/v1?session_timeout=30&surpass=95",
    ):
        built = build_opencode_server_entry(python_bin="/venv/bin/python", base_url=benign_url)
        assert built["environment"]["CC_AGY_VERTEX_BASE_URL"] == benign_url

    for secret_url in (
        "https://gw.example.internal/v1?MyAPIToken=x",
        "https://gw.example.internal/v1?userToken=x",
        "https://gw.example.internal/v1?accesstoken=x",
        "https://gw.example.internal/v1?apitoken=x",
        "https://gw.example.internal/v1?virtualkey=x",
        "https://gw.example.internal/v1?apiKey2=x",
        "https://gw.example.internal/v1?sigv4=x",
        "https://gw.example.internal/v1?jwt=x",
        "https://gw.example.internal/v1?pass=x",
        "https://gw.example.internal/v1?pw=x",
        "https://gw.example.internal/v1?sid=x",
        "https://gw.example.internal/v1?assertion=x",
    ):
        with pytest.raises(ValueError, match="must not embed"):
            build_opencode_server_entry(python_bin="/venv/bin/python", base_url=secret_url)
