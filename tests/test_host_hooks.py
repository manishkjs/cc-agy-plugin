"""Hermetic tests for Codex and Antigravity (agy) hook installs in `server/install.py`.

Codex 0.118 reads `$CODEX_HOME/hooks.json` (`{"hooks": {Event: [{matcher, hooks: [...]}]}}`)
only when `[features] codex_hooks = true`. agy reads a named map (`{"<hook name>": {Event:
[...]}}`) from `~/.gemini/config/hooks.json` and from each plugin root. Both run `command`
through a shell without substituting env vars, so the installer writes absolute paths.

No real host CLI runs here: every test uses a temporary HOME / CODEX_HOME.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest

from server import install

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = "/opt/py 3/bin/python3"  # A path with a space proves the command is shell-quoted.
OTHER_CODEX_HOOK = {"type": "command", "command": "/usr/local/bin/audit-bash", "timeout": 3}
OTHER_AGY_HOOK = {"PreToolUse": [{"matcher": "run_command", "hooks": [{"type": "command", "command": "./lint.sh"}]}]}


def _cc_timeouts() -> dict[str, int]:
    """Timeout of each event's handler in the Claude Code hooks/hooks.json (the source of truth)."""
    cc = json.loads((REPO_ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    return {event: groups[0]["hooks"][0]["timeout"] for event, groups in cc.items()}


def _commands(node: object) -> list[str]:
    if isinstance(node, dict):
        found = [node["command"]] if isinstance(node.get("command"), str) else []
        for value in node.values():
            found.extend(_commands(value))
        return found
    if isinstance(node, list):
        return [cmd for item in node for cmd in _commands(item)]
    return []


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("template", [install.CODEX_HOOKS_TEMPLATE, install.AGY_HOOKS_TEMPLATE], ids=["codex", "agy"])
def test_template_commands_point_at_the_adapter(template: Path) -> None:
    commands = _commands(_read(template))
    assert commands
    host = "codex" if template == install.CODEX_HOOKS_TEMPLATE else "agy"
    for command in commands:
        match = re.match(r"\A@PYTHON@ @PLUGIN_ROOT@/(\S+) --host (\S+) --event (\S+)\Z", command)
        assert match, command
        assert (REPO_ROOT / match.group(1)).resolve() == install.HOOK_ADAPTER.resolve()
        assert match.group(2) == host
    assert install.HOOK_ADAPTER.is_file() or pytest.skip("scripts/hook_adapter.py not landed yet")


def test_codex_template_events_matchers_and_timeouts() -> None:
    hooks = _read(install.CODEX_HOOKS_TEMPLATE)["hooks"]
    assert set(hooks) <= set(install.CODEX_HOOK_EVENTS)
    # PostToolUse is omitted on purpose: Codex only reports Bash, which the ledger matcher never selects.
    assert set(hooks) == set(_cc_timeouts()) - {"PostToolUse"}
    for event, groups in hooks.items():
        (group,) = groups
        if event in ("PreToolUse", "PostToolUse"):
            assert group["matcher"] == "^Bash$"  # Codex 0.118 only runs these for its shell tool.
        else:
            assert "matcher" not in group
        (handler,) = group["hooks"]
        assert handler["type"] == "command"
        assert handler["timeout"] == _cc_timeouts()[event]
        assert handler["command"].endswith(f"--event {event}")


def test_agy_template_is_a_named_map_with_anchored_matchers() -> None:
    data = _read(install.AGY_HOOKS_TEMPLATE)
    assert list(data) == [install.AGY_HOOKS_KEY]
    spec = data[install.AGY_HOOKS_KEY]
    assert set(spec) <= set(install.AGY_HOOK_EVENTS)
    timeouts = _cc_timeouts()
    cc_event = {"PreInvocation": "UserPromptSubmit"}
    for event, entries in spec.items():
        want = cc_event.get(event, event)
        if event in ("PreToolUse", "PostToolUse"):
            (group,) = entries
            # agy compiles each matcher as RE2 `\A(?:...)\z` (jsonhook.go MatchesToolName).
            anchored = re.compile(r"\A(?:" + group["matcher"] + r")\Z")
            handlers = group["hooks"]
        else:
            handlers = entries  # Non-tool events take a flat list of handlers.
            anchored = None
        (handler,) = handlers
        assert handler["timeout"] == timeouts[want]
        assert handler["command"].endswith(f"--event {want}")
        if event == "PreToolUse":
            for tool in ("run_command", "view_file", "write_to_file", "replace_file_content",
                         "multi_replace_file_content", "grep_search", "find_by_name", "call_mcp_tool",
                         "read_file", "write_file", "edit_file",
                         "mcp_agy_delegate_start"):
                assert anchored.match(tool), tool
            for tool in ("run_command_x", "xrun_command", "mcp_agy_list_jobs", "browser_subagent"):
                assert not anchored.match(tool), tool
        if event == "PostToolUse":
            for tool in ("call_mcp_tool", "mcp_agy_delegate_result", "mcp_other_delegate_apply"):
                assert anchored.match(tool), tool
            assert not anchored.match("run_command")


def test_render_quotes_python_and_adapter_path(tmp_path: Path) -> None:
    root = tmp_path / "with space"
    cmd = install.render_hook_command(
        "@PYTHON@ @PLUGIN_ROOT@/scripts/hook_adapter.py --host codex --event Stop", python_bin=PY, plugin_root=root
    )
    assert shlex.split(cmd) == [PY, str(root / "scripts" / "hook_adapter.py"), "--host", "codex", "--event", "Stop"]


# ---------------------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------------------


def test_codex_install_writes_hooks_and_feature_flag(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    home.mkdir()
    (home / "config.toml").write_text('model = "o3"\n\n[mcp_servers.agy]\ncommand = "py"\n', encoding="utf-8")
    result = install.install_codex_hooks(codex_home=home, python_bin=PY)
    assert result["status"] == "installed" and result["feature_flag"] == "enabled"
    assert sorted(p.name for p in home.iterdir() if ".bak-" not in p.name) == ["config.toml", "hooks.json"]
    cfg = install._parse_toml((home / "config.toml").read_text(encoding="utf-8"))
    assert cfg["features"]["codex_hooks"] is True
    assert cfg["model"] == "o3" and cfg["mcp_servers"]["agy"]["command"] == "py"
    hooks = _read(home / "hooks.json")["hooks"]
    assert set(hooks) == set(_read(install.CODEX_HOOKS_TEMPLATE)["hooks"])
    pre = hooks["PreToolUse"][0]
    assert pre["matcher"] == "^Bash$"
    assert shlex.split(pre["hooks"][0]["command"]) == [
        PY, str(install.HOOK_ADAPTER), "--host", "codex", "--event", "PreToolUse"]
    assert result["backup_paths"]  # config.toml existed and changed, so it was backed up.


def test_codex_reinstall_is_idempotent_and_preserves_other_hooks(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    home.mkdir()
    other = {"hooks": {"PreToolUse": [{"matcher": "^Bash$", "hooks": [OTHER_CODEX_HOOK]}]}, "note": "mine"}
    (home / "hooks.json").write_text(json.dumps(other), encoding="utf-8")
    (home / "config.toml").write_text("[features]\nweb_search = true\n", encoding="utf-8")
    install.install_codex_hooks(codex_home=home, python_bin=PY)
    first = (home / "hooks.json").read_text(encoding="utf-8")
    again = install.install_codex_hooks(codex_home=home, python_bin=PY)
    assert again["status"] == "unchanged" and again["feature_flag"] == "unchanged"
    assert (home / "hooks.json").read_text(encoding="utf-8") == first
    data = json.loads(first)
    assert data["note"] == "mine"
    pre_handlers = [h for g in data["hooks"]["PreToolUse"] for h in g["hooks"]]
    assert OTHER_CODEX_HOOK in pre_handlers
    assert sum(install.is_our_codex_hook(h) for h in pre_handlers) == 1
    cfg = install._parse_toml((home / "config.toml").read_text(encoding="utf-8"))
    assert cfg["features"] == {"web_search": True, "codex_hooks": True}


def test_codex_reinstall_from_another_checkout_replaces_ours(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    home.mkdir()
    install.install_codex_hooks(codex_home=home, python_bin="/old/python")
    install.install_codex_hooks(codex_home=home, python_bin=PY)
    commands = _commands(_read(home / "hooks.json"))
    assert len(commands) == len(_read(install.CODEX_HOOKS_TEMPLATE)["hooks"])
    assert all(shlex.split(c)[0] == PY for c in commands)


def test_codex_uninstall_removes_only_ours(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    home.mkdir()
    other = {"hooks": {"Stop": [{"hooks": [OTHER_CODEX_HOOK]}]}}
    (home / "hooks.json").write_text(json.dumps(other), encoding="utf-8")
    install.install_codex_hooks(codex_home=home, python_bin=PY)
    removed = install.install_codex_hooks(codex_home=home, uninstall=True)
    assert removed["status"] == "removed"
    assert _read(home / "hooks.json") == other
    assert install.install_codex_hooks(codex_home=home, uninstall=True)["status"] == "absent"
    # The feature flag stays on: other hooks may rely on it.
    assert "codex_hooks = true" in (home / "config.toml").read_text(encoding="utf-8")


def test_codex_dry_run_writes_nothing(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    result = install.install_codex_hooks(codex_home=home, python_bin=PY, dry_run=True)
    assert result["status"] == "dry_run" and result["would_change"] is True
    assert not home.exists()


def test_codex_refuses_invalid_hooks_json(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    home.mkdir()
    (home / "hooks.json").write_text("{not json", encoding="utf-8")
    result = install.install_codex_hooks(codex_home=home, python_bin=PY)
    assert result["status"] == "refused"
    assert (home / "hooks.json").read_text(encoding="utf-8") == "{not json"
    assert not (home / "config.toml").exists()


@pytest.mark.parametrize(
    ("before", "expect_features"),
    [
        ("", {"codex_hooks": True}),
        ('model = "x"', {"codex_hooks": True}),
        ("[features]\ncodex_hooks = false\nother = 1\n", {"codex_hooks": True, "other": 1}),
        ("[features]\n# comment\nother = 1\n[tui]\nx = 1\n", {"codex_hooks": True, "other": 1}),
        ("features.codex_hooks = false\n", {"codex_hooks": True}),
    ],
)
def test_enable_codex_hooks_feature(before: str, expect_features: dict) -> None:
    after = install.enable_codex_hooks_feature(before)
    parsed = install._parse_toml(after)
    assert parsed is not None and parsed["features"] == expect_features
    assert install.enable_codex_hooks_feature(after) == after


# ---------------------------------------------------------------------------------------
# agy (Antigravity CLI)
# ---------------------------------------------------------------------------------------


def test_agy_install_merges_named_key_and_preserves_others(tmp_path: Path) -> None:
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "hooks.json").write_text(json.dumps({"lint-checker": OTHER_AGY_HOOK}), encoding="utf-8")
    result = install.install_agy_hooks(agy_config_dir=cfg, python_bin=PY)
    assert result["status"] == "installed" and result["backup_paths"]
    data = _read(cfg / "hooks.json")
    assert data["lint-checker"] == OTHER_AGY_HOOK
    assert set(data[install.AGY_HOOKS_KEY]) == {"SessionStart", "PreInvocation", "PreToolUse", "PostToolUse", "Stop"}
    for command in _commands(data[install.AGY_HOOKS_KEY]):
        argv = shlex.split(command)
        assert argv[:4] == [PY, str(install.HOOK_ADAPTER), "--host", "agy"]
        assert Path(argv[1]).is_absolute()
    assert not (cfg / "plugins").exists()
    again = install.install_agy_hooks(agy_config_dir=cfg, python_bin=PY)
    assert again["status"] == "unchanged"


def test_agy_uninstall_removes_only_our_key(tmp_path: Path) -> None:
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "hooks.json").write_text(json.dumps({"lint-checker": OTHER_AGY_HOOK}), encoding="utf-8")
    install.install_agy_hooks(agy_config_dir=cfg, python_bin=PY)
    assert install.install_agy_hooks(agy_config_dir=cfg, uninstall=True)["status"] == "removed"
    assert _read(cfg / "hooks.json") == {"lint-checker": OTHER_AGY_HOOK}
    assert install.install_agy_hooks(agy_config_dir=cfg, uninstall=True)["status"] == "absent"


def test_agy_plugin_scope_moves_our_key_and_never_doubles(tmp_path: Path) -> None:
    cfg = tmp_path / "config"
    install.install_agy_hooks(agy_config_dir=cfg, python_bin=PY)
    result = install.install_agy_hooks(agy_config_dir=cfg, python_bin=PY, scope="plugin")
    assert result["status"] == "installed"
    plugin_hooks = cfg / "plugins" / install.PLUGIN_NAME / "hooks.json"
    assert list(_read(plugin_hooks)) == [install.AGY_HOOKS_KEY]
    assert install.AGY_HOOKS_KEY not in _read(cfg / "hooks.json")
    install.install_agy_hooks(agy_config_dir=cfg, uninstall=True)
    assert not plugin_hooks.exists()


def test_agy_dry_run_and_invalid_json(tmp_path: Path) -> None:
    cfg = tmp_path / "config"
    assert install.install_agy_hooks(agy_config_dir=cfg, python_bin=PY, dry_run=True)["status"] == "dry_run"
    assert not cfg.exists()
    cfg.mkdir()
    (cfg / "hooks.json").write_text("[1, 2]", encoding="utf-8")
    assert install.install_agy_hooks(agy_config_dir=cfg, python_bin=PY)["status"] == "refused"
    assert (cfg / "hooks.json").read_text(encoding="utf-8") == "[1, 2]"


# ---------------------------------------------------------------------------------------
# CLI: each host writes only its own hook file
# ---------------------------------------------------------------------------------------


def _env(home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PATH": os.path.dirname(sys.executable),
        "GEMINI_API_KEY": "host-hooks-test-dummy-0000",
        "CC_AGY_AUTO_BOOTSTRAP": "0",
        "CC_AGY_MODEL_CATALOG": "0",
        "CC_AGY_PYTHON_BIN": sys.executable,
        "PIP_NO_INDEX": "1",
    }


def _run(home: Path, *args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "server.install", "--auth", "api-key", "--no-bootstrap", "--non-interactive",
         "--json", *args],
        cwd=REPO_ROOT, env=_env(home), capture_output=True, text=True, timeout=180, check=False,
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        pytest.fail(f"install.py printed no JSON (exit {proc.returncode}):\n{proc.stderr[-2000:]}")


def test_cli_codex_install_touches_only_codex_hooks(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    report = _run(home, "--host", "codex")
    (host,) = report["hosts"]
    assert host["hooks"]["status"] == "installed"
    assert (home / ".codex" / "hooks.json").is_file()
    assert not (home / ".gemini").exists()
    removed = _run(home, "--host", "codex", "--uninstall-hooks")
    assert [h["status"] for h in removed["hooks"]] == ["removed"]
    assert _read(home / ".codex" / "hooks.json") == {"hooks": {}}


def test_cli_agy_install_touches_only_agy_hooks(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    report = _run(home, "--host", "antigravity")
    (host,) = report["hosts"]
    assert host["hooks"]["status"] == "installed"
    assert install.AGY_HOOKS_KEY in _read(home / ".gemini" / "config" / "hooks.json")
    assert not (home / ".codex").exists()


def test_cli_opt_outs_and_dry_run(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    report = _run(home, "--host", "codex", "--no-codex-hooks")
    assert "hooks" not in report["hosts"][0]
    assert not (home / ".codex" / "hooks.json").exists()
    report = _run(home, "--host", "antigravity", "--no-agy-hooks")
    assert "hooks" not in report["hosts"][0]
    assert not (home / ".gemini" / "config" / "hooks.json").exists()
    dry_home = tmp_path / "dry"
    dry_home.mkdir()
    report = _run(dry_home, "--host", "codex", "--dry-run")
    assert report["hosts"][0]["hooks"]["status"] == "dry_run"
    assert not (dry_home / ".codex").exists()


def test_codex_reinstall_drops_legacy_post_tool_use(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    home.mkdir()
    legacy_cmd = f"{PY} /old/root/scripts/hook_adapter.py --host codex --event PostToolUse"
    legacy = {"hooks": {"PostToolUse": [{"matcher": "^Bash$", "hooks": [{"type": "command", "command": legacy_cmd, "timeout": 10}]}]}}
    (home / "hooks.json").write_text(json.dumps(legacy), encoding="utf-8")
    install.install_codex_hooks(codex_home=home, python_bin=PY)
    hooks = _read(home / "hooks.json")["hooks"]
    assert not any("PostToolUse" in c for c in _commands({"hooks": hooks}))
