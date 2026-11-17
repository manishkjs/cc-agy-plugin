"""Tests for macOS Seatbelt keychain and `.gemini/config/projects` access in `server.worker_sandbox`."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from server import worker_sandbox
from server.worker_sandbox import build_bwrap_prefix, build_seatbelt_profile, plan_worker_sandbox

_FILTER_RE = re.compile(r'\((subpath|literal)\s+"((?:\\.|[^"\\])*)"\)')
_OP_RE = re.compile(r"^\((allow|deny)\s+([^(]+?)(?:\s+\(|\))")


def _unescape_sbpl(raw: str) -> str:
    return raw.replace('\\"', '"').replace("\\\\", "\\")


def _op_matches(rule_op: str, query_op: str) -> bool:
    if rule_op == "default":
        return True
    if rule_op == query_op:
        return True
    if rule_op == "file-read*" and query_op in ("file-read-data", "file-read-metadata"):
        return True
    return False


def _filter_matches(kind: str, pattern: str, target: str) -> bool:
    if kind == "literal":
        return target == pattern
    if kind == "subpath":
        return target == pattern or target.startswith(pattern.rstrip("/") + "/")
    return False


def _last_matching_rule_for_path(profile: str, target: Path | str, operation: str) -> tuple[int, str] | None:
    """Evaluate SBPL last-matching-rule-wins for `target` and `operation`."""
    target_str = str(target)
    matched: tuple[int, str] | None = None
    for idx, line in enumerate(profile.splitlines()):
        op_match = _OP_RE.match(line)
        if not op_match:
            continue
        rule_ops = op_match.group(2).split()
        if not any(_op_matches(rule_op, operation) for rule_op in rule_ops):
            continue
        filters = _FILTER_RE.findall(line)
        if not filters:
            matched = (idx, line)
            continue
        for kind, raw_path in filters:
            if _filter_matches(kind, _unescape_sbpl(raw_path), target_str):
                matched = (idx, line)
                break
    return matched


def _last_matching_rule(profile: str, token: str, operation: str) -> str | None:
    """Return the last SBPL line matching `operation` (or `default`) for `token`."""
    matched: str | None = None
    for line in profile.splitlines():
        if line == "(allow default)":
            matched = line
        elif line == f"(deny {operation})":
            matched = line
        elif operation in line and token in line:
            matched = line
    return matched


@pytest.mark.parametrize("read_only_workspace", [False, True])
def test_seatbelt_allows_macos_keychain_read_and_write(read_only_workspace: bool) -> None:
    home = Path("/Users/dev")
    workspace = home / ".cache" / "cc-agy-plugin" / "abc" / "worktrees" / "job-1"
    config_dir = home / ".gemini" / "antigravity-cli"

    profile = build_seatbelt_profile(
        workspace=workspace,
        config_dir=config_dir,
        home=home,
        adc_path=None,
        read_only_workspace=read_only_workspace,
    )
    lines = profile.splitlines()
    keychain_token = '(subpath "/Users/dev/Library/Keychains")'

    deny_lines = [line for line in lines if line.startswith("(deny ")]
    assert all("Library/Keychains" not in line for line in deny_lines)

    last_read = _last_matching_rule(profile, keychain_token, "file-read*")
    last_write = _last_matching_rule(profile, keychain_token, "file-write*")
    assert last_read is not None and last_read.startswith("(allow ")
    assert last_write is not None and last_write.startswith("(allow ")
    assert "file-read*" in last_read and "file-write*" in last_write

    hidden_deny_idx = next(idx for idx, line in enumerate(lines) if line.startswith("(deny file-read* file-write* "))
    allow_idx = lines.index(last_write)
    assert allow_idx > hidden_deny_idx


def test_seatbelt_still_denies_other_secret_stores_and_caches() -> None:
    home = Path("/Users/dev")
    workspace = home / ".cache" / "cc-agy-plugin" / "abc" / "worktrees" / "job-1"
    config_dir = home / ".gemini" / "antigravity-cli"

    profile = build_seatbelt_profile(
        workspace=workspace,
        config_dir=config_dir,
        home=home,
        adc_path=None,
    )

    for rel_dir in (".ssh", ".aws", ".gnupg", ".config/gcloud", ".cache", "Library/Caches"):
        token = f'(subpath "{home / rel_dir}")'
        last_read = _last_matching_rule(profile, token, "file-read*")
        last_write = _last_matching_rule(profile, token, "file-write*")
        assert last_read is not None and last_read.startswith("(deny "), rel_dir
        assert last_write is not None and last_write.startswith("(deny "), rel_dir

    for rel_file in (".netrc", ".git-credentials"):
        token = f'(literal "{home / rel_file}")'
        last_read = _last_matching_rule(profile, token, "file-read*")
        last_write = _last_matching_rule(profile, token, "file-write*")
        assert last_read is not None and last_read.startswith("(deny "), rel_file
        assert last_write is not None and last_write.startswith("(deny "), rel_file


def test_seatbelt_allows_keychain_when_home_is_symlink(tmp_path: Path) -> None:
    real_home = tmp_path / "real_home"
    (real_home / "Library" / "Keychains").mkdir(parents=True)
    (real_home / ".ssh").mkdir(parents=True)
    (real_home / ".aws").mkdir(parents=True)
    (real_home / ".netrc").write_text("machine example.com password secret\n", encoding="utf-8")

    sym_home = tmp_path / "sym_home"
    sym_home.symlink_to(real_home)

    workspace = tmp_path / "worktrees" / "job-1"
    workspace.mkdir(parents=True)
    config_dir = real_home / ".gemini" / "antigravity-cli"
    config_dir.mkdir(parents=True)

    profile = build_seatbelt_profile(
        workspace=workspace,
        config_dir=config_dir,
        home=sym_home,
        adc_path=None,
    )
    lines = profile.splitlines()
    deny_lines = [line for line in lines if line.startswith("(deny ")]
    assert all("Library/Keychains" not in line for line in deny_lines)

    resolved_real = real_home.resolve()
    for keychain_path in (sym_home / "Library" / "Keychains", resolved_real / "Library" / "Keychains"):
        token = f'(subpath "{keychain_path}")'
        last_read = _last_matching_rule(profile, token, "file-read*")
        last_write = _last_matching_rule(profile, token, "file-write*")
        assert last_read is not None and last_read.startswith("(allow "), keychain_path
        assert last_write is not None and last_write.startswith("(allow "), keychain_path

    for secret_base in (sym_home, resolved_real):
        for rel_dir in (".ssh", ".aws"):
            token = f'(subpath "{secret_base / rel_dir}")'
            last_read = _last_matching_rule(profile, token, "file-read*")
            last_write = _last_matching_rule(profile, token, "file-write*")
            assert last_read is not None and last_read.startswith("(deny "), rel_dir
            assert last_write is not None and last_write.startswith("(deny "), rel_dir
        netrc_token = f'(literal "{secret_base / ".netrc"}")'
        last_netrc_read = _last_matching_rule(profile, netrc_token, "file-read*")
        assert last_netrc_read is not None and last_netrc_read.startswith("(deny ")


def test_seatbelt_allows_symlinked_keychain_directory(tmp_path: Path) -> None:
    home = tmp_path / "home"
    external_keychains = tmp_path / "external_keychains"
    external_keychains.mkdir(parents=True)
    (home / "Library").mkdir(parents=True)
    (home / "Library" / "Keychains").symlink_to(external_keychains)

    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    config_dir = home / ".gemini" / "antigravity-cli"
    config_dir.mkdir(parents=True)

    profile = build_seatbelt_profile(
        workspace=workspace,
        config_dir=config_dir,
        home=home,
        adc_path=None,
    )
    deny_lines = [line for line in profile.splitlines() if line.startswith("(deny ")]
    assert all("Library/Keychains" not in line for line in deny_lines)
    assert all(str(external_keychains.resolve()) not in line for line in deny_lines)

    for path in (home / "Library" / "Keychains", external_keychains.resolve()):
        token = f'(subpath "{path}")'
        last_read = _last_matching_rule(profile, token, "file-read*")
        last_write = _last_matching_rule(profile, token, "file-write*")
        assert last_read is not None and last_read.startswith("(allow "), path
        assert last_write is not None and last_write.startswith("(allow "), path


@pytest.mark.parametrize("read_only_workspace", [False, True])
def test_seatbelt_allows_gemini_projects_while_protecting_config_and_outside_of_project(
    read_only_workspace: bool,
) -> None:
    home = Path("/Users/dev")
    workspace = home / ".cache" / "cc-agy-plugin" / "abc" / "worktrees" / "job-1"
    config_dir = home / ".gemini" / "antigravity-cli"
    gemini_cfg = home / ".gemini" / "config"
    projects_dir = gemini_cfg / "projects"

    profile = build_seatbelt_profile(
        workspace=workspace,
        config_dir=config_dir,
        home=home,
        adc_path=None,
        read_only_workspace=read_only_workspace,
    )
    lines = profile.splitlines()

    # Exact ordering: hidden `.gemini/config` deny -> metadata allow -> projects allow -> outside-of-project deny
    cfg_deny_idx = next(
        idx
        for idx, line in enumerate(lines)
        if line.startswith("(deny file-read* file-write* ") and f'(subpath "{gemini_cfg}")' in line
    )
    meta_match = _last_matching_rule_for_path(profile, gemini_cfg, "file-read-metadata")
    projects_read = _last_matching_rule_for_path(profile, projects_dir / "default-cli-project.json", "file-read-data")
    projects_write = _last_matching_rule_for_path(profile, projects_dir / "default-cli-project.json", "file-write*")
    outside_read = _last_matching_rule_for_path(profile, projects_dir / "outside-of-project.json", "file-read-data")
    outside_write = _last_matching_rule_for_path(profile, projects_dir / "outside-of-project.json", "file-write*")

    assert meta_match is not None and meta_match[1].startswith("(allow file-read-metadata ")
    assert meta_match[0] > cfg_deny_idx
    assert projects_read is not None and projects_read[1].startswith("(allow file-read* file-write* ")
    assert projects_write is not None and projects_write[1].startswith("(allow file-read* file-write* ")
    assert projects_read[0] > cfg_deny_idx and projects_write[0] > cfg_deny_idx
    assert outside_read is not None and outside_read[1].startswith("(allow file-read* file-write* ")
    assert outside_write is not None and outside_write[1].startswith("(deny file-write* ")
    assert outside_write[0] > projects_write[0]

    # `~/.gemini/config` itself and its sensitive siblings remain denied for data reads and writes.
    cfg_read_data = _last_matching_rule_for_path(profile, gemini_cfg, "file-read-data")
    cfg_write = _last_matching_rule_for_path(profile, gemini_cfg, "file-write*")
    assert cfg_read_data is not None and cfg_read_data[1].startswith("(deny ")
    assert cfg_write is not None and cfg_write[1].startswith("(deny ")

    for protected_child in (
        gemini_cfg / "config.json",
        gemini_cfg / "mcp_config.json",
        gemini_cfg / "plugins" / "my-plugin" / "plugin.json",
        gemini_cfg / "skills" / "secret-skill" / "SKILL.md",
    ):
        for op in ("file-read-data", "file-read-metadata", "file-write*"):
            res = _last_matching_rule_for_path(profile, protected_child, op)
            assert res is not None and res[1].startswith("(deny "), (protected_child, op, res)


def test_seatbelt_projects_covers_symlinked_home_and_symlinked_projects_dir(tmp_path: Path) -> None:
    real_home = tmp_path / "real_home"
    external_projects = tmp_path / "external_projects"
    external_projects.mkdir(parents=True)
    (real_home / ".gemini" / "config").mkdir(parents=True)
    (real_home / ".gemini" / "config" / "config.json").write_text("{}\n", encoding="utf-8")
    (real_home / ".gemini" / "config" / "projects").symlink_to(external_projects)

    sym_home = tmp_path / "sym_home"
    sym_home.symlink_to(real_home)

    workspace = tmp_path / "worktrees" / "job-1"
    workspace.mkdir(parents=True)
    config_dir = real_home / ".gemini" / "antigravity-cli"
    config_dir.mkdir(parents=True)

    profile = build_seatbelt_profile(
        workspace=workspace,
        config_dir=config_dir,
        home=sym_home,
        adc_path=None,
    )
    resolved_real = real_home.resolve()
    resolved_ext = external_projects.resolve()

    for cfg_path in (sym_home / ".gemini" / "config", resolved_real / ".gemini" / "config"):
        meta = _last_matching_rule_for_path(profile, cfg_path, "file-read-metadata")
        assert meta is not None and meta[1].startswith("(allow file-read-metadata "), cfg_path
        cfg_file_read = _last_matching_rule_for_path(profile, cfg_path / "config.json", "file-read-data")
        assert cfg_file_read is not None and cfg_file_read[1].startswith("(deny "), cfg_path

    for proj_path in (
        sym_home / ".gemini" / "config" / "projects",
        resolved_real / ".gemini" / "config" / "projects",
        resolved_ext,
    ):
        proj_read = _last_matching_rule_for_path(profile, proj_path / "default-cli-project.json", "file-read-data")
        proj_write = _last_matching_rule_for_path(profile, proj_path / "default-cli-project.json", "file-write*")
        out_write = _last_matching_rule_for_path(profile, proj_path / "outside-of-project.json", "file-write*")
        assert proj_read is not None and proj_read[1].startswith("(allow "), proj_path
        assert proj_write is not None and proj_write[1].startswith("(allow "), proj_path
        assert out_write is not None and out_write[1].startswith("(deny file-write* "), proj_path


def test_plan_worker_sandbox_seatbelt_creates_gemini_projects_dir_when_home_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "fresh_mac_home"
    home.mkdir()
    workspace = tmp_path / "worktrees" / "job-1"
    workspace.mkdir(parents=True)
    config_dir = home / ".gemini" / "antigravity-cli"

    monkeypatch.setattr(
        worker_sandbox,
        "detect_sandbox_backend",
        lambda env=None: (worker_sandbox.BACKEND_SEATBELT, "/usr/bin/sandbox-exec", ""),
    )

    plan = plan_worker_sandbox(
        workspace_dir=workspace,
        config_dir=config_dir,
        env={"HOME": str(home), "CC_AGY_WORKER_SANDBOX": "auto"},
    )
    assert plan.backend == worker_sandbox.BACKEND_SEATBELT
    assert (home / ".gemini" / "config" / "projects").is_dir()

    # Non-existent HOME should not be created on disk.
    missing_home = tmp_path / "missing_home"
    plan_missing = plan_worker_sandbox(
        workspace_dir=workspace,
        config_dir=config_dir,
        env={"HOME": str(missing_home), "CC_AGY_WORKER_SANDBOX": "auto"},
    )
    assert plan_missing.backend == worker_sandbox.BACKEND_SEATBELT
    assert not missing_home.exists()


def test_bwrap_still_hides_keychains_dir_if_present_and_skips_when_absent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "ws"
    config_dir = home / ".gemini" / "antigravity-cli"
    workspace.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    without_dir = " ".join(
        build_bwrap_prefix("bwrap", workspace=workspace, config_dir=config_dir, home=home, adc_path=None)
    )
    assert "Library/Keychains" not in without_dir

    keychains = home / "Library" / "Keychains"
    keychains.mkdir(parents=True)
    with_dir = " ".join(
        build_bwrap_prefix("bwrap", workspace=workspace, config_dir=config_dir, home=home, adc_path=None)
    )
    assert f"--tmpfs {keychains.resolve()}" in with_dir
