"""Tests for macOS Seatbelt keychain access in `server.worker_sandbox`."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.worker_sandbox import build_bwrap_prefix, build_seatbelt_profile


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

    last_deny_idx = max(idx for idx, line in enumerate(lines) if line.startswith("(deny "))
    allow_idx = lines.index(last_write)
    assert allow_idx > last_deny_idx


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
