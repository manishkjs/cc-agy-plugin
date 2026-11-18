"""The MCP server must start fast when the host runs in the user's home folder.

Claude Code starts the plugin's server with the session's folder as the workspace. When that
folder is the home folder and it is a git repo, every file under it is untracked, and listing
them during startup took longer than Claude Code waits for the MCP handshake.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from server import jobs, runtime_env


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo_with_tracked_change_and_untracked_file(root: Path) -> Path:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / ".zshrc").write_text("export A=1\n", encoding="utf-8")
    _git(root, "add", ".zshrc")
    _git(root, "commit", "-q", "-m", "init")
    (root / ".zshrc").write_text("export A=2\n", encoding="utf-8")
    (root / "Library").mkdir()
    (root / "Library" / "cache.txt").write_text("untracked\n", encoding="utf-8")
    return root


def test_home_folder_diff_keeps_tracked_changes_and_skips_untracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _repo_with_tracked_change_and_untracked_file(tmp_path / "home")
    monkeypatch.setattr(runtime_env, "user_home", lambda env=None: home)

    assert jobs.skips_untracked_scan(home) is True
    diff = jobs.capture_live_workspace_diff(home)
    assert ".zshrc" in diff
    assert "Library/cache.txt" not in diff


def test_project_folder_diff_still_includes_untracked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _repo_with_tracked_change_and_untracked_file(tmp_path / "project")
    monkeypatch.setattr(runtime_env, "user_home", lambda env=None: tmp_path / "someone_else")

    assert jobs.skips_untracked_scan(project) is False
    diff = jobs.capture_live_workspace_diff(project)
    assert ".zshrc" in diff
    assert "Library/cache.txt" in diff


def test_filesystem_root_skips_untracked_scan() -> None:
    assert jobs.skips_untracked_scan(Path(Path.cwd().anchor)) is True
