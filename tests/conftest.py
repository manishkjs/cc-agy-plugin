"""Shared pytest fixtures for cc-agy-plugin."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="session")
def _bare_python_agy_sdk_stub(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """Provide a lightweight `google.antigravity` namespace stub when running pytest on a bare interpreter."""
    if importlib.util.find_spec("google.antigravity") is not None:
        return None
    stub_root = tmp_path_factory.mktemp("agy_sdk_stub")
    pkg = stub_root / "google" / "antigravity"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    sys.path.insert(0, str(stub_root))
    importlib.invalidate_caches()
    return stub_root


@pytest.fixture(autouse=True)
def _isolated_agy_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _bare_python_agy_sdk_stub: Path | None,
) -> None:
    """Keep `agy` caches out of the real home directory and pin test-only defaults.

    Tests that cover the model catalog turn it back on with `CC_AGY_MODEL_CATALOG=1` in
    the env mapping they pass to the code under test.
    """
    from server import agy_state

    if _bare_python_agy_sdk_stub is not None:
        stub_s = str(_bare_python_agy_sdk_stub)
        cur_pypath = os.environ.get("PYTHONPATH", "")
        if not cur_pypath.startswith(f"{stub_s}{os.pathsep}") and cur_pypath != stub_s:
            monkeypatch.setenv("PYTHONPATH", f"{stub_s}{os.pathsep}{cur_pypath}" if cur_pypath else stub_s)

    if sys.executable:
        py_dir = str(Path(sys.executable).resolve().parent)
        cur_path = os.environ.get("PATH", "")
        if py_dir and not cur_path.startswith(f"{py_dir}{os.pathsep}"):
            monkeypatch.setenv("PATH", f"{py_dir}{os.pathsep}{cur_path}")
    if (os.environ.get("GEMINI_API_KEY") or "").strip() == "dummy-key":
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    gw_base = (os.environ.get("GOOGLE_GEMINI_BASE_URL") or "").strip()
    if "127.0.0.1" in gw_base or "localhost" in gw_base:
        monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT", raising=False)

    monkeypatch.setenv("CC_AGY_STATE_DIR", str(tmp_path / "agy-state"))
    monkeypatch.setenv("CC_AGY_MODEL_CATALOG", "0")
    # Existing worker tests check process and file behavior without a sandbox; the sandbox
    # tests in test_worker_hardening.py opt back in with CC_AGY_WORKER_SANDBOX=auto.
    monkeypatch.setenv("CC_AGY_WORKER_SANDBOX", "off")
    agy_state.clear_memory_caches()
    yield
    agy_state.clear_memory_caches()
