"""Shared pytest fixtures for cc-agy-plugin."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_agy_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `agy` caches out of the real home directory and pin test-only defaults.

    Tests that cover the model catalog turn it back on with `CC_AGY_MODEL_CATALOG=1` in
    the env mapping they pass to the code under test.
    """
    from server import agy_state

    monkeypatch.setenv("CC_AGY_STATE_DIR", str(tmp_path / "agy-state"))
    monkeypatch.setenv("CC_AGY_MODEL_CATALOG", "0")
    # Existing worker tests check process and file behavior without a sandbox; the sandbox
    # tests in test_worker_hardening.py opt back in with CC_AGY_WORKER_SANDBOX=auto.
    monkeypatch.setenv("CC_AGY_WORKER_SANDBOX", "off")
    agy_state.clear_memory_caches()
    yield
    agy_state.clear_memory_caches()
