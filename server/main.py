#!/usr/bin/env python3
"""Entry point for the cc-agy-plugin MCP stdio server."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

# Ensure plugin root is importable when launched via `python3 ${CLAUDE_PLUGIN_ROOT}/server/main.py`
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


def _maybe_reexec_into_sdk_venv() -> None:
    """Re-exec into CC_AGY_PYTHON_BIN or sibling .venv/bin/python if google-antigravity is missing."""
    if os.environ.get("_CC_AGY_REEXEC") == "1":
        return
    explicit_py = (os.environ.get("CC_AGY_PYTHON_BIN") or "").strip()
    if explicit_py and not explicit_py.startswith("${") and Path(explicit_py).is_file():
        target = Path(explicit_py)
        try:
            same = Path(sys.executable).resolve() == target.resolve() and os.path.abspath(sys.executable) == os.path.abspath(str(target))
        except OSError:
            same = False
        if not same:
            os.environ["_CC_AGY_REEXEC"] = "1"
            os.execv(str(target), [str(target), *sys.argv])
        return

    if importlib.util.find_spec("google.antigravity") is not None:
        return

    for candidate in (
        PLUGIN_ROOT / ".venv" / "bin" / "python",
        PLUGIN_ROOT.parent / ".venv" / "bin" / "python",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            os.environ["_CC_AGY_REEXEC"] = "1"
            os.execv(str(candidate), [str(candidate), *sys.argv])


_maybe_reexec_into_sdk_venv()

import atexit  # noqa: E402
import signal  # noqa: E402

from server.mcp_stdio import MCPServer, serve_stdio  # noqa: E402


def main() -> int:
    server = MCPServer()

    def _record_unfinished(*_args: object) -> None:
        try:
            server.job_manager.mark_running_interrupted()
        except Exception:
            pass

    def _on_sigterm(signum: int, frame: object) -> None:
        _record_unfinished()
        raise SystemExit(128 + signum)

    atexit.register(_record_unfinished)
    signal.signal(signal.SIGTERM, _on_sigterm)
    serve_stdio(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
