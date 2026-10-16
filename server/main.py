#!/usr/bin/env python3
"""Entry point for the cc-agy-plugin MCP stdio server."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure plugin root is importable when launched via `python3 ${CLAUDE_PLUGIN_ROOT}/server/main.py`
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

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
