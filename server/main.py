#!/usr/bin/env python3
"""Entry point for the cc-agy-plugin MCP stdio server."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

# Ensure plugin root is importable when launched via `python3 ${CLAUDE_PLUGIN_ROOT}/server/main.py`
# and isolate the MCP server's own imports from workspace/PYTHONPATH shadowing (e.g. target repos
# named `httpx/` or containing `calendar/`).
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = Path(__file__).resolve().parent


def _sanitize_server_sys_path() -> None:
    raw_pypath = {
        os.path.abspath(p)
        for p in (os.environ.get("PYTHONPATH") or "").split(os.pathsep)
        if p.strip()
    }
    cwd_abs = os.path.abspath(os.getcwd())
    server_abs = os.path.abspath(str(SERVER_DIR))
    plugin_abs = os.path.abspath(str(PLUGIN_ROOT))
    cleaned: list[str] = [str(PLUGIN_ROOT)]
    deferred: list[str] = []
    for entry in sys.path:
        abs_e = os.path.abspath(entry) if entry else cwd_abs
        if abs_e == plugin_abs:
            continue
        if abs_e in (cwd_abs, server_abs) or abs_e in raw_pypath:
            deferred.append(entry)
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


_sanitize_server_sys_path()


def _safe_has_spec(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _maybe_reexec_into_sdk_venv() -> None:
    """Re-exec into CC_AGY_PYTHON_BIN, managed user venv, or sibling .venv if google-antigravity is missing."""
    if os.environ.get("_CC_AGY_REEXEC") == "1":
        return
    from server.runtime_env import ensure_sdk_venv, python_has_plugin_sdks, resolve_sdk_python

    explicit_py = (os.environ.get("CC_AGY_PYTHON_BIN") or "").strip()
    if explicit_py and not explicit_py.startswith("${") and Path(explicit_py).is_file():
        target = Path(explicit_py)
        try:
            same = (
                Path(sys.executable).resolve() == target.resolve()
                and os.path.abspath(sys.executable) == os.path.abspath(str(target))
            )
        except OSError:
            same = False
        if not same:
            os.environ["_CC_AGY_REEXEC"] = "1"
            os.execv(str(target), [str(target), *sys.argv])
        return

    # When running on a Gemini Enterprise subscription via the standalone `agy` CLI,
    # the MCP server uses pure Python stdlib and does not need PyPI SDK wheels on startup.
    try:
        from server.regions import resolve_auth_config

        if resolve_auth_config().get("auth_mode") == "gemini_enterprise":
            return
    except Exception:
        try:
            from server.agy_cli import inspect_agy_cli_offline
            from server.regions import (
                _resolve_explicit_project_only,
                _should_auto_detect_agy_cli,
                resolve_api_key,
                resolve_configured_auth_mode,
                resolve_vertex_base_url,
            )

            if (
                resolve_configured_auth_mode() in (None, "gemini_enterprise")
                and not resolve_api_key()
                and not _resolve_explicit_project_only()
                and not resolve_vertex_base_url()
                and _should_auto_detect_agy_cli(None, os.environ)
                and inspect_agy_cli_offline().get("binary_found")
            ):
                return
        except Exception:
            pass

    if _safe_has_spec("google.antigravity") and _safe_has_spec("google.genai"):
        return

    resolved = resolve_sdk_python(PLUGIN_ROOT)
    if resolved != (sys.executable or "") and Path(resolved).is_file() and python_has_plugin_sdks(resolved):
        os.environ["_CC_AGY_REEXEC"] = "1"
        os.execv(resolved, [resolved, *sys.argv])

    auto_boot = (os.environ.get("CC_AGY_AUTO_BOOTSTRAP") or "1").strip().lower()
    if auto_boot not in ("0", "false", "no", "off") and "PYTEST_CURRENT_TEST" not in os.environ:
        boot = ensure_sdk_venv(
            PLUGIN_ROOT,
            output_fn=lambda msg: sys.stderr.write(f"[cc-agy-plugin] {msg}\n"),
        )
        boot_py = str(boot.get("python_bin") or "")
        if boot.get("ok") and boot_py and Path(boot_py).is_file():
            os.environ["_CC_AGY_REEXEC"] = "1"
            os.execv(boot_py, [boot_py, *sys.argv])
        if boot.get("error"):
            sys.stderr.write(f"[cc-agy-plugin] {boot['error']}\n")


_maybe_reexec_into_sdk_venv()

import atexit  # noqa: E402
import signal  # noqa: E402

from server.mcp_stdio import MCPServer, serve_stdio  # noqa: E402


def main() -> int:
    server = MCPServer()
    _shutting_down = False

    def _record_unfinished(*_args: object) -> None:
        nonlocal _shutting_down
        if _shutting_down:
            return
        _shutting_down = True
        try:
            server.job_manager.mark_running_interrupted()
        except Exception:
            pass
        try:
            server.job_manager.shutdown(wait=False)
        except Exception:
            pass

    def _on_sigterm(signum: int, frame: object) -> None:
        _record_unfinished()
        raise SystemExit(128 + signum)

    atexit.register(_record_unfinished)
    signal.signal(signal.SIGTERM, _on_sigterm)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _on_sigterm)
    serve_stdio(server)
    # Record unfinished jobs and terminate active workers right when stdin closes. The atexit hook alone
    # is too late: Python joins ThreadPoolExecutor worker threads before atexit handlers run.
    _record_unfinished()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
