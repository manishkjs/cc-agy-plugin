#!/bin/sh
# Starts the cc-agy-plugin MCP server with a Python that is new enough.
#
# Hosts such as Claude Code launch MCP servers with a bare `python3`. On macOS
# that often resolves to Apple's /usr/bin/python3 (3.9), which cannot import
# the server, so the host only reports CONNECTION_CLOSED. This launcher picks
# the first Python 3.10+ it can find and execs the server with it, or prints a
# clear error to stderr (hosts show MCP server stderr in their logs).
#
# Usage: run_mcp_server.sh [server args...]
# Env:   CC_AGY_PYTHON_BIN pins the interpreter.

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cc_agy_root="${script_dir%/scripts}"
server_main="${cc_agy_root}/server/main.py"
. "${script_dir}/lib_pick_python.sh"

if python_bin=$(pick_python); then
  exec "${python_bin}" "${server_main}" "$@"
fi

echo "[cc-agy-plugin] Cannot start the MCP server." >&2
cc_agy_python_missing_message >&2
exit 1
