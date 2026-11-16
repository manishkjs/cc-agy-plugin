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
server_main="${script_dir%/scripts}/server/main.py"

is_new_enough() {
  [ -n "$1" ] && [ -x "$1" ] || return 1
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

pick_python() {
  # 1. Explicit pin, unless it is an unsubstituted ${user_config...} placeholder.
  case "${CC_AGY_PYTHON_BIN:-}" in
    '' | '${'*) ;;
    *) if is_new_enough "${CC_AGY_PYTHON_BIN}"; then echo "${CC_AGY_PYTHON_BIN}"; return 0; fi ;;
  esac
  # 2. The plugin's managed virtualenv, when the installer created one.
  data_home="${XDG_DATA_HOME:-${HOME}/.local/share}"
  for candidate in "${data_home}/cc-agy-plugin/venv/bin/python" "${script_dir%/scripts}/.venv/bin/python"; do
    if is_new_enough "${candidate}"; then echo "${candidate}"; return 0; fi
  done
  # 3. Versioned interpreters on PATH, newest first, then python3.
  for name in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    candidate=$(command -v "${name}" 2>/dev/null) || continue
    if is_new_enough "${candidate}"; then echo "${candidate}"; return 0; fi
  done
  # 4. Common install locations a GUI-launched host may not have on PATH.
  for prefix in /opt/homebrew/bin /usr/local/bin /opt/local/bin "${HOME}/.local/bin"; do
    for name in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
      if is_new_enough "${prefix}/${name}"; then echo "${prefix}/${name}"; return 0; fi
    done
  done
  for candidate in /Library/Frameworks/Python.framework/Versions/3.*/bin/python3; do
    if is_new_enough "${candidate}"; then echo "${candidate}"; return 0; fi
  done
  return 1
}

if python_bin=$(pick_python); then
  exec "${python_bin}" "${server_main}" "$@"
fi

found=$(command -v python3 2>/dev/null || echo "none")
found_version=$("${found}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "unknown")
cat >&2 <<EOF
[cc-agy-plugin] Cannot start the MCP server: it needs Python 3.10 or newer.
[cc-agy-plugin] Found python3 at ${found} (version ${found_version}).
[cc-agy-plugin] Install a newer Python (for example: brew install python@3.13),
[cc-agy-plugin] or set CC_AGY_PYTHON_BIN to a Python 3.10+ interpreter, then restart.
EOF
exit 1
