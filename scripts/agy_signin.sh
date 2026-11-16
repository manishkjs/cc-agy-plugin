#!/bin/bash
# Interactive and two-step headless Gemini Enterprise OAuth helper for `agy`.
# Delegates to `python3 -m server.agy_login` so local desktops get automatic
# browser sign-in, SSH terminals get a clickable OAuth URL on /dev/tty, and
# non-TTY agent sessions (Claude Code, Cursor, OpenCode) get a live 2-step
# PKCE URL + `--complete "<CODE>"` exchange.

set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PLUGIN_ROOT="$(dirname "${SCRIPT_DIR}")"
readonly PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PLUGIN_ROOT}" || exit 1
exec "${PYTHON_BIN}" -m server.agy_login "$@"
