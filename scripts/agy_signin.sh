#!/bin/bash
# Gemini Enterprise sign-in helper for `agy`; delegates to `server.agy_login`.
# Local desktops get agy's own browser sign-in, people at a terminal get a link and a code
# prompt, and agent shells (Claude Code, Cursor, OpenCode) get a 2-step link + `--complete`.

set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PLUGIN_ROOT="$(dirname "${SCRIPT_DIR}")"

cc_agy_root="${PLUGIN_ROOT}"
# shellcheck source=lib_pick_python.sh
. "${SCRIPT_DIR}/lib_pick_python.sh"
if ! python_bin="$(pick_python)"; then
  cc_agy_python_missing_message >&2
  exit 1
fi

cd "${PLUGIN_ROOT}" || exit 1
exec "${python_bin}" -m server.agy_login "$@"
