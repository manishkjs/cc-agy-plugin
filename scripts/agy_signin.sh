#!/bin/bash
# Walks the user through the one-time Gemini Enterprise sign-in for the
# Antigravity CLI (`agy`), which cc-agy-plugin uses in gemini-enterprise mode.
#
# The sign-in is a browser OAuth flow inside the interactive `agy` app, so it
# needs a real terminal. This script picks the best way to give the user one:
#   1. A controlling terminal is available (for example `curl ... | bash` typed
#      in Terminal): run `agy` right here on /dev/tty.
#   2. No terminal (for example the installer runs inside Claude Code), but we
#      are on macOS: open a new Terminal window running `agy`, then wait for
#      the sign-in to land.
#   3. Otherwise: print the exact command to run.
#
# Usage: agy_signin.sh [--wait SECONDS]
# Env:   PYTHON_BIN, CC_AGY_SKIP_SIGNIN=1 (do nothing),
#        CC_AGY_SIGNIN_WAIT (seconds to wait for a new Terminal window, default 90,
#        which keeps an installer run by an agent under its 2-minute tool timeout).

set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PLUGIN_ROOT="$(dirname "${SCRIPT_DIR}")"
readonly PYTHON_BIN="${PYTHON_BIN:-python3}"

wait_seconds="${CC_AGY_SIGNIN_WAIT:-90}"
if [[ "${1:-}" == "--wait" && -n "${2:-}" ]]; then
  wait_seconds="$2"
fi

# Prints "<signed_in 0|1> <agy_bin>" using the plugin's own offline check.
agy_state() {
  (
    cd "${PLUGIN_ROOT}" &&
      "${PYTHON_BIN}" - <<'EOF' 2>/dev/null
from server.agy_cli import inspect_agy_cli_offline

info = inspect_agy_cli_offline()
print(1 if info.get("auth_configured") else 0, info.get("agy_bin") or "")
EOF
  )
}

print_steps() {
  local agy_bin="$1"
  cat <<EOF

  ┌─ Gemini Enterprise sign-in ─────────────────────────────────────────┐
  │ 1. Pick "Use a Google Cloud project" (not "Google OAuth").          │
  │ 2. Finish the sign-in in your browser with your licensed            │
  │    Gemini Enterprise work account.                                  │
  │ 3. When the agy prompt appears, type /exit (or press Ctrl+C).       │
  └─────────────────────────────────────────────────────────────────────┘
  Command: ${agy_bin}

EOF
}

has_controlling_tty() {
  { true </dev/tty >/dev/tty; } 2>/dev/null
}

main() {
  if [[ "${CC_AGY_SKIP_SIGNIN:-}" == "1" || -n "${PYTEST_CURRENT_TEST:-}" ]]; then
    return 0
  fi

  local state signed_in agy_bin
  state="$(agy_state)"
  signed_in="${state%% *}"
  agy_bin="${state#* }"
  if [[ -z "${agy_bin}" || "${agy_bin}" == "${state}" ]]; then
    printf 'Antigravity CLI (agy) is not installed yet; skipping sign-in.\n' >&2
    return 0
  fi
  if [[ "${signed_in}" == "1" ]]; then
    printf 'Antigravity CLI is already signed in.\n'
    return 0
  fi

  if has_controlling_tty; then
    printf '\nStarting the Antigravity CLI so you can sign in now.\n' >/dev/tty
    print_steps "${agy_bin}" >/dev/tty
    "${agy_bin}" </dev/tty >/dev/tty 2>&1 || true
  elif [[ "$(uname -s)" == "Darwin" ]] && command -v open >/dev/null 2>&1; then
    printf 'Opening a new Terminal window for the Gemini Enterprise sign-in...\n'
    print_steps "${agy_bin}"
    local launcher
    launcher="$(mktemp "${TMPDIR:-/tmp}/cc-agy-signin.XXXXXX")" || return 0
    mv "${launcher}" "${launcher}.command"
    launcher="${launcher}.command"
    {
      printf '#!/bin/bash\nclear\n'
      printf 'echo "Gemini Enterprise sign-in for cc-agy-plugin"\n'
      printf 'echo "1. Pick: Use a Google Cloud project"\n'
      printf 'echo "2. Finish in the browser with your licensed work account"\n'
      printf 'echo "3. Type /exit when the agy prompt appears"\n'
      printf 'rm -f %q\n' "${launcher}"
      printf 'exec %q\n' "${agy_bin}"
    } >"${launcher}"
    chmod 700 "${launcher}"
    open -a Terminal "${launcher}" >/dev/null 2>&1 || true
    printf 'Waiting up to %ss for the sign-in to finish' "${wait_seconds}"
    local waited=0
    while (( waited < wait_seconds )); do
      sleep 5
      waited=$(( waited + 5 ))
      printf '.'
      state="$(agy_state)"
      if [[ "${state%% *}" == "1" ]]; then
        break
      fi
    done
    printf '\n'
  else
    printf '\nNo terminal is attached, so the sign-in cannot open here.\n'
    printf 'Open a terminal on this machine and run the command below.\n'
    print_steps "${agy_bin}"
    return 0
  fi

  state="$(agy_state)"
  if [[ "${state%% *}" == "1" ]]; then
    printf 'Signed in. cc-agy-plugin can now use your Gemini Enterprise seat.\n'
  else
    printf 'Sign-in not detected yet. Run this in a terminal and follow the steps:\n'
    print_steps "${agy_bin}"
  fi
  return 0
}

main "$@"
