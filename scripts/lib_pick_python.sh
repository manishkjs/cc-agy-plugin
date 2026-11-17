#!/bin/sh
# Shared helper: find a Python 3.10+ interpreter. Source this file, then call pick_python.
#
# macOS ships /usr/bin/python3 as Python 3.9, which can't run this plugin, so nothing in the
# plugin may call a bare `python3`. pick_python prints the first new-enough interpreter it
# finds and returns 1 when there is none.
#
# Order: CC_AGY_PYTHON_BIN (or PYTHON_BIN), the plugin's managed virtualenv, versioned
# interpreters on PATH (newest first), then common install locations a GUI-launched host may
# not have on PATH (Homebrew, MacPorts, python.org framework builds).
#
# Callers must set cc_agy_root to the repository root before calling pick_python.

cc_agy_is_new_enough() {
  [ -n "$1" ] && [ -x "$1" ] || return 1
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

pick_python() {
  for pin in "${CC_AGY_PYTHON_BIN:-}" "${PYTHON_BIN:-}"; do
    case "${pin}" in
      '' | '${'*) continue ;;  # Unset, or an unsubstituted ${user_config...} placeholder.
    esac
    pin_path=$(command -v "${pin}" 2>/dev/null) || pin_path="${pin}"
    if cc_agy_is_new_enough "${pin_path}"; then echo "${pin_path}"; return 0; fi
  done
  data_home="${XDG_DATA_HOME:-${HOME}/.local/share}"
  for candidate in "${data_home}/cc-agy-plugin/venv/bin/python" "${cc_agy_root:-.}/.venv/bin/python"; do
    if cc_agy_is_new_enough "${candidate}"; then echo "${candidate}"; return 0; fi
  done
  for name in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    candidate=$(command -v "${name}" 2>/dev/null) || continue
    if cc_agy_is_new_enough "${candidate}"; then echo "${candidate}"; return 0; fi
  done
  for prefix in /opt/homebrew/bin /usr/local/bin /opt/local/bin "${HOME}/.local/bin"; do
    for name in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
      if cc_agy_is_new_enough "${prefix}/${name}"; then echo "${prefix}/${name}"; return 0; fi
    done
  done
  for candidate in /Library/Frameworks/Python.framework/Versions/3.*/bin/python3; do
    if cc_agy_is_new_enough "${candidate}"; then echo "${candidate}"; return 0; fi
  done
  return 1
}

# Prints a plain-English error for when pick_python found nothing.
cc_agy_python_missing_message() {
  found=$(command -v python3 2>/dev/null || echo "none")
  found_version=$("${found}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "unknown")
  echo "[cc-agy-plugin] This needs Python 3.10 or newer, but python3 is ${found} (version ${found_version})."
  echo "[cc-agy-plugin] Install a newer Python (for example: brew install python@3.13),"
  echo "[cc-agy-plugin] or set CC_AGY_PYTHON_BIN to a Python 3.10+ interpreter, then try again."
}
