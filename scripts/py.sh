#!/bin/sh
# Runs a plugin Python script with a Python 3.10+ interpreter.
#
# Slash commands and setup instructions call this instead of a bare `python3`, which is
# Python 3.9 on stock macOS and can't run the plugin.
#
# Usage: py.sh <script.py> [args...]
#        py.sh --which          # print the interpreter it would use
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cc_agy_root="${script_dir%/scripts}"
. "${script_dir}/lib_pick_python.sh"

if ! python_bin=$(pick_python); then
  cc_agy_python_missing_message >&2
  exit 1
fi
if [ "${1:-}" = "--which" ]; then
  echo "${python_bin}"
  exit 0
fi
exec "${python_bin}" "$@"
