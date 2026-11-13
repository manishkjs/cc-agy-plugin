#!/bin/bash
# One-command bootstrap installer for cc-agy-plugin.
# Clones or updates ~/.local/share/cc-agy-plugin/repo, runs server.install,
# and verifies readiness with --doctor.

set -euo pipefail

readonly DEFAULT_REPO_URL="https://github.com/manishkjs/cc-agy-plugin.git"

main() {
  local repo_url="${CC_AGY_REPO_URL:-${DEFAULT_REPO_URL}}"
  local data_home="${XDG_DATA_HOME:-${HOME}/.local/share}"
  local checkout_dir="${CC_AGY_INSTALL_REPO_DIR:-${data_home}/cc-agy-plugin/repo}"
  local python_bin="${PYTHON_BIN:-python3}"

  if ! command -v git >/dev/null 2>&1; then
    printf 'Error: git is required to install cc-agy-plugin.\n' >&2
    exit 1
  fi
  if ! command -v "${python_bin}" >/dev/null 2>&1; then
    printf 'Error: %s is required (Python 3.10+).\n' "${python_bin}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${checkout_dir}")"
  if [[ -d "${checkout_dir}/.git" ]]; then
    git -C "${checkout_dir}" fetch --quiet --depth=1 origin main || true
    git -C "${checkout_dir}" merge --ff-only --quiet FETCH_HEAD || true
  else
    rm -rf "${checkout_dir}"
    git clone --depth=1 "${repo_url}" "${checkout_dir}"
  fi

  (
    cd "${checkout_dir}"
    "${python_bin}" -m server.install "$@"
    "${python_bin}" -m server.install --doctor --human
  )
}

main "$@"
