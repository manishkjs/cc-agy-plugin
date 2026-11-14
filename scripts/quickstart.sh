#!/bin/bash
# One-command bootstrap installer for cc-agy-plugin.
# Safely clones or updates ~/.local/share/cc-agy-plugin/repo, runs server.install,
# and verifies readiness with --doctor against the caller's working directory.

set -euo pipefail

readonly DEFAULT_REPO_URL="https://github.com/manishkjs/cc-agy-plugin.git"
readonly MANAGED_MARKER=".cc-agy-managed-repo"

main() {
  local orig_cwd
  orig_cwd="$(pwd)"
  local repo_url="${CC_AGY_REPO_URL:-${DEFAULT_REPO_URL}}"
  local explicit_ref="${CC_AGY_REPO_REF:-}"
  local repo_ref="${explicit_ref:-main}"
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
    if [[ ! -f "${checkout_dir}/${MANAGED_MARKER}" && ! -f "${checkout_dir}/.claude-plugin/plugin.json" ]]; then
      printf 'Error: %s is an existing git repository not managed by cc-agy-plugin. Refusing to modify it.\n' \
        "${checkout_dir}" >&2
      printf 'Unset CC_AGY_INSTALL_REPO_DIR or point it at an empty path.\n' >&2
      exit 1
    fi
    if ! git -C "${checkout_dir}" -c core.hooksPath=/dev/null fetch --quiet --depth=1 origin "${repo_ref}"; then
      if [[ -n "${explicit_ref}" ]]; then
        printf 'Error: failed to fetch pinned CC_AGY_REPO_REF=%s from %s.\n' \
          "${repo_ref}" "${repo_url}" >&2
        exit 1
      fi
      printf 'Warning: could not fetch %s from %s; continuing with existing checkout at %s.\n' \
        "${repo_ref}" "${repo_url}" "${checkout_dir}" >&2
    else
      if ! git -C "${checkout_dir}" -c core.hooksPath=/dev/null checkout --quiet FETCH_HEAD 2>/dev/null; then
        if [[ -n "${explicit_ref}" ]]; then
          printf 'Error: failed to check out pinned CC_AGY_REPO_REF=%s in %s.\n' \
            "${repo_ref}" "${checkout_dir}" >&2
          exit 1
        fi
        printf 'Warning: local modifications exist in %s; keeping current checkout.\n' "${checkout_dir}" >&2
      fi
    fi
  else
    if [[ -e "${checkout_dir}" ]]; then
      if [[ -d "${checkout_dir}" && -z "$(ls -A "${checkout_dir}" 2>/dev/null || true)" ]]; then
        rmdir "${checkout_dir}"
      elif [[ -f "${checkout_dir}/${MANAGED_MARKER}" ]]; then
        rm -rf "${checkout_dir}"
      else
        printf 'Error: %s already exists and is not a cc-agy-plugin git checkout. Refusing to overwrite it.\n' \
          "${checkout_dir}" >&2
        printf 'Unset CC_AGY_INSTALL_REPO_DIR or point it at an empty path.\n' >&2
        exit 1
      fi
    fi
    if [[ "${repo_ref}" =~ ^[0-9a-fA-F]{40}$ ]]; then
      mkdir -p "${checkout_dir}"
      if ! (
        git -C "${checkout_dir}" -c core.hooksPath=/dev/null init --quiet &&
        git -C "${checkout_dir}" -c core.hooksPath=/dev/null remote add origin "${repo_url}" &&
        git -C "${checkout_dir}" -c core.hooksPath=/dev/null fetch --quiet --depth=1 origin "${repo_ref}" &&
        git -C "${checkout_dir}" -c core.hooksPath=/dev/null checkout --quiet FETCH_HEAD
      ); then
        rm -rf "${checkout_dir}"
        printf 'Error: failed to fetch pinned commit %s from %s.\n' "${repo_ref}" "${repo_url}" >&2
        exit 1
      fi
    elif [[ "${repo_ref}" =~ ^[0-9a-fA-F]{7,39}$ ]]; then
      printf 'Error: abbreviated commit SHA %s cannot be fetched directly; pass the full 40-character SHA or a tag/branch name.\n' \
        "${repo_ref}" >&2
      exit 1
    else
      git -c core.hooksPath=/dev/null clone --depth=1 --branch "${repo_ref}" "${repo_url}" "${checkout_dir}"
    fi
    touch "${checkout_dir}/${MANAGED_MARKER}"
  fi

  (
    cd "${checkout_dir}"
    "${python_bin}" -m server.install "$@"
    "${python_bin}" -m server.install --doctor --human --allow-pending-login --workspace "${orig_cwd}"
  )
}

main "$@"
