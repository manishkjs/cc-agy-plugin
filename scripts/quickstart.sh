#!/bin/bash
# One-command bootstrap installer for cc-agy-plugin.
# Safely clones or updates ~/.local/share/cc-agy-plugin/repo, runs interactive
# dropdown setup (on /dev/tty in a terminal, or via host dropdown prompts inside
# Claude Code / Cursor / OpenCode), completes Gemini Enterprise OAuth via a
# clickable sign-in URL, and verifies readiness with --doctor.

set -euo pipefail

readonly DEFAULT_REPO_URL="https://github.com/manishkjs/cc-agy-plugin.git"
readonly MANAGED_MARKER=".cc-agy-managed-repo"

# Returns 0 when the install args select Gemini Enterprise (agy CLI) and are not a dry run.
wants_enterprise_signin() {
  local arg prev="" enterprise=1
  for arg in "$@"; do
    case "${arg}" in
      --dry-run) return 1 ;;
      --auth=gemini-enterprise | --auth=agy-cli | --auth=gemini_enterprise) enterprise=0 ;;
    esac
    if [[ "${prev}" == "--auth" ]]; then
      case "${arg}" in
        gemini-enterprise | agy-cli | gemini_enterprise) enterprise=0 ;;
      esac
    fi
    prev="${arg}"
  done
  return "${enterprise}"
}

is_dry_run() {
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "--dry-run" ]]; then
      return 0
    fi
  done
  return 1
}

has_controlling_tty() {
  { true </dev/tty >/dev/tty; } 2>/dev/null
}

emit_agent_dropdown_onboarding() {
  local checkout_dir="$1"
  local python_bin="$2"
  local orig_cwd="$3"
  cat <<EOF

[CC_AGY_INTERACTIVE_ONBOARDING]
cc-agy-plugin is downloaded to ${checkout_dir}.

AGENT / CLAUDE CODE — RUN THIS INTERACTIVE DROPDOWN ONBOARDING NOW:
1. Immediately call AskUserQuestion (no preamble) to let the user pick from two dropdowns:
   - Question 1 (header: "Auth mode"): "How should cc-agy-plugin connect to Gemini?"
     Options:
       * "Gemini Enterprise Seat (Recommended)" — Included \$0 seat quota via standalone agy CLI
       * "Google AI Studio API Key" — Pay-as-you-go with GEMINI_API_KEY
       * "Google Cloud Vertex AI" — Pay-as-you-go with GOOGLE_CLOUD_PROJECT and ADC
       * "Enterprise AI Gateway" — Route through LiteLLM or Bifrost gateway URL
   - Question 2 (header: "Preset"): "Which model routing preset do you want?"
     Options:
       * "Balanced (Recommended)" — Default worker gemini-4-hydrogen, escalates to gemini-4-carbon
       * "Quality" — Default worker gemini-4-carbon
       * "Economy" — Lowest-cost worker routing

2. Map their choices to flags (--auth gemini-enterprise | api-key | vertex | gateway, --profile balanced | quality | economy) and run:
     ${python_bin} "${checkout_dir}/server/install.py" --host claude --auth <AUTH> --profile <PROFILE>

3. If they picked "Gemini Enterprise Seat":
   - Run:
       ${python_bin} "${checkout_dir}/server/agy_login.py"
   - If it prints a Google OAuth URL (https://accounts.google.com/...), show that clickable URL verbatim to the user right away and ask them (via AskUserQuestion or chat) to click it, sign in with their Gemini Enterprise account, click 'Copy to Clipboard' on the callback page, and paste back the code.
   - As soon as they paste the code, finish sign-in with:
       ${python_bin} "${checkout_dir}/server/agy_login.py" --complete "<PASTE_CODE>"

4. Verify readiness with:
     ${python_bin} "${checkout_dir}/server/install.py" --doctor --human --auth <AUTH> --workspace "${orig_cwd}"
   and remind the user to run /reload-plugins in already-open Claude Code tabs.
EOF
}

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

  # Zero-flag invocation:
  # - In a real terminal (`curl ... | bash` where stdin was consumed by curl), reattach /dev/tty
  #   so the user gets the interactive arrow-key dropdown wizard + immediate OAuth sign-in.
  # - Inside an agent tool runner without /dev/tty (e.g. pasted directly into Claude Code),
  #   emit structured instructions so Claude Code pops up native AskUserQuestion dropdowns!
  if [[ $# -eq 0 ]]; then
    if [[ -z "${PYTEST_CURRENT_TEST:-}" ]] && has_controlling_tty; then
      (
        cd "${checkout_dir}"
        "${python_bin}" -m server.install </dev/tty >/dev/tty
        if "${python_bin}" -c "from server.agy_cli import inspect_agy_cli_offline; s=inspect_agy_cli_offline(); raise SystemExit(0 if s.get('binary_found') and not s.get('auth_configured') else 1)" 2>/dev/null; then
          PYTHON_BIN="${python_bin}" bash "${checkout_dir}/scripts/agy_signin.sh"
        fi
        "${python_bin}" -m server.install --doctor --human --allow-pending-login --workspace "${orig_cwd}"
      )
      return 0
    else
      emit_agent_dropdown_onboarding "${checkout_dir}" "${python_bin}" "${orig_cwd}"
      return 0
    fi
  fi

  (
    cd "${checkout_dir}"
    "${python_bin}" -m server.install "$@"
    if wants_enterprise_signin "$@"; then
      PYTHON_BIN="${python_bin}" bash "${checkout_dir}/scripts/agy_signin.sh"
    fi
    if ! is_dry_run "$@"; then
      "${python_bin}" -m server.install --doctor --human --allow-pending-login --workspace "${orig_cwd}"
    fi
  )
}

main "$@"
