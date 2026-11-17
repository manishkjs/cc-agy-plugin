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

# True only for a person at a terminal. Agent shells (Claude Code, Cursor, OpenCode, Codex)
# often inherit the user's /dev/tty while piping stdout; prompting there hangs the agent.
is_person_at_terminal() {
  [[ -t 1 ]] || return 1
  local marker
  # CURSOR_TRACE_ID is left out: Cursor also sets it in terminals a person types in.
  for marker in CLAUDECODE CLAUDE_CODE_ENTRYPOINT OPENCODE CODEX_SANDBOX; do
    [[ -n "${!marker:-}" ]] && return 1
  done
  has_controlling_tty
}

# Prints the value of an install flag such as --auth (empty when not given).
flag_value() {
  local name="$1" arg prev=""
  shift
  for arg in "$@"; do
    case "${arg}" in
      "${name}"=*) printf '%s' "${arg#"${name}"=}"; return 0 ;;
    esac
    if [[ "${prev}" == "${name}" ]]; then
      printf '%s' "${arg}"
      return 0
    fi
    prev="${arg}"
  done
}

auth_arg() {
  flag_value --auth "$@"
}

has_arg() {
  local want="$1" arg
  shift
  for arg in "$@"; do
    [[ "${arg}" == "${want}" ]] && return 0
  done
  return 1
}

emit_agent_dropdown_onboarding() {
  local checkout_dir="$1"
  local python_bin="$2"
  local orig_cwd="$3"
  # Quoted heredoc: bash must not run the backticks or expand anything in this text.
  # The three paths are filled in afterwards with plain string replacement.
  local template
  template="$(cat <<'EOF'

[CC_AGY_INTERACTIVE_ONBOARDING]
cc-agy-plugin is downloaded to @CHECKOUT_DIR@.
Run every command below exactly as written (it picks a Python 3.10+; never call bare python3).

AGENT / CLAUDE CODE — RUN THIS INTERACTIVE DROPDOWN ONBOARDING NOW:
1. Immediately call AskUserQuestion (no preamble) with two questions:
   - Question 1 (header: "Auth mode"): "How should cc-agy-plugin connect to Gemini?"
     Options:
       * "Gemini Enterprise Seat (Recommended)" — Included $0 seat quota via standalone agy CLI
       * "Google AI Studio API Key" — Pay-as-you-go with GEMINI_API_KEY
       * "Google Cloud Vertex AI" — Pay-as-you-go with a Google Cloud project and ADC
       * "Enterprise AI Gateway" — Route through a LiteLLM or Bifrost gateway URL
   - Question 2 (header: "Preset"): "Which model routing preset do you want?"
     Options:
       * "Balanced (Recommended)" — Default worker gemini-4-hydrogen, escalates to gemini-4-carbon
       * "Quality" — Default worker gemini-4-carbon
       * "Economy" — Lowest-cost worker routing

2. Collect what the chosen mode needs BEFORE installing:
   - Gemini Enterprise: nothing.
   - Google AI Studio API Key: the key must be in the shell that starts Claude Code, never in chat.
     If GEMINI_API_KEY is not set, ask the user to add `export GEMINI_API_KEY=...` to their shell
     profile and restart Claude Code, then run step 3.
   - Vertex AI: ask (free text) for the Google Cloud project ID, and add `--project <ID>` in step 3.
     It also needs `gcloud auth application-default login` done once by the user.
   - Enterprise AI Gateway: ask (free text) for the gateway base URL, and add `--base-url <URL>`.

3. Install (AUTH = gemini-enterprise | api-key | vertex | gateway, PROFILE = balanced | quality | economy):
     sh "@CHECKOUT_DIR@/scripts/py.sh" "@CHECKOUT_DIR@/server/install.py" --host claude --auth <AUTH> --profile <PROFILE> [--project <ID>] [--base-url <URL>]
   If it prints "! ..." lines, fix what they say (or ask the user) and run it again.

4. Gemini Enterprise only — sign in:
     sh "@CHECKOUT_DIR@/scripts/py.sh" "@CHECKOUT_DIR@/server/agy_login.py" --start --json
   - "already_signed_in" or "signed_in": true -> go to step 5. (On a Mac a browser window may
     open for sign-in; the command waits for it.)
   - "auth_url" present: show "short_url" on its own line first if it is set, then the full
     "auth_url" on its own line, then [Sign in with Google](<auth_url>). Only if
     "clipboard_requested" is true, add: "it may also be on your clipboard". Ask the user to sign
     in with their Gemini Enterprise work account, click 'Copy to Clipboard' on the Antigravity
     page, and paste the code. "reused": true means an earlier link is still valid; show it again.
   - When they paste the code, run the "complete_command" from the JSON with the code in place of
     <PASTE_CODE>. It looks up their license, so the first agy call works.
   - If it prints [FAIL], show the message: it says exactly what to do (for example, sign in with
     the work account that has the seat, or set CC_AGY_ENTERPRISE_PROJECT and use --force).
   - Never tell the user to sign in with `agy -p`: print mode gives up after 60 seconds.

5. Verify:
     sh "@CHECKOUT_DIR@/scripts/py.sh" "@CHECKOUT_DIR@/server/install.py" --doctor --human --auth <AUTH> --workspace "@ORIG_CWD@"
   Then tell the user to run /reload-plugins in THIS Claude Code session (and any other open
   tabs) before using /cc-agy-plugin:agy or /cc-agy-plugin:doctor.
EOF
)"
  # Bash 5.2+ treats "&" in a replacement as "the matched text"; paths such as R&D/ must stay literal.
  shopt -u patsub_replacement 2>/dev/null || true
  template="${template//@CHECKOUT_DIR@/${checkout_dir}}"
  template="${template//@ORIG_CWD@/${orig_cwd}}"
  printf '%s\n' "${template}"
}

main() {
  local orig_cwd
  orig_cwd="$(pwd)"
  local repo_url="${CC_AGY_REPO_URL:-${DEFAULT_REPO_URL}}"
  local explicit_ref="${CC_AGY_REPO_REF:-}"
  local repo_ref="${explicit_ref:-main}"
  local data_home="${XDG_DATA_HOME:-${HOME}/.local/share}"
  local checkout_dir="${CC_AGY_INSTALL_REPO_DIR:-${data_home}/cc-agy-plugin/repo}"
  local python_bin=""

  if ! command -v git >/dev/null 2>&1; then
    printf 'Error: git is required to install cc-agy-plugin.\n' >&2
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

  # Pick a Python 3.10+ (stock macOS python3 is 3.9). Older pinned checkouts lack the helper.
  if [[ -f "${checkout_dir}/scripts/lib_pick_python.sh" ]]; then
    # shellcheck source=/dev/null
    cc_agy_root="${checkout_dir}" . "${checkout_dir}/scripts/lib_pick_python.sh"
    cc_agy_root="${checkout_dir}"
    if ! python_bin="$(pick_python)"; then
      cc_agy_python_missing_message >&2
      exit 1
    fi
  else
    python_bin="${PYTHON_BIN:-python3}"
  fi

  # Zero-flag invocation:
  # - In a real terminal (`curl ... | bash` where stdin was consumed by curl), reattach /dev/tty
  #   so the user gets the interactive arrow-key dropdown wizard + immediate OAuth sign-in.
  # - Inside an agent tool runner without /dev/tty (e.g. pasted directly into Claude Code),
  #   emit structured instructions so Claude Code pops up native AskUserQuestion dropdowns!
  if [[ $# -eq 0 ]]; then
    if [[ -z "${PYTEST_CURRENT_TEST:-}" ]] && is_person_at_terminal; then
      (
        cd "${checkout_dir}"
        "${python_bin}" -m server.install </dev/tty >/dev/tty
        # The wizard records what the user picked; use it instead of guessing.
        picked_auth="$("${python_bin}" -c "import json, os, pathlib; p = pathlib.Path(os.environ.get('XDG_DATA_HOME') or pathlib.Path.home() / '.local/share') / 'cc-agy-plugin/last-install.json'; print(json.loads(p.read_text()).get('auth_mode', '') if p.is_file() else '')" 2>/dev/null || true)"
        if [[ "${picked_auth}" == "gemini_enterprise" ]] && "${python_bin}" -c "from server.agy_cli import inspect_agy_cli_offline; s=inspect_agy_cli_offline(); raise SystemExit(0 if s.get('binary_found') and (not s.get('auth_configured') or s.get('consumer_account_detected')) else 1)" 2>/dev/null; then
          PYTHON_BIN="${python_bin}" bash "${checkout_dir}/scripts/agy_signin.sh"
        fi
        doctor_args=(--doctor --human --allow-pending-login --workspace "${orig_cwd}")
        if [[ -n "${picked_auth}" ]]; then
          doctor_args+=(--auth "${picked_auth//_/-}")
        fi
        "${python_bin}" -m server.install "${doctor_args[@]}"
      )
      return 0
    else
      emit_agent_dropdown_onboarding "${checkout_dir}" "${python_bin}" "${orig_cwd}"
      return 0
    fi
  fi

  local chosen_auth chosen_agy
  chosen_auth="$(auth_arg "$@")"
  chosen_agy="$(flag_value --agy-cli-path "$@")"
  (
    cd "${checkout_dir}"
    "${python_bin}" -m server.install "$@"
    if wants_enterprise_signin "$@"; then
      if has_arg --non-interactive "$@"; then
        export CC_AGY_NONINTERACTIVE=1
      fi
      PYTHON_BIN="${python_bin}" bash "${checkout_dir}/scripts/agy_signin.sh"
    fi
    if ! is_dry_run "$@"; then
      doctor_args=(--doctor --human --allow-pending-login --workspace "${orig_cwd}")
      if [[ -n "${chosen_auth}" ]]; then
        doctor_args+=(--auth "${chosen_auth}")
      fi
      if [[ -n "${chosen_agy}" ]]; then
        doctor_args+=(--agy-cli-path "${chosen_agy}")
      fi
      "${python_bin}" -m server.install "${doctor_args[@]}"
    fi
  )
}

main "$@"
