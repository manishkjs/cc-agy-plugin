---
name: setup
description: Configure cc-agy-plugin connection mode, model preset, and Gemini Enterprise sign-in using interactive dropdowns
---

Start each command with the `ROOT=...` lines shown below. They find the plugin even when
`CLAUDE_PLUGIN_ROOT` is unset or it was installed from a source checkout. Run everything through
`sh "$ROOT/scripts/py.sh"`: it picks a Python 3.10+ (stock macOS `python3` is 3.9).

1. Use `AskUserQuestion` immediately (no preamble) with two questions:
   - **Auth mode** (`header: "Auth mode"`): *"How should cc-agy-plugin connect to Gemini?"*
     - `Gemini Enterprise Seat (Recommended)` — Included $0 seat quota via standalone `agy` CLI
     - `Google AI Studio API Key` — Pay-as-you-go with `GEMINI_API_KEY`
     - `Google Cloud Vertex AI` — Pay-as-you-go with a Google Cloud project and ADC
     - `Enterprise AI Gateway` — Route through a LiteLLM or Bifrost gateway URL
   - **Model preset** (`header: "Preset"`): *"Which model routing preset do you want?"*
     - `Balanced (Recommended)` — Default worker `gemini-4-hydrogen`, escalates to `gemini-4-carbon`
     - `Quality` — Default worker `gemini-4-carbon`
     - `Economy` — Lowest-cost worker routing

2. Collect what the chosen mode needs before installing:
   - **Gemini Enterprise:** nothing.
   - **AI Studio API key:** the key must be in the shell that starts Claude Code, never in chat.
     Install first (step 3). Then, if `GEMINI_API_KEY` is not set, ask the user to add
     `export GEMINI_API_KEY=...` to their shell profile and restart Claude Code.
   - **Vertex AI:** ask for the Google Cloud project ID and pass `--project <ID>`. The user also
     needs `gcloud auth application-default login` once.
   - **AI gateway:** ask for the gateway base URL and pass `--base-url <URL>`.

3. Install (`<AUTH>` is `gemini-enterprise`, `api-key`, `vertex` or `gateway`; `<PROFILE>` is
   `balanced`, `quality` or `economy`):
   ```bash
   ROOT="${CLAUDE_PLUGIN_ROOT:-}"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
for candidate in "$ROOT" "$DATA/cc-agy-plugin/current" "$DATA/cc-agy-plugin/repo"; do
  [ -f "$candidate/scripts/py.sh" ] && ROOT="$candidate" && break
done
   sh "$ROOT/scripts/py.sh" "$ROOT/server/install.py" --host claude --auth <AUTH> --profile <PROFILE> [--project <ID>] [--base-url <URL>]
   ```
   If it prints `! ...` lines, fix what they say (or ask the user) and run it again.

4. **Gemini Enterprise only:** follow `/cc-agy-plugin:login` (run `agy_login.py --start --json` and
   handle its JSON exactly as that command describes).

5. Verify with the CLI doctor (the MCP `doctor` tool still runs the old settings until reload):
   ```bash
   ROOT="${CLAUDE_PLUGIN_ROOT:-}"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
for candidate in "$ROOT" "$DATA/cc-agy-plugin/current" "$DATA/cc-agy-plugin/repo"; do
  [ -f "$candidate/scripts/py.sh" ] && ROOT="$candidate" && break
done
   sh "$ROOT/scripts/py.sh" "$ROOT/server/install.py" --doctor --human --auth <AUTH> [--project <ID>] [--base-url <URL>]
   ```
   Then tell the user to run `/reload-plugins` in this session (and any other open tabs).
