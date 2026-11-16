---
name: setup
description: Configure cc-agy-plugin connection mode, model preset, and Gemini Enterprise sign-in using interactive dropdowns
---

Use `AskUserQuestion` immediately (no preamble) to present two dropdown selections:

1. **Auth mode** (`header: "Auth mode"`): *"How should cc-agy-plugin connect to Gemini?"*
   - `Gemini Enterprise Seat (Recommended)` — Included $0 seat quota via standalone `agy` CLI
   - `Google AI Studio API Key` — Pay-as-you-go with `GEMINI_API_KEY`
   - `Google Cloud Vertex AI` — Pay-as-you-go with `GOOGLE_CLOUD_PROJECT` and ADC
   - `Enterprise AI Gateway` — Route through a LiteLLM or Bifrost gateway URL

2. **Model preset** (`header: "Preset"`): *"Which model routing preset do you want?"*
   - `Balanced (Recommended)` — Default worker `gemini-4-hydrogen`, escalates to `gemini-4-carbon`
   - `Quality` — Default worker `gemini-4-carbon`
   - `Economy` — Lowest-cost worker routing

After the user selects their options:
1. Run `server/install.py` with their mapped `--auth` (`gemini-enterprise`, `api-key`, `vertex`, or `gateway`) and `--profile` (`balanced`, `quality`, or `economy`):
   ```bash
   python3 "${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/cc-agy-plugin/repo}/server/install.py" --host claude --auth <AUTH> --profile <PROFILE>
   ```
2. If `<AUTH>` is `gemini-enterprise`, run:
   ```bash
   python3 "${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/cc-agy-plugin/repo}/server/agy_login.py"
   ```
   If it prints a sign-in URL, show BOTH the **short 1-line URL** (`https://is.gd/...` on its own line so it never wraps in `Terminal.app`) AND `[Sign in with Google](<full_url>)`, tell the user it is **already copied to their clipboard (`Cmd+V` in browser)**, ask them to click **Copy to Clipboard** on the Antigravity callback page and paste the code, and run:
   ```bash
   python3 "${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/cc-agy-plugin/repo}/server/agy_login.py" --complete "<PASTE_CODE>"
   ```
3. Run `doctor` on the `vertex` MCP server (or `python3 "${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/cc-agy-plugin/repo}/server/install.py" --doctor --human --auth <AUTH>`) and tell the user if `/reload-plugins` is needed.
