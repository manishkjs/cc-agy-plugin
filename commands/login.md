---
name: login
description: Sign in to Gemini Enterprise via a clickable Google OAuth URL (or automatic desktop browser sign-in)
---

Run the Gemini Enterprise login helper:

```bash
python3 "${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/cc-agy-plugin/repo}/server/agy_login.py"
```

How to handle the output:
1. If it says `Gemini Enterprise sign-in verified` or `Signed in to Gemini Enterprise`, run `doctor` on the `vertex` MCP server and confirm `status: ok` and `tier_id: gcp-enterprise`.
2. If it prints a sign-in URL:
   - Display BOTH the **short 1-line URL** (`https://is.gd/...` on its own line so it never wraps in macOS `Terminal.app`) AND the Markdown link `[Sign in with Google](<full_url>)` (for OSC 8 terminals like iTerm2/Ghostty/VS Code), and mention that the URL was **already copied to their clipboard** (`Cmd+V` in browser, or drag-select the short 20-char link / `Fn+Cmd+Double-Click` in `Terminal.app`).
   - Ask the user to sign in with their Gemini Enterprise account, click **Copy to Clipboard** on the Antigravity callback page, and paste the code back in chat.
   - As soon as the user replies with the code, run:
     ```bash
     python3 "${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/cc-agy-plugin/repo}/server/agy_login.py" --complete "<PASTE_CODE>"
     ```
   - Then call `doctor` on the `vertex` MCP server and report `status` and `tier_id` (never echo raw tokens).
