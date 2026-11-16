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
2. If it prints a `https://accounts.google.com/...` URL:
   - Display that clickable URL verbatim to the user right away (never truncate query parameters).
   - Ask the user (via `AskUserQuestion` or a direct 1-line prompt) to click the link, sign in with their Gemini Enterprise account, and paste the authorization code (or callback URL) back in chat.
   - As soon as the user replies with the code, run:
     ```bash
     python3 "${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/cc-agy-plugin/repo}/server/agy_login.py" --complete "<PASTE_CODE_OR_URL>"
     ```
   - Then call `doctor` on the `vertex` MCP server and report `status` and `tier_id` (never echo raw tokens).
