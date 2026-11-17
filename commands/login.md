---
name: login
description: Sign in to Gemini Enterprise, or repair a saved sign-in that agy rejects. Pass --force to always start a fresh sign-in.
---

Run the Gemini Enterprise login helper. If the user typed `/cc-agy-plugin:login --force`, add `--force`:

```bash
python3 "${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/cc-agy-plugin/repo}/server/agy_login.py" --start --json $ARGUMENTS
```

How to handle the JSON output:

1. `"already_signed_in": true`: the saved sign-in works (the helper already repaired it if
   it was missing its license). Run `doctor` on the `vertex` MCP server and report `status`
   and `tier_id`.
2. `"auth_url"` is present:
   - If `short_url` is set, show it on its own line first. It is short enough not to wrap,
     so it stays clickable in macOS Terminal.app.
   - Then show `[Sign in with Google](<auth_url>)`.
   - If `copied_to_clipboard` is true, say the link is already on their clipboard.
   - Ask the user to sign in with their Gemini Enterprise work account, click
     **Copy to Clipboard** on the Antigravity page, and paste the code here.
   - When they paste the code, run:
     ```bash
     python3 "${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/cc-agy-plugin/repo}/server/agy_login.py" --complete "<PASTE_CODE>"
     ```
   - The helper looks up the user's license and project, so the first `agy` call after
     sign-in works. Report the `tier` and `project` it prints, then run `doctor`.
   - If it prints a `[WARN]` about no license, tell the user to set
     `CC_AGY_ENTERPRISE_PROJECT` to the project that holds their seats and run
     `/cc-agy-plugin:login --force`.
3. `"ok": false`: show the `error` and stop.

Never echo raw tokens. Never tell the user to run `agy -p` to sign in: print mode gives up
after 60 seconds on a headless machine.
