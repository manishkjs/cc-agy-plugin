---
description: Sign in to Gemini Enterprise for cc-agy-plugin, or repair a saved sign-in that agy rejects (pass --force for a fresh sign-in)
---

Before running the command below, tell the user:
"A browser window will open for Google sign-in; finish it there. This waits up to 90 seconds."

Then run the sign-in helper (keep `--force` if the user asked for a fresh sign-in):

```bash
ROOT="${CC_AGY_INSTALL_REPO_DIR:-}"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
for candidate in "$ROOT" "$DATA/cc-agy-plugin/current" "$DATA/cc-agy-plugin/repo"; do
  [ -f "$candidate/scripts/py.sh" ] && ROOT="$candidate" && break
done
sh "$ROOT/scripts/py.sh" "$ROOT/server/agy_login.py" --start --json $ARGUMENTS
```

1. `"already_signed_in": true` or `"signed_in": true`: done. Call `agy_doctor` on the `agy` MCP server.
2. `"auth_url"` present: show `short_url` on its own line if set, then the full `auth_url` on its own
   line, unindented. Only if `clipboard_requested` is true, add "it may also be on your clipboard".
   `"reused": true` means the earlier link is still valid. Show any `note` first (why the browser
   step was skipped). Ask the user to sign in with their Gemini
   Enterprise work account, click **Copy to Clipboard** on the Antigravity page, and paste the code.
   Then run the `complete_command` from the JSON with the code in place of `<PASTE_CODE>`, show its
   `[OK]`, `[NOTE]` or `[FAIL]` line as written, and call `agy_doctor`.
3. `"ok": false`: show the `error` and stop.

Never echo raw tokens. Never tell the user to sign in with `agy -p`.
