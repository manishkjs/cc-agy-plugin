---
description: Sign in to Gemini Enterprise for cc-agy-plugin, or repair a saved sign-in that agy rejects (pass --force for a fresh sign-in)
---

Run the sign-in helper (keep `--force` if the user asked for a fresh sign-in):

```bash
ROOT="${CC_AGY_INSTALL_REPO_DIR:-}"
for candidate in "$ROOT" "$HOME/.local/share/cc-agy-plugin/current" "$HOME/.local/share/cc-agy-plugin/repo"; do
  [ -f "$candidate/scripts/py.sh" ] && ROOT="$candidate" && break
done
sh "$ROOT/scripts/py.sh" "$ROOT/server/agy_login.py" --start --json $ARGUMENTS
```

On a local Mac or desktop this may open a browser window and wait for the user to sign in there.

1. `"already_signed_in": true` or `"signed_in": true`: done. Call `agy_doctor` on the `agy` MCP server.
2. `"auth_url"` present: show `short_url` on its own line if set, then the full `auth_url` on its own
   line, unindented. Only if `clipboard_requested` is true, add "it may also be on your clipboard".
   `"reused": true` means the earlier link is still valid. Ask the user to sign in with their Gemini
   Enterprise work account, click **Copy to Clipboard** on the Antigravity page, and paste the code.
   Then run the `complete_command` from the JSON with the code in place of `<PASTE_CODE>`, show its
   `[OK]`, `[NOTE]` or `[FAIL]` line as written, and call `agy_doctor`.
3. `"ok": false`: show the `error` and stop.

Never echo raw tokens. Never tell the user to sign in with `agy -p`.
