---
name: login
description: Sign in to Gemini Enterprise, or repair a saved sign-in that agy rejects. Pass --force to always start a fresh sign-in.
---

Run the sign-in helper. Pass through what the user typed after the command (`--force` for a fresh
sign-in, `--project <ID>` to pick the seat project):

```bash
ROOT="${CLAUDE_PLUGIN_ROOT:-}"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
for candidate in "$ROOT" "$DATA/cc-agy-plugin/current" "$DATA/cc-agy-plugin/repo"; do
  [ -f "$candidate/scripts/py.sh" ] && ROOT="$candidate" && break
done
sh "$ROOT/scripts/py.sh" "$ROOT/server/agy_login.py" --start --json $ARGUMENTS
```

On a local Mac or desktop this may open a browser window and wait for the user to sign in there.

Handle the JSON:

1. `"already_signed_in": true` or `"signed_in": true`: the sign-in works (the helper repairs a saved
   sign-in that was missing its license). Go to step 3.
2. `"auth_url"` is present:
   - If `short_url` is set, show it on its own line first.
   - Show the full `auth_url` on its own line, unindented, then `[Sign in with Google](<auth_url>)`.
   - Only if `clipboard_requested` is true, add "it may also be on your clipboard" (some terminals,
     such as macOS Terminal.app, ignore clipboard requests).
   - `"reused": true` means the link from earlier is still valid; show it again.
   - Ask the user to sign in with their Gemini Enterprise work account, click **Copy to Clipboard**
     on the Antigravity page, and paste the code here.
   - When they paste it, run the `complete_command` from the JSON with the code in place of
     `<PASTE_CODE>`. It looks up the user's license and project, so the first `agy` call works.
   - `[OK]` lines report the tier and project. Show any `[NOTE]` (other seat projects). A `[FAIL]`
     line says exactly what to do next; show it as written.
3. Verify:
   ```bash
   ROOT="${CLAUDE_PLUGIN_ROOT:-}"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
for candidate in "$ROOT" "$DATA/cc-agy-plugin/current" "$DATA/cc-agy-plugin/repo"; do
  [ -f "$candidate/scripts/py.sh" ] && ROOT="$candidate" && break
done
   sh "$ROOT/scripts/py.sh" "$ROOT/server/install.py" --doctor --human --auth gemini-enterprise
   ```
4. `"ok": false`: show the `error` and stop.

Never echo raw tokens. Never tell the user to sign in with `agy -p`: print mode gives up after
60 seconds on a headless machine.
