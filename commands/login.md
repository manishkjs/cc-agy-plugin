---
name: login
description: Sign in to the Antigravity CLI (agy) with your Gemini Enterprise account so cc-agy-plugin can use your seat
---

Run this with the Bash tool, using a 600000 ms timeout because it waits for the user to finish signing in:

```bash
bash "${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/cc-agy-plugin/repo}/scripts/agy_signin.sh" --wait 540
```

What the script does:
- If agy is already signed in, it says so and exits.
- On macOS it opens a new Terminal window running `agy`. Tell the user, in one short line, to look for that window, pick "Use a Google Cloud project", finish the browser sign-in with their licensed Gemini Enterprise work account, then type `/exit`.
- On other systems with no terminal attached it prints the exact command to run. Show that command to the user and tell them to run it in a terminal, or to type `! <command>` here.

When it finishes, run the `doctor` tool on the `vertex` MCP server and report whether `agy_cli.auth_configured` is true and the `tier_id`. Never print tokens.
