# FAQ

Short answers to the questions that come up most.

## Do I have to change how I prompt?

No. All five hosts load the plugin's routing skill automatically, so you describe the change in plain English and your lead agent decides when to delegate. The [example prompts](usage.md#plain-english-prompts-you-can-type-right-away) show what that looks like.

## Is the isolated copy a sandbox?

Two layers protect your machine:
1. **Git worktree isolation (every backend):** Workers edit a standalone git copy of your repository so unverified edits never touch your working tree. Credential environment variables, SSH agent sockets, and git credential helpers are scrubbed from the child environment, and HMAC-SHA256 receipts protect provenance and pre-apply backups.
2. **OS-level filesystem sandbox (`agy` CLI workers):** On Linux (`bwrap`) and macOS (`sandbox-exec`), `agy` workers run inside an OS filesystem sandbox where `/` is read-only, `/tmp` and `~/.cache` are private tmpfs mounts, `~/.ssh`, `~/.gnupg`, cloud credential folders, `.netrc`, `.git-credentials`, sibling worktrees, and the plugin receipt store (`.hmac_key`) are masked (including symlink targets), and `~/.gemini/antigravity-cli` config/MCP/plugin files are mounted read-only. Pass `--worker-sandbox required` (`CC_AGY_WORKER_SANDBOX=required`) to refuse execution when neither OS sandbox backend is available.

## What if I do not want anything applied without my review?

Pass `--no-auto-apply` (or set `CC_AGY_AUTO_APPLY_ON_PASS=0`). Workers still run in disposable git copies, `in_place=true` and shutdown salvage are disabled, and nothing touches your repo until `delegate_apply`.

## How do I undo a job?

Ask your agent to revert it by `job_id`, which calls `delegate_revert`. It restores every file the job touched from its signed pre-apply backup. If any of those files changed after the apply, it refuses without writing. `skip_dirty=true` reverts only the clean files, and `dry_run=true` previews the result.

## How do I cap spend?

Set `--max-daily-cents` (for example `500` for 500¢ a day) to enforce a UTC daily spend cap for the workspace. Use `--profile economy` to keep every worker call on `gemini-4-hydrogen`. `/cc-agy-plugin:cost` in Claude Code and `/cost` in OpenCode report what has been spent.

## What happens when Gemini returns a quota error?

Worker `429` and `503` quota errors stop automatic retries immediately and hand control back to your lead agent without burning retry turns.

## Are my API keys written to disk?

No. Secrets are never written to config files, and API keys and gateway tokens are redacted from receipts.

## Where do receipts live?

In `$CC_AGY_DATA_DIR` or `$CLAUDE_PLUGIN_DATA`, defaulting to `~/.cache/cc-agy-plugin/<workspace-slug>/receipts/<job_id>.json`.

## Which Gemini models does it use?

`gemini-4-hydrogen` by default. Under `balanced`, a failed check moves the job to `gemini-4-carbon-zero` (and `prewalk=true` runs Turn 1 planning plus the first edit on `gemini-4-carbon` before handing off to `gemini-4-hydrogen`). Under `quality`, jobs start on `gemini-4-carbon`. You can override the model per call or with `CC_AGY_DEFAULT_DELEGATE_MODEL` and `CC_AGY_DEFAULT_ASK_MODEL`.

## Why is the plugin's token total higher than the one `agy` prints?

The plugin counts cached prompt tokens and `agy` doesn't. In `agy`'s JSON output, `input_tokens` leaves out cached tokens and `total_tokens` is input plus output only, so a turn that read 8,113 tokens from the prompt cache shows 8,113 fewer tokens there. The plugin's total is uncached input + cached input + output (thinking included once), the same way the Gemini API counts `usage_metadata`. Cached tokens are listed on their own (`cached_content_token_count`) and are priced at the lower cached rate, so the cost in cents is not inflated.

## Why does the plugin skip untracked files in my home folder?

When Claude Code or another host starts in your home folder and that folder is a git repo, every file under it counts as untracked. Listing them all would take longer than the host waits for the plugin to start, so in the home folder (or a drive root) the plugin's baseline and provenance checks look at tracked files only. Start your host from a project folder to get the full checks, and to give delegated jobs code to work on.

[Back to the README](../README.md)
