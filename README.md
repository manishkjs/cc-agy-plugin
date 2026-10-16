# `cc-agy-plugin`: Claude Code Plugin for Vertex AI & Google Antigravity SDK

`cc-agy-plugin` is a native Claude Code plugin that lets Claude Code delegate coding, debugging, and review tasks to Vertex AI Gemini models through a long-lived Python stdio Model Context Protocol (MCP) server.

> **Status:** early. The unit test suite (45 tests) covers the MCP server, receipt provenance, Straitjacket shell containment, and both Google AI Studio API Key and Vertex AI authentication modes, and the plugin has been verified end to end on multi-turn coding tasks with Claude Opus 5.5 and Gemini 3.5 Flash-Lite.

## Why use this: Opus plans, Gemini does the grunt work

Most of the tokens in a coding session are grunt work: reading files, writing boilerplate, fixing lint, running tests, retrying. Paying flagship rates for that is the expensive part. This plugin splits the job in two:

- **Claude Opus is the lead.** It reads the task, decides what needs to change, writes a short outcome-only brief, reviews the diff, and decides whether to apply it.
- **A cheaper Gemini model is the worker.** It does the file reading, editing, and test running inside an isolated copy of your repo, then hands back a diff.

Output tokens are where flagship models cost the most, so the plugin keeps Opus from writing the code itself. In strict mode, Opus's direct file edits are blocked, and a pre-flight linter rejects delegation briefs that paste more than 8 consecutive lines of code.

### Back-of-envelope example

List prices per 1M tokens at the time of writing (check the current Anthropic and Vertex AI pricing pages before relying on these):

| Model | Input | Output |
|---|---|---|
| Claude Opus 5.5 | $4.00 | $20.00 |
| Gemini 4 Argon | $2.00 | $12.00 |
| Gemini 3.8 Flash | $0.50 | $3.00 |
| Gemini 3.5 Flash-Lite | $0.10 | $0.40 |

Take a task that needs 1M input tokens and 100k output tokens of agent work (the input figure includes the Antigravity SDK's ~11.5k-token system prompt on every worker turn). Assume Opus spends 100k input and 10k output tokens on briefing and reviewing.

| Setup | Opus cost | Worker cost | Total | vs. all-Opus |
|---|---|---|---|---|
| All Opus | $4.00 + $2.00 | none | **$6.00** | baseline |
| Opus lead + Gemini 4 Argon worker | $0.40 + $0.20 | $2.00 + $1.20 | **$3.80** | about 37% less |
| Opus lead + Gemini 3.8 Flash worker | $0.40 + $0.20 | $0.50 + $0.30 | **$1.40** | about 77% less |
| Opus lead + Gemini 3.5 Flash-Lite worker | $0.40 + $0.20 | $0.10 + $0.04 | **$0.74** | about 88% less |

These numbers are illustrative, not measured. Prompt caching lowers the all-Opus figure, a weaker worker may need extra rounds, and quality varies by task. Every call writes a receipt with real token counts, so `/cc-agy-plugin:cost` shows what you actually spent and you can measure the split on your own work.

## Quick start

### Option A: Google AI Studio API Key (Simplest Setup)

If you have a Gemini API key from [Google AI Studio](https://aistudio.google.com), you do not need a Google Cloud project, billing account, or `gcloud` CLI:

```bash
# Python packages for the MCP server (it runs with the python3 on your PATH)
python3 -m pip install google-genai pydantic google-antigravity

# Set your Gemini API key (or configure userConfig.gemini_api_key in Claude Code)
export GEMINI_API_KEY="your-api-key-here"

# Start Claude Code in a git repo with the plugin loaded
cd /path/to/your/repo
claude --plugin-dir /path/to/cc-agy-plugin
```

### Option B: Google Cloud Vertex AI (Service Account / Application Default Credentials)

If you prefer using Vertex AI with enterprise billing or service accounts:

```bash
# Python packages for the MCP server (it runs with the python3 on your PATH)
python3 -m pip install google-genai pydantic google-antigravity

# Google Cloud credentials for the Gemini calls
gcloud auth application-default login

# Start Claude Code in a git repo with the plugin loaded
cd /path/to/your/repo
claude --plugin-dir /path/to/cc-agy-plugin
```

Claude Code prompts for the plugin settings (Google AI Studio API key, GCP project, region, default models, strict mode). Run `/mcp` to confirm the `vertex` server is connected, then try a small task such as "use delegate_start to add a hello() function in hello.py, then show me get_diff".

## Isolated copy vs. `in_place`

By default every job runs on a **copy** of your repo. Think of it as handing a contractor a photocopy: they mark up the copy, you review the diff (`get_diff`), and only `delegate_apply` pastes the changes into your real files. This is safest when you want to review first, run several jobs at once, or keep editing while Gemini works. The trade-off is that the copy skips gitignored folders such as `node_modules/` and `.venv/`, so the worker may not be able to run your tests.

Pass `in_place=true` to `delegate_start` or `delegate_to_gemini` to let the worker edit your real working tree directly. It is faster on big repos and the worker sees your installed dependencies. To keep it safe:

- The tracked files must be clean (no uncommitted edits), so git can always undo the worker.
- Only one `in_place` job runs at a time, and `allowed_paths` is refused because it can't be enforced on live files.
- If the worker commits, its commits are undone (`git reset --mixed`) with the changes kept on disk, and your index is restored.
- The job counts as applied when it finishes, even after an error or timeout. The result includes an `undo_hint` with the exact `git checkout` / `rm` commands to revert.
- `delegate_continue` works on an `in_place` job only if nothing else in the tree changed since the last turn.

## Architecture

1. **Flexible Dual Authentication (Google AI Studio API Key or Vertex AI)**:
   - **`ask_gemini` (`server/genai_ask.py`)**: Single-shot call via `google-genai` (`genai.Client(api_key=...)` or `genai.Client(vertexai=True, ...)`). Avoids the 11,554-token per-turn system prompt floor of the full Antigravity agent loop when asking for code reviews or architectural second opinions.
   - **`delegate_start` / `delegate_result` / `get_diff` / `delegate_apply` / `delegate_continue` / `delegate_cancel` / `delegate_to_gemini` (`server/jobs.py`, `server/agy_runner.py`)**: Runs `google.antigravity.Agent` using `GeminiAPIEndpoint` with your API key or `VertexEndpoint` with Vertex AI project credentials inside a standalone isolated repository copy per job.
2. **Standalone Repository Isolation & Conflict-Safe Apply**:
   - Every worker job runs in its own standalone repository copy under `<data_dir>/worktrees/<job_id>` with its own independent `.git` directory and baseline commit SHA (so workers do not share `.git/hooks` or `.git/config` with the host repository, and host `.git/hooks` and `.git/config` are snapshot-verified around every worker run).
   - Diffs are collected against the recorded baseline commit SHA with `git diff --binary --no-renames <baseline_sha>`, capturing worker commits, file renames/deletions, untracked files, and binary diffs while blocking symlinks that point outside the worktree.
   - `delegate_apply(job_id)` verifies each modified file's host SHA-256 against the job's baseline hashes (falling back to `git apply --3way --binary` or aborting on conflict so uncommitted host edits or parallel worker edits are never silently overwritten).
3. **Crash-Safe `finally:` Receipts, HMAC Signing, and Scrubbing (`server/receipts.py`)**:
   - Every `ask_gemini` and `delegate_*` call writes an HMAC-SHA256 signed JSON receipt inside a `finally:` block, preserving partial token counts (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`, `thoughts_token_count`) and tool-call logs even when a turn times out or raises an exception.
   - `scrub_receipt_for_export()` (also available via `scrub_receipt=true` on MCP tool calls) redacts paying GCP project IDs (`<paying-project>`) and local host paths (`/workspace`).
4. **Automatic Per-Model Vertex Region Routing (`server/regions.py`)**:
   - Global-only models that return `404 NOT_FOUND` on regional endpoints (`gemini-4-argon`, `argon`, `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview`, `gemini-3-pro-preview`) are automatically routed to `location="global"` with an audit note on the receipt.
5. **Strict Orchestrator Mode & Provenance Verification (`scripts/pre_tool_guard.py`, `scripts/provenance_check.py`)**:
   - `classify_path()` anchors the project root via `CLAUDE_PROJECT_DIR` and `git rev-parse --show-toplevel`, and checks `out_dir` and `workdir` **before** `/tmp` so repositories located under `/tmp/...` on Linux are protected as `"workspace"` paths rather than exempted as scratch paths.
   - `hooks/hooks.json` registers two `PreToolUse` matcher groups: one for `Edit|Write|MultiEdit|NotebookEdit|Bash` and one for `mcp__plugin_cc-agy-plugin_vertex__delegate_.*|mcp__vertex__delegate_.*` so the 9-line code dictation linter runs pre-flight before any Vertex worker is billed.
   - `verify_workspace_provenance()` on the `Stop` hook uses a temporary git index copy (`git add -N .` + `git diff --binary --no-renames HEAD`) to verify added lines, deleted lines, deleted files, untracked files, and binary files against HMAC-verified applied worker receipts and the pre-existing workspace baseline.

## Plugin Layout

```text
cc-agy-plugin/
├── .claude-plugin/
│   └── plugin.json          # Claude Code plugin manifest + userConfig schema
├── .mcp.json                # Long-lived stdio MCP server definition (600s timeout)
├── server/
│   ├── main.py              # MCP stdio entry point
│   ├── mcp_stdio.py         # Concurrent JSON-RPC 2.0 MCP protocol dispatcher
│   ├── regions.py           # Model-to-region routing, pricing table, UTF-8 token pagination
│   ├── receipts.py          # Atomic finally-safe HMAC receipt store & diff provenance hashing
│   ├── genai_ask.py         # Single-shot google-genai Vertex helper
│   ├── agy_runner.py        # Antigravity SDK async context-manager runner
│   └── jobs.py              # Standalone repo isolation, conflict-safe apply, and pagination
├── skills/
│   └── vertex/
│       └── SKILL.md         # /cc-agy-plugin:vertex skill instructions
├── agents/
│   └── gemini-delegator.md  # Orchestrator subagent (disallowedTools: Write, Edit, ...)
├── commands/
│   ├── delegate.md          # /cc-agy-plugin:delegate slash command
│   ├── jobs.md              # /cc-agy-plugin:jobs slash command
│   └── cost.md              # /cc-agy-plugin:cost slash command
├── hooks/
│   └── hooks.json           # SessionStart, PreToolUse, PostToolUse, and Stop hook registrations
├── scripts/
│   ├── session_start.py     # SessionStart hook: records the per-session baseline
│   ├── pre_tool_guard.py    # Strict mode PreToolUse guard + 9-line dictation linter
│   ├── post_tool_ledger.py  # PostToolUse MCP audit logger
│   └── provenance_check.py  # Stop-hook git diff vs HMAC-receipt provenance verifier
└── tests/
    └── test_cc_agy_plugin.py
LICENSE                      # Apache License 2.0
```

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

## Acknowledgments

Special thanks to Vamsi Ramakrishnan ([github.com/vamsiramakrishnan/straitjacket](https://github.com/vamsiramakrishnan/straitjacket)) for building Straitjacket, whose bounded command execution and addressable evidence retrieval architecture inspired and powers our worker containment pipeline.
