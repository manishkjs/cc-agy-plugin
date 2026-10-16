# `cc-agy-plugin`: Claude Code Plugin for Vertex AI & Google Antigravity SDK

`cc-agy-plugin` is a native Claude Code plugin that lets Claude Code delegate coding, debugging, and review tasks to Vertex AI Gemini models through a long-lived Python stdio Model Context Protocol (MCP) server.

## Architecture

1. **Two Execution Tiers on Vertex AI**:
   - **`ask_gemini` (`server/genai_ask.py`)**: Single-shot Vertex AI call via `google-genai` (`genai.Client(vertexai=True, ...)`). Avoids the 11,554-token per-turn system prompt floor of the full Antigravity agent loop when asking for code reviews or architectural second opinions.
   - **`delegate_start` / `delegate_result` / `get_diff` / `delegate_apply` / `delegate_continue` / `delegate_cancel` / `delegate_to_gemini` (`server/jobs.py`, `server/agy_runner.py`)**: Runs `google.antigravity.Agent` (`vertex=True` on `LocalAgentConfig`) inside a standalone isolated repository copy per job.
2. **Standalone Repository Isolation & Conflict-Safe Apply**:
   - Every worker job runs in its own standalone repository copy under `<data_dir>/worktrees/<job_id>` with its own independent `.git` directory and baseline commit SHA (so workers do not share `.git/hooks` or `.git/config` with the host repository, and host `.git/hooks` and `.git/config` are snapshot-verified around every worker run).
   - Diffs are collected against the recorded baseline commit SHA with `git diff --binary --no-renames <baseline_sha>`, capturing worker commits, file renames/deletions, untracked files, and binary diffs while blocking symlinks that point outside the worktree.
   - `delegate_apply(job_id)` verifies each modified file's host SHA-256 against the job's baseline hashes (falling back to `git apply --3way --binary` or aborting on conflict so uncommitted host edits or parallel worker edits are never silently overwritten).
3. **Crash-Safe `finally:` Receipts, HMAC Signing, and Scrubbing (`server/receipts.py`)**:
   - Every `ask_gemini` and `delegate_*` call writes an HMAC-SHA256 signed JSON receipt inside a `finally:` block, preserving partial token counts (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`, `thoughts_token_count`) and tool-call logs even when a turn times out or raises an exception.
   - `scrub_receipt_for_export()` (also available via `scrub_receipt=true` on MCP tool calls) redacts paying GCP project IDs (`<paying-project>`) and local host paths (`/workspace`).
4. **Automatic Per-Model Vertex Region Routing (`server/regions.py`)**:
   - Global-only models that return `404 NOT_FOUND` on regional endpoints (`gemini-3.5-flash-lite`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview`, `gemini-3-pro-preview`) are automatically routed to `location="global"` with an audit note on the receipt.
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
│   └── hooks.json           # PreToolUse, PostToolUse, and Stop hook registrations
├── scripts/
│   ├── pre_tool_guard.py    # Strict mode PreToolUse guard + 9-line dictation linter
│   ├── post_tool_ledger.py  # PostToolUse MCP audit logger
│   └── provenance_check.py  # Stop-hook git diff vs HMAC-receipt provenance verifier
└── tests/
    └── test_cc_agy_plugin.py
```

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```
