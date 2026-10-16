# `cc-agy-plugin`: Claude Code Plugin for Vertex AI & Google Antigravity SDK

`cc-agy-plugin` is a native Claude Code plugin that lets Claude Code delegate coding, debugging, and review tasks to Vertex AI Gemini models through a long-lived Python stdio Model Context Protocol (MCP) server.

## Architecture

1. **Two Execution Tiers on Vertex AI**:
   - **`ask_gemini` (`server/genai_ask.py`)**: Single-shot Vertex AI call via `google-genai` (`genai.Client(vertexai=True, ...)`). Avoids the 11,554-token per-turn system prompt floor of the full Antigravity agent loop when asking for code reviews or architectural second opinions.
   - **`delegate_start` / `delegate_result` / `get_diff` / `delegate_apply` / `delegate_continue` / `delegate_cancel` / `delegate_to_gemini` (`server/jobs.py`, `server/agy_runner.py`)**: Runs `google.antigravity.Agent` (`vertex=True` on `LocalAgentConfig`) inside an isolated temporary `git worktree` per job.
2. **Isolated Git Worktree Sandboxing**:
   - Every worker job runs in its own detached git worktree under `.cc-agy-data/worktrees/<job_id>`.
   - Failed, timed-out, or out-of-scope (`allowed_paths`) worker runs never dirty the user's main working tree.
   - Only `delegate_apply(job_id)` copies verified changes into the main workspace and records SHA-256 per-file added-line hashes on the receipt.
3. **Crash-Safe `finally:` Receipts & Scrubbing (`server/receipts.py`)**:
   - Every `ask_gemini` and `delegate_*` call writes its JSON receipt inside a `finally:` block, preserving partial token counts (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`, `thoughts_token_count`) and tool-call logs even when a turn times out or raises an exception.
   - `scrub_receipt_for_export()` redacts paying GCP project IDs (`<paying-project>`) and local host paths (`/workspace`) before receipts are shared.
4. **Automatic Per-Model Vertex Region Routing (`server/regions.py`)**:
   - Preview and 3.5-series models that return `404 NOT_FOUND` on regional endpoints (`gemini-3.5-flash-lite`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview`) are automatically routed to `location="global"` with an audit note on the receipt.
5. **Strict Orchestrator Mode & Provenance Verification (`scripts/pre_tool_guard.py`, `scripts/provenance_check.py`)**:
   - `classify_path()` checks `out_dir` and `workdir` **before** `/tmp` so repositories located under `/tmp/...` on Linux are protected as `"workspace"` paths rather than exempted as scratch paths.
   - `detect_bash_workspace_mutation()` blocks direct host tree edits via shell redirects, `sed -i`, `patch`, `git apply/checkout/restore/reset/clean/stash`, `cp`, `mv`, `rm`, `touch`, `install`, `rsync`, `ln`, `truncate`, and inline interpreters (`python -c`, `node -e`, `perl -i`, `ruby -e`).
   - `verify_workspace_provenance()` on the `Stop` hook compares live `git diff HEAD` added-line SHA-256 hashes against applied worker receipts so unverified host edits are caught regardless of how they were made.

## Plugin Layout

```text
cc-agy-plugin/
├── .claude-plugin/
│   └── plugin.json          # Claude Code plugin manifest + userConfig schema
├── .mcp.json                # Long-lived stdio MCP server definition (600s timeout)
├── server/
│   ├── main.py              # MCP stdio entry point
│   ├── mcp_stdio.py         # JSON-RPC 2.0 MCP protocol dispatcher
│   ├── regions.py           # Model-to-region routing, pricing table, token budget caps
│   ├── receipts.py          # Atomic finally-safe receipt store & diff provenance hashing
│   ├── genai_ask.py         # Single-shot google-genai Vertex helper
│   ├── agy_runner.py        # Antigravity SDK async context-manager runner
│   └── jobs.py              # Git worktree job isolation, pagination, and apply logic
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
│   └── provenance_check.py  # Stop-hook git diff HEAD vs receipt provenance verifier
└── tests/
    └── test_cc_agy_plugin.py
```

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```
