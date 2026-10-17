---
name: vertex
description: Route coding, debugging, and architecture tasks from Claude Code to Google AI Studio or Vertex AI Gemini models via the Antigravity SDK and google-genai. Use when delegating implementation work to an isolated repo copy or in_place worker, requesting a fast second-opinion code review, inspecting worker diffs, or auditing token usage and cost receipts.
---

# Gemini & Antigravity Delegation Skill (`cc-agy-plugin`)

Use the `vertex` MCP server tools to split work between two execution tiers using either a **Google AI Studio API Key** (`GEMINI_API_KEY` / `api_key`, `auth_mode="api_key"`) or **Google Cloud Vertex AI** (`GOOGLE_CLOUD_PROJECT` / `project`, `auth_mode="vertex"`):

1. **Single-Shot Review Tier (`ask_gemini`)**:
   - Calls Gemini directly via `google-genai` without loading the Antigravity SDK's built-in agent system prompt and tool declarations.
   - Use for architecture reviews, root-cause hypotheses, and second opinions on diffs or specific `context_files`.
   - Default model: `gemini-2.5-flash` (override via `model=...` or `CC_AGY_DEFAULT_ASK_MODEL`).

2. **Agentic Worker Tier (`delegate_start`, `delegate_result`, `get_diff`, `delegate_apply`, `delegate_continue`, `delegate_cancel`, `delegate_to_gemini`)**:
   - Spawns `google.antigravity.Agent` with `vertex=False` (`GeminiAPIEndpoint`) in API-key mode or `vertex=True` (`VertexEndpoint`) in Vertex AI mode.
   - Default model: `gemini-3.1-pro-preview` (or pass `gemini-3.5-flash-lite`, `gemini-3.8-flash`, or `gemini-4-argon`).
   - By default (`in_place=false`), runs inside a standalone git repository copy under `<data_dir parent>/.cc-agy-worktrees-<slug>/<job_id>` (falling back to `/tmp/cc-agy-worktrees-<slug>/<job_id>`) initialized with `git init -b main`. Failed or timed-out runs leave your working tree untouched. Note: this provides git-level checkout isolation, not an OS container sandbox; worker commands execute with your user account's OS authority.
   - With `in_place=true`, edits your live working tree directly (requires a clean tracked tree, one `in_place` job at a time, and no `allowed_paths`), unwinds any worker commits or branch switches back to the baseline commit while keeping file edits on disk, and provides a shell-quoted `undo_hint`.
   - When `ctx-harness` (Straitjacket) is installed, replaces unbounded `RUN_COMMAND` with `run_command`, `ctx_get`, and `ctx_search` (capturing command outputs exceeding 240 lines or 4,000 tokens in a local evidence store).
   - Always pass outcome-oriented prompts (never paste more than 8 consecutive lines of implementation code into `prompt` or `plan`).
   - Pass `plan` (up to 15 outcome steps), `acceptance` (up to 5 read-only check commands such as `["pytest -q"]`), `max_tool_calls` (default `100`), `max_total_tokens` (default `5,000,000`), and `allowed_paths` when delegating scoped tasks.
   - Every run writes an HMAC-SHA256 signed receipt in `$CC_AGY_DATA_DIR` / `$CLAUDE_PLUGIN_DATA` (or `~/.cache/cc-agy-plugin/<workspace-slug>/receipts/<job_id>.json`).

## Standard Workflow

1. **Start one or more workers**:
   - Call `delegate_start(prompt=..., plan=[...], acceptance=["pytest -q"], allowed_paths=["src/module.py", "tests/test_module.py"])`.
2. **Wait for completion**:
   - Call `delegate_result(job_id=...)` to inspect the compact digest (`status`, `acceptance_passed`, `acceptance_results`, `progress`, `files_changed`, `diff_tokens_estimated`, `usage`, `cost`).
   - If `status` is `"checkpoint"` (budget cap reached) or `"needs_input"` (`report_to_lead` question/blocker), call `delegate_continue(job_id=..., follow_up_prompt=...)` to resume or redirect.
3. **Inspect the diff**:
   - Call `get_diff(job_id=..., page=1, page_tokens=4000)` to review the worker's unified git diff within a 4,000-token page budget.
4. **Verify and apply**:
   - For isolated jobs (`in_place=false`), call `delegate_apply(job_id=...)` once satisfied. Because the plugin already executed your `acceptance` commands and returned `acceptance_passed`, you do not need to re-run the same test commands unless doing final integration checks.
5. **Audit workspace spend**:
   - Call `delegate_status()` to view cumulative prompt, cached, output, and thinking token counts and USD/cents cost across all jobs for the active workspace.
