---
name: vertex
description: Route coding, debugging, and architecture tasks from Claude Code to Vertex AI Gemini models via the Antigravity SDK and google-genai. Use when delegating implementation work to an isolated git worktree worker, requesting a fast second-opinion code review on Vertex AI, inspecting worker diffs, or auditing token usage and cost receipts.
---

# Vertex AI & Antigravity Delegation Skill (`cc-agy-plugin`)

Use the `vertex` MCP server tools to split work between two execution tiers on Vertex AI:

1. **Single-Shot Review Tier (`ask_gemini`)**:
   - Calls Vertex AI directly via `google-genai` without loading the 11,554-token Antigravity SDK system prompt floor.
   - Use for architecture reviews, root-cause hypotheses, and second opinions on diffs or specific files.
   - Default model: `gemini-2.5-flash` (or pass `model="gemini-3.1-pro-preview"` for deep reviews).

2. **Isolated Agentic Worker Tier (`delegate_start`, `delegate_result`, `get_diff`, `delegate_apply`, `delegate_continue`, `delegate_cancel`, `delegate_to_gemini`)**:
   - Spawns `google.antigravity.Agent` inside an isolated temporary `git worktree` with `vertex=True` on `LocalAgentConfig`.
   - Failed or timed-out worker runs leave the main workspace untouched and still record partial token and tool-call usage inside `.cc-agy-data/receipts/<job_id>.json`.
   - Always pass outcome-oriented prompts (never paste more than 8 consecutive lines of implementation code into `prompt`).
   - Pass `allowed_paths` whenever delegating parallel jobs so each worker stays inside its assigned file or directory slot.

## Standard Workflow

1. **Start one or more isolated workers**:
   - Call `delegate_start(prompt=..., allowed_paths=["src/module.py", "tests/test_module.py"])`.
2. **Wait for completion**:
   - Call `delegate_result(job_id=...)` to inspect the compact digest (`status`, `files_changed`, `diff_tokens_estimated`, `usage`, `cost`).
3. **Inspect the diff**:
   - Call `get_diff(job_id=..., page=1, page_tokens=4000)` to review the worker's unified git diff within a 4,000-token page budget.
4. **Verify and apply**:
   - Call `delegate_apply(job_id=...)` once satisfied. This copies the verified changes into the main workspace and records SHA-256 line hashes on the receipt for `Stop`-hook provenance verification.
5. **Audit session spend**:
   - Call `delegate_status()` to view cumulative prompt, cached, output, and thinking token counts and cost across all completed and failed jobs.
