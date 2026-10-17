---
description: Delegate a coding task to a Google AI Studio or Vertex AI Gemini worker via the Antigravity SDK
argument-hint: <task description>
---

Delegate the following task to a Gemini worker using the `vertex` MCP server:

Task: $ARGUMENTS

Steps:
1. Formulate a concise, outcome-oriented prompt (at most 8 consecutive lines of code if any snippet is needed), ordered `plan` steps, read-only `acceptance` commands (such as `["pytest -q"]`), and target `allowed_paths`.
2. Call `delegate_start` (or `delegate_to_gemini` with `auto_apply=false`) on the `vertex` MCP server.
3. Wait with `delegate_result` (if `status` is `"checkpoint"` or `"needs_input"`, guide the worker with `delegate_continue`), then call `get_diff` to review the worker's unified diff in token-bounded pages.
4. Confirm `acceptance_passed` and, for isolated jobs (`in_place=false`), call `delegate_apply` to merge the verified changes.
5. Report the `job_id`, `auth_mode` (`api_key` or `vertex`), `model`, modified files, token usage (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`, `thoughts_token_count`), and total cost in USD and cents.
