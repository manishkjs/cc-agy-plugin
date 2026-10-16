---
description: Delegate a coding task to an isolated Vertex AI Gemini worker via the Antigravity SDK
argument-hint: <task description>
---

Delegate the following task to an isolated Vertex AI Gemini worker using the `vertex` MCP server:

Task: $ARGUMENTS

Steps:
1. Formulate a concise, outcome-oriented prompt (at most 8 consecutive lines of code if any snippet is needed) and identify the target `allowed_paths`.
2. Call `delegate_start` (or `delegate_to_gemini` with `auto_apply=false`) on the `vertex` MCP server.
3. Call `get_diff` to review the worker's unified diff in token-bounded pages.
4. If the diff is clean and within scope, call `delegate_apply` and run the relevant unit tests to verify.
5. Report the `job_id`, modified files, token usage (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`, `thoughts_token_count`), and total cost.
