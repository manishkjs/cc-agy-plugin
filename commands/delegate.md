---
description: Delegate a coding task to a Google AI Studio or Vertex AI Gemini worker via the Antigravity SDK
argument-hint: <task description>
---

Delegate the following task to a Gemini worker using the `vertex` MCP server:

Task: $ARGUMENTS

Steps:
1. Dispatch on Turn 1 or 2 (do not pre-read 10+ repository files on the Lead): formulate a concise, outcome-oriented prompt (at most 8 consecutive lines of code if any snippet is needed), ordered `plan` steps, read-only `acceptance` commands (such as `["pytest -q"]`), and target `allowed_paths`. For 6+ files with a clear per-file plan, write `/tmp/<task>-plan.json` and call `delegate_file_batch`.
2. For agent tasks, call `delegate_start` (or `delegate_to_gemini` with `model="gemini-3.8-flash"`, `preload_files=true` (file paths only), and `auto_apply=false` for the Flash pipeline on an existing repo) on the `vertex` MCP server. The worker automatically receives the cached repository map and Straitjacket (`ctx-harness`) bounded tools.
3. Wait with `delegate_result` (if `status` is `"checkpoint"` or `"needs_input"`, guide the worker with `delegate_continue`). If `acceptance_passed=true` and `scope_violations=[]`, call `delegate_apply` directly, or pass `file_filter="<path>"` to `get_diff` only for files you need to spot-check.
4. Confirm `acceptance_passed` and, for isolated jobs (`in_place=false`), call `delegate_apply` to merge the verified changes.
5. Report the `job_id`, `auth_mode` (`api_key` or `vertex`), `model`, modified files, token usage (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`, `thoughts_token_count`), and total cost in USD and cents.
