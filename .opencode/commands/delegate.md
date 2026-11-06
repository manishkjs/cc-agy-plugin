---
description: Delegate a coding task to an isolated Gemini worker via the agy MCP server
---

Delegate the following task to a Gemini worker using the `agy` MCP server:

Task: $ARGUMENTS

Steps:
1. Dispatch on Turn 1 or Turn 2 without pre-reading 10+ repository files on the Lead: formulate a concise outcome prompt (at most 8 consecutive lines of code if any snippet is needed), ordered `plan` steps, read-only `acceptance` commands (such as `["pytest -q"]`, `["go test ./..."]`, `["cargo test"]`, or `["npx vitest run"]`), and target `allowed_paths`. For 6+ files with a clear per-file plan, write `/tmp/<task>-plan.json` and call `agy_delegate_file_batch`.
2. For agent tasks, call `agy_delegate_start` on the `agy` MCP server (uses `gemini-flash-next` by default under `balanced` and steps up to `gemini-4-carbon-zero` on retries, or starts on `gemini-4-argon` when `cost_profile="quality"`). The worker automatically receives the cached repository map and Straitjacket (`ctx-harness`) bounded tools.
3. Wait with `agy_delegate_result` (if `status` is `"checkpoint"` or `"needs_input"`, guide the worker with `agy_delegate_continue`). When `verification="acceptance_tests"`, `acceptance_passed=true`, and `scope_violations=[]`, call `agy_delegate_apply` directly; when `verification` is `"smoke_only"`, `"related_tests"`, or `"unverified"`, review `diff_preview` or `agy_get_diff(file_filter=...)` first.
4. Confirm checks passed and, for isolated jobs (`in_place=false`), call `agy_delegate_apply` to merge the verified changes into your workspace.
5. Report the `job_id`, `auth_mode` (`gemini_enterprise`, `api_key`, or `vertex`), `model`, modified files, token usage (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`, `thoughts_token_count`), and worker cost in cents (`¢`).
