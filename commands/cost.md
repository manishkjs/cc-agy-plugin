---
description: Summarize cumulative Gemini token usage and cost across all receipts for the current workspace
---

Call `delegate_status` (with no `job_id`) on the `vertex` MCP server and report the cumulative workspace token and cost ledger from `totals`:
- Total jobs (`completed_jobs`, `in_progress_or_paused_jobs`, `failed_or_timed_out_jobs`, `applied_jobs`)
- Token breakdown (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`, `thoughts_token_count`, `total_token_count`)
- Per-model and total cost in cents and USD (including partial usage captured from paused, failed, or timed-out runs)
