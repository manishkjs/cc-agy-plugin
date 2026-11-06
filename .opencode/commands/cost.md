---
description: Report workspace Gemini token usage and cost in cents from signed receipts
---

Arguments: $ARGUMENTS

Without `--true-total`: call `agy_delegate_status` (no `job_id`) on the `agy` MCP server and report `totals`:
- Jobs (`completed_jobs`, `in_progress_or_paused_jobs`, `failed_or_timed_out_jobs`, `applied_jobs`)
- Tokens (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`, `thoughts_token_count`, `total_token_count`)
- Per-model and cumulative worker spend in cents (`¢`), including partial usage from paused, failed, or timed-out runs

With `--true-total`: call `agy_cost_report` on the `agy` MCP server and report the figures side by side, never blended:
- **Dispatched**: `dispatched.cost_cents` (every Gemini worker, batch writer, and `ask_gemini` call recorded in signed workspace receipts)
- **Lead session**: `lead.lead_cost_cents` when Claude Code transcripts exist for the workspace, plus any `lead.notes` (in OpenCode-only sessions without `~/.claude/projects/` transcripts, note that OpenCode lead token usage is tracked in OpenCode's own session stats rather than `~/.claude/projects/`)
- **True total**: `true_total_cost_cents` and `lead_share_of_true_total`
