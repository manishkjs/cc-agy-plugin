---
description: Workspace cost. Pass --true-total to add your own Claude Code session (rebuilt from transcripts) to the Gemini spend the plugin dispatched.
---

Without arguments: call `delegate_status` (no `job_id`) on the `vertex` MCP server and report `totals`:
- Jobs (`completed_jobs`, `in_progress_or_paused_jobs`, `failed_or_timed_out_jobs`, `applied_jobs`)
- Tokens (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`, `thoughts_token_count`, `total_token_count`)
- Per-model and total cost in cents and USD (including partial usage from paused, failed or timed-out runs)

With `--true-total` (arguments: $ARGUMENTS): call `cost_report` on the `vertex` MCP server and report three numbers side by side, never blended:
- **Dispatched**: `dispatched.cost_cents` (what the plugin sent to Gemini and Claude writers, from signed receipts)
- **Lead session**: `lead.lead_cost_cents` with its per-model rows (your Claude Code session, rebuilt from its transcripts)
- **True total**: `true_total_cost_cents` and `lead_share_of_true_total`

Say plainly that setups must be compared on true total, not dispatched cost. Mention any `lead.notes` (unpriced models, missing transcripts).
