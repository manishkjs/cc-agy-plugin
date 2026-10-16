---
description: List active and completed Vertex AI Gemini delegation jobs and their worktree status
argument-hint: [optional-job-id]
---

Call `delegate_status` on the `vertex` MCP server (passing `job_id="$ARGUMENTS"` if an argument was provided) and display a concise summary of each job:
- `job_id`, `kind` (`ask` or `delegate`), `status`, and `model`
- `vertex_location` (including any automatic global region reroute note)
- `files_changed`, `diff_tokens_estimated`, and whether `applied` is true
- Token usage and cost breakdown
