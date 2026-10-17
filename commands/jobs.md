---
description: List active and completed Gemini delegation jobs and their workspace status
argument-hint: [optional-job-id]
---

Call `delegate_status` on the `vertex` MCP server (passing `job_id="$ARGUMENTS"` if an argument was provided) and display a concise summary of each job for the current workspace:
- `job_id`, `kind` (`ask` or `delegate`), `status` (`ok`, `checkpoint`, `needs_input`, `running`, `interrupted`, `error`, `timeout`, `cancelled`), `auth_mode` (`api_key` or `vertex`), and `model`
- `vertex_location` (when `auth_mode="vertex"`, including any automatic global region reroute note)
- `files_changed`, `diff_tokens_estimated`, `acceptance_passed`, `in_place`, and whether `applied` is true
- Token usage and cost breakdown in USD and cents
