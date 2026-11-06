---
description: List active and completed Gemini delegation jobs and token spend for this workspace
---

Call `agy_delegate_status` on the `agy` MCP server (passing `job_id="$ARGUMENTS"` only when a non-empty job ID was provided) and display a concise summary for the current workspace:
- `job_id`, `kind` (`ask`, `delegate`, or `batch`), `status` (`ok`, `checkpoint`, `needs_input`, `running`, `interrupted`, `error`, `timeout`, `cancelled`), `auth_mode` (`gemini_enterprise`, `api_key`, or `vertex`), and `model`
- `vertex_location` (when `auth_mode="vertex"`, including any automatic `global` region reroute note)
- `files_changed`, `diff_tokens_estimated`, `acceptance_passed`, `verification`, `in_place`, and whether `applied` is true
- Token counts (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`, `thoughts_token_count`) and cost in cents (`¢`)
