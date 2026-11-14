---
description: Talk directly to Antigravity (agy) for fast Q&A with conversation memory or isolated code changes
---

Handle the following request directly through the `agy` MCP server on Turn 1 without pre-reading repository files on the Lead:

Request: $ARGUMENTS

Routing:
1. **Questions, explanations, design checks, or code review** (read-only):
   - Call `agy_ask_gemini` (or `await tools.agy.ask_gemini(...)` in OpenCode Code Mode) with `prompt` and any explicitly named `context_files`.
   - Follow-up `/agy` questions in the same OpenCode session automatically resume the prior `agy` conversation (`agy_conversation_resumed=true`).
   - Relay `agy`'s answer directly and concisely with `model`, `auth_mode`, and token usage.
2. **File edits, bug fixes, refactors, or test generation**:
   - Call `agy_delegate_to_gemini` with `prompt`, `allowed_paths`, and read-only `acceptance` commands (`auto_apply=true`) so `agy` works inside an isolated, OS-sandboxed copy and verifies checks before merging.
   - Report `job_id`, `model`, `auth_mode`, `applied_files`, verification status, and billed cost (`0.00¢` on included Gemini Enterprise seats).
