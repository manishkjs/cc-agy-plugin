---
name: agy
description: Talk directly to Antigravity (agy / Gemini) for fast Q&A, codebase questions, or delegated edits with conversation memory
argument-hint: <question or coding task>
---

Handle the following request directly through the `agy` (or `vertex` in Claude Code) MCP server without unnecessary Lead file pre-reading:

Request: $ARGUMENTS

Routing:
1. **Questions, explanations, architecture/code review, or quick second opinions** (no repository file edits needed):
   - Call `ask_gemini(prompt=..., context_files=[...])` immediately on Turn 1.
   - On Gemini Enterprise (`agy` CLI), follow-up `/agy` questions in the same session automatically resume the existing `agy` conversation (`agy_conversation_resumed=true`) so context is not re-sent.
   - Present Gemini's answer directly and concisely, noting the `model`, `auth_mode`, and token usage.
2. **Code changes, refactors, bug fixes, or test authoring**:
   - Call `delegate_to_gemini(prompt=..., allowed_paths=[...], acceptance=[...], auto_apply=true)` (or `delegate_file_batch` for 6+ planned files) immediately on Turn 1 so `agy` executes in an isolated, OS-sandboxed checkout and verifies acceptance checks before applying.
   - Report the `job_id`, `model`, `auth_mode`, modified files, verification status, and billed cost (`0.00¢` on included Gemini Enterprise seats).
