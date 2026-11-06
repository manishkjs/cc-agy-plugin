---
name: cc-agy-plugin
description: Route coding, multi-file edits, debugging, and second-opinion reviews from OpenCode to Gemini (Gemini Enterprise subscription via standalone agy CLI, or AI Studio, Vertex AI, and AI gateways via the Antigravity SDK MCP server). Use when delegating 6+ file plans (agy_delegate_file_batch), repo investigation or bug fixes (agy_delegate_start), single-shot reviews (agy_ask_gemini), readiness checks (agy_doctor), or workspace spend audits (agy_cost_report).
license: Apache-2.0
compatibility: opencode
---

# Gemini & Antigravity Delegation (`cc-agy-plugin`)

Use the `agy` MCP server tools in OpenCode (exposed as `agy_<tool>`, such as `agy_delegate_start`, `agy_delegate_file_batch`, `agy_ask_gemini`, and `agy_doctor`) to run implementation and test loops on Gemini (`gemini-flash-next` with automatic step-up to `gemini-4-carbon-zero` under `balanced`, or `gemini-4-argon` under `quality`) inside an isolated git copy of the repository.

If the `agy` MCP tools are not registered yet, ask the user to run `python3 -m server.install --host opencode` once from the plugin checkout (add `--auth gemini-enterprise` to install the standalone `agy` CLI and use an included Gemini Enterprise seat via Google sign-in).

## Pick the right tool first

<!-- SHARED:ROUTING_TABLE:BEGIN -->
| Change | Route |
|---|---|
| 1 to 5 files with a known local fix | Edit directly on the Lead outside strict mode; delegate with `delegate_start` when repo exploration, test debugging, or strict mode is active. |
| 6+ files with a clear per-file plan | Read at most 1 to 2 interface files, write `/tmp/<task>-plan.json`, and call `delegate_file_batch`. |
| Open-ended investigation, unknown files, or failing test suite | Delegate on Turn 1 or Turn 2 with `delegate_start` (or `ask_gemini` for a fast single-shot review without a worker copy). |
| One new doc, tests from named cases, or one edit repeated across files | Call `draft_document`, `write_tests_from_cases`, or `replicate_edit`. |
<!-- SHARED:ROUTING_TABLE:END -->

In OpenCode, call each MCP tool with its server prefix (for example `agy_doctor`, `agy_delegate_start`, `agy_delegate_result`, `agy_get_diff`, `agy_delegate_apply`, `agy_delegate_file_batch`, `agy_ask_gemini`, and `agy_cost_report`).

## Cost and safety rules

1. **Delegate early, do not read the repo twice.** Dispatch on Turn 1 or Turn 2. The plugin automatically injects a cached Repository Map and line-numbered file outlines into the Gemini worker's first turn (default worker tool-call cap: `55` on Flash in `balanced`, `100` in `economy` and on Argon).
2. **State outcomes, not implementation lines.** Never paste more than 8 consecutive lines of code into `prompt` or `plan`. Pass concrete requirements in `checklist` and targeted read-only test commands in `acceptance` (such as `["pytest -q tests/test_api.py"]`).
3. **Check `verification` before applying.**
   - `verification="acceptance_tests"` with `scope_violations=[]`: explicit behavioral tests passed in the isolated copy. Call `agy_delegate_apply(job_id=...)` directly without re-running the same tests.
   - `verification` is `"smoke_only"` (syntax or compile checks only), `"related_tests"` (discovered tests passed with no explicit acceptance command), or `"unverified"`: inspect `diff_preview` or `agy_get_diff(job_id=...)` against every requirement before calling `agy_delegate_apply`.
4. **Stop when escalated or capped.** If a digest returns `escalation.escalate_to_lead=true` (two failed worker turns or a `429`/`503` quota error) or `status="budget_exceeded"` (`CC_AGY_MAX_DAILY_CENTS`), stop delegating and finish the task directly in OpenCode.
5. **One-call undo.** `agy_delegate_revert(job_id=...)` restores every file touched by an applied job from its HMAC-signed pre-apply backup.
