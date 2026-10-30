---
name: vertex
description: Route coding, multi-file edits, debugging, and second-opinion reviews from Codex or Claude Code to Gemini (Gemini Enterprise subscription via standalone agy CLI, or AI Studio, Vertex AI, and AI gateways via the Antigravity SDK MCP server).
---

# Gemini & Antigravity Delegation (`cc-agy-plugin`)

Use the `agy` (Codex) or `vertex` (Claude Code) MCP tools to hand hands-on coding and investigation to Gemini (`gemini-flash-next` and `gemini-4-argon`) inside an isolated copy of the repository.

If the MCP tools (`delegate_start`, `delegate_file_batch`, `ask_gemini`, `doctor`) are not registered in Codex yet, ask the user to run `python3 -m server.install --host codex` once from the plugin checkout (add `--auth gemini-enterprise` to install the standalone `agy` CLI and use an included Gemini Enterprise seat via Google sign-in).

## Pick the right tool first

<!-- SHARED:ROUTING_TABLE:BEGIN -->
| Change | Route |
|---|---|
| 1 to 5 files with a known local fix | Edit directly on the Lead outside strict mode; delegate with `delegate_start` when repo exploration, test debugging, or strict mode is active. |
| 6+ files with a clear per-file plan | Read at most 1 to 2 interface files, write `/tmp/<task>-plan.json`, and call `delegate_file_batch`. |
| Open-ended investigation, unknown files, or failing test suite | Delegate on Turn 1 or Turn 2 with `delegate_start` (or `ask_gemini` for a fast single-shot review without a worker copy). |
| One new doc, tests from named cases, or one edit repeated across files | Call `draft_document`, `write_tests_from_cases`, or `replicate_edit`. |
<!-- SHARED:ROUTING_TABLE:END -->

For zero-token readiness diagnostics or workspace spend checks, call `doctor()` or `cost_report()`.

## Cost and safety rules

1. **Delegate early, do not read the repo twice.** Dispatch on Turn 1 or Turn 2. The plugin automatically gives the Gemini worker a cached Repository Map and line-numbered file outlines.
2. **State outcomes, not implementation lines.** Never paste more than 8 consecutive lines of code into `prompt` or `plan`. Pass concrete requirements in `checklist` and targeted test commands in `acceptance` (such as `["pytest -q tests/test_api.py"]`).
3. **Check `verification` before applying.**
   - `verification="acceptance_tests"` with `scope_violations=[]`: explicit tests passed in the isolated copy. Call `delegate_apply(job_id=...)` directly without re-running the same tests.
   - `verification` is `"smoke_only"` (syntax/compile checks only), `"related_tests"` (existing tests discovered by the plugin passed, with no explicit acceptance check), or `"unverified"` (nothing ran): review `diff_preview` or `get_diff(job_id=...)` against every requirement yourself before `delegate_apply`.
4. **Stop when escalated or capped.** If a digest returns `escalation.escalate_to_lead=true` (two failed worker turns or a `429`/`503` quota limit) or `status="budget_exceeded"` (`CC_AGY_MAX_DAILY_CENTS`), stop delegating and finish the task directly on the host agent.
5. **One-call undo.** `delegate_revert(job_id=...)` restores every file touched by an applied job.
