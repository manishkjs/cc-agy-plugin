---
name: vertex
description: Route work from Claude Code to Gemini (AI Studio or Vertex AI). Use for multi-file changes with a clear plan (delegate_file_batch), read-only investigation (delegate_start), drafting docs, writing tests from cases, repeating one edit across files, second-opinion reviews, and auditing true cost.
---

# Gemini & Antigravity Delegation Skill (`cc-agy-plugin`)

Use the `vertex` MCP server tools to split work between two execution tiers using either a **Google AI Studio API Key** (`GEMINI_API_KEY` / `api_key`, `auth_mode="api_key"`) or **Google Cloud Vertex AI** (`GOOGLE_CLOUD_PROJECT` / `project`, `auth_mode="vertex"`):

1. **Single-Shot Review Tier (`ask_gemini`)**:
   - Calls Gemini directly via `google-genai` without loading the Antigravity SDK's built-in agent system prompt and tool declarations.
   - Use for architecture reviews, root-cause hypotheses, and second opinions on diffs or specific `context_files`.
   - Default model: `gemini-2.5-flash` (override via `model=...` or `CC_AGY_DEFAULT_ASK_MODEL`).

2. **Agentic Worker Tier (`delegate_start`, `delegate_result`, `get_diff`, `delegate_apply`, `delegate_revert`, `delegate_continue`, `delegate_cancel`, `delegate_to_gemini`, `cost_report`)**:
   - Spawns `google.antigravity.Agent` with `vertex=False` (`GeminiAPIEndpoint`) in API-key mode or `vertex=True` (`VertexEndpoint`) in Vertex AI mode.
   - Default model: `gemini-3.1-pro-preview` (or pass `gemini-3.5-flash-lite`, `gemini-3.8-flash`, or `gemini-4-argon`).
   - By default (`in_place=false`), runs inside a standalone git repository copy under `<data_dir parent>/.cc-agy-worktrees-<slug>/<job_id>` (falling back to `/tmp/cc-agy-worktrees-<slug>/<job_id>`) initialized with `git init -b main`. Failed or timed-out runs leave your working tree untouched. Note: this provides git-level checkout isolation, not an OS container sandbox; worker commands execute with your user account's OS authority.
   - With `in_place=true`, edits your live working tree directly (requires a clean tracked tree, one `in_place` job at a time, and no `allowed_paths`), unwinds any worker commits or branch switches back to the baseline commit while keeping file edits on disk, and provides a shell-quoted `undo_hint`.
   - When `ctx-harness` (Straitjacket) is installed, replaces unbounded `RUN_COMMAND` with `run_command`, `ctx_get`, and `ctx_search` (capturing command outputs exceeding 240 lines or 4,000 tokens in a local evidence store).
   - Always pass outcome-oriented prompts (never paste more than 8 consecutive lines of implementation code into `prompt` or `plan`).
   - Pass `plan` (up to 15 outcome steps), `acceptance` (up to 5 read-only check commands such as `["pytest -q"]`), `max_tool_calls` (default `100`), `max_total_tokens` (default `5,000,000`), and `allowed_paths` when delegating scoped tasks.
   - Every run writes an HMAC-SHA256 signed receipt in `$CC_AGY_DATA_DIR` / `$CLAUDE_PLUGIN_DATA` (or `~/.cache/cc-agy-plugin/<workspace-slug>/receipts/<job_id>.json`).

## Pick the route first

| Change | Route |
|---|---|
| 1 to 5 files (known cause) | Do it yourself. Delegation costs more than it saves at this size. |
| 6+ files with a clear per-file plan | Read at most 1 to 2 interface files, write `/tmp/<task>-plan.json`, and call `delegate_file_batch`. |
| Read-only investigation, unknown file list, debugging a failing test suite | Delegate on Turn 1 or 2 with `delegate_start` (never read 10+ files yourself before delegating). |
| Mid-chat: one new doc, tests from named cases, one edit repeated across files | `draft_document`, `write_tests_from_cases`, `replicate_edit`. |

### Lead Cost Discipline (Turn 1 to 2 Delegation Rule)

- **Never pay twice for reading the repo.** When delegating, dispatch on Turn 1 or Turn 2 from the user's request plus at most 1 to 2 interface reads. Preserve all behavioral requirements, edge cases, and invariants in `delegate_start(prompt=...)` without writing out line-by-line implementation code. The plugin automatically injects a cached line-numbered Repository Map (`## Repository Map (Cached)`) and a Focused File Outline (`## Focused File Outline (Line-Numbered)`) into the worker's first turn.
- **Pass targeted acceptance commands.** In `acceptance`, specify the focused test file(s) covering the modified module (for example `["pytest -q tests/test_module.py"]`) rather than a slow full-repository test suite, so the worker verifies and exits as soon as the target tests pass.
- **Do not page the full diff or re-run Bash tests when checks pass.** Both `delegate_file_batch` and `delegate_start`/`delegate_result` return a compact structured receipt with `diff_preview`, `acceptance_passed`, and `scope_violations`. When `acceptance_passed=true` and `scope_violations=[]`, call `delegate_apply(job_id=...)` directly and finish; do not spend extra turns calling `get_diff` or re-running `pytest` via `Bash`.

## Batch file writing (`delegate_file_batch`)

1. **Plan in a file, not in chat.** Write the plan JSON with `Write` to a path outside the repo (for example `/tmp/<task>-plan.json`) so the per-file details never sit in your context:
   ```json
   {"shared_context": ["src/models.py"],
    "acceptance": ["python -m pytest -q"],
    "tasks": [
      {"task_id": "store", "path": "src/store.py", "action": "create",
       "intent": "SQLite-backed store for Order with get/put/list.",
       "interface": "from src.store import OrderStore  # OrderStore(path: str).put(order) -> None",
       "checks": ["python -m py_compile src/store.py"]},
      {"task_id": "api", "path": "src/api.py", "action": "modify", "depends_on": ["store"],
       "intent": "Add GET /orders/{id} using OrderStore.", "context_files": ["src/store.py"],
       "checks": ["python -m pytest -q tests/test_api.py"]}
    ]}
   ```
   - `intent` is the outcome in prose (no pasted code). `interface` is the exact import or signature other files rely on, so writers working in parallel agree on how files connect.
   - `depends_on` orders tasks; `repro_test` (optional) is a test that must fail before the change and pass after it.
   - Checks and acceptance must be read-only test or lint commands. Tasks that only run syntax/compile checks are listed in `syntax_only_tasks` in the digest.
2. **Run it:** `delegate_file_batch(plan_path="/tmp/<task>-plan.json")`. Pass `model="mechanical"` (`gemini-3.5-flash-lite`) for purely mechanical file orders, or `read_budget=6` on messy brownfield repos where writers need up to 6 read-only lookups. A bad plan comes back as `status: "invalid_plan"` with every problem and its JSON path; fix them all in one edit.
3. **Read the digest:** each task is `landed`, `unchanged` (already done in a previous run), `failed` (with a short `check_tail`) or `held` (a dependency failed). Re-sending the same plan resumes the job and skips finished tasks at no cost.
4. **Land it:** If `acceptance_passed=true` and `review_required`/`syntax_only_tasks` are empty, call `delegate_apply(job_id=...)` directly; otherwise spot-check with `get_diff(job_id=..., file_filter="...")` before `delegate_apply(job_id=...)`. Fix a failed task yourself, or adjust its intent and re-send.
5. **Cost:** writers default to `gemini-3.8-flash` with LOW thinking (`gemini-3.5-flash-lite` when `model="mechanical"`); the third attempt goes to Claude (`claude -p`, no tools). Run `probe_writers` once per session to catch auth problems before a batch.

## Agent worker workflow (`delegate_start`)

1. **Start one or more workers**:
   - Call `delegate_start(prompt=..., plan=[...], acceptance=["pytest -q"], allowed_paths=["src/module.py", "tests/test_module.py"])`.
2. **Wait for completion**:
   - Call `delegate_result(job_id=...)` to inspect the compact digest (`status`, `acceptance_passed`, `acceptance_results`, `progress`, `files_changed`, `diff_tokens_estimated`, `usage`, `cost`).
   - If `status` is `"checkpoint"` (budget cap reached) or `"needs_input"` (`report_to_lead` question/blocker), call `delegate_continue(job_id=..., follow_up_prompt=...)` to resume or redirect.
3. **Inspect the diff**:
   - Call `get_diff(job_id=..., page=1, page_tokens=4000)` to review the worker's unified git diff within a 4,000-token page budget.
4. **Verify and apply**:
   - For isolated jobs (`in_place=false`), call `delegate_apply(job_id=...)` once satisfied. Because the plugin already executed your `acceptance` commands and returned `acceptance_passed`, you do not need to re-run the same test commands unless doing final integration checks.
5. **Existing-repo Flash pipeline (`preload_files` + optional `qa_review_model`)**:
   - On existing repositories, pass `model="gemini-3.8-flash"` and `preload_files=True` to `delegate_to_gemini`, and give it `acceptance` tests that catch misses: Flash is much cheaper per attempt than Opus but solves fewer hard tasks on its own.
   - `preload_files=True` lists the top keyword-matched file paths (paths only, not contents) in Turn 1. Full file contents in Turn 1 get re-read on every later turn, which inflates cost.
   - `qa_review_model="gemini-4-argon"` is experimental: a single-shot HIGH-thinking diff audit (diff capped at 25,000 tokens) that runs `delegate_continue` as a fresh worker session on the edited worktree if it finds defects. It is not yet shown to raise solve rate.
   - Do not ask Gemini to write spec tests before implementation (they are often wrong), and do not hand a Flash draft to Opus to finish (you pay for the work twice).
6. **Audit workspace spend**:
   - Call `delegate_status()` to view cumulative prompt, cached, output, and thinking token counts and USD/cents cost across all jobs for the active workspace.
   - Call `cost_report()` for the true total: dispatched Gemini cost from receipts plus your own Claude Code session rebuilt from its transcripts. Report savings on the true total, never on dispatched cost alone.
7. **When the worker keeps failing**:
   - If a digest carries `escalation.escalate_to_lead=true` (two failed turns in a row), stop delegating that task. Inspect `get_diff(job_id=...)` and finish it yourself, or `delegate_cancel` and re-plan. Only pass `force_retry=true` to `delegate_continue` with a materially different instruction.
8. **Undo an applied job**:
   - `delegate_revert(job_id=..., dry_run=true)` shows what would be restored; drop `dry_run` to restore. It refuses if files changed after the apply; `skip_dirty=true` reverts only the clean files.
