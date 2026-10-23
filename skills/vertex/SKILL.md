---
name: vertex
description: Route work from Claude Code to Gemini (Gemini Enterprise subscription, AI Studio, Vertex AI, or an AI gateway). Use for multi-file changes with a clear plan (delegate_file_batch), read-only investigation (delegate_start), drafting docs, writing tests from cases, repeating one edit across files, second-opinion reviews, and auditing true cost.
---

# Gemini & Antigravity Delegation Skill (`cc-agy-plugin`)

Use the `vertex` MCP server tools to split work between two execution tiers using your **Gemini Enterprise subscription** (`CC_AGY_AUTH_MODE=gemini_enterprise` via the standalone Antigravity CLI `agy` Google sign-in, or auto-detected when signed into `agy` with no API key or GCP project set), a **Google AI Studio API Key** (`GEMINI_API_KEY` / `api_key`, `auth_mode="api_key"`), **Google Cloud Vertex AI** (`GOOGLE_CLOUD_PROJECT` / `project`, `auth_mode="vertex"`), or a **Vertex-compatible AI Gateway** (`CC_AGY_VERTEX_BASE_URL`):

1. **Single-Shot Review Tier (`ask_gemini`)**:
   - Calls Gemini directly via `google-genai` (or `agy --output-format json --mode plan` on Gemini Enterprise seats) without loading an agent workspace copy.
   - Use for architecture reviews, root-cause hypotheses, and second opinions on diffs or specific `context_files`.
   - Default model: `gemini-3.8-flash` (`gemini-4-argon` when `cost_profile="quality"`, or override via `model=...` / `CC_AGY_DEFAULT_ASK_MODEL`).

2. **Agentic Worker Tier (`delegate_start`, `delegate_result`, `get_diff`, `delegate_apply`, `delegate_revert`, `delegate_continue`, `delegate_cancel`, `delegate_to_gemini`, `cost_report`, `doctor`)**:
   - Routes automatically to the standalone Antigravity CLI (`agy --output-format stream-json`) on Gemini Enterprise subscriptions, or spawns `google.antigravity.Agent` (`GeminiAPIEndpoint` in API-key mode, `VertexEndpoint` in Vertex AI and AI Gateway modes).
   - Default model: `gemini-3.8-flash` under the default `balanced` cost profile, escalating automatically to `gemini-4-argon` on retries, failed checks, or overlapping follow-up fixes (`cost_profile="quality"` starts directly on `gemini-4-argon`; `cost_profile="economy"` stays on `gemini-3.8-flash`).
   - By default (`in_place=false`), runs inside a standalone git repository copy under `<data_dir parent>/.cc-agy-worktrees-<slug>/<job_id>` (falling back to `/tmp/cc-agy-worktrees-<slug>/<job_id>`) initialized with `git init -b main`. When `CC_AGY_SPARSE_WORKTREE=1` or `CC_AGY_SPARSE_THRESHOLD_FILES=<N>` is set on large monorepos, passing `allowed_paths` materializes a scoped sparse copy (including 2-hop local imports) so worktree setup stays fast without marking omitted files as deleted. Failed or timed-out runs leave your working tree untouched. Note: this provides git-level checkout isolation, not an OS container sandbox; worker commands execute with your user account's OS authority.
   - With `in_place=true`, edits your live working tree directly (requires a clean tracked tree, one `in_place` job at a time, and no `allowed_paths`), unwinds any worker commits or branch switches back to the baseline commit while keeping file edits on disk, and provides a shell-quoted `undo_hint`.
   - When `ctx-harness` (Straitjacket) is installed, replaces unbounded `RUN_COMMAND` with `run_command`, `ctx_get`, and `ctx_search` (capturing command outputs exceeding 400 lines or 6,000 tokens in a local evidence store).
   - Always pass outcome-oriented prompts (never paste more than 8 consecutive lines of implementation code into `prompt` or `plan`).
   - Pass `plan` (up to 15 outcome steps), `acceptance` (up to 5 read-only check commands such as `["pytest -q"]`, `["go test ./..."]`, `["cargo test"]`, or `["npx vitest run"]`), `max_tool_calls` (default `40` on Flash, `100` on Argon), `max_total_tokens` (default `5,000,000`), and `allowed_paths` when delegating scoped tasks.
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
- **Do not page the full diff or re-run Bash tests when checks pass.** Both `delegate_file_batch` and `delegate_start`/`delegate_result` return a compact structured receipt with `diff_preview`, `acceptance_passed`, `verification`, and `scope_violations`. When `verification="acceptance_tests"` (an explicit acceptance command passed) and `scope_violations=[]`, call `delegate_apply(job_id=...)` directly and finish; do not spend extra turns calling `get_diff` or re-running `pytest` via `Bash`. When `verification` is `smoke_only`, `related_tests`, or `unverified` (for example in a repository without tests), review `diff_preview` against every requirement before applying or finishing.

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
2. **Run it:** `delegate_file_batch(plan_path="/tmp/<task>-plan.json")`. Pass `read_budget=6` on brownfield repos where writers need up to 6 read-only lookups, or `model="gemini-4-argon"` for harder file specs. A bad plan comes back as `status: "invalid_plan"` with every problem and its JSON path; fix them all in one edit.
3. **Read the digest:** each task is `landed`, `unchanged` (already done in a previous run), `failed` (with a short `check_tail`) or `held` (a dependency failed). Re-sending the same plan resumes the job and skips finished tasks at no cost.
4. **Land it:** If `acceptance_passed=true` and `review_required`/`syntax_only_tasks` are empty, call `delegate_apply(job_id=...)` directly; otherwise spot-check with `get_diff(job_id=..., file_filter="...")` before `delegate_apply(job_id=...)`. Fix a failed task yourself, or adjust its intent and re-send.
5. **Cost:** writers default to `gemini-3.8-flash` with LOW thinking; the third attempt goes to Claude (`claude -p`, no tools). Run `doctor` for a zero-cost offline readiness check or `probe_writers` once per session to verify live writer auth before a batch.

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
