# `cc-agy-plugin`: Call Google Antigravity SDK from Claude Code via MCP

![Architecture: Claude Code calling Google Antigravity SDK via MCP](assets/architecture.svg)

`cc-agy-plugin` connects two coding agent harnesses over the Model Context Protocol (MCP): **Claude Code** acts as the lead engineer, while **Google Antigravity SDK** (defaulting to `gemini-3.1-pro-preview` for worker delegations and `gemini-2.5-flash` for single-shot reviews, with support for `gemini-3.5-flash-lite`, `gemini-3.8-flash`, and `gemini-4-argon`) acts as the execution worker inside an isolated copy of your repository.

---

## Architecture & Trade-Offs: Why Connect Two Harnesses?

`cc-agy-plugin` connects Claude Code to the Google Antigravity SDK so a single workspace can combine Anthropic and Gemini models with strict worktree isolation and cryptographic diff provenance:

1. **Claude Opus 5.5 plans and reviews**:
   - Reads the user prompt, inspects key interfaces, and writes a short outcome-only task brief with ordered `plan` steps and read-only `acceptance` checks (`pytest`, `ruff`, `mypy`).
   - Reviews the token-paginated git diff returned by the worker and decides whether to merge it (`delegate_apply`) or send follow-up instructions (`delegate_continue`).
2. **Google Antigravity SDK + Gemini executes in an isolated copy**:
   - Spins up a headless Antigravity agent inside a standalone git repository copy (or `in_place` when requested).
   - **Straitjacket (`ctx-harness`) shell containment** intercepts noisy test and build outputs (>240 lines or >4,000 tokens), stores them in a local evidence store (`run:<id>#stdout`), and gives the worker `ctx_get` and `ctx_search` tools so a single failing test suite does not flood the worker's context window.
3. **Your working tree stays clean until verified (Git-Level Isolation)**:
   - In the default isolated mode (`in_place=false`), the worker edits a standalone copy of tracked files under `<data_dir parent>/.cc-agy-worktrees-<slug>/<job_id>` (falling back to `/tmp/cc-agy-worktrees-<slug>/<job_id>`), so failed or timed-out runs leave uncommitted files in your checkout untouched. Changes are only merged when `delegate_apply` verifies baseline SHA-256 hashes (using a 3-way merge when needed) and records an HMAC-SHA256 signed receipt.
   - *Security note*: Isolated copies provide git-level checkout isolation, not an OS container or VM sandbox. Worker shell commands run with your local user account's OS permissions; HMAC receipts and child-env scrubbing enforce workflow provenance and prevent accidental secret leakage, not isolation against a hostile local process running as the same OS user. The plugin snapshots and restores host `.git/hooks` and `.git/config` around every run.

### Configured Rate Card & Measured SWE Benchmark Trade-Offs

Configured model rates in `server/regions.py` per 1M tokens (for single model calls exceeding `200,000` prompt tokens on Pro/Argon models, such as `ask_gemini` with `single_request=True` or an individual per-step call, input rates scale by `2.0x` and output rates by `1.5x`; multi-call worker turns accumulate smaller per-step prompts at the standard rate):

| Model | Role | Uncached Input / 1M | Cached Input / 1M | Output / 1M |
|---|---|---|---|---|
| **Claude Opus 5.5** | Lead Planner & Reviewer (Unverified assumed proxy list rate) | $4.00 | $0.40 | $20.00 |
| **Gemini 3.1 Pro / Gemini 4 Argon** | Frontier Reasoning Worker (Default) | $2.00 | $0.20 | $12.00 |
| **Gemini 3.8 Flash** | Fast Daily Coding Worker | $0.50 | $0.05 | $3.00 |
| **Gemini 3.5 Flash-Lite** | High-Volume Execution Worker | $0.10 | $0.01 | $0.40 |
| **Gemini 2.5 Flash** | Single-Shot Review (`ask_gemini` Default) | $0.30 | $0.03 | $2.50 |

**Empirical Benchmark Findings (2 `httpx` tasks × 3 arms × 2 runs = 12 cells)**:
Although Gemini per-token rates are lower than Opus, **delegating a self-contained single-repo coding task to a nested multi-turn worker is NOT cheaper or faster than running Claude Opus 5.5 standalone**:
- When running standalone (`opus_solo`), Claude Opus 5.5 solved the tasks in **17 to 21 turns** (**263.4s to 402.0s**, **0.83M to 1.27M tokens**) with **~87% to 92% prompt cache hits ($0.40/1M)**, costing **96.43¢ to 146.32¢** per run (`3/4` passes).
- When delegating via `cc-agy-plugin`, the Opus Lead still spent **12 to 25 turns** (**98.73¢ to 202.62¢**) exploring, writing briefs, reviewing `get_diff`, and running post-apply host checks, while the Gemini Worker cold-started a second agent loop (**41 to 70 turns / 148.49¢ to 280.72¢ on `Gemini 4 Argon`**; **125 to 207 turns / 109.68¢ to 224.41¢ on `Gemini 3.8 Flash`**).
- End-to-end, the delegated arms cost **2.22x to 3.53x more** (**244.81¢ to 422.38¢**) and took **4.6x to 12.7x longer** (**1518.7s to 3529.9s**) than `opus_solo`. Use `cc-agy-plugin` when you want **isolated worktree experimentation (`wait=false`), cross-model second opinions (`ask_gemini`), or cryptographic provenance gating**, not to reduce cost or latency on self-contained tasks that Opus can solve directly in ~20 cached turns.

Every tool call writes a signed receipt with token usage, and `/cc-agy-plugin:cost` reports cumulative workspace spend in USD and cents based on the rate table in `server/regions.py`.

---

## Quick Start

### Option A: Google AI Studio API Key (Fastest Setup)

If you have a Gemini API key from [Google AI Studio](https://aistudio.google.com), no Google Cloud project or `gcloud` CLI is required:

```bash
# 1. Install the plugin and its Antigravity + Straitjacket dependencies
python3 -m pip install -e ".[antigravity]"
# (Or directly: python3 -m pip install google-genai pydantic google-antigravity ctx-harness)

# 2. Export your Gemini API key (or set userConfig.gemini_api_key in Claude Code)
export GEMINI_API_KEY="your-api-key-here"

# 3. Launch Claude Code with the plugin loaded
cd /path/to/your/repo
claude --plugin-dir /path/to/cc-agy-plugin
```

### Option B: Google Cloud Vertex AI (ADC / Service Account)

If you use Google Cloud enterprise billing or service accounts:

```bash
# 1. Install the plugin and its Antigravity + Straitjacket dependencies
python3 -m pip install -e ".[antigravity]"

# 2. Authenticate with Google Cloud Application Default Credentials
# Note: unset GEMINI_API_KEY / GOOGLE_API_KEY if set in your shell so Vertex AI is selected by default
unset GEMINI_API_KEY GOOGLE_API_KEY
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# 3. Launch Claude Code with the plugin loaded
cd /path/to/your/repo
claude --plugin-dir /path/to/cc-agy-plugin
```

### Authentication Precedence

`resolve_auth_config()` in `server/regions.py` selects the authentication mode in this order:
1. Explicit `api_key` argument on the tool call -> **Google AI Studio** (`auth_mode="api_key"`).
2. Explicit `project` argument on the tool call -> **Vertex AI** (`auth_mode="vertex"`).
3. `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `CLAUDE_PLUGIN_OPTION_GEMINI_API_KEY` in the environment -> **Google AI Studio** (`auth_mode="api_key"`, even if `GOOGLE_CLOUD_PROJECT` is also exported in the shell).
4. `GOOGLE_CLOUD_PROJECT`, `VERTEX_PROJECT`, `GCLOUD_PROJECT`, `CLOUD_ML_PROJECT_ID`, or `CLAUDE_PLUGIN_OPTION_GOOGLE_CLOUD_PROJECT` in the environment -> **Vertex AI** (`auth_mode="vertex"`).

### Default Models & Overrides

- **`delegate_start` / `delegate_to_gemini`**: defaults to `gemini-3.1-pro-preview` (override per call with `model="gemini-3.5-flash-lite"`, `model="gemini-3.8-flash"`, or `model="gemini-4-argon"`, or globally via `CC_AGY_DEFAULT_DELEGATE_MODEL` / `userConfig.default_delegate_model`).
- **`ask_gemini`**: defaults to `gemini-2.5-flash` (override per call with `model=...` or globally via `CC_AGY_DEFAULT_ASK_MODEL` / `userConfig.default_ask_model`).

Inside Claude Code, run `/mcp` to verify the `vertex` MCP server is connected, then try:
> *"Use `delegate_start` with `model="gemini-3.5-flash-lite"` to add a `hello()` function in `hello.py`, inspect the change with `get_diff`, and apply it with `delegate_apply`."*

---

## How One Harness Calls the Other (End-to-End Flow)

1. **Hook Pre-Flight Check (`scripts/pre_tool_guard.py`)**:
   - A pre-flight linter always rejects delegation prompts (or `plan` steps) containing more than 8 consecutive lines of code so the lead agent writes concise outcome briefs instead of dictating implementation code into the prompt.
   - When strict mode is enabled (`CC_AGY_STRICT_MODE=true` or `userConfig.strict_delegation_mode=true`), Claude Code's direct `Edit`/`Write`/`MultiEdit`/`NotebookEdit` tools and mutating `Bash` commands on workspace files are blocked so implementation work goes through Gemini workers.
2. **MCP Stdio Request (`server/mcp_stdio.py`)**:
   - Claude Code sends a JSON-RPC 2.0 `tools/call` message over `stdin` to the plugin's persistent Python process.
   - An 8-worker thread pool dispatches `tools/call` requests concurrently (up to 8 simultaneous in-flight tool calls) so `delegate_status`, `ping`, and `delegate_cancel` can respond while a worker job is running.
3. **Isolated Copy Snapshot & Worker Execution (`server/jobs.py`, `server/agy_runner.py`)**:
   - `JobManager` copies tracked and non-ignored files (`git ls-files -co --exclude-standard -z`) into a standalone git repository under `<data_dir parent>/.cc-agy-worktrees-<slug>/<job_id>` (falling back to `/tmp/cc-agy-worktrees-<slug>/<job_id>`) and creates a baseline commit (`git init -b main`).
   - `run_antigravity_worker()` launches `google.antigravity.Agent` with per-turn safety caps (default: `100` tool calls and `5,000,000` total tokens) and Straitjacket's bounded shell tools (`run_command`, `ctx_get`, `ctx_search`). If `ctx-harness` is not installed, the worker falls back to standard Antigravity tools and surfaces `containment: "none"` with a `containment_warning` in the job digest.
4. **Plugin-Run Acceptance Checks & Token-Bounded Diff**:
   - When the worker finishes, the plugin automatically executes any read-only `acceptance` commands (for example, `["pytest -q"]`), cleans up new untracked test artifacts, and returns pass/fail counts with a 600-token output tail.
   - The lead agent inspects changes via `get_diff` (paginated at `4,000` tokens per page, up to `8,000` max, so large diffs never exceed Claude Code's MCP token cap) and merges verified work with `delegate_apply`.
5. **Cryptographic Provenance Check (`scripts/provenance_check.py`)**:
   - Every run writes an HMAC-SHA256 signed receipt in `$CC_AGY_DATA_DIR` / `$CLAUDE_PLUGIN_DATA` (defaulting to `~/.cache/cc-agy-plugin/<workspace-slug>/receipts/<job_id>.json`), with API keys unconditionally redacted and the `.hmac_key` stored outside the worker's worktree hierarchy.
   - When strict mode (`CC_AGY_STRICT_MODE=true`) is enabled, the `Stop` hook verifies that every modified line in the workspace matches a signed worker receipt (or pre-existing session baseline).

---

## MCP Tools Reference

| Tool | Purpose |
|---|---|
| `ask_gemini` | Fast single-shot query via `google-genai` (without loading the Antigravity SDK's agent prompt and tool declarations). Best for code reviews, architecture checks, and second opinions. Default model: `gemini-2.5-flash`. |
| `delegate_start` | Start an Antigravity worker in a standalone repo copy (or `in_place=true`). Supports `plan`, `acceptance`, `max_tool_calls`, `max_total_tokens`, `thinking_level`, `api_key`, and `project`. Default model: `gemini-3.1-pro-preview`. |
| `delegate_result` | Wait for a background job to finish (default `240s` wait cap so Claude Code's 5-minute prompt cache stays warm) and return its compact digest. |
| `delegate_status` | Inspect live progress signals (`report_to_lead`), token usage, and USD/cents cost for a single `job_id`, or omit `job_id` to list recent jobs and cumulative totals for the active workspace. |
| `get_diff` | Retrieve the unified git diff from a job's worktree, filtered by file or glob and paginated by token budget (`default: 4000`, `max: 8000` tokens per page). |
| `delegate_apply` | Verify host file SHA-256 hashes against the job's baseline, apply the diff (with 3-way merge fallback), record provenance hashes on the HMAC receipt, and clean up the worktree. |
| `delegate_continue` | Send a follow-up instruction into an existing job's worktree (preserving the job's `auth_mode`, `plan`, `acceptance`, and budget caps across turns). |
| `delegate_cancel` | Cancel an in-flight worker job and remove its temporary worktree. |
| `delegate_to_gemini` | Synchronous convenience wrapper: runs `delegate_start(..., wait=true)` and automatically calls `delegate_apply` if `status == "ok"` (leaving `checkpoint` and `needs_input` jobs unapplied without flagging `is_error`). |

---

## Isolated Copy vs. `in_place` Mode

- **Isolated copy (default, `in_place=false`)**: Every job runs in its own standalone copy of tracked files under `<data_dir parent>/.cc-agy-worktrees-<slug>/<job_id>` (falling back to `/tmp/cc-agy-worktrees-<slug>/<job_id>`). Multiple jobs can run in parallel, failed runs leave your working tree untouched, and host `.git/hooks` and `.git/config` are snapshot-verified around every run.
- **Direct working-tree mode (`in_place=true`)**: Edits your live checkout directly so the worker can use gitignored virtual environments (`.venv/`, `node_modules/`) without copying them. Requires a clean tracked tree, allows one `in_place` job at a time, automatically restores your original branch and unwinds any worker git commits (`git reset --mixed`) while keeping file edits on disk, and returns a shell-quoted `undo_hint` command to revert if needed.

---

## Repository Layout

```text
cc-agy-plugin/
├── assets/
│   └── architecture.svg     # Cross-harness MCP architecture diagram
├── .claude-plugin/
│   └── plugin.json          # Claude Code plugin manifest + userConfig schema
├── .mcp.json                # Long-lived stdio MCP server definition (600s timeout)
├── server/
│   ├── main.py              # MCP stdio entry point
│   ├── mcp_stdio.py         # Concurrent JSON-RPC 2.0 MCP protocol dispatcher
│   ├── regions.py           # Dual auth (API Key / Vertex), region routing, pricing, token pagination
│   ├── receipts.py          # Atomic finally-safe HMAC receipt store, API key redaction & diff hashing
│   ├── genai_ask.py         # Single-shot google-genai helper
│   ├── agy_runner.py        # Antigravity SDK runner + Straitjacket (ctx.sdk) tool containment
│   └── jobs.py              # Isolated repo copy, acceptance runner, 3-way apply, and pagination
├── skills/
│   └── vertex/
│       └── SKILL.md         # /cc-agy-plugin:vertex skill instructions
├── agents/
│   └── gemini-delegator.md  # Orchestrator subagent (disallowedTools: Write, Edit, ...)
├── commands/
│   ├── delegate.md          # /cc-agy-plugin:delegate slash command
│   ├── jobs.md              # /cc-agy-plugin:jobs slash command
│   └── cost.md              # /cc-agy-plugin:cost slash command
├── hooks/
│   └── hooks.json           # SessionStart, PreToolUse, PostToolUse, and Stop hook registrations
├── scripts/
│   ├── session_start.py     # SessionStart hook: records the per-session baseline
│   ├── pre_tool_guard.py    # Strict mode PreToolUse guard + 9-line code dictation linter
│   ├── post_tool_ledger.py  # PostToolUse MCP audit logger
│   └── provenance_check.py  # Stop-hook git diff vs HMAC-receipt provenance verifier
├── tests/
│   └── test_cc_agy_plugin.py
└── LICENSE                  # Apache License 2.0
```

## Running Tests

```bash
pytest tests/test_cc_agy_plugin.py -v
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgments

Special thanks to Vamsi Ramakrishnan ([github.com/vamsiramakrishnan/straitjacket](https://github.com/vamsiramakrishnan/straitjacket)) for building Straitjacket, whose bounded command execution and addressable evidence retrieval architecture inspired and powers our worker containment pipeline.
