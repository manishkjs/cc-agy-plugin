# `cc-agy-plugin`: Call Google Antigravity SDK from Claude Code via MCP

![Architecture: Claude Code calling Google Antigravity SDK via MCP](assets/architecture.svg)

`cc-agy-plugin` connects two coding agent harnesses over the Model Context Protocol (MCP): **Claude Code** acts as the lead engineer, while **Google Antigravity SDK** (powered by Gemini 3.5 Flash-Lite, Gemini 3.8 Flash, or Gemini 4 Argon) acts as the execution worker inside an isolated copy of your repository.

> **Status:** 45 unit tests cover the MCP stdio server, HMAC receipt provenance, Straitjacket shell containment, and both Google AI Studio API Key and Vertex AI authentication modes. Verified end to end on multi-turn coding tasks with Claude Opus 5.5 and Gemini 3.5 Flash-Lite.

---

## The Value Proposition: Why Connect Two Harnesses?

In a typical coding session, **90% of tokens go to mechanical execution**: reading files, grepping symbols, editing boilerplate, running test suites, reading tracebacks, and retrying. Paying flagship model rates ($4/$20 per 1M tokens) for every test loop is expensive, while running an inexpensive model alone can miss subtle architectural constraints.

`cc-agy-plugin` splits the work by role:

1. **Claude Opus 5.5 plans and reviews (~10% of session tokens)**:
   - Reads the user prompt, inspects key interfaces, and writes a short outcome-only task brief with ordered plan steps and read-only acceptance checks (`pytest`, `ruff`, `mypy`).
   - Reviews the paginated git diff returned by the worker and decides whether to merge it (`delegate_apply`) or send follow-up instructions (`delegate_continue`).
2. **Google Antigravity SDK + Gemini executes (~90% of session tokens)**:
   - Spins up a headless Antigravity agent inside an isolated git worktree copy (or `in_place` when requested).
   - Performs the file reads, code edits, and iterative test runs at Gemini rates ($0.10 to $2.00 per 1M input tokens).
   - **Straitjacket (`ctx`) shell containment** intercepts noisy test and build outputs (>240 lines or >16 KB), stores them in a local evidence store (`run:<id>#stdout`), and gives the worker `ctx_get` and `ctx_search` tools so a single failing test suite never floods the worker's context window.
3. **Your main working tree stays clean until verified**:
   - Failed worker runs never pollute your checkout. Changes are only merged when `delegate_apply` verifies baseline SHA-256 hashes (using a 3-way merge when needed) and records a cryptographic HMAC-SHA256 receipt.

### Cost Comparison (1M Input + 100k Output Worker Tokens)

List prices per 1M tokens:

| Model | Role | Input / 1M | Output / 1M |
|---|---|---|---|
| **Claude Opus 5.5** | Lead Planner & Reviewer | $4.00 | $20.00 |
| **Gemini 4 Argon** | Frontier Reasoning Worker | $2.00 | $12.00 |
| **Gemini 3.8 Flash** | Fast Daily Coding Worker | $0.50 | $3.00 |
| **Gemini 3.5 Flash-Lite** | High-Volume Execution Worker | $0.10 | $0.40 |

For a task requiring **1M input tokens and 100k output tokens** of worker execution (plus **100k input and 10k output tokens** for Opus to brief and review):

| Workflow | Lead Cost (Opus) | Worker Cost (Gemini) | Total Session Cost | Savings vs. All-Opus |
|---|---|---|---|---|
| **All Claude Opus 5.5** | $4.00 + $2.00 | $0.00 | **$6.00** | Baseline |
| **Opus Lead + Gemini 4 Argon** | $0.40 + $0.20 | $2.00 + $1.20 | **$3.80** | **37% less** |
| **Opus Lead + Gemini 3.8 Flash** | $0.40 + $0.20 | $0.50 + $0.30 | **$1.40** | **77% less** |
| **Opus Lead + Gemini 3.5 Flash-Lite** | $0.40 + $0.20 | $0.10 + $0.04 | **$0.74** | **88% less** |

Every tool call writes a signed receipt with actual token usage, and `/cc-agy-plugin:cost` reports exact session spend in USD and cents.

---

## Quick Start

### Option A: Google AI Studio API Key (Fastest Setup)

If you have a Gemini API key from [Google AI Studio](https://aistudio.google.com), no Google Cloud project or `gcloud` CLI is required:

```bash
# 1. Install Python dependencies for the MCP server
python3 -m pip install google-genai pydantic google-antigravity

# 2. Export your Gemini API key (or set userConfig.gemini_api_key in Claude Code)
export GEMINI_API_KEY="your-api-key-here"

# 3. Launch Claude Code with the plugin loaded
cd /path/to/your/repo
claude --plugin-dir /path/to/cc-agy-plugin
```

### Option B: Google Cloud Vertex AI (ADC / Service Account)

If you use Google Cloud enterprise billing or service accounts:

```bash
# 1. Install Python dependencies for the MCP server
python3 -m pip install google-genai pydantic google-antigravity

# 2. Authenticate with Google Cloud Application Default Credentials
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# 3. Launch Claude Code with the plugin loaded
cd /path/to/your/repo
claude --plugin-dir /path/to/cc-agy-plugin
```

Inside Claude Code, run `/mcp` to verify the `vertex` MCP server is connected, then try:
> *"Use `delegate_start` with `gemini-3.5-flash-lite` to add a `hello()` function in `hello.py`, inspect the change with `get_diff`, and apply it with `delegate_apply`."*

---

## How One Harness Calls the Other (End-to-End Flow)

1. **Hook Pre-Flight Check (`scripts/pre_tool_guard.py`)**:
   - In strict mode (`CC_AGY_STRICT_MODE=true`), Claude Code's direct `Edit`/`Write` tools on workspace files are blocked so Opus delegates implementation work.
   - A pre-flight linter rejects delegation prompts containing more than 8 consecutive lines of code so Opus writes concise outcome briefs instead of paying $20/1M output tokens to dictate code into the prompt.
2. **MCP Stdio Request (`server/mcp_stdio.py`)**:
   - Claude Code sends a JSON-RPC 2.0 `tools/call` message over `stdin` to the plugin's persistent Python process.
   - An 8-thread worker pool handles requests concurrently so `delegate_status`, `ping`, and `delegate_cancel` never block behind a running worker turn.
3. **Sandbox Snapshot & Worker Execution (`server/jobs.py`, `server/agy_runner.py`)**:
   - `JobManager` snapshots tracked files (`git ls-files -co --exclude-standard`) into an isolated worktree under `<data_dir>/worktrees/<job_id>` and records the baseline commit SHA.
   - `run_antigravity_worker()` launches `google.antigravity.Agent` with per-turn safety caps (default: `100` tool calls and `5,000,000` total tokens) and Straitjacket's bounded shell tools (`run_command`, `ctx_get`, `ctx_search`).
4. **Plugin-Run Acceptance Checks & Token-Bounded Diff**:
   - When the worker finishes, the plugin automatically executes any read-only `acceptance` commands (for example, `["pytest -q"]`), cleans up temporary test artifacts, and returns pass/fail counts with a compact output tail.
   - Opus inspects the changes via `get_diff` (paginated at `4,000` tokens per page so large diffs never overflow Claude Code's MCP token cap) and merges verified work with `delegate_apply`.
5. **Cryptographic Provenance Check (`scripts/provenance_check.py`)**:
   - Every run writes an HMAC-SHA256 signed receipt in `.cc-agy-data/receipts/<job_id>.json` (with API keys unconditionally redacted).
   - When Claude Code finishes its turn, the `Stop` hook verifies that every modified line in the workspace matches a signed worker receipt.

---

## MCP Tools Reference

| Tool | Purpose |
|---|---|
| `ask_gemini` | Fast single-shot query via `google-genai` (skips the 11,554-token Antigravity system prompt floor). Best for code reviews, architecture checks, and second opinions. |
| `delegate_start` | Start an Antigravity worker in an isolated repo copy (or `in_place=true`). Supports `plan`, `acceptance`, `max_tool_calls`, `max_total_tokens`, `thinking_level`, `api_key`, and `project`. |
| `delegate_result` | Wait for a background job to finish (default `240s` wait cap so Claude Code's 5-minute prompt cache stays warm) and return its compact digest. |
| `delegate_status` | Inspect live progress signals (`report_to_lead`), token usage, and USD/cents cost for a single `job_id` or list recent jobs and cumulative session totals. |
| `get_diff` | Retrieve the unified git diff from a job's worktree, filtered by file or glob and paginated by token budget (`default: 4000`, `max: 8000` tokens per page). |
| `delegate_apply` | Verify host file SHA-256 hashes against the job's baseline, apply the diff (with 3-way merge fallback), record provenance hashes on the HMAC receipt, and clean up the worktree. |
| `delegate_continue` | Send a follow-up instruction into an existing job's worktree (preserving the job's `auth_mode`, `plan`, `acceptance`, and budget caps across turns). |
| `delegate_cancel` | Cancel an in-flight worker job and remove its temporary worktree. |
| `delegate_to_gemini` | Synchronous convenience wrapper: runs `delegate_start(..., wait=true)` and automatically calls `delegate_apply` if `status == "ok"`. |

---

## Isolated Copy vs. `in_place` Mode

- **Isolated copy (default, `in_place=false`)**: Every job runs in its own standalone copy of tracked files under `<data_dir>/worktrees/<job_id>`. Multiple jobs can run in parallel, failed runs leave your working tree untouched, and host `.git/hooks` and `.git/config` are snapshot-verified around every run.
- **Direct working-tree mode (`in_place=true`)**: Edits your live checkout directly so the worker can use gitignored virtual environments (`.venv/`, `node_modules/`) without copying them. Requires a clean tracked tree, allows one `in_place` job at a time, automatically unwinds any worker git commits (`git reset --mixed`) while keeping file edits on disk, and returns an `undo_hint` command to revert if needed.

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
│   ├── genai_ask.py         # Single-shot google-genai helper (no Antigravity prompt floor)
│   ├── agy_runner.py        # Antigravity SDK runner + Straitjacket (ctx.sdk) tool containment
│   └── jobs.py              # Worktree isolation, acceptance runner, 3-way apply, and pagination
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

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

## Acknowledgments

Special thanks to Vamsi Ramakrishnan ([github.com/vamsiramakrishnan/straitjacket](https://github.com/vamsiramakrishnan/straitjacket)) for building Straitjacket, whose bounded command execution and addressable evidence retrieval architecture inspired and powers our worker containment pipeline.
