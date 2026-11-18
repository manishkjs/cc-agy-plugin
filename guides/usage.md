# Usage

Prompts, slash commands, and how changes are checked when a project has no tests.

Once installed, open your project in Claude Code, Cursor, OpenCode, Codex, or Antigravity and work normally. All five hosts load the plugin's routing skill automatically, so you do not need to memorize internal MCP tool names.

## Plain-English prompts you can type right away

| What you want to do | Example prompt | What happens behind the scenes |
|:--|:--|:--|
| **Check your setup (free)** | *"Run doctor to verify my Gemini setup."* (or `/cc-agy-plugin:doctor` in Claude Code, `/doctor` in OpenCode) | Verifies credentials, active `cost_profile`, installed SDKs, git status, and detected test runner offline at **0 tokens**. |
| **Fix a bug or build a feature** | *"Delegate adding input validation to `api/orders.py` with unit tests, review the diff, and apply it."* | Runs `gemini-4-hydrogen` in an isolated git copy (`delegate_start`), runs checks, lets your lead agent inspect `get_diff`, and merges with `delegate_apply`. |
| **Edit many files from one plan (6+ files)** | *"Plan the new webhook handlers across all 8 files and write them with `delegate_file_batch`."* | Validates the plan locally, writes files in parallel with per-file verification, retries failures automatically, and returns one combined diff. |
| **Get a fast second opinion** | *"Ask Gemini to review my current git diff for edge cases and race conditions."* | Sends a single-shot `ask_gemini` query without spinning up a worker copy or touching files. |
| **Undo an applied worker job** | *"Revert worker job `<job_id>`."* | Restores every file touched by that job from its signed pre-apply backup (`delegate_revert`) if no newer manual edits conflict. |

## Slash commands in Claude Code and OpenCode

| Claude Code | OpenCode | What it does |
|:--|:--|:--|
| `/cc-agy-plugin:setup` | n/a | Opens interactive dropdowns to choose your connection mode and model preset, then runs sign-in and `doctor`. |
| `/cc-agy-plugin:login` | `/login` | Signs in to Gemini Enterprise (agy's own browser sign-in on a local desktop, or a sign-in link plus pasted code over SSH), repairs a saved sign-in agy rejects, and records your license. `--force` starts fresh; `--project <ID>` picks the seat project. |
| `/cc-agy-plugin:agy <question or task>` | `/agy <question or task>` | Talks directly to Antigravity/Gemini: routes questions to `ask_gemini` (with conversation reuse on GE) and code edits to `delegate_to_gemini`. |
| `/cc-agy-plugin:doctor` | `/doctor` | Offline readiness check for credentials, SDKs, `cost_profile`, git workspace, and test runner (zero token cost). |
| `/cc-agy-plugin:delegate <task>` | `/delegate <task>` | Hands `<task>` to an isolated Gemini worker, verifies checks, inspects the diff, and applies clean results (you can also mention `@gemini-delegator` in OpenCode). |
| `/cc-agy-plugin:jobs` | `/jobs` | Shows running and recent worker jobs, modified files, token counts, and cost in cents. |
| `/cc-agy-plugin:cost` | `/cost` | Reports workspace Gemini spend in cents (pass `--true-total` to compare dispatched worker spend with lead session transcripts). |

## What if my project does not have tests yet?

Every worker digest reports a `verification` mode, so you always know how a change was checked.

**Projects with tests (`verification: "acceptance_tests"`).** The plugin detects and runs scoped tests across Python (`pytest`, `unittest`), TypeScript/JavaScript (`vitest`, `jest`, `node --test`), Go (`go test`), Rust (`cargo test`), Java/Kotlin (`gradlew`, `mvnw`, `mvn`), and `Makefile`.

**Projects without tests (`verification: "smoke_only"`).** When no test command is given or detected, the plugin runs fast read-only syntax and compile checks on every changed file: Python parse and import, `tsc --noEmit` with your repo's TypeScript, `node --check`, `go build`, `cargo check --offline`, `bash -n`, and JSON validation.

Files that fail to parse or compile are rejected and escalated to `gemini-4-carbon-zero` under `balanced` (`gemini-4-carbon` under `quality`). When the smoke checks pass, the digest tells your lead agent to review the diff line by line against your request before finishing.

[Back to the README](../README.md)
