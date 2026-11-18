# Configuration

Connection modes, cost profiles, the controls that keep you in charge, Gemini Enterprise options for governed organizations, and the full table of flags and environment variables.

Three choices cover most setups: how you connect, which cost profile you run, and how much the plugin may do without asking.

## Connection modes

| How you connect | Backend | Enterprise seat? | Billing | Best fit |
|:--|:--:|:--:|:--|:--|
| **Gemini Enterprise** (`--auth gemini-enterprise`) | Standalone `agy` CLI | **Yes** | Included in monthly Enterprise Standard/Plus seat | Work accounts with assigned Gemini Enterprise seats. |
| **AI Studio key** (`GEMINI_API_KEY`) | Python SDK | No | Pay-as-you-go on AI Studio | Individual developers without Google Cloud setup. |
| **Vertex AI** (`GOOGLE_CLOUD_PROJECT`) | Python SDK | No | Pay-as-you-go on GCP (`aiplatform.googleapis.com`) | Cloud teams, Service Accounts, and CI/CD pipelines. |
| **AI gateway** (`--base-url`) | Python SDK | Gateway-defined | Managed by your gateway (`LiteLLM`, `Portkey`, `Bifrost`, Apigee) | Enterprises routing model traffic through a central proxy. |

See [Authentication precedence](#authentication-precedence) for exact credential selection order when multiple variables are set.

## Model and cost profiles (`--profile` or `CC_AGY_COST_PROFILE`)

| Profile | How models are chosen | Best when |
|:--|:--|:--|
| **`balanced`** *(default)* | Starts on fast **`gemini-4-hydrogen`**. Steps up automatically to **`gemini-4-carbon-zero`** (Carbon, zero thinking) on the same git copy if a check fails or a retry is needed, then returns to Hydrogen on the next new task. Pass `prewalk=true` on difficult tasks for a `gemini-4-carbon` -> `gemini-4-hydrogen` intra-worker handoff. Quota errors (`429` / `503`) stop automatic retries and hand control back to your lead agent. | Everyday engineering where you want Hydrogen speed with Carbon backup on hard bugs. |
| **`economy`** | Keeps all worker and `ask_gemini` calls on **`gemini-4-hydrogen`** and turns off automatic step-up to Carbon Zero-Think. | Tight budgets, boilerplate generation, docs, and well-scoped tasks. |
| **`quality`** | Starts all worker and `ask_gemini` calls directly on **`gemini-4-carbon`** with deeper reasoning from turn one. | Complex multi-file architecture bugs, subtle concurrency fixes, or tricky math/logic changes. |

## You stay in full control

Every model call, file edit, and spend ceiling is governed by explicit settings you control:

| Control | How to set it | Guarantee |
|:--|:--|:--|
| **1. Lock connection & billing mode** | `--auth gemini-enterprise\|api-key\|vertex\|gateway` (`CC_AGY_AUTH_MODE`) | Explicit `--auth` never falls back to another backend; if `auto` picks Vertex AI from a shell env var, `doctor` warns you upfront. |
| **2. Allow or block GCP projects** | `--allowed-projects` / `--blocked-projects` (`CC_AGY_ALLOWED_PROJECTS`, `CC_AGY_BLOCKED_PROJECTS`) | Blocks unapproved GCP projects at **0 tokens** across Vertex AI, `gcloud` ADC `quota_project_id`, and `agy` CLI `onboarding.json` (`cloudaicompanionProject`), and scrubs ambient `GOOGLE_CLOUD_PROJECT` vars from `agy` subprocesses. |
| **3. Hard spend ceiling & model tier** | `--max-daily-cents 500` + `--profile economy\|balanced\|quality` | Enforces a UTC daily spend cap in cents and stops automatic retries immediately on `429` / `503` quota errors. |
| **4. Isolated git copy + review before apply** | Default outside `--strict`, or force everywhere with `--no-auto-apply` (`CC_AGY_AUTO_APPLY_ON_PASS=0`) | Workers run in disposable git copies, `in_place=true` and shutdown salvage are disabled when `--no-auto-apply` is set, and nothing touches your repo until `delegate_apply`. |
| **5. File scope & one-call undo** | `allowed_paths=[...]` + `delegate_revert` / `delegate_cancel` | Rejects out-of-scope file edits and restores every touched file from its HMAC-signed pre-apply backup in one call. |
| **6. Zero-token preflight audit** | `/cc-agy-plugin:doctor` (or `"Run doctor"`) | Reports active auth mode, pinned status, active/companion GCP project, allow/block lists, spend cap, and manual-apply policy offline at **0 tokens**. |

## Gemini Enterprise seats in a governed organization

These three settings are optional and only apply to the Gemini Enterprise (`agy` CLI) connection. Leave them unset and nothing changes.

**Sign in with your own Google Cloud credential.** If your organization onboards developers with `gcloud auth application-default login`, pass `--enterprise-adc` (or set `CC_AGY_ENTERPRISE_ADC=true`). The plugin then starts `agy` with `AGY_ADC_AUTH=true` and your seat project and location. It also turns on by itself when `agy` already has `userSettings.enableAdc` set and your credential file belongs to a person. A service account is refused, because it is not a licensed seat. The seat project is checked against `--allowed-projects` and `--blocked-projects` before `agy` starts. Add `--enterprise-user-tier gcp-ge-plus-tier` (or `gcp-ge-standard-tier`) so the usage shows up in Google Cloud Console under Gemini Enterprise, Developer tools metrics. Use the tier you were assigned; there is no default.

```bash
gcloud auth application-default login --project "$GE_PROJECT"
python3 -m server.install --host all --auth gemini-enterprise \
  --enterprise-adc --enterprise-project "$GE_PROJECT" --enterprise-user-tier gcp-ge-plus-tier
```

**Verify licensed Gemini Enterprise tiers and pin CLI releases.** By default, Gemini Enterprise mode blocks known consumer Google accounts (`free-tier`, `@gmail.com` personal tiers) and prints a `--doctor` warning if `cache/onboarding.json` has not recorded `enterpriseOnboardingComplete: true` (or an explicit `gcp-ge-*` enterprise tier) yet. Set `CC_AGY_REQUIRE_GE_TIER=1` to require verified enterprise onboarding at 0 tokens before any `agy` run starts. To pin exact binary versions across a fleet, set `CC_AGY_CLI_VERSION=<version from doctor>` and `CC_AGY_CLI_SHA512=<128-hex sha512>`. Because Google's public `agy` manifest serves a rolling latest release, pair older version pins with `CC_AGY_CLI_MANIFEST_URL=https://mirror.internal.example.com/agy-manifest.json`. On Windows, the plugin installs `agy.exe` through a pure-Python SHA-512 verified downloader (`server/agy_cli.py`) when `bash` is absent.

**Write workers run in an OS sandbox.** A Gemini Enterprise write worker runs `agy` without a person watching each step, so the plugin confines it at the operating system level instead of trusting the agent. On Linux it uses bubblewrap (`bwrap`): the whole disk is read-only except the job's own copy of the repo and `agy`'s state folder, `/tmp` and `~/.cache` are private and empty, and `~/.ssh`, `~/.gnupg`, cloud credential folders, `.netrc`, and `.git-credentials` are hidden. On macOS it uses `sandbox-exec` with the same rules. Other files in your home folder that are not on the hidden-credential list remain readable to the worker, so keep secrets in standard credential stores or `.env` files. The network stays on because `agy` must reach Google; instead, SSH agents, git credential helpers, and askpass programs are removed from the worker's environment, and inside the Linux `bwrap` sandbox `ssh`, `scp`, `sftp`, and git's remote HTTP/HTTPS/FTP helpers are masked with `/dev/null` alongside git config guards (outbound HTTPS from other tools remains open so `agy` can reach Google). On Windows (or whenever neither OS sandbox tool is available), the worker enables `agy --sandbox` and returns any action that needs approval as `denied_actions`. Set `CC_AGY_WORKER_SANDBOX=required` to refuse to run without an OS sandbox, or `off` to go back to the old behavior. `--doctor` shows which one is active. Tools that need to write outside the repo copy, such as `npm install` into a global cache, will fail inside the sandbox.

**When your organization blocks an action.** An admin can require review for terminal commands or for files outside the project. When `agy` hits one of those rules, the job pauses with `status: "needs_input"` and a `denied_actions` list that names each blocked command or file. Nothing is applied to your working tree. Your lead agent asks you what to do, runs the approved step inside the job's isolated copy of the repo, and resumes the same `agy` conversation with `delegate_continue`. This works the same way in Claude Code, Cursor, OpenCode, Codex, and Antigravity.

**How approvals reach you (ACP).** Workers talk to `agy` over the Agent Client Protocol (`agy acp --stdio`) by default; older `agy` builds without `acp` fall back to `--output-format stream-json`, and `CC_AGY_CLI_PROTOCOL=stream-json` forces the old path. The plugin does not pass `--dangerously-skip-permissions` to `agy acp` (set `CC_AGY_ACP_SKIP_PERMISSIONS=1` to opt back in), so every action `agy` wants approved arrives as a `session/request_permission` request. The plugin answers each one in this order:

1. Escapes from the job's repo copy, protected secret files, credential commands, and writes in a read-only job are refused, and you are never asked.
2. Actions the plugin's own policy allows (in-scope edits and commands inside the OS sandbox) are approved on the spot.
3. Anything else, such as an admin rule that needs review or an edit outside `allowed_paths`, goes to you. If your host supports MCP elicitation and the call is waiting for its result, it shows an Allow / Deny prompt and `agy` continues in the same turn. Otherwise the job pauses with `status: "needs_input"` as described above.

When a job sets `allowed_paths`, the plugin keeps `agy` out of its `accept-edits` mode, because that mode approves workspace writes without asking and would skip the scope check.

**Let a central Bifrost gateway decide who can run `agy`.** Seat traffic goes from the `agy` CLI straight to Google, so it cannot pass through a gateway. A central Bifrost can still control it. Set the gateway URL and a Virtual Key that is allowed to use `agy-cli/*` models:

```bash
export CC_AGY_BIFROST_POLICY_URL="https://bifrost.internal.example.com"
export CC_AGY_BIFROST_AGY_VIRTUAL_KEY="sk-bf-agy-..."          # falls back to CC_AGY_BIFROST_VIRTUAL_KEY
export CC_AGY_BIFROST_AGY_POLICY_MODEL="gemini-4-hydrogen-medium" # used for policy-routed when rules cannot be read
```

Before each `agy` run the plugin asks the gateway which `agy-cli/<model>` entries the key may use. A revoked key or a model that is not on the list stops the run at 0 tokens. Ask for the model `policy-routed` to run whatever model the gateway's routing rule currently selects. The laptop only makes outbound HTTPS calls, so no inbound port is needed. Two limits to know: the allowlist must contain the `agy` model names you actually run (or `policy-routed`), and Bifrost does not see the tokens these runs use. The check runs on the developer's machine, so treat it as a guard against mistakes and revoked keys, not a security boundary: someone with full control of the machine can skip it. Token counts stay in the plugin's receipts and in the Gemini Enterprise console.

**Follow-up questions reuse the same conversation.** When you ask `ask_gemini` a second question about the same files in the same session, the plugin resumes the earlier `agy` conversation and sends only the new question. The files are not sent again, so Gemini can reuse what it already cached. A conversation is reused only for the same session, model, workspace, and files, and only for 15 minutes (`CC_AGY_CONVERSATION_TTL_SECONDS`, `0` turns this off). If the resume fails, the full prompt is sent in a new conversation. Only conversation IDs and token counts are stored on disk, never prompts.

**Model names are checked before the first run.** The plugin caches the output of `agy models` for 60 seconds. If the model you asked for is not offered to your seat, it picks the closest offered one up front instead of failing a first turn. With a Bifrost policy URL set, it only picks models the gateway also allows.

Run `python3 -m server.install --doctor` to see the seat mode, tier, location, and Bifrost policy status. Add `--probe-oidc` to also query the gateway once and print how many `agy` models the key may use.

## Installer flags and runtime configuration

Pass `--host codex|claude|cursor|opencode|agy|all`, `--auth`, `--profile`, `--project`, `--allowed-projects`, `--blocked-projects`, `--no-auto-apply`, `--location`, `--base-url`, `--gateway-wire`, `--max-daily-cents`, or `--strict` to `python3 -m server.install`. Codex and Antigravity installs also add the guard hooks (see [Guard hooks on every host](reference.md#guard-hooks-on-every-host)); skip them with `--no-codex-hooks` / `--no-agy-hooks`, or remove only ours later with `--host codex|agy --uninstall-hooks`. For a single session without installing, run `claude --plugin-dir /path/to/cc-agy-plugin`.

All CLI flags, `userConfig` keys, and `CC_AGY_*` environment variables are listed in one place in the [Configuration reference](#configuration-reference), or print the grouped public catalog from your terminal with `python3 -m server.install --list-settings`.

## Authentication precedence and retry rules

<a id="authentication-precedence"></a>

### Authentication precedence

Google's Cloud Code server guardrail permits Gemini Enterprise business OAuth seats on the standalone Antigravity CLI (`agy`) while rejecting direct SDK traffic (`"The Antigravity SDK is not available on an enterprise account. Use a Vertex AI API key instead."`). `resolve_auth_config()` in `server/regions.py` selects credentials in this exact order:
1. **Pinned Gemini Enterprise subscription mode** (`CC_AGY_AUTH_MODE=gemini_enterprise`, `--auth gemini-enterprise` / `agy-cli`, `CC_AGY_GEMINI_ENTERPRISE=true`, or `userConfig.auth_mode="gemini_enterprise"`) -> **Standalone Antigravity CLI (`agy`)** (`auth_mode="gemini_enterprise"`, **fail-closed**). Rejects per-call `api_key` and unapproved `project` overrides at 0 tokens, omits pay-as-you-go override properties from MCP `tools/list`, disables Attempt-3 Claude Opus batch escalation (`claude_writer=false`), refuses starting `cc-agy-gateway`, verifies licensed GE sign-in (blocking consumer Gmail accounts), and runs `agy` subprocesses (`agy acp --stdio` by default, or `--output-format stream-json` on older `agy` builds and when `CC_AGY_CLI_PROTOCOL=stream-json`).
2. Explicit `api_key` argument on the tool call (when Gemini Enterprise mode is not pinned) -> **Google AI Studio** (`auth_mode="api_key"`, Python SDK).
3. Explicit `project` argument on the tool call -> **Vertex AI** (`auth_mode="vertex"`, Python SDK).
4. Custom AI gateway base URL (`CC_AGY_VERTEX_BASE_URL`, `GOOGLE_VERTEX_BASE_URL`, `CC_AGY_GEMINI_BASE_URL`, `GOOGLE_GEMINI_BASE_URL`, or `userConfig.vertex_base_url`) -> **Vertex- or Gemini-wire AI Gateway** (`resolve_gateway_wire()` auto-selects Gemini `/v1beta/models/...` wire format for `/genai` and `/gemini` URL paths, `CC_AGY_GEMINI_BASE_URL` / `GOOGLE_GEMINI_BASE_URL`, or `CC_AGY_BIFROST_VIRTUAL_KEY`, and Vertex `/v1beta1/projects/...` wire format otherwise; override via `--gateway-wire auto|vertex|gemini|litellm-vertex|litellm-gemini|bifrost` or `CC_AGY_GATEWAY_WIRE`). Sends `CC_AGY_VERTEX_BASE_URL_TOKEN` (as `Authorization: Bearer` and `x-goog-api-key`) and optional `CC_AGY_BIFROST_VIRTUAL_KEY` (`x-bf-vk`, composable alongside Cloud Run OIDC). `CC_AGY_VERTEX_BASE_URL` / `GOOGLE_VERTEX_BASE_URL` takes priority over an ambient shell `GEMINI_API_KEY`, while in `auto` or `gateway` mode `GOOGLE_GEMINI_BASE_URL` and `CC_AGY_GEMINI_BASE_URL` pair with `GEMINI_API_KEY` when no gateway bearer token or Bifrost virtual key is set.
5. `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `userConfig.gemini_api_key` in the environment -> **Google AI Studio** (`auth_mode="api_key"`, even if `GOOGLE_CLOUD_PROJECT` is also exported in the shell).
6. `GOOGLE_CLOUD_PROJECT`, `VERTEX_PROJECT`, `GCLOUD_PROJECT`, `CLOUD_ML_PROJECT_ID`, or `userConfig.google_cloud_project` in the environment -> **Vertex AI** (`auth_mode="vertex"`).
7. Automatic `agy` CLI fallback (`auto` mode when none of 1 to 6 are set) -> **Standalone Antigravity CLI (`agy`)** (`auth_mode="gemini_enterprise"`) when an external `agy` binary and local Google sign-in are present.

### Retry, thinking-level, and override mechanics

Profile goals (`balanced`, `economy`, `quality`) are summarized in [Options](configuration.md). Additional runtime mechanics:
- **Default reasoning depth**: Mutating worker turns use `LOW` thinking on `gemini-4-hydrogen`, zero thinking on escalated `gemini-4-carbon-zero` turns in `balanced`, `MEDIUM` on `gemini-4-carbon` in `quality`, and `HIGH` on read-only worker turns (override via `thinking_level` or `CC_AGY_DELEGATE_THINKING_LEVEL`).
- **Quota short-circuit**: Worker `429` or `503` quota errors set `escalation.escalate_to_lead=true` (`reason="quota_or_rate_limit"`) immediately without burning retry turns unless `force_retry=true` is passed.
- **Per-call overrides**: Pass `model="gemini-4-hydrogen"` or `model="gemini-4-carbon"` per call, or set `CC_AGY_DEFAULT_DELEGATE_MODEL` / `CC_AGY_DEFAULT_ASK_MODEL`.

## Configuration reference

<a id="configuration-reference"></a>

Strict mode means `--strict`, `CC_AGY_STRICT_MODE=true`, or `userConfig.strict_delegation_mode=true`.

| Setting (`install` flag / `userConfig` / env var) | Default | What it controls |
|---|---|---|
| `--host codex\|claude\|cursor\|opencode\|agy\|all` | `all` | Target host(s) for `python3 -m server.install` (`cursor` installs into `~/.cursor/mcp.json` and `~/.cursor/skills/`; `opencode` writes legacy-compatible `mcp.<name>` into `~/.config/opencode/opencode.json` [`$OPENCODE_CONFIG_DIR` / `$XDG_CONFIG_HOME/opencode` / absolute `$OPENCODE_CONFIG`, preserving V2 `mcp.servers` configs in place and leaving commented `opencode.jsonc` files untouched next to `opencode.json`] plus `skills/`, `commands/`, and `agents/` with `0o600` `.bak-*` backups for any pre-existing or user-modified files; `agy` / `antigravity` / `jetski` installs into `~/.gemini/config/`; or pass `claude --plugin-dir /path/to/cc-agy-plugin` / `codex plugin marketplace add manishkjs/cc-agy-plugin --ref main`). |
| `--auth` / `userConfig.auth_mode` / `CC_AGY_AUTH_MODE` | `auto` | Connection mode (`gemini_enterprise`, `agy_cli`, `api_key`, `vertex`, `gateway`, or `auto`; alias `CC_AGY_GEMINI_ENTERPRISE=true`). Explicit `api_key` mode always connects directly to Google AI Studio and ignores gateway URLs; use `auto` or `gateway` for custom gateways (explicit `vertex` mode also honors a configured gateway URL when a gateway token, Bifrost virtual key, or OIDC audience is present). |
| `--agy-cli-path`, `--install-agy-cli` / `userConfig.agy_cli_path` / `CC_AGY_CLI_BIN`, `CC_AGY_CLI_CONFIG_DIR` | `~/.local/share/cc-agy-plugin/bin/agy` / `~/.gemini/antigravity-cli` | External `agy` binary and OAuth config directory (`ANTIGRAVITY_OAUTH_TOKEN` is also supported and scrubbed from receipts). |
| `userConfig.gemini_api_key` / `GEMINI_API_KEY` (`GOOGLE_API_KEY`) | unset | Google AI Studio API key (redacted from child shells and receipts). |
| `--project` / `userConfig.google_cloud_project` / `GOOGLE_CLOUD_PROJECT` | unset (required on Vertex) | Google Cloud project ID for pay-as-you-go Vertex AI (`aiplatform.googleapis.com`). |
| `--location` / `userConfig.google_cloud_location` / `GOOGLE_CLOUD_LOCATION` | `us-central1` | Preferred Vertex AI region (`gemini-4-hydrogen` and `gemini-4-carbon` route to `global` automatically). |
| `--base-url` / `userConfig.vertex_base_url` / `CC_AGY_VERTEX_BASE_URL` (`GOOGLE_VERTEX_BASE_URL`, `CC_AGY_GEMINI_BASE_URL`, `GOOGLE_GEMINI_BASE_URL`) | unset | Custom AI gateway URL for LiteLLM (`/vertex_ai` or `/gemini`), Bifrost (`/genai`), Portkey, Apigee, or Cloud Run proxies (`CC_AGY_VERTEX_BASE_URL` takes priority over an ambient shell `GEMINI_API_KEY`; in `auto` or `gateway` mode, `GOOGLE_GEMINI_BASE_URL` and `CC_AGY_GEMINI_BASE_URL` also accept `GEMINI_API_KEY` when no gateway bearer token or Bifrost virtual key is set; Windows PowerShell bootstrap in `scripts/bootstrap-gateway.ps1`). |
| `--gateway-wire` / `userConfig.gateway_wire` / `CC_AGY_GATEWAY_WIRE` | `auto` | Gateway wire protocol (`auto`, `vertex`, `gemini`, `litellm-vertex`, `litellm-gemini`, or `bifrost`). `auto` selects `gemini` (`/v1beta/models/...`, `GeminiAPIEndpoint`) when the gateway path ends in `/genai` or `/gemini`, when `CC_AGY_GEMINI_BASE_URL` / `GOOGLE_GEMINI_BASE_URL` is used, or when `CC_AGY_BIFROST_VIRTUAL_KEY` is set, and selects `vertex` (`/v1beta1/projects/...`, `VertexEndpoint`) otherwise; an explicit trailing `/v1`, `/v1alpha`, `/v1beta`, or `/v1beta1` on `--base-url` sets the SDK API version without doubling the path segment. |
| `userConfig.vertex_base_url_token` / `CC_AGY_VERTEX_BASE_URL_TOKEN` | unset | Bearer token or virtual key sent to the custom gateway as `Authorization: Bearer` and `x-goog-api-key` (redacted from child shells and receipts). |
| `userConfig.bifrost_virtual_key` / `CC_AGY_BIFROST_VIRTUAL_KEY` | unset | Optional Bifrost (`maximhq/bifrost`) virtual key sent in the `x-bf-vk` header for governance and budget limits; works standalone or alongside Cloud Run OIDC `Authorization: Bearer` tokens (redacted from child shells and receipts). |
| `--gateway-audience` / `userConfig.vertex_base_url_audience` / `CC_AGY_VERTEX_BASE_URL_AUDIENCE` (`CC_AGY_VERTEX_OIDC=1`, `CC_AGY_VERTEX_OIDC_SOURCE`) | unset | OIDC target audience URL (or `auto`) for zero-key Cloud Run and Workload Identity Federation gateway auth. Mints and caches refreshing OIDC identity tokens (up to ~55 minutes per single worker job) via WIF/SA JSON, `gcloud` with `CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT`, human `gcloud auth login` on developer machines (failing closed on expired sessions or `gcloud` timeouts rather than swapping to a VM service account), or Cloud Run/GCE metadata (`CC_AGY_VERTEX_OIDC_SOURCE=auto\|gac\|gcloud\|metadata`; GKE pods without `K_SERVICE` can set `metadata` explicitly). When a custom gateway verifies an exact custom `aud` claim, use SA impersonation, WIF, or `CC_AGY_VERTEX_OIDC_SOURCE=metadata`. |
| `--telemetry-sink` / `userConfig.telemetry_sink` / `CC_AGY_TELEMETRY_SINK` (`CC_AGY_TELEMETRY_TOKEN`) | unset (off) | Opt-in absolute/`~`/`file://` local NDJSON file path (`0o600`) or `https://` collector URL for sanitized receipt telemetry (deterministic `event_id`, delta and cumulative token counts and cents, model, verification mode, ISO `timestamp`, and IAP/OIDC/self-reported caller provenance; never sends prompts, diffs, or secrets; aggregate spend across process restarts via `max(cumulative_cost_cents)` per `job_id`, with 256 KiB tail delta recovery on local NDJSON sinks). |
| `--doctor` (`python3 -m server.install --doctor [--probe-oidc]` or `python3 -m server.doctor [--probe-oidc]`) | off | Runs offline environment, gateway/OIDC, SDK, receipt-store, and telemetry-sink diagnostics as JSON (pass `--probe-oidc` to live-mint once and verify the OIDC principal claim) and exits non-zero when unhealthy. |
| `CC_AGY_LEAD_RATES` (alias `CC_AGY_LEAD_RATES_JSON`) | `server/rate_card.json` | Optional JSON override table `{model_substring: [in, cache_read, cache_5m, cache_1h, out]}` in USD per 1M tokens for Lead cost reporting; matching substrings apply longest-key-first across matching model IDs. |
| `--profile` / `userConfig.cost_profile` / `CC_AGY_COST_PROFILE` | `balanced` | Model routing profile (`balanced`, `economy`, or `quality`). |
| `CC_AGY_DEFAULT_DELEGATE_MODEL` / `CC_AGY_DEFAULT_ASK_MODEL` | from `--profile` | Default Gemini model override for worker jobs or `ask_gemini` calls (overridden by `model=` per call). |
| `--max-daily-cents` / `userConfig.max_daily_cents` / `CC_AGY_MAX_DAILY_CENTS` | unset (unlimited) | UTC daily workspace spend cap in US cents (for example `500` for 500¢/day). |
| `--allowed-projects` / `userConfig.allowed_projects` / `CC_AGY_ALLOWED_PROJECTS` | unset | Comma-separated allowlist of GCP project IDs permitted for Vertex AI or Gemini Enterprise (`agy` CLI `cloudaicompanionProject`). Any other project is rejected at 0 tokens. |
| `--blocked-projects` / `userConfig.blocked_projects` / `CC_AGY_BLOCKED_PROJECTS` | unset | Comma-separated blocklist of GCP project IDs that must never be used or billed (checked across Vertex AI, ADC `quota_project_id`, and `agy` CLI `onboarding.json`). |
| `--enterprise-adc` / `userConfig.enterprise_adc` / `CC_AGY_ENTERPRISE_ADC` | auto-detect | Runs `agy` on your Gemini Enterprise seat with your own `gcloud` ADC login (`AGY_ADC_AUTH=true`). Auto-detected from `userSettings.enableAdc` plus an `authorized_user` ADC file; `false` turns it off. Service accounts are refused. |
| `--enterprise-project` / `userConfig.enterprise_project` / `CC_AGY_ENTERPRISE_PROJECT` | ADC `quota_project_id` | Project that holds the Gemini Enterprise seats. Checked against the allowed and blocked project lists before `agy` starts. |
| `--enterprise-location` / `userConfig.enterprise_location` / `CC_AGY_ENTERPRISE_LOCATION` | `global` | Gemini Enterprise location for ADC seat mode (`global`, `us`, or `eu`). |
| `--enterprise-user-tier` / `userConfig.enterprise_user_tier` / `CC_AGY_ENTERPRISE_USER_TIER` | unset | Seat tier passed to `agy` as `BAICODE_USER_TIER` (`gcp-ge-plus-tier` or `gcp-ge-standard-tier`) so usage appears in Developer tools metrics. |
| `--bifrost-policy-url` / `userConfig.bifrost_policy_url` / `CC_AGY_BIFROST_POLICY_URL` | unset | Central Bifrost URL (`https://`, or loopback) that decides which `agy-cli/<model>` entries a Virtual Key may run. Policy lookups only; no prompts are sent. |
| `userConfig.bifrost_agy_virtual_key` / `CC_AGY_BIFROST_AGY_VIRTUAL_KEY` | `CC_AGY_BIFROST_VIRTUAL_KEY` | Bifrost Virtual Key used for the `agy` policy lookup. Never passed to the `agy` process. |
| `userConfig.bifrost_policy_user`, `userConfig.bifrost_policy_password` / `CC_AGY_BIFROST_POLICY_USER`, `CC_AGY_BIFROST_POLICY_PASSWORD` | unset | Optional read-only Bifrost login used to read routing rules and resolve `policy-routed`. |
| `--bifrost-agy-policy-model` / `userConfig.bifrost_agy_policy_model` / `CC_AGY_BIFROST_AGY_POLICY_MODEL` | unset | `agy` model to run for `policy-routed` when routing rules cannot be read. |
| `--worker-sandbox` / `userConfig.worker_sandbox` / `CC_AGY_WORKER_SANDBOX` | `auto` | OS sandbox for Gemini Enterprise write workers (bubblewrap on Linux, `sandbox-exec` on macOS). `auto` falls back to permission prompts when no sandbox works, `required` refuses to run, `off` disables it. `CC_AGY_BWRAP_BIN` overrides the `bwrap` path. |
| `CC_AGY_CLI_PROTOCOL` | `acp` | Wire protocol for Gemini Enterprise workers: `acp` (`agy acp --stdio`) or `stream-json`. Unset uses `acp` when the installed `agy` offers it and falls back to `stream-json` otherwise. |
| `CC_AGY_ACP_SKIP_PERMISSIONS` | `0` | `1` passes `--dangerously-skip-permissions` to `agy acp`. Leave it off so admin (MDM) and sandbox rules reach the plugin as `session/request_permission` callbacks. |
| `CC_AGY_CONVERSATION_TTL_SECONDS` | `900` | How long an `ask_gemini` conversation on a Gemini Enterprise seat can be resumed for a follow-up question. `0` always sends the full prompt. |
| `CC_AGY_MODEL_CATALOG` / `CC_AGY_MODEL_CATALOG_TTL_SECONDS` | on / `60` | Checks requested models against the cached `agy models` list before a run. Set `CC_AGY_MODEL_CATALOG=0` to turn the check off. |
| `CC_AGY_STATE_DIR` | next to the managed `agy` binary | Where the model list and conversation IDs are cached (files are `0600`). |
| `CC_AGY_BIFROST_POLICY_TTL_SECONDS` | `60` | How long a Bifrost policy answer is reused before it is fetched again. |
| `--no-auto-apply` / `userConfig.auto_apply_on_pass=false` / `CC_AGY_AUTO_APPLY_ON_PASS=0` | `false` outside strict mode, `true` in strict mode | Enforces manual diff review (`get_diff` then `delegate_apply`) across all tools, disables shutdown salvage, and blocks `in_place=true` unless `CC_AGY_ALLOW_IN_PLACE=1`. |
| `CC_AGY_ALLOW_IN_PLACE` | `true` (`false` when `--no-auto-apply` is set) | Set to `0` to forbid `in_place=true` direct working-tree edits so every worker runs inside a disposable git copy. |
| `--no-bootstrap` / `CC_AGY_VENV_DIR` / `CC_AGY_PYTHON_BIN` | `~/.local/share/cc-agy-plugin/venv` | Per-user virtualenv directory or explicit Python interpreter path used on PEP 668 systems. |
| `--strict` / `userConfig.strict_delegation_mode` / `CC_AGY_STRICT_MODE` | `false` | Blocks direct host edits in Claude Code so every modified line traces to a signed worker receipt. |
| `CC_AGY_CHECK_ENV_ALLOW` | unset | Comma-separated env var names allowed into acceptance checks, batch checks, and worker commands (credentials, tokens, `*_DSN`, password URLs, and `SSH_AUTH_SOCK` are scrubbed by default; removed names appear under `credential_env_removed` on failure). |
| `CC_AGY_AUTO_APPLY_ON_PASS` | on in strict mode, off otherwise | Auto-applies a finished worker diff when status is `ok`, behavioral checks pass, and `scope_violations` is empty (`0` disables). |
| `CC_AGY_SALVAGE_ON_SHUTDOWN` | on in bounded Lead sessions | Applies the best verified unapplied worker edits on MCP server shutdown when no job was applied yet. |
| `CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE` | on in strict mode | Routes a second `delegate_start` to `delegate_continue` on the latest paused (`timeout`, `checkpoint`, `needs_input`, failed acceptance) job when scopes overlap. |
| `CC_AGY_SPARSE_WORKTREE` | off (`1` or `CC_AGY_SPARSE_THRESHOLD_FILES=<N>`) | Materializes only scoped directories, 2-hop local imports, configs, and matching tests when `allowed_paths` is set on large monorepos. |
| `CC_AGY_MIN_VIEW_LINES` | `250` | Widens small `view_file` reads to at least this many lines (`0` disables). |
| `CC_AGY_COMPACTION_THRESHOLD` / `CC_AGY_COMPACTION_BUFFER` | `120000` tokens | Worker context-compaction trigger (`0` disables; accepts integer token counts like `120000` or window percentages like `85%`, or set `CC_AGY_COMPACTION_BUFFER=15%` to reserve 15% of the model context window with a 16,000-token floor; `CC_AGY_TOOL_OUTPUT_MAX_TOKENS=4000` caps per-tool output). |
| `CC_AGY_INPUT_MODALITIES` | from `server/rate_card.json` | Comma-separated input modalities declared for the active worker model (`text,image,pdf,audio,video` on Gemini models); `view_file` is always a text-only line reader that refuses binary media files cleanly and cites unsupported modalities when excluded here. |
| `CC_AGY_QA_REVIEW_MODEL` | unset | Runs a blocker-only spec-vs-diff QA check with this model before auto-apply (also `qa_review_model` per call). |
| `CC_AGY_TOOL_PROFILE` | `core` in strict mode, `full` otherwise | `core` exposes only the 9 core delegation and diagnostic tools, cutting the MCP tool schema from about 7,200 tokens to under 1,250 tokens. |
| `CC_AGY_LEAD_READ_LIMIT` (alias `CC_AGY_MAX_LEAD_READS`) | `2` (strict mode) | Caps direct Lead file/search reads before delegation (up to 6 review reads are added when a worker finishes without auto-apply; `0` allows none, `off` or `<0` removes the cap). |
| `CC_AGY_FLASH_MAX_TOOL_CALLS` | `100` in economy, `55` otherwise | Per-turn tool-call cap for Flash workers (`55` in `balanced`, `100` in `economy`, matching Pro/Argon's `CC_AGY_MAX_TOOL_CALLS=100`). |
| `CC_AGY_SAME_MODEL_RETRIES` | `1` in economy, `0` otherwise | Automatic same-model fix turns in bounded Lead sessions when a voluntary `status="ok"` turn edited files under 60% of its tool budget and failed checks. |
| `--list-settings` | off | Prints the 6-category public `CC_AGY_*` settings catalog (`--json` outputs machine-readable JSON). Internal `CC_AGY_USERCFG_*` variables are `.mcp.json` passthroughs managed automatically by `server.install`. |

[Back to the README](../README.md)
