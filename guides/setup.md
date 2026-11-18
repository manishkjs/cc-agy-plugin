# Setup

Every way to install cc-agy-plugin: the one-command installer, manual flags per host and connection, sign-in fixes, and how to check that it works.

Setup takes about a minute in a normal terminal shell.

The installer sets up either the standalone Antigravity CLI (`~/.local/share/cc-agy-plugin/bin/agy`) for Gemini Enterprise seats or an isolated Python environment (`~/.local/share/cc-agy-plugin/venv`) for AI Studio and Vertex AI. That keeps you clear of `externally-managed-environment` (PEP 668) errors on Debian, Ubuntu, and macOS Homebrew. Secrets are never written to config files.

## 1. One-command interactive setup (recommended)

Paste this single command into **Claude Code** or a **terminal**:

```bash
curl -fsSL https://raw.githubusercontent.com/manishkjs/cc-agy-plugin/main/scripts/quickstart.sh | bash
```

What happens next:
1. **Pick from dropdowns.** Inside Claude Code, two dropdowns ask how you connect (`Gemini Enterprise Seat`, `Google AI Studio API Key`, `Vertex AI`, or `AI Gateway`) and which model preset you want (`Balanced`, `Quality`, or `Economy`). In a regular terminal, an arrow-key (`↑/↓`) selector opens instead.
2. **Sign in to Gemini Enterprise in one step.** On a local Mac or desktop, your browser opens and `agy` saves your sign-in, with nothing to paste. Over SSH or on a remote VM, you get a sign-in link: open it, sign in, click **Copy to Clipboard** on the Antigravity page, and paste the code back into chat. Either way, the plugin then records your Gemini Enterprise license and project, so your first `agy` call works right away. Signing in from a second tab reuses the same link instead of breaking the first one.
3. **Automatic `--doctor` check.** Confirms MCP tools, your seat tier (for example `gcp-ge-plus-tier`), and git workspace readiness at 0 tokens.
4. **Run `/reload-plugins`** in the Claude Code session you installed from (and any other open tabs). Then `/cc-agy-plugin:doctor` and `/cc-agy-plugin:agy` are ready. Run `/cc-agy-plugin:setup` or `/cc-agy-plugin:login` anytime to change settings or sign in again.

**If calls fail with "authentication failed".** Run `/cc-agy-plugin:login`. It repairs a saved sign-in that `agy` rejects (a token saved without a license tier or project) without opening the browser, or prints a fresh link if the token is missing, damaged or expired. `/cc-agy-plugin:login --force` always starts a new sign-in. If your account holds seats in more than one project, the plugin tells you which one it picked; run `/cc-agy-plugin:login --project <ID>` (or set `CC_AGY_ENTERPRISE_PROJECT`) to switch. Don't try to sign in with `agy -p`: print mode gives up after 60 seconds on a headless machine.

**Short sign-in links are opt-in.** Set `CC_AGY_SHORT_URL=1` to also get an `is.gd` / `tinyurl` link that never wraps in macOS Terminal. It is off by default because it sends the sign-in URL to a third-party service.

<details>
<summary><b>Non-interactive / CI flags & version pinning</b></summary>

```bash
# Skip the dropdowns by passing --host and --auth directly:
curl -fsSL https://raw.githubusercontent.com/manishkjs/cc-agy-plugin/main/scripts/quickstart.sh | bash -s -- --host claude --auth gemini-enterprise
# Pin a specific release tag or 40-char commit SHA:
curl -fsSL https://raw.githubusercontent.com/manishkjs/cc-agy-plugin/main/scripts/quickstart.sh | CC_AGY_REPO_REF=v0.4.2 bash -s -- --host opencode --auth gemini-enterprise
# Or via uvx (bundles plugin assets directly inside the wheel, offline-safe after download):
uvx --from git+https://github.com/manishkjs/cc-agy-plugin@v0.4.2 cc-agy-plugin --host claude --auth gemini-enterprise
```

</details>

Or clone manually if you prefer working from your own directory:

```bash
git clone https://github.com/manishkjs/cc-agy-plugin && cd cc-agy-plugin
```

## 2. Manual setup by host and connection (optional)

If you cloned the repository manually (`git clone https://github.com/manishkjs/cc-agy-plugin && cd cc-agy-plugin`) and want to pass flags directly instead of using the dropdown wizard (`python3 -m server.install`), expand your connection type below:

<details>
<summary><b>Option A: Gemini Enterprise work subscription</b> (uses your included monthly seat)</summary>

```bash
# Install for Claude Code and download ~/.local/share/cc-agy-plugin/bin/agy:
python3 -m server.install --host claude --auth gemini-enterprise

# Or install into Cursor (~/.cursor/mcp.json + ~/.cursor/skills/):
python3 -m server.install --host cursor --auth gemini-enterprise

# Or install into OpenCode (~/.config/opencode/opencode.json + skill, /commands, and @gemini-delegator):
python3 -m server.install --host opencode --auth gemini-enterprise

# Or install into Codex (~/.codex/config.toml):
python3 -m server.install --host codex --auth gemini-enterprise

# Or install into Antigravity CLI / IDE (~/.gemini/config/):
python3 -m server.install --host agy --auth gemini-enterprise

# Or run with no flags for the arrow-key dropdown wizard:
python3 -m server.install

# Sign in with a clickable Google OAuth link (opens browser automatically on desktop, or prints URL over SSH):
python3 -m server.agy_login
```

</details>

<details>
<summary><b>Option B: Google AI Studio API key</b> (fastest personal setup, pay-as-you-go)</summary>

```bash
export GEMINI_API_KEY="your-key"
python3 -m server.install --host claude   # or --host cursor, --host opencode, --host codex, --host agy, or --host all
```

</details>

<details>
<summary><b>Option C: Google Cloud Vertex AI project</b> (pay-as-you-go)</summary>

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
python3 -m server.install --host claude   # or --host cursor, --host opencode, --host codex, --host agy, or --host all
```

</details>

<details>
<summary><b>Option D: LiteLLM or Bifrost AI gateway</b></summary>

```bash
# LiteLLM proxy (Vertex pass-through at /vertex_ai or Gemini pass-through at /gemini):
export CC_AGY_VERTEX_BASE_URL="https://litellm.example.com/vertex_ai"   # or https://litellm.example.com/gemini
export CC_AGY_VERTEX_BASE_URL_TOKEN="sk-litellm-virtual-key"
python3 -m server.install --host claude --auth gateway --base-url "$CC_AGY_VERTEX_BASE_URL"

# Bifrost gateway (maximhq/bifrost native /genai provider + optional virtual key):
export CC_AGY_VERTEX_BASE_URL="http://localhost:8080/genai"
export CC_AGY_BIFROST_VIRTUAL_KEY="bf-vk-team-budget"   # sent as x-bf-vk (works alongside Cloud Run OIDC too)
python3 -m server.install --host claude --auth gateway --base-url "$CC_AGY_VERTEX_BASE_URL"
```

The plugin detects the wire format from the gateway path: Gemini `/v1beta/models/...` for `/genai` and `/gemini` paths, and Vertex `/v1beta1/projects/...` otherwise. To set it yourself, pass `--gateway-wire auto|vertex|gemini|litellm-vertex|litellm-gemini|bifrost`.

</details>

### What `--host` sets up

| Host | Flag | What the installer writes |
|:--|:--|:--|
| Claude Code | `--host claude` | The Claude Code plugin, with your non-secret settings forwarded as plugin `userConfig` |
| Cursor | `--host cursor` | The `agy` MCP server in `~/.cursor/mcp.json` and the routing skill in `~/.cursor/skills/cc-agy-plugin/SKILL.md` |
| OpenCode | `--host opencode` | `mcp.agy` in `~/.config/opencode/opencode.json`, the global skill, the `/agy`, `/delegate`, `/doctor`, `/jobs`, and `/cost` slash commands, and the `@gemini-delegator` subagent |
| Codex | `--host codex` | Non-secret plugin settings in `~/.codex/config.toml` |
| Antigravity | `--host agy` | `~/.gemini/config/mcp_config.json`, `~/.gemini/config/skills/cc-agy-plugin/SKILL.md`, and `~/.gemini/config/plugins/cc-agy-plugin/` |
| Every host | `--host all` | All of the above (this is the default) |

<details>
<summary><b>Notes for OpenCode and Cursor</b></summary>

**OpenCode**

- The installer honours `$OPENCODE_CONFIG_DIR` and `$OPENCODE_CONFIG`.
- Fresh config files use OpenCode's legacy-compatible `mcp.<name>` format with a 30-minute tool timeout. An existing V2 `mcp.servers` layout is preserved as it is, and a commented `opencode.jsonc` next to `opencode.json` is left intact.
- For OpenCode 2.2.6 and later, add `--opencode-v2` to write the `mcp.servers.agy` envelope. Related flags: `--[no-]opencode-codemode`, `--opencode-protocol auto`, and `--[no-]opencode-pool`.
- Installed asset hashes are tracked in `.cc-agy-plugin-assets.json`. Any pre-existing or user-edited file is backed up to `*.bak-*` before it is replaced.
- Symlinked asset files are unlinked before writing, so the external dotfiles target is never modified. Symlinked directories are written through.
- The same skill, commands, and subagent are bundled in this repository under `.opencode/`.
- Optional native plugin: run `opencode plugin /path/to/cc-agy-plugin -g`, then `opencode auth login` and pick `cc-agy`, which offers four login methods: Gemini Enterprise seat, Gemini API key, Google Cloud Vertex AI project, or AI Gateway. If `cc-agy` is missing from the provider list, the plugin did not load: run `opencode --print-logs mcp list` and look for `failed to load plugin`. It registers the same `agy` MCP server, commands, and subagent without editing `opencode.json`, and keeps anything you enter in OpenCode's own `auth.json`. It opens no network port. Use either this or `--host opencode`, not both.

**Cursor**

- This repository also includes a native `.cursor-plugin/plugin.json` manifest.

</details>

> [!TIP]
> If `claude` is already open in another terminal tab, run `/reload-plugins` inside that session (or restart with `claude --continue`), or prefix shell commands inside Claude Code with `!`.

For every flag and fallback rule, see [Configuration](configuration.md), [Authentication precedence](configuration.md#authentication-precedence), and the [Configuration reference](configuration.md#configuration-reference).

## 3. Verify your setup (free, 0 tokens)

| Host | How to check |
|:--|:--|
| Claude Code | Type `/cc-agy-plugin:doctor` |
| Cursor | Open your project, check **Cursor Settings -> Tools & MCP** to see `agy`, and ask in Agent chat: `"Run doctor to verify my Gemini setup."` |
| OpenCode | Start `opencode` in your project and type `/doctor` (or run `opencode mcp list` to confirm `agy` is enabled) |
| Codex | Start `codex` in your project and ask: `"Run doctor to verify my Gemini setup."` |
| Antigravity | Start `agy` in your project (or run `agy mcp list` to confirm `agy` is registered) and ask: `"Run doctor to verify my Gemini setup."` |

[Back to the README](../README.md)
