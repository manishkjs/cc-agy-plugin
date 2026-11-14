"""Canonical catalog of `CC_AGY_*` runtime environment settings.

Groups user-facing configuration into 6 clear categories while separating internal
`.mcp.json` `CC_AGY_USERCFG_*` passthroughs and low-level harness tuning overrides.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SettingSpec:
    """Metadata for one public `CC_AGY_*` environment variable."""

    name: str
    category: str
    default: str
    summary: str


SETTINGS_CATEGORIES: tuple[str, ...] = (
    "Connection & Auth",
    "Models, Cost & Budgets",
    "Enterprise Governance & Bifrost",
    "OS Sandbox & Containment",
    "Delegation, Hooks & Verification",
    "Gateway, Rate Card & Telemetry",
)

PUBLIC_SETTINGS: tuple[SettingSpec, ...] = (
    # 1. Connection & Auth
    SettingSpec(
        "CC_AGY_AUTH_MODE",
        "Connection & Auth",
        "auto",
        "Auth backend: auto, gemini_enterprise (agy CLI), api_key, vertex, or gateway.",
    ),
    SettingSpec(
        "CC_AGY_CLI_BIN",
        "Connection & Auth",
        "~/.local/share/cc-agy-plugin/bin/agy",
        "Explicit path to the standalone Antigravity CLI (`agy`) binary.",
    ),
    SettingSpec(
        "CC_AGY_CLI_VERSION",
        "Connection & Auth",
        "unset (latest verified)",
        "Optional exact `agy` version pin enforced during install, `doctor`, and tool execution.",
    ),
    SettingSpec(
        "CC_AGY_CLI_SHA512",
        "Connection & Auth",
        "unset (manifest SHA-512)",
        "Optional 128-hex SHA-512 archive checksum pin enforced when downloading `agy`.",
    ),
    SettingSpec(
        "CC_AGY_AUTO_INSTALL_CLI",
        "Connection & Auth",
        "0",
        "Set to `1` to allow downloading `agy` inside an MCP tool call if missing (default: explicit install only).",
    ),
    SettingSpec(
        "CC_AGY_CLI_PROTOCOL",
        "Connection & Auth",
        "stream-json",
        "Wire protocol for `agy` workers: `stream-json` (default) or experimental `acp` (`agy acp --stdio`).",
    ),
    SettingSpec(
        "CC_AGY_CLI_MANIFEST_URL",
        "Connection & Auth",
        "(Google CDN manifest)",
        "Optional enterprise HTTPS mirror URL for the pinned `agy` platform manifest JSON.",
    ),
    SettingSpec(
        "CC_AGY_REQUIRE_GE_TIER",
        "Connection & Auth",
        "0",
        "Set to `1` to require an explicit `gcp-ge-*` enterprise tier in `cache/onboarding.json` for GE OAuth sign-ins.",
    ),
    SettingSpec(
        "CC_AGY_REPO_REF",
        "Connection & Auth",
        "main",
        "Git tag, branch, or commit ref pinned by `quickstart.sh` and `cc-agy-install` checkout bootstrap.",
    ),
    SettingSpec(
        "CC_AGY_CLI_CONFIG_DIR",
        "Connection & Auth",
        "~/.gemini/antigravity-cli",
        "State and OAuth directory used by the `agy` CLI.",
    ),
    SettingSpec(
        "CC_AGY_STATE_DIR",
        "Connection & Auth",
        "~/.local/share/cc-agy-plugin/state",
        "Plugin cache directory for model catalogs and resumable ask conversations.",
    ),
    SettingSpec(
        "CC_AGY_WORKSPACE_ROOT",
        "Connection & Auth",
        "git top-level / cwd",
        "Workspace root override for MCP hosts launched outside the repository.",
    ),
    SettingSpec(
        "CC_AGY_DATA_DIR",
        "Connection & Auth",
        "~/.cache/cc-agy-plugin/<slug>",
        "Directory storing HMAC-signed job receipts, diffs, and pre-apply backups.",
    ),
    # 2. Models, Cost & Budgets
    SettingSpec(
        "CC_AGY_COST_PROFILE",
        "Models, Cost & Budgets",
        "balanced",
        "Routing preset: economy (Flash only), balanced (Flash -> Carbon), or quality (Argon).",
    ),
    SettingSpec(
        "CC_AGY_DEFAULT_DELEGATE_MODEL",
        "Models, Cost & Budgets",
        "(from profile)",
        "Default Gemini model ID for `delegate_start` / `delegate_to_gemini` workers.",
    ),
    SettingSpec(
        "CC_AGY_DEFAULT_ASK_MODEL",
        "Models, Cost & Budgets",
        "(from profile)",
        "Default Gemini model ID for single-shot `ask_gemini` calls.",
    ),
    SettingSpec(
        "CC_AGY_MAX_DAILY_CENTS",
        "Models, Cost & Budgets",
        "unlimited",
        "Daily spend cap in US cents across worker and ask calls (for example 500 = $5.00).",
    ),
    SettingSpec(
        "CC_AGY_MAX_TOOL_CALLS",
        "Models, Cost & Budgets",
        "55 (Flash) / 100 (Pro)",
        "Per-turn worker tool-call cap (`0` disables).",
    ),
    SettingSpec(
        "CC_AGY_MAX_TOTAL_TOKENS",
        "Models, Cost & Budgets",
        "5000000",
        "Per-turn worker cumulative token cap (`0` disables).",
    ),
    SettingSpec(
        "CC_AGY_TOOL_PROFILE",
        "Models, Cost & Budgets",
        "full (core in strict)",
        "MCP tool schema surface exposed to the host: full or core.",
    ),
    # 3. Enterprise Governance & Bifrost
    SettingSpec(
        "CC_AGY_ENTERPRISE_ADC",
        "Enterprise Governance & Bifrost",
        "auto",
        "Use `gcloud auth application-default login` user credentials for Gemini Enterprise seats.",
    ),
    SettingSpec(
        "CC_AGY_ENTERPRISE_PROJECT",
        "Enterprise Governance & Bifrost",
        "(ADC quota project)",
        "GCP project holding Gemini Enterprise seats when `CC_AGY_ENTERPRISE_ADC` is active.",
    ),
    SettingSpec(
        "CC_AGY_ENTERPRISE_LOCATION",
        "Enterprise Governance & Bifrost",
        "global",
        "Gemini Enterprise ADC seat region: global, us, or eu.",
    ),
    SettingSpec(
        "CC_AGY_ENTERPRISE_USER_TIER",
        "Enterprise Governance & Bifrost",
        "unset",
        "Optional seat tier tag: gcp-ge-plus-tier or gcp-ge-standard-tier.",
    ),
    SettingSpec(
        "CC_AGY_ALLOWED_PROJECTS",
        "Enterprise Governance & Bifrost",
        "unset",
        "Comma-separated allowlist of permitted GCP project IDs.",
    ),
    SettingSpec(
        "CC_AGY_BLOCKED_PROJECTS",
        "Enterprise Governance & Bifrost",
        "unset",
        "Comma-separated blocklist of forbidden GCP project IDs.",
    ),
    SettingSpec(
        "CC_AGY_BIFROST_POLICY_URL",
        "Enterprise Governance & Bifrost",
        "unset",
        "Central Bifrost base URL (`https://...`) governing allowed `agy-cli/*` models.",
    ),
    SettingSpec(
        "CC_AGY_BIFROST_AGY_VIRTUAL_KEY",
        "Enterprise Governance & Bifrost",
        "(falls back to VIRTUAL_KEY)",
        "Bifrost Virtual Key queried against `/v1/models` before spawning `agy`.",
    ),
    SettingSpec(
        "CC_AGY_BIFROST_AGY_POLICY_MODEL",
        "Enterprise Governance & Bifrost",
        "unset",
        "Fallback `agy` model ID used when Bifrost returns `agy-cli/policy-routed`.",
    ),
    # 4. OS Sandbox & Containment
    SettingSpec(
        "CC_AGY_WORKER_SANDBOX",
        "OS Sandbox & Containment",
        "auto",
        "OS confinement (`bwrap` on Linux, `sandbox-exec` on macOS): auto, required, or off.",
    ),
    SettingSpec(
        "CC_AGY_CONTAINMENT",
        "OS Sandbox & Containment",
        "1",
        "Enable Straitjacket bounded output and addressable evidence tools on SDK workers.",
    ),
    SettingSpec(
        "CC_AGY_BLOCKED_PATHS",
        "OS Sandbox & Containment",
        "unset",
        "Additional colon/comma-separated host paths hidden or blocked from workers.",
    ),
    # 5. Delegation, Hooks & Verification
    SettingSpec(
        "CC_AGY_STRICT_MODE",
        "Delegation, Hooks & Verification",
        "false",
        "Enforce Lead read-only orchestration and receipt-backed provenance checks.",
    ),
    SettingSpec(
        "CC_AGY_AUTO_APPLY_ON_PASS",
        "Delegation, Hooks & Verification",
        "true",
        "Automatically apply isolated worker diffs when acceptance checks pass.",
    ),
    SettingSpec(
        "CC_AGY_LEAD_READ_LIMIT",
        "Delegation, Hooks & Verification",
        "2",
        "Pre-delegation host Read/Grep/Glob budget in strict mode (`off` removes cap).",
    ),
    SettingSpec(
        "CC_AGY_READ_GUARD",
        "Delegation, Hooks & Verification",
        "off",
        "Optional host Read size hint on large files: off, suggest, or block.",
    ),
    SettingSpec(
        "CC_AGY_CONVERSATION_TTL_SECONDS",
        "Delegation, Hooks & Verification",
        "900",
        "TTL in seconds for resumable `ask_gemini` conversations (`0` disables resume).",
    ),
    # 6. Gateway, Rate Card & Telemetry
    SettingSpec(
        "CC_AGY_VERTEX_BASE_URL",
        "Gateway, Rate Card & Telemetry",
        "unset",
        "Custom AI gateway URL (`https://...` for LiteLLM, Bifrost, Cloud Run, or Apigee).",
    ),
    SettingSpec(
        "CC_AGY_GATEWAY_WIRE",
        "Gateway, Rate Card & Telemetry",
        "auto",
        "Gateway wire format: auto, vertex (litellm-vertex), or gemini (bifrost / litellm-gemini).",
    ),
    SettingSpec(
        "CC_AGY_VERTEX_BASE_URL_TOKEN",
        "Gateway, Rate Card & Telemetry",
        "unset",
        "Bearer token or virtual key sent to `CC_AGY_VERTEX_BASE_URL`.",
    ),
    SettingSpec(
        "CC_AGY_GATEWAY_OIDC_AUDIENCE",
        "Gateway, Rate Card & Telemetry",
        "unset",
        "OIDC audience URL (or `auto`) for keyless Cloud Run / Workload Identity gateways.",
    ),
    SettingSpec(
        "CC_AGY_TELEMETRY_SINK",
        "Gateway, Rate Card & Telemetry",
        "unset",
        "Opt-in local `.ndjson` path or `https://` endpoint receiving scrubbed job receipts.",
    ),
    SettingSpec(
        "CC_AGY_RATE_CARD_PATH",
        "Gateway, Rate Card & Telemetry",
        "server/rate_card.json",
        "Optional path to a custom JSON pricing rate card.",
    ),
)


def format_settings_text() -> str:
    """Format the public settings catalog for terminal output (`--list-settings`)."""
    lines: list[str] = ["cc-agy-plugin runtime settings (public CC_AGY_* reference):"]
    for category in SETTINGS_CATEGORIES:
        lines.append(f"\n[{category}]")
        for spec in PUBLIC_SETTINGS:
            if spec.category == category:
                lines.append(f"  {spec.name:<34} (default: {spec.default})")
                lines.append(f"      {spec.summary}")
    lines.append(
        "\nNote: `CC_AGY_USERCFG_*` variables are internal `.mcp.json` passthroughs managed "
        "automatically by `server.install`; low-level timeout/cache tuning variables remain "
        "optional overrides with safe built-in defaults."
    )
    return "\n".join(lines)
