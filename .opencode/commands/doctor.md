---
description: Run zero-token offline environment, auth, SDK, workspace, and test-runner diagnostics for cc-agy-plugin
---

Call `agy_doctor` on the `agy` MCP server and report:
1. Overall `status` (`ok`, `degraded`, or `error`), active `cost_profile` (`economy`, `balanced`, or `quality`), `default_delegate_model`, and `default_ask_model`.
2. Authentication mode (`gemini_enterprise` via standalone `agy` CLI, `api_key`, or `vertex`), `agy` CLI binary and sign-in readiness or GCP project/region if configured, host config status (`mcp.host_configs.opencode`), and whether a custom AI gateway is active (never echo raw API keys or OAuth tokens).
3. Installed SDKs (`google-genai`, `google-antigravity`, `ctx-harness` Straitjacket), git workspace file count, detected language test runner, and receipt store writability.
4. Any `issues`, `warnings`, and exact `remediations`.
