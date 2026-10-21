---
description: Run offline environment, authentication, SDK, workspace, and test-runner diagnostics for cc-agy-plugin
---

Call `doctor` on the `vertex` MCP server and report:
1. Overall `status` (`ok`, `degraded`, or `error`), active `cost_profile` (`economy`, `balanced`, or `quality`), `default_delegate_model`, and `default_ask_model`.
2. Authentication mode (`api_key` or `vertex`), project/region if Vertex AI is active, and whether a custom Vertex gateway is configured (never echo raw API keys).
3. Installed SDKs (`google-genai`, `google-antigravity`, `ctx-harness` Straitjacket), git workspace file count, detected language test runner, and receipt store writability.
4. Any `issues`, `warnings`, and exact `remediations`.
