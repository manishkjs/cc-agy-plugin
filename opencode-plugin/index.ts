/**
 * Native OpenCode entry point for cc-agy-plugin.
 *
 * Install with `opencode plugin <cc-agy-plugin> -g`, then run `opencode auth login`,
 * choose "Other", and enter the provider id `cc-agy` (custom plugin providers are not in
 * OpenCode's built-in catalog, so searching for "cc-agy" shows no matches). OpenCode then
 * shows the login methods: Gemini Enterprise seat, Gemini API key, Google Cloud Vertex AI
 * project, or AI Gateway. The plugin:
 *
 * - stores what you enter in OpenCode's own `auth.json` (mode 0600), never in a config file;
 * - registers the `agy` MCP server, which launches this repository's existing stdio server
 *   (`server/main.py`). No HTTP listener is opened;
 * - registers the `/agy`, `/delegate`, `/doctor`, `/jobs`, and `/cost` commands and the
 *   `@gemini-delegator` subagent from the bundled `.opencode/` files.
 *
 * Anything you already defined in `opencode.json` (for example an entry written by
 * `python3 -m server.install --host opencode`) is left untouched.
 *
 * This file must export exactly one value, the default plugin factory: OpenCode treats
 * every export of the entry file as a plugin and fails to load on any non-function export.
 * Helpers live in `./internal.ts`.
 */

import { CcAgyPlugin } from "./internal.ts";

export default CcAgyPlugin;
