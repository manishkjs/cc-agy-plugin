/**
 * Helpers and the plugin factory for cc-agy-plugin's native OpenCode entry point.
 *
 * Kept out of `index.ts` on purpose: OpenCode (1.19.6 and later) loads every export of the
 * plugin entry file as a plugin factory and refuses to load the plugin when any export is
 * not a function ("Plugin export is not a function"). `index.ts` therefore exports only
 * the default factory, and tests import the helpers from this file.
 */

import { existsSync, readFileSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "@opencode-ai/plugin";

export const PROVIDER_ID = "cc-agy";
export const SERVER_NAME = "agy";
const SEAT_PLACEHOLDER_KEY = "gemini-enterprise-seat";
const MCP_TOOL_TIMEOUT_MS = 1_800_000;
const PLUGIN_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

type StoredAuth = { key?: string; metadata?: Record<string, string> };
type Frontmatter = Record<string, string | Record<string, string>>;

/** Read this plugin's entry from OpenCode's `auth.json`; a missing or unreadable file yields `{}`. */
export function readStoredAuth(env: NodeJS.ProcessEnv = process.env): StoredAuth {
  const dataHome = (env.XDG_DATA_HOME || "").trim() || join(env.HOME || homedir(), ".local", "share");
  try {
    const parsed = JSON.parse(readFileSync(join(dataHome, "opencode", "auth.json"), "utf8"));
    const entry = parsed?.[PROVIDER_ID];
    return entry && typeof entry === "object" ? (entry as StoredAuth) : {};
  } catch {
    return {};
  }
}

/** Turn the stored login into environment variables for the MCP server process. */
export function buildServerEnvironment(auth: StoredAuth): Record<string, string> {
  const meta = auth.metadata || {};
  const key = String(auth.key || "").trim();
  const environment: Record<string, string> = {};
  if (!meta.authMode) {
    // The API-key method has no prompts and no authorize step, so OpenCode saves the key typed
    // at its own "Enter your API key" prompt as `{type: "api", key}` with no metadata.
    if (key && key !== SEAT_PLACEHOLDER_KEY) {
      environment.CC_AGY_AUTH_MODE = "api_key";
      environment.GEMINI_API_KEY = key;
    }
    return environment;
  }
  if (meta.authMode === "api_key") {
    environment.CC_AGY_AUTH_MODE = "api_key";
    if (key) environment.GEMINI_API_KEY = key;
    return environment;
  }
  if (meta.authMode === "vertex") {
    environment.CC_AGY_AUTH_MODE = "vertex";
    if (meta.project) environment.GOOGLE_CLOUD_PROJECT = meta.project;
    if (meta.location) environment.GOOGLE_CLOUD_LOCATION = meta.location;
    return environment;
  }
  if (meta.authMode === "gateway") {
    environment.CC_AGY_AUTH_MODE = "gateway";
    if (meta.baseUrl) environment.CC_AGY_VERTEX_BASE_URL = meta.baseUrl;
    if (key && key !== SEAT_PLACEHOLDER_KEY) environment.CC_AGY_VERTEX_BASE_URL_TOKEN = key;
    if (meta.bifrostVirtualKey) environment.CC_AGY_BIFROST_VIRTUAL_KEY = meta.bifrostVirtualKey;
    return environment;
  }
  if (meta.authMode !== "gemini_enterprise") return environment;
  environment.CC_AGY_AUTH_MODE = "gemini_enterprise";
  if (meta.bifrostPolicyUrl) {
    environment.CC_AGY_BIFROST_POLICY_URL = meta.bifrostPolicyUrl;
    if (key && key !== SEAT_PLACEHOLDER_KEY) environment.CC_AGY_BIFROST_AGY_VIRTUAL_KEY = key;
  }
  return environment;
}

/** Picks the Python interpreter: `CC_AGY_PYTHON_BIN`/`CC_AGY_PYTHON`, managed user venv, plugin `.venv`, then `python3`. */
export function resolvePython(env: NodeJS.ProcessEnv = process.env): string {
  const explicit = (env.CC_AGY_PYTHON_BIN || env.CC_AGY_PYTHON || "").trim();
  if (explicit) return explicit;
  const customVenv = (env.CC_AGY_VENV_DIR || "").trim();
  const dataHome = (env.XDG_DATA_HOME || "").trim() || join(env.HOME || homedir(), ".local", "share");
  const managedVenv = customVenv || join(dataHome, "cc-agy-plugin", "venv");
  for (const venv of [managedVenv, join(PLUGIN_ROOT, ".venv")]) {
    const candidates =
      process.platform === "win32"
        ? [join(venv, "Scripts", "python.exe"), join(venv, "bin", "python")]
        : [join(venv, "bin", "python")];
    for (const candidate of candidates) {
      if (existsSync(candidate)) return candidate;
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}

/** Split a Markdown file into simple `key: value` frontmatter (one nested level) and its body. */
export function parseMarkdownAsset(text: string): { frontmatter: Frontmatter; body: string } {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
  if (!match) return { frontmatter: {}, body: text.trim() };
  const frontmatter: Frontmatter = {};
  let parentKey = "";
  for (const line of match[1].split(/\r?\n/)) {
    const pair = /^(\s*)([A-Za-z0-9_-]+):\s*(.*)$/.exec(line);
    if (!pair) continue;
    const [, indent, key, value] = pair;
    if (indent && parentKey) {
      (frontmatter[parentKey] as Record<string, string>)[key] = value.trim();
    } else if (value.trim()) {
      frontmatter[key] = value.trim();
      parentKey = "";
    } else {
      frontmatter[key] = {};
      parentKey = key;
    }
  }
  return { frontmatter, body: text.slice(match[0].length).trim() };
}

function readAssets(subdir: string): Array<{ name: string; frontmatter: Frontmatter; body: string }> {
  const dir = join(PLUGIN_ROOT, ".opencode", subdir);
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((file) => file.endsWith(".md"))
    .sort()
    .map((file) => ({ name: basename(file, ".md"), ...parseMarkdownAsset(readFileSync(join(dir, file), "utf8")) }));
}

function hasServerEntry(mcp: any): boolean {
  return Boolean(mcp && (mcp[SERVER_NAME] || (mcp.servers && mcp.servers[SERVER_NAME])));
}

/** Add the MCP server, commands, and subagent to an OpenCode config object, keeping user entries. */
export function applyConfig(cfg: any, env: NodeJS.ProcessEnv = process.env): void {
  cfg.mcp = cfg.mcp || {};
  if (!hasServerEntry(cfg.mcp)) {
    cfg.mcp[SERVER_NAME] = {
      type: "local",
      command: [resolvePython(env), join(PLUGIN_ROOT, "server", "main.py")],
      environment: buildServerEnvironment(readStoredAuth(env)),
      enabled: true,
      timeout: MCP_TOOL_TIMEOUT_MS,
    };
  }
  cfg.command = cfg.command || {};
  for (const asset of readAssets("commands")) {
    cfg.command[asset.name] = cfg.command[asset.name] || {
      description: String(asset.frontmatter.description || ""),
      template: asset.body,
    };
  }
  cfg.agent = cfg.agent || {};
  for (const asset of readAssets("agents")) {
    cfg.agent[asset.name] = cfg.agent[asset.name] || {
      description: String(asset.frontmatter.description || ""),
      mode: String(asset.frontmatter.mode || "subagent"),
      ...(typeof asset.frontmatter.permission === "object" ? { permission: asset.frontmatter.permission } : {}),
      prompt: asset.body,
    };
  }
  const skillPath = join(PLUGIN_ROOT, ".opencode", "skills", "cc-agy-plugin", "SKILL.md");
  if (existsSync(skillPath)) {
    const list = Array.isArray(cfg.instructions) ? cfg.instructions : [];
    if (!list.includes(skillPath)) list.push(skillPath);
    cfg.instructions = list;
  }
}

function isPolicyUrl(value: string): boolean {
  return /^https:\/\/[^\s?#@]+$/.test(value) || /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?(\/[^\s?#]*)?$/.test(value);
}

type SavedCredential =
  | { type: "success"; provider: string; key: string; metadata: Record<string, string> }
  | { type: "failed" };

/**
 * Wrap a credential as an OpenCode `type: "oauth"` result that needs no browser step.
 *
 * OpenCode appends its own required "Enter your API key" prompt to every `type: "api"` method,
 * but not to `type: "oauth"` ones. With an empty `url`, `opencode auth login` prints no link,
 * shows `instructions`, and saves `callback()`'s `key` and `metadata` as a `type: "api"` entry
 * in `auth.json`: the same shape earlier versions of this plugin saved, so `readStoredAuth`
 * and `buildServerEnvironment` read old and new entries the same way.
 */
export function savedWithoutBrowser(credential: SavedCredential, instructions: string) {
  return {
    url: "",
    instructions,
    method: "auto" as const,
    callback: async () => credential,
  };
}

export const CcAgyPlugin: Plugin = async () => ({
  auth: {
    provider: PROVIDER_ID,
    methods: [
      {
        type: "oauth",
        label: "Gemini Enterprise seat (sign in once with the agy CLI)",
        prompts: [
          {
            type: "text",
            key: "bifrostPolicyUrl",
            message: "Central Bifrost policy URL (optional, leave blank to skip)",
            placeholder: "https://bifrost.internal.example.com",
            validate: (value: string) =>
              !value.trim() || isPolicyUrl(value.trim()) ? undefined : "Enter an https:// URL, or leave blank",
          },
          {
            type: "text",
            key: "bifrostAgyVirtualKey",
            message: "Bifrost Virtual Key for agy (optional, only with a policy URL)",
            placeholder: "sk-bf-agy-...",
          },
        ],
        authorize: async (inputs: Record<string, string> = {}) =>
          savedWithoutBrowser(
            {
              type: "success",
              provider: PROVIDER_ID,
              key: String(inputs.bifrostAgyVirtualKey || "").trim() || SEAT_PLACEHOLDER_KEY,
              metadata: {
                authMode: "gemini_enterprise",
                bifrostPolicyUrl: String(inputs.bifrostPolicyUrl || "").trim().replace(/\/+$/, ""),
              },
            },
            "Saved. If you have not signed in to your Gemini Enterprise seat yet, run /login in OpenCode.",
          ),
      },
      {
        // No prompts and no authorize step: OpenCode asks "Enter your API key" itself (and refuses an
        // empty key), then saves it as `{type: "api", key}`. `buildServerEnvironment` maps that entry
        // to api_key mode, in both `opencode auth login` and the TUI's connect dialog.
        type: "api",
        label: "Gemini API key (Google AI Studio)",
      },
      {
        type: "oauth",
        label: "Google Cloud Vertex AI project (uses gcloud ADC)",
        prompts: [
          {
            type: "text",
            key: "project",
            message: "Google Cloud project ID",
            validate: (value: string) => (value.trim().length > 2 ? undefined : "Required"),
          },
          {
            type: "text",
            key: "location",
            message: "Vertex AI location (optional, defaults to global)",
            placeholder: "global",
          },
        ],
        authorize: async (inputs: Record<string, string> = {}) => {
          const project = String(inputs.project || "").trim();
          const location = String(inputs.location || "").trim();
          return savedWithoutBrowser(
            project
              ? {
                  type: "success",
                  provider: PROVIDER_ID,
                  key: SEAT_PLACEHOLDER_KEY,
                  metadata: { authMode: "vertex", project, ...(location ? { location } : {}) },
                }
              : { type: "failed" },
            "Saved. The plugin uses your gcloud Application Default Credentials for this project.",
          );
        },
      },
      {
        type: "oauth",
        label: "AI Gateway (LiteLLM / Bifrost / custom proxy)",
        prompts: [
          {
            type: "text",
            key: "baseUrl",
            message: "AI gateway base URL",
            placeholder: "https://litellm.example.com/vertex_ai",
            validate: (value: string) => (isPolicyUrl(value.trim()) ? undefined : "Enter an https:// or localhost URL"),
          },
          {
            type: "text",
            key: "gatewayToken",
            message: "Gateway bearer token or API key (optional)",
          },
          {
            type: "text",
            key: "bifrostVirtualKey",
            message: "Bifrost Virtual Key header x-bf-vk (optional)",
          },
        ],
        authorize: async (inputs: Record<string, string> = {}) => {
          const baseUrl = String(inputs.baseUrl || "").trim().replace(/\/+$/, "");
          const token = String(inputs.gatewayToken || "").trim();
          const bifrostVirtualKey = String(inputs.bifrostVirtualKey || "").trim();
          return savedWithoutBrowser(
            baseUrl
              ? {
                  type: "success",
                  provider: PROVIDER_ID,
                  key: token || SEAT_PLACEHOLDER_KEY,
                  metadata: {
                    authMode: "gateway",
                    baseUrl,
                    ...(bifrostVirtualKey ? { bifrostVirtualKey } : {}),
                  },
                }
              : { type: "failed" },
            "Saved. The plugin sends Gemini requests through this gateway.",
          );
        },
      },
    ],
  },
  config: async (cfg: any) => applyConfig(cfg),
});

