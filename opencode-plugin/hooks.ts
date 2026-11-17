/**
 * OpenCode hook parity for cc-agy-plugin.
 *
 * Claude Code runs the plugin's guard scripts from `hooks/hooks.json`. OpenCode has no
 * hooks.json; instead a plugin returns hook functions. This module maps those functions onto
 * the same five Claude Code events and runs the shared adapter for each one:
 *
 *   python3 scripts/hook_adapter.py --host opencode --event <Event>
 *
 * The adapter reads OpenCode-shaped JSON on stdin, runs the same script that hooks.json maps to
 * the event, and prints `{"action":"allow"|"deny","reason","additionalContext","block"}`.
 *
 * | OpenCode hook                          | Claude Code event | What happens with the result                     |
 * |----------------------------------------|-------------------|--------------------------------------------------|
 * | `tool.execute.before`                  | PreToolUse        | deny -> throw, which stops the tool call          |
 * | `tool.execute.after`                   | PostToolUse       | receipt ledger; context is buffered              |
 * | `chat.message`                         | UserPromptSubmit  | context is buffered                              |
 * | `event` `session.created`              | SessionStart      | context is buffered                              |
 * | `event` `session.idle`                 | Stop              | OpenCode cannot block here: log + toast only     |
 * | `experimental.chat.system.transform`   | (delivery)        | buffered context is added to the system prompt   |
 *
 * Every failure (spawn error, timeout, non-zero exit, unparsable output) fails open, except a
 * PreToolUse failure in strict delegation mode, which denies the tool call.
 *
 * Nothing here is exported from `index.ts`; OpenCode treats every export of the entry file as a
 * plugin factory.
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";

export type HookEvent = "SessionStart" | "UserPromptSubmit" | "PreToolUse" | "PostToolUse" | "Stop";

/** Per-event timeouts in milliseconds; they mirror the `timeout` values in `hooks/hooks.json`. */
export const HOOK_TIMEOUTS_MS: Record<HookEvent, number> = {
  SessionStart: 10_000,
  UserPromptSubmit: 5_000,
  PreToolUse: 10_000,
  PostToolUse: 10_000,
  Stop: 15_000,
};

/** OpenCode built-in tools that correspond to the hooks.json PreToolUse matcher (Read/Edit/Write/Grep/Glob/Bash). */
const PRE_TOOL_BUILTINS = new Set(["read", "edit", "write", "apply_patch", "grep", "glob", "bash"]);
/**
 * OpenCode names MCP tools `sanitize(server) + "_" + sanitize(tool)` (non `[A-Za-z0-9_-]` -> `_`),
 * so Claude Code's `mcp__<server>__delegate_*` becomes `<server>_delegate_*`.
 */
const PRE_TOOL_MCP = /^[A-Za-z0-9_-]+?_delegate_[A-Za-z0-9_-]*$/;
/** hooks.json PostToolUse: every tool of the vertex/agy server, or a delegate lifecycle tool of any server. */
const POST_TOOL_SERVER = /^(?:vertex|agy|plugin_cc-agy-plugin_vertex)_[A-Za-z0-9_-]+$/;
const POST_TOOL_DELEGATE = /^[A-Za-z0-9_-]+?_(?:delegate_start|delegate_continue|delegate_result|delegate_to_gemini|delegate_apply)$/;

const STRICT_MODE_KEYS = ["CLAUDE_PLUGIN_OPTION_STRICT_DELEGATION_MODE", "CC_AGY_STRICT_MODE", "CC_AGY_USERCFG_STRICT_MODE"];
const MAX_BUFFERED_CONTEXT = 20;
const MAX_TOOL_OUTPUT_CHARS = 65_536;

/** True when the PreToolUse guard should run for this OpenCode tool id. */
export function matchesPreToolUse(tool: string): boolean {
  return PRE_TOOL_BUILTINS.has(tool) || PRE_TOOL_MCP.test(tool);
}

/** True when the PostToolUse ledger should run for this OpenCode tool id. */
export function matchesPostToolUse(tool: string): boolean {
  return POST_TOOL_SERVER.test(tool) || POST_TOOL_DELEGATE.test(tool);
}

function isTruthy(value: string | undefined): boolean {
  return /^(1|true|yes|on)$/i.test(String(value || "").trim());
}

/** Strict delegation mode, read from the same keys as `server/env_flags.py`. */
export function isStrictMode(env: Record<string, string | undefined>): boolean {
  return STRICT_MODE_KEYS.some((key) => isTruthy(env[key]));
}

/**
 * Turn either `tool.execute.after` output shape into plain text.
 *
 * Built-in tools give `{title, output, metadata}`; MCP tools give the raw MCP result
 * `{content: [{type: "text", text}, ...]}`.
 */
export function toolOutputText(output: any): string {
  if (output == null) return "";
  if (typeof output === "string") return output.slice(0, MAX_TOOL_OUTPUT_CHARS);
  let text = "";
  if (Array.isArray(output.content)) {
    text = output.content
      .filter((item: any) => item && item.type === "text" && typeof item.text === "string")
      .map((item: any) => item.text)
      .join("\n");
  } else if (typeof output.output === "string") {
    text = output.output;
  }
  return text.slice(0, MAX_TOOL_OUTPUT_CHARS);
}

/**
 * The `output` field sent to the adapter: MCP results stay `{content: [{type: "text", text}]}`,
 * built-in results stay `{title, output, metadata}`; text is capped and binary content dropped.
 */
export function normalizeToolOutput(output: any): Record<string, unknown> {
  if (output && Array.isArray(output.content)) {
    return { content: [{ type: "text", text: toolOutputText(output) }], isError: Boolean(output.isError) };
  }
  return { title: String(output?.title ?? ""), output: toolOutputText(output), metadata: output?.metadata ?? null };
}

/** The user's prompt text from `chat.message` parts (text parts that are not synthetic). */
export function promptText(parts: any): string {
  if (!Array.isArray(parts)) return "";
  return parts
    .filter((part: any) => part && part.type === "text" && !part.synthetic && typeof part.text === "string")
    .map((part: any) => part.text)
    .join("\n")
    .trim();
}

export type AdapterResult = {
  action: "allow" | "deny";
  reason: string | null;
  additionalContext: string | null;
  block: boolean;
  /** Set when the adapter could not be run or its output could not be read. */
  failure?: string;
};

export type HookDeps = {
  pluginRoot: string;
  /** OpenCode's project directory (PluginInput.directory); falls back to `process.cwd()`. */
  directory?: string;
  /** OpenCode SDK client, used only for best-effort logs and toasts. */
  client?: any;
  /** Python interpreter for the adapter (the plugin passes `resolvePython`). */
  python: (env: NodeJS.ProcessEnv) => string;
  /** Settings the MCP server runs with (stored login + `CC_AGY_*` from the OpenCode config). */
  settingsEnvironment: () => Record<string, string>;
  env?: NodeJS.ProcessEnv;
};

function adapterPath(deps: HookDeps, env: NodeJS.ProcessEnv): string {
  return (env.CC_AGY_HOOK_ADAPTER || "").trim() || join(deps.pluginRoot, "scripts", "hook_adapter.py");
}

function timeoutFor(event: HookEvent, env: NodeJS.ProcessEnv): number {
  const override = Number(env.CC_AGY_HOOK_TIMEOUT_MS || "");
  return Number.isFinite(override) && override > 0 ? override : HOOK_TIMEOUTS_MS[event];
}

function failed(failure: string): AdapterResult {
  return { action: "allow", reason: null, additionalContext: null, block: false, failure };
}

function parseAdapterOutput(stdout: string): AdapterResult | null {
  const lines = stdout.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return null;
  try {
    const parsed = JSON.parse(lines[lines.length - 1]);
    if (!parsed || typeof parsed !== "object") return null;
    const str = (value: unknown) => (typeof value === "string" && value.trim() ? value : null);
    return {
      action: parsed.action === "deny" ? "deny" : "allow",
      reason: str(parsed.reason),
      additionalContext: str(parsed.additionalContext),
      block: parsed.block === true,
    };
  } catch {
    return null;
  }
}

/** Run the adapter for one event. Never rejects: problems come back as `failure`. */
export function runAdapter(event: HookEvent, payload: Record<string, unknown>, deps: HookDeps): Promise<AdapterResult> {
  const baseEnv = deps.env || process.env;
  const env: NodeJS.ProcessEnv = { ...baseEnv, ...deps.settingsEnvironment() };
  const script = adapterPath(deps, env);
  if (!existsSync(script)) return Promise.resolve(failed(`hook adapter not found: ${script}`));
  const timeoutMs = timeoutFor(event, env);
  return new Promise((resolvePromise) => {
    const controller = new AbortController();
    let settled = false;
    let stdout = "";
    const finish = (result: AdapterResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolvePromise(result);
    };
    const timer = setTimeout(() => {
      controller.abort();
      finish(failed(`${event} hook timed out after ${timeoutMs} ms`));
    }, timeoutMs);
    let child;
    try {
      child = spawn(deps.python(env), [script, "--host", "opencode", "--event", event], {
        env,
        // A missing cwd makes spawn fail with ENOENT, which would look like a missing interpreter.
        cwd: deps.directory && existsSync(deps.directory) ? deps.directory : process.cwd(),
        signal: controller.signal,
        stdio: ["pipe", "pipe", "ignore"],
        windowsHide: true,
      });
    } catch (error) {
      finish(failed(`could not start hook adapter: ${(error as Error).message}`));
      return;
    }
    child.on("error", (error: Error) => finish(failed(`hook adapter error: ${error.message}`)));
    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk: string) => {
      if (stdout.length < 1_000_000) stdout += chunk;
    });
    child.on("close", (code: number | null) => {
      if (code !== 0) return finish(failed(`hook adapter exited with code ${code}`));
      finish(parseAdapterOutput(stdout) || failed("hook adapter printed no JSON result"));
    });
    child.stdin?.on("error", () => {});
    child.stdin?.end(JSON.stringify(payload));
  });
}

/** Build the OpenCode hook functions. The plugin spreads the returned object into its hooks. */
export function createHookHandlers(deps: HookDeps) {
  const env = () => ({ ...(deps.env || process.env), ...deps.settingsEnvironment() });
  const directory = () => deps.directory || process.cwd();
  const pending = new Map<string, string[]>();

  const log = (level: "info" | "warn" | "error", message: string) => {
    try {
      const done = deps.client?.app?.log?.({ body: { service: "cc-agy-plugin", level, message } });
      if (done && typeof done.catch === "function") done.catch(() => {});
    } catch {
      // Logging is best effort.
    }
  };
  const toast = (message: string) => {
    try {
      const done = deps.client?.tui?.showToast?.({ body: { title: "cc-agy-plugin", message, variant: "warning" } });
      if (done && typeof done.catch === "function") done.catch(() => {});
    } catch {
      // Toasts are best effort (no TUI in `opencode run`).
    }
  };
  const remember = (sessionID: string | undefined, context: string | null) => {
    if (!context) return;
    const key = sessionID || "";
    const list = pending.get(key) || [];
    list.push(context);
    pending.set(key, list.slice(-MAX_BUFFERED_CONTEXT));
  };
  const run = async (event: HookEvent, payload: Record<string, unknown>) => {
    const result = await runAdapter(event, { directory: directory(), ...payload }, deps);
    if (result.failure) log("warn", `cc-agy ${event} hook: ${result.failure}`);
    return result;
  };

  return {
    "tool.execute.before": async (input: { tool: string; sessionID: string; callID: string }, output: { args: any }) => {
      if (!input || !matchesPreToolUse(String(input.tool || ""))) return;
      const result = await run("PreToolUse", {
        tool: input.tool,
        args: output?.args ?? {},
        sessionID: input.sessionID,
        callID: input.callID,
      });
      if (result.failure && isStrictMode(env())) {
        throw new Error(
          `cc-agy strict delegation mode blocked ${input.tool}: the PreToolUse guard did not run (${result.failure}).`,
        );
      }
      if (result.action === "deny") {
        throw new Error(result.reason || `cc-agy guard denied ${input.tool}.`);
      }
      remember(input.sessionID, result.additionalContext);
    },

    "tool.execute.after": async (
      input: { tool: string; sessionID: string; callID: string; args: any },
      output: any,
    ) => {
      if (!input || !matchesPostToolUse(String(input.tool || ""))) return;
      const result = await run("PostToolUse", {
        tool: input.tool,
        args: input.args ?? {},
        sessionID: input.sessionID,
        callID: input.callID,
        output: normalizeToolOutput(output),
      });
      remember(input.sessionID, result.additionalContext);
    },

    "chat.message": async (input: { sessionID: string; messageID?: string; agent?: string }, output: { parts: any[] }) => {
      const prompt = promptText(output?.parts);
      if (!prompt) return;
      const result = await run("UserPromptSubmit", {
        sessionID: input?.sessionID,
        messageID: input?.messageID,
        agent: input?.agent,
        prompt,
      });
      remember(input?.sessionID, result.additionalContext);
    },

    event: async ({ event }: { event: any }) => {
      const type = event?.type;
      const props = event?.properties || {};
      if (type === "session.created") {
        const sessionID = props.info?.id || props.sessionID;
        const result = await run("SessionStart", { sessionID, source: "startup" });
        remember(sessionID, result.additionalContext);
      } else if (type === "session.idle") {
        const result = await run("Stop", { sessionID: props.sessionID, stop_hook_active: false });
        if (result.block && result.reason) {
          // OpenCode publishes session.idle after the turn ended, so the session cannot be kept going.
          log("warn", `cc-agy provenance check: ${result.reason}`);
          toast(result.reason);
        }
      }
    },

    "experimental.chat.system.transform": async (input: { sessionID?: string }, output: { system: string[] }) => {
      if (!output || !Array.isArray(output.system)) return;
      for (const key of [input?.sessionID || "", ""]) {
        const list = pending.get(key);
        if (!list || !list.length) continue;
        pending.delete(key);
        output.system.push(...list);
        if (!input?.sessionID) break;
      }
    },
  };
}
