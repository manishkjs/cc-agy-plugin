#!/usr/bin/env python3
"""Shared hook adapter: run cc-agy-plugin's Claude Code hooks on other hosts.

Usage:
  python3 <root>/scripts/hook_adapter.py --host {opencode,codex,agy} \
      --event {SessionStart,UserPromptSubmit,PreToolUse,PostToolUse,Stop}

How it works:
1. Read the host's native hook JSON from stdin.
2. `normalize()` rewrites it into the Claude Code (CC) payload the existing
   scripts read (`session_id`, `cwd`, `tool_name`, `tool_input`, ...).
3. Run the same script `hooks/hooks.json` maps to the event, as a subprocess
   with the CC timeout from hooks.json. Tool events are skipped when the CC
   matcher in hooks.json does not match the canonical tool name.
4. `denormalize()` turns the script's CC output into the host's native shape.

Failures fail open (allow), except PreToolUse in strict delegation mode, which
denies with a clear reason. Stdlib only, Python 3.9+, no network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping

SCRIPTS_DIR = Path(__file__).resolve().parent
PLUGIN_ROOT = SCRIPTS_DIR.parent
for _p in (str(SCRIPTS_DIR), str(PLUGIN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hook_toolmap  # noqa: E402

HOSTS: tuple[str, ...] = ("opencode", "codex", "agy")
EVENTS: tuple[str, ...] = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop")
EVENT_ALIASES: dict[str, str] = {"PreInvocation": "UserPromptSubmit"}

# Fallbacks if hooks/hooks.json is unreadable (mirrors the shipped file).
_DEFAULT_SCRIPTS: dict[str, tuple[str, int]] = {
    "SessionStart": ("session_start.py", 10),
    "UserPromptSubmit": ("user_prompt_submit.py", 5),
    "PreToolUse": ("pre_tool_guard.py", 10),
    "PostToolUse": ("post_tool_ledger.py", 10),
    "Stop": ("provenance_check.py", 15),
}
_ENV_PREFIXES: tuple[str, ...] = ("CC_AGY_", "CLAUDE_PLUGIN_OPTION_", "GEMINI4_")
_SECRET_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
_SECRET_VALUE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"ya29\.[0-9A-Za-z_\-\.]+"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.=]{8,}"),
)
_STRICT_KEYS: tuple[str, ...] = (
    "CLAUDE_PLUGIN_OPTION_STRICT_DELEGATION_MODE",
    "CC_AGY_STRICT_MODE",
    "CC_AGY_USERCFG_STRICT_MODE",
)
_DEDUPE_MAX_ENTRIES = 256
# Set by server.agy_cli.build_agy_subprocess_env on every agy child the plugin spawns.
WORKER_ENV_MARKER = "CC_AGY_IN_WORKER"
_WORKTREE_DIR_PREFIXES: tuple[str, ...] = (".cc-agy-worktrees-", "cc-agy-worktrees-")


# --------------------------------------------------------------------------- env


def _usable(val: Any) -> bool:
    return isinstance(val, str) and bool(val.strip()) and not val.strip().startswith("${")


def _toml_string(raw: str) -> str | None:
    raw = raw.strip()
    if raw.startswith('"'):
        end = raw.find('"', 1)
        while end != -1 and raw[end - 1] == "\\":
            end = raw.find('"', end + 1)
        if end == -1:
            return None
        try:
            return json.loads(raw[: end + 1])
        except ValueError:
            return None
    if raw.startswith("'"):
        end = raw.find("'", 1)
        return raw[1:end] if end != -1 else None
    return None


def _toml_key(raw: str) -> str:
    raw = raw.strip()
    return raw[1:-1] if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'" else raw


def _toml_inline_table(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    body = raw.strip()
    if not (body.startswith("{") and "}" in body):
        return out
    for m in re.finditer(r"""([A-Za-z0-9_\-"']+)\s*=\s*("(?:[^"\\]|\\.)*"|'[^']*')""", body):
        val = _toml_string(m.group(2))
        if val is not None:
            out[_toml_key(m.group(1))] = val
    return out


def parse_codex_env(text: str, server: str = "agy") -> dict[str, str]:
    """Return `[mcp_servers.<server>.env]` string values from a Codex config.toml."""
    out: dict[str, str] = {}
    section = ""
    names = {server, f'"{server}"', f"'{server}'"}
    env_sections = {f"mcp_servers.{n}.env" for n in names}
    srv_sections = {f"mcp_servers.{n}" for n in names}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        header = re.match(r"\A\[([^\[\]]+)\]\s*(#.*)?\Z", stripped)
        if header:
            section = re.sub(r"\s*\.\s*", ".", header.group(1).strip())
            continue
        if stripped.startswith("[["):
            section = ""
            continue
        if "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        if section in env_sections:
            parsed = _toml_string(val)
            if parsed is not None:
                out[_toml_key(key)] = parsed
        elif section in srv_sections and _toml_key(key) == "env":
            out.update(_toml_inline_table(val))
    return out


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def host_config_env(host: str, base_env: Mapping[str, str]) -> dict[str, str]:
    """Return the plugin MCP server's env block from the host's own config file."""
    home = Path(base_env.get("HOME") or os.path.expanduser("~"))
    if host == "codex":
        cfg = base_env.get("CC_AGY_HOOK_CODEX_CONFIG") or str(
            Path(base_env.get("CODEX_HOME") or home / ".codex") / "config.toml"
        )
        try:
            return parse_codex_env(Path(cfg).read_text(encoding="utf-8"))
        except OSError:
            return {}
    if host == "agy":
        cfg = base_env.get("CC_AGY_HOOK_AGY_MCP_CONFIG") or str(home / ".gemini" / "config" / "mcp_config.json")
        data = _read_json_file(Path(cfg))
        servers = data.get("mcpServers") if isinstance(data, dict) else None
        entry = servers.get("agy") if isinstance(servers, dict) else None
        env = entry.get("env") if isinstance(entry, dict) else None
        return {str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {}
    return {}


def build_env(host: str, base_env: Mapping[str, str], extra: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Merge host-config and shim-provided plugin settings into `base_env` without overriding."""
    env = {k: v for k, v in base_env.items() if isinstance(v, str)}
    sources: list[Mapping[str, Any]] = []
    if extra:
        sources.append(extra)
    sources.append(host_config_env(host, env))
    for src in sources:
        for key, val in src.items():
            if not str(key).startswith(_ENV_PREFIXES) or not _usable(val):
                continue
            if not _usable(env.get(key)):
                env[str(key)] = str(val)
    env.setdefault("CLAUDE_PLUGIN_ROOT", str(PLUGIN_ROOT))
    env["CC_AGY_HOOK_HOST"] = host
    return env


def is_strict(env: Mapping[str, str]) -> bool:
    """Return True when strict delegation mode is on (same keys as `server.env_flags`)."""
    try:
        from server.env_flags import is_strict_mode_enabled

        return bool(is_strict_mode_enabled(env))
    except Exception:  # noqa: BLE001 - fall back to the plain truthy check.
        return any(str(env.get(k) or "").strip().lower() in ("1", "true", "yes", "on") for k in _STRICT_KEYS)


def redact(text: Any, env: Mapping[str, str] | None = None) -> Any:
    """Mask secret-looking env values and token patterns in `text`."""
    if not isinstance(text, str) or not text:
        return text
    for key, val in (env or {}).items():
        if _SECRET_NAME_RE.search(key) and isinstance(val, str) and len(val.strip()) >= 8:
            text = text.replace(val.strip(), "[REDACTED]")
    for pat in _SECRET_VALUE_RES:
        text = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


def data_dir(env: Mapping[str, str]) -> Path:
    """Return the adapter state dir (same env keys as `ReceiptStore`; agy `PLUGIN_DATA` is ignored)."""
    for key in ("CC_AGY_DATA_DIR", "CLAUDE_PLUGIN_DATA"):
        if _usable(env.get(key)):
            return Path(str(env[key]))
    home = Path(env.get("HOME") or os.path.expanduser("~"))
    return home / ".cache" / "cc-agy-plugin"


# --------------------------------------------------------------------- normalize


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        val = payload.get(key)
        if val not in (None, ""):
            return val
    return None


def _opencode_response(output: Any) -> Any:
    if isinstance(output, dict):
        if isinstance(output.get("content"), list):
            texts = [str(c.get("text") or "") for c in output["content"] if isinstance(c, dict)]
            return "\n".join(t for t in texts if t)
        if "output" in output:
            return output.get("output")
    return output


def _opencode_prompt(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("prompt"), str):
        return payload["prompt"]
    parts = payload.get("parts")
    if not isinstance(parts, list):
        out = payload.get("output")
        parts = out.get("parts") if isinstance(out, dict) else None
    texts = [str(p.get("text") or "") for p in parts or [] if isinstance(p, dict) and p.get("type", "text") == "text"]
    return "\n".join(t for t in texts if t)


def normalize(host: str, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Map a host-native hook payload to the Claude Code hook payload."""
    p = payload if isinstance(payload, Mapping) else {}
    cc: dict[str, Any] = {"hook_event_name": event}
    if host == "codex":
        cc.update({k: v for k, v in p.items() if k != "hook_event_name"})
        if event in ("PreToolUse", "PostToolUse"):
            cc["tool_name"], cc["tool_input"] = hook_toolmap.map_tool("codex", "Bash", p.get("tool_input"))
        cc.setdefault("cwd", os.getcwd())
        return cc
    if host == "opencode":
        cc["session_id"] = _first(p, "sessionID", "session_id") or ""
        cc["cwd"] = _first(p, "cwd", "directory", "worktree") or os.getcwd()
        if event in ("PreToolUse", "PostToolUse"):
            cc["tool_name"], cc["tool_input"] = hook_toolmap.map_tool("opencode", p.get("tool"), p.get("args"))
            if p.get("callID"):
                cc["tool_use_id"] = p.get("callID")
        if event == "PostToolUse":
            cc["tool_response"] = _opencode_response(p.get("output"))
        if event == "UserPromptSubmit":
            cc["prompt"] = _opencode_prompt(p)
        if event == "SessionStart":
            cc["source"] = p.get("source") or "startup"
        if event == "Stop":
            cc["stop_hook_active"] = bool(_first(p, "stop_hook_active", "stopHookActive"))
        return cc
    # agy
    cc["session_id"] = _first(p, "conversationId", "conversation_id") or ""
    cc["cwd"] = _first(p, "cwd") or os.getcwd()
    if p.get("transcriptPath"):
        cc["transcript_path"] = p["transcriptPath"]
    tool_call = p.get("toolCall") if isinstance(p.get("toolCall"), dict) else {}
    if event in ("PreToolUse", "PostToolUse"):
        cc["tool_name"], cc["tool_input"] = hook_toolmap.map_tool("agy", tool_call.get("name"), tool_call.get("args"))
        if tool_call.get("id"):
            cc["tool_use_id"] = tool_call["id"]
    if event == "PostToolUse":
        result, error = p.get("result"), p.get("error")
        cc["tool_response"] = result if not error else {"result": result, "error": error}
    if event == "UserPromptSubmit":
        cc["prompt"] = str(p.get("lastUserInput") or "")
    if event == "SessionStart":
        cc["source"] = p.get("source") or "startup"
    if event == "Stop":
        cc["stop_hook_active"] = bool(_first(p, "continuationActive", "stop_hook_active"))
    return cc


# ------------------------------------------------------------------- run script


def load_hook_spec(event: str, root: Path = PLUGIN_ROOT) -> tuple[Path, int, str | None]:
    """Return `(script_path, timeout_s, matcher)` for `event` from hooks/hooks.json."""
    script, timeout = _DEFAULT_SCRIPTS[event]
    matcher: str | None = None
    data = _read_json_file(root / "hooks" / "hooks.json")
    try:
        group = data["hooks"][event][0]
        hook = group["hooks"][0]
        found = re.search(r"scripts/([A-Za-z0-9_]+\.py)", hook["command"])
        if found:
            script = found.group(1)
        timeout = int(hook.get("timeout") or timeout)
        matcher = group.get("matcher")
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return root / "scripts" / script, timeout, matcher


def matcher_allows(matcher: str | None, tool_name: str) -> bool:
    """Return True when a CC hooks.json matcher selects `tool_name` (empty matches all)."""
    if not matcher:
        return True
    try:
        return re.fullmatch(matcher, tool_name) is not None
    except re.error:
        return True


class HookScriptError(RuntimeError):
    """The mapped CC hook script crashed, timed out, or printed invalid JSON."""


def run_pre_tool_guard_inprocess(cc_payload: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any] | None:
    """Run `pre_tool_guard.evaluate_pre_tool_use` in-process; None when it can't be imported.

    Saves one Python start-up per guarded tool call. Import problems fall back to
    the subprocess path; an exception from the guard itself is a script crash.
    """
    try:
        import pre_tool_guard  # pylint: disable=import-outside-toplevel
    except Exception:  # noqa: BLE001 - any import failure means "use the subprocess".
        return None
    evaluate = getattr(pre_tool_guard, "evaluate_pre_tool_use", None)
    if not callable(evaluate):
        return None
    try:
        out = evaluate(dict(cc_payload), env=dict(env))
    except Exception as exc:  # noqa: BLE001
        raise HookScriptError(f"pre_tool_guard.py raised {type(exc).__name__}: {exc}") from exc
    return out if isinstance(out, dict) else {}


def run_script(script: Path, cc_payload: Mapping[str, Any], env: Mapping[str, str], timeout: int) -> dict[str, Any]:
    """Run a CC hook script with `cc_payload` on stdin and return its parsed JSON output."""
    cwd = str(cc_payload.get("cwd") or "")
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            input=json.dumps(cc_payload),
            capture_output=True,
            text=True,
            env=dict(env),
            cwd=cwd if cwd and os.path.isdir(cwd) else str(PLUGIN_ROOT),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HookScriptError(f"{script.name} timed out after {timeout}s") from exc
    except OSError as exc:
        raise HookScriptError(f"{script.name} could not start: {exc}") from exc
    if proc.returncode == 2:
        # CC convention: exit 2 blocks with stderr as the reason.
        return {"_exit2": (proc.stderr or "").strip() or f"{script.name} blocked"}
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        raise HookScriptError(f"{script.name} exited {proc.returncode}: {tail[0][:300]}")
    out = (proc.stdout or "").strip()
    if not out:
        return {}
    try:
        parsed = json.loads(out.splitlines()[-1])
    except ValueError as exc:
        raise HookScriptError(f"{script.name} printed non-JSON output") from exc
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------- denormalize


def interpret(event: str, cc_out: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce CC script output to `{deny, block, reason, context}`."""
    hso = cc_out.get("hookSpecificOutput") if isinstance(cc_out.get("hookSpecificOutput"), dict) else {}
    res: dict[str, Any] = {"deny": False, "block": False, "reason": None, "context": None}
    if "_exit2" in cc_out:
        res["reason"] = cc_out["_exit2"]
        res["deny" if event == "PreToolUse" else "block"] = event in ("PreToolUse", "Stop")
    if event == "PreToolUse" and hso.get("permissionDecision") == "deny":
        res["deny"] = True
        res["reason"] = hso.get("permissionDecisionReason") or "Blocked by cc-agy-plugin guard."
    if event == "Stop" and cc_out.get("decision") == "block":
        res["block"] = True
        res["reason"] = cc_out.get("reason") or "Blocked by cc-agy-plugin provenance check."
    texts = [t for t in (hso.get("additionalContext"), cc_out.get("systemMessage")) if isinstance(t, str) and t.strip()]
    if texts:
        res["context"] = "\n\n".join(texts)
    return res


def denormalize(host: str, event: str, res: Mapping[str, Any]) -> dict[str, Any] | None:
    """Translate an interpreted result into the host-native stdout object (None = no output)."""
    reason, ctx = res.get("reason"), res.get("context")
    if host == "opencode":
        return {
            "action": "deny" if res.get("deny") else "allow",
            "reason": reason if (res.get("deny") or res.get("block")) else None,
            "additionalContext": ctx,
            "block": bool(res.get("deny") or res.get("block")),
        }
    if host == "codex":
        if event == "PreToolUse":
            if res.get("deny"):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }
            return None  # Codex rejects additionalContext/allow on PreToolUse.
        if event == "Stop":
            return {"decision": "block", "reason": reason} if res.get("block") else None
        if ctx:
            return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": ctx}}
        return None
    # agy
    if event == "PreToolUse" and res.get("deny"):
        return {"decision": "deny", "reason": reason}
    if event == "Stop" and res.get("block"):
        return {"decision": "block", "reason": reason}
    if event in ("SessionStart", "UserPromptSubmit") and ctx:
        return {"injectSteps": [{"systemMessage": {"systemMessage": ctx}}]}
    return {}


# ----------------------------------------------------------------------- dedupe


def agy_prompt_is_new(env: Mapping[str, str], session_id: str, prompt: str) -> bool:
    """Record `prompt` for `session_id`; return False if it was already handled.

    agy has no UserPromptSubmit; its PreInvocation fires on every model call
    with the same `lastUserInput`, so the CC script must run once per prompt.
    """
    if not prompt.strip():
        return False
    state_path = data_dir(env) / "hook_adapter" / "agy_prompts.json"
    digest = hashlib.sha256(prompt.encode("utf-8", "replace")).hexdigest()
    state = _read_json_file(state_path)
    state = state if isinstance(state, dict) else {}
    key = session_id or "_"
    entry = state.get(key)
    if isinstance(entry, dict) and entry.get("sha") == digest:
        return False
    state[key] = {"sha": digest, "ts": time.time()}
    if len(state) > _DEDUPE_MAX_ENTRIES:
        ordered = sorted(state.items(), key=lambda kv: (kv[1] or {}).get("ts", 0) if isinstance(kv[1], dict) else 0)
        state = dict(ordered[-_DEDUPE_MAX_ENTRIES:])
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, state_path)
    except OSError:
        pass
    return True


# ------------------------------------------------------------------------- core


def is_worker_context(env: Mapping[str, str], payload: Mapping[str, Any]) -> bool:
    """Return True when the hook fires inside an agy worker the plugin itself spawned.

    Workers run in plugin-owned worktrees and must not be policed by the
    orchestrator's guards (strict mode would otherwise deny the worker's own
    edits). Primary signal: `CC_AGY_IN_WORKER=1`; fallback: a worktree `cwd`.
    """
    if str(env.get(WORKER_ENV_MARKER) or "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    cwd = str(_first(payload, "cwd", "directory", "worktree") or "")
    return any(part.startswith(_WORKTREE_DIR_PREFIXES) for part in Path(cwd).parts)


def _stop_active(host: str, payload: Mapping[str, Any]) -> bool:
    keys = ("continuationActive", "stop_hook_active") if host == "agy" else ("stop_hook_active", "stopHookActive")
    return bool(_first(payload, *keys))


def handle(host: str, event: str, payload: Any, base_env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Run one hook end to end and return the host-native stdout object (None = print nothing)."""
    event = EVENT_ALIASES.get(event, event)
    raw = payload if isinstance(payload, dict) else {}
    allow = {"deny": False, "block": False, "reason": None, "context": None}
    source_env = base_env if base_env is not None else os.environ
    if is_worker_context(source_env, raw):
        return denormalize(host, event, allow)
    extra = raw.get("env") if host == "opencode" and isinstance(raw.get("env"), dict) else None
    env = build_env(host, source_env, extra)
    try:
        cc = normalize(host, event, raw)
        script, timeout, matcher = load_hook_spec(event)
        if event in ("PreToolUse", "PostToolUse") and not matcher_allows(matcher, str(cc.get("tool_name") or "")):
            return denormalize(host, event, allow)
        if host == "agy" and event == "UserPromptSubmit":
            if not agy_prompt_is_new(env, str(cc.get("session_id") or ""), str(cc.get("prompt") or "")):
                return denormalize(host, event, allow)
        cc_out = run_pre_tool_guard_inprocess(cc, env) if event == "PreToolUse" else None
        if cc_out is None:
            cc_out = run_script(script, cc, env, timeout)
        res = interpret(event, cc_out)
    except Exception as exc:  # noqa: BLE001 - hooks must never crash the host.
        res = dict(allow)
        detail = f"{type(exc).__name__}: {exc}"
        if event == "PreToolUse" and is_strict(env):
            res["deny"] = True
            res["reason"] = (
                f"cc-agy-plugin strict delegation mode: the PreToolUse guard failed ({detail}), "
                "so this tool call is denied. Fix the plugin install "
                "(run session_start.py --doctor) or turn strict mode off."
            )
        elif event == "Stop" and is_strict(env):
            # Mirrors provenance_check.main(): strict mode blocks on an internal error.
            res["block"] = True
            res["reason"] = f"Provenance verification encountered an internal error: {detail}"
        else:
            sys.stderr.write(redact(f"cc-agy-plugin hook_adapter: {detail}\n", env))
    if event == "Stop" and res.get("block") and _stop_active(host, raw):
        res["block"] = False  # Re-entry guard: never block a continuation twice.
    for key in ("reason", "context"):
        res[key] = redact(res.get(key), env)
    return denormalize(host, event, res)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; always exits 0 (decisions are carried in stdout JSON)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True, choices=HOSTS)
    parser.add_argument("--event", required=True, choices=EVENTS + tuple(EVENT_ALIASES))
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.read().strip()
        payload = json.loads(raw) if raw else {}
    except (ValueError, OSError):
        payload = {}
    out = handle(args.host, args.event, payload)
    if out is not None:
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
