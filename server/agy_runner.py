"""Antigravity SDK worker adapter with crash-safe usage/tool extraction."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable

from server.regions import resolve_vertex_location, resolve_vertex_project


async def maybe_await(value: Any) -> Any:
    """Resolve value whether it is a plain return value, coroutine, or awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value


THINKING_LEVELS = ("MINIMAL", "LOW", "MEDIUM", "HIGH", "EXTRA_HIGH")
PROGRESS_KINDS = ("milestone", "question", "blocker")
_progress_lock = threading.Lock()


def append_progress(progress_path: str | Path | None, kind: str, message: str) -> str:
    """Append one worker -> lead signal (milestone / question / blocker) as a JSON line."""
    if not progress_path:
        return "No progress channel is configured for this job."
    clean_kind = str(kind or "milestone").strip().lower()
    if clean_kind not in PROGRESS_KINDS:
        clean_kind = "milestone"
    entry = {"ts": round(time.time(), 3), "kind": clean_kind, "message": str(message or "")[:2000]}
    with _progress_lock:
        Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
        with open(progress_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    if clean_kind in ("question", "blocker"):
        return (
            "Reported to the lead. Stop now: end your turn with a short summary of what is done and what "
            "you need. The lead will reply with a follow-up."
        )
    return "Milestone recorded. Continue."


def read_progress(progress_path: str | Path | None, limit: int = 8) -> list[dict[str, Any]]:
    """Return the most recent worker -> lead signals for a job."""
    if not progress_path or not Path(progress_path).exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in Path(progress_path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-max(1, limit):]


def resolve_thinking_level(value: str | None) -> str | None:
    """Return the worker thinking level (default HIGH); "DEFAULT"/"NONE" leaves it to the model's own default."""
    raw = (value or os.environ.get("CC_AGY_DELEGATE_THINKING_LEVEL") or "HIGH").strip().upper()
    if raw.startswith("${"):
        raw = "HIGH"
    if raw in ("DEFAULT", "NONE", ""):
        return None
    if raw not in THINKING_LEVELS:
        raise ValueError(f"thinking_level must be one of {THINKING_LEVELS} or DEFAULT, got {value!r}")
    return raw


async def resolve_turn_usage(resp: Any) -> dict[str, int]:
    """Read one chat turn's usage from `resp.usage_metadata` (awaitable on google-antigravity 0.1.x)."""
    usage = {
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "total_token_count": 0,
    }
    if resp is None:
        return usage
    try:
        raw = await maybe_await(getattr(resp, "usage_metadata", None))
    except Exception:
        return usage
    if raw is None:
        return usage
    data = raw.model_dump() if hasattr(raw, "model_dump") else (dict(raw) if isinstance(raw, dict) else {})
    if not data:
        data = {k: getattr(raw, k, 0) for k in usage}
    for k in usage:
        usage[k] = int(data.get(k) or 0)
    if usage["total_token_count"] == 0:
        usage["total_token_count"] = (
            usage["prompt_token_count"] + usage["candidates_token_count"] + usage["thoughts_token_count"]
        )
    return usage


def extract_usage_from_session(agent: Any, resp: Any = None) -> dict[str, int]:
    """Extract raw Vertex token counters from an Antigravity Agent or Response.

    Safe to call inside a `finally:` block even when `resp.resolve()` timed out or
    raised an exception mid-turn, inspecting both the response object and any
    completed turns recorded on the agent session.
    """
    usage = {
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "total_token_count": 0,
    }

    candidates = []
    if resp is not None:
        candidates.append(resp)
        for attr in ("usage_metadata", "usage", "_usage"):
            sub = getattr(resp, attr, None)
            if sub is not None:
                candidates.append(sub)

    if agent is not None:
        candidates.append(agent)
        for attr in ("usage_metadata", "usage", "total_usage", "_usage", "session"):
            sub = getattr(agent, attr, None)
            if sub is not None:
                candidates.append(sub)

    for obj in candidates:
        if isinstance(obj, dict):
            for k in usage:
                val = int(obj.get(k) or 0)
                if val > usage[k]:
                    usage[k] = val
        else:
            for k in usage:
                val = int(getattr(obj, k, 0) or 0)
                if val > usage[k]:
                    usage[k] = val

    if usage["total_token_count"] == 0:
        usage["total_token_count"] = (
            usage["prompt_token_count"]
            + usage["candidates_token_count"]
            + usage["thoughts_token_count"]
        )
    return usage


async def drain_response_streams(
    resp: Any,
    tool_calls_out: list[dict[str, Any]],
    thoughts_out: list[str],
) -> None:
    """Drain async or sync `resp.tool_calls` and `resp.thoughts` streams safely."""
    if resp is None:
        return

    tc_stream = getattr(resp, "tool_calls", None)
    if tc_stream is not None:
        if hasattr(tc_stream, "__aiter__"):
            async for tc in tc_stream:
                tool_calls_out.append(_serialize_tool_call(tc))
        elif hasattr(tc_stream, "__iter__"):
            for tc in tc_stream:
                tool_calls_out.append(_serialize_tool_call(tc))

    th_stream = getattr(resp, "thoughts", None)
    if th_stream is not None:
        if hasattr(th_stream, "__aiter__"):
            async for th in th_stream:
                thoughts_out.append(str(getattr(th, "text", th)))
        elif hasattr(th_stream, "__iter__"):
            for th in th_stream:
                thoughts_out.append(str(getattr(th, "text", th)))


def _serialize_tool_call(tc: Any) -> dict[str, Any]:
    if isinstance(tc, dict):
        return {
            "name": str(tc.get("name") or "unknown"),
            "args": tc.get("args") or {},
        }
    return {
        "name": str(getattr(tc, "name", None) or getattr(tc, "tool_name", "unknown")),
        "args": getattr(tc, "args", None) or getattr(tc, "arguments", {}) or {},
    }


async def _default_antigravity_backend(request: dict[str, Any]) -> dict[str, Any]:
    """Execute an agentic task inside `workspace_dir` using `google.antigravity`."""
    from google import antigravity as ag
    from google.antigravity import policy, types

    endpoint = types.VertexEndpoint(
        project=request["project"],
        location=request["location"],
        options=(
            types.GeminiModelOptions(thinking_level=getattr(types.ThinkingLevel, request["thinking_level"]))
            if request.get("thinking_level")
            else None
        ),
    )
    # google-antigravity 0.1.x: the model is a ModelTarget(name, types, endpoint).
    model_cfg = types.ModelTarget(
        name=request["model"],
        types=[types.ModelType.TEXT],
        endpoint=endpoint,
    )

    cap_kwargs: dict[str, Any] = {
        "enable_subagents": bool(request.get("enable_subagents", False)),
    }
    if request.get("enabled_tools") is not None:
        cap_kwargs["enabled_tools"] = list(request["enabled_tools"])
    if request.get("disabled_tools") is not None:
        cap_kwargs["disabled_tools"] = list(request["disabled_tools"])

    capabilities = types.CapabilitiesConfig(**cap_kwargs)

    # Keep the SDK's own session/artifact files outside the repo so they never show up in the diff.
    save_dir = tempfile.mkdtemp(prefix="cc-agy-save-")

    # Hard caps so one job cannot read tens of millions of tokens; when hit, the turn stops with a
    # MAX_*_EXCEEDED stop reason and the job becomes a "checkpoint" the lead can continue or redirect.
    budget_kwargs = {
        k: int(request[k]) for k in ("max_tool_calls", "max_total_tokens") if request.get(k)
    }
    budget_cfg = types.BudgetConfig(**budget_kwargs) if budget_kwargs else None

    progress_path = request.get("progress_path")

    def report_to_lead(kind: str, message: str) -> str:
        """Send a short signal to the lead engineer.

        Args:
            kind: "milestone" for progress (e.g. tests now 40/42 passing), "question" when a requirement
                is ambiguous, or "blocker" when you cannot proceed. Questions and blockers end your turn.
            message: One or two sentences for the lead.
        """
        return append_progress(progress_path, kind, message)

    custom_tools: list[Any] = []
    if progress_path:
        custom_tools.append(report_to_lead)

    containment_mode = "none"
    use_containment = os.environ.get("CC_AGY_CONTAINMENT", "1").lower() not in ("0", "false", "no")
    if use_containment:
        try:
            from ctx.sdk import ToolSession

            session = ToolSession(
                Path(request["workspace_dir"]),
                timeout=float(request.get("timeout_seconds") or 600.0),
                producer="cc-agy-plugin",
            )

            async def run_command(command: str, working_directory: str = "", timeout_seconds: float = 600.0) -> str:
                """Execute a shell command with typed status and bounded output. Large output (>240 lines or >16KB)
                is saved to a local evidence store and returns a concise digest with a run address (run:<id>#stdout).
                Use ctx_get or ctx_search to retrieve omitted lines."""
                result = await session.run(command, working_directory=working_directory, timeout_seconds=timeout_seconds)
                return result.to_json()

            async def ctx_get(address: str, start_line: int = 1, end_line: int = 100) -> str:
                """Retrieve a bounded, inclusive line range from an evidence address (e.g. 'run:<id>#stdout')."""
                return await session.get(address, start_line=start_line, end_line=end_line)

            async def ctx_search(address: str, pattern: str) -> str:
                """Search stored evidence (e.g. 'run:<id>#stdout') or repository files for a regex pattern."""
                return await session.search(address, pattern)

            custom_tools.extend([run_command, ctx_get, ctx_search])
            disabled_list = list(cap_kwargs.get("disabled_tools") or [])
            if types.BuiltinTools.RUN_COMMAND not in disabled_list:
                disabled_list.append(types.BuiltinTools.RUN_COMMAND)
            cap_kwargs["disabled_tools"] = disabled_list
            capabilities = types.CapabilitiesConfig(**cap_kwargs)
            containment_mode = "straitjacket"
        except (ImportError, Exception):
            pass

    # CRITICAL (D11): Always set vertex=True at top-level of LocalAgentConfig so the
    # Antigravity SDK does not inject the default GeminiAPIEndpoint image model
    # (which crashes when GEMINI_API_KEY is unset in Vertex-only environments).
    cfg_kwargs: dict[str, Any] = dict(
        workspaces=[str(Path(request["workspace_dir"]))],
        vertex=True,
        project=request["project"],
        location=request["location"],
        model=model_cfg,
        policies=[policy.allow_all()],
        capabilities=capabilities,
        save_dir=save_dir,
    )
    if budget_cfg is not None:
        cfg_kwargs["budget_config"] = budget_cfg
    if custom_tools:
        cfg_kwargs["tools"] = custom_tools
    cfg = ag.LocalAgentConfig(**cfg_kwargs)

    tool_calls_log: list[dict[str, Any]] = []
    thoughts_log: list[str] = []
    usage: dict[str, int] = {
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "total_token_count": 0,
    }
    text_out = ""
    resp = None
    agent_ref = None
    timeout_sec = float(request.get("timeout_seconds") or 300.0)
    stop_reason: str | None = None

    try:
        async with ag.Agent(cfg) as agent:
            agent_ref = agent
            for idx, prompt_turn in enumerate(request["prompts"]):
                if idx == 0 or not hasattr(agent, "continue_session"):
                    resp = await maybe_await(agent.chat(prompt_turn))
                else:
                    resp = await maybe_await(agent.continue_session(prompt_turn))
                await asyncio.wait_for(
                    maybe_await(resp.resolve()),
                    timeout=timeout_sec,
                )
                await drain_response_streams(resp, tool_calls_log, thoughts_log)
                text_fn = getattr(resp, "text", None)
                if callable(text_fn):
                    text_out = str((await maybe_await(text_fn())) or "")
                elif isinstance(text_fn, str):
                    text_out = text_fn
                # usage_metadata is awaitable on the real SDK; each chat turn reports its own
                # usage, so turns are summed (not maxed).
                turn_usage = await resolve_turn_usage(resp)
                if not any(turn_usage.values()):
                    turn_usage = extract_usage_from_session(agent, resp)
                for k in usage:
                    usage[k] += turn_usage[k]
                stop_reason = await resolve_stop_reason(resp)
    except Exception as exc:
        # Attach partial usage and tool calls so the caller's finally block
        # records everything consumed prior to the failure or timeout.
        try:
            await drain_response_streams(resp, tool_calls_log, thoughts_log)
        except Exception:
            pass
        partial_usage = extract_usage_from_session(agent_ref, resp)
        for k in usage:
            usage[k] = max(usage[k], partial_usage[k])
        setattr(exc, "partial_usage", usage)
        setattr(exc, "partial_tool_calls", tool_calls_log)
        setattr(exc, "partial_thoughts", thoughts_log)
        raise

    return {
        "text": text_out,
        "usage": usage,
        "tool_calls": tool_calls_log,
        "thoughts": thoughts_log,
        "stop_reason": stop_reason,
        "containment": containment_mode,
    }


async def resolve_stop_reason(resp: Any) -> str | None:
    """Return the SDK stop reason name (e.g. MAX_TOOL_CALLS_EXCEEDED) or None for a normal finish."""
    try:
        raw = await maybe_await(getattr(resp, "stop_reason", None))
    except Exception:
        return None
    if raw is None:
        return None
    name = str(getattr(raw, "name", None) or getattr(raw, "value", None) or raw).upper()
    name = name.rsplit(".", 1)[-1]
    if name in ("", "NONE", "UNSPECIFIED", "STOP_REASON_UNSPECIFIED"):
        return None
    return name


def run_antigravity_worker(
    *,
    prompts: list[str],
    workspace_dir: str | Path,
    model: str,
    project: str | None = None,
    location: str | None = None,
    timeout_seconds: float = 300.0,
    enabled_tools: list[str] | None = None,
    disabled_tools: list[str] | None = None,
    enable_subagents: bool = False,
    backend: Callable[[dict[str, Any]], Any] | None = None,
    thinking_level: str | None = None,
    max_tool_calls: int | None = None,
    max_total_tokens: int | None = None,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run an Antigravity worker synchronously or via custom test backend."""
    resolved_project = resolve_vertex_project(project)
    resolved_location, reroute_note = resolve_vertex_location(model, location)
    resolved_thinking = resolve_thinking_level(thinking_level)

    req = {
        "prompts": list(prompts),
        "workspace_dir": str(Path(workspace_dir).resolve()),
        "model": model,
        "project": resolved_project,
        "location": resolved_location,
        "reroute_note": reroute_note,
        "timeout_seconds": timeout_seconds,
        "enabled_tools": enabled_tools,
        "disabled_tools": disabled_tools,
        "enable_subagents": enable_subagents,
        "thinking_level": resolved_thinking,
        "max_tool_calls": max_tool_calls,
        "max_total_tokens": max_total_tokens,
        "progress_path": str(progress_path) if progress_path else None,
    }

    active_backend = backend or _default_antigravity_backend
    started = time.time()
    result = active_backend(req)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    elapsed = round(max(0.0, time.time() - started), 3)

    return {
        "text": str(result.get("text") or ""),
        "usage": dict(result.get("usage") or {}),
        "tool_calls": list(result.get("tool_calls") or []),
        "thoughts": list(result.get("thoughts") or []),
        "stop_reason": result.get("stop_reason"),
        "containment": result.get("containment", "none"),
        "vertex_project": resolved_project,
        "vertex_location": resolved_location,
        "reroute_note": reroute_note,
        "thinking_level": resolved_thinking,
        "duration_seconds": elapsed,
    }
