"""Antigravity SDK worker adapter with crash-safe usage/tool extraction."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
import time
from typing import Any, Callable

from server.regions import resolve_vertex_location, resolve_vertex_project


async def maybe_await(value: Any) -> Any:
    """Resolve value whether it is a plain return value, coroutine, or awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value


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
    )
    model_cfg = types.ModelConfig(
        model=request["model"],
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

    # CRITICAL (D11): Always set vertex=True at top-level of LocalAgentConfig so the
    # Antigravity SDK does not inject the default GeminiAPIEndpoint image model
    # (which crashes when GEMINI_API_KEY is unset in Vertex-only environments).
    cfg = ag.LocalAgentConfig(
        workspace=Path(request["workspace_dir"]),
        vertex=True,
        model=model_cfg,
        policies=[policy.allow_all()],
        capabilities=capabilities,
    )

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
                turn_usage = extract_usage_from_session(agent, resp)
                for k in usage:
                    usage[k] = max(usage[k], turn_usage[k])
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
    }


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
) -> dict[str, Any]:
    """Run an Antigravity worker synchronously or via custom test backend."""
    resolved_project = resolve_vertex_project(project)
    resolved_location, reroute_note = resolve_vertex_location(model, location)

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
        "vertex_project": resolved_project,
        "vertex_location": resolved_location,
        "reroute_note": reroute_note,
        "duration_seconds": elapsed,
    }
