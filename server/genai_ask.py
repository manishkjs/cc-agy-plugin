"""Single-shot Vertex AI query helper using google-genai with crash-safe receipts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid

from server.genai_client import (
    build_genai_client,
    close_genai_client as _close_genai_client,
    extract_usage_metadata,
)
from server.receipts import ReceiptStore, enforce_daily_budget, resolve_workspace_root
from server.regions import (
    MAX_TOOL_RESPONSE_TOKENS,
    resolve_auth_config,
    resolve_default_ask_model,
    sanitize_placeholder_env,
    truncate_to_token_budget,
)

GenaiBackend = Callable[[dict[str, Any]], dict[str, Any]]


def _default_genai_backend(request: dict[str, Any]) -> dict[str, Any]:
    """Invoke Gemini Enterprise (`agy` CLI), Google AI Studio, or Vertex AI."""
    sanitize_placeholder_env()
    if request.get("auth_mode") == "gemini_enterprise":
        from server.agy_cli import run_agy_cli_ask

        return run_agy_cli_ask(request)

    from google.genai import types as genai_types

    timeout_ms = int(request.get("timeout_ms") or 600_000)
    req_tl = str(request.get("thinking_level") or "").strip().upper()
    is_zero_thinking = req_tl in ("ZERO", "0", "OFF")
    client = build_genai_client(
        request,
        timeout_ms=timeout_ms,
        zero_thinking=is_zero_thinking,
    )
    try:
        cfg_kwargs: dict[str, Any] = {}
        if request.get("system_instruction"):
            cfg_kwargs["system_instruction"] = request["system_instruction"]
        if request.get("temperature") is not None:
            cfg_kwargs["temperature"] = float(request["temperature"])
        if is_zero_thinking:
            cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=0
            )
        elif req_tl:
            cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_level=req_tl
            )

        config = genai_types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None
        response = client.models.generate_content(
            model=request["model"],
            contents=request["prompt"],
            config=config,
        )
        return {
            "text": getattr(response, "text", "") or "",
            "usage": extract_usage_metadata(getattr(response, "usage_metadata", None)),
        }
    finally:
        _close_genai_client(client)


def ask_gemini(
    prompt: str,
    *,
    model: str | None = None,
    context_files: list[str] | None = None,
    system_instruction: str | None = None,
    temperature: float | None = None,
    thinking_level: str | None = None,
    timeout_ms: int | None = None,
    project: str | None = None,
    location: str | None = None,
    api_key: str | None = None,
    workspace_root: str | Path | None = None,
    store: ReceiptStore | None = None,
    scrub_receipt: bool = False,
    backend: GenaiBackend | None = None,
    record_receipt: bool = True,
) -> dict[str, Any]:
    """Run a single-shot Google AI Studio or Vertex AI Gemini query without spinning up Antigravity.

    Ideal for code reviews, architecture checks, or second opinions where the
    Antigravity SDK's built-in agent system prompt and tool declarations are unnecessary.
    Always writes a receipt in a `finally:` block so errors still record partial usage.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    if context_files is not None and (
        not isinstance(context_files, list)
        or not all(isinstance(p, str) for p in context_files)
    ):
        raise ValueError("context_files must be a list of relative file path strings")

    ws_root = resolve_workspace_root(workspace_root)
    receipt_store = store or ReceiptStore(workspace_root=ws_root)
    enforce_daily_budget(receipt_store, workspace_root=ws_root)

    chosen_model = resolve_default_ask_model(model, env=os.environ)

    auth = resolve_auth_config(
        requested_project=project,
        requested_location=location,
        requested_api_key=api_key,
        model=chosen_model,
    )
    auth_mode = auth["auth_mode"]
    resolved_key = auth.get("api_key")
    resolved_project = auth.get("project")
    resolved_location = auth.get("location")
    reroute_note = auth.get("reroute_note")

    assembled_parts: list[str] = []
    included_files: list[str] = []
    for rel_path in context_files or []:
        raw_target = ws_root / rel_path
        candidate = raw_target.resolve()
        if not candidate.is_relative_to(ws_root):
            raise ValueError(
                f"context_file {rel_path!r} escapes workspace root {ws_root}"
            )
        if not candidate.is_file():
            raise ValueError(f"context_file not found: {rel_path!r}")
        content = candidate.read_text(encoding="utf-8", errors="replace")
        assembled_parts.append(f"### File: {rel_path}\n```\n{content}\n```")
        included_files.append(rel_path)

    assembled_parts.append(prompt.strip())
    full_prompt = "\n\n".join(assembled_parts)

    job_id = f"ask-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    started_at = round(time.time(), 3)
    usage: dict[str, int] = {
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "total_token_count": 0,
    }
    status = "error"
    error_msg: str | None = None
    response_text = ""

    active_backend = backend or _default_genai_backend
    try:
        req_payload: dict[str, Any] = {
            "auth_mode": auth_mode,
            "api_key": resolved_key,
            "project": resolved_project,
            "location": resolved_location,
            "location_explicit": bool(auth.get("location_explicit")),
            "model": chosen_model,
            "prompt": full_prompt,
            "system_instruction": system_instruction,
            "temperature": temperature,
        }
        if thinking_level:
            req_payload["thinking_level"] = thinking_level
        if timeout_ms is not None and int(timeout_ms) > 0:
            req_payload["timeout_ms"] = int(timeout_ms)
        result = active_backend(req_payload)
        if result.get("vertex_project"):
            resolved_project = str(result["vertex_project"])
        if result.get("vertex_location"):
            resolved_location = str(result["vertex_location"])
        if result.get("reroute_note") and not reroute_note:
            reroute_note = str(result["reroute_note"])
        response_text = str(result.get("text") or "")
        if isinstance(result.get("usage"), dict):
            for k in usage:
                usage[k] = int(result["usage"].get(k) or 0)
        if usage["total_token_count"] == 0:
            usage["total_token_count"] = (
                usage["prompt_token_count"]
                + usage["candidates_token_count"]
                + usage["thoughts_token_count"]
            )
        status = "ok"
    except (TimeoutError, asyncio.TimeoutError) as exc:
        status = "timeout"
        error_msg = f"TimeoutError: {exc}"
        partial = getattr(exc, "partial_usage", None)
        if isinstance(partial, dict):
            for k in usage:
                usage[k] = int(partial.get(k) or 0)
        raise
    except Exception as exc:
        status = "error"
        error_msg = f"{type(exc).__name__}: {exc}"
        partial = getattr(exc, "partial_usage", None)
        if isinstance(partial, dict):
            for k in usage:
                usage[k] = int(partial.get(k) or 0)
        raise
    finally:
        finished_at = round(time.time(), 3)
        clipped_text, was_truncated, orig_tokens = truncate_to_token_budget(
            response_text, MAX_TOOL_RESPONSE_TOKENS
        )
        receipt_payload = {
            "job_id": job_id,
            "kind": "ask",
            "status": status,
            "auth_mode": auth_mode,
            "model": chosen_model,
            "vertex_project": resolved_project,
            "vertex_location": resolved_location,
            "reroute_note": reroute_note,
            "context_files": included_files,
            "started_at_epoch": started_at,
            "finished_at_epoch": finished_at,
            "duration_seconds": round(max(0.0, finished_at - started_at), 3),
            "usage": usage,
            "response_tokens_estimated": orig_tokens,
            "response_truncated": was_truncated,
            "error": error_msg,
            "applied": False,
        }
        if record_receipt:
            saved_receipt = receipt_store.write_receipt(
                job_id,
                receipt_payload,
                scrub=scrub_receipt,
                workspace_root=str(ws_root),
                api_key=resolved_key,
            )
            computed_cost = saved_receipt["cost"]
        else:
            from server.regions import estimate_cost_usd

            computed_cost = estimate_cost_usd(
                chosen_model,
                usage,
                single_request=True,
            )

    return {
        "job_id": job_id,
        "status": status,
        "auth_mode": auth_mode,
        "model": chosen_model,
        "vertex_location": resolved_location,
        "reroute_note": reroute_note,
        "context_files": included_files,
        "text": clipped_text,
        "response_truncated": was_truncated,
        "response_tokens_estimated": orig_tokens,
        "usage": usage,
        "cost": computed_cost,
    }
