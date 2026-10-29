"""Single-shot file writers and the fallback ladder for batch file tasks.

A writer gets exactly one :class:`~server.batch_plan.FileTask` plus the files it needs to see, and
answers with one strict JSON object. It never explores the repository and never runs anything:
the plugin applies the answer inside the job copy and runs the task's checks itself.

Answer shape (see :data:`ANSWER_SCHEMA`)::

    {"path": "<task path>", "content": "<whole file>"}                    # create, or full rewrite
    {"path": "<task path>", "edits": [{"find": "...", "replace": "..."}]}  # modify by exact edits

Answers are never repaired: no code-fence stripping, no fuzzy matching. A wrong path, an empty file,
or a ``find`` that does not match exactly once rejects the answer and costs an attempt.

The ladder (:data:`WRITER_LADDER`) gives a task two attempts on the routed low-cost writer (the
second one sees the failing check output) and a third on Claude. Transient vendor failures wait
with jittered, capped backoff and are not attempts; a reply cut off at the output limit moves to
the next writer for free; 401/403 stops the whole batch.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable

from server.batch_plan import FileTask, is_protected_path
from server.genai_client import (
    build_genai_client as _build_genai_client,
    close_genai_client,
    extract_usage_metadata,
)

WRITER_LADDER: tuple[str, ...] = ("routed", "routed", "claude")
TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})
AUTH_STATUSES = frozenset({401, 403})
MAX_TRANSIENT_WAITS = 6
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 60.0
DEFAULT_MAX_OUTPUT_TOKENS = 32_768

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"find": {"type": "string"}, "replace": {"type": "string"}},
                "required": ["find", "replace"],
            },
        },
    },
    "required": ["path"],
}

WRITER_SYSTEM_PROMPT = (
    "You write exactly one file for a larger change planned by a lead engineer. "
    "Use only the task and the files provided; do not ask questions or explain. "
    "Reply with one JSON object: {\"path\", \"content\"} for a new or fully rewritten file, or "
    "{\"path\", \"edits\": [{\"find\", \"replace\"}]} where every find is copied verbatim from the "
    "current file and occurs exactly once."
)


# --------------------------------------------------------------------------------------------
# Errors and data
# --------------------------------------------------------------------------------------------


class AnswerRejected(ValueError):
    """The writer's answer cannot be applied as given."""


class WriterTransportError(RuntimeError):
    """Transient vendor or network failure (retry after a wait; not an attempt)."""

    def __init__(self, status: int | None, message: str = "") -> None:
        self.status = status
        super().__init__(f"transient writer failure (status={status}) {message}".strip())


class WriterConfigError(RuntimeError):
    """The writer cannot run at all (missing CLI, bad flags, unreadable output): skip the rung, not an attempt."""


class WriterAuthError(RuntimeError):
    """401/403 from a writer's vendor: stop the batch instead of burning attempts."""

    def __init__(self, status: int, message: str = "") -> None:
        self.status = status
        super().__init__(f"writer authentication failed ({status}): {message}".strip())


@dataclass
class WriterRequest:
    task: FileTask
    prompt: str
    system: str = WRITER_SYSTEM_PROMPT
    shared_prefix: str = ""
    failure_tail: str = ""
    model: str = ""
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


@dataclass
class WriterReply:
    text: str
    cut_off: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    cost_cents: float = 0.0
    model: str = ""
    pricing_known: bool = True
    diagnostics: str = ""


@dataclass
class AppliedAnswer:
    path: str
    new_bytes: bytes
    previous_bytes: bytes | None


@dataclass
class LadderOutcome:
    state: str  # "landed" | "failed" | "held" (batch stopped before the task finished)
    attempts: int
    writer_used: str
    detail: str = ""
    answer: AppliedAnswer | None = None
    cost_cents: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    transient_waits: int = 0
    pricing_known: bool = True
    history: list[dict[str, Any]] = field(default_factory=list)


Writer = Callable[[WriterRequest], WriterReply]


def classify_status(status: int | None) -> str:
    """Map an HTTP status code to ``transient``, ``auth`` or ``other``; never inspects error text."""
    if status is None:
        return "other"
    if status in AUTH_STATUSES:
        return "auth"
    if status in TRANSIENT_STATUSES or 500 <= int(status) <= 599:
        return "transient"
    return "other"


# --------------------------------------------------------------------------------------------
# Answer parsing and application
# --------------------------------------------------------------------------------------------


def _decode_answer(raw: Any) -> dict[str, Any]:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="strict")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AnswerRejected(f"answer is not a single JSON object: {exc.msg} at char {exc.pos}") from None
    if not isinstance(raw, dict):
        raise AnswerRejected("answer must be a JSON object")
    return raw


def _detect_eol(data: bytes) -> str:
    """CRLF only when every line ending in the file is CRLF; mixed files are edited as-is."""
    crlf = data.count(b"\r\n")
    return "\r\n" if crlf and crlf == data.count(b"\n") else "\n"


def symlink_component(root: str | Path, rel: str) -> str | None:
    """Return the first existing component of ``rel`` under ``root`` that is a symlink, else None.

    Walks the unresolved path one component at a time, so an alias such as ``safe.txt -> .env`` or a
    linked directory is caught before anything follows it.
    """
    base = Path(root)
    current = base
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            return current.relative_to(base).as_posix()
    return None


def apply_writer_answer(
    task: FileTask,
    raw: Any,
    root: str | Path,
    *,
    edits_only: bool = False,
    allow_existing: bool = False,
) -> AppliedAnswer:
    """Validate an answer against its task and return the bytes to write; raise AnswerRejected otherwise.

    ``allow_existing`` lets a 'create' task overwrite a file this same job created earlier (a resumed run).
    """
    answer = _decode_answer(raw)
    path = answer.get("path")
    if path != task.path:
        raise AnswerRejected(f"answer path {path!r} does not match the task path {task.path!r}")
    posix = PurePosixPath(task.path)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts or is_protected_path(task.path):
        raise AnswerRejected(f"answer path {path!r} is not a safe workspace path")
    root_path = Path(root).resolve()
    # Check every unresolved component first: resolve() follows links, so a check on the resolved
    # path alone can never see that safe.txt is an alias for .env.
    link = symlink_component(root_path, task.path)
    if link is not None:
        raise AnswerRejected(f"{task.path!r} goes through a symlink ({link!r})")
    target = (root_path / posix.as_posix()).resolve()
    if not target.is_relative_to(root_path) or target == root_path:
        raise AnswerRejected(f"answer path {path!r} is not a safe workspace path")
    resolved_rel = target.relative_to(root_path).as_posix()
    if is_protected_path(resolved_rel):
        raise AnswerRejected(f"answer path {path!r} resolves to the protected path {resolved_rel!r}")

    has_content = "content" in answer
    has_edits = "edits" in answer
    if has_content and has_edits:
        raise AnswerRejected("answer has both 'content' and 'edits'; send one")
    if not has_content and not has_edits:
        raise AnswerRejected("answer has neither 'content' nor 'edits'")

    previous = target.read_bytes() if target.is_file() else None
    if task.action == "modify" and previous is None:
        raise AnswerRejected(f"cannot modify {task.path!r}: the file does not exist")
    if task.action == "create" and previous is not None and not allow_existing:
        raise AnswerRejected(f"cannot create {task.path!r}: the file already exists")
    if task.action == "create" and has_edits:
        raise AnswerRejected("a 'create' task needs 'content', not 'edits'")
    if edits_only and has_content:
        raise AnswerRejected("this task only accepts exact 'edits', not a whole-file 'content'")

    if has_content:
        content = answer.get("content")
        if not isinstance(content, str) or not content.strip():
            raise AnswerRejected("answer 'content' is empty")
        if previous is not None and _detect_eol(previous) == "\r\n":
            content = content.replace("\r\n", "\n").replace("\n", "\r\n")
        return AppliedAnswer(task.path, content.encode("utf-8"), previous)

    edits = answer.get("edits")
    if not isinstance(edits, list) or not edits:
        raise AnswerRejected("answer 'edits' must be a non-empty list")
    if previous is None:
        raise AnswerRejected(f"cannot apply edits: {task.path!r} does not exist")
    try:
        original = previous.decode("utf-8")
    except UnicodeDecodeError:
        raise AnswerRejected(f"{task.path!r} is not UTF-8 text; edits cannot be applied") from None
    eol = _detect_eol(previous)
    text = original.replace("\r\n", "\n") if eol == "\r\n" else original
    for idx, edit in enumerate(edits):
        if not isinstance(edit, dict) or not isinstance(edit.get("find"), str) or not isinstance(edit.get("replace"), str):
            raise AnswerRejected(f"edits[{idx}] must have string 'find' and 'replace'")
        find = edit["find"].replace("\r\n", "\n")
        replace = edit["replace"].replace("\r\n", "\n")
        if not find:
            raise AnswerRejected(f"edits[{idx}].find is empty")
        count = text.count(find)
        if count != 1:
            raise AnswerRejected(f"edits[{idx}].find matches {count} times in {task.path!r}; it must match exactly once")
        text = text.replace(find, replace, 1)
    if eol == "\r\n":
        text = text.replace("\n", "\r\n")
    new_bytes = text.encode("utf-8")
    if not new_bytes.strip():
        raise AnswerRejected("edits leave the file empty")
    return AppliedAnswer(task.path, new_bytes, previous)


# --------------------------------------------------------------------------------------------
# Ladder
# --------------------------------------------------------------------------------------------


def _resolve_ladder(routed: str) -> list[str]:
    return [routed if rung == "routed" else rung for rung in WRITER_LADDER]


def _add_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    for key, val in (usage or {}).items():
        if isinstance(val, (int, float)):
            total[key] = int(total.get(key, 0)) + int(val)


def _next_distinct(rungs: list[str], idx: int) -> int:
    name = rungs[idx]
    nxt = idx + 1
    while nxt < len(rungs) and rungs[nxt] == name:
        nxt += 1
    return nxt


def run_writer_ladder(
    task: FileTask,
    *,
    routed: str,
    writers: Any,
    verify: Callable[[AppliedAnswer], tuple[bool, str]],
    build_request: Callable[[FileTask, str, str], WriterRequest] | None = None,
    root: str | Path | None = None,
    edits_only: bool = False,
    allow_existing: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
    before_call: Callable[[str], bool | None] | None = None,
    on_transient: Callable[[float], None] | None = None,
    after_call: Callable[[str, WriterReply | None], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> LadderOutcome:
    """Run one task up the writer ladder until an answer passes verification or the ladder ends.

    ``verify`` writes the answer into the job copy, runs the task's checks and returns
    ``(passed, failure_tail)``; it must restore the previous file content when checks fail.
    ``before_call`` runs before every writer call (batch-wide throttle, budget and escalation cap);
    returning False means the call is not admitted: the task is ``held`` when ``should_stop`` reports
    a batch stop (budget reached, cancel, auth failure), otherwise that rung is skipped.
    ``after_call`` runs after every admitted call, with the reply or None when the call raised, so
    spend is recorded the moment it happens rather than when the whole ladder ends.
    ``on_transient`` is told about every transient wait so other tasks can back off too.
    Network failures that carry no HTTP status (connection reset, client timeout) count as transient
    by exception type; everything else is decided from the status code only.
    """
    rungs = _resolve_ladder(routed)
    workspace = Path(root) if root is not None else Path.cwd()
    make = build_request or (lambda t, name, tail: WriterRequest(task=t, prompt=t.intent, failure_tail=tail))
    outcome = LadderOutcome(state="failed", attempts=0, writer_used="")
    failure_tail = ""
    waits_by_writer: dict[str, int] = {}
    idx = 0

    def note(name: str, result: str, detail: str = "", cost: float = 0.0) -> None:
        outcome.history.append({"writer": name, "result": result, "detail": detail[-300:], "cost_cents": round(cost, 4)})

    def hold(name: str) -> LadderOutcome:
        note(name, "stopped", "batch stopped before this call")
        outcome.state = "held"
        outcome.detail = outcome.detail or "batch stopped before this task finished"
        return outcome

    while idx < len(rungs):
        name = rungs[idx]
        writer = writers.get(name)
        if writer is None:
            idx += 1
            continue
        if should_stop is not None and should_stop():
            return hold(name)
        if before_call is not None and before_call(name) is False:
            if should_stop is not None and should_stop():
                return hold(name)
            note(name, "not_admitted", "writer call not admitted (escalation cap reached)")
            idx = _next_distinct(rungs, idx)
            continue
        reply: WriterReply | None = None
        try:
            request = make(task, name, failure_tail)
            reply = writer(request)
        except BaseException as exc:
            # Release the call's budget slot before any backoff sleep, so a waiting task is not
            # blocked by a call that is no longer running.
            if after_call is not None:
                after_call(name, None)
            if isinstance(exc, WriterConfigError):
                note(name, "writer_error", str(exc))
                outcome.detail = f"writer {name!r} could not run: {exc}"
                idx = _next_distinct(rungs, idx)
                continue
            if not isinstance(exc, WriterTransportError):
                raise
            used = waits_by_writer.get(name, 0)
            if used >= MAX_TRANSIENT_WAITS:
                note(name, "transient_exhausted", str(exc))
                outcome.detail = f"writer {name!r} unavailable after {used} waits ({exc})"
                idx = _next_distinct(rungs, idx)
                continue
            delay = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * (2 ** used))
            delay *= 0.5 + 0.5 * max(0.0, min(1.0, jitter()))
            waits_by_writer[name] = used + 1
            outcome.transient_waits += 1
            if on_transient is not None:
                on_transient(delay)
            sleep(delay)
            continue
        if after_call is not None:
            after_call(name, reply)
        outcome.cost_cents += float(reply.cost_cents or 0.0)
        outcome.pricing_known = outcome.pricing_known and bool(reply.pricing_known)
        _add_usage(outcome.usage, reply.usage)
        if reply.cut_off:
            note(name, "cut_off", "reply hit the output limit", reply.cost_cents)
            outcome.detail = f"writer {name!r} reply was cut off at its output limit"
            idx = _next_distinct(rungs, idx)
            continue
        outcome.attempts += 1
        outcome.writer_used = name
        try:
            answer = apply_writer_answer(task, reply.text, workspace, edits_only=edits_only, allow_existing=allow_existing)
        except AnswerRejected as exc:
            detail = str(exc) + (f" [{reply.diagnostics}]" if reply.diagnostics else "")
            note(name, "answer_rejected", detail, reply.cost_cents)
            failure_tail = f"Your previous answer was rejected: {exc}"
            outcome.detail = detail
            idx += 1
            continue
        passed, tail = verify(answer)
        if passed:
            note(name, "landed", "", reply.cost_cents)
            outcome.state = "landed"
            outcome.answer = answer
            outcome.detail = ""
            return outcome
        note(name, "checks_failed", tail, reply.cost_cents)
        failure_tail = tail or "checks failed"
        outcome.detail = failure_tail
        idx += 1
    return outcome


# --------------------------------------------------------------------------------------------
# Writer backends
# --------------------------------------------------------------------------------------------


def _status_from_exception(exc: BaseException) -> int | None:
    for attr in ("code", "status_code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    response = getattr(exc, "response", None)
    val = getattr(response, "status_code", None)
    return val if isinstance(val, int) else None


def _raise_classified(exc: BaseException) -> None:
    status = _status_from_exception(exc)
    kind = classify_status(status)
    if kind == "auth":
        raise WriterAuthError(int(status or 0), type(exc).__name__) from exc
    if kind == "transient":
        raise WriterTransportError(status, type(exc).__name__) from exc
    transport_types: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)
    try:
        import httpx

        transport_types = transport_types + (httpx.TransportError,)
    except ImportError:
        pass
    if isinstance(exc, transport_types):
        raise WriterTransportError(None, type(exc).__name__) from exc
    raise exc


def render_task_prompt(
    task: FileTask,
    root: Path,
    failure_tail: str = "",
    file_cap_bytes: int = 120_000,
    extra_context: tuple[str, ...] = (),
) -> str:
    """Build the per-task user message: task, interface, current file, context files, last failure.

    ``extra_context`` adds files this task depends on (the landed output of its dependencies).
    """
    parts = [f"## Task {task.task_id}: {task.action} `{task.path}`", task.intent]
    if task.interface:
        parts.append("## Interface other files rely on (match it exactly)\n" + task.interface)
    target = root / task.path
    if task.action == "modify" and target.is_file():
        parts.append(f"## Current `{task.path}`\n" + target.read_text(encoding="utf-8", errors="replace")[:file_cap_bytes])
    seen: set[str] = set()
    for rel in tuple(task.context_files) + tuple(extra_context):
        if rel in seen:
            continue
        seen.add(rel)
        p = root / rel
        if p.is_file() and rel != task.path:
            parts.append(f"## Context `{rel}`\n" + p.read_text(encoding="utf-8", errors="replace")[:file_cap_bytes])
    if failure_tail:
        parts.append("## Your previous attempt failed\n" + failure_tail)
    parts.append(f"Reply with the JSON object for `{task.path}` only.")
    return "\n\n".join(parts)


def make_agy_cli_writer(
    auth: dict[str, Any],
    model: str,
    *,
    thinking_level: str = "LOW",
    timeout_seconds: float = 300.0,
    read_budget: int = 0,
    workspace_dir: Path | None = None,
) -> Writer:
    """Writer backed by standalone Antigravity CLI (`agy --output-format json --mode plan`) for Gemini Enterprise."""
    from server.agy_cli import is_agy_auth_error, run_agy_cli_structured_writer
    from server.regions import estimate_cost_usd

    def _write(req: WriterRequest) -> WriterReply:
        sys_inst = req.system + ("\n\n" + req.shared_prefix if req.shared_prefix else "")
        try:
            text_out, usage = run_agy_cli_structured_writer(
                prompt=req.prompt,
                system_instruction=sys_inst,
                model=model,
                thinking_level=thinking_level,
                json_schema=ANSWER_SCHEMA,
                timeout_seconds=timeout_seconds,
                read_budget=read_budget,
                workspace_dir=workspace_dir,
            )
        except TimeoutError as exc:
            raise WriterTransportError(None, f"TimeoutError: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            low = msg.lower()
            if (
                "not installed" in low
                or "cc_agy_cli_bin" in low
                or "internal `jetski` binary" in low
                or "skipped internal" in low
            ):
                raise WriterConfigError(msg) from exc
            if any(
                tok in low
                for tok in (
                    "429",
                    "503",
                    "502",
                    "504",
                    "529",
                    "resource_exhausted",
                    "quota",
                    "rate limit",
                    "overloaded",
                    "unavailable",
                )
            ):
                raise WriterTransportError(429, msg) from exc
            if is_agy_auth_error(msg):
                raise WriterAuthError(401, msg) from exc
            raise

        cost = estimate_cost_usd(model, usage, single_request=(read_budget <= 0))
        return WriterReply(
            text=text_out,
            cut_off=False,
            usage=usage,
            cost_cents=float(cost.get("total_cost_cents") or 0.0),
            model=model,
            pricing_known=bool(cost.get("pricing_known", True)),
            diagnostics="" if text_out.strip() else "empty_agy_cli_output",
        )

    return _write


def make_genai_writer(
    auth: dict[str, Any],
    model: str,
    *,
    thinking_level: str = "LOW",
    cached_content: str | None = None,
    timeout_ms: int = 300_000,
) -> Writer:
    """Writer on google-genai ``generate_content`` with a response schema (or `agy` CLI on Gemini Enterprise)."""
    if auth.get("auth_mode") == "gemini_enterprise":
        return make_agy_cli_writer(
            auth,
            model,
            thinking_level=thinking_level,
            timeout_seconds=max(5.0, float(timeout_ms) / 1000.0),
            read_budget=0,
        )
    from server.regions import estimate_cost_usd

    def _write(req: WriterRequest) -> WriterReply:
        from google.genai import types as gt

        client = _build_genai_client(auth, timeout_ms=timeout_ms)
        try:
            cfg: dict[str, Any] = {
                "response_mime_type": "application/json",
                "response_schema": ANSWER_SCHEMA,
                "max_output_tokens": req.max_output_tokens,
                "thinking_config": gt.ThinkingConfig(thinking_level=thinking_level),
            }
            if cached_content:
                cfg["cached_content"] = cached_content
            else:
                cfg["system_instruction"] = req.system + ("\n\n" + req.shared_prefix if req.shared_prefix else "")
            try:
                resp = client.models.generate_content(model=model, contents=req.prompt, config=gt.GenerateContentConfig(**cfg))
            except Exception as exc:  # noqa: BLE001 - classified below by status code / type
                _raise_classified(exc)
                raise
            usage = extract_usage_metadata(getattr(resp, "usage_metadata", None))
            finish = ""
            cands = getattr(resp, "candidates", None) or []
            if cands:
                finish = str(getattr(cands[0], "finish_reason", "") or "")
            cost = estimate_cost_usd(model, usage, single_request=True)
            return WriterReply(
                text=getattr(resp, "text", "") or "",
                cut_off=finish.endswith("MAX_TOKENS"),
                usage=usage,
                cost_cents=float(cost.get("total_cost_cents") or 0.0),
                model=model,
                pricing_known=bool(cost.get("pricing_known", True)),
                diagnostics="" if getattr(resp, "text", "") else f"finish_reason={finish}",
            )
        finally:
            close_genai_client(client)

    return _write


def create_shared_cache(
    auth: dict[str, Any], model: str, system: str, shared_text: str, *, ttl_seconds: int = 3600
) -> str | None:
    """Create a Gemini explicit context cache for text every task shares; None when not worth it or refused."""
    if not shared_text.strip() or auth.get("auth_mode") == "gemini_enterprise":
        return None
    client: Any = None
    try:
        from google.genai import types as gt

        client = _build_genai_client(auth)
        cache = client.caches.create(
            model=model,
            config=gt.CreateCachedContentConfig(
                system_instruction=system,
                contents=[gt.Content(role="user", parts=[gt.Part(text=shared_text)])],
                ttl=f"{int(ttl_seconds)}s",
                display_name="cc-agy-file-batch",
            ),
        )
        return str(getattr(cache, "name", "") or "") or None
    except Exception:  # noqa: BLE001 - too small for caching or not supported: fall back to inline prefix
        return None
    finally:
        close_genai_client(client)


def delete_shared_cache(auth: dict[str, Any], name: str | None) -> None:
    if not name:
        return
    client: Any = None
    try:
        client = _build_genai_client(auth)
        client.caches.delete(name=name)
    except Exception:  # noqa: BLE001 - the TTL expires it anyway
        pass
    finally:
        close_genai_client(client)


def make_agent_writer(
    auth: dict[str, Any],
    model: str,
    *,
    thinking_level: str = "LOW",
    timeout_seconds: float = 300.0,
    read_budget: int = 0,
    workspace_dir: Path | None = None,
) -> Writer:
    """Writer on the Antigravity SDK, stripped to a writer: no subagents, FINISH only (or up to
    ``read_budget`` read-only lookups when ``read_budget > 0``), a short custom system prompt instead
    of the default agent preamble, structured FINISH output."""
    if auth.get("auth_mode") == "gemini_enterprise":
        return make_agy_cli_writer(
            auth,
            model,
            thinking_level=thinking_level,
            timeout_seconds=timeout_seconds,
            read_budget=read_budget,
            workspace_dir=workspace_dir,
        )
    from server.regions import estimate_cost_usd

    def _write(req: WriterRequest) -> WriterReply:
        from google import antigravity as ag
        from google.antigravity import policy, types
        from server.regions import resolve_vertex_base_url, resolve_vertex_base_url_token

        options = types.GeminiModelOptions(thinking_level=getattr(types.ThinkingLevel, thinking_level))
        eff_proj = auth.get("project")
        eff_loc = auth.get("location")
        if auth.get("api_key"):
            endpoint: Any = types.GeminiAPIEndpoint(api_key=auth["api_key"], options=options)
        else:
            gw_url = resolve_vertex_base_url(auth.get("base_url")) or ""
            if gw_url:
                eff_proj = eff_proj or "custom-endpoint"
                eff_loc = eff_loc or "global"
            ep_kw: dict[str, Any] = {"project": eff_proj, "location": eff_loc, "options": options}
            if gw_url:
                gw_tok = resolve_vertex_base_url_token() or "custom-endpoint"
                ep_kw["base_url"] = gw_url
                ep_kw["api_key"] = gw_tok
                ep_kw["http_headers"] = {"Authorization": f"Bearer {gw_tok}"}
            endpoint = types.VertexEndpoint(**ep_kw)
        scratch = tempfile.mkdtemp(prefix="cc-agy-writer-")
        save_dir = tempfile.mkdtemp(prefix="cc-agy-writer-save-")
        ws_root = str(Path(workspace_dir).resolve()) if (read_budget > 0 and workspace_dir is not None) else scratch
        enabled_tools = [types.BuiltinTools.FINISH]
        custom_tools: list[Any] = []
        if read_budget > 0:
            use_sj = os.environ.get("CC_AGY_CONTAINMENT", "1").lower() not in ("0", "false", "no")
            if use_sj:
                try:
                    from server.agy_runner import build_straitjacket_tools

                    sj_tools, _ = build_straitjacket_tools(
                        ws_root, timeout_seconds=timeout_seconds, profile="read-only"
                    )
                    custom_tools.extend(sj_tools)
                except Exception:
                    pass
            if not custom_tools:
                for r_name in ("VIEW_FILE", "SEARCH_DIR", "LIST_DIR", "FIND_FILE"):
                    r_enum = getattr(types.BuiltinTools, r_name, None)
                    if r_enum is not None:
                        enabled_tools.append(r_enum)
        budget_kwargs: dict[str, Any] = {"max_model_calls": 3 if read_budget <= 0 else int(read_budget) + 2}
        if read_budget > 0:
            # BudgetConfig counts every tool call "regardless of tool source", FINISH included.
            budget_kwargs["max_tool_calls"] = int(read_budget) + 1
        sys_prompt = req.system
        if read_budget > 0:
            sys_prompt += (
                f"\nYou may make at most {int(read_budget)} Straitjacket read-only lookups "
                "(view_file, search_dir, list_dir, find_file, ctx_query, ctx_get, ctx_search) "
                "if an imported signature or convention is missing from the prompt, then call FINISH with the JSON answer."
            )
        agent_policies: list[Any] = [policy.allow_all()]
        if hasattr(policy, "workspace_only"):
            try:
                ws_policies = policy.workspace_only([ws_root])
                if isinstance(ws_policies, list):
                    agent_policies = ws_policies + agent_policies
                else:
                    agent_policies = [ws_policies] + agent_policies
            except Exception:
                pass
        cfg_kwargs: dict[str, Any] = dict(
            workspaces=[ws_root],
            vertex=not bool(auth.get("api_key")),
            model=types.ModelTarget(name=model, types=[types.ModelType.TEXT], endpoint=endpoint),
            policies=agent_policies,
            capabilities=types.CapabilitiesConfig(
                enable_subagents=False,
                enabled_tools=enabled_tools,
                finish_tool_schema_json=json.dumps(ANSWER_SCHEMA),
            ),
            system_instructions=types.CustomSystemInstructions(
                text=sys_prompt + ("\n\n" + req.shared_prefix if req.shared_prefix else "")
            ),
            budget_config=types.BudgetConfig(**budget_kwargs),
            save_dir=save_dir,
        )
        if custom_tools:
            cfg_kwargs["tools"] = custom_tools
        if auth.get("api_key"):
            cfg_kwargs["api_key"] = auth["api_key"]
        else:
            cfg_kwargs["project"] = eff_proj
            cfg_kwargs["location"] = eff_loc

        async def _run() -> WriterReply:
            from server.agy_runner import maybe_await, resolve_stop_reason, resolve_turn_usage

            async with ag.Agent(ag.LocalAgentConfig(**cfg_kwargs)) as agent:
                resp = await maybe_await(agent.chat(req.prompt))
                await asyncio.wait_for(maybe_await(resp.resolve()), timeout=timeout_seconds)
                structured = getattr(resp, "structured_output", None)
                if callable(structured):
                    structured = await maybe_await(structured())
                text_fn = getattr(resp, "text", None)
                text = str((await maybe_await(text_fn())) if callable(text_fn) else (text_fn or ""))
                usage = await resolve_turn_usage(resp)
                stop = str(await resolve_stop_reason(resp) or "")
                payload = json.dumps(structured) if isinstance(structured, (dict, list)) else (structured or text)
                cost = estimate_cost_usd(model, usage)
                return WriterReply(
                    text=str(payload), cut_off="MAX_TOKENS" in stop.upper(), usage=dict(usage),
                    cost_cents=float(cost.get("total_cost_cents") or 0.0), model=model,
                )

        try:
            return asyncio.run(_run())
        except (WriterAuthError, WriterTransportError):
            raise
        except Exception as exc:  # noqa: BLE001
            _raise_classified(exc)
            raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            shutil.rmtree(save_dir, ignore_errors=True)

    return _write


def resolve_claude_binary() -> str | None:
    explicit = os.environ.get("CC_AGY_CLAUDE_BIN", "").strip()
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("claude")
    if found:
        return found
    versions = Path.home() / ".local" / "share" / "claude" / "versions"
    if versions.is_dir():
        cands = sorted((p for p in versions.iterdir() if p.is_file() and os.access(p, os.X_OK)), key=lambda p: p.name)
        if cands:
            return str(cands[-1])
    return None


def make_claude_writer(model: str = "opus", *, timeout_seconds: float = 600.0, binary: str | None = None) -> Writer | None:
    """Last-resort writer: ``claude -p`` with no tools, no MCP servers, low effort, schema-checked JSON.

    The shared context rides in the system prompt so Claude's prompt cache reuses it across tasks.
    Returns None when no ``claude`` binary is installed.
    """
    exe = binary or resolve_claude_binary()
    if not exe:
        return None

    def _write(req: WriterRequest) -> WriterReply:
        from server.true_cost import rate_for_model

        system = req.system + ("\n\n" + req.shared_prefix if req.shared_prefix else "")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CC_AGY_", "CLAUDE_PLUGIN_"))}
        # The system prompt (with the shared context) goes through a private file: argv is capped at 128 KiB
        # per argument on Linux, and a stable system prompt is what Claude's prompt cache reuses across tasks.
        with tempfile.NamedTemporaryFile("w", suffix=".md", prefix="cc-agy-writer-", delete=False, encoding="utf-8") as fh:
            fh.write(system)
            system_file = fh.name
        cmd = [
            exe, "-p", "--output-format", "json", "--model", model, "--tools", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--effort", "low",
            "--no-session-persistence", "--setting-sources", "", "--system-prompt-file", system_file,
            "--json-schema", json.dumps(ANSWER_SCHEMA),
        ]
        try:
            proc = subprocess.run(cmd, input=req.prompt, capture_output=True, text=True, timeout=timeout_seconds,
                                  env=env, cwd=tempfile.gettempdir())
        except subprocess.TimeoutExpired:
            raise WriterTransportError(None, "claude -p timed out") from None
        except OSError as exc:
            raise WriterConfigError(f"cannot start claude: {exc}") from None
        finally:
            try:
                os.unlink(system_file)
            except OSError:
                pass
        try:
            result = json.loads(proc.stdout or "")
        except json.JSONDecodeError:
            result = None
        if not isinstance(result, dict):
            raise WriterConfigError(f"claude -p exited {proc.returncode} without a JSON result: {(proc.stderr or proc.stdout)[-300:]}")
        status = result.get("api_error_status")
        if isinstance(status, int) and classify_status(status) == "auth":
            raise WriterAuthError(status, "claude -p could not authenticate")
        if isinstance(status, int) and classify_status(status) == "transient":
            raise WriterTransportError(status, "claude -p vendor error")
        usage_raw = result.get("usage") or {}
        usage = {
            "claude_input_tokens": int(usage_raw.get("input_tokens") or 0),
            "claude_cache_read_input_tokens": int(usage_raw.get("cache_read_input_tokens") or 0),
            "claude_cache_creation_input_tokens": int(usage_raw.get("cache_creation_input_tokens") or 0),
            "claude_output_tokens": int(usage_raw.get("output_tokens") or 0),
        }
        # Price every model the call billed (claude -p can add side calls, e.g. a small title model),
        # from token counts at our list rates; the CLI's own dollar figure is not trusted.
        per_model = result.get("modelUsage") if isinstance(result.get("modelUsage"), dict) else {}
        cents, priced = 0.0, True
        billed = per_model or {model: {"inputTokens": usage["claude_input_tokens"],
                                       "cacheReadInputTokens": usage["claude_cache_read_input_tokens"],
                                       "cacheCreationInputTokens": usage["claude_cache_creation_input_tokens"],
                                       "outputTokens": usage["claude_output_tokens"]}}
        for model_id, mu in billed.items():
            rates = rate_for_model(str(model_id)) if isinstance(mu, dict) else None
            if not rates:
                priced = False
                continue
            cents += (int(mu.get("inputTokens") or 0) * rates[0] + int(mu.get("cacheReadInputTokens") or 0) * rates[1]
                      + int(mu.get("cacheCreationInputTokens") or 0) * rates[2]
                      + int(mu.get("outputTokens") or 0) * rates[4]) / 10_000.0
        structured = result.get("structured_output")
        text = json.dumps(structured) if isinstance(structured, dict) else str(result.get("result") or "")
        diagnostics = ""
        if not isinstance(structured, dict):
            diagnostics = (f"claude -p subtype={result.get('subtype')} is_error={result.get('is_error')} "
                           f"stop={result.get('stop_reason')} terminal={result.get('terminal_reason')} turns={result.get('num_turns')}")
        return WriterReply(
            text=text,
            cut_off=str(result.get("stop_reason") or "") == "max_tokens",
            usage=usage,
            cost_cents=round(cents, 4),
            model=f"claude:{model}",
            pricing_known=priced,
            diagnostics=diagnostics,
        )

    return _write
