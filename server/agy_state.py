"""Small on-disk caches for the `agy` CLI adapter: the model catalog and resumable conversations.

Both caches hold only non-secret metadata (model names, conversation IDs, hashes, token
counts). Prompts, responses, and credentials are never written. Files are created with
mode `0600` under a per-user state directory, replaced atomically, and guarded by a file
lock so two MCP servers on the same machine do not corrupt each other's entries.

Model catalog: `agy models` lists the models the signed-in seat may use. Caching that list
lets a model-name mismatch be fixed before any tokens are spent, instead of after a failed
first turn.

Conversations: a follow-up question on the same context can resume the earlier `agy`
conversation and send only the new question. The earlier turns then form a stable prefix,
which keeps Gemini's implicit cache warm. An entry is reused only for the same host
session, model, effort, workspace, and context prefix, only while it is not leased by a
concurrent in-flight request, and only within a short TTL.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any
import uuid

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl; the thread lock still applies.
    fcntl = None  # type: ignore[assignment]

DEFAULT_CATALOG_TTL_SECONDS = 60.0
DEFAULT_CONVERSATION_TTL_SECONDS = 900.0
DEFAULT_LEASE_SECONDS = 300.0
CATALOG_COMMAND_TIMEOUT_SECONDS = 20.0
MAX_CONVERSATION_ENTRIES = 256
MAX_CATALOG_ENTRIES = 16

_CATALOG_FILE = "agy_model_catalog.json"
_CONVERSATIONS_FILE = "agy_conversations.json"
_MODEL_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,79}$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_EFFORT_SUFFIXES = ("-low", "-medium", "-high", "-xhigh", "-max")
_USAGE_KEYS = (
    "prompt_token_count",
    "cached_content_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)

# One MCP stdio server serves one host session, so its process is a safe default session scope.
PROCESS_SESSION_ID = f"proc-{os.getpid()}-{uuid.uuid4().hex[:12]}"

_THREAD_LOCK = threading.RLock()
_CATALOG_MEMORY: dict[str, tuple[float, tuple[str, ...] | None]] = {}


def _setting(env: Mapping[str, str] | None, name: str) -> str:
    """Read `name` from `env`, falling back to the process environment."""
    for source in (env, os.environ):
        if source is None:
            continue
        cleaned = str(source.get(name) or "").strip()
        if cleaned and not cleaned.startswith("${"):
            return cleaned
    return ""


def _seconds_setting(env: Mapping[str, str] | None, name: str, default: float) -> float:
    try:
        return max(0.0, float(_setting(env, name) or default))
    except ValueError:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def state_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the per-user directory for `agy` adapter caches (`CC_AGY_STATE_DIR` overrides it)."""
    explicit = _setting(env, "CC_AGY_STATE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    from server.agy_cli import managed_agy_bin_dir

    return managed_agy_bin_dir(env).parent / "state"


def conversation_workspace_dir(key: str | None, env: Mapping[str, str] | None = None) -> Path:
    """Return a stable directory for one conversation key so `agy --conversation` sees a consistent cwd."""
    slug = (key or "default")[:24]
    ws_dir = state_dir(env) / "ask_workspaces" / slug
    ws_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        os.chmod(ws_dir, 0o700)
        os.utime(ws_dir, None)
    return ws_dir


def _prune_stale_ask_workspaces(
    active_keys: Sequence[str] | set[str], env: Mapping[str, str] | None = None
) -> None:
    """Remove `ask_workspaces/<slug>` directories whose conversation keys have expired and are not in flight."""
    ws_root = state_dir(env) / "ask_workspaces"
    if not ws_root.is_dir():
        return
    keep_slugs = {k[:24] for k in active_keys if isinstance(k, str) and k}
    min_age = max(DEFAULT_LEASE_SECONDS, conversation_ttl_seconds(env))
    now = time.time()
    try:
        for child in ws_root.iterdir():
            if child.name in keep_slugs:
                continue
            try:
                if now - child.stat().st_mtime < min_age:
                    continue
            except OSError:
                continue
            shutil.rmtree(child, ignore_errors=True)
    except OSError:
        pass


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold a thread lock and, where available, an exclusive file lock next to `path`."""
    with _THREAD_LOCK:
        lock_fd: int | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if fcntl is not None:
                lock_fd = os.open(str(path) + ".lock", os.O_WRONLY | os.O_CREAT, 0o600)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except OSError:
            lock_fd = None
        try:
            yield
        finally:
            if lock_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(lock_fd)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically replace `path` with `data` as a `0600` file; failures are ignored."""
    try:
        if path.is_symlink():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True)
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except OSError:
        pass


def _digest(parts: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(parts)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------


def parse_agy_models_output(stdout_text: str) -> tuple[str, ...]:
    """Parse `agy models` output (`<slug><TAB or spaces><display name>` per line) into slugs."""
    slugs: list[str] = []
    for raw_line in stdout_text.splitlines():
        clean = _ANSI_RE.sub("", raw_line).strip()
        if not clean:
            continue
        slug = re.split(r"\t|\s{2,}", clean, maxsplit=1)[0].strip()
        if _MODEL_SLUG_RE.match(slug) and slug not in slugs:
            slugs.append(slug)
    return tuple(slugs)


def catalog_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return False when `CC_AGY_MODEL_CATALOG` turns the up-front catalog check off."""
    return _setting(env, "CC_AGY_MODEL_CATALOG").lower() not in ("0", "false", "no", "off")


def _catalog_key(agy_bin: str, env: Mapping[str, str] | None) -> str:
    from server.agy_cli import agy_cli_config_dir, resolve_enterprise_adc_seat

    seat = resolve_enterprise_adc_seat(env)
    return _digest([agy_bin, str(agy_cli_config_dir(env)), str(seat["project"] if seat["active"] else "")])


def _run_agy_models(agy_bin: str, env: Mapping[str, str] | None) -> tuple[str, ...] | None:
    from server.agy_cli import _custom_config_dir_args, build_agy_subprocess_env

    try:
        with tempfile.TemporaryDirectory(prefix="cc-agy-models-") as empty_cwd:
            proc = subprocess.run(
                [agy_bin, *_custom_config_dir_args(env), "models"],
                cwd=empty_cwd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=CATALOG_COMMAND_TIMEOUT_SECONDS,
                env=build_agy_subprocess_env(env),
                check=False,
            )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    slugs = parse_agy_models_output(proc.stdout.decode("utf-8", errors="replace"))
    return slugs or None


def load_agy_model_catalog(
    agy_bin: str,
    env: Mapping[str, str] | None = None,
    *,
    refresh: bool = False,
) -> tuple[str, ...] | None:
    """Return the models `agy` offers this seat, or None when the list cannot be read.

    A fresh answer is taken from memory, then from disk, and only then from `agy models`.
    A failed lookup is remembered in memory for the same TTL so a broken or offline CLI
    does not slow every call. None always means "unknown", never "no models".
    """
    if not catalog_enabled(env):
        return None
    ttl = _seconds_setting(env, "CC_AGY_MODEL_CATALOG_TTL_SECONDS", DEFAULT_CATALOG_TTL_SECONDS)
    key = _catalog_key(agy_bin, env)
    path = state_dir(env) / _CATALOG_FILE
    with _THREAD_LOCK:
        cached = _CATALOG_MEMORY.get(key)
    if not refresh and cached is not None and time.monotonic() - cached[0] < ttl:
        return cached[1]
    if not refresh:
        entry = _read_json(path).get(key)
        fetched_at = _safe_float(entry.get("fetched_at") if isinstance(entry, dict) else 0.0)
        if isinstance(entry, dict) and fetched_at > 0.0 and time.time() - fetched_at < ttl:
            models = tuple(m for m in entry.get("models") or [] if isinstance(m, str))
            if models:
                with _THREAD_LOCK:
                    _CATALOG_MEMORY[key] = (time.monotonic(), models)
                return models

    models_or_none = _run_agy_models(agy_bin, env)
    with _THREAD_LOCK:
        _CATALOG_MEMORY[key] = (time.monotonic(), models_or_none)
    if models_or_none:
        with _locked(path):
            raw_data = _read_json(path)
            valid_data = {k: v for k, v in raw_data.items() if isinstance(v, dict)}
            valid_data[key] = {"fetched_at": time.time(), "models": list(models_or_none)}
            newest = sorted(
                valid_data.items(),
                key=lambda kv: _safe_float(kv[1].get("fetched_at")),
                reverse=True,
            )
            _write_json(path, dict(newest[:MAX_CATALOG_ENTRIES]))
    return models_or_none


def catalog_offers(model: str, catalog: Sequence[str]) -> bool:
    """Return True if `model` is listed, or is the base name of a listed `<model>-<effort>` entry."""
    if model in catalog:
        return True
    return any(f"{model}{suffix}" in catalog for suffix in _EFFORT_SUFFIXES)


def inspect_agy_model_catalog(
    agy_bin: str | None,
    env: Mapping[str, str] | None = None,
    *,
    live: bool = False,
) -> dict[str, Any]:
    """Summarize the model catalog for `doctor`. Only `live=True` runs `agy models`."""
    report: dict[str, Any] = {
        "enabled": catalog_enabled(env),
        "model_count": None,
        "models": [],
        "age_seconds": None,
        "source": None,
    }
    if not agy_bin or not report["enabled"]:
        return report
    if live:
        models = load_agy_model_catalog(agy_bin, env, refresh=True)
        if models:
            report.update({"model_count": len(models), "models": list(models), "age_seconds": 0, "source": "live"})
        else:
            report["source"] = "unavailable"
        return report
    entry = _read_json(state_dir(env) / _CATALOG_FILE).get(_catalog_key(agy_bin, env))
    if isinstance(entry, dict) and entry.get("models"):
        models = [m for m in entry["models"] if isinstance(m, str)]
        age = max(0, int(time.time() - _safe_float(entry.get("fetched_at"))))
        report.update({"model_count": len(models), "models": models, "age_seconds": age, "source": "disk_cache"})
    return report


# ---------------------------------------------------------------------------
# Resumable conversations
# ---------------------------------------------------------------------------


def conversation_ttl_seconds(env: Mapping[str, str] | None = None) -> float:
    """Return how long a stored conversation may be resumed (`0` turns delta resume off)."""
    return _seconds_setting(env, "CC_AGY_CONVERSATION_TTL_SECONDS", DEFAULT_CONVERSATION_TTL_SECONDS)


def conversation_key(
    *,
    session_id: str,
    model: str,
    effort: str | None,
    workspace: str,
    prefix: str,
    config_dir: str,
) -> str:
    """Build the lookup key for one resumable conversation.

    Every field that changes what the conversation contains, or who may see it, is part of
    the key, so a different session, model, effort, workspace, or context never reuses it.
    Only a hash of `prefix` is kept.
    """
    prefix_hash = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    return _digest([session_id, model, effort or "", workspace, prefix_hash, config_dir])


def _fresh_entries(data: Mapping[str, Any], ttl: float) -> dict[str, dict[str, Any]]:
    now = time.time()
    fresh: dict[str, dict[str, Any]] = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        conv_id = entry.get("conversation_id")
        updated_at = _safe_float(entry.get("updated_at"))
        if isinstance(conv_id, str) and conv_id and updated_at > 0.0 and now - updated_at < ttl:
            fresh[key] = dict(entry)
    return fresh


def lookup_conversation(
    key: str,
    env: Mapping[str, str] | None = None,
    *,
    claim_lease: bool = True,
) -> dict[str, Any] | None:
    """Return `{conversation_id, turns, usage}` for `key` if fresh and not leased by another call.

    Args:
        key: Conversation lookup hash from `conversation_key`.
        env: Optional environment mapping.
        claim_lease: When True (default), marks the entry leased so a concurrent `ask_gemini`
            call with the same key starts a separate conversation instead of colliding.

    Returns:
        A dict with `conversation_id`, `turns`, and `usage`, or None if missing, expired, or leased.
    """
    ttl = conversation_ttl_seconds(env)
    if ttl <= 0 or not key:
        return None
    path = state_dir(env) / _CONVERSATIONS_FILE
    with _locked(path):
        now = time.time()
        data = _fresh_entries(_read_json(path), ttl)
        entry = data.get(key)
        if entry is None:
            return None
        leased_until = _safe_float(entry.get("leased_until"))
        if leased_until > now:
            return None
        if claim_lease:
            entry["leased_until"] = now + DEFAULT_LEASE_SECONDS
            data[key] = entry
            _write_json(path, data)
        raw_usage = entry.get("usage") if isinstance(entry.get("usage"), dict) else {}
        return {
            "conversation_id": entry["conversation_id"],
            "turns": max(1, _safe_int(entry.get("turns"), 1)),
            "usage": {k: max(0, _safe_int(raw_usage.get(k), 0)) for k in _USAGE_KEYS},
        }


def remember_conversation(
    key: str,
    conversation_id: str,
    *,
    turns: int,
    usage: Mapping[str, int],
    env: Mapping[str, str] | None = None,
) -> None:
    """Store or refresh the conversation for `key` (releasing any active lease)."""
    ttl = conversation_ttl_seconds(env)
    if ttl <= 0 or not conversation_id or not key:
        return
    path = state_dir(env) / _CONVERSATIONS_FILE
    with _locked(path):
        data = _fresh_entries(_read_json(path), ttl)
        data[key] = {
            "conversation_id": conversation_id,
            "turns": max(1, _safe_int(turns, 1)),
            "usage": {k: max(0, _safe_int(usage.get(k), 0)) for k in _USAGE_KEYS},
            "updated_at": time.time(),
            "leased_until": 0.0,
        }
        newest = sorted(
            data.items(),
            key=lambda kv: _safe_float(kv[1].get("updated_at")),
            reverse=True,
        )
        kept = dict(newest[:MAX_CONVERSATION_ENTRIES])
        _write_json(path, kept)
        _prune_stale_ask_workspaces(set(kept.keys()), env)


def release_conversation_lease(key: str, env: Mapping[str, str] | None = None) -> None:
    """Release an active lease on `key` without deleting the conversation entry."""
    if not key:
        return
    path = state_dir(env) / _CONVERSATIONS_FILE
    with _locked(path):
        data = _read_json(path)
        entry = data.get(key)
        if isinstance(entry, dict) and _safe_float(entry.get("leased_until")) > 0.0:
            entry["leased_until"] = 0.0
            _write_json(path, data)


def forget_conversation(key: str, env: Mapping[str, str] | None = None) -> None:
    """Drop the conversation for `key` (used when a resume fails or times out)."""
    if not key:
        return
    path = state_dir(env) / _CONVERSATIONS_FILE
    with _locked(path):
        data = _read_json(path)
        if data.pop(key, None) is not None:
            _write_json(path, data)


def clear_memory_caches() -> None:
    """Drop in-memory catalog results (used by tests)."""
    with _THREAD_LOCK:
        _CATALOG_MEMORY.clear()
