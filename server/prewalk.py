"""Worker prewalk (frontier -> fast model trajectory handoff), advisor helpers, and ACP permission policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

from server.batch_plan import is_protected_path
from server.job_git_utils import _is_ephemeral_build_or_test_artifact, _matches_allowed_paths
from server.regions import DEFAULT_FLASH_MODEL, env_flag, first_valid_env

PREWALK_BLOCK_START = "<cc_agy_prewalk_instructions>"
PREWALK_BLOCK_END = "</cc_agy_prewalk_instructions>"

DEFAULT_PREWALK_GUIDE_MODEL = "gemini-4-carbon"
DEFAULT_PREWALK_EXECUTOR_MODEL = DEFAULT_FLASH_MODEL
DEFAULT_PREWALK_MAX_ITEMS = 7
DEFAULT_ADVISOR_MODEL = "gemini-4-carbon"
DEFAULT_ADVISOR_MAX_USES = 3

_EDIT_TOOL_NAMES = frozenset(
    {
        "edit",
        "edit_file",
        "write_file",
        "write_to_file",
        "create_file",
        "replace_file_content",
        "multi_replace_file_content",
        "delete_file",
        "apply_patch",
        "str_replace_editor",
    }
)

_SCRATCH_FILE_PREFIXES = (
    "scratch",
    "tmp_",
    "_tmp",
    ".tmp",
    ".cc_agy_",
)

_SCRATCH_DIR_NAMES = frozenset(
    {
        "tmp",
        "temp",
        "scratch",
        ".scratch",
        ".tmp",
        ".pytest_cache",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
    }
)

_SECRET_CMD_TOKEN_RE = re.compile(
    r"(?:^|[\s'\"=/])(?:\.env(?:\.[A-Za-z0-9_.-]+)?|\.ssh/|\.gnupg/|\.netrc|\.git-credentials|id_rsa|id_ed25519|[A-Za-z0-9_.-]+\.(?:pem|key|p12|pfx))(?=$|[\s'\":])"
)


@dataclass(frozen=True)
class PrewalkConfig:
    """Resolved configuration for intra-worker frontier-to-fast prewalk execution."""

    enabled: bool
    guide_model: str
    executor_model: str
    max_checklist_items: int


@dataclass(frozen=True)
class AdvisorConfig:
    """Resolved configuration for mid-task frontier advisor consultations."""

    enabled: bool
    advisor_model: str
    max_uses: int


def resolve_worker_strategy(env: Mapping[str, str] | None = None) -> str:
    """Return normalized worker strategy (`single`, `prewalk`, `advisor`, or `prewalk_advisor`)."""
    source = os.environ if env is None else env
    raw_strategy = first_valid_env(
        source,
        "CC_AGY_WORKER_STRATEGY",
        "CLAUDE_PLUGIN_OPTION_WORKER_STRATEGY",
        "CC_AGY_USERCFG_WORKER_STRATEGY",
        default="",
    ).strip().lower().replace("-", "_")
    prewalk_flag = env_flag("CC_AGY_PREWALK", source)
    advisor_flag = env_flag("CC_AGY_ADVISOR", source)

    if raw_strategy in ("prewalk_advisor", "both"):
        return "prewalk_advisor"
    if raw_strategy == "prewalk":
        return "prewalk_advisor" if advisor_flag is True else "prewalk"
    if raw_strategy == "advisor":
        return "prewalk_advisor" if prewalk_flag is True else "advisor"
    if raw_strategy in ("single", "off", "none"):
        return "single"

    if prewalk_flag is True and advisor_flag is True:
        return "prewalk_advisor"
    if prewalk_flag is True:
        return "prewalk"
    if advisor_flag is True:
        return "advisor"
    return "single"


def _is_frontier_guide_family(model_name: str) -> bool:
    low = (model_name or "").strip().lower()
    return any(tok in low for tok in ("carbon", "opus", "fable", "pro"))


def resolve_prewalk_config(
    env: Mapping[str, str] | None = None,
    *,
    requested_model: str | None = None,
    read_only: bool = False,
    explicit_prewalk: bool | None = None,
) -> PrewalkConfig:
    """Resolve whether worker prewalk is active and which guide/executor models to pair."""
    source = os.environ if env is None else env
    strategy = resolve_worker_strategy(source)
    enabled = explicit_prewalk if explicit_prewalk is not None else strategy in ("prewalk", "prewalk_advisor")

    guide_model = (
        first_valid_env(
            source,
            "CC_AGY_PREWALK_GUIDE_MODEL",
            "CLAUDE_PLUGIN_OPTION_PREWALK_GUIDE_MODEL",
            default=DEFAULT_PREWALK_GUIDE_MODEL,
        ).strip()
        or DEFAULT_PREWALK_GUIDE_MODEL
    )
    req_clean = (requested_model or "").strip()
    configured_exec = first_valid_env(
        source,
        "CC_AGY_PREWALK_EXECUTOR_MODEL",
        "CLAUDE_PLUGIN_OPTION_PREWALK_EXECUTOR_MODEL",
        default="",
    ).strip()
    if configured_exec:
        executor_model = configured_exec
    elif req_clean and not _is_frontier_guide_family(req_clean):
        executor_model = req_clean
    else:
        executor_model = DEFAULT_PREWALK_EXECUTOR_MODEL

    raw_items = first_valid_env(source, "CC_AGY_PREWALK_MAX_ITEMS", default=str(DEFAULT_PREWALK_MAX_ITEMS))
    try:
        max_items = max(3, min(10, int(raw_items)))
    except ValueError:
        max_items = DEFAULT_PREWALK_MAX_ITEMS

    if read_only or guide_model.strip().lower() == executor_model.strip().lower():
        enabled = False
    if req_clean and _is_frontier_guide_family(req_clean) and explicit_prewalk is not True and not configured_exec:
        # Caller explicitly asked for a frontier-model run without configuring a distinct prewalk executor.
        enabled = False

    return PrewalkConfig(
        enabled=bool(enabled),
        guide_model=guide_model,
        executor_model=executor_model,
        max_checklist_items=max_items,
    )


def resolve_advisor_config(
    env: Mapping[str, str] | None = None,
    *,
    executor_model: str | None = None,
    read_only: bool = False,
    explicit_advisor: bool | None = None,
) -> AdvisorConfig:
    """Resolve mid-task worker advisor settings (`Hydrogen` executor consulting `Carbon` advisor)."""
    source = os.environ if env is None else env
    strategy = resolve_worker_strategy(source)
    enabled = explicit_advisor if explicit_advisor is not None else strategy in ("advisor", "prewalk_advisor")
    advisor_model = (
        first_valid_env(
            source,
            "CC_AGY_ADVISOR_MODEL",
            "CLAUDE_PLUGIN_OPTION_ADVISOR_MODEL",
            default=DEFAULT_ADVISOR_MODEL,
        ).strip()
        or DEFAULT_ADVISOR_MODEL
    )
    raw_uses = first_valid_env(source, "CC_AGY_ADVISOR_MAX_USES", default=str(DEFAULT_ADVISOR_MAX_USES))
    try:
        max_uses = max(1, min(10, int(raw_uses)))
    except ValueError:
        max_uses = DEFAULT_ADVISOR_MAX_USES

    exec_clean = (executor_model or "").strip().lower()
    if read_only or (exec_clean and exec_clean == advisor_model.strip().lower()):
        enabled = False
    return AdvisorConfig(
        enabled=bool(enabled),
        advisor_model=advisor_model,
        max_uses=max_uses,
    )


def inject_prewalk_planning_block(
    prompt: str,
    *,
    max_items: int = DEFAULT_PREWALK_MAX_ITEMS,
    acceptance: Sequence[str] | None = None,
) -> str:
    """Append the removable Phase-1 prewalk planning & checklist instruction block."""
    clean_base = strip_prewalk_planning_block(prompt).rstrip()
    acc_hint = ""
    if acceptance:
        joined = "; ".join(str(cmd).strip() for cmd in acceptance if str(cmd).strip())
        if joined:
            acc_hint = f" Include `{joined}` as the final verification item."
    planning_block = (
        f"{PREWALK_BLOCK_START}\n"
        "Before editing broadly:\n"
        f"1. Inspect the relevant code paths and write a concise numbered TODO checklist of at most {max_items} items.\n"
        f"2. Each checklist item must name the target file(s) and a concrete validation step.{acc_hint}\n"
        "3. Immediately apply the first concrete code edit from item 1 in a real repository file.\n"
        f"{PREWALK_BLOCK_END}"
    )
    return f"{clean_base}\n\n{planning_block}"


def strip_prewalk_planning_block(prompt: str) -> str:
    """Remove any `<cc_agy_prewalk_instructions>...</cc_agy_prewalk_instructions>` block from `prompt`."""
    if PREWALK_BLOCK_START not in prompt:
        return prompt
    pattern = re.compile(
        re.escape(PREWALK_BLOCK_START) + r".*?" + re.escape(PREWALK_BLOCK_END),
        flags=re.DOTALL,
    )
    stripped = pattern.sub("", prompt)
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


def build_prewalk_continuation_prompt(
    *,
    first_edited_path: str,
    acceptance: Sequence[str] | None = None,
) -> str:
    """Build the clean Phase-2 executor prompt with no planning block."""
    acc_clause = ""
    if acceptance:
        cmds = [str(cmd).strip() for cmd in acceptance if str(cmd).strip()]
        if cmds:
            acc_clause = f" Run acceptance verification ({'; '.join(cmds)}) before finishing."
    return (
        f"Continue from the checklist and first edit already applied to `{first_edited_path}`. "
        f"Complete the remaining checklist items in place without re-reading unchanged files.{acc_clause}"
    )


def build_advisor_consultation_prompt(*, failure_summary: str) -> str:
    """Build a bounded read-only advisor prompt modeled on Anthropic's `advisor_20260301` guidance."""
    trimmed = (failure_summary or "").strip()[:2000]
    return (
        "ADVISOR CONSULTATION (read-only diagnosis, <= 500 tokens, do not edit files): "
        "Review the trajectory and failure output below, identify the exact root cause, and list "
        "the minimal file-and-line corrections the executor should make next.\n\n"
        f"Failure summary:\n{trimmed}"
    )


def _normalize_workspace_rel_path(raw_path: str, workspace_dir: str | Path) -> tuple[str | None, str | None]:
    """Resolve `raw_path` relative to `workspace_dir`; return `(rel_posix, escape_error)`."""
    cleaned = str(raw_path or "").strip()
    if not cleaned:
        return None, None
    if cleaned.startswith("file://"):
        cleaned = cleaned[len("file://") :]
    ws_root = Path(workspace_dir).resolve()
    cand = Path(cleaned)
    try:
        resolved = cand.resolve(strict=False) if cand.is_absolute() else (ws_root / cand).resolve(strict=False)
    except OSError:
        return None, f"Invalid path {raw_path!r}."
    if not resolved.is_relative_to(ws_root):
        return None, f"Path {raw_path!r} escapes isolated workspace {ws_root}."
    rel_posix = resolved.relative_to(ws_root).as_posix()
    if not rel_posix or rel_posix == ".":
        return None, None
    return rel_posix, None


def is_scratch_or_test_artifact_path(rel_posix: str) -> bool:
    """Return True for scratch files, temp notes, caches, or build/test artifacts that must not trigger prewalk swap."""
    norm = PurePosixPath(rel_posix).as_posix().removeprefix("./")
    if not norm:
        return True
    if _is_ephemeral_build_or_test_artifact(norm) or is_protected_path(norm):
        return True
    parts = [p.lower() for p in PurePosixPath(norm).parts]
    if any(part in _SCRATCH_DIR_NAMES for part in parts[:-1]):
        return True
    basename = parts[-1]
    if basename.startswith(_SCRATCH_FILE_PREFIXES):
        return True
    if basename.endswith((".tmp", ".bak", ".orig", ".rej", ".log", ".out", ".coverage")):
        return True
    if basename in ("todo.md", "plan.md", "notes.md", "scratch.md", "scratch.py", "repro_tmp.py"):
        return True
    return False


def is_real_code_edit_target(
    raw_path: str,
    *,
    workspace_dir: str | Path,
    allowed_paths: Sequence[str] | None = None,
) -> str | None:
    """Return repo-relative POSIX path if `raw_path` is a real repository file edit, else `None`."""
    rel_posix, err = _normalize_workspace_rel_path(raw_path, workspace_dir)
    if err is not None or not rel_posix:
        return None
    if is_scratch_or_test_artifact_path(rel_posix):
        return None
    if allowed_paths and not _matches_allowed_paths(rel_posix, list(allowed_paths)):
        return None
    return rel_posix


def extract_tool_call_paths(tool_payload: Mapping[str, Any]) -> list[str]:
    """Extract all candidate file paths referenced by an ACP `toolCall` or `session/update` payload."""
    paths: list[str] = []
    for map_key in ("rawInput", "args"):
        arg_map = tool_payload.get(map_key)
        if isinstance(arg_map, Mapping):
            for key in ("targetFile", "Path", "path", "file_path", "filepath"):
                val = arg_map.get(key)
                if isinstance(val, str) and val.strip() and val.strip() not in paths:
                    paths.append(val.strip())
    locations = tool_payload.get("locations")
    if isinstance(locations, Sequence) and not isinstance(locations, (str, bytes)):
        for loc in locations:
            if isinstance(loc, Mapping) and isinstance(loc.get("path"), str) and loc["path"].strip():
                p_str = loc["path"].strip()
                if p_str not in paths:
                    paths.append(p_str)
            elif isinstance(loc, str) and loc.strip() and loc.strip() not in paths:
                paths.append(loc.strip())
    content_items = tool_payload.get("content")
    if isinstance(content_items, Sequence) and not isinstance(content_items, (str, bytes)):
        for item in content_items:
            if isinstance(item, Mapping) and isinstance(item.get("path"), str) and item["path"].strip():
                p_str = item["path"].strip()
                if p_str not in paths:
                    paths.append(p_str)
    return paths


def is_acp_edit_tool(tool_payload: Mapping[str, Any]) -> bool:
    """Return True if `tool_payload` represents a file-creating or file-editing tool call."""
    kind = str(tool_payload.get("kind") or "").strip().lower()
    if kind in ("edit", "write", "delete"):
        return True
    raw_name = str(tool_payload.get("name") or tool_payload.get("title") or "").strip().lower()
    if raw_name.startswith("default_api:"):
        raw_name = raw_name[len("default_api:") :]
    if raw_name in _EDIT_TOOL_NAMES:
        return True
    return any(raw_name.startswith(f"{ename} ") or raw_name.endswith(f":{ename}") for ename in _EDIT_TOOL_NAMES)


def extract_edit_path_from_acp_event(
    update_payload: Mapping[str, Any],
    *,
    workspace_dir: str | Path,
    allowed_paths: Sequence[str] | None = None,
) -> str | None:
    """Return normalized repo-relative path when an ACP update marks a completed real code edit."""
    status = str(update_payload.get("status") or "").strip().lower()
    if status != "completed":
        return None
    if not is_acp_edit_tool(update_payload):
        return None
    for raw_path in extract_tool_call_paths(update_payload):
        valid_rel = is_real_code_edit_target(
            raw_path,
            workspace_dir=workspace_dir,
            allowed_paths=allowed_paths,
        )
        if valid_rel is not None:
            return valid_rel
    return None


def evaluate_acp_permission_policy(
    tool_call: Mapping[str, Any],
    *,
    workspace_dir: str | Path,
    allowed_paths: Sequence[str] | None = None,
    read_only: bool = False,
    skip_permissions: bool = False,
) -> tuple[bool, dict[str, Any] | None]:
    """Evaluate `session/request_permission` against `read_only`, `allowed_paths`, and secret-file policy.

    Returns `(allowed, denied_entry_or_none)`. Policy checks apply even when `skip_permissions=True`.
    """
    raw_name = str(tool_call.get("name") or tool_call.get("title") or tool_call.get("kind") or "tool").strip()
    clean_name = raw_name[len("default_api:") :] if raw_name.startswith("default_api:") else raw_name
    kind_low = str(tool_call.get("kind") or "").strip().lower()
    is_exec = (
        kind_low in ("execute", "command", "unsandboxed")
        or any(tok in clean_name.lower() for tok in ("run_command", "bash", "shell", "exec"))
    )
    is_write = is_acp_edit_tool(tool_call)
    candidate_paths = extract_tool_call_paths(tool_call)

    # 1. Check all extracted file paths against workspace boundary, secret denylist, and allowed_paths.
    for raw_path in candidate_paths:
        rel_posix, esc_err = _normalize_workspace_rel_path(raw_path, workspace_dir)
        if esc_err is not None:
            return False, {
                "kind": "write" if is_write else "read",
                "tool": clean_name,
                "path": raw_path,
                "reason": esc_err,
                "hard_deny": True,
            }
        if rel_posix and is_protected_path(rel_posix):
            return False, {
                "kind": "write" if is_write else "read",
                "tool": clean_name,
                "path": rel_posix,
                "reason": f"Access to protected or secret file '{rel_posix}' is denied by policy.",
                "hard_deny": True,
            }
        if is_write and rel_posix and allowed_paths and not _matches_allowed_paths(rel_posix, list(allowed_paths)):
            return False, {
                "kind": "write",
                "tool": clean_name,
                "path": rel_posix,
                "reason": f"Edit target '{rel_posix}' is outside allowed_paths {list(allowed_paths)}.",
                "hard_deny": False,
            }

    # 2. Check shell command strings for secret file access.
    if is_exec:
        cmd_text = ""
        for map_key in ("rawInput", "args"):
            arg_map = tool_call.get(map_key)
            if isinstance(arg_map, Mapping):
                for k in ("CommandLine", "command", "cmd"):
                    if isinstance(arg_map.get(k), str) and arg_map[k].strip():
                        cmd_text = arg_map[k].strip()
                        break
        if cmd_text and _SECRET_CMD_TOKEN_RE.search(cmd_text):
            return False, {
                "kind": "command",
                "tool": clean_name,
                "command": cmd_text[:200],
                "reason": "Command references a protected secret or credential path.",
                "hard_deny": True,
            }

    if read_only and (is_write or is_exec):
        return False, {
            "kind": "command" if is_exec else "write",
            "tool": clean_name,
            "reason": "Blocked in read-only ACP worker session.",
            "hard_deny": True,
        }

    if not skip_permissions and (is_write or is_exec):
        return False, {
            "kind": "command" if is_exec else "write",
            "tool": clean_name,
            "hard_deny": False,
            **(
                {"path": candidate_paths[0]}
                if candidate_paths
                else {}
            ),
            "reason": "Action requires interactive permission approval outside an OS sandbox.",
        }

    return True, None


def record_model_phase_usage(
    by_model: dict[str, dict[str, int]],
    requests_list: list[dict[str, Any]],
    *,
    model: str,
    phase: str,
    phase_usage: Mapping[str, Any],
) -> None:
    """Accumulate per-model token usage (`by_model` and `requests` entries) across worker phases."""
    token_keys = (
        "prompt_token_count",
        "cached_content_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "total_token_count",
    )
    clean_model = (model or "unknown").strip() or "unknown"
    p_tok = max(0, int(phase_usage.get("prompt_token_count") or 0))
    c_tok = max(0, int(phase_usage.get("cached_content_token_count") or 0))
    cand_tok = max(0, int(phase_usage.get("candidates_token_count") or 0))
    th_tok = max(0, int(phase_usage.get("thoughts_token_count") or 0))
    tot_tok = max(int(phase_usage.get("total_token_count") or 0), p_tok + cand_tok + th_tok)
    if p_tok == 0 and cand_tok == 0 and th_tok == 0 and tot_tok == 0:
        return
    bucket = by_model.setdefault(clean_model, {k: 0 for k in token_keys})
    bucket["prompt_token_count"] += p_tok
    bucket["cached_content_token_count"] += c_tok
    bucket["candidates_token_count"] += cand_tok
    bucket["thoughts_token_count"] += th_tok
    bucket["total_token_count"] += tot_tok
    requests_list.append(
        {
            "model": clean_model,
            "phase": phase,
            "prompt_token_count": p_tok,
            "cached_content_token_count": c_tok,
            "candidates_token_count": cand_tok,
            "thoughts_token_count": th_tok,
            "total_token_count": tot_tok,
        }
    )
