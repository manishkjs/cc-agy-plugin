"""Session receipt state, Lead read-budget tracking, and strict-mode PreToolUse evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from server.env_flags import is_strict_mode_enabled, resolve_cost_profile

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _resolve_data_dir_fast(
    data_dir: str | Path | None,
    workspace_root: Path,
    env: dict[str, str] | None = None,
) -> Path:
    """Resolve the plugin receipt data directory without importing `server.receipts`."""
    if data_dir is not None:
        candidate = str(data_dir).strip()
        if candidate and not candidate.startswith("${"):
            return Path(candidate).resolve()
    source = env if env is not None else os.environ
    for env_key in ("CC_AGY_DATA_DIR", "CLAUDE_PLUGIN_DATA"):
        val = str(source.get(env_key) or "").strip()
        if val and not val.startswith("${"):
            return Path(val).resolve()
    legacy_in_tree = (workspace_root / ".cc-agy-data").resolve()
    if legacy_in_tree.exists():
        return legacy_in_tree
    ws_slug = hashlib.sha256(str(workspace_root).encode("utf-8")).hexdigest()[:12]
    return (Path.home() / ".cache" / "cc-agy-plugin" / ws_slug).resolve()

def receipt_owner_pid_alive(receipt: dict[str, Any]) -> bool:
    """Lazy-imported wrapper around `server.receipts.receipt_owner_pid_alive`."""
    from server.receipts import receipt_owner_pid_alive as _impl

    return _impl(receipt)


def is_verified_applied_receipt(receipt: dict[str, Any] | None) -> bool:
    """Lazy-imported wrapper around `server.verification_policy.is_verified_applied_receipt`."""
    if not isinstance(receipt, dict):
        return False
    from server.verification_policy import is_verified_applied_receipt as _impl

    return _impl(receipt)


def resolve_project_root(
    workdir: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    """Resolve the true project root, honoring CLAUDE_PROJECT_DIR and .git directory discovery."""
    source = env if env is not None else os.environ
    env_proj = (source.get("CLAUDE_PROJECT_DIR") or "").strip()
    if env_proj and not env_proj.startswith("${"):
        base = Path(env_proj).resolve()
    elif workdir is not None:
        base = Path(workdir).resolve()
    else:
        base = Path.cwd().resolve()

    if base.exists():
        if not (source.get("GIT_DIR") or source.get("GIT_WORK_TREE")):
            start_dir = base if base.is_dir() else base.parent
            for cur in (start_dir, *start_dir.parents):
                if (cur / ".git").exists():
                    return cur.resolve()
        else:
            from server.verify import run_git

            res = run_git(
                ["rev-parse", "--show-toplevel"],
                cwd=str(base if base.is_dir() else base.parent),
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return Path(res.stdout.strip()).resolve()

    # If no explicit CLAUDE_PROJECT_DIR and no .git repo, and base is a subdirectory
    # inside a top-level /tmp/<workspace-root> folder (e.g. /tmp/ws-xxxx/src), anchor
    # to /tmp/<workspace-root> so `cd src` does not lose the enclosing workspace root.
    if not (env_proj and not env_proj.startswith("${")):
        for tmp_root in (Path("/tmp").resolve(), Path("/var/tmp").resolve()):
            if base.is_relative_to(tmp_root):
                rel_parts = base.relative_to(tmp_root).parts
                if len(rel_parts) >= 2:
                    return tmp_root / rel_parts[0]
    return base


def classify_path(
    target_path: str | Path,
    workdir: str | Path,
    out_dir: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Classify a filesystem path relative to out_dir, workdir, and /tmp.

    CRITICAL ORDERING INVARIANT:
    Always check `out_dir` and `workdir` BEFORE `/tmp` (or `/var/tmp`) so that
    git repositories or worktrees located under `/tmp/...` on Linux/macOS are
    classified as `"workspace"` (and protected against direct driver mutation),
    never misclassified as `"tmp_scratch"`.
    """
    ws_resolved = resolve_project_root(workdir, env=env)
    base_cwd = Path(cwd).resolve() if cwd is not None else Path(workdir).resolve()
    raw_str = str(target_path).strip()
    candidate = Path(raw_str)
    if not candidate.is_absolute():
        resolved = (base_cwd / candidate).resolve()
    else:
        resolved = candidate.resolve()

    out_candidates: list[str] = []
    if out_dir is not None:
        out_candidates.append(str(out_dir).strip())
    source_env = env if env is not None else os.environ
    for env_key in ("CC_AGY_OUT_DIR", "CC_AGY_DATA_DIR", "CC_AGY_WORKTREES_DIR", "CLAUDE_PLUGIN_DATA"):
        val = source_env.get(env_key, "").strip()
        if val:
            out_candidates.append(val)

    for out_str in out_candidates:
        if out_str and not out_str.startswith("${"):
            out_resolved = Path(out_str).resolve()
            if resolved.is_relative_to(out_resolved):
                return "out_dir"

    for part in resolved.parts:
        if part.startswith(("cc-agy-worktrees-", ".cc-agy-worktrees-")):
            return "out_dir"

    for data_candidate in (
        ws_resolved / ".cc-agy-data",
        Path.home() / ".cache" / "cc-agy-plugin",
    ):
        if resolved.is_relative_to(data_candidate.resolve()):
            return "out_dir"

    if resolved.is_relative_to(ws_resolved):
        return "workspace"

    if not candidate.is_absolute() and ".." in candidate.parts:
        return "workspace"

    for tmp_root in (Path("/tmp").resolve(), Path("/var/tmp").resolve()):
        if resolved.is_relative_to(tmp_root):
            return "tmp_scratch"

    return "external"


_receipt_owner_pid_is_alive = receipt_owner_pid_alive
_is_verified_applied_receipt = is_verified_applied_receipt


@dataclass(frozen=True)
class SessionReceiptSummary:
    """Single-pass snapshot of current-session receipts for `PreToolUse` decisions."""

    has_any_session_job_artifact: bool = False
    has_session_json_files: bool = False
    has_parsed_session_json_receipt: bool = False
    has_live_running_any: bool = False
    live_running_job_ids: tuple[str, ...] = ()
    has_live_running_delegate: bool = False
    has_terminal_session_receipt: bool = False
    has_terminal_delegate: bool = False
    has_multifile_applied_delegate: bool = False
    has_terminal_any_job: bool = False
    has_applied_any_job: bool = False
    sticky_verified_applied: bool = False
    escalated_delegate_receipts: tuple[dict[str, Any], ...] = ()

    @property
    def has_lead_escalated_job(self) -> bool:
        return bool(self.escalated_delegate_receipts) and not self.has_live_running_delegate


def summarize_session_receipts(
    data_root: Path,
    ws_root: Path | None = None,
) -> SessionReceiptSummary:
    """Read current-session `receipts/job-*.json` files once and derive all hook state flags."""
    try:
        receipts_dir = data_root / "receipts"
        if not receipts_dir.is_dir():
            return SessionReceiptSummary()
        marker = data_root / ".lead_session_started"
        has_marker = marker.exists()
        since = marker.stat().st_mtime if has_marker else 0.0
        since_esc = (since - 1.0) if has_marker else 0.0
        target_ws = str(Path(ws_root).resolve()) if ws_root is not None else None

        has_any_artifact = False
        has_session_json = False
        has_parsed_session_json = False
        has_live_any = False
        live_running_ids: list[str] = []
        has_live_delegate = False
        has_terminal_delegate = False
        has_multifile_delegate = False
        terminal_recs: list[tuple[int, dict[str, Any]]] = []
        escalated_recs: list[dict[str, Any]] = []
        sig_store: Any = None

        for p in sorted(receipts_dir.glob("job-*")):
            try:
                st = p.stat()
            except OSError:
                continue
            mtime = st.st_mtime
            if mtime >= since:
                has_any_artifact = True
            name = p.name
            if not name.endswith(".json") or name.endswith((".progress.jsonl", ".live.json")):
                continue
            if mtime >= since:
                has_session_json = True
            if mtime < since_esc and (target_ws is None or has_marker):
                continue
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if not isinstance(rec, dict):
                continue

            kind = str(rec.get("kind") or "delegate")
            status = str(rec.get("status") or "")
            is_cancelled = status == "cancelled" or bool(rec.get("cancelled"))

            if mtime >= since:
                has_parsed_session_json = True
                if status == "running" and receipt_owner_pid_alive(rec):
                    has_live_any = True
                    jid = str(rec.get("job_id") or "")
                    if jid:
                        live_running_ids.append(jid)
                    if kind == "delegate":
                        has_live_delegate = True
                elif status != "running" and not is_cancelled:
                    terminal_recs.append((st.st_mtime_ns, rec))
                    if kind == "delegate":
                        has_terminal_delegate = True
                if kind == "delegate" and rec.get("applied"):
                    applied_files = rec.get("applied_files") or []
                    if isinstance(applied_files, list) and len(applied_files) >= 2:
                        has_multifile_delegate = True

            if (
                kind == "delegate"
                and status != "running"
                and not is_cancelled
                and not rec.get("external_worktree_tamper")
            ):
                if target_ws is not None and rec.get("workspace_root") not in (None, "/workspace", target_ws):
                    continue
                esc_info = rec.get("escalation")
                claims_esc = rec.get("escalate_to_lead") is True or (
                    isinstance(esc_info, dict) and esc_info.get("escalate_to_lead") is True
                )
                if not claims_esc:
                    continue
                if target_ws is not None:
                    rec_ts = float(
                        rec.get("updated_at_epoch")
                        or rec.get("finished_at_epoch")
                        or rec.get("started_at_epoch")
                        or 0.0
                    )
                    if rec_ts < since_esc:
                        continue
                elif mtime < since_esc:
                    continue
                if sig_store is None:
                    from server.receipts import ReceiptStore

                    sig_store = (
                        ReceiptStore(data_root, workspace_root=ws_root)
                        if ws_root is not None
                        else ReceiptStore(data_root)
                    )
                if sig_store.verify_receipt_signature(rec):
                    escalated_recs.append(rec)

        has_terminal_any = any(bool(r.get("job_id")) for _mt, r in terminal_recs)
        has_applied_any = any(
            bool(r.get("job_id")) and bool(r.get("applied")) for _mt, r in terminal_recs
        )
        sticky_verified = False
        if terminal_recs and not has_live_any:
            ordered_recs = sorted(
                terminal_recs,
                key=lambda item: (float(item[1].get("started_at_epoch") or 0.0), item[0]),
            )
            for _mt, rec in ordered_recs:
                if is_verified_applied_receipt(rec):
                    sticky_verified = True
                elif rec.get("reverted") or bool(rec.get("files_changed")):
                    sticky_verified = False

        return SessionReceiptSummary(
            has_any_session_job_artifact=has_any_artifact,
            has_session_json_files=has_session_json,
            has_parsed_session_json_receipt=has_parsed_session_json,
            has_live_running_any=has_live_any,
            live_running_job_ids=tuple(live_running_ids),
            has_live_running_delegate=has_live_delegate,
            has_terminal_session_receipt=bool(terminal_recs),
            has_terminal_delegate=has_terminal_delegate,
            has_multifile_applied_delegate=has_multifile_delegate,
            has_terminal_any_job=has_terminal_any,
            has_applied_any_job=has_applied_any,
            sticky_verified_applied=sticky_verified,
            escalated_delegate_receipts=tuple(escalated_recs),
        )
    except (OSError, ValueError, TypeError, KeyError):
        return SessionReceiptSummary()


def _session_has_multifile_delegate_job(
    data_root: Path,
    summary: SessionReceiptSummary | None = None,
) -> bool:
    """Return True when the current session has an applied delegate job touching >=2 files."""
    active = summary if summary is not None else summarize_session_receipts(data_root)
    return active.has_multifile_applied_delegate


def _session_has_live_running_delegate_job(
    data_root: Path,
    summary: SessionReceiptSummary | None = None,
) -> bool:
    """Return True when the current session has an active `running` delegate receipt owned by a live PID."""
    active = summary if summary is not None else summarize_session_receipts(data_root)
    return active.has_live_running_delegate


def _session_has_terminal_delegate_job(
    data_root: Path,
    summary: SessionReceiptSummary | None = None,
) -> bool:
    """Return True when the current session has at least one completed (non-running, non-cancelled) delegate job."""
    active = summary if summary is not None else summarize_session_receipts(data_root)
    return active.has_terminal_delegate


def _session_has_lead_escalated_job(
    data_root: Path,
    ws_root: Path | None = None,
    summary: SessionReceiptSummary | None = None,
) -> bool:
    """Return True when any completed/stopped delegate receipt in the current session has `escalate_to_lead=True`
    and no worker job is actively running."""
    active = summary if summary is not None else summarize_session_receipts(data_root, ws_root)
    return active.has_lead_escalated_job


_UNLIMITED_READ_VALUES = ("off", "none", "unlimited", "disabled")


def resolve_lead_read_limit(env: dict[str, str]) -> int | None:
    """Return the strict-mode Lead read/search budget before review bonuses, or None for no cap.

    `CC_AGY_LEAD_READ_LIMIT` (alias `CC_AGY_MAX_LEAD_READS`): a non-negative integer is the number of
    reads allowed (`0` means none before delegating); a negative number or `off` removes the cap. When
    unset or invalid, all profiles default to 2 (so Economy allows up to 2 lightweight Glob/Grep/Read
    scouting calls when entrypoints or test paths are uncertain, while pre-delegation Bash remains blocked).
    """
    default = 2
    raw = str(env.get("CC_AGY_LEAD_READ_LIMIT") or env.get("CC_AGY_MAX_LEAD_READS") or "").strip().lower()
    if not raw or raw.startswith("${"):
        return default
    if raw in _UNLIMITED_READ_VALUES:
        return None
    try:
        value = int(raw)
    except ValueError:
        return default
    return None if value < 0 else value


def _lead_review_read_bonus(
    data_root: Path,
    env: dict[str, str],
    summary: SessionReceiptSummary | None = None,
) -> int:
    """Extra strict-mode Lead reads granted once a worker job started in the current session exists.

    When a job auto-applied cleanly with passing acceptance checks and no missing symbols (and
    `CC_AGY_LEAD_REVIEW_READ_LIMIT` was not explicitly overridden), returns 0 greedily and stickily across
    subsequent 0-edit follow-up jobs so the Lead finishes immediately instead of re-reading verified files.
    """
    raw_override = (env.get("CC_AGY_LEAD_REVIEW_READ_LIMIT") or "").strip()
    try:
        bonus = int(raw_override or "6")
    except ValueError:
        bonus = 6
    if bonus <= 0:
        return 0
    active = summary if summary is not None else summarize_session_receipts(data_root)
    if not active.has_any_session_job_artifact:
        return 0
    if active.has_session_json_files:
        if (
            not active.has_parsed_session_json_receipt
            or active.has_live_running_any
            or not active.has_terminal_session_receipt
        ):
            return 0
        if not raw_override and active.sticky_verified_applied:
            return 0
    return bonus



def _check_strict_bash_test_runner_guard(
    command: str,
    ws_root: Path,
    out_dir: Path | str | None,
    env: dict[str, str],
    summary: SessionReceiptSummary | None = None,
) -> str | None:
    """Prevent Lead host test-suite runs while a worker job is running or in Economy/zero-read mode."""
    from server.guard_bash import _command_invokes_host_test_runner

    if not _command_invokes_host_test_runner(command):
        return None
    try:
        data_root = _resolve_data_dir_fast(out_dir or None, ws_root, env)
        active = summary if summary is not None else summarize_session_receipts(data_root, ws_root)
        if active.live_running_job_ids:
            jid = active.live_running_job_ids[0]
            return (
                f"Strict orchestrator mode: delegated worker job {jid!r} is still running in its isolated "
                f"worktree. Do NOT run test suites on the un-applied host workspace; call "
                f"delegate_result(job_id={jid!r}, wait=true) to wait for the worker."
            )
        if resolve_cost_profile(env) != "economy" and resolve_lead_read_limit(env) != 0:
            return None
        if active.has_lead_escalated_job:
            return None
        if not active.has_terminal_any_job or not active.has_applied_any_job:
            return (
                "Economy strict mode: no worker diff has been applied to the host workspace yet (host is still "
                "at clean HEAD, and running tests on the host can write untracked test artifacts that conflict "
                "with delegate_apply). Delegate test execution and fixes via delegate_start or delegate_continue."
            )
        if _lead_review_read_bonus(data_root, env, summary=active) == 0:
            return (
                "Economy strict mode: worker changes were already verified by the acceptance test suite and "
                "applied cleanly. Finish your response directly without re-running tests on the host."
            )
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return None



def evaluate_strict_pre_tool_use(
    tool_name: str,
    tool_input: dict[str, Any],
    cwd: str,
    ws_root: Path,
    out_dir: str | None,
    source_env: dict[str, str],
) -> dict[str, Any]:
    """Evaluate strict-mode PreToolUse guards for Read/Grep/Glob, Edit/Write, and Bash."""
    if tool_name == "Read":
        raw_fp_strict = str(tool_input.get("file_path") or tool_input.get("path") or "").strip()
        if raw_fp_strict and Path(raw_fp_strict).suffix.lower() in (".mp4", ".mov", ".webm", ".mpeg"):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Claude Read cannot inspect video file '{raw_fp_strict}'. "
                        f"Call `ask_gemini(prompt=..., media=[{raw_fp_strict!r}])` or "
                        f"`delegate_start(prompt=..., media_files=[{raw_fp_strict!r}], wait=true)`."
                    ),
                }
            }

    _cached_data_root: Path | None = None
    _cached_summary: SessionReceiptSummary | None = None

    def _get_data_root_and_summary() -> tuple[Path, SessionReceiptSummary]:
        nonlocal _cached_data_root, _cached_summary
        if _cached_data_root is None:
            _cached_data_root = _resolve_data_dir_fast(out_dir or None, ws_root, source_env)
        if _cached_summary is None:
            _cached_summary = summarize_session_receipts(_cached_data_root, ws_root)
        return _cached_data_root, _cached_summary

    # 2a. Block Read / Grep / Glob on out_dir (.hmac_key / receipts) or PLUGIN_ROOT and enforce Lead read cap in strict mode
    if tool_name in ("Read", "Grep", "Glob"):
        if (
            ws_root != PLUGIN_ROOT
            and tool_name in ("Grep", "Glob")
            and not tool_input.get("path")
            and Path(cwd).resolve().is_relative_to(PLUGIN_ROOT)
        ):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Strict orchestrator mode is active: {tool_name} on cc-agy-plugin internal source "
                        "is disabled. Use the MCP tools (`delegate_start`, `delegate_continue`, "
                        "`get_diff`, `delegate_apply`) directly."
                    ),
                }
            }
        for field in ("file_path", "path", "pattern", "glob"):
            target_val = tool_input.get(field)
            if not target_val:
                continue
            raw_t = str(target_val)
            if field in ("file_path", "path"):
                try:
                    cand_p = Path(raw_t).expanduser()
                    resolved_t = (Path(cwd).resolve() / cand_p).resolve() if not cand_p.is_absolute() else cand_p.resolve()
                    if ws_root != PLUGIN_ROOT and resolved_t.is_relative_to(PLUGIN_ROOT):
                        return {
                            "hookSpecificOutput": {
                                "hookEventName": "PreToolUse",
                                "permissionDecision": "deny",
                                "permissionDecisionReason": (
                                    f"Strict orchestrator mode is active: {tool_name} on cc-agy-plugin internal source "
                                    f"{raw_t!r} is disabled. Use the MCP tools (`delegate_start`, `delegate_continue`, "
                                    "`get_diff`, `delegate_apply`) directly."
                                ),
                            }
                        }
                except OSError:
                    pass
            classification = classify_path(raw_t, ws_root, out_dir, cwd=cwd, env=source_env)
            if (
                classification == "out_dir"
                or ".hmac_key" in raw_t
                or ".cc-agy-data" in raw_t
            ):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Strict orchestrator mode is active: {tool_name} on internal receipt/key store "
                            f"{raw_t!r} is disabled. Use delegate_status or get_diff instead."
                        ),
                    }
                }
        read_limit = resolve_lead_read_limit(source_env)
        if read_limit is not None:
            data_root, sess_summary = _get_data_root_and_summary()
            if resolve_cost_profile(source_env) == "economy" and _session_has_live_running_delegate_job(
                data_root, summary=sess_summary
            ):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Still running: a delegated worker job is actively running. Do NOT call {tool_name} on "
                            "host files while it runs; poll with delegate_result(job_id=..., wait=true) instead."
                        ),
                    }
                }
            if _session_has_lead_escalated_job(data_root, ws_root, summary=sess_summary):
                return {}
            counter_path = data_root / ".lead_read_count"
            # Once a worker job from this session exists, the Lead gets extra reads to check the
            # result against the task. Without them a Lead that spent its planning reads has to
            # accept the worker's result unchecked.
            read_limit += _lead_review_read_bonus(data_root, source_env, summary=sess_summary)
            try:
                current_reads = int(counter_path.read_text(encoding="utf-8").strip()) if counter_path.exists() else 0
            except (OSError, ValueError):
                current_reads = 0
            read_unit_cost = 1
            if (
                tool_name == "Read"
                and resolve_cost_profile(source_env) == "economy"
                and not _session_has_terminal_delegate_job(data_root, summary=sess_summary)
            ):
                raw_lim_chk = tool_input.get("limit")
                try:
                    lim_int = int(raw_lim_chk) if raw_lim_chk is not None else 999999
                except (ValueError, TypeError):
                    lim_int = 999999
                if lim_int > 150 and tool_input.get("file_path"):
                    try:
                        req_c = Path(str(tool_input["file_path"])).expanduser()
                        res_c = (Path(cwd).resolve() / req_c).resolve() if not req_c.is_absolute() else req_c.resolve()
                        if res_c.is_relative_to(ws_root) and res_c.is_file():
                            with open(res_c, "rb") as fh_c:
                                line_cnt = sum(1 for _ in fh_c)
                            if line_cnt > 150:
                                read_unit_cost = 2
                    except (ImportError, OSError, ValueError, TypeError):
                        pass
            if current_reads >= read_limit or current_reads + read_unit_cost > read_limit:
                reason_msg = (
                    f"LeadReadBudgetExceeded: Lead read/search limit ({read_limit} calls) reached in strict "
                    "delegation mode. Do NOT read more files on the Lead orchestrator. Call delegate_start "
                    "(with wait=true and a concise prompt <500 words; preload_files=true and auto_apply=true "
                    "are enabled automatically), delegate_continue, get_diff, or delegate_apply "
                    "immediately; the worker has full repository and test access in its worktree."
                )
                if tool_name == "Read" and tool_input.get("file_path"):
                    try:
                        from server.repo_map import build_focused_file_outline

                        req_p = Path(str(tool_input["file_path"]))
                        res_p = (Path(cwd).resolve() / req_p).resolve() if not req_p.is_absolute() else req_p.resolve()
                        if res_p.is_relative_to(ws_root) and res_p.is_file():
                            rel_p = str(res_p.relative_to(ws_root))
                            outline = build_focused_file_outline(ws_root, [rel_p], max_tokens=450)
                            if outline:
                                reason_msg += (
                                    "\n\nFocused File Outline (auto-attached so you can call delegate_start immediately):\n"
                                    + outline
                                )
                    except (ImportError, OSError, ValueError, TypeError):
                        pass
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason_msg,
                    }
                }
            try:
                data_root.mkdir(parents=True, exist_ok=True)
                counter_path.write_text(str(current_reads + read_unit_cost), encoding="utf-8")
                if tool_name == "Read":
                    raw_fp = str(tool_input.get("file_path") or tool_input.get("path") or "").strip()
                    if raw_fp:
                        cand_fp = Path(raw_fp).expanduser()
                        res_fp = (Path(cwd).resolve() / cand_fp).resolve() if not cand_fp.is_absolute() else cand_fp.resolve()
                        if res_fp.is_relative_to(ws_root) and res_fp.is_file():
                            rel_fp = res_fp.relative_to(ws_root).as_posix()
                            read_files_path = data_root / ".lead_read_files.json"
                            existing_list: list[str] = []
                            if read_files_path.exists():
                                try:
                                    parsed_rf = json.loads(read_files_path.read_text(encoding="utf-8"))
                                    if isinstance(parsed_rf, list):
                                        existing_list = [str(x) for x in parsed_rf if isinstance(x, str)]
                                except (OSError, ValueError, TypeError):
                                    existing_list = []
                            if rel_fp not in existing_list:
                                existing_list.append(rel_fp)
                                read_files_path.write_text(json.dumps(existing_list), encoding="utf-8")
                            raw_off = tool_input.get("offset")
                            raw_lim = tool_input.get("limit")
                            if raw_off is not None or raw_lim is not None:
                                try:
                                    w_off = max(1, int(raw_off)) if raw_off is not None else 1
                                    w_lim = max(50, int(raw_lim)) if raw_lim is not None else 250
                                    rw_path = data_root / ".lead_read_windows.json"
                                    existing_wins: dict[str, list[int]] = {}
                                    if rw_path.exists():
                                        parsed_rw = json.loads(rw_path.read_text(encoding="utf-8"))
                                        if isinstance(parsed_rw, dict):
                                            existing_wins = parsed_rw
                                    existing_wins[rel_fp] = [w_off, w_lim]
                                    rw_path.write_text(json.dumps(existing_wins), encoding="utf-8")
                                except (OSError, ValueError, TypeError):
                                    pass
            except (OSError, ValueError, TypeError):
                pass

    # 2b. Block direct Edit / Write / MultiEdit / NotebookEdit on workspace, out_dir (receipts), and PLUGIN_ROOT
    if tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        file_path = (
            tool_input.get("file_path")
            or tool_input.get("notebook_path")
            or tool_input.get("path")
            or ""
        )
        if file_path:
            cand_edit = Path(str(file_path))
            res_edit = (Path(cwd).resolve() / cand_edit).resolve() if not cand_edit.is_absolute() else cand_edit.resolve()
            if ws_root != PLUGIN_ROOT and res_edit.is_relative_to(PLUGIN_ROOT):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Strict orchestrator mode is active: direct {tool_name} on cc-agy-plugin source "
                            f"{file_path!r} is disabled."
                        ),
                    }
                }
            classification = classify_path(
                str(file_path), ws_root, out_dir, cwd=cwd, env=source_env
            )
            if classification == "workspace" and tool_name in ("Edit", "Write"):
                try:
                    from server.jobs import _is_test_file_path, _matches_allowed_paths

                    esc_data_root, sess_summary = _get_data_root_and_summary()
                    rel_edit = res_edit.relative_to(ws_root).as_posix()
                    if (
                        rel_edit
                        and not rel_edit.startswith((".git/", ".cc-agy-data/", ".worktrees/"))
                        and not res_edit.is_symlink()
                        and sess_summary.has_lead_escalated_job
                    ):
                        esc_cnt_path = esc_data_root / ".lead_escalated_edit_count"
                        try:
                            cur_esc_edits = (
                                int(esc_cnt_path.read_text(encoding="utf-8").strip())
                                if esc_cnt_path.exists()
                                else 0
                            )
                        except (OSError, ValueError):
                            cur_esc_edits = 0
                        if cur_esc_edits < 3:
                            authorized_esc = False
                            for r_esc in sess_summary.escalated_delegate_receipts:
                                if r_esc.get("reverted"):
                                    continue
                                r_allowed = list(r_esc.get("allowed_paths") or [])
                                r_applied_files = set(r_esc.get("applied_files") or []) | set(r_esc.get("files_changed") or [])
                                if (
                                    rel_edit in r_applied_files
                                    or not r_allowed
                                    or _matches_allowed_paths(rel_edit, r_allowed)
                                    or _is_test_file_path(rel_edit)
                                ):
                                    authorized_esc = True
                                    break
                            if authorized_esc:
                                esc_data_root.mkdir(parents=True, exist_ok=True)
                                esc_cnt_path.write_text(str(cur_esc_edits + 1), encoding="utf-8")
                                return {}
                except (ImportError, OSError, ValueError, TypeError):
                    pass
            if classification in ("workspace", "out_dir"):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Strict orchestrator mode is active: direct {tool_name} on {classification} file "
                            f"{file_path!r} is disabled. Delegate code modifications to a Vertex Gemini "
                            f"worker using delegate_start or delegate_to_gemini."
                        ),
                    }
                }

    # 3. Block Bash mutations targeting workspace or out_dir, and enforce Lead read cap on Bash workspace reads
    if tool_name == "Bash":
        from server.guard_bash import (
            _command_invokes_host_test_runner,
            _is_readonly_git_diff_command,
            detect_bash_workspace_mutation,
            detect_bash_workspace_read,
        )
        command = str(tool_input.get("command") or "")
        verdict = detect_bash_workspace_mutation(
            command, ws_root, out_dir, cwd=cwd, env=source_env
        )
        if verdict["blocked"]:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": verdict["reason"],
                }
            }
        if _command_invokes_host_test_runner(command):
            _bash_dr, bash_summary = _get_data_root_and_summary()
            test_guard_reason = _check_strict_bash_test_runner_guard(
                command, ws_root, out_dir, source_env, summary=bash_summary
            )
            if test_guard_reason:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": test_guard_reason,
                    }
                }
        bash_read_targets = detect_bash_workspace_read(
            command, ws_root, out_dir, cwd=cwd, env=source_env
        )
        if bash_read_targets:
            read_limit = resolve_lead_read_limit(source_env)
            if read_limit is not None:
                data_root, sess_summary = _get_data_root_and_summary()
                if _session_has_lead_escalated_job(data_root, ws_root, summary=sess_summary):
                    return {}
                if (
                    resolve_cost_profile(source_env) == "economy"
                    and read_limit > 0
                    and not _session_has_terminal_delegate_job(data_root, summary=sess_summary)
                ):
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                "Economy strict mode: pre-delegation Bash file/search inspection is disabled so "
                                "Turn 1 stays fast and clean. Use Glob, Grep, or bounded Read(limit=120) "
                                f"(up to {read_limit} calls) if target paths are uncertain, or call delegate_start directly."
                            ),
                        }
                    }
                counter_path = data_root / ".lead_read_count"
                read_limit += _lead_review_read_bonus(data_root, source_env, summary=sess_summary)
                try:
                    current_reads = int(counter_path.read_text(encoding="utf-8").strip()) if counter_path.exists() else 0
                except Exception:
                    current_reads = 0
                file_targets = [t for t in bash_read_targets if t != "<workspace-search>"]
                span_cost = 1
                for sed_m in re.finditer(r"\bsed\b[^;|&\n]*?['\"]?(\d+)\s*,\s*(\d+)\s*p", command):
                    try:
                        s_lo, s_hi = int(sed_m.group(1)), int(sed_m.group(2))
                        if s_hi >= s_lo:
                            span_cost = max(span_cost, (s_hi - s_lo + 120) // 120)
                    except ValueError:
                        pass
                if re.search(r"(?:^|[\s;|&`(])cat\b", command) and file_targets:
                    cat_lines = 0
                    for rel_ft in file_targets[:4]:
                        fp_cat = ws_root / rel_ft
                        if fp_cat.is_file():
                            try:
                                with open(fp_cat, "rb") as fh_cat:
                                    cat_lines += sum(1 for _ in fh_cat)
                            except OSError:
                                pass
                    if cat_lines > 120:
                        span_cost = max(span_cost, min(6, (cat_lines + 119) // 120))
                read_cost = max(1, len(file_targets), span_cost)
                spotcheck_marker = data_root / ".lead_git_diff_spotcheck_used"
                session_marker = data_root / ".lead_session_started"
                cur_session_sig = ""
                if session_marker.exists():
                    try:
                        cur_session_sig = session_marker.read_text(encoding="utf-8").strip() or str(
                            session_marker.stat().st_mtime_ns
                        )
                    except OSError:
                        cur_session_sig = ""
                spotcheck_used_in_session = False
                if spotcheck_marker.exists():
                    try:
                        prev_sig = spotcheck_marker.read_text(encoding="utf-8").strip()
                        if not cur_session_sig:
                            spotcheck_used_in_session = True
                        elif prev_sig == cur_session_sig:
                            spotcheck_used_in_session = True
                        elif session_marker.exists() and spotcheck_marker.stat().st_mtime >= session_marker.stat().st_mtime and prev_sig == "1":
                            spotcheck_used_in_session = True
                    except OSError:
                        spotcheck_used_in_session = True
                allow_multifile_git_diff_spotcheck = (
                    _is_readonly_git_diff_command(command)
                    and not spotcheck_used_in_session
                    and _session_has_multifile_delegate_job(data_root, summary=sess_summary)
                )
                if allow_multifile_git_diff_spotcheck:
                    try:
                        data_root.mkdir(parents=True, exist_ok=True)
                        spotcheck_marker.write_text(cur_session_sig or "1", encoding="utf-8")
                    except OSError:
                        pass
                    read_cost = 0
                elif current_reads >= read_limit or current_reads + read_cost > read_limit:
                    reason_msg = (
                        f"LeadReadBudgetExceeded: Lead read/search limit ({read_limit} calls) reached in strict "
                        "delegation mode (including Bash file/diff inspection commands). Do NOT read more files on "
                        "the Lead orchestrator. Call delegate_start (with wait=true and a concise prompt <500 words; "
                        "preload_files=true and auto_apply=true are enabled automatically), delegate_continue, "
                        "get_diff, or delegate_apply immediately; the worker has full repository and test access."
                    )
                    if file_targets:
                        try:
                            from server.repo_map import build_focused_file_outline

                            outline = build_focused_file_outline(ws_root, file_targets[:2], max_tokens=450)
                            if outline:
                                reason_msg += (
                                    "\n\nFocused File Outline (auto-attached so you can call delegate_start immediately):\n"
                                    + outline
                                )
                        except (ImportError, OSError, ValueError, TypeError):
                            pass
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": reason_msg,
                        }
                    }
                try:
                    data_root.mkdir(parents=True, exist_ok=True)
                    counter_path.write_text(str(current_reads + read_cost), encoding="utf-8")
                    if file_targets:
                        read_files_path = data_root / ".lead_read_files.json"
                        existing_list: list[str] = []
                        if read_files_path.exists():
                            try:
                                parsed_rf = json.loads(read_files_path.read_text(encoding="utf-8"))
                                if isinstance(parsed_rf, list):
                                    existing_list = [str(x) for x in parsed_rf if isinstance(x, str)]
                            except (OSError, ValueError, TypeError):
                                existing_list = []
                        changed = False
                        for rel_fp in file_targets:
                            if rel_fp not in existing_list:
                                existing_list.append(rel_fp)
                                changed = True
                        if changed:
                            read_files_path.write_text(json.dumps(existing_list), encoding="utf-8")
                except (ImportError, OSError, ValueError, TypeError):
                    pass

    return {}
