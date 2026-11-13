"""Bounded acceptance command execution, test-count parsing, and baseline regression evaluation."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from typing import Any

from server.job_git_utils import (
    ACCEPTANCE_OUTPUT_TOKENS,
    DEFAULT_ACCEPTANCE_TIMEOUT,
    DEFAULT_BOUNDED_ACCEPTANCE_TIMEOUT,
    _env_int,
    _git_raw,
    _matches_allowed_paths,
    _run_git,
)
from server.regions import resolve_cost_profile, truncate_to_token_budget
from server.test_discovery import (
    _build_workdir_exec_env,
    _collect_workspace_node_bin_dirs,
    _list_tracked_repo_files,
)
from server.verification_policy import command_strength, is_non_smoke_passing_entry

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _parse_test_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "errors": 0}
    text = _ANSI_ESCAPE_RE.sub("", output or "")

    # 1. Check anchored pytest summary lines first so prose like "expected 2 errors" in a unittest/stestr
    # log never hijacks the count parser.
    pytest_summary_lines = [
        ln
        for ln in text.splitlines()
        if re.search(r"=+.*\bin\s+\d+(?:\.\d+)?s\b.*=+", ln)
        or re.search(
            r"^\s*\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?)\b.*\bin\s+\d+(?:\.\d+)?s\b",
            ln,
        )
    ]
    if pytest_summary_lines:
        for ln in pytest_summary_lines:
            for key, pat in (
                ("passed", r"(\d+)\s+passed\b"),
                ("failed", r"(\d+)\s+failed\b"),
                ("errors", r"(\d+)\s+errors?\b"),
            ):
                for m in re.finditer(pat, ln):
                    counts[key] += int(m.group(1))
        return counts

    # 2. stestr / subunit summary blocks (e.g. "- Passed: 162", "- Failed: 1")
    st_passed = re.findall(r"-\s*Passed:\s*(\d+)", text)
    st_failed = re.findall(r"-\s*Failed:\s*(\d+)", text)
    st_errors = re.findall(r"-\s*(?:Errors?|Unexpected Success(?:es)?):\s*(\d+)", text)
    if st_passed or st_failed or st_errors:
        if st_passed:
            counts["passed"] = int(st_passed[-1])
        if st_failed:
            counts["failed"] = int(st_failed[-1])
        if st_errors:
            counts["errors"] = sum(int(x) for x in st_errors)
        return counts

    # 3. Standard library unittest summary ("Ran N tests in 0.123s" + "OK" / "FAILED (...)")
    # Sum across multiple unittest suite invocations in the same output and tolerate trailing teardown lines.
    ran_iter = list(re.finditer(r"Ran (\d+) tests? in [\d.]+s", text))
    if ran_iter:
        matched_any_suite = False
        for idx_r, m_ran in enumerate(ran_iter):
            suite_total = int(m_ran.group(1))
            block_start = m_ran.end()
            block_end = ran_iter[idx_r + 1].start() if idx_r + 1 < len(ran_iter) else len(text)
            suite_tail = text[block_start:block_end]
            m_fail = re.search(r"(?m)^\s*FAILED\s*\(([^)]+)\)", suite_tail)
            m_ok = re.search(r"(?m)^\s*OK(?:\s*\(([^)]*)\))?\s*$", suite_tail)
            if m_fail:
                matched_any_suite = True
                fail_meta = m_fail.group(1)
                f_m = re.search(r"failures=(\d+)", fail_meta)
                e_m = re.search(r"errors=(\d+)", fail_meta)
                s_m = re.search(r"skipped=(\d+)", fail_meta)
                f_cnt = int(f_m.group(1)) if f_m else 0
                e_cnt = int(e_m.group(1)) if e_m else 0
                s_cnt = int(s_m.group(1)) if s_m else 0
                counts["failed"] += f_cnt
                counts["errors"] += e_cnt
                counts["passed"] += max(0, suite_total - f_cnt - e_cnt - s_cnt)
            elif m_ok:
                matched_any_suite = True
                ok_meta = m_ok.group(1) or ""
                s_m = re.search(r"skipped=(\d+)", ok_meta)
                s_cnt = int(s_m.group(1)) if s_m else 0
                counts["passed"] += max(0, suite_total - s_cnt)
        if matched_any_suite:
            return counts

    # 4. Rust cargo test ("test result: ok. 12 passed; 0 failed; 0 ignored; ...")
    cargo_matches = list(
        re.finditer(r"test result:\s*(?:ok|FAILED)\.\s*(\d+)\s+passed;\s*(\d+)\s+failed;", text)
    )
    if cargo_matches:
        for cm in cargo_matches:
            counts["passed"] += int(cm.group(1))
            counts["failed"] += int(cm.group(2))
        return counts

    # 5. Jest / Vitest ("Tests: 1 failed, 14 passed, 15 total" or "Tests  14 passed (14)")
    jest_lines = [
        ln for ln in text.splitlines() if re.match(r"^\s*Tests:?\s+", ln) and re.search(r"\b(?:passed|failed)\b", ln)
    ]
    if jest_lines:
        last_jest = jest_lines[-1]
        m_jp = re.search(r"(\d+)\s+passed\b", last_jest)
        m_jf = re.search(r"(\d+)\s+failed\b", last_jest)
        if m_jp:
            counts["passed"] = int(m_jp.group(1))
        if m_jf:
            counts["failed"] = int(m_jf.group(1))
        return counts

    # 6. Go `go test` ("--- PASS: TestX", "--- FAIL: TestY", or package-level "ok  pkg  0.012s" / "FAIL pkg 0.012s")
    go_case_pass = len(re.findall(r"(?m)^---\s+PASS:\s+\S+", text))
    go_case_fail = len(re.findall(r"(?m)^---\s+FAIL:\s+\S+", text))
    if go_case_pass or go_case_fail:
        counts["passed"] = go_case_pass
        counts["failed"] = go_case_fail
        return counts
    go_pkg_ok = len(re.findall(r"(?m)^ok\s+\S+\s+(?:\d+(?:\.\d+)?s|\(cached\))", text))
    go_pkg_fail = len(re.findall(r"(?m)^FAIL\s+\S+\s+\d+(?:\.\d+)?s", text))
    if go_pkg_ok or go_pkg_fail:
        counts["passed"] = go_pkg_ok
        counts["failed"] = go_pkg_fail
        return counts

    # 7. Node.js `node --test` TAP/spec footer ("ℹ pass 5" / "# pass 5", "ℹ fail 1" / "# fail 1")
    node_pass = re.findall(r"(?m)^[\sℹ#]*pass\s+(\d+)\s*$", text)
    node_fail = re.findall(r"(?m)^[\sℹ#]*fail\s+(\d+)\s*$", text)
    if node_pass or node_fail:
        if node_pass:
            counts["passed"] = int(node_pass[-1])
        if node_fail:
            counts["failed"] = int(node_fail[-1])
        return counts

    # 8. Maven Surefire ("Tests run: 10, Failures: 1, Errors: 0, Skipped: 0")
    mvn_matches = list(
        re.finditer(
            r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)",
            text,
        )
    )
    if mvn_matches:
        last_m = mvn_matches[-1]
        total_m, fail_m, err_m, skip_m = (int(last_m.group(i)) for i in range(1, 5))
        counts["failed"] = fail_m
        counts["errors"] = err_m
        counts["passed"] = max(0, total_m - fail_m - err_m - skip_m)
        return counts

    # 9. Gradle ("5 tests completed, 1 failed, 1 skipped")
    gradle_matches = list(
        re.finditer(
            r"(\d+)\s+tests?\s+completed(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+skipped)?",
            text,
        )
    )
    if gradle_matches:
        total_g = sum(int(m.group(1)) for m in gradle_matches)
        fail_g = sum(int(m.group(2) or 0) for m in gradle_matches)
        skip_g = sum(int(m.group(3) or 0) for m in gradle_matches)
        counts["failed"] = fail_g
        counts["passed"] = max(0, total_g - fail_g - skip_g)
        return counts

    # 10. Fallback for minimal/custom pytest output without a duration footer
    for key, pat in (("passed", r"(\d+) passed\b"), ("failed", r"(\d+) failed\b"), ("errors", r"(\d+) errors?\b")):
        found = re.findall(pat, text)
        if found:
            counts[key] = int(found[-1])
    return counts


def _invalidate_python_bytecode_for_paths(root: Path, rel_paths: list[str] | tuple[str, ...]) -> None:
    """Delete stale `__pycache__/<stem>.*.pyc` bytecode for modified `.py` files inside `root`.

    CPython PEP 552 `.pyc` headers compare 32-bit integer `int(st_mtime)` seconds and `st_size`, so
    two edits within the same wall-clock second that keep the same byte length (`a * b` -> `a + b`)
    would otherwise execute stale bytecode from an earlier turn.
    """
    try:
        resolved_root = root.resolve()
    except OSError:
        return
    for raw_rel in rel_paths:
        norm = str(raw_rel or "").replace("\\", "/").strip("/")
        if not norm.endswith(".py"):
            continue
        src = root / norm
        try:
            resolved_src = src.resolve()
        except OSError:
            continue
        if not resolved_src.is_relative_to(resolved_root):
            continue
        stem = resolved_src.stem
        pycache = resolved_src.parent / "__pycache__"
        if pycache.is_dir() and not pycache.is_symlink():
            for pyc in pycache.glob(f"{stem}.*.pyc"):
                try:
                    pyc.unlink(missing_ok=True)
                except OSError:
                    pass
        legacy_pyc = resolved_src.with_suffix(".pyc")
        if legacy_pyc.is_file() and not legacy_pyc.is_symlink():
            try:
                legacy_pyc.unlink(missing_ok=True)
            except OSError:
                pass


_PYTEST_VALUE_FLAGS = frozenset({
    "-k",
    "-m",
    "-o",
    "-c",
    "-p",
    "--rootdir",
    "--confcutdir",
    "--import-mode",
    "--basetemp",
    "--tb",
    "--maxfail",
    "--durations",
    "--ignore",
    "--ignore-glob",
    "--deselect",
    "--cov",
    "--cov-report",
    "--cov-config",
})


def _strip_missing_pytest_file_targets(
    cmd: str,
    output: str,
    cwd: Path,
) -> tuple[str | None, list[str]]:
    """When a pytest command fails with `exit_code == 4` (`ERROR: file or directory not found: <path>`),
    strip non-existent file/dir target arguments if at least one existing test target remains in `cmd`."""
    if "file or directory not found:" not in (output or "").lower():
        return None, []
    stripped = cmd.strip()
    if any(sep in stripped for sep in ("&&", "||", ";", "|", "\n")):
        return None, []
    runner_pref = _extract_pytest_runner_prefix(stripped)
    if not runner_pref:
        return None, []
    try:
        tokens = shlex.split(stripped)
        pref_tokens = shlex.split(runner_pref)
    except ValueError:
        return None, []
    if len(tokens) <= len(pref_tokens):
        return None, []
    reported_missing = {
        m.strip().rstrip(",'\"")
        for m in re.findall(r"ERROR:\s+file or directory not found:\s+(\S+)", output or "", re.I)
        if m.strip()
    }
    kept_tokens: list[str] = list(tokens[: len(pref_tokens)])
    existing_targets: list[str] = []
    missing_targets: list[str] = []
    skip_next = False
    for tok in tokens[len(pref_tokens) :]:
        if skip_next:
            kept_tokens.append(tok)
            skip_next = False
            continue
        if tok.startswith("-"):
            kept_tokens.append(tok)
            if tok in _PYTEST_VALUE_FLAGS:
                skip_next = True
            continue
        file_part = tok.split("::", 1)[0].strip()
        cand_path = cwd / file_part
        if file_part and cand_path.exists():
            existing_targets.append(file_part)
            kept_tokens.append(tok)
        elif file_part and (
            file_part in reported_missing
            or tok in reported_missing
            or "/" in file_part
            or file_part.endswith(".py")
        ):
            missing_targets.append(file_part)
        else:
            kept_tokens.append(tok)
    if not missing_targets or not existing_targets:
        return None, []
    return shlex.join(kept_tokens), missing_targets


def run_acceptance(
    cmds: list[str],
    cwd: Path,
    timeout: float | None = None,
    *,
    total_wall_budget: float | None = None,
    restore_tracked: bool = True,
    change_set: list[str] | tuple[str, ...] = (),
    quarantine_dir: Path | None = None,
    diagnostic_remaps: dict[str, tuple[str, list[str]]] | None = None,
) -> list[dict[str, Any]]:
    """Run acceptance checks via ``verify.execute_command_suite`` so both delegation and batch paths
    share identical credential scrubbing, bounded output streaming, unconditional process-group
    termination, per-command workspace snapshots, and untracked artifact cleanup.

    If an explicit pytest acceptance command exits with code 4 because one or more requested targets
    are missing, any reduced or remapped command runs for diagnostic output only; ``passed`` remains
    ``False`` and ``exit_code`` stays ``4`` so a missing required test never becomes a passing result.
    """
    from server.verify import execute_check_process, execute_command_suite

    default_timeout = (
        DEFAULT_BOUNDED_ACCEPTANCE_TIMEOUT
        if (
            os.environ.get("CC_AGY_SESSION_DEADLINE_SECONDS", "").strip()
            or resolve_cost_profile() == "economy"
        )
        else DEFAULT_ACCEPTANCE_TIMEOUT
    )
    limit = float(timeout or _env_int("CC_AGY_ACCEPTANCE_TIMEOUT", int(default_timeout)))

    def _missing_target_diagnostics(
        cmd: str,
        rc: int | None,
        _timed_out: bool,
        out: str,
        rem_timeout: float,
        exec_env: Any,
    ) -> dict[str, Any] | None:
        if rc != 4:
            return None
        filtered_cmd, filtered_missing = _strip_missing_pytest_file_targets(cmd, out or "", cwd)
        if not filtered_cmd and diagnostic_remaps:
            remap_pair = diagnostic_remaps.get(cmd.strip())
            if remap_pair:
                filtered_cmd, filtered_missing = remap_pair[0], list(remap_pair[1])
        if not filtered_cmd:
            diag_details: list[dict[str, str]] = []
            remapped_list, did_diag_remap = remap_nonexistent_pytest_acceptance_cmds(
                [cmd],
                cwd,
                [{"command": cmd, "exit_code": 4, "passed": False, "error_tail": out or ""}],
                remap_details=diag_details,
            )
            if did_diag_remap and remapped_list and remapped_list[0].strip() != cmd.strip():
                filtered_cmd = remapped_list[0]
                filtered_missing = list(
                    dict.fromkeys(
                        str(d.get("missing_target") or "").strip()
                        for d in diag_details
                        if isinstance(d, dict) and d.get("missing_target")
                    )
                )
        if not filtered_cmd or not filtered_missing:
            return None
        diag_rc, diag_to, diag_out, _ = execute_check_process(
            filtered_cmd,
            cwd,
            exec_env,
            timeout=max(1.0, rem_timeout),
            max_tail_chars=20_000,
        )
        diag_header = (
            f"\n[cc-agy diagnostic: requested target(s) {filtered_missing} not found; "
            f"diagnostic run of {filtered_cmd}]\n"
        )
        combined = f"{(out or '').rstrip()}{diag_header}{diag_out or ''}"
        return {
            "force_failed": True,
            "effective_command": filtered_cmd,
            "filtered_missing_paths": filtered_missing,
            "diagnostic_exit_code": diag_rc,
            "diagnostic_timed_out": diag_to,
            "diagnostic_counts": _parse_test_counts(diag_out or ""),
            "combined_output": combined,
        }

    suite_items, cleanup_summary = execute_command_suite(
        cmds,
        cwd,
        timeout=limit,
        total_wall_budget=total_wall_budget,
        stop_on_failure=False,
        stop_on_timeout=True,
        clean_new_files=True,
        restore_tracked=restore_tracked,
        change_set=change_set,
        quarantine_dir=quarantine_dir,
        max_tail_chars=20_000,
        check_all_vacuous=True,
        diagnostic_hook=_missing_target_diagnostics,
    )
    results: list[dict[str, Any]] = []
    for cmd_res, meta in suite_items:
        if meta.get("wall_exhausted_skipped"):
            results.append(
                {
                    "command": cmd_res.command,
                    "strength": command_strength(cmd_res.command),
                    "exit_code": None,
                    "timed_out": True,
                    "passed": False,
                    "duration_seconds": 0.0,
                    "counts": {"passed": 0, "failed": 0, "errors": 0},
                    "output_tail": cmd_res.tail,
                }
            )
            continue
        raw_tail = str(meta.get("raw_tail") or "")
        tail, _, _ = truncate_to_token_budget(raw_tail[-20_000:], ACCEPTANCE_OUTPUT_TOKENS)
        if cmd_res.untrusted_side_effects:
            from server.verify import side_effect_note

            tail = (tail or "") + side_effect_note(cmd_res.untrusted_side_effects, restored=restore_tracked)
        filtered_cmd = meta.get("effective_command")
        filtered_missing_paths = meta.get("filtered_missing_paths")
        entry: dict[str, Any] = {
            "command": cmd_res.command,
            "strength": command_strength(cmd_res.command),
            "exit_code": cmd_res.exit_code,
            "timed_out": cmd_res.timed_out,
            "passed": cmd_res.passed,
            "duration_seconds": round(cmd_res.duration_seconds, 1),
            "counts": dict(meta.get("counts") or {"passed": 0, "failed": 0, "errors": 0}),
            "output_tail": tail,
            **({"vacuous": True} if cmd_res.vacuous else {}),
            **(
                {
                    "side_effects_restored" if restore_tracked else "tracked_side_effects": list(
                        cmd_res.side_effects_restored[:50]
                    )
                }
                if cmd_res.side_effects_restored
                else {}
            ),
            **(
                {"untrusted_side_effects": list(cmd_res.untrusted_side_effects[:50])}
                if cmd_res.untrusted_side_effects
                else {}
            ),
            **(
                {"credential_env_removed": list(cmd_res.credential_env_removed)}
                if cmd_res.credential_env_removed
                else {}
            ),
            **(
                {
                    "effective_command": filtered_cmd,
                    "filtered_missing_paths": list(filtered_missing_paths),
                    **({"diagnostic_counts": dict(meta["diagnostic_counts"])} if "diagnostic_counts" in meta else {}),
                    **({"diagnostic_exit_code": meta["diagnostic_exit_code"]} if "diagnostic_exit_code" in meta else {}),
                    **({"diagnostic_timed_out": True} if meta.get("diagnostic_timed_out") else {}),
                }
                if filtered_cmd and filtered_missing_paths
                else {}
            ),
        }
        results.append(entry)

    q_report = cleanup_summary.get("quarantine") or {}
    if q_report.get("moved") and results:
        results[-1]["quarantined_files"] = {"dir": q_report["quarantine_dir"], "files": q_report["moved"][:20]}
    if q_report.get("left_in_place") and results:
        results[-1]["untracked_left_in_place"] = q_report["left_in_place"][:20]
    created = cleanup_summary.get("created") or []
    if created and results:
        results[-1]["cleaned_artifacts"] = created[:20]
    return results


_PROBE_SKIP_TOP_DIRS = frozenset({
    ".git",
    ".venv",
    ".cc-agy-data",
    ".worktrees",
    ".ctx-session-reads",
    ".cc-agy-review",
})


def _snapshot_probe_worktree_state(cwd: Path) -> tuple[dict[str, bytes | None], set[str]]:
    """Snapshot all untracked files (including small gitignored files) and directory paths in `cwd`."""
    files_snap: dict[str, bytes | None] = {}

    def _record_untracked_file(rel: str) -> None:
        p = cwd / rel
        data: bytes | None = None
        try:
            if not p.is_symlink() and p.is_file() and p.stat().st_size <= 1_048_576:
                data = p.read_bytes()
        except OSError:
            data = None
        files_snap[rel] = data

    res = _git_raw(
        ["ls-files", "--others", "-z"],
        cwd=str(cwd),
        capture_output=True,
        check=False,
    )
    if res.returncode == 0:
        for raw in res.stdout.split(b"\x00"):
            if not raw:
                continue
            rel = raw.decode("utf-8", errors="surrogateescape")
            top = rel.split("/", 1)[0]
            if top not in _PROBE_SKIP_TOP_DIRS:
                _record_untracked_file(rel)
    dirs_set: set[str] = set()
    for cur_root, dirnames, filenames in os.walk(str(cwd)):
        rel_root = os.path.relpath(cur_root, str(cwd))
        if rel_root == ".":
            dirnames[:] = [d for d in dirnames if d not in _PROBE_SKIP_TOP_DIRS]
            rel_prefix = ""
        else:
            rel_prefix = Path(rel_root).as_posix()
            dirs_set.add(rel_prefix)
        if res.returncode != 0:
            for fn in filenames:
                _record_untracked_file(f"{rel_prefix}/{fn}" if rel_prefix else fn)
    return files_snap, dirs_set


def _restore_probe_worktree_state(
    cwd: Path,
    before_files: dict[str, bytes | None] | set[str],
    before_dirs: set[str],
    *,
    reset_tracked: bool = True,
) -> list[str]:
    """Restore clean worktree state after an acceptance probe by reverting tracked/untracked mutations and removing new files/dirs."""
    if reset_tracked:
        _run_git(["checkout", "--", "."], cwd=cwd, check=False)
    after_files, after_dirs = _snapshot_probe_worktree_state(cwd)
    before_keys = set(before_files.keys()) if isinstance(before_files, dict) else set(before_files)
    removed: list[str] = []
    for rel in sorted(set(after_files.keys()) - before_keys):
        p = cwd / rel
        try:
            if p.is_symlink() or p.is_file():
                p.unlink(missing_ok=True)
                removed.append(rel)
        except OSError:
            pass
    if isinstance(before_files, dict):
        for rel, prev_bytes in before_files.items():
            if prev_bytes is None:
                continue
            p = cwd / rel
            try:
                if p.is_symlink():
                    p.unlink(missing_ok=True)
                if not p.is_file() or after_files.get(rel) != prev_bytes:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(prev_bytes)
            except OSError:
                pass
    for rel_dir in sorted(after_dirs - before_dirs, key=lambda d: d.count("/"), reverse=True):
        dpath = cwd / rel_dir
        try:
            if dpath.is_symlink():
                dpath.unlink(missing_ok=True)
            elif dpath.is_dir():
                shutil.rmtree(dpath, ignore_errors=True)
        except OSError:
            pass
    return removed


_COMPILER_CLI_CONFIG_ERROR_RE = re.compile(
    r"(?:"
    r"Syntax:\s+tsc\s+\[options\]"
    r"|Initializes a TypeScript project and creates a tsconfig\.json"
    r"|error TS(?:5023|5024|5042|5058|5083|6053|18002|18003)\b"
    r")",
    re.I,
)


def _extract_short_error_tail(output: str, max_lines: int = 3) -> str:
    """Return a compact 3-line summary of the failure from ANSI-stripped test output."""
    clean = _ANSI_ESCAPE_RE.sub("", output or "")
    non_empty = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    if not non_empty:
        return ""
    priority = [
        ln
        for ln in non_empty
        if re.match(r"^(?:FAILED|ERROR)\b", ln)
        or ln.startswith("E   ")
        or re.search(r"\b(?:[A-Za-z]+Error|Exception|AssertionError):", ln)
        or _COMPILER_CLI_CONFIG_ERROR_RE.search(ln)
    ]
    chosen = priority[-max_lines:] if priority else non_empty[-max_lines:]
    return "\n".join(ln[:180] for ln in chosen)


def _extract_failing_test_nodes_and_files(output: str, cwd: Path) -> tuple[list[str], list[str]]:
    """Extract failing test node IDs (e.g. `tests/sub/test_x.py::test_y`) and existing relative test file paths."""
    clean = _ANSI_ESCAPE_RE.sub("", output or "")
    nodes: list[str] = []
    files: list[str] = []

    def _record_file(cand: str) -> str | None:
        norm = cand.strip().lstrip("./").replace("\\", "/")
        if not norm:
            return None
        p = (cwd / norm).resolve()
        try:
            rel_norm = p.relative_to(cwd.resolve()).as_posix()
        except ValueError:
            return None
        if p.is_file():
            if rel_norm not in files:
                files.append(rel_norm)
            return rel_norm
        return None

    for m in re.finditer(r"(?m)^(?:FAILED|ERROR(?:\s+at\s+\w+\s+of|\s+collecting)?)\s+([^\s]+\.py(?:::[^\s]+)*)", clean):
        raw_node = m.group(1).strip().rstrip(",")
        file_part = raw_node.split("::", 1)[0]
        rel_f = _record_file(file_part)
        if rel_f:
            suffix = raw_node[len(file_part):]
            norm_node = f"{rel_f}{suffix}"
            if norm_node not in nodes:
                nodes.append(norm_node)

    # Traceback location fallback for `pytest -x -q` or custom formatters that omit `FAILED <node>` summary lines
    current_func: str | None = None
    for ln in clean.splitlines():
        m_hdr = re.match(r"^_{3,}\s+([A-Za-z_][\w.\[\]-]*)\s+_{3,}$", ln.strip())
        if m_hdr:
            current_func = m_hdr.group(1)
            continue
        m_loc = re.match(r"^([\w./\\-]+?\.py):(\d+):", ln.strip())
        if m_loc:
            cand_f = m_loc.group(1)
            base_n = Path(cand_f).name
            if base_n.startswith("test_") or base_n.endswith("_test.py"):
                rel_f = _record_file(cand_f)
                if rel_f:
                    fallback_node = f"{rel_f}::{current_func.replace('.', '::')}" if current_func else rel_f
                    if not any(n == fallback_node or n.startswith(f"{rel_f}::") for n in nodes):
                        nodes.append(fallback_node)

    # Standard library unittest header: `FAIL: test_foo (pkg.sub.test_bar.TestCls)`
    for m_ut in re.finditer(r"(?m)^(?:FAIL|ERROR):\s+([A-Za-z_]\w*)\s+\(([\w.]+)\)", clean):
        func_name, qual_cls = m_ut.group(1), m_ut.group(2)
        parts = qual_cls.split(".")
        for split_idx in range(len(parts), 0, -1):
            cand_rel = "/".join(parts[:split_idx]) + ".py"
            rel_f = _record_file(cand_rel)
            if rel_f:
                cls_part = "::".join(parts[split_idx:])
                ut_node = f"{rel_f}::{cls_part}::{func_name}" if cls_part else f"{rel_f}::{func_name}"
                if ut_node not in nodes:
                    nodes.append(ut_node)
                break

    return nodes, files


def _resolve_pytest_runner_for_subdir(runner_prefix: str, root_cwd: Path) -> str:
    """Resolve repo-relative runner executables (e.g. `.venv/bin/pytest`) without dereferencing venv symlinks."""
    try:
        tokens = shlex.split(runner_prefix)
    except ValueError:
        return runner_prefix
    if not tokens:
        return runner_prefix
    changed = False
    for idx, tok in enumerate(tokens):
        if idx > 0 and tokens[idx - 1] != "run":
            if all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t) for t in tokens[:idx]):
                pass
            else:
                continue
        if "/" in tok and not os.path.isabs(tok):
            cand = os.path.abspath(os.path.join(str(root_cwd), tok))
            if os.path.lexists(cand):
                tokens[idx] = cand
                changed = True
    return shlex.join(tokens) if changed else runner_prefix


def _extract_pytest_runner_prefix(cmd: str) -> str | None:
    """If `cmd` is a single simple pytest invocation, return its runner prefix (e.g. `.venv/bin/pytest`)."""
    stripped = cmd.strip()
    if any(sep in stripped for sep in ("&&", "||", ";", "|", "\n")):
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return None
    if not tokens:
        return None
    idx = 0
    while idx < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[idx]):
        idx += 1
    if idx >= len(tokens):
        return None
    head_base = Path(tokens[idx]).name
    if head_base in ("pytest", "py.test"):
        return shlex.join(tokens[: idx + 1])
    if (
        head_base.startswith("python")
        and idx + 2 < len(tokens)
        and tokens[idx + 1] == "-m"
        and tokens[idx + 2] in ("pytest", "py.test")
    ):
        return shlex.join(tokens[: idx + 3])
    if (
        head_base == "uv"
        and idx + 2 < len(tokens)
        and tokens[idx + 1] == "run"
        and Path(tokens[idx + 2]).name in ("pytest", "py.test")
    ):
        return shlex.join(tokens[: idx + 3])
    return None


def _strip_pytest_exitfirst_flags(cmd: str) -> str:
    """Return `cmd` with `-x`, `--exitfirst`, and `--maxfail=...` removed so remaining tests all execute."""
    try:
        tokens = shlex.split(cmd.strip())
    except ValueError:
        return cmd
    cleaned: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in ("-x", "--exitfirst"):
            continue
        if tok == "--maxfail":
            skip_next = True
            continue
        if tok.startswith("--maxfail="):
            continue
        if tok.startswith("-") and not tok.startswith("--") and len(tok) > 2 and "x" in tok[1:]:
            rem = "-" + tok[1:].replace("x", "")
            if rem != "-":
                cleaned.append(rem)
            continue
        cleaned.append(tok)
    return shlex.join(cleaned) if cleaned else cmd


def _run_bounded_probe_cmd(
    cmd: str,
    *,
    cwd: Path,
    exec_env: dict[str, str],
    timeout_s: float,
) -> tuple[int | None, bool, str]:
    from server.verify import execute_check_process

    rc, timed_out, out, _ = execute_check_process(
        cmd,
        cwd,
        exec_env,
        timeout=max(0.1, float(timeout_s)),
        max_tail_chars=20_000,
    )
    return rc, timed_out, (out or "")


_LINTER_OR_STATIC_CHECK_CMD_RE = re.compile(
    r"\b(?:"
    r"go\s+vet"
    r"|golangci-lint"
    r"|staticcheck"
    r"|flake8"
    r"|ruff(?:\s+check)?"
    r"|mypy"
    r"|pyright"
    r"|pylint"
    r"|bandit"
    r"|eslint"
    r"|biome"
    r"|stylelint"
    r"|tsc"
    r"|cargo\s+clippy"
    r"|rubocop"
    r"|shellcheck"
    r")\b",
    re.I,
)

_TEST_RUNNER_CMD_RE = re.compile(
    r"\b(?:"
    r"pytest|py\.test|unittest|go\s+test|jest|vitest|mocha|ava|cargo\s+test"
    r"|npm\s+(?:run\s+)?test|yarn\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|rspec"
    r")\b",
    re.I,
)


def _is_linter_or_static_check_cmd(cmd: str) -> bool:
    if not cmd or _TEST_RUNNER_CMD_RE.search(cmd):
        return False
    return bool(_LINTER_OR_STATIC_CHECK_CMD_RE.search(cmd))


def _extract_linter_diagnostic_files(output: str, cwd: Path) -> list[str]:
    """Extract existing relative file paths from `path:line[:col]` or `path(line,col)` linter/compiler output."""
    clean = _ANSI_ESCAPE_RE.sub("", output or "")
    files: list[str] = []
    cwd_res = cwd.resolve()

    def _add_cand(raw_cand: str) -> None:
        norm = raw_cand.strip().lstrip("./").replace("\\", "/")
        if not norm or norm.startswith("-"):
            return
        try:
            p = (cwd / norm).resolve()
            rel_norm = p.relative_to(cwd_res).as_posix()
        except (ValueError, OSError):
            return
        if p.is_file() and rel_norm not in files:
            files.append(rel_norm)

    for m in re.finditer(r"(?:^|[\s(])([A-Za-z0-9_./\\-]+?\.[A-Za-z0-9]+):(\d+)(?::\d+)?(?::|\b)", clean, re.M):
        _add_cand(m.group(1))
    for m in re.finditer(r"(?:^|[\s])([A-Za-z0-9_./\\-]+?\.[A-Za-z0-9]+)\((\d+),\d+\):", clean, re.M):
        _add_cand(m.group(1))
    return files


def probe_baseline_acceptance(
    cmds: list[str],
    cwd: Path,
    *,
    per_cmd_timeout: float = 15.0,
    total_timeout: float = 25.0,
    reset_tracked: bool = True,
) -> list[dict[str, Any]]:
    """Probe explicit acceptance commands on clean HEAD (`0` model tokens) and restore 100% worktree hygiene."""
    if not cmds or not cwd.is_dir():
        return []
    before_files, before_dirs = _snapshot_probe_worktree_state(cwd)
    exec_env = _build_workdir_exec_env(cwd)
    probe_started = time.monotonic()
    probes: list[dict[str, Any]] = []
    try:
        for cmd in cmds:
            elapsed = time.monotonic() - probe_started
            remaining = total_timeout - elapsed
            if remaining < 0.2:
                break
            cmd_limit = min(per_cmd_timeout, remaining)
            rc, timed_out, out = _run_bounded_probe_cmd(
                cmd, cwd=cwd, exec_env=exec_env, timeout_s=cmd_limit
            )
            _restore_probe_worktree_state(cwd, before_files, before_dirs, reset_tracked=reset_tracked)
            if timed_out:
                probes.append({"command": cmd, "exit_code": None, "timed_out": True, "passed": False})
                continue
            counts = _parse_test_counts(out)
            if rc == 0:
                probes.append({"command": cmd, "exit_code": 0, "timed_out": False, "passed": True, "counts": counts})
                continue
            nodes, files = _extract_failing_test_nodes_and_files(out, cwd)
            if not files:
                files = _extract_linter_diagnostic_files(out, cwd)
            short_tail = _extract_short_error_tail(out, max_lines=3)
            clean_probe_out = _ANSI_ESCAPE_RE.sub("", out or "")
            is_cli_cfg_err = bool(
                _COMPILER_CLI_CONFIG_ERROR_RE.search(clean_probe_out)
                and not re.search(r"\berror TS(?:1|2|4|7)\d{3}\b", clean_probe_out)
            )
            entry: dict[str, Any] = {
                "command": cmd,
                "exit_code": rc,
                "timed_out": False,
                "passed": False,
                "counts": counts,
                "error_tail": short_tail,
                "failing_nodes": nodes[:20],
                "failing_files": files[:10],
                **({"compiler_cli_config_error": True} if is_cli_cfg_err else {}),
            }
            pytest_runner = _extract_pytest_runner_prefix(cmd)
            if pytest_runner and files:
                cand_dirs = list(
                    dict.fromkeys(
                        str(Path(f).parent.as_posix())
                        for f in files
                        if str(Path(f).parent.as_posix()) not in ("", ".")
                    )
                )
                rem_after = total_timeout - (time.monotonic() - probe_started)
                if len(cand_dirs) == 1 and rem_after >= 0.4:
                    sub_rel = cand_dirs[0]
                    sub_dir = cwd / sub_rel
                    if sub_dir.is_dir():
                        sub_runner = _resolve_pytest_runner_for_subdir(pytest_runner, cwd)
                        sub_rc, sub_to, sub_out = _run_bounded_probe_cmd(
                            f"{sub_runner} -q",
                            cwd=sub_dir,
                            exec_env=exec_env,
                            timeout_s=min(10.0, rem_after),
                        )
                        _restore_probe_worktree_state(cwd, before_files, before_dirs, reset_tracked=reset_tracked)
                        sub_counts = _parse_test_counts(sub_out)
                        if (
                            not sub_to
                            and sub_rc == 0
                            and int(sub_counts.get("passed") or 0) >= 1
                            and int(sub_counts.get("failed") or 0) == 0
                            and int(sub_counts.get("errors") or 0) == 0
                        ):
                            entry["subdir_cwd_passed"] = True
                            entry["passing_cwd"] = sub_rel
                            entry["subdir_counts"] = sub_counts
            probes.append(entry)
    finally:
        _restore_probe_worktree_state(cwd, before_files, before_dirs, reset_tracked=reset_tracked)
    return probes


_PYTEST_NOT_FOUND_RE = re.compile(r"file or directory not found:\s*([^\s'\"`]+)", re.I)


def remap_nonexistent_pytest_acceptance_cmds(
    cmds: list[str],
    cwd: Path,
    probes: list[dict[str, Any]] | None,
    *,
    allowed_paths: list[str] | None = None,
    prompt_text: str = "",
    remap_details: list[dict[str, str]] | None = None,
) -> tuple[list[str], bool]:
    """Rewrite nonexistent `exit_code == 4` pytest target paths to unique tracked basename matches."""
    if not cmds or not probes or not cwd.is_dir():
        return list(cmds), False
    probe_by_cmd = {
        str(p.get("command") or "").strip(): p
        for p in probes
        if isinstance(p, dict) and not p.get("passed") and p.get("exit_code") == 4
    }
    if not probe_by_cmd:
        return list(cmds), False
    norm_prompt = prompt_text.replace("\\", "/") if prompt_text else ""
    tracked: list[str] | None = None
    remapped_any = False
    updated_cmds: list[str] = []
    for cmd in cmds:
        pr = probe_by_cmd.get(cmd.strip())
        if not pr or "pytest" not in cmd:
            updated_cmds.append(cmd)
            continue
        tail = str(pr.get("error_tail") or "")
        missing_tokens = list(dict.fromkeys(_PYTEST_NOT_FOUND_RE.findall(tail)))
        if not missing_tokens:
            try:
                toks = shlex.split(cmd)
            except ValueError:
                toks = cmd.split()
            for tok in toks:
                if tok.startswith("-"):
                    continue
                base_tok = tok.split("::", 1)[0].strip()
                if base_tok.endswith(".py") and not (cwd / base_tok).exists():
                    missing_tokens.append(tok)
        new_cmd = cmd
        for raw_missing in missing_tokens:
            file_part, sep, node_part = raw_missing.partition("::")
            clean_missing = file_part.replace("\\", "/").strip("/")
            if not clean_missing or (cwd / clean_missing).exists():
                continue
            if allowed_paths and _matches_allowed_paths(clean_missing, allowed_paths):
                continue
            target_bname = Path(clean_missing).name
            if not target_bname:
                continue
            if norm_prompt:
                if re.search(
                    r"(?<![A-Za-z0-9_./-])" + re.escape(clean_missing) + r"(?![A-Za-z0-9_./-])",
                    norm_prompt,
                ):
                    continue
                if re.search(
                    r"\b(?:create|add|write|new)\b[^.\n]{0,120}\b" + re.escape(target_bname) + r"\b",
                    norm_prompt,
                    re.I,
                ):
                    continue
            if tracked is None:
                tracked = _list_tracked_repo_files(cwd)
            try:
                cur_toks = shlex.split(new_cmd)
            except ValueError:
                cur_toks = new_cmd.split()
            existing_cmd_targets = {
                tok.split("::", 1)[0].replace("\\", "/").strip("/")
                for tok in cur_toks
                if not tok.startswith("-") and tok.split("::", 1)[0].endswith(".py")
            }
            matches = [
                tf
                for tf in tracked
                if Path(tf).name == target_bname
                and (cwd / tf).is_file()
                and tf not in existing_cmd_targets
            ]
            chosen_tf: str | None = None
            if len(matches) == 1:
                chosen_tf = matches[0]
            elif len(matches) > 1:
                miss_parts = list(reversed(Path(clean_missing).parts[:-1]))
                ranked: list[tuple[int, int, str]] = []
                for tf in matches:
                    tf_parts = list(reversed(Path(tf).parts[:-1]))
                    shared = 0
                    for a_p, b_p in zip(miss_parts, tf_parts):
                        if a_p == b_p:
                            shared += 1
                        else:
                            break
                    ranked.append((-shared, len(Path(tf).parts), tf))
                ranked.sort()
                if len(ranked) >= 2 and ranked[0][0] < ranked[1][0]:
                    chosen_tf = ranked[0][2]
            if chosen_tf:
                replacement = f"{chosen_tf}{sep}{node_part}" if sep else chosen_tf
                if raw_missing in new_cmd:
                    prev_cmd = new_cmd
                    new_cmd = new_cmd.replace(raw_missing, replacement, 1)
                    remapped_any = True
                    if remap_details is not None:
                        remap_details.append(
                            {
                                "original_command": prev_cmd,
                                "remapped_command": new_cmd,
                                "missing_target": clean_missing,
                                "remapped_target": chosen_tf,
                            }
                        )
        updated_cmds.append(new_cmd)
    return updated_cmds, remapped_any


def evaluate_post_worker_baseline_failures(
    acceptance_results: list[dict[str, Any]],
    baseline_probe: list[dict[str, Any]] | None,
    cwd: Path,
    files_changed: list[str],
    *,
    timeout: float = 30.0,
) -> list[str]:
    """Account for pre-existing clean-HEAD `cwd` or untouched baseline test/linter failures without hiding regressions."""
    from server.verify import check_side_effects, snapshot_workspace

    if not acceptance_results or not files_changed or not cwd.is_dir():
        return []
    probe_by_cmd = {
        str(p.get("command") or "").strip(): p
        for p in (baseline_probe or [])
        if isinstance(p, dict) and not p.get("passed") and not p.get("timed_out") and p.get("exit_code") not in (None, 0)
    }
    any_real_test_passed = any(
        is_non_smoke_passing_entry(x)
        and (
            int((x.get("counts") or {}).get("passed") or 0) > 0
            or bool(_TEST_RUNNER_CMD_RE.search(str(x.get("command") or "")))
        )
        for x in acceptance_results
    )
    changed_norm = {
        str(f).strip().lstrip("./").replace("\\", "/")
        for f in files_changed
        if f
    }

    exec_env = _build_workdir_exec_env(cwd)
    warnings: list[str] = []

    for r in acceptance_results:
        if not isinstance(r, dict) or r.get("passed") or r.get("timed_out"):
            continue
        cmd = str(r.get("command") or "").strip()
        probe = probe_by_cmd.get(cmd)
        post_tail = str(r.get("output_tail") or "")
        r_counts = r.get("counts") or {}
        has_zero_test_counts = (
            int(r_counts.get("passed") or 0) == 0
            and int(r_counts.get("failed") or 0) == 0
            and int(r_counts.get("errors") or 0) == 0
        )

        # Case 2b ([R5-2]): Untouched baseline linter / static-check failure (e.g. `go vet ./pkg/...`,
        # `golangci-lint`, `flake8`, `ruff`, `mypy`, `eslint`) where the command ALSO failed on clean HEAD
        # (`probe is not None`), every `path:line` diagnostic is a subset of the clean-HEAD diagnostics and
        # outside `files_changed`, and at least one real test-runner command passed.
        if (
            probe is not None
            and any_real_test_passed
            and has_zero_test_counts
            and _is_linter_or_static_check_cmd(cmd)
            and r.get("exit_code") not in (None, 0)
        ):
            diag_files = _extract_linter_diagnostic_files(post_tail, cwd)
            probe_diag_files = [
                str(pf).strip().lstrip("./").replace("\\", "/")
                for pf in (probe.get("failing_files") or [])
                if pf
            ]
            probe_err_tail = str(probe.get("error_tail") or "").strip()
            matches_clean_head_diag = (
                set(diag_files) <= set(probe_diag_files)
                if probe_diag_files
                else (bool(probe_err_tail) and all(df in probe_err_tail for df in diag_files))
            )
            untouched_diag = (
                bool(diag_files)
                and matches_clean_head_diag
                and all(df not in changed_norm for df in diag_files)
            )
            identical_clean_head_err = bool(
                not diag_files
                and probe.get("exit_code") == r.get("exit_code")
                and probe_err_tail
                and probe_err_tail in post_tail
            )
            if untouched_diag or identical_clean_head_err:
                r["original_exit_code"] = r.get("exit_code")
                r["exit_code"] = 0
                r["passed"] = True
                r["baseline_neutralized"] = True
                r["preexisting_baseline_failure_ignored"] = True
                warnings.append(
                    f"Untouched baseline linter/static-check failure in `{cmd}` neutralized: "
                    f"diagnostic file(s) {diag_files or ['<clean-HEAD match>']} are outside `files_changed` "
                    "and matched clean-HEAD baseline diagnostics while test-runner acceptance passed."
                )
                continue

        if not probe:
            continue
        post_nodes, post_files = _extract_failing_test_nodes_and_files(post_tail, cwd)
        pytest_runner = _extract_pytest_runner_prefix(cmd)

        # Case 1: Pre-existing clean-HEAD subdir `cwd` quirk (passed on HEAD when run from `passing_cwd`).
        if probe.get("subdir_cwd_passed") and probe.get("passing_cwd") and pytest_runner:
            passing_cwd = str(probe["passing_cwd"])
            sub_dir = cwd / passing_cwd
            # Root run must only have failed inside `passing_cwd` (not on a new root test before it).
            if sub_dir.is_dir() and post_files and all(
                f == passing_cwd or f.startswith(f"{passing_cwd}/") for f in post_files
            ):
                before_files, before_dirs = _snapshot_probe_worktree_state(cwd)
                # Post-worker probes run on top of the worker's edits, so tracked files cannot be reset
                # wholesale; snapshot the change set instead and undo whatever the probe touched.
                probe_snapshot = snapshot_workspace(cwd)
                try:
                    sub_runner = _resolve_pytest_runner_for_subdir(pytest_runner, cwd)
                    sub_rc, sub_to, sub_out = _run_bounded_probe_cmd(
                        f"{sub_runner} -q",
                        cwd=sub_dir,
                        exec_env=exec_env,
                        timeout_s=min(20.0, timeout),
                    )
                finally:
                    _restore_probe_worktree_state(cwd, before_files, before_dirs, reset_tracked=False)
                    check_side_effects(probe_snapshot)
                sub_counts = _parse_test_counts(sub_out)
                sub_ok = (
                    not sub_to
                    and sub_rc == 0
                    and int(sub_counts.get("passed") or 0) >= 1
                    and int(sub_counts.get("failed") or 0) == 0
                    and int(sub_counts.get("errors") or 0) == 0
                )
                if sub_ok:
                    rest_cmd = f"{_strip_pytest_exitfirst_flags(cmd)} --ignore={shlex.quote(passing_cwd)}"
                    before_files2, before_dirs2 = _snapshot_probe_worktree_state(cwd)
                    probe_snapshot2 = snapshot_workspace(cwd)
                    try:
                        rest_rc, rest_to, rest_out = _run_bounded_probe_cmd(
                            rest_cmd,
                            cwd=cwd,
                            exec_env=exec_env,
                            timeout_s=min(30.0, timeout),
                        )
                    finally:
                        _restore_probe_worktree_state(cwd, before_files2, before_dirs2, reset_tracked=False)
                        check_side_effects(probe_snapshot2)
                    rest_counts = _parse_test_counts(rest_out)
                    rest_ok = (
                        not rest_to
                        and rest_rc in (0, 5)
                        and int(rest_counts.get("failed") or 0) == 0
                        and int(rest_counts.get("errors") or 0) == 0
                    )
                    if rest_ok:
                        total_passed = int(sub_counts.get("passed") or 0) + int(rest_counts.get("passed") or 0)
                        r["original_exit_code"] = r.get("exit_code")
                        r["exit_code"] = 0
                        r["passed"] = True
                        r["preexisting_baseline_failure_ignored"] = True
                        r["passing_cwd"] = passing_cwd
                        r["counts"] = {"passed": total_passed, "failed": 0, "errors": 0}
                        warnings.append(
                            f"Baseline `cwd` failure in `{cmd}` ignored: `{passing_cwd}` already failed from repo "
                            f"root on clean HEAD and passed from `{passing_cwd}` ({sub_counts.get('passed', 0)} "
                            f"passed in `{passing_cwd}`, {rest_counts.get('passed', 0)} passed in remaining suite)."
                        )
                        continue

        # Case 2: Pre-existing clean-HEAD compiler CLI config/help error (e.g. `tsc --noEmit` without root
        # tsconfig.json) alongside at least one passing test-runner command: ignore so real passing tests
        # are not blocked by a missing root compiler config.
        if (
            _is_compiler_cli_config_or_help_failure(r)
            and _is_compiler_cli_config_or_help_failure(probe)
            and any_real_test_passed
        ):
            r["original_exit_code"] = r.get("exit_code")
            r["exit_code"] = 0
            r["passed"] = True
            r["preexisting_baseline_failure_ignored"] = True
            warnings.append(
                f"Baseline compiler CLI config failure in `{cmd}` ignored: `{cmd}` already failed on clean HEAD "
                "without test failures while test-runner acceptance passed."
            )
            continue

        # Case 3: Non-subdir clean-HEAD failures remain failing: NEVER flip `passed` to True
        # (an unfixed TDD acceptance test must never auto-apply even if the worker also added
        # unrelated passing tests). Record an informational diagnostic for the Lead only.
        base_nodes = [str(n) for n in (probe.get("failing_nodes") or []) if n]
        if base_nodes and post_nodes and set(post_nodes) <= set(base_nodes):
            warnings.append(
                f"Diagnostic: failing node(s) {post_nodes} in `{cmd}` also failed on clean HEAD "
                f"(acceptance remains failed so unfixed TDD targets are never auto-applied)."
            )

    return warnings


def _is_compiler_cli_config_or_help_failure(res: dict[str, Any]) -> bool:
    if not isinstance(res, dict) or res.get("passed") or res.get("timed_out"):
        return False
    counts = res.get("counts") or {}
    if (
        int(counts.get("passed") or 0) > 0
        or int(counts.get("failed") or 0) > 0
        or int(counts.get("errors") or 0) > 0
    ):
        return False
    tail = str(res.get("output_tail") or res.get("error_tail") or "")
    if re.search(r"\berror TS(?:1|2|4|7)\d{3}\b", tail):
        return False
    return bool(res.get("compiler_cli_config_error")) or bool(_COMPILER_CLI_CONFIG_ERROR_RE.search(tail))


def _only_compiler_cli_config_failures(acc_results: list[dict[str, Any]]) -> bool:
    if not acc_results:
        return False
    has_passing_test_runner = any(
        is_non_smoke_passing_entry(r)
        and int(r.get("exit_code") if r.get("exit_code") is not None else 0) == 0
        and int((r.get("counts") or {}).get("passed") or 0) > 0
        for r in acc_results
    )
    if not has_passing_test_runner:
        return False
    failing = [r for r in acc_results if isinstance(r, dict) and not r.get("passed")]
    return bool(failing) and all(_is_compiler_cli_config_or_help_failure(r) for r in failing)

