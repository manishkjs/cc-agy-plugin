"""Isolated job manager for Antigravity SDK delegations."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import fcntl
import fnmatch
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable
import uuid

from server.agy_runner import (
    is_read_only_tool_set,
    is_straitjacket_available,
    read_progress,
    resolve_thinking_level,
    run_antigravity_worker,
)
from server.repo_map import build_focused_file_outline, build_repo_map
from server.receipts import (
    ReceiptStore,
    extract_diff_provenance,
    parse_diff_git_header,
    parse_unified_header_path,
    resolve_workspace_root,
    sha256_bytes,
    validate_job_id,
)
from server.regions import (
    DEFAULT_DIFF_PAGE_TOKENS,
    MAX_TOOL_RESPONSE_TOKENS,
    estimate_tokens,
    first_valid_env,
    paginate_text_by_tokens,
    resolve_api_key,
    resolve_auth_config,
    truncate_to_token_budget,
)


def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _strip_dot_slash(path_str: str) -> str:
    """Strip leading `./` prefixes without stripping `../`."""
    s = path_str.strip()
    while s.startswith("./"):
        s = s[2:]
    return s


def validate_allowed_paths(allowed_paths: Any) -> list[str]:
    """Validate allowed_paths parameter, rejecting non-lists, parent traversal, or empty/root wildcards."""
    if allowed_paths is None:
        return []
    if not isinstance(allowed_paths, list):
        raise ValueError(
            f"allowed_paths must be a list of relative path strings, got {type(allowed_paths).__name__}"
        )
    cleaned: list[str] = []
    for item in allowed_paths:
        if not isinstance(item, str):
            raise ValueError(f"allowed_paths entries must be strings, got {type(item).__name__}")
        stripped = item.strip()
        norm = _strip_dot_slash(stripped).rstrip("/")
        if (
            not norm
            or norm in (".", "..")
            or stripped.startswith("/")
            or ".." in Path(norm).parts
        ):
            raise ValueError(
                f"Invalid allowed_paths entry {item!r}: must be a non-empty workspace-relative path without '..'"
            )
        cleaned.append(norm)
    return cleaned


def _matches_allowed_paths(rel_path: str, allowed_paths: list[str] | str | None) -> bool:
    """Return True if rel_path is inside or matches any entry in allowed_paths."""
    if allowed_paths is None:
        return True
    if isinstance(allowed_paths, str):
        raw_list = [allowed_paths]
    elif isinstance(allowed_paths, list):
        raw_list = allowed_paths
    else:
        return False
    if not raw_list:
        return True
    try:
        validated = validate_allowed_paths(raw_list)
    except ValueError:
        return False
    if not validated:
        return True
    stripped = rel_path.strip()
    if stripped.startswith("/"):
        return False
    norm_path = _strip_dot_slash(stripped)
    if not norm_path or norm_path in (".", "..") or ".." in Path(norm_path).parts:
        return False
    for norm_pat in validated:
        if norm_pat == "*":
            return True
        if norm_path == norm_pat or norm_path.startswith(norm_pat + "/"):
            return True
        if fnmatch.fnmatch(norm_path, norm_pat):
            return True
    return False


def _matches_file_filter(rel_path: str, file_filter: str) -> bool:
    """Match a diff file path against file_filter without substring false positives (a.py vs data.py)."""
    norm_path = _strip_dot_slash(rel_path.strip())
    norm_filter = _strip_dot_slash(file_filter.strip()).rstrip("/")
    if not norm_filter:
        return True
    if norm_path == norm_filter or norm_path.startswith(norm_filter + "/"):
        return True
    if Path(norm_path).name == norm_filter:
        return True
    if fnmatch.fnmatch(norm_path, norm_filter):
        return True
    # Also check if norm_filter matches a path suffix at a directory boundary
    if norm_path.endswith("/" + norm_filter):
        return True
    return False


def capture_live_workspace_diff(
    workspace_root: Path,
    *,
    baseline_head_sha: str | None = None,
) -> str:
    """Capture the unified diff of workspace_root against baseline_head_sha (or HEAD) including untracked files.

    Uses a temporary GIT_INDEX_FILE copy so the host repository's real git index is
    never mutated.
    """
    git_dir_res = _run_git(["rev-parse", "--git-dir"], cwd=workspace_root, check=False)
    if git_dir_res.returncode != 0:
        return ""
    raw_git_dir = Path(git_dir_res.stdout.strip())
    git_dir = raw_git_dir if raw_git_dir.is_absolute() else (workspace_root / raw_git_dir).resolve()
    real_index = git_dir / "index"

    target_ref = "HEAD"
    if baseline_head_sha:
        verify_ref = _run_git(["rev-parse", "--verify", f"{baseline_head_sha}^{{commit}}"], cwd=workspace_root, check=False)
        if verify_ref.returncode == 0:
            target_ref = baseline_head_sha

    with tempfile.NamedTemporaryFile(prefix="cc-agy-idx-", delete=False) as tmp_idx:
        tmp_index_path = Path(tmp_idx.name)
    try:
        if real_index.exists():
            shutil.copy2(real_index, tmp_index_path)
        else:
            tmp_index_path.unlink(missing_ok=True)
        env = dict(os.environ)
        env["GIT_INDEX_FILE"] = str(tmp_index_path)
        subprocess.run(
            ["git", "-c", "core.quotePath=false", "add", "-N", "."],
            cwd=str(workspace_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        diff_res = subprocess.run(
            ["git", "-c", "core.quotePath=false", "diff", "--binary", "--no-renames", target_ref],
            cwd=str(workspace_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return diff_res.stdout if diff_res.returncode == 0 else ""
    finally:
        tmp_index_path.unlink(missing_ok=True)


def _snapshot_directory_tree(root: Path) -> dict[str, bytes]:
    """Capture a byte snapshot of files inside `root` (used to protect host .git/hooks and .git/config)."""
    snap: dict[str, bytes] = {}
    if not root.exists():
        return snap
    if root.is_file():
        try:
            snap[""] = root.read_bytes()
        except OSError:
            pass
        return snap
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            rel = str(p.relative_to(root))
            try:
                snap[rel] = p.read_bytes()
            except OSError:
                pass
    return snap


def _restore_directory_tree(root: Path, snap: dict[str, bytes]) -> bool:
    """Verify `root` matches `snap`; if tampered, restore `root` to `snap` and return True."""
    current = _snapshot_directory_tree(root)
    if current == snap:
        return False
    if root.is_file() or ("" in snap):
        if "" in snap:
            root.parent.mkdir(parents=True, exist_ok=True)
            root.write_bytes(snap[""])
        elif root.exists():
            root.unlink(missing_ok=True)
        return True

    if root.exists():
        for p in list(root.rglob("*")):
            if p.is_file() or p.is_symlink():
                rel = str(p.relative_to(root))
                if rel not in snap:
                    p.unlink(missing_ok=True)
    for rel, content in snap.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return True


_EPHEMERAL_CACHE_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".hypothesis",
        ".benchmarks",
        ".mypy_cache",
        ".ruff_cache",
        ".pytype",
        ".eggs",
        "node_modules",
        ".tox",
        ".nox",
        ".venv",
        ".gradle",
        ".next",
        ".nuxt",
        ".ctx-session-reads",
        ".cc-agy-review",
        ".cc-agy-data",
        ".worktrees",
    }
)

_EPHEMERAL_CACHE_SUFFIXES: tuple[str, ...] = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".egg-info",
    ".coverage",
    ".DS_Store",
)

_IGNORED_BUILD_OUTPUT_DIRS: frozenset[str] = frozenset(
    {
        "build",
        "dist",
        "target",
        "out",
        "coverage",
        "venv",
        ".cache",
        ".parcel-cache",
        ".turbo",
        ".svelte-kit",
        "htmlcov",
    }
)

_SAFE_NOTE_EXTENSIONS: tuple[str, ...] = (
    ".md",
    ".txt",
    ".rst",
    ".org",
    ".adoc",
    ".todo",
)


def _is_ephemeral_build_or_test_artifact(rel_path: str, root: Path | None = None) -> bool:
    """Return True if `rel_path` is a transient test/compiler cache artifact that should never be captured or applied."""
    norm = rel_path.strip("/")
    if not norm:
        return False
    parts = norm.split("/")
    for part in parts:
        if part in _EPHEMERAL_CACHE_DIRS or part.endswith(".egg-info"):
            return True
    name = parts[-1]
    if (
        name in (".coverage", ".DS_Store")
        or name.startswith(".coverage.")
        or name.endswith(_EPHEMERAL_CACHE_SUFFIXES)
    ):
        return True
    if root is not None:
        candidate = root / norm
        if candidate.is_file() and not candidate.is_symlink():
            try:
                if candidate.stat().st_size <= 16384:
                    with open(candidate, "rb") as fh:
                        head = fh.read(128)
                    if head.startswith(b"# TLS secrets log file, generated by OpenSSL"):
                        return True
            except OSError:
                pass
    return False


def _is_ignored_build_artifact(rel_path: str, root: Path | None = None) -> bool:
    """Return True if an already-.gitignored path is inside a build/output/cache directory."""
    if _is_ephemeral_build_or_test_artifact(rel_path, root=root):
        return True
    norm = rel_path.strip("/")
    if not norm:
        return False
    for part in norm.split("/"):
        if part in _IGNORED_BUILD_OUTPUT_DIRS:
            return True
    return False


DEFAULT_MAX_TOOL_CALLS = 100
DEFAULT_MAX_TOTAL_TOKENS = 5_000_000
DEFAULT_ACCEPTANCE_TIMEOUT = 300.0
MAX_ACCEPTANCE_COMMANDS = 5
MAX_PLAN_STEPS = 15
ACCEPTANCE_OUTPUT_TOKENS = 600
CHECKPOINT_STOP_REASONS = (
    "MAX_TOOL_CALLS_EXCEEDED",
    "MAX_TOTAL_TOKENS_EXCEEDED",
    "MAX_INPUT_TOKENS_EXCEEDED",
    "MAX_OUTPUT_TOKENS_EXCEEDED",
    "MAX_MODEL_CALLS_EXCEEDED",
)


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    if not raw or raw.startswith("${"):
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def resolve_budget(
    max_tool_calls: int | None, max_total_tokens: int | None
) -> tuple[int | None, int | None]:
    """Per-turn worker caps (defaults 100 tool calls / 5M tokens; env overrides; 0 disables a cap)."""
    calls = max_tool_calls if max_tool_calls is not None else _env_int("CC_AGY_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS)
    tokens = (
        max_total_tokens
        if max_total_tokens is not None
        else _env_int("CC_AGY_MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS)
    )
    if int(calls) < 0 or int(tokens) < 0:
        raise ValueError("max_tool_calls and max_total_tokens must be >= 0 (0 disables the cap)")
    return (int(calls) or None), (int(tokens) or None)


def validate_plan(plan: Any) -> list[str]:
    if plan is None:
        return []
    if not isinstance(plan, list) or not all(isinstance(s, str) for s in plan):
        raise ValueError("plan must be a list of step strings")
    steps = [s.strip() for s in plan if s.strip()]
    if len(steps) > MAX_PLAN_STEPS:
        raise ValueError(f"plan has {len(steps)} steps; keep it to at most {MAX_PLAN_STEPS}")
    return steps


def validate_acceptance(acceptance: Any, workdir: Path) -> list[str]:
    """Acceptance commands must be read-only checks (tests, linters); mutating commands are refused.

    The plugin itself runs them, so a mutating command would be an unguarded way to edit the repo.
    """
    if acceptance is None:
        return []
    if not isinstance(acceptance, list) or not all(isinstance(c, str) for c in acceptance):
        raise ValueError("acceptance must be a list of shell command strings")
    cmds = [c.strip() for c in acceptance if c.strip()]
    if len(cmds) > MAX_ACCEPTANCE_COMMANDS:
        raise ValueError(f"acceptance has {len(cmds)} commands; at most {MAX_ACCEPTANCE_COMMANDS} allowed")
    from scripts.pre_tool_guard import detect_bash_workspace_mutation

    for cmd in cmds:
        verdict = detect_bash_workspace_mutation(cmd, workdir, cwd=workdir)
        if verdict.get("blocked"):
            raise ValueError(
                f"acceptance command {cmd!r} is refused because it can modify the workspace: "
                f"{verdict.get('reason')}"
            )
    return cmds


_PRELOAD_SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".pdf",
    ".whl", ".tar", ".gz", ".zip", ".pyc", ".so", ".dylib", ".lock",
}

# Reviewer context cap for qa_review_model diff audits.
QA_REVIEW_MAX_TOKENS = 25_000


def escalate_after_failures() -> int:
    """Consecutive failed worker turns before the job is handed back to the Lead (default 2)."""
    raw = os.environ.get("CC_AGY_ESCALATE_AFTER_FAILURES", "2").strip()
    try:
        val = int(raw)
    except ValueError:
        val = 2
    return max(0, val)


def worker_turn_failed(status: str | None, acceptance_passed: bool | None) -> bool:
    """A turn fails when its acceptance checks fail or the worker errors or times out."""
    return acceptance_passed is False or status in ("error", "timeout")

QA_DIFF_AUDIT_SYSTEM = """You are a senior software engineer performing a 1-shot code review of a git diff against a task specification.
Check strictly for:
1. Missing requirements or edge cases explicitly stated in the task specification.
2. Subtle logic bugs, off-by-one errors, state leaks, or broken imports/exports.
3. Missing or weak unit test coverage for the required behavior.

Output format:
- If the diff is complete and sound, start with `VERDICT: PASS` on line 1.
- If there is at least one concrete bug or missing requirement, start with `VERDICT: NEEDS_FIXES` on line 1, followed by concise bullet points (`- ISSUE: <file:line>: <exact bug and how to fix it>`).
Do NOT nitpick formatting or style."""


def parse_qa_audit_verdict(text: str) -> dict[str, Any]:
    """Parse a 1-shot QA review response into verdict, passed boolean, and issue bullets."""
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    # Reviewers sometimes add a preamble ("Reviewed the diff.", "Format reminder: VERDICT: PASS means ...")
    # before the real verdict, so prefer the LAST line that starts with a VERDICT marker (markdown
    # emphasis allowed), then fall back to any line mentioning it, then line 0.
    starts = [i for i, ln in enumerate(lines) if re.match(r"^[\s*#>_`]*VERDICT:", ln, re.IGNORECASE)]
    mentions = [i for i, ln in enumerate(lines) if "VERDICT:" in ln.upper()]
    verdict_idx = starts[-1] if starts else (mentions[0] if mentions else 0)
    first = lines[verdict_idx].upper() if lines else ""
    passed = "VERDICT: PASS" in first and "NEEDS_FIXES" not in first and "FAIL" not in first
    rest = lines[verdict_idx + 1:]
    issues = [
        ln
        for ln in rest
        if re.match(r"^(?:[-*•]|\d+[.)])\s+", ln) or "ISSUE:" in ln.upper()
    ]
    # Fail closed: an explicit ISSUE: line anywhere contradicts a PASS verdict.
    if passed and any("ISSUE:" in ln.upper() for ln in lines):
        passed = False
    if not passed and not issues and len(rest) > 0:
        issues = [ln for ln in rest if not ln.upper().startswith("VERDICT:")]
    return {
        "verdict": "PASS" if passed else "NEEDS_FIXES",
        "passed": passed,
        "issues": issues,
    }


def select_candidate_files(
    workspace_root: Path,
    task_prompt: str,
    *,
    max_files: int = 12,
    token_budget: int = 120_000,
) -> list[tuple[str, str]]:
    """Rank tracked repository files by keyword overlap with task_prompt and return (rel_path, content) pairs."""
    res = subprocess.run(
        ["git", "ls-files"],
        cwd=str(workspace_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    all_files = [
        f.strip()
        for f in res.stdout.splitlines()
        if f.strip() and not _is_ephemeral_build_or_test_artifact(f.strip(), root=workspace_root)
    ]
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", task_prompt)
    keywords = {t.lower().strip("._/-") for t in raw_tokens if len(t) >= 3}
    prompt_low = task_prompt.lower()

    scored: list[tuple[float, str, str]] = []
    for rel in all_files[:5000]:
        p = workspace_root / rel
        if p.suffix.lower() in _PRELOAD_SKIP_EXTS or not p.is_file() or p.is_symlink():
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz == 0 or sz > 400_000:
            continue
        rel_low = rel.lower()
        score = 0.0
        if rel_low in prompt_low:
            score += 50.0
        for kw in keywords:
            if kw and kw in rel_low:
                score += 6.0
        try:
            with p.open("rb") as fh:
                sample_txt = fh.read(65_536).decode("utf-8", errors="replace")
        except OSError:
            continue
        txt_low = sample_txt.lower()
        for kw in keywords:
            if kw and kw in txt_low:
                score += 1.0
        if not rel_low.startswith(("docs/", "examples/", ".github/")):
            score += 0.5
        if score > 0:
            scored.append((score, rel, sample_txt))

    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen: list[tuple[str, str]] = []
    used_tokens = 0
    for _score, rel, sample_txt in scored[: max_files * 2]:
        if len(chosen) >= max_files:
            break
        est = estimate_tokens(sample_txt)
        if used_tokens + est > token_budget:
            continue
        chosen.append((rel, sample_txt))
        used_tokens += est
    return chosen


def render_worker_prompt(
    brief: str,
    *,
    plan: list[str],
    acceptance: list[str],
    max_tool_calls: int | None,
    max_total_tokens: int | None,
    progress_channel: bool = True,
    preloaded_files: list[tuple[str, str]] | None = None,
    repo_map: str = "",
    focused_outline: str = "",
) -> str:
    """Wrap the lead's brief with its plan, acceptance checks, budget, and reporting rules."""
    parts = [brief.strip()]
    if repo_map and repo_map.strip():
        parts.append("## Repository Map (Cached)\n" + repo_map.strip())
    if focused_outline and focused_outline.strip():
        parts.append("## Focused File Outline (Line-Numbered)\n" + focused_outline.strip())
    if preloaded_files:
        # Path hints only: full file contents pasted into turn 1 get re-read on every later turn.
        path_lines = [f"- `{rel}`" for rel, _content in preloaded_files]
        parts.append(
            "## Candidate Repository Files\n"
            "These files matched the task's identifiers. Open only the relevant sections:\n"
            + "\n".join(path_lines)
        )
    if plan:
        parts.append("## Plan from the lead\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1)))
    if acceptance:
        parts.append(
            "## Acceptance\nWhen you finish, the plugin runs these commands and reports the results to the lead. "
            "Make them pass:\n"
            + "\n".join(f"- `{c}`" for c in acceptance)
            + "\nIMPORTANT: Create or update all requested implementation and test files FIRST, and verify your "
            "new/changed test files directly. Once the Acceptance command(s) above pass and any temporary scratch "
            "files are removed, stop and report immediately; do NOT run unrequested full-repository test suites or "
            "linters (e.g. ruff, mypy, flake8, eslint) that are not listed above. Never modify unrelated files to "
            "fix pre-existing test failures outside your task scope or allowed_paths."
        )
    budget_bits = []
    if max_tool_calls:
        budget_bits.append(f"{max_tool_calls} tool calls")
    if max_total_tokens:
        budget_bits.append(f"{max_total_tokens:,} total tokens")
    if budget_bits:
        parts.append(
            "## Budget\nThis turn stops automatically after " + " or ".join(budget_bits) + ". Work efficiently: "
            "make your first file edit within 8-10 tool calls and implement changes incrementally file-by-file "
            "so your edits are saved on disk early (never read more than 6-8 files before editing, and NEVER draft "
            "full file implementations or long code blocks inside internal thinking; emit create_file/edit_file "
            "tool calls directly and iterate against test output), read files <=500 lines in a single view_file "
            "call (start_line=1, end_line=500) or use the Focused File Outline line numbers to read 250-500 line "
            "spans on larger files instead of paging in small slices, avoid interactive one-liner REPL probes, "
            "run only the focused test file(s) needed for your changes, do not re-read files you have already seen, "
            "and remove any temporary scratch scripts before finishing."
        )
    if progress_channel:
        parts.append(
            "## Reporting\nUse the report_to_lead tool: kind=\"milestone\" after each plan step or when tests "
            "pass; kind=\"question\" if a requirement is ambiguous; kind=\"blocker\" if you cannot proceed "
            "(questions and blockers end your turn). Finish with a short summary that marks each plan step "
            "DONE or BLOCKED."
        )
    if (
        os.environ.get("CC_AGY_CONTAINMENT", "1").lower() not in ("0", "false", "no")
        and is_straitjacket_available()
    ):
        parts.append(
            "## Shell Commands & Evidence\n"
            "Commands execute with bounded output. If a test or build produces large output, it returns a concise "
            "digest and an address (e.g. `run:<id>#stdout`). Use `ctx_get(address, start_line, end_line)` or "
            "`ctx_search(address, pattern)` to retrieve specific omitted lines or tracebacks instead of rerunning commands."
        )
    return "\n\n".join(parts)


def _parse_test_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "errors": 0}
    text = output or ""

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

    # 4. Fallback for minimal/custom pytest output without a duration footer
    for key, pat in (("passed", r"(\d+) passed\b"), ("failed", r"(\d+) failed\b"), ("errors", r"(\d+) errors?\b")):
        found = re.findall(pat, text)
        if found:
            counts[key] = int(found[-1])
    return counts


def _build_workdir_exec_env(cwd: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment dict that prepends cwd (and cwd/src) to PYTHONPATH and cwd/.venv/bin to PATH."""
    env = dict(os.environ if base_env is None else base_env)
    cwd_resolved = cwd.resolve()
    py_parts = [str(cwd_resolved)]
    src_dir = cwd_resolved / "src"
    if src_dir.is_dir():
        py_parts.append(str(src_dir))
    existing_py = (env.get("PYTHONPATH") or "").strip()
    if existing_py:
        py_parts.append(existing_py)
    env["PYTHONPATH"] = ":".join(py_parts)
    venv_dir = cwd_resolved / ".venv"
    venv_bin = venv_dir / "bin"
    if venv_bin.is_dir():
        existing_path = (env.get("PATH") or "").strip()
        env["PATH"] = f"{venv_bin}:{existing_path}" if existing_path else str(venv_bin)
        env["VIRTUAL_ENV"] = str(venv_dir)
    return env


def run_acceptance(cmds: list[str], cwd: Path, timeout: float | None = None) -> list[dict[str, Any]]:
    """Run acceptance checks with a hard timeout (whole process group is killed), then delete any new
    untracked files the checks themselves created (e.g. caches or TLS key logs), so they never land in
    the diff or block later turns."""
    limit = float(timeout or _env_int("CC_AGY_ACCEPTANCE_TIMEOUT", int(DEFAULT_ACCEPTANCE_TIMEOUT)))
    before = set(_snapshot_untracked_hashes(cwd))
    exec_env = _build_workdir_exec_env(cwd)
    results: list[dict[str, Any]] = []
    for cmd in cmds:
        started = time.time()
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            cwd=str(cwd),
            env=exec_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        timed_out = False
        try:
            out, _ = proc.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            out, _ = proc.communicate()
        tail, _, _ = truncate_to_token_budget((out or "")[-20000:], ACCEPTANCE_OUTPUT_TOKENS)
        results.append(
            {
                "command": cmd,
                "exit_code": None if timed_out else proc.returncode,
                "timed_out": timed_out,
                "passed": (not timed_out) and proc.returncode == 0,
                "duration_seconds": round(time.time() - started, 1),
                "counts": _parse_test_counts(out or ""),
                "output_tail": tail,
            }
        )
    created = [rel for rel in _snapshot_untracked_hashes(cwd) if rel not in before]
    for rel in created:
        p = cwd / rel
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
    if created and results:
        results[-1]["cleaned_artifacts"] = created[:20]
    return results


_PROMPT_SYMBOL_STOPLIST = frozenset({
    "True", "False", "None", "self", "cls", "args", "kwargs", "dict", "list", "tuple", "set",
    "str", "int", "float", "bool", "bytes", "object", "type", "super", "Path", "Any", "Optional",
    "Union", "Callable", "Iterable", "Iterator", "Sequence", "Mapping", "MutableMapping", "Literal",
    "Exception", "BaseException", "ValueError", "TypeError", "RuntimeError", "KeyError", "IndexError",
    "AttributeError", "FileNotFoundError", "OSError", "NotImplementedError", "StopIteration",
    "pytest", "unittest", "git", "python", "python3", "bash", "sh", "node", "npm", "npx", "cargo",
    "rustc", "go", "pip", "uv", "poetry", "ruff", "mypy", "flake8", "black", "isort",
    "main", "init", "setup", "test", "tests", "src", "pkg", "app", "lib", "core", "utils", "helpers",
    "todo", "note", "warning", "error", "info", "debug", "true", "false", "null", "none",
    "view_file", "edit_file", "create_file", "run_command", "search_dir", "list_dir", "find_file",
    "ctx_get", "ctx_search", "ctx_query", "report_to_lead", "delegate_start", "delegate_continue",
    "delegate_apply", "delegate_result", "delegate_status", "delegate_cancel", "delegate_revert",
    "delegate_to_gemini", "ask_gemini", "get_diff", "cost_report", "probe_writers",
})


def _detect_default_pytest_runner(cwd: Path) -> str:
    """Detect the best pytest command prefix for `cwd` when the caller supplied no acceptance commands.
    Returns `""` if pytest is not installed in `cwd`'s environment."""
    for rel_cand in (".venv/bin/pytest", "venv/bin/pytest"):
        if (cwd / rel_cand).is_file():
            return rel_cand
    exec_env = _build_workdir_exec_env(cwd)
    if shutil.which("pytest", path=exec_env.get("PATH")):
        return "pytest"
    for py_cand in (".venv/bin/python", "venv/bin/python"):
        if (cwd / py_cand).is_file():
            probe = subprocess.run(
                [str(cwd / py_cand), "-c", "import pytest"],
                cwd=str(cwd),
                env=exec_env,
                capture_output=True,
                check=False,
            )
            if probe.returncode == 0:
                return f"{py_cand} -m pytest"
    probe_sys = subprocess.run(
        ["python3", "-c", "import pytest"],
        cwd=str(cwd),
        env=exec_env,
        capture_output=True,
        check=False,
    )
    if probe_sys.returncode == 0:
        return "python3 -m pytest"
    return ""


def discover_related_test_commands(
    cwd: Path,
    files_changed: list[str],
    existing_cmds: list[str],
    *,
    max_files: int = 4,
) -> list[str]:
    """Discover existing tracked unit test files (`tests/test_<stem>.py`) for modified Python source
    modules that are not already covered by `existing_cmds`. When `existing_cmds` is empty, also
    discovers modified test files and synthesizes a default pytest command."""
    if not files_changed:
        return []
    pytest_cmds: list[str] = []
    if existing_cmds:
        pytest_cmds = [c for c in existing_cmds if re.search(r"\bpytest\b", c)]
        if not pytest_cmds:
            return []
        # If any existing pytest command runs the full suite or a directory without targeting specific .py files
        # and without filtering to a subset via -k, no supplemental regression command is needed.
        for pcmd in pytest_cmds:
            if not re.search(r"[\w./-]+\.py\b", pcmd) and not re.search(r"(?:^|\s)-k(?:\s|=|$)", pcmd):
                return []

    discovered: list[str] = []
    stems_to_check: list[str] = []
    for rel in files_changed:
        norm = rel.replace("\\", "/").strip("/")
        if not norm.endswith(".py"):
            continue
        bname = Path(norm).name
        if bname.startswith("test_") or bname.endswith("_test.py"):
            if not existing_cmds and (cwd / norm).is_file() and norm not in discovered:
                discovered.append(norm)
            continue
        if bname in ("__init__.py", "conftest.py", "setup.py"):
            continue
        parts = Path(norm).parts
        if any(p in ("tests", "test", "docs", "examples", "benchmarks") for p in parts[:-1]):
            continue
        stem = Path(norm).stem
        if stem and stem not in stems_to_check:
            stems_to_check.append(stem)

    if stems_to_check and len(discovered) < max_files:
        tracked_res = subprocess.run(
            ["git", "ls-files"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        tracked_files = [line.strip() for line in (tracked_res.stdout or "").splitlines() if line.strip()]
        for stem in stems_to_check:
            target_names = {f"test_{stem}.py", f"{stem}_test.py"}
            for tf in tracked_files:
                tf_name = Path(tf).name
                if tf_name in target_names and tf not in discovered:
                    already_covered = any(
                        re.search(rf"(?:^|[\s/\"'=]){re.escape(tf_name)}(?:$|[\s\"':])", c)
                        for c in pytest_cmds
                    )
                    if not already_covered and (cwd / tf).is_file():
                        discovered.append(tf)
                        if len(discovered) >= max_files:
                            break
            if len(discovered) >= max_files:
                break

    if not discovered:
        return []

    discovered = discovered[:max_files]
    if pytest_cmds:
        # Preserve the full pytest invocation prefix (including env vars, `uv run`, `poetry run`, `python3 -m pytest`, etc.)
        first_pytest = pytest_cmds[0].strip()
        m_pref = re.match(r"^(.*?\bpytest)\b", first_pytest)
        runner_prefix = m_pref.group(1).strip() if m_pref else "pytest"
    else:
        runner_prefix = _detect_default_pytest_runner(cwd)
        if not runner_prefix:
            return []
    quoted_files = " ".join(shlex.quote(f) for f in discovered)
    return [f"{runner_prefix} -q {quoted_files}"]


_IMPORT_SKIP_DIRS = frozenset({
    "tests", "test", "docs", "examples", "benchmarks", "scripts", "migrations", "tools", "bin",
})
_IMPORT_SKIP_FILES = frozenset({
    "setup.py", "conftest.py", "__main__.py", "manage.py", "noxfile.py", "tasks.py", "fabfile.py",
    "wsgi.py", "asgi.py", "sitecustomize.py", "usercustomize.py",
})


def build_import_smoke_command(
    cwd: Path,
    files_changed: list[str],
    *,
    max_modules: int = 8,
) -> list[str]:
    """Build a fast Python import smoke-check command for modified or created Python modules."""
    if not files_changed:
        return []
    modules: list[str] = []
    for rel in files_changed:
        norm = rel.replace("\\", "/").strip("/")
        if not norm.endswith(".py") or not (cwd / norm).is_file():
            continue
        bname = Path(norm).name
        if bname in _IMPORT_SKIP_FILES or bname.startswith("test_") or bname.endswith("_test.py"):
            continue
        parts = list(Path(norm).parts)
        if any(p in _IMPORT_SKIP_DIRS or p.startswith(".") for p in parts[:-1]):
            continue
        base_dir = cwd
        if parts[0] == "src" and len(parts) > 1:
            base_dir = cwd / "src"
            parts = parts[1:]
        if len(parts) <= 1:
            # Never import top-level non-package .py scripts (e.g. run.py), which can execute CLI/server side effects.
            continue
        # Verify every parent directory is a valid Python package with __init__.py
        cur = base_dir
        valid_pkg = True
        for pkg_part in parts[:-1]:
            cur = cur / pkg_part
            if not (cur / "__init__.py").is_file() or not pkg_part.isidentifier():
                valid_pkg = False
                break
        if not valid_pkg:
            continue
        if bname == "__init__.py":
            mod_name = ".".join(parts[:-1])
        else:
            stem = Path(bname).stem
            if not stem.isidentifier():
                continue
            mod_name = ".".join((*parts[:-1], stem))
        if mod_name and mod_name not in modules:
            modules.append(mod_name)
            if len(modules) >= max_modules:
                break
    if not modules:
        return []
    py_bin = "python3"
    for cand in (".venv/bin/python", "venv/bin/python"):
        if (cwd / cand).is_file():
            py_bin = cand
            break
    quoted_mods = " ".join(shlex.quote(m) for m in modules)
    return [
        f"{py_bin} -c 'import importlib, sys; [importlib.import_module(m) for m in sys.argv[1:]]' {quoted_mods}"
    ]


def find_missing_prompt_symbols(
    prompt: str,
    diff_text: str,
    *,
    worktree_root: Path | None = None,
    files_changed: list[str] | None = None,
    baseline_ref: str | None = None,
) -> list[str]:
    """Identify code symbols explicitly backticked in `prompt` (e.g. `` `my_func` ``, `` `MyClass.method` ``)
    that are completely absent from both `diff_text` and the modified files in `worktree_root`."""
    if not prompt or not prompt.strip():
        return []

    # Collect file stems mentioned in the prompt or files_changed so module names are not mistaken for missing symbols
    file_stems: set[str] = set()
    for f_match in re.findall(r"[\w./-]+\.(?:py|ts|tsx|js|jsx|rs|go|java|rb|c|cc|cpp|h|hpp)\b", prompt):
        file_stems.add(Path(f_match).stem)
    for fc in files_changed or []:
        file_stems.add(Path(fc).stem)

    candidate_symbols: list[str] = []
    # Method half of a backticked `ClassName.method` -> ClassName, used to scope the baseline check.
    member_owners: dict[str, str] = {}
    for raw_span in re.findall(r"(?<!`)`([^`\n]+)`(?!`)", prompt):
        span = raw_span.strip()
        if not span or span.startswith("-"):
            continue
        span = re.sub(r"^(?:async\s+def|def|class)\s+", "", span).rstrip(":").strip()
        has_call_parens = "(" in span and ")" in span
        if "(" in span:
            span = span.split("(", 1)[0].strip()
        if not span or len(span) < 3 or len(span) > 80:
            continue
        if " " in span or "/" in span or "\\" in span:
            continue
        if re.search(r"\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml|toml|cfg|ini|sh|txt|rst|html|css|sql)\b", span, re.IGNORECASE):
            continue
        # Handle dotted identifiers like `ClassName.method_name` or `module.func_name`
        parts = span.split(".")
        if len(parts) > 2 or not all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", p) for p in parts):
            continue
        tokens_to_check = [parts[-1]] if len(parts) == 2 else [parts[0]]
        if len(parts) == 2 and re.match(r"^[A-Z][A-Za-z0-9_]*$", parts[0]) and any(c.islower() for c in parts[0]):
            tokens_to_check.insert(0, parts[0])
            member_owners.setdefault(parts[-1], parts[0])
        for tok in tokens_to_check:
            if len(tok) < 4 or tok in _PROMPT_SYMBOL_STOPLIST or tok in file_stems:
                continue
            # Ignore all-uppercase constants/env-vars like `TRUE`, `JSON`, `HTTP`, `CC_AGY_...`
            if tok.isupper():
                continue
            is_snake = "_" in tok and any(c.islower() for c in tok)
            is_camel = bool(re.match(r"^[A-Z][A-Za-z0-9]+$", tok)) and any(c.islower() for c in tok)
            if (is_snake or is_camel or has_call_parens) and tok not in candidate_symbols:
                candidate_symbols.append(tok)

    if not candidate_symbols:
        return []

    corpus_parts: list[str] = [diff_text or ""]
    if worktree_root is not None and worktree_root.is_dir():
        for rel in files_changed or []:
            fp = worktree_root / rel
            if fp.is_file() and not fp.is_symlink():
                try:
                    corpus_parts.append(fp.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    pass
    corpus = "\n".join(corpus_parts)
    if not corpus.strip():
        return candidate_symbols[:10]

    missing: list[str] = [sym for sym in candidate_symbols if not re.search(rf"\b{re.escape(sym)}\b", corpus)]
    if missing and worktree_root is not None:
        # Symbols already *defined* in non-test baseline code are references to existing code
        # ("follow `_DiverseFailableLawSpec`", "do NOT extend `SwappableN`"), not new symbols the
        # worker was asked to create, so they must not block auto-apply or trigger extra turns.
        # Mere mentions (test imports, docs) do not count: a TDD test that imports a not-yet-written
        # function must still surface that function as missing.
        defined = _symbols_defined_in_baseline(
            worktree_root, missing, ref=baseline_ref or "HEAD", owners=member_owners
        )
        missing = [sym for sym in missing if sym not in defined]
    # Cap only after filtering so real misses are never crowded out by excluded references.
    return missing[:10]


_DOC_SUFFIXES = (".md", ".rst", ".txt", ".adoc")


def _is_test_or_doc_path(rel: str) -> bool:
    p = Path(rel)
    name = p.name
    if any(part in ("tests", "test", "testing", "__tests__", "docs", "doc") for part in p.parts[:-1]):
        return True
    if name.startswith("test_") or name.endswith(("_test.py", "_test.go", "_spec.rb")):
        return True
    if ".test." in name or ".spec." in name or name == "conftest.py":
        return True
    return name.lower().endswith(_DOC_SUFFIXES)


def _symbols_defined_in_baseline(
    repo_root: Path,
    symbols: list[str],
    *,
    ref: str = "HEAD",
    owners: dict[str, str] | None = None,
) -> set[str]:
    """Return the subset of `symbols` that are defined in non-test, non-doc files of the committed `ref`
    tree of `repo_root`.

    A definition is a declaration at the start of a non-comment line (def/class/function/type/...;
    indentation allowed so methods count) or a column-0 top-level assignment. A symbol listed in `owners`
    (the method half of a backticked `Owner.method`) counts only if one file defines both `class Owner`
    and `method`, so `Order.to_payload` is not excused by an unrelated `User.to_payload`.
    """
    if not symbols or not repo_root.is_dir():
        return set()
    owner_map = {sym: own for sym, own in (owners or {}).items() if sym in symbols}
    search_terms = list(dict.fromkeys(list(symbols) + list(owner_map.values())))
    cmd = ["git", "grep", "--no-color", "-I", "-z", "-w", "-F"]
    for term in search_terms:
        cmd.extend(["-e", term])
    cmd.extend([ref, "--"])
    try:
        res = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    # git grep exits 1 when nothing matches and >1 on errors (e.g. not a git repo); both mean no exclusions.
    if res.returncode != 0:
        return set()
    decl_prefix = r"^\s*(?:(?:export|default|async|pub|public|private|protected|static|abstract|final)\s+)*"
    def_patterns = {
        term: (
            re.compile(
                rf"{decl_prefix}(?:def|class|function|interface|type|struct|enum|trait|fn|func)\s+{re.escape(term)}\b"
            ),
            re.compile(rf"^(?:(?:export|const|let|var|final|static|pub)\s+)*{re.escape(term)}\s*(?::[^=]*)?=(?!=)"),
        )
        for term in search_terms
    }
    comment_prefixes = ("#", "//", "/*", "*", "--", '"""', "'''")
    defined_by_file: dict[str, set[str]] = {}
    prefix = f"{ref}:"
    for line in (res.stdout or "").splitlines():
        # With -z and a tree-ish, each hit is `<ref>:<path>\0<content>`.
        if "\x00" not in line:
            continue
        loc, content = line.split("\x00", 1)
        rel = loc[len(prefix):] if loc.startswith(prefix) else loc
        if _is_test_or_doc_path(rel) or content.lstrip().startswith(comment_prefixes):
            continue
        for term, (decl_re, assign_re) in def_patterns.items():
            if decl_re.search(content) or assign_re.search(content):
                defined_by_file.setdefault(rel, set()).add(term)
    found: set[str] = set()
    for sym in symbols:
        owner = owner_map.get(sym)
        for defs in defined_by_file.values():
            if sym in defs and (owner is None or owner in defs):
                found.add(sym)
                break
    return found


def _hash_path(p: Path) -> str | None:
    """Return a content fingerprint for a workspace path (symlink target, file SHA-256, or None if absent)."""
    if p.is_symlink():
        return f"symlink:{os.readlink(p)}"
    if p.exists() and p.is_file():
        return sha256_bytes(p.read_bytes())
    return None


def _snapshot_untracked_hashes(root: Path) -> dict[str, str | None]:
    """Fingerprint every untracked, non-ignored file so in_place jobs can skip ones they did not touch."""
    res = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=str(root),
        capture_output=True,
        check=False,
    )
    snap: dict[str, str | None] = {}
    if res.returncode != 0:
        return snap
    for raw in res.stdout.split(b"\x00"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="surrogateescape")
        snap[rel] = _hash_path(root / rel)
    return snap


def _split_diff_by_file(diff_text: str) -> list[tuple[str, str]]:
    """Split a unified git diff into (path, chunk) pairs keyed by the `b/` path of each header."""
    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    current_path = ""
    in_hunk = False
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                chunks.append((current_path, "".join(current)))
            current = [line]
            hdr = parse_diff_git_header(line)
            current_path = (hdr[1] or hdr[0]) if hdr else ""
            in_hunk = False
        elif current:
            if line.startswith("@@ "):
                in_hunk = True
            elif not in_hunk and line.startswith("+++ "):
                plus_path = parse_unified_header_path(line, "+++")
                if plus_path and plus_path != "/dev/null":
                    current_path = plus_path
            elif not in_hunk and line.startswith("--- ") and not current_path:
                minus_path = parse_unified_header_path(line, "---")
                if minus_path and minus_path != "/dev/null":
                    current_path = minus_path
            current.append(line)
    if current:
        chunks.append((current_path, "".join(current)))
    return chunks


class JobManager:
    """Manages isolated worker clones, background execution, conflict-safe apply, and provenance."""

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        store: ReceiptStore | None = None,
        worker_backend: Callable[[dict[str, Any]], Any] | None = None,
        max_workers: int = 4,
    ) -> None:
        self.workspace_root = resolve_workspace_root(workspace_root)
        self.store = store or ReceiptStore(workspace_root=self.workspace_root)
        self.worker_backend = worker_backend
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cc-agy-job")
        self._lock = threading.Lock()
        self._apply_lock = threading.Lock()
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._cancelled: set[str] = set()
        self._raw_projects: dict[str, str] = {}
        self._raw_api_keys: dict[str, str] = {}
        self._in_place_job: str | None = None
        self._created_at = time.monotonic()
        self._ensure_baseline_recorded()

    def _is_bounded_lead_session(self) -> bool:
        if os.environ.get("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "").strip().lower() in ("1", "true", "yes"):
            return True
        if self.worker_backend is not None:
            return False
        return (
            os.environ.get("CC_AGY_STRICT_MODE", "").strip().lower() in ("1", "true", "yes")
            or bool(os.environ.get("CC_AGY_FORCE_WORKER_THINKING_LEVEL", "").strip())
        )

    def _should_run_default_acceptance(self) -> bool:
        if os.environ.get("CC_AGY_AUTO_REGRESSION_TESTS", "1").strip().lower() in ("0", "false", "no"):
            return False
        raw = os.environ.get("CC_AGY_AUTO_DEFAULT_ACCEPTANCE", "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        return self.worker_backend is None

    def _clamp_worker_timeout(self, timeout_seconds: float) -> float:
        """Clamp worker turn timeout against the session deadline (reserving headroom for review & apply)."""
        t_val = max(0.1, float(timeout_seconds))
        raw_deadline = os.environ.get("CC_AGY_SESSION_DEADLINE_SECONDS", "").strip()
        if raw_deadline:
            try:
                session_deadline = float(raw_deadline)
            except ValueError:
                session_deadline = 1500.0
            if session_deadline <= 0:
                return t_val
        elif self._is_bounded_lead_session():
            session_deadline = 1500.0
        else:
            return t_val
        raw_reserve = os.environ.get("CC_AGY_DEADLINE_RESERVE_SECONDS", "").strip()
        try:
            reserve_sec = max(15.0, float(raw_reserve)) if raw_reserve else 120.0
        except ValueError:
            reserve_sec = 120.0
        elapsed = max(0.0, time.monotonic() - self._created_at)
        worker_budget = max(45.0, (session_deadline - reserve_sec) - elapsed)
        return min(t_val, worker_budget)

    def _ensure_baseline_recorded(self) -> None:
        if self.store.is_baseline_tampered(self.workspace_root):
            return
        head_res = _run_git(["rev-parse", "HEAD"], cwd=self.workspace_root, check=False)
        head_sha = head_res.stdout.strip() if head_res.returncode == 0 else None
        existing = self.store.read_baseline_record(self.workspace_root)
        if existing is not None:
            receipts = self.store.list_receipts(workspace_root=self.workspace_root)
            for r in receipts:
                if r.get("kind") == "delegate" and not r.get("applied") and r.get("status") == "running":
                    return
            existing_sha = existing.get("baseline_head_sha")
            if head_sha and existing_sha and head_sha != existing_sha:
                # HEAD moved since the prior baseline was recorded. Verify existing_sha is an
                # ancestor of head_sha (normal human commit between sessions vs amend/reset).
                anc_res = _run_git(
                    ["merge-base", "--is-ancestor", str(existing_sha), str(head_sha)],
                    cwd=self.workspace_root,
                    check=False,
                )
                if anc_res.returncode != 0:
                    return
            else:
                # Same HEAD commit: refuse to overwrite if tracked files or untracked code files
                # have unverified modifications (prevents laundering host edits via MCP server restart).
                existing_prov = existing.get("provenance") or {}
                existing_prov_files = set(existing_prov.get("files_changed") or [])
                applied_files: set[str] = set()
                latest_applied_hash: dict[str, str | None] = {}
                for r in receipts:
                    if r.get("applied") and self.store.verify_receipt_signature(r):
                        applied_files.update(r.get("applied_files") or [])
                        for f_k, f_h in (r.get("applied_file_hashes") or {}).items():
                            latest_applied_hash[f_k] = f_h
                    for hist_prov in r.get("applied_history") or []:
                        if isinstance(hist_prov, dict):
                            applied_files.update(hist_prov.get("files_changed") or [])

                tracked_diff_res = _run_git(
                    ["diff", "--name-only", "HEAD"],
                    cwd=self.workspace_root,
                    check=False,
                )
                if tracked_diff_res.returncode == 0 and tracked_diff_res.stdout.strip():
                    for fpath in tracked_diff_res.stdout.splitlines():
                        clean_f = fpath.strip()
                        if not clean_f:
                            continue
                        if clean_f not in existing_prov_files and clean_f not in applied_files:
                            return
                        if clean_f in latest_applied_hash:
                            expected_h = latest_applied_hash[clean_f]
                            live_p = self.workspace_root / clean_f
                            actual_h = (
                                sha256_bytes(live_p.read_bytes())
                                if live_p.exists() and live_p.is_file() and not live_p.is_symlink()
                                else None
                            )
                            if actual_h != expected_h:
                                return

                untracked_res = _run_git(
                    ["ls-files", "--others", "--exclude-standard"],
                    cwd=self.workspace_root,
                    check=False,
                )
                if untracked_res.returncode == 0 and untracked_res.stdout.strip():
                    for line in untracked_res.stdout.splitlines():
                        rel_u = line.strip()
                        if (
                            not rel_u
                            or rel_u.startswith((".cc-agy-data/", ".worktrees/"))
                            or _is_ephemeral_build_or_test_artifact(rel_u, root=self.workspace_root)
                            or rel_u in existing_prov_files
                            or rel_u in applied_files
                        ):
                            continue
                        if not rel_u.lower().endswith(_SAFE_NOTE_EXTENSIONS):
                            return

        live_diff = capture_live_workspace_diff(self.workspace_root, baseline_head_sha=head_sha)
        prov = extract_diff_provenance(live_diff)
        self.store.record_baseline_dirty_state(
            self.workspace_root,
            prov,
            baseline_head_sha=head_sha,
            overwrite=True,
        )

    def _prepare_worktree(self, job_id: str) -> tuple[Path, str, dict[str, str | None]]:
        """Create a standalone isolated git repository copy for `job_id`.

        Uses `git ls-files -co --exclude-standard -z` so gitignored directories (like
        `node_modules/` or `.venv/`) are never copied into the worker tree, and computes
        baseline file hashes on demand for changed files rather than bloating the receipt JSON.
        """
        safe_id = validate_job_id(job_id)
        wt_dir = self.store.worktree_path(safe_id).resolve()
        if wt_dir.exists():
            shutil.rmtree(wt_dir, ignore_errors=True)
        wt_dir.mkdir(parents=True, exist_ok=True)

        ls_res = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=str(self.workspace_root),
            capture_output=True,
            check=False,
        )
        if ls_res.returncode == 0:
            rel_entries = [
                entry.decode("utf-8", errors="surrogateescape")
                for entry in ls_res.stdout.split(b"\x00")
                if entry
            ]
            for rel in rel_entries:
                if rel.startswith((".git/", ".cc-agy-data/", ".worktrees/", ".ctx-session-reads/", ".cc-agy-review/", ".venv/", "venv/")) or rel in (".venv", "venv", ".ctx-session-reads", ".cc-agy-review"):
                    continue
                src_p = self.workspace_root / rel
                dst_p = wt_dir / rel
                if src_p.is_symlink():
                    dst_p.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(os.readlink(src_p), dst_p)
                elif src_p.is_file():
                    dst_p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_p, dst_p)
        else:
            shutil.rmtree(wt_dir, ignore_errors=True)
            shutil.copytree(
                self.workspace_root,
                wt_dir,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    ".git", ".cc-agy-data", ".worktrees", ".ctx-session-reads", ".cc-agy-review", "__pycache__", "node_modules", ".venv", "venv"
                ),
            )

        _run_git(["init", "-b", "main"], cwd=wt_dir, check=True)
        _run_git(["config", "user.email", "worker@cc-agy-plugin.local"], cwd=wt_dir, check=True)
        _run_git(["config", "user.name", "CC-AGY Worker"], cwd=wt_dir, check=True)
        _run_git(["add", "-A"], cwd=wt_dir, check=True)
        _run_git(["commit", "--allow-empty", "-m", "cc-agy-baseline"], cwd=wt_dir, check=True)
        sha_res = _run_git(["rev-parse", "HEAD"], cwd=wt_dir, check=True)
        baseline_sha = sha_res.stdout.strip()

        exclude_file = wt_dir / ".git" / "info" / "exclude"
        try:
            exclude_file.parent.mkdir(parents=True, exist_ok=True)
            existing_excl = exclude_file.read_text(encoding="utf-8", errors="replace") if exclude_file.exists() else ""
            extra_excl = "\n".join(
                [
                    ".venv",
                    ".venv/",
                    ".pytest_cache/",
                    "__pycache__/",
                    ".mypy_cache/",
                    ".ruff_cache/",
                    ".ctx-session-reads",
                    ".ctx-session-reads/",
                    ".cc-agy-review",
                    ".cc-agy-review/",
                ]
            )
            exclude_file.write_text(
                (existing_excl.rstrip("\n") + "\n" + extra_excl + "\n").lstrip("\n"),
                encoding="utf-8",
            )
        except OSError:
            pass

        host_venv = self.workspace_root / ".venv"
        wt_venv = wt_dir / ".venv"
        if host_venv.exists() and not wt_venv.exists() and not wt_venv.is_symlink():
            try:
                os.symlink(str(host_venv.resolve()), str(wt_venv))
            except OSError:
                pass

        return wt_dir, baseline_sha, {}

    def _compute_changed_baseline_hashes(
        self,
        wt_dir: Path,
        baseline_sha: str,
        files_changed: list[str],
        prior_hashes: dict[str, str | None] | None = None,
    ) -> dict[str, str | None]:
        """Compute baseline SHA-256 hashes only for the files touched by the worker."""
        hashes: dict[str, str | None] = dict(prior_hashes or {})
        if not wt_dir.exists():
            return hashes
        for rel in files_changed:
            if rel in hashes:
                continue
            show_res = subprocess.run(
                ["git", "show", f"{baseline_sha}:{rel}"],
                cwd=str(wt_dir),
                capture_output=True,
                check=False,
            )
            if show_res.returncode == 0:
                hashes[rel] = sha256_bytes(show_res.stdout)
            else:
                hashes[rel] = None
        return hashes

    def _collect_worktree_diff(
        self,
        wt_dir: Path,
        baseline_sha: str,
        allowed_paths: list[str] | None = None,
    ) -> tuple[str, list[str], list[str]]:
        """Collect unified binary diff with `--no-renames` against `baseline_sha`.

        Also force-stages any new `.gitignore`d files written by the worker inside `wt_dir`
        (excluding transient build/test caches such as `__pycache__/` or `.pytest_cache/`).
        Returns (diff_text, files_changed, symlink_escapes).
        """
        if not wt_dir.exists():
            return "", [], []

        symlink_escapes: list[str] = []
        wt_resolved = wt_dir.resolve()
        for p in list(wt_dir.rglob("*")):
            if ".git" in p.parts:
                continue
            rel = str(p.relative_to(wt_dir))
            if _is_ephemeral_build_or_test_artifact(rel, root=wt_dir):
                continue
            if p.is_symlink():
                try:
                    link_target = (p.parent / os.readlink(p)).resolve()
                    if not link_target.is_relative_to(wt_resolved):
                        symlink_escapes.append(rel)
                except OSError:
                    symlink_escapes.append(rel)

        _run_git(["reset", "-q", baseline_sha], cwd=wt_dir, check=False)
        untracked_to_stage: list[str] = []
        others_res = _run_git(
            ["ls-files", "--others", "--exclude-standard"],
            cwd=wt_dir,
            check=False,
        )
        if others_res.returncode == 0 and others_res.stdout.strip():
            for line in others_res.stdout.splitlines():
                rel_u = line.strip()
                if not rel_u:
                    continue
                if _is_ephemeral_build_or_test_artifact(rel_u, root=wt_dir):
                    eph_p = wt_dir / rel_u
                    if eph_p.is_file() and not eph_p.is_symlink():
                        eph_p.unlink(missing_ok=True)
                    continue
                untracked_to_stage.append(rel_u)

        ign_res = _run_git(
            ["ls-files", "--others", "--ignored", "--exclude-standard"],
            cwd=wt_dir,
            check=False,
        )
        if ign_res.returncode == 0 and ign_res.stdout.strip():
            for line in ign_res.stdout.splitlines():
                rel_i = line.strip()
                if (
                    rel_i
                    and not _is_ignored_build_artifact(rel_i, root=wt_dir)
                    and rel_i not in untracked_to_stage
                ):
                    untracked_to_stage.append(rel_i)

        if untracked_to_stage:
            _run_git(["add", "-f", "-N", "--", *untracked_to_stage], cwd=wt_dir, check=False)

        diff_res = _run_git(
            ["diff", "--binary", "--no-renames", baseline_sha],
            cwd=wt_dir,
            check=False,
        )
        diff_text = diff_res.stdout if diff_res.returncode == 0 else ""

        name_res = _run_git(
            ["diff", "--name-only", "--no-renames", baseline_sha],
            cwd=wt_dir,
            check=False,
        )
        if untracked_to_stage:
            _run_git(["reset", "-q", baseline_sha], cwd=wt_dir, check=False)
        files = [
            line.strip()
            for line in (name_res.stdout.splitlines() if name_res.returncode == 0 else [])
            if line.strip()
        ]
        for esc in symlink_escapes:
            if esc not in files:
                files.append(esc)
        return diff_text, files, symlink_escapes

    def _cleanup_worktree(self, wt_dir: Path) -> None:
        if not wt_dir.exists():
            return
        # Never delete anything outside the plugin's own worktrees directory. In in_place mode
        # the "worktree" is the user's real repository, so this guard must never be removed.
        try:
            resolved = wt_dir.resolve()
            if not resolved.is_relative_to(self.store.worktrees_dir.resolve()):
                return
            if resolved == self.workspace_root.resolve():
                return
        except (OSError, ValueError):
            return
        shutil.rmtree(wt_dir, ignore_errors=True)

    def _snapshot_branches(self) -> tuple[str | None, list[str]]:
        """Capture (current symbolic branch name or None, list of local branch names)."""
        sym_res = _run_git(["symbolic-ref", "-q", "--short", "HEAD"], cwd=self.workspace_root, check=False)
        orig_branch = sym_res.stdout.strip() if sym_res.returncode == 0 and sym_res.stdout.strip() else None
        refs_res = _run_git(
            ["for-each-ref", "--format=%(refname:short)", "refs/heads/"],
            cwd=self.workspace_root,
            check=False,
        )
        branches = [b.strip() for b in refs_res.stdout.splitlines() if b.strip()] if refs_res.returncode == 0 else []
        return orig_branch, branches

    def _check_in_place_ready(self) -> tuple[str, dict[str, str | None]]:
        """Verify the host repo can be edited in place; return (HEAD sha, untracked file hashes)."""
        head_res = _run_git(["rev-parse", "--verify", "HEAD"], cwd=self.workspace_root, check=False)
        if head_res.returncode != 0 or not head_res.stdout.strip():
            raise ValueError("in_place mode needs a git repository with at least one commit.")
        dirty_res = _run_git(
            ["status", "--porcelain", "--untracked-files=no"],
            cwd=self.workspace_root,
            check=False,
        )
        dirty = [line for line in dirty_res.stdout.splitlines() if line.strip()]
        if dirty:
            raise ValueError(
                "in_place mode needs a clean working tree for tracked files, so git can undo the "
                "worker's edits. If they come from an earlier in_place job, send it a follow-up with "
                "delegate_continue instead. Otherwise commit or stash them, or use the default isolated copy: "
                f"{dirty[:10]}"
            )
        return head_res.stdout.strip(), _snapshot_untracked_hashes(self.workspace_root)

    def _acquire_in_place_slot(self, job_id: str) -> None:
        with self._lock:
            active = self._in_place_job
            if active is not None and active != job_id:
                active_fut = self._futures.get(active)
                if active_fut is None or not active_fut.done():
                    raise RuntimeError(
                        f"Another in_place job ({active}) is still running in this repository. "
                        f"Wait for it to finish or use the default isolated copy."
                    )
            self._in_place_job = job_id

    def _release_in_place_slot(self, job_id: str) -> None:
        with self._lock:
            if self._in_place_job == job_id:
                self._in_place_job = None

    def _finalize_in_place(
        self,
        baseline_sha: str,
        pre_untracked: dict[str, str | None],
        orig_branch: str | None = None,
        pre_branches: list[str] | None = None,
    ) -> tuple[str, list[str], list[str], list[str]]:
        """Collect the worker's in-place edits against baseline_sha.

        Undoes any commits or branch switches the worker made (keeping their changes in the
        working tree), restores the index, and skips untracked files that existed before the job
        and are unchanged. Returns (diff_text, files_changed, symlink_escapes, notes).
        """
        root = self.workspace_root
        notes: list[str] = []
        sym_res = _run_git(["symbolic-ref", "-q", "--short", "HEAD"], cwd=root, check=False)
        cur_branch = sym_res.stdout.strip() if sym_res.returncode == 0 and sym_res.stdout.strip() else None
        head_res = _run_git(["rev-parse", "HEAD"], cwd=root, check=False)
        head_now = head_res.stdout.strip() if head_res.returncode == 0 else ""

        if orig_branch:
            _run_git(["update-ref", f"refs/heads/{orig_branch}", baseline_sha], cwd=root, check=False)
            if cur_branch != orig_branch:
                _run_git(["symbolic-ref", "HEAD", f"refs/heads/{orig_branch}"], cwd=root, check=False)
                notes.append(
                    f"Worker switched branch to {cur_branch or 'detached HEAD'!r}; "
                    f"restored HEAD to branch {orig_branch!r} at {baseline_sha[:12]}."
                )
        elif cur_branch:
            _run_git(["symbolic-ref", "--delete", "HEAD"], cwd=root, check=False)
            _run_git(["update-ref", "--no-deref", "HEAD", baseline_sha], cwd=root, check=False)
            notes.append(
                f"Worker switched to branch {cur_branch!r}; restored detached HEAD at {baseline_sha[:12]}."
            )

        if pre_branches is not None:
            refs_res = _run_git(
                ["for-each-ref", "--format=%(refname:short)", "refs/heads/"],
                cwd=root,
                check=False,
            )
            now_branches = (
                [b.strip() for b in refs_res.stdout.splitlines() if b.strip()]
                if refs_res.returncode == 0
                else []
            )
            pre_set = set(pre_branches)
            for b in now_branches:
                if b not in pre_set and b != orig_branch:
                    b_rev = _run_git(["rev-parse", "--verify", f"refs/heads/{b}"], cwd=root, check=False)
                    b_sha = b_rev.stdout.strip() if b_rev.returncode == 0 else ""
                    if b_sha and b_sha != baseline_sha and b_sha != head_now:
                        _run_git(["checkout", b_sha, "--", "."], cwd=root, check=False)
                        notes.append(
                            f"Recovered committed changes from worker branch {b!r} ({b_sha[:12]}) "
                            f"into working tree before deleting branch."
                        )
                    _run_git(["branch", "-D", b], cwd=root, check=False)
                    notes.append(f"Deleted worker-created branch {b!r}.")

        if head_now and head_now != baseline_sha:
            notes.append(
                f"Worker moved HEAD to {head_now[:12]}; its commits were undone with "
                f"`git reset --mixed {baseline_sha[:12]}` and the changes kept in the working tree."
            )
        _run_git(["reset", "-q", baseline_sha], cwd=root, check=False)

        live_diff = capture_live_workspace_diff(root, baseline_head_sha=baseline_sha)
        kept_chunks: list[str] = []
        files: list[str] = []
        for rel, chunk in _split_diff_by_file(live_diff):
            if not rel or rel in files:
                continue
            if rel.startswith((".cc-agy-data/", ".worktrees/", ".ctx-session-reads/", ".cc-agy-review/")):
                continue
            # Unlike the isolated copy, caches the worker writes here (e.g. __pycache__) really are
            # in the user's tree, so they are recorded rather than filtered (and never deleted).
            if rel in pre_untracked and _hash_path(root / rel) == pre_untracked[rel]:
                continue
            kept_chunks.append(chunk)
            files.append(rel)

        escapes: list[str] = []
        root_resolved = root.resolve()
        for rel in files:
            p = root / rel
            if p.is_symlink():
                try:
                    target = (p.parent / os.readlink(p)).resolve()
                    if not target.is_relative_to(root_resolved):
                        escapes.append(rel)
                except OSError:
                    escapes.append(rel)
        return "".join(kept_chunks), files, escapes, notes

    def _execute_job(
        self,
        *,
        job_id: str,
        worker_prompts: list[str],
        recorded_prompts: list[str],
        model: str,
        auth_mode: str = "vertex",
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        reroute_note: str | None,
        allowed_paths: list[str],
        timeout_seconds: float,
        enabled_tools: list[str] | None,
        disabled_tools: list[str] | None,
        enable_subagents: bool,
        wt_dir: Path,
        baseline_sha: str,
        baseline_file_hashes: dict[str, str | None],
        started_at: float,
        prior_usage: dict[str, Any] | None = None,
        prior_tool_calls: list[dict[str, Any]] | None = None,
        scrub_receipt: bool = False,
        in_place: bool = False,
        pre_untracked: dict[str, str | None] | None = None,
        thinking_level: str | None = None,
        plan: list[str] | None = None,
        acceptance: list[str] | None = None,
        max_tool_calls: int | None = None,
        max_total_tokens: int | None = None,
        preloaded_files: list[str] | None = None,
    ) -> dict[str, Any]:
        token_keys = (
            "prompt_token_count",
            "cached_content_token_count",
            "candidates_token_count",
            "thoughts_token_count",
            "total_token_count",
        )
        usage: dict[str, Any] = {
            k: int((prior_usage or {}).get(k, 0)) for k in token_keys
        }
        prior_requests = list((prior_usage or {}).get("requests") or []) if isinstance(prior_usage, dict) else []
        prior_max_req_prompt = int((prior_usage or {}).get("max_request_prompt_tokens") or 0) if isinstance(prior_usage, dict) else 0
        if prior_requests:
            usage["requests"] = prior_requests
        if prior_max_req_prompt > 0:
            usage["max_request_prompt_tokens"] = prior_max_req_prompt

        tool_calls: list[dict[str, Any]] = list(prior_tool_calls or [])
        thoughts: list[str] = []
        response_text = ""
        status = "error"
        error_msg: str | None = None
        diff_text = ""
        files_changed: list[str] = []
        scope_violations: list[str] = []
        in_place_notes: list[str] = []
        symlink_escapes: list[str] = []
        stop_reason: str | None = None
        acceptance_results: list[dict[str, Any]] = []
        progress_path = self.store.progress_path(job_id)
        progress_before = len(read_progress(progress_path, limit=10**6))

        host_git_dir = self.workspace_root / ".git"
        hooks_snap = _snapshot_directory_tree(host_git_dir / "hooks")
        config_snap = _snapshot_directory_tree(host_git_dir / "config")
        orig_branch, pre_branches = self._snapshot_branches() if in_place else (None, [])
        worker_out: dict[str, Any] = {}
        workspace_dir_accessed = False

        class _TrackedReq(dict[str, Any]):
            def __getitem__(self, key: str) -> Any:
                nonlocal workspace_dir_accessed
                if key == "workspace_dir":
                    workspace_dir_accessed = True
                return super().__getitem__(key)

            def get(self, key: str, default: Any = None) -> Any:
                nonlocal workspace_dir_accessed
                if key == "workspace_dir":
                    workspace_dir_accessed = True
                return super().get(key, default)

        raw_backend = self.worker_backend
        if raw_backend is None:
            workspace_dir_accessed = True
            tracked_backend = None
        else:
            def tracked_backend(req: dict[str, Any]) -> Any:
                return raw_backend(_TrackedReq(req))

        try:
            with self._lock:
                if job_id in self._cancelled:
                    status = "cancelled"
                    error_msg = "Job cancelled before execution."
                    raise RuntimeError(error_msg)

            worker_out = run_antigravity_worker(
                prompts=worker_prompts,
                workspace_dir=wt_dir,
                model=model,
                project=project,
                location=location,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                enabled_tools=enabled_tools,
                disabled_tools=disabled_tools,
                enable_subagents=enable_subagents,
                backend=tracked_backend,
                thinking_level=thinking_level,
                max_tool_calls=max_tool_calls,
                max_total_tokens=max_total_tokens,
                progress_path=progress_path,
            )

            response_text = str(worker_out.get("text") or "")
            new_usage = worker_out.get("usage") or {}
            for k in token_keys:
                usage[k] += int(new_usage.get(k) or 0)
            usage["total_token_count"] = max(
                usage["total_token_count"],
                usage["prompt_token_count"]
                + usage["candidates_token_count"]
                + usage["thoughts_token_count"],
            )
            new_requests = new_usage.get("requests")
            if isinstance(new_requests, list) and new_requests:
                usage.setdefault("requests", []).extend(new_requests)
            new_max_req = int(new_usage.get("max_request_prompt_tokens") or 0)
            if new_max_req > 0 or usage.get("max_request_prompt_tokens"):
                usage["max_request_prompt_tokens"] = max(
                    int(usage.get("max_request_prompt_tokens") or 0),
                    new_max_req,
                )
            tool_calls.extend(worker_out.get("tool_calls") or [])
            thoughts.extend(worker_out.get("thoughts") or [])
            stop_reason = worker_out.get("stop_reason")

            with self._lock:
                if job_id in self._cancelled:
                    status = "cancelled"
                    error_msg = "Job cancelled during execution."
                else:
                    status = "ok"
                    new_signals = read_progress(progress_path, limit=10**6)[progress_before:]
                    if stop_reason in CHECKPOINT_STOP_REASONS:
                        # Budget reached: not an error. The lead reviews and continues or redirects.
                        status = "checkpoint"
                    elif new_signals and new_signals[-1].get("kind") in ("question", "blocker"):
                        status = "needs_input"

        except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError) as exc:
            with self._lock:
                is_canc = job_id in self._cancelled
            if is_canc:
                status = "cancelled"
                error_msg = "Job cancelled during execution."
            else:
                status = "timeout"
                error_msg = f"TimeoutError: {exc}"
            self._merge_partial_exception_data(exc, usage, tool_calls, thoughts)
            if getattr(exc, "partial_containment", None):
                worker_out["containment"] = getattr(exc, "partial_containment")
            if getattr(exc, "partial_containment_warning", None):
                worker_out["containment_warning"] = getattr(exc, "partial_containment_warning")
        except Exception as exc:
            with self._lock:
                is_canc = job_id in self._cancelled
            if is_canc or status == "cancelled":
                status = "cancelled"
                error_msg = "Job cancelled by user."
            else:
                status = "error"
                error_msg = f"{type(exc).__name__}: {exc}"
            self._merge_partial_exception_data(exc, usage, tool_calls, thoughts)
            if getattr(exc, "partial_containment", None):
                worker_out["containment"] = getattr(exc, "partial_containment")
            if getattr(exc, "partial_containment_warning", None):
                worker_out["containment_warning"] = getattr(exc, "partial_containment_warning")
        finally:
            try:
                hooks_tampered = _restore_directory_tree(host_git_dir / "hooks", hooks_snap)
                config_tampered = _restore_directory_tree(host_git_dir / "config", config_snap)
                if hooks_tampered or config_tampered:
                    status = "security_violation"
                    error_msg = "Worker attempted to modify host .git/hooks or .git/config; restored."

                if in_place:
                    diff_text, files_changed, symlink_escapes, in_place_notes = self._finalize_in_place(
                        baseline_sha,
                        pre_untracked or {},
                        orig_branch=orig_branch,
                        pre_branches=pre_branches,
                    )
                    baseline_file_hashes = self._compute_changed_baseline_hashes(
                        self.workspace_root, baseline_sha, files_changed, {}
                    )
                else:
                    diff_text, files_changed, symlink_escapes = self._collect_worktree_diff(
                        wt_dir, baseline_sha, allowed_paths=allowed_paths
                    )
                    baseline_file_hashes = self._compute_changed_baseline_hashes(
                        wt_dir, baseline_sha, files_changed, baseline_file_hashes
                    )
            except Exception as fin_exc:
                diff_text, files_changed, symlink_escapes = "", [], []
                if status in ("running", "ok"):
                    status = "error"
                    error_msg = f"Failed to collect worktree diff: {type(fin_exc).__name__}: {fin_exc}"
            runnable_acc_status = (
                status in ("ok", "checkpoint", "needs_input")
                or (status == "timeout" and bool(files_changed))
            )
            effective_acceptance = list(acceptance or [])
            auto_default_acceptance = False
            target_acc_cwd = self.workspace_root if in_place else wt_dir
            if not effective_acceptance and runnable_acc_status and files_changed and self._should_run_default_acceptance():
                effective_acceptance = discover_related_test_commands(target_acc_cwd, files_changed, [])
                if os.environ.get("CC_AGY_AUTO_IMPORT_CHECK", "1").strip().lower() not in ("0", "false", "no", "off"):
                    for icmd in build_import_smoke_command(target_acc_cwd, files_changed):
                        if icmd not in effective_acceptance:
                            effective_acceptance.append(icmd)
                auto_default_acceptance = bool(effective_acceptance)
            if effective_acceptance and runnable_acc_status:
                try:
                    acceptance_results = run_acceptance(effective_acceptance, target_acc_cwd)
                    if auto_default_acceptance:
                        for dr in acceptance_results:
                            dr["auto_discovered_default"] = True
                            if dr.get("exit_code") == 5:
                                dr["passed"] = True
                                dr["no_tests_collected"] = True
                    elif (
                        files_changed
                        and os.environ.get("CC_AGY_AUTO_REGRESSION_TESTS", "1").strip().lower()
                        not in ("0", "false", "no")
                    ):
                        extra_reg_cmds = discover_related_test_commands(
                            target_acc_cwd,
                            files_changed,
                            effective_acceptance,
                        )
                        if extra_reg_cmds:
                            extra_res = run_acceptance(extra_reg_cmds, target_acc_cwd)
                            for er in extra_res:
                                er["auto_discovered_regression"] = True
                            acceptance_results.extend(extra_res)
                    if (
                        not auto_default_acceptance
                        and files_changed
                        and os.environ.get("CC_AGY_AUTO_IMPORT_CHECK", "").strip().lower()
                        in ("1", "true", "yes", "on", "always")
                    ):
                        import_cmds = build_import_smoke_command(target_acc_cwd, files_changed)
                        if import_cmds:
                            imp_res = run_acceptance(import_cmds, target_acc_cwd)
                            for ir in imp_res:
                                ir["auto_import_smoke"] = True
                            acceptance_results.extend(imp_res)
                except Exception as acc_exc:
                    acceptance_results = [
                        {"command": "<acceptance>", "passed": False, "error": f"{type(acc_exc).__name__}: {acc_exc}"}
                    ]
                try:
                    if in_place:
                        diff_text, files_changed, symlink_escapes, in_place_notes = self._finalize_in_place(
                            baseline_sha,
                            pre_untracked or {},
                            orig_branch=orig_branch,
                            pre_branches=pre_branches,
                        )
                        baseline_file_hashes = self._compute_changed_baseline_hashes(
                            self.workspace_root, baseline_sha, files_changed, {}
                        )
                    else:
                        diff_text, files_changed, symlink_escapes = self._collect_worktree_diff(
                            wt_dir, baseline_sha, allowed_paths=allowed_paths
                        )
                        baseline_file_hashes = self._compute_changed_baseline_hashes(
                            wt_dir, baseline_sha, files_changed, baseline_file_hashes
                        )
                except Exception:
                    pass
            if symlink_escapes:
                for esc in symlink_escapes:
                    if esc not in scope_violations:
                        scope_violations.append(esc)
                if status == "ok":
                    status = "scope_violation"
                    error_msg = f"Worker created symlink escaping worktree: {symlink_escapes}"

            if allowed_paths:
                for fpath in files_changed:
                    if not _matches_allowed_paths(fpath, allowed_paths):
                        if fpath not in scope_violations:
                            scope_violations.append(fpath)
                if scope_violations and status == "ok":
                    status = "scope_violation"
                    error_msg = (
                        f"Worker modified files outside allowed_paths {allowed_paths}: "
                        f"{scope_violations}. Review the diff with get_diff and either call "
                        f"delegate_apply(job_id, allow_scope_violations=True) to approve, or call "
                        f"delegate_continue(job_id, follow_up_prompt=..., allowed_paths=...) to adjust scope."
                    )

            provenance = extract_diff_provenance(diff_text)
            for fpath in files_changed:
                if fpath not in provenance["files_changed"]:
                    provenance["files_changed"].append(fpath)

            diff_path = self.store.diff_path(job_id)
            diff_path.write_text(diff_text, encoding="utf-8")

            missing_prompt_symbols = (
                find_missing_prompt_symbols(
                    "\n".join(recorded_prompts) if recorded_prompts else "",
                    diff_text,
                    worktree_root=self.workspace_root if in_place else wt_dir,
                    files_changed=files_changed,
                    baseline_ref=baseline_sha or None,
                )
                if files_changed
                else []
            )

            with self._lock:
                if job_id in self._cancelled:
                    status = "cancelled"
                    error_msg = error_msg or "Cancelled by user via delegate_cancel."

            finished_at = round(time.time(), 3)
            clipped_text, was_truncated, orig_tokens = truncate_to_token_budget(
                response_text, MAX_TOOL_RESPONSE_TOKENS
            )
            receipt_payload = {
                "job_id": job_id,
                "kind": "delegate",
                "status": status,
                "auth_mode": auth_mode,
                "model": model,
                "vertex_project": project,
                "vertex_location": location,
                "reroute_note": reroute_note,
                "prompts": recorded_prompts,
                "allowed_paths": allowed_paths,
                "enabled_tools": enabled_tools,
                "disabled_tools": disabled_tools,
                "enable_subagents": enable_subagents,
                "scrub_receipt": scrub_receipt,
                "scope_violations": scope_violations,
                "symlink_escapes": symlink_escapes,
                "worktree_dir": str(wt_dir),
                "baseline_commit_sha": baseline_sha,
                "baseline_file_hashes": baseline_file_hashes,
                "started_at_epoch": started_at,
                "finished_at_epoch": finished_at,
                "duration_seconds": round(max(0.0, finished_at - started_at), 3),
                "usage": usage,
                "tool_call_count": len(tool_calls),
                "tool_calls": tool_calls,
                "thoughts_count": len(thoughts),
                "summary": clipped_text,
                "summary_truncated": was_truncated,
                "summary_tokens_estimated": orig_tokens,
                "files_changed": files_changed,
                "diff_tokens_estimated": estimate_tokens(diff_text),
                "provenance": provenance,
                "applied": False,
                "error": error_msg,
                "thinking_level": thinking_level,
                "plan": plan or [],
                "acceptance": acceptance or [],
                "budget": {"max_tool_calls": max_tool_calls, "max_total_tokens": max_total_tokens},
                "stop_reason": stop_reason,
                "containment": worker_out.get("containment", "none") if isinstance(worker_out, dict) else "none",
                "containment_warning": (
                    worker_out.get("containment_warning") if isinstance(worker_out, dict) else None
                ),
                "acceptance_results": acceptance_results,
                "acceptance_passed": (
                    all(r.get("passed") for r in acceptance_results) if acceptance_results else None
                ),
                "missing_prompt_symbols": missing_prompt_symbols,
                "external_worktree_tamper": bool(not in_place and files_changed and not workspace_dir_accessed),
                "preloaded_files": list(preloaded_files or []),
            }
            receipt_payload.update(self._escalation_fields(job_id, receipt_payload))
            if in_place:
                # The edits are already in the user's working tree, so the job counts as applied
                # the moment it finishes (even on error or timeout, partial edits are on disk).
                applied_file_hashes = {
                    fpath: _hash_path(self.workspace_root / fpath) for fpath in files_changed
                }
                new_files = [f for f in files_changed if baseline_file_hashes.get(f) is None]
                tracked_files = [f for f in files_changed if baseline_file_hashes.get(f) is not None]
                undo_parts: list[str] = []
                if tracked_files:
                    undo_parts.append(
                        f"git checkout {baseline_sha[:12]} -- "
                        + " ".join(shlex.quote(f) for f in tracked_files)
                    )
                if new_files:
                    undo_parts.append("rm -- " + " ".join(shlex.quote(f) for f in new_files))
                receipt_payload.update(
                    {
                        "in_place": True,
                        "in_place_notes": in_place_notes,
                        "pre_untracked_hashes": pre_untracked or {},
                        "applied": True,
                        "applied_at_epoch": finished_at,
                        "applied_files": list(files_changed),
                        "applied_file_hashes": applied_file_hashes,
                        "undo_hint": " && ".join(undo_parts) if undo_parts else None,
                    }
                )
                if status not in ("ok", "checkpoint", "needs_input") and files_changed:
                    receipt_payload["error"] = (
                        f"{error_msg or status}. Partial in_place edits remain in the working tree; "
                        f"see undo_hint to revert them."
                    )
            try:
                prior_disk_rec = self.store.read_receipt(job_id)
            except Exception:
                prior_disk_rec = {}
            if prior_disk_rec.get("cancelled") or prior_disk_rec.get("status") == "cancelled" or status == "cancelled":
                receipt_payload["status"] = "cancelled"
                receipt_payload["cancelled"] = True
                if not receipt_payload.get("error"):
                    receipt_payload["error"] = prior_disk_rec.get("error") or "Cancelled by user via delegate_cancel."
            elif prior_disk_rec.get("status") == "interrupted":
                receipt_payload["status"] = "interrupted"
                if not receipt_payload.get("error"):
                    receipt_payload["error"] = prior_disk_rec.get("error")
            if prior_disk_rec.get("escalated_from_model") and not receipt_payload.get("escalated_from_model"):
                receipt_payload["escalated_from_model"] = prior_disk_rec["escalated_from_model"]
            if prior_disk_rec.get("applied") and not in_place:
                for preserve_key in (
                    "applied",
                    "applied_at_epoch",
                    "applied_files",
                    "applied_file_hashes",
                    "baseline_file_hashes",
                    "provenance",
                    "pre_apply",
                    "salvaged_on_shutdown",
                ):
                    if preserve_key in prior_disk_rec:
                        receipt_payload[preserve_key] = prior_disk_rec[preserve_key]
            saved = self.store.write_receipt(
                job_id,
                receipt_payload,
                scrub=scrub_receipt,
                workspace_root=str(self.workspace_root),
                api_key=api_key,
            )
            with self._lock:
                if job_id in self._cancelled and wt_dir.exists():
                    self._cleanup_worktree(wt_dir)
            if in_place:
                self._release_in_place_slot(job_id)

        return saved

    @staticmethod
    def _merge_partial_exception_data(
        exc: BaseException,
        usage: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        thoughts: list[str],
    ) -> None:
        partial_u = getattr(exc, "partial_usage", None)
        if isinstance(partial_u, dict):
            for k in (
                "prompt_token_count",
                "cached_content_token_count",
                "candidates_token_count",
                "thoughts_token_count",
                "total_token_count",
            ):
                usage[k] = int(usage.get(k) or 0) + int(partial_u.get(k) or 0)
            usage["total_token_count"] = max(
                usage["total_token_count"],
                usage["prompt_token_count"]
                + usage["candidates_token_count"]
                + usage["thoughts_token_count"],
            )
            partial_reqs = partial_u.get("requests")
            if isinstance(partial_reqs, list) and partial_reqs:
                usage.setdefault("requests", []).extend(partial_reqs)
            partial_max_req = int(partial_u.get("max_request_prompt_tokens") or 0)
            if partial_max_req > 0:
                usage["max_request_prompt_tokens"] = max(
                    int(usage.get("max_request_prompt_tokens") or 0),
                    partial_max_req,
                )
        partial_tc = getattr(exc, "partial_tool_calls", None)
        if isinstance(partial_tc, list):
            tool_calls.extend(partial_tc)
        partial_th = getattr(exc, "partial_thoughts", None)
        if isinstance(partial_th, list):
            thoughts.extend(partial_th)

    def delegate_start(
        self,
        prompt: str,
        *,
        model: str | None = None,
        allowed_paths: list[str] | None = None,
        project: str | None = None,
        location: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 1200.0,
        enabled_tools: list[str] | None = None,
        disabled_tools: list[str] | None = None,
        enable_subagents: bool = False,
        wait: bool = False,
        scrub_receipt: bool = False,
        in_place: bool = False,
        thinking_level: str | None = None,
        plan: list[str] | None = None,
        acceptance: list[str] | None = None,
        max_tool_calls: int | None = None,
        max_total_tokens: int | None = None,
        preload_files: bool = True,
        auto_apply: bool | None = None,
    ) -> dict[str, Any]:
        """Start an Antigravity worker job.

        By default the worker edits an isolated standalone copy of the repo and its diff is
        applied later with delegate_apply (or automatically when auto_apply=True / CC_AGY_AUTO_APPLY_ON_PASS=1).
        With in_place=True the worker edits the real working tree directly (requires a clean tracked tree,
        one in_place job at a time, and no allowed_paths); the job is recorded as applied as soon as it finishes.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if wait and not in_place and self._is_bounded_lead_session():
            delegate_receipts = [
                r
                for r in self.store.list_receipts(workspace_root=self.workspace_root)
                if r.get("kind", "delegate") == "delegate"
            ]
            if delegate_receipts:
                prev_rec = delegate_receipts[-1]
                prev_jid = str(prev_rec.get("job_id") or "")
                prev_wt = prev_rec.get("worktree_dir")
                prev_status = str(prev_rec.get("status") or "")
                with self._lock:
                    prev_is_cancelled = bool(
                        prev_rec.get("cancelled")
                        or prev_status == "cancelled"
                        or prev_jid in self._cancelled
                    )
                prev_continuable = (
                    prev_status in ("timeout", "checkpoint", "needs_input")
                    or (prev_status == "ok" and prev_rec.get("acceptance_passed") is False)
                )
                if (
                    prev_jid
                    and not prev_is_cancelled
                    and prev_continuable
                    and prev_rec.get("files_changed")
                    and not prev_rec.get("applied")
                    and not (prev_rec.get("escalation") or {}).get("escalate_to_lead")
                    and prev_wt
                    and Path(prev_wt).is_dir()
                ):
                    return self.delegate_continue(
                        job_id=prev_jid,
                        follow_up_prompt=prompt,
                        allowed_paths=allowed_paths,
                        acceptance=acceptance,
                        timeout_seconds=timeout_seconds,
                        wait=True,
                        project=project,
                        location=location,
                        api_key=api_key,
                        scrub_receipt=scrub_receipt,
                        auto_apply=auto_apply,
                    )
        timeout_seconds = self._clamp_worker_timeout(timeout_seconds)

        validated_paths = validate_allowed_paths(allowed_paths)
        if in_place and validated_paths:
            raise ValueError(
                "allowed_paths cannot be enforced in in_place mode (edits land directly in your "
                "files). Drop allowed_paths or use the default isolated copy."
            )
        req_model_clean = (model or "").strip()
        if req_model_clean and not req_model_clean.startswith("${"):
            chosen_model = req_model_clean
        else:
            chosen_model = first_valid_env(
                os.environ,
                "CC_AGY_DEFAULT_DELEGATE_MODEL",
                "CLAUDE_PLUGIN_OPTION_DEFAULT_DELEGATE_MODEL",
                "CC_AGY_USERCFG_DEFAULT_DELEGATE_MODEL",
                default="gemini-3.1-pro-preview",
            )
        resolved_thinking = resolve_thinking_level(
            thinking_level,
            model=chosen_model,
            read_only=is_read_only_tool_set(enabled_tools, disabled_tools),
        )
        plan_steps = validate_plan(plan)
        acceptance_cmds = validate_acceptance(acceptance, self.workspace_root)
        budget_calls, budget_tokens = resolve_budget(max_tool_calls, max_total_tokens)
        preloaded_pairs: list[tuple[str, str]] = (
            select_candidate_files(self.workspace_root, prompt) if preload_files else []
        )
        preloaded_rel_paths = [rel for rel, _ in preloaded_pairs]
        use_repo_map = os.environ.get("CC_AGY_REPO_MAP", "1").strip().lower() not in ("0", "false", "no")
        repo_map_text = (
            build_repo_map(self.workspace_root, cache_dir=self.store.data_dir)
            if use_repo_map
            else ""
        )
        focus_targets = list(dict.fromkeys([*(validated_paths or []), *preloaded_rel_paths]))
        if not focus_targets and use_repo_map:
            focus_targets = [rel for rel, _ in select_candidate_files(self.workspace_root, prompt, max_files=6)]
        focused_outline_text = (
            build_focused_file_outline(self.workspace_root, focus_targets, max_tokens=500)
            if use_repo_map and focus_targets
            else ""
        )
        worker_prompt = render_worker_prompt(
            prompt,
            plan=plan_steps,
            acceptance=acceptance_cmds,
            max_tool_calls=budget_calls,
            max_total_tokens=budget_tokens,
            preloaded_files=preloaded_pairs,
            repo_map=repo_map_text,
            focused_outline=focused_outline_text,
        )

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

        job_id = f"job-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        started_at = round(time.time(), 3)
        pre_untracked: dict[str, str | None] = {}
        if in_place:
            self._acquire_in_place_slot(job_id)
            try:
                baseline_sha, pre_untracked = self._check_in_place_ready()
            except Exception:
                self._release_in_place_slot(job_id)
                raise
            wt_dir, baseline_hashes = self.workspace_root, {}
        else:
            wt_dir, baseline_sha, baseline_hashes = self._prepare_worktree(job_id)

        with self._lock:
            if resolved_project:
                self._raw_projects[job_id] = resolved_project
            if resolved_key:
                self._raw_api_keys[job_id] = resolved_key

        initial_receipt = {
            "job_id": job_id,
            "kind": "delegate",
            "status": "running",
            "auth_mode": auth_mode,
            "model": chosen_model,
            "vertex_project": resolved_project,
            "vertex_location": resolved_location,
            "reroute_note": reroute_note,
            "prompts": [prompt.strip()],
            "allowed_paths": validated_paths,
            "enabled_tools": enabled_tools,
            "disabled_tools": disabled_tools,
            "enable_subagents": enable_subagents,
            "scrub_receipt": scrub_receipt,
            "scope_violations": [],
            "worktree_dir": str(wt_dir),
            "baseline_commit_sha": baseline_sha,
            "baseline_file_hashes": baseline_hashes,
            "started_at_epoch": started_at,
            "usage": {
                "prompt_token_count": 0,
                "cached_content_token_count": 0,
                "candidates_token_count": 0,
                "thoughts_token_count": 0,
                "total_token_count": 0,
            },
            "tool_call_count": 0,
            "tool_calls": [],
            "files_changed": [],
            "applied": False,
            "in_place": in_place,
            "thinking_level": resolved_thinking or "DEFAULT",
            "plan": plan_steps,
            "acceptance": acceptance_cmds,
            "budget": {"max_tool_calls": budget_calls, "max_total_tokens": budget_tokens},
            "preloaded_files": preloaded_rel_paths,
            "error": None,
        }
        self.store.write_receipt(
            job_id,
            initial_receipt,
            scrub=scrub_receipt,
            workspace_root=str(self.workspace_root),
            api_key=resolved_key,
        )

        fut = self._executor.submit(
            self._execute_job,
            job_id=job_id,
            worker_prompts=[worker_prompt],
            recorded_prompts=[prompt.strip()],
            model=chosen_model,
            auth_mode=auth_mode,
            api_key=resolved_key,
            project=resolved_project,
            location=resolved_location,
            reroute_note=reroute_note,
            allowed_paths=validated_paths,
            timeout_seconds=timeout_seconds,
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
            enable_subagents=enable_subagents,
            wt_dir=wt_dir,
            baseline_sha=baseline_sha,
            baseline_file_hashes=baseline_hashes,
            started_at=started_at,
            scrub_receipt=scrub_receipt,
            in_place=in_place,
            pre_untracked=pre_untracked,
            thinking_level=resolved_thinking or "DEFAULT",
            plan=plan_steps,
            acceptance=acceptance_cmds,
            max_tool_calls=budget_calls,
            max_total_tokens=budget_tokens,
            preloaded_files=preloaded_rel_paths,
        )
        with self._lock:
            self._futures[job_id] = fut

        if wait:
            self._wait_for_future_or_cancel(job_id, fut, timeout=max(0.5, float(timeout_seconds) + 30.0))
            receipt = self.store.read_receipt(job_id)
            digest = self._digest_with_progress(receipt)
            return self._maybe_auto_apply_digest(
                job_id,
                digest,
                auto_apply=auto_apply,
                in_place=in_place,
            )

        return {
            "job_id": job_id,
            "status": "running",
            "auth_mode": auth_mode,
            "model": chosen_model,
            "vertex_location": resolved_location,
            "reroute_note": reroute_note,
            "allowed_paths": validated_paths,
            "worktree_dir": str(wt_dir),
            "in_place": in_place,
        }

    def _wait_for_future_or_cancel(
        self,
        job_id: str,
        fut: Future[dict[str, Any]],
        *,
        timeout: float,
    ) -> None:
        """Wait for `fut` to complete or for `job_id` to be added to `self._cancelled`."""
        deadline = time.monotonic() + max(0.1, float(timeout))
        while not fut.done():
            with self._lock:
                if job_id in self._cancelled:
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                fut.result(timeout=min(0.05, remaining))
                return
            except (TimeoutError, FuturesTimeoutError):
                continue
            except (Exception, asyncio.CancelledError):
                return

    def delegate_status(
        self,
        job_id: str | None = None,
        *,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return status for a single job or list recent jobs within the MCP token cap."""
        baseline_rec = self.store.read_baseline_record(self.workspace_root) or {}
        active_baseline_sha = baseline_rec.get("baseline_head_sha")
        if job_id:
            safe_id = validate_job_id(job_id)
            receipt = self.store.read_receipt(safe_id)
            digest = self._digest_with_progress(receipt, include_summary=True)
            digest["baseline_head_sha"] = active_baseline_sha
            return digest

        all_receipts = self.store.list_receipts(workspace_root=self.workspace_root)
        max_items = max(1, min(int(limit), 50))
        recent = all_receipts[-max_items:]
        totals = self.store.summarize_costs(workspace_root=self.workspace_root)
        summaries: list[dict[str, Any]] = []
        for r in reversed(recent):
            item = self._compact_digest(r, include_summary=False, include_acceptance_details=False)
            candidate_list = [item] + summaries
            probe = {
                "jobs": candidate_list,
                "returned_jobs": len(candidate_list),
                "total_jobs": len(all_receipts),
                "baseline_head_sha": active_baseline_sha,
                "totals": totals,
            }
            if summaries and estimate_tokens(json.dumps(probe, indent=2, sort_keys=True)) > MAX_TOOL_RESPONSE_TOKENS - 200:
                break
            summaries.insert(0, item)
        return {
            "jobs": summaries,
            "returned_jobs": len(summaries),
            "total_jobs": len(all_receipts),
            "baseline_head_sha": active_baseline_sha,
            "totals": totals,
        }

    def delegate_result(
        self,
        job_id: str,
        *,
        wait: bool = True,
        timeout: float = 240.0,
    ) -> dict[str, Any]:
        """Return the compact completion digest for job_id, optionally waiting if still running."""
        safe_id = validate_job_id(job_id)
        with self._lock:
            fut = self._futures.get(safe_id)
        if fut is not None and wait and not fut.done():
            capped_timeout = max(0.1, min(float(timeout), 540.0))
            self._wait_for_future_or_cancel(safe_id, fut, timeout=capped_timeout)
        receipt = self.store.read_receipt(safe_id)
        return self._digest_with_progress(receipt)

    @staticmethod
    def _filter_diff_by_symbols(diff_text: str, focus_symbols: list[str]) -> str:
        clean_syms = [str(s).strip() for s in focus_symbols if isinstance(s, str) and str(s).strip()]
        if not clean_syms or not diff_text:
            return diff_text
        sym_patterns = [
            re.compile(rf"(?<![A-Za-z0-9_]){re.escape(sym)}(?![A-Za-z0-9_])")
            for sym in clean_syms
        ]
        matched_files: list[str] = []
        file_chunks: list[list[str]] = []
        cur_file_lines: list[str] = []
        for line in diff_text.splitlines(keepends=True):
            if line.startswith("diff --git "):
                if cur_file_lines:
                    file_chunks.append(cur_file_lines)
                cur_file_lines = [line]
            else:
                cur_file_lines.append(line)
        if cur_file_lines:
            file_chunks.append(cur_file_lines)
        for chunk in file_chunks:
            header_lines: list[str] = []
            hunks: list[list[str]] = []
            cur_hunk: list[str] = []
            for ln in chunk:
                if ln.startswith("@@ "):
                    if cur_hunk:
                        hunks.append(cur_hunk)
                    cur_hunk = [ln]
                elif cur_hunk:
                    cur_hunk.append(ln)
                else:
                    header_lines.append(ln)
            if cur_hunk:
                hunks.append(cur_hunk)
            kept_hunks = [
                "".join(h)
                for h in hunks
                if any(pat.search("".join(h[1:]) if len(h) > 1 else "".join(h)) for pat in sym_patterns)
            ]
            if kept_hunks:
                matched_files.append("".join(header_lines) + "".join(kept_hunks))
            elif not hunks and any(pat.search("".join(header_lines)) for pat in sym_patterns):
                matched_files.append("".join(header_lines))
        return "".join(matched_files)

    def get_diff(
        self,
        job_id: str,
        *,
        file_filter: str | None = None,
        page: int = 1,
        page_tokens: int = DEFAULT_DIFF_PAGE_TOKENS,
        summary_only: bool = False,
        focus_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return paginated unified git diff for job_id measured in tokens, with per-file line stats
        and optional summary-only or symbol-focused hunk filtering."""
        safe_id = validate_job_id(job_id)
        receipt = self.store.read_receipt(safe_id)
        diff_path = self.store.diff_path(safe_id)
        full_diff = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""

        if file_filter and full_diff:
            filtered_chunks: list[str] = []
            current_chunk: list[str] = []
            keep_chunk = False
            in_hunk = False
            for line in full_diff.splitlines(keepends=True):
                if line.startswith("diff --git "):
                    if current_chunk and keep_chunk:
                        filtered_chunks.append("".join(current_chunk))
                    current_chunk = [line]
                    hdr = parse_diff_git_header(line)
                    cand_path = (hdr[1] or hdr[0]) if hdr else ""
                    keep_chunk = bool(cand_path and _matches_file_filter(cand_path, file_filter))
                    in_hunk = False
                else:
                    if line.startswith("@@ "):
                        in_hunk = True
                    elif not in_hunk and line.startswith("+++ "):
                        cand_path = parse_unified_header_path(line, "+++")
                        if cand_path and cand_path != "/dev/null" and _matches_file_filter(cand_path, file_filter):
                            keep_chunk = True
                    elif not in_hunk and line.startswith("--- "):
                        cand_path = parse_unified_header_path(line, "---")
                        if cand_path and cand_path != "/dev/null" and _matches_file_filter(cand_path, file_filter):
                            keep_chunk = True
                    current_chunk.append(line)
            if current_chunk and keep_chunk:
                filtered_chunks.append("".join(current_chunk))
            full_diff = "".join(filtered_chunks)

        if focus_symbols and full_diff:
            full_diff = self._filter_diff_by_symbols(full_diff, focus_symbols)

        prov_per_file = (extract_diff_provenance(full_diff).get("per_file") or {}) if full_diff else {}
        file_stats = {
            fpath: {
                "added": int(meta.get("added_count") or 0),
                "removed": int(meta.get("removed_count") or 0),
            }
            for fpath, meta in prov_per_file.items()
        }

        pages = paginate_text_by_tokens(full_diff, page_tokens=page_tokens)
        total_pages = len(pages)
        clamped_page = max(1, min(int(page), total_pages))
        page_slice = "" if summary_only else pages[clamped_page - 1]
        effective_files = (
            list(file_stats.keys())
            if (file_filter or focus_symbols)
            else receipt.get("files_changed", [])
        )

        out: dict[str, Any] = {
            "job_id": safe_id,
            "status": receipt.get("status"),
            "file_filter": file_filter,
            "page": clamped_page,
            "total_pages": total_pages,
            "page_tokens_estimated": estimate_tokens(page_slice),
            "total_diff_tokens_estimated": estimate_tokens(full_diff),
            "files_changed": effective_files,
            "file_stats": file_stats,
            "diff": page_slice,
        }
        if summary_only:
            out["summary_only"] = True
        if focus_symbols:
            out["focus_symbols"] = list(focus_symbols)
        return out

    def delegate_apply(
        self,
        job_id: str,
        *,
        cleanup_worktree: bool = True,
        allow_scope_violations: bool = False,
    ) -> dict[str, Any]:
        """Apply a verified worker job's changes to the main workspace with serialized conflict detection."""
        safe_id = validate_job_id(job_id)
        with self._lock:
            fut = self._futures.get(safe_id)
        if fut is not None and not fut.done():
            raise RuntimeError(f"Job '{safe_id}' is still running; wait for completion before applying.")

        lock_file_path = self.store.data_dir / ".apply.lock"
        with self._apply_lock:
            with open(lock_file_path, "w", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    return self._delegate_apply_locked(
                        safe_id,
                        cleanup_worktree=cleanup_worktree,
                        allow_scope_violations=allow_scope_violations,
                    )
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _delegate_apply_locked(
        self,
        safe_id: str,
        *,
        cleanup_worktree: bool = True,
        allow_scope_violations: bool = False,
    ) -> dict[str, Any]:
        receipt = self.store.read_receipt(safe_id)
        if receipt.get("applied"):
            prov = receipt.get("provenance") or {}
            return {
                "job_id": safe_id,
                "applied": True,
                "already_applied": True,
                "applied_files": receipt.get("applied_files", []),
                "diff_sha256": prov.get("diff_sha256"),
            }

        status = receipt.get("status")
        wt_dir = Path(str(receipt.get("worktree_dir") or ""))
        baseline_sha = str(receipt.get("baseline_commit_sha") or "HEAD")
        if status == "running" and self._futures.get(safe_id) is None and wt_dir.exists():
            rec_diff, rec_files, rec_escapes = self._collect_worktree_diff(wt_dir, baseline_sha)
            if rec_files and not rec_escapes:
                self.store.diff_path(safe_id).write_text(rec_diff, encoding="utf-8")
                receipt["status"] = "timeout"
                receipt["files_changed"] = rec_files
                receipt["symlink_escapes"] = rec_escapes
                status = "timeout"
        has_clean_timeout_edits = (
            status in ("timeout", "interrupted")
            and bool(receipt.get("files_changed"))
            and (not receipt.get("scope_violations") or allow_scope_violations)
            and not receipt.get("symlink_escapes")
        )
        has_approved_scope_override = (
            status == "scope_violation"
            and allow_scope_violations
            and not receipt.get("symlink_escapes")
        )
        if (
            status not in ("ok", "checkpoint", "needs_input")
            and not has_clean_timeout_edits
            and not has_approved_scope_override
        ):
            raise ValueError(
                f"Cannot apply job '{safe_id}' with status={status!r} "
                f"(error={receipt.get('error')!r}). Only ok / checkpoint / needs_input jobs "
                f"(or timeout jobs with clean edits, or scope_violation jobs with allow_scope_violations=True) can be applied."
            )

        diff_path = self.store.diff_path(safe_id)
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""
        raw_files_changed: list[str] = list(receipt.get("files_changed") or [])
        files_changed = [
            f
            for f in raw_files_changed
            if not _is_ephemeral_build_or_test_artifact(
                f, root=wt_dir if wt_dir.exists() else self.workspace_root
            )
        ]
        allowed_paths: list[str] = validate_allowed_paths(receipt.get("allowed_paths") or [])
        baseline_hashes: dict[str, str | None] = self._compute_changed_baseline_hashes(
            wt_dir,
            baseline_sha,
            files_changed,
            dict(receipt.get("baseline_file_hashes") or {}),
        )

        # 1. Pre-flight validation: scope, path traversal, symlinks, and baseline conflict check
        conflicted_files: list[str] = []
        for rel_path in files_changed:
            if not allow_scope_violations and not _matches_allowed_paths(rel_path, allowed_paths):
                raise ValueError(
                    f"Refusing to apply out-of-scope file {rel_path!r} (allowed_paths={allowed_paths}). "
                    f"Pass allow_scope_violations=True to delegate_apply if you have reviewed the diff and want to apply it."
                )
            dst_raw = self.workspace_root / rel_path
            dst_resolved = dst_raw.resolve()
            if not dst_resolved.is_relative_to(self.workspace_root):
                raise ValueError(f"Path traversal blocked for {rel_path!r}")

            if wt_dir.exists():
                src_raw = wt_dir / rel_path
                if src_raw.is_symlink():
                    link_target = (src_raw.parent / os.readlink(src_raw)).resolve()
                    if not link_target.is_relative_to(wt_dir.resolve()):
                        raise ValueError(
                            f"Refusing to apply symlink {rel_path!r} pointing outside worktree"
                        )

            expected_baseline_sha = baseline_hashes.get(rel_path)
            if dst_raw.exists() and dst_raw.is_file() and not dst_raw.is_symlink():
                current_host_sha = sha256_bytes(dst_raw.read_bytes())
                if expected_baseline_sha != current_host_sha:
                    # If the worker file already has the exact same bytes as the host file, no merge is needed.
                    if wt_dir.exists():
                        src_raw = wt_dir / rel_path
                        if src_raw.exists() and src_raw.is_file() and not src_raw.is_symlink():
                            if sha256_bytes(src_raw.read_bytes()) == current_host_sha:
                                continue
                    conflicted_files.append(rel_path)
            elif expected_baseline_sha is not None and not dst_raw.exists():
                conflicted_files.append(rel_path)

        # 2. If host files changed since baseline, attempt in-memory 3-way merge via `git merge-file -p`
        # without ever touching the host repository's git index.
        merged_payloads: dict[str, bytes] = {}
        if conflicted_files:
            if not wt_dir.exists():
                raise ValueError(
                    f"Conflict detected: host files {conflicted_files} were modified since job '{safe_id}' started."
                )
            for rel_path in conflicted_files:
                dst_raw = self.workspace_root / rel_path
                src_raw = wt_dir / rel_path
                if not (dst_raw.exists() and dst_raw.is_file() and src_raw.exists() and src_raw.is_file()):
                    raise ValueError(
                        f"Conflict detected: host file {rel_path!r} was modified or deleted since job '{safe_id}' started."
                    )
                base_show = subprocess.run(
                    ["git", "show", f"{baseline_sha}:{rel_path}"],
                    cwd=str(wt_dir),
                    capture_output=True,
                    check=False,
                )
                if base_show.returncode != 0:
                    raise ValueError(
                        f"Conflict detected: host file {rel_path!r} was concurrently created on both host and worker."
                    )
                with tempfile.TemporaryDirectory(prefix="cc-agy-merge-") as mdir:
                    mpath = Path(mdir)
                    f_cur = mpath / "current"
                    f_base = mpath / "base"
                    f_wrk = mpath / "worker"
                    f_cur.write_bytes(dst_raw.read_bytes())
                    f_base.write_bytes(base_show.stdout)
                    f_wrk.write_bytes(src_raw.read_bytes())
                    merge_res = subprocess.run(
                        ["git", "merge-file", "-p", str(f_cur), str(f_base), str(f_wrk)],
                        capture_output=True,
                        check=False,
                    )
                    if merge_res.returncode != 0:
                        raise ValueError(
                            f"Conflict detected: host files {conflicted_files} were modified since job "
                            f"'{safe_id}' started and could not be cleanly merged."
                        )
                    merged_payloads[rel_path] = merge_res.stdout

        # Record each target's pre-apply state (bytes backed up under data_dir/backups/<job_id>/) so
        # delegate_revert can restore the workspace exactly, including uncommitted host edits.
        pre_apply = self._snapshot_pre_apply(safe_id, files_changed)

        applied_files: list[str] = []
        if wt_dir.exists():
            for rel_path in files_changed:
                src_raw = wt_dir / rel_path
                dst_raw = self.workspace_root / rel_path
                if rel_path in merged_payloads:
                    dst_raw.parent.mkdir(parents=True, exist_ok=True)
                    dst_raw.write_bytes(merged_payloads[rel_path])
                    applied_files.append(rel_path)
                elif src_raw.is_symlink():
                    dst_raw.parent.mkdir(parents=True, exist_ok=True)
                    if dst_raw.exists() or dst_raw.is_symlink():
                        dst_raw.unlink()
                    os.symlink(os.readlink(src_raw), dst_raw)
                    applied_files.append(rel_path)
                elif src_raw.exists() and src_raw.is_file():
                    dst_raw.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_raw, dst_raw)
                    applied_files.append(rel_path)
                elif not src_raw.exists():
                    if dst_raw.exists() or dst_raw.is_symlink():
                        dst_raw.unlink()
                    applied_files.append(rel_path)
        elif diff_text.strip():
            subprocess.run(
                ["git", "apply", "--binary", "--whitespace=nowarn", "-"],
                input=diff_text,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                check=True,
            )
            applied_files = list(files_changed)

        if cleanup_worktree and wt_dir.exists():
            self._cleanup_worktree(wt_dir)

        # Compute exact provenance and post-apply SHA-256 hashes of the applied files
        applied_prov = extract_diff_provenance(diff_text)
        applied_file_hashes: dict[str, str | None] = {}
        for fpath in applied_files:
            if fpath not in applied_prov["files_changed"]:
                applied_prov["files_changed"].append(fpath)
            dst_p = self.workspace_root / fpath
            if dst_p.is_symlink():
                applied_file_hashes[fpath] = f"symlink:{os.readlink(dst_p)}"
            elif dst_p.exists() and dst_p.is_file():
                applied_file_hashes[fpath] = sha256_bytes(dst_p.read_bytes())
            else:
                applied_file_hashes[fpath] = None

        receipt["applied"] = True
        receipt["applied_at_epoch"] = round(time.time(), 3)
        receipt["applied_files"] = applied_files
        receipt["applied_file_hashes"] = applied_file_hashes
        receipt["baseline_file_hashes"] = baseline_hashes
        receipt["provenance"] = applied_prov
        receipt["pre_apply"] = {k: v for k, v in pre_apply.items() if k in applied_files}
        self.store.write_receipt(
            safe_id,
            receipt,
            scrub=bool(receipt.get("scrub_receipt", False)),
            workspace_root=str(self.workspace_root),
        )

        return {
            "job_id": safe_id,
            "applied": True,
            "applied_files": applied_files,
            "diff_sha256": applied_prov["diff_sha256"],
            "revert_hint": f"delegate_revert(job_id={safe_id!r}) restores these files to their pre-apply state.",
        }

    def _escalation_fields(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Count consecutive failed turns and, at the threshold, mark the job for Lead (Opus) takeover."""
        try:
            prior = self.store.read_receipt(job_id)
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
            prior = {}
        prior_failed = int(prior.get("failed_attempts") or 0)
        failed = worker_turn_failed(payload.get("status"), payload.get("acceptance_passed"))
        count = prior_failed + 1 if failed else 0
        out: dict[str, Any] = {"failed_attempts": count}
        threshold = escalate_after_failures()
        if threshold and count >= threshold:
            out["escalation"] = {
                "escalate_to_lead": True,
                "failed_attempts": count,
                "threshold": threshold,
                "last_status": payload.get("status"),
                "last_acceptance_passed": payload.get("acceptance_passed"),
                "recommendation": (
                    f"The Gemini worker failed {count} attempts in a row. Stop delegating this task: as the Lead, "
                    f"inspect get_diff(job_id={job_id!r}), then finish it yourself (or delegate_cancel and start over). "
                    "delegate_continue refuses further Gemini attempts on this job unless force_retry=true."
                ),
            }
        return out

    def _backup_dir(self, safe_id: str) -> Path:
        d = self.store.data_dir / "backups" / safe_id
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        return d

    def _snapshot_pre_apply(self, safe_id: str, rel_paths: list[str]) -> dict[str, dict[str, Any]]:
        """Back up the host bytes of every file an apply is about to touch."""
        snap: dict[str, dict[str, Any]] = {}
        bdir: Path | None = None
        for rel in rel_paths:
            p = self.workspace_root / rel
            if p.is_symlink():
                snap[rel] = {"state": "symlink", "target": os.readlink(p), "sha256": f"symlink:{os.readlink(p)}"}
            elif p.exists() and p.is_file():
                data = p.read_bytes()
                bdir = bdir or self._backup_dir(safe_id)
                name = sha256_bytes(rel.encode("utf-8"))[:24]
                (bdir / name).write_bytes(data)
                snap[rel] = {"state": "file", "sha256": sha256_bytes(data), "backup": name,
                             "mode": p.stat().st_mode & 0o7777}
            else:
                snap[rel] = {"state": "absent", "sha256": None}
        return snap

    def delegate_revert(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
        skip_dirty: bool = False,
    ) -> dict[str, Any]:
        """Restore every file an applied job touched to its pre-apply state.

        Refuses (or skips with skip_dirty=True) any file whose current content no longer matches
        what the job applied, so later human or agent edits are never silently overwritten.
        """
        safe_id = validate_job_id(job_id)
        lock_file_path = self.store.data_dir / ".apply.lock"
        with self._apply_lock:
            with open(lock_file_path, "w", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    return self._delegate_revert_locked(safe_id, dry_run=dry_run, skip_dirty=skip_dirty)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _pre_state_for_in_place(self, receipt: dict[str, Any], rel: str) -> dict[str, Any]:
        base_hash = (receipt.get("baseline_file_hashes") or {}).get(rel)
        if base_hash is None:
            return {"state": "absent", "sha256": None}
        sha = str(receipt.get("baseline_commit_sha") or "HEAD")
        show = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=str(self.workspace_root),
                              capture_output=True, check=False)
        if show.returncode != 0 or sha256_bytes(show.stdout) != base_hash:
            return {"state": "unknown", "sha256": base_hash}
        return {"state": "git_blob", "sha256": base_hash, "data": show.stdout}

    def _delegate_revert_locked(self, safe_id: str, *, dry_run: bool, skip_dirty: bool) -> dict[str, Any]:
        receipt = self.store.read_receipt(safe_id)
        if not self.store.verify_receipt_signature(receipt):
            raise ValueError(f"Receipt for job '{safe_id}' failed its HMAC check; refusing to revert from it.")
        if not receipt.get("applied"):
            raise ValueError(f"Job '{safe_id}' was never applied; nothing to revert (use delegate_cancel).")
        if receipt.get("reverted"):
            return {"job_id": safe_id, "reverted": True, "already_reverted": True,
                    "reverted_files": receipt.get("reverted_files", [])}
        applied_hashes: dict[str, str | None] = dict(receipt.get("applied_file_hashes") or {})
        pre_apply: dict[str, dict[str, Any]] = dict(receipt.get("pre_apply") or {})
        in_place = bool(receipt.get("in_place"))
        bdir = self.store.data_dir / "backups" / safe_id

        plan: list[dict[str, Any]] = []
        dirty: list[dict[str, Any]] = []
        for rel in list(receipt.get("applied_files") or []):
            dst = self.workspace_root / rel
            if not dst.resolve().is_relative_to(self.workspace_root.resolve()) and not dst.is_symlink():
                raise ValueError(f"Path traversal blocked for {rel!r}")
            current = _hash_path(dst)
            expected = applied_hashes.get(rel)
            pre = pre_apply.get(rel) or (self._pre_state_for_in_place(receipt, rel) if in_place else {"state": "unknown"})
            if current != expected:
                diff = ""
                if dst.is_file() and not dst.is_symlink():
                    diff = f"current sha256={current}, applied sha256={expected}"
                dirty.append({"file": rel, "reason": "modified since apply", "detail": diff})
                continue
            if pre.get("state") == "unknown":
                dirty.append({"file": rel, "reason": "pre-apply content not recorded (older receipt or dirty in_place baseline)"})
                continue
            plan.append({"file": rel, "restore_to": pre.get("state"), "pre": pre})

        if dirty and not skip_dirty:
            return {
                "job_id": safe_id,
                "reverted": False,
                "refused": True,
                "dirty_files": dirty,
                "would_restore": [{"file": x["file"], "restore_to": x["restore_to"]} for x in plan],
                "message": "Some files changed after this job was applied. Nothing was written. "
                           "Pass skip_dirty=True to revert only the clean files.",
            }
        if dry_run:
            return {
                "job_id": safe_id,
                "dry_run": True,
                "would_restore": [{"file": x["file"], "restore_to": x["restore_to"]} for x in plan],
                "skipped_dirty": dirty,
            }

        reverted: list[str] = []
        for item in plan:
            rel, pre = item["file"], item["pre"]
            dst = self.workspace_root / rel
            state = pre.get("state")
            if dst.is_symlink() or (dst.exists() and state in ("absent", "symlink")):
                dst.unlink()
            if state == "absent":
                pass
            elif state == "symlink":
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(pre.get("target")), dst)
            elif state == "file":
                data = (bdir / str(pre.get("backup"))).read_bytes()
                if sha256_bytes(data) != pre.get("sha256"):
                    raise ValueError(f"Backup for {rel!r} is corrupt (hash mismatch); refusing to restore it.")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                if pre.get("mode") is not None:
                    os.chmod(dst, int(pre["mode"]))
            elif state == "git_blob":
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(pre["data"])
            reverted.append(rel)

        receipt["reverted"] = True
        receipt["reverted_at_epoch"] = round(time.time(), 3)
        receipt["reverted_files"] = reverted
        receipt["revert_skipped"] = dirty
        self.store.write_receipt(
            safe_id,
            receipt,
            scrub=bool(receipt.get("scrub_receipt", False)),
            workspace_root=str(self.workspace_root),
        )
        return {"job_id": safe_id, "reverted": True, "reverted_files": reverted, "skipped_dirty": dirty}

    def _should_auto_apply(self, auto_apply: bool | None) -> bool:
        if auto_apply is not None:
            return bool(auto_apply)
        raw_auto = os.environ.get("CC_AGY_AUTO_APPLY_ON_PASS", "").strip().lower()
        if raw_auto in ("0", "false", "no", "off"):
            return False
        if raw_auto in ("1", "true", "yes", "on"):
            return True
        for key in (
            "CLAUDE_PLUGIN_OPTION_STRICT_DELEGATION_MODE",
            "CC_AGY_STRICT_MODE",
            "CC_AGY_USERCFG_STRICT_MODE",
        ):
            if os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on"):
                return True
        return False

    def _maybe_auto_apply_digest(
        self,
        job_id: str,
        digest: dict[str, Any],
        *,
        auto_apply: bool | None,
        in_place: bool,
    ) -> dict[str, Any]:
        if not self._should_auto_apply(auto_apply) or in_place or digest.get("applied"):
            return digest
        escalated = bool((digest.get("escalation") or {}).get("escalate_to_lead"))
        if escalated:
            digest["auto_apply_skipped"] = "escalated to the Lead after repeated failed attempts"
            return digest
        if digest.get("acceptance_passed") is False:
            digest["auto_apply_skipped"] = "acceptance checks failed"
            return digest
        qa_info = digest.get("qa_review")
        if isinstance(qa_info, dict) and qa_info.get("passed") is False and not qa_info.get("resumed"):
            digest["auto_apply_skipped"] = "qa_review reported NEEDS_FIXES without a fix turn"
            return digest
        if digest.get("missing_prompt_symbols"):
            digest["auto_apply_skipped"] = f"missing prompt symbols: {digest['missing_prompt_symbols']}"
            return digest
        if (
            digest.get("status") == "ok"
            and digest.get("files_changed")
            and digest.get("acceptance_passed") is not False
            and not digest.get("scope_violations")
        ):
            apply_res = self.delegate_apply(job_id)
            digest["applied"] = True
            digest["applied_files"] = apply_res.get("applied_files", [])
            digest["diff_sha256"] = apply_res.get("diff_sha256")
            digest.pop("next_action", None)
        return digest

    def delegate_continue(
        self,
        job_id: str,
        follow_up_prompt: str,
        *,
        allowed_paths: list[str] | None = None,
        acceptance: list[str] | None = None,
        timeout_seconds: float = 600.0,
        wait: bool = True,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        force_retry: bool = False,
        scrub_receipt: bool | None = None,
        auto_apply: bool | None = None,
    ) -> dict[str, Any]:
        """Send a follow-up prompt into the same job's isolated worktree without re-running prior prompts."""
        safe_id = validate_job_id(job_id)
        if not follow_up_prompt or not follow_up_prompt.strip():
            raise ValueError("follow_up_prompt must be a non-empty string")
        timeout_seconds = self._clamp_worker_timeout(timeout_seconds)
        try:
            kind_probe = self.store.read_receipt(safe_id).get("kind")
        except FileNotFoundError:
            kind_probe = None
        if kind_probe in ("file_batch", "draft_document", "write_tests_from_cases", "replicate_edit"):
            raise ValueError(
                f"Job '{safe_id}' is a batch job; re-send it with delegate_file_batch(resume_job_id={safe_id!r}) "
                "or fix the failed files yourself. delegate_continue only drives agent jobs."
            )
        if not force_retry:
            try:
                prior_receipt = self.store.read_receipt(safe_id)
            except FileNotFoundError:
                prior_receipt = {}
            if (prior_receipt.get("escalation") or {}).get("escalate_to_lead"):
                digest = self._compact_digest(prior_receipt, include_summary=False)
                digest["status"] = "escalated"
                digest["continued"] = False
                return digest

        with self._lock:
            fut = self._futures.get(safe_id)
            if fut is not None and not fut.done():
                raise RuntimeError(f"Job '{safe_id}' is currently running; wait before continuing.")
            self._cancelled.discard(safe_id)

        receipt = self.store.read_receipt(safe_id)
        in_place = bool(receipt.get("in_place"))
        pre_untracked: dict[str, str | None] = {}
        if in_place:
            baseline_sha = str(receipt.get("baseline_commit_sha") or "HEAD")
            stray_untracked = self._verify_in_place_unchanged(receipt)
            self._acquire_in_place_slot(safe_id)
            wt_dir = self.workspace_root
            baseline_hashes: dict[str, str | None] = {}
            pre_untracked = dict(receipt.get("pre_untracked_hashes") or {})
            # New untracked files that someone else created (e.g. a test run's artifacts) are treated
            # as pre-existing: never credited to the worker, and the Stop hook still checks them.
            pre_untracked.update(stray_untracked)
        else:
            if receipt.get("applied"):
                raise ValueError(
                    f"Job '{safe_id}' has already been applied to the workspace. "
                    f"Start a new job with delegate_start to preserve applied provenance."
                )

            wt_dir = Path(str(receipt.get("worktree_dir") or ""))
            baseline_sha = str(receipt.get("baseline_commit_sha") or "HEAD")
            baseline_hashes = dict(receipt.get("baseline_file_hashes") or {})
            if not wt_dir.exists():
                wt_dir, baseline_sha, baseline_hashes = self._prepare_worktree(safe_id)

        if receipt.get("cancelled") or receipt.get("status") in ("cancelled", "interrupted"):
            # A new turn supersedes a prior cancel or a server-restart interruption; otherwise
            # _execute_job's shutdown-race guard would keep relabelling every later turn. This runs only
            # after the applied / in-place checks above, so a refused continue leaves the receipt as-is.
            receipt.pop("cancelled", None)
            receipt["status"] = "running"
            receipt["error"] = None
            self.store.write_receipt(
                safe_id,
                receipt,
                scrub=bool(receipt.get("scrub_receipt", False)),
                workspace_root=str(self.workspace_root),
            )

        prior_prompts = list(receipt.get("prompts") or [])
        clean_followup = follow_up_prompt.strip()
        recorded_prompts = prior_prompts + [clean_followup]

        prior_inspected_files: list[str] = []
        wt_str_prefix = str(wt_dir).rstrip("/") + "/"
        ws_str_prefix = str(self.workspace_root).rstrip("/") + "/"
        for tc in receipt.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            t_args = tc.get("args") or {}
            if not isinstance(t_args, dict):
                continue
            raw_fp = str(t_args.get("file_path") or t_args.get("path") or "").strip()
            if not raw_fp:
                continue
            for pref in (wt_str_prefix, ws_str_prefix):
                if raw_fp.startswith(pref):
                    raw_fp = raw_fp[len(pref):]
                    break
            raw_fp = raw_fp.lstrip("/")
            if raw_fp and raw_fp not in prior_inspected_files and len(prior_inspected_files) < 8:
                prior_inspected_files.append(raw_fp)

        # Attach a concise context note of the original task and latest turn so the fresh
        # Antigravity session retains task context without re-executing prior turns.
        if prior_prompts:
            orig_task = prior_prompts[0]
            last_turn = prior_prompts[-1] if len(prior_prompts) > 1 else None
            prior_summary = (
                f"original_task={orig_task!r}, latest_turn={last_turn!r}"
                if last_turn is not None
                else f"{orig_task}"
            )
            no_edits_warning = ""
            if not receipt.get("files_changed") and prior_inspected_files:
                no_edits_warning = (
                    f" IMPORTANT: The prior turn timed out before writing any file edits "
                    f"(already inspected: {prior_inspected_files}). Do NOT re-read all files from scratch; "
                    f"open at most 1-2 target sections and immediately apply your changes with edit_file / create_file."
                )
            context_note = (
                f"[Prior turn context (baseline {baseline_sha}, files_changed={receipt.get('files_changed', [])}, "
                f"run `git diff {baseline_sha}` to inspect prior work): "
                f"{prior_summary}]{no_edits_warning}\n{clean_followup}"
            )
            worker_prompts = [context_note]
        else:
            worker_prompts = [clean_followup]
        plan_steps = list(receipt.get("plan") or [])
        if acceptance is not None:
            acceptance_cmds = validate_acceptance(acceptance, self.workspace_root)
        else:
            acceptance_cmds = list(receipt.get("acceptance") or [])
        budget = receipt.get("budget") or {}
        # A stored None means the lead disabled that cap; a missing key (older receipt) means defaults.
        budget_calls, budget_tokens = resolve_budget(
            (budget.get("max_tool_calls") or 0) if "max_tool_calls" in budget else None,
            (budget.get("max_total_tokens") or 0) if "max_total_tokens" in budget else None,
        )
        use_repo_map = os.environ.get("CC_AGY_REPO_MAP", "1").strip().lower() not in ("0", "false", "no")
        repo_map_text = (
            build_repo_map(self.workspace_root, cache_dir=self.store.data_dir)
            if use_repo_map
            else ""
        )
        cont_focus_targets = list(
            dict.fromkeys([
                *(receipt.get("files_changed") or []),
                *prior_inspected_files,
                *(receipt.get("allowed_paths") or []),
                *(receipt.get("preloaded_files") or []),
            ])
        )
        cont_focused_outline = (
            build_focused_file_outline(wt_dir if wt_dir.exists() else self.workspace_root, cont_focus_targets, max_tokens=500)
            if use_repo_map and cont_focus_targets
            else ""
        )
        worker_prompts = [
            render_worker_prompt(
                worker_prompts[0],
                plan=plan_steps,
                acceptance=acceptance_cmds,
                max_tool_calls=budget_calls,
                max_total_tokens=budget_tokens,
                repo_map=repo_map_text,
                focused_outline=cont_focused_outline,
            )
        ]

        model = str(receipt.get("model") or "gemini-3.1-pro-preview")
        escalated_from: str | None = None
        raw_esc_model = os.environ.get("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "").strip()
        if (
            raw_esc_model
            and raw_esc_model.lower() not in ("0", "false", "no", "off")
            and "flash" in model.lower()
            and worker_turn_failed(receipt.get("status"), receipt.get("acceptance_passed"))
        ):
            target_esc_model = (
                "gemini-4-argon"
                if raw_esc_model.lower() in ("1", "true", "yes", "on")
                else raw_esc_model
            )
            if target_esc_model != model:
                escalated_from = str(receipt.get("escalated_from_model") or model)
                model = target_esc_model
        with self._lock:
            raw_proj = self._raw_projects.get(safe_id)
            raw_key = self._raw_api_keys.get(safe_id)
        prior_auth_mode = receipt.get("auth_mode")
        if prior_auth_mode == "api_key" and not project:
            if not resolve_api_key(api_key or raw_key):
                raise ValueError(
                    "Job was started in api_key mode, but no Gemini API key is available after server restart. "
                    "Pass api_key=... or set GEMINI_API_KEY."
                )
        inherited_project = project
        if not inherited_project and prior_auth_mode != "api_key":
            inherited_project = raw_proj or receipt.get("vertex_project")
        auth = resolve_auth_config(
            requested_project=inherited_project,
            requested_location=location or receipt.get("vertex_location"),
            requested_api_key=api_key or raw_key,
            model=model,
        )
        auth_mode = auth["auth_mode"]
        resolved_key = auth.get("api_key")
        resolved_project = auth.get("project")
        resolved_location = auth.get("location")
        reroute_note = auth.get("reroute_note")

        with self._lock:
            if resolved_project:
                self._raw_projects[safe_id] = resolved_project
            if resolved_key:
                self._raw_api_keys[safe_id] = resolved_key

        if allowed_paths is not None:
            if in_place and allowed_paths:
                raise ValueError("allowed_paths cannot be combined with in_place=True")
            resolved_allowed_paths = validate_allowed_paths(allowed_paths)
        else:
            resolved_allowed_paths = validate_allowed_paths(receipt.get("allowed_paths") or [])
        allowed_paths = resolved_allowed_paths
        if escalated_from is not None:
            receipt["escalated_from_model"] = escalated_from
            receipt["model"] = model
            try:
                self.store.write_receipt(
                    safe_id,
                    receipt,
                    scrub=bool(receipt.get("scrub_receipt", False)),
                    workspace_root=str(self.workspace_root),
                )
            except Exception:
                pass
        enabled_tools = receipt.get("enabled_tools")
        disabled_tools = receipt.get("disabled_tools")
        enable_subagents = bool(receipt.get("enable_subagents", False))
        eff_scrub = bool(scrub_receipt) if scrub_receipt is not None else bool(receipt.get("scrub_receipt", False))
        started_at = float(receipt.get("started_at_epoch") or time.time())

        if wait:
            updated = self._execute_job(
                job_id=safe_id,
                worker_prompts=worker_prompts,
                recorded_prompts=recorded_prompts,
                model=model,
                auth_mode=auth_mode,
                api_key=resolved_key,
                project=resolved_project,
                location=resolved_location,
                reroute_note=reroute_note,
                allowed_paths=allowed_paths,
                timeout_seconds=timeout_seconds,
                enabled_tools=enabled_tools,
                disabled_tools=disabled_tools,
                enable_subagents=enable_subagents,
                wt_dir=wt_dir,
                baseline_sha=baseline_sha,
                baseline_file_hashes=baseline_hashes,
                started_at=started_at,
                prior_usage=receipt.get("usage"),
                prior_tool_calls=receipt.get("tool_calls"),
                scrub_receipt=eff_scrub,
                in_place=in_place,
                pre_untracked=pre_untracked,
                thinking_level=receipt.get("thinking_level"),
                plan=plan_steps,
                acceptance=acceptance_cmds,
                max_tool_calls=budget_calls,
                max_total_tokens=budget_tokens,
                preloaded_files=receipt.get("preloaded_files"),
            )
            digest = self._digest_with_progress(updated)
            return self._maybe_auto_apply_digest(
                safe_id,
                digest,
                auto_apply=auto_apply,
                in_place=in_place,
            )

        fut = self._executor.submit(
            self._execute_job,
            job_id=safe_id,
            worker_prompts=worker_prompts,
            recorded_prompts=recorded_prompts,
            model=model,
            auth_mode=auth_mode,
            api_key=resolved_key,
            project=resolved_project,
            location=resolved_location,
            reroute_note=reroute_note,
            allowed_paths=allowed_paths,
            timeout_seconds=timeout_seconds,
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
            enable_subagents=enable_subagents,
            wt_dir=wt_dir,
            baseline_sha=baseline_sha,
            baseline_file_hashes=baseline_hashes,
            started_at=started_at,
            prior_usage=receipt.get("usage"),
            prior_tool_calls=receipt.get("tool_calls"),
            scrub_receipt=eff_scrub,
            in_place=in_place,
            pre_untracked=pre_untracked,
            thinking_level=receipt.get("thinking_level"),
            plan=plan_steps,
            acceptance=acceptance_cmds,
            max_tool_calls=budget_calls,
            max_total_tokens=budget_tokens,
        )
        with self._lock:
            self._futures[safe_id] = fut
        return {
            "job_id": safe_id,
            "status": "running",
            "auth_mode": auth_mode,
            "prompts_count": len(recorded_prompts),
        }

    def _verify_in_place_unchanged(self, receipt: dict[str, Any]) -> dict[str, str | None]:
        """Refuse to continue an in_place job if anyone else changed the tree since its last turn."""
        root = self.workspace_root
        baseline_sha = str(receipt.get("baseline_commit_sha") or "")
        head_res = _run_git(["rev-parse", "HEAD"], cwd=root, check=False)
        if head_res.stdout.strip() != baseline_sha:
            raise ValueError(
                "HEAD moved since this in_place job ran (a commit was made). "
                "Start a new in_place job instead of continuing this one."
            )
        applied_files = set(receipt.get("applied_files") or [])
        for fpath, expected in (receipt.get("applied_file_hashes") or {}).items():
            if _hash_path(root / fpath) != expected:
                raise ValueError(
                    f"{fpath} changed after this in_place job finished. Commit or revert that edit, "
                    f"then start a new in_place job."
                )
        tracked_res = _run_git(["diff", "--name-only", "--no-renames", baseline_sha], cwd=root, check=False)
        extra_tracked = [
            f.strip() for f in tracked_res.stdout.splitlines() if f.strip() and f.strip() not in applied_files
        ]
        pre_untracked = receipt.get("pre_untracked_hashes") or {}
        extra_untracked = [
            rel
            for rel, h in _snapshot_untracked_hashes(root).items()
            if rel not in applied_files
            and not rel.startswith((".cc-agy-data/", ".worktrees/"))
            and not _is_ephemeral_build_or_test_artifact(rel, root=root)
            and pre_untracked.get(rel, "__absent__") != h
        ]
        if extra_tracked:
            raise ValueError(
                "Tracked files outside this in_place job changed since its last turn: "
                f"{extra_tracked[:10]}. Commit or revert them, then start a new job."
            )
        current = _snapshot_untracked_hashes(root)
        return {rel: current.get(rel) for rel in extra_untracked}

    def delegate_cancel(self, job_id: str, *, cleanup_worktree: bool = True) -> dict[str, Any]:
        """Cancel a job and clean up its worktree while preserving its cancelled receipt status."""
        safe_id = validate_job_id(job_id)
        with self._lock:
            self._cancelled.add(safe_id)
            fut = self._futures.get(safe_id)
            if fut is not None:
                fut.cancel()
            still_running = fut is not None and not fut.done()

        receipt = self.store.read_receipt(safe_id)
        wt_dir = Path(str(receipt.get("worktree_dir") or ""))
        # Only remove wt_dir immediately if the worker thread is not currently running inside it;
        # if it is still running, _execute_job's finally block removes wt_dir upon exit.
        if cleanup_worktree and wt_dir.exists() and not still_running:
            self._cleanup_worktree(wt_dir)

        if not receipt.get("applied"):
            receipt["cancelled"] = True
            receipt["status"] = "cancelled"
            receipt["error"] = "Cancelled by user via delegate_cancel."
            receipt["finished_at_epoch"] = round(time.time(), 3)
            self.store.write_receipt(
                safe_id,
                receipt,
                scrub=bool(receipt.get("scrub_receipt", False)),
                workspace_root=str(self.workspace_root),
            )

        return {
            "job_id": safe_id,
            "status": receipt.get("status"),
            "cancelled": True,
        }

    def _digest_with_progress(self, receipt: dict[str, Any], *, include_summary: bool = True) -> dict[str, Any]:
        """Compact digest plus the worker's latest report_to_lead signals (visible while it is running)."""
        digest = self._compact_digest(receipt, include_summary=include_summary)
        job_id = str(receipt.get("job_id") or "")
        if job_id:
            try:
                digest["progress"] = read_progress(self.store.progress_path(job_id), limit=5)
            except ValueError:
                digest["progress"] = []
        if digest.get("status") == "running":
            digest["poll_hint"] = (
                "Still running. Poll again with delegate_result(wait=true) (waits up to 240s, which keeps "
                "your prompt cache warm), or check progress with delegate_status."
            )
        elif (
            job_id
            and digest.get("status") in ("ok", "checkpoint", "needs_input", "timeout", "scope_violation")
            and digest.get("files_changed")
        ):
            try:
                diff_path = self.store.diff_path(job_id)
                full_diff = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""
            except Exception:
                full_diff = ""
            if full_diff.strip():
                # Order non-test implementation file diffs ahead of test file diffs in the preview
                chunks: list[tuple[int, str]] = []
                cur: list[str] = []
                cur_is_test = 0
                for dline in full_diff.splitlines(keepends=True):
                    if dline.startswith("diff --git "):
                        if cur:
                            chunks.append((cur_is_test, "".join(cur)))
                        cur = [dline]
                        hdr = parse_diff_git_header(dline)
                        cpath = (hdr[1] or hdr[0]) if hdr else ""
                        cur_is_test = 1 if (cpath.startswith(("tests/", "test/")) or "/test_" in cpath or cpath.endswith(("_test.py", ".test.ts", ".spec.ts"))) else 0
                    else:
                        cur.append(dline)
                if cur:
                    chunks.append((cur_is_test, "".join(cur)))
                ordered_diff = "".join(chunk for _is_t, chunk in sorted(chunks, key=lambda x: x[0])) if chunks else full_diff
                max_preview_tokens = _env_int("CC_AGY_DIFF_PREVIEW_TOKENS", 1800)
                used_tokens = estimate_tokens(json.dumps(digest, indent=2, sort_keys=True))
                preview_budget = max(0, min(max_preview_tokens, MAX_TOOL_RESPONSE_TOKENS - used_tokens - 150))
                if preview_budget >= 80:
                    preview_txt, was_trunc, _ = truncate_to_token_budget(ordered_diff, preview_budget)
                    digest["diff_preview"] = preview_txt
                    digest["diff_preview_complete"] = not was_trunc
            missing_syms = digest.get("missing_prompt_symbols") or []
            missing_note = (
                f" Note: prompt symbol(s) {missing_syms} were not found in the modified files; verify whether "
                f"they are required and call delegate_continue(job_id={job_id!r}, follow_up_prompt=...) if missing."
                if missing_syms
                else ""
            )
            if not digest.get("applied"):
                if (
                    digest.get("status") == "ok"
                    and digest.get("acceptance_passed") is not False
                    and not digest.get("scope_violations")
                ):
                    digest["next_action"] = (
                        f"Acceptance checks passed and scope_violations=[]. Call delegate_apply(job_id={job_id!r}) "
                        f"directly to land these changes without re-running the same test suite.{missing_note}"
                    )
                elif digest.get("status") == "ok" and digest.get("acceptance_passed") is False:
                    digest["next_action"] = (
                        f"Worker modified {len(digest.get('files_changed') or [])} file(s), but an acceptance check "
                        f"failed (see acceptance_results). Call delegate_continue(job_id={job_id!r}, "
                        f"follow_up_prompt=...) to fix the failing test in the existing worktree.{missing_note}"
                    )
                elif digest.get("status") == "checkpoint":
                    digest["next_action"] = (
                        f"Worker reached the tool-call checkpoint after modifying {len(digest.get('files_changed') or [])} "
                        f"file(s) (acceptance_passed={digest.get('acceptance_passed')!r}). Call "
                        f"delegate_continue(job_id={job_id!r}, follow_up_prompt=...) if any task requirement or test "
                        f"is still unfinished, or call delegate_apply(job_id={job_id!r}) if the diff is complete.{missing_note}"
                    )
                elif digest.get("status") == "needs_input":
                    digest["next_action"] = (
                        f"Worker paused with status='needs_input' (see progress). Call "
                        f"delegate_continue(job_id={job_id!r}, follow_up_prompt=...) to answer and finish, or "
                        f"delegate_apply(job_id={job_id!r}) if the edits are already complete.{missing_note}"
                    )
                elif digest.get("status") == "scope_violation" or digest.get("scope_violations"):
                    digest["next_action"] = (
                        f"Worker modified file(s) outside allowed_paths ({digest.get('scope_violations')}). Review "
                        f"diff_preview and call delegate_apply(job_id={job_id!r}, allow_scope_violations=True) to "
                        f"approve, or call delegate_continue(job_id={job_id!r}, follow_up_prompt=..., allowed_paths=...).{missing_note}"
                    )
                elif digest.get("status") == "timeout":
                    if digest.get("acceptance_passed") is True:
                        digest["next_action"] = (
                            f"Worker timed out after modifying {len(digest.get('files_changed') or [])} file(s) and passing "
                            f"acceptance checks. Do NOT read or grep repository files yourself — immediately call "
                            f"delegate_continue(job_id={job_id!r}, follow_up_prompt=...) if any task item or unit test is "
                            f"still missing, or call delegate_apply(job_id={job_id!r}, allow_scope_violations=True) now.{missing_note}"
                        )
                    else:
                        digest["next_action"] = (
                            f"Worker timed out mid-implementation with {len(digest.get('files_changed') or [])} modified "
                            f"file(s) preserved in its worktree. Do NOT read or grep repository files yourself — "
                            f"immediately call delegate_continue(job_id={job_id!r}, follow_up_prompt=...) so the worker "
                            f"can finish the remaining edits and run acceptance tests.{missing_note}"
                        )
        elif job_id and digest.get("status") in ("timeout", "checkpoint", "needs_input") and not digest.get("applied"):
            digest["next_action"] = (
                f"Worker stopped with status={digest.get('status')!r} before modifying files. Do NOT read or grep "
                f"repository files yourself — immediately call delegate_continue(job_id={job_id!r}, "
                "follow_up_prompt=...) instructing the worker to apply its changes directly with edit_file/create_file."
            )
        return digest

    def salvage_unapplied_worktree(self) -> dict[str, Any] | None:
        """If the workspace has zero applied jobs on shutdown, salvage the best unapplied worker job
        that produced clean file edits so bounded benchmark/CLI runs never exit with an empty patch."""
        all_receipts = self.store.list_receipts(workspace_root=self.workspace_root)
        if any(bool(r.get("applied")) for r in all_receipts):
            return None
        candidates: list[dict[str, Any]] = []
        with self._lock:
            cancelled_snapshot = set(self._cancelled)
        for r in all_receipts:
            if r.get("kind", "delegate") != "delegate":
                continue
            if r.get("applied") or r.get("cancelled") or r.get("symlink_escapes") or not r.get("files_changed"):
                continue
            if r.get("status") not in ("ok", "checkpoint", "needs_input", "timeout", "interrupted", "scope_violation"):
                continue
            jid = str(r.get("job_id") or "")
            if not jid or jid in cancelled_snapshot:
                continue
            wt_p = Path(str(r.get("worktree_dir") or ""))
            if not wt_p.exists() and not self.store.diff_path(jid).exists():
                continue
            candidates.append(r)
        if not candidates:
            return None
        candidates.sort(
            key=lambda r: (
                1 if r.get("acceptance_passed") is True else (0 if r.get("acceptance_passed") is None else -1),
                1 if r.get("status") == "ok" else 0,
                len(r.get("files_changed") or []),
                float(r.get("started_at_epoch") or 0.0),
            ),
            reverse=True,
        )
        best = candidates[0]
        best_id = str(best["job_id"])
        try:
            lock_file_path = self.store.data_dir / ".apply.lock"
            with self._apply_lock:
                with open(lock_file_path, "w", encoding="utf-8") as lock_handle:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                    try:
                        res = self._delegate_apply_locked(
                            best_id,
                            cleanup_worktree=False,
                            allow_scope_violations=True,
                        )
                    finally:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            rec = self.store.read_receipt(best_id)
            rec["salvaged_on_shutdown"] = True
            self.store.write_receipt(
                best_id,
                rec,
                scrub=bool(rec.get("scrub_receipt", False)),
                workspace_root=str(self.workspace_root),
            )
            return res
        except Exception:
            return None

    def mark_running_interrupted(self) -> list[str]:
        """Called when the MCP server is shutting down: record unfinished jobs as interrupted instead of
        leaving them 'running' forever, preserve any live worktree diff, and optionally salvage unapplied edits."""
        marked: list[str] = []
        with self._lock:
            live = {jid for jid, fut in self._futures.items() if not fut.done()}
            cancelled_ids = set(self._cancelled)
        for jid in live:
            try:
                receipt = self.store.read_receipt(jid)
            except Exception:
                continue
            if jid in cancelled_ids or receipt.get("cancelled") or receipt.get("status") == "cancelled":
                # Keep the user's explicit cancel; never relabel it as a server interruption.
                continue
            if not receipt.get("in_place") and not receipt.get("applied"):
                wt_dir = Path(str(receipt.get("worktree_dir") or ""))
                baseline_sha = str(receipt.get("baseline_commit_sha") or "HEAD")
                if wt_dir.exists():
                    try:
                        rec_diff, rec_files, rec_escapes = self._collect_worktree_diff(
                            wt_dir,
                            baseline_sha,
                            allowed_paths=receipt.get("allowed_paths"),
                        )
                        if rec_files and not rec_escapes:
                            self.store.diff_path(jid).write_text(rec_diff, encoding="utf-8")
                            receipt["files_changed"] = rec_files
                            receipt["symlink_escapes"] = rec_escapes
                            receipt["provenance"] = extract_diff_provenance(rec_diff)
                    except Exception:
                        pass
            receipt["status"] = "interrupted"
            receipt["error"] = (
                "The MCP server stopped before this job finished; the unfinished turn's token usage was not "
                "recorded by the SDK."
            )
            receipt["finished_at_epoch"] = round(time.time(), 3)
            try:
                self.store.write_receipt(
                    jid, receipt, scrub=bool(receipt.get("scrub_receipt", False)),
                    workspace_root=str(self.workspace_root),
                )
                marked.append(jid)
            except Exception:
                continue
        should_salvage = (
            self._is_bounded_lead_session()
            or os.environ.get("CC_AGY_SALVAGE_ON_SHUTDOWN", "").strip().lower() in ("1", "true", "yes")
        )
        if should_salvage:
            try:
                self.salvage_unapplied_worktree()
            except Exception:
                pass
        return marked

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the underlying thread pool executor."""
        self._executor.shutdown(wait=wait)

    @staticmethod
    def _compact_digest(
        receipt: dict[str, Any],
        *,
        include_summary: bool = True,
        include_acceptance_details: bool = True,
    ) -> dict[str, Any]:
        """Return a token-bounded digest suitable for Claude Code's context window."""
        if not include_acceptance_details and not include_summary:
            raw_usage = receipt.get("usage") or {}
            raw_cost = receipt.get("cost") or {}
            slim: dict[str, Any] = {
                "job_id": receipt.get("job_id"),
                "kind": receipt.get("kind", "delegate"),
                "status": receipt.get("status"),
                "auth_mode": receipt.get("auth_mode", "vertex"),
                "model": receipt.get("model"),
                "files_changed": receipt.get("files_changed", []),
                "applied": bool(receipt.get("applied", False)),
                "in_place": bool(receipt.get("in_place", False)),
                "usage": {
                    "prompt_token_count": int(raw_usage.get("prompt_token_count") or 0),
                    "cached_content_token_count": int(raw_usage.get("cached_content_token_count") or 0),
                    "candidates_token_count": int(raw_usage.get("candidates_token_count") or 0),
                    "thoughts_token_count": int(raw_usage.get("thoughts_token_count") or 0),
                    "total_token_count": int(raw_usage.get("total_token_count") or 0),
                },
                "cost": {
                    "total_cost_usd": float(raw_cost.get("total_cost_usd") or 0.0),
                    "total_cost_cents": float(raw_cost.get("total_cost_cents") or 0.0),
                },
            }
            if receipt.get("acceptance_passed") is not None:
                slim["acceptance_passed"] = receipt.get("acceptance_passed")
            if (receipt.get("escalation") or {}).get("escalate_to_lead"):
                slim["escalate_to_lead"] = True
            if receipt.get("reverted"):
                slim["reverted"] = True
            if receipt.get("error"):
                slim["error"] = receipt.get("error")
            if receipt.get("stop_reason"):
                slim["stop_reason"] = receipt.get("stop_reason")
            if receipt.get("escalated_from_model"):
                slim["escalated_from_model"] = receipt.get("escalated_from_model")
            return slim

        digest_usage = dict(receipt.get("usage") or {})
        raw_reqs = digest_usage.get("requests")
        if isinstance(raw_reqs, list):
            def _safe_req_tokens(req_obj: Any) -> int:
                if not isinstance(req_obj, dict):
                    return 0
                try:
                    return int(float(req_obj.get("prompt_token_count") or 0))
                except (ValueError, TypeError):
                    return 0

            digest_usage["requests_count"] = len(raw_reqs)
            digest_usage["over_200k_requests_count"] = sum(
                1
                for r in raw_reqs
                if _safe_req_tokens(r) > 200_000
            )
            if len(raw_reqs) > 10:
                digest_usage["requests"] = raw_reqs[-10:]
                digest_usage["requests_truncated"] = True
        if isinstance(digest_usage.get("turns"), list) and len(digest_usage["turns"]) > 10:
            digest_usage["turns"] = digest_usage["turns"][-10:]

        digest: dict[str, Any] = {
            "job_id": receipt.get("job_id"),
            "kind": receipt.get("kind", "delegate"),
            "status": receipt.get("status"),
            "auth_mode": receipt.get("auth_mode", "vertex"),
            "model": receipt.get("model"),
            "vertex_location": receipt.get("vertex_location"),
            "reroute_note": receipt.get("reroute_note"),
            "allowed_paths": receipt.get("allowed_paths", []),
            "scope_violations": receipt.get("scope_violations", []),
            "files_changed": receipt.get("files_changed", []),
            "diff_tokens_estimated": receipt.get("diff_tokens_estimated", 0),
            "tool_call_count": receipt.get("tool_call_count", 0),
            "usage": digest_usage,
            "cost": receipt.get("cost", {}),
            "applied": bool(receipt.get("applied", False)),
            "in_place": bool(receipt.get("in_place", False)),
            "thinking_level": receipt.get("thinking_level"),
            "stop_reason": receipt.get("stop_reason"),
            "budget": receipt.get("budget"),
            "containment": receipt.get("containment", "none"),
            "containment_warning": receipt.get("containment_warning"),
            "acceptance_passed": receipt.get("acceptance_passed"),
            "error": receipt.get("error"),
        }
        if receipt.get("preloaded_files"):
            digest["preloaded_files"] = list(receipt.get("preloaded_files") or [])
        if receipt.get("escalated_from_model"):
            digest["escalated_from_model"] = receipt.get("escalated_from_model")
        if receipt.get("missing_prompt_symbols"):
            digest["missing_prompt_symbols"] = list(receipt.get("missing_prompt_symbols") or [])
        if receipt.get("applied"):
            digest["applied_files"] = list(receipt.get("applied_files") or receipt.get("files_changed") or [])
            prov = receipt.get("provenance") or {}
            if prov.get("diff_sha256"):
                digest["diff_sha256"] = prov.get("diff_sha256")
        if receipt.get("failed_attempts"):
            digest["failed_attempts"] = int(receipt.get("failed_attempts") or 0)
        if receipt.get("escalation"):
            digest["escalation"] = receipt.get("escalation")
        if receipt.get("reverted"):
            digest["reverted"] = True
            digest["reverted_files"] = receipt.get("reverted_files", [])
        if include_acceptance_details:
            digest["acceptance_results"] = receipt.get("acceptance_results") or []
        if (
            not receipt.get("applied")
            and receipt.get("files_changed")
            and receipt.get("status") in ("timeout", "checkpoint")
        ):
            digest["apply_hint"] = (
                f"Worker modified {len(receipt.get('files_changed', []))} file(s). "
                f"Inspect the diff with get_diff(job_id={receipt.get('job_id')!r}) and apply clean edits directly via "
                f"delegate_apply(job_id={receipt.get('job_id')!r}) if the implementation is ready."
            )
        if receipt.get("in_place"):
            digest["in_place_notes"] = receipt.get("in_place_notes", [])
            digest["undo_hint"] = receipt.get("undo_hint")
        if include_summary:
            summary_text = str(receipt.get("summary") or "")
            meta_tokens = estimate_tokens(json.dumps(digest, indent=2, sort_keys=True))
            max_summary_tokens = _env_int("CC_AGY_DIGEST_SUMMARY_TOKENS", 300)
            summary_budget = max(64, min(max_summary_tokens, MAX_TOOL_RESPONSE_TOKENS - meta_tokens - 250))
            clipped_summary, truncated, est_tokens = truncate_to_token_budget(
                summary_text, summary_budget
            )
            digest["summary"] = clipped_summary
            digest["summary_truncated"] = bool(receipt.get("summary_truncated") or truncated)
            digest["summary_tokens_estimated"] = est_tokens
        return digest
