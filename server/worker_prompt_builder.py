"""Context ranking, safe workspace file loading, QA verdict parsing, and worker prompt rendering."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from server.agy_runner import is_straitjacket_available
from server.job_git_utils import _git_raw, _is_ephemeral_build_or_test_artifact
from server.regions import estimate_tokens, truncate_to_token_budget

_PRELOAD_SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".pdf",
    ".whl", ".tar", ".gz", ".zip", ".pyc", ".so", ".dylib", ".lock",
}
_PRELOAD_BOILERPLATE_STEMS = frozenset({
    "license", "licence", "copying", "notice", "dockerfile", "makefile", "gnumakefile",
    "changelog", "changes", "history", "authors", "contributors", "contributing",
    "code_of_conduct", "security",
})
_PRELOAD_BOILERPLATE_EXACT = frozenset({
    ".dockerignore", ".editorconfig", ".gitattributes", ".gitignore",
    ".prettierrc", ".eslintrc", ".flake8", ".yamllint",
})
_PRELOAD_ENTRYPOINT_NAMES = frozenset({
    "__main__.py", "cli.py", "main.py", "app.py", "main.go", "root.go",
    "index.ts", "index.tsx", "index.js", "main.ts", "cli.ts", "main.rs", "lib.rs",
})


def _is_preload_boilerplate(rel_path: str, prompt_low: str) -> bool:
    """Exclude license/Docker/Makefile/README/changelog boilerplate unless explicitly referenced in the prompt."""
    norm_low = rel_path.replace("\\", "/").strip("/").lower()
    bname = norm_low.rsplit("/", 1)[-1]
    if norm_low in prompt_low or bname in prompt_low:
        return False
    if bname in _PRELOAD_BOILERPLATE_EXACT:
        return True
    stem = bname.split(".", 1)[0]
    if stem in _PRELOAD_BOILERPLATE_STEMS:
        return True
    if stem == "readme" and "readme" not in prompt_low:
        return True
    return False

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

_PROMPT_ASKS_FOR_TESTS_RE = re.compile(
    r"\b(?:unit\s+tests?|integration\s+tests?|test\s+cases?|test\s+suite|write\s+tests?|add\s+tests?|pytest|vitest|jest)\b",
    re.IGNORECASE,
)

_QUOTA_OR_RATE_LIMIT_RE = re.compile(
    r"(?:\b429\b|RESOURCE_EXHAUSTED|QUOTA_EXHAUSTED|\brate[- ]limit(?:ed|s)?\b|\b503\b.*?\b(?:UNAVAILABLE|overloaded)\b|\bUNAVAILABLE\b.*?\b503\b)",
    re.IGNORECASE | re.DOTALL,
)


def is_quota_or_rate_limit_error(error_text: str | None) -> bool:
    """Return True when an error string indicates a Gemini 429/503 quota or rate-limit failure."""
    return bool(error_text and _QUOTA_OR_RATE_LIMIT_RE.search(error_text))


QA_DIFF_AUDIT_SYSTEM = """You are a senior software engineer performing a 1-shot blocker-only code review of a git diff against a task specification.
Report ONLY High or Medium functional blockers:
1. Missing requirements, cross-module wiring, or edge cases explicitly stated in the task specification (check every branch and requirement in the Original User Specification even if abbreviated in the Lead brief).
2. Subtle logic bugs, off-by-one errors, state leaks, security bugs, or broken imports/exports.
3. Weakened or altered pre-existing test assertions/expected outputs made to match broken behavior instead of fixing production code, or missing unit test coverage when the repository has automated tests.

Do NOT report Low, Optional, polish, naming, style, formatting, or future-hardening suggestions.
Output format:
- If there are no High or Medium blockers, start with `VERDICT: PASS` on line 1.
- If there is at least one concrete blocker, start with `VERDICT: NEEDS_FIXES` on line 1, followed by concise bullet points (`- ISSUE: <file:line>: <exact bug and how to fix it>`)."""


def build_qa_diff_audit_system(*, verification: str | None = None, prompt: str = "") -> str:
    """Build the 1-shot QA review system prompt, skipping test-coverage demands on testless repos."""
    ver = str(verification or "").strip().lower()
    if ver in ("smoke_only", "unverified") and not _PROMPT_ASKS_FOR_TESTS_RE.search(prompt or ""):
        return (
            "You are a senior software engineer performing a 1-shot blocker-only code review of a git diff against a task specification.\n"
            "Report ONLY High or Medium functional blockers:\n"
            "1. Missing requirements or edge cases explicitly stated in the task specification (check every branch and requirement in the Original User Specification even if abbreviated in the Lead brief).\n"
            "2. Subtle logic bugs, off-by-one errors, state leaks, security bugs, or broken imports/exports.\n\n"
            f"This repository has no automated test suite covering this change (verification={ver!r}). "
            "Do NOT fail the diff for missing unit tests unless the task specification explicitly asked to add tests.\n"
            "Do NOT report Low, Optional, polish, naming, style, formatting, or future-hardening suggestions.\n"
            "Output format:\n"
            "- If there are no High or Medium blockers, start with `VERDICT: PASS` on line 1.\n"
            "- If there is at least one concrete blocker, start with `VERDICT: NEEDS_FIXES` on line 1, followed by concise bullet points (`- ISSUE: <file:line>: <exact bug and how to fix it>`)."
        )
    return QA_DIFF_AUDIT_SYSTEM


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
    res = _git_raw(
        ["ls-files"],
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
        if _is_preload_boilerplate(rel, prompt_low):
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz == 0 or sz > 400_000:
            continue
        rel_low = rel.lower()
        bname_low = p.name.lower()
        parts_low = rel_low.split("/")
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
        if score > 0 or bname_low in _PRELOAD_ENTRYPOINT_NAMES:
            if bname_low in _PRELOAD_ENTRYPOINT_NAMES and not rel_low.startswith(("docs/", "examples/", "tests/", "test/")):
                score += 4.0
            elif any(
                tok in ("api", "client", "queries", "query", "graphql", "rpc", "routes", "router", "handler", "handlers", "cmd", "cli")
                for seg in parts_low
                for tok in re.split(r"[._-]+", seg.split(".", 1)[0])
                if tok
            ) and score > 0:
                score += 3.0
        if score > 0 and not rel_low.startswith(("docs/", "examples/", ".github/")):
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


def _load_safe_workspace_files(
    workspace_root: Path,
    rel_paths: list[str] | None,
    *,
    max_total_tokens: int = 8000,
    line_windows: dict[str, tuple[int, int]] | None = None,
) -> list[tuple[str, str]]:
    """Safely load workspace-relative text files up to `max_total_tokens` without escaping `workspace_root`."""
    if not rel_paths:
        return []
    ws_resolved = workspace_root.resolve()
    secret_basenames = {
        ".env", ".netrc", ".npmrc", ".pypirc", ".mcp.json", ".hmac_key",
        "id_rsa", "id_ed25519", "id_ecdsa", "credentials.json",
    }
    candidates: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for raw_p in rel_paths:
        clean_p = str(raw_p or "").strip()
        if not clean_p or ".." in Path(clean_p).parts:
            continue
        try:
            cand = (ws_resolved / clean_p).resolve()
            rel_obj = cand.relative_to(ws_resolved)
        except (OSError, ValueError):
            continue
        rel_str = rel_obj.as_posix()
        if not rel_str or rel_str in seen:
            continue
        bname = rel_obj.name.lower()
        parts_low = [p.lower() for p in rel_obj.parts]
        if (
            bname in secret_basenames
            or bname.startswith(".env.")
            or bname.endswith((".pem", ".key", ".p12", ".pfx"))
            or ".git" in parts_low
            or ".claude" in parts_low
            or ".ssh" in parts_low
            or ".cc-agy-data" in parts_low
            or ".worktrees" in parts_low
            or not cand.is_file()
        ):
            continue
        try:
            raw_bytes = cand.read_bytes()
            if b"\x00" in raw_bytes[:4096]:
                continue
            text = raw_bytes.decode("utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        seen.add(rel_str)
        candidates.append((rel_str, text, estimate_tokens(text)))

    if not candidates:
        return []

    # Reserve token budget for smaller files so an oversized first file never starves subsequent small files.
    small_cutoff = max(250, max_total_tokens // 2)
    remaining_small_tokens = sum(est for _, _, est in candidates if est <= small_cutoff)
    loaded: list[tuple[str, str]] = []
    used_tokens = 0
    for rel_str, text, est in candidates:
        is_small = est <= small_cutoff
        if is_small:
            remaining_small_tokens = max(0, remaining_small_tokens - est)
        avail_for_this = max_total_tokens - used_tokens - (0 if is_small else min(remaining_small_tokens, max_total_tokens // 2))
        if avail_for_this <= 0:
            continue
        if est > avail_for_this:
            if avail_for_this < 200 and loaded:
                continue
            all_lines = text.splitlines()
            total_lines = max(1, len(all_lines))
            win = (line_windows or {}).get(rel_str)
            if win and total_lines > 1:
                w_start = max(1, min(int(win[0]), total_lines))
                w_limit = max(50, int(win[1]) if int(win[1]) > 0 else 250)
                w_end = min(total_lines, w_start + w_limit - 1)
                window_text = "\n".join(all_lines[w_start - 1 : w_end])
                trunc_text, _was_tr, _orig_t = truncate_to_token_budget(window_text, max(100, avail_for_this - 30))
                shown_count = max(1, len(trunc_text.splitlines()))
                actual_end = min(total_lines, w_start + shown_count - 1)
                text = (
                    trunc_text.rstrip("\n")
                    + f"\n... [truncated: showing lines {w_start}-{actual_end} of {total_lines}; use view_file for remaining lines]"
                )
            else:
                trunc_text, _was_tr, _orig_t = truncate_to_token_budget(text, max(100, avail_for_this - 30))
                shown_lines = max(1, len(trunc_text.splitlines()))
                text = (
                    trunc_text.rstrip("\n")
                    + f"\n... [truncated: showing lines 1-{shown_lines} of {total_lines}; use view_file for remaining lines]"
                )
            est = estimate_tokens(text)
        loaded.append((rel_str, text))
        used_tokens += est
    return loaded


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
    spec_files_inlined: list[tuple[str, str]] | None = None,
    context_files_inlined: list[tuple[str, str]] | None = None,
    checklist: list[str] | None = None,
    raw_user_prompt: str = "",
    baseline_acceptance_probe: list[dict[str, Any]] | None = None,
    allowed_paths: list[str] | None = None,
    media_files: list[str] | None = None,
) -> str:
    """Wrap the lead's brief with repository map, budget/reporting rules, inlined context, plan, and acceptance."""
    static_parts: list[str] = []
    if repo_map and repo_map.strip():
        static_parts.append("## Repository Map (Cached)\n" + repo_map.strip())
    budget_bits = []
    if max_tool_calls:
        budget_bits.append(f"{max_tool_calls} tool calls")
    if max_total_tokens:
        budget_bits.append(f"{max_total_tokens:,} total tokens")
    if budget_bits:
        static_parts.append(
            "## Budget\nThis turn stops automatically after " + " or ".join(budget_bits) + ". Work efficiently: "
            "make your first file edit within 8-10 tool calls and implement changes incrementally file-by-file "
            "so your edits are saved on disk early (never read more than 6-8 files before editing, and NEVER draft "
            "full file implementations or long code blocks inside internal thinking; emit create_file/edit_file "
            "tool calls directly and iterate against test output), read files <=260 lines in a single view_file "
            "call (start_line=1, end_line=260) or use the Focused File Outline line numbers to read 200-260 line "
            "spans on larger files instead of paging in small slices, avoid interactive one-liner REPL probes, "
            "run only the focused test file(s) needed for your changes, do not re-read files you have already seen, "
            "and remove any temporary scratch scripts before finishing."
        )
    if progress_channel:
        static_parts.append(
            "## Reporting\nUse the report_to_lead tool only when blocked (kind=\"question\" if a requirement is "
            "ambiguous; kind=\"blocker\" if you cannot proceed; questions and blockers end your turn) or "
            "kind=\"milestone\" between distinct plan steps on multi-step plans. Once your implementation and "
            "tests pass, finish your turn directly with a short summary instead of calling report_to_lead."
        )
    if (
        os.environ.get("CC_AGY_CONTAINMENT", "1").lower() not in ("0", "false", "no")
        and is_straitjacket_available()
    ):
        static_parts.append(
            "## Shell Commands & Evidence\n"
            "Commands and file reads execute with bounded output via Straitjacket (`run_command`, `view_file`, "
            "`search_dir`, `list_dir`, `find_file`, `ctx_query`, `ctx_get`, `ctx_search`). If a test or build "
            "produces large output, it returns a concise digest and an address (e.g. `run:<id>#stdout`). Use "
            "`ctx_get(address, start_line, end_line)` or `ctx_search(address, pattern)` to retrieve specific "
            "omitted lines or tracebacks instead of rerunning commands."
        )

    dynamic_parts: list[str] = []
    if raw_user_prompt and raw_user_prompt.strip():
        if allowed_paths:
            dynamic_parts.append(
                "## Background User Context (Reference Only: Scope Restricted to allowed_paths)\n"
                f"Scope restriction: implement ONLY the Lead's brief above and modify ONLY files within "
                f"`allowed_paths={list(allowed_paths)}`. Ignore any parts of this background request that "
                "target other files/directories or ask to run multiple workers:\n"
                + raw_user_prompt.strip()
            )
        else:
            dynamic_parts.append(
                "## Original User Request (Verbatim: Do Not Omit Any Requirement)\n"
                "If the Lead's brief above omits a branch or guesses an unverified flag/helper name, follow this Original User Request and existing repository helpers/callers.\n"
                + raw_user_prompt.strip()
            )
    if spec_files_inlined:
        full_specs = [(r, c) for r, c in spec_files_inlined if "[truncated:" not in c]
        trunc_specs = [(r, c) for r, c in spec_files_inlined if "[truncated:" in c]
        if full_specs:
            spec_blocks = [f"### `{rel}`\n```\n{content.rstrip()}\n```" for rel, content in full_specs]
            dynamic_parts.append(
                "## Task Specification Files (Verbatim: Do Not Re-Read)\n" + "\n\n".join(spec_blocks)
            )
        if trunc_specs:
            tspec_blocks = [f"### `{rel}`\n```\n{content.rstrip()}\n```" for rel, content in trunc_specs]
            dynamic_parts.append(
                "## Partially Inlined Specification Files (Truncated to Token Budget: Use view_file for Omitted Lines)\n"
                + "\n\n".join(tspec_blocks)
            )
    if checklist:
        dynamic_parts.append(
            "## Behavioral Checklist\n"
            + "\n".join(f"{i}. {item}" for i, item in enumerate(checklist, 1))
        )
    if media_files:
        media_lines = [f"- `{rel_m}`" for rel_m in media_files]
        dynamic_parts.append(
            "## Attached Multimodal Media Files (Present in Worktree, Excluded from Git Diffs)\n"
            + "\n".join(media_lines)
        )
    if focused_outline and focused_outline.strip():
        dynamic_parts.append("## Focused File Outline (Line-Numbered)\n" + focused_outline.strip())
    if preloaded_files:
        # Path hints only: full file contents pasted into turn 1 get re-read on every later turn.
        path_lines = [f"- `{rel}`" for rel, _content in preloaded_files]
        dynamic_parts.append(
            "## Candidate Repository Files\n"
            "These files matched the task's identifiers. Open only the relevant sections:\n"
            + "\n".join(path_lines)
        )
    if context_files_inlined:
        full_ctx = [(r, c) for r, c in context_files_inlined if "[truncated:" not in c]
        trunc_ctx = [(r, c) for r, c in context_files_inlined if "[truncated:" in c]
        if full_ctx:
            ctx_blocks = [f"### `{rel}`\n```\n{content.rstrip()}\n```" for rel, content in full_ctx]
            dynamic_parts.append(
                "## Pre-Read Context Files (Already Inlined: Do Not Call view_file on These Again; Edit Directly)\n"
                + "\n\n".join(ctx_blocks)
            )
        if trunc_ctx:
            tctx_blocks = [f"### `{rel}`\n```\n{content.rstrip()}\n```" for rel, content in trunc_ctx]
            dynamic_parts.append(
                "## Partially Inlined Context Files (Truncated to Token Budget: Use view_file with start_line/end_line for Omitted Lines)\n"
                + "\n\n".join(tctx_blocks)
            )
    if plan:
        dynamic_parts.append("## Plan from the lead\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1)))
    if acceptance:
        baseline_notes: list[str] = []
        for bp in baseline_acceptance_probe or []:
            if not isinstance(bp, dict) or bp.get("passed"):
                continue
            bp_cmd = str(bp.get("command") or "").strip()
            if bp.get("timed_out"):
                if bp_cmd:
                    baseline_notes.append(
                        f"[Baseline Acceptance Timeout Note] Command `{bp_cmd}` exceeded the 15s baseline probe "
                        f"on clean HEAD because the full test suite is large. Do NOT run the full `{bp_cmd}` "
                        f"suite during iteration; run only targeted test file(s) covering your modified modules "
                        f"(post-turn acceptance will also scope to related tests automatically)."
                    )
                continue
            bp_rc = bp.get("exit_code")
            if bp_rc in (None, 0):
                continue
            bp_tail = " | ".join(
                ln.strip() for ln in str(bp.get("error_tail") or "").splitlines() if ln.strip()
            )[:280]
            cwd_hint = ""
            if bp.get("subdir_cwd_passed") and bp.get("passing_cwd"):
                p_cwd = str(bp["passing_cwd"])
                cwd_hint = (
                    f" Note: existing tests under `{p_cwd}` pass (`rc=0`) on HEAD when run with "
                    f"`working_directory={p_cwd!r}`. Verify `{p_cwd}` with `working_directory={p_cwd!r}` and run "
                    f"your new/modified tests (e.g. `--ignore={p_cwd}`), then stop once both pass; post-worker "
                    f"acceptance automatically accounts for this pre-existing HEAD `cwd` quirk."
                )
            baseline_notes.append(
                f"[Baseline Acceptance Note] Command `{bp_cmd}` already exited non-zero (`rc={bp_rc}`) on "
                f"unmodified HEAD before any edits ({bp_tail or 'failed on baseline'}).{cwd_hint} Inspect how "
                f"existing tests configure `cwd`, fixtures, or environment variables before editing; NEVER modify "
                f"or shadow existing production class/module attributes or default paths in `__init__` just to "
                f"work around a test runner's current working directory (`cwd`)."
            )
        baseline_block = ("\n" + "\n".join(baseline_notes)) if baseline_notes else ""
        dynamic_parts.append(
            "## Acceptance\nWhen you finish, the plugin runs these commands and reports the results to the lead. "
            "Make them pass:\n"
            + "\n".join(f"- `{c}`" for c in acceptance)
            + baseline_block
            + "\nIMPORTANT: Create or update all requested implementation and test files (you may inspect how an "
            "existing test file sets up fixtures or `cwd` when needed), and verify your new/changed test files "
            "plus the Acceptance command(s) above. Once they pass and any temporary scratch files are removed, "
            "stop and report immediately; do NOT run unrequested linters (e.g. ruff, mypy, flake8, eslint) or "
            "unrelated full-repository suites not listed above. Never modify unrelated files or production default "
            "paths to work around pre-existing test failures outside your task scope or allowed_paths."
        )
    return "\n\n".join([brief.strip(), *static_parts, *dynamic_parts])

