"""Three focused helper tools for normal chat: the Lead keeps the real work and hands off one bounded job.

Each tool turns its request into one or more file tasks, adds a tool-specific gate on the written
bytes, and runs them through the same isolated job copy, writer ladder and receipts as
``delegate_file_batch``. Nothing reaches the workspace until the Lead calls ``delegate_apply``.

* ``draft_document``: a new doc from purpose, audience, sections and facts. Every fact's quote must
  already appear verbatim in its source file (checked before any model call); the draft must contain
  every section heading and may only link to paths that exist.
* ``write_tests_from_cases``: a new test file from named cases. Every case name and every expected
  value must appear verbatim (so an expectation cannot be quietly edited to make a test pass), and
  the given test command must pass in the job copy.
* ``replicate_edit``: a change the Lead already made by hand in one file, repeated across other files
  as exact find/replace edits only (no whole-file rewrites), then one check command for the set.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

from server.batch_plan import BatchPlan, BatchPlanError, FileTask, PlanProblem, validate_batch_plan

_MD_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_CODE_PATH_RE = re.compile(r"`([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+\.[A-Za-z0-9]{1,8})`")
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#", "tel:")


class AssistRequestError(ValueError):
    def __init__(self, problems: list[PlanProblem]) -> None:
        self.problems = problems
        super().__init__("; ".join(str(p) for p in problems))

    def as_dict(self) -> dict[str, Any]:
        return {"status": "invalid_request", "is_error": True,
                "problems": [{"json_path": p.json_path, "message": p.message} for p in self.problems]}


def _with_note_fingerprint(plan: BatchPlan, note: str) -> BatchPlan:
    digest = hashlib.sha256((plan.raw_fingerprint + "\n" + note).encode("utf-8")).hexdigest()
    return replace(plan, raw_fingerprint=digest)


def _plan_or_raise(raw: dict[str, Any], root: Path, allowed_paths: list[str] | None) -> BatchPlan:
    try:
        return validate_batch_plan(raw, root, allowed_paths=allowed_paths)
    except BatchPlanError as exc:
        raise AssistRequestError(exc.problems) from None


# --------------------------------------------------------------------------------------------
# draft_document
# --------------------------------------------------------------------------------------------


def prepare_document(
    root: Path,
    *,
    path: str,
    purpose: str,
    audience: str,
    sections: list[str],
    facts: list[dict[str, Any]],
    context_files: list[str] | None = None,
    allowed_paths: list[str] | None = None,
) -> tuple[BatchPlan, str, Any]:
    problems: list[PlanProblem] = []
    if not isinstance(sections, list) or not sections or not all(isinstance(s, str) and s.strip() for s in sections):
        problems.append(PlanProblem("$.sections", "must be a non-empty list of section titles"))
        sections = []
    fact_lines: list[str] = []
    for idx, fact in enumerate(facts or []):
        base = f"$.facts[{idx}]"
        if not isinstance(fact, dict):
            problems.append(PlanProblem(base, "must be an object with claim, source and quote"))
            continue
        source, quote, claim = fact.get("source"), fact.get("quote"), fact.get("claim")
        if not all(isinstance(v, str) and v.strip() for v in (source, quote, claim)):
            problems.append(PlanProblem(base, "claim, source and quote are all required strings"))
            continue
        src = (root / source).resolve()
        if not src.is_relative_to(root.resolve()) or not src.is_file():
            problems.append(PlanProblem(f"{base}.source", f"{source!r} is not a file in the workspace"))
            continue
        if quote not in src.read_text(encoding="utf-8", errors="replace"):
            problems.append(PlanProblem(f"{base}.quote", f"quote not found verbatim in {source!r}"))
            continue
        fact_lines.append(f"- {claim.strip()} (source `{source}`: \"{quote.strip()}\")")
    if not (isinstance(purpose, str) and purpose.strip()):
        problems.append(PlanProblem("$.purpose", "required"))
    if not (isinstance(audience, str) and audience.strip()):
        problems.append(PlanProblem("$.audience", "required"))
    if problems:
        raise AssistRequestError(problems)
    target_exists = (root / path).exists()
    intent = (
        f"Write a document for {audience.strip()}. Purpose: {purpose.strip()} "
        f"Use exactly these sections as headings, in order: {', '.join(sections)}. "
        "State only the facts provided, in your own words, and link only to files that exist."
    )
    note = "## Facts (each verified verbatim in the repository)\n" + ("\n".join(fact_lines) or "- (none)")
    plan = _plan_or_raise({"tasks": [{
        "task_id": "doc", "path": path, "action": "modify" if target_exists else "create",
        "intent": intent, "context_files": list(context_files or []),
    }]}, root, allowed_paths)

    def gate(task: FileTask, data: bytes, wt: Path) -> str | None:
        text = data.decode("utf-8", errors="replace")
        headings = [ln.lstrip("#").strip().lower() for ln in text.splitlines() if ln.lstrip().startswith("#")]
        missing = [s for s in sections if not any(s.strip().lower() in h for h in headings)]
        if missing:
            return f"document is missing section heading(s): {missing}"
        doc_dir = PurePosixPath(task.path).parent
        dead: list[str] = []
        for link in _MD_LINK_RE.findall(text):
            if link.startswith(_EXTERNAL_PREFIXES):
                continue
            rel = link.split("#", 1)[0]
            if rel and not ((wt / doc_dir / rel).exists() or (wt / rel.lstrip("/")).exists()):
                dead.append(link)
        for code_path in _CODE_PATH_RE.findall(text):
            if not (wt / code_path).exists():
                dead.append(code_path)
        if dead:
            return f"document references paths that do not exist: {sorted(set(dead))[:10]}"
        return None

    return _with_note_fingerprint(plan, note), note, gate


# --------------------------------------------------------------------------------------------
# write_tests_from_cases
# --------------------------------------------------------------------------------------------


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def prepare_tests(
    root: Path,
    *,
    test_path: str,
    target_path: str,
    cases: list[dict[str, Any]],
    test_command: str,
    allowed_paths: list[str] | None = None,
) -> tuple[BatchPlan, str, Any]:
    problems: list[PlanProblem] = []
    clean: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, case in enumerate(cases or []):
        base = f"$.cases[{idx}]"
        if not isinstance(case, dict) or not all(isinstance(case.get(k), str) and case.get(k).strip() for k in ("name", "given", "expect")):
            problems.append(PlanProblem(base, "each case needs string name, given and expect"))
            continue
        slug = _slug(case["name"])
        if not slug or slug in seen:
            problems.append(PlanProblem(f"{base}.name", "case names must be unique and contain letters or digits"))
            continue
        seen.add(slug)
        clean.append({"name": case["name"].strip(), "slug": slug, "given": case["given"].strip(), "expect": case["expect"].strip()})
    if not clean and not problems:
        problems.append(PlanProblem("$.cases", "at least one case is required"))
    if not (isinstance(test_command, str) and test_command.strip()):
        problems.append(PlanProblem("$.test_command", "required"))
    if problems:
        raise AssistRequestError(problems)
    note = "## Cases (each must become one test; keep every expected value exactly as written)\n" + "\n".join(
        f"- `{c['slug']}`: given {c['given']} -> expect {c['expect']}" for c in clean)
    intent = (
        f"Write a test file for `{target_path}` with one test per listed case, named after the case. "
        "Use each expected value exactly as given; never change an expectation to make a test pass."
    )
    plan = _plan_or_raise({"tasks": [{
        "task_id": "tests", "path": test_path, "action": "modify" if (root / test_path).exists() else "create",
        "intent": intent, "context_files": [target_path], "checks": [test_command.strip()],
    }]}, root, allowed_paths)

    def gate(task: FileTask, data: bytes, wt: Path) -> str | None:
        text = data.decode("utf-8", errors="replace")
        lowered = text.lower()
        missing = [c["name"] for c in clean if c["slug"] not in lowered and c["name"] not in text]
        if missing:
            return f"test file is missing case(s): {missing}"
        changed = [c["name"] for c in clean if c["expect"] not in text]
        if changed:
            return f"expected value(s) not present verbatim for case(s) {changed}; do not alter expectations"
        return None

    return _with_note_fingerprint(plan, note), note, gate


# --------------------------------------------------------------------------------------------
# replicate_edit
# --------------------------------------------------------------------------------------------


def prepare_replication(
    root: Path,
    *,
    example_path: str,
    targets: list[str],
    note: str = "",
    check_command: str = "",
    allowed_paths: list[str] | None = None,
) -> tuple[BatchPlan, str, None]:
    problems: list[PlanProblem] = []
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) and t.strip() for t in targets):
        problems.append(PlanProblem("$.targets", "must be a non-empty list of file paths"))
        targets = []
    if example_path in targets:
        problems.append(PlanProblem("$.targets", "must not include the example file"))
    diff = subprocess.run(["git", "diff", "--no-color", "HEAD", "--", example_path], cwd=str(root),
                          capture_output=True, text=True, check=False).stdout
    if not diff.strip():
        problems.append(PlanProblem("$.example_path", f"{example_path!r} has no uncommitted change to copy; make the edit there first"))
    if problems:
        raise AssistRequestError(problems)
    shared = f"## Example change already made by hand in `{example_path}`\n```diff\n{diff[:20000]}\n```"
    if note.strip():
        shared += f"\n\n## Lead's note\n{note.strip()}"
    tasks = []
    used_ids: set[str] = set()
    for target in targets:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", PurePosixPath(target).stem)[:48] or "file"
        task_id, n = stem, 2
        while task_id in used_ids:
            task_id, n = f"{stem}_{n}", n + 1
        used_ids.add(task_id)
        tasks.append({
            "task_id": task_id, "path": target, "action": "modify",
            "intent": "Apply the same change as the example to this file, as exact edits only. Change nothing else.",
        })
    raw: dict[str, Any] = {"tasks": tasks}
    if check_command:
        raw["acceptance"] = [check_command]
    plan = _plan_or_raise(raw, root, allowed_paths)
    return _with_note_fingerprint(plan, shared), shared, None
