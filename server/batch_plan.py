"""Batch plans: the Lead's file-by-file plan for a multi-file change, validated without any model call.

A batch plan is a JSON document the Lead writes to disk (so the per-file details never sit in its
own context) and hands to ``delegate_file_batch``. Each entry is a :class:`FileTask`: one file to
create or modify, the outcome wanted, the files to show the writer, the exact interface other files
will rely on, the tasks it must wait for, and the read-only checks that must pass before the file
counts as done.

``validate_batch_plan`` is deterministic. It checks every task and reports *all* problems at once,
each with a JSON path (``$.tasks[3].path``), so the Lead can fix the plan in a single turn.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

ACTIONS = ("create", "modify")
TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_TASKS = 200
MAX_CONTEXT_FILES = 12
MAX_CHECKS = 4
MAX_INTERFACE_CODE_LINES = 20

# Paths a writer may never touch, whatever allowed_paths says (matched case-insensitively, any depth).
PROTECTED_DIR_NAMES = frozenset({".git", ".cc-agy-data", ".hg", ".svn"})
PROTECTED_FILE_NAMES = frozenset({".mcp.json", ".netrc", ".npmrc", ".pypirc"})
PROTECTED_NAME_PREFIXES = (".env",)
PROTECTED_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
# Writes here are allowed but flagged for explicit review in the digest and receipt.
REVIEW_PATTERNS = (
    ".github/workflows/*", ".gitlab-ci.yml", "Jenkinsfile", "Makefile", "Dockerfile", "*.lock",
    "package-lock.json", "pnpm-lock.yaml", "pyproject.toml", "setup.py", "setup.cfg", ".claude/*",
)
FORMATTER_ALLOWLIST = (
    "ruff format", "black", "isort", "gofmt -w", "goimports -w", "prettier --write", "npx prettier --write",
    "clang-format -i", "rustfmt", "terraform fmt", "shfmt -w", "google-java-format -i",
)
_SHELL_META_RE = re.compile(r"[;&|`$<>\\\n]|\$\(")


@dataclass(frozen=True)
class FileTask:
    """One file the batch should produce or change."""

    task_id: str
    path: str
    action: str
    intent: str
    context_files: tuple[str, ...] = ()
    interface: str = ""
    depends_on: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    repro_test: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("context_files", "depends_on", "checks"):
            data[key] = list(data[key])
        return data

    def fingerprint(self) -> str:
        """SHA-256 of the canonical task JSON; identical tasks resume as already done."""
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PlanProblem:
    json_path: str
    message: str

    def __str__(self) -> str:
        return f"{self.json_path}: {self.message}"


class BatchPlanError(ValueError):
    """Raised with every problem found in a batch plan."""

    def __init__(self, problems: list[PlanProblem]) -> None:
        self.problems = list(problems)
        lines = "\n".join(f"- {p}" for p in self.problems)
        super().__init__(f"Batch plan has {len(self.problems)} problem(s):\n{lines}")

    def as_dict(self) -> dict[str, Any]:
        return {"problems": [{"json_path": p.json_path, "message": p.message} for p in self.problems]}


@dataclass(frozen=True)
class BatchPlan:
    tasks: tuple[FileTask, ...]
    shared_context: tuple[str, ...] = ()
    acceptance: tuple[str, ...] = ()
    formatter: str = ""
    raw_fingerprint: str = field(default="")


def is_protected_path(rel_path: str) -> bool:
    """True for paths no writer may touch: VCS dirs, env files, plugin data, Claude settings, key material."""
    parts = [p.casefold() for p in PurePosixPath(rel_path).parts]
    if not parts:
        return True
    if any(p in PROTECTED_DIR_NAMES for p in parts):
        return True
    name = parts[-1]
    if name in PROTECTED_FILE_NAMES or name.startswith(PROTECTED_NAME_PREFIXES) or name.endswith(PROTECTED_SUFFIXES):
        return True
    for idx in range(len(parts) - 1):
        if parts[idx] == ".claude" and fnmatch.fnmatchcase(parts[idx + 1], "settings*.json"):
            return True
    return False


def needs_review(rel_path: str) -> bool:
    posix = PurePosixPath(rel_path).as_posix()
    name = PurePosixPath(posix).name
    return any(fnmatch.fnmatchcase(posix, pat) or fnmatch.fnmatchcase(name, pat) for pat in REVIEW_PATTERNS)


def _git_ignored(root: Path, rel: str) -> bool:
    from server.verify import run_git

    res = run_git(["check-ignore", "-q", "--", rel], cwd=str(root), capture_output=True, check=False)
    return res.returncode == 0


def validate_formatter(formatter: str) -> str | None:
    """Return a problem description unless formatter is one allowlisted command with a single {path}."""
    cmd = formatter.strip()
    if cmd.count("{path}") != 1:
        return "must contain {path} exactly once"
    if _SHELL_META_RE.search(cmd.replace("{path}", "")):
        return "must be a single command without shell operators, redirection or substitution"
    if not any(cmd == f"{allowed} {{path}}" or cmd.startswith(f"{allowed} ") for allowed in FORMATTER_ALLOWLIST):
        return f"must start with an allowlisted formatter: {list(FORMATTER_ALLOWLIST)}"
    return None


def _check_rel_path(value: Any, root: Path, json_path: str, problems: list[PlanProblem]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        problems.append(PlanProblem(json_path, "must be a non-empty relative path"))
        return None
    raw = value.strip()
    if raw.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", raw):
        problems.append(PlanProblem(json_path, f"{raw!r} is absolute; use a path relative to the workspace"))
        return None
    if "\\" in raw or "\x00" in raw:
        problems.append(PlanProblem(json_path, f"{raw!r} contains a backslash or NUL byte"))
        return None
    posix = PurePosixPath(raw)
    if any(part == ".." for part in posix.parts):
        problems.append(PlanProblem(json_path, f"{raw!r} leaves the workspace"))
        return None
    norm = posix.as_posix()
    root_resolved = root.resolve()
    resolved = (root / norm).resolve()
    if not resolved.is_relative_to(root_resolved):
        problems.append(PlanProblem(json_path, f"{raw!r} resolves outside the workspace (symlink?)"))
        return None
    probe = root
    for part in posix.parts:
        probe = probe / part
        if probe.is_symlink():
            problems.append(PlanProblem(json_path, f"{raw!r} goes through a symlink ({probe.relative_to(root).as_posix()!r})"))
            return None
    if is_protected_path(norm) or is_protected_path(resolved.relative_to(root_resolved).as_posix()):
        problems.append(PlanProblem(json_path, f"{raw!r} is a protected path and can never be written or read"))
        return None
    return norm


def _str_list(value: Any, json_path: str, problems: list[PlanProblem], limit: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        problems.append(PlanProblem(json_path, "must be a list of strings"))
        return []
    if len(value) > limit:
        problems.append(PlanProblem(json_path, f"has {len(value)} entries; at most {limit} allowed"))
    return [v.strip() for v in value if v.strip()]


def _check_commands(cmds: list[str], root: Path, json_path: str, problems: list[PlanProblem]) -> None:
    from server.jobs import validate_acceptance

    for idx, cmd in enumerate(cmds):
        try:
            validate_acceptance([cmd], root)
        except ValueError as exc:
            problems.append(PlanProblem(f"{json_path}[{idx}]", str(exc)))


def _find_cycles(deps: dict[str, list[str]]) -> list[list[str]]:
    """Return each distinct dependency cycle once, as a closed list of task ids."""
    cycles: list[list[str]] = []
    seen_keys: set[frozenset[str]] = set()
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for nxt in deps.get(node, []):
            if nxt not in deps:
                continue
            if state.get(nxt) == 1:
                cycle = stack[stack.index(nxt):] + [nxt]
                key = frozenset(cycle)
                if key not in seen_keys:
                    seen_keys.add(key)
                    cycles.append(cycle)
            elif state.get(nxt) is None:
                visit(nxt)
        stack.pop()
        state[node] = 2

    for node in deps:
        if state.get(node) is None:
            visit(node)
    return cycles


def _reaches(deps: dict[str, list[str]], start: str, target: str) -> bool:
    todo, seen = [start], set()
    while todo:
        node = todo.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        todo.extend(deps.get(node, []))
    return False


def validate_batch_plan(
    raw: Any,
    workspace_root: str | Path,
    *,
    allowed_paths: list[str] | None = None,
) -> BatchPlan:
    """Validate a batch plan and return it as typed tasks, or raise BatchPlanError with every problem."""
    from scripts.pre_tool_guard import lint_delegation_prompt
    from server.jobs import _matches_allowed_paths

    root = Path(workspace_root).resolve()
    problems: list[PlanProblem] = []
    if not isinstance(raw, dict):
        raise BatchPlanError([PlanProblem("$", "batch plan must be a JSON object with a 'tasks' list")])
    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise BatchPlanError([PlanProblem("$.tasks", "must be a non-empty list of tasks")])
    if len(raw_tasks) > MAX_TASKS:
        problems.append(PlanProblem("$.tasks", f"has {len(raw_tasks)} tasks; at most {MAX_TASKS} per batch"))

    shared: list[str] = []
    for idx, entry in enumerate(_str_list(raw.get("shared_context"), "$.shared_context", problems, MAX_CONTEXT_FILES)):
        norm = _check_rel_path(entry, root, f"$.shared_context[{idx}]", problems)
        if norm and not (root / norm).is_file():
            problems.append(PlanProblem(f"$.shared_context[{idx}]", f"{norm!r} does not exist"))
        elif norm and _git_ignored(root, norm):
            problems.append(PlanProblem(f"$.shared_context[{idx}]", f"{norm!r} is gitignored and is not copied into the job"))
        elif norm:
            shared.append(norm)
    acceptance = _str_list(raw.get("acceptance"), "$.acceptance", problems, 8)
    _check_commands(acceptance, root, "$.acceptance", problems)
    formatter = raw.get("formatter") or ""
    if formatter:
        reason = validate_formatter(formatter) if isinstance(formatter, str) else "must be a string"
        if reason:
            problems.append(PlanProblem("$.formatter", reason))
            formatter = ""

    tasks: list[FileTask] = []
    seen_ids: dict[str, int] = {}
    path_owners: dict[str, list[tuple[str, int]]] = {}
    deps: dict[str, list[str]] = {}
    for idx, entry in enumerate(raw_tasks):
        base = f"$.tasks[{idx}]"
        if not isinstance(entry, dict):
            problems.append(PlanProblem(base, "must be an object"))
            continue
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
            problems.append(PlanProblem(f"{base}.task_id", "must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}"))
            task_id = None
        elif task_id in seen_ids:
            problems.append(PlanProblem(f"{base}.task_id", f"duplicate of $.tasks[{seen_ids[task_id]}].task_id"))
            task_id = None
        else:
            seen_ids[task_id] = idx

        path = _check_rel_path(entry.get("path"), root, f"{base}.path", problems)
        if path is not None and allowed_paths and not _matches_allowed_paths(path, allowed_paths):
            problems.append(PlanProblem(f"{base}.path", f"{path!r} is outside allowed_paths {allowed_paths}"))
            path = None

        action = entry.get("action")
        if action not in ACTIONS:
            problems.append(PlanProblem(f"{base}.action", f"must be one of {list(ACTIONS)}"))
            action = None

        intent = entry.get("intent")
        if not isinstance(intent, str) or not intent.strip():
            problems.append(PlanProblem(f"{base}.intent", "must describe the outcome in prose"))
            intent = ""
        else:
            lint = lint_delegation_prompt(intent)
            if not lint.get("ok", True):
                problems.append(PlanProblem(f"{base}.intent", f"dictates code: {lint.get('reason')}. Describe the outcome instead."))

        context_files: list[str] = []
        for c_idx, c_path in enumerate(_str_list(entry.get("context_files"), f"{base}.context_files", problems, MAX_CONTEXT_FILES)):
            c_norm = _check_rel_path(c_path, root, f"{base}.context_files[{c_idx}]", problems)
            if c_norm and not (root / c_norm).is_file():
                problems.append(PlanProblem(f"{base}.context_files[{c_idx}]", f"{c_norm!r} does not exist"))
            elif c_norm and _git_ignored(root, c_norm):
                problems.append(PlanProblem(f"{base}.context_files[{c_idx}]", f"{c_norm!r} is gitignored and is not copied into the job"))
            elif c_norm:
                context_files.append(c_norm)

        interface = entry.get("interface") or ""
        if not isinstance(interface, str):
            problems.append(PlanProblem(f"{base}.interface", "must be a string"))
            interface = ""
        elif interface and not lint_delegation_prompt(interface, max_consecutive_code_lines=MAX_INTERFACE_CODE_LINES).get("ok", True):
            problems.append(PlanProblem(f"{base}.interface", f"is an implementation, not an interface (over {MAX_INTERFACE_CODE_LINES} code lines)"))
            interface = ""
        depends_on = _str_list(entry.get("depends_on"), f"{base}.depends_on", problems, MAX_TASKS)
        checks = _str_list(entry.get("checks"), f"{base}.checks", problems, MAX_CHECKS)
        _check_commands(checks, root, f"{base}.checks", problems)
        repro = entry.get("repro_test") or ""
        if repro:
            if not isinstance(repro, str):
                problems.append(PlanProblem(f"{base}.repro_test", "must be a command string"))
                repro = ""
            else:
                _check_commands([repro], root, f"{base}.repro_test", problems)

        if task_id and path:
            path_owners.setdefault(path, []).append((task_id, idx))
            deps[task_id] = depends_on
        if task_id and path and action and intent:
            tasks.append(FileTask(
                task_id=task_id, path=path, action=action, intent=intent.strip(),
                context_files=tuple(context_files), interface=interface.strip(),
                depends_on=tuple(depends_on), checks=tuple(checks), repro_test=repro.strip(),
            ))

    known = set(seen_ids)
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict) or not isinstance(entry.get("depends_on"), list):
            continue
        for d_idx, dep in enumerate(entry["depends_on"]):
            if dep == entry.get("task_id"):
                problems.append(PlanProblem(f"$.tasks[{idx}].depends_on[{d_idx}]", "a task cannot depend on itself (cycle)"))
            elif dep not in known:
                problems.append(PlanProblem(f"$.tasks[{idx}].depends_on[{d_idx}]", f"unknown task_id {dep!r}"))
    for cycle in _find_cycles({k: [d for d in v if d != k] for k, v in deps.items()}):
        problems.append(PlanProblem(
            f"$.tasks[{seen_ids[cycle[0]]}].depends_on",
            "dependency cycle: " + " -> ".join(cycle),
        ))

    # Several tasks may touch one file only when every pair is ordered (a strict chain).
    actions = {e.get("task_id"): e.get("action") for e in raw_tasks if isinstance(e, dict)}
    for path, owners in path_owners.items():
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                (a_id, _), (b_id, b_idx) = owners[i], owners[j]
                if not (_reaches(deps, b_id, a_id) or _reaches(deps, a_id, b_id)):
                    problems.append(PlanProblem(
                        f"$.tasks[{b_idx}].path",
                        f"{path!r} is also written by task {a_id!r}; order them with depends_on",
                    ))
        # The first writer in dependency order must match the file on disk; later writers modify.
        ordered = sorted(owners, key=lambda o: sum(1 for other, _ in owners if other != o[0] and _reaches(deps, o[0], other)))
        exists = (root / path).exists()
        for pos, (tid, t_idx) in enumerate(ordered):
            action = actions.get(tid)
            if pos == 0 and action == "create" and exists:
                problems.append(PlanProblem(f"$.tasks[{t_idx}].action", f"{path!r} already exists; use 'modify'"))
            elif pos == 0 and action == "modify" and not exists:
                problems.append(PlanProblem(f"$.tasks[{t_idx}].action", f"{path!r} does not exist; use 'create'"))
            elif pos > 0 and action != "modify":
                problems.append(PlanProblem(f"$.tasks[{t_idx}].action", f"{path!r} is written earlier in this plan; use 'modify'"))

    if problems:
        raise BatchPlanError(problems)
    canonical = json.dumps(
        {"tasks": [t.as_dict() for t in tasks], "shared_context": shared, "acceptance": acceptance, "formatter": formatter},
        sort_keys=True, separators=(",", ":"),
    )
    return BatchPlan(
        tasks=tuple(tasks), shared_context=tuple(shared), acceptance=tuple(acceptance),
        formatter=formatter, raw_fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _plan_file_is_private(path: Path) -> bool:
    """A plan outside the workspace must be owned by us and not writable by group or others."""
    import os
    import stat

    st = path.stat()
    return st.st_uid == os.getuid() and not (st.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def load_batch_plan(
    plan_path: str | Path,
    workspace_root: str | Path,
    *,
    extra_acceptance: list[str] | None = None,
    **kwargs: Any,
) -> BatchPlan:
    """Read a batch plan JSON file (inside the workspace, or a private file under /tmp) and validate it."""
    root = Path(workspace_root).resolve()
    candidate = Path(plan_path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    tmp_root = Path("/tmp").resolve()
    if not (resolved.is_relative_to(root) or resolved.is_relative_to(tmp_root)):
        raise BatchPlanError([PlanProblem("$", f"plan file {plan_path!r} must live inside the workspace or /tmp")])
    try:
        if resolved.is_relative_to(tmp_root) and not _plan_file_is_private(resolved):
            raise BatchPlanError([PlanProblem("$", f"plan file {plan_path!r} must be owned by you and not group/world-writable")])
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BatchPlanError([PlanProblem("$", f"plan file {plan_path!r} not found")]) from None
    except json.JSONDecodeError as exc:
        raise BatchPlanError([PlanProblem("$", f"plan file is not valid JSON: {exc}")]) from None
    if extra_acceptance and isinstance(raw, dict):
        raw = dict(raw)
        raw["acceptance"] = list(raw.get("acceptance") or []) + list(extra_acceptance)
    return validate_batch_plan(raw, root, **kwargs)
