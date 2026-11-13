"""Shared Git, path-scope, artifact-filtering, and budget helpers for worker jobs."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from server.model_routes import (
    DEFAULT_ECONOMY_FLASH_MAX_TOOL_CALLS,
    DEFAULT_FLASH_MAX_TOOL_CALLS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TOTAL_TOKENS,
)
from server.receipts import (
    parse_diff_git_header,
    parse_unified_header_path,
    sha256_bytes,
)
from server.regions import resolve_cost_profile

def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    # Hooks, fsmonitor, external diff/textconv, and clean/smudge filters stay off and the environment is
    # credential-free: worker or check commands can rewrite a job copy's .git/config or hooks.
    from server.verify import run_git

    return run_git(
        ["-c", "core.quotePath=false", *args],
        cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _git_raw(args: list[str], cwd: Path | str | None = None, **kwargs: Any) -> subprocess.CompletedProcess:
    """A plugin git call with the caller's own ``subprocess.run`` options, guarded like ``_run_git``
    (see ``verify.run_git``)."""
    from server.verify import run_git

    return run_git(args, cwd, **kwargs)


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
        ".vite",
        ".vite-temp",
        ".eslintcache",
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
    ".tsbuildinfo",
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

# Report files that test runners write at the repo root (pytest-cov, coverage.py, nose). The
# host and the worker both produce them when they run the same tests, so capturing one turns
# into a "created on both sides" apply conflict that throws away the worker's real edits.
# Only matched at the repo root, so fixtures such as tests/fixtures/coverage.xml still apply.
_GENERATED_REPORT_NAMES: frozenset[str] = frozenset(
    {
        "coverage.xml",
        "coverage.json",
        "coverage.lcov",
        "lcov.info",
        "nosetests.xml",
    }
)
_FIXTURE_PATH_SEGMENTS: frozenset[str] = frozenset(
    {"testdata", "fixtures", "fixture", "snapshots", "__snapshots__", "golden"}
)
_UNTRACKED_MODEL_OR_DATA_DUMP_SUFFIXES: tuple[str, ...] = (
    ".joblib",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".h5",
    ".onnx",
    ".parquet",
)


def _is_untracked_runtime_artifact_name(rel_path: str) -> bool:
    """Return True for transient runtime log, pid, or serialized model/dataset dump files unless in a fixture dir."""
    norm = rel_path.replace("\\", "/").strip("/")
    if not norm:
        return False
    parts = norm.split("/")
    if any(p.lower() in _FIXTURE_PATH_SEGMENTS for p in parts[:-1]):
        return False
    low_name = parts[-1].lower()
    if low_name.endswith((".log", ".pid", ".pid.lock", *_UNTRACKED_MODEL_OR_DATA_DUMP_SUFFIXES)):
        return True
    return False


_GENERIC_MODULE_STEMS: frozenset[str] = frozenset(
    {"__init__", "init", "index", "main", "test", "tests", "utils", "common", "base", "types", "config", "constants"}
)
_IMPORTABLE_MODULE_EXTS: frozenset[str] = frozenset(
    {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs", ".java", ".kt", ".c", ".cc", ".cpp", ".h", ".hpp", ".rb", ".php", ".cs", ".swift"}
)


def _is_referenced_by_files_or_prompt(
    rel_cand: str,
    ref_files: list[str] | None,
    root: Path | None,
    *,
    prompt_text: str = "",
) -> bool:
    """Return True if `rel_cand` (or its basename with extension, or its module stem for importable code files) is referenced in `prompt_text` or `ref_files`."""
    cand_norm = rel_cand.replace("\\", "/").strip("/")
    if not cand_norm:
        return False
    cand_name = Path(cand_norm).name
    cand_sfx = Path(cand_norm).suffix.lower()
    cand_stem = Path(cand_norm).stem
    cand_no_ext = str(Path(cand_norm).with_suffix("")).replace("\\", "/")
    cand_dotted = cand_no_ext.replace("/", ".")
    prompt_low = (prompt_text or "").lower()
    if prompt_low:
        if cand_norm.lower() in prompt_low or (len(cand_name) >= 4 and "." in cand_name and cand_name.lower() in prompt_low):
            return True
    if not ref_files or root is None or not root.exists():
        return False
    check_stem = (
        cand_sfx in _IMPORTABLE_MODULE_EXTS
        and not _is_untracked_runtime_artifact_name(cand_norm)
        and len(cand_stem) >= 3
        and cand_stem.lower() not in _GENERIC_MODULE_STEMS
    )
    stem_re = re.compile(rf"\b{re.escape(cand_stem)}\b") if check_stem else None
    for rf in ref_files:
        rf_norm = str(rf or "").replace("\\", "/").strip("/")
        if not rf_norm or rf_norm == cand_norm:
            continue
        if cand_sfx == ".go" and Path(rf_norm).suffix.lower() == ".go" and Path(rf_norm).parent == Path(cand_norm).parent:
            return True
        rf_p = root / rf_norm
        try:
            if not rf_p.is_file() or rf_p.is_symlink():
                continue
            with rf_p.open("rb") as fh:
                raw_bytes = fh.read(262_144)
            txt = raw_bytes.decode("utf-8", errors="ignore")
        except OSError:
            continue
        if cand_norm in txt or ("." in cand_name and cand_name in txt):
            return True
        if check_stem and (
            cand_dotted in txt
            or cand_no_ext in txt
            or (stem_re is not None and stem_re.search(txt))
        ):
            return True
    return False


def _explicitly_targets_runtime_artifact(
    rel_cand: str,
    allowed_paths: list[str] | None,
    *,
    prompt_text: str = "",
    ref_files: list[str] | None = None,
    root: Path | None = None,
) -> bool:
    """Return True when `allowed_paths`, `prompt_text`, or `ref_files` explicitly names or references `rel_cand`."""
    if not _is_untracked_runtime_artifact_name(rel_cand):
        return False
    cand_norm = rel_cand.replace("\\", "/").strip("/")
    cand_sfx = Path(cand_norm).suffix.lower()
    for pat in allowed_paths or []:
        p_norm = str(pat or "").replace("\\", "/").strip("/")
        if not p_norm or p_norm in (".", "*", "**", "**/*"):
            continue
        if (
            p_norm == cand_norm
            or (
                cand_sfx
                and p_norm.lower().endswith(cand_sfx)
                and _matches_allowed_paths(cand_norm, [p_norm])
            )
            or (
                "/" in cand_norm
                and p_norm.split("/", 1)[0] == cand_norm.split("/", 1)[0]
                and _matches_allowed_paths(cand_norm, [p_norm])
            )
        ):
            return True
    if (prompt_text or ref_files) and _is_referenced_by_files_or_prompt(
        cand_norm, ref_files, root, prompt_text=prompt_text
    ):
        return True
    return False


def _is_tracked_in_git(root: Path, rel_path: str) -> bool:
    try:
        res = _git_raw(
            ["ls-files", "--error-unmatch", "--", rel_path],
            cwd=str(root),
            capture_output=True,
            check=False,
            timeout=5,
        )
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


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
        or (len(parts) == 1 and name in _GENERATED_REPORT_NAMES)
        or name.startswith(".coverage.")
        or name.endswith(_EPHEMERAL_CACHE_SUFFIXES)
    ):
        return True
    if root is not None and _is_untracked_runtime_artifact_name(norm):
        if not _is_tracked_in_git(root, norm):
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


def _is_gitignored_on_host(workspace_root: Path, rel_path: str) -> bool:
    """True if `rel_path` is ignored by the host repo's .gitignore rules (False when not a git repo)."""
    try:
        res = _git_raw(
            ["check-ignore", "-q", "--", rel_path],
            cwd=str(workspace_root),
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0


def _is_ignored_build_artifact(rel_path: str, root: Path | None = None) -> bool:
    """Return True if an already-.gitignored path is inside a build/output/cache directory or a setuptools-scm _version.py."""
    if _is_ephemeral_build_or_test_artifact(rel_path, root=root):
        return True
    norm = rel_path.strip("/")
    if not norm:
        return False
    parts = norm.split("/")
    for part in parts:
        if part in _IGNORED_BUILD_OUTPUT_DIRS:
            return True
    if parts[-1] in ("_version.py", "__version__.py"):
        return True
    return False


# DEFAULT_MAX_TOOL_CALLS, DEFAULT_FLASH_MAX_TOOL_CALLS, DEFAULT_ECONOMY_FLASH_MAX_TOOL_CALLS,
# and DEFAULT_MAX_TOTAL_TOKENS are imported from server.model_routes (single source of truth).
DEFAULT_MAX_WAIT_SLICE_SECONDS = 240.0
DEFAULT_ACCEPTANCE_TIMEOUT = 300.0
DEFAULT_BOUNDED_ACCEPTANCE_TIMEOUT = 90.0
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
    max_tool_calls: int | None,
    max_total_tokens: int | None,
    *,
    model: str | None = None,
) -> tuple[int | None, int | None]:
    """Per-turn worker caps (defaults 55 tool calls on Flash in balanced, 100 on Flash in economy and on Pro/Argon/Carbon,
    5M tokens; env overrides; 0 disables)."""
    if max_tool_calls is not None:
        calls = max_tool_calls
    elif "flash" in (model or "").lower() and not str(os.environ.get("CC_AGY_MAX_TOOL_CALLS") or "").strip():
        flash_default = (
            DEFAULT_ECONOMY_FLASH_MAX_TOOL_CALLS
            if resolve_cost_profile() == "economy"
            else DEFAULT_FLASH_MAX_TOOL_CALLS
        )
        calls = _env_int("CC_AGY_FLASH_MAX_TOOL_CALLS", flash_default)
    else:
        calls = _env_int("CC_AGY_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS)
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


_GO_BUILD_HEAD_RE = re.compile(
    r"^([\s(]*(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s;|&()]+)\s+)*"
    r"(?:\S*/)?go(?:\s+(?:-C|--C)(?:=(?:\"[^\"]*\"|'[^']*'|[^\s;|&()]+)|\s+(?:\"[^\"]*\"|'[^']*'|[^\s;|&()]+)))*"
    r"\s+build)(?=\s|$|\))"
)


def _normalize_readonly_go_build_acceptance(cmd: str) -> str:
    """Insert `-o /dev/null` into bare `go build ...` segments (with no `-o`/`--o`) without re-quoting shell syntax."""
    parts = re.split(r"(\s*(?:&&|\|\||;|\||\n)\s*)", cmd)
    changed = False
    rebuilt: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1 or not part.strip():
            rebuilt.append(part)
            continue
        m = _GO_BUILD_HEAD_RE.match(part)
        if not m:
            rebuilt.append(part)
            continue
        rest = part[m.end() :]
        rest_for_tokens = re.sub(r"\s*(?:\d*>>?\s*(?:&\d+|\S+)|\d*<\s*\S+)\s*", " ", rest.rstrip(") \t"))
        try:
            rest_tokens = shlex.split(rest_for_tokens)
        except ValueError:
            rest_tokens = rest_for_tokens.split()
        if any(a in ("-o", "--o") or a.startswith(("-o=", "--o=")) for a in rest_tokens):
            rebuilt.append(part)
            continue
        rebuilt.append(f"{m.group(1)} -o /dev/null{rest}")
        changed = True
    return "".join(rebuilt) if changed else cmd


def validate_acceptance(acceptance: Any, workdir: Path) -> list[str]:
    """Acceptance commands must be read-only checks (tests, linters); mutating commands are refused.

    The plugin itself runs them, so a mutating command would be an unguarded way to edit the repo.
    """
    if acceptance is None:
        return []
    if not isinstance(acceptance, list) or not all(isinstance(c, str) for c in acceptance):
        raise ValueError("acceptance must be a list of shell command strings")
    cmds = [_normalize_readonly_go_build_acceptance(c.strip()) for c in acceptance if c.strip()]
    if len(cmds) > MAX_ACCEPTANCE_COMMANDS:
        raise ValueError(f"acceptance has {len(cmds)} commands; at most {MAX_ACCEPTANCE_COMMANDS} allowed")
    from scripts.pre_tool_guard import detect_bash_workspace_mutation

    for cmd in cmds:
        verdict = detect_bash_workspace_mutation(
            cmd, workdir, cwd=workdir, allow_workspace_inline_imports=True
        )
        if verdict.get("blocked"):
            raise ValueError(
                f"acceptance command {cmd!r} is refused because it can modify the workspace: "
                f"{verdict.get('reason')}"
            )
    return cmds


def _hash_path(p: Path) -> str | None:
    """Return a content fingerprint for a workspace path (symlink target, file SHA-256, or None if absent)."""
    if p.is_symlink():
        return f"symlink:{os.readlink(p)}"
    if p.exists() and p.is_file():
        return sha256_bytes(p.read_bytes())
    return None


def _snapshot_untracked_hashes(root: Path) -> dict[str, str | None]:
    """Fingerprint every untracked, non-ignored file so in_place jobs can skip ones they did not touch."""
    res = _git_raw(
        ["ls-files", "--others", "--exclude-standard", "-z"],
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

