"""Commit-keyed repository map ("scout cache") for worker and batch prompts.

Instead of letting a Gemini worker spend 40+ turns running ``find_file``, ``list_dir`` and
``search_dir`` to rediscover the repository on every job, we build a compact structural map once per
``(HEAD SHA, tracked-file list)`` and hand it to the worker in Turn 1.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from server.regions import estimate_tokens, truncate_to_token_budget

DEFAULT_REPO_MAP_MAX_TOKENS = 1200

_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".hypothesis", ".benchmarks", ".mypy_cache", ".ruff_cache", ".pytype", ".eggs", ".tox", ".nox",
    "dist", "build", "target", "out", ".next", ".nuxt", ".cc-agy-data", ".worktrees", ".ctx-session-reads",
})

_SOURCE_EXTS = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt", ".rb", ".c", ".cc", ".cpp", ".h", ".hpp",
})

_PY_SIG_RE = re.compile(r"^(class\s+[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?|(?:async\s+)?def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\))", re.M)
_DEFAULT_VAL_RE = re.compile(r"""=\s*(?:["'][^"']*["']|[^,)]+)""")
_TS_SIG_RE = re.compile(
    r"^\s*export\s+(?:default\s+)?(?:(?:async\s+)?function|class|interface|type|const)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    re.M,
)
_GO_SIG_RE = re.compile(r"^(?:func\s+(?:\([^)]+\)\s*)?[A-Z][A-Za-z0-9_]*\s*\([^)]*\)|type\s+[A-Z][A-Za-z0-9_]*\s+(?:struct|interface))", re.M)
_RS_SIG_RE = re.compile(r"^\s*pub\s+(?:async\s+)?(?:fn|struct|enum|trait)\s+[A-Za-z_][A-Za-z0-9_]*", re.M)

_MAX_CACHE_ENTRIES = 16
_MEM_CACHE: dict[str, str] = {}


def _git_info(root: Path) -> tuple[str, list[str]]:
    """Return (head_sha or "nohead"/"nogit", sorted relative file paths)."""
    head_sha = "nogit"
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=5, check=False,
        )
        head_sha = head.stdout.strip() if head.returncode == 0 else "nohead"
        ls = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=str(root), capture_output=True, text=True, timeout=10, check=False,
        )
        if ls.returncode == 0 and ls.stdout.strip():
            files = sorted({
                line.strip() for line in ls.stdout.splitlines()
                if line.strip()
                and not any(part in _SKIP_DIRS or part.startswith(".cc-agy") for part in Path(line.strip()).parts)
                and (root / line.strip()).is_file()
            })
            return head_sha, files
    except (OSError, subprocess.SubprocessError):
        head_sha = "nogit"

    # No git: walk with pruning so node_modules/.venv are never descended into.
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".cc-agy"))
        rel_dir = Path(dirpath).relative_to(root)
        for name in sorted(filenames):
            files.append((rel_dir / name).as_posix())
            if len(files) >= 2000:
                return head_sha, files
    return head_sha, files


def _strip_py_defaults(param_str: str) -> str:
    """Replace default parameter values with '=...' while respecting quotes, escapes, and brackets."""
    out: list[str] = []
    i = 0
    n = len(param_str)
    while i < n:
        ch = param_str[i]
        if ch == "=":
            out.append("=...")
            i += 1
            depth = 0
            quote: str | None = None
            while i < n:
                c = param_str[i]
                if quote is not None:
                    if c == "\\" and i + 1 < n:
                        i += 2
                        continue
                    if param_str.startswith(quote, i):
                        i += len(quote)
                        quote = None
                        continue
                    i += 1
                else:
                    if param_str.startswith(('"""', "'''"), i):
                        quote = param_str[i : i + 3]
                        i += 3
                        continue
                    if c in ('"', "'"):
                        quote = c
                        i += 1
                        continue
                    if c in "([{":
                        depth += 1
                    elif c in ")]}":
                        if depth == 0:
                            break
                        depth -= 1
                    elif c == "," and depth == 0:
                        break
                    i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _extract_py_ast_signatures(text: str, limit: int = 8, include_methods: bool = False) -> list[str]:
    """Extract top-level classes and public functions with 1-indexed line numbers using ast."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    classes: list[str] = []
    pub_funcs: list[str] = []
    priv_funcs: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start) or start
            span = f"L{start}-L{end}" if end > start else f"L{start}"
            methods: list[str] = []
            max_methods = 10 if include_methods else 4
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mname = sub.name
                    if mname.startswith("_") and mname not in ("__init__", "__call__", "__iter__", "__aiter__"):
                        continue
                    ml = getattr(sub, "lineno", 0)
                    methods.append(f"{mname}@L{ml}" if include_methods else mname)
                    if len(methods) >= max_methods:
                        break
            m_suffix = f" [{', '.join(methods)}]" if methods else ""
            entry = f"class {node.name} ({span}){m_suffix}"
            if len(entry) > 120 and not include_methods:
                entry = entry[:117] + "..."
            classes.append(entry)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start) or start
            span = f"L{start}-L{end}" if end > start else f"L{start}"
            prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
            arg_names = [a.arg for a in node.args.posonlyargs + node.args.args]
            if arg_names and arg_names[0] in ("self", "cls"):
                arg_names = arg_names[1:]
            if node.args.vararg:
                arg_names.append(f"*{node.args.vararg.arg}")
            if node.args.kwonlyargs:
                arg_names.extend(f"{a.arg}=..." for a in node.args.kwonlyargs[:3])
            sig = f"{prefix}{node.name}({', '.join(arg_names[:4])}{', ...' if len(arg_names) > 4 else ''}) ({span})"
            if node.name.startswith("_"):
                priv_funcs.append(sig)
            else:
                pub_funcs.append(sig)
    combined = classes + pub_funcs + priv_funcs
    return combined[:limit]


def _line_number_at(text: str, pos: int) -> int:
    return text.count("\n", 0, max(0, pos)) + 1


def _extract_signatures(path: Path, ext: str, limit: int = 8) -> list[str]:
    try:
        if path.stat().st_size > 250_000:
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    out: list[str] = []
    if ext == ".py":
        ast_sigs = _extract_py_ast_signatures(text, limit=limit, include_methods=False)
        if ast_sigs:
            return ast_sigs
        for m in _PY_SIG_RE.finditer(text):
            lineno = _line_number_at(text, m.start())
            sig = " ".join(m.group(1).split())
            if sig.startswith(("def ", "async def ")) and "(" in sig:
                prefix, rest = sig.split("(", 1)
                sig = f"{prefix}({_strip_py_defaults(rest)}"
            if len(sig) > 88:
                sig = sig[:85] + "..."
            out.append(f"{sig} (L{lineno})")
            if len(out) >= limit:
                break
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        for m in _TS_SIG_RE.finditer(text):
            lineno = _line_number_at(text, m.start())
            out.append(f"export {m.group(1)} (L{lineno})")
            if len(out) >= limit:
                break
    elif ext == ".go":
        for m in _GO_SIG_RE.finditer(text):
            lineno = _line_number_at(text, m.start())
            sig = " ".join(m.group(0).split())
            out.append(f"{sig[:88]} (L{lineno})")
            if len(out) >= limit:
                break
    elif ext == ".rs":
        for m in _RS_SIG_RE.finditer(text):
            lineno = _line_number_at(text, m.start())
            sig = " ".join(m.group(0).split())
            out.append(f"{sig[:88]} (L{lineno})")
            if len(out) >= limit:
                break
    return out


def build_focused_file_outline(
    workspace_root: Path,
    rel_paths: list[str],
    *,
    max_tokens: int = 500,
) -> str:
    """Build a detailed line-numbered symbol outline for specific target/candidate files."""
    if not rel_paths or max_tokens <= 0:
        return ""
    root = Path(workspace_root).resolve()
    lines: list[str] = []
    seen: set[str] = set()
    for rel in rel_paths[:10]:
        clean_rel = str(rel or "").strip().strip("/")
        if not clean_rel or clean_rel in seen:
            continue
        seen.add(clean_rel)
        fpath = (root / clean_rel).resolve()
        try:
            if not fpath.is_relative_to(root) or not fpath.is_file() or fpath.stat().st_size > 250_000:
                continue
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        total_lines = text.count("\n") + 1
        ext = fpath.suffix.lower()
        if ext == ".py":
            sigs = _extract_py_ast_signatures(text, limit=16, include_methods=True)
        else:
            sigs = _extract_signatures(fpath, ext, limit=14)
        if sigs:
            lines.append(f"- `{clean_rel}` ({total_lines} lines): " + "; ".join(sigs))
        else:
            lines.append(f"- `{clean_rel}` ({total_lines} lines)")
    if not lines:
        return ""
    raw = "\n".join(lines)
    if estimate_tokens(raw) > max_tokens:
        raw, _, _ = truncate_to_token_budget(raw, max_tokens)
    return raw


def _detect_test_hints(root: Path, files: list[str]) -> list[str]:
    hints: list[str] = []
    file_set = set(files)
    if "pyproject.toml" in file_set or "pytest.ini" in file_set or any(f.startswith("tests/") and f.endswith(".py") for f in files):
        hints.append("pytest -q")
    if "package.json" in file_set:
        try:
            pkg = json.loads((root / "package.json").read_text(encoding="utf-8", errors="replace"))
            scripts = (pkg.get("scripts") or {}) if isinstance(pkg, dict) else {}
            for k in ("test", "typecheck", "lint", "build"):
                if k in scripts:
                    hints.append(f"npm run {k}")
        except Exception:
            pass
    return hints[:4]


def repo_map_cache_key(workspace_root: Path, *, _info: tuple[str, list[str]] | None = None) -> str:
    """Key on repo path, HEAD SHA, discovered file list, and content digest of dirty/untracked files."""
    root = Path(workspace_root).resolve()
    head_sha, files = _info if _info is not None else _git_info(root)
    h = hashlib.sha256(b"repo_map_v2\0")
    for part in (str(root), head_sha, "\n".join(files)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    if head_sha != "nogit":
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
                cwd=str(root), capture_output=True, text=True, timeout=10, check=False,
            )
            if status.returncode == 0:
                raw_tokens = status.stdout.split("\0")
                dirty_rows: list[str] = []
                idx = 0
                while idx < len(raw_tokens) and len(dirty_rows) < 2000:
                    entry = raw_tokens[idx]
                    idx += 1
                    if len(entry) < 4:
                        continue
                    xy = entry[:2]
                    rel_part = entry[3:]
                    if xy[0] in ("R", "C") or xy[1] in ("R", "C"):
                        idx += 1  # skip source path token in porcelain -z rename/copy
                    p = root / rel_part
                    try:
                        st = p.lstat()
                        if p.is_file() and not p.is_symlink() and st.st_size <= 250_000 and len(dirty_rows) < 500:
                            with p.open("rb") as fh:
                                f_digest = hashlib.sha256(fh.read(250_000)).hexdigest()[:16]
                            dirty_rows.append(f"{xy}:{rel_part}:{st.st_size}:{st.st_mtime_ns}:{f_digest}")
                        else:
                            dirty_rows.append(f"{xy}:{rel_part}:{st.st_size}:{st.st_mtime_ns}")
                    except OSError:
                        dirty_rows.append(f"{xy}:{rel_part}:missing")
                for row in sorted(dirty_rows):
                    h.update(f"{row}\n".encode("utf-8"))
            else:
                h.update(b"nostatus")
        except (OSError, subprocess.SubprocessError):
            h.update(b"nostatus")
    else:
        for rel in files[:500]:
            p = root / rel
            try:
                st = p.lstat()
                if p.is_file() and not p.is_symlink() and st.st_size <= 250_000:
                    with p.open("rb") as fh:
                        f_digest = hashlib.sha256(fh.read(65_536)).hexdigest()[:16]
                    h.update(f"{rel}:{st.st_size}:{f_digest}\n".encode("utf-8"))
            except OSError:
                pass
    return h.hexdigest()[:16]


def build_repo_map(
    workspace_root: Path,
    *,
    max_tokens: int = DEFAULT_REPO_MAP_MAX_TOKENS,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> str:
    """Build or load a commit-keyed repository map bounded to ``max_tokens``."""
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        return ""
    git_info = _git_info(root)
    key = f"{repo_map_cache_key(root, _info=git_info)}_{int(max_tokens)}"
    eff_cache_dir = cache_dir
    if eff_cache_dir is None:
        default_store = root / ".cc-agy-data"
        if default_store.is_dir():
            eff_cache_dir = default_store
    cache_file = (eff_cache_dir / f"repo_map_{key}.txt") if eff_cache_dir is not None else None

    if not refresh:
        if key in _MEM_CACHE:
            return _MEM_CACHE[key]
        if cache_file is not None and cache_file.is_file():
            try:
                cached = cache_file.read_text(encoding="utf-8")
                _MEM_CACHE[key] = cached
                return cached
            except OSError:
                pass

    head_sha, files = git_info
    if not files:
        return ""

    test_files = [f for f in files if f.startswith(("tests/", "test/")) or "/test_" in f or f.endswith(("_test.py", ".test.ts", ".spec.ts"))]
    src_files = [f for f in files if f not in test_files and Path(f).suffix.lower() in _SOURCE_EXTS]
    other_files = [f for f in files if f not in test_files and f not in src_files]

    lines: list[str] = [f"Repository map ({len(files)} tracked files, HEAD={head_sha[:12]}):"]
    hints = _detect_test_hints(root, files)
    if hints:
        lines.append("Detected checks: " + ", ".join(f"`{h}`" for h in hints))

    lines.append("Source files & key symbols:")
    for rel in src_files[:80]:
        ext = Path(rel).suffix.lower()
        sigs = _extract_signatures(root / rel, ext, limit=6)
        if sigs:
            lines.append(f"- `{rel}`: " + "; ".join(sigs))
        else:
            lines.append(f"- `{rel}`")
    if len(src_files) > 80:
        lines.append(f"- ... ({len(src_files) - 80} more source files)")

    if test_files:
        shown_tests = ", ".join(f"`{t}`" for t in test_files[:25])
        more = f" (+{len(test_files) - 25} more)" if len(test_files) > 25 else ""
        lines.append(f"Tests ({len(test_files)}): {shown_tests}{more}")

    if other_files:
        shown_other = ", ".join(f"`{o}`" for o in other_files[:15])
        more = f" (+{len(other_files) - 15} more)" if len(other_files) > 15 else ""
        lines.append(f"Config/docs ({len(other_files)}): {shown_other}{more}")

    raw_map = "\n".join(lines)
    if estimate_tokens(raw_map) > max_tokens:
        raw_map, _, _ = truncate_to_token_budget(raw_map, max_tokens)

    while len(_MEM_CACHE) >= _MAX_CACHE_ENTRIES:
        _MEM_CACHE.pop(next(iter(_MEM_CACHE)), None)
    _MEM_CACHE[key] = raw_map
    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(raw_map, encoding="utf-8")
            existing = sorted(cache_file.parent.glob("repo_map_*.txt"), key=lambda p: p.stat().st_mtime)
            for old in existing[:-_MAX_CACHE_ENTRIES]:
                old.unlink(missing_ok=True)
        except OSError:
            pass
    return raw_map
