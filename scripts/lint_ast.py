#!/usr/bin/env python3
"""Zero-dependency AST static linter for `server/` and `scripts/`.

Enforces core Google Python Style Guide (`go/pystyle`) safety invariants on every
`scripts/validate.sh` run without requiring external packages:
1. Valid Python 3.10+ syntax across all `server/*.py` and `scripts/*.py` modules.
2. Zero bare `except:` clauses (`go/pystyle` S2.4).
3. Zero mutable default arguments (`[]`, `{}`, `set()` literals in `def`, `go/pystyle` S2.12).
4. Zero duplicate constant keys in dictionary literals.
5. Sub-100ms cold-start guard: `scripts/pre_tool_guard.py` must not import heavy
   server modules (`server.jobs`, `server.receipts`, `server.regions`) at top level.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path
import sys

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
_BUILTINS: frozenset[str] = frozenset(dir(builtins)) | frozenset(
    {"__name__", "__file__", "__doc__", "__package__", "__all__", "__annotations__"}
)
_MUTABLE_CALL_NAMES: frozenset[str] = frozenset({"list", "dict", "set", "bytearray"})
_FORBIDDEN_HOOK_TOP_IMPORTS: frozenset[str] = frozenset(
    {"server.jobs", "server.receipts", "server.regions", "server.agy_runner"}
)


def _is_mutable_default(expr: ast.AST) -> bool:
    if isinstance(expr, (ast.List, ast.Dict, ast.Set)):
        return True
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        return expr.func.id in _MUTABLE_CALL_NAMES
    return False


def lint_file(path: Path, rel_str: str) -> list[str]:
    """Return static lint findings for one Python source file."""
    errors: list[str] = []
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=rel_str)
    except SyntaxError as exc:
        return [f"{rel_str}:{exc.lineno}: SyntaxError: {exc.msg}"]

    if rel_str == "scripts/pre_tool_guard.py":
        for top_node in tree.body:
            if isinstance(top_node, ast.ImportFrom) and top_node.module in _FORBIDDEN_HOOK_TOP_IMPORTS:
                errors.append(
                    f"{rel_str}:{top_node.lineno}: forbidden top-level import `{top_node.module}` "
                    "(regresses PreToolUse cold-start latency)"
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            errors.append(f"{rel_str}:{node.lineno}: bare `except:` is forbidden (catch specific exceptions)")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]
            for default_expr in defaults:
                if _is_mutable_default(default_expr):
                    errors.append(
                        f"{rel_str}:{node.lineno}: mutable default argument in `{node.name}()`"
                    )
        elif isinstance(node, ast.Dict):
            seen_keys: set[tuple[type[object], object]] = set()
            for key_node in node.keys:
                if isinstance(key_node, ast.Constant) and isinstance(
                    key_node.value, (str, int, bytes)
                ):
                    sig = (type(key_node.value), key_node.value)
                    if sig in seen_keys:
                        errors.append(
                            f"{rel_str}:{key_node.lineno}: duplicate dictionary key {key_node.value!r}"
                        )
                    seen_keys.add(sig)

    defined_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                defined_names.add(arg.arg)
            if node.args.vararg is not None:
                defined_names.add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                defined_names.add(node.args.kwarg.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined_names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined_names.add(node.name)
        elif hasattr(ast, "MatchAs") and isinstance(node, ast.MatchAs) and node.name:
            defined_names.add(node.name)
        elif hasattr(ast, "TypeVar") and isinstance(node, getattr(ast, "TypeVar", ())) and hasattr(node, "name"):
            defined_names.add(node.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined_names and node.id not in _BUILTINS:
                errors.append(
                    f"{rel_str}:{node.lineno}: undefined name `{node.id}` (F821)"
                )
    return errors


def main() -> int:
    """Lint all Python modules under `server/` and `scripts/`."""
    targets = sorted(PLUGIN_ROOT.glob("server/*.py")) + sorted(PLUGIN_ROOT.glob("scripts/*.py"))
    all_errors: list[str] = []
    for file_path in targets:
        rel_str = file_path.relative_to(PLUGIN_ROOT).as_posix()
        all_errors.extend(lint_file(file_path, rel_str))

    if all_errors:
        for err in all_errors:
            sys.stderr.write(f"[lint_ast] {err}\n")
        return 1
    print(f"AST lint passed ({len(targets)} modules checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
