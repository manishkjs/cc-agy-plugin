"""AST-based class-attribute shadowing, test-stub mutation, and missing-symbol regression guards."""

from __future__ import annotations

import ast
import builtins as _py_builtins
from pathlib import Path
import re
import subprocess

from server.job_git_utils import _git_raw, _run_git
from server.receipts import parse_diff_git_header

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


_DESCRIPTOR_CALL_NAMES = frozenset({"property", "classmethod", "staticmethod", "cached_property"})
_TRY_STAR_TYPES = (ast.Try, getattr(ast, "TryStar", ast.Try))
_MATCH_TYPES = (getattr(ast, "Match", ()),) if hasattr(ast, "Match") else ()


def _is_descriptor_expr(expr: ast.AST | None) -> bool:
    if not isinstance(expr, ast.Call):
        return False
    fn = expr.func
    if isinstance(fn, ast.Name):
        return fn.id in _DESCRIPTOR_CALL_NAMES
    if isinstance(fn, ast.Attribute):
        return fn.attr in _DESCRIPTOR_CALL_NAMES
    return False


def _collect_ast_class_map(tree: ast.Module) -> dict[str, ast.ClassDef]:
    """Map qualified class names (e.g. `Runner` or `Outer.Inner`) in `tree` to their `ast.ClassDef`."""
    mapping: dict[str, ast.ClassDef] = {}

    def _visit_block(stmts: list[ast.stmt], prefix: str) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.ClassDef):
                qual = f"{prefix}.{stmt.name}" if prefix else stmt.name
                mapping[qual] = stmt
                _visit_block(stmt.body, qual)

    _visit_block(tree.body, "")
    return mapping


def _collect_direct_class_attrs(cls_node: ast.ClassDef) -> set[str]:
    """Return non-dunder class-body attributes defined with concrete values directly on `cls_node`."""
    attrs: set[str] = set()

    def _add_target(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            name = t.id
            if not (name.startswith("__") and name.endswith("__")):
                attrs.add(name)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                _add_target(elt)

    for stmt in cls_node.body:
        if isinstance(stmt, ast.Assign) and not _is_descriptor_expr(stmt.value):
            for t in stmt.targets:
                _add_target(t)
        elif (
            isinstance(stmt, ast.AnnAssign)
            and stmt.value is not None
            and not _is_descriptor_expr(stmt.value)
        ):
            _add_target(stmt.target)
    return attrs


def _collect_param_is_not_none_guards(test_expr: ast.AST, param_names: set[str]) -> frozenset[str]:
    """Return optional parameter names guaranteed `is not None` when `test_expr` holds."""
    if (
        isinstance(test_expr, ast.Compare)
        and len(test_expr.ops) == 1
        and isinstance(test_expr.ops[0], ast.IsNot)
        and len(test_expr.comparators) == 1
        and isinstance(test_expr.comparators[0], ast.Constant)
        and test_expr.comparators[0].value is None
        and isinstance(test_expr.left, ast.Name)
        and test_expr.left.id in param_names
    ):
        return frozenset({test_expr.left.id})
    if isinstance(test_expr, ast.BoolOp) and isinstance(test_expr.op, ast.And):
        found: set[str] = set()
        for val in test_expr.values:
            found |= _collect_param_is_not_none_guards(val, param_names)
        return frozenset(found)
    return frozenset()


def _rhs_reads_unshadowed_class_attr(
    val_expr: ast.AST | None,
    attr_name: str,
    self_name: str,
    cls_name: str,
    *,
    already_written: bool,
) -> bool:
    """Return True if `val_expr` copies/wraps `self.<attr_name>` or `<cls_name>.<attr_name>` without conditional fallback."""
    if val_expr is None or already_written:
        return False
    has_ref = False
    for node in ast.walk(val_expr):
        if isinstance(node, (ast.IfExp, ast.BoolOp, ast.Lambda)):
            return False
        if (
            isinstance(node, ast.Attribute)
            and node.attr == attr_name
            and isinstance(node.value, ast.Name)
            and node.value.id in (self_name, cls_name)
        ):
            has_ref = True
    return has_ref


def _dotted_expr_parts(expr: ast.AST) -> list[str] | None:
    """Unroll a dotted `ast.Name` / `ast.Attribute` chain (e.g. `pkg.base.Base` -> `['pkg', 'base', 'Base']`)."""
    parts: list[str] = []
    cur: ast.AST = expr
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


def _collect_init_self_writes(
    cls_node: ast.ClassDef,
) -> dict[str, list[tuple[str, int, bool]]]:
    """Collect `self.<attr>` writes in `__init__` and `self.<helper>` methods called from `__init__`.

    Returns `{attr: [(method_name, lineno, is_guarded_or_class_default_copy), ...]}`.
    """
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        stmt.name: stmt
        for stmt in cls_node.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "__init__" not in methods:
        return {}

    writes: dict[str, list[tuple[str, int, bool]]] = {}
    visited: set[str] = set()
    queue: list[str] = ["__init__"]

    while queue:
        m_name = queue.pop(0)
        if m_name in visited or m_name not in methods:
            continue
        visited.add(m_name)
        fn_node = methods[m_name]
        all_pos = list(fn_node.args.posonlyargs) + list(fn_node.args.args)
        if not all_pos:
            continue
        self_name = all_pos[0].arg

        # Identify parameters with default `None` on `__init__` for caller-override guard detection
        none_default_params: set[str] = set()
        if m_name == "__init__":
            defaults = list(fn_node.args.defaults)
            non_self = all_pos[1:]
            pad = len(non_self) - len(defaults)
            for idx_p, arg_node in enumerate(non_self):
                def_idx = idx_p - pad
                if def_idx >= 0 and isinstance(defaults[def_idx], ast.Constant) and defaults[def_idx].value is None:
                    none_default_params.add(arg_node.arg)
            for kw_arg, kw_def in zip(fn_node.args.kwonlyargs, fn_node.args.kw_defaults):
                if isinstance(kw_def, ast.Constant) and kw_def.value is None:
                    none_default_params.add(kw_arg.arg)

        def _is_guarded_rhs(attr_name: str, val_expr: ast.AST | None, guarded_params: frozenset[str]) -> bool:
            if guarded_params and val_expr is not None:
                if any(isinstance(n, ast.Name) and n.id in guarded_params for n in ast.walk(val_expr)):
                    return True
            return _rhs_reads_unshadowed_class_attr(
                val_expr,
                attr_name,
                self_name,
                cls_node.name,
                already_written=(attr_name in writes),
            )

        def _scan_expr(expr: ast.AST | None, guarded_params: frozenset[str], fallback_lineno: int) -> None:
            if expr is None:
                return
            for sub in ast.walk(expr):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(sub, ast.Call):
                    if (
                        isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == self_name
                        and sub.func.attr in methods
                        and sub.func.attr not in visited
                    ):
                        queue.append(sub.func.attr)
                    elif (
                        isinstance(sub.func, ast.Name)
                        and sub.func.id == "setattr"
                        and len(sub.args) == 3
                        and isinstance(sub.args[0], ast.Name)
                        and sub.args[0].id == self_name
                        and isinstance(sub.args[1], ast.Constant)
                        and isinstance(sub.args[1].value, str)
                    ):
                        s_attr = sub.args[1].value
                        if s_attr and not (s_attr.startswith("__") and s_attr.endswith("__")):
                            is_g = _is_guarded_rhs(s_attr, sub.args[2], guarded_params)
                            writes.setdefault(s_attr, []).append(
                                (m_name, getattr(sub, "lineno", fallback_lineno), is_g)
                            )

        def _walk_stmts(stmts: list[ast.stmt], guarded_params: frozenset[str]) -> None:
            for st in stmts:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(st, ast.If):
                    _scan_expr(st.test, guarded_params, st.lineno)
                    g_params = guarded_params | _collect_param_is_not_none_guards(st.test, none_default_params)
                    _walk_stmts(st.body, g_params)
                    _walk_stmts(st.orelse, guarded_params)
                    continue
                if _MATCH_TYPES and isinstance(st, _MATCH_TYPES):
                    _scan_expr(getattr(st, "subject", None), guarded_params, st.lineno)
                    for mc in getattr(st, "cases", []):
                        _scan_expr(getattr(mc, "guard", None), guarded_params, st.lineno)
                        _walk_stmts(mc.body, guarded_params)
                    continue
                if isinstance(st, (ast.For, ast.AsyncFor)):
                    _scan_expr(st.iter, guarded_params, st.lineno)
                    _walk_stmts(st.body, guarded_params)
                    _walk_stmts(st.orelse, guarded_params)
                    continue
                if isinstance(st, ast.While):
                    _scan_expr(st.test, guarded_params, st.lineno)
                    _walk_stmts(st.body, guarded_params)
                    _walk_stmts(st.orelse, guarded_params)
                    continue
                if isinstance(st, (ast.With, ast.AsyncWith)):
                    for item in st.items:
                        _scan_expr(item.context_expr, guarded_params, st.lineno)
                    _walk_stmts(st.body, guarded_params)
                    continue
                if isinstance(st, _TRY_STAR_TYPES):
                    _walk_stmts(st.body, guarded_params)
                    for h in getattr(st, "handlers", []):
                        if isinstance(h, ast.ExceptHandler):
                            _walk_stmts(h.body, guarded_params)
                    _walk_stmts(st.orelse, guarded_params)
                    _walk_stmts(st.finalbody, guarded_params)
                    continue

                # Simple statement: scan expressions for helper calls and `setattr(self, ...)`
                _scan_expr(st, guarded_params, st.lineno)
                if isinstance(st, ast.Assign):
                    for tgt in st.targets:
                        targets_iter = tgt.elts if isinstance(tgt, (ast.Tuple, ast.List)) else [tgt]
                        for item_t in targets_iter:
                            if (
                                isinstance(item_t, ast.Attribute)
                                and isinstance(item_t.value, ast.Name)
                                and item_t.value.id == self_name
                            ):
                                is_g = _is_guarded_rhs(item_t.attr, st.value, guarded_params)
                                writes.setdefault(item_t.attr, []).append((m_name, st.lineno, is_g))
                elif (
                    isinstance(st, ast.AnnAssign)
                    and st.value is not None
                    and isinstance(st.target, ast.Attribute)
                    and isinstance(st.target.value, ast.Name)
                    and st.target.value.id == self_name
                ):
                    is_g = _is_guarded_rhs(st.target.attr, st.value, guarded_params)
                    writes.setdefault(st.target.attr, []).append((m_name, st.lineno, is_g))
                elif (
                    isinstance(st, ast.AugAssign)
                    and isinstance(st.target, ast.Attribute)
                    and isinstance(st.target.value, ast.Name)
                    and st.target.value.id == self_name
                ):
                    writes.setdefault(st.target.attr, []).append((m_name, st.lineno, False))

        _walk_stmts(fn_node.body, frozenset())

    return writes


def _collect_top_level_import_map(
    tree: ast.Module,
    current_rel: str,
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Return `(symbol_imports, module_aliases)` from top-level imports in `tree`.

    `symbol_imports[local_name] = (module_dotted_base, exported_symbol_name)`
    `module_aliases[local_alias] = module_dotted_base`
    """
    pkg_parts = list(Path(current_rel).parent.parts)
    if pkg_parts == ["."]:
        pkg_parts = []
    sym_map: dict[str, tuple[str, str]] = {}
    mod_map: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            base_parts = list(pkg_parts)
            if stmt.level > 0:
                trim = stmt.level - 1
                base_parts = base_parts[: max(0, len(base_parts) - trim)] if trim > 0 else base_parts
                mod_parts = base_parts + (stmt.module.split(".") if stmt.module else [])
            else:
                mod_parts = stmt.module.split(".") if stmt.module else []
            mod_dotted = ".".join(p for p in mod_parts if p)
            if not mod_dotted:
                continue
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                sym_map[local] = (mod_dotted, alias.name)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.asname:
                    mod_map[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    mod_map[top] = top
    return sym_map, mod_map


def detect_class_attribute_shadowing(
    worktree_root: Path,
    files_changed: list[str],
    *,
    baseline_ref: str = "HEAD",
) -> list[str]:
    """Detect non-test `.py` edits where `C.__init__` (or helpers it calls) newly shadows an existing `HEAD` class attribute."""
    if not files_changed or not worktree_root.is_dir():
        return []

    parsed_head_cache: dict[str, tuple[ast.Module, dict[str, ast.ClassDef]] | None] = {}

    def _load_head_module(rel_path: str) -> tuple[ast.Module, dict[str, ast.ClassDef]] | None:
        if rel_path in parsed_head_cache:
            return parsed_head_cache[rel_path]
        res = _run_git(["show", f"{baseline_ref}:{rel_path}"], cwd=worktree_root, check=False)
        if res.returncode != 0 or not res.stdout:
            parsed_head_cache[rel_path] = None
            return None
        try:
            mod_tree = ast.parse(res.stdout, filename=rel_path)
        except SyntaxError:
            parsed_head_cache[rel_path] = None
            return None
        entry = (mod_tree, _collect_ast_class_map(mod_tree))
        parsed_head_cache[rel_path] = entry
        return entry

    def _resolve_dotted_module_rel(mod_dotted: str) -> str | None:
        rel_stem = mod_dotted.replace(".", "/")
        cands = [f"{rel_stem}.py", f"{rel_stem}/__init__.py"]
        if not rel_stem.startswith("src/"):
            cands.extend([f"src/{rel_stem}.py", f"src/{rel_stem}/__init__.py"])
        for cand in cands:
            if _load_head_module(cand) is not None:
                return cand
        return None

    def _resolve_imported_chain(full_parts: list[str]) -> tuple[str, str] | None:
        for split_idx in range(len(full_parts) - 1, 0, -1):
            cand_rel = _resolve_dotted_module_rel(".".join(full_parts[:split_idx]))
            if cand_rel is not None:
                return cand_rel, ".".join(full_parts[split_idx:])
        return None

    contract_cache: dict[tuple[str, str], tuple[set[str], set[str]]] = {}

    def _effective_head_contract(
        rel_path: str,
        qual_cls: str,
        visiting: frozenset[tuple[str, str]] = frozenset(),
    ) -> tuple[set[str], set[str]]:
        key = (rel_path, qual_cls)
        if key in contract_cache:
            return contract_cache[key]
        if key in visiting:
            return set(), set()
        loaded = _load_head_module(rel_path)
        if loaded is None:
            return set(), set()
        mod_tree, mod_classes = loaded
        sym_imports, mod_aliases = _collect_top_level_import_map(mod_tree, rel_path)
        next_visiting = visiting | {key}
        cls_node = mod_classes.get(qual_cls)
        if cls_node is None:
            q_parts = qual_cls.split(".")
            reexp_chain: list[str] = []
            if q_parts[0] in sym_imports:
                imp_mod, orig_sym = sym_imports[q_parts[0]]
                reexp_chain = [*imp_mod.split("."), orig_sym, *q_parts[1:]]
            elif q_parts[0] in mod_aliases and len(q_parts) >= 2:
                reexp_chain = [*mod_aliases[q_parts[0]].split("."), *q_parts[1:]]
            resolved_reexp = _resolve_imported_chain(reexp_chain) if reexp_chain else None
            if resolved_reexp is not None:
                res = _effective_head_contract(resolved_reexp[0], resolved_reexp[1], next_visiting)
                contract_cache[key] = res
                return res
            return set(), set()

        eff_attrs = set(_collect_direct_class_attrs(cls_node))
        eff_unguarded = {
            a
            for a, locs in _collect_init_self_writes(cls_node).items()
            if any(not is_g for _, _, is_g in locs)
        }
        parent_prefix = qual_cls.rsplit(".", 1)[0] if "." in qual_cls else ""

        for base_expr in cls_node.bases:
            b_rel: str | None = None
            b_qual: str | None = None
            b_parts = _dotted_expr_parts(base_expr)
            if not b_parts:
                continue
            joined_base = ".".join(b_parts)
            scoped_cand = f"{parent_prefix}.{joined_base}" if parent_prefix else joined_base
            if scoped_cand in mod_classes:
                b_rel, b_qual = rel_path, scoped_cand
            elif joined_base in mod_classes:
                b_rel, b_qual = rel_path, joined_base
            else:
                imp_chain: list[str] = []
                if b_parts[0] in sym_imports:
                    imp_mod, orig_sym = sym_imports[b_parts[0]]
                    imp_chain = [*imp_mod.split("."), orig_sym, *b_parts[1:]]
                elif b_parts[0] in mod_aliases and len(b_parts) >= 2:
                    imp_chain = [*mod_aliases[b_parts[0]].split("."), *b_parts[1:]]
                resolved_imp = _resolve_imported_chain(imp_chain) if imp_chain else None
                if resolved_imp is not None:
                    b_rel, b_qual = resolved_imp
            if b_rel and b_qual:
                inh_attrs, inh_unguarded = _effective_head_contract(b_rel, b_qual, next_visiting)
                eff_attrs |= inh_attrs
                eff_unguarded |= inh_unguarded

        contract_cache[key] = (eff_attrs, eff_unguarded)
        return eff_attrs, eff_unguarded

    warnings: list[str] = []
    for rel in files_changed:
        if not rel.endswith(".py") or _is_test_or_doc_path(rel):
            continue
        wt_file = worktree_root / rel
        if not wt_file.is_file() or wt_file.is_symlink():
            continue
        loaded_head = _load_head_module(rel)
        if loaded_head is None:
            continue
        _head_tree, head_classes = loaded_head
        try:
            wt_tree = ast.parse(wt_file.read_text(encoding="utf-8", errors="replace"), filename=rel)
        except SyntaxError:
            continue
        wt_classes = _collect_ast_class_map(wt_tree)
        for qual_cls in head_classes:
            wt_cls = wt_classes.get(qual_cls)
            if wt_cls is None:
                continue
            head_attrs, head_unguarded_writes = _effective_head_contract(rel, qual_cls)
            if not head_attrs:
                continue
            wt_init_writes = _collect_init_self_writes(wt_cls)
            for attr, locs in wt_init_writes.items():
                if attr not in head_attrs or attr in head_unguarded_writes:
                    continue
                unguarded_locs = [(m_name, ln) for m_name, ln, is_guarded in locs if not is_guarded]
                if not unguarded_locs:
                    continue
                m_name, lineno = unguarded_locs[0]
                loc_method = f"{qual_cls}.__init__" if m_name == "__init__" else f"{qual_cls}.{m_name} (called from __init__)"
                warnings.append(
                    f"ClassAttributeShadowed: {rel}:{lineno}: {loc_method} newly assigns self.{attr}, "
                    f"shadowing existing class attribute {qual_cls}.{attr} on HEAD "
                    f"(breaks monkeypatch.setattr({qual_cls}, {attr!r}, ...) unless only set when explicitly passed by caller)."
                )
    return warnings


_TEST_DIR_SEGMENTS = frozenset({"tests", "test", "__tests__"})
_TEST_FILENAME_RE = re.compile(
    r"^(?:test_.+\.(?:py|rb|rs|c|cc|cpp|sh)|.+_(?:test|spec)\.(?:py|go|rs|ts|tsx|js|jsx|rb)|.+\.(?:test|spec)\.(?:ts|tsx|js|jsx|mjs|cjs)|.+Test\.(?:java|kt))$"
)
_NEW_TEST_BLOCK_LINE_RE = re.compile(
    r"^\+\s*(?:(?:async\s+)?def\s+test_\w+|func\s+Test\w+|(?:it|test|describe)\s*\(|#\[test\]|@Test\b|name:\s*[\"'])"
)
_STUB_OR_MOCK_MUTATION_RE = re.compile(
    r"(?:\b(?:httpmock|Mockito|mockito|GoMock|gomock|Sinon|sinon|nock|respx|mocker|monkeypatch|MagicMock|AsyncMock|Mock|mock|mocks|mocked|unmock|Stub|stub|stubs|stubbed)\b"
    r"|\b(?:mock|stub|Mock|Stub)_[A-Za-z0-9_]+\b"
    r"|\b[A-Za-z0-9_]+_(?:mock|stub|mocks|stubs)\b"
    r"|\b(?:mock|stub|Mock|Stub)[A-Z][A-Za-z0-9_]*\b"
    r"|\b[a-z][A-Za-z0-9_]*(?:Mock|Stub|Mocks|Stubs)\b"
    r"|\b(?:patch|spyOn)\s*\(|\bresponses\.add\b)"
)
_EXISTING_TEST_EXPECTATION_RE = re.compile(
    r"(?:\b(?:wantStdout|wantStderr|wantOut|wantErr|expectedOutput|expectedStdout|expectedStderr|expectedError)\s*[:=]"
    r"|\bassert(?:Equal|Equals|StrictEqual|DeepEqual|That|True|False)?\b"
    r"|\bexpect\s*\([^)]*\)\s*\.\s*(?:toEqual|toStrictEqual|toBe|toMatch|toThrow)\b)"
)
_TEST_FUNC_HEADER_RE = re.compile(
    r"(?:(?:async\s+)?def\s+(test_\w+)|func\s+(Test\w+)|(?:it|test)\s*\(\s*(['\"`][^'\"`]+['\"`])|(?:public\s+)?void\s+(test\w+))"
)
_SUBTEST_CASE_RE = re.compile(r"^\s*name\s*:\s*(['\"`][^'\"`]+['\"`])")
_EXPLICIT_TESTS_ONLY_RE = re.compile(
    r"\b(?:only|solely)\s+(?:add|write|create|update)\s+(?:unit\s+|integration\s+|regression\s+)?tests?\b"
    r"|\bwithout\s+modifying\s+(?:any\s+)?(?:production|source|implementation)\b"
    r"|\bdo\s+not\s+modify\s+(?:any\s+)?(?:production|source|implementation)\b",
    re.I,
)
_EXPLICIT_EXISTING_TEST_UPDATE_RE = re.compile(
    r"\b(?:update|modify|adjust|adapt|rewrite|sync)\s+"
    r"(?:the\s+|any\s+|all\s+)?(?:existing\s+|affected\s+|corresponding\s+|related\s+)?"
    r"(?:unit\s+|integration\s+|regression\s+)?(?:test(?:s|\s+files?|\s+cases?|\s+expectations?|\s+assertions?|\s+snapshots?)|test_\w+\.py|\S+_test\.go|\S+\.(?:test|spec)\.[jt]sx?)\b"
    r"|\b(?:change|fix)\s+(?:the\s+|any\s+|all\s+)?existing\s+(?:unit\s+|integration\s+|regression\s+)?test(?:s|\s+expectations?|\s+assertions?|\s+cases?|\s+files?)\b"
    r"|\btests?\s+to\s+expect\b",
    re.I,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[.;:!?\n]|\b(?:but|however)\b", re.I)
_CLAUSE_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|don['\u2019]t|does\s+not|doesn['\u2019]t|did\s+not|didn['\u2019]t"
    r"|never|must\s+not|mustn['\u2019]t|should\s+not|shouldn['\u2019]t"
    r"|can\s*not|cannot|can['\u2019]t|avoid|refrain\s+from|without|no|not"
    r"|rather\s+than|instead\s+of|forbid\w*|prohibit\w*)\b",
    re.I,
)
_TEST_AUTHORING_INTENT_RE = re.compile(
    r"\b(?:write|add|create|expand|increase|improve)\s+"
    r"(?:a\s+|an\s+|new\s+|more\s+|missing\s+|additional\s+)?"
    r"(?:(?:unit|integration|regression|edge[- ]case|table[- ]driven|parameterized|parametrized)\s+)*tests?\b"
    r"|\b(?:increase|improve|expand|add)\s+(?:unit\s+|branch\s+|line\s+)?test\s+coverage\b",
    re.I,
)
_PROD_CHANGE_INTENT_RE = re.compile(
    r"\b(?:fix|implement|refactor|support|resolve|repair|patch|migrate|convert|replace"
    r"|make|handle|change|ensure|rename|remove|delete|allow|enable|disable|expose|export|preserve|prevent|avoid)\b"
    r"|\badd\s+(?:a\s+|an\s+|the\s+)?(?:--?[A-Za-z0-9_-]+|[A-Za-z0-9_-]+\s+(?:flag|option|parameter|argument|field|method|function|endpoint|command|subcommand|route|header|hook|handler|helper|class|module|config|setting|support))\b"
    r"|\bupdate\s+(?:the\s+)?(?:implementation|function|method|class|handler|logic|module|behavior)\b",
    re.I,
)


def _prompt_requests_test_authoring(prompt_text: str) -> bool:
    """Return True when the prompt explicitly asks for test-only edits or ordinary test/coverage authoring."""
    if not prompt_text:
        return False
    if _EXPLICIT_TESTS_ONLY_RE.search(prompt_text):
        return True
    if _TEST_AUTHORING_INTENT_RE.search(prompt_text) and not _PROD_CHANGE_INTENT_RE.search(prompt_text):
        return True
    return False


_FORBID_EXISTING_TEST_UPDATE_RE = re.compile(
    r"\b(?:do\s+not|don['\u2019]t|never|must\s+not|should\s+not|cannot|can['\u2019]t|without|avoid|refrain\s+from)\s+"
    r"(?:ever\s+)?(?:update|updating|modify|modifying|alter|altering|change|changing|rewrite|rewriting|edit|editing|touch|touching|weaken|weakening)\s+"
    r"(?:the\s+|any\s+|all\s+)?(?:pre-?existing\s+|existing\s+)?(?:unit\s+|integration\s+|regression\s+)?test",
    re.I,
)


def _prompt_forbids_existing_test_updates(prompt_text: str) -> bool:
    """Return True when the prompt explicitly forbids modifying/rewriting existing tests."""
    if not prompt_text:
        return False
    return bool(_FORBID_EXISTING_TEST_UPDATE_RE.search(prompt_text))


def _prompt_requests_existing_test_updates(prompt_text: str) -> bool:
    """Return True when the prompt explicitly instructs updating existing test expectations/assertions (not negated)."""
    if not prompt_text:
        return False
    for m in _EXPLICIT_EXISTING_TEST_UPDATE_RE.finditer(prompt_text):
        prefix = prompt_text[: m.start()]
        clause_parts = _CLAUSE_BOUNDARY_RE.split(prefix)
        local_clause = clause_parts[-1] if clause_parts else prefix
        if _CLAUSE_NEGATION_RE.search(local_clause):
            continue
        return True
    return False


def _unexempted_stub_mutation_warnings(warnings: list[str] | None, prompt_text: str) -> list[str]:
    """Filter test-mutation warnings against explicit prompt intent."""
    if not warnings:
        return []
    if _prompt_requests_test_authoring(prompt_text):
        return []
    if _prompt_requests_existing_test_updates(prompt_text):
        return [w for w in warnings if not str(w).startswith("ExistingTestExpectationsModified:")]
    return list(warnings)


def _is_test_file_path(rel_path: str) -> bool:
    """Return True when `rel_path` is a unit/integration/spec test file across supported languages."""
    norm = rel_path.replace("\\", "/").strip("/")
    if not norm:
        return False
    parts = norm.split("/")
    if any(seg in _TEST_DIR_SEGMENTS for seg in parts[:-1]):
        return True
    return bool(_TEST_FILENAME_RE.match(parts[-1]))


def detect_existing_test_stub_mutations(
    worktree_root: Path,
    diff_text: str,
    files_changed: list[str],
    *,
    baseline_ref: str = "HEAD",
) -> list[str]:
    """Warn when a worker alters mock/stub expectations across >= 3 existing test blocks in a pre-existing test file."""
    if not files_changed or not diff_text.strip() or not worktree_root.is_dir():
        return []
    test_targets = {
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if _is_test_file_path(rel)
    }
    if not test_targets:
        return []

    preexisting_tests: set[str] = set()
    for rel in test_targets:
        cat_res = _run_git(["cat-file", "-e", f"{baseline_ref}:{rel}"], cwd=worktree_root, check=False)
        if cat_res.returncode == 0:
            preexisting_tests.add(rel)
    if not preexisting_tests:
        return []

    file_chunks: dict[str, list[str]] = {}
    cur_rel: str | None = None
    for dline in diff_text.splitlines():
        if dline.startswith("diff --git "):
            hdr = parse_diff_git_header(dline)
            cpath = ((hdr[1] or hdr[0]) if hdr else "").replace("\\", "/").strip("/")
            cur_rel = cpath if cpath in preexisting_tests else None
            if cur_rel is not None:
                file_chunks.setdefault(cur_rel, [])
            continue
        if cur_rel is not None:
            file_chunks[cur_rel].append(dline)

    from collections import Counter

    warnings: list[str] = []
    for rel in sorted(file_chunks):
        lines = file_chunks[rel]
        hunks: list[tuple[int, str, list[str]]] = []
        cur_old_len = 0
        cur_hdr_ctx = ""
        cur_hunk: list[str] | None = None
        for ln in lines:
            m_hunk = re.match(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$", ln)
            if m_hunk:
                if cur_hunk is not None:
                    hunks.append((cur_old_len, cur_hdr_ctx, cur_hunk))
                cur_old_len = int(m_hunk.group(2)) if m_hunk.group(2) is not None else 1
                cur_hdr_ctx = (m_hunk.group(5) or "").strip()
                cur_hunk = []
                continue
            if cur_hunk is not None and not ln.startswith(("--- ", "+++ ")):
                cur_hunk.append(ln)
        if cur_hunk is not None:
            hunks.append((cur_old_len, cur_hdr_ctx, cur_hunk))

        mutated_stub_blocks = 0
        mutated_expectation_funcs: set[str] = set()
        for hunk_idx, (old_len, hdr_ctx, hlines) in enumerate(hunks):
            if old_len <= 0:
                continue
            m_hdr_fn = _TEST_FUNC_HEADER_RE.search(hdr_ctx)
            base_fn_id = (
                f"fn:{next(g for g in m_hdr_fn.groups() if g)}"
                if m_hdr_fn
                else f"hunk:{hunk_idx}"
            )
            cur_block_id = base_fn_id
            clusters: list[tuple[str, list[str]]] = []
            cur_cluster: list[str] = []
            cur_cluster_block_id = cur_block_id
            for hl in hlines:
                body = hl[1:] if hl else ""
                if hl.startswith((" ", "-")):
                    m_fn = _TEST_FUNC_HEADER_RE.search(body)
                    if m_fn and not hl.startswith("+"):
                        if cur_cluster and hl.startswith(" "):
                            clusters.append((cur_cluster_block_id, cur_cluster))
                            cur_cluster = []
                        base_fn_id = f"fn:{next(g for g in m_fn.groups() if g)}"
                        cur_block_id = base_fn_id
                    else:
                        m_sub = _SUBTEST_CASE_RE.match(body)
                        if m_sub:
                            if cur_cluster and hl.startswith(" "):
                                clusters.append((cur_cluster_block_id, cur_cluster))
                                cur_cluster = []
                            cur_block_id = f"{base_fn_id}::{m_sub.group(1)}"
                if hl.startswith(("+", "-")):
                    if not cur_cluster:
                        cur_cluster_block_id = cur_block_id
                    cur_cluster.append(hl)
                elif cur_cluster:
                    clusters.append((cur_cluster_block_id, cur_cluster))
                    cur_cluster = []
            if cur_cluster:
                clusters.append((cur_cluster_block_id, cur_cluster))
            for block_id, cluster in clusters:
                if not any(cl.startswith("-") for cl in cluster):
                    continue
                if any(cl.startswith("+") and _NEW_TEST_BLOCK_LINE_RE.search(cl) for cl in cluster):
                    continue
                if any(_STUB_OR_MOCK_MUTATION_RE.search(cl[1:]) for cl in cluster):
                    mutated_stub_blocks += 1
                minus_exp = [
                    cl[1:].strip()
                    for cl in cluster
                    if cl.startswith("-") and _EXISTING_TEST_EXPECTATION_RE.search(cl[1:])
                ]
                plus_exp = [
                    cl[1:].strip()
                    for cl in cluster
                    if cl.startswith("+") and _EXISTING_TEST_EXPECTATION_RE.search(cl[1:])
                ]
                if minus_exp:
                    minus_counts = Counter(minus_exp)
                    plus_counts = Counter(plus_exp)
                    if any(plus_counts[m] < cnt for m, cnt in minus_counts.items()):
                        mutated_expectation_funcs.add(block_id)
        if mutated_stub_blocks >= 3:
            warnings.append(
                f"ExistingTestStubsModified: {rel} modified mock/stub expectations across {mutated_stub_blocks} "
                f"pre-existing test blocks; inspect get_diff (focus_symbols) to verify no unrequested API/HTTP "
                f"calls or breaking contract changes were masked."
            )
        elif len(mutated_expectation_funcs) >= 2:
            warnings.append(
                f"ExistingTestExpectationsModified: {rel} altered pre-existing test expectations across "
                f"{len(mutated_expectation_funcs)} test functions; verify production code matches the specification "
                f"rather than rewriting existing test assertions."
            )
    return warnings


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
    "delegate_file_batch", "draft_document", "write_tests_from_cases", "replicate_edit", "list_jobs",
    "allowed_paths", "files_changed", "acceptance", "acceptance_passed", "acceptance_results",
    "scope_violations", "scope_violation", "missing_prompt_symbols", "auto_apply", "in_place",
    "follow_up_prompt", "job_id", "timeout_seconds", "max_tool_calls", "max_total_tokens",
    "thinking_level", "allow_scope_violations", "include_paths", "force_retry", "resume_job_id",
} | {name for name in dir(_py_builtins) if not name.startswith("_")})


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
    cmd = ["grep", "--no-color", "-I", "-z", "-w", "-F"]
    for term in search_terms:
        cmd.extend(["-e", term])
    cmd.extend([ref, "--"])
    try:
        res = _git_raw(
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

