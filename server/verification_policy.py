"""Single source of truth for classifying worker verification results.

Separates verification into three explicit policies so reporting, auto-apply,
batch task summaries, and hook guardrails never drift on compile smoke checks,
trivial no-op commands, or discovered tests:

1. Command strength (`command_strength`, `is_behavioral_check`):
   Classifies every shell command by verification strength regardless of whether
   it was explicitly supplied or auto-discovered:
   - `behavioral`: known test runners (`pytest`, `go test`, `cargo test`, `vitest`, etc.)
   - `custom`: real project scripts/harnesses (`make test`, `./run_tests.sh`, `python3 -c "...assert..."`)
   - `syntax_only`: parse/compile/lint/type/static checks (`py_compile`, `tsc --noEmit`, `node --check`, `bash -n`, `test -f`, `grep`)
   - `noop`: trivial commands or masked/non-executing invocations (`true`, `:`, `echo`, `pytest || true`, `pytest --collect-only`)
2. Reporting (`classify_verification`, `has_real_acceptance_pass`):
   Distinguishes `failed`, `acceptance_tests` (explicit behavioral/custom check passed),
   `related_tests` (plugin-discovered tests passed >0 test cases), `smoke_only`
   (parse/compile/import/static checks only), and `unverified` (no checks or `noop` only).
3. Applying, timeout promotion & hooks (`satisfies_auto_apply_verification`,
   `satisfies_timeout_promotion`, `is_verified_applied_receipt`,
   `satisfies_post_tool_auto_stop`):
   Require at least one passing `behavioral` or `custom` test check; `syntax_only`
   and `noop` checks keep `diff_preview` visible for Lead review and never trigger
   auto-compaction, timeout promotion, or Lead read suppression.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re
import shlex
from typing import Any, Collection, Mapping

_NO_TESTS_OUTPUT_RE = re.compile(
    r"\bno tests ran\b|\bcollected 0 items\b|\bRan 0 tests\b",
    re.IGNORECASE,
)

_STRENGTH_RANK: dict[str, int] = {
    "noop": 0,
    "syntax_only": 1,
    "custom": 2,
    "behavioral": 3,
}

_NON_EXEC_TEST_FLAG_RE = re.compile(
    r"(?:^|\s)(?:--collect-only|--co|--setup-only|--no-run|--dry-run|--listTests|--fixtures|--markers|--version|-V|--help|-h|-run(?:=|\s+)['\"]?\^\$['\"]?)(?:\s|$)"
    r"|\bgo\s+test\b[^\n;|&]*\s(?:-c|-list)(?:\s|$)"
)

_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

_NOOP_CMD_RE = re.compile(
    r"^(?:"
    r":(?:\s|$)"
    r"|!\s*(?:\S+/)?false(?:\s|$)"
    r"|exit(?:\s+0)?\s*$"
    r"|wait\s*$"
    r"|(?:test|\[)\s+[^\s\]]+\s*(?:\])?\s*$"
    r"|(?:\S+/)?(?:"
    r"true|false"
    r"|echo|printf"
    r"|sleep|pwd|whoami|id|uname|date|hostname"
    r"|ls|dir|cat|head|tail|tee|wc|stat|file|which|command|type"
    r"|cd|pushd|popd|export|unset|set|umask|ulimit|alias"
    r"|mkdir|rmdir|rm|cp|mv|chmod|chown|touch|ln"
    r")\b"
    r")",
    re.IGNORECASE,
)

_SYNTAX_ONLY_CMD_RE = re.compile(
    r"^(?:(?:\S+/)?(?:"
    r"python[0-9.]*\s+-m\s+(?:py_compile|compileall|ast|json\.tool|tabnanny|mypy|ruff|flake8|black|isort|pylint|pyflakes|pyright|bandit)\b"
    r"|ruff\b|flake8\b|black\b|isort\b|mypy\b|pyright\b|pyflakes\b|pylint\b|bandit\b"
    r"|(?:(?:npx|pnpm(?:\s+dlx)?|yarn(?:\s+dlx)?|bunx)\s+)?(?:tsc|eslint|prettier|biome|stylelint|svelte-check|vue-tsc)\b"
    r"|(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:lint|build|typecheck|type-check|tsc|format|fmt|prettier|eslint|compile|check-types|syntax|stylelint)(?:[:_-][a-z0-9_:-]+)?\b"
    r"|make\s+(?:(?:-[CfIOWo]\s+\S+|--[a-z-]+=\S+|-[A-Za-z0-9-]+)\s+)*(?:lint|build|typecheck|type-check|format|fmt|compile|syntax|style)(?:\s|$)"
    r"|node\s+(?:--check|-c)\b"
    r"|(?:ba|z|da|k)?sh\s+-n\b"
    r"|php\s+-l\b|ruby\s+-c\b|perl\s+-c\b|luac\s+-p\b"
    r"|gofmt\b|go\s+(?:vet|build|fmt)\b"
    r"|cargo\s+(?:check|clippy|fmt)\b"
    r"|cargo\s+build\b"
    r"|rustfmt\b"
    r"|rustc\b[^\n;|&]*--emit=\s*(?:metadata|dep-info)\b"
    r"|(?:gcc|g\+\+|cc|c\+\+|clang|clang\+\+)\b[^\n;|&]*-fsyntax-only\b"
    r"|jq\s+empty\b|yamllint\b|shellcheck\b|hadolint\b"
    r"|(?:test|\[|\[\[)\b"
    r"|(?:grep|egrep|fgrep|rg)\b"
    r"))",
    re.IGNORECASE,
)

_BEHAVIORAL_CMD_RE = re.compile(
    r"^(?:(?:\S+/)?(?:"
    r"pytest\b|py\.test\b|python[0-9.]*\s+-m\s+(?:pytest|unittest)\b"
    r"|python[0-9.]*\s+(?:\S+/)?(?:test_\S+|\S+_test)\.py\b"
    r"|tox\b|nox\b|go\s+test\b|cargo\s+(?:test|nextest)\b"
    r"|(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test(?::[a-z0-9_:-]+)?\b"
    r"|(?:(?:npx|pnpm|yarn|bunx)\s+)?(?:vitest|jest|mocha|ava|playwright|cypress)\b"
    r"|rspec\b|rake\s+test\b"
    r"|mvn\s+test\b|gradle(?:w)?\s+(?:test|check)\b|ctest\b|dotnet\s+test\b|bats\b|phpunit\b"
    r"))",
    re.IGNORECASE,
)


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split shell ``text`` on ``sep`` (``';'``, ``'&&'``, ``'||'``, or ``'|'``) outside quotes/parens."""
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                i += 1
                buf.append(text[i])
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(ch)
            buf.append(text[i + 1])
            i += 2
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")" and depth > 0:
            depth -= 1
            buf.append(ch)
            i += 1
            continue
        if depth == 0:
            if sep == ";" and ch in (";", "\n"):
                parts.append("".join(buf))
                buf = []
                i += 1
                continue
            if sep in ("&&", "||") and text.startswith(sep, i):
                parts.append("".join(buf))
                buf = []
                i += len(sep)
                continue
            if sep == "|" and ch == "|" and not text.startswith("||", i) and (i == 0 or text[i - 1] != "|"):
                parts.append("".join(buf))
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _strip_outer_parens(text: str) -> tuple[str, bool]:
    """Strip one or more balanced outer ``(...)`` pairs and report whether any were stripped."""
    s = text.strip()
    stripped_any = False
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        quote: str | None = None
        balanced = True
        for idx, ch in enumerate(s):
            if quote is not None:
                if ch == quote and (idx == 0 or s[idx - 1] != "\\"):
                    quote = None
                continue
            if ch in ("'", '"'):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx < len(s) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        s = s[1:-1].strip()
        stripped_any = True
    return s, stripped_any


def _unwrap_command_prefix(segment: str) -> str:
    """Strip outer parens, env assignments, and common wrapper prefixes (`env`, `uv run`, `timeout`)."""
    s, _ = _strip_outer_parens(segment)
    try:
        tokens = shlex.split(s, comments=True)
    except ValueError:
        tokens = s.split()
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        tok_base = tok.rsplit("/", 1)[-1]
        if _ENV_ASSIGN_RE.match(tok):
            idx += 1
            continue
        if tok_base == "env":
            idx += 1
            while idx < len(tokens) and (tokens[idx].startswith("-") or _ENV_ASSIGN_RE.match(tokens[idx])):
                idx += 1
            continue
        if tok_base in ("time", "nohup", "setsid"):
            idx += 1
            continue
        if tok_base == "exec":
            idx += 1
            while idx < len(tokens) and tokens[idx].startswith("-"):
                idx += 2 if tokens[idx] == "-a" else 1
            continue
        if tok_base == "nice":
            idx += 1
            while idx < len(tokens) and tokens[idx].startswith("-"):
                idx += 2 if tokens[idx] == "-n" else 1
            continue
        if tok_base == "xargs":
            idx += 1
            while idx < len(tokens) and tokens[idx].startswith("-"):
                idx += 2 if tokens[idx] in ("-n", "-L", "-P", "-I", "-s", "-d", "-a", "-E") else 1
            continue
        if tok_base == "timeout" and idx + 2 <= len(tokens):
            idx += 1
            while idx < len(tokens) and tokens[idx].startswith("-"):
                idx += 2 if tokens[idx] in ("-k", "-s") else 1
            if idx < len(tokens):
                idx += 1  # duration
            continue
        if tok in ("uv", "poetry", "pipenv") and idx + 1 < len(tokens) and tokens[idx + 1] == "run":
            idx += 2
            while idx < len(tokens) and tokens[idx].startswith("-"):
                idx += 1
            continue
        if tok == "bundle" and idx + 1 < len(tokens) and tokens[idx + 1] == "exec":
            idx += 2
            continue
        break
    if idx <= 0:
        return s
    if idx >= len(tokens):
        return ""
    return shlex.join(tokens[idx:])


_INLINE_NOOP_CALLS = frozenset({"print", "time.sleep", "sleep"})
_INLINE_SYNTAX_CALLS = frozenset(
    {"compile", "ast.parse", "py_compile.compile", "json.loads", "json.load", "tomllib.loads", "tomllib.load"}
)
_INLINE_READ_HELPERS = frozenset(
    {
        "open",
        "read",
        "read_text",
        "read_bytes",
        "readlines",
        "splitlines",
        "strip",
        "lstrip",
        "rstrip",
        "Path",
        "pathlib.Path",
        "len",
        "str",
        "repr",
        "bool",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "getattr",
        "hasattr",
        "isinstance",
    }
)


def _classify_python_inline(code: str) -> str:
    """Classify ``python -c <code>`` by inspecting its AST."""
    stripped = code.strip()
    if not stripped:
        return "noop"
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return "custom"
    has_import = False
    has_syntax_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            if isinstance(node.test, ast.Constant) and bool(node.test.value) is True:
                continue
            return "custom"
        if isinstance(node, ast.Raise):
            return "custom"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            has_import = True
        elif isinstance(node, ast.Call):
            func = node.func
            call_name = ""
            if isinstance(func, ast.Name):
                call_name = func.id
            elif isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    call_name = f"{func.value.id}.{func.attr}"
                else:
                    call_name = func.attr
            if call_name in _INLINE_NOOP_CALLS or call_name in _INLINE_READ_HELPERS:
                continue
            if call_name in ("sys.exit", "exit", "quit"):
                if not node.args or (
                    len(node.args) == 1
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in (0, None)
                ):
                    continue
                return "custom"
            if call_name in _INLINE_SYNTAX_CALLS:
                has_syntax_call = True
                continue
            return "custom"
    if has_import or has_syntax_call:
        return "syntax_only"
    return "noop"


def _classify_atomic_segment(segment: str) -> str:
    s = segment.strip()
    if not s or s.startswith("#"):
        return "noop"
    # Subshell with inner compound operators: evaluate recursively.
    inner, was_paren = _strip_outer_parens(s)
    if was_paren:
        return command_strength(inner)
    unwrapped = _unwrap_command_prefix(s)
    if not unwrapped or unwrapped.startswith("#"):
        return "noop"
    try:
        toks = shlex.split(unwrapped, comments=True)
    except ValueError:
        toks = []
    canon = " ".join(toks) if toks else unwrapped
    if _NON_EXEC_TEST_FLAG_RE.search(unwrapped) or _NON_EXEC_TEST_FLAG_RE.search(canon):
        return "noop"
    if len(toks) >= 3:
        exe_base = toks[0].rsplit("/", 1)[-1].lower()
        if re.match(r"^python[0-9.]*$", exe_base) and toks[1] == "-c":
            return _classify_python_inline(toks[2])
        if exe_base in ("sh", "bash", "zsh", "dash", "ksh") and re.match(r"^-[A-Za-z]*c$", toks[1]):
            prefix = "set -e; " if "e" in toks[1][1:-1] else ""
            return command_strength(prefix + toks[2])
    for candidate in (unwrapped, canon):
        if _NOOP_CMD_RE.match(candidate):
            return "noop"
    for candidate in (unwrapped, canon):
        if _SYNTAX_ONLY_CMD_RE.match(candidate):
            return "syntax_only"
    for candidate in (unwrapped, canon):
        if _BEHAVIORAL_CMD_RE.match(candidate):
            return "behavioral"
    return "custom"


def _classify_or_expression(expr: str, *, has_pipefail: bool) -> str:
    """Classify ``A || B || ...``: passing guarantees only ``min(strength(branch))`` across branches."""
    or_parts = [p.strip() for p in _split_top_level(expr, "||") if p.strip()]
    if not or_parts:
        return "noop"
    branch_strengths: list[str] = []
    for branch in or_parts:
        branch_best = "noop"
        for and_seg in _split_top_level(branch, "&&"):
            and_clean = and_seg.strip()
            if not and_clean or and_clean.startswith("#"):
                continue
            pipe_parts = [p.strip() for p in _split_top_level(and_clean, "|") if p.strip()]
            if not pipe_parts:
                continue
            relevant_pipes = pipe_parts if has_pipefail else [pipe_parts[-1]]
            for p_seg in relevant_pipes:
                seg_strength = _classify_atomic_segment(p_seg)
                if _STRENGTH_RANK[seg_strength] > _STRENGTH_RANK[branch_best]:
                    branch_best = seg_strength
        branch_strengths.append(branch_best)
    return min(branch_strengths, key=lambda k: _STRENGTH_RANK[k])


def _split_top_level_statements(text: str) -> list[tuple[str, bool]]:
    """Split shell ``text`` into ``(statement, is_backgrounded)`` pairs on ``;``, ``\\n``, and ``&``."""
    out: list[tuple[str, bool]] = []
    buf: list[str] = []
    quote: str | None = None
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                i += 1
                buf.append(text[i])
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(ch)
            buf.append(text[i + 1])
            i += 2
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")" and depth > 0:
            depth -= 1
            buf.append(ch)
            i += 1
            continue
        if depth == 0:
            if ch in (";", "\n"):
                out.append(("".join(buf), False))
                buf = []
                i += 1
                continue
            if ch == "&":
                prev_ch = text[i - 1] if i > 0 else ""
                next_ch = text[i + 1] if i + 1 < n else ""
                if prev_ch not in ("&", ">", "<") and next_ch not in ("&", ">"):
                    out.append(("".join(buf), True))
                    buf = []
                    i += 1
                    continue
        buf.append(ch)
        i += 1
    tail = "".join(buf)
    if tail.strip():
        out.append((tail, False))
    return out


def command_strength(cmd: str) -> str:
    """Return ``'behavioral' | 'custom' | 'syntax_only' | 'noop'`` for shell command ``cmd``."""
    text = (cmd or "").strip()
    if not text or text.startswith("#"):
        return "noop"
    # POSIX `if COND; then BODY; fi` without an `else` branch exits 0 when COND fails.
    if re.match(r"^if\s", text) and re.search(r"\bfi\s*$", text) and not re.search(r"\belse\b", text):
        return "noop"
    has_pipefail = bool(re.search(r"\bset\s+-o\s+pipefail\b", text))
    has_errexit = bool(re.search(r"\bset\s+-[a-z]*e(?:\s|$|;)", text))
    stmts = [
        (s.strip(), bg)
        for s, bg in _split_top_level_statements(text)
        if s.strip() and not s.strip().startswith("#")
    ]
    if not stmts:
        return "noop"
    # Backgrounded statements (`cmd &`) always yield immediate shell status 0, even under `set -e`.
    stmt_strengths = [
        "noop" if bg else _classify_or_expression(stmt, has_pipefail=has_pipefail)
        for stmt, bg in stmts
    ]
    # Without `set -e`, only the final statement governs exit status; if that final statement is
    # a `noop` (`true`, `/bin/true`, `exit 0`, `:`, `wait`, or `cmd &`), it masks earlier failures.
    if not has_errexit and len(stmt_strengths) > 1 and stmt_strengths[-1] == "noop":
        return "noop"
    candidates = stmt_strengths if has_errexit else [stmt_strengths[-1]]
    return max(candidates, key=lambda k: _STRENGTH_RANK[k])


def is_behavioral_check(cmd: str) -> bool:
    """Return True if ``cmd`` executes a known behavioral test runner."""
    return command_strength(cmd) == "behavioral"


def has_behavioral_check(cmds: Collection[str]) -> bool:
    return any(is_behavioral_check(c) for c in cmds)


def classify_commands_for_batch(cmds: Collection[str]) -> str:
    """Classify a set of batch task/acceptance commands into ``'behavioral' | 'syntax_only' | 'unverified'``."""
    best = "noop"
    for c in cmds:
        s = command_strength(c)
        if _STRENGTH_RANK[s] > _STRENGTH_RANK[best]:
            best = s
    if best in ("behavioral", "custom"):
        return "behavioral"
    if best == "syntax_only":
        return "syntax_only"
    return "unverified"


def _passed_test_count(entry: Mapping[str, Any]) -> int:
    counts = entry.get("counts")
    if not isinstance(counts, Mapping):
        return 0
    try:
        return int(counts.get("passed") or 0)
    except (TypeError, ValueError):
        return 0


def _entry_strength(entry: Mapping[str, Any]) -> str:
    """Resolve verification strength for a result dict, classifying legacy receipts on read when unstamped."""
    if entry.get("compile_smoke") or entry.get("auto_import_smoke"):
        return "syntax_only"
    stamped = entry.get("strength")
    if stamped in _STRENGTH_RANK:
        return str(stamped)
    cmd = str(entry.get("command") or "")
    if not cmd.strip():
        # Synthetic unit-test fixtures that omit `command` default to behavioral unless marked smoke/noop.
        return "behavioral" if ("command" not in entry) else "noop"
    return command_strength(cmd)


def _is_non_skipped_passing_entry(entry: Any) -> bool:
    return (
        isinstance(entry, Mapping)
        and entry.get("passed") is True
        and not entry.get("timed_out")
        and not entry.get("vacuous")
        and not entry.get("no_tests_collected")
        and not entry.get("runner_incompatible_skipped")
    )


def is_non_smoke_passing_entry(entry: Any) -> bool:
    """True when ``entry`` passed cleanly and represents a ``behavioral`` or ``custom`` test check."""
    if not _is_non_skipped_passing_entry(entry):
        return False
    return _entry_strength(entry) in ("behavioral", "custom")


def _is_explicit_test_entry(entry: Mapping[str, Any]) -> bool:
    return (
        is_non_smoke_passing_entry(entry)
        and not entry.get("auto_discovered_default")
        and not entry.get("auto_discovered_regression")
    )


def _is_behavioral_test_entry(entry: Mapping[str, Any]) -> bool:
    """True for an explicit behavioral/custom pass or a discovered test suite that ran >=1 passing test."""
    if not is_non_smoke_passing_entry(entry) or entry.get("auto_discovered_regression"):
        return False
    if entry.get("auto_discovered_default"):
        return _passed_test_count(entry) > 0
    return True


@dataclass(frozen=True)
class VerificationClassification:
    tier: str
    explicit_acceptance_passed: bool
    behavioral_tests_passed: bool


def evaluate_verification(
    acceptance_results: Any,
    acceptance_passed: bool | None,
) -> VerificationClassification:
    """Compute the canonical verification tier and policy flags for a result list."""
    results = [r for r in acceptance_results if isinstance(r, Mapping)] if isinstance(acceptance_results, list) else []
    explicit_ok = any(_is_explicit_test_entry(r) for r in results)
    behavioral_ok = any(_is_behavioral_test_entry(r) for r in results)

    if acceptance_passed is False:
        return VerificationClassification(
            tier="failed",
            explicit_acceptance_passed=False,
            behavioral_tests_passed=False,
        )
    if explicit_ok:
        return VerificationClassification(
            tier="acceptance_tests",
            explicit_acceptance_passed=True,
            behavioral_tests_passed=True,
        )
    ran = [r for r in results if _is_non_skipped_passing_entry(r)]
    # Only promote discovered suites to `related_tests` when their command strength is behavioral/custom
    # (never `echo "5 passed"` or syntax-only linters).
    if any(
        _entry_strength(r) in ("behavioral", "custom")
        and _passed_test_count(r) > 0
        for r in ran
    ):
        return VerificationClassification(
            tier="related_tests",
            explicit_acceptance_passed=False,
            behavioral_tests_passed=behavioral_ok,
        )
    if any(_entry_strength(r) in ("syntax_only", "behavioral", "custom") for r in ran):
        return VerificationClassification(
            tier="smoke_only",
            explicit_acceptance_passed=False,
            behavioral_tests_passed=False,
        )
    return VerificationClassification(
        tier="unverified",
        explicit_acceptance_passed=False,
        behavioral_tests_passed=False,
    )


def has_real_acceptance_pass(
    acceptance_results: Any,
    *,
    reject_zero_test_output: bool = False,
) -> bool:
    """True when at least one explicit behavioral/custom acceptance command passed."""
    if not isinstance(acceptance_results, list) or not acceptance_results:
        return False
    for entry in acceptance_results:
        if not isinstance(entry, Mapping) or not _is_explicit_test_entry(entry):
            continue
        if reject_zero_test_output and _NO_TESTS_OUTPUT_RE.search(str(entry.get("output_tail") or "")):
            continue
        return True
    return False


def classify_verification(acceptance_results: Any, acceptance_passed: bool | None) -> str:
    """Summarize verification strength: `failed`, `acceptance_tests`, `related_tests`, `smoke_only`, or `unverified`."""
    return evaluate_verification(acceptance_results, acceptance_passed).tier


def satisfies_auto_apply_verification(
    acceptance_results: Any,
    acceptance_passed: bool | None,
) -> bool:
    """Policy for `_maybe_auto_apply_digest`: explicit behavioral/custom pass or discovered test suite (`passed > 0`), never syntax/noop."""
    if acceptance_passed is not True:
        return False
    return evaluate_verification(acceptance_results, acceptance_passed).behavioral_tests_passed


def satisfies_timeout_promotion(
    acceptance_results: Any,
    *,
    test_only_diff: bool = False,
) -> bool:
    """Policy for promoting a timed-out worker with clean verified edits to `status='ok'`."""
    if test_only_diff or not isinstance(acceptance_results, list) or not acceptance_results:
        return False
    if not all(isinstance(r, Mapping) and r.get("passed") is True for r in acceptance_results):
        return False
    return has_real_acceptance_pass(acceptance_results, reject_zero_test_output=True)


def is_verified_applied_receipt(rec: Mapping[str, Any]) -> bool:
    """Hook policy (`PreToolUse`): True when an applied `ok` receipt passed real behavioral tests with no warnings."""
    if (
        rec.get("status") != "ok"
        or rec.get("applied") is not True
        or rec.get("reverted")
        or rec.get("acceptance_passed") is not True
        or rec.get("vacuous_acceptance")
        or rec.get("test_only_diff")
        or rec.get("scope_violations")
        or rec.get("missing_prompt_symbols")
        or rec.get("contract_warnings")
        or rec.get("stub_mutation_warnings")
    ):
        return False
    return satisfies_auto_apply_verification(rec.get("acceptance_results"), True)


def satisfies_post_tool_auto_stop(rec: Mapping[str, Any]) -> bool:
    """Hook policy (`PostToolUse`): True when `acceptance_passed` is True via an explicit behavioral/custom command."""
    if rec.get("acceptance_passed") is not True:
        return False
    return has_real_acceptance_pass(rec.get("acceptance_results"))
