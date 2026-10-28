#!/usr/bin/env python3
"""PreToolUse hook guard and prompt dictation linter for cc-agy-plugin."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


from server.regions import is_strict_mode_enabled  # noqa: E402  (shared strict-mode check, re-exported)
from server.regions import resolve_cost_profile  # noqa: E402


def resolve_project_root(
    workdir: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    """Resolve the true project root, honoring CLAUDE_PROJECT_DIR and git rev-parse --show-toplevel."""
    source = env if env is not None else os.environ
    env_proj = (source.get("CLAUDE_PROJECT_DIR") or "").strip()
    if env_proj and not env_proj.startswith("${"):
        base = Path(env_proj).resolve()
    elif workdir is not None:
        base = Path(workdir).resolve()
    else:
        base = Path.cwd().resolve()

    if base.exists():
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(base),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return Path(res.stdout.strip()).resolve()

    # If no explicit CLAUDE_PROJECT_DIR and no .git repo, and base is a subdirectory
    # inside a top-level /tmp/<workspace-root> folder (e.g. /tmp/ws-xxxx/src), anchor
    # to /tmp/<workspace-root> so `cd src` does not lose the enclosing workspace root.
    if not (env_proj and not env_proj.startswith("${")):
        for tmp_root in (Path("/tmp").resolve(), Path("/var/tmp").resolve()):
            if base.is_relative_to(tmp_root):
                rel_parts = base.relative_to(tmp_root).parts
                if len(rel_parts) >= 2:
                    return tmp_root / rel_parts[0]
    return base


def classify_path(
    target_path: str | Path,
    workdir: str | Path,
    out_dir: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Classify a filesystem path relative to out_dir, workdir, and /tmp.

    CRITICAL ORDERING INVARIANT:
    Always check `out_dir` and `workdir` BEFORE `/tmp` (or `/var/tmp`) so that
    git repositories or worktrees located under `/tmp/...` on Linux/macOS are
    classified as `"workspace"` (and protected against direct driver mutation),
    never misclassified as `"tmp_scratch"`.
    """
    ws_resolved = resolve_project_root(workdir, env=env)
    base_cwd = Path(cwd).resolve() if cwd is not None else Path(workdir).resolve()
    raw_str = str(target_path).strip()
    candidate = Path(raw_str)
    if not candidate.is_absolute():
        resolved = (base_cwd / candidate).resolve()
    else:
        resolved = candidate.resolve()

    out_candidates: list[str] = []
    if out_dir is not None:
        out_candidates.append(str(out_dir).strip())
    source_env = env if env is not None else os.environ
    for env_key in ("CC_AGY_OUT_DIR", "CC_AGY_DATA_DIR", "CC_AGY_WORKTREES_DIR", "CLAUDE_PLUGIN_DATA"):
        val = source_env.get(env_key, "").strip()
        if val:
            out_candidates.append(val)

    for out_str in out_candidates:
        if out_str and not out_str.startswith("${"):
            out_resolved = Path(out_str).resolve()
            if resolved.is_relative_to(out_resolved):
                return "out_dir"

    for part in resolved.parts:
        if part.startswith(("cc-agy-worktrees-", ".cc-agy-worktrees-")):
            return "out_dir"

    for data_candidate in (
        ws_resolved / ".cc-agy-data",
        Path.home() / ".cache" / "cc-agy-plugin",
    ):
        if resolved.is_relative_to(data_candidate.resolve()):
            return "out_dir"

    if resolved.is_relative_to(ws_resolved):
        return "workspace"

    if not candidate.is_absolute() and ".." in candidate.parts:
        return "workspace"

    for tmp_root in (Path("/tmp").resolve(), Path("/var/tmp").resolve()):
        if resolved.is_relative_to(tmp_root):
            return "tmp_scratch"

    return "external"


_CODE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"def\s+\w+\s*\(|"
    r"async\s+def\s+\w+\s*\(|"
    r"class\s+\w+[:\(]|"
    r"import\s+[\w\.]+|"
    r"from\s+[\w\.]+\s+import\s+|"
    r"return\b|"
    r"yield\b|"
    r"raise\b|"
    r"assert\b|"
    r"(?:pass|break|continue)\s*(?:#.*)?$|"
    r"if\s+.+:\s*$|"
    r"elif\s+.+:\s*$|"
    r"else:\s*$|"
    r"for\s+.+\s+in\s+.+:\s*$|"
    r"while\s+.+:\s*$|"
    r"try:\s*$|"
    r"except\b.*:\s*$|"
    r"finally:\s*$|"
    r"with\s+.+:\s*$|"
    r"@\w+[\w\.\(\)]*\s*$|"
    r"(?:const|let|var|function|func|fn|struct|impl|interface|type)\s+\w+|"
    r"export\s+(?:default\s+)?(?:function|class|const|interface|type)\s+|"
    r"public\s+|private\s+|protected\s+|"
    r"[\w\.\[\]]+\s*(?:[+\-*/%&|^]?=)\s*[^=].*|"
    r"(?:await\s+)?[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(.*\)\s*(?:#.*)?$|"
    r"[{}();]\s*$"
    r")"
)

_COMMENT_LINE_RE = re.compile(r"^\s*(?:#|//|/\*|\*)")
_PROSE_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+\S|\d+\.\s+\S)")
_BLOCKQUOTE_PREFIX_RE = re.compile(r"^\s*(?:>\s*)+")
_LIST_ITEM_PREFIX_RE = re.compile(r"^(\s*)(?:[-*+]|\d+\.)\s+(.*)$")


def _unwrap_markdown_code_line(line: str) -> tuple[str, bool]:
    """Strip markdown blockquote (`> `), list-item (`- `), and inline backtick wrappers for code linting."""
    s = line
    had_blockquote = False
    for _ in range(3):
        prev = s
        if _BLOCKQUOTE_PREFIX_RE.match(s):
            s = _BLOCKQUOTE_PREFIX_RE.sub("", s, count=1)
            had_blockquote = True
        m_list = _LIST_ITEM_PREFIX_RE.match(s)
        if m_list:
            s = m_list.group(2)
        stripped = s.strip()
        if (
            len(stripped) >= 2
            and stripped.startswith("`")
            and stripped.endswith("`")
            and not stripped.startswith("```")
        ):
            s = stripped[1:-1]
        if s == prev:
            break
    return s, had_blockquote


_INLINE_HEADER_CODE_RE = re.compile(
    r"^\s*(?:for\s+.+\s+in\s+.+:|while\s+.+:|if\s+.+:|elif\s+.+:|else:|try:|except\b.*:|finally:|with\s+.+:)\s*\S+"
)
_LEADING_TARGET_LABEL_RE = re.compile(r"^\s*(?:in\s+)?[`'\"]?[\w\.\[\]\(\)/-]+[`'\"]?\s*:\s+", re.I)


_PY_STR_PREFIXES = frozenset({"r", "b", "u", "f", "rb", "br", "fr", "rf"})


def _has_closing_quote_on_line(text: str, start_idx: int, q: str) -> bool:
    esc = False
    for ch in text[start_idx:]:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == q:
            return True
    return False


def _is_token_start_quote(text: str, idx: int, ch: str) -> bool:
    if not _has_closing_quote_on_line(text, idx + 1, ch):
        return False
    if idx == 0:
        return True
    prev = text[idx - 1]
    if not (prev.isalnum() or prev == "_"):
        return True
    if ch in ("'", '"'):
        tok_start = idx - 1
        while tok_start > 0 and (text[tok_start - 1].isalnum() or text[tok_start - 1] == "_"):
            tok_start -= 1
        if text[tok_start:idx].lower() in _PY_STR_PREFIXES:
            return True
    return False


def _split_semicolon_outside_quotes(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    for idx, ch in enumerate(text):
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"') and _is_token_start_quote(text, idx, ch):
            quote = ch
            buf.append(ch)
            continue
        if ch == ";":
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _count_line_code_statements(line: str, unwrapped: str, is_indented_code: bool) -> int:
    """Return how many consecutive code statements `line` contributes (`0` for prose, `>=2` for `;`-chained code)."""
    if ";" in unwrapped:
        cand = _LEADING_TARGET_LABEL_RE.sub("", unwrapped, count=1).strip()
        if cand.endswith("."):
            cand = cand[:-1].rstrip()
        cand = cand.strip("`").strip()
        raw_parts = _split_semicolon_outside_quotes(cand)
        if len(raw_parts) >= 2:
            expanded_stmts: list[str] = []
            for idx_p, part in enumerate(raw_parts):
                p_clean = part[:-1].rstrip() if (idx_p == len(raw_parts) - 1 and part.endswith(".")) else part
                p_clean = p_clean.strip("`").strip()
                m_hdr = re.match(
                    r"^((?:for\s+.+\s+in\s+.+|while\s+.+|if\s+.+|elif\s+.+|else|try|except\b.*|finally|with\s+.+):)\s+(.+)$",
                    p_clean,
                )
                if m_hdr:
                    expanded_stmts.append(m_hdr.group(1))
                    expanded_stmts.append(m_hdr.group(2))
                else:
                    expanded_stmts.append(p_clean)
            matched = sum(
                1
                for st in expanded_stmts
                if _CODE_LINE_RE.match(st) or _INLINE_HEADER_CODE_RE.match(st)
            )
            if matched >= 2 and matched >= (len(expanded_stmts) + 1) // 2:
                return len(expanded_stmts)
    if _CODE_LINE_RE.match(line) or _CODE_LINE_RE.match(unwrapped) or is_indented_code:
        return 1
    return 0


_DATA_FENCE_LANGS = frozenset(
    {
        "json",
        "jsonc",
        "json5",
        "yaml",
        "yml",
        "toml",
        "xml",
        "html",
        "csv",
        "tsv",
        "ini",
        "env",
        "dotenv",
        "graphql",
        "gql",
        "proto",
        "protobuf",
    }
)
_SQL_FENCE_LANGS = frozenset({"sql", "ddl", "postgres", "postgresql", "mysql", "sqlite"})

_IMPERATIVE_CODE_GUARD_RE = re.compile(
    r"^\s*(?:"
    r"def\s+\w+\s*\(|"
    r"async\s+def\s+\w+\s*\(|"
    r"class\s+\w+[:\(]|"
    r"import\s+[\w\.]+|"
    r"from\s+[\w\.]+\s+import\s+|"
    r"return\b|yield\b|raise\b|assert\b|"
    r"if\s+.+:\s*$|elif\s+.+:\s*$|else:\s*$|"
    r"for\s+.+\s+in\s+.+:\s*$|while\s+.+:\s*$|"
    r"try:\s*$|except\b.*:\s*$|finally:\s*$|with\s+.+:\s*$|"
    r"(?:const|let|var|function|func|fn|impl)\s+\w+|"
    r"export\s+(?:default\s+)?(?:function|class|const)\s+"
    r")"
)
_JSON_STRUCT_OR_KV_RE = re.compile(
    r'^\s*(?:[\{\}\[\]],?|"(?:\\.|[^"\\])*"\s*:.*|"(?:\\.|[^"\\])*"\s*,?|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*,?|(?:true|false|null)\s*,?|//.*|\.\.\.)\s*$'
)
_YAML_KV_OR_ITEM_RE = re.compile(
    r"""^\s*(?:---\s*$|\.\.\.\s*$|#.*|-\s+\S.*|-\s*$|[A-Za-z0-9_."'/+-]+\s*:(?:\s+.*|\s*$))$"""
)
_SQL_DDL_START_RE = re.compile(
    r"\b(?:CREATE\s+(?:TEMP(?:ORARY)?\s+)?(?:UNIQUE\s+)?(?:TABLE|INDEX|VIEW|SCHEMA|TYPE|DOMAIN)\b|ALTER\s+TABLE\b)",
    re.I,
)
_SQL_PROCEDURAL_RE = re.compile(
    r"\b(?:CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE|TRIGGER)\b|LANGUAGE\s+plpgsql\b|DECLARE\b|\bBEGIN\b|\bLOOP\b)",
    re.I,
)
_PY_TB_HEADER_RE = re.compile(
    r"^\s*(?:Traceback \(most recent call last\):|During handling of the above exception.*:|The above exception was the direct cause.*:)\s*$"
)
_PY_TB_FRAME_RE = re.compile(r'^\s*File "[^"]+", line \d+(?:, in \S+)?\s*$')
_PY_TB_CARET_RE = re.compile(r"^\s*[\^~]+\s*$")
_PY_TB_EXC_RE = re.compile(
    r"^\s*[A-Za-z_][\w\.]*(?:Error|Exception|Warning|Exit|Interrupt|Fault|Failure)(?::.*|\s*)$"
)
_JS_GO_TB_FRAME_RE = re.compile(
    r"^\s*(?:at\s+\S+.*\(.*:\d+:\d+\)|at\s+[^\s()]+:\d+:\d+|goroutine\s+\d+\s+\[.*?\]:|\S+\.go:\d+\b)"
)


def _is_traceback_lines(raw_lines: list[str]) -> bool:
    """Return True if `raw_lines` form a Python/JS/Go stack trace rather than dictated source code."""
    non_empty = [
        _BLOCKQUOTE_PREFIX_RE.sub("", ln.strip(), count=1).strip()
        for ln in raw_lines
        if _BLOCKQUOTE_PREFIX_RE.sub("", ln.strip(), count=1).strip()
    ]
    if not non_empty:
        return False
    frame_hits = sum(
        1 for ln in non_empty if _PY_TB_FRAME_RE.match(ln) or _JS_GO_TB_FRAME_RE.match(ln)
    )
    if frame_hits == 0:
        return False
    echo_budget = 0
    unaccounted = 0
    for ln in non_empty:
        if _PY_TB_HEADER_RE.match(ln):
            echo_budget = 0
            continue
        if _PY_TB_FRAME_RE.match(ln) or _JS_GO_TB_FRAME_RE.match(ln):
            echo_budget = 1
            continue
        if _PY_TB_CARET_RE.match(ln) or _PY_TB_EXC_RE.match(ln):
            continue
        if echo_budget > 0:
            echo_budget -= 1
            continue
        unaccounted += 1
    return unaccounted <= 1


def _is_declarative_fence_block(fence_lang: str, raw_lines: list[str]) -> bool:
    """Return True if a fenced block is declarative data/schema or a stack trace (with zero smuggled code)."""
    if _is_traceback_lines(raw_lines):
        return True
    cleaned = [_BLOCKQUOTE_PREFIX_RE.sub("", ln, count=1) for ln in raw_lines]
    non_empty = [ln for ln in cleaned if ln.strip()]
    if not non_empty:
        return True
    if any(_IMPERATIVE_CODE_GUARD_RE.match(ln) for ln in non_empty):
        return False
    joined = "\n".join(cleaned).strip()
    if joined.startswith(("{", "[")):
        try:
            json.loads(joined)
            return True
        except Exception:
            pass
    lang = fence_lang.lower().strip()
    if lang in ("json", "jsonc", "json5"):
        return all(_JSON_STRUCT_OR_KV_RE.match(ln) for ln in non_empty)
    if lang in ("yaml", "yml"):
        return all(_YAML_KV_OR_ITEM_RE.match(ln) for ln in non_empty)
    if lang in _DATA_FENCE_LANGS:
        return True
    if lang in _SQL_FENCE_LANGS:
        return bool(_SQL_DDL_START_RE.search(joined)) and not bool(_SQL_PROCEDURAL_RE.search(joined))
    return False


def _find_unfenced_declarative_line_indices(lines: list[str]) -> set[int]:
    """Identify line indices belonging to valid unfenced JSON fixtures or stack traces."""
    exempt: set[int] = set()
    in_fence = False
    fence_marker = ""
    unfenced_indices: list[int] = []
    for idx, line in enumerate(lines):
        ufc = _BLOCKQUOTE_PREFIX_RE.sub("", line.strip(), count=1).strip()
        if ufc.startswith("```") or ufc.startswith("~~~"):
            mk = ufc[:3]
            if in_fence and mk == fence_marker:
                in_fence = False
                fence_marker = ""
            elif not in_fence:
                in_fence = True
                fence_marker = mk
            continue
        if not in_fence:
            unfenced_indices.append(idx)

    # 1. Unfenced JSON object/array blocks
    pos = 0
    while pos < len(unfenced_indices):
        line_idx = unfenced_indices[pos]
        raw_s = _BLOCKQUOTE_PREFIX_RE.sub("", lines[line_idx], count=1).strip()
        if raw_s.startswith(("{", "[")):
            depth = 0
            in_str = False
            esc = False
            block_idxs: list[int] = []
            valid_scan = True
            for scan_pos in range(pos, len(unfenced_indices)):
                s_idx = unfenced_indices[scan_pos]
                if block_idxs and s_idx != block_idxs[-1] + 1:
                    valid_scan = False
                    break
                s_line = _BLOCKQUOTE_PREFIX_RE.sub("", lines[s_idx], count=1)
                if _IMPERATIVE_CODE_GUARD_RE.match(s_line):
                    valid_scan = False
                    break
                block_idxs.append(s_idx)
                for ch in s_line:
                    if esc:
                        esc = False
                        continue
                    if ch == "\\" and in_str:
                        esc = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if not in_str:
                        if ch in ("{", "["):
                            depth += 1
                        elif ch in ("}", "]"):
                            depth -= 1
                if depth == 0 and block_idxs:
                    break
            if valid_scan and depth == 0 and len(block_idxs) >= 2:
                candidate_txt = "\n".join(_BLOCKQUOTE_PREFIX_RE.sub("", lines[k], count=1) for k in block_idxs)
                non_empty_b = [ln for ln in candidate_txt.splitlines() if ln.strip()]
                parsed_ok = False
                try:
                    json.loads(candidate_txt)
                    parsed_ok = True
                except Exception:
                    parsed_ok = all(_JSON_STRUCT_OR_KV_RE.match(ln) for ln in non_empty_b)
                if parsed_ok:
                    exempt.update(block_idxs)
                    pos += len(block_idxs)
                    continue
        pos += 1

    # 2. Unfenced stack trace lines
    in_tb = False
    echo_budget = 0
    for line_idx in unfenced_indices:
        unwrapped, _ = _unwrap_markdown_code_line(lines[line_idx])
        stripped_u = unwrapped.strip()
        if not stripped_u:
            in_tb = False
            echo_budget = 0
            continue
        if _PY_TB_HEADER_RE.match(stripped_u):
            in_tb = True
            echo_budget = 0
            exempt.add(line_idx)
            continue
        if _PY_TB_FRAME_RE.match(stripped_u) or _JS_GO_TB_FRAME_RE.match(stripped_u):
            in_tb = True
            echo_budget = 1
            exempt.add(line_idx)
            continue
        if in_tb and _PY_TB_CARET_RE.match(stripped_u):
            exempt.add(line_idx)
            continue
        if in_tb and echo_budget > 0:
            echo_budget -= 1
            exempt.add(line_idx)
            continue
        if in_tb and _PY_TB_EXC_RE.match(stripped_u):
            exempt.add(line_idx)
            in_tb = False
            echo_budget = 0
            continue
        in_tb = False
        echo_budget = 0

    return exempt


def lint_delegation_prompt(prompt_text: str, max_consecutive_code_lines: int = 8) -> dict[str, Any]:
    """Detect whether a delegation prompt dictates >= 9 consecutive lines/statements of code.

    Checks fenced code blocks (``` or ~~~), unfenced consecutive code-like or 4-space indented
    code lines, and semicolon-packed multi-statement code lines in plan/checklist items, while
    allowing concise 1-line inline API invariants (e.g. `self.records.clear()`), declarative
    JSON/YAML/SQL-DDL schemas, and runtime stack traces.
    Interleaved comments inside a code run do not reset the consecutive code counter.
    """
    lines = prompt_text.splitlines()
    exempt_unfenced = _find_unfenced_declarative_line_indices(lines)
    in_fence = False
    fence_marker = ""
    fence_lang = ""
    fence_lines: list[str] = []
    fence_run = 0
    max_fence_run = 0
    code_run = 0
    max_code_run = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        unwrapped_fence_check = _BLOCKQUOTE_PREFIX_RE.sub("", stripped, count=1).strip()
        if unwrapped_fence_check.startswith("```") or unwrapped_fence_check.startswith("~~~"):
            marker = unwrapped_fence_check[:3]
            if in_fence and marker == fence_marker:
                if not _is_declarative_fence_block(fence_lang, fence_lines):
                    if fence_run > max_fence_run:
                        max_fence_run = fence_run
                fence_run = 0
                fence_lines = []
                fence_lang = ""
                in_fence = False
                fence_marker = ""
                continue
            elif not in_fence:
                in_fence = True
                fence_marker = marker
                info = unwrapped_fence_check[3:].strip()
                fence_lang = info.split()[0].lower() if info else ""
                fence_lines = []
                fence_run = 0
                continue

        if in_fence:
            fence_lines.append(line)
            if unwrapped_fence_check:
                stmt_n = _count_line_code_statements(line, unwrapped_fence_check, True)
                fence_run += max(1, stmt_n)
            continue

        if idx in exempt_unfenced:
            code_run = 0
            continue

        if stripped == "" or unwrapped_fence_check == "":
            continue

        unwrapped, had_blockquote = _unwrap_markdown_code_line(line)
        if _COMMENT_LINE_RE.match(unwrapped):
            # Comments inside an active code block do not reset code_run
            continue

        is_indented_code = (
            ((line.startswith("    ") or line.startswith("\t")) and not _PROSE_LIST_RE.match(line))
            or (had_blockquote and (unwrapped.startswith("    ") or unwrapped.startswith("\t")))
        )
        stmt_count = _count_line_code_statements(line, unwrapped, is_indented_code)
        if stmt_count > 0:
            code_run += stmt_count
            if code_run > max_code_run:
                max_code_run = code_run
        else:
            code_run = 0

    if in_fence and not _is_declarative_fence_block(fence_lang, fence_lines) and fence_run > max_fence_run:
        max_fence_run = fence_run

    worst_run = max(max_fence_run, max_code_run)
    violated = worst_run > max_consecutive_code_lines
    return {
        "ok": not violated,
        "max_consecutive_code_lines": worst_run,
        "limit": max_consecutive_code_lines,
        "reason": (
            f"Delegation prompt contains {worst_run} consecutive code lines "
            f"(limit is {max_consecutive_code_lines}). Describe desired outcomes and "
            f"invariants instead of dictating implementation code."
            if violated
            else None
        ),
    }


def _collapse_code_lines_run(
    block_lines: list[str],
    *,
    max_consecutive_code_lines: int,
    snippet_dir: Path | None,
    snippet_paths: list[str],
) -> list[str]:
    """Collapse a single run of code lines exceeding `max_consecutive_code_lines` into signature lines + note."""
    keep_head = max(0, min(3, max_consecutive_code_lines - 1))
    full_block = "\n".join(block_lines)
    omitted_count = max(1, len(block_lines) - keep_head)
    if snippet_dir is not None:
        try:
            snippet_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(full_block.encode("utf-8", errors="replace")).hexdigest()[:10]
            target_file = snippet_dir / f"dictated_snippet_{digest}.txt"
            target_file.write_text(full_block + "\n", encoding="utf-8")
            snippet_str = str(target_file)
            if snippet_str not in snippet_paths:
                snippet_paths.append(snippet_str)
        except OSError:
            pass
    note = (
        f"[... {omitted_count} dictated code lines omitted; "
        "implement from behavioral requirements ...]"
    )
    return [*block_lines[:keep_head], note]


def repair_dictated_prompt(
    prompt_text: str,
    *,
    max_consecutive_code_lines: int = 8,
    snippet_dir: Path | None = None,
) -> dict[str, Any]:
    """Repair a delegation prompt that violates the consecutive-code-line dictation limit.

    Extracts over-limit fenced or unfenced code blocks into optional reference files under
    `snippet_dir` and collapses them in-place to their leading signature lines plus an omission
    marker, preventing empty-patch PreToolUse deadlocks while keeping the prompt within linting limits.
    """
    lint = lint_delegation_prompt(prompt_text, max_consecutive_code_lines=max_consecutive_code_lines)
    if lint["ok"]:
        return {
            "repaired": False,
            "prompt": prompt_text,
            "original_max_code_lines": lint["max_consecutive_code_lines"],
            "snippet_paths": [],
        }

    lines = prompt_text.splitlines()
    exempt_unfenced = _find_unfenced_declarative_line_indices(lines)
    out_lines: list[str] = []
    snippet_paths: list[str] = []

    in_fence = False
    fence_marker = ""
    fence_lang = ""
    fence_open_line = ""
    fence_body: list[str] = []
    fence_nonempty = 0

    unfenced_buf: list[str] = []
    unfenced_code_count = 0

    def _flush_unfenced() -> None:
        nonlocal unfenced_buf, unfenced_code_count
        if not unfenced_buf:
            return
        if unfenced_code_count > max_consecutive_code_lines:
            out_lines.extend(
                _collapse_code_lines_run(
                    unfenced_buf,
                    max_consecutive_code_lines=max_consecutive_code_lines,
                    snippet_dir=snippet_dir,
                    snippet_paths=snippet_paths,
                )
            )
        else:
            out_lines.extend(unfenced_buf)
        unfenced_buf = []
        unfenced_code_count = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        unwrapped_fence_check = _BLOCKQUOTE_PREFIX_RE.sub("", stripped, count=1).strip()
        if unwrapped_fence_check.startswith("```") or unwrapped_fence_check.startswith("~~~"):
            marker = unwrapped_fence_check[:3]
            if in_fence and marker == fence_marker:
                if fence_nonempty > max_consecutive_code_lines and not _is_declarative_fence_block(
                    fence_lang, fence_body
                ):
                    collapsed = _collapse_code_lines_run(
                        fence_body,
                        max_consecutive_code_lines=max_consecutive_code_lines,
                        snippet_dir=snippet_dir,
                        snippet_paths=snippet_paths,
                    )
                    out_lines.append(fence_open_line)
                    out_lines.extend(collapsed)
                    out_lines.append(line)
                else:
                    out_lines.append(fence_open_line)
                    out_lines.extend(fence_body)
                    out_lines.append(line)
                in_fence = False
                fence_marker = ""
                fence_lang = ""
                fence_open_line = ""
                fence_body = []
                fence_nonempty = 0
                continue
            elif not in_fence:
                _flush_unfenced()
                in_fence = True
                fence_marker = marker
                info = unwrapped_fence_check[3:].strip()
                fence_lang = info.split()[0].lower() if info else ""
                fence_open_line = line
                fence_body = []
                fence_nonempty = 0
                continue

        if in_fence:
            fence_body.append(line)
            if unwrapped_fence_check:
                fence_nonempty += 1
            continue

        if idx in exempt_unfenced:
            _flush_unfenced()
            out_lines.append(line)
            continue

        if stripped == "" or unwrapped_fence_check == "":
            if unfenced_buf:
                unfenced_buf.append(line)
            else:
                out_lines.append(line)
            continue

        unwrapped, had_blockquote = _unwrap_markdown_code_line(line)
        if _COMMENT_LINE_RE.match(unwrapped):
            if unfenced_buf:
                unfenced_buf.append(line)
            else:
                out_lines.append(line)
            continue

        is_indented_code = (
            ((line.startswith("    ") or line.startswith("\t")) and not _PROSE_LIST_RE.match(line))
            or (had_blockquote and (unwrapped.startswith("    ") or unwrapped.startswith("\t")))
        )
        stmt_count = _count_line_code_statements(line, unwrapped, is_indented_code)
        if stmt_count > 0:
            if stmt_count > 1:
                raw_semi = _LEADING_TARGET_LABEL_RE.sub("", unwrapped, count=1).strip()
                if raw_semi.endswith("."):
                    raw_semi = raw_semi[:-1].rstrip()
                raw_semi = raw_semi.strip("`").strip()
                expanded = [s.strip("`").strip() for s in _split_semicolon_outside_quotes(raw_semi)]
                unfenced_buf.extend(expanded if len(expanded) > 1 else [line])
            else:
                unfenced_buf.append(line)
            unfenced_code_count += stmt_count
        else:
            _flush_unfenced()
            out_lines.append(line)

    if in_fence:
        out_lines.append(fence_open_line)
        if fence_nonempty > max_consecutive_code_lines:
            out_lines.extend(
                _collapse_code_lines_run(
                    fence_body,
                    max_consecutive_code_lines=max_consecutive_code_lines,
                    snippet_dir=snippet_dir,
                    snippet_paths=snippet_paths,
                )
            )
        else:
            out_lines.extend(fence_body)
    _flush_unfenced()

    repaired_prompt = "\n".join(out_lines)
    if not lint_delegation_prompt(repaired_prompt, max_consecutive_code_lines=max_consecutive_code_lines)["ok"]:
        pass2_lines: list[str] = []
        p2_in_fence = False
        p2_marker = ""
        p2_code_run = 0
        keep_limit = max(0, min(3, max_consecutive_code_lines - 1))
        for ln in repaired_prompt.splitlines():
            st = ln.strip()
            ufc = _BLOCKQUOTE_PREFIX_RE.sub("", st, count=1).strip()
            if ufc.startswith("```") or ufc.startswith("~~~"):
                mk = ufc[:3]
                if p2_in_fence and mk == p2_marker:
                    p2_in_fence = False
                    p2_marker = ""
                    pass2_lines.append(ln)
                    continue
                elif not p2_in_fence:
                    p2_in_fence = True
                    p2_marker = mk
                    pass2_lines.append(ln)
                    continue
            if p2_in_fence or st == "" or ufc == "":
                pass2_lines.append(ln)
                continue
            unw, had_bq = _unwrap_markdown_code_line(ln)
            if _COMMENT_LINE_RE.match(unw):
                if p2_code_run <= keep_limit:
                    pass2_lines.append(ln)
                continue
            is_ind = (
                ((ln.startswith("    ") or ln.startswith("\t")) and not _PROSE_LIST_RE.match(ln))
                or (had_bq and (unw.startswith("    ") or unw.startswith("\t")))
            )
            if _CODE_LINE_RE.match(ln) or _CODE_LINE_RE.match(unw) or is_ind:
                p2_code_run += 1
                if p2_code_run <= keep_limit:
                    pass2_lines.append(ln)
                elif p2_code_run == keep_limit + 1:
                    pass2_lines.append(
                        "[... dictated code lines omitted; implement from behavioral requirements ...]"
                    )
            else:
                p2_code_run = 0
                pass2_lines.append(ln)
        repaired_prompt = "\n".join(pass2_lines)

    return {
        "repaired": repaired_prompt != prompt_text,
        "prompt": repaired_prompt,
        "original_max_code_lines": lint["max_consecutive_code_lines"],
        "snippet_paths": snippet_paths,
    }


_MUTATING_BINARIES: frozenset[str] = frozenset(
    {
        "cp",
        "mv",
        "rm",
        "unlink",
        "shred",
        "touch",
        "mkdir",
        "rmdir",
        "install",
        "rsync",
        "ln",
        "truncate",
        "chmod",
        "chown",
        "chgrp",
        "patch",
        "tee",
        "sponge",
        "sqlite3",
    }
)

_ALWAYS_BLOCKED_BINARIES: frozenset[str] = frozenset(
    {
        "ed",
        "ex",
        "vi",
        "vim",
        "nvim",
        "nano",
        "emacs",
        "unzip",
        "gunzip",
        "bunzip2",
        "unxz",
        "pip",
        "pip3",
        "autopep8",
        "yapf",
        "kill",
        "pkill",
        "killall",
    }
)

_BLOCKED_ENV_VARS: frozenset[str] = frozenset(
    {
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONHOME",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "PYTEST_DEBUG_TEMPROOT",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "NODE_OPTIONS",
        "RUBYOPT",
        "PERL5OPT",
        "GIT_SSH_COMMAND",
        "GIT_EXEC_PATH",
        "BASH_ENV",
        "ENV",
    }
)

_GIT_READONLY_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "status",
        "diff",
        "log",
        "show",
        "rev-parse",
        "ls-files",
        "ls-tree",
        "cat-file",
        "describe",
        "shortlog",
        "blame",
        "grep",
        "version",
        "help",
        "fetch",
    }
)

_GIT_OPTS_WITH_VALUE: frozenset[str] = frozenset(
    {
        "-C",
        "-c",
        "--git-dir",
        "--work-tree",
        "--namespace",
        "--super-prefix",
        "--config-env",
    }
)

_SUDO_OPTS_WITH_VALUE: frozenset[str] = frozenset(
    {"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-T", "-U"}
)

_STANDALONE_REDIR_OP_RE = re.compile(r"^(?:&>>?|\d*>>?|\d*>\|?|\d*>&|\d*<>|\d*<&?)$")
_ATTACHED_REDIR_TOKEN_RE = re.compile(r"^(?:&>>?|\d*>>?|\d*>\|?|\d*>&|\d*<>|\d*<&?).+$")


def _strip_shell_redirects_from_args(raw_args: list[str]) -> list[str]:
    """Remove shell redirect operators and their target tokens from tokenized command arguments."""
    cleaned: list[str] = []
    i = 0
    n = len(raw_args)
    while i < n:
        tok = raw_args[i]
        if _STANDALONE_REDIR_OP_RE.match(tok):
            i += 2
            continue
        if _ATTACHED_REDIR_TOKEN_RE.match(tok):
            i += 1
            continue
        cleaned.append(tok)
        i += 1
    return cleaned

_SAFE_INLINE_PYTHON_MODULES: frozenset[str] = frozenset(
    {
        "sys",
        "math",
        "json",
        "re",
        "time",
        "datetime",
        "collections",
        "itertools",
        "decimal",
        "fractions",
        "statistics",
        "hashlib",
        "ast",
    }
)

_FORBIDDEN_INLINE_PYTHON_NAMES: frozenset[str] = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "memoryview",
        "help",
        "license",
        "copyright",
        "credits",
    }
)

_FORBIDDEN_INLINE_PYTHON_ATTRS: frozenset[str] = frozenset(
    {
        "modules",
        "meta_path",
        "path_hooks",
        "path_importer_cache",
        "builtins",
        "audit",
        "addaudithook",
        "write",
        "writelines",
        "write_text",
        "write_bytes",
        "unlink",
        "rmdir",
        "rename",
        "replace",
        "mkdir",
        "makedirs",
        "touch",
        "chmod",
        "chown",
        "symlink_to",
        "hardlink_to",
        "system",
        "popen",
        "spawn",
        "fork",
        "kill",
        "remove",
        "truncate",
    }
)


def _is_readonly_git_diff_command(command: str) -> bool:
    """Return True if `command` is a single read-only `git diff` command without external helpers or arbitrary file reads."""
    raw = (command or "").strip()
    if not raw or any(sep in raw for sep in (";", "|", "&", "`", "$(", ">", "<", "\n")):
        return False
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return False
    if len(tokens) < 2 or Path(tokens[0]).name != "git":
        return False
    idx = 1
    while idx < len(tokens) and tokens[idx].startswith("-"):
        tok_g = tokens[idx]
        if tok_g in ("--no-pager", "-p", "--paginate"):
            idx += 1
        else:
            return False
    if idx >= len(tokens) or tokens[idx] != "diff":
        return False
    for tok in tokens[idx + 1 :]:
        if tok == "--":
            break
        if (
            tok in ("--output", "--ext-diff", "--textconv", "--no-index", "-x")
            or tok.startswith(("--output=", "--ext-diff=", "--textconv"))
        ):
            return False
    return True


def _session_has_multifile_delegate_job(data_root: Path) -> bool:
    """Return True when the current session has an applied delegate job touching >=2 files."""
    try:
        marker = data_root / ".lead_session_started"
        since = marker.stat().st_mtime if marker.exists() else 0.0
        receipts_dir = data_root / "receipts"
        if not receipts_dir.is_dir():
            return False
        for p in receipts_dir.glob("job-*.json"):
            if p.name.endswith(".progress.jsonl") or p.stat().st_mtime < since:
                continue
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(rec, dict) or rec.get("kind", "delegate") != "delegate":
                continue
            if not rec.get("applied"):
                continue
            applied_files = rec.get("applied_files") or []
            if isinstance(applied_files, list) and len(applied_files) >= 2:
                return True
    except OSError:
        pass
    return False


def _receipt_owner_pid_is_alive(rec: dict[str, Any]) -> bool:
    """Return False when `owner_instance_id` (`<pid>-<hex>` or `inst-<pid>-<hex>`) points to a dead local PID."""
    owner_id = str(rec.get("owner_instance_id") or "").strip()
    if not owner_id or "-" not in owner_id:
        return True
    parts = owner_id.split("-")
    pid_str = parts[1] if parts[0] == "inst" and len(parts) >= 3 else parts[0]
    try:
        pid = int(pid_str)
    except ValueError:
        return True
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True


def _session_has_live_running_delegate_job(data_root: Path) -> bool:
    """Return True when the current session has an active `running` delegate receipt owned by a live PID."""
    try:
        marker = data_root / ".lead_session_started"
        since = marker.stat().st_mtime if marker.exists() else 0.0
        receipts_dir = data_root / "receipts"
        if not receipts_dir.is_dir():
            return False
        for p in receipts_dir.glob("job-*.json"):
            if p.name.endswith((".progress.jsonl", ".live.json")) or p.stat().st_mtime < since:
                continue
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(rec, dict) or rec.get("kind", "delegate") != "delegate":
                continue
            if rec.get("status") == "running" and _receipt_owner_pid_is_alive(rec):
                return True
    except OSError:
        pass
    return False


def _session_has_terminal_delegate_job(data_root: Path) -> bool:
    """Return True when the current session has at least one completed (non-running, non-cancelled) delegate job."""
    try:
        marker = data_root / ".lead_session_started"
        since = marker.stat().st_mtime if marker.exists() else 0.0
        receipts_dir = data_root / "receipts"
        if not receipts_dir.is_dir():
            return False
        for p in receipts_dir.glob("job-*.json"):
            if p.name.endswith((".progress.jsonl", ".live.json")) or p.stat().st_mtime < since:
                continue
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(rec, dict) or rec.get("kind", "delegate") != "delegate":
                continue
            if rec.get("status") not in ("running", "cancelled") and not rec.get("cancelled"):
                return True
    except OSError:
        pass
    return False


def _session_has_lead_escalated_job(data_root: Path, ws_root: Path | None = None) -> bool:
    """Return True when any completed/stopped delegate receipt in the current session has `escalate_to_lead=True`
    and no worker job is actively running."""
    if _session_has_live_running_delegate_job(data_root):
        return False
    try:
        marker = data_root / ".lead_session_started"
        since = marker.stat().st_mtime - 1.0 if marker.exists() else 0.0
        if ws_root is not None:
            from server.receipts import ReceiptStore

            store = ReceiptStore(data_root, workspace_root=ws_root)
            for r_esc in store.list_receipts(workspace_root=ws_root):
                if r_esc.get("kind", "delegate") != "delegate":
                    continue
                if r_esc.get("status") in ("running", "cancelled") or r_esc.get("cancelled"):
                    continue
                if r_esc.get("external_worktree_tamper") or not store.verify_receipt_signature(r_esc):
                    continue
                rec_ts = float(
                    r_esc.get("updated_at_epoch")
                    or r_esc.get("finished_at_epoch")
                    or r_esc.get("started_at_epoch")
                    or 0.0
                )
                if rec_ts < since:
                    continue
                esc_info = r_esc.get("escalation")
                if r_esc.get("escalate_to_lead") is True or (
                    isinstance(esc_info, dict) and esc_info.get("escalate_to_lead") is True
                ):
                    return True
            return False
        receipts_dir = data_root / "receipts"
        if not receipts_dir.is_dir():
            return False
        fallback_store = ReceiptStore(data_root)
        for p in receipts_dir.glob("job-*.json"):
            if p.name.endswith((".progress.jsonl", ".live.json")) or p.stat().st_mtime < since:
                continue
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(rec, dict) or rec.get("kind", "delegate") != "delegate":
                continue
            if (
                rec.get("status") in ("running", "cancelled")
                or rec.get("cancelled")
                or rec.get("external_worktree_tamper")
                or not fallback_store.verify_receipt_signature(rec)
            ):
                continue
            esc_info = rec.get("escalation")
            if rec.get("escalate_to_lead") is True or (
                isinstance(esc_info, dict) and esc_info.get("escalate_to_lead") is True
            ):
                return True
    except Exception:
        pass
    return False


def _is_verified_applied_receipt(rec: dict[str, Any]) -> bool:
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
    acc_list = rec.get("acceptance_results")
    if not isinstance(acc_list, list) or not acc_list:
        return False
    return any(
        isinstance(r, dict)
        and r.get("passed") is True
        and not r.get("compile_smoke")
        and (
            not r.get("auto_discovered_default")
            or int((r.get("counts") or {}).get("passed") or 0) > 0
        )
        and not r.get("auto_discovered_regression")
        and not r.get("no_tests_collected")
        and not r.get("runner_incompatible_skipped")
        and not r.get("auto_import_smoke")
        for r in acc_list
    )


_UNLIMITED_READ_VALUES = ("off", "none", "unlimited", "disabled")


def resolve_lead_read_limit(env: dict[str, str]) -> int | None:
    """Return the strict-mode Lead read/search budget before review bonuses, or None for no cap.

    `CC_AGY_LEAD_READ_LIMIT` (alias `CC_AGY_MAX_LEAD_READS`): a non-negative integer is the number of
    reads allowed (`0` means none before delegating); a negative number or `off` removes the cap. When
    unset or invalid, all profiles default to 2 (so Economy allows up to 2 lightweight Glob/Grep/Read
    scouting calls when entrypoints or test paths are uncertain, while pre-delegation Bash remains blocked).
    """
    default = 2
    raw = str(env.get("CC_AGY_LEAD_READ_LIMIT") or env.get("CC_AGY_MAX_LEAD_READS") or "").strip().lower()
    if not raw or raw.startswith("${"):
        return default
    if raw in _UNLIMITED_READ_VALUES:
        return None
    try:
        value = int(raw)
    except ValueError:
        return default
    return None if value < 0 else value


def _lead_review_read_bonus(data_root: Path, env: dict[str, str]) -> int:
    """Extra strict-mode Lead reads granted once a worker job started in the current session exists.

    When a job auto-applied cleanly with passing acceptance checks and no missing symbols (and
    `CC_AGY_LEAD_REVIEW_READ_LIMIT` was not explicitly overridden), returns 0 greedily and stickily across
    subsequent 0-edit follow-up jobs so the Lead finishes immediately instead of re-reading verified files.
    """
    raw_override = (env.get("CC_AGY_LEAD_REVIEW_READ_LIMIT") or "").strip()
    try:
        bonus = int(raw_override or "6")
    except ValueError:
        bonus = 6
    if bonus <= 0:
        return 0
    try:
        marker = data_root / ".lead_session_started"
        since = marker.stat().st_mtime if marker.exists() else 0.0
        receipts_dir = data_root / "receipts"
        if not receipts_dir.is_dir():
            return 0
        session_files = [
            p for p in receipts_dir.glob("job-*")
            if p.stat().st_mtime >= since
        ]
        if not session_files:
            return 0
        json_receipts = [
            p for p in session_files
            if p.name.endswith(".json") and not p.name.endswith((".progress.jsonl", ".live.json"))
        ]
        if json_receipts:
            parsed_recs: list[tuple[int, dict[str, Any]]] = []
            for jp in json_receipts:
                try:
                    rec_obj = json.loads(jp.read_text(encoding="utf-8"))
                    if isinstance(rec_obj, dict):
                        parsed_recs.append((jp.stat().st_mtime_ns, rec_obj))
                except Exception:
                    continue
            if not parsed_recs:
                return 0
            if any(
                r.get("status") == "running" and _receipt_owner_pid_is_alive(r)
                for _mt, r in parsed_recs
            ):
                return 0
            terminal_recs = [
                (mt, r)
                for mt, r in parsed_recs
                if r.get("status") not in ("running", "cancelled") and not r.get("cancelled")
            ]
            if not terminal_recs:
                return 0
            if not raw_override:
                ordered_recs = sorted(
                    terminal_recs,
                    key=lambda item: (float(item[1].get("started_at_epoch") or 0.0), item[0]),
                )
                sticky_verified_applied = False
                for _mt, rec in ordered_recs:
                    if _is_verified_applied_receipt(rec):
                        sticky_verified_applied = True
                    elif rec.get("reverted") or bool(rec.get("files_changed")):
                        sticky_verified_applied = False
                if sticky_verified_applied:
                    return 0
        return bonus
    except OSError:
        return 0


_CMD_HEAD_PREFIX_RE = (
    r"(?:^|[;&|`(\n])\s*(?:(?:&&|\|\|)\s*)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*"
    r"(?:(?:"
    r"env(?:\s+(?:-[A-Za-z0-9]+|--[A-Za-z0-9-]+(?:=\S+)?|-C\s+\S+|-u\s+\S+|[A-Za-z_][A-Za-z0-9_]*=\S*))*"
    r"|sudo(?:\s+(?:-[A-Za-z0-9]+|--[A-Za-z0-9-]+(?:=\S+)?|-u\s+\S+))*"
    r"|command|exec|time|nohup|nice(?:\s+-n\s+\S+)?"
    r"|xargs(?:\s+(?:-[A-Za-z0-9]+(?:\s+\S+)?|--[A-Za-z0-9-]+(?:=\S+)?))*"
    r"|watch(?:\s+(?:-[A-Za-z0-9]+(?:\s+\S+)?|--[A-Za-z0-9-]+(?:=\S+)?))*"
    r"|timeout(?:\s+(?:-[A-Za-z0-9]+|--[A-Za-z0-9-]+(?:=\S+)?))?\s+\S+"
    r"|uv\s+run(?:\s+(?:--[A-Za-z0-9-]+(?:\s+\S+|=[^\s]+)?|-[A-Za-z0-9]+))*"
    r"|poetry\s+run|pipenv\s+run|pdm\s+run|hatch\s+run"
    r"|(?:pnpm|npm|yarn)(?:\s+(?:--filter(?:=\S+|\s+\S+)|-F\s+\S+|-C\s+\S+|--prefix(?:=\S+|\s+\S+)|--cwd(?:=\S+|\s+\S+)))*\s+exec"
    r"|npx(?:\s+-[A-Za-z0-9-]+)*|bunx|pnpx|bun\s+x|yarn(?:\s+--cwd\s+\S+)?\s+dlx"
    r")\s+)*"
)
_HOST_TEST_RUNNER_RE = re.compile(
    _CMD_HEAD_PREFIX_RE
    + r"(?:"
    r"(?:\S*/)?(?:pytest|py\.test|vitest|jest|tox|nox)(?=$|[\s;|&'\")`])"
    r"|(?:\S*/)?python[0-9.]*(?:\s+(?:-[XWO]\s+\S+|-[A-Za-z0-9]+|--[A-Za-z0-9-]+(?:=\S+)?))*\s+-m\s+(?:pytest|unittest)(?=$|[\s;|&'\")`])"
    r"|(?:pnpm|npm|yarn|bun)(?:\s+(?:--filter(?:=\S+|\s+\S+)|-F\s+\S+|-C\s+\S+|--prefix(?:=\S+|\s+\S+)|--cwd(?:=\S+|\s+\S+)|workspace\s+\S+|-[A-Za-z0-9-]+|--[A-Za-z0-9-]+=\S+))*\s+(?:(?:run|run-script)\s+test(?::[A-Za-z0-9_.:-]+)?|test(?::[A-Za-z0-9_.:-]+)?|t)(?=$|[\s;|&'\")`])"
    r"|hatch\s+run\s+(?:[A-Za-z0-9_.-]+:)?test(?=$|[\s;|&'\")`])"
    r"|go(?:\s+(?:-C\s+\S+|--C\s+\S+|-C=\S+|--C=\S+))*\s+test(?=$|[\s;|&'\")`])"
    r"|cargo(?:\s+\+\S+)?\s+test(?=$|[\s;|&'\")`])"
    r"|deno\s+test(?=$|[\s;|&'\")`])"
    r"|make(?:\s+-[A-Za-z0-9]+(?:\s+\S+)?)?\s+test(?=$|[\s;|&'\")`])"
    r"|node\s+--test(?=$|[\s;|&'\")`])"
    r"|(?:node|nodejs|bun)\s+(?:\./)?(?:[A-Za-z0-9_.-]+/){0,3}(?:node_modules/(?:\.pnpm/[^/]+/node_modules/)?(?:vitest/(?:vitest\.mjs|dist/cli\.c?m?js)|jest/bin/jest\.c?m?js|mocha/bin/(?:mocha|_mocha)(?:\.c?m?js)?|ava/entrypoints/cli\.mjs|tap/bin/run\.c?m?js|uvu/bin\.c?m?js|tape/bin/tape|jasmine/bin/jasmine\.c?m?js)|node_modules/\.bin/(?:vitest|jest|mocha|ava|tap|uvu|tape|jasmine))(?=$|[\s;|&'\")`])"
    r")"
)
_NODE_TEST_ENTRYPOINT_RE = re.compile(
    r"^(?:[A-Za-z0-9_.-]+/){0,3}"
    r"(?:"
    r"node_modules/(?:\.pnpm/[^/]+/node_modules/)?"
    r"(?:"
    r"vitest/(?:vitest\.mjs|dist/cli\.c?m?js)"
    r"|jest/bin/jest\.c?m?js"
    r"|mocha/bin/(?:mocha|_mocha)(?:\.c?m?js)?"
    r"|ava/entrypoints/cli\.mjs"
    r"|tap/bin/run\.c?m?js"
    r"|uvu/bin\.c?m?js"
    r"|tape/bin/tape"
    r"|jasmine/bin/jasmine\.c?m?js"
    r")"
    r"|node_modules/\.bin/(?:vitest|jest|mocha|ava|tap|uvu|tape|jasmine)"
    r")$"
)
_NODE_TSC_ENTRYPOINT_RE = re.compile(
    r"^(?:[A-Za-z0-9_.-]+/){0,3}"
    r"(?:"
    r"node_modules/(?:\.pnpm/[^/]+/node_modules/)?typescript/(?:lib|bin)/tsc\.c?m?js"
    r"|node_modules/\.bin/tsc"
    r")$"
)
_NODE_MUTATING_TEST_FLAGS = {
    "-u",
    "--u",
    "--update",
    "--updateSnapshot",
    "--update-snapshot",
    "--ci=false",
    "--init",
    "--build",
    "-b",
    "--outFile",
    "--outDir",
    "--declarationDir",
}
_NODE_MUTATING_TEST_FLAG_PREFIXES = (
    "-u=",
    "--u=",
    "--update=",
    "--updateSnapshot=",
    "--update-snapshot=",
    "--outFile=",
    "--outDir=",
    "--declarationDir=",
)
_SHELL_C_INNER_RE = re.compile(
    r"(?:^|[;&|`(\n])\s*(?:(?:&&|\|\|)\s*)?(?:\S*/)?(?:ba|z|da|k)?sh\b[^;&|\n]*?\s-[A-Za-z]*c\s+(['\"])(.*?)\1",
    re.DOTALL,
)
_WHICH_SUBSHELL_RE = re.compile(
    r"\"?\$\(\s*(?:which|command\s+-v)\s+([A-Za-z0-9_./-]+)\s*\)\"?|`\s*(?:which|command\s+-v)\s+([A-Za-z0-9_./-]+)\s*`"
)
_QUOTED_SPAN_RE = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"", re.DOTALL)


def _command_invokes_host_test_runner(command: str, _depth: int = 0) -> bool:
    if not command:
        return False
    if _depth < 3:
        for m in _SHELL_C_INNER_RE.finditer(command):
            if _command_invokes_host_test_runner(m.group(2), _depth + 1):
                return True
    expanded = _WHICH_SUBSHELL_RE.sub(lambda m: m.group(1) or m.group(2) or "", command)
    masked = _QUOTED_SPAN_RE.sub("__QUOTED_ARG__", expanded)
    return bool(_HOST_TEST_RUNNER_RE.search(masked))


def _check_strict_bash_test_runner_guard(
    command: str,
    ws_root: Path,
    out_dir: Path | str | None,
    env: dict[str, str],
) -> str | None:
    """Prevent Lead host test-suite runs while a worker job is running or in Economy/zero-read mode."""
    if not _command_invokes_host_test_runner(command):
        return None
    try:
        try:
            from server.receipts import ReceiptStore

            data_root = ReceiptStore._resolve_data_dir(out_dir or None, ws_root)
        except Exception:
            data_root = Path(out_dir) if out_dir else (ws_root / ".cc-agy-data")
        marker = data_root / ".lead_session_started"
        since = marker.stat().st_mtime if marker.exists() else 0.0
        receipts_dir = data_root / "receipts"
        parsed: list[tuple[int, dict[str, Any]]] = []
        if receipts_dir.is_dir():
            for jp in receipts_dir.glob("job-*.json"):
                if jp.name.endswith(".live.json") or jp.stat().st_mtime < since:
                    continue
                try:
                    obj = json.loads(jp.read_text(encoding="utf-8"))
                    if isinstance(obj, dict) and obj.get("job_id"):
                        parsed.append((jp.stat().st_mtime_ns, obj))
                except Exception:
                    continue
        for _mt, rec in parsed:
            if rec.get("status") == "running" and _receipt_owner_pid_is_alive(rec):
                jid = str(rec.get("job_id") or "")
                return (
                    f"Strict orchestrator mode: delegated worker job {jid!r} is still running in its isolated "
                    f"worktree. Do NOT run test suites on the un-applied host workspace; call "
                    f"delegate_result(job_id={jid!r}, wait=true) to wait for the worker."
                )
        if resolve_cost_profile(env) != "economy" and resolve_lead_read_limit(env) != 0:
            return None
        if _session_has_lead_escalated_job(data_root, ws_root):
            return None
        terminal = [
            (mt, r)
            for mt, r in parsed
            if r.get("status") not in ("running", "cancelled") and not r.get("cancelled")
        ]
        any_applied = any(bool(r.get("applied")) for _mt, r in terminal)
        if not terminal or not any_applied:
            return (
                "Economy strict mode: no worker diff has been applied to the host workspace yet (host is still "
                "at clean HEAD, and running tests on the host can write untracked test artifacts that conflict "
                "with delegate_apply). Delegate test execution and fixes via delegate_start or delegate_continue."
            )
        if _lead_review_read_bonus(data_root, env) == 0:
            return (
                "Economy strict mode: worker changes were already verified by the acceptance test suite and "
                "applied cleanly. Finish your response directly without re-running tests on the host."
            )
    except Exception:
        return None
    return None


def _is_workspace_python_package(root_mod: str, ws_root: Path | tuple[Path, ...] | None) -> bool:
    """Return True if `root_mod` is a valid Python package directory (`__init__.py`) inside `ws_root` (or `ws_root/src`)."""
    if (
        ws_root is None
        or not root_mod
        or not root_mod.isidentifier()
        or root_mod in ("server", "scripts", "tests", "test", "setup", "conftest")
    ):
        return False
    roots = ws_root if isinstance(ws_root, tuple) else (ws_root,)
    for r in roots:
        if (r / root_mod / "__init__.py").is_file() or (r / "src" / root_mod / "__init__.py").is_file():
            return True
    return False


def _is_safe_workspace_pythonpath(
    val: str,
    ws_root: Path,
    out_dir: str | Path | None,
    active_cwd: Path,
    env: dict[str, str] | None,
) -> bool:
    """Return True if every path segment in `PYTHONPATH=<val>` resolves strictly inside `ws_root`."""
    raw = (val or "").strip().strip("'\"")
    if not raw:
        return True
    segments = [s.strip() for s in raw.split(":") if s.strip() and s.strip() not in ("$PYTHONPATH", "${PYTHONPATH}")]
    if not segments:
        return True
    ws_resolved = ws_root.resolve()
    cwd_resolved = active_cwd.resolve()
    for seg in segments:
        expanded = _expand_shell_vars_in_path(seg, active_cwd, ws_root)
        if any(ch in expanded for ch in ("$", "`", "~")):
            return False
        cand = Path(expanded)
        if ".." in cand.parts:
            return False
        resolved = (cwd_resolved / cand).resolve() if not cand.is_absolute() else cand.resolve()
        if not (resolved.is_relative_to(ws_resolved) and resolved.is_relative_to(cwd_resolved)):
            return False
        if classify_path(str(resolved), ws_root, out_dir, cwd=active_cwd, env=env) != "workspace":
            return False
    return True


def _is_safe_python_inline_ast(code_str: str, ws_root: Path | tuple[Path, ...] | None = None) -> bool:
    """Validate `python -c <code>` using a two-pass AST allowlist so dynamic/obfuscated mutations are rejected."""
    try:
        tree = ast.parse(code_str, mode="exec")
    except (SyntaxError, ValueError, TypeError):
        return False

    all_nodes = list(ast.walk(tree))
    ws_imported_names: set[str] = set()
    # Pass 1: collect all imports first (ast.walk is breadth-first, so nested imports appear after top-level functions).
    for node in all_nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".", 1)[0]
                if root_mod in _SAFE_INLINE_PYTHON_MODULES:
                    continue
                if _is_workspace_python_package(root_mod, ws_root):
                    ws_imported_names.add(alias.asname or root_mod)
                    continue
                return False
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                return False
            root_mod = node.module.split(".", 1)[0]
            if root_mod not in _SAFE_INLINE_PYTHON_MODULES:
                return False
            for alias in node.names:
                if alias.name.startswith("_") or alias.name in _FORBIDDEN_INLINE_PYTHON_ATTRS:
                    return False

    # Pass 2: validate names, attributes, string constants, and calls against collected imports.
    for node in all_nodes:
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in _FORBIDDEN_INLINE_PYTHON_NAMES:
                return False
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in _FORBIDDEN_INLINE_PYTHON_ATTRS:
                return False
            # If workspace packages were imported (`allow_workspace_inline_imports=True`), forbid arbitrary
            # attribute traversal on them (e.g. `mypkg.shutil.rmtree(...)`) and allow only `<pkg>.test(...)`.
            if ws_imported_names:
                if not (isinstance(node.value, ast.Name) and node.value.id in ws_imported_names and node.attr in ("test", "doctest")):
                    return False
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            segments = [seg for seg in re.split(r"[.\[\]:!]", node.value) if seg]
            if (
                node.value.startswith("__")
                or node.value in _FORBIDDEN_INLINE_PYTHON_ATTRS
                or node.value in _FORBIDDEN_INLINE_PYTHON_NAMES
                or any(
                    seg.startswith("__")
                    or seg in _FORBIDDEN_INLINE_PYTHON_ATTRS
                    or seg in _FORBIDDEN_INLINE_PYTHON_NAMES
                    for seg in segments
                )
            ):
                return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                if any(kw.arg in ("file", None) for kw in node.keywords):
                    return False
            if ws_imported_names:
                if not (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ws_imported_names
                    and node.func.attr in ("test", "doctest")
                ):
                    return False
    return True


def _unquoted_mask(s: str) -> list[bool]:
    """Return a boolean mask where mask[i] is True iff s[i] is outside quotes and not backslash-escaped."""
    n = len(s)
    mask = [False] * n
    in_single = False
    in_double = False
    i = 0
    while i < n:
        ch = s[i]
        if in_single:
            if ch == "'":
                in_single = False
            i += 1
        elif in_double:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
        else:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            else:
                mask[i] = True
            i += 1
    return mask


def _split_shell_segments(s: str) -> list[str]:
    """Split a shell command string on unquoted ;, |, &&, ||, or newline."""
    mask = _unquoted_mask(s)
    n = len(s)
    segments: list[str] = []
    start = 0
    i = 0
    while i < n:
        if mask[i]:
            if i + 1 < n and mask[i + 1] and s[i : i + 2] in ("&&", "||"):
                segments.append(s[start:i])
                i += 2
                start = i
                continue
            if s[i] in (";", "|", "\n"):
                segments.append(s[start:i])
                i += 1
                start = i
                continue
        i += 1
    segments.append(s[start:])
    return segments


def _expand_shell_vars_in_path(target: str, active_cwd: Path, ws_root: Path) -> str:
    """Replace common shell path variables ($PWD, $(pwd), $(git rev-parse --show-toplevel)) before classifying."""
    s = target.strip()
    s = s.replace("$(git rev-parse --show-toplevel)", str(ws_root))
    s = s.replace("`git rev-parse --show-toplevel`", str(ws_root))
    s = s.replace("${PWD}", str(active_cwd)).replace("$PWD", str(active_cwd))
    s = s.replace("$(pwd)", str(active_cwd)).replace("`pwd`", str(active_cwd))
    return s


_DANGEROUS_GIT_CONFIG_SUBSTRS = (
    "!",
    "alias.",
    "hookspath",
    "fsmonitor",
    "diff.external",
    "pager",
    "sshcommand",
    "textconv",
    "credential.",
    "filter.",
    "uploadpack",
    "receivepack",
    "editor",
)


def _is_dangerous_git_config_val(val: str) -> bool:
    lower_v = val.lower()
    return any(tok in lower_v for tok in _DANGEROUS_GIT_CONFIG_SUBSTRS)


def _extract_git_subcommand_and_workdir(
    args: list[str],
    current_cwd: Path,
) -> tuple[str | None, list[str], Path, str | None]:
    """Skip git global flags (-C, -c, --git-dir, --work-tree) and return (subcommand, sub_args, effective_cwd, config_err)."""
    idx = 0
    eff_cwd = current_cwd
    while idx < len(args):
        tok = args[idx]
        if tok == "--":
            idx += 1
            break
        if tok in _GIT_OPTS_WITH_VALUE:
            if tok == "--config-env":
                return None, [], eff_cwd, "Strict orchestrator mode blocked `git --config-env`."
            if idx + 1 < len(args):
                val = args[idx + 1]
                if tok == "-C":
                    eff_cwd = (eff_cwd / val).resolve()
                elif tok == "-c":
                    if _is_dangerous_git_config_val(val):
                        return None, [], eff_cwd, f"Strict orchestrator mode blocked `git -c {val}`."
            idx += 2
            continue
        if any(tok.startswith(prefix + "=") for prefix in _GIT_OPTS_WITH_VALUE):
            if tok.startswith("--config-env="):
                return None, [], eff_cwd, f"Strict orchestrator mode blocked `git {tok}`."
            val = tok.split("=", 1)[1]
            if tok.startswith("-C="):
                eff_cwd = (eff_cwd / val).resolve()
            elif tok.startswith("-c="):
                if _is_dangerous_git_config_val(val):
                    return None, [], eff_cwd, f"Strict orchestrator mode blocked `git {tok}`."
            idx += 1
            continue
        if tok.startswith("-C") and len(tok) > 2:
            eff_cwd = (eff_cwd / tok[2:]).resolve()
            idx += 1
            continue
        if tok.startswith("-c") and len(tok) > 2:
            val = tok[2:]
            if _is_dangerous_git_config_val(val):
                return None, [], eff_cwd, f"Strict orchestrator mode blocked `git {tok}`."
            idx += 1
            continue
        if tok.startswith("-"):
            idx += 1
            continue
        return tok, args[idx + 1 :], eff_cwd, None
    return None, [], eff_cwd, None


_SAFE_PYTEST_INI_KEYS: frozenset[str] = frozenset({
    "addopts",
    "xfail_strict",
    "filterwarnings",
    "log_cli",
    "log_cli_level",
    "log_level",
    "asyncio_mode",
    "python_files",
    "python_classes",
    "python_functions",
    "markers",
    "console_output_style",
    "junit_family",
})

_BLOCKED_PYTEST_FLAG_PREFIXES: tuple[str, ...] = (
    "--confcutdir",
    "--rootdir",
    "--noconftest",
    "--basetemp",
    "--junitxml",
    "--junit-xml",
    "--result-log",
    "--resultlog",
    "--log-file",
    "--cov",
    "--html",
    "--self-contained-html",
    "--import-mode",
    "--pastebin",
    "--snapshot-update",
    "--update-snapshots",
    "--force-regen",
    "--regen-all",
    "--inline-snapshot",
    "--record-mode",
)


def _is_safe_pytest_ini_override(
    ini_expr: str,
    *,
    ws_root: Path | None = None,
    out_dir: str | Path | None = None,
    active_cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> bool:
    """Return True if a `pytest -o key=val` expression only sets a read-only ini key without unsafe flags."""
    key, sep, val = (ini_expr or "").strip().partition("=")
    clean_key = key.strip()
    if not sep or clean_key not in _SAFE_PYTEST_INI_KEYS:
        return False
    if clean_key == "addopts":
        try:
            sub_args = shlex.split(val)
        except ValueError:
            return False
        eff_ws = ws_root or Path.cwd().resolve()
        eff_cwd = active_cwd or eff_ws
        sub_v = _validate_pytest_args(
            sub_args,
            ws_root=eff_ws,
            out_dir=out_dir,
            active_cwd=eff_cwd,
            env=env,
            allow_nested_ini=False,
        )
        return not sub_v["blocked"]
    if any(
        tok in val
        for tok in (
            "--confcutdir",
            "--rootdir",
            "--override-ini",
            "--basetemp",
            "--junitxml",
            "--result-log",
            "--log-file",
        )
    ):
        return False
    return True


def _validate_pytest_args(
    args: list[str],
    *,
    ws_root: Path,
    out_dir: str | Path | None,
    active_cwd: Path,
    env: dict[str, str] | None,
    allow_nested_ini: bool = True,
) -> dict[str, Any]:
    """Validate pytest CLI arguments (used for both `pytest` and `python -m pytest` and `-o addopts=...`)."""
    skip_next = False
    for idx_p, parg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if parg == "--":
            continue
        if parg == "-c" or parg.startswith("-c="):
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `pytest {parg}` plugin/config override.",
            }
        if any(parg == pref or parg.startswith(pref + "=") for pref in _BLOCKED_PYTEST_FLAG_PREFIXES):
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `pytest {parg}` plugin/config override.",
            }
        if parg == "-p":
            next_val = args[idx_p + 1] if idx_p + 1 < len(args) else ""
            if next_val.startswith("no:"):
                skip_next = True
                continue
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `pytest {parg}` plugin/config override.",
            }
        if parg.startswith("-p") and len(parg) > 2 and not parg.startswith("--"):
            if parg[2:].lstrip("=").startswith("no:"):
                continue
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `pytest {parg}` plugin/config override.",
            }
        if parg in ("-o", "--override-ini"):
            if not allow_nested_ini:
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked nested `pytest {parg}` override.",
                }
            next_val = args[idx_p + 1] if idx_p + 1 < len(args) else ""
            if _is_safe_pytest_ini_override(
                next_val, ws_root=ws_root, out_dir=out_dir, active_cwd=active_cwd, env=env
            ):
                skip_next = True
                continue
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `pytest {parg}` plugin/config override.",
            }
        if parg.startswith("--override-ini=") or (parg.startswith("-o") and len(parg) > 2 and not parg.startswith("--")):
            if not allow_nested_ini:
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked nested `pytest {parg}` override.",
                }
            ini_val = parg.split("=", 1)[1] if parg.startswith("--override-ini=") else parg[2:].lstrip("=")
            if _is_safe_pytest_ini_override(
                ini_val, ws_root=ws_root, out_dir=out_dir, active_cwd=active_cwd, env=env
            ):
                continue
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `pytest {parg}` plugin/config override.",
            }
        if parg in ("-k", "-m", "-W", "-r", "--tb", "--maxfail", "--durations", "--durations-min", "--color", "--capture"):
            skip_next = True
            continue
        if parg.startswith("-") and not parg.startswith("--") and len(parg) > 2:
            # Inspect clustered short flags (e.g. -qo, -qp, -qc)
            short_chars = parg[1:]
            for idx_ch, ch in enumerate(short_chars):
                if ch in ("k", "m", "W", "r"):
                    if idx_ch == len(short_chars) - 1:
                        skip_next = True
                    break
                if ch == "c":
                    return {
                        "blocked": True,
                        "reason": f"Strict orchestrator mode blocked `pytest {parg}` plugin/config override.",
                    }
                if ch == "p":
                    tail = short_chars[idx_ch + 1 :].lstrip("=")
                    if tail:
                        if tail.startswith("no:"):
                            break
                    elif idx_p + 1 < len(args) and args[idx_p + 1].startswith("no:"):
                        skip_next = True
                        break
                    return {
                        "blocked": True,
                        "reason": f"Strict orchestrator mode blocked `pytest {parg}` plugin/config override.",
                    }
                if ch == "o":
                    if not allow_nested_ini:
                        return {
                            "blocked": True,
                            "reason": f"Strict orchestrator mode blocked nested `pytest {parg}` override.",
                        }
                    tail = short_chars[idx_ch + 1 :].lstrip("=")
                    if tail:
                        if _is_safe_pytest_ini_override(
                            tail, ws_root=ws_root, out_dir=out_dir, active_cwd=active_cwd, env=env
                        ):
                            break
                    elif idx_p + 1 < len(args) and _is_safe_pytest_ini_override(
                        args[idx_p + 1], ws_root=ws_root, out_dir=out_dir, active_cwd=active_cwd, env=env
                    ):
                        skip_next = True
                        break
                    return {
                        "blocked": True,
                        "reason": f"Strict orchestrator mode blocked `pytest {parg}` plugin/config override.",
                    }
            continue
        if parg.startswith("-"):
            continue
        target_file = parg.split("::", 1)[0]
        cls = classify_path(target_file, ws_root, out_dir, cwd=active_cwd, env=env)
        if cls != "workspace":
            return {
                "blocked": True,
                "reason": (
                    f"Strict orchestrator mode blocked `pytest` on non-workspace path {parg!r} ({cls})."
                ),
            }
    return {"blocked": False, "reason": None}


def detect_bash_workspace_mutation(
    command: str,
    workdir: str | Path,
    out_dir: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    allow_workspace_inline_imports: bool = False,
    _depth: int = 0,
) -> dict[str, Any]:
    """Inspect a Bash command for direct workspace or receipt-store mutations in strict mode."""
    if _depth > 8:
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked deeply nested shell command.",
        }

    cmd = (command or "").strip()
    if not cmd:
        return {"blocked": False, "reason": None}

    ws_root = resolve_project_root(workdir, env=env)
    active_cwd = Path(cwd).resolve() if cwd is not None else Path(workdir).resolve()

    # 0. Block direct access to .hmac_key, PLUGIN_ROOT, ${IFS} / $'...' obfuscation, and process substitution <( / >(
    if ".hmac_key" in cmd:
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked reading or modifying the internal `.hmac_key`.",
        }
    if ws_root != PLUGIN_ROOT and str(PLUGIN_ROOT) in cmd:
        return {
            "blocked": True,
            "reason": (
                "Strict orchestrator mode blocked accessing or modifying the cc-agy-plugin installation "
                "directory. Use the MCP tools (`delegate_start`, `delegate_continue`, `delegate_apply`, "
                "`get_diff`) directly."
            ),
        }
    if "${IFS}" in cmd or "$IFS" in cmd or "$'" in cmd:
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked `$IFS` or ANSI-C `$'...'` shell token obfuscation.",
        }
    if re.search(r"[<>]\(", cmd):
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked bash process substitution `<(...)` / `>(...)`.",
        }

    # 1. Compute unquoted mask and check inline sed/perl in-place flags outside quotes
    cmd_without_herestrings = re.sub(r"<<<", "   ", cmd)
    cmd_without_heredocs = re.sub(r"<<-?\s*['\"]?\w+['\"]?", "   ", cmd_without_herestrings)
    unquoted = _unquoted_mask(cmd_without_heredocs)
    cmd_unquoted = "".join(
        ch if (i < len(unquoted) and unquoted[i]) else " "
        for i, ch in enumerate(cmd_without_heredocs)
    )
    if re.search(r"\bsed\s+(?:-[a-zA-Z]*i|--in-place)\b", cmd_unquoted):
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked `sed -i` in-place edit.",
        }
    if re.search(r"\bperl\s+-[a-zA-Z]*[pi]", cmd_unquoted):
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked `perl -p/-i` in-place mutation.",
        }

    # 2. Check shell output redirections (>, >>, >|, &>, &>>, 1>, 2>, <>),
    # ignoring here-strings (<<<), heredocs (<<), and fd duplicates (2>&1, >/dev/null).
    redir_pattern = re.compile(
        r"(?:&>>?|\d?>>?|\d?>\|?|<>|\b(?:tee|sponge)\s+(?:-[a-zA-Z]+\s+)*)\s*(&?[^\s;|&\n]+)"
    )
    for match in redir_pattern.finditer(cmd_without_heredocs):
        if match.start() < len(unquoted) and not unquoted[match.start()]:
            continue
        raw_target = match.group(1).strip("'\"()")
        if (
            not raw_target
            or re.match(r"^&(?:\d+|-)$", raw_target)
            or raw_target in ("/dev/null", "/dev/stderr", "/dev/stdout")
        ):
            continue
        if raw_target.startswith("&"):
            raw_target = raw_target[1:].strip("'\"()")
            if not raw_target:
                continue
        target = _expand_shell_vars_in_path(raw_target, active_cwd, ws_root)
        classification = classify_path(target, ws_root, out_dir, cwd=active_cwd, env=env)
        if classification in ("workspace", "out_dir"):
            return {
                "blocked": True,
                "reason": (
                    f"Strict orchestrator mode blocked shell redirection to {classification} path {raw_target!r}. "
                    f"Use delegate_start / delegate_apply instead."
                ),
            }

    # 2b. Check input redirection feeding an interpreter or patch (e.g. `python3 < /tmp/writer.py`, `patch -p1 < /tmp/p.diff`)
    for stdin_match in re.finditer(
        r"\b(?:python\d*(?:\.\d+)?|pypy\d*|node|ruby|perl|php|sh|bash|zsh|dash|ksh|patch|git\s+apply)\b[^;|&\n]*?(<\s*([^\s<;|&\n]+))",
        cmd_without_heredocs,
    ):
        lt_pos = stdin_match.start(1)
        if lt_pos < len(unquoted) and unquoted[lt_pos]:
            return {
                "blocked": True,
                "reason": (
                    "Strict orchestrator mode blocked interpreter/patch stdin redirection. "
                    "Delegate workspace modifications via delegate_start."
                ),
            }

    # 3. Check command substitutions $( ... ) or ` ... ` recursively
    for sub_match in re.finditer(r"\$\(([^)]+)\)|`([^`]+)`", cmd):
        inner_sub = sub_match.group(1) or sub_match.group(2) or ""
        sub_verdict = detect_bash_workspace_mutation(
            inner_sub,
            ws_root,
            out_dir,
            cwd=active_cwd,
            env=env,
            allow_workspace_inline_imports=allow_workspace_inline_imports,
            _depth=_depth + 1,
        )
        if sub_verdict["blocked"]:
            return sub_verdict

    # 4. Parse shell segments separated by unquoted ;, &&, ||, |, or newlines
    segments = _split_shell_segments(cmd_without_heredocs)
    for raw_seg in segments:
        seg = raw_seg.strip()
        # Strip outer subshell / brace group wrappers `( ... )` and `{ ... }`
        while seg and (seg[0] in "({" or seg[-1] in ")}"):
            seg = seg.lstrip("({ \t").rstrip(")} \t")
        if not seg:
            continue

        # Detect obfuscated or dynamic command heads before shlex strips quotes/escapes
        # e.g. `$(echo cp) ...`, `c\p ...`, `'c'p ...`
        seg_mask = _unquoted_mask(seg)
        raw_words: list[str] = []
        w_start: int | None = None
        for idx_ch, ch in enumerate(seg):
            if seg_mask[idx_ch] and ch.isspace():
                if w_start is not None:
                    raw_words.append(seg[w_start:idx_ch])
                    w_start = None
            elif w_start is None:
                w_start = idx_ch
        if w_start is not None:
            raw_words.append(seg[w_start:])
        raw_head_idx = 0
        while (
            raw_head_idx < len(raw_words)
            and "=" in raw_words[raw_head_idx]
            and not raw_words[raw_head_idx].startswith("-")
            and raw_words[raw_head_idx].split("=", 1)[0].isidentifier()
        ):
            raw_head_idx += 1
        if raw_head_idx < len(raw_words):
            raw_head = raw_words[raw_head_idx]
            if any(ch in raw_head for ch in ("\\", "'", '"', "$", "`")):
                return {
                    "blocked": True,
                    "reason": (
                        f"Strict orchestrator mode blocked dynamic or quote/backslash-obfuscated command head {raw_head!r}."
                    ),
                }

        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if not tokens:
            continue

        tokens = [_expand_shell_vars_in_path(t, active_cwd, ws_root) for t in tokens]
        verdict, active_cwd = _inspect_segment_tokens(
            tokens,
            ws_root=ws_root,
            out_dir=out_dir,
            active_cwd=active_cwd,
            env=env,
            depth=_depth,
            allow_workspace_inline_imports=allow_workspace_inline_imports,
        )
        if verdict["blocked"]:
            return verdict

    return {"blocked": False, "reason": None}


def _inspect_segment_tokens(
    tokens: list[str],
    *,
    ws_root: Path,
    out_dir: str | Path | None,
    active_cwd: Path,
    env: dict[str, str] | None,
    depth: int,
    allow_workspace_inline_imports: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Inspect a single tokenized command segment, unwrapping prefixes and tracking `cd`."""
    idx = 0
    n = len(tokens)
    used_xargs = False

    while idx < n:
        tok = Path(tokens[idx]).name if tokens[idx].startswith("/") else tokens[idx]
        # Inspect and skip leading VAR=val assignments
        if "=" in tok and not tok.startswith("-") and tok.split("=", 1)[0].isidentifier():
            var_name, var_val = tok.split("=", 1)
            if var_name in _BLOCKED_ENV_VARS:
                if var_name == "PYTHONPATH" and _is_safe_workspace_pythonpath(
                    var_val, ws_root, out_dir, active_cwd, env
                ):
                    idx += 1
                    continue
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked environment override `{var_name}=...`.",
                }, active_cwd
            idx += 1
            continue
        if tok in ("command", "nohup", "time", "exec", "builtin", "setsid", "busybox"):
            idx += 1
            while idx < n and tokens[idx].startswith("-"):
                idx += 1
            continue
        if tok in ("nice", "ionice", "chrt", "stdbuf", "watch"):
            idx += 1
            while idx < n and tokens[idx].startswith("-"):
                if tokens[idx] in ("-n", "--adjustment", "-c", "-p", "-i", "-o", "-e") and idx + 1 < n:
                    idx += 2
                else:
                    idx += 1
            continue
        if tok == "timeout":
            idx += 1
            while idx < n and tokens[idx].startswith("-"):
                if tokens[idx] in ("-k", "--kill-after", "-s", "--signal") and idx + 1 < n:
                    idx += 2
                else:
                    idx += 1
            if idx < n:
                idx += 1  # skip duration argument
            continue
        if tok == "flock":
            idx += 1
            while idx < n and tokens[idx].startswith("-"):
                if tokens[idx] in ("-w", "--timeout", "--wait", "-E", "--conflict-exit-code") and idx + 1 < n:
                    idx += 2
                elif tokens[idx] in ("-c", "--command") and idx + 1 < n:
                    sub_v = detect_bash_workspace_mutation(
                        tokens[idx + 1], ws_root, out_dir, cwd=active_cwd, env=env, _depth=depth + 1
                    )
                    return sub_v, active_cwd
                else:
                    idx += 1
            if idx < n:
                idx += 1  # skip lockfile argument
            continue
        if tok == "script":
            for i in range(idx + 1, n):
                sarg = tokens[i]
                if (sarg in ("-c", "--command") or (sarg.startswith("-") and not sarg.startswith("--") and "c" in sarg[1:])) and i + 1 < n:
                    sub_v = detect_bash_workspace_mutation(
                        tokens[i + 1], ws_root, out_dir, cwd=active_cwd, env=env, _depth=depth + 1
                    )
                    return sub_v, active_cwd
            return {
                "blocked": True,
                "reason": "Strict orchestrator mode blocked `script` typescript capture in workspace.",
            }, active_cwd
        if tok == "env":
            idx += 1
            while idx < n:
                etok = tokens[idx]
                if etok in ("-u", "--unset", "-C", "--chdir") and idx + 1 < n:
                    if etok in ("-C", "--chdir"):
                        active_cwd = (active_cwd / tokens[idx + 1]).resolve()
                    idx += 2
                elif etok in ("-S", "--split-string") and idx + 1 < n:
                    sub_v = detect_bash_workspace_mutation(
                        tokens[idx + 1], ws_root, out_dir, cwd=active_cwd, env=env, _depth=depth + 1
                    )
                    return sub_v, active_cwd
                elif etok.startswith("-"):
                    idx += 1
                elif "=" in etok and etok.split("=", 1)[0].isidentifier():
                    var_name, var_val = etok.split("=", 1)
                    if var_name in _BLOCKED_ENV_VARS:
                        if var_name == "PYTHONPATH" and _is_safe_workspace_pythonpath(
                            var_val, ws_root, out_dir, active_cwd, env
                        ):
                            idx += 1
                            continue
                        return {
                            "blocked": True,
                            "reason": f"Strict orchestrator mode blocked `env {var_name}=...`.",
                        }, active_cwd
                    idx += 1
                else:
                    break
            continue
        if tok == "sudo":
            idx += 1
            while idx < n:
                stok = tokens[idx]
                if stok in _SUDO_OPTS_WITH_VALUE and idx + 1 < n:
                    idx += 2
                elif stok.startswith("-"):
                    idx += 1
                else:
                    break
            continue
        if tok == "xargs":
            used_xargs = True
            idx += 1
            while idx < n:
                xtok = tokens[idx]
                if xtok in ("-I", "-L", "-n", "-P", "-s", "-a", "-d", "-E") and idx + 1 < n:
                    idx += 2
                elif xtok.startswith("-"):
                    idx += 1
                else:
                    break
            continue
        break

    if idx >= n:
        return {"blocked": False, "reason": None}, active_cwd

    raw_cmd_tok = tokens[idx]
    if raw_cmd_tok in (".", "source"):
        return {
            "blocked": True,
            "reason": f"Strict orchestrator mode blocked sourcing shell script via `{raw_cmd_tok}`.",
        }, active_cwd

    binary = Path(raw_cmd_tok).name
    raw_args = tokens[idx + 1 :]

    # Block any non-`ls` command from reading, searching, or archiving `out_dir` (.hmac_key / receipts) or `PLUGIN_ROOT`
    if binary != "ls":
        for arg in raw_args:
            if arg.startswith("-"):
                continue
            clean_arg = arg.split("=", 1)[-1] if "=" in arg else arg
            if "/" in clean_arg or clean_arg.startswith("."):
                try:
                    cand_p = Path(clean_arg).expanduser()
                    res_p = (active_cwd / cand_p).resolve() if not cand_p.is_absolute() else cand_p.resolve()
                    if ws_root != PLUGIN_ROOT and res_p.is_relative_to(PLUGIN_ROOT):
                        return {
                            "blocked": True,
                            "reason": (
                                f"Strict orchestrator mode blocked `{binary}` on cc-agy-plugin internal source {clean_arg!r}. "
                                "Use the MCP tools (`delegate_start`, `delegate_continue`, `delegate_apply`, `get_diff`) directly."
                            ),
                        }, active_cwd
                except OSError:
                    pass
                if classify_path(clean_arg, ws_root, out_dir, cwd=active_cwd, env=env) == "out_dir":
                    return {
                        "blocked": True,
                        "reason": (
                            f"Strict orchestrator mode blocked `{binary}` on internal receipt/key directory {clean_arg!r}. "
                            f"Use delegate_status or get_diff instead."
                        ),
                    }, active_cwd

    args = _strip_shell_redirects_from_args(raw_args)

    # Track directory changes (`cd <dir>`) across chained segments
    if binary in ("cd", "pushd"):
        non_flags = [a for a in args if not a.startswith("-")]
        if non_flags:
            active_cwd = (active_cwd / non_flags[0]).resolve()
        return {"blocked": False, "reason": None}, active_cwd

    if binary in ("export", "declare", "typeset"):
        for earg in args:
            if earg.startswith("-"):
                continue
            var_name = earg.split("=", 1)[0]
            if var_name in _BLOCKED_ENV_VARS:
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked `{binary} {var_name}`.",
                }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    # Unwrap nested shells (`sh -c`, `bash -c`, `zsh -c`, `eval`)
    if binary in ("sh", "bash", "zsh", "dash", "ksh"):
        if used_xargs:
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `xargs {binary}` shell execution.",
            }, active_cwd
        for i, arg in enumerate(args):
            if arg == "-c" or (arg.startswith("-") and "c" in arg[1:] and not arg.startswith("--")):
                if i + 1 < len(args):
                    if "{}" in args[i + 1]:
                        return {
                            "blocked": True,
                            "reason": f"Strict orchestrator mode blocked `{binary} -c` with placeholder substitution.",
                        }, active_cwd
                    sub_v = detect_bash_workspace_mutation(
                        args[i + 1], ws_root, out_dir, cwd=active_cwd, env=env, _depth=depth + 1
                    )
                    return sub_v, active_cwd
        if any(a in ("--version", "-V", "--help", "-h") for a in args):
            return {"blocked": False, "reason": None}, active_cwd
        non_flags = [a for a in args if not a.startswith("-")]
        if non_flags:
            return {
                "blocked": True,
                "reason": (
                    f"Strict orchestrator mode blocked shell script execution `{binary} {non_flags[0]}`. "
                    f"Delegate workspace modifications via delegate_start."
                ),
            }, active_cwd
        return {
            "blocked": True,
            "reason": (
                f"Strict orchestrator mode blocked `{binary}` reading commands from stdin/pipe. "
                f"Delegate workspace modifications via delegate_start."
            ),
        }, active_cwd

    if binary == "eval":
        joined = " ".join(args)
        if "$(" in joined or "`" in joined or "base64" in joined:
            return {
                "blocked": True,
                "reason": "Strict orchestrator mode blocked dynamic `eval` command.",
            }, active_cwd
        sub_v = detect_bash_workspace_mutation(
            joined, ws_root, out_dir, cwd=active_cwd, env=env, _depth=depth + 1
        )
        return sub_v, active_cwd

    if binary in _ALWAYS_BLOCKED_BINARIES:
        return {
            "blocked": True,
            "reason": f"Strict orchestrator mode blocked mutating binary `{binary}`.",
        }, active_cwd

    # Check sed / gsed in-place flags anywhere in the argument list
    if binary in ("sed", "gsed"):
        skip_sed_arg = False
        for sarg in args:
            if skip_sed_arg:
                skip_sed_arg = False
                continue
            if sarg == "--":
                break
            if sarg in ("-e", "-f", "--expression", "--file"):
                skip_sed_arg = True
                continue
            if (
                sarg in ("-i", "--in-place")
                or sarg.startswith(("-i", "--in-place="))
                or (sarg.startswith("-") and not sarg.startswith("--") and not sarg.startswith(("-e", "-f")) and "i" in sarg[1:])
            ):
                return {
                    "blocked": True,
                    "reason": "Strict orchestrator mode blocked `sed -i` in-place edit.",
                }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    # Check awk (`-i inplace`, `system(...)`, or internal output redirection `>`)
    if binary in ("awk", "gawk", "mawk", "nawk"):
        joined_awk = " ".join(args)
        if (
            "inplace" in args
            or any(a.startswith("-i") for a in args)
            or re.search(r"\bsystem\s*\(|>\s*['\"]?", joined_awk)
        ):
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked mutating `{binary}` command.",
            }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    # Check formatters: allow read-only `--check` / `--diff` / `-l`, block in-place formatting
    if binary in ("black", "isort", "gofmt", "rustfmt", "prettier"):
        has_check = any(a in ("--check", "--diff", "-c", "-l", "--list-different") for a in args)
        has_write = any(a in ("-w", "--write") for a in args)
        if not has_check or has_write:
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked in-place formatter `{binary}` (use `--check` for read-only verification).",
            }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    # Check ruff / npm / npx / make / pytest
    if binary == "ruff":
        is_mutating_ruff = any(a in ("--fix", "--unsafe-fixes") for a in args) or (
            "format" in args and not any(a in ("--check", "--diff") for a in args)
        )
        if is_mutating_ruff:
            return {
                "blocked": True,
                "reason": "Strict orchestrator mode blocked `ruff` mutating/formatting flags.",
            }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    if binary in ("npm", "pnpm", "yarn", "bun"):
        blocked_js_flags = (
            "-u",
            "--update",
            "--updateSnapshot",
            "--update-snapshots",
            "--snapshot-update",
            "--fix",
            "--unsafe-fixes",
            "--write",
            "--force-regen",
            "--regen-all",
            "--prefix",
            "--cwd",
            "-C",
        )
        if any(a in blocked_js_flags or a.startswith(("--prefix=", "--cwd=", "--fix=")) for a in args):
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `{binary} {' '.join(args[:3])}`.",
            }, active_cwd
        pre_dd_pos: list[str] = []
        skip_next = False
        for a in args:
            if a == "--":
                break
            if skip_next:
                skip_next = False
                continue
            if a in ("--filter", "-F", "--workspace", "-w", "--loglevel", "--reporter", "--config", "-t", "--grep", "--testNamePattern"):
                skip_next = True
                continue
            if a.startswith("-"):
                continue
            pre_dd_pos.append(a)
        subcmd = pre_dd_pos[0] if pre_dd_pos else ""
        subcmd_arg = pre_dd_pos[1] if len(pre_dd_pos) > 1 else ""
        blocked_js_subcmds = (
            "install",
            "i",
            "add",
            "remove",
            "rm",
            "uninstall",
            "unlink",
            "update",
            "up",
            "upgrade",
            "ci",
            "link",
            "audit",
            "fmt",
            "format",
            "init",
            "create",
            "pm",
            "x",
            "exec",
            "dlx",
            "patch",
            "patch-commit",
            "publish",
            "pack",
            "version",
            "rebuild",
            "dedupe",
            "prune",
            "shrinkwrap",
            "import",
            "set",
            "unset",
            "delete",
            "unplug",
            "stage",
            "deploy",
            "fetch",
        )
        mutating_script_suffixes = (":fix", ":write", ":format", "-fix", "-write", "-format")
        if (
            not subcmd
            or subcmd in blocked_js_subcmds
            or subcmd.endswith(mutating_script_suffixes)
        ):
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `{binary} {' '.join(args[:3])}`.",
            }, active_cwd
        if subcmd == "pkg" and subcmd_arg != "get":
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `{binary} pkg {subcmd_arg}`.",
            }, active_cwd
        if subcmd == "plugin" and subcmd_arg != "list":
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `{binary} plugin {subcmd_arg}`.",
            }, active_cwd
        if subcmd in ("config", "c") and subcmd_arg not in ("get", "list", "ls"):
            allow_user_npm_cfg = (
                binary == "npm"
                and subcmd_arg in ("set", "delete")
                and any(a in ("--location=user", "--global", "-g") for a in args)
            )
            if not allow_user_npm_cfg:
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked `{binary} {subcmd} {subcmd_arg}`.",
                }, active_cwd
        if subcmd in ("run", "run-script"):
            script_name = subcmd_arg
            if (
                not script_name
                or script_name in ("fmt", "format", "fix", "write", "codegen", "generate", "migrate", "seed", "deploy", "publish", "release", "clean")
                or script_name.endswith(mutating_script_suffixes)
            ):
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked `{binary} {' '.join(args[:3])}`.",
                }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    if binary in ("npx", "bunx", "pnpx"):
        # Allow read-only local test/typecheck/lint runners while blocking remote package installs and snapshot writes.
        if any(
            a
            in (
                "-y",
                "--yes",
                "-p",
                "--package",
                "-c",
                "--call",
                "-u",
                "--update",
                "--updateSnapshot",
                "--update-snapshots",
                "--snapshot-update",
                "--fix",
                "--unsafe-fixes",
                "--write",
                "--force-regen",
                "--regen-all",
            )
            or a.startswith(("--package=", "--yes=", "--fix="))
            for a in args
        ):
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked mutating or remote-install `{binary}` flags: `{binary} {' '.join(args[:3])}`.",
            }, active_cwd
        npx_non_flags = [a for a in args if a != "--no-install" and not a.startswith("-")]
        runner_tok = npx_non_flags[0] if npx_non_flags else ""
        if runner_tok in ("vitest", "jest", "mocha", "ava"):
            return {"blocked": False, "reason": None}, active_cwd
        if runner_tok == "tsc" and "--noEmit" in args:
            return {"blocked": False, "reason": None}, active_cwd
        if runner_tok == "eslint":
            return {"blocked": False, "reason": None}, active_cwd
        return {
            "blocked": True,
            "reason": f"Strict orchestrator mode blocked `{binary} {' '.join(args[:2])}`.",
        }, active_cwd

    if binary == "cargo":
        cargo_non_flags = [a for a in args if not a.startswith("-") and not a.startswith("+")]
        cargo_sub = cargo_non_flags[0] if cargo_non_flags else ""
        if cargo_sub == "fmt":
            if "--check" not in args:
                return {
                    "blocked": True,
                    "reason": "Strict orchestrator mode blocked `cargo fmt` without `--check`.",
                }, active_cwd
            return {"blocked": False, "reason": None}, active_cwd
        if cargo_sub == "clippy" and "--fix" in args:
            return {
                "blocked": True,
                "reason": "Strict orchestrator mode blocked `cargo clippy --fix`.",
            }, active_cwd
        if cargo_sub == "insta":
            insta_sub = cargo_non_flags[1] if len(cargo_non_flags) > 1 else ""
            if insta_sub != "test" or any(
                a in ("--accept", "--review", "--force-update-snapshots")
                or a.startswith("--unreferenced=")
                for a in args
            ):
                return {
                    "blocked": True,
                    "reason": "Strict orchestrator mode blocked mutating `cargo insta` snapshot command.",
                }, active_cwd
            return {"blocked": False, "reason": None}, active_cwd
        if cargo_sub in ("fix", "add", "remove", "rm", "init", "new", "install", "uninstall", "clean", "update", "vendor", "publish"):
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked mutating `cargo {cargo_sub}` command.",
            }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    if binary == "go":
        idx_g = 0
        go_sub = ""
        while idx_g < len(args):
            ga = args[idx_g]
            if ga in ("-C", "--C") and idx_g + 1 < len(args):
                c_dir = _expand_shell_vars_in_path(args[idx_g + 1], active_cwd, ws_root)
                c_cand = Path(c_dir)
                active_cwd = (active_cwd / c_cand).resolve() if not c_cand.is_absolute() else c_cand.resolve()
                idx_g += 2
                continue
            if ga.startswith(("-C=", "--C=")):
                c_dir = _expand_shell_vars_in_path(ga.split("=", 1)[1], active_cwd, ws_root)
                c_cand = Path(c_dir)
                active_cwd = (active_cwd / c_cand).resolve() if not c_cand.is_absolute() else c_cand.resolve()
                idx_g += 1
                continue
            if ga.startswith("-"):
                idx_g += 1
                continue
            go_sub = ga
            break
        if go_sub in ("fmt", "generate", "get", "install", "clean", "mod", "work"):
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked mutating `go {go_sub}` command.",
            }, active_cwd
        if go_sub in ("test", "build"):
            go_sub_args = args[idx_g + 1 :]
            if go_sub == "test" and any(
                a in ("-update", "--update", "-golden", "--golden", "-u")
                or a.startswith(("-update=", "--update=", "-golden=", "--golden="))
                for a in go_sub_args
            ):
                return {
                    "blocked": True,
                    "reason": "Strict orchestrator mode blocked `go test` golden/snapshot update flag.",
                }, active_cwd
            has_explicit_go_out = False
            for i_g, garg in enumerate(go_sub_args):
                out_dest: str | None = None
                if garg in ("-o", "--o", "-coverprofile", "--coverprofile") and i_g + 1 < len(go_sub_args):
                    out_dest = go_sub_args[i_g + 1]
                    if garg in ("-o", "--o"):
                        has_explicit_go_out = True
                elif garg.startswith(("-o=", "--o=", "-coverprofile=", "--coverprofile=")):
                    out_dest = garg.split("=", 1)[1]
                    if garg.startswith(("-o=", "--o=")):
                        has_explicit_go_out = True
                if out_dest and out_dest != "/dev/null":
                    if classify_path(out_dest, ws_root, out_dir, cwd=active_cwd, env=env) in ("workspace", "out_dir"):
                        return {
                            "blocked": True,
                            "reason": f"Strict orchestrator mode blocked `go {go_sub}` writing output into workspace ({out_dest!r}).",
                        }, active_cwd
            if go_sub == "build" and not has_explicit_go_out:
                return {
                    "blocked": True,
                    "reason": "Strict orchestrator mode blocked `go build` without `-o /dev/null` or `-o /tmp/...` (use `go test` or `go vet`).",
                }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    if binary in ("mvn", "mvnw", "gradle", "gradlew"):
        bad_jvm_goals = ("clean", "deploy", "install", "spotless:apply", "spotlessApply", "rewrite:run", "rewriteRun")
        for jarg in args:
            if jarg in bad_jvm_goals or jarg.startswith("release:"):
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked mutating `{binary} {jarg}` goal.",
                }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    if binary == "dotnet":
        dn_non_flags = [a for a in args if not a.startswith("-")]
        dn_sub = dn_non_flags[0] if dn_non_flags else ""
        if dn_sub == "format" and "--verify-no-changes" not in args:
            return {
                "blocked": True,
                "reason": "Strict orchestrator mode blocked `dotnet format` without `--verify-no-changes`.",
            }, active_cwd
        if dn_sub in ("new", "add", "remove", "clean", "publish", "pack", "workload", "tool"):
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked mutating `dotnet {dn_sub}` command.",
            }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    if binary == "make":
        if any(a in ("-f", "--file", "--makefile", "-C", "--directory") or a.startswith("--file=") for a in args):
            return {
                "blocked": True,
                "reason": "Strict orchestrator mode blocked custom `make -f/-C` execution.",
            }, active_cwd
        non_flags = [a for a in args if not a.startswith("-")]
        if non_flags and not all(t in ("test", "check", "lint") for t in non_flags):
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked non-test make target `{non_flags[0]}`.",
            }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    if binary in ("pytest", "py.test") or bool(re.match(r"^(?:pytest|py\.test)(?:-\d+(?:\.\d+)?)?$", binary)):
        v_pytest = _validate_pytest_args(
            args, ws_root=ws_root, out_dir=out_dir, active_cwd=active_cwd, env=env
        )
        return v_pytest, active_cwd

    # Check dd output file `of=<path>`
    if binary == "dd":
        for arg in args:
            if arg.startswith("of="):
                target = arg[len("of=") :].strip("'\"")
                cls = classify_path(target, ws_root, out_dir, cwd=active_cwd, env=env)
                if cls in ("workspace", "out_dir"):
                    return {
                        "blocked": True,
                        "reason": f"Strict orchestrator mode blocked `dd of={target}` targeting {cls}.",
                    }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    # Check python / node / ruby / perl / php / lua inline scripts and script files
    if binary.startswith(("python", "pypy")) or binary in ("node", "ruby", "perl", "php", "lua"):
        if used_xargs:
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `xargs {binary}` execution.",
            }, active_cwd
        if binary == "node" and "--test" in args:
            blocked_node_test_flags = (
                "-e",
                "--eval",
                "-c",
                "--check",
                "-r",
                "--require",
                "--import",
                "--loader",
                "--experimental-loader",
                "--test-update-snapshots",
            )
            if any(
                a in blocked_node_test_flags
                or a.startswith(("--eval=", "--require=", "--import=", "--loader=", "--experimental-loader="))
                for a in args
            ):
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked mutating or custom-loader `node --test` flag: `node {' '.join(args[:3])}`.",
                }, active_cwd
            for narg in args:
                if narg.startswith("-"):
                    continue
                if classify_path(narg, ws_root, out_dir, cwd=active_cwd, env=env) != "workspace":
                    return {
                        "blocked": True,
                        "reason": f"Strict orchestrator mode blocked `node --test` targeting non-workspace path {narg!r}.",
                    }, active_cwd
            return {"blocked": False, "reason": None}, active_cwd

        for i, arg in enumerate(args):
            inline_code: str | None = None
            if (
                arg in ("-c", "-e", "-r", "--eval")
                or (arg.startswith("-") and not arg.startswith("--") and len(arg) > 2 and arg[1:-1].isalpha() and arg[-1] in ("c", "e"))
            ) and i + 1 < len(args):
                inline_code = args[i + 1]
            elif (
                arg.startswith(("-c", "-e", "-r"))
                and not arg.startswith("--")
                and len(arg) > 2
            ):
                inline_code = arg[2:]
            elif arg.startswith("--eval="):
                inline_code = arg.split("=", 1)[1]

            if inline_code is not None:
                if re.search(
                    r"\b(?:from\s+server\.(?:mcp_stdio|jobs|agy_runner|receipts)\s+import|import\s+server\.(?:mcp_stdio|jobs|agy_runner|receipts)|from\s+server\s+import\s+(?:mcp_stdio|jobs|agy_runner|receipts)|MCPServer\s*\(|JobManager\s*\()",
                    inline_code,
                ):
                    return {
                        "blocked": True,
                        "reason": (
                            "Strict orchestrator mode blocked direct Python import/invocation of plugin internals "
                            "(`MCPServer` / `JobManager`). Use the MCP tools (`delegate_start`, `delegate_continue`, "
                            "`delegate_apply`, `get_diff`) directly."
                        ),
                    }, active_cwd
                ws_roots_for_inline: Path | tuple[Path, ...] | None = None
                if allow_workspace_inline_imports:
                    try:
                        if active_cwd.resolve().is_relative_to(ws_root.resolve()):
                            ws_roots_for_inline = (ws_root, active_cwd)
                        else:
                            ws_roots_for_inline = (ws_root,)
                    except Exception:
                        ws_roots_for_inline = (ws_root,)
                if binary in ("node", "ruby", "perl", "php", "lua") or not _is_safe_python_inline_ast(
                    inline_code, ws_root=ws_roots_for_inline
                ):
                    return {
                        "blocked": True,
                        "reason": (
                            f"Strict orchestrator mode blocked `{binary} {arg}` containing file/system mutation. "
                            f"Delegate workspace edits via delegate_start or delegate_to_gemini."
                        ),
                    }, active_cwd
                return {"blocked": False, "reason": None}, active_cwd

            mod: str | None = None
            mod_extra_args: list[str] = []
            if arg == "-m" and i + 1 < len(args):
                mod = args[i + 1]
                mod_extra_args = args[i + 2 :]
            elif arg.startswith("-m") and not arg.startswith("--") and len(arg) > 2:
                mod = arg[2:]
                mod_extra_args = args[i + 1 :]

            if mod is not None:
                if mod == "ruff":
                    is_mutating_ruff = any(a in ("--fix", "--unsafe-fixes") for a in mod_extra_args) or (
                        "format" in mod_extra_args and not any(a in ("--check", "--diff") for a in mod_extra_args)
                    )
                    if is_mutating_ruff:
                        return {
                            "blocked": True,
                            "reason": "Strict orchestrator mode blocked `python -m ruff` mutating/formatting flags.",
                        }, active_cwd
                    return {"blocked": False, "reason": None}, active_cwd
                if mod in ("pytest", "py.test"):
                    v_py = _validate_pytest_args(
                        mod_extra_args, ws_root=ws_root, out_dir=out_dir, active_cwd=active_cwd, env=env
                    )
                    return v_py, active_cwd
                if mod in ("unittest", "mypy", "flake8", "pylint", "json.tool", "py_compile", "compileall"):
                    for mod_arg in mod_extra_args:
                        if mod_arg.startswith("-"):
                            continue
                        if mod_arg in ("discover", "tests"):
                            continue
                        target_file = mod_arg.split("::", 1)[0]
                        if "/" in target_file or target_file.endswith(".py"):
                            cls = classify_path(target_file, ws_root, out_dir, cwd=active_cwd, env=env)
                            if cls != "workspace":
                                return {
                                    "blocked": True,
                                    "reason": (
                                        f"Strict orchestrator mode blocked `{binary} -m {mod}` on non-workspace path {mod_arg!r} ({cls})."
                                    ),
                                }, active_cwd
                    return {"blocked": False, "reason": None}, active_cwd
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked `{binary} -m {mod}`.",
                }, active_cwd
        non_flags = [a for a in args if not a.startswith("-") or a == "-"]
        if non_flags:
            script_arg = non_flags[0]
            script_name = Path(script_arg).name
            script_cls = classify_path(script_arg, ws_root, out_dir, cwd=active_cwd, env=env)
            script_cand = Path(script_arg)
            cwd_resolved = active_cwd.resolve()
            ws_resolved = ws_root.resolve()
            script_resolved = (
                (cwd_resolved / script_cand).resolve()
                if not script_cand.is_absolute()
                else script_cand.resolve()
            )
            script_inside_ws = (
                not any(ch in script_arg for ch in ("$", "`", "~"))
                and ".." not in script_cand.parts
                and script_resolved.is_relative_to(ws_resolved)
                and script_resolved.is_relative_to(cwd_resolved)
            )
            script_rel = (
                str(script_resolved.relative_to(cwd_resolved)).replace("\\", "/")
                if script_inside_ws
                else ""
            )
            norm_script_arg = script_arg.replace("\\", "/")
            while norm_script_arg.startswith("./"):
                norm_script_arg = norm_script_arg[2:]
            if script_inside_ws and script_rel:
                norm_script_arg = script_rel
            tail_args = args[1:]
            has_node_mutating_flag = any(
                a in _NODE_MUTATING_TEST_FLAGS or a.startswith(_NODE_MUTATING_TEST_FLAG_PREFIXES)
                for a in tail_args
            )
            is_js_node_test_runner = (
                binary in ("node", "nodejs", "bun")
                and not any(ch in script_arg for ch in ("$", "`", "~"))
                and ".." not in script_cand.parts
                and (script_inside_ws or not script_cand.is_absolute())
                and not has_node_mutating_flag
                and (
                    bool(_NODE_TEST_ENTRYPOINT_RE.match(norm_script_arg))
                    or (
                        bool(_NODE_TSC_ENTRYPOINT_RE.match(norm_script_arg))
                        and "--noEmit" in tail_args
                    )
                )
            )
            is_allowed_test_runner = is_js_node_test_runner or (
                script_cls == "workspace"
                and script_inside_ws
                and (
                    (script_name == "manage.py" and len(non_flags) >= 2 and non_flags[1] == "test")
                    or (binary == "php" and script_name == "artisan" and len(non_flags) >= 2 and non_flags[1] == "test")
                    or (
                        binary == "ruby"
                        and script_name.endswith("_test.rb")
                        and not any(a.startswith(("-i", "-r")) for a in args)
                    )
                    or bool(re.match(r"^(?:run_tests\.py|runtests\.py|test_.*\.py|.*_test\.py)$", script_name))
                    or script_rel in ("bin/test", "bin/doctest", "tests/runtests.py", "runtests.py", "run_tests.py")
                )
            )
            if is_allowed_test_runner:
                return {"blocked": False, "reason": None}, active_cwd
            return {
                "blocked": True,
                "reason": (
                    f"Strict orchestrator mode blocked executing script `{binary} {script_arg}`. "
                    f"Use a read-only test runner (pytest, npm/pnpm/yarn/bun test, node --test, go test, cargo test, ./gradlew test, ./mvnw test) "
                    f"or delegate workspace changes via delegate_start."
                ),
            }, active_cwd
        if any(a in ("--version", "-V", "--help", "-h") for a in args):
            return {"blocked": False, "reason": None}, active_cwd
        return {
            "blocked": True,
            "reason": (
                f"Strict orchestrator mode blocked `{binary}` reading script from stdin/pipe. "
                f"Use a read-only test runner (pytest, npm/pnpm/yarn/bun test, node --test, go test, cargo test, ./gradlew test, ./mvnw test) "
                f"or delegate workspace changes via delegate_start."
            ),
        }, active_cwd

    # Check git subcommands (skipping -C / --git-dir / -c options)
    if binary == "git":
        sub, sub_args, git_cwd, cfg_err = _extract_git_subcommand_and_workdir(args, active_cwd)
        if cfg_err is not None:
            return {"blocked": True, "reason": cfg_err}, active_cwd
        if sub is None:
            return {"blocked": False, "reason": None}, active_cwd
        if sub == "branch":
            readonly_branch_flags = {
                "-a",
                "--all",
                "-r",
                "--remotes",
                "-v",
                "-vv",
                "--list",
                "--show-current",
                "--contains",
                "--merged",
                "--no-merged",
            }
            if all(a in readonly_branch_flags for a in sub_args):
                return {"blocked": False, "reason": None}, active_cwd
            return {
                "blocked": True,
                "reason": "Strict orchestrator mode blocked mutating `git branch` command.",
            }, active_cwd
        if sub == "stash":
            if sub_args and sub_args[0] in ("list", "show"):
                return {"blocked": False, "reason": None}, active_cwd
            return {
                "blocked": True,
                "reason": "Strict orchestrator mode blocked mutating `git stash` command.",
            }, active_cwd
        if sub == "remote":
            if not sub_args or sub_args == ["-v"] or sub_args[0] == "show":
                return {"blocked": False, "reason": None}, active_cwd
            return {
                "blocked": True,
                "reason": "Strict orchestrator mode blocked mutating `git remote` command.",
            }, active_cwd
        if sub == "tag":
            if not sub_args or all(a in ("-l", "--list") for a in sub_args):
                return {"blocked": False, "reason": None}, active_cwd
            return {
                "blocked": True,
                "reason": "Strict orchestrator mode blocked mutating `git tag` command.",
            }, active_cwd
        if sub not in _GIT_READONLY_SUBCOMMANDS:
            return {
                "blocked": True,
                "reason": (
                    f"Strict orchestrator mode blocked mutating git command `git {sub}`. "
                    f"Use delegate_apply to apply verified worker diffs."
                ),
            }, active_cwd
        # Even in git diff/show/log, block external diff/textconv helpers and --output targeting workspace/out_dir
        for s_idx, sarg in enumerate(sub_args):
            if sarg == "--":
                break
            if sarg in ("--ext-diff", "--textconv") or sarg.startswith(("--ext-diff=", "--textconv=")):
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked `git {sub} {sarg}`.",
                }, active_cwd
            if sarg == "--output" and s_idx + 1 < len(sub_args):
                out_t = sub_args[s_idx + 1]
                cls = classify_path(out_t, ws_root, out_dir, cwd=git_cwd, env=env)
                if cls in ("workspace", "out_dir"):
                    return {
                        "blocked": True,
                        "reason": f"Strict orchestrator mode blocked `git {sub} --output {out_t}`.",
                    }, active_cwd
            if sarg.startswith("--output="):
                out_t = sarg.split("=", 1)[1]
                cls = classify_path(out_t, ws_root, out_dir, cwd=git_cwd, env=env)
                if cls in ("workspace", "out_dir"):
                    return {
                        "blocked": True,
                        "reason": f"Strict orchestrator mode blocked `git {sub} {sarg}`.",
                    }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    # Check find (-delete, -exec, -execdir, -ok)
    if binary == "find":
        for farg in args:
            if farg in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fls", "-fprint", "-fprint0", "-fprintf"):
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked `find {farg}`.",
                }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    # Check curl / wget file outputs
    if binary == "curl":
        for i, carg in enumerate(args):
            if carg in ("-O", "--remote-name", "--remote-name-all", "-J", "--remote-header-name"):
                cls = classify_path(".", ws_root, out_dir, cwd=active_cwd, env=env)
                if cls in ("workspace", "out_dir"):
                    return {
                        "blocked": True,
                        "reason": f"Strict orchestrator mode blocked `curl {carg}` in {cls}.",
                    }, active_cwd
            if (
                carg in ("-o", "--output")
                or (carg.startswith("-") and not carg.startswith("--") and carg.endswith("o"))
            ) and i + 1 < len(args):
                cls = classify_path(args[i + 1], ws_root, out_dir, cwd=active_cwd, env=env)
                if cls in ("workspace", "out_dir"):
                    return {
                        "blocked": True,
                        "reason": f"Strict orchestrator mode blocked `curl {carg} {args[i + 1]}`.",
                    }, active_cwd
            if (carg.startswith("-o") and len(carg) > 2) or carg.startswith("--output="):
                target = carg[2:] if carg.startswith("-o") else carg.split("=", 1)[1]
                cls = classify_path(target, ws_root, out_dir, cwd=active_cwd, env=env)
                if cls in ("workspace", "out_dir"):
                    return {
                        "blocked": True,
                        "reason": f"Strict orchestrator mode blocked `curl {carg}`.",
                    }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    if binary == "wget":
        dest_targets: list[str] = []
        for i, warg in enumerate(args):
            if warg in ("-O", "--output-document", "-P", "--directory-prefix") and i + 1 < len(args):
                dest_targets.append(args[i + 1])
            elif warg.startswith(("--output-document=", "--directory-prefix=")):
                dest_targets.append(warg.split("=", 1)[1])
            elif warg.startswith(("-O", "-P")) and len(warg) > 2:
                dest_targets.append(warg[2:])
        if not dest_targets:
            dest_targets.append(".")
        for target in dest_targets:
            if target == "-":
                continue
            cls = classify_path(target, ws_root, out_dir, cwd=active_cwd, env=env)
            if cls in ("workspace", "out_dir"):
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked `wget` writing to {cls} ({target!r}).",
                }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    # Check tar extraction
    if binary == "tar":
        is_extract = False
        target_dir = "."
        for i, targ in enumerate(args):
            if targ in ("-x", "--extract", "--get") or (
                not targ.startswith("--") and "x" in targ.lstrip("-") and i == 0
            ) or (targ.startswith("-") and not targ.startswith("--") and "x" in targ[1:]):
                is_extract = True
            if targ in ("-C", "--directory") and i + 1 < len(args):
                target_dir = args[i + 1]
            elif targ.startswith("--directory="):
                target_dir = targ.split("=", 1)[1]
        if is_extract:
            cls = classify_path(target_dir, ws_root, out_dir, cwd=active_cwd, env=env)
            if cls in ("workspace", "out_dir"):
                return {
                    "blocked": True,
                    "reason": f"Strict orchestrator mode blocked `tar` extraction into {cls} ({target_dir!r}).",
                }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    # Check standard file-mutating binaries (cp, mv, rm, touch, install, rsync, ln, truncate, chmod, etc.)
    if binary in _MUTATING_BINARIES:
        target_dir_opt: str | None = None
        positional: list[str] = []
        skip_next = False
        for i, marg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if marg in ("-t", "--target-directory") and i + 1 < len(args):
                target_dir_opt = args[i + 1]
                skip_next = True
                continue
            if marg.startswith("--target-directory="):
                target_dir_opt = marg.split("=", 1)[1]
                continue
            if marg.startswith("-t") and len(marg) > 2 and not marg.startswith("--"):
                target_dir_opt = marg[2:]
                continue
            if marg in ("-m", "-o", "-g", "-S", "--suffix") and binary == "install" and i + 1 < len(args):
                skip_next = True
                continue
            if marg in ("-s", "--size", "-r", "--reference") and binary == "truncate" and i + 1 < len(args):
                skip_next = True
                continue
            if marg.startswith("-"):
                continue
            positional.append(marg)

        candidate_targets: list[str] = []
        if target_dir_opt is not None:
            candidate_targets.append(target_dir_opt)
            if binary == "mv":
                candidate_targets.extend(positional)
        elif binary in ("cp", "install", "rsync", "ln"):
            if len(positional) >= 2:
                candidate_targets.append(positional[-1])
            elif len(positional) == 1:
                candidate_targets.append(".")
        else:
            candidate_targets.extend(positional)

        if used_xargs or not candidate_targets:
            return {
                "blocked": True,
                "reason": (
                    f"Strict orchestrator mode blocked `{binary}` (piped/dynamic arguments). "
                    f"Delegate workspace file changes via delegate_start or delegate_to_gemini."
                ),
            }, active_cwd

        for target in candidate_targets:
            classification = classify_path(target, ws_root, out_dir, cwd=active_cwd, env=env)
            if classification in ("workspace", "out_dir"):
                return {
                    "blocked": True,
                    "reason": (
                        f"Strict orchestrator mode blocked `{binary}` targeting {classification} path {target!r}. "
                        f"Delegate workspace file changes via delegate_start or delegate_to_gemini."
                    ),
                }, active_cwd

    return {"blocked": False, "reason": None}, active_cwd


_BASH_READ_BINARIES: frozenset[str] = frozenset(
    {
        "cat",
        "head",
        "tail",
        "nl",
        "less",
        "more",
        "bat",
        "sed",
        "awk",
        "gawk",
        "mawk",
        "nawk",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "ripgrep",
        "ack",
        "ag",
        "cut",
        "od",
        "wc",
        "sort",
        "uniq",
        "diff",
        "comm",
        "paste",
        "join",
        "pr",
        "fmt",
        "fold",
        "expand",
        "unexpand",
        "tac",
        "rev",
        "strings",
        "hexdump",
        "xxd",
        "jq",
        "yq",
    }
)


def _expand_brace_patterns(arg_str: str, max_depth: int = 8) -> list[str]:
    cleaned = re.sub(r"\{,+\}", "", arg_str)
    if max_depth <= 0 or "{" not in cleaned or "}" not in cleaned:
        return [cleaned]
    m = re.search(r"\{([^{}]*,[^{}]*)\}", cleaned)
    if not m:
        return [cleaned]
    prefix = cleaned[: m.start()]
    suffix = cleaned[m.end() :]
    out: list[str] = []
    choices = list(dict.fromkeys(m.group(1).split(",")))[:16]
    for choice in choices:
        for exp in _expand_brace_patterns(f"{prefix}{choice}{suffix}", max_depth - 1):
            if exp not in out:
                out.append(exp)
                if len(out) >= 64:
                    return out
    return out


def _resolve_bash_read_arg(
    raw_arg: str,
    active_cwd: Path,
    ws_root: Path,
) -> tuple[list[str], bool]:
    """Expand braces/globs in `raw_arg` and return `(workspace_rel_files, saw_workspace_path)`."""
    import glob as _glob

    clean_arg = raw_arg.strip()
    if not clean_arg or clean_arg in ("-", "--", "/dev/null", "{}"):
        return [], False
    files_out: list[str] = []
    saw_ws = False
    for expanded in _expand_brace_patterns(clean_arg):
        if any(ch in expanded for ch in "*?["):
            pat_path = Path(expanded).expanduser()
            glob_pat = str((active_cwd / pat_path) if not pat_path.is_absolute() else pat_path)
            try:
                matches = sorted(_glob.glob(glob_pat, recursive=True))
            except Exception:
                matches = []
            for m_str in matches[:50]:
                try:
                    res_p = Path(m_str).resolve()
                except OSError:
                    continue
                if res_p.is_relative_to(ws_root) and res_p.exists():
                    saw_ws = True
                    if res_p.is_file():
                        rel_p = res_p.relative_to(ws_root).as_posix()
                        if rel_p not in files_out:
                            files_out.append(rel_p)
        else:
            try:
                cand_p = Path(expanded).expanduser()
                res_p = (active_cwd / cand_p).resolve() if not cand_p.is_absolute() else cand_p.resolve()
            except OSError:
                continue
            if res_p.is_relative_to(ws_root) and res_p.exists():
                saw_ws = True
                if res_p.is_file():
                    rel_p = res_p.relative_to(ws_root).as_posix()
                    if rel_p not in files_out:
                        files_out.append(rel_p)
    return files_out, saw_ws


def detect_bash_workspace_read(
    command: str,
    workdir: str | Path,
    out_dir: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    _depth: int = 0,
) -> list[str]:
    """Identify read-only Bash commands that inspect workspace file contents or diffs.

    Returns a non-empty list of workspace-relative file paths (or `["<workspace-search>"]`)
    when `command` reads or searches repository files via `cat`, `sed`, `head`, `tail`, `nl`,
    `awk`, `grep`, `rg`, `cut`, `od`, `find -exec`, subshells, redirects, or `git diff`/`show`/`log`/`cat-file`,
    so `evaluate_pre_tool_use` counts each read file against `.lead_read_count` on par with `Read`/`Grep`/`Glob`.
    """
    if not command or not command.strip() or _depth > 12:
        return []
    ws_root = Path(workdir).resolve()
    active_cwd = Path(cwd).resolve() if cwd else ws_root
    cmd_without_herestrings = re.sub(r"<<<", "   ", command)
    cmd_without_heredocs = re.sub(r"<<-?\s*['\"]?\w+['\"]?", "   ", cmd_without_herestrings)
    matched_targets: list[str] = []
    saw_read_cmd = False

    def _add_targets(new_files: list[str], saw_ws: bool = False) -> None:
        nonlocal saw_read_cmd
        if saw_ws or new_files:
            saw_read_cmd = True
        for f in new_files:
            if f == "<workspace-search>":
                saw_read_cmd = True
            elif f not in matched_targets:
                matched_targets.append(f)

    # 1. Inspect command substitutions `$(...)` and backticks `` `...` ``
    for sub_m in re.finditer(r"\$\(([^()]+)\)|`([^`]+)`", cmd_without_heredocs):
        inner_cmd = sub_m.group(1) or sub_m.group(2) or ""
        if inner_cmd.strip():
            sub_res = detect_bash_workspace_read(
                inner_cmd, ws_root, out_dir, cwd=active_cwd, env=env, _depth=_depth + 1
            )
            _add_targets(sub_res)

    # 2. Inspect input redirections `< file` (excluding heredocs/herestrings/fd-dups)
    for redir_m in re.finditer(r"(?<![<>])<\s*(?![<&>])(\"[^\"]+\"|'[^']+'|[^\s;|&><()\"']+)", cmd_without_heredocs):
        raw_redir = redir_m.group(1).strip("\"'")
        exp_redir = _expand_shell_vars_in_path(raw_redir, active_cwd, ws_root)
        r_files, r_saw = _resolve_bash_read_arg(exp_redir, active_cwd, ws_root)
        _add_targets(r_files, r_saw)

    segments = _split_shell_segments(cmd_without_heredocs)
    for raw_seg in segments:
        seg = raw_seg.strip()
        while seg and (seg[0] in "({" or seg[-1] in ")}"):
            seg = seg.lstrip("({ \t").rstrip(")} \t")
        if not seg:
            continue
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if not tokens:
            continue
        tokens = [_expand_shell_vars_in_path(t, active_cwd, ws_root) for t in tokens]
        idx = 0
        n = len(tokens)
        while idx < n:
            tok = Path(tokens[idx]).name if tokens[idx].startswith("/") else tokens[idx]
            if "=" in tok and not tok.startswith("-") and tok.split("=", 1)[0].isidentifier():
                idx += 1
                continue
            if tok in ("command", "nohup", "time", "exec", "builtin", "setsid", "busybox", "env"):
                idx += 1
                while idx < n and (
                    tokens[idx].startswith("-")
                    or ("=" in tokens[idx] and tokens[idx].split("=", 1)[0].isidentifier())
                ):
                    idx += 1
                continue
            if tok == "timeout":
                idx += 1
                while idx < n and tokens[idx].startswith("-"):
                    flag_t = tokens[idx]
                    idx += 1
                    if flag_t in ("-k", "-s", "--kill-after", "--signal") and idx < n:
                        idx += 1
                if idx < n and re.match(r"^\d+(?:\.\d+)?[smhd]?$", tokens[idx]):
                    idx += 1
                continue
            if tok in ("nice", "ionice", "chrt", "stdbuf", "watch"):
                idx += 1
                while idx < n and tokens[idx].startswith("-"):
                    flag_t = tokens[idx]
                    idx += 1
                    if flag_t in ("-n", "-c", "-p") and idx < n and re.match(r"^-?\d+$", tokens[idx]):
                        idx += 1
                continue
            if tok == "xargs":
                idx += 1
                while idx < n and tokens[idx].startswith("-"):
                    flag_t = tokens[idx]
                    idx += 1
                    if flag_t in ("-n", "-P", "-I", "-L", "-s", "-d", "-a", "--max-args", "--max-procs") and idx < n:
                        idx += 1
                continue
            break
        if idx >= n:
            continue
        binary = Path(tokens[idx]).name
        args = tokens[idx + 1 :]
        if binary == "cd":
            if args:
                cand_cd = Path(args[0]).expanduser()
                active_cwd = (active_cwd / cand_cd).resolve() if not cand_cd.is_absolute() else cand_cd.resolve()
            continue

        if binary in ("bash", "sh", "zsh", "dash", "ksh") and "-c" in args:
            c_idx = args.index("-c")
            if c_idx + 1 < len(args):
                sub_res = detect_bash_workspace_read(
                    args[c_idx + 1], ws_root, out_dir, cwd=active_cwd, env=env, _depth=_depth + 1
                )
                _add_targets(sub_res)
            continue

        if binary == "eval" and args:
            sub_res = detect_bash_workspace_read(
                " ".join(args), ws_root, out_dir, cwd=active_cwd, env=env, _depth=_depth + 1
            )
            _add_targets(sub_res)
            continue

        if binary == "find":
            search_roots: list[Path] = []
            name_pats: list[str] = []
            has_read_exec = False
            i_f = 0
            while i_f < len(args):
                a_f = args[i_f]
                if a_f in ("-name", "-iname", "-path", "-wholename") and i_f + 1 < len(args):
                    name_pats.append(args[i_f + 1])
                    i_f += 2
                    continue
                if a_f in ("-exec", "-execdir"):
                    for exec_tok in args[i_f + 1 :]:
                        if exec_tok in (";", "+"):
                            break
                        if Path(exec_tok).name in _BASH_READ_BINARIES:
                            has_read_exec = True
                            break
                    break
                if not a_f.startswith("-") and a_f not in ("!", "(", ")", ","):
                    try:
                        cand_r = Path(a_f).expanduser()
                        res_r = (active_cwd / cand_r).resolve() if not cand_r.is_absolute() else cand_r.resolve()
                        if res_r.is_relative_to(ws_root) and res_r.exists():
                            search_roots.append(res_r)
                    except OSError:
                        pass
                i_f += 1
            if has_read_exec:
                if not search_roots and active_cwd.is_relative_to(ws_root):
                    search_roots.append(active_cwd)
                for sr in search_roots:
                    saw_read_cmd = True
                    if sr.is_file():
                        _add_targets([sr.relative_to(ws_root).as_posix()], True)
                    elif sr.is_dir():
                        pats = name_pats or ["*"]
                        for pat in pats:
                            for m_p in sorted(sr.rglob(pat))[:50]:
                                if m_p.is_file() and m_p.is_relative_to(ws_root):
                                    _add_targets([m_p.relative_to(ws_root).as_posix()], True)
            continue

        if binary in ("python", "python3") or re.match(r"^python3\.\d+$", binary):
            if "-m" in args:
                m_idx = args.index("-m")
                if m_idx + 1 < len(args):
                    mod_name = args[m_idx + 1]
                    if mod_name in ("tokenize", "ast", "dis", "json.tool", "doctest", "py_compile", "tabnanny"):
                        for a_py in args[m_idx + 2 :]:
                            if a_py.startswith("-"):
                                continue
                            r_files, r_saw = _resolve_bash_read_arg(a_py, active_cwd, ws_root)
                            _add_targets(r_files, r_saw)
            continue

        if binary == "git":
            git_sub = next((a for a in args if not a.startswith("-")), "")
            if git_sub in ("diff", "show", "grep", "blame", "log", "cat-file"):
                if git_sub == "log" and not any(
                    a in ("-p", "--patch", "--stat") or a.startswith("-L") for a in args
                ):
                    continue
                if active_cwd.is_relative_to(ws_root):
                    saw_read_cmd = True
                    for a in args:
                        if a == git_sub or a.startswith("-"):
                            continue
                        cand_str = a.split(":", 1)[-1] if ":" in a and git_sub in ("show", "cat-file") else a
                        r_files, r_saw = _resolve_bash_read_arg(cand_str, active_cwd, ws_root)
                        _add_targets(r_files, r_saw)
            continue

        if binary in _BASH_READ_BINARIES:
            saw_outside_arg = False
            saw_ws_arg = False
            for a in args:
                if a.startswith("-"):
                    continue
                r_files, r_saw = _resolve_bash_read_arg(a, active_cwd, ws_root)
                if r_saw or r_files:
                    saw_ws_arg = True
                elif "/" in a or a.startswith("."):
                    try:
                        cand_out = Path(a).expanduser()
                        res_out = (active_cwd / cand_out).resolve() if not cand_out.is_absolute() else cand_out.resolve()
                        if res_out.exists() and not res_out.is_relative_to(ws_root):
                            saw_outside_arg = True
                    except OSError:
                        pass
                _add_targets(r_files, r_saw)
            if (
                not matched_targets
                and not saw_ws_arg
                and not saw_outside_arg
                and binary in ("grep", "egrep", "fgrep", "rg", "ripgrep", "ack", "ag")
                and active_cwd.is_relative_to(ws_root)
            ):
                saw_read_cmd = True

    if saw_read_cmd and not matched_targets:
        return ["<workspace-search>"]
    return matched_targets


def evaluate_pre_tool_use(payload: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    """Evaluate a Claude Code PreToolUse event and return hookSpecificOutput if blocked."""
    if not isinstance(payload, dict):
        return {}
    source_env = env if env is not None else os.environ
    tool_name = str(payload.get("tool_name") or "")
    raw_tool_input = payload.get("tool_input")
    tool_input = dict(raw_tool_input) if isinstance(raw_tool_input, dict) else {}
    cwd = str(payload.get("cwd") or os.getcwd())
    ws_root = resolve_project_root(cwd, env=source_env)
    out_dir = source_env.get("CC_AGY_DATA_DIR") or source_env.get("CLAUDE_PLUGIN_DATA")

    # 1. Always lint delegation prompts (including follow_up_prompt, plan, and checklist) for >=9-line code dictation
    if tool_name.endswith(("delegate_start", "delegate_to_gemini", "delegate_continue")):
        plan_items = tool_input.get("plan") if isinstance(tool_input.get("plan"), list) else []
        checklist_items = tool_input.get("checklist") if isinstance(tool_input.get("checklist"), list) else []
        prompt_text = "\n".join(
            [str(tool_input.get("prompt") or tool_input.get("follow_up_prompt") or "")]
            + [str(step) for step in plan_items]
            + [str(item) for item in checklist_items]
        )
        lint = lint_delegation_prompt(prompt_text)
        if not lint["ok"]:
            auto_repair = str(source_env.get("CC_AGY_REPAIR_CODE_DICTATION") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if not auto_repair:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": lint["reason"],
                    }
                }

    if not is_strict_mode_enabled(source_env):
        return {}

    # 2a. Block Read / Grep / Glob on out_dir (.hmac_key / receipts) or PLUGIN_ROOT and enforce Lead read cap in strict mode
    if tool_name in ("Read", "Grep", "Glob"):
        if (
            ws_root != PLUGIN_ROOT
            and tool_name in ("Grep", "Glob")
            and not tool_input.get("path")
            and Path(cwd).resolve().is_relative_to(PLUGIN_ROOT)
        ):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Strict orchestrator mode is active: {tool_name} on cc-agy-plugin internal source "
                        "is disabled. Use the MCP tools (`delegate_start`, `delegate_continue`, "
                        "`get_diff`, `delegate_apply`) directly."
                    ),
                }
            }
        for field in ("file_path", "path", "pattern", "glob"):
            target_val = tool_input.get(field)
            if not target_val:
                continue
            raw_t = str(target_val)
            if field in ("file_path", "path"):
                try:
                    cand_p = Path(raw_t).expanduser()
                    resolved_t = (Path(cwd).resolve() / cand_p).resolve() if not cand_p.is_absolute() else cand_p.resolve()
                    if ws_root != PLUGIN_ROOT and resolved_t.is_relative_to(PLUGIN_ROOT):
                        return {
                            "hookSpecificOutput": {
                                "hookEventName": "PreToolUse",
                                "permissionDecision": "deny",
                                "permissionDecisionReason": (
                                    f"Strict orchestrator mode is active: {tool_name} on cc-agy-plugin internal source "
                                    f"{raw_t!r} is disabled. Use the MCP tools (`delegate_start`, `delegate_continue`, "
                                    "`get_diff`, `delegate_apply`) directly."
                                ),
                            }
                        }
                except OSError:
                    pass
            classification = classify_path(raw_t, ws_root, out_dir, cwd=cwd, env=source_env)
            if (
                classification == "out_dir"
                or ".hmac_key" in raw_t
                or ".cc-agy-data" in raw_t
            ):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Strict orchestrator mode is active: {tool_name} on internal receipt/key store "
                            f"{raw_t!r} is disabled. Use delegate_status or get_diff instead."
                        ),
                    }
                }
        read_limit = resolve_lead_read_limit(source_env)
        if read_limit is not None:
            try:
                from server.receipts import ReceiptStore

                data_root = ReceiptStore._resolve_data_dir(out_dir or None, ws_root)
            except Exception:
                data_root = Path(out_dir) if out_dir else (ws_root / ".cc-agy-data")
            if resolve_cost_profile(source_env) == "economy" and _session_has_live_running_delegate_job(data_root):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Still running: a delegated worker job is actively running. Do NOT call {tool_name} on "
                            "host files while it runs; poll with delegate_result(job_id=..., wait=true) instead."
                        ),
                    }
                }
            if _session_has_lead_escalated_job(data_root, ws_root):
                return {}
            counter_path = data_root / ".lead_read_count"
            # Once a worker job from this session exists, the Lead gets extra reads to check the
            # result against the task. Without them a Lead that spent its planning reads has to
            # accept the worker's result unchecked.
            read_limit += _lead_review_read_bonus(data_root, source_env)
            try:
                current_reads = int(counter_path.read_text(encoding="utf-8").strip()) if counter_path.exists() else 0
            except Exception:
                current_reads = 0
            read_unit_cost = 1
            if (
                tool_name == "Read"
                and resolve_cost_profile(source_env) == "economy"
                and not _session_has_terminal_delegate_job(data_root)
            ):
                raw_lim_chk = tool_input.get("limit")
                try:
                    lim_int = int(raw_lim_chk) if raw_lim_chk is not None else 999999
                except (ValueError, TypeError):
                    lim_int = 999999
                if lim_int > 150 and tool_input.get("file_path"):
                    try:
                        req_c = Path(str(tool_input["file_path"])).expanduser()
                        res_c = (Path(cwd).resolve() / req_c).resolve() if not req_c.is_absolute() else req_c.resolve()
                        if res_c.is_relative_to(ws_root) and res_c.is_file():
                            with open(res_c, "rb") as fh_c:
                                line_cnt = sum(1 for _ in fh_c)
                            if line_cnt > 150:
                                read_unit_cost = 2
                    except Exception:
                        pass
            if current_reads >= read_limit or current_reads + read_unit_cost > read_limit:
                reason_msg = (
                    f"LeadReadBudgetExceeded: Lead read/search limit ({read_limit} calls) reached in strict "
                    "delegation mode. Do NOT read more files on the Lead orchestrator. Call delegate_start "
                    "(with wait=true and a concise prompt <500 words; preload_files=true and auto_apply=true "
                    "are enabled automatically), delegate_continue, get_diff, or delegate_apply "
                    "immediately; the worker has full repository and test access in its worktree."
                )
                if tool_name == "Read" and tool_input.get("file_path"):
                    try:
                        from server.repo_map import build_focused_file_outline

                        req_p = Path(str(tool_input["file_path"]))
                        res_p = (Path(cwd).resolve() / req_p).resolve() if not req_p.is_absolute() else req_p.resolve()
                        if res_p.is_relative_to(ws_root) and res_p.is_file():
                            rel_p = str(res_p.relative_to(ws_root))
                            outline = build_focused_file_outline(ws_root, [rel_p], max_tokens=450)
                            if outline:
                                reason_msg += (
                                    "\n\nFocused File Outline (auto-attached so you can call delegate_start immediately):\n"
                                    + outline
                                )
                    except Exception:
                        pass
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason_msg,
                    }
                }
            try:
                data_root.mkdir(parents=True, exist_ok=True)
                counter_path.write_text(str(current_reads + read_unit_cost), encoding="utf-8")
                if tool_name == "Read":
                    raw_fp = str(tool_input.get("file_path") or tool_input.get("path") or "").strip()
                    if raw_fp:
                        cand_fp = Path(raw_fp).expanduser()
                        res_fp = (Path(cwd).resolve() / cand_fp).resolve() if not cand_fp.is_absolute() else cand_fp.resolve()
                        if res_fp.is_relative_to(ws_root) and res_fp.is_file():
                            rel_fp = res_fp.relative_to(ws_root).as_posix()
                            read_files_path = data_root / ".lead_read_files.json"
                            existing_list: list[str] = []
                            if read_files_path.exists():
                                try:
                                    parsed_rf = json.loads(read_files_path.read_text(encoding="utf-8"))
                                    if isinstance(parsed_rf, list):
                                        existing_list = [str(x) for x in parsed_rf if isinstance(x, str)]
                                except Exception:
                                    existing_list = []
                            if rel_fp not in existing_list:
                                existing_list.append(rel_fp)
                                read_files_path.write_text(json.dumps(existing_list), encoding="utf-8")
                            raw_off = tool_input.get("offset")
                            raw_lim = tool_input.get("limit")
                            if raw_off is not None or raw_lim is not None:
                                try:
                                    w_off = max(1, int(raw_off)) if raw_off is not None else 1
                                    w_lim = max(50, int(raw_lim)) if raw_lim is not None else 250
                                    rw_path = data_root / ".lead_read_windows.json"
                                    existing_wins: dict[str, list[int]] = {}
                                    if rw_path.exists():
                                        parsed_rw = json.loads(rw_path.read_text(encoding="utf-8"))
                                        if isinstance(parsed_rw, dict):
                                            existing_wins = parsed_rw
                                    existing_wins[rel_fp] = [w_off, w_lim]
                                    rw_path.write_text(json.dumps(existing_wins), encoding="utf-8")
                                except Exception:
                                    pass
            except Exception:
                pass

    # 2b. Block direct Edit / Write / MultiEdit / NotebookEdit on workspace, out_dir (receipts), and PLUGIN_ROOT
    if tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        file_path = (
            tool_input.get("file_path")
            or tool_input.get("notebook_path")
            or tool_input.get("path")
            or ""
        )
        if file_path:
            cand_edit = Path(str(file_path))
            res_edit = (Path(cwd).resolve() / cand_edit).resolve() if not cand_edit.is_absolute() else cand_edit.resolve()
            if ws_root != PLUGIN_ROOT and res_edit.is_relative_to(PLUGIN_ROOT):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Strict orchestrator mode is active: direct {tool_name} on cc-agy-plugin source "
                            f"{file_path!r} is disabled."
                        ),
                    }
                }
            classification = classify_path(
                str(file_path), ws_root, out_dir, cwd=cwd, env=source_env
            )
            if classification == "workspace" and tool_name in ("Edit", "Write"):
                try:
                    from server.jobs import _is_test_file_path, _matches_allowed_paths
                    from server.receipts import ReceiptStore

                    store_esc = ReceiptStore(out_dir or None, workspace_root=ws_root)
                    rel_edit = res_edit.relative_to(ws_root).as_posix()
                    if (
                        rel_edit
                        and not rel_edit.startswith((".git/", ".cc-agy-data/", ".worktrees/"))
                        and not res_edit.is_symlink()
                        and _session_has_lead_escalated_job(store_esc.data_dir, ws_root)
                    ):
                        marker_esc = store_esc.data_dir / ".lead_session_started"
                        since_esc = marker_esc.stat().st_mtime - 1.0 if marker_esc.exists() else 0.0
                        esc_cnt_path = store_esc.data_dir / ".lead_escalated_edit_count"
                        try:
                            cur_esc_edits = (
                                int(esc_cnt_path.read_text(encoding="utf-8").strip())
                                if esc_cnt_path.exists()
                                else 0
                            )
                        except Exception:
                            cur_esc_edits = 0
                        if cur_esc_edits < 3:
                            authorized_esc = False
                            for r_esc in store_esc.list_receipts(workspace_root=ws_root):
                                if r_esc.get("kind", "delegate") != "delegate":
                                    continue
                                if r_esc.get("status") in ("running", "cancelled") or r_esc.get("cancelled"):
                                    continue
                                if r_esc.get("external_worktree_tamper") or not store_esc.verify_receipt_signature(r_esc):
                                    continue
                                rec_ts = float(
                                    r_esc.get("updated_at_epoch")
                                    or r_esc.get("finished_at_epoch")
                                    or r_esc.get("started_at_epoch")
                                    or 0.0
                                )
                                if rec_ts < since_esc:
                                    continue
                                esc_info = r_esc.get("escalation")
                                if not (
                                    r_esc.get("escalate_to_lead") is True
                                    or (isinstance(esc_info, dict) and esc_info.get("escalate_to_lead") is True)
                                ):
                                    continue
                                if r_esc.get("reverted"):
                                    continue
                                r_allowed = list(r_esc.get("allowed_paths") or [])
                                r_applied_files = set(r_esc.get("applied_files") or []) | set(r_esc.get("files_changed") or [])
                                if (
                                    rel_edit in r_applied_files
                                    or not r_allowed
                                    or _matches_allowed_paths(rel_edit, r_allowed)
                                    or _is_test_file_path(rel_edit)
                                ):
                                    authorized_esc = True
                                    break
                            if authorized_esc:
                                esc_cnt_path.write_text(str(cur_esc_edits + 1), encoding="utf-8")
                                return {}
                except Exception:
                    pass
            if classification in ("workspace", "out_dir"):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Strict orchestrator mode is active: direct {tool_name} on {classification} file "
                            f"{file_path!r} is disabled. Delegate code modifications to a Vertex Gemini "
                            f"worker using delegate_start or delegate_to_gemini."
                        ),
                    }
                }

    # 3. Block Bash mutations targeting workspace or out_dir, and enforce Lead read cap on Bash workspace reads
    if tool_name == "Bash":
        command = str(tool_input.get("command") or "")
        verdict = detect_bash_workspace_mutation(
            command, ws_root, out_dir, cwd=cwd, env=source_env
        )
        if verdict["blocked"]:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": verdict["reason"],
                }
            }
        test_guard_reason = _check_strict_bash_test_runner_guard(
            command, ws_root, out_dir, source_env
        )
        if test_guard_reason:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": test_guard_reason,
                }
            }
        bash_read_targets = detect_bash_workspace_read(
            command, ws_root, out_dir, cwd=cwd, env=source_env
        )
        if bash_read_targets:
            read_limit = resolve_lead_read_limit(source_env)
            if read_limit is not None:
                try:
                    from server.receipts import ReceiptStore

                    data_root = ReceiptStore._resolve_data_dir(out_dir or None, ws_root)
                except Exception:
                    data_root = Path(out_dir) if out_dir else (ws_root / ".cc-agy-data")
                if _session_has_lead_escalated_job(data_root, ws_root):
                    return {}
                if (
                    resolve_cost_profile(source_env) == "economy"
                    and read_limit > 0
                    and not _session_has_terminal_delegate_job(data_root)
                ):
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                "Economy strict mode: pre-delegation Bash file/search inspection is disabled so "
                                "Turn 1 stays fast and clean. Use Glob, Grep, or bounded Read(limit=120) "
                                f"(up to {read_limit} calls) if target paths are uncertain, or call delegate_start directly."
                            ),
                        }
                    }
                counter_path = data_root / ".lead_read_count"
                read_limit += _lead_review_read_bonus(data_root, source_env)
                try:
                    current_reads = int(counter_path.read_text(encoding="utf-8").strip()) if counter_path.exists() else 0
                except Exception:
                    current_reads = 0
                file_targets = [t for t in bash_read_targets if t != "<workspace-search>"]
                span_cost = 1
                for sed_m in re.finditer(r"\bsed\b[^;|&\n]*?['\"]?(\d+)\s*,\s*(\d+)\s*p", command):
                    try:
                        s_lo, s_hi = int(sed_m.group(1)), int(sed_m.group(2))
                        if s_hi >= s_lo:
                            span_cost = max(span_cost, (s_hi - s_lo + 120) // 120)
                    except ValueError:
                        pass
                if re.search(r"(?:^|[\s;|&`(])cat\b", command) and file_targets:
                    cat_lines = 0
                    for rel_ft in file_targets[:4]:
                        fp_cat = ws_root / rel_ft
                        if fp_cat.is_file():
                            try:
                                with open(fp_cat, "rb") as fh_cat:
                                    cat_lines += sum(1 for _ in fh_cat)
                            except OSError:
                                pass
                    if cat_lines > 120:
                        span_cost = max(span_cost, min(6, (cat_lines + 119) // 120))
                read_cost = max(1, len(file_targets), span_cost)
                spotcheck_marker = data_root / ".lead_git_diff_spotcheck_used"
                session_marker = data_root / ".lead_session_started"
                cur_session_sig = ""
                if session_marker.exists():
                    try:
                        cur_session_sig = session_marker.read_text(encoding="utf-8").strip() or str(
                            session_marker.stat().st_mtime_ns
                        )
                    except OSError:
                        cur_session_sig = ""
                spotcheck_used_in_session = False
                if spotcheck_marker.exists():
                    try:
                        prev_sig = spotcheck_marker.read_text(encoding="utf-8").strip()
                        if not cur_session_sig:
                            spotcheck_used_in_session = True
                        elif prev_sig == cur_session_sig:
                            spotcheck_used_in_session = True
                        elif session_marker.exists() and spotcheck_marker.stat().st_mtime >= session_marker.stat().st_mtime and prev_sig == "1":
                            spotcheck_used_in_session = True
                    except OSError:
                        spotcheck_used_in_session = True
                allow_multifile_git_diff_spotcheck = (
                    _is_readonly_git_diff_command(command)
                    and not spotcheck_used_in_session
                    and _session_has_multifile_delegate_job(data_root)
                )
                if allow_multifile_git_diff_spotcheck:
                    try:
                        data_root.mkdir(parents=True, exist_ok=True)
                        spotcheck_marker.write_text(cur_session_sig or "1", encoding="utf-8")
                    except OSError:
                        pass
                    read_cost = 0
                elif current_reads >= read_limit or current_reads + read_cost > read_limit:
                    reason_msg = (
                        f"LeadReadBudgetExceeded: Lead read/search limit ({read_limit} calls) reached in strict "
                        "delegation mode (including Bash file/diff inspection commands). Do NOT read more files on "
                        "the Lead orchestrator. Call delegate_start (with wait=true and a concise prompt <500 words; "
                        "preload_files=true and auto_apply=true are enabled automatically), delegate_continue, "
                        "get_diff, or delegate_apply immediately; the worker has full repository and test access."
                    )
                    if file_targets:
                        try:
                            from server.repo_map import build_focused_file_outline

                            outline = build_focused_file_outline(ws_root, file_targets[:2], max_tokens=450)
                            if outline:
                                reason_msg += (
                                    "\n\nFocused File Outline (auto-attached so you can call delegate_start immediately):\n"
                                    + outline
                                )
                        except Exception:
                            pass
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": reason_msg,
                        }
                    }
                try:
                    data_root.mkdir(parents=True, exist_ok=True)
                    counter_path.write_text(str(current_reads + read_cost), encoding="utf-8")
                    if file_targets:
                        read_files_path = data_root / ".lead_read_files.json"
                        existing_list: list[str] = []
                        if read_files_path.exists():
                            try:
                                parsed_rf = json.loads(read_files_path.read_text(encoding="utf-8"))
                                if isinstance(parsed_rf, list):
                                    existing_list = [str(x) for x in parsed_rf if isinstance(x, str)]
                            except Exception:
                                existing_list = []
                        changed = False
                        for rel_fp in file_targets:
                            if rel_fp not in existing_list:
                                existing_list.append(rel_fp)
                                changed = True
                        if changed:
                            read_files_path.write_text(json.dumps(existing_list), encoding="utf-8")
                except Exception:
                    pass

    return {}


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0

    decision = evaluate_pre_tool_use(payload)
    if decision:
        sys.stdout.write(json.dumps(decision) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
