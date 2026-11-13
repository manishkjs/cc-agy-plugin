"""Prompt dictation, self-debate, and code-block linter for delegated worker prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

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
