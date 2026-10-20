"""File-backed review evidence bundle and enclosing-symbol diff slicer.

Avoids inline prompt bloat during QA diff audits and multi-turn worker continuations by:
1. Writing full review artifacts (`task.md`, `changes.diff`, `test_output.txt`,
   `enclosing_slices.txt`, `INDEX.md`) to `<worktree>/.cc-agy-review/`.
2. Extracting AST-backed enclosing function/class slices around unified diff hunks
   so 1-shot reviewers see complete modified functions rather than 3-line diff context.
3. Producing a compact (<500-token) pointer prompt for tool-equipped reviewers.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Sequence

from server.regions import estimate_tokens, truncate_to_token_budget

REVIEW_DIR_NAME = ".cc-agy-review"
DEFAULT_SLICE_MAX_TOKENS = 6000
POINTER_PROMPT_MAX_TOKENS = 500

_DIFF_FILE_RE = re.compile(r"^\+\+\+\s+(?:b/)?(.+)$")
_DIFF_HUNK_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@")


def _parse_diff_hunks(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Map relative file path -> list of (start_line, end_line) in the post-edit file."""
    file_hunks: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    for raw_line in diff_text.splitlines():
        m_file = _DIFF_FILE_RE.match(raw_line)
        if m_file:
            candidate = m_file.group(1).strip()
            if candidate == "/dev/null":
                current_file = None
            else:
                current_file = candidate
                file_hunks.setdefault(current_file, [])
            continue
        if current_file is None:
            continue
        m_hunk = _DIFF_HUNK_RE.match(raw_line)
        if m_hunk:
            start = max(1, int(m_hunk.group(1)))
            count = int(m_hunk.group(2)) if m_hunk.group(2) is not None else 1
            end = max(start, start + max(count, 1) - 1)
            file_hunks[current_file].append((start, end))
    return file_hunks


def _merge_intervals(intervals: Sequence[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: list[tuple[int, int, str]] = [sorted_iv[0]]
    for start, end, label in sorted_iv[1:]:
        prev_start, prev_end, prev_label = merged[-1]
        if start <= prev_end + 3:
            combined_label = prev_label if label in prev_label else f"{prev_label}, {label}".strip(", ")
            merged[-1] = (prev_start, max(prev_end, end), combined_label)
        else:
            merged.append((start, end, label))
    return merged


def _python_enclosing_intervals(
    source_text: str,
    hunks: Sequence[tuple[int, int]],
    total_lines: int,
    *,
    fallback_pad: int = 25,
) -> list[tuple[int, int, str]]:
    """Find enclosing function/method/class spans for each diff hunk in a Python file."""
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return [
            (max(1, s - fallback_pad), min(total_lines, e + fallback_pad), f"lines {s}-{e}")
            for s, e in hunks
        ]

    # Collect (start, end, kind_and_name, span_len) for all functions and classes
    symbols: list[tuple[int, int, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lineno = getattr(node, "lineno", None)
            end_lineno = getattr(node, "end_lineno", None)
            if lineno is not None and end_lineno is not None:
                 decorator_start = min(
                     [lineno] + [getattr(d, "lineno", lineno) for d in getattr(node, "decorator_list", [])]
                 )
                 kind = "class" if isinstance(node, ast.ClassDef) else "def"
                 symbols.append((decorator_start, end_lineno, f"{kind} {node.name}", end_lineno - decorator_start + 1))

    intervals: list[tuple[int, int, str]] = []
    for h_start, h_end in hunks:
        # Prefer the tightest enclosing FunctionDef/AsyncFunctionDef, or tightest ClassDef if<=160 lines
        matching_funcs = [
            (s, e, label, length)
            for s, e, label, length in symbols
            if label.startswith("def ") and s <= h_end and e >= h_start and length <= 180
        ]
        if matching_funcs:
            matching_funcs.sort(key=lambda x: x[3])
            best_s, best_e, best_label, _ = matching_funcs[0]
            intervals.append((best_s, best_e, best_label))
            continue

        matching_classes = [
            (s, e, label, length)
            for s, e, label, length in symbols
            if label.startswith("class ") and s <= h_end and e >= h_start and length <= 160
        ]
        if matching_classes:
            matching_classes.sort(key=lambda x: x[3])
            best_s, best_e, best_label, _ = matching_classes[0]
            intervals.append((best_s, best_e, best_label))
            continue

        intervals.append(
            (
                max(1, h_start - fallback_pad),
                min(total_lines, h_end + fallback_pad),
                f"lines {h_start}-{h_end}",
            )
        )
    return _merge_intervals(intervals)


def extract_enclosing_diff_slices(
    worktree: Path,
    diff_text: str,
    *,
    max_tokens: int = DEFAULT_SLICE_MAX_TOKENS,
) -> str:
    """Extract enclosing function/class code slices around every modified diff hunk."""
    if not diff_text.strip():
        return ""
    root = Path(worktree).resolve()
    file_hunks = _parse_diff_hunks(diff_text)
    if not file_hunks:
        return ""

    blocks: list[str] = []
    for rel_path, hunks in file_hunks.items():
        if not hunks:
            continue
        candidate = (root / rel_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if not candidate.is_file():
            continue
        try:
            source_text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = source_text.splitlines()
        if not lines:
            continue
        total_lines = len(lines)
        if rel_path.endswith(".py"):
            intervals = _python_enclosing_intervals(source_text, hunks, total_lines)
        else:
            raw_iv = [
                (max(1, s - 25), min(total_lines, e + 25), f"lines {s}-{e}")
                for s, e in hunks
            ]
            intervals = _merge_intervals(raw_iv)

        for start_line, end_line, label in intervals:
            numbered = "\n".join(
                f"{idx:4d}: {lines[idx - 1]}"
                for idx in range(start_line, min(end_line, total_lines) + 1)
            )
            blocks.append(
                f"### {rel_path} ({label}, L{start_line}-L{end_line})\n```python\n{numbered}\n```"
                if rel_path.endswith(".py")
                else f"### {rel_path} ({label}, L{start_line}-L{end_line})\n```\n{numbered}\n```"
            )

    combined = "\n\n".join(blocks)
    if not combined:
        return ""
    sliced, _truncated, _orig = truncate_to_token_budget(combined, max_tokens)
    return sliced


def build_review_evidence_dir(
    worktree: Path,
    *,
    task_prompt: str,
    diff_text: str,
    test_output: str = "",
    files_changed: Sequence[str] = (),
    max_slice_tokens: int = DEFAULT_SLICE_MAX_TOKENS,
) -> dict[str, Any]:
    """Write `<worktree>/.cc-agy-review/` artifacts and return a <500-token pointer prompt."""
    root = Path(worktree).resolve()
    review_dir = root / REVIEW_DIR_NAME
    review_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    enclosing_slices = extract_enclosing_diff_slices(
        root, diff_text, max_tokens=max_slice_tokens
    )

    artifacts: dict[str, str] = {
        "task.md": task_prompt.strip() + "\n",
        "changes.diff": diff_text if diff_text.endswith("\n") else diff_text + "\n",
        "test_output.txt": (test_output.strip() or "(no acceptance test output recorded)") + "\n",
        "enclosing_slices.txt": (enclosing_slices.strip() or "(no enclosing slices extracted)") + "\n",
    }

    manifest_rows: list[str] = [
        "| Artifact | Path | Lines | Est. Tokens |",
        "|---|---|---:|---:|",
    ]
    for name, content in artifacts.items():
        target = review_dir / name
        target.write_text(content, encoding="utf-8")
        line_count = len(content.splitlines())
        tok_count = estimate_tokens(content)
        manifest_rows.append(
            f"| `{name}` | `{REVIEW_DIR_NAME}/{name}` | {line_count} | {tok_count} |"
        )

    changed_list = list(files_changed) or list(_parse_diff_hunks(diff_text).keys())
    changed_section = (
        "\n".join(f"- `{p}`" for p in changed_list[:25])
        if changed_list
        else "- (none)"
    )
    if len(changed_list) > 25:
        changed_section += f"\n- ... (+{len(changed_list) - 25} more files)"

    index_markdown = (
        "# Review Evidence Manifest\n\n"
        + "\n".join(manifest_rows)
        + "\n\n## Modified Files\n"
        + changed_section
        + "\n"
    )
    index_path = review_dir / "INDEX.md"
    index_path.write_text(index_markdown, encoding="utf-8")

    raw_pointer_prompt = (
        f"Review evidence is stored on disk under `{REVIEW_DIR_NAME}/`:\n"
        f"- Manifest: `{REVIEW_DIR_NAME}/INDEX.md`\n"
        f"- Task specification: `{REVIEW_DIR_NAME}/task.md` ({estimate_tokens(artifacts['task.md'])} tokens)\n"
        f"- Candidate unified diff: `{REVIEW_DIR_NAME}/changes.diff` ({estimate_tokens(artifacts['changes.diff'])} tokens)\n"
        f"- Enclosing function/class slices: `{REVIEW_DIR_NAME}/enclosing_slices.txt` ({estimate_tokens(artifacts['enclosing_slices.txt'])} tokens)\n"
        f"- Acceptance test output: `{REVIEW_DIR_NAME}/test_output.txt` ({estimate_tokens(artifacts['test_output.txt'])} tokens)\n"
        "Inspect the manifest and slices first, then verify every requirement against the candidate diff."
    )
    pointer_prompt, _trunc, _orig = truncate_to_token_budget(
        raw_pointer_prompt, POINTER_PROMPT_MAX_TOKENS
    )

    return {
        "review_dir": str(review_dir),
        "index_path": str(index_path),
        "index_markdown": index_markdown,
        "pointer_prompt": pointer_prompt,
        "pointer_prompt_tokens": estimate_tokens(pointer_prompt),
        "enclosing_slices": enclosing_slices,
        "enclosing_slices_tokens": estimate_tokens(enclosing_slices),
    }
