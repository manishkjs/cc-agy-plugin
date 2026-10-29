#!/usr/bin/env python3
"""Synchronize and verify shared routing tables and worker budget constants across docs and skills."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from server.model_routes import (  # noqa: E402
    DEFAULT_ECONOMY_FLASH_MAX_TOOL_CALLS,
    DEFAULT_FLASH_MAX_TOOL_CALLS,
    ROUTING_TABLE_BEGIN,
    ROUTING_TABLE_END,
    render_routing_markdown_table,
)

SHARED_ROUTING_DOC_PATHS: tuple[str, ...] = (
    "README.md",
    "skills/vertex/SKILL.md",
    "plugins/cc-agy-plugin/skills/vertex/SKILL.md",
    "agents/gemini-delegator.md",
)

_STALE_BUDGET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"default\s+`?40`?\s+on\s+Flash", re.IGNORECASE), "stale Flash tool-call cap 40"),
    (re.compile(r"\b75\s+calls\b", re.IGNORECASE), "stale Flash tool-call cap 75"),
)

_STALE_FIXED_TEST_COUNT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"badge/tests-\d+%20passing"), "hardcoded test count in README badge"),
    (re.compile(r"\b\d+\s+automated(?:\s+unit,)?\s+tests\b", re.IGNORECASE), "hardcoded test count in README prose"),
)


def expected_routing_block() -> str:
    return f"{ROUTING_TABLE_BEGIN}\n{render_routing_markdown_table()}\n{ROUTING_TABLE_END}"


def sync_routing_blocks(root: Path, *, write: bool = False) -> list[str]:
    """Validate (or rewrite when `write=True`) shared routing blocks and budget constants under `root`."""
    plugin_root = Path(root).resolve()
    expected_block = expected_routing_block()
    block_re = re.compile(
        re.escape(ROUTING_TABLE_BEGIN) + r".*?" + re.escape(ROUTING_TABLE_END),
        re.DOTALL,
    )
    errors: list[str] = []

    for rel in SHARED_ROUTING_DOC_PATHS:
        doc_path = plugin_root / rel
        if not doc_path.is_file():
            continue
        content = doc_path.read_text(encoding="utf-8")
        matches = list(block_re.finditer(content))
        if len(matches) != 1:
            errors.append(
                f"{rel}: expected exactly 1 {ROUTING_TABLE_BEGIN}..{ROUTING_TABLE_END} block, found {len(matches)}"
            )
            continue
        actual_block = matches[0].group(0)
        if actual_block != expected_block:
            if write:
                updated = content[: matches[0].start()] + expected_block + content[matches[0].end() :]
                doc_path.write_text(updated, encoding="utf-8")
            else:
                errors.append(
                    f"{rel}: shared routing table out of sync with server/model_routes.py "
                    "(run `python3 scripts/sync_docs.py --write`)"
                )

    for rel in SHARED_ROUTING_DOC_PATHS:
        doc_path = plugin_root / rel
        if not doc_path.is_file():
            continue
        content = doc_path.read_text(encoding="utf-8")
        for pat, desc in _STALE_BUDGET_PATTERNS:
            if pat.search(content):
                errors.append(f"{rel}: contains {desc}")
        if rel in ("README.md", "skills/vertex/SKILL.md"):
            flash_cap_str = f"`{DEFAULT_FLASH_MAX_TOOL_CALLS}`"
            econ_cap_str = f"`{DEFAULT_ECONOMY_FLASH_MAX_TOOL_CALLS}`"
            if flash_cap_str not in content or econ_cap_str not in content:
                errors.append(
                    f"{rel}: missing current Flash tool-call caps ({flash_cap_str} balanced / {econ_cap_str} economy)"
                )

    readme_path = plugin_root / "README.md"
    if readme_path.is_file():
        readme_text = readme_path.read_text(encoding="utf-8")
        for pat, desc in _STALE_FIXED_TEST_COUNT_PATTERNS:
            if pat.search(readme_text):
                errors.append(f"README.md: contains {desc}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify shared docs match server/model_routes.py (default).")
    mode.add_argument("--write", action="store_true", help="Rewrite shared routing table blocks in place.")
    parser.add_argument("--root", type=Path, default=PLUGIN_ROOT, help="Plugin repository root.")
    args = parser.parse_args(argv)

    errors = sync_routing_blocks(args.root, write=bool(args.write))
    if args.write and not errors:
        errors = sync_routing_blocks(args.root, write=False)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
