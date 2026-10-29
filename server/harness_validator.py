"""Pre-flight validator for plugin prompt budgets, skills, script references, and hooks.

Catches prompt bloat, orphan skills, broken script paths, Python syntax errors in hook
scripts, and invalid hook event keys in <50ms before any model call is dispatched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from server.agy_runner import WORKER_SYSTEM_PROMPT
from server.regions import estimate_tokens, truncate_to_token_budget

VALID_HOOK_EVENTS: frozenset[str] = frozenset(
    {
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "Notification",
        "Stop",
        "SubagentStart",
        "SubagentStop",
        "PreCompact",
        "SessionEnd",
    }
)

_TRIGGER_PHRASES: tuple[str, ...] = (
    "use when",
    "use for",
    "when to use",
    "triggers on",
    "activate when",
    "use this skill when",
)

_SCRIPT_REF_RE = re.compile(
    r"(?:(?:\$\{CLAUDE_PLUGIN_ROOT\}|\.)/)?(?:scripts|server|hooks)/[A-Za-z0-9_./-]+\.(?:py|sh)\b"
)

_TOKEN_WORD_RE = re.compile(r"[a-z0-9_]{3,}")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "when",
        "use",
        "only",
        "also",
        "are",
        "not",
        "all",
        "any",
        "can",
        "via",
        "one",
        "two",
    }
)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract simple YAML key: value frontmatter and return (metadata, body)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, text

    meta: dict[str, str] = {}
    current_key: str | None = None
    for raw in lines[1:end_idx]:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in raw and not raw.startswith((" ", "\t")):
            key, val = raw.split(":", 1)
            current_key = key.strip()
            meta[current_key] = val.strip().strip("'\"")
        elif current_key is not None:
            meta[current_key] = (meta[current_key] + " " + stripped).strip()
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return meta, body


def check_always_on_budget(texts: Sequence[str], max_tokens: int = 1200) -> list[str]:
    """Verify combined always-on instructions stay within `max_tokens`."""
    combined = "\n\n".join(t for t in texts if t)
    total = estimate_tokens(combined)
    if total > max_tokens:
        return [
            f"always-on prompt token budget exceeded: {total} tokens > max_tokens={max_tokens}"
        ]
    return []


def validate_skills(
    skill_dir: Path,
    always_on_text: str = "",
    *,
    require_trigger_in_always_on: bool = True,
) -> list[str]:
    """Validate every SKILL.md under `skill_dir` for frontmatter, trigger phrases, and references."""
    errors: list[str] = []
    root = Path(skill_dir)
    if not root.exists():
        return [f"skills directory does not exist: {root}"]

    skill_files = sorted(root.rglob("SKILL.md"))
    if not skill_files:
        return [f"no SKILL.md files found under {root}"]

    always_on_lower = always_on_text.lower()
    for skill_md in skill_files:
        try:
            content = skill_md.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append(f"{skill_md}: unreadable ({exc})")
            continue
        meta, _body = parse_frontmatter(content)
        name = (meta.get("name") or "").strip()
        description = (meta.get("description") or "").strip()
        if not name:
            errors.append(f"{skill_md}: missing required frontmatter 'name'")
        if not description:
            errors.append(f"{skill_md}: missing required frontmatter 'description'")
        else:
            desc_lower = description.lower()
            if not any(phrase in desc_lower for phrase in _TRIGGER_PHRASES):
                errors.append(
                    f"{skill_md}: description for skill '{name or '<unnamed>'}' lacks a trigger phrase "
                    f"(expected one of: {', '.join(_TRIGGER_PHRASES)})"
                )
        if name and always_on_lower and require_trigger_in_always_on:
            if name.lower() not in always_on_lower:
                errors.append(
                    f"{skill_md}: orphan skill '{name}' is not referenced in always-on instructions"
                )
    return errors


def autotrigger_skills(
    prompt: str,
    skill_dir: Path,
    *,
    allowed_paths: Sequence[str] = (),
    max_skills: int = 2,
    max_tokens_per_skill: int = 600,
    min_keyword_hits: int = 2,
) -> list[tuple[str, str]]:
    """Deterministically match up to `max_skills` skills against `prompt` and `allowed_paths`."""
    root = Path(skill_dir)
    if not root.exists():
        return []
    query_text = f"{prompt} {' '.join(allowed_paths)}".lower()
    query_tokens = {
        tok for tok in _TOKEN_WORD_RE.findall(query_text) if tok not in _STOPWORDS
    }
    if not query_tokens:
        return []

    scored: list[tuple[int, str, str]] = []
    for skill_md in sorted(root.rglob("SKILL.md")):
        try:
            content = skill_md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        meta, body = parse_frontmatter(content)
        name = (meta.get("name") or skill_md.parent.name).strip()
        description = (meta.get("description") or "").strip()
        skill_tokens = {
            tok
            for tok in _TOKEN_WORD_RE.findall(f"{name} {description}".lower())
            if tok not in _STOPWORDS
        }
        hits = len(query_tokens & skill_tokens)
        if hits >= min_keyword_hits:
            capped_body, _trunc, _orig = truncate_to_token_budget(
                body.strip(), max_tokens_per_skill
            )
            scored.append((hits, name, capped_body))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(name, body) for _hits, name, body in scored[:max_skills]]


def _check_script_file(root: Path, raw_ref: str, source_label: str) -> list[str]:
    cleaned = raw_ref.strip().strip("'\"")
    if cleaned.startswith("${CLAUDE_PLUGIN_ROOT}/"):
        cleaned = cleaned[len("${CLAUDE_PLUGIN_ROOT}/") :]
    elif cleaned.startswith("./"):
        cleaned = cleaned[2:]

    script_path = (Path(root).resolve() / cleaned).resolve()
    try:
        script_path.relative_to(Path(root).resolve())
    except ValueError:
        return [f"{source_label}: script reference '{raw_ref}' escapes plugin root"]

    if not script_path.is_file():
        return [f"{source_label}: missing referenced script '{cleaned}' ({script_path})"]

    try:
        text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"{source_label}: unable to read script '{cleaned}': {exc}"]

    if not text.strip():
        return [f"{source_label}: referenced script '{cleaned}' is empty"]

    if script_path.suffix == ".py":
        try:
            compile(text, str(script_path), "exec")
        except SyntaxError as exc:
            return [
                f"{source_label}: syntax error in '{cleaned}' at line {exc.lineno}: {exc.msg}"
            ]
    return []


def validate_script_refs(root: Path, markdown_files: Sequence[Path]) -> list[str]:
    """Scan markdown files for script paths (`scripts/...`, `server/...`, `hooks/...`) and validate them."""
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for md_path in markdown_files:
        try:
            content = Path(md_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append(f"{md_path}: unreadable ({exc})")
            continue
        for match in _SCRIPT_REF_RE.finditer(content):
            ref = match.group(0)
            key = (str(md_path), ref)
            if key in seen:
                continue
            seen.add(key)
            errors.extend(_check_script_file(root, ref, str(md_path)))
    return errors


def validate_hooks_config(
    hooks_json_path: Path,
    *,
    plugin_root: Path | None = None,
) -> list[str]:
    """Validate `hooks/hooks.json` schema, event keys, and referenced hook commands."""
    path = Path(hooks_json_path)
    if not path.is_file():
        return [f"hooks config does not exist: {path}"]

    root = Path(plugin_root).resolve() if plugin_root is not None else path.parent.parent.resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: invalid JSON ({exc})"]

    if not isinstance(data, dict):
        return [f"{path}: top-level JSON value must be an object"]

    hooks_map = data.get("hooks")
    if not isinstance(hooks_map, dict) or not hooks_map:
        return [f"{path}: missing or empty 'hooks' object"]

    errors: list[str] = []
    for event_name, event_groups in hooks_map.items():
        if event_name not in VALID_HOOK_EVENTS:
            errors.append(
                f"{path}: unknown hook event '{event_name}' (valid: {sorted(VALID_HOOK_EVENTS)})"
            )
            continue
        if not isinstance(event_groups, list) or not event_groups:
            errors.append(f"{path}: event '{event_name}' must map to a non-empty list")
            continue
        for grp_idx, group in enumerate(event_groups):
            if not isinstance(group, dict):
                errors.append(f"{path}: {event_name}[{grp_idx}] must be an object")
                continue
            inner_hooks = group.get("hooks")
            if not isinstance(inner_hooks, list) or not inner_hooks:
                errors.append(
                    f"{path}: {event_name}[{grp_idx}] must contain a non-empty 'hooks' list"
                )
                continue
            for h_idx, hook in enumerate(inner_hooks):
                label = f"{path}:{event_name}[{grp_idx}].hooks[{h_idx}]"
                if not isinstance(hook, dict):
                    errors.append(f"{label}: hook entry must be an object")
                    continue
                if hook.get("type") != "command":
                    errors.append(
                        f"{label}: unsupported hook type {hook.get('type')!r} (expected 'command')"
                    )
                cmd = hook.get("command")
                if not isinstance(cmd, str) or not cmd.strip():
                    errors.append(f"{label}: missing non-empty 'command' string")
                    continue
                normalized_cmd = cmd.replace('"', "").replace("'", "")
                for match in _SCRIPT_REF_RE.finditer(normalized_cmd):
                    errors.extend(_check_script_file(root, match.group(0), label))
    return errors


def validate_plugin_tree(root: Path) -> list[str]:
    """Run all pre-flight harness checks against the `cc-agy-plugin` repository tree."""
    plugin_root = Path(root).resolve()
    errors: list[str] = []

    # 1. Worker system prompt budget (<350 tokens) + combined always-on budget (<1200 tokens)
    errors.extend(check_always_on_budget([WORKER_SYSTEM_PROMPT], max_tokens=350))

    always_on_docs: list[str] = [WORKER_SYSTEM_PROMPT]
    delegator_md = plugin_root / "agents" / "gemini-delegator.md"
    if delegator_md.is_file():
        always_on_docs.append(delegator_md.read_text(encoding="utf-8", errors="replace"))
    errors.extend(check_always_on_budget(always_on_docs, max_tokens=1200))

    # 2. Skills frontmatter, trigger phrases, and non-orphan check against agent/command docs
    reference_corpus_parts: list[str] = list(always_on_docs)
    for cmd_md in sorted((plugin_root / "commands").glob("*.md")):
        reference_corpus_parts.append(cmd_md.read_text(encoding="utf-8", errors="replace"))
    errors.extend(
        validate_skills(
            plugin_root / "skills",
            always_on_text="\n".join(reference_corpus_parts),
            require_trigger_in_always_on=True,
        )
    )

    # 3. Markdown script reference check across skills, agents, and commands
    md_files: list[Path] = []
    for sub in ("skills", "agents", "commands"):
        sub_dir = plugin_root / sub
        if sub_dir.exists():
            md_files.extend(sorted(sub_dir.rglob("*.md")))
    errors.extend(validate_script_refs(plugin_root, md_files))

    # 4. Hooks config + referenced Python scripts check
    hooks_json = plugin_root / "hooks" / "hooks.json"
    errors.extend(validate_hooks_config(hooks_json, plugin_root=plugin_root))

    # 5. Shared routing table and worker budget constant sync across README, skills, and subagent docs
    sync_script = plugin_root / "scripts" / "sync_docs.py"
    if sync_script.is_file():
        from scripts.sync_docs import sync_routing_blocks

        errors.extend(sync_routing_blocks(plugin_root, write=False))

    return errors
