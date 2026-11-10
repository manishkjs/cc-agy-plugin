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

VALID_CURSOR_HOOK_EVENTS: frozenset[str] = frozenset(
    {
        "sessionStart",
        "sessionEnd",
        "preToolUse",
        "postToolUse",
        "postToolUseFailure",
        "subagentStart",
        "subagentStop",
        "beforeShellExecution",
        "afterShellExecution",
        "beforeMCPExecution",
        "afterMCPExecution",
        "beforeReadFile",
        "afterFileEdit",
        "beforeSubmitPrompt",
        "preCompact",
        "stop",
        "afterAgentResponse",
        "afterAgentThought",
        "beforeTabFileRead",
        "afterTabFileEdit",
        "workspaceOpen",
    }
)

VALID_OPENCODE_SKILL_KEYS: frozenset[str] = frozenset(
    {"name", "description", "license", "compatibility", "metadata", "disable-model-invocation"}
)
_OPENCODE_MAX_SAFE_INT = 9_007_199_254_740_991

VALID_OPENCODE_COMMAND_KEYS: frozenset[str] = frozenset(
    {"description", "agent", "model", "variant", "subtask", "subagent"}
)

VALID_OPENCODE_AGENT_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "variant",
        "temperature",
        "top_p",
        "prompt",
        "tools",
        "disable",
        "description",
        "mode",
        "hidden",
        "options",
        "color",
        "steps",
        "maxSteps",
        "permission",
    }
)

_OPENCODE_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_REQUIRED_OPENCODE_COMMANDS: tuple[str, ...] = ("delegate", "doctor", "jobs", "cost")

_TRIGGER_PHRASES: tuple[str, ...] = (
    "use when",
    "use for",
    "when to use",
    "triggers on",
    "activate when",
    "use this skill when",
)

_SCRIPT_REF_RE = re.compile(
    r"(?:(?:\$\{CLAUDE_PLUGIN_ROOT\}|\$\{CURSOR_PLUGIN_ROOT\}|\.)/)?(?:scripts|server|hooks)/[A-Za-z0-9_./-]+\.(?:py|sh|ps1)\b"
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


def _strip_yaml_inline_comment(value: str) -> str:
    """Strip an unquoted YAML `# ...` comment while preserving `#` inside quotes or mid-token."""
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(value):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_double:
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double and (idx == 0 or value[idx - 1].isspace()):
            return value[:idx].rstrip()
    return value.rstrip()


def _unquote_yaml_scalar(value: str) -> str:
    """Remove matching outer `'...'` or `\"...\"` quotes from a scalar YAML value."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _extract_frontmatter_lines(text: str) -> tuple[list[str], int, list[str]] | None:
    """Return `(all_lines, end_idx, frontmatter_body_lines)` when valid YAML `---` frontmatter is present."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return lines, idx, lines[1:idx]
    return None


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract simple YAML key: value frontmatter and return (metadata, body)."""
    extracted = _extract_frontmatter_lines(text)
    if extracted is None:
        return {}, text
    lines, end_idx, fm_lines = extracted

    meta: dict[str, str] = {}
    current_key: str | None = None
    for raw in fm_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in raw and not raw.startswith((" ", "\t")):
            key, val = raw.split(":", 1)
            current_key = key.strip()
            cleaned_val = _strip_yaml_inline_comment(val.strip())
            if cleaned_val in ("|", "|-", "|+", ">", ">-", ">+"):
                meta[current_key] = ""
            else:
                meta[current_key] = _unquote_yaml_scalar(cleaned_val)
        elif current_key is not None:
            cont = _strip_yaml_inline_comment(stripped)
            if cont:
                meta[current_key] = f"{meta[current_key]} {cont}".strip()
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return meta, body


def _parse_yaml_flow_rule_map(flow_str: str) -> dict[str, str] | None:
    """Parse a flat YAML flow mapping `{action: edit, resource: '*', effect: deny}`; return `None` if nested/invalid."""
    inner = flow_str.strip()
    if not (inner.startswith("{") and inner.endswith("}")):
        return None
    body = inner[1:-1].strip()
    if not body or any(ch in body for ch in ("{", "}", "[", "]")):
        return None
    rule: dict[str, str] = {}
    for part in body.split(","):
        seg = part.strip()
        if not seg:
            continue
        if ":" not in seg:
            return None
        rk, rv = seg.split(":", 1)
        clean_k = _unquote_yaml_scalar(rk.strip())
        clean_v = _unquote_yaml_scalar(_strip_yaml_inline_comment(rv.strip()))
        if not clean_k or not clean_v:
            return None
        rule[clean_k] = clean_v
    return rule if rule else None


def _parse_yaml_flow_rule_list(flow_list_str: str) -> list[dict[str, str]] | None:
    """Parse an inline YAML flow list `[{...}, {...}]` of flat rule mappings; return `None` if invalid."""
    outer = _strip_yaml_inline_comment(flow_list_str.strip())
    if not (outer.startswith("[") and outer.endswith("]")):
        return None
    inner = outer[1:-1].strip()
    if not inner:
        return None
    rules: list[dict[str, str]] = []
    depth = 0
    start: int | None = None
    for idx, ch in enumerate(inner):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return None
            if depth == 0 and start is not None:
                parsed = _parse_yaml_flow_rule_map(inner[start : idx + 1])
                if parsed is None:
                    return None
                rules.append(parsed)
                start = None
        elif depth == 0 and ch not in (" ", "\t", ","):
            return None
    if depth != 0 or not rules:
        return None
    return rules


def _has_opencode_v2_permissions_list(text: str) -> bool:
    """Return True when frontmatter defines OpenCode V2 `permissions:` as a list of `{action, resource, effect}` rules."""
    extracted = _extract_frontmatter_lines(text)
    if extracted is None:
        return False
    _, _, fm_lines = extracted
    in_permissions = False
    for raw in fm_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        expanded = raw.expandtabs(2)
        indent = len(expanded) - len(expanded.lstrip(" "))
        if indent == 0:
            if in_permissions and (stripped == "-" or stripped.startswith("- ")):
                return True
            if ":" in raw and raw.split(":", 1)[0].strip() == "permissions":
                rhs = _strip_yaml_inline_comment(raw.split(":", 1)[1].strip())
                if rhs.startswith("[") and rhs.endswith("]"):
                    return True
                in_permissions = rhs == ""
            else:
                in_permissions = False
            continue
        if in_permissions:
            return stripped == "-" or stripped.startswith("- ")
    return False


def _is_global_permission_resource(resource: str) -> bool:
    """Return True when `resource` covers all files (`*`, `**`, `**/*`, `/*`)."""
    return resource.strip() in ("*", "**", "**/*", "/*")


def _opencode_wildcard_matches(pattern: str, target: str) -> bool:
    """Match whole-value OpenCode wildcard pattern (`*` any chars, `?` single char) against `target`."""
    pat = pattern.strip()
    if not pat:
        return False
    regex_parts: list[str] = ["^"]
    for ch in pat:
        if ch == "*":
            regex_parts.append(".*")
        elif ch == "?":
            regex_parts.append(".")
        else:
            regex_parts.append(re.escape(ch))
    regex_parts.append("$")
    return bool(re.match("".join(regex_parts), target, flags=re.DOTALL))


def _action_covers_edit(action: str) -> bool:
    """Return True when whole-value wildcard `action` matches `'edit'` (`edit`, `*`, `e*`, `*dit`, `ed?t`)."""
    return _opencode_wildcard_matches(action, "edit")


def _action_touches_edit(action: str) -> bool:
    """Return True when `action` matches `'edit'` or starts with `'edit'` (conservative fail-closed prefix check)."""
    act = action.strip()
    return _action_covers_edit(act) or act.startswith("edit")


def _opencode_v2_permissions_list_edit_effect(text: str) -> str | None:
    """Evaluate OpenCode V2 `permissions:` list rules (`{action, resource, effect}`) sequentially for `edit`.

    Fails closed (`None`) when any rule is malformed, uses nested block lists/maps, or omits `action`/`effect`.
    Processes rules in order ("last matching rule wins"): a trailing wildcard `deny` overrides earlier `allow`
    rules, while any unoverridden narrow `allow`/`ask` causes effective `edit` permission to evaluate to `allow`.
    """
    extracted = _extract_frontmatter_lines(text)
    if extracted is None:
        return None
    _, _, fm_lines = extracted
    in_permissions = False
    rules: list[dict[str, str]] = []
    cur_rule: dict[str, str] | None = None
    rule_dash_indent: int | None = None

    for raw in fm_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        expanded = raw.expandtabs(2)
        indent = len(expanded) - len(expanded.lstrip(" "))
        if indent == 0 and not (in_permissions and (stripped == "-" or stripped.startswith("- "))):
            if cur_rule is not None:
                rules.append(cur_rule)
                cur_rule = None
                rule_dash_indent = None
            if ":" in raw and raw.split(":", 1)[0].strip() == "permissions":
                rhs = _strip_yaml_inline_comment(raw.split(":", 1)[1].strip())
                if rhs.startswith("["):
                    flow_rules = _parse_yaml_flow_rule_list(rhs)
                    if flow_rules is None:
                        return None
                    rules.extend(flow_rules)
                    in_permissions = False
                elif rhs == "":
                    in_permissions = True
                else:
                    return None
            else:
                in_permissions = False
            continue
        if not in_permissions:
            continue
        if stripped == "-" or stripped.startswith("- "):
            if rule_dash_indent is not None and indent > rule_dash_indent:
                return None
            if cur_rule is not None:
                rules.append(cur_rule)
            rule_dash_indent = indent
            rem = _strip_yaml_inline_comment(stripped[1:].strip())
            if not rem:
                cur_rule = {}
                continue
            if rem.startswith("{"):
                flow_map = _parse_yaml_flow_rule_map(rem)
                if flow_map is None:
                    return None
                rules.append(flow_map)
                cur_rule = None
                continue
            if ":" not in rem:
                return None
            rk, rv = rem.split(":", 1)
            clean_k = _unquote_yaml_scalar(rk.strip())
            clean_v = _unquote_yaml_scalar(_strip_yaml_inline_comment(rv.strip()))
            if not clean_k or not clean_v or any(ch in clean_v for ch in ("{", "}", "[", "]")):
                return None
            cur_rule = {clean_k: clean_v}
        elif cur_rule is not None:
            if ":" not in stripped:
                return None
            rk, rv = stripped.split(":", 1)
            clean_k = _unquote_yaml_scalar(rk.strip())
            clean_v = _unquote_yaml_scalar(_strip_yaml_inline_comment(rv.strip()))
            if not clean_k or not clean_v or any(ch in clean_v for ch in ("{", "}", "[", "]")):
                return None
            cur_rule[clean_k] = clean_v
        else:
            return None

    if cur_rule is not None:
        rules.append(cur_rule)
    if not rules:
        return None

    global_edit_effect: str | None = None
    narrow_non_deny_active = False
    for rule in rules:
        action = rule.get("action", "").strip()
        resource = rule.get("resource", "*").strip()
        effect = rule.get("effect", "").strip()
        if not action or not resource or effect not in ("allow", "deny", "ask"):
            return None
        if not _action_touches_edit(action):
            continue
        if effect == "deny":
            if _is_global_permission_resource(resource) and _action_covers_edit(action):
                global_edit_effect = "deny"
                narrow_non_deny_active = False
        else:
            if _is_global_permission_resource(resource) and _action_covers_edit(action):
                global_edit_effect = effect
                narrow_non_deny_active = False
            else:
                narrow_non_deny_active = True

    if narrow_non_deny_active:
        return "allow"
    return global_edit_effect


def _opencode_v1_permission_map_edit_effect(text: str) -> str | None:
    """Evaluate sequential OpenCode V1 `permission:` map for `edit`."""
    extracted = _extract_frontmatter_lines(text)
    if extracted is None:
        return None
    _, _, fm_lines = extracted

    in_permission = False
    child_indent: int | None = None
    current_tool: str | None = None
    effective: str | None = None
    for raw in fm_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        expanded = raw.expandtabs(2)
        indent = len(expanded) - len(expanded.lstrip(" "))
        if indent == 0:
            in_permission = False
            child_indent = None
            current_tool = None
            if ":" in raw:
                top_key, top_val = raw.split(":", 1)
                if top_key.strip() == "permission":
                    raw_scalar = _strip_yaml_inline_comment(top_val.strip())
                    if raw_scalar.startswith("{") and raw_scalar.endswith("}"):
                        inner = raw_scalar[1:-1].strip()
                        if not inner:
                            return None
                        flow_effective: str | None = None
                        for part in inner.split(","):
                            if ":" not in part:
                                continue
                            fk_raw, fv_raw = part.split(":", 1)
                            fk = _unquote_yaml_scalar(_strip_yaml_inline_comment(fk_raw.strip()))
                            fv = _unquote_yaml_scalar(_strip_yaml_inline_comment(fv_raw.strip()))
                            if fk in ("*", "edit") and fv:
                                flow_effective = fv
                        return flow_effective
                    scalar = _unquote_yaml_scalar(raw_scalar)
                    if scalar and scalar not in ("|", "|-", "|+", ">", ">-", ">+"):
                        return scalar
                    in_permission = True
            continue
        if not in_permission or ":" not in stripped:
            continue
        if child_indent is None or indent < child_indent:
            child_indent = indent
        rule_key_raw, rule_val_raw = stripped.split(":", 1)
        rule_key = _unquote_yaml_scalar(_strip_yaml_inline_comment(rule_key_raw.strip()))
        rule_val = _unquote_yaml_scalar(_strip_yaml_inline_comment(rule_val_raw.strip()))
        if indent == child_indent:
            current_tool = rule_key
            if rule_val and rule_val not in ("|", "|-", "|+", ">", ">-", ">+"):
                if rule_key in ("*", "edit"):
                    effective = rule_val
        elif indent > child_indent and rule_val:
            if current_tool == "edit":
                if rule_val != "deny":
                    return rule_val
                if rule_key == "*":
                    effective = "deny"
            elif current_tool == "*":
                if rule_val != "deny":
                    effective = rule_val
                elif rule_key == "*":
                    effective = "deny"
    return effective


def _opencode_agent_effective_edit_permission(text: str) -> str | None:
    """Evaluate sequential OpenCode `permission:` (V1 map) or `permissions:` (V2 rule list) for `edit`.

    Direct children under `permission:` (`*`, `edit`, `bash`, ...) are evaluated in order, as well
    as inline flow maps (`permission: {edit: deny}`) and V2 `{action, resource, effect}` rule lists.
    If both `permission:` and `permissions:` are present in the same frontmatter, both must evaluate
    to `'deny'` (fail closed on conflict).
    """
    extracted = _extract_frontmatter_lines(text)
    if extracted is None:
        return None
    _, _, fm_lines = extracted
    has_v1_key = any(
        len(raw.expandtabs(2)) - len(raw.expandtabs(2).lstrip(" ")) == 0
        and ":" in raw
        and raw.split(":", 1)[0].strip() == "permission"
        for raw in fm_lines
        if raw.strip() and not raw.strip().startswith("#")
    )
    has_v2_key = any(
        len(raw.expandtabs(2)) - len(raw.expandtabs(2).lstrip(" ")) == 0
        and ":" in raw
        and raw.split(":", 1)[0].strip() == "permissions"
        for raw in fm_lines
        if raw.strip() and not raw.strip().startswith("#")
    )
    if has_v1_key and has_v2_key:
        v1_eff = _opencode_v1_permission_map_edit_effect(text)
        v2_eff = _opencode_v2_permissions_list_edit_effect(text)
        return "deny" if (v1_eff == "deny" and v2_eff == "deny") else None
    if has_v2_key or _has_opencode_v2_permissions_list(text):
        return _opencode_v2_permissions_list_edit_effect(text)
    return _opencode_v1_permission_map_edit_effect(text)


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
    elif cleaned.startswith("${CURSOR_PLUGIN_ROOT}/"):
        cleaned = cleaned[len("${CURSOR_PLUGIN_ROOT}/") :]
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


def validate_cursor_plugin(plugin_root: Path) -> list[str]:
    """Validate `.cursor-plugin/plugin.json`, `.cursor-plugin/mcp.json`, and `.cursor-plugin/hooks.json`."""
    root = Path(plugin_root).resolve()
    errors: list[str] = []

    manifest_path = root / ".cursor-plugin" / "plugin.json"
    if not manifest_path.is_file():
        return [f"Cursor plugin manifest does not exist: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{manifest_path}: invalid JSON ({exc})"]
    if not isinstance(manifest, dict):
        return [f"{manifest_path}: top-level JSON value must be an object"]

    if manifest.get("name") != "cc-agy-plugin":
        errors.append(f"{manifest_path}: expected name 'cc-agy-plugin', got {manifest.get('name')!r}")

    pyproject_path = root / "pyproject.toml"
    if pyproject_path.is_file():
        m_ver = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_path.read_text(encoding="utf-8"), re.M)
        if m_ver and manifest.get("version") != m_ver.group(1):
            errors.append(
                f"{manifest_path}: version {manifest.get('version')!r} does not match pyproject.toml {m_ver.group(1)!r}"
            )

    for dir_field in ("skills", "agents", "commands"):
        rel_val = manifest.get(dir_field)
        if not isinstance(rel_val, str) or not rel_val.strip():
            errors.append(f"{manifest_path}: missing non-empty '{dir_field}' directory path")
            continue
        target_dir = (root / rel_val).resolve()
        try:
            target_dir.relative_to(root)
        except ValueError:
            errors.append(f"{manifest_path}: '{dir_field}' path {rel_val!r} escapes plugin root")
            continue
        if not target_dir.is_dir():
            errors.append(f"{manifest_path}: '{dir_field}' directory does not exist: {target_dir}")

    mcp_ref = manifest.get("mcpServers")
    if not isinstance(mcp_ref, str) or not mcp_ref.strip():
        errors.append(f"{manifest_path}: 'mcpServers' must point to a dedicated Cursor MCP config path")
    else:
        mcp_path = (root / mcp_ref).resolve()
        if mcp_path == (root / ".mcp.json").resolve():
            errors.append(
                f"{manifest_path}: 'mcpServers' must not point to Claude Code root '.mcp.json' (${{user_config.*}} trap)"
            )
        elif not mcp_path.is_file():
            errors.append(f"{manifest_path}: MCP config does not exist: {mcp_path}")
        else:
            raw_mcp = mcp_path.read_text(encoding="utf-8", errors="replace")
            if "${user_config." in raw_mcp:
                errors.append(f"{mcp_path}: contains Claude Code ${{user_config.*}} placeholder unsupported by Cursor")
            if "${PLUGIN_ROOT}" in raw_mcp or "${PLUGIN_DATA}" in raw_mcp:
                errors.append(
                    f"{mcp_path}: contains ${{PLUGIN_ROOT}} or ${{PLUGIN_DATA}}; Cursor expands ${{CURSOR_PLUGIN_ROOT}} instead"
                )
            try:
                mcp_data = json.loads(raw_mcp)
            except json.JSONDecodeError as exc:
                errors.append(f"{mcp_path}: invalid JSON ({exc})")
                mcp_data = None
            if isinstance(mcp_data, dict):
                servers = mcp_data.get("mcpServers")
                if not isinstance(servers, dict) or not servers:
                    errors.append(f"{mcp_path}: missing or empty 'mcpServers' object")
                else:
                    for srv_name, srv_cfg in servers.items():
                        if not isinstance(srv_cfg, dict):
                            errors.append(f"{mcp_path}: mcpServers.{srv_name} must be an object")
                            continue
                        for arg in srv_cfg.get("args") or []:
                            if isinstance(arg, str):
                                for match in _SCRIPT_REF_RE.finditer(arg):
                                    errors.extend(_check_script_file(root, match.group(0), f"{mcp_path}:{srv_name}"))

    # Cursor auto-discovers root `hooks/hooks.json` when `hooks` is omitted from `.cursor-plugin/plugin.json`.
    # Require an explicit Cursor-formatted hooks config so Claude Code's PascalCase `hooks/hooks.json` is never scanned.
    hooks_ref = manifest.get("hooks")
    if not isinstance(hooks_ref, str) or not hooks_ref.strip():
        errors.append(
            f"{manifest_path}: 'hooks' must point to a Cursor hooks file so Cursor does not auto-discover Claude's hooks/hooks.json"
        )
    else:
        cursor_hooks_path = (root / hooks_ref).resolve()
        if cursor_hooks_path == (root / "hooks" / "hooks.json").resolve():
            errors.append(
                f"{manifest_path}: 'hooks' must not point to Claude Code 'hooks/hooks.json' (PascalCase event schema mismatch)"
            )
        elif not cursor_hooks_path.is_file():
            errors.append(f"{manifest_path}: Cursor hooks config does not exist: {cursor_hooks_path}")
        else:
            try:
                ch_data = json.loads(cursor_hooks_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{cursor_hooks_path}: invalid JSON ({exc})")
                ch_data = None
            if isinstance(ch_data, dict):
                ch_hooks = ch_data.get("hooks")
                if not isinstance(ch_hooks, dict):
                    errors.append(f"{cursor_hooks_path}: 'hooks' field must be an object")
                else:
                    for ev_name, entries in ch_hooks.items():
                        if ev_name not in VALID_CURSOR_HOOK_EVENTS:
                            errors.append(
                                f"{cursor_hooks_path}: invalid Cursor hook event {ev_name!r}"
                            )
                        if not isinstance(entries, list):
                            errors.append(f"{cursor_hooks_path}: event {ev_name!r} must map to a list")
                            continue
                        for idx, item in enumerate(entries):
                            if not isinstance(item, dict) or not isinstance(item.get("command"), str):
                                errors.append(
                                    f"{cursor_hooks_path}:{ev_name}[{idx}] must be a flat object with string 'command'"
                                )

    return errors


def validate_opencode_bundle(plugin_root: Path) -> list[str]:
    """Validate `.opencode/` skill, commands, and subagent against OpenCode v1.x and v2.x schemas."""
    root = Path(plugin_root).resolve()
    bundle_dir = root / ".opencode"
    if not bundle_dir.is_dir():
        return [f"OpenCode bundle directory does not exist: {bundle_dir}"]

    errors: list[str] = []

    skills_dir = bundle_dir / "skills"
    skill_files = sorted(skills_dir.rglob("SKILL.md")) if skills_dir.is_dir() else []
    expected_skill = skills_dir / "cc-agy-plugin" / "SKILL.md"
    if not expected_skill.is_file():
        errors.append(f"OpenCode skill does not exist: {expected_skill}")
    for skill_md in skill_files:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
        meta, _body = parse_frontmatter(content)
        unknown_skill_keys = sorted(set(meta.keys()) - VALID_OPENCODE_SKILL_KEYS)
        if unknown_skill_keys:
            errors.append(
                f"{skill_md}: unsupported OpenCode skill frontmatter key(s) {unknown_skill_keys} "
                f"(allowed: {sorted(VALID_OPENCODE_SKILL_KEYS)})"
            )
        if "disable-model-invocation" in meta:
            dmi_val = (meta.get("disable-model-invocation") or "").strip().lower()
            if dmi_val not in ("true", "false"):
                errors.append(
                    f"{skill_md}: 'disable-model-invocation' must be a boolean ('true' or 'false'), "
                    f"got {meta.get('disable-model-invocation')!r}"
                )
        name = (meta.get("name") or "").strip()
        desc = (meta.get("description") or "").strip()
        if not name:
            errors.append(f"{skill_md}: missing required frontmatter 'name'")
        else:
            if len(name) > 64 or not _OPENCODE_SKILL_NAME_RE.fullmatch(name):
                errors.append(
                    f"{skill_md}: OpenCode skill 'name' {name!r} must match ^[a-z0-9]+(-[a-z0-9]+)*$ (1..64 chars)"
                )
            if name != skill_md.parent.name:
                errors.append(
                    f"{skill_md}: OpenCode skill 'name' {name!r} must match directory name {skill_md.parent.name!r}"
                )
        if not desc:
            errors.append(f"{skill_md}: missing required frontmatter 'description'")
        else:
            if len(desc) > 1024:
                errors.append(
                    f"{skill_md}: OpenCode skill 'description' length {len(desc)} exceeds 1024-char limit"
                )
            if not any(phrase in desc.lower() for phrase in _TRIGGER_PHRASES):
                errors.append(
                    f"{skill_md}: description for skill '{name or '<unnamed>'}' lacks a trigger phrase"
                )

    commands_dir = bundle_dir / "commands"
    for cmd_stem in _REQUIRED_OPENCODE_COMMANDS:
        cmd_path = commands_dir / f"{cmd_stem}.md"
        if not cmd_path.is_file():
            errors.append(f"OpenCode command does not exist: {cmd_path}")
    if commands_dir.is_dir():
        for cmd_md in sorted(commands_dir.glob("*.md")):
            content = cmd_md.read_text(encoding="utf-8", errors="replace")
            meta, body = parse_frontmatter(content)
            if not (meta.get("description") or "").strip():
                errors.append(f"{cmd_md}: missing required frontmatter 'description'")
            unknown_keys = sorted(set(meta.keys()) - VALID_OPENCODE_COMMAND_KEYS)
            if unknown_keys:
                errors.append(
                    f"{cmd_md}: unsupported OpenCode command frontmatter key(s) {unknown_keys} "
                    f"(allowed: {sorted(VALID_OPENCODE_COMMAND_KEYS)})"
                )
            for bool_cmd_key in ("subtask", "subagent"):
                if bool_cmd_key in meta:
                    bool_val = (meta.get(bool_cmd_key) or "").strip().lower()
                    if bool_val not in ("true", "false"):
                        errors.append(
                            f"{cmd_md}: OpenCode command '{bool_cmd_key}' must be a boolean ('true' or 'false'), "
                            f"got {meta.get(bool_cmd_key)!r}"
                        )
            if "on the `vertex` MCP server" in body:
                errors.append(
                    f"{cmd_md}: references Claude Code's 'vertex' MCP server instead of OpenCode's 'agy' server"
                )

    agents_dir = bundle_dir / "agents"
    expected_agent_path = agents_dir / "gemini-delegator.md"
    if not expected_agent_path.is_file():
        errors.append(f"OpenCode subagent does not exist: {expected_agent_path}")
    agent_files = sorted(agents_dir.glob("*.md")) if agents_dir.is_dir() else []
    for agent_path in agent_files:
        agent_content = agent_path.read_text(encoding="utf-8", errors="replace")
        meta, _agent_body = parse_frontmatter(agent_content)
        if not (meta.get("description") or "").strip():
            errors.append(f"{agent_path}: missing required frontmatter 'description'")
        if (meta.get("mode") or "").strip() != "subagent":
            errors.append(f"{agent_path}: expected mode 'subagent', got {meta.get('mode')!r}")
        if "permissions" in meta:
            if not _has_opencode_v2_permissions_list(agent_content):
                errors.append(
                    f"{agent_path}: uses plural 'permissions'; OpenCode v1 agent schema requires singular 'permission'"
                )
        allowed_agent_keys = (
            VALID_OPENCODE_AGENT_KEYS | {"permissions"}
            if _has_opencode_v2_permissions_list(agent_content)
            else VALID_OPENCODE_AGENT_KEYS
        )
        unknown_agent_keys = sorted(set(meta.keys()) - allowed_agent_keys)
        if unknown_agent_keys:
            errors.append(
                f"{agent_path}: unsupported OpenCode agent frontmatter key(s) {unknown_agent_keys}"
            )
        if _opencode_agent_effective_edit_permission(agent_content) != "deny":
            errors.append(
                f"{agent_path}: OpenCode subagent effective 'permission.edit' must evaluate to 'deny' "
                "(including V2 'permissions' rules; ensure no trailing '\"*\": allow', glob override, or conflicting V1/V2 permission keys)"
            )
        errors.extend(check_always_on_budget([WORKER_SYSTEM_PROMPT, agent_content], max_tokens=1200))

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

    # 3. Markdown script reference check across skills, agents, commands, and .opencode
    md_files: list[Path] = []
    for sub in ("skills", "agents", "commands", ".opencode"):
        sub_dir = plugin_root / sub
        if sub_dir.exists():
            md_files.extend(sorted(sub_dir.rglob("*.md")))
    errors.extend(validate_script_refs(plugin_root, md_files))

    # 4. Hooks config + referenced Python scripts check
    hooks_json = plugin_root / "hooks" / "hooks.json"
    errors.extend(validate_hooks_config(hooks_json, plugin_root=plugin_root))

    # 5. Native Cursor plugin manifest and MCP/hooks compatibility checks
    errors.extend(validate_cursor_plugin(plugin_root))

    # 5b. Native OpenCode bundle (.opencode/) schema and prompt budget checks
    errors.extend(validate_opencode_bundle(plugin_root))

    # 6. Declarative rate card schema check
    rate_card_json = plugin_root / "server" / "rate_card.json"
    if not rate_card_json.is_file():
        errors.append(f"missing declarative rate card: {rate_card_json}")
    else:
        try:
            from server.rate_card import validate_rate_card

            loaded_card = json.loads(rate_card_json.read_text(encoding="utf-8"))
            for rc_err in validate_rate_card(loaded_card):
                errors.append(f"{rate_card_json}: {rc_err}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{rate_card_json}: failed to validate ({exc})")

    # 7. Shared routing table and worker budget constant sync across README, skills, and subagent docs
    sync_script = plugin_root / "scripts" / "sync_docs.py"
    if sync_script.is_file():
        from scripts.sync_docs import sync_routing_blocks

        errors.extend(sync_routing_blocks(plugin_root, write=False))

    return errors
