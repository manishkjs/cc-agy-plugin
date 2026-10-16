#!/usr/bin/env python3
"""PreToolUse hook guard and prompt dictation linter for cc-agy-plugin."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any


def is_strict_mode_enabled(env: dict[str, str] | None = None) -> bool:
    """Check whether strict orchestrator delegation mode is enabled."""
    source = env if env is not None else os.environ
    for key in (
        "CLAUDE_PLUGIN_OPTION_STRICT_DELEGATION_MODE",
        "CC_AGY_STRICT_MODE",
    ):
        val = str(source.get(key) or "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
    return False


def classify_path(
    target_path: str | Path,
    workdir: str | Path,
    out_dir: str | Path | None = None,
) -> str:
    """Classify a filesystem path relative to out_dir, workdir, and /tmp.

    CRITICAL ORDERING INVARIANT:
    Always check `out_dir` and `workdir` BEFORE `/tmp` (or `/private/tmp`) so that
    git repositories or worktrees located under `/tmp/...` on Linux/macOS are
    classified as `"workspace"` (and protected against direct driver mutation),
    never misclassified as `"tmp_scratch"`.
    """
    ws_resolved = Path(workdir).resolve()
    raw_str = str(target_path).strip()
    candidate = Path(raw_str)
    if not candidate.is_absolute():
        resolved = (ws_resolved / candidate).resolve()
    else:
        resolved = candidate.resolve()

    if out_dir is not None:
        out_resolved = Path(out_dir).resolve()
        if resolved == out_resolved or str(resolved).startswith(str(out_resolved) + os.sep):
            return "out_dir"

    data_dir = (ws_resolved / ".cc-agy-data").resolve()
    if resolved == data_dir or str(resolved).startswith(str(data_dir) + os.sep):
        return "out_dir"

    if resolved == ws_resolved or str(resolved).startswith(str(ws_resolved) + os.sep):
        return "workspace"

    for tmp_root in (Path("/tmp").resolve(), Path("/var/tmp").resolve()):
        if resolved == tmp_root or str(resolved).startswith(str(tmp_root) + os.sep):
            return "tmp_scratch"

    return "external"


_CODE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"def\s+\w+\s*\(|"
    r"async\s+def\s+\w+\s*\(|"
    r"class\s+\w+[:\(]|"
    r"import\s+[\w\.]+|"
    r"from\s+[\w\.]+\s+import\s+|"
    r"return\s+.+|"
    r"if\s+.+:\s*$|"
    r"elif\s+.+:\s*$|"
    r"else:\s*$|"
    r"for\s+.+\s+in\s+.+:\s*$|"
    r"while\s+.+:\s*$|"
    r"try:\s*$|"
    r"except\b.*:\s*$|"
    r"finally:\s*$|"
    r"with\s+.+:\s*$|"
    r"(?:const|let|var|function)\s+\w+|"
    r"export\s+(?:default\s+)?(?:function|class|const|interface|type)\s+|"
    r"public\s+|private\s+|protected\s+|"
    r"[\w\.\[\]]+\s*=\s*[^=].*|"
    r"[{}();]\s*$"
    r")"
)


def lint_delegation_prompt(prompt_text: str, max_consecutive_code_lines: int = 8) -> dict[str, Any]:
    """Detect whether a delegation prompt dictates >= 9 consecutive lines of code.

    Checks both fenced code blocks and unfenced consecutive code-like lines.
    """
    lines = prompt_text.splitlines()
    in_fence = False
    fence_run = 0
    max_fence_run = 0
    code_run = 0
    max_code_run = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                if fence_run > max_fence_run:
                    max_fence_run = fence_run
                fence_run = 0
                in_fence = False
            else:
                in_fence = True
                fence_run = 0
            continue

        if in_fence:
            if stripped:
                fence_run += 1
                if fence_run > max_fence_run:
                    max_fence_run = fence_run
            continue

        if _CODE_LINE_RE.match(line):
            code_run += 1
            if code_run > max_code_run:
                max_code_run = code_run
        elif stripped == "":
            # Blank lines inside a code block do not reset immediately unless followed by prose
            continue
        else:
            code_run = 0

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
        "patch",
        "tee",
    }
)

_GIT_MUTATING_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "apply",
        "checkout",
        "restore",
        "reset",
        "clean",
        "stash",
        "revert",
        "cherry-pick",
        "merge",
        "rebase",
        "am",
    }
)

_INTERPRETER_INLINE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("python -c file mutation", re.compile(r"\bpython3?(?:\.\d+)?\s+-c\b")),
    ("node -e file mutation", re.compile(r"\bnode\s+(?:-e|--eval)\b")),
    ("perl in-place mutation", re.compile(r"\bperl\s+-[a-zA-Z]*[pi]")),
    ("ruby -e file mutation", re.compile(r"\bruby\s+-[a-zA-Z]*e\b")),
    ("sed -i in-place edit", re.compile(r"\bsed\s+(?:-[a-zA-Z]*i|--in-place)\b")),
    ("awk in-place edit", re.compile(r"\bawk\s+-i\s+inplace\b")),
    ("dd output file write", re.compile(r"\bdd\b[^;|&\n]*\bof=")),
    ("heredoc file write", re.compile(r"<<-?\s*['\"]?\w+['\"]?")),
)


def detect_bash_workspace_mutation(
    command: str,
    workdir: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect a Bash command for direct workspace tree mutations in strict mode."""
    cmd = (command or "").strip()
    if not cmd:
        return {"blocked": False, "reason": None}

    # 1. Check inline interpreter & in-place editor patterns
    for label, pattern in _INTERPRETER_INLINE_PATTERNS:
        if pattern.search(cmd):
            return {
                "blocked": True,
                "reason": (
                    f"Strict orchestrator mode blocked Bash command ({label}). "
                    f"Delegate workspace edits to a Vertex worker via delegate_start or delegate_to_gemini."
                ),
            }

    # 2. Check shell redirections (> or >>), ignoring harmless fd redirects like 2>&1 or >/dev/null
    redirection_matches = re.finditer(r"(?:\d?>>?\s*|\btee\s+(?:-a\s+)?)([^\s;|&\n]+)", cmd)
    for match in redirection_matches:
        target = match.group(1).strip("'\"")
        if target.startswith("&") or target in ("/dev/null", "/dev/stderr", "/dev/stdout"):
            continue
        classification = classify_path(target, workdir, out_dir)
        if classification == "workspace":
            return {
                "blocked": True,
                "reason": (
                    f"Strict orchestrator mode blocked shell redirection to workspace path {target!r}. "
                    f"Use delegate_start / delegate_apply instead."
                ),
            }

    # 3. Parse shell segments separated by ;, &&, ||, or |
    segments = re.split(r"(?:&&|\|\||[;|\n])", cmd)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if not tokens:
            continue

        # Skip leading env assignments (FOO=bar) or sudo/env wrappers
        idx = 0
        while idx < len(tokens) and (
            "=" in tokens[idx] and not tokens[idx].startswith("-")
            or tokens[idx] in ("env", "sudo", "command", "nohup")
        ):
            idx += 1
        if idx >= len(tokens):
            continue

        binary = Path(tokens[idx]).name
        args = tokens[idx + 1 :]

        if binary == "git" and args:
            sub = args[0]
            if sub in _GIT_MUTATING_SUBCOMMANDS:
                return {
                    "blocked": True,
                    "reason": (
                        f"Strict orchestrator mode blocked `git {sub}` on the host workspace. "
                        f"Use delegate_apply to apply verified worker diffs."
                    ),
                }

        if binary in _MUTATING_BINARIES:
            non_flag_args = [a for a in args if not a.startswith("-")]
            if not non_flag_args:
                continue
            # For cp/mv/install/rsync/ln, the destination is the last positional argument;
            # for rm/touch/truncate/chmod/chown/patch/mkdir, every positional path is mutated.
            if binary in ("cp", "mv", "install", "rsync", "ln") and len(non_flag_args) >= 2:
                candidate_targets = [non_flag_args[-1]]
                if binary == "mv":
                    # Moving a file out of workspace also mutates the workspace
                    candidate_targets = non_flag_args
            else:
                candidate_targets = non_flag_args

            for target in candidate_targets:
                classification = classify_path(target, workdir, out_dir)
                if classification == "workspace":
                    return {
                        "blocked": True,
                        "reason": (
                            f"Strict orchestrator mode blocked `{binary}` targeting workspace path {target!r}. "
                            f"Delegate workspace file changes via delegate_start or delegate_to_gemini."
                        ),
                    }

    return {"blocked": False, "reason": None}


def evaluate_pre_tool_use(payload: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    """Evaluate a Claude Code PreToolUse event and return hookSpecificOutput if blocked."""
    tool_name = str(payload.get("tool_name") or "")
    tool_input = dict(payload.get("tool_input") or {})
    cwd = str(payload.get("cwd") or os.getcwd())
    out_dir = (env or os.environ).get("CC_AGY_DATA_DIR") or (env or os.environ).get("CLAUDE_PLUGIN_DATA")

    # Always lint delegation prompts for >=9-line code dictation when calling delegate tools
    if tool_name.endswith(("delegate_start", "delegate_to_gemini")):
        prompt_text = str(tool_input.get("prompt") or "")
        lint = lint_delegation_prompt(prompt_text)
        if not lint["ok"]:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": lint["reason"],
                }
            }

    if not is_strict_mode_enabled(env):
        return {}

    if tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        file_path = (
            tool_input.get("file_path")
            or tool_input.get("notebook_path")
            or tool_input.get("path")
            or ""
        )
        if file_path:
            classification = classify_path(str(file_path), cwd, out_dir)
            if classification == "workspace":
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Strict orchestrator mode is active: direct {tool_name} on workspace file "
                            f"{file_path!r} is disabled. Delegate code modifications to a Vertex Gemini "
                            f"worker using delegate_start or delegate_to_gemini."
                        ),
                    }
                }

    if tool_name == "Bash":
        command = str(tool_input.get("command") or "")
        verdict = detect_bash_workspace_mutation(command, cwd, out_dir)
        if verdict["blocked"]:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": verdict["reason"],
                }
            }

    return {}


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0

    decision = evaluate_pre_tool_use(payload)
    if decision:
        sys.stdout.write(json.dumps(decision) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
