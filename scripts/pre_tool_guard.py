#!/usr/bin/env python3
"""PreToolUse hook guard and prompt dictation linter for cc-agy-plugin."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
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

    if out_dir is not None:
        out_str = str(out_dir).strip()
        if out_str and not out_str.startswith("${"):
            out_resolved = Path(out_str).resolve()
            if resolved.is_relative_to(out_resolved):
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


def lint_delegation_prompt(prompt_text: str, max_consecutive_code_lines: int = 8) -> dict[str, Any]:
    """Detect whether a delegation prompt dictates >= 9 consecutive lines of code.

    Checks both fenced code blocks (``` or ~~~) and unfenced consecutive code-like
    or 4-space indented code lines. Interleaved comments inside a code run do not
    reset the consecutive code counter.
    """
    lines = prompt_text.splitlines()
    in_fence = False
    fence_marker = ""
    fence_run = 0
    max_fence_run = 0
    code_run = 0
    max_code_run = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if in_fence and marker == fence_marker:
                if fence_run > max_fence_run:
                    max_fence_run = fence_run
                fence_run = 0
                in_fence = False
                fence_marker = ""
            elif not in_fence:
                in_fence = True
                fence_marker = marker
                fence_run = 0
            continue

        if in_fence:
            if stripped:
                fence_run += 1
                if fence_run > max_fence_run:
                    max_fence_run = fence_run
            continue

        if stripped == "":
            continue

        if _COMMENT_LINE_RE.match(line):
            # Comments inside an active code block do not reset code_run
            continue

        is_indented_code = (
            (line.startswith("    ") or line.startswith("\t"))
            and not _PROSE_LIST_RE.match(line)
        )
        if _CODE_LINE_RE.match(line) or is_indented_code:
            code_run += 1
            if code_run > max_code_run:
                max_code_run = code_run
        else:
            code_run = 0

    if in_fence and fence_run > max_fence_run:
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
        "black",
        "autopep8",
        "yapf",
        "isort",
        "prettier",
        "gofmt",
        "rustfmt",
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

_INLINE_MUTATION_KEYWORDS_RE = re.compile(
    r"(?:"
    r"\bopen\s*\(|"
    r"\b(?:exec|eval|compile|__import__|getattr|setattr|delattr|globals|locals|vars|fromhex|b64decode)\b|"
    r"\.(?:write\w*|unlink|rmdir|rename|replace|mkdir|makedirs|touch|chmod|chown|symlink_to|hardlink_to|copyFile\w*|appendFile\w*|openSync|createWriteStream)\s*\(|"
    r"\b(?:shutil|subprocess|tempfile|pathlib)\b|"
    r"\b(?:writeFile\w*|appendFile\w*|openSync|createWriteStream|file_put_contents|fopen|fwrite|unlink|rmdir|mkdir|rename|copyFile\w*)\s*\(|"
    r"\bos\.(?:remove|unlink|rmdir|removedirs|rename|renames|replace|mkdir|makedirs|chmod|chown|link|symlink|truncate|system|popen|exec\w*|spawn\w*|open)|"
    r"\bfs\.(?:write|append|unlink|rm|cp|mv|rename|mkdir|rmdir|chmod|chown|truncate|copyFile|createWriteStream)"
    r")"
)


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
            if idx + 1 < len(args):
                val = args[idx + 1]
                if tok == "-C":
                    eff_cwd = (eff_cwd / val).resolve()
                elif tok == "-c":
                    lower_v = val.lower()
                    if "!" in val or "alias." in lower_v or "hookspath" in lower_v or "fsmonitor" in lower_v:
                        return None, [], eff_cwd, f"Strict orchestrator mode blocked `git -c {val}`."
            idx += 2
            continue
        if any(tok.startswith(prefix + "=") for prefix in _GIT_OPTS_WITH_VALUE):
            val = tok.split("=", 1)[1]
            if tok.startswith("-C="):
                eff_cwd = (eff_cwd / val).resolve()
            elif tok.startswith("-c="):
                lower_v = val.lower()
                if "!" in val or "alias." in lower_v or "hookspath" in lower_v or "fsmonitor" in lower_v:
                    return None, [], eff_cwd, f"Strict orchestrator mode blocked `git {tok}`."
            idx += 1
            continue
        if tok.startswith("-C") and len(tok) > 2:
            eff_cwd = (eff_cwd / tok[2:]).resolve()
            idx += 1
            continue
        if tok.startswith("-c") and len(tok) > 2:
            val = tok[2:]
            lower_v = val.lower()
            if "!" in val or "alias." in lower_v or "hookspath" in lower_v or "fsmonitor" in lower_v:
                return None, [], eff_cwd, f"Strict orchestrator mode blocked `git {tok}`."
            idx += 1
            continue
        if tok.startswith("-"):
            idx += 1
            continue
        return tok, args[idx + 1 :], eff_cwd, None
    return None, [], eff_cwd, None


def detect_bash_workspace_mutation(
    command: str,
    workdir: str | Path,
    out_dir: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
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

    # 0. Block direct access to .hmac_key, ${IFS} obfuscation, process substitution <( / >(, and awk system()/redirect
    if ".hmac_key" in cmd:
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked reading or modifying the internal `.hmac_key`.",
        }
    if "${IFS}" in cmd or "$IFS" in cmd:
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked `$IFS` shell token obfuscation.",
        }
    if re.search(r"[<>]\(", cmd):
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked bash process substitution `<(...)` / `>(...)`.",
        }

    # 1. Check inline sed/awk/perl in-place flags or awk system()/redirection anywhere in the command
    if re.search(r"\bsed\s+(?:-[a-zA-Z]*i|--in-place)\b", cmd):
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked `sed -i` in-place edit.",
        }
    if re.search(r"\bawk\b[^;|&\n]*(?:-i\s+inplace|\bsystem\s*\(|>\s*['\"])", cmd):
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked mutating `awk` command (`-i inplace`, `system()`, or file redirect).",
        }
    if re.search(r"\bperl\s+-[a-zA-Z]*[pi]", cmd):
        return {
            "blocked": True,
            "reason": "Strict orchestrator mode blocked `perl -p/-i` in-place mutation.",
        }

    # 2. Check shell output redirections (>, >>, >|, &>, &>>, 1>, 2>, <>),
    # ignoring here-strings (<<<), heredocs (<<), and fd duplicates (2>&1, >/dev/null).
    cmd_without_herestrings = re.sub(r"<<<", "   ", cmd)
    cmd_without_heredocs = re.sub(r"<<-?\s*['\"]?\w+['\"]?", "   ", cmd_without_herestrings)
    unquoted = _unquoted_mask(cmd_without_heredocs)
    redir_pattern = re.compile(
        r"(?:&>>?|\d?>>?|\d?>[|&]?|<>|\b(?:tee|sponge)\s+(?:-[a-zA-Z]+\s+)*)\s*([^\s;|&\n]+)"
    )
    for match in redir_pattern.finditer(cmd_without_heredocs):
        if match.start() < len(unquoted) and not unquoted[match.start()]:
            continue
        raw_target = match.group(1).strip("'\"()")
        if not raw_target or raw_target.startswith("&") or raw_target in ("/dev/null", "/dev/stderr", "/dev/stdout"):
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
        raw_words = seg.split()
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
) -> tuple[dict[str, Any], Path]:
    """Inspect a single tokenized command segment, unwrapping prefixes and tracking `cd`."""
    idx = 0
    n = len(tokens)
    used_xargs = False

    while idx < n:
        tok = Path(tokens[idx]).name if tokens[idx].startswith("/") else tokens[idx]
        # Skip leading VAR=val assignments
        if "=" in tok and not tok.startswith("-") and tok.split("=", 1)[0].isidentifier():
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
    args = tokens[idx + 1 :]

    # Track directory changes (`cd <dir>`) across chained segments
    if binary in ("cd", "pushd"):
        non_flags = [a for a in args if not a.startswith("-")]
        if non_flags:
            active_cwd = (active_cwd / non_flags[0]).resolve()
        return {"blocked": False, "reason": None}, active_cwd

    # Unwrap nested shells (`sh -c`, `bash -c`, `zsh -c`, `eval`)
    if binary in ("sh", "bash", "zsh", "dash", "ksh"):
        for i, arg in enumerate(args):
            if arg == "-c" or (arg.startswith("-") and "c" in arg[1:] and not arg.startswith("--")):
                if i + 1 < len(args):
                    sub_v = detect_bash_workspace_mutation(
                        args[i + 1], ws_root, out_dir, cwd=active_cwd, env=env, _depth=depth + 1
                    )
                    return sub_v, active_cwd
        non_flags = [a for a in args if not a.startswith("-")]
        if non_flags:
            return {
                "blocked": True,
                "reason": (
                    f"Strict orchestrator mode blocked shell script execution `{binary} {non_flags[0]}`. "
                    f"Delegate workspace modifications via delegate_start."
                ),
            }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

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

    # Check ruff / npm / npx / make / pytest
    if binary == "ruff":
        if any(a in ("--fix", "--unsafe-fixes", "format") for a in args):
            return {
                "blocked": True,
                "reason": "Strict orchestrator mode blocked `ruff` mutating/formatting flags.",
            }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

    if binary in ("npm", "pnpm", "yarn", "npx"):
        if binary == "npx" or any(
            a in ("install", "i", "add", "remove", "uninstall", "update", "ci", "link", " audit", "fmt", "format")
            for a in args
        ):
            return {
                "blocked": True,
                "reason": f"Strict orchestrator mode blocked `{binary} {' '.join(args[:2])}`.",
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

    if binary in ("pytest", "py.test"):
        for parg in args:
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
                }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

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

    # Check python / node / ruby / perl / php inline scripts and script files
    if binary.startswith(("python", "pypy")) or binary in ("node", "ruby", "perl", "php"):
        for i, arg in enumerate(args):
            if arg in ("-c", "-e", "-r", "--eval") and i + 1 < len(args):
                code_str = args[i + 1]
                if binary in ("ruby", "perl", "php") or _INLINE_MUTATION_KEYWORDS_RE.search(code_str):
                    return {
                        "blocked": True,
                        "reason": (
                            f"Strict orchestrator mode blocked `{binary} {arg}` containing file/system mutation. "
                            f"Delegate workspace edits via delegate_start or delegate_to_gemini."
                        ),
                    }, active_cwd
                return {"blocked": False, "reason": None}, active_cwd
            if arg == "-m" and i + 1 < len(args):
                mod = args[i + 1]
                if mod in ("unittest", "pytest", "json.tool", "py_compile", "compileall"):
                    for mod_arg in args[i + 2 :]:
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
            is_allowed_test_runner = (
                script_cls == "workspace"
                and (
                    (script_name == "manage.py" and len(non_flags) >= 2 and non_flags[1] == "test")
                    or bool(re.match(r"^(?:run_tests\.py|test_.*\.py|.*_test\.py)$", script_name))
                )
            )
            if is_allowed_test_runner:
                return {"blocked": False, "reason": None}, active_cwd
            return {
                "blocked": True,
                "reason": (
                    f"Strict orchestrator mode blocked executing script `{binary} {script_arg}`. "
                    f"Use `python3 -m unittest` / `pytest` for tests or delegate edits to Vertex."
                ),
            }, active_cwd
        return {"blocked": False, "reason": None}, active_cwd

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
        # Even in git diff/show/log, block --output=<file> targeting workspace/out_dir
        for sarg in sub_args:
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
        elif binary in ("cp", "install", "rsync", "ln") and len(positional) >= 2:
            candidate_targets.append(positional[-1])
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


def evaluate_pre_tool_use(payload: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    """Evaluate a Claude Code PreToolUse event and return hookSpecificOutput if blocked."""
    source_env = env if env is not None else os.environ
    tool_name = str(payload.get("tool_name") or "")
    tool_input = dict(payload.get("tool_input") or {})
    cwd = str(payload.get("cwd") or os.getcwd())
    ws_root = resolve_project_root(cwd, env=source_env)
    out_dir = source_env.get("CC_AGY_DATA_DIR") or source_env.get("CLAUDE_PLUGIN_DATA")

    # 1. Always lint delegation prompts (including follow_up_prompt) for >=9-line code dictation
    if tool_name.endswith(("delegate_start", "delegate_to_gemini", "delegate_continue")):
        prompt_text = str(
            tool_input.get("prompt") or tool_input.get("follow_up_prompt") or ""
        )
        lint = lint_delegation_prompt(prompt_text)
        if not lint["ok"]:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": lint["reason"],
                }
            }

    if not is_strict_mode_enabled(source_env):
        return {}

    # 2a. Block Read / Grep / Glob on out_dir (.hmac_key / receipts)
    if tool_name in ("Read", "Grep", "Glob"):
        target_path = (
            tool_input.get("file_path")
            or tool_input.get("path")
            or tool_input.get("pattern")
            or ""
        )
        if target_path:
            raw_t = str(target_path)
            classification = classify_path(raw_t, ws_root, out_dir, cwd=cwd, env=source_env)
            if classification == "out_dir" or ".hmac_key" in raw_t:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Strict orchestrator mode is active: {tool_name} on internal receipt/key store "
                            f"{target_path!r} is disabled. Use delegate_status or get_diff instead."
                        ),
                    }
                }

    # 2b. Block direct Edit / Write / MultiEdit / NotebookEdit on both workspace and out_dir (receipts)
    if tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        file_path = (
            tool_input.get("file_path")
            or tool_input.get("notebook_path")
            or tool_input.get("path")
            or ""
        )
        if file_path:
            classification = classify_path(
                str(file_path), ws_root, out_dir, cwd=cwd, env=source_env
            )
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

    # 3. Block Bash mutations targeting workspace or out_dir
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
