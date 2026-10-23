"""Run verification commands for file tasks in a job copy.

Each command runs in its own process group with vendor credentials removed from the environment and
is killed as a group when it overruns its timeout (and after it exits, so stray background children
never outlive the check). Output is kept as a bounded tail; the full log is written next to the job
so it stays addressable when the tail is not enough.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import re
import threading
from typing import Any
from pathlib import Path
import signal
import subprocess
import time
import uuid

TAIL_CHARS = 3_600
SECRET_ENV_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "CC_AGY_VERTEX_BASE_URL_TOKEN",
    "ANTIGRAVITY_OAUTH_TOKEN",
    "CC_AGY_ENTERPRISE_ACCESS_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)


@dataclass
class CommandResult:
    command: str
    passed: bool
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    tail: str
    log_path: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "command": self.command,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "tail": self.tail,
            "log_path": self.log_path,
        }


def scrubbed_env(cwd: Path) -> dict[str, str]:
    """Environment for check commands: no vendor credentials or plugin internals, repo venv first on PATH."""
    from server.jobs import _build_workdir_exec_env

    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("CC_AGY_", "CLAUDE_PLUGIN_")) and k not in SECRET_ENV_KEYS
    }
    for key in SECRET_ENV_KEYS:
        env[key] = ""
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return _build_workdir_exec_env(cwd, env)


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _drain(stream: Any, log_handle: Any, tail: deque[str]) -> None:
    """Copy a process's output to the log file while keeping only a bounded tail in memory."""
    kept = 0
    for chunk in iter(lambda: stream.read(8192), ""):
        if log_handle is not None:
            log_handle.write(chunk)
        tail.append(chunk)
        kept += len(chunk)
        while kept - len(tail[0]) > TAIL_CHARS and len(tail) > 1:
            kept -= len(tail.popleft())


def _untracked(cwd: Path) -> set[str]:
    res = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=str(cwd),
                         capture_output=True, check=False)
    if res.returncode != 0:
        return set()
    return {
        p for p in res.stdout.decode("utf-8", "replace").split("\0")
        if p
        and not p.startswith((".ctx-session-reads/", ".cc-agy-review/"))
        and p not in (".ctx-session-reads", ".cc-agy-review")
    }


def run_commands(
    commands: list[str] | tuple[str, ...],
    cwd: Path,
    *,
    timeout: float = 300.0,
    stop_on_failure: bool = True,
    log_dir: Path | None = None,
    clean_new_files: bool = True,
) -> list[CommandResult]:
    """Run commands in order inside cwd; stop at the first failure unless stop_on_failure is False.

    Output streams to a log file (when log_dir is set) and only a bounded tail is held in memory.
    Untracked files the commands create (caches, reports) are removed afterwards so they never
    reach the job's diff.
    """
    results: list[CommandResult] = []
    env = scrubbed_env(cwd)
    before = _untracked(cwd) if clean_new_files else set()
    try:
        for cmd in commands:
            started = time.time()
            log_path = ""
            log_handle = None
            if log_dir is not None:
                log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                log_file = log_dir / f"check-{uuid.uuid4().hex[:10]}.log"
                log_handle = open(log_file, "w", encoding="utf-8")
                log_path = str(log_file)
            tail: deque[str] = deque()
            proc = subprocess.Popen(
                ["bash", "-c", cmd], cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", start_new_session=True,
            )
            reader = threading.Thread(target=_drain, args=(proc.stdout, log_handle, tail), daemon=True)
            reader.start()
            timed_out = False
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc.pid)
                proc.wait()
            _kill_group(proc.pid)
            reader.join(timeout=10)
            if log_handle is not None:
                log_handle.close()
            out = "".join(tail)
            result = CommandResult(
                command=cmd,
                passed=(not timed_out) and proc.returncode == 0,
                exit_code=None if timed_out else proc.returncode,
                timed_out=timed_out,
                duration_seconds=round(time.time() - started, 2),
                tail=out[-TAIL_CHARS:],
                log_path=log_path,
            )
            results.append(result)
            if stop_on_failure and not result.passed:
                break
    finally:
        if clean_new_files:
            for rel in _untracked(cwd) - before:
                target = cwd / rel
                if target.is_file() or target.is_symlink():
                    target.unlink(missing_ok=True)
    return results


_ERROR_LINE_RE = re.compile(
    r"(Traceback \(most recent call last\)|^E\s|Error[:\s]|error[:\[]|FAILED|FAIL:|AssertionError|panic:|undefined|cannot find)",
    re.MULTILINE,
)


_SYNTAX_ONLY_RE = re.compile(
    r"^\s*(?:python3?\s+-m\s+(?:py_compile|compileall)|ruff\b|flake8\b|black\b|isort\b|mypy\b|pyright\b|"
    r"(?:npx\s+)?(?:tsc|eslint|prettier|biome)\b|gofmt\b|go\s+vet\b|cargo\s+check\b|rustfmt\b|test\s+-[efds]\b)",
    re.I,
)

_BEHAVIORAL_TEST_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*(?:(?:uv|poetry|pipenv)\s+run\s+|bundle\s+exec\s+|env\s+)*(?:\S+/)?(?:"
    r"pytest\b|py\.test\b|python[0-9.]*\s+-m\s+(?:pytest|unittest)\b|"
    r"tox\b|nox\b|go\s+test\b|cargo\s+(?:test|nextest)\b|"
    r"(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test(?::[a-z0-9_:-]+)?\b|"
    r"(?:npx\s+)?(?:vitest|jest|mocha|ava|playwright|cypress)\b|"
    r"rspec\b|rake\s+test\b|"
    r"mvn\s+test\b|gradle(?:w)?\s+test\b|ctest\b|dotnet\s+test\b|bats\b|phpunit\b"
    r")",
    re.I,
)


_NON_EXEC_TEST_FLAG_RE = re.compile(
    r"(?:^|\s)(?:--collect-only|--co|--setup-only|--no-run|--dry-run|--listTests|--fixtures|--markers|--version|-V|--help|-h|-run=\^\$)(?:\s|$)"
    r"|\bgo\s+test\b[^\n;|&]*\s(?:-c|-list)(?:\s|$)"
)
_EXIT_MASK_RE = re.compile(
    r"\|\|\s*(?:true\b|:(?=\s|$)|exit\s+0\b|echo\b)|;\s*(?:exit\s+0\b|true\b|:(?=\s|$))",
    re.I,
)


def is_behavioral_check(cmd: str) -> bool:
    """Return True if ``cmd`` executes a behavioral test runner rather than syntax/type/lint checks."""
    text = (cmd or "").strip()
    if not text or text.startswith("#") or _SYNTAX_ONLY_RE.match(text):
        return False
    if _EXIT_MASK_RE.search(text) or _NON_EXEC_TEST_FLAG_RE.search(text):
        return False
    has_pipefail = bool(re.search(r"\bset\s+-o\s+pipefail\b", text))
    has_errexit = bool(re.search(r"\bset\s+-[a-z]*e", text))
    semi_parts = [s.strip() for s in re.split(r";|\n", text) if s.strip() and not s.strip().startswith("#")]
    if not semi_parts:
        return False
    candidate_semi = semi_parts if has_errexit else [semi_parts[-1]]
    for semi_seg in candidate_semi:
        for and_seg in semi_seg.split("&&"):
            pipe_parts = [p.strip() for p in re.split(r"(?<!\|)\|(?!\|)", and_seg) if p.strip()]
            if not pipe_parts:
                continue
            if len(pipe_parts) > 1 and not has_pipefail:
                continue
            cleaned = pipe_parts[0].lstrip("(").strip()
            if cleaned and not cleaned.startswith("#") and _BEHAVIORAL_TEST_RE.match(cleaned):
                return True
    return False


def has_behavioral_check(cmds: list[str] | tuple[str, ...]) -> bool:
    return any(is_behavioral_check(c) for c in cmds)


def failure_tail(results: list[CommandResult], max_chars: int = 1_200) -> str:
    """Short text for the first failing command: the first error block plus the closing summary."""
    for res in results:
        if res.passed:
            continue
        head = f"$ {res.command}\n" + ("(timed out)\n" if res.timed_out else f"(exit {res.exit_code})\n")
        body = res.tail
        budget = max(200, max_chars - len(head))
        match = _ERROR_LINE_RE.search(body)
        if match and len(body) > budget:
            first = body[match.start(): match.start() + budget * 2 // 3]
            summary = body[-(budget - len(first)):]
            return head + first + ("\n...\n" + summary if summary and summary not in first else "")
        return head + body[-budget:]
    return ""
