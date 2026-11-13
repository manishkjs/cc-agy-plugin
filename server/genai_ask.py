"""Single-shot Gemini query helper (`ask_gemini`) via `google-genai`, SSE long-decode, or `agy`.

Supports:
- `gemini-4-carbon` (`carbon`) sequential `continuationToken` long-decode (up to 8 continuations,
  `maxOutputTokens = max(max_output_tokens * 3, 98_304)`, empty-output `MAX_TOKENS` budget doubling,
  and per-segment `usage["requests"]` tracking for accurate >200K / <=200K pricing).
- Workspace-confined multimodal attachments (`media`: PNG, JPEG, WebP, HEIC, HEIF, MP4, MOV, WebM,
  MPEG, PDF up to 14 MB total).
- SSRF-hardened HTTPS specification fetching (`spec_url`) with DNS/IP public-host validation on every
  redirect hop, inline PDF attachment, and linear-time HTML text extraction.
- Live Google Search grounding (`google_search`) and batched parallel dependency/CVE auditing
  (`check_dependencies`, 12 packages/batch, 6 parallel workers, auto-splitting on `TOO_MANY_TOOL_CALLS`).
- Outbound secret file blocking (`is_secret_file`), linear-time secret content redaction
  (`redact_secrets`), hostile `.git/config` filter neutralization (`_git`), and priority-ordered
  round-robin directory/glob context collection (`file_priority`, `fair_order`).
"""

from __future__ import annotations

import asyncio
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import html.parser
import ipaddress
import json
import math
import os
from pathlib import Path, PurePath
import re
import shutil
import socket
import subprocess
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request
import uuid

from server.genai_client import (
    build_genai_client,
    close_genai_client as _close_genai_client,
    extract_usage_metadata,
)
from server.receipts import (
    ReceiptStore,
    enforce_daily_budget,
    resolve_workspace_root,
    validate_job_id,
)
from server.regions import (
    LONG_DECODE_MODELS,
    MAX_TOOL_RESPONSE_TOKENS,
    clamp_gemini_thinking_level,
    estimate_cost_usd,
    estimate_tokens,
    is_long_decode_model,
    normalize_wire_model_name,
    resolve_ask_thinking_level,
    resolve_auth_config,
    resolve_default_ask_model,
    resolve_effective_gateway_url,
    resolve_gateway_credentials,
    resolve_gateway_wire,
    sanitize_placeholder_env,
    truncate_to_token_budget,
)

GenaiBackend = Callable[[dict[str, Any]], dict[str, Any]]

DEFAULT_ASK_MAX_OUTPUT_TOKENS = 32_768
LONG_DECODE_MIN_OUTPUT_TOKENS = 98_304
MAX_STREAM_SEGMENT_OUTPUT_TOKENS = 65_536
MAX_CONTINUATIONS = 8
MAX_BUDGET_RETRY_OUTPUT_TOKENS = 262_144
DEFAULT_ASK_CONTEXT_TOKEN_BUDGET = 700_000
MAX_FILE_BYTES = 2_000_000
MAX_MEDIA_BYTES = 14_000_000
MAX_SPEC_BYTES = 8_000_000
DEPS_BATCH_SIZE = 12
DEPS_PARALLEL = 6

MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mpeg": "video/mpeg",
    ".pdf": "application/pdf",
}

VIDEO_MEDIA_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".webm", ".mpeg"})

SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        "__pycache__",
        ".venv",
        "venv",
        "vendor",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "coverage",
        ".Output",
        ".cc-agy-data",
        ".worktrees",
        ".ctx-session-reads",
        ".cc-agy-review",
    }
)

SECRET_FILE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*credentials*.json",
    "*service-account*.json",
    "*service_account*.json",
    "*secret*",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "id_dsa*",
    ".netrc",
    "_netrc",
    ".npmrc",
    ".pypirc",
    ".pgpass",
    "*.tfstate",
    "*.tfstate.*",
    "*.tfvars",
    "kubeconfig*",
)

SECRET_FILE_ALLOW: frozenset[str] = frozenset(
    {
        ".env.example",
        ".env.sample",
        ".env.template",
        "env.example",
    }
)

SECRET_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".ssh",
        ".gnupg",
        ".aws",
        ".azure",
        ".kube",
        ".docker",
        ".config/gcloud",
        "gcloud",
    }
)

_PEM_HEADER_RE = re.compile(r"-----BEGIN [A-Z0-9 _-]{0,40}PRIVATE KEY-----")
_SECRET_NAME_RE = re.compile(
    r"(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|"
    r"client[_-]?secret|password|passwd|private[_-]?key|bearer|x-api-key)\s*$",
    re.IGNORECASE,
)
_SECRET_PLACEHOLDERS: frozenset[str] = frozenset(
    {"true", "false", "null", "none", "undefined", "your_api_key", "changeme", "xxx", "placeholder"}
)

SECRET_CONTENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bAIza[0-9A-Za-z\-_]{31,45}\b"), "[REDACTED:GOOGLE_API_KEY]"),
    (re.compile(r"\bya29\.[0-9A-Za-z\-_]{20,512}\b"), "[REDACTED:GOOGLE_OAUTH_TOKEN]"),
    (re.compile(r"\bsk-ant-[0-9A-Za-z\-_]{16,256}\b"), "[REDACTED:ANTHROPIC_KEY]"),
    (re.compile(r"\bsk-(?:proj-|live-|test-)?[0-9A-Za-z\-_]{20,256}\b"), "[REDACTED:API_KEY]"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[0-9A-Za-z]{20,255}\b"), "[REDACTED:GITHUB_TOKEN]"),
    (re.compile(r"\bgithub_pat_[0-9A-Za-z_]{20,255}\b"), "[REDACTED:GITHUB_PAT]"),
    (re.compile(r"\bglpat-[0-9A-Za-z\-_]{20,255}\b"), "[REDACTED:GITLAB_PAT]"),
    (re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,255}\b"), "[REDACTED:SLACK_TOKEN]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED:AWS_ACCESS_KEY_ID]"),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,512}\.eyJ[A-Za-z0-9_-]{8,1024}\.[A-Za-z0-9_-]{8,512}\b"),
        "[REDACTED:JWT]",
    ),
    (
        re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://[^/\s:@]{1,128}:)[^@\s/]{4,256}@"),
        r"\1[REDACTED]@",
    ),
)

_GIT_SAFE_ARGS: tuple[str, ...] = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "diff.external=",
    "-c",
    "protocol.file.allow=never",
)

ENTRY_POINT_NAMES: frozenset[str] = frozenset(
    {
        "main.py",
        "app.py",
        "server.py",
        "cli.py",
        "__init__.py",
        "__main__.py",
        "index.ts",
        "index.tsx",
        "index.js",
        "main.ts",
        "main.js",
        "server.ts",
        "app.ts",
        "main.go",
        "main.rs",
        "lib.rs",
        "mod.rs",
        "Program.cs",
        "Main.java",
        "Application.java",
    }
)

SOURCE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".scala",
        ".sh",
        ".sql",
        ".proto",
        ".graphql",
    }
)

CONFIG_DOC_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".md",
        ".rst",
        ".txt",
        ".ini",
        ".cfg",
    }
)

LOW_PRIORITY_MARKERS: tuple[str, ...] = (
    "test",
    "tests",
    "__tests__",
    "spec",
    "specs",
    "fixture",
    "fixtures",
    "mock",
    "mocks",
    "example",
    "examples",
    "bench",
    "benchmark",
    "docs",
    "migrations",
    "generated",
)

MANIFEST_PATTERNS: tuple[str, ...] = (
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "requirements-*.txt",
    "requirements/*.txt",
    "Pipfile",
    "Cargo.toml",
    "go.mod",
    "Gemfile",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "composer.json",
)


def is_secret_file(rel_path: str) -> bool:
    """Return True if `rel_path` matches a secret/credential file or directory pattern."""
    norm = rel_path.replace("\\", "/")
    p = PurePath(norm)
    name = p.name.lower()
    if name in SECRET_FILE_ALLOW:
        return False
    parts_lower = [part.lower() for part in p.parts[:-1]]
    for part in parts_lower:
        if part in SECRET_DIRS:
            return True
    rel_dir = "/".join(parts_lower)
    if any(sd in rel_dir for sd in SECRET_DIRS if "/" in sd):
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in SECRET_FILE_PATTERNS)


def _redact_private_keys(text: str) -> tuple[str, int]:
    """Linear-time redaction for PEM private key blocks."""
    out: list[str] = []
    pos = 0
    count = 0
    while True:
        m = _PEM_HEADER_RE.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        out.append(text[pos : m.start()])
        end_idx = text.find("-----END ", m.end())
        if end_idx != -1:
            tail_dashes = text.find("-----", end_idx + 9)
            pos = (tail_dashes + 5) if tail_dashes != -1 else len(text)
        else:
            pos = len(text)
        out.append("[REDACTED:PRIVATE_KEY]")
        count += 1
    return ("".join(out) if count else text), count


def _redact_secret_assignments(text: str) -> tuple[str, int]:
    """Line-by-line assignment redaction (`api_key = "..."`) without backtracking."""
    if "\n" not in text and len(text) > 4096:
        return text, 0
    out_lines: list[str] = []
    count = 0
    for line in text.splitlines(keepends=True):
        if len(line) > 1024:
            out_lines.append(line)
            continue
        sep_idx = -1
        for ch in ("=", ":"):
            idx = line.find(ch)
            if idx != -1 and (sep_idx == -1 or idx < sep_idx):
                sep_idx = idx
        if sep_idx <= 0:
            out_lines.append(line)
            continue
        lhs = line[:sep_idx].rstrip(" \"'")
        if not _SECRET_NAME_RE.search(lhs):
            out_lines.append(line)
            continue
        rhs = line[sep_idx + 1 :].lstrip()
        if not rhs:
            out_lines.append(line)
            continue
        if rhs[0] in ('"', "'"):
            q = rhs[0]
            end_q = rhs.find(q, 1)
            val = rhs[1:end_q] if end_q != -1 else ""
            tail = rhs[end_q + 1 :] if end_q != -1 else ""
        else:
            tok_end = len(rhs)
            for i, c in enumerate(rhs):
                if c in (" ", "\t", "\r", "\n", ",", ";", "#", ")", "}"):
                    tok_end = i
                    break
            val = rhs[:tok_end]
            tail = rhs[tok_end:]
        if (
            len(val) >= 12
            and val.lower() not in _SECRET_PLACEHOLDERS
            and not val.startswith(("$", "{", "<", "[", "os.", "process.", "env.", " settings.", "config."))
            and any(c.isalnum() for c in val)
        ):
            out_lines.append(f"{line[: sep_idx + 1]} [REDACTED:SECRET_VALUE]{tail}")
            count += 1
        else:
            out_lines.append(line)
    return ("".join(out_lines) if count else text), count


def redact_secrets(text: str) -> tuple[str, int]:
    """Scrub secret patterns from `text`, returning `(redacted_text, redactions_count)`."""
    if (
        os.environ.get("CC_AGY_REDACT_SECRETS", "1").strip() == "0"
        or os.environ.get("GEMINI4_REDACT_SECRETS", "1").strip() == "0"
    ):
        return text, 0
    total = 0
    text, n = _redact_private_keys(text)
    total += n
    for pattern, replacement in SECRET_CONTENT_PATTERNS:
        text, n = pattern.subn(replacement, text)
        total += n
    text, n = _redact_secret_assignments(text)
    total += n
    return text, total


def _git_dir_trusted(ws_root: Path) -> bool:
    """Verify `.git` is not a symlink or a `.git` file pointing outside `ws_root`."""
    git_entry = ws_root / ".git"
    try:
        if git_entry.is_symlink():
            return False
        if git_entry.is_dir():
            return True
        if git_entry.is_file():
            raw = git_entry.read_text(encoding="utf-8", errors="replace").strip()
            if not raw.lower().startswith("gitdir:"):
                return False
            target = (ws_root / raw.split(":", 1)[1].strip()).resolve()
            target.relative_to(ws_root.resolve())
            return True
    except (OSError, ValueError):
        return False
    return False


def _git_filter_overrides(ws_root: Path) -> dict[str, str]:
    """Build `GIT_CONFIG_*` overrides neutralizing any repo `.git/config` `filter.*` drivers."""
    env_Pairs: list[tuple[str, str]] = []
    dot_git = ws_root / ".git"
    candidates: list[Path] = []
    if dot_git.is_dir() and not dot_git.is_symlink():
        candidates.extend([dot_git / "config", dot_git / "config.worktree"])
    elif dot_git.is_file() and not dot_git.is_symlink():
        try:
            raw = dot_git.read_text(encoding="utf-8", errors="replace").strip()
            if raw.lower().startswith("gitdir:"):
                gitdir = (ws_root / raw.split(":", 1)[1].strip()).resolve()
                gitdir.relative_to(ws_root.resolve())
                candidates.extend([gitdir / "config", gitdir / "config.worktree", gitdir.parent.parent / "config"])
        except (OSError, ValueError):
            pass
    filter_names: set[str] = set()
    for cfg in candidates:
        try:
            if cfg.is_file() and not cfg.is_symlink():
                text = cfg.read_text(encoding="utf-8", errors="replace")
                for m in re.finditer(r'\[filter\s+"([^"\n]+)"\]', text, re.IGNORECASE):
                    filter_names.add(m.group(1))
        except OSError:
            pass
    for name in sorted(filter_names):
        env_Pairs.append((f"filter.{name}.clean", ""))
        env_Pairs.append((f"filter.{name}.smudge", ""))
        env_Pairs.append((f"filter.{name}.process", ""))
        env_Pairs.append((f"filter.{name}.required", "false"))
    if not env_Pairs:
        return {}
    out = {"GIT_CONFIG_COUNT": str(len(env_Pairs))}
    for i, (k, v) in enumerate(env_Pairs):
        out[f"GIT_CONFIG_KEY_{i}"] = k
        out[f"GIT_CONFIG_VALUE_{i}"] = v
    return out


def _git(ws_root: Path, args: list[str], timeout: int = 5) -> subprocess.CompletedProcess[str] | None:
    """Run a read-only git command inside `ws_root` with hostile-config protections."""
    if not _git_dir_trusted(ws_root):
        return None
    try:
        env = {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            **_git_filter_overrides(ws_root),
        }
        return subprocess.run(
            ["git", *_GIT_SAFE_ARGS, "-C", str(ws_root), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None


def _in_git_repo(ws_root: Path) -> bool:
    res = _git(ws_root, ["rev-parse", "--is-inside-work-tree"])
    return bool(res and res.returncode == 0 and res.stdout.strip() == "true")


def _inside_git_dir(path: Path, ws_root: Path) -> bool:
    try:
        rel = path.relative_to(ws_root)
    except ValueError:
        return True
    return any(part == ".git" for part in rel.parts)


def file_priority(rel_path: str) -> tuple[int, int, str]:
    """Rank workspace files so entry points & core source precede configs, docs, and tests."""
    p = PurePath(rel_path.replace("\\", "/"))
    parts_lower = [part.lower() for part in p.parts]
    name = p.name
    ext = p.suffix.lower()
    depth = len(p.parts)

    if any(marker in parts_lower for marker in LOW_PRIORITY_MARKERS) or name.lower().startswith("test_"):
        return (3, depth, rel_path)
    if name.endswith((".lock", "-lock.json", ".min.js", ".min.css", ".map")):
        return (3, depth, rel_path)
    if name in ENTRY_POINT_NAMES and depth <= 3:
        return (0, depth, rel_path)
    if ext in SOURCE_EXTENSIONS:
        return (1, depth, rel_path)
    if ext in CONFIG_DOC_EXTENSIONS:
        return (2, depth, rel_path)
    return (2, depth, rel_path)


def fair_order(rel_paths: list[str]) -> list[str]:
    """Order files by priority tier and round-robin across top-level directories within each tier."""
    by_tier: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for rel in sorted(set(rel_paths), key=file_priority):
        tier = file_priority(rel)[0]
        parts = PurePath(rel.replace("\\", "/")).parts
        top_dir = parts[0] if len(parts) > 1 else "."
        by_tier[tier][top_dir].append(rel)

    ordered: list[str] = []
    for tier in sorted(by_tier):
        buckets = by_tier[tier]
        dir_keys = sorted(buckets.keys(), key=lambda d: (0 if d == "." else 1, d))
        idx = 0
        while True:
            added = False
            for d in dir_keys:
                if idx < len(buckets[d]):
                    ordered.append(buckets[d][idx])
                    added = True
            if not added:
                break
            idx += 1
    return ordered


def estimate_source_tokens(text: str) -> int:
    """Character-class token estimator for code and structured source files."""
    if not text:
        return 0
    sample = text if len(text) <= 65536 else text[:32768] + text[-32768:]
    scale = len(text) / len(sample)
    alnum = 0
    punct = 0
    ws = 0
    non_ascii = 0
    for ch in sample:
        code = ord(ch)
        if code > 127:
            non_ascii += 1
        elif ch.isalnum() or ch == "_":
            alnum += 1
        elif ch in (" ", "\t", "\n", "\r"):
            ws += 1
        else:
            punct += 1
    raw = (alnum / 4.2) + (punct / 1.6) + (ws / 6.0) + (non_ascii / 1.5)
    return max(1, int(raw * scale))


def _resolve_safe_workspace_target(ws_root: Path, raw_path: str, label: str = "context_file") -> Path:
    rel_str = str(raw_path or "").strip()
    if not rel_str:
        raise ValueError(f"{label} entries must be non-empty strings")
    cand = Path(rel_str)
    if cand.is_absolute():
        raise ValueError(f"{label} must be relative to workspace_root: {rel_str}")
    if ".." in cand.parts:
        raise ValueError(f"{label} escapes workspace_root: {rel_str}")
    joined = ws_root / cand
    if joined.is_symlink():
        resolved_link = joined.resolve()
        if not resolved_link.is_relative_to(ws_root):
            raise ValueError(f"{label} is a symlink escaping workspace_root: {rel_str}")
    full = joined.resolve()
    if not full.is_relative_to(ws_root):
        raise ValueError(f"{label} escapes workspace_root: {rel_str}")
    return full


def collect_context_files(
    ws_root: Path,
    raw_paths: list[str],
    token_budget: int | None = None,
    *,
    strict_missing: bool = False,
) -> tuple[str, list[str], list[str], int]:
    """Expand files/directories/globs inside `ws_root`, scrub secrets, and pack within `token_budget`.

    Returns `(formatted_context, included_rel_paths, skipped_notes, total_redactions)`.
    """
    ws_resolved = ws_root.resolve()
    if token_budget is None:
        raw_env_budget = (
            os.environ.get("CC_AGY_CONTEXT_TOKEN_BUDGET")
            or os.environ.get("GEMINI4_CONTEXT_TOKEN_BUDGET")
            or str(DEFAULT_ASK_CONTEXT_TOKEN_BUDGET)
        )
        try:
            token_budget = max(1, int(raw_env_budget))
        except ValueError:
            token_budget = DEFAULT_ASK_CONTEXT_TOKEN_BUDGET

    in_git = _in_git_repo(ws_resolved)
    include_ignored = (
        os.environ.get("CC_AGY_INCLUDE_IGNORED", "0").strip() == "1"
        or os.environ.get("GEMINI4_INCLUDE_IGNORED", "0").strip() == "1"
    )
    expanded: list[Path] = []
    skipped: list[str] = []

    for raw in raw_paths:
        rel_str = str(raw or "").strip()
        if not rel_str:
            raise ValueError("context_files entries must be non-empty strings")
        has_glob = any(ch in rel_str for ch in "*?[")
        if has_glob:
            cand_glob = Path(rel_str)
            if cand_glob.is_absolute() or ".." in cand_glob.parts:
                raise ValueError(f"context_file escapes workspace_root: {rel_str}")
            for match in sorted(ws_resolved.glob(rel_str)):
                if match.is_file() and not match.is_symlink():
                    try:
                        resolved_match = match.resolve()
                        rel_match = resolved_match.relative_to(ws_resolved)
                        if any(part in SKIP_DIRS or part in SECRET_DIRS for part in rel_match.parts):
                            continue
                        if not _inside_git_dir(resolved_match, ws_resolved):
                            expanded.append(resolved_match)
                    except ValueError:
                        skipped.append(f"{rel_str} (outside workspace)")
            continue

        target = _resolve_safe_workspace_target(ws_resolved, rel_str, label="context_file")
        rel_norm = target.relative_to(ws_resolved).as_posix()
        if _inside_git_dir(target, ws_resolved):
            skipped.append(f"{rel_norm} (skipped: .git directory)")
            continue
        if is_secret_file(rel_norm):
            skipped.append(f"{rel_norm} (blocked: secret/credential pattern)")
            continue
        if not target.exists():
            if strict_missing:
                raise ValueError(f"context_file does not exist: {rel_str}")
            skipped.append(f"{rel_str} (not found)")
            continue
        if target.is_file():
            expanded.append(target)
        elif target.is_dir():
            rel_dir = rel_norm
            git_files = (
                _git(
                    ws_resolved,
                    [
                        "ls-files",
                        "-z",
                        "--cached",
                        "--others",
                        "--exclude-standard",
                        "--",
                        "." if rel_dir in ("", ".") else rel_dir,
                    ],
                )
                if (in_git and not include_ignored)
                else None
            )
            if git_files and git_files.returncode == 0:
                for rel_item in git_files.stdout.split("\0"):
                    if not rel_item:
                        continue
                    fp = (ws_resolved / rel_item).resolve()
                    try:
                        fp.relative_to(ws_resolved)
                    except ValueError:
                        continue
                    if (ws_resolved / rel_item).is_symlink():
                        continue
                    if any(part in SKIP_DIRS for part in PurePath(rel_item).parts):
                        continue
                    if fp.is_file():
                        expanded.append(fp)
            else:
                for root_dir, dirs, files in os.walk(target, followlinks=False):
                    dirs[:] = sorted(
                        d
                        for d in dirs
                        if d not in SKIP_DIRS
                        and d not in SECRET_DIRS
                        and not (Path(root_dir) / d).is_symlink()
                    )
                    for fname in sorted(files):
                        fp = Path(root_dir) / fname
                        if fp.is_symlink():
                            continue
                        try:
                            fp.resolve().relative_to(ws_resolved)
                            expanded.append(fp.resolve())
                        except ValueError:
                            continue

    rel_to_path: dict[str, Path] = {}
    for fp in expanded:
        try:
            rel_posix = fp.relative_to(ws_resolved).as_posix()
        except ValueError:
            continue
        rel_to_path[rel_posix] = fp

    ordered_rels = fair_order(list(rel_to_path.keys()))
    candidate_rels: list[str] = []
    for rel_posix in ordered_rels:
        if is_secret_file(rel_posix):
            skipped.append(f"{rel_posix} (blocked: secret/credential pattern)")
            continue
        candidate_rels.append(rel_posix)

    ignored_set: set[str] = set()
    if in_git and not include_ignored and candidate_rels:
        try:
            env = {
                **os.environ,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
                **_git_filter_overrides(ws_resolved),
            }
            proc = subprocess.run(
                ["git", *_GIT_SAFE_ARGS, "-C", str(ws_resolved), "check-ignore", "--stdin", "-z"],
                input="\0".join(candidate_rels).encode("utf-8"),
                capture_output=True,
                timeout=5,
                check=False,
                env=env,
            )
            if proc.returncode in (0, 1) and proc.stdout:
                ignored_set = {
                    item.decode("utf-8", errors="replace")
                    for item in proc.stdout.split(b"\0")
                    if item
                }
        except (OSError, subprocess.SubprocessError):
            ignored_set = set()

    sections: list[str] = []
    included: list[str] = []
    used_tokens = 0
    total_redactions = 0

    for rel_posix in candidate_rels:
        if rel_posix in ignored_set:
            skipped.append(f"{rel_posix} (skipped: gitignored)")
            continue
        fp = rel_to_path[rel_posix]
        try:
            size = fp.stat().st_size
        except OSError:
            continue
        if size > MAX_FILE_BYTES:
            skipped.append(f"{rel_posix} (skipped: {size:,} bytes > {MAX_FILE_BYTES:,})")
            continue
        try:
            with fp.open("rb") as fh:
                head = fh.read(8192)
                if b"\x00" in head:
                    skipped.append(f"{rel_posix} (skipped: binary)")
                    continue
                raw_bytes = head + fh.read(MAX_FILE_BYTES - len(head) + 1)
            if len(raw_bytes) > MAX_FILE_BYTES:
                skipped.append(f"{rel_posix} (skipped: >{MAX_FILE_BYTES:,} bytes)")
                continue
            content = raw_bytes.decode("utf-8", errors="replace")
        except OSError as exc:
            skipped.append(f"{rel_posix} (read error: {exc})")
            continue

        content, red_count = redact_secrets(content)
        total_redactions += red_count
        file_tokens = max(estimate_source_tokens(content), estimate_tokens(content))
        if used_tokens + file_tokens > token_budget:
            skipped.append(f"{rel_posix} (skipped: token budget {token_budget:,} reached)")
            continue
        used_tokens += file_tokens
        included.append(rel_posix)
        sections.append(f"### File: {rel_posix}\n```\n{content}\n```")

    return "\n\n".join(sections), included, skipped, total_redactions


def collect_media_files(ws_root: Path, media_paths: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Load workspace-confined image, video, and PDF files as Gemini `inlineData` parts."""
    ws_resolved = ws_root.resolve()
    parts: list[dict[str, Any]] = []
    attached: list[str] = []
    total_bytes = 0
    in_git = _in_git_repo(ws_resolved)
    include_ignored = (
        os.environ.get("CC_AGY_INCLUDE_IGNORED", "0").strip() == "1"
        or os.environ.get("GEMINI4_INCLUDE_IGNORED", "0").strip() == "1"
    )

    for raw in media_paths:
        rel_str = str(raw or "").strip()
        if not rel_str:
            continue
        target = _resolve_safe_workspace_target(ws_resolved, rel_str, label="media")
        if _inside_git_dir(target, ws_resolved):
            raise ValueError(f"Refusing to attach file inside .git: {rel_str}")
        if not target.is_file():
            raise ValueError(f"Media file not found: {rel_str}")
        rel_posix = target.relative_to(ws_resolved).as_posix()
        if is_secret_file(rel_posix):
            raise ValueError(f"Refusing to attach secret-pattern file: {rel_posix}")
        if in_git and not include_ignored:
            ign = _git(ws_resolved, ["check-ignore", "-q", "--", rel_posix])
            if ign and ign.returncode == 0:
                raise ValueError(
                    f"Refusing to attach gitignored file: {rel_posix} (set CC_AGY_INCLUDE_IGNORED=1 to override)"
                )
        mime = MEDIA_TYPES.get(target.suffix.lower())
        if not mime:
            supported = ", ".join(sorted(MEDIA_TYPES))
            raise ValueError(f"Unsupported media type '{target.suffix}' for {rel_str}. Supported: {supported}")
        size = target.stat().st_size
        if total_bytes + size > MAX_MEDIA_BYTES:
            raise ValueError(f"Total media size exceeds {MAX_MEDIA_BYTES // 1_000_000} MB limit at {rel_posix}")
        with target.open("rb") as fh:
            raw_bytes = fh.read(MAX_MEDIA_BYTES - total_bytes + 1)
        if len(raw_bytes) + total_bytes > MAX_MEDIA_BYTES:
            raise ValueError(f"Total media size exceeds {MAX_MEDIA_BYTES // 1_000_000} MB limit at {rel_posix}")
        total_bytes += len(raw_bytes)
        parts.append(
            {
                "inlineData": {
                    "mimeType": mime,
                    "data": base64.b64encode(raw_bytes).decode("ascii"),
                }
            }
        )
        attached.append(rel_posix)
    return parts, attached


_MEDIA_EXT_PATTERN = "|".join(re.escape(ext.lstrip(".")) for ext in sorted(MEDIA_TYPES))
_PROMPT_MEDIA_PATH_RE = re.compile(
    rf"(?<![A-Za-z0-9_://.-])([A-Za-z0-9_./-]+\.(?:{_MEDIA_EXT_PATTERN}))(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)


def extract_workspace_media_paths(ws_root: Path, *texts: str) -> list[str]:
    """Extract workspace-relative multimodal paths referenced in `texts` that exist on disk."""
    ws_resolved = ws_root.resolve()
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in _PROMPT_MEDIA_PATH_RE.finditer(text):
            raw_cand = match.group(1).strip().lstrip("./")
            if not raw_cand or ".." in Path(raw_cand).parts or Path(raw_cand).is_absolute():
                continue
            cand_path = ws_resolved / raw_cand
            try:
                if cand_path.is_symlink():
                    continue
                resolved = cand_path.resolve()
                if not resolved.is_relative_to(ws_resolved) or not resolved.is_file():
                    continue
                rel_posix = resolved.relative_to(ws_resolved).as_posix()
                if is_secret_file(rel_posix) or _inside_git_dir(resolved, ws_resolved):
                    continue
                if rel_posix not in seen:
                    seen.add(rel_posix)
                    found.append(rel_posix)
            except OSError:
                continue
    return found


def _resolve_public_ips(hostname: str, port: int = 443) -> list[str]:
    """Resolve `hostname` and return all IPs only when every resolved IP is public/global."""
    if not hostname or hostname.lower() in ("localhost", "localhost.localdomain"):
        return []
    try:
        infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return []
    if not infos:
        return []
    ips: list[str] = []
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return []
        if not (ip.is_global and not ip.is_multicast):
            return []
        if addr not in ips:
            ips.append(addr)
    return ips


def _public_host(hostname: str) -> bool:
    """Resolve `hostname` and return True only when every resolved IP is public/global."""
    return bool(_resolve_public_ips(hostname))


import http.client as _http_client
import ssl as _ssl


class _PinnedPublicHTTPSConnection(_http_client.HTTPSConnection):
    """HTTPS connection that pins TCP socket creation to verified public IPs (anti-DNS-rebinding)."""

    def connect(self) -> None:
        target_host = self._tunnel_host if self._tunnel_host else self.host
        target_port = (self._tunnel_port or 443) if self._tunnel_host else (self.port or 443)
        ips = _resolve_public_ips(target_host, target_port)
        if not ips:
            raise urllib.error.URLError(f"Refusing connection to non-public host: {target_host}")
        if self._tunnel_host:
            super().connect()
            return
        last_err: OSError | None = None
        raw_sock = None
        for ip in ips:
            try:
                raw_sock = socket.create_connection((ip, self.port or 443), self.timeout, self.source_address)
                break
            except OSError as exc:
                last_err = exc
        if raw_sock is None:
            raise urllib.error.URLError(last_err or f"Unable to connect to {self.host}")
        self.sock = self._context.wrap_socket(raw_sock, server_hostname=self.host)


class _PinnedPublicHTTPSHandler(urllib.request.HTTPSHandler):
    """HTTPS handler that uses `_PinnedPublicHTTPSConnection`."""

    def https_open(self, req):  # type: ignore[override]
        return self.do_open(_PinnedPublicHTTPSConnection, req, context=self._context)


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    """Re-verify HTTPS scheme, no credentials, and public IP on every redirect hop."""

    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme.lower() != "https":
            raise urllib.error.URLError(f"Refusing redirect to non-https URL: {newurl}")
        if parsed.username or parsed.password:
            raise urllib.error.URLError("Refusing redirect URL containing credentials")
        if not parsed.hostname or not _public_host(parsed.hostname):
            raise urllib.error.URLError(f"Refusing redirect to non-public host: {parsed.hostname}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _TextExtractor(html.parser.HTMLParser):
    """Linear-time HTML visible-text extractor with zero regex backtracking."""

    _SKIP_TAGS: frozenset[str] = frozenset({"script", "style", "noscript", "svg", "template"})
    _BLOCK_TAGS: frozenset[str] = frozenset(
        {"p", "div", "section", "article", "header", "footer", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "br", "hr"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self._skip_depth += 1
        elif t in self._BLOCK_TAGS and self._skip_depth == 0:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif t in self._BLOCK_TAGS and self._skip_depth == 0:
            self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self.chunks.append(data)


def _html_to_text(raw_html: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw_html)
    parser.close()
    lines = [" ".join(line.split()) for line in "".join(parser.chunks).splitlines()]
    return "\n".join(line for line in lines if line)


def fetch_spec_url(
    url: str,
    *,
    opener: Any = None,
) -> tuple[str, dict[str, Any] | None, str, int]:
    """Fetch a public HTTPS specification URL safely, returning `(text, pdf_part, final_url, redactions)`."""
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme.lower() != "https":
        raise ValueError("spec_url must use https://")
    if parsed.username or parsed.password:
        raise ValueError("spec_url must not contain credentials")
    if not parsed.hostname or not _public_host(parsed.hostname):
        raise ValueError(f"Refusing non-public or unresolvable host: {parsed.hostname}")

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "cc-agy-plugin/0.3 (+https://github.com/manishkjs/cc-agy-plugin)",
            "Accept": "text/markdown, text/plain, text/html, application/pdf;q=0.9, */*;q=0.5",
        },
        method="GET",
    )
    active_opener = opener or urllib.request.build_opener(
        _PinnedPublicHTTPSHandler(context=_ssl.create_default_context()),
        _CheckedRedirect,
    )
    with active_opener.open(req, timeout=20) as resp:
        final_url = getattr(resp, "geturl", lambda: url)()
        content_type = (resp.headers.get("Content-Type") or "").lower()
        data = resp.read(MAX_SPEC_BYTES + 1)
    if len(data) > MAX_SPEC_BYTES:
        raise ValueError(f"Specification at {url} exceeds {MAX_SPEC_BYTES // 1_000_000} MB limit")

    if "application/pdf" in content_type or final_url.lower().endswith(".pdf") or data.startswith(b"%PDF-"):
        if len(data) > MAX_MEDIA_BYTES:
            raise ValueError("Specification PDF exceeds 14 MB inline media limit")
        return (
            "",
            {
                "inlineData": {
                    "mimeType": "application/pdf",
                    "data": base64.b64encode(data).decode("ascii"),
                }
            },
            final_url,
            0,
        )

    text = data.decode("utf-8", errors="replace")
    if "text/html" in content_type or "<html" in text[:1000].lower() or "<!doctype html" in text[:1000].lower():
        text = _html_to_text(text)
    text, red_count = redact_secrets(text)
    return text[:120_000], None, final_url, red_count


def find_manifests(ws_root: Path) -> list[str]:
    """Auto-discover dependency manifests inside `ws_root` (up to depth 4)."""
    ws_resolved = ws_root.resolve()
    found: list[str] = []
    for root_dir, dirs, files in os.walk(ws_resolved, followlinks=False):
        rel_root = Path(root_dir).relative_to(ws_resolved)
        if len(rel_root.parts) > 4:
            dirs[:] = []
            continue
        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP_DIRS and d not in SECRET_DIRS and not (Path(root_dir) / d).is_symlink()
        ]
        for fname in files:
            if any(fnmatch.fnmatch(fname, pat) for pat in MANIFEST_PATTERNS):
                fp = Path(root_dir) / fname
                if not fp.is_symlink():
                    found.append(fp.relative_to(ws_resolved).as_posix())
    return sorted(found)


def _safe_spec(spec: str) -> str:
    """Scrub URL/VCS/path dependency specs so private tokens or internal hosts never enter search queries."""
    s = spec.strip()
    if any(tok in s for tok in ("://", "@", "git+", "file:", "../", "./")) or s.startswith("/"):
        return "<url or path>"
    return s[:64]


def parse_declared_dependencies(rel_path: str, content: str) -> list[str]:
    """Extract compact `'ecosystem:name @ version (rel_path)'` strings from a manifest file."""
    name = PurePath(rel_path.replace("\\", "/")).name
    deps: list[str] = []

    if name == "package.json":
        try:
            pkg = json.loads(content)
            for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
                for k, v in (pkg.get(section) or {}).items():
                    if isinstance(v, str):
                        deps.append(f"npm:{k} @ {_safe_spec(v)} ({rel_path})")
        except json.JSONDecodeError:
            pass
    elif name.startswith("requirements") and name.endswith(".txt"):
        for raw in content.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith(("-", "--", "git+", "http:", "https:")):
                continue
            if " @ " in line:
                pkg_name = line.split(" @ ", 1)[0].strip()
                deps.append(f"pypi:{pkg_name} @ <url or path> ({rel_path})")
            else:
                deps.append(f"pypi:{line} ({rel_path})")
    elif name == "pyproject.toml":
        for m in re.finditer(r'^\s*"([A-Za-z0-9_.\-]+(?:\[[^\]]+\])?\s*[^"]*)"\s*,?\s*$', content, re.M):
            spec = m.group(1).strip()
            if " @ " in spec:
                pkg_name = spec.split(" @ ", 1)[0].strip()
                deps.append(f"pypi:{pkg_name} @ <url or path> ({rel_path})")
            else:
                deps.append(f"pypi:{spec} ({rel_path})")
        in_poetry = False
        for raw in content.splitlines():
            s = raw.strip()
            if s.startswith("["):
                in_poetry = s in (
                    "[tool.poetry.dependencies]",
                    "[tool.poetry.dev-dependencies]",
                    "[tool.poetry.group.dev.dependencies]",
                )
                continue
            if in_poetry and "=" in s and not s.startswith("#"):
                k, v = [p.strip().strip("\"'") for p in s.split("=", 1)]
                if k.lower() != "python":
                    deps.append(f"pypi:{k} @ {_safe_spec(v)} ({rel_path})")
    elif name == "Cargo.toml":
        in_deps = False
        for raw in content.splitlines():
            s = raw.strip()
            if s.startswith("["):
                in_deps = s.endswith("dependencies]")
                continue
            if in_deps and "=" in s and not s.startswith("#"):
                k, v = [p.strip() for p in s.split("=", 1)]
                vm = re.search(r'"([^"]+)"', v)
                ver = _safe_spec(vm.group(1)) if vm else "<workspace>"
                deps.append(f"crates:{k} @ {ver} ({rel_path})")
    elif name == "go.mod":
        for m in re.finditer(r"^\s*([a-zA-Z0-9_.\-/]+)\s+(v[0-9][^\s/]*)", content, re.M):
            mod, ver = m.group(1), m.group(2)
            if not mod.startswith(("module", "go", "toolchain", "replace", "exclude", "retract")):
                deps.append(f"go:{mod} @ {ver} ({rel_path})")
    elif name == "Gemfile":
        for m in re.finditer(r"""^\s*gem\s+['"]([^'"]+)['"](?:\s*,\s*['"]([^'"]+)['"])?""", content, re.M):
            deps.append(f"rubygems:{m.group(1)} @ {_safe_spec(m.group(2) or '*')} ({rel_path})")
    elif name == "composer.json":
        try:
            comp = json.loads(content)
            for section in ("require", "require-dev"):
                for k, v in (comp.get(section) or {}).items():
                    if isinstance(v, str) and k != "php" and not k.startswith("ext-"):
                        deps.append(f"packagist:{k} @ {_safe_spec(v)} ({rel_path})")
        except json.JSONDecodeError:
            pass
    elif name in ("build.gradle", "build.gradle.kts"):
        for m in re.finditer(
            r"""(?:implementation|api|compileOnly|runtimeOnly|testImplementation)\s*\(?['"]([^:'"]+):([^:'"]+):([^'"]+)['"]""",
            content,
        ):
            deps.append(f"maven:{m.group(1)}:{m.group(2)} @ {_safe_spec(m.group(3))} ({rel_path})")
    elif name == "pom.xml":
        for m in re.finditer(
            r"<dependency>\s*<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>"
            r"(?:\s*<version>([^<]+)</version>)?",
            content,
            re.S,
        ):
            deps.append(f"maven:{m.group(1).strip()}:{m.group(2).strip()} @ {_safe_spec((m.group(3) or '*').strip())} ({rel_path})")

    return deps


def _extract_grounding(raw_obj: Any) -> tuple[list[dict[str, str]], list[str]]:
    """Extract `(sources, search_queries)` from either a REST dict or a `google-genai` response object."""
    sources: list[dict[str, str]] = []
    queries: list[str] = []
    seen_uris: set[str] = set()

    if isinstance(raw_obj, dict):
        # Could be already extracted or raw Gemini JSON
        if isinstance(raw_obj.get("sources"), list):
            for s in raw_obj["sources"]:
                if isinstance(s, dict) and s.get("uri") and s["uri"] not in seen_uris:
                    seen_uris.add(str(s["uri"]))
                    sources.append({"title": str(s.get("title") or ""), "uri": str(s["uri"])})
        if isinstance(raw_obj.get("search_queries"), list):
            for q in raw_obj["search_queries"]:
                if isinstance(q, str) and q not in queries:
                    queries.append(q)
        for cand in raw_obj.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            gm = cand.get("groundingMetadata") or cand.get("grounding_metadata") or {}
            if not isinstance(gm, dict):
                continue
            for q in gm.get("webSearchQueries") or gm.get("web_search_queries") or []:
                if isinstance(q, str) and q not in queries:
                    queries.append(q)
            for chunk in gm.get("groundingChunks") or gm.get("grounding_chunks") or []:
                if not isinstance(chunk, dict):
                    continue
                web = chunk.get("web") or {}
                uri = web.get("uri") if isinstance(web, dict) else None
                if uri and uri not in seen_uris:
                    seen_uris.add(str(uri))
                    sources.append({"title": str(web.get("title") or ""), "uri": str(uri)})
        return sources, queries

    for cand in getattr(raw_obj, "candidates", None) or []:
        gm = getattr(cand, "grounding_metadata", None) or getattr(cand, "groundingMetadata", None)
        if gm is None:
            continue
        for q in getattr(gm, "web_search_queries", None) or getattr(gm, "webSearchQueries", None) or []:
            if isinstance(q, str) and q not in queries:
                queries.append(q)
        for chunk in getattr(gm, "grounding_chunks", None) or getattr(gm, "groundingChunks", None) or []:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None) if web is not None else None
            if uri and str(uri) not in seen_uris:
                seen_uris.add(str(uri))
                sources.append({"title": str(getattr(web, "title", "") or ""), "uri": str(uri)})
    return sources, queries


def _parse_generate_response(raw: bytes) -> dict[str, Any]:
    """Parse SSE (`alt=sse`) or single JSON `generateContent` response, merging text, usage, grounding, and `continuationToken`."""
    raw_str = raw.decode("utf-8", errors="replace").strip()
    if raw_str.startswith("{"):
        return json.loads(raw_str)

    events: list[dict[str, Any]] = []
    for line in raw_str.splitlines():
        s = line.strip()
        if s.startswith("data:"):
            payload = s[5:].strip()
            if payload and payload != "[DONE]":
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
    if not events:
        return json.loads(raw_str)

    merged_parts: list[dict[str, Any]] = []
    finish_reason: str | None = None
    continuation_token: str | None = None
    grounding_chunks: list[dict[str, Any]] = []
    search_queries: list[str] = []
    usage: dict[str, Any] = {}
    model_version: str | None = None

    for ev in events:
        if ev.get("modelVersion"):
            model_version = ev["modelVersion"]
        if ev.get("usageMetadata"):
            usage = ev["usageMetadata"]
        if ev.get("continuationToken"):
            continuation_token = ev["continuationToken"]
        for cand in ev.get("candidates") or []:
            if cand.get("finishReason"):
                finish_reason = cand["finishReason"]
            if cand.get("continuationToken"):
                continuation_token = cand["continuationToken"]
            gm = cand.get("groundingMetadata") or {}
            grounding_chunks.extend(gm.get("groundingChunks") or [])
            for q in gm.get("webSearchQueries") or []:
                if q not in search_queries:
                    search_queries.append(q)
            for p in (cand.get("content") or {}).get("parts") or []:
                merged_parts.append(p)

    cand_out: dict[str, Any] = {
        "content": {"role": "model", "parts": merged_parts},
        "finishReason": finish_reason or "STOP",
    }
    if continuation_token:
        cand_out["continuationToken"] = continuation_token
    if grounding_chunks or search_queries:
        cand_out["groundingMetadata"] = {
            "groundingChunks": grounding_chunks,
            "webSearchQueries": search_queries,
        }
    out: dict[str, Any] = {"candidates": [cand_out], "usageMetadata": usage}
    if model_version:
        out["modelVersion"] = model_version
    if continuation_token:
        out["continuationToken"] = continuation_token
    return out


def _normalize_usage_dict(raw_usage: dict[str, Any] | None) -> dict[str, int]:
    u = raw_usage or {}
    p_tok = int(u.get("prompt_token_count") or u.get("promptTokenCount") or 0)
    c_tok = int(u.get("cached_content_token_count") or u.get("cachedContentTokenCount") or 0)
    o_tok = int(u.get("candidates_token_count") or u.get("candidatesTokenCount") or 0)
    t_tok = int(u.get("thoughts_token_count") or u.get("thoughtsTokenCount") or 0)
    tot_tok = int(u.get("total_token_count") or u.get("totalTokenCount") or (p_tok + o_tok + t_tok))
    return {
        "prompt_token_count": p_tok,
        "cached_content_token_count": c_tok,
        "candidates_token_count": o_tok,
        "thoughts_token_count": t_tok,
        "total_token_count": tot_tok,
    }


def _build_sse_endpoint_and_headers(req: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Build the `v1beta` / `v1beta1` `:streamGenerateContent?alt=sse` URL and headers for `req`."""
    auth_mode = str(req.get("auth_mode") or "")
    gateway_url = resolve_effective_gateway_url(req)
    wire = resolve_gateway_wire(req) if gateway_url else ""
    custom_gw = bool(gateway_url)
    wire_model = normalize_wire_model_name(str(req.get("model") or ""), custom_gateway=custom_gw)
    safe_model = urllib.parse.quote(wire_model, safe="-_.")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    gw_token, gw_extra_headers = resolve_gateway_credentials(req) if custom_gw else (None, {})
    if gw_extra_headers:
        headers.update(gw_extra_headers)
    if gw_token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {gw_token}"

    if auth_mode == "api_key" or (custom_gw and wire == "gemini"):
        base = (gateway_url or "https://generativelanguage.googleapis.com").rstrip("/")
        url = f"{base}/v1beta/models/{safe_model}:streamGenerateContent?alt=sse"
        api_key = str(req.get("api_key") or os.environ.get("GEMINI_API_KEY") or "").strip()
        if api_key:
            headers["x-goog-api-key"] = api_key
        return url, headers

    project = urllib.parse.quote(str(req.get("project") or ""), safe="-_.")
    location = urllib.parse.quote(str(req.get("location") or "global"), safe="-_.")
    if custom_gw:
        base = gateway_url.rstrip("/")
    elif location == "global":
        base = "https://aiplatform.googleapis.com"
    else:
        base = f"https://{location}-aiplatform.googleapis.com"
    url = (
        f"{base}/v1beta1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{safe_model}:streamGenerateContent?alt=sse"
    )
    return url, headers


def call_gemini_long_decode(
    req: dict[str, Any] | None = None,
    *,
    endpoint_url: str | None = None,
    headers: dict[str, str] | None = None,
    request_body: dict[str, Any] | None = None,
    max_continuations: int = MAX_CONTINUATIONS,
    timeout: float = 180.0,
    http_transport: Callable[[dict[str, Any], int], bytes | dict[str, Any]] | None = None,
    _sse_poster: Callable[[str, dict[str, str], dict[str, Any], float], list[dict[str, Any]] | dict[str, Any] | bytes] | None = None,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Execute a Gemini request with sequential `continuationToken` SSE decoding and empty-`MAX_TOKENS` retry."""
    eff_timeout = float(timeout_seconds if timeout == 180.0 else timeout)
    if request_body is not None:
        model = str((req or {}).get("model") or "gemini-4-carbon")
        base_gen_cfg = dict(request_body.get("generationConfig") or {})
        max_output_tokens = int(base_gen_cfg.get("maxOutputTokens") or MAX_STREAM_SEGMENT_OUTPUT_TOKENS)

        def _build_body(cur_max_out: int, cont_token: str | None) -> dict[str, Any]:
            b = json.loads(json.dumps(request_body))
            b.setdefault("generationConfig", {})["maxOutputTokens"] = cur_max_out
            if cont_token:
                b["continuationToken"] = cont_token
            else:
                b.pop("continuationToken", None)
            return b
    else:
        req = req or {}
        model = str(req.get("model") or "")
        base_max = int(req.get("max_output_tokens") or DEFAULT_ASK_MAX_OUTPUT_TOKENS)
        if is_long_decode_model(model):
            max_output_tokens = max(base_max * 3, LONG_DECODE_MIN_OUTPUT_TOKENS)
        else:
            max_output_tokens = base_max

        parts: list[dict[str, Any]] = []
        for mp in req.get("media_parts") or []:
            if isinstance(mp, dict):
                parts.append(mp)
        parts.append({"text": str(req.get("prompt") or "")})

        def _build_body(cur_max_out: int, cont_token: str | None) -> dict[str, Any]:
            gen_cfg: dict[str, Any] = {
                "temperature": req.get("temperature") if req.get("temperature") is not None else 0.2,
                "maxOutputTokens": min(cur_max_out, MAX_STREAM_SEGMENT_OUTPUT_TOKENS)
                if is_long_decode_model(model)
                else cur_max_out,
            }
            thinking_lvl = req.get("thinking_level")
            if thinking_lvl:
                clamped = clamp_gemini_thinking_level(model, str(thinking_lvl))
                if clamped:
                    if str(clamped).upper() == "MINIMAL" and "carbon" in model.lower():
                        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
                    else:
                        gen_cfg["thinkingConfig"] = {"thinkingLevel": str(clamped).upper()}
            body_d: dict[str, Any] = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": gen_cfg,
            }
            sys_inst = req.get("system_instruction")
            if sys_inst:
                body_d["systemInstruction"] = {"parts": [{"text": str(sys_inst)}]}
            if req.get("google_search"):
                body_d["tools"] = [{"googleSearch": {}}]
            if cont_token:
                body_d["continuationToken"] = cont_token
            return body_d

    texts: list[str] = []
    sources: list[dict[str, str]] = []
    seen_uris: set[str] = set()
    search_queries: list[str] = []
    request_usages: list[dict[str, Any]] = []
    total_in = 0
    total_cached = 0
    total_out = 0
    total_think = 0
    total_tok = 0
    finish_reason = "STOP"
    token: str | None = None
    segments = 0
    budget_retry_done = False

    while True:
        segments += 1
        req_body = _build_body(max_output_tokens, token)
        if _sse_poster is not None:
            raw_events = _sse_poster(
                endpoint_url or "",
                headers or {"Content-Type": "application/json"},
                req_body,
                eff_timeout,
            )
            if isinstance(raw_events, bytes):
                body = _parse_generate_response(raw_events)
            elif isinstance(raw_events, list):
                sse_lines = "\n".join(f"data: {json.dumps(ev)}" for ev in raw_events).encode("utf-8")
                body = _parse_generate_response(sse_lines)
            else:
                body = dict(raw_events)
        elif http_transport is not None:
            raw_or_dict = http_transport(req_body, int(eff_timeout))
            body = _parse_generate_response(raw_or_dict) if isinstance(raw_or_dict, bytes) else dict(raw_or_dict)
        else:
            url = endpoint_url
            req_headers = headers
            if not url or req_headers is None:
                url, req_headers = _build_sse_endpoint_and_headers(req or {})
            http_req = urllib.request.Request(
                url,
                data=json.dumps(req_body).encode("utf-8"),
                headers=req_headers,
                method="POST",
            )
            with urllib.request.urlopen(http_req, timeout=eff_timeout) as resp:
                body = _parse_generate_response(resp.read())

        cands = body.get("candidates") or []
        cand = cands[0] if cands else {}
        for part in (cand.get("content") or {}).get("parts") or []:
            if not part.get("thought") and "text" in part:
                texts.append(str(part["text"]))

        seg_sources, seg_queries = _extract_grounding(body)
        for s in seg_sources:
            if s["uri"] not in seen_uris:
                seen_uris.add(s["uri"])
                sources.append(s)
        for q in seg_queries:
            if q not in search_queries:
                search_queries.append(q)

        norm_u = _normalize_usage_dict(body.get("usageMetadata") or body.get("usage"))
        total_in += norm_u["prompt_token_count"]
        total_cached += norm_u["cached_content_token_count"]
        total_out += norm_u["candidates_token_count"]
        total_think += norm_u["thoughts_token_count"]
        total_tok += norm_u["total_token_count"]
        request_usages.append(
            {
                **norm_u,
                "per_call": True,
                "segment": segments,
            }
        )

        finish_reason = str(cand.get("finishReason") or body.get("finish_reason") or "STOP")
        next_token = cand.get("continuationToken") or body.get("continuationToken") or body.get("continuation_token")

        if finish_reason == "CONTINUATION" and next_token and segments <= max_continuations:
            token = str(next_token)
            continue
        if finish_reason == "CONTINUATION" and segments > max_continuations:
            finish_reason = "MAX_TOKENS"
            break
        if (
            finish_reason == "MAX_TOKENS"
            and "".join(texts).strip() == ""
            and not budget_retry_done
            and max_output_tokens < MAX_BUDGET_RETRY_OUTPUT_TOKENS
        ):
            budget_retry_done = True
            max_output_tokens = min(max_output_tokens * 2, MAX_BUDGET_RETRY_OUTPUT_TOKENS)
            token = str(next_token) if next_token else None
            continue
        break

    combined_usage: dict[str, Any] = {
        "prompt_token_count": total_in,
        "cached_content_token_count": total_cached,
        "candidates_token_count": total_out,
        "thoughts_token_count": total_think,
        "total_token_count": total_tok,
    }
    if len(request_usages) > 1 or is_long_decode_model(model):
        combined_usage["requests"] = request_usages

    return {
        "text": "".join(texts),
        "usage": combined_usage,
        "finish_reason": finish_reason,
        "continuations": max(0, segments - 1),
        "max_output_tokens": max_output_tokens,
        "sources": sources,
        "search_queries": search_queries,
    }


def _default_genai_backend(request: dict[str, Any]) -> dict[str, Any]:
    """Invoke Gemini Enterprise (`agy` CLI), Google AI Studio, or Vertex AI."""
    sanitize_placeholder_env()
    if request.get("auth_mode") == "gemini_enterprise":
        if request.get("google_search"):
            raise RuntimeError(
                "Live Google Search grounding (`google_search` / `check_dependencies`) is not available "
                "on the standalone `agy` CLI (`--auth gemini-enterprise`). Omit `google_search` / "
                "`check_dependencies`, or switch to `--auth api-key`, `--auth vertex`, or `--auth gateway`."
            )
        from server.agy_cli import run_agy_cli_ask

        return run_agy_cli_ask(request)

    from google.genai import types as genai_types

    timeout_ms = int(request.get("timeout_ms") or 600_000)
    req_tl = str(request.get("thinking_level") or "").strip().upper().replace("-", "_")
    if req_tl.startswith("THINKING_LEVEL_"):
        req_tl = req_tl[len("THINKING_LEVEL_") :]
    is_zero_thinking = req_tl in ("ZERO", "0", "OFF")
    client = build_genai_client(
        request,
        timeout_ms=timeout_ms,
        zero_thinking=is_zero_thinking,
    )
    try:
        cfg_kwargs: dict[str, Any] = {}
        if request.get("system_instruction"):
            cfg_kwargs["system_instruction"] = request["system_instruction"]
        if request.get("temperature") is not None:
            cfg_kwargs["temperature"] = float(request["temperature"])
        if request.get("max_output_tokens") is not None:
            cfg_kwargs["max_output_tokens"] = int(request["max_output_tokens"])
        eff_gw_url = resolve_effective_gateway_url(request)
        custom_gateway = bool(eff_gw_url)
        wire_model = normalize_wire_model_name(
            str(request["model"]), custom_gateway=custom_gateway
        )
        if is_zero_thinking:
            cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=0
            )
        elif req_tl:
            clamped_tl = clamp_gemini_thinking_level(
                str(request.get("model") or wire_model),
                req_tl,
                allow_extra_high=False,
            )
            if clamped_tl:
                cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_level=clamped_tl
                )
        if request.get("google_search"):
            tool_ctor = getattr(genai_types, "Tool", None)
            gs_ctor = getattr(genai_types, "GoogleSearch", None)
            if tool_ctor is not None and gs_ctor is not None:
                try:
                    cfg_kwargs["tools"] = [tool_ctor(google_search=gs_ctor())]
                except Exception:
                    cfg_kwargs["tools"] = [{"google_search": {}}]

        config = genai_types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None
        media_parts = request.get("media_parts") or []
        if media_parts:
            part_ctor = getattr(genai_types, "Part", None)
            from_bytes_fn = getattr(part_ctor, "from_bytes", None) if part_ctor is not None else None
            sdk_contents: list[Any] = []
            for mp in media_parts:
                inline = (mp.get("inlineData") or {}) if isinstance(mp, dict) else {}
                b64_data = inline.get("data")
                mime_type = inline.get("mimeType")
                if b64_data and mime_type:
                    raw_b = base64.b64decode(b64_data)
                    if callable(from_bytes_fn):
                        sdk_contents.append(from_bytes_fn(data=raw_b, mime_type=mime_type))
                    else:
                        sdk_contents.append(mp)
            sdk_contents.append(request["prompt"])
            contents_arg: Any = sdk_contents
        else:
            contents_arg = request["prompt"]

        response = client.models.generate_content(
            model=wire_model,
            contents=contents_arg,
            config=config,
        )
        text_out = getattr(response, "text", "") or ""
        finish_reason = "STOP"
        cands = getattr(response, "candidates", None) or []
        if cands:
            raw_fr = getattr(cands[0], "finish_reason", None) or getattr(cands[0], "finishReason", None)
            if raw_fr is not None:
                finish_reason = str(getattr(raw_fr, "name", raw_fr))
        sources, search_queries = _extract_grounding(response)
        out: dict[str, Any] = {
            "text": text_out,
            "usage": extract_usage_metadata(getattr(response, "usage_metadata", None)),
            "finish_reason": finish_reason,
        }
        if sources:
            out["sources"] = sources
        if search_queries:
            out["search_queries"] = search_queries
        return out
    finally:
        _close_genai_client(client)


def _run_ask_with_continuations(
    req: dict[str, Any],
    backend_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Run `backend_fn(req)` with multi-segment `continuationToken` and empty-`MAX_TOKENS` retry support."""
    model = str(req.get("model") or "")
    explicit_max = req.get("max_output_tokens")
    if explicit_max is not None and int(explicit_max) > 0:
        max_output_tokens = int(explicit_max)
    elif is_long_decode_model(model):
        max_output_tokens = max(DEFAULT_ASK_MAX_OUTPUT_TOKENS * 3, LONG_DECODE_MIN_OUTPUT_TOKENS)
    else:
        max_output_tokens = DEFAULT_ASK_MAX_OUTPUT_TOKENS

    texts: list[str] = []
    sources: list[dict[str, str]] = []
    seen_uris: set[str] = set()
    search_queries: list[str] = []
    request_usages: list[dict[str, Any]] = []
    total_in = 0
    total_cached = 0
    total_out = 0
    total_think = 0
    total_tok = 0
    finish_reason = "STOP"
    token: str | None = None
    segments = 0
    budget_retry_done = False
    resumed_conversation = False

    while True:
        segments += 1
        hop_req = dict(req)
        hop_req["max_output_tokens"] = (
            min(max_output_tokens, MAX_STREAM_SEGMENT_OUTPUT_TOKENS)
            if (is_long_decode_model(model) and explicit_max is None and not budget_retry_done)
            else max_output_tokens
        )
        if token is not None:
            hop_req["continuation_token"] = token
            hop_req["continuationToken"] = token

        raw_res = backend_fn(hop_req) or {}
        resumed_conversation = resumed_conversation or bool(raw_res.get("resumed_conversation"))
        seg_text = str(raw_res.get("text") or "")
        if seg_text:
            texts.append(seg_text)

        seg_sources, seg_queries = _extract_grounding(raw_res)
        for s in seg_sources:
            if s["uri"] not in seen_uris:
                seen_uris.add(s["uri"])
                sources.append(s)
        for q in seg_queries:
            if q not in search_queries:
                search_queries.append(q)

        raw_u = raw_res.get("usage") or raw_res.get("usageMetadata") or {}
        norm_u = _normalize_usage_dict(raw_u)
        total_in += norm_u["prompt_token_count"]
        total_cached += norm_u["cached_content_token_count"]
        total_out += norm_u["candidates_token_count"]
        total_think += norm_u["thoughts_token_count"]
        total_tok += norm_u["total_token_count"]

        existing_reqs = raw_u.get("requests") if isinstance(raw_u, dict) else None
        if isinstance(existing_reqs, list) and existing_reqs:
            for r_item in existing_reqs:
                if isinstance(r_item, dict):
                    request_usages.append({**_normalize_usage_dict(r_item), "per_call": True})
        else:
            request_usages.append({**norm_u, "per_call": True, "segment": segments})

        finish_reason = str(raw_res.get("finish_reason") or raw_res.get("finishReason") or "STOP")
        next_token = raw_res.get("continuation_token") or raw_res.get("continuationToken")

        if finish_reason == "CONTINUATION" and next_token and segments <= MAX_CONTINUATIONS:
            token = str(next_token)
            continue
        if finish_reason == "CONTINUATION" and segments > MAX_CONTINUATIONS:
            finish_reason = "MAX_TOKENS"
            break
        if (
            finish_reason == "MAX_TOKENS"
            and "".join(texts).strip() == ""
            and not budget_retry_done
            and max_output_tokens < MAX_BUDGET_RETRY_OUTPUT_TOKENS
        ):
            budget_retry_done = True
            max_output_tokens = min(max_output_tokens * 2, MAX_BUDGET_RETRY_OUTPUT_TOKENS)
            token = str(next_token) if next_token else None
            continue
        break

    combined_usage: dict[str, Any] = {
        "prompt_token_count": total_in,
        "cached_content_token_count": total_cached,
        "candidates_token_count": total_out,
        "thoughts_token_count": total_think,
        "total_token_count": total_tok,
    }
    if len(request_usages) > 1 or is_long_decode_model(model):
        combined_usage["requests"] = request_usages

    out: dict[str, Any] = {
        "text": "".join(texts),
        "usage": combined_usage,
        "finish_reason": finish_reason,
        "continuations": max(0, segments - 1),
        "max_output_tokens": max_output_tokens,
    }
    if sources:
        out["sources"] = sources
    if search_queries:
        out["search_queries"] = search_queries
    if resumed_conversation:
        out["resumed_conversation"] = True
    return out


def _deps_batch(
    batch: list[str],
    user_note: str,
    run_single_batch: Callable[..., dict[str, Any]],
    depth: int = 0,
) -> dict[str, Any]:
    """Audit one batch of up to 12 dependencies with live Google Search, splitting in half on `TOO_MANY_TOOL_CALLS`."""
    dep_list = "\n".join(f"- {d}" for d in batch)
    batch_prompt = (
        f"Check every dependency below using Google Search ({len(batch)} packages).\n"
        "For EACH package, find the latest stable version and any known CVEs or security advisories "
        "affecting the declared version.\n\n"
        "Output ONLY Markdown table rows (no header, no separator, no prose before the table) in this exact 6-column format:\n"
        "| package | declared | latest_stable | OK / OUTDATED / VULNERABLE | CVE-IDs or None | concise upgrade note |\n\n"
        "After the rows, if any upgrade has breaking changes, add a bullet under `### Risky`.\n\n"
        f"{user_note}\n\nDependencies:\n{dep_list}"
    )
    try:
        try:
            res = run_single_batch(batch_prompt, batch)
        except TypeError:
            res = run_single_batch(batch_prompt)
    except Exception as exc:
        if "TOO_MANY_TOOL_CALLS" in str(exc) and len(batch) > 1 and depth < 2:
            res = {"finish_reason": "TOO_MANY_TOOL_CALLS"}
        else:
            raise
    finish_reason = str(res.get("finish_reason") or "STOP")
    if finish_reason == "TOO_MANY_TOOL_CALLS" and len(batch) > 1 and depth < 2:
        mid = len(batch) // 2
        left = _deps_batch(batch[:mid], user_note, run_single_batch, depth + 1)
        right = _deps_batch(batch[mid:], user_note, run_single_batch, depth + 1)
        l_u = _normalize_usage_dict(left.get("usage"))
        r_u = _normalize_usage_dict(right.get("usage"))
        merged_reqs = list((left.get("usage") or {}).get("requests") or [l_u]) + list(
            (right.get("usage") or {}).get("requests") or [r_u]
        )
        return {
            "text": (str(left.get("text") or "") + "\n" + str(right.get("text") or "")).strip(),
            "finish_reason": right.get("finish_reason") or left.get("finish_reason") or "STOP",
            "sources": list(left.get("sources") or []) + list(right.get("sources") or []),
            "search_queries": list(left.get("search_queries") or []) + list(right.get("search_queries") or []),
            "usage": {
                "prompt_token_count": l_u["prompt_token_count"] + r_u["prompt_token_count"],
                "cached_content_token_count": l_u["cached_content_token_count"] + r_u["cached_content_token_count"],
                "candidates_token_count": l_u["candidates_token_count"] + r_u["candidates_token_count"],
                "thoughts_token_count": l_u["thoughts_token_count"] + r_u["thoughts_token_count"],
                "total_token_count": l_u["total_token_count"] + r_u["total_token_count"],
                "requests": merged_reqs,
            },
        }
    return res


def check_dependencies_batched(
    ws_root: Path,
    manifest_rels: list[str],
    user_prompt: str,
    run_single_batch: Callable[[str], dict[str, Any]],
) -> dict[str, Any] | None:
    """Parse manifests, check dependencies in parallel batches of 12, and merge into a prioritized CVE table."""
    ws_resolved = ws_root.resolve()
    all_deps: list[str] = []
    for rel in manifest_rels:
        if is_secret_file(rel):
            continue
        fp = (ws_resolved / rel).resolve()
        try:
            fp.relative_to(ws_resolved)
            if fp.is_symlink() or not fp.is_file():
                continue
            content = fp.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        all_deps.extend(parse_declared_dependencies(rel, content))

    if not all_deps:
        return None

    batches = [all_deps[i : i + DEPS_BATCH_SIZE] for i in range(0, len(all_deps), DEPS_BATCH_SIZE)]
    results: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=min(DEPS_PARALLEL, len(batches))) as pool:
        futs = {pool.submit(_deps_batch, b, user_prompt, run_single_batch): idx for idx, b in enumerate(batches)}
        for fut in as_completed(futs):
            idx = futs[fut]
            results.append((idx, fut.result()))
    results.sort(key=lambda x: x[0])

    rows: list[str] = []
    risky: list[str] = []
    sources: list[dict[str, str]] = []
    seen_uris: set[str] = set()
    queries: list[str] = []
    request_usages: list[dict[str, Any]] = []
    total_in = 0
    total_cached = 0
    total_out = 0
    total_think = 0
    total_tok = 0

    for _, r in results:
        u = _normalize_usage_dict(r.get("usage"))
        total_in += u["prompt_token_count"]
        total_cached += u["cached_content_token_count"]
        total_out += u["candidates_token_count"]
        total_think += u["thoughts_token_count"]
        total_tok += u["total_token_count"]
        sub_reqs = (r.get("usage") or {}).get("requests")
        if isinstance(sub_reqs, list) and sub_reqs:
            for sr in sub_reqs:
                if isinstance(sr, dict):
                    request_usages.append({**_normalize_usage_dict(sr), "per_call": True})
        else:
            request_usages.append({**u, "per_call": True})

        for s in r.get("sources") or []:
            if isinstance(s, dict) and s.get("uri") and s["uri"] not in seen_uris:
                seen_uris.add(str(s["uri"]))
                sources.append(s)
        for q in r.get("search_queries") or []:
            if isinstance(q, str) and q not in queries:
                queries.append(q)

        in_risky = False
        for line in str(r.get("text") or "").splitlines():
            s_line = line.strip()
            if s_line.lower().startswith("### risky"):
                in_risky = True
                continue
            if s_line.startswith("|") and s_line.endswith("|") and not s_line.startswith(("|---", "| ---", "| Package", "| package")):
                rows.append(s_line)
            elif in_risky and s_line.startswith(("-", "*")):
                risky.append(s_line)

    urgent = [row for row in rows if "VULNERABLE" in row.upper() or "CVE-" in row.upper() or "GHSA-" in row.upper()]
    rest = [row for row in rows if row not in urgent]
    ordered_rows = urgent + rest

    header = (
        f"Checked **{len(all_deps)}** declared dependencies across **{len(manifest_rels)}** manifest(s) "
        f"in **{len(batches)}** parallel Google Search batch(es) (`{', '.join(manifest_rels)}`).\n\n"
        "| Package | Declared | Latest Stable | Status | CVEs / Advisory | Notes |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = header + "\n".join(ordered_rows)
    if risky:
        body += "\n\n**Risky upgrades**\n" + "\n".join(risky)

    return {
        "text": body,
        "finish_reason": "STOP",
        "continuations": 0,
        "sources": sources,
        "search_queries": queries,
        "batches": len(batches),
        "dependencies_checked": len(all_deps),
        "usage": {
            "prompt_token_count": total_in,
            "cached_content_token_count": total_cached,
            "candidates_token_count": total_out,
            "thoughts_token_count": total_think,
            "total_token_count": total_tok,
            "requests": request_usages,
        },
    }


def ask_gemini(
    prompt: str,
    *,
    model: str | None = None,
    context_files: list[str] | None = None,
    media: list[str] | None = None,
    spec_url: str | None = None,
    google_search: bool = False,
    check_dependencies: bool = False,
    system_instruction: str | None = None,
    temperature: float | None = None,
    thinking_level: str | None = None,
    max_output_tokens: int | None = None,
    timeout_ms: int | None = None,
    project: str | None = None,
    location: str | None = None,
    api_key: str | None = None,
    workspace_root: str | Path | None = None,
    store: ReceiptStore | None = None,
    scrub_receipt: bool = False,
    backend: GenaiBackend | None = None,
    http_transport: Callable[[dict[str, Any], int], bytes | dict[str, Any]] | None = None,
    record_receipt: bool = True,
    budget_job_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run a single-shot or multi-segment long-decode Gemini query and record a signed receipt.

    On a Gemini Enterprise seat, a follow-up question in the same host session
    (`session_id`, defaulting to this server process) on the same context files resumes
    the earlier `agy` conversation and sends only the new question.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    if context_files is not None and (
        not isinstance(context_files, list)
        or not all(isinstance(p, str) for p in context_files)
    ):
        raise ValueError("context_files must be a list of relative file path strings")

    ws_root = resolve_workspace_root(workspace_root)
    receipt_store = store or ReceiptStore(workspace_root=ws_root)
    enforce_daily_budget(receipt_store, workspace_root=ws_root)

    chosen_model = resolve_default_ask_model(model, env=os.environ)
    effective_thinking = thinking_level
    if effective_thinking is None and is_long_decode_model(chosen_model):
        effective_thinking = resolve_ask_thinking_level(None, chosen_model, env=os.environ)

    auth = resolve_auth_config(
        requested_project=project,
        requested_location=location,
        requested_api_key=api_key,
        model=chosen_model,
    )
    auth_mode = auth["auth_mode"]
    resolved_key = auth.get("api_key")
    resolved_project = auth.get("project")
    resolved_location = auth.get("location")
    reroute_note = auth.get("reroute_note")

    redacted_prompt, prompt_redactions = redact_secrets(prompt.strip())
    total_redactions = prompt_redactions
    redacted_sys: str | None = None
    if system_instruction:
        redacted_sys, sys_redactions = redact_secrets(system_instruction)
        total_redactions += sys_redactions

    effective_ctx_inputs = list(context_files or [])
    if check_dependencies and not effective_ctx_inputs:
        effective_ctx_inputs = find_manifests(ws_root)

    formatted_ctx = ""
    included_files: list[str] = []
    skipped_files: list[str] = []
    if effective_ctx_inputs:
        formatted_ctx, included_files, skipped_files, ctx_redactions = collect_context_files(
            ws_root,
            effective_ctx_inputs,
            strict_missing=not check_dependencies,
        )
        total_redactions += ctx_redactions

    media_parts: list[dict[str, Any]] = []
    attached_media: list[str] = []
    if media:
        media_parts, attached_media = collect_media_files(ws_root, media)

    final_spec_url: str | None = None
    spec_text = ""
    if spec_url and str(spec_url).strip():
        spec_text, spec_pdf_part, final_spec_url, spec_redactions = fetch_spec_url(str(spec_url).strip())
        total_redactions += spec_redactions
        if spec_pdf_part is not None:
            media_parts.append(spec_pdf_part)

    assembled_parts: list[str] = []
    if formatted_ctx:
        assembled_parts.append(formatted_ctx)
    if spec_text and final_spec_url:
        assembled_parts.append(f"=== SPECIFICATION ({final_spec_url}) ===\n{spec_text}\n=== END SPECIFICATION ===")
    stable_prefix = "\n\n".join(assembled_parts)
    assembled_parts.append(redacted_prompt)
    full_prompt = "\n\n".join(assembled_parts)

    effective_google_search = bool(google_search or check_dependencies)

    job_id = validate_job_id(f"ask-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}")
    effective_budget_job_id = budget_job_id or job_id
    est_input_tok = max(1, len(full_prompt) // 4)
    est_cost_cents = float(
        estimate_cost_usd(
            chosen_model,
            {"prompt_token_count": est_input_tok},
            single_request=True,
        ).get("total_cost_cents")
        or 0.0
    )
    budget_guard = getattr(receipt_store, "budget", None)
    res_id: str | None = None
    if budget_guard is not None:
        res_id = budget_guard.reserve(
            effective_budget_job_id,
            estimate_cents=est_cost_cents,
            unpriced=est_cost_cents <= 0.0,
            workspace_root=ws_root,
        )

    started_at = round(time.time(), 3)
    usage: dict[str, Any] = {
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "total_token_count": 0,
    }
    status = "error"
    error_msg: str | None = None
    response_text = ""
    sources_out: list[dict[str, str]] = []
    queries_out: list[str] = []
    continuations_out = 0
    finish_reason_out = "STOP"
    extra_meta: dict[str, Any] = {}

    active_single_backend = backend or _default_genai_backend
    try:
        req_payload: dict[str, Any] = {
            "auth_mode": auth_mode,
            "api_key": resolved_key,
            "project": resolved_project,
            "location": resolved_location,
            "location_explicit": bool(auth.get("location_explicit")),
            "base_url": auth.get("base_url"),
            "gateway_wire": auth.get("gateway_wire"),
            "base_url_token": auth.get("base_url_token"),
            "base_url_audience": auth.get("base_url_audience"),
            "bifrost_virtual_key": auth.get("bifrost_virtual_key"),
            "model": chosen_model,
            "prompt": full_prompt,
            "system_instruction": redacted_sys,
            "temperature": temperature,
        }
        if effective_thinking:
            req_payload["thinking_level"] = effective_thinking
        if max_output_tokens is not None and int(max_output_tokens) > 0:
            req_payload["max_output_tokens"] = int(max_output_tokens)
        if timeout_ms is not None and int(timeout_ms) > 0:
            req_payload["timeout_ms"] = int(timeout_ms)
        if media_parts:
            req_payload["media_parts"] = media_parts
        if effective_google_search:
            req_payload["google_search"] = True
        if stable_prefix and not check_dependencies:
            from server.agy_state import PROCESS_SESSION_ID

            req_payload["prompt_prefix"] = stable_prefix
            req_payload["prompt_delta"] = redacted_prompt
            req_payload["session_id"] = (session_id or "").strip() or PROCESS_SESSION_ID
            req_payload["workspace_root"] = str(ws_root)

        if check_dependencies and included_files:
            def _run_dep_batch(batch_prompt: str) -> dict[str, Any]:
                b_req = dict(req_payload)
                b_req["prompt"] = batch_prompt
                b_req["google_search"] = True
                return _run_ask_with_continuations(b_req, active_single_backend)

            dep_res = check_dependencies_batched(ws_root, included_files, redacted_prompt, _run_dep_batch)
            if dep_res is not None:
                result = dep_res
            else:
                result = _run_ask_with_continuations(req_payload, active_single_backend)
        elif http_transport is not None:
            result = call_gemini_long_decode(req_payload, http_transport=http_transport)
        elif is_long_decode_model(chosen_model) or max_output_tokens is not None:
            result = _run_ask_with_continuations(req_payload, active_single_backend)
        else:
            result = active_single_backend(req_payload)

        if result.get("vertex_project"):
            resolved_project = str(result["vertex_project"])
        if result.get("vertex_location"):
            resolved_location = str(result["vertex_location"])
        if result.get("reroute_note") and not reroute_note:
            reroute_note = str(result["reroute_note"])
        response_text = str(result.get("text") or "")
        sources_out = list(result.get("sources") or [])
        queries_out = list(result.get("search_queries") or [])
        continuations_out = int(result.get("continuations") or 0)
        if result.get("resumed_conversation"):
            extra_meta["agy_conversation_resumed"] = True
        finish_reason_out = str(result.get("finish_reason") or "STOP")
        if "batches" in result:
            extra_meta["batches"] = result["batches"]
            extra_meta["dependencies_checked"] = result.get("dependencies_checked", 0)
        if isinstance(result.get("usage"), dict):
            for k in (
                "prompt_token_count",
                "cached_content_token_count",
                "candidates_token_count",
                "thoughts_token_count",
                "total_token_count",
            ):
                usage[k] = int(result["usage"].get(k) or 0)
            if isinstance(result["usage"].get("requests"), list):
                usage["requests"] = result["usage"]["requests"]
        if usage["total_token_count"] == 0:
            usage["total_token_count"] = (
                usage["prompt_token_count"]
                + usage["candidates_token_count"]
                + usage["thoughts_token_count"]
            )
        status = "ok"
    except (TimeoutError, asyncio.TimeoutError) as exc:
        status = "timeout"
        error_msg = f"TimeoutError: {exc}"
        partial = getattr(exc, "partial_usage", None)
        if isinstance(partial, dict):
            for k in (
                "prompt_token_count",
                "cached_content_token_count",
                "candidates_token_count",
                "thoughts_token_count",
                "total_token_count",
            ):
                usage[k] = int(partial.get(k) or 0)
        raise
    except Exception as exc:
        status = "error"
        error_msg = f"{type(exc).__name__}: {exc}"
        partial = getattr(exc, "partial_usage", None)
        if isinstance(partial, dict):
            for k in (
                "prompt_token_count",
                "cached_content_token_count",
                "candidates_token_count",
                "thoughts_token_count",
                "total_token_count",
            ):
                usage[k] = int(partial.get(k) or 0)
        raise
    finally:
        finished_at = round(time.time(), 3)
        clipped_text, was_truncated, orig_tokens = truncate_to_token_budget(
            response_text, MAX_TOOL_RESPONSE_TOKENS
        )
        receipt_payload: dict[str, Any] = {
            "job_id": job_id,
            "kind": "ask",
            "status": status,
            "auth_mode": auth_mode,
            "model": chosen_model,
            "vertex_project": resolved_project,
            "vertex_location": resolved_location,
            "reroute_note": reroute_note,
            "context_files": included_files,
            **({"skipped_files": skipped_files} if skipped_files else {}),
            **({"media_files": attached_media} if attached_media else {}),
            **({"spec_url": final_spec_url} if final_spec_url else {}),
            **({"sources": sources_out} if sources_out else {}),
            **({"search_queries": queries_out} if queries_out else {}),
            **({"thinking_level": effective_thinking} if effective_thinking else {}),
            "redactions_count": total_redactions,
            "continuations": continuations_out,
            "finish_reason": finish_reason_out,
            "started_at_epoch": started_at,
            "finished_at_epoch": finished_at,
            "duration_seconds": round(max(0.0, finished_at - started_at), 3),
            "usage": usage,
            "response_tokens_estimated": orig_tokens,
            "response_truncated": was_truncated,
            "error": error_msg,
            "applied": False,
        }
        if record_receipt:
            saved_receipt = receipt_store.write_receipt(
                job_id,
                receipt_payload,
                scrub=scrub_receipt,
                workspace_root=str(ws_root),
                api_key=resolved_key,
            )
            computed_cost = saved_receipt["cost"]
            if budget_guard is not None and res_id is not None:
                budget_guard.settle(
                    res_id,
                    job_id=effective_budget_job_id,
                    actual_cents=0.0,
                    workspace_root=ws_root,
                    record_live=False,
                )
        else:
            computed_cost = estimate_cost_usd(
                chosen_model,
                usage,
                single_request=not isinstance(usage.get("requests"), list),
            )
            if budget_guard is not None:
                budget_guard.settle(
                    res_id,
                    job_id=effective_budget_job_id,
                    actual_cents=float(computed_cost.get("total_cost_cents") or 0.0),
                    workspace_root=ws_root,
                    record_live=True,
                )

    return {
        "job_id": job_id,
        "status": status,
        "auth_mode": auth_mode,
        "model": chosen_model,
        **({"thinking_level": effective_thinking} if effective_thinking else {}),
        "vertex_location": resolved_location,
        "reroute_note": reroute_note,
        "context_files": included_files,
        **({"skipped_files": skipped_files} if skipped_files else {}),
        **({"media_files": attached_media} if attached_media else {}),
        **({"spec_url": final_spec_url} if final_spec_url else {}),
        **({"sources": sources_out} if sources_out else {}),
        **({"search_queries": queries_out} if queries_out else {}),
        "redactions_count": total_redactions,
        "continuations": continuations_out,
        "finish_reason": finish_reason_out,
        "text": clipped_text,
        "response_truncated": was_truncated,
        "response_tokens_estimated": orig_tokens,
        "usage": usage,
        "cost": computed_cost,
        **extra_meta,
    }
