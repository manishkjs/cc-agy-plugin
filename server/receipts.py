"""Crash-safe receipt store, HMAC provenance signing, scrubbing, and diff hashing."""

from __future__ import annotations

import atexit
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
import time
from typing import Any

from server.regions import estimate_cost_usd

_VALID_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_BINARY_DIFF_RE = re.compile(r"^Binary files (.+?) and (.+?) differ$")


def unquote_git_path(raw: str) -> str:
    """Unquote a git C-quoted path (e.g. '\"a/r\\303\\251sum\\303\\251.py\"') into a UTF-8 string."""
    s = raw.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        inner = s[1:-1]
        buf = bytearray()
        i = 0
        n = len(inner)
        escape_map = {
            "a": 0x07,
            "b": 0x08,
            "t": 0x09,
            "n": 0x0A,
            "v": 0x0B,
            "f": 0x0C,
            "r": 0x0D,
            "\\": 0x5C,
            '"': 0x22,
        }
        while i < n:
            ch = inner[i]
            if ch == "\\" and i + 1 < n:
                nxt = inner[i + 1]
                if nxt in "01234567":
                    oct_digits = nxt
                    j = i + 2
                    while j < n and j < i + 4 and inner[j] in "01234567":
                        oct_digits += inner[j]
                        j += 1
                    buf.append(int(oct_digits, 8) & 0xFF)
                    i = j
                    continue
                if nxt in escape_map:
                    buf.append(escape_map[nxt])
                    i += 2
                    continue
                buf.extend(nxt.encode("utf-8"))
                i += 2
                continue
            buf.extend(ch.encode("utf-8"))
            i += 1
        return buf.decode("utf-8", errors="replace")
    return s


def _strip_ab_prefix(path: str, prefix: str) -> str:
    unq = unquote_git_path(path)
    if unq.startswith(prefix):
        return unq[len(prefix) :]
    return unq


def parse_diff_git_header(line: str) -> tuple[str, str] | None:
    """Parse `diff --git a/... b/...` (supporting C-quoted and space-containing paths)."""
    stripped = line.rstrip("\r\n")
    if not stripped.startswith("diff --git "):
        return None
    rem = stripped[len("diff --git ") :].strip()
    if not rem:
        return None
    if rem.startswith('"'):
        i = 1
        n = len(rem)
        while i < n:
            if rem[i] == "\\" and i + 1 < n:
                i += 2
                continue
            if rem[i] == '"':
                first_tok = rem[: i + 1]
                second_tok = rem[i + 1 :].strip()
                return _strip_ab_prefix(first_tok, "a/"), _strip_ab_prefix(second_tok, "b/")
            i += 1
    q_pos = rem.find(' "b/')
    if q_pos != -1:
        first_tok = rem[:q_pos]
        second_tok = rem[q_pos + 1 :].strip()
        return _strip_ab_prefix(first_tok, "a/"), _strip_ab_prefix(second_tok, "b/")
    idx = 0
    candidates: list[tuple[str, str]] = []
    while True:
        pos = rem.find(" b/", idx)
        if pos == -1:
            break
        left = _strip_ab_prefix(rem[:pos], "a/")
        right = _strip_ab_prefix(rem[pos + 1 :], "b/")
        if left == right:
            return left, right
        candidates.append((left, right))
        idx = pos + 3
    if candidates:
        return candidates[0]
    return None


def parse_unified_header_path(line: str, marker: str) -> str | None:
    """Parse `--- a/...` or `+++ b/...` or `--- /dev/null` (including C-quoted paths)."""
    prefix = f"{marker} "
    if not line.startswith(prefix):
        return None
    raw_rest = line[len(prefix) :].rstrip("\r\n")
    if not raw_rest.startswith('"') and "\t" in raw_rest:
        raw_rest = raw_rest.split("\t", 1)[0]
    unq = unquote_git_path(raw_rest)
    if unq == "/dev/null":
        return "/dev/null"
    ab = "a/" if marker == "---" else "b/"
    if unq.startswith(ab):
        return unq[len(ab) :]
    return None


def sha256_text(text: str) -> str:
    """Return hex SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def validate_job_id(job_id: str) -> str:
    """Strictly validate a job_id to prevent path traversal or flag injection."""
    if not isinstance(job_id, str):
        raise ValueError(f"Invalid job_id type: {type(job_id).__name__}")
    candidate = job_id.strip()
    if not _VALID_JOB_ID_RE.fullmatch(candidate):
        raise ValueError(
            f"Invalid job_id {job_id!r}: must match {_VALID_JOB_ID_RE.pattern}"
        )
    return candidate


def resolve_workspace_root(
    workspace_root: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
    anchor_to_git_root: bool = True,
) -> Path:
    """Resolve the active workspace root, honoring CLAUDE_PROJECT_DIR and git top-level."""
    source_env = env if env is not None else os.environ
    base: Path | None = None
    if workspace_root is not None:
        raw = str(workspace_root).strip()
        if raw and not raw.startswith("${"):
            base = Path(raw).resolve()
    if base is None:
        env_proj = (source_env.get("CLAUDE_PROJECT_DIR") or "").strip()
        if env_proj and not env_proj.startswith("${"):
            base = Path(env_proj).resolve()
    if base is None:
        base = Path.cwd().resolve()

    if anchor_to_git_root and base.exists():
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(base),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return Path(res.stdout.strip()).resolve()
    return base


def extract_diff_provenance(diff_text: str) -> dict[str, Any]:
    """Extract deterministic per-file added/removed line hashes, deletions, and binaries from a unified diff."""
    files_changed: list[str] = []
    deleted_files: list[str] = []
    binary_files: list[str] = []
    per_file_added_hashes: dict[str, list[str]] = {}
    per_file_removed_hashes: dict[str, list[str]] = {}
    current_file: str | None = None
    pending_old_file: str | None = None
    in_hunk = False

    def _register_file(path: str) -> None:
        if path and path != "/dev/null":
            if path not in files_changed:
                files_changed.append(path)
            per_file_added_hashes.setdefault(path, [])
            per_file_removed_hashes.setdefault(path, [])

    for raw_line in diff_text.splitlines():
        git_hdr = parse_diff_git_header(raw_line)
        if git_hdr is not None:
            old_p, new_p = git_hdr
            pending_old_file = old_p
            current_file = new_p or old_p
            in_hunk = False
            _register_file(current_file)
            continue

        if not in_hunk:
            bin_hdr = _BINARY_DIFF_RE.match(raw_line)
            if bin_hdr:
                old_raw = _strip_ab_prefix(bin_hdr.group(1), "a/")
                new_raw = _strip_ab_prefix(bin_hdr.group(2), "b/")
                target_b = (
                    new_raw
                    if new_raw and new_raw != "/dev/null"
                    else (old_raw if old_raw and old_raw != "/dev/null" else (current_file or ""))
                ).strip()
                if target_b:
                    _register_file(target_b)
                    if target_b not in binary_files:
                        binary_files.append(target_b)
                continue

            old_hdr = parse_unified_header_path(raw_line, "---")
            if old_hdr is not None:
                pending_old_file = None if old_hdr == "/dev/null" else old_hdr
                continue

            new_hdr = parse_unified_header_path(raw_line, "+++")
            if new_hdr is not None:
                if new_hdr == "/dev/null":
                    if pending_old_file:
                        current_file = pending_old_file
                        _register_file(current_file)
                        if current_file not in deleted_files:
                            deleted_files.append(current_file)
                else:
                    current_file = new_hdr
                    _register_file(current_file)
                continue

            if raw_line.startswith("deleted file mode"):
                if current_file and current_file not in deleted_files:
                    deleted_files.append(current_file)
                continue
            if raw_line.startswith("GIT binary patch"):
                if current_file and current_file not in binary_files:
                    binary_files.append(current_file)
                continue
            if raw_line.startswith("@@"):
                in_hunk = True
                continue
            continue

        if raw_line.startswith("@@"):
            in_hunk = True
            continue

        if current_file is not None:
            if raw_line.startswith("+"):
                per_file_added_hashes[current_file].append(sha256_text(raw_line[1:]))
            elif raw_line.startswith("-"):
                per_file_removed_hashes[current_file].append(sha256_text(raw_line[1:]))

    per_file_summary = {
        path: {
            "added_count": len(per_file_added_hashes.get(path, [])),
            "removed_count": len(per_file_removed_hashes.get(path, [])),
            "added_line_sha256": per_file_added_hashes.get(path, []),
            "removed_line_sha256": per_file_removed_hashes.get(path, []),
            "deleted": path in deleted_files,
            "binary": path in binary_files,
        }
        for path in files_changed
    }

    return {
        "diff_sha256": sha256_text(diff_text),
        "files_changed": files_changed,
        "deleted_files": deleted_files,
        "binary_files": binary_files,
        "per_file": per_file_summary,
    }


_ENUM_METADATA_KEYS: frozenset[str] = frozenset(
    {"status", "kind", "auth_mode", "model", "vertex_location", "containment", "thinking_level", "stop_reason"}
)


def _redact_obj_values(obj: Any, secrets: list[str], *, parent_key: str = "") -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "api_key":
                out[k] = "<api-key>"
            else:
                out[k] = _redact_obj_values(v, secrets, parent_key=str(k))
        return out
    if isinstance(obj, list):
        return [_redact_obj_values(item, secrets, parent_key=parent_key) for item in obj]
    if isinstance(obj, str):
        s = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "<api-key>", obj)
        for sec in secrets:
            if len(sec) >= 4 and parent_key not in _ENUM_METADATA_KEYS:
                s = s.replace(sec, "<api-key>")
        return s
    return obj


def redact_api_keys(
    receipt: dict[str, Any],
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Unconditionally redact secret API keys from a receipt before persistence without corrupting dict keys."""
    secrets: list[str] = []
    for candidate in (
        api_key,
        receipt.get("api_key"),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
        os.environ.get("CLAUDE_PLUGIN_OPTION_GEMINI_API_KEY"),
    ):
        if isinstance(candidate, str):
            cleaned = candidate.strip()
            if len(cleaned) >= 2 and not cleaned.startswith("${") and cleaned != "<api-key>":
                if cleaned not in secrets:
                    secrets.append(cleaned)
    # Sort longest secrets first to avoid partial substring replacement
    secrets.sort(key=len, reverse=True)
    cloned = json.loads(json.dumps(receipt, sort_keys=True))
    return _redact_obj_values(cloned, secrets)


def scrub_receipt_for_export(
    receipt: dict[str, Any],
    *,
    project_id: str | None = None,
    api_key: str | None = None,
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Return a copy of a receipt with paying project IDs, API keys, and host paths scrubbed."""
    redacted = redact_api_keys(receipt, api_key=api_key)
    raw_json = json.dumps(redacted, sort_keys=True)
    proj = project_id or receipt.get("vertex_project")
    if proj and isinstance(proj, str) and proj not in ("<paying-project>", ""):
        escaped_proj = json.dumps(proj)[1:-1]
        if escaped_proj:
            raw_json = raw_json.replace(escaped_proj, "<paying-project>")
    if workspace_root and isinstance(workspace_root, str):
        norm_ws = str(Path(workspace_root).resolve())
        escaped_ws = json.dumps(norm_ws)[1:-1]
        if escaped_ws:
            raw_json = raw_json.replace(escaped_ws, "/workspace")
    home_dir = str(Path.home())
    if home_dir and home_dir != "/":
        escaped_home = json.dumps(home_dir)[1:-1]
        if escaped_home:
            raw_json = raw_json.replace(escaped_home, "/home/user")
    scrubbed = json.loads(raw_json)
    if "vertex_project" in scrubbed and scrubbed["vertex_project"] is not None:
        scrubbed["vertex_project"] = "<paying-project>"
    if "api_key" in scrubbed:
        scrubbed["api_key"] = "<api-key>"
    return scrubbed


_REGISTERED_WORKTREE_DIRS: set[Path] = set()


def _cleanup_empty_worktree_dirs() -> None:
    for d in list(_REGISTERED_WORKTREE_DIRS):
        try:
            if d.exists() and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass


atexit.register(_cleanup_empty_worktree_dirs)


def _register_empty_worktree_dir_cleanup(path: Path) -> None:
    _REGISTERED_WORKTREE_DIRS.add(path)


class ReceiptStore:
    """Atomic, HMAC-signed, crash-safe receipt storage for Vertex AI delegations and queries."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.workspace_root = resolve_workspace_root(workspace_root)
        self.data_dir = self._resolve_data_dir(data_dir, self.workspace_root)
        self.receipts_dir = self.data_dir / "receipts"
        # Place worker worktrees in a separate 0o700 sibling/temp directory outside `data_dir`
        # so relative paths like `../../.hmac_key` from inside a worker's worktree cannot
        # reach `.hmac_key` or tamper with receipts/baselines.
        wt_slug = sha256_text(f"{self.data_dir}:{self.workspace_root}")[:12]
        tmp_root = Path(tempfile.gettempdir()).resolve()
        if self.data_dir.parent != tmp_root and self.data_dir.parent != self.workspace_root:
            self.worktrees_dir = self.data_dir.parent / f".cc-agy-worktrees-{wt_slug}"
        else:
            self.worktrees_dir = tmp_root / f"cc-agy-worktrees-{wt_slug}"
        self.baselines_dir = self.data_dir / "baselines"
        self.ledger_path = self.data_dir / "ledger.jsonl"
        self.hmac_key_path = self.data_dir / ".hmac_key"
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.receipts_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.worktrees_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.baselines_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for priv_dir in (self.data_dir, self.receipts_dir, self.worktrees_dir, self.baselines_dir):
            try:
                os.chmod(priv_dir, 0o700)
            except OSError:
                pass
        _register_empty_worktree_dir_cleanup(self.worktrees_dir)
        self._recorded_workspaces: set[str] = set()
        self._hmac_key = self._load_or_create_hmac_key()

    @staticmethod
    def _resolve_data_dir(
        data_dir: str | Path | None,
        workspace_root: Path | None = None,
    ) -> Path:
        if data_dir is not None:
            candidate = str(data_dir).strip()
            if candidate and not candidate.startswith("${"):
                return Path(candidate).resolve()
        for env_key in ("CC_AGY_DATA_DIR", "CLAUDE_PLUGIN_DATA"):
            val = os.environ.get(env_key, "").strip()
            if val and not val.startswith("${"):
                return Path(val).resolve()
        ws = workspace_root or resolve_workspace_root()
        # Keep default data dir outside the user's git working tree so git status stays clean,
        # unless a legacy `<workspace>/.cc-agy-data` directory already exists.
        legacy_in_tree = (ws / ".cc-agy-data").resolve()
        if legacy_in_tree.exists():
            return legacy_in_tree
        ws_slug = sha256_text(str(ws))[:12]
        return (Path.home() / ".cache" / "cc-agy-plugin" / ws_slug).resolve()

    def _load_or_create_hmac_key(self) -> bytes:
        lock_path = self.data_dir / ".hmac_key.lock"
        lock_fd: int | None = None
        try:
            lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except OSError:
            pass
        try:
            if self.hmac_key_path.exists():
                try:
                    raw = self.hmac_key_path.read_bytes().strip()
                    if len(raw) >= 32:
                        return raw
                except OSError:
                    pass
            key = secrets.token_hex(32).encode("ascii")
            try:
                fd, tmp_key_path = tempfile.mkstemp(
                    prefix=".hmac_key-",
                    suffix=".tmp",
                    dir=str(self.data_dir),
                )
                try:
                    os.chmod(tmp_key_path, 0o600)
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(key)
                    os.replace(tmp_key_path, self.hmac_key_path)
                finally:
                    if os.path.exists(tmp_key_path):
                        try:
                            os.unlink(tmp_key_path)
                        except OSError:
                            pass
            except OSError:
                pass
            if self.hmac_key_path.exists():
                try:
                    disk_raw = self.hmac_key_path.read_bytes().strip()
                    if len(disk_raw) >= 32:
                        return disk_raw
                except OSError:
                    pass
            return key
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    os.close(lock_fd)
                except OSError:
                    pass

    def _active_hmac_key(self) -> bytes:
        if self.hmac_key_path.exists():
            try:
                disk_key = self.hmac_key_path.read_bytes().strip()
                if len(disk_key) >= 32 and disk_key != self._hmac_key:
                    self._hmac_key = disk_key
            except OSError:
                pass
        return self._hmac_key

    def _compute_hmac(self, record: dict[str, Any]) -> str:
        canonical_payload = {
            "job_id": record.get("job_id"),
            "kind": record.get("kind"),
            "status": record.get("status"),
            "applied": bool(record.get("applied", False)),
            "applied_files": record.get("applied_files", []),
            "applied_file_hashes": record.get("applied_file_hashes", {}),
            "provenance": record.get("provenance"),
            "applied_history": record.get("applied_history", []),
            "prompts": record.get("prompts", []),
        }
        serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hmac.new(self._active_hmac_key(), serialized, hashlib.sha256).hexdigest()

    def verify_receipt_signature(self, record: dict[str, Any]) -> bool:
        """Verify that a receipt on disk was genuinely written by ReceiptStore (not forged)."""
        sig = record.get("receipt_hmac")
        if not sig or not isinstance(sig, str):
            return False
        expected = self._compute_hmac(record)
        return hmac.compare_digest(sig, expected)

    def receipt_path(self, job_id: str) -> Path:
        safe_id = validate_job_id(job_id)
        return self.receipts_dir / f"{safe_id}.json"

    def diff_path(self, job_id: str) -> Path:
        safe_id = validate_job_id(job_id)
        return self.receipts_dir / f"{safe_id}.diff"

    def progress_path(self, job_id: str) -> Path:
        """Worker -> lead signals (report_to_lead) for a job, one JSON object per line."""
        safe_id = validate_job_id(job_id)
        return self.receipts_dir / f"{safe_id}.progress.jsonl"

    def worktree_path(self, job_id: str) -> Path:
        safe_id = validate_job_id(job_id)
        return self.worktrees_dir / safe_id

    def baseline_path(self, workspace_root: str | Path | None = None) -> Path:
        ws = Path(workspace_root or self.workspace_root).resolve()
        ws_hash = sha256_text(str(ws))[:16]
        return self.baselines_dir / f"baseline-{ws_hash}.json"

    def session_baseline_path(self, session_id: str) -> Path:
        safe_sid = validate_job_id(session_id)
        return self.baselines_dir / f"session-{safe_sid}.json"

    def _baseline_marker_path(self, workspace_root: str | Path | None = None) -> Path:
        ws = Path(workspace_root or self.workspace_root).resolve()
        ws_hash = sha256_text(str(ws))[:16]
        return self.baselines_dir / f".recorded-{ws_hash}"

    def record_session_baseline(
        self,
        workspace_root: str | Path,
        provenance: dict[str, Any],
        *,
        baseline_head_sha: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Record an immutable per-session baseline when a new Claude Code session_id starts.

        Returns True if a new session baseline was recorded, or False if `session_id`
        already had a recorded baseline in this session.
        """
        if session_id:
            s_path = self.session_baseline_path(session_id)
            if s_path.exists():
                return False
            s_payload = {
                "workspace_root": str(Path(workspace_root).resolve()),
                "recorded_at_epoch": round(time.time(), 3),
                "baseline_head_sha": baseline_head_sha,
                "provenance": provenance,
            }
            s_ser = json.dumps(s_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            s_payload["baseline_hmac"] = hmac.new(self._active_hmac_key(), s_ser, hashlib.sha256).hexdigest()
            s_path.write_text(json.dumps(s_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        self.record_baseline_dirty_state(
            workspace_root,
            provenance,
            baseline_head_sha=baseline_head_sha,
            overwrite=True,
        )
        return True

    def record_baseline_dirty_state(
        self,
        workspace_root: str | Path,
        provenance: dict[str, Any],
        *,
        baseline_head_sha: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Record pre-existing uncommitted baseline provenance and HEAD SHA for a workspace session."""
        bpath = self.baseline_path(workspace_root)
        if bpath.exists() and not overwrite:
            return
        payload = {
            "workspace_root": str(Path(workspace_root).resolve()),
            "recorded_at_epoch": round(time.time(), 3),
            "baseline_head_sha": baseline_head_sha,
            "provenance": provenance,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["baseline_hmac"] = hmac.new(self._active_hmac_key(), serialized, hashlib.sha256).hexdigest()
        bpath.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._recorded_workspaces.add(str(Path(workspace_root).resolve()))
        try:
            self._baseline_marker_path(workspace_root).write_text("1\n", encoding="utf-8")
        except OSError:
            pass

    def is_baseline_tampered(self, workspace_root: str | Path, *, session_id: str | None = None) -> bool:
        """Return True if a baseline file exists with an invalid HMAC/JSON or was deleted after being recorded."""
        if session_id:
            try:
                s_path = self.session_baseline_path(session_id)
                if s_path.exists():
                    return self._verify_and_parse_baseline_file(s_path) is None
            except ValueError:
                return True
        bpath = self.baseline_path(workspace_root)
        marker = self._baseline_marker_path(workspace_root)
        ws_key = str(Path(workspace_root).resolve())
        if not bpath.exists():
            return marker.exists() or (ws_key in self._recorded_workspaces)
        return self.read_baseline_record(workspace_root) is None

    def _verify_and_parse_baseline_file(self, bpath: Path) -> dict[str, Any] | None:
        if not bpath.exists():
            return None
        try:
            active_key = self._active_hmac_key()
            data = json.loads(bpath.read_text(encoding="utf-8"))
            sig = data.get("baseline_hmac")
            if not sig or not isinstance(sig, str):
                return None
            check_copy = {
                "workspace_root": data.get("workspace_root"),
                "recorded_at_epoch": data.get("recorded_at_epoch"),
                "baseline_head_sha": data.get("baseline_head_sha"),
                "provenance": data.get("provenance"),
            }
            serialized = json.dumps(check_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
            expected = hmac.new(active_key, serialized, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected):
                return None
            return data
        except (OSError, json.JSONDecodeError):
            return None

    def read_baseline_record(
        self,
        workspace_root: str | Path,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Read and verify the signed baseline record (provenance + baseline_head_sha)."""
        if session_id:
            try:
                s_path = self.session_baseline_path(session_id)
                if s_path.exists():
                    return self._verify_and_parse_baseline_file(s_path)
            except ValueError:
                return None
        bpath = self.baseline_path(workspace_root)
        return self._verify_and_parse_baseline_file(bpath)

    def read_baseline_dirty_state(self, workspace_root: str | Path) -> dict[str, Any] | None:
        """Read and verify the pre-existing uncommitted baseline provenance for workspace_root."""
        rec = self.read_baseline_record(workspace_root)
        return rec.get("provenance") if isinstance(rec, dict) else None

    def write_receipt(
        self,
        job_id: str,
        payload: dict[str, Any],
        *,
        scrub: bool = False,
        workspace_root: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Atomically persist an HMAC-signed job receipt to disk and append to the ledger."""
        safe_id = validate_job_id(job_id)
        record = dict(payload)
        record["job_id"] = safe_id
        record["updated_at_epoch"] = round(time.time(), 3)
        if workspace_root:
            record.setdefault("workspace_root", str(Path(workspace_root).resolve()))

        usage = record.get("usage") or {}
        model = str(record.get("model") or "gemini-3.1-pro-preview")
        record["cost"] = estimate_cost_usd(
            model,
            usage,
            single_request=(record.get("kind") == "ask"),
        )

        # Always redact secret API keys even when scrub=False
        record = redact_api_keys(record, api_key=api_key)

        if scrub:
            record = scrub_receipt_for_export(
                record,
                project_id=str(record.get("vertex_project") or ""),
                api_key=api_key,
                workspace_root=workspace_root,
            )

        record["receipt_hmac"] = self._compute_hmac(record)

        target = self.receipt_path(safe_id)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{safe_id}-",
            suffix=".tmp",
            dir=str(self.receipts_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_path, target)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        with open(self.ledger_path, "a", encoding="utf-8") as ledger_file:
            ledger_file.write(json.dumps(record, sort_keys=True) + "\n")

        return record

    def read_receipt(self, job_id: str) -> dict[str, Any]:
        path = self.receipt_path(job_id)
        if not path.exists():
            raise FileNotFoundError(f"No receipt found for job_id '{job_id}'")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_receipts(self, *, workspace_root: str | Path | None = None) -> list[dict[str, Any]]:
        target_ws = str(Path(workspace_root).resolve()) if workspace_root else None
        receipts: list[dict[str, Any]] = []
        for path in sorted(self.receipts_dir.glob("*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if target_ws and rec.get("workspace_root") not in (None, "/workspace", target_ws):
                continue
            receipts.append(rec)
        receipts.sort(key=lambda r: float(r.get("started_at_epoch") or 0.0))
        return receipts

    def summarize_costs(
        self,
        *,
        workspace_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Aggregate token usage and cost across receipts for the active workspace, including failed runs."""
        target_ws = workspace_root if workspace_root is not None else self.workspace_root
        receipts = self.list_receipts(workspace_root=target_ws)
        totals: dict[str, Any] = {
            "total_jobs": len(receipts),
            "completed_jobs": 0,
            "in_progress_or_paused_jobs": 0,
            "checkpoint_jobs": 0,
            "needs_input_jobs": 0,
            "running_jobs": 0,
            "interrupted_jobs": 0,
            "failed_or_timed_out_jobs": 0,
            "applied_jobs": 0,
            "prompt_token_count": 0,
            "cached_content_token_count": 0,
            "candidates_token_count": 0,
            "thoughts_token_count": 0,
            "total_token_count": 0,
            "total_cost_usd": 0.0,
            "total_cost_cents": 0.0,
            "all_pricing_known": True,
            "by_model": {},
        }
        for r in receipts:
            status = str(r.get("status") or "")
            if status in ("ok", "completed"):
                totals["completed_jobs"] += 1
            elif status == "checkpoint":
                totals["checkpoint_jobs"] += 1
                totals["in_progress_or_paused_jobs"] += 1
            elif status == "needs_input":
                totals["needs_input_jobs"] += 1
                totals["in_progress_or_paused_jobs"] += 1
            elif status == "running":
                totals["running_jobs"] += 1
                totals["in_progress_or_paused_jobs"] += 1
            elif status == "interrupted":
                totals["interrupted_jobs"] += 1
                totals["failed_or_timed_out_jobs"] += 1
            elif status in ("error", "timeout", "cancelled", "scope_violation", "security_violation"):
                totals["failed_or_timed_out_jobs"] += 1
            if r.get("applied"):
                totals["applied_jobs"] += 1

            usage = r.get("usage") or {}
            cost = r.get("cost") or estimate_cost_usd(
                str(r.get("model") or "gemini-3.1-pro-preview"),
                usage,
                single_request=(r.get("kind") == "ask"),
            )
            if not cost.get("pricing_known", True):
                totals["all_pricing_known"] = False

            p_tok = int(usage.get("prompt_token_count") or 0)
            c_tok = int(usage.get("cached_content_token_count") or 0)
            o_tok = int(usage.get("candidates_token_count") or 0)
            t_tok = int(usage.get("thoughts_token_count") or 0)
            tot_tok = int(usage.get("total_token_count") or (p_tok + o_tok + t_tok))

            totals["prompt_token_count"] += p_tok
            totals["cached_content_token_count"] += c_tok
            totals["candidates_token_count"] += o_tok
            totals["thoughts_token_count"] += t_tok
            totals["total_token_count"] += tot_tok
            totals["total_cost_usd"] = round(
                totals["total_cost_usd"] + float(cost.get("total_cost_usd") or 0.0), 8
            )
            totals["total_cost_cents"] = round(totals["total_cost_usd"] * 100.0, 6)

            model = str(r.get("model") or "unknown")
            m_entry = totals["by_model"].setdefault(
                model,
                {
                    "jobs": 0,
                    "prompt_token_count": 0,
                    "cached_content_token_count": 0,
                    "candidates_token_count": 0,
                    "thoughts_token_count": 0,
                    "total_token_count": 0,
                    "total_cost_usd": 0.0,
                    "total_cost_cents": 0.0,
                    "pricing_known": bool(cost.get("pricing_known", True)),
                },
            )
            m_entry["jobs"] += 1
            m_entry["prompt_token_count"] += p_tok
            m_entry["cached_content_token_count"] += c_tok
            m_entry["candidates_token_count"] += o_tok
            m_entry["thoughts_token_count"] += t_tok
            m_entry["total_token_count"] += tot_tok
            m_entry["total_cost_usd"] = round(
                m_entry["total_cost_usd"] + float(cost.get("total_cost_usd") or 0.0), 8
            )
            m_entry["total_cost_cents"] = round(m_entry["total_cost_usd"] * 100.0, 6)

        return totals
