"""Crash-safe receipt store, HMAC provenance signing, scrubbing, and diff hashing."""

from __future__ import annotations

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
_DIFF_GIT_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
_BINARY_DIFF_RE = re.compile(r"^Binary files (?:a/(.+?)|/dev/null) and (?:b/(.+?)|/dev/null) differ$")


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

    def _register_file(path: str) -> None:
        if path and path != "/dev/null":
            if path not in files_changed:
                files_changed.append(path)
            per_file_added_hashes.setdefault(path, [])
            per_file_removed_hashes.setdefault(path, [])

    for raw_line in diff_text.splitlines():
        git_hdr = _DIFF_GIT_HEADER_RE.match(raw_line)
        if git_hdr:
            old_p, new_p = git_hdr.group(1).strip(), git_hdr.group(2).strip()
            pending_old_file = old_p
            current_file = new_p or old_p
            _register_file(current_file)
            continue

        bin_hdr = _BINARY_DIFF_RE.match(raw_line)
        if bin_hdr:
            old_b, new_b = bin_hdr.group(1), bin_hdr.group(2)
            target_b = (new_b or old_b or current_file or "").strip()
            if target_b:
                _register_file(target_b)
                if target_b not in binary_files:
                    binary_files.append(target_b)
            continue

        if raw_line.startswith("--- a/"):
            pending_old_file = raw_line[len("--- a/") :].strip()
            continue
        if raw_line.startswith("--- /dev/null"):
            pending_old_file = None
            continue
        if raw_line.startswith("+++ /dev/null"):
            if pending_old_file:
                current_file = pending_old_file
                _register_file(current_file)
                if current_file not in deleted_files:
                    deleted_files.append(current_file)
            continue
        if raw_line.startswith("+++ b/"):
            current_file = raw_line[len("+++ b/") :].strip()
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
            continue

        if current_file is not None:
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                per_file_added_hashes[current_file].append(sha256_text(raw_line[1:]))
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
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


def scrub_receipt_for_export(
    receipt: dict[str, Any],
    *,
    project_id: str | None = None,
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Return a copy of a receipt with paying project IDs and host paths scrubbed."""
    raw_json = json.dumps(receipt, sort_keys=True)
    proj = project_id or receipt.get("vertex_project")
    if proj and isinstance(proj, str) and proj not in ("<paying-project>", ""):
        raw_json = raw_json.replace(proj, "<paying-project>")
    if workspace_root and isinstance(workspace_root, str):
        norm_ws = str(Path(workspace_root).resolve())
        raw_json = raw_json.replace(norm_ws, "/workspace")
    home_dir = str(Path.home())
    if home_dir and home_dir != "/":
        raw_json = raw_json.replace(home_dir, "/home/user")
    scrubbed = json.loads(raw_json)
    if "vertex_project" in scrubbed:
        scrubbed["vertex_project"] = "<paying-project>"
    return scrubbed


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
        self.worktrees_dir = self.data_dir / "worktrees"
        self.baselines_dir = self.data_dir / "baselines"
        self.ledger_path = self.data_dir / "ledger.jsonl"
        self.hmac_key_path = self.data_dir / ".hmac_key"
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
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
        if self.hmac_key_path.exists():
            try:
                raw = self.hmac_key_path.read_bytes().strip()
                if len(raw) >= 32:
                    return raw
            except OSError:
                pass
        key = secrets.token_hex(32).encode("ascii")
        try:
            fd = os.open(
                str(self.hmac_key_path),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(key)
        except OSError:
            pass
        return key

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
        return hmac.new(self._hmac_key, serialized, hashlib.sha256).hexdigest()

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

    def worktree_path(self, job_id: str) -> Path:
        safe_id = validate_job_id(job_id)
        return self.worktrees_dir / safe_id

    def baseline_path(self, workspace_root: str | Path | None = None) -> Path:
        ws = Path(workspace_root or self.workspace_root).resolve()
        ws_hash = sha256_text(str(ws))[:16]
        return self.baselines_dir / f"baseline-{ws_hash}.json"

    def record_baseline_dirty_state(
        self,
        workspace_root: str | Path,
        provenance: dict[str, Any],
    ) -> None:
        """Record pre-existing uncommitted baseline provenance once per workspace session."""
        bpath = self.baseline_path(workspace_root)
        if bpath.exists():
            return
        payload = {
            "workspace_root": str(Path(workspace_root).resolve()),
            "recorded_at_epoch": round(time.time(), 3),
            "provenance": provenance,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["baseline_hmac"] = hmac.new(self._hmac_key, serialized, hashlib.sha256).hexdigest()
        bpath.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def read_baseline_dirty_state(self, workspace_root: str | Path) -> dict[str, Any] | None:
        """Read and verify the pre-existing uncommitted baseline provenance for workspace_root."""
        bpath = self.baseline_path(workspace_root)
        if not bpath.exists():
            return None
        try:
            data = json.loads(bpath.read_text(encoding="utf-8"))
            sig = data.get("baseline_hmac")
            if not sig or not isinstance(sig, str):
                return None
            check_copy = {
                "workspace_root": data.get("workspace_root"),
                "recorded_at_epoch": data.get("recorded_at_epoch"),
                "provenance": data.get("provenance"),
            }
            serialized = json.dumps(check_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
            expected = hmac.new(self._hmac_key, serialized, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected):
                return None
            return data.get("provenance")
        except (OSError, json.JSONDecodeError):
            return None

    def write_receipt(
        self,
        job_id: str,
        payload: dict[str, Any],
        *,
        scrub: bool = False,
        workspace_root: str | None = None,
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
        record["cost"] = estimate_cost_usd(model, usage)

        if scrub:
            record = scrub_receipt_for_export(
                record,
                project_id=str(record.get("vertex_project") or ""),
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

    def summarize_costs(self) -> dict[str, Any]:
        """Aggregate token usage and cost across all receipts, including failed runs."""
        receipts = self.list_receipts()
        totals: dict[str, Any] = {
            "total_jobs": len(receipts),
            "completed_jobs": 0,
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
            elif status in ("error", "timeout", "cancelled", "scope_violation", "security_violation"):
                totals["failed_or_timed_out_jobs"] += 1
            if r.get("applied"):
                totals["applied_jobs"] += 1

            usage = r.get("usage") or {}
            cost = r.get("cost") or estimate_cost_usd(
                str(r.get("model") or "gemini-3.1-pro-preview"), usage
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
