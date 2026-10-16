"""Crash-safe receipt store, sensitive field scrubbing, and diff provenance hashing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from server.regions import estimate_cost_usd


def sha256_text(text: str) -> str:
    """Return hex SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_diff_provenance(diff_text: str) -> dict[str, Any]:
    """Extract deterministic per-file added/removed line hashes from a unified git diff.

    Used both when a worker job finishes (or is applied via delegate_apply) and by
    the Stop-hook provenance checker to verify that every changed file and added line
    in the host repository came from an applied worker receipt.
    """
    files_changed: list[str] = []
    per_file_added_hashes: dict[str, list[str]] = {}
    per_file_removed_hashes: dict[str, list[str]] = {}
    current_file: str | None = None

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ b/"):
            current_file = raw_line[len("+++ b/") :].strip()
            if current_file and current_file not in files_changed:
                files_changed.append(current_file)
            per_file_added_hashes.setdefault(current_file, [])
            per_file_removed_hashes.setdefault(current_file, [])
        elif raw_line.startswith("--- a/"):
            continue
        elif raw_line.startswith("@@"):
            continue
        elif current_file is not None:
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                line_body = raw_line[1:]
                per_file_added_hashes[current_file].append(sha256_text(line_body))
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
                line_body = raw_line[1:]
                per_file_removed_hashes[current_file].append(sha256_text(line_body))

    per_file_summary = {
        path: {
            "added_count": len(per_file_added_hashes.get(path, [])),
            "removed_count": len(per_file_removed_hashes.get(path, [])),
            "added_line_sha256": per_file_added_hashes.get(path, []),
            "removed_line_sha256": per_file_removed_hashes.get(path, []),
        }
        for path in files_changed
    }

    return {
        "diff_sha256": sha256_text(diff_text),
        "files_changed": files_changed,
        "per_file": per_file_summary,
    }


def scrub_receipt_for_export(
    receipt: dict[str, Any],
    *,
    project_id: str | None = None,
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Return a copy of a receipt with paying project IDs and host paths scrubbed.

    Prevents leaking internal GCP project IDs or local home directory paths when
    receipts are committed to public git repositories.
    """
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
    """Atomic, crash-safe receipt storage for Vertex AI delegations and queries."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        resolved_dir = self._resolve_data_dir(data_dir)
        self.data_dir = resolved_dir
        self.receipts_dir = self.data_dir / "receipts"
        self.worktrees_dir = self.data_dir / "worktrees"
        self.ledger_path = self.data_dir / "ledger.jsonl"
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_data_dir(data_dir: str | Path | None) -> Path:
        if data_dir:
            candidate = str(data_dir).strip()
            if candidate and not candidate.startswith("${"):
                return Path(candidate).resolve()
        for env_key in ("CC_AGY_DATA_DIR", "CLAUDE_PLUGIN_DATA"):
            val = os.environ.get(env_key, "").strip()
            if val and not val.startswith("${"):
                return Path(val).resolve()
        return (Path.cwd() / ".cc-agy-data").resolve()

    def receipt_path(self, job_id: str) -> Path:
        safe_id = "".join(c for c in job_id if c.isalnum() or c in ("-", "_"))
        if not safe_id:
            raise ValueError(f"Invalid job_id: {job_id!r}")
        return self.receipts_dir / f"{safe_id}.json"

    def write_receipt(
        self,
        job_id: str,
        payload: dict[str, Any],
        *,
        scrub: bool = False,
        workspace_root: str | None = None,
    ) -> dict[str, Any]:
        """Atomically persist a job receipt to disk and append to the ledger.

        Safe to call from a `finally:` block on error, cancellation, or timeout so
        partial token usage and tool call counts are never lost.
        """
        record = dict(payload)
        record["job_id"] = job_id
        record.setdefault("updated_at_epoch", round(time.time(), 3))

        usage = record.get("usage") or {}
        model = str(record.get("model") or "gemini-3.1-pro-preview")
        record["cost"] = estimate_cost_usd(model, usage)

        if scrub:
            record = scrub_receipt_for_export(
                record,
                project_id=str(record.get("vertex_project") or ""),
                workspace_root=workspace_root,
            )

        target = self.receipt_path(job_id)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{job_id}-",
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

    def list_receipts(self) -> list[dict[str, Any]]:
        receipts: list[dict[str, Any]] = []
        for path in sorted(self.receipts_dir.glob("*.json")):
            try:
                receipts.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        receipts.sort(key=lambda r: float(r.get("started_at_epoch") or 0.0))
        return receipts

    def summarize_costs(self) -> dict[str, Any]:
        """Aggregate token usage and cost across all receipts, including failed runs."""
        receipts = self.list_receipts()
        totals = {
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
            "by_model": {},
        }
        for r in receipts:
            status = str(r.get("status") or "")
            if status in ("ok", "completed"):
                totals["completed_jobs"] += 1
            elif status in ("error", "timeout", "cancelled", "scope_violation"):
                totals["failed_or_timed_out_jobs"] += 1
            if r.get("applied"):
                totals["applied_jobs"] += 1

            usage = r.get("usage") or {}
            cost = r.get("cost") or estimate_cost_usd(
                str(r.get("model") or "gemini-3.1-pro-preview"), usage
            )

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
