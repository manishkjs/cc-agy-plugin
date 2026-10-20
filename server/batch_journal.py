"""Append-only journal of landed file tasks, so a re-sent batch skips work that is already done.

One JSONL file per job under ``<data_dir>/file_batches/<job_id>.jsonl``. A task counts as unchanged
only when both its fingerprint (the canonical task JSON) and the file's current SHA-256 match the
journal entry written when it landed; anything else runs again.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

from server.batch_plan import FileTask


def _file_sha(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BatchJournal:
    def __init__(self, data_dir: str | Path, job_id: str) -> None:
        from server.receipts import validate_job_id

        self.job_id = validate_job_id(job_id)
        self.path = Path(data_dir) / "file_batches" / f"{self.job_id}.jsonl"
        self._lock = threading.Lock()
        self._latest: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and isinstance(entry.get("task_id"), str):
                self._latest[entry["task_id"]] = entry

    def record_landed(self, task: FileTask, root: Path) -> dict[str, Any]:
        entry = {
            "task_id": task.task_id,
            "task_fingerprint": task.fingerprint(),
            "path": task.path,
            "file_sha256": _file_sha(root / task.path),
            "landed_at_epoch": round(time.time(), 3),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
            self._latest[task.task_id] = entry
        return entry

    def entry(self, task_id: str) -> dict[str, Any] | None:
        return self._latest.get(task_id)

    def landed_paths(self) -> set[str]:
        return {str(e.get("path")) for e in self._latest.values() if e.get("path")}

    def is_unchanged(self, task: FileTask, root: Path) -> bool:
        entry = self._latest.get(task.task_id)
        if not entry or entry.get("task_fingerprint") != task.fingerprint() or entry.get("path") != task.path:
            return False
        current = _file_sha(root / task.path)
        return current is not None and current == entry.get("file_sha256")
