#!/usr/bin/env python3
"""Zero-dependency parallel pytest runner scheduling largest test files first.

Runs each `tests/test_*.py` module in an isolated worker process across CPU cores
so machines without `pytest-xdist` execute the full 29-file suite ~5-6x faster.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _run_one_test_file(rel_file: str) -> tuple[str, int, float, str]:
    """Execute pytest on a single test file with an isolated temp directory."""
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cc-agy-ptest-") as tmp_dir:
        env = dict(os.environ)
        env["TMPDIR"] = tmp_dir
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--no-header",
                "-p",
                "no:cacheprovider",
                rel_file,
            ],
            cwd=str(PLUGIN_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - t0
    combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return rel_file, proc.returncode, elapsed, combined


def main(argv: list[str] | None = None) -> int:
    """Run all `tests/test_*.py` files concurrently and report per-file status."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_workers = max(2, min(10, (os.cpu_count() or 4) // 2))
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=default_workers,
        help=f"Parallel file workers (default: {default_workers}).",
    )
    args = parser.parse_args(argv)

    test_files = sorted(
        (PLUGIN_ROOT / "tests").glob("test_*.py"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    rel_files = [p.relative_to(PLUGIN_ROOT).as_posix() for p in test_files]
    t_start = time.monotonic()
    failures: list[tuple[str, str]] = []

    print(f"Running {len(rel_files)} test modules across {args.workers} parallel workers...")
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        fut_map = {pool.submit(_run_one_test_file, rf): rf for rf in rel_files}
        for fut in as_completed(fut_map):
            rel_file, rc, elapsed, output = fut.result()
            last_line = output.splitlines()[-1] if output else ""
            status = "PASS" if rc == 0 else "FAIL"
            print(f"  [{status}] {rel_file:<44} ({elapsed:5.1f}s)  {last_line}", flush=True)
            if rc != 0:
                failures.append((rel_file, output))

    total_s = time.monotonic() - t_start
    if failures:
        sys.stderr.write(f"\n{len(failures)} test module(s) FAILED in {total_s:.1f}s:\n")
        for rel_file, output in failures:
            sys.stderr.write(f"\n=== {rel_file} ===\n{output}\n")
        return 1
    print(f"\nAll {len(rel_files)} test modules passed in {total_s:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
