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


def _venv_has_pkgs(venv_dir: Path, *dotted_pkgs: str) -> bool:
    """Fast filesystem check (<1ms) for installed packages inside `venv_dir`."""
    sp_dirs = (
        [venv_dir / "Lib" / "site-packages"]
        if os.name == "nt"
        else list(venv_dir.glob("lib/python3*/site-packages"))
    )
    if not sp_dirs:
        return False
    for pkg in dotted_pkgs:
        rel = Path(*pkg.split(".")) / "__init__.py"
        if not any((sp / rel).is_file() for sp in sp_dirs):
            return False
    return True


def _resolve_test_python() -> str:
    """Select an interpreter that has both pytest and optional plugin SDKs when present."""
    import importlib.util

    try:
        if (
            sys.executable
            and importlib.util.find_spec("pytest") is not None
            and importlib.util.find_spec("google.genai") is not None
            and importlib.util.find_spec("google.antigravity") is not None
        ):
            return sys.executable
    except (ImportError, ValueError, AttributeError):
        pass
    for venv_dir in (PLUGIN_ROOT / ".venv", PLUGIN_ROOT.parent / ".venv"):
        py_cand = (
            venv_dir / "Scripts" / "python.exe"
            if os.name == "nt"
            else venv_dir / "bin" / "python"
        )
        if py_cand.is_file() and os.access(py_cand, os.X_OK):
            if _venv_has_pkgs(venv_dir, "pytest", "google.genai", "google.antigravity"):
                return str(py_cand)
    return sys.executable


def _run_one_test_file(args_tuple: tuple[str, str]) -> tuple[str, int, float, str]:
    """Execute pytest on a single test file with clean ambient credentials."""
    rel_file, py_bin = args_tuple
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ptest-parallel-") as tmp_dir:
        env = dict(os.environ)
        for scrub_key in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_GEMINI_BASE_URL",
            "GOOGLE_VERTEX_BASE_URL",
            "GOOGLE_CLOUD_PROJECT",
            "ANTHROPIC_VERTEX_PROJECT",
            "ANTHROPIC_API_KEY",
        ):
            env.pop(scrub_key, None)
        for k in [k for k in env if k.startswith("CC_AGY_")]:
            env.pop(k, None)
        py_dir = str(Path(py_bin).resolve().parent)
        env["PATH"] = f"{py_dir}{os.pathsep}{env.get('PATH', os.defpath)}"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [
                py_bin,
                "-m",
                "pytest",
                "-q",
                "--no-header",
                "-o",
                f"tmp_path_retention_count=1",
                "--basetemp",
                str(Path(tmp_dir) / "basetemp"),
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
    default_workers = max(2, min(6, (os.cpu_count() or 4) // 2))
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=default_workers,
        help=f"Parallel file workers (default: {default_workers}).",
    )
    args = parser.parse_args(argv)
    py_bin = _resolve_test_python()

    test_files = sorted(
        (PLUGIN_ROOT / "tests").glob("test_*.py"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    rel_files = [p.relative_to(PLUGIN_ROOT).as_posix() for p in test_files]
    t_start = time.monotonic()
    failures: list[tuple[str, str]] = []

    print(
        f"Running {len(rel_files)} test modules across {args.workers} parallel workers ({py_bin})..."
    )
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        fut_map = {pool.submit(_run_one_test_file, (rf, py_bin)): rf for rf in rel_files}
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
