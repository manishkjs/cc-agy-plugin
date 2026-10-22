"""Managed per-user SDK virtualenv discovery and PEP 668-safe bootstrap."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping
import venv

DEFAULT_PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def managed_venv_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the per-user managed virtualenv directory (`~/.local/share/cc-agy-plugin/venv`)."""
    source = env if env is not None else os.environ
    explicit = str(source.get("CC_AGY_VENV_DIR") or "").strip()
    if explicit and not explicit.startswith("${"):
        return Path(explicit).expanduser()
    if os.name == "nt":
        base = Path(str(source.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))).expanduser()
        return base / "cc-agy-plugin" / "venv"
    xdg_data = str(source.get("XDG_DATA_HOME") or "").strip()
    base = Path(xdg_data).expanduser() if (xdg_data and not xdg_data.startswith("${")) else (Path.home() / ".local" / "share")
    return base / "cc-agy-plugin" / "venv"


def venv_python_path(venv_dir: Path) -> Path:
    """Return the Python interpreter path inside `venv_dir`."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _site_packages_has_module(venv_dir: Path, dotted_pkg: str) -> bool:
    rel = Path(*dotted_pkg.split(".")) / "__init__.py"
    if os.name == "nt":
        return (venv_dir / "Lib" / "site-packages" / rel).is_file()
    return any((p / rel).is_file() for p in venv_dir.glob("lib/python3*/site-packages"))


def python_has_module(python_bin: str | Path, module_name: str) -> bool:
    """Return True if `python_bin` can import `module_name` without cwd shadowing."""
    py_str = str(python_bin)
    if sys.executable and Path(py_str) == Path(sys.executable):
        try:
            return importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError, AttributeError):
            return False
    py_path = Path(py_str)
    if not (py_path.is_file() and os.access(py_path, os.X_OK)):
        return False
    venv_root = py_path.parent.parent
    if _site_packages_has_module(venv_root, module_name):
        return True
    probe = subprocess.run(
        [py_str, "-c", f"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)"],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        check=False,
        timeout=20,
    )
    return probe.returncode == 0


def python_has_plugin_sdks(python_bin: str | Path) -> bool:
    """Return True when `python_bin` has both `google.antigravity` and `google.genai` installed."""
    return python_has_module(python_bin, "google.antigravity") and python_has_module(python_bin, "google.genai")


def candidate_sdk_pythons(
    plugin_root: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> list[Path]:
    """Ordered virtualenv interpreter candidates consulted when system `python3` lacks SDKs."""
    root = (plugin_root or DEFAULT_PLUGIN_ROOT).resolve()
    managed_py = venv_python_path(managed_venv_dir(env))
    local_venv_py = venv_python_path(root / ".venv")
    parent_venv_py = venv_python_path(root.parent / ".venv")
    seen: set[str] = set()
    ordered: list[Path] = []
    for cand in (managed_py, local_venv_py, parent_venv_py):
        key = str(cand)
        if key not in seen:
            seen.add(key)
            ordered.append(cand)
    return ordered


def resolve_sdk_python(
    plugin_root: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the Python interpreter that should run the MCP server and SDK probes.

    Order:
    1. Explicit `CC_AGY_PYTHON_BIN` when set and executable.
    2. Current `sys.executable` when it already has `google-antigravity` and `google-genai`.
    3. Managed per-user venv (`~/.local/share/cc-agy-plugin/venv/bin/python`) when ready.
    4. Plugin-local `.venv` or parent `.venv` when ready.
    5. First existing candidate venv interpreter, else `sys.executable` / `python3`.
    """
    source = env if env is not None else os.environ
    explicit = str(source.get("CC_AGY_PYTHON_BIN") or "").strip()
    if explicit and not explicit.startswith("${"):
        exp_path = Path(explicit).expanduser()
        if exp_path.is_file() and os.access(exp_path, os.X_OK):
            return str(exp_path)

    if sys.executable and python_has_plugin_sdks(sys.executable):
        return sys.executable

    candidates = candidate_sdk_pythons(plugin_root, env=source)
    usable = [c for c in candidates if c.is_file() and os.access(c, os.X_OK)]
    for cand in usable:
        if python_has_plugin_sdks(cand):
            return str(cand)
    if usable:
        return str(usable[0])
    return sys.executable or "python3"


def ensure_sdk_venv(
    plugin_root: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Ensure an SDK-capable Python environment exists without violating PEP 668.

    If an existing interpreter (`CC_AGY_PYTHON_BIN`, `sys.executable`, managed venv, or `.venv`)
    already has `google-antigravity` and `google-genai`, returns it immediately with zero network I/O.
    Otherwise creates `~/.local/share/cc-agy-plugin/venv` and installs `<plugin_root>[antigravity]`
    into that isolated per-user virtualenv.
    """
    root = (plugin_root or DEFAULT_PLUGIN_ROOT).resolve()
    source = env if env is not None else os.environ
    existing_py = resolve_sdk_python(root, env=source)
    if python_has_plugin_sdks(existing_py):
        return {
            "ok": True,
            "bootstrapped": False,
            "python_bin": existing_py,
            "venv_dir": str(managed_venv_dir(source)),
        }

    target_venv = managed_venv_dir(source)
    target_py = venv_python_path(target_venv)
    if output_fn is not None:
        output_fn(f"  Setting up isolated Python environment at {target_venv} ...")

    target_venv.parent.mkdir(parents=True, exist_ok=True)
    used_uv = False
    if not (target_py.is_file() and os.access(target_py, os.X_OK)):
        try:
            venv.EnvBuilder(with_pip=True, clear=False).create(str(target_venv))
        except Exception as venv_exc:
            uv_bin = shutil.which("uv")
            if not uv_bin:
                return {
                    "ok": False,
                    "bootstrapped": False,
                    "python_bin": existing_py,
                    "venv_dir": str(target_venv),
                    "error": (
                        f"Could not create virtualenv at {target_venv} ({venv_exc}). "
                        "On Debian/Ubuntu, run `sudo apt-get update && sudo apt-get install -y python3-venv` "
                        "(or install `uv`) and re-run `python3 -m server.install`."
                    ),
                }
            uv_res = subprocess.run(
                [uv_bin, "venv", str(target_venv)],
                cwd=tempfile.gettempdir(),
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            if uv_res.returncode != 0:
                tail = (uv_res.stderr or uv_res.stdout or "").strip()[-500:]
                return {
                    "ok": False,
                    "bootstrapped": False,
                    "python_bin": existing_py,
                    "venv_dir": str(target_venv),
                    "error": f"`uv venv {target_venv}` failed: {tail}",
                }
            used_uv = True

    install_spec = f"{root}[antigravity]"
    is_ephemeral_cache = "plugins/cache" in root.as_posix()
    edit_args = [] if is_ephemeral_cache else ["-e"]
    pip_check = subprocess.run(
        [str(target_py), "-m", "pip", "--version"],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        check=False,
        timeout=20,
    )
    if pip_check.returncode == 0 and not used_uv:
        install_cmd = [str(target_py), "-m", "pip", "install", "--upgrade", "pip"]
        subprocess.run(install_cmd, cwd=tempfile.gettempdir(), capture_output=True, check=False, timeout=180)
        pkg_cmd = [str(target_py), "-m", "pip", "install", *edit_args, install_spec]
    else:
        uv_bin = shutil.which("uv")
        if not uv_bin:
            return {
                "ok": False,
                "bootstrapped": False,
                "python_bin": str(target_py),
                "venv_dir": str(target_venv),
                "error": (
                    f"pip is unavailable inside {target_venv}. "
                    "On Debian/Ubuntu, install `python3-venv` (`sudo apt-get install -y python3-venv`) and re-run."
                ),
            }
        pkg_cmd = [uv_bin, "pip", "install", "--python", str(target_py), *edit_args, install_spec]

    proc = subprocess.run(
        pkg_cmd,
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if proc.returncode != 0 or not python_has_plugin_sdks(target_py):
        tail = (proc.stderr or proc.stdout or "").strip()[-600:]
        return {
            "ok": False,
            "bootstrapped": False,
            "python_bin": str(target_py),
            "venv_dir": str(target_venv),
            "error": f"Failed to install plugin dependencies into {target_venv}: {tail}",
        }
    return {
        "ok": True,
        "bootstrapped": True,
        "python_bin": str(target_py),
        "venv_dir": str(target_venv),
    }
