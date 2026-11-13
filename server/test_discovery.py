"""Language-aware test runner detection, related test discovery, and compile/import smoke builders."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any

from server.job_git_utils import MAX_ACCEPTANCE_COMMANDS, _git_raw

def _collect_workspace_node_bin_dirs(ws_root: Path, max_sub_dirs: int = 16) -> list[str]:
    """Return root `node_modules/.bin` followed by bounded 1-to-2-level monorepo subpackage `.bin` directories."""
    out: list[str] = []
    seen: set[str] = set()

    def _safe_is_dir(p: Path) -> bool:
        try:
            return p.is_dir()
        except OSError:
            return False

    def _add_bin(p: Path) -> None:
        if not _safe_is_dir(p):
            return
        resolved_str: str | None = None
        try:
            resolved_str = str(p.resolve())
        except OSError:
            resolved_str = None
        for cand in (str(p), resolved_str):
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)

    _add_bin(ws_root / "node_modules" / ".bin")
    skip_names = {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        "out",
        "coverage",
        "target",
    }
    sub_bins: list[Path] = []
    try:
        lvl1 = sorted(os.scandir(ws_root), key=lambda e: e.name)
    except OSError:
        lvl1 = []
    for e1 in lvl1:
        try:
            is_d1 = e1.is_dir(follow_symlinks=False)
        except OSError:
            is_d1 = False
        if e1.name.startswith(".") or e1.name in skip_names or not is_d1:
            continue
        p1 = Path(e1.path)
        b1 = p1 / "node_modules" / ".bin"
        if _safe_is_dir(b1):
            sub_bins.append(b1)
        try:
            lvl2 = sorted(os.scandir(p1), key=lambda e: e.name)
        except OSError:
            continue
        for e2 in lvl2:
            try:
                is_d2 = e2.is_dir(follow_symlinks=False)
            except OSError:
                is_d2 = False
            if e2.name.startswith(".") or e2.name in skip_names or not is_d2:
                continue
            b2 = Path(e2.path) / "node_modules" / ".bin"
            if _safe_is_dir(b2):
                sub_bins.append(b2)
    for sb in sub_bins[:max_sub_dirs]:
        _add_bin(sb)
    return out


def _build_workdir_exec_env(cwd: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment dict that prepends cwd (and cwd/src) to PYTHONPATH and node_modules/.bin + .venv/bin to PATH.

    Without ``base_env`` it starts from ``verify.credential_free_environ()``: every command this module
    runs in a workspace (acceptance, baseline probes, runner detection, smoke checks) gets an
    environment with plugin internals and credentials removed.
    """
    if base_env is None:
        from server.verify import credential_free_environ

        env = credential_free_environ()
    else:
        env = dict(base_env)
    cwd_resolved = cwd.resolve()
    py_parts = [str(cwd_resolved)]
    src_dir = cwd_resolved / "src"
    if src_dir.is_dir():
        py_parts.append(str(src_dir))
    existing_py = (env.get("PYTHONPATH") or "").strip()
    if existing_py:
        py_parts.append(existing_py)
    env["PYTHONPATH"] = ":".join(py_parts)
    path_prefixes: list[str] = _collect_workspace_node_bin_dirs(cwd_resolved)
    venv_dir = cwd_resolved / ".venv"
    venv_bin = venv_dir / "bin"
    if venv_bin.is_dir():
        path_prefixes.append(str(venv_bin))
        env["VIRTUAL_ENV"] = str(venv_dir)
    if path_prefixes:
        existing_path = (env.get("PATH") or "").strip()
        prefix_joined = ":".join(path_prefixes)
        env["PATH"] = f"{prefix_joined}:{existing_path}" if existing_path else prefix_joined
    return env


def _repo_has_explicit_pytest_config(cwd: Path) -> bool:
    """Return True when `cwd` declares explicit pytest configuration."""
    if (cwd / "pytest.ini").is_file() or (cwd / "conftest.py").is_file():
        return True
    for cfg_name, marker in (("pyproject.toml", "[tool.pytest"), ("setup.cfg", "[tool:pytest]"), ("tox.ini", "[pytest]")):
        cfg_p = cwd / cfg_name
        if cfg_p.is_file():
            try:
                if marker in cfg_p.read_text(encoding="utf-8", errors="replace"):
                    return True
            except OSError:
                pass
    return False


def _has_python_bin_test_runner(cwd: Path) -> bool:
    """Return True if `cwd/bin/test` is a Python test runner script."""
    bin_test = cwd / "bin" / "test"
    if not bin_test.is_file():
        return False
    try:
        head = bin_test.read_bytes()[:256].decode("utf-8", errors="replace")
    except OSError:
        return False
    first_line = head.splitlines()[0] if head else ""
    return "python" in first_line.lower()


def _detect_default_pytest_runner(cwd: Path) -> str:
    """Detect the best Python test runner prefix for `cwd` when the caller supplied no acceptance commands.
    Returns `""` if no compatible Python test runner is installed in `cwd`'s environment."""
    # Repositories with a dedicated Python `bin/test` runner and legacy `utilities/pytest.py` shims
    # (without modern pytest config) fail under modern pytest (`module 'py' has no attribute 'test'`).
    has_bin_test = _has_python_bin_test_runner(cwd)
    has_legacy_pytest_shim = (
        has_bin_test
        and not _repo_has_explicit_pytest_config(cwd)
        and next(cwd.glob("*/utilities/pytest.py"), None) is not None
    )
    if has_legacy_pytest_shim:
        for py_cand in (".venv/bin/python", "venv/bin/python"):
            if (cwd / py_cand).is_file():
                return f"{py_cand} bin/test"
        return "python3 bin/test"
    for rel_cand in (".venv/bin/pytest", "venv/bin/pytest"):
        if (cwd / rel_cand).is_file():
            return rel_cand
    exec_env = _build_workdir_exec_env(cwd)
    if shutil.which("pytest", path=exec_env.get("PATH")):
        return "pytest"
    for py_cand in (".venv/bin/python", "venv/bin/python"):
        if (cwd / py_cand).is_file():
            probe = subprocess.run(
                [str(cwd / py_cand), "-c", "import pytest"],
                cwd=str(cwd),
                env=exec_env,
                capture_output=True,
                check=False,
            )
            if probe.returncode == 0:
                return f"{py_cand} -m pytest"
    probe_sys = subprocess.run(
        ["python3", "-c", "import pytest"],
        cwd=str(cwd),
        env=exec_env,
        capture_output=True,
        check=False,
    )
    if probe_sys.returncode == 0:
        return "python3 -m pytest"
    if has_bin_test:
        for py_cand in (".venv/bin/python", "venv/bin/python"):
            if (cwd / py_cand).is_file():
                return f"{py_cand} bin/test"
        return "python3 bin/test"
    return ""


_JS_TS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_JS_TS_TEST_RE = re.compile(r"\.(?:test|spec)\.(?:ts|tsx|js|jsx|mjs|cjs)$", re.I)


def _detect_node_runner_info(cwd: Path, exec_env: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Inspect `package.json` and lockfiles to detect the Node/TypeScript test runner for `cwd`."""
    pkg_json = cwd / "package.json"
    if not pkg_json.is_file():
        return None
    env_map = exec_env if exec_env is not None else _build_workdir_exec_env(cwd)
    path_str = env_map.get("PATH")
    try:
        pkg = json.loads(pkg_json.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError, TypeError):
        pkg = {}
    if not isinstance(pkg, dict):
        pkg = {}
    scripts = pkg.get("scripts") if isinstance(pkg.get("scripts"), dict) else {}
    test_script = str((scripts or {}).get("test") or "").strip()
    is_placeholder = "no test specified" in test_script.lower()
    deps: dict[str, Any] = {}
    for dep_key in ("dependencies", "devDependencies"):
        sub = pkg.get(dep_key)
        if isinstance(sub, dict):
            deps.update(sub)
    pm_field = str(pkg.get("packageManager") or "").lower()
    if (cwd / "pnpm-lock.yaml").is_file() or pm_field.startswith("pnpm@"):
        pm = "pnpm"
    elif (cwd / "yarn.lock").is_file() or pm_field.startswith("yarn@"):
        pm = "yarn"
    elif (cwd / "bun.lockb").is_file() or (cwd / "bun.lock").is_file() or pm_field.startswith("bun@"):
        pm = "bun"
    else:
        pm = "npm"

    framework = ""
    if "vitest" in test_script or "vitest" in deps:
        framework = "vitest"
    elif "jest" in test_script or "jest" in deps:
        framework = "jest"
    elif "mocha" in test_script or "mocha" in deps:
        framework = "mocha"
    elif "node --test" in test_script:
        framework = "node-test"
    elif pm == "bun" and (not test_script or "bun test" in test_script):
        framework = "bun-test"

    node_bin_dir = cwd / "node_modules" / ".bin"
    pm_available = bool(shutil.which(pm, path=path_str))
    node_available = bool(shutil.which("node", path=path_str))
    if framework in ("vitest", "jest", "mocha"):
        local_bin = node_bin_dir / framework
        available = local_bin.is_file() or ((cwd / "node_modules").is_dir() and pm_available)
    elif framework == "node-test":
        available = node_available
    elif framework == "bun-test":
        available = bool(shutil.which("bun", path=path_str))
    else:
        available = bool(test_script and not is_placeholder and pm_available and (cwd / "node_modules").is_dir())

    primary_cmd = ""
    if framework == "bun-test":
        primary_cmd = "bun test"
    elif framework == "node-test":
        primary_cmd = "node --test"
    elif framework == "vitest" and test_script.strip() == "vitest":
        primary_cmd = "./node_modules/.bin/vitest run" if (node_bin_dir / "vitest").is_file() else "npx vitest run"
    elif test_script and not is_placeholder:
        primary_cmd = f"{pm} test"
    elif framework == "vitest":
        primary_cmd = "./node_modules/.bin/vitest run" if (node_bin_dir / "vitest").is_file() else "npx vitest run"
    elif framework == "jest":
        primary_cmd = (
            "./node_modules/.bin/jest --watchAll=false"
            if (node_bin_dir / "jest").is_file()
            else "npx jest --watchAll=false"
        )

    return {
        "ecosystem": "node",
        "package_manager": pm,
        "framework": framework or ("npm-script" if (test_script and not is_placeholder) else "none"),
        "primary_command": primary_cmd,
        "available": available,
    }


def detect_project_test_runner(cwd: Path) -> dict[str, Any]:
    """Detect repository language ecosystems and the best read-only test command for `cwd`."""
    root = Path(cwd).resolve()
    exec_env = _build_workdir_exec_env(root)
    path_str = exec_env.get("PATH")
    ecosystems: list[str] = []
    runners: list[dict[str, Any]] = []

    py_markers = ("pyproject.toml", "pytest.ini", "setup.py", "setup.cfg", "manage.py", "requirements.txt")
    has_py = (
        any((root / m).is_file() for m in py_markers)
        or _has_python_bin_test_runner(root)
        or next(root.glob("tests/test_*.py"), None) is not None
        or next(root.glob("*/tests/test_*.py"), None) is not None
    )
    if has_py:
        ecosystems.append("python")
        if (root / "manage.py").is_file() and not _repo_has_explicit_pytest_config(root):
            py_bin = ".venv/bin/python" if (root / ".venv/bin/python").is_file() else "python3"
            runners.append(
                {
                    "ecosystem": "python",
                    "framework": "django",
                    "primary_command": f"{py_bin} manage.py test",
                    "available": bool(shutil.which(py_bin, path=path_str) or (root / py_bin).is_file()),
                }
            )
        else:
            py_runner = _detect_default_pytest_runner(root)
            if py_runner:
                runners.append(
                    {
                        "ecosystem": "python",
                        "framework": "bin/test" if py_runner.endswith("bin/test") else "pytest",
                        "primary_command": py_runner if py_runner.endswith("bin/test") else f"{py_runner} -q",
                        "available": True,
                    }
                )
            else:
                runners.append(
                    {
                        "ecosystem": "python",
                        "framework": "unittest",
                        "primary_command": "python3 -m unittest discover",
                        "available": bool(shutil.which("python3", path=path_str)),
                    }
                )

    node_info = _detect_node_runner_info(root, exec_env)
    if node_info is not None:
        ecosystems.append("node")
        if node_info.get("primary_command"):
            runners.append(node_info)

    if (root / "go.mod").is_file():
        ecosystems.append("go")
        runners.append(
            {
                "ecosystem": "go",
                "framework": "go-test",
                "primary_command": "go test ./...",
                "available": bool(shutil.which("go", path=path_str)),
            }
        )

    if (root / "Cargo.toml").is_file():
        ecosystems.append("rust")
        runners.append(
            {
                "ecosystem": "rust",
                "framework": "cargo-test",
                "primary_command": "cargo test --quiet",
                "available": bool(shutil.which("cargo", path=path_str)),
            }
        )

    if (root / "gradlew").is_file() or (root / "build.gradle").is_file() or (root / "build.gradle.kts").is_file():
        ecosystems.append("jvm")
        use_wrapper = (root / "gradlew").is_file() and os.access(root / "gradlew", os.X_OK)
        cmd_jvm = "./gradlew test" if use_wrapper else "gradle test"
        avail_jvm = (use_wrapper and bool(shutil.which("java", path=path_str))) or bool(
            shutil.which("gradle", path=path_str)
        )
        runners.append(
            {
                "ecosystem": "jvm",
                "framework": "gradle",
                "primary_command": cmd_jvm,
                "available": avail_jvm,
            }
        )
    elif (root / "pom.xml").is_file():
        ecosystems.append("jvm")
        use_mvnw = (root / "mvnw").is_file() and os.access(root / "mvnw", os.X_OK)
        cmd_mvn = "./mvnw -q test" if use_mvnw else "mvn -q test"
        avail_mvn = (use_mvnw and bool(shutil.which("java", path=path_str))) or bool(
            shutil.which("mvn", path=path_str)
        )
        runners.append(
            {
                "ecosystem": "jvm",
                "framework": "maven",
                "primary_command": cmd_mvn,
                "available": avail_mvn,
            }
        )

    makefile_p = root / "Makefile"
    if makefile_p.is_file():
        try:
            mk_txt = makefile_p.read_text(encoding="utf-8", errors="replace")
            if re.search(r"(?m)^test\s*:", mk_txt):
                if "make" not in ecosystems:
                    ecosystems.append("make")
                runners.append(
                    {
                        "ecosystem": "make",
                        "framework": "make",
                        "primary_command": "make test",
                        "available": bool(shutil.which("make", path=path_str)),
                    }
                )
        except OSError:
            pass

    primary = next((r for r in runners if r.get("available")), runners[0] if runners else None)
    return {
        "ecosystems": ecosystems,
        "primary_command": str(primary.get("primary_command") or "") if primary else "",
        "framework": str(primary.get("framework") or "") if primary else "",
        "available": bool(primary.get("available")) if primary else False,
        "runners": runners,
    }


def _list_tracked_repo_files(cwd: Path) -> list[str]:
    try:
        tracked_res = _git_raw(
            ["ls-files"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if tracked_res.returncode == 0 and tracked_res.stdout.strip():
            return [line.strip() for line in tracked_res.stdout.splitlines() if line.strip()]
    except OSError:
        pass
    skip_dirs = {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "target",
        "dist",
        "build",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }
    fallback: list[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(cwd):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".agy")]
            rel_dir = Path(dirpath).relative_to(cwd)
            for fname in sorted(filenames):
                rel_p = (rel_dir / fname).as_posix() if rel_dir != Path(".") else fname
                fallback.append(rel_p)
                if len(fallback) >= 5000:
                    return fallback
    except OSError:
        return []
    return fallback


def discover_related_test_commands(
    cwd: Path,
    files_changed: list[str],
    existing_cmds: list[str],
    *,
    max_files: int = 6,
) -> list[str]:
    """Discover targeted unit/regression test commands across Python, Node/TS, Go, Rust, and JVM
    for modified source or test files that are not already covered by `existing_cmds`."""
    if not files_changed:
        return []
    exec_env = _build_workdir_exec_env(cwd)
    path_str = exec_env.get("PATH")
    tracked_cache: list[str] | None = None

    def _get_tracked() -> list[str]:
        nonlocal tracked_cache
        if tracked_cache is None:
            tracked_cache = _list_tracked_repo_files(cwd)
        return tracked_cache

    out_cmds: list[str] = []

    # --- 1. Python (.py) ---
    py_changed = [
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if rel.replace("\\", "/").strip("/").endswith(".py")
    ]
    if py_changed:
        pytest_cmds: list[str] = []
        skip_py = False
        if existing_cmds:
            pytest_cmds = [c for c in existing_cmds if re.search(r"\bpytest\b", c)]
            if not pytest_cmds:
                skip_py = True
            else:
                for pcmd in pytest_cmds:
                    if not re.search(r"[\w./-]+\.py\b", pcmd) and not re.search(r"(?:^|\s)-k(?:\s|=|$)", pcmd):
                        skip_py = True
                        break
        if not skip_py:
            discovered_py: list[str] = []
            stems_to_check: list[str] = []
            pkg_test_names: set[str] = set()
            mirrored_test_dirs: list[str] = []
            for norm in py_changed:
                bname = Path(norm).name
                if bname.startswith("test_") or bname.endswith("_test.py"):
                    if not existing_cmds and (cwd / norm).is_file() and norm not in discovered_py:
                        discovered_py.append(norm)
                    continue
                if bname in ("__init__.py", "conftest.py", "setup.py"):
                    continue
                parts = Path(norm).parts
                if any(p in ("tests", "test", "docs", "examples", "benchmarks") for p in parts[:-1]):
                    continue
                stem = Path(norm).stem
                if stem and stem not in stems_to_check:
                    stems_to_check.append(stem)
                dir_parts = list(parts[:-1])
                if dir_parts and dir_parts[0] in ("src", "lib", "python"):
                    dir_parts = dir_parts[1:]
                for pkg_seg in dir_parts:
                    if pkg_seg and pkg_seg.isidentifier():
                        pkg_test_names.add(f"test_{pkg_seg}.py")
                        pkg_test_names.add(f"{pkg_seg}_test.py")
                if dir_parts:
                    rel_pkg = "/".join(dir_parts)
                    for t_root in ("tests", "test"):
                        for cand_mir in (f"{t_root}/{rel_pkg}", *( [f"{t_root}/{'/'.join(dir_parts[1:])}"] if len(dir_parts) > 1 else [] )):
                            if cand_mir not in mirrored_test_dirs:
                                mirrored_test_dirs.append(cand_mir)

            def _is_py_test_covered(tf_rel: str) -> bool:
                tf_bname = Path(tf_rel).name
                return any(
                    re.search(rf"(?:^|[\s/\"'=]){re.escape(tf_bname)}(?:$|[\s\"':])", c)
                    or re.search(rf"(?:^|[\s\"'=]){re.escape(tf_rel)}(?:$|[\s\"':])", c)
                    for c in pytest_cmds
                )

            if stems_to_check and len(discovered_py) < max_files:
                for stem in stems_to_check:
                    target_names = {f"test_{stem}.py", f"{stem}_test.py"}
                    for tf in _get_tracked():
                        tf_name = Path(tf).name
                        if tf_name in target_names and tf not in discovered_py:
                            if not _is_py_test_covered(tf) and (cwd / tf).is_file():
                                discovered_py.append(tf)
                                if len(discovered_py) >= max_files:
                                    break
                    if len(discovered_py) >= max_files:
                        break

            # Include sibling package-level test files (e.g. tests/<pkg>/<subpkg>/test_<subpkg>.py)
            # and compact mirrored test directories (<= 8 test files) so package round-trip suites run.
            if (pkg_test_names or mirrored_test_dirs) and len(discovered_py) < max_files:
                tracked_list = _get_tracked()
                for tf in tracked_list:
                    if len(discovered_py) >= max_files:
                        break
                    tf_parent = Path(tf).parent.as_posix()
                    tf_name = Path(tf).name
                    if tf_name in pkg_test_names and (
                        tf_parent in mirrored_test_dirs
                        or tf.startswith(("tests/", "test/"))
                    ):
                        if tf not in discovered_py and not _is_py_test_covered(tf) and (cwd / tf).is_file():
                            discovered_py.append(tf)
                for mdir in mirrored_test_dirs:
                    if len(discovered_py) >= max_files:
                        break
                    dir_tests = [
                        tf
                        for tf in tracked_list
                        if Path(tf).parent.as_posix() == mdir
                        and (Path(tf).name.startswith("test_") or Path(tf).name.endswith("_test.py"))
                        and (cwd / tf).is_file()
                    ]
                    if 0 < len(dir_tests) <= 8:
                        for tf in dir_tests:
                            if len(discovered_py) >= max_files:
                                break
                            if tf not in discovered_py and not _is_py_test_covered(tf):
                                discovered_py.append(tf)

            if discovered_py:
                discovered_py = discovered_py[:max_files]
                if pytest_cmds:
                    first_pytest = pytest_cmds[0].strip()
                    m_pref = re.match(r"^(.*?\bpytest)\b", first_pytest)
                    runner_prefix = m_pref.group(1).strip() if m_pref else "pytest"
                else:
                    runner_prefix = _detect_default_pytest_runner(cwd)
                quoted_files = " ".join(shlex.quote(f) for f in discovered_py)
                if runner_prefix:
                    if runner_prefix.endswith("bin/test"):
                        out_cmds.append(f"{runner_prefix} {quoted_files}")
                    else:
                        out_cmds.append(f"{runner_prefix} -q {quoted_files}")
                elif not existing_cmds and shutil.which("python3", path=path_str):
                    out_cmds.append(f"python3 -m unittest {quoted_files}")

    # --- 2. Node.js / TypeScript (.ts, .tsx, .js, .jsx, .mjs, .cjs) ---
    js_changed = [
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if rel.replace("\\", "/").strip("/").lower().endswith(_JS_TS_EXTS)
    ]
    if js_changed and (cwd / "package.json").is_file():
        node_info = _detect_node_runner_info(cwd, exec_env)
        if node_info and node_info.get("available"):
            fw = str(node_info.get("framework") or "")
            pm = str(node_info.get("package_manager") or "npm")
            has_existing_node_cmd = any(
                re.search(r"\b(?:npm|pnpm|yarn|bun|npx|vitest|jest|mocha|node\s+--test)\b", c)
                for c in existing_cmds
            )
            # Only synthesize a Node test command when caller passed none, or when caller ran targeted
            # vitest/jest test files and missed a sibling `<stem>.test.*` / `<stem>.spec.*`.
            specific_js_cmds = [
                c
                for c in existing_cmds
                if re.search(r"\b(?:vitest|jest)\b", c) and _JS_TS_TEST_RE.search(c)
            ]
            if not existing_cmds or (has_existing_node_cmd and specific_js_cmds and len(specific_js_cmds) == len(existing_cmds)):
                discovered_js: list[str] = []
                js_stems: list[str] = []
                for norm in js_changed:
                    if _JS_TS_TEST_RE.search(norm):
                        if not existing_cmds and (cwd / norm).is_file() and norm not in discovered_js:
                            discovered_js.append(norm)
                        continue
                    bname = Path(norm).name
                    if bname.startswith(("index.", "vite.config.", "vitest.config.", "jest.config.", "eslint.config.")):
                        continue
                    stem = Path(norm).stem
                    if stem and stem not in js_stems:
                        js_stems.append(stem)
                if js_stems and len(discovered_js) < max_files:
                    for stem in js_stems:
                        cand_names = {
                            f"{stem}.test{ext}" for ext in _JS_TS_EXTS
                        } | {
                            f"{stem}.spec{ext}" for ext in _JS_TS_EXTS
                        }
                        for tf in _get_tracked():
                            tf_name = Path(tf).name
                            if tf_name in cand_names and tf not in discovered_js:
                                already_cov = any(
                                    re.search(rf"(?:^|[\s/\"'=]){re.escape(tf_name)}(?:$|[\s\"':])", c)
                                    for c in existing_cmds
                                )
                                if not already_cov and (cwd / tf).is_file():
                                    discovered_js.append(tf)
                                    if len(discovered_js) >= max_files:
                                        break
                        if len(discovered_js) >= max_files:
                            break
                if discovered_js:
                    q_js = " ".join(shlex.quote(f) for f in discovered_js[:max_files])
                    if fw == "vitest":
                        bin_v = "./node_modules/.bin/vitest" if (cwd / "node_modules/.bin/vitest").is_file() else "npx vitest"
                        out_cmds.append(f"{bin_v} run {q_js}")
                    elif fw == "jest":
                        bin_j = "./node_modules/.bin/jest" if (cwd / "node_modules/.bin/jest").is_file() else "npx jest"
                        out_cmds.append(f"{bin_j} --watchAll=false {q_js}")
                    elif fw == "bun-test":
                        out_cmds.append(f"bun test {q_js}")
                    elif fw == "node-test":
                        out_cmds.append(f"node --test {q_js}")
                    elif not existing_cmds and node_info.get("primary_command"):
                        out_cmds.append(str(node_info["primary_command"]))
                elif not existing_cmds and node_info.get("primary_command") and pm in ("npm", "pnpm", "yarn", "bun"):
                    out_cmds.append(str(node_info["primary_command"]))

    # --- 3. Go (.go) ---
    go_changed = [
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if rel.replace("\\", "/").strip("/").endswith(".go")
    ]
    if go_changed and (cwd / "go.mod").is_file() and shutil.which("go", path=path_str):
        go_cmds = [c for c in existing_cmds if re.search(r"\bgo\s+test\b", c)]
        full_go_suite = any("./..." in c for c in go_cmds)
        if (not existing_cmds) or (go_cmds and not full_go_suite):
            pkg_dirs: list[str] = []
            for norm in go_changed:
                parent_rel = Path(norm).parent.as_posix()
                pkg_dir = cwd if parent_rel in ("", ".") else (cwd / parent_rel)
                if not pkg_dir.is_dir() or next(pkg_dir.glob("*_test.go"), None) is None:
                    continue
                pkg_spec = "." if parent_rel in ("", ".") else f"./{parent_rel}"
                already_cov = any(
                    re.search(rf"(?:^|\s){re.escape(pkg_spec)}(?:$|/|\s)", c) for c in go_cmds
                )
                if not already_cov and pkg_spec not in pkg_dirs:
                    pkg_dirs.append(pkg_spec)
                    if len(pkg_dirs) >= max_files:
                        break
            if pkg_dirs:
                out_cmds.append("go test " + " ".join(shlex.quote(p) for p in pkg_dirs))

    # --- 4. Rust (.rs) ---
    rs_changed = [
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if rel.replace("\\", "/").strip("/").endswith(".rs")
    ]
    if not existing_cmds and rs_changed and (cwd / "Cargo.toml").is_file() and shutil.which("cargo", path=path_str):
        crate_names: list[str] = []
        for norm in rs_changed:
            cur_dir = (cwd / norm).parent
            while cur_dir != cwd and cur_dir.is_relative_to(cwd):
                sub_toml = cur_dir / "Cargo.toml"
                if sub_toml.is_file():
                    try:
                        toml_txt = sub_toml.read_text(encoding="utf-8", errors="replace")
                        m_pkg = re.search(r'(?ms)^\[package\].*?^\s*name\s*=\s*["\']([A-Za-z0-9_-]+)["\']', toml_txt)
                        if m_pkg and m_pkg.group(1) not in crate_names:
                            crate_names.append(m_pkg.group(1))
                    except OSError:
                        pass
                    break
                cur_dir = cur_dir.parent
        if crate_names:
            pkg_flags = " ".join(f"-p {shlex.quote(cn)}" for cn in crate_names[:max_files])
            out_cmds.append(f"cargo test --quiet {pkg_flags}")
        else:
            out_cmds.append("cargo test --quiet")

    # --- 5. Java / Kotlin (.java, .kt) ---
    jvm_changed = [
        rel.replace("\\", "/").strip("/")
        for rel in files_changed
        if rel.replace("\\", "/").strip("/").lower().endswith((".java", ".kt"))
    ]
    if not existing_cmds and jvm_changed:
        jvm_test_classes: list[str] = []
        for norm in jvm_changed:
            stem = Path(norm).stem
            if not stem:
                continue
            if stem.endswith(("Test", "Tests", "IT")):
                if stem not in jvm_test_classes:
                    jvm_test_classes.append(stem)
                continue
            cand_jvm = {f"{stem}Test.java", f"{stem}Tests.java", f"{stem}Test.kt", f"{stem}Tests.kt"}
            for tf in _get_tracked():
                tf_bname = Path(tf).name
                if tf_bname in cand_jvm and (cwd / tf).is_file():
                    cls_name = Path(tf_bname).stem
                    if cls_name not in jvm_test_classes:
                        jvm_test_classes.append(cls_name)
                        break
        jvm_test_classes = jvm_test_classes[:max_files]
        is_multi_gradle = False
        for sg_name in ("settings.gradle", "settings.gradle.kts"):
            sg_p = cwd / sg_name
            if sg_p.is_file():
                try:
                    if re.search(r"\binclude\s*[\(\"'']", sg_p.read_text(encoding="utf-8", errors="replace")):
                        is_multi_gradle = True
                        break
                except OSError:
                    pass
        if (cwd / "gradlew").is_file() and os.access(cwd / "gradlew", os.X_OK) and shutil.which("java", path=path_str):
            if jvm_test_classes and not is_multi_gradle:
                flags = " ".join(f"--tests {shlex.quote('*' + c)}" for c in jvm_test_classes)
                out_cmds.append(f"./gradlew test {flags}")
            else:
                out_cmds.append("./gradlew test")
        elif ((cwd / "build.gradle").is_file() or (cwd / "build.gradle.kts").is_file()) and shutil.which("gradle", path=path_str):
            if jvm_test_classes and not is_multi_gradle:
                flags = " ".join(f"--tests {shlex.quote('*' + c)}" for c in jvm_test_classes)
                out_cmds.append(f"gradle test {flags}")
            else:
                out_cmds.append("gradle test")
        elif (cwd / "pom.xml").is_file():
            use_mvnw = (cwd / "mvnw").is_file() and os.access(cwd / "mvnw", os.X_OK) and bool(shutil.which("java", path=path_str))
            mvn_bin = "./mvnw" if use_mvnw else ("mvn" if shutil.which("mvn", path=path_str) else "")
            if mvn_bin:
                if jvm_test_classes:
                    out_cmds.append(
                        f"{mvn_bin} -q test -Dtest={','.join(jvm_test_classes)} "
                        f"-Dsurefire.failIfNoSpecifiedTests=false -DfailIfNoTests=false"
                    )
                else:
                    out_cmds.append(f"{mvn_bin} -q test")

    return out_cmds[:MAX_ACCEPTANCE_COMMANDS]


_IMPORT_SKIP_DIRS = frozenset({
    "tests", "test", "docs", "examples", "benchmarks", "scripts", "migrations", "tools", "bin",
})
_IMPORT_SKIP_FILES = frozenset({
    "setup.py", "conftest.py", "__main__.py", "manage.py", "noxfile.py", "tasks.py", "fabfile.py",
    "wsgi.py", "asgi.py", "sitecustomize.py", "usercustomize.py",
})


def build_import_smoke_command(
    cwd: Path,
    files_changed: list[str],
    *,
    max_modules: int = 8,
) -> list[str]:
    """Build a fast Python import smoke-check command for modified or created Python modules."""
    if not files_changed:
        return []
    modules: list[str] = []
    for rel in files_changed:
        norm = rel.replace("\\", "/").strip("/")
        if not norm.endswith(".py") or not (cwd / norm).is_file():
            continue
        bname = Path(norm).name
        if bname in _IMPORT_SKIP_FILES or bname.startswith("test_") or bname.endswith("_test.py"):
            continue
        parts = list(Path(norm).parts)
        if any(p in _IMPORT_SKIP_DIRS or p.startswith(".") for p in parts[:-1]):
            continue
        base_dir = cwd
        if parts[0] == "src" and len(parts) > 1:
            base_dir = cwd / "src"
            parts = parts[1:]
        if len(parts) <= 1:
            # Never import top-level non-package .py scripts (e.g. run.py), which can execute CLI/server side effects.
            continue
        # Verify every parent directory is a valid Python package with __init__.py
        cur = base_dir
        valid_pkg = True
        for pkg_part in parts[:-1]:
            cur = cur / pkg_part
            if not (cur / "__init__.py").is_file() or not pkg_part.isidentifier():
                valid_pkg = False
                break
        if not valid_pkg:
            continue
        if bname == "__init__.py":
            mod_name = ".".join(parts[:-1])
        else:
            stem = Path(bname).stem
            if not stem.isidentifier():
                continue
            mod_name = ".".join((*parts[:-1], stem))
        if mod_name and mod_name not in modules:
            modules.append(mod_name)
            if len(modules) >= max_modules:
                break
    if not modules:
        return []
    py_bin = "python3"
    for cand in (".venv/bin/python", "venv/bin/python"):
        if (cwd / cand).is_file():
            py_bin = cand
            break
    return [
        f"{py_bin} -c 'import {', '.join(modules)}'"
    ]


# Output signatures that mean a compile check could not run in this environment (missing toolchain,
# offline dependency fetch, no lockfile entry), as opposed to the worker's code failing to compile.
COMPILE_SMOKE_ENV_FAILURE_RE = re.compile(
    r"command not found|not recognized as an internal or external command|"
    r"cannot find module 'typescript'|could not resolve host|failed to download|"
    r"no matching package named|unable to update registry|dial tcp|network is unreachable|"
    r"missing go\.sum entry|cannot find module providing package|"
    r"attempting to make an http request, but --offline|failed to get .* as a dependency",
    re.I,
)

# Exits 0 unless an error line from the compiler names one of the files passed as arguments, so errors that
# already existed elsewhere in the project never fail a smoke check for an unrelated change.
_CHANGED_FILE_ERROR_FILTER = (
    "import re,sys\n"
    "rc=int(sys.argv[1]); files=sys.argv[2:]; out=sys.stdin.read()\n"
    "pats=[re.compile(r'^\\s*(?:-->\\s*)?(?:\\./)?'+re.escape(f)+r'(?::\\d+|\\(\\d+,\\d+\\))') for f in files]\n"
    "hits=[l for l in out.splitlines() if any(p.search(l) for p in pats)]\n"
    "print('\\n'.join(hits[-40:]) if hits else out[-2000:])\n"
    "sys.exit(1 if rc and hits else 0)\n"
)


def build_compile_smoke_commands(
    cwd: Path,
    files_changed: list[str],
    *,
    max_files: int = 8,
) -> list[str]:
    """Build fast compile/syntax checks for modified files across common languages.

    These run when a job has no explicit acceptance commands (for example a repository without tests), so
    a change that does not parse or compile still fails acceptance and triggers escalation instead of
    passing silently. Every command is read-only, uses toolchains that are already installed in the
    workspace or on PATH, and never downloads packages. Results are recorded as smoke checks, never as an
    explicit test pass.
    """
    if not files_changed:
        return []
    exec_env = _build_workdir_exec_env(cwd)
    path_str = exec_env.get("PATH")
    norm_files: list[str] = []
    for rel in files_changed:
        norm = rel.replace("\\", "/").strip("/")
        if norm and norm not in norm_files and (cwd / norm).is_file():
            norm_files.append(norm)

    def _pick(exts: tuple[str, ...]) -> list[str]:
        return [f for f in norm_files if f.lower().endswith(exts)][:max_files]

    def _quoted(files: list[str]) -> str:
        return " ".join(shlex.quote(f) for f in files)

    py_bin = "python3"
    for cand in (".venv/bin/python", "venv/bin/python"):
        if (cwd / cand).is_file():
            py_bin = cand
            break

    cmds: list[str] = []

    # Python: parse every modified file, including top-level scripts the import smoke check skips.
    py_files = _pick((".py",))
    if py_files:
        parse_script = "import ast,sys\nfor p in sys.argv[1:]:\n    ast.parse(open(p,'rb').read(), p)\n"
        cmds.append(f"{py_bin} -c {shlex.quote(parse_script)} {_quoted(py_files)}")

    # TypeScript: type-check the project with the repository's own tsc, failing only on changed files.
    ts_files = _pick((".ts", ".tsx", ".mts", ".cts"))
    tsc_bin = cwd / "node_modules" / ".bin" / "tsc"
    if ts_files and (cwd / "tsconfig.json").is_file() and tsc_bin.is_file():
        cmds.append(
            "out=$(./node_modules/.bin/tsc --noEmit --pretty false -p tsconfig.json 2>&1); rc=$?; "
            f"printf '%s\\n' \"$out\" | python3 -c {shlex.quote(_CHANGED_FILE_ERROR_FILTER)} \"$rc\" "
            f"{_quoted(ts_files)}"
        )

    # Plain JavaScript: syntax check with node (JSX needs a build step, so it is skipped).
    js_files = _pick((".js", ".mjs", ".cjs"))
    if js_files and shutil.which("node", path=path_str):
        cmds.append(" && ".join(f"node --check {shlex.quote('./' + f)}" for f in js_files))

    # Go: compile the changed packages without writing a binary (test-only edits are left to `go test`).
    go_files = [f for f in _pick((".go",)) if not f.endswith("_test.go")]
    if go_files and (cwd / "go.mod").is_file() and shutil.which("go", path=path_str):
        pkgs: list[str] = []
        for f in go_files:
            parent = Path(f).parent.as_posix()
            spec = "." if parent in ("", ".") else f"./{parent}"
            if spec not in pkgs:
                pkgs.append(spec)
        cmds.append(f"go build -o /dev/null {' '.join(shlex.quote(p) for p in pkgs)}")

    # Rust: only with a warm host target directory, so the check never triggers a cold full build.
    rs_files = _pick((".rs",))
    if rs_files and (cwd / "Cargo.toml").is_file() and shutil.which("cargo", path=path_str):
        if (cwd / "target").is_dir():
            cmds.append("cargo check --offline --quiet")

    # Shell scripts and JSON documents (JSON-with-comments config files are skipped).
    sh_files = _pick((".sh", ".bash"))
    if sh_files and shutil.which("bash", path=path_str):
        cmds.append(" && ".join(f"bash -n {shlex.quote('./' + f)}" for f in sh_files))
    json_files = [
        f
        for f in _pick((".json",))
        if not re.match(r"^(?:tsconfig|jsconfig|\.eslintrc|devcontainer|\.babelrc)", Path(f).name, re.I)
        and not any(part in (".vscode", ".devcontainer") for part in Path(f).parts[:-1])
    ]
    if json_files:
        json_script = "import json,sys\nfor p in sys.argv[1:]:\n    json.load(open(p,encoding='utf-8-sig'))\n"
        cmds.append(f"python3 -c {shlex.quote(json_script)} {_quoted(json_files)}")

    return cmds

