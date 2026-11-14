"""Zero-dependency PEP 517 / PEP 660 build backend bundling all multi-host plugin assets.

Works 100% offline with `--no-build-isolation` on stock Python 3.10+ even when `setuptools`
or `wheel` is not installed, while producing standard PEP 427 wheels (`uvx`, `pip`, `pipx`).
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile

_REPO_ROOT = Path(__file__).resolve().parent
_DIST_NAME = "cc_agy_plugin"
_PROJECT_NAME = "cc-agy-plugin"
_VERSION = "0.3.0"
_DIST_INFO = f"{_DIST_NAME}-{_VERSION}.dist-info"

_BUNDLED_ASSET_DIRS: tuple[str, ...] = (
    ".claude-plugin",
    ".cursor-plugin",
    "cursor",
    ".opencode",
    "opencode-plugin",
    "plugins",
    "commands",
    "skills",
    "agents",
    "hooks",
)
_BUNDLED_ASSET_FILES: tuple[str, ...] = (
    ".mcp.json",
    "package.json",
    "README.md",
)
_SKIP_NAMES: frozenset[str] = frozenset({"__pycache__", "node_modules", ".DS_Store", "_bundled_assets"})


def _dist_info_files() -> dict[str, bytes]:
    readme_text = (_REPO_ROOT / "README.md").read_text(encoding="utf-8") if (_REPO_ROOT / "README.md").is_file() else ""
    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {_PROJECT_NAME}\n"
        f"Version: {_VERSION}\n"
        "Summary: Claude Code, Cursor, OpenCode, Antigravity, and Codex plugin for delegating tasks to Gemini.\n"
        "License: Apache-2.0\n"
        "Requires-Python: >=3.10\n"
        "Requires-Dist: google-genai>=1.0.0\n"
        "Requires-Dist: pydantic>=2.0.0\n"
        "Provides-Extra: antigravity\n"
        "Requires-Dist: google-antigravity>=0.1.0; extra == 'antigravity'\n"
        "Requires-Dist: ctx-harness>=0.39.0; (python_version >= '3.11') and extra == 'antigravity'\n"
        "Provides-Extra: straitjacket\n"
        "Requires-Dist: ctx-harness>=0.39.0; (python_version >= '3.11') and extra == 'straitjacket'\n"
        "Provides-Extra: dev\n"
        "Requires-Dist: pytest>=8.0.0; extra == 'dev'\n"
        "Requires-Dist: pytest-xdist>=3.5.0; extra == 'dev'\n"
        "Requires-Dist: jsonschema>=4.20.0; extra == 'dev'\n"
        "Requires-Dist: ruff>=0.8.0; extra == 'dev'\n"
        "Description-Content-Type: text/markdown\n\n"
        f"{readme_text}"
    ).encode("utf-8")
    wheel = (
        "Wheel-Version: 1.0\n"
        "Generator: cc-agy-plugin-pep517\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    ).encode("utf-8")
    entry_points = (
        "[console_scripts]\n"
        "cc-agy-install = server.install:main\n"
        "cc-agy-plugin = server.install:main\n"
    ).encode("utf-8")
    return {
        f"{_DIST_INFO}/METADATA": metadata,
        f"{_DIST_INFO}/WHEEL": wheel,
        f"{_DIST_INFO}/entry_points.txt": entry_points,
    }


def _iter_tree_files(base_dir: Path) -> list[tuple[Path, str]]:
    collected: list[tuple[Path, str]] = []
    if not base_dir.is_dir():
        return collected
    for p in sorted(base_dir.rglob("*")):
        if any(part in _SKIP_NAMES or part.endswith(".pyc") for part in p.parts):
            continue
        if p.is_file():
            collected.append((p, p.relative_to(base_dir).as_posix()))
    return collected


def _record_digest(data: bytes) -> tuple[str, str]:
    b64 = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={b64}", str(len(data))


def get_requires_for_build_wheel(config_settings: dict | None = None) -> list[str]:
    return []


def get_requires_for_build_editable(config_settings: dict | None = None) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict | None = None,
) -> str:
    dist_dir = Path(metadata_directory) / _DIST_INFO
    dist_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, payload in _dist_info_files().items():
        out_path = Path(metadata_directory) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(payload)
    return _DIST_INFO


prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel


def _write_wheel(wheel_directory: str, *, editable: bool) -> str:
    wheel_name = f"{_DIST_NAME}-{_VERSION}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name
    entries: dict[str, bytes] = {}

    if editable:
        pth_payload = (str(_REPO_ROOT) + "\n").encode("utf-8")
        entries[f"{_DIST_NAME}.pth"] = pth_payload
    else:
        for pkg in ("server", "scripts"):
            for src_file, rel_in_pkg in _iter_tree_files(_REPO_ROOT / pkg):
                if pkg == "server" and not rel_in_pkg.endswith((".py", ".json")):
                    continue
                if pkg == "scripts" and not rel_in_pkg.endswith((".py", ".sh", ".ps1")):
                    continue
                entries[f"{pkg}/{rel_in_pkg}"] = src_file.read_bytes()
        for asset_dir in _BUNDLED_ASSET_DIRS:
            for src_file, rel_in_asset in _iter_tree_files(_REPO_ROOT / asset_dir):
                entries[f"server/_bundled_assets/{asset_dir}/{rel_in_asset}"] = src_file.read_bytes()
        for asset_file in _BUNDLED_ASSET_FILES:
            src_f = _REPO_ROOT / asset_file
            if src_f.is_file():
                entries[f"server/_bundled_assets/{asset_file}"] = src_f.read_bytes()

    entries.update(_dist_info_files())

    record_buf = io.StringIO()
    writer = csv.writer(record_buf, lineterminator="\n")
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname in sorted(entries):
            payload = entries[arcname]
            zf.writestr(arcname, payload)
            digest, length = _record_digest(payload)
            writer.writerow([arcname, digest, length])
        record_arc = f"{_DIST_INFO}/RECORD"
        writer.writerow([record_arc, "", ""])
        zf.writestr(record_arc, record_buf.getvalue().encode("utf-8"))
    return wheel_name


def build_wheel(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _write_wheel(wheel_directory, editable=False)


def build_editable(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _write_wheel(wheel_directory, editable=True)
