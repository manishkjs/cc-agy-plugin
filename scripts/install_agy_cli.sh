#!/usr/bin/env bash
# Hardened installer for the standalone Antigravity CLI (`agy`).
# Downloads the platform manifest from the official Cloud Run auto-updater,
# verifies SHA-512 integrity before extraction, blocks archive path traversal,
# and installs atomically into a managed plugin directory without mutating
# user shell dotfiles unless `--configure-shell` is explicitly requested.
set -euo pipefail

DOWNLOAD_BASE_URL="${CC_AGY_UPDATER_BASE_URL:-https://antigravity-cli-auto-updater-974169037036.us-central1.run.app}"
DEFAULT_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
TARGET_DIR="${CC_AGY_CLI_INSTALL_DIR:-$DEFAULT_DATA_HOME/cc-agy-plugin/bin}"
MANIFEST_URL=""
CONFIGURE_SHELL=0
JSON_OUTPUT=0

usage() {
  cat <<'EOF'
Usage: install_agy_cli.sh [options]

Options:
  --target-dir <dir>     Install directory for `agy` (default: ~/.local/share/cc-agy-plugin/bin)
  --manifest-url <url>   Override release manifest URL (default: official auto-updater)
  --configure-shell      Run `agy install` after placement to update shell PATH/aliases
  --json                 Print machine-readable JSON summary to stdout
  -h, --help             Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir)
      [[ $# -ge 2 ]] || { echo "error: --target-dir requires a value" >&2; exit 2; }
      TARGET_DIR="$2"
      shift 2
      ;;
    --target-dir=*)
      TARGET_DIR="${1#*=}"
      shift
      ;;
    --manifest-url)
      [[ $# -ge 2 ]] || { echo "error: --manifest-url requires a value" >&2; exit 2; }
      MANIFEST_URL="$2"
      shift 2
      ;;
    --manifest-url=*)
      MANIFEST_URL="${1#*=}"
      shift
      ;;
    --configure-shell)
      CONFIGURE_SHELL=1
      shift
      ;;
    --json)
      JSON_OUTPUT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TARGET_DIR" ]]; then
  echo "error: target directory cannot be empty" >&2
  exit 2
fi

PYTHON_BIN="${CC_AGY_INSTALLER_PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: python3 is required to verify manifest and SHA-512 checksum" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required to download Antigravity CLI" >&2
  exit 1
fi

detect_platform() {
  if [[ -n "${CC_AGY_CLI_PLATFORM:-}" ]]; then
    printf '%s\n' "$CC_AGY_CLI_PLATFORM"
    return 0
  fi

  local os_name arch_name norm_arch is_musl
  os_name="$(uname -s)"
  arch_name="$(uname -m)"

  case "$arch_name" in
    x86_64|amd64) norm_arch="amd64" ;;
    arm64|aarch64) norm_arch="arm64" ;;
    *)
      echo "error: unsupported CPU architecture: $arch_name" >&2
      return 1
      ;;
  esac

  case "$os_name" in
    Darwin)
      printf 'darwin_%s\n' "$norm_arch"
      ;;
    Linux)
      if [[ -n "${TERMUX_VERSION:-}" ]] || [[ "$(uname -o 2>/dev/null || true)" == "Android" ]]; then
        if [[ "$norm_arch" != "arm64" ]]; then
          echo "error: unsupported Android architecture: $arch_name" >&2
          return 1
        fi
        printf 'android_arm64\n'
        return 0
      fi
      is_musl=0
      if compgen -G "/lib/ld-musl-*" >/dev/null 2>&1; then
        is_musl=1
      elif command -v ldd >/dev/null 2>&1; then
        if ldd --version 2>&1 | "$PYTHON_BIN" -c 'import sys; sys.exit(0 if "musl" in sys.stdin.read().lower() else 1)'; then
          is_musl=1
        fi
      fi
      if [[ "$is_musl" -eq 1 ]]; then
        printf 'linux_%s_musl\n' "$norm_arch"
      else
        printf 'linux_%s\n' "$norm_arch"
      fi
      ;;
    *)
      echo "error: unsupported operating system: $os_name" >&2
      return 1
      ;;
  esac
}

PLATFORM="$(detect_platform)"
if [[ -z "$MANIFEST_URL" ]]; then
  MANIFEST_URL="${DOWNLOAD_BASE_URL%/}/manifests/${PLATFORM}.json"
fi

"$PYTHON_BIN" - "$MANIFEST_URL" "${CC_AGY_ALLOW_HTTP_MANIFEST:-0}" <<'PY'
import sys

manifest_url = sys.argv[1].strip()
allow_http = sys.argv[2] == "1"
allowed_schemes = ("https://", "file://") + (("http://",) if allow_http else ())
if not manifest_url or any(ord(ch) <= 32 for ch in manifest_url):
    raise SystemExit("error: manifest URL contains invalid empty or control/whitespace characters")
if not manifest_url.startswith(allowed_schemes):
    raise SystemExit(f"error: manifest URL {manifest_url!r} must use https://")
PY

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cc-agy-cli-install.XXXXXX")"
STAGE_PATH=""
cleanup() {
  rm -rf "$TMP_DIR"
  if [[ -n "$STAGE_PATH" && -e "$STAGE_PATH" ]]; then
    rm -f "$STAGE_PATH"
  fi
}
trap cleanup EXIT

fetch_url() {
  local src_url="$1"
  local dest_file="$2"
  if [[ "$src_url" == https://* ]]; then
    curl -fsSL --proto '=https' --proto-redir '=https' --connect-timeout 15 --max-time 180 "$src_url" -o "$dest_file"
  else
    curl -fsSL --connect-timeout 15 --max-time 180 "$src_url" -o "$dest_file"
  fi
}

fetch_url "$MANIFEST_URL" "$TMP_DIR/manifest.json"

PARSED_MANIFEST="$(
  "$PYTHON_BIN" - "$TMP_DIR/manifest.json" "${CC_AGY_ALLOW_HTTP_MANIFEST:-0}" <<'PY'
import json
import re
import sys

manifest_path = sys.argv[1]
allow_http = sys.argv[2] == "1"

try:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
except (OSError, ValueError) as exc:
    raise SystemExit(f"error: manifest is not valid JSON ({exc})")

if not isinstance(data, dict):
    raise SystemExit("error: manifest root must be a JSON object")

version = str(data.get("version") or "").strip()
url = str(data.get("url") or "").strip()
sha512 = str(data.get("sha512") or "").strip().lower()

if not version or any(ord(ch) < 32 for ch in version):
    raise SystemExit("error: manifest missing valid 'version'")
if not url or any(ord(ch) <= 32 for ch in url):
    raise SystemExit("error: manifest 'url' contains invalid empty or control/whitespace characters")
allowed_schemes = ("https://", "file://") + (("http://",) if allow_http else ())
if not url.startswith(allowed_schemes):
    raise SystemExit(f"error: manifest URL {url!r} must use https://")
if not re.fullmatch(r"[0-9a-f]{128}", sha512):
    raise SystemExit("error: manifest 'sha512' must be a 128-character hex digest")

print(version)
print(url)
print(sha512)
PY
)"

VERSION="$(printf '%s\n' "$PARSED_MANIFEST" | sed -n '1p')"
ARCHIVE_URL="$(printf '%s\n' "$PARSED_MANIFEST" | sed -n '2p')"
EXPECTED_SHA512="$(printf '%s\n' "$PARSED_MANIFEST" | sed -n '3p')"

ARCHIVE_PATH="$TMP_DIR/agy.tar.gz"
EXTRACT_DIR="$TMP_DIR/extract"
mkdir -p "$EXTRACT_DIR"

fetch_url "$ARCHIVE_URL" "$ARCHIVE_PATH"

EXTRACTED_BIN="$(
  "$PYTHON_BIN" - "$ARCHIVE_PATH" "$EXPECTED_SHA512" "$EXTRACT_DIR" <<'PY'
import hashlib
import os
from pathlib import Path
import stat
import sys
import tarfile

archive_path = Path(sys.argv[1])
expected_sha = sys.argv[2].strip().lower()
extract_dir = Path(sys.argv[3]).resolve()

h = hashlib.sha512()
with archive_path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
        h.update(chunk)
actual_sha = h.hexdigest().lower()
if actual_sha != expected_sha:
    raise SystemExit(
        f"error: SHA-512 mismatch for downloaded Antigravity CLI archive "
        f"(expected {expected_sha[:16]}..., got {actual_sha[:16]}...)"
    )

with tarfile.open(archive_path, mode="r:gz") as tf:
    for member in tf.getmembers():
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"error: refusing non-regular tar member {member.name!r}")
        member_target = (extract_dir / member.name).resolve()
        if member_target != extract_dir and extract_dir not in member_target.parents:
            raise SystemExit(f"error: refusing path-traversal tar member {member.name!r}")
        if member.isdir():
            member_target.mkdir(parents=True, exist_ok=True)
        else:
            member_target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                raise SystemExit(f"error: unreadable tar member {member.name!r}")
            with src, member_target.open("wb") as out:
                while True:
                    buf = src.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)

candidates: list[Path] = []
for name in ("agy", "antigravity", "agy.exe", "antigravity.exe"):
    for p in sorted(extract_dir.rglob(name)):
        if p.is_file() and not p.is_symlink():
            candidates.append(p)

if not candidates:
    raise SystemExit("error: archive did not contain an 'agy' or 'antigravity' binary")

chosen = candidates[0]
os.chmod(chosen, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
print(str(chosen))
PY
)"

if [[ "$(uname -s)" == "Darwin" ]] && command -v xattr >/dev/null 2>&1; then
  xattr -d com.apple.quarantine "$EXTRACTED_BIN" 2>/dev/null || true
fi

mkdir -p "$TARGET_DIR"
if [[ "$EXTRACTED_BIN" == *.exe ]]; then
  DEST_BIN="${TARGET_DIR%/}/agy.exe"
  STAGE_PATH="${TARGET_DIR%/}/.agy.tmp.$$.exe"
else
  DEST_BIN="${TARGET_DIR%/}/agy"
  STAGE_PATH="${TARGET_DIR%/}/.agy.tmp.$$"
fi
cp -f "$EXTRACTED_BIN" "$STAGE_PATH"
chmod 0755 "$STAGE_PATH"

if [[ "${CC_AGY_SKIP_SMOKE_CHECK:-0}" != "1" ]]; then
  HELP_OUT="$("$STAGE_PATH" --help 2>&1)" || {
    rm -f "$STAGE_PATH"
    STAGE_PATH=""
    echo "error: staged binary failed '--help' verification (existing binary at $DEST_BIN preserved)" >&2
    exit 1
  }
  if [[ "$HELP_OUT" == *"--is_google_internal"* || "$HELP_OUT" == *"--use_stubby_auth"* ]]; then
    rm -f "$STAGE_PATH"
    STAGE_PATH=""
    echo "error: staged binary is a Google-internal jetski build rather than external agy" >&2
    exit 1
  fi
  if [[ "$HELP_OUT" != *"--output-format"* ]]; then
    rm -f "$STAGE_PATH"
    STAGE_PATH=""
    echo "error: staged binary does not advertise '--output-format' (existing binary at $DEST_BIN preserved)" >&2
    exit 1
  fi
fi

mv -f "$STAGE_PATH" "$DEST_BIN"
STAGE_PATH=""

if [[ "$CONFIGURE_SHELL" -eq 1 ]]; then
  "$DEST_BIN" install --dir "$TARGET_DIR" >/dev/null 2>&1 || true
fi

if [[ "$JSON_OUTPUT" -eq 1 ]]; then
  "$PYTHON_BIN" - "$DEST_BIN" "$VERSION" "$PLATFORM" <<'PY'
import json
import sys

print(
    json.dumps(
        {
            "ok": True,
            "agy_bin": sys.argv[1],
            "version": sys.argv[2],
            "platform": sys.argv[3],
        }
    )
)
PY
else
  printf 'Installed Antigravity CLI %s (%s) to %s\n' "$VERSION" "$PLATFORM" "$DEST_BIN"
fi
