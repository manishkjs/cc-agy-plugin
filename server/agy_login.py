"""Interactive and two-step headless Gemini Enterprise OAuth login for `agy`.

Supports three seamless sign-in paths without requiring users to navigate full-screen TUI menus:
1. Local desktop browser loopback (`agy acp` `authenticate` with `methodId="oauth-business"`):
   automatically opens the browser on local macOS/Windows/Linux desktops and saves credentials
   with zero code copy-pasting.
2. Interactive terminal PKCE (`AGY_CLI_CDE_AUTH_ACTION=login agy` attached to `/dev/tty`):
   prints a clickable Google OAuth URL directly in the terminal and reads the pasted code or
   callback URL.
3. Two-step agent/chat PKCE (`--start` / `--complete` for Claude Code, Cursor, OpenCode, or SSH
   sessions without a controlling TTY): spawns a background `agy` PKCE session holding the code
   verifier in memory over a `0600` UNIX socket, immediately returns the clickable OAuth URL so
   the host agent can show it to the user, and completes token exchange when `--complete <code>`
   is invoked.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.parse

_PLUGIN_ROOT = str(Path(__file__).resolve().parent.parent)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from server.agy_cli import agy_cli_config_dir, inspect_agy_cli_offline, resolve_agy_cli_binary

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_OAUTH_URL_RE = re.compile(r"(https://accounts\.google\.com/\S+)")
_EMAIL_RE = re.compile(r"Successfully authenticated as\s+(\S+?)\.?\s*$", re.MULTILINE)
_DEFAULT_HOLD_TIMEOUT_SECONDS = 600
_URL_CAPTURE_TIMEOUT_SECONDS = 15.0


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from `text`."""
    return _ANSI_ESCAPE_RE.sub("", text)


def extract_oauth_code(raw_value: str) -> str:
    """Extract an OAuth authorization code from a pasted code string (or URL if `?code=` is still present)."""
    cleaned = (raw_value or "").strip()
    if not cleaned:
        raise ValueError("Authorization code cannot be empty.")
    if cleaned.startswith(("http://", "https://")) or ("code=" in cleaned and ("?" in cleaned or "&" in cleaned)):
        query_str = urllib.parse.urlsplit(cleaned).query if "://" in cleaned else cleaned.lstrip("?")
        params = urllib.parse.parse_qs(query_str)
        codes = params.get("code")
        if codes and codes[0].strip():
            return codes[0].strip()
        raise ValueError(
            "The browser address bar strips `?code=` after loading. Click the 'Copy to Clipboard' "
            "button inside the Antigravity callback page and paste that 4/... code instead."
        )
    return cleaned


def _home_dir(env: Mapping[str, str]) -> Path:
    raw_home = (env.get("HOME") or "").strip()
    return Path(raw_home).expanduser() if raw_home else Path.home()


def _login_state_dir(env: Mapping[str, str]) -> Path:
    """Return a short `0700` per-user directory for the UNIX domain socket (<= 100 bytes for `AF_UNIX`)."""
    home_key = hashlib.sha256(str(_home_dir(env)).encode("utf-8")).hexdigest()[:12]
    runtime_base = Path(tempfile.gettempdir()) / f"cc-agy-login-{os.getuid()}-{home_key}"
    runtime_base.mkdir(parents=True, exist_ok=True)
    try:
        runtime_base.chmod(0o700)
    except OSError:
        pass
    return runtime_base


def _atomic_write_private(target: Path, payload_bytes: bytes) -> None:
    """Atomically write `payload_bytes` to `target` with `0600` permissions."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload_bytes)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def finalize_enterprise_onboarding(
    config_dir: Path,
    home_dir: Path,
    *,
    had_standalone_token_before: bool = False,
) -> dict[str, Any]:
    """Sync standalone CDE token into `agy`'s CLI token file and record enterprise `onboarding.json`.

    When `AGY_CLI_CDE_AUTH_ACTION=login` runs on a headless host without an OS keyring, it writes
    `~/.gemini/jetski-standalone-oauth-token` whereas `agy` CLI reads `<config_dir>/antigravity-oauth-token`.
    Both files use the identical `codeassistclient.StoredToken` JSON format.
    """
    standalone_token_file = home_dir / ".gemini" / "jetski-standalone-oauth-token"
    cli_token_file = config_dir / "antigravity-oauth-token"
    prev_auth_method = "gcp"

    if standalone_token_file.is_file() and standalone_token_file.stat().st_size > 10:
        raw_bytes = standalone_token_file.read_bytes()
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(parsed, dict) and parsed.get("wif_provider"):
                prev_auth_method = "wif"
        except (UnicodeDecodeError, ValueError):
            pass
        _atomic_write_private(cli_token_file, raw_bytes)
        if not had_standalone_token_before:
            standalone_token_file.unlink(missing_ok=True)

    onboarding_file = config_dir / "cache" / "onboarding.json"
    existing_onboarding: dict[str, Any] = {}
    if onboarding_file.is_file():
        try:
            loaded = json.loads(onboarding_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_onboarding = loaded
        except (OSError, ValueError):
            existing_onboarding = {}

    updated_onboarding = {
        **existing_onboarding,
        "consumerOnboardingComplete": False,
        "enterpriseOnboardingComplete": True,
        "onboardingComplete": bool(existing_onboarding.get("onboardingComplete", False)),
        "previousAuthMethod": prev_auth_method,
    }
    _atomic_write_private(
        onboarding_file,
        (json.dumps(updated_onboarding, indent=2) + "\n").encode("utf-8"),
    )
    return {"config_dir": str(config_dir), "previous_auth_method": prev_auth_method}


def can_use_local_browser_loopback(env: Mapping[str, str]) -> bool:
    """Return True when this machine has a local GUI browser and is not an SSH session."""
    if any((env.get(k) or "").strip() for k in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return False
    if (env.get("CC_AGY_FORCE_URL_LOGIN") or "").strip().lower() in ("1", "true", "yes"):
        return False
    if sys.platform == "darwin":
        return shutil.which("open") is not None
    if sys.platform == "win32":
        return True
    if sys.platform.startswith("linux"):
        has_display = bool((env.get("DISPLAY") or "").strip() or (env.get("WAYLAND_DISPLAY") or "").strip())
        return has_display and shutil.which("xdg-open") is not None
    return False


def try_acp_business_browser_login(
    agy_bin: str,
    *,
    env: Mapping[str, str],
    timeout_seconds: float = 110.0,
) -> tuple[bool, str]:
    """Attempt zero-copy-paste browser loopback login via `agy acp` `authenticate(oauth-business)`."""
    proc = subprocess.Popen(
        [agy_bin, "acp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env),
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientInfo": {"name": "cc-agy-plugin", "version": "0.3.0"},
            },
        }
        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()
        init_line = proc.stdout.readline()
        if not init_line:
            return False, "agy acp closed before initialize completed"

        auth_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "authenticate",
            "params": {"methodId": "oauth-business"},
        }
        proc.stdin.write(json.dumps(auth_req) + "\n")
        proc.stdin.flush()

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                return False, "agy acp exited during browser sign-in"
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("id") != 2:
                continue
            if "error" in msg and msg["error"]:
                err_obj = msg["error"]
                err_msg = err_obj.get("message") if isinstance(err_obj, Mapping) else str(err_obj)
                return False, str(err_msg or "agy acp authenticate failed")
            finalize_enterprise_onboarding(agy_cli_config_dir(env), _home_dir(env), had_standalone_token_before=True)
            return True, ""
        return False, "Timed out waiting for browser sign-in"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _spawn_cde_login_and_capture_url(
    agy_bin: str,
    *,
    env: Mapping[str, str],
    timeout_seconds: float = _URL_CAPTURE_TIMEOUT_SECONDS,
) -> tuple[subprocess.Popen[str], str]:
    """Start `AGY_CLI_CDE_AUTH_ACTION=login agy` and read stdout until the Google OAuth URL appears."""
    child_env = {**env, "AGY_CLI_CDE_AUTH_ACTION": "login"}
    child_env.pop("AGY_CDE_ENTERPRISE_AUTH", None)
    proc = subprocess.Popen(
        [agy_bin],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )
    assert proc.stdout is not None
    collected: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
            continue
        clean = strip_ansi(line)
        collected.append(clean)
        match = _OAUTH_URL_RE.search(clean)
        if match:
            return proc, match.group(1).strip()
    try:
        proc.kill()
    except OSError:
        pass
    joined = "".join(collected).strip()
    raise RuntimeError(f"Could not obtain OAuth sign-in URL from `{agy_bin}`: {joined or 'no output'}")


def _exchange_code_with_running_cde_proc(
    proc: subprocess.Popen[str],
    raw_code_or_url: str,
    *,
    config_dir: Path,
    home_dir: Path,
    had_standalone_before: bool,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Feed the extracted authorization code to the waiting `agy` login process and finalize credentials."""
    code = extract_oauth_code(raw_code_or_url)
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(code + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    remaining_out, _ = proc.communicate(timeout=timeout_seconds)
    clean_out = strip_ansi(remaining_out or "")
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": clean_out.strip() or f"agy login exited with status {proc.returncode}",
        }
    finalize_enterprise_onboarding(
        config_dir,
        home_dir,
        had_standalone_token_before=had_standalone_before,
    )
    email_match = _EMAIL_RE.search(clean_out)
    email = email_match.group(1) if email_match else None
    return {
        "ok": True,
        "email": email,
        "tier_id": "gcp-enterprise",
        "message": clean_out.strip(),
    }


def _cleanup_stale_hold_session(state_dir: Path) -> None:
    session_file = state_dir / "session.json"
    sock_file = state_dir / "submit.sock"
    if session_file.is_file():
        try:
            meta = json.loads(session_file.read_text(encoding="utf-8"))
            old_pid = int(meta.get("pid") or 0)
            if old_pid > 1 and old_pid != os.getpid():
                os.kill(old_pid, signal.SIGTERM)
                time.sleep(0.1)
        except (OSError, ValueError):
            pass
    session_file.unlink(missing_ok=True)
    sock_file.unlink(missing_ok=True)


def run_hold_daemon(state_dir: Path, *, timeout_seconds: int = _DEFAULT_HOLD_TIMEOUT_SECONDS) -> int:
    """Background PKCE holder: starts `agy` login, writes `session.json`, and waits for `--complete`."""
    env = dict(os.environ)
    agy_bin, _ = resolve_agy_cli_binary(env)
    if not agy_bin:
        _atomic_write_private(
            state_dir / "session.json",
            json.dumps({"ok": False, "error": "External `agy` binary is not installed."}).encode("utf-8"),
        )
        return 1

    home_dir = _home_dir(env)
    config_dir = agy_cli_config_dir(env)
    had_standalone = (home_dir / ".gemini" / "jetski-standalone-oauth-token").is_file()
    sock_path = state_dir / "submit.sock"
    sock_path.unlink(missing_ok=True)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    try:
        sock_path.chmod(0o600)
    except OSError:
        pass
    srv.listen(2)
    srv.settimeout(float(timeout_seconds))

    try:
        proc, auth_url = _spawn_cde_login_and_capture_url(agy_bin, env=env)
    except Exception as exc:
        srv.close()
        sock_path.unlink(missing_ok=True)
        _atomic_write_private(
            state_dir / "session.json",
            json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"),
        )
        return 1

    _atomic_write_private(
        state_dir / "session.json",
        json.dumps(
            {
                "ok": True,
                "pid": os.getpid(),
                "auth_url": auth_url,
                "socket": str(sock_path),
                "created_at": time.time(),
                "expires_at": time.time() + timeout_seconds,
            },
            indent=2,
        ).encode("utf-8"),
    )

    try:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            remaining = max(1.0, deadline - time.monotonic())
            srv.settimeout(remaining)
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                break
            with conn:
                raw_req = conn.recv(8192).decode("utf-8", errors="replace").strip()
                try:
                    extract_oauth_code(raw_req)
                except ValueError as exc:
                    conn.sendall((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode("utf-8"))
                    continue
                result = _exchange_code_with_running_cde_proc(
                    proc,
                    raw_req,
                    config_dir=config_dir,
                    home_dir=home_dir,
                    had_standalone_before=had_standalone,
                )
                conn.sendall((json.dumps(result) + "\n").encode("utf-8"))
                return 0 if result.get("ok") else 1
        return 1
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except OSError:
            pass
        srv.close()
        sock_path.unlink(missing_ok=True)
        (state_dir / "session.json").unlink(missing_ok=True)


def start_two_step_login(
    env: Mapping[str, str] | None = None,
    *,
    timeout_seconds: int = _DEFAULT_HOLD_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Launch the detached PKCE holder daemon and return the clickable Google OAuth URL."""
    source = dict(os.environ if env is None else env)
    status = inspect_agy_cli_offline(source)
    if status.get("auth_configured") and not status.get("consumer_account_detected"):
        return {
            "ok": True,
            "already_signed_in": True,
            "tier_id": status.get("tier_id"),
            "credential_source": status.get("credential_source"),
        }
    if not status.get("agy_bin"):
        return {"ok": False, "error": status.get("error") or "External `agy` binary is not installed."}

    state_dir = _login_state_dir(source)
    _cleanup_stale_hold_session(state_dir)
    session_file = state_dir / "session.json"

    with open(os.devnull, "rb") as devnull_in, open(os.devnull, "wb") as devnull_out:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "server.agy_login",
                "--hold-daemon",
                str(state_dir),
                "--timeout",
                str(timeout_seconds),
            ],
            cwd=str(Path(__file__).resolve().parent.parent),
            env=source,
            stdin=devnull_in,
            stdout=devnull_out,
            stderr=devnull_out,
            start_new_session=True,
            close_fds=True,
        )

    deadline = time.monotonic() + _URL_CAPTURE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if session_file.is_file():
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and ("auth_url" in data or "error" in data):
                    return data
            except (OSError, ValueError):
                pass
        time.sleep(0.05)
    return {"ok": False, "error": "Timed out waiting for background `agy` OAuth URL generation."}


def complete_two_step_login(
    raw_code_or_url: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Send the user's pasted authorization code (or callback URL) to the waiting PKCE holder."""
    source = dict(os.environ if env is None else env)
    try:
        clean_code = extract_oauth_code(raw_code_or_url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    state_dir = _login_state_dir(source)
    sock_path = state_dir / "submit.sock"
    if not sock_path.exists():
        return {
            "ok": False,
            "error": (
                "No active Gemini Enterprise sign-in session is waiting (it may have expired or already completed). "
                "Run `python3 -m server.agy_login --start` to generate a fresh sign-in link."
            ),
        }

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(35.0)
        client.connect(str(sock_path))
        client.sendall((clean_code + "\n").encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
    raw_reply = b"".join(chunks).decode("utf-8", errors="replace").strip()
    try:
        parsed = json.loads(raw_reply)
        if isinstance(parsed, dict):
            return parsed
    except ValueError:
        pass
    return {"ok": False, "error": raw_reply or "Empty response from login helper."}


def _open_controlling_tty() -> tuple[Any, Any] | None:
    if sys.platform == "win32":
        return None
    try:
        tty_in = open("/dev/tty", "r", encoding="utf-8")
        tty_out = open("/dev/tty", "w", encoding="utf-8", buffering=1)
        return tty_in, tty_out
    except OSError:
        return None


def run_interactive_or_url_login(
    env: Mapping[str, str] | None = None,
    *,
    browser_timeout_seconds: float = 110.0,
) -> int:
    """Automatic onboarding entrypoint used by `quickstart.sh` and `/cc-agy-plugin:login`."""
    source = dict(os.environ if env is None else env)
    if (source.get("CC_AGY_SKIP_SIGNIN") or "").strip() == "1" or source.get("PYTEST_CURRENT_TEST"):
        return 0

    status = inspect_agy_cli_offline(source)
    if status.get("auth_configured") and not status.get("consumer_account_detected"):
        print(f"Gemini Enterprise sign-in verified ({status.get('tier_id') or 'enterprise'}).")
        return 0

    agy_bin = status.get("agy_bin")
    if not agy_bin:
        print("Antigravity CLI (`agy`) is not installed yet; skipping sign-in.", file=sys.stderr)
        return 0

    # 1. Local desktop with GUI browser (not SSH): zero-copy-paste browser loopback!
    if can_use_local_browser_loopback(source):
        print("Opening your browser for one-click Gemini Enterprise sign-in...", flush=True)
        ok, _reason = try_acp_business_browser_login(
            agy_bin,
            env=source,
            timeout_seconds=browser_timeout_seconds,
        )
        if ok:
            print("Signed in to Gemini Enterprise (tier=gcp-enterprise).")
            return 0

    # 2. Interactive terminal (`/dev/tty` available, e.g. `curl ... | bash` in an SSH shell):
    tty_pair = _open_controlling_tty()
    if tty_pair is not None:
        tty_in, tty_out = tty_pair
        try:
            home_dir = _home_dir(source)
            config_dir = agy_cli_config_dir(source)
            had_standalone = (home_dir / ".gemini" / "jetski-standalone-oauth-token").is_file()
            proc, auth_url = _spawn_cde_login_and_capture_url(agy_bin, env=source)
            tty_out.write(
                "\n"
                "  Gemini Enterprise Sign-In\n"
                "  1. Open this URL in your browser and sign in with your work account:\n\n"
                f"     {auth_url}\n\n"
                "  2. Click 'Copy to Clipboard' on the Antigravity page and paste the code below:\n"
                "  Authorization code: "
            )
            tty_out.flush()
            pasted = tty_in.readline()
            if not pasted:
                proc.kill()
                tty_out.write("\nSign-in skipped (no code entered).\n")
                return 0
            result = _exchange_code_with_running_cde_proc(
                proc,
                pasted,
                config_dir=config_dir,
                home_dir=home_dir,
                had_standalone_before=had_standalone,
            )
            if result.get("ok"):
                who = f" as {result['email']}" if result.get("email") else ""
                tty_out.write(f"\nSigned in to Gemini Enterprise{who} (tier=gcp-enterprise)!\n\n")
                return 0
            tty_out.write(f"\nSign-in failed: {result.get('error')}\n")
            return 1
        finally:
            tty_in.close()
            tty_out.close()

    # 3. Non-interactive host (e.g. Claude Code / Cursor over SSH with no `/dev/tty`):
    #    Start the 2-step PKCE session immediately and print the clickable OAuth URL!
    started = start_two_step_login(source)
    if not started.get("ok"):
        print(f"Could not start Gemini Enterprise sign-in: {started.get('error')}", file=sys.stderr)
        return 1
    if started.get("already_signed_in"):
        print(f"Gemini Enterprise sign-in verified ({started.get('tier_id') or 'enterprise'}).")
        return 0

    plugin_root = Path(__file__).resolve().parent.parent
    complete_cmd = f'{sys.executable} "{plugin_root}/server/agy_login.py" --complete "<PASTE_CODE>"'
    auth_url = started["auth_url"]
    print(
        "\n"
        "========================================================================\n"
        "GEMINI ENTERPRISE SIGN-IN URL (CLICK TO AUTHORIZE):\n\n"
        f"{auth_url}\n\n"
        "HOW TO FINISH SIGN-IN (1 STEP):\n"
        "  1. Click the URL above, sign in with your Gemini Enterprise work account,\n"
        "     and click 'Copy to Clipboard' on the Antigravity callback page.\n"
        "  2. Paste that code right into chat!\n\n"
        "AGENT / CLAUDE CODE INSTRUCTION:\n"
        "  - Display the clickable URL above verbatim to the user right now and ask\n"
        "    them to reply with the copied authorization code (via AskUserQuestion or chat).\n"
        "  - As soon as the user pastes the code, run:\n"
        f"      {complete_cmd}\n"
        "========================================================================\n",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `python3 -m server.agy_login` and `python3 server/agy_login.py`."""
    parser = argparse.ArgumentParser(description="Interactive Gemini Enterprise OAuth helper for cc-agy-plugin.")
    parser.add_argument("--start", action="store_true", help="Start a 2-step PKCE login and print JSON/text with URL.")
    parser.add_argument("--complete", metavar="CODE_OR_URL", help="Complete a waiting 2-step PKCE login with the code.")
    parser.add_argument("--json", action="store_true", help="Format `--start` or `--complete` output as JSON.")
    parser.add_argument("--hold-daemon", metavar="STATE_DIR", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=_DEFAULT_HOLD_TIMEOUT_SECONDS, help="Session timeout in seconds.")
    args = parser.parse_args(argv)

    if args.hold_daemon:
        return run_hold_daemon(Path(args.hold_daemon), timeout_seconds=args.timeout)

    if args.complete is not None:
        res = complete_two_step_login(args.complete)
        if args.json:
            print(json.dumps(res, indent=2))
        elif res.get("ok"):
            who = f" as {res['email']}" if res.get("email") else ""
            print(f"[OK] Signed in to Gemini Enterprise{who} (tier={res.get('tier_id', 'gcp-enterprise')}).")
        else:
            print(f"[FAIL] {res.get('error')}", file=sys.stderr)
        return 0 if res.get("ok") else 1

    if args.start:
        res = start_two_step_login(timeout_seconds=args.timeout)
        if args.json:
            print(json.dumps(res, indent=2))
        elif res.get("already_signed_in"):
            print(f"[OK] Already signed in to Gemini Enterprise (tier={res.get('tier_id')}).")
        elif res.get("ok"):
            print(res["auth_url"])
        else:
            print(f"[FAIL] {res.get('error')}", file=sys.stderr)
        return 0 if res.get("ok") else 1

    return run_interactive_or_url_login()


if __name__ == "__main__":
    # Support direct `python3 /path/to/repo/server/agy_login.py` invocation as well as `-m server.agy_login`
    sys.exit(main())
