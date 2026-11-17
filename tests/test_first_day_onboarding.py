"""First-day onboarding regressions found by the 3-model gauntlet review (round 1).

Each test reproduces a state a real first-day user can reach, and checks that the plugin
either works or says exactly what to do, never "signed in" when agy would reject the call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest

from server import agy_cli
from server import agy_login
from server import install

REPO_ROOT = Path(__file__).resolve().parent.parent
_LICENSE_A = {
    "user_tier": "gcp-ge-plus-tier",
    "tier_display_name": "Gemini Enterprise Plus",
    "project_id": "proj-alpha",
    "region": "global",
}
_LICENSE_B = {**_LICENSE_A, "project_id": "proj-beta", "user_tier": "gcp-ge-standard-tier"}


def _env(home: Path, cfg_dir: Path, **extra: str) -> dict[str, str]:
    return {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
        "CC_AGY_ALLOW_TEST_LICENSE_FETCH": "1",
        "CC_AGY_TOKEN_IN_KEYCHAIN": "0",
        **extra,
    }


def _write_token(cfg_dir: Path, token: dict[str, Any] | None = None, **fields: Any) -> Path:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    token_file = cfg_dir / "antigravity-oauth-token"
    payload = {
        "token": token if token is not None else {"access_token": "acc", "refresh_token": "ref"},
        "auth_method": "gcp",
        **fields,
    }
    token_file.write_text(json.dumps(payload), encoding="utf-8")
    return token_file


def _fake_licenses(monkeypatch: pytest.MonkeyPatch, licenses: list[dict[str, str]]) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def _fetch(access_token: str, *, timeout_seconds: float = 8.0) -> list[dict[str, str]]:
        del timeout_seconds
        calls.append(("fetch", access_token))
        return [dict(lic) for lic in licenses]

    def _self_assign(access_token: str, project_id: str, location: str = "global", **_: Any) -> None:
        calls.append(("self_assign", project_id, location))
        return None

    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", _fetch)
    monkeypatch.setattr(agy_login, "self_assign_enterprise_license", _self_assign)
    return calls


# --- Honest sign-in status -------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        b"",  # 0-byte file
        b'{"token": {"access_token": "acc-half_writ',  # half-written JSON
        json.dumps({"token": {}, "auth_method": "gcp", "user_tier": "t", "project_id": "p"}).encode(),
        json.dumps(
            {
                "token": {"access_token": "old", "refresh_token": "", "expiry": "2020-01-01T00:00:00Z"},
                "auth_method": "gcp",
                "user_tier": "t",
                "project_id": "p",
            }
        ).encode(),
    ],
    ids=["empty", "half_written", "no_tokens", "expired_no_refresh"],
)
def test_unusable_token_files_are_not_signed_in(tmp_path: Path, raw: bytes):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "antigravity-oauth-token").write_bytes(raw)
    (cfg_dir / "cache").mkdir()
    (cfg_dir / "cache" / "onboarding.json").write_text(json.dumps({"enterpriseOnboardingComplete": True}))

    auth_ok, source = agy_cli._inspect_auth_files(cfg_dir, _env(tmp_path, cfg_dir))
    started = agy_login.start_two_step_login(_env(tmp_path, cfg_dir, CC_AGY_CLI_BIN="/nonexistent/agy"))

    assert agy_cli.stored_token_problem(cfg_dir / "antigravity-oauth-token") is not None
    assert (auth_ok, source) == (False, None)
    assert not started.get("already_signed_in")


def test_expired_token_with_refresh_token_is_still_usable(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    _write_token(
        cfg_dir,
        token={"access_token": "old", "refresh_token": "ref", "expiry": "2020-01-01T00:00:00Z"},
        user_tier="t",
        project_id="p",
    )

    assert agy_cli.stored_token_problem(cfg_dir / "antigravity-oauth-token") is None


def test_onboarding_cache_alone_counts_only_where_agy_uses_a_keychain(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    (cfg_dir / "cache").mkdir(parents=True)
    (cfg_dir / "cache" / "onboarding.json").write_text(json.dumps({"enterpriseOnboardingComplete": True}))

    linux_ok, _ = agy_cli._inspect_auth_files(cfg_dir, _env(tmp_path, cfg_dir, CC_AGY_TOKEN_IN_KEYCHAIN="0"))
    mac_ok, mac_source = agy_cli._inspect_auth_files(cfg_dir, _env(tmp_path, cfg_dir, CC_AGY_TOKEN_IN_KEYCHAIN="1"))

    assert linux_ok is False
    assert mac_ok is True and mac_source.startswith("onboarding_cache:")


# --- License choice ---------------------------------------------------------------------------


def test_explicit_project_switches_a_saved_license(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    token_file = _write_token(cfg_dir, **_LICENSE_A)
    _fake_licenses(monkeypatch, [_LICENSE_A, _LICENSE_B])

    result = agy_login.heal_enterprise_token(_env(tmp_path, cfg_dir, CC_AGY_ENTERPRISE_PROJECT="proj-beta"))

    saved = json.loads(token_file.read_text())
    assert result["healed"] is True
    assert (saved["project_id"], saved["user_tier"]) == ("proj-beta", "gcp-ge-standard-tier")


def test_stale_settings_project_falls_back_to_the_real_seat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """settings.json gcp.project is written by the plugin itself; it must not block a moved seat."""
    cfg_dir = tmp_path / "cfg"
    _write_token(cfg_dir)
    (cfg_dir / "settings.json").write_text(json.dumps({"gcp": {"project": "proj-old"}}))
    calls = _fake_licenses(monkeypatch, [_LICENSE_B])

    result = agy_login.heal_enterprise_token(_env(tmp_path, cfg_dir))

    assert result["healed"] is True
    assert result["project_id"] == "proj-beta"
    assert not [c for c in calls if c[0] == "self_assign"]


def test_several_seats_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    _write_token(cfg_dir)
    _fake_licenses(monkeypatch, [_LICENSE_A, _LICENSE_B])

    result = agy_login.heal_enterprise_token(_env(tmp_path, cfg_dir))

    assert result["project_id"] == "proj-alpha"
    assert result["alternatives"] == ["proj-beta"]


def test_lookup_failure_is_not_called_no_license(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    _write_token(cfg_dir)

    def _fail(*_a: Any, **_k: Any) -> list[dict[str, str]]:
        raise agy_login.LicenseLookupError("couldn't reach the license service (connection refused)")

    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", _fail)

    result = agy_login.heal_enterprise_token(_env(tmp_path, cfg_dir))

    assert result["healed"] is False
    assert "couldn't reach the license service" in result["reason"]


def test_fetch_reports_http_and_network_errors(monkeypatch: pytest.MonkeyPatch):
    import urllib.error

    def _http_401(*_a: Any, **_k: Any) -> None:
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(agy_login.urllib.request, "urlopen", _http_401)
    with pytest.raises(agy_login.LicenseLookupError, match="HTTP 401"):
        agy_login.fetch_enterprise_licenses("tok")

    def _down(*_a: Any, **_k: Any) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(agy_login.urllib.request, "urlopen", _down)
    with pytest.raises(agy_login.LicenseLookupError, match="couldn't reach"):
        agy_login.fetch_enterprise_licenses("tok")


# --- No seat / keychain: never claim success ---------------------------------------------------


def test_no_seat_sign_in_records_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    _write_token(cfg_dir)
    _fake_licenses(monkeypatch, [])

    onboarding = agy_login.finalize_enterprise_onboarding(
        cfg_dir, tmp_path, had_standalone_token_before=True, env=_env(tmp_path, cfg_dir)
    )

    assert onboarding["verified"] is False
    assert onboarding["problem"] == "no_license"
    assert not (cfg_dir / "cache" / "onboarding.json").exists()
    assert "no Gemini Enterprise license" in agy_login.describe_onboarding_problem(onboarding)


def test_keychain_token_is_reported_not_faked(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()

    onboarding = agy_login.finalize_enterprise_onboarding(
        cfg_dir, tmp_path, had_standalone_token_before=True, env=_env(tmp_path, cfg_dir)
    )

    assert onboarding["problem"] == "keychain"
    assert not (cfg_dir / "cache" / "onboarding.json").exists()
    assert "--force" in agy_login.describe_onboarding_problem(onboarding)


# --- Sign-in session safety ---------------------------------------------------------------------


def test_state_dir_falls_back_when_tmp_name_was_planted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agy_login.tempfile, "gettempdir", lambda: str(tmp_path))
    home = tmp_path / "home"
    home.mkdir()
    env = {"HOME": str(home)}
    planted = agy_login._login_state_dir(env)
    planted.rmdir()
    elsewhere = tmp_path / "attacker"
    elsewhere.mkdir()
    planted.symlink_to(elsewhere)

    chosen = agy_login._login_state_dir(env)

    assert chosen.parent == home / ".cache"
    assert chosen.stat().st_mode & 0o777 == 0o700


def test_state_dir_permissions_are_tightened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agy_login.tempfile, "gettempdir", lambda: str(tmp_path))
    env = {"HOME": str(tmp_path / "home")}
    target = agy_login._login_state_dir(env)
    target.chmod(0o777)

    agy_login._login_state_dir(env)

    assert target.stat().st_mode & 0o777 == 0o700


def test_cleanup_never_signals_an_unrelated_process(tmp_path: Path):
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "session.json").write_text(json.dumps({"pid": bystander.pid}))
        agy_login._cleanup_stale_hold_session(tmp_path)
        time.sleep(0.2)
        assert bystander.poll() is None
    finally:
        bystander.kill()


def test_second_start_reuses_the_waiting_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    holder = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "--hold-daemon", str(state_dir)]
    )
    try:
        (state_dir / "submit.sock").write_text("")
        (state_dir / "session.json").write_text(
            json.dumps({"ok": True, "pid": holder.pid, "auth_url": "https://accounts.google.com/x", "expires_at": time.time() + 300})
        )
        monkeypatch.setattr(agy_login, "_login_state_dir", lambda _env: state_dir)
        monkeypatch.setattr(agy_login, "can_use_local_browser_loopback", lambda _env: False)
        monkeypatch.setattr(
            agy_login, "inspect_agy_cli_offline", lambda _env: {"agy_bin": "/x/agy", "auth_configured": False}
        )
        monkeypatch.setattr(agy_login, "heal_enterprise_token", lambda _env: {})

        result = agy_login.start_two_step_login({"HOME": str(tmp_path)})

        assert result.get("reused") is True, result
        assert result["auth_url"] == "https://accounts.google.com/x"
        assert holder.poll() is None
    finally:
        holder.kill()


def test_agent_shells_are_never_treated_as_a_person_at_a_terminal():
    assert agy_login.is_interactive_terminal({"CLAUDECODE": "1"}) is False


def test_short_links_are_opt_in():
    assert agy_login.shorten_and_verify_oauth_url("https://accounts.google.com/o/oauth2/auth?x=1", {}) is None


# --- Wizard and quickstart contracts --------------------------------------------------------------


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("1", ["--auth", "api-key"]),  # Ambient GOOGLE_CLOUD_PROJECT must not turn this into Vertex.
        ("2", ["--project", "corp-prod-billing"]),  # --project selects Vertex on its own.
        ("3", ["--auth", "gateway"]),  # No URL yet: pinned, so setup asks for one.
        ("4", ["--auth", "gemini-enterprise"]),
    ],
)
def test_wizard_pins_the_chosen_auth_mode(choice: str, expected: list[str]):
    answers = iter(["3", choice, "", "1", ""])
    built = install.run_interactive_wizard(
        input_fn=lambda _prompt="": next(answers, ""),
        output_fn=lambda *_a: None,
        env={"GOOGLE_CLOUD_PROJECT": "corp-prod-billing"},
    )

    flag = expected[0]
    assert flag in built, built
    assert built[built.index(flag) + 1] == expected[1]


def _bash_function(name: str) -> str:
    script = (REPO_ROOT / "scripts" / "quickstart.sh").read_text(encoding="utf-8")
    start = script.index(f"{name}() {{")
    return script[start : script.index("\n}\n", start) + 3]


def test_quickstart_reads_auth_for_its_final_doctor():
    for args, want in ((["--host", "claude", "--auth", "gemini-enterprise"], "gemini-enterprise"), (["--auth=vertex"], "vertex"), ([], "")):
        out = subprocess.run(
            ["bash", "-c", _bash_function("flag_value") + _bash_function("auth_arg") + '\nauth_arg "$@"', "_", *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert out == want


def test_quickstart_treats_piped_stdout_as_an_agent():
    proc = subprocess.run(
        ["bash", "-c", _bash_function("has_controlling_tty") + _bash_function("is_person_at_terminal") + "\nis_person_at_terminal"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1


def test_py_wrapper_picks_a_new_enough_python(tmp_path: Path):
    old = tmp_path / "python3"
    old.write_text("#!/bin/sh\nexit 1\n")  # Stands in for macOS /usr/bin/python3 3.9.
    old.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.path.dirname(sys.executable)}:/usr/bin:/bin", "HOME": str(tmp_path), "CC_AGY_PYTHON_BIN": sys.executable}

    which = subprocess.run(["sh", str(REPO_ROOT / "scripts" / "py.sh"), "--which"], capture_output=True, text=True, env=env, check=True)

    assert Path(which.stdout.strip()).resolve() == Path(sys.executable).resolve()


def test_mcp_server_answers_the_handshake_before_any_venv_setup(tmp_path: Path):
    """Hosts give up on an MCP server that stays silent ("failed to connect"); setup must not block it."""
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "CC_AGY_PYTHON_BIN": sys.executable,
        "PIP_INDEX_URL": "http://127.0.0.1:9/simple",  # Never download anything from a test.
        "PIP_RETRIES": "0",
    }
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
        }
    )
    started = time.monotonic()
    proc = subprocess.run(
        ["sh", str(REPO_ROOT / "scripts" / "run_mcp_server.sh")],
        input=request + "\n",
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    elapsed = time.monotonic() - started

    assert '"result"' in proc.stdout, proc.stderr
    assert elapsed < 20, f"handshake took {elapsed:.1f}s"
    assert (home / ".local" / "share" / "cc-agy-plugin" / "current").resolve() == REPO_ROOT.resolve()


# --- Round 2 regressions ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"SSH_CONNECTION": "1 2 3 4"}, False),  # agy uses a file over SSH, even on a Mac.
        ({"WSL_DISTRO_NAME": "Ubuntu"}, False),
        ({"CC_AGY_TOKEN_IN_KEYCHAIN": "1"}, True),
    ],
)
def test_keyring_rule_mirrors_agy(env: dict[str, str], expected: bool):
    assert agy_cli._platform_keeps_token_in_keychain(env) is expected


def test_linux_keyring_needs_a_desktop_session(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agy_cli.sys, "platform", "linux")
    monkeypatch.setattr(agy_cli.Path, "exists", lambda self: False)  # Not in a container.
    bus = {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1/bus"}
    assert agy_cli._platform_keeps_token_in_keychain({**bus, "DISPLAY": ":0"}) is True
    # systemd gives headless sessions (Cloudtop, tmux) a bus too, with no keyring behind it.
    assert agy_cli._platform_keeps_token_in_keychain(bus) is False
    assert agy_cli._platform_keeps_token_in_keychain({"DBUS_SESSION_BUS_ADDRESS": "disabled:", "DISPLAY": ":0", "XDG_RUNTIME_DIR": "/nonexistent"}) is False


def test_plugin_pointer_moves_when_the_plugin_moves(tmp_path: Path):
    """`mv` onto a symlinked directory moves the new link *into* it; the pointer must really move."""
    from server.runtime_env import write_plugin_root_pointer

    first = tmp_path / "checkout-a"
    second = tmp_path / "checkout-b"
    first.mkdir()
    second.mkdir()
    env = {"HOME": str(tmp_path / "home")}

    write_plugin_root_pointer(first, env)
    pointer = write_plugin_root_pointer(second, env)

    assert pointer is not None and pointer.resolve() == second.resolve()
    assert not list(first.iterdir())  # Nothing was dropped inside the old checkout.


def test_browser_sign_in_gives_up_on_a_silent_agy(tmp_path: Path):
    silent_agy = tmp_path / "agy"
    silent_agy.write_text(
        f"#!{sys.executable}\nimport sys, json, time\n"
        "line = sys.stdin.readline()\n"
        "print(json.dumps({'jsonrpc': '2.0', 'id': 1, 'result': {}}), flush=True)\n"
        "sys.stdin.readline()\n"
        "time.sleep(60)\n"
    )
    silent_agy.chmod(0o755)
    started = time.monotonic()

    ok, reason = agy_login.try_acp_business_browser_login(str(silent_agy), env={"HOME": str(tmp_path)}, timeout_seconds=2)

    assert ok is False and "timed out" in reason
    assert time.monotonic() - started < 15


def _stub_new_link(monkeypatch: pytest.MonkeyPatch, state_dir: Path) -> None:
    """Make start_two_step_login 'start' a holder that immediately publishes a link."""
    state_dir.mkdir(exist_ok=True)

    class _FakeHolder:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            (state_dir / "session.json").write_text(json.dumps({"ok": True, "auth_url": "https://accounts.google.com/new"}))

    monkeypatch.setattr(agy_login, "_login_state_dir", lambda _env: state_dir)
    monkeypatch.setattr(agy_login, "_live_hold_session", lambda _d: None)
    monkeypatch.setattr(agy_login, "_cleanup_stale_hold_session", lambda _d: None)
    monkeypatch.setattr(agy_login.subprocess, "Popen", _FakeHolder)


def _stub_browser_failure(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    def _fail(*_a: Any, **_k: Any) -> tuple[bool, str]:
        calls.append("browser")
        return False, "window closed"

    monkeypatch.setattr(agy_login, "heal_enterprise_token", lambda _env: {})
    monkeypatch.setattr(agy_login, "inspect_agy_cli_offline", lambda _env: {"agy_bin": "/x/agy", "auth_configured": False})
    monkeypatch.setattr(agy_login, "can_use_local_browser_loopback", lambda _env: True)
    monkeypatch.setattr(agy_login, "try_acp_business_browser_login", _fail)


def test_linux_desktop_browser_failure_falls_back_to_a_link_and_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Remote desktops and slow SSO can't finish in the browser; agy writes a file there, so a link works."""
    calls: list[str] = []
    _stub_browser_failure(monkeypatch, calls)
    _stub_new_link(monkeypatch, tmp_path / "state")
    monkeypatch.setattr(agy_login.sys, "platform", "linux")

    result = agy_login.start_two_step_login({"HOME": str(tmp_path), "CC_AGY_TOKEN_IN_KEYCHAIN": "1"})

    assert result["auth_url"] == "https://accounts.google.com/new"
    assert "window closed" in result["note"]


def test_mac_browser_failure_gives_no_link_that_cannot_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    _stub_browser_failure(monkeypatch, calls)
    _stub_new_link(monkeypatch, tmp_path / "state")
    monkeypatch.setattr(agy_login.sys, "platform", "darwin")

    result = agy_login.start_two_step_login({"HOME": str(tmp_path), "CC_AGY_TOKEN_IN_KEYCHAIN": "1"})

    assert result["ok"] is False and "window closed" in result["error"]
    assert "auth_url" not in result


def test_browser_is_tried_once_on_the_terminal_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    _stub_browser_failure(monkeypatch, calls)
    _stub_new_link(monkeypatch, tmp_path / "state")
    monkeypatch.setattr(agy_login.sys, "platform", "linux")
    monkeypatch.setattr(agy_login, "is_interactive_terminal", lambda _env: False)

    rc = agy_login.run_interactive_or_url_login({"HOME": str(tmp_path), "CC_AGY_TOKEN_IN_KEYCHAIN": "0"})

    assert rc == 0
    assert calls == ["browser"]


def test_failed_project_switch_restores_agys_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    original = {"theme": "dark", "gcp": {"project": "team-proj", "location": "us-central1"}}
    (cfg_dir / "settings.json").write_text(json.dumps(original))
    monkeypatch.setattr(agy_login, "try_acp_business_browser_login", lambda *_a, **_k: (False, "no seat in typo-proj"))

    ok, _ = agy_login._browser_sign_in("/x/agy", {"HOME": str(tmp_path), "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir)}, "typo-proj", 5)

    assert ok is False
    assert json.loads((cfg_dir / "settings.json").read_text()) == original


def test_login_child_env_drops_ambient_projects_and_keys():
    child = agy_login._login_child_env(
        {"HOME": "/h", "GOOGLE_CLOUD_PROJECT": "p", "ANTIGRAVITY_PROJECT_ID": "uuid", "GEMINI_API_KEY": "k", "PATH": "/bin"}
    )
    assert child == {"HOME": "/h", "PATH": "/bin"}


def test_failed_project_switch_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    _write_token(cfg_dir, **_LICENSE_A)
    _fake_licenses(monkeypatch, [_LICENSE_A])  # No seat in proj-beta; self-assign fails.

    result = agy_login.start_two_step_login(
        _env(tmp_path, cfg_dir, CC_AGY_ENTERPRISE_PROJECT="proj-beta", CC_AGY_CLI_BIN="/nonexistent/agy")
    )

    assert result["ok"] is False
    assert "proj-beta" in result["error"]


def test_older_standalone_token_never_replaces_a_working_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    cli_token = _write_token(cfg_dir, **_LICENSE_A)
    standalone = tmp_path / ".gemini" / "jetski-standalone-oauth-token"
    standalone.parent.mkdir(parents=True, exist_ok=True)
    standalone.write_text(json.dumps({"token": {"access_token": "old"}, "auth_method": "gcp"}))
    os.utime(standalone, (time.time() - 3600, time.time() - 3600))
    _fake_licenses(monkeypatch, [_LICENSE_A])

    agy_login.finalize_enterprise_onboarding(cfg_dir, tmp_path, had_standalone_token_before=True, env=_env(tmp_path, cfg_dir))

    assert json.loads(cli_token.read_text())["token"]["access_token"] == "acc"


def test_reused_session_requires_the_same_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A session started for one seat project must not swallow a sign-in meant for another."""

    class _NewSession(Exception):
        pass

    def _no_new_session(_state_dir: Path) -> None:
        raise _NewSession

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "--hold-daemon", str(state_dir)])
    try:
        (state_dir / "submit.sock").write_text("")
        (state_dir / "session.json").write_text(
            json.dumps(
                {"pid": holder.pid, "auth_url": "https://accounts.google.com/x", "expires_at": time.time() + 300, "enterprise_project": "proj-alpha"}
            )
        )
        monkeypatch.setattr(agy_login, "_login_state_dir", lambda _env: state_dir)
        monkeypatch.setattr(agy_login, "can_use_local_browser_loopback", lambda _env: False)
        monkeypatch.setattr(agy_login, "inspect_agy_cli_offline", lambda _env: {"agy_bin": "/x/agy", "auth_configured": False})
        monkeypatch.setattr(agy_login, "heal_enterprise_token", lambda _env: {"already_complete": True})
        monkeypatch.setattr(agy_login, "_cleanup_stale_hold_session", _no_new_session)

        same = agy_login.start_two_step_login({"HOME": str(tmp_path), "CC_AGY_ENTERPRISE_PROJECT": "proj-alpha"})
        assert same.get("reused") is True

        with pytest.raises(_NewSession):
            agy_login.start_two_step_login({"HOME": str(tmp_path), "CC_AGY_ENTERPRISE_PROJECT": "proj-beta"})
    finally:
        holder.kill()


def test_doctor_report_ok_fails_when_claude_was_requested_but_missing():
    report = {"preflight": {"ready": True}, "hosts": [{"host": "claude", "status": "claude_cli_not_found"}]}
    assert install._report_ok({**report, "requested_host": "claude"}) is False
    assert install._report_ok({**report, "requested_host": "all"}) is True


def test_doctor_does_not_claim_sdks_the_running_server_cannot_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A venv built in the background after startup isn't loaded yet; doctor must say so."""
    from server import doctor, receipts
    import server.runtime_env as runtime_env

    monkeypatch.setattr(doctor, "_module_available", lambda name: False)
    monkeypatch.setattr(runtime_env, "python_has_module", lambda _py, _mod: True)
    monkeypatch.setattr(runtime_env, "resolve_sdk_python", lambda **_k: str(tmp_path / "venv" / "bin" / "python"))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    env = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", ""), "CC_AGY_AUTH_MODE": "api_key", "GEMINI_API_KEY": "k"}
    store = receipts.ReceiptStore(tmp_path / "st", workspace_root=repo)

    in_server = doctor.run_doctor(workspace_root=repo, env={**env, "CC_AGY_IN_MCP_SERVER": "1"}, store=store)
    one_shot = doctor.run_doctor(workspace_root=repo, env=env, store=store)

    assert in_server["sdks"]["google_genai"] is False
    assert any("started before they were ready" in issue for issue in in_server["issues"]), in_server["issues"]
    # A one-shot CLI doctor isn't the server: the next server start uses the venv, so it's fine.
    assert one_shot["sdks"]["google_genai"] is True
    assert not any("started before they were ready" in issue for issue in one_shot["issues"])


def test_mac_project_switch_goes_through_agys_own_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    (cfg_dir / "cache").mkdir(parents=True)
    (cfg_dir / "cache" / "onboarding.json").write_text(
        json.dumps({"enterpriseOnboardingComplete": True, "cloudaicompanionProject": "proj-alpha"})
    )
    seen: dict[str, Any] = {}

    def _fake_browser(_agy: str, *, env: dict[str, str], timeout_seconds: float) -> tuple[bool, str]:
        seen["settings"] = json.loads((cfg_dir / "settings.json").read_text())
        agy_login._mark_enterprise_onboarding_complete(cfg_dir)
        return True, ""

    monkeypatch.setattr(agy_login, "inspect_agy_cli_offline", lambda _env: {
        "agy_bin": "/x/agy", "auth_configured": True, "companion_project": "proj-alpha"})
    monkeypatch.setattr(agy_login, "can_use_local_browser_loopback", lambda _env: True)
    monkeypatch.setattr(agy_login, "try_acp_business_browser_login", _fake_browser)

    result = agy_login.start_two_step_login(
        _env(tmp_path, cfg_dir, CC_AGY_TOKEN_IN_KEYCHAIN="1", CC_AGY_ENTERPRISE_PROJECT="proj-beta")
    )

    assert result.get("signed_in") is True, result  # Not short-circuited to "already signed in".
    assert seen["settings"]["gcp"]["project"] == "proj-beta"
    assert json.loads((cfg_dir / "cache" / "onboarding.json").read_text())["cloudaicompanionProject"] == "proj-beta"


def test_install_never_records_a_placeholder_auth_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    install._record_install_choice("missing")
    assert not (tmp_path / "cc-agy-plugin" / "last-install.json").exists()
    install._record_install_choice("api_key")
    assert json.loads((tmp_path / "cc-agy-plugin" / "last-install.json").read_text()) == {"auth_mode": "api_key"}


def test_command_resolver_honours_xdg_data_home(tmp_path: Path):
    snippet = (REPO_ROOT / "commands" / "login.md").read_text(encoding="utf-8")
    block = snippet[snippet.index('ROOT="${CLAUDE_PLUGIN_ROOT:-}"'):snippet.index("done", snippet.index('ROOT="${CLAUDE_PLUGIN_ROOT:-}"')) + 4]
    data = tmp_path / "xdg"
    (data / "cc-agy-plugin").mkdir(parents=True)
    (data / "cc-agy-plugin" / "current").symlink_to(REPO_ROOT)

    out = subprocess.run(
        ["sh", "-c", block + '\necho "$ROOT"'],
        env={"HOME": str(tmp_path / "home"), "XDG_DATA_HOME": str(data), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert Path(out).resolve() == REPO_ROOT.resolve()


def test_install_record_keeps_project_and_gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    install._record_install_choice("vertex", project="my-proj", base_url=None)
    assert json.loads((tmp_path / "cc-agy-plugin" / "last-install.json").read_text()) == {
        "auth_mode": "vertex",
        "project": "my-proj",
    }


def test_host_saved_project_is_a_preference_not_an_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A project saved at install time must not undo a later `/login --project` switch."""
    cfg_dir = tmp_path / "cfg"
    token_file = _write_token(cfg_dir, **_LICENSE_B)  # User switched to proj-beta earlier.
    calls = _fake_licenses(monkeypatch, [_LICENSE_A, _LICENSE_B])

    result = agy_login.heal_enterprise_token(_env(tmp_path, cfg_dir, CC_AGY_HOST_ENTERPRISE_PROJECT="proj-alpha"))

    assert result["already_complete"] is True
    assert json.loads(token_file.read_text())["project_id"] == "proj-beta"
    assert not calls


def test_pointer_leaves_no_litter_when_current_is_a_real_directory(tmp_path: Path):
    from server.runtime_env import write_plugin_root_pointer

    home = tmp_path / "home"
    current = home / ".local" / "share" / "cc-agy-plugin" / "current"
    current.mkdir(parents=True)

    assert write_plugin_root_pointer(REPO_ROOT, {"HOME": str(home)}) is None
    assert sorted(p.name for p in current.parent.iterdir()) == ["current"]


def test_same_second_standalone_token_does_not_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    cli_token = _write_token(cfg_dir, **_LICENSE_A)
    standalone = tmp_path / ".gemini" / "jetski-standalone-oauth-token"
    standalone.parent.mkdir(parents=True, exist_ok=True)
    standalone.write_text(json.dumps({"token": {"access_token": "stale"}, "auth_method": "gcp"}))
    stamp = cli_token.stat().st_mtime_ns
    os.utime(standalone, ns=(stamp, stamp))
    _fake_licenses(monkeypatch, [_LICENSE_A])

    agy_login.finalize_enterprise_onboarding(cfg_dir, tmp_path, had_standalone_token_before=True, env=_env(tmp_path, cfg_dir))

    assert json.loads(cli_token.read_text())["token"]["access_token"] == "acc"


def test_cancelled_install_clears_the_previous_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    install._record_install_choice("vertex", project="day-1")
    install._record_install_choice("auto")
    assert not (tmp_path / "cc-agy-plugin" / "last-install.json").exists()


def test_doctor_with_explicit_auth_still_uses_the_saved_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from server import doctor, receipts

    monkeypatch.setattr(doctor, "_discover_installed_host_mcp_env", lambda _env: {"GOOGLE_CLOUD_PROJECT": "acme-vtx", "CC_AGY_AUTH_MODE": "api_key"})
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    env = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", ""), "CC_AGY_AUTH_MODE": "vertex"}

    report = doctor.run_doctor(
        workspace_root=repo, env=env, allow_pending_login=True, store=receipts.ReceiptStore(tmp_path / "st", workspace_root=repo)
    )

    assert not any("Missing Google Cloud project ID" in issue for issue in report["issues"]), report["issues"]
    assert report["auth"]["auth_mode"] == "vertex"


# --- Round-5 non-interactive, Ctrl+D skip, and quickstart exit-code regressions ---------------


@pytest.mark.parametrize("flag_env", [{"CI": "1"}, {"CI": "true"}, {"CC_AGY_NONINTERACTIVE": "1"}])
def test_ci_and_noninteractive_never_block_in_browser_sign_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag_env: dict[str, str], capsys: pytest.CaptureFixture[str]
):
    calls: list[str] = []
    _stub_browser_failure(monkeypatch, calls)
    monkeypatch.setattr(agy_login.sys, "platform", "darwin")

    rc = agy_login.run_interactive_or_url_login(
        {"HOME": str(tmp_path), "CC_AGY_TOKEN_IN_KEYCHAIN": "1", **flag_env}
    )

    assert rc == 0
    assert calls == []
    assert "sign-in is still needed" in capsys.readouterr().out


def test_tty_eof_skip_returns_zero_and_failed_exchange_returns_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import io

    class _DummyProc:
        def poll(self) -> int | None:
            return 0

        def kill(self) -> None:
            pass

    class _EofReader:
        def readline(self) -> str:
            raise EOFError

        def close(self) -> None:
            pass

    class _CaptureWriter:
        def __init__(self) -> None:
            self.chunks: list[str] = []

        def write(self, text: str) -> int:
            self.chunks.append(text)
            return len(text)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

        def text(self) -> str:
            return "".join(self.chunks)

    monkeypatch.setattr(agy_login, "heal_enterprise_token", lambda _env: {})
    monkeypatch.setattr(agy_login, "inspect_agy_cli_offline", lambda _env: {"agy_bin": "/x/agy", "auth_configured": False})
    monkeypatch.setattr(agy_login, "can_use_local_browser_loopback", lambda _env: False)
    monkeypatch.setattr(agy_login, "is_interactive_terminal", lambda _env: True)
    monkeypatch.setattr(agy_login, "_spawn_cde_login_and_capture_url", lambda *_a, **_k: (_DummyProc(), "https://accounts.google.com/o/oauth2/auth?x=1"))
    monkeypatch.setattr(agy_login, "copy_url_to_terminal_clipboard_osc52", lambda _u: False)

    env = {"HOME": str(tmp_path), "CC_AGY_TOKEN_IN_KEYCHAIN": "0"}

    # 1. Ctrl+D (EOFError) -> intentional skip, returns 0.
    out_buf = _CaptureWriter()
    monkeypatch.setattr(agy_login, "_open_controlling_tty", lambda: (_EofReader(), out_buf))
    rc_eof = agy_login.run_interactive_or_url_login(env)
    assert rc_eof == 0
    assert "Skipped sign-in for now. Run /cc-agy-plugin:login in Claude Code when ready." in out_buf.text()

    # 2. Pasted bad code where exchange fails -> returns 1.
    out_fail = _CaptureWriter()
    monkeypatch.setattr(agy_login, "_open_controlling_tty", lambda: (io.StringIO("4/BADCODE\n"), out_fail))
    monkeypatch.setattr(
        agy_login,
        "_exchange_code_with_running_cde_proc",
        lambda *_a, **_k: {"ok": False, "error": "invalid authorization code"},
    )
    rc_bad = agy_login.run_interactive_or_url_login(env)
    assert rc_bad == 1


def test_quickstart_propagates_signin_failure_and_still_runs_doctor():
    script = (REPO_ROOT / "scripts" / "quickstart.sh").read_text(encoding="utf-8")
    assert "agy_signin.sh\" || true" not in script
    assert script.count("signin_rc=0") >= 2

    # Exercise the exact exit-code propagation logic used in both quickstart.sh branches.
    harness = """
set -euo pipefail
run_case() {
  local install_rc="$1" signin_exit="$2" doctor_exit="$3"
  local signin_rc=0 doctor_ran=0
  (exit "${signin_exit}") || signin_rc=$?
  if [[ "${signin_rc}" -ne 0 && "${install_rc}" -eq 0 ]]; then
    install_rc="${signin_rc}"
  fi
  doctor_ran=1
  (exit "${doctor_exit}") || install_rc=$?
  printf '%s:%s\\n' "${doctor_ran}" "${install_rc}"
}
run_case 0 1 0
run_case 0 0 0
"""
    proc = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, check=True)
    assert proc.stdout.strip().splitlines() == ["1:1", "1:0"]


def test_first_time_lookup_failure_keeps_unverified_token_for_one_step_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    home = tmp_path / "home"
    cfg_dir = home / ".gemini" / "antigravity-cli"
    cfg_dir.mkdir(parents=True)
    standalone = home / ".gemini" / "jetski-standalone-oauth-token"
    standalone.write_text(
        json.dumps({"token": {"access_token": "tok-new", "refresh_token": "ref-new"}, "auth_method": "gcp"}),
        encoding="utf-8",
    )

    def _net_down(*_a: Any, **_k: Any) -> list[dict[str, str]]:
        raise agy_login.LicenseLookupError("couldn't reach the license service (offline)")

    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", _net_down)
    env = _env(home, cfg_dir)
    first = agy_login.finalize_enterprise_onboarding(cfg_dir, home, had_standalone_token_before=False, env=env)
    assert first["verified"] is False
    assert first["problem"] == "lookup_failed"
    assert (cfg_dir / "antigravity-oauth-token").is_file()
    assert agy_cli._inspect_auth_files(cfg_dir, env) == (False, None)

    _fake_licenses(monkeypatch, [_LICENSE_A])
    healed = agy_login.heal_enterprise_token(env)
    assert healed["healed"] is True
    assert healed["project_id"] == "proj-alpha"
    assert agy_cli._inspect_auth_files(cfg_dir, env)[0] is True


def test_bare_auto_mode_missing_credentials_and_doctor_remediation_work_from_any_cwd(tmp_path: Path):
    from server import doctor, receipts
    from server.regions import resolve_auth_config

    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    clean_env = {"HOME": str(empty_home), "PATH": "/usr/bin:/bin"}

    with pytest.raises(ValueError) as exc_info:
        resolve_auth_config(env=clean_env)
    err = str(exc_info.value)
    assert "Missing credentials. Provide a Google AI Studio Gemini API key" in err
    assert "/cc-agy-plugin:setup" in err
    assert "/cc-agy-plugin:login" in err

    repo = tmp_path / "user-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    report = doctor.run_doctor(
        workspace_root=repo,
        env=clean_env,
        store=receipts.ReceiptStore(tmp_path / "st", workspace_root=repo),
    )
    assert any(
        "/cc-agy-plugin:setup" in rem
        and f"{REPO_ROOT / 'scripts' / 'py.sh'}" in rem
        and f"{REPO_ROOT / 'server' / 'install.py'} --host claude --auth gemini-enterprise" in rem
        for rem in report["remediations"]
    ), report["remediations"]
