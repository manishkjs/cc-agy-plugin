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


def test_state_dir_refuses_a_symlink_planted_by_someone_else(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agy_login.tempfile, "gettempdir", lambda: str(tmp_path))
    env = {"HOME": str(tmp_path / "home")}
    target = agy_login._login_state_dir(env)
    target.rmdir()
    elsewhere = tmp_path / "attacker"
    elsewhere.mkdir()
    target.symlink_to(elsewhere)

    with pytest.raises(RuntimeError, match="not a directory owned by you"):
        agy_login._login_state_dir(env)


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
