"""Tests for Gemini Enterprise license lookup and stale-token repair.

`agy`'s `keyringAuth.TryAuth` rejects a saved token whose `auth_method` is "gcp" when both
`user_tier` and `project_id` are empty. These tests check that the plugin (1) never reports
such a token as signed in, (2) fills the license in after sign-in, and (3) repairs an
existing stale token without a new browser sign-in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from server import agy_cli
from server import agy_login

_LICENSE = {
    "user_tier": "gcp-ge-plus-tier",
    "tier_display_name": "Gemini Enterprise Plus",
    "project_id": "acme-ge-seats",
    "region": "global",
}


def _write_token(cfg_dir: Path, **fields: Any) -> Path:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    token_file = cfg_dir / "antigravity-oauth-token"
    payload = {
        "token": {"access_token": "dummy-access", "refresh_token": "dummy-refresh"},
        "auth_method": "gcp",
        "id_token_email": "alice@corp.example.com",
        **fields,
    }
    token_file.write_text(json.dumps(payload), encoding="utf-8")
    return token_file


def _env(home: Path, cfg_dir: Path, **extra: str) -> dict[str, str]:
    return {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
        "CC_AGY_ALLOW_TEST_LICENSE_FETCH": "1",
        **extra,
    }


@pytest.fixture
def fake_licenses(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the BAIC fetchLicenses call and record the access tokens it receives."""
    seen_tokens: list[str] = []

    def _fetch(access_token: str, *, timeout_seconds: float = 8.0) -> list[dict[str, str]]:
        del timeout_seconds
        seen_tokens.append(access_token)
        return [dict(_LICENSE)]

    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", _fetch)
    return seen_tokens


def test_incomplete_gcp_token_is_not_reported_signed_in(tmp_path: Path):
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    token_file = _write_token(cfg_dir)
    # A complete-looking onboarding cache must not rescue a token agy rejects.
    (cfg_dir / "cache").mkdir()
    (cfg_dir / "cache" / "onboarding.json").write_text(
        json.dumps({"enterpriseOnboardingComplete": True, "previousAuthMethod": "gcp"}),
        encoding="utf-8",
    )

    assert agy_cli.stored_token_onboarding_gap(token_file) is not None
    auth_ok, source = agy_cli._inspect_auth_files(cfg_dir, _env(tmp_path, cfg_dir))

    assert auth_ok is False
    assert source is None


@pytest.mark.parametrize(
    "fields",
    [{"user_tier": "gcp-ge-plus-tier"}, {"project_id": "acme-ge-seats"}],
)
def test_gcp_token_with_tier_or_project_is_signed_in(tmp_path: Path, fields: dict[str, str]):
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    token_file = _write_token(cfg_dir, **fields)

    auth_ok, source = agy_cli._inspect_auth_files(cfg_dir, _env(tmp_path, cfg_dir))

    assert agy_cli.stored_token_onboarding_gap(token_file) is None
    assert auth_ok is True
    assert source == f"token_file:{token_file}"


def test_legacy_plain_token_still_counts_as_signed_in(tmp_path: Path):
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "antigravity-oauth-token").write_text("oauth-token-123456", encoding="utf-8")

    auth_ok, _ = agy_cli._inspect_auth_files(cfg_dir, _env(tmp_path, cfg_dir))

    assert auth_ok is True


def test_finalize_writes_license_project_and_settings(tmp_path: Path, fake_licenses: list[str]):
    home = tmp_path
    cfg_dir = home / ".gemini" / "antigravity-cli"
    cfg_dir.mkdir(parents=True)
    standalone = home / ".gemini" / "jetski-standalone-oauth-token"
    standalone.write_text(
        json.dumps({"token": {"access_token": "fresh-access"}, "auth_method": "gcp"}),
        encoding="utf-8",
    )

    result = agy_login.finalize_enterprise_onboarding(
        cfg_dir, home, had_standalone_token_before=False, env=_env(home, cfg_dir)
    )

    assert fake_licenses == ["fresh-access"]
    assert result["user_tier"] == "gcp-ge-plus-tier"
    assert result["project_id"] == "acme-ge-seats"
    saved = json.loads((cfg_dir / "antigravity-oauth-token").read_text(encoding="utf-8"))
    assert saved["user_tier"] == "gcp-ge-plus-tier"
    assert saved["tier_display_name"] == "Gemini Enterprise Plus"
    assert saved["project_id"] == "acme-ge-seats"
    assert saved["region"] == "global"
    onboarding = json.loads((cfg_dir / "cache" / "onboarding.json").read_text(encoding="utf-8"))
    assert onboarding["enterpriseOnboardingComplete"] is True
    assert onboarding["tierId"] == "gcp-ge-plus-tier"
    assert onboarding["cloudaicompanionProject"] == "acme-ge-seats"
    settings = json.loads((cfg_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["gcp"] == {"project": "acme-ge-seats", "location": "global"}
    assert not standalone.exists()
    assert (cfg_dir / "antigravity-oauth-token").stat().st_mode & 0o777 == 0o600


def test_heal_repairs_stale_token_without_new_sign_in(tmp_path: Path, fake_licenses: list[str]):
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    _write_token(cfg_dir)
    env = _env(tmp_path, cfg_dir)

    first = agy_login.heal_enterprise_token(env)
    second = agy_login.heal_enterprise_token(env)

    assert first["healed"] is True
    assert first["project_id"] == "acme-ge-seats"
    assert second == {
        "healed": False,
        "already_complete": True,
        "user_tier": "gcp-ge-plus-tier",
        "project_id": "acme-ge-seats",
    }
    assert fake_licenses == ["dummy-access"]  # The second run makes no network call.
    auth_ok, _ = agy_cli._inspect_auth_files(cfg_dir, env)
    assert auth_ok is True


def test_heal_restores_backup_token(tmp_path: Path, fake_licenses: list[str]):
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    token_file = _write_token(cfg_dir)
    token_file.rename(cfg_dir / "antigravity-oauth-token.bak")

    result = agy_login.heal_enterprise_token(_env(tmp_path, cfg_dir))

    assert result["healed"] is True
    assert token_file.is_file()
    assert not (cfg_dir / "antigravity-oauth-token.bak").exists()
    assert fake_licenses == ["dummy-access"]


def test_heal_reports_failure_when_no_license(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    _write_token(cfg_dir)
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *a, **k: [])

    result = agy_login.heal_enterprise_token(_env(tmp_path, cfg_dir))

    assert result["healed"] is False
    assert "no license" in result["reason"]


def test_heal_explains_expired_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    _write_token(cfg_dir)
    stored = json.loads((cfg_dir / "antigravity-oauth-token").read_text(encoding="utf-8"))
    stored["token"]["expiry"] = "2020-01-01T00:00:00Z"
    (cfg_dir / "antigravity-oauth-token").write_text(json.dumps(stored), encoding="utf-8")
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *a, **k: [])

    result = agy_login.heal_enterprise_token(_env(tmp_path, cfg_dir))

    assert result["healed"] is False
    assert "expired" in result["reason"]
    assert "/cc-agy-plugin:login" in result["reason"]


def test_ambient_project_env_vars_are_ignored(tmp_path: Path, fake_licenses: list[str]):
    """ANTIGRAVITY_PROJECT_ID held an Antigravity workspace UUID on a real host and got saved."""
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    cfg_dir.mkdir(parents=True)
    env = _env(
        tmp_path,
        cfg_dir,
        ANTIGRAVITY_PROJECT_ID="1ffb6af1-941c-454f-94b8-5b4e3498e016",
        GOOGLE_CLOUD_PROJECT="unrelated-project",
        CLOUD_ML_PROJECT_ID="another-project",
    )

    resolved = agy_login.resolve_enterprise_license_for_token(
        {"token": {"access_token": "tok"}, "auth_method": "gcp"}, config_dir=cfg_dir, env=env
    )

    assert resolved["project_id"] == "acme-ge-seats"
    assert resolved["user_tier"] == "gcp-ge-plus-tier"


def test_never_saves_project_without_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A project with no tier would put agy in pay-as-you-go Agent Platform billing."""
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    token_file = _write_token(cfg_dir)
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *a, **k: [])
    monkeypatch.setattr(agy_login, "self_assign_enterprise_license", lambda *a, **k: None)
    env = _env(tmp_path, cfg_dir, CC_AGY_ENTERPRISE_PROJECT="acme-ge-seats")

    result = agy_login.heal_enterprise_token(env)

    assert result["healed"] is False
    saved = json.loads(token_file.read_text(encoding="utf-8"))
    assert "project_id" not in saved and "user_tier" not in saved
    assert not (cfg_dir / "settings.json").exists()


def test_explicit_tier_and_project_used_when_lookup_finds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    _write_token(cfg_dir)
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *a, **k: [])
    monkeypatch.setattr(agy_login, "self_assign_enterprise_license", lambda *a, **k: None)
    env = _env(
        tmp_path,
        cfg_dir,
        CC_AGY_ENTERPRISE_PROJECT="acme-ge-seats",
        CC_AGY_ENTERPRISE_USER_TIER="gcp-ge-standard-tier",
    )

    result = agy_login.heal_enterprise_token(env)

    assert result["healed"] is True
    assert result["user_tier"] == "gcp-ge-standard-tier"
    assert result["project_id"] == "acme-ge-seats"


def test_configured_project_selects_matching_license(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    cfg_dir.mkdir(parents=True)
    other = {**_LICENSE, "project_id": "other-project", "user_tier": "gcp-ge-standard-tier"}
    monkeypatch.setattr(
        agy_login, "fetch_enterprise_licenses", lambda *a, **k: [other, dict(_LICENSE)]
    )

    resolved = agy_login.resolve_enterprise_license_for_token(
        {"token": {"access_token": "tok"}, "auth_method": "gcp"},
        config_dir=cfg_dir,
        env=_env(tmp_path, cfg_dir, CC_AGY_ENTERPRISE_PROJECT="acme-ge-seats"),
    )

    assert resolved["project_id"] == "acme-ge-seats"
    assert resolved["user_tier"] == "gcp-ge-plus-tier"


def test_ensure_seat_ready_auto_heals(tmp_path: Path, fake_licenses: list[str]):
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    token_file = _write_token(cfg_dir)

    agy_cli.ensure_enterprise_seat_ready(_env(tmp_path, cfg_dir))

    assert agy_cli.stored_token_onboarding_gap(token_file) is None
    assert fake_licenses == ["dummy-access"]


def test_inspect_offline_explains_incomplete_token(tmp_path: Path):
    cfg_dir = tmp_path / ".gemini" / "antigravity-cli"
    _write_token(cfg_dir)
    fake_agy = tmp_path / "bin" / "agy"
    fake_agy.parent.mkdir()
    fake_agy.write_text("#!/bin/sh\necho 1.5.9\n", encoding="utf-8")
    fake_agy.chmod(0o755)

    status = agy_cli.inspect_agy_cli_offline(_env(tmp_path, cfg_dir, CC_AGY_CLI_BIN=str(fake_agy)))

    if status["binary_found"]:
        assert status["auth_configured"] is False
        assert "incomplete GCP onboarding" in status["error"]
        assert "/cc-agy-plugin:login" in status["remediation"]


def test_auth_remediation_points_at_login_not_print_mode(tmp_path: Path):
    message = agy_cli._format_auth_remediation(
        "Print mode: silent auth failed", "/x/agy", env={"HOME": str(tmp_path)}
    )

    assert "/cc-agy-plugin:login" in message
    assert "--force" in message
    assert "once in a terminal" not in message
