"""Tests for Gemini Enterprise ADC seat mode, denied-action recovery, and Bifrost `/agy` policy."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import textwrap
from typing import Any, Mapping
from unittest import mock

import pytest

from server import bifrost_policy
from server.agy_cli import (
    _BINARY_PROBE_CACHE,
    _MODEL_FALLBACK_CACHE,
    _resolve_cached_model_for_cli,
    build_agy_subprocess_env,
    inspect_agy_cli_offline,
    normalize_denied_actions,
    resolve_enterprise_adc_seat,
    run_agy_cli_worker,
)
from server.bifrost_policy import (
    BifrostPolicyError,
    enforce_agy_model_policy,
    inspect_bifrost_policy,
    validate_policy_url,
)
from server.install import (
    build_agy_server_entry,
    build_claude_user_config,
    build_codex_server_block,
    build_enterprise_env,
)
from server.jobs import JobManager
from server.receipts import ReceiptStore
from server.regions import enforce_project_controls, read_adc_metadata, read_adc_quota_project

SEAT_PROJECT = "ge-seat-project"
POLICY_URL = "https://bifrost.internal.example.com"
AGY_VIRTUAL_KEY = "sk-bf-agy-test-virtual-key"


@pytest.fixture(autouse=True)
def _clear_caches():
    _BINARY_PROBE_CACHE.clear()
    _MODEL_FALLBACK_CACHE.clear()
    bifrost_policy.clear_policy_cache()
    yield
    _BINARY_PROBE_CACHE.clear()
    _MODEL_FALLBACK_CACHE.clear()
    bifrost_policy.clear_policy_cache()


def _write_adc(home: Path, adc_type: str = "authorized_user", quota_project: str | None = SEAT_PROJECT) -> Path:
    adc_file = home / ".config" / "gcloud" / "application_default_credentials.json"
    adc_file.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"type": adc_type, "refresh_token": "never-returned-refresh-token"}
    if quota_project:
        payload["quota_project_id"] = quota_project
    adc_file.write_text(json.dumps(payload), encoding="utf-8")
    return adc_file


def _enable_adc_setting(home: Path) -> None:
    settings = home / ".gemini" / "config" / "config.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({"userSettings": {"enableAdc": True}}), encoding="utf-8")


def _make_fake_agy(bin_path: Path, body: str) -> Path:
    script = (
        "#!/usr/bin/env python3\n"
        "import json\nimport os\nimport sys\n\n"
        "argv = sys.argv[1:]\n"
        'if "--help" in argv or "-h" in argv:\n'
        '    sys.stderr.write("Usage: agy [options]\\n  --output-format string\\n  --mode string\\n")\n'
        "    raise SystemExit(0)\n" + textwrap.dedent(body)
    )
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_text(script, encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


def _init_git_repo(repo_dir: Path) -> Path:
    repo_dir.mkdir(parents=True, exist_ok=True)
    for cmd in (
        ["git", "init", "-b", "main"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "Test User"],
    ):
        subprocess.run(cmd, cwd=str(repo_dir), check=True, capture_output=True)
    (repo_dir / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(repo_dir), check=True, capture_output=True)
    return repo_dir


class _FakeBifrost:
    """In-memory stand-in for the two read-only Bifrost policy endpoints."""

    def __init__(
        self,
        models: list[str],
        *,
        routed_model: str | None = None,
        models_status: int = 200,
    ) -> None:
        self.models = models
        self.routed_model = routed_model
        self.models_status = models_status
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout_seconds: float) -> tuple[int, bytes]:
        self.calls.append((url, dict(headers)))
        if url.endswith("/v1/models"):
            if self.models_status != 200:
                return self.models_status, b""
            return 200, json.dumps({"data": [{"id": m} for m in self.models]}).encode("utf-8")
        if url.endswith("/api/routing/rules"):
            if self.routed_model is None:
                return 404, b""
            rules = [
                {"priority": 90, "enabled": False, "cel_expression": 'model == "policy-routed"',
                 "targets": [{"provider": "agy-cli", "model": "disabled-model"}]},
                {"priority": 50, "enabled": True, "cel_expression": 'model == "policy-routed"',
                 "targets": [{"provider": "agy-cli", "model": f"agy-cli/{self.routed_model}"}]},
            ]
            return 200, json.dumps({"rules": rules}).encode("utf-8")
        return 404, b""


def _policy_env(**extra: str) -> dict[str, str]:
    return {"CC_AGY_BIFROST_POLICY_URL": POLICY_URL, "CC_AGY_BIFROST_AGY_VIRTUAL_KEY": AGY_VIRTUAL_KEY, **extra}


# ---------------------------------------------------------------------------
# 1. Gemini Enterprise human ADC seat mode
# ---------------------------------------------------------------------------


def test_seat_mode_is_off_and_project_vars_are_scrubbed_by_default(tmp_path: Path) -> None:
    env = {
        "HOME": str(tmp_path),
        "PATH": os.defpath,
        "GOOGLE_CLOUD_PROJECT": "payg-project",
        "GOOGLE_CLOUD_QUOTA_PROJECT": "payg-project",
        "BAICODE_USER_TIER": "gcp-ge-plus-tier",
        "AGY_LLM_GATEWAY_URL": "https://gateway.example.com",
        "AGY_LLM_GATEWAY_API_KEY": "gateway-secret",
        "AGY_USE_BAIC_FETCH_CONFIG": "true",
    }
    seat = resolve_enterprise_adc_seat(env)
    sub_env = build_agy_subprocess_env(env)

    assert seat["requested"] is False and seat["active"] is False
    for scrubbed in (
        "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_QUOTA_PROJECT", "AGY_ADC_AUTH", "BAICODE_USER_TIER",
        "AGY_LLM_GATEWAY_URL", "AGY_LLM_GATEWAY_API_KEY", "AGY_USE_BAIC_FETCH_CONFIG",
    ):
        assert scrubbed not in sub_env


def test_opt_in_seat_mode_injects_validated_project_location_and_tier(tmp_path: Path) -> None:
    adc_file = _write_adc(tmp_path)
    env = {
        "HOME": str(tmp_path),
        "PATH": os.defpath,
        "CC_AGY_ENTERPRISE_ADC": "true",
        "CC_AGY_ENTERPRISE_LOCATION": "eu",
        "CC_AGY_ENTERPRISE_USER_TIER": "gcp-ge-standard-tier",
        "CC_AGY_BIFROST_AGY_VIRTUAL_KEY": AGY_VIRTUAL_KEY,
        "GEMINI_API_KEY": "payg-key",
        "GOOGLE_CLOUD_PROJECT": "some-other-project",
    }
    seat = resolve_enterprise_adc_seat(env)
    sub_env = build_agy_subprocess_env(env)

    assert seat["active"] is True and seat["adc_path"] == str(adc_file)
    assert sub_env["AGY_ADC_AUTH"] == "true"
    assert sub_env["GOOGLE_CLOUD_PROJECT"] == SEAT_PROJECT
    assert sub_env["GOOGLE_CLOUD_QUOTA_PROJECT"] == SEAT_PROJECT
    assert sub_env["GOOGLE_CLOUD_LOCATION"] == "eu"
    assert sub_env["BAICODE_USER_TIER"] == "gcp-ge-standard-tier"
    assert "GEMINI_API_KEY" not in sub_env
    assert not any(key.startswith("CC_AGY_") for key in sub_env)
    assert AGY_VIRTUAL_KEY not in json.dumps(sub_env)


def test_seat_mode_auto_detects_enable_adc_setting_and_defaults_location(tmp_path: Path) -> None:
    _write_adc(tmp_path)
    _enable_adc_setting(tmp_path)
    env = {"HOME": str(tmp_path), "PATH": os.defpath}

    seat = resolve_enterprise_adc_seat(env)
    info = inspect_agy_cli_offline(env)

    assert seat["active"] is True and seat["source"] == "auto:userSettings.enableAdc"
    assert seat["location"] == "global" and seat["user_tier"] is None
    assert "BAICODE_USER_TIER" not in build_agy_subprocess_env(env)
    assert info["adc_seat_enabled"] is True
    assert info["enterprise_location"] == "global"
    assert info["baicode_user_tier"] is None
    assert info["auth_configured"] is True and str(info["credential_source"]).startswith("enterprise_adc:")

    opted_out = {**env, "CC_AGY_ENTERPRISE_ADC": "false"}
    assert resolve_enterprise_adc_seat(opted_out)["requested"] is False
    assert inspect_agy_cli_offline(opted_out)["adc_seat_enabled"] is False


def test_service_account_adc_is_never_a_seat(tmp_path: Path) -> None:
    _write_adc(tmp_path, adc_type="service_account")
    _enable_adc_setting(tmp_path)
    env = {"HOME": str(tmp_path), "PATH": os.defpath, "CC_AGY_ENTERPRISE_ADC": "true"}

    seat = resolve_enterprise_adc_seat(env)

    assert seat["requested"] is True and seat["active"] is False
    assert "authorized_user" in seat["error"] and "service_account" in seat["error"]
    assert "AGY_ADC_AUTH" not in build_agy_subprocess_env(env)
    assert inspect_agy_cli_offline(env)["adc_seat_error"] == seat["error"]


@pytest.mark.parametrize(
    ("option", "value", "expected_fragment"),
    [
        ("CC_AGY_ENTERPRISE_LOCATION", "asia", "CC_AGY_ENTERPRISE_LOCATION"),
        ("CC_AGY_ENTERPRISE_USER_TIER", "gcp-ge-free-tier", "CC_AGY_ENTERPRISE_USER_TIER"),
    ],
)
def test_seat_mode_rejects_unknown_location_and_tier(
    tmp_path: Path, option: str, value: str, expected_fragment: str
) -> None:
    _write_adc(tmp_path)
    env = {"HOME": str(tmp_path), "CC_AGY_ENTERPRISE_ADC": "true", option: value}

    seat = resolve_enterprise_adc_seat(env)

    assert seat["active"] is False and expected_fragment in seat["error"]


def test_seat_project_is_checked_against_allowed_and_blocked_projects(tmp_path: Path) -> None:
    _write_adc(tmp_path)
    base = {"HOME": str(tmp_path), "PATH": os.defpath, "CC_AGY_ENTERPRISE_ADC": "true"}
    blocked = {**base, "CC_AGY_BLOCKED_PROJECTS": SEAT_PROJECT}
    not_allowed = {**base, "CC_AGY_ALLOWED_PROJECTS": "approved-project"}
    explicit_ok = {**not_allowed, "CC_AGY_ENTERPRISE_PROJECT": "approved-project"}

    assert resolve_enterprise_adc_seat(blocked)["active"] is False
    assert "GOOGLE_CLOUD_PROJECT" not in build_agy_subprocess_env(blocked)
    with pytest.raises(ValueError, match="CC_AGY_BLOCKED_PROJECTS"):
        enforce_project_controls(auth_mode="gemini_enterprise", project=None, source_env=blocked)
    with pytest.raises(ValueError, match="CC_AGY_ALLOWED_PROJECTS"):
        enforce_project_controls(auth_mode="gemini_enterprise", project=None, source_env=not_allowed)

    enforce_project_controls(auth_mode="gemini_enterprise", project=None, source_env=explicit_ok)
    assert build_agy_subprocess_env(explicit_ok)["GOOGLE_CLOUD_PROJECT"] == "approved-project"


def test_adc_metadata_never_returns_secrets_and_quota_helper_is_unchanged(tmp_path: Path) -> None:
    _write_adc(tmp_path)
    env = {"HOME": str(tmp_path)}

    meta = read_adc_metadata(env)

    assert set(meta) == {"type", "quota_project_id", "path"}
    assert meta["type"] == "authorized_user"
    assert "never-returned-refresh-token" not in json.dumps(meta)
    assert read_adc_quota_project(env) == SEAT_PROJECT
    assert read_adc_quota_project({"HOME": str(tmp_path / "empty-home")}) is None


def test_installer_writes_enterprise_settings_for_every_host_without_secrets() -> None:
    extra_env = build_enterprise_env(
        enterprise_adc=True,
        enterprise_project=SEAT_PROJECT,
        enterprise_location="US",
        enterprise_user_tier="gcp-ge-plus-tier",
        bifrost_policy_url=POLICY_URL + "/",
        bifrost_agy_policy_model="gemini-3.8-flash-medium",
    )

    codex_block = build_codex_server_block(auth_mode="gemini-enterprise", extra_env=extra_env)
    claude_cfg = build_claude_user_config(auth_mode="gemini-enterprise", extra_env=extra_env)
    agy_entry = build_agy_server_entry(auth_mode="gemini-enterprise", extra_env=extra_env)

    assert extra_env["CC_AGY_ENTERPRISE_LOCATION"] == "us"
    assert extra_env["CC_AGY_BIFROST_POLICY_URL"] == POLICY_URL
    assert f'CC_AGY_ENTERPRISE_PROJECT = "{SEAT_PROJECT}"' in codex_block
    assert claude_cfg["enterprise_adc"] == "true" and claude_cfg["bifrost_policy_url"] == POLICY_URL
    assert agy_entry["env"]["CC_AGY_ENTERPRISE_USER_TIER"] == "gcp-ge-plus-tier"
    assert build_claude_user_config(auth_mode="gemini-enterprise") == {"auth_mode": "gemini_enterprise"}
    with pytest.raises(ValueError, match="--enterprise-user-tier"):
        build_enterprise_env(enterprise_user_tier="platinum")
    with pytest.raises(ValueError, match="https://"):
        build_enterprise_env(bifrost_policy_url="http://bifrost.internal.example.com")


def test_plugin_manifest_mirrors_every_enterprise_option_into_the_mcp_env() -> None:
    root = Path(__file__).resolve().parent.parent
    user_config = json.loads((root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))["userConfig"]
    mcp_env = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["vertex"]["env"]

    for option in (
        "enterprise_adc", "enterprise_project", "enterprise_location", "enterprise_user_tier",
        "bifrost_policy_url", "bifrost_agy_virtual_key", "bifrost_policy_user",
        "bifrost_policy_password", "bifrost_agy_policy_model",
    ):
        assert option in user_config
        assert mcp_env[f"CC_AGY_USERCFG_{option.upper()}"] == "${user_config.%s}" % option
    assert user_config["bifrost_agy_virtual_key"]["sensitive"] is True
    assert user_config["bifrost_policy_password"]["sensitive"] is True


# ---------------------------------------------------------------------------
# 2. Structured denied-action recovery
# ---------------------------------------------------------------------------


def test_denied_actions_on_a_multi_tool_turn_are_matched_one_to_one() -> None:
    tool_calls = [
        {"name": "run_command", "args": {"CommandLine": "ls", "Cwd": "/work"}},
        {"name": "write_file", "args": {"Path": "notes.md", "Content": "SECRET FILE BODY"}},
        {"name": "run_command", "args": {"CommandLine": "npm test", "Cwd": "/work/web"}},
        {"name": "view_file", "args": {"AbsolutePath": "/etc/hosts"}},
    ]
    raw = [
        {"tool_name": "run_command", "reason": "Terminal command auto-execution policy: Require review"},
        {"tool_name": "view_file", "reason": "Outside of file access policy: Always ask"},
        {"tool_name": "run_command"},
        {"tool_name": "write_file", "step_index": 2},
    ]

    records = normalize_denied_actions(raw, tool_calls, step_indexes=[1, 2, 3, 4], conversation_id="conv-9")

    assert [r["tool"] for r in records] == ["run_command", "view_file", "run_command", "write_file"]
    assert [r["kind"] for r in records] == ["command", "read", "command", "write"]
    assert records[0]["command"] == "npm test" and records[0]["workdir"] == "/work/web"
    assert records[1]["path"] == "/etc/hosts" and "Always ask" in records[1]["reason"]
    assert records[2]["command"] == "ls"
    assert records[3]["path"] == "notes.md" and records[3]["correlation"] == "step_index"
    assert all(r["agy_conversation_id"] == "conv-9" for r in records)
    assert "SECRET FILE BODY" not in json.dumps(records)


def test_denied_actions_without_details_fall_back_safely() -> None:
    tool_calls = [{"name": "edit_file", "args": {"path": "a.py"}}]

    by_position = normalize_denied_actions(["blocked by policy"], tool_calls)
    unmatched = normalize_denied_actions([{"tool_name": "run_command", "command": "make"}], tool_calls)

    assert by_position == [
        {"tool": "edit_file", "kind": "edit", "correlation": "position", "path": "a.py", "reason": "blocked by policy"}
    ]
    assert unmatched == [{"tool": "run_command", "kind": "command", "correlation": "none", "command": "make"}]
    assert normalize_denied_actions(None, tool_calls) == []


def test_worker_returns_structured_denied_actions_and_conversation_id(tmp_path: Path) -> None:
    fake_agy = _make_fake_agy(
        tmp_path / "bin" / "agy",
        """
        _ = sys.stdin.read()
        events = [
            {"event": "step_update", "step_update": {"conversation_id": "conv-d1", "step_index": 1,
             "tool_info": {"name": "edit_file", "parameters": {"path": "app.py"}}}},
            {"event": "step_update", "step_update": {"conversation_id": "conv-d1", "step_index": 2,
             "tool_info": {"name": "run_command", "parameters": {"CommandLine": "pytest -q", "Cwd": "."}}}},
            {"event": "result", "result": {"status": "SUCCESS", "conversation_id": "conv-d1", "num_turns": 1,
             "response": "Edited app.py; tests need approval.",
             "denied_actions": [{"tool_name": "run_command", "reason": "Require review"}],
             "usage": {"input_tokens": 40, "output_tokens": 8}}},
        ]
        for event in events:
            print(json.dumps(event), flush=True)
        """,
    )
    cfg = tmp_path / "cfg"
    cfg.mkdir()

    out = run_agy_cli_worker(
        {
            "workspace_dir": str(tmp_path),
            "prompt": "Fix app.py",
            "env": {"CC_AGY_CLI_BIN": str(fake_agy), "CC_AGY_CLI_CONFIG_DIR": str(cfg)},
        }
    )

    assert out["stop_reason"] == "DENIED_ACTIONS"
    assert out["agy_conversation_id"] == "conv-d1"
    assert out["denied_actions"] == [
        {
            "tool": "run_command",
            "kind": "command",
            "correlation": "tool_name",
            "command": "pytest -q",
            "workdir": ".",
            "reason": "Require review",
            "agy_conversation_id": "conv-d1",
        }
    ]
    assert out["tool_calls"][1] == {"name": "run_command", "args": {"CommandLine": "pytest -q", "Cwd": "."}}


def test_denied_job_pauses_unapplied_and_resumes_the_same_conversation(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    store = ReceiptStore(workspace_root=repo, data_dir=tmp_path / "data")
    argv_log = tmp_path / "argv.log"
    fake_agy = _make_fake_agy(
        tmp_path / "bin" / "agy",
        f"""
        prompt = sys.stdin.read()
        with open({str(argv_log)!r}, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\\n")
        resumed = "--conversation" in argv
        with open("app.py", "w", encoding="utf-8") as fh:
            fh.write("VALUE = 3\\n" if resumed else "VALUE = 2\\n")
        result = {{"status": "SUCCESS", "conversation_id": "conv-denied-1", "num_turns": 1,
                  "response": "done" if resumed else "blocked",
                  "usage": {{"input_tokens": 30, "output_tokens": 6}}}}
        if not resumed:
            print(json.dumps({{"event": "step_update", "step_update": {{"step_index": 1,
                  "tool_info": {{"name": "run_command", "parameters": {{"CommandLine": "make test"}}}}}}}}), flush=True)
            result["denied_actions"] = [{{"tool_name": "run_command", "reason": "Require review"}}]
        print(json.dumps({{"event": "result", "result": result}}), flush=True)
        """,
    )
    env_patch = {
        "CC_AGY_AUTH_MODE": "gemini_enterprise",
        "CC_AGY_CLI_BIN": str(fake_agy),
        "ANTIGRAVITY_OAUTH_TOKEN": "seat-oauth-token",
    }
    with mock.patch.dict(os.environ, env_patch, clear=False):
        jm = JobManager(workspace_root=repo, store=store)
        try:
            paused = jm.delegate_start(prompt="Bump VALUE in app.py", wait=True)
            job_id = paused["job_id"]
            receipt = store.read_receipt(job_id)
            resumed = jm.delegate_continue(job_id=job_id, follow_up_prompt="Approved. `make test` passed.", wait=True)
        finally:
            jm.shutdown(wait=True)

    assert paused["status"] == "needs_input" and paused["applied"] is False
    assert paused["denied_actions"][0]["command"] == "make test"
    assert "delegate_continue" in paused["next_action"]
    assert str(store.worktree_path(job_id)) in paused["next_action"]
    assert receipt["denied_actions"][0]["agy_conversation_id"] == "conv-denied-1"
    assert (repo / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    second_argv = json.loads(argv_log.read_text(encoding="utf-8").splitlines()[-1])
    assert second_argv[second_argv.index("--conversation") + 1] == "conv-denied-1"
    assert resumed["status"] == "ok" and "denied_actions" not in resumed


# ---------------------------------------------------------------------------
# 3. Central Bifrost `/agy` policy governance
# ---------------------------------------------------------------------------


def test_policy_is_off_without_a_policy_url() -> None:
    verdict = enforce_agy_model_policy("gemini-4-argon", "high", {"CC_AGY_BIFROST_VIRTUAL_KEY": "sdk-key"})

    assert verdict == {"enabled": False, "model": "gemini-4-argon", "allowed": None, "policy_routed": False}
    assert inspect_bifrost_policy({})["enabled"] is False


def test_policy_allows_listed_models_and_sends_both_key_headers() -> None:
    gateway = _FakeBifrost(["agy-cli/gemini-4-argon-high", "vertex/gemini-3.8-flash"])

    verdict = enforce_agy_model_policy("gemini-4-argon", "high", _policy_env(), transport=gateway)

    url, headers = gateway.calls[0]
    assert verdict["enabled"] is True and verdict["model"] == "gemini-4-argon"
    assert verdict["allowed"] == frozenset({"gemini-4-argon-high"})
    assert url == f"{POLICY_URL}/v1/models"
    assert headers["Authorization"] == f"Bearer {AGY_VIRTUAL_KEY}" and headers["x-bf-vk"] == AGY_VIRTUAL_KEY


def test_policy_denies_unlisted_model_and_revoked_key() -> None:
    gateway = _FakeBifrost(["agy-cli/gemini-3.8-flash-medium"])
    revoked = _FakeBifrost([], models_status=401)

    with pytest.raises(BifrostPolicyError, match="'gemini-4-argon' is not allowed"):
        enforce_agy_model_policy("gemini-4-argon", "high", _policy_env(), transport=gateway)
    bifrost_policy.clear_policy_cache()
    with pytest.raises(BifrostPolicyError, match="HTTP 401"):
        enforce_agy_model_policy("gemini-3.8-flash-medium", None, _policy_env(), transport=revoked)


def test_policy_routed_resolves_from_routing_rules_then_fallback_model() -> None:
    allowed = ["agy-cli/policy-routed", "agy-cli/gemini-3.8-flash-medium", "agy-cli/gemini-4-argon-high"]
    with_rules = _FakeBifrost(allowed, routed_model="gemini-4-argon-high")
    rules_env = _policy_env(CC_AGY_BIFROST_POLICY_USER="reader", CC_AGY_BIFROST_POLICY_PASSWORD="reader-pass")
    fallback_env = _policy_env(CC_AGY_BIFROST_AGY_POLICY_MODEL="agy-cli/gemini-3.8-flash-medium")

    routed = enforce_agy_model_policy("agy-cli/policy-routed", "low", rules_env, transport=with_rules)
    bifrost_policy.clear_policy_cache()
    fell_back = enforce_agy_model_policy("policy-routed", None, fallback_env, transport=_FakeBifrost(allowed))
    bifrost_policy.clear_policy_cache()

    assert routed["policy_routed"] is True and routed["model"] == "gemini-4-argon-high"
    assert with_rules.calls[1][1]["Authorization"].startswith("Basic ")
    assert fell_back["model"] == "gemini-3.8-flash-medium"
    with pytest.raises(BifrostPolicyError, match="could not be resolved"):
        enforce_agy_model_policy("policy-routed", None, _policy_env(), transport=_FakeBifrost(allowed))
    bifrost_policy.clear_policy_cache()
    with pytest.raises(BifrostPolicyError, match="does not allow 'policy-routed'"):
        enforce_agy_model_policy(
            "policy-routed", None, fallback_env, transport=_FakeBifrost(["agy-cli/gemini-3.8-flash-medium"])
        )


def test_policy_results_are_cached_and_failures_never_allow_a_run() -> None:
    gateway = _FakeBifrost(["agy-cli/*"])

    def unreachable(url: str, headers: Mapping[str, str], timeout_seconds: float) -> tuple[int, bytes]:
        raise OSError("connection refused")

    enforce_agy_model_policy("any-model", None, _policy_env(), transport=gateway)
    enforce_agy_model_policy("another-model", None, _policy_env(), transport=gateway)
    no_cache_env = _policy_env(CC_AGY_BIFROST_POLICY_TTL_SECONDS="0")

    assert len(gateway.calls) == 1
    with pytest.raises(BifrostPolicyError, match="Could not reach"):
        enforce_agy_model_policy("any-model", None, no_cache_env, transport=unreachable)
    with pytest.raises(BifrostPolicyError, match="HTTP 302"):
        enforce_agy_model_policy("any-model", None, no_cache_env, transport=lambda *_: (302, b""))


def test_policy_url_and_key_validation() -> None:
    assert validate_policy_url("https://bifrost.example.com/") == "https://bifrost.example.com"
    assert validate_policy_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    for unsafe in (
        "http://bifrost.example.com",
        "https://user:pass@bifrost.example.com",
        "https://bifrost.example.com/?key=1",
        "ftp://bifrost.example.com",
    ):
        with pytest.raises(ValueError):
            validate_policy_url(unsafe)
    with pytest.raises(ValueError, match="no Virtual Key"):
        enforce_agy_model_policy("m", None, {"CC_AGY_BIFROST_POLICY_URL": POLICY_URL})
    assert bifrost_policy._RefuseRedirects().redirect_request(None, None, 302, "Found", {}, "https://evil") is None


def test_doctor_summary_reports_status_and_allowed_count_without_the_key() -> None:
    gateway = _FakeBifrost(["agy-cli/policy-routed", "agy-cli/gemini-3.8-flash-medium"])
    env = _policy_env(CC_AGY_BIFROST_AGY_POLICY_MODEL="gemini-3.8-flash-medium")

    offline = inspect_bifrost_policy(env)
    live = inspect_bifrost_policy(env, live=True, transport=gateway)
    broken = inspect_bifrost_policy({"CC_AGY_BIFROST_POLICY_URL": "http://bifrost.example.com"})

    assert offline["enabled"] is True and offline["allowed_model_count"] is None
    assert live["allowed_model_count"] == 2 and live["policy_routed_model"] == "gemini-3.8-flash-medium"
    assert AGY_VIRTUAL_KEY not in json.dumps(offline) + json.dumps(live)
    assert broken["enabled"] is False and "https://" in broken["error"]


def test_agy_model_resolution_enforces_policy_before_spawning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = _FakeBifrost(["agy-cli/policy-routed", "agy-cli/gemini-3.8-flash-medium"])
    monkeypatch.setattr(bifrost_policy, "_default_transport", gateway)
    env = _policy_env(
        CC_AGY_CLI_CONFIG_DIR=str(tmp_path / "cfg"),
        CC_AGY_BIFROST_AGY_POLICY_MODEL="gemini-3.8-flash-medium",
    )

    model, effort, note = _resolve_cached_model_for_cli("agy", "policy-routed", "high", env)

    assert (model, effort) == ("gemini-3.8-flash-medium", None)
    assert "policy-routed" in str(note)
    with pytest.raises(BifrostPolicyError, match="not allowed"):
        _resolve_cached_model_for_cli("agy", "gemini-4-argon", "high", env)
    with pytest.raises(BifrostPolicyError, match="not allowed"):
        run_agy_cli_worker(
            {
                "workspace_dir": str(tmp_path),
                "prompt": "never sent",
                "model": "gemini-4-argon",
                "env": {**env, "CC_AGY_CLI_BIN": str(_make_fake_agy(tmp_path / "bin" / "agy", "raise SystemExit(3)\n"))},
            }
        )


# ---------------------------------------------------------------------------
# 4. Cached `agy models` catalog, delta resume, and the OpenCode entry point
# ---------------------------------------------------------------------------

_CATALOG_AGY_BODY = """
    if argv and argv[-1] == "models":
        with open(os.environ["FAKE_AGY_LOG"], "a", encoding="utf-8") as fh:
            fh.write("models\\n")
        sys.stderr.write("Fetching available models...\\n")
        print("gemini-4-argon-high\\tGemini 4 Argon (High)")
        print("gemini-3.8-flash-medium\\tGemini 3.8 Flash (Medium)")
        print("gemini-3.8-flash-low\\tGemini 3.8 Flash (Low)")
        raise SystemExit(0)
"""


def _catalog_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "FAKE_AGY_LOG": str(tmp_path / "agy.log"),
        "CC_AGY_CLI_CONFIG_DIR": str(tmp_path / "cfg"),
        "CC_AGY_STATE_DIR": str(tmp_path / "state"),
        "CC_AGY_MODEL_CATALOG": "1",
        **extra,
    }


def test_model_catalog_is_parsed_cached_in_memory_and_on_disk(tmp_path: Path) -> None:
    from server import agy_state

    fake_agy = str(_make_fake_agy(tmp_path / "bin" / "agy", _CATALOG_AGY_BODY))
    env = _catalog_env(tmp_path)

    first = agy_state.load_agy_model_catalog(fake_agy, env)
    second = agy_state.load_agy_model_catalog(fake_agy, env)
    agy_state.clear_memory_caches()
    from_disk = agy_state.load_agy_model_catalog(fake_agy, env)
    expired = agy_state.load_agy_model_catalog(fake_agy, {**env, "CC_AGY_MODEL_CATALOG_TTL_SECONDS": "0"})
    state_file = tmp_path / "state" / "agy_model_catalog.json"

    assert first == ("gemini-4-argon-high", "gemini-3.8-flash-medium", "gemini-3.8-flash-low")
    assert second == first and from_disk == first and expired == first
    assert (tmp_path / "agy.log").read_text(encoding="utf-8").count("models") == 2
    assert stat.S_IMODE(state_file.stat().st_mode) == 0o600
    assert agy_state.catalog_offers("gemini-4-argon", first) and not agy_state.catalog_offers("gemini-flash-next", first)
    assert agy_state.inspect_agy_model_catalog(fake_agy, env)["source"] == "disk_cache"
    assert agy_state.load_agy_model_catalog(fake_agy, {**env, "CC_AGY_MODEL_CATALOG": "0"}) is None


def test_unreadable_catalog_means_unknown_and_changes_nothing(tmp_path: Path) -> None:
    from server import agy_state

    broken_agy = str(_make_fake_agy(tmp_path / "bin" / "agy", "print('not a model list')\nraise SystemExit(1)\n"))
    env = _catalog_env(tmp_path)

    assert agy_state.load_agy_model_catalog(broken_agy, env) is None
    assert agy_state.load_agy_model_catalog(str(tmp_path / "missing-agy"), env) is None
    assert _resolve_cached_model_for_cli(broken_agy, "gemini-flash-next", "low", env) == (
        "gemini-flash-next", "low", None,
    )
    assert agy_state.parse_agy_models_output("\x1b[32mgemini-x-1\x1b[0m  Name\nBad Slug Here\n\n") == ("gemini-x-1",)


def test_model_mismatch_is_fixed_from_the_catalog_before_any_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_agy = str(_make_fake_agy(tmp_path / "bin" / "agy", _CATALOG_AGY_BODY))
    env = _catalog_env(tmp_path)

    model, effort, note = _resolve_cached_model_for_cli(fake_agy, "gemini-flash-next", "low", env)
    offered = _resolve_cached_model_for_cli(fake_agy, "gemini-4-argon", "high", env)

    assert model == "gemini-3.8-flash-low" and effort is None and "gemini-flash-next" in str(note)
    assert offered == ("gemini-4-argon", "high", None)

    gateway = _FakeBifrost(["agy-cli/policy-routed", "agy-cli/gemini-3.8-flash-medium", "agy-cli/gemini-9-unoffered"])
    monkeypatch.setattr(bifrost_policy, "_default_transport", gateway)
    _MODEL_FALLBACK_CACHE.clear()
    governed = {**env, **_policy_env(CC_AGY_BIFROST_AGY_POLICY_MODEL="gemini-9-unoffered")}

    intersected, _, _ = _resolve_cached_model_for_cli(fake_agy, "gemini-flash-next", "low", governed)

    assert intersected == "gemini-3.8-flash-medium"
    with pytest.raises(BifrostPolicyError, match="does not offer"):
        _resolve_cached_model_for_cli(fake_agy, "policy-routed", None, governed)


_RESUME_AGY_BODY = """
    prompt = sys.stdin.read()
    with open(os.environ["FAKE_AGY_LOG"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"argv": argv, "prompt": prompt}) + "\\n")
    if "--conversation" in argv:
        if os.environ.get("FAKE_AGY_BREAK_RESUME"):
            print(json.dumps({"status": "ERROR", "error": "conversation not found"}))
            raise SystemExit(1)
        print(json.dumps({"status": "SUCCESS", "response": "follow-up answer", "conversation_id": "conv-ask-1",
                          "num_turns": 2, "usage": {"input_tokens": 200, "cache_read_tokens": 1900,
                                                    "output_tokens": 70}}))
    else:
        print(json.dumps({"status": "SUCCESS", "response": "first answer", "conversation_id": "conv-ask-1",
                          "num_turns": 1, "usage": {"input_tokens": 2000, "output_tokens": 40}}))
"""


def _ask_request(tmp_path: Path, question: str, **overrides: Any) -> dict[str, Any]:
    env = {
        "PATH": os.defpath,
        "FAKE_AGY_LOG": str(tmp_path / "agy.log"),
        "CC_AGY_CLI_BIN": str(tmp_path / "bin" / "agy"),
        "CC_AGY_CLI_CONFIG_DIR": str(tmp_path / "cfg"),
        "CC_AGY_STATE_DIR": str(tmp_path / "state"),
        **overrides.pop("env", {}),
    }
    context = "=== FILE: big.py ===\nSECRET_CONTEXT_BODY\n=== END FILE ==="
    return {
        "model": "gemini-3.8-flash-low",
        "prompt": f"{context}\n\n{question}",
        "prompt_prefix": context,
        "prompt_delta": question,
        "session_id": "host-session-A",
        "workspace_root": str(tmp_path),
        "env": env,
        **overrides,
    }


def _agy_calls(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "agy.log").read_text(encoding="utf-8").splitlines()]


def test_follow_up_ask_resumes_with_only_the_new_question_and_delta_usage(tmp_path: Path) -> None:
    from server.agy_cli import run_agy_cli_ask

    _make_fake_agy(tmp_path / "bin" / "agy", _RESUME_AGY_BODY)

    first = run_agy_cli_ask(_ask_request(tmp_path, "What does big.py do?"))
    second = run_agy_cli_ask(_ask_request(tmp_path, "Which function is slowest?"))
    calls = _agy_calls(tmp_path)
    stored = (tmp_path / "state" / "agy_conversations.json").read_text(encoding="utf-8")

    assert first["resumed_conversation"] is False and second["resumed_conversation"] is True
    assert "--conversation" not in calls[0]["argv"] and "SECRET_CONTEXT_BODY" in calls[0]["prompt"]
    assert calls[1]["argv"][calls[1]["argv"].index("--conversation") + 1] == "conv-ask-1"
    assert "Which function is slowest?" in calls[1]["prompt"] and "SECRET_CONTEXT_BODY" not in calls[1]["prompt"]
    assert all("SECRET_CONTEXT_BODY" not in " ".join(call["argv"]) for call in calls)
    assert second["usage"]["prompt_token_count"] == 100 and second["usage"]["candidates_token_count"] == 30
    assert second["usage"]["cached_content_token_count"] == 1900
    assert "SECRET_CONTEXT_BODY" not in stored and "slowest" not in stored
    assert stat.S_IMODE((tmp_path / "state" / "agy_conversations.json").stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "overrides",
    [
        {"session_id": "host-session-B"},
        {"model": "gemini-3.8-flash-medium"},
        {"workspace_root": "/another/workspace"},
        {"prompt_prefix": "=== FILE: other.py ==="},
        {"env": {"CC_AGY_CONVERSATION_TTL_SECONDS": "0"}},
        {"session_id": ""},
    ],
)
def test_conversation_is_not_reused_across_session_model_workspace_or_context(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    from server.agy_cli import run_agy_cli_ask

    _make_fake_agy(tmp_path / "bin" / "agy", _RESUME_AGY_BODY)

    run_agy_cli_ask(_ask_request(tmp_path, "First question"))
    other = run_agy_cli_ask(_ask_request(tmp_path, "Second question", **overrides))

    assert other["resumed_conversation"] is False
    assert "--conversation" not in _agy_calls(tmp_path)[1]["argv"]


def test_failed_or_expired_resume_falls_back_to_the_full_prompt(tmp_path: Path) -> None:
    from server import agy_state
    from server.agy_cli import run_agy_cli_ask

    _make_fake_agy(tmp_path / "bin" / "agy", _RESUME_AGY_BODY)
    run_agy_cli_ask(_ask_request(tmp_path, "First question"))

    recovered = run_agy_cli_ask(_ask_request(tmp_path, "Second question", env={"FAKE_AGY_BREAK_RESUME": "1"}))
    calls = _agy_calls(tmp_path)

    assert recovered["resumed_conversation"] is False and recovered["text"] == "first answer"
    assert "--conversation" in calls[1]["argv"]
    assert "--conversation" not in calls[2]["argv"] and "SECRET_CONTEXT_BODY" in calls[2]["prompt"]

    state_file = tmp_path / "state" / "agy_conversations.json"
    aged = json.loads(state_file.read_text(encoding="utf-8"))
    for entry in aged.values():
        entry["updated_at"] -= agy_state.DEFAULT_CONVERSATION_TTL_SECONDS + 5
    state_file.write_text(json.dumps(aged), encoding="utf-8")

    assert run_agy_cli_ask(_ask_request(tmp_path, "Third question"))["resumed_conversation"] is False


def test_opencode_entry_point_registers_server_commands_and_agent_without_secrets_on_disk(tmp_path: Path) -> None:
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    root = Path(__file__).resolve().parent.parent
    auth_dir = tmp_path / "data" / "opencode"
    auth_dir.mkdir(parents=True)
    (auth_dir / "auth.json").write_text(
        json.dumps({"cc-agy": {"type": "api", "key": AGY_VIRTUAL_KEY,
                               "metadata": {"authMode": "gemini_enterprise", "bifrostPolicyUrl": POLICY_URL}}}),
        encoding="utf-8",
    )
    script = (
        f"const m = await import({json.dumps(str(root / 'opencode-plugin' / 'index.ts'))});\n"
        "const cfg = {command: {cost: {template: 'user-defined'}}};\n"
        "m.applyConfig(cfg, process.env);\n"
        "const kept = {mcp: {servers: {agy: {command: ['mine']}}}};\n"
        "m.applyConfig(kept, process.env);\n"
        "const plugin = await m.default({});\n"
        "const seat = await plugin.auth.methods[0].authorize({});\n"
        "const noKey = await plugin.auth.methods[1].authorize({geminiApiKey: ' '});\n"
        "console.log(JSON.stringify({cfg, kept, seat, noKey, hooks: Object.keys(plugin)}));\n"
    )
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=60, check=False,
        env={"PATH": os.environ.get("PATH", os.defpath), "XDG_DATA_HOME": str(tmp_path / "data"), "HOME": str(tmp_path)},
    )
    if proc.returncode != 0 and "ERR_UNKNOWN_FILE_EXTENSION" in proc.stderr:
        pytest.skip("this node version cannot load TypeScript directly")
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    server = out["cfg"]["mcp"]["agy"]
    package = json.loads((root / "package.json").read_text(encoding="utf-8"))
    source = (root / "opencode-plugin" / "index.ts").read_text(encoding="utf-8")

    assert server["type"] == "local" and server["command"][1] == str(root / "server" / "main.py")
    assert server["environment"] == {
        "CC_AGY_AUTH_MODE": "gemini_enterprise",
        "CC_AGY_BIFROST_POLICY_URL": POLICY_URL,
        "CC_AGY_BIFROST_AGY_VIRTUAL_KEY": AGY_VIRTUAL_KEY,
    }
    assert set(out["cfg"]["command"]) == {"agy", "cost", "delegate", "doctor", "jobs"}
    assert out["cfg"]["command"]["cost"] == {"template": "user-defined"}
    assert out["cfg"]["agent"]["gemini-delegator"]["permission"] == {"edit": "deny"}
    assert "agy" not in {k for k in out["kept"]["mcp"] if k != "servers"}
    assert out["seat"]["key"] == "gemini-enterprise-seat" and out["noKey"] == {"type": "failed"}
    assert sorted(out["hooks"]) == ["auth", "config"]
    assert package["main"] == "./opencode-plugin/index.ts"
    assert "createServer" not in source and "Bun.serve" not in source and ".listen(" not in source
    assert any(str(p).endswith(".opencode/skills/cc-agy-plugin/SKILL.md") for p in out["cfg"].get("instructions", []))
    assert len(plugin_methods := out["cfg"]) >= 0


def test_corrupt_conversation_json_and_concurrent_lease_isolation(tmp_path: Path) -> None:
    from server import agy_state

    env = {"CC_AGY_STATE_DIR": str(tmp_path / "state")}
    state_path = tmp_path / "state" / "agy_conversations.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"k1": {"conversation_id": "conv-1", "updated_at": "corrupt-ts", "turns": "not-an-int"}}),
        encoding="utf-8",
    )
    assert agy_state.lookup_conversation("k1", env) is None

    agy_state.remember_conversation("k2", "conv-2", turns=1, usage={"prompt_token_count": 100}, env=env)
    claimed = agy_state.lookup_conversation("k2", env, claim_lease=True)
    assert claimed is not None and claimed["conversation_id"] == "conv-2"
    # Second concurrent caller must not collide on the in-flight conversation
    assert agy_state.lookup_conversation("k2", env, claim_lease=True) is None
    agy_state.release_conversation_lease("k2", env)
    reclaimed = agy_state.lookup_conversation("k2", env, claim_lease=True)
    assert reclaimed is not None and reclaimed["conversation_id"] == "conv-2"
    agy_state.release_conversation_lease("k2", env)


def test_failed_resume_computes_delta_usage_and_timeout_falls_back(tmp_path: Path) -> None:
    from server.agy_cli import run_agy_cli_ask

    body = """
    log_path = os.environ["FAKE_AGY_LOG"]
    prompt = sys.stdin.read()
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"argv": argv, "prompt": prompt}) + "\\n")
    if "--conversation" in argv:
        if os.environ.get("FAKE_AGY_FAIL_RESUME_WITH_USAGE") == "1":
            print(json.dumps({
                "status": "ERROR",
                "error": "resume failed after partial work",
                "conversation_id": "conv-delta-1",
                "num_turns": 2,
                "usage": {"input_tokens": 250, "cache_read_tokens": 1900, "output_tokens": 70}
            }))
            raise SystemExit(1)
    print(json.dumps({
        "status": "SUCCESS",
        "conversation_id": "conv-delta-1",
        "num_turns": 1,
        "response": "fresh answer",
        "usage": {"input_tokens": 2000, "cache_read_tokens": 0, "output_tokens": 50}
    }))
    """
    _make_fake_agy(tmp_path / "bin" / "agy", body)
    first = run_agy_cli_ask(_ask_request(tmp_path, "First question"))
    assert first["usage"]["prompt_token_count"] == 2000

    second = run_agy_cli_ask(
        _ask_request(tmp_path, "Second question", env={"FAKE_AGY_FAIL_RESUME_WITH_USAGE": "1"})
    )
    assert second["resumed_conversation"] is False
    # Failed Turn 2 cumulative prompt_token_count is 250 + 1900 = 2150; delta over Turn 1 (2000) is 150.
    # Plus fresh fallback (2000) = 2150, NOT 2150 + 2000 = 4150!
    assert second["usage"]["prompt_token_count"] == 2150
    assert second["usage"]["candidates_token_count"] == 70


def test_pinned_gemini_enterprise_never_falls_back_to_sdk_and_ignores_ambient_gcp_project(tmp_path: Path) -> None:
    from server.doctor import format_doctor_human, run_doctor
    from server.genai_ask import _default_genai_backend
    from server.install import preflight
    from server.regions import resolve_auth_config

    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", "raise SystemExit(0)")
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()

    # Unsigned agy binary during preflight: ready=True with pending_login=True and warning/next_steps
    pre = preflight(
        {"CC_AGY_CLI_CONFIG_DIR": str(cfg_dir), "HOME": str(tmp_path)},
        auth_mode="gemini-enterprise",
        agy_cli_path=str(fake_agy),
    )
    assert pre["ready"] is True
    assert pre["agy_cli"]["pending_login"] is True
    assert pre["warnings"] and pre["next_steps"]

    # Ambient GOOGLE_CLOUD_PROJECT must be ignored when auth_mode is gemini_enterprise without ADC seat mode
    auth = resolve_auth_config(
        env={
            "CC_AGY_AUTH_MODE": "gemini_enterprise",
            "CC_AGY_CLI_BIN": str(fake_agy),
            "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
            "ANTIGRAVITY_OAUTH_TOKEN": "tok-123",
            "GOOGLE_CLOUD_PROJECT": "ambient-payg-project",
            "HOME": str(tmp_path),
        }
    )
    assert auth["auth_mode"] == "gemini_enterprise"
    assert auth["project"] is None

    with pytest.raises(RuntimeError, match="Live Google Search grounding"):
        _default_genai_backend(
            {
                "auth_mode": "gemini_enterprise",
                "google_search": True,
                "prompt": "search test",
            }
        )

    rep = run_doctor(
        tmp_path,
        env={
            "CC_AGY_AUTH_MODE": "gemini_enterprise",
            "CC_AGY_CLI_BIN": str(fake_agy),
            "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
            "ANTIGRAVITY_OAUTH_TOKEN": "oauth-token-1234567890",
            "HOME": str(tmp_path),
            "CC_AGY_DATA_DIR": str(tmp_path / "data"),
        },
    )
    human = format_doctor_human(rep)
    assert "[OK] cc-agy-plugin doctor" in human or "[WARN] cc-agy-plugin doctor" in human
    assert "Auth mode:      gemini_enterprise" in human


def test_bifrost_effort_suffix_matching_gateway_controls_and_fast_pre_tool_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import time
    from server.bifrost_policy import is_model_allowed
    from server.gateway import DEFAULT_PRICES_PER_MTOK, gemini_endpoint, translate_anthropic_to_gemini_payload

    assert is_model_allowed("gemini-3.8-flash-high", None, frozenset({"gemini-3.8-flash"})) is True
    assert DEFAULT_PRICES_PER_MTOK["gemini-3.8-flash"] == (0.50, 0.05, 3.00)
    assert DEFAULT_PRICES_PER_MTOK["claude-opus-5-5"] == (4.00, 0.20, 20.00)

    translated = translate_anthropic_to_gemini_payload(
        {
            "messages": [
                {"role": "user", "content": "Part A"},
                {"role": "user", "content": "Part B"},
            ]
        },
        "gemini-flash-next",
    )
    assert len(translated["contents"]) == 1
    assert len(translated["contents"][0]["parts"]) == 2

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CC_AGY_VERTEX_BASE_URL", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setattr("server.gateway._vertex_access_token", lambda: "mock-oauth-bearer-token")
    url, headers = gemini_endpoint("gemini-flash-next")
    assert "/locations/global/" in url

    monkeypatch.setenv("CC_AGY_BLOCKED_PROJECTS", "my-proj")
    with pytest.raises(Exception, match="CC_AGY_BLOCKED_PROJECTS"):
        gemini_endpoint("gemini-flash-next")

    guard_script = Path(__file__).resolve().parent.parent / "scripts" / "pre_tool_guard.py"
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(guard_script)],
        input=json.dumps({"tool_name": "Read", "tool_input": {"file_path": "README.md"}}),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    elapsed = time.monotonic() - t0
    assert proc.returncode == 0
    assert elapsed < 0.45, f"pre_tool_guard took {elapsed:.3f}s (expected < 0.45s)"
