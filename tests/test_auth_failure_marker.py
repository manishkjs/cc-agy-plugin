"""Tests for auth-failure marker tracking, cache invalidation on re-login, and login prompt text."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from typing import Any

import pytest

from server import agy_acp
from server import agy_cli
from server import agy_login
from server import agy_state

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_BROWSER_NOTICE = (
    "A browser window will open for Google sign-in; finish it there. This waits up to 90 seconds."
)
_LICENSE_OK = {
    "user_tier": "gcp-ge-plus-tier",
    "tier_display_name": "Gemini Enterprise Plus",
    "project_id": "proj-alpha",
    "region": "global",
}


def _make_fake_agy(
    path: Path,
    *,
    mode: str = "ok",
    models_output: str = "gemini-4-hydrogen-medium  Gemini 4 Hydrogen\n",
    error_text: str = "HTTP 401 Unauthorized: token has been expired or revoked",
) -> Path:
    """Create a lightweight fake external `agy` executable (no network, no real CLI)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""#!{sys.executable}
import json
import os
import sys

args = sys.argv[1:]
if "--help" in args:
    print("Antigravity CLI --output-format --mode")
    sys.exit(0)
if "--version" in args:
    print("agy 1.2.3")
    sys.exit(0)
if "models" in args:
    mode_file = os.environ.get("FAKE_AGY_MODE_FILE")
    cur_mode = open(mode_file, encoding="utf-8").read().strip() if mode_file and os.path.exists(mode_file) else {mode!r}
    count_file = os.environ.get("FAKE_AGY_MODELS_COUNT_FILE")
    if count_file:
        prev = int(open(count_file, encoding="utf-8").read().strip()) if os.path.exists(count_file) else 0
        open(count_file, "w", encoding="utf-8").write(str(prev + 1))
    if cur_mode == "auth_error":
        sys.stderr.write({error_text!r} + "\\n")
        sys.exit(1)
    sys.stdout.write({models_output!r})
    sys.exit(0)

_prompt = sys.stdin.read()
mode_file = os.environ.get("FAKE_AGY_MODE_FILE")
cur_mode = open(mode_file, encoding="utf-8").read().strip() if mode_file and os.path.exists(mode_file) else {mode!r}
is_stream = "stream-json" in args

if cur_mode == "auth_error":
    if is_stream:
        print(json.dumps({{"event": "result", "status": "ERROR", "error": {error_text!r}}}))
    else:
        print(json.dumps({{"status": "ERROR", "error": {error_text!r}}}))
    sys.exit(1)

if cur_mode == "quota_error":
    msg = "429 RESOURCE_EXHAUSTED: quota exceeded"
    if is_stream:
        print(json.dumps({{"event": "result", "status": "ERROR", "error": msg}}))
    else:
        print(json.dumps({{"status": "ERROR", "error": msg}}))
    sys.exit(1)

used_model = ""
if "--model" in args:
    used_model = args[args.index("--model") + 1]

if is_stream:
    print(
        json.dumps(
            {{
                "event": "result",
                "status": "OK",
                "response": f"worker-ok:{{used_model}}",
                "conversation_id": "conv-worker-1",
                "num_turns": 1,
                "usage": {{"input_tokens": 10, "output_tokens": 5}},
            }}
        )
    )
else:
    print(
        json.dumps(
            {{
                "status": "OK",
                "response": f"ok:{{used_model}}",
                "conversation_id": "conv-ask-1",
                "num_turns": 1,
                "usage": {{"input_tokens": 10, "output_tokens": 5}},
            }}
        )
    )
sys.exit(0)
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def _env(home: Path, cfg_dir: Path, agy_bin: Path | None = None, **extra: str) -> dict[str, str]:
    base = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
        "CC_AGY_STATE_DIR": str(home / "agy-state"),
        "CC_AGY_ALLOW_TEST_LICENSE_FETCH": "1",
        "CC_AGY_WORKER_SANDBOX": "off",
        "CC_AGY_MODEL_CATALOG": "0",
        **extra,
    }
    if agy_bin is not None:
        base["CC_AGY_CLI_BIN"] = str(agy_bin)
    return base


def _write_onboarding(cfg_dir: Path, project: str = "proj-alpha", tier: str = "gcp-ge-plus-tier") -> Path:
    onboarding = cfg_dir / "cache" / "onboarding.json"
    onboarding.parent.mkdir(parents=True, exist_ok=True)
    onboarding.write_text(
        json.dumps(
            {
                "enterpriseOnboardingComplete": True,
                "previousAuthMethod": "gcp",
                "cloudaicompanionProject": project,
                "tierId": tier,
            }
        ),
        encoding="utf-8",
    )
    return onboarding


def _write_valid_token(cfg_dir: Path) -> Path:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    token = cfg_dir / "antigravity-oauth-token"
    token.write_text(
        json.dumps(
            {
                "token": {"access_token": "acc-1", "refresh_token": "ref-1"},
                "auth_method": "gcp",
                "user_tier": "gcp-ge-plus-tier",
                "project_id": "proj-alpha",
            }
        ),
        encoding="utf-8",
    )
    return token


# --- Auth-failure marker writing & best-effort safety -----------------------------------------


def test_auth_failure_writes_marker_across_worker_ask_and_writer(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="auth_error")
    env = _env(tmp_path, cfg_dir, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="1")
    marker = agy_cli.auth_failure_marker_path(cfg_dir)

    with pytest.raises(RuntimeError, match="/cc-agy-plugin:login"):
        agy_cli.run_agy_cli_ask({"prompt": "hi", "model": "gemini-4-hydrogen", "env": env})
    assert marker.is_file()
    ts = float(marker.read_text(encoding="utf-8").strip())
    assert ts > 0

    marker.unlink()
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(RuntimeError, match="/cc-agy-plugin:login"):
        agy_cli.run_agy_cli_worker(
            {"prompt": "edit", "workspace_dir": str(ws), "model": "gemini-4-hydrogen", "env": env}
        )
    assert marker.is_file()

    marker.unlink()
    with pytest.raises(RuntimeError, match="/cc-agy-plugin:login"):
        agy_cli.run_agy_cli_structured_writer(
            prompt="gen",
            system_instruction="sys",
            model="gemini-4-hydrogen",
            thinking_level="LOW",
            json_schema={"type": "object"},
            env=env,
        )
    assert marker.is_file()


def test_non_auth_error_does_not_write_auth_failure_marker(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="quota_error")
    env = _env(tmp_path, cfg_dir, fake_agy)
    marker = agy_cli.auth_failure_marker_path(cfg_dir)

    with pytest.raises(RuntimeError, match="429"):
        agy_cli.run_agy_cli_ask({"prompt": "hi", "model": "gemini-4-hydrogen", "env": env})
    assert not marker.exists()


def test_record_auth_failure_marker_is_best_effort_and_never_raises(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    # Block cache directory creation by placing a regular file at `<cfg_dir>/cache`.
    (cfg_dir / "cache").write_text("not-a-dir", encoding="utf-8")

    assert agy_cli.record_auth_failure_marker(cfg_dir) is None
    assert agy_cli.clear_auth_failure_marker(cfg_dir) is False


# --- Offline inspector & /login honesty -------------------------------------------------------


def test_keychain_inspector_reports_not_signed_in_when_marker_newer_than_onboarding(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="ok")
    onboarding = _write_onboarding(cfg_dir)
    env = _env(tmp_path, cfg_dir, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="1")

    before = agy_cli.inspect_agy_cli_offline(env)
    assert before["ok"] is True
    assert before["auth_configured"] is True

    marker = agy_cli.record_auth_failure_marker(cfg_dir)
    assert marker is not None and marker.is_file()

    after = agy_cli.inspect_agy_cli_offline(env)
    assert after["ok"] is False
    assert after["auth_configured"] is False
    assert "rejected the saved sign-in" in (after["error"] or "")

    # If onboarding.json is later updated (newer than the marker), inspector trusts it again.
    newer_ns = marker.stat().st_mtime_ns + 5_000_000_000
    os.utime(onboarding, ns=(newer_ns, newer_ns))
    recovered = agy_cli.inspect_agy_cli_offline(env)
    assert recovered["ok"] is True
    assert recovered["auth_configured"] is True


def test_token_file_inspector_reports_not_signed_in_when_marker_newer_than_token(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="ok")
    token_file = _write_valid_token(cfg_dir)
    _write_onboarding(cfg_dir)
    env = _env(tmp_path, cfg_dir, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="0")

    assert agy_cli.inspect_agy_cli_offline(env)["auth_configured"] is True

    marker = agy_cli.record_auth_failure_marker(cfg_dir)
    assert marker is not None
    status = agy_cli.inspect_agy_cli_offline(env)
    assert status["auth_configured"] is False
    assert "rejected the saved sign-in" in (status["error"] or "")

    # Refreshing token_file makes it newer than the marker again.
    newer_ns = marker.stat().st_mtime_ns + 5_000_000_000
    os.utime(token_file, ns=(newer_ns, newer_ns))
    assert agy_cli.inspect_agy_cli_offline(env)["auth_configured"] is True


def test_login_skips_already_signed_in_when_marker_is_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="ok")
    _write_onboarding(cfg_dir)
    agy_cli.record_auth_failure_marker(cfg_dir)
    env = _env(tmp_path, cfg_dir, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="1")

    called: list[str] = []

    def _fake_browser(_agy: str, *, env: dict[str, str], timeout_seconds: float) -> tuple[bool, str]:
        called.append("browser")
        agy_login._mark_enterprise_onboarding_complete(cfg_dir, env=env)
        return True, ""

    monkeypatch.setattr(agy_login, "can_use_local_browser_loopback", lambda _env: True)
    monkeypatch.setattr(agy_login, "try_acp_business_browser_login", _fake_browser)

    res = agy_login.start_two_step_login(env)
    assert res == {"ok": True, "signed_in": True, "method": "browser"}
    assert called == ["browser"]
    assert not agy_cli.auth_failure_marker_path(cfg_dir).exists()
    assert agy_cli.inspect_agy_cli_offline(env)["auth_configured"] is True


# --- Clearing the marker on sign-in & successful calls ---------------------------------------


def test_marker_cleared_on_verified_finalize_and_not_on_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "antigravity-oauth-token").write_text(
        json.dumps({"token": {"access_token": "acc"}, "auth_method": "gcp"}), encoding="utf-8"
    )
    marker = agy_cli.record_auth_failure_marker(cfg_dir)
    assert marker is not None and marker.exists()

    # Unverified sign-in (no license) must NOT clear the marker.
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *_a, **_k: [])
    monkeypatch.setattr(agy_login, "self_assign_enterprise_license", lambda *_a, **_k: None)
    bad = agy_login.finalize_enterprise_onboarding(
        cfg_dir,
        tmp_path,
        had_standalone_token_before=True,
        env=_env(tmp_path, cfg_dir, CC_AGY_TOKEN_IN_KEYCHAIN="0"),
    )
    assert bad["verified"] is False
    assert marker.exists()

    # Verified sign-in clears the marker.
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *_a, **_k: [dict(_LICENSE_OK)])
    good = agy_login.finalize_enterprise_onboarding(
        cfg_dir,
        tmp_path,
        had_standalone_token_before=True,
        env=_env(tmp_path, cfg_dir, CC_AGY_TOKEN_IN_KEYCHAIN="0"),
    )
    assert good["verified"] is True
    assert not marker.exists()


def test_marker_cleared_after_any_successful_agy_call(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="ok")
    _write_onboarding(cfg_dir)
    env = _env(tmp_path, cfg_dir, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="1")
    marker = agy_cli.auth_failure_marker_path(cfg_dir)

    agy_cli.record_auth_failure_marker(cfg_dir)
    assert marker.exists()
    agy_cli.run_agy_cli_ask({"prompt": "hi", "model": "gemini-4-hydrogen", "env": env})
    assert not marker.exists()

    ws = tmp_path / "ws"
    ws.mkdir()
    agy_cli.record_auth_failure_marker(cfg_dir)
    assert marker.exists()
    agy_cli.run_agy_cli_worker(
        {"prompt": "hi", "workspace_dir": str(ws), "model": "gemini-4-hydrogen", "env": env}
    )
    assert not marker.exists()

    agy_cli.record_auth_failure_marker(cfg_dir)
    assert marker.exists()
    agy_cli.run_agy_cli_structured_writer(
        prompt="hi",
        system_instruction="sys",
        model="gemini-4-hydrogen",
        thinking_level="LOW",
        json_schema={"type": "object"},
        env=env,
    )
    assert not marker.exists()


# --- Bug (c): in-process and cross-process model cache invalidation ---------------------------


def _register_dummy_warm_acp_session(job_id: str) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name != "nt"),
    )
    sess = agy_acp.WarmAcpSession(
        job_id=job_id,
        agy_bin="/fake/agy",
        cwd="/tmp",
        proc=proc,
        spawn_cookie="test-cookie",
        pipe_inodes=set(),
        line_q=queue.Queue(),
        stderr_chunks=[],
    )
    agy_acp._WARM_ACP_POOL.put(sess)
    return proc


def test_signin_invalidates_model_caches_without_touching_acp_or_conversations(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="ok")
    env = _env(tmp_path, cfg_dir, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="1", CC_AGY_MODEL_CATALOG="1")
    cfg_key = str(cfg_dir)

    # Seed model caches alongside active ACP session and conversation ledger/disk state.
    agy_cli._MODEL_FALLBACK_CACHE[(str(fake_agy), cfg_key, "gemini-4-carbon", "high")] = "stale-model"
    agy_cli._CONVERSATION_USAGE_LEDGER[(cfg_key, "conv-active")] = (1, {"prompt_token_count": 50})
    catalog = agy_state.load_agy_model_catalog(str(fake_agy), env)
    assert catalog == ("gemini-4-hydrogen-medium",)
    agy_state.remember_conversation("conv-key", "conv-active", turns=1, usage={"prompt_token_count": 50}, env=env)
    assert (Path(env["CC_AGY_STATE_DIR"]) / "agy_model_catalog.json").is_file()
    assert (Path(env["CC_AGY_STATE_DIR"]) / "agy_conversations.json").is_file()
    proc = _register_dummy_warm_acp_session("parallel-job-1")
    try:
        assert "parallel-job-1" in agy_acp.active_warm_acp_jobs()

        agy_login._mark_enterprise_onboarding_complete(cfg_dir, env=env)

        assert (str(fake_agy), cfg_key, "gemini-4-carbon", "high") not in agy_cli._MODEL_FALLBACK_CACHE
        assert not agy_state._CATALOG_MEMORY
        assert not (Path(env["CC_AGY_STATE_DIR"]) / "agy_model_catalog.json").exists()
        # Active ACP sessions and conversation state must stay intact.
        assert (cfg_key, "conv-active") in agy_cli._CONVERSATION_USAGE_LEDGER
        assert (Path(env["CC_AGY_STATE_DIR"]) / "agy_conversations.json").is_file()
        assert "parallel-job-1" in agy_acp.active_warm_acp_jobs()
        assert proc.poll() is None
    finally:
        agy_acp.close_acp_job_session("parallel-job-1")
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)
        agy_cli._CONVERSATION_USAGE_LEDGER.pop((cfg_key, "conv-active"), None)


def test_cross_process_signin_invalidates_stale_model_caches_before_first_call(tmp_path: Path):
    """Simulate sign-in running in a separate process: onboarding.json mtime change invalidates model caches."""
    cfg_dir = tmp_path / "cfg"
    mode_file = tmp_path / "mode.txt"
    count_file = tmp_path / "models_count.txt"
    mode_file.write_text("auth_error", encoding="utf-8")

    fake_agy = _make_fake_agy(
        tmp_path / "bin" / "agy",
        mode="ok",
        models_output="gemini-4-hydrogen-medium  Gemini 4 Hydrogen\n",
    )
    onboarding = _write_onboarding(cfg_dir, project="proj-old")
    env = _env(
        tmp_path,
        cfg_dir,
        fake_agy,
        CC_AGY_TOKEN_IN_KEYCHAIN="1",
        CC_AGY_MODEL_CATALOG="1",
        CC_AGY_MODEL_CATALOG_TTL_SECONDS="60",
        FAKE_AGY_MODE_FILE=str(mode_file),
        FAKE_AGY_MODELS_COUNT_FILE=str(count_file),
    )

    # 1. Long-running MCP server observes initial auth state and caches a failed `agy models` lookup
    #    (None for 60s) and a stale model fallback.
    assert agy_cli.inspect_agy_cli_offline(env)["auth_configured"] is True
    assert agy_state.load_agy_model_catalog(str(fake_agy), env) is None
    assert count_file.read_text(encoding="utf-8") == "1"
    # Second call within 60s hits the cached None without invoking `agy models`.
    assert agy_state.load_agy_model_catalog(str(fake_agy), env) is None
    assert count_file.read_text(encoding="utf-8") == "1"

    agy_cli._MODEL_FALLBACK_CACHE[(str(fake_agy), str(cfg_dir), "gemini-4-hydrogen", "")] = "stale-old-model"

    # 2. Separate process completes sign-in: updates onboarding.json on disk (new mtime/inode)
    #    and tokens now work, WITHOUT calling any in-process invalidation helper here.
    mode_file.write_text("ok", encoding="utf-8")
    new_onboarding = cfg_dir / "cache" / "onboarding.json.tmp"
    new_onboarding.write_text(
        json.dumps(
            {
                "enterpriseOnboardingComplete": True,
                "previousAuthMethod": "gcp",
                "cloudaicompanionProject": "proj-new",
                "tierId": "gcp-ge-plus-tier",
            }
        ),
        encoding="utf-8",
    )
    newer_ns = onboarding.stat().st_mtime_ns + 5_000_000_000
    os.utime(new_onboarding, ns=(newer_ns, newer_ns))
    new_onboarding.replace(onboarding)

    # 3. Very first call in the MCP server after cross-process sign-in must invalidate stale model
    #    caches, re-query `agy models`, and succeed immediately.
    res = agy_cli.run_agy_cli_ask({"prompt": "hello", "model": "gemini-4-hydrogen", "env": env})
    assert res["text"] == "ok:gemini-4-hydrogen"
    assert int(count_file.read_text(encoding="utf-8")) == 2
    assert (str(fake_agy), str(cfg_dir), "gemini-4-hydrogen", "") not in agy_cli._MODEL_FALLBACK_CACHE

    # 4. Subsequent call with unchanged onboarding.json reuses the fresh catalog cache (no extra `agy models`).
    res2 = agy_cli.run_agy_cli_ask({"prompt": "again", "model": "gemini-4-hydrogen", "env": env})
    assert res2["text"] == "ok:gemini-4-hydrogen"
    assert int(count_file.read_text(encoding="utf-8")) == 2


def test_routine_token_refresh_and_auth_error_preserve_acp_and_conversation_ledger(tmp_path: Path):
    """Routine hourly token refresh or an auth failure must not kill parallel ACP jobs or drop usage ledgers."""
    cfg_dir = tmp_path / "cfg"
    cfg_key = str(cfg_dir)
    mode_file = tmp_path / "mode.txt"
    mode_file.write_text("ok", encoding="utf-8")
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="ok")
    token_file = _write_valid_token(cfg_dir)
    env = _env(
        tmp_path,
        cfg_dir,
        fake_agy,
        CC_AGY_TOKEN_IN_KEYCHAIN="0",
        CC_AGY_MODEL_CATALOG="1",
        FAKE_AGY_MODE_FILE=str(mode_file),
    )

    assert agy_cli.inspect_agy_cli_offline(env)["auth_configured"] is True
    agy_cli._MODEL_FALLBACK_CACHE[(str(fake_agy), cfg_key, "gemini-4-carbon", "high")] = "cached-fallback"
    agy_cli._CONVERSATION_USAGE_LEDGER[(cfg_key, "conv-inflight")] = (2, {"prompt_token_count": 120})
    agy_state.remember_conversation(
        "conv-inflight-key", "conv-inflight", turns=2, usage={"prompt_token_count": 120}, env=env
    )
    proc = _register_dummy_warm_acp_session("inflight-acp-job")
    try:
        # 1. Simulate agy's routine hourly token file rewrite on Linux.
        refreshed = token_file.with_suffix(".tmp")
        refreshed.write_text(
            json.dumps(
                {
                    "token": {"access_token": "acc-refreshed", "refresh_token": "ref-1"},
                    "auth_method": "gcp",
                    "user_tier": "gcp-ge-plus-tier",
                    "project_id": "proj-alpha",
                }
            ),
            encoding="utf-8",
        )
        newer_ns = token_file.stat().st_mtime_ns + 5_000_000_000
        os.utime(refreshed, ns=(newer_ns, newer_ns))
        refreshed.replace(token_file)

        # Trigger sync path on another call.
        assert agy_cli.inspect_agy_cli_offline(env)["auth_configured"] is True
        # Model fallback cache refreshed, while ACP session and conversation state survive.
        assert (str(fake_agy), cfg_key, "gemini-4-carbon", "high") not in agy_cli._MODEL_FALLBACK_CACHE
        assert "inflight-acp-job" in agy_acp.active_warm_acp_jobs()
        assert proc.poll() is None
        assert agy_cli._CONVERSATION_USAGE_LEDGER.get((cfg_key, "conv-inflight")) == (2, {"prompt_token_count": 120})
        assert (Path(env["CC_AGY_STATE_DIR"]) / "agy_conversations.json").is_file()

        # 2. Simulate an auth error from a sibling call; ACP session and conversation ledger still survive.
        mode_file.write_text("auth_error", encoding="utf-8")
        with pytest.raises(RuntimeError, match="/cc-agy-plugin:login"):
            agy_cli.run_agy_cli_ask({"prompt": "hi", "model": "gemini-4-hydrogen", "env": env})

        assert "inflight-acp-job" in agy_acp.active_warm_acp_jobs()
        assert proc.poll() is None
        assert agy_cli._CONVERSATION_USAGE_LEDGER.get((cfg_key, "conv-inflight")) == (2, {"prompt_token_count": 120})
        assert (Path(env["CC_AGY_STATE_DIR"]) / "agy_conversations.json").is_file()
    finally:
        agy_acp.close_acp_job_session("inflight-acp-job")
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)
        agy_cli._CONVERSATION_USAGE_LEDGER.pop((cfg_key, "conv-inflight"), None)


# --- Login text contract ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_path",
    [
        "commands/login.md",
        ".opencode/commands/login.md",
    ],
)
def test_login_commands_announce_browser_wait_before_start_without_em_dashes(rel_path: str):
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    assert EXPECTED_BROWSER_NOTICE in text
    assert text.index(EXPECTED_BROWSER_NOTICE) < text.index("--start --json")
    assert "\u2014" not in text


def test_quickstart_agent_text_announces_browser_wait_before_start():
    script = (REPO_ROOT / "scripts" / "quickstart.sh").read_text(encoding="utf-8")
    step4_start = script.index("4. Gemini Enterprise only")
    step5_start = script.index("5. Verify:", step4_start)
    step4 = script[step4_start:step5_start]

    assert EXPECTED_BROWSER_NOTICE in step4
    assert step4.index(EXPECTED_BROWSER_NOTICE) < step4.index("--start --json")
    assert "\u2014" not in step4
