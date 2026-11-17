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
    "Checking your Gemini Enterprise sign-in. On a local desktop, a browser window may open "
    "for Google sign-in (waits up to 90 seconds); over SSH or headless Linux, a sign-in link will appear next."
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


# --- Round-5 edge-case fixes ------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        "GET https://api.github.com/repos/org/repo returned HTTP 403 Forbidden",
        "npm ERR! 403 Forbidden - GET https://registry.npmjs.org/@scope/pkg",
        "ERROR: HTTP 403 Forbidden from https://pypi.org/simple/pkg/",
        "fatal: unable to access 'https://github.com/org/repo.git/': The requested URL returned error: 403",
        "HTTP 403 Forbidden: Request is prohibited by organization's policy (VPC Service Controls)",
        "HTTP 403 PermissionDenied: Caller does not have permission on project foo",
        "HTTP 403 accessNotConfigured: Cloud Code API has not been used in project",
        "Action unauthorized for protected branch",
    ],
)
def test_third_party_403_and_vpc_sc_iam_denials_are_not_agy_auth_errors(msg: str):
    assert agy_cli.is_agy_auth_error(msg) is False
    assert agy_cli.is_agy_oauth_credential_rejected(msg) is False


def test_oauth_rejection_classifier_is_strict_and_never_writes_marker_on_403(tmp_path: Path):
    # Strictly rejected/missing/expired Google OAuth credential errors.
    for oauth_err in (
        "HTTP 401 from cloudcode-pa.googleapis.com",
        "invalid_grant: token has been expired or revoked",
        "invalid_token: credentials are invalid",
        "Not logged in. Please sign in with Google.",
        "incomplete GCP onboarding",
        "UNAUTHENTICATED: missing required authentication credential",
        "Reauthentication required",
    ):
        assert agy_cli.is_agy_oauth_credential_rejected(oauth_err) is True, oauth_err

    # 403s and bare 401/unauthorized without OAuth context must never trip the failure marker.
    for non_marker_err in (
        "RPC failed: status = 403",
        "HTTP 403 Forbidden",
        "cloudcode-pa.googleapis.com returned 403",
        "HTTP 401 Unauthorized",
        "unauthorized",
    ):
        assert agy_cli.is_agy_oauth_credential_rejected(non_marker_err) is False, non_marker_err

    cfg_dir = tmp_path / "cfg"
    _write_valid_token(cfg_dir)
    _write_onboarding(cfg_dir)
    marker = agy_cli.auth_failure_marker_path(cfg_dir)

    for err_text, expect_remediation in (
        ("GET https://api.github.com/repos/a/b returned HTTP 403 Forbidden", False),
        ("HTTP 403 PermissionDenied: VPC Service Controls blocked request", False),
        ("RPC failed: status = 403", True),
    ):
        fake_agy = _make_fake_agy(tmp_path / f"bin-{abs(hash(err_text))}" / "agy", mode="auth_error", error_text=err_text)
        env = _env(tmp_path, cfg_dir, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="0")
        with pytest.raises(RuntimeError) as exc_info:
            agy_cli.run_agy_cli_ask({"prompt": "hi", "model": "gemini-4-hydrogen", "env": env})
        assert not marker.exists(), err_text
        assert ("/cc-agy-plugin:login" in str(exc_info.value)) is expect_remediation


def test_auto_mode_routes_to_gemini_enterprise_when_only_failure_marker_active(tmp_path: Path):
    from server.regions import resolve_auth_config

    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="ok")

    # 1. File-based valid token + active failure marker -> doctor sees ok=False, auto mode still routes to agy.
    cfg_file = tmp_path / "cfg-file"
    _write_valid_token(cfg_file)
    _write_onboarding(cfg_file)
    agy_cli.record_auth_failure_marker(cfg_file)
    env_file = _env(tmp_path, cfg_file, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="0")
    status_file = agy_cli.inspect_agy_cli_offline(env_file)
    assert status_file["ok"] is False
    assert status_file["auth_configured"] is False
    assert status_file["auth_failure_marker_active"] is True
    assert status_file["recoverable_auto_agy"] is True
    assert resolve_auth_config(env=env_file)["auth_mode"] == "gemini_enterprise"

    # 2. Keychain enterprise onboarding.json + active failure marker -> auto mode still routes to agy.
    cfg_kc = tmp_path / "cfg-kc"
    _write_onboarding(cfg_kc)
    agy_cli.record_auth_failure_marker(cfg_kc)
    env_kc = _env(tmp_path, cfg_kc, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="1")
    status_kc = agy_cli.inspect_agy_cli_offline(env_kc)
    assert status_kc["ok"] is False
    assert status_kc["auth_failure_marker_active"] is True
    assert status_kc["recoverable_auto_agy"] is True
    assert resolve_auth_config(env=env_kc)["auth_mode"] == "gemini_enterprise"

    # 3. Consumer-only onboarding + active marker must NOT be recoverable in auto mode.
    cfg_cons = tmp_path / "cfg-cons"
    (cfg_cons / "cache").mkdir(parents=True)
    (cfg_cons / "cache" / "onboarding.json").write_text(
        json.dumps({"consumerOnboardingComplete": True, "tierId": "free-tier"}), encoding="utf-8"
    )
    agy_cli.record_auth_failure_marker(cfg_cons)
    env_cons = _env(tmp_path, cfg_cons, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="1")
    assert agy_cli.inspect_agy_cli_offline(env_cons)["recoverable_auto_agy"] is False
    with pytest.raises(ValueError, match="Missing credentials"):
        resolve_auth_config(env=env_cons)

    # 4. Version mismatch + active marker must NOT be recoverable in auto mode.
    (fake_agy.parent / "agy.manifest.json").write_text(json.dumps({"version": "1.2.3"}), encoding="utf-8")
    env_ver = {**env_file, "CC_AGY_CLI_VERSION": "9.9.9"}
    assert agy_cli.inspect_agy_cli_offline(env_ver)["recoverable_auto_agy"] is False
    with pytest.raises(ValueError, match="Missing credentials"):
        resolve_auth_config(env=env_ver)


def test_unverified_finalize_preserves_working_token_and_keeps_marker_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="ok")
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *_a, **_k: [])
    monkeypatch.setattr(agy_login, "self_assign_enterprise_license", lambda *_a, **_k: None)

    # 1. Healthy existing CLI token (no failure marker) + newer unlicensed standalone (e.g. --force
    #    on a personal account) must preserve the working CLI token untouched.
    cfg_healthy = tmp_path / "cfg-healthy"
    home_healthy = tmp_path / "home-healthy"
    cli_token = _write_valid_token(cfg_healthy)
    _write_onboarding(cfg_healthy)
    orig_bytes = cli_token.read_bytes()
    orig_mtime_ns = cli_token.stat().st_mtime_ns

    standalone = home_healthy / ".gemini" / "jetski-standalone-oauth-token"
    standalone.parent.mkdir(parents=True, exist_ok=True)
    standalone.write_text(json.dumps({"token": {"access_token": "unlicensed"}, "auth_method": "gcp"}), encoding="utf-8")
    newer_ns = orig_mtime_ns + 5_000_000_000
    os.utime(standalone, ns=(newer_ns, newer_ns))

    env_healthy = _env(home_healthy, cfg_healthy, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="0")
    res_healthy = agy_login.finalize_enterprise_onboarding(
        cfg_healthy, home_healthy, had_standalone_token_before=False, env=env_healthy
    )
    assert res_healthy["verified"] is False
    assert cli_token.read_bytes() == orig_bytes
    assert cli_token.stat().st_mtime_ns == orig_mtime_ns
    assert agy_cli.inspect_agy_cli_offline(env_healthy)["auth_configured"] is True

    # 2. Failed heal on an already-on-disk revoked token (e.g. --project switch with no seat, no
    #    newer standalone) must NEVER bump cli_token_file's mtime past cc-agy-auth-failed.
    cfg_failed = tmp_path / "cfg-failed"
    home_failed = tmp_path / "home-failed"
    revoked_token = _write_valid_token(cfg_failed)
    _write_onboarding(cfg_failed)
    revoked_bytes = revoked_token.read_bytes()
    revoked_mtime_ns = revoked_token.stat().st_mtime_ns
    marker = agy_cli.record_auth_failure_marker(cfg_failed)
    assert marker is not None

    env_failed = _env(
        home_failed,
        cfg_failed,
        fake_agy,
        CC_AGY_TOKEN_IN_KEYCHAIN="0",
        CC_AGY_ENTERPRISE_PROJECT="proj-beta",
    )
    res_failed = agy_login.finalize_enterprise_onboarding(
        cfg_failed, home_failed, had_standalone_token_before=True, env=env_failed
    )
    assert res_failed["verified"] is False
    assert revoked_token.read_bytes() == revoked_bytes
    assert revoked_token.stat().st_mtime_ns == revoked_mtime_ns
    status_failed = agy_cli.inspect_agy_cli_offline(env_failed)
    assert status_failed["auth_configured"] is False
    assert status_failed["auth_failure_marker_active"] is True


def test_unverified_first_time_and_failed_standalone_token_is_preserved_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="ok")

    # 1. First-time SSH login hits transient lookup_failed: fresh standalone token is preserved
    #    as an unverified (license-stripped) CLI token so retry via heal_enterprise_token succeeds
    #    without repeating Google OAuth, while doctor still reports auth_configured=False until healed.
    home_1 = tmp_path / "home1"
    cfg_1 = home_1 / ".gemini" / "antigravity-cli"
    cfg_1.mkdir(parents=True)
    standalone_1 = home_1 / ".gemini" / "jetski-standalone-oauth-token"
    standalone_1.write_text(
        json.dumps(
            {
                "token": {"access_token": "fresh-ssh-tok", "refresh_token": "ref-ssh"},
                "auth_method": "gcp",
                "user_tier": "",
                "project_id": "",
            }
        ),
        encoding="utf-8",
    )

    def _raise_net(*_a: Any, **_k: Any) -> list[dict[str, str]]:
        raise agy_login.LicenseLookupError("couldn't reach the license service (temporary DNS error)")

    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", _raise_net)
    env_1 = _env(home_1, cfg_1, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="0")
    res_1 = agy_login.finalize_enterprise_onboarding(cfg_1, home_1, had_standalone_token_before=False, env=env_1)

    assert res_1["verified"] is False
    assert res_1["problem"] == "lookup_failed"
    assert not standalone_1.exists()
    cli_1 = cfg_1 / "antigravity-oauth-token"
    assert cli_1.is_file()
    saved_1 = json.loads(cli_1.read_text(encoding="utf-8"))
    assert saved_1["token"]["access_token"] == "fresh-ssh-tok"
    assert not saved_1.get("user_tier") and not saved_1.get("project_id")
    assert agy_cli.inspect_agy_cli_offline(env_1)["auth_configured"] is False

    # Network recovers -> heal_enterprise_token finishes without a second Google OAuth.
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *_a, **_k: [dict(_LICENSE_OK)])
    healed_1 = agy_login.heal_enterprise_token(env_1)
    assert healed_1["healed"] is True
    assert healed_1["project_id"] == "proj-alpha"
    assert agy_cli.inspect_agy_cli_offline(env_1)["auth_configured"] is True

    # 2. Prior rejected CLI token (active marker) + fresh standalone token with no_license:
    #    fresh token replaces the rejected token (stripped of license fields) so `--project` can heal it.
    home_2 = tmp_path / "home2"
    cfg_2 = home_2 / ".gemini" / "antigravity-cli"
    _write_valid_token(cfg_2)
    _write_onboarding(cfg_2)
    marker_2 = agy_cli.record_auth_failure_marker(cfg_2)
    assert marker_2 is not None
    standalone_2 = home_2 / ".gemini" / "jetski-standalone-oauth-token"
    standalone_2.parent.mkdir(parents=True, exist_ok=True)
    standalone_2.write_text(
        json.dumps({"token": {"access_token": "fresh-relogin-tok", "refresh_token": "ref-2"}, "auth_method": "gcp"}),
        encoding="utf-8",
    )
    newer_ns = marker_2.stat().st_mtime_ns + 5_000_000_000
    os.utime(standalone_2, ns=(newer_ns, newer_ns))

    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *_a, **_k: [])
    monkeypatch.setattr(agy_login, "self_assign_enterprise_license", lambda *_a, **_k: None)
    env_2 = _env(home_2, cfg_2, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="0")
    res_2 = agy_login.finalize_enterprise_onboarding(cfg_2, home_2, had_standalone_token_before=False, env=env_2)
    assert res_2["verified"] is False
    assert res_2["problem"] == "no_license"
    saved_2 = json.loads((cfg_2 / "antigravity-oauth-token").read_text(encoding="utf-8"))
    assert saved_2["token"]["access_token"] == "fresh-relogin-tok"
    assert agy_cli.inspect_agy_cli_offline(env_2)["auth_configured"] is False


def test_keychain_browser_login_preserves_new_fallback_token_and_drops_stale_ssh_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Case A (fable-3 P2-T3): Linux desktop where agy falls back to writing a license-less
    # antigravity-oauth-token during THIS login and license lookup fails/is disabled:
    # token_file must NOT be deleted and doctor must NOT lie signed_in=True.
    cfg_a = tmp_path / "cfg-a"
    cfg_a.mkdir()
    acp_writes_token = tmp_path / "bin-a" / "agy"
    acp_writes_token.parent.mkdir(parents=True)
    acp_writes_token.write_text(
        f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
        "if '--help' in sys.argv: print('Antigravity CLI --output-format --mode'); sys.exit(0)\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw)\n"
        "    if msg.get('method') == 'authenticate':\n"
        "        p = pathlib.Path(os.environ['CC_AGY_CLI_CONFIG_DIR']) / 'antigravity-oauth-token'\n"
        "        p.write_text(json.dumps({'token': {'access_token': 'fallback-tok'}, 'auth_method': 'gcp'}))\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': {}}), flush=True)\n",
        encoding="utf-8",
    )
    acp_writes_token.chmod(0o755)
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *_a, **_k: [])
    monkeypatch.setattr(agy_login, "self_assign_enterprise_license", lambda *_a, **_k: None)
    env_a = _env(tmp_path, cfg_a, acp_writes_token, CC_AGY_TOKEN_IN_KEYCHAIN="1")

    ok_a, reason_a = agy_login.try_acp_business_browser_login(str(acp_writes_token), env=env_a, timeout_seconds=10)
    assert ok_a is False
    assert "no Gemini Enterprise license" in reason_a
    assert (cfg_a / "antigravity-oauth-token").is_file()
    assert not (cfg_a / "cache" / "onboarding.json").exists()
    assert agy_cli.inspect_agy_cli_offline(env_a)["auth_configured"] is False

    # Case B (carbon-3 P2): macOS keychain where a valid pre-existing SSH antigravity-oauth-token
    # existed BEFORE login, and agy saves the new browser token to OS keychain (leaving the file
    # untouched): stale SSH token file must be unlinked so it cannot shadow the new keychain login.
    cfg_b = tmp_path / "cfg-b"
    stale_ssh_tok = _write_valid_token(cfg_b)
    (cfg_b / "settings.json").write_text(json.dumps({"gcp": {"project": "proj-beta"}}), encoding="utf-8")
    acp_keychain = tmp_path / "bin-b" / "agy"
    acp_keychain.parent.mkdir(parents=True)
    acp_keychain.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        "if '--help' in sys.argv: print('Antigravity CLI --output-format --mode'); sys.exit(0)\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw)\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': {}}), flush=True)\n",
        encoding="utf-8",
    )
    acp_keychain.chmod(0o755)
    env_b = _env(tmp_path, cfg_b, acp_keychain, CC_AGY_TOKEN_IN_KEYCHAIN="1")

    ok_b, reason_b = agy_login.try_acp_business_browser_login(str(acp_keychain), env=env_b, timeout_seconds=10)
    assert (ok_b, reason_b) == (True, "")
    assert not stale_ssh_tok.exists()
    onboarding_b = json.loads((cfg_b / "cache" / "onboarding.json").read_text(encoding="utf-8"))
    assert onboarding_b["cloudaicompanionProject"] == "proj-beta"
    assert agy_cli.inspect_agy_cli_offline(env_b)["auth_configured"] is True


def test_generalized_third_party_http_url_errors_are_excluded():
    for third_party_err in (
        "GET https://artifactory.corp.internal/api/npm/pkg returned HTTP 401 Unauthorized",
        "POST https://api.stripe.com/v1/charges failed: HTTP 401 authentication failed",
        "GET https://evilgoogle.com/v1/token returned HTTP 401 Unauthorized",
        "HTTP 403 Forbidden from https://nexus.internal.example:8443/repository/pypi",
    ):
        assert agy_cli.is_agy_auth_error(third_party_err) is False, third_party_err
        assert agy_cli.is_agy_oauth_credential_rejected(third_party_err) is False, third_party_err

    assert agy_cli.is_agy_auth_error("GET https://accounts.google.com/o/oauth2/token returned HTTP 401 Unauthorized") is True
    assert agy_cli.is_agy_oauth_credential_rejected("GET https://cloudcode-pa.googleapis.com/v1:generateContent returned HTTP 401") is True


def test_remediations_and_host_docs_include_host_neutral_terminal_signin(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="ok")

    # Missing token remediation keeps Claude Code command and adds host-neutral terminal script.
    env = _env(tmp_path, cfg_dir, fake_agy, CC_AGY_TOKEN_IN_KEYCHAIN="0")
    s_missing = agy_cli.inspect_agy_cli_offline(env)
    assert "Run `/cc-agy-plugin:login` in Claude Code" in (s_missing["remediation"] or "")
    assert "bash ~/.local/share/cc-agy-plugin/current/scripts/agy_signin.sh" in (s_missing["remediation"] or "")

    # Active failure marker remediation also includes host-neutral terminal fallback.
    _write_valid_token(cfg_dir)
    agy_cli.record_auth_failure_marker(cfg_dir)
    s_marker = agy_cli.inspect_agy_cli_offline(env)
    assert "Run `/cc-agy-plugin:login` in Claude Code" in (s_marker["remediation"] or "")
    assert "bash ~/.local/share/cc-agy-plugin/current/scripts/agy_signin.sh" in (s_marker["remediation"] or "")

    # _format_auth_remediation also includes host-neutral terminal fallback.
    formatted = agy_cli._format_auth_remediation("HTTP 401 from cloudcode-pa.googleapis.com", str(fake_agy), env=env)
    assert "Run `/cc-agy-plugin:login` in Claude Code" in formatted
    assert "bash ~/.local/share/cc-agy-plugin/current/scripts/agy_signin.sh" in formatted

    # Cursor rule and Codex skill point to host-neutral sign-in helpers without em-dashes.
    for rel in ("cursor/rules/cc-agy-coexistence.mdc", "plugins/cc-agy-plugin/skills/vertex/SKILL.md"):
        doc_text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "agy_signin.sh" in doc_text, rel
        assert "agy_login.py --start --json" in doc_text, rel
        assert "\u2014" not in doc_text, rel


def test_keychain_browser_login_ignores_stale_standalone_and_clears_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg_dir = tmp_path / "cfg"
    _write_onboarding(cfg_dir)
    stale_cli_tok = _write_valid_token(cfg_dir)
    marker = agy_cli.record_auth_failure_marker(cfg_dir)
    assert marker is not None

    # Stale unlicensed standalone token sitting in ~/.gemini/ must not poison keychain browser login.
    standalone = tmp_path / ".gemini" / "jetski-standalone-oauth-token"
    standalone.parent.mkdir(parents=True, exist_ok=True)
    standalone.write_text(json.dumps({"token": {"access_token": "stale"}, "auth_method": "gcp"}), encoding="utf-8")

    acp_agy = tmp_path / "bin" / "agy"
    acp_agy.parent.mkdir(parents=True, exist_ok=True)
    acp_agy.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        "if '--help' in sys.argv: print('Antigravity CLI --output-format --mode'); sys.exit(0)\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw)\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': {}}), flush=True)\n",
        encoding="utf-8",
    )
    acp_agy.chmod(0o755)
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *_a, **_k: [])
    env = _env(tmp_path, cfg_dir, acp_agy, CC_AGY_TOKEN_IN_KEYCHAIN="1")

    ok, reason = agy_login.try_acp_business_browser_login(str(acp_agy), env=env, timeout_seconds=10)

    assert (ok, reason) == (True, "")
    assert not stale_cli_tok.exists()
    assert not marker.exists()
    status = agy_cli.inspect_agy_cli_offline(env)
    assert status["ok"] is True
    assert status["auth_configured"] is True


def test_explicit_project_switch_falls_through_when_token_expired_or_marker_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg_dir = tmp_path / "cfg"
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", mode="ok")
    _write_valid_token(cfg_dir)
    _write_onboarding(cfg_dir, project="proj-alpha")

    browser_calls: list[str] = []

    def _fake_browser(_agy: str, *, env: dict[str, str], timeout_seconds: float) -> tuple[bool, str]:
        del _agy, env, timeout_seconds
        browser_calls.append("browser")
        return True, ""

    monkeypatch.setattr(agy_login, "can_use_local_browser_loopback", lambda _env: True)
    monkeypatch.setattr(agy_login, "try_acp_business_browser_login", _fake_browser)

    # 1. Active failure marker + explicit project -> falls through to fresh sign-in.
    agy_cli.record_auth_failure_marker(cfg_dir)
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *_a, **_k: [dict(_LICENSE_OK)])
    monkeypatch.setattr(agy_login, "self_assign_enterprise_license", lambda *_a, **_k: None)
    env = _env(
        tmp_path,
        cfg_dir,
        fake_agy,
        CC_AGY_TOKEN_IN_KEYCHAIN="0",
        CC_AGY_ENTERPRISE_PROJECT="proj-beta",
    )
    res_marker = agy_login.start_two_step_login(env)
    assert res_marker == {"ok": True, "signed_in": True, "method": "browser"}
    assert browser_calls == ["browser"]

    # 2. No failure marker, but token rejected with HTTP 401 -> falls through to fresh sign-in
    #    in both start_two_step_login and run_interactive_or_url_login.
    agy_cli.clear_auth_failure_marker(cfg_dir)

    def _raise_401(*_a: Any, **_k: Any) -> list[dict[str, str]]:
        raise agy_login.LicenseLookupError("the license service rejected the saved token (HTTP 401); it has probably expired")

    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", _raise_401)
    browser_calls.clear()
    res_401 = agy_login.start_two_step_login(env)
    assert res_401 == {"ok": True, "signed_in": True, "method": "browser"}
    assert browser_calls == ["browser"]

    browser_calls.clear()
    rc_401 = agy_login.run_interactive_or_url_login(env)
    assert rc_401 == 0
    assert browser_calls == ["browser"]

    # 3. Live usable token with no seat on proj-beta -> run_interactive_or_url_login fails early (rc=1).
    monkeypatch.setattr(agy_login, "fetch_enterprise_licenses", lambda *_a, **_k: [dict(_LICENSE_OK)])
    browser_calls.clear()
    rc_no_seat = agy_login.run_interactive_or_url_login(env)
    assert rc_no_seat == 1
    assert browser_calls == []
