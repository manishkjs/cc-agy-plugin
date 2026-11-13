"""Tests for worker sandboxing, attempt accounting, fail-closed seats, and policy hardening."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap
from typing import Any, Mapping

import pytest

from server import agy_cli, bifrost_policy, worker_sandbox
from server.agy_cli import (
    EnterpriseSeatError,
    _BoundedLedger,
    build_agy_subprocess_env,
    inspect_agy_cli_offline,
    run_agy_cli_ask,
    run_agy_cli_worker,
)
from server.bifrost_policy import cel_selects_policy_routed, enforce_agy_model_policy, inspect_bifrost_policy
from server.install import _merge_opencode_env, build_enterprise_env
from server.jobs import JobManager
from server.receipts import ReceiptStore
from server.worker_sandbox import (
    WorkerSandboxError,
    build_bwrap_prefix,
    build_seatbelt_profile,
    git_guard_environment,
    inspect_worker_sandbox,
    plan_worker_sandbox,
    resolve_sandbox_mode,
)

POLICY_URL = "https://bifrost.internal.example.com"
# A real path outside /tmp (which the sandbox makes private), used to prove writes are blocked.
_OUTSIDE_PROBE = Path(__file__).resolve().parent / ".sandbox-write-probe"


def _bwrap_usable() -> bool:
    worker_sandbox.clear_probe_cache()
    backend, _, _ = worker_sandbox.detect_sandbox_backend({})
    return backend == worker_sandbox.BACKEND_BWRAP


requires_bwrap = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not _bwrap_usable(), reason="bubblewrap is not usable here"
)


@pytest.fixture(autouse=True)
def _reset_caches():
    worker_sandbox.clear_probe_cache()
    bifrost_policy.clear_policy_cache()
    agy_cli._MODEL_FALLBACK_CACHE.clear()
    yield
    worker_sandbox.clear_probe_cache()
    bifrost_policy.clear_policy_cache()


def _make_fake_agy(bin_path: Path, body: str) -> Path:
    script = (
        "#!/usr/bin/env python3\nimport json\nimport os\nimport sys\nimport time\n\n"
        "argv = sys.argv[1:]\n"
        'if "--help" in argv or "-h" in argv:\n'
        '    sys.stderr.write("Usage: agy [options]\\n  --output-format string\\n  --mode string\\n")\n'
        "    raise SystemExit(0)\n" + textwrap.dedent(body)
    )
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_text(script, encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


def _home_with_secrets(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE-KEY-BYTES", encoding="utf-8")
    (home / ".netrc").write_text("machine example.com password hunter2", encoding="utf-8")
    (home / ".cache").mkdir()
    (home / "notes.txt").write_text("outside the repo", encoding="utf-8")
    return home


def _workspace(tmp_path: Path, job: str = "job-1") -> Path:
    workspace = tmp_path / "data" / "worktrees" / job
    workspace.mkdir(parents=True)
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return workspace


# ---------------------------------------------------------------------------
# P0. OS sandbox for write workers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", "auto"), ("auto", "auto"), ("ON", "auto"), ("required", "required"), ("strict", "required"),
     ("off", "off"), ("0", "off")],
)
def test_sandbox_mode_values(raw: str, expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CC_AGY_WORKER_SANDBOX", raising=False)
    assert resolve_sandbox_mode({"CC_AGY_WORKER_SANDBOX": raw}) == expected


def test_unknown_sandbox_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="CC_AGY_WORKER_SANDBOX"):
        resolve_sandbox_mode({"CC_AGY_WORKER_SANDBOX": "maybe"})


def test_without_a_backend_auto_drops_permission_skipping_and_required_refuses(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    missing = {"CC_AGY_BWRAP_BIN": str(tmp_path / "no-such-bwrap"), "CC_AGY_WORKER_SANDBOX": "auto"}

    auto_plan = plan_worker_sandbox(workspace_dir=workspace, config_dir=tmp_path / "cfg", env=missing)
    off_plan = plan_worker_sandbox(
        workspace_dir=workspace, config_dir=tmp_path / "cfg", env={**missing, "CC_AGY_WORKER_SANDBOX": "off"}
    )

    assert auto_plan.backend is None and auto_plan.skip_permissions is False
    assert "denied_actions" in str(auto_plan.warning) and auto_plan.wrap(["agy"]) == ["agy"]
    assert off_plan.skip_permissions is True and "off" in str(off_plan.warning)
    if sys.platform.startswith("linux"):
        with pytest.raises(WorkerSandboxError, match="required"):
            plan_worker_sandbox(
                workspace_dir=workspace, config_dir=tmp_path / "cfg", env={**missing, "CC_AGY_WORKER_SANDBOX": "required"}
            )
        assert inspect_worker_sandbox({**missing, "CC_AGY_WORKER_SANDBOX": "required"})["error"]


def test_bwrap_arguments_hide_secrets_and_bind_only_the_repo_copy(tmp_path: Path) -> None:
    home = _home_with_secrets(tmp_path)
    workspace = _workspace(tmp_path)
    config_dir = home / ".gemini" / "antigravity-cli"
    config_dir.mkdir(parents=True)
    adc = home / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text("{}", encoding="utf-8")

    args = build_bwrap_prefix("bwrap", workspace=workspace, config_dir=config_dir, home=home, adc_path=adc)
    joined = " ".join(args)

    assert args[1:4] == ["--ro-bind", "/", "/"] and args[-1] == "--"
    assert f"--tmpfs {home / '.ssh'}" in joined and f"--tmpfs {home / '.cache'}" in joined
    assert f"--tmpfs {home / '.config' / 'gcloud'}" in joined and f"--ro-bind {adc} {adc}" in joined
    assert f"--dev-bind /dev/null {home / '.netrc'}" in joined
    assert f"--tmpfs {tmp_path / 'data'}" in joined
    assert joined.index(f"--tmpfs {tmp_path / 'data'}") < joined.index(f"--bind {workspace} {workspace}")
    assert f"--bind {config_dir} {config_dir}" in joined and "--unshare-pid" in args and "--die-with-parent" in args


def test_seatbelt_profile_denies_writes_and_secret_reads(tmp_path: Path) -> None:
    home = Path("/Users/dev")
    workspace = Path("/Users/dev/.cache/cc-agy-plugin/abc/worktrees/job-1")

    profile = build_seatbelt_profile(workspace=workspace, config_dir=home / ".gemini/antigravity-cli", home=home, adc_path=None)

    assert profile.startswith("(version 1)\n(allow default)\n(deny file-write*)")
    assert f'(subpath "{workspace}")' in profile and '(subpath "/Users/dev/.ssh")' in profile
    assert '(literal "/Users/dev/.netrc")' in profile
    assert profile.rstrip().endswith(f'(allow file-read* (subpath "{workspace}"))')


@requires_bwrap
def test_real_bwrap_blocks_writes_and_secret_reads_outside_the_repo_copy(tmp_path: Path) -> None:
    home = _home_with_secrets(tmp_path)
    workspace = _workspace(tmp_path)
    _workspace(tmp_path, "job-2")
    config_dir = tmp_path / "agy-cfg"
    plan = plan_worker_sandbox(
        workspace_dir=workspace, config_dir=config_dir, env={"HOME": str(home), "CC_AGY_WORKER_SANDBOX": "auto"}
    )
    probe = textwrap.dedent(
        f"""
        import os
        def attempt(fn):
            try:
                fn()
                return "ok"
            except OSError as exc:
                return type(exc).__name__
        print(attempt(lambda: open("app.py", "w").write("VALUE = 2\\n")))
        print(attempt(lambda: open({str(_OUTSIDE_PROBE)!r}, "w").write("x")))
        print(attempt(lambda: open({str(config_dir / 'state.log')!r}, "w").write("x")))
        print(os.path.exists({str(home / '.ssh' / 'id_ed25519')!r}))
        print(open({str(home / '.netrc')!r}).read() == "")
        print(os.path.exists({str(tmp_path / 'data' / 'worktrees' / 'job-2' / 'app.py')!r}))
        print(attempt(lambda: open("/tmp/scratch", "w").write("x")))
        """
    )

    try:
        proc = subprocess.run(plan.wrap([sys.executable, "-c", probe]), cwd=workspace, capture_output=True, text=True,
                              timeout=30, check=False)
        probe_escaped = _OUTSIDE_PROBE.exists()
    finally:
        _OUTSIDE_PROBE.unlink(missing_ok=True)

    assert proc.returncode == 0, proc.stderr
    assert plan.backend == "bwrap" and plan.skip_permissions is True
    assert proc.stdout.split() == ["ok", "OSError", "ok", "False", "True", "False", "ok"]
    assert (workspace / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert probe_escaped is False


@requires_bwrap
def test_write_worker_runs_inside_bwrap_with_permission_skipping(tmp_path: Path) -> None:
    home = _home_with_secrets(tmp_path)
    workspace = _workspace(tmp_path)
    config_dir = tmp_path / "agy-cfg"
    fake_agy = _make_fake_agy(
        tmp_path / "agy-bin" / "agy",
        f"""
        prompt = sys.stdin.read()
        open("app.py", "w").write("VALUE = 2\\n")
        try:
            open({str(_OUTSIDE_PROBE)!r}, "w").write("escaped")
            outside = "written"
        except OSError:
            outside = "blocked"
        with open({str(config_dir / 'argv.json')!r}, "w") as fh:
            json.dump({{"argv": argv, "outside": outside}}, fh)
        print(json.dumps({{"event": "result", "result": {{"status": "SUCCESS", "conversation_id": "c1",
              "num_turns": 1, "response": "done", "usage": {{"input_tokens": 10, "output_tokens": 2}}}}}}), flush=True)
        """,
    )

    try:
        out = run_agy_cli_worker(
            {
                "workspace_dir": str(workspace),
                "prompt": "Bump VALUE",
                "env": {"HOME": str(home), "PATH": os.defpath, "CC_AGY_CLI_BIN": str(fake_agy),
                        "CC_AGY_CLI_CONFIG_DIR": str(config_dir), "CC_AGY_WORKER_SANDBOX": "auto"},
            }
        )
        probe_escaped = _OUTSIDE_PROBE.exists()
    finally:
        _OUTSIDE_PROBE.unlink(missing_ok=True)
    record = json.loads((config_dir / "argv.json").read_text(encoding="utf-8"))

    assert out["worker_sandbox"] == {"mode": "auto", "backend": "bwrap", "skip_permissions": True}
    assert out["containment_warning"] is None
    assert "--dangerously-skip-permissions" in record["argv"] and record["outside"] == "blocked"
    assert (workspace / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert probe_escaped is False


def test_write_worker_without_a_sandbox_does_not_skip_permissions(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    config_dir = tmp_path / "agy-cfg"
    fake_agy = _make_fake_agy(
        config_dir / "bin" / "agy",
        f"""
        _ = sys.stdin.read()
        open({str(config_dir / 'argv.json')!r}, "w").write(json.dumps(argv))
        print(json.dumps({{"event": "result", "result": {{"status": "SUCCESS", "num_turns": 1, "response": "ok"}}}}))
        """,
    )

    out = run_agy_cli_worker(
        {
            "workspace_dir": str(workspace),
            "prompt": "Bump VALUE",
            "env": {"PATH": os.defpath, "CC_AGY_CLI_BIN": str(fake_agy), "CC_AGY_CLI_CONFIG_DIR": str(config_dir),
                    "CC_AGY_BWRAP_BIN": str(tmp_path / "no-such-bwrap"), "CC_AGY_WORKER_SANDBOX": "auto"},
        }
    )

    assert "--dangerously-skip-permissions" not in json.loads((config_dir / "argv.json").read_text(encoding="utf-8"))
    assert out["worker_sandbox"]["backend"] is None and "denied_actions" in out["containment_warning"]


def test_subprocess_env_drops_credential_agents_and_blocks_remote_git(tmp_path: Path) -> None:
    env = build_agy_subprocess_env(
        {
            "PATH": os.defpath,
            "HOME": str(tmp_path),
            "SSH_AUTH_SOCK": "/run/agent.sock",
            "GPG_AGENT_INFO": "/run/gpg",
            "GIT_ASKPASS": "/usr/bin/askpass",
            "SSH_ASKPASS": "/usr/bin/ssh-askpass",
            "GIT_SSH_COMMAND": "ssh -i key",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "credential.helper",
            "GIT_CONFIG_VALUE_0": "store",
        }
    )

    for removed in ("SSH_AUTH_SOCK", "GPG_AGENT_INFO", "GIT_ASKPASS", "SSH_ASKPASS", "GIT_SSH_COMMAND"):
        assert removed not in env
    assert env["GIT_CONFIG_KEY_0"] == "credential.helper" and env["GIT_CONFIG_VALUE_0"] == ""
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert int(env["GIT_CONFIG_COUNT"]) == len([k for k in env if k.startswith("GIT_CONFIG_KEY_")])


def test_git_guard_blocks_push_and_network_but_keeps_local_work(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    base_env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1"}
    guarded = {**base_env, **git_guard_environment()}

    def git(*args: str, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-c", "user.email=t@e.st", "-c", "user.name=T", *args], cwd=repo, env=dict(env),
                              capture_output=True, text=True, check=False)

    assert git("init", "-q", "-b", "main", env=base_env).returncode == 0
    (repo / "a.txt").write_text("1\n", encoding="utf-8")
    assert git("add", "a.txt", env=guarded).returncode == 0
    assert git("commit", "-qm", "local commit", env=guarded).returncode == 0
    assert git("remote", "add", "origin", str(remote), env=guarded).returncode == 0

    blocked_push = git("push", "origin", "main", env=guarded)
    blocked_https = git("ls-remote", "https://example.com/repo.git", env=guarded)
    allowed_push = git("push", "origin", "main", env=base_env)

    assert blocked_push.returncode != 0 and "transport 'cc-agy-push-blocked' not allowed" in blocked_push.stderr
    assert blocked_https.returncode != 0
    assert allowed_push.returncode == 0


@requires_bwrap
def test_doctor_reports_the_active_sandbox() -> None:
    report = inspect_worker_sandbox({"CC_AGY_WORKER_SANDBOX": "auto"})
    assert report["mode"] == "auto" and report["backend"] == "bwrap" and report["active"] is True
    assert inspect_worker_sandbox({"CC_AGY_WORKER_SANDBOX": "off"})["active"] is False


# ---------------------------------------------------------------------------
# P1-a / P1-b. Failed attempts count, and one deadline covers the job
# ---------------------------------------------------------------------------

_REROUTE_AGY_BODY = """
    prompt = sys.stdin.read()
    model = argv[argv.index("--model") + 1] if "--model" in argv else ""
    if model == "gemini-flash-missing":
        time.sleep(float(os.environ.get("FAKE_FIRST_SLEEP", "0")))
        print(json.dumps({"event": "step_update", "step_update": {"step_index": 1,
              "tool_info": {"name": "view_file", "parameters": {"path": "a.py"}}}}), flush=True)
        print(json.dumps({"event": "result", "result": {"status": "ERROR",
              "error": "model not recognized as a known model. Available models:\\n  - gemini-3.8-flash-low",
              "usage": {"input_tokens": 300, "output_tokens": 0}}}), flush=True)
        raise SystemExit(1)
    print(json.dumps({"event": "result", "result": {"status": "SUCCESS", "num_turns": 1, "response": "done",
          "usage": {"input_tokens": 100, "output_tokens": 20}}}), flush=True)
"""


def _worker_request(tmp_path: Path, **env_extra: str) -> dict[str, Any]:
    config_dir = tmp_path / "agy-cfg"
    fake_agy = _make_fake_agy(config_dir / "bin" / "agy", _REROUTE_AGY_BODY)
    return {
        "workspace_dir": str(_workspace(tmp_path)),
        "prompt": "Do the task",
        "model": "gemini-flash-missing",
        "env": {"PATH": os.defpath, "CC_AGY_CLI_BIN": str(fake_agy), "CC_AGY_CLI_CONFIG_DIR": str(config_dir),
                **env_extra},
    }


def test_tokens_and_tool_calls_from_a_rerouted_attempt_are_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agy_cli, "_is_read_only_worker_request", lambda request: True)

    out = run_agy_cli_worker(_worker_request(tmp_path))

    assert out["usage"]["prompt_token_count"] == 400
    assert out["usage"]["candidates_token_count"] == 20
    assert out["tool_calls"] == [{"name": "view_file", "args": {"path": "a.py"}}]
    assert "gemini-3.8-flash-low" in str(out["reroute_note"])


def test_retries_share_one_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agy_cli, "_is_read_only_worker_request", lambda request: True)
    request = _worker_request(tmp_path, FAKE_FIRST_SLEEP="2.4")
    request["timeout_seconds"] = 3

    with pytest.raises(TimeoutError, match="whole 3s timeout") as excinfo:
        run_agy_cli_worker(request)

    assert excinfo.value.partial_usage["prompt_token_count"] == 300  # type: ignore[attr-defined]


def test_ask_counts_tokens_from_a_rerouted_attempt(tmp_path: Path) -> None:
    fake_agy = _make_fake_agy(
        tmp_path / "bin" / "agy",
        """
        _ = sys.stdin.read()
        model = argv[argv.index("--model") + 1]
        if model == "gemini-flash-missing":
            print(json.dumps({"status": "ERROR", "error": "not recognized as a known model. Available models:\\n"
                              "  - gemini-3.8-flash-low", "usage": {"input_tokens": 50}}))
            raise SystemExit(1)
        print(json.dumps({"status": "SUCCESS", "response": "ok", "usage": {"input_tokens": 70, "output_tokens": 5}}))
        """,
    )

    out = run_agy_cli_ask({"model": "gemini-flash-missing", "prompt": "hi",
                           "env": {"PATH": os.defpath, "CC_AGY_CLI_BIN": str(fake_agy),
                                   "CC_AGY_CLI_CONFIG_DIR": str(tmp_path / "cfg")}})

    assert out["usage"]["prompt_token_count"] == 120 and out["usage"]["candidates_token_count"] == 5


# ---------------------------------------------------------------------------
# P1-c. A requested but invalid ADC seat stops at 0 tokens
# ---------------------------------------------------------------------------


def _service_account_home(tmp_path: Path) -> Path:
    home = tmp_path / "seat-home"
    adc = home / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text(json.dumps({"type": "service_account", "quota_project_id": "proj"}), encoding="utf-8")
    cfg = home / ".gemini" / "antigravity-cli"
    (cfg / "cache").mkdir(parents=True)
    (cfg / "cache" / "onboarding.json").write_text(json.dumps({"onboardingComplete": True}), encoding="utf-8")
    return home


def test_invalid_requested_seat_fails_closed_before_any_agy_call(tmp_path: Path) -> None:
    home = _service_account_home(tmp_path)
    calls = tmp_path / "calls.log"
    fake_agy = _make_fake_agy(tmp_path / "bin" / "agy", f"open({str(calls)!r}, 'a').write('called')\n")
    env = {"HOME": str(home), "PATH": os.defpath, "CC_AGY_CLI_BIN": str(fake_agy), "CC_AGY_ENTERPRISE_ADC": "true"}

    with pytest.raises(EnterpriseSeatError, match="Nothing was sent"):
        run_agy_cli_worker({"workspace_dir": str(_workspace(tmp_path)), "prompt": "x", "env": env})
    with pytest.raises(EnterpriseSeatError, match="service_account"):
        run_agy_cli_ask({"prompt": "x", "env": env})
    info = inspect_agy_cli_offline(env)

    assert not calls.exists()
    assert info["ok"] is False and "service_account" in info["error"]
    assert info["auth_configured"] is True  # the stale onboarding cache no longer hides the problem


# ---------------------------------------------------------------------------
# P1-d. Reinstalls drop stale enterprise settings
# ---------------------------------------------------------------------------


def test_opencode_reinstall_drops_stale_enterprise_and_sandbox_keys() -> None:
    existing = {"CC_AGY_ENTERPRISE_PROJECT": "old-project", "CC_AGY_WORKER_SANDBOX": "off",
                "CC_AGY_BIFROST_POLICY_URL": "https://old.example.com", "MY_OWN_VAR": "keep"}

    merged = _merge_opencode_env(existing, {"CC_AGY_AUTH_MODE": "gemini_enterprise"})

    assert merged == {"MY_OWN_VAR": "keep", "CC_AGY_AUTH_MODE": "gemini_enterprise"}
    assert build_enterprise_env(worker_sandbox="strict") == {"CC_AGY_WORKER_SANDBOX": "required"}
    with pytest.raises(ValueError, match="--worker-sandbox"):
        build_enterprise_env(worker_sandbox="sometimes")


def test_unsubstituted_user_config_placeholders_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    from server.regions import sanitize_placeholder_env

    monkeypatch.setenv("CC_AGY_USERCFG_WORKER_SANDBOX", "${user_config.worker_sandbox}")
    monkeypatch.setenv("CC_AGY_USERCFG_BIFROST_POLICY_URL", "${user_config.bifrost_policy_url}")

    sanitize_placeholder_env()

    assert "CC_AGY_USERCFG_WORKER_SANDBOX" not in os.environ
    assert "CC_AGY_USERCFG_BIFROST_POLICY_URL" not in os.environ


# ---------------------------------------------------------------------------
# P1-e. Bifrost policy: CEL parsing, aliases, visible fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "selects"),
    [
        ('model == "policy-routed" || model == "agy-cli/policy-routed"', True),
        ('"policy-routed" == model', True),
        ('model in ["x", "agy-cli/policy-routed"]', True),
        ('team == "a" || (model == "policy-routed")', True),
        ('model != "policy-routed"', False),
        ('!(model == "policy-routed")', False),
        ('model == "policy-routed" && team == "a"', False),
        ('model in ["policy-routed-2"]', False),
    ],
)
def test_cel_parsing_accepts_only_positive_matches(expression: str, selects: bool) -> None:
    assert cel_selects_policy_routed(expression) is selects


class _Gateway:
    def __init__(self, *, rules_status: int = 200, rule_model: str = "agy-frontier") -> None:
        self.rules_status = rules_status
        self.rule_model = rule_model

    def __call__(self, url: str, headers: Mapping[str, str], timeout_seconds: float) -> tuple[int, bytes]:
        if url.endswith("/v1/models"):
            models = ["agy-cli/policy-routed", "agy-cli/agy-frontier", "agy-cli/gemini-3.8-flash-low"]
            return 200, json.dumps({"data": [{"id": m} for m in models]}).encode()
        if url.endswith("/api/routing/rules"):
            rules = [
                {"priority": 1, "cel_expression": 'model != "policy-routed"', "targets": [{"model": "wrong-model"}]},
                {"priority": 5, "cel_expression": 'model == "policy-routed"',
                 "targets": [{"model": f"agy-cli/{self.rule_model}"}]},
            ]
            return self.rules_status, json.dumps({"rules": rules}).encode() if self.rules_status == 200 else b""
        if url.endswith("/api/providers/agy-cli"):
            return 200, json.dumps({"keys": [{"aliases": {"agy-frontier": "agy-cli/gemini-4-argon-high"}}]}).encode()
        return 404, b""


def _rules_env(**extra: str) -> dict[str, str]:
    return {"CC_AGY_BIFROST_POLICY_URL": POLICY_URL, "CC_AGY_BIFROST_AGY_VIRTUAL_KEY": "vk",
            "CC_AGY_BIFROST_POLICY_USER": "reader", "CC_AGY_BIFROST_POLICY_PASSWORD": "pw", **extra}


def test_negated_rule_is_skipped_and_tier_alias_is_resolved() -> None:
    verdict = enforce_agy_model_policy("policy-routed", None, _rules_env(), transport=_Gateway())

    assert verdict["model"] == "gemini-4-argon-high"
    assert bifrost_policy.policy_routing_source(POLICY_URL) == "routing_rules"


def test_fallback_after_rejected_credentials_is_logged_and_reported(caplog: pytest.LogCaptureFixture) -> None:
    env = _rules_env(CC_AGY_BIFROST_AGY_POLICY_MODEL="gemini-3.8-flash-low")

    with caplog.at_level(logging.WARNING, logger="server.bifrost_policy"):
        verdict = enforce_agy_model_policy("policy-routed", None, env, transport=_Gateway(rules_status=401))
    bifrost_policy.clear_policy_cache()
    report = inspect_bifrost_policy(env, live=True, transport=_Gateway(rules_status=401))

    assert verdict["model"] == "gemini-3.8-flash-low"
    assert "rules_unauthorized" in caplog.text
    assert report["policy_routed_source"] == "fallback_model_rules_unauthorized"
    assert "Checked on this machine" in report["advisory"]


# ---------------------------------------------------------------------------
# P2. Bounded ledger, estimated usage, one process scan per kill
# ---------------------------------------------------------------------------


def test_conversation_ledger_keeps_only_recent_entries() -> None:
    ledger = _BoundedLedger(max_entries=3)
    for idx in range(4):
        ledger[("cfg", f"c{idx}")] = (1, {})
    ledger.get(("cfg", "c1"))
    ledger[("cfg", "c4")] = (1, {})

    assert list(ledger) == [("cfg", "c3"), ("cfg", "c1"), ("cfg", "c4")]
    assert isinstance(agy_cli._CONVERSATION_USAGE_LEDGER, _BoundedLedger)


def test_estimated_usage_is_flagged_on_the_receipt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (["git", "init", "-q", "-b", "main"], ["git", "config", "user.email", "t@e.st"],
                ["git", "config", "user.name", "T"]):
        subprocess.run(cmd, cwd=repo, check=True)
    (repo / "a.py").write_text("A = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    store = ReceiptStore(workspace_root=repo, data_dir=tmp_path / "data")

    def worker(req: dict[str, Any]) -> dict[str, Any]:
        Path(req["workspace_dir"], "a.py").write_text("A = 2\n", encoding="utf-8")
        return {"text": "done", "usage": {"prompt_token_count": 10, "total_token_count": 10}, "usage_estimated": True}

    jm = JobManager(workspace_root=repo, store=store, worker_backend=worker)
    try:
        result = jm.delegate_start("Set A to 2", project="p", wait=True)
    finally:
        jm.shutdown(wait=True)

    assert store.read_receipt(result["job_id"])["usage_estimated"] is True


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_process_table_is_scanned_once_per_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real_scan = agy_cli._scan_process_tree

    def counting_scan(root_pid: int, **kwargs: Any) -> set[int]:
        calls.append(root_pid)
        return real_scan(root_pid, **kwargs)

    monkeypatch.setattr(agy_cli, "_scan_process_tree", counting_scan)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)

    agy_cli._terminate_process_tree(proc)
    agy_cli._terminate_process_tree(proc)

    assert proc.poll() is not None and len(calls) == 1


def test_symlinked_secrets_and_real_receipt_store_layout_are_masked_in_sandbox(tmp_path: Path) -> None:
    home = tmp_path / "home"
    dotfiles = home / "dotfiles"
    (dotfiles / "ssh").mkdir(parents=True)
    (dotfiles / "ssh" / "id_ed25519").write_text("SECRET-SSH-KEY", encoding="utf-8")
    (dotfiles / "netrc").write_text("machine api.example.com password secret", encoding="utf-8")
    (home / ".ssh").symlink_to(dotfiles / "ssh")
    (home / ".netrc").symlink_to(dotfiles / "netrc")

    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (["git", "init", "-q", "-b", "main"], ["git", "config", "user.email", "t@e.st"],
                ["git", "config", "user.name", "T"]):
        subprocess.run(cmd, cwd=repo, check=True)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    data_dir = tmp_path / "plugin-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / ".hmac_key").write_bytes(b"hmac-secret-bytes")
    wt_container = tmp_path / ".cc-agy-worktrees-repo-1234abcd"
    wt1 = wt_container / "job-1"
    wt2 = wt_container / "job-2"
    wt1.mkdir(parents=True)
    wt2.mkdir(parents=True)
    (wt1 / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (wt2 / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    config_dir = home / ".gemini" / "antigravity-cli"
    config_dir.mkdir(parents=True)

    args = build_bwrap_prefix(
        "bwrap",
        workspace=wt1,
        config_dir=config_dir,
        home=home,
        adc_path=None,
        data_dir=data_dir,
    )
    joined = " ".join(args)
    assert f"--tmpfs {(dotfiles / 'ssh').resolve()}" in joined
    assert f"--dev-bind /dev/null {(dotfiles / 'netrc').resolve()}" in joined
    assert f"--tmpfs {data_dir.resolve()}" in joined
    assert f"--tmpfs {wt1.parent.resolve()}" in joined
    assert f"--ro-bind {config_dir / 'settings.json'} {config_dir / 'settings.json'}" in joined
    assert f"--ro-bind {config_dir / 'mcp'} {config_dir / 'mcp'}" in joined

    profile = build_seatbelt_profile(
        workspace=wt1,
        config_dir=config_dir,
        home=home,
        adc_path=None,
        data_dir=data_dir,
    )
    assert f'(subpath "{(dotfiles / "ssh").resolve()}")' in profile
    assert f'(literal "{(dotfiles / "netrc").resolve()}")' in profile
    assert f'(subpath "{data_dir.resolve()}")' in profile
    assert f'(subpath "{wt1.parent.resolve()}")' in profile
    assert f'(literal "{config_dir / "settings.json"}")' in profile

    if _bwrap_usable():
        plan = plan_worker_sandbox(
            workspace_dir=wt1,
            config_dir=config_dir,
            data_dir=data_dir,
            env={"HOME": str(home), "CC_AGY_WORKER_SANDBOX": "auto"},
        )
        probe = textwrap.dedent(
            f"""
            import os
            def can_write(p):
                try:
                    open(p, "w").write("x")
                    return True
                except OSError:
                    return False
            print(os.path.exists({str((dotfiles / 'ssh' / 'id_ed25519').resolve())!r}))
            print(open({str((dotfiles / 'netrc').resolve())!r}).read() == "")
            print(os.path.exists({str((data_dir / '.hmac_key').resolve())!r}))
            print(os.path.exists({str((wt2 / 'app.py').resolve())!r}))
            print(can_write({str(config_dir / 'settings.json')!r}))
            print(can_write({str(config_dir / 'state.log')!r}))
            """
        )
        proc = subprocess.run(plan.wrap([sys.executable, "-c", probe]), cwd=wt1, capture_output=True, text=True, timeout=20, check=False)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.split() == ["False", "True", "False", "False", "False", "True"]


def test_pattern_based_secret_env_scrubbing_and_read_only_plan_mode_sandbox(tmp_path: Path) -> None:
    env = build_agy_subprocess_env(
        {
            "HOME": str(tmp_path),
            "PATH": os.defpath,
            "ANTIGRAVITY_OAUTH_TOKEN": "keep-agy-oauth",
            "NPM_TOKEN": "npm-secret",
            "HF_TOKEN": "hf-secret",
            "STRIPE_SECRET_KEY": "sk_live_123",
            "DATABASE_PASSWORD": "db-password",
            "MY_SERVICE_API_KEY": "svc-key",
            "KUBECONFIG": "/tmp/kubeconfig",
            "DOCKER_AUTH_CONFIG": "{}",
            "VAULT_TOKEN": "s.vault",
        }
    )
    assert env.get("ANTIGRAVITY_OAUTH_TOKEN") == "keep-agy-oauth"
    for leaked in (
        "NPM_TOKEN",
        "HF_TOKEN",
        "STRIPE_SECRET_KEY",
        "DATABASE_PASSWORD",
        "MY_SERVICE_API_KEY",
        "KUBECONFIG",
        "DOCKER_AUTH_CONFIG",
        "VAULT_TOKEN",
    ):
        assert leaked not in env

    ws = _workspace(tmp_path)
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    ro_args = build_bwrap_prefix(
        "bwrap", workspace=ws, config_dir=cfg, home=tmp_path, adc_path=None, read_only_workspace=True
    )
    ro_joined = " ".join(ro_args)
    assert f"--ro-bind {ws} {ws}" in ro_joined
    assert f"--bind {ws} {ws}" not in ro_joined


def test_round2_sandbox_config_guards_foreign_conversations_and_ask_media_cleanup(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".local" / "share" / "opencode").mkdir(parents=True)
    (home / ".local" / "share" / "keyrings").mkdir(parents=True)
    (home / ".config" / "anthropic").mkdir(parents=True)
    (home / ".pgpass").write_text("db:5432:*:postgres:secret", encoding="utf-8")
    (home / "proj").mkdir(parents=True)
    (home / "proj" / ".env").write_text("DOTENV_SECRET=1", encoding="utf-8")
    (home / "code" / "org" / "svc").mkdir(parents=True)
    (home / "code" / "org" / "svc" / ".env").write_text("DEEP_SECRET=1", encoding="utf-8")

    cfg = home / ".gemini" / "antigravity-cli"
    (cfg / "conversations").mkdir(parents=True)
    (cfg / "conversations" / "other.pb").write_text("OTHER_JOB", encoding="utf-8")
    (cfg / "conversations" / "other.db-wal").write_text("OTHER_WAL", encoding="utf-8")
    (cfg / "conversations" / "active.pb").write_text("ACTIVE_JOB", encoding="utf-8")
    (cfg / "conversations" / "active.db-wal").write_text("ACTIVE_WAL", encoding="utf-8")
    (cfg / "brain" / "other").mkdir(parents=True)
    (cfg / "brain" / "other" / "notes.md").write_text("FOREIGN_BRAIN", encoding="utf-8")
    (cfg / "brain" / "active").mkdir(parents=True)
    (cfg / "brain" / "active" / "notes.md").write_text("ACTIVE_BRAIN", encoding="utf-8")
    ws = _workspace(tmp_path)

    args = build_bwrap_prefix(
        "bwrap",
        workspace=ws,
        config_dir=cfg,
        home=home,
        adc_path=None,
        conversation_id="active",
    )
    joined = " ".join(args)
    assert f"--tmpfs {(home / '.local' / 'share' / 'opencode').resolve()}" in joined
    assert f"--tmpfs {(home / '.local' / 'share' / 'keyrings').resolve()}" in joined
    assert f"--tmpfs {(home / '.config' / 'anthropic').resolve()}" in joined
    assert f"--dev-bind /dev/null {(home / '.pgpass').resolve()}" in joined
    assert f"--dev-bind /dev/null {(home / 'proj' / '.env').resolve()}" in joined
    assert f"--dev-bind /dev/null {(home / 'code' / 'org' / 'svc' / '.env').resolve()}" in joined
    assert f"--tmpfs {(cfg / 'brain' / 'other').resolve()}" in joined
    assert f"--tmpfs {(cfg / 'brain' / 'active').resolve()}" not in joined
    assert f"--tmpfs {(cfg / 'brain').resolve()} " not in joined
    assert f"--dev-bind /dev/null {(cfg / 'conversations' / 'other.pb').resolve()}" in joined
    assert f"--dev-bind /dev/null {(cfg / 'conversations' / 'other.db-wal').resolve()}" in joined
    assert f"--dev-bind /dev/null {(cfg / 'conversations' / 'active.pb').resolve()}" not in joined
    assert f"--dev-bind /dev/null {(cfg / 'conversations' / 'active.db-wal').resolve()}" not in joined
    for guarded_sub in ("GEMINI.md", "skills", "extensions", "rules"):
        assert f"--ro-bind {cfg / guarded_sub} {cfg / guarded_sub}" in joined

    with pytest.raises(WorkerSandboxError, match="CC_AGY_WORKER_SANDBOX=required"):
        plan_worker_sandbox(
            workspace_dir=ws,
            config_dir=cfg,
            env={"CC_AGY_BWRAP_BIN": str(tmp_path / "missing-bwrap"), "CC_AGY_WORKER_SANDBOX": "required"},
            read_only_workspace=True,
        )

    fake_agy = _make_fake_agy(
        tmp_path / "bin" / "agy",
        """
        _ = sys.stdin.read()
        print(json.dumps({"status": "SUCCESS", "conversation_id": "c-media", "num_turns": 1,
                          "response": "inspected", "usage": {"input_tokens": 10, "output_tokens": 2}}))
        """,
    )
    state_root = tmp_path / "state"
    env = {
        "PATH": os.defpath,
        "CC_AGY_CLI_BIN": str(fake_agy),
        "CC_AGY_CLI_CONFIG_DIR": str(cfg),
        "CC_AGY_STATE_DIR": str(state_root),
    }
    res = run_agy_cli_ask(
        {
            "prompt": "check image",
            "prompt_prefix": "ctx",
            "prompt_delta": "q1",
            "session_id": "s-media",
            "workspace_root": str(ws),
            "media_parts": [{"inlineData": {"mimeType": "image/heic", "data": "aGVsbG8="}}],
            "env": env,
        }
    )
    assert res["text"] == "inspected"
    assert list(state_root.rglob(".agy_media")) == []


def test_timed_out_ask_captures_partial_usage_and_real_agy_brain_sandbox_smoke(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    state_root = tmp_path / "state"

    fake_agy = _make_fake_agy(
        tmp_path / "bin" / "agy",
        """
        import time
        _ = sys.stdin.read()
        if "--conversation" in argv:
            sys.stdout.write(json.dumps({"usage": {"input_tokens": 260, "cache_read_tokens": 900, "output_tokens": 35}}) + "\\n")
            sys.stdout.flush()
            time.sleep(5.0)
            raise SystemExit(0)
        if os.environ.get("FAKE_AGY_TIMEOUT_FRESH") == "1":
            sys.stdout.write(json.dumps({"usage": {"input_tokens": 310, "output_tokens": 12}}) + "\\n")
            sys.stdout.flush()
            time.sleep(5.0)
            raise SystemExit(0)
        print(json.dumps({
            "status": "SUCCESS",
            "conversation_id": "conv-timeout-delta",
            "num_turns": 1,
            "response": "ok-answer",
            "usage": {"input_tokens": 1000, "output_tokens": 20},
        }))
        """,
    )
    base_env = {
        "PATH": os.defpath,
        "CC_AGY_CLI_BIN": str(fake_agy),
        "CC_AGY_CLI_CONFIG_DIR": str(cfg),
        "CC_AGY_STATE_DIR": str(state_root),
    }
    # Turn 1 succeeds (1000 prompt, 20 output)
    r1 = run_agy_cli_ask(
        {
            "prompt": "full-1",
            "prompt_prefix": "ctx",
            "prompt_delta": "q1",
            "session_id": "s-to",
            "workspace_root": str(ws),
            "timeout_seconds": 3.0,
            "env": base_env,
        }
    )
    assert r1["usage"]["prompt_token_count"] == 1000
    assert r1["usage"]["candidates_token_count"] == 20

    # Turn 2 resumed attempt emits cumulative (1160 prompt, 35 output -> delta 160 prompt, 15 output),
    # times out at 60% budget (~3.6s), and falls back to fresh (1000 prompt, 20 output) -> total 1160 prompt, 35 output!
    r2 = run_agy_cli_ask(
        {
            "prompt": "full-2",
            "prompt_prefix": "ctx",
            "prompt_delta": "q2",
            "session_id": "s-to",
            "workspace_root": str(ws),
            "timeout_seconds": 6.0,
            "env": base_env,
        }
    )
    assert r2["resumed_conversation"] is False
    assert r2["usage"]["prompt_token_count"] == 1160
    assert r2["usage"]["candidates_token_count"] == 35

    # Total timeout on fresh ask still attaches partial_usage onto TimeoutError
    with pytest.raises(TimeoutError) as exc_info:
        run_agy_cli_ask(
            {
                "prompt": "full-3",
                "session_id": "",
                "workspace_root": str(ws),
                "timeout_seconds": 0.4,
                "env": {**base_env, "FAKE_AGY_TIMEOUT_FRESH": "1"},
            }
        )
    assert exc_info.value.partial_usage["prompt_token_count"] == 310  # type: ignore[attr-defined]
    assert exc_info.value.partial_usage["candidates_token_count"] == 12  # type: ignore[attr-defined]

    # Verify --list-settings CLI flag works cleanly
    from server.install import main as install_main

    assert install_main(["--list-settings", "--json"]) == 0

    # If real `agy` and usable `bwrap` exist on this host, verify `agy --version` runs inside the sandbox
    # while keeping active brain/<id> writable and foreign brain/<other> hidden.
    real_agy = Path.home() / ".local" / "bin" / "agy"
    if _bwrap_usable() and real_agy.is_file():
        (cfg / "brain" / "conv-live").mkdir(parents=True, exist_ok=True)
        (cfg / "brain" / "conv-live" / "artifact.md").write_text("KEEP_ME", encoding="utf-8")
        (cfg / "brain" / "conv-other").mkdir(parents=True, exist_ok=True)
        (cfg / "brain" / "conv-other" / "secret.md").write_text("HIDE_ME", encoding="utf-8")
        plan = plan_worker_sandbox(
            workspace_dir=ws,
            config_dir=cfg,
            readonly_paths=(real_agy,),
            conversation_id="conv-live",
            env={"HOME": str(tmp_path), "CC_AGY_WORKER_SANDBOX": "required"},
        )
        ver_proc = subprocess.run(
            plan.wrap([str(real_agy), "--version"]),
            cwd=ws,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert ver_proc.returncode == 0, ver_proc.stderr
        probe = (
            f"import pathlib; "
            f"a = pathlib.Path({str((cfg / 'brain' / 'conv-live' / 'artifact.md').resolve())!r}); "
            f"o = pathlib.Path({str((cfg / 'brain' / 'conv-other' / 'secret.md').resolve())!r}); "
            f"a.write_text(a.read_text() + '+OK'); "
            f"print(a.read_text(), o.exists())"
        )
        b_proc = subprocess.run(
            plan.wrap([sys.executable, "-c", probe]),
            cwd=ws,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert b_proc.returncode == 0, b_proc.stderr
        assert b_proc.stdout.strip() == "KEEP_ME+OK False"
        assert (cfg / "brain" / "conv-live" / "artifact.md").read_text(encoding="utf-8") == "KEEP_ME+OK"
