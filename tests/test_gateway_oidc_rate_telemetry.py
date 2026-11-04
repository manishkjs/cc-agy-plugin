"""Unit tests for per-model thinkingLevel clamping, OIDC gateway refresh, rate card, and telemetry sink."""

from __future__ import annotations

import base64
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import stat
import subprocess
import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from server.doctor import run_doctor
from server.install import (
    build_agy_server_entry,
    build_claude_user_config,
    build_codex_server_block,
    build_cursor_server_entry,
    main as install_main,
)
from server.rate_card import (
    DEFAULT_LEAD_RATES,
    MODEL_PRICING_PER_1M_USD,
    check_promotional_rate_expiry,
    clamp_gemini_thinking_level,
    load_rate_card,
    supported_gemini_thinking_levels,
    validate_rate_card,
)
from server.receipts import ReceiptStore, redact_api_keys
from server.regions import (
    clear_oidc_token_cache,
    estimate_cost_usd,
    resolve_vertex_base_url_audience,
    resolve_vertex_base_url_token,
)
from server.telemetry import (
    build_telemetry_event,
    clear_telemetry_dedup_cache,
    emit_receipt_telemetry,
    flush_telemetry,
    resolve_telemetry_identity,
    validate_webhook_sink_url,
)
from server.true_cost import lead_rates, rate_for_model


def _make_jwt(header: dict[str, Any], payload: dict[str, Any], sig: bytes = b"0123456789abcdef0123456789abcdef") -> str:
    def _enc(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    h_seg = _enc(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p_seg = _enc(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    s_seg = _enc(sig)
    return f"{h_seg}.{p_seg}.{s_seg}"


class TestThinkingLevelClamping:
    def test_clamps_minimal_to_low_on_flash_38_37_argon_25_and_pro_31(self) -> None:
        for model in (
            "gemini-4-argon",
            "gemini-4-carbon",
            "gemini-flash-next",
            "gemini-3.8-flash",
            "gemini-3.8-flash-medium",
            "publishers/google/models/gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-3.1-pro-preview",
            "gemini-3-pro-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ):
            assert clamp_gemini_thinking_level(model, "MINIMAL", allow_extra_high=False) == "LOW"
            assert clamp_gemini_thinking_level(model, "MINIMAL", allow_extra_high=True) == "LOW"
            assert (
                clamp_gemini_thinking_level(
                    model,
                    "ZERO",
                    allow_extra_high=True,
                    map_zero_to_minimal=True,
                )
                == "LOW"
            )

    def test_preserves_minimal_on_flash_36_3_and_flash_lite(self) -> None:
        for model in (
            "gemini-3.6-flash",
            "gemini-3-flash-preview",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite-preview",
        ):
            assert clamp_gemini_thinking_level(model, "MINIMAL", allow_extra_high=False) == "MINIMAL"

    def test_clamps_extra_high_and_3_pro_medium_per_published_table(self) -> None:
        assert clamp_gemini_thinking_level("gemini-4-argon", "EXTRA_HIGH", allow_extra_high=False) == "HIGH"
        assert clamp_gemini_thinking_level("gemini-4-argon", "EXTRA_HIGH", allow_extra_high=True) == "HIGH"
        assert clamp_gemini_thinking_level("gemini-4-carbon", "xhigh", allow_extra_high=True) == "HIGH"
        assert clamp_gemini_thinking_level("gemini-3.8-flash", "EXTRA_HIGH", allow_extra_high=True) == "HIGH"
        assert clamp_gemini_thinking_level("gemini-3.1-pro-preview", "EXTRA_HIGH", allow_extra_high=True) == "HIGH"
        assert clamp_gemini_thinking_level("gemini-3-pro-preview", "MEDIUM", allow_extra_high=True) == "HIGH"
        assert "EXTRA_HIGH" not in supported_gemini_thinking_levels("gemini-4-argon", allow_extra_high=False)

    def test_genai_ask_backend_clamps_minimal_and_extra_high_without_pydantic_error(self) -> None:
        from google.genai import types as genai_types

        clamped_min = clamp_gemini_thinking_level("gemini-3.8-flash", "MINIMAL", allow_extra_high=False)
        cfg_min = genai_types.ThinkingConfig(thinking_level=clamped_min)
        assert str(cfg_min.thinking_level).endswith("LOW")

        clamped_xh = clamp_gemini_thinking_level("gemini-4-argon", "EXTRA_HIGH", allow_extra_high=False)
        cfg_xh = genai_types.ThinkingConfig(thinking_level=clamped_xh)
        assert str(cfg_xh.thinking_level).endswith("HIGH")


class TestDeclarativeRateCard:
    def test_rate_card_json_loads_and_preserves_backward_compatible_rates(self) -> None:
        card = load_rate_card()
        assert validate_rate_card(card) == []
        assert MODEL_PRICING_PER_1M_USD["gemini-3.8-flash"] == {
            "input": 0.50,
            "cached_input": 0.05,
            "output": 3.00,
        }
        assert MODEL_PRICING_PER_1M_USD["gemini-3.7-flash"]["input"] == 0.50
        assert DEFAULT_LEAD_RATES["sonnet"] == (3.00, 0.30, 3.75, 6.00, 15.00)
        assert rate_for_model("claude-sonnet-4-6") == (3.00, 0.30, 3.75, 6.00, 15.00)
        assert rate_for_model("claude-sonnet-5-5-20261001") == (2.00, 0.20, 2.50, 4.00, 10.00)
        assert rate_for_model("claude-sonnet-5-0") == (2.00, 0.20, 2.50, 4.00, 10.00)
        assert rate_for_model("claude-opus-4-8") == (5.00, 0.50, 6.25, 10.00, 25.00)
        assert rate_for_model("claude-opus-5-5") == (4.00, 0.20, 5.00, 8.00, 20.00)
        assert rate_for_model("mystery-model") is None

        cost = estimate_cost_usd(
            "gemini-3.7-flash",
            {"prompt_token_count": 1_000_000, "cached_content_token_count": 0, "candidates_token_count": 0},
            single_request=True,
        )
        assert cost["pricing_known"] is True
        assert cost["total_cost_cents"] == pytest.approx(50.0)

    def test_sonnet_override_propagates_to_sonnet_5_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CC_AGY_LEAD_RATES", '{"sonnet": [3.5, 0.35, 4.0, 7.0, 17.5]}')
        merged = lead_rates()
        assert merged["sonnet-5.5"] == (3.5, 0.35, 4.0, 7.0, 17.5)

    def test_promotional_rate_expiry_check(self) -> None:
        assert check_promotional_rate_expiry(date(2026, 12, 31)) == []
        expired = check_promotional_rate_expiry("2027-01-01")
        assert len(expired) >= 2
        assert any("gemini-3.8-flash" in msg for msg in expired)
        assert any("sonnet-5.5" in msg for msg in expired)


class TestOidcGatewayAuthAndBootstrap:
    def setup_method(self) -> None:
        clear_oidc_token_cache()

    def teardown_method(self) -> None:
        clear_oidc_token_cache()

    def test_audience_resolution_and_flag_injection_guard(self) -> None:
        env = {
            "CC_AGY_VERTEX_BASE_URL": "https://router-abc123-uc.a.run.app/v1",
            "CC_AGY_VERTEX_OIDC": "1",
        }
        assert resolve_vertex_base_url_audience(env=env) == "https://router-abc123-uc.a.run.app"

        with pytest.raises(ValueError, match="must not start with '-'"):
            resolve_vertex_base_url_audience("--impersonate-service-account=evil")

        with pytest.raises(ValueError, match="whitespace"):
            resolve_vertex_base_url_audience("https://router.example.com --extra")

        with pytest.raises(ValueError, match="Unsupported OIDC audience"):
            resolve_vertex_base_url_audience("http://router.example.com")

    def test_static_token_beats_oidc_and_gcloud_mints_cached_jwt(self, tmp_path: Path) -> None:
        fake_jwt = _make_jwt(
            {"alg": "RS256", "typ": "JWT"},
            {"email": "dev@example.com", "exp": int(time.time()) + 3600},
        )
        fake_gcloud = tmp_path / "gcloud"
        counter_file = tmp_path / "count.txt"
        fake_gcloud.write_text(
            f"#!/usr/bin/env bash\n"
            f"echo x >> {counter_file}\n"
            f"printf '%s\\n' '{fake_jwt}'\n",
            encoding="utf-8",
        )
        fake_gcloud.chmod(0o755)

        env = {
            "PATH": f"{tmp_path}:{os.environ.get('PATH', '')}",
            "CC_AGY_VERTEX_BASE_URL": "https://gateway.example.com",
            "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://gateway.example.com",
        }
        tok1 = resolve_vertex_base_url_token(env=env)
        tok2 = resolve_vertex_base_url_token(env=env)
        assert tok1 == fake_jwt
        assert tok2 == fake_jwt
        assert len(counter_file.read_text(encoding="utf-8").splitlines()) == 1

        env_with_static = {**env, "CC_AGY_VERTEX_BASE_URL_TOKEN": "static-vkey-123"}
        assert resolve_vertex_base_url_token(env=env_with_static) == "static-vkey-123"

        redacted = redact_api_keys({"error": f"Upstream rejected Bearer {fake_jwt}", "status": "error"})
        assert fake_jwt not in redacted["error"]
        assert "<oidc-token>" in redacted["error"]

    def test_metadata_server_fallback_when_gcloud_unavailable(self) -> None:
        fake_jwt = _make_jwt(
            {"alg": "RS256", "typ": "JWT"},
            {"sub": "1234567890", "exp": int(time.time()) + 1800},
        )
        with patch("server.regions.shutil.which", return_value=None), patch(
            "server.regions._fetch_metadata_identity_token", return_value=fake_jwt
        ) as mock_meta:
            tok = resolve_vertex_base_url_token(
                env={
                    "PATH": "/nonexistent",
                    "CC_AGY_VERTEX_BASE_URL": "https://gateway.example.com",
                    "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://gateway.example.com",
                }
            )
            assert tok == fake_jwt
            mock_meta.assert_called_once_with("https://gateway.example.com")

    def test_install_flags_and_powershell_bootstrap_script(self, tmp_path: Path) -> None:
        sink_file = tmp_path / "events.ndjson"
        codex_block = build_codex_server_block(
            base_url="https://gateway.example.com",
            gateway_audience="https://gateway.example.com",
            telemetry_sink=str(sink_file),
            auth_mode="gateway",
        )
        assert 'CC_AGY_VERTEX_BASE_URL_AUDIENCE = "https://gateway.example.com"' in codex_block
        assert f'CC_AGY_TELEMETRY_SINK = "{sink_file}"' in codex_block

        claude_cfg = build_claude_user_config(
            auth_mode="gateway",
            base_url="https://gateway.example.com",
            gateway_audience="https://gateway.example.com",
            telemetry_sink=str(sink_file),
        )
        assert claude_cfg["vertex_base_url_audience"] == "https://gateway.example.com"
        assert claude_cfg["telemetry_sink"] == str(sink_file)

        cursor_entry = build_cursor_server_entry(
            auth_mode="gateway",
            base_url="https://gateway.example.com",
            gateway_audience="https://gateway.example.com",
            telemetry_sink=str(sink_file),
        )
        agy_entry = build_agy_server_entry(
            auth_mode="gateway",
            base_url="https://gateway.example.com",
            gateway_audience="https://gateway.example.com",
        )
        assert cursor_entry["env"]["CC_AGY_VERTEX_BASE_URL_AUDIENCE"] == "https://gateway.example.com"
        assert agy_entry["env"]["CC_AGY_VERTEX_BASE_URL_AUDIENCE"] == "https://gateway.example.com"

        rc = install_main(
            [
                "--host",
                "cursor",
                "--auth",
                "gateway",
                "--base-url",
                "https://gateway.example.com",
                "--gateway-audience",
                "https://gateway.example.com",
                "--no-bootstrap",
                "--dry-run",
                "--json",
            ]
        )
        assert rc == 0

        ps1_path = Path(__file__).resolve().parent.parent / "scripts" / "bootstrap-gateway.ps1"
        ps1_text = ps1_path.read_text(encoding="utf-8")
        assert "--gateway-audience" in ps1_text
        assert "gcloud" in ps1_text
        assert "\u2014" not in ps1_text


class TestOptInTelemetryAndIdentityPrecedence:
    def test_identity_precedence_prevents_unverified_spoofing(self) -> None:
        now = time.time()
        valid_jwt = _make_jwt(
            {"alg": "RS256", "typ": "JWT"},
            {"email": "jwt-user@example.com", "exp": now + 1800},
        )
        alg_none_jwt = _make_jwt(
            {"alg": "none", "typ": "JWT"},
            {"email": "forged@example.com", "exp": now + 1800},
        )
        expired_jwt = _make_jwt(
            {"alg": "RS256", "typ": "JWT"},
            {"email": "expired@example.com", "exp": now - 600},
        )

        # 1. IAP header beats everything and is marked verified
        ident_iap = resolve_telemetry_identity(
            headers={
                "X-Goog-Authenticated-User-Email": "accounts.google.com:iap-verified@example.com",
                "X-User-Id": "spoofed@example.com",
            },
            bearer_token=valid_jwt,
            env={"CC_AGY_USER_ID": "env-spoof@example.com"},
            now=now,
        )
        assert ident_iap == {
            "user_id": "iap-verified@example.com",
            "identity_source": "iap_header",
            "identity_verified": True,
        }

        # 2. Valid signed JWT beats self-reported X-User-Id / CC_AGY_USER_ID
        ident_jwt = resolve_telemetry_identity(
            headers={"X-User-Id": "spoofed@example.com"},
            bearer_token=valid_jwt,
            env={"CC_AGY_USER_ID": "env-spoof@example.com"},
            now=now,
        )
        assert ident_jwt == {
            "user_id": "jwt-user@example.com",
            "identity_source": "oidc_jwt",
            "identity_verified": False,
        }

        # 3. alg:none and expired JWTs are rejected and fall through to self_reported
        ident_forged = resolve_telemetry_identity(
            bearer_token=alg_none_jwt,
            env={"CC_AGY_USER_ID": "dev-local"},
            now=now,
        )
        assert ident_forged == {
            "user_id": "dev-local",
            "identity_source": "self_reported",
            "identity_verified": False,
        }
        ident_exp = resolve_telemetry_identity(
            bearer_token=expired_jwt,
            env={},
            now=now,
        )
        assert ident_exp == {
            "user_id": "anonymous",
            "identity_source": "anonymous",
            "identity_verified": False,
        }

    def test_file_sink_permissions_symlink_guard_and_receipt_store_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = tmp_path / "repo"
        ws.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        sink_path = tmp_path / "telemetry" / "receipts.ndjson"
        monkeypatch.setenv("CC_AGY_TELEMETRY_SINK", str(sink_path))
        monkeypatch.setenv("CC_AGY_USER_ID", "alice")

        store = ReceiptStore(workspace_root=ws, data_dir=tmp_path / "data")
        # Intermediate 'running' receipt must not emit
        store.write_receipt(
            "job-tel-1",
            {
                "kind": "delegate",
                "status": "running",
                "auth_mode": "vertex",
                "model": "gemini-flash-next",
                "prompt": "TOP_SECRET_PROMPT_DO_NOT_LEAK",
                "usage": {"prompt_token_count": 100, "total_token_count": 100},
                "cost": {"total_cost_cents": 0.05, "pricing_known": True},
            },
        )
        assert not sink_path.exists()

        # Terminal receipt emits sanitized NDJSON with 0o600 permissions
        rec = store.write_receipt(
            "job-tel-1",
            {
                "kind": "delegate",
                "status": "ok",
                "auth_mode": "vertex",
                "model": "gemini-flash-next",
                "prompt": "TOP_SECRET_PROMPT_DO_NOT_LEAK",
                "diff": "+SECRET_DIFF_LINE",
                "workspace_root": str(ws),
                "duration_seconds": 4.25,
                "verification_mode": "behavioral",
                "usage": {
                    "prompt_token_count": 2000,
                    "cached_content_token_count": 500,
                    "candidates_token_count": 300,
                    "thoughts_token_count": 100,
                    "total_token_count": 2400,
                },
                "cost": {"total_cost_cents": 0.21, "pricing_known": True},
            },
        )
        # Re-emitting identical receipt is deduplicated
        assert emit_receipt_telemetry(rec) is False

        lines = sink_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        raw_text = lines[0]
        assert "TOP_SECRET_PROMPT_DO_NOT_LEAK" not in raw_text
        assert "SECRET_DIFF_LINE" not in raw_text
        assert str(ws) not in raw_text
        ev = json.loads(raw_text)
        assert ev["job_id"] == "job-tel-1"
        assert ev["cost_cents"] == pytest.approx(0.1975)
        assert ev["cost_cents"] == pytest.approx(rec["cost"]["total_cost_cents"])
        assert ev["usage"]["total_token_count"] == 2400
        assert ev["user_id"] == "alice"
        assert ev["identity_source"] == "self_reported"
        assert ev["identity_verified"] is False
        assert len(ev["workspace_hash"]) == 16
        assert (stat.S_IMODE(sink_path.stat().st_mode) & 0o777) == 0o600

        # Symlink target must be rejected
        sym_path = tmp_path / "symlink.ndjson"
        sym_path.symlink_to(sink_path)
        with pytest.raises(ValueError, match="symlink"):
            emit_receipt_telemetry(
                {"job_id": "job-sym-2", "status": "ok", "cost": {"total_cost_cents": 0.1}},
                env={"CC_AGY_TELEMETRY_SINK": str(sym_path)},
            )

    def test_webhook_sink_ssrf_guards_and_loopback_delivery(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Plain HTTP"):
            validate_webhook_sink_url("http://collector.example.com/events")
        with pytest.raises(ValueError, match="blocked"):
            validate_webhook_sink_url("https://169.254.169.254/latest/meta-data")
        with pytest.raises(ValueError, match="blocked"):
            validate_webhook_sink_url("https://metadata.google.internal/computeMetadata/v1/")
        with pytest.raises(ValueError, match="userinfo"):
            validate_webhook_sink_url("https://user:secret@collector.example.com/ingest")

        received: list[tuple[dict[str, str], dict[str, Any]]] = []

        class _CollectorHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                received.append(({k.lower(): v for k, v in self.headers.items()}, payload))
                self.send_response(202)
                self.end_headers()

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _CollectorHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            ok = emit_receipt_telemetry(
                {
                    "job_id": "job-webhook-1",
                    "kind": "ask_gemini",
                    "status": "ok",
                    "model": "gemini-flash-next",
                    "cost": {"total_cost_cents": 0.12, "pricing_known": True},
                    "usage": {"total_token_count": 512},
                },
                env={
                    "CC_AGY_TELEMETRY_SINK": f"http://127.0.0.1:{port}/v1/receipts",
                    "CC_AGY_TELEMETRY_TOKEN": "tel-secret-xyz",
                },
            )
            assert ok is True
            assert flush_telemetry(timeout=3.0) is True
            assert len(received) == 1
            hdrs, body = received[0]
            assert hdrs.get("authorization") == "Bearer tel-secret-xyz"
            assert body["job_id"] == "job-webhook-1"
            assert body["cost_cents"] == pytest.approx(0.12)
        finally:
            srv.shutdown()
            srv.server_close()

        doc = run_doctor(
            tmp_path,
            env={
                "GEMINI_API_KEY": "AIzaSyTestKey12345678901234567890",
                "CC_AGY_TELEMETRY_SINK": "http://evil.example.com/plain",
            },
        )
        assert any("Plain HTTP telemetry sink" in issue for issue in doc["issues"])


class TestGauntletRound1Bugs:
    def setup_method(self) -> None:
        clear_oidc_token_cache()

    def teardown_method(self) -> None:
        clear_oidc_token_cache()

    def test_published_thinking_table_and_proto_prefix_stripping(self) -> None:
        doc_levels = {
            "gemini-4-argon": {"LOW", "MEDIUM", "HIGH"},
            "gemini-3.8-flash": {"LOW", "MEDIUM", "HIGH"},
            "gemini-3.6-flash": {"MINIMAL", "LOW", "MEDIUM", "HIGH"},
            "gemini-3.5-flash-lite": {"MINIMAL", "LOW", "MEDIUM", "HIGH"},
            "gemini-3.1-pro-preview": {"LOW", "MEDIUM", "HIGH"},
            "gemini-3-flash-preview": {"MINIMAL", "LOW", "MEDIUM", "HIGH"},
            "gemini-3-pro-preview": {"LOW", "HIGH"},
            "gemini-2.5-pro": {"LOW", "MEDIUM", "HIGH"},
            "gemini-2.5-flash": {"LOW", "MEDIUM", "HIGH"},
            "gemini-2.5-flash-lite": {"LOW", "MEDIUM", "HIGH"},
        }
        for model, allowed in doc_levels.items():
            for raw_lvl in ("MINIMAL", "LOW", "MEDIUM", "HIGH", "EXTRA_HIGH"):
                for allow_xh in (False, True):
                    out = clamp_gemini_thinking_level(model, raw_lvl, allow_extra_high=allow_xh)
                    assert out in allowed, f"{model} + {raw_lvl} -> {out} not in {allowed}"
                    if raw_lvl in allowed:
                        assert out == raw_lvl

        assert clamp_gemini_thinking_level("gemini-3.8-flash", "THINKING_LEVEL_MINIMAL") == "LOW"
        assert clamp_gemini_thinking_level("gemini-4-argon", "THINKING_LEVEL_EXTRA_HIGH", allow_extra_high=True) == "HIGH"
        assert clamp_gemini_thinking_level("gemini-3.8-flash", "THINKING_LEVEL_UNSPECIFIED") is None

    def test_claude_35_sonnet_not_shadowed_gemini_base_models_and_post_promo_rates(
        self, tmp_path: Path
    ) -> None:
        assert rate_for_model("claude-3-5-sonnet-20241022") == (3.00, 0.30, 3.75, 6.00, 15.00)
        assert rate_for_model("claude-3.5-sonnet") == (3.00, 0.30, 3.75, 6.00, 15.00)
        assert rate_for_model("claude-3-7-sonnet-20250219") == (3.00, 0.30, 3.75, 6.00, 15.00)
        assert rate_for_model("claude-sonnet-5-5-20261001") == (2.00, 0.20, 2.50, 4.00, 10.00)
        assert rate_for_model("claude-5-sonnet") == (2.00, 0.20, 2.50, 4.00, 10.00)

        for model_id, expected_in, expected_out in (
            ("gemini-3.5-flash", 0.50, 3.00),
            ("gemini-3.5-flash-lite", 0.10, 0.40),
            ("gemini-3.1-pro", 2.00, 12.00),
            ("gemini-3.1-flash-lite", 0.10, 0.40),
            ("gemini-3-pro", 2.00, 12.00),
            ("gemini-3-flash", 0.50, 3.00),
        ):
            c = estimate_cost_usd(
                model_id,
                {"prompt_token_count": 100_000, "candidates_token_count": 100_000},
                single_request=True,
            )
            assert c["pricing_known"] is True, f"{model_id} should be known"
            assert c["input_cost_usd"] == pytest.approx(expected_in * 0.1)
            assert c["output_cost_usd"] == pytest.approx(expected_out * 0.1)

        for pro_model in ("gemini-3.1-pro", "gemini-3-pro"):
            c_long = estimate_cost_usd(
                pro_model,
                {"prompt_token_count": 300_000, "candidates_token_count": 100_000},
                single_request=True,
            )
            assert c_long["long_context_multiplier_applied"] is True
            assert c_long["input_cost_usd"] == pytest.approx(2.00 * 2.0 * 0.3)

        post_promo = lead_rates(as_of=date(2027, 1, 2))
        assert post_promo["sonnet-5.5"] == (3.00, 0.30, 3.75, 6.00, 15.00)

        ws = tmp_path / "docrepo"
        ws.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        with patch("server.doctor.check_promotional_rate_expiry", return_value=["Promo expired notice"]):
            doc = run_doctor(ws, env={"GEMINI_API_KEY": "AIzaSyTestKey12345678901234567890"})
        assert doc["status"] == "ok"
        assert "Promo expired notice" in doc.get("rate_card_notices", [])

    def test_oidc_user_account_fallback_wif_coalescing_refresh_and_telemetry_bridge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from server.regions import resolve_auth_config

        # 1. Shell metacharacter guard on audience
        with pytest.raises(ValueError, match="metacharacters|Invalid OIDC audience"):
            resolve_vertex_base_url_audience("https://gw.example.com/&calc")

        # 2. User-account gcloud (`WrongAccountTypeError` on --audiences) retries without --audiences
        user_jwt = _make_jwt(
            {"alg": "RS256", "typ": "JWT"},
            {"email": "laptop-user@example.com", "exp": int(time.time()) + 3600},
        )
        argv_log = tmp_path / "gcloud_argv.log"
        fake_gcloud = tmp_path / "gcloud"
        fake_gcloud.write_text(
            "#!/usr/bin/env bash\n"
            f'echo "$@" >> "{argv_log}"\n'
            'for a in "$@"; do\n'
            '  case "$a" in\n'
            '    --audiences=*)\n'
            '      echo "ERROR: (gcloud.auth.print-identity-token) Invalid account type for `--audiences`. Requires valid service account." >&2\n'
            '      exit 1\n'
            '      ;;\n'
            '  esac\n'
            'done\n'
            f"printf '%s\\n' '{user_jwt}'\n",
            encoding="utf-8",
        )
        fake_gcloud.chmod(0o755)

        env = {
            "PATH": f"{tmp_path}:{os.environ.get('PATH', '')}",
            "CC_AGY_VERTEX_BASE_URL": "https://router-abc-uc.a.run.app",
            "CC_AGY_VERTEX_OIDC": "1",
            "CC_AGY_USER_ID": "spoofed@example.com",
        }
        with patch("server.regions._fetch_metadata_identity_token", side_effect=RuntimeError("no metadata")):
            minted = resolve_vertex_base_url_token(env=env)
        assert minted == user_jwt
        logged_calls = argv_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(logged_calls) == 2
        assert "--audiences=https://router-abc-uc.a.run.app" in logged_calls[0]
        assert "--audiences=" not in logged_calls[1]

        # 3. Zero-key OIDC populates telemetry oidc_jwt identity ahead of spoofed CC_AGY_USER_ID
        ident = resolve_telemetry_identity(env=env)
        assert ident["identity_source"] == "oidc_jwt"
        assert ident["user_id"] == "laptop-user@example.com"

        # 4. resolve_auth_config populates base_url_audience
        auth_cfg = resolve_auth_config(model="gemini-3.8-flash", requested_auth_mode="gateway", env=env)
        assert auth_cfg.get("base_url_audience") == "https://router-abc-uc.a.run.app"

        # 5. WIF external_account JSON in source_env (not os.environ) loads via _mint_from_credentials_file
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        clear_oidc_token_cache()
        wif_file = tmp_path / "wif.json"
        wif_file.write_text(
            json.dumps(
                {
                    "type": "external_account",
                    "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/gh",
                    "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                    "token_url": "https://sts.googleapis.com/v1/token",
                    "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa@proj.iam.gserviceaccount.com:generateAccessToken",
                    "credential_source": {"file": str(tmp_path / "subj.jwt")},
                }
            ),
            encoding="utf-8",
        )
        wif_jwt = _make_jwt(
            {"alg": "RS256", "typ": "JWT"},
            {"email": "wif-sa@proj.iam.gserviceaccount.com", "exp": int(time.time()) + 3600},
        )

        class _FakeIdTokenCreds:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.token = None

            def refresh(self, request: Any) -> None:
                self.token = wif_jwt

        with patch("google.auth.impersonated_credentials.IDTokenCredentials", _FakeIdTokenCreds):
            got_wif = resolve_vertex_base_url_token(
                env={
                    "GOOGLE_APPLICATION_CREDENTIALS": str(wif_file),
                    "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com",
                    "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://gw.example.com",
                    "PATH": "/nonexistent",
                }
            )
        assert got_wif == wif_jwt

        # 6. 16-thread coalescing -> 1 mint
        clear_oidc_token_cache()
        coal_dir = tmp_path / "coal"
        coal_dir.mkdir()
        coal_count = coal_dir / "count.txt"
        coal_gcloud = coal_dir / "gcloud"
        coal_gcloud.write_text(
            "#!/usr/bin/env bash\n"
            "sleep 0.15\n"
            f'echo x >> "{coal_count}"\n'
            f"printf '%s\\n' '{user_jwt}'\n",
            encoding="utf-8",
        )
        coal_gcloud.chmod(0o755)
        coal_env = {
            "PATH": f"{coal_dir}:{os.environ.get('PATH', '')}",
            "CC_AGY_VERTEX_BASE_URL": "https://coal.example.com",
            "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://coal.example.com",
        }
        results: list[str] = []
        threads = [
            threading.Thread(target=lambda: results.append(resolve_vertex_base_url_token(env=coal_env) or ""))
            for _ in range(16)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert len(coal_count.read_text(encoding="utf-8").splitlines()) == 1
        assert set(results) == {user_jwt}

    def test_telemetry_lifecycle_dedup_symlink_parent_ssrf_jwt_exp_and_doctor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = tmp_path / "repo"
        ws.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)

        # Symlinked parent directory (macOS /tmp -> /private/tmp) must succeed
        real_dir = tmp_path / "private_tmp"
        real_dir.mkdir()
        sym_dir = tmp_path / "tmp_link"
        sym_dir.symlink_to(real_dir, target_is_directory=True)
        sink_file = sym_dir / "receipts.ndjson"

        # Failed sink write must NOT poison dedup cache for subsequent valid sink write
        bad_rec = {
            "job_id": "job-retry-1",
            "kind": "delegate",
            "status": "ok",
            "cost": {"total_cost_cents": 0.25, "pricing_known": True},
            "usage": {"total_token_count": 1000},
        }
        with pytest.raises(ValueError):
            emit_receipt_telemetry(bad_rec, env={"CC_AGY_TELEMETRY_SINK": "relative/bad.ndjson"})
        assert emit_receipt_telemetry(bad_rec, env={"CC_AGY_TELEMETRY_SINK": str(sink_file)}) is True

        # ReceiptStore lifecycle rewrites (applied=False -> applied=True -> reverted=True) must NOT multi-count spend
        monkeypatch.setenv("CC_AGY_TELEMETRY_SINK", str(sink_file))
        store = ReceiptStore(workspace_root=ws, data_dir=tmp_path / "data")
        base_payload = {
            "kind": "delegate",
            "status": "ok",
            "auth_mode": "vertex",
            "model": "gemini-flash-next",
            "applied": False,
            "usage": {"prompt_token_count": 1000, "candidates_token_count": 200, "total_token_count": 1200},
        }
        store.write_receipt("job-life-1", dict(base_payload, applied=False))
        store.write_receipt("job-life-1", dict(base_payload, applied=True))
        store.write_receipt("job-life-1", dict(base_payload, applied=True, reverted=True))

        events = [
            json.loads(ln)
            for ln in sink_file.read_text(encoding="utf-8").splitlines()
            if json.loads(ln)["job_id"] == "job-life-1"
        ]
        assert len(events) == 1

        # Non-numeric usage string does not crash build_telemetry_event
        weird_ev = build_telemetry_event(
            {"job_id": "job-weird", "status": "ok", "usage": {"prompt_token_count": "12abc"}}
        )
        assert weird_ev["usage"]["prompt_token_count"] == 0

        # JWT missing 'exp' is rejected at oidc_jwt tier
        no_exp_jwt = _make_jwt({"alg": "RS256", "typ": "JWT"}, {"email": "forged@example.com"})
        ident_no_exp = resolve_telemetry_identity(bearer_token=no_exp_jwt, env={})
        assert ident_no_exp["identity_source"] == "anonymous"

        # SSRF edge cases: trailing dot, hex/decimal IPv4, NAT64, DNS-to-loopback, UNC local path
        for blocked_url in (
            "https://metadata.google.internal./computeMetadata/v1/",
            "https://0xa9fea9fe/latest/meta-data",
            "https://2852039166/latest/meta-data",
            "https://[64:ff9b::169.254.169.254]/hook",
        ):
            with pytest.raises(ValueError):
                validate_webhook_sink_url(blocked_url, check_dns=False)

        with patch(
            "server.telemetry.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("127.0.0.1", 443))],
        ):
            with pytest.raises(ValueError, match="blocked|loopback"):
                validate_webhook_sink_url("https://rebind.attacker.example/hook", check_dns=True)

        with pytest.raises(ValueError, match="UNC"):
            emit_receipt_telemetry(
                {"job_id": "job-unc", "status": "ok"},
                env={"CC_AGY_TELEMETRY_SINK": "//attacker-smb/share/events.ndjson"},
            )

        # doctor.py validates local file sinks too
        doc_rel = run_doctor(
            ws,
            env={
                "GEMINI_API_KEY": "AIzaSyTestKey12345678901234567890",
                "CC_AGY_TELEMETRY_SINK": "relative/path.ndjson",
            },
        )
        assert doc_rel["healthy"] is False
        assert any("telemetry" in issue.lower() for issue in doc_rel["issues"])

    def test_powershell_bootstrap_contract_and_agent_writer_zero_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ps1_path = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap-gateway.ps1"
        content = ps1_path.read_text(encoding="utf-8")
        assert "#Requires -Version 5.1" in content
        assert "sys.version_info >= (3, 10)" in content
        assert "WindowsApps" in content
        assert "$RoutingProfile" in content
        assert "[string]$Profile" not in content
        assert "[switch]$NonInteractive" in content
        assert 'Alias("HostTarget")' in content
        assert "GOOGLE_APPLICATION_CREDENTIALS" in content
        assert "--doctor" in content

        import sys
        import types as py_types
        from server.batch_plan import FileTask
        from server.writers import WriterRequest, make_agent_writer

        captured_ep: list[dict[str, Any]] = []
        captured_opts: list[Any] = []

        fake_types = py_types.SimpleNamespace()
        fake_types.ThinkingLevel = py_types.SimpleNamespace(
            MINIMAL="ENUM_MINIMAL",
            LOW="ENUM_LOW",
            MEDIUM="ENUM_MEDIUM",
            HIGH="ENUM_HIGH",
        )
        fake_types.GeminiModelOptions = lambda **kw: captured_opts.append(kw) or kw
        fake_types.VertexEndpoint = lambda **kw: captured_ep.append(kw) or kw
        fake_types.GeminiAPIEndpoint = lambda **kw: kw
        fake_types.BuiltinTools = py_types.SimpleNamespace(FINISH="FINISH")
        fake_types.ModelType = py_types.SimpleNamespace(TEXT="TEXT")
        fake_types.ModelTarget = lambda **kw: kw
        fake_types.CapabilitiesConfig = lambda **kw: kw
        fake_types.CustomSystemInstructions = lambda **kw: kw
        fake_types.BudgetConfig = lambda **kw: kw
        fake_policy = py_types.SimpleNamespace(allow_all=lambda: "ALLOW_ALL")

        class _FakeResp:
            async def resolve(self) -> None:
                return None

            def structured_output(self) -> dict[str, Any]:
                return {"path": "a.py", "content": "x = 1\n"}

            def text(self) -> str:
                return '{"path": "a.py", "content": "x = 1\\n"}'

            usage_metadata = {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15}

        class _FakeAgent:
            def __init__(self, cfg: Any) -> None:
                self.cfg = cfg

            async def __aenter__(self) -> "_FakeAgent":
                return self

            async def __aexit__(self, *args: Any) -> bool:
                return False

            async def chat(self, prompt: str) -> _FakeResp:
                return _FakeResp()

        fake_ag = py_types.ModuleType("google.antigravity")
        fake_ag.Agent = _FakeAgent  # type: ignore[attr-defined]
        fake_ag.LocalAgentConfig = lambda **kw: kw  # type: ignore[attr-defined]
        fake_ag.types = fake_types  # type: ignore[attr-defined]
        fake_ag.policy = fake_policy  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "google.antigravity", fake_ag)
        import google

        monkeypatch.setattr(google, "antigravity", fake_ag, raising=False)

        sample_task = FileTask(task_id="t1", path="a.py", action="create", intent="create a.py")
        writer_zero = make_agent_writer(
            {
                "auth_mode": "gateway",
                "base_url": "https://gw.example.com",
                "base_url_token": "static-gw-tok",
            },
            "gemini-3.8-flash",
            thinking_level="ZERO",
        )
        reply = writer_zero(WriterRequest(task=sample_task, prompt="write a.py", system="sys"))
        assert "a.py" in reply.text
        assert captured_ep[-1]["http_headers"]["X-CC-AGY-Thinking-Budget"] == "0"
        assert captured_opts[-1]["thinking_level"] == "ENUM_LOW"

        writer_bogus = make_agent_writer(
            {
                "auth_mode": "gateway",
                "base_url": "https://gw.example.com",
                "base_url_token": "static-gw-tok",
            },
            "gemini-3.8-flash",
            thinking_level="BOGUS_LEVEL",
        )
        writer_bogus(WriterRequest(task=sample_task, prompt="write a.py", system="sys"))
        assert captured_opts[-1]["thinking_level"] == "ENUM_LOW"


class TestGauntletRound2Bugs:
    """RED-first regression tests for Round 2 OIDC, telemetry, doctor, and rate-card invariants."""

    def setup_method(self) -> None:
        clear_oidc_token_cache()
        clear_telemetry_dedup_cache()
        flush_telemetry()

    def test_oidc_rejects_access_tokens_wif_no_impersonation_and_principal_routing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from server.regions import _jwt_refresh_deadline

        # 1. Windows cmd.exe expansion chars (%, ^, !, (, )) rejected by audience validator
        for bad_aud in (
            "https://gw.example.com/%USERNAME%",
            "https://gw.example.com/^calc",
            "https://gw.example.com/!PATH!",
            "https://gw.example.com/(sub)",
        ):
            with pytest.raises(ValueError):
                resolve_vertex_base_url_audience(bad_aud)

        # 2. Monotonic refresh deadline across near-expiry & full-hour tokens
        now = 1_760_000_000.0
        deadlines = [
            _jwt_refresh_deadline(_make_jwt({"alg": "RS256"}, {"exp": now + ttl}), now) - now
            for ttl in (120, 320, 400, 600, 1800, 3600)
        ]
        assert deadlines == sorted(deadlines)
        assert deadlines[1] < deadlines[2] < deadlines[3]

        # 3. WIF external_account WITHOUT service_account_impersonation_url or impersonate_sa raises cleanly
        #    (never ships a 2-dot OAuth2 access token and never falls through to metadata/fetch_id_token)
        wif_no_imp = tmp_path / "wif_no_imp.json"
        wif_no_imp.write_text(
            json.dumps(
                {
                    "type": "external_account",
                    "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/gh",
                    "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                    "token_url": "https://sts.googleapis.com/v1/token",
                    "credential_source": {"file": str(tmp_path / "subj.jwt")},
                }
            ),
            encoding="utf-8",
        )
        fetch_called = False

        def _forbid_fetch(*args: Any, **kwargs: Any) -> str:
            nonlocal fetch_called
            fetch_called = True
            return "SHOULD_NOT_CALL"

        with (
            patch("google.oauth2.id_token.fetch_id_token", _forbid_fetch),
            patch("server.regions._fetch_metadata_identity_token", side_effect=RuntimeError("no metadata")),
        ):
            with pytest.raises(RuntimeError, match="impersonat|CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT"):
                resolve_vertex_base_url_token(
                    env={
                        "GOOGLE_APPLICATION_CREDENTIALS": str(wif_no_imp),
                        "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com",
                        "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://gw.example.com",
                        "PATH": "/nonexistent",
                    }
                )
        assert fetch_called is False

        # 4. authorized_user JSON whose refresh() sets a 2-dot OAuth2 access token (`oauth2_access.c.xxx`) is NEVER returned
        auth_user_file = tmp_path / "authorized_user.json"
        auth_user_file.write_text(
            json.dumps(
                {
                    "type": "authorized_user",
                    "client_id": "cid",
                    "client_secret": "csec",
                    "refresh_token": "rtok",
                }
            ),
            encoding="utf-8",
        )

        class _FakeUserCreds:
            def __init__(self) -> None:
                self.token = "oauth2_access.c.FAKE_TWO_DOT_ACCESS_TOKEN"
                self.id_token = None

            def refresh(self, req: Any) -> None:
                self.token = "oauth2_access.c.FAKE_TWO_DOT_ACCESS_TOKEN"

        with (
            patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=_FakeUserCreds()),
            patch("google.oauth2.id_token.fetch_id_token", _forbid_fetch),
            patch("server.regions._fetch_metadata_identity_token", side_effect=RuntimeError("no metadata")),
        ):
            with pytest.raises(RuntimeError):
                resolve_vertex_base_url_token(
                    env={
                        "GOOGLE_APPLICATION_CREDENTIALS": str(auth_user_file),
                        "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com",
                        "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://gw.example.com",
                        "PATH": "/nonexistent",
                    }
                )
        assert fetch_called is False

        # 5. On an interactive GCE/Workstation dev machine where a human explicitly logged into gcloud,
        #    default auto mode preserves the human developer's principal (`human@example.com`) rather than
        #    silently swapping to `vm-sa@...`, while Cloud Run / GKE (`K_SERVICE`) or explicit
        #    `CC_AGY_VERTEX_OIDC_SOURCE=metadata` selects the workload SA token (`meta_jwt`).
        clear_oidc_token_cache()
        meta_jwt = _make_jwt(
            {"alg": "RS256", "typ": "JWT"},
            {"aud": "https://gw.a.run.app", "email": "vm-sa@proj.iam.gserviceaccount.com", "exp": int(time.time()) + 3600},
        )
        user_jwt = _make_jwt(
            {"alg": "RS256", "typ": "JWT"},
            {"aud": "32555940559.apps.googleusercontent.com", "email": "human@example.com", "exp": int(time.time()) + 3600},
        )
        gdir = tmp_path / "gbin"
        gdir.mkdir()
        gcloud_bin = gdir / "gcloud"
        gcloud_bin.write_text(
            "#!/usr/bin/env bash\n"
            'for a in "$@"; do\n'
            '  case "$a" in\n'
            '    --audiences=*)\n'
            '      echo "ERROR: Invalid account type for `--audiences`." >&2\n'
            '      exit 1\n'
            '      ;;\n'
            '  esac\n'
            'done\n'
            f"printf '%s\\n' '{user_jwt}'\n",
            encoding="utf-8",
        )
        gcloud_bin.chmod(0o755)
        with patch("server.regions._fetch_metadata_identity_token", return_value=meta_jwt) as meta_mock:
            picked_dev = resolve_vertex_base_url_token(
                env={
                    "PATH": f"{gdir}:{os.environ.get('PATH', '')}",
                    "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                    "CC_AGY_VERTEX_OIDC": "1",
                }
            )
            assert picked_dev == user_jwt
            assert meta_mock.call_count == 0

            picked_meta_explicit = resolve_vertex_base_url_token(
                env={
                    "PATH": f"{gdir}:{os.environ.get('PATH', '')}",
                    "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                    "CC_AGY_VERTEX_OIDC": "1",
                    "CC_AGY_VERTEX_OIDC_SOURCE": "metadata",
                }
            )
            assert picked_meta_explicit == meta_jwt

            clear_oidc_token_cache()
            picked_cloud_run = resolve_vertex_base_url_token(
                env={
                    "PATH": f"{gdir}:{os.environ.get('PATH', '')}",
                    "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                    "CC_AGY_VERTEX_OIDC": "1",
                    "K_SERVICE": "worker-svc",
                }
            )
            assert picked_cloud_run == meta_jwt

        # 6. Generic gcloud auth error that echoes `--audiences` does NOT trigger user-account fallback
        clear_oidc_token_cache()
        gdir2 = tmp_path / "gbin2"
        gdir2.mkdir()
        gcloud_err = gdir2 / "gcloud"
        gcloud_err.write_text(
            "#!/usr/bin/env bash\n"
            'for a in "$@"; do\n'
            '  case "$a" in\n'
            '    --audiences=*)\n'
            '      echo "ERROR: Reauthentication failed while running --audiences=https://gw.a.run.app" >&2\n'
            '      exit 1\n'
            '      ;;\n'
            '  esac\n'
            'done\n'
            f"printf '%s\\n' '{user_jwt}'\n",
            encoding="utf-8",
        )
        gcloud_err.chmod(0o755)
        with patch("server.regions._fetch_metadata_identity_token", side_effect=RuntimeError("no metadata")):
            with pytest.raises(RuntimeError, match="Reauthentication failed"):
                resolve_vertex_base_url_token(
                    env={
                        "PATH": f"{gdir2}:{os.environ.get('PATH', '')}",
                        "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                        "CC_AGY_VERTEX_OIDC": "1",
                    }
                )

    def test_telemetry_delta_tokens_concurrent_lock_status_transition_iso_timestamp_and_nan_jwt(
        self, tmp_path: Path
    ) -> None:
        sink = tmp_path / "delta.ndjson"
        env = {"CC_AGY_TELEMETRY_SINK": str(sink)}

        # 1. Turn 1 -> Turn 2 emits delta cost AND delta tokens + cumulative_usage + ISO timestamp, no stale 'applied'
        assert (
            emit_receipt_telemetry(
                {
                    "job_id": "job-delta-1",
                    "status": "ok",
                    "created_at_epoch": 1760000100.5,
                    "cost": {"total_cost_cents": 0.25, "pricing_known": True},
                    "usage": {"prompt_token_count": 800, "candidates_token_count": 200, "total_token_count": 1000},
                },
                env=env,
            )
            is True
        )
        assert (
            emit_receipt_telemetry(
                {
                    "job_id": "job-delta-1",
                    "status": "ok",
                    "created_at_epoch": 1760000100.5,
                    "cost": {"total_cost_cents": 0.60, "pricing_known": True},
                    "usage": {"prompt_token_count": 1900, "candidates_token_count": 600, "total_token_count": 2500},
                },
                env=env,
            )
            is True
        )
        # Status transition ok -> error at unchanged spend is NOT dropped (emits 0-delta transition)
        assert (
            emit_receipt_telemetry(
                {
                    "job_id": "job-delta-1",
                    "status": "error",
                    "created_at_epoch": 1760000100.5,
                    "cost": {"total_cost_cents": 0.60, "pricing_known": True},
                    "usage": {"prompt_token_count": 1900, "candidates_token_count": 600, "total_token_count": 2500},
                },
                env=env,
            )
            is True
        )

        rows = [json.loads(ln) for ln in sink.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 3
        assert rows[0]["timestamp"].endswith("Z")
        assert rows[0]["created_at_epoch"] == pytest.approx(1760000100.5)
        assert "applied" not in rows[0]
        assert [r["cost_cents"] for r in rows] == pytest.approx([0.25, 0.35, 0.0])
        assert [r["usage"]["total_token_count"] for r in rows] == [1000, 1500, 0]
        assert rows[1]["cumulative_usage"]["total_token_count"] == 2500
        assert rows[2]["status"] == "error"
        assert sum(r["cost_cents"] for r in rows) == pytest.approx(0.60)
        assert sum(r["usage"]["total_token_count"] for r in rows) == 2500

        # 2. Concurrent 8-thread race on same job_id emits once (no check-then-act double-spend)
        race_sink = tmp_path / "race.ndjson"
        race_env = {"CC_AGY_TELEMETRY_SINK": str(race_sink)}
        barrier = threading.Barrier(8)
        emitted_flags: list[bool] = []

        def _worker(idx: int) -> None:
            barrier.wait()
            res = emit_receipt_telemetry(
                {
                    "job_id": "job-race-1",
                    "status": "ok",
                    "applied": bool(idx % 2),
                    "cost": {"total_cost_cents": 0.40, "pricing_known": True},
                    "usage": {"total_token_count": 1000},
                },
                env=race_env,
            )
            emitted_flags.append(res)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert emitted_flags.count(True) == 1
        race_rows = [json.loads(ln) for ln in race_sink.read_text(encoding="utf-8").splitlines()]
        assert len(race_rows) == 1
        assert sum(r["cost_cents"] for r in race_rows) == pytest.approx(0.40)

        # 3. NaN / Infinity exp or nbf in JWT payload rejected, and IPv4-compatible IPv6 [::169.254.169.254] blocked
        def _raw_jwt(payload_json: str) -> str:
            h = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode("ascii")
            p = base64.urlsafe_b64encode(payload_json.encode("utf-8")).rstrip(b"=").decode("ascii")
            s = base64.urlsafe_b64encode(b"x" * 64).rstrip(b"=").decode("ascii")
            return f"{h}.{p}.{s}"

        for bad_payload in (
            '{"email":"nan@example.com","exp":NaN}',
            '{"email":"inf@example.com","exp":Infinity}',
            '{"email":"nbf@example.com","exp":9999999999,"nbf":NaN}',
        ):
            ident = resolve_telemetry_identity(bearer_token=_raw_jwt(bad_payload), env={})
            assert ident["identity_source"] == "anonymous"

        with pytest.raises(ValueError, match="blocked"):
            validate_webhook_sink_url("https://[::169.254.169.254]/hook", check_dns=False)

        # 4. doctor.py exposes telemetry_sink_valid + telemetry_sink_warning
        ws = tmp_path / "docws"
        ws.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        doc_ok = run_doctor(
            ws,
            env={
                "GEMINI_API_KEY": "AIzaSyTestKey12345678901234567890",
                "CC_AGY_TELEMETRY_SINK": str(tmp_path / "valid.ndjson"),
            },
        )
        assert doc_ok["user_controls"]["telemetry_sink_valid"] is True
        doc_bad = run_doctor(
            ws,
            env={
                "GEMINI_API_KEY": "AIzaSyTestKey12345678901234567890",
                "CC_AGY_TELEMETRY_SINK": "relative/bad.ndjson",
            },
        )
        assert doc_bad["user_controls"]["telemetry_sink_valid"] is False
        assert doc_bad["user_controls"]["telemetry_sink_warning"]

    def test_true_cost_historical_message_date_and_lead_rate_user_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from server.true_cost import collect_lead_usage

        tdir = tmp_path / "transcripts"
        tdir.mkdir()
        sess_promo = tdir / "promo.jsonl"
        sess_promo.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-12-15T12:00:00Z",
                    "message": {
                        "id": "msg-promo-1",
                        "model": "claude-sonnet-5-5-20261001",
                        "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        sess_post = tdir / "post.jsonl"
        sess_post.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2027-01-05T12:00:00Z",
                    "message": {
                        "id": "msg-post-1",
                        "model": "claude-sonnet-5-5-20261001",
                        "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        # Even if run in 2027, the 2026-12-15 message keeps the $2/1M promo rate (200 cents)
        # while the 2027-01-05 message gets the $3/1M standard rate (300 cents) -> total 500 cents
        monkeypatch.delenv("CC_AGY_LEAD_RATES", raising=False)
        res = collect_lead_usage(tmp_path, transcripts_dir=tdir)
        assert res["lead_cost_cents"] == pytest.approx(500.0)

        # User CC_AGY_LEAD_RATES overrides are NEVER shadowed by longer un-overridden default keys
        monkeypatch.setenv("CC_AGY_LEAD_RATES", json.dumps({"opus-4": [9, 0.9, 11.25, 18, 45]}))
        assert rate_for_model("claude-opus-4-8") == (9.0, 0.9, 11.25, 18.0, 45.0)

        monkeypatch.setenv("CC_AGY_LEAD_RATES", json.dumps({"4-opus": [9, 0.9, 11.25, 18, 45]}))
        assert rate_for_model("claude-4-opus") == (9.0, 0.9, 11.25, 18.0, 45.0)

        monkeypatch.setenv("CC_AGY_LEAD_RATES", json.dumps({"sonnet": [7, 0.7, 8.75, 14, 35]}))
        assert rate_for_model("claude-3-5-sonnet-20241022") == (7.0, 0.7, 8.75, 14.0, 35.0)
        assert rate_for_model("claude-3-7-sonnet-20250219") == (7.0, 0.7, 8.75, 14.0, 35.0)
        assert rate_for_model("claude-sonnet-5-5") == (7.0, 0.7, 8.75, 14.0, 35.0)


class TestGauntletRound3Bugs:
    """RED-first regression tests for Round 3 CLI doctor, OIDC validation/caching, and telemetry schema."""

    def setup_method(self) -> None:
        clear_oidc_token_cache()
        clear_telemetry_dedup_cache()
        flush_telemetry()

    def test_install_doctor_cli_flag_and_module_main(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        base_env = {
            **os.environ,
            "HOME": str(tmp_path),
            "CC_AGY_DATA_DIR": str(tmp_path / "data"),
            "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com",
            "CC_AGY_VERTEX_BASE_URL_TOKEN": "static-vkey-ok",
        }
        proc = subprocess.run(
            [sys.executable, "-m", "server.install", "--doctor"],
            cwd=str(repo_root),
            env=base_env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        report = json.loads(proc.stdout)
        assert report["healthy"] is True
        assert report["auth"]["custom_vertex_gateway"] is True

        proc_mod = subprocess.run(
            [sys.executable, "-m", "server.doctor"],
            cwd=str(repo_root),
            env=base_env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert proc_mod.returncode == 0, proc_mod.stderr
        assert json.loads(proc_mod.stdout)["healthy"] is True

        bad_env = {**base_env, "CC_AGY_TELEMETRY_SINK": "relative/bad.ndjson"}
        proc_bad = subprocess.run(
            [sys.executable, "-m", "server.install", "--doctor"],
            cwd=str(repo_root),
            env=bad_env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert proc_bad.returncode == 1
        assert json.loads(proc_bad.stdout)["healthy"] is False

    def test_oidc_rejects_non_jwt_gcloud_and_mismatched_aud_and_caps_min_remaining_above_lifetime(
        self, tmp_path: Path
    ) -> None:
        # 1. gcloud returning a 2-dot OAuth2 access token (`oauth2_access.c.xxx`) or opaque string (`OK`) is REJECTED (never cached)
        gdir = tmp_path / "g_bad"
        gdir.mkdir()
        gcloud_bad = gdir / "gcloud"
        gcloud_bad.write_text(
            "#!/usr/bin/env bash\nprintf 'oauth2_access.c.b0Aaekm1K_two_dot_oauth2_access_token\\n'\n",
            encoding="utf-8",
        )
        gcloud_bad.chmod(0o755)
        with patch("server.regions._fetch_metadata_identity_token", side_effect=RuntimeError("no metadata")):
            with pytest.raises(RuntimeError, match="JWT|OIDC"):
                resolve_vertex_base_url_token(
                    env={
                        "PATH": f"{gdir}:{os.environ.get('PATH', '')}",
                        "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com",
                        "CC_AGY_VERTEX_OIDC": "1",
                    }
                )

        # 2. authorized_user credentials file yielding an id_token with wrong `aud` (e.g. OAuth client ID)
        #    is rejected on the GAC path, and falls through cleanly to gcloud when gcloud is available!
        auth_user_file = tmp_path / "adc_user.json"
        auth_user_file.write_text(
            json.dumps(
                {
                    "type": "authorized_user",
                    "client_id": "32555940559.apps.googleusercontent.com",
                    "client_secret": "csec",
                    "refresh_token": "rtok",
                }
            ),
            encoding="utf-8",
        )
        wrong_aud_id_tok = _make_jwt(
            {"alg": "RS256", "typ": "JWT"},
            {"aud": "32555940559.apps.googleusercontent.com", "email": "dev@example.com", "exp": int(time.time()) + 3600},
        )

        class _WrongAudUserCreds:
            def __init__(self) -> None:
                self.token = "oauth2_access.c.access"
                self.id_token = wrong_aud_id_tok

            def refresh(self, req: Any) -> None:
                self.id_token = wrong_aud_id_tok

        with (
            patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=_WrongAudUserCreds()),
            patch("server.regions._fetch_metadata_identity_token", side_effect=RuntimeError("no metadata")),
        ):
            with pytest.raises(RuntimeError):
                resolve_vertex_base_url_token(
                    env={
                        "GOOGLE_APPLICATION_CREDENTIALS": str(auth_user_file),
                        "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com",
                        "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://gw.example.com",
                        "PATH": "/nonexistent",
                    }
                )

        # When gcloud IS on PATH alongside an authorized_user ADC file, gcloud fallback succeeds!
        valid_gw_jwt = _make_jwt(
            {"alg": "RS256", "typ": "JWT"},
            {"aud": "https://gw.example.com", "email": "dev@example.com", "exp": int(time.time()) + 3600},
        )
        gdir_ok = tmp_path / "g_ok"
        gdir_ok.mkdir()
        counter = tmp_path / "mint_count.txt"
        gcloud_ok = gdir_ok / "gcloud"
        gcloud_ok.write_text(
            f"#!/usr/bin/env bash\necho 1 >> {counter}\nprintf '%s\\n' '{valid_gw_jwt}'\n",
            encoding="utf-8",
        )
        gcloud_ok.chmod(0o755)
        with patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=_WrongAudUserCreds()):
            tok = resolve_vertex_base_url_token(
                env={
                    "GOOGLE_APPLICATION_CREDENTIALS": str(auth_user_file),
                    "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com",
                    "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://gw.example.com",
                    "PATH": f"{gdir_ok}:{os.environ.get('PATH', '')}",
                }
            )
            assert tok == valid_gw_jwt

        # 3. `min_remaining_seconds` above token lifetime (e.g. 3601s on a 3600s token) does NOT thrash cache
        clear_oidc_token_cache()
        counter.write_text("", encoding="utf-8")
        env_long = {
            "PATH": f"{gdir_ok}:{os.environ.get('PATH', '')}",
            "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com",
            "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://gw.example.com",
        }
        for _ in range(3):
            assert resolve_vertex_base_url_token(env=env_long, min_remaining_seconds=3601.0) == valid_gw_jwt
        assert len(counter.read_text(encoding="utf-8").splitlines()) == 1

        # Doctor surfaces cached OIDC principal email without making network calls
        from server.doctor import run_doctor

        doc = run_doctor(workspace_root=tmp_path, env={**env_long, "CC_AGY_DATA_DIR": str(tmp_path / "d")})
        assert doc["auth"].get("gateway_oidc_principal") == "dev@example.com"

    def test_telemetry_consistent_cumulative_schema_event_id_and_lead_rates_json_alias(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from server.true_cost import rate_for_model

        sink = tmp_path / "r3_events.ndjson"
        env = {"CC_AGY_TELEMETRY_SINK": str(sink)}
        assert emit_receipt_telemetry(
            {
                "job_id": "job-r3-schema",
                "status": "ok",
                "cost": {"total_cost_cents": 0.42, "pricing_known": True},
                "usage": {"prompt_token_count": 500, "candidates_token_count": 100, "total_token_count": 600},
            },
            env=env,
        )
        rows = [json.loads(ln) for ln in sink.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["cumulative_cost_cents"] == pytest.approx(0.42)
        assert rows[0]["cumulative_usage"]["total_token_count"] == 600
        assert isinstance(rows[0].get("event_id"), str) and len(rows[0]["event_id"]) == 24

        # Simulate MCP restart: re-emitting the exact same completed receipt yields the exact same deterministic event_id
        clear_telemetry_dedup_cache()
        sink2 = tmp_path / "r3_events2.ndjson"
        emit_receipt_telemetry(
            {
                "job_id": "job-r3-schema",
                "status": "ok",
                "cost": {"total_cost_cents": 0.42, "pricing_known": True},
                "usage": {"prompt_token_count": 500, "candidates_token_count": 100, "total_token_count": 600},
            },
            env={"CC_AGY_TELEMETRY_SINK": str(sink2)},
        )
        rows2 = [json.loads(ln) for ln in sink2.read_text(encoding="utf-8").splitlines()]
        assert rows2[0]["event_id"] == rows[0]["event_id"]

        # CC_AGY_LEAD_RATES_JSON is honored as a fallback alias when CC_AGY_LEAD_RATES is unset
        monkeypatch.delenv("CC_AGY_LEAD_RATES", raising=False)
        monkeypatch.setenv("CC_AGY_LEAD_RATES_JSON", json.dumps({"opus": [5.0, 0.5, 6.25, 10.0, 25.0]}))
        assert rate_for_model("claude-opus-5-5") == (5.0, 0.5, 6.25, 10.0, 25.0)


class TestGauntletRound4Bugs:
    """RED-first regression tests for Round 4 OIDC principal safety, post-restart telemetry, and CLI polish."""

    def setup_method(self) -> None:
        clear_oidc_token_cache()
        clear_telemetry_dedup_cache()

    def teardown_method(self) -> None:
        clear_oidc_token_cache()
        clear_telemetry_dedup_cache()

    def test_expired_human_gcloud_session_on_gce_does_not_swap_to_vm_metadata(
        self, tmp_path: Path
    ) -> None:
        now = int(time.time())
        user_jwt = _make_jwt(
            {"alg": "RS256"},
            {
                "iss": "https://accounts.google.com",
                "sub": "111",
                "email": "dev@example.com",
                "aud": "32555940559.apps.googleusercontent.com",
                "exp": now + 3600,
            },
        )
        meta_jwt = _make_jwt(
            {"alg": "RS256"},
            {
                "iss": "https://accounts.google.com",
                "sub": "222",
                "email": "vm-default-sa@proj.iam.gserviceaccount.com",
                "aud": "https://gw.a.run.app",
                "exp": now + 3600,
            },
        )

        # Case A: active human login fails `--audiences` with Invalid account type, then fails user print with reauth
        gdir_reauth = tmp_path / "g_reauth"
        gdir_reauth.mkdir()
        gbin_reauth = gdir_reauth / "gcloud"
        gbin_reauth.write_text(
            "#!/usr/bin/env bash\n"
            'for a in "$@"; do\n'
            '  case "$a" in\n'
            '    --audiences=*)\n'
            '      echo "ERROR: Invalid account type for `--audiences`." >&2\n'
            '      exit 1\n'
            '      ;;\n'
            '  esac\n'
            'done\n'
            'echo "ERROR: (gcloud.auth.print-identity-token) Reauthentication failed." >&2\n'
            'exit 1\n',
            encoding="utf-8",
        )
        gbin_reauth.chmod(0o755)
        with patch("server.regions._fetch_metadata_identity_token", return_value=meta_jwt) as meta_mock:
            with pytest.raises(RuntimeError, match="gcloud auth login"):
                resolve_vertex_base_url_token(
                    env={
                        "PATH": f"{gdir_reauth}:{os.environ.get('PATH', '')}",
                        "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                        "CC_AGY_VERTEX_OIDC": "1",
                    }
                )
            assert meta_mock.call_count == 0

        # Case B: first `--audiences` call itself fails with Reauthentication / invalid_grant
        clear_oidc_token_cache()
        gdir_first = tmp_path / "g_first_reauth"
        gdir_first.mkdir()
        gbin_first = gdir_first / "gcloud"
        gbin_first.write_text(
            "#!/usr/bin/env bash\n"
            'echo "ERROR: (gcloud.auth.print-identity-token) RefreshError: invalid_grant: Reauthentication required." >&2\n'
            'exit 1\n',
            encoding="utf-8",
        )
        gbin_first.chmod(0o755)
        with patch("server.regions._fetch_metadata_identity_token", return_value=meta_jwt) as meta_mock2:
            with pytest.raises(RuntimeError, match="invalid_grant"):
                resolve_vertex_base_url_token(
                    env={
                        "PATH": f"{gdir_first}:{os.environ.get('PATH', '')}",
                        "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                        "CC_AGY_VERTEX_OIDC": "1",
                    }
                )
            assert meta_mock2.call_count == 0

        # Case C: Kubernetes-hosted dev pod (KUBERNETES_SERVICE_HOST set, no K_SERVICE) preserves healthy human login
        clear_oidc_token_cache()
        gdir_ok = tmp_path / "g_user_ok"
        gdir_ok.mkdir()
        gbin_ok = gdir_ok / "gcloud"
        gbin_ok.write_text(
            "#!/usr/bin/env bash\n"
            'for a in "$@"; do\n'
            '  case "$a" in\n'
            '    --audiences=*) echo "ERROR: Invalid account type for `--audiences`." >&2; exit 1 ;;\n'
            '  esac\n'
            'done\n'
            f"printf '%s\\n' '{user_jwt}'\n",
            encoding="utf-8",
        )
        gbin_ok.chmod(0o755)
        with patch("server.regions._fetch_metadata_identity_token", return_value=meta_jwt) as meta_mock3:
            picked = resolve_vertex_base_url_token(
                env={
                    "PATH": f"{gdir_ok}:{os.environ.get('PATH', '')}",
                    "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                    "CC_AGY_VERTEX_OIDC": "1",
                    "KUBERNETES_SERVICE_HOST": "10.96.0.1",
                }
            )
            assert picked == user_jwt
            assert meta_mock3.call_count == 0

        # Case D: Headless GCE/GKE VM with gcloud installed on PATH but NO active account falls back to metadata
        # (uses real Google Cloud SDK NoActiveAccountException multi-line stderr including 'gcloud auth login' footer)
        clear_oidc_token_cache()
        gdir_none = tmp_path / "g_no_acct"
        gdir_none.mkdir()
        gbin_none = gdir_none / "gcloud"
        gbin_none.write_text(
            "#!/usr/bin/env bash\n"
            "cat >&2 <<'EOF'\n"
            "ERROR: (gcloud.auth.print-identity-token) You do not currently have an active account selected.\n"
            "Please run:\n"
            "\n"
            "  $ gcloud auth login\n"
            "\n"
            "to obtain new credentials.\n"
            "If you have already logged in with a different account, run:\n"
            "\n"
            "  $ gcloud config set account ACCOUNT\n"
            "EOF\n"
            "exit 1\n",
            encoding="utf-8",
        )
        gbin_none.chmod(0o755)
        with patch("server.regions._fetch_metadata_identity_token", return_value=meta_jwt) as meta_mock4:
            picked_headless = resolve_vertex_base_url_token(
                env={
                    "PATH": f"{gdir_none}:{os.environ.get('PATH', '')}",
                    "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                    "CC_AGY_VERTEX_OIDC": "1",
                    "KUBERNETES_SERVICE_HOST": "10.96.0.1",
                }
            )
            assert picked_headless == meta_jwt
            assert meta_mock4.call_count == 1

    def test_telemetry_file_sink_recovers_delta_across_process_restart(
        self, tmp_path: Path
    ) -> None:
        sink = tmp_path / "restart_events.ndjson"
        env = {"CC_AGY_TELEMETRY_SINK": str(sink)}

        assert emit_receipt_telemetry(
            {
                "job_id": "job-r4-restart",
                "status": "ok",
                "cost": {"total_cost_cents": 0.25, "pricing_known": True},
                "usage": {"prompt_token_count": 100, "candidates_token_count": 50, "total_token_count": 150},
            },
            env=env,
        )
        assert emit_receipt_telemetry(
            {
                "job_id": "job-r4-restart",
                "status": "applied",
                "cost": {"total_cost_cents": 0.40, "pricing_known": True},
                "usage": {"prompt_token_count": 160, "candidates_token_count": 80, "total_token_count": 240},
            },
            env=env,
        )

        # Simulate process restart: in-memory caches wiped, file sink still on disk
        clear_telemetry_dedup_cache()

        # Re-emitting identical post-restart state is deduplicated against the file sink tail
        assert (
            emit_receipt_telemetry(
                {
                    "job_id": "job-r4-restart",
                    "status": "applied",
                    "cost": {"total_cost_cents": 0.40, "pricing_known": True},
                    "usage": {"prompt_token_count": 160, "candidates_token_count": 80, "total_token_count": 240},
                },
                env=env,
            )
            is False
        )
        # Further progress after restart emits only the new delta (0.15) while keeping cumulative 0.55
        assert emit_receipt_telemetry(
            {
                "job_id": "job-r4-restart",
                "status": "applied",
                "cost": {"total_cost_cents": 0.55, "pricing_known": True},
                "usage": {"prompt_token_count": 220, "candidates_token_count": 110, "total_token_count": 330},
            },
            env=env,
        )
        rows = [json.loads(ln) for ln in sink.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 3
        assert [(r["cost_cents"], r["cumulative_cost_cents"]) for r in rows] == [
            (pytest.approx(0.25), pytest.approx(0.25)),
            (pytest.approx(0.15), pytest.approx(0.40)),
            (pytest.approx(0.15), pytest.approx(0.55)),
        ]
        assert sum(r["cost_cents"] for r in rows) == pytest.approx(0.55)
        assert len({r["event_id"] for r in rows}) == 3

    def test_doctor_probe_oidc_userinfo_rejection_cache_monotonicity_and_ps1_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        from server.regions import _is_cached_oidc_fresh, resolve_vertex_base_url_audience
        from server.true_cost import lead_rates

        # 1. URL userinfo credentials must never leak into derived OIDC audience
        with pytest.raises(ValueError, match="userinfo|credentials"):
            resolve_vertex_base_url_audience(
                None,
                env={
                    "CC_AGY_VERTEX_BASE_URL": "https://user:secret@gw.example.com/v1",
                    "CC_AGY_VERTEX_OIDC": "1",
                },
            )

        # 2. Monotonic cache headroom: 3300s min_rem on a 3600s token stays fresh 240s after mint (3360s left),
        #    refreshes at 600s after mint (3000s left < 3300s needed for ~55m worker job), and caps >TTL callers at ttl-60s
        now = time.time()
        tok_3600 = _make_jwt({"alg": "RS256"}, {"iss": "https://accounts.google.com", "sub": "1", "exp": int(now + 3600)})
        assert _is_cached_oidc_fresh(tok_3600, now + 3300.0, 3600.0, now + 240.0, 3300.0) is True
        assert _is_cached_oidc_fresh(tok_3600, now + 3300.0, 3600.0, now + 600.0, 3300.0) is False
        assert _is_cached_oidc_fresh(tok_3600, now + 3300.0, 3600.0, now + 30.0, 3601.0) is True

        # 3. CC_AGY_LEAD_RATES_JSON parse error names CC_AGY_LEAD_RATES_JSON
        monkeypatch.delenv("CC_AGY_LEAD_RATES", raising=False)
        monkeypatch.setenv("CC_AGY_LEAD_RATES_JSON", "{bad-json")
        with pytest.raises(ValueError, match="CC_AGY_LEAD_RATES_JSON"):
            lead_rates()
        monkeypatch.delenv("CC_AGY_LEAD_RATES_JSON", raising=False)

        # 4. `--doctor` stays 0-subprocess by default (`unminted_until_first_request`), while `--doctor --probe-oidc` mints once
        repo_root = Path(__file__).resolve().parents[1]
        gdir = tmp_path / "g_probe"
        gdir.mkdir()
        counter = tmp_path / "probe_count.txt"
        counter.write_text("", encoding="utf-8")
        probe_jwt = _make_jwt(
            {"alg": "RS256"},
            {
                "iss": "https://accounts.google.com",
                "sub": "99",
                "email": "probed-user@example.com",
                "aud": "https://gw.example.com",
                "exp": int(now + 3600),
            },
        )
        gbin = gdir / "gcloud"
        gbin.write_text(
            "#!/usr/bin/env bash\n"
            f"echo 1 >> '{counter}'\n"
            f"printf '%s\\n' '{probe_jwt}'\n",
            encoding="utf-8",
        )
        gbin.chmod(0o755)
        doc_env = {
            **os.environ,
            "PATH": f"{gdir}:{os.environ.get('PATH', '')}",
            "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com",
            "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://gw.example.com",
            "CC_AGY_DATA_DIR": str(tmp_path / "doc_data"),
        }
        proc_default = subprocess.run(
            [sys.executable, "-m", "server.install", "--doctor"],
            cwd=str(repo_root),
            env=doc_env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert proc_default.returncode == 0
        rep_default = json.loads(proc_default.stdout)
        assert rep_default["auth"].get("gateway_oidc_principal_status") == "unminted_until_first_request"
        assert len(counter.read_text(encoding="utf-8").splitlines()) == 0

        proc_probe = subprocess.run(
            [sys.executable, "-m", "server.install", "--doctor", "--probe-oidc"],
            cwd=str(repo_root),
            env=doc_env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert proc_probe.returncode == 0
        rep_probe = json.loads(proc_probe.stdout)
        assert rep_probe["auth"].get("gateway_oidc_principal") == "probed-user@example.com"
        assert rep_probe["auth"].get("gateway_oidc_principal_status") == "probed"
        assert len(counter.read_text(encoding="utf-8").splitlines()) == 1

        # 5. PowerShell bootstrap guards $GatewayAudience before overwriting shell env
        ps1_text = (repo_root / "scripts" / "bootstrap-gateway.ps1").read_text(encoding="utf-8")
        assert "if ($GatewayAudience)" in ps1_text


class TestGauntletRound5Bugs:
    """RED-first regression tests for Round 5 gauntlet findings (gcloud footer, impersonate_sa guard, static_token doctor)."""

    def test_scopeless_gce_token_refresh_falls_back_while_human_refresh_fails_closed(
        self, tmp_path: Path
    ) -> None:
        from server.regions import clear_oidc_token_cache, resolve_vertex_base_url_token

        now = int(time.time())
        meta_jwt = _make_jwt(
            {"alg": "RS256"},
            {
                "iss": "https://accounts.google.com",
                "sub": "222",
                "email": "vm-default-sa@proj.iam.gserviceaccount.com",
                "aud": "https://gw.a.run.app",
                "exp": now + 3600,
            },
        )

        # 1. Scopeless GCE VM where gcloud tries metadata access token and prints Google Cloud SDK's relogin footer
        clear_oidc_token_cache()
        gdir_scopeless = tmp_path / "g_scopeless"
        gdir_scopeless.mkdir()
        gbin_scopeless = gdir_scopeless / "gcloud"
        gbin_scopeless.write_text(
            "#!/usr/bin/env bash\n"
            "cat >&2 <<'EOF'\n"
            "ERROR: (gcloud.auth.print-identity-token) There was a problem refreshing your current auth tokens: "
            "('Failed to retrieve http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token?scopes=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.email "
            "from the Google Compute Engine metadata service. Compute Engine Metadata server unavailable or \"No service account scopes specified.\"')\n"
            "Please run:\n"
            "\n"
            "  $ gcloud auth login\n"
            "\n"
            "to obtain new credentials.\n"
            "EOF\n"
            "exit 1\n",
            encoding="utf-8",
        )
        gbin_scopeless.chmod(0o755)
        with patch("server.regions._fetch_metadata_identity_token", return_value=meta_jwt) as meta_mock:
            tok = resolve_vertex_base_url_token(
                env={
                    "PATH": f"{gdir_scopeless}:{os.environ.get('PATH', '')}",
                    "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                    "CC_AGY_VERTEX_OIDC": "1",
                }
            )
            assert tok == meta_jwt
            assert meta_mock.call_count == 1

        # 2. Human account refresh failure with relogin footer still fails closed (never swaps to VM metadata SA)
        clear_oidc_token_cache()
        gdir_human_err = tmp_path / "g_human_err"
        gdir_human_err.mkdir()
        gbin_human_err = gdir_human_err / "gcloud"
        gbin_human_err.write_text(
            "#!/usr/bin/env bash\n"
            "cat >&2 <<'EOF'\n"
            "ERROR: (gcloud.auth.print-identity-token) There was a problem refreshing auth tokens for account dev@example.com: "
            "('invalid_grant: Bad Request', {'error': 'invalid_grant'})\n"
            "Please run:\n"
            "\n"
            "  $ gcloud auth login\n"
            "\n"
            "to obtain new credentials.\n"
            "EOF\n"
            "exit 1\n",
            encoding="utf-8",
        )
        gbin_human_err.chmod(0o755)
        with patch("server.regions._fetch_metadata_identity_token", return_value=meta_jwt) as meta_mock2:
            with pytest.raises(RuntimeError, match="Failed to mint OIDC identity token"):
                resolve_vertex_base_url_token(
                    env={
                        "PATH": f"{gdir_human_err}:{os.environ.get('PATH', '')}",
                        "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                        "CC_AGY_VERTEX_OIDC": "1",
                    }
                )
            assert meta_mock2.call_count == 0

    def test_impersonate_service_account_never_swaps_to_unmatched_metadata_principal(
        self, tmp_path: Path
    ) -> None:
        from server.regions import (
            _OIDC_TOKEN_CACHE,
            clear_oidc_token_cache,
            resolve_vertex_base_url_token,
        )

        now = int(time.time())
        wrong_meta_jwt = _make_jwt(
            {"alg": "RS256"},
            {
                "iss": "https://accounts.google.com",
                "sub": "222",
                "email": "runtime-sa@proj.iam.gserviceaccount.com",
                "aud": "https://gw.a.run.app",
                "exp": now + 3600,
            },
        )
        matching_meta_jwt = _make_jwt(
            {"alg": "RS256"},
            {
                "iss": "https://accounts.google.com",
                "sub": "333",
                "email": "target-sa@proj.iam.gserviceaccount.com",
                "aud": "https://gw.a.run.app",
                "exp": now + 3600,
            },
        )

        # A2: gcloud impersonation fails due to expired user reauth -> fail closed (0 metadata calls)
        clear_oidc_token_cache()
        gdir_reauth = tmp_path / "g_imp_reauth"
        gdir_reauth.mkdir()
        gbin_reauth = gdir_reauth / "gcloud"
        gbin_reauth.write_text(
            "#!/usr/bin/env bash\n"
            'echo "ERROR: (gcloud.auth.print-identity-token) Reauthentication required." >&2\n'
            "exit 1\n",
            encoding="utf-8",
        )
        gbin_reauth.chmod(0o755)
        with patch("server.regions._fetch_metadata_identity_token", return_value=wrong_meta_jwt) as meta_a2:
            with pytest.raises(RuntimeError, match="Failed to mint OIDC identity token"):
                resolve_vertex_base_url_token(
                    env={
                        "PATH": f"{gdir_reauth}:{os.environ.get('PATH', '')}",
                        "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                        "CC_AGY_VERTEX_OIDC": "1",
                        "CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT": "target-sa@proj.iam.gserviceaccount.com",
                    }
                )
            assert meta_a2.call_count == 0
            assert len(_OIDC_TOKEN_CACHE) == 0

        # A2b: gcloud impersonation fails with IAM 403 -> fail closed (0 metadata calls)
        clear_oidc_token_cache()
        gdir_403 = tmp_path / "g_imp_403"
        gdir_403.mkdir()
        gbin_403 = gdir_403 / "gcloud"
        gbin_403.write_text(
            "#!/usr/bin/env bash\n"
            "echo \"ERROR: (gcloud.auth.print-identity-token) Failed to impersonate [target-sa@proj.iam.gserviceaccount.com]. Permission 'iam.serviceAccounts.getOpenIdToken' denied\" >&2\n"
            "exit 1\n",
            encoding="utf-8",
        )
        gbin_403.chmod(0o755)
        with patch("server.regions._fetch_metadata_identity_token", return_value=wrong_meta_jwt) as meta_a2b:
            with pytest.raises(RuntimeError, match="Failed to mint OIDC identity token"):
                resolve_vertex_base_url_token(
                    env={
                        "PATH": f"{gdir_403}:{os.environ.get('PATH', '')}",
                        "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                        "CC_AGY_VERTEX_OIDC": "1",
                        "CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT": "target-sa@proj.iam.gserviceaccount.com",
                    }
                )
            assert meta_a2b.call_count == 0
            assert len(_OIDC_TOKEN_CACHE) == 0

        # G1: Managed Cloud Run with NO gcloud on PATH and CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT set:
        # rejects wrong runtime-sa metadata token, accepts if runtime-sa matches target-sa
        clear_oidc_token_cache()
        empty_bin = tmp_path / "empty_bin"
        empty_bin.mkdir()
        with patch("server.regions._fetch_metadata_identity_token", return_value=wrong_meta_jwt):
            with pytest.raises(RuntimeError, match="CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT"):
                resolve_vertex_base_url_token(
                    env={
                        "PATH": str(empty_bin),
                        "K_SERVICE": "worker-svc",
                        "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                        "CC_AGY_VERTEX_OIDC": "1",
                        "CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT": "target-sa@proj.iam.gserviceaccount.com",
                    }
                )
            assert len(_OIDC_TOKEN_CACHE) == 0

        with patch("server.regions._fetch_metadata_identity_token", return_value=matching_meta_jwt):
            matched_tok = resolve_vertex_base_url_token(
                env={
                    "PATH": str(empty_bin),
                    "K_SERVICE": "worker-svc",
                    "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
                    "CC_AGY_VERTEX_OIDC": "1",
                    "CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT": "target-sa@proj.iam.gserviceaccount.com",
                }
            )
            assert matched_tok == matching_meta_jwt

    def test_doctor_static_token_probe_flag_validation_userinfo_redaction_and_restart_cache(
        self, tmp_path: Path
    ) -> None:
        import sys

        from server.doctor import run_doctor
        from server.regions import _validate_oidc_audience
        from server.telemetry import _EMITTED_JOB_SPEND

        now = int(time.time())
        static_jwt = _make_jwt(
            {"alg": "RS256"},
            {
                "iss": "https://accounts.google.com",
                "sub": "77",
                "email": "static-caller@example.com",
                "aud": "https://gw.example.com",
                "exp": now + 3600,
            },
        )

        # 1. Static opaque virtual key and static JWT both report gateway_oidc_principal_status == "static_token"
        #    even when --probe-oidc (probe_oidc=True) and CC_AGY_VERTEX_OIDC=1 are set, without invoking gcloud
        with patch("server.regions._mint_oidc_identity_token") as mint_spy:
            rep_opaque = run_doctor(
                workspace_root=tmp_path,
                probe_oidc=True,
                env={
                    "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com",
                    "CC_AGY_VERTEX_BASE_URL_TOKEN": "static-vkey-opaque",
                    "CC_AGY_VERTEX_OIDC": "1",
                    "CC_AGY_DATA_DIR": str(tmp_path / "d1"),
                },
            )
            assert rep_opaque["auth"]["gateway_oidc_principal_status"] == "static_token"
            assert "gateway_oidc_principal" not in rep_opaque["auth"]

            rep_jwt = run_doctor(
                workspace_root=tmp_path,
                probe_oidc=True,
                env={
                    "CC_AGY_VERTEX_BASE_URL": "https://gw.example.com",
                    "CC_AGY_VERTEX_BASE_URL_TOKEN": static_jwt,
                    "CC_AGY_VERTEX_OIDC": "1",
                    "CC_AGY_DATA_DIR": str(tmp_path / "d2"),
                },
            )
            assert rep_jwt["auth"]["gateway_oidc_principal_status"] == "static_token"
            assert rep_jwt["auth"]["gateway_oidc_principal"] == "static-caller@example.com"
            assert mint_spy.call_count == 0

        # 2. `python3 -m server.install --probe-oidc` without `--doctor` exits 2
        repo_root = Path(__file__).resolve().parents[1]
        proc_bad_flag = subprocess.run(
            [sys.executable, "-m", "server.install", "--probe-oidc"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert proc_bad_flag.returncode == 2
        assert "--doctor" in (proc_bad_flag.stderr or "")

        # 3. _validate_oidc_audience rejects userinfo credentials with spaces BEFORE echoing secret in metachar error
        with pytest.raises(ValueError) as exc_info:
            _validate_oidc_audience("https://u:my sec ret@gw.example.com")
        assert "userinfo" in str(exc_info.value).lower()
        assert "my sec ret" not in str(exc_info.value)

        # 4. Post-restart duplicate status poll populates _EMITTED_JOB_SPEND so subsequent polls skip file tail reads
        sink = tmp_path / "cache_recovery.ndjson"
        env = {"CC_AGY_TELEMETRY_SINK": str(sink)}
        receipt = {
            "job_id": "job-r5-cache",
            "status": "ok",
            "cost": {"total_cost_cents": 0.30, "pricing_known": True},
            "usage": {"prompt_token_count": 50, "candidates_token_count": 25, "total_token_count": 75},
        }
        assert emit_receipt_telemetry(receipt, env=env) is True
        clear_telemetry_dedup_cache()
        assert "job-r5-cache" not in _EMITTED_JOB_SPEND
        assert emit_receipt_telemetry(receipt, env=env) is False
        assert "job-r5-cache" in _EMITTED_JOB_SPEND


class TestGauntletRound6Bugs:
    """RED-first regression tests for Round 6 findings (gcloud --include-email on impersonation & OverflowError guards)."""

    def test_gcloud_impersonation_passes_include_email_and_succeeds_end_to_end(
        self, tmp_path: Path
    ) -> None:
        from server.doctor import run_doctor
        from server.regions import clear_oidc_token_cache, resolve_vertex_base_url_token

        now = int(time.time())
        jwt_no_email = _make_jwt(
            {"alg": "RS256"},
            {
                "iss": "https://accounts.google.com",
                "sub": "10987654321",
                "aud": "https://gw.a.run.app",
                "exp": now + 3600,
            },
        )
        jwt_with_email = _make_jwt(
            {"alg": "RS256"},
            {
                "iss": "https://accounts.google.com",
                "sub": "10987654321",
                "email": "target-sa@proj.iam.gserviceaccount.com",
                "email_verified": True,
                "aud": "https://gw.a.run.app",
                "exp": now + 3600,
            },
        )
        wrong_meta_jwt = _make_jwt(
            {"alg": "RS256"},
            {
                "iss": "https://accounts.google.com",
                "sub": "222",
                "email": "vm-default-sa@proj.iam.gserviceaccount.com",
                "aud": "https://gw.a.run.app",
                "exp": now + 3600,
            },
        )

        gdir = tmp_path / "g_imp_real"
        gdir.mkdir()
        argv_log = tmp_path / "gcloud_argv.log"
        gbin = gdir / "gcloud"
        # Matches Google Cloud SDK 592.0.0 surface/auth/print_identity_token.py:
        # --include-email requires --impersonate-service-account, and without --include-email
        # IAM generateIdToken omits `email` from the returned JWT.
        gbin.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' \"$*\" >> '{argv_log}'\n"
            "has_imp=0\n"
            "has_inc=0\n"
            'for a in "$@"; do\n'
            '  case "$a" in\n'
            "    --impersonate-service-account=target-sa@proj.iam.gserviceaccount.com) has_imp=1 ;;\n"
            "    --include-email) has_inc=1 ;;\n"
            "  esac\n"
            "done\n"
            'if [ "$has_inc" -eq 1 ] && [ "$has_imp" -eq 0 ]; then\n'
            '  echo "ERROR: (gcloud.auth.print-identity-token) Invalid account type for `--include-email`. Requires an impersonate service account." >&2\n'
            "  exit 1\n"
            "fi\n"
            'if [ "$has_imp" -eq 1 ] && [ "$has_inc" -eq 1 ]; then\n'
            f"  printf '%s\\n' '{jwt_with_email}'\n"
            "  exit 0\n"
            "fi\n"
            f"printf '%s\\n' '{jwt_no_email}'\n",
            encoding="utf-8",
        )
        gbin.chmod(0o755)

        env = {
            "PATH": f"{gdir}:{os.environ.get('PATH', '')}",
            "CC_AGY_VERTEX_BASE_URL": "https://gw.a.run.app",
            "CC_AGY_VERTEX_OIDC": "1",
            "CC_AGY_VERTEX_IMPERSONATE_SERVICE_ACCOUNT": "target-sa@proj.iam.gserviceaccount.com",
            "CC_AGY_DATA_DIR": str(tmp_path / "doc_imp"),
        }
        clear_oidc_token_cache()
        with patch("server.regions._fetch_metadata_identity_token", return_value=wrong_meta_jwt) as meta_mock:
            minted = resolve_vertex_base_url_token(env=env)
            assert minted == jwt_with_email
            assert meta_mock.call_count == 0

        logged_calls = argv_log.read_text(encoding="utf-8").splitlines()
        assert len(logged_calls) == 1
        assert "--impersonate-service-account=target-sa@proj.iam.gserviceaccount.com" in logged_calls[0]
        assert "--include-email" in logged_calls[0]

        clear_oidc_token_cache()
        rep = run_doctor(workspace_root=tmp_path, probe_oidc=True, env=env)
        assert rep["healthy"] is True
        assert rep["auth"]["gateway_oidc_principal"] == "target-sa@proj.iam.gserviceaccount.com"
        assert rep["auth"]["gateway_oidc_principal_status"] == "probed"

    def test_telemetry_infinity_usage_and_schemeless_userinfo_redaction(
        self, tmp_path: Path
    ) -> None:
        from server.regions import _validate_oidc_audience, resolve_vertex_base_url_audience
        from server.telemetry import _recover_file_sink_job_state

        # 1. Corrupt 1e400 (Infinity) usage value in NDJSON sink must not raise OverflowError
        sink = tmp_path / "inf_usage.ndjson"
        sink.write_text(
            '{"job_id":"job-inf","status":"ok","cumulative_cost_cents":0.25,'
            '"cumulative_usage":{"prompt_token_count":10,"total_token_count":1e400}}\n',
            encoding="utf-8",
        )
        recovered = _recover_file_sink_job_state(str(sink), "job-inf")
        assert recovered is not None
        assert recovered[0] == pytest.approx(0.25)
        assert recovered[1]["prompt_token_count"] == 10
        assert recovered[1]["total_token_count"] == 0

        assert (
            emit_receipt_telemetry(
                {
                    "job_id": "job-inf",
                    "status": "applied",
                    "cost": {"total_cost_cents": 0.40, "pricing_known": True},
                    "usage": {"prompt_token_count": 20, "candidates_token_count": 5, "total_token_count": 25},
                },
                env={"CC_AGY_TELEMETRY_SINK": str(sink)},
            )
            is True
        )

        # 2. Malformed scheme-less audience / base_url with userinfo never echoes embedded secret
        with pytest.raises(ValueError) as exc1:
            _validate_oidc_audience("u:s3cret@gw.example.com")
        assert "userinfo" in str(exc1.value).lower()
        assert "s3cret" not in str(exc1.value)

        with pytest.raises(ValueError) as exc2:
            resolve_vertex_base_url_audience("auto", base_url="u:s3cret@gw.example.com", env={})
        assert "userinfo" in str(exc2.value).lower()
        assert "s3cret" not in str(exc2.value)




