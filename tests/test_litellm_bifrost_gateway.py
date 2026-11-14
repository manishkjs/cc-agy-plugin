"""Tests for first-class LiteLLM (`/vertex_ai` + `/gemini`) and Bifrost (`/genai`) gateway support."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
import threading
import types as py_types
from typing import Any
from urllib.parse import unquote

import pytest

from server.agy_cli import build_agy_subprocess_env
from server.agy_runner import run_antigravity_worker
from server.batch_plan import FileTask
from server.doctor import run_doctor
from server.genai_ask import ask_gemini
from server.genai_client import build_genai_client
from server.install import (
    _validate_gateway_base_url_no_secrets,
    build_agy_server_entry,
    build_claude_user_config,
    build_codex_server_block,
    install_agy,
    install_cursor,
    install_opencode,
    main as install_main,
    preflight,
    run_interactive_wizard,
)
from server.receipts import redact_api_keys
from server.regions import (
    enforce_project_controls,
    resolve_auth_config,
    resolve_effective_gateway_url,
    resolve_gateway_credentials,
    resolve_gateway_wire,
    resolve_vertex_base_url_audience,
)
from server.verify import credential_free_environ
from server.writers import (
    WriterRequest,
    create_shared_cache,
    make_agent_writer,
    make_genai_writer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class _MockGatewayServer:
    """Hermetic loopback HTTP server recording every incoming path, method, header, and JSON body."""

    def __init__(self, reply_text: str = '{"content": "ok = 1\\n"}') -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply_text = reply_text
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or "0")
                raw_body = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
                outer.requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": raw_body,
                    }
                )
                payload = {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": outer.reply_text}],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 42,
                        "candidatesTokenCount": 12,
                        "totalTokenCount": 54,
                    },
                    "modelVersion": "gemini-flash-next",
                }
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> _MockGatewayServer:
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def test_resolve_gateway_wire_auto_detection_and_explicit_overrides() -> None:
    # Auto-detects "gemini" for Bifrost `/genai` and LiteLLM `/gemini` path segments (including `/gemini/v1beta`)
    assert resolve_gateway_wire(base_url="http://localhost:8080/genai", env={}) == "gemini"
    assert resolve_gateway_wire(base_url="https://gw.corp.example.com/proxy/genai/", env={}) == "gemini"
    assert resolve_gateway_wire(base_url="https://litellm.example.com/gemini", env={}) == "gemini"
    assert resolve_gateway_wire(base_url="https://litellm.example.com/gemini/v1beta", env={}) == "gemini"
    assert resolve_gateway_wire(base_url="http://localhost:8080/genai/v1beta", env={}) == "gemini"

    # Auto-detects "gemini" when CC_AGY_GEMINI_BASE_URL / GOOGLE_GEMINI_BASE_URL or CC_AGY_BIFROST_VIRTUAL_KEY (without Vertex base URL) is set
    assert resolve_gateway_wire(env={"CC_AGY_GEMINI_BASE_URL": "https://gw.example.com"}) == "gemini"
    assert resolve_gateway_wire(env={"GOOGLE_GEMINI_BASE_URL": "https://gw.example.com"}) == "gemini"
    assert (
        resolve_gateway_wire(
            base_url="https://bifrost.corp.example.com",
            env={"CC_AGY_BIFROST_VIRTUAL_KEY": "bf-vk-123"},
        )
        == "gemini"
    )
    # Leftover CC_AGY_BIFROST_VIRTUAL_KEY in shell env does NOT flip an explicit CC_AGY_VERTEX_BASE_URL (without /genai) to "gemini"
    assert (
        resolve_gateway_wire(
            env={
                "CC_AGY_VERTEX_BASE_URL": "https://vertex-proxy.example.com",
                "CC_AGY_BIFROST_VIRTUAL_KEY": "bf-vk-leftover",
            }
        )
        == "vertex"
    )

    # Auto-detects "vertex" for LiteLLM `/vertex_ai`, root URLs, and Cloud Code PA
    assert resolve_gateway_wire(base_url="https://litellm.example.com/vertex_ai", env={}) == "vertex"
    assert resolve_gateway_wire(base_url="https://litellm.example.com/vertex_ai/v1", env={}) == "vertex"
    assert resolve_gateway_wire(base_url="https://ai-gateway.example.com", env={}) == "vertex"

    # Explicit overrides win in both directions
    assert (
        resolve_gateway_wire(
            "gemini",
            base_url="https://custom-bifrost.example.com/v1-gw",
            env={},
        )
        == "gemini"
    )
    assert (
        resolve_gateway_wire(
            "vertex",
            base_url="https://weird-proxy.example.com/gemini",
            env={},
        )
        == "vertex"
    )
    assert (
        resolve_gateway_wire(
            base_url="https://gw.example.com",
            env={"CC_AGY_GATEWAY_WIRE": "bifrost"},
        )
        == "gemini"
    )

    with pytest.raises(ValueError, match="Unsupported gateway wire mode"):
        resolve_gateway_wire("openai-chat", env={})


def test_resolve_auth_config_gateway_modes_preserve_existing_vertex_and_api_key_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Existing invariant: GOOGLE_VERTEX_BASE_URL + ambient GEMINI_API_KEY stays auth_mode="vertex" without leaking GEMINI_API_KEY
    cfg_vtx = resolve_auth_config(
        env={
            "GOOGLE_VERTEX_BASE_URL": "https://litellm.example.com/vertex_ai",
            "GEMINI_API_KEY": "AIzaSyAmbientPersonalKey999999",
        }
    )
    assert cfg_vtx["auth_mode"] == "vertex"
    assert cfg_vtx["gateway_wire"] == "vertex"
    assert cfg_vtx["project"] == "custom-endpoint"
    assert not cfg_vtx.get("api_key")

    # 2. Explicit per-call `requested_api_key` and `CC_AGY_AUTH_MODE=api_key` override ambient gateway env vars
    # AND `resolve_effective_gateway_url` + `build_genai_client` never re-scrape os.environ!
    monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL", "http://127.0.0.1:19999/vertex_ai")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "http://127.0.0.1:19998/gemini")
    cfg_explicit_key = resolve_auth_config(
        requested_api_key="AIzaSyExplicitPerCallKey123456",
        env={
            "CC_AGY_VERTEX_BASE_URL": "http://127.0.0.1:19999/vertex_ai",
            "GOOGLE_GEMINI_BASE_URL": "http://127.0.0.1:19998/gemini",
        },
    )
    assert cfg_explicit_key["auth_mode"] == "api_key"
    assert cfg_explicit_key["api_key"] == "AIzaSyExplicitPerCallKey123456"
    assert cfg_explicit_key["base_url"] is None
    assert resolve_effective_gateway_url(cfg_explicit_key) == ""
    direct_client_1 = build_genai_client(cfg_explicit_key)
    assert not direct_client_1._api_client.vertexai
    assert direct_client_1._api_client._http_options.base_url == "https://generativelanguage.googleapis.com/"

    cfg_mode_apikey = resolve_auth_config(
        env={
            "CC_AGY_AUTH_MODE": "api_key",
            "GEMINI_API_KEY": "AIzaSyDirectModeKey123456",
            "CC_AGY_VERTEX_BASE_URL": "http://127.0.0.1:19999/vertex_ai",
            "CC_AGY_GEMINI_BASE_URL": "http://127.0.0.1:19998/gemini",
            "GOOGLE_GEMINI_BASE_URL": "http://127.0.0.1:19997/gemini",
        }
    )
    assert cfg_mode_apikey["auth_mode"] == "api_key"
    assert cfg_mode_apikey["base_url"] is None
    assert resolve_effective_gateway_url(cfg_mode_apikey) == ""
    direct_client_2 = build_genai_client(cfg_mode_apikey)
    assert not direct_client_2._api_client.vertexai
    assert direct_client_2._api_client._http_options.base_url == "https://generativelanguage.googleapis.com/"
    monkeypatch.delenv("CC_AGY_VERTEX_BASE_URL", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)

    # 3. GOOGLE_GEMINI_BASE_URL + GEMINI_API_KEY routes through custom gateway on "gemini" wire
    cfg_gem_gw = resolve_auth_config(
        env={
            "GOOGLE_GEMINI_BASE_URL": "https://litellm.example.com/gemini",
            "GEMINI_API_KEY": "AIzaSyGatewayVirtualKey123456",
        }
    )
    assert cfg_gem_gw["auth_mode"] == "api_key"
    assert cfg_gem_gw["gateway_wire"] == "gemini"
    assert cfg_gem_gw["base_url"] == "https://litellm.example.com/gemini"
    assert cfg_gem_gw["api_key"] == "AIzaSyGatewayVirtualKey123456"

    # 3b. GOOGLE_GEMINI_BASE_URL + CC_AGY_BIFROST_VIRTUAL_KEY (without GEMINI_API_KEY) attaches base_url + gemini wire
    cfg_gem_bf = resolve_auth_config(
        env={
            "GOOGLE_GEMINI_BASE_URL": "https://bifrost.example.com/genai",
            "CC_AGY_BIFROST_VIRTUAL_KEY": "bf-vk-only-key",
        }
    )
    assert cfg_gem_bf["base_url"] == "https://bifrost.example.com/genai"
    assert cfg_gem_bf["gateway_wire"] == "gemini"
    assert cfg_gem_bf["bifrost_virtual_key"] == "bf-vk-only-key"

    # 3c. Bare `GOOGLE_GEMINI_BASE_URL` without credentials or gateway mode raises ValueError (`N6`)
    with pytest.raises(ValueError, match="Missing credentials"):
        resolve_auth_config(env={"GOOGLE_GEMINI_BASE_URL": "https://litellm.example.com/gemini"})
    with pytest.raises(ValueError, match="Missing Google Cloud project ID"):
        resolve_auth_config(
            env={
                "GOOGLE_GEMINI_BASE_URL": "https://litellm.example.com/gemini",
                "CC_AGY_AUTH_MODE": "vertex",
            }
        )

    # 3d. Direct `resolve_gateway_credentials` precedence: static token and Bifrost VK outrank `api_key` (`N1`)
    eff_tok_static, hdrs_static = resolve_gateway_credentials(
        {"api_key": "AIzaSyAmbientPersonalKey", "base_url_token": "sk-static-gateway-key"},
        base_url="https://litellm.example.com/gemini",
        wire="gemini",
        env={},
    )
    assert eff_tok_static == "sk-static-gateway-key"
    assert hdrs_static["Authorization"] == "Bearer sk-static-gateway-key"
    assert hdrs_static["x-goog-api-key"] == "sk-static-gateway-key"
    assert "AIzaSyAmbientPersonalKey" not in json.dumps(hdrs_static)

    eff_tok_bf, hdrs_bf = resolve_gateway_credentials(
        {"api_key": "AIzaSyAmbientPersonalKey", "bifrost_virtual_key": "bf-vk-direct-test"},
        base_url="https://bifrost.example.com/genai",
        wire="gemini",
        env={},
    )
    assert eff_tok_bf == "bf-vk-direct-test"
    assert hdrs_bf["Authorization"] == "Bearer bf-vk-direct-test"
    assert hdrs_bf["x-goog-api-key"] == "bf-vk-direct-test"
    assert hdrs_bf["x-bf-vk"] == "bf-vk-direct-test"
    assert "AIzaSyAmbientPersonalKey" not in json.dumps(hdrs_bf)

    # 4. Bifrost `/genai` via CC_AGY_VERTEX_BASE_URL + CC_AGY_BIFROST_VIRTUAL_KEY ignores ambient GEMINI_API_KEY
    cfg_bf = resolve_auth_config(
        env={
            "CC_AGY_VERTEX_BASE_URL": "http://localhost:8080/genai",
            "CC_AGY_BIFROST_VIRTUAL_KEY": "bf-vk-team-alpha",
            "GEMINI_API_KEY": "AIzaSyAmbientPersonalKey999999",
        }
    )
    assert cfg_bf["gateway_wire"] == "gemini"
    assert cfg_bf["base_url"] == "http://localhost:8080/genai"
    assert cfg_bf["bifrost_virtual_key"] == "bf-vk-team-alpha"
    assert not cfg_bf.get("api_key")

    # 4b. Host sandbox `GEMINI_API_KEY=dummy-key` + bare `GOOGLE_GEMINI_BASE_URL` shim is ignored
    # across auto, explicit `CC_AGY_AUTH_MODE=vertex`, and explicit `CC_AGY_AUTH_MODE=api_key`
    # (tested on both a non-host port 55052 and 55051 so `_STARTUP_JETSKI_SHIM_GEMINI_URL` never masks the guard)!
    for shim_url in ("http://127.0.0.1:55052", "http://127.0.0.1:55051"):
        for mode_val in ("auto", "vertex"):
            cfg_shim = resolve_auth_config(
                env={
                    "CC_AGY_AUTH_MODE": mode_val,
                    "GOOGLE_GEMINI_BASE_URL": shim_url,
                    "GEMINI_API_KEY": "dummy-key",
                    "GOOGLE_CLOUD_PROJECT": "real-vertex-proj",
                }
            )
            assert cfg_shim["auth_mode"] == "vertex"
            assert cfg_shim["project"] == "real-vertex-proj"
            assert cfg_shim["base_url"] is None
            assert resolve_effective_gateway_url(cfg_shim) == ""

        cfg_shim_apikey = resolve_auth_config(
            env={
                "CC_AGY_AUTH_MODE": "api_key",
                "GOOGLE_GEMINI_BASE_URL": shim_url,
                "GEMINI_API_KEY": "dummy-key",
                "GOOGLE_API_KEY": "AIzaSyRealGoogleKey123456",
            }
        )
        assert cfg_shim_apikey["api_key"] == "AIzaSyRealGoogleKey123456"
        assert cfg_shim_apikey["base_url"] is None
        assert resolve_effective_gateway_url(cfg_shim_apikey) == ""

    # 5. Gateway mode satisfies enforce_project_controls without requiring GOOGLE_CLOUD_PROJECT even with allowlist set,
    # and audience auto-derivation works from CC_AGY_GEMINI_BASE_URL as well
    assert (
        enforce_project_controls(
            auth_mode="vertex",
            project="custom-endpoint",
            source_env={
                "CC_AGY_GEMINI_BASE_URL": "https://bifrost.run.app/genai",
                "CC_AGY_ALLOWED_PROJECTS": "approved-proj-1",
            },
            explicit_env={
                "CC_AGY_GEMINI_BASE_URL": "https://bifrost.run.app/genai",
                "CC_AGY_ALLOWED_PROJECTS": "approved-proj-1",
            },
        )
        is None
    )
    assert (
        resolve_vertex_base_url_audience(
            "auto",
            env={"CC_AGY_GEMINI_BASE_URL": "https://bifrost.run.app/genai"},
        )
        == "https://bifrost.run.app"
    )


def test_live_loopback_wire_bifrost_genai_with_oidc_and_virtual_key_and_litellm_gemini_and_vertex(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    oidc_jwt = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJodHRwczovL2JpZnJvc3QtZ3cucnVuLmFwcCJ9.sig"
    monkeypatch.setattr(
        "server.regions._mint_oidc_identity_token",
        lambda *_a, **_kw: oidc_jwt,
    )
    sample_task = FileTask(task_id="t1", path="a.py", action="create", intent="create a.py")

    # Case A: Bifrost `/genai` with BOTH Cloud Run OIDC (`Authorization: Bearer`) AND `x-bf-vk` virtual key via `ask_gemini`
    with _MockGatewayServer(reply_text="bifrost-pong") as srv:
        monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL", f"{srv.origin}/genai")
        monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL_AUDIENCE", "https://bifrost-gw.run.app")
        monkeypatch.setenv("CC_AGY_BIFROST_VIRTUAL_KEY", "bf-vk-eng-platform")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyAmbientShouldNotLeak000000")
        monkeypatch.setenv("CC_AGY_DATA_DIR", str(tmp_path / "ask_data"))
        res_bf = ask_gemini(
            "ping",
            model="gemini-flash-next",
            thinking_level="LOW",
            workspace_root=tmp_path,
        )
        assert res_bf["text"] == "bifrost-pong"
        assert len(srv.requests) == 1
        req_bf = srv.requests[0]
        assert req_bf["path"] == "/genai/v1beta/models/gemini-flash-next:generateContent"
        assert req_bf["headers"]["authorization"] == f"Bearer {oidc_jwt}"
        assert req_bf["headers"]["x-bf-vk"] == "bf-vk-eng-platform"
        # Cloud Run OIDC JWT stays strictly in Authorization: Bearer and never overwrites x-goog-api-key
        assert req_bf["headers"]["x-goog-api-key"] == "bf-vk-eng-platform"
        assert "AIzaSyAmbientShouldNotLeak000000" not in json.dumps(req_bf["headers"])

    # Clean up monkeypatched gateway vars before Case B/B2/C
    for k in ("CC_AGY_VERTEX_BASE_URL", "CC_AGY_VERTEX_BASE_URL_AUDIENCE", "CC_AGY_BIFROST_VIRTUAL_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    # Case B: LiteLLM Gemini pass-through (`/gemini`) via `GOOGLE_GEMINI_BASE_URL` + `GEMINI_API_KEY`
    with _MockGatewayServer(reply_text='{"path": "a.py", "content": "x = 42\\n"}') as srv_gem:
        auth_gem = resolve_auth_config(
            env={
                "GOOGLE_GEMINI_BASE_URL": f"{srv_gem.origin}/gemini",
                "GEMINI_API_KEY": "sk-litellm-gemini-vk",
            }
        )
        writer_gem = make_genai_writer(auth_gem, "gemini-flash-next", thinking_level="LOW")
        reply_gem = writer_gem(WriterRequest(task=sample_task, system="Return JSON.", prompt="Write x = 42"))
        assert json.loads(reply_gem.text)["content"] == "x = 42\n"
        assert len(srv_gem.requests) == 1
        req_gem = srv_gem.requests[0]
        assert req_gem["path"] == "/gemini/v1beta/models/gemini-flash-next:generateContent"
        assert req_gem["headers"]["x-goog-api-key"] == "sk-litellm-gemini-vk"
        assert req_gem["headers"]["authorization"] == "Bearer sk-litellm-gemini-vk"

    # Case B2: LiteLLM `/gemini` behind Cloud Run OIDC (`Authorization: Bearer <oidc-jwt>`) + `GEMINI_API_KEY` (`x-goog-api-key: <litellm-key>`)
    with _MockGatewayServer(reply_text='{"path": "a.py", "content": "oidc_gem = 7\\n"}') as srv_gem_oidc:
        auth_gem_oidc = resolve_auth_config(
            env={
                "CC_AGY_GEMINI_BASE_URL": f"{srv_gem_oidc.origin}/gemini",
                "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://litellm-gw.run.app",
                "GEMINI_API_KEY": "sk-litellm-gemini-vk",
            }
        )
        writer_gem_oidc = make_genai_writer(auth_gem_oidc, "gemini-flash-next", thinking_level="LOW")
        reply_gem_oidc = writer_gem_oidc(WriterRequest(task=sample_task, system="Return JSON.", prompt="Write oidc_gem = 7"))
        assert json.loads(reply_gem_oidc.text)["content"] == "oidc_gem = 7\n"
        req_gem_oidc = srv_gem_oidc.requests[0]
        assert req_gem_oidc["headers"]["authorization"] == f"Bearer {oidc_jwt}"
        assert req_gem_oidc["headers"]["x-goog-api-key"] == "sk-litellm-gemini-vk"

    # Case B3: LiteLLM `/gemini` with `CC_AGY_VERTEX_BASE_URL_TOKEN=sk-team` and ambient personal `GEMINI_API_KEY`
    # sends ONLY `sk-team` in both `Authorization: Bearer` and `x-goog-api-key` and never leaks the personal key!
    with _MockGatewayServer(reply_text='{"path": "a.py", "content": "team_key = 1\\n"}') as srv_gem_tok:
        auth_gem_tok = resolve_auth_config(
            env={
                "CC_AGY_GEMINI_BASE_URL": f"{srv_gem_tok.origin}/gemini",
                "CC_AGY_VERTEX_BASE_URL_TOKEN": "sk-team-gateway-token",
                "GEMINI_API_KEY": "AIzaSyPersonalShellKeyMustNotLeak",
            }
        )
        assert auth_gem_tok.get("api_key") is None
        writer_gem_tok = make_genai_writer(auth_gem_tok, "gemini-flash-next", thinking_level="LOW")
        reply_gem_tok = writer_gem_tok(
            WriterRequest(task=sample_task, system="Return JSON.", prompt="Write team_key = 1")
        )
        assert json.loads(reply_gem_tok.text)["content"] == "team_key = 1\n"
        req_gem_tok = srv_gem_tok.requests[0]
        assert req_gem_tok["headers"]["authorization"] == "Bearer sk-team-gateway-token"
        assert req_gem_tok["headers"]["x-goog-api-key"] == "sk-team-gateway-token"
        assert "AIzaSyPersonalShellKeyMustNotLeak" not in json.dumps(req_gem_tok["headers"])

    # Case C: LiteLLM Vertex pass-through (`/vertex_ai`) via `CC_AGY_VERTEX_BASE_URL` + `CC_AGY_VERTEX_BASE_URL_TOKEN`
    # Also verifies `make_genai_writer` strips `publishers/google/models/` on custom gateways
    with _MockGatewayServer(reply_text='{"path": "a.py", "content": "vtx = 1\\n"}') as srv_vtx:
        auth_vtx = resolve_auth_config(
            model="gemini-flash-next",
            env={
                "CC_AGY_VERTEX_BASE_URL": f"{srv_vtx.origin}/vertex_ai",
                "CC_AGY_VERTEX_BASE_URL_TOKEN": "sk-litellm-vertex-vk",
                "GOOGLE_CLOUD_PROJECT": "acme-prod-ai",
            },
        )
        writer_vtx = make_genai_writer(
            auth_vtx,
            "publishers/google/models/gemini-flash-next",
            thinking_level="LOW",
        )
        reply_vtx = writer_vtx(WriterRequest(task=sample_task, system="Return JSON.", prompt="Write vtx = 1"))
        assert json.loads(reply_vtx.text)["content"] == "vtx = 1\n"
        assert reply_vtx.model == "publishers/google/models/gemini-flash-next"
        assert len(srv_vtx.requests) == 1
        req_vtx = srv_vtx.requests[0]
        assert (
            req_vtx["path"]
            == "/vertex_ai/v1beta1/projects/acme-prod-ai/locations/global/publishers/google/models/gemini-flash-next:generateContent"
        )
        assert req_vtx["headers"]["authorization"] == "Bearer sk-litellm-vertex-vk"
        assert req_vtx["headers"]["x-goog-api-key"] == "sk-litellm-vertex-vk"


def _ensure_antigravity_sdk_or_fake(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return real `google.antigravity.types` if installed, or install a faithful fake for `/usr/bin/python3`."""
    try:
        import google.antigravity.types as real_types  # type: ignore

        return real_types
    except ImportError:
        fake_types = py_types.ModuleType("google.antigravity.types")

        class _ModelType:
            TEXT = "TEXT"
            IMAGE = "IMAGE"

        class _ThinkingLevel:
            MINIMAL = "MINIMAL"
            LOW = "LOW"
            MEDIUM = "MEDIUM"
            HIGH = "HIGH"
            EXTRA_HIGH = "EXTRA_HIGH"

        class _BuiltinTools:
            FINISH = "FINISH"
            READ_FILE = "READ_FILE"
            WRITE_FILE = "WRITE_FILE"
            EDIT_FILE = "EDIT_FILE"
            RUN_COMMAND = "RUN_COMMAND"
            GREP_SEARCH = "GREP_SEARCH"
            FIND_BY_NAME = "FIND_BY_NAME"
            LIST_DIR = "LIST_DIR"

        class GeminiModelOptions:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class GeminiAPIEndpoint:
            def __init__(
                self,
                *,
                api_key: str,
                base_url: str | None = None,
                http_headers: dict[str, str] | None = None,
                options: Any = None,
            ) -> None:
                if not api_key:
                    raise ValueError("AntigravityAuthError: api_key required")
                self.api_key = api_key
                self.base_url = base_url
                self.http_headers = dict(http_headers or {})
                self.options = options

        class VertexEndpoint:
            def __init__(
                self,
                *,
                project: str | None = None,
                location: str | None = None,
                base_url: str | None = None,
                api_key: str | None = None,
                credentials: Any = None,
                http_headers: dict[str, str] | None = None,
                options: Any = None,
                **kwargs: Any,
            ) -> None:
                self.project = project
                self.location = location
                self.base_url = base_url
                self.api_key = api_key
                self.credentials = credentials
                self.http_headers = dict(http_headers or {})
                self.options = options
                self.__dict__.update(kwargs)

        class ModelTarget:
            def __init__(
                self,
                *,
                name: str,
                types: Any,
                endpoint: Any,
                thinking_level: Any = None,
                max_output_tokens: int | None = None,
            ) -> None:
                self.name = name
                self.types = list(types)
                self.endpoint = endpoint
                self.thinking_level = thinking_level
                self.max_output_tokens = max_output_tokens

        class _KwargsBag:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class LocalAgentConfig:
            def __init__(
                self,
                *,
                workspace_dir: str | None = None,
                model: Any = None,
                models: list[Any] | None = None,
                vertex: bool = False,
                api_key: str | None = None,
                project: str | None = None,
                location: str | None = None,
                **kwargs: Any,
            ) -> None:
                self.workspace_dir = workspace_dir
                self.vertex = vertex
                self.api_key = api_key
                self.project = project
                self.location = location
                self.__dict__.update(kwargs)
                resolved_models = list(models or ([model] if model is not None else []))
                if not any(_ModelType.IMAGE in m.types for m in resolved_models):
                    if not vertex:
                        resolved_models.append(
                            ModelTarget(
                                name="gemini-3-pro-image-preview",
                                types=[_ModelType.IMAGE],
                                endpoint=GeminiAPIEndpoint(api_key=api_key or ""),
                            )
                        )
                self.models = resolved_models

        fake_types.ModelType = _ModelType  # type: ignore[attr-defined]
        fake_types.ThinkingLevel = _ThinkingLevel  # type: ignore[attr-defined]
        fake_types.BuiltinTools = _BuiltinTools  # type: ignore[attr-defined]
        fake_types.GeminiModelOptions = GeminiModelOptions  # type: ignore[attr-defined]
        fake_types.GeminiAPIEndpoint = GeminiAPIEndpoint  # type: ignore[attr-defined]
        fake_types.VertexEndpoint = VertexEndpoint  # type: ignore[attr-defined]
        fake_types.ModelTarget = ModelTarget  # type: ignore[attr-defined]
        fake_types.Model = ModelTarget  # type: ignore[attr-defined]
        fake_types.SubagentConfig = _KwargsBag  # type: ignore[attr-defined]
        fake_types.CapabilitiesConfig = _KwargsBag  # type: ignore[attr-defined]
        fake_types.ToolOutputTruncationConfig = _KwargsBag  # type: ignore[attr-defined]
        fake_types.BudgetConfig = _KwargsBag  # type: ignore[attr-defined]
        fake_types.CompactionConfig = _KwargsBag  # type: ignore[attr-defined]
        fake_types.CustomSystemInstructions = _KwargsBag  # type: ignore[attr-defined]
        fake_types.TemplatedSystemInstructions = _KwargsBag  # type: ignore[attr-defined]
        fake_types.SystemInstructionSection = _KwargsBag  # type: ignore[attr-defined]
        fake_types.RunCommandConfig = _KwargsBag  # type: ignore[attr-defined]
        fake_types.LocalAgentConfig = LocalAgentConfig  # type: ignore[attr-defined]

        fake_policy = py_types.ModuleType("google.antigravity.policy")
        fake_policy.allow_all = lambda: "allow_all"  # type: ignore[attr-defined]
        fake_policy.workspace_only = lambda paths: ["workspace_only", list(paths)]  # type: ignore[attr-defined]

        fake_agy = py_types.ModuleType("google.antigravity")
        fake_agy.types = fake_types  # type: ignore[attr-defined]
        fake_agy.policy = fake_policy  # type: ignore[attr-defined]
        fake_agy.LocalAgentConfig = LocalAgentConfig  # type: ignore[attr-defined]
        fake_agy.Agent = object  # type: ignore[attr-defined]

        import google  # type: ignore

        monkeypatch.setattr(google, "antigravity", fake_agy, raising=False)
        monkeypatch.setitem(sys.modules, "google.antigravity", fake_agy)
        monkeypatch.setitem(sys.modules, "google.antigravity.types", fake_types)
        monkeypatch.setitem(sys.modules, "google.antigravity.policy", fake_policy)
        return fake_types


def test_antigravity_sdk_gemini_vs_vertex_endpoint_construction_without_ambient_gemini_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    types = _ensure_antigravity_sdk_or_fake(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CC_AGY_VERTEX_BASE_URL", raising=False)
    monkeypatch.delenv("GOOGLE_VERTEX_BASE_URL", raising=False)

    captured_configs: list[Any] = []

    class _FakeTurnResp:
        def resolve(self) -> None:
            return

        def text(self) -> str:
            return '{"content": "ok\\n"}'

        def structured_output(self) -> dict[str, Any]:
            return {"content": "ok\n"}

    class _FakeEnteredAgent:
        def chat(self, _prompt: str) -> _FakeTurnResp:
            return _FakeTurnResp()

    class _FakeAgent:
        def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
            captured_configs.append(config)

        async def __aenter__(self) -> _FakeEnteredAgent:
            return _FakeEnteredAgent()

        async def __aexit__(self, *_: Any) -> None:
            return

    monkeypatch.setattr("google.antigravity.Agent", _FakeAgent)

    sample_task = FileTask(task_id="t1", path="a.py", action="create", intent="create a.py")
    # 1. Bifrost `/genai` in `make_agent_writer` with NO `GEMINI_API_KEY` in `os.environ`:
    # Must construct `GeminiAPIEndpoint` AND pass a non-secret placeholder `api_key` into `LocalAgentConfig(vertex=False, api_key=...)`
    # so `LocalAgentConfig`'s injected shorthand image model neither raises `AntigravityAuthError` nor carries the real gateway secret!
    auth_bf = resolve_auth_config(
        env={
            "CC_AGY_VERTEX_BASE_URL": "http://localhost:8080/genai",
            "CC_AGY_BIFROST_VIRTUAL_KEY": "bf-vk-only-no-gemini-key",
        }
    )
    agent_writer_bf = make_agent_writer(
        auth_bf,
        "publishers/google/models/gemini-flash-next",
        thinking_level="LOW",
        workspace_dir=tmp_path,
    )
    rep_bf = agent_writer_bf(WriterRequest(task=sample_task, system="Sys", prompt="Hi"))
    assert json.loads(rep_bf.text)["content"] == "ok\n"
    assert len(captured_configs) == 1
    text_models_bf = [m for m in captured_configs[0].models if types.ModelType.TEXT in m.types]
    image_models_bf = [m for m in captured_configs[0].models if types.ModelType.IMAGE in m.types]
    assert len(text_models_bf) == 1
    assert text_models_bf[0].name == "gemini-flash-next"
    assert isinstance(text_models_bf[0].endpoint, types.GeminiAPIEndpoint)
    assert text_models_bf[0].endpoint.base_url == "http://localhost:8080/genai"
    assert text_models_bf[0].endpoint.http_headers["x-bf-vk"] == "bf-vk-only-no-gemini-key"
    for img_m in image_models_bf:
        assert getattr(img_m.endpoint, "api_key", None) == "custom-gateway-no-direct-api-key"

    # 2. LiteLLM `/vertex_ai` in `make_agent_writer` constructs `VertexEndpoint`
    captured_configs.clear()
    auth_vtx = resolve_auth_config(
        env={
            "CC_AGY_VERTEX_BASE_URL": "https://litellm.example.com/vertex_ai",
            "CC_AGY_VERTEX_BASE_URL_TOKEN": "sk-litellm-tok",
            "GOOGLE_CLOUD_PROJECT": "corp-vtx-proj",
        }
    )
    agent_writer_vtx = make_agent_writer(
        auth_vtx,
        "gemini-flash-next",
        thinking_level="LOW",
        workspace_dir=tmp_path,
    )
    agent_writer_vtx(WriterRequest(task=sample_task, system="Sys", prompt="Hi"))
    assert len(captured_configs) == 1
    text_models_vtx = [m for m in captured_configs[0].models if types.ModelType.TEXT in m.types]
    assert isinstance(text_models_vtx[0].endpoint, types.VertexEndpoint)
    assert text_models_vtx[0].endpoint.base_url == "https://litellm.example.com/vertex_ai"
    assert text_models_vtx[0].endpoint.project == "corp-vtx-proj"

    # 3. `run_antigravity_worker` with Bifrost `/genai` + `CC_AGY_BIFROST_VIRTUAL_KEY` (and NO `GEMINI_API_KEY`)
    captured_configs.clear()
    monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL", "http://localhost:8080/genai")
    monkeypatch.setenv("CC_AGY_BIFROST_VIRTUAL_KEY", "bf-vk-worker-secret")
    monkeypatch.setenv("CC_AGY_CONTAINMENT", "0")
    worker_out = run_antigravity_worker(
        prompts=["Implement feature"],
        workspace_dir=tmp_path,
        model="gemini-flash-next",
    )
    assert worker_out["text"] == '{"content": "ok\\n"}'
    assert len(captured_configs) == 1
    worker_text_models = [m for m in captured_configs[0].models if types.ModelType.TEXT in m.types]
    worker_image_models = [m for m in captured_configs[0].models if types.ModelType.IMAGE in m.types]
    assert isinstance(worker_text_models[0].endpoint, types.GeminiAPIEndpoint)
    assert worker_text_models[0].endpoint.base_url == "http://localhost:8080/genai"
    assert worker_text_models[0].endpoint.http_headers["x-bf-vk"] == "bf-vk-worker-secret"
    for img_m in worker_image_models:
        assert getattr(img_m.endpoint, "api_key", None) == "custom-gateway-no-direct-api-key"

    # 4. `run_antigravity_worker` and `make_agent_writer` with explicit `api_key` or `CC_AGY_AUTH_MODE=api_key`
    # ignore ambient `CC_AGY_VERTEX_BASE_URL` / `GOOGLE_GEMINI_BASE_URL`
    captured_configs.clear()
    monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL", "http://localhost:8080/genai")
    monkeypatch.setenv("CC_AGY_AUTH_MODE", "api_key")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDirectWorkerKey")
    run_antigravity_worker(
        prompts=["Implement feature directly"],
        workspace_dir=tmp_path,
        model="gemini-flash-next",
    )
    assert len(captured_configs) == 1
    direct_text_models = [m for m in captured_configs[0].models if types.ModelType.TEXT in m.types]
    assert getattr(direct_text_models[0].endpoint, "base_url", None) is None

    captured_configs.clear()
    auth_direct_writer = resolve_auth_config(
        env={
            "CC_AGY_AUTH_MODE": "api_key",
            "GEMINI_API_KEY": "AIzaSyDirectWriterKey",
            "CC_AGY_VERTEX_BASE_URL": "http://localhost:8080/genai",
        }
    )
    direct_writer_fn = make_agent_writer(
        auth_direct_writer,
        "gemini-flash-next",
        workspace_dir=tmp_path,
    )
    direct_writer_fn(
        WriterRequest(
            task=FileTask(task_id="t_direct", path="a.py", action="create", intent="create a.py"),
            system="Return JSON.",
            prompt="Write a.py",
        )
    )
    assert len(captured_configs) == 1
    direct_writer_text_models = [m for m in captured_configs[0].models if types.ModelType.TEXT in m.types]
    assert getattr(direct_writer_text_models[0].endpoint, "base_url", None) is None
    monkeypatch.delenv("CC_AGY_AUTH_MODE", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # 4b. Two-pass `_execute_job` / `followup_job` -> `run_antigravity_worker` preserves Gemini-wire gateways (`P1-1`)
    # across (a) CC_AGY_GEMINI_BASE_URL + key, (b) CC_AGY_GEMINI_BASE_URL + key + CC_AGY_VERTEX_BASE_URL_TOKEN,
    # (c) GOOGLE_GEMINI_BASE_URL + key, (d) OIDC audience + key, and (e) follow-up pass.
    monkeypatch.setattr(
        "server.regions._mint_oidc_identity_token",
        lambda *_a, **_kw: "eyJFAKE.OIDC.WORKER_TOKEN",
    )
    two_pass_cases: list[tuple[str, dict[str, str], str, str | None]] = [
        (
            "cc_agy_gem_key",
            {"CC_AGY_GEMINI_BASE_URL": "https://litellm.corp/gemini", "GEMINI_API_KEY": "sk-litellm-vk"},
            "sk-litellm-vk",
            None,
        ),
        (
            "cc_agy_gem_key_plus_token",
            {
                "CC_AGY_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "GEMINI_API_KEY": "AIzaSyPersonalShellKeyMustNotLeak",
                "CC_AGY_VERTEX_BASE_URL_TOKEN": "sk-team-token",
            },
            "sk-team-token",
            "AIzaSyPersonalShellKeyMustNotLeak",
        ),
        (
            "google_gem_key",
            {"GOOGLE_GEMINI_BASE_URL": "https://litellm.corp/gemini", "GEMINI_API_KEY": "sk-litellm-cli-vk"},
            "sk-litellm-cli-vk",
            None,
        ),
        (
            "oidc_audience_plus_key",
            {
                "CC_AGY_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://litellm.corp",
                "GEMINI_API_KEY": "sk-litellm-oidc-vk",
            },
            "sk-litellm-oidc-vk",
            None,
        ),
        (
            "A_gp_google_gem_tok_key_proj",
            {
                "GOOGLE_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "CC_AGY_VERTEX_BASE_URL_TOKEN": "sk-team-token",
                "GEMINI_API_KEY": "AIzaSyPersonalShellKeyMustNotLeak",
                "GOOGLE_CLOUD_PROJECT": "my-proj",
            },
            "sk-team-token",
            "AIzaSyPersonalShellKeyMustNotLeak",
        ),
        (
            "T_gp_google_gem_tok_proj_no_key",
            {
                "GOOGLE_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "CC_AGY_VERTEX_BASE_URL_TOKEN": "sk-team-token",
                "GOOGLE_CLOUD_PROJECT": "my-proj",
            },
            "sk-team-token",
            None,
        ),
        (
            "BFgp_google_gem_bfvk_proj",
            {
                "GOOGLE_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "CC_AGY_BIFROST_VIRTUAL_KEY": "sk-bf-vk",
                "GOOGLE_CLOUD_PROJECT": "my-proj",
            },
            "sk-bf-vk",
            None,
        ),
        (
            "GWgp_gateway_mode_google_gem_tok_proj",
            {
                "CC_AGY_AUTH_MODE": "gateway",
                "GOOGLE_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "CC_AGY_VERTEX_BASE_URL_TOKEN": "sk-team-token",
                "GOOGLE_CLOUD_PROJECT": "my-proj",
            },
            "sk-team-token",
            None,
        ),
        (
            "GWgv_gateway_mode_google_gem_bfvk_proj",
            {
                "CC_AGY_AUTH_MODE": "gateway",
                "GOOGLE_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "CC_AGY_BIFROST_VIRTUAL_KEY": "sk-bf-vk",
                "GOOGLE_CLOUD_PROJECT": "my-proj",
            },
            "sk-bf-vk",
            None,
        ),
        (
            "GWck_gateway_mode_ccgem_key",
            {
                "CC_AGY_AUTH_MODE": "gateway",
                "CC_AGY_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "GEMINI_API_KEY": "sk-litellm-vk",
            },
            "sk-litellm-vk",
            None,
        ),
        (
            "GWgk2_gateway_mode_google_gem_key_proj",
            {
                "CC_AGY_AUTH_MODE": "gateway",
                "GOOGLE_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "GEMINI_API_KEY": "sk-litellm-vk",
                "GOOGLE_CLOUD_PROJECT": "my-proj",
            },
            "sk-litellm-vk",
            None,
        ),
        (
            "GWjk_gateway_mode_ccgem_jwt_token_and_key",
            {
                "CC_AGY_AUTH_MODE": "gateway",
                "CC_AGY_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "CC_AGY_VERTEX_BASE_URL_TOKEN": "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZWFtIn0.c2ln",
                "GEMINI_API_KEY": "AIzaSyPersonalShellKeyMustNotLeak",
            },
            "custom-endpoint",
            "AIzaSyPersonalShellKeyMustNotLeak",
        ),
        (
            "GWbk_gateway_mode_ccgem_bfvk_and_key",
            {
                "CC_AGY_AUTH_MODE": "gateway",
                "CC_AGY_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "CC_AGY_BIFROST_VIRTUAL_KEY": "sk-bf-vk",
                "GEMINI_API_KEY": "AIzaSyPersonalShellKeyMustNotLeak",
            },
            "sk-bf-vk",
            "AIzaSyPersonalShellKeyMustNotLeak",
        ),
        (
            "GWok_gateway_mode_ccgem_oidc_and_key",
            {
                "CC_AGY_AUTH_MODE": "gateway",
                "CC_AGY_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://litellm.corp",
                "GEMINI_API_KEY": "sk-litellm-oidc-vk",
            },
            "sk-litellm-oidc-vk",
            None,
        ),
        (
            "GWvg_gateway_mode_vtxurl_gemini_path_and_key",
            {
                "CC_AGY_AUTH_MODE": "gateway",
                "CC_AGY_VERTEX_BASE_URL": "https://litellm.corp/gemini",
                "GEMINI_API_KEY": "AIzaSyPersonalShellKeyMustNotLeak",
            },
            "custom-endpoint",
            "AIzaSyPersonalShellKeyMustNotLeak",
        ),
        (
            "AVg_auto_vtxurl_gemini_path_and_key",
            {
                "CC_AGY_VERTEX_BASE_URL": "https://litellm.corp/gemini",
                "GEMINI_API_KEY": "AIzaSyPersonalShellKeyMustNotLeak",
            },
            "custom-endpoint",
            "AIzaSyPersonalShellKeyMustNotLeak",
        ),
        (
            "VX_ccgem_key_vertex_mode_ccgem_proj_and_key",
            {
                "CC_AGY_AUTH_MODE": "vertex",
                "CC_AGY_GEMINI_BASE_URL": "https://litellm.corp/gemini",
                "GOOGLE_CLOUD_PROJECT": "my-proj",
                "GEMINI_API_KEY": "AIzaSyPersonalShellKeyMustNotLeak",
            },
            "custom-endpoint",
            "AIzaSyPersonalShellKeyMustNotLeak",
        ),
    ]
    routing_keys = (
        "auth_mode",
        "api_key",
        "project",
        "location",
        "base_url",
        "gateway_wire",
        "base_url_token",
        "bifrost_virtual_key",
        "base_url_audience",
    )
    for case_label, case_env, expected_ep_key, forbidden_key in two_pass_cases:
        for env_k in (
            "CC_AGY_VERTEX_BASE_URL",
            "CC_AGY_GEMINI_BASE_URL",
            "GOOGLE_GEMINI_BASE_URL",
            "CC_AGY_VERTEX_BASE_URL_TOKEN",
            "CC_AGY_VERTEX_BASE_URL_AUDIENCE",
            "CC_AGY_BIFROST_VIRTUAL_KEY",
            "GEMINI_API_KEY",
            "CC_AGY_AUTH_MODE",
            "GOOGLE_CLOUD_PROJECT",
        ):
            monkeypatch.delenv(env_k, raising=False)
        for env_k, env_v in case_env.items():
            monkeypatch.setenv(env_k, env_v)

        captured_configs.clear()
        a1 = resolve_auth_config(model="gemini-flash-next")
        assert a1["base_url"] == "https://litellm.corp/gemini", f"{case_label} pass-1 lost base_url: {a1}"
        assert a1["base_url"] is not None
        if forbidden_key:
            assert a1["auth_mode"] == "vertex"
            assert a1.get("api_key") is None
        run_antigravity_worker(
            prompts=[f"Run {case_label}"],
            workspace_dir=tmp_path,
            model="gemini-flash-next",
            project=a1.get("project"),
            location=a1.get("location"),
            api_key=a1.get("api_key"),
            auth_mode=a1.get("auth_mode"),
        )
        assert len(captured_configs) == 1, f"Missing worker config for {case_label}"
        ep = [m for m in captured_configs[0].models if types.ModelType.TEXT in m.types][0].endpoint
        assert isinstance(ep, types.GeminiAPIEndpoint), f"{case_label} expected GeminiAPIEndpoint, got {type(ep)}"
        assert ep.base_url == a1["base_url"]
        assert ep.base_url == "https://litellm.corp/gemini"
        assert ep.api_key == expected_ep_key
        if forbidden_key:
            assert forbidden_key not in json.dumps(ep.http_headers)
            assert forbidden_key != ep.api_key

        # Follow-up pass (`followup_job` -> `_execute_job` -> `run_antigravity_worker`)
        captured_configs.clear()
        a_followup = resolve_auth_config(
            requested_project=a1.get("project"),
            requested_location=a1.get("location"),
            requested_api_key=a1.get("api_key"),
            requested_auth_mode=a1.get("auth_mode"),
            model="gemini-flash-next",
        )
        for rk in routing_keys:
            assert a_followup.get(rk) == a1.get(rk), f"{case_label} routing mismatch on {rk}: {a_followup} vs {a1}"
        run_antigravity_worker(
            prompts=[f"Followup {case_label}"],
            workspace_dir=tmp_path,
            model="gemini-flash-next",
            project=a_followup.get("project"),
            location=a_followup.get("location"),
            api_key=a_followup.get("api_key"),
            auth_mode=a_followup.get("auth_mode"),
        )
        assert len(captured_configs) == 1
        ep_followup = [m for m in captured_configs[0].models if types.ModelType.TEXT in m.types][0].endpoint
        assert ep_followup.base_url == "https://litellm.corp/gemini"
        assert ep_followup.api_key == expected_ep_key
        if forbidden_key:
            assert forbidden_key not in json.dumps(ep_followup.http_headers)

    # 4c. Vertex-wire gateway in `gateway` mode never forwards ambient `GEMINI_API_KEY` (`R9h`: `GWvk` and `GWvk_gem_var`)
    for vtx_gw_label, vtx_gw_env in (
        (
            "GWvk",
            {
                "CC_AGY_AUTH_MODE": "gateway",
                "CC_AGY_VERTEX_BASE_URL": "https://litellm.corp/vertex_ai",
                "GEMINI_API_KEY": "AIzaSyPersonalShellKeyMustNotLeak",
            },
        ),
        (
            "GWvk_gem_var",
            {
                "CC_AGY_AUTH_MODE": "gateway",
                "CC_AGY_GEMINI_BASE_URL": "https://litellm.corp/vertex_ai",
                "CC_AGY_GATEWAY_WIRE": "vertex",
                "GEMINI_API_KEY": "AIzaSyPersonalShellKeyMustNotLeak",
            },
        ),
    ):
        for env_k in (
            "CC_AGY_VERTEX_BASE_URL",
            "CC_AGY_GEMINI_BASE_URL",
            "GOOGLE_GEMINI_BASE_URL",
            "CC_AGY_GATEWAY_WIRE",
            "CC_AGY_VERTEX_BASE_URL_TOKEN",
            "CC_AGY_VERTEX_BASE_URL_AUDIENCE",
            "CC_AGY_BIFROST_VIRTUAL_KEY",
            "GEMINI_API_KEY",
            "CC_AGY_AUTH_MODE",
            "GOOGLE_CLOUD_PROJECT",
        ):
            monkeypatch.delenv(env_k, raising=False)
        for env_k, env_v in vtx_gw_env.items():
            monkeypatch.setenv(env_k, env_v)

        captured_configs.clear()
        a_vtx_gw = resolve_auth_config(model="gemini-flash-next")
        assert a_vtx_gw["base_url"] == "https://litellm.corp/vertex_ai", f"{vtx_gw_label} lost base_url: {a_vtx_gw}"
        assert a_vtx_gw["gateway_wire"] == "vertex"
        assert a_vtx_gw.get("api_key") is None
        run_antigravity_worker(
            prompts=[f"Run {vtx_gw_label}"],
            workspace_dir=tmp_path,
            model="gemini-flash-next",
            project=a_vtx_gw.get("project"),
            location=a_vtx_gw.get("location"),
            api_key=a_vtx_gw.get("api_key"),
            auth_mode=a_vtx_gw.get("auth_mode"),
        )
        assert len(captured_configs) == 1
        ep_vtx_gw = [m for m in captured_configs[0].models if types.ModelType.TEXT in m.types][0].endpoint
        assert isinstance(ep_vtx_gw, types.VertexEndpoint)
        assert ep_vtx_gw.base_url == "https://litellm.corp/vertex_ai"
        assert ep_vtx_gw.api_key == "custom-endpoint"
        assert "AIzaSyPersonalShellKeyMustNotLeak" not in json.dumps(ep_vtx_gw.http_headers)

    for env_k in (
        "CC_AGY_VERTEX_BASE_URL",
        "CC_AGY_GEMINI_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
        "CC_AGY_GATEWAY_WIRE",
        "CC_AGY_VERTEX_BASE_URL_TOKEN",
        "CC_AGY_VERTEX_BASE_URL_AUDIENCE",
        "CC_AGY_BIFROST_VIRTUAL_KEY",
        "GEMINI_API_KEY",
        "CC_AGY_AUTH_MODE",
        "GOOGLE_CLOUD_PROJECT",
    ):
        monkeypatch.delenv(env_k, raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-worker")

    # 5. `create_shared_cache` normalizes `publishers/google/models/...` on custom gateways
    recorded_cache_models: list[str] = []

    class _FakeCaches:
        def create(self, *, model: str, config: Any) -> Any:
            recorded_cache_models.append(model)
            return py_types.SimpleNamespace(name="cachedContents/test-123")

    monkeypatch.setattr(
        "server.writers._build_genai_client",
        lambda _auth: py_types.SimpleNamespace(caches=_FakeCaches()),
    )
    cache_name = create_shared_cache(
        auth_bf,
        "publishers/google/models/gemini-flash-next",
        "System instructions",
        "x" * 20_000,
        ttl_seconds=300,
    )
    assert cache_name == "cachedContents/test-123"
    assert recorded_cache_models == ["gemini-flash-next"]


def test_installer_gateway_wire_all_five_hosts_doctor_and_secret_scrubbing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gw_url = "https://gateway.corp.example.com/custom-gemini-route"

    # 1. Claude userConfig & Codex TOML block carry `gateway_wire = "gemini"` and forward Bifrost/Gemini env vars without secrets
    claude_cfg = build_claude_user_config(
        auth_mode="gateway",
        base_url=gw_url,
        gateway_wire="gemini",
    )
    assert claude_cfg == {
        "auth_mode": "gateway",
        "vertex_base_url": gw_url,
        "gateway_wire": "gemini",
    }

    codex_toml = build_codex_server_block(
        python_bin="/venv/bin/python",
        auth_mode="gateway",
        base_url=gw_url,
        gateway_wire="gemini",
    )
    assert 'CC_AGY_GATEWAY_WIRE = "gemini"' in codex_toml
    assert '"CC_AGY_BIFROST_VIRTUAL_KEY"' in codex_toml
    assert '"CC_AGY_GEMINI_BASE_URL"' in codex_toml

    # 2. Cursor, OpenCode, and Antigravity installs write `CC_AGY_GATEWAY_WIRE` and never write virtual keys/tokens to disk
    monkeypatch.setenv("CC_AGY_BIFROST_VIRTUAL_KEY", "bf-vk-top-secret-never-write")
    monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL_TOKEN", "sk-litellm-top-secret-never-write")

    cur_res = install_cursor(
        cursor_config_dir=tmp_path / "cursor",
        python_bin="/venv/bin/python",
        auth_mode="gateway",
        base_url=gw_url,
        gateway_wire="gemini",
    )
    oc_res = install_opencode(
        opencode_config_dir=tmp_path / "opencode",
        python_bin="/venv/bin/python",
        auth_mode="gateway",
        base_url=gw_url,
        gateway_wire="gemini",
    )
    agy_res = install_agy(
        agy_config_dir=tmp_path / "agy",
        python_bin="/venv/bin/python",
        auth_mode="gateway",
        base_url=gw_url,
        gateway_wire="gemini",
    )
    assert cur_res["entry"]["env"]["CC_AGY_GATEWAY_WIRE"] == "gemini"
    assert oc_res["entry"]["environment"]["CC_AGY_GATEWAY_WIRE"] == "gemini"
    assert agy_res["entry"]["env"]["CC_AGY_GATEWAY_WIRE"] == "gemini"

    for written_file in tmp_path.rglob("*"):
        if written_file.is_file():
            raw = written_file.read_text(encoding="utf-8")
            assert "bf-vk-top-secret-never-write" not in raw
            assert "sk-litellm-top-secret-never-write" not in raw

    # CLI `--gateway-wire gemini --dry-run --json` works end-to-end across `--host all`
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex_cli"))
    monkeypatch.setenv("CURSOR_CONFIG_DIR", str(tmp_path / "cursor_cli"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "opencode_cli"))
    monkeypatch.setenv("AGY_CONFIG_DIR", str(tmp_path / "agy_cli"))
    rc = install_main(
        [
            "--host",
            "all",
            "--auth",
            "gateway",
            "--base-url",
            gw_url,
            "--gateway-wire",
            "gemini",
            "--dry-run",
            "--no-bootstrap",
            "--json",
        ]
    )
    assert rc == 0

    # Interactive wizard ignores Jetski loopback shim (`GEMINI_API_KEY=dummy-key` + `GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:55051`)
    wizard_prompts: list[str] = []

    def _ask_default(prompt: str) -> str:
        wizard_prompts.append(prompt)
        return ""

    wizard_argv = run_interactive_wizard(
        env={
            "GOOGLE_GEMINI_BASE_URL": "http://127.0.0.1:55052",
            "GEMINI_API_KEY": "dummy-key",
        },
        input_fn=_ask_default,
        output_fn=lambda _msg: None,
    )
    assert wizard_argv == ["--host", "all", "--profile", "balanced"]
    assert any("(api_key)" in p for p in wizard_prompts)

    # Interactive wizard preserves Gemini-CLI style `GOOGLE_GEMINI_BASE_URL` + `GEMINI_API_KEY` pairing (`I2`)
    wizard_prompts.clear()
    wizard_cli_pairing = run_interactive_wizard(
        env={
            "GOOGLE_GEMINI_BASE_URL": "https://litellm.corp/gemini",
            "GEMINI_API_KEY": "sk-litellm-vk",
        },
        input_fn=_ask_default,
        output_fn=lambda _msg: None,
    )
    assert wizard_cli_pairing == ["--host", "all", "--profile", "balanced"]
    assert any("(api_key)" in p for p in wizard_prompts)

    # Interactive wizard with `CC_AGY_GEMINI_BASE_URL` on a path-less Gemini proxy preserves `--gateway-wire gemini` (`J3`)
    wizard_prompts.clear()
    wizard_tips: list[str] = []
    wizard_gem_proxy = run_interactive_wizard(
        env={"CC_AGY_GEMINI_BASE_URL": "https://gem-proxy.corp"},
        input_fn=_ask_default,
        output_fn=wizard_tips.append,
    )
    assert wizard_gem_proxy == [
        "--host",
        "all",
        "--profile",
        "balanced",
        "--base-url",
        "https://gem-proxy.corp",
        "--gateway-wire",
        "gemini",
    ]
    assert any("CC_AGY_VERTEX_BASE_URL_TOKEN" in tip for tip in wizard_tips)

    # Preflight recognizes `CC_AGY_GEMINI_BASE_URL` in gateway mode and flags `--auth api-key` + `--base-url` conflict (`R6-2`)
    pre_gem = preflight(
        env={"CC_AGY_GEMINI_BASE_URL": "http://localhost:8080/genai"},
        auth_mode="gateway",
    )
    assert pre_gem["auth_mode"] == "gateway"

    pre_conflict = preflight(
        env={"GEMINI_API_KEY": "sk-litellm-vk"},
        auth_mode="api_key",
        base_url="https://litellm.example.com/gemini",
    )
    assert pre_conflict["ready"] is False
    assert any("ignores `--base-url`" in issue for issue in pre_conflict["issues"])

    # Ambient `GOOGLE_GEMINI_BASE_URL` in a Jetski shell does NOT fail `preflight(auth_mode="api_key")` when `--base-url` is not passed (`P2-1`)
    pre_ambient_shim = preflight(
        env={
            "GEMINI_API_KEY": "AIzaSyRealUserKey123456",
            "GOOGLE_GEMINI_BASE_URL": "http://127.0.0.1:55051",
        },
        auth_mode="api_key",
    )
    assert pre_ambient_shim["ready"] is True
    assert pre_ambient_shim["issues"] == []
    assert any("ignores `--base-url`" in w for w in pre_ambient_shim["warnings"])

    # `--auth api-key` host config builders omit `CC_AGY_VERTEX_BASE_URL` / `CC_AGY_GATEWAY_WIRE` (`Fable N2` & `Opus R8g`)
    ak_codex = build_codex_server_block(
        python_bin="/venv/bin/python",
        auth_mode="api_key",
        base_url="https://litellm.example.com/gemini",
        gateway_wire="gemini",
    )
    assert "CC_AGY_VERTEX_BASE_URL =" not in ak_codex
    assert "CC_AGY_GATEWAY_WIRE =" not in ak_codex
    ak_claude = build_claude_user_config(
        auth_mode="api_key",
        base_url="https://litellm.example.com/gemini",
        gateway_wire="gemini",
    )
    assert "vertex_base_url" not in ak_claude
    assert "gateway_wire" not in ak_claude
    ak_agy = build_agy_server_entry(
        python_bin="/venv/bin/python",
        auth_mode="api_key",
        base_url="https://litellm.example.com/gemini",
        gateway_wire="gemini",
        gateway_audience="https://litellm.example.com",
    )
    assert "CC_AGY_VERTEX_BASE_URL" not in ak_agy["env"]
    assert "CC_AGY_GATEWAY_WIRE" not in ak_agy["env"]
    assert "CC_AGY_VERTEX_BASE_URL_AUDIENCE" not in ak_agy["env"]

    # 3. Doctor reports `gateway_wire` + boolean `bifrost_virtual_key_configured` (never raw key) and flags invalid wire
    monkeypatch.setattr("server.doctor._module_available", lambda _name: True)
    doc_ok = run_doctor(
        workspace_root=REPO_ROOT,
        env={
            "HOME": str(tmp_path / "doc_home"),
            "CC_AGY_VERTEX_BASE_URL": "http://localhost:8080/genai",
            "CC_AGY_BIFROST_VIRTUAL_KEY": "bf-vk-secret-value-777",
            "CC_AGY_DATA_DIR": str(tmp_path / "doc_data"),
        },
    )
    assert doc_ok["auth"]["custom_vertex_gateway"] is True
    assert doc_ok["auth"]["gateway_wire"] == "gemini"
    assert doc_ok["auth"]["bifrost_virtual_key_configured"] is True
    assert doc_ok["auth"]["gateway_credential_source"] == "bifrost_virtual_key"
    assert "bf-vk-secret-value-777" not in json.dumps(doc_ok)

    doc_gw_gem_key = run_doctor(
        workspace_root=REPO_ROOT,
        env={
            "HOME": str(tmp_path / "doc_home_gw_key"),
            "CC_AGY_AUTH_MODE": "gateway",
            "CC_AGY_GEMINI_BASE_URL": "https://litellm.corp/gemini",
            "GEMINI_API_KEY": "sk-litellm-vk",
            "CC_AGY_DATA_DIR": str(tmp_path / "doc_data_gw_key"),
        },
    )
    assert doc_gw_gem_key["auth"]["gateway_credential_source"] == "gemini_api_key"

    doc_gw_vtx_nokey = run_doctor(
        workspace_root=REPO_ROOT,
        env={
            "HOME": str(tmp_path / "doc_home_gw_nokey"),
            "CC_AGY_AUTH_MODE": "gateway",
            "CC_AGY_VERTEX_BASE_URL": "https://litellm.corp/gemini",
            "GEMINI_API_KEY": "AIzaSyPersonalShellKeyMustNotLeak",
            "CC_AGY_DATA_DIR": str(tmp_path / "doc_data_gw_nokey"),
        },
    )
    assert doc_gw_vtx_nokey["auth"]["gateway_credential_source"] == "none"

    # Human-readable CLI text mode prints `pre["warnings"]` (`Fable [TUNE]` & `Opus NIT`)
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyRealUserKey123456")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://litellm.example.com/gemini")
    capsys.readouterr()
    rc_warn = install_main(["--host", "codex", "--auth", "api-key", "--dry-run", "--no-bootstrap"])
    assert rc_warn == 0
    warn_stdout = capsys.readouterr().out
    assert "~ Explicit `--auth api-key`" in warn_stdout
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)

    # Doctor with `CC_AGY_AUTH_MODE=api_key` and dedicated `CC_AGY_VERTEX_BASE_URL` reports `healthy=False` (`issues`),
    # whereas ambient `GOOGLE_GEMINI_BASE_URL` reports `healthy=True` (`warnings`, `Opus R8f`)
    doc_apikey_override = run_doctor(
        workspace_root=REPO_ROOT,
        env={
            "HOME": str(tmp_path / "doc_home_ak"),
            "CC_AGY_AUTH_MODE": "api_key",
            "GEMINI_API_KEY": "AIzaSyDirectDoctorKey",
            "CC_AGY_VERTEX_BASE_URL": "http://localhost:8080/genai",
            "CC_AGY_DATA_DIR": str(tmp_path / "doc_data_ak"),
        },
    )
    assert doc_apikey_override["auth"]["auth_mode"] == "api_key"
    assert doc_apikey_override["auth"]["custom_vertex_gateway"] is False
    assert doc_apikey_override["healthy"] is False
    assert any("ignores configured gateway URL" in issue for issue in doc_apikey_override["issues"])

    doc_ambient_shim = run_doctor(
        workspace_root=REPO_ROOT,
        env={
            "HOME": str(tmp_path / "doc_home_ambient"),
            "CC_AGY_AUTH_MODE": "api_key",
            "GEMINI_API_KEY": "AIzaSyDirectDoctorKey",
            "GOOGLE_GEMINI_BASE_URL": "http://127.0.0.1:55051",
            "CC_AGY_DATA_DIR": str(tmp_path / "doc_data_ambient"),
        },
    )
    assert doc_ambient_shim["auth"]["auth_mode"] == "api_key"
    assert doc_ambient_shim["healthy"] is True
    assert doc_ambient_shim["issues"] == []
    assert any("ignores configured gateway URL" in w for w in doc_ambient_shim["warnings"])

    # Doctor with `probe_oidc=True` and `CC_AGY_BIFROST_VIRTUAL_KEY` still catches broken Cloud Run OIDC!
    def _fail_oidc(*_a: Any, **_kw: Any) -> str:
        raise RuntimeError("Simulated Cloud Run OIDC minting failure")

    monkeypatch.setattr("server.regions._mint_oidc_identity_token", _fail_oidc)
    doc_broken_oidc = run_doctor(
        workspace_root=REPO_ROOT,
        probe_oidc=True,
        env={
            "HOME": str(tmp_path / "doc_home_oidc"),
            "CC_AGY_VERTEX_BASE_URL": "https://bifrost-gw.run.app/genai",
            "CC_AGY_VERTEX_BASE_URL_AUDIENCE": "https://bifrost-gw.run.app",
            "CC_AGY_BIFROST_VIRTUAL_KEY": "bf-vk-eng-platform",
            "CC_AGY_DATA_DIR": str(tmp_path / "doc_data_oidc"),
        },
    )
    assert doc_broken_oidc["healthy"] is False
    assert doc_broken_oidc["auth"]["gateway_oidc_principal_status"] == "probe_failed"

    doc_bad_wire = run_doctor(
        workspace_root=REPO_ROOT,
        env={
            "HOME": str(tmp_path / "doc_home2"),
            "CC_AGY_VERTEX_BASE_URL": "http://localhost:8080/genai",
            "CC_AGY_GATEWAY_WIRE": "invalid-protocol",
            "CC_AGY_DATA_DIR": str(tmp_path / "doc_data2"),
        },
    )
    assert doc_bad_wire["healthy"] is False
    assert any("Unsupported gateway wire mode" in issue for issue in doc_bad_wire["issues"])

    # 4. Secret scrubbing strips/blanks `CC_AGY_BIFROST_VIRTUAL_KEY` across verify, receipts, and agy_cli
    scrub_env = credential_free_environ(
        {
            "CC_AGY_BIFROST_VIRTUAL_KEY": "bf-vk-scrub-me",
            "CC_AGY_VERTEX_BASE_URL_TOKEN": "sk-scrub-me",
            "SAFE_FLAG": "1",
        }
    )
    assert scrub_env.get("CC_AGY_BIFROST_VIRTUAL_KEY") == ""
    assert scrub_env.get("CC_AGY_VERTEX_BASE_URL_TOKEN") == ""
    assert scrub_env["SAFE_FLAG"] == "1"

    monkeypatch.setenv("CC_AGY_BIFROST_VIRTUAL_KEY", "bf-vk-scrub-me-longsecret")
    monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL_TOKEN", "sk-scrub-me-longsecret")
    redacted_rec = redact_api_keys(
        {"error": "Headers: x-bf-vk=bf-vk-scrub-me-longsecret auth=sk-scrub-me-longsecret"}
    )
    redacted = json.dumps(redacted_rec)
    assert "bf-vk-scrub-me-longsecret" not in redacted
    assert "sk-scrub-me-longsecret" not in redacted

    clean_agy = build_agy_subprocess_env(
        {
            "PATH": "/usr/bin",
            "CC_AGY_BIFROST_VIRTUAL_KEY": "bf-vk-scrub-me",
            "CC_AGY_GEMINI_BASE_URL": "http://localhost:8080/genai",
            "GOOGLE_GEMINI_BASE_URL": "http://localhost:8080/genai",
        }
    )
    assert "CC_AGY_BIFROST_VIRTUAL_KEY" not in clean_agy
    assert "CC_AGY_GEMINI_BASE_URL" not in clean_agy
    assert "GOOGLE_GEMINI_BASE_URL" not in clean_agy


def test_powershell_bootstrap_gateway_secret_regex_matches_python_validator_on_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ps1_text = (REPO_ROOT / "scripts" / "bootstrap-gateway.ps1").read_text(encoding="utf-8")
    m_pat = re.search(r"^\$SecretParamPattern = '([^']+)'", ps1_text, flags=re.MULTILINE)
    assert m_pat is not None, "Could not extract $SecretParamPattern from bootstrap-gateway.ps1"
    secret_param_pattern = m_pat.group(1)
    assert secret_param_pattern.startswith("(?i)")
    core_pat = secret_param_pattern[4:]

    # Verify every `[ValidateSet(...)]` value for `$GatewayWire` in `bootstrap-gateway.ps1` is accepted by
    # both `server.install --gateway-wire` and `resolve_gateway_wire` (`R6-1`).
    m_wire_set = re.search(
        r'\[ValidateSet\(([^)]+)\)\]\s*\[string\]\$GatewayWire',
        ps1_text,
    )
    assert m_wire_set is not None, "Could not extract $GatewayWire ValidateSet from bootstrap-gateway.ps1"
    wire_values = re.findall(r'"([^"]+)"', m_wire_set.group(1))
    assert set(wire_values) == {"auto", "vertex", "gemini", "litellm-vertex", "litellm-gemini", "bifrost"}
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "ps1_codex"))
    for w_val in wire_values:
        assert resolve_gateway_wire(w_val, env={}) in ("vertex", "gemini")
        rc_wire = install_main(
            [
                "--host",
                "codex",
                "--auth",
                "gateway",
                "--base-url",
                "https://gw.example.com/genai",
                "--gateway-wire",
                w_val,
                "--dry-run",
                "--no-bootstrap",
                "--json",
            ]
        )
        assert rc_wire == 0, f"server.install rejected -GatewayWire {w_val!r}"

    # Extract the exact `-match` templates from `scripts/bootstrap-gateway.ps1` so any drift in the
    # PowerShell script's userinfo, query, or fragment regexes fails this test immediately.
    m_userinfo = re.search(r"\$DecodedGatewayUrl -match '([^']+)'", ps1_text)
    m_templates = re.findall(r'\$DecodedGatewayUrl -match "([^"]+)"', ps1_text)
    assert m_userinfo is not None, "Could not extract userinfo -match pattern from bootstrap-gateway.ps1"
    assert len(m_templates) == 2, f"Expected 2 interpolated -match templates in bootstrap-gateway.ps1, got {m_templates}"

    userinfo_re = re.compile(m_userinfo.group(1), re.IGNORECASE)
    query_re = re.compile(
        m_templates[0].replace("$SecretParamPattern", core_pat),
        re.IGNORECASE,
    )
    frag_re = re.compile(
        m_templates[1].replace("$SecretParamPattern", core_pat),
        re.IGNORECASE,
    )

    def ps1_rejects(url: str) -> bool:
        decoded = unquote(url)
        return bool(userinfo_re.search(decoded) or query_re.search(decoded) or frag_re.search(decoded))

    secret_urls = [
        "https://user:pass@gw.example.internal/v1",
        "https://gw.example.internal/v1?key=AIzaSecret",
        "https://gw.example.internal/v1?api_key=AIzaSecret",
        "https://gw.example.internal/v1?x-api-key=AIzaSecret",
        "https://gw.example.internal/v1?apiKey=AIzaSecret",
        "https://gw.example.internal/v1?apiKey2=AIzaSecret",
        "https://gw.example.internal/v1?AWSAccessKeyId=AKIAIOSFODNN7EXAMPLE",
        "https://gw.example.internal/v1?secretName=corp-secret",
        "https://gw.example.internal/v1?passwordHash=deadbeef",
        "https://gw.example.internal/v1?tokenValue=BearerSecret",
        "https://gw.example.internal/v1?token=BearerSecret",
        "https://gw.example.internal/v1?MyAPIToken=Secret",
        "https://gw.example.internal/v1?userToken=Secret",
        "https://gw.example.internal/v1?accesstoken=Secret",
        "https://gw.example.internal/v1?apitoken=Secret",
        "https://gw.example.internal/v1?virtualkey=bf-vk-1",
        "https://gw.example.internal/v1?client_secret=Secret",
        "https://gw.example.internal/v1?X-Goog-Signature=Sig",
        "https://gw.example.internal/v1?sigv4=Sig",
        "https://gw.example.internal/v1?jwt=eyJ",
        "https://gw.example.internal/v1?pass=123",
        "https://gw.example.internal/v1?pw=123",
        "https://gw.example.internal/v1?sid=abc",
        "https://gw.example.internal/v1?assertion=saml",
        "https://gw.example.internal/v1?%74oken=Secret",
        "https://gw.example.internal/v1?region=us;token=Secret",
        "https://gw.example.internal/v1#token=Secret",
        "https://gw.example.internal/v1#apikey",
    ]
    for bad_url in secret_urls:
        assert ps1_rejects(bad_url), f"PowerShell regex missed secret URL: {bad_url}"
        with pytest.raises(ValueError, match="must not embed"):
            _validate_gateway_base_url_no_secrets(bad_url)

    benign_urls = [
        "https://litellm.example.com/vertex_ai",
        "https://litellm.example.com/gemini?region=global",
        "http://localhost:8080/genai?keyword=routing",
        "https://gw.example.internal/v1?author=alice&authority=corp",
        "https://gw.example.internal/v1?authuser=0",
        "https://gw.example.internal/v1?design=v2&signal=1&monkey=true&oauth_version=1.0",
        "https://gw.example.internal/v1?bypass=1&compass=north&sidecar=true&session_timeout=30",
        "https://gw.example.internal/v1?max_tokens=5&tokenizer=t&secretary=1&tokens=1",
    ]
    for ok_url in benign_urls:
        assert not ps1_rejects(ok_url), f"PowerShell regex false-positived on benign URL: {ok_url}"
        _validate_gateway_base_url_no_secrets(ok_url)


def test_build_genai_client_pins_v1beta_and_deduplicates_versioned_gateway_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure SDK 3.0+ keeps `/v1beta` on Gemini-wire and honors explicit `/v1` / `/v1beta` / `/v1beta1` suffixes without doubling."""
    from server.genai_client import build_genai_client, close_genai_client

    for clean_env in (
        "CC_AGY_VERTEX_BASE_URL",
        "CC_AGY_GEMINI_BASE_URL",
        "GOOGLE_VERTEX_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
        "CC_AGY_BIFROST_VIRTUAL_KEY",
    ):
        monkeypatch.delenv(clean_env, raising=False)

    # 1. Direct AI Studio API key pins `v1beta` across google-genai 2.x and 3.x
    direct_client = build_genai_client({"api_key": "AIzaSyUnitTestKey0000000000000000"})
    try:
        req_direct = direct_client._api_client._build_request(
            "post", "models/gemini-flash-next:generateContent", {}, None
        )
        assert req_direct.url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-next:generateContent"
    finally:
        close_genai_client(direct_client)

    # 2. Gemini-wire gateway URLs with and without trailing `/v1beta`, `/V1BETA`, `/v1alpha`, or `/v1` never double the version segment
    for gw_url, expected_prefix in (
        ("https://litellm.example.com/gemini", "https://litellm.example.com/gemini/v1beta/"),
        ("https://litellm.example.com/gemini/v1beta", "https://litellm.example.com/gemini/v1beta/"),
        ("https://litellm.example.com/gemini/V1BETA", "https://litellm.example.com/gemini/v1beta/"),
        ("https://litellm.example.com/genai/v1alpha", "https://litellm.example.com/genai/v1alpha/"),
        ("https://litellm.example.com/gemini/v1", "https://litellm.example.com/gemini/v1/"),
        ("http://localhost:8080/genai/v1beta/", "http://localhost:8080/genai/v1beta/"),
    ):
        gw_client = build_genai_client(
            {
                "base_url": gw_url,
                "gateway_wire": "gemini",
                "base_url_token": "sk-litellm-test",
            }
        )
        try:
            req_gw = gw_client._api_client._build_request(
                "post", "models/gemini-flash-next:generateContent", {}, None
            )
            assert req_gw.url == f"{expected_prefix}models/gemini-flash-next:generateContent"
        finally:
            close_genai_client(gw_client)

    # 3. Custom gateways with spoofed `googleapis.com` path, suffix, prefix, or userinfo still set `follow_redirects=False`
    for spoof_url in (
        "https://gw.corp.internal/proxy/generativelanguage.googleapis.com/genai",
        "https://googleapis.com.evil.example/v1beta",
        "https://evilgoogleapis.com/v1beta",
        "https://x.googleapis.com@evil.example/v1beta",
    ):
        spoof_client = build_genai_client(
            {
                "base_url": spoof_url,
                "gateway_wire": "gemini",
                "base_url_token": "sk-corp-proxy",
            }
        )
        try:
            client_args = spoof_client._api_client._http_options.client_args or {}
            assert client_args.get("follow_redirects") is False, spoof_url
        finally:
            close_genai_client(spoof_client)

    # 4. Vertex-wire gateway URLs default to `v1beta1`, honor explicit `/v1`, and preserve `/v1internal` untouched
    for vx_url, expected_ver_segment in (
        ("https://litellm.example.com/vertex_ai", "/vertex_ai/v1beta1/projects/"),
        ("https://litellm.example.com/vertex_ai/v1", "/vertex_ai/v1/projects/"),
        ("https://litellm.example.com/vertex_ai/v1beta1", "/vertex_ai/v1beta1/projects/"),
        ("https://proxy.example.com/v1internal", "/v1internal/v1beta1/projects/"),
    ):
        vx_client = build_genai_client(
            {
                "base_url": vx_url,
                "gateway_wire": "vertex",
                "project": "acme-prod",
                "location": "global",
                "base_url_token": "sk-vertex-test",
            }
        )
        try:
            req_vx = vx_client._api_client._build_request(
                "post", "publishers/google/models/gemini-flash-next:generateContent", {}, None
            )
            assert expected_ver_segment in req_vx.url
        finally:
            close_genai_client(vx_client)
