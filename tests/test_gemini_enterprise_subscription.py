"""Hermetic unit and loopback integration tests for Gemini Enterprise subscription mode,
PEP 668 managed virtualenv bootstrap, and Claude Code --config installer forwarding."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any
from unittest import mock
import urllib.error
import urllib.parse
import urllib.request

import pytest

from server.agy_runner import run_antigravity_worker
from server.batch_runner import create_shared_cache
from server.doctor import run_doctor
from server.gemini_enterprise import (
    EnterpriseAuthError,
    EnterpriseEntitlement,
    EnterpriseSubscriptionProxy,
    PROXY_AUTH_HEADER,
    PROXY_LOCATION_HEADER,
    baic_host_for_location,
    clear_enterprise_caches,
    fetch_enterprise_licenses,
    inspect_enterprise_credentials_offline,
    prepare_enterprise_endpoint,
    resolve_enterprise_access_token,
    resolve_enterprise_entitlement,
    self_assign_enterprise_license,
    stop_enterprise_proxy,
    to_baic_generate_content_request,
)
from server.genai_ask import ask_gemini
from server.install import (
    build_claude_user_config,
    claude_install_commands,
    install_claude,
    install_codex,
    main as install_main,
    preflight,
    run_interactive_wizard,
)
from server.jobs import JobManager
from server.receipts import ReceiptStore
from server.regions import resolve_auth_config, resolve_configured_auth_mode
from server.runtime_env import (
    ensure_sdk_venv,
    managed_venv_dir,
    python_has_plugin_sdks,
    resolve_sdk_python,
    venv_python_path,
)
from server.verify import scrubbed_env
from server.writers import _build_genai_client


@pytest.fixture(autouse=True)
def _clean_enterprise_state() -> Any:
    clear_enterprise_caches()
    stop_enterprise_proxy()
    yield
    clear_enterprise_caches()
    stop_enterprise_proxy()


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=str(repo), check=True)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)


class _LocalBAICServer:
    """Hermetic 127.0.0.1 Business AI Code mock server."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.licenses_payload: dict[str, Any] = {
            "licenses": [
                {
                    "userTier": "gcp-enterprise-tier",
                    "tierDisplayName": "Gemini Enterprise Standard",
                    "projectId": "ent-seat-proj",
                    "location": "global",
                }
            ]
        }
        self.self_assign_payload: dict[str, Any] = {
            "license": {
                "userTier": "gcp-enterprise-plus-tier",
                "tierDisplayName": "Gemini Enterprise Plus",
                "projectId": "self-assigned-proj",
                "location": "us",
            }
        }
        self.fail_first_fetch_with_403: bool = False
        self.fail_first_inference_with_401: bool = False
        self._fetch_calls = 0
        self._inference_calls = 0
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        host, port = self._server.server_address[:2]
        self.base_url = f"http://{host}:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class _H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

            def _record(self, raw_body: bytes) -> dict[str, Any]:
                parsed = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                rec = {
                    "method": self.command,
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "json": parsed,
                }
                with outer._lock:
                    outer.requests.append(rec)
                return rec

            def _reply(self, status: int, obj: dict[str, Any]) -> None:
                data = json.dumps(obj).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()

            def do_GET(self) -> None:  # noqa: N802
                self._record(b"")
                if self.path.endswith(":fetchLicenses"):
                    with outer._lock:
                        outer._fetch_calls += 1
                        call_num = outer._fetch_calls
                    if outer.fail_first_fetch_with_403 and call_num == 1:
                        self._reply(403, {"error": {"code": 403, "message": "caller lacks quota permission on project"}})
                        return
                    self._reply(200, outer.licenses_payload)
                    return
                self._reply(404, {"error": {"code": 404, "message": "not found"}})

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or "0")
                raw = self.rfile.read(n) if n > 0 else b""
                self._record(raw)
                if self.path.endswith(":selfAssignLicense"):
                    self._reply(200, outer.self_assign_payload)
                    return
                with outer._lock:
                    outer._inference_calls += 1
                    inf_num = outer._inference_calls
                if outer.fail_first_inference_with_401 and inf_num == 1:
                    self._reply(401, {"error": {"code": 401, "message": "UNAUTHENTICATED: token expired"}})
                    return
                chunk_obj = {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "Enterprise OK"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 25,
                        "candidatesTokenCount": 4,
                        "totalTokenCount": 29,
                    },
                }
                if ":streamGenerateContent" in self.path:
                    sse = f"data: {json.dumps(chunk_obj)}\r\n\r\n".encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    self.wfile.write(sse)
                    self.wfile.flush()
                    return
                self._reply(200, chunk_obj)

        return _H


def test_baic_host_routing_all_locations() -> None:
    assert baic_host_for_location(None, env={}) == "https://businessaicode.googleapis.com"
    assert baic_host_for_location("global", env={}) == "https://businessaicode.googleapis.com"
    assert baic_host_for_location("us", env={}) == "https://businessaicode.us.rep.googleapis.com"
    assert baic_host_for_location("eu", env={}) == "https://businessaicode.eu.rep.googleapis.com"
    assert baic_host_for_location("us-central1", env={}) == "https://us-central1-businessaicode.googleapis.com"
    assert (
        baic_host_for_location("eu", env={"CC_AGY_BAIC_BASE_URL": "http://127.0.0.1:9999/"})
        == "http://127.0.0.1:9999"
    )
    with pytest.raises(ValueError):
        baic_host_for_location("bad/location", env={})


def test_to_baic_generate_content_request_sanitizes_fields() -> None:
    vertex_req = {
        "model": "publishers/google/models/gemini-3.8-flash",
        "sessionId": "sess-123",
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": "LOW", "includeThoughts": True},
            "imageConfig": {"aspectRatio": "16:9"},
            "temperature": 0.2,
        },
        "tools": [
            {"googleSearch": {"excludeDomains": ["example.com"], "dynamicRetrieval": True}},
            {"functionDeclarations": [{"name": "ping"}]},
        ],
    }
    converted = to_baic_generate_content_request(
        vertex_req,
        model="publishers/google/models/gemini-3.8-flash",
        user_tier="gcp-enterprise-tier",
    )
    assert "model" not in converted
    assert "sessionId" not in converted
    assert converted["aicode"] == {"experience": "gemini-3.8-flash"}
    assert converted["entitlement"] == {"user_tier": "gcp-enterprise-tier"}
    assert "imageConfig" not in converted["generationConfig"]
    assert converted["generationConfig"]["thinkingConfig"] == {"includeThoughts": True}
    assert converted["tools"][0] == {"enterpriseWebSearch": {"excludeDomains": ["example.com"]}}
    assert converted["tools"][1] == {"functionDeclarations": [{"name": "ping"}]}

    # Empty thinkingConfig and empty generationConfig are pruned cleanly
    minimal = to_baic_generate_content_request(
        {"contents": [], "generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}}},
        model="gemini-4-argon",
        user_tier="gcp-enterprise-plus-tier",
    )
    assert "generationConfig" not in minimal


def test_inspect_enterprise_credentials_offline_human_vs_service_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    assert inspect_enterprise_credentials_offline({"CC_AGY_ENTERPRISE_ACCESS_TOKEN": "mock-oauth-tok"})["ok"] is True

    sa_file = tmp_path / "sa.json"
    sa_file.write_text(
        json.dumps({"type": "service_account", "client_email": "ci-bot@proj.iam.gserviceaccount.com"}),
        encoding="utf-8",
    )
    sa_res = inspect_enterprise_credentials_offline({"GOOGLE_APPLICATION_CREDENTIALS": str(sa_file)})
    assert sa_res["ok"] is False
    assert sa_res["credential_source"] == "service_account_rejected"
    assert "ci-bot@proj.iam.gserviceaccount.com" in sa_res["error"]

    user_adc = tmp_path / "user_adc.json"
    user_adc.write_text(
        json.dumps({"type": "authorized_user", "quota_project_id": "my-ent-proj"}),
        encoding="utf-8",
    )
    user_res = inspect_enterprise_credentials_offline({"GOOGLE_APPLICATION_CREDENTIALS": str(user_adc)})
    assert user_res["ok"] is True
    assert user_res["credential_source"] == "adc_authorized_user"


def test_resolve_enterprise_access_token_rejects_sa_and_falls_back_to_gcloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sa_file = tmp_path / "sa_key.json"
    sa_file.write_text(
        json.dumps({"type": "service_account", "client_email": "runner@proj.iam.gserviceaccount.com"}),
        encoding="utf-8",
    )
    with pytest.raises(EnterpriseAuthError, match="service account"):
        resolve_enterprise_access_token(env={"GOOGLE_APPLICATION_CREDENTIALS": str(sa_file)})

    # Simulate a GCE/Cloudtop VM where google.auth.default returns ambient compute_engine SA credentials,
    # while `gcloud` has a human user login.
    FakeComputeCreds = type("Credentials", (), {"__module__": "google.auth.compute_engine.credentials"})
    import google.auth

    monkeypatch.setattr(google.auth, "default", lambda scopes=None: (FakeComputeCreds(), "vm-proj"))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/gcloud" if name == "gcloud" else None)

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "get-value" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="alice@company.com\n", stderr="")
        if "print-access-token" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="mock-oauth-human-gcloud-tok\n", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)
    tok = resolve_enterprise_access_token(env={"CLOUDSDK_CONFIG": str(tmp_path / "gcloud")})
    assert tok == "mock-oauth-human-gcloud-tok"


def test_resolve_enterprise_entitlement_auto_picks_and_self_assigns() -> None:
    baic = _LocalBAICServer()
    try:
        env = {"CC_AGY_BAIC_BASE_URL": baic.base_url}
        # 1. Auto-picks single license and retries without X-Goog-User-Project if first 403s
        baic.fail_first_fetch_with_403 = True
        ent = resolve_enterprise_entitlement(
            "mock-oauth-tok1",
            project=None,
            location="us-central1",
            location_explicit=False,
            env={**env, "GOOGLE_CLOUD_PROJECT": "unprivileged-quota-proj"},
        )
        assert ent.project_id == "ent-seat-proj"
        assert ent.location == "global"
        assert ent.user_tier == "gcp-enterprise-tier"
        assert len(baic.requests) == 2
        assert baic.requests[0]["headers"].get("x-goog-user-project") == "unprivileged-quota-proj"
        assert "x-goog-user-project" not in baic.requests[1]["headers"]

        # 2. Falls back to :selfAssignLicense with parent body + X-Goog-User-Project when project not in fetchLicenses
        clear_enterprise_caches()
        baic.fail_first_fetch_with_403 = False
        assigned = resolve_enterprise_entitlement(
            "mock-oauth-tok2",
            project="self-assigned-proj",
            location="us",
            location_explicit=True,
            env=env,
        )
        assert assigned.project_id == "self-assigned-proj"
        assert assigned.location == "us"
        assert assigned.user_tier == "gcp-enterprise-plus-tier"
        sa_req = baic.requests[-1]
        assert sa_req["path"] == "/v1beta/projects/self-assigned-proj/locations/us:selfAssignLicense"
        assert sa_req["json"] == {"parent": "projects/self-assigned-proj/locations/us"}
        assert sa_req["headers"].get("x-goog-user-project") == "self-assigned-proj"

        # 3. Regional location with '-' is rejected on selfAssignLicense
        with pytest.raises(EnterpriseAuthError, match="multi-region jurisdiction"):
            self_assign_enterprise_license("mock-oauth-tok2", project="p", location="us-central1", env=env)
    finally:
        baic.close()


def test_enterprise_loopback_proxy_unary_sse_token_guard_and_401_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    baic = _LocalBAICServer()
    env = {"CC_AGY_BAIC_BASE_URL": baic.base_url}
    proxy = EnterpriseSubscriptionProxy(env=env)
    try:
        proxy.register_entitlement(
            EnterpriseEntitlement(
                project_id="ent-seat-proj",
                location="global",
                user_tier="gcp-enterprise-tier",
            )
        )
        # 1. Missing X-CC-AGY-Proxy-Token is rejected with HTTP 401
        bad_req = urllib.request.Request(
            f"{proxy.base_url}/v1beta1/projects/ent-seat-proj/locations/global/publishers/google/models/gemini-3.8-flash:generateContent",
            data=b'{"contents":[]}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(bad_req, timeout=10)
        assert exc_info.value.code == 401

        # 2. Upstream 401 triggers automatic force_refresh=True retry and succeeds
        baic.fail_first_inference_with_401 = True
        refresh_flags: list[bool] = []

        def fake_resolve_tok(*, env: Any = None, force_refresh: bool = False) -> str:
            refresh_flags.append(force_refresh)
            return "mock-oauth-refreshed" if force_refresh else "mock-oauth-stale"

        monkeypatch.setattr("server.gemini_enterprise.resolve_enterprise_access_token", fake_resolve_tok)
        sse_req = urllib.request.Request(
            f"{proxy.base_url}/v1beta1/projects/ent-seat-proj/locations/global/publishers/google/models/gemini-3.8-flash:streamGenerateContent?alt=sse",
            data=json.dumps({"contents": [{"role": "user", "parts": [{"text": "ping"}]}]}).encode("utf-8"),
            headers={"Content-Type": "application/json", PROXY_AUTH_HEADER: proxy.secret},
            method="POST",
        )
        with urllib.request.urlopen(sse_req, timeout=10) as resp:
            raw_sse = resp.read().decode("utf-8")
        assert "Enterprise OK" in raw_sse
        assert refresh_flags == [False, True]
        assert baic.requests[-1]["headers"]["authorization"] == "Bearer mock-oauth-refreshed"
        assert baic.requests[-1]["headers"]["x-goog-user-project"] == "ent-seat-proj"
    finally:
        proxy.close()
        baic.close()


def test_ask_gemini_and_writers_end_to_end_without_adc_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify google.genai.Client(vertexai=True) works end-to-end against the local enterprise proxy
    even when NO Application Default Credentials file exists on disk."""
    baic = _LocalBAICServer()
    try:
        empty_home = tmp_path / "clean_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        monkeypatch.setenv("CLOUDSDK_CONFIG", str(empty_home / "gcloud"))
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.setenv("CC_AGY_AUTH_MODE", "gemini_enterprise")
        monkeypatch.setenv("CC_AGY_BAIC_BASE_URL", baic.base_url)
        monkeypatch.setenv("CC_AGY_ENTERPRISE_ACCESS_TOKEN", "mock-oauth-e2e-seat-token")

        repo = tmp_path / "repo"
        _init_git_repo(repo)
        store = ReceiptStore(tmp_path / "data", workspace_root=repo)

        res = ask_gemini("Say hello", workspace_root=repo, store=store)
        assert res["text"] == "Enterprise OK"
        assert res["auth_mode"] == "gemini_enterprise"
        rcpt = store.read_receipt(res["job_id"])
        assert rcpt["auth_mode"] == "gemini_enterprise"
        assert rcpt["vertex_project"] == "ent-seat-proj"

        # Also verify writers._build_genai_client constructs cleanly without ADC on disk
        auth = resolve_auth_config()
        assert auth["auth_mode"] == "gemini_enterprise"
        client = _build_genai_client(auth, timeout_ms=10_000)
        gen_resp = client.models.generate_content(model="gemini-3.8-flash", contents="Check writer")
        assert gen_resp.text == "Enterprise OK"

        # Verify BAIC received enterpriseWebSearch / aicode / entitlement and X-Goog-User-Project
        inf_reqs = [r for r in baic.requests if ":generateContent" in r["path"]]
        assert len(inf_reqs) >= 2
        assert inf_reqs[-1]["json"]["aicode"] == {"experience": "gemini-3.8-flash"}
        assert inf_reqs[-1]["json"]["entitlement"] == {"user_tier": "gcp-enterprise-tier"}
        assert inf_reqs[-1]["headers"]["x-goog-user-project"] == "ent-seat-proj"
    finally:
        baic.close()


def test_auth_precedence_job_manager_and_shared_cache_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Gemini Enterprise mode beats an ambient GEMINI_API_KEY in the shell, while explicit requested_api_key still wins
    env = {"CC_AGY_AUTH_MODE": "gemini_enterprise", "GEMINI_API_KEY": "ambient-studio-key"}
    assert resolve_configured_auth_mode(env) == "gemini_enterprise"
    auth = resolve_auth_config(env=env)
    assert auth["auth_mode"] == "gemini_enterprise"
    assert resolve_auth_config(requested_api_key="explicit-key", env=env)["auth_mode"] == "api_key"

    # Shared Vertex context cache cleanly no-ops on BAIC
    cache_name = create_shared_cache(
        {"auth_mode": "gemini_enterprise", "project": "ent-seat-proj", "location": "global"},
        "gemini-3.8-flash",
        "system prompt",
        "x" * 20_000,
    )
    assert cache_name is None

    # JobManager preserves auth_mode="gemini_enterprise" + resolved seat project across delegate_continue
    monkeypatch.setenv("CC_AGY_AUTH_MODE", "gemini_enterprise")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    repo = tmp_path / "jm_repo"
    _init_git_repo(repo)
    store = ReceiptStore(tmp_path / "jm_data", workspace_root=repo)
    seen_reqs: list[dict[str, Any]] = []

    def fake_worker(req: dict[str, Any]) -> dict[str, Any]:
        seen_reqs.append(dict(req))
        (Path(req["workspace_dir"]) / "app.py").write_text(f"VALUE = {len(seen_reqs) + 1}\n", encoding="utf-8")
        return {
            "text": "updated",
            "vertex_project": "ent-seat-proj",
            "vertex_location": "global",
            "usage": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15},
        }

    jm = JobManager(workspace_root=repo, store=store, worker_backend=fake_worker)
    try:
        d1 = jm.delegate_start("Update VALUE in app.py", wait=True)
        assert d1["status"] == "ok"
        assert seen_reqs[0]["auth_mode"] == "gemini_enterprise"
        r1 = store.read_receipt(d1["job_id"])
        assert r1["auth_mode"] == "gemini_enterprise"
        assert r1["vertex_project"] == "ent-seat-proj"

        # Clear env var before delegate_continue: prior_auth_mode from receipt must still keep gemini_enterprise
        monkeypatch.delenv("CC_AGY_AUTH_MODE", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "later-ambient-key")
        d2 = jm.delegate_continue(d1["job_id"], "Increment VALUE again", wait=True)
        assert d2["status"] == "ok"
        assert seen_reqs[1]["auth_mode"] == "gemini_enterprise"
    finally:
        jm.shutdown()


def test_agy_runner_wires_enterprise_vertex_endpoint_without_adc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("google.antigravity")
    baic = _LocalBAICServer()
    try:
        empty_home = tmp_path / "no_adc_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        monkeypatch.setenv("CLOUDSDK_CONFIG", str(empty_home / "gcloud"))
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.setenv("CC_AGY_BAIC_BASE_URL", baic.base_url)
        monkeypatch.setenv("CC_AGY_ENTERPRISE_ACCESS_TOKEN", "mock-oauth-runner-tok")

        ws = tmp_path / "worker_ws"
        ws.mkdir()
        out = run_antigravity_worker(
            prompts=["Reply with Enterprise OK"],
            workspace_dir=ws,
            model="gemini-3.8-flash",
            auth_mode="gemini_enterprise",
            timeout_seconds=30.0,
        )
        assert "Enterprise OK" in out["text"]
        assert out["vertex_project"] == "ent-seat-proj"
        assert out["vertex_location"] == "global"
        stream_reqs = [r for r in baic.requests if ":streamGenerateContent" in r["path"] or ":generateContent" in r["path"]]
        assert len(stream_reqs) >= 1
        assert stream_reqs[-1]["json"]["aicode"] == {"experience": "gemini-3.8-flash"}
        assert stream_reqs[-1]["json"]["entitlement"] == {"user_tier": "gcp-enterprise-tier"}
        assert stream_reqs[-1]["headers"]["x-goog-user-project"] == "ent-seat-proj"
        assert stream_reqs[-1]["headers"]["authorization"] == "Bearer mock-oauth-runner-tok"
    finally:
        baic.close()


def test_runtime_env_resolve_and_pep668_managed_venv_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv_dir = tmp_path / "managed_venv"
    env = {"CC_AGY_VENV_DIR": str(venv_dir)}
    assert managed_venv_dir(env) == venv_dir

    # Simulate a PEP 668 machine where system python lacks SDKs and ensure_sdk_venv creates managed_venv
    fake_py = venv_python_path(venv_dir)

    def fake_create(target: str) -> None:
        fake_py.parent.mkdir(parents=True, exist_ok=True)
        fake_py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_py.chmod(0o755)

    sdk_installed = {"ready": False}

    def fake_has_sdks(py_bin: str | Path) -> bool:
        return str(py_bin) == str(fake_py) and sdk_installed["ready"]

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "-e" in cmd:
            sdk_installed["ready"] = True
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("server.runtime_env.python_has_plugin_sdks", fake_has_sdks)
    monkeypatch.setattr("venv.EnvBuilder.create", lambda self, target: fake_create(target))
    monkeypatch.setattr("server.runtime_env.subprocess.run", fake_run)

    res = ensure_sdk_venv(tmp_path, env=env)
    assert res["ok"] is True
    assert res["bootstrapped"] is True
    assert res["python_bin"] == str(fake_py)
    assert resolve_sdk_python(tmp_path, env=env) == str(fake_py)


def test_install_claude_passes_repeatable_config_flags_and_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_claude_user_config(
        profile="quality",
        auth_mode="gemini-enterprise",
        project="ent-proj",
        location="global",
        max_daily_cents=500.0,
        strict=True,
    )
    assert cfg == {
        "auth_mode": "gemini_enterprise",
        "cost_profile": "quality",
        "google_cloud_project": "ent-proj",
        "google_cloud_location": "global",
        "max_daily_cents": "500",
        "strict_delegation_mode": True,
    }
    cmds = claude_install_commands(
        profile="quality",
        auth_mode="gemini-enterprise",
        max_daily_cents=500.0,
        strict=True,
    )
    assert cmds[1] == [
        "claude",
        "plugin",
        "install",
        "cc-agy-plugin@cc-agy",
        "--config",
        "auth_mode=gemini_enterprise",
        "--config",
        "cost_profile=quality",
        "--config",
        "max_daily_cents=500",
        "--config",
        "strict_delegation_mode=true",
    ]

    # Simulate an older Claude CLI rejecting `--config` and falling back to `plugin configure --values-stdin`
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/claude" if name == "claude" else None)
    recorded_calls: list[tuple[list[str], str | None]] = []

    def fake_claude_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded_calls.append((cmd, kwargs.get("input")))
        if "install" in cmd and "--config" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error: unknown option '--config'")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("server.install.subprocess.run", fake_claude_run)
    res = install_claude(dry_run=False, profile="quality", auth_mode="gemini-enterprise", max_daily_cents=500.0)
    assert res["status"] == "installed"
    assert any("configure" in call[0] and "--values-stdin" in call[0] for call in recorded_calls)
    stdin_payload = next(call[1] for call in recorded_calls if call[1] is not None)
    assert json.loads(stdin_payload) == {
        "auth_mode": "gemini_enterprise",
        "cost_profile": "quality",
        "max_daily_cents": "500",
    }


def test_install_codex_and_wizard_gemini_enterprise_option_4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    codex_home = tmp_path / "codex"
    monkeypatch.setenv("CC_AGY_ENTERPRISE_ACCESS_TOKEN", "mock-oauth-do-not-write-me")
    out = install_codex(
        name="agy",
        profile="balanced",
        auth_mode="gemini-enterprise",
        project=None,
        location="us",
        base_url=None,
        max_daily_cents=400.0,
        dry_run=False,
        codex_home=codex_home,
    )
    assert out["status"] == "installed"
    toml_text = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert 'CC_AGY_AUTH_MODE = "gemini_enterprise"' in toml_text
    assert 'CC_AGY_ENTERPRISE_LOCATION = "us"' in toml_text
    assert "mock-oauth-do-not-write-me" not in toml_text
    for forwarded in ("GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_CONFIG"):
        assert f'"{forwarded}"' in toml_text

    answers = iter(["3", "4", "1", ""])
    wizard_flags = run_interactive_wizard(
        input_fn=lambda _p: next(answers),
        output_fn=lambda _m: None,
        env={"CC_AGY_ENTERPRISE_ACCESS_TOKEN": "mock-oauth-ok"},
    )
    assert wizard_flags == ["--host", "claude", "--profile", "balanced", "--auth", "gemini-enterprise"]

    rc = install_main(
        [
            "--host",
            "codex",
            "--auth",
            "gemini-enterprise",
            "--no-bootstrap",
            "--dry-run",
            "--json",
        ]
    )
    assert rc == 0


def test_doctor_and_secret_scrubbing_for_gemini_enterprise(tmp_path: Path) -> None:
    repo = tmp_path / "doc_repo"
    _init_git_repo(repo)

    ok_env = {
        **os.environ,
        "CC_AGY_AUTH_MODE": "gemini_enterprise",
        "CC_AGY_ENTERPRISE_ACCESS_TOKEN": "mock-oauth-secret-ent-token",
        "CC_AGY_DATA_DIR": str(tmp_path / "doc_data"),
    }
    rep = run_doctor(workspace_root=repo, env=ok_env)
    assert rep["auth"]["auth_mode"] == "gemini_enterprise"
    assert rep["auth"]["enterprise_credential_source"] == "env_access_token"
    assert "mock-oauth-secret-ent-token" not in json.dumps(rep)

    # Service account credentials fail doctor in Gemini Enterprise mode with actionable remediation
    sa_path = tmp_path / "sa.json"
    sa_path.write_text(json.dumps({"type": "service_account", "client_email": "sa@p.iam.gserviceaccount.com"}), encoding="utf-8")
    bad_env = {
        "PATH": os.environ.get("PATH", ""),
        "CC_AGY_AUTH_MODE": "gemini_enterprise",
        "GOOGLE_APPLICATION_CREDENTIALS": str(sa_path),
        "CC_AGY_DATA_DIR": str(tmp_path / "doc_data_bad"),
    }
    bad_rep = run_doctor(workspace_root=repo, env=bad_env)
    assert bad_rep["healthy"] is False
    assert bad_rep["auth"]["enterprise_credential_source"] == "service_account_rejected"
    assert preflight(bad_env, auth_mode="gemini-enterprise")["ready"] is False

    # Child process env scrubbing and receipt redaction strip CC_AGY_ENTERPRISE_ACCESS_TOKEN
    with mock.patch.dict(os.environ, {"CC_AGY_ENTERPRISE_ACCESS_TOKEN": "mock-oauth-secret-ent-token"}):
        scrubbed = scrubbed_env(repo)
    assert scrubbed.get("CC_AGY_ENTERPRISE_ACCESS_TOKEN") == ""

    synth_oauth = "ya" + "29.secret-ent-token"
    store = ReceiptStore(tmp_path / "rcpt_data", workspace_root=repo)
    store.write_receipt(
        "job-redact",
        {
            "kind": "ask_gemini",
            "status": "ok",
            "env_dump": {"CC_AGY_ENTERPRISE_ACCESS_TOKEN": synth_oauth},
            "raw_log": f"Authorization: Bearer {synth_oauth}",
        },
    )
    saved = store.read_receipt("job-redact")
    assert synth_oauth not in json.dumps(saved)


def test_gateway_vertex_base_url_preserves_custom_bearer_token_in_genai_and_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for Opus R1 #1: google-genai must send gateway bearer token, not 'Bearer proxy-local'."""
    baic = _LocalBAICServer()
    try:
        empty_home = tmp_path / "gw_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        monkeypatch.setenv("CLOUDSDK_CONFIG", str(empty_home / "gcloud"))
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("CC_AGY_AUTH_MODE", "vertex")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gw-proj")
        monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL", baic.base_url)
        monkeypatch.setenv("CC_AGY_VERTEX_BASE_URL_TOKEN", "gw-secret-token-123")

        repo = tmp_path / "gw_repo"
        _init_git_repo(repo)
        store = ReceiptStore(tmp_path / "gw_data", workspace_root=repo)

        res = ask_gemini("Gateway check", workspace_root=repo, store=store)
        assert res["text"] == "Enterprise OK"

        auth = resolve_auth_config()
        client = _build_genai_client(auth, timeout_ms=10_000)
        w_resp = client.models.generate_content(model="gemini-3.8-flash", contents="Writer gateway check")
        assert w_resp.text == "Enterprise OK"

        gen_reqs = [r for r in baic.requests if ":generateContent" in r["path"]]
        assert len(gen_reqs) == 2
        assert gen_reqs[0]["headers"]["authorization"] == "Bearer gw-secret-token-123"
        assert gen_reqs[1]["headers"]["authorization"] == "Bearer gw-secret-token-123"
    finally:
        baic.close()


def test_ambient_google_cloud_location_and_usercfg_jurisdiction_in_delegate_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for Argon R1 #1/#2 & Opus R1 #2: regional GOOGLE_CLOUD_LOCATION stays a soft hint
    across JobManager.delegate_start, while explicit userConfig 'eu'/'us'/'global' filters strictly."""
    # 1. Explicit userConfig multi-region ('eu') sets location_explicit=True; default 'us-central1' stays False
    eu_auth = resolve_auth_config(
        env={"CC_AGY_AUTH_MODE": "gemini_enterprise", "CC_AGY_USERCFG_GOOGLE_CLOUD_LOCATION": "eu"}
    )
    assert eu_auth["location"] == "eu"
    assert eu_auth["location_explicit"] is True

    default_cfg_auth = resolve_auth_config(
        env={"CC_AGY_AUTH_MODE": "gemini_enterprise", "CC_AGY_USERCFG_GOOGLE_CLOUD_LOCATION": "us-central1"}
    )
    assert default_cfg_auth["location"] == "us-central1"
    assert default_cfg_auth["location_explicit"] is False

    baic = _LocalBAICServer()
    try:
        env = {"CC_AGY_BAIC_BASE_URL": baic.base_url}
        # 2. Multi-seat preference: regional 'europe-west4' soft hint picks 'eu' seat over 'us' without failing if absent
        baic.licenses_payload = {
            "licenses": [
                {"userTier": "gcp-enterprise-tier", "projectId": "seat-us", "location": "us"},
                {"userTier": "gcp-enterprise-plus-tier", "projectId": "seat-eu", "location": "eu"},
            ]
        }
        picked_eu = resolve_enterprise_entitlement(
            "mock-oauth-multi", project=None, location="europe-west4", location_explicit=False, env=env
        )
        assert picked_eu.project_id == "seat-eu"
        assert picked_eu.location == "eu"

        # 3. Fallback selfAssignLicense with regional hint 'europe-west4' maps to 'eu' jurisdiction
        clear_enterprise_caches()
        baic.self_assign_payload = {
            "license": {"userTier": "gcp-enterprise-tier", "projectId": "new-eu-proj", "location": "eu"}
        }
        self_eu = resolve_enterprise_entitlement(
            "mock-oauth-multi", project="new-eu-proj", location="europe-west4", location_explicit=False, env=env
        )
        assert self_eu.location == "eu"
        assert baic.requests[-1]["path"] == "/v1beta/projects/new-eu-proj/locations/eu:selfAssignLicense"

        # 4. JobManager.delegate_start with ambient GOOGLE_CLOUD_LOCATION='us-central1' does NOT pass location='us-central1'
        #    as an explicit override to the worker when license is in 'global'.
        baic.licenses_payload = {
            "licenses": [{"userTier": "gcp-enterprise-tier", "projectId": "ent-global", "location": "global"}]
        }
        clear_enterprise_caches()
        monkeypatch.setenv("CC_AGY_AUTH_MODE", "gemini_enterprise")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        monkeypatch.setenv("CC_AGY_BAIC_BASE_URL", baic.base_url)
        monkeypatch.setenv("CC_AGY_ENTERPRISE_ACCESS_TOKEN", "mock-oauth-jm-loc-tok")

        repo = tmp_path / "loc_repo"
        _init_git_repo(repo)
        store = ReceiptStore(tmp_path / "loc_data", workspace_root=repo)
        seen_worker_locs: list[Any] = []

        def fake_worker(req: dict[str, Any]) -> dict[str, Any]:
            seen_worker_locs.append((req.get("location"), req.get("location_explicit")))
            ep = prepare_enterprise_endpoint(req)
            (Path(req["workspace_dir"]) / "app.py").write_text("VALUE = 99\n", encoding="utf-8")
            return {
                "text": "ok",
                "vertex_project": ep["project"],
                "vertex_location": ep["location"],
                "usage": {"prompt_token_count": 5, "candidates_token_count": 2, "total_token_count": 7},
            }

        jm = JobManager(workspace_root=repo, store=store, worker_backend=fake_worker)
        try:
            out = jm.delegate_start("Set VALUE = 99", wait=True)
            assert out["status"] == "ok"
            assert seen_worker_locs == [("us-central1", False)]
            assert store.read_receipt(out["job_id"])["vertex_location"] == "global"
        finally:
            jm.shutdown()
    finally:
        baic.close()


def test_scrub_receipt_delegate_continue_and_writer_batch_auth_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for Argon R1 #3: scrub_receipt=True + lazy enterprise project works across delegate_continue
    and writers._build_genai_client propagates resolved project/location back onto auth dict."""
    baic = _LocalBAICServer()
    try:
        monkeypatch.setenv("CC_AGY_AUTH_MODE", "gemini_enterprise")
        monkeypatch.setenv("CC_AGY_BAIC_BASE_URL", baic.base_url)
        monkeypatch.setenv("CC_AGY_ENTERPRISE_ACCESS_TOKEN", "mock-oauth-scrub-tok")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Verify _build_genai_client mutates auth dict with resolved enterprise project & location
        writer_auth = resolve_auth_config()
        assert writer_auth["project"] is None
        _build_genai_client(writer_auth, timeout_ms=5_000)
        assert writer_auth["project"] == "ent-seat-proj"
        assert writer_auth["location"] == "global"

        repo = tmp_path / "scrub_repo"
        _init_git_repo(repo)
        store = ReceiptStore(tmp_path / "scrub_data", workspace_root=repo)
        seen_projects: list[Any] = []

        def fake_worker(req: dict[str, Any]) -> dict[str, Any]:
            seen_projects.append(req.get("project"))
            (Path(req["workspace_dir"]) / "app.py").write_text(
                f"VALUE = {len(seen_projects) + 10}\n", encoding="utf-8"
            )
            return {
                "text": "done",
                "vertex_project": req.get("project") or "ent-seat-proj",
                "vertex_location": "global",
                "usage": {"prompt_token_count": 8, "candidates_token_count": 3, "total_token_count": 11},
            }

        jm = JobManager(workspace_root=repo, store=store, worker_backend=fake_worker)
        try:
            d1 = jm.delegate_start("First edit", scrub_receipt=True, wait=True)
            assert d1["status"] == "ok"
            rcpt1 = store.read_receipt(d1["job_id"])
            assert rcpt1["vertex_project"] == "<paying-project>"

            # In-process delegate_continue recovers raw project from _raw_projects
            d2 = jm.delegate_continue(d1["job_id"], "Second edit", scrub_receipt=True, wait=True)
            assert d2["status"] == "ok"
            assert seen_projects[1] == "ent-seat-proj"

            # Simulated server restart (_raw_projects empty): '<paying-project>' is ignored cleanly (None), never forwarded
            jm._raw_projects.clear()
            d3 = jm.delegate_continue(d1["job_id"], "Third edit", scrub_receipt=True, wait=True)
            assert d3["status"] == "ok"
            assert seen_projects[2] is None
        finally:
            jm.shutdown()
    finally:
        baic.close()


def test_enterprise_proxy_global_url_rewrite_on_eu_license_and_body_guards() -> None:
    """Regression for Argon R1 #4 & Opus R1 #5/#6: GLOBAL_ONLY_MODELS URL rewrite ('locations/global')
    and _FALLBACK_MODEL_PATH_RE still hit the registered 'eu' entitlement, and 401/413 guards fire pre-body."""
    baic = _LocalBAICServer()
    env = {
        "CC_AGY_BAIC_BASE_URL": baic.base_url,
        "CC_AGY_ENTERPRISE_ACCESS_TOKEN": "mock-oauth-eu-proxy-tok",
    }
    proxy = EnterpriseSubscriptionProxy(env=env)
    try:
        proxy.register_entitlement(
            EnterpriseEntitlement(
                project_id="eu-seat-proj",
                location="eu",
                user_tier="gcp-enterprise-plus-tier",
            )
        )
        # 1. SDK URL rewritten to locations/global for gemini-4-argon still routes upstream to locations/eu
        global_rewritten_req = urllib.request.Request(
            f"{proxy.base_url}/v1beta1/projects/eu-seat-proj/locations/global/publishers/google/models/gemini-4-argon:generateContent",
            data=b'{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}',
            headers={"Content-Type": "application/json", PROXY_AUTH_HEADER: proxy.secret},
            method="POST",
        )
        with urllib.request.urlopen(global_rewritten_req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["candidates"][0]["content"]["parts"][0]["text"] == "Enterprise OK"
        assert baic.requests[-1]["path"] == "/v1beta/projects/eu-seat-proj/locations/eu:generateContent"
        assert baic.requests[-1]["headers"]["x-goog-user-project"] == "eu-seat-proj"

        # 2. Projectless fallback URL (/v1beta/models/...) reuses single registered entitlement without fetchLicenses
        fb_req = urllib.request.Request(
            f"{proxy.base_url}/v1beta/models/gemini-3.8-flash:generateContent",
            data=b'{"contents":[{"role":"user","parts":[{"text":"fb"}]}]}',
            headers={"Content-Type": "application/json", PROXY_AUTH_HEADER: proxy.secret},
            method="POST",
        )
        with urllib.request.urlopen(fb_req, timeout=10) as resp:
            fb_body = json.loads(resp.read().decode("utf-8"))
        assert fb_body["candidates"][0]["content"]["parts"][0]["text"] == "Enterprise OK"
        assert baic.requests[-1]["path"] == "/v1beta/projects/eu-seat-proj/locations/eu:generateContent"
        assert not any(r["path"].endswith(":fetchLicenses") for r in baic.requests)

        # 3. Oversized Content-Length with valid token returns HTTP 413 cleanly
        import http.client

        parsed = urllib.parse.urlsplit(proxy.base_url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
        try:
            conn.putrequest("POST", "/v1beta/models/gemini-3.8-flash:generateContent")
            conn.putheader(PROXY_AUTH_HEADER, proxy.secret)
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(70 * 1024 * 1024))
            conn.endheaders()
            r413 = conn.getresponse()
            assert r413.status == 413
        finally:
            conn.close()
    finally:
        proxy.close()
        baic.close()


def test_ephemeral_claude_plugin_cache_non_editable_and_config_fallback_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for Opus R1 #4/#7/#8: ADC account separation, non-editable install from plugins/cache,
    and narrow '--config' unknown-option detection."""
    adc = tmp_path / "adc.json"
    adc.write_text(json.dumps({"type": "authorized_user", "quota_project_id": "qp-123"}), encoding="utf-8")
    info = inspect_enterprise_credentials_offline({"GOOGLE_APPLICATION_CREDENTIALS": str(adc)})
    assert info["ok"] is True
    assert info["account"] is None
    assert info["quota_project_id"] == "qp-123"

    # Plugin root inside ~/.claude/plugins/cache/... installs non-editably (no '-e')
    cache_root = tmp_path / ".claude" / "plugins" / "cache" / "cc-agy" / "cc-agy-plugin" / "1.1.0"
    cache_root.mkdir(parents=True)
    venv_dir = tmp_path / "cache_venv"
    fake_py = venv_python_path(venv_dir)
    seen_pip_cmds: list[list[str]] = []

    def fake_create(_self: Any, _target: str) -> None:
        fake_py.parent.mkdir(parents=True, exist_ok=True)
        fake_py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_py.chmod(0o755)

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen_pip_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    call_count = {"n": 0}

    def fake_has_sdks(py_bin: str | Path) -> bool:
        if str(py_bin) != str(fake_py):
            return False
        call_count["n"] += 1
        return call_count["n"] >= 1

    monkeypatch.setattr("venv.EnvBuilder.create", fake_create)
    monkeypatch.setattr("server.runtime_env.subprocess.run", fake_run)
    monkeypatch.setattr("server.runtime_env.python_has_plugin_sdks", fake_has_sdks)

    boot = ensure_sdk_venv(cache_root, env={"CC_AGY_VENV_DIR": str(venv_dir)})
    assert boot["ok"] is True
    pkg_install_cmds = [c for c in seen_pip_cmds if f"{cache_root}[antigravity]" in c]
    assert len(pkg_install_cmds) == 1
    assert "-e" not in pkg_install_cmds[0]

    # Transient network error echoing '--config' in install_claude does NOT trigger configure fallback
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/claude" if name == "claude" else None)
    transient_calls: list[list[str]] = []

    def fake_transient_claude(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        transient_calls.append(list(cmd))
        if "install" in cmd:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="network timeout while running: claude plugin install --config auth_mode=auto"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("server.install.subprocess.run", fake_transient_claude)
    failed_res = install_claude(dry_run=False, auth_mode="gemini-enterprise")
    assert failed_res["status"] == "failed"
    assert not any("configure" in c for c in transient_calls)


def test_round2_gateway_auth_mode_codex_location_and_proxy_hardening(tmp_path: Path) -> None:
    """Regression for Opus R2 #1-#4: auth_mode='gateway', Codex CC_AGY_ENTERPRISE_LOCATION='us-central1',
    HTTP 400 on negative/malformed Content-Length, and multi-seat US/EU isolation via X-CC-AGY-Enterprise-Location."""
    # 1. CC_AGY_AUTH_MODE='gateway' succeeds with CC_AGY_VERTEX_BASE_URL and fails closed without it
    gw_ok = resolve_auth_config(
        env={
            "CC_AGY_AUTH_MODE": "gateway",
            "CC_AGY_VERTEX_BASE_URL": "http://127.0.0.1:19999",
            "GEMINI_API_KEY": "ambient-studio-key",
        }
    )
    assert gw_ok["auth_mode"] == "vertex"
    assert gw_ok["project"] == "custom-endpoint"

    with pytest.raises(ValueError, match="Missing gateway URL"):
        resolve_auth_config(env={"CC_AGY_AUTH_MODE": "gateway", "GOOGLE_CLOUD_PROJECT": "accidental-gcp-proj"})

    codex_gw = install_codex(
        name="agy",
        profile="balanced",
        auth_mode="gateway",
        project=None,
        location=None,
        base_url="http://127.0.0.1:19999",
        max_daily_cents=None,
        dry_run=False,
        codex_home=tmp_path / "codex_gw",
    )
    assert codex_gw["status"] == "installed"
    gw_toml = (tmp_path / "codex_gw" / "config.toml").read_text(encoding="utf-8")
    assert 'CC_AGY_AUTH_MODE = "gateway"' in gw_toml
    assert build_claude_user_config(auth_mode="gateway", base_url="http://127.0.0.1:19999")["auth_mode"] == "gateway"

    # 2. Codex CC_AGY_ENTERPRISE_LOCATION='us-central1' is treated as a soft jurisdiction hint, not an impossible BAIC filter
    codex_loc_auth = resolve_auth_config(
        env={"CC_AGY_AUTH_MODE": "gemini_enterprise", "CC_AGY_ENTERPRISE_LOCATION": "us-central1"}
    )
    assert codex_loc_auth["location"] == "us-central1"
    assert codex_loc_auth["location_explicit"] is False

    baic = _LocalBAICServer()
    env = {
        "CC_AGY_BAIC_BASE_URL": baic.base_url,
        "CC_AGY_ENTERPRISE_ACCESS_TOKEN": "mock-oauth-r2-proxy-tok",
    }
    proxy = EnterpriseSubscriptionProxy(env=env)
    try:
        baic.licenses_payload = {
            "licenses": [{"userTier": "gcp-enterprise-tier", "projectId": "only-eu-seat", "location": "eu"}]
        }
        resolved_ent = resolve_enterprise_entitlement(
            "mock-oauth-r2-proxy-tok",
            project=codex_loc_auth["project"],
            location=codex_loc_auth["location"],
            location_explicit=codex_loc_auth["location_explicit"],
            env=env,
        )
        assert resolved_ent.project_id == "only-eu-seat"
        assert resolved_ent.location == "eu"

        # 3. Negative or non-numeric Content-Length is rejected with HTTP 400 before touching upstream
        import http.client

        before_reqs = len(baic.requests)
        parsed = urllib.parse.urlsplit(proxy.base_url)
        for bad_cl in ("-5", "abc"):
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
            try:
                conn.putrequest("POST", "/v1beta/models/gemini-3.8-flash:generateContent")
                conn.putheader(PROXY_AUTH_HEADER, proxy.secret)
                conn.putheader("Content-Type", "application/json")
                conn.putheader("Content-Length", bad_cl)
                conn.endheaders()
                r400 = conn.getresponse()
                assert r400.status == 400
            finally:
                conn.close()
        assert len(baic.requests) == before_reqs

        # 4. Multi-seat project ('proj-a' in both 'us' and 'eu'): X-CC-AGY-Enterprise-Location keeps EU job on EU host
        #    even when SDK rewrites global-only model URL path to 'locations/global'.
        proxy.register_entitlement(
            EnterpriseEntitlement(project_id="proj-a", location="us", user_tier="gcp-enterprise-tier")
        )
        proxy.register_entitlement(
            EnterpriseEntitlement(project_id="proj-a", location="eu", user_tier="gcp-enterprise-plus-tier")
        )
        eu_argon_req = urllib.request.Request(
            f"{proxy.base_url}/v1beta1/projects/proj-a/locations/global/publishers/google/models/gemini-4-argon:generateContent",
            data=b'{"contents":[{"role":"user","parts":[{"text":"eu residency check"}]}]}',
            headers={
                "Content-Type": "application/json",
                PROXY_AUTH_HEADER: proxy.secret,
                PROXY_LOCATION_HEADER: "eu",
            },
            method="POST",
        )
        with urllib.request.urlopen(eu_argon_req, timeout=10) as resp:
            assert resp.status == 200
        assert baic.requests[-1]["path"] == "/v1beta/projects/proj-a/locations/eu:generateContent"
        assert baic.requests[-1]["json"]["entitlement"] == {"user_tier": "gcp-enterprise-plus-tier"}
    finally:
        proxy.close()
        baic.close()
