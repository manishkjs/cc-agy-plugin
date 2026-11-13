"""Hermetic unit tests for the 5 gemini-sidecar integration enhancements in cc-agy-plugin."""

from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts.pre_tool_guard import evaluate_pre_tool_use
from scripts.user_prompt_submit import evaluate_user_prompt_submit, record_user_prompt
from server.gateway import (
    BoundedLRU,
    GatewayRequestError,
    KNOWN_GEMINI_MODELS,
    ReusableGatewayHTTPServer,
    GatewayHandler,
    classify_and_route,
    gemini_response_to_anthropic,
    load_or_create_token,
    resolve_gemini_model,
    translate_anthropic_to_gemini_payload,
)
from server.genai_ask import (
    DEPS_BATCH_SIZE,
    MEDIA_TYPES,
    _git_dir_trusted,
    _git_filter_overrides,
    _html_to_text,
    _public_host,
    ask_gemini,
    call_gemini_long_decode,
    check_dependencies_batched,
    collect_context_files,
    collect_media_files,
    estimate_source_tokens,
    extract_workspace_media_paths,
    fair_order,
    fetch_spec_url,
    file_priority,
    find_manifests,
    is_secret_file,
    parse_declared_dependencies,
    redact_secrets,
)
from server.install import install_cursor
from server.jobs import JobManager, render_worker_prompt
from server.mcp_stdio import get_tool_definitions
from server.receipts import ReceiptStore
from server.regions import (
    GLOBAL_ONLY_MODELS,
    LONG_DECODE_MODELS,
    MODEL_PRICING_PER_1M_USD,
    estimate_cost_usd,
    estimate_tokens,
    is_long_decode_model,
    normalize_wire_model_name,
    resolve_ask_thinking_level,
)


def _git_commit_no_hooks(repo: Path, msg: str = "init") -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "--allow-empty",
            "-m",
            msg,
        ],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# 1. gemini-4-carbon + Sequential continuationToken Long-Decode
# ---------------------------------------------------------------------------


def test_carbon_model_routing_and_pricing() -> None:
    assert "gemini-4-carbon" in GLOBAL_ONLY_MODELS
    assert "carbon" in GLOBAL_ONLY_MODELS
    assert "gemini-4-carbon" in LONG_DECODE_MODELS
    assert is_long_decode_model("gemini-4-carbon")
    assert is_long_decode_model("carbon")
    assert not is_long_decode_model("gemini-3.8-flash")
    assert normalize_wire_model_name("carbon", custom_gateway=False) == "gemini-4-carbon"
    assert resolve_ask_thinking_level(None, "gemini-4-carbon", env={}) == "MEDIUM"
    assert resolve_ask_thinking_level(None, "gemini-4-argon", env={}) is None
    assert (
        resolve_ask_thinking_level(
            None, "gemini-4-carbon", env={"CC_AGY_PRO_THINKING_LEVEL": "HIGH"}
        )
        == "HIGH"
    )
    assert "gemini-4-carbon" in MODEL_PRICING_PER_1M_USD
    cost = estimate_cost_usd(
        "gemini-4-carbon",
        {
            "prompt_token_count": 100_000,
            "cached_content_token_count": 50_000,
            "candidates_token_count": 10_000,
            "thoughts_token_count": 0,
        },
        single_request=True,
    )
    assert cost["total_cost_usd"] > 0


def test_call_gemini_long_decode_multi_segment_continuation() -> None:
    captured_bodies: list[dict[str, Any]] = []

    def fake_sse_post(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
        captured_bodies.append(json.loads(json.dumps(body)))
        if len(captured_bodies) == 1:
            assert "continuationToken" not in body
            return [
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "Segment 1 "}]},
                            "finishReason": "CONTINUATION",
                        }
                    ],
                    "continuationToken": "tok-seg-2",
                    "usageMetadata": {
                        "promptTokenCount": 500,
                        "cachedContentTokenCount": 100,
                        "candidatesTokenCount": 32000,
                        "thoughtsTokenCount": 1000,
                        "totalTokenCount": 33500,
                    },
                }
            ]
        assert body.get("continuationToken") == "tok-seg-2"
        return [
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Segment 2 complete."}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 500,
                    "cachedContentTokenCount": 400,
                    "candidatesTokenCount": 18000,
                    "thoughtsTokenCount": 200,
                    "totalTokenCount": 18700,
                },
            }
        ]

    res = call_gemini_long_decode(
        endpoint_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-4-carbon:streamGenerateContent?alt=sse",
        headers={"Content-Type": "application/json"},
        request_body={
            "contents": [{"role": "user", "parts": [{"text": "Generate full module"}]}],
            "generationConfig": {"maxOutputTokens": 65536},
        },
        _sse_poster=fake_sse_post,
    )
    assert res["text"] == "Segment 1 Segment 2 complete."
    assert res["continuations"] == 1
    assert res["finish_reason"] == "STOP"
    assert res["usage"]["candidates_token_count"] == 50000
    assert len(res["usage"]["requests"]) == 2
    assert all(r["per_call"] is True for r in res["usage"]["requests"])


def test_ask_gemini_carbon_max_tokens_empty_visible_retry(tmp_path: Path) -> None:
    store = ReceiptStore(tmp_path / "data", workspace_root=tmp_path)
    calls: list[int] = []

    def fake_backend(req: dict[str, Any]) -> dict[str, Any]:
        calls.append(int(req.get("max_output_tokens", 0)))
        if len(calls) == 1:
            return {
                "text": "",
                "finish_reason": "MAX_TOKENS",
                "continuation_token": None,
                "sources": [],
                "search_queries": [],
                "usage": {
                    "prompt_token_count": 100,
                    "cached_content_token_count": 0,
                    "candidates_token_count": 0,
                    "thoughts_token_count": 8192,
                    "total_token_count": 8292,
                },
            }
        return {
            "text": "Recovered answer after doubling budget.",
            "finish_reason": "STOP",
            "continuation_token": None,
            "sources": [],
            "search_queries": [],
            "usage": {
                "prompt_token_count": 100,
                "cached_content_token_count": 80,
                "candidates_token_count": 400,
                "thoughts_token_count": 1000,
                "total_token_count": 1500,
            },
        }

    out = ask_gemini(
        "Synthesize architecture",
        model="gemini-4-carbon",
        api_key="test-api-key-value",
        max_output_tokens=8192,
        store=store,
        workspace_root=tmp_path,
        backend=fake_backend,
    )
    assert out["status"] == "ok"
    assert out["model"] == "gemini-4-carbon"
    assert out["thinking_level"] == "MEDIUM"
    assert "Recovered answer" in out["text"]
    assert calls == [8192, 16384]


# ---------------------------------------------------------------------------
# 2. Multimodal UI / Spec Debugging + Immediate Turn-1 Delegation
# ---------------------------------------------------------------------------


def test_collect_media_files_and_secret_blocking(tmp_path: Path) -> None:
    png = tmp_path / "ui_bug.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    mp4 = tmp_path / "demo.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42fake")

    parts, attached = collect_media_files(tmp_path, ["ui_bug.png", "demo.mp4"])
    assert attached == ["ui_bug.png", "demo.mp4"]
    assert parts[0]["inlineData"]["mimeType"] == "image/png"
    assert parts[1]["inlineData"]["mimeType"] == "video/mp4"
    assert set(MEDIA_TYPES) >= {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".mp4", ".mov", ".webm", ".mpeg", ".pdf"}

    extracted = extract_workspace_media_paths(
        tmp_path, "Please inspect ui_bug.png and demo.mp4 for layout shift"
    )
    assert extracted == ["ui_bug.png", "demo.mp4"]


def test_user_prompt_submit_multimodal_hook_and_return_path(tmp_path: Path) -> None:
    (tmp_path / "recording.mp4").write_bytes(b"mp4-bytes")
    payload = {
        "cwd": str(tmp_path),
        "session_id": "sess-mm-1",
        "prompt": "Fix the button overlap shown in recording.mp4",
    }
    env = {"CC_AGY_DATA_DIR": str(tmp_path / ".cc-agy-data")}
    saved = record_user_prompt(payload, env=env)
    assert isinstance(saved, Path) and saved.is_file()

    hook_out = evaluate_user_prompt_submit(payload, env=env)
    assert hook_out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    ctx = hook_out["hookSpecificOutput"]["additionalContext"]
    assert "recording.mp4" in ctx
    assert "delegate_start" in ctx
    assert "ask_gemini" in ctx


def test_worktree_copies_untracked_media_and_excludes_from_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=str(repo), check=True, capture_output=True)
    _git_commit_no_hooks(repo, "initial")

    # Untracked binary media file in workspace
    (repo / "screen.png").write_bytes(b"\x89PNG\r\n\x1a\nuntracked-image")

    store = ReceiptStore(tmp_path / "data", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store)
    wt_dir, baseline_sha, _ = jm._prepare_worktree("job-1234567890-abcdef12", media_files=["screen.png"])
    try:
        assert (wt_dir / "screen.png").is_file()
        # Simulate worker editing code while leaving untracked media in worktree
        (wt_dir / "app.py").write_text("x = 2\n", encoding="utf-8")
        diff_text, files_changed, escapes = jm._collect_worktree_diff(wt_dir, baseline_sha)
        assert escapes == []
        assert files_changed == ["app.py"]
        assert "screen.png" not in diff_text
    finally:
        jm.shutdown()

    wp = render_worker_prompt(
        "Fix UI",
        plan=None,
        acceptance=None,
        max_tool_calls=60,
        max_total_tokens=200_000,
        media_files=["screen.png"],
    )
    assert "Attached Multimodal Media Files" in wp
    assert "`screen.png`" in wp


# ---------------------------------------------------------------------------
# 3. Public Spec Checking + Live Google Search Dependency/CVE Grounding
# ---------------------------------------------------------------------------


def test_ssrf_public_host_and_html_extractor() -> None:
    assert _public_host("localhost") is False
    assert _public_host("127.0.0.1") is False
    assert _public_host("169.254.169.254") is False
    with pytest.raises(ValueError, match="non-public"):
        fetch_spec_url("https://localhost/spec.html")
    with pytest.raises(ValueError, match="non-public"):
        fetch_spec_url("https://127.0.0.1/spec.html")
    with pytest.raises(ValueError, match="https://"):
        fetch_spec_url("http://example.com/spec.html")

    html = "<html><head><style>body{color:red}</style><script>alert(1)</script></head><body><h1>RFC 9110</h1><p>Section 1: Semantics.</p></body></html>"
    extracted = _html_to_text(html)
    assert "RFC 9110" in extracted
    assert "Section 1: Semantics." in extracted
    assert "alert(1)" not in extracted


def test_parse_declared_dependencies_and_batched_cve_check(tmp_path: Path) -> None:
    req_text = "\n".join(
        [
            "requests==2.31.0",
            "git+https://user:ghp_secretTokenValue123456@github.com/org/private-pkg.git#egg=private_pkg",
            *(f"pkg-{i}==1.0.{i}" for i in range(DEPS_BATCH_SIZE + 2)),
        ]
    )
    (tmp_path / "requirements.txt").write_text(req_text + "\n", encoding="utf-8")
    (tmp_path / "package.json").write_text(
        json.dumps({"dependencies": {"express": "^4.18.2"}, "devDependencies": {"vitest": "~1.6.0"}}),
        encoding="utf-8",
    )
    manifests = find_manifests(tmp_path)
    assert set(manifests) == {"package.json", "requirements.txt"}
    req_deps = parse_declared_dependencies("requirements.txt", req_text)
    pkg_deps = parse_declared_dependencies(
        "package.json", (tmp_path / "package.json").read_text(encoding="utf-8")
    )
    assert any("requests==2.31.0" in d for d in req_deps)
    assert any("npm:express @ ^4.18.2" in d for d in pkg_deps)
    assert all("ghp_secretTokenValue123456" not in d for d in req_deps)

    call_sizes: list[int] = []

    def fake_search_caller(batch_prompt: str, batch_deps: list[str]) -> dict[str, Any]:
        count = len(batch_deps)
        call_sizes.append(count)
        if count >= DEPS_BATCH_SIZE:
            raise RuntimeError("400 INVALID_ARGUMENT: TOO_MANY_TOOL_CALLS in grounded search")
        return {
            "text": f"| Package | Declared | Latest | CVEs | Action |\n|---|---|---|---|---|\n| batch-{count} | 1.0 | 1.1 | None | ok |",
            "sources": [{"title": "OSV", "uri": "https://osv.dev"}],
            "search_queries": [f"pypi batch {count}"],
            "usage": {
                "prompt_token_count": 50,
                "cached_content_token_count": 0,
                "candidates_token_count": 30,
                "thoughts_token_count": 10,
                "total_token_count": 90,
            },
        }

    merged = check_dependencies_batched(tmp_path, manifests, "Audit deps", fake_search_caller)
    assert merged is not None
    assert 12 in call_sizes
    assert 6 in call_sizes
    assert merged["sources"] == [{"title": "OSV", "uri": "https://osv.dev"}]
    assert merged["usage"]["total_token_count"] > 0


# ---------------------------------------------------------------------------
# 4. Outbound Secret Redaction, Hostile .git Hardening, Fair File Budgeting & Read Guard
# ---------------------------------------------------------------------------


def test_secret_redaction_and_file_blocking(tmp_path: Path) -> None:
    assert is_secret_file(".env")
    assert is_secret_file("config/.env.production")
    assert is_secret_file("keys/id_rsa")
    assert not is_secret_file(".env.example")
    assert not is_secret_file("src/main.py")

    raw = (
        "api_key = 'AIza" + "SyA1234567890abcdefghijklmnopqrstuvwx'\n"
        "token = 'ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789'\n"
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----\n"
    )
    redacted, count = redact_secrets(raw)
    assert count >= 3
    assert "AIzaSy" not in redacted
    assert "ghp_" not in redacted
    assert "MIIEpAIBAAKCAQEA" not in redacted


def test_hostile_git_pointer_and_fair_context_budgeting(tmp_path: Path) -> None:
    # Untrusted .git file pointing outside workspace
    bad_ws = tmp_path / "bad_ws"
    bad_ws.mkdir()
    outside_git = tmp_path / "outside_git"
    outside_git.mkdir()
    (bad_ws / ".git").write_text(f"gitdir: {outside_git}\n", encoding="utf-8")
    assert not _git_dir_trusted(bad_ws)

    # Directory expansion + fair ordering across packages
    ws = tmp_path / "ws"
    (ws / "pkg_a").mkdir(parents=True)
    (ws / "pkg_b").mkdir(parents=True)
    (ws / "pkg_a" / "a1.py").write_text("def a1():\n    return 1\n", encoding="utf-8")
    (ws / "pkg_a" / "a2.py").write_text("def a2():\n    return 2\n", encoding="utf-8")
    (ws / "pkg_b" / "b1.py").write_text("def b1():\n    return 3\n", encoding="utf-8")
    (ws / "pkg_a" / ".env").write_text("SECRET=1234567890\n", encoding="utf-8")

    ordered = fair_order(["pkg_a/a1.py", "pkg_a/a2.py", "pkg_b/b1.py"])
    assert ordered[:2] == ["pkg_a/a1.py", "pkg_b/b1.py"]
    assert file_priority("src/core.py") < file_priority("docs/readme.md")
    assert estimate_source_tokens("{\"a\": [1, 2, 3]}") >= 1

    formatted, loaded_rels, skipped, redactions = collect_context_files(
        ws, ["pkg_a", "pkg_b"], token_budget=5000
    )
    assert "pkg_a/a1.py" in loaded_rels
    assert "pkg_b/b1.py" in loaded_rels
    assert "def a1():" in formatted
    assert any(".env" in s and "secret" in s for s in skipped)
    assert redactions == 0


def test_pre_tool_guard_video_read_and_offset_aware_line_guard(tmp_path: Path) -> None:
    vid_res = evaluate_pre_tool_use(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": "demo.mp4"},
            "cwd": str(tmp_path),
        },
        env={"CC_AGY_STRICT_MODE": "0"},
    )
    assert vid_res["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "ask_gemini" in vid_res["hookSpecificOutput"]["permissionDecisionReason"]

    big_file = tmp_path / "big.py"
    big_file.write_text("".join(f"x_{i} = {i}\n" for i in range(450)), encoding="utf-8")

    # Whole-file Read blocked when CC_AGY_READ_GUARD=block
    blk = evaluate_pre_tool_use(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": str(big_file)},
            "cwd": str(tmp_path),
        },
        env={"CC_AGY_STRICT_MODE": "0", "CC_AGY_READ_GUARD": "block", "CC_AGY_READ_GUARD_LINES": "400"},
    )
    assert blk["hookSpecificOutput"]["permissionDecision"] == "deny"

    # Scoped Read with offset/limit is allowed
    scoped = evaluate_pre_tool_use(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": str(big_file), "offset": 1, "limit": 80},
            "cwd": str(tmp_path),
        },
        env={"CC_AGY_STRICT_MODE": "0", "CC_AGY_READ_GUARD": "block", "CC_AGY_READ_GUARD_LINES": "400"},
    )
    assert scoped == {}


# ---------------------------------------------------------------------------
# 5. Cursor Coexistence Rule + Local Anthropic Gateway + Strict Schema Budget
# ---------------------------------------------------------------------------


def test_install_cursor_writes_coexistence_rule_and_refuses_symlinks(tmp_path: Path) -> None:
    cursor_dir = tmp_path / ".cursor"
    res = install_cursor(cursor_config_dir=cursor_dir)
    assert res["status"] == "installed"
    rule_path = Path(res["rule_path"])
    assert rule_path.is_file()
    assert "cc-agy-coexistence.mdc" in rule_path.name

    # Refuse symlinked mcp.json
    symlink_dir = tmp_path / ".cursor_sym"
    symlink_dir.mkdir()
    target_cfg = tmp_path / "outside.json"
    target_cfg.write_text("{}", encoding="utf-8")
    (symlink_dir / "mcp.json").symlink_to(target_cfg)
    refused = install_cursor(cursor_config_dir=symlink_dir)
    assert refused["status"] == "refused"
    assert "symlink" in refused["error"]


def test_gateway_excludes_flash_lite_and_replays_thought_signatures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert "gemini-3.5-flash-lite" not in KNOWN_GEMINI_MODELS
    with pytest.raises(GatewayRequestError):
        resolve_gemini_model("gemini-3.5-flash-lite")

    # Haiku background calls route to gemini-3.8-flash, never flash-lite
    route = classify_and_route(
        session_id="s-1",
        requested_model="claude-3-5-haiku-latest",
        messages=[{"role": "user", "content": "hi"}],
        token_count_est=10,
    )
    assert route["selected_model"] == "gemini/gemini-3.8-flash"

    # Auto router routes complex/long-decode to gemini-4-carbon and simple to gemini-3.8-flash
    auto_simple = classify_and_route(
        session_id="s-2",
        requested_model="claude-auto-gemini",
        messages=[{"role": "user", "content": "hi"}],
        token_count_est=10,
    )
    assert auto_simple["selected_model"] == "gemini/gemini-3.8-flash"

    auto_complex = classify_and_route(
        session_id="s-3",
        requested_model="claude-auto-gemini",
        messages=[{"role": "user", "content": "Please migrate all modules and generate complete test suite"}],
        token_count_est=100,
    )
    assert auto_complex["selected_model"] == "gemini/gemini-4-carbon"

    # ThoughtSignature vault replay across tool_use -> tool_result turns
    gem_resp = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {
                            "functionCall": {"name": "read_file", "args": {"path": "a.py"}},
                            "thoughtSignature": "sig-xyz-987",
                        }
                    ]
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10},
    }
    anth_msg = gemini_response_to_anthropic(gem_resp, "claude-gemini-4-carbon")
    tool_block = anth_msg["content"][0]
    assert tool_block["type"] == "tool_use"
    tool_id = tool_block["id"]

    follow_up_body = {
        "model": "gemini-4-carbon",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": "Read a.py"},
            {"role": "assistant", "content": [tool_block]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_id, "content": "print(1)"},
                    {"type": "text", "text": "System reminder: stay concise."},
                ],
            },
        ],
    }
    translated = translate_anthropic_to_gemini_payload(follow_up_body, "gemini-4-carbon")
    model_turn = next(c for c in translated["contents"] if c["role"] == "model")
    assert model_turn["parts"][0]["thoughtSignature"] == "sig-xyz-987"
    assert translated["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "MEDIUM"

    # Token file creation with 0600 permissions
    tok_file = tmp_path / "gw-token"
    monkeypatch.setenv("CC_AGY_GATEWAY_TOKEN_FILE", str(tok_file))
    monkeypatch.delenv("CC_AGY_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI4_GATEWAY_TOKEN", raising=False)
    token, path, created = load_or_create_token()
    assert created is True
    assert path == str(tok_file)
    assert len(token) >= 32
    assert (tok_file.stat().st_mode & 0o777) == 0o600


def test_strict_mode_core_schema_stays_within_token_budget() -> None:
    strict_defs = get_tool_definitions(env={"CC_AGY_STRICT_MODE": "true"})
    full_defs = get_tool_definitions(env={"CC_AGY_STRICT_MODE": "false"})
    strict_tokens = estimate_tokens(json.dumps(strict_defs))
    full_tokens = estimate_tokens(json.dumps(full_defs))
    assert strict_tokens < 1250
    assert strict_tokens < full_tokens * 0.5
