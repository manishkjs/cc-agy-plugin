"""Standalone Antigravity CLI (`agy`) runtime adapter for Gemini Enterprise seats.

Cloud Code PA rejects `useragent.SDK` requests on the Business OAuth path
(`The Antigravity SDK is not available on an enterprise account`), while
explicitly permitting `useragent.CLI` (`Antigravity CLI` / `agy`). This module
discovers or bootstraps the external `agy` binary (skipping Google-internal
`jetski` builds), inspects local OAuth readiness without echoing tokens, maps
`agy` headless JSON / `stream-json` envelopes and token usage to Vertex/GenAI
accounting without double-counting thinking or cached tokens, and runs
cancelled-cleanly subprocesses for workers, single-shot Q&A, and batch writers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence

DEFAULT_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
MANAGED_CLI_BIN_NAME = "agy.exe" if os.name == "nt" else "agy"

_USAGE_KEYS = (
    "prompt_token_count",
    "cached_content_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)

_SCRUBBED_SUBPROCESS_ENV_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_VERTEX_BASE_URL",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "VERTEX_PROJECT",
        "CLOUD_ML_PROJECT_ID",
        "GOOGLE_CLOUD_QUOTA_PROJECT",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_OPENAI_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
    }
)

_EFFORT_SUFFIXES = frozenset({"low", "medium", "high", "xhigh", "max"})

_THINKING_TO_EFFORT: dict[str, str] = {
    "ZERO": "low",
    "MINIMAL": "low",
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high",
    "EXTRA_HIGH": "xhigh",
    "XHIGH": "xhigh",
    "MAX": "max",
}

_BINARY_PROBE_LOCK = threading.Lock()
_BINARY_PROBE_CACHE: dict[tuple[str, int, int], tuple[bool, str]] = {}

_MODEL_FALLBACK_LOCK = threading.Lock()
_MODEL_FALLBACK_CACHE: dict[tuple[str, str, str, str], str] = {}

_CONV_LEDGER_LOCK = threading.Lock()
_CONVERSATION_USAGE_LEDGER: dict[tuple[str, str], tuple[int, dict[str, int]]] = {}

_QUOTA_OR_TRANSIENT_PATTERNS = (
    "429",
    "resource_exhausted",
    "quota exhausted",
    "quota exceeded",
    "rate limit",
    "too many requests",
    "overloaded",
    "503",
    "service unavailable",
)

_AUTH_ERROR_PATTERNS = (
    "not logged in",
    "not authenticated",
    "unauthenticated",
    "authentication required",
    "authentication failed",
    "login required",
    "please sign in",
    "sign in with google",
    "invalid oauth",
    "expired oauth",
    "missing oauth",
    "invalid_grant",
    "token has been expired or revoked",
    "expired or revoked",
    "credentials missing",
    "no stored credentials",
    "reauthentication required",
    "missing required authentication credential",
    "antigravity sdk is not available on an enterprise account",
    "unauthorized",
)

_AUTH_CONTEXT_HTTP_RE = re.compile(
    r"(?:googleapis\.com|cloudcode|antigravity|oauth|unauthenticated)"
    r"[^\n]{0,120}\b(?:401|403|unauthorized)\b"
    r"|\b(?:401|403|unauthorized)\b[^\n]{0,120}"
    r"(?:googleapis\.com|cloudcode|antigravity|oauth|sign[- ]in)"
    r"|(?:http(?:/\d(?:\.\d)?)?|status(?:\s+code)?|error(?:\s+code)?)\s*[:=]?\s*(?:401|403)\b"
    r"|\b(?:401\s+unauthorized|403\s+forbidden)\b",
    re.IGNORECASE,
)


def _clean_env_str(env: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        val = env.get(key)
        if val is None:
            continue
        cleaned = str(val).strip()
        if cleaned and not cleaned.startswith("${"):
            return cleaned
    return ""


def _env_flag(name: str, env: Mapping[str, str]) -> bool | None:
    raw = _clean_env_str(env, name).lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def _home_dir(env: Mapping[str, str]) -> Path:
    explicit_home = _clean_env_str(env, "HOME", "USERPROFILE")
    if explicit_home:
        return Path(explicit_home).expanduser()
    return Path.home()


def managed_agy_bin_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the per-user managed directory where `install_agy_cli.sh` places `agy`."""
    source = env if env is not None else os.environ
    explicit = _clean_env_str(source, "CC_AGY_CLI_INSTALL_DIR")
    if explicit:
        return Path(explicit).expanduser()
    home = _home_dir(source)
    if os.name == "nt":
        local_app = _clean_env_str(source, "LOCALAPPDATA")
        base = Path(local_app).expanduser() if local_app else (home / "AppData" / "Local")
        return base / "cc-agy-plugin" / "bin"
    xdg_data = _clean_env_str(source, "XDG_DATA_HOME")
    base = Path(xdg_data).expanduser() if xdg_data else (home / ".local" / "share")
    return base / "cc-agy-plugin" / "bin"


def agy_cli_config_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the active `agy` CLI state directory (`~/.gemini/antigravity-cli`)."""
    source = env if env is not None else os.environ
    explicit = _clean_env_str(source, "CC_AGY_CLI_CONFIG_DIR")
    if explicit:
        return Path(explicit).expanduser()
    home = _home_dir(source)
    default_dir = home / ".gemini" / "antigravity-cli"
    if os.name == "nt" and not default_dir.exists():
        for win_var in ("APPDATA", "LOCALAPPDATA"):
            win_base = _clean_env_str(source, win_var)
            if win_base:
                cand = Path(win_base).expanduser() / "antigravity-cli"
                if cand.exists():
                    return cand
    return default_dir


def is_external_agy_binary(
    path: str | Path,
    env: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Verify `path` is a runnable external `agy` CLI and not an internal `jetski` build."""
    cand = Path(path).expanduser()
    if not (cand.is_file() and os.access(cand, os.X_OK)):
        return False, "not_executable"
    try:
        resolved = cand.resolve()
        st = resolved.stat()
    except OSError as exc:
        return False, f"stat_error:{exc}"

    cache_key = (str(resolved), st.st_mtime_ns, st.st_size)
    with _BINARY_PROBE_LOCK:
        cached = _BINARY_PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    probe_env = build_agy_subprocess_env(env)
    try:
        proc = subprocess.run(
            [str(cand), "--help"],
            cwd=tempfile.gettempdir(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            env=probe_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        verdict = (False, f"probe_failed:{exc}")
        with _BINARY_PROBE_LOCK:
            _BINARY_PROBE_CACHE[cache_key] = verdict
        return verdict

    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    if "--is_google_internal" in combined or "--use_stubby_auth" in combined:
        verdict = (False, "internal_jetski_binary")
    elif proc.returncode != 0:
        verdict = (False, f"exit_{proc.returncode}")
    elif "--output-format" not in combined:
        verdict = (False, "missing_output_format_flag")
    else:
        verdict = (True, "ok")

    with _BINARY_PROBE_LOCK:
        _BINARY_PROBE_CACHE[cache_key] = verdict
    return verdict


def _explicit_cli_bin_override(source: Mapping[str, str]) -> str:
    return _clean_env_str(
        source,
        "CC_AGY_CLI_BIN",
        "CLAUDE_PLUGIN_OPTION_AGY_CLI_PATH",
        "CC_AGY_USERCFG_AGY_CLI_PATH",
    )


def resolve_agy_cli_binary(
    env: Mapping[str, str] | None = None,
) -> tuple[str | None, list[str]]:
    """Locate the external `agy` binary, returning `(valid_path, skipped_internal_paths)`."""
    source = env if env is not None else os.environ
    skipped_internal: list[str] = []
    explicit = _explicit_cli_bin_override(source)

    if explicit:
        ok, reason = is_external_agy_binary(explicit, env=source)
        if ok:
            return str(Path(explicit).expanduser()), skipped_internal
        if reason == "internal_jetski_binary":
            skipped_internal.append(explicit)
        return None, skipped_internal

    candidates: list[Path] = []
    candidates.append(managed_agy_bin_dir(source) / MANAGED_CLI_BIN_NAME)

    path_var = source.get("PATH") if env is not None else os.environ.get("PATH")
    if path_var:
        which_hit = shutil.which("agy", path=path_var)
        if which_hit:
            candidates.append(Path(which_hit))

    home = _home_dir(source)
    candidates.append(home / ".local" / "bin" / MANAGED_CLI_BIN_NAME)

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        ok, reason = is_external_agy_binary(cand, env=source)
        if ok:
            return str(cand), skipped_internal
        if reason == "internal_jetski_binary":
            skipped_internal.append(str(cand))

    return None, skipped_internal


def _read_onboarding_metadata(config_dir: Path) -> tuple[bool, str | None, str | None]:
    """Read non-secret onboarding flags, `cloudaicompanionProject`, and `tierId` from `cache/onboarding.json`."""
    onboarding_file = config_dir / "cache" / "onboarding.json"
    try:
        if onboarding_file.is_file() and onboarding_file.stat().st_size > 2:
            data = json.loads(onboarding_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                raw_proj = data.get("cloudaicompanionProject")
                comp_proj = str(raw_proj).strip() if isinstance(raw_proj, str) and str(raw_proj).strip() else None
                raw_tier = data.get("tierId")
                tier_id = str(raw_tier).strip() if isinstance(raw_tier, str) and str(raw_tier).strip() else None
                complete = any(
                    bool(data.get(flag_key))
                    for flag_key in (
                        "onboardingComplete",
                        "enterpriseOnboardingComplete",
                        "consumerOnboardingComplete",
                    )
                ) or bool(tier_id or comp_proj)
                return complete, comp_proj, tier_id
    except (OSError, ValueError):
        pass
    return False, None, None


def _inspect_auth_files(config_dir: Path, source: Mapping[str, str]) -> tuple[bool, str | None]:
    env_tok = _clean_env_str(source, "ANTIGRAVITY_OAUTH_TOKEN")
    if len(env_tok) >= 8:
        return True, "env:ANTIGRAVITY_OAUTH_TOKEN"

    token_file = config_dir / "antigravity-oauth-token"
    try:
        if token_file.is_file() and token_file.stat().st_size > 10:
            raw_head = token_file.read_bytes()[:256].strip()
            if raw_head:
                return True, f"token_file:{token_file}"
    except OSError:
        pass

    onboarding_ok, _comp_proj, _tier_id = _read_onboarding_metadata(config_dir)
    if onboarding_ok:
        onboarding_file = config_dir / "cache" / "onboarding.json"
        return True, f"onboarding_cache:{onboarding_file}"

    return False, None


def inspect_agy_cli_offline(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Inspect external `agy` binary and sign-in state without network calls or secret echoing."""
    source = env if env is not None else os.environ
    agy_bin, skipped_internal = resolve_agy_cli_binary(source)
    cfg_dir = agy_cli_config_dir(source)
    auth_ok, cred_source = _inspect_auth_files(cfg_dir, source)
    _onboarding_ok, companion_project, tier_id = _read_onboarding_metadata(cfg_dir)
    managed_bin = managed_agy_bin_dir(source) / MANAGED_CLI_BIN_NAME

    explicit_bin = _explicit_cli_bin_override(source)
    binary_found = agy_bin is not None
    ok = binary_found and auth_ok

    error: str | None = None
    remediation: str | None = None
    if not binary_found:
        if explicit_bin and explicit_bin in skipped_internal:
            error = (
                f"Configured CC_AGY_CLI_BIN ({explicit_bin}) points to a Google-internal `jetski` binary "
                "rather than the external Antigravity CLI (`agy`)."
            )
        elif explicit_bin:
            error = f"Configured CC_AGY_CLI_BIN ({explicit_bin}) is missing or not a valid external `agy` binary."
        elif skipped_internal:
            error = (
                f"Skipped internal `jetski` binary on PATH ({skipped_internal[0]}); "
                "no external Antigravity CLI (`agy`) binary found."
            )
        else:
            error = "External Antigravity CLI (`agy`) is not installed yet."
        remediation = (
            "Run `python3 -m server.install --auth gemini-enterprise` (or `bash scripts/install_agy_cli.sh`) "
            f"to install `agy` to `{managed_bin}`, then run `{managed_bin}` once in a terminal to sign in."
        )
    elif not auth_ok:
        error = f"Antigravity CLI is installed at `{agy_bin}`, but no Google sign-in token was found in `{cfg_dir}`."
        remediation = (
            f"Open a terminal and run `{agy_bin}` once to sign in with your Gemini Enterprise Google account "
            "(then `/exit` or Ctrl+C), or set `ANTIGRAVITY_OAUTH_TOKEN`."
        )

    return {
        "ok": ok,
        "binary_found": binary_found,
        "agy_bin": agy_bin,
        "managed_bin_path": str(managed_bin),
        "auth_configured": auth_ok,
        "credential_source": cred_source,
        "companion_project": companion_project,
        "tier_id": tier_id,
        "skipped_internal_binaries": skipped_internal,
        "config_dir": str(cfg_dir),
        "error": error,
        "remediation": remediation,
    }


def ensure_agy_cli_installed(
    plugin_root: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    manifest_url: str | None = None,
    configure_shell: bool = False,
    output_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Ensure the external `agy` binary is installed in `managed_agy_bin_dir`."""
    source = env if env is not None else os.environ
    existing_bin, skipped_internal = resolve_agy_cli_binary(source)
    if existing_bin is not None:
        return {
            "ok": True,
            "bootstrapped": False,
            "agy_bin": existing_bin,
            "skipped_internal_binaries": skipped_internal,
        }

    explicit_override = _explicit_cli_bin_override(source)
    if explicit_override:
        return {
            "ok": False,
            "bootstrapped": False,
            "agy_bin": None,
            "skipped_internal_binaries": skipped_internal,
            "error": (
                f"Explicit CC_AGY_CLI_BIN ({explicit_override}) is invalid or missing; "
                "unset CC_AGY_CLI_BIN to allow automatic installation."
            ),
        }

    root = Path(plugin_root).resolve() if plugin_root is not None else DEFAULT_PLUGIN_ROOT.resolve()
    installer = root / "scripts" / "install_agy_cli.sh"
    target_dir = managed_agy_bin_dir(source)
    if not installer.is_file():
        return {
            "ok": False,
            "bootstrapped": False,
            "agy_bin": None,
            "skipped_internal_binaries": skipped_internal,
            "error": f"Installer script missing at {installer}",
        }

    if output_fn is not None:
        output_fn(f"  Installing standalone Antigravity CLI (`agy`) to {target_dir / MANAGED_CLI_BIN_NAME} ...")

    cmd = ["bash", str(installer), "--target-dir", str(target_dir), "--json"]
    effective_manifest = manifest_url or _clean_env_str(source, "CC_AGY_CLI_MANIFEST_URL")
    if effective_manifest:
        cmd.extend(["--manifest-url", effective_manifest])
    if configure_shell:
        cmd.append("--configure-shell")

    run_env = dict(os.environ)
    run_env.update({k: str(v) for k, v in source.items() if v is not None})
    try:
        proc = subprocess.run(
            cmd,
            cwd=tempfile.gettempdir(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
            env=run_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ok": False,
            "bootstrapped": False,
            "agy_bin": None,
            "skipped_internal_binaries": skipped_internal,
            "error": f"Failed to execute install_agy_cli.sh: {exc}",
        }

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-500:]
        return {
            "ok": False,
            "bootstrapped": False,
            "agy_bin": None,
            "skipped_internal_binaries": skipped_internal,
            "error": f"install_agy_cli.sh failed (exit {proc.returncode}): {detail}",
        }

    with _BINARY_PROBE_LOCK:
        _BINARY_PROBE_CACHE.clear()

    resolved_after, skipped_after = resolve_agy_cli_binary(source)
    if resolved_after is None:
        return {
            "ok": False,
            "bootstrapped": False,
            "agy_bin": None,
            "skipped_internal_binaries": skipped_after,
            "error": f"Installed binary under {target_dir} did not pass external `agy` verification.",
        }

    payload_ver: str | None = None
    for line in reversed((proc.stdout or "").splitlines()):
        line_s = line.strip()
        if line_s.startswith("{") and line_s.endswith("}"):
            try:
                parsed = json.loads(line_s)
                if isinstance(parsed, dict):
                    payload_ver = parsed.get("version")
                    break
            except ValueError:
                pass

    return {
        "ok": True,
        "bootstrapped": True,
        "agy_bin": resolved_after,
        "version": payload_ver,
        "skipped_internal_binaries": skipped_after,
    }


def _should_auto_install_on_call(env: Mapping[str, str] | None, source: Mapping[str, str]) -> bool:
    flag = _env_flag("CC_AGY_AUTO_INSTALL_CLI", source)
    if flag is True:
        return True
    if flag is False:
        return False
    if _explicit_cli_bin_override(source):
        return False
    if env is not None or "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return True


def _resolve_or_bootstrap_agy_bin(env: Mapping[str, str] | None = None) -> str:
    source = env if env is not None else os.environ
    agy_bin, skipped_internal = resolve_agy_cli_binary(source)
    if agy_bin is not None:
        return agy_bin

    if _should_auto_install_on_call(env, source):
        boot = ensure_agy_cli_installed(DEFAULT_PLUGIN_ROOT, env=source)
        if boot.get("ok") and boot.get("agy_bin"):
            return str(boot["agy_bin"])

    status = inspect_agy_cli_offline(source)
    err_msg = status.get("error") or "Antigravity CLI (`agy`) is not installed."
    rem = status.get("remediation") or "Run `python3 -m server.install --auth gemini-enterprise`."
    if skipped_internal and "internal" not in err_msg:
        err_msg = f"{err_msg} (skipped internal binary: {skipped_internal[0]})"
    raise RuntimeError(f"{err_msg} {rem}")


def convert_agy_usage_to_vertex(agy_usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Convert `agy` `JSONUsage` into Vertex/GenAI `usage_metadata` without double-counting.

    In `agy` (`steps/json_output.go`):
    - `input_tokens` = uncached prompt tokens
    - `cache_read_tokens` = cached prompt tokens
    - `output_tokens` = total generated tokens INCLUDING `thinking_tokens`
    - `thinking_tokens` = reasoning subset of `output_tokens`
    """
    if not isinstance(agy_usage, Mapping):
        return {
            "prompt_token_count": 0,
            "cached_content_token_count": 0,
            "candidates_token_count": 0,
            "thoughts_token_count": 0,
            "total_token_count": 0,
        }

    src_map: Mapping[str, Any] = agy_usage
    if isinstance(src_map.get("usage"), Mapping):
        src_map = src_map["usage"]

    def _first_nonneg(*keys: str) -> int:
        for key in keys:
            val = src_map.get(key)
            if val is None or isinstance(val, bool):
                continue
            try:
                return max(0, int(val))
            except (TypeError, ValueError):
                continue
        return 0

    if any(
        k in src_map
        for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "cached_content_token_count")
    ) and not any(k in src_map for k in ("input_tokens", "inputTokens", "output_tokens", "outputTokens")):
        p_tok = _first_nonneg("prompt_token_count", "promptTokenCount")
        c_tok = _first_nonneg("cached_content_token_count", "cachedContentTokenCount")
        o_tok = _first_nonneg("candidates_token_count", "candidatesTokenCount")
        t_tok = _first_nonneg("thoughts_token_count", "thoughtsTokenCount")
        return {
            "prompt_token_count": p_tok,
            "cached_content_token_count": c_tok,
            "candidates_token_count": o_tok,
            "thoughts_token_count": t_tok,
            "total_token_count": p_tok + o_tok + t_tok,
        }

    uncached_in = _first_nonneg("input_tokens", "inputTokens")
    cached_in = _first_nonneg("cache_read_tokens", "cacheReadTokens")
    raw_out = _first_nonneg("output_tokens", "outputTokens")
    think_out = _first_nonneg("thinking_tokens", "thinkingTokens")

    prompt_tokens = uncached_in + cached_in
    candidates_tokens = max(0, raw_out - think_out)
    thoughts_tokens = think_out
    total_tokens = prompt_tokens + candidates_tokens + thoughts_tokens
    return {
        "prompt_token_count": prompt_tokens,
        "cached_content_token_count": cached_in,
        "candidates_token_count": candidates_tokens,
        "thoughts_token_count": thoughts_tokens,
        "total_token_count": total_tokens,
    }


def _model_has_explicit_effort(model: str | None) -> bool:
    if not model:
        return False
    m = model.strip()
    lower = m.lower()
    if lower.endswith("-effort"):
        lower = lower[: -len("-effort")]
    if "-" in lower:
        suffix = lower.rsplit("-", 1)[-1]
        if suffix in _EFFORT_SUFFIXES:
            return True
    paren = re.search(r"\((low|medium|high|xhigh|max)\)\s*$", lower)
    return paren is not None


def map_thinking_level_to_effort(
    thinking_level: str | None,
    model: str | None = None,
) -> str | None:
    """Map plugin `thinking_level` (`MINIMAL`..`EXTRA_HIGH`) to `agy --effort`."""
    if _model_has_explicit_effort(model):
        return None
    if not thinking_level:
        return None
    norm = str(thinking_level).strip().upper().replace("-", "_")
    return _THINKING_TO_EFFORT.get(norm)


def build_agy_subprocess_env(
    env: Mapping[str, str] | None = None,
    *,
    workspace_dir: str | Path | None = None,
    cwd: str | Path | None = None,
) -> dict[str, str]:
    """Build a scrubbed subprocess environment for `agy` preserving OAuth, keyring, and proxies."""
    source = env if env is not None else os.environ
    out: dict[str, str] = {}
    for k, v in source.items():
        if v is None:
            continue
        if k in _SCRUBBED_SUBPROCESS_ENV_KEYS or k in ("PWD", "OLDPWD"):
            continue
        if k.startswith("CC_AGY_") or k.startswith("CLAUDE_PLUGIN_"):
            continue
        val_str = str(v)
        if k == "ANTIGRAVITY_OAUTH_TOKEN" and val_str.strip().startswith("${"):
            continue
        out[k] = val_str
    if not out.get("PATH"):
        out["PATH"] = os.environ.get("PATH") or os.defpath
    eff_ws = workspace_dir if workspace_dir is not None else cwd
    if eff_ws is not None:
        ws_path = Path(eff_ws)
        path_prefixes: list[str] = []
        nm_bin = ws_path / "node_modules" / ".bin"
        if nm_bin.is_dir():
            nm_bin_resolved = str(nm_bin.resolve())
            path_prefixes.append(nm_bin_resolved)
            if str(nm_bin) != nm_bin_resolved:
                path_prefixes.append(str(nm_bin))
        venv_bin = ws_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
        if venv_bin.is_dir():
            path_prefixes.append(str(venv_bin))
        if path_prefixes:
            cur_path = out.get("PATH") or ""
            prefix_joined = os.pathsep.join(path_prefixes)
            out["PATH"] = f"{prefix_joined}{os.pathsep}{cur_path}" if cur_path else prefix_joined
    out.pop("_AGY_SPAWN_COOKIE", None)
    out["NO_COLOR"] = "1"
    out.setdefault("TERM", "dumb")
    return out


def _custom_config_dir_args(env: Mapping[str, str] | None = None) -> list[str]:
    source = env if env is not None else os.environ
    explicit_cfg = _clean_env_str(source, "CC_AGY_CLI_CONFIG_DIR")
    if not explicit_cfg:
        return []
    cfg_path = Path(explicit_cfg).expanduser().resolve()
    return ["--gemini_dir", str(cfg_path.parent), "--app_data_dir", cfg_path.name]


def is_agy_auth_error(message: str) -> bool:
    """Return True if `message` indicates missing, expired, or unauthorized `agy` OAuth credentials."""
    lower = (message or "").lower()
    if any(qpat in lower for qpat in _QUOTA_OR_TRANSIENT_PATTERNS):
        return False
    if any(pat in lower for pat in _AUTH_ERROR_PATTERNS):
        return True
    return bool(_AUTH_CONTEXT_HTTP_RE.search(lower))


def _extract_available_models(error_text: str) -> list[str]:
    if "Available models:" not in error_text:
        return []
    after = error_text.split("Available models:", 1)[1]
    entries: list[str] = []
    for raw_line in after.splitlines():
        if not raw_line.strip():
            if entries:
                break
            continue
        if raw_line.startswith((" ", "\t", "-", "*", "•")):
            item = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", raw_line.strip()).strip("`'\"")
            if item:
                entries.append(item)
        elif entries:
            break
    return entries


def select_fallback_agy_model(
    requested_model: str,
    requested_effort: str | None,
    available_models: Sequence[str],
) -> str | None:
    """Select the closest Gemini model from `agy`'s `Available models:` catalog."""
    if not available_models:
        return None
    req_low = (requested_model or "").lower()
    want_flash = "flash" in req_low
    want_pro = ("argon" in req_low) or ("carbon" in req_low) or ("pro" in req_low) or not want_flash
    eff_low = (requested_effort or "").strip().lower()

    best_score: tuple[int, int, int, int, int] | None = None
    best_entry: str | None = None

    for entry in available_models:
        low = entry.lower()
        if "gemini" not in low:
            continue
        if "lite" in low and "lite" not in req_low:
            continue
        is_flash = "flash" in low
        is_pro_or_argon = ("argon" in low) or ("carbon" in low) or ("pro" in low)
        if want_flash and not is_flash:
            continue
        if want_pro and not want_flash and not is_pro_or_argon:
            continue

        ver_match = re.search(r"gemini[\s\-]*(\d+)(?:\.(\d+))?", low)
        major = int(ver_match.group(1)) if ver_match else 0
        minor = int(ver_match.group(2)) if (ver_match and ver_match.group(2)) else 0
        argon_bonus = 1 if (("argon" in req_low and "argon" in low) or ("carbon" in req_low and "carbon" in low)) else 0

        eff_bonus = 0
        if eff_low:
            if f"({eff_low})" in low or low.endswith(f"-{eff_low}"):
                eff_bonus = 2
        elif "(medium)" in low or low.endswith("-medium") or "(high)" in low or low.endswith("-high"):
            eff_bonus = 1

        base_clean_bonus = 0 if _model_has_explicit_effort(entry) else 1
        score = (argon_bonus, major, minor, eff_bonus, base_clean_bonus)
        if best_score is None or score > best_score:
            best_score = score
            best_entry = entry

    return best_entry


def _close_proc_pipes_safely(
    proc: subprocess.Popen[Any],
    *,
    stdout_thread: threading.Thread | None = None,
    stderr_thread: threading.Thread | None = None,
) -> None:
    """Close subprocess pipes only when no reader thread is blocked in `.read()`/`.readline()`.

    Calling `BufferedReader.close()` from another thread while a reader thread holds the buffer's
    internal C lock deadlocks Python indefinitely if an unkilled child still holds the write fd.
    """
    if proc.stdin is not None:
        try:
            proc.stdin.close()
        except Exception:
            pass
    if proc.stdout is not None and (stdout_thread is None or not stdout_thread.is_alive()):
        try:
            proc.stdout.close()
        except Exception:
            pass
    if proc.stderr is not None and (stderr_thread is None or not stderr_thread.is_alive()):
        try:
            proc.stderr.close()
        except Exception:
            pass


def _snapshot_pipe_inodes(proc: subprocess.Popen[Any]) -> set[tuple[int, int]]:
    inodes: set[tuple[int, int]] = set()
    for pipe in (proc.stdout, proc.stderr):
        if pipe is None:
            continue
        try:
            st = os.fstat(pipe.fileno())
            inodes.add((st.st_dev, st.st_ino))
        except Exception:
            pass
    return inodes


def _find_pipe_or_cookie_survivors(
    *,
    pipe_inodes: set[tuple[int, int]] | None,
    spawn_cookie: str,
    exclude_pid: int,
) -> set[int]:
    """Locate detached (`setsid`) descendants that still hold our stdout/stderr pipes or spawn cookie."""
    found: set[int] = set()
    if os.name == "nt":
        return found
    from server.agy_runner import _proc_fs_available, _ps_pid_matches_save_dir, _ps_process_table

    cookie_token = f"_AGY_SPAWN_COOKIE={spawn_cookie}" if spawn_cookie else ""
    if _proc_fs_available():
        try:
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                if pid == exclude_pid:
                    continue
                if pipe_inodes:
                    matched_pipe = False
                    for fd_num in (1, 2):
                        try:
                            st = os.stat(f"/proc/{pid}/fd/{fd_num}")
                            if (st.st_dev, st.st_ino) in pipe_inodes:
                                found.add(pid)
                                matched_pipe = True
                                break
                        except OSError:
                            continue
                    if matched_pipe:
                        continue
                if cookie_token:
                    try:
                        raw_env = (entry / "environ").read_bytes()
                        if cookie_token.encode("utf-8") in raw_env:
                            found.add(pid)
                    except OSError:
                        continue
        except OSError:
            pass
        return found

    if cookie_token:
        for pid, (_ppid, state, args_s) in _ps_process_table(include_args=True).items():
            if pid == exclude_pid or state.upper().startswith("Z"):
                continue
            if _ps_pid_matches_save_dir(pid, cookie_token, args_s):
                found.add(pid)
    return found


def _terminate_process_tree(
    proc: subprocess.Popen[Any],
    *,
    spawn_cookie: str = "",
    pipe_inodes: set[tuple[int, int]] | None = None,
) -> None:
    """Terminate `proc`, its POSIX process group, and any detached `setsid` descendants."""
    try:
        if os.name != "nt":
            from server.agy_runner import _collect_descendant_pids, _pid_is_alive_non_zombie

            my_pid = os.getpid()
            extra_pids: set[int] = set(_collect_descendant_pids({proc.pid}, exclude_pid=my_pid))
            extra_pids.update(
                _find_pipe_or_cookie_survivors(
                    pipe_inodes=pipe_inodes,
                    spawn_cookie=spawn_cookie,
                    exclude_pid=my_pid,
                )
            )
            extra_pids.discard(proc.pid)
            extra_pids.discard(my_pid)

            pgid_alive = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pgid_alive = False

            for ep in list(extra_pids):
                try:
                    os.killpg(ep, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    os.kill(ep, signal.SIGTERM)
                except OSError:
                    pass

            if proc.poll() is None:
                try:
                    proc.wait(timeout=0.4)
                except subprocess.TimeoutExpired:
                    pass
            elif pgid_alive or extra_pids:
                deadline_term = time.monotonic() + 0.25
                while time.monotonic() < deadline_term:
                    if not any(_pid_is_alive_non_zombie(ep) for ep in extra_pids):
                        break
                    time.sleep(0.02)

            extra_pids.update(
                _find_pipe_or_cookie_survivors(
                    pipe_inodes=pipe_inodes,
                    spawn_cookie=spawn_cookie,
                    exclude_pid=my_pid,
                )
            )
            extra_pids.discard(proc.pid)
            extra_pids.discard(my_pid)

            if pgid_alive or proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
            for ep in list(extra_pids):
                if _pid_is_alive_non_zombie(ep):
                    try:
                        os.killpg(ep, signal.SIGKILL)
                    except OSError:
                        pass
                    try:
                        os.kill(ep, signal.SIGKILL)
                    except OSError:
                        pass
            if proc.poll() is None:
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
        else:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
    except OSError:
        pass


def _is_read_only_worker_request(request: Mapping[str, Any]) -> bool:
    from server.agy_runner import is_read_only_tool_set

    return is_read_only_tool_set(
        request.get("enabled_tools"),
        request.get("disabled_tools"),
    )


def _format_cli_worker_prompt(
    prompt: str,
    *,
    system_prompt: str | None,
    allowed_paths: Sequence[str] | None,
    read_only: bool,
    is_followup: bool,
) -> str:
    if is_followup:
        return prompt
    guard_lines = [
        "[Execution Guardrails]",
        "- Stay strictly inside the current workspace directory; never run `git commit`, `git reset`, `git checkout`, or `git push`.",
    ]
    if read_only:
        guard_lines.append("- Read-only task: do NOT create, modify, or delete any files.")
    elif allowed_paths:
        joined = ", ".join(str(p) for p in allowed_paths[:20])
        guard_lines.append(f"- Allowed file scope: {joined}. Do not modify files outside this scope.")
    guard_lines.append("- Keep your final summary concise (under 250 words) listing files changed and checks run.")
    header = "\n".join(guard_lines)
    if system_prompt and system_prompt.strip():
        return f"{system_prompt.strip()}\n\n{header}\n\n{prompt}"
    return f"{header}\n\n{prompt}"


def _format_auth_remediation(raw_err: str, agy_bin: str, env: Mapping[str, str] | None = None) -> str:
    cfg_dir = agy_cli_config_dir(env)
    return (
        f"Gemini Enterprise `agy` CLI authentication failed: {raw_err.strip()}. "
        f"Run `{agy_bin}` once in a terminal to complete Google sign-in (credentials stored in `{cfg_dir}`), "
        "or set `ANTIGRAVITY_OAUTH_TOKEN`."
    )


def _merge_usage(acc: dict[str, int], turn_usage: Mapping[str, int]) -> None:
    for k in _USAGE_KEYS:
        acc[k] = acc.get(k, 0) + int(turn_usage.get(k, 0) or 0)


def _sum_step_usages(step_usages: Mapping[int, dict[str, int]]) -> dict[str, int]:
    acc = convert_agy_usage_to_vertex(None)
    for u in step_usages.values():
        _merge_usage(acc, u)
    return acc


def _start_stdin_writer(proc: subprocess.Popen[bytes], prompt_text: str) -> threading.Thread:
    """Spawn a daemon thread that writes `prompt_text` (UTF-8) to `proc.stdin` and closes the pipe."""

    def _write_stdin() -> None:
        if proc.stdin is None:
            return
        try:
            proc.stdin.write(prompt_text.encode("utf-8"))
            proc.stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    thread = threading.Thread(target=_write_stdin, daemon=True)
    thread.start()
    return thread


def _start_pipe_chunk_drainer(stream: Any, chunks: list[bytes]) -> threading.Thread:
    """Spawn a daemon thread that drains binary chunks from `stream` into `chunks` until EOF."""

    def _drain() -> None:
        assert stream is not None
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)
        except (OSError, ValueError):
            pass

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    return thread


def _compute_conversation_turn_delta(
    *,
    turn_usage: Mapping[str, int],
    step_usage_sum: Mapping[str, int],
    cur_num_turns: int,
    base_usage: Mapping[str, int] | None,
    base_turns: int,
    had_resumed_conversation: bool,
) -> tuple[dict[str, int], dict[str, int], int, bool]:
    """Compute `(delta_usage, updated_cum_usage, updated_turns, unknown_cumulative_baseline)`.

    `agy --conversation` builds `JSONOutput` from `session.Snapshot()` across all trajectory steps
    (`num_turns > 1`), so `turn_usage` is cumulative when `cur_num_turns > base_turns` and every
    counter is `>= base_usage`. If any counter shrinks (e.g. a mock or CLI build reporting per-turn
    numbers despite `num_turns > 1`), treat `turn_usage` as already per-turn so Turn-1 tokens are
    never lost.
    """
    clean_turn = {k: max(0, int(turn_usage.get(k, 0) or 0)) for k in _USAGE_KEYS}
    eff_turns = max(cur_num_turns, base_turns + 1 if had_resumed_conversation else 1)

    if base_usage is not None:
        clean_base = {k: max(0, int(base_usage.get(k, 0) or 0)) for k in _USAGE_KEYS}
        is_monotonic_cumulative = (
            cur_num_turns > max(0, base_turns)
            and cur_num_turns > 1
            and all(clean_turn[k] >= clean_base[k] for k in _USAGE_KEYS)
        )
        if is_monotonic_cumulative:
            delta = {k: max(0, clean_turn[k] - clean_base[k]) for k in _USAGE_KEYS}
            delta["total_token_count"] = (
                delta["prompt_token_count"] + delta["candidates_token_count"] + delta["thoughts_token_count"]
            )
            return delta, dict(clean_turn), eff_turns, False

        cum = dict(clean_base)
        _merge_usage(cum, clean_turn)
        return dict(clean_turn), cum, eff_turns, False

    if had_resumed_conversation and cur_num_turns > 1:
        step_tot = int(step_usage_sum.get("total_token_count", 0) or 0)
        if step_tot > 0:
            delta_from_steps = {k: max(0, int(step_usage_sum.get(k, 0) or 0)) for k in _USAGE_KEYS}
            return delta_from_steps, dict(clean_turn), eff_turns, False
        return dict(clean_turn), dict(clean_turn), eff_turns, True

    return dict(clean_turn), dict(clean_turn), eff_turns, False


def _run_stream_json_turn(
    *,
    agy_bin: str,
    prompt_text: str,
    cwd: str,
    model: str,
    effort: str | None,
    mode: str,
    skip_permissions: bool,
    conversation_id: str | None,
    timeout_seconds: float,
    register_cancel: Callable[[Callable[[], None]], None] | None,
    max_tool_calls: int | None = None,
    max_total_tokens: int | None = None,
    prior_tool_calls_count: int = 0,
    prior_tokens_count: int = 0,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not prompt_text or not prompt_text.strip():
        raise ValueError("Worker prompt cannot be empty.")
    timeout_sec = max(0.05, float(timeout_seconds))
    print_timeout_sec = max(5, int(round(timeout_sec)) + 15)

    cmd: list[str] = [
        agy_bin,
        *_custom_config_dir_args(env),
        "--output-format",
        "stream-json",
        "--disable-slash-commands",
        "--mode",
        mode,
        "--print-timeout",
        f"{print_timeout_sec}s",
    ]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["--effort", effort])
    if conversation_id:
        cmd.extend(["--conversation", conversation_id])

    spawn_cookie = secrets.token_hex(16)
    sub_env = build_agy_subprocess_env(env, workspace_dir=cwd)
    sub_env["_AGY_SPAWN_COOKIE"] = spawn_cookie

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name != "nt"),
        env=sub_env,
    )
    pipe_inodes = _snapshot_pipe_inodes(proc)

    cancelled_or_timed_out = threading.Event()
    timed_out_flag = threading.Event()
    budget_stop_reason: list[str] = []

    def _cancel_proc() -> None:
        cancelled_or_timed_out.set()
        _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)

    if callable(register_cancel):
        register_cancel(_cancel_proc)

    def _on_timeout() -> None:
        timed_out_flag.set()
        _cancel_proc()

    timer = threading.Timer(timeout_sec, _on_timeout)
    timer.daemon = True
    timer.start()

    stderr_chunks: list[bytes] = []
    line_q: queue.Queue[bytes | None] = queue.Queue()

    def _drain_stdout_lines() -> None:
        assert proc.stdout is not None
        try:
            while True:
                ln = proc.stdout.readline()
                if not ln:
                    break
                line_q.put(ln)
        except (OSError, ValueError):
            pass
        finally:
            line_q.put(None)

    t_out = threading.Thread(target=_drain_stdout_lines, daemon=True)
    t_out.start()
    t_err = _start_pipe_chunk_drainer(proc.stderr, stderr_chunks)
    t_in = _start_stdin_writer(proc, prompt_text)

    resolved_conv_id = conversation_id or ""
    step_tool_calls: list[dict[str, Any]] = []
    seen_step_tools: dict[tuple[int, str], int] = {}
    step_usages: dict[int, dict[str, int]] = {}
    fallback_tool_calls: list[dict[str, Any]] = []
    thoughts: list[str] = []
    delta_text_parts: list[str] = []
    assistant_text_parts: list[str] = []
    final_result_obj: dict[str, Any] | None = None
    got_result_frame = False
    auto_step_seq = -1
    deadline = time.monotonic() + timeout_sec

    def _check_live_budgets() -> bool:
        if max_tool_calls is not None and max_tool_calls > 0:
            active_calls = len(step_tool_calls) if step_tool_calls else len(fallback_tool_calls)
            live_calls = prior_tool_calls_count + active_calls
            if live_calls > max_tool_calls:
                budget_stop_reason.append("MAX_TOOL_CALLS_EXCEEDED")
                _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
                return True
        if max_total_tokens is not None and max_total_tokens > 0 and step_usages:
            live_toks = prior_tokens_count + _sum_step_usages(step_usages)["total_token_count"]
            if live_toks > max_total_tokens:
                budget_stop_reason.append("MAX_TOTAL_TOKENS_EXCEEDED")
                _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
                return True
        return False

    def _process_ndjson_line(raw_line: bytes, *, allow_result: bool = True) -> str:
        """Parse one NDJSON line; return `'result'`, `'budget'`, or `''`."""
        nonlocal resolved_conv_id, auto_step_seq, got_result_frame, final_result_obj
        line_str = raw_line.decode("utf-8", errors="replace").strip()
        if not line_str or not line_str.startswith("{"):
            return ""
        try:
            evt = json.loads(line_str)
        except ValueError:
            return ""
        if not isinstance(evt, dict):
            return ""

        evt_kind = str(evt.get("event") or evt.get("type") or "").strip().lower()
        if isinstance(evt.get("conversation_id"), str) and evt["conversation_id"]:
            resolved_conv_id = evt["conversation_id"]

        if evt_kind == "step_update":
            su = evt.get("step_update")
            if isinstance(su, dict):
                if isinstance(su.get("conversation_id"), str) and su["conversation_id"]:
                    resolved_conv_id = su["conversation_id"]
                if isinstance(su.get("text_delta"), str) and su["text_delta"]:
                    delta_text_parts.append(su["text_delta"])
                raw_s_idx = su.get("step_index")
                if isinstance(raw_s_idx, int) and not isinstance(raw_s_idx, bool) and raw_s_idx > 0:
                    s_idx = raw_s_idx
                else:
                    s_idx = auto_step_seq
                    auto_step_seq -= 1
                if isinstance(su.get("usage"), dict):
                    step_usages[s_idx] = convert_agy_usage_to_vertex(su["usage"])
                t_info = su.get("tool_info")
                t_name = str(
                    (t_info.get("name") if isinstance(t_info, dict) else None)
                    or su.get("tool_name")
                    or ""
                ).strip()
                if t_name:
                    params = t_info.get("parameters") if isinstance(t_info, dict) else None
                    parsed_args = dict(params) if isinstance(params, dict) else {}
                    if s_idx > 0 and (s_idx, t_name) in seen_step_tools:
                        if parsed_args:
                            step_tool_calls[seen_step_tools[(s_idx, t_name)]]["args"] = parsed_args
                    else:
                        step_tool_calls.append({"name": t_name, "args": parsed_args})
                        if s_idx > 0:
                            seen_step_tools[(s_idx, t_name)] = len(step_tool_calls) - 1
                if allow_result and _check_live_budgets():
                    return "budget"
        elif evt_kind == "assistant":
            msg = evt.get("message")
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "thinking" and isinstance(block.get("thinking"), str):
                        thoughts.append(block["thinking"])
                    elif btype == "tool_use" and isinstance(block.get("name"), str):
                        inp = block.get("input")
                        fallback_tool_calls.append(
                            {"name": block["name"], "args": dict(inp) if isinstance(inp, dict) else {}}
                        )
                    elif btype == "text" and isinstance(block.get("text"), str):
                        assistant_text_parts.append(block["text"])
                if allow_result and _check_live_budgets():
                    return "budget"
        elif evt_kind == "result" and allow_result:
            got_result_frame = True
            res_payload = evt.get("result")
            if isinstance(res_payload, dict):
                final_result_obj = res_payload
                if isinstance(res_payload.get("conversation_id"), str) and res_payload["conversation_id"]:
                    resolved_conv_id = res_payload["conversation_id"]
            else:
                final_result_obj = evt
            return "result"
        return ""

    try:
        while not cancelled_or_timed_out.is_set() and not budget_stop_reason:
            rem = deadline - time.monotonic()
            if rem <= 0:
                timed_out_flag.set()
                cancelled_or_timed_out.set()
                break
            poll_wait = 0.05 if proc.poll() is not None else min(0.1, rem)
            try:
                raw_line = line_q.get(timeout=poll_wait)
            except queue.Empty:
                if proc.poll() is not None and line_q.empty():
                    # Leader already exited; only orphaned children can still hold stdout open.
                    break
                continue
            if raw_line is None:
                break

            action = _process_ndjson_line(raw_line, allow_result=True)
            if action in ("result", "budget"):
                # `result` is the terminal NDJSON frame; break immediately and give `agy` a
                # graceful window to flush conversation state before cleaning up descendants.
                break

        if got_result_frame and not cancelled_or_timed_out.is_set() and not budget_stop_reason:
            timer.cancel()
            post_result_grace = min(8.0, max(1.0, deadline - time.monotonic()))
            try:
                proc.wait(timeout=post_result_grace)
            except subprocess.TimeoutExpired:
                pass
        elif not cancelled_or_timed_out.is_set() and not budget_stop_reason:
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
    finally:
        timer.cancel()
        _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
        t_in.join(timeout=0.3)
        t_out.join(timeout=0.3)
        t_err.join(timeout=0.3)
        _close_proc_pipes_safely(proc, stdout_thread=t_out, stderr_thread=t_err)

    while True:
        try:
            rem_line = line_q.get_nowait()
        except queue.Empty:
            break
        if rem_line is not None:
            _process_ndjson_line(rem_line, allow_result=not timed_out_flag.is_set())

    stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
    tool_calls = step_tool_calls if step_tool_calls else fallback_tool_calls
    step_usage_sum = _sum_step_usages(step_usages)

    if isinstance(final_result_obj, dict) and isinstance(final_result_obj.get("usage"), dict):
        vertex_usage = convert_agy_usage_to_vertex(final_result_obj["usage"])
        usage_from_result_snapshot = True
    elif step_usages:
        vertex_usage = step_usage_sum
        usage_from_result_snapshot = False
    else:
        vertex_usage = convert_agy_usage_to_vertex(None)
        usage_from_result_snapshot = False

    if timed_out_flag.is_set() and not budget_stop_reason and not got_result_frame:
        err = TimeoutError(f"Antigravity CLI worker timed out after {timeout_sec:.1f}s")
        err.partial_usage = vertex_usage  # type: ignore[attr-defined]
        err.partial_usage_from_snapshot = usage_from_result_snapshot  # type: ignore[attr-defined]
        err.partial_step_usage_sum = step_usage_sum  # type: ignore[attr-defined]
        err.partial_tool_calls = tool_calls  # type: ignore[attr-defined]
        err.partial_thoughts = thoughts  # type: ignore[attr-defined]
        err.partial_containment = "agy_cli"  # type: ignore[attr-defined]
        err.partial_agy_conversation_id = resolved_conv_id or None  # type: ignore[attr-defined]
        raise err

    status_str = ""
    err_str = ""
    response_text = ""
    denied_actions: list[Any] = []
    num_turns = 0
    if isinstance(final_result_obj, dict):
        num_turns = int(final_result_obj.get("num_turns") or 0)
        status_str = str(final_result_obj.get("status") or final_result_obj.get("subtype") or "").upper()
        if final_result_obj.get("is_error") is True and not status_str:
            status_str = "ERROR"
        raw_err = final_result_obj.get("error")
        if isinstance(raw_err, str) and raw_err.strip():
            err_str = raw_err.strip()
        elif isinstance(final_result_obj.get("errors"), list) and final_result_obj["errors"]:
            err_str = "; ".join(str(x) for x in final_result_obj["errors"])
        if isinstance(final_result_obj.get("response"), str):
            response_text = final_result_obj["response"]
        elif isinstance(final_result_obj.get("result"), str):
            response_text = final_result_obj["result"]
        if isinstance(final_result_obj.get("denied_actions"), list):
            denied_actions = final_result_obj["denied_actions"]

    streamed_text = "".join(delta_text_parts) if delta_text_parts else "".join(assistant_text_parts)
    if not response_text and streamed_text:
        response_text = streamed_text

    if budget_stop_reason:
        return {
            "ok": True,
            "text": response_text,
            "usage": vertex_usage,
            "step_usage_sum": step_usage_sum,
            "usage_from_snapshot": usage_from_result_snapshot,
            "num_turns": num_turns,
            "tool_calls": tool_calls,
            "thoughts": thoughts,
            "stop_reason": budget_stop_reason[0],
            "conversation_id": resolved_conv_id,
        }

    missing_envelope = final_result_obj is None and not streamed_text
    has_non_error_result = got_result_frame and status_str != "ERROR" and not err_str
    failed = (status_str == "ERROR") or bool(err_str) or missing_envelope or (proc.returncode != 0 and not has_non_error_result)
    if failed:
        combined_err = (
            err_str
            or stderr_text
            or (
                "Antigravity CLI exited without emitting a result envelope"
                if missing_envelope and proc.returncode == 0
                else f"Antigravity CLI exited with code {proc.returncode}"
            )
        )
        return {
            "ok": False,
            "error": combined_err,
            "returncode": proc.returncode,
            "conversation_id": resolved_conv_id,
            "num_turns": num_turns,
            "usage": vertex_usage,
            "step_usage_sum": step_usage_sum,
            "usage_from_snapshot": usage_from_result_snapshot,
            "tool_calls": tool_calls,
            "thoughts": thoughts,
            "text": response_text,
        }

    stop_reason = "DENIED_ACTIONS" if denied_actions else "STOP"
    if stop_reason == "STOP":
        if max_tool_calls is not None and max_tool_calls > 0 and (prior_tool_calls_count + len(tool_calls)) > max_tool_calls:
            stop_reason = "MAX_TOOL_CALLS_EXCEEDED"
    return {
        "ok": True,
        "text": response_text,
        "usage": vertex_usage,
        "step_usage_sum": step_usage_sum,
        "usage_from_snapshot": usage_from_result_snapshot,
        "num_turns": num_turns,
        "tool_calls": tool_calls,
        "thoughts": thoughts,
        "stop_reason": stop_reason,
        "conversation_id": resolved_conv_id,
    }


def _resolve_cached_model_for_cli(
    agy_bin: str,
    requested_model: str,
    effort: str | None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str | None, str | None]:
    cfg_key = str(agy_cli_config_dir(env))
    cache_key = (agy_bin, cfg_key, requested_model, effort or "")
    with _MODEL_FALLBACK_LOCK:
        cached = _MODEL_FALLBACK_CACHE.get(cache_key)
    if cached:
        eff_after = None if _model_has_explicit_effort(cached) else effort
        note = f"Mapped requested model {requested_model!r} to offered Antigravity CLI model {cached!r}."
        return cached, eff_after, note
    return requested_model, effort, None


def _maybe_retry_model_error(
    *,
    err_text: str,
    agy_bin: str,
    requested_model: str,
    effort: str | None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str | None, str] | None:
    lower = err_text.lower()
    available = _extract_available_models(err_text)
    if available and ("not recognized as a known model" in lower or "invalid model selection" in lower):
        chosen = select_fallback_agy_model(requested_model, effort, available)
        if chosen and chosen != requested_model:
            cfg_key = str(agy_cli_config_dir(env))
            with _MODEL_FALLBACK_LOCK:
                _MODEL_FALLBACK_CACHE[(agy_bin, cfg_key, requested_model, effort or "")] = chosen
            eff_after = None if _model_has_explicit_effort(chosen) else effort
            note = f"Mapped requested model {requested_model!r} to offered Antigravity CLI model {chosen!r}."
            return chosen, eff_after, note

    if effort and ("has no " in lower and "effort" in lower or "effort is not supported" in lower):
        note = f"Omitted unsupported --effort={effort!r} for model {requested_model!r}."
        return requested_model, None, note

    return None


def _is_missing_conversation_error(err_text: str) -> bool:
    low = (err_text or "").lower()
    return "conversation" in low and any(
        phrase in low
        for phrase in ("not found", "does not exist", "unknown conversation", "no such conversation", "failed to load")
    )


def run_agy_cli_worker(request: dict[str, Any]) -> dict[str, Any]:
    """Execute an agentic worker job through external `agy --output-format stream-json`."""
    env = request.get("env") if isinstance(request.get("env"), Mapping) else None
    agy_bin = _resolve_or_bootstrap_agy_bin(env)
    cfg_key = str(agy_cli_config_dir(env))

    workspace_dir = str(request["workspace_dir"])
    requested_model = str(request.get("model") or "gemini-flash-next")
    thinking_level = request.get("thinking_level")
    effort = map_thinking_level_to_effort(thinking_level, requested_model)
    active_model, active_effort, reroute_note = _resolve_cached_model_for_cli(
        agy_bin, requested_model, effort, env=env
    )

    read_only = _is_read_only_worker_request(request)
    mode = "plan" if read_only else "auto"
    skip_permissions = not read_only

    prompts: list[str] = list(request.get("prompts") or [])
    if not prompts and request.get("prompt"):
        prompts = [str(request["prompt"])]
    if not prompts:
        raise ValueError("Worker prompt cannot be empty.")

    timeout_seconds = float(request.get("timeout_seconds") or 600.0)
    register_cancel = request.get("register_cancel")
    max_tool_calls = int(request["max_tool_calls"]) if request.get("max_tool_calls") else None
    max_total_tokens = int(request["max_total_tokens"]) if request.get("max_total_tokens") else None

    conv_id: str | None = (
        str(request["agy_conversation_id"]).strip()
        if request.get("agy_conversation_id")
        else None
    )

    base_conv_usage: dict[str, int] | None = None
    base_conv_turns = 0
    raw_prior_conv_u = request.get("agy_prior_conversation_usage")
    if raw_prior_conv_u is None:
        raw_prior_conv_u = request.get("prior_agy_usage")
    if isinstance(raw_prior_conv_u, Mapping) and any(int(raw_prior_conv_u.get(k, 0) or 0) > 0 for k in _USAGE_KEYS):
        base_conv_usage = {k: max(0, int(raw_prior_conv_u.get(k, 0) or 0)) for k in _USAGE_KEYS}
        base_conv_turns = max(1, int(request.get("agy_prior_conversation_turns") or 1))
    elif conv_id:
        with _CONV_LEDGER_LOCK:
            ledger_hit = _CONVERSATION_USAGE_LEDGER.get((cfg_key, conv_id))
        if ledger_hit is not None:
            base_conv_turns, ledger_u = ledger_hit
            base_conv_usage = dict(ledger_u)

    agg_usage = convert_agy_usage_to_vertex(None)
    all_tool_calls: list[dict[str, Any]] = []
    all_thoughts: list[str] = []
    last_text = ""
    stop_reason = "STOP"
    unknown_cumulative_baseline = False

    for turn_idx, raw_prompt in enumerate(prompts):
        had_resumed = bool(conv_id)
        formatted_prompt = _format_cli_worker_prompt(
            raw_prompt,
            system_prompt=request.get("system_prompt"),
            allowed_paths=request.get("allowed_paths"),
            read_only=read_only,
            is_followup=(turn_idx > 0 or had_resumed),
        )
        try:
            turn_out = _run_stream_json_turn(
                agy_bin=agy_bin,
                prompt_text=formatted_prompt,
                cwd=workspace_dir,
                model=active_model,
                effort=active_effort,
                mode=mode,
                skip_permissions=skip_permissions,
                conversation_id=conv_id,
                timeout_seconds=timeout_seconds,
                register_cancel=register_cancel if callable(register_cancel) else None,
                max_tool_calls=max_tool_calls,
                max_total_tokens=max_total_tokens,
                prior_tool_calls_count=len(all_tool_calls),
                prior_tokens_count=agg_usage["total_token_count"],
                env=env,
            )

            if not turn_out["ok"] and turn_idx == 0 and conv_id and _is_missing_conversation_error(turn_out["error"]):
                conv_id = None
                had_resumed = False
                base_conv_usage = None
                base_conv_turns = 0
                formatted_prompt = _format_cli_worker_prompt(
                    raw_prompt,
                    system_prompt=request.get("system_prompt"),
                    allowed_paths=request.get("allowed_paths"),
                    read_only=read_only,
                    is_followup=False,
                )
                turn_out = _run_stream_json_turn(
                    agy_bin=agy_bin,
                    prompt_text=formatted_prompt,
                    cwd=workspace_dir,
                    model=active_model,
                    effort=active_effort,
                    mode=mode,
                    skip_permissions=skip_permissions,
                    conversation_id=None,
                    timeout_seconds=timeout_seconds,
                    register_cancel=register_cancel if callable(register_cancel) else None,
                    max_tool_calls=max_tool_calls,
                    max_total_tokens=max_total_tokens,
                    prior_tool_calls_count=len(all_tool_calls),
                    prior_tokens_count=agg_usage["total_token_count"],
                    env=env,
                )

            seen_worker_states: set[tuple[str, str | None]] = {(active_model, active_effort)}
            for _retry_idx in range(2):
                if turn_out["ok"]:
                    break
                retry_spec = _maybe_retry_model_error(
                    err_text=turn_out["error"],
                    agy_bin=agy_bin,
                    requested_model=active_model,
                    effort=active_effort,
                    env=env,
                )
                if retry_spec is None:
                    break
                next_model, next_effort, reroute_note = retry_spec
                if (next_model, next_effort) in seen_worker_states:
                    break
                seen_worker_states.add((next_model, next_effort))
                active_model, active_effort = next_model, next_effort
                turn_out = _run_stream_json_turn(
                    agy_bin=agy_bin,
                    prompt_text=formatted_prompt,
                    cwd=workspace_dir,
                    model=active_model,
                    effort=active_effort,
                    mode=mode,
                    skip_permissions=skip_permissions,
                    conversation_id=conv_id,
                    timeout_seconds=timeout_seconds,
                    register_cancel=register_cancel if callable(register_cancel) else None,
                    max_tool_calls=max_tool_calls,
                    max_total_tokens=max_total_tokens,
                    prior_tool_calls_count=len(all_tool_calls),
                    prior_tokens_count=agg_usage["total_token_count"],
                    env=env,
                )
        except TimeoutError as exc:
            part_u = getattr(exc, "partial_usage", None) or {}
            if getattr(exc, "partial_usage_from_snapshot", False):
                delta_part, partial_cum_u, partial_cum_turns, _ = _compute_conversation_turn_delta(
                    turn_usage=part_u,
                    step_usage_sum=getattr(exc, "partial_step_usage_sum", None) or {},
                    cur_num_turns=base_conv_turns + 1,
                    base_usage=base_conv_usage,
                    base_turns=base_conv_turns,
                    had_resumed_conversation=had_resumed,
                )
            else:
                delta_part = {k: max(0, int(part_u.get(k, 0) or 0)) for k in _USAGE_KEYS}
                partial_cum_u = dict(base_conv_usage or convert_agy_usage_to_vertex(None))
                _merge_usage(partial_cum_u, delta_part)
                has_turn_toks = any(delta_part.get(k, 0) > 0 for k in _USAGE_KEYS)
                partial_cum_turns = base_conv_turns + (1 if has_turn_toks else 0)
            _merge_usage(agg_usage, delta_part)
            all_tool_calls.extend(getattr(exc, "partial_tool_calls", None) or [])
            all_thoughts.extend(getattr(exc, "partial_thoughts", None) or [])
            eff_partial_conv = getattr(exc, "partial_agy_conversation_id", None) or conv_id or None
            if eff_partial_conv and any(partial_cum_u.get(k, 0) > 0 for k in _USAGE_KEYS):
                with _CONV_LEDGER_LOCK:
                    _CONVERSATION_USAGE_LEDGER[(cfg_key, str(eff_partial_conv))] = (
                        max(1, partial_cum_turns),
                        dict(partial_cum_u),
                    )
            exc.partial_usage = agg_usage  # type: ignore[attr-defined]
            exc.partial_tool_calls = all_tool_calls  # type: ignore[attr-defined]
            exc.partial_thoughts = all_thoughts  # type: ignore[attr-defined]
            exc.partial_containment = "agy_cli"  # type: ignore[attr-defined]
            exc.partial_agy_conversation_id = eff_partial_conv  # type: ignore[attr-defined]
            if any(partial_cum_u.get(k, 0) > 0 for k in _USAGE_KEYS):
                exc.partial_agy_conversation_usage = dict(partial_cum_u)  # type: ignore[attr-defined]
                exc.partial_agy_conversation_turns = max(1, partial_cum_turns)  # type: ignore[attr-defined]
            raise

        if turn_out.get("usage_from_snapshot", True):
            delta_u, updated_cum_u, updated_turns, unk_cum = _compute_conversation_turn_delta(
                turn_usage=turn_out["usage"],
                step_usage_sum=turn_out.get("step_usage_sum") or {},
                cur_num_turns=int(turn_out.get("num_turns") or 0),
                base_usage=base_conv_usage,
                base_turns=base_conv_turns,
                had_resumed_conversation=had_resumed,
            )
            if unk_cum:
                unknown_cumulative_baseline = True
        else:
            delta_u = dict(turn_out["usage"])
            updated_cum_u = dict(base_conv_usage or convert_agy_usage_to_vertex(None))
            _merge_usage(updated_cum_u, delta_u)
            updated_turns = max(int(turn_out.get("num_turns") or 0), base_conv_turns + 1)

        _merge_usage(agg_usage, delta_u)
        all_tool_calls.extend(turn_out["tool_calls"])
        all_thoughts.extend(turn_out["thoughts"])
        if turn_out.get("conversation_id"):
            conv_id = str(turn_out["conversation_id"])

        if turn_out["ok"] or any(delta_u.get(k, 0) > 0 for k in _USAGE_KEYS):
            base_conv_usage = updated_cum_u
            base_conv_turns = updated_turns
            if conv_id and any(base_conv_usage.get(k, 0) > 0 for k in _USAGE_KEYS):
                with _CONV_LEDGER_LOCK:
                    _CONVERSATION_USAGE_LEDGER[(cfg_key, conv_id)] = (base_conv_turns, dict(base_conv_usage))

        if not turn_out["ok"]:
            err_msg = str(turn_out["error"])
            if is_agy_auth_error(err_msg):
                err_msg = _format_auth_remediation(err_msg, agy_bin, env=env)
            exc = RuntimeError(err_msg)
            exc.partial_usage = agg_usage  # type: ignore[attr-defined]
            exc.partial_tool_calls = all_tool_calls  # type: ignore[attr-defined]
            exc.partial_thoughts = all_thoughts  # type: ignore[attr-defined]
            exc.partial_containment = "agy_cli"  # type: ignore[attr-defined]
            exc.partial_agy_conversation_id = conv_id or None  # type: ignore[attr-defined]
            if base_conv_usage is not None and any(base_conv_usage.get(k, 0) > 0 for k in _USAGE_KEYS):
                exc.partial_agy_conversation_usage = dict(base_conv_usage)  # type: ignore[attr-defined]
                exc.partial_agy_conversation_turns = max(1, base_conv_turns)  # type: ignore[attr-defined]
            raise exc

        last_text = str(turn_out.get("text") or "")
        stop_reason = str(turn_out.get("stop_reason") or "STOP")
        if (
            stop_reason == "STOP"
            and max_total_tokens is not None
            and max_total_tokens > 0
            and agg_usage["total_token_count"] > max_total_tokens
        ):
            stop_reason = "MAX_TOTAL_TOKENS_EXCEEDED"
        if stop_reason in ("MAX_TOOL_CALLS_EXCEEDED", "MAX_TOTAL_TOKENS_EXCEEDED", "DENIED_ACTIONS"):
            break

    return {
        "text": last_text,
        "usage": agg_usage,
        "tool_calls": all_tool_calls,
        "thoughts": all_thoughts,
        "stop_reason": stop_reason,
        "containment": "agy_cli",
        "containment_warning": None,
        "command_runs": [],
        "agy_conversation_id": conv_id or None,
        "agy_conversation_usage": dict(base_conv_usage) if base_conv_usage is not None else None,
        "agy_conversation_turns": base_conv_turns if base_conv_turns > 0 else None,
        "usage_is_cumulative": unknown_cumulative_baseline,
        "reroute_note": reroute_note,
    }


def _run_json_envelope_once(
    *,
    agy_bin: str,
    prompt_text: str,
    cwd: str,
    model: str,
    effort: str | None,
    json_schema: dict[str, Any] | None,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not prompt_text or not prompt_text.strip():
        raise ValueError("Worker prompt cannot be empty.")
    timeout_sec = max(0.05, float(timeout_seconds))
    print_timeout_sec = max(5, int(round(timeout_sec)) + 15)

    cmd: list[str] = [
        agy_bin,
        *_custom_config_dir_args(env),
        "--output-format",
        "json",
        "--mode",
        "plan",
        "--disable-slash-commands",
        "--print-timeout",
        f"{print_timeout_sec}s",
    ]
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["--effort", effort])
    if json_schema is not None:
        cmd.extend(["--json-schema", json.dumps(json_schema, separators=(",", ":"))])

    spawn_cookie = secrets.token_hex(16)
    sub_env = build_agy_subprocess_env(env)
    sub_env["_AGY_SPAWN_COOKIE"] = spawn_cookie

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name != "nt"),
        env=sub_env,
    )
    pipe_inodes = _snapshot_pipe_inodes(proc)

    timed_out_flag = threading.Event()
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    t_in = _start_stdin_writer(proc, prompt_text)
    t_out = _start_pipe_chunk_drainer(proc.stdout, stdout_chunks)
    t_err = _start_pipe_chunk_drainer(proc.stderr, stderr_chunks)

    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out_flag.set()
        _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
    finally:
        _terminate_process_tree(proc, spawn_cookie=spawn_cookie, pipe_inodes=pipe_inodes)
        t_in.join(timeout=0.3)
        t_out.join(timeout=0.3)
        t_err.join(timeout=0.3)
        _close_proc_pipes_safely(proc, stdout_thread=t_out, stderr_thread=t_err)

    if timed_out_flag.is_set():
        raise TimeoutError(f"Antigravity CLI request timed out after {timeout_sec:.1f}s")

    stdout_text = b"".join(stdout_chunks).decode("utf-8", errors="replace").strip()
    stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()

    parsed: dict[str, Any] | None = None
    if stdout_text:
        for candidate in (stdout_text, stdout_text.splitlines()[-1]):
            cand_s = candidate.strip()
            if cand_s.startswith("{") and cand_s.endswith("}"):
                try:
                    obj = json.loads(cand_s)
                    if isinstance(obj, dict):
                        parsed = obj
                        break
                except ValueError:
                    pass

    raw_usage = parsed.get("usage") if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict) else None
    vertex_usage = convert_agy_usage_to_vertex(raw_usage)

    status_str = str(parsed.get("status") or "").upper() if isinstance(parsed, dict) else ""
    err_field = ""
    if isinstance(parsed, dict):
        if isinstance(parsed.get("error"), str) and parsed["error"].strip():
            err_field = parsed["error"].strip()
        elif parsed.get("is_error") is True and isinstance(parsed.get("result"), str):
            err_field = parsed["result"].strip()

    missing_json_envelope = parsed is None and not stdout_text
    if proc.returncode != 0 or status_str == "ERROR" or missing_json_envelope:
        return {
            "ok": False,
            "error": (
                err_field
                or stderr_text
                or stdout_text
                or (
                    "Antigravity CLI exited without emitting a JSON result envelope"
                    if missing_json_envelope and proc.returncode == 0
                    else f"Antigravity CLI exited with code {proc.returncode}"
                )
            ),
            "usage": vertex_usage,
            "parsed": parsed,
        }

    text_out = ""
    if isinstance(parsed, dict):
        if "structured_output" in parsed and parsed["structured_output"] is not None:
            so = parsed["structured_output"]
            text_out = so if isinstance(so, str) else json.dumps(so)
        elif isinstance(parsed.get("response"), str):
            text_out = parsed["response"]
        elif isinstance(parsed.get("result"), str):
            text_out = parsed["result"]
    elif stdout_text:
        text_out = stdout_text

    return {
        "ok": True,
        "text": text_out,
        "usage": vertex_usage,
        "parsed": parsed,
    }


def run_agy_cli_ask(request: dict[str, Any]) -> dict[str, Any]:
    """Execute a single-shot `ask_gemini` / QA audit call via `agy --output-format json --mode plan`."""
    env = request.get("env") if isinstance(request.get("env"), Mapping) else None
    agy_bin = _resolve_or_bootstrap_agy_bin(env)

    requested_model = str(request.get("model") or "gemini-flash-next")
    effort = map_thinking_level_to_effort(request.get("thinking_level"), requested_model)
    active_model, active_effort, reroute_note = _resolve_cached_model_for_cli(
        agy_bin, requested_model, effort, env=env
    )

    prompt = str(request.get("prompt") or "")
    sys_inst = str(request.get("system_instruction") or "").strip()
    no_tools_note = (
        "[Single-Shot Query Mode: Answer directly from the prompt context below without invoking workspace tools.]"
    )
    if sys_inst:
        full_prompt = f"{sys_inst}\n\n{no_tools_note}\n\n{prompt}"
    else:
        full_prompt = f"{no_tools_note}\n\n{prompt}"

    if request.get("timeout_seconds") is not None:
        timeout_seconds = float(request["timeout_seconds"])
    elif request.get("timeout_ms") is not None:
        timeout_seconds = float(request["timeout_ms"]) / 1000.0
    else:
        timeout_seconds = 180.0
    with tempfile.TemporaryDirectory(prefix="cc-agy-ask-") as empty_cwd:
        out = _run_json_envelope_once(
            agy_bin=agy_bin,
            prompt_text=full_prompt,
            cwd=empty_cwd,
            model=active_model,
            effort=active_effort,
            json_schema=None,
            timeout_seconds=timeout_seconds,
            env=env,
        )
        seen_ask_states: set[tuple[str, str | None]] = {(active_model, active_effort)}
        for _retry_idx in range(2):
            if out["ok"]:
                break
            retry_spec = _maybe_retry_model_error(
                err_text=out["error"],
                agy_bin=agy_bin,
                requested_model=active_model,
                effort=active_effort,
                env=env,
            )
            if retry_spec is None:
                break
            next_model, next_effort, reroute_note = retry_spec
            if (next_model, next_effort) in seen_ask_states:
                break
            seen_ask_states.add((next_model, next_effort))
            active_model, active_effort = next_model, next_effort
            out = _run_json_envelope_once(
                agy_bin=agy_bin,
                prompt_text=full_prompt,
                cwd=empty_cwd,
                model=active_model,
                effort=active_effort,
                json_schema=None,
                timeout_seconds=timeout_seconds,
                env=env,
            )

    if not out["ok"]:
        err_msg = str(out["error"])
        if is_agy_auth_error(err_msg):
            err_msg = _format_auth_remediation(err_msg, agy_bin, env=env)
        exc = RuntimeError(err_msg)
        exc.partial_usage = out["usage"]  # type: ignore[attr-defined]
        raise exc

    return {
        "text": str(out.get("text") or ""),
        "usage": out["usage"],
        "reroute_note": reroute_note,
    }


def run_agy_cli_structured_writer(
    *,
    prompt: str,
    system_instruction: str,
    model: str,
    thinking_level: str | None,
    json_schema: dict[str, Any],
    timeout_seconds: float = 300.0,
    read_budget: int = 0,
    workspace_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, int]]:
    """Run a structured JSON-schema batch writer turn via `agy --output-format json`."""
    agy_bin = _resolve_or_bootstrap_agy_bin(env)
    effort = map_thinking_level_to_effort(thinking_level, model)
    active_model, active_effort, _note = _resolve_cached_model_for_cli(
        agy_bin, model, effort, env=env
    )

    if read_budget > 0 and workspace_dir and Path(workspace_dir).is_dir():
        budget_rule = (
            f"[Read-Only Batch Writer Mode: You may read at most {int(read_budget)} workspace file(s) "
            "if needed, must NOT edit files or run commands, and must return valid JSON matching the schema.]"
        )
        full_prompt = f"{system_instruction.strip()}\n\n{budget_rule}\n\n{prompt}"
        out = _consume_writer_envelope(
            agy_bin=agy_bin,
            full_prompt=full_prompt,
            cwd=str(workspace_dir),
            active_model=active_model,
            active_effort=active_effort,
            json_schema=json_schema,
            timeout_seconds=timeout_seconds,
            env=env,
        )
    else:
        budget_rule = (
            "[Pure Generation Batch Writer Mode: Do NOT call any tools; generate the JSON object "
            "matching the schema directly from the provided prompt context.]"
        )
        full_prompt = f"{system_instruction.strip()}\n\n{budget_rule}\n\n{prompt}"
        with tempfile.TemporaryDirectory(prefix="cc-agy-writer-") as empty_cwd:
            out = _consume_writer_envelope(
                agy_bin=agy_bin,
                full_prompt=full_prompt,
                cwd=empty_cwd,
                active_model=active_model,
                active_effort=active_effort,
                json_schema=json_schema,
                timeout_seconds=timeout_seconds,
                env=env,
            )

    if not out["ok"]:
        err_msg = str(out["error"])
        if is_agy_auth_error(err_msg):
            err_msg = _format_auth_remediation(err_msg, agy_bin, env=env)
        exc = RuntimeError(err_msg)
        exc.partial_usage = out["usage"]  # type: ignore[attr-defined]
        raise exc

    return str(out.get("text") or ""), out["usage"]


def _consume_writer_envelope(
    *,
    agy_bin: str,
    full_prompt: str,
    cwd: str,
    active_model: str,
    active_effort: str | None,
    json_schema: dict[str, Any],
    timeout_seconds: float,
    env: Mapping[str, str] | None,
) -> dict[str, Any]:
    cur_model = active_model
    cur_effort = active_effort
    cur_schema: dict[str, Any] | None = json_schema
    seen_model_states: set[tuple[str, str | None]] = {(cur_model, cur_effort)}
    out: dict[str, Any] = {"ok": False, "error": "Writer did not execute", "usage": convert_agy_usage_to_vertex(None)}

    for _attempt in range(4):
        out = _run_json_envelope_once(
            agy_bin=agy_bin,
            prompt_text=full_prompt,
            cwd=cwd,
            model=cur_model,
            effort=cur_effort,
            json_schema=cur_schema,
            timeout_seconds=timeout_seconds,
            env=env,
        )
        if out["ok"]:
            return out
        err_str = str(out.get("error") or "")
        err_low = err_str.lower()
        if cur_schema is not None and "--json-schema" in err_str and (
            "not supported" in err_low
            or "unknown" in err_low
            or "unrecognized" in err_low
            or "flag provided but not defined" in err_low
        ):
            cur_schema = None
            continue
        retry_spec = _maybe_retry_model_error(
            err_text=err_str,
            agy_bin=agy_bin,
            requested_model=cur_model,
            effort=cur_effort,
            env=env,
        )
        if retry_spec is not None:
            next_model, next_effort, _ = retry_spec
            if (next_model, next_effort) not in seen_model_states:
                seen_model_states.add((next_model, next_effort))
                cur_model, cur_effort = next_model, next_effort
                continue
        return out
    return out
