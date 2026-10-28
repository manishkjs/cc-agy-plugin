"""Offline readiness and environment diagnostics for cc-agy-plugin (`doctor`).

Checks MCP tool registration, cost profile and default model routing, authentication
configuration (never exposing raw API keys), required Python SDKs (`google-genai`,
`google.antigravity`, Straitjacket `ctx`), workspace git state, language test-runner
discovery, and receipt store writability without making any network or model calls.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Mapping

from server.jobs import detect_project_test_runner
from server.receipts import ReceiptStore, resolve_workspace_root
from server.verify import run_git
from server.regions import (
    is_strict_mode_enabled,
    resolve_auth_config,
    resolve_cost_profile,
    resolve_default_ask_model,
    resolve_default_delegate_model,
    resolve_max_daily_cents,
    resolve_vertex_base_url,
)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _default_adc_candidates(env: Mapping[str, str]) -> list[Path]:
    explicit = (env.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    candidates: list[Path] = []
    if explicit and not explicit.startswith("${"):
        candidates.append(Path(explicit).expanduser())
    gcloud_cfg = (env.get("CLOUDSDK_CONFIG") or "").strip()
    if gcloud_cfg and not gcloud_cfg.startswith("${"):
        candidates.append(Path(gcloud_cfg).expanduser() / "application_default_credentials.json")
    home = Path.home()
    candidates.append(home / ".config" / "gcloud" / "application_default_credentials.json")
    appdata = (env.get("APPDATA") or "").strip()
    if appdata:
        candidates.append(Path(appdata) / "gcloud" / "application_default_credentials.json")
    return candidates


def run_doctor(
    workspace_root: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    store: ReceiptStore | None = None,
) -> dict[str, Any]:
    """Run local zero-network readiness checks and return a structured report."""
    active_env: Mapping[str, str] = os.environ if env is None else env
    issues: list[str] = []
    warnings: list[str] = []
    remediations: list[str] = []

    cost_profile = resolve_cost_profile(active_env)
    default_delegate_model = resolve_default_delegate_model(None, env=active_env)
    default_ask_model = resolve_default_ask_model(None, env=active_env)
    strict_mode = is_strict_mode_enabled(active_env)

    # 1. MCP tool catalog self-check
    try:
        from server.mcp_stdio import TOOL_DEFINITIONS, get_tool_definitions

        active_tools = get_tool_definitions(dict(active_env))
        active_names = [str(t.get("name") or "") for t in active_tools if isinstance(t, dict)]
        required = {"delegate_start", "delegate_continue", "delegate_apply", "get_diff"}
        missing = sorted(required - set(active_names))
        mcp_ok = not missing
        if missing:
            issues.append(f"Missing core MCP tools: {', '.join(missing)}")
        mcp_info = {
            "ok": mcp_ok,
            "active_tool_count": len(active_names),
            "total_registered_tools": len(TOOL_DEFINITIONS),
            "active_tools": active_names,
        }
    except Exception as exc:
        issues.append(f"MCP tool catalog failed to load: {type(exc).__name__}: {exc}")
        mcp_info = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # 2. Authentication configuration (never expose raw API key or OAuth token values)
    custom_gateway = bool(resolve_vertex_base_url(env=active_env))
    auth_info: dict[str, Any]
    agy_cli_info: dict[str, Any] | None = None
    active_auth_mode: str | None = None
    cli_detected_unsigned = False
    except_agy_cli_info: dict[str, Any] | None = None
    try:
        resolved_auth = resolve_auth_config(
            model=default_delegate_model,
            env=dict(env) if env is not None else None,
        )
        auth_mode = str(resolved_auth.get("auth_mode") or "")
        active_auth_mode = auth_mode
        adc_found: bool | None = None
        ent_cred_source: str | None = None
        auth_ok = True
        if auth_mode == "gemini_enterprise":
            from server.agy_cli import inspect_agy_cli_offline

            ent_check = inspect_agy_cli_offline(active_env)
            ent_cred_source = str(ent_check.get("credential_source") or "") or None
            agy_cli_info = {
                "binary_found": bool(ent_check.get("binary_found")),
                "agy_bin": ent_check.get("agy_bin"),
                "managed_bin_path": ent_check.get("managed_bin_path"),
                "auth_configured": bool(ent_check.get("auth_configured")),
                "credential_source": ent_cred_source,
                "config_dir": ent_check.get("config_dir"),
            }
            if not ent_check.get("ok"):
                auth_ok = False
                issues.append(str(ent_check.get("error") or "Gemini Enterprise `agy` CLI sign-in is missing."))
                if ent_check.get("remediation"):
                    remediations.append(str(ent_check["remediation"]))
        elif auth_mode == "vertex":
            adc_found = any(p.is_file() for p in _default_adc_candidates(active_env))
            if not adc_found and not custom_gateway:
                warnings.append(
                    "Vertex project is configured, but no local Application Default Credentials file was found "
                    "(metadata-server ADC may still work on GCE/Cloudtop)."
                )
                remediations.append("Run `gcloud auth application-default login` or set `GEMINI_API_KEY`.")
        auth_info = {
            "ok": auth_ok,
            "auth_mode": auth_mode,
            "api_key_configured": bool(resolved_auth.get("api_key")),
            "project": resolved_auth.get("project"),
            "location": resolved_auth.get("location"),
            "custom_vertex_gateway": custom_gateway,
            **({"adc_file_found": adc_found} if adc_found is not None else {}),
            **({"enterprise_credential_source": ent_cred_source} if ent_cred_source is not None else {}),
            **({"agy_cli": agy_cli_info} if agy_cli_info is not None else {}),
        }
    except ValueError as exc:
        issues.append(str(exc))
        try:
            from server.agy_cli import inspect_agy_cli_offline
            from server.regions import _should_auto_detect_agy_cli, resolve_configured_auth_mode

            cfg_mode = resolve_configured_auth_mode(env=active_env)
            if cfg_mode in (None, "gemini_enterprise") and _should_auto_detect_agy_cli(env, active_env):
                ent_probe = inspect_agy_cli_offline(active_env)
                if ent_probe.get("binary_found"):
                    cli_detected_unsigned = True
                    except_agy_cli_info = {
                        "ok": False,
                        "binary_found": True,
                        "agy_bin": ent_probe.get("agy_bin"),
                        "auth_configured": False,
                        "credential_source": None,
                        "config_dir": ent_probe.get("config_dir"),
                    }
                    if ent_probe.get("remediation"):
                        remediations.append(str(ent_probe["remediation"]))
        except Exception:
            pass
        if not cli_detected_unsigned:
            remediations.append(
                "Run `python3 -m server.install --auth gemini-enterprise` and sign in with `agy` "
                "(Gemini Enterprise seat), export `GEMINI_API_KEY` (AI Studio), or export `GOOGLE_CLOUD_PROJECT` (Vertex AI)."
            )
        auth_info = {
            "ok": False,
            "auth_mode": None,
            "api_key_configured": False,
            "project": None,
            "location": None,
            "custom_vertex_gateway": custom_gateway,
            "error": str(exc),
            **({"agy_cli": except_agy_cli_info} if except_agy_cli_info is not None else {}),
        }

    # 3. Python SDK availability (check both active interpreter and resolved MCP SDK venv)
    from server.runtime_env import python_has_module, resolve_sdk_python

    sdk_py = resolve_sdk_python(env=active_env)
    has_genai = _module_available("google.genai") or python_has_module(sdk_py, "google.genai")
    has_agy = _module_available("google.antigravity") or python_has_module(sdk_py, "google.antigravity")
    has_ctx = _module_available("ctx.sdk") or python_has_module(sdk_py, "ctx.sdk")
    cli_only_enterprise = (active_auth_mode == "gemini_enterprise") or cli_detected_unsigned
    if not has_genai:
        if cli_only_enterprise:
            warnings.append(
                "Optional Python package `google-genai` is not installed (not required in `gemini_enterprise` CLI mode)."
            )
        else:
            issues.append("Python package `google-genai` is not installed.")
            remediations.append("Install plugin dependencies: `python3 -m server.install`")
    if not has_agy:
        if cli_only_enterprise:
            warnings.append(
                "Optional Python package `google-antigravity` is not installed (not required in `gemini_enterprise` CLI mode)."
            )
        else:
            issues.append("Python package `google-antigravity` is not installed.")
            remediations.append("Install Antigravity SDK: `python3 -m server.install`")
    if not has_ctx and not cli_only_enterprise:
        warnings.append(
            "Optional `ctx-harness` (Straitjacket) package is not installed; worker shell output containment will fall back to raw Antigravity tools."
        )
    sdk_info = {
        "python_bin": sdk_py,
        "google_genai": has_genai,
        "google_antigravity": has_agy,
        "ctx_straitjacket": has_ctx,
    }

    # 4. Workspace git state & multi-language test runner discovery
    ws_path = resolve_workspace_root(workspace_root, env=dict(active_env))
    is_git_repo = False
    tracked_files = 0
    if ws_path.is_dir():
        git_probe = run_git(
            ["rev-parse", "--is-inside-work-tree"],
            cwd=str(ws_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        is_git_repo = git_probe.returncode == 0 and git_probe.stdout.strip().lower() == "true"
        if is_git_repo:
            ls_probe = run_git(
                ["ls-files", "-z"],
                cwd=str(ws_path),
                capture_output=True,
                check=False,
            )
            if ls_probe.returncode == 0:
                tracked_files = sum(1 for entry in ls_probe.stdout.split(b"\x00") if entry)
        else:
            warnings.append(f"Workspace `{ws_path}` is not a git repository; isolated git diff checks require git.")
    detected_runner = detect_project_test_runner(ws_path) if ws_path.is_dir() else None
    workspace_info = {
        "workspace_root": str(ws_path),
        "is_git_repo": is_git_repo,
        "tracked_files": tracked_files,
        "detected_test_runner": detected_runner,
    }

    # 5. Receipt store writability and daily spend budget
    store_writable = False
    data_dir_str = ""
    daily_budget_info: dict[str, Any] = {"max_daily_cents": None}
    try:
        active_store = store or ReceiptStore(workspace_root=ws_path)
        data_dir_str = str(active_store.data_dir)
        active_store.data_dir.mkdir(parents=True, exist_ok=True)
        probe_file = active_store.data_dir / f".doctor_write_probe_{os.getpid()}"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
        store_writable = True
        cap_cents = resolve_max_daily_cents(active_env)
        today = active_store.daily_spend_summary(workspace_root=ws_path)
        daily_budget_info = {
            **today,
            "max_daily_cents": cap_cents,
            "remaining_cents": round(max(0.0, cap_cents - float(today["spent_today_cents"])), 6) if cap_cents is not None else None,
            "exceeded": bool(cap_cents is not None and float(today["spent_today_cents"]) >= cap_cents),
        }
        if daily_budget_info["exceeded"]:
            warnings.append(
                f"Daily Gemini budget reached ({today['spent_today_cents']:.2f}c / {cap_cents:.2f}c on {today['utc_date']} UTC)."
            )
    except ValueError as exc:
        issues.append(str(exc))
    except Exception as exc:
        issues.append(f"Receipt store directory is not writable: {type(exc).__name__}: {exc}")
    receipt_store_info = {
        "data_dir": data_dir_str,
        "writable": store_writable,
    }

    healthy = len(issues) == 0
    status = "ok" if (healthy and not warnings) else ("degraded" if healthy else "error")
    return {
        "status": status,
        "healthy": healthy,
        "cost_profile": cost_profile,
        "default_delegate_model": default_delegate_model,
        "default_ask_model": default_ask_model,
        "strict_mode": strict_mode,
        "mcp": mcp_info,
        "auth": auth_info,
        "sdks": sdk_info,
        "workspace": workspace_info,
        "receipt_store": receipt_store_info,
        "daily_budget": daily_budget_info,
        "issues": issues,
        "warnings": warnings,
        "remediations": list(dict.fromkeys(remediations)),
    }
