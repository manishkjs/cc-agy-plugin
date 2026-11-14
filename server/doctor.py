"""Offline readiness and environment diagnostics for cc-agy-plugin (`doctor`).

Checks MCP tool registration, cost profile and default model routing, authentication
configuration (never exposing raw API keys), required Python SDKs (`google-genai`,
`google.antigravity`, Straitjacket `ctx`), workspace git state, language test-runner
discovery, and receipt store writability without making any network or model calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib.util
import os
from pathlib import Path
import shutil
from typing import Any

from server.install import (
    expand_path_with_env,
    resolve_home_dir,
    resolve_opencode_config_dir,
    resolve_opencode_config_path,
)
from server.jobs import detect_project_test_runner
from server.receipts import ReceiptStore, resolve_workspace_root
from server.verify import run_git
from server.regions import (
    check_promotional_rate_expiry,
    is_in_place_allowed,
    is_strict_mode_enabled,
    read_adc_quota_project,
    resolve_allowed_projects,
    resolve_auth_config,
    resolve_auto_apply_on_pass,
    resolve_auto_apply_override,
    resolve_blocked_projects,
    resolve_bifrost_virtual_key,
    resolve_configured_auth_mode,
    resolve_cost_profile,
    resolve_default_ask_model,
    resolve_default_delegate_model,
    resolve_gateway_base_url,
    resolve_gateway_wire,
    resolve_max_daily_cents,
    resolve_vertex_base_url_audience,
)
from server.telemetry import resolve_telemetry_sink, validate_telemetry_sink


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _default_adc_candidates(env: Mapping[str, str]) -> list[Path]:
    explicit = (env.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    candidates: list[Path] = []
    if explicit and not explicit.startswith("${"):
        candidates.append(expand_path_with_env(explicit, env))
    gcloud_cfg = (env.get("CLOUDSDK_CONFIG") or "").strip()
    if gcloud_cfg and not gcloud_cfg.startswith("${"):
        candidates.append(expand_path_with_env(gcloud_cfg, env) / "application_default_credentials.json")
    home = resolve_home_dir(env)
    candidates.append(home / ".config" / "gcloud" / "application_default_credentials.json")
    appdata = (env.get("APPDATA") or "").strip()
    if appdata:
        candidates.append(Path(appdata) / "gcloud" / "application_default_credentials.json")
    return candidates


def _discover_installed_host_mcp_env(base_env: Mapping[str, str]) -> dict[str, str]:
    """Extract non-secret `CC_AGY_*` / project settings from installed host MCP configs when shell env is unset."""
    import json
    from server.install import _loads_jsonc, _parse_toml

    home_dir = resolve_home_dir(base_env)
    opencode_cfg = resolve_opencode_config_path(resolve_opencode_config_dir(base_env), env=base_env)
    cursor_cfg = expand_path_with_env(base_env.get("CURSOR_CONFIG_DIR") or (home_dir / ".cursor"), base_env) / "mcp.json"
    agy_cfg = expand_path_with_env(base_env.get("AGY_CONFIG_DIR") or (home_dir / ".gemini" / "config"), base_env) / "mcp_config.json"
    codex_cfg = expand_path_with_env(base_env.get("CODEX_HOME") or (home_dir / ".codex"), base_env) / "config.toml"

    candidates: list[dict[str, Any]] = []
    if opencode_cfg.is_file():
        try:
            oc_data = _loads_jsonc(opencode_cfg.read_text(encoding="utf-8"))
            if isinstance(oc_data, dict):
                mcp_sec = oc_data.get("mcp")
                if isinstance(mcp_sec, dict):
                    servers_map = mcp_sec.get("servers") if isinstance(mcp_sec.get("servers"), dict) else mcp_sec
                    for srv in servers_map.values():
                        if isinstance(srv, dict) and isinstance(srv.get("environment"), dict):
                            cmd_str = " ".join(str(x) for x in (srv.get("command") or []) if x)
                            if "server/main.py" in cmd_str or "cc-agy-plugin" in cmd_str or "CC_AGY_" in str(srv["environment"]):
                                candidates.append(srv["environment"])
        except Exception:
            pass
    for json_cfg in (cursor_cfg, agy_cfg):
        if json_cfg.is_file():
            try:
                j_data = json.loads(json_cfg.read_text(encoding="utf-8"))
                mcp_srvs = j_data.get("mcpServers") if isinstance(j_data, dict) else None
                if isinstance(mcp_srvs, dict):
                    for srv in mcp_srvs.values():
                        if isinstance(srv, dict) and isinstance(srv.get("env"), dict):
                            args_str = " ".join(str(x) for x in (srv.get("args") or []) if x)
                            if "server/main.py" in args_str or "cc-agy-plugin" in args_str or "CC_AGY_" in str(srv["env"]):
                                candidates.append(srv["env"])
            except Exception:
                pass
    if codex_cfg.is_file():
        try:
            t_data = _parse_toml(codex_cfg.read_text(encoding="utf-8"))
            mcp_srvs = t_data.get("mcp_servers") if isinstance(t_data, dict) else None
            if isinstance(mcp_srvs, dict):
                for srv in mcp_srvs.values():
                    if isinstance(srv, dict) and isinstance(srv.get("env"), dict):
                        args_str = " ".join(str(x) for x in (srv.get("args") or []) if x)
                        if "server/main.py" in args_str or "cc-agy-plugin" in args_str or "CC_AGY_" in str(srv["env"]):
                            candidates.append(srv["env"])
        except Exception:
            pass

    allowed_prefixes = ("CC_AGY_",)
    allowed_exact = {"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"}
    secret_substrings = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    merged: dict[str, str] = {}
    for env_dict in candidates:
        for k, v in env_dict.items():
            if not isinstance(k, str) or not isinstance(v, (str, int, float, bool)):
                continue
            val_s = str(v).strip()
            if not val_s or val_s.startswith("${"):
                continue
            if any(sec in k.upper() for sec in secret_substrings):
                continue
            if k.startswith(allowed_prefixes) or k in allowed_exact:
                merged.setdefault(k, val_s)
        if merged.get("CC_AGY_AUTH_MODE"):
            break
    return merged


def run_doctor(
    workspace_root: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    store: ReceiptStore | None = None,
    probe_oidc: bool = False,
    allow_pending_login: bool = False,
) -> dict[str, Any]:
    """Run local zero-network readiness checks and return a structured report."""
    raw_env: Mapping[str, str] = os.environ if env is None else env
    used_host_cfg_env = False
    has_isolated_cfg_override = bool(
        env is not None
        and any(
            k in env
            for k in (
                "HOME",
                "USERPROFILE",
                "XDG_CONFIG_HOME",
                "OPENCODE_CONFIG_DIR",
                "OPENCODE_CONFIG",
                "CURSOR_CONFIG_DIR",
                "AGY_CONFIG_DIR",
                "CODEX_HOME",
            )
        )
    )
    has_explicit_non_ge_creds = any(
        (raw_env.get(k) or "").strip()
        for k in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "CC_AGY_VERTEX_BASE_URL",
            "CC_AGY_GEMINI_BASE_URL",
            "GOOGLE_APPLICATION_CREDENTIALS",
        )
    )
    in_unisolated_pytest = bool(
        os.environ.get("PYTEST_CURRENT_TEST")
        and not has_isolated_cfg_override
        and not allow_pending_login
    )
    raw_auth_val = (raw_env.get("CC_AGY_AUTH_MODE") or "").strip().lower()
    if (
        (env is None or has_isolated_cfg_override or allow_pending_login)
        and raw_auth_val in ("", "auto")
        and not in_unisolated_pytest
        and (allow_pending_login or not has_explicit_non_ge_creds)
    ):
        discovered_cfg_env = _discover_installed_host_mcp_env(raw_env)
        if discovered_cfg_env:
            combined = dict(discovered_cfg_env)
            combined.update({k: v for k, v in raw_env.items() if v is not None})
            if discovered_cfg_env.get("CC_AGY_AUTH_MODE") and raw_auth_val in ("", "auto"):
                combined["CC_AGY_AUTH_MODE"] = discovered_cfg_env["CC_AGY_AUTH_MODE"]
            active_env: Mapping[str, str] = combined
            used_host_cfg_env = True
        else:
            active_env = raw_env
    else:
        active_env = raw_env
    issues: list[str] = []
    warnings: list[str] = []
    remediations: list[str] = []

    cost_profile = resolve_cost_profile(active_env)
    default_delegate_model = resolve_default_delegate_model(None, env=active_env)
    default_ask_model = resolve_default_ask_model(None, env=active_env)
    strict_mode = is_strict_mode_enabled(active_env)
    allowed_projects = list(resolve_allowed_projects(active_env))
    blocked_projects = list(resolve_blocked_projects(active_env))
    auto_apply_override = resolve_auto_apply_override(active_env)
    auto_apply_on_pass = resolve_auto_apply_on_pass(active_env)
    in_place_allowed = is_in_place_allowed(active_env)
    configured_auth_mode: str | None = None
    try:
        configured_auth_mode = resolve_configured_auth_mode(env=active_env)
    except ValueError:
        configured_auth_mode = None

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
        home_dir = resolve_home_dir(active_env)
        codex_cfg = (
            expand_path_with_env(active_env.get("CODEX_HOME") or (home_dir / ".codex"), active_env)
            / "config.toml"
        )
        cursor_cfg = (
            expand_path_with_env(active_env.get("CURSOR_CONFIG_DIR") or (home_dir / ".cursor"), active_env)
            / "mcp.json"
        )
        opencode_cfg = resolve_opencode_config_path(
            resolve_opencode_config_dir(active_env),
            env=active_env,
        )
        if (active_env.get("OPENCODE_CONFIG") or "").strip() and not opencode_cfg.is_absolute():
            warnings.append(
                f"OPENCODE_CONFIG ({opencode_cfg}) is a relative path; use an absolute path so OpenCode and server.install resolve the same file."
            )
        agy_cfg = (
            expand_path_with_env(
                active_env.get("AGY_CONFIG_DIR") or (home_dir / ".gemini" / "config"),
                active_env,
            )
            / "mcp_config.json"
        )
        mcp_info = {
            "ok": mcp_ok,
            "active_tool_count": len(active_names),
            "total_registered_tools": len(TOOL_DEFINITIONS),
            "active_tools": active_names,
            "host_configs": {
                "codex": {"path": str(codex_cfg), "exists": codex_cfg.is_file()},
                "cursor": {"path": str(cursor_cfg), "exists": cursor_cfg.is_file()},
                "opencode": {"path": str(opencode_cfg), "exists": opencode_cfg.is_file()},
                "agy": {"path": str(agy_cfg), "exists": agy_cfg.is_file()},
            },
        }
    except Exception as exc:
        issues.append(f"MCP tool catalog failed to load: {type(exc).__name__}: {exc}")
        mcp_info = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # 2. Authentication configuration (never expose raw API key or OAuth token values)
    env_gateway_url = resolve_gateway_base_url(env=active_env)
    resolved_gateway_url = env_gateway_url
    custom_gateway = bool(resolved_gateway_url)
    gateway_wire: str | None = None
    try:
        resolved_wire = resolve_gateway_wire(base_url=resolved_gateway_url, env=active_env)
        if custom_gateway:
            gateway_wire = resolved_wire
    except ValueError as exc:
        issues.append(str(exc))
    gateway_oidc_audience: str | None = None
    oidc_config_ok = True
    try:
        gateway_oidc_audience = resolve_vertex_base_url_audience(
            base_url=resolved_gateway_url,
            env=active_env,
        )
    except ValueError as exc:
        oidc_config_ok = False
        issues.append(str(exc))
    auth_info: dict[str, Any]
    agy_cli_info: dict[str, Any] | None = None
    active_auth_mode: str | None = None
    cli_detected_unsigned = False
    except_agy_cli_info: dict[str, Any] | None = None
    try:
        resolved_auth = resolve_auth_config(
            model=default_delegate_model,
            env=dict(active_env) if (env is not None or used_host_cfg_env) else None,
        )
        auth_mode = str(resolved_auth.get("auth_mode") or "")
        active_auth_mode = auth_mode
        eff_gw_from_auth = resolved_auth.get("base_url")
        custom_gateway = bool(eff_gw_from_auth)
        resolved_gateway_url = str(eff_gw_from_auth) if eff_gw_from_auth else None
        gateway_wire = str(resolved_auth.get("gateway_wire") or "") or None if custom_gateway else None
        adc_found: bool | None = None
        adc_quota_project: str | None = None
        ent_cred_source: str | None = None
        active_project_id: str | None = resolved_auth.get("project")
        auth_ok = True
        if env_gateway_url and not custom_gateway and configured_auth_mode == "api_key":
            from server.regions import _clean_base_url_candidate, resolve_vertex_base_url

            has_dedicated_plugin_gateway = bool(
                resolve_vertex_base_url(env=active_env)
                or _clean_base_url_candidate(active_env.get("CC_AGY_GEMINI_BASE_URL"))
            )
            msg = (
                "Explicit `CC_AGY_AUTH_MODE=api_key` (`--auth api-key`) always connects directly to Google AI Studio "
                f"and ignores configured gateway URL '{env_gateway_url}'. Use `CC_AGY_AUTH_MODE=auto` or `gateway` "
                "to route through the gateway."
            )
            if has_dedicated_plugin_gateway:
                issues.append(msg)
            else:
                warnings.append(msg)
        elif active_project_id == "custom-endpoint" and not custom_gateway:
            issues.append(
                "Project resolved to 'custom-endpoint' without an active gateway base URL; "
                "set `CC_AGY_VERTEX_BASE_URL` or `CC_AGY_GEMINI_BASE_URL`."
            )
        if auth_mode == "gemini_enterprise":
            from server.agy_cli import inspect_agy_cli_offline

            ent_check = inspect_agy_cli_offline(active_env)
            detected_agy_ver = ent_check.get("version")
            if (
                detected_agy_ver is None
                and ent_check.get("agy_bin")
                and (probe_oidc or allow_pending_login or not os.environ.get("PYTEST_CURRENT_TEST"))
            ):
                from server.agy_cli import probe_agy_cli_version

                detected_agy_ver = probe_agy_cli_version(ent_check.get("agy_bin"), env=active_env)
            ent_cred_source = str(ent_check.get("credential_source") or "") or None
            comp_proj = ent_check.get("companion_project")
            if isinstance(comp_proj, str) and comp_proj.strip() and not active_project_id:
                active_project_id = comp_proj.strip()
            from server.regions import is_ge_mode_pinned

            ge_pinned = is_ge_mode_pinned(active_env)
            claude_esc_off = str(active_env.get("CC_AGY_ALLOW_CLAUDE_IN_GE") or "").strip().lower() not in (
                "1",
                "true",
                "yes",
            )
            agy_cli_info = {
                "binary_found": bool(ent_check.get("binary_found")),
                "agy_bin": ent_check.get("agy_bin"),
                "version": detected_agy_ver,
                "pinned_version": ent_check.get("pinned_version"),
                "managed_bin_path": ent_check.get("managed_bin_path"),
                "auth_configured": bool(ent_check.get("auth_configured")),
                "consumer_account_detected": bool(ent_check.get("consumer_account_detected")),
                "tier_verified_enterprise": bool(ent_check.get("tier_verified_enterprise")),
                "credential_source": ent_cred_source,
                "companion_project": comp_proj,
                "tier_id": ent_check.get("tier_id"),
                "config_dir": ent_check.get("config_dir"),
                "adc_seat_enabled": bool(ent_check.get("adc_seat_enabled")),
                "enterprise_project": ent_check.get("enterprise_project"),
                "enterprise_location": ent_check.get("enterprise_location"),
                "baicode_user_tier": ent_check.get("baicode_user_tier"),
                "ge_billing_lock": {
                    "pinned": ge_pinned,
                    "api_key_override_blocked": ge_pinned,
                    "claude_escalation_disabled": claude_esc_off,
                },
            }
            if ent_check.get("adc_seat_error"):
                agy_cli_info["adc_seat_error"] = ent_check["adc_seat_error"]
            from server.worker_sandbox import inspect_worker_sandbox

            agy_cli_info["worker_sandbox"] = inspect_worker_sandbox(active_env)
            if agy_cli_info["worker_sandbox"]["error"]:
                issues.append(str(agy_cli_info["worker_sandbox"]["error"]))
            from server.bifrost_policy import inspect_bifrost_policy

            bifrost_agy_policy = inspect_bifrost_policy(active_env, live=probe_oidc)
            if bifrost_agy_policy["enabled"] or bifrost_agy_policy["error"]:
                agy_cli_info["bifrost_agy_policy"] = bifrost_agy_policy
            if bifrost_agy_policy["error"]:
                issues.append(f"Bifrost /agy policy: {bifrost_agy_policy['error']}")
            from server.agy_state import inspect_agy_model_catalog

            model_catalog = inspect_agy_model_catalog(ent_check.get("agy_bin"), active_env, live=probe_oidc)
            if model_catalog["model_count"] is not None or model_catalog["source"]:
                agy_cli_info["model_catalog"] = model_catalog
            if probe_oidc and model_catalog["models"] and bifrost_agy_policy["enabled"] and not bifrost_agy_policy["error"]:
                from server.agy_cli import _filter_models_by_bifrost_policy

                model_catalog["allowed_by_bifrost_count"] = len(
                    _filter_models_by_bifrost_policy(list(model_catalog["models"]), active_env)
                )
            if not ent_check.get("ok"):
                only_pending_signin = bool(
                    allow_pending_login
                    and ent_check.get("binary_found")
                    and not ent_check.get("auth_configured")
                    and not ent_check.get("consumer_account_detected")
                    and not ent_check.get("adc_seat_error")
                )
                if only_pending_signin:
                    agy_cli_info["pending_login"] = True
                    warnings.append(
                        str(ent_check.get("error") or "One-time Antigravity CLI sign-in is still pending.")
                    )
                else:
                    auth_ok = False
                    issues.append(str(ent_check.get("error") or "Gemini Enterprise `agy` CLI sign-in is missing."))
                if ent_check.get("remediation"):
                    remediations.append(str(ent_check["remediation"]))
            elif not ent_check.get("tier_verified_enterprise"):
                warnings.append(
                    "Antigravity CLI sign-in is present, but `cache/onboarding.json` does not yet record an explicit "
                    "`gcp-ge-*` enterprise tier. Verify you signed in with your licensed Gemini Enterprise work account "
                    "(or set `CC_AGY_REQUIRE_GE_TIER=1` to enforce strict tier verification)."
                )
        elif auth_mode == "vertex":
            adc_found = any(p.is_file() for p in _default_adc_candidates(active_env))
            adc_quota_project = read_adc_quota_project(active_env, explicit_env=env)
            if configured_auth_mode is None and not custom_gateway:
                warnings.append(
                    f"Auth mode is 'auto' and selected pay-as-you-go Vertex AI project "
                    f"'{resolved_auth.get('project')}' from your shell environment. Pass `--auth vertex` to "
                    "confirm or `--auth gemini-enterprise` / `--auth api-key` to pin another backend."
                )
            if not adc_found and not custom_gateway:
                warnings.append(
                    "Vertex project is configured, but no local Application Default Credentials file was found "
                    "(metadata-server ADC may still work on GCE/Cloudtop)."
                )
                remediations.append("Run `gcloud auth application-default login` or set `GEMINI_API_KEY`.")
        gateway_oidc_principal: str | None = None
        gateway_oidc_principal_status: str | None = None
        gateway_oidc_source: str | None = None
        gateway_credential_source: str | None = None
        if custom_gateway:
            from server.regions import (
                _resolve_oidc_source,
                peek_cached_oidc_token,
                resolve_static_vertex_base_url_token,
                resolve_vertex_base_url_token,
            )
            from server.telemetry import _extract_oidc_jwt_claim

            try:
                gateway_oidc_source = _resolve_oidc_source(active_env)
            except ValueError as exc:
                oidc_config_ok = False
                issues.append(str(exc))
            static_tok = resolve_static_vertex_base_url_token(None, env=active_env) or ""
            bf_vk_tok = resolve_bifrost_virtual_key(None, env=active_env) or ""
            cached_tok = (
                peek_cached_oidc_token(
                    env=active_env,
                    audience=gateway_oidc_audience,
                    base_url=resolved_gateway_url,
                )
                or ""
            )
            if static_tok:
                gateway_oidc_principal = _extract_oidc_jwt_claim(static_tok)
                gateway_oidc_principal_status = "static_token"
            elif cached_tok:
                gateway_oidc_principal = _extract_oidc_jwt_claim(cached_tok)
                gateway_oidc_principal_status = "cached"
            elif gateway_oidc_audience:
                if probe_oidc:
                    if not oidc_config_ok:
                        gateway_oidc_principal_status = "probe_skipped"
                    else:
                        try:
                            minted_tok = resolve_vertex_base_url_token(
                                None,
                                base_url=resolved_gateway_url,
                                requested_audience=gateway_oidc_audience,
                                env=active_env,
                            )
                            if minted_tok:
                                gateway_oidc_principal = _extract_oidc_jwt_claim(minted_tok)
                                gateway_oidc_principal_status = "probed"
                        except Exception as exc:  # noqa: BLE001
                            issues.append(f"OIDC probe failed: {exc}")
                            gateway_oidc_principal_status = "probe_failed"
                else:
                    gateway_oidc_principal_status = "unminted_until_first_request"
            elif bf_vk_tok:
                gateway_oidc_principal_status = "static_token"
            if static_tok:
                gateway_credential_source = "vertex_base_url_token"
            elif bf_vk_tok and gateway_oidc_audience:
                gateway_credential_source = "oidc+bifrost_virtual_key"
            elif bf_vk_tok:
                gateway_credential_source = "bifrost_virtual_key"
            elif gateway_oidc_audience and resolved_auth.get("api_key"):
                gateway_credential_source = "oidc+gemini_api_key"
            elif gateway_oidc_audience:
                gateway_credential_source = "oidc"
            elif resolved_auth.get("api_key"):
                gateway_credential_source = "gemini_api_key"
            else:
                gateway_credential_source = "none"
        auth_info = {
            "ok": auth_ok and not issues,
            "auth_mode": auth_mode,
            "configured_auth_mode": configured_auth_mode or "auto",
            "auth_mode_pinned": configured_auth_mode is not None,
            "api_key_configured": bool(resolved_auth.get("api_key")),
            "project": active_project_id,
            "location": resolved_auth.get("location"),
            "custom_vertex_gateway": custom_gateway,
            **({"gateway_wire": gateway_wire} if custom_gateway and gateway_wire else {}),
            **(
                {"gateway_credential_source": gateway_credential_source}
                if custom_gateway and gateway_credential_source
                else {}
            ),
            **(
                {"bifrost_virtual_key_configured": bool(resolve_bifrost_virtual_key(env=active_env))}
                if custom_gateway
                else {}
            ),
            **({"gateway_oidc_audience": gateway_oidc_audience} if custom_gateway and gateway_oidc_audience else {}),
            **({"gateway_oidc_source": gateway_oidc_source} if gateway_oidc_source else {}),
            **({"gateway_oidc_principal": gateway_oidc_principal} if gateway_oidc_principal else {}),
            **(
                {"gateway_oidc_principal_status": gateway_oidc_principal_status}
                if gateway_oidc_principal_status
                else {}
            ),
            **({"adc_file_found": adc_found} if adc_found is not None else {}),
            **({"adc_quota_project": adc_quota_project} if adc_quota_project is not None else {}),
            **({"enterprise_credential_source": ent_cred_source} if ent_cred_source is not None else {}),
            **({"agy_cli": agy_cli_info} if agy_cli_info is not None else {}),
        }
    except ValueError as exc:
        issues.append(str(exc))
        try:
            from server.agy_cli import inspect_agy_cli_offline
            from server.regions import _should_auto_detect_agy_cli

            if configured_auth_mode in (None, "gemini_enterprise") and _should_auto_detect_agy_cli(env, active_env):
                ent_probe = inspect_agy_cli_offline(active_env)
                if ent_probe.get("binary_found"):
                    cli_detected_unsigned = True
                    except_agy_cli_info = {
                        "ok": False,
                        "binary_found": True,
                        "agy_bin": ent_probe.get("agy_bin"),
                        "auth_configured": False,
                        "credential_source": None,
                        "companion_project": ent_probe.get("companion_project"),
                        "tier_id": ent_probe.get("tier_id"),
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
    if not Path("/proc").is_dir() and shutil.which("pgrep") is None and shutil.which("ps") is None:
        warnings.append(
            "Process-tree cleanup fallback requires `pgrep` or `ps` on non-/proc platforms."
        )

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

    telemetry_sink = resolve_telemetry_sink(active_env)
    telemetry_sink_valid = True
    telemetry_sink_warning: str | None = None
    if telemetry_sink:
        try:
            validate_telemetry_sink(telemetry_sink, check_dns=False)
        except ValueError as exc:
            telemetry_sink_valid = False
            telemetry_sink_warning = str(exc)
            issues.append(str(exc))
    rate_card_notices = check_promotional_rate_expiry()

    user_controls: dict[str, Any] = {
        "configured_auth_mode": configured_auth_mode or "auto",
        "auth_mode_pinned": configured_auth_mode is not None,
        "active_auth_mode": active_auth_mode,
        "active_project": auth_info.get("project"),
        "allowed_projects": allowed_projects,
        "blocked_projects": blocked_projects,
        "max_daily_cents": daily_budget_info.get("max_daily_cents"),
        "cost_profile": cost_profile,
        "default_delegate_model": default_delegate_model,
        "auto_apply_on_pass": auto_apply_on_pass,
        "manual_apply_enforced": auto_apply_override is False,
        "in_place_allowed": in_place_allowed,
        "isolated_worktree_default": True,
        "strict_mode": strict_mode,
        "telemetry_sink_enabled": bool(telemetry_sink),
        "telemetry_sink_valid": telemetry_sink_valid,
    }
    if telemetry_sink_warning is not None:
        user_controls["telemetry_sink_warning"] = telemetry_sink_warning
    healthy = len(issues) == 0
    status = "ok" if (healthy and not warnings) else ("degraded" if healthy else "error")
    return {
        "status": status,
        "healthy": healthy,
        "cost_profile": cost_profile,
        "default_delegate_model": default_delegate_model,
        "default_ask_model": default_ask_model,
        "strict_mode": strict_mode,
        "user_controls": user_controls,
        "mcp": mcp_info,
        "auth": auth_info,
        "sdks": sdk_info,
        "workspace": workspace_info,
        "receipt_store": receipt_store_info,
        "daily_budget": daily_budget_info,
        "rate_card_notices": rate_card_notices,
        "issues": issues,
        "warnings": warnings,
        "remediations": list(dict.fromkeys(remediations)),
    }


def format_doctor_human(report: Mapping[str, Any]) -> str:
    """Format a `run_doctor()` report as a concise human-readable terminal summary."""
    status = str(report.get("status") or "error").upper()
    healthy = bool(report.get("healthy"))
    badge = "[OK]" if status == "OK" else ("[WARN]" if healthy else "[FAIL]")
    auth = report.get("auth") if isinstance(report.get("auth"), Mapping) else {}
    mcp = report.get("mcp") if isinstance(report.get("mcp"), Mapping) else {}
    sdks = report.get("sdks") if isinstance(report.get("sdks"), Mapping) else {}
    ws = report.get("workspace") if isinstance(report.get("workspace"), Mapping) else {}
    store = report.get("receipt_store") if isinstance(report.get("receipt_store"), Mapping) else {}
    budget = report.get("daily_budget") if isinstance(report.get("daily_budget"), Mapping) else {}

    raw_runner = ws.get("detected_test_runner")
    if isinstance(raw_runner, Mapping):
        runner_display = str(raw_runner.get("primary_command") or raw_runner.get("command") or "smoke_only")
    else:
        runner_display = str(raw_runner or "smoke_only")

    lines = [
        f"{badge} cc-agy-plugin doctor ({status.lower()})",
        f"  Auth mode:      {auth.get('auth_mode') or 'unconfigured'} (configured: {auth.get('configured_auth_mode') or 'auto'})",
        f"  Cost profile:   {report.get('cost_profile')} (worker: {report.get('default_delegate_model')}, ask: {report.get('default_ask_model')})",
        f"  MCP tools:      {'ok' if mcp.get('ok') else 'error'} ({mcp.get('active_tool_count', 0)} active)",
        f"  Python runtime: {sdks.get('python_bin')} (genai={sdks.get('google_genai')}, agy={sdks.get('google_antigravity')}, ctx={sdks.get('ctx_straitjacket')})",
        f"  Workspace:      {ws.get('workspace_root')} (git={ws.get('is_git_repo')}, runner={runner_display})",
        f"  Receipt store:  {store.get('data_dir')} (writable={store.get('writable')})",
    ]
    agy_cli = auth.get("agy_cli") if isinstance(auth.get("agy_cli"), Mapping) else None
    if agy_cli:
        sbx = agy_cli.get("worker_sandbox") if isinstance(agy_cli.get("worker_sandbox"), Mapping) else {}
        ge_lock = agy_cli.get("ge_billing_lock") if isinstance(agy_cli.get("ge_billing_lock"), Mapping) else {}
        ver_str = agy_cli.get("version") or "unknown"
        if agy_cli.get("pinned_version"):
            ver_str = f"{ver_str} (pinned={agy_cli['pinned_version']})"
        tier_str = f", tier={agy_cli['tier_id']}" if agy_cli.get("tier_id") else ""
        lines.append(
            f"  AGY CLI:        bin={agy_cli.get('agy_bin') or 'missing'}, version={ver_str}, "
            f"signed_in={agy_cli.get('auth_configured')}{tier_str}, "
            f"sandbox={sbx.get('backend') or 'agy --sandbox'} (mode={sbx.get('mode') or 'auto'})"
        )
        lines.append(
            f"  GE billing:     fail_closed={bool(ge_lock.get('api_key_override_blocked'))}, "
            f"claude_escalation={'disabled' if ge_lock.get('claude_escalation_disabled') else 'enabled'}"
        )
    if budget.get("max_daily_cents") is not None:
        lines.append(
            f"  Daily budget:   {float(budget.get('spent_today_cents') or 0.0):.2f}c / {float(budget['max_daily_cents']):.2f}c"
        )
    for issue in report.get("issues") or []:
        lines.append(f"  [FAIL] {issue}")
    for warn in report.get("warnings") or []:
        lines.append(f"  [WARN] {warn}")
    for rem in report.get("remediations") or []:
        lines.append(f"  [FIX]  {rem}")
    return "\n".join(lines)


def _apply_doctor_fixes(env: Mapping[str, str]) -> list[str]:
    """Run safe local bootstrap repairs when `doctor --fix` is requested."""
    actions: list[str] = []
    plugin_root = Path(__file__).resolve().parent.parent
    try:
        cfg_mode = resolve_configured_auth_mode(env=env)
    except ValueError:
        cfg_mode = None
    if cfg_mode == "gemini_enterprise":
        from server.agy_cli import ensure_agy_cli_installed, inspect_agy_cli_offline

        status = inspect_agy_cli_offline(env)
        if not status.get("binary_found"):
            boot = ensure_agy_cli_installed(plugin_root, env=env, output_fn=None)
            actions.append(f"ensure_agy_cli_installed: {'ok' if boot.get('ok') else boot.get('error')}")
    else:
        from server.runtime_env import ensure_sdk_venv, python_has_plugin_sdks, resolve_sdk_python

        py_bin = resolve_sdk_python(plugin_root, env=env)
        if not python_has_plugin_sdks(py_bin):
            boot = ensure_sdk_venv(plugin_root, env=env, output_fn=None)
            actions.append(f"ensure_sdk_venv: {'ok' if boot.get('ok') else boot.get('error')}")
    return actions


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run cc-agy-plugin environment and gateway diagnostics.")
    parser.add_argument(
        "--probe-oidc",
        action="store_true",
        help="Mint a live OIDC identity token once when gateway OIDC is configured and display the principal claim.",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Print a concise human-readable diagnostic summary instead of JSON.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt safe local bootstrap repairs (managed SDK venv or standalone `agy` CLI) before checking.",
    )
    args = parser.parse_args(argv)
    fix_actions: list[str] = []
    if args.fix:
        fix_actions = _apply_doctor_fixes(os.environ)
    report = run_doctor(workspace_root=os.getcwd(), probe_oidc=bool(args.probe_oidc))
    if fix_actions:
        report["fix_actions"] = fix_actions
    if args.human:
        print(format_doctor_human(report))
    else:
        print(json.dumps(report, indent=2))
    return 0 if report.get("healthy") else 1


if __name__ == "__main__":
    raise SystemExit(main())
