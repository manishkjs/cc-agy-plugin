"""Shared `google-genai` client factory, session cleanup, and token-usage normalizer."""

from __future__ import annotations

from typing import Any, Mapping
import urllib.parse

from server.regions import (
    resolve_effective_gateway_url,
    resolve_gateway_credentials,
    resolve_gateway_wire,
)

USAGE_TOKEN_KEYS: tuple[str, ...] = (
    "prompt_token_count",
    "cached_content_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)
_GATEWAY_URL_API_VERSIONS: frozenset[str] = frozenset({"v1", "v1alpha", "v1beta", "v1beta1"})


def _split_gateway_base_and_api_version(
    base_url: str,
    *,
    default_api_version: str,
) -> tuple[str, str]:
    """Strip an explicit trailing `/v1`, `/v1alpha`, `/v1beta`, or `/v1beta1` segment into `api_version`.

    `google-genai` 3.0+ changed its `vertexai=False` default `api_version` from `'v1beta'` to `'v1'`, and
    always appends `/{api_version}/...` onto `HttpOptions.base_url`. Returning `(clean_base_url, api_version)`
    keeps Gemini-wire calls on `'v1beta'` by default while honoring case-insensitive `/v1`, `/v1alpha`,
    `/v1beta`, or `/v1beta1` gateway path suffixes without doubling the version segment (leaving `/v1internal` untouched).
    """
    parsed = urllib.parse.urlsplit(base_url)
    segments = [seg for seg in parsed.path.split("/") if seg]
    if segments and segments[-1].lower() in _GATEWAY_URL_API_VERSIONS:
        api_ver = segments[-1].lower()
        prefix_segs = segments[:-1]
        new_path = ("/" + "/".join(prefix_segs)) if prefix_segs else ""
        clean_base = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, new_path, parsed.query, parsed.fragment)
        )
        return clean_base, api_ver
    return base_url, default_api_version


def build_genai_client(
    auth: Mapping[str, Any],
    *,
    timeout_ms: int | None = None,
    zero_thinking: bool = False,
) -> Any:
    """Construct a `google.genai.Client` honoring API keys, Vertex ADC, and AI gateways."""
    from google import genai
    from google.genai import types as genai_types
    from google.oauth2.credentials import Credentials as OAuth2Credentials

    base_url = resolve_effective_gateway_url(auth)
    if not base_url and auth.get("api_key"):
        direct_opts: dict[str, Any] = {
            "base_url": "https://generativelanguage.googleapis.com/",
            "api_version": "v1beta",
        }
        if timeout_ms:
            direct_opts["timeout"] = timeout_ms
        return genai.Client(
            vertexai=False,
            api_key=auth["api_key"],
            http_options=genai_types.HttpOptions(**direct_opts),
        )

    if base_url:
        gw_wire = resolve_gateway_wire(auth.get("gateway_wire"), base_url=base_url)
        token, headers = resolve_gateway_credentials(
            auth,
            base_url=base_url,
            wire=gw_wire,
            zero_thinking=zero_thinking,
        )
        default_ver = "v1beta" if gw_wire == "gemini" else "v1beta1"
        clean_base_url, api_ver = _split_gateway_base_and_api_version(
            base_url,
            default_api_version=default_ver,
        )
        opt_kwargs: dict[str, Any] = {
            "base_url": clean_base_url,
            "api_version": api_ver,
            "headers": headers,
        }
        host_low = (urllib.parse.urlsplit(clean_base_url).hostname or "").lower()
        is_custom_gateway = clean_base_url.startswith("http://") or not (
            host_low == "googleapis.com" or host_low.endswith(".googleapis.com")
        )
        if is_custom_gateway:
            opt_kwargs["client_args"] = {"follow_redirects": False}
        if timeout_ms:
            opt_kwargs["timeout"] = timeout_ms
        if gw_wire == "gemini":
            return genai.Client(
                vertexai=False,
                api_key=token,
                http_options=genai_types.HttpOptions(**opt_kwargs),
            )
        client = genai.Client(
            vertexai=True,
            project=auth.get("project") or "custom-endpoint",
            location=auth.get("location") or "global",
            credentials=OAuth2Credentials(token=token),
            http_options=genai_types.HttpOptions(**opt_kwargs),
        )
        if is_custom_gateway and getattr(client, "_api_client", None) is not None:
            client._api_client._use_google_auth_async = lambda: False  # type: ignore[method-assign]
        return client

    http_opts = genai_types.HttpOptions(timeout=timeout_ms) if timeout_ms else None
    return genai.Client(
        vertexai=True,
        project=auth.get("project"),
        location=auth.get("location"),
        **({"http_options": http_opts} if http_opts else {}),
    )


def close_genai_client(client: Any) -> None:
    """Close underlying httpx/requests sessions on a `google-genai` Client."""
    if client is None:
        return
    api_client = getattr(client, "_api_client", None)
    for owner in (
        getattr(api_client, "_httpx_client", None),
        getattr(api_client, "_authorized_session", None),
        getattr(getattr(api_client, "_auth", None), "session", None),
        client,
        api_client,
    ):
        if owner is None:
            continue
        close_fn = getattr(owner, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:  # noqa: BLE001
                pass


def extract_usage_metadata(raw: Any) -> dict[str, int]:
    """Normalize a `usage_metadata` model, dict, or attribute bag into standard token counters."""
    usage = {k: 0 for k in USAGE_TOKEN_KEYS}
    if raw is None:
        return usage
    if not any(hasattr(raw, k) for k in USAGE_TOKEN_KEYS) and not (
        isinstance(raw, Mapping) and any(k in raw for k in USAGE_TOKEN_KEYS)
    ):
        nested = (
            raw.get("usage_metadata") or raw.get("usage")
            if isinstance(raw, Mapping)
            else getattr(raw, "usage_metadata", None) or getattr(raw, "usage", None)
        )
        if nested is not None:
            raw = nested
    data = (
        raw.model_dump()
        if hasattr(raw, "model_dump")
        else (dict(raw) if isinstance(raw, Mapping) else {})
    )
    if not data:
        data = {k: getattr(raw, k, 0) for k in USAGE_TOKEN_KEYS}
    for key in USAGE_TOKEN_KEYS:
        try:
            usage[key] = int(data.get(key) or 0)
        except (TypeError, ValueError):
            usage[key] = 0
    if usage["total_token_count"] == 0:
        usage["total_token_count"] = (
            usage["prompt_token_count"]
            + usage["candidates_token_count"]
            + usage["thoughts_token_count"]
        )
    return usage
