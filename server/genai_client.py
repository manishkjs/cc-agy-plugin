"""Shared `google-genai` client factory, session cleanup, and token-usage normalizer."""

from __future__ import annotations

from typing import Any, Mapping

from server.regions import resolve_vertex_base_url, resolve_vertex_base_url_token

USAGE_TOKEN_KEYS: tuple[str, ...] = (
    "prompt_token_count",
    "cached_content_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)


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

    if auth.get("api_key"):
        http_opts = genai_types.HttpOptions(timeout=timeout_ms) if timeout_ms else None
        return genai.Client(
            api_key=auth["api_key"],
            **({"http_options": http_opts} if http_opts else {}),
        )

    base_url = resolve_vertex_base_url(auth.get("base_url")) or ""
    if base_url:
        token = resolve_vertex_base_url_token() or "custom-endpoint"
        headers: dict[str, str] = {"Authorization": f"Bearer {token}"}
        if zero_thinking:
            headers["X-CC-AGY-Thinking-Budget"] = "0"
        opt_kwargs: dict[str, Any] = {
            "base_url": base_url,
            "headers": headers,
        }
        is_custom_gateway = base_url.startswith("http://") or "googleapis.com" not in base_url
        if is_custom_gateway:
            opt_kwargs["client_args"] = {"follow_redirects": False}
        if timeout_ms:
            opt_kwargs["timeout"] = timeout_ms
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
