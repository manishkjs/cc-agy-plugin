"""Central Bifrost policy checks for local Gemini Enterprise (`agy` CLI) runs.

A Gemini Enterprise seat runs through the `agy` CLI on the developer's machine, so its
prompts and responses never pass through a central Bifrost gateway. This module lets the
gateway still decide who may run `agy` and with which model. Before `agy` starts, the
plugin makes two small outbound, read-only calls:

- `GET {url}/v1/models` with the `/agy` Virtual Key lists the `agy-cli/<slug>` models the
  key may use. A revoked key or a model outside that list stops the run at 0 tokens.
- `GET {url}/api/routing/rules` (optional, needs read-only credentials) resolves the
  logical `policy-routed` model to the concrete model the gateway currently selects.

Results are cached for a short time. Redirects are refused, the URL must be `https://`
or loopback, and no Virtual Key or password is ever passed to the `agy` subprocess.
Token usage of these runs is recorded in the plugin's receipts, not in Bifrost.

This check is advisory and runs on the developer's machine: someone who controls that
machine can skip it, and Bifrost budgets and rate limits do not apply to seat runs. It
stops honest mistakes and revoked keys; it is not a security boundary.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Mapping
import urllib.error
import urllib.parse
import urllib.request

from server.regions import resolve_bifrost_virtual_key

POLICY_ROUTED_MODEL = "policy-routed"
AGY_PROVIDER_PREFIX = "agy-cli/"
DEFAULT_POLICY_TTL_SECONDS = 60.0
POLICY_HTTP_TIMEOUT_SECONDS = 10.0
_MAX_POLICY_BODY_BYTES = 2_000_000
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_WILDCARD = "*"

# `(url, headers, timeout_seconds) -> (status_code, body_bytes)`; injectable for tests.
PolicyTransport = Callable[[str, Mapping[str, str], float], tuple[int, bytes]]

_LOG = logging.getLogger(__name__)
_CACHE_LOCK = threading.Lock()
_ALLOWED_MODELS_CACHE: dict[tuple[str, str], tuple[float, frozenset[str]]] = {}
_ROUTED_MODEL_CACHE: dict[tuple[str, str], tuple[float, str, str]] = {}
# How `policy-routed` was last resolved per gateway URL, for `doctor`.
_ROUTING_SOURCE: dict[str, str] = {}

_POLICY_ROUTED_LITERAL = r'"(?:agy-cli/)?policy-routed"'
_CEL_EQUALS_RE = re.compile(rf"^(?:model\s*==\s*{_POLICY_ROUTED_LITERAL}|{_POLICY_ROUTED_LITERAL}\s*==\s*model)$")
_CEL_IN_RE = re.compile(r"^model\s+in\s+\[(?P<items>[^\]]*)\]$")
_CEL_STRING_RE = re.compile(r'"([^"\\]*)"')


class BifrostPolicyError(RuntimeError):
    """Raised when Central Bifrost policy denies an `agy` run or cannot be checked."""


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Turn every redirect into an error so a Virtual Key is never sent to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _option(source: Mapping[str, str], suffix: str) -> str:
    for key in (f"CC_AGY_{suffix}", f"CLAUDE_PLUGIN_OPTION_{suffix}", f"CC_AGY_USERCFG_{suffix}"):
        cleaned = str(source.get(key) or "").strip()
        if cleaned and not cleaned.startswith("${"):
            return cleaned
    return ""


def validate_policy_url(raw_url: str) -> str:
    """Return `raw_url` without a trailing slash, or raise `ValueError` if it is unsafe.

    Only `https://` URLs and plain `http://` URLs on loopback are accepted. Embedded
    credentials, query strings, and fragments are rejected.
    """
    parsed = urllib.parse.urlparse(raw_url.strip())
    host = (parsed.hostname or "").lower()
    scheme = parsed.scheme.lower()
    if not host or scheme not in ("https", "http"):
        raise ValueError("CC_AGY_BIFROST_POLICY_URL must be an https:// URL (or http:// on loopback).")
    if scheme == "http" and host not in _LOOPBACK_HOSTS:
        raise ValueError("CC_AGY_BIFROST_POLICY_URL must use https:// unless it points at 127.0.0.1 or localhost.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("CC_AGY_BIFROST_POLICY_URL must not contain credentials, a query string, or a fragment.")
    return f"{scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def resolve_bifrost_policy_config(env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Read the Bifrost `/agy` policy settings, or return None when governance is off.

    Governance is on when `CC_AGY_BIFROST_POLICY_URL` is set. The Virtual Key comes from
    `CC_AGY_BIFROST_AGY_VIRTUAL_KEY`, falling back to `CC_AGY_BIFROST_VIRTUAL_KEY`.

    Raises:
        ValueError: If the URL is unsafe or no Virtual Key is configured.
    """
    source = env if env is not None else os.environ
    raw_url = _option(source, "BIFROST_POLICY_URL")
    if not raw_url:
        return None
    url = validate_policy_url(raw_url)
    virtual_key = _option(source, "BIFROST_AGY_VIRTUAL_KEY") or (resolve_bifrost_virtual_key(env=source) or "")
    if not virtual_key:
        raise ValueError(
            "CC_AGY_BIFROST_POLICY_URL is set, but no Virtual Key was found. "
            "Set CC_AGY_BIFROST_AGY_VIRTUAL_KEY (or CC_AGY_BIFROST_VIRTUAL_KEY)."
        )
    try:
        ttl_seconds = max(0.0, float(_option(source, "BIFROST_POLICY_TTL_SECONDS") or DEFAULT_POLICY_TTL_SECONDS))
    except ValueError:
        ttl_seconds = DEFAULT_POLICY_TTL_SECONDS
    return {
        "url": url,
        "virtual_key": virtual_key,
        "policy_user": _option(source, "BIFROST_POLICY_USER"),
        "policy_password": _option(source, "BIFROST_POLICY_PASSWORD"),
        "fallback_model": _agy_slug(_option(source, "BIFROST_AGY_POLICY_MODEL")),
        "ttl_seconds": ttl_seconds,
    }


def _agy_slug(model: str) -> str:
    cleaned = model.strip()
    return cleaned[len(AGY_PROVIDER_PREFIX):] if cleaned.startswith(AGY_PROVIDER_PREFIX) else cleaned


def _key_digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _default_transport(url: str, headers: Mapping[str, str], timeout_seconds: float) -> tuple[int, bytes]:
    opener = urllib.request.build_opener(_RefuseRedirects)
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            return int(response.status), response.read(_MAX_POLICY_BODY_BYTES)
    except urllib.error.HTTPError as exc:
        return int(exc.code), b""


def _get_json(url: str, headers: Mapping[str, str], transport: PolicyTransport | None) -> tuple[int, Any]:
    try:
        status, body = (transport or _default_transport)(url, headers, POLICY_HTTP_TIMEOUT_SECONDS)
    except (OSError, ValueError) as exc:
        raise BifrostPolicyError(f"Could not reach Central Bifrost policy endpoint: {type(exc).__name__}.") from exc
    if status != 200:
        return status, None
    try:
        return status, json.loads(body.decode("utf-8"))
    except ValueError as exc:
        raise BifrostPolicyError("Central Bifrost returned a policy response that is not valid JSON.") from exc


def fetch_allowed_agy_models(
    config: Mapping[str, Any],
    *,
    transport: PolicyTransport | None = None,
) -> frozenset[str]:
    """Return the `agy` model slugs the `/agy` Virtual Key may use (`*` means any).

    Raises:
        BifrostPolicyError: If the key is rejected or the gateway cannot be checked. A
            run is never allowed on the strength of a failed check.
    """
    cache_key = (str(config["url"]), _key_digest(str(config["virtual_key"])))
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _ALLOWED_MODELS_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < float(config["ttl_seconds"]):
        return cached[1]

    virtual_key = str(config["virtual_key"])
    status, payload = _get_json(
        f"{config['url']}/v1/models",
        {"Authorization": f"Bearer {virtual_key}", "x-bf-vk": virtual_key, "Accept": "application/json"},
        transport,
    )
    if status in (401, 403):
        raise BifrostPolicyError(
            f"Central Bifrost rejected the /agy Virtual Key (HTTP {status}). "
            "The key may be revoked or lack `agy-cli` access."
        )
    if status != 200:
        raise BifrostPolicyError(f"Central Bifrost policy check failed (HTTP {status}).")

    entries = payload.get("data") if isinstance(payload, dict) else payload
    allowed: set[str] = set()
    for entry in entries if isinstance(entries, list) else []:
        model_id = str(entry.get("id") or "") if isinstance(entry, dict) else str(entry or "")
        if model_id.startswith(AGY_PROVIDER_PREFIX):
            allowed.add(model_id[len(AGY_PROVIDER_PREFIX):])
    frozen = frozenset(allowed)
    with _CACHE_LOCK:
        _ALLOWED_MODELS_CACHE[cache_key] = (now, frozen)
    return frozen


def resolve_policy_routed_model(
    config: Mapping[str, Any],
    *,
    transport: PolicyTransport | None = None,
) -> str:
    """Return the concrete `agy` model the gateway currently maps `policy-routed` to.

    See `_resolve_policy_routed` for the lookup order.

    Raises:
        BifrostPolicyError: If no model can be resolved.
    """
    return _resolve_policy_routed(config, transport=transport)[0]


def _resolve_policy_routed(
    config: Mapping[str, Any],
    *,
    transport: PolicyTransport | None = None,
) -> tuple[str, str]:
    """Return the concrete `agy` model the gateway currently maps `policy-routed` to.

    Returns `(model, gateway_name)`, where `gateway_name` is the name the rule used
    (a tier alias such as `agy-frontier`, or the model itself). Reads
    `/api/routing/rules` when read-only credentials are configured and maps a tier
    alias through the `agy-cli` provider's `aliases`. Otherwise, or when the rules cannot
    be read, it uses `CC_AGY_BIFROST_AGY_POLICY_MODEL`. Falling back after a failed read
    is logged and shown by `doctor`, never silent.

    Raises:
        BifrostPolicyError: If neither source yields a model.
    """
    url = str(config["url"])
    fallback = str(config.get("fallback_model") or "")
    user, password = str(config.get("policy_user") or ""), str(config.get("policy_password") or "")
    source = "fallback_model"
    if user and password:
        cache_key = (url, _key_digest(f"{user}:{password}"))
        now = time.monotonic()
        with _CACHE_LOCK:
            cached = _ROUTED_MODEL_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < float(config["ttl_seconds"]):
            return cached[1], cached[2]
        basic = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        try:
            status, payload = _get_json(
                f"{url}/api/routing/rules",
                {"Authorization": f"Basic {basic}", "Accept": "application/json"},
                transport,
            )
        except BifrostPolicyError:
            status, payload = 0, None
        routed = _first_policy_routed_target(payload) if status == 200 else ""
        if routed:
            resolved = _resolve_alias(config, routed, basic, transport)
            with _CACHE_LOCK:
                _ROUTED_MODEL_CACHE[cache_key] = (now, resolved, routed)
                _ROUTING_SOURCE[url] = "routing_rules"
            return resolved, routed
        if status in (401, 403):
            source = "fallback_model_rules_unauthorized"
        elif status == 200:
            source = "fallback_model_no_matching_rule"
        else:
            source = "fallback_model_rules_unreachable"
        _LOG.warning(
            "Bifrost routing rules could not resolve 'policy-routed' (%s); using the fallback model %r.",
            source.removeprefix("fallback_model_"),
            fallback or None,
        )
    with _CACHE_LOCK:
        _ROUTING_SOURCE[url] = source
    if fallback:
        return fallback, fallback
    raise BifrostPolicyError(
        "Model 'policy-routed' could not be resolved. Configure read-only routing-rule credentials "
        "(CC_AGY_BIFROST_POLICY_USER and CC_AGY_BIFROST_POLICY_PASSWORD) or set CC_AGY_BIFROST_AGY_POLICY_MODEL."
    )


def _split_top_level_or(expression: str) -> list[str]:
    """Split a CEL expression on `||` that is not inside parentheses, brackets, or a string."""
    clauses: list[str] = []
    depth = 0
    in_string = False
    start = 0
    idx = 0
    while idx < len(expression):
        char = expression[idx]
        if in_string:
            if char == "\\":
                idx += 1
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "|" and depth == 0 and expression.startswith("||", idx):
            clauses.append(expression[start:idx])
            start = idx + 2
            idx += 1
        idx += 1
    clauses.append(expression[start:])
    return clauses


def _strip_outer_parens(clause: str) -> str:
    cleaned = clause.strip()
    while cleaned.startswith("(") and cleaned.endswith(")") and _split_top_level_or(cleaned[1:-1]) == [cleaned[1:-1]]:
        inner = cleaned[1:-1].strip()
        if inner.count("(") != inner.count(")"):
            break
        cleaned = inner
    return cleaned


def cel_selects_policy_routed(expression: str) -> bool:
    """Return True if a routing rule's CEL always fires for the `policy-routed` model.

    Only plain positive matches count: `model == "policy-routed"` (either side, with or
    without the `agy-cli/` prefix) or `model in [..., "policy-routed", ...]`, optionally
    joined by `||`. Negated (`!=`, `!`) or conditional (`&&`) clauses never count, because
    they cannot be evaluated here and may not apply to this request.
    """
    for raw_clause in _split_top_level_or(expression or ""):
        clause = _strip_outer_parens(raw_clause)
        if not clause or "&&" in clause or "!" in clause:
            continue
        if _CEL_EQUALS_RE.match(clause):
            return True
        in_match = _CEL_IN_RE.match(clause)
        if in_match and any(
            item in (POLICY_ROUTED_MODEL, AGY_PROVIDER_PREFIX + POLICY_ROUTED_MODEL)
            for item in _CEL_STRING_RE.findall(in_match.group("items"))
        ):
            return True
    return False


def _first_policy_routed_target(payload: Any) -> str:
    rules = payload if isinstance(payload, list) else None
    if isinstance(payload, dict):
        rules = payload.get("rules") if isinstance(payload.get("rules"), list) else payload.get("data")
    candidates = [r for r in rules if isinstance(r, dict)] if isinstance(rules, list) else []

    def _priority(rule: Mapping[str, Any]) -> int:
        try:
            return int(rule.get("priority", 100))
        except (TypeError, ValueError):
            return 100

    for rule in sorted(candidates, key=_priority):
        if rule.get("enabled", True) is False or not cel_selects_policy_routed(str(rule.get("cel_expression") or "")):
            continue
        targets = rule.get("targets")
        first = targets[0] if isinstance(targets, list) and targets and isinstance(targets[0], dict) else {}
        model = str(first.get("model") or "").strip()
        if model:
            return model.rsplit("/", 1)[-1]
    return ""


def _provider_aliases(payload: Any) -> dict[str, str]:
    """Collect `alias -> model` maps from a Bifrost provider config (`keys[*].aliases`)."""
    provider = payload.get("provider", payload) if isinstance(payload, dict) else {}
    keys = provider.get("keys") if isinstance(provider, dict) else None
    aliases: dict[str, str] = {}
    for key_cfg in keys if isinstance(keys, list) else []:
        raw = key_cfg.get("aliases") if isinstance(key_cfg, dict) else None
        if isinstance(raw, dict):
            aliases.update({str(k): _agy_slug(str(v)) for k, v in raw.items() if isinstance(v, str) and v.strip()})
    return aliases


def _resolve_alias(config: Mapping[str, Any], model: str, basic: str, transport: PolicyTransport | None) -> str:
    """Map a Bifrost tier alias (for example `agy-frontier`) to the real `agy` model, if one is defined."""
    try:
        status, payload = _get_json(
            f"{config['url']}/api/providers/agy-cli",
            {"Authorization": f"Basic {basic}", "Accept": "application/json"},
            transport,
        )
    except BifrostPolicyError:
        return model
    return _provider_aliases(payload).get(model, model) if status == 200 else model


def policy_routing_source(url: str) -> str | None:
    """Return how `policy-routed` was last resolved for `url` (for `doctor`)."""
    with _CACHE_LOCK:
        return _ROUTING_SOURCE.get(url)


_EFFORT_SUFFIXES = frozenset({"low", "medium", "high", "xhigh", "max"})


def is_model_allowed(model: str, effort: str | None, allowed: frozenset[str]) -> bool:
    """Return True if `model` (optionally with its `-<effort>` suffix or base slug) is in `allowed`."""
    slug = _agy_slug(model)
    if _WILDCARD in allowed or slug in allowed:
        return True
    if effort and f"{slug}-{effort}" in allowed:
        return True
    if "-" in slug:
        base_slug, suffix = slug.rsplit("-", 1)
        if suffix in _EFFORT_SUFFIXES and base_slug in allowed:
            return True
    return False


def enforce_agy_model_policy(
    requested_model: str,
    effort: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    transport: PolicyTransport | None = None,
) -> dict[str, Any]:
    """Check one `agy` run against Central Bifrost policy before any tokens are spent.

    Args:
        requested_model: The model the run asked for; `policy-routed` is resolved here.
        effort: The `agy --effort` value, used to match `<model>-<effort>` slugs.
        env: Environment to read settings from (defaults to `os.environ`).
        transport: Optional HTTP transport override for tests.

    Returns:
        A dict with `enabled`, `model` (the model to run), `allowed` (the allowed slugs,
        or None when governance is off), and `policy_routed` (True when resolved here).

    Raises:
        BifrostPolicyError: If the key is rejected, the model is not allowed, or the
            policy cannot be checked.
        ValueError: If the policy settings themselves are invalid.
    """
    config = resolve_bifrost_policy_config(env)
    if config is None:
        return {"enabled": False, "model": requested_model, "allowed": None, "policy_routed": False}

    allowed = fetch_allowed_agy_models(config, transport=transport)
    slug = _agy_slug(requested_model)
    policy_routed = slug == POLICY_ROUTED_MODEL
    target = slug
    if policy_routed:
        if _WILDCARD not in allowed and POLICY_ROUTED_MODEL not in allowed:
            raise BifrostPolicyError("Central Bifrost policy does not allow 'policy-routed' for this Virtual Key.")
        target, gateway_name = _resolve_policy_routed(config, transport=transport)
        if gateway_name != target and is_model_allowed(gateway_name, None, allowed):
            return {"enabled": True, "model": target, "allowed": allowed, "policy_routed": True}
    if not is_model_allowed(target, None if policy_routed else effort, allowed):
        raise BifrostPolicyError(
            f"Model '{target}' is not allowed by Central Bifrost policy for this Virtual Key "
            f"({len(allowed)} `agy-cli` model(s) allowed)."
        )
    return {"enabled": True, "model": target, "allowed": allowed, "policy_routed": policy_routed}


def inspect_bifrost_policy(
    env: Mapping[str, str] | None = None,
    *,
    live: bool = False,
    transport: PolicyTransport | None = None,
) -> dict[str, Any]:
    """Summarize Bifrost `/agy` governance for `doctor`, without printing any secret.

    With `live=False` only local settings are read. With `live=True` the gateway is
    queried once and `allowed_model_count` and `policy_routed_model` are filled in.
    """
    report: dict[str, Any] = {
        "enabled": False,
        "url": None,
        "routing_rules_credentials": False,
        "fallback_policy_model": None,
        "allowed_model_count": None,
        "policy_routed_model": None,
        "error": None,
    }
    try:
        config = resolve_bifrost_policy_config(env)
    except ValueError as exc:
        report["error"] = str(exc)
        return report
    if config is None:
        return report
    report.update(
        {
            "enabled": True,
            "url": config["url"],
            "routing_rules_credentials": bool(config["policy_user"] and config["policy_password"]),
            "fallback_policy_model": config["fallback_model"] or None,
        }
    )
    if not live:
        return report
    try:
        allowed = fetch_allowed_agy_models(config, transport=transport)
        report["allowed_model_count"] = len(allowed)
        if _WILDCARD in allowed or POLICY_ROUTED_MODEL in allowed:
            report["policy_routed_model"] = resolve_policy_routed_model(config, transport=transport)
            report["policy_routed_source"] = policy_routing_source(str(config["url"]))
    except BifrostPolicyError as exc:
        report["error"] = str(exc)
    report["advisory"] = (
        "Checked on this machine before each run; Bifrost budgets and token counts do not apply to seat runs."
    )
    return report


def clear_policy_cache() -> None:
    """Drop cached policy results (used by tests and after a settings change)."""
    with _CACHE_LOCK:
        _ALLOWED_MODELS_CACHE.clear()
        _ROUTED_MODEL_CACHE.clear()
        _ROUTING_SOURCE.clear()
