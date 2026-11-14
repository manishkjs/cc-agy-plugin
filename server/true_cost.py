"""Dispatched cost vs true total cost.

*Dispatched* is what the plugin's own receipts record: every Gemini worker turn and
ask_gemini call it sent. *True total* adds the Claude Code lead session that drove those
calls, which the plugin never sees directly. We rebuild it after the fact from Claude
Code's session transcripts (``~/.claude/projects/<hash>/*.jsonl``, or
``$CLAUDE_CONFIG_DIR/projects/<hash>/``), summing each assistant message's ``usage``
block inside a time window and pricing it per model.

The two figures are never blended silently: the report always shows dispatched, lead,
and true total side by side. The lead's own loop is often most of the bill, so a cost
claim built on dispatched spend alone overstates savings.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from server.rate_card import DEFAULT_LEAD_RATES, resolve_claude_lead_rates

_SONNET_5_ALIASES: tuple[str, ...] = (
    "sonnet-5",
    "sonnet-5.0",
    "sonnet-5-5",
    "sonnet-5.5",
    "claude-5-sonnet",
    "claude-5.5-sonnet",
    "claude-5-0-sonnet",
    "claude-5-5-sonnet",
    "claude-sonnet-5",
)


def _parse_lead_rate_overrides() -> dict[str, tuple[float, float, float, float, float]]:
    primary = (os.environ.get("CC_AGY_LEAD_RATES") or "").strip()
    alias = (os.environ.get("CC_AGY_LEAD_RATES_JSON") or "").strip()
    env_name = "CC_AGY_LEAD_RATES" if primary else "CC_AGY_LEAD_RATES_JSON"
    raw = primary or alias
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("top-level JSON value must be an object")
        explicit: dict[str, tuple[float, float, float, float, float]] = {}
        for k, v in parsed.items():
            vals = [float(x) for x in v]
            if len(vals) != 5:
                raise ValueError(f"rate for {k!r} needs 5 numbers")
            explicit[str(k).lower()] = tuple(vals)  # type: ignore[assignment]
        return explicit
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{env_name} is not valid JSON rate table: {exc}") from exc


def lead_rates(
    as_of: date | str | None = None,
) -> dict[str, tuple[float, float, float, float, float]]:
    base_rates = resolve_claude_lead_rates(as_of=as_of) if as_of is not None else resolve_claude_lead_rates()
    explicit = _parse_lead_rate_overrides()
    if not explicit:
        return dict(base_rates)
    out = dict(base_rates)
    out.update(explicit)
    sorted_explicit = sorted(explicit.items(), key=lambda kv: len(kv[0]), reverse=True)
    for base_key in list(out.keys()):
        if base_key in explicit:
            continue
        for exp_key, exp_val in sorted_explicit:
            if exp_key in base_key:
                out[base_key] = exp_val
                break
    if "opus" in explicit:
        for alias in ("opus-5-5", "opus-5.5"):
            if alias not in explicit:
                out[alias] = explicit["opus"]
    if "sonnet" in explicit:
        for alias in _SONNET_5_ALIASES:
            if alias not in explicit:
                out[alias] = explicit["sonnet"]
    return out


def rate_for_model(
    model: str,
    rates: dict[str, tuple[float, ...]] | None = None,
    *,
    as_of: date | str | None = None,
) -> tuple[float, ...] | None:
    m = (model or "").lower()
    if rates is None:
        explicit = _parse_lead_rate_overrides()
        for key, val in sorted(explicit.items(), key=lambda kv: len(kv[0]), reverse=True):
            if key in m:
                return val
        table: dict[str, tuple[float, ...]] = (
            resolve_claude_lead_rates(as_of=as_of) if as_of is not None else resolve_claude_lead_rates()
        )
    else:
        table = rates
    for key, val in sorted(table.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key in m:
            return val
    return None


def project_hash(path: str | Path) -> str:
    """Claude Code's transcript folder name: absolute path with every non-alphanumeric char replaced by '-'."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(Path(path).resolve()))


def default_transcripts_dir(workspace_root: str | Path, config_dir: str | None = None) -> Path:
    cfg = config_dir if config_dir is not None else os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    base = Path(cfg).expanduser() if cfg else Path.home() / ".claude"
    return base / "projects" / project_hash(workspace_root)


def _iso_to_epoch(ts: str) -> float | None:
    if not ts:
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _transcript_files(tdir: Path) -> list[Path]:
    if not tdir.is_dir():
        return []
    files = sorted(tdir.glob("*.jsonl"))
    files += sorted(tdir.glob("*/subagents/*.jsonl"))
    files += sorted(tdir.glob("subagents/*.jsonl"))
    return files


def _iter_opencode_assistant_usage(workspace_root: str | Path) -> Iterable[dict[str, Any]]:
    """Yield assistant message usage from OpenCode's XDG storage (`storage/session` + `storage/message`)."""
    data_home = Path((os.environ.get("XDG_DATA_HOME") or "").strip() or (Path.home() / ".local" / "share")).expanduser()
    storage_dir = data_home / "opencode" / "storage"
    session_root = storage_dir / "session"
    message_root = storage_dir / "message"
    if not session_root.is_dir() or not message_root.is_dir():
        return
    target_ws = str(Path(workspace_root).resolve())
    matching_sessions: set[str] = set()
    for sfile in session_root.rglob("*.json"):
        try:
            sdata = json.loads(sfile.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        if not isinstance(sdata, dict):
            continue
        s_dir = str(sdata.get("directory") or sdata.get("cwd") or "").strip()
        if s_dir:
            try:
                if str(Path(s_dir).resolve()) != target_ws:
                    continue
            except OSError:
                if s_dir != target_ws:
                    continue
        sid = str(sdata.get("id") or sfile.stem).strip()
        if sid:
            matching_sessions.add(sid)
    for sid in sorted(matching_sessions):
        msg_dir = message_root / sid
        if not msg_dir.is_dir():
            continue
        for mfile in sorted(msg_dir.glob("*.json")):
            try:
                mdata = json.loads(mfile.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                continue
            if not isinstance(mdata, dict) or mdata.get("role") != "assistant":
                continue
            tokens = mdata.get("tokens")
            if not isinstance(tokens, dict):
                continue
            time_obj = mdata.get("time") if isinstance(mdata.get("time"), dict) else {}
            created_ms = time_obj.get("created") or time_obj.get("completed")
            iso_ts: str | None = None
            if isinstance(created_ms, (int, float)) and created_ms > 0:
                from datetime import datetime, timezone

                sec = float(created_ms) / 1000.0 if created_ms > 1e11 else float(created_ms)
                iso_ts = datetime.fromtimestamp(sec, tz=timezone.utc).isoformat()
            elif isinstance(created_ms, str):
                iso_ts = created_ms
            cache_obj = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
            yield {
                "file": str(mfile),
                "timestamp": iso_ts,
                "message_id": str(mdata.get("id") or mfile.stem),
                "model": str(mdata.get("modelID") or (mdata.get("model") or {}).get("modelID") or ""),
                "usage": {
                    "input_tokens": int(tokens.get("input") or 0),
                    "output_tokens": int(tokens.get("output") or 0) + int(tokens.get("reasoning") or 0),
                    "cache_read_input_tokens": int(cache_obj.get("read") or 0),
                    "cache_creation_input_tokens": int(cache_obj.get("write") or 0),
                },
            }


def _iter_assistant_usage(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for p in paths:
        try:
            fh = open(p, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(obj, dict) or obj.get("type") != "assistant":
                    continue
                msg = obj.get("message") or {}
                usage = msg.get("usage") if isinstance(msg, dict) else None
                if not isinstance(usage, dict):
                    continue
                yield {
                    "file": str(p),
                    "timestamp": obj.get("timestamp"),
                    "message_id": msg.get("id") or obj.get("requestId") or obj.get("uuid"),
                    "model": msg.get("model") or "",
                    "usage": usage,
                }


def collect_lead_usage(
    workspace_root: str | Path,
    *,
    since_epoch: float | None = None,
    until_epoch: float | None = None,
    transcripts_dir: str | Path | None = None,
    session_anchor_epoch: float | None = None,
) -> dict[str, Any]:
    """Sum the Claude Code lead's token usage and list-price cost inside [since, until].

    With session_anchor_epoch (and no since_epoch), whole transcript files are counted when any of
    their assistant messages falls in [anchor, until], so a lead's planning turns before its first
    delegation are included.
    """
    tdir = Path(transcripts_dir).expanduser() if transcripts_dir else default_transcripts_dir(workspace_root)
    files = _transcript_files(tdir)
    if since_epoch is None and session_anchor_epoch is not None:
        overlapping = []
        for f in files:
            for rec in _iter_assistant_usage([f]):
                ts = _iso_to_epoch(str(rec.get("timestamp") or ""))
                if ts is not None and ts >= session_anchor_epoch and (until_epoch is None or ts <= until_epoch):
                    overlapping.append(f)
                    break
        files = overlapping
    # Validate CC_AGY_LEAD_RATES eagerly even when transcripts are empty.
    _ = lead_rates()
    rates_cache: dict[str | None, dict[str, tuple[float, ...]]] = {}
    seen: set[str] = set()
    per_model: dict[str, dict[str, Any]] = {}
    per_model_usd: dict[str, float] = {}
    unpriced_models: list[str] = []
    notes: list[str] = []
    messages = 0
    opencode_records = list(_iter_opencode_assistant_usage(workspace_root)) if transcripts_dir is None else []
    combined_records = list(_iter_assistant_usage(files)) + opencode_records
    for rec in combined_records:
        raw_ts = str(rec.get("timestamp") or "")
        ts = _iso_to_epoch(raw_ts)
        if ts is None:
            continue
        if since_epoch is not None and ts < since_epoch:
            continue
        if until_epoch is not None and ts > until_epoch:
            continue
        mid = str(rec.get("message_id") or "")
        # Claude Code writes one line per content block with the same message id and usage.
        if mid:
            if mid in seen:
                continue
            seen.add(mid)
        u = rec["usage"]
        model = str(rec.get("model") or "unknown")
        if model == "<synthetic>":
            continue
        cc = u.get("cache_creation") if isinstance(u.get("cache_creation"), dict) else {}
        w5 = int(cc.get("ephemeral_5m_input_tokens") or 0)
        w1h = int(cc.get("ephemeral_1h_input_tokens") or 0)
        w_total = int(u.get("cache_creation_input_tokens") or 0)
        if w5 + w1h < w_total:
            w5 += w_total - (w5 + w1h)  # unsplit writes priced at the 5-minute rate
        in_tok = int(u.get("input_tokens") or 0)
        cr_tok = int(u.get("cache_read_input_tokens") or 0)
        out_tok = int(u.get("output_tokens") or 0)
        row = per_model.setdefault(
            model,
            {"messages": 0, "input_tokens": 0, "cache_read_input_tokens": 0,
             "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0, "output_tokens": 0,
             "cost_cents": 0.0, "priced": True},
        )
        row["messages"] += 1
        row["input_tokens"] += in_tok
        row["cache_read_input_tokens"] += cr_tok
        row["cache_write_5m_tokens"] += w5
        row["cache_write_1h_tokens"] += w1h
        row["output_tokens"] += out_tok
        messages += 1

        msg_date = raw_ts[:10] if len(raw_ts) >= 10 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_ts[:10]) else None
        if msg_date not in rates_cache:
            rates_cache[msg_date] = lead_rates(as_of=msg_date)
        r = rate_for_model(model, rates_cache[msg_date])
        if r is None:
            row["priced"] = False
            if model not in unpriced_models:
                unpriced_models.append(model)
        else:
            msg_usd = (
                in_tok * r[0]
                + cr_tok * r[1]
                + w5 * r[2]
                + w1h * r[3]
                + out_tok * r[4]
            ) / 1_000_000.0
            per_model_usd[model] = per_model_usd.get(model, 0.0) + msg_usd

    total_cents = 0.0
    all_priced = len(unpriced_models) == 0
    for unpriced_model in unpriced_models:
        notes.append(f"No lead rate for model {unpriced_model!r}; its tokens are counted but not priced.")
    for model, row in per_model.items():
        if not row["priced"]:
            continue
        usd = per_model_usd.get(model, 0.0)
        row["cost_cents"] = round(usd * 100.0, 4)
        total_cents += usd * 100.0
    if not files and not opencode_records:
        notes.append(f"No Claude Code or OpenCode transcripts found under {tdir}.")
    return {
        "transcripts_dir": str(tdir),
        "transcript_files": len(files) + len({r["file"] for r in opencode_records}),
        "messages": messages,
        "per_model": per_model,
        "lead_cost_cents": round(total_cents, 4),
        "all_lead_pricing_known": all_priced,
        "notes": notes,
    }


def true_total_report(
    store: Any,
    workspace_root: str | Path,
    *,
    since_epoch: float | None = None,
    until_epoch: float | None = None,
    transcripts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return dispatched (plugin receipts), lead (transcripts) and true total cost for a window.

    Without since_epoch, the lead side counts every transcript session that was active between the
    first delegated job and until_epoch (planning turns before the first delegation included), and
    the dispatched side counts every receipt up to until_epoch.
    """
    from server.receipts import (
        _is_included_ge_seat_receipt,
        _receipt_cost_cents,
        _receipt_list_equivalent_cents,
    )

    receipts = store.list_receipts(workspace_root=workspace_root)
    anchor: float | None = None
    if since_epoch is None:
        starts = [float(r.get("started_at_epoch") or 0.0) for r in receipts if r.get("started_at_epoch")]
        anchor = min(starts) if starts else None
    until = until_epoch if until_epoch is not None else time.time()
    in_window = [
        r for r in receipts
        if (since_epoch is None or float(r.get("started_at_epoch") or 0.0) >= since_epoch)
        and float(r.get("started_at_epoch") or 0.0) <= until
    ]
    dispatched_cents = 0.0
    ge_equiv_cents = 0.0
    dispatched_known = True
    by_auth_mode: dict[str, dict[str, Any]] = {
        "gemini_enterprise_seat": {
            "jobs": 0,
            "billing_model": "included_ge_seat",
            "total_tokens": 0,
            "billed_cost_cents": 0.0,
            "list_equivalent_cents": 0.0,
        },
        "pay_as_you_go": {
            "jobs": 0,
            "billing_model": "pay_as_you_go",
            "total_tokens": 0,
            "billed_cost_cents": 0.0,
            "list_equivalent_cents": 0.0,
        },
    }
    for r in in_window:
        cost = r.get("cost") or {}
        usage = r.get("usage") or {}
        tot_tok = int(usage.get("total_token_count") or 0)
        is_ge = _is_included_ge_seat_receipt(r)
        billed_c = _receipt_cost_cents(r)
        equiv_c = _receipt_list_equivalent_cents(r)
        dispatched_cents += billed_c
        if is_ge:
            ge_equiv_cents += equiv_c
            ge_bucket = by_auth_mode["gemini_enterprise_seat"]
            ge_bucket["jobs"] += 1
            ge_bucket["total_tokens"] += tot_tok
            ge_bucket["billed_cost_cents"] = round(ge_bucket["billed_cost_cents"] + billed_c, 4)
            ge_bucket["list_equivalent_cents"] = round(ge_bucket["list_equivalent_cents"] + equiv_c, 4)
        else:
            payg_bucket = by_auth_mode["pay_as_you_go"]
            payg_bucket["jobs"] += 1
            payg_bucket["total_tokens"] += tot_tok
            payg_bucket["billed_cost_cents"] = round(payg_bucket["billed_cost_cents"] + billed_c, 4)
            payg_bucket["list_equivalent_cents"] = round(payg_bucket["list_equivalent_cents"] + equiv_c, 4)
        if cost.get("pricing_known") is False:
            dispatched_known = False
    lead = collect_lead_usage(
        workspace_root, since_epoch=since_epoch, until_epoch=until, transcripts_dir=transcripts_dir,
        session_anchor_epoch=anchor,
    )
    true_total = dispatched_cents + float(lead["lead_cost_cents"])
    return {
        "window": {
            "since_epoch": since_epoch,
            "until_epoch": round(until, 3),
            "mode": "explicit" if since_epoch is not None else "whole lead sessions overlapping delegated jobs",
            "first_job_epoch": anchor,
        },
        "dispatched": {
            "jobs": len(in_window),
            "cost_cents": round(dispatched_cents, 4),
            "included_ge_seat_equivalent_cents": round(ge_equiv_cents, 4),
            "by_auth_mode": by_auth_mode,
            "pricing_known": dispatched_known,
            "source": "plugin receipts (Gemini worker turns and ask_gemini calls)",
        },
        "lead": lead,
        "true_total_cost_cents": round(true_total, 4),
        "lead_share_of_true_total": round(float(lead["lead_cost_cents"]) / true_total, 4) if true_total > 0 else None,
        "note": (
            "Dispatched cost_cents counts pay-as-you-go Gemini spend only; included Gemini Enterprise seat runs "
            "bill $0.00 and report their token volume and list-rate value under dispatched.by_auth_mode. "
            "True total adds the Lead session rebuilt from Claude Code / OpenCode transcripts."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dispatched vs true total cost for a cc-agy-plugin workspace.")
    ap.add_argument("--workspace", default=os.getcwd())
    ap.add_argument("--data-dir", default=None, help="cc-agy-plugin data dir (default: CC_AGY_DATA_DIR or plugin default)")
    ap.add_argument("--since", type=float, default=None, help="window start (epoch seconds)")
    ap.add_argument("--until", type=float, default=None, help="window end (epoch seconds)")
    ap.add_argument("--transcripts-dir", default=None)
    args = ap.parse_args(argv)
    from server.receipts import ReceiptStore

    store = ReceiptStore(args.data_dir or os.environ.get("CC_AGY_DATA_DIR") or None, workspace_root=args.workspace)
    rep = true_total_report(
        store, args.workspace, since_epoch=args.since, until_epoch=args.until, transcripts_dir=args.transcripts_dir
    )
    json.dump(rep, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
