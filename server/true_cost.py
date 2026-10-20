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
from pathlib import Path
from typing import Any, Iterable

# Claude list prices in USD per 1M tokens:
# (input, cache_read, cache_write_5m, cache_write_1h, output).
# Matched by substring of the model id, first match wins. Override with
# CC_AGY_LEAD_RATES='{"opus": [5, 0.5, 6.25, 10, 25], ...}'.
DEFAULT_LEAD_RATES: dict[str, tuple[float, float, float, float, float]] = {
    "opus": (5.00, 0.50, 6.25, 10.00, 25.00),
    "sonnet": (3.00, 0.30, 3.75, 6.00, 15.00),
    "haiku": (1.00, 0.10, 1.25, 2.00, 5.00),
}


def lead_rates() -> dict[str, tuple[float, float, float, float, float]]:
    raw = os.environ.get("CC_AGY_LEAD_RATES", "").strip()
    if not raw:
        return dict(DEFAULT_LEAD_RATES)
    try:
        parsed = json.loads(raw)
        out = {}
        for k, v in parsed.items():
            vals = [float(x) for x in v]
            if len(vals) != 5:
                raise ValueError(f"rate for {k!r} needs 5 numbers")
            out[str(k).lower()] = tuple(vals)  # type: ignore[assignment]
        return out
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"CC_AGY_LEAD_RATES is not valid JSON rate table: {exc}") from exc


def rate_for_model(model: str, rates: dict[str, tuple[float, ...]] | None = None) -> tuple[float, ...] | None:
    table = rates if rates is not None else lead_rates()
    m = (model or "").lower()
    for key, val in table.items():
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
    rates = lead_rates()
    seen: set[str] = set()
    per_model: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    messages = 0
    for rec in _iter_assistant_usage(files):
        ts = _iso_to_epoch(str(rec.get("timestamp") or ""))
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
        row = per_model.setdefault(
            model,
            {"messages": 0, "input_tokens": 0, "cache_read_input_tokens": 0,
             "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0, "output_tokens": 0,
             "cost_cents": 0.0, "priced": True},
        )
        row["messages"] += 1
        row["input_tokens"] += int(u.get("input_tokens") or 0)
        row["cache_read_input_tokens"] += int(u.get("cache_read_input_tokens") or 0)
        row["cache_write_5m_tokens"] += w5
        row["cache_write_1h_tokens"] += w1h
        row["output_tokens"] += int(u.get("output_tokens") or 0)
        messages += 1
    total_cents = 0.0
    all_priced = True
    for model, row in per_model.items():
        r = rate_for_model(model, rates)
        if r is None:
            row["priced"] = False
            all_priced = False
            notes.append(f"No lead rate for model {model!r}; its tokens are counted but not priced.")
            continue
        usd = (
            row["input_tokens"] * r[0]
            + row["cache_read_input_tokens"] * r[1]
            + row["cache_write_5m_tokens"] * r[2]
            + row["cache_write_1h_tokens"] * r[3]
            + row["output_tokens"] * r[4]
        ) / 1_000_000.0
        row["cost_cents"] = round(usd * 100.0, 4)
        total_cents += usd * 100.0
    if not files:
        notes.append(f"No Claude Code transcripts found under {tdir}.")
    return {
        "transcripts_dir": str(tdir),
        "transcript_files": len(files),
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
    dispatched_known = True
    for r in in_window:
        cost = r.get("cost") or {}
        dispatched_cents += float(cost.get("total_cost_cents") or 0.0)
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
            "pricing_known": dispatched_known,
            "source": "plugin receipts (Gemini worker turns and ask_gemini calls)",
        },
        "lead": lead,
        "true_total_cost_cents": round(true_total, 4),
        "lead_share_of_true_total": round(float(lead["lead_cost_cents"]) / true_total, 4) if true_total > 0 else None,
        "note": (
            "Dispatched is what the plugin sent to Gemini. True total adds the Claude Code lead session "
            "rebuilt from its transcripts. Compare architectures on true total, not dispatched."
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
