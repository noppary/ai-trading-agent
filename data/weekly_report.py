#!/usr/bin/env python3
"""Generate a weekly markdown summary from trade history files."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

TRADE_HISTORY_DIR = Path("/root/ai-trading-agent/data/trade_history")
REPORTS_DIR = Path("/root/ai-trading-agent/data/reports")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _win_rate(trades: list[dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if _to_float(t.get("pnl")) > 0)
    return (wins / len(trades)) * 100.0


def generate_weekly_report() -> Path:
    TRADE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    cutoff = now - timedelta(days=7)

    weekly_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for file in sorted(TRADE_HISTORY_DIR.glob("*.json")):
        symbol = file.stem.upper()
        try:
            trades = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(trades, list):
            continue

        filtered: list[dict[str, Any]] = []
        for t in trades:
            close_ts = _parse_iso(str(t.get("close_time", "")))
            if close_ts and close_ts >= cutoff:
                filtered.append(t)
        if filtered:
            weekly_by_symbol[symbol] = filtered

    all_weekly_trades = [t for trades in weekly_by_symbol.values() for t in trades]
    total_pnl = sum(_to_float(t.get("pnl")) for t in all_weekly_trades)
    overall_win_rate = _win_rate(all_weekly_trades)

    lines = [
        f"# Weekly Trade Summary ({now.strftime('%Y-%m-%d')})",
        "",
        f"- Window: last 7 days (from {cutoff.strftime('%Y-%m-%d %H:%M UTC')})",
        f"- Total trades: {len(all_weekly_trades)}",
        f"- Total PnL (USD): {total_pnl:.8f}",
        f"- Overall win rate: {overall_win_rate:.2f}%",
        "",
        "## Per Symbol",
        "| Symbol | Trades | Win Rate | Total PnL (USD) | Avg PnL (USD) |",
        "|---|---:|---:|---:|---:|",
    ]

    for symbol, trades in sorted(weekly_by_symbol.items()):
        pnl_sum = sum(_to_float(t.get("pnl")) for t in trades)
        avg_pnl = pnl_sum / len(trades) if trades else 0.0
        lines.append(f"| {symbol} | {len(trades)} | {_win_rate(trades):.2f}% | {pnl_sum:.8f} | {avg_pnl:.8f} |")

    if not weekly_by_symbol:
        lines.append("| (none) | 0 | 0.00% | 0.00000000 | 0.00000000 |")

    lines += [
        "",
        "## Notes",
        "- Source: `data/trade_history/*.json`",
        "- This report includes only trades whose `close_time` falls in the last 7 days.",
    ]

    path = REPORTS_DIR / f"weekly_{now.strftime('%Y%m%d')}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    out = generate_weekly_report()
    print(f"Saved weekly report: {out}")


if __name__ == "__main__":
    main()
