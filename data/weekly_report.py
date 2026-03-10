#!/usr/bin/env python3
"""Generate a weekly markdown summary across all symbols."""

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


def _parse_iso(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def _win_rate(trades: list[dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    return (sum(1 for trade in trades if _to_float(trade.get("pnl")) > 0) / len(trades)) * 100.0


def generate_weekly_report() -> Path:
    TRADE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    cutoff = now - timedelta(days=7)
    grouped: dict[str, list[dict[str, Any]]] = {}

    for history_path in sorted(TRADE_HISTORY_DIR.glob("*.json")):
        try:
            trades = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(trades, list):
            continue

        weekly_trades = []
        for trade in trades:
            closed_at = _parse_iso(str(trade.get("close_time") or ""))
            if closed_at and closed_at >= cutoff:
                weekly_trades.append(trade)
        if weekly_trades:
            grouped[history_path.stem.upper()] = weekly_trades

    all_trades = [trade for trades in grouped.values() for trade in trades]
    total_pnl = sum(_to_float(trade.get("pnl")) for trade in all_trades)

    lines = [
        f"# Weekly Trade Summary {now.strftime('%Y-%m-%d')}",
        "",
        f"- Window start (UTC): {cutoff.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Window end (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Symbols with trades: {len(grouped)}",
        f"- Total completed trades: {len(all_trades)}",
        f"- Total PnL (USD): {total_pnl:.8f}",
        f"- Overall win rate: {_win_rate(all_trades):.2f}%",
        "",
        "## By Symbol",
        "| Symbol | Trades | Win Rate | Total PnL USD | Avg PnL USD | Avg Hold Minutes |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    if not grouped:
        lines.append("| (none) | 0 | 0.00% | 0.00000000 | 0.00000000 | 0.00 |")
    else:
        for symbol, trades in sorted(grouped.items()):
            total_symbol_pnl = sum(_to_float(trade.get("pnl")) for trade in trades)
            avg_symbol_pnl = total_symbol_pnl / len(trades)
            avg_hold = sum(_to_float(trade.get("hold_duration_minutes")) for trade in trades) / len(trades)
            lines.append(
                f"| {symbol} | {len(trades)} | {_win_rate(trades):.2f}% | "
                f"{total_symbol_pnl:.8f} | {avg_symbol_pnl:.8f} | {avg_hold:.2f} |"
            )

    report_path = REPORTS_DIR / f"weekly_{now.strftime('%Y%m%d')}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    report_path = generate_weekly_report()
    print(f"saved weekly report {report_path}")


if __name__ == "__main__":
    main()
