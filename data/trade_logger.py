#!/usr/bin/env python3
"""Fetch Hyperliquid testnet fills and build per-symbol completed trade history."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

API_URL = "https://api.hyperliquid-testnet.xyz/info"
USER_ADDRESS = "0xE332bff7eBCAC0160468654F79B6fA3176638256"

BASE_DIR = Path("/root/ai-trading-agent/data")
TRADE_HISTORY_DIR = BASE_DIR / "trade_history"
ANALYSIS_STATE_PATH = BASE_DIR / "analysis_state.json"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_iso_utc(timestamp_ms: Any) -> str:
    ts = _to_float(timestamp_ms)
    if ts <= 0:
        return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return datetime.fromtimestamp(ts / 1000, tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _side_from_fill(fill: dict[str, Any]) -> str:
    for key in ("side", "dir", "direction"):
        val = str(fill.get(key, "")).lower()
        if "buy" in val or val == "b":
            return "buy"
        if "sell" in val or val == "s":
            return "sell"
    return "buy"


def _extract_market_condition(fill: dict[str, Any]) -> str | None:
    for key in ("market_condition", "marketCondition", "regime", "volatility"):
        if key in fill and fill[key] not in (None, ""):
            return str(fill[key])
    return None


def _extract_signal_confidence(fill: dict[str, Any]) -> float | None:
    for key in ("signal_confidence", "signalConfidence", "confidence"):
        if key in fill:
            try:
                return float(fill[key])
            except (TypeError, ValueError):
                return None
    return None


def fetch_user_fills() -> list[dict[str, Any]]:
    payload = {"type": "userFills", "user": USER_ADDRESS}
    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected response format from Hyperliquid userFills endpoint")
    return data


def build_completed_trades(fills: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    symbol_fills: dict[str, list[dict[str, Any]]] = {}
    for fill in fills:
        symbol = str(fill.get("coin") or fill.get("symbol") or fill.get("asset") or "").upper()
        if not symbol:
            continue
        symbol_fills.setdefault(symbol, []).append(fill)

    completed: dict[str, list[dict[str, Any]]] = {}

    for symbol, sfills in symbol_fills.items():
        sfills.sort(key=lambda x: _to_float(x.get("time"), 0.0))
        trades: list[dict[str, Any]] = []
        position: dict[str, Any] | None = None

        for fill in sfills:
            px = _to_float(fill.get("px") or fill.get("price"))
            qty = abs(_to_float(fill.get("sz") or fill.get("size")))
            if px <= 0 or qty <= 0:
                continue

            side = _side_from_fill(fill)
            signed_qty = qty if side == "buy" else -qty
            fill_time_ms = _to_float(fill.get("time"), 0.0)

            if position is None:
                position = {
                    "direction": 1 if signed_qty > 0 else -1,
                    "size": abs(signed_qty),
                    "entry_price": px,
                    "open_time_ms": fill_time_ms,
                    "signal_confidence": _extract_signal_confidence(fill),
                    "market_condition": _extract_market_condition(fill),
                }
                continue

            direction = int(position["direction"])
            current_size = _to_float(position["size"])

            if direction * signed_qty > 0:
                new_size = current_size + abs(signed_qty)
                position["entry_price"] = ((position["entry_price"] * current_size) + (px * abs(signed_qty))) / new_size
                position["size"] = new_size
                continue

            close_qty = min(current_size, abs(signed_qty))
            entry = _to_float(position["entry_price"])
            pnl = (px - entry) * close_qty if direction > 0 else (entry - px) * close_qty
            notional = entry * close_qty
            pnl_pct = (pnl / notional * 100.0) if notional > 0 else 0.0
            hold_minutes = max(0.0, (fill_time_ms - _to_float(position["open_time_ms"])) / 60000.0)

            trades.append(
                {
                    "open_time": _to_iso_utc(position["open_time_ms"]),
                    "close_time": _to_iso_utc(fill_time_ms),
                    "side": "long" if direction > 0 else "short",
                    "entry_price": round(entry, 8),
                    "exit_price": round(px, 8),
                    "size": round(close_qty, 8),
                    "pnl": round(pnl, 8),
                    "pnl_pct": round(pnl_pct, 8),
                    "hold_duration_minutes": round(hold_minutes, 4),
                    "signal_confidence": position.get("signal_confidence"),
                    "market_condition": position.get("market_condition"),
                }
            )

            remainder = abs(signed_qty) - close_qty
            if remainder <= 1e-12:
                if close_qty >= current_size - 1e-12:
                    position = None
                else:
                    position["size"] = current_size - close_qty
            else:
                position = {
                    "direction": 1 if signed_qty > 0 else -1,
                    "size": remainder,
                    "entry_price": px,
                    "open_time_ms": fill_time_ms,
                    "signal_confidence": _extract_signal_confidence(fill),
                    "market_condition": _extract_market_condition(fill),
                }

        completed[symbol] = trades

    return completed


def update_state_and_flags(symbol_trades: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    TRADE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "updated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "total_trades": 0,
        "symbols": {},
    }

    total = 0
    for symbol, trades in sorted(symbol_trades.items()):
        path = TRADE_HISTORY_DIR / f"{symbol}.json"
        path.write_text(json.dumps(trades, indent=2), encoding="utf-8")
        count = len(trades)
        total += count

        flag_path = TRADE_HISTORY_DIR / f"{symbol}.flag"
        flag_created = False
        if count >= 10 and not flag_path.exists():
            flag_path.write_text(
                f"symbol={symbol}\nreached_trades={count}\ncreated_at={state['updated_at']}\n",
                encoding="utf-8",
            )
            flag_created = True

        state["symbols"][symbol] = {
            "trade_count": count,
            "flag_exists": flag_path.exists(),
            "flag_created_this_run": flag_created,
        }

    state["total_trades"] = total
    ANALYSIS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main() -> None:
    fills = fetch_user_fills()
    symbol_trades = build_completed_trades(fills)
    state = update_state_and_flags(symbol_trades)
    print(f"Processed fills into completed trades for {len(symbol_trades)} symbols; total trades={state['total_trades']}")


if __name__ == "__main__":
    main()
