#!/usr/bin/env python3
"""Fetch Hyperliquid testnet fills and persist completed trades by symbol."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

API_URL = "https://api.hyperliquid-testnet.xyz/info"
USER_ADDRESS = "0xE332bff7eBCAC0160468654F79B6fA3176638256"

DATA_DIR = Path("/root/ai-trading-agent/data")
TRADE_HISTORY_DIR = DATA_DIR / "trade_history"
ANALYSIS_STATE_PATH = DATA_DIR / "analysis_state.json"


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
    side_value = str(fill.get("side") or fill.get("dir") or fill.get("direction") or "").lower()
    if "sell" in side_value or side_value == "s":
        return "sell"
    return "buy"


def _extract_market_condition(fill: dict[str, Any]) -> str:
    for key in ("market_condition", "marketCondition", "regime", "volatility"):
        value = fill.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def _extract_signal_confidence(fill: dict[str, Any]) -> float | None:
    for key in ("signal_confidence", "signalConfidence", "confidence"):
        value = fill.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _normalize_symbol(fill: dict[str, Any]) -> str:
    return str(fill.get("coin") or fill.get("symbol") or fill.get("asset") or "").upper()


def fetch_user_fills() -> list[dict[str, Any]]:
    response = requests.post(
        API_URL,
        json={"type": "userFills", "user": USER_ADDRESS},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected Hyperliquid response for userFills")
    return payload


def _new_position(fill: dict[str, Any], signed_size: float, price: float, fill_time_ms: float) -> dict[str, Any]:
    return {
        "direction": 1 if signed_size > 0 else -1,
        "size": abs(signed_size),
        "entry_price": price,
        "open_time_ms": fill_time_ms,
        "signal_confidence": _extract_signal_confidence(fill),
        "market_condition": _extract_market_condition(fill),
    }


def build_completed_trades(fills: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    fills_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for fill in fills:
        symbol = _normalize_symbol(fill)
        if symbol:
            fills_by_symbol.setdefault(symbol, []).append(fill)

    completed_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for symbol, symbol_fills in fills_by_symbol.items():
        symbol_fills.sort(key=lambda item: _to_float(item.get("time")))
        completed: list[dict[str, Any]] = []
        position: dict[str, Any] | None = None

        for fill in symbol_fills:
            price = _to_float(fill.get("px") or fill.get("price"))
            size = abs(_to_float(fill.get("sz") or fill.get("size")))
            if price <= 0 or size <= 0:
                continue

            fill_side = _side_from_fill(fill)
            signed_size = size if fill_side == "buy" else -size
            fill_time_ms = _to_float(fill.get("time"))

            if position is None:
                position = _new_position(fill, signed_size, price, fill_time_ms)
                continue

            direction = int(position["direction"])
            current_size = _to_float(position["size"])

            if direction * signed_size > 0:
                new_size = current_size + abs(signed_size)
                weighted_price = ((position["entry_price"] * current_size) + (price * abs(signed_size))) / new_size
                position["entry_price"] = weighted_price
                position["size"] = new_size
                if position.get("signal_confidence") is None:
                    position["signal_confidence"] = _extract_signal_confidence(fill)
                if position.get("market_condition") in (None, "", "unknown"):
                    position["market_condition"] = _extract_market_condition(fill)
                continue

            closed_size = min(current_size, abs(signed_size))
            entry_price = _to_float(position["entry_price"])
            pnl = (price - entry_price) * closed_size if direction > 0 else (entry_price - price) * closed_size
            notional = entry_price * closed_size
            pnl_pct = (pnl / notional * 100.0) if notional > 0 else 0.0
            hold_minutes = max(0.0, (fill_time_ms - _to_float(position["open_time_ms"])) / 60000.0)

            completed.append(
                {
                    "open_time": _to_iso_utc(position["open_time_ms"]),
                    "close_time": _to_iso_utc(fill_time_ms),
                    "side": "long" if direction > 0 else "short",
                    "entry_price": round(entry_price, 8),
                    "exit_price": round(price, 8),
                    "size": round(closed_size, 8),
                    "pnl": round(pnl, 8),
                    "pnl_pct": round(pnl_pct, 8),
                    "hold_duration_minutes": round(hold_minutes, 4),
                    "signal_confidence": position.get("signal_confidence"),
                    "market_condition": position.get("market_condition", "unknown"),
                }
            )

            remaining_size = abs(signed_size) - closed_size
            if closed_size >= current_size - 1e-12:
                position = None
            else:
                position["size"] = current_size - closed_size

            if remaining_size > 1e-12:
                position = _new_position(fill, signed_size, price, fill_time_ms)
                position["size"] = remaining_size

        completed_by_symbol[symbol] = completed

    return completed_by_symbol


def update_state_and_flags(symbol_trades: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRADE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "updated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "total_trades": 0,
        "symbols": {},
    }

    total_trades = 0
    for symbol, trades in sorted(symbol_trades.items()):
        history_path = TRADE_HISTORY_DIR / f"{symbol}.json"
        history_path.write_text(json.dumps(trades, indent=2, ensure_ascii=False), encoding="utf-8")

        trade_count = len(trades)
        total_trades += trade_count
        flag_path = TRADE_HISTORY_DIR / f"{symbol}.flag"
        flag_created = False
        if trade_count >= 10 and not flag_path.exists():
            flag_path.write_text(
                f"symbol={symbol}\ntrade_count={trade_count}\ncreated_at={state['updated_at']}\n",
                encoding="utf-8",
            )
            flag_created = True

        state["symbols"][symbol] = {
            "trade_count": trade_count,
            "flag_exists": flag_path.exists(),
            "flag_created_this_run": flag_created,
        }

    state["total_trades"] = total_trades
    ANALYSIS_STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return state


def main() -> None:
    fills = fetch_user_fills()
    symbol_trades = build_completed_trades(fills)
    state = update_state_and_flags(symbol_trades)
    print(
        f"saved completed trades for {len(symbol_trades)} symbols "
        f"with {state['total_trades']} completed trades"
    )


if __name__ == "__main__":
    main()
