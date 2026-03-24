"""Atomic state persistence — all files use write-to-temp + rename for crash safety.
Files are namespaced by HYPERLIQUID_NETWORK so testnet never pollutes mainnet."""

import json
import os
import logging
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Network-aware filename resolution ─────────────────────────────────────────

def _get_network() -> str:
    """Detect network from environment. Defaults to 'mainnet'."""
    return os.getenv("HYPERLIQUID_NETWORK", "mainnet").lower()


def _fn(base: str) -> str:
    """Return network-namespaced filename. e.g. circuit_state.json → mainnet_circuit_state.json."""
    net = _get_network()
    # Strip any existing network prefix to avoid double-prefixing
    if base.startswith(f"{net}_"):
        return base
    parts = base.rsplit(".", 1)
    return f"{parts[0]}_{net}.{parts[1]}"


# ── Atomic write helpers ──────────────────────────────────────────────────────

def _atomic_write(path: str, data: dict) -> None:
    """Write dict to JSON atomically: write to .tmp, then rename."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _safe_read(path: str, default: Any) -> Any:
    """Read JSON file, return default if missing or corrupt."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to read %s: %s — returning default", path, e)
        return default


# ── Circuit-breaker state ────────────────────────────────────────────────────

_STATE_FILE = "circuit_state.json"


def load_circuit_state() -> dict:
    """Load persisted circuit-breaker state. Returns defaults if file missing."""
    return _safe_read(_fn(_STATE_FILE), {
        "initial_account_value": None,
        "peak_account_value": None,
        "daily_start_value": None,
        "daily_start_date": None,   # ISO date string
        "trading_halted": False,
        "halted_reason": None,
        "last_updated": None,
    })


def save_circuit_state(state: dict) -> None:
    """Persist circuit-breaker state atomically."""
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(_fn(_STATE_FILE), state)


# ── Active trades ────────────────────────────────────────────────────────────

_TRADES_FILE = "active_trades.json"


def load_active_trades() -> list:
    """Load persisted active_trades list."""
    return _safe_read(_fn(_TRADES_FILE), [])


def save_active_trades(trades: list) -> None:
    """Persist active_trades atomically."""
    _atomic_write(_fn(_TRADES_FILE), trades)


# ── Trade log (for Sharpe) ───────────────────────────────────────────────────

_TRADE_LOG_FILE = "trade_log.json"


def load_trade_log() -> list:
    return _safe_read(_fn(_TRADE_LOG_FILE), [])


def append_trade_log_entry(entry: dict) -> None:
    """Append one trade result to the log. Atomic append."""
    trade_log = load_trade_log()
    trade_log.append({**entry, "logged_at": datetime.now(timezone.utc).isoformat()})
    _atomic_write(_fn(_TRADE_LOG_FILE), trade_log[-1000:])  # keep last 1000 entries


# ── Startup reconciliation ───────────────────────────────────────────────────

def build_recovery_report(
    exchange_positions: list,
    exchange_orders: list,
    persisted_trades: list,
) -> dict:
    """
    Compare exchange state vs persisted active_trades.
    Returns a dict with:
      - orphaned_persisted: trades in persisted_trades but no corresponding exchange position
      - missing_from_persisted: exchange positions not tracked in persisted_trades
      - orders_missing_tp_sl: positions without a TP or SL order
    """
    exchange_coins = {p.get("coin") for p in exchange_positions if float(p.get("szi", 0) or 0) != 0}
    order_coins = {o.get("coin") for o in exchange_orders if o.get("coin")}

    tp_sl_coins = set()
    for o in exchange_orders:
        coin = o.get("coin")
        ot = o.get("orderType", {})
        if isinstance(ot, dict) and "trigger" in ot:
            tpsl = ot.get("trigger", {}).get("tpsl", "")
            if tpsl in ("tp", "sl"):
                tp_sl_coins.add(coin)

    orphaned_persisted = [
        t for t in persisted_trades
        if t.get("asset") not in exchange_coins and t.get("asset") not in order_coins
    ]
    missing_from_persisted = [
        p for p in exchange_positions
        if float(p.get("szi", 0) or 0) != 0
        and p.get("coin") not in {t.get("asset") for t in persisted_trades}
    ]
    orders_missing_tp_sl = [
        p for p in exchange_positions
        if float(p.get("szi", 0) or 0) != 0
        and p.get("coin") not in tp_sl_coins
    ]

    return {
        "orphaned_persisted": orphaned_persisted,
        "missing_from_persisted": missing_from_persisted,
        "orders_missing_tp_sl": orders_missing_tp_sl,
        "exchange_position_coins": list(exchange_coins),
        "exchange_order_coins": list(order_coins),
        "tracked_coins": [t.get("asset") for t in persisted_trades],
    }
