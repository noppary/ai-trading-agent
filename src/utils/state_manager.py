"""P3.1: SQLite-backed state persistence with JSON file fallback.

All state is stored in a single SQLite database, namespaced by HYPERLIQUID_NETWORK.
The public API is identical to the previous JSON-file implementation so callers
(main.py) don't need changes.  JSONL diary remains file-based (append-only log).
"""

import json
import os
import logging
import sqlite3
from typing import Any
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Network-aware DB path ────────────────────────────────────────────────────

def _get_network() -> str:
    return os.getenv("HYPERLIQUID_NETWORK", "mainnet").lower()


def _db_path() -> str:
    return f"state_{_get_network()}.db"


# ── DB connection (module-level singleton per process) ────────────────────────

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = _db_path()
        _conn = sqlite3.connect(path, timeout=10)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _init_tables(_conn)
        log.info("SQLite state DB opened: %s", path)
    return _conn


def _init_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kv (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trade_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            data       TEXT NOT NULL,
            logged_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS diary (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            data       TEXT NOT NULL,
            timestamp  TEXT NOT NULL
        );
    """)
    conn.commit()


# ── Generic KV helpers ───────────────────────────────────────────────────────

def _kv_get(key: str, default: Any = None) -> Any:
    conn = _get_conn()
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return default


def _kv_set(key: str, value: Any) -> None:
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, json.dumps(value, default=str), now),
    )
    conn.commit()


# ── Circuit-breaker state ────────────────────────────────────────────────────

_CIRCUIT_DEFAULTS = {
    "initial_account_value": None,
    "peak_account_value": None,
    "daily_start_value": None,
    "daily_start_date": None,
    "trading_halted": False,
    "halted_reason": None,
    "last_updated": None,
}


def load_circuit_state() -> dict:
    """Load persisted circuit-breaker state. Returns defaults if not found."""
    state = _kv_get("circuit_state")
    if isinstance(state, dict):
        return {**_CIRCUIT_DEFAULTS, **state}
    return dict(_CIRCUIT_DEFAULTS)


def save_circuit_state(state: dict) -> None:
    """Persist circuit-breaker state."""
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    _kv_set("circuit_state", state)


# ── Active trades ────────────────────────────────────────────────────────────

def load_active_trades() -> list:
    """Load persisted active_trades list."""
    trades = _kv_get("active_trades")
    return trades if isinstance(trades, list) else []


def save_active_trades(trades: list) -> None:
    """Persist active_trades."""
    _kv_set("active_trades", trades)


# ── Trade log (for Sharpe) ───────────────────────────────────────────────────

def load_trade_log() -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT data FROM trade_log ORDER BY id DESC LIMIT 1000"
    ).fetchall()
    result = []
    for row in reversed(rows):
        try:
            result.append(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError):
            continue
    return result


def append_trade_log_entry(entry: dict) -> None:
    """Append one trade result to the log."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    entry_with_ts = {**entry, "logged_at": now}
    conn.execute(
        "INSERT INTO trade_log (data, logged_at) VALUES (?, ?)",
        (json.dumps(entry_with_ts, default=str), now),
    )
    conn.commit()
    # Prune old entries beyond 5000
    conn.execute(
        "DELETE FROM trade_log WHERE id NOT IN (SELECT id FROM trade_log ORDER BY id DESC LIMIT 5000)"
    )
    conn.commit()


# ── Diary (queryable copy — JSONL file remains the write-ahead log) ──────────

def append_diary_entry(entry: dict) -> None:
    """Insert a diary entry into SQLite for querying."""
    conn = _get_conn()
    ts = entry.get("timestamp", datetime.now(timezone.utc).isoformat())
    conn.execute(
        "INSERT INTO diary (data, timestamp) VALUES (?, ?)",
        (json.dumps(entry, default=str), ts),
    )
    conn.commit()


def query_diary(limit: int = 200, asset: str | None = None) -> list:
    """Query diary entries, optionally filtered by asset."""
    conn = _get_conn()
    if asset:
        rows = conn.execute(
            "SELECT data FROM diary WHERE json_extract(data, '$.asset') = ? ORDER BY id DESC LIMIT ?",
            (asset, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT data FROM diary ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for row in reversed(rows):
        try:
            result.append(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError):
            continue
    return result


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


# ── Migration: import existing JSON files into SQLite on first run ───────────

def _fn(base: str) -> str:
    """Legacy network-namespaced filename for migration."""
    net = _get_network()
    if base.startswith(f"{net}_"):
        return base
    parts = base.rsplit(".", 1)
    return f"{parts[0]}_{net}.{parts[1]}"


def migrate_from_json():
    """One-time migration: load existing JSON state files into SQLite."""
    migrated = _kv_get("_migrated_from_json")
    if migrated:
        return

    log.info("Migrating JSON state files to SQLite...")

    # Circuit state
    circuit_path = _fn("circuit_state.json")
    if os.path.exists(circuit_path):
        try:
            with open(circuit_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                save_circuit_state(data)
                log.info("Migrated circuit_state from %s", circuit_path)
        except Exception as e:
            log.warning("Failed to migrate %s: %s", circuit_path, e)

    # Active trades
    trades_path = _fn("active_trades.json")
    if os.path.exists(trades_path):
        try:
            with open(trades_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                save_active_trades(data)
                log.info("Migrated active_trades from %s", trades_path)
        except Exception as e:
            log.warning("Failed to migrate %s: %s", trades_path, e)

    # Trade log
    tlog_path = _fn("trade_log.json")
    if os.path.exists(tlog_path):
        try:
            with open(tlog_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data[-1000:]:
                    append_trade_log_entry(entry)
                log.info("Migrated %d trade_log entries from %s", min(len(data), 1000), tlog_path)
        except Exception as e:
            log.warning("Failed to migrate %s: %s", tlog_path, e)

    # Diary
    diary_path = "diary.jsonl"
    if os.path.exists(diary_path):
        try:
            count = 0
            with open(diary_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            append_diary_entry(entry)
                            count += 1
                        except json.JSONDecodeError:
                            continue
            log.info("Migrated %d diary entries from %s", count, diary_path)
        except Exception as e:
            log.warning("Failed to migrate %s: %s", diary_path, e)

    _kv_set("_migrated_from_json", True)
    log.info("JSON → SQLite migration complete")
