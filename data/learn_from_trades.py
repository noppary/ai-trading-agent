#!/usr/bin/env python3
"""
Self-learning engine: analyzes closed trades, extracts patterns,
and updates learnings.json for the trading agent.

Run manually:
    python learn_from_trades.py [--days 7]

Integrated:
    Called after each decision cycle in main.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

BOT_DIR = Path("/root/ai-trading-agent")
DIARY_PATH = BOT_DIR / "diary.jsonl"
LEARNINGS_PATH = BOT_DIR / "data" / "learnings.json"
ANALYSIS_STATE_PATH = BOT_DIR / "data" / "analysis_state.json"
TRADE_HISTORY_DIR = BOT_DIR / "data" / "trade_history"

# ── Load helpers ─────────────────────────────────────────────────────────────

def load_learnings() -> dict[str, Any]:
    if LEARNINGS_PATH.exists():
        try:
            return json.loads(LEARNINGS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Could not load learnings.json: %s", e)
    return {"patterns": [], "strategy_changes": [], "market_correlations": []}


def save_learnings(data: dict[str, Any]) -> None:
    LEARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEARNINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved learnings to %s", LEARNINGS_PATH)


def load_analysis_state() -> dict[str, Any]:
    if ANALYSIS_STATE_PATH.exists():
        try:
            return json.loads(ANALYSIS_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "total_trades": 0,
        "symbols": {s: {"trade_count": 0, "flag_exists": False, "flag_created_this_run": False} for s in ["BTC", "ETH", "SOL", "BNB", "EIGEN"]},
    }


def save_analysis_state(state: dict[str, Any]) -> None:
    ANALYSIS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_trade_history(symbol: str) -> list[dict]:
    fpath = TRADE_HISTORY_DIR / f"{symbol.upper()}.json"
    if fpath.exists():
        try:
            return json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_trade_history(symbol: str, trades: list[dict]) -> None:
    TRADE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    fpath = TRADE_HISTORY_DIR / f"{symbol.upper()}.json"
    fpath.write_text(json.dumps(trades, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Parse diary ──────────────────────────────────────────────────────────────

def load_diary(days: int = 7) -> list[dict]:
    """Load recent diary entries (closed trades only)."""
    if not DIARY_PATH.exists():
        return []

    cutoff = datetime.now(UTC) - timedelta(days=days)
    closed_trades = []

    try:
        content = DIARY_PATH.read_text(encoding="utf-8")
        for line in reversed(content.strip().split("\n")):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = entry.get("timestamp", "")
            try:
                entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                # Normalize to UTC-aware
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=UTC)
            except (ValueError, TypeError):
                continue

            if entry_time < cutoff:
                break

            # Only closed trades (filled buy/sell, not hold, not dryrun)
            action = entry.get("action", "")
            if action in ("buy", "sell") and entry.get("filled") and not entry.get("dry_run"):
                closed_trades.append(entry)

    except IOError as e:
        logger.error("Failed to read diary: %s", e)

    return closed_trades


# ── Core analysis ────────────────────────────────────────────────────────────

def classify_outcome(entry: dict) -> str:
    """Classify trade outcome: win, loss, breakeven."""
    pnl = entry.get("unrealizedPnl") or entry.get("realizedPnl") or 0
    try:
        pnl = float(pnl)
    except (ValueError, TypeError):
        pnl = 0.0

    if pnl > 0.5:
        return "win"
    elif pnl < -0.5:
        return "loss"
    else:
        return "breakeven"


def extract_market_conditions(entry: dict) -> dict:
    """Extract market conditions from diary entry rationale or state."""
    rationale = entry.get("rationale", "") or ""
    tp = entry.get("tp_price") or 0
    sl = entry.get("sl_price") or 0
    entry_price = float(entry.get("entry_price") or 0)
    action = entry.get("action", "")

    # Determine market regime from rationale keywords
    regime = "unknown"
    if "bullish" in rationale.lower() or "uptrend" in rationale.lower():
        regime = "trending_up"
    elif "bearish" in rationale.lower() or "downtrend" in rationale.lower():
        regime = "trending_down"
    elif "ranging" in rationale.lower() or "neutral" in rationale.lower():
        regime = "ranging"
    elif "breakout" in rationale.lower():
        regime = "breakout"

    # RSI from rationale if available
    rsi_match = re.search(r"RSI[^\d]*(\d+\.?\d*)", rationale, re.IGNORECASE)
    rsi = float(rsi_match.group(1)) if rsi_match else None

    # Confidence (approximate from allocation vs max)
    alloc = float(entry.get("allocation_usd", 0))
    confidence = "low"
    if alloc >= 50:
        confidence = "high"
    elif alloc >= 20:
        confidence = "medium"

    return {
        "regime": regime,
        "rsi": rsi,
        "confidence": confidence,
        "has_tp": bool(tp),
        "has_sl": bool(sl),
        "risk_reward_estimate": _estimate_rr(tp, sl, entry_price, action),
    }


def _estimate_rr(tp, sl, entry, action) -> float | None:
    """Estimate risk:reward ratio from TP/SL distances."""
    try:
        tp, sl, entry = float(tp), float(sl), float(entry)
        if entry <= 0 or tp <= 0 or sl <= 0:
            return None
        if action == "sell":  # short
            risk = entry - sl  # sl above entry
            reward = tp - entry  # tp below entry
        else:  # buy / long
            risk = sl - entry  # sl below entry
            reward = entry - tp  # tp above entry
        if risk <= 0:
            return None
        return round(reward / risk, 2)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def derive_pattern(conditions: dict, outcome: str, symbol: str) -> dict:
    """Create a pattern entry from conditions + outcome."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "outcome": outcome,
        "regime": conditions["regime"],
        "rsi": conditions["rsi"],
        "confidence": conditions["confidence"],
        "has_tp": conditions["has_tp"],
        "has_sl": conditions["has_sl"],
        "rr": conditions["risk_reward_estimate"],
    }


def derive_strategy_change(conditions: dict, outcome: str, symbol: str) -> dict | None:
    """Derive a strategy change recommendation based on patterns."""

    # Strong win pattern: high confidence + trending + has TP/SL
    if outcome == "win" and conditions["confidence"] == "high" and conditions["regime"] in ("trending_up", "trending_down"):
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "change": f"Increase allocation in {conditions['regime']} regime for {symbol}",
            "pattern": "high_confidence_trend_follow",
            "approved": False,  # requires human review
        }

    # Loss pattern: breakeven in ranging with no TP/SL
    if outcome == "loss" and conditions["regime"] == "ranging" and not conditions["has_tp"]:
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "change": f"Set TP/SL mandatory for {symbol} — avoid ranging markets without exits",
            "pattern": "ranging_no_exit",
            "approved": False,
        }

    # Repeated loss: RSI at extremes in wrong regime
    if outcome == "loss" and conditions["rsi"] is not None:
        if conditions["regime"] == "trending_up" and conditions["rsi"] > 70:
            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "symbol": symbol,
                "change": f"Avoid buy signals when RSI > 70 in {conditions['regime']} — fading overbought in uptrend",
                "pattern": "overbought_fade_in_uptrend",
                "approved": False,
            }

    return None


# ── Update learnings ──────────────────────────────────────────────────────────

def update_learnings(closed_trades: list[dict]) -> dict[str, Any]:
    """Analyze closed trades and update learnings.json."""
    if not closed_trades:
        logger.info("No new closed trades to analyze")
        return load_learnings()

    learnings = load_learnings()
    new_patterns = []
    new_changes = []

    for trade in closed_trades:
        symbol = trade.get("asset", "UNKNOWN")
        outcome = classify_outcome(trade)
        conditions = extract_market_conditions(trade)

        # Build pattern
        pattern = derive_pattern(conditions, outcome, symbol)
        new_patterns.append(pattern)

        # Build strategy change recommendation
        change = derive_strategy_change(conditions, outcome, symbol)
        if change:
            new_changes.append(change)

        # Append to trade history
        trades = load_trade_history(symbol)
        trades.append({
            "timestamp": trade.get("timestamp"),
            "action": trade.get("action"),
            "amount": trade.get("amount"),
            "entry_price": trade.get("entry_price"),
            "tp": trade.get("tp_price"),
            "sl": trade.get("sl_price"),
            "outcome": outcome,
            "regime": conditions["regime"],
            "rsi": conditions["rsi"],
        })
        save_trade_history(symbol, trades[-100:])  # keep last 100

    # Deduplicate patterns (same regime + symbol + outcome = same pattern)
    existing = learnings.get("patterns", [])
    seen = {(p.get("regime"), p.get("symbol"), p.get("outcome")) for p in existing}
    for p in new_patterns:
        key = (p["regime"], p["symbol"], p["outcome"])
        if key not in seen:
            existing.append(p)
            seen.add(key)

    # Deduplicate strategy changes
    existing_changes = learnings.get("strategy_changes", [])
    seen_changes = {(c.get("change"), c.get("symbol")) for c in existing_changes}
    for c in new_changes:
        key = (c["change"], c["symbol"])
        if key not in seen_changes:
            existing_changes.append(c)
            seen_changes.add(key)

    # Keep only recent changes (last 20)
    learnings["patterns"] = existing[-50:]
    learnings["strategy_changes"] = existing_changes[-20:]

    # Update market correlations (simplified)
    learnings["market_correlations"] = _compute_correlations()

    save_learnings(learnings)
    return learnings


def _compute_correlations() -> list[dict]:
    """Compute basic correlations from trade history."""
    symbols = ["BTC", "ETH", "SOL", "BNB", "EIGEN"]
    correlations = []

    for sym in symbols:
        trades = load_trade_history(sym)
        if not trades:
            continue
        wins = sum(1 for t in trades if t.get("outcome") == "win")
        total = len(trades)
        win_rate = wins / total if total > 0 else 0
        correlations.append({
            "symbol": sym,
            "total_trades": total,
            "win_rate": round(win_rate, 3),
            "sample_period": "last_100_trades",
        })

    return correlations


# ── Main / CLI ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Self-learning engine: analyze trades and update learnings")
    parser.add_argument("--days", type=int, default=7, help="Days of diary to analyze (default: 7)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze but don't save")
    args = parser.parse_args()

    logger.info("=== Self-Learning Engine ===")
    closed = load_diary(days=args.days)
    logger.info("Found %d closed trades in last %d days", len(closed), args.days)

    if args.dry_run:
        for t in closed:
            print(f"  {t.get('timestamp')} | {t.get('asset')} | {t.get('action')} | filled={t.get('filled')} | pnl={t.get('unrealizedPnl',0)}")
        return

    learnings = update_learnings(closed)

    print("\n=== Updated Learnings ===")
    print(f"Patterns:    {len(learnings.get('patterns', []))}")
    print(f"Changes:     {len(learnings.get('strategy_changes', []))}")
    print(f"Correlations: {len(learnings.get('market_correlations', []))}")

    if learnings.get("strategy_changes"):
        print("\nPending strategy changes (need human approval):")
        for c in learnings["strategy_changes"][-5:]:
            print(f"  [{c['symbol']}] {c['change']}")


if __name__ == "__main__":
    main()
