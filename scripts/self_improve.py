#!/usr/bin/env python3
"""Self-improvement engine for the AI trading bot.

Runs the full cycle:
1. Collect: fetch fills + run diary analysis
2. Compute metrics from trade history
3. Recommend parameter adjustments
4. Apply safe changes (within bounds), backup first
5. Record via strategy_adjuster
6. Output JSON summary for Hana
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Paths
PROJECT_DIR = Path("/root/ai-trading-agent")
DATA_DIR = PROJECT_DIR / "data"
ENV_PATH = PROJECT_DIR / ".env"
DIARY_PATH = PROJECT_DIR / "diary.jsonl"
TRADE_HISTORY_DIR = DATA_DIR / "trade_history"
LEARNINGS_PATH = DATA_DIR / "learnings.json"

# Import sibling scripts
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(PROJECT_DIR / "data"))

# Safety bounds for auto-adjustment
SAFE_BOUNDS = {
    "MAX_LEVERAGE":            {"min": 2,  "max": 7,   "type": int},
    "MAX_POSITION_PCT":        {"min": 10, "max": 30,  "type": int},
    "MAX_TOTAL_EXPOSURE_PCT":  {"min": 40, "max": 80,  "type": int},
    "DAILY_LOSS_LIMIT_USD":    {"min": 20, "max": 100, "type": int},
    "MAX_DRAWDOWN_PCT":        {"min": 8,  "max": 20,  "type": int},
}

# Keys that must NEVER be touched
FORBIDDEN_PATTERNS = re.compile(
    r"(KEY|SECRET|PRIVATE|PASSWORD|TOKEN|NETWORK|MNEMONIC)", re.IGNORECASE
)


def load_env() -> dict[str, str]:
    """Parse .env into a dict."""
    env = {}
    if not ENV_PATH.exists():
        return env
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def backup_env() -> Path:
    """Create timestamped backup of .env."""
    backup = ENV_PATH.with_suffix(f".env.bak.{int(time.time())}")
    shutil.copy2(ENV_PATH, backup)
    return backup


def write_env(env: dict[str, str]) -> None:
    """Write env dict back to .env, preserving comments and order."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    updated_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in env:
                new_lines.append(f"{key}={env[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)
    # Add any new keys
    for k, v in env.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def validate_env() -> bool:
    """Validate .env by attempting to load config."""
    result = subprocess.run(
        ["python3", "-c", "from src.config_loader import CONFIG; print('ok')"],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.returncode == 0 and "ok" in result.stdout


def collect_trade_data() -> dict[str, list[dict]]:
    """Run trade_logger to fetch fills, return completed trades by symbol."""
    try:
        subprocess.run(
            ["python3", "data/trade_logger.py"],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        pass

    trades_by_symbol: dict[str, list[dict]] = {}
    if TRADE_HISTORY_DIR.exists():
        for f in TRADE_HISTORY_DIR.glob("*.json"):
            symbol = f.stem
            try:
                trades = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(trades, list):
                    trades_by_symbol[symbol] = trades
            except (json.JSONDecodeError, OSError):
                pass
    return trades_by_symbol


def compute_metrics(trades_by_symbol: dict[str, list[dict]]) -> dict[str, Any]:
    """Compute per-asset and overall trading metrics."""
    overall = {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "max_losing_streak": 0,
    }
    per_asset: dict[str, dict] = {}

    for symbol, trades in trades_by_symbol.items():
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losses = sum(1 for t in trades if t.get("pnl", 0) <= 0)
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        count = len(trades)

        # Losing streak
        max_streak = 0
        current_streak = 0
        for t in trades:
            if t.get("pnl", 0) <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # Confidence accuracy
        confident_correct = 0
        confident_total = 0
        for t in trades:
            conf = t.get("signal_confidence")
            if conf is not None and conf >= 0.5:
                confident_total += 1
                if t.get("pnl", 0) > 0:
                    confident_correct += 1

        per_asset[symbol] = {
            "trades": count,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / count if count > 0 else 0,
            "total_pnl": round(total_pnl, 4),
            "avg_pnl": round(total_pnl / count, 4) if count > 0 else 0,
            "max_losing_streak": max_streak,
            "confidence_accuracy": confident_correct / confident_total if confident_total > 0 else None,
        }

        overall["total_trades"] += count
        overall["wins"] += wins
        overall["losses"] += losses
        overall["total_pnl"] += total_pnl
        overall["max_losing_streak"] = max(overall["max_losing_streak"], max_streak)

    overall["win_rate"] = (
        overall["wins"] / overall["total_trades"]
        if overall["total_trades"] > 0
        else 0
    )
    overall["total_pnl"] = round(overall["total_pnl"], 4)

    return {"overall": overall, "per_asset": per_asset}


def generate_recommendations(
    metrics: dict[str, Any],
    diary_analysis: dict[str, Any],
    current_env: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Generate auto-apply and needs-approval recommendations."""
    auto_apply: list[dict] = []
    needs_approval: list[dict] = []

    overall = metrics.get("overall", {})
    per_asset = metrics.get("per_asset", {})

    # Rule: max losing streak > 5 → reduce MAX_LEVERAGE by 1
    if overall.get("max_losing_streak", 0) > 5:
        current_lev = int(current_env.get("MAX_LEVERAGE", "5"))
        new_lev = max(SAFE_BOUNDS["MAX_LEVERAGE"]["min"], current_lev - 1)
        if new_lev < current_lev:
            auto_apply.append({
                "param": "MAX_LEVERAGE",
                "old": current_lev,
                "new": new_lev,
                "reason": f"Max losing streak {overall['max_losing_streak']} > 5 — reducing leverage",
            })

    # Rule: win rate < 30% on asset with 20+ trades → remove from ASSETS
    current_assets = current_env.get("ASSETS", "").replace(",", " ").split()
    for symbol, stats in per_asset.items():
        if stats["trades"] >= 20 and stats["win_rate"] < 0.30:
            if symbol in current_assets:
                new_assets = [a for a in current_assets if a != symbol]
                if new_assets:  # Don't remove all assets
                    auto_apply.append({
                        "param": "ASSETS",
                        "old": " ".join(current_assets),
                        "new": " ".join(new_assets),
                        "reason": f"{symbol} win rate {stats['win_rate']:.0%} over {stats['trades']} trades — removing",
                    })

    # Rule: hold ratio > 90% → flag (prompt fix, not .env)
    hold_ratio = diary_analysis.get("hold_ratio", 0)
    if hold_ratio > 0.90:
        needs_approval.append({
            "type": "flag",
            "issue": f"Hold ratio {hold_ratio:.1%} — bot may be too conservative",
            "suggestion": "Stage 2 prompt may need further tuning to reduce hold bias",
        })

    # Rule: confidence accuracy < 40%
    for symbol, stats in per_asset.items():
        if stats.get("confidence_accuracy") is not None and stats["confidence_accuracy"] < 0.40 and stats["trades"] >= 10:
            needs_approval.append({
                "type": "flag",
                "issue": f"{symbol} confidence accuracy {stats['confidence_accuracy']:.0%} — signal model may be miscalibrated",
                "suggestion": "Review Stage 2 signal classification for this asset",
            })

    return auto_apply, needs_approval


def apply_changes(changes: list[dict], current_env: dict[str, str]) -> list[dict]:
    """Apply auto-approved changes to .env within safety bounds."""
    if not changes:
        return []

    applied = []
    backup_path = backup_env()

    for change in changes:
        param = change["param"]

        # Safety check: never touch forbidden keys
        if FORBIDDEN_PATTERNS.search(param):
            change["status"] = "BLOCKED — forbidden parameter"
            continue

        new_value = change["new"]

        # Validate within bounds if applicable
        if param in SAFE_BOUNDS:
            bounds = SAFE_BOUNDS[param]
            try:
                val = bounds["type"](new_value)
                if val < bounds["min"] or val > bounds["max"]:
                    change["status"] = f"BLOCKED — {val} outside safe range [{bounds['min']}, {bounds['max']}]"
                    continue
                new_value = str(val)
            except (ValueError, TypeError):
                change["status"] = "BLOCKED — invalid value type"
                continue

        current_env[param] = str(new_value)
        change["status"] = "applied"
        applied.append(change)

    if applied:
        write_env(current_env)
        if not validate_env():
            # Rollback
            shutil.copy2(backup_path, ENV_PATH)
            for change in applied:
                change["status"] = "ROLLED BACK — config validation failed"
            return applied

    return applied


def record_changes(changes: list[dict]) -> None:
    """Record applied changes via strategy_adjuster."""
    for change in changes:
        if change.get("status") != "applied":
            continue
        try:
            subprocess.run(
                [
                    "python3", "data/strategy_adjuster.py",
                    "--symbol", "ALL",
                    "--action", f"auto_adjust_{change['param']}",
                    "--value", f"{change['old']} → {change['new']}: {change['reason']}",
                    "--approved-by", "self_improve",
                ],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:
            pass


def restart_service_if_needed(changes: list[dict]) -> bool:
    """Restart ai-trading-agent service if changes were applied."""
    applied = [c for c in changes if c.get("status") == "applied"]
    if not applied:
        return False
    try:
        result = subprocess.run(
            ["systemctl", "restart", "ai-trading-agent"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def main() -> dict[str, Any]:
    """Run the full self-improvement cycle."""
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    # 1. Collect trade data
    trades_by_symbol = collect_trade_data()

    # 2. Analyze diary (always — especially useful with few trades)
    from analyze_diary import main as analyze_diary_main
    diary_analysis = analyze_diary_main(hours=4.0)

    # 3. Compute metrics from completed trades
    metrics = compute_metrics(trades_by_symbol)

    # 4. Generate recommendations
    current_env = load_env()
    auto_apply, needs_approval = generate_recommendations(metrics, diary_analysis, current_env)

    # 5. Apply safe changes
    applied = apply_changes(auto_apply, current_env)

    # 6. Record changes
    record_changes(applied)

    # 7. Restart if needed
    restarted = restart_service_if_needed(applied)

    # Build summary
    summary = {
        "timestamp": now,
        "metrics": metrics,
        "diary_analysis": {
            "total_entries": diary_analysis.get("total_entries", 0),
            "hold_ratio": diary_analysis.get("hold_ratio", 0),
            "pathologies": diary_analysis.get("pathologies", []),
            "failed_orders": diary_analysis.get("failed_orders", 0),
        },
        "auto_applied": applied,
        "needs_approval": needs_approval,
        "service_restarted": restarted,
    }

    return summary


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
