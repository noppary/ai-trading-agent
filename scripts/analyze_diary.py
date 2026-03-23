#!/usr/bin/env python3
"""Analyze diary.jsonl for pathological patterns when trade history is sparse.

Reads diary entries, counts action distributions, extracts rationale patterns,
and flags issues like excessive hold rates or repeated identical rationales.
Writes findings to data/learnings.json patterns array.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

DIARY_PATH = Path("/root/ai-trading-agent/diary.jsonl")
LEARNINGS_PATH = Path("/root/ai-trading-agent/data/learnings.json")


def load_diary(hours: float = 4.0) -> list[dict]:
    """Load diary entries from the last N hours."""
    if not DIARY_PATH.exists():
        return []
    cutoff = datetime.now(UTC).timestamp() - (hours * 3600)
    entries = []
    with open(DIARY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Try to parse timestamp
            ts = entry.get("timestamp") or entry.get("time") or entry.get("ts")
            if ts:
                try:
                    if isinstance(ts, (int, float)):
                        entry_ts = ts if ts > 1e12 else ts * 1000
                        entry_ts = entry_ts / 1000
                    else:
                        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        entry_ts = dt.timestamp()
                    if entry_ts < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            entries.append(entry)
    return entries


def analyze_entries(entries: list[dict]) -> dict:
    """Analyze diary entries for patterns and pathologies."""
    if not entries:
        return {
            "total_entries": 0,
            "period_hours": 0,
            "action_distribution": {},
            "hold_ratio": 0,
            "per_asset": {},
            "pathologies": ["No diary entries found"],
            "top_rationales": [],
        }

    # Count actions per asset
    asset_actions: dict[str, Counter] = {}
    rationales: list[str] = []
    failed_orders = 0
    total_decisions = 0

    for entry in entries:
        # Support both flat format (asset/action at top level) and nested trade_decisions
        decisions = entry.get("trade_decisions") or entry.get("decisions") or []
        if isinstance(decisions, dict):
            decisions = [decisions]
        # If flat format (single asset/action per entry), wrap it
        if not decisions and entry.get("action"):
            decisions = [entry]

        for dec in decisions:
            asset = (dec.get("asset") or dec.get("symbol") or "UNKNOWN").upper()
            action = (dec.get("action") or "hold").lower()
            if asset not in asset_actions:
                asset_actions[asset] = Counter()
            asset_actions[asset][action] += 1
            total_decisions += 1

            rationale = dec.get("rationale") or dec.get("reasoning") or ""
            if rationale:
                # Normalize: lowercase, strip whitespace, collapse spaces
                norm = re.sub(r"\s+", " ", rationale.lower().strip())
                rationales.append(norm)

        # Check for errors/failed orders
        error = entry.get("error") or entry.get("execution_error")
        if error:
            failed_orders += 1

    # Overall action distribution
    total_actions = Counter()
    for ac in asset_actions.values():
        total_actions.update(ac)

    hold_count = total_actions.get("hold", 0)
    hold_ratio = hold_count / total_decisions if total_decisions > 0 else 0

    # Per-asset stats
    per_asset = {}
    for asset, actions in asset_actions.items():
        asset_total = sum(actions.values())
        per_asset[asset] = {
            "total": asset_total,
            "actions": dict(actions),
            "hold_ratio": actions.get("hold", 0) / asset_total if asset_total > 0 else 0,
        }

    # Top rationale patterns (find repeated phrases)
    rationale_counter = Counter()
    for r in rationales:
        # Extract key phrases (first 80 chars as fingerprint)
        key = r[:80]
        rationale_counter[key] += 1
    top_rationales = rationale_counter.most_common(10)

    # Detect pathologies
    pathologies = []
    if hold_ratio > 0.95:
        pathologies.append(f"CRITICAL: Hold ratio {hold_ratio:.1%} — bot is not trading at all")
    elif hold_ratio > 0.90:
        pathologies.append(f"WARNING: Hold ratio {hold_ratio:.1%} — bot is excessively conservative")

    for phrase, count in top_rationales[:3]:
        if count > max(total_decisions * 0.3, 10):
            pathologies.append(f"Repeated rationale ({count}x): '{phrase[:60]}...'")

    if failed_orders > 0:
        pathologies.append(f"Failed order attempts: {failed_orders}")

    for asset, stats in per_asset.items():
        if stats["hold_ratio"] > 0.98 and stats["total"] > 20:
            pathologies.append(f"{asset}: {stats['hold_ratio']:.0%} hold rate over {stats['total']} decisions")

    return {
        "total_entries": len(entries),
        "total_decisions": total_decisions,
        "action_distribution": dict(total_actions),
        "hold_ratio": round(hold_ratio, 4),
        "per_asset": per_asset,
        "pathologies": pathologies,
        "top_rationales": [{"phrase": p, "count": c} for p, c in top_rationales[:10]],
        "failed_orders": failed_orders,
    }


def write_findings_to_learnings(analysis: dict) -> None:
    """Append pathological findings to learnings.json patterns array."""
    if not analysis.get("pathologies"):
        return

    try:
        learnings = json.loads(LEARNINGS_PATH.read_text(encoding="utf-8")) if LEARNINGS_PATH.exists() else {}
    except json.JSONDecodeError:
        learnings = {}

    patterns = learnings.get("patterns", [])
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    for pathology in analysis["pathologies"]:
        patterns.append({
            "timestamp": now,
            "symbol": "ALL",
            "action": "diary_pattern",
            "value": pathology,
            "approved_by": "self_improve",
        })

    learnings["patterns"] = patterns
    learnings.setdefault("strategy_changes", [])
    learnings.setdefault("market_correlations", [])
    LEARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEARNINGS_PATH.write_text(json.dumps(learnings, indent=2, ensure_ascii=False), encoding="utf-8")


def main(hours: float = 4.0) -> dict:
    entries = load_diary(hours)
    analysis = analyze_entries(entries)
    write_findings_to_learnings(analysis)
    return analysis


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    result = main(hours)
    print(json.dumps(result, indent=2))
