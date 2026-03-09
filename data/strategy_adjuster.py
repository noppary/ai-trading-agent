#!/usr/bin/env python3
"""Apply strategy adjustments with versioned snapshots."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATA_DIR = Path("/root/ai-trading-agent/data")
LEARNINGS_PATH = DATA_DIR / "learnings.json"
STRATEGY_MD_PATH = Path("/root/.openclaw/workspace/STRATEGY_ADJUSTMENTS.md")
VERSIONS_DIR = DATA_DIR / "strategy_versions"

STRATEGY_MD_TEMPLATE = """# STRATEGY ADJUSTMENTS

## Active Rules
- (none)

## Retired Rules
- (none)

## Pattern Log
| Timestamp (UTC) | Symbol | Action | Value | Approved By |
|---|---|---|---|---|
"""


def ensure_learnings_file() -> None:
    LEARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LEARNINGS_PATH.exists():
        LEARNINGS_PATH.write_text(
            json.dumps({"patterns": [], "strategy_changes": [], "market_correlations": []}, indent=2),
            encoding="utf-8",
        )


def ensure_strategy_md() -> None:
    STRATEGY_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STRATEGY_MD_PATH.exists():
        STRATEGY_MD_PATH.write_text(STRATEGY_MD_TEMPLATE, encoding="utf-8")


def create_snapshot(symbol: str, action: str) -> Path:
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_symbol = re.sub(r"[^A-Za-z0-9_.-]", "_", symbol.upper())
    safe_action = re.sub(r"[^A-Za-z0-9_.-]", "_", action.lower())
    snap_dir = VERSIONS_DIR / f"{ts}_{safe_symbol}_{safe_action}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    if LEARNINGS_PATH.exists():
        shutil.copy2(LEARNINGS_PATH, snap_dir / "learnings.json")
    if STRATEGY_MD_PATH.exists():
        shutil.copy2(STRATEGY_MD_PATH, snap_dir / "STRATEGY_ADJUSTMENTS.md")
    return snap_dir


def _load_learnings() -> dict[str, Any]:
    try:
        data = json.loads(LEARNINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        data = {"patterns": [], "strategy_changes": [], "market_correlations": []}
    for key in ("patterns", "strategy_changes", "market_correlations"):
        if key not in data or not isinstance(data[key], list):
            data[key] = []
    return data


def _insert_bullet_under_heading(text: str, heading: str, bullet: str) -> str:
    marker = f"## {heading}\n"
    if marker not in text:
        text += f"\n{marker}- {bullet}\n"
        return text
    start = text.index(marker) + len(marker)
    return text[:start] + f"- {bullet}\n" + text[start:]


def _append_pattern_log_row(text: str, row: str) -> str:
    if "| Timestamp (UTC) | Symbol | Action | Value | Approved By |" not in text:
        text += "\n## Pattern Log\n| Timestamp (UTC) | Symbol | Action | Value | Approved By |\n|---|---|---|---|---|\n"
    if not text.endswith("\n"):
        text += "\n"
    return text + row + "\n"


def apply_adjustment(symbol: str, action: str, value: str, approved_by: str) -> None:
    ensure_learnings_file()
    ensure_strategy_md()
    snapshot_dir = create_snapshot(symbol, action)

    now_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    entry = {
        "timestamp": now_utc,
        "symbol": symbol.upper(),
        "action": action,
        "value": value,
        "approved_by": approved_by,
    }

    learnings = _load_learnings()
    lower_action = action.lower()
    if "pattern" in lower_action:
        learnings["patterns"].append(entry)
    elif "correlation" in lower_action:
        learnings["market_correlations"].append(entry)
    else:
        learnings["strategy_changes"].append(entry)
    LEARNINGS_PATH.write_text(json.dumps(learnings, indent=2, ensure_ascii=False), encoding="utf-8")

    md_text = STRATEGY_MD_PATH.read_text(encoding="utf-8")
    bullet = f"{entry['symbol']} | {action} | {value} | approved by {approved_by} | {now_utc}"
    if lower_action in {"add_rule", "activate_rule", "new_rule"}:
        md_text = _insert_bullet_under_heading(md_text, "Active Rules", bullet)
    if lower_action in {"retire_rule", "deactivate_rule"}:
        md_text = _insert_bullet_under_heading(md_text, "Retired Rules", bullet)
    row = f"| {now_utc} | {entry['symbol']} | {action} | {value} | {approved_by} |"
    md_text = _append_pattern_log_row(md_text, row)
    STRATEGY_MD_PATH.write_text(md_text, encoding="utf-8")

    print(f"Snapshot: {snapshot_dir}")
    print("Applied strategy adjustment successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjust strategy with audit snapshots")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. BTC")
    parser.add_argument("--action", required=True, help="Adjustment action, e.g. add_rule/retire_rule/log_pattern")
    parser.add_argument("--value", required=True, help="Action payload or rule text")
    parser.add_argument("--approved-by", required=True, dest="approved_by", help="Approver name/id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_adjustment(args.symbol, args.action, args.value, args.approved_by)


if __name__ == "__main__":
    main()
