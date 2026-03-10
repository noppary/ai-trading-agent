#!/usr/bin/env python3
"""Track strategy adjustments with mandatory snapshots."""

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
VERSIONS_DIR = DATA_DIR / "strategy_versions"
STRATEGY_MD_PATH = Path("/root/.openclaw/workspace/STRATEGY_ADJUSTMENTS.md")

LEARNINGS_TEMPLATE = {
    "patterns": [],
    "strategy_changes": [],
    "market_correlations": [],
}

STRATEGY_TEMPLATE = """# STRATEGY_ADJUSTMENTS

## Active Rules
- (none)

## Retired Rules
- (none)

## Pattern Log
| Timestamp (UTC) | Symbol | Action | Value | Approved By |
|---|---|---|---|---|
"""


def create_snapshot(symbol: str, action: str) -> Path:
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_symbol = re.sub(r"[^A-Za-z0-9_.-]", "_", symbol.upper())
    safe_action = re.sub(r"[^A-Za-z0-9_.-]", "_", action.lower())
    snapshot_dir = VERSIONS_DIR / f"{timestamp}_{safe_symbol}_{safe_action}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if LEARNINGS_PATH.exists():
        shutil.copy2(LEARNINGS_PATH, snapshot_dir / "learnings.json")
    else:
        (snapshot_dir / "learnings.json.missing").write_text("", encoding="utf-8")
    if STRATEGY_MD_PATH.exists():
        shutil.copy2(STRATEGY_MD_PATH, snapshot_dir / "STRATEGY_ADJUSTMENTS.md")
    else:
        (snapshot_dir / "STRATEGY_ADJUSTMENTS.md.missing").write_text("", encoding="utf-8")
    return snapshot_dir


def ensure_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LEARNINGS_PATH.exists():
        LEARNINGS_PATH.write_text(json.dumps(LEARNINGS_TEMPLATE, indent=2), encoding="utf-8")
    STRATEGY_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STRATEGY_MD_PATH.exists():
        STRATEGY_MD_PATH.write_text(STRATEGY_TEMPLATE, encoding="utf-8")


def _load_learnings() -> dict[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(LEARNINGS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        payload = dict(LEARNINGS_TEMPLATE)
    for key in LEARNINGS_TEMPLATE:
        if not isinstance(payload.get(key), list):
            payload[key] = []
    return payload


def _insert_line_after_heading(document: str, heading: str, line: str) -> str:
    marker = f"## {heading}\n"
    if marker not in document:
        return document.rstrip() + f"\n\n{marker}- {line}\n"
    insert_at = document.index(marker) + len(marker)
    return document[:insert_at] + f"- {line}\n" + document[insert_at:]


def apply_adjustment(symbol: str, action: str, value: str, approved_by: str) -> Path:
    snapshot_dir = create_snapshot(symbol, action)
    ensure_files()

    now_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    entry = {
        "timestamp": now_utc,
        "symbol": symbol.upper(),
        "action": action,
        "value": value,
        "approved_by": approved_by,
    }

    learnings = _load_learnings()
    normalized_action = action.lower()
    if "pattern" in normalized_action:
        learnings["patterns"].append(entry)
    elif "correlation" in normalized_action:
        learnings["market_correlations"].append(entry)
    else:
        learnings["strategy_changes"].append(entry)
    LEARNINGS_PATH.write_text(json.dumps(learnings, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown = STRATEGY_MD_PATH.read_text(encoding="utf-8")
    bullet = f"{entry['symbol']} | {action} | {value} | approved by {approved_by} | {now_utc}"
    if normalized_action in {"activate_rule", "add_rule", "new_rule"}:
        markdown = _insert_line_after_heading(markdown, "Active Rules", bullet)
    if normalized_action in {"retire_rule", "deactivate_rule"}:
        markdown = _insert_line_after_heading(markdown, "Retired Rules", bullet)
    markdown = markdown.rstrip() + f"\n| {now_utc} | {entry['symbol']} | {action} | {value} | {approved_by} |\n"
    STRATEGY_MD_PATH.write_text(markdown, encoding="utf-8")
    return snapshot_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a strategy adjustment with snapshots")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--approved-by", required=True, dest="approved_by")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = apply_adjustment(
        symbol=args.symbol,
        action=args.action,
        value=args.value,
        approved_by=args.approved_by,
    )
    print(f"created snapshot {snapshot_dir}")
    print("applied strategy adjustment")


if __name__ == "__main__":
    main()
