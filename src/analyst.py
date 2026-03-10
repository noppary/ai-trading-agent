#!/usr/bin/env python3
"""Analyze trade history, request Thai recommendations, and persist reports."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

TRADE_HISTORY_DIR = Path("/root/ai-trading-agent/data/trade_history")
REPORTS_DIR = Path("/root/ai-trading-agent/data/reports")
INBOX_DIR = Path("/root/.openclaw/workspace/inbox")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-haiku-4-5"


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


def _pct(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return (sum(1 for flag in flags if flag) / len(flags)) * 100.0


def _infer_volatility_bucket(raw_value: Any) -> str | None:
    value = str(raw_value or "").strip().lower()
    if not value:
        return None
    if "high" in value or value in {"volatile", "hv"}:
        return "high"
    if "low" in value or value in {"calm", "lv"}:
        return "low"
    return None


def compute_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    defaults = {
        "trade_count": len(trades),
        "win_rate": 0.0,
        "avg_pnl_usd": 0.0,
        "avg_pnl_pct": 0.0,
        "total_pnl_usd": 0.0,
        "best_trade": None,
        "worst_trade": None,
        "long_win_rate": 0.0,
        "short_win_rate": 0.0,
        "avg_hold_minutes": 0.0,
        "confidence_accuracy": None,
        "losing_streak_max": 0,
        "high_vol_win_rate": None,
        "low_vol_win_rate": None,
    }
    if not trades:
        return defaults

    pnls = [_to_float(trade.get("pnl")) for trade in trades]
    pnl_pcts = [_to_float(trade.get("pnl_pct")) for trade in trades]
    hold_minutes = [_to_float(trade.get("hold_duration_minutes")) for trade in trades]
    sides = [str(trade.get("side") or "").lower() for trade in trades]

    wins = [pnl > 0 for pnl in pnls]
    best_index = max(range(len(trades)), key=lambda idx: pnls[idx])
    worst_index = min(range(len(trades)), key=lambda idx: pnls[idx])

    long_indexes = [idx for idx, side in enumerate(sides) if side == "long"]
    short_indexes = [idx for idx, side in enumerate(sides) if side == "short"]

    confidence_scored: list[bool] = []
    for idx, trade in enumerate(trades):
        confidence = trade.get("signal_confidence")
        if confidence in (None, ""):
            continue
        confidence_value = _to_float(confidence, default=-1.0)
        if confidence_value < 0:
            continue
        threshold = 50.0 if confidence_value > 1.0 else 0.5
        confidence_scored.append(
            (confidence_value >= threshold and pnls[idx] > 0)
            or (confidence_value < threshold and pnls[idx] <= 0)
        )

    losing_streak = 0
    losing_streak_max = 0
    for pnl in pnls:
        if pnl < 0:
            losing_streak += 1
            losing_streak_max = max(losing_streak_max, losing_streak)
        else:
            losing_streak = 0

    high_vol_indexes: list[int] = []
    low_vol_indexes: list[int] = []
    for idx, trade in enumerate(trades):
        bucket = _infer_volatility_bucket(trade.get("market_condition"))
        if bucket == "high":
            high_vol_indexes.append(idx)
        elif bucket == "low":
            low_vol_indexes.append(idx)

    return {
        **defaults,
        "win_rate": round(_pct(wins), 4),
        "avg_pnl_usd": round(sum(pnls) / len(pnls), 8),
        "avg_pnl_pct": round(sum(pnl_pcts) / len(pnl_pcts), 8),
        "total_pnl_usd": round(sum(pnls), 8),
        "best_trade": trades[best_index],
        "worst_trade": trades[worst_index],
        "long_win_rate": round(_pct([pnls[idx] > 0 for idx in long_indexes]), 4) if long_indexes else 0.0,
        "short_win_rate": round(_pct([pnls[idx] > 0 for idx in short_indexes]), 4) if short_indexes else 0.0,
        "avg_hold_minutes": round(sum(hold_minutes) / len(hold_minutes), 4),
        "confidence_accuracy": round(_pct(confidence_scored), 4) if confidence_scored else None,
        "losing_streak_max": losing_streak_max,
        "high_vol_win_rate": round(_pct([pnls[idx] > 0 for idx in high_vol_indexes]), 4) if high_vol_indexes else None,
        "low_vol_win_rate": round(_pct([pnls[idx] > 0 for idx in low_vol_indexes]), 4) if low_vol_indexes else None,
    }


def get_thai_recommendations(symbol: str, metrics: dict[str, Any]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "ไม่พบ OPENROUTER_API_KEY จึงไม่สามารถสร้างคำแนะนำจากโมเดลได้"

    prompt = (
        "วิเคราะห์ผลเทรดและเสนอคำแนะนำเป็นภาษาไทยแบบ actionable สำหรับผู้พัฒนาระบบเทรดอัตโนมัติ "
        "ให้เน้นการลดการขาดทุนต่อเนื่อง ปรับคุณภาพสัญญาณ และระบุสิ่งที่ควรทดลองต่อไป\n\n"
        f"สัญลักษณ์: {symbol}\n"
        f"สถิติ: {json.dumps(metrics, ensure_ascii=False)}\n\n"
        "ตอบเป็นหัวข้อย่อย 5-8 ข้อ ภาษาไทยเท่านั้น"
    )

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": "คุณเป็นนักวิเคราะห์ performance การเทรด ตอบภาษาไทยเท่านั้น"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter returned no choices")
    content = str(choices[0].get("message", {}).get("content", "")).strip()
    if not content:
        raise ValueError("OpenRouter returned empty content")
    return content


def _markdown_for_report(symbol: str, batch: str, report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    return "\n".join(
        [
            f"# Analysis {symbol} {batch}",
            "",
            "## Summary",
            f"- Generated at: {report['generated_at']}",
            f"- Trade count: {metrics['trade_count']}",
            f"- Win rate: {metrics['win_rate']}",
            f"- Avg PnL USD: {metrics['avg_pnl_usd']}",
            f"- Avg PnL %: {metrics['avg_pnl_pct']}",
            f"- Total PnL USD: {metrics['total_pnl_usd']}",
            f"- Long win rate: {metrics['long_win_rate']}",
            f"- Short win rate: {metrics['short_win_rate']}",
            f"- Avg hold minutes: {metrics['avg_hold_minutes']}",
            f"- Confidence accuracy: {metrics['confidence_accuracy']}",
            f"- Max losing streak: {metrics['losing_streak_max']}",
            f"- High vol win rate: {metrics['high_vol_win_rate']}",
            f"- Low vol win rate: {metrics['low_vol_win_rate']}",
            "",
            "## Recommendations",
            report["recommendations_th"],
            "",
        ]
    )


def analyze_symbol(history_path: Path, batch: str) -> tuple[Path, Path] | None:
    try:
        trades = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"skipping invalid trade history: {history_path}")
        return None

    if not isinstance(trades, list):
        print(f"skipping non-array trade history: {history_path}")
        return None

    if len(trades) == 0:
        print(f"skipping {history_path.stem.upper()}: 0 trades — no LLM call needed")
        return None

    symbol = history_path.stem.upper()
    metrics = compute_metrics(trades)

    latest_close = max(
        (_parse_iso(str(trade.get("close_time"))) for trade in trades if trade.get("close_time")),
        default=None,
    )
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    try:
        recommendations_th = get_thai_recommendations(symbol, metrics)
    except Exception as exc:  # noqa: BLE001
        recommendations_th = f"ไม่สามารถสร้างคำแนะนำจาก OpenRouter ได้: {exc}"

    report = {
        "symbol": symbol,
        "batch": batch,
        "generated_at": generated_at,
        "latest_close_time": latest_close.isoformat().replace("+00:00", "Z") if latest_close else None,
        "metrics": metrics,
        "recommendations_th": recommendations_th,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / f"analysis_{symbol}_{batch}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    inbox_path = INBOX_DIR / f"analysis_{symbol}_{batch}.md"
    inbox_path.write_text(_markdown_for_report(symbol, batch, report), encoding="utf-8")
    return report_path, inbox_path


def main() -> None:
    TRADE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)

    batch = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    trade_files = sorted(path for path in TRADE_HISTORY_DIR.glob("*.json") if path.is_file())
    for trade_file in trade_files:
        outputs = analyze_symbol(trade_file, batch)
        if outputs is None:
            continue
        report_path, inbox_path = outputs
        print(f"saved {report_path} and {inbox_path}")


if __name__ == "__main__":
    main()
