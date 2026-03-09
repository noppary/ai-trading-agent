#!/usr/bin/env python3
"""Analyze trade history and generate per-symbol reports plus inbox markdown."""

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


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _pct(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v) / len(values) * 100.0


def compute_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "trade_count": 0,
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

    pnls = [_to_float(t.get("pnl")) for t in trades]
    pnl_pcts = [_to_float(t.get("pnl_pct")) for t in trades]
    holds = [_to_float(t.get("hold_duration_minutes")) for t in trades]
    wins = [p > 0 for p in pnls]
    sides = [str(t.get("side", "")).lower() for t in trades]
    market = [str(t.get("market_condition", "")).lower() for t in trades]

    best_idx = max(range(len(trades)), key=lambda i: pnls[i])
    worst_idx = min(range(len(trades)), key=lambda i: pnls[i])

    long_flags = [i for i, s in enumerate(sides) if s == "long"]
    short_flags = [i for i, s in enumerate(sides) if s == "short"]

    long_win_rate = _pct([pnls[i] > 0 for i in long_flags]) if long_flags else 0.0
    short_win_rate = _pct([pnls[i] > 0 for i in short_flags]) if short_flags else 0.0

    confidence_correct: list[bool] = []
    for i, t in enumerate(trades):
        conf = t.get("signal_confidence")
        if conf is None:
            continue
        c = _to_float(conf, default=-1)
        if c < 0:
            continue
        threshold = 50.0 if c > 1.0 else 0.5
        confidence_correct.append((c >= threshold and pnls[i] > 0) or (c < threshold and pnls[i] <= 0))
    confidence_accuracy = _pct(confidence_correct) if confidence_correct else None

    max_streak = 0
    cur_streak = 0
    for p in pnls:
        if p < 0:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    high_vol_idx = [i for i, m in enumerate(market) if "high" in m]
    low_vol_idx = [i for i, m in enumerate(market) if "low" in m]
    high_vol_win_rate = _pct([pnls[i] > 0 for i in high_vol_idx]) if high_vol_idx else None
    low_vol_win_rate = _pct([pnls[i] > 0 for i in low_vol_idx]) if low_vol_idx else None

    return {
        "trade_count": len(trades),
        "win_rate": round(_pct(wins), 4),
        "avg_pnl_usd": round(sum(pnls) / len(pnls), 8),
        "avg_pnl_pct": round(sum(pnl_pcts) / len(pnl_pcts), 8),
        "total_pnl_usd": round(sum(pnls), 8),
        "best_trade": trades[best_idx],
        "worst_trade": trades[worst_idx],
        "long_win_rate": round(long_win_rate, 4),
        "short_win_rate": round(short_win_rate, 4),
        "avg_hold_minutes": round(sum(holds) / len(holds), 4),
        "confidence_accuracy": None if confidence_accuracy is None else round(confidence_accuracy, 4),
        "losing_streak_max": max_streak,
        "high_vol_win_rate": None if high_vol_win_rate is None else round(high_vol_win_rate, 4),
        "low_vol_win_rate": None if low_vol_win_rate is None else round(low_vol_win_rate, 4),
    }


def get_thai_recommendations(symbol: str, metrics: dict[str, Any]) -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return "ไม่พบ OPENROUTER_API_KEY จึงไม่สามารถสร้างคำแนะนำด้วย AI ได้"

    prompt = (
        "คุณเป็นผู้ช่วยวิเคราะห์กลยุทธ์เทรดคริปโต ให้คำแนะนำเป็นภาษาไทยแบบ actionable และสั้นกระชับ "
        "จากข้อมูลสถิติต่อไปนี้ โดยเน้นการลด drawdown และเพิ่มคุณภาพสัญญาณ\n\n"
        f"สัญลักษณ์: {symbol}\n"
        f"metrics: {json.dumps(metrics, ensure_ascii=False)}\n\n"
        "ตอบกลับเป็นหัวข้อย่อยภาษาไทย 5-8 ข้อ"
    )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "คุณเป็นนักวิเคราะห์การเทรดมืออาชีพ ตอบภาษาไทยเท่านั้น"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "ไม่สามารถดึงคำแนะนำจาก OpenRouter ได้ (choices ว่าง)"
        return str(choices[0].get("message", {}).get("content", "")).strip() or "ได้รับผลลัพธ์ว่างจากโมเดล"
    except Exception as exc:  # noqa: BLE001
        return f"เกิดข้อผิดพลาดระหว่างเรียก OpenRouter: {exc}"


def write_inbox_markdown(symbol: str, batch: str, metrics: dict[str, Any], recommendations_th: str) -> Path:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    md_path = INBOX_DIR / f"analysis_{symbol}_{batch}.md"
    lines = [
        f"# Analysis: {symbol} ({batch})",
        "",
        "## Metrics",
        f"- trade_count: {metrics.get('trade_count')}",
        f"- win_rate: {metrics.get('win_rate')}",
        f"- avg_pnl_usd: {metrics.get('avg_pnl_usd')}",
        f"- avg_pnl_pct: {metrics.get('avg_pnl_pct')}",
        f"- total_pnl_usd: {metrics.get('total_pnl_usd')}",
        f"- long_win_rate: {metrics.get('long_win_rate')}",
        f"- short_win_rate: {metrics.get('short_win_rate')}",
        f"- avg_hold_minutes: {metrics.get('avg_hold_minutes')}",
        f"- confidence_accuracy: {metrics.get('confidence_accuracy')}",
        f"- losing_streak_max: {metrics.get('losing_streak_max')}",
        f"- high_vol_win_rate: {metrics.get('high_vol_win_rate')}",
        f"- low_vol_win_rate: {metrics.get('low_vol_win_rate')}",
        "",
        "## Thai Recommendations",
        recommendations_th,
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TRADE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    batch = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    files = sorted([p for p in TRADE_HISTORY_DIR.glob("*.json") if p.is_file()])

    for path in files:
        symbol = path.stem.upper()
        try:
            trades = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON: {path}")
            continue
        if not isinstance(trades, list):
            print(f"Skipping non-array trade file: {path}")
            continue

        metrics = compute_metrics(trades)
        recommendations_th = get_thai_recommendations(symbol, metrics)

        report = {
            "symbol": symbol,
            "batch": batch,
            "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "metrics": metrics,
            "recommendations_th": recommendations_th,
        }
        report_path = REPORTS_DIR / f"analysis_{symbol}_{batch}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        inbox_path = write_inbox_markdown(symbol, batch, metrics, recommendations_th)
        print(f"Saved {report_path} and {inbox_path}")


if __name__ == "__main__":
    main()
