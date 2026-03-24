"""Entry-point script that wires together the trading agent, data feeds, and API."""

import sys
import argparse
import pathlib
import signal
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src.agent.decision_maker import TradingAgent
from src.indicators.hyperliquid_indicators import HyperliquidIndicators
from src.trading.hyperliquid_api import HyperliquidAPI
import asyncio
import logging
from collections import deque, OrderedDict
from datetime import datetime, timezone
import math  # For Sharpe
from dotenv import load_dotenv
import os
import json
from aiohttp import web
from src.utils.formatting import format_number as fmt, format_size as fmt_sz
from src.utils.prompt_utils import json_default, round_or_none, round_series
from src.utils.state_manager import (
    load_circuit_state, save_circuit_state,
    load_active_trades, save_active_trades,
    load_trade_log, append_trade_log_entry,
    build_recovery_report,
)

# Self-learning engine
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from learn_from_trades import update_learnings, load_diary

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global flag for graceful shutdown (SIGTERM handler)
is_shutting_down = False

def _sigterm_handler(signum, frame):
    global is_shutting_down
    is_shutting_down = True
    logging.warning("SIGTERM received — initiating graceful shutdown")

# Register SIGTERM handler
signal.signal(signal.SIGTERM, _sigterm_handler)

def clear_terminal():
    """Clear the terminal screen on Windows or POSIX systems."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_interval_seconds(interval_str):
    """Convert interval strings like '5m' or '1h' to seconds."""
    if interval_str.endswith('m'):
        return int(interval_str[:-1]) * 60
    elif interval_str.endswith('h'):
        return int(interval_str[:-1]) * 3600
    elif interval_str.endswith('d'):
        return int(interval_str[:-1]) * 86400
    else:
        raise ValueError(f"Unsupported interval: {interval_str}")

def main():
    """Parse CLI args, bootstrap dependencies, and launch the trading loop."""
    clear_terminal()
    parser = argparse.ArgumentParser(description="LLM-based Trading Agent on Hyperliquid")
    parser.add_argument("--assets", type=str, nargs="+", required=False, help="Assets to trade, e.g., BTC ETH")
    parser.add_argument("--interval", type=str, required=False, help="Interval period, e.g., 1h")
    args = parser.parse_args()

    # Allow assets/interval via .env (CONFIG) if CLI not provided
    from src.config_loader import CONFIG
    assets_env = CONFIG.get("assets")
    interval_env = CONFIG.get("interval")
    if (not args.assets or len(args.assets) == 0) and assets_env:
        # Support space or comma separated
        if "," in assets_env:
            args.assets = [a.strip() for a in assets_env.split(",") if a.strip()]
        else:
            args.assets = [a.strip() for a in assets_env.split(" ") if a.strip()]
    if not args.interval and interval_env:
        args.interval = interval_env

    if not args.assets or not args.interval:
        parser.error("Please provide --assets and --interval, or set ASSETS and INTERVAL in .env")

    hyperliquid = HyperliquidAPI()
    indicators = HyperliquidIndicators()
    agent = TradingAgent()

    start_time = datetime.now(timezone.utc)
    invocation_count = 0
    recent_events = deque(maxlen=200)
    diary_path = "diary.jsonl"

    # ── Load persisted state (Phase 1: crash-proof) ──────────────────────────
    circuit = load_circuit_state()
    initial_account_value = circuit.get("initial_account_value")
    peak_account_value = circuit.get("peak_account_value")
    daily_start_value = circuit.get("daily_start_value")
    daily_start_date = circuit.get("daily_start_date")  # ISO date string or None
    trading_halted = circuit.get("trading_halted", False)
    trade_log = load_trade_log()
    active_trades = load_active_trades()

    consecutive_failures = 0
    last_loop_ts = None  # for /health endpoint

    # Perp mid-price history sampled each loop (authoritative, avoids spot/perp basis mismatch)
    price_history = {}

    # ── Startup reconciliation (Phase 1 P1.3) ─────────────────────────────────
    async def _startup_reconcile():
        nonlocal active_trades, trading_halted
        try:
            state_r = await hyperliquid.get_user_state()
            positions_r = state_r.get("assetPositions", [])
            orders_r = await hyperliquid.get_open_orders()

            report = build_recovery_report(
                exchange_positions=[p["position"] for p in positions_r if "position" in p],
                exchange_orders=orders_r,
                persisted_trades=active_trades,
            )

            # 1) Positions on exchange not tracked → add to active_trades
            for pos_wrap in report["missing_from_persisted"]:
                pos = pos_wrap
                coin = pos.get("coin")
                size = float(pos.get("szi", 0) or 0)
                entry_px = float(pos.get("entryPx", 0) or 0)
                if not coin or abs(size) < 1e-9:
                    continue
                is_long = size > 0
                # TP/SL OIDs are unknown — set to None so reconciliation loop will handle
                active_trades.append({
                    "asset": coin,
                    "is_long": is_long,
                    "amount": abs(size),
                    "entry_price": entry_px,
                    "tp_oid": None,
                    "sl_oid": None,
                    "exit_plan": "",
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "_recovered": True,
                })
                add_event(f"RECOVERY: added {coin} {'long' if is_long else 'short'} x{abs(size):.6f} from exchange (TP/SL OIDs unknown)")

            # 2) Persisted trades with no exchange position → clean up
            for tr in report["orphaned_persisted"]:
                add_event(f"RECOVERY: removing stale active_trade for {tr.get('asset')} (no position on exchange)")
                try:
                    active_trades.remove(tr)
                except ValueError:
                    pass

            # 3) Positions missing TP or SL → log as warning (P1.3 P1.4)
            for pos_wrap in report["orders_missing_tp_sl"]:
                pos = pos_wrap
                coin = pos.get("coin")
                size = float(pos.get("szi", 0) or 0)
                add_event(f"WARNING: {coin} position has no TP/SL orders — manual intervention may be required")
                if trading_halted:
                    # If halted, we shouldn't be re-opening anything — just note it
                    continue
                # Note: we do NOT auto-place TP/SL here (too risky without LLM validation).
                # The loop's reconciliation will catch this every iteration.

            save_active_trades(active_trades)
            if report["missing_from_persisted"] or report["orphaned_persisted"]:
                add_event(f"Startup reconciliation done: {len(active_trades)} active_trades tracked")
        except Exception as e:
            add_event(f"Startup reconciliation error: {e}")

    # Run reconciliation before entering main loop
    # (needs hyperliquid already instantiated, before the "wallet validation" in main_async)
    # We'll call this from main_async after hyperliquid is ready.

    print(f"Starting trading agent for assets: {args.assets} at interval: {args.interval}")

    def add_event(msg: str):
        """Log an informational event and push it into the recent events deque."""
        logging.info(msg)

    def send_telegram_alert(message: str):
        """P2.9: Send Telegram alert with retry, delivery confirmation, and failure logging."""
        import subprocess, time
        cmd = ["openclaw", "agent", "--to", "kevin", "--message", message, "--deliver"]
        for attempt in range(3):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,  # capture output so we can check it
                    timeout=15,
                )
                if proc.returncode == 0:
                    logging.info("Telegram alert sent successfully")
                    return
                else:
                    stderr = proc.stderr.decode(errors="replace").strip() if proc.stderr else ""
                    logging.warning("Telegram alert attempt %d failed (exit %d): %s", attempt + 1, proc.returncode, stderr[:200])
            except subprocess.TimeoutExpired:
                logging.warning("Telegram alert attempt %d timed out", attempt + 1)
            except Exception as e:
                logging.error("Telegram alert attempt %d exception: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s
        logging.error("Telegram alert FAILED after 3 attempts — last message: %s", message[:200])

    async def run_loop():
        """Main trading loop that gathers data, calls the agent, and executes trades."""
        nonlocal invocation_count, initial_account_value, peak_account_value
        nonlocal daily_start_value, daily_start_date, consecutive_failures, trading_halted, last_loop_ts
        while True:
            # P1.8: graceful shutdown — finish current iteration, then exit cleanly
            if is_shutting_down:
                add_event("Graceful shutdown: persisting state and exiting")
                save_circuit_state({
                    "initial_account_value": initial_account_value,
                    "peak_account_value": peak_account_value,
                    "daily_start_value": daily_start_value,
                    "daily_start_date": str(daily_start_date) if daily_start_date else None,
                    "trading_halted": trading_halted,
                    "halted_reason": None,
                })
                save_active_trades(active_trades)
                break
            invocation_count += 1
            minutes_since_start = (datetime.now(timezone.utc) - start_time).total_seconds() / 60

            # Global account state
            state = await hyperliquid.get_user_state()
            total_value = state.get('total_value') or state['balance'] + sum(p.get('pnl', 0) for p in state['positions'])
            sharpe = calculate_sharpe(trade_log)

            account_value = total_value
            if initial_account_value is None:
                initial_account_value = account_value
            total_return_pct = ((account_value - initial_account_value) / initial_account_value * 100.0) if initial_account_value else 0.0

            # ── Circuit breaker checks ──
            if peak_account_value is None or account_value > peak_account_value:
                peak_account_value = account_value

            today = datetime.now(timezone.utc).date()
            if daily_start_date != today:
                daily_start_date = today
                daily_start_value = account_value

            # P1.1: persist circuit state after every update
            save_circuit_state({
                "initial_account_value": initial_account_value,
                "peak_account_value": peak_account_value,
                "daily_start_value": daily_start_value,
                "daily_start_date": str(daily_start_date) if daily_start_date else None,
                "trading_halted": trading_halted,
                "halted_reason": None,
            })

            drawdown_pct = (peak_account_value - account_value) / peak_account_value * 100 if peak_account_value else 0
            daily_loss = (daily_start_value - account_value) if daily_start_value else 0
            # P2.3: %-based daily loss limit (supersedes hardcoded USD)
            daily_loss_limit_pct = CONFIG.get("daily_loss_limit_pct", 3.0)
            daily_loss_pct = (daily_loss / daily_start_value * 100.0) if daily_start_value and daily_start_value > 0 else 0

            if drawdown_pct > CONFIG.get("max_drawdown_pct", 15):
                trading_halted = True
                add_event(f"CIRCUIT BREAKER: drawdown {drawdown_pct:.1f}% exceeds limit")
            if daily_loss_pct > daily_loss_limit_pct:
                trading_halted = True
                add_event(f"CIRCUIT BREAKER: daily loss {daily_loss_pct:.2f}% (${daily_loss:.2f}) exceeds {daily_loss_limit_pct}% limit")

            if trading_halted:
                add_event(f"CIRCUIT BREAKER ACTIVE: halting trading (drawdown={drawdown_pct:.1f}%, daily_loss=${daily_loss:.2f})")
                try:
                    await hyperliquid.close_all_positions()
                except Exception as e:
                    add_event(f"Failed to close positions during circuit break: {e}")
                send_telegram_alert(
                    f"🚨 CIRCUIT BREAKER TRIGGERED\n"
                    f"Drawdown: {drawdown_pct:.1f}%\n"
                    f"Daily loss: {daily_loss_pct:.2f}% (${daily_loss:.2f})\n"
                    f"Account value: ${account_value:.2f}\n"
                    f"All positions closed. Trading halted."
                )
                # P1.1: persist halted state so it survives restart
                save_circuit_state({
                    "initial_account_value": initial_account_value,
                    "peak_account_value": peak_account_value,
                    "daily_start_value": daily_start_value,
                    "daily_start_date": str(daily_start_date) if daily_start_date else None,
                    "trading_halted": True,
                    "halted_reason": f"drawdown={drawdown_pct:.1f}% daily_loss_pct={daily_loss_pct:.2f}%",
                })
                await asyncio.sleep(get_interval_seconds(args.interval))
                continue

            positions = []
            for pos_wrap in state['positions']:
                pos = pos_wrap
                coin = pos.get('coin')
                current_px = await hyperliquid.get_current_price(coin) if coin else None
                positions.append({
                    "symbol": coin,
                    "quantity": round_or_none(pos.get('szi'), 6),
                    "entry_price": round_or_none(pos.get('entryPx'), 2),
                    "current_price": round_or_none(current_px, 2),
                    "liquidation_price": round_or_none(pos.get('liquidationPx') or pos.get('liqPx'), 2),
                    "unrealized_pnl": round_or_none(pos.get('pnl'), 4),
                    "leverage": pos.get('leverage')
                })

            recent_diary = []
            try:
                with open(diary_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        entry = json.loads(line)
                        recent_diary.append(entry)
            except Exception:
                pass

            open_orders_struct = []
            try:
                open_orders = await hyperliquid.get_open_orders()
                for o in open_orders[:50]:
                    open_orders_struct.append({
                        "coin": o.get('coin'),
                        "oid": o.get('oid'),
                        "is_buy": o.get('isBuy'),
                        "size": round_or_none(o.get('sz'), 6),
                        "price": round_or_none(o.get('px'), 2),
                        "trigger_price": round_or_none(o.get('triggerPx'), 2),
                        "order_type": o.get('orderType')
                    })
            except Exception:
                open_orders = []

            # Reconcile active trades
            try:
                # Fetch fresh state specifically for reconciliation
                state_for_reconcile = await hyperliquid.get_user_state()
                positions_for_reconcile = state_for_reconcile.get("assetPositions", [])
                assets_with_positions = set()
                for pos_wrap in positions_for_reconcile:
                    try:
                        # Accessing position details from the wrapped structure
                        pos = pos_wrap["position"]
                        # Check if there's an actual position size
                        if abs(float(pos.get('szi') or 0)) > 0:
                            assets_with_positions.add(pos.get('coin'))
                    except Exception as e:
                        add_event(f"Error processing position for reconciliation: {e}")
                        continue
                assets_with_orders = {o.get('coin') for o in (open_orders or []) if o.get('coin')}
                for tr in active_trades[:]:
                    asset = tr.get('asset')
                    if asset not in assets_with_positions and asset not in assets_with_orders:
                        add_event(f"Reconciling stale active trade for {asset} (no position, no orders)")
                        active_trades.remove(tr)
                        save_active_trades(active_trades)
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": "reconcile_close",
                                "reason": "no_position_no_orders",
                                "opened_at": tr.get('opened_at')
                            }) + "\n")
            except Exception:
                pass

            recent_fills_struct = []
            try:
                fills = await hyperliquid.get_recent_fills(limit=50)
                for f_entry in fills[-20:]:
                    try:
                        t_raw = f_entry.get('time') or f_entry.get('timestamp')
                        timestamp = None
                        if t_raw is not None:
                            try:
                                t_int = int(t_raw)
                                if t_int > 1e12:
                                    timestamp = datetime.fromtimestamp(t_int / 1000, tz=timezone.utc).isoformat()
                                else:
                                    timestamp = datetime.fromtimestamp(t_int, tz=timezone.utc).isoformat()
                            except Exception:
                                timestamp = str(t_raw)
                        recent_fills_struct.append({
                            "timestamp": timestamp,
                            "coin": f_entry.get('coin') or f_entry.get('asset'),
                            "is_buy": f_entry.get('isBuy'),
                            "size": round_or_none(f_entry.get('sz') or f_entry.get('size'), 6),
                            "price": round_or_none(f_entry.get('px') or f_entry.get('price'), 2)
                        })
                    except Exception:
                        continue
            except Exception:
                pass

            dashboard = {
                "total_return_pct": round(total_return_pct, 2),
                "balance": round_or_none(state['balance'], 2),
                "account_value": round_or_none(account_value, 2),
                "sharpe_ratio": round_or_none(sharpe, 3),
                "positions": positions,
                "active_trades": [
                    {
                        "asset": tr.get('asset'),
                        "is_long": tr.get('is_long'),
                        "amount": round_or_none(tr.get('amount'), 6),
                        "entry_price": round_or_none(tr.get('entry_price'), 2),
                        "tp_oid": tr.get('tp_oid'),
                        "sl_oid": tr.get('sl_oid'),
                        "exit_plan": tr.get('exit_plan'),
                        "opened_at": tr.get('opened_at')
                    }
                    for tr in active_trades
                ],
                "open_orders": open_orders_struct,
                "recent_diary": recent_diary,
                "recent_fills": recent_fills_struct,
            }

            # Gather data for ALL assets first
            market_sections = []
            asset_prices = {}
            for asset in args.assets:
                try:
                    current_price = await hyperliquid.get_current_price(asset)
                    asset_prices[asset] = current_price
                    if asset not in price_history:
                        price_history[asset] = deque(maxlen=60)
                    price_history[asset].append({"t": datetime.now(timezone.utc).isoformat(), "mid": round_or_none(current_price, 2)})
                    oi = await hyperliquid.get_open_interest(asset)
                    funding = await hyperliquid.get_funding_rate(asset)

                    intraday_tf = "5m"
                    ema_series = indicators.fetch_series("ema", asset, intraday_tf, results=10, params={"period": 20})
                    macd_series = indicators.fetch_series("macd", asset, intraday_tf, results=10)
                    rsi7_series = indicators.fetch_series("rsi", asset, intraday_tf, results=10, params={"period": 7})
                    rsi14_series = indicators.fetch_series("rsi", asset, intraday_tf, results=10, params={"period": 14})

                    lt_ema20 = indicators.fetch_value("ema", asset, "4h", params={"period": 20})
                    lt_ema50 = indicators.fetch_value("ema", asset, "4h", params={"period": 50})
                    lt_atr3 = indicators.fetch_value("atr", asset, "4h", params={"period": 3})
                    lt_atr14 = indicators.fetch_value("atr", asset, "4h", params={"period": 14})
                    lt_macd_series = indicators.fetch_series("macd", asset, "4h", results=10)
                    lt_rsi_series = indicators.fetch_series("rsi", asset, "4h", results=10, params={"period": 14})

                    recent_mids = [entry["mid"] for entry in list(price_history.get(asset, []))[-10:]]
                    funding_annualized = round(funding * 24 * 365 * 100, 2) if funding else None

                    market_sections.append({
                        "asset": asset,
                        "current_price": round_or_none(current_price, 2),
                        "intraday": {
                            "ema20": round_or_none(ema_series[-1], 2) if ema_series else None,
                            "macd": round_or_none(macd_series[-1], 2) if macd_series else None,
                            "rsi7": round_or_none(rsi7_series[-1], 2) if rsi7_series else None,
                            "rsi14": round_or_none(rsi14_series[-1], 2) if rsi14_series else None,
                            "series": {
                                "ema20": round_series(ema_series, 2),
                                "macd": round_series(macd_series, 2),
                                "rsi7": round_series(rsi7_series, 2),
                                "rsi14": round_series(rsi14_series, 2)
                            }
                        },
                        "long_term": {
                            "ema20": round_or_none(lt_ema20, 2),
                            "ema50": round_or_none(lt_ema50, 2),
                            "atr3": round_or_none(lt_atr3, 2),
                            "atr14": round_or_none(lt_atr14, 2),
                            "macd_series": round_series(lt_macd_series, 2),
                            "rsi_series": round_series(lt_rsi_series, 2)
                        },
                        "open_interest": round_or_none(oi, 2),
                        "funding_rate": round_or_none(funding, 8),
                        "funding_annualized_pct": funding_annualized,
                        "recent_mid_prices": recent_mids
                    })
                except Exception as e:
                    add_event(f"Data gather error {asset}: {e}")
                    continue

            # Single LLM call with all assets
            context_payload = OrderedDict([
                ("invocation", {
                    "minutes_since_start": round(minutes_since_start, 2),
                    "current_time": datetime.now(timezone.utc).isoformat(),
                    "invocation_count": invocation_count
                }),
                ("account", dashboard),
                ("market_data", market_sections),
                ("instructions", {
                    "assets": args.assets,
                    "requirement": "Decide actions for all assets and return a strict JSON array matching the schema."
                })
            ])
            context = json.dumps(context_payload, default=json_default)
            add_event(f"Combined prompt length: {len(context)} chars for {len(args.assets)} assets")
            with open("prompts.log", "a") as f:
                f.write(f"\n\n--- {datetime.now()} - ALL ASSETS ---\n{json.dumps(context_payload, indent=2, default=json_default)}\n")

            def _is_failed_outputs(outs):
                """Return True when outputs are missing or clearly invalid."""
                if not isinstance(outs, dict):
                    return True
                decisions = outs.get("trade_decisions")
                if not isinstance(decisions, list) or not decisions:
                    return True
                try:
                    return all(
                        isinstance(o, dict)
                        and (o.get('action') == 'hold')
                        and ('parse error' in (o.get('rationale', '').lower()))
                        for o in decisions
                    )
                except Exception:
                    return True

            try:
                outputs = agent.decide_trade(args.assets, context)
                if not isinstance(outputs, dict):
                    add_event(f"Invalid output format (expected dict): {outputs}")
                    outputs = {}
            except Exception as e:
                import traceback
                add_event(f"Agent error: {e}")
                add_event(f"Traceback: {traceback.format_exc()}")
                outputs = {}

            # Track consecutive failures
            if _is_failed_outputs(outputs):
                consecutive_failures += 1
                add_event(f"Pipeline failure #{consecutive_failures}")
                if consecutive_failures >= CONFIG.get("consecutive_failure_limit", 10):
                    trading_halted = True
                    add_event(f"CIRCUIT BREAKER: {consecutive_failures} consecutive pipeline failures")
                    send_telegram_alert(
                        f"🚨 PIPELINE FAILURE HALT\n"
                        f"{consecutive_failures} consecutive failures.\n"
                        f"Trading halted. Check OpenRouter/model status."
                    )
                    await asyncio.sleep(get_interval_seconds(args.interval))
                    continue

            # Retry once on failure/parse error with a stricter instruction prefix
            if _is_failed_outputs(outputs):
                add_event("Retrying LLM once due to invalid/parse-error output")
                context_retry_payload = OrderedDict([
                    ("retry_instruction", "Return ONLY the JSON array per schema with no prose."),
                    ("original_context", context_payload)
                ])
                context_retry = json.dumps(context_retry_payload, default=json_default)
                try:
                    outputs = agent.decide_trade(args.assets, context_retry)
                    if not isinstance(outputs, dict):
                        add_event(f"Retry invalid format: {outputs}")
                        outputs = {}
                except Exception as e:
                    import traceback
                    add_event(f"Retry agent error: {e}")
                    add_event(f"Retry traceback: {traceback.format_exc()}")
                    outputs = {}

            # Reset failure counter on successful pipeline run
            if not _is_failed_outputs(outputs):
                consecutive_failures = 0

            reasoning_text = outputs.get("reasoning", "") if isinstance(outputs, dict) else ""
            if reasoning_text:
                add_event(f"LLM reasoning summary: {reasoning_text}")

            # Execute trades for each asset
            for output in outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []:
                try:
                    asset = output.get("asset")
                    if not asset or asset not in args.assets:
                        continue
                    action = output.get("action")
                    current_price = asset_prices.get(asset, 0)
                    action = output["action"]
                    rationale = output.get("rationale", "")
                    if rationale:
                        add_event(f"Decision rationale for {asset}: {rationale}")
                    if action in ("buy", "sell"):
                        is_buy = action == "buy"
                        alloc_usd = float(output.get("allocation_usd", 0.0))
                        if alloc_usd <= 0:
                            add_event(f"Holding {asset}: zero/negative allocation")
                            continue

                        # ── P2.10: Max open positions cap ──
                        max_open = CONFIG.get("max_open_positions", 3)
                        if len(active_trades) >= max_open:
                            add_event(f"Holding {asset}: max open positions ({max_open}) reached — skipping new entry")
                            continue

                        # ── P2.5: Per-asset cooldown enforcement ──
                        cooldown_bars = CONFIG.get("trade_cooldown_bars", 3)
                        cooldown_active = False
                        for tr in active_trades:
                            if tr.get("asset") == asset:
                                opened_at = tr.get("opened_at")
                                if opened_at:
                                    try:
                                        opened_ts = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
                                        bars_elapsed = (datetime.now(timezone.utc) - opened_ts).total_seconds() / (
                                            get_interval_seconds(args.interval)
                                        )
                                        if bars_elapsed < cooldown_bars:
                                            add_event(f"Holding {asset}: cooldown active ({bars_elapsed:.1f}/{cooldown_bars} bars elapsed)")
                                            cooldown_active = True
                                            break
                                    except Exception:
                                        pass
                        if cooldown_active:
                            continue

                        # ── Allocation caps ──
                        max_per_asset = account_value * (CONFIG.get("max_position_pct", 25) / 100.0)
                        current_exposure = sum(
                            abs(float(p.get('szi', 0)) * float(p.get('entryPx', 0)))
                            for p in state.get('positions', [])
                        )
                        max_remaining = account_value * (CONFIG.get("max_total_exposure_pct", 75) / 100.0) - current_exposure
                        alloc_usd = min(alloc_usd, max_per_asset, max(0, max_remaining), state['balance'] * 0.9)
                        if alloc_usd <= 0:
                            add_event(f"Holding {asset}: allocation capped to zero")
                            continue

                        amount = alloc_usd / current_price

                        # ── DRY RUN MODE: Skip actual order placement ──
                        if CONFIG.get("dry_run_mode", False):
                            add_event(f"DRY RUN: Would {action.upper()} {asset} amount {amount:.4f} at ~{current_price}")
                            add_event(f"DRY RUN: Would set TP={output.get('tp_price')} SL={output.get('sl_price')}")
                            # Log to diary as DRY RUN
                            with open(diary_path, "a") as f:
                                diary_entry = {
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": asset,
                                    "action": f"dryrun_{action}",
                                    "allocation_usd": alloc_usd,
                                    "amount": amount,
                                    "entry_price": current_price,
                                    "tp_price": output.get("tp_price"),
                                    "sl_price": output.get("sl_price"),
                                    "dry_run": True
                                }
                                f.write(json.dumps(diary_entry) + "\n")
                            continue

                        # ── P2.6: Slippage tracking ─────────────────────────────────────────
                        intended_price = current_price  # price when decision was made
                        order = await hyperliquid.place_buy_order(asset, amount) if is_buy else await hyperliquid.place_sell_order(asset, amount)
                        # Confirm by checking recent fills for this asset shortly after placing
                        await asyncio.sleep(1)
                        fills_check = await hyperliquid.get_recent_fills(limit=10)
                        filled = False
                        fill_price = None
                        slippage_bps = None
                        for fc in reversed(fills_check):
                            try:
                                if (fc.get('coin') == asset or fc.get('asset') == asset):
                                    filled = True
                                    fill_price = float(fc.get('px') or fc.get('price') or intended_price)
                                    slippage_bps = round(abs(fill_price - intended_price) / intended_price * 10000, 2) if intended_price else None
                                    break
                            except Exception:
                                continue
                        trade_log.append({
                            "type": action, "price": intended_price, "fill_price": fill_price,
                            "slippage_bps": slippage_bps,
                            "amount": amount, "exit_plan": output["exit_plan"], "filled": filled
                        })
                        append_trade_log_entry({
                            "type": action, "price": intended_price, "fill_price": fill_price,
                            "slippage_bps": slippage_bps,
                            "amount": amount, "exit_plan": output["exit_plan"], "filled": filled
                        })
                        if slippage_bps is not None:
                            add_event(f"SLIPPAGE: {asset} fill @ {fill_price} vs intent {intended_price} = {slippage_bps} bps")
                        tp_oid = None
                        sl_oid = None

                        # ── P1.4: TP/SL sanity validation ─────────────────────────────
                        tp_price = output.get("tp_price")
                        sl_price = output.get("sl_price")
                        tp_msg = ""
                        sl_msg = ""
                        if tp_price is not None and sl_price is not None:
                            if is_buy:
                                # Long: TP must be > entry, SL must be < entry
                                if tp_price <= current_price:
                                    tp_msg = f"TP {tp_price} <= entry {current_price} — skipping TP"
                                    tp_price = None
                                if sl_price >= current_price:
                                    sl_msg = f"SL {sl_price} >= entry {current_price} — skipping SL"
                                    sl_price = None
                            else:
                                # Short: TP must be < entry, SL must be > entry
                                if tp_price >= current_price:
                                    tp_msg = f"TP {tp_price} >= entry {current_price} — skipping TP"
                                    tp_price = None
                                if sl_price <= current_price:
                                    sl_msg = f"SL {sl_price} <= entry {current_price} — skipping SL"
                                    sl_price = None
                            # Minimum TP/SL distance: 0.5% of price
                            min_distance = current_price * 0.005
                            if tp_price and abs(tp_price - current_price) < min_distance:
                                tp_msg = f"TP distance {abs(tp_price-current_price)/current_price*100:.2f}% < 0.5% minimum — skipping TP"
                                tp_price = None
                            if sl_price and abs(sl_price - current_price) < min_distance:
                                sl_msg = f"SL distance {abs(sl_price-current_price)/current_price*100:.2f}% < 0.5% minimum — skipping SL"
                                sl_price = None
                        if tp_msg:
                            add_event(f"WARNING: {asset} {tp_msg}")
                        if sl_msg:
                            add_event(f"WARNING: {asset} {sl_msg}")
                        # ── End P1.4 ────────────────────────────────────────────

                        if tp_price is not None and tp_price:
                            tp_order = await hyperliquid.place_take_profit(asset, is_buy, amount, tp_price)
                            tp_oids = hyperliquid.extract_oids(tp_order)
                            tp_oid = tp_oids[0] if tp_oids else None
                            add_event(f"TP placed {asset} at {tp_price}")
                        if sl_price is not None and sl_price:
                            sl_order = await hyperliquid.place_stop_loss(asset, is_buy, amount, sl_price)
                            sl_oids = hyperliquid.extract_oids(sl_order)
                            sl_oid = sl_oids[0] if sl_oids else None
                            add_event(f"SL placed {asset} at {sl_price}")
                        # Reconcile: if opposite-side position exists or TP/SL just filled, clear stale active_trades for this asset
                        for existing in active_trades[:]:
                            if existing.get('asset') == asset:
                                try:
                                    active_trades.remove(existing)
                                except ValueError:
                                    pass
                        active_trades.append({
                            "asset": asset,
                            "is_long": is_buy,
                            "amount": amount,
                            "entry_price": current_price,
                            "tp_oid": tp_oid,
                            "sl_oid": sl_oid,
                            "exit_plan": output["exit_plan"],
                            "opened_at": datetime.now(timezone.utc).isoformat()
                        })
                        save_active_trades(active_trades)  # P1.2: persist active_trades
                        add_event(f"{action.upper()} {asset} amount {amount:.4f} at ~{current_price}")
                        if rationale:
                            add_event(f"Post-trade rationale for {asset}: {rationale}")
                        # Write to diary after confirming fills status
                        with open(diary_path, "a") as f:
                            diary_entry = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": action,
                                "allocation_usd": alloc_usd,
                                "amount": amount,
                                "entry_price": intended_price,
                                "fill_price": fill_price,
                                "slippage_bps": slippage_bps,
                                "tp_price": tp_price,
                                "tp_oid": tp_oid,
                                "sl_price": sl_price,
                                "sl_oid": sl_oid,
                                "exit_plan": output.get("exit_plan", ""),
                                "rationale": output.get("rationale", ""),
                                "order_result": str(order),
                                "opened_at": datetime.now(timezone.utc).isoformat(),
                                "filled": filled
                            }
                            f.write(json.dumps(diary_entry) + "\n")
                        slippage_str = f" | Slip: {slippage_bps} bps" if slippage_bps is not None else ""
                        send_telegram_alert(
                            f"📊 {action.upper()} {asset}\n"
                            f"Amount: {amount:.4f} (~${alloc_usd:.0f})\n"
                            f"Price: ${current_price:.2f}{slippage_str}\n"
                            f"TP: {output.get('tp_price')}, SL: {output.get('sl_price')}"
                        )
                    else:
                        add_event(f"Hold {asset}: {output.get('rationale', '')}")
                        # Throttle hold diary writes: once/hour instead of every 5min
                        if invocation_count % 12 == 0:
                            with open(diary_path, "a") as f:
                                diary_entry = {
                                    "timestamp": datetime.now().isoformat(),
                                    "asset": asset,
                                    "action": "hold",
                                    "rationale": output.get("rationale", "")
                                }
                                f.write(json.dumps(diary_entry) + "\n")
                except Exception as e:
                    import traceback
                    add_event(f"Execution error {asset}: {e}")

            # ── Self-learning: analyze closed trades every ~1 hour ──
            if invocation_count % 12 == 0:
                try:
                    closed = load_diary(days=7)
                    if closed:
                        learnings = update_learnings(closed)
                        n_patterns = len(learnings.get("patterns", []))
                        n_changes = len(learnings.get("strategy_changes", []))
                        if n_changes > 0:
                            add_event(f"Learnings updated: {n_patterns} patterns, {n_changes} strategy changes")
                except Exception as e:
                    logging.warning("Self-learning error: %s", e)

            # P1.6: update last_loop_ts for /health endpoint
            last_loop_ts = datetime.now(timezone.utc).isoformat()

            # P1.8: graceful shutdown check after each iteration
            if is_shutting_down:
                add_event("Graceful shutdown: persisting state and exiting")
                save_circuit_state({
                    "initial_account_value": initial_account_value,
                    "peak_account_value": peak_account_value,
                    "daily_start_value": daily_start_value,
                    "daily_start_date": str(daily_start_date) if daily_start_date else None,
                    "trading_halted": trading_halted,
                    "halted_reason": None,
                })
                save_active_trades(active_trades)
                break

            await asyncio.sleep(get_interval_seconds(args.interval))

    async def handle_diary(request):
        """Return diary entries as JSON or newline-delimited text."""
        try:
            raw = request.query.get('raw')
            download = request.query.get('download')
            if raw or download:
                if not os.path.exists(diary_path):
                    return web.Response(text="", content_type="text/plain")
                with open(diary_path, "r") as f:
                    data = f.read()
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename=diary.jsonl"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(request.query.get('limit', '200'))
            with open(diary_path, "r") as f:
                lines = f.readlines()
            start = max(0, len(lines) - limit)
            entries = [json.loads(l) for l in lines[start:]]
            return web.json_response({"entries": entries})
        except FileNotFoundError:
            return web.json_response({"entries": []})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    ALLOWED_LOG_FILES = {"llm_requests.log", "prompts.log", "diary.jsonl"}

    async def handle_logs(request):
        """Stream log files with optional download or tailing behaviour."""
        try:
            path = request.query.get('path', 'llm_requests.log')
            if path not in ALLOWED_LOG_FILES:
                return web.json_response({"error": "forbidden"}, status=403)
            download = request.query.get('download')
            limit_param = request.query.get('limit')
            if not os.path.exists(path):
                return web.Response(text="", content_type="text/plain")
            with open(path, "r") as f:
                data = f.read()
            if download or (limit_param and (limit_param.lower() == 'all' or limit_param == '-1')):
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename={os.path.basename(path)}"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(limit_param) if limit_param else 2000
            return web.Response(text=data[-limit:], content_type="text/plain")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_emergency_close(request):
        """Close all positions, cancel all orders, and halt trading."""
        nonlocal trading_halted
        trading_halted = True
        try:
            results = await hyperliquid.close_all_positions()
            add_event("EMERGENCY CLOSE: all positions closed via API endpoint")
            send_telegram_alert("🚨 EMERGENCY CLOSE triggered via API. All positions closed. Trading halted.")
            return web.json_response({"status": "halted", "close_results": [str(r) for r in results]})
        except Exception as e:
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    async def handle_health(request):
        """P1.6: Health check endpoint for external monitoring."""
        try:
            # Check that the bot is still alive by verifying last_loop_ts is recent
            if last_loop_ts:
                from datetime import datetime as dt, timezone as tz
                last = datetime.fromisoformat(last_loop_ts.replace("Z", "+00:00"))
                age_sec = (datetime.now(tz.utc) - last).total_seconds()
            else:
                age_sec = None

            # Quick exchange reachability check
            try:
                state_h = await hyperliquid.get_user_state()
                exchange_ok = True
            except Exception:
                exchange_ok = False

            return web.json_response({
                "status": "ok",
                "last_loop_ts": last_loop_ts,
                "loop_age_sec": round(age_sec, 1) if age_sec is not None else None,
                "exchange_ok": exchange_ok,
                "positions_tracked": len(active_trades),
                "trading_halted": trading_halted,
                "invocation_count": invocation_count,
            })
        except Exception as e:
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    async def start_api(app):
        """Register HTTP endpoints for observing diary entries and logs."""
        app.router.add_get('/diary', handle_diary)
        app.router.add_get('/logs', handle_logs)
        app.router.add_post('/emergency-close', handle_emergency_close)
        app.router.add_get('/health', handle_health)  # P1.6

    async def main_async():
        """Start the aiohttp server and kick off the trading loop."""
        # Startup checks: validate wallet and fund perp account
        wallet_ok = await hyperliquid.validate_wallet()
        if not wallet_ok:
            logging.error("FATAL: Wallet validation failed. Re-authorize the API wallet on Hyperliquid.")
            sys.exit(1)
        await hyperliquid.ensure_perp_funded()

        # Phase 1 P1.3: reconcile persisted state with exchange state on startup
        await _startup_reconcile()

        app = web.Application()
        await start_api(app)
        from src.config_loader import CONFIG as CFG
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, CFG.get("api_host"), int(CFG.get("api_port")))
        await site.start()
        await run_loop()

    def calculate_total_return(state, trade_log):
        """Compute percent return relative to an assumed initial balance."""
        initial = 10000
        current = state['balance'] + sum(p.get('pnl', 0) for p in state.get('positions', []))
        return ((current - initial) / initial) * 100 if initial else 0

    def calculate_sharpe(returns):
        """Compute a naive Sharpe-like ratio from the trade log."""
        if not returns:
            return 0
        vals = [r.get('pnl', 0) if 'pnl' in r else 0 for r in returns]
        if not vals:
            return 0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 0
        return mean / std if std > 0 else 0

    async def check_exit_condition(trade, indicators, hyperliquid):
        """Evaluate whether a given trade's exit plan triggers a close."""
        plan = (trade.get("exit_plan") or "").lower()
        if not plan:
            return False
        try:
            if "macd" in plan and "below" in plan:
                macd = indicators.get_indicators(trade["asset"], "4h")["macd"].get("valueMACD")
                threshold = float(plan.split("below")[-1].strip())
                return macd is not None and macd < threshold
            if "close above ema50" in plan:
                ema50 = indicators.fetch_value("ema", trade["asset"], "4h", params={"period": 50})
                current = await hyperliquid.get_current_price(trade["asset"])
                return ema50 is not None and current > ema50
        except Exception:
            return False
        return False

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
