"""3-stage LLM pipeline for trading decisions.

Stage 1: Qwen3-8B   — Parse + normalize raw market data     ($0.05/M)
Stage 2: Qwen3-32B  — Signal classification + risk checks   ($0.12/M)
Stage 3: Kimi K2.5  — Final trade decisions via Together AI  (213ms TTFT)
"""

import json
import logging
import time
import concurrent.futures
from collections import deque
from datetime import datetime
from pathlib import Path

import requests

from src.config_loader import CONFIG
from src.utils.metrics import LLM_LATENCY, LLM_ERRORS

_STAGE3_TIMEOUT = 90  # hard wall-clock limit for Stage 3 (seconds)


class TradingAgent:
    """Multi-model trading agent with a 3-stage decision pipeline."""

    def __init__(self):
        # OpenRouter (Stage 1 + 2)
        self.openrouter_key = CONFIG["openrouter_api_key"]
        self.openrouter_url = f"{CONFIG['openrouter_base_url']}/chat/completions"
        self.referer = CONFIG.get("openrouter_referer")
        self.app_title = CONFIG.get("openrouter_app_title")

        # Together AI (Stage 3)
        self.together_key = CONFIG.get("together_api_key")
        self.together_url = f"{CONFIG['together_base_url']}/chat/completions"

        # MiniMax (Direct)
        self.minimax_key = CONFIG.get("minimax_api_key")
        self.minimax_url = CONFIG.get("minimax_base_url")

        # P2.8: Per-stage latency tracking (keep last 20 per stage for p95)
        self._latency: dict[str, deque] = {
            "stage1": deque(maxlen=20),
            "stage2": deque(maxlen=20),
            "stage3": deque(maxlen=20),
        }
        self._latency_thresholds = {"stage1": 30, "stage2": 45, "stage3": 60}  # seconds

        # Models & Providers
        self.stage1_model = CONFIG["stage1_model"]
        self.stage1_provider = CONFIG.get("stage1_provider", "openrouter")
        self.stage2_model = CONFIG["stage2_model"]
        self.stage2_provider = CONFIG.get("stage2_provider", "openrouter")
        self.stage3_model = CONFIG["stage3_model"]
        self.stage3_provider = CONFIG.get("stage3_provider", "together")

    # ── P2.8: Latency helpers ──────────────────────────────────

    def _record_latency(self, stage: str, elapsed: float):
        """Record stage latency and log a warning if p95 exceeds threshold."""
        self._latency[stage].append(elapsed)
        LLM_LATENCY.labels(stage=stage).observe(elapsed)
        logging.info("LLM latency %s: %.1fs", stage, elapsed)
        if len(self._latency[stage]) >= 5:
            sorted_vals = sorted(self._latency[stage])
            p95_idx = int(len(sorted_vals) * 0.95)
            p95 = sorted_vals[min(p95_idx, len(sorted_vals) - 1)]
            threshold = self._latency_thresholds.get(stage, 60)
            if p95 > threshold:
                logging.warning("LLM DEGRADATION: %s p95=%.1fs exceeds %ds threshold", stage, p95, threshold)

    def get_latency_stats(self) -> dict:
        """Return latency stats per stage for /metrics or /health."""
        stats = {}
        for stage, vals in self._latency.items():
            if vals:
                sorted_vals = sorted(vals)
                p95_idx = int(len(sorted_vals) * 0.95)
                stats[stage] = {
                    "count": len(vals),
                    "last": round(sorted_vals[-1], 2),
                    "p50": round(sorted_vals[len(sorted_vals) // 2], 2),
                    "p95": round(sorted_vals[min(p95_idx, len(sorted_vals) - 1)], 2),
                }
            else:
                stats[stage] = {"count": 0, "last": None, "p50": None, "p95": None}
        return stats

    # ── P3.8: Rate limit tracking ────────────────────────────

    def _check_rate_limits(self, resp: requests.Response, provider: str):
        """Parse rate limit headers and warn when approaching limits."""
        remaining = resp.headers.get("x-ratelimit-remaining") or resp.headers.get("X-RateLimit-Remaining")
        limit = resp.headers.get("x-ratelimit-limit") or resp.headers.get("X-RateLimit-Limit")
        retry_after = resp.headers.get("retry-after") or resp.headers.get("Retry-After")

        if retry_after:
            logging.warning("RATE LIMIT: %s Retry-After=%s — backing off", provider, retry_after)
            try:
                time.sleep(min(float(retry_after), 30))
            except (ValueError, TypeError):
                time.sleep(5)
            return

        if remaining is not None and limit is not None:
            try:
                rem, lim = int(remaining), int(limit)
                usage_pct = ((lim - rem) / lim * 100) if lim > 0 else 0
                if rem <= 5 or usage_pct > 80:
                    logging.warning("RATE LIMIT: %s %d/%d remaining (%.0f%% used)", provider, rem, lim, usage_pct)
            except (ValueError, TypeError):
                pass

    # ── HTTP helpers ──────────────────────────────────────────

    def _get_poster(self, provider: str):
        """Return the appropriate posting method for the given provider."""
        provider = provider.lower()
        if provider == "openrouter":
            return self._post_openrouter
        if provider == "together":
            return self._post_together
        if provider == "minimax":
            return self._post_minimax
        raise ValueError(f"Unsupported provider: {provider}")

    def _post_openrouter(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.app_title:
            headers["X-Title"] = self.app_title

        model = payload.get("model", "?")
        logging.info("Stage request → OpenRouter (%s)", model)
        self._log_request(model, payload)

        resp = requests.post(self.openrouter_url, headers=headers, json=payload, timeout=60)
        self._check_rate_limits(resp, "openrouter")
        if resp.status_code != 200:
            logging.error("OpenRouter %s error: %s - %s", model, resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return resp.json()

    def _post_together(self, payload: dict) -> dict:
        if not self.together_key:
            raise RuntimeError("TOGETHER_API_KEY not set — required for Together provider")
        headers = {
            "Authorization": f"Bearer {self.together_key}",
            "Content-Type": "application/json",
        }
        model = payload.get("model", "?")
        logging.info("Stage request → Together AI (%s)", model)
        self._log_request(model, payload)

        resp = requests.post(self.together_url, headers=headers, json=payload, timeout=60)
        self._check_rate_limits(resp, "together")
        if resp.status_code != 200:
            logging.error("Together %s error: %s - %s", model, resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return resp.json()

    def _post_minimax(self, payload: dict) -> dict:
        if not self.minimax_key:
            raise RuntimeError("MINIMAX_API_KEY not set — required for MiniMax provider")
        headers = {
            "Authorization": f"Bearer {self.minimax_key}",
            "Content-Type": "application/json",
        }
        model = payload.get("model", "?")
        logging.info("Stage request → MiniMax Direct (%s)", model)
        self._log_request(model, payload)

        resp = requests.post(self.minimax_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # MiniMax V2 API specific error handling (they return 200 even for logic errors)
        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            err_msg = base_resp.get("status_msg", "Unknown MiniMax error")
            logging.error("MiniMax API error (%s): %s - %s", model, base_resp.get("status_code"), err_msg)
            raise RuntimeError(f"MiniMax API error: {err_msg}")

        return data

    def _post_stage3(self, payload: dict) -> dict:
        """P2.7: Try primary provider, then fallback chain before giving up."""
        # Build fallback chain: primary → openrouter (if primary isn't already openrouter)
        providers = [self.stage3_provider]
        if self.stage3_provider != "openrouter":
            providers.append("openrouter")

        last_err = None
        for provider in providers:
            try:
                poster = self._get_poster(provider)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(poster, payload)
                    try:
                        result = future.result(timeout=_STAGE3_TIMEOUT)
                        if provider != self.stage3_provider:
                            logging.warning("Stage 3 succeeded via FALLBACK provider: %s", provider)
                        return result
                    except concurrent.futures.TimeoutError:
                        last_err = TimeoutError(f"Stage 3 timed out after {_STAGE3_TIMEOUT}s on {provider}")
                        logging.warning("Stage 3 timeout on %s — trying next provider", provider)
            except Exception as e:
                last_err = e
                logging.warning("Stage 3 failed on %s: %s — trying next provider", provider, e)

        raise last_err or RuntimeError("Stage 3 all providers exhausted")

    def _log_request(self, model: str, payload: dict):
        try:
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n=== {datetime.now()} ===\n")
                f.write(f"Model: {model}\n")
                f.write(f"Payload size: {len(json.dumps(payload))} chars\n")
        except Exception:
            pass

    def _extract_content(self, resp_json: dict) -> str:
        try:
            return resp_json["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    # ── Stage 1: Parse + Normalize (Qwen3-8B) ────────────────

    def _stage1_normalize(self, raw_context: str) -> str:
        payload = {
            "model": self.stage1_model,
            "messages": [
                {"role": "system", "content": (
                    "You are a market data normalizer for a crypto trading system.\n"
                    "Input: raw JSON with account state, positions, market data (prices, indicators, funding, OI).\n"
                    "Output: a concise, structured JSON summary with these sections:\n"
                    "1. account: {balance, total_value, return_pct, open_positions: [{symbol, side, size, entry, current, pnl, leverage}]}\n"
                    "2. markets: [{asset, price, trend_5m, trend_4h, indicators: {rsi14, macd, ema20, ema50, atr14}, funding_rate, open_interest}]\n"
                    "3. active_trades: [{asset, side, entry, tp, sl, exit_plan}]\n"
                    "4. recent_fills: [{asset, side, price, time}]\n\n"
                    "Rules:\n"
                    "- trend_5m/trend_4h: classify as 'bullish', 'bearish', or 'neutral' based on EMA crossovers and MACD\n"
                    "- Round prices to 2 decimals, indicators to 4\n"
                    "- Strip noise, keep only actionable data\n"
                    "- Output ONLY valid JSON, no markdown or prose\n"
                    "- Do NOT use /no_think or any special tokens"
                )},
                {"role": "user", "content": raw_context},
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
        try:
            t0 = time.monotonic()
            poster = self._get_poster(self.stage1_provider)
            resp = poster(payload)
            self._record_latency("stage1", time.monotonic() - t0)
            content = self._extract_content(resp)
            # Validate it's parseable JSON
            json.loads(content)
            logging.info("Stage 1 (normalize): %d chars → %d chars", len(raw_context), len(content))
            return content
        except Exception as e:
            LLM_ERRORS.labels(stage="stage1").inc()
            logging.error("Stage 1 failed: %s — passing raw context", e)
            return raw_context

    # ── Stage 2: Signal Classification (Qwen3-32B) ───────────

    def _load_learnings(self) -> str:
        """Load learnings.json and format relevant entries for prompt injection."""
        try:
            learnings_path = Path("/root/ai-trading-agent/data/learnings.json")
            if not learnings_path.exists():
                return ""
            data = json.loads(learnings_path.read_text(encoding="utf-8"))
            parts = []
            patterns = data.get("patterns", [])
            if patterns:
                recent = patterns[-10:]  # last 10 patterns
                parts.append("Recent patterns observed:\n" + "\n".join(
                    f"- [{p.get('symbol', '?')}] {p.get('value', p.get('action', ''))}" for p in recent
                ))
            changes = data.get("strategy_changes", [])
            if changes:
                recent = changes[-5:]
                parts.append("Recent strategy adjustments:\n" + "\n".join(
                    f"- [{c.get('symbol', '?')}] {c.get('action', '')}: {c.get('value', '')}" for c in recent
                ))
            return "\n\n".join(parts)
        except Exception as e:
            logging.warning("Failed to load learnings: %s", e)
            return ""

    def _stage2_signals(self, normalized: str, assets: list) -> str:
        # Load learnings for prompt injection
        learnings_block = self._load_learnings()
        learnings_section = ""
        if learnings_block:
            learnings_section = (
                "\n\n## Learned Patterns & Adjustments (from self-improvement system)\n"
                "Incorporate these observations into your analysis:\n"
                f"{learnings_block}\n"
            )

        payload = {
            "model": self.stage2_model,
            "messages": [
                {"role": "system", "content": (
                    "You are a quantitative signal classifier for crypto perpetual futures.\n"
                    "Input: normalized market data JSON.\n"
                    "Output: a JSON object with signal analysis per asset.\n\n"
                    "For each asset, provide:\n"
                    "{\n"
                    '  "signals": [{\n'
                    '    "asset": "BTC",\n'
                    '    "bias": "long" | "short" | "neutral",\n'
                    '    "confidence": 0.0-1.0,\n'
                    '    "market_regime": "trending_up" | "trending_down" | "ranging" | "breakout",\n'
                    '    "structure": "bullish/bearish/ranging — describe EMA alignment, HH/HL vs LH/LL",\n'
                    '    "momentum": "describe MACD regime + RSI slope",\n'
                    '    "volatility": "ATR context — expanding/contracting/normal",\n'
                    '    "funding_tilt": "positive/negative/neutral — is funding aiding or hurting?",\n'
                    '    "risk_flags": ["list any warnings: divergence, extreme RSI, high leverage, etc."],\n'
                    '    "suggested_action": "buy/sell/hold",\n'
                    '    "suggested_leverage": 1-10,\n'
                    '    "suggested_allocation_pct": 0-100,\n'
                    '    "tp_zone": "price range or null",\n'
                    '    "sl_zone": "price range or null",\n'
                    '    "reasoning": "2-3 sentence first-principles analysis"\n'
                    "  }]\n"
                    "}\n\n"
                    "## Market Regime Classification\n"
                    "FIRST classify the market regime for each asset using these rules:\n"
                    "- trending_up: EMA20 > EMA50 AND price > EMA20 — momentum is bullish\n"
                    "- trending_down: EMA20 < EMA50 AND price < EMA20 — momentum is bearish\n"
                    "- ranging: EMA20 and EMA50 are within 0.5% of each other, ATR is contracting\n"
                    "- breakout: ATR expanding >20% above its 14-period average, price breaking key EMA\n\n"
                    "## RSI Interpretation by Regime (CRITICAL)\n"
                    "- In trending_up: RSI 55-75 CONFIRMS bullish momentum — this is NOT overbought\n"
                    "  Only flag overbought if RSI > 80 in a trend or RSI > 70 in ranging market\n"
                    "- In trending_down: RSI 25-45 CONFIRMS bearish momentum — this is NOT oversold\n"
                    "  Only flag oversold if RSI < 20 in a trend or RSI < 30 in ranging market\n"
                    "- In ranging: standard thresholds apply (overbought >70, oversold <30)\n"
                    "- DO NOT default to neutral/hold just because RSI is above 50\n\n"
                    "## Anti-Stall Rule\n"
                    "- You are part of a TRADING system, not a watching system\n"
                    "- If the data shows clear trend alignment (EMA crossover + confirming RSI + MACD),\n"
                    "  you MUST suggest buy or sell — do not hold in confirmed trends\n"
                    "- Only suggest hold when there is genuine conflicting evidence across indicators\n"
                    "- Confidence for trend-aligned signals should be >= 0.5\n\n"
                    "Rules:\n"
                    "- Require multi-timeframe confluence (5m + 4h alignment) for high confidence\n"
                    "- Counter-trend signals need extra confirmation → lower confidence\n"
                    "- If existing position has an exit_plan, respect it unless hard invalidation occurred\n"
                    "- Hysteresis: require stronger evidence to flip than to hold\n"
                    "- Funding is a tilt, not a trigger\n"
                    "- Output ONLY valid JSON, no markdown\n"
                    "- Do NOT use /no_think or any special tokens"
                    + learnings_section
                )},
                {"role": "user", "content": normalized},
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
        try:
            t0 = time.monotonic()
            poster = self._get_poster(self.stage2_provider)
            resp = poster(payload)
            self._record_latency("stage2", time.monotonic() - t0)
            content = self._extract_content(resp)
            json.loads(content)
            logging.info("Stage 2 (signals): %d chars", len(content))
            return content
        except Exception as e:
            LLM_ERRORS.labels(stage="stage2").inc()
            logging.error("Stage 2 failed: %s — passing normalized data through", e)
            return normalized

    # ── Stage 3: Trade Decisions (Kimi K2.5 via Together AI) ──

    def _stage3_decide(self, signals: str, assets: list) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "trade_decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "asset": {"type": "string"},
                            "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                            "allocation_usd": {"type": "number"},
                            "tp_price": {"type": "number"},
                            "sl_price": {"type": "number"},
                            "exit_plan": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["asset", "action", "allocation_usd", "tp_price", "sl_price", "exit_plan", "rationale"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["reasoning", "trade_decisions"],
            "additionalProperties": False,
        }

        payload = {
            "model": self.stage3_model,
            "messages": [
                {"role": "system", "content": (
                    "You are the EXECUTION LAYER of a quantitative trading system for crypto perpetual futures.\n"
                    "You receive pre-analyzed signals from upstream classifiers. Your job: make the FINAL trade decision.\n\n"
                    f"Assets: {json.dumps(assets)}\n\n"
                    "Rules:\n"
                    "1. Trust the upstream signal analysis — don't second-guess the data, focus on execution quality.\n"
                    "2. For each asset, decide: buy, sell, or hold.\n"
                    "3. Set allocation_usd based on confidence and suggested_allocation_pct from signals.\n"
                    "   - Confidence >= 0.5: use full suggested allocation\n"
                    "   - Confidence 0.35-0.5: use 50% of suggested allocation (reduced-size entry)\n"
                    "   - Confidence < 0.35: hold\n"
                    "4. USE LEVERAGE: minimum 2x, maximum 10x. Reduce leverage when risk_flags are present or volatility is high.\n"
                    "5. TP/SL rules:\n"
                    "   - BUY: tp_price > current_price, sl_price < current_price\n"
                    "   - SELL: tp_price < current_price, sl_price > current_price\n"
                    "   - Use signal tp_zone/sl_zone as guidance\n"
                    "6. exit_plan must include at least ONE explicit invalidation trigger + optional cooldown.\n"
                    "7. Cooldown: after opening/flipping, wait at least 3 bars before changing direction.\n"
                    "8. If suggested_action is 'hold' and confidence < 0.35, always hold.\n"
                    "   If confidence is 0.35-0.5, enter at HALF allocation to test the signal.\n\n"
                    "Output a JSON object with 'reasoning' (string) and 'trade_decisions' (array).\n"
                    "Each decision: {asset, action, allocation_usd, tp_price, sl_price, exit_plan, rationale}\n"
                    "Output ONLY valid JSON. No markdown, no prose outside the JSON."
                )},
                {"role": "user", "content": signals},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }

        # Try with response_format first, fall back without
        for attempt in range(2):
            try:
                if attempt == 0:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "trade_decisions", "strict": True, "schema": schema},
                    }
                else:
                    payload.pop("response_format", None)

                t0 = time.monotonic()
                resp = self._post_stage3(payload)
                self._record_latency("stage3", time.monotonic() - t0)
                content = self._extract_content(resp)
                parsed = json.loads(content)

                if isinstance(parsed, dict) and "trade_decisions" in parsed:
                    decisions = parsed["trade_decisions"]
                    for item in decisions:
                        item.setdefault("allocation_usd", 0.0)
                        item.setdefault("tp_price", None)
                        item.setdefault("sl_price", None)
                        item.setdefault("exit_plan", "")
                        item.setdefault("rationale", "")
                    logging.info("Stage 3 (decide): %d decisions", len(decisions))
                    return parsed

                logging.warning("Stage 3: unexpected format, retrying")
            except (requests.HTTPError, RuntimeError) as e:
                if attempt == 0:
                    logging.warning("Stage 3: response_format rejected (%s), retrying without", e)
                    continue
                raise
            except TimeoutError as e:
                logging.error("Stage 3 timeout: %s — defaulting to hold", e)
                break
            except (json.JSONDecodeError, KeyError) as e:
                logging.error("Stage 3 parse error: %s", e)
                if attempt == 0:
                    continue
                break

        # Fallback: hold everything
        logging.error("Stage 3 failed — defaulting to hold")
        return {
            "reasoning": "Stage 3 pipeline failure — holding all positions",
            "trade_decisions": [{
                "asset": a, "action": "hold", "allocation_usd": 0.0,
                "tp_price": None, "sl_price": None,
                "exit_plan": "", "rationale": "pipeline failure"
            } for a in assets]
        }

    # ── Public API ────────────────────────────────────────────

    def decide_trade(self, assets, context) -> dict:
        """Run the full 3-stage pipeline: normalize → classify → decide."""
        logging.info("═══ Pipeline start: %d assets ═══", len(assets))

        # Stage 1: Qwen3-8B normalizes raw data
        normalized = self._stage1_normalize(context)

        # Stage 2: Qwen3-32B classifies signals + risk
        signals = self._stage2_signals(normalized, assets)

        # Stage 3: Kimi K2.5 makes final trade decisions
        result = self._stage3_decide(signals, assets)

        logging.info("═══ Pipeline complete ═══")
        return result
