"""3-stage LLM pipeline for trading decisions.

Stage 1: Qwen3-8B   — Parse + normalize raw market data     ($0.05/M)
Stage 2: Qwen3-32B  — Signal classification + risk checks   ($0.12/M)
Stage 3: Kimi K2.5  — Final trade decisions via Together AI  (213ms TTFT)
"""

import json
import logging
import concurrent.futures
from datetime import datetime

import requests

from src.config_loader import CONFIG

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

        # Models & Providers
        self.stage1_model = CONFIG["stage1_model"]
        self.stage1_provider = CONFIG.get("stage1_provider", "openrouter")
        self.stage2_model = CONFIG["stage2_model"]
        self.stage2_provider = CONFIG.get("stage2_provider", "openrouter")
        self.stage3_model = CONFIG["stage3_model"]
        self.stage3_provider = CONFIG.get("stage3_provider", "together")

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

        # MiniMax V2 ChatCompletion payload mapping
        # MiniMax uses 'model' and 'messages' same as OpenAI format in their V2 API
        resp = requests.post(self.minimax_url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            logging.error("MiniMax %s error: %s - %s", model, resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return resp.json()

    def _post_stage3(self, payload: dict) -> dict:
        poster = self._get_poster(self.stage3_provider)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(poster, payload)
            try:
                return future.result(timeout=_STAGE3_TIMEOUT)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Stage 3 timed out after {_STAGE3_TIMEOUT}s")

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
        except (KeyError, IndexError):
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
            resp = self._post_openrouter(payload)
            content = self._extract_content(resp)
            # Validate it's parseable JSON
            json.loads(content)
            logging.info("Stage 1 (normalize): %d chars → %d chars", len(raw_context), len(content))
            return content
        except Exception as e:
            logging.error("Stage 1 failed: %s — passing raw context", e)
            return raw_context

    # ── Stage 2: Signal Classification (Qwen3-32B) ───────────

    def _stage2_signals(self, normalized: str, assets: list) -> str:
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
                    "Rules:\n"
                    "- Require multi-timeframe confluence (5m + 4h alignment) for high confidence\n"
                    "- Counter-trend signals need extra confirmation → lower confidence\n"
                    "- If existing position has an exit_plan, respect it unless hard invalidation occurred\n"
                    "- Hysteresis: require stronger evidence to flip than to hold\n"
                    "- Funding is a tilt, not a trigger\n"
                    "- Output ONLY valid JSON, no markdown\n"
                    "- Do NOT use /no_think or any special tokens"
                )},
                {"role": "user", "content": normalized},
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
        try:
            resp = self._post_openrouter(payload)
            content = self._extract_content(resp)
            json.loads(content)
            logging.info("Stage 2 (signals): %d chars", len(content))
            return content
        except Exception as e:
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
                            "tp_price": {"type": ["number", "null"]},
                            "sl_price": {"type": ["number", "null"]},
                            "exit_plan": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["asset", "action", "allocation_usd", "tp_price", "sl_price", "exit_plan", "rationale"],
                    },
                }
            },
            "required": ["reasoning", "trade_decisions"],
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
                    "4. USE LEVERAGE: minimum 3x, maximum 10x. Reduce leverage when risk_flags are present or volatility is high.\n"
                    "5. TP/SL rules:\n"
                    "   - BUY: tp_price > current_price, sl_price < current_price\n"
                    "   - SELL: tp_price < current_price, sl_price > current_price\n"
                    "   - Use signal tp_zone/sl_zone as guidance\n"
                    "6. exit_plan must include at least ONE explicit invalidation trigger + optional cooldown.\n"
                    "7. Cooldown: after opening/flipping, wait at least 3 bars before changing direction.\n"
                    "8. If suggested_action is 'hold' and confidence < 0.5, always hold.\n\n"
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

                resp = self._post_stage3(payload)
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
            except requests.HTTPError as e:
                if attempt == 0 and e.response.status_code in (400, 422):
                    logging.warning("Stage 3: response_format rejected, retrying without")
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
