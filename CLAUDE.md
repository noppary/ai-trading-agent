# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Install dependencies
~/.local/bin/poetry install --no-root

# Run the trading agent
poetry run python src/main.py --assets BTC ETH SOL --interval 5m

# Assets and interval can also be set via .env (ASSETS, INTERVAL)
poetry run python src/main.py

# Docker
docker build --platform linux/amd64 -t trading-agent .
docker run --rm -p 3000:3000 --env-file .env trading-agent
```

No test suite or linter is configured.

## Architecture

### 3-Stage LLM Pipeline

The core decision engine (`src/agent/decision_maker.py`) uses a cost-tiered multi-model pipeline, all routed through OpenRouter (or optionally Together AI for Stage 3):

```
Stage 1: Qwen3-8B   → Normalize raw market JSON into compact structured data
Stage 2: Qwen3-32B  → Classify signals per asset (bias, confidence, risk flags)
Stage 3: Kimi K2.5  → Final trade decisions with JSON Schema enforcement
```

Each stage falls back gracefully: Stage 1 passes raw data through, Stage 2 passes normalized data through, Stage 3 defaults to "hold all" on failure.

### Trading Loop (`src/main.py`)

1. Fetch account state + market data (prices, funding, OI) from Hyperliquid
2. Compute technical indicators locally via `HyperliquidIndicators` (pandas-ta on Hyperliquid candles)
3. Build context JSON with account, positions, indicators (5m + 4h timeframes)
4. Run 3-stage LLM pipeline → get `{reasoning, trade_decisions}` per asset
5. Execute buy/sell orders + place TP/SL triggers via `HyperliquidAPI`
6. Reconcile local active_trades against exchange state
7. Log to `diary.jsonl`, sleep until next interval

### Module Map

- **`src/config_loader.py`** — Loads `.env` into a `CONFIG` dict. All env vars are accessed through this module.
- **`src/agent/decision_maker.py`** — `TradingAgent` class with `decide_trade(assets, context)` public API. Routes through `_stage1_normalize` → `_stage2_signals` → `_stage3_decide`.
- **`src/indicators/hyperliquid_indicators.py`** — Fetches OHLCV candles from Hyperliquid REST API (`POST https://api.hyperliquid.xyz/info` with `candleSnapshot`), computes EMA/SMA/RSI/MACD/ATR/BBands via pandas-ta. Drop-in replacement for the deprecated `taapi_client.py`.
- **`src/trading/hyperliquid_api.py`** — Async wrapper around `hyperliquid-python-sdk`. Handles order placement, position queries, fills, funding rates, open interest. Uses websocket for price feeds.
- **`src/utils/formatting.py`** / **`prompt_utils.py`** — Number formatting and JSON serialization helpers (handles numpy floats, etc.).

### Observability

Local HTTP API runs on port 3000 (configurable via `API_PORT`):
- `GET /diary` — Trading decisions log (from `diary.jsonl`)
- `GET /logs` — LLM request logs

Local files: `diary.jsonl` (trade log), `prompts.log` (full LLM context), `llm_requests.log` (request metadata).

## Key Environment Variables

Required: `HYPERLIQUID_PRIVATE_KEY`, `OPENROUTER_API_KEY`

Pipeline models (with defaults): `STAGE1_MODEL` (qwen/qwen3-8b), `STAGE2_MODEL` (qwen/qwen3-32b), `STAGE3_MODEL` (moonshotai/kimi-k2.5), `STAGE3_PROVIDER` (together|openrouter)

Optional: `TOGETHER_API_KEY` (if Stage 3 uses Together), `HYPERLIQUID_NETWORK` (mainnet|testnet), `ASSETS`, `INTERVAL`

## Design Principles

- **Exchange state is truth**: Reconciliation always defers to Hyperliquid positions/orders over local tracking.
- **Perp-only pricing**: All prices come from perpetual mid-prices to avoid spot/perp basis mismatch.
- **Local indicators**: No external indicator API — candles fetched directly from Hyperliquid, computed with pandas-ta.
- **Structured outputs**: Stage 3 uses JSON Schema with `strict: true`; falls back to unstructured on 400/422.
- **Credential safety**: Never log API keys. The `taapi_client.py` has `_redact_url()` for URL-embedded secrets.
