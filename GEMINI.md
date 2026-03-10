# GEMINI.md

This file provides project-specific guidance, architectural overview, and development standards for the AI Trading Agent (Nocturne) on Hyperliquid.

## Project Overview

Nocturne is an AI-powered trading agent that uses a multi-stage LLM pipeline to analyze market data from the Hyperliquid decentralized exchange and execute automated trading decisions. It utilizes technical indicators computed locally and leverages models like Qwen and Kimi via OpenRouter for decision-making.

- **Main Technologies**: Python 3.12+, Poetry, Hyperliquid SDK, pandas-ta, OpenRouter (LLM API), Docker.
- **Key Concepts**: 3-Stage LLM Pipeline (Normalize → Signal → Decide), Local indicators (Hyperliquid candles), Perpetual-only pricing.

## Getting Started

### Building and Running

Ensure you have [Poetry](https://python-poetry.org/) installed.

```bash
# Install dependencies
poetry install --no-root

# Run the trading agent (specify assets and interval)
poetry run python src/main.py --assets BTC ETH SOL --interval 5m

# Run with defaults from .env
poetry run python src/main.py

# Docker Build & Run
docker build --platform linux/amd64 -t trading-agent .
docker run --rm -p 3000:3000 --env-file .env trading-agent
```

### Tests and Linting
- **TODO**: Currently, no formal test suite or linter is configured. Adding `pytest` and `ruff` is recommended for future development.

## Architecture

### 3-Stage LLM Pipeline (`src/agent/decision_maker.py`)
1. **Stage 1 (Normalize)**: Processes raw market JSON into compact structured data (e.g., using `qwen/qwen3-8b`).
2. **Stage 2 (Signal)**: Classifies per-asset signals (bias, confidence, risk) (e.g., using `qwen/qwen3-32b`).
3. **Stage 3 (Decide)**: Final trade decisions with JSON Schema enforcement (e.g., using `moonshotai/kimi-k2.5`).

### Key Modules
- `src/main.py`: Entry point and main trading loop (Account state → Context → Pipeline → Execution → Log).
- `src/agent/decision_maker.py`: Core decision engine implementing the 3-stage pipeline.
- `src/indicators/hyperliquid_indicators.py`: Fetches OHLCV candles from Hyperliquid and computes indicators (EMA, RSI, MACD, etc.) using `pandas-ta`.
- `src/trading/hyperliquid_api.py`: Async wrapper for Hyperliquid SDK (orders, positions, fills, funding).
- `src/config_loader.py`: Centralized configuration management from `.env`.

## Development Conventions

- **State Management**: Always defer to Hyperliquid's exchange state (positions/orders) as the single source of truth; reconcile local state against the exchange.
- **Indicator Source**: Use local indicators computed from Hyperliquid candles via `pandas-ta`. Do not rely on external indicator APIs.
- **Pricing**: Use perpetual mid-prices for all calculations to avoid spot/perp basis mismatch.
- **Structured Output**: Ensure Stage 3 LLM responses strictly follow the defined JSON Schema.
- **Security**: Never log or commit private keys or API keys. Use `src/config_loader.py` for all environment variable access.
- **Logging**: Detailed logs are maintained in `diary.jsonl` (trades), `prompts.log` (LLM context), and `llm_requests.log` (metadata).

## Environment Configuration

Copy `.env.example` to `.env` and configure:
- `HYPERLIQUID_PRIVATE_KEY`: Your trading account private key.
- `OPENROUTER_API_KEY`: For LLM access.
- `STAGE[1,2,3]_MODEL`: Specify models (defaults to Qwen and Kimi models).
- `ASSETS`: Comma-separated list (e.g., `BTC,ETH,SOL`).
- `INTERVAL`: Trading interval (e.g., `5m`, `1h`).

## Observability

The agent runs a minimal HTTP API (default port `3000`):
- `GET /diary`: Returns recent trading decisions from `diary.jsonl`.
- `GET /logs`: Tails specified log files (e.g., `llm_requests.log`).
