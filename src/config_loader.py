"""Centralized environment variable loading for the trading agent configuration."""

import json
import os
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Fetch an environment variable with optional default and required validation."""
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"Invalid float for {name}: {raw}")

def _get_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {raw}") from exc


def _get_json(name: str, default: dict | None = None) -> dict | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Environment variable {name} must be a JSON object")
        return parsed
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON for {name}: {raw}") from exc


def _get_list(name: str, default: list[str] | None = None) -> list[str] | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    # Support JSON-style lists
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise RuntimeError(f"Environment variable {name} must be a list if using JSON syntax")
            return [str(item).strip().strip('"\'') for item in parsed if str(item).strip()]
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON list for {name}: {raw}") from exc
    # Fallback: comma separated string
    values = []
    for item in raw.split(","):
        cleaned = item.strip().strip('"\'')
        if cleaned:
            values.append(cleaned)
    return values or default


CONFIG = {
    "taapi_api_key": _get_env("TAAPI_API_KEY"),  # No longer required — using local indicators
    "hyperliquid_private_key": _get_env("HYPERLIQUID_PRIVATE_KEY") or _get_env("LIGHTER_PRIVATE_KEY"),
    "mnemonic": _get_env("MNEMONIC"),
    # Hyperliquid network/base URL overrides
    "hyperliquid_base_url": _get_env("HYPERLIQUID_BASE_URL"),
    "hyperliquid_network": _get_env("HYPERLIQUID_NETWORK", "mainnet"),
    "hyperliquid_main_wallet": _get_env("HYPERLIQUID_MAIN_WALLET"),
    "hyperliquid_api_wallet": _get_env("HYPERLIQUID_API_WALLET") or _get_env("HYPERLIQUID_WALLET_ADDRESS"),
    # LLM via OpenRouter (Stage 1 + Stage 2)
    "openrouter_api_key": _get_env("OPENROUTER_API_KEY", required=True),
    "openrouter_base_url": _get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    "openrouter_referer": _get_env("OPENROUTER_REFERER"),
    "openrouter_app_title": _get_env("OPENROUTER_APP_TITLE", "trading-agent"),
    # Together AI (Stage 3 — trade decisions)
    "together_api_key": _get_env("TOGETHER_API_KEY"),
    "together_base_url": _get_env("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
    # MiniMax API (Direct)
    "minimax_api_key": _get_env("MINIMAX_API_KEY"),
    "minimax_base_url": _get_env("MINIMAX_BASE_URL", "https://api.minimax.io/v1/text/chatcompletion_v2"),
    # Multi-model pipeline
    "stage1_model": _get_env("STAGE1_MODEL", "qwen/qwen3-8b"),            # Parse + normalize ($0.05/M)
    "stage1_provider": _get_env("STAGE1_PROVIDER", "openrouter"),
    "stage2_model": _get_env("STAGE2_MODEL", "qwen/qwen3-32b"),           # Signals + risk ($0.12/M)
    "stage2_provider": _get_env("STAGE2_PROVIDER", "openrouter"),
    "stage3_model": _get_env("STAGE3_MODEL", "moonshotai/kimi-k2.5"),     # Trade decisions (Together AI)
    "stage3_provider": _get_env("STAGE3_PROVIDER", "together"),            # "together" or "openrouter" or "minimax"
    # Legacy single-model (fallback)
    "llm_model": _get_env("LLM_MODEL", "x-ai/grok-4"),
    # Reasoning tokens
    "reasoning_enabled": _get_bool("REASONING_ENABLED", False),
    "reasoning_effort": _get_env("REASONING_EFFORT", "high"),
    # Provider routing
    "provider_config": _get_json("PROVIDER_CONFIG"),
    "provider_quantizations": _get_list("PROVIDER_QUANTIZATIONS"),
    # Runtime controls via env
    "assets": _get_env("ASSETS"),  # e.g., "BTC ETH SOL" or "BTC,ETH,SOL"
    "interval": _get_env("INTERVAL"),  # e.g., "5m", "1h"
    # API server
    "api_host": _get_env("API_HOST", "127.0.0.1"),
    "dry_run_mode": _get_bool("DRY_RUN_MODE", True),
    "api_port": _get_env("APP_PORT") or _get_env("API_PORT") or "3000",
    # Circuit breaker / risk limits
    "max_drawdown_pct": _get_int("MAX_DRAWDOWN_PCT", 15),
    "daily_loss_limit_usd": _get_int("DAILY_LOSS_LIMIT_USD", 50),   # legacy, superseded by daily_loss_limit_pct
    "daily_loss_limit_pct": _get_float("DAILY_LOSS_LIMIT_PCT", 3.0),  # P2.3: % of daily_start_value
    "max_position_pct": _get_int("MAX_POSITION_PCT", 25),
    "max_total_exposure_pct": _get_int("MAX_TOTAL_EXPOSURE_PCT", 75),
    "max_open_positions": _get_int("MAX_OPEN_POSITIONS", 3),          # P2.10: cap simultaneous positions
    "max_leverage": _get_int("MAX_LEVERAGE", 5),
    "consecutive_failure_limit": _get_int("CONSECUTIVE_FAILURE_LIMIT", 10),
    "trade_cooldown_bars": _get_int("TRADE_COOLDOWN_BARS", 3),        # P2.5: min bars between trades per asset
}
