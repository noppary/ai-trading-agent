"""P2.2: Prometheus metrics for the trading agent."""

from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
)

# ── Trading metrics ──────────────────────────────────────────────────────────

ACCOUNT_VALUE = Gauge("trading_account_value_usd", "Current account value in USD")
POSITIONS_COUNT = Gauge("trading_positions_count", "Number of active tracked positions")
TRADING_HALTED = Gauge("trading_halted", "1 if circuit breaker is active, 0 otherwise")

ORDERS_TOTAL = Counter("trading_orders_total", "Total orders placed", ["asset", "side"])
SLIPPAGE_BPS = Summary("trading_slippage_bps", "Order slippage in basis points")

NET_EXPOSURE_PCT = Gauge("trading_net_exposure_pct", "Net directional exposure as % of account")
DRAWDOWN_PCT = Gauge("trading_drawdown_pct", "Current drawdown from peak")

# ── Loop metrics ─────────────────────────────────────────────────────────────

LOOP_DURATION = Histogram(
    "trading_loop_duration_seconds",
    "Duration of one trading loop iteration",
    buckets=[5, 10, 20, 30, 60, 90, 120, 180, 300],
)
LOOP_COUNT = Counter("trading_loop_iterations_total", "Total loop iterations")
LOOP_ERRORS = Counter("trading_loop_errors_total", "Total loop-level errors")

# ── LLM metrics ──────────────────────────────────────────────────────────────

LLM_LATENCY = Histogram(
    "trading_llm_latency_seconds",
    "LLM call latency per stage",
    ["stage"],
    buckets=[1, 2, 5, 10, 15, 20, 30, 45, 60, 90],
)
LLM_ERRORS = Counter("trading_llm_errors_total", "LLM call failures", ["stage"])
PIPELINE_FAILURES = Counter("trading_pipeline_failures_total", "Consecutive pipeline failures")


def metrics_response():
    """Return (body, content_type) for an aiohttp response."""
    return generate_latest(), CONTENT_TYPE_LATEST
