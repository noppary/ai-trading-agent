"""Local indicator calculations using Hyperliquid candle data + pandas-ta.

Replaces the external TAAPI.io dependency — no API key needed, no rate limits.
"""

import logging
import time

import pandas as pd
import pandas_ta as ta
import requests


# Hyperliquid REST API base URL
_HL_INFO_URL = "https://api.hyperliquid.xyz/info"


class HyperliquidIndicators:
    """Fetch OHLCV candles from Hyperliquid and compute TA indicators locally."""

    def __init__(self, hyperliquid_api=None):
        # We use direct HTTP — no SDK dependency needed for candles
        pass

    def _fetch_candles(self, asset: str, interval: str, count: int = 100) -> pd.DataFrame:
        """Fetch recent candles from Hyperliquid and return as a DataFrame.

        Args:
            asset: Market symbol (e.g. "BTC").
            interval: Candle interval (e.g. "5m", "1h", "4h").
            count: Number of candles to request.

        Returns:
            DataFrame with columns: open, high, low, close, volume.
        """
        now_ms = int(time.time() * 1000)
        interval_ms = self._interval_to_ms(interval)
        start_ms = now_ms - (count * interval_ms)

        payload = {
            "type": "candleSnapshot",
            "req": {"coin": asset, "interval": interval, "startTime": start_ms, "endTime": now_ms}
        }
        resp = requests.post(_HL_INFO_URL, json=payload, timeout=10)
        resp.raise_for_status()
        candles = resp.json()

        if not candles:
            logging.warning("No candles returned for %s %s", asset, interval)
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        # Hyperliquid candle format: {t, T, s, i, o, c, h, l, v, n}
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        return df

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        unit = interval[-1]
        val = int(interval[:-1])
        multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
        return val * multipliers.get(unit, 60_000)

    def _safe_series(self, series, n: int = 10) -> list:
        """Extract the last n values from a pandas Series, rounded."""
        if series is None or series.empty:
            return []
        values = series.dropna().tail(n).tolist()
        return [round(v, 4) if isinstance(v, (int, float)) else v for v in values]

    def _safe_value(self, series):
        """Extract the latest value from a pandas Series."""
        if series is None or series.empty:
            return None
        val = series.dropna().iloc[-1] if not series.dropna().empty else None
        if val is not None:
            try:
                return round(float(val), 4)
            except (TypeError, ValueError):
                return val
        return None

    def fetch_series(self, indicator: str, asset: str, interval: str,
                     results: int = 10, params: dict | None = None,
                     value_key: str = "value") -> list:
        """Drop-in replacement for TAAPIClient.fetch_series."""
        try:
            # Need enough candles for the indicator period + results
            period = (params or {}).get("period", 20)
            count = max(results + period + 50, 150)
            df = self._fetch_candles(asset, interval, count=count)
            if df.empty:
                return []

            series = self._compute_indicator(df, indicator, params)
            return self._safe_series(series, results)
        except Exception as e:
            logging.error("Indicator error %s %s %s: %s", indicator, asset, interval, e)
            return []

    def fetch_value(self, indicator: str, asset: str, interval: str,
                    params: dict | None = None, key: str = "value"):
        """Drop-in replacement for TAAPIClient.fetch_value."""
        try:
            period = (params or {}).get("period", 20)
            count = max(period + 50, 150)
            df = self._fetch_candles(asset, interval, count=count)
            if df.empty:
                return None

            series = self._compute_indicator(df, indicator, params)
            return self._safe_value(series)
        except Exception as e:
            logging.error("Indicator value error %s %s %s: %s", indicator, asset, interval, e)
            return None

    def get_indicators(self, asset: str, interval: str) -> dict:
        """Drop-in replacement for TAAPIClient.get_indicators."""
        try:
            df = self._fetch_candles(asset, interval, count=150)
            if df.empty:
                return {"rsi": None, "macd": {}, "sma": None, "ema": None, "bbands": {}}

            rsi = ta.rsi(df["close"], length=14)
            macd_df = ta.macd(df["close"])
            sma = ta.sma(df["close"], length=20)
            ema = ta.ema(df["close"], length=20)
            bbands_df = ta.bbands(df["close"], length=20)

            macd_result = {}
            if macd_df is not None and not macd_df.empty:
                macd_result = {
                    "valueMACD": self._safe_value(macd_df.iloc[:, 0]),
                    "valueMACDSignal": self._safe_value(macd_df.iloc[:, 1]),
                    "valueMACDHist": self._safe_value(macd_df.iloc[:, 2]),
                }

            bbands_result = {}
            if bbands_df is not None and not bbands_df.empty:
                bbands_result = {
                    "valueUpperBand": self._safe_value(bbands_df.iloc[:, 2]),
                    "valueMiddleBand": self._safe_value(bbands_df.iloc[:, 1]),
                    "valueLowerBand": self._safe_value(bbands_df.iloc[:, 0]),
                }

            return {
                "rsi": self._safe_value(rsi),
                "macd": macd_result,
                "sma": self._safe_value(sma),
                "ema": self._safe_value(ema),
                "bbands": bbands_result,
            }
        except Exception as e:
            logging.error("get_indicators error %s %s: %s", asset, interval, e)
            return {"rsi": None, "macd": {}, "sma": None, "ema": None, "bbands": {}}

    def get_historical_indicator(self, indicator: str, symbol: str, interval: str,
                                  results: int = 10, params: dict | None = None) -> list:
        """Drop-in replacement for TAAPIClient.get_historical_indicator."""
        asset = symbol.replace("/USDT", "").replace("/USD", "")
        series = self.fetch_series(indicator, asset, interval, results=results, params=params)
        return [{"value": v} for v in series]

    def _compute_indicator(self, df: pd.DataFrame, indicator: str,
                           params: dict | None = None) -> pd.Series:
        """Route an indicator name to the correct pandas-ta calculation."""
        params = params or {}
        period = params.get("period", 14)

        if indicator == "ema":
            return ta.ema(df["close"], length=period)
        elif indicator == "sma":
            return ta.sma(df["close"], length=period)
        elif indicator == "rsi":
            return ta.rsi(df["close"], length=period)
        elif indicator == "macd":
            result = ta.macd(df["close"])
            if result is not None and not result.empty:
                return result.iloc[:, 0]  # MACD line
            return pd.Series(dtype=float)
        elif indicator == "atr":
            return ta.atr(df["high"], df["low"], df["close"], length=period)
        elif indicator == "bbands":
            result = ta.bbands(df["close"], length=period)
            if result is not None and not result.empty:
                return result.iloc[:, 1]  # Middle band
            return pd.Series(dtype=float)
        else:
            logging.warning("Unknown indicator: %s", indicator)
            return pd.Series(dtype=float)
