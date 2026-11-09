"""Market data providers.

Every tool talks to a :class:`MarketDataProvider`, never to a data vendor directly, so the
source can be swapped (another API, a database) and replaced by a fake in tests.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

import pandas as pd

from .errors import DataUnavailableError

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
T = TypeVar("T")


@runtime_checkable
class MarketDataProvider(Protocol):
    """Source of OHLCV history and company metadata."""

    def get_history(self, ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
        """Return a DataFrame indexed by date with columns Open, High, Low, Close, Volume.

        Prices are split/dividend adjusted. Raises DataUnavailableError when nothing is found.
        """
        ...

    def get_info(self, ticker: str) -> dict[str, Any]:
        """Return descriptive metadata (name, currency, exchange, sector, market cap...)."""
        ...


def clean_history(df: pd.DataFrame | None, ticker: str) -> pd.DataFrame:
    """Normalise a raw OHLCV frame: known columns, naive daily DatetimeIndex, sorted."""
    if df is None or df.empty or "Close" not in df:
        raise DataUnavailableError(
            f"No price data found for '{ticker}'. Check the symbol (Yahoo Finance format, e.g. "
            "'SHOP.TO' for Toronto listings) or try a longer period."
        )
    df = df[[c for c in OHLCV_COLUMNS if c in df.columns]].dropna(subset=["Close"]).copy()
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    df.index = index.normalize()
    df.index.name = "Date"
    return df.sort_index()


class YFinanceProvider:
    """Free market data from Yahoo Finance through the ``yfinance`` package."""

    _INFO_KEYS = {
        "shortName": "name",
        "longName": "long_name",
        "currency": "currency",
        "exchange": "exchange",
        "quoteType": "type",
        "sector": "sector",
        "industry": "industry",
        "marketCap": "market_cap",
        "trailingPE": "trailing_pe",
        "forwardPE": "forward_pe",
        "dividendYield": "dividend_yield",
        "beta": "beta",
    }

    def get_history(self, ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
        import yfinance as yf  # imported lazily: tests never touch the network stack

        try:
            raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        except Exception as exc:  # yfinance raises many error types on bad symbols / network
            raise DataUnavailableError(f"Could not download data for '{ticker}': {exc}") from exc
        return clean_history(raw, ticker)

    def get_info(self, ticker: str) -> dict[str, Any]:
        import yfinance as yf

        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            return {}  # metadata is optional; prices remain the source of truth
        return {out: info[key] for key, out in self._INFO_KEYS.items() if info.get(key) is not None}


class CachedProvider:
    """Thread-safe in-memory TTL cache in front of another provider.

    A conversation often asks for the same series several times (quote, then indicators,
    then a backtest); caching keeps the server fast and polite to the upstream API.
    """

    def __init__(
        self, inner: MarketDataProvider, ttl_seconds: float = 300.0, max_entries: int = 256
    ) -> None:
        self.inner = inner
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[tuple, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _copy(value: T) -> T:
        return value.copy() if isinstance(value, pd.DataFrame | dict) else value

    def _cached(self, key: tuple, loader: Callable[[], T]) -> T:
        with self._lock:
            entry = self._store.get(key)
            if entry and time.monotonic() - entry[0] < self.ttl:
                self.hits += 1
                return self._copy(entry[1])
        value = loader()  # outside the lock: network calls must not serialise all requests
        with self._lock:
            self.misses += 1
            if key not in self._store and len(self._store) >= self.max_entries:
                oldest = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest]
            self._store[key] = (time.monotonic(), value)
        return self._copy(value)

    def get_history(self, ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
        return self._cached(
            ("history", ticker, period, interval),
            lambda: self.inner.get_history(ticker, period, interval),
        )

    def get_info(self, ticker: str) -> dict[str, Any]:
        return self._cached(("info", ticker), lambda: self.inner.get_info(ticker))

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def default_provider() -> MarketDataProvider:
    """Yahoo Finance behind a cache whose TTL can be set with STOCK_MCP_CACHE_TTL (seconds)."""
    ttl = float(os.environ.get("STOCK_MCP_CACHE_TTL", "300"))
    return CachedProvider(YFinanceProvider(), ttl_seconds=ttl)
