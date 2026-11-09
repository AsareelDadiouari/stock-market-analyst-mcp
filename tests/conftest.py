"""Shared fixtures: a deterministic, offline market data provider."""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stock_market_analyst_mcp.errors import DataUnavailableError

PERIOD_SESSIONS = {
    "1mo": 21,
    "3mo": 63,
    "6mo": 126,
    "ytd": 200,
    "1y": 252,
    "2y": 504,
    "5y": 1260,
    "10y": 2520,
    "max": 2520,
}


def make_history(
    seed: int, sessions: int, drift: float = 0.0004, vol: float = 0.015
) -> pd.DataFrame:
    """Synthetic but realistic-looking OHLCV bars (geometric random walk + a slow cycle)."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end="2026-10-30", periods=sessions, name="Date")
    cycle = 0.004 * np.sin(np.arange(sessions) / 40)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, sessions) + cycle / 10))
    spread = close * rng.uniform(0.002, 0.02, sessions)
    open_ = close * (1 + rng.normal(0, 0.004, sessions))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + spread / 2,
            "Low": np.minimum(open_, close) - spread / 2,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, sessions).astype(float),
        },
        index=index,
    )


class FakeProvider:
    """Offline provider: every ticker gets its own reproducible price path."""

    def __init__(self, missing: set[str] | None = None) -> None:
        self.missing = missing or {"NOPE"}
        self.calls: list[tuple[str, ...]] = []

    def get_history(self, ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
        self.calls.append(("history", ticker, period, interval))
        if ticker in self.missing:
            raise DataUnavailableError(f"No price data found for '{ticker}'.")
        return make_history(zlib.crc32(ticker.encode()), PERIOD_SESSIONS[period])

    def get_info(self, ticker: str) -> dict[str, Any]:
        self.calls.append(("info", ticker))
        return {"name": f"{ticker} Corp.", "currency": "USD", "exchange": "NMS"}

    def get_profile(self, ticker: str) -> dict[str, Any]:
        self.calls.append(("profile", ticker))
        if ticker in self.missing:
            raise DataUnavailableError(f"No company profile found for '{ticker}'.")
        return {
            "name": f"{ticker} Corporation",
            "business_summary": f"{ticker} builds rockets and satellites.",
            "sector": "Industrials",
            "current_price": 50.0,
            "target_mean": 60.0,
            "analyst_recommendation": "buy",
            "next_earnings": {"date": "2026-11-10", "eps_estimate": -0.02},
        }

    def get_news(self, ticker: str, limit: int = 10) -> list[dict[str, Any]]:
        self.calls.append(("news", ticker, limit))
        items = [
            {
                "title": f"{ticker} headline {i}",
                "summary": "Something happened.",
                "publisher": "Newswire",
                "published_at": f"2026-11-0{i + 1}T12:00:00Z",
                "url": f"https://example.com/{ticker}/{i}",
            }
            for i in range(3)
        ]
        return items[:limit]

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        self.calls.append(("search", query, limit))
        catalog = [
            {
                "symbol": "RKLB",
                "name": "Rocket Lab Corporation",
                "type": "EQUITY",
                "exchange": "NASDAQ",
            },
            {"symbol": "AAPL", "name": "Apple Inc.", "type": "EQUITY", "exchange": "NASDAQ"},
        ]
        q = query.lower()
        return [c for c in catalog if q in c["name"].lower() or q in c["symbol"].lower()][:limit]


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
