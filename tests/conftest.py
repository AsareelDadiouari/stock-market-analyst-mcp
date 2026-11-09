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


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
