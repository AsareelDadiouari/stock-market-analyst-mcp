"""Probabilistic price ranges by Monte Carlo simulation.

This module does not "predict" a price. It answers a narrower, honest question:
*if the next N sessions behave statistically like the last few years, what range of prices
is plausible, and how often does the price end up above a given level?*

Two simulation methods are available:

- ``bootstrap`` (default): resample the stock's own historical daily log returns. Keeps fat
  tails and skew, assumes no autocorrelation.
- ``gbm``: geometric Brownian motion with the historical mean and volatility of log returns
  (the classic Black-Scholes assumption of normally distributed returns).

Both assume the future looks like the sampled past. They know nothing about earnings, news,
macro events or regime changes, so the probabilities are descriptive, not reliable odds.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from . import indicators as ind
from .errors import DataUnavailableError, InvalidInputError

Method = Literal["bootstrap", "gbm"]

FORECAST_DISCLAIMER = (
    "Statistical simulation, not a prediction and not financial advice. It assumes future "
    "returns resemble past returns and ignores news, earnings and market regime changes. "
    "Real outcomes can fall outside every range shown. Do not make investment decisions "
    "based on this output."
)

MIN_HISTORY = 60
PERCENTILES = (5, 25, 50, 75, 95)


def _simulate_log_paths(
    log_returns: np.ndarray, horizon: int, simulations: int, method: Method, seed: int | None
) -> np.ndarray:
    """Cumulative log-return paths, shape (simulations, horizon)."""
    rng = np.random.default_rng(seed)
    if method == "bootstrap":
        draws = rng.choice(log_returns, size=(simulations, horizon), replace=True)
    elif method == "gbm":
        draws = rng.normal(log_returns.mean(), log_returns.std(ddof=1), (simulations, horizon))
    else:
        raise InvalidInputError(f"Unknown method '{method}'. Use 'bootstrap' or 'gbm'.")
    return np.cumsum(draws, axis=1)


def historical_base_rate(close: pd.Series, horizon: int) -> dict[str, Any]:
    """How often, in the sampled history, the price was higher ``horizon`` sessions later."""
    forward = (close.shift(-horizon) / close - 1.0).dropna()
    if forward.empty:
        return {"windows": 0, "share_positive": None, "median_return": None}
    return {
        "windows": len(forward),
        "share_positive": float((forward > 0).mean()),
        "median_return": float(forward.median()),
    }


def monte_carlo_forecast(
    close: pd.Series,
    horizon: int = 21,
    *,
    simulations: int = 10_000,
    method: Method = "bootstrap",
    target_price: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate ``simulations`` price paths ``horizon`` sessions ahead and summarise them."""
    close = close.dropna().astype(float)
    if len(close) < MIN_HISTORY:
        raise DataUnavailableError(
            f"At least {MIN_HISTORY} sessions of history are needed to simulate (got "
            f"{len(close)}); use a longer lookback."
        )
    if target_price is not None and target_price <= 0:
        raise InvalidInputError("target_price must be a positive number.")

    price = float(close.iloc[-1])
    log_returns = np.diff(np.log(close.to_numpy()))
    paths = _simulate_log_paths(log_returns, horizon, simulations, method, seed)
    final = price * np.exp(paths[:, -1])
    lowest = price * np.exp(np.minimum(paths.min(axis=1), 0.0))
    highest = price * np.exp(np.maximum(paths.max(axis=1), 0.0))

    result: dict[str, Any] = {
        "current_price": price,
        "horizon_sessions": horizon,
        "method": method,
        "simulations": simulations,
        "inputs": {
            "history_start": close.index[0],
            "history_end": close.index[-1],
            "sessions_used": len(log_returns),
            "annualized_drift": float(log_returns.mean() * ind.TRADING_DAYS),
            "annualized_volatility": float(log_returns.std(ddof=1) * np.sqrt(ind.TRADING_DAYS)),
        },
        "price_percentiles": {
            f"p{p}": float(v)
            for p, v in zip(PERCENTILES, np.percentile(final, PERCENTILES), strict=True)
        },
        "expected_price": float(final.mean()),
        "probability_up": float((final > price).mean()),
        "probability_down_10pct_or_more": float((final <= 0.9 * price).mean()),
        "probability_up_10pct_or_more": float((final >= 1.1 * price).mean()),
        "probability_of_10pct_drawdown_along_the_way": float((lowest <= 0.9 * price).mean()),
        "historical_base_rate": historical_base_rate(close, horizon),
    }
    if target_price is not None:
        touched = highest >= target_price if target_price >= price else lowest <= target_price
        result["target"] = {
            "price": target_price,
            "probability_above_at_horizon": float((final >= target_price).mean()),
            "probability_touched_before_horizon": float(touched.mean()),
        }
    result["how_to_read"] = (
        "90% of simulated paths ended between p5 and p95. probability_up is the share of "
        "simulated paths ending above today's price; historical_base_rate is how often this "
        "stock actually rose over the same horizon in the sampled history."
    )
    result["disclaimer"] = FORECAST_DISCLAIMER
    return result
