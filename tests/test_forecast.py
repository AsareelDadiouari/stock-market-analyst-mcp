import numpy as np
import pandas as pd
import pytest

from stock_market_analyst_mcp.errors import DataUnavailableError, InvalidInputError
from stock_market_analyst_mcp.forecast import historical_base_rate, monte_carlo_forecast

from .conftest import make_history


def series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_constant_growth_is_deterministic():
    # Every daily log return is identical, so every simulated path is the same.
    closes = series(100 * np.exp(0.01 * np.arange(100)))
    for method in ("bootstrap", "gbm"):
        out = monte_carlo_forecast(closes, 10, simulations=1000, method=method, seed=0)
        expected = closes.iloc[-1] * np.exp(0.1)
        assert out["price_percentiles"]["p5"] == pytest.approx(expected)
        assert out["price_percentiles"]["p95"] == pytest.approx(expected)
        assert out["probability_up"] == 1.0
        assert out["probability_of_10pct_drawdown_along_the_way"] == 0.0


def test_target_probabilities():
    closes = series(100 * np.exp(0.01 * np.arange(100)))
    price = closes.iloc[-1]
    up = monte_carlo_forecast(closes, 10, simulations=1000, target_price=price * 1.05, seed=0)
    assert up["target"]["probability_above_at_horizon"] == 1.0
    assert up["target"]["probability_touched_before_horizon"] == 1.0
    far = monte_carlo_forecast(closes, 10, simulations=1000, target_price=price * 2, seed=0)
    assert far["target"]["probability_above_at_horizon"] == 0.0


def test_random_walk_is_roughly_symmetric_and_reproducible():
    closes = make_history(42, 504, drift=0.0, vol=0.02)["Close"]
    a = monte_carlo_forecast(closes, 21, simulations=20_000, method="gbm", seed=7)
    b = monte_carlo_forecast(closes, 21, simulations=20_000, method="gbm", seed=7)
    assert a == b
    assert 0.3 < a["probability_up"] < 0.7
    pct = a["price_percentiles"]
    assert pct["p5"] < a["current_price"] < pct["p95"]
    assert a["probability_down_10pct_or_more"] <= a["probability_of_10pct_drawdown_along_the_way"]


def test_historical_base_rate():
    rate = historical_base_rate(series([1, 2, 3, 2, 1]), 1)
    assert rate["windows"] == 4
    assert rate["share_positive"] == 0.5
    assert historical_base_rate(series([1, 2]), 5)["windows"] == 0


def test_validation():
    with pytest.raises(DataUnavailableError):
        monte_carlo_forecast(series(range(1, 20)), 5)
    closes = make_history(1, 100)["Close"]
    with pytest.raises(InvalidInputError):
        monte_carlo_forecast(closes, 5, target_price=0)
    with pytest.raises(InvalidInputError):
        monte_carlo_forecast(closes, 5, method="crystal_ball")
