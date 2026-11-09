import numpy as np
import pandas as pd
import pytest

from stock_market_analyst_mcp import indicators as ind

# Classic RSI worked example (J. Welles Wilder / StockCharts "RSI" ChartSchool article).
WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
]  # fmt: skip
# Published values; the article rounds intermediate averages, hence a ~0.1 tolerance.
WILDER_RSI = [70.53, 66.32, 66.55, 69.41, 66.36, 57.97]


def series(values):
    return pd.Series(values, index=pd.bdate_range("2026-01-01", periods=len(values)), dtype=float)


def test_sma_known_values():
    result = ind.sma(series(range(1, 11)), 3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == pytest.approx([2, 3, 4, 5, 6, 7, 8, 9])


def test_ema_seeded_with_sma_and_recursive():
    result = ind.ema(series([1, 2, 3, 4, 5, 6]), 3)  # alpha = 0.5
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == pytest.approx([2.0, 3.0, 4.0, 5.0])
    result = ind.ema(series([2, 4, 6, 10]), 3)
    assert result.iloc[3] == pytest.approx(0.5 * 10 + 0.5 * 4)


def test_rsi_matches_wilder_reference():
    result = ind.rsi(series(WILDER_CLOSES), 14)
    assert result.iloc[:14].isna().all()
    assert result.iloc[14:].tolist() == pytest.approx(WILDER_RSI, abs=0.1)


def test_rsi_first_value_exact():
    closes = series(WILDER_CLOSES)
    deltas = closes.diff().iloc[1:15]
    rs = deltas.clip(lower=0).mean() / (-deltas.clip(upper=0)).mean()
    assert ind.rsi(closes).iloc[14] == pytest.approx(100 - 100 / (1 + rs))


def test_rsi_edge_cases():
    assert ind.rsi(series(range(1, 30))).iloc[-1] == 100.0  # only gains
    assert ind.rsi(series(range(30, 1, -1))).iloc[-1] == pytest.approx(0.0)  # only losses
    assert ind.rsi(series([5.0] * 30)).iloc[-1] == 50.0  # flat
    assert ind.rsi(series([1, 2, 3])).isna().all()  # too short


def test_macd_is_difference_of_emas():
    closes = series(np.linspace(10, 40, 60) + np.sin(np.arange(60)))
    m = ind.macd(closes)
    expected = ind.ema(closes, 12) - ind.ema(closes, 26)
    pd.testing.assert_series_equal(m["macd"], expected, check_names=False)
    assert m["macd"].iloc[:25].isna().all() and not np.isnan(m["macd"].iloc[25])
    first_signal = 25 + 8
    seed = m["macd"].iloc[25 : first_signal + 1].mean()
    assert m["signal"].iloc[first_signal] == pytest.approx(seed)
    assert (m["histogram"] - (m["macd"] - m["signal"])).abs().max() < 1e-12


def test_bollinger_known_values():
    closes = series([1, 2, 3, 4, 5])
    bands = ind.bollinger_bands(closes, window=5, num_std=2)
    std = np.std([1, 2, 3, 4, 5])  # population std = sqrt(2)
    last = bands.iloc[-1]
    assert last["middle"] == pytest.approx(3)
    assert last["upper"] == pytest.approx(3 + 2 * std)
    assert last["lower"] == pytest.approx(3 - 2 * std)
    assert last["percent_b"] == pytest.approx((5 - (3 - 2 * std)) / (4 * std))


def test_bollinger_flat_series_has_no_percent_b():
    bands = ind.bollinger_bands(series([10.0] * 25))
    assert bands["upper"].iloc[-1] == bands["lower"].iloc[-1] == 10
    assert np.isnan(bands["percent_b"].iloc[-1])


def test_max_drawdown_and_volatility():
    assert ind.max_drawdown(series([100, 120, 90, 110, 60, 130])) == pytest.approx(-0.5)
    assert ind.max_drawdown(series([1, 2, 3])) == 0
    assert ind.annualized_volatility(series([100, 100, 100])) == 0
    returns = np.array([0.01, -0.02, 0.015, 0.0])
    closes = series(100 * np.cumprod(np.r_[1, 1 + returns]))
    assert ind.annualized_volatility(closes) == pytest.approx(returns.std(ddof=1) * np.sqrt(252))
