import numpy as np
import pandas as pd
import pytest

from stock_market_analyst_mcp.analysis import analyze, compare, to_jsonable
from stock_market_analyst_mcp.backtest import run_backtest
from stock_market_analyst_mcp.errors import DataUnavailableError, InvalidInputError
from stock_market_analyst_mcp.providers import CachedProvider, clean_history

from .conftest import FakeProvider, make_history


def frame(closes):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1.0},
        index=pd.bdate_range("2020-01-01", periods=len(closes)),
    )


def test_analyze_detects_uptrend():
    result = analyze(frame(np.linspace(50, 150, 300)), {})
    assert result["trend"]["label"] == "strong uptrend"
    assert result["momentum"]["rsi_state"] == "overbought"
    assert result["returns"]["1y"] > 0
    assert any("200-day" in s for s in result["signals"])


def test_analyze_detects_downtrend():
    assert analyze(frame(np.linspace(150, 50, 300)), {})["trend"]["label"] == "strong downtrend"


def test_analyze_requires_history():
    with pytest.raises(DataUnavailableError):
        analyze(frame(range(1, 10)), {})


def test_compare_aligns_dates_and_ranks():
    a = frame(np.linspace(100, 200, 100))
    b = frame(np.linspace(100, 120, 100)).iloc[10:]  # starts later
    result = compare({"A": a, "B": b})
    assert result["sessions"] == 90
    assert result["ranking_by_return"] == ["A", "B"]
    assert result["correlation"]["A"]["A"] == pytest.approx(1)


def test_backtest_buy_and_hold_math():
    closes = np.r_[np.full(30, 100.0), np.linspace(100, 200, 100)]
    result = run_backtest(frame(closes), "sma_crossover", fast=5, slow=10, fee_bps=0)
    bh = result["buy_and_hold_result"]
    assert bh["total_return"] == pytest.approx(1.0)
    assert bh["final_equity"] == pytest.approx(20_000)
    # A single entry once the rally starts, held to the end (no look-ahead: misses day one).
    strat = result["strategy_result"]
    assert strat["number_of_trades"] == 1
    assert 0.8 < strat["total_return"] < 1.0
    assert result["recent_trades"][0]["open"] is True


def test_backtest_no_lookahead_on_single_spike():
    # The price jumps once; a strategy must not capture the jump on the bar that caused the signal.
    closes = np.r_[np.full(40, 100.0), [200.0], np.full(40, 200.0)]
    result = run_backtest(frame(closes), "sma_crossover", fast=2, slow=5, fee_bps=0)
    assert result["strategy_result"]["total_return"] == pytest.approx(0.0)


def test_backtest_fees_reduce_returns():
    history = make_history(7, 800)
    free = run_backtest(history, "rsi_reversion", fee_bps=0)
    costly = run_backtest(history, "rsi_reversion", fee_bps=50)
    assert free["strategy_result"]["number_of_trades"] > 0
    assert costly["strategy_result"]["total_return"] < free["strategy_result"]["total_return"]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"strategy": "sma_crossover", "fast": 50, "slow": 20}, InvalidInputError),
        ({"strategy": "rsi_reversion", "oversold": 80, "overbought": 20}, InvalidInputError),
        ({"strategy": "magic"}, InvalidInputError),
        ({"strategy": "sma_crossover", "slow": 400}, DataUnavailableError),
    ],
)
def test_backtest_validation(kwargs, error):
    with pytest.raises(error):
        run_backtest(make_history(1, 300), **kwargs)


def test_to_jsonable_handles_nan_and_numpy():
    out = to_jsonable({"a": np.float64("nan"), "b": np.int64(3), "c": pd.Timestamp("2026-01-02")})
    assert out == {"a": None, "b": 3, "c": "2026-01-02"}


def test_cached_provider_hits_and_isolation():
    inner = FakeProvider()
    cached = CachedProvider(inner, ttl_seconds=60)
    first = cached.get_history("AAPL", "1y")
    first["Close"] = 0  # mutating a result must not corrupt the cache
    second = cached.get_history("AAPL", "1y")
    assert (second["Close"] > 0).all()
    assert len([c for c in inner.calls if c[0] == "history"]) == 1
    assert cached.hits == 1


def test_cached_provider_expires():
    inner = FakeProvider()
    cached = CachedProvider(inner, ttl_seconds=0)
    cached.get_info("AAPL")
    cached.get_info("AAPL")
    assert len(inner.calls) == 2


def test_clean_history_normalises_index():
    raw = make_history(3, 5)
    raw.index = raw.index.tz_localize("America/New_York") + pd.Timedelta(hours=9)
    cleaned = clean_history(raw, "X")
    assert cleaned.index.tz is None
    assert (cleaned.index == cleaned.index.normalize()).all()
    with pytest.raises(DataUnavailableError):
        clean_history(pd.DataFrame(), "X")
