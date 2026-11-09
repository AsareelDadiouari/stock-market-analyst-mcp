"""Technical indicators implemented with pandas.

Every function takes a price Series and returns values aligned on the same index, with NaN
during the warm-up period. Conventions follow the original definitions (Wilder's RSI,
population standard deviation for Bollinger Bands) so results match common charting tools.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window=window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (alpha = 2 / (span + 1)), seeded with the SMA of the first
    ``span`` values, as in most charting packages."""
    values = close.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) >= span:
        alpha = 2.0 / (span + 1)
        out[span - 1] = values[:span].mean()
        for i in range(span, len(values)):
            out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=close.index, name=f"ema_{span}")


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing.

    The first average gain/loss is the simple mean of the first ``window`` changes; each
    following value is ``(previous * (window - 1) + current) / window``.
    """
    delta = close.diff().to_numpy(dtype=float)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    out = np.full(len(close), np.nan)
    if len(close) <= window:
        return pd.Series(out, index=close.index, name="rsi")

    avg_gain = gains[1 : window + 1].mean()
    avg_loss = losses[1 : window + 1].mean()
    for i in range(window, len(close)):
        if i > window:
            avg_gain = (avg_gain * (window - 1) + gains[i]) / window
            avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        if avg_loss == 0:
            out[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return pd.Series(out, index=close.index, name="rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line (EMA fast - EMA slow), signal line (EMA of the MACD line) and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line.dropna(), signal).reindex(close.index)
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}
    )


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: SMA ± ``num_std`` population standard deviations, plus %B and bandwidth."""
    middle = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = upper - lower
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "percent_b": (close - lower) / width.replace(0, np.nan),
            "bandwidth": width / middle,
        }
    )


def annualized_volatility(close: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Standard deviation of simple returns, annualised."""
    returns = close.pct_change().dropna()
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(close: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative fraction (e.g. -0.25 for -25 %)."""
    if close.empty:
        return float("nan")
    return float((close / close.cummax() - 1.0).min())
