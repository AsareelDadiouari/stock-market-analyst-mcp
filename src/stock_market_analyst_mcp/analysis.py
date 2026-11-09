"""Higher-level analytics built on the indicators: quotes, stock summaries, comparisons."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import indicators as ind
from .errors import DataUnavailableError

DISCLAIMER = (
    "Educational analysis of historical data only — not financial advice. "
    "Past performance does not predict future results."
)


def to_jsonable(value: Any, digits: int = 4) -> Any:
    """Recursively convert numpy/pandas values to JSON-friendly Python types (NaN -> None)."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v, digits) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else round(f, digits)
    return value


def pct_change_over(close: pd.Series, sessions: int) -> float:
    """Simple return over the last ``sessions`` bars (NaN if the series is too short)."""
    if len(close) <= sessions:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-1 - sessions] - 1.0)


def _year_extreme(history: pd.DataFrame, column: str) -> float:
    """52-week high (``High``) or low (``Low``), falling back to closes."""
    year = history.iloc[-ind.TRADING_DAYS :]
    series = year[column] if column in year else year["Close"]
    return float(series.max() if column == "High" else series.min())


def quote_summary(history: pd.DataFrame, info: dict[str, Any]) -> dict[str, Any]:
    close = history["Close"]
    last = history.iloc[-1]
    previous_close = close.iloc[-2] if len(close) > 1 else float("nan")
    return {
        **info,
        "as_of": history.index[-1],
        "price": last["Close"],
        "previous_close": previous_close,
        "change": last["Close"] - previous_close,
        "change_pct": last["Close"] / previous_close - 1.0,
        "day_open": last.get("Open"),
        "day_high": last.get("High"),
        "day_low": last.get("Low"),
        "volume": last.get("Volume"),
        "avg_volume_20d": history["Volume"].iloc[-20:].mean() if "Volume" in history else None,
        "fifty_two_week_high": _year_extreme(history, "High"),
        "fifty_two_week_low": _year_extreme(history, "Low"),
    }


def _trend_label(price: float, sma50: float, sma200: float) -> str:
    if any(math.isnan(x) for x in (sma50, sma200)):
        if math.isnan(sma50):
            return "insufficient data"
        return "uptrend" if price > sma50 else "downtrend"
    if price > sma50 > sma200:
        return "strong uptrend"
    if price < sma50 < sma200:
        return "strong downtrend"
    if price > sma200:
        return "uptrend (consolidating)"
    return "downtrend (recovering)" if price > sma50 else "downtrend"


def _rsi_label(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if value >= 70:
        return "overbought"
    if value <= 30:
        return "oversold"
    return "bullish" if value >= 50 else "bearish"


def _last_cross(fast: pd.Series, slow: pd.Series) -> dict[str, Any] | None:
    """Most recent crossing of ``fast`` over/under ``slow``."""
    diff = (fast - slow).dropna()
    if len(diff) < 2:
        return None
    sign = np.sign(diff)
    changes = sign[sign != sign.shift()].iloc[1:]
    changes = changes[changes != 0]
    if changes.empty:
        return None
    date = changes.index[-1]
    return {
        "type": "bullish" if changes.iloc[-1] > 0 else "bearish",
        "date": date,
        "sessions_ago": int(len(diff) - 1 - diff.index.get_loc(date)),
    }


def analyze(history: pd.DataFrame, info: dict[str, Any]) -> dict[str, Any]:
    """One structured snapshot: trend, momentum, volatility, key levels and plain signals."""
    close = history["Close"]
    if len(close) < 30:
        raise DataUnavailableError(
            "At least 30 sessions of history are needed for an analysis; use a longer period."
        )
    price = float(close.iloc[-1])
    sma20, sma50, sma200 = (ind.sma(close, w) for w in (20, 50, 200))
    last = {k: float(s.iloc[-1]) for k, s in {"20": sma20, "50": sma50, "200": sma200}.items()}
    rsi14 = float(ind.rsi(close, 14).iloc[-1])
    macd_df = ind.macd(close)
    bands = ind.bollinger_bands(close)
    year = history.iloc[-ind.TRADING_DAYS :]
    recent = history.iloc[-20:]

    trend = _trend_label(price, last["50"], last["200"])
    momentum = {
        "rsi_14": rsi14,
        "rsi_state": _rsi_label(rsi14),
        "macd": macd_df["macd"].iloc[-1],
        "macd_signal": macd_df["signal"].iloc[-1],
        "macd_histogram": macd_df["histogram"].iloc[-1],
        "last_macd_cross": _last_cross(macd_df["macd"], macd_df["signal"]),
    }
    volatility = {
        "annualized_volatility_1y": ind.annualized_volatility(year["Close"]),
        "annualized_volatility_20d": ind.annualized_volatility(close.iloc[-21:]),
        "max_drawdown_1y": ind.max_drawdown(year["Close"]),
        "bollinger_percent_b": bands["percent_b"].iloc[-1],
        "bollinger_bandwidth": bands["bandwidth"].iloc[-1],
    }
    levels = {
        "support_20d": recent["Low"].min() if "Low" in recent else recent["Close"].min(),
        "resistance_20d": recent["High"].max() if "High" in recent else recent["Close"].max(),
        "fifty_two_week_low": _year_extreme(history, "Low"),
        "fifty_two_week_high": _year_extreme(history, "High"),
        "sma_50": last["50"],
        "sma_200": last["200"],
        "bollinger_lower": bands["lower"].iloc[-1],
        "bollinger_upper": bands["upper"].iloc[-1],
    }

    signals: list[str] = []
    if not math.isnan(last["200"]):
        side = "above" if price > last["200"] else "below"
        signals.append(f"Price is {abs(price / last['200'] - 1):.1%} {side} its 200-day average.")
    cross = _last_cross(sma50, sma200)
    if cross and cross["sessions_ago"] <= 60:
        name = "Golden cross" if cross["type"] == "bullish" else "Death cross"
        signals.append(f"{name} (SMA50 vs SMA200) {cross['sessions_ago']} sessions ago.")
    if momentum["rsi_state"] in ("overbought", "oversold"):
        signals.append(f"RSI(14) at {rsi14:.0f}: {momentum['rsi_state']}.")
    mc = momentum["last_macd_cross"]
    if mc and mc["sessions_ago"] <= 10:
        signals.append(f"MACD {mc['type']} crossover {mc['sessions_ago']} sessions ago.")
    pb = volatility["bollinger_percent_b"]
    if not math.isnan(pb) and (pb > 1 or pb < 0):
        signals.append(
            f"Price is outside the Bollinger Bands ({'above upper' if pb > 1 else 'below lower'})."
        )
    high = levels["fifty_two_week_high"]
    if price >= 0.98 * high:
        signals.append("Trading within 2% of its 52-week high.")

    return {
        "ticker_info": info,
        "as_of": history.index[-1],
        "price": price,
        "returns": {
            "1w": pct_change_over(close, 5),
            "1m": pct_change_over(close, 21),
            "3m": pct_change_over(close, 63),
            "6m": pct_change_over(close, 126),
            "1y": pct_change_over(close, 252),
        },
        "trend": {
            "label": trend,
            "sma_20": last["20"],
            "sma_50": last["50"],
            "sma_200": last["200"],
            "last_sma50_sma200_cross": cross,
        },
        "momentum": momentum,
        "volatility": volatility,
        "key_levels": levels,
        "signals": signals,
        "disclaimer": DISCLAIMER,
    }


def performance_metrics(close: pd.Series) -> dict[str, float]:
    """Return / risk statistics for a price (or equity) curve of daily bars."""
    returns = close.pct_change().dropna()
    years = max(len(returns) / ind.TRADING_DAYS, 1e-9)
    total = float(close.iloc[-1] / close.iloc[0] - 1.0)
    std = returns.std(ddof=1)
    return {
        "total_return": total,
        "cagr": (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0,
        "annualized_volatility": ind.annualized_volatility(close),
        "sharpe_ratio": float(returns.mean() / std * np.sqrt(ind.TRADING_DAYS)) if std else 0.0,
        "max_drawdown": ind.max_drawdown(close),
    }


def compare(histories: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Side-by-side performance and correlation of several tickers on their common dates."""
    closes = pd.DataFrame({t: h["Close"] for t, h in histories.items()}).dropna()
    if len(closes) < 10:
        raise DataUnavailableError(
            "The tickers share fewer than 10 common trading days; try a longer period."
        )
    metrics = {t: performance_metrics(closes[t]) for t in closes.columns}
    ranking = sorted(metrics, key=lambda t: metrics[t]["total_return"], reverse=True)
    corr = closes.pct_change().dropna().corr()
    return {
        "start": closes.index[0],
        "end": closes.index[-1],
        "sessions": len(closes),
        "metrics": metrics,
        "ranking_by_return": ranking,
        "best_risk_adjusted": max(metrics, key=lambda t: metrics[t]["sharpe_ratio"]),
        "correlation": {t: corr[t].to_dict() for t in corr.columns},
        "normalized_performance": (closes / closes.iloc[0] * 100).iloc[-1].to_dict(),
        "disclaimer": DISCLAIMER,
    }
