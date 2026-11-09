"""Vectorised long-only backtests compared against buy-and-hold.

Signals are computed on a bar's close and applied from the next bar (``position.shift(1)``),
so a strategy never trades on information it could not have had.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from . import indicators as ind
from .analysis import DISCLAIMER, performance_metrics
from .errors import DataUnavailableError, InvalidInputError

Strategy = Literal["sma_crossover", "rsi_reversion"]


def sma_crossover_positions(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Long (1) while the fast SMA is above the slow SMA, flat (0) otherwise."""
    if fast >= slow:
        raise InvalidInputError("'fast' window must be shorter than the 'slow' window.")
    return (ind.sma(close, fast) > ind.sma(close, slow)).astype(float)


def rsi_reversion_positions(
    close: pd.Series, window: int, oversold: float, overbought: float
) -> pd.Series:
    """Enter long when RSI drops below ``oversold``; exit when it rises above ``overbought``."""
    if not 0 < oversold < overbought < 100:
        raise InvalidInputError("Require 0 < oversold < overbought < 100.")
    rsi = ind.rsi(close, window)
    position, out = 0.0, []
    for value in rsi:
        if position == 0 and value < oversold:
            position = 1.0
        elif position == 1 and value > overbought:
            position = 0.0
        out.append(position)
    return pd.Series(out, index=close.index)


def _trades(close: pd.Series, held: pd.Series) -> list[dict[str, Any]]:
    """Round trips from the effective (already shifted) position series."""
    trades, entry = [], None
    for i in range(1, len(held)):
        if held.iloc[i] == 1 and held.iloc[i - 1] == 0:
            entry = i - 1  # bought at the previous close
        elif held.iloc[i] == 0 and held.iloc[i - 1] == 1 and entry is not None:
            trades.append((entry, i - 1))
            entry = None
    if entry is not None:
        trades.append((entry, len(held) - 1))  # still open: mark to market
    return [
        {
            "entry_date": close.index[a],
            "entry_price": close.iloc[a],
            "exit_date": close.index[b],
            "exit_price": close.iloc[b],
            "return": close.iloc[b] / close.iloc[a] - 1.0,
            "open": bool(b == len(held) - 1 and held.iloc[-1] == 1),
        }
        for a, b in trades
    ]


def run_backtest(
    history: pd.DataFrame,
    strategy: Strategy = "sma_crossover",
    *,
    fast: int = 50,
    slow: int = 200,
    rsi_window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    initial_capital: float = 10_000.0,
    fee_bps: float = 5.0,
) -> dict[str, Any]:
    close = history["Close"].astype(float)
    if strategy == "sma_crossover":
        signal = sma_crossover_positions(close, fast, slow)
        warmup = slow
        params: dict[str, Any] = {"fast": fast, "slow": slow}
    elif strategy == "rsi_reversion":
        signal = rsi_reversion_positions(close, rsi_window, oversold, overbought)
        warmup = rsi_window + 1
        params = {"rsi_window": rsi_window, "oversold": oversold, "overbought": overbought}
    else:
        raise InvalidInputError(
            f"Unknown strategy '{strategy}'. Use 'sma_crossover' or 'rsi_reversion'."
        )
    if len(close) < warmup + 20:
        raise DataUnavailableError(
            f"Need at least {warmup + 20} sessions for this strategy (got {len(close)}); "
            "use a longer period or shorter windows."
        )
    if fee_bps < 0 or initial_capital <= 0:
        raise InvalidInputError("fee_bps must be >= 0 and initial_capital > 0.")

    # Evaluate both curves from the first bar where the strategy can have an opinion.
    close = close.iloc[warmup - 1 :]
    held = signal.iloc[warmup - 1 :].shift(1).fillna(0.0)
    returns = close.pct_change().fillna(0.0)
    turnover = held.diff().abs().fillna(held.iloc[0])
    strat_returns = held * returns - turnover * fee_bps / 10_000
    equity = initial_capital * (1 + strat_returns).cumprod()
    buy_hold = initial_capital * close / close.iloc[0]

    trades = _trades(close, held)
    closed = [t for t in trades if not t["open"]]
    strat = performance_metrics(equity)
    bench = performance_metrics(buy_hold)
    return {
        "strategy": strategy,
        "parameters": params,
        "start": close.index[0],
        "end": close.index[-1],
        "sessions": len(close),
        "fee_bps": fee_bps,
        "strategy_result": {
            **strat,
            "final_equity": equity.iloc[-1],
            "number_of_trades": len(trades),
            "win_rate": (sum(t["return"] > 0 for t in closed) / len(closed)) if closed else None,
            "exposure": float(held.mean()),
        },
        "buy_and_hold_result": {**bench, "final_equity": buy_hold.iloc[-1]},
        "outperformance": strat["total_return"] - bench["total_return"],
        "recent_trades": trades[-10:],
        "disclaimer": DISCLAIMER,
    }
