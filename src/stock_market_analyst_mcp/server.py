"""MCP server exposing the stock analysis tools over stdio.

Run with ``uv run stock-market-analyst-mcp`` (or ``uvx --from <repo> stock-market-analyst-mcp``)
and point any MCP client (Claude Desktop, Claude Code, MCP Inspector) at it.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Annotated, Any, Literal, ParamSpec, TypeVar

import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import __version__
from . import indicators as ind
from .analysis import DISCLAIMER, analyze, compare, quote_summary, to_jsonable
from .backtest import run_backtest
from .errors import AnalystError
from .forecast import monte_carlo_forecast
from .providers import MarketDataProvider, default_provider
from .validation import (
    normalize_ticker,
    normalize_tickers,
    validate_int,
    validate_interval,
    validate_period,
)

P = ParamSpec("P")
R = TypeVar("R")

INSTRUCTIONS = """\
Stock Market Analyst: tools to fetch market data (Yahoo Finance) and run technical analysis.
Start with `analyze_stock` for a one-shot overview of a ticker, `compare_tickers` to rank
several symbols, `backtest_strategy` to test a rule on history, and `forecast_price_range` for
a probabilistic range of future prices. Tickers use Yahoo Finance symbols (AAPL, BRK-B,
SHOP.TO, ^GSPC, BTC-USD). Always present results as educational analysis, not financial
advice. Forecast probabilities come from simulations that assume the future resembles the
past: never present them as predictions or reliable odds, and always relay their disclaimer."""

Ticker = Annotated[str, Field(description="Yahoo Finance symbol, e.g. 'AAPL', 'SHOP.TO', '^GSPC'")]
Period = Annotated[
    Literal["1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y", "10y", "max"],
    Field(description="How far back to look"),
]
IndicatorName = Literal["sma", "ema", "rsi", "macd", "bollinger"]

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True, idempotentHint=True)


def _user_errors(func: Callable[P, R]) -> Callable[P, R]:
    """Turn anticipated domain errors into MCP tool errors the model can read and act on."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except AnalystError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


def create_server(provider: MarketDataProvider | None = None) -> MCPServer:
    """Build the server. Inject a provider to use another data source (or a fake in tests)."""
    data = provider or default_provider()
    mcp = MCPServer(
        name="stock-market-analyst",
        title="Stock Market Analyst",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    @mcp.tool(annotations=READ_ONLY)
    @_user_errors
    def get_quote(ticker: Ticker) -> dict[str, Any]:
        """Latest price, daily change, volume, 52-week range and company details for a ticker."""
        symbol = normalize_ticker(ticker)
        history = data.get_history(symbol, "1y", "1d")
        return to_jsonable({"ticker": symbol, **quote_summary(history, data.get_info(symbol))})

    @mcp.tool(annotations=READ_ONLY)
    @_user_errors
    def get_price_history(
        ticker: Ticker,
        period: Period = "6mo",
        interval: Annotated[Literal["1d", "1wk", "1mo"], Field(description="Bar size")] = "1d",
        limit: Annotated[int, Field(description="Most recent bars to return (1-500)")] = 60,
    ) -> dict[str, Any]:
        """OHLCV price bars (split/dividend adjusted) plus a summary of the whole period."""
        symbol = normalize_ticker(ticker)
        validate_period(period)
        validate_interval(interval)
        validate_int("limit", limit, minimum=1, maximum=500)
        history = data.get_history(symbol, period, interval)
        close = history["Close"]
        bars = history.tail(limit).reset_index()
        return to_jsonable(
            {
                "ticker": symbol,
                "period": period,
                "interval": interval,
                "summary": {
                    "start": history.index[0],
                    "end": history.index[-1],
                    "bars": len(history),
                    "first_close": close.iloc[0],
                    "last_close": close.iloc[-1],
                    "period_return": close.iloc[-1] / close.iloc[0] - 1,
                    "high": history["High"].max() if "High" in history else close.max(),
                    "low": history["Low"].min() if "Low" in history else close.min(),
                },
                "bars": bars.to_dict(orient="records"),
            }
        )

    @mcp.tool(annotations=READ_ONLY)
    @_user_errors
    def technical_indicators(
        ticker: Ticker,
        indicators: Annotated[
            list[IndicatorName] | None,
            Field(description="Indicators to compute (default: all five)"),
        ] = None,
        period: Period = "1y",
        sma_window: Annotated[int, Field(description="SMA window in sessions")] = 50,
        ema_span: Annotated[int, Field(description="EMA span in sessions")] = 20,
        rsi_window: Annotated[int, Field(description="RSI window in sessions")] = 14,
        bollinger_window: Annotated[int, Field(description="Bollinger window")] = 20,
        last_n: Annotated[int, Field(description="Recent rows of the series to return")] = 10,
    ) -> dict[str, Any]:
        """SMA, EMA, RSI (Wilder), MACD (12/26/9) and Bollinger Bands (20, 2σ) on daily closes.

        Returns the latest value of each indicator with a short interpretation, plus the
        last `last_n` rows of the series."""
        if indicators is None:
            indicators = ["sma", "ema", "rsi", "macd", "bollinger"]
        symbol = normalize_ticker(ticker)
        validate_period(period)
        for name, value in (
            ("sma_window", sma_window),
            ("ema_span", ema_span),
            ("rsi_window", rsi_window),
            ("bollinger_window", bollinger_window),
        ):
            validate_int(name, value, minimum=2, maximum=300)
        validate_int("last_n", last_n, minimum=1, maximum=250)
        if not indicators:
            raise AnalystError("Ask for at least one indicator.")

        history = data.get_history(symbol, period, "1d")
        close = history["Close"]
        columns: dict[str, Any] = {"close": close}
        latest: dict[str, Any] = {}
        price = float(close.iloc[-1])

        if "sma" in indicators:
            s = ind.sma(close, sma_window)
            columns[f"sma_{sma_window}"] = s
            latest[f"sma_{sma_window}"] = {
                "value": s.iloc[-1],
                "price_vs_sma": "above" if price > s.iloc[-1] else "below",
            }
        if "ema" in indicators:
            e = ind.ema(close, ema_span)
            columns[f"ema_{ema_span}"] = e
            latest[f"ema_{ema_span}"] = {
                "value": e.iloc[-1],
                "price_vs_ema": "above" if price > e.iloc[-1] else "below",
            }
        if "rsi" in indicators:
            r = ind.rsi(close, rsi_window)
            columns[f"rsi_{rsi_window}"] = r
            v = r.iloc[-1]
            state = "overbought" if v >= 70 else "oversold" if v <= 30 else "neutral"
            latest[f"rsi_{rsi_window}"] = {"value": v, "state": state}
        if "macd" in indicators:
            m = ind.macd(close)
            for col in m.columns:
                columns[f"macd_{col}" if col != "macd" else "macd"] = m[col]
            hist = m["histogram"]
            latest["macd"] = {
                **m.iloc[-1].to_dict(),
                "momentum": "bullish" if hist.iloc[-1] > 0 else "bearish",
                "histogram_rising": bool(hist.iloc[-1] > hist.iloc[-2]) if len(hist) > 1 else None,
            }
        if "bollinger" in indicators:
            b = ind.bollinger_bands(close, bollinger_window)
            for col in b.columns:
                columns[f"bb_{col}"] = b[col]
            latest["bollinger"] = b.iloc[-1].to_dict()

        frame = pd.DataFrame(columns).tail(last_n).reset_index()
        return to_jsonable(
            {
                "ticker": symbol,
                "as_of": history.index[-1],
                "price": price,
                "latest": latest,
                "series": frame.to_dict(orient="records"),
                "disclaimer": DISCLAIMER,
            }
        )

    @mcp.tool(annotations=READ_ONLY)
    @_user_errors
    def analyze_stock(ticker: Ticker) -> dict[str, Any]:
        """One-shot technical overview of a ticker from 2 years of daily data: returns over
        several horizons, trend (SMA 20/50/200, golden/death cross), momentum (RSI, MACD),
        volatility and drawdown, key support/resistance levels and plain-language signals."""
        symbol = normalize_ticker(ticker)
        history = data.get_history(symbol, "2y", "1d")
        return to_jsonable({"ticker": symbol, **analyze(history, data.get_info(symbol))})

    @mcp.tool(annotations=READ_ONLY)
    @_user_errors
    def compare_tickers(
        tickers: Annotated[list[str], Field(description="2 to 10 Yahoo Finance symbols")],
        period: Period = "1y",
    ) -> dict[str, Any]:
        """Compare 2-10 tickers over the same dates: total return, CAGR, volatility, Sharpe
        ratio, max drawdown, ranking and the correlation matrix of daily returns."""
        symbols = normalize_tickers(tickers, min_count=2, max_count=10)
        validate_period(period)
        histories = {s: data.get_history(s, period, "1d") for s in symbols}
        return to_jsonable({"tickers": symbols, "period": period, **compare(histories)})

    @mcp.tool(annotations=READ_ONLY)
    @_user_errors
    def backtest_strategy(
        ticker: Ticker,
        strategy: Annotated[
            Literal["sma_crossover", "rsi_reversion"],
            Field(
                description="sma_crossover: long while fast SMA > slow SMA. "
                "rsi_reversion: buy when RSI < oversold, sell when RSI > overbought."
            ),
        ] = "sma_crossover",
        period: Period = "5y",
        fast: Annotated[int, Field(description="Fast SMA window (sma_crossover)")] = 50,
        slow: Annotated[int, Field(description="Slow SMA window (sma_crossover)")] = 200,
        rsi_window: Annotated[int, Field(description="RSI window (rsi_reversion)")] = 14,
        oversold: Annotated[float, Field(description="RSI entry level (rsi_reversion)")] = 30,
        overbought: Annotated[float, Field(description="RSI exit level (rsi_reversion)")] = 70,
        fee_bps: Annotated[float, Field(description="Cost per trade in basis points")] = 5,
        initial_capital: Annotated[float, Field(description="Starting capital")] = 10_000,
    ) -> dict[str, Any]:
        """Backtest a long-only rule on daily closes and compare it with buy-and-hold:
        total return, CAGR, volatility, Sharpe, max drawdown, trades, win rate and exposure.
        Signals are applied on the next session (no look-ahead) and fees are deducted."""
        symbol = normalize_ticker(ticker)
        validate_period(period)
        validate_int("fast", fast, minimum=2, maximum=300)
        validate_int("slow", slow, minimum=3, maximum=400)
        validate_int("rsi_window", rsi_window, minimum=2, maximum=100)
        history = data.get_history(symbol, period, "1d")
        result = run_backtest(
            history,
            strategy,
            fast=fast,
            slow=slow,
            rsi_window=rsi_window,
            oversold=oversold,
            overbought=overbought,
            fee_bps=fee_bps,
            initial_capital=initial_capital,
        )
        return to_jsonable({"ticker": symbol, **result})

    @mcp.tool(annotations=READ_ONLY)
    @_user_errors
    def forecast_price_range(
        ticker: Ticker,
        horizon_days: Annotated[
            int, Field(description="Trading sessions ahead to simulate (1-252; 21 ≈ 1 month)")
        ] = 21,
        target_price: Annotated[
            float | None,
            Field(description="Optional price level: probability of ending above / touching it"),
        ] = None,
        method: Annotated[
            Literal["bootstrap", "gbm"],
            Field(
                description="bootstrap: resample the stock's own daily returns (fat tails). "
                "gbm: geometric Brownian motion with historical drift and volatility."
            ),
        ] = "bootstrap",
        lookback: Annotated[
            Literal["1y", "2y", "5y", "10y"], Field(description="History to learn from")
        ] = "2y",
        simulations: Annotated[int, Field(description="Number of simulated paths")] = 10_000,
        seed: Annotated[
            int | None, Field(description="Random seed, for reproducible results")
        ] = None,
    ) -> dict[str, Any]:
        """Probabilistic price range by Monte Carlo simulation: percentile range (p5-p95) of the
        price after `horizon_days`, probability of ending up/down, of hitting a target, of a
        10% drawdown along the way, plus how often the stock actually rose over that horizon
        historically. NOT a prediction: it assumes the future resembles the sampled past."""
        symbol = normalize_ticker(ticker)
        validate_int("horizon_days", horizon_days, minimum=1, maximum=252)
        validate_int("simulations", simulations, minimum=1_000, maximum=20_000)
        history = data.get_history(symbol, validate_period(lookback), "1d")
        result = monte_carlo_forecast(
            history["Close"],
            horizon_days,
            simulations=simulations,
            method=method,
            target_price=target_price,
            seed=seed,
        )
        return to_jsonable({"ticker": symbol, **result})

    @mcp.prompt(title="Analyse a stock")
    def analyze_ticker(ticker: str) -> str:
        """Full technical review of one ticker, written for a non-expert."""
        return (
            f"Analyse {ticker} for me. Call `get_quote` and `analyze_stock`, then explain in plain "
            "language: the current trend, momentum, volatility and risk, and the key price "
            "levels to watch. Finish with a short bullish vs bearish summary and remind me "
            "this is educational analysis of past data, not financial advice."
        )

    @mcp.prompt(title="Compare stocks")
    def compare_stocks(tickers: str) -> str:
        """Compare several tickers (comma-separated) on performance and risk."""
        symbols = [t.strip() for t in tickers.split(",") if t.strip()]
        return (
            f"Compare these stocks over the last year: {', '.join(symbols)}. Use "
            "`compare_tickers`, then present a table of return, volatility, Sharpe ratio and "
            "max drawdown, comment on how correlated they are and which looks strongest on a "
            "risk-adjusted basis. Remind me this is not financial advice."
        )

    @mcp.prompt(title="Evaluate a trading strategy")
    def evaluate_strategy(ticker: str) -> str:
        """Backtest SMA crossover and RSI mean reversion on a ticker versus buy-and-hold."""
        return (
            f"Backtest both the `sma_crossover` (50/200) and `rsi_reversion` strategies on "
            f"{ticker} over 5 years with `backtest_strategy`. Compare each with buy-and-hold on "
            "return, drawdown and Sharpe ratio, explain why they won or lost, and mention the "
            "limits of backtesting (overfitting, costs, regime changes)."
        )

    @mcp.prompt(title="Probabilistic outlook")
    def probability_outlook(ticker: str, horizon_days: str = "21") -> str:
        """Range of plausible prices for a ticker, with probabilities and their limits."""
        return (
            f"Use `forecast_price_range` on {ticker} with horizon_days={horizon_days}, once with "
            "method 'bootstrap' and once with 'gbm'. Explain the p5-p95 price range, the "
            "probability of ending higher, of a 10% drawdown along the way, and compare with the "
            "historical base rate. Be explicit that this is a simulation assuming the future "
            "resembles the past, not a prediction, and quote the disclaimer."
        )

    return mcp


def main() -> None:
    # stdout is the protocol channel: logs go to stderr, and yfinance's own error logs are
    # silenced because failures are already reported to the client as tool errors.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    create_server().run("stdio")


if __name__ == "__main__":
    main()
