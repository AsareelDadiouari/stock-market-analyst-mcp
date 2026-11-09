"""Input validation shared by every tool."""

from __future__ import annotations

import re

from .errors import InvalidInputError

VALID_PERIODS = ("1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y", "10y", "max")
VALID_INTERVALS = ("1d", "1wk", "1mo")

# Letters/digits plus the punctuation used by Yahoo symbols: BRK-B, RY.TO, ^GSPC, EURUSD=X, BTC-USD
_TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")


def normalize_ticker(ticker: str) -> str:
    if not isinstance(ticker, str) or not ticker.strip():
        raise InvalidInputError("Ticker must be a non-empty string, e.g. 'AAPL'.")
    symbol = ticker.strip().upper()
    if not _TICKER_RE.match(symbol):
        raise InvalidInputError(
            f"'{ticker}' is not a valid ticker symbol. Use a Yahoo Finance symbol such as "
            "'AAPL', 'BRK-B', 'SHOP.TO', '^GSPC' or 'BTC-USD'."
        )
    return symbol


def normalize_tickers(tickers: list[str], *, min_count: int = 1, max_count: int = 10) -> list[str]:
    if not isinstance(tickers, list):
        raise InvalidInputError("Tickers must be a list of symbols, e.g. ['AAPL', 'MSFT'].")
    symbols = list(dict.fromkeys(normalize_ticker(t) for t in tickers))  # dedupe, keep order
    if not min_count <= len(symbols) <= max_count:
        raise InvalidInputError(
            f"Provide between {min_count} and {max_count} distinct tickers (got {len(symbols)})."
        )
    return symbols


def validate_period(period: str) -> str:
    if period not in VALID_PERIODS:
        raise InvalidInputError(
            f"Invalid period '{period}'. Choose one of: {', '.join(VALID_PERIODS)}."
        )
    return period


def validate_interval(interval: str) -> str:
    if interval not in VALID_INTERVALS:
        raise InvalidInputError(
            f"Invalid interval '{interval}'. Choose one of: {', '.join(VALID_INTERVALS)}."
        )
    return interval


def validate_int(name: str, value: int, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise InvalidInputError(f"'{name}' must be an integer between {minimum} and {maximum}.")
    return value
