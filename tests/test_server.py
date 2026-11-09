"""End-to-end tests through a real MCP client, connected in memory to the server."""

import json

import pytest
from mcp.client import Client

from stock_market_analyst_mcp.server import create_server

from .conftest import FakeProvider

pytestmark = pytest.mark.anyio

EXPECTED_TOOLS = {
    "get_quote",
    "get_price_history",
    "technical_indicators",
    "analyze_stock",
    "compare_tickers",
    "backtest_strategy",
}


@pytest.fixture
async def client():
    async with Client(create_server(FakeProvider())) as c:
        yield c


async def call(client, name, **arguments):
    result = await client.call_tool(name, arguments)
    text = result.content[0].text
    return result.is_error, (text if result.is_error else json.loads(text))


async def test_lists_tools_and_prompts(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == EXPECTED_TOOLS
    assert all(t.description for t in tools.values())
    assert all(t.annotations.read_only_hint for t in tools.values())
    prompts = {p.name for p in (await client.list_prompts()).prompts}
    assert prompts == {"analyze_ticker", "compare_stocks", "evaluate_strategy"}


async def test_get_quote(client):
    error, data = await call(client, "get_quote", ticker=" aapl ")
    assert not error
    assert data["ticker"] == "AAPL"
    assert data["name"] == "AAPL Corp."
    assert data["price"] > 0
    assert data["change_pct"] == pytest.approx(data["price"] / data["previous_close"] - 1, abs=1e-3)


async def test_get_price_history_limit(client):
    error, data = await call(client, "get_price_history", ticker="MSFT", period="3mo", limit=5)
    assert not error
    assert len(data["bars"]) == 5
    assert data["summary"]["bars"] == 63
    assert set(data["bars"][0]) == {"Date", "Open", "High", "Low", "Close", "Volume"}


async def test_technical_indicators_subset(client):
    error, data = await call(
        client, "technical_indicators", ticker="NVDA", indicators=["rsi", "macd"], last_n=3
    )
    assert not error
    assert set(data["latest"]) == {"rsi_14", "macd"}
    assert 0 <= data["latest"]["rsi_14"]["value"] <= 100
    assert len(data["series"]) == 3


async def test_analyze_stock(client):
    error, data = await call(client, "analyze_stock", ticker="SHOP.TO")
    assert not error
    for section in ("returns", "trend", "momentum", "volatility", "key_levels", "signals"):
        assert section in data
    assert "not financial advice" in data["disclaimer"]


async def test_compare_tickers(client):
    error, data = await call(client, "compare_tickers", tickers=["AAPL", "MSFT", "aapl"])
    assert not error
    assert data["tickers"] == ["AAPL", "MSFT"]  # deduplicated
    assert set(data["metrics"]) == {"AAPL", "MSFT"}


async def test_backtest_strategy(client):
    error, data = await call(client, "backtest_strategy", ticker="SPY", strategy="sma_crossover")
    assert not error
    assert {"strategy_result", "buy_and_hold_result", "outperformance"} <= set(data)


@pytest.mark.parametrize(
    ("tool", "arguments", "message"),
    [
        ("get_quote", {"ticker": "NOT A TICKER!"}, "not a valid ticker"),
        ("get_quote", {"ticker": "NOPE"}, "No price data"),
        ("compare_tickers", {"tickers": ["AAPL"]}, "between 2 and 10"),
        ("backtest_strategy", {"ticker": "SPY", "fast": 200, "slow": 50}, "shorter than"),
        ("get_price_history", {"ticker": "SPY", "limit": 0}, "'limit'"),
        ("get_price_history", {"ticker": "SPY", "period": "7y"}, "period"),
    ],
)
async def test_errors_are_readable(client, tool, arguments, message):
    error, text = await call(client, tool, **arguments)
    assert error
    assert message in text


async def test_prompt_renders(client):
    result = await client.get_prompt("analyze_ticker", {"ticker": "AAPL"})
    assert "AAPL" in result.messages[0].content.text


async def test_technical_indicators_defaults_to_all(client):
    error, data = await call(client, "technical_indicators", ticker="AMD")
    assert not error
    assert set(data["latest"]) == {"sma_50", "ema_20", "rsi_14", "macd", "bollinger"}
