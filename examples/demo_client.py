"""Smoke test / demo: start the server over stdio, like Claude Desktop does, and call its tools.

    uv run python examples/demo_client.py AAPL MSFT

Needs internet access (live Yahoo Finance data).
"""

from __future__ import annotations

import asyncio
import json
import sys

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters


async def main(tickers: list[str]) -> None:
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "stock_market_analyst_mcp.server"]
    )
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        print("Tools:", ", ".join(t.name for t in tools))

        calls = [
            ("get_quote", {"ticker": tickers[0]}),
            ("analyze_stock", {"ticker": tickers[0]}),
            ("compare_tickers", {"tickers": tickers}),
            ("backtest_strategy", {"ticker": tickers[0], "strategy": "sma_crossover"}),
            ("get_quote", {"ticker": "NOT_A_TICKER!"}),  # error path
        ]
        for name, arguments in calls:
            result = await client.call_tool(name, arguments)
            text = result.content[0].text
            print(f"\n=== {name}({arguments}) error={result.is_error}")
            if result.is_error:
                print(text)
                continue
            data = json.loads(text)
            summary = {
                "get_quote": lambda d: {k: d.get(k) for k in ("name", "price", "change_pct")},
                "analyze_stock": lambda d: {"trend": d["trend"]["label"], "signals": d["signals"]},
                "compare_tickers": lambda d: {
                    t: m["total_return"] for t, m in d["metrics"].items()
                },
                "backtest_strategy": lambda d: {
                    "strategy": d["strategy_result"]["total_return"],
                    "buy_and_hold": d["buy_and_hold_result"]["total_return"],
                },
            }[name](data)
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["AAPL", "MSFT"]))
