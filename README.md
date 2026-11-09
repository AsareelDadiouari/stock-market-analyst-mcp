# Stock Market Analyst MCP

[![CI](https://github.com/AsareelDadiouari/stock-market-analyst-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/AsareelDadiouari/stock-market-analyst-mcp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![MCP](https://img.shields.io/badge/MCP-server-8A2BE2)
![License](https://img.shields.io/badge/license-MIT-green)

A **[Model Context Protocol](https://modelcontextprotocol.io) server** that turns Claude (or any
MCP client) into a stock market analyst. It fetches market data, computes technical indicators,
compares tickers, backtests trading rules and simulates probabilistic price ranges — then hands
structured results to the model, which explains them in plain language.

> [!CAUTION]
> **Not financial advice. Use at your own risk.** This is an educational project. Its output can
> be wrong, and its "probabilities" are simulations based on past data, not predictions. The
> author accepts no liability for any loss arising from its use. See the full
> [disclaimer](#disclaimer).

> Ask *"Is Microsoft in an uptrend? Compare it with Apple and Shopify over the last year"* and
> Claude calls the tools below instead of guessing from its training data.

## Demo

```text
You     Analyse MSFT for me.

Claude  ▸ get_quote(ticker="MSFT")
        ▸ analyze_stock(ticker="MSFT")

        Microsoft closed at $530.61. It is in a strong uptrend: the price sits above its
        50-day ($516.65) and 200-day ($438.85) averages, and the 50-day crossed above the
        200-day ("golden cross") 49 sessions ago. Returns: +5% over 3 months, +28% over 6.

        Momentum is cooling: RSI(14) is a neutral 52 and the MACD made a bearish crossover
        14 sessions ago. Key levels: support around $505 (20-day low), resistance near $559,
        Bollinger range $512–$551.

        Bullish case: long-term trend intact. Bearish case: short-term momentum fading.
        This is an educational reading of past data, not financial advice.
```

Numbers above come from a real run of `analyze_stock` (November 2026).

## Tools

| Tool | What it does |
| --- | --- |
| `get_quote` | Latest price, daily change, volume, 52-week range, company details |
| `get_price_history` | Adjusted OHLCV bars (daily / weekly / monthly) plus a period summary |
| `technical_indicators` | SMA, EMA, RSI (Wilder), MACD (12/26/9), Bollinger Bands (20, 2σ) with interpretation |
| `analyze_stock` | One structured overview: returns, trend, momentum, volatility, drawdown, support/resistance, signals |
| `compare_tickers` | 2–10 tickers on common dates: return, CAGR, volatility, Sharpe, drawdown, ranking, correlation matrix |
| `backtest_strategy` | SMA crossover or RSI mean-reversion vs buy-and-hold, with fees and no look-ahead bias |
| `forecast_price_range` | Monte Carlo simulation: p5–p95 price range after N sessions, probability of ending up/down, of hitting a target, of a 10 % drawdown, vs the historical base rate |

**Prompts** (reusable templates exposed to the client): `analyze_ticker`, `compare_stocks`,
`evaluate_strategy`, `probability_outlook`.

Every tool validates its inputs (ticker format, periods, windows, ranges) and returns clear,
actionable error messages the model can recover from — e.g. *"'NOT A TICKER!' is not a valid
ticker symbol. Use a Yahoo Finance symbol such as 'AAPL', 'BRK-B', 'SHOP.TO'…"*.
Tickers use Yahoo Finance symbols: `AAPL`, `BRK-B`, `SHOP.TO`, `^GSPC`, `BTC-USD`, `EURUSD=X`.

## Probabilistic forecasts: how they work (and why they are not predictions)

`forecast_price_range` answers a narrow, honest question: *if the next N sessions behave
statistically like the last few years, what range of prices is plausible?*

1. Take the stock's daily log returns over the lookback window (default 2 years).
2. Simulate 10,000 price paths N sessions ahead, either by **bootstrap** (resampling the
   stock's own past daily returns, which keeps fat tails) or **GBM** (geometric Brownian
   motion with the historical drift and volatility).
3. Summarise the simulated outcomes: percentiles of the final price, share of paths ending
   above today's price or a target, share of paths that dip 10 % along the way.
4. Report the **historical base rate** next to it: how often the stock *actually* rose over
   the same horizon in the sampled history, as a sanity check.

Example (NVDA, 21 sessions, live data, November 2026): 90 % of simulated paths end between
$152 and $233, 54 % end above today's price, and 30 % suffer a 10 % drawdown at some point.

**Limits:** the simulation assumes the future resembles the sampled past. It knows nothing about
earnings, news, interest rates or regime changes, and assumes daily returns are independent.
Real prices regularly move outside the simulated range. Read the numbers as a description of
historical volatility, never as odds of what will happen.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AsareelDadiouari/stock-market-analyst-mcp.git
cd stock-market-analyst-mcp
uv sync
uv run stock-market-analyst-mcp   # starts the server on stdio (used by MCP clients)
```

### Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "stock-market-analyst": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/AsareelDadiouari/stock-market-analyst-mcp",
        "stock-market-analyst-mcp"
      ]
    }
  }
}
```

Or, from a local clone: `"command": "uv", "args": ["--directory", "/path/to/stock-market-analyst-mcp", "run", "stock-market-analyst-mcp"]`.

### Claude Code

```bash
claude mcp add stock-market-analyst -- uvx --from git+https://github.com/AsareelDadiouari/stock-market-analyst-mcp stock-market-analyst-mcp
```

### MCP Inspector / scripted client

```bash
npx @modelcontextprotocol/inspector uv run stock-market-analyst-mcp
uv run python examples/demo_client.py AAPL MSFT   # real stdio round-trip with live data
```

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `STOCK_MCP_CACHE_TTL` | `300` | Seconds market data stays in the in-memory cache |

## Architecture

```text
 MCP client (Claude Desktop, Claude Code, Inspector)
        │  JSON-RPC over stdio
        ▼
 server.py ─ MCPServer: tool & prompt definitions, input validation, error mapping
        │
        ├── analysis.py   quotes, stock overview, multi-ticker comparison
        ├── backtest.py   vectorised long-only backtests vs buy-and-hold
        ├── indicators.py SMA, EMA, RSI, MACD, Bollinger, volatility, drawdown (pandas)
        │
        ▼
 providers.py ─ MarketDataProvider (Protocol)
        ├── CachedProvider   thread-safe TTL cache (decorator)
        └── YFinanceProvider Yahoo Finance via yfinance (swappable)
```

Design choices:

- **Provider interface** — tools depend on a `MarketDataProvider` protocol, not on yfinance. The
  data source can be swapped (paid API, database) and tests inject an offline fake.
- **Pure analytics core** — indicators, analysis and backtests are plain functions on pandas
  objects with no I/O, which makes them easy to unit-test.
- **Correct by construction** — RSI uses Wilder's smoothing and is checked against the classic
  published example; Bollinger Bands use population σ; backtests apply signals on the *next*
  session so they never trade on information they could not have had, and deduct fees.
- **Model-friendly output** — structured JSON (NaN → `null`, rounded floats, ISO dates) plus
  plain-language labels (`"strong uptrend"`, `"overbought"`) and signals the model can quote.
- **Typed tool schemas** — `Literal` types and pydantic `Field` descriptions generate precise JSON
  schemas, so the client knows the allowed periods, strategies and indicators up front.

## Development

```bash
uv sync                 # install runtime + dev dependencies
uv run pytest           # 48 tests, fully offline (mocked market data)
uv run ruff check .     # lint
uv run ruff format .    # format
```

Tests cover indicator maths against known values, backtest correctness (look-ahead, fees),
Monte Carlo properties (deterministic edge cases, reproducible seeds), caching, input validation, and every tool end-to-end through a real MCP client connected
in memory to the server. CI runs lint and tests on Python 3.11–3.13 with GitHub Actions.

Built with Python, the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk),
pandas, NumPy and yfinance.

## Disclaimer

**This software is provided for educational and informational purposes only. It is not
financial, investment, tax or legal advice, and it is not a recommendation to buy, sell or hold
any security.**

- **No predictions.** All analysis is based on historical data. Forecast "probabilities" are
  statistical simulations that assume the future resembles the past; they are not predictions,
  guarantees or reliable odds of any outcome.
- **No accuracy guarantee.** Market data comes from Yahoo Finance through the unofficial
  `yfinance` library and may be delayed, incomplete or wrong. Calculations may contain bugs.
  Language models can also misread or misreport tool output.
- **Your decisions, your risk.** Investing involves risk, including the loss of principal.
  Do your own research and consult a licensed financial professional before making any
  investment decision.
- **No liability.** To the maximum extent permitted by law, the author and contributors accept
  no liability for any loss or damage, direct or indirect, arising from the use of this
  software or its output. The software is provided "as is", without warranty of any kind, as
  stated in the [MIT License](LICENSE).
- **Data terms.** Yahoo Finance data is intended for personal use; check its terms before any
  other use.

By using this software you acknowledge that you have read and accepted this disclaimer.

## License

[MIT](LICENSE) © 2026 Asaréel Dadiouari
