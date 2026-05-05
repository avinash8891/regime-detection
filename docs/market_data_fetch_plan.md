# Market Data Fetch Plan (V1, Direct From Market-Data-Hub Alpaca REST)

Date range: `2015-01-01` through “today” (script default: `--end <today>`). We verify the actual earliest returned trading date per required symbols in the fetch report.

| Dataset | Symbols | Columns (all required) | Source | Output (repo-local) |
|---|---|---|---|---|
| Daily OHLCV (market anchor) | `SPY` | `date,symbol,open,high,low,close,volume,adjusted_close` | Alpaca REST (StockHistoricalDataClient / daily bars) | `data/raw/daily_ohlcv/` (Parquet dataset partitioned by `symbol`) |
| Daily OHLCV (breadth proxy) | `RSP` | `date,symbol,open,high,low,close,volume,adjusted_close` | Alpaca REST (daily bars) | `data/raw/daily_ohlcv/` |
| Daily close (vol proxy) | `VIX` or `^VIX` | `date,symbol,close` | Alpaca REST only (no fallback) | `data/raw/daily_ohlcv/` (same dataset; script errors if neither symbol is returned) |
| Daily OHLCV (stock universe) | 10B+ US stocks (target `762`) | `date,symbol,open,high,low,close,volume,adjusted_close` | Alpaca REST (daily bars; batched requests) | `data/raw/daily_ohlcv/` |

Universe source: derived from the `market-data-hub` US universe seed list, optionally refreshed into a repo-local cache JSON (`data/raw/universe/us_universe_cache.json`) using yfinance market-cap checks (one-time build via `--build-universe`).

Non-goals for this plan: intraday/5m candles (not fetched).

