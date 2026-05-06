#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import sys

# Allow running as a script without requiring PYTHONPATH/installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from regime_data_fetch.cli_common import load_env_file, parse_date
from regime_data_fetch.fetch_workflow import run_macro_fetch, run_market_fetch
from regime_data_fetch.pmi import run_pmi_fetch
from regime_data_fetch.universe import build_or_load_us_universe_10b_cache


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch regime-engine raw market and macro data for V1/V2 scopes.")
    ap.add_argument("--market-data-hub-root", default=None, help="Path to the market-data-hub repo (required when stock-universe fetch is enabled).")
    ap.add_argument("--out-dir", default="data/raw", help="Output directory for Parquet + reports.")
    ap.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD).")
    ap.add_argument("--end", default=dt.date.today().isoformat(), help="End date (YYYY-MM-DD).")
    ap.add_argument("--scope", default="v1", help="Data scope: v1|v2|all.")
    ap.add_argument("--fetch", default="market", help="What to fetch: market|macro|pmi|all.")
    ap.add_argument("--min-cap-b", type=float, default=10.0, help="Universe filter threshold in $B.")
    ap.add_argument("--adjustment", default="raw", help="Alpaca adjustment: raw|split|dividend|all.")
    ap.add_argument("--alpaca-feed", default=None, help="Alpaca data feed: sip|iex|otc. Omit to use SDK default.")
    ap.add_argument("--fred-api-key", default=None, help="Optional FRED API key for macro fetches.")
    ap.add_argument("--include-cpi-vintages", action="store_true", help="Also fetch CPI vintages via ALFRED-style realtime observations.")
    ap.add_argument("--list-symbols", action="store_true", help="Only print symbol counts and exit.")
    ap.add_argument("--build-universe", action="store_true", help="Force-refresh the 10B+ universe cache (network: yfinance).")
    ap.add_argument("--env-file", default=None, help="Optional .env file to load (for Alpaca creds).")
    ap.add_argument(
        "--universe-json",
        default=None,
        help="Optional path to a JSON list[str] of symbols to fetch (use this for the 762-symbol final universe).",
    )
    ap.add_argument(
        "--vix-symbol",
        default="VIX",
        help="Volatility proxy symbol to fetch from Alpaca. Use VIX when your account supports it, or VIXY as the documented proxy.",
    )
    ap.add_argument(
        "--allow-vix-proxy",
        action="store_true",
        help="Allow proceeding when true VIX is unavailable, using --vix-symbol (e.g. VIXY).",
    )
    ap.add_argument("--verbose", action="store_true", help="Print progress while fetching.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = parse_date(args.start)
    end = parse_date(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    if args.scope not in {"v1", "v2", "all"}:
        raise SystemExit("--scope must be v1|v2|all")
    if args.fetch not in {"market", "macro", "pmi", "all"}:
        raise SystemExit("--fetch must be market|macro|pmi|all")

    if args.env_file:
        load_env_file(Path(args.env_file))

    if args.list_symbols:
        stocks = _resolve_stock_universe(args)
        print(
            json.dumps(
                {
                    "scope": args.scope,
                    "stocks_count": len(stocks),
                    "note": (
                        "V1/all stock-universe fetches use the market-data-hub-derived cache. "
                        "V2 scope adds the fixed ETF/cross-asset universe in code."
                    ),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.fetch in {"market", "all"}:
        stocks = _resolve_stock_universe(args) if args.scope in {"v1", "all"} else []
        market_report = run_market_fetch(
            out_dir=out_dir,
            scope=args.scope,
            stock_symbols=stocks,
            start=start,
            end=end,
            adjustment=args.adjustment,
            alpaca_feed=args.alpaca_feed,
            vix_symbol=args.vix_symbol,
            allow_vix_proxy=args.allow_vix_proxy,
            verbose=args.verbose,
        )
        print(str(market_report))

    if args.fetch in {"macro", "all"}:
        macro_report = run_macro_fetch(
            out_dir=out_dir,
            start=start,
            end=end,
            fred_api_key=args.fred_api_key,
            include_cpi_vintages=args.include_cpi_vintages,
        )
        print(str(macro_report))

    if args.fetch in {"pmi", "all"}:
        pmi_report = run_pmi_fetch(
            out_dir=out_dir,
            as_of_date=end,
        )
        print(str(pmi_report))
    return 0


def _resolve_stock_universe(args: argparse.Namespace) -> list[str]:
    if args.universe_json:
        universe_path = Path(args.universe_json)
        stocks = json.loads(universe_path.read_text())
        if not isinstance(stocks, list) or not all(isinstance(symbol, str) for symbol in stocks):
            raise SystemExit("--universe-json must be a JSON list[str]")
        return stocks

    if not args.market_data_hub_root:
        raise SystemExit("--market-data-hub-root is required for V1/all stock-universe fetches unless --universe-json is provided")

    uni = build_or_load_us_universe_10b_cache(
        market_data_hub_root=args.market_data_hub_root,
        out_dir=Path(args.out_dir) / "universe",
        min_cap_b=args.min_cap_b,
        allow_update=args.build_universe or (not args.list_symbols),
    )
    return uni.symbols


if __name__ == "__main__":
    raise SystemExit(main())
