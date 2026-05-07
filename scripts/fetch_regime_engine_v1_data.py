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
from regime_data_fetch.bls_schedule import build_bls_local_archive_page_fetcher
from regime_data_fetch.event_calendar import run_us_event_calendar_fetch
from regime_data_fetch.fetch_workflow import run_macro_fetch, run_market_fetch
from regime_data_fetch.aggregate_eps import run_aggregate_eps_fetch, run_wayback_aggregate_eps_fetch
from regime_data_fetch.fomc_minutes import run_fomc_minutes_fetch
from regime_data_fetch.local_usd_index import run_local_usd_index_import
from regime_data_fetch.pmi import run_pmi_fetch
from regime_data_fetch.pit_constituents import run_pit_constituents_fetch
from regime_data_fetch.powell_speeches import run_powell_speeches_fetch
from regime_data_fetch.universe import build_or_load_us_universe_10b_cache


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch regime-engine raw market and macro data for V1/V2 scopes.")
    ap.add_argument("--market-data-hub-root", default=None, help="Path to the market-data-hub repo (required when stock-universe fetch is enabled).")
    ap.add_argument("--out-dir", default="data/raw", help="Output directory for Parquet + reports.")
    ap.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD).")
    ap.add_argument("--end", default=dt.date.today().isoformat(), help="End date (YYYY-MM-DD).")
    ap.add_argument("--scope", default="v1", help="Data scope: v1|v2|all.")
    ap.add_argument("--fetch", default="market", help="What to fetch: market|macro|events|pmi|pit|fomc|powell|eps|eps-wayback|usd-index-local|all.")
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
    ap.add_argument(
        "--eps-workbook",
        default=None,
        help="Path to a manually downloaded S&P aggregate EPS workbook (.xlsx). Required for --fetch eps.",
    )
    ap.add_argument("--eps-wayback-max-snapshots", type=int, default=None, help="Optional maximum number of Wayback EPS snapshots to process.")
    ap.add_argument("--eps-wayback-from", default=None, help="Optional lower bound date (YYYY-MM-DD) for Wayback EPS snapshot dates.")
    ap.add_argument("--eps-wayback-to", default=None, help="Optional upper bound date (YYYY-MM-DD) for Wayback EPS snapshot dates.")
    ap.add_argument("--eps-wayback-stop-after-first-success", action="store_true", help="Stop Wayback EPS processing after the first successfully parsed snapshot.")
    ap.add_argument("--usd-index-csv", default=None, help="Path to a local Yahoo Finance ^NYICDX historical CSV export. Required for --fetch usd-index-local.")
    ap.add_argument("--acquisition-db", default=None, help="Optional SQLite path for raw acquisition/provenance recording.")
    ap.add_argument("--bls-schedule-dir", default=None, help="Optional local directory containing bls_schedule_YYYY.html files for BLS historical release schedules.")
    ap.add_argument("--bls-start-year", type=int, default=2000, help="Start year for BLS CPI/NFP schedule generation.")
    ap.add_argument("--bls-end-year", type=int, default=None, help="End year for BLS CPI/NFP schedule generation. Defaults to the current year.")
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
    if args.fetch not in {"market", "macro", "events", "pmi", "pit", "fomc", "powell", "eps", "eps-wayback", "usd-index-local", "all"}:
        raise SystemExit("--fetch must be market|macro|events|pmi|pit|fomc|powell|eps|eps-wayback|usd-index-local|all")

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
            acquisition_db_path=Path(args.acquisition_db) if args.acquisition_db else None,
        )
        print(str(market_report))

    if args.fetch in {"macro", "all"}:
        macro_report = run_macro_fetch(
            out_dir=out_dir,
            start=start,
            end=end,
            fred_api_key=args.fred_api_key,
            include_cpi_vintages=args.include_cpi_vintages,
            acquisition_db_path=Path(args.acquisition_db) if args.acquisition_db else None,
        )
        print(str(macro_report))

    if args.fetch in {"events", "all"}:
        bls_page_fetcher = None
        if args.bls_schedule_dir:
            bls_page_fetcher = build_bls_local_archive_page_fetcher(
                schedule_dir=Path(args.bls_schedule_dir),
            )
        event_report = run_us_event_calendar_fetch(
            repo_root=REPO_ROOT,
            fred_api_key=args.fred_api_key or None,
            bls_page_fetcher=bls_page_fetcher,
            acquisition_db_path=Path(args.acquisition_db) if args.acquisition_db else None,
            bls_start_year=args.bls_start_year,
            bls_end_year=args.bls_end_year,
        )
        print(str(event_report))

    if args.fetch in {"pmi", "all"}:
        pmi_report = run_pmi_fetch(
            out_dir=out_dir,
            as_of_date=end,
        )
        print(str(pmi_report))

    if args.fetch in {"pit", "all"}:
        pit_report = run_pit_constituents_fetch(
            out_dir=out_dir,
        )
        print(str(pit_report))

    if args.fetch in {"fomc", "all"}:
        fomc_report = run_fomc_minutes_fetch(
            out_dir=out_dir,
        )
        print(str(fomc_report))

    if args.fetch in {"powell", "all"}:
        powell_report = run_powell_speeches_fetch(
            out_dir=out_dir,
        )
        print(str(powell_report))

    if args.fetch in {"eps", "all"}:
        if not args.eps_workbook:
            raise SystemExit("--eps-workbook is required for eps fetches")
        eps_report = run_aggregate_eps_fetch(
            out_dir=out_dir,
            workbook_path=Path(args.eps_workbook),
            acquisition_db_path=Path(args.acquisition_db) if args.acquisition_db else None,
        )
        print(str(eps_report))

    if args.fetch in {"eps-wayback", "all"}:
        eps_wayback_report = run_wayback_aggregate_eps_fetch(
            out_dir=out_dir,
            max_snapshots=args.eps_wayback_max_snapshots,
            from_date=parse_date(args.eps_wayback_from) if args.eps_wayback_from else None,
            to_date=parse_date(args.eps_wayback_to) if args.eps_wayback_to else None,
            stop_after_first_success=args.eps_wayback_stop_after_first_success,
        )
        print(str(eps_wayback_report))

    if args.fetch == "usd-index-local":
        if not args.usd_index_csv:
            raise SystemExit("--usd-index-csv is required for usd-index-local fetches")
        usd_index_report = run_local_usd_index_import(
            out_dir=out_dir,
            csv_path=Path(args.usd_index_csv),
            acquisition_db_path=Path(args.acquisition_db) if args.acquisition_db else None,
        )
        print(str(usd_index_report))
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
