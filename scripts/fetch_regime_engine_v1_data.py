#!/usr/bin/env python3
# ruff: noqa: E402
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
from regime_data_fetch.artifact_export import emit_manifest_for_report_paths
from regime_data_fetch.aaii_sentiment import run_sentiment_fetch
from regime_data_fetch.fetch_workflow import run_macro_fetch, run_market_fetch
from regime_data_fetch.aggregate_eps import (
    run_aggregate_eps_fetch,
    run_aggregate_eps_auto_fetch,
    run_wayback_aggregate_eps_fetch,
)
from regime_data_fetch.fomc_minutes import run_fomc_minutes_fetch
from regime_data_fetch.investing_archive import run_local_investing_archive_import
from regime_data_fetch.investing_live import run_investing_live_fetch
from regime_data_fetch.cleveland_fed_nowcast import run_cleveland_fed_nowcast_fetch
from regime_data_fetch.local_daily_ohlcv_sqlite import (
    run_alpaca_constituent_daily_ohlcv_fetch,
    run_local_daily_ohlcv_sqlite_import,
)
from regime_data_fetch.local_usd_index import run_local_usd_index_import
from regime_data_fetch.pmi import run_pmi_fetch
from regime_data_fetch.pit_constituents import run_pit_constituents_fetch
from regime_data_fetch.powell_speeches import run_powell_speeches_fetch
from regime_data_fetch.sf_fed_news_sentiment import run_sf_fed_news_sentiment_fetch
from regime_data_fetch.universe import (
    FIXED_UNIVERSE_SYMBOL_COUNT,
    FIXED_UNIVERSE_TREE_NAME,
    load_symbols_from_daily_ohlcv_tree,
    load_symbols_from_pit_constituents_parquet,
)


UNATTENDED_FETCH_MODES = frozenset(
    {
        "market",
        "macro",
        "events",
        "pmi",
        "pit",
        "fomc",
        "powell",
        "sentiment",
        "cleveland-fed-nowcast",
        "sf-fed-news-sentiment",
        "daily-ohlcv-constituents-alpaca",
    }
)
OPERATOR_ASSISTED_FETCH_MODES = frozenset(
    {
        "eps",
        "eps-spglobal-auto",
        "eps-wayback",
        "usd-index-local",
        "daily-ohlcv-local-sqlite",
        "investing-archive-local",
        "investing-live",
    }
)
FETCH_MODES = UNATTENDED_FETCH_MODES | OPERATOR_ASSISTED_FETCH_MODES | {"all"}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch regime-engine raw market and macro data for V1/V2 scopes. "
            "--fetch all is reserved for unattended autonomous refreshes; browser-session, "
            "local-file, and historical-backfill tools remain explicit operator-assisted fetches."
        )
    )
    ap.add_argument("--out-dir", default="data/raw", help="Output directory for Parquet + reports.")
    ap.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD).")
    ap.add_argument("--end", default=dt.date.today().isoformat(), help="End date (YYYY-MM-DD).")
    ap.add_argument("--scope", default="v1", help="Data scope: v1|v2|all.")
    ap.add_argument("--fetch", default="market", help=f"What to fetch: {'|'.join(sorted(FETCH_MODES))}.")
    ap.add_argument("--min-cap-b", type=float, default=10.0, help="Universe filter threshold in $B.")
    ap.add_argument("--adjustment", default="raw", help="Alpaca adjustment: raw|split|dividend|all.")
    ap.add_argument("--alpaca-feed", default=None, help="Alpaca data feed: sip|iex|otc. Omit to use SDK default.")
    ap.add_argument("--fred-api-key", default=None, help="Optional FRED API key for macro fetches.")
    ap.add_argument(
        "--include-cpi-vintages",
        dest="include_cpi_vintages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fetch CPI vintages via FRED ALFRED-style realtime observations. "
            "Required by V2 §2A lines 2587-2593 for first-release replay PIT "
            "accuracy (see docs/spec_code_data_audit_2026_05_15.md §3.2 / M2). "
            "Default True; pass --no-include-cpi-vintages to skip."
        ),
    )
    ap.add_argument("--list-symbols", action="store_true", help="Only print symbol counts and exit.")
    ap.add_argument("--env-file", default=None, help="Optional .env file to load (for Alpaca creds).")
    ap.add_argument(
        "--universe-json",
        default=None,
        help=f"Optional JSON list[str] of symbols to fetch. For constituent OHLCV, this should be the fixed {FIXED_UNIVERSE_SYMBOL_COUNT}-symbol universe.",
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
        help="Operator-assisted manual import: path to a downloaded S&P aggregate EPS workbook (.xlsx). Required for --fetch eps.",
    )
    ap.add_argument("--eps-wayback-max-snapshots", type=int, default=None, help="Operator-assisted backfill: optional maximum number of Wayback EPS snapshots to process.")
    ap.add_argument("--eps-wayback-from", default=None, help="Optional lower bound date (YYYY-MM-DD) for Wayback EPS snapshot dates.")
    ap.add_argument("--eps-wayback-to", default=None, help="Optional upper bound date (YYYY-MM-DD) for Wayback EPS snapshot dates.")
    ap.add_argument("--eps-wayback-stop-after-first-success", action="store_true", help="Stop Wayback EPS processing after the first successfully parsed snapshot.")
    ap.add_argument("--eps-browser-user-data-dir", default=None, help="Operator-assisted EPS auto fetch: persistent browser profile directory for S&P browser fallback.")
    ap.add_argument("--eps-browser-executable", default=None, help="Operator-assisted EPS auto fetch: Chrome/Chromium executable for S&P browser fallback.")
    ap.add_argument("--eps-browser-headless", action=argparse.BooleanOptionalAction, default=True, help="Operator-assisted EPS auto fetch: run browser fallback headless/headful. Default headless.")
    ap.add_argument("--eps-browser-timeout-ms", type=int, default=120000, help="Operator-assisted EPS auto fetch: timeout in milliseconds for S&P browser fallback.")
    ap.add_argument(
        "--usd-index-csv",
        default=None,
        help=(
            "Optional/manual Yahoo Finance ^NYICDX historical CSV export for --fetch usd-index-local. "
            "Routine future USD ingestion uses FRED DTWEXBGS through --fetch macro."
        ),
    )
    ap.add_argument("--daily-ohlcv-dir", default=None, help="Operator-assisted manual import: path to a local partitioned daily_ohlcv parquet directory. Required for --fetch daily-ohlcv-local-sqlite.")
    ap.add_argument("--pit-parquet", default=None, help="PIT constituent parquet for --fetch daily-ohlcv-constituents-alpaca. Defaults to <out-dir>/pit_constituents/sp500_ticker_intervals.parquet.")
    ap.add_argument(
        "--constituent-universe-dir",
        default=None,
        help=f"Fixed partitioned {FIXED_UNIVERSE_TREE_NAME} tree to use as the Alpaca constituent refresh universe.",
    )
    ap.add_argument(
        "--allow-pit-constituent-universe",
        action="store_true",
        help="Explicitly allow PIT-constituent parquet expansion as a bootstrap universe for Alpaca constituent OHLCV.",
    )
    ap.add_argument(
        "--constituent-universe-expected-count",
        type=int,
        default=FIXED_UNIVERSE_SYMBOL_COUNT,
        help=f"Expected fixed constituent universe size for Alpaca refreshes. Default {FIXED_UNIVERSE_SYMBOL_COUNT}.",
    )
    ap.add_argument("--allow-missing-constituent-symbols", action="store_true", help="Allow daily-ohlcv-constituents-alpaca to continue when Alpaca returns no bars for some PIT symbols.")
    ap.add_argument("--pmi-history-dir", default=None, help="Optional manual Investing PMI history directory. Omit for live DBnomics/TradingEconomics PMI ingestion.")
    ap.add_argument("--investing-archive-root", default=None, help="Operator-assisted manual import: path to archived Investing.com source_pages root. Required for --fetch investing-archive-local.")
    ap.add_argument("--investing-earnings-loaded-page", default=None, help="Operator-assisted Investing fetch: path to a browser-loaded earnings calendar HTML page containing __NEXT_DATA__. Optional for --fetch investing-live.")
    ap.add_argument("--investing-earnings-browser-capture", action=argparse.BooleanOptionalAction, default=True, help="Operator-assisted Investing fetch: capture a fresh earnings page with Playwright when no page/token is supplied. Default True.")
    ap.add_argument("--investing-browser-user-data-dir", default=None, help="Operator-assisted Investing fetch: persistent browser profile directory. Defaults to archive-local browser_profile or INVESTING_BROWSER_USER_DATA_DIR.")
    ap.add_argument("--investing-browser-executable", default=None, help="Operator-assisted Investing fetch: Chrome/Chromium executable. Defaults to Playwright browser or INVESTING_BROWSER_EXECUTABLE.")
    ap.add_argument("--investing-browser-headless", action=argparse.BooleanOptionalAction, default=None, help="Operator-assisted Investing fetch: run browser capture headless/headful. Defaults to INVESTING_BROWSER_HEADLESS or headful.")
    ap.add_argument("--investing-browser-timeout-ms", type=int, default=None, help="Operator-assisted Investing fetch: timeout in milliseconds while waiting for accessToken. Defaults to INVESTING_BROWSER_TIMEOUT_MS or 120000.")
    ap.add_argument("--acquisition-db", default=None, help="Optional SQLite path for raw acquisition/provenance recording.")
    ap.add_argument(
        "--artifact-store",
        default=None,
        help="Optional object-store root for emitted artifacts, e.g. s3://regime-data or /mnt/regime-data.",
    )
    ap.add_argument(
        "--emit-manifest",
        default=None,
        help="Optional manifest YAML path to write after fetch outputs are uploaded to --artifact-store.",
    )
    ap.add_argument(
        "--manifest-artifact-set",
        default=None,
        help="Optional artifact_set name for --emit-manifest. Defaults to regime_engine_<end-date>.",
    )
    ap.add_argument(
        "--manifest-required-for",
        default="profile_engine_30d,v2_calibration,historical_walkforward,audit_layer2_30d",
        help="Comma-separated use cases attached to emitted manifest artifacts.",
    )
    ap.add_argument("--bls-schedule-dir", default=None, help="Optional local directory containing bls_schedule_YYYY.html files for BLS historical release schedules.")
    ap.add_argument("--bls-start-year", type=int, default=2000, help="Start year for BLS CPI/NFP schedule generation.")
    ap.add_argument("--bls-end-year", type=int, default=None, help="End year for BLS CPI/NFP schedule generation. Defaults to --end year.")
    ap.add_argument(
        "--include-v2-curated-event-candidates",
        action="store_true",
        help="Add deterministic V2 curated event candidates: elections, budget deadlines, and official global-rate calendars.",
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
    _validate_fetch_modes()
    if args.fetch not in FETCH_MODES:
        raise SystemExit(f"--fetch must be {'|'.join(sorted(FETCH_MODES))}")
    if args.emit_manifest and not args.artifact_store:
        raise SystemExit("--artifact-store is required when --emit-manifest is set")

    # TODO(simplify): under --fetch all, the per-source fetches (market, macro,
    # sentiment, events, pmi, pit, fomc, powell, cleveland-fed-nowcast,
    # sf-fed-news-sentiment, daily-ohlcv-constituents-alpaca) run strictly in
    # series, even though they hit independent upstreams. A ThreadPoolExecutor
    # would cut wall-clock to ~max(N) instead of sum(N). Held back because
    # Alpaca shares a rate-limit budget with daily-ohlcv-constituents-alpaca —
    # confirm upstream rate-limit policy per source before parallelizing.
    if args.env_file:
        load_env_file(Path(args.env_file))

    if args.list_symbols:
        stocks = _resolve_stock_universe(args, out_dir=out_dir)
        print(
            json.dumps(
                {
                    "scope": args.scope,
                    "stocks_count": len(stocks),
                    "note": (
                        "V1/all stock-universe listings use --universe-json when supplied, otherwise "
                        "the PIT constituent parquet ticker set. Constituent OHLCV refreshes require "
                        f"the fixed {FIXED_UNIVERSE_SYMBOL_COUNT}-symbol artifact unless --allow-pit-constituent-universe is explicit."
                    ),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    report_paths: list[Path] = []
    acquisition_db_path = Path(args.acquisition_db) if args.acquisition_db else None
    acquisition_artifact_store_root = args.artifact_store if acquisition_db_path and args.artifact_store else None

    if _should_fetch(args.fetch, "market"):
        stocks = _resolve_stock_universe(args, out_dir=out_dir) if args.scope in {"v1", "all"} else []
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
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(market_report)
        print(str(market_report))

    if _should_fetch(args.fetch, "macro"):
        macro_report = run_macro_fetch(
            out_dir=out_dir,
            start=start,
            end=end,
            fred_api_key=args.fred_api_key,
            include_cpi_vintages=args.include_cpi_vintages,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(macro_report)
        print(str(macro_report))

    if _should_fetch(args.fetch, "sentiment"):
        sentiment_report = run_sentiment_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(sentiment_report)
        print(str(sentiment_report))

    if _should_fetch(args.fetch, "events"):
        bls_page_fetcher = None
        if args.bls_schedule_dir:
            bls_page_fetcher = build_bls_local_archive_page_fetcher(
                schedule_dir=Path(args.bls_schedule_dir),
            )
        event_report = run_us_event_calendar_fetch(
            repo_root=REPO_ROOT,
            fred_api_key=args.fred_api_key or None,
            bls_page_fetcher=bls_page_fetcher,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
            bls_start_year=args.bls_start_year,
            bls_end_year=args.bls_end_year,
            include_v2_curated_candidates=args.include_v2_curated_event_candidates,
            as_of_date=end,
        )
        report_paths.append(event_report)
        print(str(event_report))

    if _should_fetch(args.fetch, "pmi"):
        pmi_report = run_pmi_fetch(
            out_dir=out_dir,
            as_of_date=end,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
            manual_history_dir=Path(args.pmi_history_dir) if args.pmi_history_dir else None,
        )
        report_paths.append(pmi_report)
        print(str(pmi_report))

    if _should_fetch(args.fetch, "pit"):
        pit_report = run_pit_constituents_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(pit_report)
        print(str(pit_report))

    if _should_fetch(args.fetch, "fomc"):
        fomc_report = run_fomc_minutes_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(fomc_report)
        print(str(fomc_report))

    if _should_fetch(args.fetch, "powell"):
        powell_report = run_powell_speeches_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(powell_report)
        print(str(powell_report))

    if _should_fetch(args.fetch, "cleveland-fed-nowcast"):
        nowcast_report = run_cleveland_fed_nowcast_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(nowcast_report)
        print(str(nowcast_report))

    if _should_fetch(args.fetch, "sf-fed-news-sentiment"):
        news_sentiment_report = run_sf_fed_news_sentiment_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(news_sentiment_report)
        print(str(news_sentiment_report))

    if args.fetch == "eps":
        # Operator-assisted/manual import only; excluded from --fetch all.
        if not args.eps_workbook:
            raise SystemExit("--eps-workbook is required for eps fetches")
        eps_report = run_aggregate_eps_fetch(
            out_dir=out_dir,
            workbook_path=Path(args.eps_workbook),
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(eps_report)
        print(str(eps_report))

    if args.fetch == "eps-spglobal-auto":
        # Opt-in only; intentionally excluded from --fetch all because S&P
        # publishes the workbook weekly and the spdji URL can require a real
        # browser session. Cadence is WEEKLY; see
        # regime_data_fetch.aggregate_eps.download_spglobal_eps_workbook
        # docstring for the 4-week revision-direction rationale.
        eps_auto_report = run_aggregate_eps_auto_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
            browser_user_data_dir=Path(args.eps_browser_user_data_dir) if args.eps_browser_user_data_dir else None,
            browser_executable=Path(args.eps_browser_executable) if args.eps_browser_executable else None,
            browser_headless=args.eps_browser_headless,
            browser_timeout_ms=args.eps_browser_timeout_ms,
        )
        report_paths.append(eps_auto_report)
        print(str(eps_auto_report))

    if args.fetch == "eps-wayback":
        # Operator-assisted historical backfill only; excluded from --fetch all.
        eps_wayback_report = run_wayback_aggregate_eps_fetch(
            out_dir=out_dir,
            max_snapshots=args.eps_wayback_max_snapshots,
            from_date=parse_date(args.eps_wayback_from) if args.eps_wayback_from else None,
            to_date=parse_date(args.eps_wayback_to) if args.eps_wayback_to else None,
            stop_after_first_success=args.eps_wayback_stop_after_first_success,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(eps_wayback_report)
        print(str(eps_wayback_report))

    if args.fetch == "usd-index-local":
        # Optional diagnostic import only. The regime-engine USD input is
        # broad_usd_index from FRED DTWEXBGS, fetched by the macro workflow.
        if not args.usd_index_csv:
            raise SystemExit("--usd-index-csv is required for usd-index-local fetches")
        usd_index_report = run_local_usd_index_import(
            out_dir=out_dir,
            csv_path=Path(args.usd_index_csv),
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(usd_index_report)
        print(str(usd_index_report))

    if args.fetch == "investing-archive-local":
        # Operator-assisted archive import only; excluded from --fetch all.
        if not args.investing_archive_root:
            raise SystemExit("--investing-archive-root is required for investing-archive-local fetches")
        if not args.acquisition_db:
            raise SystemExit("--acquisition-db is required for investing-archive-local fetches")
        investing_report = run_local_investing_archive_import(
            out_dir=out_dir,
            archive_root=Path(args.investing_archive_root),
            acquisition_db_path=Path(args.acquisition_db),
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(investing_report)
        print(str(investing_report))

    if args.fetch == "investing-live":
        # Operator-assisted browser/session fetch only; excluded from --fetch all.
        if not args.acquisition_db:
            raise SystemExit("--acquisition-db is required for investing-live fetches")
        investing_report = run_investing_live_fetch(
            out_dir=out_dir,
            start=start,
            end=end,
            acquisition_db_path=Path(args.acquisition_db),
            artifact_store_root=acquisition_artifact_store_root,
            earnings_loaded_page_path=Path(args.investing_earnings_loaded_page) if args.investing_earnings_loaded_page else None,
            earnings_browser_capture=args.investing_earnings_browser_capture,
            earnings_browser_user_data_dir=Path(args.investing_browser_user_data_dir) if args.investing_browser_user_data_dir else None,
            earnings_browser_executable=Path(args.investing_browser_executable) if args.investing_browser_executable else None,
            earnings_browser_headless=args.investing_browser_headless,
            earnings_browser_timeout_ms=args.investing_browser_timeout_ms,
        )
        report_paths.append(investing_report)
        print(str(investing_report))

    if args.fetch == "daily-ohlcv-local-sqlite":
        # Operator-assisted local materialization/import only; excluded from --fetch all.
        if not args.daily_ohlcv_dir:
            raise SystemExit("--daily-ohlcv-dir is required for daily-ohlcv-local-sqlite fetches")
        if not args.acquisition_db:
            raise SystemExit("--acquisition-db is required for daily-ohlcv-local-sqlite fetches")
        ohlcv_import_report = run_local_daily_ohlcv_sqlite_import(
            out_dir=out_dir,
            source_dir=Path(args.daily_ohlcv_dir),
            acquisition_db_path=Path(args.acquisition_db),
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(ohlcv_import_report)
        print(str(ohlcv_import_report))

    if _should_fetch(args.fetch, "daily-ohlcv-constituents-alpaca"):
        if not args.acquisition_db:
            raise SystemExit("--acquisition-db is required for daily-ohlcv-constituents-alpaca fetches")
        pit_parquet_path = (
            Path(args.pit_parquet)
            if args.pit_parquet
            else out_dir / "pit_constituents" / "sp500_ticker_intervals.parquet"
        )
        ohlcv_constituent_report = run_alpaca_constituent_daily_ohlcv_fetch(
            out_dir=out_dir,
            pit_parquet_path=pit_parquet_path,
            start=start,
            end=end,
            adjustment=args.adjustment,
            alpaca_feed=args.alpaca_feed,
            acquisition_db_path=Path(args.acquisition_db),
            artifact_store_root=acquisition_artifact_store_root,
            allow_missing_symbols=args.allow_missing_constituent_symbols,
            fixed_universe_symbols=_load_json_symbol_list(Path(args.universe_json)) if args.universe_json else None,
            fixed_universe_dir=Path(args.constituent_universe_dir) if args.constituent_universe_dir else None,
            allow_pit_universe=args.allow_pit_constituent_universe,
            expected_universe_count=args.constituent_universe_expected_count,
            verbose=args.verbose,
        )
        report_paths.append(ohlcv_constituent_report)
        print(str(ohlcv_constituent_report))
    if args.emit_manifest:
        required_for = [item.strip() for item in args.manifest_required_for.split(",") if item.strip()]
        manifest = emit_manifest_for_report_paths(
            report_paths=report_paths,
            out_dir=out_dir,
            artifact_store_root=args.artifact_store,
            manifest_path=Path(args.emit_manifest),
            artifact_set=args.manifest_artifact_set or f"regime_engine_{end.isoformat()}",
            required_for=required_for,
            repo_root=REPO_ROOT,
        )
        print(str(Path(args.emit_manifest)))
        print(f"manifest_artifacts={len(manifest.artifacts)}")
    return 0


def _resolve_stock_universe(args: argparse.Namespace, *, out_dir: Path) -> list[str]:
    if args.universe_json:
        return _load_json_symbol_list(Path(args.universe_json))
    if args.constituent_universe_dir:
        return load_symbols_from_daily_ohlcv_tree(Path(args.constituent_universe_dir))
    pit_parquet = (
        Path(args.pit_parquet)
        if args.pit_parquet
        else out_dir / "pit_constituents" / "sp500_ticker_intervals.parquet"
    )
    return load_symbols_from_pit_constituents_parquet(pit_parquet)


def _should_fetch(selected: str, mode: str) -> bool:
    return selected == mode or (selected == "all" and mode in UNATTENDED_FETCH_MODES)


def _validate_fetch_modes() -> None:
    overlap = UNATTENDED_FETCH_MODES & OPERATOR_ASSISTED_FETCH_MODES
    if overlap:
        raise RuntimeError(f"Fetch mode sets overlap: {sorted(overlap)}")


def _load_json_symbol_list(universe_path: Path) -> list[str]:
    stocks = json.loads(universe_path.read_text())
    if not isinstance(stocks, list) or not all(isinstance(symbol, str) for symbol in stocks):
        raise SystemExit("--universe-json must be a JSON list[str]")
    return stocks


if __name__ == "__main__":
    raise SystemExit(main())
