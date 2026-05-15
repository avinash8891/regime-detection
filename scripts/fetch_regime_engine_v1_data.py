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
from regime_data_fetch.local_daily_ohlcv_sqlite import run_local_daily_ohlcv_sqlite_import
from regime_data_fetch.local_usd_index import run_local_usd_index_import
from regime_data_fetch.pmi import DEFAULT_MANUAL_PMI_HISTORY_DIR, run_pmi_fetch
from regime_data_fetch.pit_constituents import run_pit_constituents_fetch
from regime_data_fetch.powell_speeches import run_powell_speeches_fetch
from regime_data_fetch.sf_fed_news_sentiment import run_sf_fed_news_sentiment_fetch


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch regime-engine raw market and macro data for V1/V2 scopes.")
    ap.add_argument("--out-dir", default="data/raw", help="Output directory for Parquet + reports.")
    ap.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD).")
    ap.add_argument("--end", default=dt.date.today().isoformat(), help="End date (YYYY-MM-DD).")
    ap.add_argument("--scope", default="v1", help="Data scope: v1|v2|all.")
    ap.add_argument("--fetch", default="market", help="What to fetch: market|macro|events|pmi|pit|fomc|powell|eps|eps-spglobal-auto|eps-wayback|usd-index-local|daily-ohlcv-local-sqlite|sentiment|investing-archive-local|investing-live|cleveland-fed-nowcast|sf-fed-news-sentiment|all.")
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
        help="Path to a JSON list[str] of symbols to fetch. Required for V1/all stock-universe market fetches.",
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
    ap.add_argument("--daily-ohlcv-dir", default=None, help="Path to a local partitioned daily_ohlcv parquet directory. Required for --fetch daily-ohlcv-local-sqlite.")
    ap.add_argument("--investing-archive-root", default=None, help="Path to archived Investing.com source_pages root. Required for --fetch investing-archive-local.")
    ap.add_argument("--investing-earnings-loaded-page", default=None, help="Path to a browser-loaded Investing.com earnings calendar HTML page containing __NEXT_DATA__. Optional for --fetch investing-live.")
    ap.add_argument("--investing-earnings-browser-capture", action=argparse.BooleanOptionalAction, default=True, help="For --fetch investing-live, capture a fresh Investing.com earnings page with Playwright when no page/token is supplied. Default True.")
    ap.add_argument("--investing-browser-user-data-dir", default=None, help="Persistent browser profile directory for Investing.com browser capture. Defaults to archive-local browser_profile or INVESTING_BROWSER_USER_DATA_DIR.")
    ap.add_argument("--investing-browser-executable", default=None, help="Chrome/Chromium executable for Investing.com browser capture. Defaults to Playwright browser or INVESTING_BROWSER_EXECUTABLE.")
    ap.add_argument("--investing-browser-headless", action=argparse.BooleanOptionalAction, default=None, help="Run Investing.com browser capture headless/headful. Defaults to INVESTING_BROWSER_HEADLESS or headful.")
    ap.add_argument("--investing-browser-timeout-ms", type=int, default=None, help="Timeout in milliseconds while waiting for Investing.com accessToken. Defaults to INVESTING_BROWSER_TIMEOUT_MS or 120000.")
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
        default="profile_engine_30d,v2_calibration",
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
    if args.fetch not in {"market", "macro", "events", "pmi", "pit", "fomc", "powell", "eps", "eps-spglobal-auto", "eps-wayback", "usd-index-local", "daily-ohlcv-local-sqlite", "sentiment", "investing-archive-local", "investing-live", "cleveland-fed-nowcast", "sf-fed-news-sentiment", "all"}:
        raise SystemExit("--fetch must be market|macro|events|pmi|pit|fomc|powell|eps|eps-spglobal-auto|eps-wayback|usd-index-local|daily-ohlcv-local-sqlite|sentiment|investing-archive-local|investing-live|cleveland-fed-nowcast|sf-fed-news-sentiment|all")
    if args.emit_manifest and not args.artifact_store:
        raise SystemExit("--artifact-store is required when --emit-manifest is set")

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
                        "V1/all stock-universe fetches require an explicit --universe-json symbol list. "
                        "V2 scope adds the fixed ETF/cross-asset universe in code."
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
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(market_report)
        print(str(market_report))

    if args.fetch in {"macro", "all"}:
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

    if args.fetch in {"sentiment", "all"}:
        sentiment_report = run_sentiment_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(sentiment_report)
        print(str(sentiment_report))

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
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
            bls_start_year=args.bls_start_year,
            bls_end_year=args.bls_end_year,
            include_v2_curated_candidates=args.include_v2_curated_event_candidates,
            as_of_date=end,
        )
        report_paths.append(event_report)
        print(str(event_report))

    if args.fetch in {"pmi", "all"}:
        pmi_report = run_pmi_fetch(
            out_dir=out_dir,
            as_of_date=end,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
            manual_history_dir=DEFAULT_MANUAL_PMI_HISTORY_DIR,
        )
        report_paths.append(pmi_report)
        print(str(pmi_report))

    if args.fetch in {"pit", "all"}:
        pit_report = run_pit_constituents_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(pit_report)
        print(str(pit_report))

    if args.fetch in {"fomc", "all"}:
        fomc_report = run_fomc_minutes_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(fomc_report)
        print(str(fomc_report))

    if args.fetch in {"powell", "all"}:
        powell_report = run_powell_speeches_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(powell_report)
        print(str(powell_report))

    if args.fetch in {"cleveland-fed-nowcast", "all"}:
        nowcast_report = run_cleveland_fed_nowcast_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(nowcast_report)
        print(str(nowcast_report))

    if args.fetch in {"sf-fed-news-sentiment", "all"}:
        news_sentiment_report = run_sf_fed_news_sentiment_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(news_sentiment_report)
        print(str(news_sentiment_report))

    if args.fetch == "eps":
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
        # Opt-in only; intentionally excluded from --fetch all because the
        # spdji URL is Akamai-protected and returns HTTP 403 to programmatic
        # clients. Including it in `all` would abort the full fetch midway in
        # any environment without a pre-staged manual-drop file. Cadence is
        # WEEKLY (S&P publishes weekly); see
        # regime_data_fetch.aggregate_eps.download_spglobal_eps_workbook
        # docstring for the 4-week revision-direction rationale.
        eps_auto_report = run_aggregate_eps_auto_fetch(
            out_dir=out_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(eps_auto_report)
        print(str(eps_auto_report))

    if args.fetch == "eps-wayback":
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


def _resolve_stock_universe(args: argparse.Namespace) -> list[str]:
    if not args.universe_json:
        raise SystemExit("--universe-json is required for V1/all stock-universe fetches")
    universe_path = Path(args.universe_json)
    stocks = json.loads(universe_path.read_text())
    if not isinstance(stocks, list) or not all(isinstance(symbol, str) for symbol in stocks):
        raise SystemExit("--universe-json must be a JSON list[str]")
    return stocks


if __name__ == "__main__":
    raise SystemExit(main())
