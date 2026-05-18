from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Collection


def build_fetch_arg_parser(
    *,
    fetch_modes: Collection[str],
    operator_env_pointer_file: str,
    fixed_universe_symbol_count: int,
    fixed_universe_tree_name: str,
    auto_emit_manifest: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch regime-engine raw market and macro data for V1/V2 scopes. "
            "--fetch all is reserved for unattended autonomous refreshes; browser-session, "
            "local-file, and historical-backfill tools remain explicit operator-assisted fetches."
        )
    )
    parser.add_argument(
        "--out-dir", default="data/raw", help="Output directory for Parquet + reports."
    )
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument(
        "--end", default=dt.date.today().isoformat(), help="End date (YYYY-MM-DD)."
    )
    parser.add_argument("--scope", default="v1", help="Data scope: v1|v2|all.")
    parser.add_argument(
        "--fetch",
        default="market",
        help=f"What to fetch: {'|'.join(sorted(fetch_modes))}.",
    )
    parser.add_argument(
        "--min-cap-b", type=float, default=10.0, help="Universe filter threshold in $B."
    )
    parser.add_argument(
        "--adjustment", default="raw", help="Alpaca adjustment: raw|split|dividend|all."
    )
    parser.add_argument(
        "--alpaca-feed",
        default=None,
        help="Alpaca data feed: sip|iex|otc. Omit to use SDK default.",
    )
    parser.add_argument(
        "--fred-api-key", default=None, help="Optional FRED API key for macro fetches."
    )
    parser.add_argument(
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
    parser.add_argument(
        "--list-symbols", action="store_true", help="Only print symbol counts and exit."
    )
    parser.add_argument(
        "--env-file", default=None, help="Optional direct .env file to load."
    )
    parser.add_argument(
        "--operator-env-file",
        default=None,
        help=(
            "Optional non-secret pointer file listing repo credential env files. "
            f"Defaults to {operator_env_pointer_file} or ~/.config/regime-detection/operator.env."
        ),
    )
    parser.add_argument(
        "--universe-json",
        default=None,
        help=(
            "Optional JSON list[str] of symbols to fetch. For constituent OHLCV, "
            f"this should be the fixed {fixed_universe_symbol_count}-symbol universe."
        ),
    )
    parser.add_argument(
        "--vix-symbol",
        default="VIX",
        help=(
            "Volatility proxy symbol to fetch from Alpaca. Use VIX when your account "
            "supports it, or VIXY as the documented proxy."
        ),
    )
    parser.add_argument(
        "--allow-vix-proxy",
        action="store_true",
        help="Allow proceeding when true VIX is unavailable, using --vix-symbol (e.g. VIXY).",
    )
    parser.add_argument(
        "--eps-workbook",
        default=None,
        help=(
            "Operator-assisted manual import: path to a downloaded S&P aggregate EPS "
            "workbook (.xlsx). Required for --fetch eps."
        ),
    )
    parser.add_argument(
        "--eps-wayback-max-snapshots",
        type=int,
        default=None,
        help="Operator-assisted backfill: optional maximum number of Wayback EPS snapshots to process.",
    )
    parser.add_argument(
        "--eps-wayback-from",
        default=None,
        help="Optional lower bound date (YYYY-MM-DD) for Wayback EPS snapshot dates.",
    )
    parser.add_argument(
        "--eps-wayback-to",
        default=None,
        help="Optional upper bound date (YYYY-MM-DD) for Wayback EPS snapshot dates.",
    )
    parser.add_argument(
        "--eps-wayback-stop-after-first-success",
        action="store_true",
        help="Stop Wayback EPS processing after the first successfully parsed snapshot.",
    )
    parser.add_argument(
        "--eps-browser-user-data-dir",
        default=None,
        help=(
            "Operator-assisted EPS auto fetch: persistent browser profile directory for "
            "S&P browser fallback."
        ),
    )
    parser.add_argument(
        "--eps-browser-executable",
        default=None,
        help="Operator-assisted EPS auto fetch: Chrome/Chromium executable for S&P browser fallback.",
    )
    parser.add_argument(
        "--eps-browser-headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Operator-assisted EPS auto fetch: run browser fallback headless/headful. Default headless.",
    )
    parser.add_argument(
        "--eps-browser-timeout-ms",
        type=int,
        default=120000,
        help="Operator-assisted EPS auto fetch: timeout in milliseconds for S&P browser fallback.",
    )
    parser.add_argument(
        "--usd-index-csv",
        default=None,
        help=(
            "Optional/manual Yahoo Finance ^NYICDX historical CSV export for --fetch usd-index-local. "
            "Routine future USD ingestion uses FRED DTWEXBGS through --fetch macro."
        ),
    )
    parser.add_argument(
        "--daily-ohlcv-dir",
        default=None,
        help=(
            "Operator-assisted manual import: path to a local partitioned daily_ohlcv "
            "parquet directory. Required for --fetch daily-ohlcv-local-sqlite."
        ),
    )
    parser.add_argument(
        "--pit-parquet",
        default=None,
        help=(
            "PIT constituent parquet for --fetch daily-ohlcv-constituents-alpaca. "
            "Defaults to <out-dir>/pit_constituents/sp500_ticker_intervals.parquet."
        ),
    )
    parser.add_argument(
        "--constituent-universe-dir",
        default=None,
        help=(
            f"Fixed partitioned {fixed_universe_tree_name} tree to use as the Alpaca "
            "constituent refresh universe."
        ),
    )
    parser.add_argument(
        "--allow-pit-constituent-universe",
        action="store_true",
        help=(
            "Explicitly allow PIT-constituent parquet expansion as a bootstrap universe "
            "for Alpaca constituent OHLCV."
        ),
    )
    parser.add_argument(
        "--constituent-universe-expected-count",
        type=int,
        default=fixed_universe_symbol_count,
        help=(
            "Expected fixed constituent universe size for Alpaca refreshes. "
            f"Default {fixed_universe_symbol_count}."
        ),
    )
    parser.add_argument(
        "--allow-missing-constituent-symbols",
        action="store_true",
        help=(
            "Allow daily-ohlcv-constituents-alpaca to continue when Alpaca returns no bars "
            "for some PIT symbols."
        ),
    )
    parser.add_argument(
        "--pmi-history-dir",
        default=None,
        help="Optional manual Investing PMI history directory. Omit for live DBnomics/TradingEconomics PMI ingestion.",
    )
    parser.add_argument(
        "--investing-archive-root",
        default=None,
        help=(
            "Operator-assisted manual import: path to archived Investing.com source_pages "
            "root. Required for --fetch investing-archive-local."
        ),
    )
    parser.add_argument(
        "--investing-earnings-loaded-page",
        default=None,
        help=(
            "Operator-assisted Investing fetch: path to a browser-loaded earnings calendar "
            "HTML page containing __NEXT_DATA__. Optional for --fetch investing-live."
        ),
    )
    parser.add_argument(
        "--investing-earnings-browser-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Operator-assisted Investing fetch: capture a fresh earnings page with "
            "Playwright when no page/token is supplied. Default True."
        ),
    )
    parser.add_argument(
        "--investing-browser-user-data-dir",
        default=None,
        help=(
            "Operator-assisted Investing fetch: persistent browser profile directory. "
            "Defaults to archive-local browser_profile or INVESTING_BROWSER_USER_DATA_DIR."
        ),
    )
    parser.add_argument(
        "--investing-browser-executable",
        default=None,
        help=(
            "Operator-assisted Investing fetch: Chrome/Chromium executable. Defaults to "
            "Playwright browser or INVESTING_BROWSER_EXECUTABLE."
        ),
    )
    parser.add_argument(
        "--investing-browser-headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Operator-assisted Investing fetch: run browser capture headless/headful. "
            "Defaults to INVESTING_BROWSER_HEADLESS or headful."
        ),
    )
    parser.add_argument(
        "--investing-browser-timeout-ms",
        type=int,
        default=None,
        help=(
            "Operator-assisted Investing fetch: timeout in milliseconds while waiting for "
            "accessToken. Defaults to INVESTING_BROWSER_TIMEOUT_MS or 120000."
        ),
    )
    parser.add_argument(
        "--acquisition-db",
        default=None,
        help="Optional SQLite path for raw acquisition/provenance recording.",
    )
    parser.add_argument(
        "--artifact-store",
        default=None,
        help="Optional object-store root for emitted artifacts, e.g. s3://regime-data or /mnt/regime-data.",
    )
    parser.add_argument(
        "--emit-manifest",
        nargs="?",
        const=auto_emit_manifest,
        default=None,
        help=(
            "Optional manifest YAML path to write after fetch outputs are uploaded to "
            "--artifact-store. If supplied without a value, writes an immutable tracked "
            "lockfile under manifests/runs/."
        ),
    )
    parser.add_argument(
        "--manifest-artifact-set",
        default=None,
        help="Optional artifact_set name for --emit-manifest. Defaults to regime_engine_<end-date>.",
    )
    parser.add_argument(
        "--manifest-required-for",
        default="profile_engine,v2_calibration,historical_walkforward,audit_layer2_30d",
        help="Comma-separated use cases attached to emitted manifest artifacts.",
    )
    parser.add_argument(
        "--bls-schedule-dir",
        default=None,
        help="Optional local directory containing bls_schedule_YYYY.html files for BLS historical release schedules.",
    )
    parser.add_argument(
        "--bls-start-year",
        type=int,
        default=2000,
        help="Start year for BLS CPI/NFP schedule generation.",
    )
    parser.add_argument(
        "--bls-end-year",
        type=int,
        default=None,
        help="End year for BLS CPI/NFP schedule generation. Defaults to --end year.",
    )
    parser.add_argument(
        "--include-layer-event-candidates",
        "--include-v2-curated-event-candidates",
        dest="include_layer_event_candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include routine layer event candidates in event fetch output: elections, "
            "budget deadlines, and official global-rate calendars. Default True; "
            "use --no-include-layer-event-candidates for core FOMC/CPI/NFP debug runs."
        ),
    )
    parser.add_argument(
        "--conservative-concurrent-fetches",
        action="store_true",
        help=(
            "Opt in to parallel execution for registry-marked independent unattended modes "
            "under --fetch all. Default fetch execution remains serial."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print progress while fetching."
    )
    return parser
