#!/usr/bin/env python3
# ruff: noqa: E402
# pyright: reportUnknownVariableType=false, reportUnusedFunction=false
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import datetime as dt
import json
import os
from pathlib import Path

import sys
from collections.abc import Callable
from typing import Any, Literal, cast

# Allow running as a script without requiring PYTHONPATH/installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from scripts._fetch_regime_engine_v1_args import build_fetch_arg_parser
except ModuleNotFoundError:
    from _fetch_regime_engine_v1_args import build_fetch_arg_parser
from regime_data_fetch.cli_common import (
    OPERATOR_ENV_POINTER_FILE,
    load_env_file,
    load_operator_env_files,
    parse_date,
)
from regime_data_fetch.bls_schedule import (
    build_bls_local_archive_page_fetcher,
)  # pyright: ignore[reportUnknownVariableType]
from regime_data_fetch.event_calendar import (
    run_us_event_calendar_fetch,
)  # pyright: ignore[reportUnknownVariableType]
from regime_data_fetch.artifact_export import emit_manifest_for_report_paths
from regime_data_fetch.aaii_sentiment import run_sentiment_fetch
from regime_data_fetch.fetch_workflow import run_macro_fetch, run_market_fetch
from regime_data_fetch.aggregate_eps import (
    run_aggregate_eps_fetch,
    run_aggregate_eps_auto_fetch,  # pyright: ignore[reportUnknownVariableType]
    run_wayback_aggregate_eps_fetch,  # pyright: ignore[reportUnknownVariableType]
)
from regime_data_fetch.fomc_minutes import (
    run_fomc_minutes_fetch,
)  # pyright: ignore[reportUnknownVariableType]
from regime_data_fetch.investing_archive import run_local_investing_archive_import
from regime_data_fetch.investing_live import run_investing_live_fetch
from regime_data_fetch.cleveland_fed_nowcast import run_cleveland_fed_nowcast_fetch
from regime_data_fetch.local_daily_ohlcv_sqlite import (
    run_alpaca_constituent_daily_ohlcv_fetch,
    run_local_daily_ohlcv_sqlite_import,
)
from regime_data_fetch.local_usd_index import run_local_usd_index_import
from regime_data_fetch.pmi import (
    run_pmi_fetch,
)  # pyright: ignore[reportUnknownVariableType]
from regime_data_fetch.pit_constituents import (
    run_pit_constituents_fetch,
)  # pyright: ignore[reportUnknownVariableType]
from regime_data_fetch.powell_speeches import (
    run_powell_speeches_fetch,
)  # pyright: ignore[reportUnknownVariableType]
from regime_data_fetch.sf_fed_news_sentiment import run_sf_fed_news_sentiment_fetch
from regime_data_fetch.universe import (
    FIXED_UNIVERSE_SYMBOL_COUNT,
    FIXED_UNIVERSE_TREE_NAME,
    load_symbols_from_daily_ohlcv_tree,
    load_symbols_from_pit_constituents_parquet,
)

build_bls_local_archive_page_fetcher = cast(Any, build_bls_local_archive_page_fetcher)
run_us_event_calendar_fetch = cast(Any, run_us_event_calendar_fetch)
run_aggregate_eps_auto_fetch = cast(Any, run_aggregate_eps_auto_fetch)
run_wayback_aggregate_eps_fetch = cast(Any, run_wayback_aggregate_eps_fetch)
run_fomc_minutes_fetch = cast(Any, run_fomc_minutes_fetch)
run_pmi_fetch = cast(Any, run_pmi_fetch)
run_pit_constituents_fetch = cast(Any, run_pit_constituents_fetch)
run_powell_speeches_fetch = cast(Any, run_powell_speeches_fetch)

FetchModeCategory = Literal["unattended", "operator-assisted"]
UNATTENDED: FetchModeCategory = "unattended"
OPERATOR_ASSISTED: FetchModeCategory = "operator-assisted"


@dataclass(frozen=True)
class FetchModeInvocation:
    args: argparse.Namespace
    out_dir: Path
    start: dt.date
    end: dt.date
    acquisition_db_path: Path | None
    acquisition_artifact_store_root: str | None


@dataclass(frozen=True)
class FetchModeSpec:
    name: str
    category: FetchModeCategory
    conservative_concurrent: bool = False
    invoke: Callable[[FetchModeInvocation], Path] | None = None


@dataclass(frozen=True)
class FetchExecutionGroup:
    modes: tuple[str, ...]
    concurrent: bool = False


def _invoke_market_fetch(context: FetchModeInvocation) -> Path:
    args = context.args
    stocks = (
        _resolve_stock_universe(args, out_dir=context.out_dir)
        if args.scope in {"v1", "all"}
        else []
    )
    return run_market_fetch(
        out_dir=context.out_dir,
        scope=args.scope,
        stock_symbols=stocks,
        start=context.start,
        end=context.end,
        adjustment=args.adjustment,
        alpaca_feed=args.alpaca_feed,
        daily_bars_provider=args.daily_bars_provider,
        vix_symbol=args.vix_symbol,
        allow_vix_proxy=args.allow_vix_proxy,
        verbose=args.verbose,
        acquisition_db_path=context.acquisition_db_path,
        artifact_store_root=context.acquisition_artifact_store_root,
    )


def _invoke_macro_fetch(context: FetchModeInvocation) -> Path:
    args = context.args
    return run_macro_fetch(
        out_dir=context.out_dir,
        start=context.start,
        end=context.end,
        fred_api_key=args.fred_api_key,
        include_cpi_vintages=args.include_cpi_vintages,
        acquisition_db_path=context.acquisition_db_path,
        artifact_store_root=context.acquisition_artifact_store_root,
    )


def _invoke_sentiment_fetch(context: FetchModeInvocation) -> Path:
    return run_sentiment_fetch(
        out_dir=context.out_dir,
        acquisition_db_path=context.acquisition_db_path,
        artifact_store_root=context.acquisition_artifact_store_root,
        required=context.args.fetch == "sentiment",
    )


def _invoke_event_fetch(context: FetchModeInvocation) -> Path:
    args = context.args
    bls_page_fetcher = None
    if args.bls_schedule_dir:
        bls_page_fetcher = build_bls_local_archive_page_fetcher(
            schedule_dir=Path(args.bls_schedule_dir),
        )
    return run_us_event_calendar_fetch(
        repo_root=REPO_ROOT,
        fred_api_key=args.fred_api_key or None,
        bls_page_fetcher=bls_page_fetcher,
        acquisition_db_path=context.acquisition_db_path,
        artifact_store_root=context.acquisition_artifact_store_root,
        bls_start_year=args.bls_start_year,
        bls_end_year=args.bls_end_year,
        include_v2_curated_candidates=args.include_layer_event_candidates,
        as_of_date=context.end,
    )


def _invoke_pmi_fetch(context: FetchModeInvocation) -> Path:
    args = context.args
    pmi_history_dir = args.pmi_history_dir or os.environ.get("REGIME_PMI_HISTORY_DIR")
    return run_pmi_fetch(
        out_dir=context.out_dir,
        as_of_date=context.end,
        acquisition_db_path=context.acquisition_db_path,
        artifact_store_root=context.acquisition_artifact_store_root,
        manual_history_dir=Path(pmi_history_dir) if pmi_history_dir else None,
    )


def _invoke_pit_fetch(context: FetchModeInvocation) -> Path:
    return run_pit_constituents_fetch(
        out_dir=context.out_dir,
        acquisition_db_path=context.acquisition_db_path,
        artifact_store_root=context.acquisition_artifact_store_root,
    )


def _invoke_fomc_fetch(context: FetchModeInvocation) -> Path:
    return run_fomc_minutes_fetch(
        out_dir=context.out_dir,
        acquisition_db_path=context.acquisition_db_path,
        artifact_store_root=context.acquisition_artifact_store_root,
    )


def _invoke_powell_fetch(context: FetchModeInvocation) -> Path:
    return run_powell_speeches_fetch(
        out_dir=context.out_dir,
        acquisition_db_path=context.acquisition_db_path,
        artifact_store_root=context.acquisition_artifact_store_root,
    )


def _invoke_cleveland_fed_nowcast_fetch(context: FetchModeInvocation) -> Path:
    return run_cleveland_fed_nowcast_fetch(
        out_dir=context.out_dir,
        acquisition_db_path=context.acquisition_db_path,
        artifact_store_root=context.acquisition_artifact_store_root,
    )


def _invoke_sf_fed_news_sentiment_fetch(context: FetchModeInvocation) -> Path:
    return run_sf_fed_news_sentiment_fetch(
        out_dir=context.out_dir,
        acquisition_db_path=context.acquisition_db_path,
        artifact_store_root=context.acquisition_artifact_store_root,
    )


def _invoke_constituent_daily_ohlcv_fetch(context: FetchModeInvocation) -> Path:
    args = context.args
    if not context.acquisition_db_path:
        raise SystemExit(
            "--acquisition-db is required for daily-ohlcv-constituents-alpaca fetches"
        )
    pit_parquet_path = (
        Path(args.pit_parquet)
        if args.pit_parquet
        else context.out_dir / "pit_constituents" / "sp500_ticker_intervals.parquet"
    )
    return run_alpaca_constituent_daily_ohlcv_fetch(
        out_dir=context.out_dir,
        pit_parquet_path=pit_parquet_path,
        start=context.start,
        end=context.end,
        adjustment=args.adjustment,
        alpaca_feed=args.alpaca_feed,
        daily_bars_provider=args.daily_bars_provider,
        acquisition_db_path=context.acquisition_db_path,
        artifact_store_root=context.acquisition_artifact_store_root,
        allow_missing_symbols=args.allow_missing_constituent_symbols,
        fixed_universe_symbols=(
            _load_json_symbol_list(Path(args.universe_json))
            if args.universe_json
            else None
        ),
        fixed_universe_dir=(
            Path(args.constituent_universe_dir)
            if args.constituent_universe_dir
            else None
        ),
        allow_pit_universe=args.allow_pit_constituent_universe,
        expected_universe_count=args.constituent_universe_expected_count,
        verbose=args.verbose,
    )


FETCH_MODE_REGISTRY = {
    spec.name: spec
    for spec in (
        FetchModeSpec("market", UNATTENDED, invoke=_invoke_market_fetch),
        FetchModeSpec(
            "macro",
            UNATTENDED,
            conservative_concurrent=True,
            invoke=_invoke_macro_fetch,
        ),
        FetchModeSpec(
            "sentiment",
            UNATTENDED,
            conservative_concurrent=True,
            invoke=_invoke_sentiment_fetch,
        ),
        FetchModeSpec(
            "events",
            UNATTENDED,
            conservative_concurrent=True,
            invoke=_invoke_event_fetch,
        ),
        FetchModeSpec(
            "pmi", UNATTENDED, conservative_concurrent=True, invoke=_invoke_pmi_fetch
        ),
        FetchModeSpec(
            "pit", UNATTENDED, conservative_concurrent=True, invoke=_invoke_pit_fetch
        ),
        FetchModeSpec(
            "fomc",
            UNATTENDED,
            conservative_concurrent=True,
            invoke=_invoke_fomc_fetch,
        ),
        FetchModeSpec(
            "powell",
            UNATTENDED,
            conservative_concurrent=True,
            invoke=_invoke_powell_fetch,
        ),
        FetchModeSpec(
            "cleveland-fed-nowcast",
            UNATTENDED,
            conservative_concurrent=True,
            invoke=_invoke_cleveland_fed_nowcast_fetch,
        ),
        FetchModeSpec(
            "sf-fed-news-sentiment",
            UNATTENDED,
            conservative_concurrent=True,
            invoke=_invoke_sf_fed_news_sentiment_fetch,
        ),
        FetchModeSpec(
            "daily-ohlcv-constituents-alpaca",
            UNATTENDED,
            invoke=_invoke_constituent_daily_ohlcv_fetch,
        ),
        FetchModeSpec("eps", OPERATOR_ASSISTED),
        FetchModeSpec("eps-spglobal-auto", OPERATOR_ASSISTED),
        FetchModeSpec("eps-wayback", OPERATOR_ASSISTED),
        FetchModeSpec("usd-index-local", OPERATOR_ASSISTED),
        FetchModeSpec("daily-ohlcv-local-sqlite", OPERATOR_ASSISTED),
        FetchModeSpec("investing-archive-local", OPERATOR_ASSISTED),
        FetchModeSpec("investing-live", OPERATOR_ASSISTED),
    )
}
UNATTENDED_FETCH_MODES = frozenset(
    name for name, spec in FETCH_MODE_REGISTRY.items() if spec.category == UNATTENDED
)
OPERATOR_ASSISTED_FETCH_MODES = frozenset(
    name
    for name, spec in FETCH_MODE_REGISTRY.items()
    if spec.category == OPERATOR_ASSISTED
)
FETCH_MODES = frozenset(FETCH_MODE_REGISTRY) | {"all"}
AUTO_EMIT_MANIFEST = "__auto_emit_manifest__"
MANIFEST_LOCKFILE_ROOT = REPO_ROOT / "manifests"
RUN_MANIFEST_DIR = MANIFEST_LOCKFILE_ROOT / "runs"


def main() -> int:
    args = build_fetch_arg_parser(
        fetch_modes=FETCH_MODES,
        operator_env_pointer_file=OPERATOR_ENV_POINTER_FILE,
        fixed_universe_symbol_count=FIXED_UNIVERSE_SYMBOL_COUNT,
        fixed_universe_tree_name=FIXED_UNIVERSE_TREE_NAME,
        auto_emit_manifest=AUTO_EMIT_MANIFEST,
    ).parse_args()

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
    emit_manifest_path = _resolve_emit_manifest_path(args.emit_manifest, end=end)
    if emit_manifest_path is not None and not args.artifact_store:
        raise SystemExit("--artifact-store is required when --emit-manifest is set")
    if emit_manifest_path is not None:
        _validate_manifest_artifact_store(
            manifest_path=emit_manifest_path,
            artifact_store_root=args.artifact_store,
        )

    if args.env_file:
        load_env_file(Path(args.env_file))
    load_operator_env_files(
        repo_root=REPO_ROOT,
        explicit_path=Path(args.operator_env_file) if args.operator_env_file else None,
    )

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
    acquisition_artifact_store_root = (
        args.artifact_store if acquisition_db_path and args.artifact_store else None
    )

    for group in _plan_fetch_mode_execution(
        args.fetch,
        conservative_concurrency=args.conservative_concurrent_fetches,
    ):
        if group.concurrent:
            max_workers = min(len(group.modes), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:

                def _run_mode(mode: str) -> Path:
                    return _invoke_unattended_fetch_mode(
                        mode,
                        args=args,
                        out_dir=out_dir,
                        start=start,
                        end=end,
                        acquisition_db_path=acquisition_db_path,
                        acquisition_artifact_store_root=acquisition_artifact_store_root,
                    )

                reports = list(executor.map(_run_mode, group.modes))
        else:
            reports = [
                _invoke_unattended_fetch_mode(
                    mode,
                    args=args,
                    out_dir=out_dir,
                    start=start,
                    end=end,
                    acquisition_db_path=acquisition_db_path,
                    acquisition_artifact_store_root=acquisition_artifact_store_root,
                )
                for mode in group.modes
            ]
        report_paths.extend(reports)
        for report in reports:
            print(str(report))

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
            browser_user_data_dir=(
                Path(args.eps_browser_user_data_dir)
                if args.eps_browser_user_data_dir
                else None
            ),
            browser_executable=(
                Path(args.eps_browser_executable)
                if args.eps_browser_executable
                else None
            ),
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
            from_date=(
                parse_date(args.eps_wayback_from) if args.eps_wayback_from else None
            ),
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
            raise SystemExit(
                "--investing-archive-root is required for investing-archive-local fetches"
            )
        if not args.acquisition_db:
            raise SystemExit(
                "--acquisition-db is required for investing-archive-local fetches"
            )
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
            earnings_loaded_page_path=(
                Path(args.investing_earnings_loaded_page)
                if args.investing_earnings_loaded_page
                else None
            ),
            earnings_browser_capture=args.investing_earnings_browser_capture,
            earnings_browser_user_data_dir=(
                Path(args.investing_browser_user_data_dir)
                if args.investing_browser_user_data_dir
                else None
            ),
            earnings_browser_executable=(
                Path(args.investing_browser_executable)
                if args.investing_browser_executable
                else None
            ),
            earnings_browser_headless=args.investing_browser_headless,
            earnings_browser_timeout_ms=args.investing_browser_timeout_ms,
        )
        report_paths.append(investing_report)
        print(str(investing_report))

    if args.fetch == "daily-ohlcv-local-sqlite":
        # Operator-assisted local materialization/import only; excluded from --fetch all.
        if not args.daily_ohlcv_dir:
            raise SystemExit(
                "--daily-ohlcv-dir is required for daily-ohlcv-local-sqlite fetches"
            )
        if not args.acquisition_db:
            raise SystemExit(
                "--acquisition-db is required for daily-ohlcv-local-sqlite fetches"
            )
        ohlcv_import_report = run_local_daily_ohlcv_sqlite_import(
            out_dir=out_dir,
            source_dir=Path(args.daily_ohlcv_dir),
            acquisition_db_path=Path(args.acquisition_db),
            artifact_store_root=acquisition_artifact_store_root,
        )
        report_paths.append(ohlcv_import_report)
        print(str(ohlcv_import_report))

    if emit_manifest_path is not None:
        required_for = [
            item.strip()
            for item in args.manifest_required_for.split(",")
            if item.strip()
        ]
        manifest = emit_manifest_for_report_paths(
            report_paths=report_paths,
            out_dir=out_dir,
            artifact_store_root=args.artifact_store,
            manifest_path=emit_manifest_path,
            artifact_set=args.manifest_artifact_set
            or f"regime_engine_{end.isoformat()}",
            required_for=required_for,
            repo_root=REPO_ROOT,
        )
        print(str(emit_manifest_path))
        print(f"manifest_artifacts={len(manifest.artifacts)}")
    return 0


def _resolve_emit_manifest_path(value: str | None, *, end: dt.date) -> Path | None:
    if value is None:
        return None
    path = (
        _default_run_manifest_path(end) if value == AUTO_EMIT_MANIFEST else Path(value)
    )
    if not path.is_absolute():
        path = REPO_ROOT / path
    _validate_manifest_lockfile_path(path)
    return path


def _default_run_manifest_path(end: dt.date) -> Path:
    return RUN_MANIFEST_DIR / f"regime_engine_{end.isoformat()}.yaml"


def _validate_manifest_lockfile_path(path: Path) -> None:
    repo_root = REPO_ROOT.resolve()
    resolved = path if path.is_absolute() else (repo_root / path)
    try:
        relative = resolved.resolve(strict=False).relative_to(repo_root)
    except ValueError:
        return
    if relative.parts[:1] == ("data",):
        raise SystemExit(
            "manifest lockfiles must be written outside ignored data/; use manifests/runs/<name>.yaml"
        )
    if relative.parts[:1] == (".context",):
        raise SystemExit(
            "manifest lockfiles must be written outside ignored .context/; use manifests/runs/<name>.yaml"
        )


def _validate_manifest_artifact_store(
    *, manifest_path: Path, artifact_store_root: str
) -> None:
    if "://" in artifact_store_root:
        return
    repo_root = REPO_ROOT.resolve()
    try:
        manifest_relative = manifest_path.resolve(strict=False).relative_to(repo_root)
    except ValueError:
        return
    if manifest_relative.parts[:1] != ("manifests",):
        return
    store_path = Path(artifact_store_root)
    if not store_path.is_absolute():
        store_path = repo_root / store_path
    try:
        store_relative = store_path.resolve(strict=False).relative_to(repo_root)
    except ValueError:
        return
    if store_relative.parts[:1] == (".context",):
        raise SystemExit(
            "tracked manifests require durable artifact storage; .context artifact stores are local scratch only"
        )


def _plan_fetch_mode_execution(
    selected: str,
    *,
    conservative_concurrency: bool,
) -> list[FetchExecutionGroup]:
    if selected != "all":
        if selected in UNATTENDED_FETCH_MODES:
            return [FetchExecutionGroup((selected,))]
        return []
    modes = tuple(
        name
        for name, spec in FETCH_MODE_REGISTRY.items()
        if spec.category == UNATTENDED
    )
    if not conservative_concurrency:
        return [FetchExecutionGroup((mode,)) for mode in modes]

    groups: list[FetchExecutionGroup] = []
    concurrent_batch: list[str] = []
    for mode in modes:
        spec = FETCH_MODE_REGISTRY[mode]
        if spec.conservative_concurrent:
            concurrent_batch.append(mode)
            continue
        if concurrent_batch:
            groups.append(FetchExecutionGroup(tuple(concurrent_batch), concurrent=True))
            concurrent_batch = []
        groups.append(FetchExecutionGroup((mode,)))
    if concurrent_batch:
        groups.append(FetchExecutionGroup(tuple(concurrent_batch), concurrent=True))
    return groups


def _invoke_unattended_fetch_mode(
    mode: str,
    *,
    args: argparse.Namespace,
    out_dir: Path,
    start: dt.date,
    end: dt.date,
    acquisition_db_path: Path | None,
    acquisition_artifact_store_root: str | None,
) -> Path:
    spec = FETCH_MODE_REGISTRY.get(mode)
    if spec is None or spec.category != UNATTENDED or spec.invoke is None:
        raise RuntimeError(f"Unsupported unattended fetch mode: {mode}")
    return spec.invoke(
        FetchModeInvocation(
            args=args,
            out_dir=out_dir,
            start=start,
            end=end,
            acquisition_db_path=acquisition_db_path,
            acquisition_artifact_store_root=acquisition_artifact_store_root,
        )
    )


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


def _should_fetch(
    selected: str, mode: str
) -> bool:  # pyright: ignore[reportUnusedFunction]
    return selected == mode or any(
        mode in group.modes
        for group in _plan_fetch_mode_execution(
            selected, conservative_concurrency=False
        )
    )


def _validate_fetch_modes() -> None:
    overlap = UNATTENDED_FETCH_MODES & OPERATOR_ASSISTED_FETCH_MODES
    if overlap:
        raise RuntimeError(f"Fetch mode sets overlap: {sorted(overlap)}")


def _load_json_symbol_list(universe_path: Path) -> list[str]:
    stocks = cast(list[Any], json.loads(universe_path.read_text()))
    if not all(isinstance(symbol, str) for symbol in stocks):
        raise SystemExit("--universe-json must be a JSON list[str]")
    return [str(symbol) for symbol in stocks]


if __name__ == "__main__":
    raise SystemExit(main())
