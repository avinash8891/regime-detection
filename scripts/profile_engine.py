#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from regime_data_fetch.pit_constituents import read_pit_intervals
from regime_data_fetch.manifest_inputs import ManifestInputResolutionError
from regime_data_fetch.cli_common import (
    OPERATOR_ENV_POINTER_FILE,
    load_operator_env_files,
)
from regime_data_fetch.universe import FIXED_UNIVERSE_TREE_NAME
from regime_detection.engine import RegimeEngine
from regime_detection.fragility_universe import SECTOR_ETFS
from regime_detection.loaders import (
    load_central_bank_text_score,
    load_cpi_vintages_first_release,
    load_event_calendar,
    load_news_sentiment_series,
)
from regime_detection.market_context import build_market_context
from regime_detection.timeline import ENGINE_MINIMUM_HISTORY
from scripts._v2_calibration_helpers import (
    CROSS_ASSET_SYMBOLS,
    add_manifest_args,
    apply_manifest_input_defaults,
    apply_manifest_input_paths,
    default_pmi_path,
    load_close_dict,
    load_macro_series,
    load_market_data,
    manifest_input_overrides,
    materialize_manifest_from_args,
    positive_int,
    register_manifest_input_args,
)
from scripts.profile_engine_reporting import (
    PROFILE_INPUT_SEAM_NAMES,
    _build_json_report,
    _compact_timeline_rows,
    _format_stage_rows,
    _input_status,
    _profile_input_seam_values,
    _reporting_label,  # noqa: F401  re-exported; tests access via profile_engine._reporting_label
    _trailing_v2_status,
    _verify_invariants,
    _write_json_report,
)
from scripts.profile_engine_timers import (
    _timed_inflation_growth_builder,  # noqa: F401  re-exported; tests access via profile_engine.*
    install_timers as _install_timers,
)


_RUNNER_NAME = "profile_engine"

DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "src" / "regime_detection" / "configs" / "core3-v2.0.0.yaml"
)
DEFAULT_DAILY_DIR = REPO_ROOT / "data" / "raw" / FIXED_UNIVERSE_TREE_NAME
DEFAULT_CONSTITUENT_TREE = REPO_ROOT / "data" / "raw" / FIXED_UNIVERSE_TREE_NAME
DEFAULT_MACRO_PARQUET = (
    REPO_ROOT / "data" / "raw" / "macro" / "fred_macro_series.parquet"
)
DEFAULT_PIT_PARQUET = (
    REPO_ROOT / "data" / "raw" / "pit_constituents" / "sp500_ticker_intervals.parquet"
)
DEFAULT_PMI_PATH = default_pmi_path(REPO_ROOT / "data" / "raw")
DEFAULT_EVENT_CALENDAR = REPO_ROOT / "configs" / "events" / "us_events.yaml"
DEFAULT_RUN_TIMEOUT_SECONDS = 300

@dataclass(init=False)
class StageTimer:
    totals: defaultdict[str, float]
    counts: defaultdict[str, int]
    timeout_seconds: int
    current_stage: str | None

    def __init__(
        self, *, timeout_seconds: int = DEFAULT_RUN_TIMEOUT_SECONDS
    ) -> None:
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.timeout_seconds = timeout_seconds
        self.current_stage = None

    @contextlib.contextmanager
    def measure(self, stage_name: str):
        start = time.perf_counter()
        previous_stage = self.current_stage
        self.current_stage = stage_name
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.totals[stage_name] += elapsed
            self.counts[stage_name] += 1
            self.current_stage = previous_stage


@dataclass(frozen=True)
class ProfileInputBundle:
    market_data: pd.DataFrame
    end_date: dt.date
    required_sessions: int
    working_start_date: dt.date
    selected_dates: list[dt.date]
    sector_etf_closes: dict[str, pd.Series]
    cross_asset_closes: dict[str, pd.Series]
    macro_series: dict[str, pd.Series]
    event_calendar: pd.DataFrame | None
    aaii_sentiment: pd.DataFrame | None
    news_sentiment: pd.Series | None
    implied_vol_30d: pd.Series | None
    central_bank_text_releases: pd.DataFrame | None
    cpi_first_release: pd.Series | None
    pit_constituent_intervals: pd.DataFrame
    constituent_ohlcv: dict[str, pd.DataFrame]
    constituent_tickers: list[str]


class RunTimeout(RuntimeError):
    pass


def _timeout_handler(timer: StageTimer, *_args: Any) -> None:
    stage = timer.current_stage or "<outside instrumented stage>"
    raise RunTimeout(
        f"Profiling run exceeded {timer.timeout_seconds}s while in stage: {stage}"
    )


def _require_path(path: Path, *, kind: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path


def _build_required_sessions(
    config: Any, session_count: int, lookback_days: int
) -> int:
    v2_min_history = ENGINE_MINIMUM_HISTORY
    trailing_component_lookback = 0
    if config.change_point is not None:
        v2_min_history = max(
            v2_min_history, config.change_point.training_window_days + 21
        )
    if config.hmm is not None:
        v2_min_history = max(v2_min_history, config.hmm.training_window_days + 63)
        trailing_component_lookback = max(trailing_component_lookback, 5)
    if config.clustering is not None:
        v2_min_history = max(
            v2_min_history, config.clustering.training_window_days + 63
        )
    return min(
        session_count, v2_min_history + lookback_days - 1 + trailing_component_lookback
    )


def _read_symbol_ohlcv(tree_root: Path, symbol: str) -> pd.DataFrame:
    symbol_dir = tree_root / f"symbol={symbol}"
    parquet_path = symbol_dir / "ohlcv.parquet"
    if parquet_path.exists():
        source = parquet_path
    else:
        partition_files = sorted(symbol_dir.glob("*.parquet"))
        if not partition_files:
            raise FileNotFoundError(parquet_path)
        source = symbol_dir
    frame = pd.read_parquet(source)
    required_cols = ["date", "open", "high", "low", "close", "volume", "adjusted_close"]
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")
    frame = frame[required_cols].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def _load_optional_aaii_sentiment(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    required_cols = {"bull_bear_spread_8w_ma"}
    missing = sorted(required_cols - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required AAII sentiment columns: {missing}")
    if "publication_date" not in frame.columns and "date" not in frame.columns:
        raise ValueError(f"{path} must contain either publication_date or date")
    return frame


def _load_optional_event_calendar(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return load_event_calendar(path)


def _resolve_news_sentiment_path(path: Path) -> Path:
    canonical = path.with_name("sf_fed_news_sentiment.parquet")
    if path.exists() or path == canonical or not canonical.exists():
        return path
    return canonical


def _load_optional_news_sentiment(path: Path | None) -> pd.Series | None:
    if path is None:
        return None
    path = _resolve_news_sentiment_path(path)
    if not path.exists():
        return None
    return load_news_sentiment_series(path)


def _load_optional_cpi_first_release(path: Path | None) -> pd.Series | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return load_cpi_vintages_first_release(path)


def _load_optional_central_bank_text_releases(
    *,
    fomc_path: Path | None,
    powell_path: Path | None,
) -> pd.DataFrame | None:
    releases = load_central_bank_text_score(
        fomc_minutes_source=(
            fomc_path if fomc_path is not None and fomc_path.exists() else None
        ),
        powell_speeches_source=(
            powell_path if powell_path is not None and powell_path.exists() else None
        ),
    )
    if releases.empty:
        return None
    return releases



def _load_constituent_ohlcv_from_tree(
    tree_root: Path,
    intervals: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    overlap_mask = (intervals["start_date"] <= end_date) & (
        intervals["end_date"].isna() | (intervals["end_date"] >= start_date)
    )
    tickers = sorted({str(t) for t in intervals.loc[overlap_mask, "ticker"].tolist()})
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        parquet_path = tree_root / f"symbol={ticker}" / "ohlcv.parquet"
        try:
            frame = _read_symbol_ohlcv(tree_root, ticker)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"PIT-constituent OHLCV missing: {parquet_path}. "
                "Re-materialize the manifest to fetch the full 1193-symbol tree."
            ) from None
        frame = frame[
            (frame["date"] >= pd.Timestamp(start_date))
            & (frame["date"] <= pd.Timestamp(end_date))
        ].copy()
        if frame.empty:
            continue
        for col in ("open", "high", "low", "close", "adjusted_close"):
            frame[col] = frame[col].astype("float64")
        frame["volume"] = frame["volume"].astype("int64")
        frame = frame.set_index("date")[
            ["open", "high", "low", "close", "volume", "adjusted_close"]
        ]
        frame.index.name = "date"
        out[ticker] = frame
    return out, tickers


def _load_profile_inputs(
    args: argparse.Namespace, *, config: Any
) -> ProfileInputBundle:
    market_data = load_market_data(args.daily_dir)
    if market_data.empty:
        raise ValueError(f"market_data is empty from {args.daily_dir}")
    end_date = max(market_data["date"])

    bootstrap_context = build_market_context(
        end_date=end_date,
        market_data=market_data,
        config=config,
    )
    spy_index = bootstrap_context.spy_ohlcv.index
    required_sessions = _build_required_sessions(
        config,
        len(bootstrap_context.sessions),
        args.lookback_days,
    )
    working_start_date = bootstrap_context.sessions[-required_sessions]
    selected_dates = list(bootstrap_context.sessions[-args.lookback_days :])

    cross_asset_symbols = CROSS_ASSET_SYMBOLS
    # Load the union of sector ETFs and cross-asset symbols in one pass to avoid
    # reading files for symbols that appear in both lists (XLY, XLI, XLP, XLU).
    all_close_symbols = list(dict.fromkeys([*SECTOR_ETFS, *cross_asset_symbols]))
    all_closes = load_close_dict(args.daily_dir, all_close_symbols, spy_index)
    sector_etf_closes = {s: all_closes[s] for s in SECTOR_ETFS if s in all_closes}
    cross_asset_closes = {s: all_closes[s] for s in cross_asset_symbols if s in all_closes}
    macro_series = load_macro_series(
        args.macro_parquet,
        args.pmi_path if args.pmi_path is not None and args.pmi_path.exists() else None,
        cpi_nowcast_parquet=args.cpi_nowcast_parquet,
        eps_weekly_history_parquet=args.aggregate_forward_eps_weekly_history_parquet,
    )
    if args.disable_aggregate_forward_eps_revision:
        # EPS revision is intentionally operator-disabled until the weekly S&P
        # snapshot accumulator has reliable fresh coverage. Keep the macro key
        # absent so earnings_expansion / earnings_contraction stay silent via
        # existing NaN-falsifies behavior.
        macro_series.pop("aggregate_forward_eps_revision", None)
    event_calendar = _load_optional_event_calendar(args.event_calendar)
    aaii_sentiment = _load_optional_aaii_sentiment(args.aaii_sentiment_parquet)
    news_sentiment = _load_optional_news_sentiment(args.news_sentiment_parquet)
    implied_vol_30d = macro_series.get("implied_vol_30d")
    central_bank_text_releases = _load_optional_central_bank_text_releases(
        fomc_path=args.fomc_minutes_parquet,
        powell_path=args.powell_speeches_parquet,
    )
    cpi_first_release = _load_optional_cpi_first_release(args.cpi_vintages_parquet)
    pit_constituent_intervals = read_pit_intervals(args.pit_parquet)
    constituent_ohlcv, constituent_tickers = _load_constituent_ohlcv_from_tree(
        args.constituent_tree,
        pit_constituent_intervals,
        start_date=working_start_date,
        end_date=end_date,
    )

    return ProfileInputBundle(
        market_data=market_data,
        end_date=end_date,
        required_sessions=required_sessions,
        working_start_date=working_start_date,
        selected_dates=selected_dates,
        sector_etf_closes=sector_etf_closes,
        cross_asset_closes=cross_asset_closes,
        macro_series=macro_series,
        event_calendar=event_calendar,
        aaii_sentiment=aaii_sentiment,
        news_sentiment=news_sentiment,
        implied_vol_30d=implied_vol_30d,
        central_bank_text_releases=central_bank_text_releases,
        cpi_first_release=cpi_first_release,
        pit_constituent_intervals=pit_constituent_intervals,
        constituent_ohlcv=constituent_ohlcv,
        constituent_tickers=constituent_tickers,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile RegimeEngine.classify_window() wall-clock stages over a configurable lookback window."
    )
    parser.add_argument("--lookback-days", type=positive_int, default=30)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--daily-dir", type=Path, default=None)
    parser.add_argument("--constituent-tree", type=Path, default=None)
    parser.add_argument("--event-calendar", type=Path, default=DEFAULT_EVENT_CALENDAR)
    # Optional manifest-routed inputs are declared once in
    # MANIFEST_INPUT_SPECS (regime_data_fetch.manifest_inputs); the
    # helper below registers them all so the per-runner argparse list
    # cannot silently drift behind the registry.
    register_manifest_input_args(parser, include_required_paths=False)
    parser.add_argument("--macro-parquet", type=Path, default=None)
    parser.add_argument("--pit-parquet", type=Path, default=None)
    parser.add_argument(
        "--operator-env-file",
        type=Path,
        default=None,
        help=(
            "Optional non-secret pointer file listing repo credential env files. "
            f"Defaults to {OPERATOR_ENV_POINTER_FILE} or ~/.config/regime-detection/operator.env."
        ),
    )
    add_manifest_args(parser, data_root_default=REPO_ROOT / "data" / "raw", action="profiling")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable profiling report JSON artifact.",
    )
    parser.add_argument(
        "--disable-aggregate-forward-eps-revision",
        action="store_true",
        help=(
            "Operator-disable the aggregate forward EPS revision seam. "
            "This leaves earnings_expansion / earnings_contraction silent "
            "without changing inflation/growth rule logic."
        ),
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=int,
        default=DEFAULT_RUN_TIMEOUT_SECONDS,
        help=(
            "SIGALRM budget for the instrumented classify_window() block. "
            f"Defaults to {DEFAULT_RUN_TIMEOUT_SECONDS}s (preserves prior "
            "behavior). Raise this for full-history profiling runs; pass 0 "
            "or a negative value to disable the alarm entirely. When the "
            "alarm fires, a RunTimeout exception is raised so callers can "
            "see that clustering/HMM was not allowed to complete."
        ),
    )
    args = parser.parse_args()
    args.manifest_input_overrides = manifest_input_overrides(sys.argv[1:])
    if args.daily_dir is None:
        args.daily_dir = args.data_root / FIXED_UNIVERSE_TREE_NAME
    if args.constituent_tree is None:
        args.constituent_tree = args.data_root / FIXED_UNIVERSE_TREE_NAME
    apply_manifest_input_defaults(args, args.data_root)
    args.news_sentiment_parquet = _resolve_news_sentiment_path(
        args.news_sentiment_parquet
    )
    return args


def _apply_manifest_input_paths(
    args: argparse.Namespace,
    *,
    runner_name: str,
    required_fields: frozenset[str] | None = None,
) -> None:
    apply_manifest_input_paths(
        args,
        runner_name=runner_name,
        repo_root=REPO_ROOT,
        required_fields=required_fields,
    )


def _is_manifest_resolution_error(error: Exception) -> bool:
    """Detect both the typed resolver error and the un-typed
    ``ValueError("manifest has no artifacts required for ...")`` that
    ``materialize_manifest`` raises before the typed resolver gets a chance to
    run. Both share the same root cause from the runner's perspective: the
    manifest does not satisfy this runner's contract."""
    if isinstance(error, ManifestInputResolutionError):
        return True
    if isinstance(error, ValueError) and str(error).startswith(
        "manifest has no artifacts required for "
    ):
        return True
    return False


def _emit_manifest_resolution_failure(
    args: argparse.Namespace, error: Exception
) -> None:
    """Emit a structured manifest_resolution_failure record to ``--json-output``
    when the runner aborts before classification. Without this, downstream
    regression dashboards that look for the JSON file see nothing and cannot
    distinguish "runner crashed" from "runner skipped"."""
    if args.json_output is None:
        return
    failure_report: dict[str, Any] = {
        "status": "manifest_resolution_failure",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "runner_name": _RUNNER_NAME,
        "data_root": str(args.data_root) if args.data_root is not None else None,
    }
    _write_json_report(args.json_output, failure_report)


def main() -> int:
    args = _parse_args()
    load_operator_env_files(repo_root=REPO_ROOT, explicit_path=args.operator_env_file)
    try:
        materialize_manifest_from_args(
            args,
            repo_root=REPO_ROOT,
            required_for=_RUNNER_NAME,
        )
        _apply_manifest_input_paths(args, runner_name=_RUNNER_NAME)
    except (ManifestInputResolutionError, ValueError) as error:
        if not _is_manifest_resolution_error(error):
            raise
        _emit_manifest_resolution_failure(args, error)
        print(
            f"manifest_resolution_failure: {error}",
            file=sys.stderr,
        )
        return 2

    _require_path(args.config_path, kind="config path")
    _require_path(args.daily_dir, kind="daily OHLCV path")
    _require_path(args.constituent_tree, kind="constituent tree path")
    _require_path(args.macro_parquet, kind="macro parquet")
    _require_path(args.pit_parquet, kind="PIT parquet")

    engine = RegimeEngine(config_path=args.config_path)
    config = engine.config

    inputs = _load_profile_inputs(args, config=config)

    timer = StageTimer(timeout_seconds=args.run_timeout_seconds)
    alarm_enabled = args.run_timeout_seconds > 0
    previous_handler = signal.getsignal(signal.SIGALRM) if alarm_enabled else None
    if alarm_enabled:
        signal.signal(signal.SIGALRM, lambda *a: _timeout_handler(timer, *a))
        signal.alarm(args.run_timeout_seconds)
    wall_start = time.perf_counter()
    try:
        with _install_timers(timer):
            timeline = engine.classify_window(
                end_date=inputs.end_date,
                market_data=inputs.market_data,
                lookback_days=args.lookback_days,
                sector_etf_closes=inputs.sector_etf_closes,
                cross_asset_closes=inputs.cross_asset_closes,
                macro_series=inputs.macro_series,
                event_calendar=inputs.event_calendar,
                aaii_sentiment=inputs.aaii_sentiment,
                implied_vol_30d=inputs.implied_vol_30d,
                central_bank_text_releases=inputs.central_bank_text_releases,
                cpi_first_release=inputs.cpi_first_release,
                news_sentiment=inputs.news_sentiment,
                pit_constituent_intervals=inputs.pit_constituent_intervals,
                constituent_ohlcv=inputs.constituent_ohlcv,
            )
    finally:
        if alarm_enabled:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    total_wall_clock = time.perf_counter() - wall_start

    context = build_market_context(
        end_date=inputs.end_date,
        market_data=inputs.market_data,
        config=config,
        sector_etf_closes=inputs.sector_etf_closes,
        cross_asset_closes=inputs.cross_asset_closes,
        macro_series=inputs.macro_series,
        event_calendar=inputs.event_calendar,
        aaii_sentiment=inputs.aaii_sentiment,
        implied_vol_30d=inputs.implied_vol_30d,
        central_bank_text_releases=inputs.central_bank_text_releases,
        cpi_first_release=inputs.cpi_first_release,
        news_sentiment=inputs.news_sentiment,
        pit_constituent_intervals=inputs.pit_constituent_intervals,
        constituent_ohlcv=inputs.constituent_ohlcv,
    )
    import regime_detection.market_context as market_context_module
    import regime_detection.feature_store as feature_store_module

    working_context = market_context_module.slice_context_to_recent_sessions(
        context=context,
        required_sessions=inputs.required_sessions,
    )
    feature_store = feature_store_module.build_feature_store(
        working_context,
        network_fragility_config=config.network_fragility,
        trend_direction_v2_config=config.trend_direction_v2,
        volatility_state_v2_config=config.volatility_state_v2,
        breadth_state_v2_config=config.breadth_state_v2,
        volume_liquidity_v2_config=config.volume_liquidity_v2,
        monetary_pressure_v2_config=config.monetary_pressure_v2,
        credit_funding_config=config.credit_funding,
        inflation_growth_config=config.inflation_growth,
        central_bank_text_config=config.central_bank_text,
        news_sentiment_config=config.news_sentiment,
    )

    per_day_emission_total = max(
        0.0,
        timer.totals.get("build_regime_timeline_total", 0.0)
        - timer.totals.get("slice_context_to_recent_sessions", 0.0)
        - timer.totals.get("build_feature_store_total", 0.0)
        - timer.totals.get("build_axis_series_bundle", 0.0)
        - timer.totals.get("build_transition_risk_series", 0.0),
    )
    per_day_avg_ms = per_day_emission_total / args.lookback_days * 1000.0

    stage_rows = _format_stage_rows(
        [
            "build_market_context",
            "slice_context_to_recent_sessions",
            "build_feature_store_total",
            "build_axis_series_bundle",
            "build_transition_risk_series",
            "build_regime_timeline_total",
        ],
        timer,
        total_wall_clock,
    )
    stage_rows.append(
        f"per_day_output_emission_loop_residual | {per_day_emission_total:.6f} | "
        f"{(per_day_emission_total / total_wall_clock * 100.0) if total_wall_clock > 0 else 0.0:6.2f}%"
    )

    feature_rows = _format_stage_rows(
        [
            "feature_store.network_fragility",
            "feature_store.trend_direction_v2",
            "feature_store.volatility_state_v2",
            "feature_store.breadth_state_v2",
            "feature_store.volume_liquidity_v2",
            "feature_store.monetary_pressure_v2",
            "feature_store.credit_funding",
            "feature_store.inflation_growth",
            "feature_store.hmm",
            "feature_store.gmm_clustering",
            "feature_store.change_point",
        ],
        timer,
        timer.totals.get("build_feature_store_total", 0.0),
    )

    verification_issues = _verify_invariants(timeline, feature_store, inputs)
    if args.json_output is not None:
        _write_json_report(
            args.json_output,
            _build_json_report(
                args=args,
                inputs=inputs,
                timeline=timeline,
                timer=timer,
                total_wall_clock=total_wall_clock,
                per_day_emission_total=per_day_emission_total,
                per_day_avg_ms=per_day_avg_ms,
                verification_issues=verification_issues,
                feature_store=feature_store,
            ),
        )

    print(f"config_path={args.config_path}")
    print(f"market_data_source={args.daily_dir}")
    print(f"constituent_tree_source={args.constituent_tree}")
    print(f"macro_source={args.macro_parquet}")
    print(
        f"event_calendar_source={args.event_calendar if inputs.event_calendar is not None else '<absent>'}"
    )
    print(
        f"aaii_sentiment_source={args.aaii_sentiment_parquet if inputs.aaii_sentiment is not None else '<absent>'}"
    )
    print(
        f"news_sentiment_source={args.news_sentiment_parquet if inputs.news_sentiment is not None else '<absent>'}"
    )
    print(
        f"implied_vol_30d_source={'macro_series[implied_vol_30d]' if inputs.implied_vol_30d is not None else '<absent>'}"
    )
    print(
        f"fomc_minutes_source={args.fomc_minutes_parquet if args.fomc_minutes_parquet is not None and args.fomc_minutes_parquet.exists() else '<absent>'}"
    )
    print(
        f"powell_speeches_source={args.powell_speeches_parquet if args.powell_speeches_parquet is not None and args.powell_speeches_parquet.exists() else '<absent>'}"
    )
    print(
        f"cpi_vintages_source={args.cpi_vintages_parquet if inputs.cpi_first_release is not None else '<absent>'}"
    )
    print(f"pit_source={args.pit_parquet}")
    print(f"end_date={inputs.end_date.isoformat()}")
    print(f"selected_window_start={inputs.selected_dates[0].isoformat()}")
    print(f"selected_window_end={inputs.selected_dates[-1].isoformat()}")
    print(f"working_window_start={inputs.working_start_date.isoformat()}")
    print(f"required_sessions={inputs.required_sessions}")
    print(f"lookback_days={args.lookback_days}")
    print()

    print("Input seam status")
    input_values = _profile_input_seam_values(inputs)
    for name in PROFILE_INPUT_SEAM_NAMES:
        print(_input_status(name, input_values[name]))
    print(f"pit_overlap_tickers_requested={len(inputs.constituent_tickers)}")
    print(f"constituent_tickers_loaded={len(inputs.constituent_ohlcv)}")
    print()

    print("Timing table")
    for row in stage_rows:
        print(row)
    print()

    print("build_feature_store sub-breakdown")
    for row in feature_rows:
        print(row)
    print()

    print("build_axis_series_bundle sub-breakdown")
    for row in _format_stage_rows(
        [
            "axis_series.trend_direction",
            "axis_series.trend_character",
            "axis_series.volatility_state",
            "axis_series.breadth_state",
            "axis_series.event_calendar",
            "axis_series.credit_funding",
            "axis_series.network_fragility",
            "axis_series.volume_liquidity_state",
            "axis_series.monetary_pressure_state",
            "axis_series.inflation_growth",
        ],
        timer,
        timer.totals.get("build_axis_series_bundle", 0.0),
    ):
        print(row)
    print()

    print("axis_series.inflation_growth sub-breakdown")
    for row in _format_stage_rows(
        [
            "axis_series.inflation_growth.build_rule_inputs_by_date",
            "axis_series.inflation_growth.assess_series_input_quality",
            "axis_series.inflation_growth.evaluate_rules",
        ],
        timer,
        timer.totals.get("axis_series.inflation_growth", 0.0),
    ):
        print(row)
    print()

    print(
        "per_day_output_emission | "
        f"total_seconds={per_day_emission_total:.6f} | avg_ms_per_day={per_day_avg_ms:.3f}"
    )
    print()

    print("Compact RegimeTimeline")
    for row in _compact_timeline_rows(timeline.outputs):
        print(row)
    print()

    print("Trailing-session V2 field status")
    for row in _trailing_v2_status(timeline.outputs[-1]):
        print(row)
    print()

    if verification_issues:
        print("Verification issues")
        for issue in verification_issues:
            print(f"- {issue}")
    else:
        print("Verification issues")
        print("- none")
    print()
    print(f"bottom_line_total_wall_clock_seconds={total_wall_clock:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
