#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, get_args

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from regime_data_fetch.pit_constituents import read_pit_intervals
from regime_data_fetch.materialization import materialize_if_requested
from regime_data_fetch.universe import FIXED_UNIVERSE_TREE_NAME
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import FeatureStore
from regime_detection.fragility_universe import CROSS_ASSET_SYMBOLS, SECTOR_ETFS
from regime_detection.loaders import (
    load_central_bank_text_score,
    load_cpi_vintages_first_release,
    load_event_calendar,
    load_news_sentiment_series,
)
from regime_detection.market_context import build_market_context
from regime_detection.models import ClassificationStatus, RegimeOutput, RegimeTimeline
from regime_detection.timeline import ENGINE_MINIMUM_HISTORY
from scripts._v2_calibration_helpers import (
    default_pmi_path,
    load_close_dict,
    load_macro_series,
    load_market_data,
    positive_int,
)


DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "src" / "regime_detection" / "configs" / "core3-v2.0.0.yaml"
)
DEFAULT_DAILY_DIR = REPO_ROOT / "data" / "raw" / "daily_ohlcv"
DEFAULT_CONSTITUENT_TREE = REPO_ROOT / "data" / "raw" / FIXED_UNIVERSE_TREE_NAME
DEFAULT_MACRO_PARQUET = (
    REPO_ROOT / "data" / "raw" / "macro" / "fred_macro_series.parquet"
)
DEFAULT_PIT_PARQUET = (
    REPO_ROOT / "data" / "raw" / "pit_constituents" / "sp500_ticker_intervals.parquet"
)
DEFAULT_PMI_PATH = default_pmi_path(REPO_ROOT / "data" / "raw")
DEFAULT_EVENT_CALENDAR = REPO_ROOT / "configs" / "events" / "us_events.yaml"
RUN_TIMEOUT_SECONDS = 300
NON_CLASSIFIED_REPORTING_LABELS = set(get_args(ClassificationStatus)) - {"classified"}


@dataclass
class StageTimer:
    totals: dict[str, float]
    counts: dict[str, int]
    current_stage: str | None = None

    def __init__(self) -> None:
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
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
    missing_constituent_paths: list[Path]
    input_kwargs: dict[str, Any]


class RunTimeout(RuntimeError):
    pass


def _timeout_handler(timer: StageTimer, *_args: Any) -> None:
    stage = timer.current_stage or "<outside instrumented stage>"
    raise RunTimeout(
        f"Profiling run exceeded {RUN_TIMEOUT_SECONDS}s while in stage: {stage}"
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
    parquet_path = tree_root / f"symbol={symbol}" / "ohlcv.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    frame = pd.read_parquet(parquet_path)
    required_cols = ["date", "open", "high", "low", "close", "volume", "adjusted_close"]
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"{parquet_path} missing required columns: {missing}")
    frame = frame[required_cols].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def _load_optional_aaii_sentiment(path: Path) -> pd.DataFrame | None:
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


def _load_optional_news_sentiment(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    return load_news_sentiment_series(path)


def _load_optional_cpi_first_release(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    return load_cpi_vintages_first_release(path)


def _load_optional_central_bank_text_releases(
    *,
    fomc_path: Path,
    powell_path: Path,
) -> pd.DataFrame | None:
    releases = load_central_bank_text_score(
        fomc_minutes_source=fomc_path if fomc_path.exists() else None,
        powell_speeches_source=powell_path if powell_path.exists() else None,
    )
    if releases.empty:
        return None
    return releases


def _load_market_data_from_tree(tree_root: Path, symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        frame = _read_symbol_ohlcv(tree_root, symbol)
        frame["symbol"] = symbol
        frames.append(
            frame[["date", "symbol", "open", "high", "low", "close", "volume"]]
        )
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _load_constituent_ohlcv_from_tree(
    tree_root: Path,
    intervals: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
    *,
    allow_missing_files: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str], list[Path]]:
    overlap_mask = (intervals["start_date"] <= end_date) & (
        intervals["end_date"].isna() | (intervals["end_date"] >= start_date)
    )
    tickers = sorted({str(t) for t in intervals.loc[overlap_mask, "ticker"].tolist()})
    out: dict[str, pd.DataFrame] = {}
    missing_paths: list[Path] = []
    for ticker in tickers:
        parquet_path = tree_root / f"symbol={ticker}" / "ohlcv.parquet"
        if not parquet_path.exists():
            if not allow_missing_files:
                raise FileNotFoundError(parquet_path)
            missing_paths.append(parquet_path)
            continue
        frame = _read_symbol_ohlcv(tree_root, ticker)
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
    return out, tickers, missing_paths


def _input_status(name: str, value: Any) -> str:
    if value is None:
        return f"{name}: NONE"
    if isinstance(value, dict):
        if not value:
            return f"{name}: EMPTY_DICT"
        return f"{name}: {len(value)} keys"
    if isinstance(value, pd.DataFrame):
        return f"{name}: {len(value)} rows"
    if isinstance(value, pd.Series):
        return f"{name}: {len(value)} rows"
    return f"{name}: type={type(value).__name__}"


def _input_status_report(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {"name": name, "status": "none", "count": 0}
    if isinstance(value, dict):
        return {
            "name": name,
            "status": "empty_dict" if not value else "present",
            "kind": "dict",
            "count": len(value),
        }
    if isinstance(value, pd.DataFrame):
        return {
            "name": name,
            "status": "present",
            "kind": "dataframe",
            "rows": len(value),
            "columns": list(value.columns),
        }
    if isinstance(value, pd.Series):
        return {
            "name": name,
            "status": "present",
            "kind": "series",
            "rows": len(value),
            "name_in_series": value.name,
        }
    return {
        "name": name,
        "status": "present",
        "kind": type(value).__name__,
    }


def _timed_wrapper(
    timer: StageTimer,
    stage_name: str,
    func: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with timer.measure(stage_name):
            return func(*args, **kwargs)

    return wrapped


def _timed_method_wrapper(
    timer: StageTimer,
    stage_name: str,
    method: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with timer.measure(stage_name):
            return method(*args, **kwargs)

    return wrapped


def _timed_inflation_growth_builder(
    timer: StageTimer,
    method: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapped(
        self,
        context: Any,
        feature_store: Any,
        credit_funding_active_labels_by_date: Any = None,
    ) -> Any:
        import regime_detection.axis_series as axis_series_module

        original_assess = axis_series_module.assess_series_input_quality
        original_build_inputs = (
            axis_series_module.build_inflation_growth_rule_inputs_by_date
        )
        original_eval = axis_series_module.evaluate_inflation_growth_rules

        def timed_assess(*args: Any, **kwargs: Any) -> Any:
            with timer.measure(
                "axis_series.inflation_growth.assess_series_input_quality"
            ):
                return original_assess(*args, **kwargs)

        def timed_build_inputs(*args: Any, **kwargs: Any) -> Any:
            with timer.measure(
                "axis_series.inflation_growth.build_rule_inputs_by_date"
            ):
                return original_build_inputs(*args, **kwargs)

        def timed_eval(*args: Any, **kwargs: Any) -> Any:
            with timer.measure("axis_series.inflation_growth.evaluate_rules"):
                return original_eval(*args, **kwargs)

        with timer.measure("axis_series.inflation_growth"):
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    _patched_attr(
                        axis_series_module, "assess_series_input_quality", timed_assess
                    )
                )
                stack.enter_context(
                    _patched_attr(
                        axis_series_module,
                        "build_inflation_growth_rule_inputs_by_date",
                        timed_build_inputs,
                    )
                )
                stack.enter_context(
                    _patched_attr(
                        axis_series_module,
                        "evaluate_inflation_growth_rules",
                        timed_eval,
                    )
                )
                return method(
                    self,
                    context,
                    feature_store,
                    credit_funding_active_labels_by_date=credit_funding_active_labels_by_date,
                )

    return wrapped


@contextlib.contextmanager
def _patched_attr(module: Any, attr_name: str, replacement: Any):
    original = getattr(module, attr_name)
    setattr(module, attr_name, replacement)
    try:
        yield
    finally:
        setattr(module, attr_name, original)


@contextlib.contextmanager
def _install_timers(timer: StageTimer):
    import regime_detection.engine as engine_module
    import regime_detection.axis_series as axis_series_module
    import regime_detection.feature_store as feature_store_module
    import regime_detection.timeline as timeline_module

    patches = [
        (
            engine_module,
            "build_market_context",
            _timed_wrapper(
                timer, "build_market_context", engine_module.build_market_context
            ),
        ),
        (
            engine_module,
            "build_regime_timeline",
            _timed_wrapper(
                timer,
                "build_regime_timeline_total",
                engine_module.build_regime_timeline,
            ),
        ),
        (
            timeline_module,
            "slice_context_to_recent_sessions",
            _timed_wrapper(
                timer,
                "slice_context_to_recent_sessions",
                timeline_module.slice_context_to_recent_sessions,
            ),
        ),
        (
            timeline_module,
            "build_feature_store",
            _timed_wrapper(
                timer, "build_feature_store_total", timeline_module.build_feature_store
            ),
        ),
        (
            timeline_module,
            "build_axis_series_bundle",
            _timed_wrapper(
                timer,
                "build_axis_series_bundle",
                timeline_module.build_axis_series_bundle,
            ),
        ),
        (
            timeline_module,
            "build_transition_risk_series",
            _timed_wrapper(
                timer,
                "build_transition_risk_series",
                timeline_module.build_transition_risk_series,
            ),
        ),
        (
            feature_store_module,
            "compute_network_fragility_features",
            _timed_wrapper(
                timer,
                "feature_store.network_fragility",
                feature_store_module.compute_network_fragility_features,
            ),
        ),
        (
            feature_store_module,
            "compute_trend_v2_features",
            _timed_wrapper(
                timer,
                "feature_store.trend_direction_v2",
                feature_store_module.compute_trend_v2_features,
            ),
        ),
        (
            feature_store_module,
            "compute_volatility_v2_features",
            _timed_wrapper(
                timer,
                "feature_store.volatility_state_v2",
                feature_store_module.compute_volatility_v2_features,
            ),
        ),
        (
            feature_store_module,
            "compute_breadth_v2_features",
            _timed_wrapper(
                timer,
                "feature_store.breadth_state_v2",
                feature_store_module.compute_breadth_v2_features,
            ),
        ),
        (
            feature_store_module,
            "compute_volume_liquidity_v2_features",
            _timed_wrapper(
                timer,
                "feature_store.volume_liquidity_v2",
                feature_store_module.compute_volume_liquidity_v2_features,
            ),
        ),
        (
            feature_store_module,
            "compute_monetary_pressure_features",
            _timed_wrapper(
                timer,
                "feature_store.monetary_pressure_v2",
                feature_store_module.compute_monetary_pressure_features,
            ),
        ),
        (
            feature_store_module,
            "compute_credit_funding_features",
            _timed_wrapper(
                timer,
                "feature_store.credit_funding",
                feature_store_module.compute_credit_funding_features,
            ),
        ),
        (
            feature_store_module,
            "compute_inflation_growth_features",
            _timed_wrapper(
                timer,
                "feature_store.inflation_growth",
                feature_store_module.compute_inflation_growth_features,
            ),
        ),
        (
            feature_store_module,
            "compute_hmm_features",
            _timed_wrapper(
                timer, "feature_store.hmm", feature_store_module.compute_hmm_features
            ),
        ),
        (
            feature_store_module,
            "compute_clustering_features",
            _timed_wrapper(
                timer,
                "feature_store.gmm_clustering",
                feature_store_module.compute_clustering_features,
            ),
        ),
        (
            feature_store_module,
            "compute_change_point_features",
            _timed_wrapper(
                timer,
                "feature_store.change_point",
                feature_store_module.compute_change_point_features,
            ),
        ),
        (
            axis_series_module.TrendDirectionSeriesClassifier,
            "build",
            _timed_method_wrapper(
                timer,
                "axis_series.trend_direction",
                axis_series_module.TrendDirectionSeriesClassifier.build,
            ),
        ),
        (
            axis_series_module.TrendCharacterSeriesClassifier,
            "build",
            _timed_method_wrapper(
                timer,
                "axis_series.trend_character",
                axis_series_module.TrendCharacterSeriesClassifier.build,
            ),
        ),
        (
            axis_series_module.VolatilitySeriesClassifier,
            "build",
            _timed_method_wrapper(
                timer,
                "axis_series.volatility_state",
                axis_series_module.VolatilitySeriesClassifier.build,
            ),
        ),
        (
            axis_series_module.BreadthSeriesClassifier,
            "build",
            _timed_method_wrapper(
                timer,
                "axis_series.breadth_state",
                axis_series_module.BreadthSeriesClassifier.build,
            ),
        ),
        (
            axis_series_module.CreditFundingSeriesClassifier,
            "build",
            _timed_method_wrapper(
                timer,
                "axis_series.credit_funding",
                axis_series_module.CreditFundingSeriesClassifier.build,
            ),
        ),
        (
            axis_series_module.NetworkFragilitySeriesClassifier,
            "build",
            _timed_method_wrapper(
                timer,
                "axis_series.network_fragility",
                axis_series_module.NetworkFragilitySeriesClassifier.build,
            ),
        ),
        (
            axis_series_module.VolumeLiquidityStateSeriesClassifier,
            "build",
            _timed_method_wrapper(
                timer,
                "axis_series.volume_liquidity_state",
                axis_series_module.VolumeLiquidityStateSeriesClassifier.build,
            ),
        ),
        (
            axis_series_module.MonetaryPressureV2SeriesClassifier,
            "build",
            _timed_method_wrapper(
                timer,
                "axis_series.monetary_pressure_state",
                axis_series_module.MonetaryPressureV2SeriesClassifier.build,
            ),
        ),
        (
            axis_series_module.InflationGrowthSeriesClassifier,
            "build",
            _timed_inflation_growth_builder(
                timer, axis_series_module.InflationGrowthSeriesClassifier.build
            ),
        ),
        (
            axis_series_module,
            "build_event_calendar_series",
            _timed_wrapper(
                timer,
                "axis_series.event_calendar",
                axis_series_module.build_event_calendar_series,
            ),
        ),
    ]
    with contextlib.ExitStack() as stack:
        for module, attr_name, replacement in patches:
            stack.enter_context(_patched_attr(module, attr_name, replacement))
        yield


def _format_stage_rows(
    stage_names: list[str], timer: StageTimer, total: float
) -> list[str]:
    rows = ["stage_name | wall_clock_seconds | % of total"]
    for stage_name in stage_names:
        seconds = timer.totals.get(stage_name, 0.0)
        pct = (seconds / total * 100.0) if total > 0 else 0.0
        rows.append(f"{stage_name} | {seconds:.6f} | {pct:6.2f}%")
    return rows


def _stage_report(
    stage_names: list[str], timer: StageTimer, total: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_name in stage_names:
        seconds = timer.totals.get(stage_name, 0.0)
        pct = (seconds / total * 100.0) if total > 0 else 0.0
        rows.append(
            {
                "stage_name": stage_name,
                "wall_clock_seconds": seconds,
                "percent_of_total": pct,
                "call_count": timer.counts.get(stage_name, 0),
            }
        )
    return rows


def _reporting_label(output: Any) -> str | None:
    if output is None:
        return None
    reporting = getattr(output, "reporting_label", None)
    if reporting is not None:
        return reporting
    classification_status = getattr(output, "classification_status", "classified")
    if classification_status != "classified":
        return classification_status
    return getattr(output, "active_label", None)


def _compact_timeline_rows(outputs: list[RegimeOutput]) -> list[str]:
    rows = [
        "as_of_date | trend_direction | volatility_state | transition_risk | activated_v2_seams"
    ]
    for out in outputs:
        seams: list[str] = []
        network_fragility_label = _reporting_label(out.network_fragility)
        if (
            network_fragility_label is not None
            and network_fragility_label not in NON_CLASSIFIED_REPORTING_LABELS
        ):
            seams.append(f"network_fragility={network_fragility_label}")
        if out.volume_liquidity_state is not None:
            seams.append(
                f"volume_liquidity_state={_reporting_label(out.volume_liquidity_state)}"
            )
        if out.credit_funding_state is not None:
            seams.append(
                f"credit_funding_state={_reporting_label(out.credit_funding_state)}"
            )
        if out.credit_funding_state_proxy is not None:
            seams.append(
                f"credit_funding_state_proxy={_reporting_label(out.credit_funding_state_proxy)}"
            )
        if out.credit_funding_effective_state is not None:
            source = out.credit_funding_effective_state.evidence.get(
                "source_used", "not_recorded"
            )
            seams.append(
                "credit_funding_effective_state="
                f"{_reporting_label(out.credit_funding_effective_state)}"
                f"({source})"
            )
        if out.inflation_growth_state is not None:
            seams.append(
                f"inflation_growth_state={_reporting_label(out.inflation_growth_state)}"
            )
        if out.monetary_pressure_state is not None:
            seams.append(
                f"monetary_pressure_state={_reporting_label(out.monetary_pressure_state)}"
            )
        if out.cluster is not None:
            seams.append(f"cluster={out.cluster.cluster_id}")
        if out.change_point is not None:
            seams.append(f"change_point={out.change_point.score:.4f}")
        if out.transition_risk.score is not None:
            seams.append(f"transition_score={out.transition_risk.score:.4f}")
        seam_text = ", ".join(seams) if seams else "-"
        rows.append(
            f"{out.as_of_date.isoformat()} | "
            f"{_reporting_label(out.trend_direction)} | "
            f"{_reporting_label(out.volatility_state)} | "
            f"{out.transition_risk.label} | "
            f"{seam_text}"
        )
    return rows


def _compact_timeline_report(outputs: list[RegimeOutput]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for out in outputs:
        seams: dict[str, Any] = {}
        network_fragility_label = _reporting_label(out.network_fragility)
        if (
            network_fragility_label is not None
            and network_fragility_label not in NON_CLASSIFIED_REPORTING_LABELS
        ):
            seams["network_fragility"] = network_fragility_label
        if out.volume_liquidity_state is not None:
            seams["volume_liquidity_state"] = _reporting_label(
                out.volume_liquidity_state
            )
        if out.credit_funding_state is not None:
            seams["credit_funding_state"] = _reporting_label(out.credit_funding_state)
        if out.credit_funding_state_proxy is not None:
            seams["credit_funding_state_proxy"] = _reporting_label(
                out.credit_funding_state_proxy
            )
        if out.credit_funding_effective_state is not None:
            seams["credit_funding_effective_state"] = {
                "reported": _reporting_label(out.credit_funding_effective_state),
                "source_used": out.credit_funding_effective_state.evidence.get(
                    "source_used", "not_recorded"
                ),
            }
        if out.inflation_growth_state is not None:
            seams["inflation_growth_state"] = _reporting_label(
                out.inflation_growth_state
            )
        if out.monetary_pressure_state is not None:
            seams["monetary_pressure_state"] = _reporting_label(
                out.monetary_pressure_state
            )
        if out.cluster is not None:
            seams["cluster"] = out.cluster.cluster_id
        if out.change_point is not None:
            seams["change_point"] = out.change_point.score
        if out.transition_risk.score is not None:
            seams["transition_score"] = out.transition_risk.score
        rows.append(
            {
                "as_of_date": out.as_of_date.isoformat(),
                "trend_direction": _reporting_label(out.trend_direction),
                "volatility_state": _reporting_label(out.volatility_state),
                "transition_risk": out.transition_risk.label,
                "activated_v2_seams": seams,
            }
        )
    return rows


def _trailing_v2_status(out: RegimeOutput) -> list[str]:
    rows = ["field | status"]

    def add(name: str, value: Any) -> None:
        if value is None:
            rows.append(f"{name} | NONE")
            return
        if isinstance(value, float) and math.isnan(value):
            rows.append(f"{name} | NaN")
            return
        if hasattr(value, "active_label"):
            rows.append(f"{name} | reported={_reporting_label(value)}")
            return
        rows.append(f"{name} | present")

    add("network_fragility", out.network_fragility)
    add("volume_liquidity_state", out.volume_liquidity_state)
    add("credit_funding_state", out.credit_funding_state)
    add("credit_funding_state_proxy", out.credit_funding_state_proxy)
    add("credit_funding_effective_state", out.credit_funding_effective_state)
    add("inflation_growth_state", out.inflation_growth_state)
    add("monetary_pressure_state", out.monetary_pressure_state)
    add("cluster", out.cluster)
    add("change_point", out.change_point)
    add("transition_risk.score", out.transition_risk.score)
    add("transition_risk.score_components", out.transition_risk.score_components)
    return rows


def _trailing_v2_status_report(out: RegimeOutput) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, value: Any) -> None:
        if value is None:
            rows.append({"field": name, "status": "none"})
            return
        if isinstance(value, float) and math.isnan(value):
            rows.append({"field": name, "status": "nan"})
            return
        if hasattr(value, "active_label"):
            rows.append(
                {
                    "field": name,
                    "status": "present",
                    "reported": _reporting_label(value),
                }
            )
            return
        rows.append({"field": name, "status": "present"})

    add("network_fragility", out.network_fragility)
    add("volume_liquidity_state", out.volume_liquidity_state)
    add("credit_funding_state", out.credit_funding_state)
    add("credit_funding_state_proxy", out.credit_funding_state_proxy)
    add("credit_funding_effective_state", out.credit_funding_effective_state)
    add("inflation_growth_state", out.inflation_growth_state)
    add("monetary_pressure_state", out.monetary_pressure_state)
    add("cluster", out.cluster)
    add("change_point", out.change_point)
    add("transition_risk.score", out.transition_risk.score)
    add("transition_risk.score_components", out.transition_risk.score_components)
    return rows


def _verify_invariants(
    timeline: RegimeTimeline,
    feature_store: FeatureStore,
    input_kwargs: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    for out in timeline.outputs:
        if out.trend_direction.active_label is None:
            issues.append(
                f"{out.as_of_date.isoformat()}: trend_direction.active_label is None"
            )
    trailing = timeline.outputs[-1]
    expected_non_none = [
        ("network_fragility", trailing.network_fragility),
        ("volume_liquidity_state", trailing.volume_liquidity_state),
        ("credit_funding_state", trailing.credit_funding_state),
        ("credit_funding_effective_state", trailing.credit_funding_effective_state),
        ("inflation_growth_state", trailing.inflation_growth_state),
        ("monetary_pressure_state", trailing.monetary_pressure_state),
    ]
    for name, value in expected_non_none:
        if value is None:
            issues.append(f"Trailing session missing expected V2 field: {name}")
    seam_expectations = [
        ("network_fragility", feature_store.network_fragility, ["sector_etf_closes"]),
        ("trend_direction_v2", feature_store.trend_direction_v2, []),
        ("volatility_state_v2", feature_store.volatility_state_v2, []),
        (
            "breadth_state_v2",
            feature_store.breadth_state_v2,
            ["sector_etf_closes", "pit_constituent_intervals", "constituent_ohlcv"],
        ),
        ("volume_liquidity_v2", feature_store.volume_liquidity_v2, []),
        ("monetary_pressure_v2", feature_store.monetary, ["macro_series"]),
        (
            "credit_funding",
            feature_store.credit_funding,
            ["cross_asset_closes", "macro_series"],
        ),
        (
            "inflation_growth",
            feature_store.inflation_growth,
            ["cross_asset_closes", "macro_series"],
        ),
        ("hmm", feature_store.hmm, []),
        ("gmm_clustering", feature_store.clustering, []),
        ("change_point", feature_store.change_point, []),
    ]
    for seam_name, seam_value, deps in seam_expectations:
        if seam_value is None:
            missing = [dep for dep in deps if not input_kwargs.get(dep)]
            if missing:
                issues.append(
                    f"{seam_name} is None; missing inputs: {', '.join(missing)}"
                )
    return issues


def _path_text(path: Path | None, *, present: bool = True) -> str | None:
    if path is None or not present:
        return None
    return str(path)


def _build_json_report(
    *,
    args: argparse.Namespace,
    inputs: ProfileInputBundle,
    timeline: RegimeTimeline,
    timer: StageTimer,
    total_wall_clock: float,
    per_day_emission_total: float,
    per_day_avg_ms: float,
    verification_issues: list[str],
) -> dict[str, Any]:
    stage_names = [
        "build_market_context",
        "slice_context_to_recent_sessions",
        "build_feature_store_total",
        "build_axis_series_bundle",
        "build_transition_risk_series",
        "build_regime_timeline_total",
    ]
    stage_rows = _stage_report(stage_names, timer, total_wall_clock)
    residual_pct = (
        per_day_emission_total / total_wall_clock * 100.0
        if total_wall_clock > 0
        else 0.0
    )
    stage_rows.append(
        {
            "stage_name": "per_day_output_emission_loop_residual",
            "wall_clock_seconds": per_day_emission_total,
            "percent_of_total": residual_pct,
            "call_count": args.lookback_days,
        }
    )

    input_names = [
        "sector_etf_closes",
        "cross_asset_closes",
        "macro_series",
        "event_calendar",
        "aaii_sentiment",
        "news_sentiment",
        "implied_vol_30d",
        "central_bank_text_releases",
        "cpi_first_release",
        "pit_constituent_intervals",
        "constituent_ohlcv",
    ]

    return {
        "sources": {
            "config_path": str(args.config_path),
            "market_data": str(args.daily_dir),
            "constituent_tree": str(args.constituent_tree),
            "macro": str(args.macro_parquet),
            "event_calendar": _path_text(
                args.event_calendar, present=inputs.event_calendar is not None
            ),
            "aaii_sentiment": _path_text(
                args.aaii_sentiment_parquet,
                present=inputs.aaii_sentiment is not None,
            ),
            "news_sentiment": _path_text(
                args.news_sentiment_parquet,
                present=inputs.news_sentiment is not None,
            ),
            "implied_vol_30d": (
                "macro_series[implied_vol_30d]"
                if inputs.implied_vol_30d is not None
                else None
            ),
            "fomc_minutes": _path_text(
                args.fomc_minutes_parquet,
                present=args.fomc_minutes_parquet.exists(),
            ),
            "powell_speeches": _path_text(
                args.powell_speeches_parquet,
                present=args.powell_speeches_parquet.exists(),
            ),
            "cpi_vintages": _path_text(
                args.cpi_vintages_parquet,
                present=inputs.cpi_first_release is not None,
            ),
            "pit": str(args.pit_parquet),
        },
        "window": {
            "end_date": inputs.end_date.isoformat(),
            "selected_window_start": inputs.selected_dates[0].isoformat(),
            "selected_window_end": inputs.selected_dates[-1].isoformat(),
            "working_window_start": inputs.working_start_date.isoformat(),
            "required_sessions": inputs.required_sessions,
            "lookback_days": args.lookback_days,
        },
        "inputs": {
            "seams": [
                _input_status_report(name, inputs.input_kwargs[name])
                for name in input_names
            ],
            "pit_overlap_tickers_requested": len(inputs.constituent_tickers),
            "constituent_tickers_loaded": len(inputs.constituent_ohlcv),
            "missing_constituent_files": [
                str(path) for path in inputs.missing_constituent_paths
            ],
        },
        "timing": {
            "stages": stage_rows,
            "feature_store": _stage_report(
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
            ),
            "axis_series_bundle": _stage_report(
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
            ),
            "inflation_growth": _stage_report(
                [
                    "axis_series.inflation_growth.build_rule_inputs_by_date",
                    "axis_series.inflation_growth.assess_series_input_quality",
                    "axis_series.inflation_growth.evaluate_rules",
                ],
                timer,
                timer.totals.get("axis_series.inflation_growth", 0.0),
            ),
            "per_day_output_emission": {
                "total_seconds": per_day_emission_total,
                "avg_ms_per_day": per_day_avg_ms,
            },
            "bottom_line_total_wall_clock_seconds": total_wall_clock,
        },
        "timeline": _compact_timeline_report(timeline.outputs),
        "trailing_v2_field_status": _trailing_v2_status_report(timeline.outputs[-1]),
        "verification_issues": verification_issues,
    }


def _write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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

    sector_etf_closes = load_close_dict(args.daily_dir, list(SECTOR_ETFS), spy_index)
    cross_asset_symbols = [
        *CROSS_ASSET_SYMBOLS,
        "DBC",
        "KRE",
        "XLY",
        "XLI",
        "XLP",
        "XLU",
    ]
    cross_asset_closes = load_close_dict(args.daily_dir, cross_asset_symbols, spy_index)
    macro_series = load_macro_series(
        args.macro_parquet, args.pmi_path if args.pmi_path.exists() else None
    )
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
    constituent_ohlcv, constituent_tickers, missing_constituent_paths = (
        _load_constituent_ohlcv_from_tree(
            args.constituent_tree,
            pit_constituent_intervals,
            start_date=working_start_date,
            end_date=end_date,
            allow_missing_files=args.allow_missing_constituent_files,
        )
    )

    input_kwargs = {
        "sector_etf_closes": sector_etf_closes,
        "cross_asset_closes": cross_asset_closes,
        "macro_series": macro_series,
        "event_calendar": event_calendar,
        "aaii_sentiment": aaii_sentiment,
        "news_sentiment": news_sentiment,
        "implied_vol_30d": implied_vol_30d,
        "central_bank_text_releases": central_bank_text_releases,
        "cpi_first_release": cpi_first_release,
        "pit_constituent_intervals": pit_constituent_intervals,
        "constituent_ohlcv": constituent_ohlcv,
    }
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
        missing_constituent_paths=missing_constituent_paths,
        input_kwargs=input_kwargs,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile 30-session RegimeEngine.classify_window() wall-clock stages."
    )
    parser.add_argument("--lookback-days", type=positive_int, default=30)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--daily-dir", type=Path, default=None)
    parser.add_argument("--constituent-tree", type=Path, default=None)
    parser.add_argument("--macro-parquet", type=Path, default=None)
    parser.add_argument("--pit-parquet", type=Path, default=None)
    parser.add_argument("--pmi-path", type=Path, default=None)
    parser.add_argument("--event-calendar", type=Path, default=DEFAULT_EVENT_CALENDAR)
    parser.add_argument("--aaii-sentiment-parquet", type=Path, default=None)
    parser.add_argument("--news-sentiment-parquet", type=Path, default=None)
    parser.add_argument("--fomc-minutes-parquet", type=Path, default=None)
    parser.add_argument("--powell-speeches-parquet", type=Path, default=None)
    parser.add_argument("--cpi-vintages-parquet", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional artifact manifest to materialize before profiling.",
    )
    parser.add_argument(
        "--artifact-store",
        default=None,
        help="Optional artifact-store root override for --manifest.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Local data/raw root used for manifest materialization.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable profiling report JSON artifact.",
    )
    parser.add_argument("--allow-missing-constituent-files", action="store_true")
    args = parser.parse_args()
    if args.daily_dir is None:
        args.daily_dir = args.data_root / "daily_ohlcv"
    if args.constituent_tree is None:
        args.constituent_tree = args.data_root / FIXED_UNIVERSE_TREE_NAME
    if args.macro_parquet is None:
        args.macro_parquet = args.data_root / "macro" / "fred_macro_series.parquet"
    if args.pmi_path is None:
        args.pmi_path = default_pmi_path(args.data_root)
    if args.pit_parquet is None:
        args.pit_parquet = (
            args.data_root / "pit_constituents" / "sp500_ticker_intervals.parquet"
        )
    if args.aaii_sentiment_parquet is None:
        args.aaii_sentiment_parquet = (
            args.data_root / "sentiment" / "aaii_sentiment.parquet"
        )
    if args.news_sentiment_parquet is None:
        args.news_sentiment_parquet = (
            args.data_root / "news_sentiment" / "sf_fed_news_sentiment.parquet"
        )
    if args.fomc_minutes_parquet is None:
        args.fomc_minutes_parquet = (
            args.data_root / "fomc_minutes" / "fomc_minutes.parquet"
        )
    if args.powell_speeches_parquet is None:
        args.powell_speeches_parquet = (
            args.data_root / "powell_speeches" / "powell_speeches.parquet"
        )
    if args.cpi_vintages_parquet is None:
        args.cpi_vintages_parquet = (
            args.data_root / "macro_vintages" / "cpi_all_items_vintages.parquet"
        )
    return args


def main() -> int:
    args = _parse_args()
    materialize_if_requested(
        manifest_path=args.manifest,
        local_root=args.data_root,
        repo_root=REPO_ROOT,
        store_root=args.artifact_store,
        required_for="profile_engine_30d",
    )

    _require_path(args.config_path, kind="config path")
    _require_path(args.daily_dir, kind="daily OHLCV path")
    _require_path(args.constituent_tree, kind="constituent tree path")
    _require_path(args.macro_parquet, kind="macro parquet")
    _require_path(args.pit_parquet, kind="PIT parquet")

    engine = RegimeEngine(config_path=args.config_path)
    config = engine.config

    inputs = _load_profile_inputs(args, config=config)

    timer = StageTimer()
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *a: _timeout_handler(timer, *a))
    signal.alarm(RUN_TIMEOUT_SECONDS)
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
            "per_day_output_emission_loop_residual",
        ],
        StageTimer(),
        total_wall_clock,
    )
    stage_rows[-1] = (
        f"per_day_output_emission_loop_residual | {per_day_emission_total:.6f} | "
        f"{(per_day_emission_total / total_wall_clock * 100.0) if total_wall_clock > 0 else 0.0:6.2f}%"
    )
    for idx, stage_name in enumerate(
        [
            "build_market_context",
            "slice_context_to_recent_sessions",
            "build_feature_store_total",
            "build_axis_series_bundle",
            "build_transition_risk_series",
            "build_regime_timeline_total",
        ],
        start=1,
    ):
        seconds = timer.totals.get(stage_name, 0.0)
        pct = (seconds / total_wall_clock * 100.0) if total_wall_clock > 0 else 0.0
        stage_rows[idx] = f"{stage_name} | {seconds:.6f} | {pct:6.2f}%"

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

    verification_issues = _verify_invariants(
        timeline, feature_store, inputs.input_kwargs
    )
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
        f"fomc_minutes_source={args.fomc_minutes_parquet if args.fomc_minutes_parquet.exists() else '<absent>'}"
    )
    print(
        f"powell_speeches_source={args.powell_speeches_parquet if args.powell_speeches_parquet.exists() else '<absent>'}"
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
    for name in [
        "sector_etf_closes",
        "cross_asset_closes",
        "macro_series",
        "event_calendar",
        "aaii_sentiment",
        "news_sentiment",
        "implied_vol_30d",
        "central_bank_text_releases",
        "cpi_first_release",
        "pit_constituent_intervals",
        "constituent_ohlcv",
    ]:
        print(_input_status(name, inputs.input_kwargs[name]))
    print(f"pit_overlap_tickers_requested={len(inputs.constituent_tickers)}")
    print(f"constituent_tickers_loaded={len(inputs.constituent_ohlcv)}")
    print(f"missing_constituent_files={len(inputs.missing_constituent_paths)}")
    if inputs.missing_constituent_paths:
        for path in inputs.missing_constituent_paths[:20]:
            print(f"missing_constituent_file={path}")
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
