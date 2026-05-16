#!/usr/bin/env python3
"""Generate Layer 2 wiring/label audit artifacts from the 30-day runner path."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import Counter
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
from regime_data_fetch.materialization import materialize_if_requested
from regime_detection.axis_series import build_axis_series_bundle
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import FeatureStore, build_feature_store
from regime_detection.fragility_universe import CROSS_ASSET_SYMBOLS, SECTOR_ETFS
from regime_detection.market_context import (
    MarketContext,
    build_market_context,
    slice_context_to_recent_sessions,
)
from scripts._v2_calibration_helpers import (
    default_pmi_path,
    load_close_dict,
    load_macro_series,
    load_market_data,
    positive_int,
)
from scripts.profile_engine_30d import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CONSTITUENT_TREE,
    DEFAULT_DAILY_DIR,
    DEFAULT_EVENT_CALENDAR,
    DEFAULT_MACRO_PARQUET,
    DEFAULT_PIT_PARQUET,
    _build_required_sessions,
    _load_constituent_ohlcv_from_tree,
    _load_optional_aaii_sentiment,
    _load_optional_central_bank_text_releases,
    _load_optional_cpi_first_release,
    _load_optional_event_calendar,
    _load_optional_news_sentiment,
)


LAYER2_FEATURES: dict[str, tuple[str, ...]] = {
    "monetary_pressure": (
        "yield_change_zscore_2y_63d",
        "yield_change_zscore_10y_63d",
        "broad_usd_index_zscore_63d",
        "yield_change_zscore_21d_2y",
        "yield_change_zscore_21d_10y",
        "central_bank_text_score",
    ),
    "credit_funding": (
        "hy_oas_63d",
        "ig_oas_63d",
        "hy_oas_percentile_504d",
        "hy_oas_slope_21d",
        "ig_oas_slope_21d",
        "hy_tr_differential_63d",
        "ig_tr_differential_63d",
        "hy_tr_differential_percentile_504d",
        "hy_tr_differential_slope_21d",
        "ig_tr_differential_slope_21d",
        "kre_spy_ratio",
        "kre_spy_slope_63d",
        "nfci_daily_carried",
        "sofr_iorb_spread",
        "sofr_iorb_slope_21d",
        "broad_usd_index_zscore_21d",
        "spy_21d_return",
        "tlt_21d_return",
    ),
    "inflation_growth": (
        "cpi_3m_change_pct",
        "cpi_6m_change_pct",
        "cpi_6m_change_pct_slope_21d",
        "inflation_surprise_zscore",
        "pmi_manufacturing",
        "pmi_manufacturing_slope_21d",
        "aggregate_forward_eps_revision_direction_4w",
        "commodity_return_63d",
        "treasury_10y_yield_slope_21d",
        "cyclical_defensive_ratio",
        "cyclical_defensive_slope_21d",
        "spy_21d_return",
        "tlt_21d_return",
    ),
}


def _json_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _reporting_label(output: Any | None) -> str:
    if output is None:
        return "not_wired"
    reporting_label = getattr(output, "reporting_label", None)
    if reporting_label is not None:
        return str(reporting_label)
    classification_status = getattr(output, "classification_status", None)
    if classification_status != "classified":
        return str(classification_status or "not_wired")
    active_label = getattr(output, "active_label", None)
    if active_label is not None:
        return str(active_label)
    return str(getattr(output, "label", "not_wired"))


def _feature_row(
    *,
    axis: str,
    metric: str,
    series: pd.Series | None,
    selected_dates: list[dt.date],
) -> dict[str, Any]:
    if series is None:
        return {
            "axis": axis,
            "metric": metric,
            "role": "feature",
            "present_days": 0,
            "total_days": len(selected_dates),
            "status": "missing",
            "min": None,
            "median": None,
            "max": None,
            "true_days": None,
        }

    selected_index = pd.to_datetime(selected_dates)
    aligned = series.reindex(selected_index)
    present = aligned.notna()
    non_null = aligned[present]
    true_days: int | None = None
    min_value: float | None = None
    median_value: float | None = None
    max_value: float | None = None
    if len(non_null) > 0:
        if pd.api.types.is_bool_dtype(non_null):
            true_days = int(non_null.astype(bool).sum())
        else:
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numeric) > 0:
                min_value = _finite_or_none(numeric.min())
                median_value = _finite_or_none(numeric.median())
                max_value = _finite_or_none(numeric.max())
    return {
        "axis": axis,
        "metric": metric,
        "role": "feature",
        "present_days": int(present.sum()),
        "total_days": len(selected_dates),
        "status": "ok" if int(present.sum()) == len(selected_dates) else "missing",
        "min": min_value,
        "median": median_value,
        "max": max_value,
        "true_days": true_days,
    }


def build_wiring_presence_rows(
    *,
    feature_store: FeatureStore,
    selected_dates: list[dt.date],
) -> list[dict[str, Any]]:
    feature_objects = {
        "monetary_pressure": feature_store.monetary,
        "credit_funding": feature_store.credit_funding,
        "inflation_growth": feature_store.inflation_growth,
    }
    rows: list[dict[str, Any]] = []
    for axis, metrics in LAYER2_FEATURES.items():
        features = feature_objects[axis]
        for metric in metrics:
            series = getattr(features, metric, None) if features is not None else None
            rows.append(
                _feature_row(
                    axis=axis,
                    metric=metric,
                    series=series,
                    selected_dates=selected_dates,
                )
            )
    return rows


def _summarize_output_series(
    series: dict[dt.date, Any] | None,
    selected_dates: list[dt.date],
) -> dict[str, Any]:
    reported: Counter[str] = Counter()
    active: Counter[str | None] = Counter()
    raw: Counter[str | None] = Counter()
    stable: Counter[str | None] = Counter()
    quality_status: Counter[str | None] = Counter()
    quality_reasons: Counter[str | None] = Counter()
    classification_status: Counter[str | None] = Counter()
    rule_evidence_present: Counter[str] = Counter()
    source_used: Counter[str | None] = Counter()

    for day in selected_dates:
        output = series.get(day) if series is not None else None
        reported[_reporting_label(output)] += 1
        active[output.active_label if output is not None else None] += 1
        raw[output.raw_label if output is not None else None] += 1
        stable[output.stable_label if output is not None else None] += 1
        if output is None:
            quality_status[None] += 1
            classification_status[None] += 1
            continue
        quality_status[output.data_quality.status] += 1
        quality_reasons[output.data_quality.reason] += 1
        classification_status[output.classification_status] += 1
        evidence = output.evidence or {}
        for metric, value in dict(evidence.get("rule_evidence", {})).items():
            if value is not None and not (
                isinstance(value, float) and math.isnan(value)
            ):
                rule_evidence_present[str(metric)] += 1
        if "source_used" in evidence:
            source_used[evidence.get("source_used")] += 1

    summary = {
        "reported": _json_counter(reported),
        "active": _json_counter(active),
        "raw": _json_counter(raw),
        "stable": _json_counter(stable),
        "data_quality_status": _json_counter(quality_status),
        "data_quality_reasons": _json_counter(quality_reasons),
        "classification_status": _json_counter(classification_status),
        "rule_evidence_present": _json_counter(rule_evidence_present),
    }
    if source_used:
        summary["source_used"] = _json_counter(source_used)
    return summary


def build_label_rule_summary(
    *,
    axis_bundle: Any,
    selected_dates: list[dt.date],
    missing_constituent_files: int,
) -> dict[str, Any]:
    series_by_name = {
        "monetary_pressure_state": axis_bundle.monetary_pressure_state,
        "credit_funding_state": axis_bundle.credit_funding,
        "credit_funding_state_proxy": axis_bundle.credit_funding_proxy,
        "credit_funding_effective_state": axis_bundle.credit_funding_effective,
        "inflation_growth_state": axis_bundle.inflation_growth,
    }
    return {
        "run": {
            "days": len(selected_dates),
            "selected_start": selected_dates[0].isoformat(),
            "selected_end": selected_dates[-1].isoformat(),
            "missing_constituent_files": missing_constituent_files,
        },
        "axes": {
            name: _summarize_output_series(series, selected_dates)
            for name, series in series_by_name.items()
        },
    }


def _build_current_layer2_state(
    args: argparse.Namespace,
) -> tuple[MarketContext, FeatureStore, Any, list[dt.date], int]:
    engine = RegimeEngine(config_path=args.config_path)
    config = engine.config
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
        config, len(bootstrap_context.sessions), args.lookback_days
    )
    working_start_date = bootstrap_context.sessions[-required_sessions]

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
        args.macro_parquet,
        args.pmi_path if args.pmi_path.exists() else None,
    )
    pit_constituent_intervals = read_pit_intervals(args.pit_parquet)
    constituent_ohlcv, _constituent_tickers, missing_constituent_paths = (
        _load_constituent_ohlcv_from_tree(
            args.constituent_tree,
            pit_constituent_intervals,
            start_date=working_start_date,
            end_date=end_date,
            allow_missing_files=args.allow_missing_constituent_files,
        )
    )

    context = build_market_context(
        end_date=end_date,
        market_data=market_data,
        config=config,
        sector_etf_closes=sector_etf_closes,
        cross_asset_closes=cross_asset_closes,
        macro_series=macro_series,
        event_calendar=_load_optional_event_calendar(args.event_calendar),
        aaii_sentiment=_load_optional_aaii_sentiment(args.aaii_sentiment_parquet),
        implied_vol_30d=macro_series.get("implied_vol_30d"),
        central_bank_text_releases=_load_optional_central_bank_text_releases(
            fomc_path=args.fomc_minutes_parquet,
            powell_path=args.powell_speeches_parquet,
        ),
        cpi_first_release=_load_optional_cpi_first_release(args.cpi_vintages_parquet),
        news_sentiment=_load_optional_news_sentiment(args.news_sentiment_parquet),
        pit_constituent_intervals=pit_constituent_intervals,
        constituent_ohlcv=constituent_ohlcv,
    )
    working_context = slice_context_to_recent_sessions(
        context=context,
        required_sessions=required_sessions,
    )
    feature_store = build_feature_store(
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
    axis_bundle = build_axis_series_bundle(
        context=working_context,
        feature_store=feature_store,
    )
    selected_dates = list(working_context.sessions[-args.lookback_days :])
    return (
        working_context,
        feature_store,
        axis_bundle,
        selected_dates,
        len(missing_constituent_paths),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Layer 2 feature/label audit artifacts from profile_engine_30d inputs."
    )
    parser.add_argument("--lookback-days", type=positive_int, default=30)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument(
        "--constituent-tree", type=Path, default=DEFAULT_CONSTITUENT_TREE
    )
    parser.add_argument("--macro-parquet", type=Path, default=DEFAULT_MACRO_PARQUET)
    parser.add_argument("--pit-parquet", type=Path, default=DEFAULT_PIT_PARQUET)
    parser.add_argument("--pmi-path", type=Path, default=None)
    parser.add_argument("--event-calendar", type=Path, default=DEFAULT_EVENT_CALENDAR)
    parser.add_argument("--aaii-sentiment-parquet", type=Path, default=None)
    parser.add_argument("--news-sentiment-parquet", type=Path, default=None)
    parser.add_argument("--fomc-minutes-parquet", type=Path, default=None)
    parser.add_argument("--powell-speeches-parquet", type=Path, default=None)
    parser.add_argument("--cpi-vintages-parquet", type=Path, default=None)
    parser.add_argument("--allow-missing-constituent-files", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional artifact manifest to materialize before audit.",
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
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / ".context")
    parser.add_argument("--stamp", default=dt.date.today().strftime("%Y%m%d"))
    args = parser.parse_args()
    if args.pmi_path is None:
        args.pmi_path = default_pmi_path(args.data_root)
    return args


def main() -> int:
    args = _parse_args()
    materialize_if_requested(
        manifest_path=args.manifest,
        local_root=args.data_root,
        repo_root=REPO_ROOT,
        store_root=args.artifact_store,
        required_for="audit_layer2_30d",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _working_context, feature_store, axis_bundle, selected_dates, missing_files = (
        _build_current_layer2_state(args)
    )
    wiring_rows = build_wiring_presence_rows(
        feature_store=feature_store,
        selected_dates=selected_dates,
    )
    label_summary = build_label_rule_summary(
        axis_bundle=axis_bundle,
        selected_dates=selected_dates,
        missing_constituent_files=missing_files,
    )
    wiring_path = args.out_dir / f"layer2_wiring_presence_audit_{args.stamp}.csv"
    summary_path = args.out_dir / f"layer2_label_rule_summary_{args.stamp}.json"
    pd.DataFrame(wiring_rows).to_csv(wiring_path, index=False)
    summary_path.write_text(json.dumps(label_summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {wiring_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
