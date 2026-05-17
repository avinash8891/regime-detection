from __future__ import annotations

import argparse
import datetime as dt
import enum
import json
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import pandas as pd

from regime_detection.feature_store import FeatureStore
from regime_detection.models import ClassificationStatus
from regime_detection.models import RegimeOutput, RegimeTimeline

if TYPE_CHECKING:
    from scripts.profile_engine_30d import ProfileInputBundle, StageTimer

NON_CLASSIFIED_REPORTING_LABELS = set(get_args(ClassificationStatus)) - {"classified"}
PROFILE_INPUT_SEAM_NAMES = [
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
REPORTING_SUMMARY_FIELDS = [
    "trend_direction",
    "trend_character",
    "volatility_state",
    "breadth_state",
    "network_fragility",
    "volume_liquidity_state",
    "credit_funding_state",
    "credit_funding_state_proxy",
    "credit_funding_effective_state",
    "inflation_growth_state",
    "monetary_pressure_state",
]


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}

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


def _profile_input_seam_values(inputs: ProfileInputBundle) -> dict[str, Any]:
    return {name: getattr(inputs, name) for name in PROFILE_INPUT_SEAM_NAMES}


def _input_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, pd.DataFrame | pd.Series):
        return not value.empty
    return True


def _series_metric_summary(
    series: pd.Series | None, selected_dates: list[dt.date]
) -> dict[str, Any]:
    if series is None:
        return {"status": "missing", "non_null": 0, "total": len(selected_dates)}
    selected_index = pd.DatetimeIndex(selected_dates)
    aligned = series.reindex(selected_index)
    non_null = aligned.dropna()
    summary: dict[str, Any] = {
        "status": "present",
        "non_null": int(non_null.size),
        "total": int(aligned.size),
    }
    if not non_null.empty:
        summary["first_date"] = non_null.index[0].date().isoformat()
        summary["first_value"] = _json_safe_value(non_null.iloc[0])
        summary["last_date"] = non_null.index[-1].date().isoformat()
        summary["last_value"] = _json_safe_value(non_null.iloc[-1])
    return summary


def _feature_metric_summary_report(
    feature_store: FeatureStore | None, selected_dates: list[dt.date]
) -> dict[str, Any]:
    if feature_store is None or feature_store.trend_direction_v2 is None:
        return {}
    trend_direction_v2 = feature_store.trend_direction_v2
    return {
        "trend_direction_v2": {
            "sentiment_score": _series_metric_summary(
                trend_direction_v2.sentiment_score, selected_dates
            ),
            "news_sentiment_score": _series_metric_summary(
                trend_direction_v2.news_sentiment_score, selected_dates
            ),
            "sentiment_concordance": _series_metric_summary(
                trend_direction_v2.sentiment_concordance, selected_dates
            ),
        }
    }


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


def _json_safe_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        fields = getattr(type(value), "model_fields", None)
        if fields is not None:
            payload = {
                field_name: _json_safe_value(getattr(value, field_name))
                for field_name in fields
            }
        else:
            payload = _json_safe_value(
                value.model_dump(mode="json", exclude_none=False)
            )
        reporting_label = _reporting_label(value)
        if reporting_label is not None:
            payload["reporting_label"] = reporting_label
        return payload
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        payload = {
            key: _json_safe_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        reporting_label = _reporting_label(value)
        if reporting_label is not None:
            payload["reporting_label"] = reporting_label
        return payload
    return value


def _full_timeline_report(outputs: list[RegimeOutput]) -> list[dict[str, Any]]:
    return [_json_safe_value(output) for output in outputs]


def _label_summary_report(outputs: list[RegimeOutput]) -> dict[str, dict[str, dict[str, int]]]:
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for field_name in REPORTING_SUMMARY_FIELDS:
        active_counts: Counter[str] = Counter()
        reported_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        for output in outputs:
            value = getattr(output, field_name, None)
            if value is None:
                active_counts["missing"] += 1
                reported_counts["missing"] += 1
                status_counts["missing"] += 1
                continue
            active_label = getattr(value, "active_label", getattr(value, "label", None))
            active_counts[str(active_label or "missing")] += 1
            reported_counts[str(_reporting_label(value) or "missing")] += 1
            status_counts[
                str(getattr(value, "classification_status", "classified") or "missing")
            ] += 1
        summary[field_name] = {
            "active": _counter_dict(active_counts),
            "reported": _counter_dict(reported_counts),
            "status": _counter_dict(status_counts),
        }
    return summary


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
    inputs: ProfileInputBundle,
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
    input_values = _profile_input_seam_values(inputs)
    for seam_name, seam_value, deps in seam_expectations:
        if seam_value is None:
            missing = [
                dep for dep in deps if not _input_is_present(input_values.get(dep))
            ]
            if missing:
                issues.append(
                    f"{seam_name} is None; missing inputs: {', '.join(missing)}"
                )
    trend_direction_v2 = feature_store.trend_direction_v2
    if trend_direction_v2 is not None:
        sentiment_checks = [
            ("sentiment_score", ["aaii_sentiment"]),
            ("news_sentiment_score", ["news_sentiment"]),
            ("sentiment_concordance", ["aaii_sentiment", "news_sentiment"]),
        ]
        for metric_name, deps in sentiment_checks:
            if getattr(trend_direction_v2, metric_name) is not None:
                continue
            missing = [
                dep for dep in deps if not _input_is_present(input_values.get(dep))
            ]
            reason = ", ".join(missing) if missing else "not_computed"
            issues.append(
                f"trend_direction.{metric_name} missing; missing inputs: {reason}"
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
    feature_store: FeatureStore | None = None,
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

    input_values = _profile_input_seam_values(inputs)

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
                _input_status_report(name, input_values[name])
                for name in PROFILE_INPUT_SEAM_NAMES
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
        "label_summary": _label_summary_report(timeline.outputs),
        "feature_metrics": _feature_metric_summary_report(
            feature_store, inputs.selected_dates
        ),
        "full_timeline": _full_timeline_report(timeline.outputs),
        "trailing_v2_field_status": _trailing_v2_status_report(timeline.outputs[-1]),
        "verification_issues": verification_issues,
    }


def _write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
