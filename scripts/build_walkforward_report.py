#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from regime_detection.calendar import nyse_calendar

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "src" / "regime_detection" / "configs" / "core3-v1.0.0.yaml"

LABEL_COLUMNS = [
    "trend_direction_active",
    "trend_character_active",
    "volatility_state_active",
    "breadth_state_active",
    "transition_risk_state",
]

HYSTERESIS_BY_COLUMN = {
    "trend_direction_active": 3,
    "trend_character_active": 3,
    "volatility_state_active": 2,
    "breadth_state_active": 2,
    "transition_risk_state": 3,
}

BETTER_DIRECTION = {
    "max_drawdown": "lower",
    "false_switch_rate": "lower",
    "average_detection_lag": "lower",
    "wrong_environment_trades": "lower",
    "sharpe": "higher",
    "strategy_return": "higher",
    "hit_rate": "higher",
    "strategy_pnl_improvement": "higher",
}


def _load_hysteresis_days() -> dict[str, int]:
    if not CONFIG_PATH.exists():
        return HYSTERESIS_BY_COLUMN.copy()
    data = yaml.safe_load(CONFIG_PATH.read_text())
    hysteresis = data.get("hysteresis", {})
    return {
        "trend_direction_active": int(
            hysteresis.get("trend_direction_deescalation_days", 3)
        ),
        "trend_character_active": int(
            hysteresis.get("trend_character_deescalation_days", 3)
        ),
        "volatility_state_active": int(
            hysteresis.get("volatility_deescalation_days", 2)
        ),
        "breadth_state_active": int(hysteresis.get("breadth_deescalation_days", 2)),
        "transition_risk_state": int(hysteresis.get("composite_deescalation_days", 3)),
    }


def _load_summary(output_root: Path) -> pd.DataFrame:
    summary_path = output_root / "reports" / "walkforward_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"walkforward summary not found: {summary_path}")
    pandas_module: Any = pd
    df = pandas_module.read_csv(summary_path)
    parsed_date_values = cast(Any, pd.to_datetime(df["as_of_date"]))
    parsed_dates = pd.DatetimeIndex(parsed_date_values)
    df["as_of_date"] = [pd.Timestamp(value).date() for value in parsed_dates.tolist()]
    return df.sort_values("as_of_date").reset_index(drop=True)


def _load_runs_from_db(output_root: Path) -> pd.DataFrame:
    db_path = output_root / "regime_walkforward.db"
    if not db_path.exists():
        raise FileNotFoundError(f"walkforward db not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        pandas_module: Any = pd
        df = pandas_module.read_sql_query(
            "SELECT as_of_date, status, failure_reason, engine_version, config_version, input_archive_path, output_path FROM runs ORDER BY as_of_date",
            conn,
        )
    parsed_date_values = cast(Any, pd.to_datetime(df["as_of_date"]))
    parsed_dates = pd.DatetimeIndex(parsed_date_values)
    df["as_of_date"] = [pd.Timestamp(value).date() for value in parsed_dates.tolist()]
    return df


def _expected_sessions(summary_df: pd.DataFrame) -> list[date]:
    start_date = min(summary_df["as_of_date"])
    end_date = max(summary_df["as_of_date"])
    calendar: Any = nyse_calendar()
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    schedule_index = pd.DatetimeIndex(pd.Index(schedule.index))
    return [pd.Timestamp(ts).date() for ts in schedule_index.tolist()]


def _missing_sessions(summary_df: pd.DataFrame) -> list[str]:
    expected = set(_expected_sessions(summary_df))
    actual = set(summary_df["as_of_date"])
    return [d.isoformat() for d in sorted(expected - actual)]


def _label_distribution(success_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for col in LABEL_COLUMNS:
        if col in success_df.columns:
            counts = cast(
                dict[Any, int],
                cast(Any, success_df[col].value_counts(dropna=False)).to_dict(),
            )
            out[col] = {str(k): int(v) for k, v in counts.items()}
    return out


def _longest_runs(series: pd.Series) -> dict[str, int]:
    runs: dict[str, int] = {}
    current_label: str | None = None
    current_len = 0
    for raw_value in series.tolist():
        label = "null" if pd.isna(raw_value) else str(raw_value)
        if label == current_label:
            current_len += 1
        else:
            if current_label is not None:
                runs[current_label] = max(runs.get(current_label, 0), current_len)
            current_label = label
            current_len = 1
    if current_label is not None:
        runs[current_label] = max(runs.get(current_label, 0), current_len)
    return runs


def _switch_count(series: pd.Series) -> int:
    clean = ["null" if pd.isna(value) else str(value) for value in series.tolist()]
    if not clean:
        return 0
    switches = sum(1 for idx in range(1, len(clean)) if clean[idx] != clean[idx - 1])
    return switches


def _false_switch_count(series: pd.Series, horizon: int) -> int:
    clean = ["null" if pd.isna(value) else str(value) for value in series.tolist()]
    count = 0
    for idx in range(1, len(clean) - 1):
        prev_label = clean[idx - 1]
        current = clean[idx]
        if current == prev_label:
            continue
        window_end = min(len(clean), idx + horizon + 1)
        if prev_label in clean[idx + 1 : window_end]:
            count += 1
    return count


def _unknown_stretch(series: pd.Series) -> int:
    max_len = 0
    cur = 0
    for raw_value in series.tolist():
        label = "null" if pd.isna(raw_value) else str(raw_value)
        if label == "unknown":
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return max_len


def _series_summaries(
    success_df: pd.DataFrame, hysteresis_days: dict[str, int]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col in LABEL_COLUMNS:
        if col not in success_df.columns:
            continue
        series = success_df[col]
        switch_count = _switch_count(series)
        false_switch_count = _false_switch_count(series, hysteresis_days[col])
        out[col] = {
            "longest_runs": _longest_runs(series),
            "switch_count": switch_count,
            "false_switch_count": false_switch_count,
            "false_switch_rate": (
                0.0 if switch_count == 0 else false_switch_count / switch_count
            ),
            "max_unknown_stretch": _unknown_stretch(series),
        }
    return out


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text())


def _baseline_comparison(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    metrics = payload.get("metrics", {})
    improved_metrics: list[str] = []
    materially_worse_metrics: list[str] = []
    comparisons: dict[str, dict[str, Any]] = {}
    for metric, values in metrics.items():
        with_regime = values["with_regime_gating"]
        baseline = values["no_regime_baseline"]
        direction = BETTER_DIRECTION.get(metric)
        if direction == "lower":
            improved = with_regime < baseline
            materially_worse = with_regime > baseline
            delta = with_regime - baseline
        else:
            improved = with_regime > baseline
            materially_worse = with_regime < baseline
            delta = with_regime - baseline
        if improved:
            improved_metrics.append(metric)
        if materially_worse:
            materially_worse_metrics.append(metric)
        comparisons[metric] = {
            "with_regime_gating": with_regime,
            "no_regime_baseline": baseline,
            "delta": delta,
            "better_direction": direction or "higher",
        }
    return {
        "comparisons": comparisons,
        "improved_metrics": sorted(improved_metrics),
        "materially_worse_metrics": sorted(materially_worse_metrics),
        "all_metrics_materially_worse": bool(comparisons)
        and len(materially_worse_metrics) == len(comparisons),
    }


def _failure_reasons(
    *,
    summary_df: pd.DataFrame,
    missing_sessions: list[str],
    golden_results: dict[str, Any] | None,
    baseline_comparison: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if summary_df["status"].eq("failure").any():
        reasons.append("run_failures_present")
    if missing_sessions:
        reasons.append("missing_sessions")
    if golden_results is None:
        reasons.append("missing_golden_results")
    elif not bool(golden_results.get("all_passed")):
        reasons.append("golden_results_failed")
    if baseline_comparison is None:
        reasons.append("missing_baseline_metrics")
    elif baseline_comparison["all_metrics_materially_worse"]:
        reasons.append("materially_worse_than_baseline")
    return reasons


def _build_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Historical Walk-Forward Analysis",
        "",
        f"- status: `{analysis['status']}`",
        f"- session_count: `{analysis['session_count']}`",
        f"- success_count: `{analysis['success_count']}`",
        f"- failure_count: `{analysis['failure_count']}`",
        f"- missing_session_count: `{len(analysis['missing_sessions'])}`",
        "",
        "## Failure Reasons",
        "",
    ]
    if analysis["failure_reasons"]:
        lines.extend([f"- `{reason}`" for reason in analysis["failure_reasons"]])
    else:
        lines.append("- none")

    lines.extend(["", "## Label Distributions", ""])
    for key, value in analysis["label_distributions"].items():
        lines.extend(
            [
                f"### {key}",
                "",
                "```json",
                json.dumps(value, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )

    lines.extend(["## Series Summaries", ""])
    for key, value in analysis["series_summaries"].items():
        lines.extend(
            [
                f"### {key}",
                "",
                "```json",
                json.dumps(value, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Golden Results",
            "",
            "```json",
            json.dumps(analysis["golden_results"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    lines.extend(
        [
            "## Baseline Comparison",
            "",
            "```json",
            json.dumps(analysis["baseline_comparison"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_walkforward_report(
    *,
    output_root: Path,
    golden_results_path: Path | None = None,
    baseline_metrics_path: Path | None = None,
) -> dict[str, Any]:
    summary_df = _load_summary(output_root)
    runs_df = _load_runs_from_db(output_root)
    success_df = summary_df[summary_df["status"] == "success"].copy()
    missing_sessions = _missing_sessions(summary_df)
    hysteresis_days = _load_hysteresis_days()

    golden_results = _load_optional_json(golden_results_path)
    baseline_comparison = _baseline_comparison(
        _load_optional_json(baseline_metrics_path)
    )
    failure_reasons = _failure_reasons(
        summary_df=summary_df,
        missing_sessions=missing_sessions,
        golden_results=golden_results,
        baseline_comparison=baseline_comparison,
    )

    analysis = {
        "status": "pass" if not failure_reasons else "fail",
        "session_count": int(len(summary_df)),
        "success_count": int(summary_df["status"].eq("success").sum()),
        "failure_count": int(summary_df["status"].eq("failure").sum()),
        "missing_sessions": missing_sessions,
        "failure_reasons": failure_reasons,
        "engine_version": (
            str(runs_df["engine_version"].dropna().iloc[0])
            if not runs_df.empty
            else None
        ),
        "config_version": (
            str(runs_df["config_version"].dropna().iloc[0])
            if not runs_df.empty
            else None
        ),
        "label_distributions": _label_distribution(success_df),
        "series_summaries": _series_summaries(success_df, hysteresis_days),
        "golden_results": golden_results
        or {"all_passed": False, "reason": "not_provided"},
        "baseline_comparison": baseline_comparison
        or {
            "all_metrics_materially_worse": False,
            "reason": "not_provided",
            "improved_metrics": [],
        },
    }

    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = reports_dir / "walkforward_analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = reports_dir / "walkforward_report.md"
    report_path.write_text(_build_markdown(analysis), encoding="utf-8")
    return analysis


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a walk-forward qualification report from archived artifacts."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--golden-results", type=Path, default=None)
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_walkforward_report(
        output_root=args.output_root,
        golden_results_path=args.golden_results,
        baseline_metrics_path=args.baseline_metrics,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
