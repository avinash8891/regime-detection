#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any
from contextlib import closing

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

MIN_OOS_SESSIONS = 252
LABEL_DOMINANCE_THRESHOLD = 0.95
UNKNOWN_STRETCH_THRESHOLD = 20
TRANSITION_RISK_WARNING_STATES = frozenset(
    {
        "watch",
        "weakening",
        "transition_warning",
        "high_transition_risk",
        "crisis",
        "bear_stress",
        "fragile_bull",
        "recovery_attempt",
    }
)
# F-050: configured crash windows where the §6 walk-forward MUST surface a crisis-
# equivalent label, else §8 "defensible label distribution" is violated. Drawn from
# the spec §9.4 stress dates / golden-date crisis rows (Volmageddon, the Q4-2018
# selloff, the COVID crash, and the Jun-2022 bear capitulation). Each window is the
# canonical multi-session episode, not a single date, so a near-date predicate
# boundary does not hide the absence. A window only arms the check when OOS success
# rows actually cover it — we never flag a window the walk-forward never reached.
CRASH_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("volmageddon_2018", "2018-02-05", "2018-02-12"),
    ("q4_2018_selloff", "2018-12-10", "2018-12-26"),
    ("covid_crash_2020", "2020-02-24", "2020-03-23"),
    ("jun_2022_capitulation", "2022-06-13", "2022-06-17"),
)
# A crisis-equivalent label on EITHER the volatility or transition-risk axis satisfies
# the window (crisis_vol is the §3 emergency-override volatility label; crisis is the
# §9 transition-risk crisis state).
CRISIS_EQUIVALENT_VOLATILITY_LABELS = frozenset({"crisis_vol"})
CRISIS_EQUIVALENT_TRANSITION_LABELS = frozenset({"crisis"})

REQUIRED_SUMMARY_COLUMNS = frozenset(
    {
        "as_of_date",
        "run_timestamp",  # F-019 / §5: per-artifact provenance
        "status",
        *LABEL_COLUMNS,
        "transition_risk_score",
        "transition_risk_primary_drivers",
        "transition_risk_triggered_rules",
        "transition_risk_data_quality_status",
        "transition_risk_axis_switch_count",
        "transition_risk_recent_axis_switch_count",
    }
)

HYSTERESIS_BY_COLUMN = {
    "trend_direction_active": 3,
    "trend_character_active": 3,
    "volatility_state_active": 2,
    "breadth_state_active": 2,
    "transition_risk_state": 3,
}

# F-016: relative-delta magnitude below which a metric change vs the no-regime baseline
# is a TIE (not "material") — so an infinitesimal delta cannot flip a metric to
# improved/worse, and a tie cannot rescue an otherwise-regressed run.
_BASELINE_MATERIALITY_REL_EPSILON = 0.01

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
    df = pd.read_csv(summary_path)
    df = df.assign(as_of_date=pd.to_datetime(df["as_of_date"]).dt.date)
    return df.sort_values("as_of_date").reset_index(drop=True)


def _missing_summary_columns(summary_df: pd.DataFrame) -> list[str]:
    return sorted(REQUIRED_SUMMARY_COLUMNS - set(summary_df.columns))


def _load_runs_from_db(output_root: Path) -> pd.DataFrame:
    db_path = output_root / "regime_walkforward.db"
    if not db_path.exists():
        raise FileNotFoundError(f"walkforward db not found: {db_path}")
    with closing(sqlite3.connect(db_path)) as conn:
        df = pd.read_sql_query(
            "SELECT as_of_date, status, failure_reason, engine_version, config_version, run_timestamp, input_archive_path, output_path FROM runs ORDER BY as_of_date",
            conn,
        )
    df = df.assign(as_of_date=pd.to_datetime(df["as_of_date"]).dt.date)
    return df


def _expected_sessions(summary_df: pd.DataFrame) -> list[date]:
    start_date = min(summary_df["as_of_date"])
    end_date = max(summary_df["as_of_date"])
    schedule = nyse_calendar().schedule(start_date=start_date, end_date=end_date)
    return list(schedule.index.date)


def _missing_sessions(summary_df: pd.DataFrame) -> list[str]:
    expected = set(_expected_sessions(summary_df))
    actual = set(summary_df["as_of_date"])
    return [d.isoformat() for d in sorted(expected - actual)]


def _label_distribution(success_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for col in LABEL_COLUMNS:
        if col in success_df.columns:
            out[col] = {
                str(k): int(v)
                for k, v in success_df[col].value_counts(dropna=False).to_dict().items()
            }
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
    clean = series.fillna("null").astype(str)
    if clean.empty:
        return 0
    return int(clean.ne(clean.shift(1)).sum() - 1)


def _false_switch_count(series: pd.Series, horizon: int) -> int:
    clean = series.fillna("null").astype(str).tolist()
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


def _json_nan_paths(value: Any, prefix: str) -> list[str]:
    if isinstance(value, float) and math.isnan(value):
        return [prefix]
    if isinstance(value, dict):
        out: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_json_nan_paths(child, child_prefix))
        return out
    if isinstance(value, list):
        out = []
        for idx, child in enumerate(value):
            out.extend(_json_nan_paths(child, f"{prefix}[{idx}]"))
        return out
    return []


def _nan_leakage(
    summary_df: pd.DataFrame, runs_df: pd.DataFrame, output_root: Path
) -> list[str]:
    leaks: list[str] = []
    for col in summary_df.columns:
        if col == "as_of_date" or not pd.api.types.is_numeric_dtype(summary_df[col]):
            continue
        for _, row in summary_df.loc[summary_df[col].isna()].iterrows():
            leaks.append(f"summary.{col}@{row['as_of_date'].isoformat()}")

    # F-020: §6 contract — degraded/missing data must surface as the explicit `unknown`
    # / `insufficient_history` LABELS, never as an empty/None/NaN cell. The numeric scan
    # above skips string label columns, so a silently-blank label would otherwise pass.
    # Flag any successful-row LABEL_COLUMN value that is missing or blank.
    success_df = (
        summary_df[summary_df["status"] == "success"]
        if "status" in summary_df.columns
        else summary_df
    )
    for col in LABEL_COLUMNS:
        if col not in summary_df.columns:
            continue
        for _, row in success_df.iterrows():
            value = row[col]
            text = ("" if value is None else str(value)).strip().lower()
            if text in ("", "nan", "none", "null"):
                leaks.append(
                    f"summary.{col}@{row['as_of_date'].isoformat()}:label_contract_violation"
                )

    for _, row in runs_df.iterrows():
        output_path = row.get("output_path")
        if pd.isna(output_path) or output_path is None:
            continue
        path = Path(str(output_path))
        if not path.is_absolute():
            path = output_root / path
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        as_of = row["as_of_date"].isoformat()
        for leak in _json_nan_paths(payload, ""):
            leaks.append(f"output.{leak}@{as_of}")
    return sorted(leaks)


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


def _frozen_versions(runs_df: pd.DataFrame) -> dict[str, Any]:
    engine_versions = sorted(
        str(v) for v in runs_df["engine_version"].dropna().unique()
    )
    config_versions = sorted(
        str(v) for v in runs_df["config_version"].dropna().unique()
    )
    return {
        "engine_versions": engine_versions,
        "config_versions": config_versions,
        "engine_version": engine_versions[0] if len(engine_versions) == 1 else None,
        "config_version": config_versions[0] if len(config_versions) == 1 else None,
        "is_single_pair": len(engine_versions) == 1 and len(config_versions) == 1,
    }


def _per_date_provenance(runs_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in runs_df.sort_values("as_of_date").iterrows():
        rows.append(
            {
                "as_of_date": row["as_of_date"].isoformat(),
                # F-019 / §5: surface the run_timestamp the runner records in the DB so
                # every per-date provenance row carries all five mandated fields.
                "run_timestamp": (
                    None if pd.isna(row["run_timestamp"]) else str(row["run_timestamp"])
                ),
                "engine_version": (
                    None
                    if pd.isna(row["engine_version"])
                    else str(row["engine_version"])
                ),
                "config_version": (
                    None
                    if pd.isna(row["config_version"])
                    else str(row["config_version"])
                ),
                "input_archive_path": (
                    None
                    if pd.isna(row["input_archive_path"])
                    else str(row["input_archive_path"])
                ),
                "output_path": (
                    None if pd.isna(row["output_path"]) else str(row["output_path"])
                ),
            }
        )
    return rows


def _expected_golden_dates() -> list[str]:
    golden_path = REPO_ROOT / "tests" / "fixtures" / "derived" / "golden_dates.yaml"
    data = yaml.safe_load(golden_path.read_text())
    # The walkforward gate is the core/V1-replay golden gate. expected_v2_fields
    # rows run through the separate V2 harness and are not walkforward-gated.
    return sorted(str(row["as_of_date"]) for row in data["rows"] if "expected" in row)


def _single_golden_gate_reasons(
    golden_results: Any, runs_df: pd.DataFrame
) -> list[str]:
    if not isinstance(golden_results, dict):
        return ["golden_results_missing_before_after"]

    reasons: list[str] = []
    if not bool(golden_results.get("all_passed")):
        reasons.append("golden_results_failed")

    result_dates = sorted(
        str(row.get("as_of_date")) for row in golden_results.get("results", [])
    )
    if result_dates != _expected_golden_dates():
        reasons.append("golden_results_dates_mismatch")

    frozen_versions = _frozen_versions(runs_df)
    expected_engine = frozen_versions["engine_version"]
    expected_config = frozen_versions["config_version"]
    if (
        golden_results.get("engine_version") != expected_engine
        or golden_results.get("config_version") != expected_config
    ):
        reasons.append("golden_results_version_mismatch")
    return reasons


def _golden_gate_reasons(
    golden_results: dict[str, Any] | None, runs_df: pd.DataFrame
) -> list[str]:
    if golden_results is None:
        return ["missing_golden_results"]
    if "pre_batch" not in golden_results or "post_batch" not in golden_results:
        return ["golden_results_missing_before_after"]

    reasons: list[str] = []
    reasons.extend(_single_golden_gate_reasons(golden_results["pre_batch"], runs_df))
    reasons.extend(_single_golden_gate_reasons(golden_results["post_batch"], runs_df))
    return sorted(set(reasons))


def _replay_gate_reasons(
    replay_results: dict[str, Any] | None, runs_df: pd.DataFrame
) -> list[str]:
    """§6 replay gate (F-001). The replay verdict must be produced by
    run_walkforward_replay_check (recompute from archived inputs vs stored output),
    cover every SUCCESSFUL walk-forward date, and be bound to the frozen
    engine/config pair — not merely carry a truthy all_passed."""
    if replay_results is None:
        return ["missing_replay_verification"]

    success_dates = sorted(
        row["as_of_date"].isoformat()
        for _, row in runs_df.iterrows()
        if str(row.get("status")) == "success"
    )
    # CR-005: an empty batch (zero successful runs) had nothing to replay. The producer's
    # `all_passed = bool([]) and …` is False, but reporting that as replay_mismatch_detected
    # conflates "nothing to verify" with "verification failed". Emit a distinct reason
    # (the build still fails — insufficient_oos_sessions also fires).
    if not success_dates:
        return ["no_successful_runs_to_replay"]
    reasons: list[str] = []
    if not bool(replay_results.get("all_passed")):
        reasons.append("replay_mismatch_detected")
    result_dates = sorted(
        str(item.get("as_of_date")) for item in replay_results.get("results", [])
    )
    # CR-010: result_dates is the producer's snapshot (the success set when
    # run_walkforward_replay_check ran); success_dates is recomputed live from runs_df at
    # report-build time. A DB status flip between the two steps fires replay_dates_mismatch
    # — this is INTENDED fail-closed: a verdict that does not cover every currently-
    # successful date is stale and must not promote. The §6 ordering contract is to run
    # run_walkforward_replay_check immediately before build_walkforward_report (re-run the
    # producer after any late/retried date) so the two snapshots align.
    if result_dates != success_dates:
        reasons.append("replay_dates_mismatch")

    frozen_versions = _frozen_versions(runs_df)
    if (
        replay_results.get("engine_version") != frozen_versions["engine_version"]
        or replay_results.get("config_version") != frozen_versions["config_version"]
    ):
        reasons.append("replay_version_mismatch")
    return sorted(set(reasons))


def _baseline_comparison(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    metrics = payload.get("metrics", {})
    improved_metrics: list[str] = []
    materially_worse_metrics: list[str] = []
    unknown_metrics: list[str] = []
    comparisons: dict[str, dict[str, Any]] = {}
    for metric, values in metrics.items():
        with_regime = values["with_regime_gating"]
        baseline = values["no_regime_baseline"]
        direction = BETTER_DIRECTION.get(metric)
        if direction is None:
            unknown_metrics.append(metric)
            comparisons[metric] = {
                "with_regime_gating": with_regime,
                "no_regime_baseline": baseline,
                "delta": with_regime - baseline,
                "relative_delta": (
                    None if baseline == 0 else (with_regime - baseline) / baseline
                ),
                "better_direction": None,
            }
            continue
        delta = with_regime - baseline
        relative_delta = None if baseline == 0 else delta / baseline
        # F-016: "materially" needs a magnitude threshold — a change counts only when it
        # exceeds the relative epsilon (or any nonzero change when baseline is 0). A
        # within-epsilon TIE is neither improved nor materially_worse, and crucially
        # counts as NO benefit (it cannot rescue an otherwise-regressed run).
        material = (
            abs(relative_delta) > _BASELINE_MATERIALITY_REL_EPSILON
            if relative_delta is not None
            else delta != 0
        )
        if direction == "lower":
            improved = material and with_regime < baseline
            materially_worse = material and with_regime > baseline
        else:
            improved = material and with_regime > baseline
            materially_worse = material and with_regime < baseline
        if improved:
            improved_metrics.append(metric)
        if materially_worse:
            materially_worse_metrics.append(metric)
        comparisons[metric] = {
            "with_regime_gating": with_regime,
            "no_regime_baseline": baseline,
            "delta": delta,
            "relative_delta": relative_delta,
            "better_direction": direction,
        }
    return {
        "comparisons": comparisons,
        "improved_metrics": sorted(improved_metrics),
        "materially_worse_metrics": sorted(materially_worse_metrics),
        "unknown_metrics": sorted(unknown_metrics),
        # F-016: the §6/§9 gate is "no material benefit anywhere + a regression" — fail
        # when no metric is materially improved AND at least one is materially worse (a
        # tie on one metric no longer rescues an otherwise all-worse run). Pass still
        # requires a clear benefit on >=1 material dimension (improved_metrics non-empty).
        "all_metrics_materially_worse": (
            bool(comparisons)
            and not unknown_metrics
            and not improved_metrics
            and bool(materially_worse_metrics)
        ),
    }


def _crash_window_red_flags(success_df: pd.DataFrame) -> list[dict[str, Any]]:
    """F-050: flag any configured crash window, covered by OOS success rows, in which
    no crisis-equivalent label appears on the volatility or transition-risk axis."""
    flags: list[dict[str, Any]] = []
    if success_df.empty or "as_of_date" not in success_df.columns:
        return flags
    as_of = pd.to_datetime(success_df["as_of_date"], errors="coerce")
    crisis_by_axis = (
        ("volatility_state_active", CRISIS_EQUIVALENT_VOLATILITY_LABELS),
        ("transition_risk_state", CRISIS_EQUIVALENT_TRANSITION_LABELS),
    )
    for name, start, end in CRASH_WINDOWS:
        in_window = (as_of >= pd.Timestamp(start)) & (as_of <= pd.Timestamp(end))
        covered = int(in_window.sum())
        if covered == 0:
            # The walk-forward never reached this window — cannot assert anything.
            continue
        window_df = success_df[in_window]
        has_crisis = any(
            col in window_df.columns
            and bool(window_df[col].astype(str).isin(labels).any())
            for col, labels in crisis_by_axis
        )
        if not has_crisis:
            flags.append(
                {
                    "type": "crisis_label_missing_in_crash_window",
                    "window": name,
                    "window_start": start,
                    "window_end": end,
                    "covered_sessions": covered,
                }
            )
    return flags


def _red_flags(
    success_df: pd.DataFrame, hysteresis_days: dict[str, int]
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    if success_df.empty:
        return flags
    session_count = len(success_df)
    for col in LABEL_COLUMNS:
        if col not in success_df.columns:
            continue
        counts = success_df[col].fillna("null").astype(str).value_counts()
        if not counts.empty:
            label = str(counts.index[0])
            count = int(counts.iloc[0])
            share = count / session_count
            if share >= LABEL_DOMINANCE_THRESHOLD:
                flags.append(
                    {
                        "type": "label_dominance",
                        "column": col,
                        "label": label,
                        "count": count,
                        "share": share,
                    }
                )
        unknown_run = _unknown_stretch(success_df[col])
        if unknown_run > UNKNOWN_STRETCH_THRESHOLD:
            flags.append(
                {
                    "type": "long_unknown_run",
                    "column": col,
                    "max_unknown_stretch": unknown_run,
                }
            )
        false_switch_count = _false_switch_count(success_df[col], hysteresis_days[col])
        if false_switch_count > 1:
            flags.append(
                {
                    "type": "repeated_one_day_flip_flops",
                    "column": col,
                    "false_switch_count": false_switch_count,
                }
            )

    flags.extend(_crash_window_red_flags(success_df))

    if "transition_risk_state" in success_df.columns:
        transition_states = (
            success_df["transition_risk_state"].fillna("null").astype(str)
        )
        warning_count = int(
            transition_states.isin(TRANSITION_RISK_WARNING_STATES).sum()
        )
        if warning_count == 0:
            flags.append(
                {
                    "type": "transition_risk_never_fires",
                    "column": "transition_risk_state",
                }
            )
        elif warning_count / session_count >= LABEL_DOMINANCE_THRESHOLD:
            flags.append(
                {
                    "type": "transition_risk_almost_always_fires",
                    "column": "transition_risk_state",
                    "count": warning_count,
                    "share": warning_count / session_count,
                }
            )
    return flags


def _failure_reasons(
    *,
    summary_df: pd.DataFrame,
    missing_sessions: list[str],
    nan_leakage: list[str],
    missing_summary_columns: list[str],
    frozen_versions: dict[str, Any],
    replay_gate_reasons: list[str],
    golden_gate_reasons: list[str],
    baseline_comparison: dict[str, Any] | None,
    red_flags: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if summary_df["status"].eq("failure").any():
        reasons.append("run_failures_present")
    if missing_sessions:
        reasons.append("missing_sessions")
    if nan_leakage:
        reasons.append("nan_leakage_detected")
    if missing_summary_columns:
        reasons.append("missing_report_columns")
    if not frozen_versions["is_single_pair"]:
        reasons.append("mixed_frozen_versions")
    if int(summary_df["status"].eq("success").sum()) < MIN_OOS_SESSIONS:
        reasons.append("insufficient_oos_sessions")
    reasons.extend(replay_gate_reasons)
    reasons.extend(golden_gate_reasons)
    if baseline_comparison is None:
        reasons.append("missing_baseline_metrics")
    elif baseline_comparison["unknown_metrics"]:
        reasons.append("unknown_baseline_metric_direction")
    elif baseline_comparison["all_metrics_materially_worse"]:
        reasons.append("materially_worse_than_baseline")
    if red_flags:
        reasons.append("red_flags_detected")
    return reasons


def _build_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Historical Walk-Forward Analysis",
        "",
        f"- status: `{analysis['status']}`",
        f"- engine_version: `{analysis['engine_version']}`",
        f"- config_version: `{analysis['config_version']}`",
        f"- session_count: `{analysis['session_count']}`",
        f"- oos_session_count: `{analysis['oos_session_count']}`",
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

    lines.extend(
        [
            "",
            "## Frozen Version",
            "",
            "```json",
            json.dumps(analysis["frozen_versions"], indent=2, sort_keys=True),
            "```",
            "",
            "## Missing Sessions",
            "",
            "```json",
            json.dumps(analysis["missing_sessions"], indent=2, sort_keys=True),
            "```",
            "",
            "## NaN Leakage",
            "",
            "```json",
            json.dumps(analysis["nan_leakage"], indent=2, sort_keys=True),
            "```",
            "",
            "## Red Flags",
            "",
            "```json",
            json.dumps(analysis["red_flags"], indent=2, sort_keys=True),
            "```",
            "",
            "## Replay Verification",
            "",
            "```json",
            json.dumps(analysis["replay_verification"], indent=2, sort_keys=True),
            "```",
            "",
            "## Per-Date Provenance",
            "",
            "```json",
            json.dumps(analysis["per_date_provenance"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )

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
    replay_results_path: Path | None = None,
) -> dict[str, Any]:
    summary_df = _load_summary(output_root)
    runs_df = _load_runs_from_db(output_root)
    success_df = summary_df[summary_df["status"] == "success"].copy()
    missing_sessions = _missing_sessions(summary_df)
    nan_leakage = _nan_leakage(summary_df, runs_df, output_root)
    missing_summary_columns = _missing_summary_columns(summary_df)
    frozen_versions = _frozen_versions(runs_df)
    hysteresis_days = _load_hysteresis_days()

    golden_results = _load_optional_json(golden_results_path)
    golden_gate_reasons = _golden_gate_reasons(golden_results, runs_df)
    replay_results = _load_optional_json(replay_results_path)
    replay_gate_reasons = _replay_gate_reasons(replay_results, runs_df)
    baseline_comparison = _baseline_comparison(
        _load_optional_json(baseline_metrics_path)
    )
    red_flags = _red_flags(success_df, hysteresis_days)
    failure_reasons = _failure_reasons(
        summary_df=summary_df,
        missing_sessions=missing_sessions,
        nan_leakage=nan_leakage,
        missing_summary_columns=missing_summary_columns,
        frozen_versions=frozen_versions,
        replay_gate_reasons=replay_gate_reasons,
        golden_gate_reasons=golden_gate_reasons,
        baseline_comparison=baseline_comparison,
        red_flags=red_flags,
    )

    analysis = {
        "status": "pass" if not failure_reasons else "fail",
        "session_count": int(len(summary_df)),
        "oos_session_count": int(summary_df["status"].eq("success").sum()),
        "success_count": int(summary_df["status"].eq("success").sum()),
        "failure_count": int(summary_df["status"].eq("failure").sum()),
        "missing_sessions": missing_sessions,
        "nan_leakage": nan_leakage,
        "missing_summary_columns": missing_summary_columns,
        "failure_reasons": failure_reasons,
        "engine_version": frozen_versions["engine_version"],
        "config_version": frozen_versions["config_version"],
        "frozen_versions": frozen_versions,
        "per_date_provenance": _per_date_provenance(runs_df),
        "label_distributions": _label_distribution(success_df),
        "series_summaries": _series_summaries(success_df, hysteresis_days),
        "red_flags": red_flags,
        "golden_results": golden_results
        or {"all_passed": False, "reason": "not_provided"},
        "replay_verification": replay_results
        or {"all_passed": False, "reason": "not_provided"},
        "baseline_comparison": baseline_comparison
        or {
            "all_metrics_materially_worse": False,
            "reason": "not_provided",
            "improved_metrics": [],
            "unknown_metrics": [],
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
    parser.add_argument("--replay-results", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_walkforward_report(
        output_root=args.output_root,
        golden_results_path=args.golden_results,
        baseline_metrics_path=args.baseline_metrics,
        replay_results_path=args.replay_results,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
