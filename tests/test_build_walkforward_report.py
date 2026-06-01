from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import yaml


def _load_module(name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _session_rows(report_mod, *, count: int = 252) -> list[dict[str, object]]:
    schedule = report_mod.nyse_calendar().schedule(
        start_date=date(2023, 1, 3),
        end_date=date(2024, 3, 31),
    )
    sessions = list(schedule.index.date)[:count]
    assert len(sessions) == count
    trend_labels = ("bull", "sideways", "bear", "recovery")
    character_labels = ("trending", "chop", "transition", "recovery_attempt")
    vol_labels = ("low_vol", "normal_vol", "high_vol", "crisis_vol")
    breadth_labels = (
        "healthy_breadth",
        "neutral_breadth",
        "weak_breadth",
        "divergent_fragile",
    )
    transition_labels = ("stable", "watch", "elevated", "high_transition_risk")
    rows: list[dict[str, object]] = []
    for idx, session in enumerate(sessions):
        label_idx = idx // 4
        rows.append(
            {
                "as_of_date": session.isoformat(),
                "run_timestamp": f"{session.isoformat()}T12:00:00Z",  # F-019
                "status": "success",
                "trend_direction_active": trend_labels[label_idx % len(trend_labels)],
                "trend_character_active": character_labels[
                    label_idx % len(character_labels)
                ],
                "volatility_state_active": vol_labels[label_idx % len(vol_labels)],
                "breadth_state_active": breadth_labels[label_idx % len(breadth_labels)],
                "transition_risk_state": transition_labels[
                    label_idx % len(transition_labels)
                ],
                "transition_risk_score": round((idx % 10) / 10, 2),
                "transition_risk_primary_drivers": "[]",
                "transition_risk_triggered_rules": "[]",
                "transition_risk_data_quality_status": "ok",
                "transition_risk_axis_switch_count": idx % 3,
                "transition_risk_recent_axis_switch_count": idx % 2,
            }
        )
    return rows


def _prepare_walkforward_root(
    tmp_path: Path,
    report_mod=None,
    *,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    """Create the report builder's persisted inputs without running the engine."""
    if rows is None:
        if report_mod is None:
            rows = [
                {
                    "as_of_date": "2023-12-12",
                    "run_timestamp": "2023-12-12T12:00:00Z",
                    "status": "success",
                    "trend_direction_active": "uptrend",
                    "trend_character_active": "trend",
                    "volatility_state_active": "low_volatility",
                    "breadth_state_active": "healthy",
                    "transition_risk_state": "normal",
                    "transition_risk_score": 0.1,
                    "transition_risk_primary_drivers": "[]",
                    "transition_risk_triggered_rules": "[]",
                    "transition_risk_data_quality_status": "ok",
                    "transition_risk_axis_switch_count": 0,
                    "transition_risk_recent_axis_switch_count": 0,
                },
                {
                    "as_of_date": "2023-12-13",
                    "run_timestamp": "2023-12-13T12:00:00Z",
                    "status": "success",
                    "trend_direction_active": "uptrend",
                    "trend_character_active": "trend",
                    "volatility_state_active": "low_volatility",
                    "breadth_state_active": "healthy",
                    "transition_risk_state": "normal",
                    "transition_risk_score": 0.1,
                    "transition_risk_primary_drivers": "[]",
                    "transition_risk_triggered_rules": "[]",
                    "transition_risk_data_quality_status": "ok",
                    "transition_risk_axis_switch_count": 0,
                    "transition_risk_recent_axis_switch_count": 0,
                },
                {
                    "as_of_date": "2023-12-14",
                    "run_timestamp": "2023-12-14T12:00:00Z",
                    "status": "success",
                    "trend_direction_active": "uptrend",
                    "trend_character_active": "trend",
                    "volatility_state_active": "low_volatility",
                    "breadth_state_active": "healthy",
                    "transition_risk_state": "elevated",
                    "transition_risk_score": 0.4,
                    "transition_risk_primary_drivers": "[]",
                    "transition_risk_triggered_rules": "[]",
                    "transition_risk_data_quality_status": "ok",
                    "transition_risk_axis_switch_count": 0,
                    "transition_risk_recent_axis_switch_count": 0,
                },
            ]
        else:
            rows = _session_rows(report_mod)

    out_root = tmp_path / "walkforward"
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True)
    fieldnames = list(rows[0])
    (reports_dir / "walkforward_summary.csv").write_text(
        "\n".join(
            [
                ",".join(fieldnames),
                *[",".join(str(row[name]) for name in fieldnames) for row in rows],
                "",
            ],
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(out_root / "regime_walkforward.db") as conn:
        conn.execute("""
            CREATE TABLE runs (
                as_of_date TEXT NOT NULL,
                run_timestamp TEXT,
                status TEXT NOT NULL,
                failure_reason TEXT,
                engine_version TEXT,
                config_version TEXT,
                input_archive_path TEXT,
                output_path TEXT
            )
            """)
        conn.executemany(
            """
            INSERT INTO runs (
                as_of_date, run_timestamp, status, failure_reason, engine_version,
                config_version, input_archive_path, output_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["as_of_date"]),
                    str(row["run_timestamp"]),
                    str(row["status"]),
                    None,
                    "regime-engine-vtest",
                    "core3-test",
                    f"input_archives/{row['as_of_date']}",
                    f"outputs/{row['as_of_date']}.json",
                )
                for row in rows
            ],
        )
    return out_root


def _write_replay_results(
    output_root: Path, *, all_passed: bool = True, mismatches: list[str] | None = None
) -> Path:
    # F-001: the hardened §6 gate requires the replay verdict to cover every
    # SUCCESSFUL walk-forward date and be bound to the frozen engine/config pair,
    # mirroring what run_walkforward_replay_check emits.
    with sqlite3.connect(output_root / "regime_walkforward.db") as conn:
        success_dates = [
            str(row[0])
            for row in conn.execute(
                "SELECT as_of_date FROM runs WHERE status='success' ORDER BY as_of_date"
            ).fetchall()
        ]
    path = output_root / "reports" / "replay_verification.json"
    path.write_text(
        json.dumps(
            {
                "engine_version": "regime-engine-vtest",
                "config_version": "core3-test",
                "all_passed": all_passed,
                "results": [
                    {"as_of_date": as_of, "matches": all_passed}
                    for as_of in success_dates
                ],
                "mismatches": mismatches or [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _golden_results_payload(*, all_passed: bool = True) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    return {
        "all_passed": all_passed,
        "engine_version": "regime-engine-vtest",
        "config_version": "core3-test",
        "results": [
            {"as_of_date": row["as_of_date"], "passed": all_passed}
            for row in golden["rows"]
            if "expected" in row
        ],
    }


def _golden_results_batch_payload(*, all_passed: bool = True) -> dict[str, object]:
    return {
        "pre_batch": _golden_results_payload(all_passed=all_passed),
        "post_batch": _golden_results_payload(all_passed=all_passed),
    }


def test_build_walkforward_report_fails_without_required_gates(
    tmp_path: Path,
) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    out_root = _prepare_walkforward_root(tmp_path)

    result = report_mod.build_walkforward_report(output_root=out_root)

    assert result["status"] == "fail"
    assert "missing_golden_results" in result["failure_reasons"]
    assert "missing_baseline_metrics" in result["failure_reasons"]
    assert "missing_replay_verification" in result["failure_reasons"]
    assert "insufficient_oos_sessions" in result["failure_reasons"]

    analysis_path = out_root / "reports" / "walkforward_analysis.json"
    report_path = out_root / "reports" / "walkforward_report.md"
    assert analysis_path.exists()
    assert report_path.exists()

    payload = json.loads(analysis_path.read_text())
    assert payload["status"] == "fail"
    assert payload["session_count"] == 3
    assert payload["success_count"] == 3
    assert payload["missing_sessions"] == []
    assert payload["label_distributions"]["transition_risk_state"]
    assert payload["per_date_provenance"][0] == {
        "as_of_date": "2023-12-12",
        "run_timestamp": "2023-12-12T12:00:00Z",
        "engine_version": "regime-engine-vtest",
        "config_version": "core3-test",
        "input_archive_path": "input_archives/2023-12-12",
        "output_path": "outputs/2023-12-12.json",
    }
    report_text = report_path.read_text()
    assert "## Frozen Version" in report_text
    assert "## Per-Date Provenance" in report_text


def test_build_walkforward_report_passes_with_golden_and_baseline_inputs(
    tmp_path: Path,
) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    out_root = _prepare_walkforward_root(tmp_path, report_mod)

    golden_path = tmp_path / "golden_results.json"
    golden_path.write_text(
        json.dumps(_golden_results_batch_payload(), indent=2) + "\n",
        encoding="utf-8",
    )

    baseline_path = tmp_path / "baseline_metrics.json"
    baseline_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "max_drawdown": {
                        "with_regime_gating": 0.12,
                        "no_regime_baseline": 0.18,
                    },
                    "sharpe": {"with_regime_gating": 1.10, "no_regime_baseline": 0.95},
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        golden_results_path=golden_path,
        baseline_metrics_path=baseline_path,
        replay_results_path=_write_replay_results(out_root),
    )

    assert result["status"] == "pass"
    assert result["failure_reasons"] == []
    assert result["baseline_comparison"]["improved_metrics"] == [
        "max_drawdown",
        "sharpe",
    ]

    payload = json.loads(
        (out_root / "reports" / "walkforward_analysis.json").read_text()
    )
    assert payload["golden_results"]["pre_batch"]["all_passed"] is True
    assert payload["golden_results"]["post_batch"]["all_passed"] is True
    assert payload["baseline_comparison"]["all_metrics_materially_worse"] is False
    assert (
        payload["baseline_comparison"]["comparisons"]["max_drawdown"]["relative_delta"]
        == -0.3333333333333333
    )
    assert payload["oos_session_count"] == 252
    assert payload["red_flags"] == []
    assert payload["replay_verification"]["all_passed"] is True


def test_build_walkforward_report_rejects_replay_mismatch(tmp_path: Path) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    out_root = _prepare_walkforward_root(tmp_path, report_mod)
    golden_path = tmp_path / "golden_results.json"
    golden_path.write_text(
        json.dumps(_golden_results_batch_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline_metrics.json"
    baseline_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "max_drawdown": {
                        "with_regime_gating": 0.12,
                        "no_regime_baseline": 0.18,
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        golden_results_path=golden_path,
        baseline_metrics_path=baseline_path,
        replay_results_path=_write_replay_results(
            out_root, all_passed=False, mismatches=["2023-01-04"]
        ),
    )

    assert result["status"] == "fail"
    assert "replay_mismatch_detected" in result["failure_reasons"]


def test_build_walkforward_report_rejects_short_oos_window(tmp_path: Path) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    out_root = _prepare_walkforward_root(
        tmp_path, report_mod, rows=_session_rows(report_mod, count=251)
    )
    golden_path = tmp_path / "golden_results.json"
    golden_path.write_text(
        json.dumps(_golden_results_batch_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline_metrics.json"
    baseline_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "sharpe": {"with_regime_gating": 1.10, "no_regime_baseline": 0.95}
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        golden_results_path=golden_path,
        baseline_metrics_path=baseline_path,
        replay_results_path=_write_replay_results(out_root),
    )

    assert result["status"] == "fail"
    assert "insufficient_oos_sessions" in result["failure_reasons"]


def test_build_walkforward_report_rejects_mixed_frozen_versions(
    tmp_path: Path,
) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    out_root = _prepare_walkforward_root(tmp_path, report_mod)
    with sqlite3.connect(out_root / "regime_walkforward.db") as conn:
        conn.execute(
            "UPDATE runs SET config_version = 'core3-other' WHERE as_of_date = ?",
            ("2023-01-04",),
        )

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        replay_results_path=_write_replay_results(out_root),
    )

    assert result["status"] == "fail"
    assert "mixed_frozen_versions" in result["failure_reasons"]


def test_build_walkforward_report_rejects_unknown_baseline_metric_direction(
    tmp_path: Path,
) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    out_root = _prepare_walkforward_root(tmp_path, report_mod)
    baseline_path = tmp_path / "baseline_metrics.json"
    baseline_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "time_spent_in_each_regime": {
                        "with_regime_gating": 10,
                        "no_regime_baseline": 8,
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        baseline_metrics_path=baseline_path,
        replay_results_path=_write_replay_results(out_root),
    )

    assert result["status"] == "fail"
    assert "unknown_baseline_metric_direction" in result["failure_reasons"]
    assert result["baseline_comparison"]["unknown_metrics"] == [
        "time_spent_in_each_regime"
    ]


def _crash_window_rows(*, covid_volatility: str, covid_transition: str) -> pd.DataFrame:
    # Three sessions inside the F-050 covid_crash_2020 window (2020-02-24..2020-03-23)
    # plus a calm session well outside any window. The covid sessions carry the
    # labels under test; the calm session must never arm the check.
    return pd.DataFrame(
        [
            {
                "as_of_date": "2020-03-16",
                "volatility_state_active": covid_volatility,
                "transition_risk_state": covid_transition,
            },
            {
                "as_of_date": "2020-03-18",
                "volatility_state_active": covid_volatility,
                "transition_risk_state": covid_transition,
            },
            {
                "as_of_date": "2020-03-20",
                "volatility_state_active": covid_volatility,
                "transition_risk_state": covid_transition,
            },
            {
                "as_of_date": "2019-06-28",
                "volatility_state_active": "low_vol",
                "transition_risk_state": "stable",
            },
        ]
    )


def test_crash_window_flags_missing_crisis_label() -> None:
    # F-050: COVID-crash sessions present in OOS but NO crisis-equivalent label on
    # either the volatility or transition-risk axis ⇒ the §8 red flag fires.
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    success_df = _crash_window_rows(
        covid_volatility="high_vol", covid_transition="bear_stress"
    )

    flags = report_mod._crash_window_red_flags(success_df)

    assert flags == [
        {
            "type": "crisis_label_missing_in_crash_window",
            "window": "covid_crash_2020",
            "window_start": "2020-02-24",
            "window_end": "2020-03-23",
            "covered_sessions": 3,
        }
    ]


def test_crash_window_accepts_crisis_on_either_axis() -> None:
    # F-050: a crisis-equivalent label on EITHER axis satisfies the window — no flag.
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    vol_satisfies = report_mod._crash_window_red_flags(
        _crash_window_rows(
            covid_volatility="crisis_vol", covid_transition="bear_stress"
        )
    )
    transition_satisfies = report_mod._crash_window_red_flags(
        _crash_window_rows(covid_volatility="high_vol", covid_transition="crisis")
    )

    assert vol_satisfies == []
    assert transition_satisfies == []


def test_crash_window_does_not_arm_when_window_uncovered() -> None:
    # F-050: a window the walk-forward never reached must NOT be flagged (we cannot
    # assert a missing crisis label for sessions that were never classified).
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    success_df = pd.DataFrame(
        [
            {
                "as_of_date": "2019-06-28",
                "volatility_state_active": "low_vol",
                "transition_risk_state": "stable",
            }
        ]
    )

    assert report_mod._crash_window_red_flags(success_df) == []


def test_build_walkforward_report_rejects_red_flags(tmp_path: Path) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    rows = _session_rows(report_mod)
    for row in rows:
        row["trend_direction_active"] = "bull"
        row["transition_risk_state"] = "stable"
    out_root = _prepare_walkforward_root(tmp_path, report_mod, rows=rows)

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        replay_results_path=_write_replay_results(out_root),
    )

    assert result["status"] == "fail"
    assert "red_flags_detected" in result["failure_reasons"]
    assert {
        "type": "label_dominance",
        "column": "trend_direction_active",
        "label": "bull",
        "count": 252,
        "share": 1.0,
    } in result["red_flags"]
    assert {
        "type": "transition_risk_never_fires",
        "column": "transition_risk_state",
    } in result["red_flags"]


def test_build_walkforward_report_counts_current_transition_risk_states(
    tmp_path: Path,
) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    rows = _session_rows(report_mod)
    for row in rows:
        row["transition_risk_state"] = "weakening"
    out_root = _prepare_walkforward_root(tmp_path, report_mod, rows=rows)

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        replay_results_path=_write_replay_results(out_root),
    )

    assert {
        "type": "transition_risk_almost_always_fires",
        "column": "transition_risk_state",
        "count": 252,
        "share": 1.0,
    } in result["red_flags"]
    assert {
        "type": "transition_risk_never_fires",
        "column": "transition_risk_state",
    } not in result["red_flags"]


def test_build_walkforward_report_rejects_incomplete_golden_results(
    tmp_path: Path,
) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    out_root = _prepare_walkforward_root(tmp_path)
    golden_path = tmp_path / "golden_results.json"
    golden_path.write_text(
        json.dumps(
            {
                "pre_batch": _golden_results_payload(),
                "post_batch": {
                    "all_passed": True,
                    "engine_version": "regime-engine-vtest",
                    "config_version": "core3-test",
                    "results": [{"as_of_date": "2023-12-14", "passed": True}],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        golden_results_path=golden_path,
    )

    assert result["status"] == "fail"
    assert "golden_results_dates_mismatch" in result["failure_reasons"]


def test_build_walkforward_report_requires_before_and_after_golden_results(
    tmp_path: Path,
) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    out_root = _prepare_walkforward_root(tmp_path)
    golden_path = tmp_path / "golden_results.json"
    golden_path.write_text(
        json.dumps(_golden_results_payload(), indent=2) + "\n",
        encoding="utf-8",
    )

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        golden_results_path=golden_path,
    )

    assert result["status"] == "fail"
    assert "golden_results_missing_before_after" in result["failure_reasons"]


def test_build_walkforward_report_rejects_golden_version_mismatch(
    tmp_path: Path,
) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    out_root = _prepare_walkforward_root(tmp_path)
    payload = _golden_results_batch_payload()
    payload["post_batch"]["engine_version"] = "different-engine"
    golden_path = tmp_path / "golden_results.json"
    golden_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        golden_results_path=golden_path,
    )

    assert result["status"] == "fail"
    assert "golden_results_version_mismatch" in result["failure_reasons"]


def test_build_walkforward_report_rejects_json_output_nan_leakage(
    tmp_path: Path,
) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    out_root = _prepare_walkforward_root(tmp_path)
    output_path = out_root / "outputs" / "2023-12-13.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"transition_risk": {"score": NaN}}\n', encoding="utf-8")

    result = report_mod.build_walkforward_report(output_root=out_root)

    assert result["status"] == "fail"
    assert "nan_leakage_detected" in result["failure_reasons"]
    assert result["nan_leakage"] == [
        "output.transition_risk.score@2023-12-13",
    ]


def test_build_walkforward_report_fails_on_nan_leakage(tmp_path: Path) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    out_root = _prepare_walkforward_root(tmp_path)
    summary_path = out_root / "reports" / "walkforward_summary.csv"
    summary_path.write_text(
        "\n".join(
            [
                "as_of_date,status,trend_direction_active,trend_character_active,volatility_state_active,breadth_state_active,transition_risk_state,transition_risk_score",
                "2023-12-12,success,uptrend,trend,low_volatility,healthy,normal,0.10",
                "2023-12-13,success,uptrend,trend,low_volatility,healthy,normal,NaN",
                "2023-12-14,success,uptrend,trend,low_volatility,healthy,elevated,0.40",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = report_mod.build_walkforward_report(output_root=out_root)

    assert result["status"] == "fail"
    assert "nan_leakage_detected" in result["failure_reasons"]
    assert result["nan_leakage"] == [
        "summary.transition_risk_score@2023-12-13",
    ]


def test_build_walkforward_report_rejects_long_unknown_stretch(
    tmp_path: Path,
) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    rows = _session_rows(report_mod)
    for row in rows[: report_mod.UNKNOWN_STRETCH_THRESHOLD + 1]:
        row["breadth_state_active"] = "unknown"
    out_root = _prepare_walkforward_root(tmp_path, report_mod, rows=rows)

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        replay_results_path=_write_replay_results(out_root),
    )

    assert result["status"] == "fail"
    assert "red_flags_detected" in result["failure_reasons"]
    assert {
        "type": "long_unknown_run",
        "column": "breadth_state_active",
        "max_unknown_stretch": report_mod.UNKNOWN_STRETCH_THRESHOLD + 1,
    } in result["red_flags"]


def test_build_walkforward_report_rejects_repeated_one_day_flip_flops(
    tmp_path: Path,
) -> None:
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    rows = _session_rows(report_mod)
    for idx, row in enumerate(rows):
        row["trend_direction_active"] = "bull" if idx % 2 == 0 else "bear"
    out_root = _prepare_walkforward_root(tmp_path, report_mod, rows=rows)

    result = report_mod.build_walkforward_report(
        output_root=out_root,
        replay_results_path=_write_replay_results(out_root),
    )

    assert result["status"] == "fail"
    assert "red_flags_detected" in result["failure_reasons"]
    assert {
        "type": "repeated_one_day_flip_flops",
        "column": "trend_direction_active",
        "false_switch_count": 250,
    } in result["red_flags"]


def test_replay_gate_emits_no_successful_runs_distinctly() -> None:
    """CR-005: an empty batch (zero successful runs) yields a distinct
    no_successful_runs_to_replay reason — not the misleading replay_mismatch_detected,
    which implies a verification FAILURE rather than nothing to verify."""
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    runs_df = pd.DataFrame(
        [
            {"as_of_date": date(2023, 12, 12), "status": "failure"},
            {"as_of_date": date(2023, 12, 13), "status": "failure"},
        ]
    )
    replay_results = {
        "all_passed": False,
        "results": [],
        "engine_version": "regime-engine-vtest",
        "config_version": "core3-test",
    }

    reasons = report_mod._replay_gate_reasons(replay_results, runs_df)

    assert reasons == ["no_successful_runs_to_replay"]
    assert "replay_mismatch_detected" not in reasons


def test_baseline_comparison_flags_all_materially_worse() -> None:
    """F-009: with-regime worse on EVERY tracked metric → all_metrics_materially_worse
    (the central economic-improvement fail gate), and no metric is improved."""
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    payload = {
        "metrics": {
            "max_drawdown": {"with_regime_gating": 0.25, "no_regime_baseline": 0.10},
            "sharpe": {"with_regime_gating": 0.50, "no_regime_baseline": 1.20},
        }
    }

    comp = report_mod._baseline_comparison(payload)

    assert comp is not None
    assert comp["all_metrics_materially_worse"] is True
    assert comp["improved_metrics"] == []
    assert comp["materially_worse_metrics"] == ["max_drawdown", "sharpe"]


def test_baseline_comparison_tie_does_not_rescue_regression() -> None:
    """F-016: a within-epsilon tie on one metric is NOT a material benefit and must not
    rescue a run that is materially worse on every other metric — the gate still fires.
    """
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    payload = {
        "metrics": {
            # ~0.2% delta → within the 1% materiality epsilon → a TIE (no benefit).
            "sharpe": {"with_regime_gating": 1.000, "no_regime_baseline": 1.002},
            "max_drawdown": {"with_regime_gating": 0.25, "no_regime_baseline": 0.10},
            "hit_rate": {"with_regime_gating": 0.40, "no_regime_baseline": 0.55},
        }
    }

    comp = report_mod._baseline_comparison(payload)

    assert comp is not None
    assert "sharpe" not in comp["improved_metrics"]
    assert "sharpe" not in comp["materially_worse_metrics"]  # tie, not material
    assert comp["improved_metrics"] == []  # no material benefit anywhere
    assert comp["all_metrics_materially_worse"] is True


def test_nan_leakage_flags_blank_label_but_accepts_unknown(tmp_path: Path) -> None:
    """F-020: §6 NaN-leakage gate — a success row whose label cell is blank/None (not
    surfaced as the explicit unknown/insufficient_history contract) is flagged; the
    explicit unknown / insufficient_history labels are accepted."""
    report_mod = _load_module(
        "build_walkforward_report", "scripts/build_walkforward_report.py"
    )
    summary_df = pd.DataFrame(
        [
            {
                "as_of_date": date(2023, 12, 12),
                "status": "success",
                "trend_direction_active": "bull",
                "trend_character_active": "trending",
                "volatility_state_active": "unknown",
                "breadth_state_active": "healthy_breadth",
                "transition_risk_state": "insufficient_history",
            },
            {
                "as_of_date": date(2023, 12, 13),
                "status": "success",
                "trend_direction_active": None,  # blank → contract violation
                "trend_character_active": "trending",
                "volatility_state_active": "",  # blank → contract violation
                "breadth_state_active": "healthy_breadth",
                "transition_risk_state": "stable",
            },
        ]
    )
    runs_df = pd.DataFrame({"as_of_date": [], "output_path": []})

    leaks = report_mod._nan_leakage(summary_df, runs_df, tmp_path)

    assert "summary.trend_direction_active@2023-12-13:label_contract_violation" in leaks
    assert (
        "summary.volatility_state_active@2023-12-13:label_contract_violation" in leaks
    )
    # the explicit unknown / insufficient_history contract labels are accepted
    assert not any("@2023-12-12" in leak for leak in leaks)
