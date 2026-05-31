from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import date
from pathlib import Path

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
                as_of_date, status, failure_reason, engine_version,
                config_version, input_archive_path, output_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["as_of_date"]),
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
    path = output_root / "reports" / "replay_verification.json"
    path.write_text(
        json.dumps(
            {
                "all_passed": all_passed,
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
