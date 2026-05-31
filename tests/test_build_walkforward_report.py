from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


def _load_module(name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _prepare_walkforward_root(tmp_path: Path) -> Path:
    """Create the report builder's persisted inputs without running the engine."""
    out_root = tmp_path / "walkforward"
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "walkforward_summary.csv").write_text(
        "\n".join(
            [
                "as_of_date,status,trend_direction_active,trend_character_active,volatility_state_active,breadth_state_active,transition_risk_state",
                "2023-12-12,success,uptrend,trend,low_volatility,healthy,normal",
                "2023-12-13,success,uptrend,trend,low_volatility,healthy,normal",
                "2023-12-14,success,uptrend,trend,low_volatility,healthy,elevated",
                "",
            ]
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
                    "2023-12-12",
                    "success",
                    None,
                    "regime-engine-vtest",
                    "core3-test",
                    "input_archives/2023-12-12",
                    "outputs/2023-12-12.json",
                ),
                (
                    "2023-12-13",
                    "success",
                    None,
                    "regime-engine-vtest",
                    "core3-test",
                    "input_archives/2023-12-13",
                    "outputs/2023-12-13.json",
                ),
                (
                    "2023-12-14",
                    "success",
                    None,
                    "regime-engine-vtest",
                    "core3-test",
                    "input_archives/2023-12-14",
                    "outputs/2023-12-14.json",
                ),
            ],
        )
    return out_root


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


def test_build_walkforward_report_passes_with_golden_and_baseline_inputs(
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
                "all_passed": True,
                "results": [
                    {"as_of_date": "2023-12-14", "passed": True},
                ],
            },
            indent=2,
        )
        + "\n",
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
    assert payload["golden_results"]["all_passed"] is True
    assert payload["baseline_comparison"]["all_metrics_materially_worse"] is False


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
