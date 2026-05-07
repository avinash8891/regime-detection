from __future__ import annotations

import importlib.util
import json
from datetime import date
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
    runner = _load_module("run_historical_walkforward", "scripts/run_historical_walkforward.py")
    repo_root = Path(__file__).resolve().parents[1]
    market_data_path = repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    out_root = tmp_path / "walkforward"
    runner.run_walkforward(
        market_data_path=market_data_path,
        output_root=out_root,
        start_date=date(2023, 12, 12),
        end_date=date(2023, 12, 14),
    )
    return out_root


def test_build_walkforward_report_fails_without_required_gates(tmp_path: Path) -> None:
    report_mod = _load_module("build_walkforward_report", "scripts/build_walkforward_report.py")
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
    assert payload["label_distributions"]["transition_risk_label"]


def test_build_walkforward_report_passes_with_golden_and_baseline_inputs(tmp_path: Path) -> None:
    report_mod = _load_module("build_walkforward_report", "scripts/build_walkforward_report.py")
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
                    "max_drawdown": {"with_regime_gating": 0.12, "no_regime_baseline": 0.18},
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
    assert result["baseline_comparison"]["improved_metrics"] == ["max_drawdown", "sharpe"]

    payload = json.loads((out_root / "reports" / "walkforward_analysis.json").read_text())
    assert payload["golden_results"]["all_passed"] is True
    assert payload["baseline_comparison"]["all_metrics_materially_worse"] is False
