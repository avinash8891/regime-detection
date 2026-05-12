from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_historical_walkforward.py"
    spec = importlib.util.spec_from_file_location("run_historical_walkforward", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_historical_walkforward_runner_writes_expected_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    market_data_path = repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    out_root = tmp_path / "walkforward"

    mod = _load_runner_module()
    result = mod.run_walkforward(
        market_data_path=market_data_path,
        output_root=out_root,
        start_date=date(2023, 12, 12),
        end_date=date(2023, 12, 14),
    )

    assert result["session_count"] == 3
    assert result["success_count"] == 3
    assert result["failure_count"] == 0

    db_path = out_root / "regime_walkforward.db"
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT as_of_date, status, output_path, input_archive_path FROM runs ORDER BY as_of_date"
        ).fetchall()
    assert [row[0] for row in rows] == ["2023-12-12", "2023-12-13", "2023-12-14"]
    assert {row[1] for row in rows} == {"success"}

    output_path = out_root / "outputs" / "2023-12-14.json"
    payload = json.loads(output_path.read_text())
    assert payload["as_of_date"] == "2023-12-14"
    assert payload["engine_version"].startswith("regime-engine-v")
    from regime_detection.config import load_default_regime_config

    assert payload["config_version"] == load_default_regime_config().config_version

    archive_market = out_root / "input_archives" / "2023-12-14" / "market_data.parquet"
    archived_df = pd.read_parquet(archive_market)
    archived_df["date"] = pd.to_datetime(archived_df["date"]).dt.date
    assert archived_df["date"].max() == date(2023, 12, 14)

    checksums_path = out_root / "input_archives" / "2023-12-14" / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    assert "market_data.parquet" in checksums
    assert "events.yaml" in checksums

    report_path = out_root / "reports" / "walkforward_report.md"
    summary_path = out_root / "reports" / "walkforward_summary.csv"
    assert report_path.exists()
    assert summary_path.exists()


def test_historical_walkforward_runner_records_failures_without_silent_skip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    original = pd.read_parquet(repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet")
    original["date"] = pd.to_datetime(original["date"]).dt.date

    broken = original.copy()
    broken = broken[~((broken["symbol"] == "SPY") & (broken["date"] == date(2023, 12, 14)))]
    broken_path = tmp_path / "broken_market_data.parquet"
    broken.to_parquet(broken_path, index=False)

    out_root = tmp_path / "walkforward"
    mod = _load_runner_module()
    result = mod.run_walkforward(
        market_data_path=broken_path,
        output_root=out_root,
        start_date=date(2023, 12, 13),
        end_date=date(2023, 12, 14),
    )

    assert result["session_count"] == 2
    assert result["success_count"] == 1
    assert result["failure_count"] == 1

    with sqlite3.connect(out_root / "regime_walkforward.db") as conn:
        rows = conn.execute(
            "SELECT as_of_date, status, failure_reason FROM runs ORDER BY as_of_date"
        ).fetchall()

    assert rows[0][0] == "2023-12-13"
    assert rows[0][1] == "success"
    assert rows[1][0] == "2023-12-14"
    assert rows[1][1] == "failure"
    assert "SPY row" in rows[1][2]
