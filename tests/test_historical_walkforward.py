from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from contextlib import closing

import pandas as pd
import pytest

pytestmark = pytest.mark.slow


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_historical_walkforward.py"
    spec = importlib.util.spec_from_file_location(
        "run_historical_walkforward", script_path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _write_v2_macro_parquet(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    source = pd.read_csv(
        repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    )
    dates = pd.to_datetime(sorted(source["date"].unique()))
    trend = pd.Series(range(len(dates)), index=dates, dtype="float64")
    logical_series = {
        "2y_yield": 4.00 + trend * 0.0002,
        "10y_yield": 4.25 + trend * 0.0001,
        "broad_usd_index": 100.0 + trend * 0.01,
        "sofr": 5.25 + trend * 0.0001,
        "iorb": 5.40 + trend * 0.0001,
        "nfci": -0.35 + trend * 0.00001,
        "hy_oas": 3.8 + trend * 0.0001,
        "ig_bbb_oas": 1.6 + trend * 0.0001,
        "cpi_all_items": 300.0 + trend * 0.01,
        "pmi_manufacturing": 50.0 + trend * 0.0001,
    }
    rows = [
        {
            "date": observed_date,
            "series_id": logical_name.upper(),
            "logical_name": logical_name,
            "value": value,
        }
        for logical_name, series in logical_series.items()
        for observed_date, value in series.items()
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_historical_summary_cells_include_event_calendar_primary_and_matching_labels() -> (
    None
):
    mod = _load_runner_module()
    output = SimpleNamespace(
        structural_causal_state=SimpleNamespace(
            event_calendar=SimpleNamespace(
                primary_label="fed_week",
                matching_labels=("fed_week", "cpi_week", "expiry_week"),
            )
        )
    )

    assert mod._event_calendar_summary_cells(output) == {
        "event_calendar_primary_label": "fed_week",
        "event_calendar_matching_labels": '["fed_week", "cpi_week", "expiry_week"]',
    }


def test_historical_walkforward_runner_writes_expected_artifacts(
    walkforward_2023_dec_template: Path,
) -> None:
    out_root = walkforward_2023_dec_template

    db_path = out_root / "regime_walkforward.db"
    assert db_path.exists()
    with closing(sqlite3.connect(db_path)) as conn:
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
    summary_df = pd.read_csv(summary_path)
    assert {
        "event_calendar_primary_label",
        "event_calendar_matching_labels",
        "v2_dependency_payload_contracts",
        "transition_risk_score",
        "transition_risk_primary_drivers",
        "transition_risk_triggered_rules",
        "transition_risk_data_quality_status",
        "transition_risk_axis_switch_count",
        "transition_risk_recent_axis_switch_count",
    }.issubset(summary_df.columns)


def test_historical_walkforward_supplies_macro_series_for_configured_v2_axes(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    v2_daily_path = repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    config_path = repo_root / "tests" / "fixtures" / "configs" / "core3-v2-fast.yaml"
    macro_path = _write_v2_macro_parquet(tmp_path / "fred_macro_series.parquet")
    out_root = tmp_path / "walkforward"

    mod = _load_runner_module()
    result = mod.run_walkforward(
        market_data_path=v2_daily_path,
        output_root=out_root,
        start_date=date(2026, 5, 13),
        end_date=date(2026, 5, 13),
        event_calendar_path=event_calendar_path,
        config_path=config_path,
        v2_daily_ohlcv_path=v2_daily_path,
        macro_parquet_path=macro_path,
    )

    assert result["success_count"] == 1
    payload = json.loads((out_root / "outputs" / "2026-05-13.json").read_text())
    assert payload["credit_funding_state"] is not None
    assert payload["inflation_growth_state"] is not None


def test_historical_walkforward_runner_records_failures_without_silent_skip(
    tmp_path: Path,
    v2_macro_parquet_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    original = pd.read_parquet(
        repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    )
    original["date"] = pd.to_datetime(original["date"]).dt.date

    broken = original.copy()
    broken = broken[
        ~((broken["symbol"] == "SPY") & (broken["date"] == date(2023, 12, 14)))
    ]
    broken_path = tmp_path / "broken_market_data.parquet"
    broken.to_parquet(broken_path, index=False)

    out_root = tmp_path / "walkforward"
    mod = _load_runner_module()
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    v2_daily_path = repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    config_path = repo_root / "tests" / "fixtures" / "configs" / "core3-v2-fast.yaml"
    result = mod.run_walkforward(
        market_data_path=broken_path,
        output_root=out_root,
        start_date=date(2023, 12, 13),
        end_date=date(2023, 12, 14),
        event_calendar_path=event_calendar_path,
        config_path=config_path,
        v2_daily_ohlcv_path=v2_daily_path,
        macro_parquet_path=v2_macro_parquet_path,
    )

    assert result["session_count"] == 2
    assert result["success_count"] == 1
    assert result["failure_count"] == 1

    with closing(sqlite3.connect(out_root / "regime_walkforward.db")) as conn:
        rows = conn.execute(
            "SELECT as_of_date, status, failure_reason FROM runs ORDER BY as_of_date"
        ).fetchall()

    assert rows[0][0] == "2023-12-13"
    assert rows[0][1] == "success"
    assert rows[1][0] == "2023-12-14"
    assert rows[1][1] == "failure"
    assert "SPY row" in rows[1][2]


def test_historical_walkforward_failure_only_summary_keeps_success_columns(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    original = pd.read_parquet(
        repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    )
    original["date"] = pd.to_datetime(original["date"]).dt.date

    broken = original[original["symbol"] != "SPY"].copy()
    broken_path = tmp_path / "broken_market_data.parquet"
    broken.to_parquet(broken_path, index=False)

    out_root = tmp_path / "walkforward"
    mod = _load_runner_module()
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    config_path = repo_root / "tests" / "fixtures" / "configs" / "core3-v2-fast.yaml"
    result = mod.run_walkforward(
        market_data_path=broken_path,
        output_root=out_root,
        start_date=date(2023, 12, 14),
        end_date=date(2023, 12, 14),
        event_calendar_path=event_calendar_path,
        config_path=config_path,
    )

    assert result["success_count"] == 0
    assert result["failure_count"] == 1
    summary_df = pd.read_csv(out_root / "reports" / "walkforward_summary.csv")
    assert {
        "classification_coverage",
        "rule_provenance",
    }.issubset(summary_df.columns)


def test_historical_walkforward_requires_event_calendar_unless_explicitly_allowed(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    market_path = repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    out_root = tmp_path / "walkforward"
    mod = _load_runner_module()

    with pytest.raises(ValueError, match="event_calendar_path is required"):
        mod.run_walkforward(
            market_data_path=market_path,
            output_root=out_root,
            start_date=date(2023, 12, 13),
            end_date=date(2023, 12, 14),
        )
    assert not out_root.exists()


def test_historical_walkforward_cli_defaults_event_calendar_to_manifest_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_runner_module()
    data_root = tmp_path / "data" / "raw"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_historical_walkforward.py",
            "--market-data",
            str(tmp_path / "market.parquet"),
            "--output-root",
            str(tmp_path / "out"),
            "--start-date",
            "2023-12-13",
            "--end-date",
            "2023-12-14",
            "--data-root",
            str(data_root),
        ],
    )

    args = mod._parse_args()

    assert args.event_calendar == data_root / "event_calendar" / "us_events.yaml"
