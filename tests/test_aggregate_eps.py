from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from regime_data_fetch.aggregate_eps import (
    AggregateEPSFetchError,
    parse_sp500_eps_workbook,
    run_aggregate_eps_fetch,
)


FIXTURES = Path("tests/fixtures/raw/eps")


def test_parse_sp500_eps_workbook_extracts_current_and_historical_snapshots() -> None:
    parsed = parse_sp500_eps_workbook(FIXTURES / "sp500_eps_est_fixture.xlsx")

    assert parsed.workbook_as_of_date == dt.date(2026, 1, 30)
    assert parsed.public_files_discontinued is True
    assert parsed.current_snapshot.observation_label == "current"
    assert parsed.current_snapshot.observation_date == dt.date(2026, 1, 30)
    assert parsed.current_snapshot.estimate_2025e == 265.41
    assert parsed.current_snapshot.estimate_2026e == 310.24
    assert parsed.current_snapshot.change_vs_prior_observation_2026e == 0.02362412564339467
    assert len(parsed.historical_snapshots) == 7
    assert parsed.historical_snapshots[0].observation_date == dt.date(2024, 3, 31)
    assert parsed.historical_snapshots[-1].observation_date == dt.date(2025, 9, 30)
    assert parsed.historical_snapshots[-1].estimate_2026e == 303.08


def test_run_aggregate_eps_fetch_writes_parquet_and_report(tmp_path: Path) -> None:
    report_path = run_aggregate_eps_fetch(
        out_dir=tmp_path,
        workbook_path=FIXTURES / "sp500_eps_est_fixture.xlsx",
    )

    report = json.loads(report_path.read_text())
    assert report["counts"]["historical_snapshots"] == 7
    assert report["counts"]["current_snapshots"] == 1
    assert report["current_snapshot"]["estimate_2026e"] == 310.24
    assert report["current_snapshot"]["change_vs_prior_observation_2026e"] == 0.02362412564339467
    assert report["limitations"]["aggregate_forward_eps_revision_direction_4w_available"] is False
    assert report["paths"]["aggregate_eps_parquet"] == str(
        tmp_path / "aggregate_forward_eps" / "sp500_eps_snapshots.parquet"
    )

    df = pd.read_parquet(tmp_path / "aggregate_forward_eps" / "sp500_eps_snapshots.parquet")
    assert list(df.columns) == [
        "workbook_as_of_date",
        "observation_date",
        "observation_label",
        "estimate_2025e",
        "estimate_q4_2025e",
        "estimate_2026e",
        "price",
        "pe_2025e",
        "pe_2026e",
        "change_vs_prior_observation_2025e",
        "change_vs_prior_observation_q4_2025e",
        "change_vs_prior_observation_2026e",
        "change_vs_prior_observation_price",
        "change_vs_prior_observation_pe_2025e",
        "change_vs_prior_observation_pe_2026e",
        "source",
        "source_path",
        "public_files_discontinued",
    ]
    current = df[df["observation_label"] == "current"].iloc[0]
    assert current["observation_date"] == dt.date(2026, 1, 30)
    assert current["estimate_2026e"] == 310.24
    assert bool(current["public_files_discontinued"]) is True


def test_parse_sp500_eps_workbook_raises_when_expected_sheet_is_missing(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.xlsx"
    wb = Workbook()
    wb.active.title = "Sheet1"
    wb.save(bad_path)

    try:
        parse_sp500_eps_workbook(bad_path)
    except AggregateEPSFetchError as exc:
        assert "ESTIMATES&PEs" in str(exc)
    else:
        raise AssertionError("Expected AggregateEPSFetchError")
