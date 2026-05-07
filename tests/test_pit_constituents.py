from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sqlite3

import pandas as pd

from regime_data_fetch.pit_constituents import (
    PITConstituentFetchError,
    parse_sp500_ticker_start_end_csv,
    run_pit_constituents_fetch,
)


FIXTURES = Path("tests/fixtures/raw/pit")


def test_parse_sp500_ticker_start_end_csv_extracts_intervals() -> None:
    csv_text = (FIXTURES / "sp500_ticker_start_end.csv").read_text()

    rows = parse_sp500_ticker_start_end_csv(
        csv_text,
        source_url="https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv",
    )

    assert len(rows) == 12
    assert rows[0].ticker == "A"
    assert rows[0].start_date == dt.date(2000, 6, 5)
    assert rows[0].end_date is None
    assert rows[2].ticker == "AAL"
    assert rows[2].end_date == dt.date(1997, 1, 15)
    assert rows[-1].ticker == "ZBRA"
    assert rows[-1].source == "fja05680/sp500"


def test_run_pit_constituents_fetch_writes_parquet_and_report(tmp_path: Path) -> None:
    csv_text = (FIXTURES / "sp500_ticker_start_end.csv").read_text()

    def fake_fetcher() -> str:
        return csv_text

    report_path = run_pit_constituents_fetch(
        out_dir=tmp_path,
        csv_fetcher=fake_fetcher,
    )

    report = json.loads(report_path.read_text())
    assert report["counts"]["rows"] == 12
    assert report["counts"]["open_intervals"] == 4
    assert report["bias_warning"] == "survivorship_biased_constituent_universe"
    assert report["paths"]["pit_constituents_parquet"] == str(tmp_path / "pit_constituents" / "sp500_ticker_intervals.parquet")

    df = pd.read_parquet(tmp_path / "pit_constituents" / "sp500_ticker_intervals.parquet")
    assert list(df.columns) == [
        "ticker",
        "start_date",
        "end_date",
        "source",
        "source_url",
        "bias_warning",
    ]
    assert df.iloc[0]["ticker"] == "A"
    assert df.iloc[0]["bias_warning"] == "survivorship_biased_constituent_universe"


def test_run_pit_constituents_fetch_raises_on_invalid_csv(tmp_path: Path) -> None:
    def bad_fetcher() -> str:
        return "ticker,start_date,end_date\nA,not-a-date,\n"

    try:
        run_pit_constituents_fetch(
            out_dir=tmp_path,
            csv_fetcher=bad_fetcher,
        )
    except PITConstituentFetchError as exc:
        assert "invalid start_date" in str(exc).lower()
    else:
        raise AssertionError("Expected PITConstituentFetchError")


def test_run_pit_constituents_fetch_records_raw_csv_and_outputs_in_sqlite(tmp_path: Path) -> None:
    acquisition_db = tmp_path / "acquisition.db"
    csv_text = "\n".join(
        [
            "ticker,start_date,end_date",
            "MMM,1976-08-09,",
            "AAPL,1982-11-30,",
        ]
    )

    report_path = run_pit_constituents_fetch(
        out_dir=tmp_path,
        csv_fetcher=lambda: csv_text,
        acquisition_db_path=acquisition_db,
    )

    report = json.loads(report_path.read_text())
    assert report["paths"]["acquisition_db"] == str(acquisition_db)

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall()
        artifacts = conn.execute(
            "SELECT source_name, artifact_kind, count(*) FROM artifacts GROUP BY source_name, artifact_kind"
        ).fetchall()
        outputs = conn.execute("SELECT output_kind FROM derived_outputs ORDER BY output_id").fetchall()

    assert fetch_runs == [("pit_constituents", "ok")]
    assert artifacts == [("github_raw:sp500_ticker_start_end", "csv", 1)]
    assert outputs == [
        ("pit_constituents_parquet",),
        ("pit_constituents_report",),
    ]
