from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from regime_data_fetch.pit_constituents import (
    PITConstituentFetchError,
    parse_sp500_ticker_start_end_csv,
    read_pit_intervals,
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


def test_parse_sp500_ticker_start_end_csv_persists_raw_open_intervals() -> None:
    """GNXfc: the parser must NOT bake SOURCE_END_DATE_CORRECTIONS into the persisted
    source — correction-eligible open intervals stay open. Corrections are applied
    solely on read (read_pit_intervals patch-on-read) so the §1D/§10 survivorship gate
    validates a genuine SOURCE closure, not a write-time-fabricated one."""
    csv_text = "\n".join(
        [
            "ticker,start_date,end_date",
            "DAY,2024-02-01,",
            "HOLX,2016-03-30,",
            "CTRA,2021-10-04,",
            "AAPL,1982-11-30,",
        ]
    )

    rows = parse_sp500_ticker_start_end_csv(csv_text, source_url="source")
    by_ticker = {row.ticker: row for row in rows}

    # Correction-eligible tickers stay OPEN (raw), not closed at their correction dates.
    assert by_ticker["DAY"].end_date is None
    assert by_ticker["HOLX"].end_date is None
    assert by_ticker["CTRA"].end_date is None
    assert by_ticker["AAPL"].end_date is None


def test_run_pit_fetch_persists_raw_so_correction_only_source_fails_gate(
    tmp_path: Path,
) -> None:
    """GNXfc end-to-end: a source whose only closure candidates are
    SOURCE_END_DATE_CORRECTIONS tickers (open in the raw feed) is persisted RAW by the
    writer, so read_pit_intervals' survivorship gate rejects it as current-only — the
    write path can no longer fabricate closures that fool the gate."""
    csv_text = "\n".join(
        [
            "ticker,start_date,end_date",
            "DAY,2024-02-01,",  # correction-eligible, but OPEN in the raw source
            "AAPL,1982-11-30,",  # genuinely open
        ]
    )
    out_dir = tmp_path / "out"
    run_pit_constituents_fetch(out_dir=out_dir, csv_fetcher=lambda: csv_text)
    parquet_path = out_dir / "pit_constituents" / "sp500_ticker_intervals.parquet"

    # Persisted RAW: DAY stays open (not closed at its 2026-02-03 correction).
    written = pd.read_parquet(parquet_path)
    assert bool(written.loc[written["ticker"] == "DAY", "end_date"].isna().all())

    # The survivorship gate therefore rejects the correction-only universe.
    with pytest.raises(ValueError, match="source contains no closed intervals"):
        read_pit_intervals(parquet_path)


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
    assert report["paths"]["pit_constituents_parquet"] == str(
        tmp_path / "pit_constituents" / "sp500_ticker_intervals.parquet"
    )

    df = pd.read_parquet(
        tmp_path / "pit_constituents" / "sp500_ticker_intervals.parquet"
    )
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


def test_run_pit_constituents_fetch_records_raw_csv_and_outputs_in_sqlite(
    tmp_path: Path,
) -> None:
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

    with closing(sqlite3.connect(acquisition_db)) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status FROM fetch_runs"
        ).fetchall()
        artifacts = conn.execute(
            "SELECT source_name, artifact_kind, count(*) FROM artifacts GROUP BY source_name, artifact_kind"
        ).fetchall()
        outputs = conn.execute(
            "SELECT output_kind FROM derived_outputs ORDER BY output_id"
        ).fetchall()

    assert fetch_runs == [("pit_constituents", "ok")]
    assert artifacts == [("github_raw:sp500_ticker_start_end", "csv", 1)]
    assert outputs == [
        ("pit_constituents_parquet",),
        ("pit_constituents_report",),
    ]
