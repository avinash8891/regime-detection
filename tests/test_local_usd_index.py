from __future__ import annotations

import csv
import datetime as dt
import json
import sqlite3
from pathlib import Path

from regime_data_fetch.local_usd_index import load_yahoo_usd_index_csv, run_local_usd_index_import


def test_load_yahoo_usd_index_csv_validates_and_normalizes(tmp_path: Path) -> None:
    csv_path = tmp_path / "NYICDX_history.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        writer.writerow(["2026-05-05", "100.0", "101.0", "99.0", "100.5", "100.5", "0"])
        writer.writerow(["2026-05-06", "100.5", "101.5", "100.0", "101.0", "101.0", "0"])

    result = load_yahoo_usd_index_csv(csv_path)
    frame = result.frame

    assert list(frame.columns) == ["date", "symbol", "open", "high", "low", "close", "adjusted_close", "volume", "source"]
    assert len(frame) == 2
    assert frame.iloc[0]["symbol"] == "^NYICDX"
    assert frame.iloc[0]["source"] == "yahoo_finance"
    assert result.quarantined_rows == []


def test_load_yahoo_usd_index_csv_quarantines_blank_rows_under_threshold(tmp_path: Path) -> None:
    csv_path = tmp_path / "NYICDX_history.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        for i in range(100):
            day = dt.date(2026, 1, 1) + dt.timedelta(days=i)
            writer.writerow([day.isoformat(), "100.0", "101.0", "99.0", "100.5", "100.5", "0"])
        writer.writerow(["2026-04-11", "", "", "", "", "", ""])

    result = load_yahoo_usd_index_csv(csv_path)

    assert len(result.frame) == 100
    assert len(result.quarantined_rows) == 1
    assert result.quarantined_rows[0]["reason"] == "blank_price_row"


def test_run_local_usd_index_import_records_sqlite_artifact_and_outputs(tmp_path: Path) -> None:
    csv_path = tmp_path / "NYICDX_history.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        writer.writerow(["2026-05-05", "100.0", "101.0", "99.0", "100.5", "100.5", "0"])
        writer.writerow(["2026-05-06", "100.5", "101.5", "100.0", "101.0", "101.0", "0"])

    acquisition_db = tmp_path / "acquisition.db"
    report_path = run_local_usd_index_import(
        out_dir=tmp_path,
        csv_path=csv_path,
        acquisition_db_path=acquisition_db,
    )

    report = json.loads(report_path.read_text())
    assert report["symbol"] == "^NYICDX"
    assert report["counts"]["rows"] == 2
    assert report["counts"]["quarantined_rows"] == 0
    assert report["paths"]["acquisition_db"] == str(acquisition_db)

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall()
        artifact_sources = conn.execute("SELECT source_name, artifact_kind FROM artifacts").fetchall()
        outputs = conn.execute("SELECT output_kind FROM derived_outputs ORDER BY output_id").fetchall()

    assert fetch_runs == [("usd_index_local", "ok")]
    assert artifact_sources == [("yahoo:^NYICDX", "csv_manual")]
    assert outputs == [
        ("usd_index_parquet",),
        ("usd_index_report",),
    ]
