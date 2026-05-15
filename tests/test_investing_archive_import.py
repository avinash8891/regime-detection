from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from regime_data_fetch.investing_archive import run_local_investing_archive_import


def test_run_local_investing_archive_import_records_raw_and_canonical_artifacts(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _write_archive_fixture(archive_root)
    out_dir = tmp_path / "data" / "raw"
    db_path = out_dir / "acquisition" / "acquisition.db"

    report_path = run_local_investing_archive_import(
        out_dir=out_dir,
        archive_root=archive_root,
        acquisition_db_path=db_path,
        artifact_store_root=tmp_path / "store",
    )

    report = json.loads(report_path.read_text())
    assert report["counts"] == {
        "economic_events_rows": 2,
        "holiday_rows": 1,
        "earnings_rows": 2,
        "raw_files": 9,
    }
    assert report["date_range"]["economic_events"] == {"min_date": "2016-01-01", "max_date": "2016-01-02"}
    assert report["paths"]["raw_archive"]["local_path"] == "data/raw/investing/raw_archive"
    assert pd.read_parquet(out_dir / "investing" / "economic_events.parquet").shape[0] == 2
    assert pd.read_parquet(out_dir / "investing" / "holidays.parquet").shape[0] == 1
    assert pd.read_parquet(out_dir / "investing" / "earnings.parquet").shape[0] == 2

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall() == [
            ("investing_archive_local", "ok")
        ]
        sources = conn.execute(
            """
            SELECT source_name, artifact_kind, count(*)
            FROM artifact_records
            GROUP BY source_name, artifact_kind
            ORDER BY source_name, artifact_kind
            """
        ).fetchall()
        assert ("investing.com", "parquet", 3) in sources
        assert ("investing.com:archive", "csv", 3) in sources
        assert ("investing.com:archive", "json", 3) in sources
        assert ("investing.com:archive", "jsonl", 3) in sources
        outputs = conn.execute(
            "SELECT output_kind, row_count, min_date, max_date FROM derived_outputs ORDER BY output_kind"
        ).fetchall()
        assert ("investing_earnings_parquet", 2, "2016-01-04", "2016-01-05") in outputs
        assert ("investing_archive_import_report", 5, "2016-01-01", "2016-01-05") in outputs


def _write_archive_fixture(root: Path) -> None:
    calendar = root / "investing_calendar_structured_2016_2026"
    earnings = root / "investing_earnings_2016_2026"
    raw_instruments = earnings / "raw_instruments"
    calendar.mkdir(parents=True)
    raw_instruments.mkdir(parents=True)

    (calendar / "investing_economic_events_2016-01-01_2026-05-15.csv").write_text(
        "\n".join(
            [
                "event,occurrence_time_utc,country,kind",
                "A,2016-01-01T00:00:00Z,US,event",
                "B,2016-01-02T00:00:00Z,US,event",
            ]
        )
        + "\n"
    )
    (calendar / "investing_holidays_2016-01-01_2026-05-15.csv").write_text(
        "name,holiday_start_utc,country,kind\nNew Year,2016-01-01T00:00:00Z,US,holiday\n"
    )
    (calendar / "investing_calendar_combined_2016-01-01_2026-05-15.jsonl").write_text(
        '{"kind":"event"}\n'
    )
    (calendar / "fetch_report.json").write_text('{"ok": true}\n')
    (earnings / "investing_earnings_2016-01-01_2026-05-15.csv").write_text(
        "\n".join(
            [
                "symbol,date,company,kind",
                "AAPL,2016-01-04,Apple,earnings",
                "MSFT,2016-01-05,Microsoft,earnings",
            ]
        )
        + "\n"
    )
    (earnings / "investing_earnings_2016-01-01_2026-05-15.jsonl").write_text(
        '{"kind":"earnings"}\n'
    )
    (earnings / "quarantine_earnings_fetch_errors.jsonl").write_text('{"error":"sample"}\n')
    (earnings / "fetch_report.json").write_text('{"ok": true}\n')
    (raw_instruments / "instruments_batch_0001.json").write_text('{"items":[]}\n')
