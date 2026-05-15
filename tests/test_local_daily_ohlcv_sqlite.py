from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from regime_data_fetch.local_daily_ohlcv_sqlite import run_local_daily_ohlcv_sqlite_import


def test_run_local_daily_ohlcv_sqlite_import_records_rows_and_artifacts(tmp_path: Path) -> None:
    source_dir = tmp_path / "daily_ohlcv_762"
    symbol_dir = source_dir / "symbol=AAPL"
    symbol_dir.mkdir(parents=True)
    parquet_path = symbol_dir / "part-0.parquet"
    pd.DataFrame(
        [
            {
                "date": "2026-05-05",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
                "adjusted_close": 100.5,
            },
            {
                "date": "2026-05-06",
                "open": 100.5,
                "high": 101.5,
                "low": 100.0,
                "close": 101.0,
                "volume": 1200,
                "adjusted_close": 101.0,
            },
        ]
    ).to_parquet(parquet_path, index=False)

    acquisition_db = tmp_path / "acquisition.db"
    report_path = run_local_daily_ohlcv_sqlite_import(
        out_dir=tmp_path,
        source_dir=source_dir,
        acquisition_db_path=acquisition_db,
    )

    report = json.loads(report_path.read_text())
    assert report["counts"]["parquet_files"] == 1
    assert report["counts"]["symbols"] == 1
    assert report["counts"]["imported_rows"] == 2
    assert report["date_range"]["min_date"] == "2026-05-05"
    assert report["date_range"]["max_date"] == "2026-05-06"
    assert report["paths"]["profile_constituent_tree"] == {
        "path": str(source_dir),
        "local_path": "data/raw/daily_ohlcv_762",
    }

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall()
        artifact_sources = conn.execute("SELECT source_name, artifact_kind FROM artifacts").fetchall()
        outputs = conn.execute("SELECT output_kind FROM derived_outputs ORDER BY output_id").fetchall()
        ohlcv_rows = conn.execute(
            "SELECT symbol, date, close FROM daily_ohlcv_rows ORDER BY symbol, date"
        ).fetchall()

    assert fetch_runs == [("daily_ohlcv_local_sqlite", "ok")]
    assert artifact_sources == [("local:daily_ohlcv", "parquet_local")]
    assert outputs == [("daily_ohlcv_local_sqlite_import_report",)]
    assert ohlcv_rows == [
        ("AAPL", "2026-05-05", 100.5),
        ("AAPL", "2026-05-06", 101.0),
    ]
