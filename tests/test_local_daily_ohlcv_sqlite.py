from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from regime_data_fetch.alpaca_daily import DailyBarsFetchResult
from regime_data_fetch.local_daily_ohlcv_sqlite import (
    run_local_daily_ohlcv_sqlite_import,
)
from regime_data_fetch.local_daily_ohlcv_sqlite import (
    run_alpaca_constituent_daily_ohlcv_fetch,
)
from regime_data_fetch.universe import (
    FIXED_UNIVERSE_LOCAL_PATH,
    FIXED_UNIVERSE_TREE_NAME,
)


def test_run_local_daily_ohlcv_sqlite_import_records_rows_and_artifacts(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / FIXED_UNIVERSE_TREE_NAME
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
        "local_path": FIXED_UNIVERSE_LOCAL_PATH,
    }

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status FROM fetch_runs"
        ).fetchall()
        artifact_sources = conn.execute(
            "SELECT source_name, artifact_kind FROM artifacts"
        ).fetchall()
        outputs = conn.execute(
            "SELECT output_kind FROM derived_outputs ORDER BY output_id"
        ).fetchall()
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


def test_run_alpaca_constituent_daily_ohlcv_fetch_materializes_profile_tree_and_sqlite(
    tmp_path: Path,
) -> None:
    pit_path = tmp_path / "pit_constituents.parquet"
    pd.DataFrame(
        [
            {"ticker": "MSFT", "start_date": "2015-01-01", "end_date": None},
            {"ticker": "AAPL", "start_date": "2015-01-01", "end_date": "2026-12-31"},
            {"ticker": "AAPL", "start_date": "2010-01-01", "end_date": "2014-12-31"},
        ]
    ).to_parquet(pit_path, index=False)

    def fake_fetcher(**kwargs) -> DailyBarsFetchResult:
        assert kwargs["symbols"] == ["AAPL", "MSFT"]
        assert kwargs["start_date"].isoformat() == "2026-05-05"
        assert kwargs["end_date"].isoformat() == "2026-05-06"
        return DailyBarsFetchResult(
            df=pd.DataFrame(
                [
                    {
                        "date": "2026-05-05",
                        "symbol": "AAPL",
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 1000,
                        "adjusted_close": 100.5,
                    },
                    {
                        "date": "2026-05-06",
                        "symbol": "MSFT",
                        "open": 200.0,
                        "high": 202.0,
                        "low": 199.0,
                        "close": 201.5,
                        "volume": 2000,
                        "adjusted_close": 201.5,
                    },
                ]
            ),
            missing_symbols=[],
        )

    acquisition_db = tmp_path / "acquisition.db"
    report_path = run_alpaca_constituent_daily_ohlcv_fetch(
        out_dir=tmp_path / "data" / "raw",
        pit_parquet_path=pit_path,
        start=pd.Timestamp("2026-05-05").date(),
        end=pd.Timestamp("2026-05-06").date(),
        adjustment="split",
        alpaca_feed="sip",
        acquisition_db_path=acquisition_db,
        bars_fetcher=fake_fetcher,
        allow_pit_universe=True,
        expected_universe_count=None,
    )

    report = json.loads(report_path.read_text())
    assert report["counts"] == {
        "symbols_requested": 2,
        "symbols_returned": 2,
        "rows": 2,
        "missing_symbols": 0,
    }
    assert report["paths"]["profile_constituent_tree"] == {
        "path": str(tmp_path / "data" / "raw" / FIXED_UNIVERSE_TREE_NAME),
        "local_path": FIXED_UNIVERSE_LOCAL_PATH,
    }
    assert (
        tmp_path
        / "data"
        / "raw"
        / FIXED_UNIVERSE_TREE_NAME
        / "symbol=AAPL"
        / "ohlcv.parquet"
    ).exists()
    assert (
        tmp_path
        / "data"
        / "raw"
        / FIXED_UNIVERSE_TREE_NAME
        / "symbol=MSFT"
        / "ohlcv.parquet"
    ).exists()

    with sqlite3.connect(acquisition_db) as conn:
        ohlcv_rows = conn.execute(
            "SELECT symbol, date, close FROM daily_ohlcv_rows ORDER BY symbol, date"
        ).fetchall()
        fetch_runs = conn.execute(
            "SELECT fetch_type, status FROM fetch_runs ORDER BY run_id"
        ).fetchall()

    assert fetch_runs == [
        ("daily_ohlcv_constituents_alpaca", "ok"),
        ("daily_ohlcv_local_sqlite", "ok"),
    ]
    assert ohlcv_rows == [
        ("AAPL", "2026-05-05", 100.5),
        ("MSFT", "2026-05-06", 201.5),
    ]


def test_run_alpaca_constituent_daily_ohlcv_fetch_uses_fixed_universe_before_pit(
    tmp_path: Path,
) -> None:
    pit_path = tmp_path / "pit_constituents.parquet"
    pd.DataFrame(
        [
            {"ticker": "MSFT", "start_date": "2015-01-01", "end_date": None},
            {"ticker": "AAPL", "start_date": "2015-01-01", "end_date": None},
        ]
    ).to_parquet(pit_path, index=False)

    def fake_fetcher(**kwargs) -> DailyBarsFetchResult:
        assert kwargs["symbols"] == ["AAPL"]
        return DailyBarsFetchResult(
            df=pd.DataFrame(
                [
                    {
                        "date": "2026-05-05",
                        "symbol": "AAPL",
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 1000,
                        "adjusted_close": 100.5,
                    }
                ]
            ),
            missing_symbols=[],
        )

    report_path = run_alpaca_constituent_daily_ohlcv_fetch(
        out_dir=tmp_path / "data" / "raw",
        pit_parquet_path=pit_path,
        start=pd.Timestamp("2026-05-05").date(),
        end=pd.Timestamp("2026-05-05").date(),
        adjustment="split",
        alpaca_feed="sip",
        acquisition_db_path=tmp_path / "acquisition.db",
        bars_fetcher=fake_fetcher,
        fixed_universe_symbols=["AAPL"],
        expected_universe_count=1,
    )

    report = json.loads(report_path.read_text())
    assert report["universe_source"] == "fixed_symbol_list"
    assert report["counts"]["symbols_requested"] == 1


def test_run_alpaca_constituent_daily_ohlcv_fetch_merges_incremental_rows(
    tmp_path: Path,
) -> None:
    tree_root = tmp_path / "data" / "raw" / FIXED_UNIVERSE_TREE_NAME
    symbol_dir = tree_root / "symbol=AAPL"
    symbol_dir.mkdir(parents=True)
    (symbol_dir / "ohlcv.parquet").write_bytes(b"")
    pd.DataFrame(
        [
            {
                "date": "2026-05-04",
                "open": 99.0,
                "high": 100.0,
                "low": 98.0,
                "close": 99.5,
                "volume": 900,
                "adjusted_close": 99.5,
            },
            {
                "date": "2026-05-05",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
                "adjusted_close": 100.5,
            },
        ]
    ).to_parquet(symbol_dir / "ohlcv.parquet", index=False)

    def fake_fetcher(**kwargs) -> DailyBarsFetchResult:
        assert kwargs["symbols"] == ["AAPL"]
        assert kwargs["start_date"].isoformat() == "2026-05-05"
        return DailyBarsFetchResult(
            df=pd.DataFrame(
                [
                    {
                        "date": "2026-05-05",
                        "symbol": "AAPL",
                        "open": 100.1,
                        "high": 101.1,
                        "low": 99.1,
                        "close": 100.7,
                        "volume": 1100,
                        "adjusted_close": 100.7,
                    },
                    {
                        "date": "2026-05-06",
                        "symbol": "AAPL",
                        "open": 101.0,
                        "high": 102.0,
                        "low": 100.0,
                        "close": 101.5,
                        "volume": 1200,
                        "adjusted_close": 101.5,
                    },
                ]
            ),
            missing_symbols=[],
        )

    run_alpaca_constituent_daily_ohlcv_fetch(
        out_dir=tmp_path / "data" / "raw",
        pit_parquet_path=tmp_path / "unused.parquet",
        start=pd.Timestamp("2026-05-05").date(),
        end=pd.Timestamp("2026-05-06").date(),
        adjustment="split",
        alpaca_feed="sip",
        acquisition_db_path=tmp_path / "acquisition.db",
        bars_fetcher=fake_fetcher,
        fixed_universe_symbols=["AAPL"],
        expected_universe_count=1,
    )

    merged = pd.read_parquet(symbol_dir / "ohlcv.parquet").sort_values("date")
    assert merged[["date", "close"]].to_dict(orient="records") == [
        {"date": "2026-05-04", "close": 99.5},
        {"date": "2026-05-05", "close": 100.7},
        {"date": "2026-05-06", "close": 101.5},
    ]
