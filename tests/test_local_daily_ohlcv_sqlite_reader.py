from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_data_fetch.local_daily_ohlcv_sqlite import EXPECTED_COLUMNS
from regime_data_fetch.local_daily_ohlcv_sqlite_reader import (
    LocalDailyOHLCVReadError,
    read_constituent_ohlcv,
)


# AAPL session prices (hand-picked plausible values for 2024-01-02..2024-01-10).
_AAPL_ROWS: list[tuple[str, float, float, float, float, int, float]] = [
    ("2024-01-02", 187.15, 188.44, 183.89, 185.64, 82488700, 185.40),
    ("2024-01-03", 184.22, 185.88, 183.43, 184.25, 58414500, 184.02),
    ("2024-01-04", 182.15, 183.09, 180.88, 181.91, 71983600, 181.69),
    ("2024-01-05", 181.99, 182.76, 180.17, 181.18, 62303300, 180.96),
    ("2024-01-08", 182.09, 185.60, 181.50, 185.56, 59144500, 185.34),
    ("2024-01-09", 183.92, 185.15, 182.73, 185.14, 42841800, 184.92),
    ("2024-01-10", 184.35, 186.40, 183.92, 186.19, 46792900, 185.97),
]


# MSFT session prices (hand-picked plausible values for 2024-01-02..2024-01-08).
_MSFT_ROWS: list[tuple[str, float, float, float, float, int, float]] = [
    ("2024-01-02", 373.86, 375.90, 366.50, 370.87, 25258600, 369.80),
    ("2024-01-03", 369.30, 371.95, 367.35, 370.60, 23083500, 369.53),
    ("2024-01-04", 370.62, 373.48, 365.78, 367.94, 20901500, 366.88),
    ("2024-01-05", 368.55, 374.66, 367.50, 367.75, 21302500, 366.69),
    ("2024-01-08", 369.30, 376.50, 369.00, 374.69, 22765200, 373.61),
]


def _create_real_schema_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS daily_ohlcv_rows (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                adjusted_close REAL NOT NULL,
                source_file TEXT NOT NULL,
                PRIMARY KEY (symbol, date)
            );
            CREATE INDEX IF NOT EXISTS idx_daily_ohlcv_rows_date
                ON daily_ohlcv_rows (date);
            """
        )

        def _rows_for(symbol: str, rows):
            return [
                (symbol, d, o, h, l, c, v, ac, f"fixture://{symbol}.parquet")
                for (d, o, h, l, c, v, ac) in rows
            ]

        conn.executemany(
            """
            INSERT INTO daily_ohlcv_rows (
                symbol, date, open, high, low, close, volume, adjusted_close, source_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _rows_for("AAPL", _AAPL_ROWS) + _rows_for("MSFT", _MSFT_ROWS),
        )
        conn.commit()


def test_read_constituent_ohlcv_returns_dict_keyed_by_ticker(tmp_path: Path) -> None:
    db = tmp_path / "ohlcv.db"
    _create_real_schema_db(db)

    result = read_constituent_ohlcv(
        db, ["AAPL", "MSFT"], dt.date(2024, 1, 2), dt.date(2024, 1, 10)
    )

    assert set(result.keys()) == {"AAPL", "MSFT"}
    assert isinstance(result["AAPL"], pd.DataFrame)
    assert isinstance(result["MSFT"], pd.DataFrame)
    assert len(result["AAPL"]) == 7
    assert len(result["MSFT"]) == 5


def test_read_constituent_ohlcv_dataframe_has_expected_columns(tmp_path: Path) -> None:
    db = tmp_path / "ohlcv.db"
    _create_real_schema_db(db)

    result = read_constituent_ohlcv(
        db, ["AAPL"], dt.date(2024, 1, 2), dt.date(2024, 1, 10)
    )

    df = result["AAPL"]
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "adjusted_close"]
    # Sanity-check alignment with the writer's EXPECTED_COLUMNS (date is the index).
    assert list(df.columns) == [c for c in EXPECTED_COLUMNS if c != "date"]


def test_read_constituent_ohlcv_index_is_datetimeindex(tmp_path: Path) -> None:
    db = tmp_path / "ohlcv.db"
    _create_real_schema_db(db)

    result = read_constituent_ohlcv(
        db, ["AAPL"], dt.date(2024, 1, 2), dt.date(2024, 1, 10)
    )

    df = result["AAPL"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None
    # First row should be 2024-01-02 normalized to midnight (UTC-naive).
    assert df.index[0] == pd.Timestamp("2024-01-02")


def test_read_constituent_ohlcv_filters_by_date_range_inclusive(tmp_path: Path) -> None:
    db = tmp_path / "ohlcv.db"
    _create_real_schema_db(db)

    result = read_constituent_ohlcv(
        db, ["AAPL"], dt.date(2024, 1, 3), dt.date(2024, 1, 8)
    )

    df = result["AAPL"]
    # AAPL sessions within [2024-01-03, 2024-01-08]: 01-03, 01-04, 01-05, 01-08 = 4 rows.
    # (01-06 and 01-07 are weekend / non-session, not in fixture.)
    # Re-checked: rows are 01-03, 01-04, 01-05, 01-08 -> 4 inclusive both ends.
    assert len(df) == 4
    assert df.index[0] == pd.Timestamp("2024-01-03")
    assert df.index[-1] == pd.Timestamp("2024-01-08")


def test_read_constituent_ohlcv_omits_tickers_not_in_store(tmp_path: Path) -> None:
    db = tmp_path / "ohlcv.db"
    _create_real_schema_db(db)
    # Drop MSFT rows so only AAPL is in the store.
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM daily_ohlcv_rows WHERE symbol = 'MSFT'")
        conn.commit()

    result = read_constituent_ohlcv(
        db, ["AAPL", "TSLA"], dt.date(2024, 1, 2), dt.date(2024, 1, 10)
    )

    assert set(result.keys()) == {"AAPL"}


def test_read_constituent_ohlcv_volume_is_int64(tmp_path: Path) -> None:
    db = tmp_path / "ohlcv.db"
    _create_real_schema_db(db)

    result = read_constituent_ohlcv(
        db, ["AAPL"], dt.date(2024, 1, 2), dt.date(2024, 1, 10)
    )

    assert result["AAPL"]["volume"].dtype == np.int64


def test_read_constituent_ohlcv_adjusted_close_is_float64(tmp_path: Path) -> None:
    db = tmp_path / "ohlcv.db"
    _create_real_schema_db(db)

    result = read_constituent_ohlcv(
        db, ["AAPL"], dt.date(2024, 1, 2), dt.date(2024, 1, 10)
    )

    assert result["AAPL"]["adjusted_close"].dtype == np.float64


def test_read_constituent_ohlcv_raises_filenotfound_on_missing_db(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.db"

    with pytest.raises(FileNotFoundError):
        read_constituent_ohlcv(missing, ["AAPL"], dt.date(2024, 1, 2), dt.date(2024, 1, 10))


def test_read_constituent_ohlcv_raises_on_wrong_schema(tmp_path: Path) -> None:
    db = tmp_path / "wrong.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE unrelated_table (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO unrelated_table (payload) VALUES ('not ohlcv')")
        conn.commit()

    with pytest.raises(LocalDailyOHLCVReadError):
        read_constituent_ohlcv(db, ["AAPL"], dt.date(2024, 1, 2), dt.date(2024, 1, 10))


def test_read_constituent_ohlcv_handles_empty_ticker_list(tmp_path: Path) -> None:
    db = tmp_path / "ohlcv.db"
    _create_real_schema_db(db)

    result = read_constituent_ohlcv(db, [], dt.date(2024, 1, 2), dt.date(2024, 1, 10))

    assert result == {}
