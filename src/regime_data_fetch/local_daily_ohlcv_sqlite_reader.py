from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Iterable
from contextlib import closing

import numpy as np
import pandas as pd

from regime_data_fetch.local_daily_ohlcv_sqlite import EXPECTED_COLUMNS
from regime_shared.pandas_compat import cow_safe_assign

DAILY_OHLCV_ROWS_TABLE = "daily_ohlcv_rows"
_REQUIRED_COLUMNS = frozenset(["symbol", "source_file", *EXPECTED_COLUMNS])
_VALUE_COLUMNS = [c for c in EXPECTED_COLUMNS if c != "date"]


class LocalDailyOHLCVReadError(RuntimeError):
    """Raised when the SQLite store has an unexpected schema."""


def read_constituent_ohlcv(
    db_path: Path,
    tickers: Iterable[str],
    start_date: dt.date,
    end_date: dt.date,
) -> dict[str, pd.DataFrame]:
    """Read per-ticker OHLCV from the local SQLite store.

    Returns a dict mapping ticker -> DataFrame for tickers present within
    ``[start_date, end_date]`` inclusive. Tickers absent from the store are
    silently omitted. An empty ticker list returns ``{}`` without querying.
    """
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    ticker_list = list(tickers)
    if not ticker_list:
        return {}

    with closing(sqlite3.connect(db_path)) as conn:
        _validate_schema(conn)

        placeholders = ",".join("?" for _ in ticker_list)
        query = (
            f"SELECT symbol, date, open, high, low, close, volume, adjusted_close "
            f"FROM {DAILY_OHLCV_ROWS_TABLE} "
            f"WHERE symbol IN ({placeholders}) AND date BETWEEN ? AND ? "
            f"ORDER BY symbol, date"
        )
        params = [*ticker_list, start_date.isoformat(), end_date.isoformat()]
        rows = pd.read_sql_query(query, conn, params=params)

    if rows.empty:
        return {}

    rows = cow_safe_assign(
        rows,
        {
            "date": pd.to_datetime(rows["date"]).dt.normalize(),
            "open": rows["open"].astype(np.float64),
            "high": rows["high"].astype(np.float64),
            "low": rows["low"].astype(np.float64),
            "close": rows["close"].astype(np.float64),
            "adjusted_close": rows["adjusted_close"].astype(np.float64),
            "volume": rows["volume"].astype(np.int64),
        },
    )

    result: dict[str, pd.DataFrame] = {}
    for symbol, group in rows.groupby("symbol", sort=False):
        frame = group.set_index("date")[_VALUE_COLUMNS].copy()
        frame.index.name = "date"
        result[str(symbol)] = frame
    return result


def _validate_schema(conn: sqlite3.Connection) -> None:
    cursor = conn.execute(f"PRAGMA table_info({DAILY_OHLCV_ROWS_TABLE})")
    info = cursor.fetchall()
    if not info:
        raise LocalDailyOHLCVReadError(
            f"Missing required table '{DAILY_OHLCV_ROWS_TABLE}' in SQLite store"
        )
    present = {row[1] for row in info}
    missing = _REQUIRED_COLUMNS - present
    if missing:
        raise LocalDailyOHLCVReadError(
            f"Table '{DAILY_OHLCV_ROWS_TABLE}' missing required columns: {sorted(missing)!r}"
        )
