"""Import derived acquisition outputs into normalized consolidation tables."""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

EVENT_CALENDAR_ROWS_TABLE = "event_calendar_rows"
MACRO_ROWS_TABLE = "macro_rows"
PMI_ROWS_TABLE = "pmi_rows"
PIT_CONSTITUENT_ROWS_TABLE = "pit_constituent_rows"
FOMC_MINUTES_ROWS_TABLE = "fomc_minutes_rows"
POWELL_SPEECHES_ROWS_TABLE = "powell_speeches_rows"
USD_INDEX_ROWS_TABLE = "usd_index_rows"
AGGREGATE_EPS_SNAPSHOT_ROWS_TABLE = "aggregate_eps_snapshot_rows"
AGGREGATE_EPS_WAYBACK_ROWS_TABLE = "aggregate_eps_wayback_rows"
ALPACA_MARKET_ROWS_TABLE = "alpaca_market_rows"

_NORMALIZED_TABLES = (
    EVENT_CALENDAR_ROWS_TABLE,
    MACRO_ROWS_TABLE,
    PMI_ROWS_TABLE,
    PIT_CONSTITUENT_ROWS_TABLE,
    FOMC_MINUTES_ROWS_TABLE,
    POWELL_SPEECHES_ROWS_TABLE,
    USD_INDEX_ROWS_TABLE,
    AGGREGATE_EPS_SNAPSHOT_ROWS_TABLE,
    AGGREGATE_EPS_WAYBACK_ROWS_TABLE,
    ALPACA_MARKET_ROWS_TABLE,
)


def _ensure_normalized_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS event_calendar_rows (
            run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
            event_date TEXT NOT NULL,
            release_timestamp_et TEXT NOT NULL,
            market TEXT NOT NULL,
            event_type TEXT NOT NULL,
            importance TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (run_id, event_date, event_type, release_timestamp_et)
        );
        CREATE INDEX IF NOT EXISTS idx_event_calendar_rows_date
            ON event_calendar_rows (event_date);

        CREATE TABLE IF NOT EXISTS macro_rows (
            run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
            dataset_kind TEXT NOT NULL,
            date TEXT NOT NULL,
            series_id TEXT NOT NULL,
            value REAL,
            realtime_start TEXT,
            realtime_end TEXT,
            logical_name TEXT NOT NULL,
            PRIMARY KEY (run_id, dataset_kind, date, series_id, realtime_start, realtime_end)
        );
        CREATE INDEX IF NOT EXISTS idx_macro_rows_date
            ON macro_rows (date);

        CREATE TABLE IF NOT EXISTS pmi_rows (
            run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
            dataset_kind TEXT NOT NULL,
            series_name TEXT NOT NULL,
            period TEXT NOT NULL,
            value REAL NOT NULL,
            release_timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            PRIMARY KEY (run_id, dataset_kind, series_name, period)
        );
        CREATE INDEX IF NOT EXISTS idx_pmi_rows_period
            ON pmi_rows (period);

        CREATE TABLE IF NOT EXISTS pit_constituent_rows (
            run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
            ticker TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            bias_warning TEXT NOT NULL,
            PRIMARY KEY (run_id, ticker, start_date)
        );

        CREATE TABLE IF NOT EXISTS fomc_minutes_rows (
            run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
            meeting_end_date TEXT NOT NULL,
            release_timestamp TEXT NOT NULL,
            title TEXT NOT NULL,
            meeting_date_text TEXT NOT NULL,
            body_text TEXT NOT NULL,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            pdf_url TEXT,
            PRIMARY KEY (run_id, meeting_end_date, release_timestamp)
        );

        CREATE TABLE IF NOT EXISTS powell_speeches_rows (
            run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
            speech_date TEXT NOT NULL,
            publication_timestamp TEXT NOT NULL,
            publication_timestamp_precision TEXT NOT NULL,
            title TEXT NOT NULL,
            speaker TEXT NOT NULL,
            location TEXT NOT NULL,
            body_text TEXT NOT NULL,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            PRIMARY KEY (run_id, speech_date, source_url)
        );

        CREATE TABLE IF NOT EXISTS usd_index_rows (
            run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            adjusted_close REAL NOT NULL,
            volume INTEGER NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (run_id, symbol, date)
        );
        CREATE INDEX IF NOT EXISTS idx_usd_index_rows_date
            ON usd_index_rows (date);

        CREATE TABLE IF NOT EXISTS aggregate_eps_snapshot_rows (
            run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
            workbook_as_of_date TEXT NOT NULL,
            observation_date TEXT NOT NULL,
            observation_label TEXT NOT NULL,
            forward_estimate_label TEXT NOT NULL,
            forward_estimate_value REAL,
            estimate_2025e REAL,
            estimate_q4_2025e REAL,
            estimate_2026e REAL,
            price REAL,
            pe_2025e REAL,
            pe_2026e REAL,
            change_vs_prior_observation_2025e REAL,
            change_vs_prior_observation_q4_2025e REAL,
            change_vs_prior_observation_2026e REAL,
            change_vs_prior_observation_price REAL,
            change_vs_prior_observation_pe_2025e REAL,
            change_vs_prior_observation_pe_2026e REAL,
            source TEXT NOT NULL,
            source_path TEXT NOT NULL,
            public_files_discontinued INTEGER NOT NULL,
            PRIMARY KEY (run_id, workbook_as_of_date, observation_date, observation_label)
        );

        CREATE TABLE IF NOT EXISTS aggregate_eps_wayback_rows (
            run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
            snapshot_date TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            archive_url TEXT NOT NULL,
            workbook_as_of_date TEXT NOT NULL,
            forward_estimate_label TEXT NOT NULL,
            forward_estimate_value REAL,
            estimate_2025e REAL,
            estimate_q4_2025e REAL,
            estimate_2026e REAL,
            price REAL,
            pe_2025e REAL,
            pe_2026e REAL,
            change_vs_prior_observation_2025e REAL,
            change_vs_prior_observation_q4_2025e REAL,
            change_vs_prior_observation_2026e REAL,
            change_vs_prior_observation_price REAL,
            change_vs_prior_observation_pe_2025e REAL,
            change_vs_prior_observation_pe_2026e REAL,
            public_files_discontinued INTEGER NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (run_id, snapshot_date, timestamp)
        );

        CREATE TABLE IF NOT EXISTS alpaca_market_rows (
            run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            adjusted_close REAL NOT NULL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (run_id, symbol, date)
        );
        CREATE INDEX IF NOT EXISTS idx_alpaca_market_rows_date
            ON alpaca_market_rows (date);
        """
    )


def _import_normalized_output(
    *,
    dst_conn: sqlite3.Connection,
    run_id: int,
    output_kind: str,
    path: Path,
) -> str | None:
    if output_kind == "event_calendar_yaml":
        _import_event_calendar_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return EVENT_CALENDAR_ROWS_TABLE
    if output_kind == "fred_macro_parquet":
        _import_macro_rows(dst_conn=dst_conn, run_id=run_id, path=path, dataset_kind="series")
        return MACRO_ROWS_TABLE
    if output_kind == "fred_cpi_vintages_parquet":
        _import_macro_rows(dst_conn=dst_conn, run_id=run_id, path=path, dataset_kind="cpi_vintages")
        return MACRO_ROWS_TABLE
    if output_kind == "pmi_parquet":
        _import_pmi_rows(dst_conn=dst_conn, run_id=run_id, path=path, dataset_kind="latest")
        return PMI_ROWS_TABLE
    if output_kind == "pmi_history_parquet":
        _import_pmi_rows(dst_conn=dst_conn, run_id=run_id, path=path, dataset_kind="history")
        return PMI_ROWS_TABLE
    if output_kind == "pit_constituents_parquet":
        _import_pit_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return PIT_CONSTITUENT_ROWS_TABLE
    if output_kind == "fomc_minutes_parquet":
        _import_fomc_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return FOMC_MINUTES_ROWS_TABLE
    if output_kind == "powell_speeches_parquet":
        _import_powell_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return POWELL_SPEECHES_ROWS_TABLE
    if output_kind == "usd_index_parquet":
        _import_usd_index_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return USD_INDEX_ROWS_TABLE
    if output_kind == "aggregate_eps_parquet":
        _import_aggregate_eps_snapshot_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return AGGREGATE_EPS_SNAPSHOT_ROWS_TABLE
    if output_kind == "aggregate_eps_wayback_timeline":
        _import_aggregate_eps_wayback_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return AGGREGATE_EPS_WAYBACK_ROWS_TABLE
    if output_kind == "alpaca_daily_ohlcv_parquet":
        _import_alpaca_market_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return ALPACA_MARKET_ROWS_TABLE
    return None


def _import_event_calendar_rows(*, dst_conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    payload = _read_yaml_events(path)
    rows = [
        (
            run_id,
            row["date"],
            row["release_timestamp_et"],
            row["market"],
            row["type"],
            row["importance"],
            row["source"],
        )
        for row in payload
    ]
    dst_conn.executemany(
        """
        INSERT INTO event_calendar_rows (
            run_id, event_date, release_timestamp_et, market, event_type, importance, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _import_macro_rows(
    *,
    dst_conn: sqlite3.Connection,
    run_id: int,
    path: Path,
    dataset_kind: str,
) -> None:
    frame = _read_parquet(path)
    rows = [
        (
            run_id,
            dataset_kind,
            _sql_value(row["date"]),
            _sql_value(row["series_id"]),
            _sql_value(row["value"]),
            _sql_value(row["realtime_start"]),
            _sql_value(row["realtime_end"]),
            _sql_value(row["logical_name"]),
        )
        for row in frame.to_dict(orient="records")
    ]
    dst_conn.executemany(
        """
        INSERT INTO macro_rows (
            run_id, dataset_kind, date, series_id, value, realtime_start, realtime_end, logical_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _import_pmi_rows(
    *,
    dst_conn: sqlite3.Connection,
    run_id: int,
    path: Path,
    dataset_kind: str,
) -> None:
    frame = _read_parquet(path)
    rows = [
        (
            run_id,
            dataset_kind,
            _sql_value(row["series_name"]),
            _sql_value(row["period"]),
            _sql_value(row["value"]),
            _sql_value(row["release_timestamp"]),
            _sql_value(row["source"]),
            _sql_value(row["source_url"]),
        )
        for row in frame.to_dict(orient="records")
    ]
    dst_conn.executemany(
        """
        INSERT INTO pmi_rows (
            run_id, dataset_kind, series_name, period, value, release_timestamp, source, source_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _import_pit_rows(*, dst_conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    frame = _read_parquet(path)
    rows = [
        (
            run_id,
            _sql_value(row["ticker"]),
            _sql_value(row["start_date"]),
            _sql_value(row["end_date"]),
            _sql_value(row["source"]),
            _sql_value(row["source_url"]),
            _sql_value(row["bias_warning"]),
        )
        for row in frame.to_dict(orient="records")
    ]
    dst_conn.executemany(
        """
        INSERT INTO pit_constituent_rows (
            run_id, ticker, start_date, end_date, source, source_url, bias_warning
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _import_fomc_rows(*, dst_conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    frame = _read_parquet(path)
    rows = [
        (
            run_id,
            _sql_value(row["meeting_end_date"]),
            _sql_value(row["release_timestamp"]),
            _sql_value(row["title"]),
            _sql_value(row["meeting_date_text"]),
            _sql_value(row["body_text"]),
            _sql_value(row["source"]),
            _sql_value(row["source_url"]),
            _sql_value(row["pdf_url"]),
        )
        for row in frame.to_dict(orient="records")
    ]
    dst_conn.executemany(
        """
        INSERT INTO fomc_minutes_rows (
            run_id, meeting_end_date, release_timestamp, title, meeting_date_text, body_text, source, source_url, pdf_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _import_powell_rows(*, dst_conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    frame = _read_parquet(path)
    rows = [
        (
            run_id,
            _sql_value(row["speech_date"]),
            _sql_value(row["publication_timestamp"]),
            _sql_value(row["publication_timestamp_precision"]),
            _sql_value(row["title"]),
            _sql_value(row["speaker"]),
            _sql_value(row["location"]),
            _sql_value(row["body_text"]),
            _sql_value(row["source"]),
            _sql_value(row["source_url"]),
        )
        for row in frame.to_dict(orient="records")
    ]
    dst_conn.executemany(
        """
        INSERT INTO powell_speeches_rows (
            run_id, speech_date, publication_timestamp, publication_timestamp_precision, title, speaker, location, body_text, source, source_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _import_usd_index_rows(*, dst_conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    frame = _read_parquet(path)
    rows = [
        (
            run_id,
            _sql_value(row["date"]),
            _sql_value(row["symbol"]),
            _sql_value(row["open"]),
            _sql_value(row["high"]),
            _sql_value(row["low"]),
            _sql_value(row["close"]),
            _sql_value(row["adjusted_close"]),
            int(_sql_value(row["volume"]) or 0),
            _sql_value(row["source"]),
        )
        for row in frame.to_dict(orient="records")
    ]
    dst_conn.executemany(
        """
        INSERT INTO usd_index_rows (
            run_id, date, symbol, open, high, low, close, adjusted_close, volume, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _import_aggregate_eps_snapshot_rows(*, dst_conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    frame = _read_parquet(path)
    rows = [
        (
            run_id,
            _sql_value(row["workbook_as_of_date"]),
            _sql_value(row["observation_date"]),
            _sql_value(row["observation_label"]),
            _sql_value(row["forward_estimate_label"]),
            _sql_value(row["forward_estimate_value"]),
            _sql_value(row["estimate_2025e"]),
            _sql_value(row["estimate_q4_2025e"]),
            _sql_value(row["estimate_2026e"]),
            _sql_value(row["price"]),
            _sql_value(row["pe_2025e"]),
            _sql_value(row["pe_2026e"]),
            _sql_value(row["change_vs_prior_observation_2025e"]),
            _sql_value(row["change_vs_prior_observation_q4_2025e"]),
            _sql_value(row["change_vs_prior_observation_2026e"]),
            _sql_value(row["change_vs_prior_observation_price"]),
            _sql_value(row["change_vs_prior_observation_pe_2025e"]),
            _sql_value(row["change_vs_prior_observation_pe_2026e"]),
            _sql_value(row["source"]),
            _sql_value(row["source_path"]),
            int(bool(_sql_value(row["public_files_discontinued"]))),
        )
        for row in frame.to_dict(orient="records")
    ]
    dst_conn.executemany(
        """
        INSERT INTO aggregate_eps_snapshot_rows (
            run_id, workbook_as_of_date, observation_date, observation_label, forward_estimate_label,
            forward_estimate_value, estimate_2025e, estimate_q4_2025e, estimate_2026e, price,
            pe_2025e, pe_2026e, change_vs_prior_observation_2025e, change_vs_prior_observation_q4_2025e,
            change_vs_prior_observation_2026e, change_vs_prior_observation_price,
            change_vs_prior_observation_pe_2025e, change_vs_prior_observation_pe_2026e,
            source, source_path, public_files_discontinued
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _import_aggregate_eps_wayback_rows(*, dst_conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    frame = _read_parquet(path)
    rows = [
        (
            run_id,
            _sql_value(row["snapshot_date"]),
            _sql_value(row["timestamp"]),
            _sql_value(row["archive_url"]),
            _sql_value(row["workbook_as_of_date"]),
            _sql_value(row["forward_estimate_label"]),
            _sql_value(row["forward_estimate_value"]),
            _sql_value(row["estimate_2025e"]),
            _sql_value(row["estimate_q4_2025e"]),
            _sql_value(row["estimate_2026e"]),
            _sql_value(row["price"]),
            _sql_value(row["pe_2025e"]),
            _sql_value(row["pe_2026e"]),
            _sql_value(row["change_vs_prior_observation_2025e"]),
            _sql_value(row["change_vs_prior_observation_q4_2025e"]),
            _sql_value(row["change_vs_prior_observation_2026e"]),
            _sql_value(row["change_vs_prior_observation_price"]),
            _sql_value(row["change_vs_prior_observation_pe_2025e"]),
            _sql_value(row["change_vs_prior_observation_pe_2026e"]),
            int(bool(_sql_value(row["public_files_discontinued"]))),
            _sql_value(row["source"]),
        )
        for row in frame.to_dict(orient="records")
    ]
    dst_conn.executemany(
        """
        INSERT INTO aggregate_eps_wayback_rows (
            run_id, snapshot_date, timestamp, archive_url, workbook_as_of_date, forward_estimate_label,
            forward_estimate_value, estimate_2025e, estimate_q4_2025e, estimate_2026e, price, pe_2025e,
            pe_2026e, change_vs_prior_observation_2025e, change_vs_prior_observation_q4_2025e,
            change_vs_prior_observation_2026e, change_vs_prior_observation_price,
            change_vs_prior_observation_pe_2025e, change_vs_prior_observation_pe_2026e,
            public_files_discontinued, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _import_alpaca_market_rows(*, dst_conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    frame = _read_parquet(path)
    symbol = _infer_symbol_from_output_path(path)
    rows = [
        (
            run_id,
            symbol,
            _sql_value(row["date"]),
            _sql_value(row["open"]),
            _sql_value(row["high"]),
            _sql_value(row["low"]),
            _sql_value(row["close"]),
            int(_sql_value(row["volume"]) or 0),
            _sql_value(row["adjusted_close"]),
            str(path),
        )
        for row in frame.to_dict(orient="records")
    ]
    dst_conn.executemany(
        """
        INSERT INTO alpaca_market_rows (
            run_id, symbol, date, open, high, low, close, volume, adjusted_close, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing derived output parquet during consolidation: {path}")
    return pd.read_parquet(path)


def _read_yaml_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing derived output YAML during consolidation: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise RuntimeError(f"Unexpected event calendar YAML shape: {path}")
    return payload["events"]


def _infer_symbol_from_output_path(path: Path) -> str:
    parent = path.parent.name
    if not parent.startswith("symbol="):
        raise RuntimeError(f"Could not infer symbol from derived output path: {path}")
    return parent.split("=", 1)[1]


def _sql_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value
