from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.local_daily_ohlcv_sqlite import _ensure_daily_ohlcv_table


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsolidationSource:
    label: str
    db_path: Path


DEFAULT_CONSOLIDATION_SOURCES = [
    ConsolidationSource("events", Path("/private/tmp/regime_events_acquisition/acquisition.db")),
    ConsolidationSource("macro_full_history", Path("/private/tmp/regime_macro_full_history/acquisition.db")),
    ConsolidationSource("market_smoke", Path("/private/tmp/regime_market_sqlite_smoke/acquisition.db")),
    ConsolidationSource("local_ohlcv", Path("/private/tmp/regime_local_ohlcv_sqlite/acquisition.db")),
    ConsolidationSource("pmi_manual", Path("/private/tmp/regime_pmi_manual_sqlite/acquisition.db")),
    ConsolidationSource("pit", Path("/private/tmp/regime_pit_sqlite_smoke/acquisition.db")),
    ConsolidationSource("fomc", Path("/private/tmp/regime_fomc_sqlite_smoke/acquisition.db")),
    ConsolidationSource("powell", Path("/private/tmp/regime_powell_sqlite_smoke/acquisition.db")),
    ConsolidationSource("usd_index", Path("/private/tmp/regime_usd_index_sqlite_smoke/acquisition.db")),
    ConsolidationSource("eps_snapshot", Path("/private/tmp/regime_eps_sqlite_smoke/acquisition.db")),
    ConsolidationSource("eps_wayback", Path("/private/tmp/regime_eps_wayback_sqlite_smoke/acquisition.db")),
]


def consolidate_acquisition_dbs(
    *,
    target_db_path: Path,
    sources: list[ConsolidationSource] | None = None,
) -> dict[str, object]:
    selected_sources = sources or DEFAULT_CONSOLIDATION_SOURCES
    target_db_path.parent.mkdir(parents=True, exist_ok=True)
    if target_db_path.exists():
        target_db_path.unlink()

    AcquisitionStore(target_db_path)
    with sqlite3.connect(target_db_path) as conn:
        _ensure_daily_ohlcv_table(conn)
        _ensure_normalized_tables(conn)

    summary_sources: list[dict[str, object]] = []
    total_daily_ohlcv_rows = 0

    for source in selected_sources:
        if not source.db_path.exists():
            raise FileNotFoundError(f"Missing acquisition db for consolidation: {source.db_path}")
        counts = _import_one_source(target_db_path=target_db_path, source=source)
        total_daily_ohlcv_rows += counts.get("daily_ohlcv_rows", 0)
        summary_sources.append(
            {
                "label": source.label,
                "db_path": str(source.db_path),
                **counts,
            }
        )

    with sqlite3.connect(target_db_path) as conn:
        final_counts = {
            "fetch_runs": conn.execute("SELECT count(*) FROM fetch_runs").fetchone()[0],
            "artifacts": conn.execute("SELECT count(*) FROM artifacts").fetchone()[0],
            "artifact_blobs": conn.execute("SELECT count(*) FROM artifact_blobs").fetchone()[0],
            "derived_outputs": conn.execute("SELECT count(*) FROM derived_outputs").fetchone()[0],
            "daily_ohlcv_rows": conn.execute("SELECT count(*) FROM daily_ohlcv_rows").fetchone()[0],
            "event_calendar_rows": conn.execute("SELECT count(*) FROM event_calendar_rows").fetchone()[0],
            "macro_rows": conn.execute("SELECT count(*) FROM macro_rows").fetchone()[0],
            "pmi_rows": conn.execute("SELECT count(*) FROM pmi_rows").fetchone()[0],
            "pit_constituent_rows": conn.execute("SELECT count(*) FROM pit_constituent_rows").fetchone()[0],
            "fomc_minutes_rows": conn.execute("SELECT count(*) FROM fomc_minutes_rows").fetchone()[0],
            "powell_speeches_rows": conn.execute("SELECT count(*) FROM powell_speeches_rows").fetchone()[0],
            "usd_index_rows": conn.execute("SELECT count(*) FROM usd_index_rows").fetchone()[0],
            "aggregate_eps_snapshot_rows": conn.execute("SELECT count(*) FROM aggregate_eps_snapshot_rows").fetchone()[0],
            "aggregate_eps_wayback_rows": conn.execute("SELECT count(*) FROM aggregate_eps_wayback_rows").fetchone()[0],
            "alpaca_market_rows": conn.execute("SELECT count(*) FROM alpaca_market_rows").fetchone()[0],
        }

    report = {
        "target_db": str(target_db_path),
        "sources": summary_sources,
        "final_counts": final_counts,
    }
    report_path = target_db_path.parent / "consolidation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report


def _import_one_source(*, target_db_path: Path, source: ConsolidationSource) -> dict[str, int]:
    with sqlite3.connect(target_db_path) as dst_conn, sqlite3.connect(source.db_path) as src_conn:
        dst_conn.execute("PRAGMA foreign_keys = ON")
        src_conn.row_factory = sqlite3.Row
        fetch_run_id_map: dict[int, int] = {}
        artifact_id_map: dict[int, int] = {}
        normalized_counts = {
            "event_calendar_rows": 0,
            "macro_rows": 0,
            "pmi_rows": 0,
            "pit_constituent_rows": 0,
            "fomc_minutes_rows": 0,
            "powell_speeches_rows": 0,
            "usd_index_rows": 0,
            "aggregate_eps_snapshot_rows": 0,
            "aggregate_eps_wayback_rows": 0,
            "alpaca_market_rows": 0,
        }

        for row in src_conn.execute("SELECT * FROM fetch_runs ORDER BY run_id"):
            params_json = _augment_params_json(row["params_json"], source_label=source.label, source_db_path=str(source.db_path))
            notes = _merge_notes(row["notes"], f"imported_from={source.label}:{source.db_path}")
            cursor = dst_conn.execute(
                """
                INSERT INTO fetch_runs (
                    fetch_type,
                    started_at_utc,
                    finished_at_utc,
                    status,
                    params_json,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["fetch_type"],
                    row["started_at_utc"],
                    row["finished_at_utc"],
                    row["status"],
                    params_json,
                    notes,
                ),
            )
            fetch_run_id_map[int(row["run_id"])] = int(cursor.lastrowid)

        for row in src_conn.execute("SELECT * FROM artifacts ORDER BY artifact_id"):
            new_run_id = fetch_run_id_map[int(row["run_id"])]
            notes = _merge_notes(row["notes"], f"imported_from={source.label}:{source.db_path}")
            cursor = dst_conn.execute(
                """
                INSERT INTO artifacts (
                    run_id,
                    source_name,
                    artifact_kind,
                    source_identifier,
                    content_text,
                    content_sha256,
                    downloaded_at_utc,
                    effective_date,
                    start_date,
                    end_date,
                    timezone,
                    calendar_assumption,
                    adjustment_policy,
                    license_note,
                    notes,
                    local_path,
                    content_size_bytes,
                    content_encoding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_run_id,
                    row["source_name"],
                    row["artifact_kind"],
                    row["source_identifier"],
                    row["content_text"],
                    row["content_sha256"],
                    row["downloaded_at_utc"],
                    row["effective_date"],
                    row["start_date"],
                    row["end_date"],
                    row["timezone"],
                    row["calendar_assumption"],
                    row["adjustment_policy"],
                    row["license_note"],
                    notes,
                    _row_value(row, "local_path"),
                    _row_value(row, "content_size_bytes"),
                    _row_value(row, "content_encoding"),
                ),
            )
            artifact_id_map[int(row["artifact_id"])] = int(cursor.lastrowid)

        if _table_exists(src_conn, "artifact_blobs"):
            for row in src_conn.execute("SELECT * FROM artifact_blobs ORDER BY artifact_id"):
                old_artifact_id = int(row["artifact_id"])
                if old_artifact_id not in artifact_id_map:
                    continue
                dst_conn.execute(
                    """
                    INSERT INTO artifact_blobs (
                        artifact_id,
                        content_bytes
                    ) VALUES (?, ?)
                    """,
                    (artifact_id_map[old_artifact_id], row["content_bytes"]),
                )

        for row in src_conn.execute("SELECT * FROM derived_outputs ORDER BY output_id"):
            new_run_id = fetch_run_id_map[int(row["run_id"])]
            notes = _merge_notes(row["notes"], f"imported_from={source.label}:{source.db_path}")
            dst_conn.execute(
                """
                INSERT INTO derived_outputs (
                    run_id,
                    output_kind,
                    path,
                    content_sha256,
                    row_count,
                    min_date,
                    max_date,
                    recorded_at_utc,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_run_id,
                    row["output_kind"],
                    row["path"],
                    row["content_sha256"],
                    row["row_count"],
                    row["min_date"],
                    row["max_date"],
                    row["recorded_at_utc"],
                    notes,
                ),
            )
            imported = _import_normalized_output(
                dst_conn=dst_conn,
                run_id=new_run_id,
                output_kind=row["output_kind"],
                path=Path(row["path"]),
            )
            if imported is not None:
                normalized_counts[imported] += int(row["row_count"] or 0)

        imported_daily_ohlcv_rows = 0
        if _table_exists(src_conn, "daily_ohlcv_rows"):
            rows = src_conn.execute("SELECT * FROM daily_ohlcv_rows ORDER BY symbol, date").fetchall()
            if rows:
                dst_conn.executemany(
                    """
                    INSERT OR REPLACE INTO daily_ohlcv_rows (
                        symbol,
                        date,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        adjusted_close,
                        source_file
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            row["symbol"],
                            row["date"],
                            row["open"],
                            row["high"],
                            row["low"],
                            row["close"],
                            row["volume"],
                            row["adjusted_close"],
                            row["source_file"],
                        )
                        for row in rows
                    ],
                )
                imported_daily_ohlcv_rows = len(rows)

        dst_conn.commit()
        return {
            "fetch_runs": len(fetch_run_id_map),
            "artifacts": len(artifact_id_map),
            "artifact_blobs": _count_rows(src_conn, "artifact_blobs"),
            "derived_outputs": _count_rows(src_conn, "derived_outputs"),
            "daily_ohlcv_rows": imported_daily_ohlcv_rows,
            **normalized_counts,
        }


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
        return "event_calendar_rows"
    if output_kind == "fred_macro_parquet":
        _import_macro_rows(dst_conn=dst_conn, run_id=run_id, path=path, dataset_kind="series")
        return "macro_rows"
    if output_kind == "fred_cpi_vintages_parquet":
        _import_macro_rows(dst_conn=dst_conn, run_id=run_id, path=path, dataset_kind="cpi_vintages")
        return "macro_rows"
    if output_kind == "pmi_parquet":
        _import_pmi_rows(dst_conn=dst_conn, run_id=run_id, path=path, dataset_kind="latest")
        return "pmi_rows"
    if output_kind == "pmi_history_parquet":
        _import_pmi_rows(dst_conn=dst_conn, run_id=run_id, path=path, dataset_kind="history")
        return "pmi_rows"
    if output_kind == "pit_constituents_parquet":
        _import_pit_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return "pit_constituent_rows"
    if output_kind == "fomc_minutes_parquet":
        _import_fomc_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return "fomc_minutes_rows"
    if output_kind == "powell_speeches_parquet":
        _import_powell_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return "powell_speeches_rows"
    if output_kind == "usd_index_parquet":
        _import_usd_index_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return "usd_index_rows"
    if output_kind == "aggregate_eps_parquet":
        _import_aggregate_eps_snapshot_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return "aggregate_eps_snapshot_rows"
    if output_kind == "aggregate_eps_wayback_timeline":
        _import_aggregate_eps_wayback_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return "aggregate_eps_wayback_rows"
    if output_kind == "alpaca_daily_ohlcv_parquet":
        _import_alpaca_market_rows(dst_conn=dst_conn, run_id=run_id, path=path)
        return "alpaca_market_rows"
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


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _count_rows(conn: sqlite3.Connection, table_name: str) -> int:
    if not _table_exists(conn, table_name):
        return 0
    return int(conn.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0])


def _merge_notes(existing: str | None, extra: str) -> str:
    if existing:
        return f"{existing} | {extra}"
    return extra


def _augment_params_json(params_json: str, *, source_label: str, source_db_path: str) -> str:
    try:
        payload = json.loads(params_json)
    except json.JSONDecodeError:
        LOG.warning(
            "params_json unparseable in _augment_params_json, using raw fallback (first 200 chars): %s",
            params_json[:200],
        )
        payload = {"raw_params_json": params_json}
    payload["consolidated_from_label"] = source_label
    payload["consolidated_from_db"] = source_db_path
    return json.dumps(payload, sort_keys=True)


def _row_value(row: sqlite3.Row, key: str) -> object | None:
    if key in row.keys():
        return row[key]
    return None


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
