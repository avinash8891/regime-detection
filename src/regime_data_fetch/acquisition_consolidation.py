from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from regime_data_fetch.acquisition_consolidation_normalized import (
    AGGREGATE_EPS_SNAPSHOT_ROWS_TABLE,
    AGGREGATE_EPS_WAYBACK_ROWS_TABLE,
    ALPACA_MARKET_ROWS_TABLE,
    EVENT_CALENDAR_ROWS_TABLE,
    FOMC_MINUTES_ROWS_TABLE,
    MACRO_ROWS_TABLE,
    PIT_CONSTITUENT_ROWS_TABLE,
    PMI_ROWS_TABLE,
    POWELL_SPEECHES_ROWS_TABLE,
    USD_INDEX_ROWS_TABLE,
    _NORMALIZED_TABLES,
    _ensure_normalized_tables,
    _import_normalized_output,
)
from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.local_daily_ohlcv_sqlite import _ensure_daily_ohlcv_table

LOGGER = logging.getLogger(__name__)
FETCH_RUNS_TABLE = "fetch_runs"
ARTIFACTS_TABLE = "artifacts"
ARTIFACT_BLOBS_TABLE = "artifact_blobs"
DERIVED_OUTPUTS_TABLE = "derived_outputs"
DAILY_OHLCV_ROWS_TABLE = "daily_ohlcv_rows"

_COUNTABLE_TABLES = frozenset(
    {
        FETCH_RUNS_TABLE,
        ARTIFACTS_TABLE,
        ARTIFACT_BLOBS_TABLE,
        DERIVED_OUTPUTS_TABLE,
        DAILY_OHLCV_ROWS_TABLE,
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
    }
)

@dataclass(frozen=True)
class ConsolidationSource:
    label: str
    db_path: Path


def consolidate_acquisition_dbs(
    *,
    target_db_path: Path,
    sources: list[ConsolidationSource] | None = None,
) -> dict[str, object]:
    if sources is None:
        raise ValueError("consolidate_acquisition_dbs requires explicit sources")
    selected_sources = sources
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
        total_daily_ohlcv_rows += counts.get(DAILY_OHLCV_ROWS_TABLE, 0)
        summary_sources.append(
            {
                "label": source.label,
                "db_path": str(source.db_path),
                **counts,
            }
        )

    with sqlite3.connect(target_db_path) as conn:
        final_counts = {
            FETCH_RUNS_TABLE: _count_rows(conn, FETCH_RUNS_TABLE),
            ARTIFACTS_TABLE: _count_rows(conn, ARTIFACTS_TABLE),
            ARTIFACT_BLOBS_TABLE: _count_rows(conn, ARTIFACT_BLOBS_TABLE),
            DERIVED_OUTPUTS_TABLE: _count_rows(conn, DERIVED_OUTPUTS_TABLE),
            DAILY_OHLCV_ROWS_TABLE: _count_rows(conn, DAILY_OHLCV_ROWS_TABLE),
            EVENT_CALENDAR_ROWS_TABLE: _count_rows(conn, EVENT_CALENDAR_ROWS_TABLE),
            MACRO_ROWS_TABLE: _count_rows(conn, MACRO_ROWS_TABLE),
            PMI_ROWS_TABLE: _count_rows(conn, PMI_ROWS_TABLE),
            PIT_CONSTITUENT_ROWS_TABLE: _count_rows(conn, PIT_CONSTITUENT_ROWS_TABLE),
            FOMC_MINUTES_ROWS_TABLE: _count_rows(conn, FOMC_MINUTES_ROWS_TABLE),
            POWELL_SPEECHES_ROWS_TABLE: _count_rows(conn, POWELL_SPEECHES_ROWS_TABLE),
            USD_INDEX_ROWS_TABLE: _count_rows(conn, USD_INDEX_ROWS_TABLE),
            AGGREGATE_EPS_SNAPSHOT_ROWS_TABLE: _count_rows(conn, AGGREGATE_EPS_SNAPSHOT_ROWS_TABLE),
            AGGREGATE_EPS_WAYBACK_ROWS_TABLE: _count_rows(conn, AGGREGATE_EPS_WAYBACK_ROWS_TABLE),
            ALPACA_MARKET_ROWS_TABLE: _count_rows(conn, ALPACA_MARKET_ROWS_TABLE),
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
        normalized_counts = dict.fromkeys(_NORMALIZED_TABLES, 0)

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

        if _table_exists(src_conn, ARTIFACT_BLOBS_TABLE):
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
        if _table_exists(src_conn, DAILY_OHLCV_ROWS_TABLE):
            rows = src_conn.execute(
                f"SELECT * FROM {DAILY_OHLCV_ROWS_TABLE} ORDER BY symbol, date"
            ).fetchall()
            if rows:
                dst_conn.executemany(
                    f"""
                    INSERT OR REPLACE INTO {DAILY_OHLCV_ROWS_TABLE} (
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
            FETCH_RUNS_TABLE: len(fetch_run_id_map),
            ARTIFACTS_TABLE: len(artifact_id_map),
            ARTIFACT_BLOBS_TABLE: _count_rows(src_conn, ARTIFACT_BLOBS_TABLE),
            DERIVED_OUTPUTS_TABLE: _count_rows(src_conn, DERIVED_OUTPUTS_TABLE),
            DAILY_OHLCV_ROWS_TABLE: imported_daily_ohlcv_rows,
            **normalized_counts,
        }


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _count_rows(conn: sqlite3.Connection, table_name: str) -> int:
    if table_name not in _COUNTABLE_TABLES:
        raise ValueError(f"Unexpected SQLite table for count: {table_name!r}")
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
        LOGGER.warning(
            "params_json unparseable; using raw fallback source_label=%s source_db_path=%s",
            source_label,
            source_db_path,
            exc_info=True,
        )
        payload = {"raw_params_json": params_json}
    payload["consolidated_from_label"] = source_label
    payload["consolidated_from_db"] = source_db_path
    return json.dumps(payload, sort_keys=True)


def _row_value(row: sqlite3.Row, key: str) -> object | None:
    if key in row.keys():
        return row[key]
    return None

