from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore


EXPECTED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "adjusted_close"]


def run_local_daily_ohlcv_sqlite_import(
    *,
    out_dir: Path,
    source_dir: Path,
    acquisition_db_path: Path,
    artifact_store_root: str | Path | None = None,
) -> Path:
    if not source_dir.exists():
        raise SystemExit(f"Missing OHLCV source directory: {source_dir}")

    parquet_files = sorted(source_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files found under: {source_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    store = AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
    fetch_run = store.start_fetch_run(
        fetch_type="daily_ohlcv_local_sqlite",
        params={
            "source_dir": str(source_dir),
            "parquet_files": len(parquet_files),
        },
    )

    imported_rows = 0
    symbol_count = 0
    min_date: str | None = None
    max_date: str | None = None
    artifact_records: list[tuple[Path, str, str, str]] = []

    try:
        with sqlite3.connect(acquisition_db_path) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            _ensure_daily_ohlcv_table(conn)

            for parquet_path in parquet_files:
                symbol = _infer_symbol_from_path(parquet_path)
                frame = pd.read_parquet(parquet_path)
                _validate_ohlcv_frame(frame=frame, parquet_path=parquet_path)
                normalized = frame.copy()
                normalized["date"] = pd.to_datetime(normalized["date"]).dt.date.astype(str)
                rows = [
                    (
                        symbol,
                        row["date"],
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        int(row["volume"]),
                        float(row["adjusted_close"]),
                        str(parquet_path),
                    )
                    for row in normalized.to_dict(orient="records")
                ]
                conn.executemany(
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
                    rows,
                )
                imported_rows += len(rows)
                symbol_count += 1
                file_min = normalized["date"].min()
                file_max = normalized["date"].max()
                min_date = file_min if min_date is None else min(min_date, file_min)
                max_date = file_max if max_date is None else max(max_date, file_max)

                artifact_records.append((parquet_path, file_min, file_max, symbol))
            conn.commit()

        for parquet_path, file_min, file_max, symbol in artifact_records:
            store.record_file_artifact(
                run_id=fetch_run.run_id,
                source_name="local:daily_ohlcv",
                artifact_kind="parquet_local",
                source_identifier=str(parquet_path),
                file_path=parquet_path,
                start_date=file_min,
                end_date=file_max,
                timezone="UTC",
                adjustment_policy="raw_or_precomputed_source",
                license_note="Local partitioned OHLCV parquet artifact imported into SQLite row store",
                notes=f"symbol={symbol}",
                store_bytes=False,
            )

        report = {
            "source_dir": str(source_dir),
            "counts": {
                "parquet_files": len(parquet_files),
                "symbols": symbol_count,
                "imported_rows": imported_rows,
            },
            "date_range": {
                "min_date": min_date,
                "max_date": max_date,
            },
            "paths": {
                "acquisition_db": str(acquisition_db_path),
                "profile_constituent_tree": {
                    "path": str(source_dir),
                    "local_path": "data/raw/daily_ohlcv_762",
                },
            },
        }
        report_path = out_dir / "daily_ohlcv_local_sqlite_import_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        store.record_output(
            run_id=fetch_run.run_id,
            output_kind="daily_ohlcv_local_sqlite_import_report",
            path=report_path,
            row_count=imported_rows,
            min_date=min_date,
            max_date=max_date,
            notes="Local OHLCV parquet import report",
        )
        store.finish_fetch_run(
            run_id=fetch_run.run_id,
            status="ok",
            notes=f"symbols={symbol_count};rows={imported_rows}",
        )
        return report_path
    except Exception as exc:
        store.finish_fetch_run(run_id=fetch_run.run_id, status="failed", notes=str(exc))
        raise


def _ensure_daily_ohlcv_table(conn: sqlite3.Connection) -> None:
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


def _infer_symbol_from_path(parquet_path: Path) -> str:
    parent = parquet_path.parent.name
    if not parent.startswith("symbol="):
        raise RuntimeError(f"Could not infer symbol from parquet path: {parquet_path}")
    return parent.split("=", 1)[1]


def _validate_ohlcv_frame(*, frame: pd.DataFrame, parquet_path: Path) -> None:
    got = list(frame.columns)
    if got != EXPECTED_COLUMNS:
        raise RuntimeError(f"Unexpected OHLCV parquet columns in {parquet_path}: {got!r}")
