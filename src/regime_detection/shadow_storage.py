from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from regime_detection.loaders import load_event_calendar


RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY,
    run_timestamp TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    config_version TEXT NOT NULL,
    status TEXT NOT NULL,
    failure_reason TEXT,
    input_archive_path TEXT NOT NULL,
    output_path TEXT,
    output_sha256 TEXT,
    UNIQUE (as_of_date, engine_version, config_version)
)
"""

REPLAY_CHECKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS replay_checks (
    check_id INTEGER PRIMARY KEY,
    check_timestamp TEXT NOT NULL,
    original_run_id INTEGER REFERENCES runs(run_id),
    matches BOOLEAN NOT NULL,
    diff TEXT
)
"""

INCIDENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id INTEGER PRIMARY KEY,
    incident_date TEXT NOT NULL,
    description TEXT NOT NULL,
    resolution TEXT,
    breaks_qualification BOOLEAN NOT NULL
)
"""


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_shadow_layout(output_root: Path) -> dict[str, Path]:
    paths = {
        "db": output_root / "regime_shadow.db",
        "outputs": output_root / "outputs",
        "input_archives": output_root / "input_archives",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    paths["outputs"].mkdir(parents=True, exist_ok=True)
    paths["input_archives"].mkdir(parents=True, exist_ok=True)
    return paths


def open_shadow_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(RUNS_SCHEMA)
    conn.execute(REPLAY_CHECKS_SCHEMA)
    conn.execute(INCIDENTS_SCHEMA)
    conn.commit()
    return conn


def event_rows_for_yaml(event_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if event_df is None or event_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row in event_df.to_dict(orient="records"):
        out: dict[str, Any] = {}
        for key, value in row.items():
            out[key] = event_value_for_yaml(value)
        rows.append(out)
    return rows


def event_value_for_yaml(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: event_value_for_yaml(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [event_value_for_yaml(item) for item in value]
    if value is None:
        return None
    if bool(pd.isna(value)):
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).date().isoformat()
    return value


def write_archived_inputs(
    *,
    archive_dir: Path,
    market_slice: pd.DataFrame,
    event_df: pd.DataFrame | None,
) -> tuple[Path, Path, Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    market_path = archive_dir / "market_data.parquet"
    events_path = archive_dir / "events.yaml"
    checksums_path = archive_dir / "checksums.json"

    market_slice.to_parquet(market_path, index=False)
    events_path.write_text(
        yaml.safe_dump({"events": event_rows_for_yaml(event_df)}, sort_keys=False),
        encoding="utf-8",
    )
    checksums_path.write_text(
        json.dumps(
            {
                "market_data.parquet": sha256_file(market_path),
                "events.yaml": sha256_file(events_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return market_path, events_path, checksums_path


def insert_run_row(
    *,
    conn: sqlite3.Connection,
    run_timestamp: str,
    as_of_date: date,
    engine_version: str,
    config_version: str,
    archive_dir: Path,
) -> None:
    conn.execute(
        """
        INSERT INTO runs (
            run_timestamp, as_of_date, engine_version, config_version,
            status, failure_reason, input_archive_path, output_path, output_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_timestamp,
            as_of_date.isoformat(),
            engine_version,
            config_version,
            "in_progress",
            None,
            str(archive_dir),
            None,
            None,
        ),
    )
    conn.commit()


def update_run_row_success(
    *,
    conn: sqlite3.Connection,
    as_of_date: date,
    engine_version: str,
    config_version: str,
    output_path: Path,
) -> None:
    conn.execute(
        """
        UPDATE runs
        SET status = ?, output_path = ?, output_sha256 = ?, failure_reason = NULL
        WHERE as_of_date = ? AND engine_version = ? AND config_version = ?
        """,
        (
            "success",
            str(output_path),
            sha256_file(output_path),
            as_of_date.isoformat(),
            engine_version,
            config_version,
        ),
    )
    conn.commit()


def update_run_row_failure(
    *,
    conn: sqlite3.Connection,
    as_of_date: date,
    engine_version: str,
    config_version: str,
    failure_reason: str,
) -> None:
    conn.execute(
        """
        UPDATE runs
        SET status = ?, failure_reason = ?
        WHERE as_of_date = ? AND engine_version = ? AND config_version = ?
        """,
        (
            "failure",
            failure_reason,
            as_of_date.isoformat(),
            engine_version,
            config_version,
        ),
    )
    conn.commit()


def load_archived_market_data(path: Path) -> pd.DataFrame:
    archived = pd.read_parquet(path)
    archived["date"] = pd.to_datetime(archived["date"]).dt.date
    return archived


def load_archived_event_calendar(path: Path) -> pd.DataFrame | None:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not loaded:
        return None
    events = loaded.get("events", []) if isinstance(loaded, dict) else loaded
    if not events:
        return None
    return load_event_calendar(path)


def fetch_run_row(
    *,
    conn: sqlite3.Connection,
    as_of_date: date,
) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT run_id, as_of_date, status, engine_version, config_version, input_archive_path, output_path
        FROM runs
        WHERE as_of_date = ?
        """,
        (as_of_date.isoformat(),),
    ).fetchone()
    conn.row_factory = None
    return row


def insert_replay_check(
    *,
    conn: sqlite3.Connection,
    check_timestamp: str,
    original_run_id: int,
    matches: bool,
    diff: dict[str, Any] | None,
) -> None:
    conn.execute(
        """
        INSERT INTO replay_checks (check_timestamp, original_run_id, matches, diff)
        VALUES (?, ?, ?, ?)
        """,
        (
            check_timestamp,
            original_run_id,
            int(matches),
            None if diff is None else json.dumps(diff, sort_keys=True),
        ),
    )
    conn.commit()


def insert_incident(
    *,
    conn: sqlite3.Connection,
    incident_date: date,
    description: str,
    resolution: str | None,
    breaks_qualification: bool,
) -> None:
    conn.execute(
        """
        INSERT INTO incidents (incident_date, description, resolution, breaks_qualification)
        VALUES (?, ?, ?, ?)
        """,
        (
            incident_date.isoformat(),
            description,
            resolution,
            int(breaks_qualification),
        ),
    )
    conn.commit()
