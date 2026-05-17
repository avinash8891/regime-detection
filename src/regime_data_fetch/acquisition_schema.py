from __future__ import annotations

import sqlite3


ACQUISITION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fetch_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetch_type TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    status TEXT NOT NULL,
    params_json TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
    source_name TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    content_text TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    downloaded_at_utc TEXT NOT NULL,
    effective_date TEXT,
    start_date TEXT,
    end_date TEXT,
    timezone TEXT,
    calendar_assumption TEXT,
    adjustment_policy TEXT,
    license_note TEXT,
    notes TEXT,
    local_path TEXT,
    content_size_bytes INTEGER,
    content_encoding TEXT
);

CREATE TABLE IF NOT EXISTS artifact_blobs (
    artifact_id INTEGER PRIMARY KEY REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    content_bytes BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS derived_outputs (
    output_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
    output_kind TEXT NOT NULL,
    path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    row_count INTEGER,
    min_date TEXT,
    max_date TEXT,
    recorded_at_utc TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS artifact_records (
    artifact_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    stage TEXT NOT NULL,
    uri TEXT NOT NULL,
    local_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    source_name TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    row_count INTEGER,
    min_date TEXT,
    max_date TEXT,
    schema_version TEXT,
    recorded_at_utc TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS artifact_lineage (
    lineage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    output_artifact_record_id INTEGER NOT NULL REFERENCES artifact_records(artifact_record_id) ON DELETE CASCADE,
    input_artifact_record_id INTEGER NOT NULL REFERENCES artifact_records(artifact_record_id) ON DELETE CASCADE,
    transform_name TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS canonical_versions (
    canonical_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL,
    version TEXT NOT NULL,
    artifact_record_id INTEGER NOT NULL REFERENCES artifact_records(artifact_record_id) ON DELETE CASCADE,
    manifest_uri TEXT,
    status TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    UNIQUE(dataset_name, version)
);

CREATE TABLE IF NOT EXISTS source_checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    cursor_key TEXT NOT NULL,
    cursor_value TEXT NOT NULL,
    successful_run_id INTEGER NOT NULL REFERENCES fetch_runs(run_id) ON DELETE CASCADE,
    updated_at_utc TEXT NOT NULL,
    UNIQUE(source_name, cursor_key)
);
"""

ARTIFACTS_COMPAT_COLUMNS = {
    "local_path": "ALTER TABLE artifacts ADD COLUMN local_path TEXT",
    "content_size_bytes": "ALTER TABLE artifacts ADD COLUMN content_size_bytes INTEGER",
    "content_encoding": "ALTER TABLE artifacts ADD COLUMN content_encoding TEXT",
}


def init_acquisition_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(ACQUISITION_SCHEMA_SQL)
    ensure_artifacts_compat_columns(conn)


def ensure_artifacts_compat_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(artifacts)")}
    for column_name, ddl in ARTIFACTS_COMPAT_COLUMNS.items():
        if column_name not in existing:
            conn.execute(ddl)
