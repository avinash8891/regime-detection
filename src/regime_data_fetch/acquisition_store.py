from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass(frozen=True)
class RecordedArtifact:
    artifact_id: int
    content_sha256: str


@dataclass(frozen=True)
class FetchRun:
    run_id: int
    started_at_utc: str


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_record_id: int
    content_sha256: str


class AcquisitionStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def start_fetch_run(self, *, fetch_type: str, params: dict[str, object]) -> FetchRun:
        started_at_utc = utc_now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO fetch_runs (
                    fetch_type,
                    started_at_utc,
                    status,
                    params_json
                ) VALUES (?, ?, ?, ?)
                """,
                (fetch_type, started_at_utc, "running", json.dumps(params, sort_keys=True)),
            )
            return FetchRun(run_id=int(cursor.lastrowid), started_at_utc=started_at_utc)

    def finish_fetch_run(
        self,
        *,
        run_id: int,
        status: str,
        notes: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE fetch_runs
                SET finished_at_utc = ?, status = ?, notes = ?
                WHERE run_id = ?
                """,
                (utc_now_iso(), status, notes, run_id),
            )

    def record_text_artifact(
        self,
        *,
        run_id: int,
        source_name: str,
        artifact_kind: str,
        source_identifier: str,
        content_text: str,
        effective_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        timezone: str | None = None,
        calendar_assumption: str | None = None,
        adjustment_policy: str | None = None,
        license_note: str | None = None,
        notes: str | None = None,
    ) -> RecordedArtifact:
        sha256 = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            cursor = conn.execute(
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
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    source_name,
                    artifact_kind,
                    source_identifier,
                    content_text,
                    sha256,
                    utc_now_iso(),
                    effective_date,
                    start_date,
                    end_date,
                    timezone,
                    calendar_assumption,
                    adjustment_policy,
                    license_note,
                    notes,
                ),
            )
            return RecordedArtifact(artifact_id=int(cursor.lastrowid), content_sha256=sha256)

    def record_file_artifact(
        self,
        *,
        run_id: int,
        source_name: str,
        artifact_kind: str,
        source_identifier: str,
        file_path: Path,
        effective_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        timezone: str | None = None,
        calendar_assumption: str | None = None,
        adjustment_policy: str | None = None,
        license_note: str | None = None,
        notes: str | None = None,
        store_bytes: bool = True,
    ) -> RecordedArtifact:
        payload = file_path.read_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
        with self._connect() as conn:
            cursor = conn.execute(
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
                    run_id,
                    source_name,
                    artifact_kind,
                    source_identifier,
                    "",
                    sha256,
                    utc_now_iso(),
                    effective_date,
                    start_date,
                    end_date,
                    timezone,
                    calendar_assumption,
                    adjustment_policy,
                    license_note,
                    notes,
                    str(file_path),
                    len(payload),
                    "binary",
                ),
            )
            artifact_id = int(cursor.lastrowid)
            if store_bytes:
                conn.execute(
                    """
                    INSERT INTO artifact_blobs (
                        artifact_id,
                        content_bytes
                    ) VALUES (?, ?)
                    """,
                    (artifact_id, payload),
                )
            return RecordedArtifact(artifact_id=artifact_id, content_sha256=sha256)

    def record_output(
        self,
        *,
        run_id: int,
        output_kind: str,
        path: Path,
        row_count: int | None = None,
        min_date: str | None = None,
        max_date: str | None = None,
        notes: str | None = None,
    ) -> None:
        content_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with self._connect() as conn:
            conn.execute(
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
                    run_id,
                    output_kind,
                    str(path),
                    content_sha256,
                    row_count,
                    min_date,
                    max_date,
                    utc_now_iso(),
                    notes,
                ),
            )

    def record_artifact_record(
        self,
        *,
        run_id: int,
        name: str,
        stage: str,
        uri: str,
        local_path: str,
        content_sha256: str,
        size_bytes: int,
        source_name: str,
        artifact_kind: str,
        row_count: int | None = None,
        min_date: str | None = None,
        max_date: str | None = None,
        schema_version: str | None = None,
        notes: str | None = None,
    ) -> ArtifactRecord:
        if stage not in {"raw_capture", "normalized", "canonical", "run_inputs"}:
            raise ValueError(f"unknown artifact stage: {stage}")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO artifact_records (
                    run_id,
                    name,
                    stage,
                    uri,
                    local_path,
                    content_sha256,
                    size_bytes,
                    source_name,
                    artifact_kind,
                    row_count,
                    min_date,
                    max_date,
                    schema_version,
                    recorded_at_utc,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    name,
                    stage,
                    uri,
                    local_path,
                    content_sha256,
                    size_bytes,
                    source_name,
                    artifact_kind,
                    row_count,
                    min_date,
                    max_date,
                    schema_version,
                    utc_now_iso(),
                    notes,
                ),
            )
            return ArtifactRecord(
                artifact_record_id=int(cursor.lastrowid),
                content_sha256=content_sha256,
            )

    def record_artifact_lineage(
        self,
        *,
        output_artifact_record_id: int,
        input_artifact_record_id: int,
        transform_name: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifact_lineage (
                    output_artifact_record_id,
                    input_artifact_record_id,
                    transform_name,
                    recorded_at_utc
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    output_artifact_record_id,
                    input_artifact_record_id,
                    transform_name,
                    utc_now_iso(),
                ),
            )

    def record_canonical_version(
        self,
        *,
        dataset_name: str,
        version: str,
        artifact_record_id: int,
        manifest_uri: str | None,
        status: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO canonical_versions (
                    dataset_name,
                    version,
                    artifact_record_id,
                    manifest_uri,
                    status,
                    recorded_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_name, version) DO UPDATE SET
                    artifact_record_id = excluded.artifact_record_id,
                    manifest_uri = excluded.manifest_uri,
                    status = excluded.status,
                    recorded_at_utc = excluded.recorded_at_utc
                """,
                (
                    dataset_name,
                    version,
                    artifact_record_id,
                    manifest_uri,
                    status,
                    utc_now_iso(),
                ),
            )

    def set_source_checkpoint(
        self,
        *,
        source_name: str,
        cursor_key: str,
        cursor_value: str,
        successful_run_id: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO source_checkpoints (
                    source_name,
                    cursor_key,
                    cursor_value,
                    successful_run_id,
                    updated_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_name, cursor_key) DO UPDATE SET
                    cursor_value = excluded.cursor_value,
                    successful_run_id = excluded.successful_run_id,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    source_name,
                    cursor_key,
                    cursor_value,
                    successful_run_id,
                    utc_now_iso(),
                ),
            )

    def get_source_checkpoint(self, *, source_name: str, cursor_key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT cursor_value
                FROM source_checkpoints
                WHERE source_name = ? AND cursor_key = ?
                """,
                (source_name, cursor_key),
            ).fetchone()
        return str(row[0]) if row else None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
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
            )
            self._ensure_artifact_columns(conn)

    def _ensure_artifact_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(artifacts)")
        }
        required_columns = {
            "local_path": "ALTER TABLE artifacts ADD COLUMN local_path TEXT",
            "content_size_bytes": "ALTER TABLE artifacts ADD COLUMN content_size_bytes INTEGER",
            "content_encoding": "ALTER TABLE artifacts ADD COLUMN content_encoding TEXT",
        }
        for column_name, ddl in required_columns.items():
            if column_name not in existing:
                conn.execute(ddl)
