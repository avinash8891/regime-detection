from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from regime_data_fetch.artifact_store import (
    ArtifactStore,
    StoredArtifact,
    build_artifact_store,
    sha256_bytes,
)
from regime_data_fetch.acquisition_schema import init_acquisition_schema


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass(frozen=True)
class RecordedArtifact:
    artifact_id: int
    content_sha256: str
    artifact_record_id: int | None = None


@dataclass(frozen=True)
class FetchRun:
    run_id: int
    started_at_utc: str


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_record_id: int
    content_sha256: str


class AcquisitionStore:
    def __init__(
        self,
        db_path: Path,
        *,
        artifact_store: ArtifactStore | None = None,
        artifact_store_root: str | Path | None = None,
    ) -> None:
        if artifact_store is not None and artifact_store_root is not None:
            raise ValueError(
                "pass either artifact_store or artifact_store_root, not both"
            )
        self.db_path = db_path
        self.artifact_store = artifact_store or (
            build_artifact_store(artifact_store_root)
            if artifact_store_root is not None
            else None
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            init_acquisition_schema(conn)

    def start_fetch_run(
        self, *, fetch_type: str, params: dict[str, object]
    ) -> FetchRun:
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
                (
                    fetch_type,
                    started_at_utc,
                    "running",
                    json.dumps(params, sort_keys=True),
                ),
            )
            return FetchRun(
                run_id=_last_insert_rowid(cursor),
                started_at_utc=started_at_utc,
            )

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
        payload = content_text.encode("utf-8")
        sha256 = sha256_bytes(payload)
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
            artifact_id = _last_insert_rowid(cursor)
        artifact_record = self._store_raw_artifact(
            run_id=run_id,
            source_name=source_name,
            artifact_kind=artifact_kind,
            source_identifier=source_identifier,
            payload=payload,
            content_sha256=sha256,
            effective_date=effective_date,
            start_date=start_date,
            end_date=end_date,
            notes=notes,
        )
        return RecordedArtifact(
            artifact_id=artifact_id,
            content_sha256=sha256,
            artifact_record_id=artifact_record.artifact_record_id
            if artifact_record
            else None,
        )

    # TODO(simplify, owner=regime-maintainers): record_file_artifact + record_output read the full file
    # into RAM (`path.read_bytes()`) just to sha256 it and feed _store_raw_artifact.
    # For the 762-parquet daily_ohlcv_762 import (store_bytes=False), each
    # parquet can be 10s of MB — peak RSS scales with file size for no reason.
    # Switch to streaming sha256_file + accept an already-hashed payload path
    # in _store_raw_artifact; only read_bytes when artifact_store + store_bytes
    # actually need the blob in memory.
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
        sha256 = sha256_bytes(payload)
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
            artifact_id = _last_insert_rowid(cursor)
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
        artifact_record = self._store_raw_artifact(
            run_id=run_id,
            source_name=source_name,
            artifact_kind=artifact_kind,
            source_identifier=source_identifier,
            payload=payload,
            content_sha256=sha256,
            effective_date=effective_date,
            start_date=start_date,
            end_date=end_date,
            local_path=str(file_path),
            notes=notes,
        )
        return RecordedArtifact(
            artifact_id=artifact_id,
            content_sha256=sha256,
            artifact_record_id=artifact_record.artifact_record_id
            if artifact_record
            else None,
        )

    def record_output(
        self,
        *,
        run_id: int,
        output_kind: str,
        path: Path,
        row_count: int | None = None,
        min_date: str | None = None,
        max_date: str | None = None,
        artifact_name: str | None = None,
        source_name: str = "derived_output",
        artifact_kind: str | None = None,
        record_artifact: bool = True,
        notes: str | None = None,
    ) -> ArtifactRecord | None:
        payload = path.read_bytes()
        content_sha256 = sha256_bytes(payload)
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
        if not record_artifact:
            return None
        key = str(
            Path("canonical")
            / _safe_path_part(output_kind)
            / f"run_id={run_id}"
            / path.name
        )
        stored = (
            self._put_file_artifact(
                path,
                key,
                context=f"canonical artifact {output_kind} for run_id={run_id}",
            )
            if self.artifact_store is not None
            else None
        )
        return self.record_artifact_record(
            run_id=run_id,
            name=artifact_name or output_kind,
            stage="canonical",
            uri=stored.uri if stored else key,
            local_path=str(path),
            content_sha256=stored.sha256 if stored else content_sha256,
            size_bytes=stored.size_bytes if stored else len(payload),
            source_name=source_name,
            artifact_kind=artifact_kind or output_kind,
            row_count=row_count,
            min_date=min_date,
            max_date=max_date,
            notes=notes,
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
                artifact_record_id=_last_insert_rowid(cursor),
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

    def _store_raw_artifact(
        self,
        *,
        run_id: int,
        source_name: str,
        artifact_kind: str,
        source_identifier: str,
        payload: bytes,
        content_sha256: str,
        effective_date: str | None,
        start_date: str | None,
        end_date: str | None,
        local_path: str | None = None,
        notes: str | None = None,
    ) -> ArtifactRecord | None:
        suffix = _artifact_suffix(artifact_kind=artifact_kind, local_path=local_path)
        key = str(
            Path("raw_capture")
            / _safe_path_part(source_name)
            / f"run_id={run_id}"
            / f"{_safe_path_part(source_identifier)}-{content_sha256[:12]}{suffix}"
        )
        stored = (
            self._put_bytes_artifact(
                payload,
                key,
                context=f"raw artifact {source_name}/{artifact_kind} for run_id={run_id}",
            )
            if self.artifact_store is not None
            else None
        )
        return self.record_artifact_record(
            run_id=run_id,
            name=f"{source_name}_{artifact_kind}",
            stage="raw_capture",
            uri=stored.uri if stored else key,
            local_path=local_path or key,
            content_sha256=stored.sha256 if stored else content_sha256,
            size_bytes=stored.size_bytes if stored else len(payload),
            source_name=source_name,
            artifact_kind=artifact_kind,
            min_date=start_date or effective_date,
            max_date=end_date or effective_date,
            notes=notes,
        )

    def _put_file_artifact(
        self, path: Path, key: str, *, context: str
    ) -> StoredArtifact:
        assert self.artifact_store is not None
        try:
            return self.artifact_store.put_file(path, key)
        except Exception as exc:
            raise RuntimeError(f"failed to store {context}: {key}") from exc

    def _put_bytes_artifact(
        self, payload: bytes, key: str, *, context: str
    ) -> StoredArtifact:
        assert self.artifact_store is not None
        try:
            return self.artifact_store.put_bytes(payload, key)
        except Exception as exc:
            raise RuntimeError(f"failed to store {context}: {key}") from exc


def _safe_path_part(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "artifact"


def _artifact_suffix(*, artifact_kind: str, local_path: str | None) -> str:
    if local_path:
        suffix = Path(local_path).suffix
        if suffix:
            return suffix
    safe_kind = _safe_path_part(artifact_kind)
    if safe_kind in {"json", "csv", "html", "txt", "xml", "cfb"}:
        return f".{safe_kind}"
    return ".bin"


def _last_insert_rowid(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:
        raise RuntimeError("sqlite insert did not return a row id")
    return cursor.lastrowid
