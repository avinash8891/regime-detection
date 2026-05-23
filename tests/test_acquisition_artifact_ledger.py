from __future__ import annotations

import sqlite3
from pathlib import Path

from regime_data_fetch.acquisition_schema import init_acquisition_schema
from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.artifact_store import (
    ArtifactStore,
    LocalArtifactStore,
    StoredArtifact,
    sha256_file,
)

_STORE_ROOT_URI = "file:///tmp/regime-data"


class FailingArtifactStore(ArtifactStore):
    def put_file(self, source_path: Path, key: str) -> StoredArtifact:
        del source_path, key
        raise OSError("disk full")

    def put_bytes(self, payload: bytes, key: str) -> StoredArtifact:
        del payload, key
        raise OSError("disk full")


def test_acquisition_schema_helper_creates_tables_and_migrates_legacy_artifacts(
    tmp_path: Path,
) -> None:
    assert not hasattr(AcquisitionStore, "_init_schema")

    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE artifacts (
                artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
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
                notes TEXT
            )
            """
        )

        init_acquisition_schema(conn)

        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        artifact_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(artifacts)")
        }

    assert {
        "fetch_runs",
        "artifacts",
        "artifact_blobs",
        "derived_outputs",
        "artifact_records",
        "artifact_lineage",
        "canonical_versions",
        "source_checkpoints",
    }.issubset(table_names)
    assert {"local_path", "content_size_bytes", "content_encoding"}.issubset(
        artifact_columns
    )


def test_acquisition_store_records_artifact_ledger_checkpoint_and_lineage(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "acquisition.db"
    store = AcquisitionStore(db_path)
    run = store.start_fetch_run(fetch_type="sentiment", params={"source": "aaii"})

    raw = store.record_artifact_record(
        run_id=run.run_id,
        name="aaii_raw_cfb",
        stage="raw_capture",
        uri=f"{_STORE_ROOT_URI}/raw_capture/aaii/2026-05-15/sentiment.cfb",
        local_path="data/raw/sentiment/aaii_sentiment_historical.cfb",
        content_sha256="a" * 64,
        size_bytes=1140736,
        source_name="aaii",
        artifact_kind="cfb",
    )
    canonical = store.record_artifact_record(
        run_id=run.run_id,
        name="aaii_sentiment",
        stage="canonical",
        uri=f"{_STORE_ROOT_URI}/canonical/sentiment/aaii_sentiment/as_of=2026-05-15/aaii_sentiment.parquet",
        local_path="data/raw/sentiment/aaii_sentiment.parquet",
        content_sha256="b" * 64,
        size_bytes=2048,
        source_name="aaii",
        artifact_kind="parquet",
        row_count=1900,
        min_date="1987-07-24",
        max_date="2026-05-15",
        schema_version="aaii_sentiment.v1",
    )
    store.record_artifact_lineage(
        output_artifact_record_id=canonical.artifact_record_id,
        input_artifact_record_id=raw.artifact_record_id,
        transform_name="normalize_aaii_sentiment",
    )
    store.record_canonical_version(
        dataset_name="aaii_sentiment",
        version="as_of=2026-05-15",
        artifact_record_id=canonical.artifact_record_id,
        manifest_uri="manifests/regime_engine_2026-05-15.yaml",
        status="active",
    )
    store.set_source_checkpoint(
        source_name="aaii",
        cursor_key="survey_week",
        cursor_value="2026-05-15",
        successful_run_id=run.run_id,
    )

    with sqlite3.connect(db_path) as conn:
        artifact_rows = conn.execute(
            "SELECT name, stage, uri, local_path, row_count FROM artifact_records ORDER BY artifact_record_id"
        ).fetchall()
        lineage_rows = conn.execute(
            "SELECT output_artifact_record_id, input_artifact_record_id, transform_name FROM artifact_lineage"
        ).fetchall()
        version_rows = conn.execute(
            "SELECT dataset_name, version, artifact_record_id, manifest_uri, status FROM canonical_versions"
        ).fetchall()
        checkpoint_rows = conn.execute(
            "SELECT source_name, cursor_key, cursor_value, successful_run_id FROM source_checkpoints"
        ).fetchall()

    assert artifact_rows == [
        (
            "aaii_raw_cfb",
            "raw_capture",
            f"{_STORE_ROOT_URI}/raw_capture/aaii/2026-05-15/sentiment.cfb",
            "data/raw/sentiment/aaii_sentiment_historical.cfb",
            None,
        ),
        (
            "aaii_sentiment",
            "canonical",
            f"{_STORE_ROOT_URI}/canonical/sentiment/aaii_sentiment/as_of=2026-05-15/aaii_sentiment.parquet",
            "data/raw/sentiment/aaii_sentiment.parquet",
            1900,
        ),
    ]
    assert lineage_rows == [
        (
            canonical.artifact_record_id,
            raw.artifact_record_id,
            "normalize_aaii_sentiment",
        )
    ]
    assert version_rows == [
        (
            "aaii_sentiment",
            "as_of=2026-05-15",
            canonical.artifact_record_id,
            "manifests/regime_engine_2026-05-15.yaml",
            "active",
        )
    ]
    assert checkpoint_rows == [("aaii", "survey_week", "2026-05-15", run.run_id)]


def test_source_checkpoint_upserts_latest_successful_run(tmp_path: Path) -> None:
    store = AcquisitionStore(tmp_path / "acquisition.db")
    first = store.start_fetch_run(fetch_type="macro", params={"run": 1})
    second = store.start_fetch_run(fetch_type="macro", params={"run": 2})

    store.set_source_checkpoint(
        source_name="fred",
        cursor_key="DGS10",
        cursor_value="2026-05-14",
        successful_run_id=first.run_id,
    )
    store.set_source_checkpoint(
        source_name="fred",
        cursor_key="DGS10",
        cursor_value="2026-05-15",
        successful_run_id=second.run_id,
    )

    assert (
        store.get_source_checkpoint(source_name="fred", cursor_key="DGS10")
        == "2026-05-15"
    )
    with sqlite3.connect(tmp_path / "acquisition.db") as conn:
        row = conn.execute(
            """
            SELECT successful_run_id
            FROM source_checkpoints
            WHERE source_name = 'fred' AND cursor_key = 'DGS10'
            """
        ).fetchone()

    assert row == (second.run_id,)


def test_acquisition_store_uploads_raw_and_output_artifacts_to_configured_store(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "acquisition.db"
    artifact_root = tmp_path / "artifact-store"
    store = AcquisitionStore(db_path, artifact_store=LocalArtifactStore(artifact_root))
    run = store.start_fetch_run(fetch_type="sentiment", params={"source": "aaii"})

    raw = store.record_text_artifact(
        run_id=run.run_id,
        source_name="aaii",
        artifact_kind="html",
        source_identifier="https://www.aaii.com/sentimentsurvey",
        content_text="<html>survey</html>",
        effective_date="2026-05-15",
    )
    output_path = tmp_path / "data" / "raw" / "sentiment" / "aaii_sentiment.parquet"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"canonical")
    canonical = store.record_output(
        run_id=run.run_id,
        output_kind="aaii_sentiment_parquet",
        path=output_path,
        row_count=1,
        min_date="2026-05-15",
        max_date="2026-05-15",
    )

    assert raw.artifact_record_id is not None
    assert canonical is not None
    assert (
        list(
            (artifact_root / "raw_capture" / "aaii" / f"run_id={run.run_id}").iterdir()
        )[0].read_text()
        == "<html>survey</html>"
    )
    assert (
        artifact_root
        / "canonical"
        / "aaii_sentiment_parquet"
        / f"run_id={run.run_id}"
        / "aaii_sentiment.parquet"
    ).read_bytes() == b"canonical"

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT stage, uri, content_sha256 FROM artifact_records ORDER BY artifact_record_id"
        ).fetchall()

    assert [row[0] for row in rows] == ["raw_capture", "canonical"]
    assert rows[0][1].startswith(
        (
            artifact_root.resolve() / "raw_capture" / "aaii" / f"run_id={run.run_id}"
        ).as_uri()
    )
    assert (
        rows[1][1]
        == (
            artifact_root.resolve()
            / "canonical"
            / "aaii_sentiment_parquet"
            / f"run_id={run.run_id}"
            / "aaii_sentiment.parquet"
        ).as_uri()
    )


def test_record_file_artifact_without_blob_storage_does_not_read_entire_file(
    tmp_path: Path, monkeypatch
) -> None:
    store = AcquisitionStore(tmp_path / "acquisition.db")
    run = store.start_fetch_run(fetch_type="daily_ohlcv", params={"source": "local"})
    source = tmp_path / "large.parquet"
    source.write_bytes(b"large parquet payload")

    original_read_bytes = Path.read_bytes

    def fail_read_bytes(path: Path) -> bytes:
        if path == source:
            raise AssertionError("record_file_artifact must stream-hash the file")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    recorded = store.record_file_artifact(
        run_id=run.run_id,
        source_name="polygon",
        artifact_kind="parquet",
        source_identifier="SPY",
        file_path=source,
        store_bytes=False,
    )

    assert recorded.content_sha256 == sha256_file(source)


def test_record_output_hashes_file_without_reading_entire_payload(
    tmp_path: Path, monkeypatch
) -> None:
    store = AcquisitionStore(tmp_path / "acquisition.db")
    run = store.start_fetch_run(fetch_type="daily_ohlcv", params={"source": "local"})
    output = tmp_path / "canonical.parquet"
    output.write_bytes(b"canonical parquet payload")

    original_read_bytes = Path.read_bytes

    def fail_read_bytes(path: Path) -> bytes:
        if path == output:
            raise AssertionError("record_output must stream-hash the file")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    recorded = store.record_output(
        run_id=run.run_id,
        output_kind="daily_ohlcv",
        path=output,
        row_count=1,
    )

    assert recorded is not None
    assert recorded.content_sha256 == sha256_file(output)


def test_acquisition_store_adds_context_to_artifact_store_failures(
    tmp_path: Path,
) -> None:
    store = AcquisitionStore(
        tmp_path / "acquisition.db", artifact_store=FailingArtifactStore()
    )
    run = store.start_fetch_run(fetch_type="sentiment", params={"source": "aaii"})

    try:
        store.record_text_artifact(
            run_id=run.run_id,
            source_name="aaii",
            artifact_kind="html",
            source_identifier="https://www.aaii.com/sentimentsurvey",
            content_text="<html>survey</html>",
        )
    except RuntimeError as exc:
        assert f"failed to store raw artifact aaii/html for run_id={run.run_id}" in str(
            exc
        )
        assert isinstance(exc.__cause__, OSError)
    else:
        raise AssertionError("expected contextual artifact-store failure")
