from __future__ import annotations

import sqlite3
from contextlib import closing

from regime_data_fetch.acquisition_schema import init_acquisition_schema

EXPECTED_SCHEMA_COLUMNS = {
    "fetch_runs": (
        "run_id",
        "fetch_type",
        "started_at_utc",
        "finished_at_utc",
        "status",
        "params_json",
        "notes",
    ),
    "artifacts": (
        "artifact_id",
        "run_id",
        "source_name",
        "artifact_kind",
        "source_identifier",
        "content_text",
        "content_sha256",
        "downloaded_at_utc",
        "effective_date",
        "start_date",
        "end_date",
        "timezone",
        "calendar_assumption",
        "adjustment_policy",
        "license_note",
        "notes",
        "local_path",
        "content_size_bytes",
        "content_encoding",
    ),
    "artifact_blobs": (
        "artifact_id",
        "content_bytes",
    ),
    "derived_outputs": (
        "output_id",
        "run_id",
        "output_kind",
        "path",
        "content_sha256",
        "row_count",
        "min_date",
        "max_date",
        "recorded_at_utc",
        "notes",
    ),
    "artifact_records": (
        "artifact_record_id",
        "run_id",
        "name",
        "stage",
        "uri",
        "local_path",
        "content_sha256",
        "size_bytes",
        "source_name",
        "artifact_kind",
        "row_count",
        "min_date",
        "max_date",
        "schema_version",
        "recorded_at_utc",
        "notes",
    ),
    "artifact_lineage": (
        "lineage_id",
        "output_artifact_record_id",
        "input_artifact_record_id",
        "transform_name",
        "recorded_at_utc",
    ),
    "canonical_versions": (
        "canonical_version_id",
        "dataset_name",
        "version",
        "artifact_record_id",
        "manifest_uri",
        "status",
        "recorded_at_utc",
    ),
    "source_checkpoints": (
        "checkpoint_id",
        "source_name",
        "cursor_key",
        "cursor_value",
        "successful_run_id",
        "updated_at_utc",
    ),
}


def test_acquisition_schema_initializes_expected_tables_and_columns() -> None:
    with closing(sqlite3.connect(":memory:")) as conn:
        init_acquisition_schema(conn)

        table_names = {row[0] for row in conn.execute("""
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """)}
        table_columns = {
            table_name: tuple(
                row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")
            )
            for table_name in EXPECTED_SCHEMA_COLUMNS
        }

    assert table_names == set(EXPECTED_SCHEMA_COLUMNS)
    assert table_columns == EXPECTED_SCHEMA_COLUMNS
