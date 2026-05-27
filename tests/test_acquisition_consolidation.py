from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from contextlib import closing

import pandas as pd
import pytest

import regime_data_fetch.acquisition_consolidation as acquisition_consolidation
from regime_data_fetch.acquisition_consolidation import (
    ConsolidationSource,
    DAILY_OHLCV_ROWS_TABLE,
    consolidate_acquisition_dbs,
)
from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.local_daily_ohlcv_sqlite import _ensure_daily_ohlcv_table


def test_consolidate_acquisition_dbs_requires_explicit_sources(tmp_path: Path) -> None:
    target = tmp_path / "canonical.db"

    with pytest.raises(ValueError, match="requires explicit sources"):
        consolidate_acquisition_dbs(target_db_path=target)

    assert not target.exists()


def test_consolidate_acquisition_dbs_merges_runs_artifacts_outputs_and_ohlcv(
    tmp_path: Path,
) -> None:
    src1 = tmp_path / "src1.db"
    src2 = tmp_path / "src2.db"
    _build_source_db_one(src1)
    _build_source_db_two(src2)

    target = tmp_path / "canonical.db"
    report = consolidate_acquisition_dbs(
        target_db_path=target,
        sources=[
            ConsolidationSource("one", src1),
            ConsolidationSource("two", src2),
        ],
    )

    assert report["final_counts"] == {
        "fetch_runs": 2,
        "artifacts": 2,
        "artifact_blobs": 1,
        "derived_outputs": 3,
        "daily_ohlcv_rows": 1,
        "event_calendar_rows": 1,
        "macro_rows": 0,
        "pmi_rows": 1,
        "pit_constituent_rows": 0,
        "fomc_minutes_rows": 0,
        "powell_speeches_rows": 0,
        "usd_index_rows": 0,
        "aggregate_eps_snapshot_rows": 0,
        "aggregate_eps_wayback_rows": 0,
        "alpaca_market_rows": 0,
    }

    with closing(sqlite3.connect(target)) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status, params_json, notes FROM fetch_runs ORDER BY run_id"
        ).fetchall()
        artifacts = conn.execute(
            "SELECT source_name, artifact_kind, notes FROM artifacts ORDER BY artifact_id"
        ).fetchall()
        outputs = conn.execute(
            "SELECT output_kind, notes FROM derived_outputs ORDER BY output_kind"
        ).fetchall()
        ohlcv = conn.execute(
            f"SELECT symbol, date, close FROM {DAILY_OHLCV_ROWS_TABLE}"
        ).fetchall()
        events = conn.execute(
            "SELECT event_date, event_type FROM event_calendar_rows"
        ).fetchall()
        pmi = conn.execute(
            "SELECT dataset_kind, series_name, period, value FROM pmi_rows"
        ).fetchall()

    assert len(fetch_runs) == 2
    assert '"consolidated_from_label": "one"' in fetch_runs[0][2]
    assert "imported_from=one:" in fetch_runs[0][3]
    assert len(artifacts) == 2
    assert "imported_from=one:" in artifacts[0][2]
    assert len(outputs) == 3
    assert ohlcv == [("SPY", "2026-05-05", 565.0)]
    assert events == [("2026-05-01", "CPI")]
    assert pmi == [("history", "manufacturing", "2026-04", 52.7)]


def test_augment_params_json_logs_unparseable_json_without_raw_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_payload = '{"api_key": "secret-token"'
    caplog.set_level(
        logging.WARNING, logger="regime_data_fetch.acquisition_consolidation"
    )

    augmented = acquisition_consolidation._augment_params_json(
        raw_payload,
        source_label="source-one",
        source_db_path="/tmp/source-one.db",
    )

    assert '"raw_params_json": "{\\"api_key\\": \\"secret-token\\""' in augmented
    assert "params_json unparseable" in caplog.text
    assert "source_label=source-one" in caplog.text
    assert "secret-token" not in caplog.text


def test_consolidate_acquisition_dbs_preserves_multiple_artifact_blobs_per_source(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src_multi_blob.db"
    _build_source_db_with_multiple_blobs(src)

    target = tmp_path / "canonical.db"
    consolidate_acquisition_dbs(
        target_db_path=target,
        sources=[ConsolidationSource("multi", src)],
    )

    with closing(sqlite3.connect(target)) as conn:
        target_blobs = conn.execute("""
            SELECT artifacts.source_identifier, artifact_blobs.content_bytes
            FROM artifact_blobs
            JOIN artifacts ON artifacts.artifact_id = artifact_blobs.artifact_id
            ORDER BY artifacts.source_identifier
            """).fetchall()

    assert target_blobs == [
        ("ident-A", b"blob-bytes-A"),
        ("ident-B", b"blob-bytes-B"),
        ("ident-C", b"blob-bytes-C"),
    ]


def _build_source_db_one(path: Path) -> None:
    store = AcquisitionStore(path)
    run = store.start_fetch_run(fetch_type="events", params={"a": 1})
    store.record_text_artifact(
        run_id=run.run_id,
        source_name="source:one",
        artifact_kind="html",
        source_identifier="one",
        content_text="<html>one</html>",
        notes="artifact-one",
    )
    event_yaml = path.parent / "events.yaml"
    event_yaml.write_text(
        "events:\n"
        "  - date: '2026-05-01'\n"
        "    release_timestamp_et: '2026-05-01T08:30:00-04:00'\n"
        "    market: US\n"
        "    type: CPI\n"
        "    importance: high\n"
        "    source: bls\n"
    )
    store.record_output(
        run_id=run.run_id,
        output_kind="event_calendar_yaml",
        path=event_yaml,
        row_count=1,
    )
    pmi_parquet = path.parent / "pmi.parquet"
    pd.DataFrame(
        [
            {
                "series_name": "manufacturing",
                "period": "2026-04",
                "value": 52.7,
                "release_timestamp": "2026-05-01T10:00:00-04:00",
                "source": "investing_manual",
                "source_url": "https://example.com/pmi",
            }
        ]
    ).to_parquet(pmi_parquet, index=False)
    store.record_output(
        run_id=run.run_id,
        output_kind="pmi_history_parquet",
        path=pmi_parquet,
        row_count=1,
    )
    store.finish_fetch_run(run_id=run.run_id, status="ok", notes="done-one")


def _build_source_db_two(path: Path) -> None:
    store = AcquisitionStore(path)
    run = store.start_fetch_run(fetch_type="ohlcv", params={"b": 2})
    store.record_file_artifact(
        run_id=run.run_id,
        source_name="source:two",
        artifact_kind="csv_manual",
        source_identifier="two",
        file_path=_write_binary_fixture(path.parent / "two.bin"),
        notes="artifact-two",
    )
    with closing(sqlite3.connect(path)) as conn:
        _ensure_daily_ohlcv_table(conn)
        conn.execute(
            """
            INSERT INTO daily_ohlcv_rows (
                symbol, date, open, high, low, close, volume, adjusted_close, source_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "SPY",
                "2026-05-05",
                560.0,
                566.0,
                559.0,
                565.0,
                100,
                565.0,
                "/tmp/x.parquet",
            ),
        )
        conn.commit()
    output = path.parent / "two.json"
    output.write_text("{}")
    store.record_output(
        run_id=run.run_id, output_kind="report_two", path=output, row_count=1
    )
    store.finish_fetch_run(run_id=run.run_id, status="ok", notes="done-two")


def _write_binary_fixture(path: Path) -> Path:
    path.write_bytes(b"fixture")
    return path


def _build_source_db_with_multiple_blobs(path: Path) -> None:
    store = AcquisitionStore(path)
    run = store.start_fetch_run(fetch_type="binary_multi", params={"n": 3})
    blob_a = _write_blob_fixture(path.parent / "a.bin", b"blob-bytes-A")
    blob_b = _write_blob_fixture(path.parent / "b.bin", b"blob-bytes-B")
    blob_c = _write_blob_fixture(path.parent / "c.bin", b"blob-bytes-C")
    store.record_file_artifact(
        run_id=run.run_id,
        source_name="multi:A",
        artifact_kind="binary",
        source_identifier="ident-A",
        file_path=blob_a,
        notes="blob A",
    )
    store.record_file_artifact(
        run_id=run.run_id,
        source_name="multi:B",
        artifact_kind="binary",
        source_identifier="ident-B",
        file_path=blob_b,
        notes="blob B",
    )
    store.record_file_artifact(
        run_id=run.run_id,
        source_name="multi:C",
        artifact_kind="binary",
        source_identifier="ident-C",
        file_path=blob_c,
        notes="blob C",
    )
    store.finish_fetch_run(run_id=run.run_id, status="ok", notes="done-multi")


def _write_blob_fixture(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path
