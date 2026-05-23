from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from regime_detection.shadow_storage import (
    fetch_run_row,
    insert_incident,
    insert_replay_check,
    insert_run_row,
    load_archived_market_data,
    open_shadow_db,
    sha256_file,
    update_run_row_failure,
    update_run_row_success,
    write_archived_inputs,
)


def test_open_shadow_db_creates_durable_shadow_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "shadow" / "regime_shadow.db"
    db_path.parent.mkdir()

    with open_shadow_db(db_path) as conn:
        table_names = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name IN ('runs', 'replay_checks', 'incidents')
            ORDER BY name
            """
        ).fetchall()

    with sqlite3.connect(db_path) as conn:
        durable_table_names = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name IN ('runs', 'replay_checks', 'incidents')
            ORDER BY name
            """
        ).fetchall()

    assert table_names == [("incidents",), ("replay_checks",), ("runs",)]
    assert durable_table_names == table_names


def test_write_archived_inputs_persists_files_and_checksums(tmp_path: Path) -> None:
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    market_slice = pd.DataFrame(
        [
            {"date": date(2023, 12, 13), "symbol": "SPY", "close": 470.50},
            {"date": date(2023, 12, 14), "symbol": "SPY", "close": 472.01},
        ]
    )
    event_df = pd.DataFrame(
        [
            {
                "event_id": "fomc_2023_12",
                "event_date": pd.Timestamp("2023-12-13"),
                "source": "fomc",
                "metadata": {"release_date": pd.Timestamp("2023-12-13")},
            }
        ]
    )

    market_path, events_path, checksums_path = write_archived_inputs(
        archive_dir=archive_dir,
        market_slice=market_slice,
        event_df=event_df,
    )

    archived_market = pd.read_parquet(market_path)
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    events_payload = events_path.read_text(encoding="utf-8")

    assert archived_market.to_dict(orient="records") == [
        {"date": date(2023, 12, 13), "symbol": "SPY", "close": 470.50},
        {"date": date(2023, 12, 14), "symbol": "SPY", "close": 472.01},
    ]
    assert "event_id: fomc_2023_12" in events_payload
    assert "event_date: '2023-12-13'" in events_payload
    assert checksums == {
        "events.yaml": sha256_file(events_path),
        "market_data.parquet": sha256_file(market_path),
    }


def test_run_row_success_and_failure_updates_are_durable(tmp_path: Path) -> None:
    db_path = tmp_path / "regime_shadow.db"
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    archive_dir.mkdir(parents=True)
    output_path = tmp_path / "outputs" / "2023-12-14.json"
    output_path.parent.mkdir()
    output_path.write_text('{"as_of_date": "2023-12-14"}\n', encoding="utf-8")

    with open_shadow_db(db_path) as conn:
        insert_run_row(
            conn=conn,
            run_timestamp="2026-05-17T05:15:00+00:00",
            as_of_date=date(2023, 12, 14),
            engine_version="regime-engine-test",
            config_version="core3-v2.0.0",
            archive_dir=archive_dir,
        )
        update_run_row_success(
            conn=conn,
            as_of_date=date(2023, 12, 14),
            engine_version="regime-engine-test",
            config_version="core3-v2.0.0",
            output_path=output_path,
        )
        run_row = fetch_run_row(conn=conn, as_of_date=date(2023, 12, 14))
        assert run_row is not None
        assert run_row["status"] == "success"
        assert run_row["input_archive_path"] == str(archive_dir)

        insert_run_row(
            conn=conn,
            run_timestamp="2026-05-17T05:16:00+00:00",
            as_of_date=date(2023, 12, 15),
            engine_version="regime-engine-test",
            config_version="core3-v2.0.0",
            archive_dir=tmp_path / "input_archives" / "2023-12-15",
        )
        update_run_row_failure(
            conn=conn,
            as_of_date=date(2023, 12, 15),
            engine_version="regime-engine-test",
            config_version="core3-v2.0.0",
            failure_reason="forced classify failure",
        )

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT as_of_date, status, failure_reason, output_path, output_sha256
            FROM runs
            ORDER BY as_of_date
            """
        ).fetchall()

    assert rows == [
        (
            "2023-12-14",
            "success",
            None,
            str(output_path),
            sha256_file(output_path),
        ),
        (
            "2023-12-15",
            "failure",
            "forced classify failure",
            None,
            None,
        ),
    ]


def test_fetch_run_row_rejects_partial_identity_tuple(tmp_path: Path) -> None:
    db_path = tmp_path / "regime_shadow.db"

    with open_shadow_db(db_path) as conn:
        with pytest.raises(ValueError, match="engine_version and config_version"):
            fetch_run_row(
                conn=conn,
                as_of_date=date(2023, 12, 14),
                engine_version="regime-engine-test",
            )
        with pytest.raises(ValueError, match="engine_version and config_version"):
            fetch_run_row(
                conn=conn,
                as_of_date=date(2023, 12, 14),
                config_version="core3-v2.0.0",
            )


def test_replay_check_and_incident_insertions_are_durable(tmp_path: Path) -> None:
    db_path = tmp_path / "regime_shadow.db"

    with open_shadow_db(db_path) as conn:
        insert_run_row(
            conn=conn,
            run_timestamp="2026-05-17T05:15:00+00:00",
            as_of_date=date(2023, 12, 14),
            engine_version="regime-engine-test",
            config_version="core3-v2.0.0",
            archive_dir=tmp_path / "input_archives" / "2023-12-14",
        )
        run_id = conn.execute(
            "SELECT run_id FROM runs WHERE as_of_date = ?",
            ("2023-12-14",),
        ).fetchone()[0]
        insert_replay_check(
            conn=conn,
            check_timestamp="2026-05-17T05:20:00+00:00",
            original_run_id=run_id,
            matches=False,
            diff={"transition_risk_label": {"stored": "calm", "replayed": "risk"}},
        )
        insert_incident(
            conn=conn,
            incident_date=date(2023, 12, 15),
            description="Replay mismatch for 2023-12-14",
            resolution=None,
            breaks_qualification=True,
        )

    with sqlite3.connect(db_path) as conn:
        replay_rows = conn.execute(
            """
            SELECT original_run_id, matches, diff
            FROM replay_checks
            """
        ).fetchall()
        incident_rows = conn.execute(
            """
            SELECT incident_date, description, resolution, breaks_qualification
            FROM incidents
            """
        ).fetchall()

    assert replay_rows == [
        (
            run_id,
            0,
            json.dumps(
                {"transition_risk_label": {"stored": "calm", "replayed": "risk"}},
                sort_keys=True,
            ),
        )
    ]
    assert incident_rows == [
        ("2023-12-15", "Replay mismatch for 2023-12-14", None, 1)
    ]


def test_load_archived_market_data_missing_archive_path_raises(tmp_path: Path) -> None:
    missing_market_archive = tmp_path / "input_archives" / "missing" / "market_data.parquet"

    with pytest.raises(FileNotFoundError):
        load_archived_market_data(missing_market_archive)
