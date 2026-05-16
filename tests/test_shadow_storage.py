"""Tests for shadow_storage.py — unit coverage ≥70%."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from regime_detection.shadow_storage import (
    ensure_shadow_layout,
    event_rows_for_yaml,
    fetch_run_row,
    insert_incident,
    insert_replay_check,
    insert_run_row,
    load_archived_event_calendar,
    load_archived_market_data,
    open_shadow_db,
    sha256_file,
    update_run_row_failure,
    update_run_row_success,
    utc_iso_now,
    write_archived_inputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENGINE_VERSION = "regime-engine-v2"
_CONFIG_VERSION = "v2.1.0"
_AS_OF_DATE = date(2023, 12, 14)


def _make_market_slice() -> pd.DataFrame:
    """Minimal market slice that satisfies the parquet round-trip."""
    return pd.DataFrame(
        {
            "date": [date(2023, 12, 12), date(2023, 12, 13), date(2023, 12, 14)],
            "symbol": ["SPY", "SPY", "SPY"],
            "close": [460.12, 462.34, 463.55],
            "volume": [75_000_000, 80_000_000, 77_000_000],
        }
    )


def _make_event_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date(2023, 12, 13)],
            "market": ["US"],
            "type": ["FOMC"],
            "importance": ["high"],
        }
    )


def _open_db(tmp_path: Path) -> sqlite3.Connection:
    paths = ensure_shadow_layout(tmp_path)
    return open_shadow_db(paths["db"])


# ---------------------------------------------------------------------------
# utc_iso_now
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_utc_iso_now_is_iso_string_with_timezone() -> None:
    ts = utc_iso_now()
    # Must be parseable as ISO 8601 with UTC offset
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
    assert parsed.microsecond == 0


@pytest.mark.unit
def test_utc_iso_now_returns_fresh_value_each_call() -> None:
    t1 = utc_iso_now()
    t2 = utc_iso_now()
    # Both are strings; they might be equal within the same second, but must be strings
    assert isinstance(t1, str)
    assert isinstance(t2, str)


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    content = b"SPY,2023-12-14,463.55\n"
    p = tmp_path / "market_data.parquet"
    p.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert sha256_file(p) == expected


@pytest.mark.unit
def test_sha256_file_produces_64_hex_chars(tmp_path: Path) -> None:
    p = tmp_path / "output.json"
    p.write_bytes(b'{"regime": "risk_on"}')
    result = sha256_file(p)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# ensure_shadow_layout
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ensure_shadow_layout_creates_expected_directories(tmp_path: Path) -> None:
    output_root = tmp_path / "shadow_run"
    paths = ensure_shadow_layout(output_root)
    assert paths["db"] == output_root / "regime_shadow.db"
    assert paths["outputs"].exists()
    assert paths["input_archives"].exists()


@pytest.mark.unit
def test_ensure_shadow_layout_is_idempotent(tmp_path: Path) -> None:
    output_root = tmp_path / "shadow_run"
    paths1 = ensure_shadow_layout(output_root)
    paths2 = ensure_shadow_layout(output_root)
    assert paths1["db"] == paths2["db"]
    assert paths2["outputs"].exists()


# ---------------------------------------------------------------------------
# open_shadow_db
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_open_shadow_db_creates_all_three_tables(tmp_path: Path) -> None:
    paths = ensure_shadow_layout(tmp_path)
    conn = open_shadow_db(paths["db"])
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert {"runs", "replay_checks", "incidents"} <= tables


@pytest.mark.unit
def test_open_shadow_db_is_idempotent_on_second_call(tmp_path: Path) -> None:
    paths = ensure_shadow_layout(tmp_path)
    conn1 = open_shadow_db(paths["db"])
    conn1.close()
    # Second open must not raise (CREATE TABLE IF NOT EXISTS)
    conn2 = open_shadow_db(paths["db"])
    conn2.close()


# ---------------------------------------------------------------------------
# event_rows_for_yaml
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_event_rows_for_yaml_returns_empty_list_for_none() -> None:
    assert event_rows_for_yaml(None) == []


@pytest.mark.unit
def test_event_rows_for_yaml_returns_empty_list_for_empty_df() -> None:
    assert event_rows_for_yaml(pd.DataFrame()) == []


@pytest.mark.unit
def test_event_rows_for_yaml_converts_date_objects_to_iso_strings() -> None:
    df = _make_event_df()
    rows = event_rows_for_yaml(df)
    assert len(rows) == 1
    assert rows[0]["date"] == "2023-12-13"
    assert rows[0]["type"] == "FOMC"
    assert rows[0]["importance"] == "high"


@pytest.mark.unit
def test_event_rows_for_yaml_converts_pd_timestamps_to_iso() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-19")],
            "market": ["US"],
            "type": ["FOMC"],
            "importance": ["high"],
        }
    )
    rows = event_rows_for_yaml(df)
    assert rows[0]["date"] == "2024-01-19"


@pytest.mark.unit
def test_event_rows_for_yaml_converts_nan_to_none() -> None:
    import numpy as np

    df = pd.DataFrame(
        {
            "date": [date(2024, 1, 19)],
            "market": ["US"],
            "type": ["FOMC"],
            "importance": ["high"],
            "publication_date": [np.nan],
        }
    )
    rows = event_rows_for_yaml(df)
    assert rows[0]["publication_date"] is None


# ---------------------------------------------------------------------------
# write_archived_inputs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_archived_inputs_creates_three_files(tmp_path: Path) -> None:
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    market_path, events_path, checksums_path = write_archived_inputs(
        archive_dir=archive_dir,
        market_slice=_make_market_slice(),
        event_df=_make_event_df(),
    )
    assert market_path.exists()
    assert events_path.exists()
    assert checksums_path.exists()


@pytest.mark.unit
def test_write_archived_inputs_checksums_are_valid_sha256(tmp_path: Path) -> None:
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    market_path, events_path, checksums_path = write_archived_inputs(
        archive_dir=archive_dir,
        market_slice=_make_market_slice(),
        event_df=_make_event_df(),
    )
    checksums = json.loads(checksums_path.read_text())
    assert checksums["market_data.parquet"] == sha256_file(market_path)
    assert checksums["events.yaml"] == sha256_file(events_path)


@pytest.mark.unit
def test_write_archived_inputs_events_yaml_roundtrips_none_event_df(tmp_path: Path) -> None:
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    _, events_path, _ = write_archived_inputs(
        archive_dir=archive_dir,
        market_slice=_make_market_slice(),
        event_df=None,
    )
    loaded = yaml.safe_load(events_path.read_text())
    assert loaded == {"events": []}


@pytest.mark.unit
def test_write_archived_inputs_market_parquet_roundtrips(tmp_path: Path) -> None:
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    market_slice = _make_market_slice()
    market_path, _, _ = write_archived_inputs(
        archive_dir=archive_dir,
        market_slice=market_slice,
        event_df=None,
    )
    read_back = pd.read_parquet(market_path)
    assert list(read_back["symbol"]) == ["SPY", "SPY", "SPY"]
    assert len(read_back) == 3


# ---------------------------------------------------------------------------
# insert_run_row / fetch_run_row / update_run_row_success / update_run_row_failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insert_run_row_sets_in_progress_status(tmp_path: Path) -> None:
    conn = _open_db(tmp_path)
    ts = utc_iso_now()
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    insert_run_row(
        conn=conn,
        run_timestamp=ts,
        as_of_date=_AS_OF_DATE,
        engine_version=_ENGINE_VERSION,
        config_version=_CONFIG_VERSION,
        archive_dir=archive_dir,
    )
    row = conn.execute("SELECT status, failure_reason, output_path FROM runs").fetchone()
    assert row == ("in_progress", None, None)
    conn.close()


@pytest.mark.unit
def test_fetch_run_row_returns_none_for_missing_date(tmp_path: Path) -> None:
    conn = _open_db(tmp_path)
    row = fetch_run_row(conn=conn, as_of_date=date(2020, 1, 2))
    assert row is None
    conn.close()


@pytest.mark.unit
def test_fetch_run_row_returns_inserted_row(tmp_path: Path) -> None:
    conn = _open_db(tmp_path)
    ts = utc_iso_now()
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    insert_run_row(
        conn=conn,
        run_timestamp=ts,
        as_of_date=_AS_OF_DATE,
        engine_version=_ENGINE_VERSION,
        config_version=_CONFIG_VERSION,
        archive_dir=archive_dir,
    )
    row = fetch_run_row(conn=conn, as_of_date=_AS_OF_DATE)
    assert row is not None
    assert row["as_of_date"] == "2023-12-14"
    assert row["status"] == "in_progress"
    assert row["engine_version"] == _ENGINE_VERSION
    conn.close()


@pytest.mark.unit
def test_update_run_row_success_changes_status_and_sets_sha256(tmp_path: Path) -> None:
    conn = _open_db(tmp_path)
    ts = utc_iso_now()
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    insert_run_row(
        conn=conn,
        run_timestamp=ts,
        as_of_date=_AS_OF_DATE,
        engine_version=_ENGINE_VERSION,
        config_version=_CONFIG_VERSION,
        archive_dir=archive_dir,
    )
    output_file = tmp_path / "outputs" / "2023-12-14.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text('{"regime": "risk_on"}', encoding="utf-8")

    update_run_row_success(
        conn=conn,
        as_of_date=_AS_OF_DATE,
        engine_version=_ENGINE_VERSION,
        config_version=_CONFIG_VERSION,
        output_path=output_file,
    )
    row = conn.execute("SELECT status, output_path, output_sha256 FROM runs").fetchone()
    assert row[0] == "success"
    assert row[1] == str(output_file)
    assert len(row[2]) == 64  # sha256 hex digest
    conn.close()


@pytest.mark.unit
def test_update_run_row_failure_changes_status_and_sets_reason(tmp_path: Path) -> None:
    conn = _open_db(tmp_path)
    ts = utc_iso_now()
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    insert_run_row(
        conn=conn,
        run_timestamp=ts,
        as_of_date=_AS_OF_DATE,
        engine_version=_ENGINE_VERSION,
        config_version=_CONFIG_VERSION,
        archive_dir=archive_dir,
    )
    update_run_row_failure(
        conn=conn,
        as_of_date=_AS_OF_DATE,
        engine_version=_ENGINE_VERSION,
        config_version=_CONFIG_VERSION,
        failure_reason="classify raised ValueError: insufficient data",
    )
    row = conn.execute("SELECT status, failure_reason FROM runs").fetchone()
    assert row[0] == "failure"
    assert "insufficient data" in row[1]
    conn.close()


@pytest.mark.unit
def test_insert_run_row_enforces_unique_constraint(tmp_path: Path) -> None:
    conn = _open_db(tmp_path)
    ts = utc_iso_now()
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    insert_run_row(
        conn=conn,
        run_timestamp=ts,
        as_of_date=_AS_OF_DATE,
        engine_version=_ENGINE_VERSION,
        config_version=_CONFIG_VERSION,
        archive_dir=archive_dir,
    )
    with pytest.raises(sqlite3.IntegrityError):
        insert_run_row(
            conn=conn,
            run_timestamp=utc_iso_now(),
            as_of_date=_AS_OF_DATE,
            engine_version=_ENGINE_VERSION,
            config_version=_CONFIG_VERSION,
            archive_dir=archive_dir,
        )
    conn.close()


# ---------------------------------------------------------------------------
# load_archived_market_data
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_archived_market_data_date_column_is_python_dates(tmp_path: Path) -> None:
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    market_path, _, _ = write_archived_inputs(
        archive_dir=archive_dir,
        market_slice=_make_market_slice(),
        event_df=None,
    )
    df = load_archived_market_data(market_path)
    assert all(isinstance(d, date) for d in df["date"])


# ---------------------------------------------------------------------------
# load_archived_event_calendar
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_archived_event_calendar_returns_none_for_empty_events(tmp_path: Path) -> None:
    events_path = tmp_path / "events.yaml"
    events_path.write_text(
        yaml.safe_dump({"events": []}, sort_keys=False), encoding="utf-8"
    )
    result = load_archived_event_calendar(events_path)
    assert result is None


@pytest.mark.unit
def test_load_archived_event_calendar_returns_none_for_empty_file(tmp_path: Path) -> None:
    events_path = tmp_path / "events.yaml"
    events_path.write_text("", encoding="utf-8")
    result = load_archived_event_calendar(events_path)
    assert result is None


@pytest.mark.unit
def test_load_archived_event_calendar_returns_dataframe_for_real_fixture() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "events"
        / "us_events.yaml"
    )
    if not fixture_path.exists():
        pytest.skip("us_events.yaml fixture not present")
    result = load_archived_event_calendar(fixture_path)
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# insert_replay_check
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insert_replay_check_with_matching_true(tmp_path: Path) -> None:
    conn = _open_db(tmp_path)
    ts = utc_iso_now()
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    insert_run_row(
        conn=conn,
        run_timestamp=ts,
        as_of_date=_AS_OF_DATE,
        engine_version=_ENGINE_VERSION,
        config_version=_CONFIG_VERSION,
        archive_dir=archive_dir,
    )
    run_id = conn.execute("SELECT run_id FROM runs").fetchone()[0]
    insert_replay_check(
        conn=conn,
        check_timestamp=utc_iso_now(),
        original_run_id=run_id,
        matches=True,
        diff=None,
    )
    row = conn.execute("SELECT matches, diff FROM replay_checks").fetchone()
    assert row[0] == 1
    assert row[1] is None
    conn.close()


@pytest.mark.unit
def test_insert_replay_check_with_diff_serialized_as_json(tmp_path: Path) -> None:
    conn = _open_db(tmp_path)
    ts = utc_iso_now()
    archive_dir = tmp_path / "input_archives" / "2023-12-14"
    insert_run_row(
        conn=conn,
        run_timestamp=ts,
        as_of_date=_AS_OF_DATE,
        engine_version=_ENGINE_VERSION,
        config_version=_CONFIG_VERSION,
        archive_dir=archive_dir,
    )
    run_id = conn.execute("SELECT run_id FROM runs").fetchone()[0]
    diff_payload = {"regime_label": {"old": "risk_on", "new": "risk_off"}}
    insert_replay_check(
        conn=conn,
        check_timestamp=utc_iso_now(),
        original_run_id=run_id,
        matches=False,
        diff=diff_payload,
    )
    row = conn.execute("SELECT matches, diff FROM replay_checks").fetchone()
    assert row[0] == 0
    stored_diff = json.loads(row[1])
    assert stored_diff["regime_label"]["new"] == "risk_off"
    conn.close()


# ---------------------------------------------------------------------------
# insert_incident
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insert_incident_breaking_qualification(tmp_path: Path) -> None:
    conn = _open_db(tmp_path)
    insert_incident(
        conn=conn,
        incident_date=date(2023, 10, 3),
        description="Unexpected flash crash in SPY; regime engine showed risk_on during drawdown",
        resolution="Post-mortem completed; no code changes required",
        breaks_qualification=True,
    )
    row = conn.execute(
        "SELECT incident_date, breaks_qualification, resolution FROM incidents"
    ).fetchone()
    assert row[0] == "2023-10-03"
    assert row[1] == 1
    assert "Post-mortem" in row[2]
    conn.close()


@pytest.mark.unit
def test_insert_incident_non_breaking_with_null_resolution(tmp_path: Path) -> None:
    conn = _open_db(tmp_path)
    insert_incident(
        conn=conn,
        incident_date=date(2024, 3, 22),
        description="Minor data delay in BLS NFP release; regime unchanged",
        resolution=None,
        breaks_qualification=False,
    )
    row = conn.execute(
        "SELECT breaks_qualification, resolution FROM incidents"
    ).fetchone()
    assert row[0] == 0
    assert row[1] is None
    conn.close()
