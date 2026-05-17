from __future__ import annotations

import sqlite3

from regime_data_fetch.acquisition_consolidation_normalized import (
    EVENT_CALENDAR_ROWS_TABLE,
    _ensure_normalized_tables,
    _import_normalized_output,
)


def test_import_normalized_output_loads_event_calendar_yaml(tmp_path) -> None:
    path = tmp_path / "events.yaml"
    path.write_text(
        "\n".join(
            [
                "events:",
                '  - date: "2026-01-28"',
                '    release_timestamp_et: "2026-01-28T14:00:00-05:00"',
                '    market: "US"',
                '    type: "FOMC"',
                '    importance: "high"',
                '    source: "federalreserve.gov:fomccalendars"',
            ]
        )
        + "\n"
    )
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE fetch_runs (run_id INTEGER PRIMARY KEY, fetch_type TEXT)"
    )
    conn.execute("INSERT INTO fetch_runs (run_id, fetch_type) VALUES (7, 'events')")
    _ensure_normalized_tables(conn)

    table = _import_normalized_output(
        dst_conn=conn,
        run_id=7,
        output_kind="event_calendar_yaml",
        path=path,
    )

    assert table == EVENT_CALENDAR_ROWS_TABLE
    assert conn.execute(
        "SELECT run_id, event_date, event_type, source FROM event_calendar_rows"
    ).fetchall() == [
        (7, "2026-01-28", "FOMC", "federalreserve.gov:fomccalendars")
    ]


def test_import_normalized_output_returns_none_for_unknown_kind(tmp_path) -> None:
    conn = sqlite3.connect(":memory:")
    _ensure_normalized_tables(conn)

    table = _import_normalized_output(
        dst_conn=conn,
        run_id=1,
        output_kind="unsupported_output",
        path=tmp_path / "ignored.parquet",
    )

    assert table is None
