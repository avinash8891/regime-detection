from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from regime_detection.calendar import nyse_calendar
from regime_detection.shadow_storage import open_shadow_db

ENGINE_VERSION = "regime-engine-vtest"
CONFIG_VERSION = "core3-test"


def _sessions(count: int) -> list[date]:
    schedule = nyse_calendar().schedule(
        start_date=date(2023, 1, 3),
        end_date=date(2024, 6, 30),
    )
    sessions = list(schedule.index.date)[:count]
    assert len(sessions) == count
    return sessions


def _insert_success_runs(conn: sqlite3.Connection, sessions: list[date]) -> list[int]:
    run_ids: list[int] = []
    for session in sessions:
        cursor = conn.execute(
            """
            INSERT INTO runs (
                run_timestamp, as_of_date, engine_version, config_version,
                status, failure_reason, input_archive_path, output_path,
                output_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{session.isoformat()}T22:00:00+00:00",
                session.isoformat(),
                ENGINE_VERSION,
                CONFIG_VERSION,
                "success",
                None,
                f"input_archives/{session.isoformat()}",
                f"outputs/{session.isoformat()}.json",
                "sha",
            ),
        )
        run_ids.append(int(cursor.lastrowid))
    conn.commit()
    return run_ids


def test_shadow_qualification_passes_after_252_consecutive_clean_sessions(
    tmp_path: Path,
) -> None:
    from regime_detection.shadow_qualification import evaluate_shadow_qualification

    db_path = tmp_path / "regime_shadow.db"
    sessions = _sessions(252)
    with open_shadow_db(db_path) as conn:
        _insert_success_runs(conn, sessions)
        result = evaluate_shadow_qualification(
            conn=conn,
            end_date=sessions[-1],
            engine_version=ENGINE_VERSION,
            config_version=CONFIG_VERSION,
        )

    assert result == {
        "qualifies": True,
        "current_consecutive_sessions": 252,
        "required_sessions": 252,
        "window_start": sessions[0].isoformat(),
        "window_end": sessions[-1].isoformat(),
        "blocking_reasons": [],
    }


def test_shadow_qualification_resets_after_missing_session_inside_window(
    tmp_path: Path,
) -> None:
    from regime_detection.shadow_qualification import evaluate_shadow_qualification

    db_path = tmp_path / "regime_shadow.db"
    sessions = _sessions(260)
    kept_sessions = sessions[:200] + sessions[201:]
    with open_shadow_db(db_path) as conn:
        _insert_success_runs(conn, kept_sessions)
        result = evaluate_shadow_qualification(
            conn=conn,
            end_date=sessions[-1],
            engine_version=ENGINE_VERSION,
            config_version=CONFIG_VERSION,
        )

    assert result["qualifies"] is False
    assert result["current_consecutive_sessions"] == 59
    assert result["window_start"] == sessions[201].isoformat()
    assert "missing_session_gap" in result["blocking_reasons"]


def test_shadow_qualification_resets_after_interior_failure_run(
    tmp_path: Path,
) -> None:
    # F-029: an interior status=='failure' session breaks the clean window — it is a
    # non_success_run, qualifies is False, and the count resets to the sessions after
    # the failure (distinct from a missing-session gap).
    from regime_detection.shadow_qualification import evaluate_shadow_qualification

    db_path = tmp_path / "regime_shadow.db"
    sessions = _sessions(260)
    with open_shadow_db(db_path) as conn:
        _insert_success_runs(conn, sessions)
        conn.execute(
            "UPDATE runs SET status = 'failure', failure_reason = ? "
            "WHERE as_of_date = ?",
            ("forced interior classify failure", sessions[200].isoformat()),
        )
        conn.commit()
        result = evaluate_shadow_qualification(
            conn=conn,
            end_date=sessions[-1],
            engine_version=ENGINE_VERSION,
            config_version=CONFIG_VERSION,
        )

    assert result["qualifies"] is False
    assert result["current_consecutive_sessions"] == 59
    assert result["window_start"] == sessions[201].isoformat()
    assert "non_success_run" in result["blocking_reasons"]


def test_shadow_qualification_resets_after_breaking_incident(
    tmp_path: Path,
) -> None:
    from regime_detection.shadow_qualification import evaluate_shadow_qualification

    db_path = tmp_path / "regime_shadow.db"
    sessions = _sessions(252)
    with open_shadow_db(db_path) as conn:
        _insert_success_runs(conn, sessions)
        conn.execute(
            """
            INSERT INTO incidents (
                incident_date, description, resolution, breaks_qualification
            ) VALUES (?, ?, ?, ?)
            """,
            (sessions[100].isoformat(), "threshold changed", None, 1),
        )
        conn.commit()
        result = evaluate_shadow_qualification(
            conn=conn,
            end_date=sessions[-1],
            engine_version=ENGINE_VERSION,
            config_version=CONFIG_VERSION,
        )

    assert result["qualifies"] is False
    assert result["current_consecutive_sessions"] == 151
    assert result["window_start"] == sessions[101].isoformat()
    assert "qualification_breaking_incident" in result["blocking_reasons"]


def test_shadow_qualification_resets_after_off_session_breaking_incident(
    tmp_path: Path,
) -> None:
    from regime_detection.shadow_qualification import evaluate_shadow_qualification

    db_path = tmp_path / "regime_shadow.db"
    sessions = _sessions(252)
    sunday_after_window = date(2024, 2, 18)
    assert sessions[-1] < sunday_after_window
    with open_shadow_db(db_path) as conn:
        _insert_success_runs(conn, sessions)
        conn.execute(
            """
            INSERT INTO incidents (
                incident_date, description, resolution, breaks_qualification
            ) VALUES (?, ?, ?, ?)
            """,
            (sunday_after_window.isoformat(), "deadman detected missing run", None, 1),
        )
        conn.commit()
        result = evaluate_shadow_qualification(
            conn=conn,
            end_date=sunday_after_window,
            engine_version=ENGINE_VERSION,
            config_version=CONFIG_VERSION,
        )

    assert result["qualifies"] is False
    assert result["current_consecutive_sessions"] == 0
    assert "qualification_breaking_incident" in result["blocking_reasons"]


def test_shadow_qualification_resets_after_replay_mismatch(tmp_path: Path) -> None:
    from regime_detection.shadow_qualification import evaluate_shadow_qualification

    db_path = tmp_path / "regime_shadow.db"
    sessions = _sessions(252)
    with open_shadow_db(db_path) as conn:
        run_ids = _insert_success_runs(conn, sessions)
        conn.execute(
            """
            INSERT INTO replay_checks (
                check_timestamp, original_run_id, matches, diff
            ) VALUES (?, ?, ?, ?)
            """,
            ("2023-05-25T00:00:00+00:00", run_ids[100], 0, "{}"),
        )
        conn.commit()
        result = evaluate_shadow_qualification(
            conn=conn,
            end_date=sessions[-1],
            engine_version=ENGINE_VERSION,
            config_version=CONFIG_VERSION,
        )

    assert result["qualifies"] is False
    assert result["current_consecutive_sessions"] == 151
    assert result["window_start"] == sessions[101].isoformat()
    assert "replay_mismatch" in result["blocking_reasons"]
