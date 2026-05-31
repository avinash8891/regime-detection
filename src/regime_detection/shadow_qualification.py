from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from typing import Any

import pandas as pd

from regime_detection.calendar import nyse_calendar

REQUIRED_SHADOW_SESSIONS = 252


def _expected_sessions(end_date: date, required_sessions: int) -> list[date]:
    end = pd.Timestamp(end_date)
    # Two calendar years comfortably covers 252 NYSE sessions plus gaps.
    start = end - pd.Timedelta(days=730)
    schedule = nyse_calendar().schedule(start_date=start.date(), end_date=end.date())
    return [session for session in schedule.index.date if session <= end_date]


def _run_status_by_date(
    conn: sqlite3.Connection, *, engine_version: str, config_version: str
) -> dict[date, str]:
    rows = conn.execute(
        """
        SELECT as_of_date, status
        FROM runs
        WHERE engine_version = ? AND config_version = ?
        """,
        (engine_version, config_version),
    ).fetchall()
    return {date.fromisoformat(str(row[0])): str(row[1]) for row in rows}


def _replay_mismatch_dates(
    conn: sqlite3.Connection, *, engine_version: str, config_version: str
) -> set[date]:
    rows = conn.execute(
        """
        SELECT runs.as_of_date
        FROM replay_checks
        JOIN runs ON runs.run_id = replay_checks.original_run_id
        WHERE runs.engine_version = ?
          AND runs.config_version = ?
          AND replay_checks.matches = 0
        """,
        (engine_version, config_version),
    ).fetchall()
    return {date.fromisoformat(str(row[0])) for row in rows}


def _breaking_incident_dates(conn: sqlite3.Connection) -> set[date]:
    rows = conn.execute("""
        SELECT incident_date
        FROM incidents
        WHERE breaks_qualification = 1
        """).fetchall()
    return {date.fromisoformat(str(row[0])) for row in rows}


def evaluate_shadow_qualification(
    *,
    conn: sqlite3.Connection,
    end_date: date,
    engine_version: str,
    config_version: str,
    required_sessions: int = REQUIRED_SHADOW_SESSIONS,
) -> dict[str, Any]:
    """Count the current consecutive clean shadow-qualification window."""

    sessions = _expected_sessions(end_date, required_sessions)
    if not sessions:
        return {
            "qualifies": False,
            "current_consecutive_sessions": 0,
            "required_sessions": required_sessions,
            "window_start": None,
            "window_end": None,
            "blocking_reasons": ["no_nyse_sessions"],
        }

    run_status = _run_status_by_date(
        conn, engine_version=engine_version, config_version=config_version
    )
    replay_mismatches = _replay_mismatch_dates(
        conn, engine_version=engine_version, config_version=config_version
    )
    breaking_incidents = _breaking_incident_dates(conn)

    count = 0
    window_start: date | None = None
    blocking_reasons: list[str] = []
    incident_window_end = end_date
    for session in reversed(sessions):
        if any(
            session <= incident <= incident_window_end
            for incident in breaking_incidents
        ):
            blocking_reasons.append("qualification_breaking_incident")
            break
        if session in replay_mismatches:
            blocking_reasons.append("replay_mismatch")
            break

        status = run_status.get(session)
        if status is None:
            blocking_reasons.append("missing_session_gap")
            break
        if status != "success":
            blocking_reasons.append("non_success_run")
            break

        count += 1
        window_start = session
        incident_window_end = session - timedelta(days=1)
        if count == required_sessions:
            break

    if (
        count < required_sessions
        and "insufficient_consecutive_sessions" not in blocking_reasons
    ):
        blocking_reasons.append("insufficient_consecutive_sessions")

    qualifies = count >= required_sessions
    return {
        "qualifies": qualifies,
        "current_consecutive_sessions": count,
        "required_sessions": required_sessions,
        "window_start": None if window_start is None else window_start.isoformat(),
        "window_end": None if count == 0 else sessions[-1].isoformat(),
        "blocking_reasons": [] if qualifies else blocking_reasons,
    }
