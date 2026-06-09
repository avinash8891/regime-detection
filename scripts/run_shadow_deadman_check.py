#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from regime_detection.calendar import nyse_calendar
from regime_detection.shadow_storage import (
    ensure_shadow_layout,
    fetch_run_row,
    insert_incident,
    open_shadow_db,
)
from regime_detection.shadow_qualification import evaluate_shadow_qualification


def _previous_nyse_session(check_date: date) -> date:
    end = pd.Timestamp(check_date)
    start = end - pd.Timedelta(days=10)
    schedule = nyse_calendar().schedule(start_date=start.date(), end_date=end.date())
    sessions = schedule.index.date
    previous = [session for session in sessions if session < check_date]
    if not previous:
        raise ValueError(
            f"Unable to determine previous NYSE session before {check_date.isoformat()}"
        )
    return previous[-1]


def _interior_session_gaps(
    conn: Any,
    *,
    engine_version: str,
    config_version: str,
) -> list[date]:
    """F-017: NYSE sessions with no run row that fall *between* two recorded sessions.

    A contiguity break is a hole inside the recorded history — distinct from the
    natural left edge of a young shadow window (sessions predating the program),
    which is NOT a gap. We therefore only flag missing sessions strictly between the
    earliest and latest recorded session for this engine/config pair.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT as_of_date
        FROM runs
        WHERE engine_version = ? AND config_version = ?
        """,
        (engine_version, config_version),
    ).fetchall()
    recorded = sorted(date.fromisoformat(str(row[0])) for row in rows)
    if len(recorded) < 2:
        return []
    schedule = nyse_calendar().schedule(start_date=recorded[0], end_date=recorded[-1])
    expected = [session for session in schedule.index.date]
    recorded_set = set(recorded)
    return [session for session in expected if session not in recorded_set]


def _should_insert_blocking_incident(reason: str) -> bool:
    return reason not in {
        "qualification_breaking_incident",
        "insufficient_consecutive_sessions",
    }


def run_deadman_check(
    *,
    output_root: Path,
    check_date: date,
) -> dict[str, Any]:
    paths = ensure_shadow_layout(output_root)
    conn = open_shadow_db(paths["db"])
    try:
        expected_as_of_date = _previous_nyse_session(check_date)
        run_row = fetch_run_row(conn=conn, as_of_date=expected_as_of_date)
        if run_row is not None:
            qualification = evaluate_shadow_qualification(
                conn=conn,
                end_date=expected_as_of_date,
                engine_version=str(run_row["engine_version"]),
                config_version=str(run_row["config_version"]),
            )
            # F-017: the previous session ran, but an interior session may be missing
            # — a hole between two recorded sessions silently resets the consecutive-
            # clean window. Surface it as a non-ok status so main() exits nonzero
            # (the deadman fails loudly) instead of the previous unconditional "ok".
            interior_gaps = _interior_session_gaps(
                conn,
                engine_version=str(run_row["engine_version"]),
                config_version=str(run_row["config_version"]),
            )
            if interior_gaps:
                gap_list = ", ".join(session.isoformat() for session in interior_gaps)
                alert = (
                    "Shadow qualification window broken by interior "
                    f"missing_session_gap ({gap_list})"
                )
                insert_incident(
                    conn=conn,
                    incident_date=check_date,
                    description=alert,
                    resolution=None,
                    breaks_qualification=True,
                )
                return {
                    "status": "window_gap",
                    "check_date": check_date.isoformat(),
                    "expected_as_of_date": expected_as_of_date.isoformat(),
                    "alert": alert,
                    "qualification": qualification,
                }
            if not bool(qualification.get("qualifies")):
                blocking_reasons = qualification.get("blocking_reasons") or [
                    "qualification_blocked"
                ]
                reason = str(blocking_reasons[0])
                alert = (
                    "Shadow qualification window broken by "
                    f"{reason} for previous NYSE session "
                    f"{expected_as_of_date.isoformat()}"
                )
                if _should_insert_blocking_incident(reason):
                    insert_incident(
                        conn=conn,
                        incident_date=check_date,
                        description=alert,
                        resolution=None,
                        breaks_qualification=True,
                    )
                return {
                    "status": reason,
                    "check_date": check_date.isoformat(),
                    "expected_as_of_date": expected_as_of_date.isoformat(),
                    "alert": alert,
                    "qualification": qualification,
                }
            return {
                "status": "ok",
                "check_date": check_date.isoformat(),
                "expected_as_of_date": expected_as_of_date.isoformat(),
                "alert": None,
                "qualification": qualification,
            }

        alert = f"Missing shadow run for previous NYSE session {expected_as_of_date.isoformat()}"
        insert_incident(
            conn=conn,
            incident_date=check_date,
            description=alert,
            resolution=None,
            breaks_qualification=True,
        )
        return {
            "status": "alert",
            "check_date": check_date.isoformat(),
            "expected_as_of_date": expected_as_of_date.isoformat(),
            "alert": alert,
            "qualification": None,
        }
    finally:
        conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that the previous NYSE session produced a shadow run row."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--check-date", required=True, type=date.fromisoformat)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_deadman_check(
        output_root=args.output_root,
        check_date=args.check_date,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    # F-017: both a missing previous run ("alert") and an interior contiguity gap
    # ("window_gap") are deadman failures and must exit nonzero.
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
