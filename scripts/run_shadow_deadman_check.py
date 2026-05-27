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


def _previous_nyse_session(check_date: date) -> date:
    end = pd.Timestamp(check_date)
    start = end - pd.Timedelta(days=10)
    calendar: Any = nyse_calendar()
    schedule = calendar.schedule(start_date=start.date(), end_date=end.date())
    schedule_index = pd.DatetimeIndex(pd.Index(schedule.index))
    sessions = [pd.Timestamp(value).date() for value in schedule_index.tolist()]
    previous = [session for session in sessions if session < check_date]
    if not previous:
        raise ValueError(
            f"Unable to determine previous NYSE session before {check_date.isoformat()}"
        )
    return previous[-1]


def run_deadman_check(
    *,
    output_root: Path,
    check_date: date,
) -> dict[str, str | None]:
    paths = ensure_shadow_layout(output_root)
    conn = open_shadow_db(paths["db"])
    try:
        expected_as_of_date = _previous_nyse_session(check_date)
        run_row = fetch_run_row(conn=conn, as_of_date=expected_as_of_date)
        if run_row is not None:
            return {
                "status": "ok",
                "check_date": check_date.isoformat(),
                "expected_as_of_date": expected_as_of_date.isoformat(),
                "alert": None,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
