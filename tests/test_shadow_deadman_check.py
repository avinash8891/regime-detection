from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from contextlib import closing

from regime_detection.calendar import nyse_calendar


def _load_module(name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _record_shadow_run(
    out_root: Path, as_of_date: date, *, status: str = "success"
) -> None:
    storage = _load_module("shadow_storage", "src/regime_detection/shadow_storage.py")
    paths = storage.ensure_shadow_layout(out_root)
    with storage.open_shadow_db(paths["db"]) as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_timestamp, as_of_date, engine_version, config_version,
                status, failure_reason, input_archive_path, output_path,
                output_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-05-28T00:00:00+00:00",
                as_of_date.isoformat(),
                "regime-engine-vtest",
                "core3-test",
                status,
                None if status == "success" else f"{status} fixture",
                str(out_root / "input_archives" / as_of_date.isoformat()),
                str(out_root / "outputs" / f"{as_of_date.isoformat()}.json"),
                None,
            ),
        )
        conn.commit()


def test_deadman_check_passes_when_previous_session_has_run(
    tmp_path: Path,
) -> None:
    monitor = _load_module(
        "run_shadow_deadman_check", "scripts/run_shadow_deadman_check.py"
    )
    out_root = tmp_path / "shadow_run"
    _record_shadow_run(out_root, date(2023, 12, 14))

    result = monitor.run_deadman_check(
        output_root=out_root,
        check_date=date(2023, 12, 15),
    )

    assert result["status"] == "missing_session_gap"
    assert result["expected_as_of_date"] == "2023-12-14"
    assert "missing_session_gap" in result["alert"]

    with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
        incidents = conn.execute(
            "SELECT incident_date, description FROM incidents"
        ).fetchall()
    assert incidents == [
        (
            "2023-12-15",
            "Shadow qualification window broken by missing_session_gap for previous NYSE session 2023-12-14",
        )
    ]


def test_deadman_check_fails_when_previous_session_run_is_not_success(
    tmp_path: Path, monkeypatch
) -> None:
    monitor = _load_module(
        "run_shadow_deadman_check", "scripts/run_shadow_deadman_check.py"
    )
    out_root = tmp_path / "shadow_run"
    _record_shadow_run(out_root, date(2023, 12, 14), status="in_progress")

    result = monitor.run_deadman_check(
        output_root=out_root,
        check_date=date(2023, 12, 15),
    )

    assert result["status"] == "non_success_run"
    assert result["qualification"]["qualifies"] is False
    assert "non_success_run" in result["qualification"]["blocking_reasons"]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_shadow_deadman_check.py",
            "--output-root",
            str(out_root),
            "--check-date",
            "2023-12-15",
        ],
    )
    assert monitor.main() == 1


def test_deadman_check_alerts_and_records_incident_when_previous_session_missing(
    tmp_path: Path,
) -> None:
    monitor = _load_module(
        "run_shadow_deadman_check", "scripts/run_shadow_deadman_check.py"
    )
    out_root = tmp_path / "shadow_run"

    result = monitor.run_deadman_check(
        output_root=out_root,
        check_date=date(2023, 12, 15),
    )

    assert result["status"] == "alert"
    assert result["expected_as_of_date"] == "2023-12-14"
    assert "Missing shadow run" in result["alert"]

    with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
        incidents = conn.execute(
            "SELECT incident_date, description, breaks_qualification FROM incidents"
        ).fetchall()
    assert incidents == [
        (
            "2023-12-15",
            "Missing shadow run for previous NYSE session 2023-12-14",
            1,
        )
    ]


def test_deadman_main_returns_nonzero_on_alert(tmp_path: Path, monkeypatch) -> None:
    monitor = _load_module(
        "run_shadow_deadman_check", "scripts/run_shadow_deadman_check.py"
    )
    out_root = tmp_path / "shadow_run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_shadow_deadman_check.py",
            "--output-root",
            str(out_root),
            "--check-date",
            "2023-12-15",
        ],
    )

    assert monitor.main() == 1


def test_deadman_check_flags_interior_gap_and_exits_nonzero(
    tmp_path: Path, monkeypatch
) -> None:
    # F-017: the previous NYSE session ran, but an interior session is missing, so
    # the consecutive-clean qualification window is broken. The deadman must NOT
    # report "ok" — it returns window_gap, records a breaking incident, and main()
    # exits nonzero so the scheduled job fails loudly.
    monitor = _load_module(
        "run_shadow_deadman_check", "scripts/run_shadow_deadman_check.py"
    )
    out_root = tmp_path / "shadow_run"
    schedule = nyse_calendar().schedule(
        start_date=date(2023, 11, 1),
        end_date=date(2023, 12, 15),
    )
    sessions = list(schedule.index.date)
    interior_gap = sessions[-3]
    for session in sessions:
        if session == interior_gap:
            continue
        _record_shadow_run(out_root, session)

    check_date = sessions[-1] + timedelta(days=1)
    result = monitor.run_deadman_check(output_root=out_root, check_date=check_date)

    assert result["status"] == "window_gap"
    assert result["expected_as_of_date"] == sessions[-1].isoformat()
    assert "missing_session_gap" in result["alert"]
    assert result["qualification"]["qualifies"] is False
    assert "missing_session_gap" in result["qualification"]["blocking_reasons"]

    with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
        incidents = conn.execute(
            "SELECT incident_date, breaks_qualification FROM incidents"
        ).fetchall()
    assert incidents == [(check_date.isoformat(), 1)]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_shadow_deadman_check.py",
            "--output-root",
            str(out_root),
            "--check-date",
            check_date.isoformat(),
        ],
    )
    assert monitor.main() == 1


def test_deadman_check_does_not_insert_incident_for_existing_incident_block(
    tmp_path: Path,
) -> None:
    monitor = _load_module(
        "run_shadow_deadman_check", "scripts/run_shadow_deadman_check.py"
    )
    storage = _load_module("shadow_storage", "src/regime_detection/shadow_storage.py")
    out_root = tmp_path / "shadow_run"
    schedule = nyse_calendar().schedule(
        start_date=date(2023, 1, 3),
        end_date=date(2024, 3, 31),
    )
    sessions = list(schedule.index.date)[:252]
    for session in sessions:
        _record_shadow_run(out_root, session)

    paths = storage.ensure_shadow_layout(out_root)
    with storage.open_shadow_db(paths["db"]) as conn:
        storage.insert_incident(
            conn=conn,
            incident_date=sessions[-2],
            description="pre-existing qualification breaker",
            resolution=None,
            breaks_qualification=True,
        )

    result = monitor.run_deadman_check(
        output_root=out_root,
        check_date=sessions[-1] + timedelta(days=1),
    )

    assert result["status"] == "qualification_breaking_incident"
    with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
        incidents = conn.execute("SELECT description FROM incidents").fetchall()
    assert incidents == [("pre-existing qualification breaker",)]


def test_deadman_check_uses_previous_friday_for_weekend_check(
    tmp_path: Path,
) -> None:
    monitor = _load_module(
        "run_shadow_deadman_check", "scripts/run_shadow_deadman_check.py"
    )
    out_root = tmp_path / "shadow_run"
    _record_shadow_run(out_root, date(2023, 12, 15))

    result = monitor.run_deadman_check(
        output_root=out_root,
        check_date=date(2023, 12, 17),
    )

    assert result["status"] == "missing_session_gap"
    assert result["expected_as_of_date"] == "2023-12-15"


def test_deadman_check_reports_shadow_qualification_counter(
    tmp_path: Path,
) -> None:
    monitor = _load_module(
        "run_shadow_deadman_check", "scripts/run_shadow_deadman_check.py"
    )
    out_root = tmp_path / "shadow_run"
    schedule = nyse_calendar().schedule(
        start_date=date(2023, 1, 3),
        end_date=date(2024, 3, 31),
    )
    sessions = list(schedule.index.date)[:252]
    assert len(sessions) == 252
    for session in sessions:
        _record_shadow_run(out_root, session)

    result = monitor.run_deadman_check(
        output_root=out_root,
        check_date=sessions[-1] + timedelta(days=1),
    )

    assert result["status"] == "ok"
    assert result["qualification"]["qualifies"] is True
    assert result["qualification"]["current_consecutive_sessions"] == 252
