from __future__ import annotations

import importlib.util
import sqlite3
from datetime import date
from pathlib import Path
from contextlib import closing


def _load_module(name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _record_shadow_run(out_root: Path, as_of_date: date) -> None:
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
                "success",
                None,
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

    assert result["status"] == "ok"
    assert result["expected_as_of_date"] == "2023-12-14"
    assert result["alert"] is None

    with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
        incidents = conn.execute(
            "SELECT incident_date, description FROM incidents"
        ).fetchall()
    assert incidents == []


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

    assert result["status"] == "ok"
    assert result["expected_as_of_date"] == "2023-12-15"
