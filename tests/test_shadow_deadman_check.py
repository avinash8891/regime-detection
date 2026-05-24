from __future__ import annotations

import importlib.util
import sqlite3
from datetime import date
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def _load_module(name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _run_shadow(out_root: Path, as_of_date: date) -> None:
    runner = _load_module("run_shadow_regime", "scripts/run_shadow_regime.py")
    repo_root = Path(__file__).resolve().parents[1]
    market_data_path = repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    v2_daily_path = repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    config_path = repo_root / "tests" / "fixtures" / "configs" / "core3-v2-fast.yaml"
    result = runner.run_shadow(
        as_of_date=as_of_date,
        market_data_path=market_data_path,
        event_calendar_path=event_calendar_path,
        output_root=out_root,
        config_path=config_path,
        v2_daily_ohlcv_path=v2_daily_path,
    )
    assert result["status"] == "success"


def test_deadman_check_passes_when_previous_session_has_run(tmp_path: Path) -> None:
    monitor = _load_module("run_shadow_deadman_check", "scripts/run_shadow_deadman_check.py")
    out_root = tmp_path / "shadow_run"
    _run_shadow(out_root, date(2023, 12, 14))

    result = monitor.run_deadman_check(
        output_root=out_root,
        check_date=date(2023, 12, 15),
    )

    assert result["status"] == "ok"
    assert result["expected_as_of_date"] == "2023-12-14"
    assert result["alert"] is None

    with sqlite3.connect(out_root / "regime_shadow.db") as conn:
        incidents = conn.execute("SELECT incident_date, description FROM incidents").fetchall()
    assert incidents == []


def test_deadman_check_alerts_and_records_incident_when_previous_session_missing(tmp_path: Path) -> None:
    monitor = _load_module("run_shadow_deadman_check", "scripts/run_shadow_deadman_check.py")
    out_root = tmp_path / "shadow_run"

    result = monitor.run_deadman_check(
        output_root=out_root,
        check_date=date(2023, 12, 15),
    )

    assert result["status"] == "alert"
    assert result["expected_as_of_date"] == "2023-12-14"
    assert "Missing shadow run" in result["alert"]

    with sqlite3.connect(out_root / "regime_shadow.db") as conn:
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


def test_deadman_check_uses_previous_friday_for_weekend_check(tmp_path: Path) -> None:
    monitor = _load_module("run_shadow_deadman_check", "scripts/run_shadow_deadman_check.py")
    out_root = tmp_path / "shadow_run"
    _run_shadow(out_root, date(2023, 12, 15))

    result = monitor.run_deadman_check(
        output_root=out_root,
        check_date=date(2023, 12, 17),
    )

    assert result["status"] == "ok"
    assert result["expected_as_of_date"] == "2023-12-15"
