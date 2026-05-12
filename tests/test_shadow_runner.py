from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_shadow_regime.py"
    spec = importlib.util.spec_from_file_location("run_shadow_regime", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_shadow_runner_writes_expected_artifacts_and_ledger(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    market_data_path = repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    out_root = tmp_path / "shadow_run"

    mod = _load_runner_module()
    result = mod.run_shadow(
        as_of_date=date(2023, 12, 14),
        market_data_path=market_data_path,
        event_calendar_path=event_calendar_path,
        output_root=out_root,
    )

    assert result["status"] == "success"
    assert result["as_of_date"] == "2023-12-14"

    db_path = out_root / "regime_shadow.db"
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT as_of_date, status, output_path, input_archive_path FROM runs"
        ).fetchall()
        replay_tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('replay_checks', 'incidents') ORDER BY name"
        ).fetchall()

    assert rows == [
        (
            "2023-12-14",
            "success",
            str(out_root / "outputs" / "2023-12-14.json"),
            str(out_root / "input_archives" / "2023-12-14"),
        )
    ]
    assert replay_tables == [("incidents",), ("replay_checks",)]

    output_path = out_root / "outputs" / "2023-12-14.json"
    payload = json.loads(output_path.read_text())
    assert payload["as_of_date"] == "2023-12-14"
    assert payload["engine_version"].startswith("regime-engine-v")
    from regime_detection.config import load_default_regime_config

    assert payload["config_version"] == load_default_regime_config().config_version

    archive_market = out_root / "input_archives" / "2023-12-14" / "market_data.parquet"
    archived_df = pd.read_parquet(archive_market)
    archived_df["date"] = pd.to_datetime(archived_df["date"]).dt.date
    assert archived_df["date"].max() == date(2023, 12, 14)

    checksums_path = out_root / "input_archives" / "2023-12-14" / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    assert "market_data.parquet" in checksums
    assert "events.yaml" in checksums


def test_shadow_runner_archives_inputs_and_inserts_in_progress_before_classify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    market_data_path = repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    out_root = tmp_path / "shadow_run"

    mod = _load_runner_module()
    real_classify = mod.RegimeEngine.classify

    def _checked_classify(self, *, as_of_date, market_data, event_calendar, **kwargs):
        archive_dir = out_root / "input_archives" / "2023-12-14"
        assert (archive_dir / "market_data.parquet").exists()
        assert (archive_dir / "events.yaml").exists()
        assert (archive_dir / "checksums.json").exists()
        with sqlite3.connect(out_root / "regime_shadow.db") as conn:
            row = conn.execute(
                "SELECT status, input_archive_path FROM runs WHERE as_of_date = ?",
                ("2023-12-14",),
            ).fetchone()
        assert row == ("in_progress", str(archive_dir))
        return real_classify(
            self,
            as_of_date=as_of_date,
            market_data=market_data,
            event_calendar=event_calendar,
            **kwargs,
        )

    monkeypatch.setattr(mod.RegimeEngine, "classify", _checked_classify)

    result = mod.run_shadow(
        as_of_date=date(2023, 12, 14),
        market_data_path=market_data_path,
        event_calendar_path=event_calendar_path,
        output_root=out_root,
    )

    assert result["status"] == "success"


def test_shadow_runner_records_failures_without_silent_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    market_data_path = repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    out_root = tmp_path / "shadow_run"

    mod = _load_runner_module()

    def _boom(self, **kwargs):
        raise RuntimeError("forced classify failure")

    monkeypatch.setattr(mod.RegimeEngine, "classify", _boom)

    result = mod.run_shadow(
        as_of_date=date(2023, 12, 14),
        market_data_path=market_data_path,
        event_calendar_path=event_calendar_path,
        output_root=out_root,
    )

    assert result["status"] == "failure"
    assert "forced classify failure" in result["failure_reason"]

    with sqlite3.connect(out_root / "regime_shadow.db") as conn:
        rows = conn.execute(
            "SELECT as_of_date, status, failure_reason FROM runs"
        ).fetchall()

    assert rows == [("2023-12-14", "failure", "forced classify failure")]
    assert (out_root / "input_archives" / "2023-12-14" / "market_data.parquet").exists()
    assert not (out_root / "outputs" / "2023-12-14.json").exists()


def test_shadow_runner_rejects_duplicate_versioned_run_and_non_trading_day(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    market_data_path = repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    out_root = tmp_path / "shadow_run"

    mod = _load_runner_module()
    first = mod.run_shadow(
        as_of_date=date(2023, 12, 14),
        market_data_path=market_data_path,
        output_root=out_root,
    )
    assert first["status"] == "success"

    with pytest.raises(sqlite3.IntegrityError):
        mod.run_shadow(
            as_of_date=date(2023, 12, 14),
            market_data_path=market_data_path,
            output_root=out_root,
        )

    with pytest.raises(ValueError, match="NYSE trading day"):
        mod.run_shadow(
            as_of_date=date(2023, 12, 16),
            market_data_path=market_data_path,
            output_root=tmp_path / "weekend_shadow",
        )
