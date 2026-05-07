from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import date
from pathlib import Path


def _load_module(name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _prepare_shadow_root(tmp_path: Path) -> Path:
    runner = _load_module("run_shadow_regime", "scripts/run_shadow_regime.py")
    repo_root = Path(__file__).resolve().parents[1]
    market_data_path = repo_root / "tests" / "fixtures" / "raw" / "market_data.parquet"
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    out_root = tmp_path / "shadow_run"
    result = runner.run_shadow(
        as_of_date=date(2023, 12, 14),
        market_data_path=market_data_path,
        event_calendar_path=event_calendar_path,
        output_root=out_root,
    )
    assert result["status"] == "success"
    return out_root


def test_shadow_replay_check_records_exact_match(tmp_path: Path) -> None:
    replay_mod = _load_module("run_shadow_replay_check", "scripts/run_shadow_replay_check.py")
    out_root = _prepare_shadow_root(tmp_path)

    result = replay_mod.run_replay_check(
        output_root=out_root,
        as_of_date=date(2023, 12, 14),
    )

    assert result["matches"] is True
    assert result["diff"] is None
    assert result["as_of_date"] == "2023-12-14"

    with sqlite3.connect(out_root / "regime_shadow.db") as conn:
        rows = conn.execute(
            "SELECT original_run_id, matches, diff FROM replay_checks"
        ).fetchall()
        run_row = conn.execute(
            "SELECT run_id FROM runs WHERE as_of_date = ?",
            ("2023-12-14",),
        ).fetchone()

    assert rows == [(run_row[0], 1, None)]


def test_shadow_replay_check_records_mismatch_with_diff(tmp_path: Path) -> None:
    replay_mod = _load_module("run_shadow_replay_check", "scripts/run_shadow_replay_check.py")
    out_root = _prepare_shadow_root(tmp_path)

    output_path = out_root / "outputs" / "2023-12-14.json"
    payload = json.loads(output_path.read_text())
    payload["transition_risk_label"] = "tampered_transition_risk"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_mod.run_replay_check(
        output_root=out_root,
        as_of_date=date(2023, 12, 14),
    )

    assert result["matches"] is False
    assert result["diff"] is not None
    assert result["diff"]["transition_risk_label"]["stored"] == "tampered_transition_risk"

    with sqlite3.connect(out_root / "regime_shadow.db") as conn:
        row = conn.execute(
            "SELECT matches, diff FROM replay_checks ORDER BY check_id DESC LIMIT 1"
        ).fetchone()

    assert row is not None
    assert row[0] == 0
    diff = json.loads(row[1])
    assert diff["transition_risk_label"]["replayed"] != diff["transition_risk_label"]["stored"]


def test_shadow_replay_check_uses_only_archived_inputs(tmp_path: Path) -> None:
    replay_mod = _load_module("run_shadow_replay_check", "scripts/run_shadow_replay_check.py")
    out_root = _prepare_shadow_root(tmp_path)

    archive_dir = out_root / "input_archives" / "2023-12-14"
    market_archive = archive_dir / "market_data.parquet"
    events_archive = archive_dir / "events.yaml"
    original_read_parquet = replay_mod.pd.read_parquet
    original_load_archived_event_calendar = replay_mod.load_archived_event_calendar

    seen = {"market": None, "events": None}

    def _checked_read_parquet(path, *args, **kwargs):
        seen["market"] = str(Path(path))
        return original_read_parquet(path, *args, **kwargs)

    def _checked_load_archived_event_calendar(path, *args, **kwargs):
        seen["events"] = str(Path(path))
        return original_load_archived_event_calendar(path, *args, **kwargs)

    replay_mod.pd.read_parquet = _checked_read_parquet
    replay_mod.load_archived_event_calendar = _checked_load_archived_event_calendar
    try:
        result = replay_mod.run_replay_check(
            output_root=out_root,
            as_of_date=date(2023, 12, 14),
        )
    finally:
        replay_mod.pd.read_parquet = original_read_parquet
        replay_mod.load_archived_event_calendar = original_load_archived_event_calendar

    assert result["matches"] is True
    assert seen["market"] == str(market_archive)
    assert seen["events"] == str(events_archive)
