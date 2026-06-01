from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
from datetime import date
from pathlib import Path

import pytest


def _load(name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, repo_root / rel_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_walkforward_replay_matches_then_detects_corruption(
    tmp_path: Path, v2_macro_parquet_path: Path
) -> None:
    """F-001: run_walkforward_replay_check recomputes every successful date from the
    archived inputs (incl. the v2_daily slice) over regime_walkforward.db, agrees
    with the stored outputs, and flags a corrupted output as a qualification break."""
    repo_root = Path(__file__).resolve().parents[1]
    v2_daily_path = repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    config_path = repo_root / "tests" / "fixtures" / "configs" / "core3-v2-fast.yaml"
    out_root = tmp_path / "walkforward"

    runner = _load(
        "run_historical_walkforward", "scripts/run_historical_walkforward.py"
    )
    result = runner.run_walkforward(
        market_data_path=v2_daily_path,
        output_root=out_root,
        start_date=date(2026, 5, 12),
        end_date=date(2026, 5, 13),
        event_calendar_path=event_calendar_path,
        config_path=config_path,
        v2_daily_ohlcv_path=v2_daily_path,
        macro_parquet_path=v2_macro_parquet_path,
    )
    assert result["success_count"] == 2

    replay = _load(
        "run_walkforward_replay_check", "scripts/run_walkforward_replay_check.py"
    )
    verdict = replay.run_walkforward_replay_check(
        output_root=out_root, config_path=config_path
    )

    assert verdict["all_passed"] is True
    assert {r["as_of_date"] for r in verdict["results"]} == {"2026-05-12", "2026-05-13"}
    assert all(r["matches"] for r in verdict["results"])
    assert verdict["engine_version"].startswith("regime-engine-v")
    from regime_detection.config import load_regime_config

    assert verdict["config_version"] == load_regime_config(config_path).config_version

    # CR-006: snapshot the uncorrupted batch to verify below that it replays from its own
    # output_root even after the original (whose absolute paths the DB recorded) is gone.
    relocated = tmp_path / "relocated"
    shutil.copytree(out_root, relocated)

    # Corrupt one stored output → replay must flag a classification mismatch.
    corrupt_path = out_root / "outputs" / "2026-05-13.json"
    payload = json.loads(corrupt_path.read_text())
    payload["trend_direction"]["active_label"] = "__corrupted__"
    corrupt_path.write_text(json.dumps(payload, indent=2))

    verdict2 = replay.run_walkforward_replay_check(
        output_root=out_root, config_path=config_path
    )
    assert verdict2["all_passed"] is False
    corrupted = next(r for r in verdict2["results"] if r["as_of_date"] == "2026-05-13")
    assert corrupted["matches"] is False
    assert corrupted["breaks_qualification"] is True
    # the untouched date still replays cleanly
    clean = next(r for r in verdict2["results"] if r["as_of_date"] == "2026-05-12")
    assert clean["matches"] is True

    # CR-003: the batch archived its FROZEN config; the replay drives the engine from it.
    assert (relocated / "frozen_config.yaml").read_text() == config_path.read_text()
    # CR-006 + CR-003: delete the original (its DB-recorded absolute paths are now dead),
    # then the relocated copy must still replay cleanly WITHOUT an operator-supplied
    # config_path — paths resolve from output_root's layout and the engine from the
    # archived frozen config.
    shutil.rmtree(out_root)
    relocated_verdict = replay.run_walkforward_replay_check(output_root=relocated)
    assert relocated_verdict["all_passed"] is True
    assert all(r["matches"] for r in relocated_verdict["results"])


def _write_minimal_runs_db(
    out_root: Path, *, input_archive_path: str | None, output_path: str | None
) -> None:
    """A raw regime_walkforward.db with a single success row (no NOT NULL constraints,
    so either archive path can be NULL to exercise the per-run guards)."""
    out_root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(out_root / "regime_walkforward.db") as conn:
        conn.execute(
            "CREATE TABLE runs (as_of_date TEXT, status TEXT, engine_version TEXT, "
            "config_version TEXT, input_archive_path TEXT, output_path TEXT)"
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            (
                "2026-05-13",
                "success",
                "regime-engine-vtest",
                "core3-test",
                input_archive_path,
                output_path,
            ),
        )


@pytest.mark.parametrize("null_field", ["input_archive_path", "output_path"])
def test_replay_raises_clean_error_on_null_archive_path(
    tmp_path: Path, null_field: str
) -> None:
    """CR-007 / CR-011: a success row with a NULL archive/output path fails closed with a
    clean ValueError naming the date, instead of an opaque TypeError from Path(None) that
    aborts the whole batch before any replay_verification.json is written."""
    fields: dict[str, str | None] = {
        "input_archive_path": "input_archives/2026-05-13",
        "output_path": "outputs/2026-05-13.json",
    }
    fields[null_field] = None
    out_root = tmp_path / "wf"
    _write_minimal_runs_db(out_root, **fields)

    replay = _load(
        "run_walkforward_replay_check", "scripts/run_walkforward_replay_check.py"
    )
    with pytest.raises(ValueError, match=f"has no {null_field}"):
        replay.run_walkforward_replay_check(output_root=out_root)
