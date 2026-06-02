from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
from datetime import date
from pathlib import Path
from contextlib import closing

import pytest


def _load_module(name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@pytest.fixture(scope="session")
def shadow_root_template(
    tmp_path_factory: pytest.TempPathFactory,
    v2_macro_parquet_path: Path,
) -> Path:
    runner = _load_module("run_shadow_regime", "scripts/run_shadow_regime.py")
    repo_root = Path(__file__).resolve().parents[1]
    market_data_path = (
        repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    )
    event_calendar_path = repo_root / "tests" / "fixtures" / "events" / "us_events.yaml"
    v2_daily_path = repo_root / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    config_path = repo_root / "tests" / "fixtures" / "configs" / "core3-v2-fast.yaml"
    out_root = tmp_path_factory.mktemp("shadow_replay_template")
    result = runner.run_shadow(
        as_of_date=date(2023, 12, 14),
        market_data_path=market_data_path,
        event_calendar_path=event_calendar_path,
        output_root=out_root,
        config_path=config_path,
        v2_daily_ohlcv_path=v2_daily_path,
        macro_parquet_path=v2_macro_parquet_path,
    )
    assert result["status"] == "success"
    return out_root


def _prepare_shadow_root(tmp_path: Path, template: Path) -> Path:
    out_root = tmp_path / "shadow_run"
    shutil.copytree(template, out_root)
    with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
        conn.execute(
            """
            UPDATE runs
            SET input_archive_path = ?, output_path = ?
            WHERE as_of_date = ?
            """,
            (
                str(out_root / "input_archives" / "2023-12-14"),
                str(out_root / "outputs" / "2023-12-14.json"),
                "2023-12-14",
            ),
        )
        conn.commit()
    return out_root


def test_shadow_replay_check_records_exact_match(
    tmp_path: Path, shadow_root_template: Path
) -> None:
    replay_mod = _load_module(
        "run_shadow_replay_check", "scripts/run_shadow_replay_check.py"
    )
    out_root = _prepare_shadow_root(tmp_path, shadow_root_template)
    config_path = (
        Path(__file__).resolve().parent / "fixtures" / "configs" / "core3-v2-fast.yaml"
    )

    result = replay_mod.run_replay_check(
        output_root=out_root,
        as_of_date=date(2023, 12, 14),
        config_path=config_path,
    )

    assert result["matches"] is True
    assert result["diff"] is None
    assert result["as_of_date"] == "2023-12-14"

    output_payload = json.loads((out_root / "outputs" / "2023-12-14.json").read_text())
    assert output_payload["v2_dependency_payload_contracts"]["network_fragility"] == {
        "breadth_state": "label_only",
        "credit_funding_effective": "label_only",
        "volatility_state": "label_only",
    }

    with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
        rows = conn.execute(
            "SELECT original_run_id, matches, diff FROM replay_checks"
        ).fetchall()
        run_row = conn.execute(
            "SELECT run_id FROM runs WHERE as_of_date = ?",
            ("2023-12-14",),
        ).fetchone()

    assert rows == [(run_row[0], 1, None)]


def test_shadow_replay_check_records_mismatch_with_diff(
    tmp_path: Path, shadow_root_template: Path
) -> None:
    replay_mod = _load_module(
        "run_shadow_replay_check", "scripts/run_shadow_replay_check.py"
    )
    out_root = _prepare_shadow_root(tmp_path, shadow_root_template)
    config_path = (
        Path(__file__).resolve().parent / "fixtures" / "configs" / "core3-v2-fast.yaml"
    )

    output_path = out_root / "outputs" / "2023-12-14.json"
    payload = json.loads(output_path.read_text())
    payload["transition_risk"]["state"] = "tampered_transition_risk"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = replay_mod.run_replay_check(
        output_root=out_root,
        as_of_date=date(2023, 12, 14),
        config_path=config_path,
    )

    assert result["matches"] is False
    assert result["diff"] is not None
    assert result["diff"]["transition_risk"]["state"]["stored"] == (
        "tampered_transition_risk"
    )

    with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
        row = conn.execute(
            "SELECT matches, diff FROM replay_checks ORDER BY check_id DESC LIMIT 1"
        ).fetchone()
        incident = conn.execute("""
            SELECT incident_date, description, breaks_qualification
            FROM incidents
            ORDER BY incident_id DESC
            LIMIT 1
            """).fetchone()

    assert row is not None
    assert row[0] == 0
    diff = json.loads(row[1])
    assert diff["transition_risk"]["state"]["replayed"] != (
        diff["transition_risk"]["state"]["stored"]
    )
    assert incident is not None
    assert incident[0] == "2023-12-14"
    assert "Replay mismatch" in incident[1]
    assert incident[2] == 1


def test_shadow_replay_contracts_only_drift_breaks_window_via_replay_mismatch(
    tmp_path: Path, shadow_root_template: Path
) -> None:
    # F-026: SR-029 — a v2_dependency_payload_contracts-ONLY drift (no state/label
    # change) must still report matches=False. Because the drifted key
    # (credit_funding_effective) is NOT a classification field, breaks_qualification
    # stays 0 — but evaluate_shadow_qualification still breaks the window through the
    # replay_mismatch path. This is the load-bearing, previously-untested behavior.
    from regime_detection.shadow_qualification import evaluate_shadow_qualification
    from regime_detection.shadow_storage import open_shadow_db

    replay_mod = _load_module(
        "run_shadow_replay_check", "scripts/run_shadow_replay_check.py"
    )
    out_root = _prepare_shadow_root(tmp_path, shadow_root_template)
    config_path = (
        Path(__file__).resolve().parent / "fixtures" / "configs" / "core3-v2-fast.yaml"
    )

    output_path = out_root / "outputs" / "2023-12-14.json"
    payload = json.loads(output_path.read_text())
    # Drift ONLY a non-classification contract value (does not end in _state/_label).
    payload["v2_dependency_payload_contracts"]["network_fragility"][
        "credit_funding_effective"
    ] = "tampered_contract_only"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = replay_mod.run_replay_check(
        output_root=out_root,
        as_of_date=date(2023, 12, 14),
        config_path=config_path,
    )

    assert result["matches"] is False
    assert result["diff"] is not None

    with closing(sqlite3.connect(out_root / "regime_shadow.db")) as conn:
        replay_row = conn.execute(
            "SELECT matches FROM replay_checks ORDER BY check_id DESC LIMIT 1"
        ).fetchone()
        incident = conn.execute(
            "SELECT breaks_qualification FROM incidents ORDER BY incident_id DESC "
            "LIMIT 1"
        ).fetchone()
        run = conn.execute(
            "SELECT engine_version, config_version FROM runs WHERE as_of_date = ?",
            ("2023-12-14",),
        ).fetchone()

    assert replay_row[0] == 0  # matches=False recorded
    assert incident[0] == 0  # contracts-only drift does NOT set breaks_qualification

    # ...yet the window IS broken via the replay_mismatch path.
    with open_shadow_db(out_root / "regime_shadow.db") as conn:
        qualification = evaluate_shadow_qualification(
            conn=conn,
            end_date=date(2023, 12, 14),
            engine_version=str(run[0]),
            config_version=str(run[1]),
        )
    assert qualification["qualifies"] is False
    assert "replay_mismatch" in qualification["blocking_reasons"]


def test_shadow_replay_classification_diff_detection_recurses_into_lists() -> None:
    replay_mod = _load_module(
        "run_shadow_replay_check", "scripts/run_shadow_replay_check.py"
    )

    diff = {
        "timeline": [
            {
                "transition_risk": {
                    "state": {
                        "replayed": "stable",
                        "stored": "transition_warning",
                    }
                }
            }
        ]
    }

    assert replay_mod._diff_touches_classification_fields(diff) is True


def test_shadow_replay_check_uses_only_archived_inputs(
    tmp_path: Path, shadow_root_template: Path
) -> None:
    replay_mod = _load_module(
        "run_shadow_replay_check", "scripts/run_shadow_replay_check.py"
    )
    out_root = _prepare_shadow_root(tmp_path, shadow_root_template)
    config_path = (
        Path(__file__).resolve().parent / "fixtures" / "configs" / "core3-v2-fast.yaml"
    )

    archive_dir = out_root / "input_archives" / "2023-12-14"
    market_archive = archive_dir / "market_data.parquet"
    events_archive = archive_dir / "events.yaml"
    macro_archive = archive_dir / "macro_series.parquet"
    original_read_parquet = replay_mod.pd.read_parquet
    original_load_archived_event_calendar = replay_mod.load_archived_event_calendar
    original_load_archived_macro_series = replay_mod.load_archived_macro_series

    seen = {"market": None, "events": None, "macro": None}

    def _checked_read_parquet(path, *args, **kwargs):
        seen["market"] = str(Path(path))
        return original_read_parquet(path, *args, **kwargs)

    def _checked_load_archived_event_calendar(path, *args, **kwargs):
        seen["events"] = str(Path(path))
        return original_load_archived_event_calendar(path, *args, **kwargs)

    def _checked_load_archived_macro_series(path, *args, **kwargs):
        seen["macro"] = str(Path(path))
        return original_load_archived_macro_series(path, *args, **kwargs)

    replay_mod.pd.read_parquet = _checked_read_parquet
    replay_mod.load_archived_event_calendar = _checked_load_archived_event_calendar
    replay_mod.load_archived_macro_series = _checked_load_archived_macro_series
    try:
        result = replay_mod.run_replay_check(
            output_root=out_root,
            as_of_date=date(2023, 12, 14),
            config_path=config_path,
        )
    finally:
        replay_mod.pd.read_parquet = original_read_parquet
        replay_mod.load_archived_event_calendar = original_load_archived_event_calendar
        replay_mod.load_archived_macro_series = original_load_archived_macro_series

    assert result["matches"] is True
    assert seen["market"] == str(market_archive)
    assert seen["events"] == str(events_archive)
    assert seen["macro"] == str(macro_archive)
