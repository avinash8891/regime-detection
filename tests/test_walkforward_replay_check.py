from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path


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
