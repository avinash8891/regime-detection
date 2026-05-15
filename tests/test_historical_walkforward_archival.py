from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml

from regime_detection.loaders import load_event_calendar


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_historical_walkforward.py"
    spec = importlib.util.spec_from_file_location("run_historical_walkforward", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_walkforward_archives_production_event_window_days(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_runner_module()
    event_df = load_event_calendar(repo_root / "configs" / "events" / "us_events.yaml")
    market_slice = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-05").date(),
                "symbol": "SPY",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000,
            }
        ]
    )

    market_path, events_path, checksums_path = mod._write_archived_inputs(
        archive_dir=tmp_path / "archive",
        market_slice=market_slice,
        event_df=event_df,
    )

    archived = yaml.safe_load(events_path.read_text(encoding="utf-8"))
    window_rows = [row for row in archived["events"] if row.get("window_days") == [-5, 10]]
    assert window_rows
    assert market_path.exists()
    assert json.loads(checksums_path.read_text(encoding="utf-8")) == {
        "events.yaml": mod._sha256_file(events_path),
        "market_data.parquet": mod._sha256_file(market_path),
    }
