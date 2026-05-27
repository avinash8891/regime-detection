from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pandas as pd
import yaml

from regime_detection.loaders import load_event_calendar
from regime_detection.shadow_storage import sha256_file


_REMOVED_LOCAL_STORAGE_HELPERS = (
    "RUNS_SCHEMA",
    "_utc_iso_now",
    "_sha256_file",
    "_ensure_layout",
    "_open_db",
    "_write_archived_inputs",
    "_insert_run_row",
    "_update_run_row_success",
    "_update_run_row_failure",
)


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_historical_walkforward.py"
    spec = importlib.util.spec_from_file_location(
        "run_historical_walkforward", script_path
    )
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

    market_path, events_path, checksums_path = mod.write_archived_inputs(
        archive_dir=tmp_path / "archive",
        market_slice=market_slice,
        event_df=event_df,
    )

    archived = yaml.safe_load(events_path.read_text(encoding="utf-8"))
    window_rows = [
        row for row in archived["events"] if row.get("window_days") == [-5, 10]
    ]
    assert window_rows
    assert market_path.exists()
    assert json.loads(checksums_path.read_text(encoding="utf-8")) == {
        "events.yaml": sha256_file(events_path),
        "market_data.parquet": sha256_file(market_path),
    }


def test_historical_walkforward_uses_public_shadow_storage_api() -> None:
    mod = _load_runner_module()
    source = inspect.getsource(mod)

    for helper_name in _REMOVED_LOCAL_STORAGE_HELPERS:
        assert not hasattr(mod, helper_name)
        assert f"def {helper_name}(" not in source
