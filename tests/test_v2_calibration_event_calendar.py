from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_v2_calibration.py"
    spec = importlib.util.spec_from_file_location("run_v2_calibration", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_v2_calibration_requires_event_calendar_file(tmp_path: Path) -> None:
    mod = _load_runner_module()

    with pytest.raises(FileNotFoundError, match="event_calendar"):
        mod._load_required_event_calendar(tmp_path / "missing-us-events.yaml")
