from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "profile_engine_30d.py"
    spec = importlib.util.spec_from_file_location("profile_engine_30d", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile_engine_30d = _load_script_module()


def test_profile_engine_rejects_non_positive_lookback_days(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["profile_engine_30d.py", "--lookback-days", "0"])

    with pytest.raises(SystemExit) as exc_info:
        profile_engine_30d.main()

    assert exc_info.value.code == 2
