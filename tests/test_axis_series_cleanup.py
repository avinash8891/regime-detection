from __future__ import annotations

import pandas as pd

import regime_detection.axis_series as axis_series
import regime_detection._staleness_utils as _staleness_utils


def test_axis_series_removes_unused_protocol_and_scalar_staleness_helper() -> None:
    assert not hasattr(axis_series, "AxisSeriesClassifier")
    assert not hasattr(axis_series, "_calendar_staleness_days")


def test_axis_series_staleness_helpers_moved_to_staleness_utils() -> None:
    """Staleness helpers are now in _staleness_utils, not axis_series (F001 refactor)."""
    idx = pd.date_range("2024-01-02", periods=3, freq="B")

    calendar = _staleness_utils.calendar_staleness_days_series(None, idx)
    trading = _staleness_utils.trading_staleness_series(None, idx)

    assert _staleness_utils._STALENESS_SENTINEL == 10**9
    assert calendar.tolist() == [_staleness_utils._STALENESS_SENTINEL] * 3
    assert trading.tolist() == [_staleness_utils._STALENESS_SENTINEL] * 3
