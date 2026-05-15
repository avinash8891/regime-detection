from __future__ import annotations

import pandas as pd

import regime_detection.axis_series as axis_series


def test_axis_series_removes_unused_protocol_and_scalar_staleness_helper() -> None:
    assert not hasattr(axis_series, "AxisSeriesClassifier")
    assert not hasattr(axis_series, "_calendar_staleness_days")


def test_axis_series_staleness_helpers_use_named_sentinel() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")

    calendar = axis_series._calendar_staleness_days_series(None, idx)
    trading = axis_series._trading_staleness_series(None, idx)

    assert axis_series._STALENESS_SENTINEL == 10**9
    assert calendar.tolist() == [axis_series._STALENESS_SENTINEL] * 3
    assert trading.tolist() == [axis_series._STALENESS_SENTINEL] * 3
