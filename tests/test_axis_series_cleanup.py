from __future__ import annotations

import importlib.util
import inspect

import pandas as pd

import regime_detection.axis_series as axis_series


_MOVED_AXIS_BUILDER_NAMES = (
    "build_trend_direction_axis_series",
    "build_trend_character_axis_series",
    "build_volatility_axis_series",
    "build_breadth_axis_series",
    "build_network_fragility_axis_series",
    "build_volume_liquidity_axis_series",
    "build_credit_funding_axis_series",
    "build_credit_funding_proxy_axis_series",
    "build_inflation_growth_axis_series",
    "build_monetary_pressure_axis_series",
)


def test_axis_series_removes_unused_protocol_and_scalar_staleness_helper() -> None:
    assert not hasattr(axis_series, "AxisSeriesClassifier")
    assert not hasattr(axis_series, "_calendar_staleness_days")


def test_axis_series_reexports_moved_builders_without_local_bodies() -> None:
    axis_series_source = inspect.getsource(axis_series)

    for builder_name in _MOVED_AXIS_BUILDER_NAMES:
        builder = getattr(axis_series, builder_name)
        assert f"def {builder_name}(" not in axis_series_source
        assert builder.__module__.startswith("regime_detection.axis_builders.")
        assert builder.__module__ != "regime_detection.axis_builders.series"


def test_axis_builder_series_shim_is_removed() -> None:
    assert importlib.util.find_spec("regime_detection.axis_builders.series") is None


def test_axis_series_staleness_helpers_use_named_sentinel() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")

    calendar = axis_series._calendar_staleness_days_series(None, idx)
    trading = axis_series._trading_staleness_series(None, idx)

    assert axis_series._STALENESS_SENTINEL == 10**9
    assert calendar.tolist() == [axis_series._STALENESS_SENTINEL] * 3
    assert trading.tolist() == [axis_series._STALENESS_SENTINEL] * 3
