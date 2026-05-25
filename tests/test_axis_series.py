from __future__ import annotations

from dataclasses import fields
from datetime import date

import pandas as pd

from regime_detection.axis_series import (
    AXIS_BUILD_ORDER,
    AXIS_DEPENDENCIES,
    AxisSeriesBundle,
    _build_axis_outputs,
    _validate_axis_dependency_order,
    build_axis_series_bundle,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context


def test_axis_series_bundle_contract_names_every_timeline_axis() -> None:
    bundle_fields = {field.name for field in fields(AxisSeriesBundle)}

    assert bundle_fields == {
        "trend_direction",
        "trend_character",
        "volatility_state",
        "breadth_state",
        "event_calendar",
        "network_fragility",
        "volume_liquidity_state",
        "credit_funding",
        "credit_funding_proxy",
        "credit_funding_effective",
        "monetary_pressure_state",
        "inflation_growth",
    }


def test_axis_build_order_satisfies_declared_dependencies() -> None:
    assert AXIS_DEPENDENCIES["network_fragility"] == (
        "breadth_state",
        "volatility_state",
        "credit_funding_effective",
    )
    assert AXIS_DEPENDENCIES["inflation_growth"] == ("credit_funding_effective",)
    _validate_axis_dependency_order(AXIS_BUILD_ORDER, AXIS_DEPENDENCIES)


def test_build_axis_series_bundle_emits_session_keyed_outputs_for_core_axes(
    market_df_for_asof,
) -> None:
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=RegimeEngine().config,
    )
    feature_store = build_feature_store(context)

    bundle = build_axis_series_bundle(context=context, feature_store=feature_store)
    expected_dates = set(context.sessions)

    assert set(bundle.trend_direction.outputs_by_date) == expected_dates
    assert set(bundle.trend_character.outputs_by_date) == expected_dates
    assert set(bundle.volatility_state.outputs_by_date) == expected_dates
    assert set(bundle.breadth_state.outputs_by_date) == expected_dates
    assert set(bundle.event_calendar) == expected_dates


def test_core_axis_output_builder_freezes_hysteresis_state_across_short_data_gaps() -> (
    None
):
    dates = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
    ]
    index = pd.DatetimeIndex(dates)
    required = pd.Series([1.0, None, None, None], index=index)

    result = _build_axis_outputs(
        dates=dates,
        raw_labels=["bear", "unknown", "unknown", "unknown"],
        raw_evidence=[{}, {}, {}, {}],
        risk_rank={"unknown": 2, "bull": 0, "bear": 3},
        deescalation_days_by_label={"bear": 5, "unknown": 0},
        default_deescalation_days=0,
        max_unknown_freeze_days=2,
        required_inputs=[required],
        required_trading_days=1,
        max_freshness_days=0,
        min_completeness=1.0,
    )

    assert result.outputs_by_date[dates[1]].raw_label == "unknown"
    assert result.outputs_by_date[dates[1]].stable_label == "bear"
    assert result.outputs_by_date[dates[1]].active_label == "bear"
    assert result.outputs_by_date[dates[1]].evidence["data_quality_freeze"] is True
    assert result.outputs_by_date[dates[2]].stable_label == "bear"
    assert result.outputs_by_date[dates[3]].stable_label == "unknown"
