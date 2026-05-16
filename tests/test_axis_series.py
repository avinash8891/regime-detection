from __future__ import annotations

from dataclasses import fields
from datetime import date

from regime_detection.axis_series import AxisSeriesBundle, build_axis_series_bundle
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
