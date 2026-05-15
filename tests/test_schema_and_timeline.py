from __future__ import annotations

from datetime import date

from pathlib import Path

import pytest

from regime_detection.calendar import nyse_calendar
from regime_detection.axis_series import build_axis_series_bundle
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context
from regime_detection.models import RegimeTimeline
from regime_detection.timeline import ENGINE_MINIMUM_HISTORY, build_regime_timeline
from regime_detection.transition_risk_series import build_transition_risk_history
from regime_detection.versioning import engine_version


@pytest.fixture(scope="module")
def shared_timeline_pipeline(market_df_for_asof):
    end_date = date(2023, 12, 14)
    engine = RegimeEngine()
    market_data = market_df_for_asof(end_date)
    context = build_market_context(
        end_date=end_date,
        market_data=market_data,
        config=engine.config,
    )
    feature_store = build_feature_store(context)
    bundle = build_axis_series_bundle(context=context, feature_store=feature_store)
    timeline = build_regime_timeline(
        context=context,
        lookback_days=ENGINE_MINIMUM_HISTORY,
        config=engine.config,
    )
    point_output = engine.classify(as_of_date=end_date, market_data=market_data)
    return {
        "end_date": end_date,
        "market_data": market_data,
        "context": context,
        "feature_store": feature_store,
        "bundle": bundle,
        "timeline": timeline,
        "point_output": point_output,
    }


def test_core3_v1_regime_output_keeps_legacy_placeholder_wire_shapes(market_df_for_asof) -> None:
    """V2 extends V1 cumulatively, but the core3-v1.0.0 archive replay wire
    contract stays byte-identical for the legacy placeholder fields.
    """
    as_of = date(2023, 12, 14)
    engine = RegimeEngine(
        config_path=Path("src/regime_detection/configs/core3-v1.0.0.yaml")
    )
    out = engine.classify(as_of_date=as_of, market_data=market_df_for_asof(as_of))
    dumped = out.model_dump()

    assert dumped["structural_causal_state"]["monetary_pressure"] == {
        "label": "unknown",
        "reason": "not_implemented_v1",
    }
    assert dumped["network_fragility"] == {
        "label": "not_implemented_v1",
        "reason": "breadth_state_used_as_v1_fragility_proxy",
    }
    # Strategy-response conditional modifiers still omitted when not applicable.
    assert "hard_max_loss_required" not in dumped["strategy_response"]
    # Under the archived V1 config, cumulative V2 top-level axes stay omitted.
    for v2_field in (
        "inflation_growth_state",
        "credit_funding_state",
        "volume_liquidity_state",
        "monetary_pressure_state",
    ):
        assert v2_field not in dumped, f"V2 optional field {v2_field!r} should be omitted until its slice ships"
    # TransitionRisk V2 optional fields stay omitted too (no score until slice 3).
    for v2_field in ("score", "score_interpretation", "score_components"):
        assert v2_field not in dumped["transition_risk"], f"transition_risk.{v2_field} should be omitted until v2 slice 3"


def test_classify_window_returns_one_output_per_nyse_trading_day(market_df_for_asof) -> None:
    engine = RegimeEngine()
    end_date = date(2023, 12, 14)
    market_data = market_df_for_asof(end_date)

    timeline = engine.classify_window(
        end_date=end_date,
        market_data=market_data,
        lookback_days=5,
    )

    expected_days = nyse_calendar().schedule(
        start_date=date(2023, 12, 8),
        end_date=end_date,
    ).index.date
    assert [row.as_of_date for row in timeline.outputs] == list(expected_days)
    assert timeline.start_date == expected_days[0]
    assert timeline.end_date == end_date


def test_classify_window_uses_lookback_days_not_fixed_calendar_span(market_df_for_asof) -> None:
    engine = RegimeEngine()
    end_date = date(2023, 12, 14)
    market_data = market_df_for_asof(end_date)

    timeline = engine.classify_window(
        end_date=end_date,
        market_data=market_data,
        lookback_days=23,
    )

    assert len(timeline.outputs) == 23


def test_market_context_builds_normalized_series_once(shared_timeline_pipeline) -> None:
    end_date = shared_timeline_pipeline["end_date"]
    context = shared_timeline_pipeline["context"]

    assert context.end_date == end_date
    assert context.sessions[-1] == end_date
    assert len(context.sessions) >= ENGINE_MINIMUM_HISTORY
    assert list(context.spy_ohlcv.columns) == ["open", "high", "low", "close", "volume"]
    assert context.rsp_close.name == "close"


def test_feature_store_precomputes_aligned_axis_features(shared_timeline_pipeline) -> None:
    context = shared_timeline_pipeline["context"]
    feature_store = shared_timeline_pipeline["feature_store"]

    assert feature_store.spy_index.equals(context.spy_ohlcv.index)
    assert feature_store.trend_direction.close.index.equals(context.spy_ohlcv.index)
    assert feature_store.trend_character.close.index.equals(context.spy_ohlcv.index)
    assert feature_store.volatility.close.index.equals(context.spy_ohlcv.index)
    assert feature_store.breadth.spy_close.index.equals(context.spy_ohlcv.index)


def test_axis_series_bundle_reuses_feature_store_for_all_axes(shared_timeline_pipeline) -> None:
    end_date = shared_timeline_pipeline["end_date"]
    bundle = shared_timeline_pipeline["bundle"]
    point_output = shared_timeline_pipeline["point_output"]

    assert bundle.trend_direction.outputs_by_date[end_date].model_dump() == point_output.trend_direction.model_dump()
    assert bundle.trend_character.outputs_by_date[end_date].model_dump() == point_output.trend_character.model_dump()
    assert bundle.volatility_state.outputs_by_date[end_date].model_dump() == point_output.volatility_state.model_dump()
    assert bundle.breadth_state.outputs_by_date[end_date].model_dump() == point_output.breadth_state.model_dump()
    assert bundle.event_calendar[end_date].model_dump() == point_output.structural_causal_state.event_calendar.model_dump()


def test_classify_matches_last_output_of_shared_timeline_pipeline(shared_timeline_pipeline) -> None:
    timeline = shared_timeline_pipeline["timeline"]
    point_output = shared_timeline_pipeline["point_output"]

    assert timeline.outputs[-1].model_dump() == point_output.model_dump()


def test_classify_delegates_to_classify_window_with_single_day_lookback(mocker, market_df_for_asof) -> None:
    engine = RegimeEngine()
    as_of = date(2023, 12, 14)
    expected_timeline = RegimeTimeline(
        engine_version=engine_version(),
        config_version=engine.config.config_version,
        market="SPY",
        start_date=as_of,
        end_date=as_of,
        trading_calendar="NYSE",
        outputs=[engine.classify_window(end_date=as_of, market_data=market_df_for_asof(as_of), lookback_days=1).outputs[0]],
    )
    spy = mocker.patch.object(engine, "classify_window", return_value=expected_timeline)

    output = engine.classify(as_of_date=as_of, market_data=market_df_for_asof(as_of))

    spy.assert_called_once()
    assert spy.call_args.kwargs["end_date"] == as_of
    assert spy.call_args.kwargs["lookback_days"] == 1
    assert output.model_dump() == expected_timeline.outputs[-1].model_dump()


def test_transition_risk_history_precomputes_axis_switch_and_prior_bear_flags() -> None:
    sessions = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]
    bull = {day: "bull" for day in sessions}
    stable_history = {
        sessions[0]: "bull",
        sessions[1]: "bull",
        sessions[2]: "bear",
        sessions[3]: "bear",
        sessions[4]: "bull",
    }

    history = build_transition_risk_history(
        sessions=sessions,
        trend_direction_stable_by_date=stable_history,
        trend_character_stable_by_date=bull,
        volatility_stable_by_date=bull,
        breadth_stable_by_date=bull,
    )

    assert history.stable_changed_by_date[sessions[0]] is False
    assert history.stable_changed_by_date[sessions[2]] is True
    assert history.stable_changed_by_date[sessions[4]] is True
    assert history.days_since_axis_switch_by_date[sessions[1]] is None
    assert history.days_since_axis_switch_by_date[sessions[2]] == 0
    assert history.days_since_axis_switch_by_date[sessions[3]] == 1
    assert history.days_since_axis_switch_by_date[sessions[4]] == 0
    # v1 §9.4 (post-Q2 fix): prior_bear excludes today's stable_label from the
    # 60-session lookback. sessions[2] is the FIRST day stable goes bear, so
    # the prior window (sessions[0..1] = bull, bull) has no bear and the flag
    # is False. By sessions[3] (the day AFTER the first bear print), the prior
    # window includes sessions[2]=bear, so the flag is True. sessions[4]
    # (bull again) still has bear in its prior window.
    assert history.prior_bear_by_date[sessions[1]] is False
    assert history.prior_bear_by_date[sessions[2]] is False
    assert history.prior_bear_by_date[sessions[3]] is True
    assert history.prior_bear_by_date[sessions[4]] is True
