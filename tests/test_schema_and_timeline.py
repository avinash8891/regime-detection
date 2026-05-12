from __future__ import annotations

from datetime import date

from regime_detection.calendar import nyse_calendar
from regime_detection.axis_series import build_axis_series_bundle
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context
from regime_detection.models import RegimeTimeline
from regime_detection.timeline import ENGINE_MINIMUM_HISTORY, build_regime_timeline
from regime_detection.transition_risk_series import build_transition_risk_history
from regime_detection.versioning import engine_version


def test_regime_output_emits_v2_unknown_placeholders_until_classifiers_ship(market_df_for_asof) -> None:
    """Until V2 slice 1 (network_fragility) and slice 4 (monetary_pressure) ship,
    those wire fields emit the V2 shape with `unknown` labels per V1 §2.7
    NaN-cold-start pattern. Optional V2 top-level fields stay omitted.
    """
    as_of = date(2023, 12, 14)
    out = RegimeEngine().classify(as_of_date=as_of, market_data=market_df_for_asof(as_of))
    dumped = out.model_dump()

    assert dumped["structural_causal_state"]["monetary_pressure"] == {
        "label": "unknown",
        "evidence": {"reason": "v2_classifier_not_yet_implemented"},
        "data_quality": {
            "status": "insufficient_history",
            "reason": "required_feature_is_nan",
        },
    }
    assert dumped["network_fragility"] == {
        "raw_label": "unknown",
        "stable_label": "unknown",
        "active_label": "unknown",
        "evidence": {"reason": "v2_classifier_not_yet_implemented"},
        "data_quality": {
            "status": "insufficient_history",
            "reason": "required_feature_is_nan",
        },
        "mode": "sector_cross_asset_22",
    }
    # Strategy-response conditional modifiers still omitted when not applicable.
    assert "hard_max_loss_required" not in dumped["strategy_response"]
    # New V2 top-level fields default to None → omitted via exclude_none=True.
    # `volume_liquidity_state` ships in Slice 2.7 and IS populated when the v2
    # config carries the axis block (default core3-v2.0.0.yaml does).
    for v2_field in ("inflation_growth_state", "credit_funding_state", "change_point"):
        assert v2_field not in dumped, f"V2 optional field {v2_field!r} should be omitted until its slice ships"
    # Slice 2.7: volume_liquidity_state is now populated end-to-end.
    assert "volume_liquidity_state" in dumped
    assert dumped["volume_liquidity_state"]["mode"] == "volume_zscore_v1"
    assert dumped["volume_liquidity_state"]["raw_label"] in {
        "normal_volume", "panic_volume", "liquidity_gap_behavior", "unknown",
    }
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


def test_market_context_builds_normalized_series_once(market_df_for_asof) -> None:
    end_date = date(2023, 12, 14)
    context = build_market_context(
        end_date=end_date,
        market_data=market_df_for_asof(end_date),
        config=RegimeEngine().config,
    )

    assert context.end_date == end_date
    assert context.sessions[-1] == end_date
    assert len(context.sessions) >= ENGINE_MINIMUM_HISTORY
    assert list(context.spy_ohlcv.columns) == ["open", "high", "low", "close", "volume"]
    assert context.rsp_close.name == "close"


def test_feature_store_precomputes_aligned_axis_features(market_df_for_asof) -> None:
    end_date = date(2023, 12, 14)
    context = build_market_context(
        end_date=end_date,
        market_data=market_df_for_asof(end_date),
        config=RegimeEngine().config,
    )
    feature_store = build_feature_store(context)

    assert feature_store.spy_index.equals(context.spy_ohlcv.index)
    assert feature_store.trend_direction.close.index.equals(context.spy_ohlcv.index)
    assert feature_store.trend_character.close.index.equals(context.spy_ohlcv.index)
    assert feature_store.volatility.close.index.equals(context.spy_ohlcv.index)
    assert feature_store.breadth.spy_close.index.equals(context.spy_ohlcv.index)


def test_axis_series_bundle_reuses_feature_store_for_all_axes(market_df_for_asof) -> None:
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
    point_output = engine.classify(as_of_date=end_date, market_data=market_data)

    assert bundle.trend_direction.outputs_by_date[end_date].model_dump() == point_output.trend_direction.model_dump()
    assert bundle.trend_character.outputs_by_date[end_date].model_dump() == point_output.trend_character.model_dump()
    assert bundle.volatility_state.outputs_by_date[end_date].model_dump() == point_output.volatility_state.model_dump()
    assert bundle.breadth_state.outputs_by_date[end_date].model_dump() == point_output.breadth_state.model_dump()
    assert bundle.event_calendar[end_date].model_dump() == point_output.structural_causal_state.event_calendar.model_dump()


def test_classify_matches_last_output_of_shared_timeline_pipeline(market_df_for_asof) -> None:
    end_date = date(2023, 12, 14)
    engine = RegimeEngine()
    market_data = market_df_for_asof(end_date)

    context = build_market_context(
        end_date=end_date,
        market_data=market_data,
        config=engine.config,
    )
    timeline = build_regime_timeline(
        context=context,
        lookback_days=ENGINE_MINIMUM_HISTORY,
        config=engine.config,
    )
    point_output = engine.classify(as_of_date=end_date, market_data=market_data)

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
    assert history.prior_bear_by_date[sessions[1]] is False
    assert history.prior_bear_by_date[sessions[2]] is True
    assert history.prior_bear_by_date[sessions[4]] is True
