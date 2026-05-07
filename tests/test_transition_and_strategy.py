from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context
from regime_detection.axis_series import build_axis_series_bundle
from regime_detection.transition_risk_series import TransitionRiskHistory, build_transition_risk_outputs_by_date


def test_transition_risk_matches_real_data_cases(classified_golden_outputs, market_df_for_asof) -> None:
    cases = {
        date(2018, 2, 9): "crisis_override",
        date(2018, 12, 20): "bear_stress_warning",
        date(2019, 9, 11): "post_switch_cooldown",
        date(2020, 4, 29): "recovery_attempt",
        date(2021, 11, 12): "stable",
    }

    for as_of, expected in cases.items():
        out = classified_golden_outputs.get(as_of)
        if out is None:
            out = RegimeEngine().classify(as_of_date=as_of, market_data=market_df_for_asof(as_of))
        assert out.transition_risk.label == expected


def test_strategy_response_matches_crisis_fixture(classified_golden_outputs) -> None:
    as_of = date(2018, 2, 9)
    out = classified_golden_outputs[as_of]

    assert out.transition_risk.label == "crisis_override"
    assert out.strategy_response.position_size_multiplier == 0.25
    assert out.strategy_response.leverage_allowed is False
    assert out.strategy_response.allow_buy_dip is False
    assert out.strategy_response.hard_max_loss_required is True
    assert out.strategy_response.modifiers_applied == ["crisis"]


def test_strategy_response_matches_recovery_attempt_fixture(classified_golden_outputs) -> None:
    as_of = date(2020, 4, 29)
    out = classified_golden_outputs[as_of]

    assert out.transition_risk.label == "recovery_attempt"
    assert out.strategy_response.position_size_multiplier == 0.5
    assert out.strategy_response.require_breadth_confirmation is True
    assert out.strategy_response.allow_leverage_expansion is False
    assert out.strategy_response.modifiers_applied == ["recovery_attempt"]


def test_strategy_response_matches_bull_healthy_low_vol_fixture(classified_golden_outputs) -> None:
    as_of = date(2023, 12, 14)
    out = classified_golden_outputs[as_of]

    assert out.transition_risk.label == "stable"
    assert out.strategy_response.position_size_multiplier == 1.0
    assert out.strategy_response.allow_leverage_expansion is True
    assert out.strategy_response.modifiers_applied == ["bull_healthy_low_vol"]


def test_transition_risk_series_classifier_applies_precedence_from_prepared_inputs() -> None:
    sessions = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
        date(2024, 1, 9),
    ]
    outputs = build_transition_risk_outputs_by_date(
        sessions=sessions,
        trend_direction_active_by_date={
            sessions[0]: "bull",
            sessions[1]: "bear",
            sessions[2]: "bull",
            sessions[3]: "bull",
            sessions[4]: "sideways",
            sessions[5]: "unknown",
        },
        trend_character_active_by_date={
            sessions[0]: "trending",
            sessions[1]: "trending",
            sessions[2]: "transition",
            sessions[3]: "transition",
            sessions[4]: "recovery_attempt",
            sessions[5]: "trending",
        },
        volatility_state_active_by_date={
            sessions[0]: "crisis_vol",
            sessions[1]: "high_vol",
            sessions[2]: "normal_vol",
            sessions[3]: "low_vol",
            sessions[4]: "normal_vol",
            sessions[5]: "normal_vol",
        },
        breadth_state_active_by_date={
            sessions[0]: "healthy_breadth",
            sessions[1]: "weak_breadth",
            sessions[2]: "divergent_fragile",
            sessions[3]: "healthy_breadth",
            sessions[4]: "recovery_breadth",
            sessions[5]: "healthy_breadth",
        },
        close_by_date={
            sessions[0]: 95.0,
            sessions[1]: 95.0,
            sessions[2]: 103.0,
            sessions[3]: 103.0,
            sessions[4]: 104.0,
            sessions[5]: 100.0,
        },
        sma_50_by_date={
            sessions[0]: 100.0,
            sessions[1]: 100.0,
            sessions[2]: 100.0,
            sessions[3]: 100.0,
            sessions[4]: 100.0,
            sessions[5]: 100.0,
        },
        history=TransitionRiskHistory(
            stable_changed_by_date={
                sessions[0]: True,
                sessions[1]: False,
                sessions[2]: True,
                sessions[3]: False,
                sessions[4]: True,
                sessions[5]: False,
            },
            days_since_axis_switch_by_date={
                sessions[0]: 0,
                sessions[1]: None,
                sessions[2]: 0,
                sessions[3]: 1,
                sessions[4]: 0,
                sessions[5]: None,
            },
            prior_bear_by_date={
                sessions[0]: False,
                sessions[1]: True,
                sessions[2]: False,
                sessions[3]: True,
                sessions[4]: True,
                sessions[5]: False,
            },
        ),
    )

    assert outputs[sessions[0]].label == "crisis_override"
    assert outputs[sessions[1]].label == "bear_stress_warning"
    assert outputs[sessions[2]].label == "bull_fragile_warning"
    assert outputs[sessions[3]].label == "recovery_attempt"
    assert outputs[sessions[4]].label == "recovery_attempt"
    assert outputs[sessions[5]].label == "unknown"
    assert outputs[sessions[0]].evidence["warnings_active"] == ["crisis_override"]
    assert outputs[sessions[4]].evidence["warnings_active"] == ["recovery_attempt", "post_switch_cooldown"]


def test_transition_risk_series_fails_fast_on_price_index_misalignment(market_df_for_asof) -> None:
    as_of = date(2023, 12, 14)
    engine = RegimeEngine()
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=engine.config,
    )
    feature_store = build_feature_store(context)
    axis_bundle = build_axis_series_bundle(context=context, feature_store=feature_store)

    misaligned_context = context.model_copy(
        update={
            "spy_ohlcv": context.spy_ohlcv.rename(
                index=lambda ts: pd.Timestamp(ts).tz_localize("America/New_York") + pd.Timedelta(hours=12)
            )
        }
    )
    misaligned_feature_store = feature_store.model_copy(
        update={
            "sma_50": feature_store.sma_50.rename(
                index=lambda ts: pd.Timestamp(ts).tz_localize("America/New_York") + pd.Timedelta(hours=12)
            )
        }
    )

    from regime_detection.transition_risk_series import build_transition_risk_series

    try:
        build_transition_risk_series(
            context=misaligned_context,
            feature_store=misaligned_feature_store,
            axis_bundle=axis_bundle,
        )
    except ValueError as exc:
        assert "transition-risk" in str(exc)
    else:
        raise AssertionError("Expected transition-risk strict series lookup to fail on misaligned indexes")
