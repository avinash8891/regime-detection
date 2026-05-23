from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context
from regime_detection.axis_series import build_axis_series_bundle
from regime_detection.transition_risk_series import (
    TransitionRiskHistory,
    TransitionScoreInputs,
    build_transition_risk_outputs_by_date,
)
from regime_detection.strategy_response import build_strategy_response

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GOLDEN_PATH = _REPO_ROOT / "tests" / "fixtures" / "derived" / "golden_dates.yaml"


def _golden_date(intent_id: str) -> date:
    golden = yaml.safe_load(_GOLDEN_PATH.read_text())
    for row in golden["rows"]:
        if row["intent_id"] == intent_id:
            return date.fromisoformat(row["as_of_date"])
    raise KeyError(f"intent_id {intent_id!r} not found in golden_dates.yaml")


def test_transition_risk_golden_fixture_without_v2_score_inputs_fails_loudly(
    market_df_for_asof,
) -> None:
    as_of = _golden_date("early2024_bull_lowvol")
    engine = RegimeEngine()
    with pytest.raises((RuntimeError, ValueError)):
        engine.classify(as_of_date=as_of, market_data=market_df_for_asof(as_of))


def test_strategy_response_de_risks_crisis_final_state() -> None:
    response = build_strategy_response(
        trend_direction_active="bear",
        trend_character_active="transition",
        volatility_state_active="crisis_vol",
        breadth_state_active="weak_breadth",
        transition_risk_state="crisis",
            )

    assert response.position_size_multiplier == 0.25
    assert response.leverage_allowed is False
    assert response.allow_buy_dip is False
    assert response.hard_max_loss_required is True
    assert response.modifiers_applied == ["crisis"]


def test_strategy_response_handles_recovery_attempt_final_state() -> None:
    response = build_strategy_response(
        trend_direction_active="sideways",
        trend_character_active="recovery_attempt",
        volatility_state_active="normal_vol",
        breadth_state_active="recovery_breadth",
        transition_risk_state="recovery_attempt",
            )

    assert response.position_size_multiplier == 0.5
    assert response.leverage_allowed is False
    assert "recovery_attempt" in response.modifiers_applied


def test_strategy_response_de_risks_high_transition_risk_final_state() -> None:
    response = build_strategy_response(
        trend_direction_active="bull",
        trend_character_active="trending",
        volatility_state_active="normal_vol",
        breadth_state_active="healthy_breadth",
        transition_risk_state="high_transition_risk",
            )

    assert response.position_size_multiplier == 0.5
    assert response.leverage_allowed is False
    assert response.allow_buy_dip is False
    assert response.prefer_cash_or_hedges is True
    assert response.modifiers_applied == ["bull_healthy_low_vol", "high_transition_risk"]


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
            sessions[0]: "normal_vol",
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
            axis_switch_count_by_date={
                sessions[0]: 1,
                sessions[1]: 0,
                sessions[2]: 1,
                sessions[3]: 0,
                sessions[4]: 1,
                sessions[5]: 0,
            },
            recent_axis_switch_count_by_date={
                sessions[0]: 1,
                sessions[1]: 1,
                sessions[2]: 2,
                sessions[3]: 2,
                sessions[4]: 3,
                sessions[5]: 2,
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
        transition_score_inputs_by_date={
            session: TransitionScoreInputs(
                realized_vol_short=10.0,
                realized_vol_long=10.0,
                pct_above_50dma=0.50,
                avg_pairwise_corr_percentile_504d=0.0,
                largest_eigenvalue_share_percentile_504d=0.0,
                effective_rank_percentile_504d=1.0,
                absorption_ratio_top3=0.50,
                drawdown_252d=0.0,
                spy_close=100.0,
                spy_sma_50=100.0,
                event_calendar_labels=("normal_calendar",),
                credit_funding_label="credit_calm",
                volume_liquidity_label="normal_volume",
                volume_zscore_20d=1.0,
                gap_frequency_percentile_252d=0.0,
                intraday_range_percentile_252d=0.0,
                hmm_top_state_prob_now=0.50,
                hmm_top_state_prob_5d_ago=0.50,
                change_point_score=0.0,
                cluster_id_now=1,
                cluster_id_5d_ago=1,
            )
            for session in sessions
        },
        transition_score_config=RegimeEngine().config.transition_score,
    )

    assert outputs[sessions[0]].state == "watch"
    assert outputs[sessions[1]].state == "bear_stress"
    assert outputs[sessions[2]].state == "bear_stress"
    assert outputs[sessions[2]].triggered_rules == [
        "fragile_bull",
        "post_switch_cooldown",
        "state_confirmation_pending",
    ]
    assert outputs[sessions[3]].state == "bear_stress"
    assert outputs[sessions[3]].triggered_rules == [
        "recovery_attempt",
        "post_switch_cooldown",
        "state_confirmation_pending",
    ]
    assert outputs[sessions[4]].state == "recovery_attempt"
    assert outputs[sessions[5]].state == "insufficient_data"
    assert outputs[sessions[0]].evidence["triggered_rules"] == ["post_switch_cooldown"]
    assert outputs[sessions[4]].evidence["triggered_rules"] == ["recovery_attempt", "post_switch_cooldown"]


def test_transition_risk_series_restores_absolute_crisis_override() -> None:
    session = date(2024, 1, 2)
    outputs = build_transition_risk_outputs_by_date(
        sessions=[session],
        trend_direction_active_by_date={session: "bull"},
        trend_character_active_by_date={session: "trending"},
        volatility_state_active_by_date={session: "crisis_vol"},
        breadth_state_active_by_date={session: "healthy_breadth"},
        close_by_date={session: 100.0},
        sma_50_by_date={session: 100.0},
        history=TransitionRiskHistory(
            stable_changed_by_date={session: False},
            days_since_axis_switch_by_date={session: None},
            axis_switch_count_by_date={session: 0},
            recent_axis_switch_count_by_date={session: 0},
            prior_bear_by_date={session: False},
        ),
        transition_score_inputs_by_date={
            session: TransitionScoreInputs(
                realized_vol_short=10.0,
                realized_vol_long=10.0,
                pct_above_50dma=0.50,
                avg_pairwise_corr_percentile_504d=0.0,
                largest_eigenvalue_share_percentile_504d=0.0,
                effective_rank_percentile_504d=1.0,
                absorption_ratio_top3=0.50,
                drawdown_252d=0.0,
                spy_close=100.0,
                spy_sma_50=100.0,
                event_calendar_labels=("normal_calendar",),
                credit_funding_label="credit_calm",
                volume_liquidity_label="normal_volume",
                volume_zscore_20d=1.0,
                gap_frequency_percentile_252d=0.0,
                intraday_range_percentile_252d=0.0,
                hmm_top_state_prob_now=0.50,
                hmm_top_state_prob_5d_ago=0.50,
                change_point_score=0.0,
                cluster_id_now=1,
                cluster_id_5d_ago=1,
            )
        },
        transition_score_config=RegimeEngine().config.transition_score,
    )

    assert outputs[session].state == "crisis"
    assert outputs[session].triggered_rules == ["crisis"]


def test_transition_risk_series_maps_sideways_stress_to_watch() -> None:
    session = date(2024, 1, 2)
    outputs = build_transition_risk_outputs_by_date(
        sessions=[session],
        trend_direction_active_by_date={session: "sideways"},
        trend_character_active_by_date={session: "chop"},
        volatility_state_active_by_date={session: "high_vol"},
        breadth_state_active_by_date={session: "weak_breadth"},
        close_by_date={session: 100.0},
        sma_50_by_date={session: 100.0},
        history=TransitionRiskHistory(
            stable_changed_by_date={session: False},
            days_since_axis_switch_by_date={session: None},
            axis_switch_count_by_date={session: 0},
            recent_axis_switch_count_by_date={session: 0},
            prior_bear_by_date={session: False},
        ),
        transition_score_inputs_by_date={
            session: TransitionScoreInputs(
                realized_vol_short=10.0,
                realized_vol_long=10.0,
                pct_above_50dma=0.50,
                avg_pairwise_corr_percentile_504d=0.0,
                largest_eigenvalue_share_percentile_504d=0.0,
                effective_rank_percentile_504d=1.0,
                absorption_ratio_top3=0.50,
                drawdown_252d=0.0,
                spy_close=100.0,
                spy_sma_50=100.0,
                event_calendar_labels=("normal_calendar",),
                credit_funding_label="credit_calm",
                volume_liquidity_label="normal_volume",
                volume_zscore_20d=1.0,
                gap_frequency_percentile_252d=0.0,
                intraday_range_percentile_252d=0.0,
                hmm_top_state_prob_now=0.50,
                hmm_top_state_prob_5d_ago=0.50,
                change_point_score=0.0,
                cluster_id_now=1,
                cluster_id_5d_ago=1,
            )
        },
        transition_score_config=RegimeEngine().config.transition_score,
    )

    assert outputs[session].state == "watch"
    assert outputs[session].triggered_rules == ["sideways_stress"]


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
