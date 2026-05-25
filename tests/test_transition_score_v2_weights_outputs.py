from __future__ import annotations

from datetime import date

import pytest

from regime_detection.config import TransitionScoreConfig, load_default_regime_config
from regime_detection.transition_risk_series import (
    TransitionRiskHistory,
    TransitionScoreInputs,
    build_transition_risk_outputs_by_date,
)
from regime_detection.transition_score import compose_transition_score_for_session


@pytest.fixture(scope="module")
def transition_score_config() -> TransitionScoreConfig:
    cfg = load_default_regime_config().transition_score
    assert cfg is not None
    return cfg


def test_model_instability_collapses_hmm_change_point_and_cluster_evidence(
    transition_score_config: TransitionScoreConfig,
) -> None:
    out = compose_transition_score_for_session(
        realized_vol_short=10.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.50,
        avg_pairwise_corr_percentile_504d=0.0,
        drawdown_252d=0.0,
        event_calendar_labels=("normal_calendar",),
        spy_close=100.0,
        spy_sma_50=100.0,
        largest_eigenvalue_share_percentile_504d=0.0,
        effective_rank_percentile_504d=1.0,
        absorption_ratio_top3=0.50,
        credit_funding_label="credit_calm",
        volume_liquidity_label="normal_volume",
        volume_zscore_20d=1.0,
        gap_frequency_percentile_252d=0.0,
        intraday_range_percentile_252d=0.0,
        hmm_top_state_prob_now=0.80,
        hmm_top_state_prob_5d_ago=0.50,
        change_point_score=0.40,
        cluster_id_now=7,
        cluster_id_5d_ago=7,
        config=transition_score_config,
    )

    assert out.components is not None
    assert out.components["model_instability"] == pytest.approx(0.40)


def test_raw_cluster_id_change_does_not_drive_model_instability(
    transition_score_config: TransitionScoreConfig,
) -> None:
    out = compose_transition_score_for_session(
        realized_vol_short=10.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.50,
        avg_pairwise_corr_percentile_504d=0.0,
        drawdown_252d=0.0,
        event_calendar_labels=("normal_calendar",),
        spy_close=100.0,
        spy_sma_50=100.0,
        largest_eigenvalue_share_percentile_504d=0.0,
        effective_rank_percentile_504d=1.0,
        absorption_ratio_top3=0.50,
        credit_funding_label="credit_calm",
        volume_liquidity_label="normal_volume",
        volume_zscore_20d=1.0,
        gap_frequency_percentile_252d=0.0,
        intraday_range_percentile_252d=0.0,
        hmm_top_state_prob_now=0.50,
        hmm_top_state_prob_5d_ago=0.50,
        change_point_score=0.00,
        cluster_id_now=8,
        cluster_id_5d_ago=7,
        config=transition_score_config,
    )

    assert out.components is not None
    assert out.components["model_instability"] == pytest.approx(0.0)


def test_build_transition_risk_outputs_surfaces_score_components_and_state(
    transition_score_config: TransitionScoreConfig,
) -> None:
    session = date(2024, 1, 9)
    score_inputs = TransitionScoreInputs(
        realized_vol_short=15.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.20,
        avg_pairwise_corr_percentile_504d=0.80,
        largest_eigenvalue_share_percentile_504d=0.90,
        effective_rank_percentile_504d=0.10,
        absorption_ratio_top3=0.95,
        drawdown_252d=-0.15,
        spy_close=95.0,
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

    outputs = build_transition_risk_outputs_by_date(
        sessions=[session],
        trend_direction_active_by_date={session: "bull"},
        trend_character_active_by_date={session: "trending"},
        volatility_state_active_by_date={session: "normal_vol"},
        breadth_state_active_by_date={session: "healthy_breadth"},
        close_by_date={session: 95.0},
        sma_50_by_date={session: 100.0},
        history=TransitionRiskHistory(
            stable_changed_by_date={session: False},
            days_since_axis_switch_by_date={session: None},
            axis_switch_count_by_date={session: 0},
            recent_axis_switch_count_by_date={session: 0},
            prior_bear_by_date={session: False},
        ),
        transition_score_inputs_by_date={session: score_inputs},
        transition_score_config=transition_score_config,
    )

    out = outputs[session]
    assert out.state == "fragile_bull"
    assert out.triggered_rules == ["fragile_bull"]
    assert out.score is not None
    assert out.score_components is not None
    assert set(out.score_components) == set(transition_score_config.weights)
    assert out.primary_drivers[:3] == [
        "trend_break",
        "volatility_acceleration",
        "breadth_deterioration",
    ]
