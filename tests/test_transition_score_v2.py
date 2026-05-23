from __future__ import annotations

import math

import pytest

from regime_detection.config import TransitionScoreConfig, load_default_regime_config
from regime_detection.transition_score import (
    ComposedTransitionScore,
    compose_transition_score_for_session,
    compute_transition_score,
    interpret_transition_score,
)


@pytest.fixture(scope="module")
def transition_score_config() -> TransitionScoreConfig:
    cfg = load_default_regime_config().transition_score
    assert cfg is not None
    return cfg


def _compose(
    transition_score_config: TransitionScoreConfig,
    **overrides: object,
) -> ComposedTransitionScore:
    kwargs = {
        "realized_vol_short": 10.0,
        "realized_vol_long": 10.0,
        "pct_above_50dma": 0.50,
        "avg_pairwise_corr_percentile_504d": 0.0,
        "drawdown_252d": 0.0,
        "event_calendar_labels": ("normal_calendar",),
        "spy_close": 100.0,
        "spy_sma_50": 100.0,
        "largest_eigenvalue_share_percentile_504d": 0.0,
        "effective_rank_percentile_504d": 1.0,
        "absorption_ratio_top3": 0.50,
        "credit_funding_label": "credit_calm",
        "volume_liquidity_label": "normal_volume",
        "volume_zscore_20d": 1.0,
        "gap_frequency_percentile_252d": 0.0,
        "intraday_range_percentile_252d": 0.0,
        "hmm_top_state_prob_now": 0.50,
        "hmm_top_state_prob_5d_ago": 0.50,
        "change_point_score": 0.0,
        "cluster_id_now": 1,
        "cluster_id_5d_ago": 1,
        "config": transition_score_config,
    }
    kwargs.update(overrides)
    return compose_transition_score_for_session(**kwargs)


def test_component_formulas_have_expected_boundaries(
    transition_score_config: TransitionScoreConfig,
) -> None:
    out = _compose(
        transition_score_config,
        realized_vol_short=15.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.20,
        avg_pairwise_corr_percentile_504d=0.40,
        largest_eigenvalue_share_percentile_504d=0.70,
        effective_rank_percentile_504d=0.10,
        absorption_ratio_top3=0.95,
        drawdown_252d=-0.075,
        spy_close=95.0,
        spy_sma_50=100.0,
        event_calendar_labels=("cpi_week",),
        credit_funding_label="credit_stress",
        volume_liquidity_label="liquidity_gap_behavior",
        volume_zscore_20d=3.0,
        gap_frequency_percentile_252d=0.80,
        intraday_range_percentile_252d=0.60,
        hmm_top_state_prob_now=0.90,
        hmm_top_state_prob_5d_ago=0.20,
        change_point_score=0.40,
        cluster_id_now=2,
        cluster_id_5d_ago=1,
    )

    assert out.components == {
        "trend_break": pytest.approx(1.0),
        "volatility_acceleration": pytest.approx(1.0),
        "breadth_deterioration": pytest.approx(1.0),
        "correlation_fragility": pytest.approx(1.0),
        "credit_stress": pytest.approx(0.75),
        "liquidity_stress": pytest.approx(1.0),
        "macro_event": pytest.approx(1.0),
        "model_instability": pytest.approx(1.0),
    }


def test_macro_event_uses_any_matching_event_calendar_label(
    transition_score_config: TransitionScoreConfig,
) -> None:
    out = _compose(
        transition_score_config,
        event_calendar_labels=("earnings_season", "global_rate_decision"),
    )

    assert out.components is not None
    assert out.components["macro_event"] == pytest.approx(1.0)


def test_compute_transition_score_reweights_available_components(
    transition_score_config: TransitionScoreConfig,
) -> None:
    weights = transition_score_config.weights
    components = {
        "trend_break": 0.50,
        "volatility_acceleration": 1.00,
        "breadth_deterioration": 0.25,
        "correlation_fragility": 0.75,
        "credit_stress": None,
        "liquidity_stress": 0.40,
        "macro_event": 0.00,
        "model_instability": 0.20,
    }

    score, present, missing, coverage = compute_transition_score(
        components=components,
        weights=weights,
        minimum_component_weight_coverage=transition_score_config.minimum_component_weight_coverage,
    )

    present_weight = sum(weights[key] for key, value in components.items() if value is not None)
    expected = sum(
        components[key] * weights[key] / present_weight
        for key, value in components.items()
        if value is not None
    )
    assert score == pytest.approx(expected)
    assert present is not None
    assert "credit_stress" not in present
    assert missing == ("credit_stress",)
    assert coverage == pytest.approx(present_weight / sum(weights.values()))


def test_compute_transition_score_returns_none_when_component_coverage_too_low(
    transition_score_config: TransitionScoreConfig,
) -> None:
    score, present, missing, coverage = compute_transition_score(
        components={
            "trend_break": 0.50,
            "volatility_acceleration": 1.00,
            "breadth_deterioration": 0.25,
            "correlation_fragility": 0.75,
            "credit_stress": None,
            "liquidity_stress": None,
            "macro_event": 0.00,
            "model_instability": None,
        },
        weights=transition_score_config.weights,
        minimum_component_weight_coverage=transition_score_config.minimum_component_weight_coverage,
    )

    assert score is None
    assert present is None
    assert missing == ("credit_stress", "liquidity_stress", "model_instability")
    assert coverage < transition_score_config.minimum_component_weight_coverage


def test_compute_transition_score_rejects_unconfigured_component_key(
    transition_score_config: TransitionScoreConfig,
) -> None:
    with pytest.raises(ValueError, match="unweighted"):
        compute_transition_score(
            components={"trend_break": 0.1, "unweighted": 0.2},
            weights=transition_score_config.weights,
            minimum_component_weight_coverage=transition_score_config.minimum_component_weight_coverage,
        )


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, "stable"),
        (0.35, "weakening"),
        (0.55, "transition_warning"),
        (0.75, "high"),
        (1.0, "high"),
    ],
)
def test_interpret_transition_score_uses_configured_band_boundaries(
    transition_score_config: TransitionScoreConfig,
    score: float,
    expected: str,
) -> None:
    assert interpret_transition_score(score, transition_score_config.bands) == expected


def test_compose_transition_score_exposes_only_present_components(
    transition_score_config: TransitionScoreConfig,
) -> None:
    out = _compose(
        transition_score_config,
        credit_funding_label=None,
    )

    assert out.score is not None
    assert out.components is not None
    assert set(out.components) == set(transition_score_config.weights) - {"credit_stress"}
    assert out.missing_components == ("credit_stress",)
    assert out.component_weight_coverage >= transition_score_config.minimum_component_weight_coverage
    for value in out.components.values():
        assert not math.isnan(value)


def test_compose_transition_score_returns_insufficient_when_many_components_missing(
    transition_score_config: TransitionScoreConfig,
) -> None:
    out = _compose(
        transition_score_config,
        credit_funding_label=None,
        volume_liquidity_label=None,
        volume_zscore_20d=None,
        gap_frequency_percentile_252d=None,
        intraday_range_percentile_252d=None,
        hmm_top_state_prob_now=None,
        hmm_top_state_prob_5d_ago=None,
        change_point_score=None,
        cluster_id_now=None,
        cluster_id_5d_ago=None,
    )

    assert out == ComposedTransitionScore(
        score=None,
        interpretation=None,
        components=None,
        missing_components=("credit_stress", "liquidity_stress", "model_instability"),
        component_weight_coverage=pytest.approx(0.70),
    )
