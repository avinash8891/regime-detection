from __future__ import annotations

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
    assert out.macro_event_labels == ("global_rate_decision",)


def test_macro_event_audit_ignores_non_scoring_calendar_labels(
    transition_score_config: TransitionScoreConfig,
) -> None:
    out = _compose(
        transition_score_config,
        event_calendar_labels=(
            "earnings_season",
            "fed_week",
            "expiry_week",
            "cpi_week",
        ),
    )

    assert out.components is not None
    assert out.components["macro_event"] == pytest.approx(1.0)
    assert out.macro_event_labels == ("fed_week", "cpi_week")


def test_compute_transition_score_rejects_missing_components(
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

    with pytest.raises(RuntimeError, match="transition score missing components"):
        compute_transition_score(
            components=components,
            weights=weights,
            minimum_component_weight_coverage=transition_score_config.minimum_component_weight_coverage,
        )


def test_compute_transition_score_rejects_low_coverage_missing_components(
    transition_score_config: TransitionScoreConfig,
) -> None:
    with pytest.raises(RuntimeError, match="transition score missing components"):
        compute_transition_score(
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


def test_compose_transition_score_rejects_missing_credit_component(
    transition_score_config: TransitionScoreConfig,
) -> None:
    with pytest.raises(RuntimeError, match="credit_stress"):
        _compose(
            transition_score_config,
            credit_funding_label=None,
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "hmm_top_state_prob_now",
        "hmm_top_state_prob_5d_ago",
        "change_point_score",
        "cluster_id_now",
        "cluster_id_5d_ago",
    ],
)
def test_compose_transition_score_raises_on_missing_model_evidence(
    transition_score_config: TransitionScoreConfig,
    missing_field: str,
) -> None:
    """F-002 / §4.2 / §4.0 / §10 rule 3: once the transition_score seam is enabled,
    per-session model evidence (HMM probability, change-point score, cluster id) is
    MANDATORY. A missing per-session value must fail loudly — it must NEVER be
    silently renormalized away into a normal score or hidden behind a concrete
    transition-risk state."""
    with pytest.raises(RuntimeError, match="transition score missing components"):
        _compose(transition_score_config, **{missing_field: None})


def test_cluster_flip_does_not_fire_on_same_aligned_id(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """Regression: cluster_flip must be 0.0 when cluster_id_now == cluster_id_5d_ago
    (same aligned ID = no genuine regime flip) and 1.0 when IDs differ.

    After Task 1 stabilised cluster IDs across refit boundaries, this comparison
    is meaningful. Locks the fix so a revert cannot silently break it."""
    no_flip = _compose(
        transition_score_config,
        cluster_id_now=2,
        cluster_id_5d_ago=2,
        # Zero out all other model-instability sub-signals so cluster_flip drives
        # the component value directly.
        hmm_top_state_prob_now=0.50,
        hmm_top_state_prob_5d_ago=0.50,
        change_point_score=0.0,
    )
    assert no_flip.components is not None
    assert no_flip.components["model_instability"] == pytest.approx(0.0)

    genuine_flip = _compose(
        transition_score_config,
        cluster_id_now=2,
        cluster_id_5d_ago=5,
        hmm_top_state_prob_now=0.50,
        hmm_top_state_prob_5d_ago=0.50,
        change_point_score=0.0,
    )
    assert genuine_flip.components is not None
    assert genuine_flip.components["model_instability"] == pytest.approx(1.0)


def test_compose_transition_score_rejects_many_missing_components(
    transition_score_config: TransitionScoreConfig,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="trend_break, credit_stress, liquidity_stress",
    ):
        _compose(
            transition_score_config,
            drawdown_252d=None,
            spy_close=None,
            spy_sma_50=None,
            credit_funding_label=None,
            volume_liquidity_label=None,
            volume_zscore_20d=None,
            gap_frequency_percentile_252d=None,
            intraday_range_percentile_252d=None,
        )
