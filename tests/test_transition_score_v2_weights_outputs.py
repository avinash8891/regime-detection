"""V2 Slice 3 — failing tests for ``transition_score`` (TDD RED phase).

Spec pins: ``docs/regime_engine_v2_spec.md`` §4 (lines 2324–2455), composite
``Without HMM`` weights from §4.3, interpretation bands from §4.4, and NaN
cold-start propagation per V1 §2.7.
"""

from __future__ import annotations


import pytest

from regime_detection.config import (
    TransitionScoreConfig,
    load_default_regime_config,
)
from regime_detection.transition_score import (
    compose_transition_score_for_session,
    compute_transition_score,
)


# Default valid inputs used to "hold other inputs fixed" while exercising a
# single component. None of these are NaN; each yields a well-defined score.
_VALID_VOL_SHORT = 10.0
_VALID_VOL_LONG = 10.0
_VALID_BREADTH = 0.50
_VALID_CORR = 0.0
_VALID_DRAWDOWN = 0.0
_VALID_EVENT = "normal"


@pytest.fixture(scope="module")
def transition_score_config() -> TransitionScoreConfig:
    """Real production config — no hand-built TransitionScoreConfig (AGENTS B)."""
    cfg = load_default_regime_config().transition_score
    assert cfg is not None, "default config must populate transition_score (§4.3/§4.4)"
    return cfg


# ---------------------------------------------------------------------------
# Group A — component formulas (§4.2)
# ---------------------------------------------------------------------------


def test_compose_transition_score_uses_with_hmm_weights_when_hmm_present(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """When both hmm probabilities are supplied, the composer must use the
    ``weights_with_hmm`` 6-key table and emit the ``hmm_probability_shift``
    component. Hand-computed expectation matches.
    """
    out = compose_transition_score_for_session(
        realized_vol_short=10.0,
        realized_vol_long=10.0,        # vol_acc → 0.0
        pct_above_50dma=0.50,           # breadth_det → 0.0
        avg_pairwise_corr_percentile_504d=0.0,  # corr_conc → 0.0
        drawdown_252d=0.0,              # trend_break → 0.0
        event_calendar_label="normal",  # macro_event → 0.0
        hmm_top_state_prob_now=0.7,
        hmm_top_state_prob_5d_ago=0.3,
        config=transition_score_config,
    )
    assert out.components is not None
    assert "hmm_probability_shift" in out.components
    # |0.7 - 0.3| = 0.4 → clipped to [0,1] → 0.4
    assert out.components["hmm_probability_shift"] == pytest.approx(0.4)
    # Score = sum of weighted zeros + weights_with_hmm.hmm_probability_shift * 0.4
    expected = (
        transition_score_config.weights_with_hmm["hmm_probability_shift"] * 0.4
    )
    assert out.score == pytest.approx(expected)
    # All 6 keys present.
    assert set(out.components.keys()) == {
        "volatility_acceleration",
        "breadth_deterioration",
        "correlation_concentration",
        "trend_break",
        "macro_event",
        "hmm_probability_shift",
    }


def test_compose_transition_score_falls_back_to_without_hmm_weights_when_hmm_none(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """Passing ``hmm_top_state_prob_now=None`` must reproduce V1+V2 5-component
    behavior byte-identical to the no-arg (default-None) call path.
    """
    out_explicit_none = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        hmm_top_state_prob_now=None,
        hmm_top_state_prob_5d_ago=None,
        config=transition_score_config,
    )
    out_default = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        config=transition_score_config,
    )
    assert out_explicit_none.score == out_default.score
    assert out_explicit_none.components == out_default.components
    assert out_explicit_none.components is not None
    assert "hmm_probability_shift" not in out_explicit_none.components


def test_hmm_probability_shift_score_formula_at_boundaries(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """Slice 6 §4.2 formula: hmm_probability_shift_score = clip(|p_now - p_5d|, 0, 1)."""
    out_max = compose_transition_score_for_session(
        realized_vol_short=10.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.50,
        avg_pairwise_corr_percentile_504d=0.0,
        drawdown_252d=0.0,
        event_calendar_label="normal",
        hmm_top_state_prob_now=1.0,
        hmm_top_state_prob_5d_ago=0.0,
        config=transition_score_config,
    )
    assert out_max.components is not None
    assert out_max.components["hmm_probability_shift"] == pytest.approx(1.0)

    out_zero = compose_transition_score_for_session(
        realized_vol_short=10.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.50,
        avg_pairwise_corr_percentile_504d=0.0,
        drawdown_252d=0.0,
        event_calendar_label="normal",
        hmm_top_state_prob_now=0.5,
        hmm_top_state_prob_5d_ago=0.5,
        config=transition_score_config,
    )
    assert out_zero.components is not None
    assert out_zero.components["hmm_probability_shift"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Group E — Slice 8.x change_point 7th-component wiring (Ambiguity Log #66)
# ---------------------------------------------------------------------------


def test_compose_transition_score_uses_with_change_point_weights_when_cp_only(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """When change_point_score is supplied but HMM probabilities are absent,
    the composer must use ``weights_with_change_point`` (6-key table) and
    emit ``change_point`` in components — NOT ``hmm_probability_shift``.
    """
    out = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,        # vol_acc → 0.4
        pct_above_50dma=0.45,           # breadth_det → 1/6
        avg_pairwise_corr_percentile_504d=0.60,  # corr → 0.60
        drawdown_252d=-0.075,            # trend → 0.5
        event_calendar_label="cpi_week",  # macro → 1.0
        change_point_score=0.5,
        config=transition_score_config,
    )
    assert out.components is not None
    assert "change_point" in out.components
    assert "hmm_probability_shift" not in out.components
    assert set(out.components.keys()) == {
        "volatility_acceleration",
        "breadth_deterioration",
        "correlation_concentration",
        "trend_break",
        "macro_event",
        "change_point",
    }
    assert out.components["change_point"] == pytest.approx(0.5, abs=1e-9)
    w = transition_score_config.weights_with_change_point
    expected = (
        w["volatility_acceleration"] * 0.4
        + w["breadth_deterioration"] * (1.0 / 6.0)
        + w["correlation_concentration"] * 0.60
        + w["trend_break"] * 0.5
        + w["macro_event"] * 1.0
        + w["change_point"] * 0.5
    )
    assert out.score == pytest.approx(expected, abs=1e-9)


def test_compose_transition_score_uses_with_hmm_with_cp_weights_when_both_present(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """Both HMM and change_point supplied → 7-component
    ``weights_with_hmm_with_change_point`` table.
    """
    out = compose_transition_score_for_session(
        realized_vol_short=10.0,
        realized_vol_long=10.0,        # vol_acc → 0.0
        pct_above_50dma=0.50,           # breadth_det → 0.0
        avg_pairwise_corr_percentile_504d=0.0,  # corr_conc → 0.0
        drawdown_252d=0.0,              # trend_break → 0.0
        event_calendar_label="normal",  # macro_event → 0.0
        hmm_top_state_prob_now=0.7,
        hmm_top_state_prob_5d_ago=0.3,  # hmm_shift = 0.4
        change_point_score=0.5,
        config=transition_score_config,
    )
    assert out.components is not None
    assert set(out.components.keys()) == {
        "volatility_acceleration",
        "breadth_deterioration",
        "correlation_concentration",
        "trend_break",
        "macro_event",
        "hmm_probability_shift",
        "change_point",
    }
    w = transition_score_config.weights_with_hmm_with_change_point
    expected = (
        w["hmm_probability_shift"] * 0.4
        + w["change_point"] * 0.5
    )
    assert out.score == pytest.approx(expected, abs=1e-9)


def test_compose_transition_score_falls_back_to_without_hmm_when_neither_present(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """Explicitly passing hmm=None and change_point=None must reproduce the
    pre-Log#66 5-component byte-identity path (V1 fallback).
    """
    out_explicit = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        hmm_top_state_prob_now=None,
        hmm_top_state_prob_5d_ago=None,
        change_point_score=None,
        config=transition_score_config,
    )
    out_default = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        config=transition_score_config,
    )
    assert out_explicit.score == out_default.score
    assert out_explicit.components == out_default.components
    assert out_explicit.components is not None
    assert "change_point" not in out_explicit.components
    assert "hmm_probability_shift" not in out_explicit.components


def test_compute_transition_score_raises_on_unknown_weight_key(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """Unknown component names in weights dict must raise ValueError that
    mentions the offending key.
    """
    bogus_weights = {
        "volatility_acceleration": 0.5,
        "bogus_component": 0.5,
    }
    with pytest.raises(ValueError, match="bogus_component"):
        compute_transition_score(
            volatility_acceleration_score=0.1,
            breadth_deterioration_score=0.2,
            correlation_concentration_score=0.3,
            trend_break_score=0.4,
            macro_event_score=0.0,
            weights=bogus_weights,
        )


def test_compose_transition_score_components_dict_keys_match_selected_weight_table(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """The components dict must contain exactly the same keys as the selected
    weight table across all 4 evidence configurations.
    """
    # Path 1: neither HMM nor CP → weights_without_hmm (5 keys).
    out_none = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        config=transition_score_config,
    )
    assert out_none.components is not None
    assert set(out_none.components.keys()) == set(
        transition_score_config.weights_without_hmm.keys()
    )

    # Path 2: HMM only → weights_with_hmm (6 keys).
    out_hmm = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        hmm_top_state_prob_now=0.7,
        hmm_top_state_prob_5d_ago=0.3,
        config=transition_score_config,
    )
    assert out_hmm.components is not None
    assert set(out_hmm.components.keys()) == set(
        transition_score_config.weights_with_hmm.keys()
    )

    # Path 3: CP only → weights_with_change_point (6 keys).
    out_cp = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        change_point_score=0.5,
        config=transition_score_config,
    )
    assert out_cp.components is not None
    assert set(out_cp.components.keys()) == set(
        transition_score_config.weights_with_change_point.keys()
    )

    # Path 4: both → weights_with_hmm_with_change_point (7 keys).
    out_both = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        hmm_top_state_prob_now=0.7,
        hmm_top_state_prob_5d_ago=0.3,
        change_point_score=0.5,
        config=transition_score_config,
    )
    assert out_both.components is not None
    assert set(out_both.components.keys()) == set(
        transition_score_config.weights_with_hmm_with_change_point.keys()
    )


def test_compose_transition_score_nan_hmm_with_non_nan_cp_uses_with_change_point(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """NaN HMM probability + valid change_point → weights_with_change_point.

    The HMM/CP optional inputs are evaluated independently for None/NaN to
    decide the weight table (Log #66).
    """
    out = compose_transition_score_for_session(
        realized_vol_short=10.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.50,
        avg_pairwise_corr_percentile_504d=0.0,
        drawdown_252d=0.0,
        event_calendar_label="normal",
        hmm_top_state_prob_now=float("nan"),
        hmm_top_state_prob_5d_ago=float("nan"),
        change_point_score=0.5,
        config=transition_score_config,
    )
    assert out.components is not None
    assert "hmm_probability_shift" not in out.components
    assert "change_point" in out.components
    assert set(out.components.keys()) == set(
        transition_score_config.weights_with_change_point.keys()
    )


def test_build_transition_risk_outputs_emits_change_point_when_seam_lit(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """Integration: the per-session outputs builder threads
    ``change_point_score`` from typed inputs into the composer and
    surfaces ``change_point`` in the ``score_components`` of the
    resulting ``TransitionRiskOutput`` row.

    Drives the wire-in end-to-end (typed inputs → composer → output row)
    without depending on the full feature-store seam being lit by the
    test fixtures.
    """
    from datetime import date as _date

    from regime_detection.transition_risk_series import (
        TransitionRiskHistory,
        TransitionScoreInputs,
        build_transition_risk_outputs_by_date,
    )

    session = _date(2023, 12, 14)
    sessions = [session]
    history = TransitionRiskHistory(
        stable_changed_by_date={session: False},
        days_since_axis_switch_by_date={session: None},
        prior_bear_by_date={session: False},
    )
    inputs = {
        session: TransitionScoreInputs(
            realized_vol_short=12.0,
            realized_vol_long=10.0,
            pct_above_50dma=0.45,
            avg_pairwise_corr_percentile_504d=0.60,
            drawdown_252d=-0.10,
            event_calendar_label="cpi_week",
            change_point_score=0.5,
        )
    }
    outputs = build_transition_risk_outputs_by_date(
        sessions=sessions,
        trend_direction_active_by_date={session: "bull"},
        trend_character_active_by_date={session: "trending"},
        volatility_state_active_by_date={session: "normal_vol"},
        breadth_state_active_by_date={session: "healthy_breadth"},
        close_by_date={session: 450.0},
        sma_50_by_date={session: 440.0},
        history=history,
        transition_score_inputs_by_date=inputs,
        transition_score_config=transition_score_config,
    )
    out = outputs[session]
    assert out.score_components is not None
    assert "change_point" in out.score_components
    assert out.score_components["change_point"] == pytest.approx(0.5, abs=1e-9)
    assert set(out.score_components.keys()) == set(
        transition_score_config.weights_with_change_point.keys()
    )
