"""V2 Slice 3 — failing tests for ``transition_score`` (TDD RED phase).

Spec pins: ``docs/regime_engine_v2_spec.md`` §4 (lines 2324–2455), composite
``Without HMM`` weights from §4.3, interpretation bands from §4.4, and NaN
cold-start propagation per V1 §2.7.
"""

from __future__ import annotations

import math

import pytest

from regime_detection.config import (
    TransitionScoreConfig,
    load_default_regime_config,
)
from regime_detection.transition_score import (
    ComposedTransitionScore,
    compose_transition_score_for_session,
    compute_transition_score,
    interpret_transition_score,
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


def test_volatility_acceleration_score_at_boundary_triples(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """§4.2 vol_acc = clip((short/long - 1.0) / 0.5, 0, 1).

    Boundary triple: ratio 1.0 -> 0.0, ratio 1.25 -> 0.5, ratio 1.5 -> 1.0.
    """
    out_zero = compose_transition_score_for_session(
        realized_vol_short=10.0,
        realized_vol_long=10.0,
        pct_above_50dma=_VALID_BREADTH,
        avg_pairwise_corr_percentile_504d=_VALID_CORR,
        drawdown_252d=_VALID_DRAWDOWN,
        event_calendar_label=_VALID_EVENT,
        config=transition_score_config,
    )
    out_half = compose_transition_score_for_session(
        realized_vol_short=12.5,
        realized_vol_long=10.0,
        pct_above_50dma=_VALID_BREADTH,
        avg_pairwise_corr_percentile_504d=_VALID_CORR,
        drawdown_252d=_VALID_DRAWDOWN,
        event_calendar_label=_VALID_EVENT,
        config=transition_score_config,
    )
    out_one = compose_transition_score_for_session(
        realized_vol_short=15.0,
        realized_vol_long=10.0,
        pct_above_50dma=_VALID_BREADTH,
        avg_pairwise_corr_percentile_504d=_VALID_CORR,
        drawdown_252d=_VALID_DRAWDOWN,
        event_calendar_label=_VALID_EVENT,
        config=transition_score_config,
    )
    assert out_zero.components["volatility_acceleration"] == pytest.approx(0.0, abs=1e-9)
    assert out_half.components["volatility_acceleration"] == pytest.approx(0.5, abs=1e-9)
    assert out_one.components["volatility_acceleration"] == pytest.approx(1.0, abs=1e-9)


def test_breadth_deterioration_score_at_boundary_triples(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """§4.2 breadth_det = clip((0.50 - pct_above_50dma) / 0.30, 0, 1)."""
    out_zero = compose_transition_score_for_session(
        realized_vol_short=_VALID_VOL_SHORT,
        realized_vol_long=_VALID_VOL_LONG,
        pct_above_50dma=0.50,
        avg_pairwise_corr_percentile_504d=_VALID_CORR,
        drawdown_252d=_VALID_DRAWDOWN,
        event_calendar_label=_VALID_EVENT,
        config=transition_score_config,
    )
    out_half = compose_transition_score_for_session(
        realized_vol_short=_VALID_VOL_SHORT,
        realized_vol_long=_VALID_VOL_LONG,
        pct_above_50dma=0.35,
        avg_pairwise_corr_percentile_504d=_VALID_CORR,
        drawdown_252d=_VALID_DRAWDOWN,
        event_calendar_label=_VALID_EVENT,
        config=transition_score_config,
    )
    out_one = compose_transition_score_for_session(
        realized_vol_short=_VALID_VOL_SHORT,
        realized_vol_long=_VALID_VOL_LONG,
        pct_above_50dma=0.20,
        avg_pairwise_corr_percentile_504d=_VALID_CORR,
        drawdown_252d=_VALID_DRAWDOWN,
        event_calendar_label=_VALID_EVENT,
        config=transition_score_config,
    )
    assert out_zero.components["breadth_deterioration"] == pytest.approx(0.0, abs=1e-9)
    assert out_half.components["breadth_deterioration"] == pytest.approx(0.5, abs=1e-9)
    assert out_one.components["breadth_deterioration"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "corr_percentile, expected",
    [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.68, 0.68)],
)
def test_correlation_concentration_score_passthrough(
    transition_score_config: TransitionScoreConfig,
    corr_percentile: float,
    expected: float,
) -> None:
    """§4.2 correlation_concentration = avg_pairwise_corr_percentile_504d (pass-through)."""
    out = compose_transition_score_for_session(
        realized_vol_short=_VALID_VOL_SHORT,
        realized_vol_long=_VALID_VOL_LONG,
        pct_above_50dma=_VALID_BREADTH,
        avg_pairwise_corr_percentile_504d=corr_percentile,
        drawdown_252d=_VALID_DRAWDOWN,
        event_calendar_label=_VALID_EVENT,
        config=transition_score_config,
    )
    assert out.components["correlation_concentration"] == pytest.approx(expected, abs=1e-9)


def test_trend_break_score_at_boundary_triples(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """§4.2 trend_break = clip(-drawdown_252d / 0.15, 0, 1)."""
    out_zero = compose_transition_score_for_session(
        realized_vol_short=_VALID_VOL_SHORT,
        realized_vol_long=_VALID_VOL_LONG,
        pct_above_50dma=_VALID_BREADTH,
        avg_pairwise_corr_percentile_504d=_VALID_CORR,
        drawdown_252d=0.0,
        event_calendar_label=_VALID_EVENT,
        config=transition_score_config,
    )
    out_half = compose_transition_score_for_session(
        realized_vol_short=_VALID_VOL_SHORT,
        realized_vol_long=_VALID_VOL_LONG,
        pct_above_50dma=_VALID_BREADTH,
        avg_pairwise_corr_percentile_504d=_VALID_CORR,
        drawdown_252d=-0.075,
        event_calendar_label=_VALID_EVENT,
        config=transition_score_config,
    )
    out_one = compose_transition_score_for_session(
        realized_vol_short=_VALID_VOL_SHORT,
        realized_vol_long=_VALID_VOL_LONG,
        pct_above_50dma=_VALID_BREADTH,
        avg_pairwise_corr_percentile_504d=_VALID_CORR,
        drawdown_252d=-0.15,
        event_calendar_label=_VALID_EVENT,
        config=transition_score_config,
    )
    out_clip = compose_transition_score_for_session(
        realized_vol_short=_VALID_VOL_SHORT,
        realized_vol_long=_VALID_VOL_LONG,
        pct_above_50dma=_VALID_BREADTH,
        avg_pairwise_corr_percentile_504d=_VALID_CORR,
        drawdown_252d=-0.30,
        event_calendar_label=_VALID_EVENT,
        config=transition_score_config,
    )
    assert out_zero.components["trend_break"] == pytest.approx(0.0, abs=1e-9)
    assert out_half.components["trend_break"] == pytest.approx(0.5, abs=1e-9)
    assert out_one.components["trend_break"] == pytest.approx(1.0, abs=1e-9)
    assert out_clip.components["trend_break"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "label, expected",
    [
        ("fed_week", 1.0),
        ("cpi_week", 1.0),
        ("nfp_week", 1.0),
        ("budget_week", 1.0),
        ("election_window", 1.0),
        ("global_rate_decision", 1.0),
        ("normal", 0.0),
        ("unknown", 0.0),
        ("geopolitical_event", 0.0),
        ("expiry_week", 0.0),
    ],
)
def test_macro_event_score_event_set_pins(
    transition_score_config: TransitionScoreConfig,
    label: str,
    expected: float,
) -> None:
    """§4.2 macro_event EVENT_SET pin (6 in-set labels → 1.0; everything else → 0.0)."""
    out = compose_transition_score_for_session(
        realized_vol_short=_VALID_VOL_SHORT,
        realized_vol_long=_VALID_VOL_LONG,
        pct_above_50dma=_VALID_BREADTH,
        avg_pairwise_corr_percentile_504d=_VALID_CORR,
        drawdown_252d=_VALID_DRAWDOWN,
        event_calendar_label=label,
        config=transition_score_config,
    )
    assert out.components["macro_event"] == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Group B — composite weighted sum (§4.1 + §4.3 "Without HMM")
# ---------------------------------------------------------------------------


def test_compute_transition_score_all_components_at_max_yields_one_minus_remainder(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """All five components at 1.0 -> composite = 0.225*4 + 0.10*1 = 1.0."""
    score = compute_transition_score(
        volatility_acceleration_score=1.0,
        breadth_deterioration_score=1.0,
        correlation_concentration_score=1.0,
        trend_break_score=1.0,
        macro_event_score=1.0,
        weights=transition_score_config.weights_without_hmm,
    )
    assert score == pytest.approx(1.0, abs=1e-9)


def test_compute_transition_score_hand_computed_mixed_components(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """Hand-computed mixed components per spec §4.3 Without HMM weights.

    0.225*0.40 + 0.225*0.71 + 0.225*0.68 + 0.225*0.20 + 0.10*1.0
        = 0.09 + 0.15975 + 0.153 + 0.045 + 0.10 = 0.54775
    """
    score = compute_transition_score(
        volatility_acceleration_score=0.40,
        breadth_deterioration_score=0.71,
        correlation_concentration_score=0.68,
        trend_break_score=0.20,
        macro_event_score=1.0,
        weights=transition_score_config.weights_without_hmm,
    )
    assert score == pytest.approx(0.54775, abs=1e-9)


# ---------------------------------------------------------------------------
# Group C — interpretation bands (§4.4, half-open intervals)
# ---------------------------------------------------------------------------


def test_interpret_transition_score_exactly_at_0_35_is_weakening(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """§4.4 lower bound of weakening band is inclusive: exactly 0.35 → weakening."""
    assert (
        interpret_transition_score(0.35, transition_score_config.bands) == "weakening"
    )


def test_interpret_transition_score_exactly_at_0_55_is_transition_warning(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """§4.4 exactly 0.55 → transition_warning (lower bound inclusive)."""
    assert (
        interpret_transition_score(0.55, transition_score_config.bands)
        == "transition_warning"
    )


def test_interpret_transition_score_exactly_at_0_75_is_high(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """§4.4 exactly 0.75 → high (lower bound inclusive)."""
    assert interpret_transition_score(0.75, transition_score_config.bands) == "high"


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, "stable"),
        (0.34, "stable"),
        (0.36, "weakening"),
        (0.54, "weakening"),
        (0.56, "transition_warning"),
        (0.74, "transition_warning"),
        (0.76, "high"),
        (1.0, "high"),
    ],
)
def test_interpret_transition_score_extremes(
    transition_score_config: TransitionScoreConfig,
    score: float,
    expected: str,
) -> None:
    """§4.4 half-open band coverage including 1.0 (top band is closed at the right)."""
    assert interpret_transition_score(score, transition_score_config.bands) == expected


# ---------------------------------------------------------------------------
# Group D — compose_transition_score_for_session integration
# ---------------------------------------------------------------------------


def test_compose_transition_score_full_session_with_all_inputs(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """Realistic session — hand-computed composite and interpretation band.

    Components:
        vol_acc  = clip((1.2 - 1.0)/0.5, 0, 1)        = 0.4
        breadth  = clip((0.50 - 0.45)/0.30, 0, 1)     = 1/6 ≈ 0.166666...
        corr     = 0.60 (pass-through)
        trend    = clip(0.10/0.15, 0, 1)              = 2/3 ≈ 0.666666...
        macro    = 1.0 ("cpi_week" ∈ EVENT_SET)

    Composite (§4.3 Without HMM):
        0.225*0.4 + 0.225*(1/6) + 0.225*0.60 + 0.225*(2/3) + 0.10*1.0
            = 0.09 + 0.0375 + 0.135 + 0.15 + 0.10
            = 0.5125
        → band [0.35, 0.55) → "weakening"
    """
    out = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        config=transition_score_config,
    )

    assert out.score == pytest.approx(0.5125, abs=1e-4)
    assert out.interpretation == "weakening"
    assert out.components is not None
    assert out.components["volatility_acceleration"] == pytest.approx(0.4, abs=1e-9)
    assert out.components["breadth_deterioration"] == pytest.approx(1.0 / 6.0, abs=1e-9)
    assert out.components["correlation_concentration"] == pytest.approx(0.60, abs=1e-9)
    assert out.components["trend_break"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert out.components["macro_event"] == pytest.approx(1.0, abs=1e-9)


def test_compose_transition_score_nan_input_returns_all_none(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """V1 §2.7 cold-start: any NaN numeric feature → composite is fully None."""
    out = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=float("nan"),
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        config=transition_score_config,
    )
    assert out == ComposedTransitionScore(
        score=None, interpretation=None, components=None
    )


def test_compose_transition_score_components_dict_has_canonical_keys(
    transition_score_config: TransitionScoreConfig,
) -> None:
    """§4.1 + §8 row 6: 5 components present, NO hmm_probability_shift key (HMM deferred)."""
    out = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        config=transition_score_config,
    )
    assert out.components is not None
    assert set(out.components.keys()) == {
        "volatility_acceleration",
        "breadth_deterioration",
        "correlation_concentration",
        "trend_break",
        "macro_event",
    }
    # Defensive: sanity-check no NaN sneaks into components dict on success path.
    for value in out.components.values():
        assert not math.isnan(value)


# ---------------------------------------------------------------------------
# Group D — Slice 6 HMM 6th-component wiring (§4.2 line 2396 + §6.1)
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
    ``change_point_score`` from its inputs dict into the composer and
    surfaces ``change_point`` in the ``score_components`` of the
    resulting ``TransitionRiskOutput`` row.

    Drives the wire-in end-to-end (inputs dict → composer → output row)
    without depending on the full feature-store seam being lit by the
    test fixtures.
    """
    from datetime import date as _date

    from regime_detection.transition_risk_series import (
        TransitionRiskHistory,
        build_transition_risk_outputs_by_date,
    )
    from regime_detection.transition_score_inputs import TransitionScoreInputs

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
            hmm_top_state_prob_now=float("nan"),
            hmm_top_state_prob_5d_ago=float("nan"),
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
