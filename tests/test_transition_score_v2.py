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
