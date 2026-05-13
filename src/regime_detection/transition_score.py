"""v2 §4 — Transition Score evidence composer (Slice 3, Without-HMM rhythm).

Implements the five 5-of-6 deterministic component scores defined verbatim
in v2 §4.2, the §4.3 weighted composite (5-component, post-HMM-deferral
renormalization), and the §4.4 half-open interpretation bands.
HMM-driven ``hmm_probability_shift_score`` (the 6th component) is deferred
to Slice 6; the §4.3 "Without HMM" weight table sums to 1.0 over the 5
components shipped here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from regime_detection.config import TransitionScoreConfig


EVENT_SET: frozenset[str] = frozenset(
    {
        "fed_week",            # v2 §4.2 + V1
        "cpi_week",            # v2 §4.2 + V1
        "nfp_week",            # v2 §4.2 + V1
        "budget_week",         # v2 §2D additions
        "election_window",     # v2 §2D additions
        "global_rate_decision",  # v2 §2D additions
    }
)


@dataclass(frozen=True)
class ComposedTransitionScore:
    """Composed transition-score evidence for a single session.

    All three fields are ``None`` together on NaN cold-start propagation
    (see ``compose_transition_score_for_session``).
    """

    score: float | None
    interpretation: str | None
    components: dict[str, float] | None


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_transition_score(
    *,
    volatility_acceleration_score: float,
    breadth_deterioration_score: float,
    correlation_concentration_score: float,
    trend_break_score: float,
    macro_event_score: float,
    weights: dict[str, float],
) -> float:
    """v2 §4.3 weighted composite over the 5 Without-HMM components."""
    if weights.get("hmm_probability_shift", 0.0) != 0.0:
        raise ValueError(
            "Slice 3 Without-HMM composer received weights_with_hmm — "
            "pass weights_without_hmm only."
        )
    return (
        weights["volatility_acceleration"] * volatility_acceleration_score
        + weights["breadth_deterioration"] * breadth_deterioration_score
        + weights["correlation_concentration"] * correlation_concentration_score
        + weights["trend_break"] * trend_break_score
        + weights["macro_event"] * macro_event_score
    )


def interpret_transition_score(
    score: float, bands: dict[str, tuple[float, float]]
) -> str:
    """v2 §4.4 half-open band lookup; top band (``high``) is closed at 1.0."""
    if score < 0.0 or score > 1.0:
        raise ValueError(
            f"transition_score must be in [0.0, 1.0], got {score!r}"
        )
    if score < bands["weakening"][0]:
        return "stable"
    if score < bands["transition_warning"][0]:
        return "weakening"
    if score < bands["high"][0]:
        return "transition_warning"
    return "high"


def compose_transition_score_for_session(
    *,
    realized_vol_short: float,
    realized_vol_long: float,
    pct_above_50dma: float,
    avg_pairwise_corr_percentile_504d: float,
    drawdown_252d: float,
    event_calendar_label: str,
    config: TransitionScoreConfig,
) -> ComposedTransitionScore:
    """Compose a single session's transition score from v2 §4.2 inputs.

    Returns ``ComposedTransitionScore(None, None, None)`` if any numeric
    input is NaN (V1 §2.7 cold-start propagation) or if
    ``realized_vol_long == 0.0`` (ratio undefined → treat as NaN
    propagation; avoids ZeroDivisionError on cold-start when both
    realised-vol windows are still warming up).
    """
    numeric_inputs = (
        float(realized_vol_short),
        float(realized_vol_long),
        float(pct_above_50dma),
        float(avg_pairwise_corr_percentile_504d),
        float(drawdown_252d),
    )
    if any(math.isnan(v) for v in numeric_inputs):
        return ComposedTransitionScore(score=None, interpretation=None, components=None)

    rv_short, rv_long, pct_above_50, corr_pct, dd_252 = numeric_inputs

    # Cold-start guard: rv_long == 0.0 → ratio undefined; propagate as NaN.
    if rv_long == 0.0:
        return ComposedTransitionScore(score=None, interpretation=None, components=None)

    # --- v2 §4.2 component formulas (literal thresholds per spec) --------
    ratio = rv_short / rv_long
    vol_acc = _clip((ratio - 1.0) / 0.5, 0.0, 1.0)              # v2 §4.2
    breadth_det = _clip((0.50 - pct_above_50) / 0.30, 0.0, 1.0)  # v2 §4.2
    corr_conc = corr_pct                                          # v2 §4.2 pass-through
    trend_break = _clip(-dd_252 / 0.15, 0.0, 1.0)                # v2 §4.2
    macro_event = 1.0 if event_calendar_label in EVENT_SET else 0.0  # v2 §4.2

    components: dict[str, float] = {
        "volatility_acceleration": vol_acc,
        "breadth_deterioration": breadth_det,
        "correlation_concentration": corr_conc,
        "trend_break": trend_break,
        "macro_event": macro_event,
    }

    score = compute_transition_score(
        volatility_acceleration_score=vol_acc,
        breadth_deterioration_score=breadth_det,
        correlation_concentration_score=corr_conc,
        trend_break_score=trend_break,
        macro_event_score=macro_event,
        weights=config.weights_without_hmm,
    )
    # Composite is a weighted average of [0,1] values with non-negative
    # weights summing to 1.0; defensive clip to absorb FP noise so the
    # §4.4 [0.0, 1.0] precondition never trips on a 1.0 + eps.
    score = _clip(score, 0.0, 1.0)
    interpretation = interpret_transition_score(score, config.bands)
    return ComposedTransitionScore(
        score=score, interpretation=interpretation, components=components
    )
