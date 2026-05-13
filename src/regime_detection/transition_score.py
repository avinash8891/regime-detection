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
    hmm_probability_shift_score: float | None = None,
    change_point_score: float | None = None,
) -> float:
    """v2 §4.3 weighted composite (Ambiguity Log #66 4-table system).

    The ``weights`` dict drives both the keys composed and the per-key
    weights. Valid keys are the canonical 7 components:
    ``volatility_acceleration``, ``breadth_deterioration``,
    ``correlation_concentration``, ``trend_break``, ``macro_event``,
    ``hmm_probability_shift``, ``change_point``. Unknown keys raise
    ``ValueError``. Any optional component referenced by ``weights`` must
    have a non-None value supplied.
    """
    values: dict[str, float | None] = {
        "volatility_acceleration": volatility_acceleration_score,
        "breadth_deterioration": breadth_deterioration_score,
        "correlation_concentration": correlation_concentration_score,
        "trend_break": trend_break_score,
        "macro_event": macro_event_score,
        "hmm_probability_shift": hmm_probability_shift_score,
        "change_point": change_point_score,
    }
    for key in weights:
        if key not in values:
            raise ValueError(
                f"Unknown component in weights: {key!r}. "
                f"Valid components: {sorted(values.keys())}"
            )
        if values[key] is None:
            raise ValueError(
                f"Weight references {key!r} but no value was provided"
            )
    return sum(weight * float(values[key]) for key, weight in weights.items())


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
    hmm_top_state_prob_now: float | None = None,
    hmm_top_state_prob_5d_ago: float | None = None,
    change_point_score: float | None = None,
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

    # v2 §4.2 line 2396 + §6.1 (Slice 6) — 6th component when both HMM
    # probabilities are present and non-NaN. Permutation-invariant
    # |top_state_prob[t] - top_state_prob[t-5]|, defensively clipped to
    # [0, 1] (the formula is already in-range by construction since
    # probabilities ∈ [0, 1]).
    hmm_shift: float | None = None
    if (
        hmm_top_state_prob_now is not None
        and hmm_top_state_prob_5d_ago is not None
        and not math.isnan(float(hmm_top_state_prob_now))
        and not math.isnan(float(hmm_top_state_prob_5d_ago))
    ):
        hmm_shift = _clip(
            abs(float(hmm_top_state_prob_now) - float(hmm_top_state_prob_5d_ago)),
            0.0,
            1.0,
        )

    # Ambiguity Log #66 — 7th component change_point_score wired in.
    # `change_point.score` is already a posterior probability ∈ [0, 1] by
    # construction (5-session rolling max per Log #64); no clip needed but
    # we defensively clip to absorb any FP edge case at the seam.
    cp_score: float | None = None
    if change_point_score is not None and not math.isnan(float(change_point_score)):
        cp_score = _clip(float(change_point_score), 0.0, 1.0)

    # Log #66 — 4-table weight selection gated on (hmm_present, cp_present).
    if hmm_shift is not None and cp_score is not None:
        weights = config.weights_with_hmm_with_change_point
    elif hmm_shift is not None:
        weights = config.weights_with_hmm
    elif cp_score is not None:
        weights = config.weights_with_change_point
    else:
        weights = config.weights_without_hmm

    # Build components dict to mirror the selected weight table (no
    # spurious keys whose weight wasn't selected).
    if "hmm_probability_shift" in weights:
        components["hmm_probability_shift"] = hmm_shift  # type: ignore[assignment]
    if "change_point" in weights:
        components["change_point"] = cp_score  # type: ignore[assignment]

    score = compute_transition_score(
        volatility_acceleration_score=vol_acc,
        breadth_deterioration_score=breadth_det,
        correlation_concentration_score=corr_conc,
        trend_break_score=trend_break,
        macro_event_score=macro_event,
        weights=weights,
        hmm_probability_shift_score=hmm_shift,
        change_point_score=cp_score,
    )
    # Composite is a weighted average of [0,1] values with non-negative
    # weights summing to 1.0; defensive clip to absorb FP noise so the
    # §4.4 [0.0, 1.0] precondition never trips on a 1.0 + eps.
    score = _clip(score, 0.0, 1.0)
    interpretation = interpret_transition_score(score, config.bands)
    return ComposedTransitionScore(
        score=score, interpretation=interpretation, components=components
    )
