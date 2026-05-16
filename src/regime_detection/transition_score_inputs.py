from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionScoreInputs:
    """Per-session v2 §4.2 inputs for :func:`compose_transition_score_for_session`.

    All numeric fields are ``float`` (NaN is used to signal missing / cold-start).
    ``event_calendar_label`` is the string active label from
    :class:`~regime_detection.models.EventCalendarOutput`.
    """

    realized_vol_short: float
    realized_vol_long: float
    pct_above_50dma: float
    avg_pairwise_corr_percentile_504d: float
    drawdown_252d: float
    event_calendar_label: str
    hmm_top_state_prob_now: float
    hmm_top_state_prob_5d_ago: float
    change_point_score: float
