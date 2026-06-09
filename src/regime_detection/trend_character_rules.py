"""v2 §1A Trend Character axis — classify layer.

V1+V2 raw-label walker (``raw_label_for_day`` / ``build_raw_outputs``).
The features layer lives in ``trend_character.py``.

Precedence (V2 §1A implementation decision #67 extension):
    breakout_expansion > recovery_attempt > trending > mild_trend >
    range_bound > chop > volatile_chop > transition > unknown
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_detection._rule_helpers import ev_float as _ev_float
from regime_detection.trend_character import TrendCharacterFeatures
from regime_shared.pandas_compat import require_single_session

# V2 §1A Trend Character (implementation decision #67 pin) extends the V1 5-label set.
# Precedence: breakout_expansion > recovery_attempt > trending > mild_trend >
# range_bound > chop > volatile_chop > transition > unknown.
TrendCharacterLabel = Literal[
    "breakout_expansion",
    "trending",
    "mild_trend",
    "recovery_attempt",
    "range_bound",
    "chop",
    "volatile_chop",
    "transition",
    "unknown",
]


# Per implementation decision #67 risk-rank extension. breakout_expansion shares rank 0
# with trending (both are "high-conviction directional" labels). range_bound
# shares rank 1 with recovery_attempt/chop (mid risk — calm but live regimes).
# transition/unknown remain at rank 2 (catch-all / cold-start).
_RISK_RANK: dict[TrendCharacterLabel, int] = {
    "trending": 0,
    "breakout_expansion": 0,
    "mild_trend": 0,
    "recovery_attempt": 1,
    "range_bound": 1,
    "chop": 1,
    "volatile_chop": 1,
    "transition": 2,
    "unknown": 2,
}


# Classify-side thresholds — consumed by the precedence walker.
_DEFAULT_FOLLOWTHROUGH_RATE_THRESHOLD = 0.60
_DEFAULT_RANGE_BOUND_RETURN_63D_THRESHOLD = 0.05
_DEFAULT_RANGE_BOUND_MIDPOINT_EXCURSION_THRESHOLD = 0.05
_DEFAULT_RANGE_BOUND_ADX_THRESHOLD = 20.0


def raw_label_for_day(
    f: TrendCharacterFeatures, dt: pd.Timestamp, *, allow_v2_labels: bool = True
) -> tuple[TrendCharacterLabel, dict[str, Any]]:
    # Guard: dt must resolve to exactly one session — a duplicate-date index would make
    # .loc[[dt]] return multiple rows and labels[0] silently mask the data issue.
    require_single_session(f.close.index, dt)
    day_features = TrendCharacterFeatures(
        close=f.close.loc[[dt]],
        sma_50=f.sma_50.loc[[dt]],
        return_10d=f.return_10d.loc[[dt]],
        return_21d=f.return_21d.loc[[dt]],
        prior_63d_drawdown=f.prior_63d_drawdown.loc[[dt]],
        adx_14=f.adx_14.loc[[dt]],
        return_63d=f.return_63d.loc[[dt]],
        midpoint_excursion_20d=f.midpoint_excursion_20d.loc[[dt]],
        breakout_20d_or_50d=f.breakout_20d_or_50d.loc[[dt]],
        bb_width_expanding=f.bb_width_expanding.loc[[dt]],
        volume_above_20d_average=f.volume_above_20d_average.loc[[dt]],
        followthrough_rate=f.followthrough_rate.loc[[dt]],
    )
    labels, evidence = build_raw_outputs(day_features, allow_v2_labels=allow_v2_labels)
    return labels[0], evidence[0]


def build_raw_outputs(
    f: TrendCharacterFeatures,
    *,
    allow_v2_labels: bool = True,
    followthrough_rate_threshold: float = _DEFAULT_FOLLOWTHROUGH_RATE_THRESHOLD,
    range_bound_return_63d_threshold: float = _DEFAULT_RANGE_BOUND_RETURN_63D_THRESHOLD,
    range_bound_midpoint_excursion_threshold: float = _DEFAULT_RANGE_BOUND_MIDPOINT_EXCURSION_THRESHOLD,
    range_bound_adx_threshold: float = _DEFAULT_RANGE_BOUND_ADX_THRESHOLD,
) -> tuple[list[TrendCharacterLabel], list[dict[str, Any]]]:
    close = f.close
    sma50 = f.sma_50
    ret10 = f.return_10d
    ret21 = f.return_21d
    ret63 = f.return_63d
    prior_dd = f.prior_63d_drawdown
    adx = f.adx_14
    midpoint_ex = f.midpoint_excursion_20d
    breakout_flag = f.breakout_20d_or_50d.fillna(False).astype(bool)
    bb_expanding = f.bb_width_expanding.fillna(False).astype(bool)
    vol_available = f.volume_above_20d_average.notna()
    vol_above = f.volume_above_20d_average.fillna(False).astype(bool)
    vol_gate = vol_above | ~vol_available
    ft_rate = f.followthrough_rate

    valid = ~(
        close.isna()
        | sma50.isna()
        | ret10.isna()
        | ret21.isna()
        | prior_dd.isna()
        | adx.isna()
    )

    recovery_attempt = valid & prior_dd.le(-0.10) & close.gt(sma50) & ret10.ge(0.05)
    trending = valid & adx.ge(20) & ret21.abs().ge(0.05)
    mild_trend = valid & adx.ge(20) & ret21.abs().lt(0.05)
    chop = valid & adx.lt(20) & ret10.abs().lt(0.03) & ret21.abs().lt(0.05)
    volatile_chop = valid & adx.lt(20) & ~chop & ~recovery_attempt

    breakout_expansion = (
        valid
        & breakout_flag
        & bb_expanding
        & vol_gate
        & ft_rate.notna()
        & ft_rate.ge(followthrough_rate_threshold)
    )
    range_bound = (
        valid
        & ret63.notna()
        & midpoint_ex.notna()
        & ret63.abs().lt(range_bound_return_63d_threshold)
        & midpoint_ex.le(range_bound_midpoint_excursion_threshold)
        & adx.lt(range_bound_adx_threshold)
    )
    if not allow_v2_labels:
        breakout_expansion = breakout_expansion & False
        mild_trend = mild_trend & False
        range_bound = range_bound & False
        volatile_chop = volatile_chop & False

    transition = valid & ~(
        breakout_expansion
        | recovery_attempt
        | trending
        | mild_trend
        | range_bound
        | chop
        | volatile_chop
    )

    labels = np.full(len(close), "unknown", dtype=object)
    labels[transition.to_numpy()] = "transition"
    labels[volatile_chop.to_numpy()] = "volatile_chop"
    labels[chop.to_numpy()] = "chop"
    labels[range_bound.to_numpy()] = "range_bound"
    labels[mild_trend.to_numpy()] = "mild_trend"
    labels[trending.to_numpy()] = "trending"
    labels[recovery_attempt.to_numpy()] = "recovery_attempt"
    labels[breakout_expansion.to_numpy()] = "breakout_expansion"

    evidence: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        if label == "unknown":
            evidence.append({"reason": "insufficient_history"})
            continue
        evidence.append(
            {
                "adx_14": _ev_float(adx.iat[idx]),
                "return_10d": _ev_float(ret10.iat[idx]),
                "return_21d": _ev_float(ret21.iat[idx]),
                "prior_63d_drawdown": _ev_float(prior_dd.iat[idx]),
                "recovery_attempt": bool(recovery_attempt.iat[idx]),
                "trending": bool(trending.iat[idx]),
                "mild_trend": bool(mild_trend.iat[idx]),
                "chop": bool(chop.iat[idx]),
                "volatile_chop": bool(volatile_chop.iat[idx]),
                "range_bound": bool(range_bound.iat[idx]),
                "breakout_expansion": bool(breakout_expansion.iat[idx]),
            }
        )

    return list(labels), evidence
