from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_detection._rolling_stats import period_return, simple_moving_average
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


# Spec-pinned constants (implementation decision #46/#47 + §1A formulas at spec lines
# 130-157, 175-179). Mirrored by yaml defaults in TrendCharacterV2Config.
# The classifier reads these defaults when no v2 config is threaded through;
# yaml may retune via §9.1 walk-forward.
_DEFAULT_FOLLOWTHROUGH_RATE_THRESHOLD = 0.60
_DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS = 504
_DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT = 20
_DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS = 5
_DEFAULT_BB_WIDTH_EXPANDING_LOOKBACK = 5
_DEFAULT_BB_WIDTH_PERIOD = 20
_DEFAULT_BB_WIDTH_MULTIPLIER = 2.0
_DEFAULT_RANGE_BOUND_RETURN_63D_THRESHOLD = 0.05
_DEFAULT_RANGE_BOUND_MIDPOINT_EXCURSION_THRESHOLD = 0.05
_DEFAULT_RANGE_BOUND_ADX_THRESHOLD = 20.0


@dataclass(frozen=True)
class TrendCharacterFeatures:
    close: pd.Series
    sma_50: pd.Series
    return_10d: pd.Series
    return_21d: pd.Series
    prior_63d_drawdown: pd.Series
    adx_14: pd.Series
    # V2 §1A Trend Character inputs (implementation decision #46/#47/#67).
    return_63d: pd.Series
    midpoint_excursion_20d: pd.Series
    breakout_20d_or_50d: pd.Series
    bb_width_expanding: pd.Series
    volume_above_20d_average: pd.Series
    followthrough_rate: pd.Series


def _ev_float(x: float) -> float:
    return round(float(x), 8)


def _wilder_ewm(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _compute_adx_14(*, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index)
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=close.index
    )

    n = 14
    atr = _wilder_ewm(tr, n)
    atr_safe = atr.replace(0.0, np.nan)
    plus_di = 100 * _wilder_ewm(plus_dm, n) / atr_safe
    minus_di = 100 * _wilder_ewm(minus_dm, n) / atr_safe
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    return _wilder_ewm(dx, n)


def _compute_midpoint_excursion_20d(close: pd.Series) -> pd.Series:
    """v2 §1A lines 175-176:
    midpoint_20d[t] = (max(close[t-19..t]) + min(close[t-19..t])) / 2
    excursion[t]    = max(|close[i] - midpoint_20d[t]| / midpoint_20d[t])
                      for i in t-19..t
    """
    window = 20
    rolling_max = close.rolling(window).max()
    rolling_min = close.rolling(window).min()
    midpoint = (rolling_max + rolling_min) / 2.0
    # Excursion within the same 20d window using the window's midpoint as anchor.
    # Approximated via (max - mid) and (mid - min) — equivalent to the max over
    # the window since close[i] is in [min, max] and |c - mid|/mid is monotone
    # at the extremes when mid > 0.
    upper_dev = (rolling_max - midpoint).abs()
    lower_dev = (midpoint - rolling_min).abs()
    raw_dev = pd.concat([upper_dev, lower_dev], axis=1).max(axis=1)
    return raw_dev / midpoint.replace(0.0, np.nan)


def _compute_breakout_20d_or_50d(close: pd.Series) -> pd.Series:
    """Strict, prior-window-exclusive breakout flag (v2 §1A lines 137-140)."""
    prior_max_20 = close.shift(1).rolling(20).max()
    prior_max_50 = close.shift(1).rolling(50).max()
    flag = (close > prior_max_20) | (close > prior_max_50)
    # Mask invalid early sessions (NaN max → cannot determine).
    valid = ~(prior_max_20.isna() & prior_max_50.isna())
    return flag.where(valid, False)


def _compute_bb_width_expanding(
    close: pd.Series,
    *,
    period: int = _DEFAULT_BB_WIDTH_PERIOD,
    multiplier: float = _DEFAULT_BB_WIDTH_MULTIPLIER,
    lookback: int = _DEFAULT_BB_WIDTH_EXPANDING_LOOKBACK,
) -> pd.Series:
    """Textbook BB width: 2 * multiplier * std(close[t-period+1..t], ddof=0).
    For multiplier=2 this equals 4 * std (v2 §1A line 143)."""
    std = close.rolling(period).std(ddof=0)
    bb_width = 2.0 * multiplier * std
    return bb_width > bb_width.shift(lookback)


def _compute_volume_above_20d_average(volume: pd.Series) -> pd.Series:
    prior_mean = volume.shift(1).rolling(20).mean()
    return (volume > prior_mean).where(~prior_mean.isna(), False)


def _compute_followthrough_rate(
    close: pd.Series,
    breakout_20d_or_50d: pd.Series,
    *,
    lookback_sessions: int = _DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
    window_count: int = _DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
    hold_sessions: int = _DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
) -> pd.Series:
    """v2 §1A lines 151-157 + implementation decision #47 followthrough_rate pin.

    Walk backwards through history collecting up to `window_count` past
    sessions where breakout_20d_or_50d fired. For each, "held" iff every
    session in b+1..b+hold_sessions has close > breakout_level[b], where
    breakout_level is the prior-window max that close[b] crossed (20d
    preferred, else 50d). Cap lookback at `lookback_sessions`. If fewer than
    `window_count` breakouts in trailing window → NaN (cold-start).
    """
    n = len(close)
    close_arr = close.to_numpy(dtype=float)
    is_breakout = breakout_20d_or_50d.fillna(False).to_numpy(dtype=bool)

    # Precompute prior-window maxima (exclusive of session b).
    prior_max_20 = close.shift(1).rolling(20).max().to_numpy(dtype=float)
    prior_max_50 = close.shift(1).rolling(50).max().to_numpy(dtype=float)

    # breakout_level[b]: the level that close[b] crossed at b.
    breakout_level = np.full(n, np.nan, dtype=float)
    for b in range(n):
        if not is_breakout[b]:
            continue
        m20 = prior_max_20[b]
        m50 = prior_max_50[b]
        # Tie-break (F-055, pinned): when close[b] crosses BOTH the 20d and 50d
        # prior-window maxima (m50 >= m20 always, so close > m50 implies close > m20),
        # the §1A phrase "the max-of-prior-window that close crossed" is ambiguous.
        # We deterministically choose the 20d level — the lower, more lenient hold
        # bar — so a shallow follow-through still counts as held. test_followthrough_
        # rate_breakout_level_tie_break_prefers_20d locks this against regression.
        if not np.isnan(m20) and close_arr[b] > m20:
            breakout_level[b] = m20
        elif not np.isnan(m50) and close_arr[b] > m50:
            breakout_level[b] = m50

    # held[b]: True iff close[b+1..b+hold_sessions] strictly > breakout_level[b].
    held = np.zeros(n, dtype=bool)
    for b in range(n):
        if np.isnan(breakout_level[b]):
            continue
        end = b + hold_sessions
        if end >= n:
            # Cannot determine hold for breakouts too close to the end of the
            # series. Treat as not-held for followthrough_rate accounting per
            # spec — they aren't yet validated.
            continue
        level = breakout_level[b]
        if np.all(close_arr[b + 1 : end + 1] > level):
            held[b] = True

    # Vectorized lookup: index every breakout, precompute a cumulative sum of
    # held breakouts, then for each session t fetch the most-recent
    # window_count breakouts strictly before t via bisect_right and the
    # held_count via two cumulative-sum reads.
    #
    # The lookback bound is enforced by a single check on the oldest element
    # of the window — older breakouts than `t - lookback_sessions` were
    # unreachable in the original backward walk, so requiring
    # breakout_idx[k - window_count] >= t - lookback_sessions reproduces the
    # original semantics exactly.
    out = np.full(n, np.nan, dtype=float)
    breakout_idx = np.flatnonzero(~np.isnan(breakout_level))
    if breakout_idx.size >= window_count:
        held_at_breakouts = held[breakout_idx].astype(np.int64)
        held_cum = np.empty(breakout_idx.size + 1, dtype=np.int64)
        held_cum[0] = 0
        np.cumsum(held_at_breakouts, out=held_cum[1:])
        breakout_idx_list = breakout_idx.tolist()
        for t in range(n):
            k = bisect_right(breakout_idx_list, t - 1)
            if k < window_count:
                continue
            j = k - window_count
            if breakout_idx_list[j] < t - lookback_sessions:
                continue
            out[t] = (held_cum[k] - held_cum[j]) / window_count
    return pd.Series(out, index=close.index)


def compute_features(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series | None = None,
    bb_width_period: int = _DEFAULT_BB_WIDTH_PERIOD,
    bb_width_multiplier: float = _DEFAULT_BB_WIDTH_MULTIPLIER,
    bb_width_expanding_lookback: int = _DEFAULT_BB_WIDTH_EXPANDING_LOOKBACK,
    followthrough_lookback_sessions: int = _DEFAULT_FOLLOWTHROUGH_LOOKBACK_SESSIONS,
    followthrough_window_count: int = _DEFAULT_FOLLOWTHROUGH_WINDOW_COUNT,
    followthrough_hold_sessions: int = _DEFAULT_FOLLOWTHROUGH_HOLD_SESSIONS,
) -> TrendCharacterFeatures:
    sma_50 = simple_moving_average(close, window=50)
    return_10d = period_return(close, periods=10)
    return_21d = period_return(close, periods=21)
    return_63d = period_return(close, periods=63)
    prior_63d_drawdown = close / close.rolling(63).max() - 1
    adx_14 = _compute_adx_14(high=high, low=low, close=close)

    midpoint_excursion_20d = _compute_midpoint_excursion_20d(close)
    breakout_20d_or_50d = _compute_breakout_20d_or_50d(close)
    bb_width_expanding = _compute_bb_width_expanding(
        close,
        period=bb_width_period,
        multiplier=bb_width_multiplier,
        lookback=bb_width_expanding_lookback,
    )
    if volume is None:
        volume_above_20d_average = pd.Series(False, index=close.index)
    else:
        volume_above_20d_average = _compute_volume_above_20d_average(volume)
    followthrough_rate = _compute_followthrough_rate(
        close,
        breakout_20d_or_50d,
        lookback_sessions=followthrough_lookback_sessions,
        window_count=followthrough_window_count,
        hold_sessions=followthrough_hold_sessions,
    )

    return TrendCharacterFeatures(
        close=close,
        sma_50=sma_50,
        return_10d=return_10d,
        return_21d=return_21d,
        prior_63d_drawdown=prior_63d_drawdown,
        adx_14=adx_14,
        return_63d=return_63d,
        midpoint_excursion_20d=midpoint_excursion_20d,
        breakout_20d_or_50d=breakout_20d_or_50d,
        bb_width_expanding=bb_width_expanding,
        volume_above_20d_average=volume_above_20d_average,
        followthrough_rate=followthrough_rate,
    )


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
    vol_above = f.volume_above_20d_average.fillna(False).astype(bool)
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
        & vol_above
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
