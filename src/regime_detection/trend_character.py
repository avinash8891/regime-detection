from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_detection._rolling_stats import period_return, simple_moving_average
from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.models import AxisOutput, DataQuality


# V2 §1B (documented implementation decision) extends the V1 5-label set with
# `breakout_expansion` and `range_bound`. Members ordered by precedence:
# breakout_expansion > recovery_attempt > trending > range_bound > chop >
# transition > unknown.
TrendCharacterLabel = Literal[
    "breakout_expansion",
    "trending",
    "recovery_attempt",
    "range_bound",
    "chop",
    "transition",
    "unknown",
]


# Per documented implementation decision. breakout_expansion shares rank 0 with trending (both are
# "high-conviction directional" labels). range_bound shares rank 1 with
# recovery_attempt/chop (mid risk — calm but live regimes). transition/unknown
# remain at rank 2 (catch-all / cold-start).
_RISK_RANK: dict[TrendCharacterLabel, int] = {
    "trending": 0,
    "breakout_expansion": 0,
    "recovery_attempt": 1,
    "range_bound": 1,
    "chop": 1,
    "transition": 2,
    "unknown": 2,
}


# Spec-pinned constants (documented implementation decision). Mirrored by yaml defaults in
# TrendCharacterV2Config. The classifier reads these defaults when no v2
# config is threaded through; yaml may retune via §9.1 walk-forward.
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
    high: pd.Series
    low: pd.Series
    sma_50: pd.Series
    return_10d: pd.Series
    return_21d: pd.Series
    prior_63d_drawdown: pd.Series
    adx_14: pd.Series
    # V2 §1B (documented implementation decision).
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
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)

    n = 14
    atr = _wilder_ewm(tr, n)
    atr_safe = atr.replace(0.0, np.nan)
    plus_di = 100 * _wilder_ewm(plus_dm, n) / atr_safe
    minus_di = 100 * _wilder_ewm(minus_dm, n) / atr_safe
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    return _wilder_ewm(dx, n)


def _compute_midpoint_excursion_20d(close: pd.Series) -> pd.Series:
    """v2 §1B lines 132-138:
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
    """Strict, prior-window-exclusive breakout flag (spec line 97-99)."""
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
    For multiplier=2 this equals 4 * std (spec line 102)."""
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
    """v2 §1B lines 110-116 + documented implementation decision.

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
        # Prefer the 20d window if close crossed it; else use the 50d level.
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

    # Now compute followthrough_rate per session t: walk backwards from t-1
    # up to lookback_sessions, collect up to window_count breakouts, compute
    # held_count / window_count. If <window_count → NaN.
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        collected = 0
        held_count = 0
        start = max(0, t - lookback_sessions)
        for b in range(t - 1, start - 1, -1):
            if np.isnan(breakout_level[b]):
                continue
            collected += 1
            if held[b]:
                held_count += 1
            if collected >= window_count:
                break
        if collected >= window_count:
            out[t] = held_count / window_count
    return pd.Series(out, index=close.index)


def compute_features(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series | None = None,
) -> TrendCharacterFeatures:
    sma_50 = simple_moving_average(close, window=50)
    return_10d = period_return(close, periods=10)
    return_21d = period_return(close, periods=21)
    return_63d = period_return(close, periods=63)
    prior_63d_drawdown = close / close.rolling(63).max() - 1
    adx_14 = _compute_adx_14(high=high, low=low, close=close)

    midpoint_excursion_20d = _compute_midpoint_excursion_20d(close)
    breakout_20d_or_50d = _compute_breakout_20d_or_50d(close)
    bb_width_expanding = _compute_bb_width_expanding(close)
    if volume is None:
        # Volume not threaded through — fall back to an all-False mask. The
        # breakout_expansion rule cannot fire without volume confirmation.
        volume_above_20d_average = pd.Series(False, index=close.index)
    else:
        volume_above_20d_average = _compute_volume_above_20d_average(volume)
    followthrough_rate = _compute_followthrough_rate(close, breakout_20d_or_50d)

    return TrendCharacterFeatures(
        close=close,
        high=high,
        low=low,
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
    close = f.close.loc[dt]
    sma50 = f.sma_50.loc[dt]
    ret10 = f.return_10d.loc[dt]
    ret21 = f.return_21d.loc[dt]
    ret63 = f.return_63d.loc[dt]
    prior_dd = f.prior_63d_drawdown.loc[dt]
    adx = f.adx_14.loc[dt]

    # V1 inputs gate the V1 labels.
    if any(pd.isna(x) for x in [close, sma50, ret10, ret21, prior_dd, adx]):
        return "unknown", {"reason": "insufficient_history"}

    # V1 labels (preserved verbatim).
    recovery_attempt = bool((prior_dd <= -0.10) and (close > sma50) and (ret10 >= 0.05))
    trending = bool((adx >= 20) and (abs(ret21) >= 0.05))
    chop = bool((adx < 20) and (abs(ret10) < 0.03) and (abs(ret21) < 0.05))

    # V2 §1B labels. Cold-start: any NaN input falsifies the rule.
    midpoint_ex = f.midpoint_excursion_20d.loc[dt]
    breakout_flag = f.breakout_20d_or_50d.loc[dt]
    bb_expanding = f.bb_width_expanding.loc[dt]
    vol_above = f.volume_above_20d_average.loc[dt]
    ft_rate = f.followthrough_rate.loc[dt]

    breakout_expansion = bool(
        allow_v2_labels
        and not pd.isna(ft_rate)
        and bool(breakout_flag)
        and bool(bb_expanding)
        and bool(vol_above)
        and ft_rate >= _DEFAULT_FOLLOWTHROUGH_RATE_THRESHOLD
    )
    range_bound = bool(
        allow_v2_labels
        and (not pd.isna(ret63))
        and (not pd.isna(midpoint_ex))
        and abs(ret63) < _DEFAULT_RANGE_BOUND_RETURN_63D_THRESHOLD
        and midpoint_ex <= _DEFAULT_RANGE_BOUND_MIDPOINT_EXCURSION_THRESHOLD
        and adx < _DEFAULT_RANGE_BOUND_ADX_THRESHOLD
    )

    # Precedence (documented implementation decision):
    # breakout_expansion > recovery_attempt > trending > range_bound > chop >
    # transition > unknown.
    if breakout_expansion:
        label: TrendCharacterLabel = "breakout_expansion"
    elif recovery_attempt:
        label = "recovery_attempt"
    elif trending:
        label = "trending"
    elif range_bound:
        label = "range_bound"
    elif chop:
        label = "chop"
    else:
        label = "transition"

    return label, {
        "adx_14": _ev_float(adx),
        "return_10d": _ev_float(ret10),
        "return_21d": _ev_float(ret21),
        "prior_63d_drawdown": _ev_float(prior_dd),
        "recovery_attempt": recovery_attempt,
        "trending": trending,
        "chop": chop,
        "range_bound": range_bound,
        "breakout_expansion": breakout_expansion,
    }


def build_raw_outputs(
    f: TrendCharacterFeatures, *, allow_v2_labels: bool = True
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

    valid = ~(close.isna() | sma50.isna() | ret10.isna() | ret21.isna() | prior_dd.isna() | adx.isna())

    recovery_attempt = valid & prior_dd.le(-0.10) & close.gt(sma50) & ret10.ge(0.05)
    trending = valid & adx.ge(20) & ret21.abs().ge(0.05)
    chop = valid & adx.lt(20) & ret10.abs().lt(0.03) & ret21.abs().lt(0.05)

    breakout_expansion = (
        valid
        & breakout_flag
        & bb_expanding
        & vol_above
        & ft_rate.notna()
        & ft_rate.ge(_DEFAULT_FOLLOWTHROUGH_RATE_THRESHOLD)
    )
    range_bound = (
        valid
        & ret63.notna()
        & midpoint_ex.notna()
        & ret63.abs().lt(_DEFAULT_RANGE_BOUND_RETURN_63D_THRESHOLD)
        & midpoint_ex.le(_DEFAULT_RANGE_BOUND_MIDPOINT_EXCURSION_THRESHOLD)
        & adx.lt(_DEFAULT_RANGE_BOUND_ADX_THRESHOLD)
    )
    if not allow_v2_labels:
        breakout_expansion = breakout_expansion & False
        range_bound = range_bound & False

    transition = valid & ~(
        breakout_expansion | recovery_attempt | trending | range_bound | chop
    )

    labels = np.full(len(close), "unknown", dtype=object)
    labels[transition.to_numpy()] = "transition"
    labels[chop.to_numpy()] = "chop"
    labels[range_bound.to_numpy()] = "range_bound"
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
                "chop": bool(chop.iat[idx]),
                "range_bound": bool(range_bound.iat[idx]),
                "breakout_expansion": bool(breakout_expansion.iat[idx]),
            }
        )

    return list(labels), evidence


def classify_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    as_of_date: date,
    escalation_days: int = 1,
    deescalation_days: int,
    volume: pd.Series | None = None,
    allow_v2_labels: bool = False,
) -> AxisOutput:
    """Per-day raw/stable/active trend_character labels for ``as_of_date``.

    ``allow_v2_labels`` defaults to ``False`` so the per-axis API is V1-safe:
    direct callers cannot accidentally surface the V2-only `range_bound` /
    `breakout_expansion` labels (the latter requires volume confirmation,
    but `range_bound` does not — leaving it as a leak surface for V1-only
    consumers). The engine path (`axis_series.build_axis_series_bundle`)
    bypasses this function and threads V2 labels through `build_raw_outputs`
    explicitly when the V2 config is active. Set to ``True`` only when the
    caller is intentionally in V2 mode AND has confirmed it consumes V2
    labels.
    """
    dt = pd.Timestamp(as_of_date)
    close = close.copy()
    high = high.copy()
    low = low.copy()
    close.index = pd.to_datetime(close.index)
    high.index = pd.to_datetime(high.index)
    low.index = pd.to_datetime(low.index)
    if volume is not None:
        volume = volume.copy()
        volume.index = pd.to_datetime(volume.index)

    if dt not in close.index:
        raise ValueError(f"as_of_date missing from close series: {as_of_date.isoformat()}")

    close = close.loc[:dt]
    high = high.loc[:dt]
    low = low.loc[:dt]
    if volume is not None:
        volume = volume.loc[:dt]

    f = compute_features(close=close, high=high, low=low, volume=volume)
    raw_labels: list[TrendCharacterLabel] = []
    raw_evidence: list[dict[str, Any]] = []
    for day in close.index:
        lbl, ev = raw_label_for_day(f, day, allow_v2_labels=allow_v2_labels)
        raw_labels.append(lbl)
        raw_evidence.append(ev)

    stable_labels, active_labels = apply_asymmetric_hysteresis(
        raw_labels=raw_labels,
        risk_rank=_RISK_RANK,
        escalation_days=escalation_days,
        deescalation_days=deescalation_days,
    )

    raw = raw_labels[-1]
    stable = stable_labels[-1]
    active = active_labels[-1]

    if raw == "unknown":
        dq = DataQuality(status="insufficient_history", freshness_days=0, completeness=1.0, reason="insufficient_history")
    else:
        dq = DataQuality(status="ok", freshness_days=0, completeness=1.0, reason=None)

    return AxisOutput(
        raw_label=raw,
        stable_label=stable,
        active_label=active,
        evidence={
            "rule_evidence": raw_evidence[-1],
            "risk_rank": _RISK_RANK,
            "deescalation_days": deescalation_days,
        },
        data_quality=dq,
    )
