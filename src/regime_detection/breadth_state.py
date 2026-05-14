from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd

from regime_detection.data_quality import assess_series_input_quality
from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.models import BreadthStateOutput, DataQuality


# V2 §1D (Ambiguity Log #21-#26, #68, #69, #70) extends the V1 5-label set
# with four PIT-derived labels. Members ordered by precedence (spec line 284):
#   breadth_thrust > divergent_fragile > narrowing_breadth > recovery_breadth >
#   broadening_breadth > weak_breadth > healthy_breadth > neutral_breadth >
#   unknown.
BreadthLabel = Literal[
    "breadth_thrust",
    "divergent_fragile",
    "narrowing_breadth",
    "recovery_breadth",
    "broadening_breadth",
    "weak_breadth",
    "healthy_breadth",
    "neutral_breadth",
    "unknown",
]


# Deteriorating labels get higher risk-rank. breadth_thrust is a bullish
# initiation signal so it co-ranks with healthy_breadth (rank 0); recovery_breadth
# is mid-recovery so it co-ranks with neutral_breadth (rank 1); broadening_breadth
# is a benign recovery confirmation (rank 0). narrowing_breadth is deterioration
# at mid-severity (rank 2, same as weak_breadth). divergent_fragile remains the
# highest-risk V1 label at rank 3.
_RISK_RANK: dict[BreadthLabel, int] = {
    "breadth_thrust": 0,        # bullish initiation (Log #69)
    "healthy_breadth": 0,
    "broadening_breadth": 0,    # V2 recovery confirmation (Log #21-#26)
    "neutral_breadth": 1,
    "recovery_breadth": 1,      # mid-recovery (Log #70)
    "weak_breadth": 2,
    "narrowing_breadth": 2,     # V2 deterioration — mid-severity
    "divergent_fragile": 3,
    "unknown": 2,
}


@dataclass(frozen=True)
class BreadthFeatures:
    spy_close: pd.Series
    rsp_close: pd.Series
    relative_breadth_ratio: pd.Series
    relative_breadth_sma50: pd.Series
    relative_breadth_return_20d: pd.Series
    index_distance_from_63d_high: pd.Series


def compute_features(*, spy_close: pd.Series, rsp_close: pd.Series) -> BreadthFeatures:
    ratio = rsp_close / spy_close
    ratio_sma50 = ratio.rolling(50).mean()
    ratio_ret20 = ratio / ratio.shift(20) - 1
    idx_dist = spy_close / spy_close.rolling(63, min_periods=50).max() - 1
    return BreadthFeatures(
        spy_close=spy_close,
        rsp_close=rsp_close,
        relative_breadth_ratio=ratio,
        relative_breadth_sma50=ratio_sma50,
        relative_breadth_return_20d=ratio_ret20,
        index_distance_from_63d_high=idx_dist,
    )


def raw_label_for_day(f: BreadthFeatures, dt: pd.Timestamp) -> tuple[BreadthLabel, dict[str, Any]]:
    ratio = f.relative_breadth_ratio.loc[dt]
    ratio_sma = f.relative_breadth_sma50.loc[dt]
    ratio_ret20 = f.relative_breadth_return_20d.loc[dt]
    idx_dist = f.index_distance_from_63d_high.loc[dt]

    if any(pd.isna(x) for x in [ratio, ratio_sma, ratio_ret20, idx_dist]):
        return "unknown", {"reason": "insufficient_history"}

    divergent_fragile = bool((idx_dist >= -0.05) and (ratio < ratio_sma) and (ratio_ret20 <= -0.03))
    weak_breadth = bool((ratio < ratio_sma) and (ratio_ret20 < 0))
    healthy_breadth = bool((ratio > ratio_sma) and (ratio_ret20 >= 0))
    neutral_breadth = not (divergent_fragile or weak_breadth or healthy_breadth)

    if divergent_fragile:
        label: BreadthLabel = "divergent_fragile"
    elif weak_breadth:
        label = "weak_breadth"
    elif healthy_breadth:
        label = "healthy_breadth"
    else:
        label = "neutral_breadth"

    return label, {
        "divergent_fragile": divergent_fragile,
        "weak_breadth": weak_breadth,
        "healthy_breadth": healthy_breadth,
        "neutral_breadth": neutral_breadth,
    }


def build_raw_outputs(f: BreadthFeatures) -> tuple[list[BreadthLabel], list[dict[str, Any]]]:
    ratio = f.relative_breadth_ratio
    ratio_sma = f.relative_breadth_sma50
    ratio_ret20 = f.relative_breadth_return_20d
    idx_dist = f.index_distance_from_63d_high

    valid = ~(ratio.isna() | ratio_sma.isna() | ratio_ret20.isna() | idx_dist.isna())
    divergent_fragile = valid & idx_dist.ge(-0.05) & ratio.lt(ratio_sma) & ratio_ret20.le(-0.03)
    weak_breadth = valid & ratio.lt(ratio_sma) & ratio_ret20.lt(0)
    healthy_breadth = valid & ratio.gt(ratio_sma) & ratio_ret20.ge(0)
    neutral_breadth = valid & ~(divergent_fragile | weak_breadth | healthy_breadth)

    labels = pd.Series("unknown", index=ratio.index, dtype="object")
    labels.loc[neutral_breadth] = "neutral_breadth"
    labels.loc[healthy_breadth] = "healthy_breadth"
    labels.loc[weak_breadth] = "weak_breadth"
    labels.loc[divergent_fragile] = "divergent_fragile"

    evidence: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        if label == "unknown":
            evidence.append({"reason": "insufficient_history"})
            continue
        evidence.append(
            {
                "divergent_fragile": bool(divergent_fragile.iat[idx]),
                "weak_breadth": bool(weak_breadth.iat[idx]),
                "healthy_breadth": bool(healthy_breadth.iat[idx]),
                "neutral_breadth": bool(neutral_breadth.iat[idx]),
            }
        )

    return list(labels), evidence


def classify_series(
    *,
    spy_close: pd.Series,
    rsp_close: pd.Series,
    as_of_date: date,
    deescalation_days: int,
) -> BreadthStateOutput:
    dt = pd.Timestamp(as_of_date)
    spy_close = spy_close.copy()
    rsp_close = rsp_close.copy()
    spy_close.index = pd.to_datetime(spy_close.index)
    rsp_close.index = pd.to_datetime(rsp_close.index)
    spy_close = spy_close.sort_index()
    rsp_close = rsp_close.sort_index()

    if dt not in spy_close.index:
        raise ValueError(f"as_of_date missing from SPY close series: {as_of_date.isoformat()}")

    spy_close = spy_close.loc[:dt]
    rsp_close = rsp_close.loc[:dt].reindex(spy_close.index)

    f = compute_features(spy_close=spy_close, rsp_close=rsp_close)
    raw_labels: list[BreadthLabel] = []
    raw_evidence: list[dict[str, Any]] = []
    for day in spy_close.index:
        lbl, ev = raw_label_for_day(f, day)
        raw_labels.append(lbl)
        raw_evidence.append(ev)

    stable_labels, active_labels = apply_asymmetric_hysteresis(
        raw_labels=raw_labels,
        risk_rank=_RISK_RANK,
        deescalation_days=deescalation_days,
    )

    raw = raw_labels[-1]
    stable = stable_labels[-1]
    active = active_labels[-1]

    if raw == "unknown":
        return BreadthStateOutput(
            mode="etf_proxy",
            raw_label="unknown",
            stable_label="unknown",
            active_label="unknown",
            evidence={"reason": "insufficient_history", "proxy": "RSP/SPY"},
            data_quality=DataQuality(
                status="insufficient_history",
                freshness_days=None,
                completeness=None,
                reason="required_feature_is_nan",
            ),
        )

    dq = _data_quality_for_asof(
        spy_close=spy_close,
        rsp_close=rsp_close.reindex(spy_close.index),
        as_of_date=as_of_date,
        required_trading_days=50,
        raw_label=raw,
        max_freshness_days=3,
        min_completeness=0.90,
    )

    return BreadthStateOutput(
        mode="etf_proxy",
        raw_label=raw,
        stable_label=stable,
        active_label=active,
        evidence={
            "proxy": "RSP/SPY",
            "rule_evidence": raw_evidence[-1],
            "risk_rank": _RISK_RANK,
            "deescalation_days": deescalation_days,
        },
        data_quality=dq,
    )


# ---------------------------------------------------------------------------
# V2 §1D rule predicates (Ambiguity Log #21-#26, #68). The two predicates that
# ship today read the PIT-aware features from FeatureStore.breadth_state_v2
# and gate on strict 5-session rate-of-change (Log #68 pin: "rising"/"falling"
# = strict change over `label_rate_of_change_lookback_sessions` sessions).
#
# Inputs at `t` AND at `t - lookback_sessions` must both be non-NaN; any NaN
# endpoint short-circuits the predicate to False (no V2 label fires on the
# cold-start window).
# ---------------------------------------------------------------------------


def _lookback_endpoint_values(
    series: pd.Series, *, dt: pd.Timestamp, lookback_sessions: int
) -> tuple[float, float] | None:
    """Return (value_at_t_minus_lookback, value_at_t) iff both are non-NaN AND
    the lookback position is reachable; otherwise None.
    """
    if dt not in series.index:
        return None
    pos_now = series.index.get_loc(dt)
    if isinstance(pos_now, slice) or pos_now < lookback_sessions:
        return None
    val_now = series.iloc[pos_now]
    val_then = series.iloc[pos_now - lookback_sessions]
    if pd.isna(val_now) or pd.isna(val_then):
        return None
    return float(val_then), float(val_now)


def _evaluate_narrowing_breadth(
    pct_above_50dma: pd.Series,
    pct_above_200dma: pd.Series,
    nh_nl_ratio: pd.Series,
    *,
    dt: pd.Timestamp,
    lookback_sessions: int,
    nh_nl_threshold: float,
) -> bool:
    """v2 §1D line 280 — narrowing_breadth predicate.

    Fires iff:
      pct_above_50dma is FALLING over `lookback_sessions` (strict decrease)
      AND pct_above_200dma is FALLING over `lookback_sessions`
      AND nh_nl_ratio at `dt` < nh_nl_threshold (default 0.4).

    "Falling" = strict 5-session decrease per Ambiguity Log #68.
    """
    pct50_pts = _lookback_endpoint_values(
        pct_above_50dma, dt=dt, lookback_sessions=lookback_sessions
    )
    pct200_pts = _lookback_endpoint_values(
        pct_above_200dma, dt=dt, lookback_sessions=lookback_sessions
    )
    if pct50_pts is None or pct200_pts is None:
        return False
    if dt not in nh_nl_ratio.index:
        return False
    nh_nl_now = nh_nl_ratio.loc[dt]
    if pd.isna(nh_nl_now):
        return False

    pct50_then, pct50_now = pct50_pts
    pct200_then, pct200_now = pct200_pts
    return bool(
        pct50_now < pct50_then
        and pct200_now < pct200_then
        and float(nh_nl_now) < nh_nl_threshold
    )


def _evaluate_broadening_breadth(
    nh_nl_ratio: pd.Series,
    ad_line_slope_20d: pd.Series,
    *,
    dt: pd.Timestamp,
    lookback_sessions: int,
) -> bool:
    """v2 §1D line 279 — broadening_breadth predicate.

    Fires iff:
      nh_nl_ratio is RISING over `lookback_sessions` (strict increase)
      AND ad_line_slope_20d at `dt` > 0 (strictly positive).

    "Rising" = strict 5-session increase per Ambiguity Log #68.
    """
    nh_nl_pts = _lookback_endpoint_values(
        nh_nl_ratio, dt=dt, lookback_sessions=lookback_sessions
    )
    if nh_nl_pts is None:
        return False
    if dt not in ad_line_slope_20d.index:
        return False
    slope_now = ad_line_slope_20d.loc[dt]
    if pd.isna(slope_now):
        return False

    nh_nl_then, nh_nl_now = nh_nl_pts
    return bool(nh_nl_now > nh_nl_then and float(slope_now) > 0.0)


def _evaluate_recovery_breadth(
    nh_nl_ratio: pd.Series,
    ad_line_slope_20d: pd.Series,
    *,
    dt: pd.Timestamp,
    lookback_sessions: int,
) -> bool:
    """v2 §1D ADR 0003 / Log #70 — `recovery_breadth` predicate.

    Fires iff:
      nh_nl_ratio is RISING over `lookback_sessions` (strict increase)
      AND ad_line_slope_20d at `dt` <= 0 (not yet strictly positive).

    "Rising" = strict 5-session increase per Ambiguity Log #68.

    Disjoint from `broadening_breadth` by construction: the slope conjuncts
    `<= 0` (recovery) and `> 0` (broadening) partition the real line at zero.
    Recovery sits above broadening in the §1D precedence (line 284) so the
    earlier turning-point signal surfaces before the lagging cumulative-AD
    confirmation.
    """
    nh_nl_pts = _lookback_endpoint_values(
        nh_nl_ratio, dt=dt, lookback_sessions=lookback_sessions
    )
    if nh_nl_pts is None:
        return False
    if dt not in ad_line_slope_20d.index:
        return False
    slope_now = ad_line_slope_20d.loc[dt]
    if pd.isna(slope_now):
        return False

    nh_nl_then, nh_nl_now = nh_nl_pts
    return bool(nh_nl_now > nh_nl_then and float(slope_now) <= 0.0)


# Zweig-style `breadth_thrust` LABEL thresholds — spec-fixed, NOT configurable
# (ADR 0003 / Log #69 pin). Values match the V2 §1D Breadth Thrust block.
_BREADTH_THRUST_LOW_THRESHOLD = 0.40
_BREADTH_THRUST_HIGH_THRESHOLD = 0.615
_BREADTH_THRUST_LOOKBACK_SESSIONS = 10


def _evaluate_breadth_thrust(
    breadth_thrust_feature: pd.Series,
    *,
    dt: pd.Timestamp,
) -> bool:
    """v2 §1D ADR 0003 / Log #69 — `breadth_thrust` LABEL predicate.

    Fires at session t iff:
      EXISTS b in [t-10, t-1] with breadth_thrust_feature[b] < 0.40
      AND breadth_thrust_feature[t] > 0.615

    Both inequalities strict per Zweig's canonical 1986 formulation. The
    thresholds (0.40, 0.615) and the 10-session lookback are spec-fixed
    (not configurable). NaN at `feature[t]` or at every `b` in the
    trailing 10-session window falsifies the rule (V1 §2.7 cold-start).
    """
    if dt not in breadth_thrust_feature.index:
        return False
    try:
        pos_t = breadth_thrust_feature.index.get_loc(dt)
    except KeyError:
        return False
    if not isinstance(pos_t, int):
        # Non-unique index entry — refuse to guess which row.
        return False
    feature_now = breadth_thrust_feature.iloc[pos_t]
    if pd.isna(feature_now):
        return False
    if not (float(feature_now) > _BREADTH_THRUST_HIGH_THRESHOLD):
        return False
    # Scan trailing window [t-10, t-1] for ANY past session with feature < 0.40.
    start_pos = max(0, pos_t - _BREADTH_THRUST_LOOKBACK_SESSIONS)
    if start_pos >= pos_t:
        return False  # cold-start: no prior history at all
    window = breadth_thrust_feature.iloc[start_pos:pos_t]
    non_nan = window.dropna()
    if non_nan.empty:
        return False
    return bool((non_nan < _BREADTH_THRUST_LOW_THRESHOLD).any())


def _data_quality_for_asof(
    *,
    spy_close: pd.Series,
    rsp_close: pd.Series,
    as_of_date: date,
    required_trading_days: int,
    raw_label: BreadthLabel,
    max_freshness_days: int,
    min_completeness: float,
) -> DataQuality:
    return assess_series_input_quality(
        as_of_date=as_of_date,
        required_inputs=[spy_close, rsp_close],
        required_trading_days=required_trading_days,
        raw_label=raw_label,
        max_freshness_days=max_freshness_days,
        min_completeness=min_completeness,
    )
