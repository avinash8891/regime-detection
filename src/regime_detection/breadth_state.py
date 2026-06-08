from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_detection._rule_helpers import ev_float as _ev_float
from regime_detection.config import BreadthV2Config
from regime_detection.data_quality import assess_series_input_quality
from regime_detection.fragility_universe import SECTOR_ETFS
from regime_detection.models import DataQuality
from regime_shared.pandas_compat import (
    cow_safe_assign,
    optional_date,
    require_single_session,
)
from regime_shared.pit_provenance import (
    BIAS_WARNING as _PIT_BIAS_WARNING_CODE,
    SOURCE_NAME as _PIT_BIAS_SOURCE,
    SOURCE_URL as _PIT_BIAS_SOURCE_URL,
    make_bias_warnings_frame,
)

# V2 §1D (ADR 0003 / decisions 69, 70) extends the V1 5-label set with four
# PIT-derived labels. Members ordered by precedence (spec line 385):
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
    "breadth_thrust": 0,  # bullish initiation (ADR 0003 / decision 69)
    "healthy_breadth": 0,
    "broadening_breadth": 0,  # V2 recovery confirmation
    "neutral_breadth": 1,
    "recovery_breadth": 1,  # mid-recovery (ADR 0003 / decision 70)
    "weak_breadth": 2,
    "narrowing_breadth": 2,  # V2 deterioration — mid-severity
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
    # F-011: pin min_periods to the spec values, not "complete window". §6.6
    # (index_distance_from_63d_high) and §6.8 both specify
    # ``close.rolling(63, min_periods=50)`` — the 63d high requires 50 observations,
    # NOT a full 63. relative_breadth_sma50 is a true 50d SMA (min_periods=50). No
    # emitted-output change (the fixture is fully warmed by 2016), but the warm-up
    # mask now matches the spec instead of masking 13 extra early sessions.
    ratio_sma50 = ratio.rolling(50, min_periods=50).mean()
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


def raw_label_for_day(
    f: BreadthFeatures, dt: pd.Timestamp
) -> tuple[BreadthLabel, dict[str, Any]]:
    """Per-day raw breadth label.

    F-043: thin wrapper over :func:`build_raw_outputs` so the §6.9 ETF-proxy
    rule predicates and evidence shape have a single encoding. Each feature is
    sliced to ``[dt]``; the vectorized builder reads only the target session.
    """
    # Guard: dt must resolve to exactly one session — a duplicate-date index would make
    # .loc[[dt]] return multiple rows and labels[0] silently mask the data issue.
    require_single_session(f.spy_close.index, dt)
    day_features = BreadthFeatures(
        spy_close=f.spy_close.loc[[dt]],
        rsp_close=f.rsp_close.loc[[dt]],
        relative_breadth_ratio=f.relative_breadth_ratio.loc[[dt]],
        relative_breadth_sma50=f.relative_breadth_sma50.loc[[dt]],
        relative_breadth_return_20d=f.relative_breadth_return_20d.loc[[dt]],
        index_distance_from_63d_high=f.index_distance_from_63d_high.loc[[dt]],
    )
    labels, evidence = build_raw_outputs(day_features)
    return labels[0], evidence[0]


def build_raw_outputs(
    f: BreadthFeatures,
) -> tuple[list[BreadthLabel], list[dict[str, Any]]]:
    ratio = f.relative_breadth_ratio
    ratio_sma = f.relative_breadth_sma50
    ratio_ret20 = f.relative_breadth_return_20d
    idx_dist = f.index_distance_from_63d_high

    valid = ~(ratio.isna() | ratio_sma.isna() | ratio_ret20.isna() | idx_dist.isna())
    divergent_fragile = (
        valid & idx_dist.ge(-0.05) & ratio.lt(ratio_sma) & ratio_ret20.le(-0.03)
    )
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
                "proxy": "RSP/SPY",
                "relative_breadth_ratio": _ev_float(ratio.iat[idx]),
                "relative_breadth_sma50": _ev_float(ratio_sma.iat[idx]),
                "relative_breadth_return_20d": _ev_float(ratio_ret20.iat[idx]),
                "index_distance_from_63d_high": _ev_float(idx_dist.iat[idx]),
                "divergent_fragile": bool(divergent_fragile.iat[idx]),
                "weak_breadth": bool(weak_breadth.iat[idx]),
                "healthy_breadth": bool(healthy_breadth.iat[idx]),
            }
        )

    return list(labels), evidence


def resolve_v2_raw_outputs(
    *,
    dates: pd.Index,
    raw_labels: list[BreadthLabel],
    raw_evidence: list[dict[str, Any]],
    pct_above_50dma: pd.Series,
    pct_above_200dma: pd.Series,
    nh_nl_ratio: pd.Series,
    ad_line_slope_20d: pd.Series,
    breadth_thrust: pd.Series | None,
    lookback_sessions: int,
    nh_nl_threshold: float,
) -> tuple[list[BreadthLabel], list[dict[str, Any]]]:
    updated_labels: list[BreadthLabel] = []
    updated_evidence = [dict(evidence) for evidence in raw_evidence]
    for idx_pos, day in enumerate(dates):
        v1_raw = raw_labels[idx_pos]
        thrust_fires = (
            _evaluate_breadth_thrust(breadth_thrust, dt=day)
            if breadth_thrust is not None
            else False
        )
        narrowing_fires = _evaluate_narrowing_breadth(
            pct_above_50dma=pct_above_50dma,
            pct_above_200dma=pct_above_200dma,
            nh_nl_ratio=nh_nl_ratio,
            dt=day,
            lookback_sessions=lookback_sessions,
            nh_nl_threshold=nh_nl_threshold,
        )
        recovery_fires = _evaluate_recovery_breadth(
            nh_nl_ratio=nh_nl_ratio,
            ad_line_slope_20d=ad_line_slope_20d,
            dt=day,
            lookback_sessions=lookback_sessions,
        )
        broadening_fires = _evaluate_broadening_breadth(
            nh_nl_ratio=nh_nl_ratio,
            ad_line_slope_20d=ad_line_slope_20d,
            dt=day,
            lookback_sessions=lookback_sessions,
        )
        if thrust_fires:
            resolved: BreadthLabel = "breadth_thrust"
        elif v1_raw == "divergent_fragile":
            resolved = "divergent_fragile"
        elif narrowing_fires:
            resolved = "narrowing_breadth"
        elif (
            v1_raw in {"weak_breadth", "healthy_breadth", "neutral_breadth", "unknown"}
            and recovery_fires
        ):
            resolved = "recovery_breadth"
        elif (
            v1_raw in {"weak_breadth", "healthy_breadth", "neutral_breadth", "unknown"}
            and broadening_fires
        ):
            resolved = "broadening_breadth"
        else:
            resolved = v1_raw
        updated_evidence[idx_pos].update(
            {
                "v2_breadth_thrust": thrust_fires,
                "v2_narrowing_breadth": narrowing_fires,
                "v2_recovery_breadth": recovery_fires,
                "v2_broadening_breadth": broadening_fires,
                "v1_raw_label": v1_raw,
            }
        )
        updated_labels.append(resolved)
    return updated_labels, updated_evidence


# ---------------------------------------------------------------------------
# V2 §1D rule predicates (ADR 0003 / decision 69, #70). The two predicates that
# ship today read the PIT-aware features from FeatureStore.breadth_state_v2
# and gate on strict 5-session rate-of-change ("rising"/"falling" = strict
# change over `label_rate_of_change_lookback_sessions` sessions).
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
    """v2 §1D line 381 — narrowing_breadth predicate.

    Fires iff:
      pct_above_50dma is FALLING over `lookback_sessions` (strict decrease)
      AND pct_above_200dma is FALLING over `lookback_sessions`
      AND nh_nl_ratio at `dt` < nh_nl_threshold (default 0.4).

    "Falling" = strict 5-session decrease per ADR 0003.
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
    """v2 §1D line 379 — broadening_breadth predicate.

    Fires iff:
      nh_nl_ratio is RISING over `lookback_sessions` (strict increase)
      AND ad_line_slope_20d at `dt` > 0 (strictly positive).

    "Rising" = strict 5-session increase per ADR 0003.
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
    """v2 §1D line 380 — `recovery_breadth` predicate (ADR 0003 / decision 70).

    Fires iff:
      nh_nl_ratio is RISING over `lookback_sessions` (strict increase)
      AND ad_line_slope_20d at `dt` <= 0 (not yet strictly positive).

    "Rising" = strict 5-session increase per ADR 0003.

    Disjoint from `broadening_breadth` by construction: the slope conjuncts
    `<= 0` (recovery) and `> 0` (broadening) partition the real line at zero.
    Recovery sits above broadening in the §1D precedence (line 385) so the
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
# (ADR 0003 / decision 69). Values match the V2 §1D Breadth Thrust block (spec lines 366-368).
_BREADTH_THRUST_LOW_THRESHOLD = 0.40
_BREADTH_THRUST_HIGH_THRESHOLD = 0.615
_BREADTH_THRUST_LOOKBACK_SESSIONS = 10


def _evaluate_breadth_thrust(
    breadth_thrust_feature: pd.Series,
    *,
    dt: pd.Timestamp,
) -> bool:
    """v2 §1D lines 366-368 — `breadth_thrust` LABEL predicate (ADR 0003 / decision 69).

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


# PIT feature names in spec order (v2 §1D lines 328-368).
_PIT_FEATURE_NAMES: tuple[str, ...] = (
    "pct_above_50dma",
    "pct_above_200dma",
    "ad_line",
    "ad_line_slope_20d",
    "nh_nl_ratio",
    "upvol_downvol_ratio",
    "breadth_thrust",
)

# breadth_thrust 10-session rolling mean window (v2 §1D line 360).
_BREADTH_THRUST_WINDOW = 10

# ad_line_slope_20d lookback (v2 §1D line 340).
_AD_LINE_SLOPE_LOOKBACK = 20


@dataclass(frozen=True)
class BreadthV2Features:
    """v2 §1D — per-session continuous breadth features.

    ``sector_breadth`` is always present. Optional PIT fields are ``None`` on
    the v1+v2-sector-only callsite and
    materialised when both ``pit_constituent_intervals`` and
    ``constituent_ohlcv`` are threaded through ``compute_breadth_v2_features``.
    """

    sector_breadth: pd.Series
    available_sector_breadth: pd.Series | None = None
    available_sector_count: pd.Series | None = None
    missing_sector_count: pd.Series | None = None
    missing_sector_symbols: pd.Series | None = None
    bias_warnings: pd.DataFrame | None = None
    pct_above_50dma: pd.Series | None = None
    pct_above_200dma: pd.Series | None = None
    ad_line: pd.Series | None = None
    ad_line_slope_20d: pd.Series | None = None
    nh_nl_ratio: pd.Series | None = None
    upvol_downvol_ratio: pd.Series | None = None
    breadth_thrust: pd.Series | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = ["sector_breadth"]
        for proxy_name in (
            "available_sector_breadth",
            "available_sector_count",
            "missing_sector_count",
            "missing_sector_symbols",
        ):
            if getattr(self, proxy_name) is not None:
                names.append(proxy_name)
        for pit_name in _PIT_FEATURE_NAMES:
            if getattr(self, pit_name) is not None:
                names.append(pit_name)
        return tuple(names)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({name: getattr(self, name) for name in self.feature_names})


_AVAILABLE_SECTOR_BREADTH_WARNING_CODE = "available_sector_breadth_proxy"
_AVAILABLE_SECTOR_BREADTH_SOURCE = "sector_etf_available_denominator_backtest_proxy"
_AVAILABLE_SECTOR_BREADTH_SOURCE_URL = "docs/regime_engine_v2_spec.md#sector-breadth"


def _available_sector_proxy_bias_warning() -> dict[str, str]:
    return {
        "warning_code": _AVAILABLE_SECTOR_BREADTH_WARNING_CODE,
        "feature_name": "available_sector_breadth",
        "source": _AVAILABLE_SECTOR_BREADTH_SOURCE,
        "source_url": _AVAILABLE_SECTOR_BREADTH_SOURCE_URL,
    }


def _compute_sector_breadth_features(
    *,
    sector_etf_closes: dict[str, pd.Series],
    sector_universe: tuple[str, ...],
    reference_index: pd.DatetimeIndex,
    lookback: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    returns_frame = pd.DataFrame(
        {
            symbol: sector_etf_closes[symbol]
            .astype(float)
            .pct_change(
                periods=lookback,
                fill_method=None,
            )
            for symbol in sector_universe
        },
        index=reference_index,
    )
    positive_mask = returns_frame > 0.0
    valid_mask = returns_frame.notna()
    expected_universe_size = len(sector_universe)

    sector_breadth = positive_mask.sum(axis=1).astype(float) / float(
        expected_universe_size
    )
    sector_breadth = sector_breadth.where(~returns_frame.isna().any(axis=1))
    sector_breadth.name = "sector_breadth"

    available_sector_count = valid_mask.sum(axis=1).astype("int64")
    available_sector_count.name = "available_sector_count"
    missing_sector_count = (expected_universe_size - available_sector_count).astype(
        "int64"
    )
    missing_sector_count.name = "missing_sector_count"

    available_sector_breadth = positive_mask.sum(axis=1).astype(
        float
    ) / available_sector_count.where(available_sector_count > 0).astype(float)
    available_sector_breadth = available_sector_breadth.where(
        available_sector_count > 0
    )
    available_sector_breadth.name = "available_sector_breadth"

    def _missing_symbols_for_row(row: pd.Series) -> str:
        missing = [symbol for symbol, is_valid in row.items() if not bool(is_valid)]
        return ",".join(missing)

    missing_sector_symbols = valid_mask.apply(_missing_symbols_for_row, axis=1)
    missing_sector_symbols.name = "missing_sector_symbols"
    return (
        sector_breadth,
        available_sector_breadth,
        available_sector_count,
        missing_sector_count,
        missing_sector_symbols,
    )


def compute_breadth_v2_features(
    *,
    sector_etf_closes: dict[str, pd.Series],
    config: BreadthV2Config,
    pit_constituent_intervals: pd.DataFrame | None = None,
    constituent_ohlcv: dict[str, pd.DataFrame] | None = None,
) -> BreadthV2Features:
    """Compute the v2 §1D sector_breadth feature from sector ETF closes.

    Parameters
    ----------
    sector_etf_closes
        Mapping ``symbol -> close series`` for the 11 US GICS sector ETFs
        (``SECTOR_ETFS``). All series must share a common DatetimeIndex.
    config
        ``BreadthV2Config`` instance — supplies ``sector_breadth_lookback_days``
        (no magic numbers in the function body).

    Returns
    -------
    BreadthV2Features
        Frozen dataclass with ``sector_breadth: pd.Series`` aligned to the
        first present sector's index.
    """
    lookback = config.sector_breadth_lookback_days
    sector_universe = SECTOR_ETFS

    # Resolve a reference index from the first present sector. If NO sectors
    # are present we cannot return a typed series at all — callers must guard
    # via feature_store. Treat as "fail loud" here.
    present_symbols = [s for s in sector_universe if s in sector_etf_closes]
    if not present_symbols:
        raise ValueError(
            "sector_etf_closes must contain at least one of the 11 SECTOR_ETFS "
            f"(got: {sorted(sector_etf_closes.keys())})."
        )
    reference_index = sector_etf_closes[present_symbols[0]].index
    aligned_sector_closes = {
        symbol: (
            sector_etf_closes[symbol]
            if symbol in sector_etf_closes
            else pd.Series(np.nan, index=reference_index, name=symbol)
        )
        for symbol in sector_universe
    }
    (
        sector_breadth,
        available_sector_breadth,
        available_sector_count,
        missing_sector_count,
        missing_sector_symbols,
    ) = _compute_sector_breadth_features(
        sector_etf_closes=aligned_sector_closes,
        sector_universe=sector_universe,
        reference_index=reference_index,
        lookback=lookback,
    )
    bias_warning_rows = [_available_sector_proxy_bias_warning()]

    # PIT-aware §1D features. Both new kwargs must be supplied
    # for the seven survivorship-biased features to materialise; otherwise
    # the v1+v2-sector-only callsite is preserved (all-None PIT fields).
    if pit_constituent_intervals is None or constituent_ohlcv is None:
        return BreadthV2Features(
            sector_breadth=sector_breadth,
            available_sector_breadth=available_sector_breadth,
            available_sector_count=available_sector_count,
            missing_sector_count=missing_sector_count,
            missing_sector_symbols=missing_sector_symbols,
            bias_warnings=make_bias_warnings_frame(bias_warning_rows),
        )

    pit_features = _compute_pit_features(
        reference_index=reference_index,
        pit_constituent_intervals=pit_constituent_intervals,
        constituent_ohlcv=constituent_ohlcv,
        config=config,
    )
    bias_warnings = make_bias_warnings_frame(
        bias_warning_rows
        + [
            {
                "warning_code": _PIT_BIAS_WARNING_CODE,
                "feature_name": feat,
                "source": _PIT_BIAS_SOURCE,
                "source_url": _PIT_BIAS_SOURCE_URL,
            }
            for feat in _PIT_FEATURE_NAMES
        ]
    )
    return BreadthV2Features(
        sector_breadth=sector_breadth,
        available_sector_breadth=available_sector_breadth,
        available_sector_count=available_sector_count,
        missing_sector_count=missing_sector_count,
        missing_sector_symbols=missing_sector_symbols,
        bias_warnings=bias_warnings,
        **pit_features,
    )


# ---------------------------------------------------------------------------
# PIT feature helpers. Each takes a (sessions × members)
# adjusted_close DataFrame plus the precomputed members_by_session mapping
# and returns a per-session pd.Series aligned to ``reference_index``.
# ---------------------------------------------------------------------------


def _normalize_interval_dates(intervals: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``start_date``/``end_date`` columns to ``datetime.date`` objects.

    The on-disk PIT parquet stores ISO date strings; ``read_pit_intervals``
    converts them to ``dt.date``. Test fixtures construct the frame directly
    with ISO strings, so we normalize defensively before calling
    ``members_on`` (which compares to ``dt.date``).
    """

    return cow_safe_assign(
        intervals,
        {
            "start_date": intervals["start_date"].map(optional_date),
            "end_date": intervals["end_date"].map(optional_date),
        },
    )


def _compute_pit_features(
    *,
    reference_index: pd.DatetimeIndex,
    pit_constituent_intervals: pd.DataFrame,
    constituent_ohlcv: dict[str, pd.DataFrame],
    config: BreadthV2Config,
) -> dict[str, pd.Series]:
    intervals = _normalize_interval_dates(pit_constituent_intervals)
    all_member_tickers = sorted(set(intervals["ticker"].tolist()))
    session_days = list(reference_index.date)
    ticker_to_col = {ticker: idx for idx, ticker in enumerate(all_member_tickers)}

    # Build the (sessions × members) adjusted_close and volume frames by
    # filling a preallocated dense matrix. This preserves the exact
    # ticker-missing => all-NaN semantics without paying DataFrame-of-Series
    # alignment costs for every member on every run.
    adj_close_frame = _build_member_frame(
        reference_index=reference_index,
        all_member_tickers=all_member_tickers,
        constituent_ohlcv=constituent_ohlcv,
        column_name="adjusted_close",
    )
    volume_frame = _build_member_frame(
        reference_index=reference_index,
        all_member_tickers=all_member_tickers,
        constituent_ohlcv=constituent_ohlcv,
        column_name="volume",
    )

    # PIT membership mask: derive directly from the interval rows. This keeps
    # the exact members_on semantics but avoids per-session set construction.
    membership_array = np.zeros(
        (len(reference_index), len(all_member_tickers)),
        dtype=bool,
    )
    for row in intervals.itertuples(index=False):
        ticker = str(row.ticker)
        col_idx = ticker_to_col.get(ticker)
        if col_idx is None:
            continue
        start_pos = bisect_left(session_days, row.start_date)
        if start_pos >= len(session_days):
            continue
        if row.end_date is None:
            end_pos_exclusive = len(session_days)
        else:
            end_pos_exclusive = bisect_right(session_days, row.end_date)
        if start_pos >= end_pos_exclusive:
            continue
        membership_array[start_pos:end_pos_exclusive, col_idx] = True
    membership_mask = pd.DataFrame(
        membership_array,
        index=reference_index,
        columns=all_member_tickers,
        dtype=bool,
    )

    # Per-ticker daily direction sign on adjusted_close (implementation decision #56):
    # +1 advance, -1 decline, 0 unchanged. Diff produces NaN at t=0 → mapped
    # to NaN (no prior close → excluded from advance / decline counts).
    diff_frame = adj_close_frame.diff()
    advance_mask = diff_frame > 0.0
    decline_mask = diff_frame < 0.0
    unchanged_mask = diff_frame == 0.0
    # Note: NaN values yield False on all three comparisons — exactly the
    # "no prior close → excluded" behavior required.

    pct_above_50dma = _compute_pct_above_sma(
        adj_close_frame=adj_close_frame,
        membership_mask=membership_mask,
        sma_window=config.sma_lookback_50,
    )
    pct_above_200dma = _compute_pct_above_sma(
        adj_close_frame=adj_close_frame,
        membership_mask=membership_mask,
        sma_window=config.sma_lookback_200,
    )
    ad_line, ad_line_slope_20d = _compute_ad_line(
        advance_mask=advance_mask,
        decline_mask=decline_mask,
        membership_mask=membership_mask,
    )
    nh_nl_ratio = _compute_nh_nl_ratio(
        adj_close_frame=adj_close_frame,
        membership_mask=membership_mask,
        window=config.nh_nl_lookback_sessions,
    )
    upvol_downvol_ratio = _compute_upvol_downvol_ratio(
        advance_mask=advance_mask,
        decline_mask=decline_mask,
        membership_mask=membership_mask,
        volume_frame=volume_frame,
    )
    breadth_thrust = _compute_breadth_thrust(
        advance_mask=advance_mask,
        decline_mask=decline_mask,
        unchanged_mask=unchanged_mask,
        membership_mask=membership_mask,
    )

    return {
        "pct_above_50dma": pct_above_50dma,
        "pct_above_200dma": pct_above_200dma,
        "ad_line": ad_line,
        "ad_line_slope_20d": ad_line_slope_20d,
        "nh_nl_ratio": nh_nl_ratio,
        "upvol_downvol_ratio": upvol_downvol_ratio,
        "breadth_thrust": breadth_thrust,
    }


def _build_member_frame(
    *,
    reference_index: pd.DatetimeIndex,
    all_member_tickers: list[str],
    constituent_ohlcv: dict[str, pd.DataFrame],
    column_name: str,
) -> pd.DataFrame:
    data = np.full(
        (len(reference_index), len(all_member_tickers)),
        np.nan,
        dtype=float,
    )
    for col_idx, ticker in enumerate(all_member_tickers):
        frame = constituent_ohlcv.get(ticker)
        if frame is None:
            continue
        aligned = frame[column_name].reindex(reference_index)
        data[:, col_idx] = aligned.to_numpy(dtype=float, na_value=np.nan)
    return pd.DataFrame(
        data,
        index=reference_index,
        columns=all_member_tickers,
        dtype=float,
    )


def _compute_pct_above_sma(
    *,
    adj_close_frame: pd.DataFrame,
    membership_mask: pd.DataFrame,
    sma_window: int,
) -> pd.Series:
    """v2 §1D pct_above_{N}dma — strict ``>`` against the per-ticker SMA.

    NaN-SMA tickers are excluded from BOTH numerator AND denominator
    (implementation decision #58); zero-denominator → NaN.
    """
    sma_frame = adj_close_frame.rolling(sma_window, min_periods=sma_window).mean()
    # A ticker is "valid" at session D if it's a member AND has a defined SMA
    # AND has a defined close. (Defined close is implied by defined SMA but
    # we guard explicitly.)
    valid_mask = membership_mask & sma_frame.notna() & adj_close_frame.notna()
    above_mask = valid_mask & (adj_close_frame > sma_frame)
    above_count = above_mask.sum(axis=1).astype(float)
    valid_count = valid_mask.sum(axis=1).astype(float)
    ratio = above_count / valid_count.where(valid_count > 0)
    ratio.name = f"pct_above_{sma_window}dma"
    return ratio


def _compute_ad_line(
    *,
    advance_mask: pd.DataFrame,
    decline_mask: pd.DataFrame,
    membership_mask: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """v2 §1D ad_line / ad_line_slope_20d (implementation decision #56 direction; #57 t=0 anchor).

    ad_line[0] = 0 anchor; ad_line[t] = ad_line[t-1] + (advances - declines)
    where advances/declines count PIT members with strict direction.
    """
    advances = (advance_mask & membership_mask).sum(axis=1).astype(float)
    declines = (decline_mask & membership_mask).sum(axis=1).astype(float)
    delta = advances - declines
    # Anchor at 0 on the first session (implementation decision #57).
    delta.iloc[0] = 0.0
    ad_line = delta.cumsum()
    ad_line.name = "ad_line"

    slope = (ad_line - ad_line.shift(_AD_LINE_SLOPE_LOOKBACK)) / float(
        _AD_LINE_SLOPE_LOOKBACK
    )
    slope.name = "ad_line_slope_20d"
    return ad_line, slope


def _compute_nh_nl_ratio(
    *,
    adj_close_frame: pd.DataFrame,
    membership_mask: pd.DataFrame,
    window: int,
) -> pd.Series:
    """v2 §1D nh_nl_ratio over a 252-session inclusive window.

    A ticker contributes only when it has at least ``window`` non-NaN
    sessions in the trailing inclusive window. ``ratio = nh / max(nh+nl, 1)``;
    when no member has sufficient history, the value is NaN.
    """
    rolling_max = adj_close_frame.rolling(window, min_periods=window).max()
    rolling_min = adj_close_frame.rolling(window, min_periods=window).min()

    sufficient = rolling_max.notna() & rolling_min.notna()
    valid_mask = membership_mask & sufficient
    new_high_mask = valid_mask & (adj_close_frame == rolling_max)
    new_low_mask = valid_mask & (adj_close_frame == rolling_min)

    new_highs = new_high_mask.sum(axis=1).astype(float)
    new_lows = new_low_mask.sum(axis=1).astype(float)
    valid_count = valid_mask.sum(axis=1).astype(float)

    denom = (new_highs + new_lows).where(lambda s: s > 0, other=1.0)
    ratio = new_highs / denom
    # Where no member has sufficient history, surface NaN (not 0/1).
    ratio = ratio.where(valid_count > 0)
    ratio.name = "nh_nl_ratio"
    return ratio


def _compute_upvol_downvol_ratio(
    *,
    advance_mask: pd.DataFrame,
    decline_mask: pd.DataFrame,
    membership_mask: pd.DataFrame,
    volume_frame: pd.DataFrame,
) -> pd.Series:
    """v2 §1D upvol_downvol_ratio (implementation decision #56 direction; §1D line 350 raw volume).

    Direction uses adjusted_close (strict ``>`` / ``<``); volume is RAW
    shares from constituent_ohlcv[ticker]['volume'].
    """
    advance_vol_mask = advance_mask & membership_mask
    decline_vol_mask = decline_mask & membership_mask
    upvol = volume_frame.where(advance_vol_mask, other=0.0).sum(axis=1)
    downvol = volume_frame.where(decline_vol_mask, other=0.0).sum(axis=1)
    ratio = upvol / downvol.where(downvol > 0, other=1.0)
    ratio.name = "upvol_downvol_ratio"
    return ratio


def _compute_breadth_thrust(
    *,
    advance_mask: pd.DataFrame,
    decline_mask: pd.DataFrame,
    unchanged_mask: pd.DataFrame,
    membership_mask: pd.DataFrame,
) -> pd.Series:
    """v2 §1D breadth_thrust — 10-session MA of pct_advancing.

    ``pct_advancing = advances / max(advances + declines + unchanged, 1)``
    where the denominator counts PIT members with a valid prior close
    (implementation decision #56).
    """
    advances = (advance_mask & membership_mask).sum(axis=1).astype(float)
    declines = (decline_mask & membership_mask).sum(axis=1).astype(float)
    unchanged = (unchanged_mask & membership_mask).sum(axis=1).astype(float)
    valid = advances + declines + unchanged
    pct_advancing = advances / valid.where(valid > 0, other=1.0)
    thrust = pct_advancing.rolling(
        _BREADTH_THRUST_WINDOW, min_periods=_BREADTH_THRUST_WINDOW
    ).mean()
    thrust.name = "breadth_thrust"
    return thrust
