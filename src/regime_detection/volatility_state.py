from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.models import AxisOutput, DataQuality

if TYPE_CHECKING:  # avoid runtime cycle: volatility_state_v2 → config → ...
    from regime_detection.config import VolatilityV2RulesConfig
    from regime_detection.volatility_state_v2 import VolatilityV2Features


# v2 §1C line 191 precedence:
#   crisis_vol > vol_crush > high_vol > rising_vol > low_vol > normal_vol > unknown
# `rising_vol` lands in slice 2.6 (v2 §1C lines 146-148);
# `vol_crush` lands via ADR 0005 / Ambiguity Log #20 using FRED VIXCLS as
# implied_vol_30d plus the event-window seam.
VolatilityLabel = Literal[
    "low_vol",
    "normal_vol",
    "high_vol",
    "crisis_vol",
    "unknown",
    "rising_vol",
    "vol_crush",
]


# v2 §1C — annualization constant. Pinned here as the single source of truth
# for the shared ``realized_vol`` helper so v1 (existing slice-1 path) and v2
# (slice 2.6 `rising_vol` rule) consume one annualization convention.
_TRADING_DAYS_PER_YEAR = 252


def realized_vol(
    close: pd.Series, window: int, *, ddof: int = 1
) -> pd.Series:
    """Rolling annualised realised volatility of ``close`` log/pct returns.

    Shared helper for v1 volatility classifiers (slice 1) and the v2 §1C
    ``rising_vol`` rule (slice 2.6). One implementation prevents the
    "second-system" pattern flagged in slice 2.2 review.

    Algorithm:
        daily_returns = close.pct_change(fill_method=None)
        realized_vol[t] = rolling(window).std(ddof=ddof) * sqrt(252)

    Args:
        close: per-session close prices (date-indexed).
        window: rolling-window length in trading days.
        ddof: delta degrees-of-freedom for ``Series.std``. Pinned default
            ``1`` (sample std) matches pandas/numpy convention for
            financial time series; v2 §1C is silent so the convention is
            recorded in the Implementation Ambiguity Log.

    Returns a date-indexed ``pd.Series`` aligned to ``close.index``; NaN
    until ``t >= window`` (cold-start: pandas ``rolling.std`` requires
    ``window`` observations before emitting).
    """
    if window <= 0:
        raise ValueError(f"window must be > 0: got {window}")
    daily_returns = close.pct_change(fill_method=None)
    return daily_returns.rolling(window).std(ddof=ddof) * np.sqrt(
        _TRADING_DAYS_PER_YEAR
    )


def wilders_atr(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Wilder's Average True Range over `period` sessions.

    Shared helper for v1 volatility classifiers and v2 §1C features
    (atr_ratio = ATR_14 / ATR_50). Wilder's smoothing is the standard
    estimator referenced by v2 §1C line 142 (Implementation Ambiguity
    Log entry for "ATR estimator choice").

    NOTE: a separate ``_wilder_ewm`` helper lives in
    ``regime_detection.trend_character`` for the v1 ADX cold-start
    path. ``_wilder_ewm`` uses pandas-EWM seeding (first value of the
    TR series), whereas ``wilders_atr`` here uses the textbook
    Wilder-1978 mean-seeded form (seed = simple-mean(TR[0..period-1])).
    Both converge for large ``t`` but differ at cold-start. The two
    implementations intentionally coexist (v1 ADX cold-start values are
    frozen; V2 §1C ATR ratio uses the more faithful textbook form). A
    future cleanup may unify them after V2 walk-forward validation per
    v2 §9.1. See Implementation Ambiguity Log entry #15.

    Algorithm:
        TR[t] = max(
            high[t] - low[t],
            abs(high[t] - close[t-1]),
            abs(low[t]  - close[t-1]),
        )
        ATR[0..period-2] = NaN
        ATR[period-1]    = mean(TR[0..period-1])              # seed
        ATR[t]           = (ATR[t-1] * (period-1) + TR[t]) / period   # t >= period

    Returns a date-indexed pd.Series aligned to ``close.index``.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0: got {period}")
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, close must have identical length")

    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    n = len(tr)
    out = np.full(n, np.nan, dtype=float)
    tr_arr = tr.to_numpy(dtype=float)
    if n < period:
        return pd.Series(out, index=close.index, name=f"atr_{period}")

    # Seed: simple mean of the first `period` true-range values. If any of
    # those is NaN the seed is NaN and Wilder's recursion stays NaN forever.
    seed_window = tr_arr[:period]
    if np.isnan(seed_window).any():
        seed = float("nan")
    else:
        seed = float(seed_window.mean())
    out[period - 1] = seed

    for t in range(period, n):
        prev = out[t - 1]
        cur = tr_arr[t]
        if np.isnan(prev) or np.isnan(cur):
            out[t] = float("nan")
            continue
        out[t] = (prev * (period - 1) + cur) / period

    return pd.Series(out, index=close.index, name=f"atr_{period}")


# v2 §1C line 191 precedence:
#   crisis_vol > vol_crush > high_vol > rising_vol > low_vol >
#   normal_vol > unknown.
# V1 risk-rank contract is frozen in replay fixtures; crisis_vol remains 3.
# V2 crisis-vs-vol_crush precedence is resolved before hysteresis in
# volatility_state_v2, not by changing the V1 evidence rank.
_RISK_RANK: dict[VolatilityLabel, int] = {
    "low_vol": 0,
    "normal_vol": 1,
    "high_vol": 2,
    "crisis_vol": 3,
    "unknown": 2,
    "rising_vol": 2,
    "vol_crush": 3,
}


def _ev_float(x: float) -> float:
    return round(float(x), 8)


def _pct_rank_last(arr: np.ndarray) -> float:
    x = arr[-1]
    if np.isnan(x):
        return float("nan")
    arr2 = arr[~np.isnan(arr)]
    if arr2.size == 0:
        return float("nan")
    return float(np.mean(arr2 <= x))


@dataclass(frozen=True)
class VolatilityFeatures:
    close: pd.Series
    return_1d: pd.Series
    return_5d: pd.Series
    return_21d: pd.Series
    realized_vol_21d: pd.Series
    realized_vol_percentile_252d: pd.Series
    vix_percentile_252d: pd.Series | None


def compute_features(*, close: pd.Series, vix_proxy_close: pd.Series | None) -> VolatilityFeatures:
    close = close.astype(float)
    vix_pct: pd.Series | None = None
    if vix_proxy_close is not None:
        vix_proxy_close = vix_proxy_close.astype(float)
        # Align VIX proxy series to the SPY trading-day index; missing dates become NaN.
        vix_proxy_close = vix_proxy_close.reindex(close.index)
        vix_pct = vix_proxy_close.rolling(252, min_periods=252).apply(_pct_rank_last, raw=True)

    return_1d = close / close.shift(1) - 1
    return_5d = close / close.shift(5) - 1
    return_21d = close / close.shift(21) - 1

    # v1 RV percentile feeds the v1 high_vol/low_vol thresholds. Uses the
    # shared ``realized_vol`` helper (slice 2.6) — preserves the v1 byte-
    # identical output (window=21, ddof default — pandas .std() is ddof=1).
    realized_vol_21d = realized_vol(close, window=21)
    realized_vol_percentile_252d = realized_vol_21d.rolling(252, min_periods=252).apply(_pct_rank_last, raw=True)

    return VolatilityFeatures(
        close=close,
        return_1d=return_1d,
        return_5d=return_5d,
        return_21d=return_21d,
        realized_vol_21d=realized_vol_21d,
        realized_vol_percentile_252d=realized_vol_percentile_252d,
        vix_percentile_252d=vix_pct,
    )


def raw_label_for_day(
    f: VolatilityFeatures,
    dt: pd.Timestamp,
    *,
    volatility_state_v2_features: "VolatilityV2Features | None" = None,
    volatility_state_v2_rules: "VolatilityV2RulesConfig | None" = None,
) -> tuple[VolatilityLabel, dict[str, Any]]:
    """Per-day raw volatility_state label.

    When ``volatility_state_v2_features`` AND ``volatility_state_v2_rules``
    are both supplied, the v2 §1C precedence (line 191:
    ``crisis_vol > vol_crush > high_vol > rising_vol > low_vol >
    normal_vol > unknown``) is layered ON TOP of the v1 label. When either
    is ``None`` the function returns the v1 label and evidence unchanged
    — byte-identical to the pre-slice-2.6 implementation.
    """
    ret1 = f.return_1d.loc[dt]
    ret5 = f.return_5d.loc[dt]
    ret21 = f.return_21d.loc[dt]
    rv21 = f.realized_vol_21d.loc[dt]
    vol_pct = f.realized_vol_percentile_252d.loc[dt]
    vix_pct = None if f.vix_percentile_252d is None else f.vix_percentile_252d.loc[dt]

    # V1 required features exclude VIX percentile (it is optional).
    if any(pd.isna(x) for x in [ret1, ret5, ret21, vol_pct]):
        return "unknown", {"reason": "insufficient_history"}

    vix_crisis = False
    vix_high = False
    if vix_pct is not None and not pd.isna(vix_pct):
        vix_crisis = bool(vix_pct >= 0.95)
        vix_high = bool(vix_pct >= 0.80)

    crisis = bool(
        (ret1 <= -0.05)
        or (ret5 <= -0.08)
        or ((vol_pct >= 0.90) and (ret21 <= -0.05))
        or vix_crisis
    )
    high_vol = bool((vol_pct >= 0.80) or vix_high)
    low_vol = bool(vol_pct <= 0.30)
    normal_vol = not (crisis or high_vol or low_vol)

    if crisis:
        label: VolatilityLabel = "crisis_vol"
    elif high_vol:
        label = "high_vol"
    elif low_vol:
        label = "low_vol"
    else:
        label = "normal_vol"

    evidence: dict[str, Any] = {
        "realized_vol_21d": _ev_float(rv21),
        "realized_vol_percentile_252d": _ev_float(vol_pct),
        "vix_percentile_252d": _ev_float(vix_pct) if (vix_pct is not None and not pd.isna(vix_pct)) else None,
        "crisis_vol": crisis,
        "high_vol": high_vol,
        "low_vol": low_vol,
    }

    if (
        volatility_state_v2_features is not None
        and volatility_state_v2_rules is not None
    ):
        # Local import keeps the v1 path free of v2 module load on cold
        # callers and avoids a circular import (volatility_state_v2 imports
        # VolatilityV2RulesConfig from config).
        from regime_detection.volatility_state_v2 import (
            evaluate_v2_volatility_label,
        )

        v2_label = evaluate_v2_volatility_label(
            v1_label=label,
            features=volatility_state_v2_features,
            dt=dt,
            rules_config=volatility_state_v2_rules,
        )
        if v2_label is not None:
            evidence["v2_override"] = {
                "from": label,
                "to": v2_label,
                "rule": v2_label,  # the winning v2 §1C rule (rising_vol / vol_crush)
            }
            label = v2_label  # type: ignore[assignment]

    return label, evidence


def build_raw_outputs(
    f: VolatilityFeatures,
    *,
    volatility_state_v2_features: "VolatilityV2Features | None" = None,
    volatility_state_v2_rules: "VolatilityV2RulesConfig | None" = None,
) -> tuple[list[VolatilityLabel], list[dict[str, Any]]]:
    """Vectorized v1 raw labels + optional v2 §1C `rising_vol` override.

    The v1 pass is unchanged from pre-slice-2.6. When both v2 args are
    supplied, the v2 §1C precedence at line 191 is applied per-day AFTER
    the v1 pass — `rising_vol` overrides v1 `low_vol` / `normal_vol` /
    `unknown` (NOT `crisis_vol` / `high_vol`, which outrank `rising_vol`).
    When either argument is None, output is byte-identical to v1.
    """
    ret1 = f.return_1d
    ret5 = f.return_5d
    ret21 = f.return_21d
    vol_pct = f.realized_vol_percentile_252d
    valid = ~(ret1.isna() | ret5.isna() | ret21.isna() | vol_pct.isna())

    rv21 = f.realized_vol_21d
    vix_pct_series = pd.Series(float("nan"), index=ret1.index, dtype=float)
    vix_present = pd.Series(False, index=ret1.index, dtype="bool")
    vix_crisis = pd.Series(False, index=ret1.index, dtype="bool")
    vix_high = pd.Series(False, index=ret1.index, dtype="bool")
    if f.vix_percentile_252d is not None:
        vix_pct_series = f.vix_percentile_252d.reindex(ret1.index)
        vix_present = vix_pct_series.notna()
        vix_crisis = vix_present & vix_pct_series.ge(0.95)
        vix_high = vix_present & vix_pct_series.ge(0.80)

    crisis = valid & (
        ret1.le(-0.05)
        | ret5.le(-0.08)
        | (vol_pct.ge(0.90) & ret21.le(-0.05))
        | vix_crisis
    )
    high_vol = valid & (vol_pct.ge(0.80) | vix_high)
    low_vol = valid & vol_pct.le(0.30)
    normal_vol = valid & ~(crisis | high_vol | low_vol)

    labels = np.full(len(ret1), "unknown", dtype=object)
    labels[normal_vol.to_numpy()] = "normal_vol"
    labels[low_vol.to_numpy()] = "low_vol"
    labels[high_vol.to_numpy()] = "high_vol"
    labels[crisis.to_numpy()] = "crisis_vol"

    evidence: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        if label == "unknown":
            evidence.append({"reason": "insufficient_history"})
            continue
        evidence.append(
            {
                "realized_vol_21d": _ev_float(rv21.iat[idx]),
                "realized_vol_percentile_252d": _ev_float(vol_pct.iat[idx]),
                "vix_percentile_252d": _ev_float(vix_pct_series.iat[idx]) if bool(vix_present.iat[idx]) else None,
                "crisis_vol": bool(crisis.iat[idx]),
                "high_vol": bool(high_vol.iat[idx]),
                "low_vol": bool(low_vol.iat[idx]),
            }
        )

    if (
        volatility_state_v2_features is not None
        and volatility_state_v2_rules is not None
    ):
        # v2 §1C line 191 precedence — applied per-day on top of v1.
        # Localize import to avoid a runtime cycle with volatility_state_v2.
        from regime_detection.volatility_state_v2 import (
            evaluate_v2_volatility_label,
        )

        for idx, dt in enumerate(ret1.index):
            v1_label = str(labels[idx])
            v2_label = evaluate_v2_volatility_label(
                v1_label=v1_label,
                features=volatility_state_v2_features,
                dt=dt,
                rules_config=volatility_state_v2_rules,
            )
            if v2_label is None:
                continue
            evidence[idx]["v2_override"] = {
                "from": v1_label,
                "to": v2_label,
                "rule": v2_label,  # the winning v2 §1C rule (rising_vol / vol_crush)
            }
            labels[idx] = v2_label

    return list(labels), evidence


def classify_series(
    *,
    close: pd.Series,
    vix_proxy_close: pd.Series | None,
    as_of_date: date,
    deescalation_days: int,
) -> AxisOutput:
    dt = pd.Timestamp(as_of_date)
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    if vix_proxy_close is not None:
        vix_proxy_close = vix_proxy_close.copy()
        vix_proxy_close.index = pd.to_datetime(vix_proxy_close.index)
        vix_proxy_close = vix_proxy_close.sort_index()

    if dt not in close.index:
        raise ValueError(f"as_of_date missing from close series: {as_of_date.isoformat()}")
    # If VIX proxy is missing the as-of date, we still proceed; the label will become unknown.

    close = close.loc[:dt]
    if vix_proxy_close is not None:
        vix_proxy_close = vix_proxy_close.loc[:dt]

    f = compute_features(close=close, vix_proxy_close=vix_proxy_close)
    raw_labels: list[VolatilityLabel] = []
    raw_evidence: list[dict[str, Any]] = []
    for day in close.index:
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
        return AxisOutput(
            raw_label="unknown",
            stable_label="unknown",
            active_label="unknown",
            evidence={"reason": "insufficient_history"},
            data_quality=DataQuality(
                status="insufficient_history",
                freshness_days=None,
                completeness=None,
                reason="required_feature_is_nan",
            ),
        )
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
