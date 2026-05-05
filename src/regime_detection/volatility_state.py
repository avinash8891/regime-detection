from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.models import AxisOutput, DataQuality


VolatilityLabel = Literal["low_vol", "normal_vol", "high_vol", "crisis_vol", "unknown"]


_RISK_RANK: dict[VolatilityLabel, int] = {
    "low_vol": 0,
    "normal_vol": 1,
    "high_vol": 2,
    "crisis_vol": 3,
    "unknown": 2,
}


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
    vix_proxy_close: pd.Series
    return_1d: pd.Series
    return_5d: pd.Series
    return_21d: pd.Series
    realized_vol_percentile_252d: pd.Series
    vix_percentile_252d: pd.Series


def compute_features(*, close: pd.Series, vix_proxy_close: pd.Series) -> VolatilityFeatures:
    close = close.astype(float)
    vix_proxy_close = vix_proxy_close.astype(float)
    # Align VIX proxy series to the SPY trading-day index; missing dates become NaN.
    vix_proxy_close = vix_proxy_close.reindex(close.index)

    return_1d = close / close.shift(1) - 1
    return_5d = close / close.shift(5) - 1
    return_21d = close / close.shift(21) - 1

    daily_returns = close.pct_change()
    realized_vol_21d = daily_returns.rolling(21).std() * np.sqrt(252)
    realized_vol_percentile_252d = realized_vol_21d.rolling(252, min_periods=252).apply(_pct_rank_last, raw=True)

    vix_percentile_252d = vix_proxy_close.rolling(252, min_periods=252).apply(_pct_rank_last, raw=True)

    return VolatilityFeatures(
        close=close,
        vix_proxy_close=vix_proxy_close,
        return_1d=return_1d,
        return_5d=return_5d,
        return_21d=return_21d,
        realized_vol_percentile_252d=realized_vol_percentile_252d,
        vix_percentile_252d=vix_percentile_252d,
    )


def raw_label_for_day(f: VolatilityFeatures, dt: pd.Timestamp) -> tuple[VolatilityLabel, dict[str, Any]]:
    ret1 = f.return_1d.loc[dt]
    ret5 = f.return_5d.loc[dt]
    ret21 = f.return_21d.loc[dt]
    vol_pct = f.realized_vol_percentile_252d.loc[dt]
    vix_pct = f.vix_percentile_252d.loc[dt]

    if any(pd.isna(x) for x in [ret1, ret5, ret21, vol_pct, vix_pct]):
        return "unknown", {"reason": "insufficient_history"}

    crisis = bool(
        (ret1 <= -0.05)
        or (ret5 <= -0.08)
        or ((vol_pct >= 0.90) and (ret21 <= -0.05))
        or (vix_pct >= 0.95)
    )
    high_vol = bool((vol_pct >= 0.80) or (vix_pct >= 0.80))
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

    return label, {
        "crisis_vol": crisis,
        "high_vol": high_vol,
        "low_vol": low_vol,
        "normal_vol": normal_vol,
    }


def classify_series(
    *,
    close: pd.Series,
    vix_proxy_close: pd.Series,
    as_of_date: date,
    deescalation_days: int,
) -> AxisOutput:
    dt = pd.Timestamp(as_of_date)
    close = close.copy()
    vix_proxy_close = vix_proxy_close.copy()
    close.index = pd.to_datetime(close.index)
    vix_proxy_close.index = pd.to_datetime(vix_proxy_close.index)
    close = close.sort_index()
    vix_proxy_close = vix_proxy_close.sort_index()

    if dt not in close.index:
        raise ValueError(f"as_of_date missing from close series: {as_of_date.isoformat()}")
    # If VIX proxy is missing the as-of date, we still proceed; the label will become unknown.

    close = close.loc[:dt]
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

    dq = _data_quality_for_asof(close=close, as_of_date=as_of_date, required_trading_days=252, raw_label=raw)

    return AxisOutput(
        raw_label=raw,
        stable_label=stable,
        active_label=active,
        evidence={
            "rule_evidence": raw_evidence[-1],
            "risk_rank": _RISK_RANK,
            "deescalation_days": deescalation_days,
            "vix_proxy_symbol": "VIXY",
        },
        data_quality=dq,
    )


def _data_quality_for_asof(
    *,
    close: pd.Series,
    as_of_date: date,
    required_trading_days: int,
    raw_label: VolatilityLabel,
) -> DataQuality:
    dt = pd.Timestamp(as_of_date)
    window = close.loc[:dt].tail(required_trading_days)
    completeness = float(window.notna().mean()) if len(window) else 0.0

    if raw_label == "unknown":
        return DataQuality(status="insufficient_history", freshness_days=0, completeness=completeness, reason="insufficient_history")
    if completeness < 0.70:
        return DataQuality(status="insufficient_data", freshness_days=0, completeness=completeness, reason="insufficient_data")
    if completeness < 0.90:
        return DataQuality(status="degraded", freshness_days=0, completeness=completeness, reason="incomplete_data")
    return DataQuality(status="ok", freshness_days=0, completeness=completeness, reason=None)
