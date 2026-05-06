from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.models import AxisOutput, DataQuality


TrendDirectionLabel = Literal["bull", "bear", "sideways", "transition", "unknown"]


_RISK_RANK: dict[TrendDirectionLabel, int] = {
    "bull": 0,
    "sideways": 1,
    "transition": 2,
    "bear": 3,
    "unknown": 2,
}


@dataclass(frozen=True)
class TrendDirectionFeatures:
    close: pd.Series
    sma_50: pd.Series
    sma_200: pd.Series
    return_63d: pd.Series


def compute_features(close: pd.Series) -> TrendDirectionFeatures:
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    return_63d = close / close.shift(63) - 1
    return TrendDirectionFeatures(close=close, sma_50=sma_50, sma_200=sma_200, return_63d=return_63d)


def raw_label_for_day(f: TrendDirectionFeatures, dt: pd.Timestamp) -> tuple[TrendDirectionLabel, dict[str, Any]]:
    close = f.close.loc[dt]
    sma50 = f.sma_50.loc[dt]
    sma200 = f.sma_200.loc[dt]
    ret63 = f.return_63d.loc[dt]

    if any(pd.isna(x) for x in [close, sma50, sma200, ret63]):
        return "unknown", {"reason": "insufficient_history"}

    within_5pct_sma200 = bool((close >= sma200 * 0.95) and (close <= sma200 * 1.05))

    bull = bool((close > sma50) and (close > sma200) and (sma50 > sma200))
    bear = bool((close < sma50) and (close < sma200) and (sma50 < sma200))
    sideways = bool((abs(ret63) < 0.05) and within_5pct_sma200)
    transition = not (bull or bear or sideways)

    if bull:
        label: TrendDirectionLabel = "bull"
    elif bear:
        label = "bear"
    elif sideways:
        label = "sideways"
    else:
        label = "transition"

    return label, {
        "bull": bull,
        "bear": bear,
        "sideways": sideways,
        "transition": transition,
        "within_5pct_sma200": within_5pct_sma200,
    }


def build_raw_outputs(f: TrendDirectionFeatures) -> tuple[list[TrendDirectionLabel], list[dict[str, Any]]]:
    close = f.close
    sma50 = f.sma_50
    sma200 = f.sma_200
    ret63 = f.return_63d

    valid = ~(close.isna() | sma50.isna() | sma200.isna() | ret63.isna())
    within_5pct_sma200 = valid & close.between(sma200 * 0.95, sma200 * 1.05)
    bull = valid & close.gt(sma50) & close.gt(sma200) & sma50.gt(sma200)
    bear = valid & close.lt(sma50) & close.lt(sma200) & sma50.lt(sma200)
    sideways = valid & ret63.abs().lt(0.05) & within_5pct_sma200
    transition = valid & ~(bull | bear | sideways)

    labels = np.full(len(close), "unknown", dtype=object)
    labels[transition.to_numpy()] = "transition"
    labels[sideways.to_numpy()] = "sideways"
    labels[bear.to_numpy()] = "bear"
    labels[bull.to_numpy()] = "bull"

    evidence: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        if label == "unknown":
            evidence.append({"reason": "insufficient_history"})
            continue
        evidence.append(
            {
                "bull": bool(bull.iat[idx]),
                "bear": bool(bear.iat[idx]),
                "sideways": bool(sideways.iat[idx]),
                "transition": bool(transition.iat[idx]),
                "within_5pct_sma200": bool(within_5pct_sma200.iat[idx]),
            }
        )

    return list(labels), evidence


def apply_hysteresis(
    *,
    dates: pd.DatetimeIndex,
    raw_labels: list[TrendDirectionLabel],
    deescalation_days: int,
) -> tuple[list[TrendDirectionLabel], list[TrendDirectionLabel]]:
    if len(dates) != len(raw_labels):
        raise ValueError("dates/raw_labels length mismatch")
    return apply_asymmetric_hysteresis(
        raw_labels=raw_labels,
        risk_rank=_RISK_RANK,
        deescalation_days=deescalation_days,
    )


def classify_series(
    *,
    close: pd.Series,
    as_of_date: date,
    deescalation_days: int,
) -> AxisOutput:
    """
    Computes raw/stable/active labels for `as_of_date` by replaying across history
    up to that date.
    """
    s = close.copy()
    s.index = pd.to_datetime(s.index)
    dt = pd.Timestamp(as_of_date)
    if dt not in s.index:
        raise ValueError(f"as_of_date missing from close series: {as_of_date.isoformat()}")
    s = s.loc[:dt]

    f = compute_features(s)
    raw_labels: list[TrendDirectionLabel] = []
    raw_evidence: list[dict[str, Any]] = []
    for day in s.index:
        lbl, ev = raw_label_for_day(f, day)
        raw_labels.append(lbl)
        raw_evidence.append(ev)

    stable_labels, active_labels = apply_hysteresis(
        dates=s.index, raw_labels=raw_labels, deescalation_days=deescalation_days
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
