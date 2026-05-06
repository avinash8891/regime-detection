from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

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


def apply_hysteresis(
    *,
    dates: pd.DatetimeIndex,
    raw_labels: list[TrendDirectionLabel],
    deescalation_days: int,
) -> tuple[list[TrendDirectionLabel], list[TrendDirectionLabel]]:
    """
    Deterministic asymmetric hysteresis:
    - Escalation (higher risk_rank) updates stable immediately.
    - De-escalation requires `deescalation_days` consecutive days where raw_label has
      risk_rank <= candidate_rank. When met, stable becomes that raw_label.
    - active_label uses the spec fast-path: if raw is riskier than stable, active=raw else active=stable.
    """
    if len(dates) != len(raw_labels):
        raise ValueError("dates/raw_labels length mismatch")
    if deescalation_days < 0:
        raise ValueError("deescalation_days must be >= 0")

    stable: list[TrendDirectionLabel] = []
    active: list[TrendDirectionLabel] = []

    stable_label: TrendDirectionLabel = raw_labels[0]
    pending_label: TrendDirectionLabel | None = None
    pending_count = 0

    for raw in raw_labels:
        raw_rank = _RISK_RANK[raw]
        stable_rank = _RISK_RANK[stable_label]

        if raw_rank > stable_rank:
            # Escalate immediately.
            stable_label = raw
            pending_label = None
            pending_count = 0
        elif raw_rank < stable_rank:
            # Candidate de-escalation.
            if deescalation_days == 0:
                stable_label = raw
                pending_label = None
                pending_count = 0
            else:
                if pending_label != raw:
                    pending_label = raw
                    pending_count = 1
                else:
                    pending_count += 1
                if pending_count >= deescalation_days:
                    stable_label = raw
                    pending_label = None
                    pending_count = 0
        else:
            # Equal rank: treat as de-escalation candidate only if label differs.
            if raw != stable_label:
                if deescalation_days == 0:
                    stable_label = raw
                    pending_label = None
                    pending_count = 0
                else:
                    if pending_label != raw:
                        pending_label = raw
                        pending_count = 1
                    else:
                        pending_count += 1
                    if pending_count >= deescalation_days:
                        stable_label = raw
                        pending_label = None
                        pending_count = 0
            else:
                pending_label = None
                pending_count = 0

        stable.append(stable_label)
        # active_label per spec fast-path
        if _RISK_RANK[raw] > _RISK_RANK[stable_label]:
            active.append(raw)
        else:
            active.append(stable_label)

    return stable, active


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
