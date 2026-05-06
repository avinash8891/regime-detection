from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd

from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.models import BreadthStateOutput, DataQuality


BreadthLabel = Literal[
    "healthy_breadth",
    "neutral_breadth",
    "weak_breadth",
    "divergent_fragile",
    "unknown",
]


_RISK_RANK: dict[BreadthLabel, int] = {
    "healthy_breadth": 0,
    "neutral_breadth": 1,
    "weak_breadth": 2,
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
    idx_dist = spy_close / spy_close.rolling(63).max() - 1
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

    dq = _data_quality_for_asof(
        spy_close=spy_close,
        rsp_close=rsp_close.reindex(spy_close.index),
        as_of_date=as_of_date,
        required_trading_days=63,
        raw_label=raw,
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


def _data_quality_for_asof(
    *,
    spy_close: pd.Series,
    rsp_close: pd.Series,
    as_of_date: date,
    required_trading_days: int,
    raw_label: BreadthLabel,
) -> DataQuality:
    dt = pd.Timestamp(as_of_date)
    w_spy = spy_close.loc[:dt].tail(required_trading_days)
    w_rsp = rsp_close.loc[:dt].tail(required_trading_days)
    c_spy = float(w_spy.notna().mean()) if len(w_spy) else 0.0
    c_rsp = float(w_rsp.notna().mean()) if len(w_rsp) else 0.0
    completeness = min(c_spy, c_rsp)

    if raw_label == "unknown":
        return DataQuality(status="insufficient_history", freshness_days=0, completeness=completeness, reason="insufficient_history")
    if completeness < 0.70:
        return DataQuality(status="insufficient_data", freshness_days=0, completeness=completeness, reason="insufficient_data")
    if completeness < 0.90:
        return DataQuality(status="degraded", freshness_days=0, completeness=completeness, reason="incomplete_data")
    return DataQuality(status="ok", freshness_days=0, completeness=completeness, reason=None)
