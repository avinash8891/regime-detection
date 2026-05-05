from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.models import AxisOutput, DataQuality


TrendCharacterLabel = Literal["trending", "chop", "recovery_attempt", "transition", "unknown"]


_RISK_RANK: dict[TrendCharacterLabel, int] = {
    "trending": 0,
    "chop": 1,
    "recovery_attempt": 1,
    "transition": 2,
    "unknown": 2,
}


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


def _wilder_ewm(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _compute_adx_14(*, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    # Align to a common index so np.where arrays match index lengths.
    common = close.index.intersection(high.index).intersection(low.index)
    close = close.reindex(common)
    high = high.reindex(common)
    low = low.reindex(common)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=common)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=common)

    n = 14
    atr = _wilder_ewm(tr, n)
    plus_di = 100 * _wilder_ewm(plus_dm, n) / atr
    minus_di = 100 * _wilder_ewm(minus_dm, n) / atr
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _wilder_ewm(dx, n)


def compute_features(*, close: pd.Series, high: pd.Series, low: pd.Series) -> TrendCharacterFeatures:
    sma_50 = close.rolling(50).mean()
    return_10d = close / close.shift(10) - 1
    return_21d = close / close.shift(21) - 1
    prior_63d_drawdown = close / close.rolling(63).max() - 1
    adx_14 = _compute_adx_14(high=high, low=low, close=close).reindex(close.index)
    return TrendCharacterFeatures(
        close=close,
        high=high,
        low=low,
        sma_50=sma_50,
        return_10d=return_10d,
        return_21d=return_21d,
        prior_63d_drawdown=prior_63d_drawdown,
        adx_14=adx_14,
    )


def raw_label_for_day(
    f: TrendCharacterFeatures, dt: pd.Timestamp
) -> tuple[TrendCharacterLabel, dict[str, Any]]:
    close = f.close.loc[dt]
    sma50 = f.sma_50.loc[dt]
    ret10 = f.return_10d.loc[dt]
    ret21 = f.return_21d.loc[dt]
    prior_dd = f.prior_63d_drawdown.loc[dt]
    adx = f.adx_14.loc[dt]

    if any(pd.isna(x) for x in [close, sma50, ret10, ret21, prior_dd, adx]):
        return "unknown", {"reason": "insufficient_history"}

    recovery_attempt = bool((prior_dd <= -0.10) and (close > sma50) and (ret10 >= 0.05))
    trending = bool((adx >= 20) and (abs(ret21) >= 0.05))
    chop = bool((adx < 20) and (abs(ret10) < 0.03) and (abs(ret21) < 0.05))
    transition = not (recovery_attempt or trending or chop)

    # precedence: recovery_attempt > trending > chop > transition > unknown
    if recovery_attempt:
        label: TrendCharacterLabel = "recovery_attempt"
    elif trending:
        label = "trending"
    elif chop:
        label = "chop"
    else:
        label = "transition"

    return label, {
        "recovery_attempt": recovery_attempt,
        "trending": trending,
        "chop": chop,
        "transition": transition,
    }


def classify_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    as_of_date: date,
    deescalation_days: int,
) -> AxisOutput:
    dt = pd.Timestamp(as_of_date)
    close = close.copy()
    high = high.copy()
    low = low.copy()
    close.index = pd.to_datetime(close.index)
    high.index = pd.to_datetime(high.index)
    low.index = pd.to_datetime(low.index)
    close = close.sort_index()
    high = high.sort_index()
    low = low.sort_index()

    if dt not in close.index:
        raise ValueError(f"as_of_date missing from close series: {as_of_date.isoformat()}")

    close = close.loc[:dt]
    high = high.loc[:dt]
    low = low.loc[:dt]

    f = compute_features(close=close, high=high, low=low)
    raw_labels: list[TrendCharacterLabel] = []
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

    dq = _data_quality_for_asof(close=close, as_of_date=as_of_date, required_trading_days=63, raw_label=raw)

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


def _data_quality_for_asof(
    *,
    close: pd.Series,
    as_of_date: date,
    required_trading_days: int,
    raw_label: TrendCharacterLabel,
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
