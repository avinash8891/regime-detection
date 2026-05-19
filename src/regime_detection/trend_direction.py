from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from regime_detection._rolling_stats import period_return, simple_moving_average
from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.models import AxisOutput, DataQuality

if TYPE_CHECKING:  # avoid runtime cycle: trend_direction_v2 → config → ...
    from regime_detection.config import TrendDirectionV2RulesConfig
    from regime_detection.trend_direction_v2 import TrendDirectionV2Features


# v2 §1A line 132-134 precedence: euphoria > bull > recovery > bear > sideways > transition > unknown.
TrendDirectionLabel = Literal[
    "euphoria",
    "bull",
    "bear",
    "sideways",
    "transition",
    "unknown",
    "recovery",
]


# v2 §1A line 132 places `recovery` between `bull` (0) and `bear` (3): bull > recovery > bear.
# Risk-rank intuition: recovery is mid-rally off a deep drawdown — riskier than steady bull
# but less risky than a bear. Slot at 1 (matching sideways) so existing v1 hysteresis
# behavior for non-recovery labels is unchanged.
_RISK_RANK: dict[TrendDirectionLabel, int] = {
    "bull": 0,
    "sideways": 1,
    "recovery": 1,
    "transition": 2,
    "euphoria": 4,
    "bear": 3,
    "unknown": 2,
}


@dataclass(frozen=True)
class TrendDirectionFeatures:
    close: pd.Series
    sma_50: pd.Series
    sma_200: pd.Series
    return_63d: pd.Series


def _ev_float(x: float) -> float:
    return round(float(x), 8)


def compute_features(close: pd.Series) -> TrendDirectionFeatures:
    sma_50 = simple_moving_average(close, window=50)
    sma_200 = simple_moving_average(close, window=200)
    return_63d = period_return(close, periods=63)
    return TrendDirectionFeatures(close=close, sma_50=sma_50, sma_200=sma_200, return_63d=return_63d)


def raw_label_for_day(
    f: TrendDirectionFeatures,
    dt: pd.Timestamp,
    *,
    trend_direction_v2_features: "TrendDirectionV2Features | None" = None,
    trend_direction_v2_rules: "TrendDirectionV2RulesConfig | None" = None,
) -> tuple[TrendDirectionLabel, dict[str, Any]]:
    """Per-day raw trend_direction label.

    When ``trend_direction_v2_features`` and ``trend_direction_v2_rules`` are
    both supplied, the v2 §1A precedence (line 132-134:
    ``bull > recovery > bear > sideways > transition > unknown``) is layered
    ON TOP of the v1 label. When either is ``None`` the function returns
    the v1 label and evidence unchanged.
    """
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

    if bull:
        label: TrendDirectionLabel = "bull"
    elif bear:
        label = "bear"
    elif sideways:
        label = "sideways"
    else:
        label = "transition"

    evidence: dict[str, Any] = {
        "sma_50": _ev_float(sma50),
        "sma_200": _ev_float(sma200),
        "return_63d": _ev_float(ret63),
        "close_gt_sma50": bool(close > sma50),
        "close_gt_sma200": bool(close > sma200),
        "sma50_gt_sma200": bool(sma50 > sma200),
        "within_5pct_sma200": within_5pct_sma200,
    }

    if trend_direction_v2_features is not None and trend_direction_v2_rules is not None:
        evidence.update(_v2_evidence_for_day(trend_direction_v2_features, dt))
        # Local import keeps the v1 path free of v2 module load on cold
        # callers (e.g., the frozen v1 replay shim) and avoids a circular
        # import (trend_direction_v2 imports TrendDirectionV2RulesConfig
        # from config; we only need its functions here).
        from regime_detection.trend_direction_v2 import evaluate_v2_trend_label

        v2_label = evaluate_v2_trend_label(
            v1_label=label,
            features=trend_direction_v2_features,
            close=f.close,
            dt=dt,
            rules_config=trend_direction_v2_rules,
        )
        if v2_label is not None:
            evidence["v2_override"] = {
                "from": label,
                "to": v2_label,
                "rule": v2_label,
            }
            label = v2_label  # type: ignore[assignment]

    return label, evidence


def _v2_evidence_for_day(
    features: "TrendDirectionV2Features", dt: pd.Timestamp
) -> dict[str, Any]:
    evidence = {
        "efficiency_ratio_20d": _ev_float(features.efficiency_ratio_20d.loc[dt]),
        "hurst_250d": _ev_float(features.hurst_250d.loc[dt]),
        "slope_sma_50": _ev_float(features.slope_sma_50.loc[dt]),
        "slope_sma_200": _ev_float(features.slope_sma_200.loc[dt]),
        "return_126d": _ev_float(features.return_126d.loc[dt]),
        "drawdown_252d": _ev_float(features.drawdown_252d.loc[dt]),
        "realized_vol_21d": _ev_float(features.realized_vol_21d.loc[dt]),
    }
    if features.sentiment_score is not None:
        evidence["sentiment_score"] = _ev_float(features.sentiment_score.loc[dt])
    if features.news_sentiment_score is not None:
        evidence["news_sentiment_score"] = _ev_float(
            features.news_sentiment_score.loc[dt]
        )
    if features.sentiment_concordance is not None:
        evidence["sentiment_concordance"] = _ev_float(
            features.sentiment_concordance.loc[dt]
        )
    return evidence


def build_raw_outputs(
    f: TrendDirectionFeatures,
    *,
    trend_direction_v2_features: "TrendDirectionV2Features | None" = None,
    trend_direction_v2_rules: "TrendDirectionV2RulesConfig | None" = None,
) -> tuple[list[TrendDirectionLabel], list[dict[str, Any]]]:
    """Vectorized v1 raw labels + optional v2 §1A `recovery` override.

    When both ``trend_direction_v2_features`` and ``trend_direction_v2_rules`` are
    supplied, the v2 §1A precedence at line 132-134 is applied per-day
    AFTER the v1 pass — `recovery` overrides v1 `bear` / `sideways` /
    `transition` / `unknown` (NOT `bull`, which outranks `recovery`).
    When either argument is None, output is byte-identical to v1.
    """
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
    close_gt_sma50 = valid & close.gt(sma50)
    close_gt_sma200 = valid & close.gt(sma200)
    sma50_gt_sma200 = valid & sma50.gt(sma200)

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
                "sma_50": _ev_float(sma50.iat[idx]),
                "sma_200": _ev_float(sma200.iat[idx]),
                "return_63d": _ev_float(ret63.iat[idx]),
                "close_gt_sma50": bool(close_gt_sma50.iat[idx]),
                "close_gt_sma200": bool(close_gt_sma200.iat[idx]),
                "sma50_gt_sma200": bool(sma50_gt_sma200.iat[idx]),
                "within_5pct_sma200": bool(within_5pct_sma200.iat[idx]),
            }
        )

    if trend_direction_v2_features is not None and trend_direction_v2_rules is not None:
        # v2 §1A line 132-134 precedence — applied per-day on top of v1.
        # Localize import to avoid a runtime cycle with trend_direction_v2.
        from regime_detection.trend_direction_v2 import evaluate_v2_trend_label

        for idx, dt in enumerate(close.index):
            if labels[idx] != "unknown":
                evidence[idx].update(_v2_evidence_for_day(trend_direction_v2_features, dt))
            v1_label = str(labels[idx])
            v2_label = evaluate_v2_trend_label(
                v1_label=v1_label,
                features=trend_direction_v2_features,
                close=close,
                dt=dt,
                rules_config=trend_direction_v2_rules,
            )
            if v2_label is None:
                continue
            evidence[idx]["v2_override"] = {
                "from": v1_label,
                "to": v2_label,
                "rule": v2_label,
            }
            labels[idx] = v2_label

    return list(labels), evidence


def apply_hysteresis(
    *,
    dates: pd.DatetimeIndex,
    raw_labels: list[TrendDirectionLabel],
    escalation_days: int = 1,
    deescalation_days: int,
) -> tuple[list[TrendDirectionLabel], list[TrendDirectionLabel]]:
    if len(dates) != len(raw_labels):
        raise ValueError("dates/raw_labels length mismatch")
    return apply_asymmetric_hysteresis(
        raw_labels=raw_labels,
        risk_rank=_RISK_RANK,
        escalation_days=escalation_days,
        deescalation_days=deescalation_days,
    )


