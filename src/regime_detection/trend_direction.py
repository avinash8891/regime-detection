from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from regime_detection._rolling_stats import period_return, simple_moving_average
from regime_detection._rule_helpers import ev_float as _ev_float

if TYPE_CHECKING:  # avoid runtime cycle: trend_direction_v2 → config → ...
    from regime_detection.config import TrendDirectionV2RulesConfig
    from regime_detection.trend_direction_v2 import TrendDirectionV2Features


# v2 §1A line 239 precedence: euphoria > bull > recovery > bear > sideways > transition > unknown.
TrendDirectionLabel = Literal[
    "euphoria",
    "bull",
    "bear",
    "sideways",
    "transition",
    "unknown",
    "recovery",
]


# `_RISK_RANK` encodes DOWNSIDE RISK for hysteresis decay, NOT spec
# precedence (which is in trend_direction_v2._V2_TREND_PRECEDENCE).
# The two orderings are intentionally different — see ADR 0016
# "trend_direction risk-rank vs precedence". Higher rank = more dangerous
# for risk management: euphoria=4 (speculative excess collapses fastest),
# bear=3 (established downtrend), unknown=2 (mid-rank — neither escalates
# past bear nor strands across cold-start), recovery/sideways=1, bull=0.
_RISK_RANK: dict[TrendDirectionLabel, int] = {
    "bull": 0,
    "sideways": 1,
    "recovery": 1,
    "transition": 2,
    "euphoria": 4,
    "bear": 3,
    "unknown": 2,
}

# V1-legacy trend-direction thresholds (sideways gate + close-to-SMA200 band).
# Not restated in v2 spec; v1 contract frozen by §2.2 stateless replay.
_SIDEWAYS_ABS_RETURN_63D = 0.05
_WITHIN_PCT_SMA200 = 0.05


@dataclass(frozen=True)
class TrendDirectionFeatures:
    close: pd.Series
    sma_50: pd.Series
    sma_200: pd.Series
    return_63d: pd.Series


def compute_features(close: pd.Series) -> TrendDirectionFeatures:
    sma_50 = simple_moving_average(close, window=50)
    sma_200 = simple_moving_average(close, window=200)
    return_63d = period_return(close, periods=63)
    return TrendDirectionFeatures(
        close=close, sma_50=sma_50, sma_200=sma_200, return_63d=return_63d
    )


def raw_label_for_day(
    f: TrendDirectionFeatures,
    dt: pd.Timestamp,
    *,
    trend_direction_v2_features: "TrendDirectionV2Features | None" = None,
    trend_direction_v2_rules: "TrendDirectionV2RulesConfig | None" = None,
) -> tuple[TrendDirectionLabel, dict[str, Any]]:
    """Per-day raw trend_direction label.

    When ``trend_direction_v2_features`` and ``trend_direction_v2_rules`` are
    both supplied, the v2 §1A precedence (line 239:
    ``euphoria > bull > recovery > bear > sideways > transition > unknown``)
    is layered ON TOP of the v1 label. When either is ``None`` the function
    returns the v1 label and evidence unchanged.

    F-043: this is a thin wrapper over :func:`build_raw_outputs` so the §3.5
    rule predicates, evidence shape, and v2 override have a single encoding.
    Slicing each feature to ``[dt]`` is safe because the vectorized builder and
    the v2 ``evaluate_recovery`` / ``evaluate_euphoria`` predicates only read
    ``close.loc[dt]`` and ``features.*.loc[dt]`` at the target session.
    """
    day_features = TrendDirectionFeatures(
        close=f.close.loc[[dt]],
        sma_50=f.sma_50.loc[[dt]],
        sma_200=f.sma_200.loc[[dt]],
        return_63d=f.return_63d.loc[[dt]],
    )
    labels, evidence = build_raw_outputs(
        day_features,
        trend_direction_v2_features=trend_direction_v2_features,
        trend_direction_v2_rules=trend_direction_v2_rules,
    )
    return labels[0], evidence[0]


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
    """Vectorized v1 raw labels + optional v2 §1A `recovery` / `euphoria` override.

    When both ``trend_direction_v2_features`` and ``trend_direction_v2_rules``
    are supplied, the v2 §1A precedence at line 239 is applied per-day AFTER
    the v1 pass: ``euphoria > bull > recovery > bear > sideways > transition
    > unknown``. ``recovery`` overrides v1 ``bear`` / ``sideways`` /
    ``transition`` / ``unknown`` (NOT ``bull`` or ``euphoria``, which both
    outrank ``recovery``). When either argument is None, output is
    byte-identical to v1.
    """
    close = f.close
    sma50 = f.sma_50
    sma200 = f.sma_200
    ret63 = f.return_63d

    valid = ~(close.isna() | sma50.isna() | sma200.isna() | ret63.isna())
    within_5pct_sma200 = valid & close.between(
        sma200 * (1 - _WITHIN_PCT_SMA200),
        sma200 * (1 + _WITHIN_PCT_SMA200),
    )
    bull = valid & close.gt(sma50) & close.gt(sma200) & sma50.gt(sma200)
    bear = valid & close.lt(sma50) & close.lt(sma200) & sma50.lt(sma200)
    sideways = valid & ret63.abs().lt(_SIDEWAYS_ABS_RETURN_63D) & within_5pct_sma200
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
        # v2 §1A line 239 precedence — applied per-day on top of v1.
        # Localize import to avoid a runtime cycle with trend_direction_v2.
        from regime_detection.trend_direction_v2 import evaluate_v2_trend_label

        for idx, dt in enumerate(close.index):
            # Skip v2 evidence on "unknown" rows: v1 produced unknown only when
            # one of close/sma50/sma200/ret63 is NaN at cold-start, and the
            # evidence dict already carries `{"reason": "insufficient_history"}`.
            # Adding v2 features here would surface NaN floats and obscure the
            # cold-start signal. v2 rule eval below still runs on unknown rows
            # so recovery/euphoria can override unknown per spec §1A line 239
            # precedence — matches per-day `raw_label_for_day` behavior.
            if labels[idx] != "unknown":
                evidence[idx].update(
                    _v2_evidence_for_day(trend_direction_v2_features, dt)
                )
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
