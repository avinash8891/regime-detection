"""v2 §1A Trend Direction axis — classify layer.

V1 raw-label walker (``raw_label_for_day`` / ``build_raw_outputs``) plus the
v2 §1A precedence overlay (``evaluate_recovery`` / ``evaluate_euphoria`` /
``evaluate_v2_trend_label``). The features layer lives in
``trend_direction.py``.

Precedence (v2 §1A line 239):
    euphoria > bull > recovery > bear > sideways > transition > unknown
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_detection._rule_helpers import ev_float as _ev_float
from regime_detection.config import TrendDirectionV2RulesConfig
from regime_detection.trend_direction import (
    TrendDirectionFeatures,
    TrendDirectionV2Features,
)

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
# precedence (which is in this module's _V2_TREND_PRECEDENCE).
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


def raw_label_for_day(
    f: TrendDirectionFeatures,
    dt: pd.Timestamp,
    *,
    trend_direction_v2_features: TrendDirectionV2Features | None = None,
    trend_direction_v2_rules: TrendDirectionV2RulesConfig | None = None,
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
    features: TrendDirectionV2Features, dt: pd.Timestamp
) -> dict[str, Any]:
    # Guard: V2 features may cover a shorter date range than V1 close.index
    # due to cold-start lookbacks. Return empty dict rather than KeyError.
    if dt not in features.efficiency_ratio_20d.index:
        return {}
    evidence = {
        "efficiency_ratio_20d": _ev_float(features.efficiency_ratio_20d.loc[dt]),
        "hurst_250d": _ev_float(features.hurst_250d.loc[dt]),
        "slope_sma_50": _ev_float(features.slope_sma_50.loc[dt]),
        "slope_sma_200": _ev_float(features.slope_sma_200.loc[dt]),
        "return_126d": _ev_float(features.return_126d.loc[dt]),
        "drawdown_252d": _ev_float(features.drawdown_252d.loc[dt]),
        "realized_vol_21d": _ev_float(features.realized_vol_21d.loc[dt]),
    }
    if features.sentiment_score is not None and dt in features.sentiment_score.index:
        evidence["sentiment_score"] = _ev_float(features.sentiment_score.loc[dt])
    if (
        features.news_sentiment_score is not None
        and dt in features.news_sentiment_score.index
    ):
        evidence["news_sentiment_score"] = _ev_float(
            features.news_sentiment_score.loc[dt]
        )
    if (
        features.sentiment_concordance is not None
        and dt in features.sentiment_concordance.index
    ):
        evidence["sentiment_concordance"] = _ev_float(
            features.sentiment_concordance.loc[dt]
        )
    return evidence


def build_raw_outputs(
    f: TrendDirectionFeatures,
    *,
    trend_direction_v2_features: TrendDirectionV2Features | None = None,
    trend_direction_v2_rules: TrendDirectionV2RulesConfig | None = None,
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


# ---------------------------------------------------------------------------
# v2 §1A `recovery` rule + precedence wrapper.
#
# Rule (v2 §1A lines 193-197, verbatim):
#     prior 252d drawdown <= -0.15
#     AND return_63d > 0.10
#     AND close > SMA_50
#
# Precedence (v2 §1A line 239):
#     euphoria > bull > recovery > bear > sideways > transition > unknown
#
# `euphoria` is wired to a real predicate via `evaluate_euphoria` below
# (post ADR 0004 amendment). The predicate consumes the Optional
# `sentiment_score` (AAII) feature; when the feature is absent it
# falsifies per V2 §10 no-hallucination rule.
# ---------------------------------------------------------------------------


def evaluate_recovery(
    features: TrendDirectionV2Features,
    close: pd.Series,
    *,
    dt: pd.Timestamp,
    rules_config: TrendDirectionV2RulesConfig,
) -> bool:
    """v2 §1A lines 193-197 `recovery` predicate at a single session.

    Returns False when any of the three inputs is NaN (cold-start
    contract — no silent "unknown → True" substitution). Spec citations:

    * line 195 — ``drawdown_252d <= recovery_drawdown_threshold`` (-0.15)
    * line 196 — ``return_63d   >  recovery_return_threshold``    ( 0.10)
    * line 197 — ``close        >  SMA_50``
    """
    if dt not in features.drawdown_252d.index:
        return False
    drawdown = features.drawdown_252d.loc[dt]
    return_63d = features.return_63d.loc[dt]
    sma_50 = features.sma_50.loc[dt]
    if dt not in close.index:
        return False
    close_t = close.loc[dt]

    # Cold-start / NaN propagation: any missing input falsifies the rule.
    if any(pd.isna(x) for x in (drawdown, return_63d, sma_50, close_t)):
        return False

    drawdown_ok = bool(drawdown <= rules_config.recovery_drawdown_threshold)  # line 195
    return_ok = bool(return_63d > rules_config.recovery_return_threshold)  # line 196
    above_sma = bool(close_t > sma_50)  # line 197
    return drawdown_ok and return_ok and above_sma


# v2 §1A line 239 ranking (lower index = higher precedence).
# `euphoria` (index 0) was reserved-but-inert before ADR 0004 closure;
# it is now wired to a real predicate.
_V2_TREND_PRECEDENCE: tuple[str, ...] = (
    "euphoria",
    "bull",
    "recovery",
    "bear",
    "sideways",
    "transition",
    "unknown",
)


def evaluate_euphoria(
    features: TrendDirectionV2Features,
    close: pd.Series,
    *,
    dt: pd.Timestamp,
    rules_config: TrendDirectionV2RulesConfig,
) -> bool:
    """v2 §1A lines 200-205 `euphoria` predicate at a single session.

    Returns False when any of the four inputs is NaN or when the
    Optional ``sentiment_score`` feature is absent (V2 §10
    no-hallucination rule — ADR 0004 closure).

    Spec citations (post ADR 0004 amendment):

    * line 202 — ``close > SMA_200`` (strict)
    * line 203 — ``return_126d > euphoria_return_126d_threshold`` (0.20, strict)
    * line 204 — ``realized_vol_21d rising`` (strict 5-session change
      per implementation decision #68 §1D analogue: ``vol[t] > vol[t - N]`` where
      ``N = euphoria_vol_rising_lookback_sessions``)
    * line 205 — ``sentiment_score >= euphoria_sentiment_threshold``
      (+20 default; non-strict at boundary)
    """
    sentiment_series = features.sentiment_score
    if sentiment_series is None:
        return False

    if dt not in features.return_126d.index or dt not in close.index:
        return False
    if dt not in features.sma_200.index or dt not in features.realized_vol_21d.index:
        return False
    if dt not in sentiment_series.index:
        return False

    lookback = rules_config.euphoria_vol_rising_lookback_sessions
    vol_index = features.realized_vol_21d.index
    try:
        pos_t = vol_index.get_loc(dt)
    except KeyError:
        return False
    pos_back = pos_t - lookback
    if pos_back < 0:
        return False
    vol_t = features.realized_vol_21d.iloc[pos_t]
    vol_back = features.realized_vol_21d.iloc[pos_back]

    close_t = close.loc[dt]
    sma_200_t = features.sma_200.loc[dt]
    return_126d_t = features.return_126d.loc[dt]
    sentiment_t = sentiment_series.loc[dt]

    # Cold-start / NaN propagation: any missing input falsifies the rule.
    if any(
        pd.isna(x)
        for x in (close_t, sma_200_t, return_126d_t, vol_t, vol_back, sentiment_t)
    ):
        return False

    close_above_sma = bool(close_t > sma_200_t)  # line 202
    return_ok = bool(
        return_126d_t > rules_config.euphoria_return_126d_threshold
    )  # line 203
    vol_rising = bool(vol_t > vol_back)  # line 204
    sentiment_ok = bool(
        sentiment_t >= rules_config.euphoria_sentiment_threshold
    )  # line 205
    return close_above_sma and return_ok and vol_rising and sentiment_ok


def evaluate_v2_trend_label(
    *,
    v1_label: str,
    features: TrendDirectionV2Features,
    close: pd.Series,
    dt: pd.Timestamp,
    rules_config: TrendDirectionV2RulesConfig,
) -> str | None:
    """Apply v2 §1A trend precedence on top of a v1 raw label.

    Returns the winning v2 label per the §1A line 239 ordering, or
    ``None`` when no v2 rule fires and the caller should keep ``v1_label``.

    Precedence (line 239): ``euphoria > bull > recovery > bear >
    sideways > transition > unknown``.

    Dispatch order: euphoria first (top of precedence — outranks every
    v1 label including bull); then recovery (only fires when v1 is
    strictly lower-ranked, i.e. ``bear`` / ``sideways`` / ``transition``
    / ``unknown``).
    """
    euphoria_fires = evaluate_euphoria(
        features, close, dt=dt, rules_config=rules_config
    )
    if euphoria_fires:
        return "euphoria"

    recovery_fires = evaluate_recovery(
        features, close, dt=dt, rules_config=rules_config
    )
    if not recovery_fires:
        return None

    try:
        v1_rank = _V2_TREND_PRECEDENCE.index(v1_label)
    except ValueError:
        # Unknown v1 label — treat as lowest precedence and let recovery win.
        v1_rank = len(_V2_TREND_PRECEDENCE)
    recovery_rank = _V2_TREND_PRECEDENCE.index("recovery")
    if v1_rank < recovery_rank:
        # v1 label outranks recovery (only possible value: bull).
        return None
    return "recovery"
