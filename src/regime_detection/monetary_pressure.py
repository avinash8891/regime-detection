"""v2 §2A Layer 2A Monetary / Liquidity V2 features + axis classifier rules.

Feature slice (4.1) ships the line-896 yield z-score template; the
classifier slice (this file) extends it with the three additional
features pinned in Ambiguity Log #46 (a) and the §2A rule engine:

  Features (Log #46 a — mechanical generalizations of the line-896 template):
    - yield_change_zscore_2y_63d      (existing)
    - yield_change_zscore_10y_63d     (existing)
    - broad_usd_index_zscore_63d      (NEW, 63d-change z-score on USD index)
    - yield_change_zscore_21d_2y      (NEW, 21d-change z-score on DGS2)
    - yield_change_zscore_21d_10y     (NEW, 21d-change z-score on DGS10)

  Labels (Log #46 b):
    {tightening_pressure, easing_pressure, rate_shock, neutral_monetary, unknown}

  Precedence (Log #46 c):
    rate_shock > tightening_pressure > easing_pressure > neutral_monetary > unknown

  Risk rank (Log #46 d):
    {neutral_monetary: 0, easing_pressure: 1, unknown: 1,
     tightening_pressure: 2, rate_shock: 3}

  Per-label hysteresis (Log #46 e; carried on the yaml):
    {rate_shock: 5, tightening_pressure: 3, easing_pressure: 2,
     neutral_monetary: 0, unknown: 2}

Rules (verbatim §2A lines 1093-1104):

  tightening_pressure:
    yield_change_zscore_2y_63d > +1.5
    OR yield_change_zscore_10y_63d > +1.5
    OR broad_usd_index_zscore_63d > +1.5

  easing_pressure:
    yield_change_zscore_2y_63d < -1.5
    AND yield_change_zscore_10y_63d < -1.5

  rate_shock:
    abs(yield_change_zscore_21d_2y) > +2.0
    OR abs(yield_change_zscore_21d_10y) > +2.0

NaN-safe: ``<`` / ``>`` on NaN evaluates False in Python, so a NaN
input naturally falsifies the predicate and the precedence falls
through to ``neutral_monetary``. The data-quality gate above
``evaluate_rules`` catches the cold-start case and maps to ``unknown``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from regime_detection._rolling_stats import (
    _ZSCORE_DDOF as _ZSCORE_DDOF,
    rolling_change_zscore as _rolling_change_zscore,
)
from regime_detection.config import (
    MonetaryPressureV2FeaturesConfig,
    MonetaryPressureV2RulesConfig,
)


# ---------------------------------------------------------------------------
# Label set + risk rank (Ambiguity Log #46 b/d).
# ---------------------------------------------------------------------------


MonetaryPressureV2Label = Literal[
    "tightening_pressure",
    "easing_pressure",
    "rate_shock",
    "neutral_monetary",
    "unknown",
]


# v2 §2A risk rank per Ambiguity Log #46 (d). Pinned constant (NOT a tunable).
MONETARY_PRESSURE_V2_RISK_RANK: dict[MonetaryPressureV2Label, int] = {
    "neutral_monetary": 0,
    "easing_pressure": 1,
    "unknown": 1,
    "tightening_pressure": 2,
    "rate_shock": 3,
}


# v2 §2A precedence (highest-severity-first walk) per Ambiguity Log #46 (c).
RULE_PRECEDENCE: tuple[MonetaryPressureV2Label, ...] = (
    "rate_shock",
    "tightening_pressure",
    "easing_pressure",
    "neutral_monetary",
)


@dataclass(frozen=True)
class MonetaryPressureV2Features:
    """v2 §2A — per-session continuous monetary-pressure features.

    Five z-score series aligned to the input DatetimeIndex. NaN cold-start
    at the head of each series until the corresponding (lookback +
    normalizer) window fills.
    """

    yield_change_zscore_2y_63d: pd.Series
    yield_change_zscore_10y_63d: pd.Series
    broad_usd_index_zscore_63d: pd.Series
    yield_change_zscore_21d_2y: pd.Series
    yield_change_zscore_21d_10y: pd.Series

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "yield_change_zscore_2y_63d",
            "yield_change_zscore_10y_63d",
            "broad_usd_index_zscore_63d",
            "yield_change_zscore_21d_2y",
            "yield_change_zscore_21d_10y",
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: getattr(self, name) for name in self.feature_names}
        )


def _yield_change_zscore(
    *,
    yield_series: pd.Series,
    lookback: int,
    normalizer_window: int,
    output_name: str,
) -> pd.Series:
    """Thin §2A wrapper over the shared `rolling_change_zscore` helper.

    One home per concept lives in `_rolling_stats.py`; §2A and §2C only
    differ in their change_window / normalizer_window defaults.
    """
    return _rolling_change_zscore(
        yield_series,
        change_window=lookback,
        normalizer_window=normalizer_window,
        output_name=output_name,
    )


def compute_monetary_pressure_features(
    *,
    dgs2: pd.Series,
    dgs10: pd.Series,
    broad_usd_index: pd.Series | None = None,
    config: MonetaryPressureV2FeaturesConfig,
) -> MonetaryPressureV2Features:
    """Compute the v2 §2A yield + USD z-score features.

    Parameters
    ----------
    dgs2
        FRED ``DGS2`` (2y constant-maturity Treasury yield) series.
    dgs10
        FRED ``DGS10`` (10y constant-maturity Treasury yield) series.
    broad_usd_index
        FRED broad USD index level (e.g. ``DTWEXBGS``). When ``None``,
        the ``broad_usd_index_zscore_63d`` output is an all-NaN series
        aligned to the dgs2 index (Ambiguity Log #46 a graceful fallback
        when the USD seam is absent).
    config
        ``MonetaryPressureV2FeaturesConfig`` — supplies all four window
        lengths (yield 63d, normalizer 1260d, rate-shock 21d, broad-USD 63d).
    """
    z_2y = _yield_change_zscore(
        yield_series=dgs2,
        lookback=config.yield_change_lookback_days,
        normalizer_window=config.zscore_normalizer_window_days,
        output_name="yield_change_zscore_2y_63d",
    )
    z_10y = _yield_change_zscore(
        yield_series=dgs10,
        lookback=config.yield_change_lookback_days,
        normalizer_window=config.zscore_normalizer_window_days,
        output_name="yield_change_zscore_10y_63d",
    )
    z_21d_2y = _yield_change_zscore(
        yield_series=dgs2,
        lookback=config.rate_shock_lookback_days,
        normalizer_window=config.zscore_normalizer_window_days,
        output_name="yield_change_zscore_21d_2y",
    )
    z_21d_10y = _yield_change_zscore(
        yield_series=dgs10,
        lookback=config.rate_shock_lookback_days,
        normalizer_window=config.zscore_normalizer_window_days,
        output_name="yield_change_zscore_21d_10y",
    )
    if broad_usd_index is None:
        usd_z = pd.Series(
            float("nan"),
            index=dgs2.index,
            name="broad_usd_index_zscore_63d",
        )
    else:
        usd_z = _rolling_change_zscore(
            broad_usd_index,
            change_window=config.broad_usd_lookback_days,
            normalizer_window=config.zscore_normalizer_window_days,
            output_name="broad_usd_index_zscore_63d",
        )
    return MonetaryPressureV2Features(
        yield_change_zscore_2y_63d=z_2y,
        yield_change_zscore_10y_63d=z_10y,
        broad_usd_index_zscore_63d=usd_z,
        yield_change_zscore_21d_2y=z_21d_2y,
        yield_change_zscore_21d_10y=z_21d_10y,
    )


# ---------------------------------------------------------------------------
# Rule predicates (§2A lines 1093-1104, Ambiguity Log #46 b/c).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonetaryPressureRuleInputs:
    """Per-day scalar inputs the §2A rule engine consumes.

    Materialized by ``build_rule_inputs_for_date`` from
    ``MonetaryPressureV2Features`` at one as-of date.
    """

    zscore_2y_63d: float
    zscore_10y_63d: float
    broad_usd_zscore_63d: float
    zscore_21d_2y: float
    zscore_21d_10y: float


def _scalar_at(series: pd.Series, dt: pd.Timestamp) -> float:
    if dt not in series.index:
        return float("nan")
    val = series.loc[dt]
    if pd.isna(val):
        return float("nan")
    return float(val)


def build_rule_inputs_for_date(
    *,
    features: MonetaryPressureV2Features,
    dt: pd.Timestamp,
) -> MonetaryPressureRuleInputs:
    return MonetaryPressureRuleInputs(
        zscore_2y_63d=_scalar_at(features.yield_change_zscore_2y_63d, dt),
        zscore_10y_63d=_scalar_at(features.yield_change_zscore_10y_63d, dt),
        broad_usd_zscore_63d=_scalar_at(features.broad_usd_index_zscore_63d, dt),
        zscore_21d_2y=_scalar_at(features.yield_change_zscore_21d_2y, dt),
        zscore_21d_10y=_scalar_at(features.yield_change_zscore_21d_10y, dt),
    )


def evaluate_rules(
    *,
    inputs: MonetaryPressureRuleInputs,
    config: MonetaryPressureV2RulesConfig,
) -> MonetaryPressureV2Label:
    """Walk §2A precedence and return the first matching label.

    Precedence (Log #46 c):
      rate_shock > tightening_pressure > easing_pressure > neutral_monetary

    NaN inputs naturally falsify each ``<`` / ``>`` comparison; the walker
    then falls through to ``neutral_monetary``. The data-quality gate in
    the classifier maps that to ``unknown`` when required inputs are
    insufficient.
    """
    # rate_shock — abs(21d-change z) > +2.0 on either tenor (highest precedence).
    if (
        abs(inputs.zscore_21d_2y) > config.rate_shock_zscore_threshold
        or abs(inputs.zscore_21d_10y) > config.rate_shock_zscore_threshold
    ):
        return "rate_shock"
    # tightening_pressure — OR across the three 63d-change signals.
    if (
        inputs.zscore_2y_63d > config.tightening_pressure_zscore_threshold
        or inputs.zscore_10y_63d > config.tightening_pressure_zscore_threshold
        or inputs.broad_usd_zscore_63d > config.tightening_pressure_zscore_threshold
    ):
        return "tightening_pressure"
    # easing_pressure — AND on the two 63d-change yield signals.
    if (
        inputs.zscore_2y_63d < config.easing_pressure_zscore_threshold
        and inputs.zscore_10y_63d < config.easing_pressure_zscore_threshold
    ):
        return "easing_pressure"
    return "neutral_monetary"
