"""v2 §2A Layer 2A Monetary / Liquidity axis — classify layer.

Labels, risk rank, per-day rule input materialisation, and the precedence
walker. The features layer lives in ``monetary_pressure.py`` and ships
``MonetaryPressureV2Features`` + ``compute_monetary_pressure_features``;
this module consumes that dataclass to build per-day rule inputs and
evaluate the §2A precedence.

Rules (verbatim §2A lines 2881-2892):

  tightening_pressure:
    yield_change_zscore_2y_63d > +1.5
    OR yield_change_zscore_10y_63d > +1.5
    OR broad_usd_index_zscore_63d > +1.5

  easing_pressure:
    yield_change_zscore_2y_63d < -1.5
    OR yield_change_zscore_10y_63d < -1.5

  rate_shock:
    abs(yield_change_zscore_21d_2y) > 2.0
    OR abs(yield_change_zscore_21d_10y) > 2.0

Precedence (implementation decision(c)):
  rate_shock > tightening_pressure > easing_pressure > neutral_monetary > unknown

NaN-safe: a missing rule input returns ``unknown`` instead of falling through
to ``neutral_monetary``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from regime_detection._rule_helpers import any_nan as _any_nan
from regime_detection._rule_helpers import scalar_at as _scalar_at
from regime_detection.config import MonetaryPressureV2RulesConfig
from regime_detection.monetary_pressure import MonetaryPressureV2Features

# ---------------------------------------------------------------------------
# Label set + risk rank (implementation decision(b)/(d)).
# ---------------------------------------------------------------------------


MonetaryPressureV2Label = Literal[
    "tightening_pressure",
    "easing_pressure",
    "rate_shock",
    "neutral_monetary",
    "unknown",
]


# v2 §2A risk rank per implementation decision(d). Pinned constant (NOT a tunable).
MONETARY_PRESSURE_V2_RISK_RANK: dict[MonetaryPressureV2Label, int] = {
    "neutral_monetary": 0,
    "easing_pressure": 1,
    "unknown": 1,
    "tightening_pressure": 2,
    "rate_shock": 3,
}


# ---------------------------------------------------------------------------
# Rule predicates (§2A lines 2881-2892, implementation decision(b)/(c)).
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

    Precedence (implementation decision(c)):
      rate_shock > tightening_pressure > easing_pressure > neutral_monetary

    Missing inputs fail closed to ``unknown`` instead of silently emitting
    ``neutral_monetary`` on incomplete rule evidence.
    """
    if _any_nan(
        inputs.zscore_2y_63d,
        inputs.zscore_10y_63d,
        inputs.broad_usd_zscore_63d,
        inputs.zscore_21d_2y,
        inputs.zscore_21d_10y,
    ):
        return "unknown"
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
    # easing_pressure — OR on the two 63d-change yield signals.
    if (
        inputs.zscore_2y_63d < config.easing_pressure_zscore_threshold
        or inputs.zscore_10y_63d < config.easing_pressure_zscore_threshold
    ):
        return "easing_pressure"
    return "neutral_monetary"
