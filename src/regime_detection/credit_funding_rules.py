"""v2 §2C Credit/Funding axis — classify layer.

Labels, risk rank, per-day rule input materialisation, predicates, and the
precedence walker. The features layer lives in ``credit_funding.py`` and
ships ``CreditFundingFeatures`` + ``compute_credit_funding_features``;
this module consumes that dataclass to build per-day rule inputs and
evaluate predicates.

Spec references (docs/regime_engine_v2_spec.md):
    §2C lines 3173-3178  Labels
    §2C line  3183       Precedence:
        deleveraging > funding_squeeze > credit_stress > spread_widening >
        credit_divergence > credit_recovery > credit_calm > unknown
    §2C lines 3249-3271  Rules
    §2C lines 3277-3283  Risk Rank

Required-input keys (HYG_KEY, LQD_KEY, ...) and source provenance codes
live here as the public vocabulary axis_builders / feature_store consume
to thread the right macro_series + cross_asset_closes into the features
function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from regime_detection._rule_helpers import (
    any_nan as _any_nan,
    scalar_at as _scalar_at,
)
from regime_detection._series_alignment import (
    aligned_float_values,
    optional_aligned_float_values,
)
from regime_detection.config import CreditFundingRulesConfig
from regime_detection.credit_funding import CreditFundingFeatures

# ---------------------------------------------------------------------------
# Spec labels (§2C lines 3173-3178) + risk rank (§2C lines 3277-3283).
# ---------------------------------------------------------------------------

CreditFundingLabel = Literal[
    "credit_calm",
    "credit_recovery",
    "credit_divergence",
    "spread_widening",
    "credit_stress",
    "funding_squeeze",
    "deleveraging",
    "unknown",
]


# v2 §2C lines 3277-3283 verbatim. ``deleveraging: 4`` is the ONLY V2 axis label
# with risk_rank>3 — reflects that the rule fires only when five distinct stress
# signals coincide (spec line 3286).
CREDIT_FUNDING_RISK_RANK: dict[CreditFundingLabel, int] = {
    "credit_calm": 0,
    "credit_recovery": 0,
    "credit_divergence": 1,
    "unknown": 1,
    "spread_widening": 1,
    "credit_stress": 2,
    "funding_squeeze": 3,
    "deleveraging": 4,
}


# ---------------------------------------------------------------------------
# Required FRED / cross-asset symbol keys. Pinned here as single source of
# truth so feature_store + classifier read one constant.
# ---------------------------------------------------------------------------

HYG_KEY = "HYG"
LQD_KEY = "LQD"
TLT_KEY = "TLT"
KRE_KEY = "KRE"

SOFR_KEY = "sofr"
IORB_KEY = "iorb"
FEDFUNDS_KEY = "fedfunds"
IOER_LEGACY_KEY = "ioer_legacy"
NFCI_KEY = "nfci"
BROAD_USD_INDEX_KEY = "broad_usd_index"
# ICE BofA Option-Adjusted Spread series — FRED-redistributed under ICE
# license, free at the FRED endpoint. macro_series keys set by
# `V2_FRED_SERIES` in `regime_data_fetch.fetch_workflow`.
HY_OAS_KEY = "hy_oas"  # FRED BAMLH0A0HYM2 — ICE BofA US High Yield OAS
IG_OAS_KEY = "ig_bbb_oas"  # FRED BAMLC0A4CBBB — ICE BofA BBB Corporate OAS


REQUIRED_CROSS_ASSET_KEYS: tuple[str, ...] = (HYG_KEY, LQD_KEY, TLT_KEY, KRE_KEY)
REQUIRED_MACRO_KEYS: tuple[str, ...] = (
    SOFR_KEY,
    IORB_KEY,
    NFCI_KEY,
    BROAD_USD_INDEX_KEY,
)


# ---------------------------------------------------------------------------
# Per-day scalar rule inputs (mirrors network_fragility_rules pattern).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreditFundingRuleInputs:
    """Per-day scalars consumed by the §2C rule predicates."""

    hy_spread_percentile_504d: float
    hy_spread_slope_21d: float
    ig_spread_slope_21d: float
    broad_usd_index_zscore_21d: float
    sofr_iorb_slope_21d: float
    spy_21d_return: float
    tlt_21d_return: float
    realized_vol_21d_percentile_252d: float
    realized_vol_21d: float
    avg_pairwise_corr_percentile_504d: float
    avg_pairwise_corr_63d: float


@dataclass(frozen=True)
class CreditFundingRuleEvaluation:
    label: CreditFundingLabel
    rule_path: str
    reason: str | None = None


def build_rule_inputs_for_date(
    *,
    features: CreditFundingFeatures,
    dt: pd.Timestamp,
    hy_spread_percentile_504d: pd.Series,
    hy_spread_slope_21d: pd.Series,
    ig_spread_slope_21d: pd.Series,
    realized_vol_21d_percentile_252d: pd.Series,
    avg_pairwise_corr_percentile_504d: pd.Series,
    realized_vol_21d: pd.Series | None = None,
    avg_pairwise_corr_63d: pd.Series | None = None,
) -> CreditFundingRuleInputs:
    """Materialize the per-day scalar rule inputs at session ``dt``.

    The spread triple is passed explicitly (source-neutral) so the same
    builder serves both the real-OAS run (pass ``features.hy_oas_*``) and
    the proxy run (pass ``features.hy_tr_differential_*``) — ADR 0007; implementation decision + #71.
    """
    return CreditFundingRuleInputs(
        hy_spread_percentile_504d=_scalar_at(hy_spread_percentile_504d, dt),
        hy_spread_slope_21d=_scalar_at(hy_spread_slope_21d, dt),
        ig_spread_slope_21d=_scalar_at(ig_spread_slope_21d, dt),
        broad_usd_index_zscore_21d=_scalar_at(features.broad_usd_index_zscore_21d, dt),
        sofr_iorb_slope_21d=_scalar_at(features.sofr_iorb_slope_21d, dt),
        spy_21d_return=_scalar_at(features.spy_21d_return, dt),
        tlt_21d_return=_scalar_at(features.tlt_21d_return, dt),
        realized_vol_21d_percentile_252d=_scalar_at(
            realized_vol_21d_percentile_252d, dt
        ),
        realized_vol_21d=(
            _scalar_at(realized_vol_21d, dt)
            if realized_vol_21d is not None
            else float("nan")
        ),
        avg_pairwise_corr_percentile_504d=_scalar_at(
            avg_pairwise_corr_percentile_504d, dt
        ),
        avg_pairwise_corr_63d=(
            _scalar_at(avg_pairwise_corr_63d, dt)
            if avg_pairwise_corr_63d is not None
            else float("nan")
        ),
    )


def build_rule_inputs_by_date(
    *,
    features: CreditFundingFeatures,
    hy_spread_percentile_504d: pd.Series,
    hy_spread_slope_21d: pd.Series,
    ig_spread_slope_21d: pd.Series,
    realized_vol_21d_percentile_252d: pd.Series,
    avg_pairwise_corr_percentile_504d: pd.Series,
    realized_vol_21d: pd.Series | None = None,
    avg_pairwise_corr_63d: pd.Series | None = None,
) -> dict[pd.Timestamp, CreditFundingRuleInputs]:
    """Per-date rule inputs. The spread triple is source-neutral — pass
    ``features.hy_oas_*`` for the real-OAS run or ``features.hy_tr_differential_*``
    for the proxy run (ADR 0007; implementation decision + #71)."""
    index = hy_spread_percentile_504d.index
    hy_spread_percentile_values = aligned_float_values(hy_spread_percentile_504d, index)
    hy_spread_slope_values = aligned_float_values(hy_spread_slope_21d, index)
    ig_spread_slope_values = aligned_float_values(ig_spread_slope_21d, index)
    broad_usd_zscore_values = aligned_float_values(
        features.broad_usd_index_zscore_21d, index
    )
    sofr_iorb_slope_values = aligned_float_values(features.sofr_iorb_slope_21d, index)
    spy_return_values = aligned_float_values(features.spy_21d_return, index)
    tlt_return_values = aligned_float_values(features.tlt_21d_return, index)
    realized_vol_percentile_values = aligned_float_values(
        realized_vol_21d_percentile_252d, index
    )
    realized_vol_values = optional_aligned_float_values(realized_vol_21d, index)
    avg_corr_percentile_values = aligned_float_values(
        avg_pairwise_corr_percentile_504d, index
    )
    avg_corr_values = optional_aligned_float_values(avg_pairwise_corr_63d, index)

    outputs: dict[pd.Timestamp, CreditFundingRuleInputs] = {}
    for pos, dt in enumerate(index):
        outputs[dt] = CreditFundingRuleInputs(
            hy_spread_percentile_504d=float(hy_spread_percentile_values[pos]),
            hy_spread_slope_21d=float(hy_spread_slope_values[pos]),
            ig_spread_slope_21d=float(ig_spread_slope_values[pos]),
            broad_usd_index_zscore_21d=float(broad_usd_zscore_values[pos]),
            sofr_iorb_slope_21d=float(sofr_iorb_slope_values[pos]),
            spy_21d_return=float(spy_return_values[pos]),
            tlt_21d_return=float(tlt_return_values[pos]),
            realized_vol_21d_percentile_252d=float(realized_vol_percentile_values[pos]),
            realized_vol_21d=float(realized_vol_values[pos]),
            avg_pairwise_corr_percentile_504d=float(avg_corr_percentile_values[pos]),
            avg_pairwise_corr_63d=float(avg_corr_values[pos]),
        )
    return outputs


# ---------------------------------------------------------------------------
# Rule predicates (§2C lines 3249-3271).
# ---------------------------------------------------------------------------


def evaluate_credit_calm(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 3249-3251.

    ``hy_spread_percentile_504d < 0.50
       AND hy_spread_slope_21d <= 0`` (non-rising slope).
    """
    if _any_nan(
        inputs.hy_spread_percentile_504d,
        inputs.hy_spread_slope_21d,
    ):
        return False
    return bool(
        inputs.hy_spread_percentile_504d < config.hy_percentile_calm_max
        and inputs.hy_spread_slope_21d <= 0.0
    )


def spread_widening_rule_path(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> str | None:
    """Return the spread-widening path for confirmed or HY-led widening.

    The spec path requires both HY and IG widening. The elevated HY-led path is
    still a widening state when the HY spread percentile is already at or above
    the calm boundary; IG lag is common during early deterioration.
    """
    if _any_nan(inputs.hy_spread_slope_21d, inputs.hy_spread_percentile_504d):
        return None
    if not np.isnan(inputs.ig_spread_slope_21d) and (
        inputs.hy_spread_slope_21d > 0.0 and inputs.ig_spread_slope_21d > 0.0
    ):
        return "standard"
    if (
        inputs.hy_spread_slope_21d > 0.0
        and inputs.hy_spread_percentile_504d >= config.hy_percentile_calm_max
    ):
        return "hy_led_elevated"
    if config.spread_widening_hy_only and (inputs.hy_spread_slope_21d > 0.0):
        return "hy_only_config"
    return None


def evaluate_spread_widening(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 3253-3255 plus elevated HY-led widening."""
    return spread_widening_rule_path(inputs, config) is not None


def evaluate_credit_recovery(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """Elevated spreads (percentile >=0.50) that are narrowing (slope < 0).

    Economically: credit conditions are improving from stressed levels.
    Distinct from credit_calm (percentile < 0.50) and spread_widening (slope > 0).
    """
    if _any_nan(
        inputs.hy_spread_percentile_504d,
        inputs.hy_spread_slope_21d,
    ):
        return False
    return bool(
        inputs.hy_spread_percentile_504d >= config.hy_percentile_calm_max
        and inputs.hy_spread_slope_21d < 0.0
    )


def evaluate_credit_divergence(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """Low-spread HY-only widening with no IG confirmation.

    This is benign/divergent credit behavior: absolute spread level is still
    calm, HY is softening, and IG is not confirming a broad widening impulse.
    """
    if _any_nan(
        inputs.hy_spread_percentile_504d,
        inputs.hy_spread_slope_21d,
        inputs.ig_spread_slope_21d,
    ):
        return False
    return bool(
        inputs.hy_spread_percentile_504d < config.hy_percentile_calm_max
        and inputs.hy_spread_slope_21d > 0.0
        and inputs.ig_spread_slope_21d <= 0.0
    )


def evaluate_credit_stress(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 3257-3259.

    ``hy_spread_percentile_504d > 0.80 AND spy_21d_return < -0.05``.
    """
    if _any_nan(
        inputs.hy_spread_percentile_504d,
        inputs.spy_21d_return,
    ):
        return False
    return bool(
        inputs.hy_spread_percentile_504d > config.hy_percentile_stress_min
        and inputs.spy_21d_return < config.spy_drop_threshold
    )


def evaluate_funding_squeeze(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 3261-3264.

    ``broad_usd_index_zscore_21d > +1.5 AND sofr_iorb_slope_21d > 0
       AND spy_21d_return < 0``.
    """
    if _any_nan(
        inputs.broad_usd_index_zscore_21d,
        inputs.sofr_iorb_slope_21d,
        inputs.spy_21d_return,
    ):
        return False
    return bool(
        inputs.broad_usd_index_zscore_21d > config.broad_usd_zscore_funding_threshold
        and inputs.sofr_iorb_slope_21d > 0.0
        and inputs.spy_21d_return < 0.0
    )


def _deleveraging_percentile_path(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    if _any_nan(
        inputs.spy_21d_return,
        inputs.tlt_21d_return,
        inputs.broad_usd_index_zscore_21d,
        inputs.realized_vol_21d_percentile_252d,
        inputs.avg_pairwise_corr_percentile_504d,
    ):
        return False
    return bool(
        inputs.spy_21d_return < config.spy_drop_threshold
        and inputs.tlt_21d_return < 0.0
        and inputs.broad_usd_index_zscore_21d
        > config.broad_usd_zscore_deleveraging_threshold
        and inputs.realized_vol_21d_percentile_252d
        > config.realized_vol_percentile_threshold
        and inputs.avg_pairwise_corr_percentile_504d
        > config.correlation_percentile_threshold
    )


def _deleveraging_cold_start_path(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    if not config.cold_start_deleveraging_enabled:
        return False
    if not (
        np.isnan(inputs.realized_vol_21d_percentile_252d)
        or np.isnan(inputs.avg_pairwise_corr_percentile_504d)
    ):
        return False
    if _any_nan(
        inputs.spy_21d_return,
        inputs.tlt_21d_return,
        inputs.broad_usd_index_zscore_21d,
        inputs.realized_vol_21d,
        inputs.avg_pairwise_corr_63d,
    ):
        return False
    return bool(
        inputs.spy_21d_return < config.spy_drop_threshold
        and inputs.tlt_21d_return < 0.0
        and inputs.broad_usd_index_zscore_21d
        > config.broad_usd_zscore_deleveraging_threshold
        and inputs.realized_vol_21d
        >= config.cold_start_deleveraging_realized_vol_21d_min
        and inputs.avg_pairwise_corr_63d
        >= config.cold_start_deleveraging_avg_corr_63d_min
    )


def deleveraging_rule_path(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> str | None:
    if _deleveraging_percentile_path(inputs, config):
        return "percentile"
    if _deleveraging_cold_start_path(inputs, config):
        return "cold_start_fallback"
    return None


def evaluate_deleveraging(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 3266-3271 — 5-condition composite.

    ``spy_21d_return < -0.05 AND tlt_21d_return < 0
       AND broad_usd_index_zscore_21d > 0
       AND realized_vol_21d_percentile_252d > 0.75
       AND avg_pairwise_corr_percentile_504d > 0.75``.
    """
    return deleveraging_rule_path(inputs, config) is not None


def evaluate_rules(
    *,
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> CreditFundingLabel:
    """Walk v2 §2C precedence and return the first matching label.

    Falls through to ``unknown`` only when the supplied valid inputs escape
    the intended rule partition.
    """
    return evaluate_rules_with_evidence(inputs=inputs, config=config).label


def evaluate_rules_with_evidence(
    *,
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> CreditFundingRuleEvaluation:
    """Walk v2 §2C precedence and return the label plus matched rule path."""
    deleveraging_path = deleveraging_rule_path(inputs, config)
    if deleveraging_path is not None:
        return CreditFundingRuleEvaluation(
            label="deleveraging",
            rule_path=deleveraging_path,
        )
    if evaluate_funding_squeeze(inputs, config):
        return CreditFundingRuleEvaluation(
            label="funding_squeeze", rule_path="standard"
        )
    if evaluate_credit_stress(inputs, config):
        return CreditFundingRuleEvaluation(label="credit_stress", rule_path="standard")
    widening_path = spread_widening_rule_path(inputs, config)
    if widening_path is not None:
        return CreditFundingRuleEvaluation(
            label="spread_widening", rule_path=widening_path
        )
    if evaluate_credit_divergence(inputs, config):
        return CreditFundingRuleEvaluation(
            label="credit_divergence", rule_path="hy_only_low_spread"
        )
    if evaluate_credit_recovery(inputs, config):
        return CreditFundingRuleEvaluation(
            label="credit_recovery", rule_path="elevated_narrowing"
        )
    if evaluate_credit_calm(inputs, config):
        return CreditFundingRuleEvaluation(label="credit_calm", rule_path="standard")
    return CreditFundingRuleEvaluation(
        label="unknown",
        rule_path="unpartitioned_rule_space",
        reason="unpartitioned_credit_funding_rule_space",
    )
