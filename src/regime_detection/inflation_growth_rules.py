from __future__ import annotations

import numpy as np

from regime_detection.config import InflationGrowthRulesConfig
from regime_detection.inflation_growth import (
    InflationGrowthLabel,
    InflationGrowthRuleInputs,
)


def _any_nan(*values: float) -> bool:
    return any(np.isnan(v) for v in values)


def evaluate_goldilocks(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 2233-2238."""
    if inputs.credit_funding_active_label is None:
        return False
    if inputs.credit_funding_active_label != "credit_calm":
        return False
    if _any_nan(
        inputs.pmi_manufacturing,
        inputs.spy_21d_return,
    ):
        return False
    drift_ok = False
    if not _any_nan(inputs.cpi_6m_change_pct, inputs.cpi_6m_change_pct_lag_21):
        drift_ok = (
            abs(inputs.cpi_6m_change_pct - inputs.cpi_6m_change_pct_lag_21)
            <= config.cpi_drift_threshold
        )
    slope_ok = False
    if not np.isnan(inputs.cpi_6m_change_pct_slope_21d):
        slope_ok = inputs.cpi_6m_change_pct_slope_21d <= 0.0
    if not (drift_ok or slope_ok):
        return False
    return bool(
        inputs.pmi_manufacturing > config.pmi_goldilocks_threshold
        and inputs.spy_21d_return > 0.0
    )


def evaluate_inflation_shock(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 2550-2555 — `inflation_shock` two-limb OR rule."""
    if not _any_nan(inputs.inflation_surprise_zscore) and (
        inputs.inflation_surprise_zscore
        > config.inflation_surprise_zscore_threshold
    ):
        return True

    if _any_nan(
        inputs.commodity_return_63d,
        inputs.treasury_10y_yield_slope_21d,
        inputs.spy_21d_return,
        inputs.tlt_21d_return,
    ):
        return False
    return bool(
        inputs.commodity_return_63d > config.commodity_return_threshold
        and inputs.treasury_10y_yield_slope_21d > 0.0
        and inputs.spy_21d_return < 0.0
        and inputs.tlt_21d_return < 0.0
    )


def evaluate_disinflation(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 2247-2250."""
    if _any_nan(
        inputs.cpi_6m_change_pct_slope_21d,
        inputs.treasury_10y_yield_slope_21d,
        inputs.pmi_manufacturing,
    ):
        return False
    return bool(
        inputs.cpi_6m_change_pct_slope_21d < 0.0
        and inputs.treasury_10y_yield_slope_21d < 0.0
        and inputs.pmi_manufacturing > config.pmi_disinflation_threshold
    )


def evaluate_recession_scare(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 2252-2256."""
    if inputs.credit_funding_active_label is None:
        return False
    if inputs.credit_funding_active_label not in {"spread_widening", "credit_stress"}:
        return False
    if _any_nan(
        inputs.treasury_10y_yield_slope_21d,
        inputs.cyclical_defensive_slope_21d,
        inputs.spy_21d_return,
    ):
        return False
    return bool(
        inputs.treasury_10y_yield_slope_21d < 0.0
        and inputs.cyclical_defensive_slope_21d < 0.0
        and inputs.spy_21d_return < config.spy_recession_threshold
    )


def evaluate_recovery_growth(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 2258-2261."""
    if inputs.credit_funding_active_label is None:
        return False
    if inputs.credit_funding_active_label != "credit_calm":
        return False
    if _any_nan(
        inputs.pmi_manufacturing_slope_21d,
        inputs.pmi_manufacturing,
        inputs.cyclical_defensive_slope_21d,
    ):
        return False
    return bool(
        inputs.pmi_manufacturing_slope_21d > 0.0
        and inputs.pmi_manufacturing > config.pmi_recovery_threshold
        and inputs.cyclical_defensive_slope_21d > 0.0
    )


def evaluate_earnings_expansion(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B line 2605 — strict positive aggregate forward-EPS revision."""
    if _any_nan(inputs.aggregate_forward_eps_revision_direction_4w):
        return False
    return bool(
        inputs.aggregate_forward_eps_revision_direction_4w
        > config.eps_revision_expansion_threshold
    )


def evaluate_earnings_contraction(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B line 2609 — strict negative aggregate forward-EPS revision."""
    if _any_nan(inputs.aggregate_forward_eps_revision_direction_4w):
        return False
    return bool(
        inputs.aggregate_forward_eps_revision_direction_4w
        < config.eps_revision_contraction_threshold
    )


def evaluate_rules(
    *,
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> InflationGrowthLabel:
    """Walk v2 §2B precedence and return the first matching label."""
    if evaluate_inflation_shock(inputs, config):
        return "inflation_shock"
    if evaluate_recession_scare(inputs, config):
        return "recession_scare"
    if evaluate_disinflation(inputs, config):
        return "disinflation"
    if evaluate_goldilocks(inputs, config):
        return "goldilocks"
    if evaluate_recovery_growth(inputs, config):
        return "recovery_growth"
    if evaluate_earnings_contraction(inputs, config):
        return "earnings_contraction"
    if evaluate_earnings_expansion(inputs, config):
        return "earnings_expansion"
    return "unknown"
